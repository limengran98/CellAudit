"""Compact adapters for local perturbation-response task families."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import file_fingerprint, normalize_condition_layout
from .schemas import SchemaError, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_PATH = PROJECT_ROOT / "configs" / "tasks.json"


def _require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{module_name} is required for this command. Install the runtime extras "
            "in a CUDA-capable environment: pip install -e '.[runtime]'"
        ) from exc


def load_task_registry(path: str | Path = DEFAULT_TASKS_PATH) -> Mapping[str, Mapping[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SchemaError("Task configuration must be a JSON object")
    return payload


def get_task(task_id: str, path: str | Path = DEFAULT_TASKS_PATH) -> Mapping[str, Any]:
    registry = load_task_registry(path)
    try:
        task = registry[task_id]
    except KeyError as exc:
        raise SchemaError(f"Unknown task {task_id}; available: {', '.join(sorted(registry))}") from exc
    if not isinstance(task, Mapping):
        raise SchemaError(f"Task {task_id} must be an object")
    return task


def _decode(values: Any) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _categorical(h5_group: Any, name: str):
    """Decode both AnnData categorical encodings seen in local H5AD files.

    AnnData 0.7 commonly stores integer codes in a dataset whose ``categories``
    attribute is an HDF5 object reference.  Newer files instead store a group
    containing explicit ``codes`` and ``categories`` datasets.  Treating the
    former as plain strings silently turns a perturbation label such as
    ``AHR+FEV`` into ``"262"``; the raw RNA adapters must never do that.
    """
    node = h5_group[name]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        categories = _decode(node["categories"][:])
        codes = node["codes"][:]
        return [categories[int(code)] if int(code) >= 0 else "<missing>" for code in codes]
    if hasattr(node, "attrs") and "categories" in node.attrs:
        categories_ref = node.attrs["categories"]
        categories = _decode(node.file[categories_ref][:])
        codes = node[:]
        return [categories[int(code)] if int(code) >= 0 else "<missing>" for code in codes]
    return _decode(node[:])


def _stable_fold(label: str, folds: int = 5) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16) % folds


def _hash_features(strings: list[str], dimensions: int = 128):
    np = _require("numpy")
    features = np.zeros((len(strings), dimensions), dtype=np.float32)
    for row, value in enumerate(strings):
        # character n-grams make an inexpensive deterministic condition encoder.
        padded = f"^{value}$"
        grams = [padded[index : index + width] for width in (1, 2, 3) for index in range(max(0, len(padded) - width + 1))]
        for gram in grams:
            bucket = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16) % dimensions
            features[row, bucket] += 1.0
    norms = np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    return features / norms


@dataclass(frozen=True)
class TaskArrays:
    task_id: str
    train_pre: Any
    train_condition: Any
    train_target: Any
    validation_pre: Any
    validation_condition: Any
    validation_target: Any
    test_pre: Any
    test_condition: Any
    test_target: Any
    metadata: Mapping[str, Any]

    @property
    def input_dimensions(self) -> tuple[int, int, int]:
        return (
            int(self.train_pre.shape[1]),
            int(self.train_condition.shape[1]),
            int(self.train_target.shape[1]),
        )


def inspect_task(task_id: str, config_path: str | Path = DEFAULT_TASKS_PATH) -> dict[str, Any]:
    task = get_task(task_id, config_path)
    if task.get("adapter") == "cpg_replicate_profiles":
        return inspect_cpg_task(task_id, task)
    if task.get("adapter") in {"norman_h5ad_count_response", "norman_h5ad_pseudobulk"}:
        return inspect_norman_task(task_id, task)
    source = Path(str(task["path"]))
    result: dict[str, Any] = {"task_id": task_id, "adapter": task.get("adapter"), "path": str(source), "exists": source.exists()}
    if not source.exists():
        return result
    result["bytes"] = source.stat().st_size
    if task.get("adapter") == "bbbc_h5":
        h5py = _require("h5py")
        with h5py.File(source, "r") as handle:
            result["groups"] = sorted(handle.keys())
            result["combined"] = {key: list(handle["combined"][key].shape) for key in handle["combined"]}
    elif str(task.get("adapter", "")).endswith("h5ad") or "h5ad" in str(task.get("adapter", "")):
        h5py = _require("h5py")
        try:
            with h5py.File(source, "r") as handle:
                result["groups"] = sorted(handle.keys())
                x = handle.get("X")
                if x is not None:
                    encoding = x.attrs.get("encoding-type", "dense") if hasattr(x, "attrs") else "dense"
                    result["x_encoding"] = encoding.decode("utf-8") if isinstance(encoding, bytes) else str(encoding)
                    shape = x.attrs.get("shape", x.shape if hasattr(x, "shape") else [])
                    result["x_shape"] = [int(value) for value in shape]
        except OSError as exc:
            result["readable"] = False
            result["error"] = type(exc).__name__
        else:
            result["readable"] = True
    return result


def _subsample(indices: Any, maximum: int | None, seed: int):
    np = _require("numpy")
    if maximum is None or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def load_bbbc_arrays(
    task_id: str,
    task: Mapping[str, Any],
    *,
    test_fold: int | None = None,
    validation_fold: int | None = None,
    seed: int = 0,
) -> TaskArrays:
    """Read a BBBC HDF5 task using independent train/validation/test folds."""
    np = _require("numpy")
    h5py = _require("h5py")
    fold = int(test_fold or task.get("default_test_fold", 5))
    val_fold = int(validation_fold or (fold - 1 if fold > 1 else 5))
    with h5py.File(task["path"], "r") as handle:
        group = handle["combined"]
        split = group["split_id"][:]
        train_indices = np.flatnonzero((split != fold) & (split != val_fold))
        val_indices = np.flatnonzero(split == val_fold)
        test_indices = np.flatnonzero(split == fold)
        if not len(train_indices) or not len(val_indices) or not len(test_indices):
            raise SchemaError(f"BBBC task {task_id} does not contain required fold roles")
        train_indices = _subsample(train_indices, int(task.get("max_train_samples", len(train_indices))), seed)
        # Validation/test stay complete unless data itself is unusually large.
        pre = group["morphology_pre"][:].astype(np.float32, copy=False)
        post = group["morphology_post"][:].astype(np.float32, copy=False)
        smiles = _decode(group["smiles"][:])
        dose = group["dose"][:].astype(np.float32, copy=False)
    condition = _hash_features(smiles)
    dose = (dose - dose.mean()) / max(float(dose.std()), 1e-6)
    condition = np.concatenate([condition, dose[:, None]], axis=1)
    return TaskArrays(
        task_id=task_id,
        train_pre=pre[train_indices],
        train_condition=condition[train_indices],
        train_target=post[train_indices],
        validation_pre=pre[val_indices],
        validation_condition=condition[val_indices],
        validation_target=post[val_indices],
        test_pre=pre[test_indices],
        test_condition=condition[test_indices],
        test_target=post[test_indices],
        metadata={"family": "cell_painting", "test_fold": fold, "validation_fold": val_fold, "n_features": int(pre.shape[1])},
    )


# ---------------------------------------------------------------------------
# CPG0003 release-profile adapter
# ---------------------------------------------------------------------------


def _cpg_value(task: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read a CPG setting from the preferred nested block or legacy flat key."""
    block = task.get("cpg", {})
    if isinstance(block, Mapping) and name in block:
        return block[name]
    return task.get(name, default)


def _cpg_source(task: Mapping[str, Any], modality: str) -> Path:
    sources = task.get("sources", {})
    candidate = sources.get(modality) if isinstance(sources, Mapping) else None
    candidate = candidate or task.get(f"{modality}_path")
    if not candidate:
        raise SchemaError(f"CPG task has no registered {modality} source")
    source = Path(str(candidate))
    if not source.exists():
        raise FileNotFoundError(f"Registered CPG source does not exist: {source}")
    return source


def _cpg_join_column(task: Mapping[str, Any], modality: str) -> str:
    joined = _cpg_value(task, "join_columns", {})
    if isinstance(joined, Mapping) and modality in joined:
        return str(joined[modality])
    fallback = _cpg_value(task, f"{modality}_join_column")
    if not fallback:
        raise SchemaError(f"CPG task is missing {modality} join column")
    return str(fallback)


def _cpg_condition_key_spec(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return an explicit derived condition-key specification when registered.

    Most CPG0003 tasks share a literal identifier column between assays.  The
    LINCS-Pilot1 task instead declares a condition key assembled from the
    registered compound, dose, and cell-line columns in each raw table.  This
    is deliberately a deterministic canonicalization, never a fuzzy ID map.
    """
    value = _cpg_value(task, "condition_key")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaError("CPG condition_key must be an object")
    return value


def _cpg_condition_key_columns(task: Mapping[str, Any], modality: str) -> list[str]:
    """Registered raw columns needed to construct one assay condition key."""
    spec = _cpg_condition_key_spec(task)
    if spec is None:
        return [_cpg_join_column(task, modality)]
    if str(spec.get("kind", "")) != "compound_dose_cell":
        raise SchemaError(f"Unsupported CPG condition-key kind: {spec.get('kind')!r}")
    components = spec.get("components")
    if not isinstance(components, Mapping) or not isinstance(components.get(modality), Mapping):
        raise SchemaError(f"CPG compound_dose_cell key is missing {modality} components")
    registered = components[modality]
    names = [registered.get("compound_column"), registered.get("dose_column"), registered.get("cell_column")]
    if any(value is None or not str(value).strip() for value in names):
        raise SchemaError(f"CPG compound_dose_cell key is incomplete for {modality}")
    return [str(value) for value in names]


def _cpg_canonical_compound(value: Any) -> str:
    """Accept only a registered BRD identifier; do not infer compound identity."""
    text = str(value).strip().upper()
    match = re.search(r"(BRD-[A-Z]\d{8})", text)
    if match is None:
        match = re.search(r"(BRDN\d{10})", text)
    return match.group(1).lower() if match is not None else ""


def _cpg_canonical_dose(value: Any, *, decimals: int) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "-666"}:
        return ""
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    if abs(numeric + 666.0) < 1e-9:
        return ""
    if abs(numeric) < 10 ** (-(decimals + 1)):
        numeric = 0.0
    return f"{numeric:.{decimals}f}".rstrip("0").rstrip(".") or "0"


def _cpg_canonical_cell(value: Any) -> str:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "na", "n/a"}:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _cpg_condition_keys(task: Mapping[str, Any], modality: str, frame: Any):
    """Build raw-table keys used for independent assay aggregation and exact join."""
    pd = _require("pandas")
    spec = _cpg_condition_key_spec(task)
    if spec is None:
        return frame[_cpg_join_column(task, modality)].astype(str).str.strip()
    kind = str(spec.get("kind", ""))
    if kind != "compound_dose_cell":
        raise SchemaError(f"Unsupported CPG condition-key kind: {kind!r}")
    components = spec["components"][modality]
    decimals = int(spec.get("dose_decimals", 6))
    if decimals < 0 or decimals > 12:
        raise SchemaError("CPG condition_key dose_decimals must be between 0 and 12")
    prefix = str(spec.get("dataset_token", "cpg")).strip().lower()
    compound = frame[str(components["compound_column"])].tolist()
    dose = frame[str(components["dose_column"])].tolist()
    cell = frame[str(components["cell_column"])].tolist()
    values = []
    for compound_value, dose_value, cell_value in zip(compound, dose, cell):
        compound_token = _cpg_canonical_compound(compound_value)
        dose_token = _cpg_canonical_dose(dose_value, decimals=decimals)
        cell_token = _cpg_canonical_cell(cell_value)
        values.append(
            f"{prefix}|brd:{compound_token}|{dose_token}|{cell_token}"
            if compound_token and dose_token and cell_token
            else ""
        )
    return pd.Series(values, index=frame.index, dtype="object")


def _cpg_condition_key_description(task: Mapping[str, Any], modality: str) -> Any:
    """JSON-safe key description included in task inspection and cache manifests."""
    spec = _cpg_condition_key_spec(task)
    if spec is None:
        return _cpg_join_column(task, modality)
    components = spec.get("components", {})
    return {
        "kind": str(spec.get("kind", "")),
        "dataset_token": str(spec.get("dataset_token", "cpg")),
        "dose_decimals": int(spec.get("dose_decimals", 6)),
        "components": dict(components.get(modality, {})) if isinstance(components, Mapping) else {},
    }


def _cpg_perturbation_column(task: Mapping[str, Any], modality: str) -> str:
    columns = _cpg_value(task, "perturbation_columns", {})
    if isinstance(columns, Mapping) and modality in columns:
        return str(columns[modality])
    fallback = _cpg_value(task, f"{modality}_perturbation_column", "pert_type")
    return str(fallback)


def _cpg_plate_column(task: Mapping[str, Any], modality: str) -> str:
    columns = _cpg_value(task, "plate_columns", {})
    if isinstance(columns, Mapping) and modality in columns:
        return str(columns[modality])
    fallback = _cpg_value(task, f"{modality}_plate_column")
    if not fallback:
        raise SchemaError(f"CPG task is missing {modality} plate column")
    return str(fallback)


def _cpg_feature_columns(columns: list[str], modality: str, task: Mapping[str, Any]) -> list[str]:
    explicit = _cpg_value(task, f"{modality}_feature_columns")
    if explicit is not None:
        selected = [str(column) for column in explicit]
        missing = sorted(set(selected).difference(columns))
        if missing:
            raise SchemaError(f"Configured {modality} features are absent: {missing[:5]}")
        return selected
    if modality == "cell_painting":
        selected = [
            column
            for column in columns
            if column.startswith(("Cells_", "Cytoplasm_", "Nuclei_"))
        ]
    else:
        endpoint = str(_cpg_value(task, "l1000_feature_end_before", "pert_id"))
        if endpoint not in columns:
            raise SchemaError(f"L1000 feature boundary {endpoint!r} is absent")
        start_marker = _cpg_value(task, "l1000_feature_start_after")
        if start_marker is not None and str(start_marker) not in columns:
            raise SchemaError(f"L1000 feature start marker {start_marker!r} is absent")
        start = columns.index(str(start_marker)) + 1 if start_marker is not None else 0
        excluded = set(str(value) for value in _cpg_value(task, "l1000_exclude_columns", ["id"]))
        selected = [column for column in columns[start : columns.index(endpoint)] if column not in excluded]
    if not selected:
        raise SchemaError(f"No profile features found for {modality}")
    return selected


def _cpg_response_transform(task: Mapping[str, Any], modality: str) -> str:
    """Return the explicitly registered assay-level response transform.

    The historical adapter subtracts the same-plate control median.  That
    remains the default so existing task registrations and caches are
    unchanged.  A task may opt one assay into the bounded empirical-control
    transform through either a scalar ``response_transform`` value or a
    modality-keyed mapping.
    """

    configured = _cpg_value(task, "response_transform")
    if configured is None:
        return "control_median_centered"
    if isinstance(configured, Mapping):
        value = configured.get(modality, "control_median_centered")
    else:
        value = configured
    transform = str(value)
    allowed = {"control_median_centered", "empirical_control_cdf_centered"}
    if transform not in allowed:
        raise SchemaError(
            f"Unsupported {modality} response transform {transform!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return transform


def _empirical_control_cdf_centered(np: Any, values: Any, reference: Any):
    """Map rows to bounded same-feature control mid-CDF scores.

    For one feature with ``n`` registered controls, the score is

    ``(count(control < x) + 0.5 * count(control == x)) / n - 0.5``.

    This is the empirical mid-distribution function centered at zero.  It is
    deterministic, bounded in ``[-0.5, 0.5]``, preserves direction relative
    to controls, and cannot let a raw plate-specific magnitude dominate a
    cross-plate condition median.  Only controls from the registered
    reference set are used.
    """

    values = np.asarray(values, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2 or reference.ndim != 2:
        raise SchemaError("Empirical-control response transform requires two-dimensional arrays")
    if values.shape[1] != reference.shape[1] or reference.shape[0] < 1:
        raise SchemaError("Empirical-control response transform has an invalid reference shape")
    transformed = np.empty_like(values, dtype=np.float32)
    denominator = float(reference.shape[0])
    for feature_index in range(values.shape[1]):
        ordered = np.sort(reference[:, feature_index])
        column = values[:, feature_index]
        left = np.searchsorted(ordered, column, side="left")
        right = np.searchsorted(ordered, column, side="right")
        transformed[:, feature_index] = (
            (left + 0.5 * (right - left)) / denominator - 0.5
        ).astype(np.float32, copy=False)
    return transformed


def _cpg_control_relative_response(
    np: Any,
    *,
    treatment: Any,
    controls: Any,
    features: list[str],
    plate_column: str,
    missing_policy: str,
    transform: str,
):
    """Apply one registered same-plate control transform to treatment rows."""

    treatment_values = treatment[features].to_numpy(dtype=np.float32)
    treatment_plates = treatment[plate_column].astype(str).to_numpy()
    control_plates = controls[plate_column].astype(str).to_numpy()
    control_values = controls[features].to_numpy(dtype=np.float32)

    if transform == "control_median_centered":
        plate_control = controls.assign(
            **{plate_column: controls[plate_column].astype(str)}
        ).groupby(plate_column, sort=False)[features].median()
        aligned_controls = plate_control.reindex(treatment_plates).to_numpy(dtype=np.float32)
        if missing_policy == "global_median":
            missing_rows = np.isnan(aligned_controls).all(axis=1)
            aligned_controls[missing_rows] = np.median(control_values, axis=0)
        return treatment_values - aligned_controls

    if transform != "empirical_control_cdf_centered":
        raise SchemaError(f"Unsupported response transform {transform!r}")
    transformed = np.empty_like(treatment_values, dtype=np.float32)
    global_reference = control_values if missing_policy == "global_median" else None
    for plate in sorted(set(treatment_plates.tolist())):
        treatment_mask = treatment_plates == plate
        reference = control_values[control_plates == plate]
        if not len(reference):
            if global_reference is None:
                raise SchemaError(
                    "Empirical-control response transform encountered a treatment plate "
                    "without registered controls"
                )
            reference = global_reference
        transformed[treatment_mask] = _empirical_control_cdf_centered(
            np,
            treatment_values[treatment_mask],
            reference,
        )
    return transformed


def _cpg_metadata_columns(task: Mapping[str, Any], modality: str) -> list[str]:
    values = _cpg_value(task, f"{modality}_metadata_columns", [])
    if not isinstance(values, (list, tuple)):
        raise SchemaError(f"{modality}_metadata_columns must be a list")
    return [str(value) for value in values]


def _read_cpg_profile(task: Mapping[str, Any], modality: str):
    """Read one release CSV.GZ, selecting only profile and registered metadata columns."""
    np = _require("numpy")
    pd = _require("pandas")
    source = _cpg_source(task, modality)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    features = _cpg_feature_columns(header, modality, task)
    required = {
        *_cpg_condition_key_columns(task, modality),
        _cpg_plate_column(task, modality),
        _cpg_perturbation_column(task, modality),
        *_cpg_metadata_columns(task, modality),
    }
    missing = sorted(required.difference(header))
    if missing:
        raise SchemaError(f"Required {modality} metadata are absent: {missing}")
    use_columns = list(dict.fromkeys([*features, *sorted(required)]))
    frame = pd.read_csv(source, usecols=use_columns, low_memory=False)
    numeric = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    nonfinite = ~np.isfinite(numeric)
    # A release-profile column containing non-finite values is removed as a
    # whole rather than silently imputed.  This keeps the profile response
    # definition stable and leaves a complete manifest of the filtering.
    invalid_features = np.any(nonfinite, axis=0)
    dropped_features = [column for column, invalid in zip(features, invalid_features) if invalid]
    if dropped_features:
        keep = ~invalid_features
        frame = frame.drop(columns=dropped_features)
        features = [column for column, valid in zip(features, keep) if valid]
        numeric = numeric[:, keep]
    frame.loc[:, features] = numeric
    frame.attrs["dropped_nonfinite_features"] = tuple(dropped_features)
    return frame, features, source


def _first_present(values: Any) -> str:
    for value in values:
        if value is not None:
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text
    return ""


def _aggregate_cpg_assay(
    task: Mapping[str, Any],
    *,
    modality: str,
    prefix: str,
):
    """Control-center and condition-median one assay before cross-assay joining."""
    np = _require("numpy")
    pd = _require("pandas")
    frame, features, source = _read_cpg_profile(task, modality)
    key_columns = _cpg_condition_key_columns(task, modality)
    plate_column = _cpg_plate_column(task, modality)
    perturbation_column = _cpg_perturbation_column(task, modality)
    metadata_columns = _cpg_metadata_columns(task, modality)
    treatment_values = {str(value) for value in _cpg_value(task, "treatment_values", ["trt"])}
    control_values = {str(value) for value in _cpg_value(task, "control_values", ["control"])}
    response_transform = _cpg_response_transform(task, modality)
    labels = _cpg_condition_keys(task, modality, frame).astype(str).str.strip()
    valid_label = ~labels.str.lower().isin({"", "nan", "none"})
    pert_type = frame[perturbation_column].astype(str).str.strip()
    raw_treatment_mask = pert_type.isin(treatment_values)
    treatment_mask = valid_label & pert_type.isin(treatment_values)
    control_mask = pert_type.isin(control_values)
    controls = frame.loc[control_mask, [plate_column, *features]]
    if controls.empty:
        raise SchemaError(f"{modality} has no registered controls")
    treatment_columns = list(dict.fromkeys([*key_columns, plate_column, *metadata_columns, *features]))
    treatment = frame.loc[treatment_mask, treatment_columns].copy()
    treatment["__condition_key"] = labels.loc[treatment_mask].to_numpy()
    treatment["__source_row_id"] = frame.index[treatment_mask].astype(str)
    covered_plates = set(controls[plate_column].astype(str))
    treatment_plate = treatment[plate_column].astype(str)
    missing_plate_mask = ~treatment_plate.isin(covered_plates)
    missing_policy = str(_cpg_value(task, "missing_control_policy", "drop"))
    if missing_policy == "drop":
        treatment = treatment.loc[~missing_plate_mask].copy()
    elif missing_policy == "global_median":
        pass
    else:
        raise SchemaError("missing_control_policy must be 'drop' or 'global_median'")
    if treatment.empty:
        raise SchemaError(f"{modality} has no treatment rows after registered control policy")
    centered = _cpg_control_relative_response(
        np,
        treatment=treatment,
        controls=controls,
        features=features,
        plate_column=plate_column,
        missing_policy=missing_policy,
        transform=response_transform,
    )
    response_columns = [f"{prefix}{column}" for column in features]
    centered_frame = pd.DataFrame(centered, columns=response_columns, index=treatment.index)
    centered_frame["__condition_key"] = treatment["__condition_key"].to_numpy()
    centered_frame["__source_row_id"] = treatment["__source_row_id"].to_numpy()
    centered_frame["__plate"] = treatment[plate_column].astype(str).to_numpy()
    for column in metadata_columns:
        centered_frame[f"__meta_{prefix}{column}"] = treatment[column].to_numpy()
    grouped = centered_frame.groupby("__condition_key", sort=True)
    response = grouped[response_columns].median()
    summary = pd.DataFrame(
        {
            f"{prefix}replicate_count": grouped.size(),
            f"{prefix}plate_ids": grouped["__plate"].agg(lambda values: "|".join(sorted(set(map(str, values))))),
            f"{prefix}source_row_ids": grouped["__source_row_id"].agg(lambda values: "|".join(map(str, values))),
        }
    )
    for column in metadata_columns:
        name = f"__meta_{prefix}{column}"
        summary[f"{prefix}{column}"] = grouped[name].agg(_first_present)
    aggregate = response.join(summary).reset_index()
    diagnostics = {
        "source_path": str(source),
        "source_fingerprint": file_fingerprint(source),
        "source_rows": int(len(frame)),
        "condition_key": _cpg_condition_key_description(task, modality),
        "treatment_rows_before_control_policy": int(treatment_mask.sum()),
        "invalid_condition_key_treatment_rows": int((raw_treatment_mask & ~valid_label).sum()),
        "control_rows": int(control_mask.sum()),
        "treatment_plates": int(frame.loc[treatment_mask, plate_column].astype(str).nunique()),
        "control_plates": int(len(covered_plates)),
        "dropped_missing_control_rows": int(missing_plate_mask.sum()) if missing_policy == "drop" else 0,
        "condition_count": int(len(aggregate)),
        "feature_count": int(len(features)),
        "dropped_nonfinite_feature_count": int(len(frame.attrs.get("dropped_nonfinite_features", ()))),
        "dropped_nonfinite_features": list(frame.attrs.get("dropped_nonfinite_features", ())),
        "missing_control_policy": missing_policy,
    }
    if _cpg_value(task, "response_transform") is not None:
        diagnostics["response_transform"] = {
            "name": response_transform,
            "reference": "registered_same_plate_controls",
            "aggregation": "condition_median_after_transform",
            **(
                {
                    "empirical_distribution": "mid_cdf",
                    "center": 0.5,
                    "range": [-0.5, 0.5],
                }
                if response_transform == "empirical_control_cdf_centered"
                else {}
            ),
        }
    return aggregate, response_columns, diagnostics


def _aggregate_cpg_same_plate_control_context(
    task: Mapping[str, Any],
    *,
    prefix: str,
):
    """Aggregate an observed same-plate CP control profile for each condition.

    This is deliberately distinct from the response transform.  For every
    retained treatment row, it reads *only* registered control wells on that
    row's Cell Painting plate, takes their feature-wise median, and then
    robust-median aggregates those observed control contexts by condition.
    Treatment profile values never enter the context calculation.  The
    resulting profile is therefore an explicit context input that remains
    available for held-out treatment conditions when same-plate controls are
    observed at prediction time.
    """

    np = _require("numpy")
    pd = _require("pandas")
    modality = "cell_painting"
    frame, features, source = _read_cpg_profile(task, modality)
    key_columns = _cpg_condition_key_columns(task, modality)
    plate_column = _cpg_plate_column(task, modality)
    perturbation_column = _cpg_perturbation_column(task, modality)
    treatment_values = {str(value) for value in _cpg_value(task, "treatment_values", ["trt"])}
    control_values = {str(value) for value in _cpg_value(task, "control_values", ["control"])}
    labels = _cpg_condition_keys(task, modality, frame).astype(str).str.strip()
    valid_label = ~labels.str.lower().isin({"", "nan", "none"})
    pert_type = frame[perturbation_column].astype(str).str.strip()
    raw_treatment_mask = pert_type.isin(treatment_values)
    treatment_mask = valid_label & raw_treatment_mask
    control_mask = pert_type.isin(control_values)
    controls = frame.loc[control_mask, [plate_column, *features]].copy()
    if controls.empty:
        raise SchemaError("cell_painting has no registered controls for the context input")

    treatment_columns = list(dict.fromkeys([*key_columns, plate_column, *features]))
    treatment = frame.loc[treatment_mask, treatment_columns].copy()
    treatment["__condition_key"] = labels.loc[treatment_mask].to_numpy()
    treatment["__source_row_id"] = frame.index[treatment_mask].astype(str)
    treatment["__plate"] = treatment[plate_column].astype(str).to_numpy()

    controls["__plate"] = controls[plate_column].astype(str).to_numpy()
    control_profiles = controls.groupby("__plate", sort=True)[features].median()
    control_counts = controls.groupby("__plate", sort=True).size()
    missing_plate_mask = ~treatment["__plate"].isin(control_profiles.index)
    missing_policy = str(_cpg_value(task, "missing_control_policy", "drop"))
    if missing_policy == "drop":
        treatment = treatment.loc[~missing_plate_mask].copy()
    elif missing_policy != "global_median":
        raise SchemaError("missing_control_policy must be 'drop' or 'global_median'")
    if treatment.empty:
        raise SchemaError("cell_painting has no treatment rows after the registered context policy")

    context = control_profiles.reindex(treatment["__plate"].tolist())
    if missing_policy == "global_median":
        fallback = controls[features].median()
        context = context.fillna(fallback)
    if context.isnull().to_numpy().any():
        raise SchemaError("same-plate control context contains unresolved missing values")

    context_columns = [f"{prefix}control_context__{column}" for column in features]
    context_frame = pd.DataFrame(
        context.to_numpy(dtype=np.float32),
        columns=context_columns,
        index=treatment.index,
    )
    context_frame["__condition_key"] = treatment["__condition_key"].to_numpy()
    context_frame["__treatment_plate"] = treatment["__plate"].to_numpy()
    context_frame["__treatment_source_row_id"] = treatment["__source_row_id"].to_numpy()
    context_frame["__control_row_count"] = [
        int(control_counts.get(plate, 0)) for plate in treatment["__plate"].tolist()
    ]
    grouped = context_frame.groupby("__condition_key", sort=True)
    response = grouped[context_columns].median()
    summary = pd.DataFrame(
        {
            f"{prefix}context_treatment_plate_ids": grouped["__treatment_plate"].agg(
                lambda values: "|".join(sorted(set(map(str, values))))
            ),
            f"{prefix}context_treatment_source_row_ids": grouped["__treatment_source_row_id"].agg(
                lambda values: "|".join(map(str, values))
            ),
            f"{prefix}context_control_row_counts_by_plate": grouped["__control_row_count"].agg(
                lambda values: "|".join(map(str, values))
            ),
            f"{prefix}context_row_count": grouped.size(),
        }
    )
    aggregate = response.join(summary).reset_index()
    diagnostics = {
        "source_path": str(source),
        "source_fingerprint": file_fingerprint(source),
        "source_rows": int(len(frame)),
        "condition_key": _cpg_condition_key_description(task, modality),
        "treatment_rows_before_control_policy": int(treatment_mask.sum()),
        "invalid_condition_key_treatment_rows": int((raw_treatment_mask & ~valid_label).sum()),
        "control_rows": int(control_mask.sum()),
        "control_plates": int(len(control_profiles)),
        "dropped_missing_control_rows": int(missing_plate_mask.sum()) if missing_policy == "drop" else 0,
        "condition_count": int(len(aggregate)),
        "feature_count": int(len(features)),
        "dropped_nonfinite_feature_count": int(len(frame.attrs.get("dropped_nonfinite_features", ()))),
        "dropped_nonfinite_features": list(frame.attrs.get("dropped_nonfinite_features", ())),
        "missing_control_policy": missing_policy,
        "construction": {
            "source_rows": "registered_control_rows_only",
            "per_plate_aggregation": "featurewise_median",
            "per_condition_aggregation": "featurewise_median_over_treatment_row_contexts",
            "treatment_values_used_to_construct_context": False,
            "availability": "observed_same_plate_control_profile_for_held_out_treatment_conditions",
        },
    }
    return aggregate, features, context_columns, diagnostics


def _morgan_or_hash_features(values: list[str], *, dimensions: int):
    """Use a standard molecular fingerprint when RDKit is present; otherwise hash SMILES deterministically."""
    np = _require("numpy")
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=dimensions)
        output = np.zeros((len(values), dimensions), dtype=np.float32)
        for index, value in enumerate(values):
            molecule = Chem.MolFromSmiles(value) if value else None
            if molecule is None:
                output[index] = _hash_features([value or "<missing_smiles>"], dimensions)[0]
            else:
                output[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.float32, copy=False)
        return output, "morgan_radius2"
    except (ImportError, AttributeError):
        return _hash_features(values, dimensions), "hashed_smiles_fallback"


def _cpg_condition_features(task: Mapping[str, Any], merged: Any):
    np = _require("numpy")
    encoder = str(_cpg_value(task, "condition_encoder", "constant_context"))
    if encoder == "constant_context":
        return np.zeros((len(merged), 1), dtype=np.float32), {"encoder": encoder}
    if encoder != "smiles_morgan_plus_dose":
        raise SchemaError(f"Unknown CPG condition encoder: {encoder}")
    source = str(_cpg_value(task, "condition_metadata_source", "l1000"))
    prefix = "l1k__" if source == "l1000" else "cp__"
    smiles_column = str(_cpg_value(task, "smiles_column", "CPD_SMILES"))
    dose_column = str(_cpg_value(task, "dose_column", "pert_dose"))
    source_smiles = prefix + smiles_column
    source_dose = prefix + dose_column
    if source_smiles not in merged or source_dose not in merged:
        raise SchemaError("Chemical encoder requires registered SMILES and dose metadata")
    dimensions = int(_cpg_value(task, "chemical_feature_dim", 256))
    smiles = [str(value) if str(value).lower() not in {"nan", "none"} else "" for value in merged[source_smiles]]
    fingerprints, fingerprint_kind = _morgan_or_hash_features(smiles, dimensions=dimensions)
    dose = np.asarray(merged[source_dose], dtype=np.float32)
    dose = np.log10(np.clip(dose, 1e-8, None))[:, None]
    return np.concatenate([fingerprints, dose], axis=1), {
        "encoder": encoder,
        "fingerprint": fingerprint_kind,
        "dimensions": dimensions,
    }


def _registered_condition_layout_for_array(
    task: Mapping[str, Any],
    *,
    condition_dim: int,
) -> dict[str, Any] | None:
    """Validate an optional task-level condition layout against real arrays.

    The adapter never assumes that a 257-dimensional vector has a particular
    layout.  If a task registers the prefix-plus-final-scalar contract, the
    cache must have exactly the declared width; otherwise the layout remains
    absent and only generic architectures are executable.
    """

    layout = normalize_condition_layout(task.get("condition_layout"))
    if layout is None:
        return None
    expected = int(layout["prefix_dim"]) + 1
    if int(condition_dim) != expected:
        raise SchemaError(
            "Registered condition_layout expects "
            f"{expected} columns (prefix_dim + final scalar), but cache has {condition_dim}"
        )
    return layout


def _cpg_split_groups(task: Mapping[str, Any], merged: Any) -> tuple[list[str], str]:
    split = _cpg_value(task, "split", {})
    if not isinstance(split, Mapping):
        raise SchemaError("CPG split must be an object")
    source = str(split.get("group_source", "cell_painting"))
    column = str(split.get("group_column", ""))
    if not column:
        raise SchemaError("CPG split is missing group_column")
    prefix = "cp__" if source in {"cell_painting", "cp"} else "l1k__"
    source_column = prefix + column
    if source_column not in merged:
        raise SchemaError(f"CPG split group column is absent after aggregation: {source_column}")
    kind = str(split.get("kind", "stable_group_hash"))
    values = [str(value).strip() for value in merged[source_column]]
    if kind == "murcko_scaffold":
        # Use actual Murcko strings, not the fingerprint values, as split groups.
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            groups = []
            for value in values:
                molecule = Chem.MolFromSmiles(value) if value else None
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) if molecule is not None else ""
                # Acyclic valid compounds have an empty Murcko core; retain
                # their canonical string as a non-overlapping split group.
                groups.append(scaffold or value)
        except ImportError:
            groups = values
    elif kind == "stable_group_hash":
        groups = values
    else:
        raise SchemaError(f"Unknown CPG split kind: {kind}")
    if any(not group or group.lower() in {"nan", "none"} for group in groups):
        raise SchemaError("CPG split group contains missing values; register a complete group field")
    return groups, kind


def _cpg_registered_count_gate(task: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return an explicitly registered replicate-count gate, if a task has one."""
    value = _cpg_value(task, "count_gate")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaError("CPG count_gate must be an object")
    minima = value.get("minimum_replicates", value.get("min_replicates"))
    if not isinstance(minima, Mapping):
        raise SchemaError("CPG count_gate must declare minimum_replicates by assay")
    try:
        cp_minimum = int(minima["cell_painting"])
        l1k_minimum = int(minima["l1000"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError("CPG count_gate requires integer cell_painting and l1000 minima") from exc
    if cp_minimum < 1 or l1k_minimum < 1:
        raise SchemaError("CPG count-gate replicate minima must be positive")
    return cp_minimum, l1k_minimum


def _cpg_split_manifest(task: Mapping[str, Any], *, kind: str, group_count: int) -> dict[str, Any]:
    """Expose the registered split source and human-facing fold roles in cache provenance."""
    split = _cpg_value(task, "split", {})
    if not isinstance(split, Mapping):
        raise SchemaError("CPG split must be an object")
    manifest: dict[str, Any] = {"kind": kind, "group_count": group_count, "fold_count": 5}
    for key in ("group_source", "group_column", "source_semantics", "assignment", "roles"):
        if key in split:
            manifest[key] = split[key]
    default_validation = task.get("default_validation_fold")
    default_test = task.get("default_test_fold")
    if default_validation is not None or default_test is not None:
        manifest["default_roles"] = {
            "validation_fold": int(default_validation if default_validation is not None else 4),
            "test_fold": int(default_test if default_test is not None else 5),
        }
    return manifest


def _apply_cpg_registered_count_gate(task: Mapping[str, Any], merged: Any):
    """Remove only preregistered low-replicate paired conditions, with records."""
    gate = _cpg_registered_count_gate(task)
    if gate is None:
        return merged, [], {"registered": False, "pre_gate_matched_conditions": int(len(merged))}
    cp_minimum, l1k_minimum = gate
    keep = (
        merged["cp__replicate_count"].astype(int) >= cp_minimum
    ) & (
        merged["l1k__replicate_count"].astype(int) >= l1k_minimum
    )
    excluded = merged.loc[~keep]
    records = [
        {
            "condition_key": str(row["__condition_key"]),
            "disposition": "excluded",
            "reason": "registered_replicate_count_gate",
            "cp_replicates": int(row["cp__replicate_count"]),
            "l1000_replicates": int(row["l1k__replicate_count"]),
            "minimum_cp_replicates": cp_minimum,
            "minimum_l1000_replicates": l1k_minimum,
            "cp_plate_ids": str(row["cp__plate_ids"]),
            "l1000_plate_ids": str(row["l1k__plate_ids"]),
            "cp_source_row_ids": str(row["cp__source_row_ids"]),
            "l1000_source_row_ids": str(row["l1k__source_row_ids"]),
        }
        for _, row in excluded.iterrows()
    ]
    return (
        merged.loc[keep].copy(),
        records,
        {
            "registered": True,
            "minimum_replicates": {"cell_painting": cp_minimum, "l1000": l1k_minimum},
            "pre_gate_matched_conditions": int(len(merged)),
            "excluded_conditions": int(len(records)),
            "retained_conditions": int(keep.sum()),
        },
    )


def prepare_cpg_replicate_profiles(task: Mapping[str, Any], output: str | Path) -> Path:
    """Build a condition-level, control-relative dual-assay cache from CPG CSV.GZ sources.

    Replicates are centered and median-aggregated within each assay *before*
    the exact condition join.  This prevents a row-level Cartesian product and
    keeps every source condition/provenance record explicit.
    """
    np = _require("numpy")
    direction = str(_cpg_value(task, "direction", "cp_to_l1000"))
    cp, cp_features, cp_info = _aggregate_cpg_assay(task, modality="cell_painting", prefix="cp__")
    context_columns: list[str] | None = None
    context_info: Mapping[str, Any] | None = None
    if direction == "joint_from_condition_with_cp_control_context":
        context, context_features, context_columns, context_info = _aggregate_cpg_same_plate_control_context(
            task,
            prefix="cp_",
        )
        if [f"cp__{feature}" for feature in context_features] != cp_features:
            raise SchemaError(
                "same-plate control context features differ from the registered Cell Painting response features"
            )
        source_condition_count = len(cp)
        cp = cp.merge(context, on="__condition_key", how="inner", validate="one_to_one")
        if len(cp) != source_condition_count:
            raise SchemaError(
                "same-plate control context does not cover every retained Cell Painting condition"
            )
    l1k, l1k_features, l1k_info = _aggregate_cpg_assay(task, modality="l1000", prefix="l1k__")
    merged = cp.merge(l1k, on="__condition_key", how="inner", validate="one_to_one")
    if merged.empty:
        raise SchemaError("No exact paired conditions remain after assay-level aggregation")
    merged, count_gate_exclusions, count_gate_info = _apply_cpg_registered_count_gate(task, merged)
    if merged.empty:
        raise SchemaError("No paired conditions remain after the registered replicate-count gate")
    cp_response = merged[cp_features].to_numpy(dtype=np.float32)
    l1k_response = merged[l1k_features].to_numpy(dtype=np.float32)
    condition, condition_info = _cpg_condition_features(task, merged)
    condition_layout = _registered_condition_layout_for_array(
        task,
        condition_dim=int(condition.shape[1]),
    )
    if direction == "cp_to_l1000":
        pre, target = cp_response, l1k_response
    elif direction == "l1000_to_cp":
        pre, target = l1k_response, cp_response
    elif direction == "joint_from_condition":
        pre = np.zeros((len(merged), 1), dtype=np.float32)
        target = np.concatenate([cp_response, l1k_response], axis=1)
    elif direction == "joint_from_condition_with_cp_control_context":
        if context_columns is None:
            raise SchemaError("same-plate control context columns are missing")
        pre = merged[context_columns].to_numpy(dtype=np.float32)
        target = np.concatenate([cp_response, l1k_response], axis=1)
    else:
        raise SchemaError(
            "CPG direction must be cp_to_l1000, l1000_to_cp, joint_from_condition, "
            "or joint_from_condition_with_cp_control_context"
        )
    groups, split_kind = _cpg_split_groups(task, merged)
    folds = np.asarray([_stable_fold(group) for group in groups], dtype=np.int8)
    labels = [str(value) for value in merged["__condition_key"]]
    target_path = Path(output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    label_width = max(1, max(map(len, labels)))
    group_width = max(1, max(map(len, groups)))
    cache_arrays = {
        "pre": pre.astype(np.float32),
        "condition": condition.astype(np.float32),
        "target": target.astype(np.float32),
        "cp_response": cp_response,
        "l1000_response": l1k_response,
        "fold": folds,
        "labels": np.asarray(labels, dtype=f"<U{label_width}"),
        "split_groups": np.asarray(groups, dtype=f"<U{group_width}"),
        "cp_replicate_count": merged["cp__replicate_count"].to_numpy(dtype=np.int32),
        "l1000_replicate_count": merged["l1k__replicate_count"].to_numpy(dtype=np.int32),
    }
    if direction == "joint_from_condition_with_cp_control_context":
        # Make the observed input explicit rather than relying on the generic
        # evaluator alias alone.  It is byte-identical to ``pre``.
        cache_arrays["cp_same_plate_control_context"] = pre.astype(np.float32)
    np.savez_compressed(target_path, **cache_arrays)
    manifest = {
        "schema_version": 2,
        "adapter": "cpg_replicate_profiles",
        "response_semantics": "within-assay control-relative condition median",
        "direction": direction,
        "profile_selection": _cpg_value(task, "profile_selection", {}),
        "join": {
            "cell_painting": _cpg_condition_key_description(task, "cell_painting"),
            "l1000": _cpg_condition_key_description(task, "l1000"),
            "association_unit": "condition_level_cross_assay_not_same_well",
            "pre_count_gate_matched_conditions": int(count_gate_info["pre_gate_matched_conditions"]),
            "matched_conditions": int(len(merged)),
        },
        "count_gate": count_gate_info,
        "split": _cpg_split_manifest(task, kind=split_kind, group_count=int(len(set(groups)))),
        "condition_encoder": condition_info,
        "cell_painting": {**cp_info, "response_feature_count": int(cp_response.shape[1])},
        "l1000": {**l1k_info, "response_feature_count": int(l1k_response.shape[1])},
        "cache_shapes": {
            "pre": list(pre.shape),
            "condition": list(condition.shape),
            "target": list(target.shape),
        },
        "provenance": {
            "retained_conditions": str(target_path.with_suffix(".provenance.jsonl")),
            "excluded_conditions": str(target_path.with_suffix(".excluded_conditions.jsonl")),
        },
    }
    if direction == "joint_from_condition_with_cp_control_context":
        if context_info is None:
            raise SchemaError("same-plate control context diagnostics are missing")
        manifest["context"] = {
            "input": "same_plate_cell_painting_control_profile",
            "input_feature_count": int(pre.shape[1]),
            "source": context_info,
            "target_relation": (
                "context is an observed input only; the target remains the registered "
                "joint control-rank Cell Painting plus control-relative L1000 response"
            ),
            "target_mode_requirement": "absolute",
        }
    if condition_layout is not None:
        # The layout belongs in the immutable cache provenance as well as the
        # task contract.  This makes a stale cache fail loudly instead of
        # silently treating a final feature as an intensity scalar.
        manifest["condition_layout"] = condition_layout
    write_json(target_path.with_suffix(".manifest.json"), manifest)
    provenance_path = target_path.with_suffix(".provenance.jsonl")
    with provenance_path.open("w", encoding="utf-8") as handle:
        for _, row in merged.iterrows():
            record = {
                "condition_key": str(row["__condition_key"]),
                "disposition": "retained",
                "association_unit": "condition_level_cross_assay_not_same_well",
                "cp_replicates": int(row["cp__replicate_count"]),
                "l1000_replicates": int(row["l1k__replicate_count"]),
                "cp_plate_ids": str(row["cp__plate_ids"]),
                "l1000_plate_ids": str(row["l1k__plate_ids"]),
                "cp_source_row_ids": str(row["cp__source_row_ids"]),
                "l1000_source_row_ids": str(row["l1k__source_row_ids"]),
            }
            if direction == "joint_from_condition_with_cp_control_context":
                record["cp_control_context"] = {
                    "source_rows": "registered_control_rows_only",
                    "treatment_plate_ids": str(row["cp_context_treatment_plate_ids"]),
                    "treatment_source_row_ids": str(row["cp_context_treatment_source_row_ids"]),
                    "control_row_counts_by_plate": str(
                        row["cp_context_control_row_counts_by_plate"]
                    ),
                    "aggregated_context_rows": int(row["cp_context_row_count"]),
                }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    excluded_path = target_path.with_suffix(".excluded_conditions.jsonl")
    with excluded_path.open("w", encoding="utf-8") as handle:
        for record in count_gate_exclusions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target_path


def inspect_cpg_task(task_id: str, task: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect raw CPG sources without constructing an analysis cache."""
    pd = _require("pandas")
    result: dict[str, Any] = {"task_id": task_id, "adapter": "cpg_replicate_profiles", "sources": {}}
    for modality in ("cell_painting", "l1000"):
        source = _cpg_source(task, modality)
        header = pd.read_csv(source, nrows=0).columns.tolist()
        result["sources"][modality] = {
            "path": str(source),
            "exists": source.exists(),
            "bytes": source.stat().st_size,
            "feature_count": len(_cpg_feature_columns(header, modality, task)),
            "condition_key": _cpg_condition_key_description(task, modality),
        }
    return result


def inspect_norman_task(task_id: str, task: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect the source metadata used by the raw Norman adapter."""
    np = _require("numpy")
    h5py = _require("h5py")
    rna = _task_block(task, "rna")
    source = Path(str(task["path"]))
    result: dict[str, Any] = {
        "task_id": task_id,
        "adapter": task.get("adapter"),
        "path": str(source),
        "exists": source.exists(),
    }
    if not source.exists():
        return result
    with h5py.File(source, "r") as handle:
        obs = handle["obs"]
        condition_column = str(rna.get("condition_column", "condition"))
        control_column = str(rna.get("control_column", "control"))
        conditions = _categorical(obs, condition_column)
        controls = np.asarray(obs[control_column][:], dtype=bool)
        matrix = _h5_path(handle, str(rna.get("matrix_path", "layers/counts")))
        cell_types = sorted(set(_categorical(obs, "cell_type"))) if "cell_type" in obs else []
        result.update(
            {
                "bytes": source.stat().st_size,
                "raw_count_matrix": str(rna.get("matrix_path", "layers/counts")),
                "raw_count_shape": [int(value) for value in matrix.shape],
                "condition_column": condition_column,
                "source_condition_count": len(set(conditions)),
                "control_column": control_column,
                "control_cell_count": int(controls.sum()),
                "cell_type_values": cell_types,
                "categorical_encoding": "decoded_legacy_reference_or_group_style",
            }
        )
    return result


def _task_block(task: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = task.get(name, {})
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} settings must be an object")
    return value


def _h5_path(handle: Any, path: str):
    node = handle
    for part in path.strip("/").split("/"):
        if not part:
            continue
        if not hasattr(node, "__contains__") or part not in node:
            raise SchemaError(f"H5AD source has no registered matrix path {path!r}")
        node = node[part]
    return node


def _canonical_norman_condition(value: str, control_token: str) -> str:
    """Make a perturbation combination order-invariant without biological lookup.

    The supplied source contains both ``A+ctrl`` and ``ctrl+A`` conventions.
    Canonicalization only sorts observed text tokens and removes the registered
    control token; it adds no gene sets, pathways, or external evidence.
    """
    normalized_control = control_token.strip().lower()
    tokens = [token.strip() for token in str(value).split("+") if token.strip()]
    if not tokens:
        raise SchemaError("Norman condition contains no perturbation tokens")
    non_control = [token for token in tokens if token.lower() != normalized_control]
    return "+".join(sorted(non_control)) if non_control else control_token


def _norman_split_folds(
    labels: list[str],
    *,
    control_token: str,
    seed: int,
    fold_count: int,
):
    """Assign only canonical double perturbations to seeded disjoint folds."""
    np = _require("numpy")
    folds = np.full(len(labels), -1, dtype=np.int8)
    double_labels: list[str] = []
    for label in labels:
        if label == control_token:
            continue
        tokens = label.split("+")
        if len(tokens) == 1:
            continue
        if len(tokens) != 2:
            raise SchemaError(
                "Norman raw adapter is registered for single and double perturbations only; "
                f"found {label!r}"
            )
        double_labels.append(label)
    if len(double_labels) < fold_count:
        raise SchemaError("Not enough double perturbations for the registered five-fold split")
    label_index = {label: index for index, label in enumerate(labels)}
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(double_labels), dtype=object)[rng.permutation(len(double_labels))]
    for fold, fold_labels in enumerate(np.array_split(shuffled, fold_count)):
        for label in fold_labels.tolist():
            folds[label_index[str(label)]] = fold
    return folds, sorted(double_labels)


def _string_array(np: Any, values: list[str]):
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def prepare_norman_count_response(task: Mapping[str, Any], output: str | Path) -> Path:
    """Build a raw-count-normalized Norman condition-response cache.

    This adapter deliberately reads ``layers/counts`` rather than the already
    normalized ``X`` matrix.  It calculates ``log1p(10,000 * counts / library
    size)`` per cell, then computes deterministic condition means and subtracts
    the source-registered control mean.  The output retains all 5,045 supplied
    genes, avoiding test-informed feature selection.
    """
    np = _require("numpy")
    h5py = _require("h5py")
    rna = _task_block(task, "rna")
    source = Path(str(task["path"]))
    if not source.exists():
        raise FileNotFoundError(f"Registered Norman source does not exist: {source}")
    matrix_path = str(rna.get("matrix_path", "layers/counts"))
    condition_column = str(rna.get("condition_column", "condition"))
    control_column = str(rna.get("control_column", "control"))
    control_token = str(rna.get("control_token", "ctrl"))
    library_size = float(rna.get("library_size", 10_000.0))
    chunk_rows = int(rna.get("chunk_rows", 1_024))
    fold_count = int(rna.get("fold_count", 5))
    split_seed = int(rna.get("split_seed", 20_260_728))
    expected_cell_type = rna.get("expected_cell_type")
    condition_feature_dim = int(rna.get("condition_feature_dim", 128))
    if library_size <= 0 or chunk_rows <= 0 or fold_count != 5:
        raise SchemaError("Norman task requires positive normalization/chunk settings and exactly five folds")

    with h5py.File(source, "r") as handle:
        if "obs" not in handle or "var" not in handle:
            raise SchemaError("Norman H5AD is missing obs or var metadata")
        obs = handle["obs"]
        if condition_column not in obs or control_column not in obs:
            raise SchemaError("Norman H5AD is missing the registered condition/control metadata")
        source_conditions = _categorical(obs, condition_column)
        control_flags = np.asarray(obs[control_column][:], dtype=bool)
        if len(source_conditions) != len(control_flags):
            raise SchemaError("Norman condition and control arrays have inconsistent lengths")
        canonical_conditions = [
            _canonical_norman_condition(value, control_token) for value in source_conditions
        ]
        if not control_flags.any():
            raise SchemaError("Norman H5AD has no source-registered control cells")
        is_control_label = np.asarray([label == control_token for label in canonical_conditions], dtype=bool)
        if np.any(control_flags & ~is_control_label):
            raise SchemaError("Registered control cells do not use the configured control condition token")
        if np.any(~control_flags & is_control_label):
            raise SchemaError("The configured control condition includes non-control observations")

        cell_type_values: list[str] = []
        if "cell_type" in obs:
            cell_type_values = sorted(set(_categorical(obs, "cell_type")))
        if expected_cell_type is not None and cell_type_values != [str(expected_cell_type)]:
            raise SchemaError(
                f"Norman source cell-type metadata is {cell_type_values}, not the registered {expected_cell_type!r}"
            )

        matrix = _h5_path(handle, matrix_path)
        if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
            raise SchemaError(f"Registered count matrix {matrix_path!r} is not a dense two-dimensional dataset")
        n_cells, n_genes = (int(value) for value in matrix.shape)
        if n_cells != len(canonical_conditions):
            raise SchemaError("Norman count matrix row count disagrees with obs metadata")
        var = handle["var"]
        gene_column = str(rna.get("gene_column", "gene_name"))
        if gene_column in var:
            genes = _categorical(var, gene_column)
        elif "_index" in var:
            genes = _decode(var["_index"][:])
        else:
            genes = [f"gene_{index}" for index in range(n_genes)]
        if len(genes) != n_genes:
            raise SchemaError("Norman gene metadata length disagrees with count matrix")

        labels = sorted(set(canonical_conditions))
        if control_token not in labels:
            raise SchemaError("Norman canonical conditions contain no control label")
        label_index = {label: index for index, label in enumerate(labels)}
        condition_codes = np.asarray([label_index[label] for label in canonical_conditions], dtype=np.int32)
        condition_counts = np.bincount(condition_codes, minlength=len(labels)).astype(np.int64)
        if np.any(condition_counts == 0):
            raise SchemaError("Norman condition cache contains an empty canonical condition")

        sums = np.zeros((len(labels), n_genes), dtype=np.float64)
        for start in range(0, n_cells, chunk_rows):
            stop = min(start + chunk_rows, n_cells)
            counts = np.asarray(matrix[start:stop], dtype=np.float32)
            if not np.isfinite(counts).all() or np.any(counts < 0):
                raise SchemaError("Norman raw count matrix contains non-finite or negative values")
            library = counts.sum(axis=1, dtype=np.float64)
            if np.any(library <= 0):
                raise SchemaError("Norman raw count matrix contains zero-library observations")
            normalized = np.log1p(counts * (library_size / library)[:, None])
            np.add.at(sums, condition_codes[start:stop], normalized)

    normalized_profiles = (sums / condition_counts[:, None]).astype(np.float32)
    control_index = label_index[control_token]
    control_profile = normalized_profiles[control_index]
    response = normalized_profiles - control_profile[None, :]
    pre = np.repeat(control_profile[None, :], len(labels), axis=0).astype(np.float32)
    condition = _hash_features(labels, dimensions=condition_feature_dim)
    folds, double_labels = _norman_split_folds(
        labels,
        control_token=control_token,
        seed=split_seed,
        fold_count=fold_count,
    )
    source_labels_by_canonical: dict[str, set[str]] = {label: set() for label in labels}
    control_count_by_canonical = np.zeros(len(labels), dtype=np.int64)
    for source_label, canonical_label, is_control in zip(source_conditions, canonical_conditions, control_flags):
        index = label_index[canonical_label]
        source_labels_by_canonical[canonical_label].add(source_label)
        control_count_by_canonical[index] += int(is_control)

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = target.with_name(f".{target.stem}.tmp.npz")
    np.savez_compressed(
        temporary_cache,
        pre=pre,
        condition=condition.astype(np.float32),
        target=response.astype(np.float32),
        normalized_condition_profile=normalized_profiles,
        control_profile=control_profile.astype(np.float32),
        fold=folds,
        labels=_string_array(np, labels),
        split_groups=_string_array(np, labels),
        genes=_string_array(np, genes),
        condition_cell_count=condition_counts,
        control_cell_count=control_count_by_canonical,
        is_double=np.asarray([label != control_token and "+" in label for label in labels], dtype=np.bool_),
    )
    temporary_cache.replace(target)
    manifest = {
        "schema_version": 2,
        "adapter": "norman_h5ad_count_response",
        "source": {
            "path": str(source),
            "fingerprint": file_fingerprint(source),
            "raw_count_matrix": matrix_path,
            "shape": [n_cells, n_genes],
            "cell_type_values": cell_type_values,
        },
        "profile": {
            "normalization": "per_cell_library_size_then_log1p",
            "library_size": library_size,
            "aggregation": "condition_mean",
            "feature_policy": "all_source_genes_no_test_informed_selection",
            "gene_count": n_genes,
        },
        "conditions": {
            "source_condition_column": condition_column,
            "source_condition_count": len(set(source_conditions)),
            "canonical_condition_count": len(labels),
            "control_column": control_column,
            "control_token": control_token,
            "control_cell_count": int(control_flags.sum()),
            "canonicalization": "split_plus_sort_tokens_remove_control_token",
            "response": "condition_mean_minus_registered_control_mean",
        },
        "split": {
            "kind": "seeded_pair_disjoint_five_fold_over_double_perturbations",
            "seed": split_seed,
            "fold_count": fold_count,
            "double_condition_count": len(double_labels),
            "single_and_control_role": "train_only",
            "default_validation_fold": 4,
            "default_test_fold": 5,
        },
        "cache_shapes": {
            "pre": list(pre.shape),
            "condition": list(condition.shape),
            "target": list(response.shape),
        },
    }
    manifest_path = target.with_suffix(".manifest.json")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    write_json(temporary_manifest, manifest)
    temporary_manifest.replace(manifest_path)
    provenance_path = target.with_suffix(".provenance.jsonl")
    temporary_provenance = provenance_path.with_name(f".{provenance_path.name}.tmp")
    with temporary_provenance.open("w", encoding="utf-8") as handle:
        for index, label in enumerate(labels):
            record = {
                "canonical_condition": label,
                "source_conditions": sorted(source_labels_by_canonical[label]),
                "cell_count": int(condition_counts[index]),
                "control_cell_count": int(control_count_by_canonical[index]),
                "fold": int(folds[index]),
                "role": "train_only" if int(folds[index]) < 0 else f"double_fold_{int(folds[index]) + 1}",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary_provenance.replace(provenance_path)
    return target


def prepare_norman_pseudobulk(task: Mapping[str, Any], output: str | Path) -> Path:
    """Backward-compatible name for the raw Norman count-response adapter."""
    return prepare_norman_count_response(task, output)


def prepare_crisprmap_rna(task: Mapping[str, Any], output: str | Path) -> Path:
    """Create a compact RNA reporter-response cache from the complete local H5AD."""
    np = _require("numpy")
    h5py = _require("h5py")
    source = Path(str(task["path"]))
    with h5py.File(source, "r") as handle:
        modalities = _categorical(handle["var"], "modality")
        rna_indices = np.asarray([idx for idx, modality in enumerate(modalities) if modality == "RNA"], dtype=np.int32)
        if not len(rna_indices):
            raise SchemaError("CRISPRmap source has no registered RNA features")
        x = handle["X"][:, rna_indices].astype(np.float32)
        genes = _categorical(handle["obs"], "perturbation_gene")
        treatments = _categorical(handle["obs"], "treatment")
        controls = handle["obs"]["is_control"][:].astype(bool)
    treatments_unique = sorted(set(treatments))
    control_by_treatment = {
        treatment: x[np.asarray([(row_treatment == treatment and is_control) for row_treatment, is_control in zip(treatments, controls)])].mean(axis=0)
        for treatment in treatments_unique
    }
    labels = sorted(set(f"{gene}|{treatment}" for gene, treatment in zip(genes, treatments)))
    label_index = {label: idx for idx, label in enumerate(labels)}
    codes = np.asarray([label_index[f"{gene}|{treatment}"] for gene, treatment in zip(genes, treatments)], dtype=np.int32)
    sums = np.zeros((len(labels), len(rna_indices)), dtype=np.float64)
    np.add.at(sums, codes, x)
    counts = np.bincount(codes, minlength=len(labels))
    means = (sums / counts[:, None]).astype(np.float32)
    label_treatments = [label.split("|", 1)[1] for label in labels]
    pre = np.stack([control_by_treatment[treatment] for treatment in label_treatments]).astype(np.float32)
    response = means - pre
    condition = _hash_features(labels)
    folds = np.asarray([_stable_fold(label.split("|", 1)[0]) for label in labels], dtype=np.int8)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        pre=pre,
        condition=condition,
        target=response,
        fold=folds,
        labels=_string_array(np, labels),
        source_path=str(source),
    )
    return target


def default_cache_path(task_id: str) -> Path:
    return PROJECT_ROOT / "cache" / f"{task_id}.npz"


def load_rna_arrays(
    task_id: str,
    task: Mapping[str, Any],
    *,
    cache_path: str | Path | None = None,
    test_fold: int | None = None,
    validation_fold: int | None = None,
    seed: int = 0,
) -> TaskArrays:
    np = _require("numpy")
    cache = Path(cache_path or default_cache_path(task_id))
    if not cache.exists():
        raise FileNotFoundError(f"RNA cache {cache} does not exist; run prepare-rna first")
    with np.load(cache, allow_pickle=False) as data:
        pre, condition, target, fold = (data[name].astype(np.float32, copy=False) if name != "fold" else data[name] for name in ("pre", "condition", "target", "fold"))
    test_number = int(test_fold if test_fold is not None else task.get("default_test_fold", 5))
    validation_number = int(
        validation_fold
        if validation_fold is not None
        else task.get("default_validation_fold", ((test_number % 5) + 1))
    )
    test_index = (test_number - 1) % 5
    val_index = (validation_number - 1) % 5
    if test_index == val_index:
        raise SchemaError("RNA validation and test folds must be distinct")
    train_indices = np.flatnonzero((fold != test_index) & (fold != val_index))
    val_indices = np.flatnonzero(fold == val_index)
    test_indices = np.flatnonzero(fold == test_index)
    if not len(train_indices) or not len(val_indices) or not len(test_indices):
        raise SchemaError(f"RNA task {task_id} does not contain all fold roles")
    train_indices = _subsample(train_indices, task.get("max_train_samples"), seed)
    return TaskArrays(
        task_id=task_id,
        train_pre=pre[train_indices],
        train_condition=condition[train_indices],
        train_target=target[train_indices],
        validation_pre=pre[val_indices],
        validation_condition=condition[val_indices],
        validation_target=target[val_indices],
        test_pre=pre[test_indices],
        test_condition=condition[test_indices],
        test_target=target[test_indices],
        metadata={"family": "rna_perturbation", "test_fold": test_index, "validation_fold": val_index, "cache": str(cache)},
    )


def load_cpg_arrays(
    task_id: str,
    task: Mapping[str, Any],
    *,
    cache_path: str | Path | None = None,
    test_fold: int | None = None,
    validation_fold: int | None = None,
    seed: int = 0,
) -> TaskArrays:
    """Load an immutable cache created directly from CPG release CSV.GZ files."""
    np = _require("numpy")
    cache = Path(cache_path or default_cache_path(task_id))
    if not cache.exists():
        raise FileNotFoundError(f"CPG cache {cache} does not exist; run prepare-cpg first")
    with np.load(cache, allow_pickle=False) as data:
        pre = data["pre"].astype(np.float32, copy=False)
        condition = data["condition"].astype(np.float32, copy=False)
        target = data["target"].astype(np.float32, copy=False)
        fold = data["fold"]
        labels = data["labels"].astype(str)
        split_groups = data["split_groups"].astype(str)
    condition_layout = _registered_condition_layout_for_array(
        task,
        condition_dim=int(condition.shape[1]),
    )
    manifest_path = cache.with_suffix(".manifest.json")
    if condition_layout is not None:
        if not manifest_path.exists():
            raise SchemaError(
                "CPG cache is missing the condition-layout manifest; rerun prepare-cpg "
                "for this registered task before using intensity_gated_response_tower"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"CPG cache manifest is invalid JSON: {manifest_path}") from exc
        recorded_layout = normalize_condition_layout(manifest.get("condition_layout"))
        if recorded_layout != condition_layout:
            raise SchemaError(
                "CPG cache condition_layout does not match the registered task layout; "
                "rerun prepare-cpg before evaluation"
            )
    test_number = int(test_fold if test_fold is not None else task.get("default_test_fold", 5))
    validation_number = int(
        validation_fold
        if validation_fold is not None
        else task.get("default_validation_fold", ((test_number % 5) + 1))
    )
    test_index = (test_number - 1) % 5
    val_index = (validation_number - 1) % 5
    if test_index == val_index:
        raise SchemaError("CPG validation and test folds must be distinct")
    train_indices = np.flatnonzero((fold != test_index) & (fold != val_index))
    val_indices = np.flatnonzero(fold == val_index)
    test_indices = np.flatnonzero(fold == test_index)
    if not len(train_indices) or not len(val_indices) or not len(test_indices):
        raise SchemaError(f"CPG task {task_id} does not contain all condition-level fold roles")
    train_indices = _subsample(train_indices, task.get("max_train_samples"), seed)
    metadata = {
        "family": "paired_multi_assay_response",
        "test_fold": test_number,
        "validation_fold": validation_number,
        "cache": str(cache),
        "condition_count": int(len(labels)),
        "split_group_count": int(len(set(split_groups.tolist()))),
        "manifest": str(manifest_path),
    }
    if condition_layout is not None:
        metadata["condition_layout"] = condition_layout
    return TaskArrays(
        task_id=task_id,
        train_pre=pre[train_indices],
        train_condition=condition[train_indices],
        train_target=target[train_indices],
        validation_pre=pre[val_indices],
        validation_condition=condition[val_indices],
        validation_target=target[val_indices],
        test_pre=pre[test_indices],
        test_condition=condition[test_indices],
        test_target=target[test_indices],
        metadata=metadata,
    )


def load_task_arrays(
    task_id: str,
    *,
    config_path: str | Path = DEFAULT_TASKS_PATH,
    cache_path: str | Path | None = None,
    test_fold: int | None = None,
    validation_fold: int | None = None,
    seed: int = 0,
) -> TaskArrays:
    task = get_task(task_id, config_path)
    adapter = str(task.get("adapter"))
    if adapter == "bbbc_h5":
        return load_bbbc_arrays(task_id, task, test_fold=test_fold, validation_fold=validation_fold, seed=seed)
    if adapter == "cpg_replicate_profiles":
        return load_cpg_arrays(
            task_id,
            task,
            cache_path=cache_path,
            test_fold=test_fold,
            validation_fold=validation_fold,
            seed=seed,
        )
    if adapter in {"norman_h5ad_count_response", "norman_h5ad_pseudobulk", "crisprmap_h5ad_rna"}:
        return load_rna_arrays(
            task_id,
            task,
            cache_path=cache_path,
            test_fold=test_fold,
            validation_fold=validation_fold,
            seed=seed,
        )
    raise SchemaError(f"No adapter implementation for {adapter}")
