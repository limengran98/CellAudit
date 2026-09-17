"""Metadata-preserving BBBC036 views and target-blind semantic sham maps.

This module provides the data boundary for the perturbation-hypothesis loop: a model sees
the same registered Cell Painting control profile, chemical fingerprint, and
dose as the real task, while paired semantic shams can replace *only* the
meaningful association under a fully recorded, target-blind map.

The map builders accept no response target.  They therefore cannot select a
counterfactual from outcome values.  They operate only on inputs and stable
row identities, and expose the complete source-to-donor mapping for later
audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import SchemaError, canonical_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.npz"
DEFAULT_MANIFEST = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.manifest.json"
DEFAULT_PROVENANCE = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.provenance.jsonl"
TASK_ID = "cpg036_cp_plate_control_context"


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("Perturbation-loop data loading requires numpy") from exc
    return np


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _vector_hash(vector: Any, *, namespace: str) -> str:
    """Hash a vector in one canonical float32 representation.

    The identifier describes an observed input profile, not a response target.
    Including the shape guards against a theoretical collision between a flat
    array and a differently shaped byte-identical array.
    """

    np = _require_numpy()
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return f"{namespace}:{digest.hexdigest()}"


def _stable_key(*parts: object) -> str:
    return _sha256_text("|".join(str(part) for part in parts))


def _decode(values: Any) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


@dataclass(frozen=True)
class PerturbationRowProvenance:
    """Raw-source identifiers retained alongside one condition-level row."""

    cache_index: int
    cache_fold: int
    fold: int
    label: str
    context_profile_id: str
    profile_id: str
    context_source_id: str
    cp_plate_ids: str
    l1000_plate_ids: str
    cp_source_row_ids: str
    l1000_source_row_ids: str


@dataclass(frozen=True)
class PerturbationPartition:
    """One immutable BBBC036 partition with semantic input fields separated."""

    name: str
    pre: Any
    chemical_prefix: Any
    dose: Any
    target: Any
    cp_target: Any
    l1000_target: Any
    labels: tuple[str, ...]
    cache_indices: Any
    cache_folds: Any
    folds: Any
    context_profile_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    context_source_ids: tuple[str, ...]
    provenance: tuple[PerturbationRowProvenance, ...]

    @property
    def count(self) -> int:
        return int(self.pre.shape[0])

    @property
    def fold(self) -> int | None:
        """Return the human-readable fold for a single-fold partition only."""

        np = _require_numpy()
        values = np.unique(self.folds)
        return int(values[0]) if len(values) == 1 else None

    @property
    def condition(self) -> Any:
        """Reconstruct the registered [chemical fingerprint | dose] layout."""

        np = _require_numpy()
        return np.concatenate([self.chemical_prefix, self.dose[:, None]], axis=1)


@dataclass(frozen=True)
class PerturbationSelectionArrays:
    """Fit folds 1--2 plus feedback fold 3; no endpoint partition is exposed."""

    task_id: str
    fit: PerturbationPartition
    feedback: PerturbationPartition
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PerturbationEndpointArrays:
    """Fit folds 1--2 plus the registered endpoint fold 4 only."""

    task_id: str
    fit: PerturbationPartition
    endpoint: PerturbationPartition
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ControlProfileShamMap:
    """A target-blind map replacing each control profile by a different one.

    ``donor_indices`` index the same local partition as ``source_indices``.
    The map may reuse a donor profile when profile-class counts are imbalanced;
    this is explicit in ``audit`` rather than silently pretending it is a
    bijective row permutation.
    """

    source_indices: tuple[int, ...]
    donor_indices: tuple[int, ...]
    source_row_ids: tuple[str, ...]
    donor_row_ids: tuple[str, ...]
    source_profile_ids: tuple[str, ...]
    donor_profile_ids: tuple[str, ...]
    audit: Mapping[str, Any]

    def apply(self, pre: Any) -> Any:
        np = _require_numpy()
        values = np.asarray(pre)
        if len(values) != len(self.donor_indices):
            raise SchemaError("Control-profile sham map does not match input row count")
        return values[np.asarray(self.donor_indices, dtype=np.int64)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "control_profile_sham",
            "source_to_donor": [
                {
                    "source_index": source,
                    "donor_index": donor,
                    "source_row_id": source_id,
                    "donor_row_id": donor_id,
                    "source_context_profile_id": source_profile,
                    "donor_context_profile_id": donor_profile,
                }
                for source, donor, source_id, donor_id, source_profile, donor_profile in zip(
                    self.source_indices,
                    self.donor_indices,
                    self.source_row_ids,
                    self.donor_row_ids,
                    self.source_profile_ids,
                    self.donor_profile_ids,
                )
            ],
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class ChemicalShamMap:
    """A target-blind chemical-prefix sham map with a recorded fallback mode."""

    source_indices: tuple[int, ...]
    donor_indices: tuple[int, ...]
    source_row_ids: tuple[str, ...]
    donor_row_ids: tuple[str, ...]
    source_profile_ids: tuple[str, ...]
    donor_profile_ids: tuple[str, ...]
    source_chemical_ids: tuple[str, ...]
    donor_chemical_ids: tuple[str, ...]
    modes: tuple[str, ...]
    audit: Mapping[str, Any]

    def apply(self, chemical_prefix: Any) -> Any:
        np = _require_numpy()
        values = np.asarray(chemical_prefix)
        if len(values) != len(self.donor_indices):
            raise SchemaError("Chemical sham map does not match input row count")
        return values[np.asarray(self.donor_indices, dtype=np.int64)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "chemical_prefix_sham",
            "source_to_donor": [
                {
                    "source_index": source,
                    "donor_index": donor,
                    "source_row_id": source_id,
                    "donor_row_id": donor_id,
                    "source_context_profile_id": source_profile,
                    "donor_context_profile_id": donor_profile,
                    "source_chemical_id": source_chemical,
                    "donor_chemical_id": donor_chemical,
                    "mode": mode,
                }
                for source, donor, source_id, donor_id, source_profile, donor_profile, source_chemical, donor_chemical, mode in zip(
                    self.source_indices,
                    self.donor_indices,
                    self.source_row_ids,
                    self.donor_row_ids,
                    self.source_profile_ids,
                    self.donor_profile_ids,
                    self.source_chemical_ids,
                    self.donor_chemical_ids,
                    self.modes,
                )
            ],
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class MatchedDoseShamMap:
    """Target-blind dose reassignment among rows with the same chemical.

    Dose is only counterfactually identifiable for chemical identities observed
    at at least two distinct registered dose values.  The map therefore covers
    the eligible subset rather than fabricating a dose perturbation for
    single-dose compounds.  Every donor shares the source chemical identity
    and has a different dose value.
    """

    source_indices: tuple[int, ...]
    donor_indices: tuple[int, ...]
    source_row_ids: tuple[str, ...]
    donor_row_ids: tuple[str, ...]
    chemical_ids: tuple[str, ...]
    source_dose_ids: tuple[str, ...]
    donor_dose_ids: tuple[str, ...]
    audit: Mapping[str, Any]

    def apply(self, dose: Any) -> Any:
        np = _require_numpy()
        values = np.asarray(dose)
        if values.ndim not in (1, 2):
            raise SchemaError("Matched-dose sham requires a one- or two-dimensional dose array")
        if not len(values) or max(self.source_indices, default=-1) >= len(values):
            raise SchemaError("Matched-dose sham does not match the supplied dose array")
        return values[np.asarray(self.donor_indices, dtype=np.int64)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "matched_dose_sham",
            "source_to_donor": [
                {
                    "source_index": source,
                    "donor_index": donor,
                    "source_row_id": source_id,
                    "donor_row_id": donor_id,
                    "chemical_id": chemical_id,
                    "source_dose_id": source_dose,
                    "donor_dose_id": donor_dose,
                }
                for source, donor, source_id, donor_id, chemical_id, source_dose, donor_dose in zip(
                    self.source_indices,
                    self.donor_indices,
                    self.source_row_ids,
                    self.donor_row_ids,
                    self.chemical_ids,
                    self.source_dose_ids,
                    self.donor_dose_ids,
                )
            ],
            "audit": dict(self.audit),
        }


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Perturbation loop requires cache manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Perturbation-loop cache manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SchemaError("Perturbation-loop cache manifest must be an object")
    return payload


def _registered_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    shapes = manifest.get("cache_shapes")
    cp = manifest.get("cell_painting")
    l1000 = manifest.get("l1000")
    layout = manifest.get("condition_layout")
    if not isinstance(shapes, Mapping) or not isinstance(cp, Mapping) or not isinstance(l1000, Mapping):
        raise SchemaError("Cache manifest is missing response block metadata")
    if not isinstance(layout, Mapping):
        raise SchemaError("Cache manifest is missing registered chemical-prefix/dose layout")
    prefix_dim = int(layout.get("prefix_dim", 0))
    cp_dim = int(cp.get("response_feature_count", 0))
    l1000_dim = int(l1000.get("response_feature_count", 0))
    target_shape = shapes.get("target")
    condition_shape = shapes.get("condition")
    if (
        prefix_dim <= 0
        or cp_dim <= 0
        or l1000_dim <= 0
        or not isinstance(target_shape, list)
        or len(target_shape) != 2
        or not isinstance(condition_shape, list)
        or len(condition_shape) != 2
    ):
        raise SchemaError("Cache manifest has incomplete perturbation-loop dimensions")
    if cp_dim + l1000_dim != int(target_shape[1]):
        raise SchemaError("Registered response blocks do not reconstruct target width")
    if prefix_dim + 1 != int(condition_shape[1]):
        raise SchemaError("Registered condition layout is not chemical prefix plus one dose scalar")
    return {
        "cache_schema_version": manifest.get("schema_version"),
        "condition_layout": dict(layout),
        "chemical_prefix_dim": prefix_dim,
        "dose_semantics": layout.get("scalar_semantics"),
        "cp_target_dim": cp_dim,
        "l1000_target_dim": l1000_dim,
        "target_dim": int(target_shape[1]),
        "response_semantics": manifest.get("response_semantics"),
        "context": dict(manifest.get("context", {})) if isinstance(manifest.get("context"), Mapping) else {},
        "split": dict(manifest.get("split", {})) if isinstance(manifest.get("split"), Mapping) else {},
    }


def _provenance_id(record: Mapping[str, Any], *, kind: str) -> str:
    if kind == "profile":
        payload = {
            "condition_key": record.get("condition_key", ""),
            "cp_source_row_ids": record.get("cp_source_row_ids", ""),
            "l1000_source_row_ids": record.get("l1000_source_row_ids", ""),
        }
    elif kind == "context":
        payload = record.get("cp_control_context", {})
    else:  # pragma: no cover - internal exhaustive guard
        raise ValueError(kind)
    return f"{kind}_source:{_sha256_text(canonical_json(payload))}"


def _read_provenance(path: Path, labels: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Perturbation loop requires registered provenance: {path}") from exc
    records: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"Invalid JSON in provenance at line {line_number}") from exc
        if not isinstance(record, Mapping) or not isinstance(record.get("condition_key"), str):
            raise SchemaError(f"Provenance line {line_number} lacks condition_key")
        label = str(record["condition_key"])
        if label in records:
            raise SchemaError(f"Provenance has duplicate condition_key: {label}")
        records[label] = record
    missing = [label for label in labels if label not in records]
    if missing:
        raise SchemaError(f"Provenance is missing {len(missing)} cached labels; first={missing[0]!r}")
    return tuple(records[label] for label in labels)


def _load_cache(
    *,
    cache_path: str | Path,
    manifest_path: str | Path,
    provenance_path: str | Path,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    np = _require_numpy()
    cache = Path(cache_path)
    metadata = _registered_metadata(_read_manifest(Path(manifest_path)))
    try:
        with np.load(cache, allow_pickle=False) as data:
            required = {
                "pre",
                "condition",
                "target",
                "cp_response",
                "l1000_response",
                "fold",
                "labels",
            }
            missing = required.difference(data.files)
            if missing:
                raise SchemaError(f"Perturbation-loop cache missing arrays: {sorted(missing)}")
            arrays = {
                "pre": data["pre"].astype(np.float32, copy=False),
                "condition": data["condition"].astype(np.float32, copy=False),
                "target": data["target"].astype(np.float32, copy=False),
                "cp_response": data["cp_response"].astype(np.float32, copy=False),
                "l1000_response": data["l1000_response"].astype(np.float32, copy=False),
                "fold": data["fold"].astype(np.int8, copy=False),
                "labels": _decode(data["labels"]),
            }
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Perturbation loop requires BBBC036 cache: {cache}") from exc
    row_count = int(arrays["pre"].shape[0])
    for key in ("condition", "target", "cp_response", "l1000_response", "fold"):
        if int(arrays[key].shape[0]) != row_count:
            raise SchemaError(f"Cache array {key} has inconsistent row count")
    if len(arrays["labels"]) != row_count or len(set(arrays["labels"])) != row_count:
        raise SchemaError("Perturbation-loop cache labels must be unique and row-aligned")
    if int(arrays["condition"].shape[1]) != int(metadata["chemical_prefix_dim"]) + 1:
        raise SchemaError("Cache condition width differs from registered chemical-prefix/dose layout")
    if int(arrays["target"].shape[1]) != int(metadata["target_dim"]):
        raise SchemaError("Cache target width differs from registered response blocks")
    if int(arrays["cp_response"].shape[1]) != int(metadata["cp_target_dim"]):
        raise SchemaError("Cache Cell Painting block width differs from manifest")
    if int(arrays["l1000_response"].shape[1]) != int(metadata["l1000_target_dim"]):
        raise SchemaError("Cache L1000 block width differs from manifest")
    provenance = _read_provenance(Path(provenance_path), arrays["labels"])
    return arrays, provenance, metadata


def _partition(
    arrays: Mapping[str, Any],
    provenance: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    indices: Any,
    *,
    name: str,
) -> PerturbationPartition:
    np = _require_numpy()
    index = np.asarray(indices, dtype=np.int64)
    if not len(index):
        raise SchemaError(f"Requested empty perturbation-loop partition: {name}")
    labels = tuple(arrays["labels"][int(position)] for position in index)
    pre = arrays["pre"][index]
    context_profile_ids = tuple(_vector_hash(row, namespace="pre") for row in pre)
    row_provenance: list[PerturbationRowProvenance] = []
    profile_ids: list[str] = []
    context_source_ids: list[str] = []
    cache_folds = arrays["fold"][index].astype(np.int8, copy=False)
    folds = (cache_folds.astype(np.int16) + 1).astype(np.int8)
    for local_index, cache_index in enumerate(index.tolist()):
        record = provenance[int(cache_index)]
        profile_id = _provenance_id(record, kind="profile")
        context_source_id = _provenance_id(record, kind="context")
        profile_ids.append(profile_id)
        context_source_ids.append(context_source_id)
        row_provenance.append(
            PerturbationRowProvenance(
                cache_index=int(cache_index),
                cache_fold=int(cache_folds[local_index]),
                fold=int(folds[local_index]),
                label=labels[local_index],
                context_profile_id=context_profile_ids[local_index],
                profile_id=profile_id,
                context_source_id=context_source_id,
                cp_plate_ids=str(record.get("cp_plate_ids", "")),
                l1000_plate_ids=str(record.get("l1000_plate_ids", "")),
                cp_source_row_ids=str(record.get("cp_source_row_ids", "")),
                l1000_source_row_ids=str(record.get("l1000_source_row_ids", "")),
            )
        )
    prefix_dim = int(metadata["chemical_prefix_dim"])
    condition = arrays["condition"][index]
    return PerturbationPartition(
        name=name,
        pre=pre,
        chemical_prefix=condition[:, :prefix_dim],
        dose=condition[:, prefix_dim].astype(np.float32, copy=False),
        target=arrays["target"][index],
        cp_target=arrays["cp_response"][index],
        l1000_target=arrays["l1000_response"][index],
        labels=labels,
        cache_indices=index,
        cache_folds=cache_folds,
        folds=folds,
        context_profile_ids=context_profile_ids,
        profile_ids=tuple(profile_ids),
        context_source_ids=tuple(context_source_ids),
        provenance=tuple(row_provenance),
    )


def _fit_indices(arrays: Mapping[str, Any]) -> Any:
    np = _require_numpy()
    return np.flatnonzero(np.isin(arrays["fold"], [0, 1]))


def load_perturbation_loop_selection_arrays(
    *,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
) -> PerturbationSelectionArrays:
    """Load fit folds 1--2 and feedback fold 3, excluding endpoint and fold 5."""

    arrays, provenance, metadata = _load_cache(
        cache_path=cache_path or DEFAULT_CACHE,
        manifest_path=manifest_path or DEFAULT_MANIFEST,
        provenance_path=provenance_path or DEFAULT_PROVENANCE,
    )
    np = _require_numpy()
    return PerturbationSelectionArrays(
        task_id=TASK_ID,
        fit=_partition(arrays, provenance, metadata, _fit_indices(arrays), name="fit_folds_1_2"),
        feedback=_partition(
            arrays,
            provenance,
            metadata,
            np.flatnonzero(arrays["fold"] == 2),
            name="feedback_fold_3",
        ),
        metadata=metadata,
    )


def load_perturbation_loop_endpoint_arrays(
    *,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
) -> PerturbationEndpointArrays:
    """Load fit folds 1--2 and endpoint fold 4, without exposing feedback."""

    arrays, provenance, metadata = _load_cache(
        cache_path=cache_path or DEFAULT_CACHE,
        manifest_path=manifest_path or DEFAULT_MANIFEST,
        provenance_path=provenance_path or DEFAULT_PROVENANCE,
    )
    np = _require_numpy()
    return PerturbationEndpointArrays(
        task_id=TASK_ID,
        fit=_partition(arrays, provenance, metadata, _fit_indices(arrays), name="fit_folds_1_2"),
        endpoint=_partition(
            arrays,
            provenance,
            metadata,
            np.flatnonzero(arrays["fold"] == 3),
            name="endpoint_fold_4",
        ),
        metadata=metadata,
    )


def _normalized_row_ids(row_ids: Sequence[object] | None, count: int) -> tuple[str, ...]:
    if row_ids is None:
        return tuple(str(index) for index in range(count))
    values = tuple(str(value) for value in row_ids)
    if len(values) != count:
        raise SchemaError("Row identifiers do not match sham-map row count")
    if len(set(values)) != len(values):
        raise SchemaError("Sham-map row identifiers must be unique")
    return values


def _map_audit(
    *,
    kind: str,
    seed: int,
    source_indices: Sequence[int],
    donor_indices: Sequence[int],
    source_ids: Sequence[str],
    donor_ids: Sequence[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "seed": int(seed),
        "source_indices": list(source_indices),
        "donor_indices": list(donor_indices),
        "source_ids": list(source_ids),
        "donor_ids": list(donor_ids),
        **dict(extra),
    }
    return {
        "kind": kind,
        "seed": int(seed),
        "target_blind": True,
        "map_hash": _sha256_text(canonical_json(payload)),
        **dict(extra),
    }


def build_control_profile_sham_map(
    pre: Any,
    *,
    row_ids: Sequence[object] | None = None,
    seed: int = 0,
) -> ControlProfileShamMap:
    """Map every row to a different observed control-profile vector.

    This operation consumes ``pre`` and optional stable row IDs only.  It does
    not receive target, feedback metric, or model outcome information.  Donor
    reuse is allowed when necessary to maintain the stronger invariant that
    *every* source profile hash differs from its donor profile hash.
    """

    np = _require_numpy()
    values = np.asarray(pre)
    if values.ndim != 2 or not len(values):
        raise SchemaError("Control-profile sham requires a non-empty two-dimensional pre array")
    source_indices = tuple(range(len(values)))
    source_row_ids = _normalized_row_ids(row_ids, len(values))
    profile_ids = tuple(_vector_hash(row, namespace="pre") for row in values)
    by_profile: dict[str, list[int]] = {}
    for index, profile_id in enumerate(profile_ids):
        by_profile.setdefault(profile_id, []).append(index)
    if len(by_profile) < 2:
        raise SchemaError("Control-profile sham is impossible: partition has only one pre-vector hash")
    profile_order = sorted(by_profile, key=lambda key: _stable_key("control_profile", seed, key))
    donor_profile = {
        profile_id: profile_order[(position + 1) % len(profile_order)]
        for position, profile_id in enumerate(profile_order)
    }
    donor_indices: list[int] = []
    for source_index, profile_id in enumerate(profile_ids):
        candidates = by_profile[donor_profile[profile_id]]
        selected = min(
            candidates,
            key=lambda donor: _stable_key("control_donor", seed, source_row_ids[source_index], source_row_ids[donor]),
        )
        donor_indices.append(selected)
    donor_profiles = tuple(profile_ids[index] for index in donor_indices)
    if any(source == donor for source, donor in zip(profile_ids, donor_profiles)):
        raise AssertionError("Internal error: control-profile sham retained a source profile")
    source_counts = {profile: len(indices) for profile, indices in sorted(by_profile.items())}
    audit = _map_audit(
        kind="control_profile_sham",
        seed=seed,
        source_indices=source_indices,
        donor_indices=donor_indices,
        source_ids=source_row_ids,
        donor_ids=tuple(source_row_ids[index] for index in donor_indices),
        extra={
            "mapping_policy": "different_profile_class_cycle_with_reuse_if_needed",
            "row_count": len(values),
            "profile_class_count": len(by_profile),
            "profile_class_counts": source_counts,
            "unique_donor_rows": len(set(donor_indices)),
            "donor_reuse_count": len(donor_indices) - len(set(donor_indices)),
            "all_source_profile_ids_differ": True,
        },
    )
    return ControlProfileShamMap(
        source_indices=source_indices,
        donor_indices=tuple(donor_indices),
        source_row_ids=source_row_ids,
        donor_row_ids=tuple(source_row_ids[index] for index in donor_indices),
        source_profile_ids=profile_ids,
        donor_profile_ids=donor_profiles,
        audit=audit,
    )


def _cohort_class_permutation(
    indices: Sequence[int],
    chemical_ids: Sequence[str],
    row_ids: Sequence[str],
    *,
    seed: int,
) -> dict[int, int] | None:
    """Return a within-cohort class derangement when a bijection is feasible."""

    classes: dict[str, list[int]] = {}
    for index in indices:
        classes.setdefault(chemical_ids[index], []).append(index)
    count = len(indices)
    largest = max((len(members) for members in classes.values()), default=0)
    if len(classes) < 2 or largest > count - largest:
        return None
    # Group the largest class first; rotating by its size is a deterministic
    # class derangement when no class occupies more than half the cohort.
    class_order = sorted(
        classes,
        key=lambda chemical: (-len(classes[chemical]), _stable_key("chemical_class", seed, chemical)),
    )
    source_order: list[int] = []
    for chemical in class_order:
        source_order.extend(
            sorted(classes[chemical], key=lambda index: _stable_key("chemical_source", seed, row_ids[index]))
        )
    donor_order = source_order[largest:] + source_order[:largest]
    mapping = dict(zip(source_order, donor_order))
    if any(chemical_ids[source] == chemical_ids[donor] for source, donor in mapping.items()):
        raise AssertionError("Internal error: cohort chemical derangement was not class-disjoint")
    return mapping


def build_chemical_prefix_sham_map(
    pre: Any,
    chemical_prefix: Any,
    *,
    row_ids: Sequence[object] | None = None,
    seed: int = 0,
) -> ChemicalShamMap:
    """Shuffle chemical fingerprint classes within a control-profile cohort.

    A cohort receives an exact within-cohort class permutation when possible.
    A singleton or class-imbalanced cohort instead uses a deterministic donor
    from the complete partition with a different fingerprint class.  Every
    such fallback row and reason is retained in the returned audit.  If the
    whole partition has one chemical class, this semantic sham is impossible
    and the function fails rather than silently retaining the true chemical.
    """

    np = _require_numpy()
    pre_values = np.asarray(pre)
    chemical_values = np.asarray(chemical_prefix)
    if pre_values.ndim != 2 or chemical_values.ndim != 2 or len(pre_values) != len(chemical_values) or not len(pre_values):
        raise SchemaError("Chemical-prefix sham requires aligned non-empty two-dimensional pre and chemical arrays")
    source_indices = tuple(range(len(pre_values)))
    source_row_ids = _normalized_row_ids(row_ids, len(pre_values))
    profile_ids = tuple(_vector_hash(row, namespace="pre") for row in pre_values)
    chemical_ids = tuple(_vector_hash(row, namespace="chemical") for row in chemical_values)
    all_chemical_classes = sorted(set(chemical_ids))
    if len(all_chemical_classes) < 2:
        raise SchemaError("Chemical-prefix sham is impossible: partition has only one fingerprint class")
    by_chemical: dict[str, list[int]] = {}
    for index, chemical_id in enumerate(chemical_ids):
        by_chemical.setdefault(chemical_id, []).append(index)
    # A fallback must remain target-blind and class-disjoint, but it does not
    # need to enumerate every row outside a source class.  The previous
    # implementation did exactly that for every singleton context cohort,
    # which made a registered semantic sham quadratic in the endpoint size.
    # This fixed class cycle chooses a different chemical class first, then a
    # deterministic member of that class.  It preserves the intended semantic
    # contract while making the fallback linear in the number of rows.
    chemical_class_order = sorted(by_chemical, key=lambda key: _stable_key("chemical_partition_class", seed, key))
    fallback_donor_class = {
        chemical_id: chemical_class_order[(position + 1) % len(chemical_class_order)]
        for position, chemical_id in enumerate(chemical_class_order)
    }
    by_profile: dict[str, list[int]] = {}
    for index, profile_id in enumerate(profile_ids):
        by_profile.setdefault(profile_id, []).append(index)
    mapping: dict[int, int] = {}
    modes: list[str] = ["" for _ in source_indices]
    fallback_reasons: dict[str, str] = {}
    for profile_id in sorted(by_profile, key=lambda key: _stable_key("pre_cohort", seed, key)):
        indices = by_profile[profile_id]
        local = _cohort_class_permutation(indices, chemical_ids, source_row_ids, seed=seed)
        if local is not None:
            mapping.update(local)
            for index in indices:
                modes[index] = "within_pre_cohort_class_derangement"
            continue
        classes = {chemical_ids[index] for index in indices}
        largest = max(sum(chemical_ids[index] == chemical for index in indices) for chemical in classes)
        reason = "singleton_cohort" if len(indices) == 1 or len(classes) == 1 else "class_imbalanced_cohort"
        if len(classes) >= 2 and largest <= len(indices) - largest:  # pragma: no cover - defensive consistency check
            reason = "unclassified_infeasible_cohort"
        fallback_reasons[profile_id] = reason
        for source in indices:
            source_class = chemical_ids[source]
            donor_class = fallback_donor_class[source_class]
            candidates = by_chemical[donor_class]
            # The target class is different by construction.  A source-keyed
            # modular index supplies deterministic donor diversity without a
            # per-source scan of the full endpoint partition.
            choice_key = int(
                _stable_key("chemical_partition_fallback", seed, source_row_ids[source], donor_class),
                16,
            )
            donor = candidates[choice_key % len(candidates)]
            mapping[source] = donor
            modes[source] = "partition_wide_fallback"
    if set(mapping) != set(source_indices):
        raise AssertionError("Internal error: chemical sham has incomplete source coverage")
    donor_indices = tuple(mapping[index] for index in source_indices)
    donor_chemicals = tuple(chemical_ids[index] for index in donor_indices)
    if any(source == donor for source, donor in zip(chemical_ids, donor_chemicals)):
        raise AssertionError("Internal error: chemical sham retained a source fingerprint class")
    fallback_rows = [index for index, mode in enumerate(modes) if mode == "partition_wide_fallback"]
    cohort_summary = []
    for profile_id in sorted(by_profile):
        indices = by_profile[profile_id]
        cohort_summary.append(
            {
                "context_profile_id": profile_id,
                "row_count": len(indices),
                "chemical_class_count": len({chemical_ids[index] for index in indices}),
                "mode": "partition_wide_fallback" if profile_id in fallback_reasons else "within_pre_cohort_class_derangement",
                "fallback_reason": fallback_reasons.get(profile_id),
            }
        )
    audit = _map_audit(
        kind="chemical_prefix_sham",
        seed=seed,
        source_indices=source_indices,
        donor_indices=donor_indices,
        source_ids=source_row_ids,
        donor_ids=tuple(source_row_ids[index] for index in donor_indices),
        extra={
            "mapping_policy": "within_pre_cohort_class_derangement_then_partition_wide_cyclic_different_class_fallback_v2",
            "row_count": len(pre_values),
            "context_profile_cohort_count": len(by_profile),
            "chemical_class_count": len(all_chemical_classes),
            "fallback_row_count": len(fallback_rows),
            "fallback_rows": [source_row_ids[index] for index in fallback_rows],
            "fallback_cohorts": cohort_summary,
            "unique_donor_rows": len(set(donor_indices)),
            "donor_reuse_count": len(donor_indices) - len(set(donor_indices)),
            "all_source_chemical_ids_differ": True,
        },
    )
    return ChemicalShamMap(
        source_indices=source_indices,
        donor_indices=donor_indices,
        source_row_ids=source_row_ids,
        donor_row_ids=tuple(source_row_ids[index] for index in donor_indices),
        source_profile_ids=profile_ids,
        donor_profile_ids=tuple(profile_ids[index] for index in donor_indices),
        source_chemical_ids=chemical_ids,
        donor_chemical_ids=donor_chemicals,
        modes=tuple(modes),
        audit=audit,
    )


def build_matched_dose_sham_map(
    chemical_prefix: Any,
    dose: Any,
    *,
    row_ids: Sequence[object] | None = None,
    seed: int = 0,
    minimum_absolute_dose_delta: float = 0.0,
) -> MatchedDoseShamMap:
    """Derange registered doses *within* each multi-dose chemical identity.

    This is deliberately narrower than a global dose permutation.  It keeps
    chemical identity fixed and only includes compounds for which the endpoint
    itself contains more than one observed dose.  Thus it tests dose reliance
    without assigning a chemically unrelated dose to a compound.  It receives
    no response target, metric, model, or Fold-5 data.
    """

    np = _require_numpy()
    chemical_values = np.asarray(chemical_prefix)
    dose_values = np.asarray(dose)
    if chemical_values.ndim != 2 or dose_values.ndim not in (1, 2):
        raise SchemaError("Matched-dose sham requires a two-dimensional chemical array and scalar-dose array")
    if dose_values.ndim == 2:
        if dose_values.shape[1] != 1:
            raise SchemaError("Matched-dose sham requires exactly one registered dose scalar")
        dose_values = dose_values[:, 0]
    if len(chemical_values) != len(dose_values) or not len(chemical_values):
        raise SchemaError("Matched-dose sham requires aligned non-empty chemical and dose arrays")
    minimum_delta = float(minimum_absolute_dose_delta)
    if not np.isfinite(minimum_delta) or minimum_delta < 0:
        raise SchemaError("Matched-dose minimum delta must be finite and non-negative")
    source_row_ids = _normalized_row_ids(row_ids, len(chemical_values))
    chemical_ids = tuple(_vector_hash(row, namespace="chemical") for row in chemical_values)
    dose_ids = tuple(_vector_hash(np.asarray([value], dtype=np.float32), namespace="dose") for value in dose_values)
    by_chemical: dict[str, list[int]] = {}
    for index, chemical_id in enumerate(chemical_ids):
        by_chemical.setdefault(chemical_id, []).append(index)
    eligible_chemicals = {
        chemical_id: indices
        for chemical_id, indices in by_chemical.items()
        if len({dose_ids[index] for index in indices}) >= 2
    }
    if not eligible_chemicals:
        raise SchemaError("Matched-dose sham is not identifiable: no chemical has multiple observed dose values")

    mapping: dict[int, int] = {}
    for chemical_id in sorted(eligible_chemicals, key=lambda value: _stable_key("dose_chemical", seed, value)):
        indices = eligible_chemicals[chemical_id]
        if minimum_delta == 0.0:
            # Preserve the registered v1 mapping exactly for frozen historical
            # audits.  New contracts can request a scientifically meaningful
            # minimum intervention size below.
            by_dose: dict[str, list[int]] = {}
            for index in indices:
                by_dose.setdefault(dose_ids[index], []).append(index)
            dose_order = sorted(by_dose, key=lambda value: _stable_key("dose_class", seed, chemical_id, value))
            donor_dose = {
                dose_id: dose_order[(position + 1) % len(dose_order)]
                for position, dose_id in enumerate(dose_order)
            }
            for source in indices:
                target_dose = donor_dose[dose_ids[source]]
                candidates = by_dose[target_dose]
                choice_key = int(_stable_key("dose_donor", seed, source_row_ids[source], target_dose), 16)
                mapping[source] = candidates[choice_key % len(candidates)]
            continue

        for source in indices:
            candidates = [
                donor for donor in indices
                if donor != source
                and abs(float(dose_values[source]) - float(dose_values[donor])) >= minimum_delta
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda donor: _stable_key(
                "dose_donor_minimum_delta", seed, source_row_ids[source], source_row_ids[donor]
            ))
            mapping[source] = candidates[0]

    if not mapping:
        raise SchemaError(
            "Matched-dose sham is not identifiable at the registered minimum dose intervention"
        )

    source_indices = tuple(sorted(mapping))
    donor_indices = tuple(mapping[index] for index in source_indices)
    source_chemical_ids = tuple(chemical_ids[index] for index in source_indices)
    donor_chemical_ids = tuple(chemical_ids[index] for index in donor_indices)
    source_dose_ids = tuple(dose_ids[index] for index in source_indices)
    donor_dose_ids = tuple(dose_ids[index] for index in donor_indices)
    if any(source != donor for source, donor in zip(source_chemical_ids, donor_chemical_ids)):
        raise AssertionError("Internal error: matched-dose sham changed chemical identity")
    if any(source == donor for source, donor in zip(source_dose_ids, donor_dose_ids)):
        raise AssertionError("Internal error: matched-dose sham retained a source dose")

    eligible_rows = len(source_indices)
    active_chemical_count = len(set(source_chemical_ids))
    audit = _map_audit(
        kind="matched_dose_sham",
        seed=seed,
        source_indices=source_indices,
        donor_indices=donor_indices,
        source_ids=tuple(source_row_ids[index] for index in source_indices),
        donor_ids=tuple(source_row_ids[index] for index in donor_indices),
        extra={
            "mapping_policy": (
                "within_chemical_cyclic_different_dose_class_v1"
                if minimum_delta == 0.0
                else "within_chemical_target_blind_minimum_absolute_dose_delta_v2"
            ),
            "minimum_absolute_dose_delta": minimum_delta,
            "partition_row_count": len(chemical_values),
            "eligible_row_count": eligible_rows,
            "eligible_row_fraction": eligible_rows / len(chemical_values),
            "eligible_chemical_count": active_chemical_count,
            "ineligible_chemical_count": len(by_chemical) - active_chemical_count,
            "unique_donor_rows": len(set(donor_indices)),
            "donor_reuse_count": len(donor_indices) - len(set(donor_indices)),
            "all_source_chemical_ids_match": True,
            "all_source_dose_ids_differ": True,
            "all_absolute_dose_deltas_meet_minimum": all(
                abs(float(dose_values[source]) - float(dose_values[donor])) >= minimum_delta
                for source, donor in zip(source_indices, donor_indices)
            ),
        },
    )
    return MatchedDoseShamMap(
        source_indices=source_indices,
        donor_indices=donor_indices,
        source_row_ids=tuple(source_row_ids[index] for index in source_indices),
        donor_row_ids=tuple(source_row_ids[index] for index in donor_indices),
        chemical_ids=source_chemical_ids,
        source_dose_ids=source_dose_ids,
        donor_dose_ids=donor_dose_ids,
        audit=audit,
    )
