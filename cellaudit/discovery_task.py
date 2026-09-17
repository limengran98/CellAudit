"""Raw BBBC036 task boundary for open model discovery.

The discovery phase is allowed to inspect only fit folds 1--2 and discovery
fold 3.  This module verifies that the compact NPZ view is still derived from
the registered CPG0003 Cell Painting and L1000 ``.csv.gz`` sources before it
returns any arrays.  Fold-4 audit and fold-5 endpoint targets have no public
loader here: their roles are recorded in the immutable task contract but are
not members of :class:`BBBC036DiscoveryArrays`.

This is an API/provenance boundary, not a claim that a user with filesystem
access cannot open a local NPZ archive.  It prevents accidental leakage
through the discovery code path and makes a stale raw-derived cache fail
before an LLM or model builder sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import file_fingerprint, normalize_condition_layout
from .schemas import SchemaError, TaskContract, canonical_json, digest
from .tasks import get_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "cpg036_cp_plate_control_context"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.npz"
DEFAULT_MANIFEST = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.manifest.json"
DEFAULT_PROVENANCE = PROJECT_ROOT / "cache" / "cpg036_cp_plate_control_context.provenance.jsonl"

FIT_CACHE_FOLDS = (0, 1)
DISCOVERY_CACHE_FOLD = 2
AUDIT_CACHE_FOLD = 3
ENDPOINT_CACHE_FOLD = 4

_REQUIRED_CACHE_ARRAYS = {
    "pre",
    "condition",
    "target",
    "cp_response",
    "l1000_response",
    "fold",
    "labels",
}


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("Raw BBBC036 discovery boundary requires numpy") from exc
    return np


def _decode(values: Any) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


def _is_csv_gz(path: Path) -> bool:
    return path.name.endswith(".csv.gz")


def _resolve_registered_path(value: str | Path, *, manifest_path: Path) -> Path:
    """Resolve a cache-manifest relative provenance path without guessing.

    Historical manifests record paths relative to the project root (for
    example ``cache/foo.provenance.jsonl``), whereas synthetic tests and
    external callers often register an absolute path.  The candidates below
    are deterministic and an absent path remains absent so verification can
    fail loudly.
    """

    source = Path(value)
    if source.is_absolute():
        return source
    candidates = (PROJECT_ROOT / source, manifest_path.parent / source, manifest_path.parent / source.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / source


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw BBBC036 discovery boundary requires {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Raw BBBC036 {description} is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SchemaError(f"Raw BBBC036 {description} must be a JSON object")
    return payload


def _default_raw_sources() -> tuple[Path, Path]:
    task = get_task(TASK_ID)
    sources = task.get("sources")
    if not isinstance(sources, Mapping):
        raise SchemaError(f"Registered task {TASK_ID} lacks raw source mapping")
    try:
        cp = Path(str(sources["cell_painting"]))
        l1000 = Path(str(sources["l1000"]))
    except KeyError as exc:  # pragma: no cover - configuration guard
        raise SchemaError(f"Registered task {TASK_ID} lacks a required raw source") from exc
    return cp, l1000


def _assert_expected_source(path: Path, *, modality: str) -> None:
    if not _is_csv_gz(path):
        raise SchemaError(f"{modality} raw source must be the registered CPG0003 .csv.gz file, got {path}")
    if not path.exists():
        raise FileNotFoundError(f"Registered {modality} raw source does not exist: {path}")


def _path_equal(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:  # pragma: no cover - malformed paths still compare below
        return str(left) == str(right)


def _require_manifest_source(
    manifest: Mapping[str, Any],
    *,
    section: str,
    expected_path: Path,
) -> str:
    block = manifest.get(section)
    if not isinstance(block, Mapping):
        raise SchemaError(f"Raw-derived cache manifest lacks {section} source block")
    source_path = block.get("source_path")
    source_fingerprint = block.get("source_fingerprint")
    if not isinstance(source_path, str) or not _path_equal(source_path, expected_path):
        raise SchemaError(f"Cache manifest {section} source path does not match registered raw source")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise SchemaError(f"Cache manifest {section} source fingerprint is missing")
    observed = file_fingerprint(expected_path)
    if source_fingerprint != observed:
        raise SchemaError(f"Cache manifest {section} source fingerprint does not match current raw CSV.GZ")
    return observed


def _read_provenance_labels(path: Path) -> tuple[str, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw BBBC036 discovery boundary requires retained-condition provenance: {path}") from exc
    labels: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"Retained-condition provenance is invalid JSON at line {line_number}") from exc
        if not isinstance(record, Mapping) or not isinstance(record.get("condition_key"), str):
            raise SchemaError(f"Retained-condition provenance line {line_number} lacks condition_key")
        if record.get("disposition", "retained") != "retained":
            raise SchemaError(f"Retained-condition provenance line {line_number} is not retained")
        labels.append(str(record["condition_key"]))
    if not labels or len(labels) != len(set(labels)):
        raise SchemaError("Retained-condition provenance must contain one unique retained condition per row")
    return tuple(labels)


def _required_manifest_dimensions(manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    shapes = manifest.get("cache_shapes")
    cp = manifest.get("cell_painting")
    l1000 = manifest.get("l1000")
    layout = manifest.get("condition_layout")
    if not isinstance(shapes, Mapping) or not isinstance(cp, Mapping) or not isinstance(l1000, Mapping):
        raise SchemaError("Raw-derived cache manifest lacks required shapes or response blocks")
    normalized_layout = normalize_condition_layout(layout)
    if normalized_layout is None:
        raise SchemaError("Raw-derived cache manifest lacks registered chemical-prefix/dose layout")
    cp_dim = int(cp.get("response_feature_count", 0))
    l1000_dim = int(l1000.get("response_feature_count", 0))
    prefix_dim = int(normalized_layout["prefix_dim"])
    target_shape = shapes.get("target")
    condition_shape = shapes.get("condition")
    if (
        cp_dim <= 0
        or l1000_dim <= 0
        or not isinstance(target_shape, list)
        or len(target_shape) != 2
        or not isinstance(condition_shape, list)
        or len(condition_shape) != 2
        or cp_dim + l1000_dim != int(target_shape[1])
        or prefix_dim + 1 != int(condition_shape[1])
    ):
        raise SchemaError("Raw-derived cache manifest has inconsistent dimensions")
    return prefix_dim, cp_dim, l1000_dim


@dataclass(frozen=True)
class BBBC036RawVerification:
    """JSON-safe provenance of one verified raw-derived BBBC036 task view."""

    task_id: str
    cache_path: str
    manifest_path: str
    provenance_path: str
    cell_painting_source_path: str
    l1000_source_path: str
    cell_painting_source_fingerprint: str
    l1000_source_fingerprint: str
    cache_fingerprint: str
    manifest_fingerprint: str
    provenance_fingerprint: str
    row_count: int
    fold_counts: Mapping[str, int]
    chemical_prefix_dim: int
    cp_target_dim: int
    l1000_target_dim: int
    data_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cache_path": self.cache_path,
            "manifest_path": self.manifest_path,
            "provenance_path": self.provenance_path,
            "raw_sources": {
                "cell_painting": {
                    "path": self.cell_painting_source_path,
                    "fingerprint": self.cell_painting_source_fingerprint,
                },
                "l1000": {
                    "path": self.l1000_source_path,
                    "fingerprint": self.l1000_source_fingerprint,
                },
            },
            "cache_fingerprint": self.cache_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "provenance_fingerprint": self.provenance_fingerprint,
            "row_count": int(self.row_count),
            "fold_counts": {str(key): int(value) for key, value in self.fold_counts.items()},
            "chemical_prefix_dim": int(self.chemical_prefix_dim),
            "cp_target_dim": int(self.cp_target_dim),
            "l1000_target_dim": int(self.l1000_target_dim),
            "data_fingerprint": self.data_fingerprint,
        }


@dataclass(frozen=True)
class BBBC036DiscoveryPartition:
    """One discovery-visible partition; all rows have target access by design."""

    name: str
    pre: Any
    condition: Any
    target: Any
    cp_target: Any
    l1000_target: Any
    labels: tuple[str, ...]
    cache_indices: Any
    cache_folds: Any
    folds: Any

    @property
    def count(self) -> int:
        return int(self.pre.shape[0])


@dataclass(frozen=True)
class BBBC036DiscoveryArrays:
    """Only the data allowed before the fold-4 audit and fold-5 endpoint."""

    task_id: str
    contract: TaskContract
    fit: BBBC036DiscoveryPartition
    discovery: BBBC036DiscoveryPartition
    metadata: Mapping[str, Any]


def verify_bbbc036_raw_discovery_inputs(
    *,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    cell_painting_source_path: str | Path | None = None,
    l1000_source_path: str | Path | None = None,
) -> BBBC036RawVerification:
    """Verify raw CSV.GZ fingerprints, cache manifest, and per-condition provenance.

    The cache is accepted only when both manifest source fingerprints equal a
    freshly computed fingerprint of the original registered raw files, and
    every cached condition label occurs exactly once in retained provenance.
    """

    np = _require_numpy()
    cache = Path(cache_path or DEFAULT_CACHE)
    manifest_file = Path(manifest_path or (cache.with_suffix(".manifest.json") if cache_path else DEFAULT_MANIFEST))
    manifest = _read_json(manifest_file, description="cache manifest")
    default_cp, default_l1000 = _default_raw_sources()
    cp_source = Path(cell_painting_source_path) if cell_painting_source_path is not None else default_cp
    l1000_source = Path(l1000_source_path) if l1000_source_path is not None else default_l1000
    _assert_expected_source(cp_source, modality="Cell Painting")
    _assert_expected_source(l1000_source, modality="L1000")
    if cache.suffix != ".npz":
        raise SchemaError(f"Raw BBBC036 discovery cache must be an NPZ adapter output, got {cache}")
    if manifest.get("adapter") != "cpg_replicate_profiles":
        raise SchemaError("Discovery task accepts only the registered raw CPG replicate-profile adapter")
    if manifest.get("direction") != "joint_from_condition_with_cp_control_context":
        raise SchemaError("Discovery task cache does not implement the registered BBBC036 control-context task")
    cp_fingerprint = _require_manifest_source(
        manifest,
        section="cell_painting",
        expected_path=cp_source,
    )
    l1000_fingerprint = _require_manifest_source(
        manifest,
        section="l1000",
        expected_path=l1000_source,
    )
    prefix_dim, cp_dim, l1000_dim = _required_manifest_dimensions(manifest)
    provenance_value = provenance_path
    if provenance_value is None:
        block = manifest.get("provenance")
        if not isinstance(block, Mapping) or not isinstance(block.get("retained_conditions"), str):
            raise SchemaError("Raw-derived cache manifest lacks retained-condition provenance registration")
        provenance_file = _resolve_registered_path(str(block["retained_conditions"]), manifest_path=manifest_file)
    else:
        provenance_file = Path(provenance_value)
    provenance_labels = _read_provenance_labels(provenance_file)
    try:
        with np.load(cache, allow_pickle=False) as archive:
            missing = _REQUIRED_CACHE_ARRAYS.difference(archive.files)
            if missing:
                raise SchemaError(f"Raw-derived cache misses arrays: {sorted(missing)}")
            # Fold/label access contains no response values.  Target arrays are
            # opened only for shape validation here; no values are returned by
            # this verifier or its metadata.
            folds = archive["fold"].astype(np.int8, copy=False)
            labels = _decode(archive["labels"])
            pre_shape = tuple(int(value) for value in archive["pre"].shape)
            condition_shape = tuple(int(value) for value in archive["condition"].shape)
            target_shape = tuple(int(value) for value in archive["target"].shape)
            cp_shape = tuple(int(value) for value in archive["cp_response"].shape)
            l1000_shape = tuple(int(value) for value in archive["l1000_response"].shape)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Raw BBBC036 discovery boundary requires cache: {cache}") from exc
    row_count = len(labels)
    if row_count == 0 or len(labels) != len(set(labels)):
        raise SchemaError("Raw-derived cache labels must be non-empty and unique")
    if len(folds) != row_count or pre_shape[0] != row_count or condition_shape[0] != row_count:
        raise SchemaError("Raw-derived cache input arrays have inconsistent row counts")
    if target_shape[0] != row_count or cp_shape[0] != row_count or l1000_shape[0] != row_count:
        raise SchemaError("Raw-derived cache response arrays have inconsistent row counts")
    if condition_shape[1] != prefix_dim + 1 or cp_shape[1] != cp_dim or l1000_shape[1] != l1000_dim:
        raise SchemaError("Raw-derived cache array widths differ from registered manifest dimensions")
    if target_shape[1] != cp_dim + l1000_dim:
        raise SchemaError("Raw-derived cache joint target width differs from registered response blocks")
    if tuple(labels) != provenance_labels:
        raise SchemaError("Cache labels and retained-condition provenance are not exactly row-aligned")
    expected_folds = set(range(5))
    observed_folds = {int(value) for value in folds.tolist()}
    if observed_folds != expected_folds:
        raise SchemaError("Raw BBBC036 discovery cache must contain each registered cache fold 0--4")
    fold_counts = {str(fold + 1): int((folds == fold).sum()) for fold in range(5)}
    components = {
        "task_id": TASK_ID,
        "cell_painting_source_path": str(cp_source.resolve()),
        "l1000_source_path": str(l1000_source.resolve()),
        "cell_painting_source_fingerprint": cp_fingerprint,
        "l1000_source_fingerprint": l1000_fingerprint,
        "cache_fingerprint": file_fingerprint(cache),
        "manifest_fingerprint": file_fingerprint(manifest_file),
        "provenance_fingerprint": file_fingerprint(provenance_file),
        "fold_counts": fold_counts,
        "dimensions": {"chemical_prefix": prefix_dim, "cp": cp_dim, "l1000": l1000_dim},
    }
    data_fingerprint = digest(components)
    return BBBC036RawVerification(
        task_id=TASK_ID,
        cache_path=str(cache.resolve()),
        manifest_path=str(manifest_file.resolve()),
        provenance_path=str(provenance_file.resolve()),
        cell_painting_source_path=str(cp_source.resolve()),
        l1000_source_path=str(l1000_source.resolve()),
        cell_painting_source_fingerprint=cp_fingerprint,
        l1000_source_fingerprint=l1000_fingerprint,
        cache_fingerprint=components["cache_fingerprint"],
        manifest_fingerprint=components["manifest_fingerprint"],
        provenance_fingerprint=components["provenance_fingerprint"],
        row_count=row_count,
        fold_counts=fold_counts,
        chemical_prefix_dim=prefix_dim,
        cp_target_dim=cp_dim,
        l1000_target_dim=l1000_dim,
        data_fingerprint=data_fingerprint,
    )


def _contract_from_verification(
    verification: BBBC036RawVerification,
    *,
    manifest_path: Path,
    optimization_budget: Mapping[str, Any] | None,
) -> TaskContract:
    manifest = _read_json(manifest_path, description="cache manifest")
    condition_layout = normalize_condition_layout(manifest.get("condition_layout"))
    if condition_layout is None:  # defensive; verifier already rejects this
        raise SchemaError("Cannot create discovery contract without registered condition layout")
    protocol = {
        "protocol_id": "bbbc036_raw_discovery_boundary_v1",
        "cache_fold_roles": {
            "fit": [1, 2],
            "discovery": 3,
            "audit": 4,
            "endpoint": 5,
        },
        "target_access_before_audit": [1, 2, 3],
        "withheld_target_folds": [4, 5],
        "source_verification": verification.to_dict(),
    }
    budget = {
        "partition_protocol": protocol,
        "optimization": dict(optimization_budget or {}),
    }
    return TaskContract(
        task_id=TASK_ID,
        task_family="paired_multi_assay_response",
        data_path=verification.cell_painting_source_path,
        target="joint_same_plate_control_rank_cell_painting_and_control_relative_l1000_profile",
        condition="observed_same_plate_cell_painting_control_profile_plus_smiles_morgan2048_and_dose",
        metric="global_pcc",
        test_fold=5,
        allowed_target_modes=("absolute",),
        condition_layout=condition_layout,
        protected_fields=(
            "data_path",
            "target",
            "condition",
            "metric",
            "test_fold",
            "allowed_target_modes",
            "condition_layout",
            "training_budget",
        ),
        training_budget=budget,
        data_fingerprint=verification.data_fingerprint,
    )


def build_bbbc036_discovery_contract(
    *,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    cell_painting_source_path: str | Path | None = None,
    l1000_source_path: str | Path | None = None,
    optimization_budget: Mapping[str, Any] | None = None,
) -> TaskContract:
    """Return a JSON-safe immutable contract after source/cache verification."""

    verification = verify_bbbc036_raw_discovery_inputs(
        cache_path=cache_path,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        cell_painting_source_path=cell_painting_source_path,
        l1000_source_path=l1000_source_path,
    )
    return _contract_from_verification(
        verification,
        manifest_path=Path(verification.manifest_path),
        optimization_budget=optimization_budget,
    )


def _partition(
    *,
    name: str,
    indices: Any,
    source_cache_indices: Any,
    pre: Any,
    condition: Any,
    target: Any,
    cp_target: Any,
    l1000_target: Any,
    labels: Sequence[str],
    cache_folds: Any,
) -> BBBC036DiscoveryPartition:
    np = _require_numpy()
    index = np.asarray(indices, dtype=np.int64)
    if not len(index):
        raise SchemaError(f"Raw BBBC036 discovery requested empty partition {name}")
    selected_folds = cache_folds[index].astype(np.int8, copy=False)
    return BBBC036DiscoveryPartition(
        name=name,
        pre=pre[index].astype(np.float32, copy=False),
        condition=condition[index].astype(np.float32, copy=False),
        target=target[index].astype(np.float32, copy=False),
        cp_target=cp_target[index].astype(np.float32, copy=False),
        l1000_target=l1000_target[index].astype(np.float32, copy=False),
        labels=tuple(labels[int(position)] for position in index),
        cache_indices=np.asarray(source_cache_indices, dtype=np.int64)[index],
        cache_folds=selected_folds,
        folds=(selected_folds.astype(np.int16) + 1).astype(np.int8),
    )


def load_bbbc036_discovery_arrays(
    *,
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    cell_painting_source_path: str | Path | None = None,
    l1000_source_path: str | Path | None = None,
    optimization_budget: Mapping[str, Any] | None = None,
) -> BBBC036DiscoveryArrays:
    """Load only fit (folds 1--2) and discovery (fold 3) targets.

    The verifier runs first.  The returned object intentionally has no
    ``audit`` or ``endpoint`` member, and metadata contains only registered
    paths, dimensions, counts, hashes, and role declarations--never response
    values from folds 4 or 5.
    """

    np = _require_numpy()
    verification = verify_bbbc036_raw_discovery_inputs(
        cache_path=cache_path,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        cell_painting_source_path=cell_painting_source_path,
        l1000_source_path=l1000_source_path,
    )
    contract = _contract_from_verification(
        verification,
        manifest_path=Path(verification.manifest_path),
        optimization_budget=optimization_budget,
    )
    cache = Path(verification.cache_path)
    # The archive format stores a whole target array in one compressed member;
    # NumPy cannot read a row slice directly.  We immediately materialize only
    # registered discovery-visible rows and never retain/archive any fold-4/5
    # target array in the returned object.
    with np.load(cache, allow_pickle=False) as archive:
        folds = archive["fold"].astype(np.int8, copy=False)
        labels = _decode(archive["labels"])
        fit_indices = np.flatnonzero(np.isin(folds, FIT_CACHE_FOLDS))
        discovery_indices = np.flatnonzero(folds == DISCOVERY_CACHE_FOLD)
        allowed = np.concatenate([fit_indices, discovery_indices])
        if set(int(value) for value in folds[allowed].tolist()).difference({*FIT_CACHE_FOLDS, DISCOVERY_CACHE_FOLD}):
            raise SchemaError("Internal leakage guard failed: discovery loader selected a withheld fold")
        # Slice every response array with ``allowed`` before any partition is
        # constructed.  Index maps below refer to this short allowed view.
        allowed_pre = archive["pre"][allowed]
        allowed_condition = archive["condition"][allowed]
        allowed_target = archive["target"][allowed]
        allowed_cp_target = archive["cp_response"][allowed]
        allowed_l1000_target = archive["l1000_response"][allowed]
        allowed_labels = tuple(labels[int(index)] for index in allowed)
        allowed_folds = folds[allowed]
    fit_local = np.flatnonzero(np.isin(allowed_folds, FIT_CACHE_FOLDS))
    discovery_local = np.flatnonzero(allowed_folds == DISCOVERY_CACHE_FOLD)
    # Both conditions are independent guards: no returned cache index or fold
    # can refer to audit/endpoint data even if a future refactor changes how
    # ``allowed`` is assembled.
    if any(int(value) not in {*FIT_CACHE_FOLDS, DISCOVERY_CACHE_FOLD} for value in allowed_folds.tolist()):
        raise SchemaError("Internal leakage guard failed after discovery slicing")
    fit = _partition(
        name="fit_folds_1_2",
        indices=fit_local,
        source_cache_indices=allowed,
        pre=allowed_pre,
        condition=allowed_condition,
        target=allowed_target,
        cp_target=allowed_cp_target,
        l1000_target=allowed_l1000_target,
        labels=allowed_labels,
        cache_folds=allowed_folds,
    )
    discovery = _partition(
        name="discovery_fold_3",
        indices=discovery_local,
        source_cache_indices=allowed,
        pre=allowed_pre,
        condition=allowed_condition,
        target=allowed_target,
        cp_target=allowed_cp_target,
        l1000_target=allowed_l1000_target,
        labels=allowed_labels,
        cache_folds=allowed_folds,
    )
    metadata = {
        "task_id": TASK_ID,
        "raw_verification": verification.to_dict(),
        "discovery_visible_folds": [1, 2, 3],
        "withheld_target_folds": [4, 5],
        "fit_row_count": fit.count,
        "discovery_row_count": discovery.count,
        "contract_hash": contract.contract_hash,
    }
    # JSON round-trip is a cheap final guard against accidental NumPy scalar or
    # array leakage into metadata that will later be serialized into a trace.
    canonical_json(metadata)
    return BBBC036DiscoveryArrays(
        task_id=TASK_ID,
        contract=contract,
        fit=fit,
        discovery=discovery,
        metadata=metadata,
    )
