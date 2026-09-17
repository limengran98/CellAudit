"""Task contracts and deterministic local data fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import CandidateArtifact, SchemaError, TaskContract


CONDITION_PREFIX_FINAL_SCALAR = "feature_prefix_plus_final_scalar"


def normalize_condition_layout(value: Any) -> dict[str, Any] | None:
    """Validate a registered condition-vector layout without assay hardcoding.

    The intensity-gated response tower consumes a generic feature prefix and
    exactly one final scalar.  This helper records that boundary in the task
    contract rather than assuming a particular task ID, fingerprint length, or
    scalar meaning in model code.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SchemaError("condition_layout must be an object when supplied")
    kind = str(value.get("kind", ""))
    if kind != CONDITION_PREFIX_FINAL_SCALAR:
        raise SchemaError(
            "condition_layout.kind must be "
            f"{CONDITION_PREFIX_FINAL_SCALAR!r}"
        )
    prefix_dim = value.get("prefix_dim")
    if not isinstance(prefix_dim, int) or isinstance(prefix_dim, bool) or prefix_dim < 1:
        raise SchemaError("condition_layout.prefix_dim must be a positive integer")
    scalar_position = str(value.get("scalar_position", ""))
    if scalar_position != "last":
        raise SchemaError("condition_layout.scalar_position must be 'last'")
    scalar_semantics = str(value.get("scalar_semantics", "")).strip()
    if not scalar_semantics:
        raise SchemaError("condition_layout.scalar_semantics must be a non-empty string")
    return {
        "kind": kind,
        "prefix_dim": int(prefix_dim),
        "scalar_position": scalar_position,
        "scalar_semantics": scalar_semantics,
    }


def file_fingerprint(path: str | Path) -> str:
    """Cheap stable fingerprint: absolute path, size, mtime and first/last blocks.

    It is not a cryptographic full-file checksum, but it detects the common
    accidental dataset replacement that otherwise makes an audit misleading.
    """
    source = Path(path)
    if not source.exists():
        return "missing"
    stat = source.stat()
    hasher = hashlib.sha256()
    hasher.update(str(source.resolve()).encode())
    hasher.update(str(stat.st_size).encode())
    hasher.update(str(stat.st_mtime_ns).encode())
    with source.open("rb") as handle:
        hasher.update(handle.read(1_048_576))
        if stat.st_size > 1_048_576:
            handle.seek(max(0, stat.st_size - 1_048_576))
            hasher.update(handle.read(1_048_576))
    return hasher.hexdigest()


def task_data_fingerprint(task: Mapping[str, Any]) -> str:
    """Fingerprint every raw source registered by a task, not only its first file."""
    source_paths = task.get("source_paths")
    if source_paths is None:
        source_paths = [task["path"]]
    if not isinstance(source_paths, (list, tuple)) or not source_paths:
        raise SchemaError("source_paths must be a non-empty list when supplied")
    return hashlib.sha256(
        json.dumps(
            {str(path): file_fingerprint(path) for path in source_paths},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def contract_from_task(task_id: str, task: Mapping[str, Any], training_budget: Mapping[str, Any] | None = None) -> TaskContract:
    required = {"family", "path", "target", "condition"}
    missing = required.difference(task)
    if missing:
        raise SchemaError(f"Task {task_id} missing required settings: {sorted(missing)}")
    allowed_target_modes = tuple(str(value) for value in task.get("allowed_target_modes", ("delta", "absolute")))
    if not allowed_target_modes or not set(allowed_target_modes).issubset({"delta", "absolute"}):
        raise SchemaError("allowed_target_modes must be a non-empty subset of {'delta', 'absolute'}")
    condition_layout = normalize_condition_layout(task.get("condition_layout"))
    protected_fields = (
        "data_path",
        "target",
        "condition",
        "metric",
        "test_fold",
        "allowed_target_modes",
    )
    if condition_layout is not None:
        protected_fields += ("condition_layout",)
    return TaskContract(
        task_id=task_id,
        task_family=str(task["family"]),
        data_path=str(task["path"]),
        target=str(task["target"]),
        condition=str(task["condition"]),
        metric=str(task.get("metric", "global_pcc")),
        test_fold=int(task.get("default_test_fold", 5)),
        allowed_target_modes=allowed_target_modes,
        condition_layout=condition_layout,
        protected_fields=protected_fields,
        training_budget=dict(training_budget or {}),
        data_fingerprint=task_data_fingerprint(task),
    )


def ensure_artifact_matches_contract(artifact: CandidateArtifact, contract: TaskContract) -> None:
    if artifact.task_id != contract.task_id:
        raise SchemaError(f"Artifact task {artifact.task_id} != contract task {contract.task_id}")
    if artifact.contract_hash != contract.contract_hash:
        raise SchemaError(
            "Artifact contract hash does not match current task contract. "
            "Recreate the artifact rather than silently transferring it."
        )
    if dict(artifact.training_budget) != dict(contract.training_budget):
        raise SchemaError("Artifact training budget differs from the frozen task contract")


def write_contract(path: str | Path, contract: TaskContract) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
