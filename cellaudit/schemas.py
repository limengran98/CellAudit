"""Small typed records shared across discovery, compilation, and audit.

The records deliberately store immutable provenance rather than mutable agent
state.  This is the boundary that lets an exploratory proposal later enter a
reproducible audit without pretending that the discovery itself was frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when an artifact is malformed or violates a task contract."""


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation suitable for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class TaskContract:
    """Fields that discovery and compilation are not allowed to silently alter."""

    task_id: str
    task_family: str
    data_path: str
    target: str
    condition: str
    metric: str = "global_pcc"
    test_fold: int = 5
    allowed_target_modes: tuple[str, ...] = ("delta", "absolute")
    protected_fields: tuple[str, ...] = (
        "data_path",
        "target",
        "condition",
        "metric",
        "test_fold",
        "allowed_target_modes",
    )
    training_budget: Mapping[str, Any] = field(default_factory=dict)
    data_fingerprint: str | None = None
    # Most response tasks expose an undifferentiated condition vector.  A
    # task may opt into this explicit layout only when one final scalar has a
    # registered, distinct role.  It is deliberately appended to preserve the
    # positional order of historical TaskContract construction sites.
    # ``contract_hash`` below omits it when absent.
    condition_layout: Mapping[str, Any] | None = None

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        # ``condition_layout`` was introduced for a new sibling task.  Do not
        # make a missing optional field retroactively alter an already frozen
        # task contract or its compiled candidate identity.
        if payload.get("condition_layout") is None:
            payload.pop("condition_layout", None)
        return digest(payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if result.get("condition_layout") is None:
            result.pop("condition_layout", None)
        result["protected_fields"] = list(self.protected_fields)
        result["allowed_target_modes"] = list(self.allowed_target_modes)
        result["contract_hash"] = self.contract_hash
        return result


@dataclass(frozen=True)
class CandidateArtifact:
    """One exploratory proposal before it becomes a frozen candidate record."""

    artifact_id: str
    task_id: str
    contract_hash: str
    source_model: str
    prompt_hash: str
    response_hash: str
    operator_spec: Mapping[str, Any]
    training_budget: Mapping[str, Any]
    rationale: str = ""
    parent_id: str | None = None
    code_hash: str | None = None
    created_at: str = field(default_factory=utcnow)
    status: str = "proposed"

    @property
    def artifact_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        return digest(payload)

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if include_hash:
            result["artifact_hash"] = self.artifact_hash
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateArtifact":
        required = {
            "artifact_id",
            "task_id",
            "contract_hash",
            "source_model",
            "prompt_hash",
            "response_hash",
            "operator_spec",
            "training_budget",
        }
        missing = required.difference(payload)
        if missing:
            raise SchemaError(f"CandidateArtifact missing required fields: {sorted(missing)}")
        kwargs = {name: payload[name] for name in cls.__dataclass_fields__ if name in payload}
        artifact = cls(**kwargs)
        claimed = payload.get("artifact_hash")
        if claimed is not None and claimed != artifact.artifact_hash:
            raise SchemaError(f"Artifact hash mismatch for {artifact.artifact_id}")
        return artifact


@dataclass(frozen=True)
class CompiledCandidate:
    """An immutable executable node produced by the compiler."""

    node_id: str
    artifact_id: str
    artifact_hash: str
    task_id: str
    contract_hash: str
    component_addresses: tuple[str, ...]
    normalized_spec: Mapping[str, Any]
    compiler_version: str = "0.1.0"

    @property
    def node_hash(self) -> str:
        return digest(self.to_dict(include_hash=False))

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["component_addresses"] = list(self.component_addresses)
        if include_hash:
            result["node_hash"] = self.node_hash
        return result


@dataclass(frozen=True)
class EvaluationRecord:
    """A frozen outcome from the selection or final-evaluation phase.

    ``selection_only`` records deliberately carry training/validation values
    only.  A final evaluation may reference one of those records through
    ``selection_record_hash``; downstream selection code consumes only its
    explicit training/validation feedback allow-list.
    """

    node_id: str
    task_id: str
    seed: int
    role: str
    metrics: Mapping[str, float]
    train_seconds: float
    device: str
    candidate_hash: str
    created_at: str = field(default_factory=utcnow)
    # Epoch zero denotes the shared mean-response reference before any optimizer
    # step.  It is a valid frozen selection outcome when no later epoch improves
    # validation performance.
    selection_epoch: int | None = None
    selection_record_hash: str | None = None
    # Hashes candidate identity, registered evaluator settings, selected split
    # roles, and train/validation shapes.  It contains no held-out values.
    selection_context_hash: str | None = None

    @property
    def record_hash(self) -> str:
        return digest(self.to_dict(include_hash=False))

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if include_hash:
            result["record_hash"] = self.record_hash
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationRecord":
        required = {
            "node_id",
            "task_id",
            "seed",
            "role",
            "metrics",
            "train_seconds",
            "device",
            "candidate_hash",
        }
        missing = required.difference(payload)
        if missing:
            raise SchemaError(f"EvaluationRecord missing required fields: {sorted(missing)}")
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping):
            raise SchemaError("EvaluationRecord metrics must be an object")
        kwargs = {name: payload[name] for name in cls.__dataclass_fields__ if name in payload}
        record = cls(**kwargs)
        claimed = payload.get("record_hash")
        if claimed is not None and claimed != record.record_hash:
            raise SchemaError("EvaluationRecord hash mismatch")
        return record


@dataclass(frozen=True)
class Probe:
    """A low-cost diagnostic measurement with registered likelihoods."""

    probe_id: str
    cost: float
    likelihoods: Mapping[str, Mapping[str, float]]
    description: str


@dataclass(frozen=True)
class FailureHypothesis:
    """Operational, not causal, explanations for a model's observed residual."""

    hypothesis_id: str
    target_address: str
    description: str
    prior: float


@dataclass(frozen=True)
class ProbeObservation:
    probe_id: str
    outcome: str


class JsonlEventStore:
    """Append-only JSONL log.  It is intentionally simple and portable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event_type": event_type, "timestamp": utcnow(), "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")

    def read(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SchemaError(f"Invalid JSONL at {self.path}:{line_number}") from error
            if event_type is None or record.get("event_type") == event_type:
                records.append(record)
        return records


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
