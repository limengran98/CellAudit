"""Typed contracts for candidate-source changes.

Open discovery may produce a free-form, multi-edit model-development step.
These data-free records preserve hashes, typed manifests, dependencies, and
parent preconditions without storing arrays, metrics, checkpoints, or model
state.  The public discovery entry point uses only the typed precondition
contract; the remaining records are retained for backwards-compatible receipt
parsing.

The module is deliberately data-free.  It stores hashes, typed manifests and
replay statuses, but no arrays, metrics, scores, checkpoints, or model state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any

from .schemas import SchemaError, canonical_json, digest


CANDIDATE_CHANGE_SCHEMA_VERSION = "cellscientist_open_candidate_change_set_v3"
MULTI_EDIT_MANIFEST_SCHEMA_VERSION = "cellscientist_structured_multi_edit_manifest_v1"

STATUS_BUNDLE_ONLY = "bundle_only"
STATUS_DEPENDENCY_LOCKED = "dependency_locked"
STATUS_REPLAYABLE = "replayable"
STATUS_NON_REPLAYABLE = "non_replayable"
STATUS_REJECTED_CONTRACT = "rejected_contract"
REPLAY_STATUSES = frozenset(
    {
        STATUS_BUNDLE_ONLY,
        STATUS_DEPENDENCY_LOCKED,
        STATUS_REPLAYABLE,
        STATUS_NON_REPLAYABLE,
        STATUS_REJECTED_CONTRACT,
    }
)

OBSERVED_BUNDLE_STATUSES = frozenset(
    {"completed", "failed", "accepted", "rejected", "inconclusive"}
)
REPLAY_REPORT_STATUSES = frozenset(
    {"passed", "failed", "not_attempted", "not_materializable", "dependency_qualified"}
)
RECONSTRUCTION_STATUSES = frozenset(
    {"exact", "dependency_qualified", "not_materializable"}
)
PRECONDITION_KINDS = frozenset(
    {
        "address_present",
        "address_absent",
        "parent_source_hash",
        "parent_config_hash",
        "typed_io_contract",
        "parameter_budget",
        "compute_budget",
    }
)
# The open-discovery manifest has a slightly richer vocabulary than the
# executable compiler. Keep the boundary explicit: a precondition is either
# checked by both the static change set and runtime replayer, or it is not admitted
# as replayable at all.  Treating an uninterpreted type/budget assertion as a
# comment would make the change set claim a legal state that runtime cannot verify.
RUNTIME_ENFORCEABLE_PRECONDITION_KINDS = frozenset(
    {
        "address_present",
        "address_absent",
        "parent_source_hash",
        "parent_config_hash",
    }
)
UNSUPPORTED_RUNTIME_PRECONDITION_KINDS = (
    PRECONDITION_KINDS - RUNTIME_ENFORCEABLE_PRECONDITION_KINDS
)

# Addresses are intentionally open within a unified semantic namespace.  This
# is not a fixed BBBC036 operator bank: discovery may introduce a new address
# under a supported perturbation-modeling prefix, subject to manifest validation.
# Keep the change set's open address language aligned with the source-candidate
# validation layer.  New suffixes remain legal; only the top-level semantic
# namespace is protected from generic ``encoder``/``layer`` naming.
ADDRESS_PREFIXES = (
    "input.",
    "perturbation.",
    "response.",
    "objective.",
    "optimization.",
    "nuisance.",
    "reliability.",
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,159}$")
_ADDRESS_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


class CandidateChangeError(SchemaError):
    """Raised for invalid open-discovery change set records."""


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise CandidateChangeError(f"{name} must be a non-empty identifier")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise CandidateChangeError(f"{name} must be a lowercase SHA-256 digest")
    result = value
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CandidateChangeError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _strict_object(
    value: Any,
    *,
    name: str,
    required: set[str] | frozenset[str] | tuple[str, ...],
    allowed: set[str] | frozenset[str] | tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CandidateChangeError(f"{name} must be a JSON object with string keys")
    missing = set(required).difference(value)
    unknown = set(value).difference(allowed)
    if missing:
        raise CandidateChangeError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise CandidateChangeError(f"{name} has unknown fields: {sorted(unknown)}")
    return value


def _identifier_tuple(value: Any, *, name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise CandidateChangeError(f"{name} must be an array")
    values = tuple(_identifier(item, name=f"{name}[]") for item in value)
    if not allow_empty and not values:
        raise CandidateChangeError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise CandidateChangeError(f"{name} must not contain duplicates")
    if tuple(sorted(values)) != values:
        raise CandidateChangeError(f"{name} must be lexicographically sorted")
    return values


_FORBIDDEN_PAYLOAD_TERMS = (
    "fold3",
    "fold_3",
    "feedback_score",
    "selection_score",
    "validation_score",
    "global_pcc",
    "model_weight",
    "modelweight",
    "weights",
    "state_dict",
    "checkpoint",
    "optimizer_state",
    "gradient",
)


def _assert_payload_safe(value: Any, *, name: str = "payload") -> None:
    """Reject data, score, and learned-state payloads at the change set boundary."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CandidateChangeError(f"{name} keys must be strings")
            if any(term in key.lower() for term in _FORBIDDEN_PAYLOAD_TERMS):
                raise CandidateChangeError(f"{name} may not contain Fold-3 score or model-weight field {key!r}")
            _assert_payload_safe(item, name=f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _assert_payload_safe(item, name=f"{name}[]")
        return
    if isinstance(value, str):
        if any(term in value.lower() for term in _FORBIDDEN_PAYLOAD_TERMS):
            raise CandidateChangeError(f"{name} may not contain Fold-3 score or model-weight content")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CandidateChangeError(f"{name} must not contain NaN or infinity")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    raise CandidateChangeError(f"{name} must contain JSON values only")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def validate_semantic_address(value: Any) -> str:
    """Validate a discovery-supplied semantic address under the common schema."""

    if not isinstance(value, str) or not _ADDRESS_RE.fullmatch(value):
        raise CandidateChangeError("semantic address must use lowercase dotted components")
    if not value.startswith(ADDRESS_PREFIXES):
        raise CandidateChangeError(
            "semantic address must start with one of: " + ", ".join(ADDRESS_PREFIXES)
        )
    return value


def _object_or_attr(value: Any, names: Sequence[str], *, name: str) -> Any:
    for candidate in names:
        if isinstance(value, Mapping) and candidate in value:
            return value[candidate]
        if hasattr(value, candidate):
            return getattr(value, candidate)
    raise CandidateChangeError(f"{name} is missing one of {list(names)}")


@dataclass(frozen=True)
class Precondition:
    """A typed requirement for applying one candidate-source change."""

    precondition_id: str
    kind: str
    value: str

    def __post_init__(self) -> None:
        _identifier(self.precondition_id, name="precondition_id")
        if self.kind not in PRECONDITION_KINDS:
            raise CandidateChangeError(f"unknown precondition kind {self.kind!r}")
        if self.kind in {"address_present", "address_absent"}:
            validate_semantic_address(self.value)
        elif self.kind in {"parent_source_hash", "parent_config_hash"}:
            _sha256(self.value, name=f"precondition {self.kind}")
        else:
            _identifier(self.value, name=f"precondition {self.kind}")

    def to_dict(self) -> dict[str, str]:
        return {"precondition_id": self.precondition_id, "kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, payload: Any) -> "Precondition":
        record = _strict_object(
            payload,
            name="precondition",
            required={"precondition_id", "kind", "value"},
            allowed={"precondition_id", "kind", "value"},
        )
        return cls(
            precondition_id=_identifier(record["precondition_id"], name="precondition_id"),
            kind=str(record["kind"]),
            value=str(record["value"]),
        )


@dataclass(frozen=True)
class ReplayReport:
    """Data-free executable materialization report for a single atomic patch."""

    status: str
    validator_hash: str
    replay_source_hash: str | None = None
    replay_config_hash: str | None = None
    detail_code: str = "none"

    def __post_init__(self) -> None:
        if self.status not in REPLAY_REPORT_STATUSES:
            raise CandidateChangeError(f"unknown replay report status {self.status!r}")
        _sha256(self.validator_hash, name="validator_hash")
        if self.replay_source_hash is not None:
            _sha256(self.replay_source_hash, name="replay_source_hash")
        if self.replay_config_hash is not None:
            _sha256(self.replay_config_hash, name="replay_config_hash")
        _identifier(self.detail_code, name="replay detail_code")
        if self.status == "passed" and (
            self.replay_source_hash is None or self.replay_config_hash is None
        ):
            raise CandidateChangeError("passed replay reports require source and config replay hashes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "validator_hash": self.validator_hash,
            "replay_source_hash": self.replay_source_hash,
            "replay_config_hash": self.replay_config_hash,
            "detail_code": self.detail_code,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ReplayReport":
        record = _strict_object(
            payload,
            name="replay_report",
            required={
                "status",
                "validator_hash",
                "replay_source_hash",
                "replay_config_hash",
                "detail_code",
            },
            allowed={
                "status",
                "validator_hash",
                "replay_source_hash",
                "replay_config_hash",
                "detail_code",
            },
        )
        return cls(
            status=str(record["status"]),
            validator_hash=_sha256(record["validator_hash"], name="validator_hash"),
            replay_source_hash=(
                None
                if record["replay_source_hash"] is None
                else _sha256(record["replay_source_hash"], name="replay_source_hash")
            ),
            replay_config_hash=(
                None
                if record["replay_config_hash"] is None
                else _sha256(record["replay_config_hash"], name="replay_config_hash")
            ),
            detail_code=_identifier(record["detail_code"], name="detail_code"),
        )


@dataclass(frozen=True)
class ManifestEdit:
    """One validated atomic edit extracted from a free-form discovery proposal."""

    edit_id: str
    semantic_address: str
    semantic_interpretation: str
    source_patch_hash: str
    config_patch_hash: str
    preconditions: tuple[Precondition, ...]
    dependencies: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    replay_report: ReplayReport

    def __post_init__(self) -> None:
        _identifier(self.edit_id, name="edit_id")
        validate_semantic_address(self.semantic_address)
        if not isinstance(self.semantic_interpretation, str) or not self.semantic_interpretation.strip():
            raise CandidateChangeError("semantic_interpretation must be non-empty")
        _assert_payload_safe(self.semantic_interpretation, name="semantic_interpretation")
        _sha256(self.source_patch_hash, name="source_patch_hash")
        _sha256(self.config_patch_hash, name="config_patch_hash")
        if not isinstance(self.replay_report, ReplayReport):
            raise CandidateChangeError("manifest edit requires a ReplayReport")
        if len({item.precondition_id for item in self.preconditions}) != len(self.preconditions):
            raise CandidateChangeError("manifest edit has duplicate precondition IDs")
        if not all(isinstance(item, Precondition) for item in self.preconditions):
            raise CandidateChangeError("manifest edit preconditions must be typed")
        object.__setattr__(self, "dependencies", _identifier_tuple(self.dependencies, name="dependencies"))
        object.__setattr__(self, "incompatibilities", _identifier_tuple(self.incompatibilities, name="incompatibilities"))
        if self.edit_id in self.dependencies or self.edit_id in self.incompatibilities:
            raise CandidateChangeError("manifest edit cannot depend on or conflict with itself")
        if set(self.dependencies).intersection(self.incompatibilities):
            raise CandidateChangeError("manifest edit cannot both depend on and conflict with an edit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "semantic_address": self.semantic_address,
            "semantic_interpretation": self.semantic_interpretation,
            "source_patch_hash": self.source_patch_hash,
            "config_patch_hash": self.config_patch_hash,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "dependencies": list(self.dependencies),
            "incompatibilities": list(self.incompatibilities),
            "replay_report": self.replay_report.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ManifestEdit":
        fields = {
            "edit_id",
            "semantic_address",
            "semantic_interpretation",
            "source_patch_hash",
            "config_patch_hash",
            "preconditions",
            "dependencies",
            "incompatibilities",
            "replay_report",
        }
        record = _strict_object(payload, name="manifest_edit", required=fields, allowed=fields)
        if not isinstance(record["preconditions"], (list, tuple)):
            raise CandidateChangeError("manifest_edit.preconditions must be an array")
        return cls(
            edit_id=_identifier(record["edit_id"], name="edit_id"),
            semantic_address=validate_semantic_address(record["semantic_address"]),
            semantic_interpretation=str(record["semantic_interpretation"]),
            source_patch_hash=_sha256(record["source_patch_hash"], name="source_patch_hash"),
            config_patch_hash=_sha256(record["config_patch_hash"], name="config_patch_hash"),
            preconditions=tuple(Precondition.from_dict(item) for item in record["preconditions"]),
            dependencies=_identifier_tuple(record["dependencies"], name="dependencies"),
            incompatibilities=_identifier_tuple(record["incompatibilities"], name="incompatibilities"),
            replay_report=ReplayReport.from_dict(record["replay_report"]),
        )


@dataclass(frozen=True)
class BundleReconstruction:
    """Whether applying all manifest edits recovers the observed child hashes."""

    status: str
    applied_edit_ids: tuple[str, ...]
    reconstructed_source_hash: str | None
    reconstructed_config_hash: str | None

    def __post_init__(self) -> None:
        if self.status not in RECONSTRUCTION_STATUSES:
            raise CandidateChangeError(f"unknown bundle reconstruction status {self.status!r}")
        object.__setattr__(self, "applied_edit_ids", _identifier_tuple(self.applied_edit_ids, name="applied_edit_ids"))
        if self.reconstructed_source_hash is not None:
            _sha256(self.reconstructed_source_hash, name="reconstructed_source_hash")
        if self.reconstructed_config_hash is not None:
            _sha256(self.reconstructed_config_hash, name="reconstructed_config_hash")
        if self.status == "exact" and (
            self.reconstructed_source_hash is None or self.reconstructed_config_hash is None
        ):
            raise CandidateChangeError("exact reconstruction requires source and configuration hashes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "applied_edit_ids": list(self.applied_edit_ids),
            "reconstructed_source_hash": self.reconstructed_source_hash,
            "reconstructed_config_hash": self.reconstructed_config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "BundleReconstruction":
        record = _strict_object(
            payload,
            name="bundle_reconstruction",
            required={
                "status",
                "applied_edit_ids",
                "reconstructed_source_hash",
                "reconstructed_config_hash",
            },
            allowed={
                "status",
                "applied_edit_ids",
                "reconstructed_source_hash",
                "reconstructed_config_hash",
            },
        )
        return cls(
            status=str(record["status"]),
            applied_edit_ids=_identifier_tuple(record["applied_edit_ids"], name="applied_edit_ids"),
            reconstructed_source_hash=(
                None
                if record["reconstructed_source_hash"] is None
                else _sha256(record["reconstructed_source_hash"], name="reconstructed_source_hash")
            ),
            reconstructed_config_hash=(
                None
                if record["reconstructed_config_hash"] is None
                else _sha256(record["reconstructed_config_hash"], name="reconstructed_config_hash")
            ),
        )


@dataclass(frozen=True)
class StructuredMultiEditManifest:
    """Validated, source/config-hash-only manifest attached to one observed edge."""

    manifest_id: str
    edits: tuple[ManifestEdit, ...]
    reconstruction: BundleReconstruction

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, name="manifest_id")
        if not self.edits:
            raise CandidateChangeError("structured manifest must contain at least one edit")
        if len({item.edit_id for item in self.edits}) != len(self.edits):
            raise CandidateChangeError("structured manifest has duplicate edit IDs")
        if not all(isinstance(item, ManifestEdit) for item in self.edits):
            raise CandidateChangeError("structured manifest edits must be typed")
        ids = {item.edit_id for item in self.edits}
        for edit in self.edits:
            if not set(edit.dependencies).issubset(ids):
                raise CandidateChangeError("manifest edit dependency references an unknown edit")
            if not set(edit.incompatibilities).issubset(ids):
                raise CandidateChangeError("manifest edit incompatibility references an unknown edit")
        if not set(self.reconstruction.applied_edit_ids).issubset(ids):
            raise CandidateChangeError("reconstruction references an unknown manifest edit")
        object.__setattr__(self, "edits", tuple(sorted(self.edits, key=lambda item: item.edit_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MULTI_EDIT_MANIFEST_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "edits": [item.to_dict() for item in self.edits],
            "reconstruction": self.reconstruction.to_dict(),
        }

    @property
    def manifest_hash(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> "StructuredMultiEditManifest":
        fields = {"schema_version", "manifest_id", "edits", "reconstruction"}
        record = _strict_object(payload, name="structured_manifest", required=fields, allowed=fields)
        if record["schema_version"] != MULTI_EDIT_MANIFEST_SCHEMA_VERSION:
            raise CandidateChangeError("structured manifest has an unsupported schema_version")
        if not isinstance(record["edits"], (list, tuple)):
            raise CandidateChangeError("structured_manifest.edits must be an array")
        return cls(
            manifest_id=_identifier(record["manifest_id"], name="manifest_id"),
            edits=tuple(ManifestEdit.from_dict(item) for item in record["edits"]),
            reconstruction=BundleReconstruction.from_dict(record["reconstruction"]),
        )


@dataclass(frozen=True)
class BundleTransition:
    """The complete observed discovery edge; it is never dropped during compilation."""

    transition_id: str
    parent_state_id: str
    child_state_id: str
    task_contract_hash: str
    parent_source_hash: str
    parent_config_hash: str
    child_source_hash: str
    child_config_hash: str
    observed_status: str
    manifest: StructuredMultiEditManifest | None = None
    parent_revision_ids: tuple[str, ...] = ()
    manifest_validation_error: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("transition_id", self.transition_id),
            ("parent_state_id", self.parent_state_id),
            ("child_state_id", self.child_state_id),
        ):
            _identifier(value, name=name)
        for name, value in (
            ("task_contract_hash", self.task_contract_hash),
            ("parent_source_hash", self.parent_source_hash),
            ("parent_config_hash", self.parent_config_hash),
            ("child_source_hash", self.child_source_hash),
            ("child_config_hash", self.child_config_hash),
        ):
            _sha256(value, name=name)
        if self.observed_status not in OBSERVED_BUNDLE_STATUSES:
            raise CandidateChangeError("bundle transition has an unknown observed_status")
        if self.manifest is not None and not isinstance(self.manifest, StructuredMultiEditManifest):
            raise CandidateChangeError("bundle transition manifest must be a StructuredMultiEditManifest")
        object.__setattr__(
            self,
            "parent_revision_ids",
            _identifier_tuple(self.parent_revision_ids, name="parent_revision_ids"),
        )
        if self.manifest_validation_error is not None:
            _identifier(self.manifest_validation_error, name="manifest_validation_error")
            if self.manifest is not None:
                raise CandidateChangeError("manifest_validation_error requires no admitted structured manifest")

    @classmethod
    def from_open_discovery_response(
        cls,
        response: Any,
        *,
        transition_id: str,
        parent_state_id: str,
        child_state_id: str,
        task_contract_hash: str,
        parent_source_hash: str,
        parent_config_hash: str,
        child_source_hash: str | None = None,
        child_config_hash: str | None = None,
        manifest: Mapping[str, Any] | StructuredMultiEditManifest | None = None,
        observed_status: str | None = None,
        parent_revision_ids: Sequence[str] = (),
    ) -> "BundleTransition":
        """Loose adapter for an OpenDiscoveryResponse or equivalent mapping.

        The adapter needs only stable hashes and a validated structured manifest;
        it intentionally does not ingest a response's diagnostics, score fields,
        source text, or trained state.  This keeps it compatible with the final
        `open_discovery.py` record without coupling the two modules.
        """

        if child_source_hash is None:
            child_source_hash = _object_or_attr(
                response,
                ("child_source_hash", "candidate_source_hash", "source_hash"),
                name="open discovery response",
            )
        if child_config_hash is None:
            child_config_hash = _object_or_attr(
                response,
                ("child_config_hash", "candidate_config_hash", "config_hash"),
                name="open discovery response",
            )
        if observed_status is None:
            try:
                observed_status = _object_or_attr(
                    response, ("observed_status", "execution_status", "status"), name="open discovery response"
                )
            except CandidateChangeError:
                observed_status = "completed"
        if manifest is None:
            try:
                manifest = _object_or_attr(
                    response,
                    ("structured_change_manifest", "change_manifest", "manifest"),
                    name="open discovery response",
                )
            except CandidateChangeError:
                manifest = None
        manifest_error: str | None = None
        try:
            typed_manifest = (
                manifest
                if isinstance(manifest, StructuredMultiEditManifest)
                else (StructuredMultiEditManifest.from_dict(manifest) if manifest is not None else None)
            )
        except CandidateChangeError:
            # The observed transition is still valuable provenance.  Preserve
            # it as a rejected-contract bundle rather than silently dropping a
            # malformed free-form manifest from the discovery trajectory.
            typed_manifest = None
            manifest_error = "invalid_manifest"
        return cls(
            transition_id=transition_id,
            parent_state_id=parent_state_id,
            child_state_id=child_state_id,
            task_contract_hash=_sha256(task_contract_hash, name="task_contract_hash"),
            parent_source_hash=_sha256(parent_source_hash, name="parent_source_hash"),
            parent_config_hash=_sha256(parent_config_hash, name="parent_config_hash"),
            child_source_hash=_sha256(child_source_hash, name="child_source_hash"),
            child_config_hash=_sha256(child_config_hash, name="child_config_hash"),
            observed_status=str(observed_status),
            manifest=typed_manifest,
            parent_revision_ids=tuple(parent_revision_ids),
            manifest_validation_error=manifest_error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "parent_state_id": self.parent_state_id,
            "child_state_id": self.child_state_id,
            "task_contract_hash": self.task_contract_hash,
            "parent_source_hash": self.parent_source_hash,
            "parent_config_hash": self.parent_config_hash,
            "child_source_hash": self.child_source_hash,
            "child_config_hash": self.child_config_hash,
            "observed_status": self.observed_status,
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
            "parent_revision_ids": list(self.parent_revision_ids),
            "manifest_validation_error": self.manifest_validation_error,
        }

    @property
    def transition_hash(self) -> str:
        """Bind private executable artifacts to this exact observed edge."""

        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> "BundleTransition":
        fields = {
            "transition_id",
            "parent_state_id",
            "child_state_id",
            "task_contract_hash",
            "parent_source_hash",
            "parent_config_hash",
            "child_source_hash",
            "child_config_hash",
            "observed_status",
            "manifest",
            "parent_revision_ids",
            "manifest_validation_error",
        }
        record = _strict_object(payload, name="bundle_transition", required=fields, allowed=fields)
        return cls(
            transition_id=_identifier(record["transition_id"], name="transition_id"),
            parent_state_id=_identifier(record["parent_state_id"], name="parent_state_id"),
            child_state_id=_identifier(record["child_state_id"], name="child_state_id"),
            task_contract_hash=_sha256(record["task_contract_hash"], name="task_contract_hash"),
            parent_source_hash=_sha256(record["parent_source_hash"], name="parent_source_hash"),
            parent_config_hash=_sha256(record["parent_config_hash"], name="parent_config_hash"),
            child_source_hash=_sha256(record["child_source_hash"], name="child_source_hash"),
            child_config_hash=_sha256(record["child_config_hash"], name="child_config_hash"),
            observed_status=str(record["observed_status"]),
            manifest=(
                None if record["manifest"] is None else StructuredMultiEditManifest.from_dict(record["manifest"])
            ),
            parent_revision_ids=_identifier_tuple(record["parent_revision_ids"], name="parent_revision_ids"),
            manifest_validation_error=(
                None
                if record["manifest_validation_error"] is None
                else _identifier(record["manifest_validation_error"], name="manifest_validation_error")
            ),
        )


@dataclass(frozen=True)
class CrossParentApplicability:
    """Declared conditions for replaying an atomic edit on another parent state."""

    same_contract_only: bool = True
    allow_same_address_reapply: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.same_contract_only, bool) or not isinstance(self.allow_same_address_reapply, bool):
            raise CandidateChangeError("cross-parent applicability flags must be boolean")

    def to_dict(self) -> dict[str, bool]:
        return {
            "same_contract_only": self.same_contract_only,
            "allow_same_address_reapply": self.allow_same_address_reapply,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "CrossParentApplicability":
        record = _strict_object(
            payload,
            name="cross_parent",
            required={"same_contract_only", "allow_same_address_reapply"},
            allowed={"same_contract_only", "allow_same_address_reapply"},
        )
        return cls(
            same_contract_only=record["same_contract_only"],
            allow_same_address_reapply=record["allow_same_address_reapply"],
        )


@dataclass(frozen=True)
class AtomicRevision:
    """Data-free, replay-addressable fragment extracted from a manifest edit."""

    revision_id: str
    source_transition_id: str
    manifest_hash: str
    task_contract_hash: str
    semantic_address: str
    semantic_interpretation: str
    source_patch_hash: str
    config_patch_hash: str
    preconditions: tuple[Precondition, ...]
    dependencies: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    replay_report: ReplayReport
    cross_parent: CrossParentApplicability = CrossParentApplicability()

    def __post_init__(self) -> None:
        for name, value in (
            ("revision_id", self.revision_id),
            ("source_transition_id", self.source_transition_id),
        ):
            _identifier(value, name=name)
        for name, value in (
            ("manifest_hash", self.manifest_hash),
            ("task_contract_hash", self.task_contract_hash),
            ("source_patch_hash", self.source_patch_hash),
            ("config_patch_hash", self.config_patch_hash),
        ):
            _sha256(value, name=name)
        validate_semantic_address(self.semantic_address)
        if not isinstance(self.semantic_interpretation, str) or not self.semantic_interpretation.strip():
            raise CandidateChangeError("atomic semantic_interpretation must be non-empty")
        _assert_payload_safe(self.semantic_interpretation, name="atomic semantic_interpretation")
        if not all(isinstance(item, Precondition) for item in self.preconditions):
            raise CandidateChangeError("atomic preconditions must be typed")
        if not isinstance(self.replay_report, ReplayReport):
            raise CandidateChangeError("atomic replay_report must be typed")
        if not isinstance(self.cross_parent, CrossParentApplicability):
            raise CandidateChangeError("atomic cross_parent must be typed")
        object.__setattr__(self, "dependencies", _identifier_tuple(self.dependencies, name="atomic dependencies"))
        object.__setattr__(self, "incompatibilities", _identifier_tuple(self.incompatibilities, name="atomic incompatibilities"))
        if self.revision_id in self.dependencies or self.revision_id in self.incompatibilities:
            raise CandidateChangeError("atomic revision cannot depend on or conflict with itself")
        if set(self.dependencies).intersection(self.incompatibilities):
            raise CandidateChangeError("atomic dependencies and incompatibilities overlap")

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "source_transition_id": self.source_transition_id,
            "manifest_hash": self.manifest_hash,
            "task_contract_hash": self.task_contract_hash,
            "semantic_address": self.semantic_address,
            "semantic_interpretation": self.semantic_interpretation,
            "source_patch_hash": self.source_patch_hash,
            "config_patch_hash": self.config_patch_hash,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "dependencies": list(self.dependencies),
            "incompatibilities": list(self.incompatibilities),
            "replay_report": self.replay_report.to_dict(),
            "cross_parent": self.cross_parent.to_dict(),
        }


def extract_atomic_revisions(transition: BundleTransition) -> tuple[AtomicRevision, ...]:
    """Extract all candidate atomics from a validated multi-edit manifest.

    A transition with no manifest is an observed ``bundle_only`` edge.  It is
    still retained by :func:`compile_candidate_change_set` rather than discarded.
    """

    if transition.manifest is None:
        return ()
    manifest = transition.manifest
    revision_by_edit = {edit.edit_id: f"{transition.transition_id}.{edit.edit_id}" for edit in manifest.edits}
    return tuple(
        AtomicRevision(
            revision_id=revision_by_edit[edit.edit_id],
            source_transition_id=transition.transition_id,
            manifest_hash=manifest.manifest_hash,
            task_contract_hash=transition.task_contract_hash,
            semantic_address=edit.semantic_address,
            semantic_interpretation=edit.semantic_interpretation,
            source_patch_hash=edit.source_patch_hash,
            config_patch_hash=edit.config_patch_hash,
            preconditions=edit.preconditions,
            dependencies=tuple(sorted(revision_by_edit[item] for item in edit.dependencies)),
            incompatibilities=tuple(sorted(revision_by_edit[item] for item in edit.incompatibilities)),
            replay_report=edit.replay_report,
        )
        for edit in manifest.edits
    )


def _expected_atomic(transition: BundleTransition, revision: AtomicRevision) -> ManifestEdit | None:
    if transition.manifest is None or revision.manifest_hash != transition.manifest.manifest_hash:
        return None
    for edit in transition.manifest.edits:
        if revision.revision_id == f"{transition.transition_id}.{edit.edit_id}":
            return edit
    return None


def _contract_reasons(
    transition: BundleTransition,
    revision: AtomicRevision,
    *,
    task_contract_hash: str,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if transition.task_contract_hash != task_contract_hash:
        reasons.append("transition_contract_mismatch")
    if revision.task_contract_hash != task_contract_hash:
        reasons.append("revision_contract_mismatch")
    if revision.source_transition_id != transition.transition_id:
        reasons.append("source_transition_mismatch")
    expected = _expected_atomic(transition, revision)
    if expected is None:
        reasons.append("bundle_reconstruction_failed")
    else:
        if (
            revision.semantic_address != expected.semantic_address
            or revision.semantic_interpretation != expected.semantic_interpretation
            or revision.source_patch_hash != expected.source_patch_hash
            or revision.config_patch_hash != expected.config_patch_hash
            or revision.preconditions != expected.preconditions
            or revision.replay_report != expected.replay_report
        ):
            reasons.append("atomic_manifest_mismatch")
    try:
        _assert_payload_safe(revision.to_dict(), name="atomic_revision")
    except CandidateChangeError:
        reasons.append("public_payload_unsafe")
    return tuple(sorted(set(reasons)))


def _reconstruction_reason(transition: BundleTransition) -> str | None:
    manifest = transition.manifest
    if manifest is None:
        return "no_manifest"
    reconstruction = manifest.reconstruction
    edit_ids = tuple(sorted(item.edit_id for item in manifest.edits))
    if reconstruction.status == "not_materializable":
        return "not_materializable"
    if reconstruction.status == "dependency_qualified":
        return "dependency_qualified"
    if reconstruction.applied_edit_ids != edit_ids:
        return "exact_reconstruction_edit_set_mismatch"
    if (
        reconstruction.reconstructed_source_hash != transition.child_source_hash
        or reconstruction.reconstructed_config_hash != transition.child_config_hash
    ):
        return "exact_reconstruction_hash_mismatch"
    return None


def _atomic_source_status(
    revision: AtomicRevision,
    *,
    parent_revision_ids: Sequence[str],
    known_revisions: Mapping[str, AtomicRevision],
) -> str:
    if revision.replay_report.status in {"failed", "not_attempted", "not_materializable"}:
        return STATUS_NON_REPLAYABLE
    if revision.replay_report.status == "dependency_qualified":
        return STATUS_DEPENDENCY_LOCKED
    if not set(parent_revision_ids).issubset(known_revisions):
        return STATUS_DEPENDENCY_LOCKED
    if not set(revision.dependencies).issubset(parent_revision_ids):
        return STATUS_DEPENDENCY_LOCKED
    if _conflicts_with_parent(
        revision,
        parent_revision_ids=parent_revision_ids,
        known_revisions=known_revisions,
    ):
        return STATUS_DEPENDENCY_LOCKED
    return STATUS_REPLAYABLE


def _conflicts_with_parent(
    revision: AtomicRevision,
    *,
    parent_revision_ids: Sequence[str],
    known_revisions: Mapping[str, AtomicRevision],
) -> bool:
    """Check incompatibilities declared by either endpoint.

    Discovery manifests may name an exclusion from only one side.  The change set
    nevertheless represents an undirected exclusion relation at replay time:
    a later revision cannot bypass a conflict merely because the previously
    applied revision carried the declaration.
    """

    parent_ids = set(parent_revision_ids)
    if set(revision.incompatibilities).intersection(parent_ids):
        return True
    return any(
        revision.revision_id in known_revisions[parent_id].incompatibilities
        for parent_id in parent_ids
        if parent_id in known_revisions
    )


def _preconditions_hold(
    revision: AtomicRevision,
    *,
    parent_revision_ids: Sequence[str],
    parent_source_hash: str | None,
    parent_config_hash: str | None,
    known_revisions: Mapping[str, AtomicRevision],
) -> bool:
    if not set(parent_revision_ids).issubset(known_revisions):
        return False
    # ``allow_same_address_reapply`` permits a later *revision* at the same
    # address, never application of the identical frozen revision twice.
    if revision.revision_id in parent_revision_ids:
        return False
    addresses = {known_revisions[item].semantic_address for item in parent_revision_ids}
    if not revision.cross_parent.allow_same_address_reapply and revision.semantic_address in addresses:
        return False
    if not set(revision.dependencies).issubset(parent_revision_ids):
        return False
    if _conflicts_with_parent(
        revision,
        parent_revision_ids=parent_revision_ids,
        known_revisions=known_revisions,
    ):
        return False
    for precondition in revision.preconditions:
        if precondition.kind in UNSUPPORTED_RUNTIME_PRECONDITION_KINDS:
            return False
        if precondition.kind not in RUNTIME_ENFORCEABLE_PRECONDITION_KINDS:
            return False
        if precondition.kind == "address_present" and precondition.value not in addresses:
            return False
        if precondition.kind == "address_absent" and precondition.value in addresses:
            return False
        if precondition.kind == "parent_source_hash" and precondition.value != parent_source_hash:
            return False
        if precondition.kind == "parent_config_hash" and precondition.value != parent_config_hash:
            return False
    return True


def _dependency_ancestry(
    revision_ids: Sequence[str],
    *,
    known_revisions: Mapping[str, AtomicRevision],
) -> frozenset[str]:
    """Return known parent/dependency IDs without relying on transition order."""

    pending = list(revision_ids)
    result: set[str] = set()
    while pending:
        revision_id = pending.pop()
        if revision_id in result or revision_id not in known_revisions:
            continue
        result.add(revision_id)
        pending.extend(known_revisions[revision_id].dependencies)
    return frozenset(result)


def _canonicalize_atomic_constraints(
    atomics: Sequence[AtomicRevision],
    *,
    transitions_by_id: Mapping[str, BundleTransition],
) -> tuple[AtomicRevision, ...]:
    """Compile observed lineage into runtime-enforceable atomic constraints.

    A repeated semantic address is legal only when the observed parent lineage
    (or an explicit local dependency) proves that it is a sequential update.
    This turns a formerly implicit source-transition fact into the same
    ``cross_parent`` flag consumed by static selection and runtime replay.

    Incompatibilities are canonicalized bidirectionally for known atomics.  A
    one-sided declaration remains traceable in the manifest, while every
    compiled consumer sees the same exclusion relation.
    """

    known = {item.revision_id: item for item in atomics}
    incompatible_by_id = {
        item.revision_id: set(item.incompatibilities) for item in atomics
    }
    for revision_id, incompatible_ids in tuple(incompatible_by_id.items()):
        for incompatible_id in tuple(incompatible_ids):
            if incompatible_id in incompatible_by_id:
                incompatible_by_id[incompatible_id].add(revision_id)

    normalized: list[AtomicRevision] = []
    for revision in atomics:
        transition = transitions_by_id.get(revision.source_transition_id)
        observed_parents = () if transition is None else transition.parent_revision_ids
        ancestry = _dependency_ancestry(
            tuple(observed_parents) + tuple(revision.dependencies),
            known_revisions=known,
        )
        observed_same_address_update = any(
            known[parent_id].semantic_address == revision.semantic_address
            for parent_id in ancestry
        )
        cross_parent = revision.cross_parent
        if observed_same_address_update and not cross_parent.allow_same_address_reapply:
            cross_parent = CrossParentApplicability(
                same_contract_only=cross_parent.same_contract_only,
                allow_same_address_reapply=True,
            )
        incompatibilities = tuple(sorted(incompatible_by_id[revision.revision_id]))
        overlap = set(revision.dependencies).intersection(incompatibilities)
        if overlap:
            raise CandidateChangeError(
                "compiled dependency/incompatibility overlap for "
                f"{revision.revision_id!r}: {sorted(overlap)}"
            )
        normalized.append(
            replace(
                revision,
                incompatibilities=incompatibilities,
                cross_parent=cross_parent,
            )
        )
    return tuple(normalized)


@dataclass(frozen=True)
class AtomicChangeRecord:
    revision: AtomicRevision
    replay_status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.replay_status not in REPLAY_STATUSES - {STATUS_BUNDLE_ONLY}:
            raise CandidateChangeError("atomic change set nodes cannot have bundle_only status")
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision.to_dict(),
            "replay_status": self.replay_status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BundleEdge:
    transition: BundleTransition
    replay_status: str
    atomic_revision_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.replay_status not in REPLAY_STATUSES:
            raise CandidateChangeError("bundle edge has an unknown replay_status")
        object.__setattr__(self, "atomic_revision_ids", _identifier_tuple(self.atomic_revision_ids, name="atomic_revision_ids"))
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))
        if self.replay_status == STATUS_BUNDLE_ONLY and self.atomic_revision_ids:
            raise CandidateChangeError("bundle_only edge cannot name atomic revisions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition": self.transition.to_dict(),
            "replay_status": self.replay_status,
            "atomic_revision_ids": list(self.atomic_revision_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CandidateChangeSet:
    """Frozen observed-bundle and atomic-candidate change set for later selection."""

    task_contract_hash: str
    bundle_edges: tuple[BundleEdge, ...]
    atomic_nodes: tuple[AtomicChangeRecord, ...]

    def __post_init__(self) -> None:
        _sha256(self.task_contract_hash, name="change set task_contract_hash")
        if len({edge.transition.transition_id for edge in self.bundle_edges}) != len(self.bundle_edges):
            raise CandidateChangeError("change set contains duplicate transition IDs")
        if len({node.revision.revision_id for node in self.atomic_nodes}) != len(self.atomic_nodes):
            raise CandidateChangeError("change set contains duplicate atomic revision IDs")
        object.__setattr__(self, "bundle_edges", tuple(sorted(self.bundle_edges, key=lambda edge: edge.transition.transition_id)))
        object.__setattr__(self, "atomic_nodes", tuple(sorted(self.atomic_nodes, key=lambda node: node.revision.revision_id)))

    @property
    def atomic_by_id(self) -> Mapping[str, AtomicChangeRecord]:
        return MappingProxyType({item.revision.revision_id: item for item in self.atomic_nodes})

    def replay_status_for_parent(
        self,
        revision_id: str,
        *,
        parent_revision_ids: Sequence[str] = (),
        parent_contract_hash: str | None = None,
        parent_source_hash: str | None = None,
        parent_config_hash: str | None = None,
    ) -> str:
        """Check cross-parent applicability with no score/rank access."""

        node = self.atomic_by_id.get(revision_id)
        if node is None:
            raise CandidateChangeError(f"unknown atomic revision {revision_id!r}")
        if node.replay_status in {STATUS_REJECTED_CONTRACT, STATUS_NON_REPLAYABLE}:
            return node.replay_status
        contract = self.task_contract_hash if parent_contract_hash is None else _sha256(parent_contract_hash, name="parent_contract_hash")
        if node.revision.cross_parent.same_contract_only and contract != node.revision.task_contract_hash:
            return STATUS_REJECTED_CONTRACT
        known = {key: item.revision for key, item in self.atomic_by_id.items()}
        if not _preconditions_hold(
            node.revision,
            parent_revision_ids=tuple(parent_revision_ids),
            parent_source_hash=parent_source_hash,
            parent_config_hash=parent_config_hash,
            known_revisions=known,
        ):
            return STATUS_DEPENDENCY_LOCKED
        return STATUS_REPLAYABLE

    def public_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": CANDIDATE_CHANGE_SCHEMA_VERSION,
            "task_contract_hash": self.task_contract_hash,
            "address_schema": {"allowed_prefixes": list(ADDRESS_PREFIXES)},
            "bundle_edges": [item.to_dict() for item in self.bundle_edges],
            "atomic_nodes": [item.to_dict() for item in self.atomic_nodes],
        }
        _assert_payload_safe(payload, name="public_payload")
        return payload

    def executable_registry_requirements(self) -> dict[str, Any]:
        """Return hash-only requirements for a separately stored patch registry.

        Executable source/config patch bodies deliberately do not enter the
        public payload.  This binding record lets a private registry prove that
        it covers the exact compiled atomics and source transitions without
        exposing those bodies to the consumer.
        """

        transition_by_id = {
            edge.transition.transition_id: edge.transition for edge in self.bundle_edges
        }
        bindings: list[dict[str, Any]] = []
        for node in self.atomic_nodes:
            revision = node.revision
            transition = transition_by_id.get(revision.source_transition_id)
            if transition is None:
                # Orphan atomics remain visible as rejected change set nodes, but no
                # executable registry entry can be bound to a missing edge.
                continue
            bindings.append(
                {
                    "revision_id": revision.revision_id,
                    "source_transition_id": revision.source_transition_id,
                    "source_transition_hash": transition.transition_hash,
                    "manifest_hash": revision.manifest_hash,
                    "source_patch_hash": revision.source_patch_hash,
                    "config_patch_hash": revision.config_patch_hash,
                }
            )
        payload = {
            "task_contract_hash": self.task_contract_hash,
            "change_set_hash": self.change_set_hash,
            "revision_bindings": sorted(
                bindings, key=lambda item: str(item["revision_id"])
            ),
        }
        _assert_payload_safe(payload, name="executable_registry_requirements")
        return payload

    @property
    def change_set_hash(self) -> str:
        return digest(self.public_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.public_payload()
        return {**payload, "change_set_hash": digest(payload)}

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def compile_candidate_change_set(
    *,
    task_contract_hash: str,
    transitions: Sequence[BundleTransition],
    atomic_revisions: Sequence[AtomicRevision] | None = None,
) -> CandidateChangeSet:
    """Compile observed bundles and score-free atomic replay records.

    ``atomic_revisions=None`` extracts all manifest edits.  An explicit empty
    sequence leaves valid observed transitions as ``bundle_only``.  In either
    case every input transition is retained as a bundle edge.
    """

    contract = _sha256(task_contract_hash, name="task_contract_hash")
    transitions = tuple(transitions)
    if len({item.transition_id for item in transitions}) != len(transitions):
        raise CandidateChangeError("duplicate observed transition IDs")
    derived = atomic_revisions is None
    atomics = (
        tuple(item for transition in transitions for item in extract_atomic_revisions(transition))
        if derived
        else tuple(atomic_revisions or ())
    )
    if len({item.revision_id for item in atomics}) != len(atomics):
        raise CandidateChangeError("duplicate atomic revision IDs")
    transition_by_id = {item.transition_id: item for item in transitions}
    atomics = _canonicalize_atomic_constraints(
        atomics,
        transitions_by_id=transition_by_id,
    )
    by_transition: dict[str, list[AtomicRevision]] = {item.transition_id: [] for item in transitions}
    orphan_nodes: list[AtomicChangeRecord] = []
    for atomic in atomics:
        if atomic.source_transition_id in by_transition:
            by_transition[atomic.source_transition_id].append(atomic)
        else:
            orphan_nodes.append(
                AtomicChangeRecord(atomic, STATUS_REJECTED_CONTRACT, ("unknown_source_transition",))
            )
    known = {item.revision_id: item for item in atomics}
    edges: list[BundleEdge] = []
    nodes: list[AtomicChangeRecord] = list(orphan_nodes)
    for transition in transitions:
        candidates = tuple(sorted(by_transition[transition.transition_id], key=lambda item: item.revision_id))
        if transition.task_contract_hash != contract:
            reasons = ("transition_contract_mismatch",)
            nodes.extend(AtomicChangeRecord(item, STATUS_REJECTED_CONTRACT, reasons) for item in candidates)
            edges.append(BundleEdge(transition, STATUS_REJECTED_CONTRACT, tuple(item.revision_id for item in candidates), reasons))
            continue
        if transition.manifest is None:
            # A source/config-hash observed transition without a validated
            # structured manifest remains visible but cannot be atomized.
            reason = transition.manifest_validation_error or "missing_manifest"
            nodes.extend(AtomicChangeRecord(item, STATUS_REJECTED_CONTRACT, (reason,)) for item in candidates)
            edge_status = STATUS_REJECTED_CONTRACT if transition.manifest_validation_error else STATUS_BUNDLE_ONLY
            edges.append(BundleEdge(transition, edge_status, (), (reason,)))
            continue
        if not candidates:
            edges.append(BundleEdge(transition, STATUS_BUNDLE_ONLY, (), ("atomization_not_requested",)))
            continue
        # Materialize a multi-edit bundle in dependency order.  An edit may
        # require a preceding edit in the *same* bundle; treating every edit
        # as if it were attached directly to M0 would incorrectly mark valid
        # multi-edit discoveries as dependency-locked.
        atom_nodes: list[AtomicChangeRecord] = []
        pending = {item.revision_id: item for item in candidates}
        available = set(transition.parent_revision_ids)
        while pending:
            progressed = False
            for revision_id in sorted(tuple(pending)):
                atomic = pending[revision_id]
                reasons = _contract_reasons(transition, atomic, task_contract_hash=contract)
                if reasons:
                    node = AtomicChangeRecord(atomic, STATUS_REJECTED_CONTRACT, reasons)
                    atom_nodes.append(node)
                    nodes.append(node)
                    pending.pop(revision_id)
                    progressed = True
                    continue
                if not set(atomic.dependencies).issubset(available):
                    continue
                source_status = _atomic_source_status(
                    atomic,
                    parent_revision_ids=tuple(sorted(available)),
                    known_revisions=known,
                )
                if source_status == STATUS_REPLAYABLE and not _preconditions_hold(
                    atomic,
                    parent_revision_ids=tuple(sorted(available)),
                    parent_source_hash=transition.parent_source_hash,
                    parent_config_hash=transition.parent_config_hash,
                    known_revisions=known,
                ):
                    source_status = STATUS_DEPENDENCY_LOCKED
                node = AtomicChangeRecord(atomic, source_status, ())
                atom_nodes.append(node)
                nodes.append(node)
                pending.pop(revision_id)
                if source_status == STATUS_REPLAYABLE:
                    available.add(revision_id)
                progressed = True
            if progressed:
                continue
            # A dependency cycle or an external predecessor not encoded in the
            # observed parent remains a valid bundle record but cannot replay
            # as an independent atomic edge.
            for revision_id in sorted(pending):
                atomic = pending[revision_id]
                node = AtomicChangeRecord(atomic, STATUS_DEPENDENCY_LOCKED, ("unresolved_dependency",))
                atom_nodes.append(node)
                nodes.append(node)
            pending.clear()
        reconstruction_reason = _reconstruction_reason(transition)
        statuses = {item.replay_status for item in atom_nodes}
        if any(status == STATUS_REJECTED_CONTRACT for status in statuses):
            edge_status = STATUS_REJECTED_CONTRACT
            reasons = tuple(sorted({reason for item in atom_nodes for reason in item.reasons}))
        elif reconstruction_reason == "not_materializable" or STATUS_NON_REPLAYABLE in statuses:
            edge_status = STATUS_NON_REPLAYABLE
            reasons = tuple(item for item in (reconstruction_reason,) if item is not None)
        elif reconstruction_reason == "dependency_qualified" or STATUS_DEPENDENCY_LOCKED in statuses:
            edge_status = STATUS_DEPENDENCY_LOCKED
            reasons = tuple(item for item in (reconstruction_reason,) if item is not None)
        elif reconstruction_reason is not None:
            edge_status = STATUS_REJECTED_CONTRACT
            reasons = (reconstruction_reason,)
        else:
            edge_status = STATUS_REPLAYABLE
            reasons = ()
        edges.append(
            BundleEdge(
                transition=transition,
                replay_status=edge_status,
                atomic_revision_ids=tuple(item.revision_id for item in candidates),
                reasons=reasons,
            )
        )
    return CandidateChangeSet(contract, tuple(edges), tuple(nodes))


def with_atomic_constraints(
    revision: AtomicRevision,
    *,
    dependencies: Sequence[str] | None = None,
    incompatibilities: Sequence[str] | None = None,
    cross_parent: CrossParentApplicability | None = None,
) -> AtomicRevision:
    """Attach explicit score-free dependency/incompatibility constraints."""

    return replace(
        revision,
        dependencies=revision.dependencies if dependencies is None else tuple(dependencies),
        incompatibilities=revision.incompatibilities if incompatibilities is None else tuple(incompatibilities),
        cross_parent=revision.cross_parent if cross_parent is None else cross_parent,
    )
