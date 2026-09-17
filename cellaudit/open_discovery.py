"""Trace contract for free-form, execution-grounded open discovery.

This module records a fixed initial entry and a hash-linked sequence of
requests and responses. A request supplies the protected task, the *full*
incumbent source and configuration, real Fold-3 diagnostics, and every prior
round. A response can contain an arbitrary new Python model source and a
multi-edit manifest; provider access, execution, and training remain in their
dedicated runtime modules.

The contract makes data, split, and evaluator identity immutable.  Architecture,
loss, optimizer, scheduler, and other training-model choices intentionally
remain open.  Candidate source is parsed only for Python syntax; it is never
compiled, imported, evaluated, or executed here.  Complete interactions remain
in the immutable trace archive; later discovery prompts receive bounded,
deterministic trajectory-memory summaries of earlier rounds.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import re
from types import MappingProxyType
from typing import Any

from .discovery_candidate import (
    ALLOWED_SEMANTIC_ROOTS,
    CANDIDATE_SCHEMA_VERSION,
    DiscoveryCandidateError,
    validate_candidate_source,
)
from .candidate_change_contracts import Precondition
from .candidate_materialization import (
    ExecutablePatchManifestProposal,
    PatchEditProposal,
    RevisionMaterializationError,
    TopLevelSymbolOperation,
    candidate_api_wrapper_symbols,
    compiler_support_symbols,
    json_merge_patch,
    normalize_patch_manifest,
)
from .schemas import canonical_json, digest


OPEN_DISCOVERY_SCHEMA_VERSION = "cellscientist_open_discovery_v1"
OPEN_DISCOVERY_RESPONSE_SCHEMA_VERSION = "cellscientist_open_discovery_response_v2"
# The response parser reserves bounded headroom for a compiler-requested
# manifest completion.  Initial discovery prompts are limited more tightly so
# a legal repair does not become unparsable solely because the original model
# used every manifest slot.
MAX_DISCOVERY_MANIFEST_EDITS = 12
MAX_INITIAL_DISCOVERY_MANIFEST_EDITS = 8
FOLD3_PARTITION = "feedback_fold_3"
UNEXECUTED = "unexecuted"

# Prompt-time trajectory memory is a bounded decision record derived from the
# raw discovery archive. These character limits remain deterministic across
# API providers.
PROMPT_HISTORY_SCHEMA_VERSION = "cellscientist_prompt_history_v1"
MAX_PROMPT_HISTORY_RECORD_CHARS = 3_200
MAX_PROMPT_HISTORY_HYPOTHESIS_CHARS = 280
MAX_PROMPT_HISTORY_SUMMARY_CHARS = 240
MAX_PROMPT_HISTORY_ADDRESS_CHARS = 96
MAX_PROMPT_HISTORY_ADDRESS_COUNT = 4
MAX_PROMPT_HISTORY_METRIC_NAME_CHARS = 64
MAX_PROMPT_HISTORY_METRIC_COUNT = 4
MAX_PROMPT_HISTORY_FAILURE_CHARS = 280
MAX_PROMPT_HISTORY_MODEL_CHARS = 96


def prompt_history_memory_specification() -> dict[str, Any]:
    """Return the bounded, source-free discovery-memory contract.

    The full interaction archive remains immutable on disk.  This record
    specifies only what may be copied back into a later discovery prompt, so
    its hash can be frozen independently of an LLM provider.
    """

    return {
        "schema_version": PROMPT_HISTORY_SCHEMA_VERSION,
        "policy": "bounded_provenance_v1",
        "raw_prompt_or_response_in_prompt_history": False,
        "candidate_source_in_prompt_history": False,
        "full_trace_archive_retained": True,
        "max_record_chars": MAX_PROMPT_HISTORY_RECORD_CHARS,
        "max_hypothesis_chars": MAX_PROMPT_HISTORY_HYPOTHESIS_CHARS,
        "max_summary_chars": MAX_PROMPT_HISTORY_SUMMARY_CHARS,
        "max_address_chars": MAX_PROMPT_HISTORY_ADDRESS_CHARS,
        "max_address_count": MAX_PROMPT_HISTORY_ADDRESS_COUNT,
        "max_metric_name_chars": MAX_PROMPT_HISTORY_METRIC_NAME_CHARS,
        "max_metric_count": MAX_PROMPT_HISTORY_METRIC_COUNT,
        "max_failure_chars": MAX_PROMPT_HISTORY_FAILURE_CHARS,
        "max_model_chars": MAX_PROMPT_HISTORY_MODEL_CHARS,
    }


def prompt_history_memory_contract_hash() -> str:
    """Stable digest of the source-free trajectory-memory contract."""

    return digest(prompt_history_memory_specification())


class OpenDiscoveryError(ValueError):
    """Raised when a discovery trace would violate its protected contract."""


# These names are intentionally checked in patches and edit addresses rather
# than used as an allowed-model grammar.  The model-design space remains open;
# only task/evaluation semantics are unavailable for mutation.
_PROTECTED_FIELD_NAMES = frozenset(
    {
        "data",
        "dataset",
        "datapath",
        "datapaths",
        "sourcepath",
        "sourcepaths",
        "rawdata",
        "rawsource",
        "split",
        "splits",
        "fold",
        "folds",
        "fitfold",
        "fitfolds",
        "feedbackfold",
        "validationfold",
        "testfold",
        "holdout",
        "holdoutfold",
        "endpoint",
        "evaluator",
        "evaluation",
        "metric",
        "metrics",
        "target",
        "targets",
        "condition",
        "conditions",
        "input",
        "inputs",
        "taskcontract",
        "protectedtask",
    }
)
_FORBIDDEN_DIAGNOSTIC_TOKENS = frozenset(
    {"fold4", "fold5", "endpoint", "test", "holdout", "outer"}
)
_DIAGNOSTIC_STATUSES = frozenset({"completed", "failed", "timed_out"})
_FAILURE_SUMMARY_FIELDS = frozenset({"failure_type", "failure_message"})
_MAX_FAILURE_MESSAGE_CHARS = 1_200
_SECRET_LIKE_FAILURE_TOKEN = re.compile(
    r"(?:\b(?:api[_-]?key|access[_-]?token|authorization|bearer|password|passwd|secret|credential)\b"
    r"|\b(?:sk|rk|pk)-[A-Za-z0-9_-]{4,}\b|\bghp_[A-Za-z0-9_-]{4,}\b|\bAIza[A-Za-z0-9_-]{4,}\b)",
    flags=re.IGNORECASE,
)
_MAX_SOURCE_CHARS = 2_000_000
_MAX_RESPONSE_CHARS = 4_000_000
_MAX_PROMPT_CHARS = 16_000_000
_REQUIRED_CANDIDATE_EXPORTS = frozenset(
    {"candidate_metadata", "build_model", "compute_loss", "build_optimizer"}
)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _text(value: Any, *, name: str, allow_empty: bool = False, limit: int = 100_000) -> str:
    if not isinstance(value, str):
        raise OpenDiscoveryError(f"{name} must be a string")
    if "\x00" in value:
        raise OpenDiscoveryError(f"{name} must not contain NUL bytes")
    if len(value) > limit:
        raise OpenDiscoveryError(f"{name} exceeds the {limit}-character contract limit")
    if not allow_empty and not value.strip():
        raise OpenDiscoveryError(f"{name} must not be empty")
    return value


def _identifier(value: Any, *, name: str) -> str:
    result = _text(value, name=name, limit=240)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", result):
        raise OpenDiscoveryError(
            f"{name} may contain only letters, digits, '_', '.', ':', or '-'"
        )
    return result


def _hash(value: Any, *, name: str) -> str:
    result = _text(value, name=name, limit=64)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise OpenDiscoveryError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise OpenDiscoveryError(f"{name} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OpenDiscoveryError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise OpenDiscoveryError(f"{name} must be finite and non-negative")
    return result


def _json_copy(value: Any, *, name: str) -> Any:
    """Make a validated, ordinary JSON-compatible deep copy.

    The contract stores no mutable caller-owned mapping.  Finite numeric values
    are accepted, but arbitrary objects, bytes, NumPy arrays, and raw tensors
    are deliberately excluded from requests and diagnostics.
    """

    def copy(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise OpenDiscoveryError(f"{path} contains a non-finite float")
            return float(item)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, member in item.items():
                if not isinstance(key, str):
                    raise OpenDiscoveryError(f"{path} mapping keys must be strings")
                result[key] = copy(member, f"{path}.{key}")
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [copy(member, f"{path}[{index}]") for index, member in enumerate(item)]
        raise OpenDiscoveryError(f"{path} must be JSON-compatible")

    return copy(value, name)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(member) for key, member in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(member) for member in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_thaw(member) for member in value]
    return value


def _json_mapping(value: Any, *, name: str, allow_empty: bool = False) -> Mapping[str, Any]:
    copied = _json_copy(value, name=name)
    if not isinstance(copied, dict):
        raise OpenDiscoveryError(f"{name} must be a JSON object")
    if not allow_empty and not copied:
        raise OpenDiscoveryError(f"{name} must not be empty")
    return _freeze(copied)


def _fold3_partition(value: Any, *, name: str) -> str:
    raw = _text(value, name=name, limit=80)
    normalised = _normalise_name(raw)
    if normalised not in {"fold3", "feedbackfold3"}:
        raise OpenDiscoveryError(f"{name} must identify Fold 3, not {raw!r}")
    return FOLD3_PARTITION


def _assert_no_protected_mutation(value: Any, *, name: str) -> None:
    """Reject any configuration patch that reaches a protected task field."""

    if isinstance(value, Mapping):
        for key, member in value.items():
            if not isinstance(key, str):
                raise OpenDiscoveryError(f"{name} keys must be strings")
            normalised = _normalise_name(key)
            if normalised in _PROTECTED_FIELD_NAMES:
                raise OpenDiscoveryError(
                    f"{name}.{key} attempts to modify a protected data/split/evaluator field"
                )
            _assert_no_protected_mutation(member, name=f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _assert_no_protected_mutation(member, name=f"{name}[{index}]")


def _sanitized_failure_summary(
    value: Any,
    *,
    status: str,
) -> Mapping[str, str] | None:
    """Validate a compact, path- and secret-free execution failure summary.

    The execution layer is responsible for redaction.  This trace boundary
    still rejects common path, URL, multiline traceback, and credential
    patterns so raw process errors cannot quietly reach the discovery model or
    immutable trajectory archive.
    """

    if status == "completed":
        if value is not None:
            raise OpenDiscoveryError(
                "completed Fold-3 diagnostics must not carry failure_summary"
            )
        return None
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OpenDiscoveryError("failure_summary must be an object with string keys")
    missing = _FAILURE_SUMMARY_FIELDS.difference(value)
    unknown = set(value).difference(_FAILURE_SUMMARY_FIELDS)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields {sorted(missing)}")
        if unknown:
            detail.append(f"unknown fields {sorted(unknown)}")
        raise OpenDiscoveryError("failure_summary has " + "; ".join(detail))
    failure_type = _text(value["failure_type"], name="failure_summary.failure_type", limit=160)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", failure_type):
        raise OpenDiscoveryError(
            "failure_summary.failure_type must be a compact exception/category identifier"
        )
    failure_message = _text(
        value["failure_message"],
        name="failure_summary.failure_message",
        limit=_MAX_FAILURE_MESSAGE_CHARS,
    )
    if any(character in failure_message for character in ("\n", "\r", "\t")):
        raise OpenDiscoveryError(
            "failure_summary.failure_message must be a single-line sanitized summary"
        )
    if "/" in failure_message or "\\" in failure_message:
        raise OpenDiscoveryError(
            "failure_summary.failure_message must not contain filesystem paths or URLs"
        )
    if _SECRET_LIKE_FAILURE_TOKEN.search(failure_message):
        raise OpenDiscoveryError(
            "failure_summary.failure_message must not contain secret-like tokens"
        )
    return _freeze(
        {
            "failure_type": failure_type,
            "failure_message": failure_message,
        }
    )


def _semantic_address(value: Any, *, name: str) -> str:
    """Validate an open semantic address under one frozen top-level root.

    The suffix deliberately remains unconstrained: discovery may introduce a
    new perturbation-modeling concept under a legal root, but cannot relabel a
    data source, split, or evaluator as a model component.
    """

    address = _text(value, name=name, limit=500)
    if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", address):
        raise OpenDiscoveryError(f"{name} contains invalid semantic-address characters")
    root, separator, suffix = address.partition(".")
    if not separator or not suffix or root not in ALLOWED_SEMANTIC_ROOTS:
        raise OpenDiscoveryError(
            f"{name} must begin with one of {sorted(ALLOWED_SEMANTIC_ROOTS)} followed by '.'; "
            "protected task roots are not addressable"
        )
    return address


def _assert_python_source(value: Any, *, name: str) -> str:
    source = _text(value, name=name, limit=_MAX_SOURCE_CHARS)
    try:
        # This is a static AST-only candidate-interface check.  It neither
        # imports nor executes source, but does reject filesystem/network/
        # process access before the trace accepts a candidate declaration.
        validate_candidate_source(source)
    except DiscoveryCandidateError as exc:
        raise OpenDiscoveryError(f"{name} violates the protected candidate API: {exc}") from exc
    return source


def _module_callable_exports(source: str) -> frozenset[str]:
    """Return statically declared module-level callable names without executing."""

    # ``CandidateState`` has already parsed the source before this helper is
    # reached.  Re-parsing remains safe and keeps this metadata validation
    # independent from any source import or runtime evaluation.
    tree = ast.parse(source, filename="<open-discovery-candidate>", mode="exec")
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )


def _validate_candidate_metadata(value: Any, *, source: str) -> Mapping[str, Any]:
    """Validate the declared source-to-semantic-component provenance mapping.

    The declaration is intentionally validated without calling the source's
    ``candidate_metadata()`` function.  A later isolated candidate validator
    cross-checks the executable return value; this trace layer only records a
    complete, static declaration and ensures referenced symbols exist in the
    module AST.
    """

    metadata = _json_mapping(value, name="candidate_metadata")
    required = {
        "schema_version",
        "candidate_name",
        "semantic_components",
        "component_symbols",
        "change_summary",
    }
    missing = required.difference(metadata)
    if missing:
        raise OpenDiscoveryError(
            f"candidate_metadata missing fields: {sorted(missing)}"
        )
    if metadata["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise OpenDiscoveryError("candidate_metadata schema_version is unsupported")
    _identifier(metadata["candidate_name"], name="candidate_metadata.candidate_name")
    components = metadata["semantic_components"]
    if not isinstance(components, (list, tuple)) or not components:
        raise OpenDiscoveryError("candidate_metadata.semantic_components must be a non-empty array")
    checked_components = tuple(
        _semantic_address(component, name="candidate_metadata.semantic_components[]")
        for component in components
    )
    if len(set(checked_components)) != len(checked_components):
        raise OpenDiscoveryError("candidate_metadata.semantic_components must be unique")

    component_symbols = metadata["component_symbols"]
    if not isinstance(component_symbols, Mapping):
        raise OpenDiscoveryError("candidate_metadata.component_symbols must be an object")
    if set(component_symbols) != set(checked_components):
        raise OpenDiscoveryError(
            "candidate_metadata.component_symbols keys must exactly match semantic_components"
        )
    exports = _module_callable_exports(source)
    missing_required_exports = _REQUIRED_CANDIDATE_EXPORTS.difference(exports)
    if missing_required_exports:
        raise OpenDiscoveryError(
            "candidate source is missing required callable exports: "
            f"{sorted(missing_required_exports)}"
        )
    for address in checked_components:
        symbols = component_symbols[address]
        if not isinstance(symbols, (list, tuple)) or not symbols:
            raise OpenDiscoveryError(
                f"candidate_metadata.component_symbols[{address!r}] must be a non-empty array"
            )
        checked_symbols: list[str] = []
        for symbol in symbols:
            symbol_name = _text(
                symbol,
                name=f"candidate_metadata.component_symbols[{address!r}][]",
                limit=240,
            )
            if not symbol_name.isidentifier():
                raise OpenDiscoveryError(
                    f"candidate_metadata.component_symbols[{address!r}] values must be Python callable names"
                )
            checked_symbols.append(symbol_name)
        checked_symbols = tuple(checked_symbols)
        if len(set(checked_symbols)) != len(checked_symbols):
            raise OpenDiscoveryError(
                f"candidate_metadata.component_symbols[{address!r}] must not contain duplicates"
            )
        missing_exports = set(checked_symbols).difference(exports)
        if missing_exports:
            raise OpenDiscoveryError(
                f"candidate_metadata.component_symbols[{address!r}] names non-exported callables: "
                f"{sorted(missing_exports)}"
            )
    _text(metadata["change_summary"], name="candidate_metadata.change_summary")
    return metadata


def _merge_patch(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply an RFC-7396-style JSON merge patch without touching source code."""

    result = _json_copy(base, name="training_config")
    assert isinstance(result, dict)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_patch(result[key], value)
        else:
            result[key] = _json_copy(value, name=f"training_config_patch.{key}")
    return _freeze(result)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


_METADATA_SUPPORT_ADDRESS_ALIASES = frozenset({"metadata", "candidate_metadata"})
_METADATA_SUPPORT_CANONICAL_ADDRESS = "reliability.candidate_metadata"


def _canonicalize_metadata_support_address(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Canonicalize one unambiguous API-support address without changing science.

    Some otherwise valid OpenAI-compatible models emit a separate pure
    ``candidate_metadata`` support edit at the intuitive but non-semantic
    address ``metadata``. That wrapper has no candidate-level scientific
    meaning: the compiler classifies it as support and derives its real
    dependencies from the immutable source/metadata references.  Mapping this
    one syntactically unambiguous form to a registered namespace avoids an LLM
    repair call that can neither improve the source nor alter a scientific
    action.  The raw reply remains archived and the transformation is recorded
    in ``OpenDiscoveryRoundTrace.response_normalizations``.

    No edit carrying another symbol or a configuration effect is normalized.
    Those proposals remain subject to the ordinary schema and source checks.
    """

    manifest = payload.get("change_manifest")
    if not isinstance(manifest, Mapping):
        return payload, ()
    edits = manifest.get("edits")
    if not isinstance(edits, list):
        return payload, ()
    rewritten: list[Any] = []
    changed = False
    for raw_edit in edits:
        if not isinstance(raw_edit, Mapping):
            rewritten.append(raw_edit)
            continue
        address = raw_edit.get("address")
        symbol_patch = raw_edit.get("symbol_patch")
        parent_symbols = (
            symbol_patch.get("parent_symbols")
            if isinstance(symbol_patch, Mapping)
            else None
        )
        child_symbols = (
            symbol_patch.get("child_symbols")
            if isinstance(symbol_patch, Mapping)
            else None
        )
        pure_metadata_support = (
            isinstance(address, str)
            and address in _METADATA_SUPPORT_ADDRESS_ALIASES
            and isinstance(parent_symbols, list)
            and isinstance(child_symbols, list)
            and set(parent_symbols).union(child_symbols) == {"candidate_metadata"}
            and raw_edit.get("config_merge_patch") == {}
        )
        if not pure_metadata_support:
            rewritten.append(raw_edit)
            continue
        replacement = dict(raw_edit)
        replacement["address"] = _METADATA_SUPPORT_CANONICAL_ADDRESS
        rewritten.append(replacement)
        changed = True
    if not changed:
        return payload, ()
    normalized_manifest = dict(manifest)
    normalized_manifest["edits"] = rewritten
    normalized_payload = dict(payload)
    normalized_payload["change_manifest"] = normalized_manifest
    return normalized_payload, ("metadata_support_address_normalized",)


def _decode_open_discovery_response_payload(
    raw_response: str,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """Decode a raw reply and return compiler-owned non-scientific normalizations."""

    text = _text(
        raw_response,
        name="raw_response",
        allow_empty=True,
        limit=_MAX_RESPONSE_CHARS,
    )
    stripped = _strip_json_fence(text)
    normalizations: list[str] = []
    if stripped != text.strip():
        normalizations.append("json_fence_stripped")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OpenDiscoveryError("response was not a standalone JSON object") from exc
    if not isinstance(payload, Mapping):
        raise OpenDiscoveryError("response must be a JSON object")
    canonical_payload, support_normalizations = _canonicalize_metadata_support_address(
        payload
    )
    return canonical_payload, (*normalizations, *support_normalizations)


def _contains_raw_interaction_body(value: Any) -> bool:
    """Prevent archived raw interaction or source bodies from re-entering a prompt."""

    if isinstance(value, Mapping):
        if (
            "raw_prompt" in value
            or "raw_response" in value
            or "candidate_source" in value
        ):
            return True
        return any(_contains_raw_interaction_body(member) for member in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_interaction_body(member) for member in value)
    return False


def _bounded_history_text(value: str, *, name: str, limit: int) -> str:
    """Return a deterministic display-safe prefix with a content fingerprint.

    The suffix makes two long summaries that share a prefix distinguishable to
    the model while ensuring that trajectory memory has a fixed footprint.
    The full text remains in the raw trace, never in this prompt-side record.
    """

    if limit < 20:
        raise RuntimeError("prompt-history text limit must reserve a hash suffix")
    text = _text(value, name=name, allow_empty=True)
    if len(text) <= limit:
        return text
    suffix = "...#" + digest(text)[:12]
    return text[: limit - len(suffix)] + suffix


def _bounded_history_value(value: Any, *, name: str, limit: int) -> str:
    """Render a JSON-compatible value within the deterministic memory envelope."""

    if isinstance(value, str):
        return _bounded_history_text(value, name=name, limit=limit)
    return _bounded_history_text(
        canonical_json(_json_copy(value, name=name)),
        name=name,
        limit=limit,
    )


def _history_hash(value: Any) -> str:
    """Return a stable fingerprint for prompt-omitted archival detail."""

    return digest(_json_copy(value, name="prompt_history_hash_input"))


def _bounded_history_labels(
    values: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Summarize a variable-size label collection without losing its identity."""

    normalized = tuple(
        _text(value, name=f"{label}[]", limit=240) for value in values
    )
    return {
        f"{label}_count": len(normalized),
        f"{label}_hash": digest(list(normalized)),
        label: [
            _bounded_history_text(
                value,
                name=f"{label}[]",
                limit=MAX_PROMPT_HISTORY_ADDRESS_CHARS,
            )
            for value in normalized[:MAX_PROMPT_HISTORY_ADDRESS_COUNT]
        ],
    }


@dataclass(frozen=True)
class ProtectedDiscoveryTask:
    """Immutable data, split, evaluator, and response-semantics identity."""

    task_id: str
    task_semantics: Mapping[str, Any]
    data: Mapping[str, Any]
    splits: Mapping[str, Any]
    evaluator: Mapping[str, Any]
    feedback_partition: str = FOLD3_PARTITION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier(self.task_id, name="task_id"))
        object.__setattr__(
            self,
            "task_semantics",
            _json_mapping(self.task_semantics, name="task_semantics"),
        )
        object.__setattr__(self, "data", _json_mapping(self.data, name="data"))
        object.__setattr__(self, "splits", _json_mapping(self.splits, name="splits"))
        object.__setattr__(self, "evaluator", _json_mapping(self.evaluator, name="evaluator"))
        object.__setattr__(
            self,
            "feedback_partition",
            _fold3_partition(self.feedback_partition, name="feedback_partition"),
        )

    @property
    def data_hash(self) -> str:
        return digest(_thaw(self.data))

    @property
    def split_hash(self) -> str:
        return digest(_thaw(self.splits))

    @property
    def evaluator_hash(self) -> str:
        return digest(_thaw(self.evaluator))

    def _core_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_semantics": _thaw(self.task_semantics),
            "data": _thaw(self.data),
            "splits": _thaw(self.splits),
            "evaluator": _thaw(self.evaluator),
            "feedback_partition": self.feedback_partition,
        }

    @property
    def protected_task_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        if include_hash:
            result.update(
                {
                    "data_hash": self.data_hash,
                    "split_hash": self.split_hash,
                    "evaluator_hash": self.evaluator_hash,
                    "protected_task_hash": self.protected_task_hash,
                }
            )
        return result


@dataclass(frozen=True)
class CandidateState:
    """One full source/configuration state; source is opaque and unexecuted."""

    candidate_source: str
    candidate_metadata: Mapping[str, Any]
    training_config: Mapping[str, Any]
    parent_candidate_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_source",
            _assert_python_source(self.candidate_source, name="candidate_source"),
        )
        object.__setattr__(
            self,
            "candidate_metadata",
            _validate_candidate_metadata(
                self.candidate_metadata,
                source=self.candidate_source,
            ),
        )
        config = _json_mapping(self.training_config, name="training_config", allow_empty=True)
        _assert_no_protected_mutation(config, name="training_config")
        object.__setattr__(self, "training_config", config)
        if self.parent_candidate_hash is not None:
            object.__setattr__(
                self,
                "parent_candidate_hash",
                _hash(self.parent_candidate_hash, name="parent_candidate_hash"),
            )

    @property
    def source_hash(self) -> str:
        return digest(self.candidate_source)

    @property
    def training_config_hash(self) -> str:
        return digest(_thaw(self.training_config))

    @property
    def candidate_metadata_hash(self) -> str:
        return digest(_thaw(self.candidate_metadata))

    @property
    def candidate_hash(self) -> str:
        return digest(
            {
                "source_hash": self.source_hash,
                "candidate_metadata_hash": self.candidate_metadata_hash,
                "training_config_hash": self.training_config_hash,
                "parent_candidate_hash": self.parent_candidate_hash,
            }
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_source": self.candidate_source,
            "source_hash": self.source_hash,
            "candidate_metadata": _thaw(self.candidate_metadata),
            "candidate_metadata_hash": self.candidate_metadata_hash,
            "training_config": _thaw(self.training_config),
            "training_config_hash": self.training_config_hash,
            "parent_candidate_hash": self.parent_candidate_hash,
        }
        if include_hash:
            result["candidate_hash"] = self.candidate_hash
        return result


def candidate_history_summary(candidate: CandidateState) -> dict[str, Any]:
    """Return the bounded source-free provenance record permitted in a later prompt.

    The next request already carries the complete *current* candidate source.
    Repeating every historical source inflates later prompts, creates an
    unnecessary prompt-sensitive context window, and is not needed to replay
    the immutable archive.  This summary retains bounded lineage, hashes, and
    semantic coverage, while the full candidate remains only in the
    append-only local trace archive.
    """

    if not isinstance(candidate, CandidateState):
        raise OpenDiscoveryError("candidate history summary requires a CandidateState")
    metadata = candidate.candidate_metadata
    components = metadata.get("semantic_components")
    if not isinstance(components, tuple):  # Defensive after CandidateState validation.
        raise OpenDiscoveryError("candidate metadata semantic components are invalid")
    component_summary = _bounded_history_labels(
        components,
        label="semantic_components",
    )
    return {
        "candidate_hash": candidate.candidate_hash,
        "parent_candidate_hash": candidate.parent_candidate_hash,
        "candidate_name": _bounded_history_text(
            str(metadata["candidate_name"]),
            name="candidate_metadata.candidate_name",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        **component_summary,
        "change_summary": _bounded_history_text(
            str(metadata["change_summary"]),
            name="candidate_metadata.change_summary",
            limit=MAX_PROMPT_HISTORY_SUMMARY_CHARS,
        ),
    }


def _patchable_symbol_set(candidate: CandidateState) -> frozenset[str]:
    """Return callable names that a manifest may legally patch.

    Component symbols cover semantic model pieces.  The protected Candidate
    API wrappers must also be patchable because a source transition can update
    metadata, model construction, objective, or optimizer glue even when the
    underlying component name remains stable.
    """

    # The AST is the authoritative before/after source inventory used by the
    # executable compiler.  Restricting this check to metadata declarations
    # made a real parent wrapper look like a new child symbol when older h0
    # metadata did not enumerate every helper class.
    exports = _module_callable_exports(candidate.candidate_source)
    return frozenset(exports)


@dataclass(frozen=True)
class FixedInitialEntry:
    """The common, reproducible h0 entry that every discovery run retains."""

    protected_task: ProtectedDiscoveryTask
    initial_candidate: CandidateState
    initial_prompt: str
    exploration_budget: int
    entry_id: str = "h0"

    def __post_init__(self) -> None:
        if not isinstance(self.protected_task, ProtectedDiscoveryTask):
            raise OpenDiscoveryError("initial entry requires a ProtectedDiscoveryTask")
        if not isinstance(self.initial_candidate, CandidateState):
            raise OpenDiscoveryError("initial entry requires a CandidateState")
        if self.initial_candidate.parent_candidate_hash is not None:
            raise OpenDiscoveryError("the fixed initial candidate must not have a parent")
        object.__setattr__(self, "initial_prompt", _text(self.initial_prompt, name="initial_prompt"))
        object.__setattr__(
            self,
            "exploration_budget",
            _positive_int(self.exploration_budget, name="exploration_budget"),
        )
        object.__setattr__(self, "entry_id", _identifier(self.entry_id, name="entry_id"))

    @property
    def initial_prompt_hash(self) -> str:
        return digest(self.initial_prompt)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "protected_task_hash": self.protected_task.protected_task_hash,
            "initial_candidate": self.initial_candidate.to_dict(),
            "initial_prompt": self.initial_prompt,
            "initial_prompt_hash": self.initial_prompt_hash,
            "exploration_budget": self.exploration_budget,
        }

    @property
    def initial_entry_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        # Include the full protected task once at the fixed root.  Requests also
        # repeat it explicitly so an individual prompt stays self-contained.
        result["protected_task"] = self.protected_task.to_dict()
        if include_hash:
            result["initial_entry_hash"] = self.initial_entry_hash
        return result


@dataclass(frozen=True)
class Fold3Diagnostics:
    """A Fold-3 scalar record with an optional sanitized failed-run summary."""

    partition: str
    candidate_hash: str
    protected_task_hash: str
    split_hash: str
    evaluator_hash: str
    execution_id: str
    metrics: Mapping[str, float | None]
    runtime_seconds: float
    status: str = "completed"
    failure_summary: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition", _fold3_partition(self.partition, name="partition"))
        for name in (
            "candidate_hash",
            "protected_task_hash",
            "split_hash",
            "evaluator_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        object.__setattr__(self, "execution_id", _identifier(self.execution_id, name="execution_id"))
        if self.status not in _DIAGNOSTIC_STATUSES:
            raise OpenDiscoveryError(
                f"diagnostic status must be one of {sorted(_DIAGNOSTIC_STATUSES)}"
            )
        failure_summary = _sanitized_failure_summary(
            self.failure_summary,
            status=self.status,
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            _finite_nonnegative(self.runtime_seconds, name="runtime_seconds"),
        )
        raw_metrics = _json_mapping(self.metrics, name="metrics", allow_empty=True)
        checked: dict[str, float | None] = {}
        for key, value in raw_metrics.items():
            metric_name = _identifier(key, name="metrics key")
            token = _normalise_name(metric_name)
            if any(forbidden in token for forbidden in _FORBIDDEN_DIAGNOSTIC_TOKENS):
                raise OpenDiscoveryError(
                    f"metrics.{metric_name} attempts to expose Fold-4/Fold-5 or endpoint data"
                )
            if value is None:
                checked[metric_name] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                checked[metric_name] = float(value)
            else:
                raise OpenDiscoveryError(f"metrics.{metric_name} must be a finite scalar or null")
        if self.status == "completed" and not checked:
            raise OpenDiscoveryError("completed Fold-3 diagnostics must contain at least one scalar metric")
        object.__setattr__(self, "metrics", _freeze(dict(sorted(checked.items()))))
        object.__setattr__(
            self,
            "failure_summary",
            failure_summary,
        )

    def _core_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "candidate_hash": self.candidate_hash,
            "protected_task_hash": self.protected_task_hash,
            "split_hash": self.split_hash,
            "evaluator_hash": self.evaluator_hash,
            "execution_id": self.execution_id,
            "metrics": _thaw(self.metrics),
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
            "failure_summary": (
                None if self.failure_summary is None else _thaw(self.failure_summary)
            ),
        }

    @property
    def diagnostics_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        if include_hash:
            result["diagnostics_hash"] = self.diagnostics_hash
        return result


def fold3_history_summary(diagnostics: Fold3Diagnostics) -> dict[str, Any]:
    """Return the bounded Fold-3 result memory allowed into a later prompt.

    The complete diagnostics object is retained in the trace.  Historical
    prompt context needs only status, a few deterministic headline metrics,
    bounded failure evidence, and fingerprints for the omitted detail.
    """

    if not isinstance(diagnostics, Fold3Diagnostics):
        raise OpenDiscoveryError("Fold-3 history summary requires Fold3Diagnostics")
    metrics = _thaw(diagnostics.metrics)
    ordered_names = sorted(
        metrics,
        key=lambda name: (_normalise_name(name) != "globalpcc", name),
    )
    metric_summary = [
        {
            "name": _bounded_history_text(
                metric_name,
                name="Fold-3 metric name",
                limit=MAX_PROMPT_HISTORY_METRIC_NAME_CHARS,
            ),
            "value": metrics[metric_name],
        }
        for metric_name in ordered_names[:MAX_PROMPT_HISTORY_METRIC_COUNT]
    ]
    failure = None
    if diagnostics.failure_summary is not None:
        full_failure = _thaw(diagnostics.failure_summary)
        failure = {
            "failure_hash": digest(full_failure),
            "failure_type": _bounded_history_text(
                str(full_failure["failure_type"]),
                name="failure_summary.failure_type",
                limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
            ),
            "failure_reason": _bounded_history_text(
                str(full_failure["failure_message"]),
                name="failure_summary.failure_message",
                limit=MAX_PROMPT_HISTORY_FAILURE_CHARS,
            ),
        }
    return {
        "diagnostics_hash": diagnostics.diagnostics_hash,
        "status": diagnostics.status,
        "metrics_hash": digest(metrics),
        "metric_count": len(metrics),
        "metrics": metric_summary,
        "runtime_seconds": diagnostics.runtime_seconds,
        "failure": failure,
    }


@dataclass(frozen=True)
class SymbolPatch:
    """Per-edit declaration of parent and child callable-symbol changes.

    This is a provenance declaration, not an executable diff.  The source is
    still supplied in full; later compilation can use these named symbols to
    recover a candidate atomic edit without guessing from a prose rationale.
    """

    operation: str
    parent_symbols: tuple[str, ...]
    child_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.operation not in {"add", "replace", "remove", "modify"}:
            raise OpenDiscoveryError(
                "symbol_patch.operation must be one of ['add', 'modify', 'remove', 'replace']"
            )
        for name in ("parent_symbols", "child_symbols"):
            raw = getattr(self, name)
            if not isinstance(raw, (list, tuple)):
                raise OpenDiscoveryError(f"symbol_patch.{name} must be an array")
            values: list[str] = []
            for symbol in raw:
                value = _text(symbol, name=f"symbol_patch.{name}[]", limit=240)
                if not value.isidentifier():
                    raise OpenDiscoveryError(
                        f"symbol_patch.{name} values must be Python callable names"
                    )
                values.append(value)
            if len(set(values)) != len(values):
                raise OpenDiscoveryError(f"symbol_patch.{name} must not contain duplicates")
            object.__setattr__(self, name, tuple(values))
        if self.operation == "add" and (self.parent_symbols or not self.child_symbols):
            raise OpenDiscoveryError("add symbol patches require no parent symbols and non-empty child symbols")
        if self.operation == "remove" and (not self.parent_symbols or self.child_symbols):
            raise OpenDiscoveryError("remove symbol patches require parent symbols and no child symbols")
        if self.operation == "replace" and (
            not self.parent_symbols or not self.child_symbols
        ):
            raise OpenDiscoveryError("replace symbol patches require parent and child symbols")
        if self.operation == "modify" and bool(self.parent_symbols) != bool(self.child_symbols):
            raise OpenDiscoveryError(
                "modify symbol patches require both parent/child symbols or neither for a config-only edit"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "parent_symbols": list(self.parent_symbols),
            "child_symbols": list(self.child_symbols),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SymbolPatch":
        if not isinstance(payload, Mapping):
            raise OpenDiscoveryError("symbol_patch must be an object")
        # The operation is compiler-normalized from the before/after symbol
        # sets.  Asking a discovery model to repeat this deterministic fact
        # created avoidable contract failures without adding scientific
        # information.  ``operation`` remains accepted for legacy responses,
        # but it is never trusted over the symbol sets.
        required = {"parent_symbols", "child_symbols"}
        allowed = {*required, "operation"}
        missing = required.difference(payload)
        unknown = set(payload).difference(allowed)
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing fields {sorted(missing)}")
            if unknown:
                detail.append(f"unknown fields {sorted(unknown)}")
            raise OpenDiscoveryError("symbol_patch has " + "; ".join(detail))
        parent_symbols = (
            tuple(payload["parent_symbols"])
            if isinstance(payload["parent_symbols"], (list, tuple))
            else payload["parent_symbols"]
        )
        child_symbols = (
            tuple(payload["child_symbols"])
            if isinstance(payload["child_symbols"], (list, tuple))
            else payload["child_symbols"]
        )
        if not isinstance(parent_symbols, (list, tuple)) or not isinstance(
            child_symbols, (list, tuple)
        ):
            raise OpenDiscoveryError(
                "symbol_patch parent_symbols and child_symbols must be arrays"
            )
        if not parent_symbols and child_symbols:
            operation = "add"
        elif parent_symbols and not child_symbols:
            operation = "remove"
        elif parent_symbols and child_symbols:
            operation = "modify" if tuple(parent_symbols) == tuple(child_symbols) else "replace"
        else:
            operation = "modify"
        return cls(
            operation=operation,
            parent_symbols=parent_symbols,
            child_symbols=child_symbols,
        )


@dataclass(frozen=True)
class ChangeEdit:
    """One unrestricted model/training edit, except at protected addresses."""

    edit_id: str
    address: str
    kind: str
    description: str
    symbol_patch: SymbolPatch
    dependencies: tuple[str, ...] = ()
    incompatibilities: tuple[str, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    config_merge_patch: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edit_id", _identifier(self.edit_id, name="edit_id"))
        object.__setattr__(self, "address", _semantic_address(self.address, name="address"))
        kind = _identifier(self.kind, name="kind")
        if _normalise_name(kind) in _PROTECTED_FIELD_NAMES:
            raise OpenDiscoveryError("change kind may not modify protected task semantics")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "description", _text(self.description, name="description"))
        if not isinstance(self.symbol_patch, SymbolPatch):
            raise OpenDiscoveryError("change edits require a SymbolPatch")
        if not isinstance(self.dependencies, (list, tuple)):
            raise OpenDiscoveryError("dependencies must be an array")
        dependencies = tuple(
            _identifier(value, name="dependencies[]") for value in self.dependencies
        )
        if len(set(dependencies)) != len(dependencies):
            raise OpenDiscoveryError("dependencies must not contain duplicates")
        if self.edit_id in dependencies:
            raise OpenDiscoveryError("an edit must not depend on itself")
        object.__setattr__(self, "dependencies", dependencies)
        if not isinstance(self.incompatibilities, (list, tuple)):
            raise OpenDiscoveryError("incompatibilities must be an array")
        incompatibilities = tuple(
            _identifier(value, name="incompatibilities[]") for value in self.incompatibilities
        )
        if len(set(incompatibilities)) != len(incompatibilities):
            raise OpenDiscoveryError("incompatibilities must not contain duplicates")
        if self.edit_id in incompatibilities:
            raise OpenDiscoveryError("an edit must not conflict with itself")
        if set(dependencies).intersection(incompatibilities):
            raise OpenDiscoveryError("an edit cannot both depend on and conflict with another edit")
        object.__setattr__(self, "incompatibilities", incompatibilities)
        if not isinstance(self.preconditions, (list, tuple)) or not all(
            isinstance(item, Precondition) for item in self.preconditions
        ):
            raise OpenDiscoveryError("preconditions must be an array of typed Precondition records")
        preconditions = tuple(self.preconditions)
        if len({item.precondition_id for item in preconditions}) != len(preconditions):
            raise OpenDiscoveryError("preconditions must not contain duplicate precondition_id values")
        object.__setattr__(self, "preconditions", preconditions)
        config_merge_patch = _json_mapping(
            self.config_merge_patch,
            name="config_merge_patch",
            allow_empty=True,
        )
        _assert_no_protected_mutation(config_merge_patch, name="config_merge_patch")
        object.__setattr__(self, "config_merge_patch", config_merge_patch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "address": self.address,
            "kind": self.kind,
            "description": self.description,
            "symbol_patch": self.symbol_patch.to_dict(),
            "dependencies": list(self.dependencies),
            "incompatibilities": list(self.incompatibilities),
            "preconditions": [item.to_dict() for item in self.preconditions],
            "config_merge_patch": _thaw(self.config_merge_patch),
        }


@dataclass(frozen=True)
class ChangeManifest:
    """A structured manifest that may contain one or many coupled edits."""

    summary: str
    edits: tuple[ChangeEdit, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _text(self.summary, name="change_manifest.summary"))
        if not isinstance(self.edits, (list, tuple)) or not self.edits:
            raise OpenDiscoveryError("change_manifest.edits must be a non-empty array")
        edits = tuple(self.edits)
        if not all(isinstance(edit, ChangeEdit) for edit in edits):
            raise OpenDiscoveryError("change_manifest.edits must contain ChangeEdit records")
        identifiers = tuple(edit.edit_id for edit in edits)
        if len(set(identifiers)) != len(identifiers):
            raise OpenDiscoveryError("change_manifest edit_id values must be unique")
        if len(edits) > MAX_DISCOVERY_MANIFEST_EDITS:
            raise OpenDiscoveryError(
                "change_manifest.edits exceeds the registered maximum of "
                f"{MAX_DISCOVERY_MANIFEST_EDITS} edits"
            )
        available = set(identifiers)
        for edit in edits:
            unknown = set(edit.dependencies).difference(available)
            if unknown:
                raise OpenDiscoveryError(
                    f"change_manifest edit {edit.edit_id} has unknown dependencies: {sorted(unknown)}"
                )
            unknown_incompatibilities = set(edit.incompatibilities).difference(available)
            if unknown_incompatibilities:
                raise OpenDiscoveryError(
                    f"change_manifest edit {edit.edit_id} has unknown incompatibilities: "
                    f"{sorted(unknown_incompatibilities)}"
                )
        object.__setattr__(self, "edits", edits)

    @property
    def manifest_hash(self) -> str:
        return digest(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {"summary": self.summary, "edits": [edit.to_dict() for edit in self.edits]}
        if include_hash:
            result["manifest_hash"] = self.manifest_hash
        return result

    @classmethod
    def from_dict(cls, payload: Any) -> "ChangeManifest":
        if not isinstance(payload, Mapping):
            raise OpenDiscoveryError("change_manifest must be an object")
        allowed = {"summary", "edits"}
        missing = allowed.difference(payload)
        unknown = set(payload).difference(allowed)
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing fields {sorted(missing)}")
            if unknown:
                detail.append(f"unknown fields {sorted(unknown)}")
            raise OpenDiscoveryError("change_manifest has " + "; ".join(detail))
        raw_edits = payload["edits"]
        if not isinstance(raw_edits, (list, tuple)):
            raise OpenDiscoveryError("change_manifest.edits must be an array")
        edits: list[ChangeEdit] = []
        for index, raw_edit in enumerate(raw_edits):
            if not isinstance(raw_edit, Mapping):
                raise OpenDiscoveryError(f"change_manifest.edits[{index}] must be an object")
            required = {
                "edit_id",
                "address",
                "kind",
                "description",
                "symbol_patch",
                "dependencies",
                "incompatibilities",
                "preconditions",
                "config_merge_patch",
            }
            unknown = set(raw_edit).difference(required)
            missing = required.difference(raw_edit)
            if missing or unknown:
                detail = []
                if missing:
                    detail.append(f"missing fields {sorted(missing)}")
                if unknown:
                    detail.append(f"unknown fields {sorted(unknown)}")
                raise OpenDiscoveryError(
                    f"change_manifest.edits[{index}] has " + "; ".join(detail)
                )
            if not isinstance(raw_edit["dependencies"], (list, tuple)):
                raise OpenDiscoveryError(
                    f"change_manifest.edits[{index}].dependencies must be an array"
                )
            if not isinstance(raw_edit["incompatibilities"], (list, tuple)):
                raise OpenDiscoveryError(
                    f"change_manifest.edits[{index}].incompatibilities must be an array"
                )
            if not isinstance(raw_edit["preconditions"], (list, tuple)):
                raise OpenDiscoveryError(
                    f"change_manifest.edits[{index}].preconditions must be an array"
                )
            if not isinstance(raw_edit["config_merge_patch"], Mapping):
                raise OpenDiscoveryError(
                    f"change_manifest.edits[{index}].config_merge_patch must be an object"
                )
            edits.append(
                ChangeEdit(
                    edit_id=raw_edit["edit_id"],
                    address=raw_edit["address"],
                    kind=raw_edit["kind"],
                    description=raw_edit["description"],
                    symbol_patch=SymbolPatch.from_dict(raw_edit["symbol_patch"]),
                    dependencies=tuple(raw_edit["dependencies"]),
                    incompatibilities=tuple(raw_edit["incompatibilities"]),
                    preconditions=tuple(
                        Precondition.from_dict(item) for item in raw_edit["preconditions"]
                    ),
                    config_merge_patch=raw_edit["config_merge_patch"],
                )
            )
        return cls(summary=payload["summary"], edits=tuple(edits))


def _topological_change_edits(edits: Sequence[ChangeEdit]) -> tuple[ChangeEdit, ...]:
    """Return one deterministic dependency order or reject a cycle."""

    by_id = {edit.edit_id: edit for edit in edits}
    remaining = set(by_id)
    ordered: list[ChangeEdit] = []
    completed: set[str] = set()
    while remaining:
        ready = sorted(
            edit_id
            for edit_id in remaining
            if set(by_id[edit_id].dependencies).issubset(completed)
        )
        if not ready:
            raise OpenDiscoveryError("change_manifest dependencies contain a cycle")
        for edit_id in ready:
            ordered.append(by_id[edit_id])
            completed.add(edit_id)
            remaining.remove(edit_id)
    return tuple(ordered)


def _merged_edit_config_patch(edits: Sequence[ChangeEdit]) -> Mapping[str, Any]:
    """Merge per-edit configuration patches in their deterministic dependency order."""

    merged: dict[str, Any] = {}
    for edit in _topological_change_edits(edits):
        # ``json_merge_patch`` is the exact merge operation the materializer
        # will use when it later applies the individual edit proposal.
        merged = json_merge_patch(merged, _thaw(edit.config_merge_patch))
    return _freeze(merged)


@dataclass(frozen=True)
class OpenDiscoveryResponse:
    """Parsed response content, separate from its raw response text trace."""

    parent_candidate_hash: str
    candidate_source: str
    candidate_metadata: Mapping[str, Any]
    training_config_patch: Mapping[str, Any]
    hypothesis: str
    change_manifest: ChangeManifest

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_candidate_hash",
            _hash(self.parent_candidate_hash, name="parent_candidate_hash"),
        )
        object.__setattr__(
            self,
            "candidate_source",
            _assert_python_source(self.candidate_source, name="candidate_source"),
        )
        object.__setattr__(
            self,
            "candidate_metadata",
            _validate_candidate_metadata(
                self.candidate_metadata,
                source=self.candidate_source,
            ),
        )
        patch = _json_mapping(
            self.training_config_patch,
            name="training_config_patch",
            allow_empty=True,
        )
        _assert_no_protected_mutation(patch, name="training_config_patch")
        object.__setattr__(self, "training_config_patch", patch)
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, name="hypothesis"))
        if not isinstance(self.change_manifest, ChangeManifest):
            raise OpenDiscoveryError("change_manifest must be a ChangeManifest")
        expected_patch = _merged_edit_config_patch(self.change_manifest.edits)
        if canonical_json(_thaw(patch)) != canonical_json(_thaw(expected_patch)):
            raise OpenDiscoveryError(
                "training_config_patch must equal the dependency-ordered merge of all edit config_merge_patch values"
            )

    @property
    def source_hash(self) -> str:
        return digest(self.candidate_source)

    @property
    def training_config_patch_hash(self) -> str:
        return digest(_thaw(self.training_config_patch))

    def child_candidate(self, parent: CandidateState) -> CandidateState:
        if parent.candidate_hash != self.parent_candidate_hash:
            raise OpenDiscoveryError("response parent_candidate_hash does not match the current candidate")
        child = CandidateState(
            candidate_source=self.candidate_source,
            candidate_metadata=self.candidate_metadata,
            training_config=_merge_patch(parent.training_config, self.training_config_patch),
            parent_candidate_hash=parent.candidate_hash,
        )
        parent_exports = _patchable_symbol_set(parent)
        child_exports = _patchable_symbol_set(child)
        for edit in self.change_manifest.edits:
            missing_parent = set(edit.symbol_patch.parent_symbols).difference(parent_exports)
            if missing_parent:
                raise OpenDiscoveryError(
                    f"symbol_patch for {edit.edit_id!r} references undeclared parent symbols: "
                    f"{sorted(missing_parent)}"
                )
            missing_child = set(edit.symbol_patch.child_symbols).difference(child_exports)
            if missing_child:
                raise OpenDiscoveryError(
                    f"symbol_patch for {edit.edit_id!r} references undeclared child symbols: "
                    f"{sorted(missing_child)}"
                )
        return child

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPEN_DISCOVERY_RESPONSE_SCHEMA_VERSION,
            "parent_candidate_hash": self.parent_candidate_hash,
            "candidate_source": self.candidate_source,
            "source_hash": self.source_hash,
            "candidate_metadata": _thaw(self.candidate_metadata),
            "candidate_metadata_hash": digest(_thaw(self.candidate_metadata)),
            "training_config_patch": _thaw(self.training_config_patch),
            "training_config_patch_hash": self.training_config_patch_hash,
            "hypothesis": self.hypothesis,
            "change_manifest": self.change_manifest.to_dict(),
        }


def proposal_history_summary(response: OpenDiscoveryResponse) -> dict[str, Any]:
    """Return a bounded, source-free summary of one parsed proposal."""

    if not isinstance(response, OpenDiscoveryResponse):
        raise OpenDiscoveryError("proposal history summary requires OpenDiscoveryResponse")
    manifest = response.change_manifest
    addresses = tuple(edit.address for edit in manifest.edits)
    return {
        "proposal_hash": digest(response.to_dict()),
        "hypothesis": _bounded_history_text(
            response.hypothesis,
            name="hypothesis",
            limit=MAX_PROMPT_HISTORY_HYPOTHESIS_CHARS,
        ),
        "manifest_hash": manifest.manifest_hash,
        "manifest_summary": _bounded_history_text(
            manifest.summary,
            name="change_manifest.summary",
            limit=MAX_PROMPT_HISTORY_SUMMARY_CHARS,
        ),
        **_bounded_history_labels(addresses, label="addresses"),
        "training_config_patch_hash": response.training_config_patch_hash,
    }


def _history_digest_or_value(value: Any) -> str | None:
    """Preserve a valid digest or fingerprint arbitrary archived detail."""

    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    return _history_hash(value)


def _history_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _history_metric_pairs(value: Any) -> list[tuple[str, float | None]]:
    """Read either archived metric maps or already-compact metric lists."""

    if isinstance(value, Mapping):
        return [
            (str(name), metric if metric is None or isinstance(metric, (int, float)) else None)
            for name, metric in value.items()
        ]
    if isinstance(value, (list, tuple)):
        pairs: list[tuple[str, float | None]] = []
        for item in value:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                continue
            metric = item.get("value")
            pairs.append(
                (
                    item["name"],
                    metric if metric is None or isinstance(metric, (int, float)) else None,
                )
            )
        return pairs
    return []


def _compact_fold3_history_record(value: Any) -> Mapping[str, Any] | None:
    """Canonicalize archived or current diagnostics into prompt-safe memory."""

    raw = _history_mapping(value)
    if raw is None:
        return None
    raw_metrics = raw.get("metrics", {})
    metric_pairs = _history_metric_pairs(raw_metrics)
    metric_pairs.sort(key=lambda item: (_normalise_name(item[0]) != "globalpcc", item[0]))
    failure_raw = _history_mapping(raw.get("failure")) or _history_mapping(
        raw.get("failure_summary")
    )
    failure = None
    if failure_raw is not None:
        failure_type = failure_raw.get("failure_type", failure_raw.get("type", "unknown"))
        failure_reason = failure_raw.get(
            "failure_reason",
            failure_raw.get("failure_message", failure_raw.get("message", failure_raw.get("reason", ""))),
        )
        failure = {
            "failure_hash": _history_digest_or_value(
                failure_raw.get("failure_hash", failure_raw)
            ),
            "failure_type": _bounded_history_value(
                failure_type,
                name="prompt_history.failure_type",
                limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
            ),
            "failure_reason": _bounded_history_value(
                failure_reason,
                name="prompt_history.failure_reason",
                limit=MAX_PROMPT_HISTORY_FAILURE_CHARS,
            ),
        }
    metric_hash_input = raw_metrics if raw_metrics is not None else {}
    metric_count = raw.get("metric_count")
    if not isinstance(metric_count, int) or isinstance(metric_count, bool) or metric_count < 0:
        metric_count = len(metric_pairs)
    runtime = raw.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or isinstance(runtime, bool) or not math.isfinite(float(runtime)):
        runtime = None
    return {
        "diagnostics_hash": _history_digest_or_value(
            raw.get("diagnostics_hash", raw.get("fold3_diagnostics_hash", raw))
        ),
        "status": _bounded_history_value(
            raw.get("status", "unknown"),
            name="prompt_history.diagnostic_status",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        "metrics_hash": _history_digest_or_value(raw.get("metrics_hash", metric_hash_input)),
        "metric_count": metric_count,
        "metrics": [
            {
                "name": _bounded_history_text(
                    name,
                    name="prompt_history.metric_name",
                    limit=MAX_PROMPT_HISTORY_METRIC_NAME_CHARS,
                ),
                "value": value,
            }
            for name, value in metric_pairs[:MAX_PROMPT_HISTORY_METRIC_COUNT]
        ],
        "runtime_seconds": None if runtime is None else float(runtime),
        "failure": failure,
    }


def _compact_label_fields(
    raw: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    values = raw.get(label, ())
    if not isinstance(values, (list, tuple)):
        values = ()
    normalized = tuple(value for value in values if isinstance(value, str))
    count = raw.get(f"{label}_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        count = len(normalized)
    return {
        f"{label}_count": count,
        f"{label}_hash": _history_digest_or_value(
            raw.get(f"{label}_hash", list(normalized))
        ),
        label: [
            _bounded_history_text(
                value,
                name=f"prompt_history.{label}[]",
                limit=MAX_PROMPT_HISTORY_ADDRESS_CHARS,
            )
            for value in normalized[:MAX_PROMPT_HISTORY_ADDRESS_COUNT]
        ],
    }


def _compact_proposal_history_record(value: Any) -> Mapping[str, Any] | None:
    """Reduce archived manifests or current summaries to one memory shape."""

    raw = _history_mapping(value)
    if raw is None:
        return None
    manifest = _history_mapping(raw.get("change_manifest"))
    raw_addresses: list[Any] = list(raw.get("addresses", ())) if isinstance(
        raw.get("addresses"), (list, tuple)
    ) else []
    if not raw_addresses and manifest is not None:
        edits = manifest.get("edits")
        if isinstance(edits, (list, tuple)):
            raw_addresses = [
                edit.get("address")
                for edit in edits
                if isinstance(edit, Mapping) and isinstance(edit.get("address"), str)
            ]
    address_input = {
        "addresses": raw_addresses,
        "addresses_count": raw.get("addresses_count"),
        "addresses_hash": raw.get("addresses_hash"),
    }
    manifest_summary = raw.get("manifest_summary")
    if manifest_summary is None and manifest is not None:
        manifest_summary = manifest.get("summary", "")
    manifest_hash_input = manifest if manifest is not None else raw
    return {
        "proposal_hash": _history_digest_or_value(raw.get("proposal_hash", raw)),
        "hypothesis": _bounded_history_value(
            raw.get("hypothesis", ""),
            name="prompt_history.hypothesis",
            limit=MAX_PROMPT_HISTORY_HYPOTHESIS_CHARS,
        ),
        "manifest_hash": _history_digest_or_value(
            raw.get(
                "manifest_hash",
                manifest.get("manifest_hash") if manifest is not None else manifest_hash_input,
            )
        ),
        "manifest_summary": _bounded_history_value(
            manifest_summary if manifest_summary is not None else "",
            name="prompt_history.manifest_summary",
            limit=MAX_PROMPT_HISTORY_SUMMARY_CHARS,
        ),
        **_compact_label_fields(address_input, label="addresses"),
        "training_config_patch_hash": _history_digest_or_value(
            raw.get(
                "training_config_patch_hash",
                raw.get("training_config_patch", {}),
            )
        ),
    }


def _compact_candidate_history_record(value: Any) -> Mapping[str, Any] | None:
    """Canonicalize a source-free child-candidate record for prompt memory."""

    raw = _history_mapping(value)
    if raw is None:
        return None
    metadata = _history_mapping(raw.get("candidate_metadata"))
    source_fields = raw
    if metadata is not None:
        source_fields = {
            **raw,
            "candidate_name": raw.get("candidate_name", metadata.get("candidate_name", "")),
            "semantic_components": raw.get(
                "semantic_components", metadata.get("semantic_components", ())
            ),
            "change_summary": raw.get("change_summary", metadata.get("change_summary", "")),
        }
    return {
        "candidate_hash": _history_digest_or_value(raw.get("candidate_hash", raw)),
        "parent_candidate_hash": _history_digest_or_value(raw.get("parent_candidate_hash")),
        "candidate_name": _bounded_history_value(
            source_fields.get("candidate_name", ""),
            name="prompt_history.candidate_name",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        **_compact_label_fields(source_fields, label="semantic_components"),
        "change_summary": _bounded_history_value(
            source_fields.get("change_summary", ""),
            name="prompt_history.change_summary",
            limit=MAX_PROMPT_HISTORY_SUMMARY_CHARS,
        ),
    }


def _compact_observed_outcome(value: Any) -> Mapping[str, Any] | None:
    """Keep outcome ownership/reason visible without replaying full artifacts."""

    raw = _history_mapping(value)
    if raw is None:
        return None
    failure = _history_mapping(raw.get("failure"))
    owner = raw.get("failure_owner", raw.get("owner", None))
    reason: Any = raw.get("failure_reason", raw.get("reason", None))
    if reason is None:
        for key in ("parse_error", "budget_error", "config_error", "binding_error"):
            if key in raw:
                reason = raw[key]
                break
    if reason is None and failure is not None:
        reason = failure.get(
            "failure_reason",
            failure.get("failure_message", failure.get("message", failure.get("reason", None))),
        )
        if owner is None:
            owner = failure.get("failure_owner", failure.get("owner", failure.get("stage", None)))
    pcc = raw.get("fold3_global_pcc")
    if not isinstance(pcc, (int, float)) or isinstance(pcc, bool) or not math.isfinite(float(pcc)):
        pcc = None
    result: dict[str, Any] = {
        "outcome_hash": _history_digest_or_value(raw.get("outcome_hash", raw)),
        "status": _bounded_history_value(
            raw.get("status", "unknown"),
            name="prompt_history.outcome_status",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        "candidate_hash": _history_digest_or_value(raw.get("candidate_hash")),
        "scientific_iteration_consumed": raw.get("scientific_iteration_consumed")
        if isinstance(raw.get("scientific_iteration_consumed"), bool)
        else None,
        "accepted_as_incumbent": raw.get("accepted_as_incumbent")
        if isinstance(raw.get("accepted_as_incumbent"), bool)
        else None,
        "fold3_global_pcc": None if pcc is None else float(pcc),
    }
    if owner is not None:
        result["failure_owner"] = _bounded_history_value(
            owner,
            name="prompt_history.failure_owner",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        )
    if reason is not None:
        result["failure_reason"] = _bounded_history_value(
            reason,
            name="prompt_history.outcome_reason",
            limit=MAX_PROMPT_HISTORY_FAILURE_CHARS,
        )
    return result


def _fit_compact_prompt_history_record(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Guarantee the memory envelope while retaining a legal long proposal.

    Normal records retain the bounded natural-language evidence above.  A
    pathological record can contain every optional failure field at once.  In
    that case, progressively shorten only display text and list previews; the
    corresponding hashes and counts stay intact, and the complete evidence is
    still available in the immutable trace archive.
    """

    result = _json_copy(compact, name="compact_prompt_history_record")
    assert isinstance(result, dict)

    def size() -> int:
        return len(canonical_json(result))

    if size() <= MAX_PROMPT_HISTORY_RECORD_CHARS:
        return result

    def shorten(mapping: Any, key: str, limit: int) -> None:
        if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
            mapping[key] = _bounded_history_text(
                mapping[key],
                name=f"compact_prompt_history.{key}",
                limit=limit,
            )

    proposal = result.get("proposal")
    child = result.get("child_candidate")
    diagnostics = result.get("fold3_result")
    outcome = result.get("observed_outcome")
    for mapping, key, limit in (
        (proposal, "hypothesis", 96),
        (proposal, "manifest_summary", 96),
        (child, "candidate_name", 48),
        (child, "change_summary", 96),
        (diagnostics, "status", 48),
        (outcome, "status", 48),
        (outcome, "failure_owner", 48),
        (outcome, "failure_reason", 96),
        (result, "source_model", 48),
        (result, "parse_error", 96),
        (result, "execution_status", 48),
    ):
        shorten(mapping, key, limit)
    failure = diagnostics.get("failure") if isinstance(diagnostics, dict) else None
    shorten(failure, "failure_type", 48)
    shorten(failure, "failure_reason", 96)
    if size() <= MAX_PROMPT_HISTORY_RECORD_CHARS:
        return result

    # Keep one headline item per collection; exact full identities are retained
    # by the collection hash and count fields.
    for mapping, key in (
        (proposal, "addresses"),
        (child, "semantic_components"),
        (diagnostics, "metrics"),
    ):
        if isinstance(mapping, dict) and isinstance(mapping.get(key), list):
            mapping[key] = mapping[key][:1]
    if size() <= MAX_PROMPT_HISTORY_RECORD_CHARS:
        return result

    # This final branch is intentionally rare.  It preserves the proof hashes
    # and minimal status while collapsing verbose display evidence to markers.
    for mapping, key in (
        (proposal, "hypothesis"),
        (proposal, "manifest_summary"),
        (child, "candidate_name"),
        (child, "change_summary"),
        (outcome, "failure_owner"),
        (outcome, "failure_reason"),
        (result, "source_model"),
        (result, "parse_error"),
    ):
        if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
            mapping[key] = "#" + digest(mapping[key])[:16]
    if isinstance(failure, dict):
        for key in ("failure_type", "failure_reason"):
            if isinstance(failure.get(key), str):
                failure[key] = "#" + digest(failure[key])[:16]
    for mapping, key in (
        (proposal, "addresses"),
        (child, "semantic_components"),
        (diagnostics, "metrics"),
    ):
        if isinstance(mapping, dict) and isinstance(mapping.get(key), list):
            mapping[key] = []
    if size() > MAX_PROMPT_HISTORY_RECORD_CHARS:
        raise OpenDiscoveryError(
            "compact prompt-history record cannot fit the registered "
            f"{MAX_PROMPT_HISTORY_RECORD_CHARS}-character limit"
        )
    return result


def _compact_prompt_history_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one prior-round archive record into a fixed-size memory item.

    This also accepts the source-free records written by earlier runs.  It is
    therefore safe for recovery, but never admits raw prompt, response, or
    candidate-source bodies into a later model request.
    """

    copied = _json_copy(record, name="prior_trajectory record")
    if not isinstance(copied, Mapping):  # Defensive after _json_copy.
        raise OpenDiscoveryError("prior_trajectory records must be JSON objects")
    if _contains_raw_interaction_body(copied):
        raise OpenDiscoveryError(
            "prior_trajectory must use compact history records, not raw prompt/response bodies"
        )
    round_index = copied.get("round_index")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise OpenDiscoveryError("prior_trajectory history record round_index must be positive")
    diagnostics_raw = copied.get("fold3_result", copied.get("fold3_diagnostics"))
    outcome_raw = copied.get("observed_outcome")
    # Interrupted runner recovery records predate an explicit outcome envelope.
    if outcome_raw is None and copied.get("status") not in {None, UNEXECUTED}:
        outcome_raw = {
            "status": copied.get("status"),
            "scientific_iteration_consumed": copied.get("scientific_iteration_consumed"),
        }
    compact: dict[str, Any] = {
        "history_schema_version": PROMPT_HISTORY_SCHEMA_VERSION,
        "round_index": round_index,
        "source_model": _bounded_history_value(
            copied.get("source_model", "unknown"),
            name="prompt_history.source_model",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        "parent_candidate_hash": _history_digest_or_value(
            copied.get("parent_candidate_hash", copied.get("current_candidate_hash"))
        ),
        "fold3_result": _compact_fold3_history_record(diagnostics_raw),
        "proposal": _compact_proposal_history_record(copied.get("proposal")),
        "child_candidate": _compact_candidate_history_record(copied.get("child_candidate")),
        "observed_outcome": _compact_observed_outcome(outcome_raw),
        "parse_error": (
            None
            if copied.get("parse_error") is None
            else _bounded_history_value(
                copied.get("parse_error"),
                name="prompt_history.parse_error",
                limit=MAX_PROMPT_HISTORY_FAILURE_CHARS,
            )
        ),
        "execution_status": _bounded_history_value(
            copied.get("execution_status", UNEXECUTED),
            name="prompt_history.execution_status",
            limit=MAX_PROMPT_HISTORY_MODEL_CHARS,
        ),
        "trace_hash": _history_digest_or_value(copied.get("trace_hash")),
    }
    return _fit_compact_prompt_history_record(compact)


@dataclass(frozen=True)
class OpenDiscoveryRoundRequest:
    """All context supplied to one free-form language-model discovery round."""

    round_index: int
    protected_task: ProtectedDiscoveryTask
    fixed_initial_entry: FixedInitialEntry
    current_candidate: CandidateState
    fold3_diagnostics: Fold3Diagnostics
    prior_trajectory: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_index", _positive_int(self.round_index, name="round_index"))
        if not isinstance(self.protected_task, ProtectedDiscoveryTask):
            raise OpenDiscoveryError("round request requires a ProtectedDiscoveryTask")
        if not isinstance(self.fixed_initial_entry, FixedInitialEntry):
            raise OpenDiscoveryError("round request requires its fixed initial entry")
        if not isinstance(self.current_candidate, CandidateState):
            raise OpenDiscoveryError("round request requires a CandidateState")
        if not isinstance(self.fold3_diagnostics, Fold3Diagnostics):
            raise OpenDiscoveryError("round request requires Fold3Diagnostics")
        if (
            self.fixed_initial_entry.protected_task.protected_task_hash
            != self.protected_task.protected_task_hash
        ):
            raise OpenDiscoveryError("round request attempts to replace the fixed protected task")
        if self.fold3_diagnostics.partition != self.protected_task.feedback_partition:
            raise OpenDiscoveryError("round diagnostics do not belong to the protected Fold-3 partition")
        if self.fold3_diagnostics.candidate_hash != self.current_candidate.candidate_hash:
            raise OpenDiscoveryError("Fold-3 diagnostics do not belong to the current candidate")
        if self.fold3_diagnostics.protected_task_hash != self.protected_task.protected_task_hash:
            raise OpenDiscoveryError("Fold-3 diagnostics do not match the protected task")
        if self.fold3_diagnostics.split_hash != self.protected_task.split_hash:
            raise OpenDiscoveryError("Fold-3 diagnostics do not match the frozen split plan")
        if self.fold3_diagnostics.evaluator_hash != self.protected_task.evaluator_hash:
            raise OpenDiscoveryError("Fold-3 diagnostics do not match the frozen evaluator")
        if not isinstance(self.prior_trajectory, (list, tuple)):
            raise OpenDiscoveryError("prior_trajectory must be an array")
        prior_records: list[Mapping[str, Any]] = []
        for index, record in enumerate(self.prior_trajectory):
            copied = _json_copy(record, name=f"prior_trajectory[{index}]")
            if not isinstance(copied, Mapping):
                raise OpenDiscoveryError(f"prior_trajectory[{index}] must be an object")
            # Old source-free campaign summaries are deterministically reduced
            # here as well, so recovery cannot re-inflate later prompt context.
            prior_records.append(_freeze(_compact_prompt_history_record(copied)))
        prior = tuple(prior_records)
        if len(prior) != self.round_index - 1:
            raise OpenDiscoveryError(
                "round_index must be exactly one greater than the number of prior trajectory records"
            )
        object.__setattr__(self, "prior_trajectory", prior)

    @property
    def prior_trajectory_hash(self) -> str:
        return digest([_thaw(record) for record in self.prior_trajectory])

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPEN_DISCOVERY_SCHEMA_VERSION,
            "round_index": self.round_index,
            "protected_task": self.protected_task.to_dict(),
            "fixed_initial_entry": self.fixed_initial_entry.to_dict(),
            "current_candidate": self.current_candidate.to_dict(),
            "fold3_diagnostics": self.fold3_diagnostics.to_dict(),
            "prior_trajectory": [_thaw(record) for record in self.prior_trajectory],
            "prior_trajectory_hash": self.prior_trajectory_hash,
        }

    @property
    def request_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        if include_hash:
            result["request_hash"] = self.request_hash
        return result


@dataclass(frozen=True)
class OpenDiscoveryRoundTrace:
    """One raw prompt/response interaction, including invalid-response traces."""

    round_index: int
    source_model: str
    request_hash: str
    protected_task_hash: str
    initial_entry_hash: str
    current_candidate_hash: str
    fold3_diagnostics_hash: str
    fold3_diagnostics: Fold3Diagnostics
    prior_trajectory_hash: str
    raw_prompt: str
    raw_response: str
    parsed_response: OpenDiscoveryResponse | None
    child_candidate: CandidateState | None
    parse_error: str | None
    execution_status: str = UNEXECUTED
    response_normalizations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_index", _positive_int(self.round_index, name="round_index"))
        object.__setattr__(self, "source_model", _identifier(self.source_model, name="source_model"))
        for name in (
            "request_hash",
            "protected_task_hash",
            "initial_entry_hash",
            "current_candidate_hash",
            "fold3_diagnostics_hash",
            "prior_trajectory_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        if not isinstance(self.fold3_diagnostics, Fold3Diagnostics):
            raise OpenDiscoveryError("trace requires the complete Fold3Diagnostics record")
        if self.fold3_diagnostics.diagnostics_hash != self.fold3_diagnostics_hash:
            raise OpenDiscoveryError("trace Fold-3 diagnostics hash does not match its diagnostics record")
        if self.fold3_diagnostics.candidate_hash != self.current_candidate_hash:
            raise OpenDiscoveryError("trace Fold-3 diagnostics do not belong to its parent candidate")
        if self.fold3_diagnostics.protected_task_hash != self.protected_task_hash:
            raise OpenDiscoveryError("trace Fold-3 diagnostics do not match the protected task")
        object.__setattr__(
            self,
            "raw_prompt",
            _text(self.raw_prompt, name="raw_prompt", limit=_MAX_PROMPT_CHARS),
        )
        object.__setattr__(
            self,
            "raw_response",
            _text(self.raw_response, name="raw_response", allow_empty=True, limit=_MAX_RESPONSE_CHARS),
        )
        if not isinstance(self.response_normalizations, (list, tuple)):
            raise OpenDiscoveryError("trace response_normalizations must be an array")
        normalizations = tuple(
            _identifier(value, name="response_normalizations[]")
            for value in self.response_normalizations
        )
        if len(set(normalizations)) != len(normalizations):
            raise OpenDiscoveryError("trace response_normalizations must not contain duplicates")
        allowed_normalizations = {
            "json_fence_stripped",
            "metadata_support_address_normalized",
        }
        if set(normalizations).difference(allowed_normalizations):
            raise OpenDiscoveryError("trace has an unknown response normalization")
        object.__setattr__(self, "response_normalizations", normalizations)
        if self.execution_status != UNEXECUTED:
            raise OpenDiscoveryError("open-discovery traces must mark candidate source as unexecuted")
        success = self.parsed_response is not None or self.child_candidate is not None
        if success:
            if not isinstance(self.parsed_response, OpenDiscoveryResponse) or not isinstance(
                self.child_candidate, CandidateState
            ):
                raise OpenDiscoveryError("successful traces require parsed response and child candidate")
            if self.parse_error is not None:
                raise OpenDiscoveryError("successful traces must not contain parse_error")
            if self.parsed_response.parent_candidate_hash != self.current_candidate_hash:
                raise OpenDiscoveryError("trace response parent does not match its request")
            if self.child_candidate.parent_candidate_hash != self.current_candidate_hash:
                raise OpenDiscoveryError("trace child does not point to its requested parent")
        else:
            if not isinstance(self.parse_error, str) or not self.parse_error:
                raise OpenDiscoveryError("invalid-response traces require a parse_error")

    @property
    def prompt_hash(self) -> str:
        return digest(self.raw_prompt)

    @property
    def response_hash(self) -> str:
        return digest(self.raw_response)

    def _core_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "source_model": self.source_model,
            "request_hash": self.request_hash,
            "protected_task_hash": self.protected_task_hash,
            "initial_entry_hash": self.initial_entry_hash,
            "current_candidate_hash": self.current_candidate_hash,
            "fold3_diagnostics_hash": self.fold3_diagnostics_hash,
            "fold3_diagnostics": self.fold3_diagnostics.to_dict(),
            "prior_trajectory_hash": self.prior_trajectory_hash,
            "raw_prompt": self.raw_prompt,
            "prompt_hash": self.prompt_hash,
            "raw_response": self.raw_response,
            "response_hash": self.response_hash,
            "response_normalizations": list(self.response_normalizations),
            "parsed_response": (
                None if self.parsed_response is None else self.parsed_response.to_dict()
            ),
            "child_candidate": (
                None if self.child_candidate is None else self.child_candidate.to_dict()
            ),
            "parse_error": self.parse_error,
            "execution_status": self.execution_status,
        }

    def history_record_for_prompt(self) -> dict[str, Any]:
        """Return the bounded history memory allowed into the next prompt.

        Full raw prompt/response bodies remain in :meth:`to_dict` for the
        immutable archive.  This compact record deliberately carries only
        hashes and bounded decision evidence; it never replays a full manifest,
        diagnostic payload, or historical candidate source.
        """

        return _compact_prompt_history_record({
            "round_index": self.round_index,
            "source_model": self.source_model,
            "request_hash": self.request_hash,
            "protected_task_hash": self.protected_task_hash,
            "initial_entry_hash": self.initial_entry_hash,
            "parent_candidate_hash": self.current_candidate_hash,
            "fold3_result": fold3_history_summary(self.fold3_diagnostics),
            "proposal": (
                None
                if self.parsed_response is None
                else proposal_history_summary(self.parsed_response)
            ),
            "child_candidate": (
                None
                if self.child_candidate is None
                else candidate_history_summary(self.child_candidate)
            ),
            "parse_error": self.parse_error,
            "execution_status": self.execution_status,
            "prompt_hash": self.prompt_hash,
            "response_hash": self.response_hash,
            "trace_hash": self.trace_hash,
        })

    @property
    def trace_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        if include_hash:
            result["trace_hash"] = self.trace_hash
        return result


@dataclass(frozen=True)
class OpenDiscoveryTrajectory:
    """Hash-linked trajectory rooted at exactly one fixed initial entry."""

    fixed_initial_entry: FixedInitialEntry
    rounds: tuple[OpenDiscoveryRoundTrace, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_initial_entry, FixedInitialEntry):
            raise OpenDiscoveryError("trajectory requires a FixedInitialEntry")
        if not isinstance(self.rounds, (list, tuple)):
            raise OpenDiscoveryError("trajectory rounds must be an array")
        rounds = tuple(self.rounds)
        if not all(isinstance(round_record, OpenDiscoveryRoundTrace) for round_record in rounds):
            raise OpenDiscoveryError("trajectory rounds must contain OpenDiscoveryRoundTrace records")
        if len(rounds) > self.fixed_initial_entry.exploration_budget:
            raise OpenDiscoveryError("trajectory exceeds its fixed exploration budget")

        expected_candidate = self.fixed_initial_entry.initial_candidate
        prior: list[dict[str, Any]] = []
        task = self.fixed_initial_entry.protected_task
        for expected_index, round_record in enumerate(rounds, start=1):
            if round_record.round_index != expected_index:
                raise OpenDiscoveryError("trajectory rounds must use consecutive indices")
            if round_record.protected_task_hash != task.protected_task_hash:
                raise OpenDiscoveryError("trajectory round uses a different protected task")
            if round_record.initial_entry_hash != self.fixed_initial_entry.initial_entry_hash:
                raise OpenDiscoveryError("trajectory round uses a different fixed initial entry")
            if round_record.current_candidate_hash != expected_candidate.candidate_hash:
                raise OpenDiscoveryError("trajectory round has the wrong current candidate")
            if round_record.prior_trajectory_hash != digest(prior):
                raise OpenDiscoveryError("trajectory round does not contain the complete prior trajectory")
            if round_record.child_candidate is not None:
                expected_candidate = round_record.child_candidate
            prior.append(round_record.history_record_for_prompt())
        object.__setattr__(self, "rounds", rounds)

    @property
    def current_candidate(self) -> CandidateState:
        for round_record in reversed(self.rounds):
            if round_record.child_candidate is not None:
                return round_record.child_candidate
        return self.fixed_initial_entry.initial_candidate

    @property
    def prior_trajectory(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            _freeze(round_record.history_record_for_prompt()) for round_record in self.rounds
        )

    def next_request(self, diagnostics: Fold3Diagnostics) -> OpenDiscoveryRoundRequest:
        if len(self.rounds) >= self.fixed_initial_entry.exploration_budget:
            raise OpenDiscoveryError("fixed exploration budget is exhausted")
        return OpenDiscoveryRoundRequest(
            round_index=len(self.rounds) + 1,
            protected_task=self.fixed_initial_entry.protected_task,
            fixed_initial_entry=self.fixed_initial_entry,
            current_candidate=self.current_candidate,
            fold3_diagnostics=diagnostics,
            prior_trajectory=self.prior_trajectory,
        )

    def append(self, round_record: OpenDiscoveryRoundTrace) -> "OpenDiscoveryTrajectory":
        return OpenDiscoveryTrajectory(
            fixed_initial_entry=self.fixed_initial_entry,
            rounds=(*self.rounds, round_record),
        )

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OPEN_DISCOVERY_SCHEMA_VERSION,
            "fixed_initial_entry": self.fixed_initial_entry.to_dict(),
            "rounds": [round_record.to_dict() for round_record in self.rounds],
            "current_candidate_hash": self.current_candidate.candidate_hash,
        }

    @property
    def trajectory_hash(self) -> str:
        return digest(self._core_dict())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = self._core_dict()
        if include_hash:
            result["trajectory_hash"] = self.trajectory_hash
        return result


def build_open_discovery_prompt(request: OpenDiscoveryRoundRequest) -> str:
    """Render the complete, hashable prompt for one language-model round."""

    if not isinstance(request, OpenDiscoveryRoundRequest):
        raise OpenDiscoveryError("build_open_discovery_prompt requires an OpenDiscoveryRoundRequest")
    context = canonical_json(request.to_dict())
    return f"""You are running an open-ended perturbation-response model discovery round.

The design space is open. You may propose a genuinely new full Python model
source, including architecture, loss, optimizer, scheduler, and other
source-level training choices, in response to the real Fold-3 diagnostics.
For a failed or timed-out run, those diagnostics may include a sanitized
``failure_summary``; use that concrete failure evidence to repair the child.
Do not change any protected task data, data fingerprints, split roles, target
semantics, evaluator, metric, or endpoint access.  Do not request Fold 4/5
values.  Your code will be stored only; it will not be executed by this call.

Return exactly one JSON object using this schema:
{{
  "schema_version": "{OPEN_DISCOVERY_RESPONSE_SCHEMA_VERSION}",
  "parent_candidate_hash": "copy the current candidate hash exactly",
  "candidate_source": "complete syntactically valid Python model source",
  "candidate_metadata": {{
    "schema_version": "{CANDIDATE_SCHEMA_VERSION}",
    "candidate_name": "descriptive_name",
    "semantic_components": ["perturbation.new_semantic_address"],
    "component_symbols": {{"perturbation.new_semantic_address": ["ModuleOrFunctionExport"]}},
    "change_summary": "source-level candidate summary"
  }},
  "hypothesis": "free-form scientific/modeling hypothesis",
  "change_manifest": {{
    "summary": "summary of the complete coupled change",
    "edits": [
      {{
        "edit_id": "edit_1",
        "address": "perturbation.some_component",
        "kind": "architecture",
        "description": "what changed",
        "symbol_patch": {{
          "parent_symbols": ["OldCallable"],
          "child_symbols": ["NewCallable"]
        }},
        "dependencies": [],
        "incompatibilities": [],
        "preconditions": [],
        "config_merge_patch": {{"max_epochs": 100}}
      }}
    ]
  }}
}}

``candidate_metadata.component_symbols`` must have exactly the same keys as
``semantic_components``.  Each value is a non-empty list of callable names
exported by the submitted module.  Every semantic address must start with one
of: {", ".join(sorted(ALLOWED_SEMANTIC_ROOTS))}.  You may freely define new
suffixes below those roots; no finite operator bank constrains them.

Candidate API (enforced later by the isolated executor, and statically checked
here without execution): your module must export callable
``candidate_metadata()``, ``build_model(task_spec)``,
``compute_loss(prediction, target, state)``, and
``build_optimizer(model, state)``; ``build_scheduler(optimizer, state)`` is
optional. ``build_model`` must return a model whose forward signature is
``forward(pre, chemical, dose)``. Candidate code receives only tensor inputs,
the dimensions-only task spec with exactly these keys:
``context_dim``, ``chemical_dim``, ``dose_dim``, ``cp_dim``, ``l1000_dim``,
and ``target_dim``. It receives narrow training state with only ``epoch`` and
``max_epochs``. It must not read
files, networks, processes, source paths, split/fold identities, endpoint
arrays, or evaluator objects; the fixed executor owns those controls.

Every manifest edit requires a ``symbol_patch`` containing ``parent_symbols``
and ``child_symbols``: lists of the semantic-component callable exports or
Candidate API wrappers before and after that edit. Use an empty parent list
for an addition, an empty child list for a removal, both lists for a source
change, and two empty lists for a configuration-only edit. The compiler
deterministically derives whether this is add/replace/remove/modify; do not
include an ``operation`` field. Semantic component symbols must agree with the
candidate metadata declarations.

Architecture, loss, optimizer, scheduler, and regularization choices must be
implemented in ``candidate_source`` and represented by their changed callable
symbols. ``config_merge_patch`` is reserved only for the executor-consumed
top-level training controls: ``batch_size``, ``max_epochs``, ``patience``,
``gradient_clip`` (or ``gradient_clip_norm``), and ``min_delta``. Descriptive
fields such as ``architecture.name`` or ``optimizer.lr`` in that JSON patch do
not change execution and are not valid candidate changes.

For exact reconstruction, assign every newly added, modified, or deleted
module-level callable to exactly one edit's ``symbol_patch``. This includes
the ``candidate_metadata`` and ``build_model`` API wrappers. Any other helper
callable must first be declared in ``candidate_metadata.component_symbols``;
the required API exports and optional ``build_scheduler``/``step_scheduler``
are legal patch symbols directly. Do not assign an unchanged callable to an
edit. Modify existing component names in place rather than deleting or renaming
them, leave top-level imports and
module assignments unchanged, and list every new helper callable under exactly
one metadata semantic address and its matching manifest edit.

Submit at most {MAX_INITIAL_DISCOVERY_MANIFEST_EDITS} initial edit records for
one proposal. The parser retains four additional bounded slots exclusively for
a later compiler-requested manifest completion; do not use that reserve in an
initial response.
Keep the proposal as one coherent modeling hypothesis rather than a long list
of unrelated micro-edits. When a changed ``candidate_metadata`` or
``build_model`` wrapper only connects a scientific component to the fixed
candidate API, declare it as a separate ``metadata`` or ``candidate_api`` edit;
its address must still be a valid dotted semantic address (for example
``reliability.candidate_metadata`` or ``reliability.candidate_api``), never
the bare word ``metadata``. A pure ``candidate_metadata`` support edit is not a
scientific component: the compiler, rather than the model, determines whether
it is support-only and dependency-locks it to the scientific edit it serves.
Such API support may never be a prerequisite of a scientific edit.

Each edit must also declare ``incompatibilities`` (edit IDs), typed
``preconditions`` (objects with ``precondition_id``, ``kind``, ``value``), and
its own JSON ``config_merge_patch``. Do not submit a top-level
``training_config_patch``: the compiler derives it by dependency-ordered merge
of the per-edit patches. Do not submit any replay report, validator hash,
source patch, or ``symbol_source``: those are compiler-owned and will be
extracted from your full child source after parsing.

Use an empty ``preconditions`` array unless a condition is actually required
for the proposed edit. The only runtime-enforced kinds are
``address_present``, ``address_absent``, ``parent_source_hash``, and
``parent_config_hash``. Do not use ``typed_io_contract``, ``parameter_budget``,
or ``compute_budget`` here; the source compiler accepts executable replay
preconditions from the registered set above.

The immutable request context below includes the fixed initial entry, current
full source/hash, real Fold-3 execution provenance and diagnostics, and the
complete compact prior trajectory. Raw historical prompt/response bodies remain
in the append-only trace archive and are represented here only by their hashes.

{context}
"""


def parse_open_discovery_response(
    raw_response: str,
    *,
    request: OpenDiscoveryRoundRequest,
) -> OpenDiscoveryResponse:
    """Parse a response but never execute, import, or compile candidate source."""

    payload, _ = _decode_open_discovery_response_payload(raw_response)
    return _parse_open_discovery_response_payload(payload, request=request)


def _parse_open_discovery_response_payload(
    payload: Mapping[str, Any],
    *,
    request: OpenDiscoveryRoundRequest,
) -> OpenDiscoveryResponse:
    """Validate one decoded response payload without executing its source."""

    if not isinstance(request, OpenDiscoveryRoundRequest):
        raise OpenDiscoveryError("response parser requires an OpenDiscoveryRoundRequest")
    if not isinstance(payload, Mapping):
        raise OpenDiscoveryError("response must be a JSON object")
    common_required = {
        "schema_version",
        "parent_candidate_hash",
        "candidate_source",
        "candidate_metadata",
        "hypothesis",
        "change_manifest",
    }
    schema_version = payload.get("schema_version")
    if schema_version != OPEN_DISCOVERY_RESPONSE_SCHEMA_VERSION:
        raise OpenDiscoveryError("response has an unsupported schema_version")
    required = common_required
    allowed = required
    missing = required.difference(payload)
    unknown = set(payload).difference(allowed)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing fields {sorted(missing)}")
        if unknown:
            detail.append(f"unknown fields {sorted(unknown)}")
        raise OpenDiscoveryError("response has " + "; ".join(detail))
    manifest = ChangeManifest.from_dict(payload["change_manifest"])
    derived_patch = _merged_edit_config_patch(manifest.edits)
    response = OpenDiscoveryResponse(
        parent_candidate_hash=payload["parent_candidate_hash"],
        candidate_source=payload["candidate_source"],
        candidate_metadata=payload["candidate_metadata"],
        training_config_patch=derived_patch,
        hypothesis=payload["hypothesis"],
        change_manifest=manifest,
    )
    # Build the child solely to validate the hash binding and merge patch.  It
    # performs no training and never evaluates the supplied Python source.
    response.child_candidate(request.current_candidate)
    return response


def _top_level_callable_sources(source: str, *, name: str) -> Mapping[str, str]:
    """Extract deterministic source fragments for module-level callable exports."""

    try:
        tree = ast.parse(source, filename=f"<{name}>", mode="exec")
    except SyntaxError as exc:  # Defensive: response source was already parsed.
        raise OpenDiscoveryError(f"{name} has invalid Python syntax: {exc.msg}") from exc
    result: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        symbol = statement.name
        if symbol in result:
            raise OpenDiscoveryError(f"{name} has duplicate top-level callable {symbol!r}")
        # AST rendering is intentional: the materializer applies one isolated
        # top-level statement and canonicalizes source before hash comparison.
        result[symbol] = ast.unparse(statement).strip() + "\n"
    return _freeze(result)


def unassigned_changed_callable_symbols(
    response: OpenDiscoveryResponse,
    parent_source: str,
) -> tuple[str, ...]:
    """Return changed child callables absent from a response manifest.

    This is a compiler-facing diagnostic only.  It deliberately derives the
    set from immutable parent/child source ASTs and the parsed manifest rather
    than trusting an error string or model explanation.  A later bounded
    manifest-completion repair may assign *only* these symbols to addresses
    already declared in the frozen child metadata; it may not create code or
    alter the candidate itself.
    """

    if not isinstance(response, OpenDiscoveryResponse):
        raise OpenDiscoveryError(
            "unassigned callable diagnostic requires an OpenDiscoveryResponse"
        )
    _assert_python_source(parent_source, name="parent_source")
    parent_sources = _top_level_callable_sources(
        parent_source, name="parent_source"
    )
    child_sources = _top_level_callable_sources(
        response.candidate_source, name="child_source"
    )
    changed_child_symbols = {
        symbol
        for symbol in child_sources
        if parent_sources.get(symbol) != child_sources.get(symbol)
    }
    assigned = {
        symbol
        for edit in response.change_manifest.edits
        for symbol in (*edit.symbol_patch.parent_symbols, *edit.symbol_patch.child_symbols)
    }
    return tuple(sorted(changed_child_symbols.difference(assigned)))


def validate_manifest_child_symbol_address_bindings(
    response: OpenDiscoveryResponse,
    parent_source: str,
) -> None:
    """Bind each changed child symbol to an address declared by child metadata.

    Source execution alone cannot tell whether a changed callable was declared as
    an objective, fusion, or response-program modification.  The immutable
    candidate metadata provides that semantic binding.  Fixed Candidate API
    wrappers and a compiler-inferred direct root wrapper are exceptions because
    they are structural plumbing rather than scientific candidate components.
    """

    if not isinstance(response, OpenDiscoveryResponse):
        raise OpenDiscoveryError(
            "manifest address binding requires an OpenDiscoveryResponse"
        )
    _assert_python_source(parent_source, name="parent_source")
    component_symbols = response.candidate_metadata.get("component_symbols")
    if not isinstance(component_symbols, Mapping):  # Defensive after parser validation.
        raise OpenDiscoveryError("candidate metadata lacks component_symbols")
    addresses_by_symbol: dict[str, set[str]] = {}
    for address, symbols in component_symbols.items():
        if not isinstance(address, str) or not isinstance(symbols, (list, tuple)):
            raise OpenDiscoveryError("candidate metadata component symbol map is invalid")
        for symbol in symbols:
            if isinstance(symbol, str):
                addresses_by_symbol.setdefault(symbol, set()).add(address)
    # ``candidate_api_wrapper_symbols`` deliberately includes every direct
    # build-model return class.  That broad classification is useful while
    # normalizing API glue, but it would hide a genuine root-architecture
    # change from this semantic-address check.  The compiler's narrower,
    # transition-specific support classifier is the authoritative exception:
    # a changed root is skipped only when it merely wires separately changed
    # scientific children.  A root changed in its own right must be registered
    # at an immutable metadata address like every other scientific callable.
    api_or_root_support = set(
        compiler_support_symbols(parent_source, response.candidate_source)
    ).union({"build_model", "build_scheduler", "step_scheduler"})
    for edit in response.change_manifest.edits:
        for symbol in edit.symbol_patch.child_symbols:
            if symbol in api_or_root_support:
                continue
            if edit.address not in addresses_by_symbol.get(symbol, set()):
                raise OpenDiscoveryError(
                    "child symbol is not bound to its manifest semantic address: "
                    f"edit={edit.edit_id!r}, symbol={symbol!r}, address={edit.address!r}"
                )


def unassigned_parent_only_callable_symbols(
    response: OpenDiscoveryResponse,
    parent_source: str,
) -> tuple[str, ...]:
    """Return omitted parent-only callable deltas that cannot be safely repaired.

    The bounded manifest-completion protocol can only register a changed child
    callable at an address already declared in immutable child metadata. A
    deletion or rename also leaves an old parent-only symbol; representing that
    correctly requires a distinct paired rename/delete protocol. Rather than
    spend LLM calls on an impossible child-only repair, the formal runner uses
    this diagnostic to reject those proposals before training.
    """

    if not isinstance(response, OpenDiscoveryResponse):
        raise OpenDiscoveryError(
            "unassigned callable diagnostic requires an OpenDiscoveryResponse"
        )
    _assert_python_source(parent_source, name="parent_source")
    parent_sources = _top_level_callable_sources(
        parent_source, name="parent_source"
    )
    child_sources = _top_level_callable_sources(
        response.candidate_source, name="child_source"
    )
    assigned = {
        symbol
        for edit in response.change_manifest.edits
        for symbol in (*edit.symbol_patch.parent_symbols, *edit.symbol_patch.child_symbols)
    }
    parent_changed = {
        symbol
        for symbol in parent_sources
        if parent_sources.get(symbol) != child_sources.get(symbol)
    }
    child_changed = {
        symbol
        for symbol in child_sources
        if parent_sources.get(symbol) != child_sources.get(symbol)
    }
    return tuple(
        sorted(parent_changed.difference(assigned).difference(child_changed))
    )


def changed_noncallable_module_statement_kinds(
    parent_source: str,
    child_source: str,
) -> tuple[str, ...]:
    """Describe unrepresentable module-level deltas for early formal rejection.

    The executable patch compiler represents class/function edits. Imports and
    module assignments are intentionally outside its action language; allowing
    them to drift would make exact reconstruction fail only after a model call
    or, worse, create a source dependency not carried by the candidate receipt. A
    narrative module docstring is excluded because it has no executable role.
    """

    def noncallable_identities(source: str, *, label: str) -> tuple[tuple[str, str], ...]:
        _assert_python_source(source, name=label)
        tree = ast.parse(source, filename=f"<{label}>", mode="exec")
        statements: list[tuple[str, str]] = []
        for index, statement in enumerate(tree.body):
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if (
                index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                kind = "import"
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                kind = "assignment"
            else:
                kind = type(statement).__name__.lower()
            statements.append(
                (
                    kind,
                    ast.dump(statement, annotate_fields=True, include_attributes=False),
                )
            )
        return tuple(statements)

    before = noncallable_identities(parent_source, label="parent_source")
    after = noncallable_identities(child_source, label="child_source")
    if before == after:
        return ()
    kinds = {kind for kind, _ in before}.union(kind for kind, _ in after)
    return tuple(sorted(kinds))


def _symbol_operations_from_patch(
    patch: SymbolPatch,
    *,
    parent_sources: Mapping[str, str],
    child_sources: Mapping[str, str],
    edit_id: str,
) -> tuple[TopLevelSymbolOperation, ...]:
    """Translate a declarative symbol patch into compiler-owned operations."""

    parent_symbols = tuple(sorted(patch.parent_symbols))
    child_symbols = tuple(sorted(patch.child_symbols))
    missing_parent = set(parent_symbols).difference(parent_sources)
    missing_child = set(child_symbols).difference(child_sources)
    if missing_parent:
        raise OpenDiscoveryError(
            f"symbol_patch for {edit_id!r} cannot extract parent symbols: {sorted(missing_parent)}"
        )
    if missing_child:
        raise OpenDiscoveryError(
            f"symbol_patch for {edit_id!r} cannot extract child symbols: {sorted(missing_child)}"
        )

    operations: list[TopLevelSymbolOperation] = []
    if patch.operation == "add":
        operations.extend(
            TopLevelSymbolOperation("add", symbol, child_sources[symbol])
            for symbol in child_symbols
        )
    elif patch.operation == "remove":
        operations.extend(
            TopLevelSymbolOperation("delete", symbol, None)
            for symbol in parent_symbols
        )
    else:  # replace/modify: same names replace; differing names become rename ops.
        shared = tuple(sorted(set(parent_symbols).intersection(child_symbols)))
        operations.extend(
            TopLevelSymbolOperation("replace", symbol, child_sources[symbol])
            for symbol in shared
        )
        operations.extend(
            TopLevelSymbolOperation("delete", symbol, None)
            for symbol in sorted(set(parent_symbols).difference(child_symbols))
        )
        operations.extend(
            TopLevelSymbolOperation("add", symbol, child_sources[symbol])
            for symbol in sorted(set(child_symbols).difference(parent_symbols))
        )
    return tuple(operations)


def _assert_exact_callable_change_coverage(
    *,
    parent_sources: Mapping[str, str],
    child_sources: Mapping[str, str],
    edits: Sequence[PatchEditProposal],
    allowed_unchanged_support_symbols: Sequence[str] = (),
) -> None:
    """Require each changed top-level callable to belong to exactly one edit."""

    changed = {
        symbol
        for symbol in set(parent_sources).union(child_sources)
        if parent_sources.get(symbol) != child_sources.get(symbol)
    }
    allowed_unchanged = set(allowed_unchanged_support_symbols)
    assigned: list[str] = [
        operation.symbol_name
        for edit in edits
        for operation in edit.symbol_operations
        if (
            operation.symbol_name in changed
            or operation.symbol_name not in allowed_unchanged
        )
    ]
    duplicate_assignments = sorted(
        symbol for symbol in set(assigned) if assigned.count(symbol) != 1
    )
    if duplicate_assignments:
        raise OpenDiscoveryError(
            "each changed top-level callable must belong to exactly one edit; duplicate assignments: "
            f"{duplicate_assignments}"
        )
    assigned_set = set(assigned)
    missing = sorted(changed.difference(assigned_set))
    extra = sorted(assigned_set.difference(changed))
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"unassigned changed callables {missing}")
        if extra:
            detail.append(f"unchanged callables assigned to an edit {extra}")
        raise OpenDiscoveryError(
            "top-level callable change coverage is not exact: " + "; ".join(detail)
        )


def executable_patch_proposal_from_response(
    response: OpenDiscoveryResponse,
    parent_source: str,
    *,
    skip_address_binding: bool = False,
) -> ExecutablePatchManifestProposal:
    """Derive a deterministic materializer proposal from one model response.

    The response supplies only semantic edit declarations, symbol names and
    per-edit configuration patches. The compiler extracts all ``symbol_source``
    fragments from the full child source itself and never accepts an LLM replay
    claim or hand-authored executable operation list.
    """

    if not isinstance(response, OpenDiscoveryResponse):
        raise OpenDiscoveryError("executable patch compiler requires an OpenDiscoveryResponse")
    _assert_python_source(parent_source, name="parent_source")
    expected_patch = _merged_edit_config_patch(response.change_manifest.edits)
    if canonical_json(_thaw(response.training_config_patch)) != canonical_json(
        _thaw(expected_patch)
    ):
        raise OpenDiscoveryError(
            "training_config_patch must equal the dependency-ordered merge of all edit config_merge_patch values"
        )
    if not isinstance(skip_address_binding, bool):
        raise TypeError("skip_address_binding must be boolean")
    if not skip_address_binding:
        validate_manifest_child_symbol_address_bindings(response, parent_source)
    parent_sources = _top_level_callable_sources(parent_source, name="parent_source")
    child_sources = _top_level_callable_sources(response.candidate_source, name="child_source")
    support_symbols = candidate_api_wrapper_symbols(
        parent_source, response.candidate_source
    )
    try:
        edits = tuple(
            PatchEditProposal(
                edit_id=edit.edit_id,
                address=edit.address,
                kind=edit.kind,
                description=edit.description,
                dependencies=edit.dependencies,
                incompatibilities=edit.incompatibilities,
                preconditions=edit.preconditions,
                symbol_operations=_symbol_operations_from_patch(
                    edit.symbol_patch,
                    parent_sources=parent_sources,
                    child_sources=child_sources,
                    edit_id=edit.edit_id,
                ),
                config_merge_patch=_thaw(edit.config_merge_patch),
            )
            for edit in _topological_change_edits(response.change_manifest.edits)
        )
        manifest_id = "manifest_" + digest(
            {
                "parent_candidate_hash": response.parent_candidate_hash,
                "child_source_hash": response.source_hash,
                "change_manifest_hash": response.change_manifest.manifest_hash,
            }
        )[:20]
        _assert_exact_callable_change_coverage(
            parent_sources=parent_sources,
            child_sources=child_sources,
            edits=edits,
            allowed_unchanged_support_symbols=tuple(sorted(support_symbols)),
        )
        proposal = ExecutablePatchManifestProposal(
            manifest_id=manifest_id,
            edits=edits,
        )
        return normalize_patch_manifest(
            proposal,
            parent_source=parent_source,
            observed_child_source=response.candidate_source,
            infer_support_dependencies=False,
        )
    except RevisionMaterializationError as exc:
        raise OpenDiscoveryError(f"could not derive executable patch proposal: {exc}") from exc


def record_open_discovery_response(
    request: OpenDiscoveryRoundRequest,
    *,
    raw_response: str,
    source_model: str,
) -> OpenDiscoveryRoundTrace:
    """Record raw interaction material even if the response fails validation."""

    if not isinstance(request, OpenDiscoveryRoundRequest):
        raise OpenDiscoveryError("record_open_discovery_response requires a round request")
    prompt = build_open_discovery_prompt(request)
    raw = _text(raw_response, name="raw_response", allow_empty=True, limit=_MAX_RESPONSE_CHARS)
    try:
        payload, response_normalizations = _decode_open_discovery_response_payload(raw)
        parsed = _parse_open_discovery_response_payload(payload, request=request)
        child = parsed.child_candidate(request.current_candidate)
        error: str | None = None
    except OpenDiscoveryError as exc:
        parsed = None
        child = None
        response_normalizations = ()
        error = f"{type(exc).__name__}: {exc}"
    return OpenDiscoveryRoundTrace(
        round_index=request.round_index,
        source_model=source_model,
        request_hash=request.request_hash,
        protected_task_hash=request.protected_task.protected_task_hash,
        initial_entry_hash=request.fixed_initial_entry.initial_entry_hash,
        current_candidate_hash=request.current_candidate.candidate_hash,
        fold3_diagnostics_hash=request.fold3_diagnostics.diagnostics_hash,
        fold3_diagnostics=request.fold3_diagnostics,
        prior_trajectory_hash=request.prior_trajectory_hash,
        raw_prompt=prompt,
        raw_response=raw,
        parsed_response=parsed,
        child_candidate=child,
        parse_error=error,
        response_normalizations=response_normalizations,
    )


def append_open_discovery_response(
    trajectory: OpenDiscoveryTrajectory,
    *,
    diagnostics: Fold3Diagnostics,
    raw_response: str,
    source_model: str,
) -> tuple[OpenDiscoveryTrajectory, OpenDiscoveryRoundTrace]:
    """Build one complete request, record its response, and extend the trace."""

    if not isinstance(trajectory, OpenDiscoveryTrajectory):
        raise OpenDiscoveryError("append_open_discovery_response requires a trajectory")
    request = trajectory.next_request(diagnostics)
    trace = record_open_discovery_response(
        request,
        raw_response=raw_response,
        source_model=source_model,
    )
    return trajectory.append(trace), trace


__all__ = [
    "FOLD3_PARTITION",
    "MAX_DISCOVERY_MANIFEST_EDITS",
    "MAX_INITIAL_DISCOVERY_MANIFEST_EDITS",
    "MAX_PROMPT_HISTORY_ADDRESS_COUNT",
    "MAX_PROMPT_HISTORY_HYPOTHESIS_CHARS",
    "MAX_PROMPT_HISTORY_METRIC_COUNT",
    "MAX_PROMPT_HISTORY_RECORD_CHARS",
    "MAX_PROMPT_HISTORY_SUMMARY_CHARS",
    "OPEN_DISCOVERY_RESPONSE_SCHEMA_VERSION",
    "OPEN_DISCOVERY_SCHEMA_VERSION",
    "PROMPT_HISTORY_SCHEMA_VERSION",
    "UNEXECUTED",
    "CandidateState",
    "ChangeEdit",
    "ChangeManifest",
    "FixedInitialEntry",
    "Fold3Diagnostics",
    "OpenDiscoveryError",
    "OpenDiscoveryResponse",
    "OpenDiscoveryRoundRequest",
    "OpenDiscoveryRoundTrace",
    "OpenDiscoveryTrajectory",
    "ProtectedDiscoveryTask",
    "SymbolPatch",
    "append_open_discovery_response",
    "build_open_discovery_prompt",
    "candidate_history_summary",
    "changed_noncallable_module_statement_kinds",
    "executable_patch_proposal_from_response",
    "fold3_history_summary",
    "parse_open_discovery_response",
    "prompt_history_memory_contract_hash",
    "prompt_history_memory_specification",
    "proposal_history_summary",
    "record_open_discovery_response",
    "unassigned_changed_callable_symbols",
    "unassigned_parent_only_callable_symbols",
    "validate_manifest_child_symbol_address_bindings",
]
