"""Deterministic materialization for open-discovery candidate changes.

The model-facing proposal contains only top-level symbol operations and JSON
configuration merge patches. It cannot provide compiler-owned hashes or
validation outcomes. This module derives operations from authoritative parent
and child sources and materializes the resulting candidate deterministically.

Only top-level add/replace/delete is supported deliberately. A free-form
change that cannot be represented at that boundary remains one coupled
candidate change instead of being assigned unsupported independent semantics.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from .discovery_candidate import CandidateTaskSpec, DiscoveryCandidateError, validate_candidate_file
from .candidate_change_contracts import (
    BundleReconstruction,
    ManifestEdit,
    Precondition,
    ReplayReport,
    StructuredMultiEditManifest,
    CandidateChangeError,
    validate_semantic_address,
)
from .schemas import canonical_json, digest


PATCH_MANIFEST_PROPOSAL_SCHEMA_VERSION = "cellscientist_executable_patch_manifest_proposal_v1"
MATERIALIZER_VERSION = "cellscientist_top_level_symbol_materializer_v3"
SYMBOL_OPERATIONS = frozenset({"add", "replace", "delete"})
_CANDIDATE_API_SUPPORT_SYMBOLS = frozenset(
    {"candidate_metadata", "build_model"}
)


class RevisionMaterializationError(CandidateChangeError):
    """Raised when a proposed source/config patch cannot be materialized."""


# A dependency-closed probe intentionally applies only part of a coupled
# transition.  Its temporary source can therefore instantiate an old wrapper
# against a new constructor (or vice versa).  Those ordinary Python errors are
# evidence that the atomic edit is not independently replayable; they must not
# abort compilation of the complete, exact bundle.
_PARTIAL_REPLAY_EXCEPTIONS = (
    RevisionMaterializationError,
    DiscoveryCandidateError,
    RuntimeError,
    OSError,
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    NameError,
)


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or not value.replace("_", "a").replace("-", "a").replace(".", "a").isalnum():
        raise RevisionMaterializationError(f"{name} must be a non-empty identifier")
    return value


def _strict_object(
    value: Any,
    *,
    name: str,
    required: set[str] | frozenset[str] | tuple[str, ...],
    allowed: set[str] | frozenset[str] | tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RevisionMaterializationError(f"{name} must be a JSON object with string keys")
    missing = set(required).difference(value)
    unknown = set(value).difference(allowed)
    if missing:
        raise RevisionMaterializationError(f"{name} missing fields: {sorted(missing)}")
    if unknown:
        raise RevisionMaterializationError(f"{name} has unknown fields: {sorted(unknown)}")
    return value


def _json_value(value: Any, *, name: str) -> Any:
    """Return a deep JSON value without arrays/tensors or non-finite values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not value == value or value in {float("inf"), float("-inf")}:
            raise RevisionMaterializationError(f"{name} contains a non-finite float")
        return 0.0 if value == 0.0 else float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RevisionMaterializationError(f"{name} keys must be strings")
            result[key] = _json_value(item, name=f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name=f"{name}[]") for item in value]
    raise RevisionMaterializationError(f"{name} must contain JSON values only")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonicalize_python_source(source: str) -> str:
    """Canonical AST rendering used for full-source provenance hashes."""

    if not isinstance(source, str) or not source.strip():
        raise RevisionMaterializationError("candidate source must be non-empty")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RevisionMaterializationError(f"candidate source has invalid syntax: {exc.msg}") from exc
    return ast.unparse(ast.fix_missing_locations(tree)).strip() + "\n"


def canonical_source_hash(source: str) -> str:
    """Hash the complete canonical source, including its narrative docstring."""

    return _sha256_text(canonicalize_python_source(source))


def _without_module_docstring(tree: ast.Module) -> ast.Module:
    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return ast.Module(body=body, type_ignores=[])


def canonicalize_executable_python_source(source: str) -> str:
    """Canonical executable structure with a narrative module docstring removed."""

    if not isinstance(source, str) or not source.strip():
        raise RevisionMaterializationError("candidate source must be non-empty")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RevisionMaterializationError(
            f"candidate source has invalid syntax: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id == "__doc__"
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "__doc__"
        ):
            raise RevisionMaterializationError(
                "candidate source may not read __doc__; the module docstring is "
                "narrative provenance and is excluded from the executable hash"
            )
    executable = _without_module_docstring(tree)
    return ast.unparse(ast.fix_missing_locations(executable)).strip() + "\n"


def canonical_executable_source_hash(source: str) -> str:
    """Hash executable structure while retaining ``canonical_source_hash`` for provenance."""

    return _sha256_text(canonicalize_executable_python_source(source))


def canonical_config(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_value(value, name="configuration")
    if not isinstance(copied, dict):
        raise RevisionMaterializationError("configuration must be a JSON object")
    return copied


def canonical_config_hash(value: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(canonical_config(value)))


def json_merge_patch(target: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """RFC-7396-style recursive JSON merge patch with canonical JSON inputs."""

    base = canonical_config(target)
    change = canonical_config(patch)

    def merge(left: Any, right: Any) -> Any:
        if not isinstance(right, Mapping):
            return _json_value(right, name="merge value")
        result = dict(left) if isinstance(left, Mapping) else {}
        for key, value in right.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = merge(result.get(key), value)
        return result

    result = merge(base, change)
    assert isinstance(result, dict)
    return canonical_config(result)


def _top_level_symbol_name(statement: ast.stmt) -> str | None:
    if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return statement.name
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
        return statement.targets[0].id
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def _symbol_statement(symbol_name: str, source: str) -> ast.stmt:
    try:
        parsed = ast.parse(source)
    except SyntaxError as exc:
        raise RevisionMaterializationError(f"symbol_source for {symbol_name!r} has invalid syntax: {exc.msg}") from exc
    body = list(parsed.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    if len(body) != 1:
        raise RevisionMaterializationError("symbol_source must define exactly one top-level symbol")
    statement = body[0]
    actual_name = _top_level_symbol_name(statement)
    if actual_name != symbol_name:
        raise RevisionMaterializationError(
            f"symbol_source must define top-level symbol {symbol_name!r}, got {actual_name!r}"
        )
    return statement


@dataclass(frozen=True)
class TopLevelSymbolOperation:
    """One syntactically constrained top-level source operation."""

    operation: str
    symbol_name: str
    symbol_source: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in SYMBOL_OPERATIONS:
            raise RevisionMaterializationError(f"unknown symbol operation {self.operation!r}")
        if not isinstance(self.symbol_name, str) or not self.symbol_name.isidentifier():
            raise RevisionMaterializationError("symbol_name must be a valid Python identifier")
        if self.operation == "delete":
            if self.symbol_source is not None:
                raise RevisionMaterializationError("delete symbol operation must use null symbol_source")
        else:
            if not isinstance(self.symbol_source, str) or not self.symbol_source.strip():
                raise RevisionMaterializationError("add/replace symbol operation requires symbol_source")
            _symbol_statement(self.symbol_name, self.symbol_source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "symbol_name": self.symbol_name,
            "symbol_source": self.symbol_source,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "TopLevelSymbolOperation":
        record = _strict_object(
            payload,
            name="symbol_operation",
            required={"operation", "symbol_name", "symbol_source"},
            allowed={"operation", "symbol_name", "symbol_source"},
        )
        return cls(
            operation=str(record["operation"]),
            symbol_name=str(record["symbol_name"]),
            symbol_source=record["symbol_source"],
        )


@dataclass(frozen=True)
class PatchEditProposal:
    """LLM-proposed edit content, deliberately without replay claims/hashes."""

    edit_id: str
    address: str
    kind: str
    description: str
    dependencies: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    preconditions: tuple[Precondition, ...]
    symbol_operations: tuple[TopLevelSymbolOperation, ...]
    config_merge_patch: Mapping[str, Any]
    support_only: bool = False

    def __post_init__(self) -> None:
        _identifier(self.edit_id, name="edit_id")
        validate_semantic_address(self.address)
        _identifier(self.kind, name="kind")
        if not isinstance(self.description, str) or not self.description.strip():
            raise RevisionMaterializationError("edit description must be non-empty")
        if not isinstance(self.dependencies, (list, tuple)) or not isinstance(self.incompatibilities, (list, tuple)):
            raise RevisionMaterializationError("dependencies and incompatibilities must be arrays")
        dependencies = tuple(_identifier(item, name="dependencies[]") for item in self.dependencies)
        incompatibilities = tuple(_identifier(item, name="incompatibilities[]") for item in self.incompatibilities)
        if len(set(dependencies)) != len(dependencies) or len(set(incompatibilities)) != len(incompatibilities):
            raise RevisionMaterializationError("edit dependency/incompatibility IDs must be unique")
        if self.edit_id in dependencies or self.edit_id in incompatibilities:
            raise RevisionMaterializationError("edit may not reference itself")
        if set(dependencies).intersection(incompatibilities):
            raise RevisionMaterializationError("edit cannot both depend on and conflict with another edit")
        if not all(isinstance(item, Precondition) for item in self.preconditions):
            raise RevisionMaterializationError("edit preconditions must be typed")
        if not all(isinstance(item, TopLevelSymbolOperation) for item in self.symbol_operations):
            raise RevisionMaterializationError("edit symbol_operations must be typed")
        if not self.symbol_operations and not self.config_merge_patch:
            raise RevisionMaterializationError("edit must modify source or configuration")
        if not isinstance(self.support_only, bool):
            raise RevisionMaterializationError("support_only must be boolean")
        object.__setattr__(self, "dependencies", tuple(sorted(dependencies)))
        object.__setattr__(self, "incompatibilities", tuple(sorted(incompatibilities)))
        object.__setattr__(self, "config_merge_patch", canonical_config(self.config_merge_patch))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "address": self.address,
            "kind": self.kind,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "incompatibilities": list(self.incompatibilities),
            "preconditions": [item.to_dict() for item in self.preconditions],
            "symbol_operations": [item.to_dict() for item in self.symbol_operations],
            "config_merge_patch": canonical_config(self.config_merge_patch),
            "support_only": self.support_only,
        }

    @property
    def source_patch_hash(self) -> str:
        return _sha256_text(canonical_json([item.to_dict() for item in self.symbol_operations]))

    @property
    def config_patch_hash(self) -> str:
        return _sha256_text(canonical_json(canonical_config(self.config_merge_patch)))

    @classmethod
    def from_dict(cls, payload: Any) -> "PatchEditProposal":
        fields = {
            "edit_id",
            "address",
            "kind",
            "description",
            "dependencies",
            "incompatibilities",
            "preconditions",
            "symbol_operations",
            "config_merge_patch",
            "support_only",
        }
        legacy_fields = fields.difference({"support_only"})
        record = _strict_object(
            payload,
            name="patch_edit",
            required=legacy_fields,
            allowed=fields,
        )
        if not isinstance(record["preconditions"], (list, tuple)) or not isinstance(record["symbol_operations"], (list, tuple)):
            raise RevisionMaterializationError("patch_edit preconditions and symbol_operations must be arrays")
        if not isinstance(record["config_merge_patch"], Mapping):
            raise RevisionMaterializationError("patch_edit.config_merge_patch must be an object")
        return cls(
            edit_id=_identifier(record["edit_id"], name="edit_id"),
            address=validate_semantic_address(record["address"]),
            kind=_identifier(record["kind"], name="kind"),
            description=str(record["description"]),
            dependencies=tuple(record["dependencies"]),
            incompatibilities=tuple(record["incompatibilities"]),
            preconditions=tuple(Precondition.from_dict(item) for item in record["preconditions"]),
            symbol_operations=tuple(TopLevelSymbolOperation.from_dict(item) for item in record["symbol_operations"]),
            config_merge_patch=record["config_merge_patch"],
            support_only=record.get("support_only", False),
        )


@dataclass(frozen=True)
class ExecutablePatchManifestProposal:
    """Strict LLM-facing proposal; compiler-only fields are intentionally absent."""

    manifest_id: str
    edits: tuple[PatchEditProposal, ...]

    def __post_init__(self) -> None:
        _identifier(self.manifest_id, name="manifest_id")
        if not self.edits or not all(isinstance(item, PatchEditProposal) for item in self.edits):
            raise RevisionMaterializationError("patch manifest requires at least one typed edit")
        ids = [item.edit_id for item in self.edits]
        if len(set(ids)) != len(ids):
            raise RevisionMaterializationError("patch manifest has duplicate edit IDs")
        known = set(ids)
        for edit in self.edits:
            if not set(edit.dependencies).issubset(known) or not set(edit.incompatibilities).issubset(known):
                raise RevisionMaterializationError("patch manifest edit references an unknown dependency/incompatibility")
        object.__setattr__(self, "edits", tuple(sorted(self.edits, key=lambda item: item.edit_id)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PATCH_MANIFEST_PROPOSAL_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "edits": [item.to_dict() for item in self.edits],
        }

    @property
    def proposal_hash(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> "ExecutablePatchManifestProposal":
        fields = {"schema_version", "manifest_id", "edits"}
        record = _strict_object(payload, name="patch_manifest", required=fields, allowed=fields)
        if record["schema_version"] != PATCH_MANIFEST_PROPOSAL_SCHEMA_VERSION:
            raise RevisionMaterializationError("patch manifest has an unsupported schema_version")
        if not isinstance(record["edits"], (list, tuple)):
            raise RevisionMaterializationError("patch_manifest.edits must be an array")
        return cls(
            manifest_id=_identifier(record["manifest_id"], name="manifest_id"),
            edits=tuple(PatchEditProposal.from_dict(item) for item in record["edits"]),
        )


def _named_statements(source: str, *, name: str) -> Mapping[str, ast.stmt]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RevisionMaterializationError(
            f"{name} has invalid syntax: {exc.msg}"
        ) from exc
    result: dict[str, ast.stmt] = {}
    for statement in tree.body:
        symbol = _top_level_symbol_name(statement)
        if symbol is None:
            continue
        if symbol in result:
            raise RevisionMaterializationError(
                f"{name} has duplicate top-level symbol {symbol!r}"
            )
        result[symbol] = statement
    return result


def _statement_source(statement: ast.stmt) -> str:
    return ast.unparse(ast.fix_missing_locations(statement)).strip() + "\n"


def _statement_identity(statement: ast.stmt) -> str:
    return ast.dump(statement, annotate_fields=True, include_attributes=False)


def _build_model_constructor_symbol(
    statements: Mapping[str, ast.stmt],
) -> str | None:
    """Return the direct top-level constructor used by ``build_model``.

    The root model class is candidate-API glue: its job is to wire the named
    scientific components into the executor's fixed ``build_model`` entry
    point.  We infer it only for the unambiguous direct-return form, otherwise
    leave it independently selectable rather than guessing.
    """

    builder = statements.get("build_model")
    if not isinstance(builder, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    returns = [node for node in ast.walk(builder) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        return None
    callee = returns[0].value.func
    if not isinstance(callee, ast.Name) or callee.id not in statements:
        return None
    return callee.id


def candidate_api_wrapper_symbols(*sources: str) -> frozenset[str]:
    """Infer fixed API symbols plus direct root-model wrappers from source.

    This is compiler-owned classification.  It does not rely on an LLM's edit
    kind or semantic address, and it only marks a root wrapper when both the
    source AST and the registered ``build_model`` function make that relation
    explicit.
    """

    symbols = set(_CANDIDATE_API_SUPPORT_SYMBOLS)
    for index, source in enumerate(sources):
        statements = _named_statements(source, name=f"candidate_source[{index}]")
        constructor = _build_model_constructor_symbol(statements)
        if constructor is not None:
            symbols.add(constructor)
    return frozenset(symbols)


def _top_level_load_names(statement: ast.stmt) -> frozenset[str]:
    return frozenset(
        node.id
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )


def compiler_support_symbols(
    parent_source: str,
    observed_child_source: str,
) -> frozenset[str]:
    """Return source-proven structural support symbols for one transition.

    ``candidate_metadata`` is always provenance-only. A changed root model
    class is support only in the conservative case where an unchanged, direct
    ``build_model`` return still points to the same root class and that class
    merely wires at least one separately changed scientific component. A root
    class changed on its own (for example to alter hidden width) remains a
    scientific primary rather than being hidden as wrapper plumbing.
    """

    parent = _named_statements(parent_source, name="parent_source")
    child = _named_statements(observed_child_source, name="observed_child_source")
    support = {"candidate_metadata"}
    parent_root = _build_model_constructor_symbol(parent)
    child_root = _build_model_constructor_symbol(child)
    builders_match = (
        "build_model" in parent
        and "build_model" in child
        and _statement_identity(parent["build_model"])
        == _statement_identity(child["build_model"])
    )
    if not builders_match or parent_root is None or parent_root != child_root:
        return frozenset(support)
    root = child_root
    if root not in parent or root not in child:
        return frozenset(support)
    root_changed = _statement_identity(parent[root]) != _statement_identity(child[root])
    if not root_changed:
        return frozenset(support)
    changed_scientific = {
        symbol
        for symbol in set(parent).union(child)
        if symbol not in _CANDIDATE_API_SUPPORT_SYMBOLS
        and (
            symbol not in parent
            or symbol not in child
            or _statement_identity(parent[symbol])
            != _statement_identity(child[symbol])
        )
    }
    if _top_level_load_names(child[root]).intersection(changed_scientific):
        support.add(root)
    return frozenset(support)


def _support_only_edit(edit: PatchEditProposal) -> bool:
    """Return the compiler-owned support classification of an edit."""

    return bool(edit.support_only)


def _depends_on(
    edit_id: str,
    target_id: str,
    *,
    by_id: Mapping[str, PatchEditProposal],
) -> bool:
    pending = list(by_id[edit_id].dependencies)
    visited: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency == target_id:
            return True
        if dependency in visited:
            continue
        visited.add(dependency)
        pending.extend(by_id[dependency].dependencies)
    return False


def _support_symbol_dependencies(
    edit: PatchEditProposal,
    *,
    symbol_owner: Mapping[str, str],
) -> set[str]:
    """Infer only the scientific edits directly referenced by support code.

    A candidate root wrapper often instantiates a newly introduced fusion or
    response component.  That relation is visible in the frozen AST and is
    narrower than making every metadata/wrapper update depend on every edit in
    the bundle.  Unknown Python names are deliberately ignored; exact dummy
    validation remains the final feasibility check.
    """

    dependencies: set[str] = set()
    for operation in edit.symbol_operations:
        if operation.symbol_source is None:
            continue
        try:
            tree = ast.parse(operation.symbol_source)
        except SyntaxError as exc:  # Defensive: proposal already validated.
            raise RevisionMaterializationError(
                f"support symbol {operation.symbol_name!r} has invalid syntax: {exc.msg}"
            ) from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            owner = symbol_owner.get(node.id)
            if owner is not None and owner != edit.edit_id:
                dependencies.add(owner)
        if operation.symbol_name == "candidate_metadata":
            # Candidate metadata is a literal provenance map in the registered
            # API. Its component-symbol strings are not AST ``Name`` nodes,
            # but they still bind the metadata update to the executable edits
            # it describes. Restricting this to strings that exactly match a
            # changed top-level symbol avoids inferring dependencies from prose.
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                ):
                    continue
                owner = symbol_owner.get(node.value)
                if owner is not None and owner != edit.edit_id:
                    dependencies.add(owner)
    return dependencies


def normalize_patch_manifest(
    proposal: ExecutablePatchManifestProposal,
    *,
    parent_source: str,
    observed_child_source: str,
    infer_support_dependencies: bool = True,
) -> ExecutablePatchManifestProposal:
    """Compile symbol operations from authoritative parent/child ASTs.

    The LLM assigns changed symbols to semantic edits, but it does not control
    whether a symbol is an add, replacement, or deletion.  That fact is
    deterministic from the full parent and child sources. Candidate-API-only
    metadata/build-model/root-wrapper plumbing is dependency-locked to the
    scientific edits in the same bundle so it cannot masquerade as a independently selectable
    primary. An unchanged API wrapper mistakenly named by an LLM is dropped
    deterministically; an unknown unchanged symbol remains a hard error.
    """

    if not isinstance(proposal, ExecutablePatchManifestProposal):
        raise RevisionMaterializationError(
            "normalization requires an ExecutablePatchManifestProposal"
        )
    parent = _named_statements(parent_source, name="parent_source")
    child = _named_statements(observed_child_source, name="observed_child_source")
    ignorable_wrapper_symbols = candidate_api_wrapper_symbols(
        parent_source, observed_child_source
    )
    support_symbols = compiler_support_symbols(
        parent_source, observed_child_source
    )
    changed = {
        symbol
        for symbol in set(parent).union(child)
        if (
            symbol not in parent
            or symbol not in child
            or _statement_identity(parent[symbol])
            != _statement_identity(child[symbol])
        )
    }
    declared_operations = {
        edit.edit_id: tuple(
            operation
            for operation in edit.symbol_operations
            if (
                operation.symbol_name in changed
                or operation.symbol_name not in ignorable_wrapper_symbols
            )
        )
        for edit in proposal.edits
    }
    empty_support_declarations = [
        edit.edit_id
        for edit in proposal.edits
        if not declared_operations[edit.edit_id] and not edit.config_merge_patch
    ]
    if empty_support_declarations:
        raise RevisionMaterializationError(
            "manifest contains a no-op candidate API support edit: "
            + ", ".join(sorted(empty_support_declarations))
        )
    assigned: list[str] = [
        operation.symbol_name
        for edit in proposal.edits
        for operation in declared_operations[edit.edit_id]
    ]
    duplicates = sorted(
        symbol for symbol in set(assigned) if assigned.count(symbol) != 1
    )
    if duplicates:
        raise RevisionMaterializationError(
            "changed top-level symbols must belong to exactly one edit; "
            f"duplicate assignments: {duplicates}"
        )
    missing = sorted(changed.difference(assigned))
    extra = sorted(set(assigned).difference(changed))
    if missing or extra:
        details = []
        if missing:
            details.append(f"unassigned changed symbols {missing}")
        if extra:
            details.append(f"unchanged or absent symbols assigned to edits {extra}")
        raise RevisionMaterializationError(
            "top-level symbol coverage is not exact: " + "; ".join(details)
        )

    normalized_edits: list[PatchEditProposal] = []
    for edit in proposal.edits:
        operations: list[TopLevelSymbolOperation] = []
        for declared in declared_operations[edit.edit_id]:
            symbol = declared.symbol_name
            in_parent = symbol in parent
            in_child = symbol in child
            if in_parent and in_child:
                operation = "replace"
                symbol_source = _statement_source(child[symbol])
            elif in_child:
                operation = "add"
                symbol_source = _statement_source(child[symbol])
            elif in_parent:
                operation = "delete"
                symbol_source = None
            else:  # Defensive; exact coverage already rejects this case.
                raise RevisionMaterializationError(
                    f"assigned symbol {symbol!r} exists in neither source"
                )
            operations.append(
                TopLevelSymbolOperation(operation, symbol, symbol_source)
            )
        normalized_edits.append(
            PatchEditProposal(
                edit_id=edit.edit_id,
                address=edit.address,
                kind=edit.kind,
                description=edit.description,
                dependencies=edit.dependencies,
                incompatibilities=edit.incompatibilities,
                preconditions=edit.preconditions,
                symbol_operations=tuple(operations),
                config_merge_patch=edit.config_merge_patch,
                support_only=(
                    bool(operations)
                    and {
                        operation.symbol_name for operation in operations
                    }.issubset(support_symbols)
                    and not edit.config_merge_patch
                ),
            )
        )

    if infer_support_dependencies:
        by_id = {edit.edit_id: edit for edit in normalized_edits}
        support_ids = {
            edit.edit_id for edit in normalized_edits if _support_only_edit(edit)
        }
        scientific_ids = set(by_id).difference(support_ids)
        symbol_owner = {
            operation.symbol_name: edit.edit_id
            for edit in normalized_edits
            for operation in edit.symbol_operations
        }
        with_support_dependencies: list[PatchEditProposal] = []
        for edit in normalized_edits:
            dependencies = set(edit.dependencies)
            if edit.edit_id in support_ids:
                for scientific_id in scientific_ids:
                    if _depends_on(
                        scientific_id,
                        edit.edit_id,
                        by_id=by_id,
                    ):
                        raise RevisionMaterializationError(
                            "support-only candidate API plumbing cannot be a "
                            f"prerequisite of scientific edit {scientific_id!r}"
                        )
                dependencies.update(
                    _support_symbol_dependencies(
                        edit,
                        symbol_owner=symbol_owner,
                    ).intersection(scientific_ids)
                )
            with_support_dependencies.append(
                PatchEditProposal(
                    edit_id=edit.edit_id,
                    address=edit.address,
                    kind=edit.kind,
                    description=edit.description,
                    dependencies=tuple(dependencies),
                    incompatibilities=edit.incompatibilities,
                    preconditions=edit.preconditions,
                    symbol_operations=edit.symbol_operations,
                    config_merge_patch=edit.config_merge_patch,
                    support_only=edit.support_only,
                )
            )
        normalized_edits = with_support_dependencies

    return ExecutablePatchManifestProposal(
        manifest_id=proposal.manifest_id,
        edits=tuple(normalized_edits),
    )


def _topological_order(edits: Sequence[PatchEditProposal]) -> tuple[PatchEditProposal, ...]:
    by_id = {item.edit_id: item for item in edits}
    remaining = set(by_id)
    ordered: list[PatchEditProposal] = []
    while remaining:
        ready = sorted(item for item in remaining if set(by_id[item].dependencies).issubset({x.edit_id for x in ordered}))
        if not ready:
            raise RevisionMaterializationError("patch manifest dependencies contain a cycle or unavailable predecessor")
        for edit_id in ready:
            ordered.append(by_id[edit_id])
            remaining.remove(edit_id)
    return tuple(ordered)


def _apply_symbol_operations(tree: ast.Module, operations: Sequence[TopLevelSymbolOperation]) -> ast.Module:
    body = list(tree.body)

    def index() -> dict[str, int]:
        result: dict[str, int] = {}
        for position, statement in enumerate(body):
            name = _top_level_symbol_name(statement)
            if name is not None:
                if name in result:
                    raise RevisionMaterializationError(f"parent source has duplicate top-level symbol {name!r}")
                result[name] = position
        return result

    for operation in operations:
        locations = index()
        present = operation.symbol_name in locations
        if operation.operation == "add":
            if present:
                raise RevisionMaterializationError(f"cannot add existing top-level symbol {operation.symbol_name!r}")
            assert operation.symbol_source is not None
            body.append(_symbol_statement(operation.symbol_name, operation.symbol_source))
        elif operation.operation == "replace":
            if not present:
                raise RevisionMaterializationError(f"cannot replace missing top-level symbol {operation.symbol_name!r}")
            assert operation.symbol_source is not None
            body[locations[operation.symbol_name]] = _symbol_statement(operation.symbol_name, operation.symbol_source)
        else:
            if not present:
                raise RevisionMaterializationError(f"cannot delete missing top-level symbol {operation.symbol_name!r}")
            body.pop(locations[operation.symbol_name])
    return ast.Module(body=body, type_ignores=[])


def apply_patch_edits(
    parent_source: str,
    parent_config: Mapping[str, Any],
    edits: Sequence[PatchEditProposal],
) -> tuple[str, dict[str, Any]]:
    """Apply a topologically ordered edit sequence without running the source."""

    try:
        tree = ast.parse(canonicalize_python_source(parent_source))
    except RevisionMaterializationError:
        raise
    config = canonical_config(parent_config)
    for edit in edits:
        tree = _apply_symbol_operations(tree, edit.symbol_operations)
        config = json_merge_patch(config, edit.config_merge_patch)
    source = ast.unparse(ast.fix_missing_locations(tree)).strip() + "\n"
    return source, config


def _is_module_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _align_to_observed_child_layout(
    reconstructed_source: str,
    observed_child_source: str,
    *,
    added_symbols: frozenset[str],
) -> str:
    """Use child AST order as a compiler-owned layout anchor.

    All executable statements come from the reconstructed source.  The only
    child-only material copied is the optional narrative module docstring.
    Named statements must already be structurally identical, and imports or
    other unnamed top-level statements are matched by AST identity before they
    can be reordered.  This makes added-symbol placement deterministic without
    trusting the child as an executable patch.
    """

    try:
        reconstructed_tree = ast.parse(reconstructed_source)
        child_tree = ast.parse(observed_child_source)
    except SyntaxError as exc:
        raise RevisionMaterializationError(
            f"cannot align invalid reconstructed/child source: {exc.msg}"
        ) from exc

    reconstructed_body = list(reconstructed_tree.body)
    child_body = list(child_tree.body)
    child_docstring: ast.stmt | None = None
    if child_body and _is_module_docstring(child_body[0]):
        child_docstring = child_body.pop(0)
    if reconstructed_body and _is_module_docstring(reconstructed_body[0]):
        reconstructed_body.pop(0)

    reconstructed_added: dict[str, ast.stmt] = {}
    reconstructed_base: list[ast.stmt] = []
    for statement in reconstructed_body:
        symbol = _top_level_symbol_name(statement)
        if symbol in added_symbols:
            if symbol in reconstructed_added:
                raise RevisionMaterializationError(
                    f"reconstructed source has duplicate added symbol {symbol!r}"
                )
            reconstructed_added[symbol] = statement
        else:
            reconstructed_base.append(statement)
    if set(reconstructed_added) != set(added_symbols):
        raise RevisionMaterializationError(
            "reconstructed source does not contain the normalized added-symbol set"
        )

    child_base = [
        statement
        for statement in child_body
        if _top_level_symbol_name(statement) not in added_symbols
    ]
    if [
        _statement_identity(statement) for statement in reconstructed_base
    ] != [
        _statement_identity(statement) for statement in child_base
    ]:
        raise RevisionMaterializationError(
            "observed child reorders or changes non-added top-level statements"
        )

    aligned: list[ast.stmt] = []
    if child_docstring is not None:
        aligned.append(child_docstring)
    base_index = 0
    seen_added: set[str] = set()
    for child_statement in child_body:
        symbol = _top_level_symbol_name(child_statement)
        if symbol in added_symbols:
            reconstructed_statement = reconstructed_added[symbol]
            if _statement_identity(reconstructed_statement) != _statement_identity(
                child_statement
            ):
                raise RevisionMaterializationError(
                    f"reconstructed added symbol {symbol!r} differs from observed child"
                )
            aligned.append(reconstructed_statement)
            seen_added.add(symbol)
        else:
            aligned.append(reconstructed_base[base_index])
            base_index += 1
    if seen_added != set(added_symbols) or base_index != len(reconstructed_base):
        raise RevisionMaterializationError(
            "child layout anchor did not consume the reconstructed module exactly"
        )

    result = ast.Module(body=aligned, type_ignores=[])
    rendered = ast.unparse(ast.fix_missing_locations(result)).strip() + "\n"
    if canonical_executable_source_hash(rendered) != canonical_executable_source_hash(
        observed_child_source
    ):
        raise RevisionMaterializationError(
            "compiler-aligned executable structure differs from observed child"
        )
    return rendered


def _dependency_closure(edit: PatchEditProposal, by_id: Mapping[str, PatchEditProposal]) -> set[str]:
    result: set[str] = set()

    def visit(edit_id: str) -> None:
        if edit_id in result:
            return
        result.add(edit_id)
        for dependency in by_id[edit_id].dependencies:
            visit(dependency)

    visit(edit.edit_id)
    return result


def _validator_hash(task: CandidateTaskSpec, *, max_parameters: int, batch_size: int) -> str:
    return digest(
        {
            "materializer_version": MATERIALIZER_VERSION,
            "validator": "discovery_candidate.validate_candidate_file",
            "task": task.to_dict(),
            "max_parameters": int(max_parameters),
            "batch_size": int(batch_size),
        }
    )


def _dummy_validate(
    source: str,
    *,
    task: CandidateTaskSpec,
    max_parameters: int,
    batch_size: int,
) -> None:
    """Execute the protected dummy interface validator in an isolated temp file."""

    with tempfile.TemporaryDirectory(prefix="cellscientist-materialize-") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(source, encoding="utf-8")
        validate_candidate_file(
            candidate,
            task=task,
            max_parameters=max_parameters,
            batch_size=batch_size,
        )


@dataclass(frozen=True)
class MaterializationResult:
    """Private compiler result; only ``structured_manifest`` enters the receipt."""

    structured_manifest: StructuredMultiEditManifest
    reconstructed_source_hash: str | None
    reconstructed_config_hash: str | None
    full_validation_status: str
    materializer_hash: str

    def __post_init__(self) -> None:
        if self.full_validation_status not in {"passed", "failed", "not_materializable"}:
            raise RevisionMaterializationError("unknown full_validation_status")
        _identifier(self.materializer_hash, name="materializer_hash")
        if len(self.materializer_hash) != 64:
            raise RevisionMaterializationError("materializer_hash must be SHA-256")
        if self.reconstructed_source_hash is not None and len(self.reconstructed_source_hash) != 64:
            raise RevisionMaterializationError("reconstructed_source_hash must be SHA-256")
        if self.reconstructed_config_hash is not None and len(self.reconstructed_config_hash) != 64:
            raise RevisionMaterializationError("reconstructed_config_hash must be SHA-256")


def materialize_patch_manifest(
    proposal: ExecutablePatchManifestProposal,
    *,
    parent_source: str,
    parent_config: Mapping[str, Any],
    observed_child_source: str,
    observed_child_config: Mapping[str, Any],
    task: CandidateTaskSpec,
    max_parameters: int,
    batch_size: int = 4,
) -> MaterializationResult:
    """Generate compiler-owned replay reports and a receipt-ready manifest.

    Each atomic report validates the edit's dependency closure.  The final
    bundle reconstruction is exact only if applying *all* edits in topological
    order produces canonical source/configuration hashes equal to the observed
    child.  Scores are never requested or returned.
    """

    if not isinstance(proposal, ExecutablePatchManifestProposal):
        raise RevisionMaterializationError("materializer requires an ExecutablePatchManifestProposal")
    if not isinstance(task, CandidateTaskSpec):
        raise RevisionMaterializationError("materializer requires a CandidateTaskSpec")
    if max_parameters <= 0 or batch_size <= 1:
        raise RevisionMaterializationError("max_parameters and batch_size must be positive")
    proposal = normalize_patch_manifest(
        proposal,
        parent_source=parent_source,
        observed_child_source=observed_child_source,
        infer_support_dependencies=True,
    )
    ordered: tuple[PatchEditProposal, ...]
    try:
        ordered = _topological_order(proposal.edits)
    except RevisionMaterializationError:
        ordered = ()
    validator_hash = _validator_hash(task, max_parameters=max_parameters, batch_size=batch_size)
    by_id = {item.edit_id: item for item in proposal.edits}
    reports: dict[str, ReplayReport] = {}
    if ordered:
        for edit in proposal.edits:
            try:
                closure = _dependency_closure(edit, by_id)
                closure_edits = tuple(item for item in ordered if item.edit_id in closure)
                atomic_source, atomic_config = apply_patch_edits(parent_source, parent_config, closure_edits)
                _dummy_validate(
                    atomic_source,
                    task=task,
                    max_parameters=max_parameters,
                    batch_size=batch_size,
                )
                reports[edit.edit_id] = ReplayReport(
                    status="passed",
                    validator_hash=validator_hash,
                    replay_source_hash=canonical_source_hash(atomic_source),
                    replay_config_hash=canonical_config_hash(atomic_config),
                    detail_code="dummy_contract_passed",
                )
            except _PARTIAL_REPLAY_EXCEPTIONS:
                reports[edit.edit_id] = ReplayReport(
                    status="failed",
                    validator_hash=validator_hash,
                    detail_code="dummy_contract_failed",
                )
    else:
        reports = {
            edit.edit_id: ReplayReport(
                status="not_materializable",
                validator_hash=validator_hash,
                detail_code="dependency_cycle",
            )
            for edit in proposal.edits
        }

    reconstructed_source_hash: str | None = None
    reconstructed_config_hash: str | None = None
    full_status = "not_materializable"
    reconstruction_status = "not_materializable"
    try:
        if not ordered:
            raise RevisionMaterializationError("cannot materialize cyclic edit dependencies")
        full_source, full_config = apply_patch_edits(parent_source, parent_config, ordered)
        added_symbols = frozenset(
            operation.symbol_name
            for edit in ordered
            for operation in edit.symbol_operations
            if operation.operation == "add"
        )
        full_source = _align_to_observed_child_layout(
            full_source,
            observed_child_source,
            added_symbols=added_symbols,
        )
        _dummy_validate(full_source, task=task, max_parameters=max_parameters, batch_size=batch_size)
        reconstructed_source_hash = canonical_source_hash(full_source)
        reconstructed_config_hash = canonical_config_hash(full_config)
        observed_source_hash = canonical_source_hash(observed_child_source)
        observed_config_hash = canonical_config_hash(observed_child_config)
        full_status = "passed"
        reconstruction_status = (
            "exact"
            if (
                reconstructed_source_hash == observed_source_hash
                and reconstructed_config_hash == observed_config_hash
            )
            else "dependency_qualified"
        )
    except (RevisionMaterializationError, DiscoveryCandidateError, RuntimeError, OSError):
        full_status = "failed" if ordered else "not_materializable"
        reconstruction_status = "not_materializable"

    structured = StructuredMultiEditManifest(
        manifest_id=proposal.manifest_id,
        edits=tuple(
            ManifestEdit(
                edit_id=edit.edit_id,
                semantic_address=edit.address,
                semantic_interpretation=edit.description,
                source_patch_hash=edit.source_patch_hash,
                config_patch_hash=edit.config_patch_hash,
                preconditions=edit.preconditions,
                dependencies=edit.dependencies,
                incompatibilities=edit.incompatibilities,
                replay_report=reports[edit.edit_id],
            )
            for edit in proposal.edits
        ),
        reconstruction=BundleReconstruction(
            status=reconstruction_status,
            applied_edit_ids=tuple(sorted(edit.edit_id for edit in proposal.edits)),
            reconstructed_source_hash=reconstructed_source_hash if reconstruction_status == "exact" else None,
            reconstructed_config_hash=reconstructed_config_hash if reconstruction_status == "exact" else None,
        ),
    )
    return MaterializationResult(
        structured_manifest=structured,
        reconstructed_source_hash=reconstructed_source_hash,
        reconstructed_config_hash=reconstructed_config_hash,
        full_validation_status=full_status,
        materializer_hash=digest(
            {
                "version": MATERIALIZER_VERSION,
                "proposal_hash": proposal.proposal_hash,
                "validator_hash": validator_hash,
                "structured_manifest_hash": structured.manifest_hash,
                "reconstructed_source_hash": reconstructed_source_hash,
                "reconstructed_config_hash": reconstructed_config_hash,
            }
        ),
    )


def _response_change_manifest(response: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get("change_manifest")
    return getattr(response, "change_manifest", None)


def validate_open_discovery_alignment(
    response: Any,
    proposal: ExecutablePatchManifestProposal,
) -> None:
    """Ensure executable patch content matches the response's public edit plan."""

    manifest = _response_change_manifest(response)
    if manifest is None:
        raise RevisionMaterializationError("open discovery response has no change_manifest")
    raw_edits = getattr(manifest, "edits", None)
    if raw_edits is None and isinstance(manifest, Mapping):
        raw_edits = manifest.get("edits")
    if not isinstance(raw_edits, (list, tuple)):
        raise RevisionMaterializationError("open discovery change_manifest.edits must be an array")
    normalized: dict[str, tuple[str, str, str, tuple[str, ...]]] = {}
    for raw in raw_edits:
        if isinstance(raw, Mapping):
            edit_id, address, kind, description, dependencies = (
                raw.get("edit_id"), raw.get("address"), raw.get("kind"), raw.get("description"), raw.get("dependencies")
            )
        else:
            edit_id, address, kind, description, dependencies = (
                getattr(raw, "edit_id", None),
                getattr(raw, "address", None),
                getattr(raw, "kind", None),
                getattr(raw, "description", None),
                getattr(raw, "dependencies", None),
            )
        if not isinstance(dependencies, (list, tuple)):
            raise RevisionMaterializationError("open discovery edit dependencies must be an array")
        normalized[str(edit_id)] = (
            str(address),
            str(kind),
            str(description),
            # Dependency order is not part of an edit's semantics.  The
            # executable proposal canonicalizes it, so compare the same
            # canonical set here rather than rejecting an otherwise identical
            # LLM declaration that listed predecessors in another order.
            tuple(sorted(str(item) for item in dependencies)),
        )
    if set(normalized) != {edit.edit_id for edit in proposal.edits}:
        raise RevisionMaterializationError("executable patch manifest edit IDs differ from open discovery response")
    for edit in proposal.edits:
        if normalized[edit.edit_id] != (
            edit.address,
            edit.kind,
            edit.description,
            tuple(sorted(edit.dependencies)),
        ):
            raise RevisionMaterializationError("executable patch edit differs from open discovery response manifest")


def materialize_open_discovery_response(
    response: Any,
    proposal: ExecutablePatchManifestProposal,
    **kwargs: Any,
) -> MaterializationResult:
    """Validate response/patch alignment, then run the real local materializer."""

    validate_open_discovery_alignment(response, proposal)
    return materialize_patch_manifest(proposal, **kwargs)
