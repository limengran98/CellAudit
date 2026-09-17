"""Deterministic, source-grounded claim checking for exploratory audits.

This module deliberately contains no provider client and never executes a
candidate.  An open-discovery candidate may *declare* a compact structured
claim manifest in ``candidate_metadata.claim_manifest``.  The checker then
validates the manifest against AST-visible symbols and emits a conservative
test contract.  Unknown structures are marked unresolved rather than guessed.

The checker is the deterministic source-analysis component of the registered
``Discovery -> Check -> Falsify`` path.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .open_discovery import CandidateState
from .schemas import digest


CHECKER_SCHEMA = "cellscientist_deterministic_claim_checker_v1"
CLAIM_MANIFEST_SCHEMA = "cellscientist_perturbation_claim_manifest_v1"

CONTROL_RELIANCE = "control_context_reliance"
CHEMICAL_RELIANCE = "chemical_identity_reliance"
CHEMICAL_CONTROL_INTERACTION = "chemical_control_interaction"
DOSE_RELIANCE = "dose_reliance"
REGISTERED_CLAIMS = (
    CONTROL_RELIANCE,
    CHEMICAL_RELIANCE,
    CHEMICAL_CONTROL_INTERACTION,
    DOSE_RELIANCE,
)

TASK_REQUIRED_CLAIMS = (CONTROL_RELIANCE, CHEMICAL_RELIANCE)
REQUIRED_TESTS = {
    CONTROL_RELIANCE: ("matched_control_swap",),
    CHEMICAL_RELIANCE: ("within_context_chemical_shuffle",),
    CHEMICAL_CONTROL_INTERACTION: (
        "matched_control_swap",
        "within_context_chemical_shuffle",
        "chemical_control_2x2_interaction",
    ),
    DOSE_RELIANCE: ("matched_dose_shuffle",),
}

IMPLEMENTED = "implemented"
IMPLEMENTATION_CONTRADICTED = "implementation-contradicted"
UNRESOLVED = "unresolved"
NOT_CLAIMED = "not-claimed"
STATIC_STATUSES = (
    IMPLEMENTED,
    IMPLEMENTATION_CONTRADICTED,
    UNRESOLVED,
    NOT_CLAIMED,
)


class DeterministicClaimError(ValueError):
    """Raised when the static claim contract is malformed or ambiguous."""


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise DeterministicClaimError(f"{field} must be a Python identifier")
    return value


def _text(value: Any, *, field: str, limit: int = 2_000) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise DeterministicClaimError(f"{field} must be non-empty text")
    value = value.strip()
    if len(value) > limit:
        raise DeterministicClaimError(f"{field} exceeds its length limit")
    return value


def _source_tree(source: str) -> ast.Module:
    try:
        return ast.parse(source, filename="<deterministic-claim-check>", mode="exec")
    except SyntaxError as exc:
        raise DeterministicClaimError("candidate source is not parseable") from exc


def source_symbol_anchors(source: str) -> Mapping[str, str]:
    """Hash each top-level class/function without importing the candidate."""

    tree = _source_tree(source)
    anchors: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        segment = ast.get_source_segment(source, node)
        if not isinstance(segment, str) or not segment.strip():
            raise DeterministicClaimError("cannot recover a source symbol segment")
        anchors[node.name] = digest(segment)
    if not anchors:
        raise DeterministicClaimError("candidate source has no top-level audit symbols")
    return MappingProxyType(dict(sorted(anchors.items())))


def _top_level_nodes(tree: ast.Module) -> Mapping[str, ast.AST]:
    return MappingProxyType({
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    })


def _forward(node: ast.AST) -> ast.FunctionDef | None:
    if not isinstance(node, ast.ClassDef):
        return None
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            return item
    return None


def _forward_arguments(node: ast.AST) -> set[str]:
    forward = _forward(node)
    if forward is None:
        return set()
    return {
        argument.arg
        for argument in (*forward.args.posonlyargs, *forward.args.args, *forward.args.kwonlyargs)
        if argument.arg != "self"
    }


def _loaded_names(node: ast.AST) -> tuple[str, ...]:
    return tuple(
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    )


def _input_is_consumed(node: ast.AST, *, aliases: Sequence[str]) -> bool:
    """Return whether a public fusion input reaches an executable expression.

    The checker deliberately makes only a small, source-local claim here.  A
    parameter that is immediately deleted (the fixed diagnostic controls do
    this explicitly) is not counted as a used input merely because it appears
    in the function signature.  Conversely, passing an input to any nested
    call or returning/combining it is counted as consumption; deeper semantic
    interpretation is left to the behavioral audit.
    """

    accepted = set(aliases)
    return any(
        isinstance(item, ast.Name)
        and isinstance(item.ctx, ast.Load)
        and item.id in accepted
        for item in ast.walk(node)
    )


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_int_one(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == 1


def _is_unsqueeze_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unsqueeze"
        and len(node.args) == 1
        and _is_int_one(node.args[0])
    )


def _call_uses_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == name
        for item in ast.walk(node)
    )


def _singleton_attention_certificate(node: ast.AST) -> Mapping[str, Any] | None:
    """Recognize only a narrow, provable singleton softmax cancellation.

    The certificate is intentionally strict: it establishes that the named
    chemical input appears only in a singleton attention-query path inside the
    cited fusion callable.  It does *not* assert that no other model module
    receives chemical information.
    """

    forward = _forward(node)
    if forward is None:
        return None
    arguments = _forward_arguments(node)
    control_name = "control" if "control" in arguments else "pre" if "pre" in arguments else None
    if control_name is None or "chemical" not in arguments:
        return None

    singleton_origins: dict[str, set[str]] = {}
    has_softmax_last_axis = False
    has_attention_matmul = False
    for item in ast.walk(forward):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            target = item.targets[0].id
            if _is_unsqueeze_one(item.value):
                singleton_origins[target] = {
                    name for name in (control_name, "chemical") if _call_uses_name(item.value, name)
                }
        if isinstance(item, ast.Call):
            func_name = _attribute_name(item.func)
            if func_name == "torch.softmax":
                dim_keyword = next((keyword.value for keyword in item.keywords if keyword.arg == "dim"), None)
                has_softmax_last_axis = isinstance(dim_keyword, ast.UnaryOp) and isinstance(dim_keyword.op, ast.USub) and _is_int_one(dim_keyword.operand)
                has_softmax_last_axis = has_softmax_last_axis or (
                    isinstance(dim_keyword, ast.Constant) and type(dim_keyword.value) is int and dim_keyword.value == -1
                )
            if func_name == "torch.matmul":
                has_attention_matmul = True
    chemical_singletons = [name for name, origins in singleton_origins.items() if origins == {"chemical"}]
    control_singletons = [name for name, origins in singleton_origins.items() if origins == {control_name}]
    chemical_uses = sum(1 for name in _loaded_names(forward) if name == "chemical")
    if not (
        chemical_singletons
        and len(control_singletons) >= 2
        and has_softmax_last_axis
        and has_attention_matmul
        and chemical_uses == 1
    ):
        return None
    return MappingProxyType({
        "rule_id": "singleton_softmax_query_cancellation_v1",
        "proof_scope": (
            "Within this cited fusion callable, chemical occurs only as a singleton query; "
            "softmax over one key has constant weight, so this attention-weight pathway cannot "
            "transmit chemical dependence. Other modules are outside this certificate."
        ),
        "chemical_singleton_variables": tuple(chemical_singletons),
        "control_singleton_variables": tuple(control_singletons),
        "control_argument": control_name,
    })


def _fusion_symbols(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    component_symbols = metadata.get("component_symbols")
    if not isinstance(component_symbols, Mapping):
        raise DeterministicClaimError("candidate metadata has no component-symbol map")
    symbols: list[str] = []
    for address, values in component_symbols.items():
        if not isinstance(address, str) or not (
            "fusion" in address or address.startswith("perturbation.")
        ):
            continue
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            symbols.extend(_identifier(value, field="fusion source symbol") for value in values)
    return tuple(dict.fromkeys(symbols))


@dataclass(frozen=True)
class DeclaredClaim:
    claim_type: str
    statement: str
    source_symbols: tuple[str, ...]
    origin: str

    def __post_init__(self) -> None:
        if self.claim_type not in REGISTERED_CLAIMS:
            raise DeterministicClaimError("claim_type is not registered")
        _text(self.statement, field="claim statement")
        symbols = tuple(_identifier(item, field="claim source symbol") for item in self.source_symbols)
        if not symbols or len(symbols) != len(set(symbols)):
            raise DeterministicClaimError("claim source symbols must be unique and non-empty")
        if self.origin not in {
            "task_contract",
            "candidate_manifest",
            "registered_method_contract",
        }:
            raise DeterministicClaimError("claim origin is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "statement": self.statement,
            "source_symbols": list(self.source_symbols),
            "origin": self.origin,
        }


def _candidate_manifest(metadata: Mapping[str, Any]) -> tuple[DeclaredClaim, ...]:
    raw = metadata.get("claim_manifest")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "claims"}:
        raise DeterministicClaimError("claim_manifest fields are invalid")
    if raw["schema_version"] != CLAIM_MANIFEST_SCHEMA:
        raise DeterministicClaimError("claim_manifest schema_version is unsupported")
    items = raw["claims"]
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise DeterministicClaimError("claim_manifest claims must be an array")
    result: list[DeclaredClaim] = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"claim_type", "statement", "source_symbols"}:
            raise DeterministicClaimError("a manifest claim has invalid fields")
        if (
            not isinstance(item["source_symbols"], Sequence)
            or isinstance(item["source_symbols"], (str, bytes))
            or not item["source_symbols"]
        ):
            raise DeterministicClaimError("manifest claim source_symbols must be a non-empty array")
        result.append(DeclaredClaim(
            claim_type=str(item["claim_type"]),
            statement=_text(item["statement"], field="claim statement"),
            source_symbols=tuple(item["source_symbols"]),
            origin="candidate_manifest",
        ))
    types = [claim.claim_type for claim in result]
    if len(types) != len(set(types)):
        raise DeterministicClaimError("claim_manifest cannot repeat a claim type")
    if any(claim_type in TASK_REQUIRED_CLAIMS for claim_type in types):
        raise DeterministicClaimError("task-required input claims are checker-owned, not model-authored")
    return tuple(result)


def _task_contract_claims(metadata: Mapping[str, Any]) -> tuple[DeclaredClaim, ...]:
    symbols = _fusion_symbols(metadata)
    if not symbols:
        # The universal behavioral maps still run; their source route is simply
        # not identifiable from the candidate's own semantic declaration.
        symbols = ("build_model",)
    return (
        DeclaredClaim(
            claim_type=CONTROL_RELIANCE,
            statement="The task requires prediction to be audited for matched-control reliance.",
            source_symbols=symbols,
            origin="task_contract",
        ),
        DeclaredClaim(
            claim_type=CHEMICAL_RELIANCE,
            statement="The task requires prediction to be audited for chemical-identity reliance.",
            source_symbols=symbols,
            origin="task_contract",
        ),
    )


def _registered_method_claims(
    values: Sequence[Mapping[str, Any]] | None,
) -> tuple[DeclaredClaim, ...]:
    """Validate claims fixed by a method contract rather than model prose."""

    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise DeterministicClaimError("registered_method_claims must be an array")
    result: list[DeclaredClaim] = []
    for item in values:
        if not isinstance(item, Mapping) or set(item) != {
            "claim_type",
            "statement",
            "source_symbols",
        }:
            raise DeterministicClaimError("a registered method claim has invalid fields")
        symbols = item["source_symbols"]
        if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)) or not symbols:
            raise DeterministicClaimError("registered method claim source_symbols must be non-empty")
        result.append(
            DeclaredClaim(
                claim_type=str(item["claim_type"]),
                statement=_text(item["statement"], field="registered method claim statement"),
                source_symbols=tuple(symbols),
                origin="registered_method_contract",
            )
        )
    types = [claim.claim_type for claim in result]
    if len(types) != len(set(types)):
        raise DeterministicClaimError("registered_method_claims cannot repeat a claim type")
    return tuple(result)


def _inspect_claim(claim: DeclaredClaim, *, nodes: Mapping[str, ast.AST]) -> Mapping[str, Any]:
    absent = [symbol for symbol in claim.source_symbols if symbol not in nodes]
    if absent:
        return {
            "static_status": IMPLEMENTATION_CONTRADICTED,
            "reason": "manifest cites symbols absent from candidate source",
            "absent_symbols": absent,
            "findings": [],
        }
    findings: list[Mapping[str, Any]] = []
    if claim.claim_type == CHEMICAL_CONTROL_INTERACTION:
        for symbol in claim.source_symbols:
            node = nodes[symbol]
            arguments = _forward_arguments(node)
            control_name = "control" if "control" in arguments else "pre" if "pre" in arguments else None
            if control_name is None or "chemical" not in arguments:
                continue
            certificate = _singleton_attention_certificate(node)
            if certificate is not None:
                findings.append({"source_symbol": symbol, **dict(certificate)})
            else:
                return {
                    "static_status": IMPLEMENTED,
                    "reason": "cited fusion callable jointly accepts control and chemical inputs",
                    "findings": findings,
                }
        if findings:
            return {
                "static_status": IMPLEMENTATION_CONTRADICTED,
                "reason": "all cited interaction paths satisfy a registered singleton-attention contradiction",
                "findings": findings,
            }
        return {
            "static_status": UNRESOLVED,
            "reason": "the cited source does not expose a recognized joint control/chemical forward interface",
            "findings": [],
        }
    input_names = {
        CONTROL_RELIANCE: ("control", "pre"),
        CHEMICAL_RELIANCE: ("chemical",),
        DOSE_RELIANCE: ("dose",),
    }[claim.claim_type]
    for symbol in claim.source_symbols:
        node = nodes[symbol]
        arguments = _forward_arguments(node)
        present = tuple(name for name in input_names if name in arguments)
        if present and _input_is_consumed(node, aliases=present):
            return {
                "static_status": IMPLEMENTED,
                "reason": f"cited source callable consumes {present[0]}",
                "findings": [],
            }
    return {
        "static_status": UNRESOLVED,
        "reason": f"no cited source callable consumes a recognized {claim.claim_type} input",
        "findings": [],
    }


def check_source_claims(
    *,
    candidate_source: str,
    candidate_metadata: Mapping[str, Any],
    candidate_hash: str,
    candidate_id: str,
    fold3_global_pcc: float,
    registered_method_claims: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Compile a deterministic source-to-test receipt without running code.

    A missing manifest is an accepted, conservative result: universal chemical
    and control behavioral maps remain applicable, while optional mechanism
    claims are not inferred from natural-language descriptions.
    """

    _text(candidate_id, field="candidate_id", limit=240)
    if not isinstance(fold3_global_pcc, (int, float)) or not math.isfinite(float(fold3_global_pcc)):
        raise DeterministicClaimError("fold3_global_pcc must be finite")
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64:
        raise DeterministicClaimError("candidate_hash must be a SHA-256 digest")
    source = candidate_source
    tree = _source_tree(source)
    nodes = _top_level_nodes(tree)
    anchors = source_symbol_anchors(source)
    declarations = (
        *_task_contract_claims(candidate_metadata),
        *_candidate_manifest(candidate_metadata),
        *_registered_method_claims(registered_method_claims),
    )
    declaration_types = [claim.claim_type for claim in declarations]
    if len(declaration_types) != len(set(declaration_types)):
        raise DeterministicClaimError("a claim type is registered by more than one source contract")
    results: list[dict[str, Any]] = []
    for claim in declarations:
        inspection = _inspect_claim(claim, nodes=nodes)
        results.append({
            **claim.to_dict(),
            "source_anchor_hashes": {symbol: anchors[symbol] for symbol in claim.source_symbols if symbol in anchors},
            "required_tests": list(REQUIRED_TESTS[claim.claim_type]),
            **inspection,
        })
    for claim_type in REGISTERED_CLAIMS:
        if claim_type in declaration_types:
            continue
        results.append({
            "claim_type": claim_type,
            "statement": "No source-level claim was registered for this optional input-use contract.",
            "source_symbols": [],
            "origin": "not_claimed",
            "source_anchor_hashes": {},
            "required_tests": list(REQUIRED_TESTS[claim_type]),
            "static_status": NOT_CLAIMED,
            "reason": "the optional claim is absent from the frozen claim registry",
            "findings": [],
        })
    manifest_present = candidate_metadata.get("claim_manifest") is not None
    receipt = {
        "schema_version": CHECKER_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "fold3_global_pcc": float(fold3_global_pcc),
        "manifest_present": manifest_present,
        "registered_claim_count": len(declarations),
        "checker_mode": "deterministic_ast_only_no_candidate_execution",
        "claims": results,
        "universal_behavioral_tests": [
            "matched_control_swap",
            "within_context_chemical_shuffle",
            "chemical_control_2x2_interaction",
        ],
        "abstention_policy": (
            "Unknown source structures remain unresolved; no optional claim is inferred when "
            "claim_manifest is absent."
        ),
    }
    return {**receipt, "receipt_hash": digest(receipt)}


def check_candidate_claims(
    *,
    candidate: CandidateState,
    candidate_id: str,
    fold3_global_pcc: float,
    registered_method_claims: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """CandidateState convenience wrapper for :func:`check_source_claims`."""

    return check_source_claims(
        candidate_source=candidate.candidate_source,
        candidate_metadata=candidate.candidate_metadata,
        candidate_hash=candidate.candidate_hash,
        candidate_id=candidate_id,
        fold3_global_pcc=fold3_global_pcc,
        registered_method_claims=registered_method_claims,
    )


__all__ = [
    "CHECKER_SCHEMA",
    "CLAIM_MANIFEST_SCHEMA",
    "CHEMICAL_CONTROL_INTERACTION",
    "CHEMICAL_RELIANCE",
    "CONTROL_RELIANCE",
    "DOSE_RELIANCE",
    "IMPLEMENTATION_CONTRADICTED",
    "IMPLEMENTED",
    "NOT_CLAIMED",
    "REQUIRED_TESTS",
    "STATIC_STATUSES",
    "UNRESOLVED",
    "DeclaredClaim",
    "DeterministicClaimError",
    "check_candidate_claims",
    "check_source_claims",
    "source_symbol_anchors",
]
