"""Protected CellAudit runtime contract for free-form discovery candidates.

A discovery candidate may implement an arbitrary PyTorch architecture, loss,
optimizer and scheduler inside this interface.  It never receives source paths,
split assignments, endpoint arrays, or evaluator objects.  The fixed executor
owns data loading, normalization, training/evaluation roles and metric
calculation.

When ``build_scheduler`` returns a scheduler, a candidate may additionally
export ``step_scheduler(scheduler, state)``.  The executor supplies only the
current epoch, maximum epoch, and current Fold-3 global PCC.  Without that
hook, the executor calls ``scheduler.step()`` with no arguments.

The AST checks and tensor-only executor are a research-integrity boundary, not
a security sandbox for hostile code.  Generated candidates are validated
before the fixed training runtime imports them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import types
from typing import Any, Mapping


CANDIDATE_SCHEMA_VERSION = "cellscientist_open_candidate_v1"
ALLOWED_IMPORT_ROOTS = frozenset({"torch", "math", "typing", "dataclasses", "collections"})
FORBIDDEN_CALL_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
    }
)
FORBIDDEN_ATTRIBUTE_NAMES = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "fork",
        "socket",
        "urlopen",
        "request",
        "requests",
        "save",
        "load",
        "hub",
        "distributed",
        "multiprocessing",
    }
)
FORBIDDEN_DUNDER_ATTRIBUTES = frozenset(
    {"__class__", "__code__", "__dict__", "__globals__", "__mro__", "__subclasses__"}
)
REQUIRED_EXPORTS = (
    "candidate_metadata",
    "build_model",
    "compute_loss",
    "build_optimizer",
)
ALLOWED_SEMANTIC_ROOTS = frozenset(
    {
        "input",
        "perturbation",
        "response",
        "objective",
        "optimization",
        "nuisance",
        "reliability",
    }
)


class DiscoveryCandidateError(ValueError):
    """Raised when generated candidate source violates the protected API."""


@dataclass(frozen=True)
class CandidateTaskSpec:
    """Only tensor dimensions and response layout exposed to candidate code."""

    context_dim: int
    chemical_dim: int
    dose_dim: int
    cp_dim: int
    l1000_dim: int

    def __post_init__(self) -> None:
        for name in ("context_dim", "chemical_dim", "dose_dim", "cp_dim", "l1000_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DiscoveryCandidateError(f"{name} must be a positive integer")

    @property
    def target_dim(self) -> int:
        return self.cp_dim + self.l1000_dim

    def to_dict(self) -> dict[str, int]:
        return {
            "context_dim": self.context_dim,
            "chemical_dim": self.chemical_dim,
            "dose_dim": self.dose_dim,
            "cp_dim": self.cp_dim,
            "l1000_dim": self.l1000_dim,
            "target_dim": self.target_dim,
        }


@dataclass(frozen=True)
class CandidateValidation:
    """Replay-safe result of source, interface and dummy-forward validation."""

    source_sha256: str
    metadata: Mapping[str, Any]
    parameter_count: int
    output_shape: tuple[int, int]
    loss_value: float
    optimizer_class: str
    scheduler_class: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "metadata": dict(self.metadata),
            "parameter_count": self.parameter_count,
            "output_shape": list(self.output_shape),
            "loss_value": self.loss_value,
            "optimizer_class": self.optimizer_class,
            "scheduler_class": self.scheduler_class,
        }


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _root_module(name: str | None) -> str:
    return str(name or "").split(".", 1)[0]


def validate_candidate_source(source: str) -> ast.Module:
    """Parse generated source and reject data, filesystem and network access."""

    if not isinstance(source, str) or not source.strip():
        raise DiscoveryCandidateError("candidate source must be a non-empty string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DiscoveryCandidateError(f"candidate source has invalid Python syntax: {exc.msg}") from exc

    allowed_top_level = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
    )
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            raise DiscoveryCandidateError(
                f"candidate source has executable top-level node {type(node).__name__}"
            )
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            raise DiscoveryCandidateError("candidate source may only use a module docstring at top level")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise DiscoveryCandidateError(f"candidate import {root!r} is not permitted")
        elif isinstance(node, ast.ImportFrom):
            root = _root_module(node.module)
            if root not in ALLOWED_IMPORT_ROOTS:
                raise DiscoveryCandidateError(f"candidate import {root!r} is not permitted")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                raise DiscoveryCandidateError(f"candidate call {node.func.id!r} is not permitted")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTE_NAMES:
                raise DiscoveryCandidateError(
                    f"candidate attribute call {node.func.attr!r} is not permitted"
                )
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DUNDER_ATTRIBUTES:
            raise DiscoveryCandidateError(
                f"candidate dunder attribute {node.attr!r} is not permitted"
            )
        elif (
            isinstance(node, ast.Name)
            and node.id == "__doc__"
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr == "__doc__"
        ):
            raise DiscoveryCandidateError(
                "candidate source may not read __doc__; the module docstring is "
                "narrative provenance only"
            )
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise DiscoveryCandidateError("candidate global/nonlocal mutation is not permitted")
    return tree


def _load_module(path: Path, source: str) -> types.ModuleType:
    validate_candidate_source(source)
    module_name = f"_cellscientist_candidate_{source_sha256(source)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise DiscoveryCandidateError(f"cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in REQUIRED_EXPORTS:
        if not callable(getattr(module, name, None)):
            raise DiscoveryCandidateError(f"candidate module must export callable {name}()")
    return module


def _metadata(module: types.ModuleType) -> Mapping[str, Any]:
    payload = module.candidate_metadata()
    if not isinstance(payload, Mapping):
        raise DiscoveryCandidateError("candidate_metadata() must return a mapping")
    required = {
        "schema_version",
        "candidate_name",
        "semantic_components",
        "component_symbols",
        "change_summary",
    }
    missing = required.difference(payload)
    unknown_schema = payload.get("schema_version") != CANDIDATE_SCHEMA_VERSION
    if missing:
        raise DiscoveryCandidateError(f"candidate metadata missing fields: {sorted(missing)}")
    if unknown_schema:
        raise DiscoveryCandidateError("candidate metadata schema_version is unsupported")
    if not isinstance(payload["candidate_name"], str) or not payload["candidate_name"].strip():
        raise DiscoveryCandidateError("candidate_name must be a non-empty string")
    components = payload["semantic_components"]
    if not isinstance(components, (list, tuple)) or not components or not all(
        isinstance(item, str) and item for item in components
    ):
        raise DiscoveryCandidateError("semantic_components must be a non-empty string array")
    if len(set(components)) != len(components):
        raise DiscoveryCandidateError("semantic_components must be unique")
    for address in components:
        root, separator, remainder = address.partition(".")
        if not separator or not remainder or root not in ALLOWED_SEMANTIC_ROOTS:
            raise DiscoveryCandidateError(
                f"semantic component {address!r} must use a registered perturbation-model root"
            )
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in address
        ):
            raise DiscoveryCandidateError(
                f"semantic component {address!r} has invalid characters"
            )
    component_symbols = payload["component_symbols"]
    if not isinstance(component_symbols, Mapping):
        raise DiscoveryCandidateError("component_symbols must be a mapping")
    if set(component_symbols) != set(components):
        raise DiscoveryCandidateError(
            "component_symbols keys must exactly match semantic_components"
        )
    for address, symbols in component_symbols.items():
        if not isinstance(symbols, (list, tuple)) or not symbols:
            raise DiscoveryCandidateError(
                f"component_symbols[{address!r}] must be a non-empty string array"
            )
        for symbol in symbols:
            if (
                not isinstance(symbol, str)
                or not symbol.isidentifier()
                or not callable(getattr(module, symbol, None))
            ):
                raise DiscoveryCandidateError(
                    f"component symbol {symbol!r} for {address!r} is not an exported callable"
                )
    if not isinstance(payload["change_summary"], str):
        raise DiscoveryCandidateError("change_summary must be a string")
    try:
        json.dumps(payload, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise DiscoveryCandidateError("candidate metadata must be JSON serializable") from exc
    return dict(payload)


def validate_candidate_file(
    path: str | Path,
    *,
    task: CandidateTaskSpec,
    max_parameters: int,
    batch_size: int = 4,
) -> CandidateValidation:
    """Validate one candidate without exposing any real task arrays."""

    candidate_path = Path(path)
    source = candidate_path.read_text(encoding="utf-8")
    module = _load_module(candidate_path, source)
    metadata = _metadata(module)
    if max_parameters <= 0 or batch_size <= 1:
        raise DiscoveryCandidateError("max_parameters and batch_size must be positive")

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("candidate validation requires PyTorch") from exc

    torch.manual_seed(0)
    model = module.build_model(task.to_dict())
    if not isinstance(model, torch.nn.Module):
        raise DiscoveryCandidateError("build_model() must return torch.nn.Module")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count <= 0:
        raise DiscoveryCandidateError("candidate model has no parameters")
    if parameter_count > max_parameters:
        raise DiscoveryCandidateError(
            f"candidate parameter count exceeds budget: {parameter_count} > {max_parameters}"
        )

    pre = torch.zeros(batch_size, task.context_dim)
    chemical = torch.zeros(batch_size, task.chemical_dim)
    dose = torch.zeros(batch_size, task.dose_dim)
    target = torch.zeros(batch_size, task.target_dim)
    prediction = model(pre, chemical, dose)
    if not isinstance(prediction, torch.Tensor):
        raise DiscoveryCandidateError("candidate forward() must return a tensor")
    if tuple(prediction.shape) != (batch_size, task.target_dim):
        raise DiscoveryCandidateError(
            f"candidate output shape {tuple(prediction.shape)} differs from "
            f"{(batch_size, task.target_dim)}"
        )
    if not torch.isfinite(prediction).all():
        raise DiscoveryCandidateError("candidate dummy prediction is non-finite")

    loss = module.compute_loss(prediction, target, {"epoch": 0, "max_epochs": 1})
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise DiscoveryCandidateError("compute_loss() must return one finite scalar tensor")
    optimizer = module.build_optimizer(model, {"max_epochs": 1})
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise DiscoveryCandidateError("build_optimizer() must return torch.optim.Optimizer")
    scheduler = None
    if callable(getattr(module, "build_scheduler", None)):
        scheduler = module.build_scheduler(optimizer, {"max_epochs": 1})
        if scheduler is not None and not hasattr(scheduler, "step"):
            raise DiscoveryCandidateError("build_scheduler() must return None or a scheduler with step()")
    return CandidateValidation(
        source_sha256=source_sha256(source),
        metadata=metadata,
        parameter_count=parameter_count,
        output_shape=tuple(int(value) for value in prediction.shape),
        loss_value=float(loss.detach().cpu()),
        optimizer_class=type(optimizer).__name__,
        scheduler_class=None if scheduler is None else type(scheduler).__name__,
    )
