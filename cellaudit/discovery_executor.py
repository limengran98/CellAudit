"""Protected, fit-only training executor for open CPG discovery candidates.

Candidates are intentionally free to implement a PyTorch model, loss,
optimizer, and optional scheduler through :mod:`discovery_candidate`.  This
executor owns every datum and every experimental control that could otherwise
make a discovery comparison ambiguous: split roles, fit-only normalization,
seeding, batching, epoch count, device placement, early stopping, and metric
calculation.  A candidate receives tensors plus a dimensions-only task spec;
it never receives a path, cache index, fold identifier, evaluator, or any
fold-4/fold-5 array.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from .discovery_candidate import (
    CandidateTaskSpec,
    CandidateValidation,
    DiscoveryCandidateError,
    _load_module,
    validate_candidate_file,
)
from .discovery_task import BBBC036DiscoveryArrays, BBBC036DiscoveryPartition
from .open_discovery import CandidateState
from .perturbation_loop_data import (
    build_chemical_prefix_sham_map,
    build_control_profile_sham_map,
)
from .schemas import SchemaError, canonical_json
from .training_config_contract import RUNTIME_TRAINING_CONTROL_KEYS


class DiscoveryExecutorError(RuntimeError):
    """Raised when the protected discovery execution contract is violated."""


@dataclass(frozen=True)
class DiscoveryExecutorSettings:
    """Executor-owned immutable identity plus registered candidate hard caps.

    ``max_epochs``, ``batch_size``, ``early_stopping_patience``, and
    ``gradient_clip_norm`` are *caps*, not an instruction that every open
    discovery candidate must run identically.  A candidate may request smaller
    legal values through ``CandidateState.training_config``; seed, device,
    split roles, standardization, evaluator, and the caps themselves remain
    executor-owned and fixed for the run.
    """

    seed: int = 20260729
    max_epochs: int = 30
    batch_size: int = 128
    device: str = "cpu"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-5
    max_parameters: int = 2_500_000
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DiscoveryExecutorError("executor seed must be an integer")
        if self.max_epochs <= 0 or self.batch_size <= 0 or self.max_parameters <= 0:
            raise DiscoveryExecutorError("executor epochs, batch size, and parameter cap must be positive")
        if self.early_stopping_patience < 0 or self.early_stopping_min_delta < 0:
            raise DiscoveryExecutorError("executor early-stopping controls must be non-negative")
        if self.gradient_clip_norm <= 0:
            raise DiscoveryExecutorError("executor gradient clip norm must be positive")
        if str(self.device) not in {"cpu", "cuda"} and not str(self.device).startswith("cuda:"):
            raise DiscoveryExecutorError("executor device must be 'cpu', 'cuda', or 'cuda:<index>'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "max_epochs": int(self.max_epochs),
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "early_stopping_patience": int(self.early_stopping_patience),
            "early_stopping_min_delta": float(self.early_stopping_min_delta),
            "max_parameters": int(self.max_parameters),
            "gradient_clip_norm": float(self.gradient_clip_norm),
        }


@dataclass(frozen=True)
class CandidateTrainingControls:
    """One candidate's legal training request after executor cap validation."""

    batch_size: int
    max_epochs: int
    patience: int
    gradient_clip: float
    min_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "gradient_clip": float(self.gradient_clip),
            "min_delta": float(self.min_delta),
        }


@dataclass(frozen=True)
class EpochDiscoveryMetrics:
    """Train and fold-3 feedback metrics after one executor epoch."""

    epoch: int
    train: Mapping[str, float]
    discovery: Mapping[str, float]
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "train": {str(key): float(value) for key, value in self.train.items()},
            "discovery": {str(key): float(value) for key, value in self.discovery.items()},
            "runtime_seconds": float(self.runtime_seconds),
        }


@dataclass(frozen=True)
class DiscoveryExecutionResult:
    """Replay-safe record of one candidate fit/feedback trajectory.

    No trained model or raw array is returned.  The result contains only the
    validation record, fixed executor controls, normalized-statistics hash,
    and per-epoch scalar metrics from fit/fold-3 data.
    """

    task_id: str
    contract_hash: str
    candidate_hash: str | None
    candidate_validation: CandidateValidation
    settings: Mapping[str, Any]
    candidate_training_controls: Mapping[str, Any]
    standardization: Mapping[str, Any]
    epoch_metrics: tuple[EpochDiscoveryMetrics, ...]
    selected_epoch: int
    selected_train: Mapping[str, float]
    selected_discovery: Mapping[str, float]
    input_reliance_audit: Mapping[str, Any]
    stopped_early: bool
    completed_epochs: int
    runtime_seconds: float
    access_audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "contract_hash": self.contract_hash,
            "candidate_hash": self.candidate_hash,
            "candidate_validation": self.candidate_validation.to_dict(),
            "settings": dict(self.settings),
            "candidate_training_controls": dict(self.candidate_training_controls),
            "standardization": dict(self.standardization),
            "epoch_metrics": [item.to_dict() for item in self.epoch_metrics],
            "selected_epoch": int(self.selected_epoch),
            "selected_train": {str(key): float(value) for key, value in self.selected_train.items()},
            "selected_discovery": {
                str(key): float(value) for key, value in self.selected_discovery.items()
            },
            "input_reliance_audit": dict(self.input_reliance_audit),
            "stopped_early": bool(self.stopped_early),
            "completed_epochs": int(self.completed_epochs),
            "runtime_seconds": float(self.runtime_seconds),
            "access_audit": dict(self.access_audit),
        }


@dataclass(frozen=True)
class _TensorStandardizer:
    pre_mean: Any
    pre_scale: Any
    condition_mean: Any
    condition_scale: Any
    target_mean: Any
    target_scale: Any
    fingerprint: str

    def normalize_pre(self, value: Any) -> Any:
        return (value - self.pre_mean) / self.pre_scale

    def normalize_condition(self, value: Any) -> Any:
        return (value - self.condition_mean) / self.condition_scale

    def normalize_target(self, value: Any) -> Any:
        return (value - self.target_mean) / self.target_scale

    def inverse_target(self, value: Any) -> Any:
        scale = self.target_scale.to(device=value.device, dtype=value.dtype)
        mean = self.target_mean.to(device=value.device, dtype=value.dtype)
        return value * scale + mean


@dataclass(frozen=True)
class _ProtectedTensors:
    fit_pre: Any
    fit_condition: Any
    fit_target_raw: Any
    feedback_pre: Any
    feedback_condition: Any
    feedback_target_raw: Any
    standardizer: _TensorStandardizer
    task_spec: CandidateTaskSpec


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("Discovery executor requires PyTorch") from exc
    return torch, DataLoader, TensorDataset


@dataclass(frozen=True)
class _CandidateMaterial:
    """Source/configuration supplied by the open-discovery trace, not data."""

    source: str
    candidate_metadata: Mapping[str, Any] | None
    training_config: Mapping[str, Any]
    candidate_hash: str | None


def _candidate_material(candidate: CandidateState | str | Path) -> _CandidateMaterial:
    """Accept a hash-linked CandidateState or a local h0 source for smoke use."""

    if isinstance(candidate, CandidateState):
        return _CandidateMaterial(
            source=candidate.candidate_source,
            candidate_metadata=dict(candidate.candidate_metadata),
            training_config=dict(candidate.training_config),
            candidate_hash=candidate.candidate_hash,
        )
    if isinstance(candidate, (str, Path)):
        path = Path(candidate)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DiscoveryExecutorError(f"Could not read discovery candidate source: {path}") from exc
        return _CandidateMaterial(
            source=source,
            candidate_metadata=None,
            training_config={},
            candidate_hash=None,
        )
    raise TypeError("Discovery executor candidate must be CandidateState or a local candidate source path")


def _json_plain(value: Any) -> Any:
    """Thaw CandidateState's immutable MappingProxy-backed JSON fields."""

    if isinstance(value, Mapping):
        return {str(key): _json_plain(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_json_plain(member) for member in value]
    if isinstance(value, list):
        return [_json_plain(member) for member in value]
    return value


_SELECTED_DISCOVERY_ARTIFACT_SCHEMA_VERSION = "cellscientist_selected_discovery_artifacts_v1"


def _artifact_file_sha256(path: Path) -> str:
    """Return a content digest without retaining an additional copy in memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _export_selected_discovery_artifacts(
    *,
    export_root: str | Path,
    torch: Any,
    model: Any,
    tensors: _ProtectedTensors,
    arrays: BBBC036DiscoveryArrays,
    prefix_dim: int,
    result: DiscoveryExecutionResult,
    candidate_source_hash: str,
    candidate_source: str,
    candidate_metadata: Mapping[str, Any],
    candidate_training_config: Mapping[str, Any],
) -> None:
    """Persist exactly one post-search selected replay, atomically.

    Discovery evaluation deliberately keeps every candidate model and output in
    memory only.  This routine is reachable solely through the explicit
    ``export_selected_artifacts`` executor argument, after the selected state
    has been restored.  It writes a checkpoint and the corresponding Fold-3
    prediction/target arrays exactly once, so downstream DEG calculations can
    be performed without re-running discovery or contacting an LLM.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise DiscoveryExecutorError("Selected-artifact export requires NumPy") from exc

    root = Path(export_root).expanduser()
    if root.exists():
        raise DiscoveryExecutorError(
            f"Selected-artifact destination already exists and will not be overwritten: {root}"
        )
    if root.name in {"", ".", ".."}:
        raise DiscoveryExecutorError("Selected-artifact destination must name a new directory")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
    try:
        model.eval()
        with torch.no_grad():
            normalized_pre = tensors.standardizer.normalize_pre(tensors.feedback_pre)
            normalized_condition = tensors.standardizer.normalize_condition(tensors.feedback_condition)
            chemical, dose = _split_inputs(normalized_condition, prefix_dim=prefix_dim)
            device = next(model.parameters()).device
            normalized_prediction = model(
                normalized_pre.to(device),
                chemical.to(device),
                dose.to(device),
            )
            prediction = tensors.standardizer.inverse_target(normalized_prediction)
            target = tensors.feedback_target_raw.to(device)

        prediction_array = prediction.detach().cpu().to(dtype=torch.float32).numpy()
        target_array = target.detach().cpu().to(dtype=torch.float32).numpy()
        if tuple(prediction_array.shape) != tuple(target_array.shape):
            raise DiscoveryExecutorError("Selected Fold-3 prediction shape differs from target shape")
        if prediction_array.shape[0] != len(arrays.discovery.labels):
            raise DiscoveryExecutorError("Selected Fold-3 prediction row count differs from protected labels")
        if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
            raise DiscoveryExecutorError("Selected Fold-3 export contains non-finite values")

        checkpoint_path = temporary / "selected_model.pt"
        prediction_path = temporary / "selected_fold3_predictions.npy"
        target_path = temporary / "selected_fold3_targets.npy"
        torch.save(
            {
                "schema_version": _SELECTED_DISCOVERY_ARTIFACT_SCHEMA_VERSION,
                "artifact_policy": "post_search_selected_candidate_only",
                "task_id": result.task_id,
                "contract_hash": result.contract_hash,
                "candidate_hash": result.candidate_hash,
                "candidate_source_hash": candidate_source_hash,
                # This is the one selected source/configuration necessary to
                # rebuild the module around its state dict; no process source
                # is exported.
                "candidate_source": str(candidate_source),
                "candidate_metadata": _json_plain(candidate_metadata),
                "candidate_training_config": _json_plain(candidate_training_config),
                "selected_epoch": int(result.selected_epoch),
                "candidate_training_controls": dict(result.candidate_training_controls),
                "standardization_fingerprint": result.standardization["statistics_sha256"],
                "model_state_dict": {
                    str(name): value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            },
            checkpoint_path,
        )
        np.save(prediction_path, prediction_array, allow_pickle=False)
        np.save(target_path, target_array, allow_pickle=False)
        receipt = {
            "schema_version": _SELECTED_DISCOVERY_ARTIFACT_SCHEMA_VERSION,
            "artifact_policy": "post_search_selected_candidate_only",
            "intermediate_candidate_models_or_arrays_saved": False,
            "selection_partition": "feedback_fold_3",
            "selection_folds": [3],
            "withheld_target_folds": [4, 5],
            "task_id": result.task_id,
            "contract_hash": result.contract_hash,
            "candidate_hash": result.candidate_hash,
            "candidate_source_hash": candidate_source_hash,
            "selected_epoch": int(result.selected_epoch),
            "selected_discovery": dict(result.selected_discovery),
            "prediction_target_space": "raw_control_relative_joint_response",
            "prediction_row_labels": [str(label) for label in arrays.discovery.labels],
            "cp_target_dim": int(tensors.task_spec.cp_dim),
            "l1000_target_dim": int(tensors.task_spec.l1000_dim),
            "files": [
                {
                    "name": checkpoint_path.name,
                    "sha256": _artifact_file_sha256(checkpoint_path),
                    "kind": "selected_checkpoint",
                },
                {
                    "name": prediction_path.name,
                    "sha256": _artifact_file_sha256(prediction_path),
                    "kind": "fold3_raw_prediction",
                    "shape": [int(value) for value in prediction_array.shape],
                    "dtype": str(prediction_array.dtype),
                },
                {
                    "name": target_path.name,
                    "sha256": _artifact_file_sha256(target_path),
                    "kind": "fold3_raw_target",
                    "shape": [int(value) for value in target_array.shape],
                    "dtype": str(target_array.dtype),
                },
            ],
        }
        (temporary / "selected_artifact_receipt.json").write_text(
            canonical_json(receipt) + "\n", encoding="utf-8"
        )
        temporary.replace(root)
    except Exception:
        # Keep an unpublished temporary directory for local diagnosis rather
        # than ever overwriting an existing finalized selection.
        raise


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise DiscoveryExecutorError(f"candidate training_config.{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _bounded_float(value: Any, *, name: str, minimum_exclusive: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiscoveryExecutorError(
            f"candidate training_config.{name} must be numeric in ({minimum_exclusive}, {maximum}]"
        )
    result = float(value)
    if not result > minimum_exclusive or not result <= maximum:
        raise DiscoveryExecutorError(
            f"candidate training_config.{name} must be numeric in ({minimum_exclusive}, {maximum}]"
        )
    return result


def candidate_training_controls(
    training_config: Mapping[str, Any],
    *,
    settings: DiscoveryExecutorSettings,
) -> CandidateTrainingControls:
    """Read legal candidate-level runtime controls without exposing evaluator data.

    The controls live at the top level of ``CandidateState.training_config``:
    ``batch_size``, ``max_epochs``, ``patience``, ``gradient_clip``, and
    ``min_delta``.  ``gradient_clip_norm`` is accepted as an unambiguous
    spelling alias.  All are optional.  Missing values resolve to the
    executor's registered hard caps.  Other candidate configuration fields
    (architecture, loss, optimizer, scheduler, etc.) stay opaque to this
    executor and are not executable controls.
    """

    if not isinstance(training_config, Mapping):
        raise DiscoveryExecutorError("Candidate training_config must be a mapping")
    # Keep this assertion next to the real reader: the candidate compiler imports
    # the same constant when classifying configuration-only edits, so a field
    # cannot silently become executable before it has an execution semantics.
    if not RUNTIME_TRAINING_CONTROL_KEYS:
        raise DiscoveryExecutorError("runtime training control contract is empty")
    if "gradient_clip" in training_config and "gradient_clip_norm" in training_config:
        raise DiscoveryExecutorError("Candidate training_config must not specify both gradient_clip and gradient_clip_norm")
    batch_size = _bounded_int(
        training_config.get("batch_size", settings.batch_size),
        name="batch_size",
        minimum=1,
        maximum=int(settings.batch_size),
    )
    max_epochs = _bounded_int(
        training_config.get("max_epochs", settings.max_epochs),
        name="max_epochs",
        minimum=1,
        maximum=int(settings.max_epochs),
    )
    patience = _bounded_int(
        training_config.get("patience", settings.early_stopping_patience),
        name="patience",
        minimum=0,
        maximum=int(settings.early_stopping_patience),
    )
    gradient_value = training_config.get(
        "gradient_clip",
        training_config.get("gradient_clip_norm", settings.gradient_clip_norm),
    )
    gradient_clip = _bounded_float(
        gradient_value,
        name="gradient_clip",
        minimum_exclusive=0.0,
        maximum=float(settings.gradient_clip_norm),
    )
    min_delta_value = training_config.get(
        "min_delta", settings.early_stopping_min_delta
    )
    if isinstance(min_delta_value, bool) or not isinstance(
        min_delta_value, (int, float)
    ):
        raise DiscoveryExecutorError(
            "candidate training_config.min_delta must be numeric in [0.0, 1.0]"
        )
    min_delta = float(min_delta_value)
    if not 0.0 <= min_delta <= 1.0:
        raise DiscoveryExecutorError(
            "candidate training_config.min_delta must be numeric in [0.0, 1.0]"
        )
    return CandidateTrainingControls(
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        gradient_clip=gradient_clip,
        min_delta=min_delta,
    )


def _validate_arrays(arrays: BBBC036DiscoveryArrays) -> None:
    if not isinstance(arrays, BBBC036DiscoveryArrays):
        raise TypeError("Discovery executor accepts only BBBC036DiscoveryArrays")
    if arrays.task_id != arrays.contract.task_id:
        raise DiscoveryExecutorError("Discovery arrays and task contract disagree")
    protocol = dict(arrays.contract.training_budget).get("partition_protocol")
    if not isinstance(protocol, Mapping):
        raise DiscoveryExecutorError("Discovery arrays lack a registered partition protocol")
    expected_roles = {"fit": [1, 2], "discovery": 3, "audit": 4, "endpoint": 5}
    if dict(protocol.get("cache_fold_roles", {})) != expected_roles:
        raise DiscoveryExecutorError("Discovery contract does not register the required five-fold role boundary")
    for name, partition, allowed_folds in (
        ("fit", arrays.fit, {0, 1}),
        ("discovery", arrays.discovery, {2}),
    ):
        if not isinstance(partition, BBBC036DiscoveryPartition):
            raise DiscoveryExecutorError(f"Discovery {name} partition has an unexpected type")
        observed = {int(value) for value in partition.cache_folds.tolist()}
        if not observed or not observed.issubset(allowed_folds):
            raise DiscoveryExecutorError(f"Discovery {name} partition contains a withheld or unregistered fold")
        if name == "fit" and observed != {0, 1}:
            raise DiscoveryExecutorError("Fit partition must contain both registered fit folds 1 and 2")
        if name == "discovery" and observed != {2}:
            raise DiscoveryExecutorError("Discovery feedback must contain fold 3 only")
        count = partition.count
        tensors = (partition.pre, partition.condition, partition.target, partition.cp_target, partition.l1000_target)
        if count <= 1 or any(int(value.shape[0]) != count for value in tensors):
            raise DiscoveryExecutorError(f"Discovery {name} tensor rows are inconsistent or insufficient")
        if len(partition.labels) != count or len(partition.cache_indices) != count:
            raise DiscoveryExecutorError(f"Discovery {name} labels/indices are inconsistent")
        if int(partition.target.shape[1]) != int(partition.cp_target.shape[1]) + int(partition.l1000_target.shape[1]):
            raise DiscoveryExecutorError(f"Discovery {name} target blocks do not reconstruct joint target")
    if int(arrays.fit.pre.shape[1]) != int(arrays.discovery.pre.shape[1]):
        raise DiscoveryExecutorError("Fit and discovery context dimensions differ")
    if int(arrays.fit.condition.shape[1]) != int(arrays.discovery.condition.shape[1]):
        raise DiscoveryExecutorError("Fit and discovery condition dimensions differ")
    if int(arrays.fit.target.shape[1]) != int(arrays.discovery.target.shape[1]):
        raise DiscoveryExecutorError("Fit and discovery target dimensions differ")
    layout = arrays.contract.condition_layout
    if not isinstance(layout, Mapping) or int(layout.get("prefix_dim", 0)) + 1 != int(arrays.fit.condition.shape[1]):
        raise DiscoveryExecutorError("Discovery task contract lacks the registered prefix-plus-dose condition layout")


def _tensor_stats(torch: Any, fit: Any) -> tuple[Any, Any]:
    mean = fit.mean(dim=0, keepdim=True)
    scale = fit.std(dim=0, unbiased=False, keepdim=True)
    # A constant fit feature carries no train-derived scale.  Mapping it to a
    # unit scale is deterministic and avoids using any feedback statistic.
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    return mean, scale


def _standardizer_fingerprint(*tensors: Any) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        detached = tensor.detach().to("cpu").contiguous().numpy()
        digest.update(str(tuple(int(value) for value in detached.shape)).encode("ascii"))
        digest.update(detached.tobytes())
    return digest.hexdigest()


def _prepare_feedback_tensors(
    *,
    fit: BBBC036DiscoveryPartition,
    feedback: BBBC036DiscoveryPartition,
    contract: Any,
) -> _ProtectedTensors:
    """Create protected tensors using fit statistics and one feedback partition.

    Discovery and the later held-out audit share the exact same
    normalization and tensor boundary.  Their only difference is which
    already-authorized feedback partition is supplied by the caller.  Keeping
    this helper role-agnostic avoids a second training implementation while
    leaving :func:`execute_discovery_candidate` and its Fold-3-only public API
    unchanged.
    """

    torch, _, _ = _require_torch()
    fit_pre = torch.as_tensor(fit.pre, dtype=torch.float32, device="cpu")
    fit_condition = torch.as_tensor(fit.condition, dtype=torch.float32, device="cpu")
    fit_target = torch.as_tensor(fit.target, dtype=torch.float32, device="cpu")
    feedback_pre = torch.as_tensor(feedback.pre, dtype=torch.float32, device="cpu")
    feedback_condition = torch.as_tensor(feedback.condition, dtype=torch.float32, device="cpu")
    feedback_target = torch.as_tensor(feedback.target, dtype=torch.float32, device="cpu")
    pre_mean, pre_scale = _tensor_stats(torch, fit_pre)
    condition_mean, condition_scale = _tensor_stats(torch, fit_condition)
    target_mean, target_scale = _tensor_stats(torch, fit_target)
    standardizer = _TensorStandardizer(
        pre_mean=pre_mean,
        pre_scale=pre_scale,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fingerprint=_standardizer_fingerprint(
            pre_mean,
            pre_scale,
            condition_mean,
            condition_scale,
            target_mean,
            target_scale,
        ),
    )
    prefix_dim = int(contract.condition_layout["prefix_dim"])
    task_spec = CandidateTaskSpec(
        context_dim=int(fit_pre.shape[1]),
        chemical_dim=prefix_dim,
        dose_dim=1,
        cp_dim=int(fit.cp_target.shape[1]),
        l1000_dim=int(fit.l1000_target.shape[1]),
    )
    return _ProtectedTensors(
        fit_pre=fit_pre,
        fit_condition=fit_condition,
        fit_target_raw=fit_target,
        feedback_pre=feedback_pre,
        feedback_condition=feedback_condition,
        feedback_target_raw=feedback_target,
        standardizer=standardizer,
        task_spec=task_spec,
    )


def _prepare_tensors(arrays: BBBC036DiscoveryArrays) -> _ProtectedTensors:
    """Create standardised Fold-1/2 and Fold-3 tensors on CPU."""

    return _prepare_feedback_tensors(
        fit=arrays.fit,
        feedback=arrays.discovery,
        contract=arrays.contract,
    )


def _resolve_device(torch: Any, setting: str):
    device = torch.device(setting)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DiscoveryExecutorError("CUDA executor device was requested but is unavailable")
    return device


def _seed_executor(torch: Any, seed: int, *, device: Any) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _split_inputs(condition: Any, *, prefix_dim: int) -> tuple[Any, Any]:
    chemical = condition[:, :prefix_dim]
    dose = condition[:, prefix_dim : prefix_dim + 1]
    if int(dose.shape[1]) != 1:
        raise DiscoveryExecutorError("Registered discovery condition does not expose exactly one dose scalar")
    return chemical, dose


def _global_pcc(torch: Any, predicted: Any, observed: Any) -> float:
    prediction = predicted.reshape(-1)
    target = observed.reshape(-1)
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = torch.sqrt(
        (centered_prediction.square().sum() * centered_target.square().sum()).clamp_min(1e-12)
    )
    return float((centered_prediction * centered_target).sum().div(denominator).detach().cpu())


def _candidate_loss(module: Any, prediction: Any, target: Any, *, epoch: int, max_epochs: int) -> Any:
    # This is the only state object candidate loss/optimizer code sees.  It
    # deliberately contains no row IDs, fold ID, path, metric, or evaluator.
    state = {"epoch": int(epoch), "max_epochs": int(max_epochs)}
    loss = module.compute_loss(prediction, target, state)
    if not hasattr(loss, "ndim") or int(loss.ndim) != 0:
        raise DiscoveryExecutorError("Candidate compute_loss() must return one scalar tensor at runtime")
    if not bool(loss.isfinite()):
        raise DiscoveryExecutorError("Candidate compute_loss() returned a non-finite value at runtime")
    return loss


def _metrics(
    torch: Any,
    module: Any,
    model: Any,
    *,
    raw_pre: Any,
    raw_condition: Any,
    raw_target: Any,
    standardizer: _TensorStandardizer,
    prefix_dim: int,
    cp_dim: int,
    device: Any,
    epoch: int,
    max_epochs: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        normalized_pre = standardizer.normalize_pre(raw_pre).to(device)
        normalized_condition = standardizer.normalize_condition(raw_condition).to(device)
        normalized_target = standardizer.normalize_target(raw_target).to(device)
        chemical, dose = _split_inputs(normalized_condition, prefix_dim=prefix_dim)
        normalized_prediction = model(normalized_pre, chemical, dose)
        if tuple(normalized_prediction.shape) != tuple(normalized_target.shape):
            raise DiscoveryExecutorError("Candidate output shape changed after validation")
        loss = _candidate_loss(module, normalized_prediction, normalized_target, epoch=epoch, max_epochs=max_epochs)
        prediction = standardizer.inverse_target(normalized_prediction)
        observed = raw_target.to(device)
        return {
            "optimization_loss": float(loss.detach().cpu()),
            "global_pcc": _global_pcc(torch, prediction, observed),
            "cp_pcc": _global_pcc(torch, prediction[:, :cp_dim], observed[:, :cp_dim]),
            "l1000_pcc": _global_pcc(torch, prediction[:, cp_dim:], observed[:, cp_dim:]),
            "mse": float((prediction - observed).square().mean().detach().cpu()),
        }


def _input_global_pcc(
    torch: Any,
    model: Any,
    *,
    raw_pre: Any,
    raw_condition: Any,
    raw_target: Any,
    standardizer: _TensorStandardizer,
    prefix_dim: int,
    device: Any,
) -> float:
    """Score an input counterfactual without exposing its target to candidate code.

    The candidate model receives only its normal three input tensors.  In
    particular, this helper deliberately does not call ``compute_loss``: the
    response target remains inside the executor while it calculates the
    target-blind input-reliance PCC diagnostic.
    """

    model.eval()
    with torch.no_grad():
        normalized_pre = standardizer.normalize_pre(raw_pre).to(device)
        normalized_condition = standardizer.normalize_condition(raw_condition).to(device)
        chemical, dose = _split_inputs(normalized_condition, prefix_dim=prefix_dim)
        normalized_prediction = model(normalized_pre, chemical, dose)
        expected = (int(raw_target.shape[0]), int(raw_target.shape[1]))
        if tuple(normalized_prediction.shape) != expected:
            raise DiscoveryExecutorError("Candidate output shape changed during input-reliance diagnostics")
        prediction = standardizer.inverse_target(normalized_prediction)
        return _global_pcc(torch, prediction, raw_target.to(device))


def _input_reliance_diagnostics(
    torch: Any,
    model: Any,
    *,
    arrays: BBBC036DiscoveryArrays,
    tensors: _ProtectedTensors,
    device: Any,
    map_seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure Fold-3 control/chemical reliance under target-blind sham maps.

    The semantic maps consume only Fold-3 input arrays, stable row labels, and
    an executor-owned seed.  They are constructed after the selected model
    state has been restored, and their PCC drops neither alter training nor
    feed early stopping.  The executor retains maps, targets, and PCC
    calculation; candidate source receives none of those objects.
    """

    prefix_dim = int(tensors.task_spec.chemical_dim)
    observed_pcc = _input_global_pcc(
        torch,
        model,
        raw_pre=tensors.feedback_pre,
        raw_condition=tensors.feedback_condition,
        raw_target=tensors.feedback_target_raw,
        standardizer=tensors.standardizer,
        prefix_dim=prefix_dim,
        device=device,
    )
    metrics: dict[str, float] = {"observed_global_pcc": float(observed_pcc)}
    audit: dict[str, Any] = {
        "partition": "feedback_fold_3",
        "fold": 3,
        "map_seed": int(map_seed),
        "target_blind_construction": True,
        "calculation_stage": "after_restoring_fold3_selected_state",
        "selection_effect": "diagnostic_only_not_used_for_training_or_early_stopping",
        "observed_global_pcc": float(observed_pcc),
    }

    # The map builders intentionally receive raw inputs and labels only.  The
    # response target is absent from both calls, and map objects are retained
    # by this executor rather than supplied to candidate code.
    raw_pre = arrays.discovery.pre
    raw_condition = arrays.discovery.condition
    row_labels = arrays.discovery.labels
    try:
        control_map = build_control_profile_sham_map(raw_pre, row_ids=row_labels, seed=map_seed)
        swapped_pre = torch.as_tensor(control_map.apply(raw_pre), dtype=tensors.feedback_pre.dtype, device="cpu")
        control_swap_pcc = _input_global_pcc(
            torch,
            model,
            raw_pre=swapped_pre,
            raw_condition=tensors.feedback_condition,
            raw_target=tensors.feedback_target_raw,
            standardizer=tensors.standardizer,
            prefix_dim=prefix_dim,
            device=device,
        )
        control_drop = float(observed_pcc - control_swap_pcc)
        metrics["correct_control_swap_pcc_drop"] = control_drop
        audit["control_swap"] = {
            "status": "observed",
            "counterfactual_global_pcc": float(control_swap_pcc),
            "pcc_drop": control_drop,
            "map_hash": str(control_map.audit["map_hash"]),
            "map_audit": dict(control_map.audit),
        }
    except SchemaError as exc:
        audit["control_swap"] = {
            "status": "not_available",
            "reason": str(exc),
        }

    try:
        chemical_map = build_chemical_prefix_sham_map(
            raw_pre,
            raw_condition[:, :prefix_dim],
            row_ids=row_labels,
            seed=map_seed,
        )
        shuffled_prefix = torch.as_tensor(
            chemical_map.apply(raw_condition[:, :prefix_dim]),
            dtype=tensors.feedback_condition.dtype,
            device="cpu",
        )
        shuffled_condition = tensors.feedback_condition.clone()
        shuffled_condition[:, :prefix_dim] = shuffled_prefix
        chemical_shuffle_pcc = _input_global_pcc(
            torch,
            model,
            raw_pre=tensors.feedback_pre,
            raw_condition=shuffled_condition,
            raw_target=tensors.feedback_target_raw,
            standardizer=tensors.standardizer,
            prefix_dim=prefix_dim,
            device=device,
        )
        chemical_drop = float(observed_pcc - chemical_shuffle_pcc)
        metrics["correct_chemical_shuffle_pcc_drop"] = chemical_drop
        audit["chemical_shuffle"] = {
            "status": "observed",
            "counterfactual_global_pcc": float(chemical_shuffle_pcc),
            "pcc_drop": chemical_drop,
            "map_hash": str(chemical_map.audit["map_hash"]),
            "map_audit": dict(chemical_map.audit),
        }
    except SchemaError as exc:
        audit["chemical_shuffle"] = {
            "status": "not_available",
            "reason": str(exc),
        }
    return metrics, audit


def _scheduler_step(
    module: Any,
    scheduler: Any,
    *,
    epoch: int,
    max_epochs: int,
    fold3_global_pcc: float,
) -> None:
    """Step a candidate scheduler without mistaking a metric for an epoch.

    PyTorch schedulers disagree on ``step`` signatures: ``StepLR.step(x)``
    silently treats ``x`` as an epoch, whereas ``ReduceLROnPlateau.step(x)``
    needs a monitored metric.  A candidate that wants Fold-3-driven scheduling
    must therefore explicitly export ``step_scheduler(scheduler, state)``.
    Otherwise the fixed executor invokes the ordinary parameterless
    ``scheduler.step()``.  The optional callback gets only scalar Fold-3 PCC
    and execution counters, never an evaluator or raw feedback tensors.
    """

    if scheduler is None:
        return
    callback = getattr(module, "step_scheduler", None)
    if callable(callback):
        callback(
            scheduler,
            {
                "epoch": int(epoch),
                "max_epochs": int(max_epochs),
                "fold3_global_pcc": float(fold3_global_pcc),
            },
        )
        return
    scheduler.step()


def _strict_candidate_module(source: str, *, task_spec: CandidateTaskSpec, max_parameters: int):
    """Validate source in a temporary module path, then expose only tensor API.

    A ``CandidateState`` carries source text, rather than a filesystem path.
    The executor creates a private temporary module only for the protected
    validator/importer.  Candidate code is never given the source path, task
    arrays, provenance, or evaluator.
    """

    with tempfile.TemporaryDirectory(prefix="cellscientist-discovery-candidate-") as temporary:
        candidate_path = Path(temporary) / "candidate.py"
        candidate_path.write_text(source, encoding="utf-8")
        validation = validate_candidate_file(
            candidate_path,
            task=task_spec,
            max_parameters=max_parameters,
            batch_size=3,
        )
        try:
            module = _load_module(candidate_path, source)
        except DiscoveryCandidateError as exc:
            raise DiscoveryExecutorError(f"Candidate failed protected module load: {exc}") from exc
    return validation, module


def preflight_discovery_candidate_interface(
    candidate: CandidateState | str | Path,
    *,
    task_spec: CandidateTaskSpec,
    max_parameters: int,
) -> CandidateValidation:
    """Run the protected CPU-only candidate interface check before execution.

    This is deliberately narrower than :func:`execute_discovery_candidate`:
    it receives no task arrays, split labels, evaluator, device, or training
    controls.  It imports the candidate in a private temporary module and
    performs the same source/metadata/dummy-forward/loss/optimizer validation
    that the executor performs before creating the real model.  Calling it
    after exact manifest materialization lets the runner route a concrete
    callable-body interface failure through its bounded code-repair loop
    without first allocating Fold-3/GPU execution work.

    The returned validation record is source-hash bound.  A ``CandidateState``
    must still agree byte-for-byte with ``candidate_metadata()`` in the loaded
    module; this function never normalizes or rewrites either artifact.
    """

    if not isinstance(task_spec, CandidateTaskSpec):
        raise TypeError("task_spec must be CandidateTaskSpec")
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int) or max_parameters <= 0:
        raise DiscoveryExecutorError("max_parameters must be a positive integer")
    material = _candidate_material(candidate)
    try:
        validation, _ = _strict_candidate_module(
            material.source,
            task_spec=task_spec,
            max_parameters=max_parameters,
        )
    except (DiscoveryCandidateError, OSError) as exc:
        raise DiscoveryExecutorError(f"Candidate failed protected interface preflight: {exc}") from exc
    if material.candidate_metadata is not None and canonical_json(
        _json_plain(validation.metadata)
    ) != canonical_json(_json_plain(material.candidate_metadata)):
        raise DiscoveryExecutorError(
            "CandidateState metadata does not match candidate_metadata() in executable source"
        )
    return validation


_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SCHEMA_VERSION = (
    "cellscientist_chemical_dose_mixed_effect_probe_v1"
)
_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SEED = 20260802
_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE = 3
_CHEMICAL_DOSE_MIXED_EFFECT_EPSILON = 1e-12
_CHEMICAL_DOSE_MIXED_EFFECT_MIN_RATIO = 0.005


def preflight_chemical_dose_mixed_effect(
    candidate: CandidateState | str | Path,
    *,
    task_spec: CandidateTaskSpec,
    max_parameters: int,
    primary_symbol: str,
) -> dict[str, Any]:
    """Probe one named candidate module for a real chemical-dose interaction.

    This is a CPU-only functional preflight, not an evaluator.  It receives no
    BBBC arrays, targets, split identity, fold scalar, optimizer state, or
    training controls.  After the ordinary strict candidate validation, it
    builds a fresh CPU model under a fixed seed and forwards one fixed batch of
    synthetic ``pre``, ``chemical``, and ``dose`` inputs through the four
    corners ``(C0,D0)``, ``(C1,D0)``, ``(C0,open discovery)``, and ``(C1,open discovery)``.  A forward
    hook observes exactly one registered module with class name
    ``primary_symbol``.

    Let ``Yij`` be that hooked primary-module output at ``(Ci,Dj)``.  The
    chemical and dose marginals are respectively
    ``0.5 * ((Y10 - Y00) + (Y11 - Y01))`` and
    ``0.5 * ((Y01 - Y00) + (Y11 - Y10))``; the four-corner mixed effect is
    ``Y11 - Y10 - Y01 + Y00``.  The returned ratio is its L2 norm divided by
    the sum of the two marginal L2 norms plus ``1e-12``.  Both marginals must
    be nonzero and the ratio must be at least ``0.005``.  All values returned
    are scalar JSON primitives; no synthetic tensors are retained.
    """

    if not isinstance(task_spec, CandidateTaskSpec):
        raise TypeError("task_spec must be CandidateTaskSpec")
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, int) or max_parameters <= 0:
        raise DiscoveryExecutorError("max_parameters must be a positive integer")
    if not isinstance(primary_symbol, str) or not primary_symbol.isidentifier():
        raise DiscoveryExecutorError("primary_symbol must be a Python class identifier")

    torch, _, _ = _require_torch()
    # The strict validator owns a separate fixed dummy validation seed.  Reset
    # immediately afterwards so the probe model and all three synthetic input
    # tensors are deterministic and cannot inherit any prior executor state.
    torch.manual_seed(_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SEED)
    material = _candidate_material(candidate)
    try:
        validation, module = _strict_candidate_module(
            material.source,
            task_spec=task_spec,
            max_parameters=max_parameters,
        )
    except (DiscoveryCandidateError, OSError) as exc:
        raise DiscoveryExecutorError(
            f"Candidate failed protected chemical-dose preflight: {exc}"
        ) from exc
    if material.candidate_metadata is not None and canonical_json(
        _json_plain(validation.metadata)
    ) != canonical_json(_json_plain(material.candidate_metadata)):
        raise DiscoveryExecutorError(
            "CandidateState metadata does not match candidate_metadata() in executable source"
        )

    torch.manual_seed(_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SEED)
    try:
        model = module.build_model(task_spec.to_dict())
    except Exception as exc:
        raise DiscoveryExecutorError(
            f"Candidate failed CPU mixed-effect model construction: {type(exc).__name__}"
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise DiscoveryExecutorError("Candidate build_model() must return torch.nn.Module")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count <= 0 or parameter_count > max_parameters:
        raise DiscoveryExecutorError(
            "Candidate model parameter count violates the protected mixed-effect budget"
        )
    if parameter_count != int(validation.parameter_count):
        raise DiscoveryExecutorError(
            "Candidate model parameter count differs from protected interface validation"
        )
    try:
        model = model.to(torch.device("cpu"))
        model.eval()
    except Exception as exc:
        raise DiscoveryExecutorError(
            f"Candidate cannot enter the CPU mixed-effect probe: {type(exc).__name__}"
        ) from exc

    matching_modules = [
        (name, item)
        for name, item in model.named_modules()
        if item.__class__.__name__ == primary_symbol
    ]
    if len(matching_modules) != 1:
        raise DiscoveryExecutorError(
            "mixed-effect probe requires exactly one named module with class name "
            f"{primary_symbol!r}; found {len(matching_modules)}"
        )
    primary_name, primary_module = matching_modules[0]
    captured_outputs: list[Any] = []

    def _capture_primary_output(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        # Preserve the module's direct forward result before any later in-place
        # downstream operation can alter a view of it.
        if isinstance(output, torch.Tensor):
            captured_outputs.append(output.detach().clone())
        else:
            captured_outputs.append(output)

    def _validated_probe_output(value: Any, *, label: str, expected_width: int | None) -> Any:
        if not isinstance(value, torch.Tensor):
            raise DiscoveryExecutorError(
                f"mixed-effect probe {label} must be a tensor"
            )
        if value.device.type != "cpu":
            raise DiscoveryExecutorError(
                f"mixed-effect probe {label} must remain on CPU"
            )
        if value.ndim != 2 or int(value.shape[0]) != _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE:
            raise DiscoveryExecutorError(
                f"mixed-effect probe {label} has invalid shape {tuple(value.shape)}"
            )
        if int(value.shape[1]) <= 0 or (
            expected_width is not None and int(value.shape[1]) != int(expected_width)
        ):
            raise DiscoveryExecutorError(
                f"mixed-effect probe {label} has invalid shape {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise DiscoveryExecutorError(
                f"mixed-effect probe {label} is non-finite"
            )
        return value.detach().clone()

    torch.manual_seed(_CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SEED)
    pre = torch.randn(
        _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE,
        task_spec.context_dim,
        device="cpu",
    )
    chemical_0 = torch.randn(
        _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE,
        task_spec.chemical_dim,
        device="cpu",
    )
    chemical_1 = chemical_0 + torch.randn_like(chemical_0)
    dose_0 = torch.randn(
        _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE,
        task_spec.dose_dim,
        device="cpu",
    )
    dose_1 = dose_0 + torch.randn_like(dose_0)

    def _corner_output(*, chemical: Any, dose: Any, label: str) -> Any:
        captured_outputs.clear()
        try:
            with torch.no_grad():
                model_output = model(pre, chemical, dose)
        except DiscoveryExecutorError:
            raise
        except Exception as exc:
            raise DiscoveryExecutorError(
                f"Candidate failed CPU mixed-effect forward at {label}: {type(exc).__name__}"
            ) from exc
        _validated_probe_output(
            model_output,
            label=f"model output at {label}",
            expected_width=task_spec.target_dim,
        )
        if len(captured_outputs) != 1:
            raise DiscoveryExecutorError(
                f"mixed-effect probe primary module executed {len(captured_outputs)} times at {label}; expected exactly one"
            )
        return _validated_probe_output(
            captured_outputs[0],
            label=f"primary output at {label}",
            expected_width=None,
        )

    handle = primary_module.register_forward_hook(_capture_primary_output)
    try:
        y00 = _corner_output(chemical=chemical_0, dose=dose_0, label="C0D0")
        y10 = _corner_output(chemical=chemical_1, dose=dose_0, label="C1D0")
        y01 = _corner_output(chemical=chemical_0, dose=dose_1, label="C0D1")
        y11 = _corner_output(chemical=chemical_1, dose=dose_1, label="C1D1")
    finally:
        handle.remove()

    chemical_marginal = 0.5 * ((y10 - y00) + (y11 - y01))
    dose_marginal = 0.5 * ((y01 - y00) + (y11 - y10))
    mixed_effect = y11 - y10 - y01 + y00

    def _l2_norm(value: Any) -> float:
        result = float(torch.sqrt(value.square().sum()).item())
        if result < 0.0 or result == float("inf") or result != result:
            raise DiscoveryExecutorError("mixed-effect probe produced a non-finite summary")
        return result

    chemical_marginal_l2 = _l2_norm(chemical_marginal)
    dose_marginal_l2 = _l2_norm(dose_marginal)
    mixed_effect_l2 = _l2_norm(mixed_effect)
    if chemical_marginal_l2 <= _CHEMICAL_DOSE_MIXED_EFFECT_EPSILON:
        raise DiscoveryExecutorError("mixed-effect probe has zero chemical marginal")
    if dose_marginal_l2 <= _CHEMICAL_DOSE_MIXED_EFFECT_EPSILON:
        raise DiscoveryExecutorError("mixed-effect probe has zero dose marginal")
    denominator = (
        chemical_marginal_l2
        + dose_marginal_l2
        + _CHEMICAL_DOSE_MIXED_EFFECT_EPSILON
    )
    mixed_effect_ratio = mixed_effect_l2 / denominator
    if mixed_effect_ratio < _CHEMICAL_DOSE_MIXED_EFFECT_MIN_RATIO:
        raise DiscoveryExecutorError(
            "mixed-effect probe interaction ratio is below the registered minimum "
            f"{_CHEMICAL_DOSE_MIXED_EFFECT_MIN_RATIO:.3f}"
        )

    return {
        "schema_version": _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_SCHEMA_VERSION,
        "candidate_hash": material.candidate_hash,
        "source_sha256": str(validation.source_sha256),
        "primary_symbol": primary_symbol,
        "primary_module_name": str(primary_name),
        "device": "cpu",
        "synthetic_input_names": ["pre", "chemical", "dose"],
        "synthetic_batch_size": _CHEMICAL_DOSE_MIXED_EFFECT_PROBE_BATCH_SIZE,
        "corner_count": 4,
        "primary_output_shape": [int(value) for value in y00.shape],
        "chemical_marginal_l2": float(chemical_marginal_l2),
        "dose_marginal_l2": float(dose_marginal_l2),
        "mixed_effect_l2": float(mixed_effect_l2),
        "mixed_effect_denominator_l2": float(denominator),
        "mixed_effect_ratio": float(mixed_effect_ratio),
        "minimum_mixed_effect_ratio": _CHEMICAL_DOSE_MIXED_EFFECT_MIN_RATIO,
        "ratio_definition": (
            "||Y11-Y10-Y01+Y00||_2 / "
            "(||0.5*((Y10-Y00)+(Y11-Y01))||_2 + "
            "||0.5*((Y01-Y00)+(Y11-Y10))||_2 + 1e-12)"
        ),
        "candidate_metadata_verified": material.candidate_metadata is not None,
        "uses_targets": False,
        "uses_folds": False,
        "uses_training": False,
    }


def execute_discovery_candidate(
    arrays: BBBC036DiscoveryArrays,
    candidate: CandidateState | str | Path,
    *,
    settings: DiscoveryExecutorSettings | None = None,
    export_selected_artifacts: str | Path | None = None,
) -> DiscoveryExecutionResult:
    """Train one protected candidate on folds 1--2 and select on fold 3.

    ``arrays`` is the sole data argument.  No cache path, raw source path,
    endpoint loader, evaluator callback, target fold, or model-selection hook
    is accepted.  Fold 3 is evaluated after each candidate-requested legal
    epoch and is the only signal used for early stopping and model selection.

    ``export_selected_artifacts`` is reserved for a post-search finalization
    replay of one already frozen candidate. When supplied, it writes exactly
    one selected checkpoint and Fold-3 prediction/target pair after the
    executor restores its selected state. Discovery runners never pass this
    argument for intermediate candidates.
    """

    _validate_arrays(arrays)
    settings = settings or DiscoveryExecutorSettings()
    if not isinstance(settings, DiscoveryExecutorSettings):
        raise TypeError("Discovery executor settings must be DiscoveryExecutorSettings")
    tensors = _prepare_tensors(arrays)
    torch, DataLoader, TensorDataset = _require_torch()
    device = _resolve_device(torch, settings.device)
    _seed_executor(torch, settings.seed, device=device)
    material = _candidate_material(candidate)
    controls = candidate_training_controls(material.training_config, settings=settings)
    execution_started = time.perf_counter()
    try:
        validation, module = _strict_candidate_module(
            material.source,
            task_spec=tensors.task_spec,
            max_parameters=settings.max_parameters,
        )
    except (DiscoveryCandidateError, OSError) as exc:
        raise DiscoveryExecutorError(f"Candidate failed protected validation: {exc}") from exc
    if material.candidate_metadata is not None and canonical_json(_json_plain(validation.metadata)) != canonical_json(
        _json_plain(material.candidate_metadata)
    ):
        raise DiscoveryExecutorError("CandidateState metadata does not match candidate_metadata() in executable source")
    # Candidate validation intentionally seeds its dummy forward separately;
    # reset the executor-owned seed before building the real model.
    _seed_executor(torch, settings.seed, device=device)
    # Candidate receives dimensions only.  The executor retains all raw data,
    # normalization statistics, roles, and metrics outside this call.
    model = module.build_model(tensors.task_spec.to_dict())
    if not isinstance(model, torch.nn.Module):
        raise DiscoveryExecutorError("Candidate build_model() must return torch.nn.Module")
    actual_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if actual_parameter_count <= 0 or actual_parameter_count > settings.max_parameters:
        raise DiscoveryExecutorError("Candidate model parameter count differs from its protected validation budget")
    model = model.to(device)
    # Match the validated candidate interface exactly: optimizer/scheduler
    # construction receives the fixed epoch budget only, never feedback.
    optimizer_state = {"max_epochs": int(controls.max_epochs)}
    optimizer = module.build_optimizer(model, dict(optimizer_state))
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise DiscoveryExecutorError("Candidate build_optimizer() must return torch.optim.Optimizer")
    scheduler = None
    if callable(getattr(module, "build_scheduler", None)):
        scheduler = module.build_scheduler(optimizer, dict(optimizer_state))
        if scheduler is not None and not callable(getattr(scheduler, "step", None)):
            raise DiscoveryExecutorError("Candidate build_scheduler() must return None or a scheduler with step()")

    prefix_dim = tensors.task_spec.chemical_dim
    cp_dim = tensors.task_spec.cp_dim
    normalized_fit_pre = tensors.standardizer.normalize_pre(tensors.fit_pre)
    normalized_fit_condition = tensors.standardizer.normalize_condition(tensors.fit_condition)
    normalized_fit_target = tensors.standardizer.normalize_target(tensors.fit_target_raw)
    # Data-loader order is seeded by the executor and contains only fold-1/2
    # tensors.  Candidate code receives one tensor batch at a time.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.seed)
    loader = DataLoader(
        TensorDataset(normalized_fit_pre, normalized_fit_condition, normalized_fit_target),
        batch_size=min(int(controls.batch_size), int(normalized_fit_pre.shape[0])),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    initial_train = _metrics(
        torch,
        module,
        model,
        raw_pre=tensors.fit_pre,
        raw_condition=tensors.fit_condition,
        raw_target=tensors.fit_target_raw,
        standardizer=tensors.standardizer,
        prefix_dim=prefix_dim,
        cp_dim=cp_dim,
        device=device,
        epoch=0,
        max_epochs=controls.max_epochs,
    )
    initial_discovery = _metrics(
        torch,
        module,
        model,
        raw_pre=tensors.feedback_pre,
        raw_condition=tensors.feedback_condition,
        raw_target=tensors.feedback_target_raw,
        standardizer=tensors.standardizer,
        prefix_dim=prefix_dim,
        cp_dim=cp_dim,
        device=device,
        epoch=0,
        max_epochs=controls.max_epochs,
    )
    history: list[EpochDiscoveryMetrics] = [
        EpochDiscoveryMetrics(epoch=0, train=initial_train, discovery=initial_discovery, runtime_seconds=0.0)
    ]
    best_epoch = 0
    best_discovery = initial_discovery
    best_train = initial_train
    best_state = copy.deepcopy(model.state_dict())
    best_score = float(initial_discovery["global_pcc"])
    stale_epochs = 0
    stopped_early = False

    for epoch in range(1, int(controls.max_epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        for batch_pre, batch_condition, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_pre = batch_pre.to(device)
            batch_condition = batch_condition.to(device)
            batch_target = batch_target.to(device)
            chemical, dose = _split_inputs(batch_condition, prefix_dim=prefix_dim)
            prediction = model(batch_pre, chemical, dose)
            if tuple(prediction.shape) != tuple(batch_target.shape):
                raise DiscoveryExecutorError("Candidate output shape differs from protected target during training")
            loss = _candidate_loss(module, prediction, batch_target, epoch=epoch, max_epochs=controls.max_epochs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(controls.gradient_clip))
            optimizer.step()
        train_metrics = _metrics(
            torch,
            module,
            model,
            raw_pre=tensors.fit_pre,
            raw_condition=tensors.fit_condition,
            raw_target=tensors.fit_target_raw,
            standardizer=tensors.standardizer,
            prefix_dim=prefix_dim,
            cp_dim=cp_dim,
            device=device,
            epoch=epoch,
            max_epochs=controls.max_epochs,
        )
        discovery_metrics = _metrics(
            torch,
            module,
            model,
            raw_pre=tensors.feedback_pre,
            raw_condition=tensors.feedback_condition,
            raw_target=tensors.feedback_target_raw,
            standardizer=tensors.standardizer,
            prefix_dim=prefix_dim,
            cp_dim=cp_dim,
            device=device,
            epoch=epoch,
            max_epochs=controls.max_epochs,
        )
        history.append(
            EpochDiscoveryMetrics(
                epoch=epoch,
                train=train_metrics,
                discovery=discovery_metrics,
                runtime_seconds=time.perf_counter() - epoch_started,
            )
        )
        _scheduler_step(
            module,
            scheduler,
            epoch=epoch,
            max_epochs=controls.max_epochs,
            fold3_global_pcc=float(discovery_metrics["global_pcc"]),
        )
        if float(discovery_metrics["global_pcc"]) > best_score + float(controls.min_delta):
            best_epoch = epoch
            best_score = float(discovery_metrics["global_pcc"])
            best_train = train_metrics
            best_discovery = discovery_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(controls.patience):
                stopped_early = True
                break
    # The model state is deliberately not returned; restoring it here keeps the
    # executor's selection semantics explicit without opening a hidden endpoint
    # evaluation path in this module.
    model.load_state_dict(best_state)
    input_reliance_metrics, input_reliance_audit = _input_reliance_diagnostics(
        torch,
        model,
        arrays=arrays,
        tensors=tensors,
        device=device,
        # This executor-owned seed is registered in the result and cannot be
        # set through CandidateState.training_config or candidate source.
        map_seed=settings.seed,
    )
    # Do not mutate ``best_discovery``: that exact mapping is also retained in
    # the per-epoch history.  These post-selection diagnostics intentionally
    # belong only to the selected model state.
    selected_discovery = dict(best_discovery)
    selected_discovery.update(input_reliance_metrics)
    standardization = {
        "fit_only": True,
        "statistics_sha256": tensors.standardizer.fingerprint,
        "pre_dim": tensors.task_spec.context_dim,
        "condition_dim": tensors.task_spec.chemical_dim + tensors.task_spec.dose_dim,
        "target_dim": tensors.task_spec.target_dim,
    }
    access_audit = {
        "received_array_type": "BBBC036DiscoveryArrays",
        "fit_target_folds": [1, 2],
        "feedback_target_folds": [3],
        "withheld_target_folds": [4, 5],
        "candidate_received": [
            "tensor_batches",
            "dimension_only_task_spec",
            "epoch_state",
            "optional_fold3_scheduler_scalar",
        ],
        "candidate_not_received": ["paths", "cache_indices", "fold_ids", "evaluator", "fold4_targets", "fold5_targets"],
        "candidate_not_received_for_input_reliance": [
            "control_sham_map",
            "chemical_sham_map",
            "counterfactual_targets",
            "counterfactual_pcc",
        ],
        "selection_metric": "fold3_global_pcc",
    }
    result = DiscoveryExecutionResult(
        task_id=arrays.task_id,
        contract_hash=arrays.contract.contract_hash,
        candidate_hash=material.candidate_hash,
        candidate_validation=validation,
        settings=settings.to_dict(),
        candidate_training_controls=controls.to_dict(),
        standardization=standardization,
        epoch_metrics=tuple(history),
        selected_epoch=best_epoch,
        selected_train=best_train,
        selected_discovery=selected_discovery,
        input_reliance_audit=input_reliance_audit,
        stopped_early=stopped_early,
        completed_epochs=len(history) - 1,
        runtime_seconds=time.perf_counter() - execution_started,
        access_audit=access_audit,
    )
    # Protect trace serialization from leaking a tensor or an executor object.
    canonical_json(result.to_dict())
    if export_selected_artifacts is not None:
        _export_selected_discovery_artifacts(
            export_root=export_selected_artifacts,
            torch=torch,
            model=model,
            tensors=tensors,
            arrays=arrays,
            prefix_dim=prefix_dim,
            result=result,
            candidate_source_hash=str(validation.source_sha256),
            candidate_source=material.source,
            candidate_metadata=validation.metadata,
            candidate_training_config=material.training_config,
        )
    return result
