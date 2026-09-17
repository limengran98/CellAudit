"""Joint-response candidate training and evaluation on registered task folds.

The runtime exposes Folds 1--2 for fitting and Fold 3 for checkpoint
selection.  Held-out partitions are opened only through explicit audit
loaders.  Every candidate predicts the complete cellular-response vector with
one source-validated model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping

from .contracts import file_fingerprint, normalize_condition_layout
from .discovery_candidate import CandidateTaskSpec
from .discovery_executor import (
    _global_pcc,
    _json_plain,
    _metrics,
    _seed_executor,
    _split_inputs,
    _standardizer_fingerprint,
    _strict_candidate_module,
    _tensor_stats,
    _TensorStandardizer,
)
from .open_discovery import CandidateState
from .schemas import SchemaError, canonical_json, digest
from .tasks import get_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCHEMA = "cellscientist_joint_response_runtime_v1"
SUPPORTED_TASKS = frozenset(("cpg036_cp_plate_control_context", "cpg047_cp_plate_control_context"))
REGISTERED_STARTS = frozenset(("h0",))


class JointResponseRuntimeError(RuntimeError):
    """Raised when the protected joint-response protocol is violated."""


@dataclass(frozen=True)
class CPGPartition:
    """Tensor-compatible subset with no retained reference to withheld folds."""

    pre: Any
    condition: Any
    target: Any
    cp_target: Any
    l1000_target: Any
    labels: tuple[str, ...]
    cache_folds: Any

    @property
    def count(self) -> int:
        return int(self.pre.shape[0])


@dataclass(frozen=True)
class CPGFold13Arrays:
    task_id: str
    fit: CPGPartition
    selection: CPGPartition
    condition_layout: Mapping[str, Any]
    data_fingerprint: str
    metadata: Mapping[str, Any]

    @property
    def task_spec(self) -> CandidateTaskSpec:
        return CandidateTaskSpec(
            context_dim=int(self.fit.pre.shape[1]),
            chemical_dim=int(self.condition_layout["prefix_dim"]),
            dose_dim=1,
            cp_dim=int(self.fit.cp_target.shape[1]),
            l1000_dim=int(self.fit.l1000_target.shape[1]),
        )


@dataclass(frozen=True)
class CPGFold14Arrays(CPGFold13Arrays):
    """Fold-1--3 training arrays plus a sealed Fold-4 endpoint partition.

    This type is intentionally produced by a separate loader.  Discovery and
    Fold-3 selection code continue to use :class:`CPGFold13Arrays`, so opening
    the final evaluation partition is an explicit runner-level operation.
    """

    endpoint: CPGPartition


@dataclass(frozen=True)
class JointResponseSettings:
    seed: int
    max_epochs: int
    batch_size: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    gradient_clip_norm: float
    max_parameters: int
    device: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, device: str) -> "JointResponseSettings":
        value = config.get("training")
        if not isinstance(value, Mapping):
            raise JointResponseRuntimeError("runtime config lacks training settings")
        try:
            result = cls(
                seed=int(value["seed"]),
                max_epochs=int(value["max_epochs"]),
                batch_size=int(value["batch_size"]),
                early_stopping_patience=int(value["early_stopping_patience"]),
                early_stopping_min_delta=float(value["early_stopping_min_delta"]),
                gradient_clip_norm=float(value["gradient_clip_norm"]),
                max_parameters=int(value["max_parameters"]),
                device=str(device),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JointResponseRuntimeError("runtime training settings are invalid") from exc
        if (
            result.max_epochs < 1
            or result.batch_size < 2
            or result.early_stopping_patience < 0
            or result.gradient_clip_norm <= 0.0
            or result.max_parameters < 1
        ):
            raise JointResponseRuntimeError("runtime training settings are outside legal bounds")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_parameters": self.max_parameters,
            "device": self.device,
        }


def registered_candidate_seed(
    trajectory_seed: int | None,
    candidate_position: int,
    *,
    default_seed: int,
) -> int:
    """Return the registered training seed for one candidate slot.

    A discovery trajectory is the statistical unit, while its ten candidates
    must still receive a pre-registered, paired training-seed sequence.  The
    h0 evaluation is position 1 and the nine proposed candidates are positions
    2--10.  When no trajectory seed is supplied, the runtime uses the
    registered single-run seed from the task contract.
    """

    if isinstance(candidate_position, bool) or not isinstance(candidate_position, int):
        raise JointResponseRuntimeError("candidate position must be an integer")
    if candidate_position < 1 or candidate_position > 10:
        raise JointResponseRuntimeError("candidate position must lie in [1, 10]")
    if trajectory_seed is None:
        return int(default_seed)
    if isinstance(trajectory_seed, bool) or not isinstance(trajectory_seed, int):
        raise JointResponseRuntimeError("trajectory seed must be an integer")
    if trajectory_seed < 1:
        raise JointResponseRuntimeError("trajectory seed must be positive")
    seed = int(trajectory_seed) * 100 + int(candidate_position)
    if seed >= 2**63:
        raise JointResponseRuntimeError("registered candidate seed exceeds PyTorch range")
    return seed


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_runtime_config(path: str | Path) -> Mapping[str, Any]:
    config_path = project_path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointResponseRuntimeError(f"cannot read joint-response runtime config: {config_path}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != RUNTIME_SCHEMA:
        raise JointResponseRuntimeError("joint-response runtime config schema is invalid")
    expected = {
        "schema_version",
        "tasks",
        "fold_roles",
        "training",
        "starting_candidates",
        "selection",
        "prohibited_actions",
    }
    if set(value) != expected:
        raise JointResponseRuntimeError("joint-response runtime config fields are invalid")
    if tuple(value["tasks"]) != ("cpg036_cp_plate_control_context", "cpg047_cp_plate_control_context"):
        raise JointResponseRuntimeError("joint-response runtime task order changed")
    if value["fold_roles"] != {
        "fit_folds": [1, 2],
        "selection_fold": 3,
        "withheld_folds": [4, 5],
    }:
        raise JointResponseRuntimeError("joint-response runtime fold roles changed")
    if set(value["starting_candidates"]) != REGISTERED_STARTS:
        raise JointResponseRuntimeError("registered starting candidate changed")
    if value["selection"] != {
        "metric": "global_pcc",
        "partition": "fold_3_only",
        "checkpoint_rule": "strict_global_pcc_improvement",
    }:
        raise JointResponseRuntimeError("registered checkpoint-selection contract changed")
    if set(value["prohibited_actions"]) != {
        "fold4_target_access",
        "fold5_target_access",
        "per_target_model_training",
        "data_or_fold_modification",
        "evaluator_modification",
    }:
        raise JointResponseRuntimeError("joint-response runtime prohibitions changed")
    canonical_json(value)
    return value


def _decode(values: Any) -> tuple[str, ...]:
    return tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values)


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise JointResponseRuntimeError("joint-response execution requires numpy") from exc
    return np


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise JointResponseRuntimeError("joint-response execution requires torch") from exc
    return torch, DataLoader, TensorDataset


def _manifest_verified_cache(task_id: str) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], str]:
    if task_id not in SUPPORTED_TASKS:
        raise JointResponseRuntimeError(f"unsupported joint-response task: {task_id}")
    task = get_task(task_id)
    cache = project_path("cache") / f"{task_id}.npz"
    manifest_path = cache.with_suffix(".manifest.json")
    if not cache.is_file() or not manifest_path.is_file():
        raise JointResponseRuntimeError(f"registered CPG cache/manifest missing for {task_id}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JointResponseRuntimeError(f"invalid CPG manifest for {task_id}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("adapter") != "cpg_replicate_profiles":
        raise JointResponseRuntimeError("runtime requires the raw CPG replicate-profile adapter")
    if manifest.get("direction") != "joint_from_condition_with_cp_control_context":
        raise JointResponseRuntimeError("runtime requires the registered control-context response task")
    task_layout = normalize_condition_layout(task.get("condition_layout"))
    manifest_layout = normalize_condition_layout(manifest.get("condition_layout"))
    if task_layout is None or task_layout != manifest_layout:
        raise JointResponseRuntimeError("registered CPG condition layout does not replay")
    sources = task.get("sources")
    if not isinstance(sources, Mapping):
        raise JointResponseRuntimeError("registered CPG task lacks source mapping")
    for modality in ("cell_painting", "l1000"):
        source = Path(str(sources.get(modality, ""))).resolve()
        record = manifest.get(modality)
        if not source.is_file() or not isinstance(record, Mapping):
            raise JointResponseRuntimeError(f"registered {modality} source/manifest entry is absent")
        if Path(str(record.get("source_path", ""))).resolve() != source:
            raise JointResponseRuntimeError(f"registered {modality} source path changed")
        if record.get("source_fingerprint") != file_fingerprint(source):
            raise JointResponseRuntimeError(f"registered {modality} source fingerprint changed")
    data_fingerprint = digest(
        {
            "task_id": task_id,
            "cache_sha256": sha256_file(cache),
            "manifest_sha256": sha256_file(manifest_path),
            "condition_layout": manifest_layout,
            "fold_roles": {"fit": [1, 2], "selection": 3, "withheld": [4, 5]},
        }
    )
    return cache, task, manifest, data_fingerprint


def _load_fold_arrays(task_id: str, *, include_endpoint: bool) -> CPGFold13Arrays | CPGFold14Arrays:
    """Load the registered fit/selection partitions and, optionally, Fold 4.

    ``include_endpoint`` is used solely by the frozen endpoint runner after a
    discovery campaign has been sealed.  It never exposes Fold 5.
    """

    np = _require_numpy()
    cache, _task, manifest, data_fingerprint = _manifest_verified_cache(task_id)
    with np.load(cache, allow_pickle=False) as archive:
        required = {"pre", "condition", "target", "cp_response", "l1000_response", "fold", "labels"}
        missing = required.difference(archive.files)
        if missing:
            raise JointResponseRuntimeError(f"CPG cache is missing {sorted(missing)}")
        folds = archive["fold"].astype(np.int8, copy=False)
        labels = _decode(archive["labels"])
        if set(int(value) for value in folds.tolist()) != {0, 1, 2, 3, 4}:
            raise JointResponseRuntimeError("CPG cache does not retain all registered folds")
        fit_indices = np.flatnonzero(np.isin(folds, [0, 1]))
        selection_indices = np.flatnonzero(folds == 2)
        endpoint_indices = np.flatnonzero(folds == 3) if include_endpoint else None
        if not len(fit_indices) or not len(selection_indices):
            raise JointResponseRuntimeError("registered fit/selection folds are empty")
        if include_endpoint and (endpoint_indices is None or not len(endpoint_indices)):
            raise JointResponseRuntimeError("registered Fold-4 endpoint partition is empty")
        # The cache is a compressed, condition-level NPZ.  The executor owns
        # the archive and slices only its registered fit/selection indices;
        # no withheld partition object is constructed or returned.
        raw = {
            name: archive[name].astype(np.float32, copy=False)
            for name in ("pre", "condition", "target", "cp_response", "l1000_response")
        }
    if len(labels) != int(raw["pre"].shape[0]) or len(set(labels)) != len(labels):
        raise JointResponseRuntimeError("CPG cache labels are not unique and row aligned")
    layout = normalize_condition_layout(manifest.get("condition_layout"))
    if layout is None or int(layout["prefix_dim"]) + 1 != int(raw["condition"].shape[1]):
        raise JointResponseRuntimeError("CPG cache does not replay the chemical-prefix/dose layout")
    if int(raw["target"].shape[1]) != int(raw["cp_response"].shape[1]) + int(raw["l1000_response"].shape[1]):
        raise JointResponseRuntimeError("CPG joint target blocks do not reconstruct the target")

    def partition(indices: Any) -> CPGPartition:
        return CPGPartition(
            pre=raw["pre"][indices],
            condition=raw["condition"][indices],
            target=raw["target"][indices],
            cp_target=raw["cp_response"][indices],
            l1000_target=raw["l1000_response"][indices],
            labels=tuple(labels[int(index)] for index in indices.tolist()),
            cache_folds=folds[indices].copy(),
        )

    fit = partition(fit_indices)
    selection = partition(selection_indices)
    for name, value, allowed in (("fit", fit, {0, 1}), ("selection", selection, {2})):
        if {int(item) for item in value.cache_folds.tolist()} != allowed:
            raise JointResponseRuntimeError(f"{name} partition fold boundary is invalid")
    endpoint = partition(endpoint_indices) if endpoint_indices is not None else None
    if endpoint is not None and {int(item) for item in endpoint.cache_folds.tolist()} != {3}:
        raise JointResponseRuntimeError("endpoint partition fold boundary is invalid")
    metadata = {
        "task_id": task_id,
        "data_fingerprint": data_fingerprint,
        "cache_path": str(cache),
        "loaded_target_folds": [1, 2, 3, 4] if include_endpoint else [1, 2, 3],
        "withheld_target_folds": [5] if include_endpoint else [4, 5],
        "fit_rows": fit.count,
        "selection_rows": selection.count,
        "condition_layout": dict(layout),
    }
    if endpoint is not None:
        return CPGFold14Arrays(
            task_id=task_id,
            fit=fit,
            selection=selection,
            endpoint=endpoint,
            condition_layout=dict(layout),
            data_fingerprint=data_fingerprint,
            metadata={**metadata, "endpoint_rows": endpoint.count},
        )
    return CPGFold13Arrays(
        task_id=task_id,
        fit=fit,
        selection=selection,
        condition_layout=dict(layout),
        data_fingerprint=data_fingerprint,
        metadata=metadata,
    )


def load_fold13_arrays(task_id: str) -> CPGFold13Arrays:
    """Return only Fold-1/2 fit and Fold-3 selection tensors for one task."""

    arrays = _load_fold_arrays(task_id, include_endpoint=False)
    if not isinstance(arrays, CPGFold13Arrays) or isinstance(arrays, CPGFold14Arrays):
        raise JointResponseRuntimeError("Fold-1--3 loader returned an invalid partition type")
    return arrays


def load_fold14_arrays(task_id: str) -> CPGFold14Arrays:
    """Return Fold-1--3 training arrays and the previously sealed Fold-4 endpoint."""

    arrays = _load_fold_arrays(task_id, include_endpoint=True)
    if not isinstance(arrays, CPGFold14Arrays):
        raise JointResponseRuntimeError("Fold-1--4 loader returned an invalid partition type")
    return arrays


def _candidate_from_payload(payload: Mapping[str, Any]) -> CandidateState:
    fields = {
        "candidate_source",
        "source_hash",
        "candidate_metadata",
        "candidate_metadata_hash",
        "training_config",
        "training_config_hash",
        "parent_candidate_hash",
        "candidate_hash",
    }
    if set(payload) != fields:
        raise JointResponseRuntimeError("frozen candidate payload fields are invalid")
    candidate = CandidateState(
        candidate_source=str(payload["candidate_source"]),
        candidate_metadata=payload["candidate_metadata"],
        training_config=payload["training_config"],
        parent_candidate_hash=payload["parent_candidate_hash"],
    )
    if candidate.to_dict() != dict(payload):
        raise JointResponseRuntimeError("frozen candidate source/metadata no longer replay")
    return candidate


def load_starting_candidate(model_id: str, config: Mapping[str, Any]) -> CandidateState:
    sources = config["starting_candidates"]
    if model_id not in sources:
        raise JointResponseRuntimeError(f"missing registered starting candidate for {model_id}")
    entry = sources[model_id]
    path = project_path(entry["path"])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointResponseRuntimeError(f"cannot read starting candidate source {path}") from exc
    payload = artifact.get("candidate_state", artifact) if model_id == "h0" else artifact
    if not isinstance(payload, Mapping):
        raise JointResponseRuntimeError("starting candidate artifact lacks candidate_state")
    candidate = _candidate_from_payload(payload)
    if candidate.candidate_hash != entry["candidate_hash"]:
        raise JointResponseRuntimeError("starting candidate hash differs from registered source")
    return candidate


def load_candidate_state_path(path: str | Path, *, expected_hash: str | None = None) -> CandidateState:
    """Load one immutable discovery candidate for provider-free replay.

    This is deliberately source-only: it accepts a previously saved candidate
    state and never imports a discovery policy, prompt, provider, or trajectory
    feedback.  ``expected_hash`` binds the replay to the frozen manifest.
    """

    source_path = project_path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointResponseRuntimeError(f"cannot read frozen candidate source {source_path}") from exc
    if not isinstance(payload, Mapping):
        raise JointResponseRuntimeError("frozen candidate state must be a JSON object")
    candidate = _candidate_from_payload(payload)
    if expected_hash is not None and candidate.candidate_hash != str(expected_hash):
        raise JointResponseRuntimeError("frozen candidate hash differs from endpoint manifest")
    return candidate


def build_candidate_runner(
    candidate: CandidateState,
    *,
    task_spec: CandidateTaskSpec,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build one source-validated joint-output candidate under the shared runner.

    Candidate source may determine the architecture, loss,
    optimizer and scheduler, but never receives a dataset, a fold identifier,
    a target outside Fold 1--3, or an evaluator callback.
    """

    validation, module = _strict_candidate_module(
        candidate.candidate_source,
        task_spec=task_spec,
        max_parameters=int(config["training"]["max_parameters"]),
    )
    if canonical_json(_json_plain(validation.metadata)) != canonical_json(_json_plain(candidate.candidate_metadata)):
        raise JointResponseRuntimeError("candidate metadata differs from protected source")
    model = module.build_model(task_spec.to_dict())
    optimizer = module.build_optimizer(model, {"max_epochs": int(config["training"]["max_epochs"])})
    scheduler = module.build_scheduler(optimizer, {"max_epochs": int(config["training"]["max_epochs"])}) if callable(getattr(module, "build_scheduler", None)) else None
    return {
        "model": model,
        "forward": lambda pre, chemical, dose: model(pre, chemical, dose),
        "loss": lambda prediction, target, epoch: module.compute_loss(
            prediction, target, {
                "epoch": int(epoch),
                "max_epochs": int(config["training"]["max_epochs"]),
                "cp_dim": int(task_spec.cp_dim),
                "l1000_dim": int(task_spec.l1000_dim),
            }
        ),
        "optimizer": optimizer,
        "scheduler": scheduler,
        "module": module,
        "metadata": {
            "implementation": "source_validated_joint_output_candidate",
            "candidate_hash": candidate.candidate_hash,
            "candidate_validation": validation.to_dict(),
            "output_mode": "single_joint_vector",
        },
    }


def prepare_joint_tensors(arrays: CPGFold13Arrays):
    torch, _, _ = _require_torch()
    fit_pre = torch.as_tensor(arrays.fit.pre, dtype=torch.float32, device="cpu")
    fit_condition = torch.as_tensor(arrays.fit.condition, dtype=torch.float32, device="cpu")
    fit_target = torch.as_tensor(arrays.fit.target, dtype=torch.float32, device="cpu")
    selection_pre = torch.as_tensor(arrays.selection.pre, dtype=torch.float32, device="cpu")
    selection_condition = torch.as_tensor(arrays.selection.condition, dtype=torch.float32, device="cpu")
    selection_target = torch.as_tensor(arrays.selection.target, dtype=torch.float32, device="cpu")
    values = (_tensor_stats(torch, value) for value in (fit_pre, fit_condition, fit_target))
    pre_mean, pre_scale = next(values)
    condition_mean, condition_scale = next(values)
    target_mean, target_scale = next(values)
    standardizer = _TensorStandardizer(
        pre_mean=pre_mean,
        pre_scale=pre_scale,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        fingerprint=_standardizer_fingerprint(
            pre_mean, pre_scale, condition_mean, condition_scale, target_mean, target_scale
        ),
    )
    return SimpleNamespace(
        fit_pre=fit_pre,
        fit_condition=fit_condition,
        fit_target=fit_target,
        selection_pre=selection_pre,
        selection_condition=selection_condition,
        selection_target=selection_target,
        standardizer=standardizer,
        task_spec=arrays.task_spec,
    )


def evaluate_joint_model(
    torch: Any,
    runner: Mapping[str, Any],
    *,
    pre: Any,
    condition: Any,
    target: Any,
    tensors: Any,
    arrays: CPGFold13Arrays,
    device: Any,
    epoch: int,
    max_epochs: int,
) -> dict[str, float]:
    model = runner["model"]
    model.eval()
    with torch.no_grad():
        normalized_pre = tensors.standardizer.normalize_pre(pre).to(device)
        normalized_condition = tensors.standardizer.normalize_condition(condition).to(device)
        normalized_target = tensors.standardizer.normalize_target(target).to(device)
        chemical, dose = _split_inputs(normalized_condition, prefix_dim=tensors.task_spec.chemical_dim)
        prediction = runner["forward"](normalized_pre, chemical, dose)
        if tuple(prediction.shape) != tuple(normalized_target.shape):
            raise JointResponseRuntimeError("candidate changed the registered joint output shape")
        loss = runner.get("evaluation_loss", runner["loss"])(prediction, normalized_target, epoch)
        if not bool(torch.isfinite(loss).item()):
            raise JointResponseRuntimeError("candidate produced non-finite validation loss")
        raw_prediction = tensors.standardizer.inverse_target(prediction)
        raw_target = target.to(device)
        cp_dim = tensors.task_spec.cp_dim
        metrics = {
            "optimization_loss": float(loss.detach().cpu()),
            "global_pcc": _global_pcc(torch, raw_prediction, raw_target),
            "mse": float((raw_prediction - raw_target).square().mean().detach().cpu()),
            "cp_pcc": _global_pcc(torch, raw_prediction[:, :cp_dim], raw_target[:, :cp_dim]),
            "l1000_pcc": _global_pcc(torch, raw_prediction[:, cp_dim:], raw_target[:, cp_dim:]),
        }
        metrics.update(
            _response_sensitive_metrics(
                torch,
                prediction=raw_prediction,
                target=raw_target,
                fit_target=tensors.fit_target.to(device),
                cp_dim=cp_dim,
            )
        )
        return metrics


def evaluate_endpoint_model(
    torch: Any,
    runner: Mapping[str, Any],
    *,
    endpoint: CPGPartition | None,
    tensors: Any,
    arrays: CPGFold13Arrays,
    device: Any,
    epoch: int,
    max_epochs: int,
) -> dict[str, float] | None:
    """Evaluate one selected checkpoint on Fold 4 after Fold-3 selection.

    The conversion to tensors occurs only here, after the selected state has
    been fixed.  The returned scalars are therefore never available to model
    construction, optimizer updates, early stopping, or checkpoint selection.
    """

    if endpoint is None:
        return None
    endpoint_pre = torch.as_tensor(endpoint.pre, dtype=torch.float32, device="cpu")
    endpoint_condition = torch.as_tensor(endpoint.condition, dtype=torch.float32, device="cpu")
    endpoint_target = torch.as_tensor(endpoint.target, dtype=torch.float32, device="cpu")
    return evaluate_joint_model(
        torch,
        runner,
        pre=endpoint_pre,
        condition=endpoint_condition,
        target=endpoint_target,
        tensors=tensors,
        arrays=arrays,
        device=device,
        epoch=epoch,
        max_epochs=max_epochs,
    )


def _response_sensitive_metrics(
    torch: Any,
    *,
    prediction: Any,
    target: Any,
    fit_target: Any,
    cp_dim: int,
) -> dict[str, float]:
    """Evaluate pre-registered response-sensitive top-k feature subsets.

    Feature membership is derived only from the absolute mean control-relative
    response over fit folds 1--2.  This avoids using Fold-3 labels to define
    the feature subsets.  We report both Cell Painting and L1000 blocks; the
    names intentionally avoid calling morphology features ``DEGs``.
    """

    metrics: dict[str, float] = {}
    blocks = (("cp", 0, cp_dim), ("l1000", cp_dim, int(target.shape[1])))
    for block_name, start, stop in blocks:
        fit_block = fit_target[:, start:stop]
        response_strength = fit_block.abs().mean(dim=0)
        for k in (20, 50):
            actual_k = min(int(k), int(response_strength.numel()))
            if actual_k < 1:
                raise JointResponseRuntimeError("registered response-sensitive block is empty")
            top_indices = torch.topk(response_strength, k=actual_k, largest=True, sorted=True).indices
            prediction_block = prediction[:, start:stop].index_select(1, top_indices)
            target_block = target[:, start:stop].index_select(1, top_indices)
            metrics[f"{block_name}_response_rmse_top{k}"] = float(
                (prediction_block - target_block).square().mean().sqrt().detach().cpu()
            )
            metrics[f"{block_name}_response_pcc_top{k}"] = _global_pcc(torch, prediction_block, target_block)
    return metrics


def train_joint_candidate(
    model_id: str,
    *,
    arrays: CPGFold13Arrays,
    config: Mapping[str, Any],
    settings: JointResponseSettings,
    checkpoint_path: Path | None = None,
    candidate_override: CandidateState | None = None,
    endpoint: CPGPartition | None = None,
) -> dict[str, Any]:
    """Fit one frozen joint-response candidate and select its checkpoint."""

    candidate = candidate_override
    if candidate is None:
        if model_id not in REGISTERED_STARTS:
            raise JointResponseRuntimeError("candidate source must be supplied for an unregistered model id")
        candidate = load_starting_candidate(model_id, config)
    if candidate_override is not None:
        fixed_controls = {
            "batch_size": int(settings.batch_size),
            "max_epochs": int(settings.max_epochs),
            "patience": int(settings.early_stopping_patience),
            "gradient_clip_norm": float(settings.gradient_clip_norm),
            "min_delta": float(settings.early_stopping_min_delta),
        }
        if _json_plain(candidate_override.training_config) != fixed_controls:
            raise JointResponseRuntimeError(
                "the protected runtime requires every candidate to retain the registered training controls"
            )
    torch, DataLoader, TensorDataset = _require_torch()
    device = torch.device(settings.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise JointResponseRuntimeError("CUDA candidate execution was requested but unavailable")
    _seed_executor(torch, settings.seed, device=device)
    tensors = prepare_joint_tensors(arrays)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runner = build_candidate_runner(candidate, task_spec=tensors.task_spec, config=config)
    model = runner["model"]
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameters < 1 or parameters > settings.max_parameters:
        raise JointResponseRuntimeError("candidate parameter count violates the registered limit")
    model = model.to(device)
    dataset = TensorDataset(
        tensors.standardizer.normalize_pre(tensors.fit_pre),
        tensors.standardizer.normalize_condition(tensors.fit_condition),
        tensors.standardizer.normalize_target(tensors.fit_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(settings.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(settings.seed),
        drop_last=False,
    )
    best_state = None
    best_metrics: dict[str, float] | None = None
    best_epoch = -1
    epochs: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for batch_pre, batch_condition, batch_target in loader:
            batch_pre = batch_pre.to(device)
            batch_condition = batch_condition.to(device)
            batch_target = batch_target.to(device)
            chemical, dose = _split_inputs(batch_condition, prefix_dim=tensors.task_spec.chemical_dim)
            runner["optimizer"].zero_grad(set_to_none=True)
            prediction = runner["forward"](batch_pre, chemical, dose)
            if tuple(prediction.shape) != tuple(batch_target.shape):
                raise JointResponseRuntimeError("candidate changed the joint output shape in training")
            loss = runner["loss"](prediction, batch_target, epoch)
            if not bool(torch.isfinite(loss).item()):
                raise JointResponseRuntimeError("candidate produced non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            runner["optimizer"].step()
        selection_metrics = evaluate_joint_model(
            torch,
            runner,
            pre=tensors.selection_pre,
            condition=tensors.selection_condition,
            target=tensors.selection_target,
            tensors=tensors,
            arrays=arrays,
            device=device,
            epoch=epoch,
            max_epochs=settings.max_epochs,
        )
        epochs.append({"epoch": epoch, "selection": selection_metrics})
        scheduler = runner.get("scheduler")
        if scheduler is not None:
            callback = getattr(runner.get("module"), "step_scheduler", None)
            if callable(callback):
                callback(scheduler, {"epoch": epoch, "max_epochs": settings.max_epochs, "fold3_global_pcc": selection_metrics["global_pcc"]})
            else:
                scheduler.step()
        if best_metrics is None or selection_metrics["global_pcc"] > best_metrics["global_pcc"] + settings.early_stopping_min_delta:
            best_metrics = selection_metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= settings.early_stopping_patience:
                break
    if best_state is None or best_metrics is None:
        raise JointResponseRuntimeError("candidate did not produce a selected Fold-3 checkpoint")
    model.load_state_dict(best_state, strict=True)
    endpoint_metrics = evaluate_endpoint_model(
        torch,
        runner,
        endpoint=endpoint,
        tensors=tensors,
        arrays=arrays,
        device=device,
        epoch=best_epoch,
        max_epochs=settings.max_epochs,
    )
    stopped_early = len(epochs) < settings.max_epochs
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "cellscientist_joint_response_checkpoint_v1",
                "task_id": arrays.task_id,
                "model_id": model_id,
                "data_fingerprint": arrays.data_fingerprint,
                "settings": settings.to_dict(),
                "selected_epoch": best_epoch,
                "selection_metrics": best_metrics,
                "standardization_fingerprint": tensors.standardizer.fingerprint,
                "model_state_dict": best_state,
            },
            checkpoint_path,
        )
    return {
        "schema_version": "cellscientist_joint_response_record_v1",
        "status": "complete",
        "model_id": model_id,
        "task_id": arrays.task_id,
        "data_fingerprint": arrays.data_fingerprint,
        "selection_metrics": best_metrics,
        "endpoint_metrics": endpoint_metrics,
        "selected_epoch": best_epoch,
        "completed_epochs": len(epochs),
        "stopped_early": stopped_early,
        "parameter_count": parameters,
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "training_settings": settings.to_dict(),
        "implementation": runner["metadata"],
        "candidate_hash": candidate.candidate_hash,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "access_audit": {
            "fit_target_folds": [1, 2],
            "selection_target_folds": [3],
            "withheld_target_folds": [4, 5],
            "model_receives": ["normalized_fit_tensors", "normalized_fold3_input_tensors"],
            "model_does_not_receive": ["fold4_target", "fold5_target", "fold_identifier", "evaluator"],
            "per_target_training": False,
        },
        "epochs": epochs,
    }


def _safe_message(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").replace("\r", " ")
    return text[:320]


__all__ = [
    "CPGFold13Arrays",
    "CPGFold14Arrays",
    "CPGPartition",
    "JointResponseSettings",
    "JointResponseRuntimeError",
    "build_candidate_runner",
    "evaluate_endpoint_model",
    "evaluate_joint_model",
    "load_runtime_config",
    "load_starting_candidate",
    "load_fold13_arrays",
    "load_fold14_arrays",
    "load_candidate_state_path",
    "prepare_joint_tensors",
    "project_path",
    "registered_candidate_seed",
    "train_joint_candidate",
    "sha256_file",
]
