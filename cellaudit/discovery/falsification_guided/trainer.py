"""Fold-1--3-only training and target-related diagnostics for falsification-guided discovery."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellaudit.discovery_executor import _global_pcc, _seed_executor, _split_inputs
from cellaudit.perturbation_loop_data import (
    build_chemical_prefix_sham_map,
    build_matched_dose_sham_map,
)
from cellaudit.schemas import SchemaError, canonical_json
from cellaudit.joint_response_runtime import (
    JointResponseSettings,
    evaluate_joint_model,
    prepare_joint_tensors,
    load_runtime_config,
    load_fold13_arrays,
    project_path,
)

from .model import FalsificationGuidedDesign, PerturbationIncrementModel


PROTOCOL_PATH = "configs/falsification_guided_protocol.json"
ANCHOR_HIDDEN_DIM = 384
ANCHOR_SEED_OFFSET = 1
TRAIN_MAP_SEED = 47101
DEVELOPMENT_MAP_SEEDS = tuple(range(47201, 47209))


class FalsificationGuidedTrainingError(RuntimeError):
    """falsification-guided discovery cannot satisfy its protected training or evidence contract."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _protocol() -> Mapping[str, Any]:
    path = project_path(PROTOCOL_PATH)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FalsificationGuidedTrainingError("falsification-guided discovery protocol is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != "cellscientist_falsification_guided_protocol":
        raise FalsificationGuidedTrainingError("falsification-guided discovery protocol schema changed")
    if value.get("data_boundary", {}).get("fold5_opened") is not False:
        raise FalsificationGuidedTrainingError("falsification-guided discovery protocol no longer seals Fold 5")
    return value


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _corr_loss(prediction: Any, target: Any) -> Any:
    pred = prediction.reshape(-1) - prediction.mean()
    truth = target.reshape(-1) - target.mean()
    corr = (pred * truth).sum() / torch.sqrt(
        (pred.square().sum() * truth.square().sum()).clamp_min(1e-12)
    )
    return 1.0 - corr


def _group_objective(prediction: Any, target: Any, *, cp_dim: int, correlation_weight: float) -> Any:
    losses = []
    for pred_group, target_group in (
        (prediction[:, :cp_dim], target[:, :cp_dim]),
        (prediction[:, cp_dim:], target[:, cp_dim:]),
    ):
        losses.append(torch.nn.functional.mse_loss(pred_group, target_group) + correlation_weight * _corr_loss(pred_group, target_group))
    return 0.5 * (losses[0] + losses[1])


def _runner(model: Any, *, cp_dim: int, objective: str) -> Mapping[str, Any]:
    correlation_weight = 0.05 if objective == "correlation_residual" else 0.02
    return {
        "model": model,
        "forward": lambda pre, chemical, dose: model(pre, chemical, dose),
        "loss": lambda prediction, target, _epoch: _group_objective(
            prediction, target, cp_dim=cp_dim, correlation_weight=correlation_weight
        ),
    }


def _metrics(model: Any, *, arrays: Any, tensors: Any, settings: JointResponseSettings, objective: str, epoch: int) -> Mapping[str, float]:
    return evaluate_joint_model(
        torch,
        _runner(model, cp_dim=int(tensors.task_spec.cp_dim), objective=objective),
        pre=tensors.selection_pre,
        condition=tensors.selection_condition,
        target=tensors.selection_target,
        tensors=tensors,
        arrays=arrays,
        device=torch.device(settings.device),
        epoch=epoch,
        max_epochs=settings.max_epochs,
    )


def _references(tensors: Any) -> tuple[Any, Any]:
    normalized = tensors.standardizer.normalize_condition(tensors.fit_condition)
    prefix = int(tensors.task_spec.chemical_dim)
    chemical, dose = _split_inputs(normalized, prefix_dim=prefix)
    return chemical.mean(dim=0, keepdim=True), dose.mean(dim=0, keepdim=True)


def _new_model(*, tensors: Any, design: FalsificationGuidedDesign) -> PerturbationIncrementModel:
    chemical_reference, dose_reference = _references(tensors)
    spec = tensors.task_spec
    return PerturbationIncrementModel(
        context_dim=int(spec.context_dim),
        chemical_dim=int(spec.chemical_dim),
        dose_dim=int(spec.dose_dim),
        cp_dim=int(spec.cp_dim),
        l1000_dim=int(spec.l1000_dim),
        chemical_reference=chemical_reference,
        dose_reference=dose_reference,
        design=design,
        anchor_hidden_dim=ANCHOR_HIDDEN_DIM,
    )


def train_context_anchor(
    *,
    task_id: str,
    checkpoint_path: str | Path,
    device: str,
    seed: int,
    max_epochs: int | None = None,
) -> Mapping[str, Any]:
    """Fit one shared context-only anchor without exposing Fold 4 or Fold 5."""

    _protocol()
    path = project_path(checkpoint_path)
    if path.exists() or not path.is_relative_to(project_path("runs")):
        raise FalsificationGuidedTrainingError("falsification-guided discovery anchor checkpoint must be a new file below runs/")
    arrays = load_fold13_arrays(task_id)
    config = load_runtime_config("configs/joint_response_runtime.json")
    settings = JointResponseSettings.from_config(config, device=device)
    if max_epochs is not None:
        settings = replace(settings, max_epochs=int(max_epochs), early_stopping_patience=min(settings.early_stopping_patience, int(max_epochs)))
    device_obj = torch.device(device)
    _seed_executor(torch, int(seed), device=device_obj)
    tensors = prepare_joint_tensors(arrays)
    design = FalsificationGuidedDesign()
    model = _new_model(tensors=tensors, design=design).to(device_obj)
    optimizer = torch.optim.AdamW(model.anchor_parameters(), lr=1e-3, weight_decay=1e-5)
    dataset = TensorDataset(
        tensors.standardizer.normalize_pre(tensors.fit_pre),
        tensors.standardizer.normalize_target(tensors.fit_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(settings.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    epochs: list[Mapping[str, Any]] = []
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for pre, target in loader:
            pre, target = pre.to(device_obj), target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.anchor_prediction(pre)
            loss = _group_objective(prediction, target, cp_dim=int(tensors.task_spec.cp_dim), correlation_weight=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.anchor_parameters(), settings.gradient_clip_norm)
            optimizer.step()
        model.eval()
        metrics = _metrics(model, arrays=arrays, tensors=tensors, settings=settings, objective=design.objective, epoch=epoch)
        epochs.append({"epoch": epoch, "selection": dict(metrics)})
        if best_metrics is None or float(metrics["global_pcc"]) > float(best_metrics["global_pcc"]) + settings.early_stopping_min_delta:
            best_state = _state_dict_cpu(model.anchor)
            best_metrics = dict(metrics)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= settings.early_stopping_patience:
                break
    if best_state is None or best_metrics is None:
        raise FalsificationGuidedTrainingError("falsification-guided discovery context anchor did not produce a Fold-3 checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "cellscientist_falsification_guided_context_anchor_checkpoint_v1",
        "status": "complete",
        "provider_calls": 0,
        "task_id": task_id,
        "data_fingerprint": arrays.data_fingerprint,
        "seed": int(seed),
        "selected_epoch": best_epoch,
        "selection_metrics": best_metrics,
        "anchor_hidden_dim": ANCHOR_HIDDEN_DIM,
        "standardizer_fingerprint": tensors.standardizer.fingerprint,
        "model_state_dict": best_state,
        "completed_epochs": len(epochs),
        "runtime_seconds": time.perf_counter() - started,
        "fold4_or_fold5_used": False,
        "epochs": epochs,
    }
    torch.save(payload, path)
    return {**payload, "model_state_dict": "stored_in_checkpoint", "checkpoint_path": str(path)}


def _load_anchor(*, path: Path, task_id: str, arrays: Any, tensors: Any) -> Mapping[str, Any]:
    if not path.is_file():
        raise FalsificationGuidedTrainingError("falsification-guided discovery context anchor checkpoint is missing")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "cellscientist_falsification_guided_context_anchor_checkpoint_v1"
        or payload.get("task_id") != task_id
        or payload.get("data_fingerprint") != arrays.data_fingerprint
        or payload.get("standardizer_fingerprint") != tensors.standardizer.fingerprint
        or payload.get("fold4_or_fold5_used") is not False
    ):
        raise FalsificationGuidedTrainingError("falsification-guided discovery context anchor checkpoint violates its frozen identity")
    return payload


def _dose_identifiability_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    value = protocol.get("development_diagnostics", {}).get("dose_identifiability")
    if not isinstance(value, Mapping):
        raise FalsificationGuidedTrainingError("falsification-guided discovery dose-identifiability contract is missing")
    return value


def _training_shams(
    *, arrays: Any, tensors: Any, protocol: Mapping[str, Any]
) -> tuple[Any, Any, Any, Any, Any, Any, Mapping[str, Any]]:
    prefix = int(tensors.task_spec.chemical_dim)
    raw = np.asarray(arrays.fit.condition, dtype=np.float32)
    chemical_map = build_chemical_prefix_sham_map(
        arrays.fit.pre, raw[:, :prefix], row_ids=arrays.fit.labels, seed=TRAIN_MAP_SEED
    )
    chemical_condition = raw.copy()
    chemical_condition[:, :prefix] = chemical_map.apply(raw[:, :prefix])
    dose_condition = raw.copy()
    dose_mask = np.zeros((len(raw),), dtype=np.bool_)
    dose_donor_pre = np.asarray(arrays.fit.pre, dtype=np.float32).copy()
    dose_donor_condition = raw.copy()
    dose_donor_target = np.asarray(arrays.fit.target, dtype=np.float32).copy()
    dose_contract = _dose_identifiability_contract(protocol)
    dose_audit: dict[str, Any]
    try:
        dose_map = build_matched_dose_sham_map(
            raw[:, :prefix], raw[:, prefix:], row_ids=arrays.fit.labels, seed=TRAIN_MAP_SEED,
            minimum_absolute_dose_delta=float(dose_contract["minimum_absolute_log10_delta"]),
        )
        source = np.asarray(dose_map.source_indices, dtype=np.int64)
        donor = np.asarray(dose_map.donor_indices, dtype=np.int64)
        dose_audit = dict(dose_map.audit)
    except SchemaError as exc:
        source = np.asarray([], dtype=np.int64)
        donor = np.asarray([], dtype=np.int64)
        dose_audit = {"reason": str(exc), "eligible_row_count": 0, "eligible_chemical_count": 0}
    fit_sufficient = (
        len(source) >= int(dose_contract["minimum_fit_rows"])
        and int(dose_audit.get("eligible_chemical_count", 0)) >= int(dose_contract["minimum_fit_chemicals"])
    )
    selection_raw = np.asarray(arrays.selection.condition, dtype=np.float32)
    try:
        selection_map = build_matched_dose_sham_map(
            selection_raw[:, :prefix], selection_raw[:, prefix:],
            row_ids=arrays.selection.labels, seed=TRAIN_MAP_SEED,
            minimum_absolute_dose_delta=float(dose_contract["minimum_absolute_log10_delta"]),
        )
        development_audit = dict(selection_map.audit)
    except SchemaError as exc:
        development_audit = {
            "reason": str(exc), "eligible_row_count": 0, "eligible_chemical_count": 0
        }
    development_sufficient = (
        int(development_audit.get("eligible_row_count", 0))
        >= int(dose_contract["minimum_development_rows"])
        and int(development_audit.get("eligible_chemical_count", 0))
        >= int(dose_contract["minimum_development_chemicals"])
    )
    sufficient = fit_sufficient and development_sufficient
    dose_audit["status"] = "identifiable" if sufficient else "not_identifiable"
    dose_audit["fit_requirement"] = {
        "minimum_rows": int(dose_contract["minimum_fit_rows"]),
        "minimum_chemicals": int(dose_contract["minimum_fit_chemicals"]),
    }
    dose_audit["development_observation"] = development_audit
    dose_audit["development_requirement"] = {
        "minimum_rows": int(dose_contract["minimum_development_rows"]),
        "minimum_chemicals": int(dose_contract["minimum_development_chemicals"]),
    }
    if sufficient:
        dose_condition[source, prefix:] = raw[donor, prefix:]
        dose_donor_pre[source] = np.asarray(arrays.fit.pre, dtype=np.float32)[donor]
        dose_donor_condition[source] = raw[donor]
        dose_donor_target[source] = np.asarray(arrays.fit.target, dtype=np.float32)[donor]
        dose_mask[source] = True
    chemical_tensor = tensors.standardizer.normalize_condition(torch.as_tensor(chemical_condition, dtype=torch.float32))
    dose_tensor = tensors.standardizer.normalize_condition(torch.as_tensor(dose_condition, dtype=torch.float32))
    donor_pre_tensor = tensors.standardizer.normalize_pre(torch.as_tensor(dose_donor_pre, dtype=torch.float32))
    donor_condition_tensor = tensors.standardizer.normalize_condition(torch.as_tensor(dose_donor_condition, dtype=torch.float32))
    donor_target_tensor = tensors.standardizer.normalize_target(torch.as_tensor(dose_donor_target, dtype=torch.float32))
    return (
        chemical_tensor, dose_tensor, torch.as_tensor(dose_mask, dtype=torch.bool),
        donor_pre_tensor, donor_condition_tensor, donor_target_tensor, dose_audit,
    )


def _raw_predictions(*, model: Any, pre: Any, condition: Any, tensors: Any, device: Any) -> tuple[Any, Any, Any]:
    normalized_pre = tensors.standardizer.normalize_pre(pre).to(device)
    normalized_condition = tensors.standardizer.normalize_condition(condition).to(device)
    chemical, dose = _split_inputs(normalized_condition, prefix_dim=int(tensors.task_spec.chemical_dim))
    with torch.no_grad():
        full, anchor, increment = model.forward_parts(normalized_pre, chemical, dose)
        return tuple(tensors.standardizer.inverse_target(value).cpu() for value in (full, anchor, increment))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise FalsificationGuidedTrainingError("falsification-guided discovery diagnostic distribution is empty")
    position = (len(ordered) - 1) * float(probability)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _pairwise_map_variation(values: Sequence[float]) -> list[float]:
    return [abs(float(left) - float(right)) for index, left in enumerate(values) for right in values[index + 1 :]]


def _effect_status(
    gains: Sequence[float], sham_losses: Sequence[float], *, reference_loss: float,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    qualification = protocol["qualification"]
    interval = qualification["effect_interval"]
    map_variation_threshold = _quantile(
        _pairwise_map_variation(sham_losses),
        float(qualification["map_variation_quantile"]),
    )
    practical_threshold = float(reference_loss) * float(qualification["minimum_relative_target_loss_gain"])
    threshold = max(map_variation_threshold, practical_threshold)
    lower, upper = _quantile(gains, float(interval[0])), _quantile(gains, float(interval[1]))
    status = "claim_supported" if lower > threshold else "behaviorally_unsupported" if upper <= threshold else "inconclusive"
    return {
        "status": status,
        "mean_target_loss_gain": sum(float(value) for value in gains) / len(gains),
        "effect_interval": [lower, upper],
        "map_variation_threshold": map_variation_threshold,
        "practical_threshold": practical_threshold,
        "decision_threshold": threshold,
    }


def development_diagnostics(*, model: Any, arrays: Any, tensors: Any, device: str) -> Mapping[str, Any]:
    """Evaluate target-related chemical/dose increment on Fold 3 only."""

    protocol = _protocol()
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    raw_pre = tensors.selection_pre
    raw_condition = tensors.selection_condition
    raw_target = tensors.selection_target
    full, anchor, increment = _raw_predictions(
        model=model, pre=raw_pre, condition=raw_condition, tensors=tensors, device=device_obj
    )
    true_mse = float((full - raw_target).square().mean())
    anchor_mse = float((anchor - raw_target).square().mean())
    residual_target = raw_target - anchor
    residual_pcc = _global_pcc(torch, increment, residual_target)
    prefix = int(tensors.task_spec.chemical_dim)
    raw_condition_np = raw_condition.numpy()
    raw_pre_np = raw_pre.numpy()
    chemical_gains: list[float] = []
    chemical_sham_losses: list[float] = []
    chemical_pcc_drops: list[float] = []
    dose_gains: list[float] = []
    dose_sham_losses: list[float] = []
    dose_pcc_drops: list[float] = []
    dose_true_losses: list[float] = []
    dose_coverage = None
    dose_contract = _dose_identifiability_contract(protocol)
    dose_identifiable = True
    true_pcc = _global_pcc(torch, full, raw_target)
    for map_seed in DEVELOPMENT_MAP_SEEDS:
        chemical_map = build_chemical_prefix_sham_map(
            raw_pre_np, raw_condition_np[:, :prefix], row_ids=arrays.selection.labels, seed=int(map_seed)
        )
        chemical_condition = raw_condition_np.copy()
        chemical_condition[:, :prefix] = chemical_map.apply(raw_condition_np[:, :prefix])
        chemical_prediction, _anchor, _increment = _raw_predictions(
            model=model,
            pre=raw_pre,
            condition=torch.as_tensor(chemical_condition, dtype=torch.float32),
            tensors=tensors,
            device=device_obj,
        )
        chemical_loss = float((chemical_prediction - raw_target).square().mean())
        chemical_sham_losses.append(chemical_loss)
        chemical_gains.append(chemical_loss - true_mse)
        chemical_pcc_drops.append(true_pcc - _global_pcc(torch, chemical_prediction, raw_target))

        try:
            dose_map = build_matched_dose_sham_map(
                raw_condition_np[:, :prefix], raw_condition_np[:, prefix:],
                row_ids=arrays.selection.labels, seed=int(map_seed),
                minimum_absolute_dose_delta=float(dose_contract["minimum_absolute_log10_delta"]),
            )
        except SchemaError as exc:
            dose_coverage = {
                "status": "not_identifiable",
                "reason": str(exc),
                "eligible_row_count": 0,
                "eligible_chemical_count": 0,
            }
            dose_identifiable = False
            continue
        source = torch.as_tensor(dose_map.source_indices, dtype=torch.long)
        donor = np.asarray(dose_map.donor_indices, dtype=np.int64)
        dose_coverage = dict(dose_map.audit)
        if (
            len(source) < int(dose_contract["minimum_development_rows"])
            or int(dose_coverage.get("eligible_chemical_count", 0))
            < int(dose_contract["minimum_development_chemicals"])
        ):
            dose_identifiable = False
            continue
        if not len(source):
            continue
        dose_condition = raw_condition_np[np.asarray(dose_map.source_indices, dtype=np.int64)].copy()
        dose_condition[:, prefix:] = raw_condition_np[donor, prefix:]
        dose_prediction, _dose_anchor, _dose_increment = _raw_predictions(
            model=model,
            pre=raw_pre[source],
            condition=torch.as_tensor(dose_condition, dtype=torch.float32),
            tensors=tensors,
            device=device_obj,
        )
        true_subset = full[source]
        target_subset = raw_target[source]
        true_subset_loss = float((true_subset - target_subset).square().mean())
        dose_true_losses.append(true_subset_loss)
        dose_loss = float((dose_prediction - target_subset).square().mean())
        dose_sham_losses.append(dose_loss)
        dose_gains.append(dose_loss - true_subset_loss)
        dose_pcc_drops.append(
            _global_pcc(torch, true_subset, target_subset) - _global_pcc(torch, dose_prediction, target_subset)
        )
    chemical = _effect_status(
        chemical_gains, chemical_sham_losses, reference_loss=true_mse, protocol=protocol
    )
    if not dose_identifiable or len(dose_gains) < len(DEVELOPMENT_MAP_SEEDS):
        dose = {
            "status": "not_identifiable",
            "reason": "Fold-3 has too few scientifically separated within-chemical dose interventions",
            "development_requirement": {
                "minimum_rows": int(dose_contract["minimum_development_rows"]),
                "minimum_chemicals": int(dose_contract["minimum_development_chemicals"]),
                "minimum_absolute_log10_delta": float(dose_contract["minimum_absolute_log10_delta"]),
            },
        }
    else:
        dose = _effect_status(
            dose_gains, dose_sham_losses,
            reference_loss=sum(dose_true_losses) / len(dose_true_losses),
            protocol=protocol,
        )
    return {
        "schema_version": "cellscientist_falsification_guided_fold3_increment_diagnostic_v1",
        "fold": 3,
        "fold4_or_fold5_used": False,
        "true_global_pcc": true_pcc,
        "true_mse": true_mse,
        "anchor_mse": anchor_mse,
        "residual_response_pcc": residual_pcc,
        "chemical": {
            **chemical,
            "mean_pcc_drop": sum(chemical_pcc_drops) / len(chemical_pcc_drops),
            "map_count": len(chemical_gains),
        },
        "dose": {
            **dose,
            "mean_pcc_drop": (sum(dose_pcc_drops) / len(dose_pcc_drops)) if dose_pcc_drops else None,
            "map_count": len(dose_gains),
            "identifiability": dose_coverage,
        },
    }


def _performance_gate(*, full: Mapping[str, float], anchor: Mapping[str, float], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    qualification = protocol["qualification"]
    checks = {
        "global_pcc": float(full["global_pcc"]) >= float(anchor["global_pcc"]) - float(qualification["global_pcc_margin_below_anchor"]),
        "cp_pcc": float(full["cp_pcc"]) >= float(anchor["cp_pcc"]) - float(qualification["group_pcc_margin_below_anchor"]),
        "l1000_pcc": float(full["l1000_pcc"]) >= float(anchor["l1000_pcc"]) - float(qualification["group_pcc_margin_below_anchor"]),
        "mse": float(full["mse"]) <= float(anchor["mse"]) + float(qualification["mse_margin_above_anchor"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _qualification_checks(
    *, performance: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> Mapping[str, bool]:
    dose_identifiable = diagnostics["dose"]["status"] != "not_identifiable"
    return {
        "prediction_noninferior": bool(performance["passed"]),
        "chemical_gain_supported": diagnostics["chemical"]["status"] == "claim_supported",
        "dose_claim_applicable": dose_identifiable,
        "dose_gain_supported_if_identifiable": (
            diagnostics["dose"]["status"] == "claim_supported" if dose_identifiable else True
        ),
    }


def _is_qualified(checks: Mapping[str, bool]) -> bool:
    return bool(
        checks["prediction_noninferior"]
        and checks["chemical_gain_supported"]
        and checks["dose_gain_supported_if_identifiable"]
    )


def run_guided_candidate(
    *,
    task_id: str,
    anchor_checkpoint: str | Path,
    output_root: str | Path,
    design: FalsificationGuidedDesign,
    device: str,
    seed: int,
    max_epochs: int | None = None,
) -> Mapping[str, Any]:
    """Train one residual design against one immutable context anchor."""

    protocol = _protocol()
    design.validate()
    root = project_path(output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise FalsificationGuidedTrainingError("falsification-guided discovery candidate output root must be new and below runs/")
    arrays = load_fold13_arrays(task_id)
    config = load_runtime_config("configs/joint_response_runtime.json")
    settings = JointResponseSettings.from_config(config, device=device)
    if max_epochs is not None:
        settings = replace(settings, max_epochs=int(max_epochs), early_stopping_patience=min(settings.early_stopping_patience, int(max_epochs)))
    device_obj = torch.device(device)
    _seed_executor(torch, int(seed), device=device_obj)
    tensors = prepare_joint_tensors(arrays)
    anchor_path = project_path(anchor_checkpoint)
    anchor_payload = _load_anchor(path=anchor_path, task_id=task_id, arrays=arrays, tensors=tensors)
    model = _new_model(tensors=tensors, design=design)
    model.anchor.load_state_dict(anchor_payload["model_state_dict"], strict=True)
    model.freeze_anchor()
    model.reset_increment()
    model.to(device_obj)
    anchor_metrics = _metrics(model, arrays=arrays, tensors=tensors, settings=settings, objective=design.objective, epoch=0)
    model.set_increment_scale(0.0)
    best_performance_state = _state_dict_cpu(model)
    model.set_increment_scale(1.0)
    best_performance_metrics = dict(anchor_metrics)
    best_performance_epoch = 0
    best_performance_scale = 0.0
    best_performance_diagnostics = development_diagnostics(
        model=model, arrays=arrays, tensors=tensors, device=device
    )
    best_qualified_state = None
    best_qualified_metrics = None
    best_qualified_epoch = None
    best_qualified_scale = None
    best_qualified_diagnostics = None
    (
        chemical_sham, dose_sham, dose_mask,
        dose_donor_pre, dose_donor_condition, dose_donor_target,
        dose_training_identifiability,
    ) = _training_shams(arrays=arrays, tensors=tensors, protocol=protocol)
    dataset = TensorDataset(
        tensors.standardizer.normalize_pre(tensors.fit_pre),
        tensors.standardizer.normalize_condition(tensors.fit_condition),
        tensors.standardizer.normalize_target(tensors.fit_target),
        chemical_sham,
        dose_sham,
        dose_mask,
        dose_donor_pre,
        dose_donor_condition,
        dose_donor_target,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(settings.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.increment_readout_parameters()), "lr": 4e-4},
            {"params": list(model.increment_feature_parameters()), "lr": 8e-5},
        ],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(settings.max_epochs)), eta_min=1e-6
    )
    stale = 0
    epochs: list[Mapping[str, Any]] = [{
        "epoch": 0,
        "selection": dict(anchor_metrics),
        "anchor_checkpoint": True,
        "development_diagnostics": best_performance_diagnostics,
        "qualified": False,
    }]
    started = time.perf_counter()
    cp_dim = int(tensors.task_spec.cp_dim)
    scale_grid = tuple(float(value) for value in protocol["model_contract"]["increment_scale_grid"])
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for (
            pre, condition, target, chemical_condition, dose_condition, batch_dose_mask,
            donor_pre, donor_condition, donor_target,
        ) in loader:
            pre = pre.to(device_obj)
            condition = condition.to(device_obj)
            target = target.to(device_obj)
            chemical_condition = chemical_condition.to(device_obj)
            dose_condition = dose_condition.to(device_obj)
            batch_dose_mask = batch_dose_mask.to(device_obj)
            donor_pre = donor_pre.to(device_obj)
            donor_condition = donor_condition.to(device_obj)
            donor_target = donor_target.to(device_obj)
            chemical, dose = _split_inputs(condition, prefix_dim=int(tensors.task_spec.chemical_dim))
            chemical_shuffled, _same_dose = _split_inputs(chemical_condition, prefix_dim=int(tensors.task_spec.chemical_dim))
            _same_chemical, dose_shuffled = _split_inputs(dose_condition, prefix_dim=int(tensors.task_spec.chemical_dim))
            donor_chemical, donor_dose = _split_inputs(donor_condition, prefix_dim=int(tensors.task_spec.chemical_dim))
            optimizer.zero_grad(set_to_none=True)
            prediction, anchor, chemical_increment, dose_increment = model.forward_components(pre, chemical, dose)
            increment = chemical_increment + dose_increment
            chemical_prediction = model(pre, chemical_shuffled, dose)
            dose_prediction = model(pre, chemical, dose_shuffled)
            _donor_prediction, donor_anchor, _donor_chemical_increment, donor_dose_increment = model.forward_components(
                donor_pre, donor_chemical, donor_dose
            )
            residual_target = target - anchor.detach()
            true_row_loss = (prediction - target).square().mean(dim=1)
            chemical_row_loss = (chemical_prediction - target).square().mean(dim=1)
            dose_row_loss = (dose_prediction - target).square().mean(dim=1)
            base_loss = _group_objective(
                prediction, target, cp_dim=cp_dim,
                correlation_weight=0.05 if design.objective == "correlation_residual" else 0.02,
            )
            residual_loss = torch.nn.functional.mse_loss(increment, residual_target)
            chemical_rank = torch.relu(0.001 + true_row_loss - chemical_row_loss).mean()
            dose_rank = (
                torch.relu(0.001 + true_row_loss[batch_dose_mask] - dose_row_loss[batch_dose_mask]).mean()
                if bool(batch_dose_mask.any().item()) else prediction.new_zeros(())
            )
            dose_pair_loss = (
                torch.nn.functional.mse_loss(
                    (dose_increment - donor_dose_increment)[batch_dose_mask],
                    (
                        (target - anchor.detach())
                        - (donor_target - donor_anchor.detach())
                    )[batch_dose_mask],
                )
                if bool(batch_dose_mask.any().item()) else prediction.new_zeros(())
            )
            increment_penalty = increment.square().mean()
            loss = (
                base_loss + 0.50 * residual_loss
                + 0.10 * chemical_rank + 0.10 * dose_rank
                + 0.25 * dose_pair_loss + 0.01 * increment_penalty
            )
            if not bool(torch.isfinite(loss).item()):
                raise FalsificationGuidedTrainingError("falsification-guided discovery residual training produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.increment_parameters(), settings.gradient_clip_norm)
            optimizer.step()
        scheduler.step()
        model.eval()
        calibration = []
        epoch_best = None
        epoch_any_qualified = False
        improved = False
        for increment_scale in scale_grid:
            model.set_increment_scale(increment_scale)
            metrics = _metrics(
                model, arrays=arrays, tensors=tensors, settings=settings,
                objective=design.objective, epoch=epoch,
            )
            epoch_diagnostics = development_diagnostics(
                model=model, arrays=arrays, tensors=tensors, device=device
            )
            epoch_performance = _performance_gate(
                full=metrics, anchor=anchor_metrics, protocol=protocol
            )
            epoch_qualification = _qualification_checks(
                performance=epoch_performance, diagnostics=epoch_diagnostics
            )
            epoch_qualified = _is_qualified(epoch_qualification)
            epoch_any_qualified = epoch_any_qualified or epoch_qualified
            entry = {
                "increment_scale": increment_scale,
                "selection": dict(metrics),
                "development_diagnostics": epoch_diagnostics,
                "performance_gate": epoch_performance,
                "qualification_checks": epoch_qualification,
                "qualified": epoch_qualified,
            }
            calibration.append(entry)
            key = (float(metrics["global_pcc"]), -float(metrics["mse"]))
            if epoch_best is None or key > epoch_best[0]:
                epoch_best = (key, entry)
            best_key = (
                float(best_performance_metrics["global_pcc"]),
                -float(best_performance_metrics["mse"]),
            )
            if key > best_key:
                best_performance_state = _state_dict_cpu(model)
                best_performance_metrics = dict(metrics)
                best_performance_epoch = epoch
                best_performance_scale = increment_scale
                best_performance_diagnostics = epoch_diagnostics
                improved = True
            if epoch_qualified:
                qualified_key = (
                    float(best_qualified_metrics["global_pcc"]),
                    -float(best_qualified_metrics["mse"]),
                ) if best_qualified_metrics is not None else (-float("inf"), -float("inf"))
                if key > qualified_key:
                    best_qualified_state = _state_dict_cpu(model)
                    best_qualified_metrics = dict(metrics)
                    best_qualified_epoch = epoch
                    best_qualified_scale = increment_scale
                    best_qualified_diagnostics = epoch_diagnostics
                    improved = True
        model.set_increment_scale(1.0)
        if epoch_best is None:
            raise FalsificationGuidedTrainingError("falsification-guided discovery increment-scale calibration produced no state")
        epoch_best_entry = epoch_best[1]
        epochs.append({
            "epoch": epoch,
            "learning_rates": [float(value) for value in scheduler.get_last_lr()],
            "selection": epoch_best_entry["selection"],
            "selected_increment_scale_within_epoch": epoch_best_entry["increment_scale"],
            "development_diagnostics": epoch_best_entry["development_diagnostics"],
            "performance_gate": epoch_best_entry["performance_gate"],
            "qualification_checks": epoch_best_entry["qualification_checks"],
            "qualified": epoch_any_qualified,
            "increment_scale_calibration": calibration,
        })
        if improved:
            stale = 0
        else:
            stale += 1
        if stale >= settings.early_stopping_patience and best_qualified_state is not None:
            break
    qualified = best_qualified_state is not None
    if qualified:
        selected_state = best_qualified_state
        selected_metrics = best_qualified_metrics
        selected_epoch = best_qualified_epoch
        selected_scale = best_qualified_scale
        diagnostics = best_qualified_diagnostics
    else:
        selected_state = best_performance_state
        selected_metrics = best_performance_metrics
        selected_epoch = best_performance_epoch
        selected_scale = best_performance_scale
        diagnostics = best_performance_diagnostics
    if (
        selected_state is None or selected_metrics is None or selected_epoch is None
        or selected_scale is None or diagnostics is None
    ):
        raise FalsificationGuidedTrainingError("falsification-guided discovery did not retain a selectable checkpoint")
    model.load_state_dict(selected_state, strict=True)
    performance = _performance_gate(full=selected_metrics, anchor=anchor_metrics, protocol=protocol)
    qualification = _qualification_checks(performance=performance, diagnostics=diagnostics)
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "selected_checkpoint.pt"
    torch.save({
        "schema_version": "cellscientist_increment_candidate_checkpoint_v1",
        "task_id": task_id,
        "data_fingerprint": arrays.data_fingerprint,
        "design": design.to_dict(),
        "anchor_checkpoint": str(anchor_path),
        "anchor_checkpoint_sha256": _sha256(anchor_path),
        "selected_epoch": selected_epoch,
        "selected_increment_scale": selected_scale,
        "model_state_dict": selected_state,
        "fold4_or_fold5_used": False,
    }, checkpoint)
    result = {
        "schema_version": "cellscientist_increment_candidate_run_v1",
        "status": "complete",
        "provider_calls": 0,
        "task_id": task_id,
        "seed": int(seed),
        "design": design.to_dict(),
        "anchor_checkpoint": {"path": str(anchor_path), "sha256": _sha256(anchor_path)},
        "anchor_metrics": anchor_metrics,
        "selected_metrics": selected_metrics,
        "selected_epoch": selected_epoch,
        "selected_increment_scale": selected_scale,
        "development_diagnostics": diagnostics,
        "performance_gate": performance,
        "qualification_checks": qualification,
        "dose_training_identifiability": dose_training_identifiability,
        "qualified": qualified,
        "selection_status": "qualified_pareto_endpoint" if qualified else "no_qualified_model",
        "completed_epochs": len(epochs) - 1,
        "runtime_seconds": time.perf_counter() - started,
        "fold4_or_fold5_used": False,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "epochs": epochs,
    }
    (root / "summary.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return result
