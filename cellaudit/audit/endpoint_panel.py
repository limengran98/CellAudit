"""All-endpoint deterministic audits for the formal source-constrained and falsification-guided discovery study.

The discovery process is allowed to be open-ended.  This module starts only
after every Fold-3-selected endpoint has been frozen.  It never constructs a
prompt, calls a provider, rewrites a model, or uses one endpoint's Fold-4 result
to alter another.  For the score-driven five-refit panel, Fold 5 refits the same
frozen source with the same seeds on Folds 1--2 and selects checkpoints on Fold
3; the remaining studies replay frozen checkpoints.  Fold 5 is additionally
protected by a post-Fold-4 integrity receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from cellaudit.discovery_executor import _global_pcc, _split_inputs
from cellaudit.deterministic_claim_checker import (
    CHEMICAL_CONTROL_INTERACTION,
    DOSE_RELIANCE,
    STATIC_STATUSES,
    check_candidate_claims,
    check_source_claims,
)
from cellaudit.open_discovery import CandidateState
from cellaudit.perturbation_loop_data import (
    build_chemical_prefix_sham_map,
    build_control_profile_sham_map,
    build_matched_dose_sham_map,
)
from cellaudit.schemas import canonical_json, digest
from cellaudit.joint_response_runtime import (
    JointResponseSettings,
    build_candidate_runner,
    prepare_joint_tensors,
    load_runtime_config,
    load_candidate_state_path,
    load_fold13_arrays,
    load_fold14_arrays,
    project_path,
    train_joint_candidate,
)
from cellaudit.audit.data_boundaries import load_fold5_arrays
from cellaudit.discovery.falsification_guided.model import FalsificationGuidedDesign
from cellaudit.discovery.falsification_guided.trainer import _new_model


CHEMICAL_CONTROL_MAP_SEEDS_F4 = tuple(range(28101, 28133))
CHEMICAL_CONTROL_MAP_SEEDS_F5 = tuple(range(29101, 29133))
DOSE_MAP_SEEDS_F4 = tuple(range(39101, 39133))
DOSE_MAP_SEEDS_F5 = tuple(range(40101, 40133))
MIN_ELIGIBLE_ROWS = 64
MIN_ELIGIBLE_FRACTION = 0.10
GUIDED_MINIMUM_ABSOLUTE_DOSE_DELTA = 0.1
GUIDED_MINIMUM_ELIGIBLE_CHEMICALS = 8
REGISTRY_FILE = "ALL_ENDPOINT_REGISTRY.json"
FOLD4_FREEZE_FILE = "FOLD4_AUDIT_FREEZE.json"
FOLD4_SUMMARY_FILE = "FOLD4_AUDIT_SUMMARY.json"
FOLD5_FREEZE_FILE = "FOLD5_REPLICATION_FREEZE.json"
FOLD5_SUMMARY_FILE = "FOLD5_REPLICATION_SUMMARY.json"
METHOD_ENDPOINT_FREEZE_FILE = "METHOD_ENDPOINT_FREEZE.json"
OPEN_REFIT_COUNT = 5
CONSTRAINED_ENDPOINT_COUNT = 10
SUPPORTED_METHODS = frozenset({"open", "source_constrained", "falsification_guided"})
OPEN_METHOD_SELECTION_RULE = (
    "maximum Fold-3 Global PCC across every executable candidate in all ten "
    "trajectories; then minimum Fold-3 MSE; then minimum parameter count; "
    "then stable trajectory/candidate/hash identifier"
)


class FullEndpointAuditError(RuntimeError):
    """A discovery receipt, checkpoint, or data boundary is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise FullEndpointAuditError("PyTorch is required for the endpoint audit") from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullEndpointAuditError(f"cannot read JSON receipt: {path}") from exc
    if not isinstance(value, Mapping):
        raise FullEndpointAuditError(f"JSON receipt is not an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(project_path(".")))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= probability <= 1:
        raise FullEndpointAuditError("invalid quantile input")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _t_critical_95(count: int) -> float:
    # Two-sided Student-t 97.5% quantiles for n=2..10.  Larger panels use a
    # normal approximation only if the exact table is unavailable.
    table = {2: 12.706205, 3: 4.302653, 4: 3.182446, 5: 2.776445,
             6: 2.570582, 7: 2.446912, 8: 2.364624, 9: 2.306004, 10: 2.262157}
    return table.get(count, 1.959964)


def _mean_ci(values: Sequence[float]) -> Mapping[str, float]:
    data = [float(value) for value in values]
    if not data:
        raise FullEndpointAuditError("cannot summarize an empty panel")
    center = mean(data)
    spread = stdev(data) if len(data) > 1 else 0.0
    half = _t_critical_95(len(data)) * spread / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {"mean": center, "std": spread, "ci95_low": center - half, "ci95_high": center + half, "n": len(data)}


def coordinate_utility_decomposition(
    *,
    correct_chemical_correct_dose: float,
    shuffled_chemical_correct_dose: float,
    correct_chemical_shuffled_dose: float,
    shuffled_chemical_shuffled_dose: float,
    context_anchor: float,
) -> Mapping[str, float]:
    """Resolve frozen predictive utility across chemical and dose coordinates.

    The two Shapley terms and the both-shuffled residual close exactly to the
    full predictor's improvement over the frozen context anchor.  The
    factorial interaction is returned as a separate diagnostic because it is
    already shared across the two Shapley terms.
    """

    values = {
        "u11": float(correct_chemical_correct_dose),
        "u01": float(shuffled_chemical_correct_dose),
        "u10": float(correct_chemical_shuffled_dose),
        "u00": float(shuffled_chemical_shuffled_dose),
        "anchor": float(context_anchor),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FullEndpointAuditError("coordinate decomposition requires finite utilities")
    chemical = 0.5 * ((values["u11"] - values["u01"]) + (values["u10"] - values["u00"]))
    dose = 0.5 * ((values["u11"] - values["u10"]) + (values["u01"] - values["u00"]))
    residual = values["u00"] - values["anchor"]
    full_over_anchor = values["u11"] - values["anchor"]
    interaction = values["u11"] - values["u10"] - values["u01"] + values["u00"]
    closure_error = chemical + dose + residual - full_over_anchor
    if abs(closure_error) > 1e-12:
        raise FullEndpointAuditError("coordinate decomposition failed algebraic closure")
    return {
        "full_over_anchor": full_over_anchor,
        "chemical_shapley": chemical,
        "dose_shapley": dose,
        "both_shuffled_over_anchor": residual,
        "chemical_dose_interaction": interaction,
        "closure_error": closure_error,
    }


def _pairwise_absolute_differences(values: Sequence[float]) -> list[float]:
    data = [float(value) for value in values]
    if len(data) < 2:
        raise FullEndpointAuditError("map-variation reference requires at least two maps")
    return [abs(left - right) for index, left in enumerate(data) for right in data[index + 1 :]]


def _decision(values: Sequence[float], reference: Sequence[float], *, absolute: bool = False) -> Mapping[str, Any]:
    effect = [abs(float(value)) if absolute else float(value) for value in values]
    threshold = _quantile(reference, 0.95)
    lower, upper = _quantile(effect, 0.025), _quantile(effect, 0.975)
    status = "claim_supported" if lower > threshold else "behaviorally_unsupported" if upper <= threshold else "inconclusive"
    return {
        "status": status,
        "mean_effect": mean(effect),
        "replicate_interval": [lower, upper],
        "map_variation_threshold": threshold,
    }


def _dataset_from_task(task_id: str) -> str:
    if task_id.startswith("cpg036_"):
        return "bbbc036"
    if task_id.startswith("cpg047_"):
        return "bbbc047"
    raise FullEndpointAuditError(f"unsupported CPG task for full audit: {task_id}")


def _fold3_global_pcc(selected: Mapping[str, Any]) -> float:
    feedback = selected.get("feedback")
    values = feedback if isinstance(feedback, Mapping) else {}
    raw = next(
        (
            values[key]
            for key in ("global_pcc", "fold3_global_pcc", "feedback_global_pcc")
            if key in values
        ),
        selected.get("fold3_global_pcc"),
    )
    if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise FullEndpointAuditError("selected endpoint lacks a finite Fold-3 Global PCC")
    return float(raw)


def _method_claims(source_symbol: str) -> list[dict[str, Any]]:
    """Claims explicitly fixed by the constrained discovery formulations."""

    return [
        {
            "claim_type": CHEMICAL_CONTROL_INTERACTION,
            "statement": "The registered fusion jointly conditions response prediction on context and chemical identity.",
            "source_symbols": [source_symbol],
        },
        {
            "claim_type": DOSE_RELIANCE,
            "statement": "The registered response predictor consumes the dose attribute.",
            "source_symbols": [source_symbol],
        },
    ]


def _registered_fusion_symbol(candidate: CandidateState) -> str:
    symbols = candidate.candidate_metadata.get("component_symbols")
    if not isinstance(symbols, Mapping):
        raise FullEndpointAuditError("candidate has no registered component-symbol map")
    for address, values in symbols.items():
        if not isinstance(address, str) or not (
            "fusion" in address or address.startswith("perturbation.")
        ):
            continue
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
            return str(values[0])
    raise FullEndpointAuditError("source-constrained candidate has no registered fusion symbol")


def _checker_contract(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    claims = receipt.get("claims")
    if not isinstance(claims, list) or not claims:
        raise FullEndpointAuditError("deterministic checker produced no claim dispositions")
    contract: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, Mapping) or str(claim.get("static_status")) not in STATIC_STATUSES:
            raise FullEndpointAuditError("deterministic checker produced an invalid source disposition")
        contract.append({
            "claim_type": str(claim["claim_type"]),
            "origin": str(claim["origin"]),
            "static_status": str(claim["static_status"]),
            "reason": str(claim["reason"]),
            "required_tests": list(claim.get("required_tests") or []),
        })
    return {
        "checker_schema": str(receipt["schema_version"]),
        "checker_receipt_hash": str(receipt["receipt_hash"]),
        "claims": contract,
    }


def _guided_checker_receipt(
    *, candidate_id: str, design: Mapping[str, Any], fold3_global_pcc: float
) -> tuple[Mapping[str, Any], str]:
    model_path = project_path("cellaudit/discovery/falsification_guided/model.py")
    source = model_path.read_text(encoding="utf-8")
    source_symbol = "PerturbationIncrementModel"
    metadata = {
        "component_symbols": {"perturbation.falsification_guided_increment": [source_symbol]},
    }
    candidate_hash = digest({"model_source_sha256": _sha256(model_path), "design": dict(design)})
    receipt = check_source_claims(
        candidate_source=source,
        candidate_metadata=metadata,
        candidate_hash=candidate_hash,
        candidate_id=candidate_id,
        fold3_global_pcc=fold3_global_pcc,
        registered_method_claims=_method_claims(source_symbol),
    )
    return receipt, candidate_hash


def _write_checker_receipt(
    *, root: Path, audit_unit_id: str, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    path = root / "registry" / f"{audit_unit_id}.checker.json"
    _write(path, receipt)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "receipt_hash": str(receipt["receipt_hash"]),
    }


def _validate_discovery_summary(summary: Mapping[str, Any], *, task_id: str) -> None:
    if summary.get("status") != "complete" or summary.get("mode") != "formal":
        raise FullEndpointAuditError("discovery trace is not a completed formal trajectory")
    if summary.get("task_id") != task_id or summary.get("fold4_or_fold5_used") is not False:
        raise FullEndpointAuditError("discovery trace violates task or sealed-fold boundary")
    selected = summary.get("selected_candidate_id")
    if not isinstance(selected, str):
        selected_record = summary.get("selected_candidate")
        selected = selected_record.get("candidate_id") if isinstance(selected_record, Mapping) else None
    records = summary.get("records")
    if not isinstance(selected, str) or not isinstance(records, list):
        raise FullEndpointAuditError("discovery summary has no selected candidate record")
    matches = [row for row in records if isinstance(row, Mapping) and row.get("candidate_id") == selected]
    if len(matches) != 1 or matches[0].get("status") != "complete":
        raise FullEndpointAuditError("discovery selected candidate is not complete")


def _campaign_value(campaign: Mapping[str, Any], modern: str, legacy: str | None = None) -> Any:
    value = campaign.get(modern)
    return campaign.get(legacy) if value is None and legacy is not None else value


def _fold3_scalar(record: Mapping[str, Any], names: Sequence[str], *, field: str) -> float:
    feedback = record.get("feedback")
    values = feedback if isinstance(feedback, Mapping) else {}
    raw = next((values[name] for name in names if name in values), record.get(field))
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise FullEndpointAuditError(f"executable open candidate lacks finite {field}")
    return float(raw)


def _fold3_parameter_count(record: Mapping[str, Any]) -> int:
    feedback = record.get("feedback")
    values = feedback if isinstance(feedback, Mapping) else {}
    raw = values.get("parameter_count", record.get("parameter_count"))
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise FullEndpointAuditError("executable open candidate lacks a positive parameter_count")
    return int(raw)


def _select_open_method_endpoint(
    *, discovery: Path, task_id: str, trajectory_count: int
) -> tuple[Mapping[str, Any], CandidateState, Mapping[str, Any]]:
    """Freeze the method-level Fold-3 winner over every executable candidate."""

    if trajectory_count != CONSTRAINED_ENDPOINT_COUNT:
        raise FullEndpointAuditError(
            "automatic open endpoint selection requires the complete ten-trajectory campaign"
        )
    candidates: list[dict[str, Any]] = []
    trajectory_receipts: list[dict[str, Any]] = []
    for trajectory in range(1, trajectory_count + 1):
        result_root = discovery / f"trajectory_{trajectory:02d}" / "result"
        summary_path = result_root / "summary.json"
        summary = _read(summary_path)
        _validate_discovery_summary(summary, task_id=task_id)
        trajectory_receipts.append({
            "trajectory": trajectory,
            "path": _relative(summary_path),
            "sha256": _sha256(summary_path),
        })
        records = summary.get("records")
        if not isinstance(records, list):
            raise FullEndpointAuditError("open discovery summary has no candidate records")
        for record in records:
            if not isinstance(record, Mapping) or record.get("status") != "complete":
                continue
            candidate_id = record.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise FullEndpointAuditError("executable open candidate has no stable candidate id")
            state_path = result_root / "candidates" / candidate_id / "candidate_state.json"
            expected_hash = record.get("candidate_hash")
            candidate = load_candidate_state_path(
                state_path,
                expected_hash=str(expected_hash) if isinstance(expected_hash, str) else None,
            )
            global_pcc = _fold3_scalar(
                record, ("fold3_global_pcc", "global_pcc", "feedback_global_pcc"),
                field="fold3_global_pcc",
            )
            mse = _fold3_scalar(
                record, ("fold3_mse", "mse", "feedback_mse"), field="fold3_mse"
            )
            parameter_count = _fold3_parameter_count(record)
            stable_id = f"trajectory_{trajectory:02d}/{candidate_id}/{candidate.candidate_hash}"
            candidates.append({
                "trajectory": trajectory,
                "candidate_id": candidate_id,
                "candidate_hash": candidate.candidate_hash,
                "candidate_state_path": _relative(state_path),
                "candidate_state_sha256": _sha256(state_path),
                "discovery_summary_path": _relative(summary_path),
                "discovery_summary_sha256": _sha256(summary_path),
                "fold3_global_pcc": global_pcc,
                "fold3_mse": mse,
                "parameter_count": parameter_count,
                "stable_id": stable_id,
            })
    if not candidates:
        raise FullEndpointAuditError("complete open campaign contains no executable candidate")
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["fold3_global_pcc"]),
            float(row["fold3_mse"]),
            int(row["parameter_count"]),
            str(row["stable_id"]),
        ),
    )
    selected = dict(ranked[0])
    candidate = load_candidate_state_path(
        project_path(str(selected["candidate_state_path"])),
        expected_hash=str(selected["candidate_hash"]),
    )
    receipt = {
        "schema_version": "cellscientist_open_method_endpoint_freeze_v1",
        "status": "frozen_before_refitting",
        "method": "open",
        "task_id": task_id,
        "selection_partition": "Fold-3 only",
        "trajectory_count": trajectory_count,
        "executable_candidate_count": len(candidates),
        "selection_rule": OPEN_METHOD_SELECTION_RULE,
        "trajectory_receipts": trajectory_receipts,
        "selected": selected,
        "fold4_or_fold5_used": False,
        "provider_calls": 0,
    }
    return selected, candidate, receipt


def prepare_open_refits(
    *,
    task_id: str,
    discovery_root: str | Path,
    output_root: str | Path,
    seed_base: int,
    device: str,
    source_trajectory: int | None = None,
) -> Mapping[str, Any]:
    """Freeze the method-level Fold-3 winner and refit it under five paired seeds.

    This preparation step runs before the Fold-4 audit registry is frozen.  It
    ranks every executable candidate in the completed ten-trajectory open
    campaign using the registered Fold-3 rule, freezes the winning source, then
    refits that unchanged source on Folds 1--2 with five registered seeds and
    uses Fold 3 only for checkpoint selection.  ``source_trajectory`` retains a
    compatibility path for an already registered external receipt; automatic
    method-level selection is the default.  This function never opens a held-out
    endpoint and never calls the discovery provider.
    """

    discovery = project_path(discovery_root)
    root = project_path(output_root)
    if not discovery.is_relative_to(project_path("runs")) or not (discovery / "_SUCCESS").is_file():
        raise FullEndpointAuditError("open refit preparation requires a completed campaign below runs/")
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise FullEndpointAuditError("open refit output must be a new directory below runs/")
    campaign = _read(discovery / "campaign_freeze.json")
    if _campaign_value(campaign, "method") != "open" or _campaign_value(campaign, "task_id", "task") != task_id:
        raise FullEndpointAuditError("source campaign is not the registered open-discovery task")
    source_count = int(campaign.get("trajectory_count", len(campaign.get("trajectory_seeds", []))))
    if source_trajectory is None:
        selection, candidate, endpoint_freeze = _select_open_method_endpoint(
            discovery=discovery, task_id=task_id, trajectory_count=source_count
        )
        source_trajectory_value = int(selection["trajectory"])
        candidate_id = str(selection["candidate_id"])
        source_summary_path = project_path(str(selection["discovery_summary_path"]))
        source_summary = _read(source_summary_path)
        selected = next(
            row for row in source_summary["records"]
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        )
    else:
        if not 1 <= int(source_trajectory) <= source_count:
            raise FullEndpointAuditError("source trajectory is outside the completed campaign")
        source_trajectory_value = int(source_trajectory)
        source_result = discovery / f"trajectory_{source_trajectory_value:02d}" / "result"
        source_summary_path = source_result / "summary.json"
        source_summary = _read(source_summary_path)
        _validate_discovery_summary(source_summary, task_id=task_id)
        selected_id = source_summary.get("selected_candidate_id")
        if not isinstance(selected_id, str):
            selected_record = source_summary.get("selected_candidate")
            selected_id = selected_record.get("candidate_id") if isinstance(selected_record, Mapping) else None
        if not isinstance(selected_id, str):
            raise FullEndpointAuditError("compatibility source trajectory lacks a selected candidate")
        candidate_id = selected_id
        source_candidate_root = source_result / "candidates" / candidate_id
        candidate = load_candidate_state_path(source_candidate_root / "candidate_state.json")
        selected = next(
            row for row in source_summary["records"]
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        )
        endpoint_freeze = {
            "schema_version": "cellscientist_open_method_endpoint_freeze_v1",
            "status": "frozen_before_refitting",
            "method": "open",
            "task_id": task_id,
            "selection_partition": "Fold-3 only",
            "selection_mode": "explicit_source_trajectory_compatibility",
            "selection_rule": "use the source trajectory bound by an external pre-registration receipt",
            "selected": {
                "trajectory": source_trajectory_value,
                "candidate_id": candidate_id,
                "candidate_hash": candidate.candidate_hash,
                "candidate_state_path": _relative(source_candidate_root / "candidate_state.json"),
                "candidate_state_sha256": _sha256(source_candidate_root / "candidate_state.json"),
                "discovery_summary_path": _relative(source_summary_path),
                "discovery_summary_sha256": _sha256(source_summary_path),
                "fold3_global_pcc": _fold3_global_pcc(selected),
            },
            "fold4_or_fold5_used": False,
            "provider_calls": 0,
        }

    config = load_runtime_config("configs/joint_response_runtime.json")
    arrays = load_fold13_arrays(task_id)
    base_settings = JointResponseSettings.from_config(config, device=device)
    seeds = [int(seed_base) + index for index in range(OPEN_REFIT_COUNT)]
    root.mkdir(parents=True, exist_ok=False)
    _write(root / METHOD_ENDPOINT_FREEZE_FILE, endpoint_freeze)
    campaign_freeze = {
        "schema_version": "cellscientist_open_source_refit_campaign_v1",
        "status": "frozen_before_refitting",
        "method": "open",
        "task_id": task_id,
        "trajectory_count": OPEN_REFIT_COUNT,
        "trajectory_seeds": seeds,
        "candidate_budget": 1,
        "candidate_hash": candidate.candidate_hash,
        "method_endpoint_freeze": {
            "path": _relative(root / METHOD_ENDPOINT_FREEZE_FILE),
            "sha256": _sha256(root / METHOD_ENDPOINT_FREEZE_FILE),
        },
        "source_campaign": {"path": _relative(discovery / "campaign_freeze.json"), "sha256": _sha256(discovery / "campaign_freeze.json")},
        "source_summary": {"path": _relative(source_summary_path), "sha256": _sha256(source_summary_path)},
        "source_trajectory": source_trajectory_value,
        "source_candidate_id": candidate_id,
        "selection_rule": endpoint_freeze["selection_rule"],
        "fold_roles": {"fit": [1, 2], "selection": 3, "withheld": [4, 5]},
        "provider_calls": 0,
    }
    _write(root / "campaign_freeze.json", campaign_freeze)

    completed: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        result_root = root / f"trajectory_{index:02d}" / "result"
        candidate_root = result_root / "candidates" / candidate_id
        checkpoint = candidate_root / "selected_checkpoint.pt"
        settings = replace(base_settings, seed=int(seed))
        fit_record = train_joint_candidate(
            "cellscientist_open_refit",
            arrays=arrays,
            config=config,
            settings=settings,
            checkpoint_path=checkpoint,
            candidate_override=candidate,
        )
        _write(candidate_root / "candidate_state.json", candidate.to_dict())
        feedback = dict(fit_record["selection_metrics"])
        record = {
            "candidate_id": candidate_id,
            "status": "complete",
            "feedback": feedback,
            "candidate_hash": candidate.candidate_hash,
            "source_campaign_candidate": {
                "trajectory": source_trajectory_value,
                "candidate_id": candidate_id,
                "fold3_feedback": dict(selected.get("feedback") or {}),
            },
            "fit_record": fit_record,
        }
        summary = {
            "schema_version": "cellscientist_open_source_refit_summary_v1",
            "status": "complete",
            "mode": "formal",
            "task_id": task_id,
            "trajectory_seed": int(seed),
            "fold4_or_fold5_used": False,
            "provider_calls": 0,
            "selected_candidate_id": candidate_id,
            "records": [record],
        }
        _write(result_root / "summary.json", summary)
        (result_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        completed.append({"refit": index, "seed": int(seed), "candidate_id": candidate_id, "status": "complete"})

    campaign_summary = {
        "schema_version": "cellscientist_open_source_refit_campaign_summary_v1",
        "status": "complete",
        "method": "open",
        "task_id": task_id,
        "candidate_hash": candidate.candidate_hash,
        "method_endpoint_freeze": {
            "path": _relative(root / METHOD_ENDPOINT_FREEZE_FILE),
            "sha256": _sha256(root / METHOD_ENDPOINT_FREEZE_FILE),
        },
        "source_trajectory": source_trajectory_value,
        "source_candidate_id": candidate_id,
        "refit_count": OPEN_REFIT_COUNT,
        "refits": completed,
        "provider_calls": 0,
    }
    _write(root / "campaign_summary.json", campaign_summary)
    (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return campaign_summary


def freeze_panel(
    *,
    method: str,
    task_id: str,
    discovery_root: str | Path,
    output_root: str | Path,
    trajectory_count: int = 10,
) -> Mapping[str, Any]:
    """Bind every selected discovery endpoint before exposing Fold 4."""
    if method not in SUPPORTED_METHODS:
        raise FullEndpointAuditError("method must be open, source_constrained, or falsification_guided")
    if trajectory_count < 2:
        raise FullEndpointAuditError("full endpoint audit requires at least two discovery trajectories")
    expected_count = OPEN_REFIT_COUNT if method == "open" else CONSTRAINED_ENDPOINT_COUNT
    if trajectory_count != expected_count:
        unit = "paired refits" if method == "open" else "trajectory endpoints"
        raise FullEndpointAuditError(f"{method} audit requires exactly {expected_count} frozen {unit}")
    root = project_path(output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise FullEndpointAuditError("audit root must be a new directory under runs/")
    discovery = project_path(discovery_root)
    if not discovery.is_relative_to(project_path("runs")):
        raise FullEndpointAuditError("discovery root must be a completed directory below runs/")
    campaign = _read(discovery / "campaign_freeze.json")
    seeds = list(campaign.get("trajectory_seeds", []))
    if len(seeds) != trajectory_count:
        raise FullEndpointAuditError("discovery campaign does not match the registered trajectory count")
    campaign_method = _campaign_value(campaign, "method")
    campaign_task = _campaign_value(campaign, "task_id", "task")
    if campaign_method not in (None, method) or campaign_task != task_id or not (discovery / "_SUCCESS").is_file():
        raise FullEndpointAuditError("discovery campaign is incomplete or does not match the registered audit cell")
    method_endpoint_receipt: Mapping[str, Any] | None = None
    if method == "open":
        method_endpoint_receipt = campaign.get("method_endpoint_freeze")
        if not isinstance(method_endpoint_receipt, Mapping):
            raise FullEndpointAuditError("open refit campaign lacks the selected-model freeze receipt")
        method_endpoint_path = project_path(str(method_endpoint_receipt.get("path")))
        if (
            not method_endpoint_path.is_file()
            or _sha256(method_endpoint_path) != str(method_endpoint_receipt.get("sha256"))
        ):
            raise FullEndpointAuditError("open selected-model freeze changed before Fold 4")
        method_endpoint = _read(method_endpoint_path)
        selected_endpoint = method_endpoint.get("selected")
        if (
            method_endpoint.get("status") != "frozen_before_refitting"
            or not isinstance(selected_endpoint, Mapping)
            or str(selected_endpoint.get("candidate_hash")) != str(campaign.get("candidate_hash"))
        ):
            raise FullEndpointAuditError("open refit campaign does not bind its selected method source")
    root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    global_trajectory = 0
    for source_trajectory in range(1, len(seeds) + 1):
        global_trajectory += 1
        result_root = discovery / f"trajectory_{source_trajectory:02d}" / "result"
        summary_path = result_root / "summary.json"
        summary = _read(summary_path)
        _validate_discovery_summary(summary, task_id=task_id)
        candidate_id = str(summary["selected_candidate_id"])
        selected = next(row for row in summary["records"] if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id)
        candidate_root = result_root / "candidates" / candidate_id
        checkpoint = candidate_root / "selected_checkpoint.pt"
        if not checkpoint.is_file():
            # falsification-guided discovery records the execution location explicitly; retain this narrow
            # fallback because it is part of the registered discovery receipt.
            execution_root = selected.get("execution_root")
            checkpoint = Path(str(execution_root)) / "selected_checkpoint.pt" if execution_root else checkpoint
        if not checkpoint.is_file():
            raise FullEndpointAuditError(f"selected checkpoint is missing for trajectory {global_trajectory}")
        audit_unit_id = f"{method}_{'refit' if method == 'open' else 't'}{global_trajectory:02d}_{candidate_id}"
        fold3_global_pcc = _fold3_global_pcc(selected)
        common = {
            "audit_unit_id": audit_unit_id,
            "trajectory": global_trajectory,
            "unit_role": "paired_frozen_source_refit" if method == "open" else "independent_trajectory_endpoint",
            "source_trajectory": source_trajectory,
            "source_campaign": _relative(discovery),
            "trajectory_seed": int(summary["trajectory_seed"]),
            "selected_candidate_id": candidate_id,
            "fold3_metrics": dict(selected.get("feedback") or {}),
            "discovery_summary": {"path": _relative(summary_path), "sha256": _sha256(summary_path)},
            "checkpoint": {"path": _relative(checkpoint), "sha256": _sha256(checkpoint)},
        }
        if method in {"open", "source_constrained"}:
            state_path = candidate_root / "candidate_state.json"
            candidate = load_candidate_state_path(state_path)
            copy_path = root / "registry" / f"trajectory_{global_trajectory:02d}_{candidate_id}.candidate_state.json"
            _write(copy_path, candidate.to_dict())
            registered = _method_claims(_registered_fusion_symbol(candidate)) if method == "source_constrained" else None
            checker = check_candidate_claims(
                candidate=candidate,
                candidate_id=candidate_id,
                fold3_global_pcc=fold3_global_pcc,
                registered_method_claims=registered,
            )
            common.update({
                "candidate_hash": candidate.candidate_hash,
                "candidate_state": {"path": _relative(copy_path), "sha256": _sha256(copy_path)},
                "checker_receipt": _write_checker_receipt(root=root, audit_unit_id=audit_unit_id, receipt=checker),
                "source_contract": _checker_contract(checker),
            })
        else:
            execution = _read(Path(str(selected["execution_root"])) / "summary.json")
            design = dict(execution.get("design") or selected.get("design") or {})
            FalsificationGuidedDesign(**design).validate()
            checker, candidate_hash = _guided_checker_receipt(
                candidate_id=candidate_id,
                design=design,
                fold3_global_pcc=fold3_global_pcc,
            )
            common.update({
                "candidate_hash": candidate_hash,
                "design": design,
                "checker_receipt": _write_checker_receipt(root=root, audit_unit_id=audit_unit_id, receipt=checker),
                "source_contract": _checker_contract(checker),
            })
        records.append(common)
    if global_trajectory != trajectory_count:
        raise FullEndpointAuditError("combined discovery registry is incomplete")
    if method == "open" and len({str(record["candidate_hash"]) for record in records}) != 1:
        raise FullEndpointAuditError(
            "open audit requires five refits of one frozen source, not five independently selected sources"
        )
    registry = {
        "schema_version": "cellscientist_endpoint_audit_all_endpoint_registry_v1",
        "status": "frozen_before_fold4",
        "method": method,
        "dataset": _dataset_from_task(task_id),
        "task_id": task_id,
        "provider_calls": 0,
        "panel_design": "five_paired_refits_of_one_frozen_source" if method == "open" else "ten_independent_trajectory_endpoints",
        "records": records,
    }
    _write(root / REGISTRY_FILE, registry)
    freeze = {
        "schema_version": "cellscientist_endpoint_audit_all_endpoint_fold4_freeze_v1",
        "status": "frozen_before_fold4",
        "method": method,
        "task_id": task_id,
        "provider_calls": 0,
        "registry": {"path": _relative(root / REGISTRY_FILE), "sha256": _sha256(root / REGISTRY_FILE)},
        "discovery_campaign": {"path": _relative(discovery / "campaign_freeze.json"), "sha256": _sha256(discovery / "campaign_freeze.json")},
        "audit_implementation": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "checker_implementation": {
            "path": _relative(project_path("cellaudit/deterministic_claim_checker.py")),
            "sha256": _sha256(project_path("cellaudit/deterministic_claim_checker.py")),
        },
        "runtime_config": {
            "path": _relative(project_path("configs/joint_response_runtime.json")),
            "sha256": _sha256(project_path("configs/joint_response_runtime.json")),
        },
        "data_protocol": {
            "fit_folds": [1, 2],
            "selection_fold": 3,
            "audit_fold": 4,
            "fold5_opened": False,
            "fold4_model_state": "five registered Fold-3-selected refits" if method == "open" else "frozen discovery checkpoints",
            "fold5_model_state": "refit frozen source with the same five-seed schedule; select checkpoints on Fold 3" if method == "open" else "replay frozen discovery checkpoints",
        },
        "chemical_control_counterfactual": {"map_seeds": list(CHEMICAL_CONTROL_MAP_SEEDS_F4), "design": "2x2 correct/shuffled chemical x correct/matched-wrong control"},
        "dose_counterfactual": {
            "map_seeds": list(DOSE_MAP_SEEDS_F4),
            "design": "within-chemical observed-dose derangement",
            "min_eligible_rows": MIN_ELIGIBLE_ROWS,
            "min_eligible_fraction": MIN_ELIGIBLE_FRACTION,
            "guided_minimum_absolute_dose_delta": GUIDED_MINIMUM_ABSOLUTE_DOSE_DELTA,
            "guided_minimum_eligible_chemicals": GUIDED_MINIMUM_ELIGIBLE_CHEMICALS,
        },
        "decision_rule": {
            "map_variation_reference": "model-local pairwise variation among fixed shuffled-input maps",
            "threshold": "95th map-variation quantile",
            "per_model_status": "input-effect 2.5/97.5 percentiles versus threshold",
            "aggregate": (
                "four-of-five refit agreement; otherwise inconclusive"
                if method == "open"
                else "continuous model distribution and status counts; no audit-driven reselection"
            ),
        },
        "prohibited": ["provider call", "candidate rewrite", "candidate reselection", "unregistered checkpoint training", "Fold-5 access", "manual result exclusion"],
        "expected_units": len(records),
    }
    if method_endpoint_receipt is not None:
        freeze["method_endpoint_freeze"] = dict(method_endpoint_receipt)
    _write(root / FOLD4_FREEZE_FILE, freeze)
    (root / "_FOLD4_READY").write_text("frozen; deterministic Fold-4 replay is authorized\n", encoding="utf-8")
    return freeze


def _load_frozen(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    freeze = _read(root / FOLD4_FREEZE_FILE)
    if freeze.get("schema_version") != "cellscientist_endpoint_audit_all_endpoint_fold4_freeze_v1" or freeze.get("status") != "frozen_before_fold4":
        raise FullEndpointAuditError("invalid Fold-4 freeze")
    implementation = freeze.get("audit_implementation")
    if not isinstance(implementation, Mapping) or _sha256(Path(__file__)) != str(implementation.get("sha256")):
        raise FullEndpointAuditError("audit implementation changed after the Fold-4 freeze")
    checker_implementation = freeze.get("checker_implementation")
    checker_path = project_path("cellaudit/deterministic_claim_checker.py")
    if (
        not isinstance(checker_implementation, Mapping)
        or str(checker_implementation.get("path")) != _relative(checker_path)
        or _sha256(checker_path) != str(checker_implementation.get("sha256"))
    ):
        raise FullEndpointAuditError("deterministic checker changed after the Fold-4 freeze")
    runtime_config = freeze.get("runtime_config")
    runtime_config_path = project_path("configs/joint_response_runtime.json")
    if (
        not isinstance(runtime_config, Mapping)
        or str(runtime_config.get("path")) != _relative(runtime_config_path)
        or _sha256(runtime_config_path) != str(runtime_config.get("sha256"))
    ):
        raise FullEndpointAuditError("training configuration changed after the Fold-4 freeze")
    if freeze.get("method") == "open":
        method_endpoint_receipt = freeze.get("method_endpoint_freeze")
        if not isinstance(method_endpoint_receipt, Mapping):
            raise FullEndpointAuditError("open Fold-4 freeze lacks the selected-model receipt")
        method_endpoint_path = project_path(str(method_endpoint_receipt.get("path")))
        if (
            not method_endpoint_path.is_file()
            or _sha256(method_endpoint_path) != str(method_endpoint_receipt.get("sha256"))
        ):
            raise FullEndpointAuditError("open selected-model receipt changed after Fold 4")
    receipt = freeze.get("registry")
    if not isinstance(receipt, Mapping):
        raise FullEndpointAuditError("missing audit registry receipt")
    registry_path = project_path(str(receipt["path"]))
    if _sha256(registry_path) != str(receipt["sha256"]):
        raise FullEndpointAuditError("audit registry changed after freeze")
    registry = _read(registry_path)
    records = registry.get("records")
    if not isinstance(records, list) or not records:
        raise FullEndpointAuditError("frozen audit registry has no endpoint records")
    for record in records:
        checker = record.get("checker_receipt") if isinstance(record, Mapping) else None
        if not isinstance(checker, Mapping):
            raise FullEndpointAuditError("frozen endpoint lacks a deterministic checker receipt")
        checker_path = project_path(str(checker.get("path")))
        if _sha256(checker_path) != str(checker.get("sha256")):
            raise FullEndpointAuditError("deterministic checker receipt changed after freeze")
        payload = _read(checker_path)
        if (
            str(payload.get("receipt_hash")) != str(checker.get("receipt_hash"))
            or str(payload.get("candidate_hash")) != str(record.get("candidate_hash"))
        ):
            raise FullEndpointAuditError("deterministic checker receipt does not bind the frozen endpoint")
        _checker_contract(payload)
    return freeze, registry


def _load_model(*, record: Mapping[str, Any], method: str, arrays: Any, device: str) -> tuple[Any, Any, Mapping[str, Any]]:
    torch = _torch()
    checkpoint = project_path(str(record["checkpoint"]["path"]))
    if _sha256(checkpoint) != str(record["checkpoint"]["sha256"]):
        raise FullEndpointAuditError("frozen checkpoint changed after registration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = prepare_joint_tensors(arrays)
    if payload.get("task_id") != arrays.task_id or payload.get("data_fingerprint") != arrays.data_fingerprint:
        raise FullEndpointAuditError("checkpoint does not bind registered task/data")
    if method in {"open", "source_constrained"}:
        state = record.get("candidate_state")
        if not isinstance(state, Mapping):
            raise FullEndpointAuditError("source-constrained discovery registry lacks frozen candidate state")
        state_path = project_path(str(state["path"]))
        if _sha256(state_path) != str(state["sha256"]):
            raise FullEndpointAuditError("frozen candidate state changed")
        candidate = load_candidate_state_path(state_path, expected_hash=str(record["candidate_hash"]))
        runner = build_candidate_runner(candidate, task_spec=tensors.task_spec, config=load_runtime_config("configs/joint_response_runtime.json"))
        runner["model"].load_state_dict(payload["model_state_dict"], strict=True)
        runner["model"].to(torch.device(device)).eval()
        return runner, tensors, {"kind": method, "checkpoint": payload}
    design = FalsificationGuidedDesign(**dict(record["design"]))
    design.validate()
    if payload.get("schema_version") != "cellscientist_increment_candidate_checkpoint_v1" or dict(payload.get("design") or {}) != design.to_dict():
        raise FullEndpointAuditError("falsification-guided discovery checkpoint design differs from frozen registry")
    model = _new_model(tensors=tensors, design=design)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model, tensors, {"kind": "falsification_guided", "checkpoint": payload, "selected_increment_scale": float(payload["selected_increment_scale"])}


def _predict(*, torch: Any, model: Any, method: str, tensors: Any, pre: Any, condition: Any, device: Any) -> Any:
    with torch.no_grad():
        normalized_pre = tensors.standardizer.normalize_pre(pre).to(device)
        normalized_condition = tensors.standardizer.normalize_condition(condition).to(device)
        chemical, dose = _split_inputs(normalized_condition, prefix_dim=int(tensors.task_spec.chemical_dim))
        prediction = model["forward"](normalized_pre, chemical, dose) if method in {"open", "source_constrained"} else model(normalized_pre, chemical, dose)
        return tensors.standardizer.inverse_target(prediction)


def _metrics(*, torch: Any, prediction: Any, target: Any, cp_dim: int) -> Mapping[str, float]:
    raw_target = target.to(prediction.device)
    if tuple(prediction.shape) != tuple(raw_target.shape):
        raise FullEndpointAuditError("audit prediction does not preserve joint output shape")
    return {"global_pcc": float(_global_pcc(torch, prediction, raw_target)), "mse": float((prediction - raw_target).square().mean().detach().cpu()), "cp_pcc": float(_global_pcc(torch, prediction[:, :cp_dim], raw_target[:, :cp_dim])), "l1000_pcc": float(_global_pcc(torch, prediction[:, cp_dim:], raw_target[:, cp_dim:]))}


def _effect(cc: Mapping[str, float], cw: Mapping[str, float], sc: Mapping[str, float], sw: Mapping[str, float]) -> Mapping[str, float]:
    return {"chemical_effect": 0.5 * ((cc["global_pcc"] - sc["global_pcc"]) + (cw["global_pcc"] - sw["global_pcc"])), "control_effect": 0.5 * ((cc["global_pcc"] - cw["global_pcc"]) + (sc["global_pcc"] - sw["global_pcc"])), "chemical_control_interaction": cc["global_pcc"] - sc["global_pcc"] - cw["global_pcc"] + sw["global_pcc"]}


def _dose_evidence(*, torch: Any, model: Any, method: str, tensors: Any, endpoint: Any, device: Any, map_seeds: Sequence[int]) -> Mapping[str, Any]:
    prefix_dim = int(tensors.task_spec.chemical_dim)
    dose_dim = int(tensors.task_spec.dose_dim)
    if dose_dim <= 0 or endpoint.condition.shape[1] != prefix_dim + dose_dim:
        return {"status": "not_identifiable", "reason": "registered task does not expose chemical-plus-dose input layout"}
    minimum_delta = GUIDED_MINIMUM_ABSOLUTE_DOSE_DELTA if method == "falsification_guided" else 0.0
    probe = build_matched_dose_sham_map(
        endpoint.condition[:, :prefix_dim],
        endpoint.condition[:, prefix_dim:],
        row_ids=endpoint.labels,
        seed=int(map_seeds[0]),
        minimum_absolute_dose_delta=minimum_delta,
    )
    coverage = {"eligible_row_count": int(probe.audit["eligible_row_count"]), "eligible_row_fraction": float(probe.audit["eligible_row_fraction"]), "eligible_chemical_count": int(probe.audit["eligible_chemical_count"]), "partition_row_count": int(probe.audit["partition_row_count"])}
    if (
        coverage["eligible_row_count"] < MIN_ELIGIBLE_ROWS
        or coverage["eligible_row_fraction"] < MIN_ELIGIBLE_FRACTION
        or (method == "falsification_guided" and coverage["eligible_chemical_count"] < GUIDED_MINIMUM_ELIGIBLE_CHEMICALS)
    ):
        return {"status": "not_identifiable", "coverage": coverage, "reason": "registered endpoint lacks sufficient within-chemical dose variation"}
    device_obj = torch.device(device)
    indices = torch.as_tensor(probe.source_indices, dtype=torch.long, device="cpu")
    pre = torch.as_tensor(endpoint.pre, dtype=torch.float32, device="cpu")[indices]
    condition = torch.as_tensor(endpoint.condition, dtype=torch.float32, device="cpu")[indices]
    target = torch.as_tensor(endpoint.target, dtype=torch.float32, device="cpu")[indices]
    cc = _metrics(torch=torch, prediction=_predict(torch=torch, model=model, method=method, tensors=tensors, pre=pre, condition=condition, device=device_obj), target=target, cp_dim=int(tensors.task_spec.cp_dim))
    maps: list[dict[str, Any]] = []
    for seed in map_seeds:
        dose_map = build_matched_dose_sham_map(
            endpoint.condition[:, :prefix_dim],
            endpoint.condition[:, prefix_dim:],
            row_ids=endpoint.labels,
            seed=int(seed),
            minimum_absolute_dose_delta=minimum_delta,
        )
        if tuple(dose_map.source_indices) != tuple(probe.source_indices):
            raise FullEndpointAuditError("dose eligibility changed across fixed sham maps")
        shuffled = condition.clone()
        shuffled[:, prefix_dim:] = torch.as_tensor(endpoint.condition[list(dose_map.donor_indices), prefix_dim:], dtype=torch.float32, device="cpu")
        sd = _metrics(torch=torch, prediction=_predict(torch=torch, model=model, method=method, tensors=tensors, pre=pre, condition=shuffled, device=device_obj), target=target, cp_dim=int(tensors.task_spec.cp_dim))
        maps.append({"map_seed": int(seed), "CC": cc, "CS": sd, "dose_effect": float(cc["global_pcc"] - sd["global_pcc"]), "dose_map_hash": str(dose_map.audit["map_hash"])})
    reference = _pairwise_absolute_differences([row["CS"]["global_pcc"] for row in maps])
    return {"status": "identifiable", "coverage": coverage, "cc_metrics": cc, "counterfactual_maps": maps, "map_variation_reference": reference, "decision": _decision([row["dose_effect"] for row in maps], reference)}


def _refit_open_source_for_fold5(
    *, record: Mapping[str, Any], task_id: str, unit_root: Path, device: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Refit one frozen open source before Fold 5 is materialized."""

    state = record.get("candidate_state")
    if not isinstance(state, Mapping):
        raise FullEndpointAuditError("open Fold-5 refit lacks a frozen candidate state")
    state_path = project_path(str(state["path"]))
    if _sha256(state_path) != str(state["sha256"]):
        raise FullEndpointAuditError("open Fold-5 source changed after registration")
    candidate = load_candidate_state_path(
        state_path, expected_hash=str(record["candidate_hash"])
    )
    config = load_runtime_config("configs/joint_response_runtime.json")
    settings = replace(
        JointResponseSettings.from_config(config, device=device),
        seed=int(record["trajectory_seed"]),
    )
    fold13 = load_fold13_arrays(task_id)
    checkpoint = unit_root / "fold3_selected_checkpoint.pt"
    fit_receipt = train_joint_candidate(
        "cellscientist_open_fold5_refit",
        arrays=fold13,
        config=config,
        settings=settings,
        checkpoint_path=checkpoint,
        candidate_override=candidate,
    )
    if not checkpoint.is_file():
        raise FullEndpointAuditError(
            "registered open Fold-5 refit did not write its Fold-3 checkpoint"
        )
    replay_record = dict(record)
    replay_record["checkpoint"] = {
        "path": _relative(checkpoint),
        "sha256": _sha256(checkpoint),
    }
    receipt = {
        "protocol": "refit frozen source on Folds 1-2 and select checkpoint on Fold 3",
        "training_seed": int(record["trajectory_seed"]),
        "candidate_hash": candidate.candidate_hash,
        "source_candidate_state": dict(state),
        "checkpoint": dict(replay_record["checkpoint"]),
        "fit_receipt": fit_receipt,
        "provider_calls": 0,
        "fold4_or_fold5_used_for_fitting_or_selection": False,
    }
    return replay_record, receipt


def _run_endpoint(*, root: Path, record: Mapping[str, Any], fold: int, device: str) -> Mapping[str, Any]:
    freeze, registry = _load_frozen(root)
    method, task_id = str(registry["method"]), str(registry["task_id"])
    unit_root = root / f"fold{fold}_units" / str(record["audit_unit_id"])
    execution_record: Mapping[str, Any] = record
    fold5_refit: Mapping[str, Any] | None = None
    if fold == 5 and method == "open":
        # Refitting uses only Folds 1--3.  Fold 5 is loaded below only after the
        # selected source has produced its new Fold-3-selected checkpoint.
        unit_root.mkdir(parents=True, exist_ok=False)
        execution_record, fold5_refit = _refit_open_source_for_fold5(
            record=record, task_id=task_id, unit_root=unit_root, device=device
        )
    if fold == 4:
        arrays, map_seeds, dose_seeds = load_fold14_arrays(task_id), CHEMICAL_CONTROL_MAP_SEEDS_F4, DOSE_MAP_SEEDS_F4
    elif fold == 5:
        arrays, map_seeds, dose_seeds = load_fold5_arrays(task_id), CHEMICAL_CONTROL_MAP_SEEDS_F5, DOSE_MAP_SEEDS_F5
    else:  # pragma: no cover
        raise FullEndpointAuditError("fold must be 4 or 5")
    torch = _torch()
    model, tensors, model_meta = _load_model(record=execution_record, method=method, arrays=arrays, device=device)
    endpoint, device_obj = arrays.endpoint, torch.device(device)
    pre = torch.as_tensor(endpoint.pre, dtype=torch.float32, device="cpu")
    condition = torch.as_tensor(endpoint.condition, dtype=torch.float32, device="cpu")
    target = torch.as_tensor(endpoint.target, dtype=torch.float32, device="cpu")
    cp_dim, prefix_dim = int(tensors.task_spec.cp_dim), int(tensors.task_spec.chemical_dim)
    started = time.perf_counter()

    def score(value_pre: Any, value_condition: Any) -> Mapping[str, float]:
        return _metrics(torch=torch, prediction=_predict(torch=torch, model=model, method=method, tensors=tensors, pre=value_pre, condition=value_condition, device=device_obj), target=target, cp_dim=cp_dim)

    cc = score(pre, condition)
    maps: list[dict[str, Any]] = []
    for seed in map_seeds:
        control_map = build_control_profile_sham_map(endpoint.pre, row_ids=endpoint.labels, seed=int(seed))
        chemical_map = build_chemical_prefix_sham_map(endpoint.pre, endpoint.condition[:, :prefix_dim], row_ids=endpoint.labels, seed=int(seed))
        wrong_pre = torch.as_tensor(control_map.apply(endpoint.pre), dtype=torch.float32, device="cpu")
        shuffled = condition.clone()
        shuffled[:, :prefix_dim] = torch.as_tensor(chemical_map.apply(endpoint.condition[:, :prefix_dim]), dtype=torch.float32, device="cpu")
        cw, sc, sw = score(wrong_pre, condition), score(pre, shuffled), score(wrong_pre, shuffled)
        maps.append({"map_seed": int(seed), "CC": cc, "CW": cw, "SC": sc, "SW": sw, **_effect(cc, cw, sc, sw), "chemical_target_loss_gain": 0.5 * ((sc["mse"] - cc["mse"]) + (sw["mse"] - cw["mse"])), "chemical_map_hash": str(chemical_map.audit["map_hash"]), "control_map_hash": str(control_map.audit["map_hash"])})
    map_variation = {"chemical_effect": _pairwise_absolute_differences([0.5 * (row["SC"]["global_pcc"] + row["SW"]["global_pcc"]) for row in maps]), "control_effect": _pairwise_absolute_differences([0.5 * (row["CW"]["global_pcc"] + row["SW"]["global_pcc"]) for row in maps]), "chemical_control_interaction": _pairwise_absolute_differences([row["SC"]["global_pcc"] - row["SW"]["global_pcc"] for row in maps])}
    context_anchor: Mapping[str, Any] | None = None
    if method == "falsification_guided":
        selected_scale = float(model_meta["selected_increment_scale"])
        model.set_increment_scale(0.0)
        anchor = score(pre, condition)
        model.set_increment_scale(selected_scale)
        context_anchor = {"cc_metrics": anchor, "increment_over_anchor": float(cc["global_pcc"] - anchor["global_pcc"])}
    checker_receipt = _read(project_path(str(record["checker_receipt"]["path"])))
    result = {"schema_version": "cellscientist_endpoint_audit_all_endpoint_unit_v1", "status": "complete", "provider_calls": 0, "method": method, "task_id": task_id, "fold": fold, "audit_unit_id": record["audit_unit_id"], "trajectory": int(record["trajectory"]), "trajectory_seed": int(record["trajectory_seed"]), "checkpoint": dict(execution_record["checkpoint"]), "fold5_refit": fold5_refit, "checker_receipt": dict(record["checker_receipt"]), "source_contract": _checker_contract(checker_receipt), "fold3_metrics": dict(record["fold3_metrics"]), "cc_metrics": cc, "context_anchor": context_anchor, "counterfactual_maps": maps, "map_variation_references": map_variation, "chemical_identity": _decision([row["chemical_effect"] for row in maps], map_variation["chemical_effect"]), "control_context": _decision([row["control_effect"] for row in maps], map_variation["control_effect"]), "chemical_control_interaction": _decision([row["chemical_control_interaction"] for row in maps], map_variation["chemical_control_interaction"], absolute=True), "dose": _dose_evidence(torch=torch, model=model, method=method, tensors=tensors, endpoint=endpoint, device=device, map_seeds=dose_seeds), "elapsed_seconds": time.perf_counter() - started}
    if not unit_root.exists():
        unit_root.mkdir(parents=True, exist_ok=False)
    _write(unit_root / "summary.json", result)
    (unit_root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return result


def run_unit(*, output_root: str | Path, audit_unit_id: str, fold: int, device: str) -> Mapping[str, Any]:
    root = project_path(output_root)
    if fold == 5:
        _load_fold5_authorization(root)
    _freeze, registry = _load_frozen(root)
    matches = [row for row in registry["records"] if row["audit_unit_id"] == audit_unit_id]
    if len(matches) != 1:
        raise FullEndpointAuditError("unknown or ambiguous audit unit")
    unit_root = root / f"fold{fold}_units" / audit_unit_id
    if unit_root.exists():
        raise FullEndpointAuditError("audit unit output exists; refusing overwrite")
    return _run_endpoint(root=root, record=matches[0], fold=fold, device=device)


def run_all(*, output_root: str | Path, fold: int, device: str) -> Mapping[str, Any]:
    root = project_path(output_root)
    if fold == 5:
        _load_fold5_authorization(root)
    _freeze, registry = _load_frozen(root)
    completed: list[str] = []
    for record in registry["records"]:
        unit_id = str(record["audit_unit_id"])
        unit_root = root / f"fold{fold}_units" / unit_id
        if (unit_root / "_SUCCESS").is_file() and (unit_root / "summary.json").is_file():
            receipt = _read(unit_root / "summary.json")
            try:
                _validate_completed_unit(
                    unit=receipt, record=record, fold=fold, method=str(registry["method"])
                )
            except FullEndpointAuditError as exc:
                raise FullEndpointAuditError("existing audit unit does not validate for resume") from exc
        elif unit_root.exists():
            raise FullEndpointAuditError("incomplete audit unit exists; refusing overwrite")
        else:
            _run_endpoint(root=root, record=record, fold=fold, device=device)
        completed.append(unit_id)
    return {"status": "complete", "provider_calls": 0, "fold": fold, "completed_units": completed}


def _status_counts(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, int]:
    labels = ("claim_supported", "behaviorally_unsupported", "inconclusive", "not_identifiable")
    def status(row: Mapping[str, Any]) -> str:
        item = row[key]
        # Dose is a two-stage record: identifiability is determined first, then
        # an identifiable endpoint receives the same map-calibrated decision as
        # chemical/control.  Count the decision rather than the intermediary
        # ``identifiable`` label.
        if item.get("status") == "identifiable" and isinstance(item.get("decision"), Mapping):
            return str(item["decision"].get("status", ""))
        return str(item.get("status", ""))
    return {label: sum(status(row) == label for row in rows) for label in labels}


def _aggregate_five_refit_status(counts: Mapping[str, int]) -> str:
    """Return the registered four-of-five consensus for one input-use test."""

    if sum(int(value) for value in counts.values()) != OPEN_REFIT_COUNT:
        raise FullEndpointAuditError("open aggregate requires exactly five refit decisions")
    for label in (
        "claim_supported",
        "behaviorally_unsupported",
        "not_identifiable",
        "inconclusive",
    ):
        if int(counts.get(label, 0)) >= 4:
            return label
    return "inconclusive"


def _validate_completed_unit(
    *, unit: Mapping[str, Any], record: Mapping[str, Any], fold: int, method: str
) -> None:
    """Validate a completed audit unit before resume or aggregation."""

    if (
        unit.get("status") != "complete"
        or unit.get("audit_unit_id") != record.get("audit_unit_id")
        or int(unit.get("fold", -1)) != fold
        or unit.get("provider_calls") != 0
    ):
        raise FullEndpointAuditError("audit unit violates deterministic replay protocol")
    if method != "open" or fold != 5:
        return
    refit = unit.get("fold5_refit")
    checkpoint = unit.get("checkpoint")
    if not isinstance(refit, Mapping) or not isinstance(checkpoint, Mapping):
        raise FullEndpointAuditError("open Fold-5 unit lacks its registered refit receipt")
    if (
        str(refit.get("candidate_hash")) != str(record.get("candidate_hash"))
        or int(refit.get("training_seed", -1)) != int(record.get("trajectory_seed", -2))
        or dict(refit.get("checkpoint") or {}) != dict(checkpoint)
    ):
        raise FullEndpointAuditError("open Fold-5 refit does not bind the frozen source and seed")
    checkpoint_path = project_path(str(checkpoint.get("path")))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != str(checkpoint.get("sha256")):
        raise FullEndpointAuditError("open Fold-5 refit checkpoint changed after completion")


def summarize(*, output_root: str | Path, fold: int) -> Mapping[str, Any]:
    root = project_path(output_root)
    _freeze, registry = _load_frozen(root)
    file_name = FOLD4_SUMMARY_FILE if fold == 4 else FOLD5_SUMMARY_FILE
    output = root / file_name
    if output.exists():
        raise FullEndpointAuditError("summary already exists; preserve its frozen receipt")
    rows: list[Mapping[str, Any]] = []
    for record in registry["records"]:
        path = root / f"fold{fold}_units" / str(record["audit_unit_id"]) / "summary.json"
        if not path.is_file() or not (path.parent / "_SUCCESS").is_file():
            raise FullEndpointAuditError(f"missing completed Fold-{fold} unit: {path.parent}")
        unit = _read(path)
        _validate_completed_unit(
            unit=unit, record=record, fold=fold, method=str(registry["method"])
        )
        rows.append(unit)
    predictive = {metric: _mean_ci([float(row["cc_metrics"][metric]) for row in rows]) for metric in ("global_pcc", "mse", "cp_pcc", "l1000_pcc")}
    continuous = {"chemical_pcc_drop": _mean_ci([float(row["chemical_identity"]["mean_effect"]) for row in rows]), "control_pcc_drop": _mean_ci([float(row["control_context"]["mean_effect"]) for row in rows]), "interaction_magnitude": _mean_ci([float(row["chemical_control_interaction"]["mean_effect"]) for row in rows]), "chemical_target_loss_gain": _mean_ci([mean(float(item["chemical_target_loss_gain"]) for item in row["counterfactual_maps"]) for row in rows])}
    identifiable_dose = [row for row in rows if row["dose"].get("status") == "identifiable"]
    if identifiable_dose:
        continuous["dose_pcc_drop"] = _mean_ci([float(row["dose"]["decision"]["mean_effect"]) for row in identifiable_dose])
    guided_rows = [row for row in rows if row.get("context_anchor") is not None]
    if guided_rows:
        continuous["increment_over_anchor"] = _mean_ci([float(row["context_anchor"]["increment_over_anchor"]) for row in guided_rows])
    status_counts = {
        "chemical": _status_counts(rows, "chemical_identity"),
        "control": _status_counts(rows, "control_context"),
        "interaction": _status_counts(rows, "chemical_control_interaction"),
        "dose": _status_counts(rows, "dose"),
    }
    aggregate_status = None
    if registry["method"] == "open":
        aggregate_status = {
            key: _aggregate_five_refit_status(counts)
            for key, counts in status_counts.items()
        }
    result = {"schema_version": "cellscientist_endpoint_audit_all_endpoint_summary_v1", "status": "complete", "provider_calls": 0, "method": registry["method"], "dataset": registry["dataset"], "task_id": registry["task_id"], "fold": fold, "endpoint_count": len(rows), "predictive": predictive, "continuous_effects": continuous, "status_counts": status_counts, "aggregate_status": aggregate_status, "rows": rows}
    _write(output, result)
    lines = [f"# {str(registry['method']).upper()} all-endpoint Fold-{fold} deterministic audit", "", "| Trajectory | PCC | Chemical drop | Control drop | Chemical | Control | Interaction | Dose |", "|---:|---:|---:|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['trajectory']} | {row['cc_metrics']['global_pcc']:.4f} | {row['chemical_identity']['mean_effect']:.4f} | {row['control_context']['mean_effect']:.4f} | {row['chemical_identity']['status']} | {row['control_context']['status']} | {row['chemical_control_interaction']['status']} | {row['dose'].get('decision', row['dose']).get('status', row['dose'].get('status'))} |")
    (root / f"FOLD{fold}_AUDIT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def freeze_fold5(*, output_root: str | Path) -> Mapping[str, Any]:
    root = project_path(output_root)
    if (root / FOLD5_FREEZE_FILE).exists():
        raise FullEndpointAuditError("Fold-5 authorization already exists")
    freeze, registry = _load_frozen(root)
    summary = root / FOLD4_SUMMARY_FILE
    if not summary.is_file() or not all((root / "fold4_units" / str(row["audit_unit_id"]) / "_SUCCESS").is_file() for row in registry["records"]):
        raise FullEndpointAuditError("all Fold-4 units and summary must complete before Fold-5 authorization")
    authorization = {
        "schema_version": "cellscientist_endpoint_audit_all_endpoint_fold5_freeze_v1",
        "status": "frozen_after_complete_fold4_before_fold5",
        "fold5_authorized": True,
        "provider_calls": 0,
        "parent_fold4_freeze": {"path": _relative(root / FOLD4_FREEZE_FILE), "sha256": _sha256(root / FOLD4_FREEZE_FILE)},
        "parent_fold4_summary": {"path": _relative(summary), "sha256": _sha256(summary)},
        "registry": {"path": _relative(root / REGISTRY_FILE), "sha256": _sha256(root / REGISTRY_FILE)},
        "fold5_model_state": "refit frozen source on Folds 1-2 under the same five-seed schedule and select checkpoints on Fold 3" if registry["method"] == "open" else "replay the frozen discovery checkpoints",
        "chemical_control_counterfactual": {"map_seeds": list(CHEMICAL_CONTROL_MAP_SEEDS_F5), "design": "same 2x2 intervention family with newly registered target-blind map seeds"},
        "dose_counterfactual": {"map_seeds": list(DOSE_MAP_SEEDS_F5), "design": "same within-chemical observed-dose derangement with newly registered target-blind map seeds"},
        "prohibited": ["provider call", "candidate rewrite", "candidate reselection", "unregistered checkpoint training", "Fold-5-driven rule modification", "manual result exclusion"],
        "expected_units": len(registry["records"]),
    }
    _write(root / FOLD5_FREEZE_FILE, authorization)
    (root / "_FOLD5_READY").write_text("frozen; deterministic Fold-5 replay is authorized\n", encoding="utf-8")
    return authorization


def _load_fold5_authorization(root: Path) -> Mapping[str, Any]:
    freeze = _read(root / FOLD5_FREEZE_FILE)
    if freeze.get("schema_version") != "cellscientist_endpoint_audit_all_endpoint_fold5_freeze_v1" or freeze.get("status") != "frozen_after_complete_fold4_before_fold5" or freeze.get("fold5_authorized") is not True:
        raise FullEndpointAuditError("invalid Fold-5 authorization")
    for key in ("parent_fold4_freeze", "parent_fold4_summary", "registry"):
        receipt = freeze.get(key)
        if not isinstance(receipt, Mapping) or _sha256(project_path(str(receipt["path"]))) != str(receipt["sha256"]):
            raise FullEndpointAuditError("Fold-5 authorization parent receipt changed")
    return freeze
