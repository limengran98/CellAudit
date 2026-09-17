"""Agentic falsification-guided discovery with deterministic compilation and qualification."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Mapping

from cellaudit.runtime_utils import _provider, _safe_error, _write_json
from cellaudit.provider_client import DiscoveryError
from cellaudit.schemas import canonical_json, digest
from cellaudit.joint_response_runtime import project_path

from .model import FalsificationGuidedDesign
from .trainer import FalsificationGuidedTrainingError, run_guided_candidate, train_context_anchor


class FalsificationGuidedDiscoveryError(RuntimeError):
    """The constrained falsification-guided discovery discovery trajectory violated its frozen contract."""


INITIAL_FUSIONS = ("centered_additive", "factorized_gate", "low_rank_bilinear")


def _config(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(project_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FalsificationGuidedDiscoveryError("cannot read falsification-guided discovery discovery configuration") from exc
    required = {"schema_version", "provider", "search", "candidate_contract"}
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema_version") != "cellscientist_falsification_guided_discovery"
    ):
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery discovery configuration schema is invalid")
    if value["provider"] != {
        "model_panel": "configs/model_panel.json",
        "panel_key": "deepseek_v4_flash",
        "allow_fallback": False,
    }:
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery requires the registered DeepSeek V4 Flash lock without fallback")
    if int(value["search"].get("candidate_budget_including_anchor", 0)) != 10:
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery requires one anchor plus nine physical candidate fits")
    return value


def _view(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = result["selected_metrics"]
    diagnostics = result["development_diagnostics"]
    return {
        "global_pcc": float(metrics["global_pcc"]),
        "cp_pcc": float(metrics["cp_pcc"]),
        "l1000_pcc": float(metrics["l1000_pcc"]),
        "mse": float(metrics["mse"]),
        "chemical_status": diagnostics["chemical"]["status"],
        "chemical_target_loss_gain": float(diagnostics["chemical"]["mean_target_loss_gain"]),
        "dose_status": diagnostics["dose"]["status"],
        "qualified": bool(result["qualified"]),
        "selected_epoch": int(result["selected_epoch"]),
        "selected_increment_scale": float(result["selected_increment_scale"]),
    }


def _prompt(
    *, task_id: str, slot: int, incumbent: Mapping[str, Any] | None,
    history: list[Mapping[str, Any]], repair: str | None,
) -> list[dict[str, str]]:
    exact_fusion = INITIAL_FUSIONS[slot - 1] if slot <= 3 else None
    system = (
        "You are the CellScientist falsification-guided discovery perturbation-increment discovery policy. "
        "Return exactly one JSON object and no Markdown. You propose a semantic design card, never source code. "
        "A deterministic compiler owns the frozen context anchor, source code, tensor shapes, optimizer, folds, "
        "counterfactual maps, identifiability rules, performance gate, and Pareto scale calibration. "
        "The scientific goal is lexicographic: retain predictive utility while obtaining target-related evidence "
        "that the predictor uses identifiable perturbation inputs beyond context. Never optimize output sensitivity alone. "
        "Allowed fusion={centered_additive,factorized_gate,low_rank_bilinear}; "
        "hidden_dim={256,384,512}; readout={shared,dual_heads}; "
        "objective={balanced_residual,correlation_residual}."
    )
    phase = (
        "independent complementary hypothesis" if slot <= 3
        else "bounded evidence-conditioned revision of the current frontier"
    )
    constraints = [
        "Return every design field, even when it is inherited conceptually.",
        "Use only registered values.",
        "Do not request changes to data, folds, anchor, thresholds, optimizer, or evaluator.",
    ]
    if exact_fusion is not None:
        constraints.append(f"fusion must equal {exact_fusion}")
    if slot >= 4:
        constraints.append("Change one or two design fields relative to a prior executable design.")
    payload = {
        "task_id": task_id,
        "slot": slot,
        "phase": phase,
        "incumbent": incumbent,
        "bounded_history": history[-5:],
        "constraints": constraints,
        "response_schema": {
            "hypothesis": "string",
            "rationale": "string",
            "design": {
                "fusion": "registered string",
                "hidden_dim": "registered integer",
                "readout": "registered string",
                "objective": "registered string",
            },
        },
    }
    if repair:
        payload["previous_validation_failure"] = repair
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": canonical_json(payload)},
    ]


def _parse(text: str, *, slot: int) -> tuple[FalsificationGuidedDesign, Mapping[str, Any]]:
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise FalsificationGuidedDiscoveryError("provider response is not one standalone JSON design card") from exc
    if not isinstance(value, Mapping):
        raise FalsificationGuidedDiscoveryError("provider design card must be an object")
    hypothesis = str(value.get("hypothesis", "")).strip()
    rationale = str(value.get("rationale", "")).strip()
    raw_design = value.get("design")
    if not hypothesis or not rationale or not isinstance(raw_design, Mapping):
        raise FalsificationGuidedDiscoveryError("provider card lacks a hypothesis, rationale, or design object")
    try:
        design = FalsificationGuidedDesign(
            fusion=str(raw_design.get("fusion", "")),
            hidden_dim=int(raw_design.get("hidden_dim", 0)),
            readout=str(raw_design.get("readout", "")),
            objective=str(raw_design.get("objective", "")),
        )
        design.validate()
    except (TypeError, ValueError) as exc:
        raise FalsificationGuidedDiscoveryError("provider card contains an unregistered falsification-guided discovery design") from exc
    if slot <= 3 and design.fusion != INITIAL_FUSIONS[slot - 1]:
        raise FalsificationGuidedDiscoveryError("initial falsification-guided discovery card violates the registered complementary fusion schedule")
    return design, {"hypothesis": hypothesis, "rationale": rationale, "design": design.to_dict()}


def _better(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_key = (bool(left["qualified"]), float(left["global_pcc"]), -float(left["mse"]))
    right_key = (bool(right["qualified"]), float(right["global_pcc"]), -float(right["mse"]))
    return left_key > right_key


def run_falsification_guided_discovery(
    *, task_id: str, output_root: str | Path, mode: str, device: str,
    config_path: str | Path, trajectory_seed: int, proposal_slots: int | None = None,
) -> Mapping[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery mode must be smoke or formal")
    config = _config(config_path)
    max_slots = 9 if proposal_slots is None else int(proposal_slots)
    if max_slots not in range(1, 10) or (mode == "formal" and max_slots != 9):
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery proposal slots violate the registered physical-fit budget")
    root = project_path(output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise FalsificationGuidedDiscoveryError("falsification-guided discovery output root must be a new directory beneath runs/")
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "falsification_guided_config_snapshot.json", config)
    _write_json(root / "seed_registration.json", {
        "trajectory_seed": int(trajectory_seed),
        "candidate_seed_rule": "trajectory_seed*100+slot+1",
    })
    client, provider = _provider(config)
    _write_json(root / "provider_receipt.json", provider)
    max_epochs = (
        int(config["search"]["smoke_max_epochs"])
        if mode == "smoke" else int(config["search"]["formal_max_epochs"])
    )
    anchor_path = root / "context_anchor.pt"
    anchor = train_context_anchor(
        task_id=task_id, checkpoint_path=anchor_path, device=device,
        seed=int(trajectory_seed) * 100 + 1, max_epochs=max_epochs,
    )
    records: list[dict[str, Any]] = []
    calls: list[Mapping[str, Any]] = []
    seen_designs: set[str] = set()
    incumbent: Mapping[str, Any] | None = None
    started = time.perf_counter()
    for slot in range(1, max_slots + 1):
        candidate_id = f"increment_{slot:02d}"
        candidate_root = root / "candidates" / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=False)
        failures: list[Mapping[str, Any]] = []
        for attempt in range(1, 4):
            try:
                response = client.chat(
                    _prompt(
                        task_id=task_id, slot=slot,
                        incumbent=(dict(incumbent) if incumbent is not None else None),
                        history=[{
                            "candidate_id": item["candidate_id"],
                            "design": item.get("design"),
                            "feedback": item.get("feedback"),
                        } for item in records if item.get("status") == "complete"],
                        repair=(str(failures[-1]["message"]) if failures else None),
                    ),
                    temperature=float(config["search"]["temperature"]),
                    max_tokens=int(config["search"]["max_output_tokens"]),
                )
                calls.append({
                    "slot": slot, "attempt": attempt, "purpose": "falsification_guided_design_card",
                    "audit": dict(client.last_call_audit or {}),
                })
                design, card = _parse(response, slot=slot)
                design_digest = digest(design.to_dict())
                if design_digest in seen_designs:
                    raise FalsificationGuidedDiscoveryError("provider proposed a duplicate compiled falsification-guided discovery design")
                _write_json(candidate_root / "design_card.json", {
                    **card, "design_digest": design_digest,
                    "compiler": "deterministic_FalsificationGuidedDesign_v1",
                })
                result = run_guided_candidate(
                    task_id=task_id, anchor_checkpoint=anchor_path,
                    output_root=candidate_root / f"execution_attempt_{attempt:02d}", design=design,
                    device=device, seed=int(trajectory_seed) * 100 + slot + 1,
                    max_epochs=max_epochs,
                )
                feedback = _view(result)
                record = {
                    "candidate_id": candidate_id, "status": "complete",
                    "hypothesis": card["hypothesis"], "rationale": card["rationale"],
                    "design": design.to_dict(), "design_digest": design_digest,
                    "execution_root": str(candidate_root / f"execution_attempt_{attempt:02d}"),
                    "feedback": feedback, "card_attempts": attempt,
                    "card_failures": failures,
                }
                records.append(record)
                seen_designs.add(design_digest)
                if incumbent is None or _better(feedback, incumbent["feedback"]):
                    incumbent = record
                break
            except (DiscoveryError, FalsificationGuidedDiscoveryError, FalsificationGuidedTrainingError, ValueError) as exc:
                failure = {"attempt": attempt, "message": _safe_error(exc)}
                failures.append(failure)
                _write_json(candidate_root / f"card_failure_{attempt:02d}.json", failure)
        else:
            records.append({
                "candidate_id": candidate_id, "status": "failed", "feedback": None,
                "card_failures": failures,
            })
    qualified = [item for item in records if item.get("status") == "complete" and item["feedback"]["qualified"]]
    selected = max(
        qualified,
        key=lambda item: (float(item["feedback"]["global_pcc"]), -float(item["feedback"]["mse"])),
        default=None,
    )
    summary = {
        "schema_version": "cellscientist_falsification_guided_constrained_discovery_run_v1",
        "status": "complete", "mode": mode, "task_id": task_id,
        "trajectory_seed": int(trajectory_seed),
        "provider_calls": len(calls), "provider_calls_receipt": calls,
        "candidate_budget_including_anchor": max_slots + 1,
        "anchor": {
            "checkpoint": str(anchor_path),
            "selection_metrics": anchor["selection_metrics"],
        },
        "records": records,
        "selection_status": "qualified_endpoint" if selected is not None else "no_qualified_model",
        "selected_candidate_id": selected["candidate_id"] if selected is not None else None,
        "fold4_or_fold5_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(root / "summary.json", summary)
    (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return summary
