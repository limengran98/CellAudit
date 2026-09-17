"""Source-constrained multimodal discovery with parent-relative evidence memory."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

from cellaudit.runtime_utils import _feedback, _provider, _safe_error, _write_json
from cellaudit.provider_client import DiscoveryError
from cellaudit.open_discovery import CandidateState, OpenDiscoveryError
from cellaudit.schemas import canonical_json, digest
from cellaudit.joint_response_runtime import (
    JointResponseSettings, JointResponseRuntimeError, load_runtime_config,
    load_starting_candidate, load_fold13_arrays, project_path,
    registered_candidate_seed, train_joint_candidate,
)

from .selection import Fold3CandidateScore, select_endpoint
from .types import DesignCard, DesignCardError
from .grouped_compiler import (
    FUSIONS, HIDDEN, LOSSES, READOUTS, compile_card,
    materialize_candidate_source,
)
from .grouped_schedule import instruction_for


class SourceConstrainedDiscoveryError(RuntimeError):
    pass


_METRICS = (
    "fold3_global_pcc", "fold3_mse", "fold3_cp_pcc", "fold3_l1000_pcc",
)


def _config(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(project_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceConstrainedDiscoveryError("cannot read source-constrained discovery grouped configuration") from exc
    required = {
        "schema_version", "runtime_config", "provider", "search", "controller",
        "candidate_contract", "mechanism_contract",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("schema_version") != "cellscientist_source_constrained_discovery"
    ):
        raise SourceConstrainedDiscoveryError("source-constrained discovery grouped configuration schema is invalid")
    if value["provider"] != {
        "model_panel": "configs/model_panel.json",
        "panel_key": "deepseek_v4_flash",
        "allow_fallback": False,
    }:
        raise SourceConstrainedDiscoveryError("source-constrained discovery requires the registered DeepSeek V4 Flash lock without fallback")
    if int(value["search"].get("candidate_budget_including_h0", 0)) != 10:
        raise SourceConstrainedDiscoveryError("the grouped study requires ten physical fit slots including h0")
    expected_contract = {
        "grouped_output_diagnostics": True,
        "parent_relative_evidence": True,
        "require_distinct_initial_fusions": True,
        "forbid_single_token_attention": True,
        "endpoint_heldout_behavioral_audit_required": True,
        "selection": "fold3_global_pcc_then_mse_then_block_lcb",
    }
    if value["mechanism_contract"] != expected_contract:
        raise SourceConstrainedDiscoveryError("the grouped multimodal mechanism contract is invalid")
    return value


def _metric_view(feedback: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(feedback[key]) for key in _METRICS}


def _metric_delta(child: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, float]:
    return {key: round(float(child[key]) - float(parent[key]), 8) for key in _METRICS}


def _block_summary(feedback: Mapping[str, Any]) -> Mapping[str, Any]:
    cp = float(feedback["fold3_cp_pcc"])
    second = float(feedback["fold3_l1000_pcc"])
    weakest = "group_1" if cp < second else "group_2"
    return {
        "group_1_pcc": cp,
        "group_2_pcc": second,
        "weakest_group": weakest,
        "absolute_gap": round(abs(cp - second), 8),
    }


def _frontier(records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    complete = [item for item in records if item.get("status") == "complete" and isinstance(item.get("feedback"), Mapping)]
    definitions = {
        "best_global": ("fold3_global_pcc", True),
        "best_group_1": ("fold3_cp_pcc", True),
        "best_group_2": ("fold3_l1000_pcc", True),
        "lowest_mse": ("fold3_mse", False),
    }
    result: dict[str, Any] = {}
    for label, (metric, maximize) in definitions.items():
        chosen = sorted(
            complete,
            key=lambda item: float(item["feedback"][metric]),
            reverse=maximize,
        )[0]
        result[label] = {
            "candidate_id": chosen["candidate_id"],
            "metrics": _metric_view(chosen["feedback"]),
            "compiler_choices": chosen.get("compiler_choices"),
        }
    return result


def _trial_evidence(records: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    evidence: list[Mapping[str, Any]] = []
    for item in records:
        if item.get("status") != "complete" or item.get("candidate_id") == "h0":
            continue
        evidence.append({
            "candidate_id": item["candidate_id"],
            "parent_candidate_id": item.get("parent_candidate_id"),
            "edits": item.get("edits", []),
            "compiler_choices": item.get("compiler_choices"),
            "parent_relative_delta": item.get("parent_relative_delta"),
        })
    # Preserve every positive global trial plus the three most recent trials.
    positive_ids = {
        str(item["candidate_id"])
        for item in evidence
        if isinstance(item.get("parent_relative_delta"), Mapping)
        and float(item["parent_relative_delta"]["fold3_global_pcc"]) > 0.0
    }
    recent_ids = {str(item["candidate_id"]) for item in evidence[-3:]}
    return [item for item in evidence if str(item["candidate_id"]) in positive_ids | recent_ids]


def _attempted_choices(records: list[Mapping[str, Any]]) -> Mapping[str, list[object]]:
    keys = {
        "fusion", "readout", "hidden_dim", "loss", "deep_chemical_encoder",
        "bottleneck", "residual_response",
    }
    result: dict[str, set[object]] = {key: set() for key in keys}
    for item in records:
        choices = item.get("compiler_choices")
        if not isinstance(choices, Mapping):
            continue
        for key in keys:
            result[key].add(choices[key])
    return {key: sorted(values, key=str) for key, values in result.items()}


def _prompt(
    *, task_id: str, slot: int, parent_id: str | None,
    incumbent_feedback: Mapping[str, Any], incumbent_choices: Mapping[str, object] | None,
    frontier: Mapping[str, Any], evidence: list[Mapping[str, Any]],
    attempted: Mapping[str, list[object]], repair: str | None,
    initial_fusions: tuple[str, ...],
) -> list[dict[str, str]]:
    instruction = instruction_for(slot)
    remaining_fusions = sorted(FUSIONS - set(initial_fusions))
    system = (
        "You are the CellScientist source-constrained discovery grouped-response discovery policy. Return exactly one JSON object and no Markdown. "
        "You propose a falsifiable design card, never source code. A deterministic compiler owns source, metadata, tensor contracts, optimizer implementation, and training interfaces. "
        "Use parent-relative evidence rather than repeating an unsuccessful edit. Preserve strong output groups while addressing the weakest group. "
        "Valid components and mechanisms are: "
        "model_architecture={hidden_256,hidden_384,hidden_512}; "
        "chemical_encoder={standard,deep_two_layer}; "
        "fusion={film,gated_residual,multitoken_attention,bilinear_gate,context_residual_adapter}; "
        "response_program={standard,bottleneck_64,residual}; "
        "response_readout={shared,dual_heads,group_experts}; "
        "loss={mse,pcc_mse,group_balanced_mse,group_balanced_pcc_mse}; "
        "optimizer={adam,adamw}; scheduler={cosine,none}. "
        "The output groups are generic registered target blocks; do not assume dataset-specific biology. "
        "The context_residual_adapter preserves a context-dose anchor and adds a gated perturbation residual. "
        "The group_experts readout shares a response program but gives each registered output group a residual expert."
    )
    constraints: list[str] = []
    if instruction.exact_edits:
        constraints.append(
            "Return exactly these component/mechanism edits and no others: "
            + canonical_json([
                {"component": component, "mechanism": mechanism}
                for component, mechanism in instruction.exact_edits
            ])
            + ". Supply your own falsifiable rationale for each edit."
        )
    if instruction.required_components:
        constraints.append(
            "At least one edit must address one of these components: " + ",".join(instruction.required_components) + "."
        )
    if slot >= 4:
        constraints.append("Change one or two components relative to the explicit incumbent choices.")
    if slot == 9 and not instruction.exact_edits:
        constraints.append("Integrate only choices with positive observed evidence in the supplied frontier or parent-relative trials.")
    repair_text = "" if repair is None else " Previous validation failure to correct: " + repair
    payload = {
        "task": task_id,
        "slot": slot,
        "mode": instruction.mode,
        "goal": instruction.goal,
        "parent": parent_id,
        "incumbent": {
            "compiler_choices": incumbent_choices,
            "metrics": _metric_view(incumbent_feedback),
            "group_diagnostic": _block_summary(incumbent_feedback),
        },
        "metric_frontier": frontier,
        "parent_relative_trial_evidence": evidence,
        "attempted_choice_values": attempted,
        "initial_fusions_completed": list(initial_fusions),
        "remaining_initial_fusions": remaining_fusions,
        "constraints": constraints,
    }
    user = (
        canonical_json(payload)
        + "\nResponse schema: {\"hypothesis\":string,\"edits\":[{\"component\":string,\"mechanism\":string,\"rationale\":string}]}"
        + repair_text
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_card(text: str, *, candidate_id: str, slot: int, parent_id: str | None,
                evidence_ids: tuple[str, ...]) -> DesignCard:
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DesignCardError("provider response is not one standalone JSON design card") from exc
    if not isinstance(value, Mapping):
        raise DesignCardError("provider design card must be an object")
    payload = dict(value.get("design_card", value))
    # Normalize only harmless schema shorthands.  This repairs a field name,
    # never a scientific mechanism or a compiler choice.
    component_aliases = {
        "architecture": "model_architecture",
        "readout": "response_readout",
        "objective": "loss",
    }
    raw_edits = payload.get("edits")
    if isinstance(raw_edits, list):
        normalized_edits: list[Any] = []
        for raw_edit in raw_edits:
            if not isinstance(raw_edit, Mapping):
                normalized_edits.append(raw_edit)
                continue
            normalized = dict(raw_edit)
            component = str(normalized.get("component", "")).strip().lower().replace("-", "_")
            normalized["component"] = component_aliases.get(component, component)
            normalized_edits.append(normalized)
        payload["edits"] = normalized_edits
    payload.update({
        "candidate_id": candidate_id, "slot": slot,
        "parent_candidate_id": parent_id,
        "mode": instruction_for(slot).mode,
        "evidence_candidate_ids": list(evidence_ids) if slot == 9 else [],
    })
    return DesignCard.from_mapping(payload)


def _failure_owner(exc: Exception) -> str:
    if isinstance(exc, DiscoveryError):
        return "provider_or_infrastructure"
    if isinstance(exc, DesignCardError):
        return "design_card"
    if isinstance(exc, SourceConstrainedDiscoveryError):
        return "compiler_contract"
    if isinstance(exc, JointResponseRuntimeError):
        return "candidate_execution"
    if isinstance(exc, OpenDiscoveryError):
        return "candidate_contract"
    return "workflow_runtime"


def run_source_constrained_discovery(
    *, task_id: str, output_root: str | Path, mode: str, device: str,
    config_path: str | Path, trajectory_seed: int,
    proposal_slots: int | None = None,
) -> Mapping[str, Any]:
    if mode not in {"smoke", "formal"}:
        raise SourceConstrainedDiscoveryError("mode must be smoke or formal")
    config = _config(config_path)
    root = project_path(output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise SourceConstrainedDiscoveryError("output root must be a new directory beneath runs/")
    max_slots = 9 if proposal_slots is None else int(proposal_slots)
    if max_slots not in range(1, 10):
        raise SourceConstrainedDiscoveryError("proposal slots must lie in 1..9")
    if mode == "formal" and max_slots != 9:
        raise SourceConstrainedDiscoveryError("formal source-constrained discovery requires all nine proposal slots")

    runtime_config = load_runtime_config(config["runtime_config"])
    arrays = load_fold13_arrays(task_id)
    settings = JointResponseSettings.from_config(runtime_config, device=device)
    if mode == "smoke":
        settings = replace(settings, max_epochs=3, early_stopping_patience=3)
    fixed_training = {
        "batch_size": int(settings.batch_size), "max_epochs": int(settings.max_epochs),
        "patience": int(settings.early_stopping_patience),
        "gradient_clip_norm": float(settings.gradient_clip_norm),
        "min_delta": float(settings.early_stopping_min_delta),
    }
    client, provider = _provider(config)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "source_constrained_config_snapshot.json", config)
    _write_json(root / "provider_receipt.json", provider)
    _write_json(root / "seed_registration.json", {
        "trajectory_seed": trajectory_seed,
        "candidate_seed_rule": "trajectory_seed*100+position",
    })
    started = time.perf_counter()

    def execute(candidate_id: str, candidate: CandidateState, position: int, destination: Path) -> Mapping[str, Any]:
        result = train_joint_candidate(
            candidate_id, arrays=arrays, config=runtime_config,
            settings=replace(settings, seed=registered_candidate_seed(trajectory_seed, position, default_seed=settings.seed)),
            checkpoint_path=destination / "selected_checkpoint.pt",
            candidate_override=candidate,
        )
        _write_json(destination / "execution.json", result)
        return result

    source_h0 = load_starting_candidate("h0", runtime_config)
    h0 = source_h0 if mode == "formal" else CandidateState(
        candidate_source=source_h0.candidate_source,
        candidate_metadata=source_h0.candidate_metadata,
        training_config=fixed_training,
    )
    records: list[dict[str, Any]] = []
    h0_root = root / "candidates" / "h0"
    h0_root.mkdir(parents=True)
    (h0_root / "candidate_source.py").write_text(h0.candidate_source, encoding="utf-8")
    _write_json(h0_root / "candidate_state.json", h0.to_dict())
    h0_result = execute("h0", h0, 1, h0_root)
    h0_feedback = _feedback("h0", h0_result)
    records.append({
        "candidate_id": "h0", "status": "complete",
        "candidate_hash": h0.candidate_hash, "hypothesis": "Registered h0",
        "edits": [], "feedback": h0_feedback, "parent_candidate_id": None,
        "compiler_choices": None, "parent_relative_delta": None,
    })
    incumbent, incumbent_record = h0, records[0]
    calls: list[Mapping[str, Any]] = []
    seen_hashes = {h0.candidate_hash}
    seen_designs: set[str] = set()
    choices_by_candidate: dict[str, Mapping[str, object]] = {}
    initial_fusions: set[str] = set()

    for slot in range(1, max_slots + 1):
        candidate_id = f"candidate_{slot:02d}"
        candidate_root = root / "candidates" / candidate_id
        candidate_root.mkdir(parents=True)
        instruction = instruction_for(slot)
        if slot <= 3:
            parent_record = records[0]
            parent_id = None
        elif instruction.parent_policy != "incumbent":
            matches = [item for item in records if item.get("candidate_id") == instruction.parent_policy and item.get("status") == "complete"]
            parent_record = matches[0] if matches else incumbent_record
            parent_id = str(parent_record["candidate_id"])
        else:
            parent_record = incumbent_record
            parent_id = str(parent_record["candidate_id"])
        parent_feedback = parent_record["feedback"]
        base_choices = None if slot <= 3 else choices_by_candidate.get(parent_id)
        frontier = _frontier(records)
        evidence = _trial_evidence(records)
        evidence_ids = tuple(dict.fromkeys(
            item["candidate_id"] for item in frontier.values()
            if item["candidate_id"] != "h0"
        ))
        failures: list[Mapping[str, Any]] = []
        for attempt in range(1, 4):
            repair = None if not failures else str(failures[-1]["message"])
            try:
                response = client.chat(
                    _prompt(
                        task_id=task_id, slot=slot, parent_id=parent_id,
                        incumbent_feedback=parent_feedback,
                        incumbent_choices=base_choices, frontier=frontier,
                        evidence=evidence, attempted=_attempted_choices(records),
                        repair=repair, initial_fusions=tuple(sorted(initial_fusions)),
                    ),
                    temperature=float(config["search"]["temperature"]),
                    max_tokens=int(config["search"]["max_output_tokens"]),
                )
                calls.append({
                    "slot": slot, "attempt": attempt, "purpose": "design_card",
                    "audit": dict(client.last_call_audit or {}),
                })
                card = _parse_card(
                    response, candidate_id=candidate_id, slot=slot,
                    parent_id=parent_id, evidence_ids=evidence_ids,
                )
                compiled = compile_card(
                    card,
                    condition_dim=int(arrays.task_spec.chemical_dim + arrays.task_spec.dose_dim),
                    target_dim=int(arrays.task_spec.target_dim),
                    base_choices=base_choices,
                )
                fusion = str(compiled.metadata["compiler_choices"]["fusion"])
                if slot <= 3 and fusion in initial_fusions:
                    raise SourceConstrainedDiscoveryError("initial hypotheses must use distinct fusion mechanisms")
                design_digest = digest({"compiler_choices": compiled.metadata["compiler_choices"]})
                if design_digest in seen_designs:
                    raise SourceConstrainedDiscoveryError("compiler generated a duplicate semantic design")
                source, metadata = materialize_candidate_source(card, base_choices=base_choices)
                candidate = CandidateState(
                    candidate_source=source, candidate_metadata=metadata,
                    training_config=fixed_training,
                )
                if candidate.candidate_hash in seen_hashes:
                    raise SourceConstrainedDiscoveryError("compiler generated a duplicate candidate")
                _write_json(candidate_root / "design_card.json", {
                    "card": asdict(card), "compiler_digest": compiled.digest,
                    "metadata": metadata,
                })
                _write_json(candidate_root / "compiled_contract.json", compiled.metadata)
                (candidate_root / "candidate_source.py").write_text(source, encoding="utf-8")
                _write_json(candidate_root / "candidate_state.json", candidate.to_dict())
                result = execute(candidate_id, candidate, slot + 1, candidate_root)
                candidate_feedback = _feedback(candidate_id, result)
                # A design only becomes prior evidence after a successful
                # physical fit.  Failed interface/execution attempts must not
                # poison deterministic duplicate detection during repair.
                seen_hashes.add(candidate.candidate_hash)
                seen_designs.add(design_digest)
                delta = _metric_delta(candidate_feedback, parent_feedback)
                record = {
                    "candidate_id": candidate_id, "status": "complete",
                    "candidate_hash": candidate.candidate_hash,
                    "parent_candidate_id": card.parent_candidate_id,
                    "hypothesis": card.hypothesis,
                    "edits": [asdict(edit) for edit in card.edits],
                    "compiled_design_digest": design_digest,
                    "compiler_choices": dict(compiled.metadata["compiler_choices"]),
                    "feedback": candidate_feedback,
                    "group_diagnostic": _block_summary(candidate_feedback),
                    "parent_relative_delta": delta,
                    "card_attempts": attempt, "card_failures": failures,
                }
                records.append(record)
                choices_by_candidate[candidate_id] = dict(compiled.metadata["compiler_choices"])
                if slot <= 3:
                    initial_fusions.add(fusion)
                child_key = (
                    float(candidate_feedback["fold3_global_pcc"]),
                    -float(candidate_feedback["fold3_mse"]),
                )
                incumbent_key = (
                    float(incumbent_record["feedback"]["fold3_global_pcc"]),
                    -float(incumbent_record["feedback"]["fold3_mse"]),
                )
                if child_key > incumbent_key:
                    incumbent, incumbent_record = candidate, record
                break
            except Exception as exc:
                failure = {
                    "attempt": attempt, "failure_owner": _failure_owner(exc),
                    "message": _safe_error(exc),
                }
                failures.append(failure)
                _write_json(candidate_root / f"card_failure_{attempt:02d}.json", failure)
        else:
            records.append({
                "candidate_id": candidate_id, "status": "failed",
                "parent_candidate_id": parent_id, "hypothesis": "", "edits": [],
                "feedback": None, "card_failures": failures,
            })

    scores = [
        Fold3CandidateScore(
            item["candidate_id"], float(item["feedback"]["fold3_global_pcc"]),
            float(item["feedback"]["fold3_mse"]),
            (float(item["feedback"]["fold3_cp_pcc"]), float(item["feedback"]["fold3_l1000_pcc"])),
        )
        for item in records if item["status"] == "complete"
    ]
    selected = select_endpoint(scores)
    result = {
        "schema_version": "cellscientist_source_constrained_grouped_discovery_run_v3",
        "status": "complete", "mode": mode, "task_id": task_id,
        "trajectory_seed": trajectory_seed, "provider_calls": len(calls),
        "fold4_or_fold5_used": False,
        "candidate_budget_including_h0": max_slots + 1,
        "records": records, "provider_calls_receipt": calls,
        "initial_fusions_completed": sorted(initial_fusions),
        "selected_candidate_id": selected.candidate_id,
        "final_evidence_memory": {
            "frontier": _frontier(records),
            "trials": _trial_evidence(records),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(root / "summary.json", result)
    (root / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    return result
