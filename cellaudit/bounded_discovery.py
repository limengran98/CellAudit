"""Bounded CellScientist discovery under the shared CPG protocol.

CellScientist receives only protected task dimensions and Fold-3 diagnostics,
then iteratively revises the current incumbent with a compact record of prior
modeling hypotheses.  The executor retains ownership of raw data, folds,
preprocessing, training, checkpoint selection, and all metrics.  Fold 4 and
Fold 5 are never materialized by this module.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .runtime_utils import (
    CandidateResponseError,
    _feedback,
    _parse_candidate_response,
    _plain,
    _provider_accounting,
    _record_provider_call,
    _runtime_environment,
    _safe_error,
    _usage_totals,
    _write_json,
)
from .provider_client import DiscoveryError, OpenAICompatibleClient, load_json
from .open_discovery import CandidateState, OpenDiscoveryError
from .schemas import canonical_json
from .joint_response_runtime import (
    JointResponseSettings,
    JointResponseRuntimeError,
    load_runtime_config,
    load_starting_candidate,
    load_fold13_arrays,
    project_path,
    registered_candidate_seed,
    train_joint_candidate,
)
from .provider import provider_lock_for_panel_key, resolve_provider_client_parameters


DISCOVERY_CONFIG_SCHEMAS = frozenset((
    "cellscientist_bounded_discovery_v1",
    "cellscientist_bounded_discovery_v2",
    "cellscientist_open_discovery",
))
DISCOVERY_RUN_SCHEMA = "cellscientist_bounded_discovery_run_v2"
_ALLOWED_ACTIONS = (
    "model_architecture",
    "fusion",
    "response_readout",
    "loss",
    "optimizer",
    "scheduler",
)


class BoundedDiscoveryError(RuntimeError):
    """Raised when the bounded CellScientist discovery contract is violated."""


def _read_config(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(project_path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundedDiscoveryError("cannot read bounded-discovery configuration") from exc
    expected = {
        "schema_version", "runtime_config", "provider", "search", "controller",
        "candidate_contract", "artifact_policy",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("schema_version") not in DISCOVERY_CONFIG_SCHEMAS:
        raise BoundedDiscoveryError("bounded-discovery configuration schema is invalid")
    if value["provider"] != {
        "model_panel": "configs/model_panel.json",
        "panel_key": "deepseek_v4_flash",
        "allow_fallback": False,
    }:
        raise BoundedDiscoveryError("bounded discovery requires the registered DeepSeek V4 Flash no-fallback lock")
    search = value["search"]
    if not isinstance(search, Mapping) or int(search.get("max_evaluated_candidates_including_h0", 0)) != 10:
        raise BoundedDiscoveryError("the formal discovery budget must be exactly ten candidates including h0")
    if int(search.get("max_repairs_per_candidate", -1)) < 0 or int(search.get("max_repairs_per_candidate", 9)) > 3:
        raise BoundedDiscoveryError("the local repair budget must lie in [0, 3]")
    controller = value["controller"]
    if not isinstance(controller, Mapping) or tuple(controller.get("allowed_revision_scopes", ())) != _ALLOWED_ACTIONS:
        raise BoundedDiscoveryError("the registered modeling revision scope drifted")
    if not isinstance(controller.get("history_window"), int) or int(controller["history_window"]) < 1:
        raise BoundedDiscoveryError("bounded-discovery history window is invalid")
    agenda = controller.get("design_agenda", ())
    if agenda and (
        not isinstance(agenda, (list, tuple))
        or len(agenda) < 1
        or any(not isinstance(item, str) or not item.strip() or len(item) > 1200 for item in agenda)
    ):
        raise BoundedDiscoveryError("bounded-discovery design agenda is invalid")
    canonical_json(value)
    return value


def _provider_client(config: Mapping[str, Any]) -> tuple[OpenAICompatibleClient, Mapping[str, Any]]:
    provider = config["provider"]
    panel = load_json(project_path(str(provider["model_panel"])))
    lock = provider_lock_for_panel_key(str(provider["panel_key"]))
    params = resolve_provider_client_parameters(model_panel=panel, local_endpoints=None, provider_lock=lock)
    return (
        OpenAICompatibleClient(timeout_seconds=180, max_http_attempts=3, **params.client_kwargs()),
        {**params.safe_metadata(), "provider_lock": lock},
    )


def _record_call(
    client: OpenAICompatibleClient,
    calls: list[Mapping[str, Any]],
    slot_calls: list[Mapping[str, Any]],
    *,
    slot: int,
    purpose: str,
    attempt: int,
) -> None:
    _record_provider_call(
        client,
        calls,
        slot_calls,
        slot=slot,
        purpose=purpose,
        attempt=attempt,
        policy="cellscientist_diagnostic_revision",
    )


def _compact_memory(records: list[Mapping[str, Any]], *, window: int) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for record in records[-window:]:
        feedback = record.get("feedback")
        if not isinstance(feedback, Mapping):
            continue
        result.append({
            "candidate_id": record.get("candidate_id"),
            "status": record.get("status"),
            "hypothesis": str(record.get("hypothesis", ""))[:500],
            "fold3_global_pcc": feedback.get("fold3_global_pcc"),
            "fold3_mse": feedback.get("fold3_mse"),
            "fold3_cp_pcc": feedback.get("fold3_cp_pcc"),
            "fold3_l1000_pcc": feedback.get("fold3_l1000_pcc"),
        })
    return result


def _proposal_prompt(
    *,
    task_id: str,
    dimensions: Mapping[str, int],
    parent: CandidateState,
    feedback: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    fixed_training_config: Mapping[str, Any],
    step_agenda: str | None,
    repair_error: str | None,
) -> list[dict[str, str]]:
    system = """You are the CellScientist bounded discovery policy for a multimodal perturbation-response predictor. Return exactly one standalone JSON object, without Markdown. You may propose one coherent, source-local modeling revision based on the current model's Fold-3 diagnostics and the compact history. The local runner, not you, owns raw data, preprocessing, fold identities, targets, evaluation, checkpoint selection, and stopping. Do not access files, networks, subprocesses, data loading, private metrics, Fold 4, or Fold 5. Candidate source may import only torch, math, typing, dataclasses or collections. The model forward signature must be model(pre, chemical, dose) and return the complete joint tensor [batch, target_dim]. Required callables: candidate_metadata, build_model, compute_loss, build_optimizer; build_scheduler is optional. If a scheduler is defined, the optional callback has the exact signature step_scheduler(scheduler, state), where state contains epoch, max_epochs, and fold3_global_pcc; omit step_scheduler entirely if the scheduler's normal scheduler.step() is sufficient. Metadata must describe exported symbols using semantic roots input, perturbation, response, objective, optimization, nuisance, reliability. Preserve the exact training configuration."""
    repair = "" if repair_error is None else "\nThe previous local attempt was rejected. Correct only this stated local issue: " + repair_error
    agenda = "" if step_agenda is None else (
        "\nPre-registered exploration agenda for this slot: " + step_agenda
        + " Register this exploration direction as a testable hypothesis and implement one complete, coherent model with a substantive modeling change."
    )
    user = (
        f"Task: {task_id}\nDimensions: {canonical_json(dimensions)}\n"
        f"Current incumbent Fold-3 diagnostics: {canonical_json(feedback)}\n"
        f"Compact prior evaluated history: {canonical_json(history)}\n"
        f"Fixed training configuration: {canonical_json(_plain(fixed_training_config))}\n"
        "Allowed revision scopes: model_architecture, fusion, response_readout, loss, optimizer, scheduler. "
        "Choose one modeling hypothesis that addresses a diagnostic or complements earlier attempts; produce a source-distinct candidate.\n"
        "Current incumbent source:\n--- SOURCE ---\n" + parent.candidate_source + "\n--- END SOURCE ---\n"
        "Response schema: {\"candidate_source\": string, \"candidate_metadata\": object, \"training_config\": object, \"hypothesis\": string}."
        + agenda
        + repair
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_bounded_discovery(
    *,
    task_id: str,
    output_root: str | Path,
    mode: str,
    device: str,
    config_path: str | Path = "configs/open_discovery.json",
    trajectory_seed: int | None = None,
) -> Mapping[str, Any]:
    """Run one CellScientist discovery trajectory with exactly ten fit slots."""

    if mode not in {"smoke", "formal"}:
        raise BoundedDiscoveryError("mode must be smoke or formal")
    config = _read_config(config_path)
    runtime_config = load_runtime_config(config["runtime_config"])
    if task_id not in runtime_config["tasks"]:
        raise BoundedDiscoveryError("task is not registered by the shared CPG protocol")
    root = project_path(output_root)
    if root.exists() or not root.is_relative_to(project_path("runs")):
        raise BoundedDiscoveryError("output root must be a new path beneath runs/")
    arrays = load_fold13_arrays(task_id)
    settings = JointResponseSettings.from_config(runtime_config, device=device)
    if mode == "smoke":
        settings = replace(settings, max_epochs=3, early_stopping_patience=3)
    fixed_training_config = {
        "batch_size": int(settings.batch_size), "max_epochs": int(settings.max_epochs),
        "patience": int(settings.early_stopping_patience), "gradient_clip_norm": float(settings.gradient_clip_norm),
        "min_delta": float(settings.early_stopping_min_delta),
    }

    def settings_for_position(position: int) -> JointResponseSettings:
        return replace(
            settings,
            seed=registered_candidate_seed(
                trajectory_seed,
                position,
                default_seed=settings.seed,
            ),
        )

    seed_registration = {
        "trajectory_seed": trajectory_seed,
        "candidate_seed_rule": "trajectory_seed * 100 + candidate_position; h0 is position 1",
        "candidate_seeds": {
            "h0": settings_for_position(1).seed,
            **{
                f"cellscientist_{slot:02d}": settings_for_position(slot + 1).seed
                for slot in range(1, 10)
            },
        },
    }
    source_h0 = load_starting_candidate("h0", runtime_config)
    h0 = source_h0 if mode == "formal" else CandidateState(
        candidate_source=source_h0.candidate_source,
        candidate_metadata=source_h0.candidate_metadata,
        training_config=fixed_training_config,
    )
    if _plain(h0.training_config) != fixed_training_config:
        raise BoundedDiscoveryError("registered h0 configuration does not match common physical budget")
    client, provider_metadata = _provider_client(config)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "discovery_config_snapshot.json", config)
    _write_json(root / "runtime_config_snapshot.json", runtime_config)
    _write_json(root / "provider_receipt.json", provider_metadata)

    calls: list[Mapping[str, Any]] = []
    records: list[dict[str, Any]] = []
    best_candidate = h0
    best_record: Mapping[str, Any] | None = None
    seen_hashes = {h0.candidate_hash}
    started = time.perf_counter()

    def execute(
        candidate_id: str,
        candidate: CandidateState,
        hypothesis: str,
        destination: Path,
        candidate_position: int,
    ) -> Mapping[str, Any]:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "candidate_source.py").write_text(candidate.candidate_source, encoding="utf-8")
        _write_json(destination / "candidate_state.json", candidate.to_dict())
        _write_json(destination / "hypothesis.json", {"hypothesis": hypothesis})
        result = train_joint_candidate(candidate_id, arrays=arrays, config=runtime_config, settings=settings_for_position(candidate_position),
                                     checkpoint_path=destination / "selected_checkpoint.pt", candidate_override=candidate)
        _write_json(destination / "execution.json", result)
        return result

    try:
        h0_result = execute("h0", h0, "Registered shared h0; no discovery revision.", root / "candidates" / "h0", 1)
        h0_feedback = _feedback("h0", h0_result)
        best_record = {
            "candidate_id": "h0", "status": "complete", "candidate_hash": h0.candidate_hash,
            "parent_candidate_hash": None, "hypothesis": "Registered shared h0; no discovery revision.",
            "feedback": h0_feedback, "provider_calls": [],
        }
        records.append(dict(best_record))
    except Exception as exc:
        raise BoundedDiscoveryError(f"shared h0 failed before discovery: {_safe_error(exc)}") from exc

    for slot in range(1, 10):
        candidate_id = f"cellscientist_{slot:02d}"
        candidate_root = root / "candidates" / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=False)
        slot_calls: list[Mapping[str, Any]] = []
        parent = best_candidate
        feedback = dict(best_record["feedback"]) if best_record is not None else h0_feedback
        history = _compact_memory(records, window=int(config["controller"]["history_window"]))
        agenda_items = tuple(config["controller"].get("design_agenda", ()))
        step_agenda = agenda_items[(slot - 1) % len(agenda_items)] if agenda_items else None
        _write_json(candidate_root / "decision_context.json", {
            "parent_candidate_hash": parent.candidate_hash,
            "incumbent_feedback": feedback,
            "compact_history": history,
            "step_agenda": step_agenda,
        })
        failures: list[Mapping[str, Any]] = []
        repair_error: str | None = None
        completed = False
        for attempt in range(int(config["search"]["max_repairs_per_candidate"]) + 1):
            attempt_number = attempt + 1
            attempt_root = candidate_root / "attempts" / f"attempt_{attempt_number:02d}"
            attempt_root.mkdir(parents=True, exist_ok=False)
            candidate: CandidateState | None = None
            hypothesis = ""
            try:
                response = client.chat(
                    _proposal_prompt(task_id=task_id, dimensions=arrays.task_spec.to_dict(), parent=parent,
                                     feedback=feedback, history=history, fixed_training_config=fixed_training_config,
                                     step_agenda=step_agenda, repair_error=repair_error),
                    temperature=float(config["search"]["temperature"]),
                    max_tokens=int(config["search"]["max_output_tokens"]),
                )
                _record_call(client, calls, slot_calls, slot=slot, purpose="candidate" if attempt == 0 else "candidate_repair", attempt=attempt_number)
                candidate, hypothesis = _parse_candidate_response(response, parent=parent, fixed_training_config=fixed_training_config)
                if candidate.candidate_hash in seen_hashes:
                    raise BoundedDiscoveryError("candidate duplicates an already proposed source/configuration")
                seen_hashes.add(candidate.candidate_hash)
                (attempt_root / "candidate_source.py").write_text(candidate.candidate_source, encoding="utf-8")
                _write_json(attempt_root / "candidate_state.json", candidate.to_dict())
                _write_json(attempt_root / "hypothesis.json", {"hypothesis": hypothesis})
            except (CandidateResponseError, DiscoveryError, BoundedDiscoveryError, OpenDiscoveryError, JointResponseRuntimeError) as exc:
                _record_call(client, calls, slot_calls, slot=slot, purpose="candidate" if attempt == 0 else "candidate_repair", attempt=attempt_number)
                repair_error = _safe_error(exc)
                failure = {
                    "stage": "proposal_or_contract",
                    "failure_owner": "provider" if isinstance(exc, DiscoveryError) else "candidate_contract",
                    "attempt": attempt_number,
                    "failure_type": type(exc).__name__,
                    "failure_message": repair_error,
                }
                failures.append(failure)
                _write_json(attempt_root / "failure.json", failure)
                continue
            try:
                execution_started = time.perf_counter()
                result = train_joint_candidate(candidate_id, arrays=arrays, config=runtime_config, settings=settings_for_position(slot + 1),
                                             checkpoint_path=attempt_root / "selected_checkpoint.pt", candidate_override=candidate)
                _write_json(attempt_root / "execution.json", result)
                candidate_feedback = _feedback(candidate_id, result)
                (candidate_root / "candidate_source.py").write_text(candidate.candidate_source, encoding="utf-8")
                _write_json(candidate_root / "candidate_state.json", candidate.to_dict())
                _write_json(candidate_root / "hypothesis.json", {"hypothesis": hypothesis})
                _write_json(candidate_root / "execution.json", result)
                record = {
                    "candidate_id": candidate_id, "status": "complete", "parent_candidate_hash": parent.candidate_hash,
                    "candidate_hash": candidate.candidate_hash, "hypothesis": hypothesis, "feedback": candidate_feedback,
                    "provider_calls": slot_calls, "repair_attempts_used": attempt, "repair_failures": failures,
                    "selected_attempt": attempt_number,
                }
                records.append(record)
                if float(candidate_feedback["fold3_global_pcc"]) > float(best_record["feedback"]["fold3_global_pcc"]):
                    best_candidate, best_record = candidate, record
                completed = True
                break
            except Exception as exc:
                repair_error = _safe_error(exc)
                failure = {
                    "stage": "execution", "failure_owner": "candidate_execution_or_runtime", "attempt": attempt_number, "failure_type": type(exc).__name__,
                    "failure_message": repair_error, "candidate_hash": candidate.candidate_hash, "hypothesis": hypothesis,
                    "execution_wall_seconds": round(time.perf_counter() - execution_started, 5),
                }
                failures.append(failure)
                _write_json(attempt_root / "failure.json", failure)
        if not completed:
            status = "execution_failed" if any(item["stage"] == "execution" for item in failures) else "proposal_failed"
            record = {
                "candidate_id": candidate_id, "status": status, "parent_candidate_hash": parent.candidate_hash,
                "failure_log": failures, "provider_calls": slot_calls,
                "repair_attempts_used": int(config["search"]["max_repairs_per_candidate"]),
            }
            records.append(record)
            _write_json(candidate_root / "failure.json", record)

    if best_record is None:
        raise BoundedDiscoveryError("discovery produced no h0 selection record")
    completed_gpu_wall_seconds = round(
        sum(float(record.get("feedback", {}).get("gpu_wall_seconds", 0.0)) for record in records), 5
    )
    failed_execution_wall_seconds = round(
        sum(
            float(failure.get("execution_wall_seconds", 0.0))
            for record in records
            for failure in record.get("repair_failures", record.get("failure_log", []))
            if isinstance(failure, Mapping) and failure.get("stage") == "execution"
        ),
        5,
    )
    result = {
        "schema_version": DISCOVERY_RUN_SCHEMA, "status": "complete", "policy": "cellscientist_diagnostic_revision",
        "policy_specification": config["controller"], "task_id": task_id, "mode": mode,
        "data_fingerprint": arrays.data_fingerprint, "data_access": dict(arrays.metadata),
        "training_settings": settings.to_dict(), "training_seed_registration": seed_registration,
        "fixed_training_config": fixed_training_config,
        "candidate_budget_including_h0": 10, "selection_metric": "fold3_global_pcc_strict_improvement",
        "selected_candidate_id": str(best_record["candidate_id"]),
        "selected_candidate": best_record, "records": records, "provider": provider_metadata,
        "provider_calls": calls, "provider_usage": _usage_totals(calls),
        "provider_accounting": _provider_accounting(calls),
        "gpu_wall_seconds": completed_gpu_wall_seconds,
        "training_wall_seconds": completed_gpu_wall_seconds,
        "failed_execution_wall_seconds": failed_execution_wall_seconds,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 5), "fold4_or_fold5_used": False,
        "per_target_training": False, "prohibited_actions": list(runtime_config["prohibited_actions"]),
        "runtime_environment": _runtime_environment(),
    }
    _write_json(root / "summary.json", result)
    _write_json(root / "selected_candidate_receipt.json", best_record)
    return result


__all__ = ["BoundedDiscoveryError", "run_bounded_discovery"]
