"""Shared, secret-safe runtime utilities for CellAudit discovery."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from .provider import provider_lock_for_panel_key, resolve_provider_client_parameters
from .provider_client import OpenAICompatibleClient, load_json
from .open_discovery import CandidateState, OpenDiscoveryError
from .schemas import canonical_json
from .joint_response_runtime import project_path


_MAX_SAFE_ERROR = 360
_MAX_HYPOTHESIS = 1200


class CandidateResponseError(RuntimeError):
    """Raised when a provider response violates the candidate contract."""


def _provider(config: Mapping[str, Any]) -> tuple[OpenAICompatibleClient, Mapping[str, Any]]:
    """Construct the campaign-locked client without exposing endpoint secrets."""

    registered = config["provider"]
    panel = load_json(project_path(str(registered["model_panel"])))
    lock = provider_lock_for_panel_key(str(registered["panel_key"]))
    params = resolve_provider_client_parameters(
        model_panel=panel,
        local_endpoints=None,
        provider_lock=lock,
    )
    client = OpenAICompatibleClient(
        timeout_seconds=180,
        max_http_attempts=3,
        **params.client_kwargs(),
    )
    return client, {**params.safe_metadata(), "provider_lock": lock}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(member) for key, member in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(member) for member in value]
    return value


def _safe_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?:[A-Za-z]:)?/[A-Za-z0-9_./-]+", "<path>", text)
    text = re.sub(r"\b(?:sk|api)[-_A-Za-z0-9]{12,}\b", "<secret>", text, flags=re.IGNORECASE)
    return text[:_MAX_SAFE_ERROR] or type(exc).__name__


def _strip_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text.strip()


def _parse_candidate_response(
    text: str,
    *,
    parent: CandidateState,
    fixed_training_config: Mapping[str, Any],
) -> tuple[CandidateState, str]:
    """Parse one source proposal while leaving executable validation local."""

    try:
        payload = json.loads(_strip_fence(text))
    except json.JSONDecodeError as exc:
        raise CandidateResponseError("provider response is not a standalone JSON candidate") from exc
    if not isinstance(payload, Mapping):
        raise CandidateResponseError("provider response must be a JSON object")
    if "candidate" in payload:
        payload = payload["candidate"]
    expected = {"candidate_source", "candidate_metadata", "training_config", "hypothesis"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise CandidateResponseError(
            "provider response fields must be exactly candidate_source, candidate_metadata, training_config, hypothesis"
        )
    hypothesis = payload["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > _MAX_HYPOTHESIS:
        raise CandidateResponseError("provider hypothesis must be a compact non-empty string")
    if _plain(payload["training_config"]) != _plain(fixed_training_config):
        raise CandidateResponseError("provider attempted to alter the fixed physical training configuration")
    try:
        candidate = CandidateState(
            candidate_source=str(payload["candidate_source"]),
            candidate_metadata=payload["candidate_metadata"],
            training_config=fixed_training_config,
            parent_candidate_hash=parent.candidate_hash,
        )
    except (OpenDiscoveryError, TypeError, ValueError) as exc:
        raise CandidateResponseError(f"candidate source/metadata contract rejected: {_safe_error(exc)}") from exc
    return candidate, hypothesis.strip()


def _feedback(candidate_id: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = result["selection_metrics"]
    feedback: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_hash": result["implementation"]["candidate_hash"],
        "selected_epoch": int(result["selected_epoch"]),
        "completed_epochs": int(result["completed_epochs"]),
        "stopped_early": bool(result["stopped_early"]),
        "parameter_count": int(result["parameter_count"]),
        "training_wall_seconds": round(float(result["runtime_seconds"]), 5),
        "gpu_wall_seconds": round(float(result["runtime_seconds"]), 5),
        "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
    }
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
            feedback[f"fold3_{metric_name}"] = round(float(metric_value), 8)
    return feedback


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _usage_totals(calls: list[Mapping[str, Any]]) -> Mapping[str, int | bool | None]:
    prompt = completion = total = 0
    complete = True
    for call in calls:
        usage = call.get("usage")
        if not isinstance(usage, Mapping):
            complete = False
            continue
        try:
            prompt += int(usage.get("prompt_tokens", 0))
            completion += int(usage.get("completion_tokens", 0))
            total += int(usage.get("total_tokens", 0))
        except (TypeError, ValueError):
            complete = False
    return {
        "prompt_tokens": prompt if complete else None,
        "completion_tokens": completion if complete else None,
        "total_tokens": total if complete else None,
        "usage_complete": complete,
    }


def _call_identity(audit: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        audit.get("slot"), audit.get("purpose"), audit.get("attempt"),
        audit.get("request_payload_sha256"), audit.get("response_payload_sha256"),
        audit.get("status"), audit.get("actual_http_attempts"),
    )


def _record_provider_call(
    client: OpenAICompatibleClient,
    calls: list[Mapping[str, Any]],
    slot_calls: list[Mapping[str, Any]],
    *,
    slot: int,
    purpose: str,
    attempt: int,
    policy: str | None = None,
) -> None:
    audit = dict(client.last_call_audit or {})
    if not audit:
        return
    audit.update({"slot": slot, "purpose": purpose, "attempt": attempt})
    if policy is not None:
        audit["policy"] = policy
    identity = _call_identity(audit)
    if any(_call_identity(existing) == identity for existing in slot_calls):
        return
    calls.append(audit)
    slot_calls.append(audit)


def _provider_accounting(calls: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    by_purpose: dict[str, list[Mapping[str, Any]]] = {}
    for call in calls:
        by_purpose.setdefault(str(call.get("purpose", "unspecified")), []).append(call)

    def summarize(group: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        usage = _usage_totals(group)
        return {
            "logical_calls": len(group),
            "successful_calls": sum(call.get("status") == "success" for call in group),
            "failed_calls": sum(call.get("status") == "failed" for call in group),
            "http_attempts": sum(int(call.get("actual_http_attempts", 0) or 0) for call in group),
            "llm_elapsed_seconds": round(sum(float(call.get("elapsed_seconds", 0.0) or 0.0) for call in group), 5),
            **usage,
        }

    return {**summarize(calls), "by_purpose": {name: summarize(group) for name, group in sorted(by_purpose.items())}}


def _runtime_environment() -> Mapping[str, Any]:
    result: dict[str, Any] = {"python_version": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        result.update({
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        })
        if torch.cuda.is_available():
            result["visible_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_metadata_error"] = type(exc).__name__
    return result
