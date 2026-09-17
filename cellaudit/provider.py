"""Provider-agnostic, secret-safe discovery backend resolution.

The discovery controller is intentionally model-agnostic: it receives a
single preflight-selected backend for a whole campaign, records its public
identity, and never switches it mid-campaign.  Provider choice is therefore
an experimental runtime binding rather than a source of candidate sharing or
controller privilege.  Credentials and endpoint URLs remain process-local.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .local_config import LocalConfigError, configure_local_endpoint
from .schemas import digest


PROVIDER_RESOLVER_SCHEMA_VERSION = "cellscientist_provider_resolver"
CREDENTIAL_POLICY = "machine_local_credentials_only"
PROVIDER_PRIORITY = (
    "gemini",
    "deepseek_v4_flash",
    "qwen35_4b",
)


@dataclass(frozen=True)
class ProviderProfile:
    """One public model-panel identity eligible for a discovery campaign."""

    panel_key: str
    provider_family: str
    model_id: str
    local_endpoint_id: str
    supports_reasoning_effort: bool = False


_PROFILES = {
    "gemini": ProviderProfile(
        panel_key="gemini",
        provider_family="gemini",
        model_id="gemini-3-pro-preview",
        local_endpoint_id="gemini_compatible",
        supports_reasoning_effort=True,
    ),
    "deepseek_v4_flash": ProviderProfile(
        panel_key="deepseek_v4_flash",
        provider_family="deepseek",
        model_id="deepseek-v4-flash",
        local_endpoint_id="deepseek_v4_flash",
    ),
    "qwen35_4b": ProviderProfile(
        panel_key="qwen35_4b",
        provider_family="qwen",
        model_id="Qwen/Qwen3.5-4B",
        local_endpoint_id="qwen35_4b",
    ),
}
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FALLBACK_ENTRY_KEYS = frozenset(
    {
        "fallback",
        "fallback_model",
        "fallback_models",
        "fallback_panel_key",
        "fallback_model_panel_key",
        "fallback_provider",
    }
)


class ProviderResolverError(RuntimeError):
    """Raised without serializing a credential or endpoint value."""


def _plain_json(value: Any, *, name: str) -> Any:
    def detach(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): detach(member) for key, member in item.items()}
        if isinstance(item, (tuple, list)):
            return [detach(member) for member in item]
        return item

    try:
        return json.loads(json.dumps(detach(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ProviderResolverError(f"{name} must be finite JSON-compatible metadata") from exc


def _env_name(value: Any, *, name: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _ENV_KEY_RE.fullmatch(value):
        raise ProviderResolverError(f"{name} is not a valid environment-variable name")
    return value


def profile_for_panel_key(value: str) -> ProviderProfile:
    if not isinstance(value, str) or value not in _PROFILES:
        raise ProviderResolverError("current discovery selected provider is not a registered campaign backend")
    return _PROFILES[value]


def profile_for_provider_identity(*, provider_family: str, model_id: str) -> ProviderProfile:
    for profile in _PROFILES.values():
        if profile.provider_family == provider_family and profile.model_id == model_id:
            return profile
    raise ProviderResolverError("current discovery provider family/model identity is not registered")


def provider_lock_for_panel_key(panel_key: str) -> dict[str, Any]:
    """Construct the public no-mid-campaign-fallback lock for one backend."""

    profile = profile_for_panel_key(panel_key)
    forbidden = [
        family
        for family in ("gemini", "deepseek", "qwen")
        if family != profile.provider_family
    ]
    return {
        "model_panel_key": profile.panel_key,
        "model_id": profile.model_id,
        "local_endpoint_id": profile.local_endpoint_id,
        "credential_policy": CREDENTIAL_POLICY,
        "allow_fallback": False,
        "all_llm_conditions_use_lock": True,
        "forbidden_provider_families": forbidden,
    }


def validate_provider_lock(value: Mapping[str, Any]) -> ProviderProfile:
    """Validate a single selected provider lock; it never performs preflight I/O."""

    if not isinstance(value, Mapping):
        raise ProviderResolverError("current discovery provider lock must be an object")
    expected_fields = set(provider_lock_for_panel_key("gemini"))
    if set(value) != expected_fields:
        raise ProviderResolverError("current discovery provider lock fields do not match the registered contract")
    panel_key = value.get("model_panel_key")
    profile = profile_for_panel_key(panel_key)
    expected = provider_lock_for_panel_key(profile.panel_key)
    for field in (
        "model_panel_key",
        "model_id",
        "local_endpoint_id",
        "credential_policy",
        "allow_fallback",
        "all_llm_conditions_use_lock",
    ):
        if value.get(field) != expected[field]:
            raise ProviderResolverError("current discovery provider lock drifted from its selected backend")
    forbidden = value.get("forbidden_provider_families")
    if not isinstance(forbidden, list) or forbidden != expected["forbidden_provider_families"]:
        raise ProviderResolverError("current discovery provider lock must prohibit every other registered backend")
    return profile


@dataclass(frozen=True)
class ProviderClientParameters:
    """Private client construction values plus only public serializable metadata."""

    _base_url: str = field(repr=False)
    _api_key: str = field(repr=False)
    profile: ProviderProfile
    reasoning_effort: str | None
    stream: bool
    base_url_env: str
    api_key_env: str
    model_env: str | None
    model_panel_entry_hash: str
    provider_lock_hash: str

    def __post_init__(self) -> None:
        parsed = urlparse(self._base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ProviderResolverError("resolved provider endpoint is not a valid HTTP(S) endpoint")
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise ProviderResolverError("resolved provider credential is unavailable")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high"}:
            raise ProviderResolverError("resolved provider reasoning effort is invalid")
        if not self.profile.supports_reasoning_effort and self.reasoning_effort is not None:
            raise ProviderResolverError("selected provider must not declare unsupported reasoning effort")
        if not isinstance(self.stream, bool):
            raise ProviderResolverError("resolved provider stream setting must be boolean")
        _env_name(self.base_url_env, name="base_url_env")
        _env_name(self.api_key_env, name="api_key_env")
        _env_name(self.model_env, name="model_env", required=False)
        for name in ("model_panel_entry_hash", "provider_lock_hash"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ProviderResolverError(f"{name} must be a SHA-256 digest")

    @property
    def model(self) -> str:
        return self.profile.model_id

    @property
    def provider_id(self) -> str:
        return self.profile.local_endpoint_id

    @property
    def provider_family(self) -> str:
        return self.profile.provider_family

    @property
    def resolved_base_url_hash(self) -> str:
        return digest(self._base_url.rstrip("/"))

    @property
    def candidate_reasoning_effort(self) -> str | None:
        return "minimal" if self.profile.supports_reasoning_effort else None

    def client_kwargs(self) -> dict[str, Any]:
        return {
            "base_url": self._base_url.rstrip("/"),
            "api_key": self._api_key,
            "model": self.model,
            "provider_id": self.provider_id,
            "reasoning_effort": self.reasoning_effort,
            "stream": self.stream,
        }

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PROVIDER_RESOLVER_SCHEMA_VERSION,
            "model_panel_key": self.profile.panel_key,
            "model": self.model,
            "provider_family": self.provider_family,
            "provider_id": self.provider_id,
            "reasoning_effort": self.reasoning_effort,
            "candidate_reasoning_effort": self.candidate_reasoning_effort,
            "stream": self.stream,
            "allow_fallback": False,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "model_env": self.model_env,
            "credential_present": True,
            "resolved_base_url_hash": self.resolved_base_url_hash,
            "model_panel_entry_hash": self.model_panel_entry_hash,
            "provider_lock_hash": self.provider_lock_hash,
            "provider_call_performed": False,
        }


def resolve_provider_client_parameters(
    *,
    model_panel: Mapping[str, Any],
    local_endpoints: str | Path | None,
    provider_lock: Mapping[str, Any],
) -> ProviderClientParameters:
    """Resolve one selected backend without printing or serializing secrets."""

    profile = validate_provider_lock(provider_lock)
    if not isinstance(model_panel, Mapping):
        raise ProviderResolverError("model panel must be an object")
    raw_entry = model_panel.get(profile.panel_key)
    if not isinstance(raw_entry, Mapping):
        raise ProviderResolverError("selected provider model-panel entry is unavailable")
    entry = _plain_json(raw_entry, name="selected provider model-panel entry")
    if entry.get("kind") not in {"openai_compatible", "local_or_openai_compatible"}:
        raise ProviderResolverError("selected provider entry is not OpenAI-compatible")
    if entry.get("local_endpoint_id") != profile.local_endpoint_id:
        raise ProviderResolverError("selected provider model-panel endpoint binding drifted")
    for key in _FALLBACK_ENTRY_KEYS:
        if key in entry and entry[key] not in (None, False, "", [], {}):
            raise ProviderResolverError("selected provider entry must not declare a fallback")
    try:
        configure_local_endpoint(entry, path=local_endpoints)
    except (LocalConfigError, OSError, ValueError):
        raise ProviderResolverError("local selected-provider configuration could not be resolved") from None
    base_url_env = _env_name(entry.get("base_url_env"), name="base_url_env")
    api_key_env = _env_name(entry.get("api_key_env"), name="api_key_env")
    model_env = _env_name(entry.get("model_env"), name="model_env", required=False)
    base_url = os.environ.get(base_url_env or "", "")
    api_key = os.environ.get(api_key_env or "", "")
    resolved_model = os.environ.get(model_env, entry.get("model", "")) if model_env else entry.get("model", "")
    if resolved_model != profile.model_id:
        raise ProviderResolverError("local selected-provider endpoint did not resolve the registered model identity")
    if not isinstance(base_url, str) or not base_url.strip() or not isinstance(api_key, str) or not api_key.strip():
        raise ProviderResolverError("local selected-provider endpoint lacks a required non-secret binding")
    reasoning_effort = entry.get("reasoning_effort")
    return ProviderClientParameters(
        _base_url=base_url,
        _api_key=api_key,
        profile=profile,
        reasoning_effort=reasoning_effort,
        stream=bool(entry.get("stream", False)),
        base_url_env=base_url_env or "",
        api_key_env=api_key_env or "",
        model_env=model_env,
        model_panel_entry_hash=digest(entry),
        provider_lock_hash=digest(_plain_json(provider_lock, name="provider_lock")),
    )


__all__ = [
    "CREDENTIAL_POLICY",
    "PROVIDER_PRIORITY",
    "PROVIDER_RESOLVER_SCHEMA_VERSION",
    "ProviderClientParameters",
    "ProviderProfile",
    "ProviderResolverError",
    "profile_for_panel_key",
    "profile_for_provider_identity",
    "provider_lock_for_panel_key",
    "resolve_provider_client_parameters",
    "validate_provider_lock",
]
