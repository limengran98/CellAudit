"""Open exploration client with strict, replayable proposal artifacts.

Discovery is deliberately allowed to be model-dependent.  Its outputs become
scientific inputs only after compilation; the deterministic audit never calls
this module.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping
from urllib import error, request
from urllib.parse import urlparse

from .local_config import configure_local_endpoint


class DiscoveryError(RuntimeError):
    pass


def load_json(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DiscoveryError(f"Expected JSON object in {path}")
    return payload


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def parse_candidate_response(text: str) -> Mapping[str, Any]:
    """Parse only a JSON proposal; prose is rejected rather than guessed."""
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise DiscoveryError("LLM response was not a standalone JSON object") from exc
    if not isinstance(payload, Mapping):
        raise DiscoveryError("LLM response must be a JSON object")
    if "candidate" in payload:
        payload = payload["candidate"]
    if not isinstance(payload, Mapping):
        raise DiscoveryError("candidate field must be an object")
    if "operator_spec" not in payload:
        raise DiscoveryError("candidate response is missing operator_spec")
    if not isinstance(payload["operator_spec"], Mapping):
        raise DiscoveryError("operator_spec must be an object")
    return payload


def _stream_text(value: Any) -> tuple[str, bool]:
    """Extract text from common OpenAI-compatible streaming content shapes."""
    if isinstance(value, str):
        return value, True
    if isinstance(value, list):
        parts: list[str] = []
        found = False
        for item in value:
            text, present = _stream_text(item)
            if present:
                parts.append(text)
                found = True
        return "".join(parts), found
    if isinstance(value, Mapping):
        for key in ("text", "content", "value"):
            if key in value:
                return _stream_text(value[key])
    return "", False


def _needs_socks_compatible_transport(endpoint: str) -> bool:
    """Return whether a remote endpoint needs curl for an inherited SOCKS proxy.

    ``urllib`` supports HTTP proxies but not the ``socks5h`` proxy convention
    commonly installed on the GPU host.  The check is deliberately limited to
    non-loopback endpoints, so local OpenAI-compatible Qwen servers retain the
    normal Python transport.
    """

    host = (urlparse(endpoint).hostname or "").lower()
    if host in {"", "localhost", "127.0.0.1", "::1"}:
        return False
    # Private RFC1918 endpoints commonly bypass workstation-wide proxies.
    # Keep those endpoints on the direct Python transport while retaining the
    # curl compatibility path for public gateways reached through SOCKS.
    try:
        if ipaddress.ip_address(host).is_private:
            return False
    except ValueError:
        pass
    proxy_values = (
        os.environ.get(name, "").strip().lower()
        for name in (
            "ALL_PROXY",
            "all_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
        )
    )
    return any(value.startswith(("socks4://", "socks5://", "socks5h://")) for value in proxy_values)


def _http_proxy_override(endpoint: str) -> str | None:
    """Prefer an explicit HTTP(S) proxy over a stale lower-case SOCKS value.

    Some GPU launch environments expose an active HTTP CONNECT listener via
    upper-case ``HTTPS_PROXY`` while also inheriting obsolete lower-case SOCKS
    variables.  Curl gives the lower-case value precedence, which turns an
    otherwise valid remote-provider call into a local connection refusal.  We
    use an explicit HTTP(S) proxy only for this conflicting case; normal proxy
    environments retain their ordinary behavior.
    """

    scheme = (urlparse(endpoint).scheme or "https").lower()
    lower_socks = any(
        os.environ.get(name, "").strip().lower().startswith(
            ("socks4://", "socks5://", "socks5h://")
        )
        for name in ("all_proxy", "https_proxy", "http_proxy")
    )
    if not lower_socks:
        return None
    names = (
        ("HTTPS_PROXY", "HTTP_PROXY")
        if scheme == "https"
        else ("HTTP_PROXY", "HTTPS_PROXY")
    )
    for name in names:
        value = os.environ.get(name, "").strip()
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return value
    return None


class OpenAICompatibleClient:
    """Minimal client so the project does not depend on vendor SDKs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        provider_id: str | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
        max_http_attempts: int = 3,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise DiscoveryError("OpenAI-compatible base URL must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.provider_id = (
            provider_id
            if provider_id is not None
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", provider_id)
            else None
        )
        if reasoning_effort is not None and reasoning_effort not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise DiscoveryError(
                "reasoning_effort must be one of minimal, low, medium, or high"
            )
        self.reasoning_effort = reasoning_effort
        if not isinstance(stream, bool):
            raise DiscoveryError("stream must be a boolean")
        self.stream = stream
        if not isinstance(max_http_attempts, int) or isinstance(max_http_attempts, bool):
            raise DiscoveryError("max_http_attempts must be an integer")
        if max_http_attempts < 1 or max_http_attempts > 3:
            raise DiscoveryError("max_http_attempts must lie between 1 and 3")
        self.max_http_attempts = max_http_attempts
        # This mapping deliberately contains no URL, key, prompt, request body,
        # response body, or provider error message.  It is replaced atomically
        # after every call, including a call that exhausts all retries.
        self.last_call_audit: Mapping[str, Any] | None = None

    @classmethod
    def from_panel_entry(cls, entry: Mapping[str, Any], *, local_endpoints: str | Path | None = None) -> "OpenAICompatibleClient":
        configure_local_endpoint(entry, path=local_endpoints)
        base_var = str(entry.get("base_url_env", ""))
        key_var = str(entry.get("api_key_env", ""))
        model = str(entry.get("model_env") and os.environ.get(str(entry["model_env"])) or entry.get("model", ""))
        base_url = os.environ.get(base_var, "")
        api_key = os.environ.get(key_var, "")
        missing = [name for name, value in ((base_var, base_url), (key_var, api_key), ("model", model)) if not value]
        if missing:
            raise DiscoveryError("Missing model configuration: " + ", ".join(missing))
        stream = entry.get("stream", False)
        if not isinstance(stream, bool):
            raise DiscoveryError("model panel stream must be a boolean")
        endpoint_id = entry.get("local_endpoint_id")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider_id=str(endpoint_id) if endpoint_id else None,
            reasoning_effort=(
                str(entry["reasoning_effort"])
                if entry.get("reasoning_effort") is not None
                else None
            ),
            stream=stream,
        )

    def _chat_via_curl(
        self,
        *,
        endpoint: str,
        body: bytes,
        request_hash: str,
        active_reasoning_effort: str | None,
        call_started: float,
        proxy_override: str | None = None,
    ) -> str:
        """Make one non-streaming SOCKS-capable OpenAI-compatible request.

        This is a transport compatibility path, not a provider fallback: it
        preserves the selected endpoint, credential, model, and payload.  It
        is used only when the host explicitly configures a SOCKS proxy that
        Python's standard library cannot speak.
        """

        curl = shutil.which("curl")
        if curl is None:
            raise DiscoveryError("Host SOCKS proxy requires curl, but curl is unavailable")
        attempt_started = time.perf_counter()
        # Do not pass the bearer token as a curl argv element: host process
        # listings can expose argv to other local users.  A restrictive,
        # short-lived curl config preserves the exact request while keeping the
        # credential out of process metadata and all experiment artifacts.
        config_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="cellscientist_curl_",
                delete=False,
            ) as handle:
                config_path = Path(handle.name)
                os.chmod(config_path, 0o600)
                handle.write("silent\n")
                handle.write("show-error\n")
                handle.write(f'max-time = "{max(1, int(self.timeout_seconds))}"\n')
                handle.write('request = "POST"\n')
                handle.write(f'url = {json.dumps(endpoint)}\n')
                if proxy_override is not None:
                    handle.write(f'proxy = {json.dumps(proxy_override)}\n')
                handle.write('header = "Content-Type: application/json"\n')
                # Some compatible gateways acknowledge larger request bodies
                # with an interim HTTP 100 response. Suppress curl's implicit
                # Expect: 100-continue handshake so design-card prompts use one
                # ordinary POST.
                handle.write('header = "Expect:"\n')
                handle.write(f'header = {json.dumps(f"Authorization: Bearer {self.api_key}")}\n')
                handle.write('data-binary = "@-"\n')
                handle.write('write-out = "\\\\n%{http_code}"\n')
            result = subprocess.run(
                [curl, "--config", str(config_path)],
                input=body,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        finally:
            if config_path is not None:
                try:
                    config_path.unlink(missing_ok=True)
                except OSError:
                    pass
        raw_output = bytes(result.stdout)
        raw_payload, separator, raw_status = raw_output.rpartition(b"\n")
        http_status = raw_status.decode("ascii", errors="ignore") if separator else ""
        response_hash = hashlib.sha256(raw_payload).hexdigest() if raw_payload else None
        attempt_audit: dict[str, Any] = {
            "attempt": 1,
            "transport": "curl_socks_compatible",
            "elapsed_seconds": time.perf_counter() - attempt_started,
        }
        if http_status.isdigit():
            attempt_audit["http_status"] = int(http_status)
        if response_hash is not None:
            attempt_audit["response_payload_sha256"] = response_hash
        if result.returncode != 0:
            attempt_audit.update({"status": "error", "error_class": "CurlTransportError"})
            self.last_call_audit = {
                "schema_version": "openai_compatible_call_audit.v1",
                "status": "failed",
                "actual_http_attempts": 1,
                "elapsed_seconds": time.perf_counter() - call_started,
                "requested_model": self.model,
                "response_model": None,
                "provider_id": self.provider_id,
                "reasoning_effort": active_reasoning_effort,
                "stream": False,
                "configured_stream": self.stream,
                "transport": "curl_socks_compatible",
                "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
                "request_payload_sha256": request_hash,
                "response_payload_sha256": response_hash,
                "response_stream_sha256": None,
                "finish_reason": None,
                "usage_reported": False,
                "usage": None,
                "attempts": [attempt_audit],
            }
            raise DiscoveryError("SOCKS-compatible transport failed before an HTTP response")
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            attempt_audit.update({"status": "error", "error_class": type(exc).__name__})
            raise DiscoveryError("SOCKS-compatible transport returned malformed JSON") from exc
        if not isinstance(payload, Mapping):
            attempt_audit.update({"status": "error", "error_class": "DiscoveryError"})
            raise DiscoveryError("SOCKS-compatible transport response is not a JSON object")
        if not (http_status.isdigit() and 200 <= int(http_status) < 300):
            attempt_audit.update({"status": "error", "error_class": "HTTPError"})
            message = ""
            if isinstance(payload.get("error"), Mapping):
                message = str(payload["error"].get("message", ""))[:240]
            raise DiscoveryError(f"HTTP {http_status or 'transport'}" + (f": {message}" if message else ""))
        choices = payload.get("choices", [])
        first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, Mapping) else {}
        content = message.get("content") if isinstance(message, Mapping) else None
        text, present = _stream_text(content)
        if not present:
            attempt_audit.update({"status": "error", "error_class": "DiscoveryError"})
            raise DiscoveryError("Response has no textual choices[0].message.content")
        attempt_audit["status"] = "success"
        self.last_call_audit = {
            "schema_version": "openai_compatible_call_audit.v1",
            "status": "success",
            "actual_http_attempts": 1,
            "elapsed_seconds": time.perf_counter() - call_started,
            "requested_model": self.model,
            "response_model": str(payload.get("model"))[:256] if payload.get("model") is not None else None,
            "provider_id": self.provider_id,
            "reasoning_effort": active_reasoning_effort,
            "stream": False,
            "configured_stream": self.stream,
            "transport": "curl_socks_compatible",
            "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
            "request_payload_sha256": request_hash,
            "response_payload_sha256": response_hash,
            "response_stream_sha256": None,
            "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, Mapping) else None,
            "usage_reported": payload.get("usage") is not None,
            "usage": payload.get("usage"),
            "attempts": [attempt_audit],
        }
        return text

    def chat(
        self,
        messages: list[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        reasoning_effort_override: str | None = None,
    ) -> str:
        """Call the configured provider and archive secret-safe transport metadata.

        ``reasoning_effort_override`` is deliberately per-call rather than a
        mutable client setting.  A discovery runner can therefore reserve
        adequate visible-output budget for a complete executable artifact
        without changing the provider identity, endpoint, model, or the
        policy used by unrelated calls.  ``None`` preserves the panel-bound
        default exactly.
        """

        if reasoning_effort_override is not None and reasoning_effort_override not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise DiscoveryError(
                "reasoning_effort_override must be one of minimal, low, medium, or high"
            )
        active_reasoning_effort = (
            self.reasoning_effort
            if reasoning_effort_override is None
            else reasoning_effort_override
        )
        endpoint = self.base_url + "/chat/completions"
        # Do not require response_format=json_object: several otherwise
        # OpenAI-compatible research gateways reject that newer optional field.
        # JSON validity is enforced by parse_candidate_response instead.
        payload_request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if active_reasoning_effort is not None:
            payload_request["reasoning_effort"] = active_reasoning_effort
        # Only the two configured remote campaign gateways opt into this
        # compatibility transport.  Generic OpenAI-compatible clients retain
        # their normal urllib semantics (including test/mock behavior).
        proxy_override = _http_proxy_override(endpoint)
        use_curl_socks_transport = (
            self.provider_id in {"deepseek_v4_flash", "gemini_compatible"}
            and (proxy_override is not None or _needs_socks_compatible_transport(endpoint))
        )
        if self.stream and not use_curl_socks_transport:
            payload_request["stream"] = True
        body = json.dumps(payload_request).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        call_started = time.perf_counter()
        request_hash = hashlib.sha256(body).hexdigest()
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
        if use_curl_socks_transport:
            return self._chat_via_curl(
                endpoint=endpoint,
                body=body,
                request_hash=request_hash,
                active_reasoning_effort=active_reasoning_effort,
                call_started=call_started,
                proxy_override=proxy_override,
            )
        attempts: list[dict[str, Any]] = []
        response_payload_hash: str | None = None
        finish_reason: Any = None
        usage: Any = None
        response_model: str | None = None
        last_error: Exception | None = None
        for attempt in range(self.max_http_attempts):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            attempt_started = time.perf_counter()
            attempt_audit: dict[str, Any] = {"attempt": attempt + 1}
            attempt_response_hash: str | None = None
            stream_hasher: Any = None
            stream_bytes = 0
            # Metadata belongs to one HTTP attempt.  A malformed attempt must
            # not contaminate a later successful retry.
            finish_reason = None
            usage = None
            response_model = None
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    status = getattr(response, "status", None)
                    if status is None and hasattr(response, "getcode"):
                        status = response.getcode()
                    if isinstance(status, int):
                        attempt_audit["http_status"] = status
                    if self.stream:
                        stream_hasher = hashlib.sha256()
                        event_data: list[str] = []
                        content_parts: list[str] = []
                        textual_content_seen = False
                        saw_done = False

                        def consume_event() -> None:
                            nonlocal finish_reason, usage, response_model
                            nonlocal textual_content_seen, saw_done
                            if not event_data:
                                return
                            data = "\n".join(event_data)
                            event_data.clear()
                            if data.strip() == "[DONE]":
                                saw_done = True
                                return
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError as exc:
                                raise DiscoveryError(
                                    "Streaming response contained malformed JSON"
                                ) from exc
                            if not isinstance(payload, Mapping):
                                raise DiscoveryError(
                                    "Streaming response event is not a JSON object"
                                )
                            if isinstance(payload.get("error"), Mapping):
                                raise DiscoveryError(
                                    "Streaming provider returned an error event"
                                )
                            payload_model = payload.get("model")
                            if payload_model is not None:
                                response_model = str(payload_model)[:256]
                            if payload.get("usage") is not None:
                                usage = payload.get("usage")
                            choices = payload.get("choices", [])
                            if not choices:
                                return
                            first_choice = (
                                choices[0]
                                if isinstance(choices, list)
                                and isinstance(choices[0], Mapping)
                                else {}
                            )
                            if not first_choice:
                                return
                            if first_choice.get("finish_reason") is not None:
                                finish_reason = first_choice.get("finish_reason")
                            text = ""
                            present = False
                            if "delta" in first_choice:
                                delta = first_choice.get("delta")
                                if isinstance(delta, Mapping):
                                    text, present = _stream_text(
                                        delta.get("content")
                                        if "content" in delta
                                        else delta.get("text")
                                    )
                                else:
                                    text, present = _stream_text(delta)
                            elif isinstance(first_choice.get("message"), Mapping):
                                text, present = _stream_text(
                                    first_choice["message"].get("content")
                                )
                            elif "text" in first_choice:
                                text, present = _stream_text(first_choice.get("text"))
                            if present:
                                content_parts.append(text)
                                textual_content_seen = True

                        while not saw_done:
                            raw_line = response.readline()
                            if not raw_line:
                                break
                            raw_bytes = (
                                raw_line.encode("utf-8")
                                if isinstance(raw_line, str)
                                else bytes(raw_line)
                            )
                            stream_hasher.update(raw_bytes)
                            stream_bytes += len(raw_bytes)
                            line = raw_bytes.decode("utf-8").rstrip("\r\n")
                            if not line:
                                consume_event()
                            elif line.startswith(":"):
                                continue
                            elif line.startswith("data:"):
                                event_data.append(line[5:].lstrip(" "))
                        if event_data and not saw_done:
                            consume_event()
                        attempt_response_hash = stream_hasher.hexdigest()
                        response_payload_hash = attempt_response_hash
                        if not saw_done:
                            raise DiscoveryError(
                                "Streaming response ended before the [DONE] marker"
                            )
                        if not textual_content_seen:
                            raise DiscoveryError(
                                "Streaming response has no textual choice content"
                            )
                        content = "".join(content_parts)
                    else:
                        raw_payload = response.read()
                        attempt_response_hash = hashlib.sha256(raw_payload).hexdigest()
                        response_payload_hash = attempt_response_hash
                        payload = json.loads(raw_payload.decode("utf-8"))
                        if not isinstance(payload, Mapping):
                            raise DiscoveryError("Response payload is not a JSON object")
                        choices = payload.get("choices", [])
                        first_choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
                        message = first_choice.get("message", {})
                        content = message.get("content") if isinstance(message, Mapping) else None
                        if not isinstance(content, str):
                            raise DiscoveryError("Response has no textual choices[0].message.content")
                        finish_reason = first_choice.get("finish_reason")
                        usage = payload.get("usage")
                        payload_model = payload.get("model")
                        response_model = str(payload_model)[:256] if payload_model is not None else None
                attempt_audit.update(
                    {
                        "status": "success",
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                        "response_payload_sha256": attempt_response_hash,
                    }
                )
                if self.stream:
                    attempt_audit["response_stream_sha256"] = attempt_response_hash
                    attempt_audit["response_stream_bytes"] = stream_bytes
                attempts.append(attempt_audit)
                self.last_call_audit = {
                    "schema_version": "openai_compatible_call_audit.v1",
                    "status": "success",
                    "actual_http_attempts": len(attempts),
                    "elapsed_seconds": time.perf_counter() - call_started,
                    "requested_model": self.model,
                    "response_model": response_model,
                    "provider_id": self.provider_id,
                    "reasoning_effort": active_reasoning_effort,
                    "stream": self.stream,
                    "endpoint_sha256": endpoint_hash,
                    "request_payload_sha256": request_hash,
                    "response_payload_sha256": response_payload_hash,
                    "response_stream_sha256": (
                        response_payload_hash if self.stream else None
                    ),
                    "finish_reason": finish_reason,
                    "usage_reported": usage is not None,
                    "usage": usage,
                    "attempts": attempts,
                }
                return content
            except (
                error.HTTPError,
                error.URLError,
                TimeoutError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                DiscoveryError,
            ) as exc:
                if stream_hasher is not None:
                    attempt_response_hash = stream_hasher.hexdigest()
                    response_payload_hash = attempt_response_hash
                attempt_audit.update(
                    {
                        "status": "error",
                        "error_class": type(exc).__name__,
                        "elapsed_seconds": time.perf_counter() - attempt_started,
                    }
                )
                if attempt_response_hash is not None:
                    attempt_audit["response_payload_sha256"] = attempt_response_hash
                    if self.stream:
                        attempt_audit["response_stream_sha256"] = attempt_response_hash
                        attempt_audit["response_stream_bytes"] = stream_bytes
                if isinstance(exc, error.HTTPError):
                    attempt_audit["http_status"] = int(exc.code)
                    detail = ""
                    try:
                        raw_error_payload = exc.read()
                        attempt_response_hash = hashlib.sha256(raw_error_payload).hexdigest()
                        response_payload_hash = attempt_response_hash
                        attempt_audit["response_payload_sha256"] = attempt_response_hash
                        payload = json.loads(raw_error_payload.decode("utf-8", errors="replace"))
                        if isinstance(payload, Mapping):
                            body_error = payload.get("error", {})
                            if isinstance(body_error, Mapping):
                                detail = str(body_error.get("message", ""))[:240]
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass
                    last_error = DiscoveryError(f"HTTP {exc.code}" + (f": {detail}" if detail else ""))
                else:
                    last_error = exc
                    if isinstance(exc, error.URLError):
                        attempt_audit["reason_class"] = type(exc.reason).__name__
                attempts.append(attempt_audit)
                if attempt < self.max_http_attempts - 1:
                    time.sleep(2**attempt)
        self.last_call_audit = {
            "schema_version": "openai_compatible_call_audit.v1",
            "status": "failed",
            "actual_http_attempts": len(attempts),
            "elapsed_seconds": time.perf_counter() - call_started,
            "requested_model": self.model,
            "response_model": response_model,
            "provider_id": self.provider_id,
            "reasoning_effort": active_reasoning_effort,
            "stream": self.stream,
            "endpoint_sha256": endpoint_hash,
            "request_payload_sha256": request_hash,
            "response_payload_sha256": response_payload_hash,
            "response_stream_sha256": (
                response_payload_hash if self.stream else None
            ),
            "finish_reason": finish_reason,
            "usage_reported": usage is not None,
            "usage": usage,
            "attempts": attempts,
        }
        detail = str(last_error) if last_error is not None else "unknown transport error"
        raise DiscoveryError(f"LLM request failed after 3 attempts: {detail}") from last_error
