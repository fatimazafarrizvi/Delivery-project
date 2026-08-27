"""Optional OpenAI-compatible LLM client.

The app is fully usable without a key. When OPENAI_API_KEY is set, Task 1 and
Task 2 can call a hosted model with temperature 0 and JSON responses.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .pii import redact_text


SECRET_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "SUPPORT_AI_USE_LLM",
    "SUPPORT_AI_EVAL_LLM",
)


def _from_streamlit_secrets(name: str) -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        return ""
    return ""


def apply_streamlit_secrets() -> None:
    """Copy Streamlit Cloud / local secrets.toml into env if env is empty."""

    for name in SECRET_NAMES:
        if os.getenv(name, "").strip():
            continue
        value = _from_streamlit_secrets(name)
        if value:
            os.environ[name] = value


def get_setting(name: str, default: str = "") -> str:
    load_dotenv_if_present()
    apply_streamlit_secrets()
    value = os.getenv(name, "").strip()
    return value if value else default


def use_llm() -> bool:
    flag = get_setting("SUPPORT_AI_USE_LLM", "1").lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(get_setting("OPENAI_API_KEY"))


def _endpoint() -> str:
    base = get_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return f"{base}/chat/completions"


def _model() -> str:
    return get_setting("OPENAI_MODEL", "gpt-4o-mini")


def complete_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    seed: int = 7,
    timeout: int = 45,
) -> dict[str, Any]:
    payload = {
        "model": _model(),
        "temperature": temperature,
        "seed": seed,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": redact_text(system)},
            {"role": "user", "content": redact_text(user)},
        ],
    }
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM request failed with HTTP status {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("LLM request failed because the provider was unavailable.") from exc

    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM did not return a JSON object.")
    return parsed


def stream_text(text: str, chunk_size: int = 28) -> Iterator[str]:
    """Deterministic streaming helper used by the UI for Task 1 drafts."""

    value = text or ""
    if not value:
        return
    for index in range(0, len(value), chunk_size):
        yield value[index : index + chunk_size]


def env_status() -> dict[str, str | bool]:
    key_present = bool(get_setting("OPENAI_API_KEY"))
    flag = get_setting("SUPPORT_AI_USE_LLM", "1").lower()
    disabled = flag in {"0", "false", "no", "off"}
    if disabled:
        reason = "SUPPORT_AI_USE_LLM is off"
    elif not key_present:
        reason = "no key in .env or Streamlit secrets"
    else:
        reason = "hosted overlay enabled"
    return {
        "enabled": bool(use_llm()),
        "key_present": key_present,
        "reason": reason,
    }


def load_dotenv_if_present() -> None:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8-sig") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            current = os.environ.get(key, "")
            if not current.strip():
                os.environ[key] = value
