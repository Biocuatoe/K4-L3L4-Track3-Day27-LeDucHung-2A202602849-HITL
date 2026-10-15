"""LLM factory: an OpenAI-compatible chat model for Groq or OpenRouter.

Provider selection (explicit, never guessed across providers):
  LLM_PROVIDER=groq        -> GROQ_API_KEY,       default base https://api.groq.com/openai/v1
  LLM_PROVIDER=openrouter  -> OPENROUTER_API_KEY, default base https://openrouter.ai/api/v1
If LLM_PROVIDER is unset: OpenRouter when OPENROUTER_API_KEY is set (backward
compatible), otherwise Groq when GROQ_API_KEY is set, otherwise a clear error.
A provider's key is only ever sent to that provider's own endpoint.
"""

import os
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

SECRET_ENV_VARS = ("OPENROUTER_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN")

PROVIDERS = {
    "groq": {
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "host": "groq.com",
        "model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "host": "openrouter.ai",
        "model": "openai/gpt-4o-mini",
    },
}


def redact(text: object) -> str:
    """Replace any configured secret value found in `text` (for logs / UI errors)."""
    out = str(text)
    for name in SECRET_ENV_VARS:
        val = os.environ.get(name)
        if val and len(val) >= 8:
            out = out.replace(val, "***")
    return out


def resolve_provider() -> str:
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER {explicit!r}; expected one of: {', '.join(PROVIDERS)}"
            )
        return explicit
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    raise RuntimeError(
        "No LLM configured: set LLM_PROVIDER=groq and GROQ_API_KEY "
        "(or LLM_PROVIDER=openrouter and OPENROUTER_API_KEY) — see .env.example"
    )


def llm_settings() -> dict:
    """Resolved provider, model, base_url and api_key (api_key is never logged)."""
    provider = resolve_provider()
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['key_env']} is not set (required for LLM_PROVIDER={provider}) — see .env.example")
    base_url = os.environ.get("LLM_BASE_URL") or cfg["base_url"]
    host = urlparse(base_url).hostname or ""
    if not (host == cfg["host"] or host.endswith("." + cfg["host"])):
        raise RuntimeError(
            f"LLM_BASE_URL host {host!r} does not match LLM_PROVIDER={provider}; refusing to send "
            f"{cfg['key_env']} there. Set LLM_BASE_URL={cfg['base_url']} (or unset it)."
        )
    return {
        "provider": provider,
        "model": os.environ.get("LLM_MODEL") or cfg["model"],
        "base_url": base_url,
        "api_key": api_key,
    }


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    s = llm_settings()
    return ChatOpenAI(
        model=s["model"], base_url=s["base_url"], api_key=s["api_key"], temperature=temperature,
    )
