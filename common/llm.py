"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI

SECRET_ENV_VARS = ("OPENROUTER_API_KEY", "GITHUB_TOKEN")


def redact(text: object) -> str:
    """Replace any configured secret value found in `text` (for logs / UI errors)."""
    out = str(text)
    for name in SECRET_ENV_VARS:
        val = os.environ.get(name)
        if val and len(val) >= 8:
            out = out.replace(val, "***")
    return out


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )
