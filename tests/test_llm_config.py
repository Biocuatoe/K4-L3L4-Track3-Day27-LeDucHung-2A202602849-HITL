import pytest

from common import llm
from common.llm import get_llm, llm_settings, redact, resolve_provider

ENV = ("LLM_PROVIDER", "GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def test_groq_uses_groq_key_and_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123456")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-other-key-123")
    s = llm_settings()
    assert s["api_key"] == "gsk_test_key_123456"
    assert s["base_url"] == "https://api.groq.com/openai/v1" and s["model"] == "openai/gpt-oss-120b"
    m = get_llm()
    assert m.model_name == "openai/gpt-oss-120b"
    assert m.openai_api_base == "https://api.groq.com/openai/v1"


def test_openrouter_backward_compatible(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-123")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert resolve_provider() == "openrouter"
    assert llm_settings()["api_key"] == "sk-or-test-key-123"


def test_autodetect_groq_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123456")
    assert resolve_provider() == "groq"


def test_missing_credentials_name_the_variable(monkeypatch):
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        resolve_provider()
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        get_llm()
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        get_llm()


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        resolve_provider()


def test_key_never_sent_to_other_providers_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123456")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")  # stale OpenRouter .env value
    with pytest.raises(RuntimeError, match="refusing to send") as e:
        llm_settings()
    assert "gsk_test" not in str(e.value)


def test_redacts_both_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123456")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-123")
    out = redact("a gsk_test_key_123456 b sk-or-test-key-123")
    assert "gsk_test" not in out and "sk-or-test" not in out
    assert "GROQ_API_KEY" in llm.SECRET_ENV_VARS
