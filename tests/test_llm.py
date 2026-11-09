import json

import httpx
import pytest

from regrag.llm import LLMConfigError, llm_from_env

REPLY = {
    "model": "served-model",
    "choices": [{"message": {"role": "assistant", "content": "Rule 4.2 [Doc 2, 4.2]"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 9},
}


def _capture():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    return seen, httpx.Client(transport=httpx.MockTransport(handler))


def test_no_provider_means_no_llm():
    assert llm_from_env({}) is None
    assert llm_from_env({"REGRAG_LLM_PROVIDER": "none"}) is None


def test_unknown_provider_and_missing_key_are_clear_errors():
    with pytest.raises(LLMConfigError, match="Unknown"):
        llm_from_env({"REGRAG_LLM_PROVIDER": "bedrock", "REGRAG_LLM_MODEL": "m"})
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        llm_from_env({"REGRAG_LLM_PROVIDER": "openai", "REGRAG_LLM_MODEL": "m"})


def test_openai_request_and_response():
    seen, client = _capture()
    env = {"REGRAG_LLM_PROVIDER": "openai", "REGRAG_LLM_MODEL": "m1", "OPENAI_API_KEY": "sk-test"}
    llm = llm_from_env(env, client=client)
    out = llm.complete("system text", "user text", max_tokens=64)
    request = seen[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "m1" and body["max_completion_tokens"] == 64
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert out.text == "Rule 4.2 [Doc 2, 4.2]"
    assert (out.input_tokens, out.output_tokens) == (120, 9)


def test_azure_ai_foundry_uses_the_v1_endpoint_and_api_key_header():
    seen, client = _capture()
    env = {
        "REGRAG_LLM_PROVIDER": "azure",
        "REGRAG_LLM_MODEL": "my-deployment",
        "AZURE_OPENAI_ENDPOINT": "https://myres.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "az-test",
    }
    llm_from_env(env, client=client).complete("s", "u")
    request = seen[0]
    assert str(request.url) == "https://myres.openai.azure.com/openai/v1/chat/completions"
    assert request.headers["api-key"] == "az-test"
    assert "authorization" not in request.headers


def test_ollama_needs_no_key():
    seen, client = _capture()
    env = {"REGRAG_LLM_PROVIDER": "ollama", "REGRAG_LLM_MODEL": "qwen"}
    llm_from_env(env, client=client).complete("s", "u", max_tokens=32)
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"
    assert json.loads(seen[0].content)["max_tokens"] == 32


def test_repr_never_shows_the_key():
    env = {"REGRAG_LLM_PROVIDER": "openai", "REGRAG_LLM_MODEL": "m", "OPENAI_API_KEY": "sk-secret"}
    assert "sk-secret" not in repr(llm_from_env(env))
