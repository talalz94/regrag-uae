"""Pluggable LLM backend for the cited answer slice. Keys come from environment variables only.

REGRAG_LLM_PROVIDER picks the backend:

    none     no LLM (the default). Retrieval and evals need no key at all.
    openai   OpenAI, or any server with the same API via OPENAI_BASE_URL
    azure    Azure OpenAI in Azure AI Foundry, through its v1 endpoint
    ollama   a local Ollama server, no key needed

All four speak the same chat completions request shape, so one small client covers them.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx


class LLMConfigError(RuntimeError):
    """The environment does not describe a usable backend."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> Completion: ...


class ChatCompletionsLLM:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        headers: Mapping[str, str] | None = None,
        token_param: str = "max_completion_tokens",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.token_param = token_param
        self._headers = dict(headers or {})
        self._client = client or httpx.Client(timeout=timeout)

    def __repr__(self) -> str:  # never print headers, they hold the key
        return f"ChatCompletionsLLM(base_url={self.base_url!r}, model={self.model!r})"

    def complete(self, system: str, user: str, *, max_tokens: int = 512) -> Completion:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self.token_param: max_tokens,
        }
        start = time.perf_counter()
        response = self._client.post(
            f"{self.base_url}/chat/completions", json=payload, headers=self._headers
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        return Completion(
            text=data["choices"][0]["message"].get("content") or "",
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )


def _require(env: Mapping[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise LLMConfigError(f"{name} is not set. See .env.example.")
    return value


def llm_from_env(
    env: Mapping[str, str] | None = None, *, client: httpx.Client | None = None
) -> LLM | None:
    """Build the backend described by the environment, or None when no LLM is configured."""
    env = os.environ if env is None else env
    provider = (env.get("REGRAG_LLM_PROVIDER") or "none").strip().lower()
    if provider == "none":
        return None
    if provider not in {"openai", "azure", "ollama"}:
        raise LLMConfigError(
            f"Unknown REGRAG_LLM_PROVIDER {provider!r}. Use none, openai, azure or ollama."
        )
    model = _require(env, "REGRAG_LLM_MODEL")
    if provider == "openai":
        key = _require(env, "OPENAI_API_KEY")
        return ChatCompletionsLLM(
            base_url=env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            model=model,
            headers={"Authorization": f"Bearer {key}"},
            client=client,
        )
    if provider == "azure":
        endpoint = _require(env, "AZURE_OPENAI_ENDPOINT").rstrip("/")
        key = _require(env, "AZURE_OPENAI_API_KEY")
        # On Azure the model name is your deployment name.
        return ChatCompletionsLLM(
            base_url=f"{endpoint}/openai/v1", model=model, headers={"api-key": key}, client=client
        )
    return ChatCompletionsLLM(
        base_url=env.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1",
        model=model,
        token_param="max_tokens",
        client=client,
    )
