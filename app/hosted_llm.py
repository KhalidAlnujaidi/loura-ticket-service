"""Optional generative backend speaking the OpenAI chat-completions format.

Selected with LOURA_LLM_BACKEND=openai; key/endpoint/model come from
LOURA_LLM_API_KEY / LOURA_LLM_BASE_URL / LOURA_LLM_MODEL, so any
OpenAI-compatible server works (OpenAI, OpenRouter, vLLM, Ollama's /v1).

Design notes:
- This module is imported only when the backend is selected (see build_llm),
  so the default laya path never touches httpx.
- The client sets NO timeout of its own: the worker's wait_for(LOURA_LLM_TIMEOUT)
  stays the single deadline for every backend.
- Failures surface as ordinary exceptions (HTTP status, bad JSON, missing
  keys); the worker already funnels those through reason_for() into
  retry -> failed. Output still crosses the same strict validation gate.
"""

from __future__ import annotations

import httpx


class HostedLLM:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        if not api_key:
            raise ValueError(
                "LOURA_LLM_BACKEND=openai but LOURA_LLM_API_KEY is not set. "
                "Export a key (point LOURA_LLM_BASE_URL at any OpenAI-compatible "
                "endpoint), or use LOURA_LLM_BACKEND=laya / fake."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
        content = payload["choices"][0]["message"].get("content")
        return content or ""  # empty -> worker treats it as invalid output
