from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(raw: str, name: str) -> int:
    """Parse a strictly positive integer env override, or fail loudly.

    Without this, `LOURA_WORKERS=0` yields zero worker loops: the API answers
    201, reports healthy, and no ticket is ever classified. Configuration
    mistakes must not be able to express a silently-degraded service.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _positive_float(raw: str, name: str) -> float:
    """Parse a strictly positive float env override, or fail loudly.

    A non-positive `LOURA_LLM_TIMEOUT` makes `asyncio.wait_for` raise before
    the model is ever awaited, so every ticket would burn its attempts and fail.
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    db_path: str = "tickets.db"
    llm_backend: str = "laya"  # "laya" | "fake"
    max_attempts: int = 3
    worker_count: int = 2
    llm_timeout: float = 15.0
    retry_base_delay: float = 0.05
    retry_max_delay: float = 1.0
    default_page_size: int = 20
    max_page_size: int = 100
    # Optional generative backend (LOURA_LLM_BACKEND=openai): key/endpoint/model
    # come from the environment; defaults keep the zero-key laya path intact.
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # Optional gate for the admin UI's data endpoints (open when unset).
    admin_token: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        kwargs: dict[str, object] = {}
        if value := os.getenv("LOURA_DB_PATH"):
            kwargs["db_path"] = value
        if value := os.getenv("LOURA_LLM_BACKEND"):
            kwargs["llm_backend"] = value
        # Each numeric override is range-checked at the edge: a bad value here
        # (0 workers, negative timeout) would otherwise start a service that
        # accepts traffic and silently never classifies anything.
        if value := os.getenv("LOURA_MAX_ATTEMPTS"):
            kwargs["max_attempts"] = _positive_int(value, "LOURA_MAX_ATTEMPTS")
        if value := os.getenv("LOURA_WORKERS"):
            kwargs["worker_count"] = _positive_int(value, "LOURA_WORKERS")
        if value := os.getenv("LOURA_LLM_TIMEOUT"):
            kwargs["llm_timeout"] = _positive_float(value, "LOURA_LLM_TIMEOUT")
        if value := os.getenv("LOURA_LLM_API_KEY"):
            kwargs["llm_api_key"] = value
        if value := os.getenv("LOURA_LLM_BASE_URL"):
            kwargs["llm_base_url"] = value
        if value := os.getenv("LOURA_LLM_MODEL"):
            kwargs["llm_model"] = value
        if value := os.getenv("LOURA_ADMIN_TOKEN"):
            kwargs["admin_token"] = value
        return cls(**kwargs)