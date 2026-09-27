from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["billing", "technical", "account", "other"]
Priority = Literal["low", "medium", "high"]
Status = Literal["pending", "classified", "failed"]


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Category
    priority: Priority
    summary: str = Field(min_length=1, max_length=300)


class TicketIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    subject: str = Field(max_length=300)
    body: str = Field(max_length=20_000)


class TicketOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    subject: str
    body: str
    status: Status
    attempts: int
    failure_reason: str | None
    classification: Classification | None
    created_at: str
    classified_at: str | None


class TicketListOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TicketOut]
    page: int
    page_size: int
    total: int