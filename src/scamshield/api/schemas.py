"""Request/response models. A payment arrives with real account numbers; they are pseudonymised on arrival."""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..data import FORMATS

PaymentFormat = Literal[tuple(FORMATS)]  # type: ignore[valid-type]


class PaymentIn(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    payment_id: str = Field(min_length=1, max_length=64)
    ts: dt.datetime
    from_bank: str = Field(min_length=1, max_length=16)
    from_acct: str = Field(min_length=1, max_length=32)
    to_bank: str = Field(min_length=1, max_length=16)
    to_acct: str = Field(min_length=1, max_length=32)
    amount_paid: float = Field(ge=0, le=1e12)
    currency_paid: str = Field(min_length=1, max_length=32)
    amount_received: float = Field(ge=0, le=1e12)
    currency_received: str = Field(min_length=1, max_length=32)
    payment_format: PaymentFormat

    @property
    def sender(self) -> str:
        return f"{self.from_bank}:{self.from_acct}"

    @property
    def receiver(self) -> str:
        return f"{self.to_bank}:{self.to_acct}"


Action = Literal["ALLOW", "WARN", "HOLD"]


class PaymentOut(BaseModel):
    payment_token: str
    action: Action
    message: str | None
    reasons: dict[str, Any]
    risk: float
    anomaly: float
    latency_ms: float
    model_ms: float
    backend: str
    batch_size: int


class LoginIn(BaseModel):
    username: str
    password: str


class ResolveIn(BaseModel):
    status: Literal["released", "stopped", "contact_customer"]
    comment: str = Field(default="", max_length=500)
