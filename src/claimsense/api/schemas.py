"""Request/response models. A claim arrives with real identifiers; they are pseudonymised on arrival."""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimIn(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    ClaimID: str = Field(min_length=1, max_length=64)
    BeneID: str = Field(min_length=1, max_length=64)
    Provider: str = Field(min_length=1, max_length=64)
    ClaimStartDt: dt.date
    ClaimEndDt: dt.date | None = None
    AdmissionDt: dt.date | None = None
    DischargeDt: dt.date | None = None
    InscClaimAmtReimbursed: float = Field(ge=0, le=10_000_000)
    DeductibleAmtPaid: float | None = Field(default=0.0, ge=0, le=10_000_000)
    AttendingPhysician: str | None = None
    OperatingPhysician: str | None = None
    OtherPhysician: str | None = None
    ClmAdmitDiagnosisCode: str | None = None
    DiagnosisGroupCode: str | None = None
    ClmDiagnosisCode_1: str | None = None
    ClmDiagnosisCode_2: str | None = None
    ClmDiagnosisCode_3: str | None = None
    ClmDiagnosisCode_4: str | None = None
    ClmDiagnosisCode_5: str | None = None
    ClmDiagnosisCode_6: str | None = None
    ClmDiagnosisCode_7: str | None = None
    ClmDiagnosisCode_8: str | None = None
    ClmDiagnosisCode_9: str | None = None
    ClmDiagnosisCode_10: str | None = None
    ClmProcedureCode_1: str | None = None
    ClmProcedureCode_2: str | None = None
    ClmProcedureCode_3: str | None = None
    ClmProcedureCode_4: str | None = None
    ClmProcedureCode_5: str | None = None
    ClmProcedureCode_6: str | None = None
    is_inpatient: int = Field(ge=0, le=1)

    @field_validator("*", mode="before")
    @classmethod
    def blank_to_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().upper() in {"", "NA", "NAN", "NONE"}:
            return None
        if isinstance(v, float) and v != v:  # NaN
            return None
        return v

    @field_validator("ClmProcedureCode_1", "ClmProcedureCode_2", "ClmProcedureCode_3", "ClmProcedureCode_4",
                     "ClmProcedureCode_5", "ClmProcedureCode_6", "ClmDiagnosisCode_1", "ClmAdmitDiagnosisCode",
                     "DiagnosisGroupCode", mode="before")
    @classmethod
    def code_to_str(cls, v: Any) -> Any:
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        if isinstance(v, int):
            return str(v)
        return v


Action = Literal["PAY", "RETURN_WITH_FIX", "HOLD_FOR_REVIEW"]


class ClaimOut(BaseModel):
    claim_token: str
    provider_token: str
    action: Action
    reasons: dict[str, Any]
    anomaly: float
    provider_risk: float | None
    latency_ms: float
    model_ms: float
    backend: str
    batch_size: int


class LoginIn(BaseModel):
    username: str
    password: str


class ResolveIn(BaseModel):
    status: Literal["released", "confirmed_issue", "needs_more_info"]
    comment: str = Field(default="", max_length=500)
