"""Runtime settings, all from environment variables (set in docker-compose.yml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    policy_file: Path
    history_file: Path
    history_end: str
    db_path: Path
    checkpoint_path: Path
    secrets_dir: Path
    anomaly_backend: str
    nnpa_min_batch: int
    batch_window_ms: float
    max_batch: int
    jwt_ttl_minutes: int

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        data = Path(e("SCAMSHIELD_DATA_DIR", "/data"))
        return cls(
            models_dir=Path(e("SCAMSHIELD_MODELS", "/models")),
            policy_file=Path(e("SCAMSHIELD_POLICY", "/app/config/policy.json")),
            history_file=Path(e("SCAMSHIELD_HISTORY", "/data-in/history.csv")),
            history_end=e("HISTORY_END", "2022-09-09"),
            db_path=Path(e("SCAMSHIELD_DB", str(data / "scamshield.db"))),
            checkpoint_path=Path(e("SCAMSHIELD_AUDIT_CHECKPOINTS", str(data / "audit_checkpoints.log"))),
            secrets_dir=Path(e("SCAMSHIELD_SECRETS", "/run/secrets")),
            anomaly_backend=e("ANOMALY_BACKEND", "auto"),
            nnpa_min_batch=int(e("NNPA_MIN_BATCH", "1024")),
            batch_window_ms=float(e("BATCH_WINDOW_MS", "3")),
            max_batch=int(e("MAX_BATCH", "4096")),
            jwt_ttl_minutes=int(e("JWT_TTL_MINUTES", "480")),
        )
