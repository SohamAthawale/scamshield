"""Runtime settings, all from environment variables (set in docker-compose.yml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    policy_file: Path
    raw_dir: Path
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
        data = Path(e("CLAIMSENSE_DATA_DIR", "/data"))
        return cls(
            models_dir=Path(e("CLAIMSENSE_MODELS", "/models")),
            policy_file=Path(e("CLAIMSENSE_POLICY", "/app/config/policy.json")),
            raw_dir=Path(e("CLAIMSENSE_RAW", "/raw")),
            db_path=Path(e("CLAIMSENSE_DB", str(data / "claimsense.db"))),
            checkpoint_path=Path(e("CLAIMSENSE_AUDIT_CHECKPOINTS", str(data / "audit_checkpoints.log"))),
            secrets_dir=Path(e("CLAIMSENSE_SECRETS", "/run/secrets")),
            anomaly_backend=e("ANOMALY_BACKEND", "auto"),
            nnpa_min_batch=int(e("NNPA_MIN_BATCH", "256")),
            batch_window_ms=float(e("BATCH_WINDOW_MS", "3")),
            max_batch=int(e("MAX_BATCH", "512")),
            jwt_ttl_minutes=int(e("JWT_TTL_MINUTES", "480")),
        )
