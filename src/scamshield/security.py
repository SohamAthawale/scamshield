"""Security primitives: pseudonymisation, encryption at rest, password hashing, secret loading.

On IBM Z, AES-GCM and SHA-256 in OpenSSL (which `cryptography` links against) run on CPACF,
the on-core crypto instructions. See docs/SECURITY.md for the evidence files.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

TOKEN_PREFIX = {"account": "A", "payment": "T"}


class Pseudonymizer:
    """Keyed, deterministic tokens for identifiers (HMAC-SHA256).

    The same identifier always maps to the same token, so features and joins still work, but the raw
    identifier cannot be recovered without the key.
    """

    def __init__(self, key: bytes, length: int = 16):
        if len(key) < 32:
            raise ValueError("pseudonymisation key must be at least 32 bytes")
        self._key = key
        self.length = length

    def token(self, kind: str, value: str | None) -> str | None:
        if value is None or str(value).strip() == "":
            return None
        mac = hmac.new(self._key, f"{kind}:{value}".encode(), hashlib.sha256).digest()
        b32 = base64.b32encode(mac).decode().rstrip("=")
        return TOKEN_PREFIX.get(kind, "X") + b32[: self.length]


@dataclass(frozen=True)
class Sealed:
    key_id: str
    nonce: bytes
    ciphertext: bytes


class Envelope:
    """AES-256-GCM with a random 96-bit nonce per record and a key id per row (for rotation).

    `aad` (associated data, e.g. the claim token) is authenticated but not encrypted, so a ciphertext
    cannot be moved to another row without detection.
    """

    def __init__(self, keys: dict[str, bytes], active_key_id: str):
        for kid, k in keys.items():
            if len(k) != 32:
                raise ValueError(f"key {kid!r} must be 32 bytes for AES-256")
        if active_key_id not in keys:
            raise ValueError("active key id not in keyring")
        self._keys = {kid: AESGCM(k) for kid, k in keys.items()}
        self.active_key_id = active_key_id

    def seal(self, plaintext: bytes, aad: bytes = b"") -> Sealed:
        nonce = os.urandom(12)
        ct = self._keys[self.active_key_id].encrypt(nonce, plaintext, aad)
        return Sealed(self.active_key_id, nonce, ct)

    def open(self, sealed: Sealed, aad: bytes = b"") -> bytes:
        try:
            aes = self._keys[sealed.key_id]
        except KeyError as e:
            raise KeyError(f"unknown key id {sealed.key_id!r}") from e
        return aes.decrypt(sealed.nonce, sealed.ciphertext, aad)

    def seal_json(self, obj: dict, aad: str = "") -> Sealed:
        return self.seal(json.dumps(obj, sort_keys=True, default=str).encode(), aad.encode())

    def open_json(self, sealed: Sealed, aad: str = "") -> dict:
        return json.loads(self.open(sealed, aad.encode()))


# ---------------------------------------------------------------------------- passwords
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_b64, dk_b64 = stored.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    try:
        salt, expected = base64.b64decode(salt_b64, validate=True), base64.b64decode(dk_b64, validate=True)
    except (ValueError, TypeError):
        return False
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------- secrets
SECRET_FILES = {
    "pseudo_key": "pseudo_key",        # 32+ random bytes, hex
    "jwt_secret": "jwt_secret",        # 32+ random bytes, hex
    "data_keys": "data_keys.json",     # {"active": "k1", "keys": {"k1": "<hex 32 bytes>"}}
    "users": "users.json",             # {"name": {"role": ..., "password_hash": ..., "provider_id": ...}}
    "service_token": "service_token",  # bearer token the simulator uses
}


def secrets_dir() -> Path:
    return Path(os.environ.get("SCAMSHIELD_SECRETS", "/run/secrets"))


def read_secret(name: str, base: Path | None = None) -> str:
    path = (base or secrets_dir()) / SECRET_FILES[name]
    return path.read_text().strip()


def load_keyring(base: Path | None = None) -> Envelope:
    spec = json.loads(read_secret("data_keys", base))
    return Envelope({k: bytes.fromhex(v) for k, v in spec["keys"].items()}, spec["active"])


def generate_secrets(base: Path, demo_users: dict[str, dict]) -> dict[str, str]:
    """Create a fresh secrets folder. Returns the generated demo passwords (print once, never store)."""
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)

    def write(name: str, content: str) -> None:
        p = base / SECRET_FILES[name]
        p.write_text(content)
        os.chmod(p, 0o600)

    write("pseudo_key", secrets.token_hex(32))
    write("jwt_secret", secrets.token_hex(32))
    write("data_keys", json.dumps({"active": "k1", "keys": {"k1": secrets.token_hex(32)}}))
    write("service_token", secrets.token_urlsafe(32))
    passwords, users = {}, {}
    for name, spec in demo_users.items():
        pw = secrets.token_urlsafe(9)
        passwords[name] = pw
        users[name] = {**spec, "password_hash": hash_password(pw)}
    write("users", json.dumps(users, indent=2))
    return passwords
