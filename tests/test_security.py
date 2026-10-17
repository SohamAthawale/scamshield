"""security.py: pseudonymisation, envelope encryption, passwords, secrets folder."""
import base64
import json
import os
import stat

import pytest
from cryptography.exceptions import InvalidTag

from scamshield import security
from scamshield.security import (
    SECRET_FILES, TOKEN_PREFIX, Envelope, Pseudonymizer, Sealed, generate_secrets, hash_password, load_keyring,
    read_secret, verify_password,
)

KEY_A = bytes(range(32))
KEY_B = bytes(range(1, 33))
B32 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")


# ------------------------------------------------------------------------------ Pseudonymizer
def test_pseudonymizer_is_deterministic():
    p = Pseudonymizer(KEY_A)
    assert p.token("bene", "BENE11001") == p.token("bene", "BENE11001")
    assert Pseudonymizer(KEY_A).token("bene", "BENE11001") == p.token("bene", "BENE11001")


def test_pseudonymizer_separates_kinds():
    p = Pseudonymizer(KEY_A)
    toks = {k: p.token(k, "SAME123") for k in TOKEN_PREFIX}
    assert {t[0] for t in toks.values()} == set(TOKEN_PREFIX.values())
    assert len({t[1:] for t in toks.values()}) == len(TOKEN_PREFIX)   # the HMAC part differs too


def test_pseudonymizer_different_keys_give_different_tokens():
    assert Pseudonymizer(KEY_A).token("provider", "PRV51001") != Pseudonymizer(KEY_B).token("provider", "PRV51001")


def test_pseudonymizer_token_shape_and_no_leak():
    p = Pseudonymizer(KEY_A)
    t = p.token("account", "0119:8000E0380")
    assert t.startswith("A") and len(t) == 17 and set(t[1:]) <= B32
    assert "8000E0380" not in t and "0119" not in t
    assert p.token("payment", "T1").startswith("T")
    assert len(Pseudonymizer(KEY_A, length=8).token("payment", "T1")) == 9
    assert p.token("weird-kind", "x").startswith("X")


def test_pseudonymizer_distinct_values_distinct_tokens():
    p = Pseudonymizer(KEY_A)
    toks = {p.token("bene", f"BENE{i}") for i in range(2000)}
    assert len(toks) == 2000


@pytest.mark.parametrize("value", [None, "", "   "])
def test_pseudonymizer_empty_is_none(value):
    assert Pseudonymizer(KEY_A).token("bene", value) is None


def test_pseudonymizer_rejects_short_key():
    with pytest.raises(ValueError):
        Pseudonymizer(b"x" * 31)
    Pseudonymizer(b"x" * 32)


# ------------------------------------------------------------------------------ Envelope
@pytest.fixture
def env():
    return Envelope({"k1": KEY_A}, "k1")


def test_envelope_round_trip(env):
    s = env.seal(b"secret claim payload", b"C123")
    assert s.key_id == "k1" and len(s.nonce) == 12
    assert b"secret claim payload" not in s.ciphertext
    assert env.open(s, b"C123") == b"secret claim payload"


def test_envelope_json_round_trip(env):
    obj = {"amount": 26000, "dx": ["4019", "V5866"], "when": "2009-04-12"}
    assert env.open_json(env.seal_json(obj, aad="C1"), aad="C1") == obj


def test_envelope_wrong_aad_fails(env):
    s = env.seal(b"payload", b"C123")
    with pytest.raises(InvalidTag):
        env.open(s, b"C124")
    with pytest.raises(InvalidTag):
        env.open(s)
    with pytest.raises(InvalidTag):
        env.open_json(env.seal_json({"a": 1}, aad="row-1"), aad="row-2")


@pytest.mark.parametrize("field", ["ciphertext", "nonce"])
def test_envelope_tampering_fails(env, field):
    s = env.seal(b"payload", b"aad")
    raw = bytearray(getattr(s, field))
    raw[0] ^= 0x01
    bad = Sealed(s.key_id, bytes(raw) if field == "nonce" else s.nonce,
                 bytes(raw) if field == "ciphertext" else s.ciphertext)
    with pytest.raises(InvalidTag):
        env.open(bad, b"aad")


def test_envelope_truncated_tag_fails(env):
    s = env.seal(b"payload")
    with pytest.raises(InvalidTag):
        env.open(Sealed(s.key_id, s.nonce, s.ciphertext[:-1]))


def test_envelope_unknown_key_id(env):
    s = env.seal(b"payload")
    with pytest.raises(KeyError, match="k9"):
        env.open(Sealed("k9", s.nonce, s.ciphertext))


def test_envelope_key_rotation_keeps_old_rows_readable():
    old = Envelope({"k1": KEY_A}, "k1")
    s1 = old.seal_json({"row": 1}, aad="C1")
    rotated = Envelope({"k1": KEY_A, "k2": KEY_B}, "k2")
    s2 = rotated.seal_json({"row": 2}, aad="C2")
    assert s2.key_id == "k2"
    assert rotated.open_json(s1, aad="C1") == {"row": 1}
    assert rotated.open_json(s2, aad="C2") == {"row": 2}
    with pytest.raises(KeyError):
        old.open(s2, b"C2")


def test_envelope_same_key_bytes_under_wrong_id_fails():
    s = Envelope({"k1": KEY_A}, "k1").seal(b"x")
    with pytest.raises(InvalidTag):
        Envelope({"k1": KEY_B}, "k1").open(s)


def test_envelope_nonces_are_unique(env):
    seals = [env.seal(b"same plaintext") for _ in range(2000)]
    assert len({s.nonce for s in seals}) == 2000
    assert len({s.ciphertext for s in seals}) == 2000


@pytest.mark.parametrize("keys,active", [({"k1": b"x" * 16}, "k1"), ({"k1": b"x" * 33}, "k1"), ({"k1": KEY_A}, "k2")])
def test_envelope_rejects_bad_keyring(keys, active):
    with pytest.raises(ValueError):
        Envelope(keys, active)


# ------------------------------------------------------------------------------ passwords
def test_password_hash_and_verify():
    h = hash_password("correct horse")
    assert h.startswith("scrypt$") and h.count("$") == 2
    assert "correct horse" not in h
    assert verify_password("correct horse", h)
    assert not verify_password("correct horsE", h)
    assert not verify_password("", h)


def test_password_hash_is_salted():
    assert hash_password("pw") != hash_password("pw")
    salt = b"s" * 16
    assert hash_password("pw", salt) == hash_password("pw", salt)
    assert base64.b64decode(hash_password("pw", salt).split("$")[1]) == salt


@pytest.mark.parametrize("stored", ["", "plaintext", "bcrypt$abc$def", "scrypt$a$b$c"])
def test_verify_password_rejects_malformed(stored):
    assert verify_password("pw", stored) is False


# ------------------------------------------------------------------------------ secrets folder
DEMO_USERS = {"alice": {"role": "investigator"}, "bob": {"role": "provider", "provider_id": "P123"}}


@pytest.fixture
def secrets_path(tmp_path):
    base = tmp_path / "secrets"
    passwords = generate_secrets(base, DEMO_USERS)
    return base, passwords


def test_generate_secrets_files_and_permissions(secrets_path):
    base, _ = secrets_path
    assert stat.S_IMODE(os.stat(base).st_mode) == 0o700
    for name, fname in SECRET_FILES.items():
        p = base / fname
        assert p.exists(), name
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600, name


def test_generate_secrets_contents(secrets_path):
    base, passwords = secrets_path
    assert len(bytes.fromhex(read_secret("pseudo_key", base))) == 32
    assert len(bytes.fromhex(read_secret("jwt_secret", base))) == 32
    assert len(read_secret("service_token", base)) >= 32
    Pseudonymizer(bytes.fromhex(read_secret("pseudo_key", base)))   # long enough for the pseudonymizer
    users = json.loads(read_secret("users", base))
    assert set(users) == set(passwords) == set(DEMO_USERS)
    assert users["bob"]["provider_id"] == "P123" and users["bob"]["role"] == "provider"
    raw = (base / SECRET_FILES["users"]).read_text()
    for name, pw in passwords.items():
        assert pw not in raw
        assert verify_password(pw, users[name]["password_hash"])
    assert passwords["alice"] != passwords["bob"]


def test_generate_secrets_twice_gives_new_values(tmp_path):
    generate_secrets(tmp_path / "s", {})
    first = read_secret("pseudo_key", tmp_path / "s")
    generate_secrets(tmp_path / "s", {})
    assert read_secret("pseudo_key", tmp_path / "s") != first


def test_load_keyring(secrets_path):
    base, _ = secrets_path
    env = load_keyring(base)
    assert env.active_key_id == "k1"
    assert env.open_json(env.seal_json({"x": 1}, "a"), "a") == {"x": 1}


def test_load_keyring_with_rotated_keys(tmp_path):
    (tmp_path / "data_keys.json").write_text(json.dumps({"active": "k2", "keys": {"k1": KEY_A.hex(),
                                                                                    "k2": KEY_B.hex()}}))
    old = Envelope({"k1": KEY_A}, "k1").seal(b"old")
    env = load_keyring(tmp_path)
    assert env.active_key_id == "k2" and env.open(old) == b"old"


def test_read_secret_strips_whitespace_and_uses_env_dir(tmp_path, monkeypatch):
    (tmp_path / "jwt_secret").write_text("  abc123\n")
    assert read_secret("jwt_secret", tmp_path) == "abc123"
    monkeypatch.setenv("SCAMSHIELD_SECRETS", str(tmp_path))
    assert security.secrets_dir() == tmp_path
    assert read_secret("jwt_secret") == "abc123"
    with pytest.raises(KeyError):
        read_secret("not_a_secret", tmp_path)


def test_secrets_dir_default(monkeypatch):
    monkeypatch.delenv("SCAMSHIELD_SECRETS", raising=False)
    assert str(security.secrets_dir()) == "/run/secrets"


def test_account_and_payment_tokens():
    p = Pseudonymizer(KEY_A)
    acct = p.token("account", "0119:8C0FFEE01")
    pay = p.token("payment", "T17")
    assert acct != p.token("payment", "0119:8C0FFEE01")             # kinds are separated by the HMAC input
    assert "8C0FFEE01" not in acct and "0119" not in acct and acct == p.token("account", "0119:8C0FFEE01")
    assert len(acct) == len(pay) == 17
    assert acct != p.token("account", "119:8C0FFEE01")               # bank codes keep their leading zeros


def test_make_secrets_script_writes_customer_account(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_secrets.py"
    out = tmp_path / "secrets"
    r = subprocess.run([sys.executable, str(script), "--out", str(out), "--customer-account", "0119:8C0FFEE01"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    users = json.loads((out / "users.json").read_text())
    assert users["customer"] == {**users["customer"], "role": "customer", "account": "0119:8C0FFEE01"}
    assert {u["role"] for u in users.values()} == {"analyst", "auditor", "customer"}
    pw = dict(line.split(": ", 1) for line in (out / "demo_passwords.txt").read_text().splitlines())
    assert verify_password(pw["customer"], users["customer"]["password_hash"])
    assert stat.S_IMODE(os.stat(out / "demo_passwords.txt").st_mode) == 0o600
    assert pw["customer"] not in r.stdout
    again = subprocess.run([sys.executable, str(script), "--out", str(out), "--customer-account", "1:2"],
                           capture_output=True, text=True)
    assert again.returncode != 0 and "already has secrets" in again.stderr
