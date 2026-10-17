"""Create ./secrets (keys, JWT secret, service token, demo users). Run once on the VM; never commit it.

    python3 scripts/make_secrets.py --customer-account 0119:812345678

Demo passwords are written to secrets/demo_passwords.txt (readable only by you), not printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scamshield.security import generate_secrets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "secrets")
    ap.add_argument("--customer-account", required=True, help='raw "bank:account" the demo customer login owns')
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if (a.out / "pseudo_key").exists() and not a.force:
        sys.exit(f"{a.out} already has secrets; use --force to replace them (old data becomes unreadable)")
    users = {
        "analyst": {"role": "analyst"},
        "auditor": {"role": "auditor"},
        "customer": {"role": "customer", "account": a.customer_account},
    }
    pw = generate_secrets(a.out, users)
    p = a.out / "demo_passwords.txt"
    p.write_text("".join(f"{u}: {pw[u]}\n" for u in users))
    p.chmod(0o600)
    print(f"secrets written to {a.out}; demo logins in {p}")


if __name__ == "__main__":
    main()
