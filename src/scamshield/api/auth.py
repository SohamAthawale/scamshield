"""JWT (HS256) login and role checks.

Roles:
  customer   a bank customer; sees only their own warned/held payments and the bank's message to them
  analyst    the bank's scam analyst: held payments with decrypted details, mule watchlist, briefs
  auditor    aggregates and the audit chain; never account numbers or payment details
  service    the payment channel (simulator / core banking); can only POST /payments
"""
from __future__ import annotations

import datetime as dt
import hmac
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..security import verify_password

ROLES = {"customer", "analyst", "auditor", "service"}
ALGO = "HS256"
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    name: str
    role: str
    account_token: str | None = None   # customers: the account they own


class Authenticator:
    def __init__(self, jwt_secret: str, users: dict[str, dict], service_token: str, ttl_minutes: int,
                 account_token_for):
        self.secret = jwt_secret
        self.users = users
        self.service_token = service_token
        self.ttl = dt.timedelta(minutes=ttl_minutes)
        self.account_token_for = account_token_for  # raw "bank:account" -> pseudonymised token

    def login(self, username: str, password: str) -> tuple[str, Principal]:
        u = self.users.get(username)
        if not u or not verify_password(password, u.get("password_hash", "")):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
        atok = self.account_token_for(u["account"]) if u.get("account") else None
        p = Principal(username, u["role"], atok)
        now = dt.datetime.now(dt.timezone.utc)
        claims = {"sub": p.name, "role": p.role, "atok": p.account_token, "iat": now, "exp": now + self.ttl}
        return jwt.encode(claims, self.secret, algorithm=ALGO), p

    def decode(self, token: str) -> Principal:
        if hmac.compare_digest(token.encode(), self.service_token.encode()):
            return Principal("simulator", "service")
        try:
            c = jwt.decode(token, self.secret, algorithms=[ALGO], options={"require": ["exp", "sub", "role"]})
        except jwt.PyJWTError as e:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e
        if c["role"] not in ROLES:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown role")
        return Principal(c["sub"], c["role"], c.get("atok"))


def current_principal(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
                      token: str | None = Query(default=None, include_in_schema=False)) -> Principal:
    """Bearer header, or ?token= for EventSource (browsers can't set headers on SSE)."""
    raw = creds.credentials if creds else token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing credentials")
    return request.app.state.auth.decode(raw)


def require(*roles: str):
    def dep(p: Principal = Depends(current_principal)) -> Principal:
        if p.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"role '{p.role}' cannot access this")
        return p
    return dep
