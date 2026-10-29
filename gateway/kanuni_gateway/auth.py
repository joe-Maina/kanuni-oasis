"""API-key check, shared with the Kanuni backend (SIM_GATEWAY_API_KEY
there). Every /v1 request carries `Authorization: Bearer <key>`; HTTPS
(the TLS proxy in front) keeps it private on the wire."""
from __future__ import annotations

import hmac


def check(api_key: str, authorization: str | None) -> bool:
    scheme, _, given = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(given.strip().encode(), api_key.encode())
