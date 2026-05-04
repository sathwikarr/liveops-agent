"""Small wrapper around Fernet for at-rest secret storage.

The encryption key comes from `LIVEOPS_FERNET_KEY` (preferred) or
`SESSION_SECRET` as a fallback so dev environments don't crash.  In
production the key MUST be set explicitly — without it, a fresh deploy
would generate a new key on every boot and old ciphertexts would become
unreadable.
"""
from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


def _derive_key() -> bytes:
    """Return a Fernet-shaped 32-byte url-safe-base64 key.

    Order of preference:
      1. LIVEOPS_FERNET_KEY  — already a 32-byte url-safe-base64 string
      2. SESSION_SECRET      — derive via SHA-256 → base64url
      3. process-local random — dev only, NEVER on a real deploy
    """
    raw = os.environ.get("LIVEOPS_FERNET_KEY") or ""
    if raw:
        try:
            # Validate: Fernet keys are 32 bytes url-safe-base64.
            Fernet(raw.encode() if isinstance(raw, str) else raw)
            return raw.encode() if isinstance(raw, str) else raw
        except Exception:
            pass
    seed = os.environ.get("SESSION_SECRET") or "dev-only-do-not-use-in-prod"
    return base64.urlsafe_b64encode(hashlib.sha256(seed.encode()).digest())


_FERNET: Optional[Fernet] = None


def _f() -> Fernet:
    global _FERNET
    if _FERNET is None:
        _FERNET = Fernet(_derive_key())
    return _FERNET


def encrypt(plain: str) -> bytes:
    return _f().encrypt(plain.encode("utf-8"))


def decrypt(token: bytes) -> Optional[str]:
    """Returns None on InvalidToken so callers can treat it as 'not configured'
    rather than crashing the request."""
    try:
        return _f().decrypt(token).decode("utf-8")
    except (InvalidToken, ValueError, TypeError):
        return None


__all__ = ["encrypt", "decrypt"]
