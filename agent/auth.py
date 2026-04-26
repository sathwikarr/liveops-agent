"""Username + bcrypt-password auth backed by SQLite (agent.db.users)."""
from __future__ import annotations

import re
from typing import Tuple

import bcrypt

from agent import db

USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_username(username: str) -> Tuple[bool, str]:
    if not USERNAME_RE.fullmatch(username or ""):
        return False, "Username must be 3–32 chars: letters, digits, `_`, `-` only."
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters."
    return True, ""


def signup(username: str, password: str) -> Tuple[bool, str]:
    """Create a new account. Returns (ok, message)."""
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg
    if not db.create_user(username, _hash(password)):
        return False, "Username is already taken."
    return True, "Account created — you can log in now."


def login(username: str, password: str) -> Tuple[bool, str]:
    """Verify credentials. Returns (ok, message)."""
    ok, msg = validate_username(username)
    if not ok:
        return False, "Invalid username or password."
    row = db.get_user(username)
    if not row or not _verify(password, row["password_hash"]):
        return False, "Invalid username or password."
    return True, "Welcome back."
