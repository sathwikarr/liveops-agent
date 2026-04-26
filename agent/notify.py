"""Unified notification fan-out: Slack + SMTP email.

Single public API:
    notify(subject, body, severity="medium", recipients=None) -> dict[str, bool]

Each channel is enabled if its env vars are present; missing channels are
silently skipped (returns False for that channel). Failures in one channel
don't break the others.

Channels:
- Slack: SLACK_WEBHOOK
- Email: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM,
         SMTP_TO (comma-separated default recipients), SMTP_USE_TLS=1

Severity thresholds:
- ALERT_EMAIL_MIN_SEVERITY (default 'high')  — emails below this are skipped
- ALERT_SLACK_MIN_SEVERITY (default 'low')   — Slack always except 'none'
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Iterable, Optional

import requests

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _at_least(severity: str, threshold: str) -> bool:
    return SEVERITY_ORDER.get(severity, 1) >= SEVERITY_ORDER.get(threshold, 0)


# --------------------------------------------------------------------------- #
# Slack
# --------------------------------------------------------------------------- #

def _slack_send(text: str) -> bool:
    webhook = os.getenv("SLACK_WEBHOOK")
    if not webhook:
        return False
    try:
        r = requests.post(webhook, json={"text": text}, timeout=5)
        return 200 <= r.status_code < 300
    except Exception as e:
        print(f"[notify:slack] error: {e}")
        return False


# --------------------------------------------------------------------------- #
# SMTP email
# --------------------------------------------------------------------------- #

def _smtp_config() -> Optional[dict]:
    host = os.getenv("SMTP_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASS", ""),
        "sender": os.getenv("SMTP_FROM") or os.getenv("SMTP_USER", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "1") not in ("0", "false", "False", ""),
        "default_to": [r.strip() for r in os.getenv("SMTP_TO", "").split(",") if r.strip()],
    }


def _email_send(subject: str, body: str, recipients: Iterable[str]) -> bool:
    cfg = _smtp_config()
    if not cfg:
        return False
    to_list = list(recipients) or cfg["default_to"]
    if not to_list:
        print("[notify:email] no recipients configured")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["sender"]
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    try:
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context(), timeout=10) as s:
                if cfg["user"]:
                    s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as s:
                s.ehlo()
                if cfg["use_tls"]:
                    s.starttls(context=ssl.create_default_context())
                    s.ehlo()
                if cfg["user"]:
                    s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        return True
    except Exception as e:
        print(f"[notify:email] error: {e}")
        return False


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #

def notify(
    subject: str,
    body: str,
    severity: str = "medium",
    recipients: Optional[Iterable[str]] = None,
) -> dict[str, bool]:
    """Fan-out to enabled channels. Returns {channel: succeeded?}."""
    severity = (severity or "medium").lower()
    results: dict[str, bool] = {"slack": False, "email": False}

    slack_min = os.getenv("ALERT_SLACK_MIN_SEVERITY", "low").lower()
    email_min = os.getenv("ALERT_EMAIL_MIN_SEVERITY", "high").lower()

    if _at_least(severity, slack_min):
        results["slack"] = _slack_send(f"*[{severity.upper()}]* {subject}\n{body}")

    if _at_least(severity, email_min):
        results["email"] = _email_send(
            subject=f"[LiveOps {severity.upper()}] {subject}",
            body=body,
            recipients=recipients or [],
        )

    return results
