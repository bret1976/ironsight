"""Outbound email — SMTP when configured, otherwise writes to data/mail_outbox/."""
from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from app.core.config import get_settings, ROOT

log = logging.getLogger("ironsight.mailer")


def _outbox() -> Path:
    d = get_settings().data_dir / "mail_outbox"
    d.mkdir(parents=True, exist_ok=True)
    return d


def send_email(
    to: str,
    subject: str,
    body: str,
    *,
    html: Optional[str] = None,
) -> dict:
    s = get_settings()
    host = getattr(s, "smtp_host", "") or ""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = getattr(s, "smtp_from", None) or "noreply@ironsight.local"
    msg["To"] = to
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")

    record = {
        "to": to,
        "subject": subject,
        "body": body,
        "at": datetime.now(timezone.utc).isoformat(),
        "via": "smtp" if host else "outbox",
    }

    if host:
        try:
            port = int(getattr(s, "smtp_port", 587) or 587)
            user = getattr(s, "smtp_user", "") or ""
            password = getattr(s, "smtp_password", "") or ""
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls()
                if user:
                    smtp.login(user, password)
                smtp.send_message(msg)
            record["ok"] = True
            log.info("email sent to %s via SMTP", to)
            return record
        except Exception as e:
            record["ok"] = False
            record["error"] = str(e)
            log.warning("SMTP failed, writing outbox: %s", e)

    # Dev / fallback: durable outbox file (operator can read reset links)
    path = _outbox() / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{to.replace('@','_')}.json"
    path.write_text(json.dumps(record, indent=2))
    record["ok"] = True
    record["outbox_path"] = str(path)
    log.info("email written to outbox %s", path)
    return record
