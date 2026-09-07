"""
Email alerts for the trading bot.

Deliberately matches the SMTP conventions already used by marketing-bot
(MAIL_USER / MAIL_PASSWORD, host inferred from the sender domain, 3 attempts
with backoff) so there is one way to send mail across the projects rather
than two.

Alerts fire on EVENTS ONLY -- an order placed, an order rejected, the daily
loss limit tripping, an overnight flatten. Never on "no signal", which happens
48 times a day and would train you to ignore the emails.
"""

import os
import ssl
import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Sequence

log = logging.getLogger("notify")

SMTP_ATTEMPTS = 3
SMTP_BACKOFF = (5, 15, 45)


def _smtp_settings(user: str):
    dom = user.split("@")[-1].lower()
    if dom in ("gmail.com", "googlemail.com"):
        return "smtp.gmail.com", 465
    if dom in ("outlook.com", "hotmail.com", "live.com", "msn.com"):
        return "smtp-mail.outlook.com", 587
    return "smtp.office365.com", 587


def _recipients() -> List[str]:
    raw = os.environ.get("ALERT_RECIPIENTS", "") or os.environ.get("MAIL_USER", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


def _deliver(host, port, user, pw, recipients, msg) -> None:
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=60) as server:
            server.login(user, pw)
            server.sendmail(user, recipients, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=60) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(user, pw)
            server.sendmail(user, recipients, msg.as_string())


def send_alert(subject: str, lines: Sequence[str]) -> bool:
    """
    Send one alert. Returns False (never raises) on any failure -- a mail
    problem must never take down the trading loop or mask a trade result.
    """
    user = (os.environ.get("MAIL_USER") or "").strip()
    pw = (os.environ.get("MAIL_PASSWORD") or "").replace(" ", "").strip()
    to = _recipients()

    if not user or not pw or not to:
        log.info("mail not configured (MAIL_USER/MAIL_PASSWORD/ALERT_RECIPIENTS) - skipping alert")
        return False

    host, port = _smtp_settings(user)
    body_text = "\n".join(lines)
    body_html = (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:14px;line-height:1.6">'
        + "<br>".join(html_escape(l) for l in lines)
        + "</div>"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = "AHD Trading Bot <%s>" % user
    msg["To"] = ", ".join(to)
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    for attempt in range(1, SMTP_ATTEMPTS + 1):
        try:
            _deliver(host, port, user, pw, to, msg)
            log.info("alert sent to %s", ", ".join(to))
            return True
        except smtplib.SMTPAuthenticationError as exc:
            # A bad app password cannot be fixed by retrying, and repeated
            # failed logins make Gmail harden further.
            log.error("SMTP auth rejected for %s - not retrying: %s", user, exc)
            return False
        except (smtplib.SMTPException, OSError) as exc:
            if attempt == SMTP_ATTEMPTS:
                log.error("SMTP failed after %d attempts: %s", attempt, exc)
                return False
            wait = SMTP_BACKOFF[min(attempt - 1, len(SMTP_BACKOFF) - 1)]
            log.warning("SMTP attempt %d failed (%s) - retrying in %ds", attempt, exc, wait)
            time.sleep(wait)
    return False


def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
