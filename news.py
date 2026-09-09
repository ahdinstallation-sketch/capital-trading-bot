"""
High-impact economic events, so the bot does not open a position into one.

Source: the free ForexFactory weekly feed. No key, no account. Fetched once
per pass and cached in-process. If it cannot be fetched the filter fails OPEN
-- a calendar outage must not stop the bot -- and the log says so.

Stops on open positions stay where they are; this only gates NEW entries.
"""

import datetime as dt
import logging
from typing import Dict, List, Optional, Set

import requests

log = logging.getLogger("news")

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_cache: Dict[str, object] = {"events": None, "fetched": None}


def _currencies(epic: str) -> Set[str]:
    epic = epic.upper()
    if len(epic) == 6:
        return {epic[:3], epic[3:]}
    return {epic[-3:]}


def _events() -> Optional[List[Dict]]:
    if _cache["events"] is not None:
        return _cache["events"]  # type: ignore[return-value]
    try:
        resp = requests.get(FEED_URL, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("news calendar unavailable (%s) - filter is OFF this pass", exc)
        return None
    out = []
    for e in raw:
        if e.get("impact") != "High":
            continue
        try:
            when = dt.datetime.fromisoformat(e["date"]).astimezone(dt.timezone.utc).replace(tzinfo=None)
        except (KeyError, ValueError, TypeError):
            continue
        out.append({"when": when, "ccy": (e.get("country") or "").upper(),
                    "title": e.get("title") or ""})
    _cache["events"] = out
    log.info("news calendar: %d high-impact events this week", len(out))
    return out


def blackout(epic: str, minutes: int, now: Optional[dt.datetime] = None) -> Optional[str]:
    """A reason string if a high-impact event for this epic's currencies is within +/- minutes."""
    events = _events()
    if not events:
        return None
    now = now or dt.datetime.utcnow()
    window = dt.timedelta(minutes=minutes)
    wanted = _currencies(epic)
    for e in events:
        if e["ccy"] in wanted and abs(e["when"] - now) <= window:
            return "%s %s at %s UTC is within %d min" % (
                e["ccy"], e["title"], e["when"].strftime("%H:%M"), minutes)
    return None
