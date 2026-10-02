"""
Thin client for the Capital.com public API.

Safety design: the environment is chosen by CAPITAL_ENV and defaults to "demo".
Pointing this at real money takes two deliberate env vars, not one, so that a
typo or a stray shell export can never silently move from demo to live.

Credentials are read from the environment only. Never hardcode them here and
never paste them into a chat window.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

DEMO_BASE = "https://demo-api-capital.backend-capital.com"
LIVE_BASE = "https://api-capital.backend-capital.com"

# A session token is good for ~10 minutes of inactivity. Refresh well before.
SESSION_TTL_SECONDS = 540
LOGIN_ATTEMPTS = 3
LOGIN_BACKOFF = (15, 30)   # seconds; total 45s, well inside the 5-min job timeout


class CapitalError(RuntimeError):
    pass


def resolve_base_url() -> str:
    """Pick the demo or live endpoint. Connecting is not the dangerous part."""
    env = os.getenv("CAPITAL_ENV", "demo").strip().lower()

    if env == "demo":
        return DEMO_BASE
    if env == "live":
        return LIVE_BASE

    raise CapitalError("CAPITAL_ENV must be 'demo' or 'live', got %r" % env)


def is_armed() -> bool:
    """
    True when the operator has explicitly consented to real-money orders.

    This gates ORDER PLACEMENT only, not connecting or reading. Reading a
    balance on a live account is harmless; sending an order is not. Gating the
    connection instead would push people to arm real money just to run a
    read-only check, which is exactly backwards.
    """
    return (
        os.getenv("CAPITAL_I_UNDERSTAND_THIS_IS_REAL_MONEY", "").strip().lower()
        == "yes"
    )


class CapitalClient:
    def __init__(self) -> None:
        self.base_url = resolve_base_url()
        self.is_demo = self.base_url == DEMO_BASE

        self.api_key = os.getenv("CAPITAL_API_KEY", "")
        self.identifier = os.getenv("CAPITAL_IDENTIFIER", "")
        self.password = os.getenv("CAPITAL_PASSWORD", "")

        missing = [
            name
            for name, value in (
                ("CAPITAL_API_KEY", self.api_key),
                ("CAPITAL_IDENTIFIER", self.identifier),
                ("CAPITAL_PASSWORD", self.password),
            )
            if not value
        ]
        if missing:
            raise CapitalError(
                "Missing credentials in environment: %s. "
                "Copy .env.example to .env and fill it in." % ", ".join(missing)
            )

        self.session = requests.Session()
        self._cst: Optional[str] = None
        self._security_token: Optional[str] = None
        self._authenticated_at: float = 0.0

    # ---------------------------------------------------------------- auth

    def login(self) -> None:
        # Capital.com rate-limits session creation. Two triggers landing a
        # minute apart, or a batch of backtests, is enough to draw a 429.
        # That is transient, so wait it out rather than fail the whole pass.
        for attempt in range(1, LOGIN_ATTEMPTS + 1):
            resp = self.session.post(
                self.base_url + "/api/v1/session",
                headers={
                    "X-CAP-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"identifier": self.identifier, "password": self.password},
                timeout=20,
            )
            if resp.status_code == 429 and attempt < LOGIN_ATTEMPTS:
                wait = LOGIN_BACKOFF[min(attempt - 1, len(LOGIN_BACKOFF) - 1)]
                log.warning("Login rate-limited (429), attempt %d/%d - retrying in %ds",
                            attempt, LOGIN_ATTEMPTS, wait)
                time.sleep(wait)
                continue
            break
        if resp.status_code != 200:
            raise CapitalError(
                "Login failed (%s): %s" % (resp.status_code, resp.text[:300])
            )

        self._cst = resp.headers.get("CST")
        self._security_token = resp.headers.get("X-SECURITY-TOKEN")
        if not self._cst or not self._security_token:
            raise CapitalError("Login succeeded but session tokens were not returned.")

        self._authenticated_at = time.time()

        if self.is_demo:
            log.info("Authenticated against DEMO")
        elif is_armed():
            log.warning("=" * 62)
            log.warning("LIVE ACCOUNT, ORDERS ARMED - REAL MONEY IS AT RISK")
            log.warning("=" * 62)
        else:
            log.info("Authenticated against LIVE (read-only: orders not armed)")

    def _auth_headers(self) -> Dict[str, str]:
        if not self._cst or time.time() - self._authenticated_at > SESSION_TTL_SECONDS:
            self.login()
        return {
            "X-SECURITY-TOKEN": self._security_token or "",
            "CST": self._cst or "",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            resp = self.session.request(
                method,
                self.base_url + path,
                headers=self._auth_headers(),
                params=params,
                json=payload,
                timeout=20,
            )
        except requests.RequestException as exc:
            # Timeouts and DNS hiccups are broker-side failures as far as the
            # bot is concerned: one instrument's slow reply must not take the
            # whole pass down with a raw traceback.
            raise CapitalError("%s %s: network error: %s" % (method, path, exc))
        if resp.status_code >= 400:
            raise CapitalError(
                "%s %s failed (%s): %s"
                % (method, path, resp.status_code, resp.text[:300])
            )
        if not resp.text:
            return {}
        return resp.json()

    # ------------------------------------------------------------- account

    def accounts(self) -> List[Dict[str, Any]]:
        """Every trading account on this login, in the order the API returns."""
        data = self._request("GET", "/api/v1/accounts")
        accounts = data.get("accounts", [])
        if not accounts:
            raise CapitalError("No trading accounts returned.")
        return accounts

    def account(self) -> Dict[str, Any]:
        """
        The account this bot trades. Pin it with CAPITAL_ACCOUNT_ID.

        Unpinned, it is whichever account the API happens to list first, and
        that order is not ours to control: switching the active account in the
        Capital.com app can reorder it. On 2026-10-01 it did, and the bot spent
        a day reading a different, empty account -- reporting a balance of zero
        while the money sat untouched in the other one. It halted instead of
        trading, which is the right failure, but it should not have been able
        to happen quietly. Hence the warning below.
        """
        accounts = self.accounts()

        preferred = os.getenv("CAPITAL_ACCOUNT_ID", "").strip()
        if preferred:
            for acct in accounts:
                if acct.get("accountId") == preferred:
                    return acct
            raise CapitalError(
                "CAPITAL_ACCOUNT_ID %r not found. This login has: %s"
                % (preferred, ", ".join(a.get("accountId", "?") for a in accounts))
            )

        if len(accounts) > 1:
            log.warning(
                "%d accounts on this login and CAPITAL_ACCOUNT_ID is not set - "
                "trading %r (%s) because the API listed it first. Pin it: run "
                "`python bot.py --accounts` and set CAPITAL_ACCOUNT_ID.",
                len(accounts),
                accounts[0].get("accountId", "?"),
                (accounts[0].get("currency") or "?"),
            )
        return accounts[0]

    def balance(self) -> float:
        acct = self.account()
        return float(acct.get("balance", {}).get("balance", 0.0))

    def available(self) -> float:
        acct = self.account()
        return float(acct.get("balance", {}).get("available", 0.0))

    # -------------------------------------------------------------- market

    def candles(
        self, epic: str, resolution: str = "MINUTE_5", count: int = 100
    ) -> List[Dict[str, Any]]:
        data = self._request(
            "GET",
            "/api/v1/prices/%s" % epic,
            params={"resolution": resolution, "max": count},
        )
        return data.get("prices", [])

    def closes(self, epic: str, resolution: str = "MINUTE_5", count: int = 100
               ) -> List[float]:
        """Mid close prices, oldest first."""
        out = []
        for candle in self.candles(epic, resolution, count):
            close = candle.get("closePrice", {})
            bid, ask = close.get("bid"), close.get("ask")
            if bid is None or ask is None:
                continue
            out.append((float(bid) + float(ask)) / 2.0)
        return out

    def market(self, epic: str) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/markets/%s" % epic)

    def search_markets(self, term: str) -> List[Dict[str, Any]]:
        """Find tradeable epics by name. Read-only."""
        data = self._request("GET", "/api/v1/markets", params={"searchTerm": term})
        return data.get("markets", [])

    # ----------------------------------------------------------- positions

    def positions(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/api/v1/positions").get("positions", [])

    def transactions(self, since_iso: str) -> List[Dict[str, Any]]:
        """Account transactions (closed trades, financing, deposits) since a UTC ISO time."""
        return self._request(
            "GET", "/api/v1/history/transactions", params={"from": since_iso}
        ).get("transactions", [])

    def open_position(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_level: Optional[float] = None,
        profit_level: Optional[float] = None,
    ) -> Dict[str, Any]:
        if direction not in ("BUY", "SELL"):
            raise CapitalError("direction must be BUY or SELL")

        # The one place real money can actually leave. Everything else in this
        # client is a read.
        if not self.is_demo and not is_armed():
            raise CapitalError(
                "Refusing to place a LIVE order: "
                "CAPITAL_I_UNDERSTAND_THIS_IS_REAL_MONEY is not set to 'yes'. "
                "Uncomment that line in .env to arm real-money trading."
            )

        payload: Dict[str, Any] = {
            "epic": epic,
            "direction": direction,
            "size": size,
        }
        # A stop is not optional in this bot's risk model, but the API allows
        # placing without one, so we only include what the caller supplied.
        if stop_level is not None:
            payload["stopLevel"] = round(stop_level, 5)
        if profit_level is not None:
            payload["profitLevel"] = round(profit_level, 5)

        return self._request("POST", "/api/v1/positions", payload=payload)

    def close_position(self, deal_id: str) -> Dict[str, Any]:
        return self._request("DELETE", "/api/v1/positions/%s" % deal_id)
