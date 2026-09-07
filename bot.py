"""
Capital.com trading bot.

Two halves: a strategy that proposes trades, and a risk engine that vetoes them.
The risk engine is the important half. A mediocre strategy with hard risk limits
survives; a good strategy without them eventually does not.

Defaults are deliberately timid: demo environment, dry run on, 1% risk per trade,
3% daily loss limit, mandatory stop on every position.

Run:  python3 bot.py --once      (one evaluation pass, prints what it would do)
      python3 bot.py             (loop forever at POLL_SECONDS)
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime as dt
from typing import Dict, List, Optional, Tuple

from capital_client import CapitalClient, CapitalError
from notify import send_alert

# ------------------------------------------------------------------ config

EPIC = os.getenv("CAPITAL_EPIC", "GOLD")
RESOLUTION = os.getenv("CAPITAL_RESOLUTION", "MINUTE_5")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))

# Risk. These are the numbers that matter.
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "1"))
STOP_DISTANCE_PCT = float(os.getenv("STOP_DISTANCE_PCT", "1.0"))
REWARD_TO_RISK = float(os.getenv("REWARD_TO_RISK", "1.5"))

DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() != "false"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))

# ---- overnight policy
# Capital.com charges financing daily on open positions. On BTCUSD the long
# rate is negative (you pay) and the short rate positive (you are paid), so
# "flatten everything" would throw away a small credit on shorts. The policy
# is therefore: flatten longs before the cutoff unless the position is large
# enough to justify the fee; leave shorts alone.
SESSION_CUTOFF_UTC = os.getenv("SESSION_CUTOFF_UTC", "20:00")
NO_NEW_TRADES_MINS_BEFORE_CUTOFF = int(
    os.getenv("NO_NEW_TRADES_MINS_BEFORE_CUTOFF", "60")
)
# Positions at or above this size may be held overnight. 0 = never hold longs.
OVERNIGHT_HOLD_MIN_SIZE = float(os.getenv("OVERNIGHT_HOLD_MIN_SIZE", "0"))
HOLD_SHORTS_OVERNIGHT = os.getenv("HOLD_SHORTS_OVERNIGHT", "true").strip().lower() != "false"

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
log = logging.getLogger("bot")


# ----------------------------------------------------------------- indicator


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    """Wilder's RSI on the closing series. None if there is not enough data."""
    if len(values) < period + 1:
        return None

    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)

    avg_gain, avg_loss = gains / period, losses / period

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# --------------------------------------------------------------------- state


def load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        log.warning("state.json unreadable, starting fresh")
        return {}


def save_state(state: Dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_PATH)


def today_key() -> str:
    return dt.date.today().isoformat()


# ---------------------------------------------------------------- risk gate


class RiskEngine:
    """Every proposed trade passes through here. Any veto is final."""

    def __init__(self, client: CapitalClient) -> None:
        self.client = client
        self.state = load_state()

    def _day(self) -> Dict:
        key = today_key()
        day = self.state.get("day", {})
        if day.get("date") != key:
            day = {"date": key, "start_balance": None, "trades": 0}
            self.state["day"] = day
            save_state(self.state)
        return day

    def mark_session_start(self, balance: float) -> None:
        day = self._day()
        if day.get("start_balance") is None:
            day["start_balance"] = balance
            save_state(self.state)

    def veto(self, balance: float, open_positions: int) -> Optional[str]:
        """Return a reason string if trading must not happen, else None."""
        day = self._day()
        start = day.get("start_balance")

        if start:
            drawdown_pct = (start - balance) / start * 100.0
            if drawdown_pct >= DAILY_LOSS_LIMIT_PCT:
                return (
                    "daily loss limit hit: down %.2f%% today (limit %.2f%%). "
                    "No more trades until tomorrow."
                    % (drawdown_pct, DAILY_LOSS_LIMIT_PCT)
                )

        if open_positions >= MAX_CONCURRENT_POSITIONS:
            return "already holding %d position(s), max is %d" % (
                open_positions,
                MAX_CONCURRENT_POSITIONS,
            )

        if balance <= 0:
            return "account balance is zero or negative"

        return None

    def record_trade(self) -> None:
        day = self._day()
        day["trades"] = day.get("trades", 0) + 1
        save_state(self.state)

    def first_time_today(self, key: str) -> bool:
        """
        True only the first time `key` is seen today. Used so a condition that
        persists across every poll -- the daily loss halt, for instance -- sends
        one email rather than one every 30 minutes.
        """
        day = self._day()
        seen = day.setdefault("alerted", [])
        if key in seen:
            return False
        seen.append(key)
        save_state(self.state)
        return True


# ------------------------------------------------------------------- sizing


def position_size(
    balance: float, entry: float, stop: float
) -> Tuple[float, float]:
    """
    Size so that being stopped out costs exactly RISK_PER_TRADE_PCT of balance.

    This is the whole point: the stop distance determines the size, not a fixed
    lot. Returns (size, cash_at_risk).
    """
    risk_cash = balance * (RISK_PER_TRADE_PCT / 100.0)
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        raise ValueError("stop distance must be positive")
    return risk_cash / stop_distance, risk_cash


# ----------------------------------------------------------------- strategy


def signal(closes: List[float]) -> Optional[str]:
    """
    Starting skeleton: RSI mean reversion. This is a placeholder with no proven
    edge -- it is here so the plumbing can be tested end to end. Replace the
    body once you have backtested something you actually believe in.
    """
    value = rsi(closes, RSI_PERIOD)
    if value is None:
        log.info("not enough candles for RSI(%d)", RSI_PERIOD)
        return None

    log.info("RSI(%d) = %.1f", RSI_PERIOD, value)

    if value <= RSI_OVERSOLD:
        return "BUY"
    if value >= RSI_OVERBOUGHT:
        return "SELL"
    return None


# --------------------------------------------------------------------- loop


def _cutoff_today(now: dt.datetime) -> dt.datetime:
    hour, _, minute = SESSION_CUTOFF_UTC.partition(":")
    return now.replace(
        hour=int(hour), minute=int(minute or 0), second=0, microsecond=0
    )


def minutes_to_cutoff(now: Optional[dt.datetime] = None) -> float:
    """Minutes until today's session cutoff. Negative once it has passed."""
    now = now or dt.datetime.utcnow()
    return (_cutoff_today(now) - now).total_seconds() / 60.0


def _unpack(raw: Dict) -> Tuple[str, str, float]:
    """Capital.com nests position data; tolerate both shapes."""
    pos = raw.get("position", raw)
    return (
        pos.get("dealId", ""),
        (pos.get("direction") or "").upper(),
        float(pos.get("size") or 0),
    )


def manage_overnight(client: CapitalClient, positions: List[Dict]) -> None:
    """
    Flatten positions that should not be carried through the financing charge.

    Longs pay to hold, so they are closed unless big enough to be worth it.
    Shorts are paid to hold, so they stay (configurable).
    """
    if not positions or minutes_to_cutoff() > 0:
        return

    for raw in positions:
        deal_id, direction, size = _unpack(raw)
        if not deal_id:
            continue

        if direction == "SELL" and HOLD_SHORTS_OVERNIGHT:
            log.info(
                "overnight: holding short %s (size %.4f) - short financing is a credit",
                deal_id,
                size,
            )
            continue

        if OVERNIGHT_HOLD_MIN_SIZE and size >= OVERNIGHT_HOLD_MIN_SIZE:
            log.info(
                "overnight: holding %s %s (size %.4f >= %.4f threshold)",
                direction,
                deal_id,
                size,
                OVERNIGHT_HOLD_MIN_SIZE,
            )
            continue

        log.info(
            "overnight: closing %s %s (size %.4f) before cutoff %s UTC",
            direction,
            deal_id,
            size,
            SESSION_CUTOFF_UTC,
        )
        if DRY_RUN:
            log.info("DRY RUN - not closing")
            continue
        try:
            client.close_position(deal_id)
            send_alert(
                "Trading bot: closed %s before overnight" % EPIC,
                [
                    "Flattened ahead of the %s UTC cutoff to avoid the overnight"
                    % SESSION_CUTOFF_UTC,
                    "financing charge.",
                    "",
                    "direction  %s" % direction,
                    "size       %.4f" % size,
                    "deal       %s" % deal_id,
                ],
            )
        except CapitalError as exc:
            # A failed close means the position carries overnight unintentionally
            # -- exactly the thing this routine exists to prevent. Say so loudly.
            log.error("could not close %s: %s", deal_id, exc)
            send_alert(
                "Trading bot: FAILED to close %s before overnight" % EPIC,
                [
                    "The overnight flatten did NOT succeed. This position will",
                    "carry through the financing charge unless you close it",
                    "manually in the platform.",
                    "",
                    "direction  %s" % direction,
                    "size       %.4f" % size,
                    "deal       %s" % deal_id,
                    "error      %s" % exc,
                ],
            )


def evaluate_once(client: CapitalClient, risk: RiskEngine) -> None:
    balance = client.balance()
    risk.mark_session_start(balance)
    open_positions = client.positions()

    log.info(
        "balance=%.2f  available=%.2f  open=%d  mode=%s%s",
        balance,
        client.available(),
        len(open_positions),
        "DEMO" if client.is_demo else "LIVE",
        "  [DRY RUN]" if DRY_RUN else "",
    )

    # Session housekeeping before anything else: close what should not be
    # carried overnight.
    manage_overnight(client, open_positions)

    blocked = risk.veto(balance, len(open_positions))
    if blocked:
        log.warning("RISK VETO - %s", blocked)
        # The daily halt persists for the rest of the day, so alert once.
        if "daily loss limit" in blocked and risk.first_time_today("daily_loss"):
            send_alert(
                "Trading bot: DAILY LOSS LIMIT HIT - trading halted",
                [
                    blocked,
                    "",
                    "balance   %.2f" % balance,
                    "limit     %.1f%%" % DAILY_LOSS_LIMIT_PCT,
                    "",
                    "No further trades will be opened today. Open positions keep",
                    "their broker-side stop and target.",
                ],
            )
        return

    # Do not open anything that would immediately need closing at the cutoff.
    remaining = minutes_to_cutoff()
    if 0 < remaining <= NO_NEW_TRADES_MINS_BEFORE_CUTOFF:
        log.info(
            "session: %.0f min to cutoff (%s UTC) - no new trades",
            remaining,
            SESSION_CUTOFF_UTC,
        )
        return

    closes = client.closes(EPIC, RESOLUTION, count=max(RSI_PERIOD * 4, 60))
    if not closes:
        log.warning("no price data for %s", EPIC)
        return

    direction = signal(closes)
    if not direction:
        log.info("no signal")
        return

    entry = closes[-1]
    if direction == "BUY":
        stop = entry * (1 - STOP_DISTANCE_PCT / 100.0)
        target = entry + (entry - stop) * REWARD_TO_RISK
    else:
        stop = entry * (1 + STOP_DISTANCE_PCT / 100.0)
        target = entry - (stop - entry) * REWARD_TO_RISK

    size, risk_cash = position_size(balance, entry, stop)

    # The broker has a minimum size and a step. An "ideal" size below the
    # minimum cannot be sent -- it must be rounded up to the minimum, which
    # means risking MORE than the cap allows. That is a veto, not a rounding
    # detail: silently taking extra risk is how small accounts die.
    try:
        rules = client.market(EPIC).get("dealingRules", {})
        min_size = float(rules.get("minDealSize", {}).get("value", 0) or 0)
    except (CapitalError, TypeError, ValueError):
        min_size = 0.0

    if min_size and size < min_size:
        actual_loss = min_size * abs(entry - stop)
        log.warning(
            "sizing: ideal %.4f is below broker minimum %.4f. "
            "Minimum position would risk %.2f, cap is %.2f.",
            size,
            min_size,
            actual_loss,
            risk_cash,
        )
        if actual_loss > risk_cash:
            log.warning(
                "RISK VETO - cannot open %s without exceeding the %.1f%% risk cap "
                "(would risk %.2f of a %.2f balance = %.2f%%). "
                "Fund the account or widen RISK_PER_TRADE_PCT deliberately.",
                EPIC,
                RISK_PER_TRADE_PCT,
                actual_loss,
                balance,
                actual_loss / balance * 100.0,
            )
            return
        size = min_size
        risk_cash = actual_loss

    log.info(
        "SIGNAL %s %s  entry=%.4f stop=%.4f target=%.4f size=%.4f risking=%.2f (%.1f%% of balance)",
        direction,
        EPIC,
        entry,
        stop,
        target,
        size,
        risk_cash,
        risk_cash / balance * 100.0 if balance else 0.0,
    )

    if DRY_RUN:
        log.info("DRY RUN - not sending the order. Set DRY_RUN=false to arm.")
        return

    detail = [
        "%s %s" % (direction, EPIC),
        "",
        "entry    %.4f" % entry,
        "stop     %.4f  (-%.2f%%)" % (stop, STOP_DISTANCE_PCT),
        "target   %.4f  (%.1f:1)" % (target, REWARD_TO_RISK),
        "size     %.4f" % size,
        "risking  %.2f of %.2f balance (%.2f%%)"
        % (risk_cash, balance, risk_cash / balance * 100.0 if balance else 0.0),
        "",
        "RSI(%d)  %.1f" % (RSI_PERIOD, rsi(closes, RSI_PERIOD) or 0.0),
        "account  %s" % ("DEMO" if client.is_demo else "LIVE"),
    ]

    try:
        result = client.open_position(
            epic=EPIC,
            direction=direction,
            size=size,
            stop_level=stop,
            profit_level=target,
        )
    except CapitalError as exc:
        log.error("ORDER REJECTED: %s", exc)
        send_alert(
            "Trading bot: ORDER REJECTED (%s)" % EPIC,
            ["The broker refused the order.", "", str(exc), ""] + detail,
        )
        return

    risk.record_trade()
    log.info("order sent: %s", result)
    send_alert("Trading bot: opened %s %s" % (direction, EPIC), detail)


def preflight(client: CapitalClient) -> None:
    """
    Answer one question before any money moves: can this account actually open
    the smallest permitted position in EPIC, and does that respect the risk cap?

    Reads the broker's own dealing rules rather than assuming them.
    """
    balance = client.balance()
    available = client.available()

    print("")
    print("  environment      : %s" % ("DEMO" if client.is_demo else "LIVE"))
    print("  balance          : %.2f" % balance)
    print("  available        : %.2f" % available)
    print("  instrument       : %s" % EPIC)

    try:
        info = client.market(EPIC)
    except CapitalError as exc:
        print("\n  could not read market details: %s" % exc)
        return

    rules = info.get("dealingRules", {})
    snapshot = info.get("snapshot", {})
    instrument = info.get("instrument", {})

    min_size = rules.get("minDealSize", {}).get("value")
    price = snapshot.get("offer") or snapshot.get("bid")
    margin_pct = instrument.get("marginFactor")

    print("  min deal size    : %s" % (min_size if min_size is not None else "unknown"))
    print("  price            : %s" % (price if price is not None else "unknown"))
    print("  margin factor    : %s%%" % (margin_pct if margin_pct is not None else "?"))

    verdict = []

    if min_size is not None and price is not None and margin_pct:
        notional = float(min_size) * float(price)
        margin_needed = notional * float(margin_pct) / 100.0
        print("  min notional     : %.2f" % notional)
        print("  margin needed    : %.2f" % margin_needed)
        if margin_needed > available:
            verdict.append(
                "CANNOT TRADE: smallest position needs %.2f margin, you have %.2f "
                "available (short by %.2f)."
                % (margin_needed, available, margin_needed - available)
            )
        else:
            verdict.append("margin OK for the minimum position size.")

    risk_cash = balance * (RISK_PER_TRADE_PCT / 100.0)
    print("  risk per trade   : %.2f (%.1f%% of balance)" % (risk_cash, RISK_PER_TRADE_PCT))

    if min_size is not None and price is not None:
        # Loss if the minimum position is stopped out at STOP_DISTANCE_PCT.
        min_loss = float(min_size) * float(price) * (STOP_DISTANCE_PCT / 100.0)
        print("  loss if stopped  : %.2f (minimum position)" % min_loss)
        if min_loss > risk_cash:
            verdict.append(
                "RISK CAP EXCEEDED: the smallest position loses %.2f if stopped, but "
                "your %.1f%% cap allows only %.2f. Every trade would be vetoed."
                % (min_loss, RISK_PER_TRADE_PCT, risk_cash)
            )

    print("")
    if verdict:
        for line in verdict:
            print("  -> %s" % line)
    else:
        print("  -> no blocking issues found.")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capital.com trading bot")
    parser.add_argument(
        "--once", action="store_true", help="single evaluation pass, then exit"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="report whether this account can trade EPIC at all, then exit",
    )
    parser.add_argument(
        "--search",
        metavar="TERM",
        help="search tradeable instruments by name, print their epics, then exit",
    )
    args = parser.parse_args()

    try:
        client = CapitalClient()
        client.login()
    except CapitalError as exc:
        log.error("%s", exc)
        return 1

    risk = RiskEngine(client)

    if args.search:
        results = client.search_markets(args.search)
        if not results:
            print("no instruments matched %r" % args.search)
            return 0
        print("\n  %-22s %-38s %s" % ("EPIC", "NAME", "STATUS"))
        print("  " + "-" * 74)
        for market in results[:25]:
            print(
                "  %-22s %-38s %s"
                % (
                    market.get("epic", "?"),
                    (market.get("instrumentName", "?"))[:38],
                    market.get("marketStatus", "?"),
                )
            )
        print("")
        return 0

    if args.preflight:
        preflight(client)
        return 0

    if args.once:
        evaluate_once(client, risk)
        return 0

    log.info("polling every %ds - ctrl-c to stop", POLL_SECONDS)
    try:
        while True:
            try:
                evaluate_once(client, risk)
            except CapitalError as exc:
                log.error("api error: %s", exc)
            except Exception as exc:  # keep the loop alive on transient failures
                log.exception("unexpected error: %s", exc)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        # Stopping the bot does NOT close open positions -- their stops and
        # targets stay with the broker. Say so plainly rather than letting the
        # operator assume shutting down means flat.
        log.info("")
        log.info("stopped by user.")
        try:
            still_open = client.positions()
            if still_open:
                log.warning(
                    "%d position(s) STILL OPEN at the broker. Their stop/target "
                    "remain active, but this bot is no longer managing them "
                    "(no overnight flatten). Close them in the platform if you "
                    "do not want them running unattended.",
                    len(still_open),
                )
                for raw in still_open:
                    deal_id, direction, size = _unpack(raw)
                    log.warning("   open: %s %s size=%.4f", direction, deal_id, size)
            else:
                log.info("no open positions - account is flat.")
        except CapitalError as exc:
            log.error("could not check open positions on exit: %s", exc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
