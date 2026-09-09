"""
Capital.com trading bot.

Two halves: a strategy that proposes trades, and a risk engine that vetoes them.
The risk engine is the important half. A mediocre strategy with hard risk limits
survives; a good strategy without them eventually does not.

Defaults are deliberately timid: demo environment, dry run on, 1% risk per trade,
3% daily loss limit, mandatory stop on every position.

Trades a LIST of instruments, each with its own stop distance and RSI band,
because the broker's minimum deal size differs wildly between them: BTCUSD
can be traded in 0.0001 lots, FX pairs not below 100 units. At a 1% stop,
100 EURUSD lose $1.16 when stopped -- three times the risk cap on a $37
account -- so FX needs a tighter stop just to be permitted, and its spread
is cheap enough (0.006%) that a 0.3% stop still costs less in spread than
BTC does at 1%.

Run:  python3 bot.py --once      (one evaluation pass, prints what it would do)
      python3 bot.py             (loop forever at POLL_SECONDS)
"""

import os
import sys
import json
import time
import logging
import math
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, Set, Tuple

from capital_client import CapitalClient, CapitalError
from notify import send_alert
import news

# ------------------------------------------------------------------ config

RESOLUTION = os.getenv("CAPITAL_RESOLUTION", "MINUTE_5")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))

# Fallbacks for any epic not in INSTRUMENTS below.
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
STOP_DISTANCE_PCT = float(os.getenv("STOP_DISTANCE_PCT", "1.0"))

# Per-instrument settings. Stops are sized so the broker's MINIMUM position
# fits inside the 1% risk cap on the current balance (measured 7 Sep 2026:
# EURUSD 100u @0.3% = $0.35, GBPUSD 100u needs 0.25% = $0.34, AUDUSD $0.22,
# USDJPY $0.30, cap $0.37). RSI bands come from backtest.py over 166 days of
# 4-hour candles: 40/60 was positive on EURUSD/GBPUSD/AUDUSD, 30/70 was the
# least-bad row on BTCUSD and USDJPY. ~100 trades each -- directional, not
# proof of an edge. Re-run backtest.py before trusting any of these.
INSTRUMENTS: Dict[str, Dict[str, float]] = {
    "BTCUSD": {"stop_pct": 1.0,  "oversold": 30, "overbought": 70},
    "EURUSD": {"stop_pct": 0.3,  "oversold": 40, "overbought": 60},
    "GBPUSD": {"stop_pct": 0.25, "oversold": 40, "overbought": 60},
    "AUDUSD": {"stop_pct": 0.3,  "oversold": 40, "overbought": 60},
    "USDJPY": {"stop_pct": 0.3,  "oversold": 30, "overbought": 70},
}
# BTCUSD and USDJPY were removed 9 Sep 2026 (both negative in every backtest
# configuration; BTC also ate half the free margin per trade). Their rows stay
# in INSTRUMENTS so CAPITAL_EPICS can bring them back deliberately.
DEFAULT_EPICS = "EURUSD,GBPUSD,AUDUSD"


def _epics_from_env() -> List[str]:
    """CAPITAL_EPICS (list) wins; CAPITAL_EPIC (single) is the legacy fallback."""
    raw = os.getenv("CAPITAL_EPICS", "").strip()
    if not raw:
        raw = os.getenv("CAPITAL_EPIC", "").strip() or DEFAULT_EPICS
    seen: List[str] = []
    for part in raw.split(","):
        epic = part.strip().upper()
        if epic and epic not in seen:
            seen.append(epic)
    return seen


EPICS = _epics_from_env()


def instrument_config(epic: str) -> Dict[str, float]:
    cfg = INSTRUMENTS.get(epic.upper())
    if cfg:
        return cfg
    return {"stop_pct": STOP_DISTANCE_PCT, "oversold": RSI_OVERSOLD,
            "overbought": RSI_OVERBOUGHT}


# Risk. These are the numbers that matter.
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
REWARD_TO_RISK = float(os.getenv("REWARD_TO_RISK", "1.5"))

DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() != "false"
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))

# ---- overnight policy
# Capital.com charges financing daily, and the SIGN differs per instrument and
# per side, and it moves (EURUSD shorts paid on 7 Sep, were paid on 9 Sep). So
# the rule reads the broker's live rate every time and never assumes.
#
# Whether to PAY it is arithmetic, not principle. On a $450 FX position the
# charge is 3-4 cents a night; re-entering tomorrow costs ~3 cents of spread
# and resets a trade that needs days to reach a 4-hour-bar target. Flattening
# FX nightly was the single biggest live-vs-backtest divergence found in the
# 9 Sep audit: the backtest held through nights, the bot did not. So: hold
# anything that pays LESS than OVERNIGHT_MAX_PAY_PCT per night; flatten only
# the expensive ones (BTC longs at 0.06%/day are eight times the threshold).
# Anything that EARNS is always held. Fridays are different: FX is shut from
# 21:00 UTC to Sunday 21:00 and Monday can open anywhere, so everything on a
# market that closes for the weekend is flattened, whatever it pays.
SESSION_CUTOFF_UTC = os.getenv("SESSION_CUTOFF_UTC", "20:00")
NO_NEW_TRADES_MINS_BEFORE_CUTOFF = int(
    os.getenv("NO_NEW_TRADES_MINS_BEFORE_CUTOFF", "60")
)
# When the broker actually takes the financing charge. Read per instrument from
# its swapChargeTimestamp; this is only the fallback. Measured 21:00 UTC on all
# five instruments (7 Sep 2026). The flatten is pointless AFTER this moment --
# the charge is already taken -- so the flatten window is [cutoff, swap) and
# the no-new-payers window is [cutoff - NO_NEW_TRADES_MINS, swap). The first
# live night ran a pass at 22:07, closed two positions that had been charged
# at 21:00, paid the spread for nothing and re-opened one of them. Hence this.
SWAP_CHARGE_UTC = os.getenv("SWAP_CHARGE_UTC", "21:00")
# Hold a paying position overnight if it pays no more than this (% of
# notional per night). Measured 9 Sep 2026: EURUSD long 0.016%, GBPUSD ~0.004%,
# AUDUSD ~0.005%, BTCUSD long 0.062%. 0.02 keeps every FX pair, drops BTC.
OVERNIGHT_MAX_PAY_PCT = float(os.getenv("OVERNIGHT_MAX_PAY_PCT", "0.02"))
# Positions at or above this size are held regardless ("bigger trades"). 0 = off.
OVERNIGHT_HOLD_MIN_SIZE = float(os.getenv("OVERNIGHT_HOLD_MIN_SIZE", "0"))
HOLD_PAID_OVERNIGHT = (
    os.getenv("HOLD_PAID_OVERNIGHT", os.getenv("HOLD_SHORTS_OVERNIGHT", "true"))
    .strip().lower() != "false"
)
# Flatten everything on a weekend-closing market before Friday's 21:00 UTC close.
WEEKEND_FLATTEN = os.getenv("WEEKEND_FLATTEN", "true").strip().lower() != "false"

# ---- re-entry
# After a position in an instrument closes (stop, target, flatten -- anything),
# do not open another in it for this long. One 4-hour bar. The backtester never
# re-opens on the bar that closed a trade; the live bot, polling every 30 min,
# re-entered USDJPY four minutes after a stop-out on 9 Sep. Same RSI reading,
# second loss, spread paid twice. This closes that gap.
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "240"))

# ---- correlation
# EURUSD, GBPUSD and AUDUSD all move with the dollar (roughly 0.6-0.8), and the
# RSI fires on the same dollar move for all three at once. Three "independent"
# 1% positions in the same direction are one ~3% bet, which is exactly the
# daily halt. So at most this many positions on the same side of USD.
MAX_SAME_USD_SIDE = int(os.getenv("MAX_SAME_USD_SIDE", "2"))

# ---- news
# Stops are not guaranteed; NFP or an ECB surprise can gap 50 pips through a
# 35-pip stop. Cheapest mitigation: do not OPEN inside a window around
# high-impact events for the instrument's currencies. Open positions keep
# their stops -- this is not a reason to flatten. Fails OPEN if the calendar
# is unreachable (a nicety must not halt the bot), and says so in the log.
NEWS_FILTER = os.getenv("NEWS_FILTER", "true").strip().lower() != "false"
NEWS_BLACKOUT_MINUTES = int(os.getenv("NEWS_BLACKOUT_MINUTES", "30"))

# ---- kill switch
# The strategy's parameters are frozen from this moment. Every trade the
# broker reports closed after it counts, in R (profit / 1% of balance). If
# after KILL_AFTER_TRADES the running total is at or below KILL_BELOW_R, the
# bot stops opening positions and says why on every pass until a human
# changes this line. Decide the exit before the entry -- for the system too.
# Rebuilt from the broker's own history every pass, so a lost cache cannot
# reset it. Change STRATEGY_FROZEN_AT only when you change the strategy.
STRATEGY_FROZEN_AT = os.getenv("STRATEGY_FROZEN_AT", "2026-09-09T08:00:00")
KILL_AFTER_TRADES = int(os.getenv("KILL_AFTER_TRADES", "60"))
KILL_BELOW_R = float(os.getenv("KILL_BELOW_R", "-5"))

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")

log = logging.getLogger("bot")


def _setup_logging() -> None:
    """Only main() writes bot.log; importing this module (backtest.py) must not."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
    )


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
    # UTC on purpose: the runner is UTC, the cutoff is UTC, the swap is UTC.
    return dt.datetime.utcnow().date().isoformat()


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

    def veto(self, equity: float, open_positions: int) -> Optional[str]:
        """
        Return a reason string if trading must not happen, else None.

        `equity` is balance plus open P&L. Measuring the day on realised
        balance alone let three underwater positions not count until they
        closed, which could admit a fourth trade into a bad day.
        """
        day = self._day()
        start = day.get("start_balance")

        if start:
            drawdown_pct = (start - equity) / start * 100.0
            if drawdown_pct >= DAILY_LOSS_LIMIT_PCT:
                return (
                    "daily loss limit hit: down %.2f%% today on equity (limit %.2f%%). "
                    "No more trades until tomorrow."
                    % (drawdown_pct, DAILY_LOSS_LIMIT_PCT)
                )

        if open_positions >= MAX_CONCURRENT_POSITIONS:
            return "already holding %d position(s), max is %d" % (
                open_positions,
                MAX_CONCURRENT_POSITIONS,
            )

        if equity <= 0:
            return "account equity is zero or negative"

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
    balance: float, entry: float, stop: float, quote_factor: float = 1.0
) -> Tuple[float, float]:
    """
    Size so that being stopped out costs exactly RISK_PER_TRADE_PCT of balance.

    This is the whole point: the stop distance determines the size, not a fixed
    lot. `quote_factor` converts one unit of the instrument's quote currency
    into the account currency (1.0 for BTCUSD/EURUSD, ~1/154 for USDJPY --
    the price moves in yen, the balance is in dollars). Returns (size, cash_at_risk).
    """
    risk_cash = balance * (RISK_PER_TRADE_PCT / 100.0)
    stop_distance = abs(entry - stop) * quote_factor
    if stop_distance <= 0:
        raise ValueError("stop distance must be positive")
    return risk_cash / stop_distance, risk_cash


def quote_factor(
    epic: str, info: Dict[str, Any], price: float, account_ccy: str
) -> Optional[float]:
    """
    How many units of the account currency one unit of the instrument's
    quote currency is worth. Read from the broker's instrument.currency,
    never assumed from the epic name.

      quote == account (BTCUSD, EURUSD, GBPUSD, AUDUSD)  -> 1.0
      epic is ACCOUNT/quote (USDJPY, USDCHF, USDCAD)     -> 1 / price
      anything else (EURJPY, EURGBP on a USD account)    -> None: refused
                                                            rather than guessed.
    """
    quote = ((info.get("instrument", {}) or {}).get("currency") or "").upper()
    account_ccy = (account_ccy or "").upper()
    if not quote or not account_ccy:
        return None
    if quote == account_ccy:
        return 1.0
    if epic.upper().startswith(account_ccy) and price:
        return 1.0 / float(price)
    return None


# ----------------------------------------------------------------- strategy


def signal(closes: List[float], epic: str, cfg: Dict[str, float]) -> Optional[str]:
    """
    RSI mean reversion with a per-instrument band. Backtested (backtest.py)
    but not proven -- ~100 trades per instrument over 166 days is a direction,
    not an edge. Replace the body once you have something you actually believe.
    """
    value = rsi(closes, RSI_PERIOD)
    if value is None:
        log.info("%s: not enough candles for RSI(%d)", epic, RSI_PERIOD)
        return None

    log.info("%s: RSI(%d) = %.1f  (band %.0f/%.0f)", epic, RSI_PERIOD, value,
             cfg["oversold"], cfg["overbought"])

    if value <= cfg["oversold"]:
        return "BUY"
    if value >= cfg["overbought"]:
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


def _hhmm_today(hhmm: str, now: dt.datetime) -> dt.datetime:
    hour, _, minute = hhmm.partition(":")
    return now.replace(hour=int(hour), minute=int(minute or 0), second=0, microsecond=0)


def swap_time_today(info: Dict[str, Any], now: dt.datetime) -> dt.datetime:
    """
    The moment the broker charges overnight financing, today, UTC. Taken from
    the instrument's own swapChargeTimestamp (time of day only); falls back
    to SWAP_CHARGE_UTC if the field is missing.
    """
    try:
        ts = (info.get("instrument", {}) or {}).get("overnightFee", {}).get("swapChargeTimestamp")
        if ts:
            t = dt.datetime.utcfromtimestamp(float(ts) / 1000.0)
            return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return _hhmm_today(SWAP_CHARGE_UTC, now)


def in_flatten_window(info: Dict[str, Any], now: Optional[dt.datetime] = None) -> bool:
    """[cutoff, swap): close payers now, the charge has not been taken yet."""
    now = now or dt.datetime.utcnow()
    return _cutoff_today(now) <= now < swap_time_today(info, now)


def in_no_open_window(info: Dict[str, Any], now: Optional[dt.datetime] = None) -> bool:
    """[cutoff - N min, swap): do not open a position that would just be flattened."""
    now = now or dt.datetime.utcnow()
    start = _cutoff_today(now) - dt.timedelta(minutes=NO_NEW_TRADES_MINS_BEFORE_CUTOFF)
    return start <= now < swap_time_today(info, now)


def _unpack(raw: Dict) -> Tuple[str, str, float, str]:
    """Capital.com nests position data; tolerate both shapes."""
    pos = raw.get("position", raw)
    market = raw.get("market", {}) or {}
    return (
        pos.get("dealId", ""),
        (pos.get("direction") or "").upper(),
        float(pos.get("size") or 0),
        (market.get("epic") or pos.get("epic") or "").upper(),
    )


def overnight_rate(
    client: CapitalClient, epic: str, direction: str, cache: Dict[str, Dict]
) -> Optional[float]:
    """
    The broker's daily financing rate for holding `direction` in `epic`.
    Positive = you are paid to hold, negative = you pay. None if unreadable.
    """
    try:
        info = cache.get(epic)
        if info is None:
            info = client.market(epic)
            cache[epic] = info
        fee = (info.get("instrument", {}) or {}).get("overnightFee", {}) or {}
        key = "longRate" if direction == "BUY" else "shortRate"
        rate = fee.get(key)
        return None if rate is None else float(rate)
    except (CapitalError, TypeError, ValueError):
        return None


def closes_for_weekend(info: Dict[str, Any]) -> bool:
    """True if the broker lists no Saturday hours for this market (FX does; crypto does not)."""
    hours = (info.get("instrument", {}) or {}).get("openingHours") or {}
    return not hours.get("sat")


def overnight_decision(
    rate: Optional[float], size: Optional[float], info: Dict[str, Any],
    now: Optional[dt.datetime] = None,
) -> Tuple[bool, str]:
    """
    (hold, reason) for carrying a position through tonight's financing charge.
    One source of truth, used both to flatten before the swap and to refuse
    opening something that would just be flattened. `size` None skips the
    size rule (we are deciding about a position that does not exist yet).
    """
    now = now or dt.datetime.utcnow()
    if WEEKEND_FLATTEN and now.weekday() == 4 and closes_for_weekend(info):
        return False, "market closes for the weekend at 21:00 UTC; Monday can gap"
    if rate is not None and rate > 0 and HOLD_PAID_OVERNIGHT:
        return True, "financing is a credit (%+.4f%%/day)" % rate
    if size is not None and OVERNIGHT_HOLD_MIN_SIZE and size >= OVERNIGHT_HOLD_MIN_SIZE:
        return True, "size %.4f >= %.4f hold threshold" % (size, OVERNIGHT_HOLD_MIN_SIZE)
    if rate is None:
        return False, "financing rate unreadable; not carrying an unknown cost"
    if -rate <= OVERNIGHT_MAX_PAY_PCT:
        return True, "pays %.4f%%/day, under the %.3f%% threshold" % (-rate, OVERNIGHT_MAX_PAY_PCT)
    return False, "pays %.4f%%/day, over the %.3f%% threshold" % (-rate, OVERNIGHT_MAX_PAY_PCT)


def usd_side(epic: str, direction: str) -> Optional[str]:
    """
    Which side of the US dollar a position is on. BUY EURUSD and BUY AUDUSD
    are both SHORT_USD; BUY USDJPY is LONG_USD. None for pairs without USD.
    """
    epic = epic.upper()
    if epic.startswith("USD"):
        return "LONG_USD" if direction == "BUY" else "SHORT_USD"
    if epic.endswith("USD"):
        return "SHORT_USD" if direction == "BUY" else "LONG_USD"
    return None


def manage_overnight(
    client: CapitalClient, positions: List[Dict], cache: Optional[Dict[str, Dict]] = None
) -> int:
    """
    Flatten positions that should not be carried through the financing charge.

    Per position: look up the broker's overnight rate for that side. If it is
    a credit, hold (unless HOLD_PAID_OVERNIGHT is off). If it is a charge,
    close -- unless the position is big enough (OVERNIGHT_HOLD_MIN_SIZE) to
    be worth paying for. Returns the number of positions closed.
    """
    if not positions:
        return 0

    cache = cache if cache is not None else {}
    closed = 0
    now = dt.datetime.utcnow()

    for raw in positions:
        deal_id, direction, size, epic = _unpack(raw)
        if not deal_id:
            continue

        rate = overnight_rate(client, epic, direction, cache)
        info = cache.get(epic) or {}

        if not in_flatten_window(info, now):
            # Either the session is still open, or the charge has already
            # been taken -- closing now would pay the spread for nothing.
            continue

        hold, why = overnight_decision(rate, size, info, now)
        if hold:
            log.info("overnight: holding %s %s %s (size %.4f) - %s",
                     direction, epic, deal_id, size, why)
            continue

        log.info("overnight: closing %s %s %s (size %.4f) - %s",
                 direction, epic, deal_id, size, why)
        if DRY_RUN:
            log.info("DRY RUN - not closing")
            continue
        try:
            client.close_position(deal_id)
            closed += 1
            send_alert(
                "Trading bot: closed %s before overnight" % epic,
                [
                    "Flattened ahead of the %s UTC cutoff: %s." % (SESSION_CUTOFF_UTC, why),
                    "",
                    "instrument %s" % epic,
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
                "Trading bot: FAILED to close %s before overnight" % epic,
                [
                    "The overnight flatten did NOT succeed. This position will",
                    "carry through the financing charge unless you close it",
                    "manually in the platform.",
                    "",
                    "instrument %s" % epic,
                    "direction  %s" % direction,
                    "size       %.4f" % size,
                    "deal       %s" % deal_id,
                    "error      %s" % exc,
                ],
            )
    return closed


def evaluate_epic(
    client: CapitalClient,
    risk: RiskEngine,
    epic: str,
    balance: float,
    account_ccy: str,
    cache: Dict[str, Dict],
    usd_exposure: Dict[str, int],
) -> Optional[str]:
    """
    One instrument, one pass. Returns the direction if a trade was opened (or,
    in dry run, would have been) so the caller can count it against the
    position cap and the same-side-of-USD cap. None otherwise.
    """
    cfg = instrument_config(epic)
    stop_pct = float(cfg["stop_pct"])

    closes = client.closes(epic, RESOLUTION, count=max(RSI_PERIOD * 4, 60))
    if len(closes) < 2:
        log.warning("%s: no price data", epic)
        return None

    # The last candle the broker returns is the one still forming. Deciding on
    # it means the RSI repaints eight times inside a 4-hour bar and the bot
    # trades dips that vanish when the bar closes. The backtester only ever
    # sees completed bars; so, now, does the bot.
    completed = closes[:-1]
    direction = signal(completed, epic, cfg)
    if not direction:
        log.info("%s: no signal", epic)
        return None

    side = usd_side(epic, direction)
    if side and usd_exposure.get(side, 0) >= MAX_SAME_USD_SIDE:
        log.info("%s: %s signal ignored - already %d position(s) %s, cap %d "
                 "(the dollar pairs are one bet, not three)",
                 epic, direction, usd_exposure[side], side, MAX_SAME_USD_SIDE)
        return None

    try:
        info = cache.get(epic)
        if info is None:
            info = client.market(epic)
            cache[epic] = info
        rules = info.get("dealingRules", {})
        min_size = float(rules.get("minDealSize", {}).get("value", 0) or 0)
        step = float(rules.get("minSizeIncrement", {}).get("value", 0) or 0)
    except (CapitalError, TypeError, ValueError):
        info, min_size, step = {}, 0.0, 0.0

    # Do not open something that the overnight routine would close within the
    # hour. Same decision function as the flatten, so they cannot disagree.
    if in_no_open_window(info):
        rate = overnight_rate(client, epic, direction, cache)
        hold, why = overnight_decision(rate, None, info)
        if not hold:
            log.info("%s: %s signal ignored - would be flattened within the hour (%s)",
                     epic, direction, why)
            return None

    if NEWS_FILTER:
        blocked = news.blackout(epic, NEWS_BLACKOUT_MINUTES)
        if blocked:
            log.info("%s: %s signal ignored - %s", epic, direction, blocked)
            return None

    # Size and place the stop off the LIVE price we will actually be filled at,
    # not the last candle close. The first live USDJPY fill was 15 pips away
    # from the candle, which put the true risk a few cents over the cap.
    snap = info.get("snapshot", {}) or {}
    live = snap.get("offer") if direction == "BUY" else snap.get("bid")
    entry = float(live) if live else closes[-1]
    if live and abs(entry - closes[-1]) / closes[-1] > 0.0005:
        log.info("%s: candle close %.5f vs live %s %.5f - using live",
                 epic, closes[-1], "offer" if direction == "BUY" else "bid", entry)

    if direction == "BUY":
        stop = entry * (1 - stop_pct / 100.0)
        target = entry + (entry - stop) * REWARD_TO_RISK
    else:
        stop = entry * (1 + stop_pct / 100.0)
        target = entry - (stop - entry) * REWARD_TO_RISK

    # Price moves are in the QUOTE currency; the balance is in the ACCOUNT
    # currency. Sizing in the wrong one is off by the exchange rate (154x on
    # USDJPY), so refuse the instrument outright if we cannot convert.
    factor = quote_factor(epic, info, entry, account_ccy)
    if factor is None:
        log.warning(
            "RISK VETO - %s quotes in %s and this account is in %s; no conversion "
            "rule for that pair, so it is not traded.",
            epic, (info.get("instrument", {}) or {}).get("currency"), account_ccy,
        )
        return None

    size, risk_cash = position_size(balance, entry, stop, factor)

    # Sizes must be a whole number of the broker's step (100 units on FX,
    # 0.0001 on BTC). Round DOWN: a smaller position risks less than the cap,
    # never more. The minimum-size check below then decides whether what is
    # left is even sendable.
    if step:
        size = round(math.floor(size / step + 1e-9) * step, 8)
        risk_cash = size * abs(entry - stop) * factor

    # The broker has a minimum size and a step. An "ideal" size below the
    # minimum cannot be sent -- it must be rounded up to the minimum, which
    # means risking MORE than the cap allows. That is a veto, not a rounding
    # detail: silently taking extra risk is how small accounts die.
    if min_size and size < min_size:
        actual_loss = min_size * abs(entry - stop) * factor
        log.warning(
            "%s sizing: ideal %.4f is below broker minimum %.4f. "
            "Minimum position would risk %.2f, cap is %.2f.",
            epic, size, min_size, actual_loss, risk_cash,
        )
        if actual_loss > risk_cash:
            log.warning(
                "RISK VETO - cannot open %s without exceeding the %.1f%% risk cap "
                "(would risk %.2f of a %.2f balance = %.2f%%). "
                "Fund the account or tighten this instrument's stop_pct deliberately.",
                epic, RISK_PER_TRADE_PCT, actual_loss, balance,
                actual_loss / balance * 100.0,
            )
            return None
        size = min_size
        risk_cash = actual_loss

    log.info(
        "SIGNAL %s %s  entry=%.5f stop=%.5f target=%.5f size=%.4f risking=%.2f (%.1f%% of balance)",
        direction, epic, entry, stop, target, size, risk_cash,
        risk_cash / balance * 100.0 if balance else 0.0,
    )

    if DRY_RUN:
        log.info("DRY RUN - not sending the order. Set DRY_RUN=false to arm.")
        return direction

    detail = [
        "%s %s" % (direction, epic),
        "",
        "entry    %.5f" % entry,
        "stop     %.5f  (-%.2f%%)" % (stop, stop_pct),
        "target   %.5f  (%.1f:1)" % (target, REWARD_TO_RISK),
        "size     %.4f" % size,
        "risking  %.2f of %.2f balance (%.2f%%)"
        % (risk_cash, balance, risk_cash / balance * 100.0 if balance else 0.0),
        "",
        "RSI(%d)  %.1f  (band %.0f/%.0f)"
        % (RSI_PERIOD, rsi(completed, RSI_PERIOD) or 0.0, cfg["oversold"], cfg["overbought"]),
        "account  %s" % ("DEMO" if client.is_demo else "LIVE"),
    ]

    try:
        result = client.open_position(
            epic=epic,
            direction=direction,
            size=size,
            stop_level=stop,
            profit_level=target,
        )
    except CapitalError as exc:
        log.error("ORDER REJECTED: %s", exc)
        send_alert(
            "Trading bot: ORDER REJECTED (%s)" % epic,
            ["The broker refused the order.", "", str(exc), ""] + detail,
        )
        return None

    risk.record_trade()
    log.info("order sent: %s", result)
    send_alert("Trading bot: opened %s %s" % (direction, epic), detail)
    return direction


def _parse_tx_time(t: Dict[str, Any]) -> Optional[dt.datetime]:
    raw = (t.get("dateUtc") or t.get("dateUTC") or "")[:19]
    try:
        return dt.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def closed_trades_since(client: CapitalClient, since: dt.datetime) -> Optional[List[Dict[str, Any]]]:
    """Closed-trade transactions from the broker since `since` (UTC). None if unreadable."""
    try:
        tx = client.transactions(since.strftime("%Y-%m-%dT%H:%M:%S"))
    except CapitalError as exc:
        log.warning("could not read trade history: %s", exc)
        return None
    return [t for t in tx if "closed" in (t.get("note") or "").lower()]


def scoreboard(closed: List[Dict[str, Any]], balance: float) -> Optional[str]:
    """
    Every trade closed since STRATEGY_FROZEN_AT, scored in R against 1% of
    the current balance (the risk each was sized to, near enough while the
    balance is stable). Logs the running total on every pass so the record is
    always in the log, and returns a veto once the kill rule is met.
    """
    risk_cash = balance * RISK_PER_TRADE_PCT / 100.0
    realised = sum(float(t.get("size") or 0) for t in closed)
    total_r = realised / risk_cash if risk_cash else 0.0
    n = len(closed)
    log.info("record since %s: %d trades closed, %+.2f realised = %+.1fR  "
             "(kill rule: <= %+.0fR after %d trades)",
             STRATEGY_FROZEN_AT[:10], n, realised, total_r, KILL_BELOW_R, KILL_AFTER_TRADES)
    if n >= KILL_AFTER_TRADES and total_r <= KILL_BELOW_R:
        return ("KILL SWITCH: %d trades since %s total %+.1fR, at or below %+.0fR. "
                "The strategy has failed its own test. No new positions until "
                "STRATEGY_FROZEN_AT is changed by a human." % (n, STRATEGY_FROZEN_AT[:10],
                                                               total_r, KILL_BELOW_R))
    return None


def evaluate_once(client: CapitalClient, risk: RiskEngine) -> None:
    account = client.account()
    bal = account.get("balance", {}) or {}
    balance = float(bal.get("balance", 0.0))
    upl = float(bal.get("profitLoss") or 0.0)
    equity = balance + upl
    account_ccy = (account.get("currency") or "USD").upper()
    risk.mark_session_start(equity)
    open_positions = client.positions()
    cache: Dict[str, Dict] = {}   # market() lookups, one per epic per pass
    now = dt.datetime.utcnow()

    log.info(
        "balance=%.2f  open P&L=%+.2f  equity=%.2f  available=%.2f  open=%d  instruments=%s  mode=%s%s",
        balance, upl, equity, float(bal.get("available", 0.0)), len(open_positions),
        ",".join(EPICS), "DEMO" if client.is_demo else "LIVE",
        "  [DRY RUN]" if DRY_RUN else "",
    )

    # Session housekeeping before anything else: close what should not be
    # carried overnight. If anything was closed, re-read so the cap is right.
    if manage_overnight(client, open_positions, cache):
        open_positions = client.positions()

    # One history read serves three things: the kill switch, the cooldown,
    # and a running scoreboard in every log.
    try:
        frozen_at = dt.datetime.strptime(STRATEGY_FROZEN_AT[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        frozen_at = now
    closed = closed_trades_since(client, min(frozen_at, now - dt.timedelta(minutes=COOLDOWN_MINUTES)))
    cooling: Set[str] = set()
    killed: Optional[str] = None
    if closed is not None:
        cutoff = now - dt.timedelta(minutes=COOLDOWN_MINUTES)
        for t in closed:
            when = _parse_tx_time(t)
            if when and when >= cutoff:
                cooling.add((t.get("instrumentName") or "").upper())
        killed = scoreboard([t for t in closed if (_parse_tx_time(t) or now) >= frozen_at], balance)
    else:
        log.warning("trade history unreadable this pass: no cooldown, no kill check")

    blocked = killed or risk.veto(equity, len(open_positions))
    if blocked:
        log.error("RISK VETO - %s", blocked) if killed else log.warning("RISK VETO - %s", blocked)
        # Persistent conditions alert once a day, not every 30 minutes.
        if killed and risk.first_time_today("kill_switch"):
            send_alert("Trading bot: KILL SWITCH TRIPPED - no new trades",
                       [blocked, "", "Open positions keep their broker-side stop and target."])
        elif "daily loss limit" in blocked and risk.first_time_today("daily_loss"):
            send_alert(
                "Trading bot: DAILY LOSS LIMIT HIT - trading halted",
                [
                    blocked,
                    "",
                    "equity    %.2f" % equity,
                    "limit     %.1f%%" % DAILY_LOSS_LIMIT_PCT,
                    "",
                    "No further trades will be opened today. Open positions keep",
                    "their broker-side stop and target.",
                ],
            )
        return

    held: Set[str] = set()
    usd_exposure: Dict[str, int] = {}
    for raw in open_positions:
        _, direction, _, epic = _unpack(raw)
        held.add(epic)
        side = usd_side(epic, direction)
        if side:
            usd_exposure[side] = usd_exposure.get(side, 0) + 1
    open_count = len(open_positions)

    for epic in EPICS:
        # One position per instrument, and a hard cap across all of them. The
        # cap is re-checked after every order so a single pass cannot overshoot.
        if open_count >= MAX_CONCURRENT_POSITIONS:
            log.info("position cap %d reached - not evaluating further instruments",
                     MAX_CONCURRENT_POSITIONS)
            break
        if epic in held:
            log.info("%s: already holding a position - skipping", epic)
            continue
        if epic in cooling:
            log.info("%s: a position closed here within the last %d min - cooling down",
                     epic, COOLDOWN_MINUTES)
            continue
        try:
            opened = evaluate_epic(client, risk, epic, balance, account_ccy, cache, usd_exposure)
            if opened:
                open_count += 1
                held.add(epic)
                side = usd_side(epic, opened)
                if side:
                    usd_exposure[side] = usd_exposure.get(side, 0) + 1
        except CapitalError as exc:
            # One instrument's API trouble must not stop the others.
            log.error("%s: api error: %s", epic, exc)


def preflight(client: CapitalClient) -> None:
    """
    Answer one question before any money moves: can this account actually open
    the smallest permitted position in each instrument, and does that respect
    the risk cap?

    Reads the broker's own dealing rules rather than assuming them. This is the
    check that catches "the FX minimum is 100 units" before the risk engine
    quietly vetoes every trade.
    """
    account = client.account()
    balance = float(account.get("balance", {}).get("balance", 0.0))
    available = float(account.get("balance", {}).get("available", 0.0))
    account_ccy = (account.get("currency") or "USD").upper()
    risk_cash = balance * (RISK_PER_TRADE_PCT / 100.0)

    print("")
    print("  environment      : %s" % ("DEMO" if client.is_demo else "LIVE"))
    print("  balance          : %.2f %s" % (balance, account_ccy))
    print("  available        : %.2f" % available)
    print("  risk per trade   : %.2f (%.1f%% of balance)" % (risk_cash, RISK_PER_TRADE_PCT))
    print("  instruments      : %s" % ", ".join(EPICS))

    for epic in EPICS:
        cfg = instrument_config(epic)
        stop_pct = float(cfg["stop_pct"])
        print("")
        print("  --- %s  (stop %.2f%%, RSI %.0f/%.0f)"
              % (epic, stop_pct, cfg["oversold"], cfg["overbought"]))

        try:
            info = client.market(epic)
        except CapitalError as exc:
            print("  could not read market details: %s" % exc)
            continue

        rules = info.get("dealingRules", {})
        snapshot = info.get("snapshot", {})
        instrument = info.get("instrument", {})

        min_size = rules.get("minDealSize", {}).get("value")
        bid, offer = snapshot.get("bid"), snapshot.get("offer")
        price = offer or bid
        margin_pct = instrument.get("marginFactor")
        fee = instrument.get("overnightFee", {}) or {}

        print("  min deal size    : %s" % (min_size if min_size is not None else "unknown"))
        print("  price            : %s" % (price if price is not None else "unknown"))
        if bid and offer:
            print("  spread           : %.4f%%" % (100.0 * (float(offer) - float(bid)) / float(price)))
        print("  margin factor    : %s%%" % (margin_pct if margin_pct is not None else "?"))
        if fee:
            print("  overnight        : long %s%%  short %s%%  (+ = paid to hold)"
                  % (fee.get("longRate"), fee.get("shortRate")))

        verdict = []

        factor = quote_factor(epic, info, float(price or 0), account_ccy)
        quote = instrument.get("currency")
        if factor is None:
            verdict.append(
                "NOT TRADEABLE HERE: quotes in %s, account is %s, and there is no "
                "conversion rule for that pair." % (quote, account_ccy)
            )
            price = None   # skip the money maths below; they would be in the wrong currency
        elif factor != 1.0:
            print("  quote currency   : %s  (1 %s = %.6f %s)" % (quote, quote, factor, account_ccy))

        if min_size is not None and price is not None and margin_pct:
            notional = float(min_size) * float(price) * factor
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

        if min_size is not None and price is not None:
            # Loss if the minimum position is stopped out at this instrument's stop.
            min_loss = float(min_size) * float(price) * factor * (stop_pct / 100.0)
            print("  loss if stopped  : %.2f (minimum position, %.2f%% stop)" % (min_loss, stop_pct))
            if min_loss > risk_cash:
                verdict.append(
                    "RISK CAP EXCEEDED: the smallest position loses %.2f if stopped, but "
                    "your %.1f%% cap allows only %.2f. Every trade would be vetoed."
                    % (min_loss, RISK_PER_TRADE_PCT, risk_cash)
                )
            else:
                verdict.append("risk cap OK (%.2f of %.2f)." % (min_loss, risk_cash))

        for line in verdict:
            print("  -> %s" % line)
    print("")


def status(client: CapitalClient, hours: int = 24) -> None:
    """Balance, open positions, and what closed recently -- straight from the broker."""
    account = client.account()
    bal = account.get("balance", {})
    ccy = account.get("currency", "")
    print("")
    print("  %s account   balance %.2f %s   available %.2f   open P&L %+.2f"
          % ("DEMO" if client.is_demo else "LIVE", float(bal.get("balance", 0)), ccy,
             float(bal.get("available", 0)), float(bal.get("profitLoss", 0) or 0)))

    positions = client.positions()
    print("")
    print("  open positions: %d  (cap %d)" % (len(positions), MAX_CONCURRENT_POSITIONS))
    for raw in positions:
        pos = raw.get("position", raw)
        mkt = raw.get("market", {}) or {}
        direction = (pos.get("direction") or "").upper()
        now_px = mkt.get("bid") if direction == "BUY" else mkt.get("offer")
        print("    %-7s %-4s size %-6s entry %-10s now %-10s stop %-10s target %-10s P&L %+.2f"
              % (mkt.get("epic"), direction, pos.get("size"), pos.get("level"), now_px,
                 pos.get("stopLevel"), pos.get("profitLevel"), float(pos.get("upl") or 0)))

    since = (dt.datetime.utcnow() - dt.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        tx = client.transactions(since)
    except CapitalError as exc:
        print("\n  could not read history: %s" % exc)
        return
    closed = [t for t in tx if "closed" in (t.get("note") or "").lower()]
    fees = sum(float(t.get("size") or 0) for t in tx if "fee" in (t.get("note") or "").lower())
    realised = sum(float(t.get("size") or 0) for t in closed)
    print("")
    print("  last %dh: %d trades closed, realised %+.2f, financing %+.2f"
          % (hours, len(closed), realised, fees))
    for t in reversed(closed):
        print("    %s  %-7s %+.2f" % ((t.get("date") or "")[:16], t.get("instrumentName"),
                                     float(t.get("size") or 0)))
    print("")
    print("  Runs: gh run list --workflow=trade.yml --limit 10")
    print("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capital.com trading bot")
    parser.add_argument(
        "--status", action="store_true",
        help="balance, open positions, last 24h of closed trades, then exit",
    )
    parser.add_argument(
        "--once", action="store_true", help="single evaluation pass, then exit"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="report whether this account can trade each instrument at all, then exit",
    )
    parser.add_argument(
        "--search",
        metavar="TERM",
        help="search tradeable instruments by name, print their epics, then exit",
    )
    args = parser.parse_args()

    _setup_logging()

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

    if args.status:
        status(client)
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
                    deal_id, direction, size, epic = _unpack(raw)
                    log.warning("   open: %s %s %s size=%.4f", direction, epic, deal_id, size)
            else:
                log.info("no open positions - account is flat.")
        except CapitalError as exc:
            log.error("could not check open positions on exit: %s", exc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
