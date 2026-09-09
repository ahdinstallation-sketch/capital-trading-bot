#!/usr/bin/env python3
"""
Backtest the RSI rule on real candles from the account's own price feed.

Built to answer one question: "how do I get more trades?"

Trade COUNT is easy to raise -- loosen the thresholds and signals appear.
Trade VALUE is the thing that matters, and on BTCUSD the spread is charged
in full on every single trade. So this reports both, side by side, and
lets the numbers say whether more trades is more money or just more spread.

Everything below is measured, not assumed:
  * entries use the ASK, exits use the BID -- the real spread, per candle,
    from the broker's own feed. No modelled cost constant.
  * if a candle's range touches BOTH the stop and the target, it is scored
    as a LOSS. We cannot see intrabar order, so we take the bad branch.
  * no position is opened while one is open (MAX_CONCURRENT_POSITIONS=1),
    which is how the live bot actually behaves.
  * protective stops (break-even, trailing) are armed at the CLOSE of the
    bar that earned them and bite from the next bar. The live bot polls
    every 30 minutes inside a 4-hour bar, so it would often react sooner;
    this is the pessimistic reading.

Results are in R -- multiples of the money risked per trade. +1R is one
winning trade's worth of risk. At 1% risk on a $135 account, 1R = $1.35.

    python backtest.py --epic EURUSD --resolution HOUR_4   # threshold grid
    python backtest.py --epic EURUSD,GBPUSD --protect       # stop-protection grid
    python backtest.py --epic EURUSD --single 40 60         # one config, every trade
"""

import argparse
import sys
from typing import Any, Dict, List, Optional

from capital_client import CapitalClient
from bot import INSTRUMENTS, instrument_config


# ------------------------------------------------------------------ helpers


def rsi_series(closes: List[float], period: int) -> List[Optional[float]]:
    """
    Wilder's RSI at every bar, so the walk-forward loop does not recompute
    the whole series 1,000 times. Index i holds the RSI as known at the
    close of bar i -- element i is None until there is enough history.
    """
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return out

    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return out


def _px(candle: Dict[str, Any], field: str, side: str) -> Optional[float]:
    v = (candle.get(field) or {}).get(side)
    return None if v is None else float(v)


class Bar(object):
    """One candle, keeping bid and ask apart so the spread stays honest."""

    __slots__ = ("time", "mid_close", "bid_high", "bid_low", "ask_high",
                 "ask_low", "ask_open", "bid_open", "spread")

    def __init__(self, candle: Dict[str, Any]):
        self.time = candle.get("snapshotTime", "")
        cb, ca = _px(candle, "closePrice", "bid"), _px(candle, "closePrice", "ask")
        self.mid_close = (cb + ca) / 2.0
        self.bid_high = _px(candle, "highPrice", "bid")
        self.bid_low = _px(candle, "lowPrice", "bid")
        self.ask_high = _px(candle, "highPrice", "ask")
        self.ask_low = _px(candle, "lowPrice", "ask")
        self.ask_open = _px(candle, "openPrice", "ask")
        self.bid_open = _px(candle, "openPrice", "bid")
        self.spread = ca - cb


def load_bars(candles: List[Dict[str, Any]]) -> List[Bar]:
    bars = []
    for c in candles:
        try:
            bar = Bar(c)
        except TypeError:
            continue  # incomplete candle, skip it
        if None in (bar.bid_high, bar.bid_low, bar.ask_high,
                    bar.ask_low, bar.ask_open, bar.bid_open):
            continue
        bars.append(bar)
    return bars


# ----------------------------------------------------------------- the test


class Trade(object):
    __slots__ = ("side", "entry", "stop0", "stop", "target", "best", "opened",
                 "opened_idx", "closed", "exit_px", "result_r", "outcome",
                 "bars_held")


def run(bars: List[Bar], period: int, oversold: float, overbought: float,
        stop_pct: float, reward_to_risk: float,
        breakeven_at: Optional[float] = None,
        trail: Optional[float] = None) -> List[Trade]:
    """
    Walk the candles once, left to right, opening and closing trades exactly
    the way bot.py would. No lookahead: the decision at bar i uses only
    closes up to and including bar i, and the fill happens at bar i+1's open.

    breakeven_at: once the trade is this many R in profit, move the stop to
                  the entry price. The trade can then end at 0R instead of -1R.
    trail:        keep the stop this many R behind the best price seen. It
                  only ever tightens. trail=1.0 starts identical to the fixed
                  stop and rises with price; smaller values are tighter.
    """
    closes = [b.mid_close for b in bars]
    rsis = rsi_series(closes, period)

    trades: List[Trade] = []
    open_trade: Optional[Trade] = None

    for i in range(len(bars) - 1):
        nxt = bars[i + 1]

        # --- manage the open position first; one position at a time
        if open_trade is not None:
            t = open_trade
            risk = abs(t.entry - t.stop0)
            if t.side == "BUY":
                # A long exits by SELLING, so the stop and target sit on the bid.
                hit_stop = nxt.bid_low <= t.stop
                hit_target = nxt.bid_high >= t.target
            else:
                # A short exits by BUYING, so both sit on the ask.
                hit_stop = nxt.ask_high >= t.stop
                hit_target = nxt.ask_low <= t.target

            if hit_stop or hit_target:
                # Both touched in one candle: we cannot know which came
                # first, so we take the loss. Optimism here is how
                # backtests lie.
                t.exit_px = t.stop if hit_stop else t.target
                gain = (t.exit_px - t.entry) if t.side == "BUY" else (t.entry - t.exit_px)
                t.result_r = gain / risk if risk else 0.0
                if not hit_stop:
                    t.outcome = "TARGET"
                elif t.stop == t.stop0:
                    t.outcome = "STOP"
                else:
                    t.outcome = "TRAIL" if t.result_r > 1e-9 else "BE"
                t.closed = nxt.time
                t.bars_held = (i + 1) - t.opened_idx
                trades.append(t)
                open_trade = None
                continue  # never open on the same bar we just closed

            # --- still open: tighten the protective stop off THIS bar's best
            # price. Takes effect on the next bar, never this one.
            if t.side == "BUY":
                t.best = nxt.bid_high if t.best is None else max(t.best, nxt.bid_high)
                if breakeven_at is not None and t.best >= t.entry + breakeven_at * risk:
                    t.stop = max(t.stop, t.entry)
                if trail is not None:
                    t.stop = max(t.stop, t.best - trail * risk)
            else:
                t.best = nxt.ask_low if t.best is None else min(t.best, nxt.ask_low)
                if breakeven_at is not None and t.best <= t.entry - breakeven_at * risk:
                    t.stop = min(t.stop, t.entry)
                if trail is not None:
                    t.stop = min(t.stop, t.best + trail * risk)
            continue

        # --- look for a new entry
        value = rsis[i]
        if value is None:
            continue
        if value <= oversold:
            side = "BUY"
        elif value >= overbought:
            side = "SELL"
        else:
            continue

        t = Trade()
        t.side = side
        if side == "BUY":
            t.entry = nxt.ask_open           # we buy at the ask
            t.stop0 = t.entry * (1 - stop_pct / 100.0)
            t.target = t.entry * (1 + stop_pct * reward_to_risk / 100.0)
        else:
            t.entry = nxt.bid_open           # we sell at the bid
            t.stop0 = t.entry * (1 + stop_pct / 100.0)
            t.target = t.entry * (1 - stop_pct * reward_to_risk / 100.0)
        t.stop = t.stop0
        t.best = None
        t.opened = nxt.time
        t.opened_idx = i + 1
        t.outcome = "OPEN"
        t.result_r = 0.0
        open_trade = t

    return trades


def summarise(trades: List[Trade], days: float) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "per_day": 0.0, "wins": 0, "win_pct": 0.0, "be": 0,
                "total_r": 0.0, "avg_r": 0.0}
    wins = sum(1 for t in trades if t.result_r > 1e-9)
    be = sum(1 for t in trades if t.outcome == "BE")
    total = sum(t.result_r for t in trades)
    return {
        "n": len(trades),
        "per_day": len(trades) / days if days else 0.0,
        "wins": wins,
        "win_pct": 100.0 * wins / len(trades),
        "be": be,
        "total_r": total,
        "avg_r": total / len(trades),
    }


# ---------------------------------------------------------------------- CLI

RESOLUTION_MINUTES = {"MINUTE": 1, "MINUTE_5": 5, "MINUTE_15": 15, "MINUTE_30": 30,
                      "HOUR": 60, "HOUR_4": 240, "DAY": 1440}

# The protection variants worth knowing about. (label, breakeven_at, trail)
PROTECT_GRID = (
    ("fixed stop (live now)",     None, None),
    ("break-even at +0.5R",       0.5,  None),
    ("break-even at +0.75R",      0.75, None),
    ("break-even at +1.0R",       1.0,  None),
    ("trail 1.0R behind best",    None, 1.0),
    ("trail 0.75R behind best",   None, 0.75),
    ("BE +0.75R, then trail 1R",  0.75, 1.0),
)


def test_epic(client: CapitalClient, epic: str, args, balance: float) -> None:
    live_cfg = instrument_config(epic)
    stop_pct = args.stop_pct if args.stop_pct is not None else float(live_cfg["stop_pct"])
    live_band = (int(live_cfg["oversold"]), int(live_cfg["overbought"]))

    bars = load_bars(client.candles(epic, args.resolution, args.count))
    if len(bars) < args.period + 10:
        print("%s: not enough candles came back (%d). Nothing to test." % (epic, len(bars)))
        return

    minutes = RESOLUTION_MINUTES.get(args.resolution, 5)
    days = len(bars) * minutes / 1440.0
    avg_spread = sum(b.spread for b in bars) / len(bars)
    avg_price = sum(b.mid_close for b in bars) / len(bars)
    risk_cash = balance * args.risk_pct / 100.0

    # The single most important number on the page. The stop distance is
    # what you risk; the spread is what you pay to find out. Their ratio is
    # the tax on every trade, before the strategy has done anything at all.
    stop_distance = avg_price * stop_pct / 100.0
    spread_r = avg_spread / stop_distance

    print("")
    print("%s  %s  %d candles  ~%.1f days" % (epic, args.resolution, len(bars), days))
    print("avg price %.5g   avg spread %.5g (%.4f%% of price)"
          % (avg_price, avg_spread, 100.0 * avg_spread / avg_price))
    print("stop %.2f%% = %.5g points   ->  SPREAD COSTS %.3fR PER TRADE"
          % (stop_pct, stop_distance, spread_r))
    print("1R = $%.2f at %.1f%% risk on $%.2f" % (risk_cash, args.risk_pct, balance))

    # The bar the strategy has to clear. At reward:risk R, a coin-flip entry
    # breaks even at 1/(1+R) wins. The spread widens every loss and shortens
    # every win, so the real bar sits above the textbook one. If the measured
    # win% below does not beat this number, the rule is losing on purpose and
    # trading it more often only gets there faster.
    rr = args.reward_to_risk
    textbook = 100.0 / (1.0 + rr)
    with_spread = 100.0 * (1.0 + spread_r) / ((rr - spread_r) + (1.0 + spread_r))
    print("BREAKEVEN WIN RATE at %.1f:1 = %.1f%%  (%.1f%% once the spread is paid)"
          % (rr, textbook, with_spread))
    print("")

    if args.single:
        oversold, overbought = args.single
        trades = run(bars, args.period, oversold, overbought, stop_pct, rr,
                     args.breakeven_at, args.trail)
        print("RSI(%d) %.0f/%.0f -- every trade:" % (args.period, oversold, overbought))
        print("")
        for t in trades:
            print("  %-20s %-4s entry %10.5g  exit %10.5g  %-6s %+6.2fR  $%+6.2f"
                  % (t.opened[:19], t.side, t.entry, t.exit_px, t.outcome,
                     t.result_r, t.result_r * risk_cash))
        s = summarise(trades, days)
        print("")
        print("  %d trades, %.1f/day, %d wins (%.0f%%), %d break-even, total %+.2fR = $%+.2f"
              % (s["n"], s["per_day"], s["wins"], s["win_pct"], s["be"],
                 s["total_r"], s["total_r"] * risk_cash))
        return

    if args.protect:
        oversold, overbought = live_band
        print("RSI(%d) %d/%d, stop %.2f%%, target %.1fR -- how each protection rule changes the result:"
              % (args.period, oversold, overbought, stop_pct, rr))
        print("")
        print("%-28s %6s %5s %6s %7s %8s %9s" %
              ("rule", "trades", "win%", "BE", "avg R", "total R", "P&L $"))
        print("-" * 75)
        for label, be_at, tr in PROTECT_GRID:
            trades = run(bars, args.period, float(oversold), float(overbought),
                         stop_pct, rr, be_at, tr)
            s = summarise(trades, days)
            print("%-28s %6d %5.0f %6d %+7.2f %+8.2f %+9.2f"
                  % (label, s["n"], s["win_pct"], s["be"], s["avg_r"],
                     s["total_r"], s["total_r"] * risk_cash))
        return

    print("%-12s %7s %8s %7s %9s %10s" %
          ("RSI band", "trades", "per day", "win%", "total R", "P&L $"))
    print("-" * 58)
    for oversold, overbought in ((20, 80), (25, 75), (30, 70), (35, 65),
                                 (40, 60), (45, 55)):
        trades = run(bars, args.period, float(oversold), float(overbought),
                     stop_pct, rr, args.breakeven_at, args.trail)
        s = summarise(trades, days)
        marker = "   <- live now" if (oversold, overbought) == live_band else ""
        print("%-12s %7d %8.1f %7.0f %+9.2f %+10.2f%s"
              % ("%d/%d" % (oversold, overbought), s["n"], s["per_day"],
                 s["win_pct"], s["total_r"], s["total_r"] * risk_cash, marker))

    print("")
    print("Trades go up as the band widens. Whether MONEY goes up is the")
    print("column on the right, and it is the only one worth reading.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epic", default="BTCUSD",
                    help="one epic, or several comma-separated (one login for all)")
    ap.add_argument("--resolution", default="HOUR_4")
    ap.add_argument("--count", type=int, default=1000,
                    help="candles to fetch (broker caps this, usually 1000)")
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--stop-pct", type=float, default=None,
                    help="default: the instrument's entry in bot.INSTRUMENTS")
    ap.add_argument("--reward-to-risk", type=float, default=1.5)
    ap.add_argument("--balance", type=float, default=None,
                    help="default: the account's live balance")
    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--breakeven-at", type=float, default=None, metavar="R",
                    help="move stop to entry once this many R in profit")
    ap.add_argument("--trail", type=float, default=None, metavar="R",
                    help="trail the stop this many R behind the best price")
    ap.add_argument("--protect", action="store_true",
                    help="compare the protection rules on the live band")
    ap.add_argument("--single", nargs=2, type=float, metavar=("OVERSOLD", "OVERBOUGHT"),
                    help="test one threshold pair and print every trade")
    args = ap.parse_args()

    # ONE login for the whole run. Capital.com rate-limits session creation,
    # and a shell loop of separate invocations is exactly what drew a 429
    # against the live bot on 9 Sep 2026.
    client = CapitalClient()
    client.login()   # read-only from here on; no arming needed to backtest
    balance = args.balance if args.balance is not None else client.balance()

    for epic in [e.strip().upper() for e in args.epic.split(",") if e.strip()]:
        test_epic(client, epic, args, balance)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
