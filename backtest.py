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
  * overnight financing is charged at the broker's live rate for the side
    held, every 21:00 UTC crossing, in R (rate / stop%). Positions that pay
    more than OVERNIGHT_MAX_PAY_PCT are flattened at the 20:00 bar, exactly
    as bot.py does; everything on a weekend-closing market is flattened on
    the last bar before a gap of 8h+ (Friday close, holidays).

Results are in R -- multiples of the money risked per trade. +1R is one
winning trade's worth of risk. At 1% risk on a $135 account, 1R = $1.35.

    python backtest.py --epic EURUSD --resolution HOUR_4   # threshold grid
    python backtest.py --epic EURUSD,GBPUSD --protect       # stop-protection grid
    python backtest.py --epic EURUSD --single 40 60         # one config, every trade
"""

import argparse
import datetime as dt
import sys
from typing import Any, Dict, List, Optional

from capital_client import CapitalClient, CapitalError
from bot import INSTRUMENTS, instrument_config, OVERNIGHT_MAX_PAY_PCT, REGIME_ADX_MAX, TREND_MODE, adx_series, dmi_series


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


def sma_series(values: List[float], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= n:
            run -= values[i - n]
        if i >= n - 1:
            out[i] = run / n
    return out


def _px(candle: Dict[str, Any], field: str, side: str) -> Optional[float]:
    v = (candle.get(field) or {}).get(side)
    return None if v is None else float(v)


class Bar(object):
    """One candle, keeping bid and ask apart so the spread stays honest."""

    __slots__ = ("time", "ts", "mid_close", "bid_close", "ask_close", "bid_high",
                 "bid_low", "ask_high", "ask_low", "ask_open", "bid_open", "spread")

    def __init__(self, candle: Dict[str, Any]):
        self.time = candle.get("snapshotTime", "")
        self.ts = dt.datetime.strptime(self.time[:19], "%Y-%m-%dT%H:%M:%S")
        cb, ca = _px(candle, "closePrice", "bid"), _px(candle, "closePrice", "ask")
        self.bid_close, self.ask_close = cb, ca
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
        except (TypeError, ValueError):
            continue  # incomplete candle, skip it
        if None in (bar.bid_high, bar.bid_low, bar.ask_high,
                    bar.ask_low, bar.ask_open, bar.bid_open):
            continue
        bars.append(bar)
    return bars


# ----------------------------------------------------------------- the test


class Trade(object):
    __slots__ = ("side", "entry", "stop0", "stop", "target", "best", "opened",
                 "opened_idx", "closed", "exit_px", "result_r", "fin_r", "outcome",
                 "bars_held")


class Costs(object):
    """What the broker charges for time, per instrument. Rates in %/day, + = paid to hold."""

    def __init__(self, long_rate: Optional[float] = None, short_rate: Optional[float] = None,
                 weekend_close: bool = False, bar_minutes: int = 240,
                 flatten_pay_over: float = OVERNIGHT_MAX_PAY_PCT):
        self.long_rate, self.short_rate = long_rate, short_rate
        self.weekend_close, self.bar_minutes = weekend_close, bar_minutes
        self.flatten_pay_over = flatten_pay_over

    def rate(self, side: str) -> Optional[float]:
        return self.long_rate if side == "BUY" else self.short_rate

    def flattens(self, side: str) -> bool:
        r = self.rate(side)
        return r is None or -r > self.flatten_pay_over


def _bar_contains(bar: Bar, hour: int, bar_minutes: int) -> bool:
    """Does [bar open, bar open + duration) contain today's HH:00 UTC?"""
    mark = bar.ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    return bar.ts <= mark < bar.ts + dt.timedelta(minutes=bar_minutes)


def run(bars: List[Bar], period: int, oversold: float, overbought: float,
        stop_pct: float, reward_to_risk: float,
        breakeven_at: Optional[float] = None,
        trail: Optional[float] = None,
        costs: Optional[Costs] = None,
        regime: Optional[str] = None,
        since: Optional[dt.datetime] = None,
        trend_mode: Optional[str] = None,
        trend_rr: Optional[float] = None,
        hours: Optional[List[int]] = None,
        atr_max_pctile: Optional[float] = None) -> List[Trade]:
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
    costs = costs or Costs()
    # Regime filters. "adx": only trade when ADX(14) < 25 (ranging market --
    # the only place mean reversion has a reason to work). "sma": only fade
    # toward the 50-bar mean, i.e. BUY only above SMA50, SELL only below.
    # "both": both conditions.
    sma50 = sma_series(closes, 50)
    # ATR(14) percentile over the trailing 200 bars: a volatility-regime gauge.
    trs = [0.0] + [max(bars[i].ask_high - bars[i].bid_low, abs(bars[i].ask_high - closes[i - 1]),
                       abs(bars[i].bid_low - closes[i - 1])) for i in range(1, len(bars))]
    atr = [None] * len(bars)
    a = None
    for i, t in enumerate(trs):
        if i == 14:
            a = sum(trs[1:15]) / 14
        elif i > 14:
            a = (a * 13 + t) / 14
        atr[i] = a
    adx14, pdi, mdi = dmi_series([(b.bid_high + b.ask_high) / 2 for b in bars],
                                 [(b.bid_low + b.ask_low) / 2 for b in bars], closes, 14)
    # trend_mode: what to do when ADX says the market is TRENDING (>= limit).
    #   None       stand aside (v2)
    #   "invert"   take the RSI signal the other way round, with the trend:
    #              RSI oversold in a downtrend = SELL (sell weakness), and v.v.
    #   "pullback" enter with the trend when RSI pulls back to the middle:
    #              downtrend and RSI >= 50 = SELL; uptrend and RSI <= 50 = BUY.
    # Trend direction = which of +DI / -DI is bigger. trend_rr overrides the
    # reward-to-risk on trend trades (they are expected to run further).

    trades: List[Trade] = []
    open_trade: Optional[Trade] = None

    def _close(t: Trade, px: float, outcome: str, idx: int, when: str) -> None:
        risk = abs(t.entry - t.stop0)
        gain = (px - t.entry) if t.side == "BUY" else (t.entry - px)
        t.exit_px, t.outcome = px, outcome
        t.result_r = (gain / risk if risk else 0.0) + t.fin_r
        t.closed, t.bars_held = when, idx - t.opened_idx
        trades.append(t)

    for i in range(len(bars) - 1):
        nxt = bars[i + 1]
        # Last bar before a gap of 8h+ = Friday close (or a holiday).
        before_gap = (costs.weekend_close and i + 2 < len(bars)
                      and (bars[i + 2].ts - nxt.ts) > dt.timedelta(hours=8))

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
                if not hit_stop:
                    outcome = "TARGET"
                elif t.stop == t.stop0:
                    outcome = "STOP"
                else:
                    gain = (t.stop - t.entry) if t.side == "BUY" else (t.entry - t.stop)
                    outcome = "TRAIL" if gain > 1e-9 else "BE"
                _close(t, t.stop if hit_stop else t.target, outcome, i + 1, nxt.time)
                open_trade = None
                continue  # never open on the same bar we just closed

            # --- time costs, in the order the live bot meets them: the 20:00
            # flatten for expensive payers, then the 21:00 charge for holders,
            # then the Friday close for anything on a weekend-closing market.
            exit_px = nxt.bid_close if t.side == "BUY" else nxt.ask_close
            if costs.flattens(t.side) and _bar_contains(nxt, 20, costs.bar_minutes):
                _close(t, exit_px, "FLAT", i + 1, nxt.time)
                open_trade = None
                continue
            if _bar_contains(nxt, 21, costs.bar_minutes):
                r = costs.rate(t.side)
                if r is not None:
                    t.fin_r += r / stop_pct
            if before_gap:
                _close(t, exit_px, "WKND", i + 1, nxt.time)
                open_trade = None
                continue

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

        # --- look for a new entry. Not into a weekend: it would be flattened
        # at this bar's close and pay the spread for nothing.
        value = rsis[i]
        if value is None or before_gap:
            continue
        if since is not None and bars[i].ts < since:
            continue
        if hours is not None and bars[i + 1].ts.hour not in hours:
            continue   # the ENTRY bar's open hour must be in the allowed set
        if atr_max_pctile is not None and atr[i] is not None and i >= 200:
            window = [x for x in atr[i - 200:i] if x is not None]
            if window and sum(1 for x in window if x <= atr[i]) / len(window) * 100 > atr_max_pctile:
                continue
        rr_here = reward_to_risk
        trending = (regime in ("adx", "both")) and (adx14[i] is None or adx14[i] >= (REGIME_ADX_MAX or 25))
        if trending and trend_mode:
            if pdi[i] is None or mdi[i] is None:
                continue
            up = pdi[i] > mdi[i]
            if trend_mode == "invert":
                if value <= oversold and not up:
                    side = "SELL"
                elif value >= overbought and up:
                    side = "BUY"
                else:
                    continue
            elif trend_mode == "pullback":
                if not up and value >= 50:
                    side = "SELL"
                elif up and value <= 50:
                    side = "BUY"
                else:
                    continue
            else:
                continue
            if trend_rr:
                rr_here = trend_rr
        else:
            if value <= oversold:
                side = "BUY"
            elif value >= overbought:
                side = "SELL"
            else:
                continue
            if trending:
                continue
        if regime in ("sma", "both"):
            if sma50[i] is None:
                continue
            if side == "BUY" and closes[i] < sma50[i]:
                continue
            if side == "SELL" and closes[i] > sma50[i]:
                continue

        t = Trade()
        t.side = side
        if side == "BUY":
            t.entry = nxt.ask_open           # we buy at the ask
            t.stop0 = t.entry * (1 - stop_pct / 100.0)
            t.target = t.entry * (1 + stop_pct * rr_here / 100.0)
        else:
            t.entry = nxt.bid_open           # we sell at the bid
            t.stop0 = t.entry * (1 + stop_pct / 100.0)
            t.target = t.entry * (1 - stop_pct * rr_here / 100.0)
        t.stop = t.stop0
        t.best = None
        t.fin_r = 0.0
        t.opened = nxt.time
        t.opened_idx = i + 1
        t.outcome = "OPEN"
        t.result_r = 0.0
        open_trade = t

    return trades


def summarise(trades: List[Trade], days: float) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "per_day": 0.0, "wins": 0, "win_pct": 0.0, "be": 0,
                "total_r": 0.0, "avg_r": 0.0, "fin_r": 0.0, "flat": 0, "nights": 0}
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
        "fin_r": sum(t.fin_r for t in trades),
        "flat": sum(1 for t in trades if t.outcome in ("FLAT", "WKND")),
        "nights": sum(1 for t in trades if t.fin_r != 0.0),
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

    # Time costs from the broker's own instrument record, not assumptions.
    try:
        info = client.market(epic)
    except CapitalError:
        info = {}
    fee = (info.get("instrument", {}) or {}).get("overnightFee", {}) or {}
    hours = (info.get("instrument", {}) or {}).get("openingHours") or {}
    costs = Costs(
        long_rate=None if fee.get("longRate") is None else float(fee["longRate"]),
        short_rate=None if fee.get("shortRate") is None else float(fee["shortRate"]),
        weekend_close=not hours.get("sat"),
        bar_minutes=minutes,
    )
    if args.no_costs:
        costs = Costs(bar_minutes=minutes)
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
    if args.no_costs:
        print("time costs: OFF (--no-costs)")
    else:
        def _fmt(side):
            r = costs.rate(side)
            if r is None:
                return "unknown -> flattened nightly"
            return "%+.4f%%/day = %+.3fR/night, %s" % (
                r, r / stop_pct, "flattened at 20:00" if costs.flattens(side) else "held")
        print("financing  long: %s" % _fmt("BUY"))
        print("           short: %s" % _fmt("SELL"))
        print("weekend    %s" % ("flattened before the Friday close" if costs.weekend_close
                                  else "trades through (market open Saturday)"))

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
                     args.breakeven_at, args.trail, costs, live_regime, since, live_trend, None, live_hours)
        print("RSI(%d) %.0f/%.0f -- every trade:" % (args.period, oversold, overbought))
        print("")
        for t in trades:
            print("  %-20s %-4s entry %10.5g  exit %10.5g  %-6s %+6.2fR  $%+6.2f"
                  % (t.opened[:19], t.side, t.entry, t.exit_px, t.outcome,
                     t.result_r, t.result_r * risk_cash))
        s = summarise(trades, days)
        print("")
        print("  %d trades, %.1f/day, %d wins (%.0f%%), %d break-even, %d flattened, "
              "financing %+.2fR, total %+.2fR = $%+.2f"
              % (s["n"], s["per_day"], s["wins"], s["win_pct"], s["be"], s["flat"],
                 s["fin_r"], s["total_r"], s["total_r"] * risk_cash))
        return

    since = dt.datetime.strptime(args.since, "%Y-%m-%d") if args.since else None
    # Default regime = whatever bot.py runs, so a plain run scores the live strategy.
    live_hours = None if args.no_regime else live_cfg.get("hours")
    if live_hours:
        print("session    entries only on bars opening %s UTC (bot.py; the pair's home session)"
              % "/".join("%02d" % h for h in live_hours))
    live_regime = None if (args.no_regime or REGIME_ADX_MAX <= 0) else "adx"
    live_trend = TREND_MODE if (live_regime and TREND_MODE == "invert") else None
    if live_regime:
        print("regime     ADX(14) < %.0f = fade the move; >= %.0f = %s  (bot.py; --no-regime for v1)"
              % (REGIME_ADX_MAX, REGIME_ADX_MAX, "trade WITH the trend" if live_trend else "stand aside"))
    else:
        print("regime     none")
    if args.filters:
        oversold, overbought = live_band
        print("Live strategy (v3) with extra entry filters%s:" % (" (entries since %s)" % args.since if args.since else ""))
        print("")
        print("%-40s %6s %5s %7s %8s %9s" % ("filter", "trades", "win%", "avg R", "total R", "P&L $"))
        print("-" * 82)
        # This broker's 4-hour bars open at 02/06/10/14/18/22 UTC.
        for label, hrs, atrp in (("none (v3)", None, None),
                                 ("entries on 10:00/14:00 bars only (London+NY)", [10, 14], None),
                                 ("entries 06:00-14:00 bars (London day)", [6, 10, 14], None),
                                 ("no entries on 22:00/02:00 bars (Asia night)", [6, 10, 14, 18], None),
                                 ("ATR(14) below 90th pctile", None, 90.0),
                                 ("ATR(14) below 80th pctile", None, 80.0),
                                 ("ATR(14) below 70th pctile", None, 70.0),
                                 ("ATR(14) below 60th pctile", None, 60.0),
                                 ("ATR(14) below 50th pctile", None, 50.0)):
            trades = run(bars, args.period, float(oversold), float(overbought), stop_pct, rr, None, None,
                         costs, live_regime, since, live_trend, None, hrs, atrp)
            s = summarise(trades, days)
            print("%-40s %6d %5.0f %+7.2f %+8.2f %+9.2f" % (label, s["n"], s["win_pct"], s["avg_r"], s["total_r"], s["total_r"] * risk_cash))
        return

    if args.trend:
        oversold, overbought = live_band
        print("RSI(%d) %d/%d, ranging = mean reversion; what to do when ADX >= %.0f%s:"
              % (args.period, oversold, overbought, REGIME_ADX_MAX or 25,
                 " (entries since %s)" % args.since if args.since else ""))
        print("")
        print("%-34s %6s %5s %7s %8s %9s" % ("trending-market rule", "trades", "win%", "avg R", "total R", "P&L $"))
        print("-" * 76)
        for label, tm, trr in (("stand aside (v2)", None, None),
                               ("v1: fade it anyway (no filter)", "V1", None),
                               ("invert: sell weakness, RR 1.5 (v3 live)", "invert", None),
                               ("invert: sell weakness, RR 2.0", "invert", 2.0),
                               ("pullback: sell RSI>=50, RR 1.5", "pullback", None),
                               ("pullback: sell RSI>=50, RR 2.0", "pullback", 2.0)):
            if tm == "V1":
                trades = run(bars, args.period, float(oversold), float(overbought), stop_pct, rr, None, None, costs, None, since)
            else:
                trades = run(bars, args.period, float(oversold), float(overbought), stop_pct, rr, None, None, costs, "adx", since, tm, trr)
            s = summarise(trades, days)
            print("%-34s %6d %5.0f %+7.2f %+8.2f %+9.2f" % (label, s["n"], s["win_pct"], s["avg_r"], s["total_r"], s["total_r"] * risk_cash))
        return

    if args.regime:
        oversold, overbought = live_band
        print("RSI(%d) %d/%d -- regime filters%s:" % (args.period, oversold, overbought,
              " (entries since %s)" % args.since if args.since else ""))
        print("")
        print("%-22s %6s %5s %7s %8s %9s" % ("filter", "trades", "win%", "avg R", "total R", "P&L $"))
        print("-" * 64)
        for label, rg in (("none (v1, to 26 Sep)", None), ("ADX(14) < %.0f  (v2, live)" % (REGIME_ADX_MAX or 25), "adx"),
                          ("side of SMA50", "sma"), ("both", "both")):
            trades = run(bars, args.period, float(oversold), float(overbought),
                         stop_pct, rr, None, None, costs, rg, since)
            s = summarise(trades, days)
            print("%-22s %6d %5.0f %+7.2f %+8.2f %+9.2f" % (label, s["n"], s["win_pct"],
                  s["avg_r"], s["total_r"], s["total_r"] * risk_cash))
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
                         stop_pct, rr, be_at, tr, costs, live_regime, since, live_trend, None, live_hours)
            s = summarise(trades, days)
            print("%-28s %6d %5.0f %6d %+7.2f %+8.2f %+9.2f"
                  % (label, s["n"], s["win_pct"], s["be"], s["avg_r"],
                     s["total_r"], s["total_r"] * risk_cash))
        return

    print("%-12s %7s %8s %7s %6s %8s %9s %10s" %
          ("RSI band", "trades", "per day", "win%", "flat", "fin R", "total R", "P&L $"))
    print("-" * 74)
    for oversold, overbought in ((20, 80), (25, 75), (30, 70), (35, 65),
                                 (40, 60), (45, 55)):
        trades = run(bars, args.period, float(oversold), float(overbought),
                     stop_pct, rr, args.breakeven_at, args.trail, costs, live_regime, since, live_trend, None, live_hours)
        s = summarise(trades, days)
        marker = "   <- live now" if (oversold, overbought) == live_band else ""
        print("%-12s %7d %8.1f %7.0f %6d %+8.2f %+9.2f %+10.2f%s"
              % ("%d/%d" % (oversold, overbought), s["n"], s["per_day"],
                 s["win_pct"], s["flat"], s["fin_r"], s["total_r"],
                 s["total_r"] * risk_cash, marker))

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
    ap.add_argument("--regime", action="store_true",
                    help="compare regime filters (none / adx / sma50 side / both) on the live band")
    ap.add_argument("--trend", action="store_true",
                    help="compare what to do in a TRENDING market: stand aside / invert / pullback, at 1.5 and 2.0 RR")
    ap.add_argument("--filters", action="store_true",
                    help="on the live strategy, compare entry-hour and volatility filters")
    ap.add_argument("--no-regime", action="store_true",
                    help="ignore the ADX filter bot.py applies (the pre-26-Sep strategy)")
    ap.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                    help="only open trades from this date (score a specific window)")
    ap.add_argument("--no-costs", action="store_true",
                    help="ignore financing and weekend/overnight flattens (the pre-9-Sep model)")
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
