#!/usr/bin/env python3
"""
Edge search: does ANY well-known trading rule make money on EUR/GBP/AUD,
on history it was never tuned on, after real costs?

The pass rules are fixed in PASS below and printed before any result.
Every strategy runs at its textbook settings -- nothing is tuned here.

Stage 1 (out of sample): 4-hour rules 2023-01-01 -> start of the tuning
window; daily rules 2013-01-01 -> same. Stage 2 (holdout): the tuning window
to today, looked at ONLY for strategies that pass stage 1.

Costs: every fill on the real bid/ask of its candle (spread paid both ways),
stops that gap are filled at the open, a bar that touches stop and target
counts as a stop, and financing at TODAY's broker rates for every night
held (weekends counted). Risk unit R = the initial stop distance.
"""

import datetime as dt
import math
import sys
from collections import defaultdict

from capital_client import CapitalClient, CapitalError

EPICS = ("EURUSD", "GBPUSD", "AUDUSD")
PASS = {
    "min_trades": {"HOUR_4": 150, "DAY": 100},
    "min_avg_r": 0.05,
    "min_t": 2.6,           # ~5% significance after allowing for 10 strategies
    "min_pairs": 2,         # of 3 positive
    "min_year_share": 2 / 3.0,
}


# ------------------------------------------------------------------ data

class Bar(object):
    __slots__ = ("ts", "ob", "oa", "hb", "ha", "lb", "la", "cb", "ca", "o", "h", "l", "c")


def load(candles):
    out, seen = [], set()
    for c in candles:
        t = c.get("snapshotTimeUTC") or c.get("snapshotTime")
        if not t or t in seen:
            continue
        try:
            b = Bar()
            b.ts = dt.datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")
            b.ob, b.oa = float(c["openPrice"]["bid"]), float(c["openPrice"]["ask"])
            b.hb, b.ha = float(c["highPrice"]["bid"]), float(c["highPrice"]["ask"])
            b.lb, b.la = float(c["lowPrice"]["bid"]), float(c["lowPrice"]["ask"])
            b.cb, b.ca = float(c["closePrice"]["bid"]), float(c["closePrice"]["ask"])
        except (KeyError, TypeError, ValueError):
            continue
        seen.add(t)
        b.o, b.h, b.l, b.c = (b.ob + b.oa) / 2, (b.hb + b.ha) / 2, (b.lb + b.la) / 2, (b.cb + b.ca) / 2
        out.append(b)
    out.sort(key=lambda b: b.ts)
    return out


def fetch(client, epic, res, start, end):
    chunk = dt.timedelta(days=150 if res == "HOUR_4" else 900)
    raw, cur = [], start
    while cur < end:
        nxt = min(cur + chunk, end)
        try:
            raw += client._request("GET", "/api/v1/prices/%s" % epic, params={
                "resolution": res, "max": 1000,
                "from": cur.strftime("%Y-%m-%dT%H:%M:%S"), "to": nxt.strftime("%Y-%m-%dT%H:%M:%S")}).get("prices", [])
        except CapitalError as exc:
            if "not-found" not in str(exc):     # a period before the market existed is empty, not an error
                raise
        cur = nxt
    return load(raw)


# ------------------------------------------------------------ indicators

def sma(x, n):
    out, s = [None] * len(x), 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def stdev(x, n, mean):
    out = [None] * len(x)
    for i in range(n - 1, len(x)):
        m = mean[i]
        out[i] = math.sqrt(sum((x[k] - m) ** 2 for k in range(i - n + 1, i + 1)) / n)
    return out


def ema(x, n):
    out, k, prev = [None] * len(x), 2.0 / (n + 1), None
    vals = [(i, v) for i, v in enumerate(x) if v is not None]
    if len(vals) < n:
        return out
    start = vals[n - 1][0]
    prev = sum(v for _, v in vals[:n]) / n
    out[start] = prev
    for i in range(start + 1, len(x)):
        if x[i] is None:
            continue
        prev = x[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(x, n):
    out = [None] * len(x)
    if len(x) <= n:
        return out
    g = sum(max(x[i] - x[i - 1], 0) for i in range(1, n + 1)) / n
    l = sum(max(x[i - 1] - x[i], 0) for i in range(1, n + 1)) / n
    out[n] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    for i in range(n + 1, len(x)):
        d = x[i] - x[i - 1]
        g = (g * (n - 1) + max(d, 0)) / n
        l = (l * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


def atr(bars, n=14):
    out, a = [None] * len(bars), None
    trs = [bars[0].h - bars[0].l] + [max(b.h - b.l, abs(b.h - p.c), abs(b.l - p.c))
                                      for p, b in zip(bars, bars[1:])]
    for i, t in enumerate(trs):
        if i == n - 1:
            a = sum(trs[:n]) / n
        elif i >= n:
            a = (a * (n - 1) + t) / n
        out[i] = a
    return out


def prior_high(bars, n):
    return [None if i < n else max(b.h for b in bars[i - n:i]) for i in range(len(bars))]


def prior_low(bars, n):
    return [None if i < n else min(b.l for b in bars[i - n:i]) for i in range(len(bars))]


# ------------------------------------------------------------ strategies
# entry(j) -> (side, stop_level or None) decided at bar j's CLOSE; fill at j+1 open.
# exit(j, side) -> True to close at j+1 open. stop_atr: initial stop in ATRs.

class Strategy(object):
    name, res, stop_atr, target_r, max_bars = "", "", 2.0, None, None

    def __init__(self, bars):
        self.bars = bars
        self.c = [b.c for b in bars]
        self.atr = atr(bars)

    def entry(self, j):
        return None

    def exit(self, j, side):
        return False


class RSIFade(Strategy):
    name, res, target_r = "RSI(14) 30/70 fade", "HOUR_4", 1.5

    def __init__(self, bars):
        Strategy.__init__(self, bars); self.r = rsi(self.c, 14)

    def entry(self, j):
        r = self.r[j]
        return None if r is None else ("BUY", None) if r <= 30 else ("SELL", None) if r >= 70 else None


class BollingerFade(Strategy):
    name, res = "Bollinger(20,2) fade to middle", "HOUR_4"

    def __init__(self, bars):
        Strategy.__init__(self, bars)
        self.mid = sma(self.c, 20); sd = stdev(self.c, 20, self.mid)
        self.up = [None if m is None else m + 2 * s for m, s in zip(self.mid, sd)]
        self.dn = [None if m is None else m - 2 * s for m, s in zip(self.mid, sd)]

    def entry(self, j):
        if self.mid[j] is None:
            return None
        return ("BUY", None) if self.c[j] < self.dn[j] else ("SELL", None) if self.c[j] > self.up[j] else None

    def exit(self, j, side):
        return self.c[j] >= self.mid[j] if side == "BUY" else self.c[j] <= self.mid[j]


class LondonBreakout(Strategy):
    """Asian range (bars opening 22:00-06:59 UTC); the first London bar (07-10 UTC)
    closing outside it -> trade the break; stop at the other side; target 1R;
    out after two bars (end of London/NY overlap)."""
    name, res, target_r, max_bars = "London breakout of the Asian range", "HOUR_4", 1.0, 2

    def __init__(self, bars):
        Strategy.__init__(self, bars)
        self.sig = [None] * len(bars)
        hi = lo = None
        for j, b in enumerate(bars):
            h = b.ts.hour
            asian = h >= 22 or h <= 6
            if asian:
                if hi is None or (j and bars[j - 1].ts.hour not in range(0, 7) and bars[j - 1].ts.hour < 22):
                    hi, lo = b.h, b.l
                else:
                    hi, lo = max(hi, b.h), min(lo, b.l)
            elif 7 <= h <= 10 and hi is not None:
                if b.c > hi:
                    self.sig[j] = ("BUY", lo)
                elif b.c < lo:
                    self.sig[j] = ("SELL", hi)
                hi = lo = None
            else:
                hi = lo = None

    def entry(self, j):
        return self.sig[j]


class ConnorsRSI2(Strategy):
    name, res = "Connors RSI(2) with 200-day trend", "DAY"

    def __init__(self, bars):
        Strategy.__init__(self, bars)
        self.r2, self.s200, self.s5 = rsi(self.c, 2), sma(self.c, 200), sma(self.c, 5)

    def entry(self, j):
        if self.s200[j] is None or self.r2[j] is None:
            return None
        if self.c[j] > self.s200[j] and self.r2[j] < 10:
            return ("BUY", None)
        if self.c[j] < self.s200[j] and self.r2[j] > 90:
            return ("SELL", None)
        return None

    def exit(self, j, side):
        return self.c[j] > self.s5[j] if side == "BUY" else self.c[j] < self.s5[j]


class Turtle(Strategy):
    res = "DAY"

    def __init__(self, bars, n_in, n_out):
        Strategy.__init__(self, bars)
        self.name = "Turtle %d-day breakout / %d-day exit" % (n_in, n_out)
        self.hi, self.lo = prior_high(bars, n_in), prior_low(bars, n_in)
        self.xhi, self.xlo = prior_high(bars, n_out), prior_low(bars, n_out)

    def entry(self, j):
        if self.hi[j] is None:
            return None
        return ("BUY", None) if self.c[j] > self.hi[j] else ("SELL", None) if self.c[j] < self.lo[j] else None

    def exit(self, j, side):
        return self.c[j] < self.xlo[j] if side == "BUY" else self.c[j] > self.xhi[j]


class StateStrategy(Strategy):
    """Long while state is +1, short while -1; re-enters after a stop if the state holds."""
    res = "DAY"

    def state(self, j):
        return 0

    def entry(self, j):
        s = self.state(j)
        return ("BUY", None) if s > 0 else ("SELL", None) if s < 0 else None

    def exit(self, j, side):
        s = self.state(j)
        return s <= 0 if side == "BUY" else s >= 0


class MACross(StateStrategy):
    name = "50/200-day moving-average trend"

    def __init__(self, bars):
        StateStrategy.__init__(self, bars); self.f, self.s = sma(self.c, 50), sma(self.c, 200)

    def state(self, j):
        return 0 if self.s[j] is None else (1 if self.f[j] > self.s[j] else -1)


class TSMOM(StateStrategy):
    def __init__(self, bars, n):
        StateStrategy.__init__(self, bars); self.n = n
        self.name = "Time-series momentum, %d-day" % n

    def state(self, j):
        return 0 if j < self.n else (1 if self.c[j] > self.c[j - self.n] else -1)


class MACD(StateStrategy):
    name = "MACD(12,26,9) signal cross"

    def __init__(self, bars):
        StateStrategy.__init__(self, bars)
        e12, e26 = ema(self.c, 12), ema(self.c, 26)
        self.m = [None if a is None or b is None else a - b for a, b in zip(e12, e26)]
        self.sig = ema(self.m, 9)

    def state(self, j):
        return 0 if self.sig[j] is None else (1 if self.m[j] > self.sig[j] else -1)


STRATEGIES = [
    RSIFade, BollingerFade, LondonBreakout,
    ConnorsRSI2, lambda b: Turtle(b, 20, 10), lambda b: Turtle(b, 55, 20),
    MACross, lambda b: TSMOM(b, 63), lambda b: TSMOM(b, 252), MACD,
]


# ---------------------------------------------------------------- engine

def nights(t0, t1):
    first = t0.replace(hour=21, minute=0, second=0, microsecond=0)
    if first <= t0:
        first += dt.timedelta(days=1)
    return 0 if t1 < first else int((t1 - first).total_seconds() // 86400) + 1


def simulate(strat, rates, bar_minutes):
    bars, trades = strat.bars, []
    pos, pend_entry, pend_exit = None, None, False

    def close(px, ts, why):
        side, e, r = pos["side"], pos["entry"], pos["risk"]
        move = (px - e) if side == "BUY" else (e - px)
        rate = rates[0] if side == "BUY" else rates[1]
        fin = nights(pos["ts"], ts) * (rate / 100.0) * e / r
        trades.append({"ts": pos["ts"], "r": move / r + fin, "r_gross": move / r, "why": why})

    for j in range(1, len(bars)):
        b = bars[j]
        if pos and pend_exit:
            close(b.ob if pos["side"] == "BUY" else b.oa, b.ts, "signal"); pos = None
        pend_exit = False
        if pos is None and pend_entry:
            side, stop_lvl, atr_v = pend_entry
            e = b.oa if side == "BUY" else b.ob
            if stop_lvl is None:
                risk = strat.stop_atr * atr_v
                stop = e - risk if side == "BUY" else e + risk
            else:
                stop, risk = stop_lvl, (e - stop_lvl) if side == "BUY" else (stop_lvl - e)
            if risk > 0:
                tgt = None if not strat.target_r else (e + strat.target_r * risk if side == "BUY" else e - strat.target_r * risk)
                pos = {"side": side, "entry": e, "stop": stop, "risk": risk, "target": tgt, "j0": j, "ts": b.ts}
        pend_entry = None
        if pos:
            end_ts = b.ts + dt.timedelta(minutes=bar_minutes)
            if pos["side"] == "BUY":
                if b.ob <= pos["stop"] and j > pos["j0"]:
                    close(b.ob, b.ts, "stop-gap"); pos = None
                elif b.lb <= pos["stop"]:
                    close(pos["stop"], end_ts, "stop"); pos = None
                elif pos["target"] and b.hb >= pos["target"]:
                    close(pos["target"], end_ts, "target"); pos = None
            else:
                if b.oa >= pos["stop"] and j > pos["j0"]:
                    close(b.oa, b.ts, "stop-gap"); pos = None
                elif b.ha >= pos["stop"]:
                    close(pos["stop"], end_ts, "stop"); pos = None
                elif pos["target"] and b.la <= pos["target"]:
                    close(pos["target"], end_ts, "target"); pos = None
        if j == len(bars) - 1:
            break
        if pos:
            held = j - pos["j0"] + 1
            if strat.exit(j, pos["side"]) or (strat.max_bars and held >= strat.max_bars):
                pend_exit = True
        elif strat.atr[j] is not None:
            sig = strat.entry(j)
            if sig:
                pend_entry = (sig[0], sig[1], strat.atr[j])
    if pos:
        last = bars[-1]
        close(last.cb if pos["side"] == "BUY" else last.ca, last.ts, "open-at-end")
    return trades


def stats(rs):
    n = len(rs)
    if n < 2:
        return n, 0.0, 0.0, 0.0, 0.0
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    return n, sum(rs), m, (m / (sd / math.sqrt(n)) if sd else 0.0), 100.0 * sum(1 for x in rs if x > 0) / n


# ------------------------------------------------------------------ main

def main():
    client = CapitalClient()
    client.login()
    recent = client.candles("EURUSD", "HOUR_4", 1000)
    holdout = dt.datetime.strptime((recent[0].get("snapshotTimeUTC") or recent[0]["snapshotTime"])[:19],
                                   "%Y-%m-%dT%H:%M:%S")
    now = dt.datetime.utcnow()
    start = {"HOUR_4": dt.datetime(2023, 1, 1), "DAY": dt.datetime(2013, 1, 1)}
    warm = {"HOUR_4": dt.datetime(2022, 10, 1), "DAY": dt.datetime(2012, 1, 1)}
    minutes = {"HOUR_4": 240, "DAY": 1440}

    print("")
    print("PASS RULES, fixed before any result:")
    print("  stage 1 (out of sample, 4H from 2023 / daily from 2013, to %s):" % holdout.date())
    print("    >= %d trades (4H) / %d (daily); average >= +%.2fR per trade; t-stat >= %.1f;"
          % (PASS["min_trades"]["HOUR_4"], PASS["min_trades"]["DAY"], PASS["min_avg_r"], PASS["min_t"]))
    print("    positive on >= %d of 3 pairs; positive in >= 2/3 of full years" % PASS["min_pairs"])
    print("  stage 2 (holdout %s -> today): total R > 0, checked ONLY for stage-1 passers" % holdout.date())
    print("")

    data, rates = {}, {}
    for epic in EPICS:
        info = client.market(epic)
        fee = (info.get("instrument", {}) or {}).get("overnightFee", {}) or {}
        rates[epic] = (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0))
        for res in ("HOUR_4", "DAY"):
            data[(epic, res)] = fetch(client, epic, res, warm[res], now)
        print("  %s: 4H %d bars, daily %d bars (%s -> %s); financing long %+.4f%% short %+.4f%% /day"
              % (epic, len(data[(epic, "HOUR_4")]), len(data[(epic, "DAY")]),
                 data[(epic, "DAY")][0].ts.date(), data[(epic, "DAY")][-1].ts.date(), rates[epic][0], rates[epic][1]))
    print("")

    hdr = "%-40s %4s %6s %5s %7s %8s %6s %6s %6s  %s" % (
        "strategy", "tf", "trades", "win%", "avg R", "total R", "t", "pairs", "years", "stage 1")
    print(hdr); print("-" * len(hdr))
    passers = []
    for make in STRATEGIES:
        per_pair, holdout_r, rows, gross = {}, [], [], []
        name = res = None
        for epic in EPICS:
            s = make(data[(epic, RSIFade.res if False else ("HOUR_4" if make in (RSIFade, BollingerFade, LondonBreakout) else "DAY"))])
            name, res = s.name, s.res
            for t in simulate(s, rates[epic], minutes[res]):
                if t["ts"] < start[res]:
                    continue
                if t["ts"] >= holdout:
                    holdout_r.append(t["r"])
                else:
                    rows.append((epic, t["ts"].year, t["r"])); gross.append(t["r_gross"])
                    per_pair[epic] = per_pair.get(epic, 0.0) + t["r"]
        rs = [r for _, _, r in rows]
        n, tot, avg, tstat, win = stats(rs)
        by_year = defaultdict(float)
        for _, y, r in rows:
            by_year[y] += r
        full_years = [y for y in by_year if start[res].year <= y < holdout.year]
        yrs_pos = sum(1 for y in full_years if by_year[y] > 0)
        pairs_pos = sum(1 for e in EPICS if per_pair.get(e, 0) > 0)
        ok = (n >= PASS["min_trades"][res] and avg >= PASS["min_avg_r"] and tstat >= PASS["min_t"]
              and pairs_pos >= PASS["min_pairs"] and full_years and yrs_pos >= PASS["min_year_share"] * len(full_years))
        print("%-40s %4s %6d %5.0f %+7.3f %+8.1f %+6.2f %4d/3 %3d/%-2d  %s" % (
            name[:40], "4H" if res == "HOUR_4" else "1D", n, win, avg, tot, tstat, pairs_pos,
            yrs_pos, len(full_years), "PASS" if ok else "fail"))
        if ok:
            passers.append((name, holdout_r, by_year, per_pair))
        _, gtot, gavg, _, _ = stats(gross)
        if abs(gavg - avg) > 0.02:
            print("%-40s %46s" % ("", "(before financing: avg %+.3fR, total %+.1fR)" % (gavg, gtot)))

    print("")
    if not passers:
        print("RESULT: no strategy passed stage 1. The holdout was not looked at.")
    for name, hr, by_year, per_pair in passers:
        hn, htot, havg, _, _ = stats(hr)
        print("STAGE 2  %s: holdout %d trades, total %+.1fR -> %s" % (name, hn, htot, "PASS" if htot > 0 else "fail"))
        print("         by year: " + "  ".join("%d %+.1fR" % (y, by_year[y]) for y in sorted(by_year)))
        print("         by pair: " + "  ".join("%s %+.1fR" % (e, per_pair.get(e, 0)) for e in EPICS))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
