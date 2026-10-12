#!/usr/bin/env python3
"""
Search every 2- and 3-way combination of the indicators tested so far, then
confirm the best ones on data they were not picked on.

DISCOVERY: EUR/GBP/AUD 4-hour, 2023-01-01 -> start of the tuning window.
CONFIRMATION (never used to pick): USDJPY/USDCAD/NZDUSD/EURGBP 2023 -> today,
plus EUR/GBP/AUD from the tuning window -> today. 7 segments.

Every combination trades the same way so they are comparable: enter at the
next open when ALL its conditions agree, stop 2x ATR(14), target 1.5R, out
after 12 bars (2 days) at the latest. Same costs as edge_search.py.

Fixed before running:
  discovery shortlist: >= 100 trades, avg >= +0.05R, positive on >= 2 of 3 pairs;
                       the TOP 5 by t-stat go to confirmation.
  confirmation pass:   >= 50 trades, avg >= +0.05R, t >= 2.3 (5 candidates),
                       positive on >= 4 of 7 segments.
"""

import datetime as dt
import itertools
import sys

import bot
import edge_search as es
from capital_client import CapitalClient, CapitalError

DISC = ("EURUSD", "GBPUSD", "AUDUSD")
NEW = ("USDJPY", "USDCAD", "NZDUSD", "EURGBP")
N_FIB = 50


def conditions(bars):
    """name -> (long[], short[]). Filters have long == short."""
    c = [b.c for b in bars]
    n = len(bars)
    r14, r2 = es.rsi(c, 14), es.rsi(c, 2)
    mid = es.sma(c, 20); sd = es.stdev(c, 20, mid)
    s200 = es.sma(c, 200)
    e12, e26 = es.ema(c, 12), es.ema(c, 26)
    m = [None if a is None or b is None else a - b for a, b in zip(e12, e26)]
    sig = es.ema(m, 9)
    hi20, lo20 = es.prior_high(bars, 20), es.prior_low(bars, 20)
    adx = bot.adx_series([b.h for b in bars], [b.l for b in bars], c, 14)
    fl, fs = [False] * n, [False] * n
    for j in range(N_FIB, n):
        win = range(j - N_FIB + 1, j + 1)
        ih = max(win, key=lambda k: (bars[k].h, k)); il = min(win, key=lambda k: (bars[k].l, -k))
        H, L = bars[ih].h, bars[il].l
        if H > L:
            fl[j] = il < ih <= j - 2 and H - 0.618 * (H - L) <= c[j] <= H - 0.5 * (H - L)
            fs[j] = ih < il <= j - 2 and L + 0.5 * (H - L) <= c[j] <= L + 0.618 * (H - L)
    ok = lambda v: v is not None
    out = {
        "RSI14 30/70":   ([ok(x) and x <= 30 for x in r14], [ok(x) and x >= 70 for x in r14]),
        "RSI2 10/90":    ([ok(x) and x <= 10 for x in r2], [ok(x) and x >= 90 for x in r2]),
        "Bollinger out": ([ok(mid[j]) and c[j] < mid[j] - 2 * sd[j] for j in range(n)],
                          [ok(mid[j]) and c[j] > mid[j] + 2 * sd[j] for j in range(n)]),
        "Fib 50-61.8":   (fl, fs),
        "above/below SMA200": ([ok(s200[j]) and c[j] > s200[j] for j in range(n)],
                               [ok(s200[j]) and c[j] < s200[j] for j in range(n)]),
        "MACD side":     ([ok(sig[j]) and m[j] > sig[j] for j in range(n)], [ok(sig[j]) and m[j] < sig[j] for j in range(n)]),
        "20-bar breakout": ([ok(hi20[j]) and c[j] > hi20[j] for j in range(n)], [ok(lo20[j]) and c[j] < lo20[j] for j in range(n)]),
        "63-bar momentum": ([j >= 63 and c[j] > c[j - 63] for j in range(n)], [j >= 63 and c[j] < c[j - 63] for j in range(n)]),
    }
    rng = [ok(x) and x < 25 for x in adx]; trd = [ok(x) and x >= 25 for x in adx]
    ses = [7 <= b.ts.hour <= 12 for b in bars]
    out["ADX<25 (ranging)"] = (rng, rng)
    out["ADX>=25 (trending)"] = (trd, trd)
    out["London/NY hours"] = (ses, ses)
    return out


FILTERS = {"ADX<25 (ranging)", "ADX>=25 (trending)", "London/NY hours"}


class Combo(es.Strategy):
    res, stop_atr, target_r, max_bars = "HOUR_4", 2.0, 1.5, 12

    def __init__(self, bars, sig):
        es.Strategy.__init__(self, bars); self.sig = sig

    def entry(self, j):
        return self.sig[j]


def main():
    client = CapitalClient()
    client.login()
    recent = client.candles("EURUSD", "HOUR_4", 1000)
    holdout = dt.datetime.strptime((recent[0].get("snapshotTimeUTC") or recent[0]["snapshotTime"])[:19],
                                   "%Y-%m-%dT%H:%M:%S")
    now, warm, start = dt.datetime.utcnow(), dt.datetime(2022, 10, 1), dt.datetime(2023, 1, 1)

    pairs = {}
    for epic in DISC + NEW:
        fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
        bars = es.fetch(client, epic, "HOUR_4", warm, now)
        pairs[epic] = (bars, (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0)), conditions(bars))

    names = list(pairs["EURUSD"][2].keys())
    combos = []
    for k in (2, 3):
        for cmb in itertools.combinations(names, k):
            if all(x in FILTERS for x in cmb) or {"ADX<25 (ranging)", "ADX>=25 (trending)"} <= set(cmb):
                continue
            combos.append(cmb)

    results = []
    for cmb in combos:
        disc, conf = {e: [] for e in DISC}, {}
        for epic, (bars, rates, conds) in pairs.items():
            n = len(bars)
            L = [all(conds[x][0][j] for x in cmb) for j in range(n)]
            S = [all(conds[x][1][j] for x in cmb) for j in range(n)]
            sig = [("BUY", None) if L[j] and not S[j] else ("SELL", None) if S[j] and not L[j] else None for j in range(n)]
            for t in es.simulate(Combo(bars, sig), rates, 240):
                if t["ts"] < start:
                    continue
                if epic in DISC and t["ts"] < holdout:
                    disc[epic].append(t["r"])
                else:
                    seg = epic if epic in NEW else epic + " recent"
                    conf.setdefault(seg, []).append(t["r"])
        allr = [r for v in disc.values() for r in v]
        n, tot, avg, t, win = es.stats(allr)
        pp = sum(1 for v in disc.values() if sum(v) > 0)
        results.append((cmb, n, tot, avg, t, win, pp, conf))

    looked_good = [r for r in results if r[1] >= 100 and r[3] >= 0.05 and r[6] >= 2]
    looked_good.sort(key=lambda r: -r[4])
    print("")
    print("Combinations tested: %d (2-way and 3-way, %d indicators/filters)" % (len(results), len(names)))
    print("Looked good on the discovery data (>=100 trades, avg >= +0.05R, >=2 of 3 pairs): %d" % len(looked_good))
    print("")
    print("%-62s %6s %5s %7s %7s %6s" % ("top of the discovery list", "trades", "win%", "avg R", "total", "t"))
    for cmb, n, tot, avg, t, win, pp, _ in looked_good[:10]:
        print("%-62s %6d %5.0f %+7.3f %+7.1f %+6.2f" % (" + ".join(cmb)[:62], n, win, avg, tot, t))
    print("")
    print("CONFIRMATION of the top 5, on data they were not picked on (7 segments):")
    any_pass = False
    for cmb, n, tot, avg, t, win, pp, conf in looked_good[:5]:
        allc = [r for v in conf.values() for r in v]
        cn, ctot, cavg, ct, cwin = es.stats(allc)
        cpos = sum(1 for v in conf.values() if sum(v) > 0)
        ok = cn >= 50 and cavg >= 0.05 and ct >= 2.3 and cpos >= 4
        any_pass = any_pass or ok
        print("  %-60s" % " + ".join(cmb)[:60])
        print("     discovery %+.3fR/trade (t %+.2f)  ->  confirmation %d trades, %+.3fR/trade, total %+.1fR, t %+.2f, %d/7 segments +  -> %s"
              % (avg, t, cn, cavg, ctot, ct, cpos, "PASS" if ok else "FAIL"))
    print("")
    print("RESULT: %s" % ("at least one combination confirmed" if any_pass else "no combination confirmed on fresh data"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
