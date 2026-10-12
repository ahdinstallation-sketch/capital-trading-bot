#!/usr/bin/env python3
"""
Fibonacci + Bollinger confluence -- one pre-specified rule, tested once.

At a 4-hour close: the last 50-bar swing is an up-swing (high confirmed by 2
bars), price has retraced into the 50%-61.8% golden zone, AND the close is
below the lower Bollinger(20,2) band -> BUY at the next open. Stop at the
swing low (100%). Exit when price closes back at the middle band. Down-swings
mirrored. Same engine and costs as edge_search.py.

Pass marks fixed before running. This is the 16th rule tried, so the bar is
the strictest yet and it must hold on ALL seven pairs' unseen data.
"""

import datetime as dt
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError

PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "NZDUSD", "EURGBP")
N = 50
MARKS = {"min_trades": 150, "min_avg": 0.05, "min_t": 2.7, "min_pairs": 5, "min_years": 2}


class FibBollinger(es.BollingerFade):
    name, res = "Fib 50-61.8% zone + below/above Bollinger band", "HOUR_4"

    def __init__(self, bars):
        es.BollingerFade.__init__(self, bars)
        self.sig = [None] * len(bars)
        for j in range(N, len(bars)):
            if self.mid[j] is None:
                continue
            win = range(j - N + 1, j + 1)
            ih = max(win, key=lambda k: (bars[k].h, k))
            il = min(win, key=lambda k: (bars[k].l, -k))
            H, L = bars[ih].h, bars[il].l
            if H <= L:
                continue
            c = self.c[j]
            if il < ih <= j - 2 and H - 0.618 * (H - L) <= c <= H - 0.5 * (H - L) and c < self.dn[j]:
                self.sig[j] = ("BUY", L)
            elif ih < il <= j - 2 and L + 0.5 * (H - L) <= c <= L + 0.618 * (H - L) and c > self.up[j]:
                self.sig[j] = ("SELL", H)

    def entry(self, j):
        return self.sig[j]


def main():
    client = CapitalClient()
    client.login()
    recent = client.candles("EURUSD", "HOUR_4", 1000)
    holdout = dt.datetime.strptime((recent[0].get("snapshotTimeUTC") or recent[0]["snapshotTime"])[:19],
                                   "%Y-%m-%dT%H:%M:%S")
    now, warm, start = dt.datetime.utcnow(), dt.datetime(2022, 10, 1), dt.datetime(2023, 1, 1)
    print("")
    print("PASS MARKS, fixed before running:")
    print("  stage 1 (unseen: EUR/GBP/AUD 2023 -> %s, the 4 new pairs 2023 -> today): >= %d trades;"
          " avg >= +%.2fR; t >= %.1f; positive on >= %d of 7 pairs; positive in >= %d of 3 years"
          % (holdout.date(), MARKS["min_trades"], MARKS["min_avg"], MARKS["min_t"], MARKS["min_pairs"], MARKS["min_years"]))
    print("  stage 2: EUR/GBP/AUD holdout %s -> today total R > 0, only if stage 1 passes" % holdout.date())
    print("")
    rows, hold, per_pair = [], [], {}
    for epic in PAIRS:
        fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
        rates = (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0))
        bars = es.fetch(client, epic, "HOUR_4", warm, now)
        trades = es.simulate(FibBollinger(bars), rates, 240)
        orig = epic in es.EPICS
        mine = []
        for t in trades:
            if t["ts"] < start:
                continue
            if orig and t["ts"] >= holdout:
                hold.append(t["r"])
            else:
                mine.append(t); rows.append((t["ts"].year, t["r"]))
        n, tot, avg, ts, win = es.stats([t["r"] for t in mine])
        per_pair[epic] = tot
        print("  %-7s %4d trades  win %3.0f%%  avg %+.3fR  total %+6.1fR" % (epic, n, win, avg, tot))
    n, tot, avg, t, win = es.stats([r for _, r in rows])
    by_year = defaultdict(float)
    for y, r in rows:
        by_year[y] += r
    ypos = sum(1 for y in (2023, 2024, 2025) if by_year[y] > 0)
    ppos = sum(1 for v in per_pair.values() if v > 0)
    ok = (n >= MARKS["min_trades"] and avg >= MARKS["min_avg"] and t >= MARKS["min_t"]
          and ppos >= MARKS["min_pairs"] and ypos >= MARKS["min_years"])
    print("  ALL 7   %4d trades  win %3.0f%%  avg %+.3fR  total %+6.1fR  t %+.2f  pairs+ %d/7  years+ %d/3 -> stage 1 %s"
          % (n, win, avg, tot, t, ppos, ypos, "PASS" if ok else "FAIL"))
    print("          by year: " + "  ".join("%d %+.1fR" % (y, by_year[y]) for y in sorted(by_year)))
    if ok:
        hn, htot, _, _, _ = es.stats(hold)
        print("  STAGE 2 holdout %d trades, %+.1fR -> %s" % (hn, htot, "PASS" if htot > 0 else "FAIL"))
    else:
        print("  stage 2 not looked at.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
