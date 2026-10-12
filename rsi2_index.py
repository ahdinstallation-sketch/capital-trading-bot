#!/usr/bin/env python3
"""
Connors RSI(2) on stock indices -- the one documented short-term edge left
untested. Textbook rule, long only, daily bars:
  BUY at the next open when the close is above its 200-day average and
  RSI(2) < 10; SELL at the next open once the close is above its 5-day
  average. The original has no stop; a wide 3x ATR(14) stop is added only
  as the risk unit and disaster brake.
Real bid/ask fills, today's financing for every night held (longs pay).

Pass marks fixed before running -- ALL three must pass:
  1. US 500, 2013-01-01 -> 2026-02-27: >= 100 trades, avg >= +0.05R,
     t >= 2.0, positive in >= 9 of the 13 full years.
  2. Same rule, unchanged, on US Tech 100 and Germany 40, 2013 -> today:
     both positive.
  3. US 500 from 2026-02-27 -> today (never looked at): total R > 0.
"""

import datetime as dt
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError


class LongRSI2(es.ConnorsRSI2):
    stop_atr = 3.0

    def entry(self, j):
        s = es.ConnorsRSI2.entry(self, j)
        return s if s and s[0] == "BUY" else None


def run(client, epic, warm, now):
    fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
    rates = (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0))
    bars = es.fetch(client, epic, "DAY", warm, now)
    return es.simulate(LongRSI2(bars), rates, 1440), bars, rates


def main():
    client = CapitalClient()
    client.login()
    now, warm, start = dt.datetime.utcnow(), dt.datetime(2012, 1, 1), dt.datetime(2013, 1, 1)
    holdout = dt.datetime(2026, 2, 27)
    print(__doc__.split("Pass marks")[1].join(["\nPASS MARKS", ""]) if False else "")
    print("PASS MARKS (fixed before running): 1) US500 2013->2026-02-27: >=100 trades, avg >= +0.05R,"
          " t >= 2.0, >= 9/13 years positive;  2) US100 and DE40 2013->today both positive;"
          "  3) US500 2026-02-27->today total R > 0.  All three must pass.")
    print("")

    trades, bars, rates = run(client, "US500", warm, now)
    print("US500: %d daily bars %s -> %s; financing long %+.4f%%/day" % (len(bars), bars[0].ts.date(), bars[-1].ts.date(), rates[0]))
    s1 = [t for t in trades if start <= t["ts"] < holdout]
    s3 = [t for t in trades if t["ts"] >= holdout]
    n, tot, avg, tt, win = es.stats([t["r"] for t in s1])
    by_year = defaultdict(float)
    for t in s1:
        by_year[t["ts"].year] += t["r"]
    full = [y for y in range(2013, 2026)]
    ypos = sum(1 for y in full if by_year[y] > 0)
    _, gtot, gavg, _, _ = es.stats([t["r_gross"] for t in s1])
    ok1 = n >= 100 and avg >= 0.05 and tt >= 2.0 and ypos >= 9
    print("1) US500 2013 -> 2026-02-27: %d trades, win %.0f%%, avg %+.3fR (before financing %+.3fR), total %+.1fR, t %+.2f, years+ %d/13 -> %s"
          % (n, win, avg, gavg, tot, tt, ypos, "PASS" if ok1 else "FAIL"))
    print("   by year: " + "  ".join("%d %+.1f" % (y, by_year[y]) for y in full))

    ok2 = True
    for epic in ("US100", "DE40"):
        tr, b, _ = run(client, epic, warm, now)
        rr = [t["r"] for t in tr if t["ts"] >= start]
        n2, tot2, avg2, t2, w2 = es.stats(rr)
        ok2 = ok2 and tot2 > 0
        print("2) %-6s 2013 -> today: %d trades, win %.0f%%, avg %+.3fR, total %+.1fR, t %+.2f -> %s"
              % (epic, n2, w2, avg2, tot2, t2, "positive" if tot2 > 0 else "NEGATIVE"))
    print("   replication -> %s" % ("PASS" if ok2 else "FAIL"))

    n3, tot3, avg3, _, w3 = es.stats([t["r"] for t in s3])
    ok3 = tot3 > 0
    print("3) US500 2026-02-27 -> today (never looked at): %d trades, total %+.1fR -> %s" % (n3, tot3, "PASS" if ok3 else "FAIL"))
    print("")
    print("RESULT: %s" % ("PASS on all three" if ok1 and ok2 and ok3 else "FAIL"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
