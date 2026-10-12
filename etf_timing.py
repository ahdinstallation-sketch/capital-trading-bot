#!/usr/bin/env python3
"""
The fee-free alternative: hold the index itself (an ETF, no leverage, no
overnight financing) while it is above its 10-month average, cash otherwise.
Checked once a month (Faber's textbook rule, untuned). Index prices from the
broker's daily feed 2012 -> today; dividends NOT included (conservative);
0.10%/yr fund fee while invested; 0.05% cost per switch; cash earns 0.

Pass marks, fixed before running -- on EACH of US500, US100, DE40:
  money made overall, positive in >= 2/3 of full years,
  and a smaller worst drop (max drawdown) than simply holding.
"""

import datetime as dt
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError


def month_ends(bars):
    out = {}
    for b in bars:
        out[(b.ts.year, b.ts.month)] = (b.ts, b.c)
    return [out[k] for k in sorted(out)]


def path(me, timed):
    eq, peak, mdd, inv = 1.0, 1.0, 0.0, False
    years = defaultdict(lambda: 1.0)
    for i in range(10, len(me) - 1):
        sma10 = sum(c for _, c in me[i - 9:i + 1]) / 10.0
        want = (me[i][1] > sma10) if timed else True
        if want != inv:
            eq *= (1 - 0.0005); inv = want
        r = (me[i + 1][1] / me[i][1] - 1 - 0.001 / 12) if inv else 0.0
        eq *= (1 + r)
        years[me[i + 1][0].year] *= (1 + r)
        peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
    return eq, mdd, years


def main():
    client = CapitalClient()
    client.login()
    now = dt.datetime.utcnow()
    print("")
    print("PASS MARKS (fixed first), each index: money made; >= 2/3 of full years positive; smaller max drop than holding.")
    print("")
    print("%-6s %-22s %10s %8s %10s %9s" % ("index", "", "growth", "per yr", "worst drop", "years +"))
    ok_all = True
    for epic in ("US500", "US100", "DE40"):
        bars = es.fetch(client, epic, "DAY", dt.datetime(2012, 1, 1), now)
        me = month_ends(bars)
        span = (me[-1][0] - me[10][0]).days / 365.25
        res = {}
        for label, timed in (("hold", False), ("10-month rule", True)):
            eq, mdd, yrs = path(me, timed)
            full = [y for y in yrs if me[11][0].year <= y < now.year]
            ypos = sum(1 for y in full if yrs[y] > 1)
            res[label] = (eq, mdd, ypos, len(full))
            print("%-6s %-22s %+9.0f%% %+7.1f%% %+9.1f%% %5d/%-2d" % (
                epic, label, (eq - 1) * 100, (eq ** (1 / span) - 1) * 100, mdd * 100, ypos, len(full)))
        eq, mdd, ypos, nfull = res["10-month rule"]
        ok = eq > 1 and ypos >= 2 / 3.0 * nfull and mdd > res["hold"][1]
        ok_all = ok_all and ok
        print("%-6s %-22s -> %s" % ("", "", "PASS" if ok else "FAIL"))
    print("")
    print("RESULT: %s" % ("PASS on all three" if ok_all else "FAIL"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
