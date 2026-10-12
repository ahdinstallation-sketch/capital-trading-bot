#!/usr/bin/env python3
"""
Replication test for the one near-miss from edge_search.py: the Bollinger
(20,2) fade to the middle band on 4-hour bars (+0.068R/trade on EUR/GBP/AUD,
t=1.98 -- short of the bar for ten tries at once).

ONE pre-specified hypothesis, unchanged rule, two kinds of data it has never
touched:
  A. four new currency pairs, 2023-01-01 -> today: USDJPY USDCAD NZDUSD EURGBP
  B. the original three pairs' holdout (the tuning window -> today), which no
     Bollinger result has ever looked at.
Pass marks are fixed here and printed before anything runs.
"""

import datetime as dt
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError

NEW = ("USDJPY", "USDCAD", "NZDUSD", "EURGBP")
MARKS = {"A_min_trades": 300, "A_min_avg": 0.05, "A_min_t": 2.0, "A_min_pairs": 3, "A_min_years": 2,
         "B_min_total": 0.0}


def main():
    client = CapitalClient()
    client.login()
    recent = client.candles("EURUSD", "HOUR_4", 1000)
    holdout = dt.datetime.strptime((recent[0].get("snapshotTimeUTC") or recent[0]["snapshotTime"])[:19],
                                   "%Y-%m-%dT%H:%M:%S")
    now, warm, start = dt.datetime.utcnow(), dt.datetime(2022, 10, 1), dt.datetime(2023, 1, 1)

    print("")
    print("Rule: Bollinger(20,2) fade on 4-hour bars -- close below the lower band BUY / above the upper")
    print("band SELL, exit at the middle band, stop 2x ATR(14). Same engine and costs as edge_search.py.")
    print("PASS MARKS, fixed before running:")
    print("  A (4 new pairs, 2023 -> today): >= %d trades; avg >= +%.2fR; t >= %.1f; positive on >= %d of 4"
          " pairs; positive in >= %d of 3 full years (2023-25)"
          % (MARKS["A_min_trades"], MARKS["A_min_avg"], MARKS["A_min_t"], MARKS["A_min_pairs"], MARKS["A_min_years"]))
    print("  B (EUR/GBP/AUD holdout %s -> today): total R > 0" % holdout.date())
    print("  The rule passes only if BOTH pass.")
    print("")

    def run(epic, since, until):
        fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
        rates = (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0))
        bars = es.fetch(client, epic, "HOUR_4", warm, now)
        s = es.BollingerFade(bars)
        out = [t for t in es.simulate(s, rates, 240) if since <= t["ts"] < until]
        return out, len(bars), rates

    # ---- A
    rows, per_pair = [], {}
    for epic in NEW:
        try:
            ts, nb, rates = run(epic, start, now)
        except CapitalError as exc:
            print("  %s: unavailable (%s)" % (epic, str(exc)[:80])); continue
        n, tot, avg, t, win = es.stats([x["r"] for x in ts])
        per_pair[epic] = tot
        rows += [(epic, x["ts"].year, x["r"]) for x in ts]
        print("  %-7s %5d bars  %4d trades  win %3.0f%%  avg %+.3fR  total %+6.1fR  t %+.2f   (financing L %+.4f S %+.4f)"
              % (epic, nb, n, win, avg, tot, t, rates[0], rates[1]))
    n, tot, avg, t, win = es.stats([r for _, _, r in rows])
    by_year = defaultdict(float)
    for _, y, r in rows:
        by_year[y] += r
    ypos = sum(1 for y in (2023, 2024, 2025) if by_year[y] > 0)
    ppos = sum(1 for v in per_pair.values() if v > 0)
    a_ok = (n >= MARKS["A_min_trades"] and avg >= MARKS["A_min_avg"] and t >= MARKS["A_min_t"]
            and ppos >= MARKS["A_min_pairs"] and ypos >= MARKS["A_min_years"])
    print("  ALL 4    %4d trades  win %3.0f%%  avg %+.3fR  total %+6.1fR  t %+.2f  pairs+ %d/4  years+ %d/3  -> A %s"
          % (n, win, avg, tot, t, ppos, ypos, "PASS" if a_ok else "FAIL"))
    print("           by year: " + "  ".join("%d %+.1fR" % (y, by_year[y]) for y in sorted(by_year)))
    print("")

    # ---- B
    hrows = []
    for epic in es.EPICS:
        ts, _, _ = run(epic, holdout, now)
        hrows += [x["r"] for x in ts]
        hn, htot, havg, _, _ = es.stats([x["r"] for x in ts])
        print("  %-7s holdout %3d trades  total %+5.1fR" % (epic, hn, htot))
    hn, htot, havg, ht, hwin = es.stats(hrows)
    b_ok = htot > MARKS["B_min_total"]
    print("  ALL 3    holdout %3d trades  win %3.0f%%  avg %+.3fR  total %+5.1fR  -> B %s"
          % (hn, hwin, havg, htot, "PASS" if b_ok else "FAIL"))
    print("")
    print("RESULT: %s" % ("PASS -- the Bollinger fade replicates" if a_ok and b_ok else "FAIL -- it does not replicate"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
