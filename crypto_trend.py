#!/usr/bin/env python3
"""
Trend-following TRADING bot on SPOT crypto -- no overnight fee, because you
own the coin. Long or cash (spot cannot short). Daily bars, decided at the
close, filled at the next open. Costs: 0.15% per side (0.10% exchange fee +
0.05% half-spread). No financing. Cash earns 0. Prices: the broker's daily
feed (the price is the same everywhere; only the fees differ).

Three textbook rules, untuned:
  A  50/200-day average: in while the 50-day is above the 200-day
  B  Turtle: in on a close above the 20-day high, out on a close below the 10-day low
  C  Momentum: in while the close is above the close 63 days ago

Pass marks, fixed before running -- a rule passes only if, on BOTH BTC and ETH,
it makes money AND its worst drop is smaller than just holding. The idea
passes if at least 2 of the 3 rules pass (a real trend effect, not one lucky rule).
"""

import datetime as dt
import sys

import edge_search as es
from capital_client import CapitalClient, CapitalError

COST = 0.0015


def run(bars, rule):
    c = [b.c for b in bars]
    o = [b.o for b in bars]
    s50, s200 = es.sma(c, 50), es.sma(c, 200)
    hi20, lo10 = es.prior_high(bars, 20), es.prior_low(bars, 10)
    eq, peak, mdd, pos, trades = 1.0, 1.0, 0.0, False, 0
    for j in range(200, len(bars) - 1):
        if rule == "A":
            want = s50[j] > s200[j]
        elif rule == "B":
            want = (c[j] > hi20[j]) if not pos else not (c[j] < lo10[j])
        elif rule == "C":
            want = c[j] > c[j - 63]
        else:                                   # hold
            want = True
        # move from today's open-to-open: position chosen at close j acts from open j+1
        if want != pos:
            eq *= (1 - COST)
            pos = want
            trades += 1
        if pos and j + 2 < len(bars):
            eq *= o[j + 2] / o[j + 1]
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return eq, mdd, trades


def main():
    client = CapitalClient()
    client.login()
    now = dt.datetime.utcnow()
    print("")
    print("PASS MARKS (fixed first): a rule passes if on BOTH coins it makes money AND has a smaller worst drop than holding;")
    print("the idea passes if >= 2 of 3 rules pass.")
    print("")
    passed = {"A": True, "B": True, "C": True}
    for epic in ("BTCUSD", "ETHUSD"):
        bars = es.fetch(client, epic, "DAY", dt.datetime(2012, 1, 1), now)
        yrs = (bars[-1].ts - bars[200].ts).days / 365.25
        print("%s: %d daily bars %s -> %s (scored from %s)" % (epic, len(bars), bars[0].ts.date(), bars[-1].ts.date(), bars[200].ts.date()))
        h_eq, h_mdd, _ = run(bars, "hold")
        print("   %-34s growth %+8.0f%%  per yr %+6.1f%%  worst drop %+6.1f%%" % ("just hold", (h_eq - 1) * 100, (h_eq ** (1 / yrs) - 1) * 100, h_mdd * 100))
        for rule, label in (("A", "A 50/200-day average"), ("B", "B Turtle 20/10"), ("C", "C 63-day momentum")):
            eq, mdd, n = run(bars, rule)
            ok = eq > 1 and mdd > h_mdd
            passed[rule] = passed[rule] and ok
            print("   %-34s growth %+8.0f%%  per yr %+6.1f%%  worst drop %+6.1f%%  %3d switches  %s"
                  % (label, (eq - 1) * 100, (eq ** (1 / yrs) - 1) * 100, mdd * 100, n, "ok" if ok else "x"))
        print("")
    n_pass = sum(passed.values())
    print("Rules passing on both coins: %s  ->  RESULT: %s" % (
        ", ".join(k for k, v in passed.items() if v) or "none", "PASS" if n_pass >= 2 else "FAIL"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
