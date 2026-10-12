#!/usr/bin/env python3
"""
"Smarter" crypto trend bot = structure, not a new rule.

Universe (fixed by DATA, not performance): every coin below whose daily
history starts before 2021-06-01. Same three textbook rules per coin (50/200,
Turtle 20/10, 63-day momentum), long or cash, spot, 0.15% on all turnover,
decided at the close and filled at the next open.

  V0  current plan: BTC + ETH, equal weight
  V1  all coins, weight per coin = 1/volatility (60-day), no leverage
  V2  each week hold the 3 coins with the strongest positive 63-day return, equal weight
  HOLD  equal-weight basket of all coins, bought once

PASS MARKS, fixed before running (realistic window = from the Nov-2021 peak):
  a smarter version passes if, in that window, it earns MORE per year than V0
  AND its worst drop is no worse than V0's, AND it made money over the full period.
Caveat printed with results: coins still listed today are survivors (dead coins
are gone), which flatters every long-only result, V1/V2/HOLD most.
"""

import datetime as dt
import math
import sys

import edge_search as es
from capital_client import CapitalClient, CapitalError

CANDIDATES = ("BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "BCHUSD", "ADAUSD", "SOLUSD", "DOGEUSD", "DOTUSD",
              "LINKUSD", "AVAXUSD", "BNBUSD", "XLMUSD", "TRXUSD", "ETCUSD", "ATOMUSD", "UNIUSD")
COST = 0.0015


def rule_states(bars):
    cl = [b.c for b in bars]
    s50, s200 = es.sma(cl, 50), es.sma(cl, 200)
    hi20, lo10 = es.prior_high(bars, 20), es.prior_low(bars, 10)
    out, pos_b = {}, False
    for j in range(200, len(bars)):
        a = s50[j] > s200[j]
        pos_b = (cl[j] > hi20[j]) if not pos_b else not (cl[j] < lo10[j])
        c = cl[j] > cl[j - 63]
        r63 = cl[j] / cl[j - 63] - 1
        vol = math.sqrt(sum(math.log(cl[k] / cl[k - 1]) ** 2 for k in range(j - 59, j + 1)) / 60)
        out[bars[j].ts.date()] = ((a + pos_b + c) / 3.0, r63, vol)
    return out


def main():
    client = CapitalClient()
    client.login()
    now = dt.datetime.utcnow()
    coins, states, opens = [], {}, {}
    for epic in CANDIDATES:
        try:
            bars = es.fetch(client, epic, "DAY", dt.datetime(2014, 1, 1), now)
        except CapitalError:
            continue
        if len(bars) < 400 or bars[0].ts >= dt.datetime(2021, 6, 1):
            continue
        coins.append(epic)
        states[epic] = rule_states(bars)
        opens[epic] = {b.ts.date(): b.o for b in bars}
    print("")
    print("Universe (history before 2021-06-01): %s" % ", ".join(coins))

    days = sorted(set().union(*[set(states[c]) for c in coins]))

    def simulate(kind, start):
        eq, peak, mdd, w_prev, curve = 1.0, 1.0, 0.0, {}, []
        held = {}
        ds = [d for d in days if d >= start]
        for i in range(len(ds) - 2):
            d, d1, d2 = ds[i], ds[i + 1], ds[i + 2]
            avail = [c for c in coins if d in states[c] and d1 in opens[c] and d2 in opens[c]]
            if kind == "V0":
                pool = [c for c in avail if c in ("BTCUSD", "ETHUSD")]
                w = {c: states[c][d][0] / len(pool) for c in pool} if pool else {}
            elif kind == "V1":
                inv = {c: 1 / states[c][d][2] for c in avail if states[c][d][2] > 0}
                tot = sum(inv.values())
                w = {c: inv[c] / tot * states[c][d][0] for c in inv} if tot else {}
            elif kind == "V2":
                if i % 7 == 0 or not held:
                    ranked = sorted((c for c in avail if states[c][d][1] > 0), key=lambda c: -states[c][d][1])[:3]
                    held = {c: 1 / 3.0 for c in ranked}
                w = {c: v for c, v in held.items() if c in avail}
            else:   # HOLD basket bought at the start
                if not held:
                    held = {c: 1.0 / len(avail) for c in avail}
                w = held
            turnover = sum(abs(w.get(c, 0) - w_prev.get(c, 0)) for c in set(w) | set(w_prev)) if kind != "HOLD" or i == 0 else 0
            eq *= (1 - COST * turnover)
            ret = lambda c: (opens[c][d2] / opens[c][d1] - 1) if (d1 in opens[c] and d2 in opens[c]) else 0.0
            r = sum(wt * ret(c) for c, wt in w.items())
            eq *= (1 + r)
            if kind == "HOLD":   # let the basket drift
                held = {c: wt * (1 + ret(c)) / (1 + r) for c, wt in held.items()}
            w_prev = w if kind != "HOLD" else held
            peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
            curve.append((d2, eq))
        yrs = (curve[-1][0] - curve[0][0]).days / 365.25
        return eq, eq ** (1 / yrs) - 1, mdd

    results = {}
    for label, start in (("full period (from 2018)", dt.date(2018, 1, 1)), ("from the Nov-2021 peak", dt.date(2021, 12, 18))):
        print("")
        print("=== %s ===" % label)
        for kind, name in (("V0", "V0 current plan: BTC+ETH"), ("V1", "V1 all coins, sized by risk"),
                           ("V2", "V2 top-3 strongest coins weekly"), ("HOLD", "hold equal basket")):
            eq, cagr, mdd = simulate(kind, start)
            results[(label, kind)] = (eq, cagr, mdd)
            print("  %-34s growth %+8.0f%%  per yr %+6.1f%%  worst drop %+6.1f%%" % (name, (eq - 1) * 100, cagr * 100, mdd * 100))
    print("")
    full, rec = "full period (from 2018)", "from the Nov-2021 peak"
    for kind in ("V1", "V2"):
        ok = (results[(rec, kind)][1] > results[(rec, "V0")][1] and results[(rec, kind)][2] >= results[(rec, "V0")][2]
              and results[(full, kind)][0] > 1)
        print("%s vs V0: %s" % (kind, "PASS -- smarter" if ok else "FAIL"))
    print("Caveat: only coins still listed today are included (survivors), which flatters V1, V2 and HOLD.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
