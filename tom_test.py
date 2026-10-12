#!/usr/bin/env python3
"""
Turn-of-the-month (TOM) effect on capital.com index CFDs -- the account the
bot already trades. McConnell & Xu (2008): equity returns concentrate from
the last trading day of a month to the third trading day of the next.

Trade, textbook, untuned: BUY at the open of the month's last trading day,
SELL at the open of the 4th trading day of the new month (= hold day -1..+3).
Real bid/ask fills; today's overnight financing for every night held.
Position = 1x the account (no leverage) -- returns are % of the account.

PASS MARKS, fixed before running:
  1. US500, 2013 -> today: money made after costs; positive in >= 9 of 13
     full years; t-stat of the monthly trades >= 2.0.
  2. Replication, same rule unchanged: US100 and DE40 both make money.
  3. TOM days must beat the other days on US500 (the effect is real, not
     just "stocks went up").
"""

import datetime as dt
import math
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError


def tom_trades(bars, long_rate):
    by_month = defaultdict(list)
    for i, b in enumerate(bars):
        by_month[(b.ts.year, b.ts.month)].append(i)
    months = sorted(by_month)
    out = []
    for m0, m1 in zip(months, months[1:]):
        if len(by_month[m1]) < 4:
            continue
        i_in, i_out = by_month[m0][-1], by_month[m1][3]
        e, x = bars[i_in].oa, bars[i_out].ob
        fin = es.nights(bars[i_in].ts, bars[i_out].ts) * long_rate / 100.0
        out.append((bars[i_in].ts, x / e - 1 + fin, i_in, i_out))
    return out


def day_split(bars, trades):
    tom = set()
    for _, _, a, b in trades:
        tom.update(range(a, b))
    on = [bars[k + 1].o / bars[k].o - 1 for k in range(len(bars) - 1) if k in tom]
    off = [bars[k + 1].o / bars[k].o - 1 for k in range(len(bars) - 1) if k not in tom]
    return sum(on) / len(on), sum(off) / len(off)


def main():
    client = CapitalClient()
    client.login()
    now = dt.datetime.utcnow()
    print(__doc__.split("PASS MARKS")[1].join(["PASS MARKS", ""]))
    ok = {}
    for epic in ("US500", "US100", "DE40"):
        fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
        lr = float(fee.get("longRate") or 0)
        bars = es.fetch(client, epic, "DAY", dt.datetime(2012, 12, 1), now)
        tr = [t for t in tom_trades(bars, lr) if t[0] >= dt.datetime(2013, 1, 1)]
        rs = [t[1] for t in tr]
        eq, peak, mdd = 1.0, 1.0, 0.0
        yr = defaultdict(lambda: 1.0)
        for ts, r, _, _ in tr:
            eq *= 1 + r; peak = max(peak, eq); mdd = min(mdd, eq / peak - 1); yr[ts.year] *= 1 + r
        n = len(rs); m = sum(rs) / n; sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1)); t = m / (sd / math.sqrt(n))
        full = [y for y in range(2013, now.year)]
        ypos = sum(1 for y in full if yr[y] > 1)
        yrs = (tr[-1][0] - tr[0][0]).days / 365.25
        on, off = day_split(bars, tr)
        win = 100.0 * sum(1 for x in rs if x > 0) / n
        print("%s  financing long %+.4f%%/day" % (epic, lr))
        print("   %d monthly trades, win %.0f%%, avg %+.2f%% per trade, t %+.2f" % (n, win, m * 100, t))
        print("   growth %+.0f%% (%+.1f%%/yr) while in the market ~4 days a month; worst drop %+.1f%%; years up %d/%d"
              % ((eq - 1) * 100, (eq ** (1 / yrs) - 1) * 100, mdd * 100, ypos, len(full)))
        print("   avg day: TOM %+.3f%%  vs  other days %+.3f%%  (%.1fx)" % (on * 100, off * 100, on / off if off else float("inf")))
        print("   by year: " + "  ".join("%d %+.1f%%" % (y, (yr[y] - 1) * 100) for y in sorted(yr)))
        ok[epic] = (eq > 1, ypos, t, on > off)
    p1 = ok["US500"][0] and ok["US500"][1] >= 9 and ok["US500"][2] >= 2.0
    p2 = ok["US100"][0] and ok["DE40"][0]
    p3 = ok["US500"][3]
    print("")
    print("1) US500 money, >= 9/13 years, t >= 2.0: %s" % ("PASS" if p1 else "FAIL"))
    print("2) US100 and DE40 both make money:       %s" % ("PASS" if p2 else "FAIL"))
    print("3) TOM days beat other days on US500:     %s" % ("PASS" if p3 else "FAIL"))
    print("RESULT: %s" % ("PASS" if p1 and p2 and p3 else "FAIL"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
