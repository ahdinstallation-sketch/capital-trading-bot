#!/usr/bin/env python3
"""
Out-of-sample test: run the LIVE rule, unchanged, on 4-hour history the
rules were never tuned on.

Every version of the strategy (bands, ADX switch, home sessions, the
exhaustion filter) was chosen on the most recent 1,000 bars (~166 days).
This pages older history from the broker in ~5-month chunks and scores
the live rule on everything BEFORE that window, year by year, with the
same costs and the same pessimistic fills as backtest.py.

    python oos.py                 # from 2023-01-01
    python oos.py --start 2024-01-01

Caveat printed with the results: financing uses today's broker rates for
all years (historical rates are not available); spreads are historical,
since every candle carries its own bid and ask.
"""

import argparse
import datetime as dt
import sys
from collections import defaultdict

import backtest
import bot
from capital_client import CapitalClient, CapitalError

EPICS = ("EURUSD", "GBPUSD", "AUDUSD")
CHUNK_DAYS = 150          # ~640 four-hour bars, under the broker's 1,000 cap


def fetch(client, epic, start, end):
    seen, out = set(), []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=CHUNK_DAYS), end)
        data = client._request("GET", "/api/v1/prices/%s" % epic, params={
            "resolution": "HOUR_4", "max": 1000,
            "from": cur.strftime("%Y-%m-%dT%H:%M:%S"), "to": nxt.strftime("%Y-%m-%dT%H:%M:%S")})
        for c in data.get("prices", []):
            t = c.get("snapshotTime")
            if t and t not in seen:
                seen.add(t)
                out.append(c)
        cur = nxt
    out.sort(key=lambda c: c["snapshotTime"])
    return out


def score(bars, epic, costs, exhaust_on=True, regime=True):
    cfg = bot.instrument_config(epic)
    saved = bot.TREND_EXHAUST_RSI
    if not exhaust_on:
        bot.TREND_EXHAUST_RSI = 0
    try:
        return backtest.run(
            bars, bot.RSI_PERIOD, float(cfg["oversold"]), float(cfg["overbought"]),
            float(cfg["stop_pct"]), bot.REWARD_TO_RISK, None, None, costs,
            "adx" if regime else None, None, bot.TREND_MODE if regime else None,
            None, cfg.get("hours") if regime else None)
    finally:
        bot.TREND_EXHAUST_RSI = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--risk-cash", type=float, default=None, help="1R in $ (default 1%% of live balance)")
    args = ap.parse_args()

    client = CapitalClient()
    client.login()
    risk_cash = args.risk_cash or client.balance() * bot.RISK_PER_TRADE_PCT / 100.0

    # The in-sample window = the most recent 1,000 bars, exactly what backtest.py uses.
    recent = client.candles("EURUSD", "HOUR_4", 1000)
    insample_start = dt.datetime.strptime(recent[0]["snapshotTime"][:19], "%Y-%m-%dT%H:%M:%S")
    start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    # 60 extra days before the start so RSI/ADX are warm on the first scored bar.
    warm = start - dt.timedelta(days=60)

    print("")
    print("Live rule, scored OUT OF SAMPLE: %s -> %s (the tuning window starts %s)"
          % (start.date(), insample_start.date(), insample_start.date()))
    print("1R = $%.2f. Costs: historical bid/ask spread per candle; financing at TODAY's rates." % risk_cash)
    print("")

    rows = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))   # version -> (epic, year) -> [n, wins, R]
    for epic in EPICS:
        candles = fetch(client, epic, warm, insample_start)
        bars = backtest.load_bars(candles)
        info = client.market(epic)
        fee = (info.get("instrument", {}) or {}).get("overnightFee", {}) or {}
        hours = (info.get("instrument", {}) or {}).get("openingHours") or {}
        costs = backtest.Costs(float(fee.get("longRate", 0) or 0), float(fee.get("shortRate", 0) or 0),
                               not hours.get("sat"), 240)
        print("%s: %d bars %s -> %s" % (epic, len(bars), bars[0].time[:10], bars[-1].time[:10]))
        for version, kw in (("v5 live", {}), ("v4 (no exhaustion)", {"exhaust_on": False}),
                            ("v1 (RSI only)", {"regime": False})):
            for t in score(bars, epic, costs, **kw):
                when = dt.datetime.strptime(t.opened[:19], "%Y-%m-%dT%H:%M:%S")
                if when < start:
                    continue
                cell = rows[version][(epic, when.year)]
                cell[0] += 1
                cell[1] += 1 if t.result_r > 1e-9 else 0
                cell[2] += t.result_r

    years = sorted({y for v in rows.values() for (_, y) in v})
    for version in ("v5 live", "v4 (no exhaustion)", "v1 (RSI only)"):
        print("")
        print("=== %s ===" % version)
        print("%-8s" % "pair" + "".join("%17s" % y for y in years) + "%19s" % "TOTAL")
        tot_all = [0, 0, 0.0]
        for epic in EPICS:
            line, tot = "%-8s" % epic, [0, 0, 0.0]
            for y in years:
                n, w, r = rows[version][(epic, y)]
                line += "%17s" % ("%+6.1fR (%3d)" % (r, n) if n else "-")
                tot = [tot[0] + n, tot[1] + w, tot[2] + r]
            line += "%19s" % ("%+6.1fR (%3d)" % (tot[2], tot[0]))
            tot_all = [tot_all[0] + tot[0], tot_all[1] + tot[1], tot_all[2] + tot[2]]
            print(line)
        yline = "%-8s" % "ALL"
        for y in years:
            n = sum(rows[version][(e, y)][0] for e in EPICS)
            r = sum(rows[version][(e, y)][2] for e in EPICS)
            yline += "%17s" % ("%+6.1fR (%3d)" % (r, n))
        yline += "%19s" % ("%+6.1fR (%3d)" % (tot_all[2], tot_all[0]))
        print(yline)
        if tot_all[0]:
            print("win rate %.0f%%, avg %+.3fR/trade, total $%+.2f at 1R=$%.2f"
                  % (100.0 * tot_all[1] / tot_all[0], tot_all[2] / tot_all[0], tot_all[2] * risk_cash, risk_cash))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
