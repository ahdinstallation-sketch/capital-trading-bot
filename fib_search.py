#!/usr/bin/env python3
"""
Fibonacci retracement test, same discipline as edge_search.py.

The textbook trade: find the last swing (highest high and lowest low of the
last 50 bars, the far end confirmed by 2 bars after it). In an up-swing, put
a BUY limit at a retracement level, stop further down the ladder, target the
swing high. Down-swings mirrored. Two textbook ladders, two timeframes:

  F1  entry 61.8%, stop 100% (swing origin), target 0% (swing extreme)  ~1.6:1
  F2  entry 50.0%, stop 78.6%,               target 0%                  ~1.75:1

Limit orders fill only when the right side of the spread reaches the level
(BUY on the ask, SELL on the bid); a gap through the level fills at the open.
On the fill bar only the stop is checked (worst case). Costs and rules as in
edge_search.py. Pass marks are fixed and printed first; now 14 strategies
have been tried, so the t-stat bar rises to 2.7.
"""

import datetime as dt
import sys
from collections import defaultdict

import edge_search as es
from capital_client import CapitalClient, CapitalError

N = 50
LADDERS = (("F1 61.8% entry / 100% stop / 0% target", 0.618, 1.0),
           ("F2 50% entry / 78.6% stop / 0% target", 0.5, 0.786))
MIN_T = 2.7


def simulate(bars, rates, bar_minutes, entry_f, stop_f):
    trades, pos, order = [], None, None

    def close(px, ts):
        e, r, side = pos["entry"], pos["risk"], pos["side"]
        move = (px - e) if side == "BUY" else (e - px)
        rate = rates[0] if side == "BUY" else rates[1]
        fin = es.nights(pos["ts"], ts) * (rate / 100.0) * e / r
        trades.append({"ts": pos["ts"], "r": move / r + fin, "r_gross": move / r})

    for j in range(1, len(bars)):
        b = bars[j]
        end_ts = b.ts + dt.timedelta(minutes=bar_minutes)
        if pos:                                         # manage an open trade
            if pos["side"] == "BUY":
                if b.ob <= pos["stop"]:
                    close(b.ob, b.ts); pos = None
                elif b.lb <= pos["stop"]:
                    close(pos["stop"], end_ts); pos = None
                elif b.hb >= pos["target"]:
                    close(max(pos["target"], b.ob), end_ts); pos = None
            else:
                if b.oa >= pos["stop"]:
                    close(b.oa, b.ts); pos = None
                elif b.ha >= pos["stop"]:
                    close(pos["stop"], end_ts); pos = None
                elif b.la <= pos["target"]:
                    close(min(pos["target"], b.oa), end_ts); pos = None
        elif order:                                     # try to fill the limit
            side, lvl, stop, tgt = order
            fill = None
            if side == "BUY" and b.la <= lvl:
                fill = min(lvl, b.oa)
            elif side == "SELL" and b.hb >= lvl:
                fill = max(lvl, b.ob)
            if fill is not None:
                risk = (fill - stop) if side == "BUY" else (stop - fill)
                if risk > 0:
                    pos = {"side": side, "entry": fill, "stop": stop, "target": tgt, "risk": risk, "ts": b.ts}
                    hit = b.lb <= stop if side == "BUY" else b.ha >= stop
                    if hit:                             # worst case on the fill bar
                        close(stop, end_ts); pos = None
        order = None
        if pos or j < N:
            continue
        win = range(j - N + 1, j + 1)                   # set tomorrow's order at today's close
        ih = max(win, key=lambda k: (bars[k].h, k))
        il = min(win, key=lambda k: (bars[k].l, -k))
        H, L = bars[ih].h, bars[il].l
        if H <= L:
            continue
        if il < ih and ih <= j - 2:                      # up-swing, high confirmed
            lvl, stop, tgt = H - entry_f * (H - L), H - stop_f * (H - L), H
            if b.c > lvl:
                order = ("BUY", lvl, stop, tgt)
        elif ih < il and il <= j - 2:                    # down-swing, low confirmed
            lvl, stop, tgt = L + entry_f * (H - L), L + stop_f * (H - L), L
            if b.c < lvl:
                order = ("SELL", lvl, stop, tgt)
    return trades


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
    P = es.PASS

    print("")
    print("PASS RULES, fixed before any result (same as the other 10, t raised to %.1f for 14 tries):" % MIN_T)
    print("  stage 1: >= %d trades (4H) / %d (daily); avg >= +%.2fR; t >= %.1f; >= 2 of 3 pairs; >= 2/3 of years"
          % (P["min_trades"]["HOUR_4"], P["min_trades"]["DAY"], P["min_avg_r"], MIN_T))
    print("  stage 2: holdout %s -> today total R > 0, looked at ONLY for stage-1 passers" % holdout.date())
    print("")

    data, rates = {}, {}
    for epic in es.EPICS:
        fee = (client.market(epic).get("instrument", {}) or {}).get("overnightFee", {}) or {}
        rates[epic] = (float(fee.get("longRate") or 0), float(fee.get("shortRate") or 0))
        for res in ("HOUR_4", "DAY"):
            data[(epic, res)] = es.fetch(client, epic, res, warm[res], now)

    hdr = "%-40s %4s %6s %5s %7s %8s %6s %6s %6s  %s" % (
        "strategy", "tf", "trades", "win%", "avg R", "total R", "t", "pairs", "years", "stage 1")
    print(hdr); print("-" * len(hdr))
    passers = []
    for res in ("HOUR_4", "DAY"):
        for name, ef, sf in LADDERS:
            rows, gross, hold, per_pair = [], [], [], defaultdict(float)
            for epic in es.EPICS:
                for t in simulate(data[(epic, res)], rates[epic], minutes[res], ef, sf):
                    if t["ts"] < start[res]:
                        continue
                    if t["ts"] >= holdout:
                        hold.append(t["r"])
                    else:
                        rows.append((epic, t["ts"].year, t["r"])); gross.append(t["r_gross"])
                        per_pair[epic] += t["r"]
            n, tot, avg, tstat, win = es.stats([r for _, _, r in rows])
            by_year = defaultdict(float)
            for _, y, r in rows:
                by_year[y] += r
            full = [y for y in by_year if start[res].year <= y < holdout.year]
            ypos = sum(1 for y in full if by_year[y] > 0)
            ppos = sum(1 for e in es.EPICS if per_pair[e] > 0)
            ok = (n >= P["min_trades"][res] and avg >= P["min_avg_r"] and tstat >= MIN_T
                  and ppos >= P["min_pairs"] and full and ypos >= P["min_year_share"] * len(full))
            print("%-40s %4s %6d %5.0f %+7.3f %+8.1f %+6.2f %4d/3 %3d/%-2d  %s" % (
                name, "4H" if res == "HOUR_4" else "1D", n, win, avg, tot, tstat, ppos, ypos, len(full),
                "PASS" if ok else "fail"))
            _, gtot, gavg, _, _ = es.stats(gross)
            if abs(gavg - avg) > 0.02:
                print("%-40s %46s" % ("", "(before financing: avg %+.3fR, total %+.1fR)" % (gavg, gtot)))
            if ok:
                passers.append((name, res, hold, by_year, per_pair))
    print("")
    if not passers:
        print("RESULT: no Fibonacci rule passed stage 1. The holdout was not looked at.")
    for name, res, hold, by_year, per_pair in passers:
        hn, htot, _, _, _ = es.stats(hold)
        print("STAGE 2  %s (%s): holdout %d trades, %+.1fR -> %s" % (name, res, hn, htot, "PASS" if htot > 0 else "fail"))
        print("         by year: " + "  ".join("%d %+.1fR" % (y, by_year[y]) for y in sorted(by_year)))
        print("         by pair: " + "  ".join("%s %+.1fR" % (e, per_pair[e]) for e in es.EPICS))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CapitalError as exc:
        print("broker error: %s" % exc)
        sys.exit(1)
