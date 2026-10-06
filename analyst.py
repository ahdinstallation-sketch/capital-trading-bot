"""
The bot's analyst. Not its brain -- its diary.

Once a day, after the FX session closes, this gathers everything the broker
and the calendar know about the day -- trades closed and why, positions
still open, what each market did, which high-impact events fired, the
running record against the kill rule -- and asks Claude to write the
post-mortem a good desk analyst would hand the owner: what happened, why
the rule did what it did, and what to watch. Plain English, no jargon.

It never places, closes, or sizes a trade, and it is told not to recommend
parameter changes from one day's evidence -- that is what the backtester
and the 60-trade rule are for. A trading rule you cannot backtest is a
rule you cannot trust, and an LLM reading headlines is exactly that.

    python analyst.py --if-due        # once a day after 21:00 UTC (the workflow)
    python analyst.py --now           # write today's report immediately
    python analyst.py --print-input   # show what Claude would be given, no API call

Cost: ~4k tokens in, ~1k out, once a day -- about $0.05/day on claude-opus-5.
Report lands in reports/YYYY-MM-DD.md and is committed by the workflow.
"""

import argparse
import datetime as dt
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import bot
import news
from capital_client import CapitalClient, CapitalError

log = logging.getLogger("analyst")

MODEL = os.getenv("ANALYST_MODEL", "claude-opus-5")
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
RUN_AFTER_UTC_HOUR = 21   # FX daily close; the swap has been charged, the day is done

SYSTEM = """You are the analyst for a small automated FX trading bot. Your reader is the bot's owner: a business owner, not a trader or a programmer. He wants to understand what his money did today and why, in plain English.

What the bot is (fixed facts; do not restate them, use them):
- Two regimes on 4-hour candles, chosen by ADX(14). RANGING (ADX below the limit): mean reversion -- buy when RSI(14) is below the instrument's oversold band, sell when above its overbought band. TRENDING (ADX at or above the limit): with the trend -- the same RSI reading is taken the other way, so oversold in a downtrend (-DI above +DI) is weakness to sell, overbought in an uptrend is strength to buy; a reading stretched in the trend's own direction does nothing. A with-the-trend entry is also skipped when the RSI reading is already exhausted (at or below 30, or at or above 70): a move that far along bounces more often than it continues. Each pair only ENTERS during its home session (EUR and GBP on the London-day bars, AUD on the Asia-night bars); open positions are managed around the clock.
- Every position carries a broker-side stop and target from the moment it opens (reward-to-risk given in the data). Position size is 1% of balance at risk per trade. At most a few positions at once, at most two on the same side of the US dollar.
- It holds through nights when financing is cheap, flattens before the Friday close, waits four hours after any close before re-entering the same instrument, and stands aside for 30 minutes either side of high-impact news.
- The strategy's parameters are frozen. A kill rule stops it if the record is bad enough after enough trades. Whether to change the rule is decided by backtests over months of data, never by one day.

Write the day's post-mortem. Cover, in this order and with these headings: What happened (trades opened and closed, with the outcome and the cash), Why the rule did that (what RSI/ADX/news made it act or hold back -- tie each trade to the conditions in the data), What the market did (each instrument's move and any event that mattered), Record (the running total versus the kill rule, said plainly), Watch (anything anomalous or worth the owner's attention tomorrow -- a close that was not the stop or target, a market that is trending hard, a filter that blocked everything).

Be concrete and use the numbers given. If a day was quiet, say so in a sentence or two and do not pad. Do not recommend changing any parameter, threshold or instrument; if the evidence tempts you, say what you see and note that the decision belongs to the backtest and the 60-trade rule. Do not speculate about news you are not given. About 250-400 words."""


# ------------------------------------------------------------ gather


def _tx_time(t: Dict[str, Any]) -> Optional[dt.datetime]:
    return bot._parse_tx_time(t)


def gather(client: CapitalClient, now: dt.datetime) -> Dict[str, Any]:
    day_start = now - dt.timedelta(hours=24)
    account = client.account()
    bal = account.get("balance", {}) or {}

    positions = []
    for raw in client.positions():
        p, m = raw.get("position", raw), raw.get("market", {}) or {}
        positions.append({
            "instrument": m.get("epic"), "direction": p.get("direction"), "size": p.get("size"),
            "entry": p.get("level"), "stop": p.get("stopLevel"), "target": p.get("profitLevel"),
            "open_pnl": p.get("upl"), "opened": (p.get("createdDate") or "")[:16],
        })

    try:
        frozen = dt.datetime.strptime(bot.STRATEGY_FROZEN_AT[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        frozen = now
    tx = client.transactions(min(frozen, day_start).strftime("%Y-%m-%dT%H:%M:%S"))
    closed_all = [t for t in tx if "closed" in (t.get("note") or "").lower()]
    since_freeze = [t for t in closed_all if (_tx_time(t) or now) >= frozen]
    today_closed = [t for t in closed_all if (_tx_time(t) or now) >= day_start]
    today_fees = [t for t in tx if "fee" in (t.get("note") or "").lower()
                  and (_tx_time(t) or now) >= day_start]

    risk_cash = float(bal.get("balance", 0.0)) * bot.RISK_PER_TRADE_PCT / 100.0
    realised = sum(float(t.get("size") or 0) for t in since_freeze)

    # The rolling stand-down window, reported so the owner can see the short-run
    # brake coming before it fires rather than being surprised by it.
    roll_cutoff = now - dt.timedelta(days=bot.ROLLING_LOSS_DAYS)
    rolling = [t for t in since_freeze if (_tx_time(t) or now) >= roll_cutoff]
    rolling_r = (sum(float(t.get("size") or 0) for t in rolling) / risk_cash
                 if risk_cash else None)

    markets = []
    for epic in bot.EPICS:
        try:
            candles = client.candles(epic, bot.RESOLUTION, count=200)
            closes, highs, lows = bot.ohlc(candles)
            completed = closes[:-1]
            daily = client.candles(epic, "DAY", 6)
            dcl = [(float(c["closePrice"]["bid"]) + float(c["closePrice"]["ask"])) / 2 for c in daily]
            cfg = bot.instrument_config(epic)
            markets.append({
                "instrument": epic,
                "rsi14_last_completed_4h": round(bot.rsi(completed, bot.RSI_PERIOD) or 0, 1),
                "adx14_last_completed_4h": round(bot.adx_series(highs[:-1], lows[:-1], completed, bot.ADX_PERIOD)[-1] or 0, 1),
                "band": "%.0f/%.0f" % (cfg["oversold"], cfg["overbought"]),
                "stop_pct": cfg["stop_pct"],
                "last_close": round(closes[-1], 5),
                "change_24h_pct": round((dcl[-1] / dcl[-2] - 1) * 100, 2) if len(dcl) >= 2 else None,
                "change_5d_pct": round((dcl[-1] / dcl[0] - 1) * 100, 2) if len(dcl) >= 6 else None,
                "daily_closes": [round(x, 5) for x in dcl],
            })
        except (CapitalError, KeyError, TypeError, ValueError, IndexError) as exc:
            markets.append({"instrument": epic, "error": str(exc)[:120]})

    events = []
    for e in (news._events() or []):
        if day_start - dt.timedelta(hours=12) <= e["when"] <= now + dt.timedelta(hours=24):
            events.append({"when_utc": e["when"].strftime("%Y-%m-%d %H:%M"), "currency": e["ccy"], "event": e["title"]})

    return {
        "as_of_utc": now.strftime("%Y-%m-%d %H:%M"),
        "account": {"balance": bal.get("balance"), "open_pnl": bal.get("profitLoss"),
                    "equity": (float(bal.get("balance") or 0) + float(bal.get("profitLoss") or 0)),
                    "risk_per_trade_cash": round(risk_cash, 2)},
        "rule": {"instruments": bot.EPICS, "candles": bot.RESOLUTION, "reward_to_risk": bot.REWARD_TO_RISK,
                 "adx_limit": bot.REGIME_ADX_MAX, "cooldown_minutes": bot.COOLDOWN_MINUTES,
                 "max_positions": bot.MAX_CONCURRENT_POSITIONS, "max_same_usd_side": bot.MAX_SAME_USD_SIDE,
                 "daily_loss_limit_pct": bot.DAILY_LOSS_LIMIT_PCT, "news_blackout_minutes": bot.NEWS_BLACKOUT_MINUTES},
        "record": {"strategy_frozen_at": bot.STRATEGY_FROZEN_AT, "trades_closed": len(since_freeze),
                   "realised_cash": round(realised, 2), "realised_R": round(realised / risk_cash, 1) if risk_cash else None,
                   "kill_rule": "stop if <= %+.0fR after %d trades" % (bot.KILL_BELOW_R, bot.KILL_AFTER_TRADES),
                   "rolling_window_days": bot.ROLLING_LOSS_DAYS,
                   "rolling_R": round(rolling_r, 1) if rolling_r is not None else None,
                   "rolling_trades": len(rolling),
                   "rolling_stand_down_at_R": bot.ROLLING_LOSS_R},
        "today": {
            "closed": [{"when_utc": (t.get("dateUtc") or "")[:16], "instrument": t.get("instrumentName"),
                        "pnl": float(t.get("size") or 0)} for t in sorted(today_closed, key=lambda t: t.get("dateUtc", ""))],
            "financing": [{"instrument": t.get("instrumentName"), "amount": float(t.get("size") or 0)} for t in today_fees],
            "open_positions": positions,
        },
        "markets": markets,
        "high_impact_events_window": events,
        "note_on_closes": "A close at the stop shows as roughly -1x risk_per_trade_cash; at the target roughly +1.5x. Anything else was a flatten (Friday close, overnight cost, or a manual close).",
    }


# ------------------------------------------------------------ render


def _outcome(pnl: float, risk_cash: float) -> str:
    """
    What a close probably was, from its size in R. A heuristic, and labelled as
    one in the report: the broker's transaction row does not say "stop" or
    "target", it just says closed and by how much.
    """
    if not risk_cash:
        return "?"
    r = pnl / risk_cash
    if r <= -0.8:
        return "stop"
    if r >= bot.REWARD_TO_RISK * 0.8:
        return "target"
    if abs(r) < 0.15:
        return "flat (break-even or flatten)"
    return "closed early (flatten, or the stop had moved)"


def render_facts(data: Dict[str, Any]) -> str:
    """
    The day, straight from the broker's numbers, with no model involved.

    This is the report. The narrative from write_report() is an extra on top
    when an API key happens to be set -- it is not what makes the day legible,
    and the record has to survive a missing key, an expired card or an API
    outage. Five dated files by Friday is the whole point.
    """
    acct, rec, today = data["account"], data["record"], data["today"]
    risk_cash = float(acct.get("risk_per_trade_cash") or 0)
    out: List[str] = []

    out.append("## What happened")
    closed = today.get("closed") or []
    if closed:
        total = sum(c["pnl"] for c in closed)
        out.append("%d position(s) closed, %+.2f in total:\n" % (len(closed), total))
        for c in closed:
            out.append("- `%s`  **%s**  %+.2f  (%.1fR, %s)"
                       % (c["when_utc"][11:16], c["instrument"], c["pnl"],
                          c["pnl"] / risk_cash if risk_cash else 0.0,
                          _outcome(c["pnl"], risk_cash)))
    else:
        out.append("No positions closed today.")
    fees = today.get("financing") or []
    if fees:
        out.append("\nOvernight financing: " +
                   ", ".join("%s %+.2f" % (f["instrument"], f["amount"]) for f in fees))

    out.append("\n## Open now")
    positions = today.get("open_positions") or []
    if positions:
        for p in positions:
            stop = p.get("stop")
            out.append("- **%s %s** size %s @ %s — stop %s, target %s, open P&L %s%s (since %s)"
                       % (p.get("instrument"), p.get("direction"), p.get("size"), p.get("entry"),
                          stop if stop not in (None, "") else "**MISSING**",
                          p.get("target"), p.get("open_pnl"),
                          "" if stop not in (None, "") else "  <- no broker-side stop",
                          p.get("opened")))
    else:
        out.append("Flat — no open positions.")

    out.append("\n## Where the market is")
    out.append("| pair | last | RSI(14) | ADX(14) | band | stop | 24h | 5d |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m in data.get("markets") or []:
        if m.get("error"):
            out.append("| %s | — | — | — | — | — | — | *%s* |" % (m["instrument"], m["error"]))
            continue
        out.append("| %s | %s | %s | %s | %s | %.2f%% | %s%% | %s%% |"
                   % (m["instrument"], m["last_close"], m["rsi14_last_completed_4h"],
                      m["adx14_last_completed_4h"], m["band"], m["stop_pct"],
                      m.get("change_24h_pct"), m.get("change_5d_pct")))
    out.append("\nRSI and ADX are as of the last COMPLETED 4-hour bar — the values the "
               "rule actually decides on. Below the band's low is a buy signal when "
               "ranging, above its high a sell; ADX at or above %s means trending, and "
               "the signal is taken with the trend instead."
               % data["rule"]["adx_limit"])

    out.append("\n## Record")
    out.append("Since the rule was frozen (%s): **%d trades, %+.2f = %sR**."
               % (rec["strategy_frozen_at"][:10], rec["trades_closed"],
                  rec["realised_cash"], rec["realised_R"]))
    out.append("- Kill rule: %s. *(Needs a human to clear.)*" % rec["kill_rule"])
    if rec.get("rolling_R") is not None:
        out.append("- Last %.0f days: %d trades, %+.1fR. Stand-down at %+.1fR. "
                   "*(Clears itself as trades age out.)*"
                   % (rec["rolling_window_days"], rec["rolling_trades"],
                      rec["rolling_R"], rec["rolling_stand_down_at_R"]))
    out.append("- 1R = $%.2f (%.1f%% of a $%s balance)."
               % (risk_cash, bot.RISK_PER_TRADE_PCT, acct.get("balance")))

    events = data.get("high_impact_events_window") or []
    if events:
        out.append("\n## High-impact events in the window")
        for e in events:
            out.append("- `%s` **%s** %s" % (e["when_utc"], e["currency"], e["event"]))

    out.append("\n*Outcome labels (stop / target / flatten) are inferred from each close's "
               "size in R; the broker's history does not state which it was.*")
    return "\n".join(out)


# ------------------------------------------------------------ write


def write_report(data: Dict[str, Any]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": "Today's data:\n\n" + json.dumps(data, indent=1)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined: %s" % (response.stop_details and response.stop_details.explanation))
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("empty report (stop_reason=%s)" % response.stop_reason)
    log.info("analyst: %s in=%d out=%d", response.model, response.usage.input_tokens, response.usage.output_tokens)
    return text


def save(text: str, data: Dict[str, Any], day: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "%s.md" % day)
    rec = data["record"]
    header = ("# %s\n\n*Balance %.2f · record since %s: %d trades, %+.2f (%sR) · written %s UTC*\n\n"
              % (day, float(data["account"]["balance"] or 0), rec["strategy_frozen_at"][:10],
                 rec["trades_closed"], rec["realised_cash"], rec["realised_R"], data["as_of_utc"][11:]))
    with open(path, "w") as fh:
        fh.write(header + text + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--if-due", action="store_true", help="run once per UTC day, after %d:00" % RUN_AFTER_UTC_HOUR)
    g.add_argument("--now", action="store_true", help="run immediately")
    g.add_argument("--print-input", action="store_true", help="gather and print the data, no API call")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    now = dt.datetime.utcnow()
    day = now.date().isoformat()
    state = bot.load_state()
    if args.if_due:
        if now.hour < RUN_AFTER_UTC_HOUR:
            log.info("analyst: not due (before %02d:00 UTC)", RUN_AFTER_UTC_HOUR); return 0
        if state.get("analyst_last") == day:
            log.info("analyst: already written today"); return 0

    client = CapitalClient(); client.login()
    data = gather(client, now)
    if args.print_input:
        print(json.dumps(data, indent=1)); return 0

    # The facts always get written. The narrative is added on top when a key is
    # set and the call works -- it must never be the reason a day goes unrecorded.
    text = render_facts(data)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            text = write_report(data) + "\n\n---\n\n" + text
        except Exception as exc:
            log.warning("analyst: narrative failed (%s) - writing the facts alone", exc)
    else:
        log.info("analyst: ANTHROPIC_API_KEY not set - writing the facts without a narrative")

    path = save(text, data, day)
    state["analyst_last"] = day
    bot.save_state(state)
    log.info("analyst: wrote %s", path)
    print("")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CapitalError, RuntimeError) as exc:
        log.error("analyst failed: %s", exc)
        sys.exit(1)
