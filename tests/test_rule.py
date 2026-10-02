"""
The rule's own tests. No network, no credentials, no broker -- so they can run
on every push and in a one-minute Actions job.

Why these exist: v2 and v3 both backtested, went live, and never placed a
single trade. Nothing failed; the live rule and the scored rule had simply
drifted apart. The same class of bug produced zero trades on daily bars (the
pre-gap check) and re-entry four minutes after a stop-out (the cooldown). All
of them were silent and all of them cost money or opportunity.

So: prove the live decision and the backtested decision are the same decision,
prove the brakes engage when they should, and prove the watchdog notices.

    python3 -m unittest discover -s tests -v
"""

import datetime as dt
import logging
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest
import bot
import capital_client

logging.disable(logging.CRITICAL)      # the rule logs on every bar; not useful here


# --------------------------------------------------------------- fixtures


# A Wednesday, mid-afternoon UTC: a normal trading hour with nothing special
# about it. The live pass refuses to open inside the no-new-trades window
# (from 19:00 UTC) and refuses outright on a Friday evening, because a fresh
# position would only be flattened again before the weekend. Both are correct
# trading behaviour -- but a positive control that asks "did an order get
# placed?" against the wall clock quietly becomes a calendar test, and fails
# every Friday between 19:00 and the swap. So TestLivePass pins the clock.
A_QUIET_WEDNESDAY = dt.datetime(2026, 9, 30, 14, 0)


class FrozenClock:
    """
    Stands in for bot's `dt` module with utcnow() pinned. Everything else is
    proxied through to the real datetime module, which is left untouched --
    patching dt.datetime globally would reach every other import in the
    process.
    """

    def __init__(self, frozen):
        class _DateTime(dt.datetime):
            @classmethod
            def utcnow(cls):
                return frozen
        self.datetime = _DateTime

    def __getattr__(self, name):
        return getattr(dt, name)



def candle(ts, close, high=None, low=None, open_=None, spread=0.00008):
    """
    One broker candle in the exact shape /api/v1/prices returns: bid and ask
    kept apart, prices nested under closePrice/highPrice/lowPrice/openPrice.
    `close` is the MID -- the spread is split either side of it.
    """
    high = close if high is None else high
    low = close if low is None else low
    open_ = close if open_ is None else open_
    half = spread / 2.0

    def pair(mid):
        return {"bid": round(mid - half, 6), "ask": round(mid + half, 6)}

    return {
        "snapshotTime": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "closePrice": pair(close), "highPrice": pair(high),
        "lowPrice": pair(low), "openPrice": pair(open_),
    }


def series(mids, start=None, step_hours=4, spread=0.00008):
    """A run of 4-hour candles whose open hours line up with the real feed."""
    # The broker's 4-hour bars open at 02/06/10/14/18/22 UTC; start on one of
    # them so the home-session hours in bot.INSTRUMENTS mean what they mean live.
    start = start or dt.datetime(2026, 1, 5, 2, 0, 0)
    out = []
    for i, mid in enumerate(mids):
        ts = start + dt.timedelta(hours=step_hours * i)
        prev = mids[i - 1] if i else mid
        out.append(candle(ts, mid,
                          high=max(mid, prev) + 0.0004,
                          low=min(mid, prev) - 0.0004,
                          open_=prev, spread=spread))
    return out


def wave(n=400, base=1.1000, amp=0.004, period=17):
    """
    A mean-reverting sine wave with a slow drift. Chosen because it drives RSI
    through both bands repeatedly and ADX across the limit, so the rule's
    branches all get exercised instead of one of them.
    """
    return [base + amp * math.sin(2 * math.pi * i / period) + 0.000015 * i
            for i in range(n)]


def trending(n=400, base=1.1000, step=0.0004, period=11, amp=0.0006):
    """A strong trend with small pullbacks: the ADX-above-limit branch."""
    return [base + step * i + amp * math.sin(2 * math.pi * i / period)
            for i in range(n)]


# ------------------------------------------------- live rule == backtest rule


class TestLiveMatchesBacktest(unittest.TestCase):
    """
    The same candles through both paths must yield the same entries.

    bot.entry_side() is the live decision (what evaluate_epic acts on) and
    backtest.run() is the scored decision. They now share regime_decision(),
    and these tests are what prove the sharing is real rather than nominal.
    """

    def _entries_from_backtest(self, candles, epic):
        cfg = bot.instrument_config(epic)
        bars = backtest.load_bars(candles)
        trades = backtest.run(
            bars, bot.RSI_PERIOD, cfg["oversold"], cfg["overbought"],
            cfg["stop_pct"], bot.REWARD_TO_RISK,
            costs=backtest.Costs(bar_minutes=240, weekend_close=False),
            regime="adx", trend_mode=bot.TREND_MODE, hours=cfg.get("hours"),
        )
        # Keyed by the bar the trade opened on: run() fills at bar i+1's open,
        # and its Trade.opened is that bar's timestamp.
        return {t.opened: t.side for t in trades}

    def _live_side_at(self, candles, epic, i):
        """
        The live decision as of bar i: the bot sees completed bars plus the one
        forming. Slicing to i+2 makes bar i+1 the forming bar, which is the bar
        the backtester would fill on.
        """
        window = candles[: i + 2]
        closes, highs, lows = bot.ohlc(window)
        bar_hour = int(window[-1]["snapshotTime"][11:13])
        return bot.entry_side(closes, highs, lows, epic, bar_hour=bar_hour)

    def _assert_parity(self, mids, epic):
        cfg = bot.instrument_config(epic)
        candles = series(mids)
        opened = self._entries_from_backtest(candles, epic)
        checked = agreed = wanted = 0

        for i in range(60, len(candles) - 1):     # 60 = past ADX/RSI warm-up
            live = self._live_side_at(candles, epic, i)
            entry_bar = candles[i + 1]["snapshotTime"]
            scored = opened.get(entry_bar)
            checked += 1

            # Direction: where the backtester took a trade, the live rule must
            # agree. It holds one position at a time, so it can legitimately
            # MISS an entry the rule allows; it can never take a different side.
            if scored is not None:
                self.assertEqual(
                    live, scored,
                    "bar %d (%s): backtest entered %s, live rule says %s"
                    % (i, entry_bar, scored, live))
                agreed += 1

            # Frequency: the live rule may not want in on a bar the backtester's
            # own gates forbid. Checked separately because the one-position rule
            # means "the backtest has no trade here" is not evidence either way
            # -- and a live rule that fires MORE often than the scored one is
            # the dangerous direction: the backtested edge stops describing it.
            if live is not None and cfg.get("hours"):
                self.assertIn(
                    int(entry_bar[11:13]), cfg["hours"],
                    "bar %d: live rule wants %s on the %s bar, but the backtester "
                    "only ever enters on hours %s, so it never scored this trade"
                    % (i, live, entry_bar[11:16], cfg["hours"]))
                wanted += 1

        self.assertGreater(checked, 100, "not enough bars exercised to mean anything")
        self.assertGreater(agreed, 0,
                           "the backtester took no trades at all on this series, so "
                           "parity was never actually tested")
        self.assertGreater(wanted, 0,
                           "the live rule never wanted a trade on this series, so the "
                           "frequency check never actually ran")

    def test_parity_ranging_eurusd(self):
        self._assert_parity(wave(), "EURUSD")

    def test_parity_trending_eurusd(self):
        self._assert_parity(trending(), "EURUSD")

    def test_parity_ranging_audusd(self):
        # Different home session (Asia-night bars) and a different stop.
        self._assert_parity(wave(base=0.6600, amp=0.003), "AUDUSD")

    def test_backtest_respects_the_home_session(self):
        """Every entry the backtester takes must open on a home-session bar."""
        cfg = bot.instrument_config("EURUSD")
        candles = series(wave())
        bars = backtest.load_bars(candles)
        trades = backtest.run(
            bars, bot.RSI_PERIOD, cfg["oversold"], cfg["overbought"],
            cfg["stop_pct"], bot.REWARD_TO_RISK,
            costs=backtest.Costs(bar_minutes=240, weekend_close=False),
            regime="adx", trend_mode=bot.TREND_MODE, hours=cfg["hours"],
        )
        self.assertTrue(trades, "no trades to check")
        for t in trades:
            hour = int(t.opened[11:13])
            self.assertIn(hour, cfg["hours"],
                          "entered on the %02d:00 bar, outside %s" % (hour, cfg["hours"]))

    def test_live_rule_refuses_outside_the_home_session(self):
        """
        The same candles that produce a signal on a home-session bar must
        produce nothing when that bar is outside the session.
        """
        epic = "EURUSD"
        cfg = bot.instrument_config(epic)
        candles = series(wave())
        found = None
        for i in range(60, len(candles) - 1):
            side = self._live_side_at(candles, epic, i)
            if side:
                found = i
                break
        self.assertIsNotNone(found, "no signal found to test the session filter with")

        closes, highs, lows = bot.ohlc(candles[: found + 2])
        outside = next(h for h in range(24) if h not in cfg["hours"])
        self.assertIsNone(
            bot.entry_side(closes, highs, lows, epic, bar_hour=outside),
            "traded on the %02d:00 bar, which is not in %s" % (outside, cfg["hours"]))


# ----------------------------------------------------------- the regime rule


class TestRegimeDecision(unittest.TestCase):
    """
    The branch table, stated once. If someone edits regime_decision(), these
    say what the rule was supposed to be.
    """

    def test_ranging_fades_the_move(self):
        call = bot.regime_decision("BUY", adx=12.0, pdi=20.0, mdi=18.0, adx_max=25)
        self.assertEqual(call.side, "BUY")
        self.assertFalse(call.trending)

    def test_trending_against_the_signal_flips_it(self):
        # RSI oversold (BUY) in a downtrend: sell weakness, with the trend.
        call = bot.regime_decision("BUY", adx=30.0, pdi=15.0, mdi=25.0,
                                   adx_max=25, trend_mode="invert")
        self.assertEqual(call.side, "SELL")
        self.assertTrue(call.trending)

    def test_trending_with_the_signal_does_nothing(self):
        # RSI oversold (BUY) in an uptrend: stretched with the trend, no trade.
        call = bot.regime_decision("BUY", adx=30.0, pdi=25.0, mdi=15.0,
                                   adx_max=25, trend_mode="invert")
        self.assertIsNone(call.side)
        self.assertTrue(call.trending)

    def test_trending_stands_aside_when_not_inverting(self):
        call = bot.regime_decision("BUY", adx=30.0, pdi=15.0, mdi=25.0,
                                   adx_max=25, trend_mode="")
        self.assertIsNone(call.side)

    def test_missing_adx_is_not_a_trade(self):
        """An unmeasured regime is not a ranging one."""
        for adx, pdi, mdi in ((None, 20.0, 10.0), (30.0, None, 10.0), (30.0, 20.0, None)):
            self.assertIsNone(
                bot.regime_decision("BUY", adx, pdi, mdi, adx_max=25).side,
                "traded with adx=%s pdi=%s mdi=%s" % (adx, pdi, mdi))

    def test_filter_off_passes_the_signal_through(self):
        call = bot.regime_decision("SELL", adx=None, pdi=None, mdi=None, adx_max=0)
        self.assertEqual(call.side, "SELL")


# ----------------------------------------------------------------- brakes


def closed_trade(when, pnl, name="EURUSD"):
    """A closed-position row in the shape client.transactions() returns."""
    return {"dateUtc": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "instrumentName": name, "size": str(pnl), "note": "Position closed"}


class TestRollingStandDown(unittest.TestCase):

    def setUp(self):
        self.now = dt.datetime(2026, 9, 28, 12, 0, 0)
        self.balance = 127.0                      # 1R = $1.27 at 1% risk
        self.r = self.balance * bot.RISK_PER_TRADE_PCT / 100.0

    def test_quiet_window_does_not_stand_down(self):
        closed = [closed_trade(self.now - dt.timedelta(days=1), -self.r)]
        self.assertIsNone(bot.rolling_veto(closed, self.balance, self.now))

    def test_bad_window_stands_down(self):
        closed = [closed_trade(self.now - dt.timedelta(hours=6 * i), -self.r)
                  for i in range(5)]            # -5R inside five days
        veto = bot.rolling_veto(closed, self.balance, self.now)
        self.assertIsNotNone(veto)
        self.assertIn("ROLLING STAND-DOWN", veto)

    def test_it_expires_by_itself(self):
        """The whole point: no human has to clear it."""
        old = [closed_trade(self.now - dt.timedelta(days=9, hours=i), -self.r)
               for i in range(5)]
        self.assertIsNone(bot.rolling_veto(old, self.balance, self.now),
                          "still standing down on trades older than the window")

    def test_wins_offset_losses_in_the_window(self):
        closed = [closed_trade(self.now - dt.timedelta(hours=2), -self.r * 5),
                  closed_trade(self.now - dt.timedelta(hours=1), self.r * 3)]
        self.assertIsNone(bot.rolling_veto(closed, self.balance, self.now))

    def test_disabled_by_env(self):
        original = bot.ROLLING_LOSS_R
        bot.ROLLING_LOSS_R = 0.0
        try:
            closed = [closed_trade(self.now, -self.r * 99)]
            self.assertIsNone(bot.rolling_veto(closed, self.balance, self.now))
        finally:
            bot.ROLLING_LOSS_R = original


class TestKillSwitch(unittest.TestCase):

    def setUp(self):
        self.balance = 127.0
        self.r = self.balance * bot.RISK_PER_TRADE_PCT / 100.0
        self.now = dt.datetime(2026, 9, 28, 12, 0, 0)

    def test_the_gate_is_twenty_trades_not_sixty(self):
        """
        v1 lost 7.3R in 22 trades. At ~6 trades a week, a 60-trade gate is ten
        weeks before the brake may fire. This is that fix, pinned.
        """
        self.assertLessEqual(bot.KILL_AFTER_TRADES, 20)

    def test_bad_record_past_the_gate_kills(self):
        closed = [closed_trade(self.now, -self.r * 0.5)
                  for _ in range(bot.KILL_AFTER_TRADES)]      # -10R over 20
        veto = bot.scoreboard(closed, self.balance)
        self.assertIsNotNone(veto)
        self.assertIn("KILL SWITCH", veto)

    def test_bad_record_before_the_gate_does_not_kill(self):
        closed = [closed_trade(self.now, -self.r)
                  for _ in range(bot.KILL_AFTER_TRADES - 1)]
        self.assertIsNone(bot.scoreboard(closed, self.balance))

    def test_good_record_past_the_gate_does_not_kill(self):
        closed = [closed_trade(self.now, self.r)
                  for _ in range(bot.KILL_AFTER_TRADES + 5)]
        self.assertIsNone(bot.scoreboard(closed, self.balance))


# ---------------------------------------------------------------- watchdog


class FakeRisk(object):
    """Enough RiskEngine for the watchdog: alert-once-a-day, no disk."""

    def __init__(self):
        self.seen = set()
        self.alerted = []

    def first_time_today(self, key):
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


class FakeClient(object):
    def __init__(self, weekend_market=True):
        self.weekend_market = weekend_market

    def market(self, epic):
        hours = {} if self.weekend_market else {"sat": ["00:00-23:59"]}
        return {"instrument": {"openingHours": hours}}


def position(epic="EURUSD", direction="BUY", size=300.0, stop=1.09, deal="DEAL1"):
    return {"position": {"dealId": deal, "direction": direction, "size": size,
                         "level": 1.1, "stopLevel": stop, "profitLevel": 1.12},
            "market": {"epic": epic}}


class TestWatchdog(unittest.TestCase):

    def setUp(self):
        self.risk = FakeRisk()
        self.client = FakeClient()
        self.monday = dt.datetime(2026, 9, 28, 12, 0, 0)    # a Monday
        self.saturday = dt.datetime(2026, 9, 26, 12, 0, 0)  # a Saturday
        self.sent = []
        self._real_send = bot.send_alert
        bot.send_alert = lambda subject, lines: self.sent.append((subject, lines))

    def tearDown(self):
        bot.send_alert = self._real_send

    def test_healthy_pass_finds_nothing(self):
        found = bot.watchdog([position()], 30.0, self.risk, {}, self.client, self.monday)
        self.assertEqual(found, [])
        self.assertEqual(self.sent, [])

    def test_position_without_a_stop_is_found(self):
        found = bot.watchdog([position(stop=None)], 30.0, self.risk, {},
                             self.client, self.monday)
        self.assertEqual(len(found), 1)
        self.assertIn("NO broker-side stop", found[0])
        self.assertEqual(len(self.sent), 1, "no email sent for an unstopped position")

    def test_zero_stop_counts_as_missing(self):
        found = bot.watchdog([position(stop=0)], 30.0, self.risk, {},
                             self.client, self.monday)
        self.assertEqual(len(found), 1)

    def test_pinger_gap_is_found(self):
        found = bot.watchdog([], 95.0, self.risk, {}, self.client, self.monday)
        self.assertEqual(len(found), 1)
        self.assertIn("95 minutes", found[0])

    def test_first_ever_pass_is_not_a_gap(self):
        self.assertEqual(bot.watchdog([], None, self.risk, {}, self.client, self.monday), [])

    def test_weekend_position_on_a_closing_market_is_found(self):
        found = bot.watchdog([position()], 30.0, self.risk, {},
                             self.client, self.saturday)
        self.assertEqual(len(found), 1)
        self.assertIn("still open over the weekend", found[0])

    def test_weekend_position_on_a_saturday_market_is_fine(self):
        found = bot.watchdog([position(epic="BTCUSD")], 30.0, self.risk, {},
                             FakeClient(weekend_market=False), self.saturday)
        self.assertEqual(found, [])

    def test_the_same_problem_emails_once_a_day(self):
        for _ in range(5):
            bot.watchdog([position(stop=None)], 30.0, self.risk, {},
                         self.client, self.monday)
        self.assertEqual(len(self.sent), 1, "an unstopped position emailed every pass")

    def test_a_broker_error_in_the_weekend_check_is_swallowed(self):
        """The one call the watchdog makes to the broker must not be fatal."""
        class Sulking(object):
            def market(self, epic):
                raise bot.CapitalError("503 from the broker")
        found = bot.watchdog([position()], 30.0, self.risk, {}, Sulking(), self.saturday)
        self.assertEqual(found, [])       # cannot tell, so says nothing

    def test_an_unreadable_stop_counts_as_missing(self):
        """A stop we cannot parse is a stop we cannot rely on."""
        for bad in ("", "0", "not-a-number", 0.0):
            self.risk = FakeRisk()
            found = bot.watchdog([position(stop=bad)], 30.0, self.risk, {},
                                 self.client, self.monday)
            self.assertEqual(len(found), 1, "stopLevel=%r was treated as a real stop" % bad)

    def test_a_real_stop_as_a_string_is_accepted(self):
        found = bot.watchdog([position(stop="1.0925")], 30.0, self.risk, {},
                             self.client, self.monday)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ------------------------------------------------------------ daily report


import analyst      # noqa: E402  (after the sys.path insert above)


def sample_data(closed=None, positions=None, markets=None, events=None, **rec):
    """A gather() payload in the shape analyst.gather() returns."""
    record = {"strategy_frozen_at": "2026-09-26T11:00:00", "trades_closed": 0,
              "realised_cash": 0.0, "realised_R": 0.0,
              "kill_rule": "stop if <= -5R after 20 trades",
              "rolling_window_days": 5.0, "rolling_R": 0.0, "rolling_trades": 0,
              "rolling_stand_down_at_R": -4.0}
    record.update(rec)
    return {
        "as_of_utc": "2026-09-28 21:07",
        "account": {"balance": 127.27, "open_pnl": 0.21, "equity": 127.48,
                    "risk_per_trade_cash": 1.27},
        "rule": {"instruments": ["EURUSD"], "candles": "HOUR_4", "reward_to_risk": 1.5,
                 "adx_limit": 25.0, "cooldown_minutes": 240, "max_positions": 3,
                 "max_same_usd_side": 2, "daily_loss_limit_pct": 3.0,
                 "news_blackout_minutes": 30},
        "record": record,
        "today": {"closed": closed or [], "financing": [],
                  "open_positions": positions or []},
        "markets": markets if markets is not None else [
            {"instrument": "EURUSD", "rsi14_last_completed_4h": 41.1,
             "adx14_last_completed_4h": 18.3, "band": "40/60", "stop_pct": 0.3,
             "last_close": 1.17242, "change_24h_pct": -0.12, "change_5d_pct": 0.44,
             "daily_closes": [1.17]}],
        "high_impact_events_window": events or [],
        "note_on_closes": "...",
    }


class TestDailyReport(unittest.TestCase):
    """
    The report runs unattended at 21:00 with no one watching. A crash on a None
    field means a day with no record, which is exactly what this work was for.
    """

    def test_quiet_day(self):
        text = analyst.render_facts(sample_data())
        self.assertIn("No positions closed today.", text)
        self.assertIn("Flat — no open positions.", text)
        self.assertIn("## Record", text)

    def test_day_with_trades(self):
        text = analyst.render_facts(sample_data(closed=[
            {"when_utc": "2026-09-28T14:32", "instrument": "EURUSD", "pnl": 1.90},
            {"when_utc": "2026-09-28T16:02", "instrument": "AUDUSD", "pnl": -1.27},
        ]))
        self.assertIn("2 position(s) closed", text)
        self.assertIn("target", text)      # +1.90 on 1.27 risk = 1.5R
        self.assertIn("stop", text)        # -1.27 = -1.0R

    def test_missing_stop_is_shouted_about(self):
        text = analyst.render_facts(sample_data(positions=[
            {"instrument": "EURUSD", "direction": "BUY", "size": 300, "entry": 1.172,
             "stop": None, "target": 1.177, "open_pnl": 0.21, "opened": "2026-09-28 10:07"}]))
        self.assertIn("MISSING", text)
        self.assertIn("no broker-side stop", text)

    def test_survives_a_market_error(self):
        text = analyst.render_facts(sample_data(markets=[
            {"instrument": "GBPUSD", "error": "price feed timeout"}]))
        self.assertIn("price feed timeout", text)

    def test_survives_missing_numbers(self):
        """None in any of the reported fields must not raise."""
        data = sample_data(realised_R=None, rolling_R=None)
        data["markets"][0]["change_24h_pct"] = None
        data["markets"][0]["change_5d_pct"] = None
        data["account"]["risk_per_trade_cash"] = 0
        text = analyst.render_facts(data)
        self.assertIn("## Record", text)

    def test_no_api_key_still_produces_a_report(self):
        """The whole point of the change: the facts do not need a model."""
        key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            text = analyst.render_facts(sample_data())
            self.assertGreater(len(text), 300)
        finally:
            if key is not None:
                os.environ["ANTHROPIC_API_KEY"] = key

    def test_events_are_listed(self):
        text = analyst.render_facts(sample_data(events=[
            {"when_utc": "2026-09-28 12:30", "currency": "USD", "event": "Nonfarm payrolls"}]))
        self.assertIn("Nonfarm payrolls", text)

    def test_outcome_labels(self):
        self.assertEqual(analyst._outcome(-1.27, 1.27), "stop")
        self.assertEqual(analyst._outcome(1.90, 1.27), "target")
        self.assertIn("flat", analyst._outcome(0.01, 1.27))
        self.assertIn("closed early", analyst._outcome(-0.50, 1.27))


# -------------------------------------------------------- the live pass


class FakeBroker(object):
    """
    Enough of CapitalClient to run a whole pass. Records the orders it is asked
    to place, so a test can assert on what the bot would actually have done.
    """

    is_demo = False

    def __init__(self, mids=None, balance=127.27, positions=None, transactions=None,
                 epic_hours=None):
        self.mids = mids or wave()
        self.balance = balance
        self._positions = positions or []
        self._transactions = transactions or []
        self.orders = []
        self.closed = []
        # The forming bar's hour decides the home session; default to one that
        # is in EURUSD's set so the rule is allowed to fire.
        self.epic_hours = epic_hours
        self.candles_override = None

    def account(self):
        return {"currency": "USD",
                "balance": {"balance": self.balance, "profitLoss": 0.0,
                            "available": self.balance - 4.0}}

    def positions(self):
        return list(self._positions)

    def transactions(self, since):
        return list(self._transactions)

    def market(self, epic):
        return {
            "snapshot": {"marketStatus": "TRADEABLE", "bid": 1.1000, "offer": 1.1001},
            "instrument": {"currency": "USD", "openingHours": {},
                           "overnightFee": {"longRate": -0.005, "shortRate": -0.005,
                                            "swapChargeTimestamp": None}},
            "dealingRules": {"minDealSize": {"value": 100.0},
                             "minSizeIncrement": {"value": 100.0}},
        }

    def candles(self, epic, resolution, count=100):
        if self.candles_override is not None:
            return list(self.candles_override)
        start = dt.datetime(2026, 1, 5, 2, 0, 0)
        if self.epic_hours is not None:
            # Land the forming bar on a chosen hour.
            shift = (self.epic_hours - 2) % 24
            start += dt.timedelta(hours=shift)
        return series(self.mids, start=start)

    def open_position(self, epic, direction, size, stop_level=None, profit_level=None):
        self.orders.append({"epic": epic, "direction": direction, "size": size,
                            "stop": stop_level, "target": profit_level})
        return {"dealReference": "REF1"}

    def close_position(self, deal_id, direction, size):
        self.closed.append(deal_id)
        return {"dealReference": "REF2"}


def signalling_broker(**kwargs):
    """
    A FakeBroker whose next pass opens a position: the candles are truncated at
    a bar where the live rule fires AND the forming bar is in EURUSD's home
    session. Used as the positive control, so that a test asserting "no order
    was placed" is actually testing the brake and not an absence of signal.
    """
    mids = wave(400)
    for n in range(70, len(mids)):
        candles = series(mids[:n])
        closes, highs, lows = bot.ohlc(candles)
        hour = int(candles[-1]["snapshotTime"][11:13])
        if bot.entry_side(closes, highs, lows, "EURUSD", bar_hour=hour):
            broker = FakeBroker(**kwargs)
            broker.candles_override = candles
            return broker
    raise AssertionError("no candle series found that triggers an entry")


class TestLivePass(unittest.TestCase):
    """
    evaluate_once() is the function that runs every 30 minutes against real
    money. These do not check the strategy -- they check that a pass completes,
    that the brakes stop it, and that nothing in it raises.
    """

    def setUp(self):
        self.tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_state_test.json")
        self._real_state_path = bot.STATE_PATH
        bot.STATE_PATH = self.tmp                       # never touch the real state.json
        self._real_send = bot.send_alert
        self.sent = []
        bot.send_alert = lambda subject, lines: self.sent.append((subject, lines))
        self._real_news = bot.NEWS_FILTER
        bot.NEWS_FILTER = False                          # no calendar fetch in tests
        self._real_dry = bot.DRY_RUN
        # OFF on purpose: FakeBroker.open_position is the safety net, and with
        # DRY_RUN on the order path never runs -- which would make every
        # "did not open a position" assertion below pass for the wrong reason.
        bot.DRY_RUN = False
        # A pass reads the clock to decide whether it is too late in the day,
        # or too late in the week, to open anything. Left on the wall clock
        # these tests pass or fail by the hour they happen to run at.
        self._real_dt = bot.dt
        bot.dt = FrozenClock(A_QUIET_WEDNESDAY)

    def tearDown(self):
        bot.STATE_PATH = self._real_state_path
        bot.send_alert = self._real_send
        bot.NEWS_FILTER = self._real_news
        bot.DRY_RUN = self._real_dry
        bot.dt = self._real_dt
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _pass(self, broker):
        risk = bot.RiskEngine(broker)
        bot.evaluate_once(broker, risk)
        return risk

    def test_a_pass_completes(self):
        broker = FakeBroker()
        self._pass(broker)                               # must not raise

    def test_a_clean_pass_opens_a_position(self):
        """
        The positive control for every "no order was placed" test below: with
        nothing blocking, this exact broker DOES get an order, and the order
        carries a stop and a target.
        """
        broker = signalling_broker()
        self._pass(broker)
        self.assertGreaterEqual(len(broker.orders), 1,
                                "the rule fired but no order was placed")
        for order in broker.orders:
            self.assertIn(order["direction"], ("BUY", "SELL"))
            self.assertGreater(order["size"], 0)
            self.assertIsNotNone(order["stop"], "order placed with no stop level")
            self.assertIsNotNone(order["target"], "order placed with no target level")

    def test_nothing_new_opens_late_on_a_friday(self):
        """
        What broke the Friday 19:07 run: inside the no-new-trades window the
        bot still opens if it would carry the position overnight, but on a
        Friday it would not -- the market closes for the weekend and Monday
        can gap through the stop. So late Friday it must stand aside, and the
        identical pass one day earlier must not.
        """
        friday_evening = dt.datetime(2026, 10, 2, 19, 7)
        thursday_evening = friday_evening - dt.timedelta(days=1)
        self.assertEqual(friday_evening.weekday(), 4)

        bot.dt = FrozenClock(friday_evening)
        friday = signalling_broker()
        self._pass(friday)

        bot.dt = FrozenClock(thursday_evening)
        thursday = signalling_broker()
        self._pass(thursday)

        self.assertEqual(len(friday.orders), 0,
                         "opened a position late on a Friday that the weekend flatten would close")
        self.assertGreaterEqual(len(thursday.orders), 1,
                                "the same hour on a Thursday must still trade")

    def test_the_same_usd_side_cap_holds(self):
        """
        The fake broker serves identical candles for every pair, so all three
        fire at once -- which is the real correlation risk this cap exists for:
        three dollar pairs the same way round is one ~3% bet, not three 1% ones.
        """
        broker = signalling_broker()
        self._pass(broker)
        sides = {}
        for order in broker.orders:
            side = bot.usd_side(order["epic"], order["direction"])
            sides[side] = sides.get(side, 0) + 1
        for side, count in sides.items():
            self.assertLessEqual(count, bot.MAX_SAME_USD_SIDE,
                                 "%d positions %s, cap is %d"
                                 % (count, side, bot.MAX_SAME_USD_SIDE))

    def test_daily_loss_limit_blocks_the_pass(self):
        broker = signalling_broker()
        risk = bot.RiskEngine(broker)
        risk.mark_session_start(200.0)                   # started the day far higher
        bot.evaluate_once(broker, risk)
        self.assertEqual(broker.orders, [], "opened a position after the daily halt")

    def test_a_pass_records_its_own_timestamp(self):
        broker = FakeBroker()
        risk = self._pass(broker)
        self.assertIn("last_pass_utc", risk.state)

    def test_second_pass_sees_no_gap(self):
        broker = FakeBroker()
        risk = bot.RiskEngine(broker)
        self.assertIsNone(risk.mark_pass())              # first ever pass
        gap = risk.mark_pass()
        self.assertIsNotNone(gap)
        self.assertLess(gap, 1.0)

    def test_rolling_stand_down_blocks_the_pass(self):
        """A bad week must stop new entries even when the rule likes the setup."""
        now = dt.datetime.utcnow()
        r = 127.27 * bot.RISK_PER_TRADE_PCT / 100.0
        losses = [closed_trade(now - dt.timedelta(hours=3 * i), -r) for i in range(5)]
        broker = signalling_broker(transactions=losses)
        self._pass(broker)
        self.assertEqual(broker.orders, [], "opened a position while standing down")

    def test_kill_switch_blocks_the_pass(self):
        now = dt.datetime.utcnow()
        r = 127.27 * bot.RISK_PER_TRADE_PCT / 100.0
        # Past the trade gate and past -5R, but spread beyond the rolling window
        # so this is the kill switch talking and not the stand-down.
        losses = [closed_trade(now - dt.timedelta(days=20 + i), -r * 0.4)
                  for i in range(bot.KILL_AFTER_TRADES)]
        frozen = (now - dt.timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S")
        original = bot.STRATEGY_FROZEN_AT
        bot.STRATEGY_FROZEN_AT = frozen
        try:
            broker = signalling_broker(transactions=losses)
            self._pass(broker)
            self.assertEqual(broker.orders, [], "opened a position after the kill switch")
            self.assertTrue(any("KILL SWITCH" in s for s, _ in self.sent),
                            "no kill-switch email")
        finally:
            bot.STRATEGY_FROZEN_AT = original

    def test_unstopped_position_is_reported_during_a_real_pass(self):
        broker = FakeBroker(positions=[position(stop=None)])
        self._pass(broker)
        self.assertTrue(any("watchdog" in s.lower() for s, _ in self.sent),
                        "a live position with no stop did not raise an alert")

    def test_unreadable_history_does_not_stop_the_pass(self):
        class NoHistory(FakeBroker):
            def transactions(self, since):
                raise bot.CapitalError("503 from the broker")
        self._pass(NoHistory())                          # must not raise

    def test_a_watchdog_that_explodes_does_not_stop_the_pass(self):
        """
        The real invariant: a bug in the watchdog must never be the reason the
        bot stops trading. evaluate_once() is what has to survive it.
        """
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_state_wd.json")
        real_path, real_watchdog = bot.STATE_PATH, bot.watchdog
        real_news, real_dry = bot.NEWS_FILTER, bot.DRY_RUN
        bot.STATE_PATH, bot.NEWS_FILTER, bot.DRY_RUN = tmp, False, False

        def exploding(*a, **k):
            raise RuntimeError("bug in the watchdog")
        bot.watchdog = exploding
        try:
            broker = FakeBroker()
            bot.evaluate_once(broker, bot.RiskEngine(broker))     # must not raise
        finally:
            bot.STATE_PATH, bot.watchdog = real_path, real_watchdog
            bot.NEWS_FILTER, bot.DRY_RUN = real_news, real_dry
            if os.path.exists(tmp):
                os.remove(tmp)


class TestBacktestStillRuns(unittest.TestCase):
    """
    The entry block in run() was rewritten to call bot.regime_decision, so every
    mode it supports needs to still execute. Cannot be run against the real
    price feed without credentials, so: synthetic bars, all combinations, and an
    assertion that the trades it returns are internally consistent.
    """

    def setUp(self):
        self.bars = backtest.load_bars(series(wave(300)))
        self.costs = backtest.Costs(bar_minutes=240, weekend_close=False)

    def _run(self, **kwargs):
        return backtest.run(self.bars, 14, 40, 60, 0.3, 1.5, costs=self.costs, **kwargs)

    def test_every_mode_executes(self):
        combos = []
        for regime in (None, "adx", "sma", "both"):
            for trend_mode in (None, "invert", "pullback"):
                combos.append({"regime": regime, "trend_mode": trend_mode})
        for strategy in ("vcp", "momentum", "donchian", "cycle"):
            combos.append({"strategy": strategy})
        for kwargs in combos:
            trades = self._run(**kwargs)          # must not raise
            for t in trades:
                self.assertIn(t.side, ("BUY", "SELL"), "bad side with %s" % kwargs)
                if t.side == "BUY":
                    self.assertLess(t.stop0, t.entry)
                    self.assertGreater(t.target, t.entry)
                else:
                    self.assertGreater(t.stop0, t.entry)
                    self.assertLess(t.target, t.entry)

    def test_the_live_configuration_takes_trades(self):
        """If v4's own settings produce nothing, the refactor broke the rule."""
        trades = self._run(regime="adx", trend_mode="invert")
        self.assertTrue(trades, "the live configuration scored no trades at all")

    def test_standing_aside_trades_less_than_inverting(self):
        """
        v2 (stand aside when trending) must take fewer trades than v4 (trade
        with the trend). Run on a TRENDING series on purpose: on a ranging one
        ADX never reaches the limit, the trending branch never executes, and the
        two modes are identical -- which would make this pass for no reason.

        This is also what proves `trend_mode or ""` works: if None fell through
        to the env default of "invert", the two would be equal here.
        """
        trending_bars = backtest.load_bars(series(trending(300)))
        aside = backtest.run(trending_bars, 14, 40, 60, 0.3, 1.5, costs=self.costs,
                             regime="adx", trend_mode=None)
        invert = backtest.run(trending_bars, 14, 40, 60, 0.3, 1.5, costs=self.costs,
                              regime="adx", trend_mode="invert")
        self.assertEqual(len(aside), 0, "v2 took trades in a trend; it must stand aside")
        self.assertGreater(len(invert), 0, "v4 took no trades in a trend")

    def test_ranging_series_never_reaches_the_trend_branch(self):
        """States the assumption the parity fixtures rely on."""
        closes = [b.mid_close for b in self.bars]
        adx, _, _ = bot.dmi_series([(b.bid_high + b.ask_high) / 2 for b in self.bars],
                                   [(b.bid_low + b.ask_low) / 2 for b in self.bars],
                                   closes, 14)
        self.assertTrue(all(a < bot.REGIME_ADX_MAX for a in adx if a is not None),
                        "the 'ranging' fixture trends somewhere")

    def test_one_position_at_a_time(self):
        trades = self._run(regime="adx", trend_mode="invert")
        for earlier, later in zip(trades, trades[1:]):
            self.assertLessEqual(earlier.closed, later.opened,
                                 "two trades overlapped: %s closed after %s opened"
                                 % (earlier.closed, later.opened))


class TestAccountSelection(unittest.TestCase):
    """
    Which account the bot trades. On 2026-10-01 this silently changed: the
    selection was "whatever /api/v1/accounts lists first", the order moved,
    and the bot read a different, empty account for a day -- reporting a
    balance of zero while the money sat untouched in the other one. It halted
    rather than trading, which is the right failure, but nothing said why.
    """

    def _client(self, accounts):
        client = object.__new__(capital_client.CapitalClient)
        client._request = lambda method, path, **kw: {"accounts": accounts}
        return client

    def setUp(self):
        self._real_env = os.environ.get("CAPITAL_ACCOUNT_ID")
        os.environ.pop("CAPITAL_ACCOUNT_ID", None)
        self.rich = {"accountId": "REAL", "currency": "USD",
                     "balance": {"balance": 134.76, "available": 134.76}}
        self.empty = {"accountId": "OTHER", "currency": "USD",
                      "balance": {"balance": 0.0, "available": 0.0}}

    def tearDown(self):
        os.environ.pop("CAPITAL_ACCOUNT_ID", None)
        if self._real_env is not None:
            os.environ["CAPITAL_ACCOUNT_ID"] = self._real_env

    def test_a_pinned_account_survives_a_reorder(self):
        """The regression test for the incident: order must stop mattering."""
        os.environ["CAPITAL_ACCOUNT_ID"] = "REAL"
        for order in ([self.rich, self.empty], [self.empty, self.rich]):
            chosen = self._client(order).account()
            self.assertEqual(chosen["accountId"], "REAL")
            self.assertEqual(chosen["balance"]["balance"], 134.76)

    def test_unpinned_follows_the_api_order(self):
        """Documents the behaviour that bit us, so a change to it is deliberate."""
        self.assertEqual(self._client([self.empty, self.rich]).account()["accountId"], "OTHER")
        self.assertEqual(self._client([self.rich, self.empty]).account()["accountId"], "REAL")

    def test_an_unpinned_choice_between_accounts_is_logged(self):
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs(capital_client.log, level="WARNING") as caught:
                self._client([self.empty, self.rich]).account()
            self.assertIn("CAPITAL_ACCOUNT_ID", "\n".join(caught.output))
        finally:
            logging.disable(logging.CRITICAL)

    def test_one_account_is_not_ambiguous(self):
        logging.disable(logging.NOTSET)
        try:
            with self.assertNoLogs(capital_client.log, level="WARNING"):
                self._client([self.rich]).account()
        finally:
            logging.disable(logging.CRITICAL)

    def test_a_pin_that_matches_nothing_names_what_is_there(self):
        os.environ["CAPITAL_ACCOUNT_ID"] = "TYPO"
        with self.assertRaises(capital_client.CapitalError) as caught:
            self._client([self.rich, self.empty]).account()
        self.assertIn("REAL", str(caught.exception))
        self.assertIn("OTHER", str(caught.exception))

    def test_no_accounts_at_all_raises(self):
        with self.assertRaises(capital_client.CapitalError):
            self._client([]).account()
