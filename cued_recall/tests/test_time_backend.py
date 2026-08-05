"""The keyless clock backend that answers date/time web_search calls.

Search engines serve snippets from stale caches, so a "current date" query used
to come back months out of date and the model repeated it. These tests pin the
narrow intent matcher and the formatter that turns the clock API payload into
an authoritative statement -- the two pure parts that can be tested without
the network.
"""

import pytest

from cued_recall.pipeline import Pipeline, _time_intent


class TestTimeIntent:
    def test_plain_date_queries_match(self):
        for q in ["current date", "current date and time", "today's date",
                  "what is the date", "what day is it", "what time is it",
                  "current time"]:
            assert _time_intent(q), f"{q!r} should be a clock query"

    def test_azerbaijani_date_queries_match(self):
        for q in ["hazırkı tarix", "bugünün tarixi", "saat neçədir",
                  "tarix və saat"]:
            assert _time_intent(q), f"{q!r} should be a clock query"

    def test_unrelated_queries_do_not_match(self):
        for q in ["weather for baku", "current status of the volcano",
                  "send me the latest news today please", "current account",
                  "time machine for sale", ""]:
            assert not _time_intent(q), f"{q!r} must not be hijacked"

    def test_long_prose_never_matches(self):
        # A query embedding "current date" inside a paragraph is a search, not
        # a clock request; only a short, date-shaped query may be routed.
        q = ("can you please find me the current date and time at which the "
             "last lunar eclipse was visible from Baku so I can compare")
        assert not _time_intent(q)


class TestFormatClock:
    def test_full_payload_is_authoritative_and_has_today(self):
        now = {"year": 2026, "month": 8, "day": 6, "dayOfWeek": "Thursday",
               "time": "00:43", "timeZone": "Asia/Baku",
               "dateTime": "2026-08-06T00:43:41"}
        out = Pipeline._format_clock(now, "current date and time")
        assert "Today is Thursday, August 6, 2026." in out
        assert "2026-08-06" in out
        assert "authoritative" in out.lower()
        assert "cached search snippet" in out.lower()

    def test_no_day_of_week_still_gives_the_date(self):
        now = {"year": 2026, "month": 8, "day": 6, "dayOfWeek": "",
               "time": "00:00", "timeZone": "UTC", "dateTime": "2026-08-06T00:00"}
        out = Pipeline._format_clock(now, "current date")
        assert "2026-08-06" in out
        assert "authoritative" in out.lower()

    def test_ignores_older_dates(self):
        # The point of the whole backend: the model must not prefer a stale
        # snippet, a recalled date, or a system hint over the clock answer.
        now = {"year": 2026, "month": 8, "day": 6, "dayOfWeek": "Thursday",
               "time": "00:43", "timeZone": "Asia/Baku",
               "dateTime": "2026-08-06T00:43:41"}
        out = Pipeline._format_clock(now, "current date and time").lower()
        assert "ignore any older date" in out


class TestTimeIntentConfig:
    def test_web_search_config_exposes_the_switch(self, config):
        ws = config.web_search
        assert ws.time_intent is True
        assert ws.time_timezone == ""

    @pytest.mark.asyncio
    async def test_disabling_time_intent_is_respected(self, config, monkeypatch):
        config.web_search.time_intent = False
        # _web_search must skip the clock path when disabled; it falls through
        # to the search chain (which the stub below lets us observe).
        hits = []
        p = Pipeline.__new__(Pipeline)
        p.config = config
        p.wal = None
        monkeypatch.setattr(
            p, "_search_chain", lambda: ["stub"] * (hits.append("chain") or 1))
        monkeypatch.setattr(
            p, "_run_backend",
            lambda *a: (hits.append("backend") or ([], False)))
        out = await p._web_search("current date and time")
        assert "chain" in hits and "backend" in hits
        assert "clock API" not in out
