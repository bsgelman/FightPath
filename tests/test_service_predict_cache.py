"""Tests for service.predict()'s process-lifetime prediction cache.

Never touches real models/joblibs — predict_fight is monkeypatched to a fast
fake and service._state is populated with minimal fixtures.
"""
import sys
import threading
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

import ufc.api.service as service
import ufc.inference.predict_core as predict_core


@pytest.fixture(autouse=True)
def _reset_cache_and_state(monkeypatch):
    """Every test gets a clean cache and a minimal fake service._state."""
    service._predict_cache.clear()
    monkeypatch.setitem(service._state, "models", {})
    monkeypatch.setitem(service._state, "fighters_df", "fake-fighters-df")
    monkeypatch.setitem(service._state, "pre_fight_state", "fake-pre-fight-state")
    monkeypatch.setitem(service._state, "ref_history_df", "fake-ref-history-df")
    monkeypatch.setitem(service._state, "records", {"1": (10, 2, 0), "2": (5, 5, 1)})
    monkeypatch.setitem(service._state, "prop_models_loaded", True)  # skip real prop-model loading
    yield
    service._predict_cache.clear()


def _fake_pred(red_id="1", blue_id="2", **overrides):
    base = dict(red_id=red_id, blue_id=blue_id, red_name="Red Fighter", blue_name="Blue Fighter",
                record_red=None, record_blue=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def _predict(**overrides):
    kwargs = dict(red="A", blue="B", rounds=3, is_title=False, event_date=date(2026, 7, 11))
    kwargs.update(overrides)
    return service.predict(**kwargs)


class TestPredictCacheHits:
    def test_second_identical_call_is_a_cache_hit(self, monkeypatch):
        calls = []
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: calls.append(kw) or _fake_pred())

        p1 = _predict()
        p2 = _predict()

        assert len(calls) == 1
        assert p1 is p2

    def test_different_fighters_are_separate_entries(self, monkeypatch):
        calls = []
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: calls.append(kw) or _fake_pred())

        _predict(red="A", blue="B")
        _predict(red="C", blue="D")

        assert len(calls) == 2

    def test_different_rounds_are_separate_entries(self, monkeypatch):
        calls = []
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: calls.append(kw) or _fake_pred())

        _predict(rounds=3)
        _predict(rounds=5)

        assert len(calls) == 2

    def test_different_event_date_is_a_separate_entry(self, monkeypatch):
        calls = []
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: calls.append(kw) or _fake_pred())

        _predict(event_date=date(2026, 7, 11))
        _predict(event_date=date(2026, 8, 1))

        assert len(calls) == 2

    def test_none_vs_empty_string_weight_class_still_distinct_from_real_value(self, monkeypatch):
        calls = []
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: calls.append(kw) or _fake_pred())

        _predict(weight_class=None)
        _predict(weight_class="Lightweight")

        assert len(calls) == 2


class TestRecordsOverlay:
    def test_applied_on_miss_and_preserved_on_hit(self, monkeypatch):
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: _fake_pred(red_id="1", blue_id="2"))

        p1 = _predict()
        assert p1.record_red == (10, 2, 0)
        assert p1.record_blue == (5, 5, 1)

        p2 = _predict()
        assert p2 is p1
        assert p2.record_red == (10, 2, 0)


class TestLruEviction:
    def test_caps_cache_size_evicting_oldest_first(self, monkeypatch):
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: _fake_pred())
        monkeypatch.setattr(service, "_PREDICT_CACHE_MAX", 3)

        for i in range(5):
            _predict(red=f"F{i}")

        assert len(service._predict_cache) == 3
        keys = [k[0] for k in service._predict_cache.keys()]
        assert keys == ["F2", "F3", "F4"]

    def test_cache_hit_touches_lru_order(self, monkeypatch):
        monkeypatch.setattr(predict_core, "predict_fight", lambda **kw: _fake_pred())
        monkeypatch.setattr(service, "_PREDICT_CACHE_MAX", 2)

        _predict(red="F0")
        _predict(red="F1")
        _predict(red="F0")  # cache hit -> F0 becomes most-recently-used
        _predict(red="F2")  # miss -> evicts least-recently-used (F1), not F0

        keys = [k[0] for k in service._predict_cache.keys()]
        assert keys == ["F0", "F2"]


class TestThreadSafety:
    def test_concurrent_identical_calls_never_crash_or_corrupt(self, monkeypatch):
        call_lock = threading.Lock()
        calls = []

        def fake_predict_fight(**kw):
            with call_lock:
                calls.append(kw)
            return _fake_pred()

        monkeypatch.setattr(predict_core, "predict_fight", fake_predict_fight)

        results = []
        results_lock = threading.Lock()

        def worker():
            r = _predict()
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is not None for r in results)
        # A rare simultaneous-first-hit may compute more than once, but caching
        # must do SOMETHING — it must not literally recompute for every call.
        assert len(calls) < 20
        assert len(service._predict_cache) == 1
