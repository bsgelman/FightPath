"""Tests for the in-memory rate limiter (no FastAPI app / model load needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import ufc.api.ratelimit as ratelimit


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    """Duck-types the subset of starlette.Request that check_rate_limit uses."""
    def __init__(self, path="/api/predict", xff=None, peer="10.0.0.1"):
        self._path = path
        self._headers = {"x-forwarded-for": xff} if xff else {}
        self.client = _FakeClient(peer)

    @property
    def headers(self):
        return self._headers

    @property
    def url(self):
        class _U:
            path = self._path
        return _U()


def _reset():
    ratelimit._buckets.clear()
    ratelimit._global_bucket.clear()
    ratelimit._last_prune = 0.0


class TestClientIp:
    def test_rightmost_xff_hop_used_not_leftmost(self):
        # Client-forgeable leftmost entry must NOT determine the bucket key —
        # only the trusted-proxy-appended rightmost hop should.
        req = _FakeRequest(xff="1.2.3.4, 9.9.9.9")
        assert ratelimit._client_ip(req) == "9.9.9.9"

    def test_single_hop_xff(self):
        req = _FakeRequest(xff="5.5.5.5")
        assert ratelimit._client_ip(req) == "5.5.5.5"

    def test_falls_back_to_socket_peer(self):
        req = _FakeRequest(xff=None, peer="7.7.7.7")
        assert ratelimit._client_ip(req) == "7.7.7.7"


class TestRateLimitBypass:
    def test_rotating_leftmost_xff_still_hits_the_limit(self):
        # The bug this fixes: bucket key = leftmost XFF entry (client-controlled).
        # Rotating it used to yield a fresh per-key bucket every request — total
        # bypass. The rightmost hop stays fixed (the real client, appended by the
        # trusted proxy), so per-key limiting now actually limits.
        _reset()
        hit_429 = 0
        for i in range(100):
            spoofed_leftmost = f"6{i}.6{i}.6{i}.6{i}"
            req = _FakeRequest(path="/api/predict", xff=f"{spoofed_leftmost}, 9.9.9.9")
            resp = ratelimit.check_rate_limit(req)
            if resp is not None:
                assert resp.status_code == 429
                hit_429 += 1
        assert hit_429 > 0, "rotating leftmost XFF bypassed the rate limit entirely"

    def test_stable_ip_hits_strict_limit_at_configured_count(self):
        _reset()
        max_req, _ = ratelimit.STRICT_LIMIT
        allowed = 0
        for _ in range(max_req + 5):
            req = _FakeRequest(path="/api/predict", xff="1.1.1.1")
            resp = ratelimit.check_rate_limit(req)
            if resp is None:
                allowed += 1
        assert allowed == max_req

    def test_global_ceiling_caps_many_distinct_keys(self):
        # Many genuinely distinct keys (real rightmost IPs, not spoofed
        # leftmost), each well under its own per-key strict limit, must still
        # be capped in aggregate by GLOBAL_LIMIT -- this is what stays true
        # even at the _MAX_TRACKED_KEYS eviction point, where the per-key
        # dict gets cleared and per-key counters alone would offer no
        # protection at all.
        _reset()
        g_max, _ = ratelimit.GLOBAL_LIMIT
        n_keys = g_max + 50  # one request per distinct key, more keys than g_max
        allowed = 0
        for i in range(n_keys):
            req = _FakeRequest(path="/api/predict", xff=f"real-client-{i}")
            resp = ratelimit.check_rate_limit(req)
            if resp is None:
                allowed += 1
        assert allowed <= g_max
        assert allowed < n_keys, "global ceiling never triggered despite exceeding it in aggregate"
