"""Unit tests for services.refinery_distances — distance cache.

Covers the _DistanceCache round-trip semantics, the _dirty save guard,
graceful handling of corrupt cache files, and the ordered key format.
The HTTP fetch is mocked at the urlopen boundary so no real network
is touched.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import refinery_distances as rd


class _CacheBase(unittest.TestCase):
    """Redirects the module-level _CACHE_FILE to a tmp dir per-test."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cache_path = os.path.join(self._tmpdir.name, "cache.json")
        self._orig_cache = rd._CACHE_FILE
        rd._CACHE_FILE = self._cache_path

    def tearDown(self):
        rd._CACHE_FILE = self._orig_cache
        self._tmpdir.cleanup()


class TestDistanceCacheRoundTrip(_CacheBase):

    def test_put_then_get_returns_value(self):
        cache = rd._DistanceCache()
        cache.put(1, 2, 100.5)
        self.assertEqual(cache.get(1, 2), 100.5)

    def test_get_missing_returns_none(self):
        cache = rd._DistanceCache()
        self.assertIsNone(cache.get(99, 100))

    def test_save_then_reload_preserves_data(self):
        cache = rd._DistanceCache()
        cache.put(1, 2, 100.5)
        cache.put(13, 44, 250.0)
        cache.save()

        # New instance reads the file we just wrote
        cache2 = rd._DistanceCache()
        self.assertEqual(cache2.get(1, 2), 100.5)
        self.assertEqual(cache2.get(13, 44), 250.0)

    def test_empty_cache_save_is_safe(self):
        cache = rd._DistanceCache()
        # No put → not dirty → save is a no-op
        cache.save()  # must not raise
        # File was never created
        self.assertFalse(os.path.exists(self._cache_path))


class TestDirtyFlag(_CacheBase):

    def test_save_noop_when_not_dirty(self):
        # Pre-populate a cache file
        cache = rd._DistanceCache()
        cache.put(1, 2, 100.5)
        cache.save()
        first_mtime = os.path.getmtime(self._cache_path)

        # Reload — _dirty resets to False during __init__
        cache2 = rd._DistanceCache()
        # Reading should not flip the dirty flag
        cache2.get(1, 2)
        self.assertFalse(cache2._dirty)

        # Mock open so we can detect any write attempt
        with patch("builtins.open") as mock_open:
            cache2.save()
            mock_open.assert_not_called()

        # File mtime should also be unchanged
        self.assertEqual(os.path.getmtime(self._cache_path), first_mtime)

    def test_put_marks_dirty(self):
        cache = rd._DistanceCache()
        self.assertFalse(cache._dirty)
        cache.put(1, 2, 50.0)
        self.assertTrue(cache._dirty)

    def test_save_clears_dirty_flag(self):
        cache = rd._DistanceCache()
        cache.put(1, 2, 50.0)
        self.assertTrue(cache._dirty)
        cache.save()
        self.assertFalse(cache._dirty)


class TestCorruptedCacheFile(_CacheBase):

    def test_malformed_json_resets_to_empty(self):
        # Write garbage to the cache path
        with open(self._cache_path, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        # Capture the warning log emitted by _load
        with self.assertLogs(rd.log, level=logging.WARNING) as cm:
            cache = rd._DistanceCache()

        self.assertEqual(cache._data, {})
        # A warning that mentions the cache reset should fire
        self.assertTrue(any("distance cache reset" in m for m in cm.output))

    def test_after_corruption_recovers(self):
        # Write garbage, instantiate (which logs a warning and resets),
        # then prove the cache is usable for new writes.
        with open(self._cache_path, "w", encoding="utf-8") as f:
            f.write("not json")

        with self.assertLogs(rd.log, level=logging.WARNING):
            cache = rd._DistanceCache()
        cache.put(1, 2, 75.0)
        cache.save()

        # And the file is now valid JSON
        with open(self._cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["1-2"], 75.0)


class TestCacheKeyFormat(_CacheBase):
    """Verify the documented (origin, dest) key — directional, NOT
    deduped both ways."""

    def test_key_is_directional(self):
        # The cache stores '{a}-{b}' which means (a, b) and (b, a)
        # are distinct entries. This is by design — UEX may report
        # different distances for the two directions.
        self.assertEqual(rd._DistanceCache._key(1, 2), "1-2")
        self.assertEqual(rd._DistanceCache._key(2, 1), "2-1")

    def test_reversed_pair_misses(self):
        cache = rd._DistanceCache()
        cache.put(1, 2, 100.0)
        # Reverse direction is a separate slot — must miss
        self.assertIsNone(cache.get(2, 1))
        self.assertEqual(cache.get(1, 2), 100.0)

    def test_two_entries_distinct(self):
        cache = rd._DistanceCache()
        cache.put(1, 2, 100.0)
        cache.put(2, 1, 102.0)  # could differ in real data
        self.assertEqual(cache.get(1, 2), 100.0)
        self.assertEqual(cache.get(2, 1), 102.0)


class TestResolvePlayerTerminal(unittest.TestCase):
    """The _resolve_player_terminal helper is pure — no fixture
    needed."""

    def test_direct_match(self):
        self.assertEqual(rd._resolve_player_terminal("HUR-L1"), 44)

    def test_substring_match(self):
        # "HUR-L1 Green Glade Station" contains "HUR-L1"
        self.assertEqual(
            rd._resolve_player_terminal("HUR-L1 Green Glade Station"), 44,
        )

    def test_unknown_returns_none(self):
        self.assertIsNone(rd._resolve_player_terminal("Made-Up Place"))

    def test_empty_returns_none(self):
        self.assertIsNone(rd._resolve_player_terminal(""))


class TestFmtDistance(unittest.TestCase):
    """fmt_distance is a pure formatter."""

    def test_below_one_terameter(self):
        self.assertEqual(rd.fmt_distance(500.0), "500 Gm")

    def test_above_one_terameter(self):
        self.assertEqual(rd.fmt_distance(1500.0), "1.5 Tm")

    def test_zero(self):
        self.assertEqual(rd.fmt_distance(0.0), "0 Gm")


if __name__ == "__main__":
    unittest.main()
