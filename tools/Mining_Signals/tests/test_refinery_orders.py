"""Unit tests for services.refinery_orders — order data model + store.

Covers RefineryOrder serialization, the dedup / fingerprint helpers,
RefineryOrderStore CRUD, and match_log_completion station + window
matching. Time-dependent tests use freshly-computed ISO timestamps
so the 30-min window passes deterministically.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.refinery_orders import (
    RefineryOrder,
    RefineryOrderStore,
    _commodities_fingerprint,
    _commodities_match,
    _default_name,
    match_log_completion,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _commodity(name: str, scu: int = 10, quality: int = 0) -> dict:
    return {"name": name, "scu": scu, "quality": quality}


class TestRefineryOrderRoundTrip(unittest.TestCase):

    def test_to_dict_round_trip(self):
        order = RefineryOrder(
            id="abc123",
            name="Corundum @ Station",
            station="HUR-L1 Green Glade Station",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            method="Dinyx Solventation",
            cost=125.5,
            processing_seconds=3600,
            submitted_at="2026-01-01T00:00:00+00:00",
            expected_completion="2026-01-01T01:00:00+00:00",
            status="in_process",
        )
        data = order.to_dict()
        restored = RefineryOrder.from_dict(data)
        self.assertEqual(restored.id, order.id)
        self.assertEqual(restored.name, order.name)
        self.assertEqual(restored.commodities, order.commodities)
        self.assertEqual(restored.cost, order.cost)
        self.assertEqual(restored.status, order.status)

    def test_from_dict_supplies_defaults(self):
        # Bare minimum fields — others default
        order = RefineryOrder.from_dict({
            "id": "x",
            "name": "Y",
            "station": "Z",
        })
        self.assertEqual(order.commodities, [])
        self.assertEqual(order.method, "")
        self.assertEqual(order.cost, 0.0)
        self.assertEqual(order.status, "in_process")
        self.assertIsNone(order.completed_at)


class TestTimeRemaining(unittest.TestCase):

    def test_zero_when_no_expected(self):
        order = RefineryOrder(
            id="x", name="x", station="x", expected_completion="",
        )
        self.assertEqual(order.time_remaining_seconds(), 0)

    def test_clamped_to_zero_when_past(self):
        past = _iso(_now() - timedelta(hours=1))
        order = RefineryOrder(
            id="x", name="x", station="x", expected_completion=past,
        )
        self.assertEqual(order.time_remaining_seconds(), 0)

    def test_positive_when_future(self):
        future = _iso(_now() + timedelta(hours=1))
        order = RefineryOrder(
            id="x", name="x", station="x", expected_completion=future,
        )
        secs = order.time_remaining_seconds()
        # Roughly an hour, allowing tiny clock skew across the call
        self.assertGreater(secs, 3500)
        self.assertLessEqual(secs, 3600)

    def test_malformed_returns_zero(self):
        order = RefineryOrder(
            id="x", name="x", station="x", expected_completion="garbage",
        )
        self.assertEqual(order.time_remaining_seconds(), 0)

    def test_complete_status_string(self):
        order = RefineryOrder(
            id="x", name="x", station="x", status="complete",
        )
        self.assertEqual(order.time_remaining_str(), "Complete")

    def test_expected_string_when_zero(self):
        order = RefineryOrder(id="x", name="x", station="x")
        self.assertEqual(order.time_remaining_str(), "Expected")


class TestCommoditiesSummary(unittest.TestCase):

    def test_with_quality(self):
        order = RefineryOrder(
            id="x", name="x", station="x",
            commodities=[
                _commodity("Corundum", scu=24, quality=597),
                _commodity("Aluminum", scu=38, quality=443),
            ],
        )
        summary = order.commodities_summary()
        self.assertIn("Corundum Q597 24 cSCU", summary)
        self.assertIn("Aluminum Q443 38 cSCU", summary)

    def test_without_quality(self):
        order = RefineryOrder(
            id="x", name="x", station="x",
            commodities=[_commodity("Iron", scu=10, quality=0)],
        )
        self.assertEqual(order.commodities_summary(), "Iron 10 cSCU")

    def test_empty_dash(self):
        order = RefineryOrder(id="x", name="x", station="x")
        self.assertEqual(order.commodities_summary(), "—")


class TestFingerprint(unittest.TestCase):

    def test_includes_quality(self):
        a = [_commodity("Corundum", quality=597)]
        b = [_commodity("Corundum", quality=726)]
        self.assertNotEqual(
            _commodities_fingerprint(a), _commodities_fingerprint(b),
        )

    def test_order_independent(self):
        a = [_commodity("Aluminum", quality=443),
             _commodity("Corundum", quality=597)]
        b = [_commodity("Corundum", quality=597),
             _commodity("Aluminum", quality=443)]
        self.assertEqual(
            _commodities_fingerprint(a), _commodities_fingerprint(b),
        )

    def test_empty(self):
        self.assertEqual(_commodities_fingerprint([]), "")

    def test_match_helper(self):
        a = [_commodity("Corundum", quality=597)]
        b = [_commodity("Corundum", quality=597)]
        c = [_commodity("Corundum", quality=720)]
        self.assertTrue(_commodities_match(a, b))
        self.assertFalse(_commodities_match(a, c))


class TestDefaultName(unittest.TestCase):

    def test_with_station_and_commodity(self):
        name = _default_name(
            "HUR-L1 Green Glade Station",
            [_commodity("Corundum")],
        )
        # "{primary} @ {last word of station}"
        self.assertEqual(name, "Corundum @ Station")

    def test_no_commodities(self):
        self.assertEqual(_default_name("HUR-L1 Foo", []), "Order @ Foo")

    def test_no_station(self):
        self.assertEqual(_default_name("", [_commodity("Iron")]), "Iron @ ?")


class TestRefineryOrderStoreCrud(unittest.TestCase):

    def test_init_from_empty_config(self):
        store = RefineryOrderStore({})
        self.assertEqual(store.get_in_process(), [])

    def test_init_loads_existing_orders(self):
        config = {
            "refinery_orders": [
                {
                    "id": "a",
                    "name": "A",
                    "station": "HUR-L1",
                    "status": "in_process",
                },
                {
                    "id": "b",
                    "name": "B",
                    "station": "CRU-L1",
                    "status": "complete",
                },
            ],
        }
        store = RefineryOrderStore(config)
        self.assertEqual(len(store.get_in_process()), 1)
        self.assertEqual(len(store.get_complete()), 1)

    def test_init_skips_malformed(self):
        # Missing required fields → caught and skipped
        config = {
            "refinery_orders": [
                {"id": "a", "name": "A", "station": "HUR-L1"},  # OK
                {"unrecognised_garbage": True},  # rejected
            ],
        }
        store = RefineryOrderStore(config)
        self.assertEqual(len(store.get_in_process()), 1)

    def test_add_order_returns_order(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            method="X",
            cost=100.0,
            processing_seconds=3600,
        )
        self.assertIsNotNone(order)
        self.assertEqual(order.station, "HUR-L1")
        self.assertEqual(order.cost, 100.0)
        self.assertEqual(order.status, "in_process")

    def test_add_order_rejects_all_zero(self):
        store = RefineryOrderStore({})
        result = store.add_order(
            station="HUR-L1",
            commodities=[
                {"name": "Corundum", "qty": 0, "scu": 0},
            ],
        )
        self.assertIsNone(result)
        self.assertEqual(store.get_in_process(), [])

    def test_dedup_same_fingerprint_updates_existing(self):
        store = RefineryOrderStore({})
        first = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            method="Dinyx",
            cost=100.0,
            processing_seconds=3600,
        )
        # Same commodities — should reuse the order, update method/cost
        second = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            method="Cormack",
            cost=200.0,
            processing_seconds=7200,
        )
        self.assertIs(first, second)
        self.assertEqual(first.method, "Cormack")
        self.assertEqual(first.cost, 200.0)
        self.assertEqual(len(store.get_in_process()), 1)

    def test_different_quality_creates_new_order(self):
        store = RefineryOrderStore({})
        store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            processing_seconds=3600,
        )
        store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=720)],
            processing_seconds=3600,
        )
        self.assertEqual(len(store.get_in_process()), 2)

    def test_rename_order(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum")],
        )
        self.assertTrue(store.rename_order(order.id, "My Order"))
        self.assertEqual(order.name, "My Order")

    def test_rename_blank_rejected(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum")],
        )
        self.assertFalse(store.rename_order(order.id, "   "))

    def test_complete_then_pickup(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum")],
        )
        self.assertTrue(store.complete_order(
            order.id, "2026-04-30T10:00:00+00:00", log_event_id="L1",
        ))
        self.assertEqual(order.status, "complete")
        self.assertEqual(order.log_event_id, "L1")
        self.assertEqual(len(store.get_complete()), 1)

        self.assertTrue(store.pickup_order(order.id))
        self.assertEqual(order.status, "picked_up")
        self.assertEqual(len(store.get_picked_up()), 1)
        self.assertEqual(len(store.get_complete()), 0)

    def test_pickup_only_works_on_complete(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum")],
        )
        # Still in_process — pickup must refuse
        self.assertFalse(store.pickup_order(order.id))
        self.assertEqual(order.status, "in_process")

    def test_delete_order(self):
        store = RefineryOrderStore({})
        order = store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum")],
        )
        self.assertTrue(store.delete_order(order.id))
        self.assertIsNone(store.get_order(order.id))
        # Second delete is a no-op
        self.assertFalse(store.delete_order(order.id))

    def test_to_config_list_round_trip(self):
        store = RefineryOrderStore({})
        store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Corundum", scu=24, quality=597)],
            cost=100.0,
            processing_seconds=3600,
        )
        config_list = store.to_config_list()
        self.assertEqual(len(config_list), 1)

        # Rebuild from the serialized form
        store2 = RefineryOrderStore({"refinery_orders": config_list})
        self.assertEqual(len(store2.get_in_process()), 1)


class TestMatchLogCompletion(unittest.TestCase):

    def _store_with_pending(self, station: str, eta: datetime):
        store = RefineryOrderStore({})
        order = store.add_order(
            station=station,
            commodities=[_commodity("Corundum")],
            processing_seconds=3600,
        )
        # Override the auto-computed expected_completion to a known time
        order.expected_completion = _iso(eta)
        return store, order

    def test_matches_on_station_and_window(self):
        eta = _now() + timedelta(seconds=60)
        store, order = self._store_with_pending("HUR-L1", eta)
        log_event = {
            "location": "HUR-L1",
            "count": 1,
            "timestamp": _iso(_now()),
        }
        matched = match_log_completion(store, log_event)
        self.assertEqual(matched, [order.id])

    def test_no_match_when_outside_window(self):
        # Order ETA is 2h away — outside the 30-min window
        eta = _now() + timedelta(hours=2)
        store, _ = self._store_with_pending("HUR-L1", eta)
        log_event = {
            "location": "HUR-L1",
            "count": 1,
            "timestamp": _iso(_now()),
        }
        self.assertEqual(match_log_completion(store, log_event), [])

    def test_no_match_when_station_differs(self):
        eta = _now() + timedelta(seconds=60)
        store, _ = self._store_with_pending("HUR-L1", eta)
        log_event = {
            "location": "CRU-L1",
            "count": 1,
            "timestamp": _iso(_now()),
        }
        self.assertEqual(match_log_completion(store, log_event), [])

    def test_partial_station_match(self):
        # Long stored station name; log gives the short variant.
        eta = _now() + timedelta(seconds=60)
        store, order = self._store_with_pending(
            "HUR-L1 Green Glade Station", eta,
        )
        log_event = {
            "location": "HUR-L1",
            "count": 1,
            "timestamp": _iso(_now()),
        }
        matched = match_log_completion(store, log_event)
        self.assertEqual(matched, [order.id])

    def test_malformed_timestamp_returns_empty(self):
        eta = _now() + timedelta(seconds=60)
        store, _ = self._store_with_pending("HUR-L1", eta)
        log_event = {
            "location": "HUR-L1",
            "count": 1,
            "timestamp": "garbage",
        }
        self.assertEqual(match_log_completion(store, log_event), [])

    def test_count_caps_matches(self):
        # Two pending orders at the same station, both inside window.
        # Log says count=1 → only one match returned.
        eta = _now() + timedelta(seconds=60)
        store, _ = self._store_with_pending("HUR-L1", eta)
        store.add_order(
            station="HUR-L1",
            commodities=[_commodity("Iron")],  # different fingerprint
            processing_seconds=3600,
        ).expected_completion = _iso(_now() + timedelta(seconds=120))

        log_event = {
            "location": "HUR-L1",
            "count": 1,
            "timestamp": _iso(_now()),
        }
        matched = match_log_completion(store, log_event)
        self.assertEqual(len(matched), 1)


class TestLogOnlyCompletion(unittest.TestCase):

    def test_creates_complete_order(self):
        store = RefineryOrderStore({})
        order = store.add_log_only_completion({
            "id": "evt-1",
            "location": "HUR-L1",
            "timestamp": "2026-04-30T10:00:00+00:00",
        })
        self.assertEqual(order.id, "evt-1")
        self.assertEqual(order.status, "complete")
        self.assertEqual(order.log_event_id, "evt-1")
        self.assertEqual(order.completed_at, "2026-04-30T10:00:00+00:00")
        self.assertEqual(len(store.get_complete()), 1)


if __name__ == "__main__":
    unittest.main()
