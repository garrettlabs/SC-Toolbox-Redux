"""Unit tests for services.ledger_store — JSON round-trip + persistence.

Covers _crew_to_dict / _crew_from_dict, ledger_to_dict / ledger_from_dict,
and the save_ledger / load_ledger persistence pair. All disk I/O uses
tempfile.TemporaryDirectory so no fixtures are required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ledger_store import (
    CrewAssignment,
    FleetSupportShip,
    LedgerData,
    MiningTeam,
    PlayerEntry,
    StrikeGroupData,
    _crew_from_dict,
    _crew_to_dict,
    export_player_roster,
    import_player_roster,
    ledger_from_dict,
    ledger_to_dict,
    load_ledger,
    save_ledger,
)


def _make_full_crew() -> CrewAssignment:
    return CrewAssignment(
        loadout_path="loadouts/mole.json",
        ship_name="Big Mole",
        ship_type="MOLE",
        crew=["Alice", "Bob"],
        _pos={"x": 12.5, "y": -7.0},
        model_crew=4,
        unique_id="abc123",
        mothership_id="ms-1",
        strike_group="Strike Group 1",
        laser_crew={0: ["Alice"], 1: ["Bob"]},
    )


class TestCrewRoundTrip(unittest.TestCase):

    def test_to_dict_preserves_all_fields(self):
        c = _make_full_crew()
        raw = _crew_to_dict(c)
        self.assertEqual(raw["loadout_path"], "loadouts/mole.json")
        self.assertEqual(raw["ship_name"], "Big Mole")
        self.assertEqual(raw["ship_type"], "MOLE")
        self.assertEqual(raw["crew"], ["Alice", "Bob"])
        self.assertEqual(raw["_pos"], {"x": 12.5, "y": -7.0})
        self.assertEqual(raw["model_crew"], 4)
        self.assertEqual(raw["unique_id"], "abc123")
        self.assertEqual(raw["mothership_id"], "ms-1")
        self.assertEqual(raw["strike_group"], "Strike Group 1")
        # JSON keys are strings — even though source dict had int keys
        self.assertEqual(raw["laser_crew"], {"0": ["Alice"], "1": ["Bob"]})

    def test_round_trip_matches(self):
        original = _make_full_crew()
        restored = _crew_from_dict(_crew_to_dict(original))
        self.assertEqual(restored.loadout_path, original.loadout_path)
        self.assertEqual(restored.ship_name, original.ship_name)
        self.assertEqual(restored.ship_type, original.ship_type)
        self.assertEqual(restored.crew, original.crew)
        self.assertEqual(restored._pos, original._pos)
        self.assertEqual(restored.model_crew, original.model_crew)
        self.assertEqual(restored.unique_id, original.unique_id)
        self.assertEqual(restored.mothership_id, original.mothership_id)
        self.assertEqual(restored.strike_group, original.strike_group)
        # laser_crew must come back with int keys (not str)
        self.assertEqual(restored.laser_crew, {0: ["Alice"], 1: ["Bob"]})

    def test_crew_from_dict_supplies_defaults(self):
        # Bare-minimum dict — every other field defaults.
        restored = _crew_from_dict({"ship_name": "Tiny", "ship_type": "Prospector"})
        self.assertEqual(restored.ship_name, "Tiny")
        self.assertEqual(restored.crew, [])
        self.assertEqual(restored.laser_crew, {})
        self.assertEqual(restored.model_crew, 0)

    def test_laser_crew_drops_bad_keys(self):
        raw = {
            "ship_name": "X",
            "ship_type": "MOLE",
            "laser_crew": {
                "0": ["Alice"],
                "not-an-int": ["Bob"],
                "1": "not-a-list",  # value type rejected
                "2": ["Carol", "", None],  # empty / None entries dropped
            },
        }
        restored = _crew_from_dict(raw)
        self.assertEqual(set(restored.laser_crew.keys()), {0, 2})
        self.assertEqual(restored.laser_crew[0], ["Alice"])
        self.assertEqual(restored.laser_crew[2], ["Carol"])


class TestLedgerRoundTrip(unittest.TestCase):

    def test_empty_ledger_round_trip(self):
        original = LedgerData()
        raw = ledger_to_dict(original)
        restored = ledger_from_dict(raw)
        self.assertEqual(restored.foreman_name, "Foreman")
        self.assertEqual(restored.foreman_ships, [])
        self.assertEqual(restored.teams, [])
        self.assertEqual(restored.players, [])
        self.assertEqual(restored.fleet_support_ships, [])
        self.assertEqual(restored.unassigned_ships, [])
        self.assertEqual(restored.assigned_user, "")
        self.assertEqual(restored.strike_groups, [])

    def test_full_ledger_round_trip(self):
        original = LedgerData(
            foreman_name="Captain Reynolds",
            foreman_pos={"x": 10.0, "y": 20.0},
            foreman_ships=[_make_full_crew()],
            teams=[
                MiningTeam(
                    name="Alpha",
                    leader="Alice",
                    ships=[_make_full_crew()],
                    _pos={"x": 5.0, "y": 5.0},
                    parent_leader="",
                    cluster="A",
                ),
            ],
            players=[
                PlayerEntry(
                    name="Alice",
                    is_leader=True,
                    profession="Foreman",
                    can_reassign=True,
                    auto_reassign=False,
                ),
            ],
            fleet_support_ships=[
                FleetSupportShip(
                    name="Caterpillar 1",
                    support_type="Hauling",
                    ship_model="Caterpillar",
                    model_crew=3,
                ),
            ],
            unassigned_ships=[_make_full_crew()],
            assigned_user="Alice",
            strike_groups=[
                StrikeGroupData(
                    name="SG-A",
                    mothership_id="abc123",
                    leader="Alice",
                    _pos={"x": 1.0, "y": 2.0},
                ),
            ],
        )

        # Force JSON serialization by going through dumps + loads
        # to prove the dict shape is JSON-compatible.
        raw = ledger_to_dict(original)
        roundtripped = json.loads(json.dumps(raw))
        restored = ledger_from_dict(roundtripped)

        self.assertEqual(restored.foreman_name, "Captain Reynolds")
        self.assertEqual(restored.foreman_pos, {"x": 10.0, "y": 20.0})
        self.assertEqual(len(restored.foreman_ships), 1)
        self.assertEqual(restored.foreman_ships[0].ship_name, "Big Mole")

        self.assertEqual(len(restored.teams), 1)
        self.assertEqual(restored.teams[0].name, "Alpha")
        self.assertEqual(restored.teams[0].cluster, "A")
        self.assertEqual(len(restored.teams[0].ships), 1)

        self.assertEqual(len(restored.players), 1)
        self.assertEqual(restored.players[0].name, "Alice")
        self.assertTrue(restored.players[0].is_leader)
        self.assertTrue(restored.players[0].can_reassign)

        self.assertEqual(len(restored.fleet_support_ships), 1)
        self.assertEqual(
            restored.fleet_support_ships[0].ship_model, "Caterpillar",
        )

        self.assertEqual(len(restored.unassigned_ships), 1)
        self.assertEqual(restored.assigned_user, "Alice")

        self.assertEqual(len(restored.strike_groups), 1)
        self.assertEqual(restored.strike_groups[0].name, "SG-A")

    def test_from_dict_supplies_defaults_for_missing(self):
        # Empty dict — every list defaults to []
        restored = ledger_from_dict({})
        self.assertEqual(restored.foreman_name, "Foreman")
        self.assertEqual(restored.teams, [])
        self.assertEqual(restored.players, [])


class TestSaveLoadLedger(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "mining_ledger.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load_empty(self):
        save_ledger(LedgerData(), self._path)
        loaded = load_ledger(self._path)
        self.assertIsInstance(loaded, LedgerData)
        self.assertEqual(loaded.foreman_name, "Foreman")

    def test_save_then_load_preserves_fields(self):
        original = LedgerData(
            foreman_name="Skipper",
            assigned_user="Alice",
            players=[PlayerEntry(name="Alice", is_leader=True)],
        )
        save_ledger(original, self._path)
        self.assertTrue(os.path.exists(self._path))

        loaded = load_ledger(self._path)
        self.assertEqual(loaded.foreman_name, "Skipper")
        self.assertEqual(loaded.assigned_user, "Alice")
        self.assertEqual(len(loaded.players), 1)
        self.assertEqual(loaded.players[0].name, "Alice")

    def test_load_missing_file_returns_default(self):
        loaded = load_ledger(self._path)  # file doesn't exist
        self.assertIsInstance(loaded, LedgerData)
        self.assertEqual(loaded.foreman_name, "Foreman")

    def test_load_corrupted_returns_default(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("{garbage")
        loaded = load_ledger(self._path)
        self.assertIsInstance(loaded, LedgerData)
        self.assertEqual(loaded.teams, [])

    def test_load_non_dict_returns_default(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        loaded = load_ledger(self._path)
        self.assertIsInstance(loaded, LedgerData)


class TestRosterImportExport(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = os.path.join(self._tmp.name, "roster.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_export_then_import(self):
        players = [
            PlayerEntry(name="Alice", is_leader=True, profession="Foreman"),
            PlayerEntry(name="Bob", profession="Miner"),
        ]
        export_player_roster(players, self._path)
        restored = import_player_roster(self._path)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].name, "Alice")
        self.assertTrue(restored[0].is_leader)
        self.assertEqual(restored[1].name, "Bob")

    def test_import_drops_blank_names(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([
                {"name": "Real"},
                {"name": ""},  # dropped
                {"profession": "Miner"},  # missing name → dropped
            ], f)
        restored = import_player_roster(self._path)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].name, "Real")

    def test_import_corrupted_returns_empty(self):
        with open(self._path, "w", encoding="utf-8") as f:
            f.write("not json")
        self.assertEqual(import_player_roster(self._path), [])


if __name__ == "__main__":
    unittest.main()
