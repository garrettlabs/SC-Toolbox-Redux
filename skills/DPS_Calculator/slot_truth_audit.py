#!/usr/bin/env python3
"""slot_truth_audit.py - audit slot_extractor against erkul's RENDERED slots.

slot_parity_audit.py compares slot_extractor against count_erkul_raw_slots() - a
second heuristic over the raw API loadout tree, not erkul's actual output. So it
measures one guess against another and can never converge.

This audit compares slot_extractor against erkul_slot_truth.json - the slots
erkul's live calculator actually displays, captured by erkul_slot_truth.py.

erkul's display conventions are reconciled before counting:
  - a multi-gun turret shows ONE 'weapons' entry; an empty weapon mount shows an
    'Empty' entry under its gimbal-turret -> weapons + empty mounts
  - a missile rack shows one 'missiles' entry plus 'missile' stacks tagged 'xN'
    -> racks + sum of the xN multipliers

Usage:
    python erkul_slot_truth.py     # 1. capture / refresh the ground truth
    python refresh_erkul_cache.py  # 2. ensure .erkul_cache.json exists
    python slot_truth_audit.py     # 3. run this audit -> slot_truth_report.txt
"""
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..")))
from shared.app_bootstrap import bootstrap_skill  # noqa: E402
bootstrap_skill(__file__)

from services.slot_extractor import extract_slots_by_type  # noqa: E402

CACHE_FILE = os.path.join(SCRIPT_DIR, ".erkul_cache.json")
TRUTH_FILE = os.path.join(SCRIPT_DIR, "erkul_slot_truth.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "slot_truth_report.txt")

# audit type -> slot_extractor accept_types.
# erkul renders BombLauncher bomb racks inside its 'missiles' section, so the
# MissileLauncher row accepts BombLauncher too (the calculator shows bombs in a
# separate tab, but the items are the same).
TYPE_MAP = [
    ("WeaponGun",       {"WeaponGun", "Turret"}),
    ("MissileLauncher", {"MissileLauncher", "BombLauncher"}),
    ("Shield",          {"Shield"}),
    ("Cooler",          {"Cooler"}),
    ("PowerPlant",      {"PowerPlant"}),
    ("QuantumDrive",    {"QuantumDrive"}),
    ("Radar",           {"Radar"}),
]

_XN = re.compile(r"\sx(\d+)$")


def _mult(item):
    """Trailing 'xN' on an erkul item label -> N (default 1)."""
    m = _XN.search(item or "")
    return int(m.group(1)) if m else 1


def load_cache_ships():
    """{ship name -> ship data dict} from the erkul API cache."""
    with open(CACHE_FILE, encoding="utf-8") as f:
        data = json.load(f).get("data", {})
    ships = {}
    for entry in data.get("/live/ships", []):
        d = entry.get("data", {})
        name = d.get("name", "")
        if name and name not in ships:
            ships[name] = d
    return ships


def load_truth():
    with open(TRUTH_FILE, encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("ships", {}), obj.get("captured_at", "?")


# d0 categories that mark the end of the weapon region (missiles + components).
_NON_WEAPON_CATS = {"missiles", "shields", "coolers", "power-plants",
                    "quantum-drives", "radars", "emps"}


def erkul_counts(slots):
    """Reconcile erkul's grouped display into per-type counts comparable to
    slot_extractor's output (see module docstring for the conventions)."""
    # parent category of each slot, via a depth stack (depth -> last category)
    parent = []
    stack = {}
    for s in slots:
        d = s.get("depth", 0)
        parent.append(stack.get(d - 1))
        stack[d] = s.get("category", "")
        for deeper in [k for k in stack if k > d]:
            del stack[deeper]

    # erkul renders an EMPTY weapon hardpoint as a slot with no category icon
    # (category ""), indistinguishable from any other empty slot except by
    # position: weapon slots render before the first missile/component slot.
    # Find that boundary so empty weapon hardpoints can be counted.
    weapon_region_end = len(slots)
    for i, s in enumerate(slots):
        if (s.get("depth", 0) == 0
                and s.get("category", "") in _NON_WEAPON_CATS):
            weapon_region_end = i
            break

    weapons = missiles = missile_xn = 0
    cat = {}
    for i, s in enumerate(slots):
        c = s.get("category", "")
        cat[c] = cat.get(c, 0) + 1
        if c == "weapons":
            weapons += 1
        elif c == "" and parent[i] == "gimbal-turret":
            weapons += 1  # empty weapon mount inside a turret
        elif c == "" and s.get("depth", 0) == 0 and i < weapon_region_end:
            weapons += 1  # empty top-level weapon hardpoint
        elif c == "missiles":
            missiles += 1
        elif c == "missile":
            missile_xn += _mult(s.get("item", ""))
    return {
        "WeaponGun": weapons,
        "MissileLauncher": missiles + missile_xn,
        "Shield": cat.get("shields", 0),
        "Cooler": cat.get("coolers", 0),
        "PowerPlant": cat.get("power-plants", 0),
        "QuantumDrive": cat.get("quantum-drives", 0),
        "Radar": cat.get("radars", 0),
    }


def main():
    if not os.path.isfile(TRUTH_FILE):
        print(f"ERROR: {TRUTH_FILE} not found - run erkul_slot_truth.py first.")
        return 1
    if not os.path.isfile(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found - run refresh_erkul_cache.py first.")
        return 1

    cache_ships = load_cache_ships()
    truth_ships, captured_at = load_truth()

    lines = []
    out = lines.append

    out("=" * 100)
    out("  SLOT TRUTH AUDIT - slot_extractor vs erkul's rendered slots")
    out("=" * 100)
    out(f"erkul_slot_truth.json captured: {captured_at}")
    out(f"ships - cache: {len(cache_ships)}, erkul truth: {len(truth_ships)}")
    out("")

    only_cache = sorted(set(cache_ships) - set(truth_ships))
    only_truth = sorted(set(truth_ships) - set(cache_ships))
    if only_cache:
        out(f"In cache but NOT erkul truth ({len(only_cache)}): {', '.join(only_cache)}")
    if only_truth:
        out(f"In erkul truth but NOT cache ({len(only_truth)}): {', '.join(only_truth)}")
    out("")

    common = sorted(set(cache_ships) & set(truth_ships))
    skipped = []
    issue_rows = {}
    weapon_under = {}  # ship -> (ours, erkul) where slot_extractor finds FEWER weapons
    type_summary = {t[0]: [0, 0] for t in TYPE_MAP}  # name -> [match, mismatch]

    for ship in common:
        tinfo = truth_ships[ship]
        if tinfo.get("error") or tinfo.get("name_mismatch"):
            skipped.append(ship)
            continue
        loadout = cache_ships[ship].get("loadout", [])
        tcounts = erkul_counts(tinfo.get("slots", []))
        ship_issues = []
        for typ, accept in TYPE_MAP:
            ours = len(extract_slots_by_type(loadout, accept))
            theirs = tcounts[typ]
            if ours == theirs:
                type_summary[typ][0] += 1
            else:
                type_summary[typ][1] += 1
                ship_issues.append(
                    f"    {typ:18s}: ours={ours:3d}  erkul={theirs:3d}  delta={ours - theirs:+d}"
                )
                if typ == "WeaponGun" and ours < theirs:
                    weapon_under[ship] = (ours, theirs)
        if ship_issues:
            issue_rows[ship] = ship_issues

    out("=" * 100)
    out("  HOW TO READ THIS")
    out("=" * 100)
    out("  - WeaponGun and MissileLauncher counts are reconciled for erkul's display")
    out("    conventions (turret grouping, empty weapon mounts, xN missile stacks),")
    out("    so deltas in those types are real discrepancies, not counting artifacts.")
    out("  - A NEGATIVE delta (erkul > ours) means erkul renders a hardpoint that")
    out("    slot_extractor failed to find - the highest-value bugs; listed first.")
    out("  - Shield/Cooler/PowerPlant/Radar use raw counts; a small delta there can")
    out("    be an empty component slot that erkul labels generically.")
    out("")

    out("=" * 100)
    out(f"  HIGH-CONFIDENCE: slot_extractor finds FEWER weapons than erkul  ({len(weapon_under)} ships)")
    out("=" * 100)
    out("")
    if not weapon_under:
        out("  none")
    for ship in sorted(weapon_under):
        ours, theirs = weapon_under[ship]
        out(f"  {ship:34s} ours={ours:3d}  erkul={theirs:3d}  -> MISSING {theirs - ours}")
    out("")

    out("=" * 100)
    out(f"  SLOT COUNT MISMATCHES vs erkul truth  ({len(issue_rows)} ships)")
    out("=" * 100)
    out("")
    if not issue_rows:
        out("  ALL AUDITED SHIPS MATCH erkul's rendered slot counts.")
    for ship in sorted(issue_rows):
        out(f"  {ship}:")
        lines.extend(issue_rows[ship])
        out("")

    out("=" * 100)
    out("  PER-TYPE SUMMARY")
    out("=" * 100)
    out(f"  {'Type':18s} {'Match':>7s} {'Mismatch':>10s}")
    out("  " + "-" * 38)
    for typ, _ in TYPE_MAP:
        match, mismatch = type_summary[typ]
        out(f"  {typ:18s} {match:>7d} {mismatch:>10d}")
    out("")

    out("=" * 100)
    out("  SUMMARY")
    out("=" * 100)
    out(f"  ships audited       : {len(common) - len(skipped)}")
    out(f"  ships with mismatch : {len(issue_rows)}")
    out(f"  skipped (scrape error / name mismatch): {len(skipped)}")
    if skipped:
        out(f"    {', '.join(skipped)}")

    report = "\n".join(lines)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"))
    print(f"\nReport saved to: {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
