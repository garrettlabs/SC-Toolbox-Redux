#!/usr/bin/env python3
"""pak_cross_audit.py - confirm slot counts against pak-derived ship data.

slot_truth_audit.py compares slot_extractor against erkul's RENDERED slots.
But erkul applies a hand-maintained corrector layer, so erkul is not a perfect
oracle. This audit adds a third, independent reference: scunpacked-data
(StarCitizenWiki/scunpacked-data), a mechanical extraction of the game paks
with NO corrector layer.

For every ship where the calculator and erkul disagree, it prints a 3-way
line so the pak count can act as a tie-breaker:
  calc   - extract_slots_by_type() on erkul's API loadout (.erkul_cache.json)
  erkul  - erkul's rendered slots (erkul_slot_truth.json)
  paks   - scunpacked ships.json top-level hardpoint count

Only WeaponGun and PowerPlant are cross-checked: scunpacked's Loadout is a
FLAT top-level hardpoint list (a turret counts once, inner guns and missile
rack contents are not enumerated), so its missile count is not comparable to
erkul's expanded display.

Usage:
    python pak_cross_audit.py        # -> pak_cross_audit_report.txt
"""
import json
import os
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..")))
from shared.app_bootstrap import bootstrap_skill  # noqa: E402
bootstrap_skill(__file__)
from services.slot_extractor import extract_slots_by_type  # noqa: E402
from slot_truth_audit import erkul_counts  # noqa: E402

CACHE_FILE = os.path.join(SCRIPT_DIR, ".erkul_cache.json")
TRUTH_FILE = os.path.join(SCRIPT_DIR, "erkul_slot_truth.json")
SCUN_FILE = os.path.join(SCRIPT_DIR, ".scunpacked_ships_cache.json")
REPORT_FILE = os.path.join(SCRIPT_DIR, "pak_cross_audit_report.txt")
SCUN_URL = ("https://raw.githubusercontent.com/StarCitizenWiki/"
            "scunpacked-data/master/ships.json")


def load_scunpacked():
    """scunpacked ships.json, fetched and cached on first use."""
    if not os.path.isfile(SCUN_FILE):
        req = urllib.request.Request(SCUN_URL, headers={"User-Agent": "audit"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(SCUN_FILE, "wb") as f:
            f.write(data)
    with open(SCUN_FILE, encoding="utf-8") as f:
        return json.load(f)


def scun_counts(loadout):
    """Top-level weapon/power hardpoint counts from a scunpacked flat Loadout."""
    w = pp = 0
    for e in loadout or []:
        cat = (e.get("Type") or "").split(".")[0]
        if cat in ("Turret", "WeaponGun", "TurretBase"):
            w += 1
        elif cat == "PowerPlant":
            pp += 1
    return {"WeaponGun": w, "PowerPlant": pp}


def main():
    for path, tool in ((CACHE_FILE, "refresh_erkul_cache.py"),
                       (TRUTH_FILE, "erkul_slot_truth.py")):
        if not os.path.isfile(path):
            print(f"ERROR: {path} not found - run {tool} first.")
            return 1

    with open(CACHE_FILE, encoding="utf-8") as f:
        ecache = json.load(f)["data"]
    erkul_ships = {}
    for e in ecache.get("/live/ships", []):
        x = e.get("data", {})
        if x.get("name"):
            erkul_ships.setdefault(x["name"], x)

    with open(TRUTH_FILE, encoding="utf-8") as f:
        truth = json.load(f).get("ships", {})

    scun_by_uuid = {s.get("UUID"): s for s in load_scunpacked()}

    lines = ["=" * 92,
             "  PAK CROSS-AUDIT - ships where the calculator and erkul disagree",
             "=" * 92,
             "  calc = slot_extractor   erkul = rendered slots   paks = scunpacked (game data)",
             "  A turret counts once in the pak column (inner guns not enumerated).",
             ""]
    calc_right = calc_wrong = ambiguous = no_pak = 0

    for name in sorted(erkul_ships):
        es = erkul_ships[name]
        loadout = es.get("loadout", [])
        calc = {
            "WeaponGun": len(extract_slots_by_type(loadout, {"WeaponGun", "Turret"})),
            "PowerPlant": len(extract_slots_by_type(loadout, {"PowerPlant"})),
        }
        erk = erkul_counts(truth.get(name, {}).get("slots", []))
        scun = scun_by_uuid.get(es.get("ref", ""))
        pak = scun_counts(scun.get("Loadout", [])) if scun else None

        for typ in ("WeaponGun", "PowerPlant"):
            c, e = calc[typ], erk[typ]
            if c == e:
                continue
            p = pak[typ] if pak else None
            if p is None:
                verdict = "no pak match"
                no_pak += 1
            elif c == p:
                verdict = "calc matches game data; erkul diverges"
                calc_right += 1
            elif e == p:
                verdict = "erkul matches game data; calculator diverges"
                calc_wrong += 1
            else:
                verdict = "all three differ"
                ambiguous += 1
            pk = "  --" if p is None else f"{p:4d}"
            lines.append(f"  {name:32s} {typ:14s} "
                         f"calc={c:4d} erkul={e:4d} paks={pk}   {verdict}")

    lines += ["", "=" * 92,
              f"  calc matches game data (erkul diverges) : {calc_right}",
              f"  erkul matches game data (calc diverges) : {calc_wrong}",
              f"  all three differ                        : {ambiguous}",
              f"  no pak match                            : {no_pak}"]
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
