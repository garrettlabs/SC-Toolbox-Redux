#!/usr/bin/env python3
"""erkul_slot_truth.py - capture erkul.games ground-truth slot structure per ship.

Drives erkul's live calculator with Playwright: opens the in-app ship picker,
selects every ship, and records the rendered slot list (size / category /
equipped item / locked) exactly as erkul displays it.

Output: erkul_slot_truth.json - the ground-truth reference that
slot_parity_audit.py diffs slot_extractor against. Unlike count_erkul_raw_slots()
(a heuristic over the raw API tree), this is what erkul's UI actually shows.

erkul loads every ship + component once on page load and selects ships
client-side, so a full run touches erkul's server only for that single load.

Setup:  pip install playwright  &&  python -m playwright install chromium
Usage:  python erkul_slot_truth.py [--limit N] [--headed]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

URL = "https://www.erkul.games/live/calculator"
OUT_FILE = Path(__file__).with_name("erkul_slot_truth.json")

# Walk every <app-item> (erkul's slot component). depth = number of app-item
# ancestors, which preserves the turret -> inner-gun nesting. category comes
# from the type icon's filename (weapons.png, shields.png, turrets.png, ...).
EXTRACT_JS = r"""() => {
  const catOf = (src) => {
    const m = (src || '').match(/icons\/([a-z0-9-]+)\.png/i);
    return m ? m[1] : '';
  };
  const out = [];
  for (const ai of document.querySelectorAll('app-item')) {
    let depth = 0, p = ai.parentElement;
    while (p) { if (p.tagName === 'APP-ITEM') depth++; p = p.parentElement; }
    let btn = null;
    for (const b of ai.querySelectorAll('button.item-button')) {
      if (b.closest('app-item') === ai) { btn = b; break; }
    }
    if (!btn) continue;
    const sizeEl = btn.querySelector('.size');
    const img = btn.querySelector('app-item-icons img');
    let item = '';
    const titleEl = btn.querySelector('.mat-title');
    if (titleEl) {
      const c = titleEl.cloneNode(true);
      c.querySelectorAll('mat-icon').forEach((x) => x.remove());
      item = c.textContent.replace(/\s+/g, ' ').trim();
    }
    out.push({
      depth: depth,
      size: sizeEl ? sizeEl.textContent.trim() : '',
      category: img ? catOf(img.getAttribute('src')) : '',
      item: item,
      locked: btn.disabled === true || btn.classList.contains('mat-button-disabled'),
    });
  }
  return out;
}"""


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def dismiss_modals(page, rounds: int = 6) -> None:
    """erkul shows giveaway + donation modals on load; clear all of them."""
    for _ in range(rounds):
        if not page.locator("mat-dialog-container, .mat-dialog-container").count():
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        for pat in ("close", "no thanks", "skip", "dismiss"):
            loc = page.get_by_role("button", name=re.compile(pat, re.I))
            if loc.count():
                try:
                    loc.first.click(timeout=1800)
                except PWError:
                    pass
        page.wait_for_timeout(900)


def close_menus(page, rounds: int = 6) -> None:
    """Close any open mat-menu/overlay (Escape closes one level at a time)."""
    for _ in range(rounds):
        if not page.locator(".cdk-overlay-backdrop").count():
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    bd = page.locator(".cdk-overlay-backdrop")
    if bd.count():
        try:
            bd.last.click(force=True, timeout=1500)
        except PWError:
            pass
    page.wait_for_timeout(300)


def open_picker(page) -> None:
    """Open the manufacturer/ship picker menu."""
    close_menus(page)
    btn = page.get_by_role("button", name=re.compile("select ship", re.I))
    if btn.count():
        btn.first.click(timeout=8000)
    else:
        # once a ship is loaded the trigger may be the ship card itself
        page.locator(".ship-name, .ship-container").first.click(timeout=8000)
    page.wait_for_timeout(900)
    if not page.get_by_role("menuitem").count():
        raise RuntimeError("ship picker menu did not open")


def enumerate_ships(page) -> list[tuple[str, str]]:
    """Walk manufacturer -> ship. Returns (manufacturer, ship) pairs."""
    open_picker(page)
    top = [t.strip() for t in page.get_by_role("menuitem").all_inner_texts() if t.strip()]
    pairs: list[tuple[str, str]] = []
    for mfr in top:
        try:
            page.get_by_role("menuitem", name=mfr, exact=True).first.hover(timeout=4000)
        except PWError:
            continue
        page.wait_for_timeout(1000)
        now = [t.strip() for t in page.get_by_role("menuitem").all_inner_texts() if t.strip()]
        ships = [s for s in now if s not in top]
        if ships:
            pairs.extend((mfr, s) for s in ships)
        else:
            pairs.append((mfr, mfr))  # top-level leaf is itself a ship
    close_menus(page)
    return pairs


def select_ship(page, manufacturer: str, ship: str) -> None:
    open_picker(page)
    if manufacturer != ship:
        page.get_by_role("menuitem", name=manufacturer, exact=True).first.hover(timeout=5000)
        page.wait_for_timeout(1000)
    page.get_by_role("menuitem", name=ship, exact=True).first.click(timeout=6000)


def wait_for_ship(page, ship: str, timeout_ms: int = 15000) -> bool:
    """Poll until the ship-name header equals the requested ship."""
    deadline = time.time() + timeout_ms / 1000
    want = _norm(ship)
    while time.time() < deadline:
        nm = page.locator(".ship-name")
        if nm.count():
            try:
                if _norm(nm.first.inner_text()) == want:
                    return True
            except PWError:
                pass
        page.wait_for_timeout(300)
    return False


def main() -> int:
    try:  # erkul ship names include non-cp1252 chars (accented names)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Capture erkul slot ground truth.")
    ap.add_argument("--limit", type=int, default=0, help="only first N ships (testing)")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--resume", action="store_true",
                    help="skip ships already captured in erkul_slot_truth.json")
    args = ap.parse_args()

    result: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": URL,
        "ships": {},
    }
    errors: list[dict] = []

    done: set[str] = set()
    if args.resume and OUT_FILE.exists() and not args.limit:
        try:
            prev = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            result["ships"] = prev.get("ships", {})
            done = {s for s, v in result["ships"].items()
                    if v.get("slots") and not v.get("error")}
            if done:
                log(f"resuming: {len(done)} ships already captured")
        except Exception:
            result["ships"] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        log(f"loading {URL} ...")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=60000)
        except PWError:
            pass
        page.wait_for_timeout(3000)
        dismiss_modals(page)

        pairs = enumerate_ships(page)
        mfr_count = len({m for m, _ in pairs})
        log(f"erkul lists {len(pairs)} ships across {mfr_count} manufacturers")
        if args.limit:
            pairs = pairs[: args.limit]

        for idx, (mfr, ship) in enumerate(pairs, 1):
            tag = f"[{idx}/{len(pairs)}] {ship} ({mfr})"
            if ship in done:
                continue
            try:
                select_ship(page, mfr, ship)
                matched = wait_for_ship(page, ship)
                page.wait_for_timeout(700)  # let slots settle
                erkul_name = ""
                nm = page.locator(".ship-name")
                if nm.count():
                    erkul_name = " ".join(nm.first.inner_text().split())
                slots = page.evaluate(EXTRACT_JS)
                entry = {"manufacturer": mfr, "erkul_name": erkul_name, "slots": slots}
                if not matched:
                    entry["name_mismatch"] = True
                    errors.append({"ship": ship, "error": f"name mismatch (got {erkul_name!r})"})
                result["ships"][ship] = entry
                flag = "  !name-mismatch" if not matched else ""
                log(f"{tag}: {len(slots)} slots{flag}")
            except Exception as ex:  # noqa: BLE001 - one ship must not kill the run
                msg = " ".join(str(ex).split())[:300]
                errors.append({"ship": ship, "error": msg})
                result["ships"][ship] = {"manufacturer": mfr, "slots": [], "error": msg}
                log(f"{tag}: ERROR {msg}")
            if idx % 20 == 0:
                OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
                log(f"  ... checkpoint saved ({idx} ships)")

        browser.close()

    result["ship_count"] = len(result["ships"])
    result["error_count"] = len(errors)
    OUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"\ndone: {result['ship_count']} ships -> {OUT_FILE.name} ({len(errors)} errors)")
    if errors:
        for e in errors[:15]:
            log(f"  - {e['ship']}: {e['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
