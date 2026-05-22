#!/usr/bin/env python3
"""erkul_dropdown_truth.py - capture every erkul slot's item-picker contents.

For each ship, opens every editable slot's item picker (the overlay table erkul
shows when a slot is clicked) and records the list of selectable items. This is
the ground truth for "do the calculator's dropdowns match erkul" — the items a
player can equip in each slot.

Output: erkul_dropdown_truth.json

Usage:  python erkul_dropdown_truth.py [--limit N] [--headed] [--resume]
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
OUT_FILE = Path(__file__).with_name("erkul_dropdown_truth.json")

# Per-slot metadata for every app-item (matches erkul_slot_truth.py's fields).
SLOTS_JS = r"""() => {
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
    const disabled = btn.disabled === true ||
                     btn.classList.contains('mat-button-disabled');
    // tag THIS slot's own button so the click and the metadata can never
    // drift apart (nested turret app-items otherwise offset a flat index).
    btn.setAttribute('data-scrape', String(out.length));
    out.push({
      depth: depth,
      size: sizeEl ? sizeEl.textContent.trim() : '',
      category: img ? catOf(img.getAttribute('src')) : '',
      item: item,
      disabled: disabled,
    });
  }
  return out;
}"""

# Read the open item-picker overlay table -> list of selectable item names,
# taken from the 'Name' column (erkul also shows a weapon-class column).
PICKER_JS = r"""() => {
  const t = document.querySelector('.cdk-overlay-pane app-overlay-mount table')
         || document.querySelector('.cdk-overlay-pane table');
  if (!t) return null;
  const hdr = [...t.querySelectorAll('thead th')]
                .map((e) => e.textContent.replace(/\s+/g, ' ').trim());
  let nameIdx = hdr.indexOf('Name');
  if (nameIdx < 0) nameIdx = 2;
  const opts = [];
  for (const tr of t.querySelectorAll('tbody tr')) {
    const tds = [...tr.querySelectorAll('td')];
    if (tds.length <= nameIdx) continue;
    const name = tds[nameIdx].textContent.replace(/\s+/g, ' ').trim();
    if (name) opts.push(name);
  }
  return opts;
}"""


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def dismiss_modals(page, rounds: int = 6) -> None:
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
    for _ in range(rounds):
        if not page.locator(".cdk-overlay-backdrop").count():
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)


def open_picker(page) -> None:
    close_menus(page)
    btn = page.get_by_role("button", name=re.compile("select ship", re.I))
    if btn.count():
        btn.first.click(timeout=8000)
    else:
        page.locator(".ship-name, .ship-container").first.click(timeout=8000)
    page.wait_for_timeout(900)
    if not page.get_by_role("menuitem").count():
        raise RuntimeError("ship picker did not open")


def enumerate_ships(page) -> list[tuple[str, str]]:
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
        pairs.extend((mfr, s) for s in ships) if ships else pairs.append((mfr, mfr))
    close_menus(page)
    return pairs


def select_ship(page, mfr: str, ship: str) -> None:
    open_picker(page)
    if mfr != ship:
        page.get_by_role("menuitem", name=mfr, exact=True).first.hover(timeout=5000)
        page.wait_for_timeout(1000)
    page.get_by_role("menuitem", name=ship, exact=True).first.click(timeout=6000)


def wait_for_ship(page, ship: str, timeout_ms: int = 15000) -> bool:
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


_PICKER = ".cdk-overlay-pane:has(table)"


def _picker_open(page) -> bool:
    return page.locator(_PICKER).count() > 0


def _close_overlay(page) -> bool:
    """Close any open picker overlay (one with a table). True once none remain."""
    for _ in range(12):
        if not _picker_open(page):
            return True
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        bd = page.locator(".cdk-overlay-backdrop")
        if bd.count():
            try:
                bd.last.click(force=True, timeout=800)
            except PWError:
                pass
        page.wait_for_timeout(150)
    return not _picker_open(page)


def capture_dropdowns(page) -> list[dict]:
    """For the loaded ship, open every editable slot picker and read it.

    Each slot is isolated: every overlay is closed BEFORE the next click, and
    after clicking we wait for the picker table to actually appear - so a slot
    can never read a stale neighbouring overlay.
    """
    slots = page.evaluate(SLOTS_JS)            # also tags each slot's button
    result = []
    for idx, meta in enumerate(slots):
        if meta.get("disabled"):
            continue
        opts = None
        for _attempt in range(3):              # a click can fail to open the picker
            _close_overlay(page)
            page.wait_for_timeout(160)         # let any close animation finish
            btn = page.locator(f'button[data-scrape="{idx}"]')
            if not btn.count():
                break
            try:
                btn.first.scroll_into_view_if_needed(timeout=3000)
                btn.first.click(timeout=5000)
            except PWError:
                continue
            for _ in range(30):                # wait for the picker to open
                page.wait_for_timeout(120)
                if _picker_open(page):
                    page.wait_for_timeout(220)  # let the table settle
                    opts = page.evaluate(PICKER_JS)
                    break
            if opts is not None:
                break
        _close_overlay(page)
        result.append({
            "index": idx,
            "size": meta.get("size", ""),
            "category": meta.get("category", ""),
            "depth": meta.get("depth", 0),
            "equipped": meta.get("item", ""),
            "options": opts or [],
        })
    return result


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Capture erkul slot dropdowns.")
    ap.add_argument("--limit", type=int, default=0, help="only first N ships")
    ap.add_argument("--only", default="", help="only ships whose name contains this")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    result: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": URL, "ships": {},
    }
    done: set[str] = set()
    if args.resume and OUT_FILE.exists() and not args.limit:
        try:
            prev = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            result["ships"] = prev.get("ships", {})
            done = {s for s, v in result["ships"].items() if v.get("slots")}
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
        log(f"erkul lists {len(pairs)} ships")
        if args.only:
            pairs = [p for p in pairs if args.only.lower() in p[1].lower()]
        if args.limit:
            pairs = pairs[: args.limit]

        for idx, (mfr, ship) in enumerate(pairs, 1):
            if ship in done:
                continue
            tag = f"[{idx}/{len(pairs)}] {ship}"
            try:
                select_ship(page, mfr, ship)
                wait_for_ship(page, ship)
                page.wait_for_timeout(700)
                slots = capture_dropdowns(page)
                result["ships"][ship] = {"manufacturer": mfr, "slots": slots}
                log(f"{tag}: {len(slots)} dropdowns")
            except Exception as ex:  # noqa: BLE001
                msg = " ".join(str(ex).split())[:200]
                result["ships"][ship] = {"manufacturer": mfr, "slots": [], "error": msg}
                log(f"{tag}: ERROR {msg}")
            if idx % 10 == 0:
                OUT_FILE.write_text(json.dumps(result, indent=1), encoding="utf-8")
                log(f"  ... checkpoint ({idx} ships)")

        browser.close()

    result["ship_count"] = len(result["ships"])
    OUT_FILE.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log(f"\ndone: {result['ship_count']} ships -> {OUT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
