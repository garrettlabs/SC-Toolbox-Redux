# SC Toolbox v2.2.13

## Mining Signals — Major HUD lock-rate fix

Validated on 241 captured panels: **84% effective lock rate -> 96.2%** on real mining HUDs (+27 percentage points). The scanner now locks reliably on panels that previously fell back to "no panel found" log spam.

The fix is a stack of six independent improvements to the label-matcher and tracker:

- **LSQ outlier rejection** -- when the rigid-body solver fails because one anchor is geometrically inconsistent with the rest, drop the worst-residual anchor and re-solve. Recovers the most common failure pattern (a stale SCAN RESULTS title elsewhere on screen poisoning the fit).
- **Dual-polarity MASS NCC** -- the old Otsu-driven polarity heuristic mis-flipped some panels, dropping the MASS NCC from 0.92 at the correct position to 0.59 at a spurious one. We now run NCC against both polarities and take whichever scores higher.
- **RGB voting** -- the SC HUD text is mint-cyan; previously the matcher operated only in grayscale, so spurious matches on same-luminance white text (UNKNOWN footer, mineral names) sometimes outscored the actual MASS row. RGB and grayscale now vote -- when both detectors agree on a position within 15 pixels, that's high confidence; when they disagree, we look for an alternative.
- **RESIST / INSTA truncated templates** -- when the full RESISTANCE: / INSTABILITY: words fail NCC (perspective skew, partial occlusion), shorter "RESIST" / "INSTA" templates sliced from the existing assets serve as fallback row anchors.
- **Colon-anchor fallback** -- when one of the three label rows fails entirely, an independent colon-glyph NCC search locates the row by its trailing ":" instead. Now also fires when only SCAN RESULTS matched, using it as the column anchor.
- **MASS y-position sanity check** -- reject MASS detections that fall implausibly far below SCAN RESULTS (where no real HUD geometry could put them).

Also: a panel-presence pre-filter now silently skips non-HUD frames (transitions, menus, the inventory screen) instead of error-spamming the logs every frame.

## Mining Signals — Signature scanner "scanning forever" fix

The signature scanner had a UI bug where the "Scanning..." placeholder bubble would never dismiss, making the scanner appear stuck even when OCR ran fine downstream. The worker thread tried to clear the bubble via `QMetaObject.invokeMethod` but the target method was missing the `@Slot()` decorator that PyQt needs to expose it as cross-thread invokable. Qt logged 300+ "No such method" warnings per session in the failing user's log; the user saw a perpetual "Scanning..." state and assumed no signature results were being produced.

One-line fix: added the missing decorator. Signature scans now complete and dismiss the placeholder properly.

## Trade Hub — Starting investment filter

New **STARTING INVESTMENT (aUEC)** field on the trade hub sidebar. Set your available budget (e.g. `2000000` or `2,000,000`) and the route list hides any route whose first-leg buy cost would exceed it. Subsequent legs in a loop or chain are paid for with the proceeds from earlier sales, so only the starting outlay matters for the check.

Applies to ROUTES, LOOPS, MIXED ROUTES, and MIXED LOOPS views.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
