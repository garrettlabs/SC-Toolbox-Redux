# Detector Inventory: Existing Anchors / Feature Extractors for HUD Tracker

This document inventories all detectors and feature extractors found in
`ocr/sc_ocr/` and `ocr/onnx_hud_reader.py` with respect to their reusability
as anchors in a multi-anchor HUD tracker.

For the purposes of this audit, an "anchor" is a function that:
1. Takes the captured panel/region image as input.
2. Returns a position/bbox + confidence score (or `None` when the target is
   not visible).
3. Fails gracefully on absence — no exception, no false-positive lock-in.
4. Can be one of several voting features used to localize the panel via
   known geometric proportions (see `hud_tracker/world_model.json`).

---

## 1. Existing detectors

### 1.1 `ocr/sc_ocr/signal_anchor.py::find_icon`

- **Detects:** the location-pin icon (signal scanner) via multi-scale NCC
  against blacklist PNGs in `training_data_blacklist/`. Templates pre-built
  at 13 widths from 12 px to 72 px.
- **Input:** grayscale `np.ndarray` (the signal-region capture); polarity
  is canonicalized internally via `label_match._canonicalize` so either
  bright-on-dark or dark-on-bright works. Search restricted to the leftmost
  `search_left_fraction=0.55` columns and a vertical band `[10%, 90%)` of
  the image height.
- **Output:** `Optional[(x_left, y_top, x_right, y_bot, score)]`. Score is
  zero-mean unit-variance NCC. Threshold default `min_score=0.55`. Includes
  temporal smoothing via `_RECENT_ANCHOR_CACHE` (8-entry deque, 5-second
  TTL, ≥3-entry stability rule, 25-px reset tolerance).
- **Active:** Yes. Live in the signature-scanner pipeline. Called from
  `find_digit_crop_box`.
- **Known limitations (from comments):**
  - Multi-character digit clusters like `"10,"`, `"16,"`, `"11"` score
    comparably to the icon at small template scales because the
    brightness-weighted `correlate2d/area` formula favours fill area; the
    code mitigates this with a CNN re-rank (`_cnn_filter_icon_candidates`)
    that demands the candidate's argmax class be `"@"`, plus a
    leftmost-only post-filter when several candidates pass.
  - Connected-component shape filter rejects 2+ component candidates.
  - Original SC HUD font's `9` and `,0` glyphs have circle-on-stick shapes
    that NCC scores nearly as high as the icon — handled by the CNN
    validator.
  - Cache median fallback used when current-frame score < 0.60.

### 1.2 `ocr/sc_ocr/signal_anchor.py::find_digit_cluster`

- **Detects:** the 4–7-glyph digit cluster of the signal value via
  structural pattern (row projection → column segmentation →
  digit-typical bbox filter → leftmost cluster).
- **Input:** grayscale `np.ndarray`. Calls `_canonicalize_polarity` and
  `_adaptive_binarize` from `api.py`.
- **Output:** `Optional[(x1, y1, x2, y2)]` — no confidence score returned.
- **Active:** Yes. Called from `find_digit_crop_box` as a complementary
  anchor to `find_icon`.
- **Known limitations:** No score returned, so it can't directly feed a
  weighted-vote anchor system without an adapter. Requires height in
  range `[8, 50]` and 4–7 digit-shaped column spans clustered together.

### 1.3 `ocr/sc_ocr/signal_anchor.py::find_digit_crop_box`

- **Detects:** combo wrapper that runs both `find_icon` and
  `find_digit_cluster` and cross-validates them.
- **Input:** grayscale `np.ndarray`.
- **Output:** `Optional[(x1, y1, x2, y2)]` — bbox of the digit cluster
  with vertical/horizontal padding refinements; no exposed score.
- **Active:** Yes.
- **Limitations:** Designed for the signal-scanner panel only, not the
  scan-results panel. Could be repurposed but the proportional offsets
  in `world_model.json` are anchored to scan_results, not signal.

### 1.4 `ocr/sc_ocr/scan_results_match.py::find_scan_results_anchor`

- **Detects:** the "SCAN RESULTS" title text via NCC against the template
  pack at `ocr/sc_templates/scan_results.npz` (multi-scale, 8 scale
  factors from 0.6× to 2.0×).
- **Input:** PIL `Image.Image`. Search restricted to top
  `_SEARCH_TOP_FRACTION=0.42` of the image. Detection channel is
  `rgb.max(axis=2)` (max-of-RGB) rather than luma to preserve
  chromatic-aberrated stroke shape.
- **Output:** `Optional[dict]` of shape
  `{"title_x": int, "title_y": int, "title_w": int, "title_h": int, "score": float}`
  or `None`. Threshold `_MIN_MATCH_SCORE=0.40`. Top-bias of 0.30 favours
  near-y=0 candidates.
- **Active:** Yes. Used as one tier in the legacy
  `_label_rows_from_anchor` chain.
- **Limitations:** The 0.40 threshold catches 99.2% of the calibration
  set; 5% miss at 0.60. Misses occur on heavy particle/snow occlusion
  of the title. Per-image cache keyed on `(id(img), img.size, img.mode)`
  prevents redundant sweeps.

### 1.5 `ocr/sc_ocr/label_match.py::find_label_positions`

- **Detects:** the three label words `MASS:`, `RESISTANCE:`,
  `INSTABILITY:` via NCC against `ocr/sc_templates/labels.npz`.
- **Input:** PIL `Image.Image`. Search restricted to left 65% of image
  width (the value column lives in the right half). Polarity-canonicalized
  via `_canonicalize`.
- **Output:** `dict[str, dict]` mapping label name → match info
  `{"x", "y", "w", "h", "score", "scale"}`. Missing labels are absent
  from the dict. Empty dict on a complete miss.
- **Algorithm:** MASS-first anchoring. Find best MASS match at any of
  8 scales, then at the SAME scale + within ±35% of pitch, search for
  RESISTANCE and INSTABILITY in narrow Y windows (predicted at MASS_y +
  pitch and MASS_y + 2×pitch). FFT-based correlation via
  `scipy.signal.fftconvolve` gives 70× speed-up over naive
  `correlate2d`.
- **Active:** Yes. Surfaced through `_find_label_rows_by_ncc` in
  `onnx_hud_reader.py` as the preferred label-row finder.
- **Limitations:** MASS template required. Threshold `_MIN_MATCH_SCORE=0.45`
  — below that, label is not reported.

### 1.6 `ocr/onnx_hud_reader.py::_find_panel_lines`

- **Detects:** thin horizontal HUD chrome separator lines (the green
  tick-lines bounding the SCAN RESULTS data area).
- **Input:** grayscale `np.ndarray`.
- **Output:** `list[(y_center, x_left, x_right)]` sorted top-to-bottom.
  Returns `[]` on miss.
- **Algorithm:** Build polarity-aware text mask via `_build_text_mask`,
  count row densities, find consecutive runs of ≥`min_width_frac=0.18`
  columns. Each run kept if thickness 1–3 px, span ≥18% of width,
  and ≥80% fill within span (rejects text rows where letter caps span
  wide but only ~50–70% of columns are lit).
- **Active:** Yes. Exposed via `_get_panel_lines_cached`. Used by
  `_find_label_rows_by_hud_grid` and `_find_label_rows_by_position`.
- **Limitations:** Returns no per-line confidence; downstream code picks
  "best pair" by spacing heuristics. Works on either polarity. No
  end-notch detection (just span + density).

### 1.7 `ocr/onnx_hud_reader.py::_find_mineral_row` / `api.py::_find_mineral_row_universal`

- **Detects:** the mineral-name row (e.g. `TORITE (ORE)`,
  `ALUMINUM (ORE)`) by row-projection peaks of the text mask.
- **Input:** PIL `Image.Image` (legacy) or `Image.Image` (universal). The
  universal version uses local-contrast (Gaussian high-pass) for light
  backgrounds and brightness-threshold for dark backgrounds.
- **Output:** `Optional[(y1, y2)]` — y-band only, no x extent or
  confidence score.
- **Active:** Yes. Both versions live and are called from `api.py`.
- **Limitations:** No bbox (caller must supply x-range), no confidence
  score, doesn't validate which mineral was matched (just locates the
  ordinal row after SCAN RESULTS).

### 1.8 `ocr/sc_ocr/api.py::_ocr_mineral_name`

- **Detects:** reads the mineral-name STRING via Tesseract (multi-pass)
  and fuzzy-matches to the known minerals list.
- **Input:** PIL `Image.Image` plus `(y1, y2, x_min)` of the row.
- **Output:** `Optional[str]` — canonical mineral name only. No bbox
  or score.
- **Active:** Yes. Used inside `scan_hud_onnx`.
- **Limitations:** This is a value extractor, NOT an anchor. Doesn't
  return a position.

### 1.9 `ocr/onnx_hud_reader.py::_find_scan_results_anchor` (Tesseract)

- **Detects:** "SCAN RESULTS" title via Tesseract OCR with PSM 11 +
  uppercase whitelist over both polarity variants.
- **Input:** PIL `Image.Image`.
- **Output:** `Optional[dict]` `{"title_x", "title_y", "title_h", "title_w"}`.
- **Active:** Effectively superseded by `scan_results_match` (NCC
  version) which is the preferred entry point in
  `_label_rows_from_anchor`. Tesseract anchor is still wired as a
  legacy fallback; expensive (50–200 ms) and gated by 8-second timeout.
- **Limitations:** Slow, prone to bbox inflation on tilted / chromatically
  aberrated text. Already superseded.

### 1.10 Supporting helpers — preprocessing, segmentation, matching

These are internals the anchors above stand on; they are not anchors
themselves but are the building blocks for any new anchors:

- `ocr/sc_ocr/preprocess.py::isolate_channel`, `binarize`,
  `denoise_if_needed`, `preprocess_rgb` — colour-aware text isolation
  with auto-mode (5 background regimes including magenta-on-warm,
  cyan-on-dark, orange-on-dark).
- `ocr/sc_ocr/preprocess.py::otsu_threshold` and `_build_text_mask` —
  polarity-aware text mask.
- `ocr/sc_ocr/label_match.py::_canonicalize`, `_ncc_search`,
  `_resize_template` — generic NCC primitives (scipy fftconvolve
  fallback to numpy stride trick).
- `ocr/sc_ocr/segment.py::find_rows`, `split_glyphs_in_row` — horizontal
  band finder + connected-component glyph splitter (used by the
  classifier, not the anchors).
- `ocr/sc_ocr/api.py::_canonicalize_polarity`, `_adaptive_binarize` —
  Otsu-minority polarity rule + locally adaptive binarization.
- `ocr/sc_ocr/api.py::_crop_fingerprint` and `_crop_buffer_consistent` —
  downsampled NCC fingerprint for crop stability checks; reusable for
  anchor temporal smoothing.

---

## 2. Reusability assessment

| Detector | Anchor-shaped today? | Adapter needed |
|---|---|---|
| `find_icon` | Yes | None — already returns bbox + score, fails gracefully |
| `find_digit_cluster` | Almost | Tiny adapter to attach a confidence (e.g. cluster glyph count / 7) |
| `find_digit_crop_box` | Almost | Same — exposes bbox but not score |
| `find_scan_results_anchor` | Yes | None — dict output already has `score` |
| `find_label_positions` (per-row) | Yes | Already returns `{x,y,w,h,score}` per matched label; treat each label as its OWN anchor |
| `_find_panel_lines` | Partial | Returns list with no confidence; need adapter that picks the line pair, scores by `(span_frac, fill_ratio, thickness_inverse)` and emits a synthetic `(top_line_box, bot_line_box, score)` |
| `_find_mineral_row` / `_find_mineral_row_universal` | Partial | Returns y-band only; adapter must add x-range (full image width) and synthesize a confidence from peak count |
| `_ocr_mineral_name` | No | String-only extractor; not an anchor — needs to be combined with the mineral-row finder to yield bbox+confidence |
| `_find_scan_results_anchor` (Tesseract) | Yes (slow) | Already returns dict with bbox; conf must be synthesized from word-bounding scores |

Conclusion of section 2:

- **Already anchor-shaped (returns bbox + confidence + graceful absence):** 4
  - `find_icon`, `find_scan_results_anchor` (NCC), each individual label
    in `find_label_positions` (counts as 3 anchors: MASS, RESISTANCE,
    INSTABILITY), and `_find_scan_results_anchor` (Tesseract).
- **Needs trivial adapter (existing detection logic, missing score or
  cleanup):** 4
  - `find_digit_cluster`, `find_digit_crop_box`,
    `_find_panel_lines` (split into top_line and bot_line entries),
    `_find_mineral_row` / `_find_mineral_row_universal`.
- **Internal helpers — not anchors but reusable primitives:** ~7 modules
  (preprocess, label_match NCC primitives, segment, canon/binarize,
  crop fingerprints).

Counting the label_match output as 3 distinct anchors (MASS,
RESISTANCE, INSTABILITY all have independent (x,y,w,h,score)),
**total existing anchors usable today = 6** (find_icon, find_scan_results
NCC, MASS, RESISTANCE, INSTABILITY, Tesseract scan_results) and
**4 more become usable with an adapter wrapper** (digit_cluster,
digit_crop_box, panel_lines, mineral_row).

---

## 3. Gaps — anchors NOT implemented

The following anchor types listed in the task brief are absent from the
codebase. For each, an effort estimate.

### 3.1 Chrome-line detection (top_line, bot_line individually)

- **Status:** `_find_panel_lines` returns ALL detected lines as a list,
  not a labelled top/bot pair. There's no "find_top_line" or
  "find_bot_line" function.
- **End-notch detection (the small vertical hooks at line ends):** NOT
  implemented. `_find_panel_lines` records only `(y_center, x_left, x_right)`
  — span-only, no notch geometry. Notches are a structurally distinctive
  feature for sub-pixel localization that the codebase ignores.
- **Effort:** **Trivial (~50 LOC) for splitting the list into
  top_line/bot_line by spacing heuristics + scoring; moderate
  (~150–200 LOC) if end-notch refinement is wanted.**

### 3.2 HSV / Lab colour-segmentation panel-blob detection

- **Status:** Absent. `preprocess.isolate_channel` does per-channel
  selection (R/G/B/max) but never converts to HSV or Lab. No
  connected-component panel-blob detector exists.
- **Effort:** **Moderate (~200 LOC).** Needs HSV conversion (numpy +
  PIL — no opencv currently used outside a comment in `api.py`),
  hue/saturation thresholds for the SC HUD-green chrome colour, scipy
  `ndimage.label` for components (already a soft dependency elsewhere
  in the codebase), and geometric validation rules (aspect ratio,
  minimum area, expected position).

### 3.3 COMPOSITION green bar detection

- **Status:** Absent. The string "composition" appears in comments
  only. No bar detector exists.
- **Effort:** **Trivial (~50 LOC)** if we go HSV-mask + horizontal
  projection + bar geometry validation (assumes 3.2's HSV primitives
  exist, otherwise add ~50 LOC for HSV).

### 3.4 Outcome-bar detection (EASY / MEDIUM / HARD bar)

- **Status:** No localizer for the bar's bbox. Difficulty word reading
  exists (`scan_hud_onnx` reads OCR text via 4 Tesseract variants over
  a calibrated `needle` rectangle, with cache, in `api.py` ~lines
  7280–7415). The bbox-finding step is delegated to a manually
  calibrated rectangle, NOT a detector.
- **Effort:** **Moderate (~200 LOC)** for an automatic detector — needs
  HSV mask + connected component + position prior. The text-reading
  side can be left as-is (string output is downstream of bbox
  detection).

### 3.5 Signal-pill rounded-rectangle outline detection

- **Status:** Absent. Grep finds zero occurrences of `pill` or
  `rounded_rect` in detector code. The signal panel is currently
  located by `find_icon` only (the pin-icon NCC anchor); the pill
  outline that contains the icon + digits is never detected as a
  geometric shape.
- **Effort:** **Trivial-to-moderate (~100 LOC).** Edge map + Hough-style
  parallel-line pair detection or, more cheaply, colour-mask + contour
  fitting. Would benefit from 3.2's HSV primitives.

### 3.6 Label-row NCC for MASS / RESISTANCE / INSTABILITY

- **Status:** **Already implemented** as `find_label_positions` in
  `ocr/sc_ocr/label_match.py` (see 1.5 above). NOT a gap.

### 3.7 Mineral-name (`ALUMINUM (ORE)` / `IRON (ORE)`) NCC anchor

- **Status:** Reading the mineral name as a string is implemented; using
  it as a positional anchor is not. `_find_mineral_row` returns a
  y-band only; there is no NCC against per-mineral templates.
- **Effort:** **Trivial-to-moderate (~50–150 LOC).** Need a template
  pack (one per known mineral, similar to `labels.npz`) plus the same
  NCC machinery `label_match` already uses. Pack-build script would
  mirror `scripts/build_label_templates.py` (referenced in
  `label_match.py` docstring).

---

## Summary of gap counts

| Gap | Effort |
|---|---|
| Chrome-line top/bot split + scoring | Trivial (~50) |
| Chrome-line end-notch geometry refinement | Moderate (~200) |
| HSV/Lab panel-blob | Moderate (~200) |
| COMPOSITION bar | Trivial (~50, on top of HSV) |
| Outcome bar bbox | Moderate (~200) |
| Signal-pill rounded-rect | Trivial-to-moderate (~100) |
| Mineral-name NCC anchor | Trivial-to-moderate (~50–150) |

Of those, the **highest-leverage missing anchor** is the chrome-line
top/bot split (item 3.1), because:

1. The top_line and bot_line are the two FATTEST pieces of fixed HUD
   geometry — they bound the entire SCAN RESULTS data area
   (see `world_model.json` where `top_line` y_frac=0.875 and
   `bot_line` y_frac=7.6875 relative to scan_results).
2. The existing `_find_panel_lines` already detects them; only the
   labelling/scoring layer is missing.
3. They are visible even when the title is occluded by particles/snow
   (one of the documented `find_scan_results_anchor` failure modes).
4. End-notches give sub-pixel x-localization the title NCC cannot
   provide.

This makes chrome-line split + score the cheapest path to a second
high-confidence anchor that's robust where the title fails — the
single most impactful gap to fill first.
