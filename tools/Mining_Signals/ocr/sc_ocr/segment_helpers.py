"""HUD-side segmentation post-processing helpers.

Ports the wide-span split / narrow-span merge logic from the signature
pipeline (``_split_wide_signature_spans`` / ``_merge_narrow_signature_spans``
in :mod:`ocr.sc_ocr.api`) into reusable functions the HUD per-glyph
path can call without dragging in the signature-only assumptions
(comma handling, ``D,DDD`` / ``DD,DDD`` structural prior, lexicon).

The trigger for adding these on the HUD side is the binarization-fusion
failure mode the user observed on ``mass=27265`` — the adaptive
binarizer merged all five digits into a single ~180-px span, and the
HUD's ``_segment_glyphs`` then filtered it out by max-width to ~22 px,
leaving only the leading isolated "2". Result: the per-glyph CNN read
"2" while the CRNN read "27265" correctly. The signature pipeline
survives this because the CRNN runs first as a count oracle ("expect
5 digits"), and the helpers here forcibly split the 180-px mega-span
into 5×36-px sub-spans for re-classification.

To avoid a circular import with ``api.py``, these helpers take all
their inputs (the gray canvas, the binary mask, the boxes list) as
arguments. They return the new box list only when a successful
transformation was made; otherwise ``None`` so callers can keep the
original segmenter output.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def split_wide_spans_to_count(
    boxes: list[tuple[int, int, int, int]],
    target_count: int,
    binary: Optional[np.ndarray] = None,
    min_piece_width: int = 3,
) -> Optional[list[tuple[int, int, int, int]]]:
    """Force ``boxes`` to ``target_count`` by splitting the widest spans.

    Mirror of ``_split_wide_signature_spans`` simplified for HUD (no
    comma handling). Each output piece must be at least
    ``min_piece_width`` px wide.

    Strategy: iteratively pick the widest span, split it at its lowest-
    ink interior column when ``binary`` is provided, else split at the
    geometric midpoint. Repeat until the box count hits ``target_count``
    or no further splits are safe.

    Args:
        boxes:        list of ``(x, y, w, h)`` tuples (x-sorted).
        target_count: desired final count (from CRNN count oracle).
        binary:       optional binary mask used to snap split positions
                      to the lowest-ink column. When omitted we split at
                      the geometric midpoint. Same coordinate system as
                      the boxes' x values (i.e. value-crop columns).
        min_piece_width: floor for each output piece's width. Pieces
                      narrower than this would be sub-stroke fragments,
                      not digits.

    Returns:
        new box list of length ``target_count``, or ``None`` if no
        valid split sequence was found.
    """
    if not boxes:
        return None
    n = len(boxes)
    if n >= target_count:
        return None
    if target_count - n > 6:
        # Would need to manufacture >6 spans from existing ones — too
        # large a gap to bridge reliably. The mass=27265 case has
        # n=1, target=5 so delta=4, well under this cap.
        return None

    # Mutable list of [x_start, x_end] pairs we'll work with.
    spans: list[list[int]] = [
        [int(b[0]), int(b[0] + b[2])] for b in boxes
    ]
    # Preserve y/h from each parent box so that when we split a span,
    # all sub-pieces inherit the parent's vertical bounds. Each entry
    # is parallel to ``spans``.
    yh: list[tuple[int, int]] = [(int(b[1]), int(b[3])) for b in boxes]

    safety = 16
    while len(spans) < target_count and safety > 0:
        safety -= 1
        widest_idx = max(
            range(len(spans)),
            key=lambda i: spans[i][1] - spans[i][0],
        )
        s, e = spans[widest_idx]
        widest_w = e - s

        # How many sub-pieces does this span need to become?
        # Estimate from current count gap and span width vs median.
        widths = [span[1] - span[0] for span in spans]
        widths_sorted = sorted(widths)
        median_w = max(1, widths_sorted[len(widths_sorted) // 2])
        room = target_count - len(spans) + 1
        est = int(round(widest_w / max(1, median_w)))
        # When the widest span IS the median (e.g. n=1 with one
        # mega-span, or all spans equal width), the est/median heuristic
        # gives us est=1 and forces a 2-way split per iteration. For
        # the binarization-fusion failure mode (single mega-span
        # representing all N digits) we want a single N-way equal-width
        # split that lands all digit slots in one pass. ``room`` is the
        # number of pieces this span can become without overshooting
        # target_count.
        if widest_w == median_w and len(spans) == 1 and room >= 2:
            expected_subcount = room
        else:
            expected_subcount = max(2, min(est, room))

        # Floor: each piece must be at least ``min_piece_width``.
        if widest_w < min_piece_width * expected_subcount:
            # Fall back to a 2-way split if 2-way pieces are still big
            # enough; else bail entirely.
            if widest_w >= min_piece_width * 2:
                expected_subcount = 2
            else:
                break

        piece_w = widest_w // expected_subcount
        cut_positions: list[int] = []
        for k in range(1, expected_subcount):
            nominal = s + k * piece_w
            if binary is not None:
                # Snap to lowest-ink column within a ±piece_w/4 window
                # so cuts land in actual valleys between glyphs (per
                # signature-side rationale).
                snap_half = max(1, piece_w // 4)
                lo = max(s + min_piece_width, nominal - snap_half)
                hi = min(e - min_piece_width, nominal + snap_half)
                if hi > lo:
                    proj = (binary[:, lo:hi] > 0).astype(np.int32).sum(axis=0)
                    snap = lo + int(np.argmin(proj))
                    # Guard against snap crossing previous cut.
                    prev = cut_positions[-1] if cut_positions else s
                    if snap - prev < min_piece_width:
                        snap = prev + min_piece_width
                    cut_positions.append(snap)
                else:
                    cut_positions.append(nominal)
            else:
                cut_positions.append(nominal)

        new_pieces: list[list[int]] = []
        cur = s
        for cut in cut_positions:
            new_pieces.append([cur, cut])
            cur = cut
        new_pieces.append([cur, e])
        piece_widths = [p[1] - p[0] for p in new_pieces]

        # Validate every piece meets the floor.
        if any(pw < min_piece_width for pw in piece_widths):
            # Fall back to equal-width splits with no snap.
            new_pieces = []
            cur = s
            for k in range(1, expected_subcount):
                new_pieces.append([cur, s + k * piece_w])
                cur = s + k * piece_w
            new_pieces.append([cur, e])
            piece_widths = [p[1] - p[0] for p in new_pieces]
            if any(pw < min_piece_width for pw in piece_widths):
                break

        parent_y, parent_h = yh[widest_idx]
        spans = (
            spans[:widest_idx]
            + new_pieces
            + spans[widest_idx + 1:]
        )
        yh = (
            yh[:widest_idx]
            + [(parent_y, parent_h)] * len(new_pieces)
            + yh[widest_idx + 1:]
        )
        log.info(
            "segment_helpers.split_wide: split widest_w=%d into %d pieces "
            "%s (target=%d, now=%d)",
            widest_w, expected_subcount, piece_widths,
            target_count, len(spans),
        )

    if len(spans) != target_count:
        return None

    # Rebuild (x, y, w, h) tuples preserving each span's inherited
    # vertical bounds.
    new_boxes: list[tuple[int, int, int, int]] = []
    for (sx, ex), (sy, sh) in zip(spans, yh):
        new_boxes.append((int(sx), int(sy), int(ex - sx), int(sh)))
    return new_boxes


def merge_narrow_spans_to_count(
    boxes: list[tuple[int, int, int, int]],
    target_count: int,
) -> Optional[list[tuple[int, int, int, int]]]:
    """Force ``boxes`` to ``target_count`` by merging adjacent narrow pairs.

    Mirror of ``_merge_narrow_signature_spans`` simplified for HUD.

    Strategy: walk the (x-sorted) box list, score each adjacent pair by
    how close the combined width is to the median digit width, and
    merge the best pair. Repeat until count matches.

    Aborts safely when no candidate pair would produce a combined width
    within 1.8× the median (signals we'd be merging two genuine digits
    rather than two halves of one).

    Args:
        boxes:        list of ``(x, y, w, h)`` tuples.
        target_count: desired final count.

    Returns:
        new box list of length ``target_count``, or ``None`` if no
        valid merge sequence was found.
    """
    if not boxes or target_count <= 0:
        return None
    if len(boxes) <= target_count:
        return None
    if len(boxes) - target_count > 6:
        # Don't try to collapse too many spans; that suggests a
        # different failure mode than over-splitting.
        return None

    # Sort by x for adjacency.
    cur = sorted(boxes, key=lambda b: b[0])
    cur = [tuple(int(v) for v in b) for b in cur]

    widths = [b[2] for b in cur]
    widths_sorted = sorted(widths)
    median_w = max(4, widths_sorted[len(widths_sorted) // 2])

    safety = 16
    n_merges = 0
    while safety > 0 and len(cur) > target_count:
        safety -= 1
        best_idx = -1
        best_score = float("inf")
        for i in range(len(cur) - 1):
            a = cur[i]
            b = cur[i + 1]
            combined_w = (b[0] + b[2]) - a[0]
            if combined_w > median_w * 1.8:
                continue
            score = abs(combined_w - median_w)
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            log.info(
                "segment_helpers.merge_narrow: no safe merge candidate "
                "(count=%d expected=%d median_w=%d)",
                len(cur), target_count, median_w,
            )
            break
        a = cur[best_idx]
        b = cur[best_idx + 1]
        nx = a[0]
        ny = min(a[1], b[1])
        nx2 = b[0] + b[2]
        ny2 = max(a[1] + a[3], b[1] + b[3])
        cur = (
            cur[:best_idx]
            + [(int(nx), int(ny), int(nx2 - nx), int(ny2 - ny))]
            + cur[best_idx + 2:]
        )
        n_merges += 1
        log.info(
            "segment_helpers.merge_narrow: merged pair [%d,%d]+[%d,%d] "
            "-> [%d,%d] (combined_w=%d, median_w=%d, count=%d/expected=%d)",
            a[0], a[0] + a[2], b[0], b[0] + b[2],
            nx, nx2, nx2 - nx, median_w, len(cur), target_count,
        )

    if n_merges == 0 or len(cur) != target_count:
        return None
    return cur


def extract_crops_from_boxes(
    gray: np.ndarray,
    binary: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """Re-extract 28×28 float32 [0,1] crops from boxes.

    Mirrors the tail of :func:`ocr.sc_ocr.api._segment_glyphs` (the
    pad=2 + ``PIL.resize(28,28, BILINEAR)`` recipe the CNNs were
    trained against). For each box, re-tighten y to the actual ink
    rows in the binary mask, then pad the gray crop with white and
    resize.

    Args:
        gray:   canonical-polarity grayscale image (bright text on
                dark bg).
        binary: binary mask aligned with ``gray``.
        boxes:  list of ``(x, y, w, h)`` tuples.

    Returns:
        list of 28×28 float32 [0,1] crops, parallel to ``boxes``.
        Boxes whose crop would be empty are skipped so the returned
        list may be shorter than the input.
    """
    # Imported lazily so importing this module doesn't pull PIL when
    # the helpers aren't being invoked.
    from PIL import Image as _PILImage

    crops: list[np.ndarray] = []
    H, W = gray.shape[:2]
    for (bx, by, bw, bh) in boxes:
        x1 = max(0, int(bx))
        x2 = min(W, int(bx + bw))
        if x2 <= x1:
            continue
        y_lo = max(0, int(by))
        y_hi = min(H, int(by + bh))
        if y_hi <= y_lo:
            continue
        # Re-tighten y to actual ink rows within the box's column range.
        col_strip = binary[y_lo:y_hi, x1:x2]
        ys = np.where(np.any(col_strip > 0, axis=1))[0]
        if len(ys) >= 1:
            y1 = y_lo + int(ys[0])
            y2 = y_lo + int(ys[-1]) + 1
        else:
            y1, y2 = y_lo, y_hi
        crop = gray[y1:y2, x1:x2].astype(np.float32)
        if crop.size == 0:
            continue
        pad = 2
        padded = np.full(
            (crop.shape[0] + pad * 2, crop.shape[1] + pad * 2),
            255.0, dtype=np.float32,
        )
        padded[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
        pil = _PILImage.fromarray(padded.astype(np.uint8)).resize(
            (28, 28), _PILImage.BILINEAR,
        )
        crops.append(np.array(pil, dtype=np.float32) / 255.0)
    return crops
