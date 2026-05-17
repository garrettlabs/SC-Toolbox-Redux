"""Promote reviewed (non-quarantined) signal-icon crops into the
active signal training pool.

Workflow:
  1. ``scripts/extract_pending_icons.py`` crops the ``icon`` bbox
     out of every labeled region2 capture and writes
     ``pending_<source_id>_{gray,rgb}.png`` files into
     ``training_data_pending_review_signal/icon/``.
  2. User runs ``scripts/review_glyphs.py pending_signal_icon`` and
     clicks any crop that doesn't show the location-pin icon. Marked
     crops get moved to ``icon/_quarantine/``.
  3. THIS script moves the SURVIVING (non-quarantined)
     ``pending_*.png`` files into ``training_data_user_sig/icon/``
     so the trainer picks them up on the next run. Files are
     renamed with a ``user_promoted_*`` prefix so their provenance
     stays visible (matches the existing ``promote_reviewed.py``
     convention for HUD digits).

NOTE — synthetic seed cleanup:
  ``training_data_user_sig/icon/`` currently holds 600 synthetic
  ``aug_bad_crop_*.png`` samples that bootstrapped the ``@`` class
  from the blacklist's "bad crop" template. Once enough real
  user-promoted icon crops accumulate (default: at least 30 per the
  RegionSpec floor; ideally 60+ for working coverage) those
  synthetic samples should be removed so the model trains purely on
  real data. Pass ``--purge-synthetic`` to delete them after the
  promote step, gated by ``--min-real`` so we don't strip the seed
  before the replacement pool is large enough. Default off — review
  the promoted count first, then re-run with ``--purge-synthetic``.

SAFETY:
  * Only MOVES files matching ``pending_*.png`` (extractor's prefix).
    Anything else in the staging dir is left alone.
  * Skips files in ``icon/_quarantine/`` — those are user-rejected.
  * Dedupes by content hash against the active pool — won't promote
    a crop that's already byte-identical to one already there.
  * Pass ``--dry-run`` to preview the move without touching anything.

After this script runs (and once you've confirmed the active pool
has enough real samples), retrain the signal model:
    python ocr/train_torch.py --region signal
    python ocr/train_torch.py --region signal_inv
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent

PENDING_DIR = TOOL / "training_data_pending_review_signal"
ACTIVE_DIR = TOOL / "training_data_user_sig"
PENDING_PREFIX = "pending_"
SYNTHETIC_PREFIX = "aug_bad_crop_"

# This pool only contains the icon class; folder name on disk is
# ``icon/`` (per review_glyphs._CHAR_TO_DIRNAME mapping ``@`` → ``icon``).
ICON_CLASS = "icon"


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                buf = f.read(8192)
                if not buf:
                    break
                h.update(buf)
    except OSError:
        return ""
    return h.hexdigest()


def _build_active_hash_set(active_cls: Path) -> set[str]:
    hashes: set[str] = set()
    if not active_cls.is_dir():
        return hashes
    for f in active_cls.glob("*.png"):
        h = _file_hash(f)
        if h:
            hashes.add(h)
    return hashes


def _count_real_in_active(active_cls: Path) -> int:
    """Number of non-synthetic real PNGs currently in the active
    icon pool. Synthetic seed = ``aug_bad_crop_*.png``."""
    if not active_cls.is_dir():
        return 0
    return sum(
        1 for f in active_cls.glob("*.png")
        if not f.name.startswith(SYNTHETIC_PREFIX)
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be promoted without moving any files.",
    )
    p.add_argument(
        "--purge-synthetic", action="store_true",
        help="After promoting, DELETE the synthetic aug_bad_crop_*.png "
             "samples from the active icon pool. Gated by --min-real "
             "so we never strip the seed unless real samples meet the "
             "floor. Default OFF.",
    )
    p.add_argument(
        "--min-real", type=int, default=30,
        help="Minimum number of real (non-aug_bad_crop_) samples that "
             "must exist in the active icon pool before "
             "--purge-synthetic is allowed to delete the synthetic "
             "seeds. Default 30 (matches the RegionSpec floor).",
    )
    args = p.parse_args()

    src_cls = PENDING_DIR / ICON_CLASS
    dst_cls = ACTIVE_DIR / ICON_CLASS

    print("=== Promote reviewed signal-icon crops -> active pool ===")
    print(f"  staging dir: {src_cls}")
    print(f"  active dir:  {dst_cls}")
    print(f"  mode:        {'DRY-RUN' if args.dry_run else 'live (files will be moved)'}")
    print()

    if not src_cls.is_dir():
        print(f"No staging dir at {src_cls}; nothing to promote.")
        return 0

    active_hashes = _build_active_hash_set(dst_cls)
    print(f"  active pool currently has {len(active_hashes)} unique PNG(s)")
    print()

    n_promoted = 0
    n_skipped_dup = 0
    n_skipped_quar = 0
    n_skipped_err = 0

    for f in sorted(src_cls.glob(f"{PENDING_PREFIX}*.png")):
        # Glob doesn't recurse, but defensive guard for _quarantine
        # subdirs anywhere in the path parts.
        if "_quarantine" in f.parts:
            n_skipped_quar += 1
            continue
        h = _file_hash(f)
        if not h:
            n_skipped_err += 1
            continue
        if h in active_hashes:
            n_skipped_dup += 1
            continue
        active_hashes.add(h)

        # Promote: pending_<source>.png → user_promoted_<source>.png
        new_name = "user_promoted_" + f.name[len(PENDING_PREFIX):]
        target = dst_cls / new_name
        # Avoid clobber — append __dN until a free name appears.
        counter = 1
        while target.exists():
            target = dst_cls / (
                new_name[:-len(".png")] + f"__d{counter}.png"
            )
            counter += 1

        if not args.dry_run:
            try:
                dst_cls.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target))
            except OSError as exc:
                print(f"  [warn] move failed {f.name}: {exc}")
                n_skipped_err += 1
                continue
        n_promoted += 1

    print("=== Result ===")
    print(f"  promoted:                          {n_promoted}")
    print(f"  skipped (duplicate of active):     {n_skipped_dup}")
    print(f"  skipped (in _quarantine):          {n_skipped_quar}")
    print(f"  skipped (errors):                  {n_skipped_err}")
    print()

    if args.dry_run:
        print("(dry-run — re-run without --dry-run to actually move the files.)")
        return 0

    real_count = _count_real_in_active(dst_cls)
    print(f"  active icon pool: {real_count} real + "
          f"{sum(1 for f in dst_cls.glob(f'{SYNTHETIC_PREFIX}*.png')) if dst_cls.is_dir() else 0}"
          f" synthetic = {len(list(dst_cls.glob('*.png'))) if dst_cls.is_dir() else 0} total")
    print()

    if args.purge_synthetic:
        if not dst_cls.is_dir():
            print("  [purge] active dir doesn't exist yet — nothing to purge.")
        elif real_count < args.min_real:
            print(
                f"  [purge] BLOCKED: only {real_count} real sample(s) "
                f"in active pool (need >= {args.min_real}). The "
                f"synthetic aug_bad_crop_*.png seed stays put — "
                f"removing it now would leave the @ class undertrained."
            )
        else:
            removed = 0
            for syn in dst_cls.glob(f"{SYNTHETIC_PREFIX}*.png"):
                try:
                    syn.unlink()
                    removed += 1
                except OSError as exc:
                    print(f"  [purge warn] {syn.name}: {exc}")
            print(
                f"  [purge] removed {removed} synthetic "
                f"{SYNTHETIC_PREFIX}*.png sample(s) from "
                f"{dst_cls}. The @ class now trains on real "
                f"user-promoted crops only."
            )

    print()
    print("Next:")
    print("  python ocr/train_torch.py --region signal")
    print("  python ocr/train_torch.py --region signal_inv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
