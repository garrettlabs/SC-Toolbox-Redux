"""Record N seconds of debug_panel_overlay.png frames as a GIF + mosaic.

The OCR pipeline writes ``debug_panel_overlay.png`` after every scan,
and the in-app Panel Finder popout polls that file at 400 ms. This
script taps the same source — but at 50 ms — so we can see every
write the pipeline does, dedupe by mtime, and emit a side-by-side
record of what the Panel Finder is actually seeing.

Outputs land in ``panel_finder_recording/<timestamp>/``:

  * ``mosaic.png``     — every unique frame in a grid, captioned with
                         elapsed-ms-since-recording-start. Best for
                         pasting into a chat with an LLM that can read
                         still images but not video.
  * ``recording.gif``  — animated playback at the real frame timings,
                         for human review.
  * ``frame_NN.png``   — individual frames in case you want to crop
                         or inspect them in an image viewer.

Usage:
    python scripts/record_panel_finder.py             # 5 s, 3 s pre-roll
    python scripts/record_panel_finder.py 10          # 10 s recording
    python scripts/record_panel_finder.py 5 --no-preroll
    python scripts/record_panel_finder.py 5 --preroll 5
    python scripts/record_panel_finder.py 5 --bundle  # zip everything for Claude

Hit Ctrl+C to stop early -- whatever frames have been collected so far
are still saved.

The ``--bundle`` flag adds a "package for Claude" pass after recording:
pops up a small dialog asking what kind of jitter you saw and offering a
place to paste the Panel Finder Debug Mode log, then zips the mosaic +
notes + log + full-resolution frames + a README into one
``claude_bundle_<timestamp>.zip`` you can drag straight into a chat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
TOOL_DIR = _THIS_DIR.parent
OVERLAY_PATH = TOOL_DIR / "debug_panel_overlay.png"
OUT_ROOT = TOOL_DIR / "panel_finder_recording"

POLL_INTERVAL_S = 0.05      # 50 ms — faster than panel finder's 400 ms
                            # so we never miss a write.
MAX_MOSAIC_COLS = 4         # Frames per row in the mosaic.


# ───────────────────────────── helpers ──────────────────────────────

def _try_open(path: Path) -> Optional[Image.Image]:
    """Read the overlay PNG, tolerating mid-write reads.

    The OCR pipeline saves via Pillow which is mostly atomic, but on
    Windows a reader can occasionally hit the file mid-flush. Retry a
    couple times with a tiny sleep before giving up on this tick.
    """
    for attempt in range(3):
        try:
            with Image.open(path) as im:
                im.load()
                return im.copy()
        except Exception:
            if attempt == 2:
                return None
            time.sleep(0.01)
    return None


def _font(size: int) -> ImageFont.ImageFont:
    """Best-available monospaced font; falls back to PIL default."""
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _annotate(frame: Image.Image, label: str) -> Image.Image:
    """Stamp a small caption in the top-left corner of a frame."""
    out = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, max(120, len(label) * 8 + 8), 18), fill=(0, 0, 0))
    draw.text((4, 2), label, fill=(0, 255, 120), font=_font(12))
    return out


def _diff_mask(a: Image.Image, b: Image.Image) -> Optional[Image.Image]:
    """Return a tinted overlay of pixels that changed from ``a`` to ``b``.

    None if the two frames are the same size mismatch (defensive).
    """
    if a.size != b.size:
        return None
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    # Any channel changed by more than 24/255 counts as movement.
    bbox_mask = diff.point(lambda p: 255 if p > 24 else 0).convert("L")
    if not bbox_mask.getbbox():
        return None
    overlay = Image.new("RGB", a.size, (255, 0, 200))
    return Image.composite(overlay, b.convert("RGB"), bbox_mask)


def _build_mosaic(
    frames: List[Tuple[float, Image.Image]],
    *,
    show_diff: bool,
) -> Image.Image:
    """Lay out frames in a grid, captioned with elapsed time.

    With ``show_diff=True``, every frame after the first is overlaid
    with a magenta tint on pixels that changed since the previous
    frame — that is what HUD jitter looks like at a glance.
    """
    if not frames:
        return Image.new("RGB", (320, 60), (32, 32, 32))
    w, h = frames[0][1].size
    n = len(frames)
    cols = min(MAX_MOSAIC_COLS, n)
    rows = (n + cols - 1) // cols
    pad = 6
    header_h = 32
    mw = cols * w + (cols + 1) * pad
    mh = rows * h + (rows + 1) * pad + header_h
    canvas = Image.new("RGB", (mw, mh), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"panel_finder recording  -  {n} unique frames  -  "
        f"span {frames[-1][0] - frames[0][0]:.2f}s  -  "
        f"diff overlay: {'on' if show_diff else 'off'}"
    )
    draw.text((pad, 8), title, fill=(220, 220, 220), font=_font(13))

    prev_img = None
    for idx, (ts, img) in enumerate(frames):
        r, c = divmod(idx, cols)
        x = pad + c * (w + pad)
        y = header_h + pad + r * (h + pad)
        if show_diff and prev_img is not None:
            tinted = _diff_mask(prev_img, img)
            display = tinted if tinted is not None else img
        else:
            display = img
        gap_ms = 0 if idx == 0 else int((ts - frames[idx - 1][0]) * 1000)
        caption = f"#{idx:02d}  +{int(ts*1000):>4d}ms  d{gap_ms:>3d}"
        canvas.paste(_annotate(display, caption), (x, y))
        prev_img = img
    return canvas


# ────────────────────── reusable capture / write ───────────────────
# Public so other front-ends (the GUI launcher in particular) can run
# the same capture-and-write logic without going through the CLI's
# ``record()`` orchestrator.

def capture_frames(
    duration_s: float,
    *,
    on_frame: Optional[Callable[[int, float, "Image.Image"], None]] = None,
) -> List[Tuple[float, "Image.Image"]]:
    """Poll the overlay PNG for ``duration_s`` and return unique frames.

    Each frame is a ``(elapsed_seconds_since_start, PIL.Image)`` tuple.
    ``on_frame`` (if given) is called with ``(1-based-index, elapsed, img)``
    each time a new frame lands -- useful for live progress UIs. Callback
    exceptions are swallowed so a broken UI hook can't crash capture.
    """
    if not OVERLAY_PATH.exists():
        return []
    start = time.monotonic()
    last_mtime = -1.0
    frames: List[Tuple[float, Image.Image]] = []
    try:
        while time.monotonic() - start < duration_s:
            try:
                mtime = OVERLAY_PATH.stat().st_mtime
            except FileNotFoundError:
                time.sleep(POLL_INTERVAL_S)
                continue
            if mtime != last_mtime:
                img = _try_open(OVERLAY_PATH)
                if img is not None:
                    elapsed = time.monotonic() - start
                    frames.append((elapsed, img))
                    last_mtime = mtime
                    if on_frame is not None:
                        try:
                            on_frame(len(frames), elapsed, img)
                        except Exception:
                            pass
            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    return frames


def write_loose_outputs(
    session: Path,
    frames: List[Tuple[float, "Image.Image"]],
    *,
    show_diff: bool = True,
) -> dict:
    """Write per-frame PNGs, mosaic(s), and GIF into ``session``.

    Returns a dict mapping a label (``mosaic``, ``mosaic_no_diff``,
    ``gif``, ``frame_NN``) to the absolute path that was written.
    Empty dict if ``frames`` is empty.
    """
    out: dict = {}
    if not frames:
        return out

    for idx, (_, img) in enumerate(frames):
        p = session / f"frame_{idx:02d}.png"
        img.convert("RGB").save(p)
        out[f"frame_{idx:02d}"] = p

    mosaic_path = session / "mosaic.png"
    _build_mosaic(frames, show_diff=show_diff).save(mosaic_path)
    out["mosaic"] = mosaic_path

    if show_diff:
        plain_path = session / "mosaic_no_diff.png"
        _build_mosaic(frames, show_diff=False).save(plain_path)
        out["mosaic_no_diff"] = plain_path

    if len(frames) >= 2:
        durations = []
        for i in range(len(frames) - 1):
            d = max(40, int((frames[i + 1][0] - frames[i][0]) * 1000))
            durations.append(d)
        durations.append(durations[-1])
        gif_path = session / "recording.gif"
        first = frames[0][1].convert("RGB")
        rest = [f[1].convert("RGB") for f in frames[1:]]
        first.save(
            gif_path,
            save_all=True,
            append_images=rest,
            duration=durations,
            loop=0,
            optimize=False,
        )
        out["gif"] = gif_path

    return out


# ───────────────────────── bundle for Claude ───────────────────────

def _collect_notes_via_dialog() -> Tuple[str, str]:
    """Pop up a Tkinter dialog. Returns ``(notes, debug_log)``.

    Both strings are empty if the user closed the dialog without typing
    anything, or if Tkinter is not importable on this Python build.
    The bundle is still written either way -- the dialog is a
    convenience, not a gate.
    """
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        print("[bundle] tkinter unavailable -- skipping notes dialog")
        return "", ""

    result = {"notes": "", "debug_log": ""}

    root = tk.Tk()
    root.title("Bundle for Claude -- what did you see?")
    root.geometry("700x620")
    try:
        # Bring to front so it doesn't hide behind the game window.
        root.attributes("-topmost", True)
        root.after(200, lambda: root.attributes("-topmost", False))
    except Exception:
        pass

    tk.Label(
        root,
        text="What did you see jittering? (1-3 sentences)",
        anchor="w",
        font=("Segoe UI", 10, "bold"),
    ).pack(fill="x", padx=10, pady=(10, 2))
    tk.Label(
        root,
        text=(
            "Examples:\n"
            "  the whole panel rectangle shifts ~3px right every other frame\n"
            "  MASS reads 27.43 then 27.48 then 27.43 in alternating frames\n"
            "  the green TOP_LINE bar disappears for one frame and the bubble"
            " flickers off"
        ),
        anchor="w",
        justify="left",
        fg="#666",
        font=("Segoe UI", 8),
    ).pack(fill="x", padx=10)
    notes_box = scrolledtext.ScrolledText(
        root, height=6, font=("Consolas", 10), wrap="word",
    )
    notes_box.pack(fill="x", expand=False, padx=10, pady=(4, 10))

    tk.Label(
        root,
        text="Optional: Panel Finder debug log",
        anchor="w",
        font=("Segoe UI", 10, "bold"),
    ).pack(fill="x", padx=10, pady=(0, 2))
    tk.Label(
        root,
        text=(
            "In the app: open Panel Finder popout -> tick Debug Mode ->\n"
            "click 'Copy log + image' -> paste here (Ctrl+V)."
        ),
        anchor="w",
        justify="left",
        fg="#666",
        font=("Segoe UI", 8),
    ).pack(fill="x", padx=10)
    log_box = scrolledtext.ScrolledText(
        root, height=18, font=("Consolas", 9), wrap="none",
    )
    log_box.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))

    def on_save():
        result["notes"] = notes_box.get("1.0", "end").strip()
        result["debug_log"] = log_box.get("1.0", "end").strip()
        root.destroy()

    def on_skip():
        root.destroy()

    tk.Button(
        btn_frame, text="Save bundle", width=14, command=on_save,
        bg="#3a7", fg="white", relief="flat",
    ).pack(side="right", padx=(8, 0))
    tk.Button(
        btn_frame, text="Skip notes", width=14, command=on_skip,
    ).pack(side="right")

    notes_box.focus_set()
    root.mainloop()

    return result["notes"], result["debug_log"]


def build_bundle(
    session: Path,
    frames: List[Tuple[float, Image.Image]],
    notes: str,
    debug_log: str,
) -> Optional[Path]:
    """Zip mosaic + frames + notes + debug_log into a single archive.

    Returns the path to the zip, or None if there were no frames to bundle.
    """
    if not frames:
        return None

    import zipfile

    span = frames[-1][0] - frames[0][0]
    readme = (
        "Panel Finder recording bundle for Claude\n"
        "=========================================\n"
        f"Captured: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Frames:   {len(frames)} unique\n"
        f"Span:     {span:.2f}s\n"
        "\n"
        "Files:\n"
        "  mosaic.png         All captured frames in a grid. Pixels that\n"
        "                     changed from the previous frame are tinted\n"
        "                     magenta -- this is what HUD jitter looks like\n"
        "                     at a glance. Each cell is captioned with its\n"
        "                     elapsed-ms-from-start and gap-from-previous.\n"
        "  mosaic_no_diff.png Same grid without the magenta tint, in case\n"
        "                     the tint is hiding what the user wanted to\n"
        "                     show.\n"
        "  frames/            Individual full-resolution frames in capture\n"
        "                     order. Use these for sub-pixel inspection of\n"
        "                     anything the mosaic shrinks too small to read.\n"
        "  notes.txt          The user's plain-language description of what\n"
        "                     they saw jittering. Read this first -- it\n"
        "                     points at which layer of the pipeline (anchor,\n"
        "                     row finder, OCR voter, gate state machine) is\n"
        "                     most likely involved.\n"
        "  debug_log.txt      Optional. Logs from the OCR pipeline\n"
        "                     (ocr.sc_ocr.api, hud_panel_tracker,\n"
        "                     chrome_lines, etc.) captured by the in-app\n"
        "                     Panel Finder popout's Debug Mode. Tells you\n"
        "                     what the pipeline THOUGHT it read each frame,\n"
        "                     which the mosaic alone cannot show. May be\n"
        "                     '(no debug log provided)' if the user did not\n"
        "                     paste one.\n"
    )

    bundle_path = session / f"claude_bundle_{session.name}.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme)
        zf.writestr("notes.txt", notes or "(no notes provided)\n")
        zf.writestr(
            "debug_log.txt",
            debug_log or "(no debug log provided)\n",
        )
        for name in ("mosaic.png", "mosaic_no_diff.png"):
            p = session / name
            if p.exists():
                zf.write(p, name)
        for idx in range(len(frames)):
            p = session / f"frame_{idx:02d}.png"
            if p.exists():
                zf.write(p, f"frames/frame_{idx:02d}.png")
    return bundle_path


# Back-compat alias for any out-of-tree callers that imported the
# private name during the brief window it was private.
_build_bundle = build_bundle


def open_in_explorer(folder: Path) -> None:
    """Best-effort: open the output folder in the user's file manager."""
    _open_in_explorer(folder)


def _open_in_explorer(folder: Path) -> None:
    """Best-effort: open the output folder in the user's file manager."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f"open {folder!s}")
        else:
            os.system(f"xdg-open {folder!s}")
    except Exception:
        pass


# ───────────────────────────── recorder ─────────────────────────────

def record(
    duration_s: float,
    *,
    preroll_s: float,
    show_diff: bool,
    open_after: bool,
    bundle: bool,
) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    session = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)

    print(f"[record] watching {OVERLAY_PATH}")
    print(f"[record] writing  {session}")
    if not OVERLAY_PATH.exists():
        print("[record] !! overlay PNG does not exist yet")
        print("[record] !! make sure mining_signals_app.py is running and")
        print("[record] !! the panel anchor has fired at least once.")
        return session

    if preroll_s > 0:
        print(f"[record] starting in {preroll_s:.0f}s -- switch to the game now")
        for remaining in range(int(preroll_s), 0, -1):
            print(f"[record]   ...{remaining}")
            time.sleep(1.0)

    print(f"[record] recording for {duration_s:.1f}s -- Ctrl+C to stop early")

    def _on_frame(idx: int, elapsed: float, img: Image.Image) -> None:
        print(
            f"[record] frame #{idx:02d} captured "
            f"@ {elapsed:5.2f}s  ({img.size[0]}x{img.size[1]})"
        )

    frames = capture_frames(duration_s, on_frame=_on_frame)

    if not frames:
        print("[record] !! no frames captured -- overlay PNG never updated.")
        print("[record] !! is the OCR pipeline actually scanning right now?")
        print("[record] !! (the SCAN RESULTS panel needs to be visible)")
        return session

    outputs = write_loose_outputs(session, frames, show_diff=show_diff)
    if "mosaic" in outputs:
        print(f"[record] wrote mosaic: {outputs['mosaic']}")
    if "mosaic_no_diff" in outputs:
        print(f"[record] wrote plain:  {outputs['mosaic_no_diff']}")
    if "gif" in outputs:
        print(f"[record] wrote GIF:    {outputs['gif']}")
    else:
        print("[record] only 1 unique frame -- skipping GIF")

    print(f"[record] DONE -- {len(frames)} frames in {session}")

    if bundle:
        print("[bundle] opening notes dialog -- fill it in then click 'Save bundle'")
        notes, debug_log = _collect_notes_via_dialog()
        bundle_path = build_bundle(session, frames, notes, debug_log)
        if bundle_path is not None:
            print(f"[bundle] wrote zip:    {bundle_path}")
            print(f"[bundle]   notes:      {len(notes)} chars")
            print(f"[bundle]   debug_log:  {len(debug_log)} chars")
            print(f"[bundle] drag the zip into a chat with Claude.")

    if open_after:
        _open_in_explorer(session)
    return session


# ───────────────────────────── entry ────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "duration", nargs="?", type=float, default=5.0,
        help="Seconds to record (default: 5).",
    )
    p.add_argument(
        "--preroll", type=float, default=3.0,
        help="Countdown seconds before recording starts (default: 3).",
    )
    p.add_argument(
        "--no-preroll", action="store_true",
        help="Skip the pre-roll countdown -- start recording immediately.",
    )
    p.add_argument(
        "--no-diff", action="store_true",
        help=(
            "Disable the magenta diff overlay on the mosaic. "
            "Use if the tint is hiding what you wanted to see."
        ),
    )
    p.add_argument(
        "--no-open", action="store_true",
        help="Do not open Explorer to the output folder when finished.",
    )
    p.add_argument(
        "--bundle", action="store_true",
        help=(
            "After recording, pop up a notes dialog and zip mosaic + frames "
            "+ notes + (optional) debug log into claude_bundle_<ts>.zip "
            "for sharing with Claude in one drop."
        ),
    )
    args = p.parse_args()
    preroll = 0.0 if args.no_preroll else max(0.0, args.preroll)
    record(
        args.duration,
        preroll_s=preroll,
        show_diff=not args.no_diff,
        open_after=not args.no_open,
        bundle=args.bundle,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
