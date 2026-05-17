"""In-process delivery of live OCR crops to UI viewers.

The disk-PNG path (``debug_value_<field>_crop.png`` etc.) is for
cross-process viewers (``scripts/*.bat``). For in-process viewers
(calibration dialog, panel finder popout) we want real-time delivery
without the encode→write→decode roundtrip.

Usage from the OCR side:

    from ocr.sc_ocr import live_broadcast
    live_broadcast.broadcast_crop("mass", pil_image)

Usage from the UI side:

    from ocr.sc_ocr import live_broadcast

    def on_crop(field, image):
        # Runs on the OCR worker thread. Convert + emit Qt signal here.
        ...

    live_broadcast.register_listener(on_crop)
    # ...later, on dialog close...
    live_broadcast.unregister_listener(on_crop)

Listeners are called on the OCR worker thread that produced the crop.
They MUST be thread-safe — the typical pattern is to forward the crop
to the UI thread via a Qt signal with the default QueuedConnection.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

CropListener = Callable[[str, Any], None]

_lock = threading.Lock()
_listeners: list[CropListener] = []


def register_listener(fn: CropListener) -> None:
    with _lock:
        if fn not in _listeners:
            _listeners.append(fn)


def unregister_listener(fn: CropListener) -> None:
    with _lock:
        try:
            _listeners.remove(fn)
        except ValueError:
            pass


def has_listeners() -> bool:
    with _lock:
        return bool(_listeners)


def broadcast_crop(field: str, image: Any) -> None:
    """Deliver a crop to all registered in-process listeners.

    No-op when no listener is registered (the common case — the dialog
    is rarely open). A failing listener never blocks delivery to the
    others.
    """
    if not field or image is None:
        return
    with _lock:
        listeners = list(_listeners)
    if not listeners:
        return
    for fn in listeners:
        try:
            fn(field, image)
        except Exception as exc:
            log.debug("live_broadcast listener failed: %s", exc)


def deliver_crop(field: str, image: Any) -> None:
    """Broadcast in-process AND save to ``debug_value_<field>_crop.png``
    when the ``crops`` tag is active.

    The dialog and other in-process viewers receive the crop instantly
    via the broadcast. The disk write happens only when a cross-process
    viewer is watching (heartbeat) or the user pressed "Record Next
    Scan" (capture counter). When neither, the disk write is skipped —
    that's where the lag savings come from.
    """
    if not field or image is None:
        return
    broadcast_crop(field, image)
    try:
        from . import debug_overlay as _dbg_gate
        if not _dbg_gate.is_tag_active("crops"):
            return
    except Exception:
        return
    try:
        image.save(f"debug_value_{field}_crop.png")
    except Exception:
        pass
