"""Train an RGB CRNN for SC mining-HUD value reading.

Mirror of ``train_signal_crnn_rgb.py`` but trained on HUD value crops
(mass / resistance / instability) instead of signature digit strips.
Same architecture, different alphabet, different data source.

Alphabet ``"0123456789.%"`` covers all three HUD field formats:
  * mass        -> integer e.g. ``"1071"``
  * resistance  -> integer-percent e.g. ``"21%"``
  * instability -> 2-decimal e.g. ``"1.43"``

Training data comes from ``training_data_hud_crops/`` (built by
``scripts/extract_hud_value_crops.py``). Each crop's filename
encodes the user-confirmed label as the third ``__``-delimited
segment, with ``p`` substituted for ``.`` (e.g. ``..._1p43.png``
for label ``1.43``).

Run::

    %LOCALAPPDATA%\\Python\\pythoncore-3.14-64\\python.exe \\
        ocr/train_hud_crnn_rgb.py

Outputs (under ocr/models/):
  * model_hud_crnn_rgb.onnx      - exported ONNX (dynamic batch + width)
  * model_hud_crnn_rgb.json      - alphabet / blank index / metadata
  * model_hud_crnn_rgb_best.pth  - best checkpoint (warm-restartable)
  * model_hud_crnn_rgb_train.log - training log
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Paths
THIS_DIR = Path(__file__).resolve().parent          # .../tools/Mining_Signals/ocr
TOOL_DIR = THIS_DIR.parent                          # .../tools/Mining_Signals
MODELS_DIR = THIS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_ROOT = TOOL_DIR / "training_data_hud_crops"

OUT_ONNX = MODELS_DIR / "model_hud_crnn_rgb.onnx"
OUT_META = MODELS_DIR / "model_hud_crnn_rgb.json"
OUT_BEST_PTH = MODELS_DIR / "model_hud_crnn_rgb_best.pth"
OUT_LOG = MODELS_DIR / "model_hud_crnn_rgb_train.log"

if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

# Hyperparameters
ALPHABET = "0123456789.%"       # 12 chars
NUM_CLASSES = len(ALPHABET) + 1  # +1 blank
BLANK_IDX = NUM_CLASSES - 1
H_TARGET = 48
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15
AUG_PER_CAPTURE = 5
VAL_FRAC = 0.15
SEED = 1234

# Logging
log = logging.getLogger("train_hud_crnn_rgb")


def _setup_logging() -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    h_stdout = logging.StreamHandler(sys.stdout)
    h_stdout.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"),
    )
    log.addHandler(h_stdout)
    h_file = logging.FileHandler(str(OUT_LOG), mode="w", encoding="utf-8")
    h_file.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s"),
    )
    log.addHandler(h_file)


# --------------------------------------------------------------------
# Polarity-canonicalize per channel (matches inference preprocessing)
# --------------------------------------------------------------------
def _canon_text_bright(arr: np.ndarray) -> np.ndarray:
    """Force text to be the BRIGHT class via Otsu — same trick the
    runtime ``_canonicalize_polarity`` uses on signature panels."""
    flat = arr.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    total = flat.size
    sum_total = float(np.sum(np.arange(256) * hist))
    sum_bg, w_bg = 0.0, 0
    max_var, threshold = 0.0, 127
    for t in range(256):
        w_bg += int(hist[t])
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * int(hist[t])
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    bright = int((arr > threshold).sum())
    dark = arr.size - bright
    if dark < bright:
        return (255 - arr).astype(np.uint8)
    return arr.astype(np.uint8)


# --------------------------------------------------------------------
# Label parsing from filename
# --------------------------------------------------------------------
def _decode_label_from_filename(name: str) -> Optional[str]:
    """Filename schema: ``<user>__<capture>__<label>.png`` where
    ``<label>`` has ``.`` replaced by ``p``. Decode back to the
    canonical string with dots restored.

    Examples:
        ``user_X__cap_Y__1071.png`` -> ``"1071"``
        ``user_X__cap_Y__1p43.png`` -> ``"1.43"``
        ``user_X__cap_Y__0.png``    -> ``"0"``
    """
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    raw = parts[-1]
    # Reverse the safe-substitution from extract_hud_value_crops.py
    raw = raw.replace("p", ".").replace("n", "-")
    # Validate every char is in alphabet (or '-' which we drop).
    canon = "".join(c for c in raw if c in ALPHABET)
    if not canon:
        return None
    return canon


def _walk_training_pairs() -> list[tuple[Path, str, str]]:
    """Yield ``(png_path, label, field)`` triples across all 3 fields."""
    out: list[tuple[Path, str, str]] = []
    for field in ("mass", "resistance", "instability"):
        field_dir = CROPS_ROOT / field
        if not field_dir.exists():
            continue
        for png in sorted(field_dir.glob("*.png")):
            label = _decode_label_from_filename(png.name)
            if label is None:
                continue
            # For resistance/instability training we want the model to
            # learn the unit suffix where it's rendered. Examine the
            # filename suffix; if the original label was a percent or
            # decimal value we need to reproduce that in the training
            # target. Mass labels are pure integers (no suffix).
            #
            # The extractor stored normalized labels — so a rendered
            # "0%" became filename "0", and rendered "0.00" became
            # filename "0" too. The CRNN has to learn to ignore the
            # unit suffix and just emit the numeric content. We feed
            # the normalized form as the target and let the model
            # learn the mapping.
            out.append((png, label, field))
    return out


# --------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------
class HUDCropDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, str, str]],
        augment: bool = False,
    ) -> None:
        self.items = items
        self.augment = augment
        # Pre-build char->idx map for fast encoding.
        self._ch2idx = {c: i for i, c in enumerate(ALPHABET)}

    def __len__(self) -> int:
        # Each item yields 1 base sample + (AUG_PER_CAPTURE-1) augmented
        # variants when augment=True (matches the signature trainer).
        return len(self.items) * (AUG_PER_CAPTURE if self.augment else 1)

    def __getitem__(self, idx: int):
        if self.augment:
            base_idx = idx // AUG_PER_CAPTURE
            aug_seed = idx % AUG_PER_CAPTURE
        else:
            base_idx = idx
            aug_seed = 0
        png, label, field = self.items[base_idx]
        img = Image.open(png).convert("RGB")
        if self.augment and aug_seed > 0:
            img = self._augment(img, aug_seed)
        arr = np.asarray(img, dtype=np.uint8)
        # Resize to H=H_TARGET via Lanczos, preserve aspect ratio.
        h0 = arr.shape[0]
        if h0 != H_TARGET:
            scale = H_TARGET / max(1, h0)
            new_w = max(16, int(round(arr.shape[1] * scale)))
            pil = Image.fromarray(arr, mode="RGB").resize(
                (new_w, H_TARGET), Image.LANCZOS,
            )
            arr = np.asarray(pil, dtype=np.uint8)
        # Polarity-canonicalize per channel (matches inference).
        canon = np.empty_like(arr)
        for c in range(3):
            canon[..., c] = _canon_text_bright(arr[..., c])
        # Normalize to float32 [0, 1] and channel-first.
        tensor = torch.from_numpy(
            canon.astype(np.float32).transpose(2, 0, 1) / 255.0,
        )
        # Encode label.
        label_ids = torch.tensor(
            [self._ch2idx[c] for c in label if c in self._ch2idx],
            dtype=torch.long,
        )
        return tensor, label_ids, tensor.shape[-1]  # (3, H, W), labels, W

    def _augment(self, img: Image.Image, seed: int) -> Image.Image:
        rng = random.Random(seed * 1000 + id(img) % 999983)
        W, H = img.size
        # Random small horizontal shift (-2..+2 px).
        shift = rng.randint(-2, 2)
        if shift != 0:
            new = Image.new("RGB", (W, H), (0, 0, 0))
            if shift > 0:
                new.paste(img.crop((0, 0, W - shift, H)), (shift, 0))
            else:
                new.paste(img.crop((-shift, 0, W, H)), (0, 0))
            img = new
        # Random vertical pad/crop (-1..+1 px).
        vshift = rng.randint(-1, 1)
        if vshift != 0:
            new = Image.new("RGB", (W, H), (0, 0, 0))
            if vshift > 0:
                new.paste(img.crop((0, 0, W, H - vshift)), (0, vshift))
            else:
                new.paste(img.crop((0, -vshift, W, H)), (0, 0))
            img = new
        # Random brightness jitter (0.85..1.15).
        if rng.random() < 0.5:
            from PIL import ImageEnhance
            factor = 0.85 + rng.random() * 0.30
            img = ImageEnhance.Brightness(img).enhance(factor)
        return img


def _collate(batch):
    """Pad batch images to the same width and pack CTC inputs."""
    images, labels, widths = zip(*batch)
    H = images[0].shape[1]
    max_w = max(int(w) for w in widths)
    B = len(images)
    out = torch.zeros((B, 3, H, max_w), dtype=torch.float32)
    for i, img in enumerate(images):
        out[i, :, :, : img.shape[-1]] = img
    targets = torch.cat(labels) if any(len(l) for l in labels) else torch.zeros(
        (0,), dtype=torch.long,
    )
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    input_widths = torch.tensor(widths, dtype=torch.long)
    return out, targets, input_widths, target_lengths


# --------------------------------------------------------------------
# Model (identical architecture to signature CRNN)
# --------------------------------------------------------------------
class RGBCRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 2, padding=0)
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=hidden, num_layers=2,
            bidirectional=True, batch_first=False,
        )
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = x.squeeze(2)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def time_dim_for_width(w: int) -> int:
    return max(0, (w // 4) - 1)


# --------------------------------------------------------------------
# Greedy CTC decode (training-only metric)
# --------------------------------------------------------------------
def _ctc_decode_greedy(preds: torch.Tensor) -> list[str]:
    """preds: (T, B) int64. Returns list of decoded strings."""
    out = []
    preds = preds.cpu().numpy()
    T, B = preds.shape
    for b in range(B):
        chars = []
        prev = -1
        for t in range(T):
            p = int(preds[t, b])
            if p == prev:
                continue
            prev = p
            if p == BLANK_IDX:
                continue
            if 0 <= p < len(ALPHABET):
                chars.append(ALPHABET[p])
        out.append("".join(chars))
    return out


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------
def main() -> int:
    _setup_logging()
    log.info("HUD CRNN training (RGB, alphabet=%r blank=%d)",
             ALPHABET, BLANK_IDX)

    items = _walk_training_pairs()
    log.info("found %d labeled crops across all fields", len(items))
    if len(items) < 20:
        log.error("too few items to train")
        return 1
    # Stratify val split by field so each field gets representation
    # in both train and val.
    rng = random.Random(SEED)
    by_field: dict[str, list] = {"mass": [], "resistance": [], "instability": []}
    for trip in items:
        by_field[trip[2]].append(trip)
    train_items: list = []
    val_items: list = []
    for f, lst in by_field.items():
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * VAL_FRAC)))
        val_items.extend(lst[:n_val])
        train_items.extend(lst[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    log.info("split: train=%d val=%d", len(train_items), len(val_items))

    train_ds = HUDCropDataset(train_items, augment=True)
    val_ds = HUDCropDataset(val_items, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=0, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)
    model = RGBCRNN().to(device)
    # Warm-restart from checkpoint when present.
    if OUT_BEST_PTH.is_file():
        try:
            ckpt = torch.load(str(OUT_BEST_PTH), map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                log.info("warm-restart from %s (val_acc=%.3f)",
                         OUT_BEST_PTH.name, ckpt.get("val_acc", -1.0))
        except Exception as exc:
            log.warning("warm-restart failed: %s", exc)
    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=4,
    )

    best_val_acc = -1.0
    no_improve = 0
    t_train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for images, targets, input_widths, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)  # (T, B, C)
            T = logits.shape[0]
            B = logits.shape[1]
            # Per-sample valid T from input widths.
            input_lengths = torch.tensor(
                [min(T, time_dim_for_width(int(w))) for w in input_widths],
                dtype=torch.long,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            loss = ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        train_loss = ep_loss / max(1, n_batches)

        # Validation.
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets, input_widths, target_lengths in val_loader:
                images_d = images.to(device)
                logits = model(images_d)
                T = logits.shape[0]
                input_lengths = torch.tensor(
                    [min(T, time_dim_for_width(int(w))) for w in input_widths],
                    dtype=torch.long,
                )
                log_probs = F.log_softmax(logits, dim=-1)
                vloss = ctc_loss(
                    log_probs, targets.to(device),
                    input_lengths, target_lengths,
                )
                val_loss += float(vloss.item())
                preds = logits.argmax(dim=-1)  # (T, B)
                decoded = _ctc_decode_greedy(preds)
                # Reconstitute targets per sample.
                idx = 0
                for i, tl in enumerate(target_lengths.tolist()):
                    tgt_ids = targets[idx:idx + tl].tolist()
                    tgt_str = "".join(ALPHABET[j] for j in tgt_ids)
                    idx += tl
                    if decoded[i] == tgt_str:
                        val_correct += 1
                    val_total += 1
        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        scheduler.step(val_loss)

        elapsed = time.time() - t_train_start
        log.info(
            "epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  "
            "elapsed=%.0fs",
            epoch, EPOCHS, train_loss, val_loss, val_acc, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "epoch": epoch,
                "alphabet": ALPHABET,
                "blank_idx": BLANK_IDX,
                "h_target": H_TARGET,
            }, str(OUT_BEST_PTH))
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                log.info("early stop: no val_acc improvement for %d epochs",
                         EARLY_STOP_PATIENCE)
                break

    log.info("Training done. best_val_acc=%.3f", best_val_acc)

    # ── Export ONNX ──
    if not OUT_BEST_PTH.is_file():
        log.error("no checkpoint to export")
        return 1
    ckpt = torch.load(str(OUT_BEST_PTH), map_location="cpu")
    model_cpu = RGBCRNN()
    model_cpu.load_state_dict(ckpt["state_dict"])
    model_cpu.eval()
    dummy = torch.randn(1, 3, H_TARGET, 200, dtype=torch.float32)
    log.info("exporting ONNX to %s", OUT_ONNX)
    torch.onnx.export(
        model_cpu, dummy, str(OUT_ONNX),
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "logits": {1: "batch", 0: "time"},
        },
        dynamo=False,
    )
    meta = {
        "schema": "crnn_rgb_v1",
        "alphabet": ALPHABET,
        "blank_idx": BLANK_IDX,
        "input_shape": [None, 3, H_TARGET, None],
        "input_normalization": "rgb_u8_to_float[0,1] then polarity-canonicalized per-channel",
        "checkpoint_val_acc": float(ckpt.get("val_acc", -1.0)),
        "checkpoint_val_loss": float(ckpt.get("val_loss", -1.0)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "source_checkpoint": OUT_BEST_PTH.name,
        "training_corpus": str(CROPS_ROOT),
        "training_items": len(items),
        "fields": ["mass", "resistance", "instability"],
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote metadata: %s", OUT_META)
    return 0


if __name__ == "__main__":
    sys.exit(main())
