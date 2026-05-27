# SC Toolbox v2.2.14

## Mining Signals — auto-heal capture + guided region selection

The big fix users on v2.2.13 needed: the scanner now finds the SCAN RESULTS panel automatically, even when your saved capture region is the wrong size or position. No more "44 mass" / "444,444" garbage reads from a too-tight region cropping out the SCAN RESULTS title.

### Auto-heal capture (silent, no action required)

At scan time, the capture layer now:

1. Expands your saved region by 200 pixels on each side
2. Runs an RGB color-mask HUD locator on the expanded capture to find the actual panel position
3. Crops precisely to the detected panel + a 20-pixel padding
4. Feeds the precisely-cropped image to the full anchor pipeline

If the locator can't find a panel (confidence too low, scanner not visible, transition frame, etc.), the capture falls back to your saved region — pre-v2.2.14 behavior, no regression.

**Existing users with bad regions get fixed automatically on their next scan.** No re-calibration needed.

Opt-out: set `auto_heal_region: false` in the config, or `SC_OCR_AUTO_HEAL_REGION=0` as an environment variable.

### Game-resolution-aware sizing

New "Game Resolution" button in the OCR settings row. Auto-populated from your primary monitor on first run (so most users never need to touch it). The wire-diagram region selector uses this to size its initial bubble proportionally — 1080p users see a small bubble, 4K users see a large one, both showing the correct HUD aspect ratio. Click the button if you run SC windowed at a non-native size or on a non-primary monitor.

### Guided region selector (when you do recalibrate)

When you click "Set Mining HUD Region" or "Set Scanning Region", you now get a pre-sized bubble with a wire-diagram showing what shape to fit:

- **HUD mode**: 5 labeled rows for SCAN RESULTS / Resource / MASS / RESISTANCE / INSTABILITY
- **Signature mode**: Icon box (left) + Numbers box (right)

Interactions:
- **Click and drag the bubble body** to reposition
- **Drag corners or edges** to resize manually
- **Scroll wheel** to scale while keeping the locked aspect ratio
- **Press Enter or click outside** to save
- **Press Esc** to cancel

The aspect ratio is locked to the HUD's actual proportions (~448:670), so you can't accidentally pick a square when you need a tall rectangle. Combined with auto-heal at scan time, you don't need pixel-perfect placement — just get the bubble roughly over the panel.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
