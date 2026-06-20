<p align="center">
  <img src="assets/sc_toolbox_logo.png" alt="SC Toolbox" width="128">
</p>

<h1 align="center">SC Toolbox Redux Mining</h1>

<p align="center">
  A mining-focused Star Citizen desktop toolkit built from the SC Toolbox codebase.
</p>

<p align="center">
  <a href="https://github.com/garrettlabs/SC-Toolbox-Redux/releases/latest">
    <img src="https://img.shields.io/badge/Download-Latest%20Release-00D4FF?style=for-the-badge&logo=github" alt="Download latest release">
  </a>
  <a href="https://discord.gg/D3hqGU5hNt">
    <img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</p>

---

## What this repo is

This repository is the Redux mining edition of SC Toolbox. It keeps the mining workflows and shared runtime needed to run them, while intentionally leaving out the legacy full-suite launcher surface, installer flow, and non-mining tools.

After reading this README, you should be able to:

1. run the Redux mining launcher from source, or
2. build a lightweight Redux mining distributable.

If you want the original all-tools SC Toolbox experience, use the upstream SC Toolbox project instead.

---

## Included tools

### Mining Signals

Mining Signals is the scanner-assist overlay for mining sessions. It reads the Star Citizen HUD and helps identify scan results, signal strength, instability, resistance, mass, and related mining context.

<p align="center">
  <img src="assets/screenshots/mining_signals.png" alt="Mining Signals" width="800"><br>
  <em>Mining Signals — OCR-assisted mining scanner support.</em>
</p>

### Mining Loadout

Mining Loadout helps compare mining heads, modules, modifiers, and ship setups so you can tune a mining configuration before heading out.

<p align="center">
  <img src="assets/screenshots/mining_loadout.png" alt="Mining Loadout" width="800"><br>
  <em>Mining Loadout — mining component planning and comparison.</em>
</p>

---

## Requirements

For source runs and local builds, use Windows with Python 3.9 or newer.

The source launcher searches common local Python installs and prints the interpreter it selected. If Python is missing, install Python first and rerun the command.

---

## Run from source

From the repository root, run:

```powershell
.\RUN_REDUX_MINING.bat
```

The source launcher:

- starts only the Redux mining surface;
- verifies the mining entrypoint can import before launching;
- avoids the legacy full installer, bootstrapper, and user-selected script paths;
- keeps the visible launcher allowlist limited to Mining Loadout and Mining Signals.

Use this path for fast local iteration and smoke testing.

---

## Build the Redux mining distributable

From the repository root, run:

```powershell
.\build\build_redux_mining.bat
```

The Redux build creates a lightweight mining-only distributable under the local build output folder. It is separate from the legacy SC Toolbox installer path.

Use this when you want a portable Redux mining package rather than a source run.

---

## What is intentionally excluded

This repo is not the legacy full SC Toolbox distribution. It intentionally excludes:

- non-mining tool source trees;
- legacy root launchers and installer scripts;
- archived source snapshots;
- generated debug samples and local OCR scratch data;
- full-suite installer UI assets and packaging files.

That boundary keeps the Redux mining repo smaller and makes the supported launch path explicit.

---

## Development checks

Before publishing a cleaned Redux commit, run the test suite:

```powershell
python -m pytest -q --tb=short
```

At minimum, the Redux checks should prove that the mining build surface exists, the non-mining source boundary stays closed, and the cleanup verifiers agree with the current repository shape.

---

## Support

- Report issues in GitHub Issues.
- Join Discord for usage questions and testing feedback.
- Include your Star Citizen resolution, Windows version, Python version, and what you were doing when the issue appeared.
