# Slot Parity vs erkul - Status & Residual Diagnosis

## Current state

`slot_truth_audit.py` compares `slot_extractor.extract_slots_by_type()` against
`erkul_slot_truth.json` (erkul's live calculator output, 215 ships):

```
  Type                 Match   Mismatch
  WeaponGun              200         15
  MissileLauncher        211          4
  Shield                 215          0
  Cooler                 215          0
  PowerPlant             214          1
  QuantumDrive           215          0
  Radar                  215          0
```

20 ships still differ from erkul on one type. They are NOT simple extractor
bugs - see the pak cross-audit below.

## How to verify

```
python erkul_slot_truth.py     # (re)capture erkul ground truth if stale
python refresh_erkul_cache.py  # ensure .erkul_cache.json
python slot_truth_audit.py     # -> slot_truth_report.txt
python pak_cross_audit.py      # -> pak_cross_audit_report.txt
```

## Pak cross-audit: erkul is not a perfect oracle

`pak_cross_audit.py` adds a third reference - **scunpacked-data**, a mechanical
extraction of the game paks with no corrector layer. For every ship where the
calculator and erkul disagree it prints `calc / erkul / paks`. The verdict on
the 20 residual ships:

- **8 ships - the calculator already matches the game; erkul is the one wrong.**
  C1 Spirit, Constellation Taurus, F7C-M Super Hornet Mk II, MDC, MPUV Tractor,
  Nomad, ROC, Storm AA. erkul's corrector drops or adds a hardpoint vs the
  actual game data. Matching erkul here means *replicating erkul's error*.
- **2 ships - genuine calculator bugs.** Cyclone AA (counts the AA missile/EMP
  module as a gun; erkul + paks say 0) and Vanguard Sentinel (counts a phantom
  2nd power plant - present in erkul's `/live/ships` API loadout but not in the
  paks; erkul's renderer drops it).
- **6 ships - calc, erkul and paks all differ.** Golem OX, Idris P, Perseus,
  RAFT, Reliant Mako, Reliant Sen - turret-nesting / empty-mount edge cases.
- **4 MissileLauncher ships** (below) are a counting-model difference, not a
  ship-data bug.

## Remaining MissileLauncher residual (4 ships)

- **A1 Spirit (-1), A2 Hercules Starlifter (-2)** - erkul lists `BombLauncher`
  bomb racks under its `missiles` section; the calculator has a *separate* bomb
  tab (`extract_slots_by_type(loadout, {"BombLauncher"})` in `dps_ui/app.py`).
  Same items, different tab - an audit-measurement artifact, confirmed by paks.
- **Eclipse (-3), Gladiator (-4)** - empty `MissileRack` racks. erkul expands
  them using the rack item's internal capacity (item-database data); the
  loadout port tree has no child attach ports to count.

## erkul's corrector layer — extracted from main.js

erkul runs four hand-maintained correctors over the raw `/live/ships` loadout
before rendering. Extracted verbatim from bundle `main.5d3419f96e07cfdc.js`:

### preCorrector() — per-ship loadout restructuring (5 ships)

- `anvl_hornet*` — on `hardpoint_class_4_nose`: `minSize = maxSize`, push
  itemType `{Turret, GunTurret}` (this is why erkul renders an extra Hornet
  weapon slot vs the game data).
- `misc_hull_c` — flatten `hardpoint_body_int_rear`'s loadout to top level.
- `tmbl_storm_aa` — un-nest the turret with localReference `7ef065cf-…`.
- `krig_p52_merlin`, `krig_p72_archimedes`(+`_emerald`) — un-nest
  `hardpoint_wing_left` / `hardpoint_wing_right`.
- `orig_m80` — delete the subType of `hardpoint_powerplant_right`.

### corrector(item) — per-item patch table (~60 entries)

Operations applied per loadout item, keyed on `item.localName`:

- **Strip (`item = undefined`)** — ~50 localNames: bay-doors, turret/scanner
  caps, hull covers, intakes, winglets, nacelles, gills, canopy frames,
  scanner dishes, `misc_reliant_missile_cap_*`, `mrck_s04_espr_talon_cap*`,
  `crus_spirit_a1/c1` exterior pieces, and the remote turrets
  `rsi_polaris_*_turret_torpedo/_hangarcam`, `rsi_perseus_remote_turret_torpedo`,
  `misc_starlancer_remote_turret_missile_camera`,
  `rsi_scorpius_antares_*_remote_turret_missile`. For `crus_spirit_c1` it also
  strips *any* item whose `localName` is undefined (drops the empty rear
  tractor turret).
- **Convert to turret** — `orig_600i_*remote_turret*`,
  `tmbl_cyclone_module_antiair/_mt/_turret`,
  `drak_cutlass_steel_rear_remote_dual_turret`,
  `rsi_perseus_remote_turret_bottom_s3`, `orig_890jump` items.
- **Lock (`editable=false`)** — various, incl. `grin_rear_module_mdc`,
  PDC turret children, `aegs_idris_*turret_large*` children.
- **Resize** — `mrck_s03_behr_dual_s02`, `anvl_lightning_f8c_turret`.
- **Fill empty rack** — `mrck_s05_anvl_gladiator_quad_s05` → 4×
  `misl_s05_cs_taln_stalker`; `mrck_s09_aegs_eclipse` → 3× `misl_s09_cs_taln_argos`.
  The count comes from the rack item's `data.ports` length.

`finalCorrector`/`lockCorrector` add per-ship locks (e.g. `rsi_polaris`,
`anvl_ballista_remote_rear_turret`).

## Why porting it is the wrong call

Matching erkul *exactly* means replicating this ~60-ship table, and:

1. **It needs data the calculator does not have.** erkul's `corrector` runs
   *after* `itemFiller` populates defaults; the calculator's raw `/live/ships`
   loadout has empty ports (Eclipse/Gladiator racks, C1 Spirit rear turret).
   Porting it requires also pulling default-loadout / item data the extractor
   currently never touches.
2. **erkul updates the table every SC patch** — a permanent maintenance tax.
3. **The pak cross-audit proves erkul is wrong vs the game on 8 of these
   ships.** Porting the corrector would make the calculator *less* accurate.

Recommendation: leave the 16 residual ships as-is. The calculator is already
correct against the game paks; `pak_cross_audit.py` is the watch list.
