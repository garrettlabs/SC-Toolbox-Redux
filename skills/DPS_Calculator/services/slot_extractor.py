"""Ship loadout slot extraction — pure logic, no UI."""
import re

# Port name keyword → component type inferred when itemTypes is absent.
# Some ships (e.g. Paladin) return ALL hardpoints without itemTypes from Erkul.
# Inference is only activated for a given type when there are ZERO explicitly-typed
# ports of that type in the entire loadout tree (two-phase approach).
_PORT_NAME_TYPES: list[tuple[tuple[str, ...], str]] = [
    (("shield",),        "Shield"),
    (("cooler",),        "Cooler"),
    (("power_plant",),   "PowerPlant"),
    (("powerplant",),    "PowerPlant"),
    (("quantum_drive",), "QuantumDrive"),
    (("radar",),         "Radar"),
]

# Keywords that disqualify a port from type inference regardless of name.
# "cockpit" → cockpit display radars are not user-replaceable slots.
# "screen"  → screen_radar inside manned turrets is a display, not a slot.
# "display" → same reasoning.
# "controller" → shield/cooler controllers are not component slots.
# "helper" → radar_helper/landingpad_helper are internal helper ports, not slots.
_INFERENCE_SKIP_KEYWORDS = ("cockpit", "screen", "display", "controller",
                            "blastshield", "helper")

# Non-canonical itemTypes.type strings normalised to their canonical type.
# The Mule's battery port reports "Powerplant - Power"; erkul renders it as a
# power-plant slot.
_TYPE_ALIASES = {"Powerplant - Power": "PowerPlant"}

# Installed-item localName pattern for blanking caps / cover plates (Talon leg
# caps, Reliant missile caps). erkul strips these — they are not real slots.
# Matched on the item localName, not the port name: the same port can hold a
# real rack on a different ship variant.  'umnt_' (universal mount) items are
# excluded — erkul keeps a capped universal-mount port as an empty slot
# (e.g. F7C-S Ghost centre turret with umnt_anvl_s5_cap).
_CAP_ITEM_RE = re.compile(r"_cap(_(l|r|left|right))?$", re.I)


def _infer_type_from_port(pname: str) -> str | None:
    """Return a component type inferred from the port name, or None."""
    lower = pname.lower()
    # Skip ports whose names contain disqualifying keywords.
    if any(kw in lower for kw in _INFERENCE_SKIP_KEYWORDS):
        return None
    for keywords, type_name in _PORT_NAME_TYPES:
        if all(kw in lower for kw in keywords):
            return type_name
    return None


def _count_explicit_types(loadout: list) -> set:
    """Return the set of itemTypes that appear with explicit type info anywhere
    in the loadout tree.  Used to disable inference for types that already have
    proper ports."""
    found: set[str] = set()

    def _walk(ports):
        for port in (ports or []):
            for t in port.get("itemTypes", []):
                tp = t.get("type", "")
                if tp:
                    found.add(tp)
            _walk(port.get("loadout", []))

    _walk(loadout)
    return found


def _has_editable_typed_port(loadout: list, type_name: str) -> bool:
    """True if any port in the tree is editable and explicitly typed `type_name`."""
    def _walk(ports) -> bool:
        for port in (ports or []):
            if port.get("editable") and any(
                    t.get("type") == type_name
                    for t in port.get("itemTypes", []) or []):
                return True
            if _walk(port.get("loadout", [])):
                return True
        return False

    return _walk(loadout)


_TURRET_HOUSING_SUBTYPES = {
    "TopTurret", "MannedTurret", "BallTurret", "NoseTurret",
    "RemoteTurret", "UpperTurret", "LowerTurret",
}


def _port_label(name: str) -> str:
    s = re.sub(r"hardpoint_|_weapon$|weapon_", "", name, flags=re.I)
    s = re.sub(r"_+", " ", s).strip()
    return s.title() if s else name.replace("_", " ").title()


def _gun_position_count(port: dict) -> int:
    """Count direct children that look like independent gun attachment points.

    A child is a gun position if its name starts with one of the known gun-arm
    prefixes or exactly matches a known gun-arm name.  Returns > 1 when this
    port is a compound housing (multiple independent gun slots inside) rather
    than a single gun mount.

    Prefixes:
      "turret_"             – classic turret arm (turret_left, turret_right, …)
      "hardpoint_class"     – direct gun hardpoint (hardpoint_class_2, …)
      "hardpoint_turret_"   – Paladin-style named arm (hardpoint_turret_weapon_left_a, …)
      "joint_turret_"       – Paladin left/right turret joint arms (joint_turret_weapon_left, …)
      "hardpoint_weapon_"   – generic weapon arm children in compound turrets
                              (e.g. Corsair tail remote turret: hardpoint_weapon_left/right,
                               F7C-M nose turret: hardpoint_weapon_s1_left/right)
      "hardpoint_gimbal_"   – gimbal arm positions inside remote turrets
                              (e.g. Perseus remote turrets: hardpoint_gimbal_left/right)
      "hardpoint_gun_"      – gun arm positions inside remote turrets
                              (e.g. Zeus Mk II remote turret: hardpoint_gun_left/right)
    Exact names:
      "hardpoint_left", "hardpoint_right", "hardpoint_upper", "hardpoint_lower"
      – named gun arms used on some turrets (e.g. Asgard pilot turret)
    """
    _GUN_POS_PREFIXES = ("turret_", "hardpoint_class", "hardpoint_turret_",
                         "joint_turret_", "hardpoint_weapon_",
                         "hardpoint_gimbal_", "hardpoint_gun_")
    _GUN_POS_EXACT = {"hardpoint_left", "hardpoint_right", "hardpoint_upper", "hardpoint_lower"}
    count = 0
    for child in port.get("loadout", []):
        cpname = child.get("itemPortName", "")
        if cpname in _GUN_POS_EXACT or any(cpname.startswith(p) for p in _GUN_POS_PREFIXES):
            count += 1
    return count


def _is_missile_rack_port(port: dict) -> bool:
    """Ground-vehicle missile racks (Ballista, Storm AA, MOTH, Nova, ...) carry
    no itemTypes; identify one by a leaf child whose port name marks it as a
    missile attach point."""
    for child in port.get("loadout", []) or []:
        cname = child.get("itemPortName", "").lower()
        if "missile" in cname and not child.get("loadout"):
            return True
    return False


def _is_pdc_turret_port(pname: str) -> bool:
    """Point-defence turret ports carry no itemTypes. They are named
    hardpoint_pdc or hardpoint_pdc_* ; the paired controller / ai-module
    ports (hardpoint_pdc_wc/_mc/_aim) are not weapons and are excluded."""
    p = pname.lower()
    if p != "hardpoint_pdc" and not p.startswith("hardpoint_pdc_"):
        return False
    return not any(x in p for x in ("controller", "aimodule", "_wc", "_mc", "_aim"))


def _holds_tractor_beam(port: dict) -> bool:
    """True if the port or any descendant has a tractor-beam item installed.
    erkul renders such a turret in its tractor-beam section, not as a weapon
    (e.g. the MPUV Tractor's gun turret holds grin_tractorbeam_s1)."""
    if (port.get("localName") or "").lower().startswith("grin_tractorbeam"):
        return True
    return any(_holds_tractor_beam(c) for c in port.get("loadout", []) or [])


def extract_slots_by_type(loadout: list, accept_types: set) -> list:
    """
    Walk the loadout tree and return slots whose itemTypes match accept_types.
    For turret housings that contain weapon/gun ports, recurse into them.
    Returns list of { id, label, max_size, editable, local_ref }.
    """
    # ── Pre-scan: find which types have at least one explicitly-typed port ──
    # Inference is only used for a component type when NO explicit port of that
    # type exists anywhere in the loadout (handles ships like the Paladin whose
    # entire loadout has no itemTypes, while avoiding phantom slots on ships
    # such as Caterpillar that have both explicit and un-typed shield ports).
    _explicit_types = _count_explicit_types(loadout)
    # Whether the ship has a real (editable) radar slot. Used to drop built-in
    # non-editable cockpit-radar ports that erkul does not render as slots.
    _has_editable_radar = _has_editable_typed_port(loadout, "Radar")

    slots = []

    def _resolve_weapon_ref(port, depth=0):
        """Resolve the actual weapon/missile ref from a gun, turret, or missile port.
        Recursively searches up to 3 levels deep for the innermost weapon ref.

        Hierarchy examples:
          Gun port → hardpoint_class_2 → localReference = weapon UUID
          Turret → turret_left → hardpoint_class_2 → localReference = weapon UUID
          Missile rack → missile_01_attach → localName = missile localName
        """
        if depth > 4:
            return ""

        ln = port.get("localName", "")
        lr = port.get("localReference", "")
        children = port.get("loadout", [])

        # Missile racks: localName starts with 'mrck_', missile is in children
        if ln and ln.startswith("mrck_") and children:
            for child in children:
                child_ln = child.get("localName", "")
                if child_ln and child_ln.startswith("misl_"):
                    return child_ln
            return ln

        # If this port has localName that looks like a weapon/missile, use it
        # Skip names that are gimbal mounts, controllers, bomb racks, or other non-weapons
        _SKIP_PREFIXES = ("controller_", "bmbrck_", "mount_gimbal_", "mount_fixed_",
                          "turret_", "relay_", "vehicle_screen", "radar_display",
                          "grin_tractorbeam", "tmbl_emp", "umnt_", "gmisl_")
        _SKIP_SUBSTRINGS = ("_scoop_", "_camera_mount", "_sensor_mount",
                            "_cap", "blanking", "_blade", "missilerack_blade",
                            "missile_cap")
        if ln and not any(ln.startswith(pfx) for pfx in _SKIP_PREFIXES):
            if ln and any(s in ln for s in _SKIP_SUBSTRINGS):
                return ""
            # Also skip if it has children (it's a housing, not a weapon)
            if not children:
                return ln

        # Search children recursively for the deepest weapon ref
        for child in children:
            child_ipn = child.get("itemPortName", "")
            child_ln = child.get("localName", "")
            child_lr = child.get("localReference", "")
            child_children = child.get("loadout", [])

            # If child has its own children (deeper nesting), recurse
            if child_children:
                result = _resolve_weapon_ref(child, depth + 1)
                if result:
                    return result

            # Child has a localName (weapon/missile) — skip non-weapon names
            if child_ln and not any(child_ln.startswith(pfx) for pfx in _SKIP_PREFIXES):
                return child_ln

            # Child has a localReference (weapon UUID on hardpoint_class_*,
            # hardpoint_left/right, turret_weapon, etc.)
            is_weapon_port = ("class" in child_ipn or "weapon" in child_ipn
                              or "gun" in child_ipn or "turret" in child_ipn
                              or "missile" in child_ipn
                              or child_ipn in ("hardpoint_left", "hardpoint_right",
                                               "hardpoint_upper", "hardpoint_lower"))
            if is_weapon_port:
                if child_lr:
                    return child_lr
                else:
                    # Found the weapon port but it's empty — no stock weapon equipped.
                    # Return "" to prevent falling back to parent's mount UUID.
                    return ""

        # Fall back to this port's localReference
        return lr

    # Port names to skip entirely — not real weapon/missile slots
    _SKIP_PORT_PATTERNS = ("camera", "tractor", "self_destruct", "landing",
                            "fuel_port", "fuel_intake", "docking", "air_traffic", "relay",
                            "salvage", "mining", "scan", "torpedo_storage",
                            "vehicle_screen")

    def walk(ports, parent_label="", inherited_size=None):
        for port in (ports or []):
            pname     = port.get("itemPortName", "")
            pname_lower = pname.lower()

            # Skip non-weapon ports - but keep one whose name matches a skip
            # pattern yet is genuinely a component: a missile rack sharing a
            # fuel-intake port (Starfarer/Gemini), or a Radar on a 'scanner'
            # port (MPUV).
            if any(pat in pname_lower for pat in _SKIP_PORT_PATTERNS):
                _ptypes = {t.get("type", "") for t in port.get("itemTypes", []) or []}
                _keep = (
                    (("fuel_intake" in pname_lower or "fuel_port" in pname_lower)
                     and _is_missile_rack_port(port))
                    or ("scan" in pname_lower and "Radar" in _ptypes)
                )
                if not _keep:
                    continue

            types     = port.get("itemTypes", [])
            editable  = port.get("editable", False)
            max_sz    = port.get("maxSize") or inherited_size or 1
            local_ref = port.get("localName", port.get("localReference", ""))
            children  = port.get("loadout", [])

            # Skip blanking-cap ports — the installed item is a cover plate
            # (Talon leg caps, Reliant missile caps) that erkul does not render
            # as a slot. Keyed on the item localName, not the port name: the
            # same port holds a real rack on other variants (Talon Shrike).
            _ln = port.get("localName", "")
            if _ln and not _ln.startswith("umnt_") and _CAP_ITEM_RE.search(_ln):
                continue

            # A turret occupied by a tractor beam is rendered by erkul in its
            # tractor-beam section, not as a weapon slot (MPUV Tractor).
            if "WeaponGun" in accept_types and _holds_tractor_beam(port):
                continue

            type_names = {_TYPE_ALIASES.get(t.get("type", ""), t.get("type", ""))
                          for t in types}
            sub_names  = {t.get("subType", "") for t in types}

            # Infer component type from port name when itemTypes is absent.
            # Only fires when NO explicitly-typed port of the inferred type exists
            # anywhere in the loadout (two-phase guard), and port name doesn't
            # contain any disqualifying keyword (cockpit, screen, display, controller).
            # The `not children` restriction is intentionally omitted so that ports
            # like Paladin's hardpoint_quantum_drive (which has a jump_drive child)
            # are correctly inferred.
            if not type_names:
                inferred = _infer_type_from_port(pname)
                # Untyped shield/cooler/power-plant/qdrive ports are real slots
                # and are always inferred. Untyped 'radar'-named ports are
                # usually cockpit radar displays, not slots - infer Radar only
                # when the ship has no explicitly-typed Radar port at all.
                if inferred and (inferred != "Radar" or inferred not in _explicit_types):
                    type_names = {inferred}

            # A missile rack carries its missiles as leaf attach ports. Ground
            # vehicles tag the rack with no itemTypes; ships like the Apollo and
            # Hermes tag it MissileLauncher but omit the MissileRack subtype.
            # Either way, route it through the MissileRack handling below.
            if _is_missile_rack_port(port):
                if not type_names:
                    type_names = {"MissileLauncher"}
                # Only genuine missile racks expand into per-missile sub-slots.
                # A pure bomb rack (BombLauncher with no MissileLauncher type)
                # is a single entry to erkul — leave it un-expanded.
                if "MissileLauncher" in type_names:
                    sub_names = sub_names | {"MissileRack"}

            # Point-defence (M2C 'Swarm') turrets carry no itemTypes; identify
            # them by name so they extract as one weapon slot each.
            if not type_names and _is_pdc_turret_port(pname):
                type_names = {"WeaponGun"}

            label = _port_label(pname)
            if re.match(r'^Class \d+$', label, re.I):
                # Generic "Class N" gun hardpoint names add no useful info —
                # inherit the parent's descriptive label unchanged.
                label = parent_label
            elif parent_label:
                label = f"{parent_label} / {label}"

            # Determine what this port actually is
            is_gun         = "WeaponGun" in type_names
            is_missile     = "MissileLauncher" in type_names
            is_bomb        = "BombLauncher" in type_names
            is_gun_turret  = "Turret" in type_names and bool(sub_names & {"Gun", "GunTurret"})
            is_housing     = ("Turret" in type_names or "TurretBase" in type_names) and bool(
                sub_names & (_TURRET_HOUSING_SUBTYPES - {"GunTurret"})
            )
            is_inner_gun   = (
                pname.startswith("turret_")
                or pname.startswith("hardpoint_class")
                or pname.startswith("hardpoint_weapon")
                or pname.startswith("hardpoint_gimbal_")
                or pname.startswith("hardpoint_gun_")
            ) and not types and inherited_size is not None

            # Skip bomb launchers from weapon extraction (they're not guns)
            if is_bomb and "WeaponGun" in accept_types and "BombLauncher" not in accept_types:
                continue

            # Skip real guns from missile extraction. The Idris nose railgun is
            # typed WeaponGun/Gun + MissileLauncher/MissileRack; erkul renders
            # it as a weapon, not a missile.
            if (is_gun and "Gun" in sub_names
                    and "MissileLauncher" in accept_types
                    and "WeaponGun" not in accept_types):
                continue

            # Skip empty, non-editable missile ports — erkul drops a missile
            # hardpoint with no rack, no default item and no way to fill it
            # (Talon Shrike's vestigial wing missile mounts).
            if (is_missile and not editable and not children
                    and not port.get("localName")
                    and not port.get("localReference")):
                continue

            # Skip PURE missile turrets (PDS/CIWS) from gun extraction.
            # Hybrid turrets that have BOTH GunTurret AND MissileTurret subtypes
            # (e.g. Scorpius remote turret) are NOT skipped — they can hold guns.
            is_missile_turret = ("Turret" in type_names
                                 and "MissileTurret" in sub_names
                                 and "GunTurret" not in sub_names)
            if is_missile_turret and "WeaponGun" in accept_types:
                continue

            is_match = bool(type_names & accept_types)

            if "WeaponGun" in accept_types or "MissileLauncher" in accept_types:
                want_guns = "WeaponGun" in accept_types
                want_missiles = "MissileLauncher" in accept_types

                # Skip missile-named ports when extracting guns
                if want_guns and not want_missiles:
                    if ("missile" in pname_lower or "missilerack" in pname_lower
                            or "bombrack" in pname_lower or "bomb_" in pname_lower):
                        if not is_gun or is_missile:
                            continue

                # For missile-only extraction: only extract direct MissileLauncher
                # ports. Don't recurse into turret housings or extract inner gun ports.
                missile_only = want_missiles and not want_guns

                if is_match or (is_gun_turret and not missile_only):
                    # Compound GunTurret / Turret detection:
                    # If the port contains multiple independent gun positions as
                    # children (e.g. 400i remote turrets, Scorpius remote turret,
                    # F7A canard nose), recurse into children instead of adding a
                    # single slot.
                    # A port that is itself WeaponGun-typed is a single weapon to
                    # erkul even when it has multiple gun-position children (e.g.
                    # Asgard pilot turret, Freelancer side cannons) — count it once,
                    # do not recurse.
                    if (children and _gun_position_count(port) > 1 and not is_gun):
                        # Multi-gun turret housing: erkul groups the identical
                        # inner guns into a single 'weapons' entry ("Gun xN").
                        # Emit one slot carrying the gun count to mirror that,
                        # rather than recursing into N separate slots.
                        # Arm-style turrets (hardpoint_turret_*/joint_turret_*
                        # children) have guns one size class smaller than the
                        # housing port.
                        first_child = children[0].get("itemPortName", "") if children else ""
                        size_adjust = (first_child.startswith("hardpoint_turret_") or
                                       first_child.startswith("joint_turret_"))
                        inner_sz = max(max_sz - 1, 1) if size_adjust else max_sz
                        slots.append({
                            "id":        f"{parent_label}:{pname}",
                            "label":     label,
                            "max_size":  inner_sz,
                            "editable":  True,
                            "local_ref": _resolve_weapon_ref(port),
                            "outer_ref": port.get("localReference", "")
                                         or port.get("localName", ""),
                            "gun_count": _gun_position_count(port),
                        })
                    elif "MissileRack" in sub_names and children:
                        # Rack hardware slot (the rack itself, e.g. MSD-423)
                        rack_ref = port.get("localReference", "") or port.get("localName", "")
                        slots.append({
                            "id":        f"rack:{parent_label}:{pname}",
                            "label":     label,
                            "max_size":  max_sz,
                            "editable":  editable,
                            "local_ref": rack_ref,
                            "is_rack":   True,
                        })
                        # Individual missile sub-slots. Most racks name these
                        # missile_NN_attach; torpedo launchers (Vanguard
                        # Harbinger) use torpedo_tray_NN_attach_node.
                        # Only loaded sub-slots count: erkul shows a rack's
                        # equipped missiles, not its empty capacity (a rack's
                        # loadout tree can list more attach ports than the rack
                        # item actually holds, e.g. F7C-M Heartseeker).
                        for child in children:
                            child_ipn = child.get("itemPortName", "")
                            child_ref = child.get("localReference", "") or child.get("localName", "")
                            child_lower = child_ipn.lower()
                            if (("missile" in child_lower or "torpedo" in child_lower)
                                    and child_ref):
                                slots.append({
                                    "id":         f"{parent_label}:{pname}:{child_ipn}",
                                    "label":      label,
                                    "max_size":   max_sz,
                                    "editable":   True,
                                    "local_ref":  child_ref,
                                    "is_missile": True,
                                })
                    else:
                        weapon_ref = _resolve_weapon_ref(port)
                        # outer_ref: what's directly equipped in this hardpoint port.
                        # May be a gimbal UUID (≠ weapon_ref) or a weapon UUID (== weapon_ref).
                        outer_ref = port.get("localReference", "") or port.get("localName", "")
                        slots.append({
                            "id":        f"{parent_label}:{pname}",
                            "label":     label,
                            "max_size":  max_sz,
                            "editable":  editable,
                            "local_ref": weapon_ref,
                            "outer_ref": outer_ref,
                        })
                elif is_housing and not missile_only:
                    # Manned/ball/remote turret housing: collect all identical
                    # inner gun positions and emit ONE grouped slot with gun_count=N
                    # instead of N separate slots (Erkul shows VariPuck S4 ×4 style).
                    _HP_PFXS = ("turret_", "hardpoint_class", "hardpoint_turret_",
                                "joint_turret_", "hardpoint_weapon_",
                                "hardpoint_gimbal_", "hardpoint_gun_")
                    _HP_EXACT = {"hardpoint_left", "hardpoint_right",
                                 "hardpoint_upper", "hardpoint_lower"}
                    inner_guns = [
                        cp for cp in children
                        if not cp.get("itemTypes")
                        and (cp.get("itemPortName", "") in _HP_EXACT
                             or any(cp.get("itemPortName", "").startswith(pfx)
                                    for pfx in _HP_PFXS))
                    ]
                    n = len(inner_guns)
                    if n >= 1:
                        first      = inner_guns[0]
                        weapon_ref = _resolve_weapon_ref(first)
                        outer_ref  = (first.get("localReference", "")
                                      or first.get("localName", ""))
                        # Use the first inner gun's declared size if > 0;
                        # otherwise inherit the housing size (app.py gimbal
                        # resolution will correct it if a gimbal is equipped).
                        inner_sz   = first.get("maxSize") or max_sz
                        slots.append({
                            "id":        f"{parent_label}:{pname}",
                            "label":     label,
                            "max_size":  inner_sz,
                            "editable":  True,
                            "local_ref": weapon_ref,
                            "outer_ref": outer_ref,
                            "gun_count": n,
                        })
                    else:
                        # No recognised inner gun ports — fall back to recursion
                        walk(children, label, max_sz)
                elif is_inner_gun and not missile_only:
                    # Only extract inner gun ports for gun extraction
                    weapon_ref = _resolve_weapon_ref(port)
                    outer_ref  = port.get("localReference", "") or port.get("localName", "")
                    slots.append({
                        "id":        f"{parent_label}:{pname}_{len(slots)}",
                        "label":     label,
                        "max_size":  inherited_size,
                        "editable":  True,
                        "local_ref": weapon_ref,
                        "outer_ref": outer_ref,
                    })
                else:
                    if children:
                        # Pass max_sz (not inherited_size) so size is correctly
                        # inherited by child gun ports (e.g. canard nose children).
                        # For named gun arm ports, pass this arm's own label so the
                        # inner gun hardpoint inherits the direction name (Left A, Right B…)
                        # rather than the generic turret label.
                        is_gun_arm = (pname_lower.startswith("hardpoint_turret_weapon") or
                                      pname_lower.startswith("joint_turret_weapon"))
                        walk(children, label if is_gun_arm else parent_label, max_sz)
            else:
                # Component tab logic (Shield, Cooler, Radar, PowerPlant, QuantumDrive…)
                # '_fake' local names are display/dummy components (e.g. MOLE
                # mining-cab radars) that erkul does not count.
                # A non-editable Radar port is a built-in cockpit display, not a
                # swappable slot — erkul drops it when the ship also has a real
                # editable radar (e.g. Prospector); it is kept when it is the
                # ship's only radar (e.g. Hull A).
                is_builtin_radar = ("Radar" in type_names and not editable
                                    and _has_editable_radar)
                if is_match and "_fake" not in (local_ref or "") and not is_builtin_radar:
                    slots.append({
                        "id":        f"{pname}",
                        "label":     label,
                        "max_size":  max_sz,
                        "editable":  editable,
                        "local_ref": local_ref,
                    })
                elif children:
                    walk(children, parent_label, inherited_size)

    walk(loadout)
    return slots


def extract_mining_laser_slots(loadout: list) -> list:
    """Extract mining laser slots from ship loadout.

    Mining laser ports use 'weapon_mining' or 'mining_laser' in their name and
    are normally blocked by the ``"mining"`` entry in _SKIP_PORT_PATTERNS.
    This dedicated extractor bypasses that skip so mining ships show their
    swappable laser slots.

    The port's ``localReference`` is the mining laser UUID (indexed in the
    repository's mining_lasers_by_ref dict).

    Slot dict: {id, label, max_size, editable, local_ref}
    """
    # Ports whose localReference IS a mining laser (extract directly)
    _ML_DIRECT_KW = ("weapon_mining", "mining_laser")
    # Ports that are containers whose CHILD holds the mining laser
    _ML_CONTAINER_KW = ("mining_arm",)

    slots: list[dict] = []

    def _walk(ports, parent_label=""):
        for port in (ports or []):
            pname    = port.get("itemPortName", "")
            pname_lo = pname.lower()
            children = port.get("loadout", [])

            is_direct    = any(kw in pname_lo for kw in _ML_DIRECT_KW)
            is_container = any(kw in pname_lo for kw in _ML_CONTAINER_KW)

            if is_direct:
                # This port holds the mining laser directly via localReference.
                lr    = port.get("localReference", "") or port.get("localName", "")
                label = _port_label(pname)
                if parent_label:
                    label = f"{parent_label} / {label}"
                slots.append({
                    "id":        f"ml:{parent_label}:{pname}",
                    "label":     label,
                    "max_size":  port.get("maxSize") or 1,
                    "editable":  True,
                    "local_ref": lr,
                })
            elif is_container:
                # e.g. hardpoint_mining_arm (ToolArm) — recurse to find the
                # hardpoint_mining_laser child port.
                label = _port_label(pname)
                if parent_label:
                    label = f"{parent_label} / {label}"
                _walk(children, label)
            elif children:
                # Generic recursion (e.g. UtilityTurret/MannedTurret housings)
                label = _port_label(pname)
                if parent_label and not re.match(r'^Class \d+$', label, re.I):
                    label = f"{parent_label} / {label}"
                _walk(children, label)

    _walk(loadout)
    return slots


def extract_utility_slots(loadout: list, accept_types: set) -> list:
    """Extract slots for utility component types (Container/Cargo ore pods, Module,
    ToolArm, ExternalFuelTank, etc.).

    Unlike extract_slots_by_type, does NOT apply weapon-specific skip patterns,
    so it can find mining pods, fuel pods, salvage arms, and ship modules.

    Returns list of {id, label, max_size, editable, local_ref}.
    """
    slots: list[dict] = []

    def _walk(ports, parent_label=""):
        for port in (ports or []):
            pname    = port.get("itemPortName", "")
            types    = port.get("itemTypes", [])
            children = port.get("loadout", [])
            max_sz   = port.get("maxSize") or 1
            editable = port.get("editable", False)
            lr       = port.get("localReference", "") or port.get("localName", "")

            type_names = {t.get("type", "") for t in types}
            label = _port_label(pname)
            if parent_label:
                label = f"{parent_label} / {label}"

            if accept_types & type_names:
                slots.append({
                    "id":        f"util:{pname}:{parent_label}",
                    "label":     label,
                    "max_size":  max_sz,
                    "editable":  editable,
                    "local_ref": lr,
                })
            elif children:
                _walk(children, label)

    _walk(loadout)
    return slots


def extract_salvage_head_slots(loadout: list) -> list:
    """Extract SalvageHead sub-slots from ToolArm containers.

    The Vulture's structure:
      hardpoint_salvage_arm_left  [ToolArm, no types on head]
        hardpoint_salvage_laser   [no itemTypes, localName=salvage_head_standard]
          hardpoint_salvage_subitem01  [no types, localName=salvage_modifier_*]

    This extractor finds 'hardpoint_salvage_laser' ports nested inside
    ToolArm containers and returns them as SalvageHead slots.
    """
    slots: list[dict] = []
    _SALVAGE_HEAD_KW = ("salvage_laser", "salvage_head")

    def _walk(ports, parent_label="", inside_toolarm=False):
        for port in (ports or []):
            pname    = port.get("itemPortName", "").lower()
            children = port.get("loadout", [])
            types    = {t.get("type", "") for t in port.get("itemTypes", [])}
            max_sz   = port.get("maxSize") or 1
            lr       = port.get("localReference", "") or port.get("localName", "")

            label = _port_label(port.get("itemPortName", ""))
            if parent_label:
                label = f"{parent_label} / {label}"

            if inside_toolarm and any(kw in pname for kw in _SALVAGE_HEAD_KW):
                slots.append({
                    "id":        f"svhd:{parent_label}:{port.get('itemPortName','')}",
                    "label":     label,
                    "max_size":  max_sz,
                    "editable":  port.get("editable", False),
                    "local_ref": lr,
                })
                # Don't recurse deeper — sub-slots are separate
            elif "ToolArm" in types:
                _walk(children, label, inside_toolarm=True)
            elif children:
                _walk(children, label, inside_toolarm)

    _walk(loadout)
    return slots


def extract_fuel_pod_slots(loadout: list) -> list:
    """Extract ExternalFuelTank slots from Starfarer/Gemini loadout.

    Starfarer fuel pod ports (hardpoint_fuel_pod_*) have no itemTypes in
    Erkul's loadout JSON but carry localName pointing to the fuel pod component.
    """
    slots: list[dict] = []
    _FUEL_POD_KW = ("fuel_pod",)

    def _walk(ports, parent_label=""):
        for port in (ports or []):
            pname    = port.get("itemPortName", "").lower()
            children = port.get("loadout", [])
            lr       = port.get("localReference", "") or port.get("localName", "")
            max_sz   = port.get("maxSize") or 1

            label = _port_label(port.get("itemPortName", ""))
            if parent_label:
                label = f"{parent_label} / {label}"

            if any(kw in pname for kw in _FUEL_POD_KW) and lr:
                slots.append({
                    "id":        f"fpod:{port.get('itemPortName','')}:{parent_label}",
                    "label":     label,
                    "max_size":  max_sz,
                    "editable":  port.get("editable", False),
                    "local_ref": lr,
                })
            elif children:
                _walk(children, label)

    _walk(loadout)
    return slots


# Prefixes / substrings that identify gimbal/mount local names
_GIMBAL_PREFIXES  = ("mount_gimbal_", "mrai_pulse_mount_gimbal_")
_MOUNT_SUBSTRINGS = ("_mount_gimbal_",)


def _is_gimbal_local_name(ln: str) -> bool:
    lo = ln.lower()
    return any(lo.startswith(p) for p in _GIMBAL_PREFIXES) or any(s in lo for s in _MOUNT_SUBSTRINGS)


def extract_mount_slots(loadout: list) -> list:
    """Extract gimbal/mount slots from weapon hardpoints.

    Returns one slot per weapon hardpoint that accepts a gimbal.  The slot's
    ``local_ref`` is the gimbal's localName if one is already equipped
    (from the ship's default loadout), otherwise ``""``.

    Slot dict: {id, label, max_size, editable, local_ref}
    """
    slots: list[dict] = []

    # Port names that are never user-accessible weapon hardpoints
    _SKIP_PORT_PATTERNS = ("camera", "tractor", "self_destruct", "landing",
                            "fuel_port", "fuel_intake", "docking", "air_traffic",
                            "relay", "salvage", "mining", "scan", "torpedo_storage")

    def _walk(ports, parent_label=""):
        for port in (ports or []):
            pname    = port.get("itemPortName", "")
            pname_lo = pname.lower()
            if any(pat in pname_lo for pat in _SKIP_PORT_PATTERNS):
                continue

            types    = port.get("itemTypes", [])
            max_sz   = port.get("maxSize") or 1
            children = port.get("loadout", [])
            # localReference holds the UUID of the equipped gimbal/mount (or weapon
            # on fixed mounts).  localName is always "" on weapon ports in Erkul's API.
            ln       = port.get("localReference", "") or port.get("localName", "")

            type_names = {t.get("type", "") for t in types}
            label = _port_label(pname)
            if parent_label and not re.match(r'^Class \d+$', label, re.I):
                label = f"{parent_label} / {label}"

            is_weapon_hp = "WeaponGun" in type_names

            if is_weapon_hp:
                # Pass the UUID/localRef as-is; app.py's find_mount() will resolve it
                # and determine required_tags.  Empty means no gimbal equipped.
                current_gimbal = ln
                slots.append({
                    "id":        f"mount:{parent_label}:{pname}",
                    "label":     label,
                    "max_size":  max_sz,
                    "editable":  True,
                    "local_ref": current_gimbal,
                })
                # Don't recurse — this port is accounted for
            elif children:
                _walk(children, label)

    _walk(loadout)
    return slots
