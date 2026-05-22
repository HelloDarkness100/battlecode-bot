import sys
import math
from cambc import Direction, EntityType, Environment, GameConstants as GC
from constants import (
    PATROL_RADIUS, PATROL_MAX_DISTANCE,
    PATROL_GAP_WAIT_ROUNDS, PANIC_HP_THRESHOLD,
    PATROL_TEMP_ROUND, TEMP_SELF_DESTRUCT_TURNS,
    MIN_TITANIUM, MIN_AXIONITE_FOR_ARMOURED,
    DEBUG, ECONOMY_TEST_MODE,
)

_PATROL_MAX_DSQ = PATROL_MAX_DISTANCE * PATROL_MAX_DISTANCE
from utils import (
    euclidean_dist_sq, xy_to_pos, neighbors_4,
    DIRECTION_DELTAS, direction_between, cardinal_direction_between,
)
from pathfinding import _WALKABLE_BUILDINGS

# Turret types we respond to in priority 2. Launchers can't attack us,
# so there's no point disrupting their supply.
_TURRET_TYPES_THREAT = frozenset({
    EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH,
})

# Enemy walkable buildings — bots can stand on these tiles and attack
# from them. Markers are excluded: they're not walkable, and the correct
# way to deal with an enemy marker is to build over it (engine
# auto-destroys the marker on the build action).
_ENEMY_WALKABLE = frozenset({
    EntityType.ROAD, EntityType.CONVEYOR, EntityType.SPLITTER,
    EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
})

# Enemy buildings patrol bot will destroy (exclude CORE — too tough;
# exclude ROAD — cheap and not worth the 2 Ti / shot cost; exclude
# MARKER — builder bots can't stand on a marker to fire at it and
# can't use ct.destroy on enemy buildings, so markers are unreachable).
_ENEMY_DESTROY_TARGETS = frozenset({
    EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE,
})

_ALL_DIRS_8 = [
    Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
    Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST,
]

# Cap the number of buildings we query HP on per turn in priority 2.
# Prior behaviour queried every allied building in vision (~100 FFI calls
# per turn on mature maps, the main patrol TLE source). 20 covers the
# typical vision footprint (up to ~20 allied buildings in a mature
# economy) at ~20μs per turn.
_HEAL_HP_CHECK_LIMIT = 20


class PatrolMixin:
    """Patrol bot logic — 5-priority state machine around the core."""

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def _run_patrol(self, ct):
        my_xy = self._my_xy

        # Enemy launcher detection: if we ended last turn on a tile adjacent
        # to an enemy launcher but started this turn somewhere else, the
        # launcher picked us up. Blacklist the launcher + its 8 neighbors so
        # pathfinding routes around them, and drop the current path so we
        # replan on the next priority that sets one.
        if (self.prev_intended_pos is not None
                and my_xy != self.prev_intended_pos):
            launcher_xy = self._detect_enemy_launcher_near(self.prev_intended_pos)
            if launcher_xy is not None:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        self.blocked_launcher_tiles.add(
                            (launcher_xy[0] + dx, launcher_xy[1] + dy))
                self.path = None
                self.path_index = 0
                if DEBUG:
                    print(f"[{self.corner}] ENEMY LAUNCH: launcher={launcher_xy} "
                          f"from={self.prev_intended_pos} to={my_xy}",
                          file=sys.stderr)

        # Panic vacate: if we're sitting on one of the 9 core tiles and
        # the core is damaged enough that it needs to spawn more patrols,
        # step off to free up the spawn space. Runs before heal so damaged
        # bots still get healed first via priority 0's self-heal.
        if self._panic_vacate_core(ct, my_xy):
            self._patrol_log("PANIC vacate", my_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 0: self-heal (always runs)
        self._patrol_priority_0_heal(ct, my_xy)

        # Panic heal override: if the core is below PANIC_HP_THRESHOLD
        # and we're already close enough to heal it, do that before
        # running turret response. Otherwise a handful of enemy turrets
        # will soak every bot's priority 1 and nobody gets round to
        # healing the core. Only fires when we're already in range —
        # bots that are far from the core still go do turret duty.
        if self._panic_heal_core(ct, my_xy):
            self._patrol_log("PANIC heal_core", self.core_pos)
            self._patrol_idle_turns = 0
            return

        # TOP PRIORITY: armour the core-adjacent ring. Any allied
        # CONVEYOR in the 12-tile core ring gets destroyed+rebuilt
        # as an ARMOURED_CONVEYOR in the same turn (so enemies can't
        # squat on the tile). Only fires when resources allow it.
        if self._patrol_priority_armour_core_ring(ct, my_xy):
            self._patrol_log("P* armour_core", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 1: respond to enemy turrets (cut their ammo supply,
        # drop a sentinel). Runs before heal so patrol bots address the
        # root cause of damage instead of getting stuck repairing the
        # core's huge HP pool while turrets keep chewing through it.
        if self._patrol_priority_1_turret_response(ct, my_xy):
            self._patrol_log("P1 turret_response", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 2: heal damaged allied buildings
        if self._patrol_priority_2_heal_buildings(ct, my_xy):
            self._patrol_log("P2 heal_bldg", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 3: destroy enemy infrastructure
        if self._patrol_priority_3_destroy_infra(ct, my_xy):
            self._patrol_log("P3 destroy_infra", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 4: repair broken chains
        if self._patrol_priority_4_repair_chain(ct, my_xy):
            self._patrol_log("P4 repair_chain", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 4c: reconnect orphaned harvesters that have no
        # cardinal feeder (conveyor / bridge / splitter / armoured).
        # Tries a conveyor first, falls back to a bridge, and as a
        # last resort destroys the harvester so the tile can be reused.
        if self._patrol_priority_4c_repair_orphan_harvester(ct, my_xy):
            self._patrol_log("P4c repair_orphan", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 4b: opportunistic armour upgrade — if team axionite
        # is comfortably above MIN_AXIONITE_FOR_ARMOURED, upgrade any
        # allied CONVEYOR in action range to ARMOURED_CONVEYOR. No
        # pathfinding — runs only when the bot already sits next to a
        # target. Keeps the upgrade cost bounded and lets circling
        # bots naturally sweep the economy.
        if self._patrol_priority_armour_opportunistic(ct, my_xy):
            self._patrol_log("P4b armour_upgrade", self.patrol_target_xy)
            self._patrol_idle_turns = 0
            return

        # Priority 5: default patrol
        self._patrol_priority_5_circle(ct, my_xy)
        if self.patrol_action == 'return_to_core':
            self._patrol_log("P5 return_to_core", self.core_pos)
        else:
            self._patrol_log("P5 circle", self._patrol_last_clamped)

        # Temporary patrol self-destruct: bots spawned after PATROL_TEMP_ROUND
        # that spend TEMP_SELF_DESTRUCT_TURNS doing nothing but circling.
        # Higher-priority actions (P1-P4) return early and reset the counter
        # next turn via the block below.
        if (self._patrol_spawn_round is not None
                and self._patrol_spawn_round >= PATROL_TEMP_ROUND):
            self._patrol_idle_turns += 1
            if self._patrol_idle_turns >= TEMP_SELF_DESTRUCT_TURNS:
                if DEBUG:
                    print(f"[{self.corner}] temp patrol self-destruct after "
                          f"{self._patrol_idle_turns} idle turns", file=sys.stderr)
                ct.self_destruct()
                return  # self_destruct terminates execution, but be safe

    def _patrol_log(self, label, target):
        """Print this patrol bot's current priority + target waypoint to stdout."""
        my_xy = self._my_xy
        bid = id(self) & 0xFFFF
        print(f"[P{bid:04x}] {label} my=({my_xy[0]},{my_xy[1]}) target={target}")

    # ------------------------------------------------------------------ #
    #  Priority 0: Self-heal                                              #
    # ------------------------------------------------------------------ #

    def _patrol_priority_0_heal(self, ct, my_xy):
        cur_hp = ct.get_hp()
        max_hp = GC.BUILDER_BOT_MAX_HP
        # Only self-heal when critically damaged. A 1-HP chip would
        # otherwise burn the action cooldown every turn and starve
        # priority 2 (heal buildings) — the bot would heal itself and
        # never heal the core.
        #
        # Also skip if we're mid-turret-response: substep_destroy needs
        # the cooldown to fire on the enemy ammo source. Dying while
        # firing is better than never firing at all.
        if self.turret_response_substep > 0:
            self.patrol_prev_hp = cur_hp
            return
        if cur_hp * 2 < max_hp and ct.get_action_cooldown() == 0:
            my_pos = xy_to_pos(my_xy)
            if ct.can_heal(my_pos):
                ct.heal(my_pos)
        self.patrol_prev_hp = cur_hp

    # ------------------------------------------------------------------ #
    #  Panic helpers                                                      #
    # ------------------------------------------------------------------ #

    def _core_hp_fraction(self, ct):
        """Return allied core HP / max HP as a float, or None if unknown.
        Only queries the engine when the core is visible."""
        if self.core_pos is None:
            return None
        entry = self.tile_cache.get(self.core_pos)
        if entry is None or entry[1] is None or entry[2] != EntityType.CORE:
            return None
        core_bid = entry[1]
        try:
            hp = ct.get_hp(core_bid)
        except Exception:
            return None
        max_hp = self._max_hp_by_type.get(EntityType.CORE)
        if max_hp is None:
            try:
                max_hp = ct.get_max_hp(core_bid)
            except Exception:
                return None
            self._max_hp_by_type[EntityType.CORE] = max_hp
        if not max_hp:
            return None
        return hp / max_hp

    def _panic_vacate_core(self, ct, my_xy):
        """If panic mode is active and we're standing on a core tile,
        step off to any adjacent non-core walkable tile so the core can
        spawn more patrol bots in our place."""
        if self.core_pos is None:
            return False
        cx, cy = self.core_pos
        on_core = abs(my_xy[0] - cx) <= 1 and abs(my_xy[1] - cy) <= 1
        if not on_core:
            return False
        hp_frac = self._core_hp_fraction(ct)
        if hp_frac is None or hp_frac >= PANIC_HP_THRESHOLD:
            return False
        # Attempt 1: move directly off the 3x3 core footprint, preferring
        # directions that maximise distance from the center so we don't
        # oscillate back in next turn.
        best_dir = None
        best_score = -1
        for d in _ALL_DIRS_8:
            dx, dy = DIRECTION_DELTAS[d]
            nxy = (my_xy[0] + dx, my_xy[1] + dy)
            if abs(nxy[0] - cx) <= 1 and abs(nxy[1] - cy) <= 1:
                continue  # still on the core
            if not ct.can_move(d):
                continue
            score = euclidean_dist_sq(nxy, self.core_pos)
            if score > best_score:
                best_score = score
                best_dir = d
        if best_dir is not None:
            ct.move(best_dir)
            if DEBUG:
                print(f"[P] panic vacate core from ({my_xy[0]},{my_xy[1]}) "
                      f"(core {int(hp_frac * 100)}% HP)",
                      file=sys.stderr)
            return True

        # Attempt 2: our current core tile has no walkable off-core
        # neighbours (e.g. walls on every side). Step to ANOTHER core
        # tile that does have a free exit, so we can leave next turn.
        relay_dir = None
        for d in _ALL_DIRS_8:
            dx, dy = DIRECTION_DELTAS[d]
            nxy = (my_xy[0] + dx, my_xy[1] + dy)
            if not (abs(nxy[0] - cx) <= 1 and abs(nxy[1] - cy) <= 1):
                continue  # must stay on the core this step
            if not ct.can_move(d):
                continue
            if self._tile_has_off_core_exit(nxy):
                relay_dir = d
                break
        if relay_dir is not None:
            ct.move(relay_dir)
            if DEBUG:
                print(f"[P] panic vacate relay from ({my_xy[0]},{my_xy[1]}) "
                      f"(core {int(hp_frac * 100)}% HP)",
                      file=sys.stderr)
            return True
        return False

    def _panic_heal_core(self, ct, my_xy):
        """If the core is below the panic threshold and we're already in
        heal range, heal it this turn. Skips the whole priority chain
        below so turret response can't starve the heal queue.

        Skipped when we're mid-turret-response so substep_destroy's fire
        loop still gets access to the cooldown — otherwise a bot standing
        on an enemy ammo source next to the core heals forever and never
        actually destroys the turret supply."""
        if self.turret_response_substep > 0:
            return False
        if self.core_pos is None:
            return False
        hp_frac = self._core_hp_fraction(ct)
        if hp_frac is None or hp_frac >= PANIC_HP_THRESHOLD:
            return False
        # Distance to the closest tile of the 3x3 footprint.
        cx, cy = self.core_pos
        closest_x = max(cx - 1, min(cx + 1, my_xy[0]))
        closest_y = max(cy - 1, min(cy + 1, my_xy[1]))
        ddx = my_xy[0] - closest_x
        ddy = my_xy[1] - closest_y
        range_dsq = ddx * ddx + ddy * ddy
        if range_dsq > GC.ACTION_RADIUS_SQ:
            return False  # too far to heal — let turret response run
        # In range. If cooldown is up, heal; otherwise hold the turn so
        # we don't wander off before the cooldown clears.
        if ct.get_action_cooldown() == 0:
            target_pos = xy_to_pos((closest_x, closest_y))
            if ct.can_heal(target_pos):
                ct.heal(target_pos)
                if DEBUG:
                    print(f"[P] panic heal core from ({my_xy[0]},{my_xy[1]}) "
                          f"({int(hp_frac * 100)}% HP)", file=sys.stderr)
        return True

    def _tile_has_off_core_exit(self, xy):
        """True if xy has at least one 8-neighbour that's off the core
        footprint and is passable per tile_cache (not a wall, not another
        core tile, not a known non-walkable building). Used by
        _panic_vacate_core's relay step."""
        if self.core_pos is None:
            return False
        cx, cy = self.core_pos
        tc = self.tile_cache
        for d in _ALL_DIRS_8:
            dx, dy = DIRECTION_DELTAS[d]
            nxy = (xy[0] + dx, xy[1] + dy)
            if abs(nxy[0] - cx) <= 1 and abs(nxy[1] - cy) <= 1:
                continue
            if nxy in self.known_walls:
                continue
            entry = tc.get(nxy)
            if entry is None:
                return True  # unknown — assume passable
            if entry[0] == Environment.WALL:
                continue
            if entry[1] is None:
                return True  # empty / ore — passable
            if entry[2] in _WALKABLE_BUILDINGS:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Priority 2: Heal damaged allied buildings                          #
    # ------------------------------------------------------------------ #

    def _patrol_priority_2_heal_buildings(self, ct, my_xy):
        tc = self.tile_cache
        my_team = self.my_team_cache
        core_pos = self.core_pos

        # Collect allied candidates by distance WITHOUT calling get_hp yet
        # — FFI hp queries on every visible allied building were the main
        # TLE source here (50+ calls × 1μs every turn per patrol bot).
        candidates = []  # (dsq, bid, xy, etype)
        for xy in self._last_vision_set:
            entry = tc.get(xy)
            if entry is None:
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is None or team != my_team:
                continue
            # Roads are cheap, plentiful, and their HP doesn't matter for
            # combat — don't waste a heal action on them.
            if etype == EntityType.ROAD:
                continue
            if core_pos is not None and euclidean_dist_sq(xy, core_pos) > _PATROL_MAX_DSQ:
                continue
            dsq = euclidean_dist_sq(my_xy, xy)
            candidates.append((dsq, bid, xy, etype))

        # Sort by distance and query HP only on the top few — we only ever
        # heal one building per turn, so paying 2 FFI calls × N is wasteful.
        # Among the top few, pick the lowest-HP damaged building (tiebreak
        # on distance) so we triage the most-about-to-die structure first
        # rather than just the closest-damaged one.
        candidates.sort(key=lambda c: c[0])
        mhc = self._max_hp_by_type
        best_bid = None
        best_xy = None
        best_dsq = 999999
        best_etype = None
        best_hp = None
        for dsq, bid, xy, etype in candidates[:_HEAL_HP_CHECK_LIMIT]:
            max_hp = mhc.get(etype)
            if max_hp is None:
                try:
                    max_hp = ct.get_max_hp(bid)
                except Exception:
                    continue
                mhc[etype] = max_hp
            try:
                hp = ct.get_hp(bid)
            except Exception:
                continue
            if hp >= max_hp:
                continue
            if best_hp is None or hp < best_hp or (hp == best_hp and dsq < best_dsq):
                best_hp = hp
                best_bid = bid
                best_xy = xy
                best_dsq = dsq
                best_etype = etype

        if best_bid is None:
            if self.patrol_action == 'heal_bldg':
                self.patrol_action = None
                self.patrol_target_bid = None
                self.path = None
                self.path_index = 0
            return False

        # For the 3x3 core, heal() needs to target a *specific* tile of
        # the footprint — the engine measures distance from bot to the
        # passed target, so targeting the center from an adjacent ring
        # tile (dsq=4 to center, but dsq≤2 to the closest footprint tile)
        # would fail can_heal. Pick the closest footprint tile instead.
        if best_etype == EntityType.CORE:
            cx, cy = best_xy
            closest_x = max(cx - 1, min(cx + 1, my_xy[0]))
            closest_y = max(cy - 1, min(cy + 1, my_xy[1]))
            ddx = my_xy[0] - closest_x
            ddy = my_xy[1] - closest_y
            range_dsq = ddx * ddx + ddy * ddy
            target_pos = xy_to_pos((closest_x, closest_y))
        else:
            range_dsq = best_dsq
            target_pos = xy_to_pos(best_xy)

        # Within action radius — heal
        if range_dsq <= GC.ACTION_RADIUS_SQ:
            if ct.get_action_cooldown() == 0 and ct.can_heal(target_pos):
                ct.heal(target_pos)
            return True

        # Need to move adjacent
        if self.patrol_action != 'heal_bldg' or self.patrol_target_xy != best_xy:
            self.patrol_action = 'heal_bldg'
            self.patrol_target_xy = best_xy
            self.patrol_target_bid = best_bid
            self.path = None
            self.path_index = 0

        if self.path is None or self.path_index >= len(self.path):
            # Pathfind to a neighbor of the target. The target is in
            # vision (Chebyshev ≤ ~5), so cap the A* budget tight —
            # 100 nodes covers the visible footprint and stops runaway
            # searches when allied barriers / launcher zones block the
            # straight-line route.
            goal = self._patrol_find_adjacent(my_xy, best_xy)
            if goal is None:
                return False
            self.path = self._compute_path(my_xy, goal, max_nodes=100)
            self.path_index = 0
            if self.path is None:
                # Couldn't pathfind to this damaged building — let
                # lower priorities (destroy_infra / repair_chain /
                # circle) run instead of standing still.
                return False

        result = self._follow_path(ct)
        if result == 'blocked':
            # Another bot or a freshly-built structure blocked us. Drop
            # the cached path so next turn re-paths around it, and let
            # lower priorities have a shot this turn instead of idling.
            self.path = None
            self.path_index = 0
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Priority 1: Respond to enemy turrets                               #
    # ------------------------------------------------------------------ #

    def _patrol_priority_1_turret_response(self, ct, my_xy):
        """Supply-disruption turret response.

        Multi-turn pipeline:
        0. Identify a new unhandled enemy turret in vision (gunner/sentinel/
           breach — launchers can't attack us).
        1. Find its ammo source (conveyor facing it, adjacent harvester,
           or bridge targeting it).
        2. Pathfind adjacent to the ammo source / sentinel tile.
        3. Destroy or clear the blocking tile (free for allied, fire for
           enemy walkable).
        4. Vacate the sentinel tile if we're standing on it (sentinels
           are non-walkable — can't build under the bot).
        5. Build a sentinel on the cleared tile, facing the turret. If
           the ammo source was a conveyor/bridge the sentinel inherits
           the supply chain behind it; if it was a harvester the sentinel
           sits adjacent and draws ammo from there.

        Defense sentinels are left standing after the turret dies so they
        can defend against future waves. Destroyed allied infrastructure
        around them is tracked in building_cache and rebuilt automatically
        via the economy repair scan.
        """
        if ECONOMY_TEST_MODE:
            return False

        # Each turn, prune turrets from active_turret_threats that we can
        # see and are dead.
        self._update_active_turret_threats()

        # Continue an in-progress response if we're mid-pipeline.
        if self.turret_response_substep > 0 and self.turret_target is not None:
            # Bail out if our turret died while we were working on it.
            if self.turret_target not in self.active_turret_threats:
                # Stale state — reset and re-scan this turn
                self._turret_response_reset()
            else:
                return self._turret_response_continue(ct, my_xy)

        # Find a new turret that we haven't started responding to.
        turret_xy = self._find_unhandled_turret()
        if turret_xy is None:
            if self.patrol_action == 'turret_response':
                self.patrol_action = None
                self.path = None
                self.path_index = 0
            return False

        if not self._turret_response_start(ct, turret_xy):
            return False
        # Don't also pathfind on the same turn — _turret_response_start
        # already burned budget on get_attackable_tiles_from. Let the
        # goto substep fire next turn.
        return True

    # ------------------------------------------------------------------ #
    #  Turret response helpers                                           #
    # ------------------------------------------------------------------ #

    def _update_active_turret_threats(self):
        """Remove turrets from active_turret_threats that are visibly dead."""
        if not self.active_turret_threats:
            return
        tc = self.tile_cache
        my_team = self.my_team_cache
        for txy in list(self.active_turret_threats):
            if txy not in self._last_vision_set:
                continue  # out of vision — keep tracking
            entry = tc.get(txy)
            if entry is None:
                self.active_turret_threats.discard(txy)
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if (bid is None or team == my_team
                    or etype not in _TURRET_TYPES_THREAT):
                self.active_turret_threats.discard(txy)
                if DEBUG:
                    print(f"[P] turret ({txy[0]},{txy[1]}) destroyed",
                          file=sys.stderr)

    def _find_unhandled_turret(self):
        """Return the closest enemy turret in vision that isn't already
        in active_turret_threats. None if there are no new threats."""
        tc = self.tile_cache
        my_team = self.my_team_cache
        my_xy = self._my_xy
        core_pos = self.core_pos
        best_xy = None
        best_dsq = 999999
        for xy in self._last_vision_set:
            if xy in self.active_turret_threats:
                continue
            entry = tc.get(xy)
            if entry is None:
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is None or team == my_team:
                continue
            if etype not in _TURRET_TYPES_THREAT:
                continue
            if core_pos is not None and euclidean_dist_sq(xy, core_pos) > _PATROL_MAX_DSQ:
                continue
            dsq = euclidean_dist_sq(my_xy, xy)
            if dsq < best_dsq:
                best_dsq = dsq
                best_xy = xy
        return best_xy

    def _find_turret_ammo_source(self, ct, turret_xy):
        """Return (source_xy, source_type) feeding turret_xy, or (None, None).

        Priorities:
        1. Conveyor cardinally adjacent to the turret AND pointing at it
        2. Bridge whose target is the turret
        3. Harvester cardinally adjacent to the turret
        """
        tc = self.tile_cache
        tx, ty = turret_xy
        # Conveyor check — prefer this because destroying it inherits the chain
        harvester_fallback = None
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxy = (tx + dx, ty + dy)
            entry = tc.get(nxy)
            if entry is None or entry[1] is None:
                continue
            bid, etype = entry[1], entry[2]
            if etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR,
                         EntityType.SPLITTER):
                try:
                    conv_dir = ct.get_direction(bid)
                except Exception:
                    continue
                cdx, cdy = DIRECTION_DELTAS.get(conv_dir, (0, 0))
                if (nxy[0] + cdx, nxy[1] + cdy) == turret_xy:
                    return (nxy, 'conveyor')
            elif etype == EntityType.HARVESTER and harvester_fallback is None:
                harvester_fallback = nxy

        # Bridge check — any bridge whose output is the turret
        for bxy, btarget in self.bridge_target_cache.items():
            if btarget == turret_xy:
                return (bxy, 'bridge')

        if harvester_fallback is not None:
            return (harvester_fallback, 'harvester')
        return (None, None)

    def _pick_turret_facing_dir(self, ct, sentinel_xy, turret_xy, harvester_xy=None):
        """Return the sentinel facing direction that puts the turret in
        attack range. Returns None if no direction reaches the turret.

        Capped at 3 candidate directions (geometric best + 2 adjacent in
        the 8-dir ring). `get_attackable_tiles_from` is an FFI call that
        returns a list of Position objects — each Position crosses the
        FFI boundary at ~0.4μs and a sentinel cone holds 20-40 tiles, so
        an unbounded 8-direction scan was a 200-400μs spike inside
        `_turret_response_start`. The geometric best-guess hits in the
        vast majority of cases; if the closest 3 directions all fail
        the turret is essentially unreachable from this sentinel spot
        and we abandon (caller adds the turret to active_turret_threats
        so we don't retry the same impossible response next turn)."""
        sentinel_pos = xy_to_pos(sentinel_xy)
        excluded = None
        if harvester_xy is not None:
            excluded = direction_between(sentinel_xy, harvester_xy)

        # Best guess: direction from sentinel toward turret
        guess = direction_between(sentinel_xy, turret_xy)
        idx = _ALL_DIRS_8.index(guess) if guess in _ALL_DIRS_8 else 0
        ordered = [
            _ALL_DIRS_8[idx],
            _ALL_DIRS_8[(idx + 1) % 8],
            _ALL_DIRS_8[(idx - 1) % 8],
        ]

        for d in ordered:
            if d == excluded:
                continue
            try:
                tiles = ct.get_attackable_tiles_from(
                    sentinel_pos, d, EntityType.SENTINEL)
            except Exception:
                continue
            for t in tiles:
                if (t.x, t.y) == turret_xy:
                    return d
        return None

    def _turret_response_reset(self):
        """Clear all per-response state (keeps active_turret_threats,
        which persists across responses)."""
        self.turret_response_substep = 0
        self.turret_target = None
        self.turret_ammo_source = None
        self.turret_ammo_source_type = None
        self.turret_sentinel_tile = None
        self.turret_sentinel_dir = None
        self.patrol_action = None
        self.path = None
        self.path_index = 0
        # If we abandon a response mid-flight (e.g. target already dead)
        # and had stashed an economy cache entry for the sentinel tile,
        # push it back into building_cache so the original conveyor is
        # not forgotten by this patrol's repair pipeline.
        stash = getattr(self, '_turret_response_bc_stash', None)
        if stash is not None and stash[1] is not None:
            self.building_cache[stash[0]] = stash[1]
        self._turret_response_bc_stash = None

    def _turret_response_start(self, ct, turret_xy):
        """Identify ammo source and sentinel tile for a new turret.
        Returns False if we couldn't find a workable response.

        On failure, the turret is still added to active_turret_threats
        to prevent re-attempting the same impossible response every turn
        (the main TLE risk: 4 turrets surrounding a harvester means
        _find_turret_ammo_source + _pick_turret_facing_dir run their
        FFI calls every turn with no progress)."""
        ammo_xy, ammo_type = self._find_turret_ammo_source(ct, turret_xy)
        if ammo_xy is None:
            self.active_turret_threats.add(turret_xy)
            return False

        # Where does the sentinel go?
        #   conveyor/bridge → replace them in place (inherit chain)
        #   harvester → adjacent cardinal tile
        if ammo_type in ('conveyor', 'bridge'):
            sentinel_xy = ammo_xy
        else:
            sentinel_xy = self._pick_harvester_sentinel_tile(ammo_xy)
            if sentinel_xy is None:
                # Primary harvester is fully surrounded — search for an
                # alternative ammo source (another harvester or unclaimed
                # titanium ore) that has a free adjacent tile from which a
                # sentinel can reach the turret.
                alt = self._find_alt_ammo_source(turret_xy)
                if alt is None:
                    self.active_turret_threats.add(turret_xy)
                    return False
                ammo_xy, ammo_type, sentinel_xy = alt

        # Can we actually aim a sentinel from sentinel_xy at the turret?
        harvester_xy = ammo_xy if ammo_type in ('harvester', 'alt_ore') else None
        direction = self._pick_turret_facing_dir(
            ct, sentinel_xy, turret_xy, harvester_xy)
        if direction is None:
            self.active_turret_threats.add(turret_xy)
            return False

        self.turret_target = turret_xy
        self.turret_ammo_source = ammo_xy
        self.turret_ammo_source_type = ammo_type
        self.turret_sentinel_tile = sentinel_xy
        self.turret_sentinel_dir = direction
        # alt_ore: need to build a harvester on the ore first (substep 5→6),
        # then proceed to sentinel placement (substep 1→2→3→4).
        self.turret_response_substep = 5 if ammo_type == 'alt_ore' else 1
        self.active_turret_threats.add(turret_xy)
        self.patrol_action = 'turret_response'
        self.patrol_target_xy = ammo_xy if ammo_type == 'alt_ore' else sentinel_xy
        self.path = None
        self.path_index = 0
        if DEBUG:
            print(f"[P] turret_response start turret=({turret_xy[0]},"
                  f"{turret_xy[1]}) ammo={ammo_type}@({ammo_xy[0]},{ammo_xy[1]}) "
                  f"sentinel=({sentinel_xy[0]},{sentinel_xy[1]}) dir={direction}",
                  file=sys.stderr)
        return True

    def _pick_harvester_sentinel_tile(self, harvester_xy):
        """Find a cardinal neighbor of the harvester suitable for placing
        a sentinel. Prefers genuinely empty tiles that are NOT in our
        building_cache — tracked infrastructure will be rebuilt by other
        bots as soon as we destroy it, racing our sentinel build."""
        tc = self.tile_cache
        my_team = self.my_team_cache
        bc = self.building_cache
        empty_candidates = []
        fallback_candidates = []
        for nxy in neighbors_4(harvester_xy[0], harvester_xy[1]):
            if nxy in self.known_walls:
                continue
            entry = tc.get(nxy)
            if entry is None:
                continue
            env = entry[0]
            if env == Environment.WALL:
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is None:
                if nxy in bc:
                    # Currently empty but we expect to rebuild here —
                    # avoid, economy/patrol repair scan will race us.
                    fallback_candidates.append(nxy)
                else:
                    empty_candidates.append(nxy)
                continue
            # Walkable building we can clear
            if team == my_team and etype in (EntityType.ROAD, EntityType.MARKER):
                fallback_candidates.append(nxy)
                continue
            if team != my_team and etype in _ENEMY_WALKABLE:
                fallback_candidates.append(nxy)
                continue
        if empty_candidates:
            return empty_candidates[0]
        if fallback_candidates:
            return fallback_candidates[0]
        return None

    def _find_alt_ammo_source(self, turret_xy):
        """Search tile_cache for an alternative ammo source when the primary
        harvester's cardinal tiles are all blocked.

        Looks for allied titanium harvesters or unclaimed titanium ore
        within sentinel attack range of the turret that have an available
        adjacent tile (empty or allied road) from which a sentinel can
        reach the turret.

        Returns (ammo_xy, ammo_type, sentinel_xy) or None.
        ammo_type is 'harvester' for existing harvesters, 'alt_ore' for
        unclaimed ore (caller must build a harvester first)."""
        tc = self.tile_cache
        my_team = self.my_team_cache
        my_xy = self._my_xy
        # Sentinel range² = 32 → radius ~5.6 → scan ±6 around turret
        r = 6
        tx, ty = turret_xy
        best = None
        best_dsq = 999999

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xy = (tx + dx, ty + dy)
                entry = tc.get(xy)
                if entry is None:
                    continue
                env, bid, etype, team = entry

                is_harvester = (bid is not None and etype == EntityType.HARVESTER
                                and team == my_team)
                is_ore = (env == Environment.ORE_TITANIUM and bid is None)
                if not is_harvester and not is_ore:
                    continue

                # Find an adjacent tile that is free (empty or allied road)
                for nxy in neighbors_4(xy[0], xy[1]):
                    ne = tc.get(nxy)
                    if ne is None:
                        continue
                    if ne[0] == Environment.WALL or nxy in self.known_walls:
                        continue
                    n_bid, n_etype, n_team = ne[1], ne[2], ne[3]
                    if n_bid is not None:
                        if not (n_team == my_team
                                and n_etype in (EntityType.ROAD, EntityType.MARKER)):
                            continue
                    dsq = euclidean_dist_sq(my_xy, nxy)
                    if dsq < best_dsq:
                        atype = 'harvester' if is_harvester else 'alt_ore'
                        best = (xy, atype, nxy)
                        best_dsq = dsq

        return best

    def _turret_response_continue(self, ct, my_xy):
        """Dispatch to the current substep handler and return True if a
        turn-consuming action was performed."""
        substep = self.turret_response_substep
        if substep == 1:
            return self._turret_response_substep_goto(ct, my_xy)
        if substep == 2:
            return self._turret_response_substep_destroy(ct, my_xy)
        if substep == 3:
            return self._turret_response_substep_vacate(ct, my_xy)
        if substep == 4:
            return self._turret_response_substep_build(ct, my_xy)
        if substep == 5:
            return self._turret_response_substep_goto_ore(ct, my_xy)
        if substep == 6:
            return self._turret_response_substep_build_harvester(ct, my_xy)
        # Unknown — reset defensively.
        self._turret_response_reset()
        return False

    def _turret_response_substep_goto(self, ct, my_xy):
        """Walk within action radius (dsq ≤ 2) of the sentinel tile."""
        sxy = self.turret_sentinel_tile
        dsq = euclidean_dist_sq(my_xy, sxy)
        if dsq <= GC.ACTION_RADIUS_SQ:
            self.turret_response_substep = 2
            self.path = None
            self.path_index = 0
            return self._turret_response_substep_destroy(ct, my_xy)
        if self.path is None or self.path_index >= len(self.path):
            goal = self._patrol_find_adjacent(my_xy, sxy)
            if goal is None:
                self._turret_response_reset()
                return False
            # Sentinel tile is in vision (we found it via vision scan).
            # Tight A* cap — fail fast if blocked.
            self.path = self._compute_path(my_xy, goal, max_nodes=100)
            self.path_index = 0
            if self.path is None:
                self._turret_response_reset()
                return False
        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
        return True

    def _turret_response_substep_destroy(self, ct, my_xy):
        """Clear whatever is on the sentinel tile so we can build there."""
        sxy = self.turret_sentinel_tile
        entry = self.tile_cache.get(sxy)
        if entry is None:
            # Lost vision on the target — try again next turn.
            return True
        bid, etype, team = entry[1], entry[2], entry[3]

        if bid is None:
            # Nothing to clear — move on.
            self.turret_response_substep = 3
            return self._turret_response_substep_vacate(ct, my_xy)

        s_pos = xy_to_pos(sxy)

        # Allied infrastructure: free destroy. destroy() needs dsq ≤ 2.
        # Destroys are DEFERRED to turn-end by the engine, so we can't
        # build the sentinel in the same turn — the engine would still
        # see the old building. Advance to substep 3 and build next
        # turn. Stash the building_cache entry (instead of popping) so
        # the original economy building is remembered: after the
        # sentinel we're placing eventually self-destructs, the
        # repair pipeline will restore the conveyor/bridge/splitter
        # that used to be here.
        if team == self.my_team_cache:
            if euclidean_dist_sq(my_xy, sxy) > GC.ACTION_RADIUS_SQ:
                # Shouldn't happen after goto, but be defensive.
                self.turret_response_substep = 1
                return True
            if ct.can_destroy(s_pos):
                ct.destroy(s_pos)
                self.tile_cache[sxy] = (entry[0], None, None, None)
                # Stash the cache entry aside so our own repair scan
                # doesn't flag sxy in the 1-turn gap before we build the
                # sentinel. Restored after the sentinel is in place so
                # the repair pipeline can rebuild when it self-destructs.
                stash = self.building_cache.pop(sxy, None)
                self._turret_response_bc_stash = (sxy, stash)
                if sxy in self.pending_repairs:
                    self.pending_repairs.remove(sxy)
                if DEBUG:
                    print(f"[P] turret_response destroy ally at "
                          f"({sxy[0]},{sxy[1]})", file=sys.stderr)
            self.turret_response_substep = 3
            return True


        # Enemy walkable: stand on it and fire until destroyed.
        if etype in _ENEMY_WALKABLE:
            if my_xy == sxy:
                if ct.get_action_cooldown() == 0 and ct.can_fire(s_pos):
                    ct.fire(s_pos)
                    if DEBUG:
                        print(f"[P] turret_response fire enemy "
                              f"({sxy[0]},{sxy[1]})", file=sys.stderr)
                return True
            # Walk onto it first. In-vision target — tight cap.
            if self.path is None or self.path_index >= len(self.path):
                self.path = self._compute_path(my_xy, sxy, max_nodes=100)
                self.path_index = 0
                if self.path is None:
                    self._turret_response_reset()
                    return False
            self._follow_path(ct)
            return True

        # Enemy non-walkable (harvester, turret, barrier) — we can't
        # destroy this target, bail and try another angle.
        self._turret_response_reset()
        return False

    def _turret_response_substep_vacate(self, ct, my_xy):
        """If the bot is standing on the sentinel tile, step off to any
        adjacent walkable tile so we can build a non-walkable sentinel
        there next turn."""
        sxy = self.turret_sentinel_tile
        if my_xy != sxy:
            self.turret_response_substep = 4
            return self._turret_response_substep_build(ct, my_xy)
        # Move to any adjacent walkable tile.
        for d in _ALL_DIRS_8:
            if ct.can_move(d):
                ct.move(d)
                return True
        # Stuck — try again next turn (something might clear).
        return True

    def _turret_response_substep_build(self, ct, my_xy):
        """Place the sentinel facing the turret."""
        sxy = self.turret_sentinel_tile
        if my_xy == sxy:
            # Bot drifted back onto the tile — vacate again.
            self.turret_response_substep = 3
            return self._turret_response_substep_vacate(ct, my_xy)

        if euclidean_dist_sq(my_xy, sxy) > GC.ACTION_RADIUS_SQ:
            # Drifted out of range — walk back.
            self.turret_response_substep = 1
            return self._turret_response_substep_goto(ct, my_xy)

        if ct.get_action_cooldown() != 0:
            return True

        # Direction was computed once in _turret_response_start and stored
        # in turret_sentinel_dir. Turrets don't move, so it's still valid.
        direction = self.turret_sentinel_dir

        s_pos = xy_to_pos(sxy)
        if not self._can_spend(ct, GC.SENTINEL_BASE_COST[0]):
            return False  # Hard titanium floor — wait for reserve
        if ct.can_build_sentinel(s_pos, direction):
            ct.build_sentinel(s_pos, direction)
            # Restore the stashed economy cache entry so that when the
            # sentinel eventually self-destructs, Pass B sees the
            # empty tile + cached conveyor/bridge and pushes sxy back
            # into pending_repairs to rebuild the original building.
            stash = getattr(self, '_turret_response_bc_stash', None)
            if stash is not None and stash[0] == sxy and stash[1] is not None:
                self.building_cache[sxy] = stash[1]
            self._turret_response_bc_stash = None
            if DEBUG:
                print(f"[P] defense sentinel @({sxy[0]},{sxy[1]}) "
                      f"vs turret ({self.turret_target[0]},"
                      f"{self.turret_target[1]})",
                      file=sys.stderr)
            self._turret_response_reset()
            return True

        # Build failed. Most common reason: an allied bot built a road
        # or marker on the tile while we were walking. Clear it for free
        # and retry next turn.
        s_now = self.tile_cache.get(sxy)
        if (s_now is not None and s_now[1] is not None
                and s_now[3] == self.my_team_cache
                and s_now[2] in (EntityType.ROAD, EntityType.MARKER,
                                 EntityType.BARRIER)):
            if ct.can_destroy(s_pos):
                ct.destroy(s_pos)
                self.tile_cache[sxy] = (s_now[0], None, None, None)
                self.building_cache.pop(sxy, None)
                if DEBUG:
                    print(f"[P] substep_build clear stale ally "
                          f"{s_now[2]} at ({sxy[0]},{sxy[1]})",
                          file=sys.stderr)
            return True

        # Tile now holds an allied transport (economy rebuilt our chain
        # link) or some other blocker — abandon and persistently skip
        # this turret so we don't loop on the same sentinel_tile every
        # turn. The turret stays in active_turret_threats so
        # _find_unhandled_turret skips it on subsequent scans.
        if (s_now is not None and s_now[1] is not None
                and s_now[3] == self.my_team_cache):
            if DEBUG:
                print(f"[P] substep_build abandoned, ally {s_now[2]} "
                      f"at ({sxy[0]},{sxy[1]})", file=sys.stderr)
            # Keep self.turret_target in active_turret_threats so we
            # don't re-spot this same turret next turn.
            self.turret_response_substep = 0
            self.turret_target = None
            self.turret_ammo_source = None
            self.turret_ammo_source_type = None
            self.turret_sentinel_tile = None
            self.turret_sentinel_dir = None
            self.patrol_action = None
            self.path = None
            self.path_index = 0
            return False

        # Otherwise keep retrying (resources / cooldown).
        return True

    def _turret_response_substep_goto_ore(self, ct, my_xy):
        """Walk onto the unclaimed ore tile so we can build a harvester."""
        ore_xy = self.turret_ammo_source
        if my_xy == ore_xy:
            self.turret_response_substep = 6
            return self._turret_response_substep_build_harvester(ct, my_xy)
        if self.path is None or self.path_index >= len(self.path):
            # In-vision target — tight A* cap.
            self.path = self._compute_path(my_xy, ore_xy, max_nodes=100)
            self.path_index = 0
            if self.path is None:
                self._turret_response_reset()
                return False
        self._follow_path(ct)
        return True

    def _turret_response_substep_build_harvester(self, ct, my_xy):
        """Build a harvester on the ore tile, then transition to sentinel
        placement (substep 1 → goto sentinel tile)."""
        ore_xy = self.turret_ammo_source
        if my_xy != ore_xy:
            self.turret_response_substep = 5
            return self._turret_response_substep_goto_ore(ct, my_xy)
        if ct.get_action_cooldown() != 0:
            return True
        # Destroy any road on the ore (free) so we can build the harvester
        entry = self.tile_cache.get(ore_xy)
        if (entry and entry[1] is not None
                and entry[3] == self.my_team_cache
                and entry[2] in (EntityType.ROAD, EntityType.MARKER)):
            ore_pos = xy_to_pos(ore_xy)
            if ct.can_destroy(ore_pos):
                ct.destroy(ore_pos)
                self.tile_cache[ore_xy] = (entry[0], None, None, None)
            return True  # destroy is deferred — build next turn
        if not self._can_spend(ct, GC.HARVESTER_BASE_COST[0]):
            return True  # wait for titanium
        ore_pos = xy_to_pos(ore_xy)
        if ct.can_build_harvester(ore_pos):
            ct.build_harvester(ore_pos)
            if DEBUG:
                print(f"[P] turret_response built harvester @({ore_xy[0]},"
                      f"{ore_xy[1]})", file=sys.stderr)
            # Now transition to sentinel placement: move off ore, goto
            # sentinel tile, destroy/vacate/build as normal.
            self.turret_ammo_source_type = 'harvester'
            self.turret_response_substep = 1
            self.patrol_target_xy = self.turret_sentinel_tile
            self.path = None
            self.path_index = 0
            return True
        # Can't build — maybe another bot claimed it. Abandon.
        self._turret_response_reset()
        return False

    # ------------------------------------------------------------------ #
    #  Priority 3: Destroy enemy infrastructure                           #
    # ------------------------------------------------------------------ #

    def _patrol_priority_3_destroy_infra(self, ct, my_xy):
        tc = self.tile_cache
        my_team = self.my_team_cache
        core_pos = self.core_pos

        # Find closest enemy building in vision (within patrol range from core)
        best_xy = None
        best_dsq = 999999
        best_etype = None
        for xy in self._last_vision_set:
            entry = tc.get(xy)
            if entry is None:
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is None or team == my_team:
                continue
            if etype not in _ENEMY_DESTROY_TARGETS:
                continue
            if core_pos is not None and euclidean_dist_sq(xy, core_pos) > _PATROL_MAX_DSQ:
                continue
            dsq = euclidean_dist_sq(my_xy, xy)
            if dsq < best_dsq:
                best_dsq = dsq
                best_xy = xy
                best_etype = etype

        if best_xy is None:
            if self.patrol_action == 'destroy_infra':
                self.patrol_action = None
                self.path = None
                self.path_index = 0
            return False

        target_pos = xy_to_pos(best_xy)
        walkable = best_etype in _ENEMY_WALKABLE

        # If walkable: walk onto tile then fire(my_pos). If non-walkable: pathfind adjacent then fire(target_pos).
        if walkable:
            if my_xy == best_xy:
                # Standing on it — fire
                if ct.get_action_cooldown() == 0:
                    my_pos = xy_to_pos(my_xy)
                    if ct.can_fire(my_pos):
                        ct.fire(my_pos)
                return True
            goal = best_xy
        else:
            if best_dsq <= GC.ACTION_RADIUS_SQ:
                if ct.get_action_cooldown() == 0 and ct.can_fire(target_pos):
                    ct.fire(target_pos)
                return True
            goal = self._patrol_find_adjacent(my_xy, best_xy)
            if goal is None:
                return False

        if (self.patrol_action != 'destroy_infra'
                or self.patrol_target_xy != best_xy):
            self.patrol_action = 'destroy_infra'
            self.patrol_target_xy = best_xy
            self.path = None
            self.path_index = 0

        if self.path is None or self.path_index >= len(self.path):
            # Target is in vision (≤ ~5 tiles). Cap A* tight so a
            # blocked route fails fast instead of burning the budget.
            self.path = self._compute_path(my_xy, goal, max_nodes=100)
            self.path_index = 0
            if self.path is None:
                return False

        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Priority 4: Repair broken chains                                   #
    # ------------------------------------------------------------------ #

    def _patrol_priority_4_repair_chain(self, ct, my_xy):
        """Repair destroyed buildings.

        Two sources of repairs:
        1. building_cache-based: pending_repairs populated by _scan_turn
        2. 1-tile gap detection: scan conveyors/bridges in vision for broken targets
        """
        # --- Source 1: building_cache repairs (pending_repairs) ---
        if self.pending_repairs:
            core_pos = self.core_pos
            best_rxy = None
            best_dsq = 999999
            for rxy in self.pending_repairs:
                if core_pos is not None and euclidean_dist_sq(rxy, core_pos) > _PATROL_MAX_DSQ:
                    continue
                dsq = euclidean_dist_sq(my_xy, rxy)
                if dsq < best_dsq:
                    best_dsq = dsq
                    best_rxy = rxy

            if best_rxy is not None:
                if self.patrol_action != 'repair_chain' or self.repair_target != best_rxy:
                    self.patrol_action = 'repair_chain'
                    self.repair_target = best_rxy
                    if best_rxy in self.pending_repairs:
                        self.pending_repairs.remove(best_rxy)
                    self.path = None
                    self.path_index = 0
                self._step_repair(ct)
                if self.repair_target is None:
                    self.patrol_action = None
                return True

        # --- Source 2: 1-tile gap detection (scan vision for broken chain links) ---
        return self._patrol_scan_chain_gaps(ct, my_xy)

    def _patrol_scan_chain_gaps(self, ct, my_xy):
        """Scan allied conveyors/bridges in vision for 1-tile chain gaps.

        For each, check if placing a conveyor at the broken target tile
        in some cardinal direction (not pointing back at the source) would
        reconnect to another allied transport tile.

        Economy bots often chain conveyors one tile at a time — a freshly
        placed conveyor pointing at empty space is *not* a gap to fix; it's
        a chain-in-progress. We wait PATROL_GAP_WAIT_ROUNDS after first
        spotting a gap before committing to repair, and drop entries from
        the watch dict once something gets built there.
        """
        tc = self.tile_cache
        my_team = self.my_team_cache
        bpc = self.bridge_target_cache
        core_pos = self.core_pos

        _HEALTHY = (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE,
                    EntityType.ARMOURED_CONVEYOR, EntityType.CORE)

        # Clean up watch entries: drop an entry only when the gap tile now
        # holds a *healthy allied transport* — that's the unambiguous signal
        # an economy bot extended their chain into the tile. Roads, markers,
        # enemy walkables, and ores are all still valid gap candidates and
        # must stay in the watch dict so the wait timer can elapse.
        if self._patrol_gap_watch:
            for gxy in list(self._patrol_gap_watch):
                g_entry = tc.get(gxy)
                if g_entry is None or gxy not in self._last_vision_set:
                    continue  # out of vision — keep watching
                g_bid, g_etype, g_team = g_entry[1], g_entry[2], g_entry[3]
                if (g_bid is not None and g_team == my_team
                        and g_etype in _HEALTHY):
                    if DEBUG:
                        print(f"[P] gap watch ({gxy[0]},{gxy[1]}) resolved → back to patrol",
                              file=sys.stderr)
                    del self._patrol_gap_watch[gxy]

        repair_target = None
        repair_dir = None
        repair_kind = None

        for xy in self._last_vision_set:
            entry = tc.get(xy)
            if entry is None:
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is None or team != my_team:
                continue

            if etype == EntityType.BRIDGE:
                tgt = bpc.get(xy)
                if tgt is None:
                    continue
                src_xy = xy
                target_xy = tgt
            elif etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                # Read direction from building_cache (populated once per
                # build by scanning.py Pass A). Falling back to ct.get_direction
                # here would cost an FFI call for every visible conveyor,
                # every turn — a major TLE source on mature maps.
                cached = self.building_cache.get(xy)
                if cached is None or cached[0] != etype or cached[1] is None:
                    continue
                dx, dy = DIRECTION_DELTAS.get(cached[1], (0, 0))
                if dx == 0 and dy == 0:
                    continue
                src_xy = xy
                target_xy = (xy[0] + dx, xy[1] + dy)
            else:
                continue

            # Only flag a gap we can currently *see*. Tiles out of vision
            # may hold stale tile_cache data (scanning.py eagerly clears a
            # building out of vision), so acting on them produces phantom
            # gaps on tiles that still hold the original building.
            if target_xy not in self._last_vision_set:
                continue

            t_entry = tc.get(target_xy)
            if t_entry is None:
                continue
            t_env, t_bid, t_etype, t_team = t_entry

            if (t_bid is not None and t_team == my_team and t_etype in _HEALTHY):
                continue

            if core_pos is not None:
                if (abs(target_xy[0] - core_pos[0]) <= 1
                        and abs(target_xy[1] - core_pos[1]) <= 1):
                    continue
                if euclidean_dist_sq(target_xy, core_pos) > _PATROL_MAX_DSQ:
                    continue

            if t_env == Environment.WALL or target_xy in self.known_walls:
                continue

            is_empty = (t_bid is None and t_env == Environment.EMPTY)
            is_ore = (t_bid is None and t_env in (
                Environment.ORE_TITANIUM, Environment.ORE_AXIONITE))
            is_allied_road = (t_bid is not None and t_team == my_team
                              and t_etype == EntityType.ROAD)
            is_enemy_walkable = (t_bid is not None and t_team != my_team
                                 and t_etype in _ENEMY_WALKABLE)

            if not (is_empty or is_ore or is_allied_road or is_enemy_walkable):
                continue

            forbidden = (src_xy[0] - target_xy[0], src_xy[1] - target_xy[1])

            chosen_dir = None
            for d_new in (Direction.NORTH, Direction.EAST,
                          Direction.SOUTH, Direction.WEST):
                ndx, ndy = DIRECTION_DELTAS[d_new]
                if (ndx, ndy) == forbidden:
                    continue
                new_tgt = (target_xy[0] + ndx, target_xy[1] + ndy)
                # Accept core 3x3 footprint directly — only the core's
                # center tile is written to tile_cache as CORE, so the
                # 8 surrounding footprint tiles would otherwise fail the
                # _HEALTHY check and block "conveyor → gap → core" repairs.
                if (core_pos is not None
                        and abs(new_tgt[0] - core_pos[0]) <= 1
                        and abs(new_tgt[1] - core_pos[1]) <= 1):
                    chosen_dir = d_new
                    break
                n_entry = tc.get(new_tgt)
                if n_entry is None:
                    continue
                n_bid, n_etype, n_team = n_entry[1], n_entry[2], n_entry[3]
                if (n_bid is not None and n_team == my_team
                        and n_etype in _HEALTHY):
                    chosen_dir = d_new
                    break

            if chosen_dir is None:
                continue

            repair_target = target_xy
            repair_dir = chosen_dir
            if is_enemy_walkable:
                repair_kind = 'enemy_walkable'
            elif is_allied_road:
                repair_kind = 'allied_road'
            elif is_ore:
                repair_kind = 'ore'
            else:
                repair_kind = 'empty'
            break

        if repair_target is None:
            if self.patrol_action == 'repair_chain':
                self.patrol_action = None
                self.path = None
                self.path_index = 0
            return False

        # Wait-gate: delay the *build action* on newly-spotted gaps so an
        # economy bot chaining through this tile has time to extend their
        # conveyor. The bot still commits to walking toward the target
        # during the wait window so it's in range when the timer elapses.
        # Bypass: if an enemy bot is within 3 Chebyshev tiles of the gap,
        # repair immediately — waiting just hands them a free hop into
        # our chain.
        current_round = ct.get_current_round()
        first_seen = self._patrol_gap_watch.get(repair_target)
        if first_seen is None:
            self._patrol_gap_watch[repair_target] = current_round
            first_seen = current_round
            if DEBUG:
                print(f"[P] gap spotted ({repair_target[0]},{repair_target[1]}) "
                      f"— waiting {PATROL_GAP_WAIT_ROUNDS}r before repair",
                      file=sys.stderr)
        waiting = (current_round - first_seen < PATROL_GAP_WAIT_ROUNDS)
        if waiting and self._enemy_bot_within(repair_target, 3):
            waiting = False
            if DEBUG:
                print(f"[P] gap ({repair_target[0]},{repair_target[1]}) — "
                      f"enemy bot in range, repairing immediately",
                      file=sys.stderr)

        # Commit to this repair target (even during the wait window) so we
        # pathfind toward it and are in range when the timer elapses.
        if (self.patrol_action != 'repair_chain'
                or self.patrol_target_xy != repair_target):
            self.patrol_action = 'repair_chain'
            self.patrol_target_xy = repair_target
            self.path = None
            self.path_index = 0

        target_pos = xy_to_pos(repair_target)
        dsq = euclidean_dist_sq(my_xy, repair_target)

        if dsq > GC.ACTION_RADIUS_SQ:
            if self.path is None or self.path_index >= len(self.path):
                goal = self._patrol_find_adjacent(my_xy, repair_target)
                if goal is None:
                    return False
                # In-vision target — tight A* cap.
                self.path = self._compute_path(my_xy, goal, max_nodes=100)
                self.path_index = 0
                if self.path is None:
                    return False
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
                return False
            return True

        # In action range. If still within the wait window, hold position
        # without acting so an economy bot has room to place its own conveyor.
        if waiting:
            return True

        if ct.get_action_cooldown() != 0:
            return True

        if repair_kind == 'enemy_walkable':
            if ct.can_fire(target_pos):
                ct.fire(target_pos)
            return True

        if repair_kind == 'allied_road':
            if ct.can_destroy(target_pos):
                ct.destroy(target_pos)
                old = self.tile_cache.get(repair_target)
                if old:
                    self.tile_cache[repair_target] = (old[0], None, None, None)

        if ct.can_build_conveyor(target_pos, repair_dir):
            if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                return True  # Wait for reserve
            ct.build_conveyor(target_pos, repair_dir)
            if DEBUG:
                print(f"[P] gap-repaired at ({repair_target[0]},{repair_target[1]}) facing {repair_dir}",
                      file=sys.stderr)
            # Done with this gap — forget the watch entry so a future break
            # on the same tile starts a fresh wait.
            self._patrol_gap_watch.pop(repair_target, None)
            self.patrol_action = None
            self.path = None
            self.path_index = 0
        return True

    # ------------------------------------------------------------------ #
    #  Priority 4c: Reconnect orphaned harvesters                         #
    # ------------------------------------------------------------------ #

    def _patrol_priority_4c_repair_orphan_harvester(self, ct, my_xy):
        """Find allied harvesters with no cardinal-adjacent chain piece
        (conveyor / bridge / splitter / armoured_conveyor) and try to
        reconnect them to the existing chain network.

        Cascade:
          1. Conveyor — pick a cardinal of the harvester that's empty
             or holds an allied road. If a build there can output
             (cardinal-adjacent) into an existing chain piece, place
             a conveyor facing that direction.
          2. Bridge — same set of placement tiles. Bridge target must
             be an existing chain piece within BRIDGE_TARGET_RADIUS_SQ.
             Closest target wins on ties.
          3. Destroy — last resort. Frees the tile so future economy
             work can claim the ore differently.
        """
        tc = self.tile_cache
        my_team = self.my_team_cache
        core_pos = self.core_pos
        _CHAIN_TYPES = (
            EntityType.CONVEYOR, EntityType.BRIDGE,
            EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR,
        )
        _CARDINAL_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))

        # Find closest orphaned harvester in vision.
        orphan = None
        orphan_dsq = 999999
        for xy in self._last_vision_set:
            info = tc.get(xy)
            if info is None:
                continue
            bid, etype, team = info[1], info[2], info[3]
            if (bid is None or team != my_team
                    or etype != EntityType.HARVESTER):
                continue
            has_feeder = False
            for dx, dy in _CARDINAL_DELTAS:
                ne = tc.get((xy[0] + dx, xy[1] + dy))
                if (ne is not None and ne[1] is not None
                        and ne[3] == my_team and ne[2] in _CHAIN_TYPES):
                    has_feeder = True
                    break
            if has_feeder:
                continue
            if (core_pos is not None
                    and euclidean_dist_sq(xy, core_pos) > _PATROL_MAX_DSQ):
                continue
            dsq = euclidean_dist_sq(my_xy, xy)
            if dsq < orphan_dsq:
                orphan_dsq = dsq
                orphan = xy
        if orphan is None:
            if self.patrol_action == 'repair_orphan':
                self.patrol_action = None
                self.path = None
                self.path_index = 0
            return False

        self.patrol_target_xy = orphan
        # Walk into action range of the harvester first.
        if orphan_dsq > GC.ACTION_RADIUS_SQ:
            if (self.patrol_action != 'repair_orphan'
                    or self.path is None or self.path_index >= len(self.path)):
                self.patrol_action = 'repair_orphan'
                goal = self._patrol_find_adjacent(my_xy, orphan)
                if goal is None:
                    return False
                # In-vision target — tight A* cap.
                self.path = self._compute_path(my_xy, goal, max_nodes=100)
                self.path_index = 0
                if self.path is None:
                    return False
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
                return False
            return True

        # In range. Pick a placement: conveyor first, then bridge.
        place_xy = None
        place_kind = None       # 'conveyor' | 'bridge' | 'destroy'
        place_arg = None        # Direction for conveyor, target xy for bridge

        # Pass 1 — conveyor that targets an existing chain piece.
        for dx, dy in _CARDINAL_DELTAS:
            cand = (orphan[0] + dx, orphan[1] + dy)
            pe = tc.get(cand)
            if pe is None:
                continue
            env, p_bid, p_etype, p_team = pe
            is_empty = (p_bid is None and env != Environment.WALL
                        and env not in (Environment.ORE_TITANIUM,
                                        Environment.ORE_AXIONITE))
            is_road = (p_bid is not None and p_etype == EntityType.ROAD
                       and p_team == my_team)
            if not (is_empty or is_road):
                continue
            for out_dx, out_dy in _CARDINAL_DELTAS:
                if (out_dx, out_dy) == (-dx, -dy):
                    continue  # would point back at the harvester
                out_xy = (cand[0] + out_dx, cand[1] + out_dy)
                oe = tc.get(out_xy)
                if (oe is not None and oe[1] is not None
                        and oe[3] == my_team and oe[2] in _CHAIN_TYPES):
                    place_xy = cand
                    place_kind = 'conveyor'
                    place_arg = cardinal_direction_between(cand, out_xy)
                    break
            if place_xy is not None:
                break

        # Pass 2 — bridge with closest-by-Euclidean chain target.
        if place_xy is None:
            best_dsq = None
            for dx, dy in _CARDINAL_DELTAS:
                cand = (orphan[0] + dx, orphan[1] + dy)
                pe = tc.get(cand)
                if pe is None:
                    continue
                env, p_bid, p_etype, p_team = pe
                is_empty = (p_bid is None and env != Environment.WALL
                            and env not in (Environment.ORE_TITANIUM,
                                            Environment.ORE_AXIONITE))
                is_road = (p_bid is not None and p_etype == EntityType.ROAD
                           and p_team == my_team)
                if not (is_empty or is_road):
                    continue
                for tdx in (-3, -2, -1, 0, 1, 2, 3):
                    for tdy in (-3, -2, -1, 0, 1, 2, 3):
                        tdsq = tdx * tdx + tdy * tdy
                        if tdsq == 0 or tdsq > GC.BRIDGE_TARGET_RADIUS_SQ:
                            continue
                        target_xy = (cand[0] + tdx, cand[1] + tdy)
                        te = tc.get(target_xy)
                        if (te is None or te[1] is None
                                or te[3] != my_team
                                or te[2] not in _CHAIN_TYPES):
                            continue
                        if best_dsq is None or tdsq < best_dsq:
                            best_dsq = tdsq
                            place_xy = cand
                            place_kind = 'bridge'
                            place_arg = target_xy

        # Pass 3 — destroy the harvester. Same PATROL_GAP_WAIT_ROUNDS
        # window as P4's chain-gap repair so an in-flight economy
        # rebuild has a chance, but the destroy is intentional even on
        # our own harvester: leaving an orphaned harvester on an ore
        # tile lets the enemy land a sentinel cardinally adjacent to
        # it and fuel that sentinel from the harvester's output. Better
        # to lose the harvester than feed an enemy turret.
        if place_xy is None:
            current_round = ct.get_current_round()
            first_seen = self._patrol_orphan_watch.get(orphan)
            if first_seen is None:
                self._patrol_orphan_watch[orphan] = current_round
                return True
            if current_round - first_seen < PATROL_GAP_WAIT_ROUNDS:
                return True
            place_xy = orphan
            place_kind = 'destroy'
        # Resolved by a build below — drop the watch entry so a future
        # orphan cycle on the same tile starts a fresh timer.
        elif orphan in self._patrol_orphan_watch:
            del self._patrol_orphan_watch[orphan]

        if ct.get_action_cooldown() != 0:
            return True

        # Destroy allied road on placement tile first (free, doesn't
        # consume the action cooldown — same turn build works).
        if place_kind != 'destroy':
            pe = tc.get(place_xy)
            if (pe is not None and pe[1] is not None
                    and pe[2] == EntityType.ROAD and pe[3] == my_team):
                rpos = xy_to_pos(place_xy)
                if ct.can_destroy(rpos):
                    ct.destroy(rpos)
                    tc[place_xy] = (pe[0], None, None, None)

        ppos = xy_to_pos(place_xy)
        if place_kind == 'conveyor':
            if (self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0])
                    and ct.can_build_conveyor(ppos, place_arg)):
                ct.build_conveyor(ppos, place_arg)
                if DEBUG:
                    print(f"[{self.corner}] orphan-fix conveyor@{place_xy} "
                          f"dir={place_arg.value} for harv@{orphan}",
                          file=sys.stderr)
        elif place_kind == 'bridge':
            tpos = xy_to_pos(place_arg)
            if (self._has_economy_reserve(ct, GC.BRIDGE_BASE_COST[0])
                    and ct.can_build_bridge(ppos, tpos)):
                ct.build_bridge(ppos, tpos)
                if DEBUG:
                    print(f"[{self.corner}] orphan-fix bridge@{place_xy}"
                          f"→{place_arg} for harv@{orphan}",
                          file=sys.stderr)
        else:  # destroy
            if ct.can_destroy(ppos):
                ct.destroy(ppos)
                tc[orphan] = (Environment.EMPTY, None, None, None)
                if DEBUG:
                    print(f"[{self.corner}] orphan-fix destroy harv@{orphan} "
                          f"(no feeder placement possible)",
                          file=sys.stderr)
        return True

    # ------------------------------------------------------------------ #
    #  Priority *: Armour the core-adjacent ring                          #
    # ------------------------------------------------------------------ #

    def _patrol_priority_armour_core_ring(self, ct, my_xy):
        """Destroy + replace allied CONVEYORs in the 12-tile core ring
        with ARMOURED_CONVEYORs, preserving direction. Destroy and
        build happen in the same turn so the tile never sits empty
        (enemies can't squat on it). Walks to the nearest upgrade
        candidate when not already in action range.

        Returns True if this priority consumed the turn."""
        if self.core_pos is None:
            return False

        # Resource check: 5 Ti + 5 Ax (scaled), plus the normal MIN_TITANIUM
        # reserve floor so we don't starve bot/building spawning.
        ti, ax = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0
        arm_ti = int(GC.ARMOURED_CONVEYOR_BASE_COST[0] * scale)
        arm_ax = int(GC.ARMOURED_CONVEYOR_BASE_COST[1] * scale)
        floor = int(MIN_TITANIUM * scale)
        if ti < arm_ti + floor or ax < arm_ax:
            return False

        # Find upgrade candidates on the core ring.
        targets = []
        for xy in self._core_adjacent_tiles():
            e = self.tile_cache.get(xy)
            if e is None or e[1] is None:
                continue
            if e[3] != self.my_team_cache:
                continue
            if e[2] != EntityType.CONVEYOR:
                continue
            targets.append(xy)
        if not targets:
            return False

        # Pick closest candidate.
        targets.sort(key=lambda xy: euclidean_dist_sq(my_xy, xy))
        target = targets[0]

        # If out of action range, walk toward it.
        if euclidean_dist_sq(my_xy, target) > GC.ACTION_RADIUS_SQ:
            if self.patrol_target_xy != target:
                self.path = None
                self.path_index = 0
                self.patrol_target_xy = target
            if self.path is None or self.path_index >= len(self.path):
                # Core ring is at most 1 tile from a 3x3 core — short
                # path. Tight A* cap so a momentary block doesn't burn
                # the budget.
                self.path = self._compute_path(my_xy, target, max_nodes=100)
                self.path_index = 0
                if self.path is None:
                    # Couldn't pathfind — let lower priorities (heal /
                    # destroy / repair / circle) run instead of standing
                    # still here.
                    return False
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
                return False
            return True

        # In range — destroy + rebuild as armoured on the same turn.
        # If cooldown isn't ready, let lower-priority actions (heal /
        # destroy / circle) run this turn instead of idling. The
        # candidate will still be on the ring next turn for us to
        # come back and upgrade once cooldown clears.
        if ct.get_action_cooldown() > 0:
            return False
        tpos = xy_to_pos(target)
        te = self.tile_cache.get(target)
        if te is None or te[1] is None:
            return False
        bc = self.building_cache.get(target)
        direction = bc[1] if bc else None
        if direction is None:
            try:
                direction = ct.get_direction(te[1])
            except Exception:
                return False
        if not ct.can_destroy(tpos):
            return True
        ct.destroy(tpos)
        self.tile_cache[target] = (te[0], None, None, None)
        self.building_cache.pop(target, None)
        if ct.can_build_armoured_conveyor(tpos, direction):
            ct.build_armoured_conveyor(tpos, direction)
            self.tile_cache[target] = (
                te[0], -1, EntityType.ARMOURED_CONVEYOR, self.my_team_cache,
            )
            self.building_cache[target] = (EntityType.ARMOURED_CONVEYOR, direction)
            print(f"[{self.corner}] armour ring ({target[0]},{target[1]}) "
                  f"dir={direction.value}")
        self.patrol_target_xy = target
        return True

    # ------------------------------------------------------------------ #
    #  Priority 4b: Opportunistic armour upgrade (action-range only)      #
    # ------------------------------------------------------------------ #

    def _patrol_priority_armour_opportunistic(self, ct, my_xy):
        """If we sit next to an allied CONVEYOR and team axionite is
        above MIN_AXIONITE_FOR_ARMOURED, destroy + rebuild it as an
        ARMOURED_CONVEYOR. Action-range only — no pathfinding. Runs
        just above the circling step so regular patrols naturally
        armour the economy over time without starving higher-priority
        actions."""
        ti, ax = ct.get_global_resources()
        if ax <= MIN_AXIONITE_FOR_ARMOURED:
            return False
        scale = ct.get_scale_percent() / 100.0
        arm_ti = int(GC.ARMOURED_CONVEYOR_BASE_COST[0] * scale)
        arm_ax = int(GC.ARMOURED_CONVEYOR_BASE_COST[1] * scale)
        floor = int(MIN_TITANIUM * scale)
        if ti < arm_ti + floor or ax < arm_ax:
            return False
        if ct.get_action_cooldown() > 0:
            return False

        # Scan 5x5 around us (ACTION_RADIUS_SQ = 8 → radius ~2.83).
        target = None
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > GC.ACTION_RADIUS_SQ:
                    continue
                xy = (my_xy[0] + dx, my_xy[1] + dy)
                e = self.tile_cache.get(xy)
                if e is None or e[1] is None:
                    continue
                if e[3] != self.my_team_cache:
                    continue
                if e[2] != EntityType.CONVEYOR:
                    continue
                target = xy
                break
            if target is not None:
                break
        if target is None:
            return False

        tpos = xy_to_pos(target)
        te = self.tile_cache.get(target)
        bc = self.building_cache.get(target)
        direction = bc[1] if bc else None
        if direction is None:
            try:
                direction = ct.get_direction(te[1])
            except Exception:
                return False
        if not ct.can_destroy(tpos):
            return False
        ct.destroy(tpos)
        self.tile_cache[target] = (te[0], None, None, None)
        self.building_cache.pop(target, None)
        if ct.can_build_armoured_conveyor(tpos, direction):
            ct.build_armoured_conveyor(tpos, direction)
            self.tile_cache[target] = (
                te[0], -1, EntityType.ARMOURED_CONVEYOR, self.my_team_cache,
            )
            self.building_cache[target] = (EntityType.ARMOURED_CONVEYOR, direction)
            print(f"[{self.corner}] armour upg ({target[0]},{target[1]}) "
                  f"dir={direction.value}")
        self.patrol_target_xy = target
        return True

    # ------------------------------------------------------------------ #
    #  Priority 5: Default patrol — circle core at radius 6               #
    # ------------------------------------------------------------------ #

    def _patrol_priority_5_circle(self, ct, my_xy):
        """Walk a circle of radius PATROL_RADIUS clockwise around the core.

        Behavior:
        - Precompute waypoints at radius R around the core (one-time per bot)
        - On spawn / after returning from a task: pathfind back to core center,
          then pick a random starting waypoint
        - Each turn, advance the waypoint clockwise. The bot makes one move
          (8-directional) that brings it closest to the current waypoint,
          building a road if needed.
        """
        if self.core_pos is None:
            return

        cx, cy = self.core_pos
        dx_c = my_xy[0] - cx
        dy_c = my_xy[1] - cy
        within_max = (dx_c * dx_c + dy_c * dy_c) <= _PATROL_MAX_DSQ

        # One-time waypoint init (per bot)
        if self._patrol_waypoints is None:
            self._patrol_waypoints = self._build_patrol_waypoints()
            if not self._patrol_waypoints:
                return
            # Start circling IMMEDIATELY from the nearest waypoint so the
            # bot's vision starts expanding outward on turn 1. Previously
            # we forced a "return to core center" detour here which cost
            # 1-2 turns where the bot sat in the core area with no vision
            # expansion — meaning P1-P4 priorities had no chance to see
            # targets. Higher priorities still run each turn and preempt
            # this circle when a target comes into view.
            self.patrol_waypoint_idx = self._patrol_nearest_waypoint_idx(my_xy)
            self.patrol_action = 'circle'
            self.path = None
            self.path_index = 0
            self._patrol_last_clamped = None
            self._patrol_clamp_count = 0

        # Transitioning into circle from another (non-circle, non-return) action.
        # Only force return-to-core if we're outside PATROL_MAX_DISTANCE.
        # Otherwise resume circling immediately from the current position.
        if (self.patrol_action != 'circle'
                and self.patrol_action != 'return_to_core'):
            if within_max:
                self.patrol_action = 'circle'
                self.patrol_waypoint_idx = self._patrol_nearest_waypoint_idx(my_xy)
                self.path = None
                self.path_index = 0
                self._patrol_last_clamped = None
                self._patrol_clamp_count = 0
            else:
                self.patrol_action = 'return_to_core'
                self.path = None
                self.path_index = 0

        # --- State: returning to core ---
        if self.patrol_action == 'return_to_core':
            if my_xy == self.core_pos:
                # Arrived — pick a random starting waypoint and start circling
                self.patrol_waypoint_idx = self._patrol_random_index()
                self.patrol_action = 'circle'
                self.path = None
                self.path_index = 0
            else:
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(my_xy, self.core_pos)
                    self.path_index = 0
                if self.path:
                    result = self._follow_path(ct)
                    if result == 'blocked':
                        self.path = None
                        self.path_index = 0
                return

        # --- State: circle ---
        # Advance waypoint clockwise each turn (skip an extra if clamped target
        # has been the same for >2 turns — likely stuck against map edge)
        n = len(self._patrol_waypoints)
        advance = 2 if self._patrol_clamp_count > 2 else 1
        self.patrol_waypoint_idx = (self.patrol_waypoint_idx + advance) % n
        if advance > 1:
            self._patrol_clamp_count = 0
        raw_target = self._patrol_waypoints[self.patrol_waypoint_idx]

        # Clamp target to nearest in-bounds tile (waypoints may be off-map)
        clamped = self._patrol_clamp_to_bounds(raw_target)

        # Track if clamped target hasn't changed (stuck on map edge)
        if clamped == self._patrol_last_clamped:
            self._patrol_clamp_count += 1
        else:
            self._patrol_clamp_count = 0
            self._patrol_last_clamped = clamped

        # One move toward the (clamped) target
        self._patrol_step_toward(ct, my_xy, clamped)

    def _patrol_radius(self):
        """Dynamic patrol radius scaled by map size.

        20x20 map -> radius 3, 50x50 map -> radius 6. Linearly interpolated
        based on max(map_w, map_h). Falls back to PATROL_RADIUS if map
        dimensions are unknown.
        """
        if self.map_w is None or self.map_h is None:
            return PATROL_RADIUS
        size = max(self.map_w, self.map_h)
        # Linear: 20 -> 3, 50 -> 6; clamped to [3, 6]
        r = 3 + (size - 20) * (6 - 3) / (50 - 20)
        return max(3, min(6, int(round(r))))

    def _build_patrol_waypoints(self):
        """Precompute enough waypoints around the core to cover a full circle
        at the dynamic patrol radius with ~1-tile spacing. Out-of-bounds
        waypoints are KEPT — the bot clamps to in-bounds when stepping
        toward them. Consecutive duplicates are dedup'd to avoid stalling.
        """
        cx, cy = self.core_pos
        R = self._patrol_radius()
        # Circumference ≈ 2πR — pick n so each step is ~1 tile
        n = max(8, int(math.ceil(2 * math.pi * R)))
        waypoints = []
        for i in range(n):
            a = (2 * math.pi * i) / n
            tx = cx + int(round(R * math.cos(a)))
            ty = cy + int(round(R * math.sin(a)))
            xy = (tx, ty)
            if waypoints and waypoints[-1] == xy:
                continue
            waypoints.append(xy)
        # Dedup wraparound
        if len(waypoints) > 1 and waypoints[0] == waypoints[-1]:
            waypoints.pop()
        return waypoints

    def _patrol_clamp_to_bounds(self, xy):
        """Clamp (x,y) to the nearest in-bounds tile."""
        x, y = xy
        if self.map_w is None:
            return xy
        cx = max(0, min(self.map_w - 1, x))
        cy = max(0, min(self.map_h - 1, y))
        return (cx, cy)

    def _patrol_nearest_waypoint_idx(self, xy):
        """Return the index of the waypoint closest to xy (Euclidean)."""
        best_i = 0
        best_dsq = 999999
        for i, wp in enumerate(self._patrol_waypoints):
            d = (wp[0] - xy[0]) * (wp[0] - xy[0]) + (wp[1] - xy[1]) * (wp[1] - xy[1])
            if d < best_dsq:
                best_dsq = d
                best_i = i
        return best_i

    def _patrol_random_index(self):
        """Pick a random starting waypoint index. Uses bot id-derived seed."""
        # Use self id + map dimensions for variety without import random module
        # (random is in stdlib, but tiny optimisation: use hash instead)
        h = id(self) ^ ((self.map_w or 0) * 31 + (self.map_h or 0))
        return h % len(self._patrol_waypoints)

    def _patrol_step_toward(self, ct, my_xy, target_xy):
        """Make one 8-directional move that minimises Euclidean distance to target_xy.

        Builds a road first if the chosen tile is empty/ore (and an action is
        available). Skips walls, known_walls, non-walkable buildings, and
        bot-occupied tiles.
        """
        if my_xy == target_xy:
            return

        tc = self.tile_cache
        tx, ty = target_xy
        mx, my = my_xy

        # Score each of 8 neighbors by distance to target
        candidates = []
        for d in _ALL_DIRS_8:
            ddx, ddy = DIRECTION_DELTAS[d]
            nx, ny = mx + ddx, my + ddy
            nxy = (nx, ny)
            if nxy in self.known_walls:
                continue
            if (self.map_w is not None and
                    (nx < 0 or ny < 0 or nx >= self.map_w or ny >= self.map_h)):
                continue
            entry = tc.get(nxy)
            if entry and entry[0] == Environment.WALL:
                continue
            # Skip non-walkable buildings — markers included, since the
            # bot can't step onto a marker tile. (To route through a marker
            # we'd have to build a road over it first; _follow_path in
            # movement.py handles that on A* paths, but not in the
            # patrol step-toward loop.)
            if entry and entry[1] is not None and entry[2] not in _WALKABLE_BUILDINGS:
                continue
            # Skip tiles occupied by another bot
            if nxy in self.bot_pos_cache and nxy != my_xy:
                continue
            ddist = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty)
            candidates.append((ddist, d, nxy, entry))

        if not candidates:
            return

        candidates.sort(key=lambda c: c[0])

        for ddist, d, nxy, entry in candidates:
            if ct.can_move(d):
                ct.move(d)
                return
            # Can't move — try building a road on this tile if empty/ore
            if (entry is None or entry[1] is None) and ct.get_action_cooldown() == 0:
                target_pos = xy_to_pos(nxy)
                if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                        and ct.can_build_road(target_pos)):
                    ct.build_road(target_pos)
                    if ct.can_move(d):
                        ct.move(d)
                    return

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _enemy_bot_within(self, target_xy, radius):
        """True iff any enemy builder bot in vision is within `radius`
        Chebyshev tiles of `target_xy`. bot_pos_cache is keyed by xy
        with values (uid, team) and only holds bots in vision, so
        iteration is bounded (~10 entries typical)."""
        my_team = self.my_team_cache
        tx, ty = target_xy
        for (bx, by), (_uid, team) in self.bot_pos_cache.items():
            if team == my_team or team is None:
                continue
            if max(abs(bx - tx), abs(by - ty)) <= radius:
                return True
        return False

    def _patrol_find_adjacent(self, my_xy, target_xy):
        """Find a reachable 8-neighbor tile of target_xy, preferring closest to my_xy."""
        tc = self.tile_cache
        best = None
        best_dsq = 999999
        tx, ty = target_xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (tx + dx, ty + dy)
                if nxy in self.known_walls:
                    continue
                entry = tc.get(nxy)
                if entry and entry[0] == Environment.WALL:
                    continue
                dsq = euclidean_dist_sq(my_xy, nxy)
                if dsq < best_dsq:
                    best_dsq = dsq
                    best = nxy
        return best
