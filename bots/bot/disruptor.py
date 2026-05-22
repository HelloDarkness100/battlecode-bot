"""Disruptor builder bot — pathfinds to the enemy core and harasses on the way.

Scope now includes Step 3 (intercept enemy titanium harvesters). Steps 4-10
remain stubbed for Parts 2 (b/c) and 3.
"""
import heapq
import math
import sys
from cambc import EntityType, Environment, GameConstants as GC
from constants import (
    DEBUG, SELF_DESTRUCT_DISRUPTOR_TURNS, ECONOMY_TEST_MODE,
    INTERCEPT_DETOUR, IGNORE_ENEMY_CONVEYOR_THRESHOLD,
    LAUNCHER_PROTOCOL_PREFIX, HEALED_DETECTION_TURNS,
    HARASS_RADIUS, DISRUPTOR_CIRCLE_RADIUS, DISRUPT_INTERVAL,
    DISRUPTOR_DANGER_HP_FRAC, DISRUPTOR_DANGER_HEURISTIC_PENALTY,
)
from utils import (
    mirror_xy, xy_to_pos, direction_between, cardinal_direction_between,
    euclidean_dist_sq, DIRECTION_DELTAS,
)
from cambc import Direction
from pathfinding import _WALKABLE_BUILDINGS


# 8 movement directions in fixed order (excludes CENTRE).
_BUGNAV_DIRS = [
    Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
    Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST,
]

# Tuple offsets for the 8 neighbors — used by greedy best-first.
_NEIGHBOR_DELTAS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

# Safety cap on nodes expanded per greedy pathfind. ~2000 is plenty for
# a 50x50 map; each node is ~20 dict/heap ops, so ~400μs worst case.
_GREEDY_MAX_NODES = 2000


_SYM_ORDER = ['diag', 'vert', 'horiz']

# Cardinal offsets for the "feeder must be cardinally adjacent" scan.
_CARDINAL_OFFSETS = ((0, -1), (1, 0), (0, 1), (-1, 0))

# Turret types that can shoot us. LAUNCHER is excluded — it displaces
# bots but doesn't deal damage, so a launcher picking us up wouldn't
# register as turret damage.
_ENEMY_TURRET_TYPES = frozenset({
    EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH,
})


class DisruptorMixin:
    # ------------------------------------------------------------------ #
    #  Replay / console logging helpers                                   #
    # ------------------------------------------------------------------ #
    #  stdout → captured into the replay, viewable per-unit in the
    #           visualiser. Always on.
    #  stderr → console only, gated on DEBUG.
    # ------------------------------------------------------------------ #

    def _ds_event(self, msg):
        """Emit a disruptor action/event to replay (stdout) + console (stderr)."""
        print(msg)
        if DEBUG:
            print(msg, file=sys.stderr)

    def _print_disruptor_status(self, ct):
        """Per-turn compact status line, sent to stdout so the replay
        visualiser can show the bot's current action every turn."""
        my_xy = self._my_xy
        step = self.disruptor_step
        parts = [f"DS ({my_xy[0]},{my_xy[1]}) step={step}"]

        if step == 2:
            target = self.enemy_core_pos or self.enemy_core_guess
            parts.append(f"pathfind->{target}")
            parts.append(f"s2t={self.disruptor_step2_turns}")
        elif step == 3:
            case = "clean" if self.intercept_clean_case else "dirty"
            parts.append(f"intercept sub={self.intercept_substep} case={case}")
            if self.intercept_target_harvester is not None:
                parts.append(f"harv={self.intercept_target_harvester}")
            if self.intercept_target_conveyor is not None:
                parts.append(f"feeder={self.intercept_target_conveyor}")
            if self.intercept_sentinel_tile is not None:
                parts.append(f"stile={self.intercept_sentinel_tile}")
        elif step == 5:
            parts.append("HARASS")
            if self.disrupt_target is not None:
                tag = "post-destroy" if self.disrupt_target_destroyed else "attacking"
                parts.append(f"disrupt={self.disrupt_target} ({tag})")
            else:
                parts.append(f"ctr={self.disrupt_counter}/{DISRUPT_INTERVAL}")
            if self.step10_wander_target is not None:
                parts.append(f"wander={self.step10_wander_target}")
        elif step == 8:
            parts.append(f"launcher sub={self.launcher_place_substep}")
            if self.launcher_place_target is not None:
                parts.append(f"tgt={self.launcher_place_target}")
            if self.launcher_place_road_type is not None:
                parts.append(f"road={self.launcher_place_road_type}")

        # Cross-step overlays
        if self.case_b_active:
            parts.append(f"CASE_B:{self.case_b_phase}")
            if self.case_b_target is not None:
                parts.append(f"cbtgt={self.case_b_target}")
        if self.launch_protocol_step > 0 and step != 9:
            parts.append(f"proto={self.launch_protocol_step}")
            if self.launch_target_xy is not None:
                parts.append(f"ltgt={self.launch_target_xy}")
        if self.launched_this_turn:
            parts.append("LAUNCHED")
        if self._acted_on_land:
            parts.append("ACTED_ON_LAND")

        ti, _ = ct.get_global_resources()
        parts.append(f"ti={ti}")
        if self.blocked_launcher_tiles:
            parts.append(f"blocked={len(self.blocked_launcher_tiles)}")
        print(" ".join(parts))

    def _run_disruptor(self, ct):
        if ECONOMY_TEST_MODE:
            return

        my_xy = self._my_xy
        self.launched_this_turn = False

        # --- Enemy launcher detection (Part 3) ---
        # If our start-of-turn position doesn't match the position we ended
        # last turn on, something moved us in-between. Check prev_intended_pos's
        # 8-neighborhood for an enemy LAUNCHER and, if found, mark the area
        # as permanently blocked, try attack-on-land / build-on-land, and
        # enter Case B if we were on an attack/build task we can't finish
        # on-land.
        if self.prev_intended_pos is not None and my_xy != self.prev_intended_pos:
            launcher_xy = self._detect_enemy_launcher_near(self.prev_intended_pos)
            if launcher_xy is not None:
                self._handle_enemy_launch(ct, launcher_xy)
                # If we already fired / built on-land, we've spent this
                # turn's action cooldown — skip the normal step logic so we
                # don't trip assertions.
                if self._acted_on_land:
                    return

        # Turret-damage response: if our HP dropped since the last
        # turn and an enemy turret is in vision, treat it as the
        # shooter. Abandon the current disrupt target (a conveyor /
        # bridge we're firing at is not worth burning HP against a
        # turret that can keep shooting forever) and add every tile
        # the turret can reach to `disrupt_attack_blacklist` so
        # future attack pickers (disruption / ore-block / chain-
        # extension / intercept) skip that zone.
        self._disruptor_turret_damage_check(ct, my_xy)

        # Refresh the harvester-fed enemy sentinel danger zone for
        # this turn. Drives both the greedy passable check (impassable
        # below the HP threshold, heuristic penalty above) and the
        # flee override below. Recomputed every turn so a sentinel
        # that gets killed or rotates immediately stops blocking us.
        self._disruptor_recompute_danger_zone(ct)
        # Cache the HP gate once per turn so the per-tile greedy
        # passable check doesn't re-pay the get_hp FFI on every node.
        self._disruptor_low_hp_cached = self._disruptor_below_danger_hp(ct)

        # Top-priority safety: if we're inside the danger zone AND
        # below DISRUPTOR_DANGER_HP_FRAC HP, drop everything and walk
        # out. The end-of-turn self-heal in main.py runs unconditionally
        # for damaged disruptors, so we heal as we move.
        if (self._disruptor_below_danger_hp(ct)
                and self._disruptor_flee_danger_zone(ct, my_xy)):
            return

        # Case B counter hijacks normal step dispatch until it completes.
        if self.case_b_active:
            self._run_case_b_counter(ct)
            return

        # Step 1 runs exactly once per bot. Fall through so we can also
        # walk during the same turn.
        if self.disruptor_step == 1:
            self._disruptor_step1_init()

        # If scanning.py cycled self._symmetry (it just invalidated a
        # symmetry via a terrain mismatch), treat that as evidence the
        # disruptor's current guess is also wrong. Mark the PRIOR scanning
        # sym as tried and — if our current guess matches it — advance to
        # the next-best symmetry from the distance-sorted order.
        if self._disruptor_last_scanning_sym != self._symmetry:
            prev = self._disruptor_last_scanning_sym
            self._disruptor_last_scanning_sym = self._symmetry
            if prev is not None and prev in _SYM_ORDER:
                self._sym_tried_disruptor.add(prev)
            if self.disruptor_sym_when_guessed in self._sym_tried_disruptor:
                self._disruptor_apply_next_symmetry()

        # Env-mismatch validation against the DISRUPTOR's own symmetry
        # (which may differ from scanning.py's symmetry during the
        # distance-sorted rotation). Every real observation whose mirror
        # under `disruptor_sym_when_guessed` is ALSO a real observation
        # is a free consistency check: walls mirror to walls, ore mirrors
        # to the same ore type, empty to empty. A single mismatch proves
        # the current guess is wrong — rotate before we walk any further.
        if self.enemy_core_pos is None:
            self._disruptor_check_sym_env()

        # Prefer the real enemy core position (scanning.py sets it when seen).
        target_xy = self.enemy_core_pos or self.enemy_core_guess

        # Drive an in-progress ore-block target across turns. Only fires
        # in Step 2 and Step 5 — Step 3 has its own substep dispatch, and
        # Case-B / launch-protocol ran above.
        if (self.disrupt_ore_block_target is not None
                and self.disruptor_step in (2, 5)):
            result = self._disruptor_block_ore(ct, self.disrupt_ore_block_target)
            if result == 'in_progress':
                return
            self.disrupt_ore_block_target = None
            if result == 'done':
                return
            # 'abandon' falls through to the normal step dispatch.

        # Drive an in-progress chain-extension intercept across turns.
        if (self.disrupt_chain_block_target is not None
                and self.disruptor_step in (2, 5)):
            result = self._disruptor_block_chain_extension(
                ct, self.disrupt_chain_block_target,
                self.disrupt_chain_block_source)
            if result == 'in_progress':
                return
            # Blacklist on both done and abandon so the picker doesn't
            # immediately re-select the same target (especially after a
            # heal-detection abandon on a constantly-healed enemy road).
            if self.disrupt_chain_block_target is not None:
                self.disrupt_blacklist.add(self.disrupt_chain_block_target)
            self.disrupt_chain_block_target = None
            self.disrupt_chain_block_source = None
            if result == 'done':
                return
            # 'abandon' falls through to the normal step dispatch.

        # Drive an in-progress splitter-sentinel placement across turns.
        if (self.disrupt_splitter_target is not None
                and self.disruptor_step in (2, 5)):
            placement_xy, splitter_xy = self.disrupt_splitter_target
            result = self._disruptor_block_splitter_sentinel(
                ct, placement_xy, splitter_xy)
            if result == 'in_progress':
                return
            # Same rationale as chain-extension: blacklist on both done
            # and abandon so a healed enemy road on the splitter approach
            # doesn't keep getting re-picked.
            self.disrupt_blacklist.add(placement_xy)
            self.disrupt_splitter_target = None
            if result == 'done':
                return

        if self.disruptor_step == 2:
            # Intercept hook: if a reachable harvester is within
            # INTERCEPT_DETOUR path length, jump into Step 3. No round gate
            # — disruptors intercept from turn 1 onward. ONE sentinel per
            # trip: once we've already placed a Step-2 intercept this
            # lifetime, skip the hook and commit to reaching the core.
            pick = None if self._step2_intercepted else self._disruptor_pick_intercept(my_xy)
            if pick is not None:
                (self.intercept_target_harvester,
                 self.intercept_target_conveyor,
                 self.intercept_clean_case,
                 self.intercept_sentinel_tile) = pick
                self.intercept_substep = 0
                self.intercept_sentinel_dir = None
                self.intercept_destroyed_tile = (
                    self.intercept_target_conveyor
                    if not self.intercept_clean_case else None)
                self.intercept_post_step = 2
                self.disruptor_step = 3
                self.path = None
                self.path_index = 0
                self._ds_event(
                    f"[DS] intercept harvester={self.intercept_target_harvester} "
                    f"clean={self.intercept_clean_case} "
                    f"target={self.intercept_sentinel_tile or self.intercept_target_conveyor}")

            if self.disruptor_step == 2:
                # Chain-extension intercept: enemy conveyor / bridge
                # whose output lands on an empty / road tile is about
                # to extend their network. Drop a sentinel on the build
                # site to block (and shoot at) that chain.
                chain_pick = self._disruptor_pick_chain_extension_intercept(
                    my_xy, INTERCEPT_DETOUR)
                if chain_pick is not None:
                    self.disrupt_chain_block_target = chain_pick[0]
                    self.disrupt_chain_block_source = chain_pick[1]
                    self._ds_event(
                        f"[DS] chain-block target {chain_pick[0]} "
                        f"src={chain_pick[1]}")
                    result = self._disruptor_block_chain_extension(
                        ct, chain_pick[0], chain_pick[1])
                    if result != 'abandon':
                        return
                    self.disrupt_chain_block_target = None
                    self.disrupt_chain_block_source = None

                # Splitter-sentinel: place a sentinel adjacent to an
                # enemy splitter (excluding its back tile) — splitter
                # feeds the sentinel ammo, sentinel shoots whatever is
                # in its 7-direction cone.
                split_pick = self._disruptor_pick_splitter_sentinel(
                    my_xy, INTERCEPT_DETOUR)
                if split_pick is not None:
                    self.disrupt_splitter_target = split_pick
                    self._ds_event(
                        f"[DS] splitter-sent target placement={split_pick[0]} "
                        f"split={split_pick[1]}")
                    result = self._disruptor_block_splitter_sentinel(
                        ct, split_pick[0], split_pick[1])
                    if result != 'abandon':
                        return
                    self.disrupt_splitter_target = None

                # Opportunistic ore-block: same detour budget as the
                # harvester intercept above. Bounded by INTERCEPT_DETOUR
                # so we never wander far off the route to the core.
                ore_xy = self._disruptor_pick_ore_block(my_xy, INTERCEPT_DETOUR)
                if ore_xy is not None:
                    self.disrupt_ore_block_target = ore_xy
                    self._ds_event(f"[DS] ore-block target {ore_xy}")
                    if self._disruptor_block_ore(ct, ore_xy) != 'abandon':
                        return
                    self.disrupt_ore_block_target = None

                self.disruptor_step2_turns += 1
                if self.disruptor_step2_turns >= SELF_DESTRUCT_DISRUPTOR_TURNS:
                    self._ds_event(f"[DS] self-destruct after {self.disruptor_step2_turns}t "
                                   f"at ({my_xy[0]},{my_xy[1]})")
                    ct.self_destruct()
                    return
                self._disruptor_step2_pathfind(ct, target_xy)
                return

        if self.disruptor_step == 3:
            # Step 3 turns do NOT count toward self-destruct timer.
            self._disruptor_step3_intercept(ct)
            return

        if self.disruptor_step == 5:
            self._disruptor_step5_harass(ct)
            return

        # Any step not matched above (shouldn't happen) — idle.
        if not self._disruptor_step_stub_logged:
            if DEBUG:
                print(f"[DS] step {self.disruptor_step} unknown; idling",
                      file=sys.stderr)
            self._disruptor_step_stub_logged = True

    # ------------------------------------------------------------------ #
    #  Step 1: initial enemy-core guess from scanning.py's symmetry       #
    # ------------------------------------------------------------------ #

    def _disruptor_step1_init(self):
        # Distance-sorted symmetry priority, closest mirror first. Shortest
        # potential trip to verify → failing guesses abandoned fast.
        self._disruptor_sym_order = self._disruptor_symmetry_order()
        self._sym_tried_disruptor = set()
        # Lock onto scanning's current sym as a baseline so we can detect
        # future scanning-driven invalidations.
        self._disruptor_last_scanning_sym = self._symmetry
        self._disruptor_apply_next_symmetry()
        self.disruptor_step = 2

    def _disruptor_symmetry_order(self):
        """Return `_SYM_ORDER` sorted ASC by squared Euclidean distance
        from our core to the mirrored enemy-core position. The nearest
        mirror is tried first: if it's the right symmetry we reach the
        enemy core fastest, if it's wrong we discover and move on fastest.
        """
        core = self.core_pos

        def mirror_d2(sym):
            mxy = mirror_xy(core, sym, self.map_w, self.map_h)
            dx = core[0] - mxy[0]
            dy = core[1] - mxy[1]
            return dx * dx + dy * dy

        return sorted(_SYM_ORDER, key=mirror_d2)

    def _disruptor_apply_next_symmetry(self):
        """Pick the first untried symmetry from the distance-sorted order
        and set the guess + bookkeeping. Returns True if one was applied,
        False if every candidate has been exhausted.
        """
        for sym in getattr(self, '_disruptor_sym_order', _SYM_ORDER):
            if sym in self._sym_tried_disruptor:
                continue
            self._sym_tried_disruptor.add(sym)
            self.enemy_core_guess = mirror_xy(
                self.core_pos, sym, self.map_w, self.map_h)
            self.disruptor_sym_when_guessed = sym
            self.path = None
            self.path_index = 0
            if DEBUG:
                print(f"[DS] sym={sym} guess={self.enemy_core_guess} "
                      f"(order={self._disruptor_sym_order})",
                      file=sys.stderr)
            return True
        return False

    def _disruptor_recompute_guess(self):
        # Legacy entry point kept for compatibility with the scanning-sym-
        # change hook above. Delegates to the distance-ordered picker.
        if not getattr(self, '_disruptor_sym_order', None):
            self._disruptor_sym_order = self._disruptor_symmetry_order()
        if not hasattr(self, '_sym_tried_disruptor'):
            self._sym_tried_disruptor = set()
        self._disruptor_apply_next_symmetry()

    def _disruptor_check_sym_env(self):
        """Cross-check the disruptor's current symmetry guess against every
        directly-observed tile in current vision. For each observed tile A,
        compute B = mirror(A, disruptor_sym_when_guessed). If B is also a
        direct observation (not a scan-side prediction) and their envs
        disagree, the guess is wrong — rotate to the next symmetry.
        """
        sym = self.disruptor_sym_when_guessed
        if sym is None or self.map_w is None or self.map_h is None:
            return
        mirror_src = getattr(self, '_mirror_src', set())
        tc = self.tile_cache
        mw = self.map_w
        mh = self.map_h
        for xy in self._last_vision_set:
            if xy in mirror_src:
                continue
            info_a = tc.get(xy)
            if info_a is None:
                continue
            mxy = mirror_xy(xy, sym, mw, mh)
            if mxy == xy or mxy in mirror_src:
                continue
            info_b = tc.get(mxy)
            if info_b is None:
                continue
            if info_a[0] != info_b[0]:
                if DEBUG:
                    print(f"[DS] sym={sym} env mismatch "
                          f"{xy}={info_a[0]} vs {mxy}={info_b[0]} → rotate",
                          file=sys.stderr)
                self._disruptor_force_next_symmetry()
                return

    # ------------------------------------------------------------------ #
    #  Step 2: pathfind toward target_xy; self-destruct timer handled above #
    # ------------------------------------------------------------------ #

    def _disruptor_step2_pathfind(self, ct, target_xy):
        my_xy = self._my_xy

        if target_xy is None:
            # Can happen if core_pos / map dims not yet populated.
            self.disruptor_step = 1
            return

        # Harass-radius transition: only fires once we've actually SIGHTED
        # the enemy core. Using enemy_core_guess here is wrong on small
        # maps or when the symmetry guess is off — the bot would "enter
        # harass" halfway through its trip, then wander backwards looking
        # for frontiers inside the guess's HARASS_RADIUS. The guess is for
        # pathfinding only; harass-mode transition needs the real sighting.
        ec = self.enemy_core_pos
        if ec is not None:
            dx = my_xy[0] - ec[0]
            dy = my_xy[1] - ec[1]
            if dx * dx + dy * dy <= HARASS_RADIUS * HARASS_RADIUS:
                self.disruptor_step = 5
                self.disrupt_counter = 0
                self.path = None
                self.path_index = 0
                return

        if self.enemy_core_pos is not None:
            # Known core — aim at the closest ring tile; _disruptor_navigate
            # handles the rest until the harass-radius check above fires.
            goal_xy = self._disruptor_ring_goal(my_xy)
        else:
            # Early invalidation: the moment the guessed core tile lands
            # in our direct vision, check whether an enemy core is on it.
            # If not, the symmetry guess is wrong — rotate NOW rather
            # than walk the remaining tiles. scanning.py would set
            # enemy_core_pos the same turn a core became visible, so
            # falling into this branch means we have line of sight and
            # nothing is there.
            if target_xy in self._last_vision_set:
                info = self.tile_cache.get(target_xy)
                is_enemy_core = (info is not None
                                 and info[2] == EntityType.CORE
                                 and info[3] is not None
                                 and info[3] != self.my_team_cache)
                if not is_enemy_core:
                    if DEBUG:
                        print(f"[DS] guess {target_xy} in vision but no "
                              f"enemy core → rotate symmetry",
                              file=sys.stderr)
                    self._disruptor_force_next_symmetry()
                    return
            goal_xy = target_xy

        if goal_xy is None:
            # No reachable ring tile yet — wait for scan to expand vision.
            return

        # Hybrid: known-only A* inside vision, bugnav into the unknown.
        result = self._disruptor_navigate(ct, goal_xy)
        if result == 'blocked' and self.enemy_core_pos is None:
            # Can't make progress AND we don't know the real enemy core yet
            # — the guess might be under the wrong symmetry. Try the next.
            self._disruptor_force_next_symmetry()

    # ------------------------------------------------------------------ #
    #  Greedy best-first nav, with bugnav as a final safety fallback      #
    # ------------------------------------------------------------------ #
    #  Greedy best-first expands 8-direction neighbors ordered by
    #  squared-Euclidean distance to goal, treating unseen tiles as
    #  passable (optimistic). It returns a full planned path in one pass,
    #  so the bot "sees around" concave walls (like tetris-shaped
    #  obstacles in we_love_tetris). Much better paths than step-by-step
    #  bugnav, and cheap: bounded by _GREEDY_MAX_NODES.
    #
    #  Bugnav is kept only as a last-resort fallback when even the
    #  optimistic greedy returns None (truly walled in).
    # ------------------------------------------------------------------ #

    def _disruptor_navigate(self, ct, target_xy):
        """One navigation step toward target_xy. Returns:
          'arrived'    — my_xy == target_xy
          'moved' / 'built_road' — progress made this turn
          'blocked'    — can't progress; caller should blacklist/abandon
        """
        my_xy = self._my_xy
        if my_xy == target_xy:
            return 'arrived'

        # Cached path still valid? Greedy is optimistic about unknown
        # tiles, so the cached path may run through tiles that have
        # since been confirmed as walls. Walking blindly into them
        # adds 1+ wasted turns before the bot notices and repaths.
        # Re-validate every remaining tile against known_walls each
        # turn — if any is a wall, drop the cache and recompute.
        if self.path and self.path[-1] == target_xy:
            kw = self.known_walls
            stale = False
            for pxy in self.path[self.path_index:]:
                if pxy in kw:
                    stale = True
                    break
            if stale:
                self.path = None
                self.path_index = 0
            else:
                result = self._follow_path(ct)
                if result == 'blocked':
                    self.path = None
                    self.path_index = 0
                else:
                    return result

        # Compute a new greedy best-first path. Optimistic about unknowns,
        # so we get a plan even when the target is beyond current vision.
        path = self._disruptor_greedy_path(my_xy, target_xy)
        if path:
            self.path = path
            self.path_index = 0
            result = self._follow_path(ct)
            if result != 'blocked':
                return result
            self.path = None
            self.path_index = 0

        # Greedy returned None (truly unreachable even optimistically).
        # Bugnav gives us a one-step bail-out in case we're walled in.
        return self._disruptor_bugnav_step(ct, target_xy)

    def _disruptor_greedy_passable(self, xy, target_xy):
        """Passable check for greedy — optimistic about unknowns.

        Excludes walls, blocked launcher footprints, oscillation walls,
        tiles held by other bots, and (unless it IS the target) the
        enemy-core 3x3 footprint. Unknown tiles (not yet seen) are
        assumed passable so the plan can extend beyond current vision.

        Below DISRUPTOR_DANGER_HP_FRAC HP, harvester-fed enemy sentinel
        cones become impassable too — at that HP a single hit can
        finish us, so we route around even if the detour is long.
        Above the threshold the danger zone is still passable here;
        the heuristic penalty in `_disruptor_greedy_path` gives a soft
        push toward safer routes when one exists.
        """
        if xy in self.known_walls:
            return False
        if xy in self.blocked_launcher_tiles:
            return False
        if xy in self.oscillation_walls:
            return False
        if (xy in self.dangerous_sentinel_tiles
                and xy != target_xy
                and self._disruptor_low_hp_cached):
            return False
        if xy != target_xy and xy in self.bot_pos_cache:
            return False
        if self.enemy_core_pos is not None and xy != target_xy:
            ecx, ecy = self.enemy_core_pos
            if abs(xy[0] - ecx) <= 1 and abs(xy[1] - ecy) <= 1:
                return False
        info = self.tile_cache.get(xy)
        if info is None:
            return True                 # optimistic: unseen = passable
        env, bid, etype, team = info
        if env == Environment.WALL:
            return False
        if bid is None:
            return True                 # empty / ore land — road-able
        if etype in _WALKABLE_BUILDINGS:
            return True
        # Non-walkable allied/enemy building: only passable as a target
        # (for attack-on-tile). Otherwise route around.
        return xy == target_xy

    def _disruptor_greedy_path(self, start, target_xy, max_nodes=None):
        """Greedy best-first 8-dir from start to target_xy. Heuristic is
        squared-Euclidean to the target. Unknowns are passable (optimistic).
        Returns list[(x,y)] excluding start, including target, or None.

        `max_nodes` overrides the default `_GREEDY_MAX_NODES` cap. Picker
        callers pass a smaller budget (typically a few hundred) so a
        single failing candidate can't burn 1.6 ms on a 2000-node
        unreachable search; navigation callers leave it at the default
        because they need the full budget to route around real obstacles.
        """
        if start == target_xy:
            return []
        tx, ty = target_xy
        mw = self.map_w
        mh = self.map_h
        cap = _GREEDY_MAX_NODES if max_nodes is None else max_nodes
        # Dynamic CPU budget: scale `cap` down by remaining turn time so
        # a navigation greedy invoked late in a busy turn (e.g. right
        # after an enemy-launcher response) can't burn the rest of the
        # 2 ms budget on a 2000-node search. ~0.8μs/node, leave 200μs
        # for whatever runs after this. Bot has been TLEing on ladder
        # after being launched — the post-launch repath blew the budget.
        ct = getattr(self, '_ct', None)
        if ct is not None:
            elapsed = ct.get_cpu_time_elapsed()
            remaining = 1800 - elapsed
            cap = min(cap, max(50, int(remaining / 0.8)))
            if cap < 50:
                return None

        # Heuristic = squared Euclidean to target, plus a penalty for
        # tiles in the harvester-fed enemy sentinel danger zone. The
        # penalty steers greedy expansion toward safe tiles when the
        # detour is cheap; danger tiles are still passable above the
        # HP threshold (the passable check makes them impassable below
        # it). Cache locally so the closure doesn't read self each call.
        danger = self.dangerous_sentinel_tiles
        penalty = DISRUPTOR_DANGER_HEURISTIC_PENALTY

        def h(xy):
            dx = xy[0] - tx
            dy = xy[1] - ty
            d2 = dx * dx + dy * dy
            if xy in danger:
                d2 += penalty
            return d2

        counter = 0
        open_heap = [(h(start), counter, start)]
        parent = {start: None}
        visited = {start}
        expanded = 0

        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur == target_xy:
                path = []
                node = cur
                while parent[node] is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            expanded += 1
            if expanded > cap:
                return None
            cx, cy = cur
            for dx, dy in _NEIGHBOR_DELTAS_8:
                nxy = (cx + dx, cy + dy)
                if nxy in visited:
                    continue
                if mw is not None:
                    if nxy[0] < 0 or nxy[0] >= mw or nxy[1] < 0 or nxy[1] >= mh:
                        continue
                if not self._disruptor_greedy_passable(nxy, target_xy):
                    continue
                visited.add(nxy)
                parent[nxy] = cur
                counter += 1
                heapq.heappush(open_heap, (h(nxy), counter, nxy))
        return None

    # ------------------------------------------------------------------ #
    #  Bug-nav: greedy 8-direction step toward a target. Flat CPU cost.   #
    # ------------------------------------------------------------------ #
    #  Replaces A* navigation for disruptors. A* kept exploring hundreds
    #  of unknown tiles (cost=3 each) when pathfinding across a 50x50
    #  map to the enemy core, blowing the 2ms server budget. Bugnav is
    #  O(8) per turn: check the 8 neighbors, pick the closest-to-target
    #  passable one, move (building a road first if needed).
    #
    #  Tradeoff: no look-ahead — can get stuck at concave obstacles. The
    #  step-2 self-destruct timer plus per-target blacklists bound the
    #  damage.
    # ------------------------------------------------------------------ #

    def _disruptor_bugnav_step(self, ct, target_xy):
        """One greedy step toward target_xy. Returns:
          'arrived'    — my_xy == target_xy
          'moved'      — ct.move succeeded this turn
          'built_road' — built a road, waiting for next turn's move cooldown
          'blocked'    — no passable neighbor; caller may blacklist
        """
        my_xy = self._my_xy
        if my_xy == target_xy:
            return 'arrived'

        my_team = self.my_team_cache
        mw = self.map_w
        mh = self.map_h
        ec = self.enemy_core_pos
        candidates = []

        for d in _BUGNAV_DIRS:
            dx, dy = DIRECTION_DELTAS[d]
            nxy = (my_xy[0] + dx, my_xy[1] + dy)

            # Bounds + known walls
            if mw is not None:
                if nxy[0] < 0 or nxy[0] >= mw or nxy[1] < 0 or nxy[1] >= mh:
                    continue
            if nxy in self.known_walls:
                continue
            # Blocked enemy launcher zones
            if nxy in self.blocked_launcher_tiles:
                continue
            # Oscillation wall (set by _check_oscillation in main.py)
            if nxy in self.oscillation_walls:
                continue
            # Enemy core 3x3 footprint (unless target IS inside it)
            if ec is not None and nxy != target_xy:
                ecx, ecy = ec
                if abs(nxy[0] - ecx) <= 1 and abs(nxy[1] - ecy) <= 1:
                    continue
            # Other bots — skip unless target is on that tile
            if nxy in self.bot_pos_cache and nxy != target_xy:
                continue

            entry = self.tile_cache.get(nxy)
            if entry is not None:
                env, bid, etype, team = entry
                if env == Environment.WALL:
                    continue
                if bid is not None and etype not in _WALKABLE_BUILDINGS:
                    # Allow stepping onto an enemy attack-target (conveyor,
                    # bridge, etc. are walkable — filter already passed. This
                    # branch catches harvesters, sentinels, barriers, etc.)
                    if nxy != target_xy:
                        continue
            d2 = (nxy[0] - target_xy[0]) ** 2 + (nxy[1] - target_xy[1]) ** 2
            candidates.append((d2, d, nxy, entry))

        # Closest-to-target first. Explicit key to avoid comparing Direction
        # enums on d2 ties (TypeError on Python 3.13).
        candidates.sort(key=lambda c: c[0])

        for _, d, nxy, entry in candidates:
            if ct.can_move(d):
                ct.move(d)
                return 'moved'
            # Empty / ore tile with no building — build a road so it's walkable.
            if entry is None or entry[1] is None:
                if ct.get_action_cooldown() == 0:
                    npos = xy_to_pos(nxy)
                    if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                            and ct.can_build_road(npos)):
                        ct.build_road(npos)
                        env = entry[0] if entry else Environment.EMPTY
                        self.tile_cache[nxy] = (env, None, None, None)
                        if ct.can_move(d):
                            ct.move(d)
                            return 'moved'
                        return 'built_road'
        return 'blocked'

    def _disruptor_ring_goal(self, my_xy):
        """Closest non-wall ring tile adjacent to the enemy core 3x3 footprint.

        Ring = tiles at Chebyshev distance 2 from the enemy core center, on
        the 4 cardinal faces (3 tiles each = 12 tiles). We skip the 4
        diagonal-corner tiles at (cx±2, cy±2); the cardinal-face tiles are
        a superset of "reachable neighbors of the footprint edge."
        """
        ecx, ecy = self.enemy_core_pos
        candidates = []
        for dx in (-2, 2):
            for dy in (-1, 0, 1):
                candidates.append((ecx + dx, ecy + dy))
        for dy in (-2, 2):
            for dx in (-1, 0, 1):
                candidates.append((ecx + dx, ecy + dy))
        best = None
        best_d2 = None
        mw = self.map_w
        mh = self.map_h
        for rxy in candidates:
            rx, ry = rxy
            if rx < 0 or ry < 0 or (mw is not None and rx >= mw) or (mh is not None and ry >= mh):
                continue
            if rxy in self.known_walls:
                continue
            entry = self.tile_cache.get(rxy)
            if entry is not None:
                from cambc import Environment
                if entry[0] == Environment.WALL:
                    continue
            d2 = (rxy[0] - my_xy[0]) ** 2 + (rxy[1] - my_xy[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = rxy
        return best

    # ------------------------------------------------------------------ #
    #  Local symmetry fallback: only used when scanning.py has NOT cycled #
    #  but our arrival-check tells us the guess is wrong.                 #
    # ------------------------------------------------------------------ #

    def _disruptor_force_next_symmetry(self):
        """Called when the current guess has been invalidated by arrival
        (reached the guessed core tile with no core there). Advance to
        the next candidate in our distance-sorted priority order."""
        self.disruptor_forced_sym_failures += 1
        if not self._disruptor_apply_next_symmetry():
            if DEBUG:
                print(f"[DS] exhausted symmetries, waiting for self-destruct",
                      file=sys.stderr)
            self.path = None
            self.path_index = 0

    # ------------------------------------------------------------------ #
    #  Step 3: intercept enemy titanium harvesters                        #
    # ------------------------------------------------------------------ #

    def _disruptor_pick_intercept(self, my_xy):
        """Pick the best harvester to intercept from Step 2.

        Eligibility: enemy titanium harvester reachable via A* within
        INTERCEPT_DETOUR path tiles. We Chebyshev-prefilter to avoid A*ing
        obvious rejects. Returns a 4-tuple:

            (harvester_xy, feeder_xy_or_None, clean_case, sentinel_tile_or_None)

        Priority A (clean_case=True, sentinel_tile set): an empty cardinal
        neighbour of the harvester is available; we will walk there and drop
        a sentinel. Priority B (clean_case=False, feeder set): no empty tile
        adjacent, so we attack a cardinally-adjacent enemy conveyor/bridge.

        Returns None if no reachable harvester has either option.
        """
        best = None
        best_len = INTERCEPT_DETOUR + 1
        my_team = self.my_team_cache

        # Iterate current vision only. tile_cache accumulates thousands of
        # entries over a game; scanning a harvester out of it every turn
        # was the TLE source observed on the ladder (first sight of an
        # enemy harvester would pay full-map A* on top of a full-cache scan).
        for xy in self._last_vision_set:
            if xy in self.disrupt_attack_blacklist:
                continue
            info = self.tile_cache.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if etype != EntityType.HARVESTER:
                continue
            if team == my_team or team is None:
                continue
            if env != Environment.ORE_TITANIUM:
                continue
            if max(abs(xy[0] - my_xy[0]), abs(xy[1] - my_xy[1])) > INTERCEPT_DETOUR:
                continue
            # Skip harvesters whose defensive ring already has 2+
            # allied sentinels — adding more would oversaturate.
            if self._harvester_at_sentinel_cap(xy):
                continue
            empty = self._find_intercept_sentinel_tile(xy, my_xy)
            feeder = None if empty is not None else self._disruptor_find_feeder(xy)
            if empty is None and feeder is None:
                continue
            if feeder is not None and feeder in self.intercept_feeder_blacklist:
                continue
            # Greedy (not A*) for the eligibility check — optimistic about
            # unknowns, no dynamic-budget overhead, ~5x cheaper per call.
            # Tight node cap: search radius is INTERCEPT_DETOUR ≈ 6, so
            # 150 nodes is plenty for any reachable target. Caps each
            # failing candidate at ~120μs instead of the 1600μs that an
            # unbounded 2000-node greedy would burn — late game can have
            # many enemy harvesters in vision and we'd otherwise blow
            # the per-turn budget on a row of unreachable candidates.
            path = self._disruptor_greedy_path(my_xy, xy, max_nodes=150)
            if path is None or len(path) > INTERCEPT_DETOUR:
                continue
            plen = len(path)
            if plen >= best_len:
                continue
            best_len = plen
            if empty is not None:
                best = (xy, None, True, empty)
            else:
                best = (xy, feeder, False, None)
        return best

    def _disruptor_pick_ore_block(self, my_xy, max_path):
        """Find the closest enemy/unclaimed ore tile (titanium or axionite,
        no building, no bot standing on it) reachable via greedy path
        within `max_path` tiles. Returns (ore_xy) or None.

        Side filter: only consider ores whose squared-Euclidean distance
        to the enemy core (sighted or guessed) is strictly less than to
        our own core. Without this gate the picker happily barriers the
        ores on our spawn side — slowing our own economy because allied
        bots then have to bust the barrier on their way to harvest.

        Same eligibility shape as `_disruptor_pick_intercept`: Chebyshev
        prefilter to skip obvious far candidates, then run a greedy path
        on the survivors to find the cheapest.
        """
        # Side filter — strictly closer to the enemy core than to us.
        # Pre-sighting we infer the enemy core from scanning's
        # `self._symmetry` (rotated whenever scan observes a terrain
        # mismatch, so it's our best calibrated guess). The disruptor's
        # own `enemy_core_guess` is the distance-sorted guess that
        # rotates independently and can lag behind scan, which is
        # exactly how the previous version blocked ores on our side.
        # Once `enemy_core_pos` is sighted we use it directly.
        oc = self.core_pos
        if oc is None:
            return None
        ec = self.enemy_core_pos
        if ec is None:
            sym = self._symmetry
            if sym is None or self.map_w is None or self.map_h is None:
                return None
            ec = mirror_xy(oc, sym, self.map_w, self.map_h)
        my_team = self.my_team_cache
        best = None
        best_len = max_path + 1
        for xy in self._last_vision_set:
            if xy in self.disrupt_blacklist:
                continue
            if xy in self.disrupt_attack_blacklist:
                continue
            info = self.tile_cache.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            # Titanium only — axionite ores aren't worth burning a
            # barrier on (they don't drive the enemy economy directly,
            # and barriering them might block our own future foundry).
            if env != Environment.ORE_TITANIUM:
                continue
            # Empty ore only — anything on the tile (allied/enemy
            # harvester, road, marker, our own placed barrier) means
            # we either don't need to block or can't.
            if bid is not None:
                continue
            # A bot standing on the tile blocks build_barrier.
            if xy in self.bot_pos_cache:
                continue
            # Side filter — strictly closer to the enemy core than to
            # ours. The midline (d2_us == d2_them) goes to our side so
            # we never barrier symmetric centre-line ores.
            d2_us = (xy[0] - oc[0]) ** 2 + (xy[1] - oc[1]) ** 2
            d2_them = (xy[0] - ec[0]) ** 2 + (xy[1] - ec[1]) ** 2
            if d2_us <= d2_them:
                continue
            if max(abs(xy[0] - my_xy[0]), abs(xy[1] - my_xy[1])) > max_path:
                continue
            # Enclosed-ore guard: skip ores whose entire 8-neighbour
            # ring is known walls or non-walkable buildings. Without
            # this check the picker would still hit `_pick_adjacent_goal`
            # / greedy below, but if any neighbour is UNKNOWN the goal
            # picker accepts it optimistically — then the unbounded
            # greedy search burns its budget trying to route to a tile
            # that's actually walled off.
            if self._ore_neighbours_all_blocked(xy):
                continue
            adj = self._disruptor_pick_adjacent_goal(my_xy, xy)
            if adj is None:
                continue
            path = self._disruptor_greedy_path(my_xy, adj, max_nodes=150)
            if path is None:
                continue
            plen = len(path)
            if plen >= best_len:
                continue
            best_len = plen
            best = xy
        return best

    def _ore_neighbours_all_blocked(self, ore_xy):
        """True if every 8-neighbour of `ore_xy` is a known wall or a
        non-walkable building. Used to skip enemy-side ores that are
        completely walled off — placing a barrier on them would
        require an A* path that doesn't exist, which costs us a full
        greedy budget per failed candidate."""
        ox, oy = ore_xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (ox + dx, oy + dy)
                if nxy in self.known_walls:
                    continue
                info = self.tile_cache.get(nxy)
                if info is None:
                    return False
                env, bid, etype, _team = info
                if env == Environment.WALL:
                    continue
                if bid is not None and etype not in _WALKABLE_BUILDINGS:
                    continue
                return False
        return True

    def _disruptor_block_ore(self, ct, ore_xy):
        """Drive the navigate→barrier flow for ore-block targets.
        Returns:
          'done'        — barrier placed (or target invalidated by us)
          'in_progress' — consumed this turn navigating / waiting for cd
          'abandon'     — target no longer valid; caller should clear
        """
        info = self.tile_cache.get(ore_xy)
        if info is None:
            return 'abandon'
        env, bid, etype, team = info
        # Lost the window — someone built or it stopped being ore.
        if bid is not None:
            return 'abandon'
        if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            return 'abandon'

        my_xy = self._my_xy
        dxr = my_xy[0] - ore_xy[0]
        dyr = my_xy[1] - ore_xy[1]
        dsq = dxr * dxr + dyr * dyr

        # Within action radius — try the build.
        if dsq <= GC.ACTION_RADIUS_SQ:
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.BARRIER_BASE_COST[0])):
                ore_pos = xy_to_pos(ore_xy)
                if ct.can_build_barrier(ore_pos):
                    ct.build_barrier(ore_pos)
                    # Locally mark the tile as a barrier so the same-turn
                    # picker can't re-target it before scan refreshes.
                    self.tile_cache[ore_xy] = (
                        env, -1, EntityType.BARRIER, self.my_team_cache)
                    self.allied_barriers.add(ore_xy)
                    self._ds_event(f"[DS] ore-block barrier@{ore_xy}")
                    return 'done'
            # Cooldown / Ti reserve / can_build mismatch — hold the turn.
            return 'in_progress'

        # Need to walk closer.
        adj = self._disruptor_pick_adjacent_goal(my_xy, ore_xy)
        if adj is None:
            return 'abandon'
        result = self._disruptor_navigate(ct, adj)
        if result == 'blocked':
            return 'abandon'
        return 'in_progress'

    # ------------------------------------------------------------------ #
    #  Chain-extension intercept                                          #
    # ------------------------------------------------------------------ #
    #  Looks for any enemy CONVEYOR/SPLITTER/ARMOURED_CONVEYOR whose
    #  output tile is empty or a road, or any enemy BRIDGE whose target
    #  tile is empty or a road. That tile is where the enemy plans to
    #  extend their chain next, so a sentinel there blocks the build
    #  AND attacks (sentinel faces our enemy_core_pos so the attack
    #  cone covers the supplying chain behind us).
    # ------------------------------------------------------------------ #

    def _disruptor_pick_chain_extension_intercept(self, my_xy, max_path):
        """Find the closest enemy chain-extension target (empty or road
        downstream of an enemy conveyor/bridge) reachable within
        max_path. Returns (target_xy, source_xy) or None.

        Each enemy conveyor/splitter costs one ct.get_direction FFI
        call; bridges hit the cached `bridge_target_cache` first and
        only fall back to ct.get_bridge_target on miss. Source tiles
        are Chebyshev-prefiltered against `max_path + 1` so most enemy
        buildings are dropped before the FFI.
        """
        my_team = self.my_team_cache
        ct = self._ct
        ec = self.enemy_core_pos
        best = None
        best_len = max_path + 1

        for src_xy in self._last_vision_set:
            info = self.tile_cache.get(src_xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team == my_team or team is None:
                continue
            if etype not in (EntityType.CONVEYOR, EntityType.SPLITTER,
                             EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE):
                continue
            # Source must itself be inside max_path + 1 (target ≤ max_path).
            if max(abs(src_xy[0] - my_xy[0]),
                   abs(src_xy[1] - my_xy[1])) > max_path + 1:
                continue

            # Output / target tile. Per-bid lazy caches eliminate the
            # FFI hit on subsequent turns (conveyor direction and
            # bridge target are fixed for a bid's lifetime).
            if etype == EntityType.BRIDGE:
                target_xy = (self.bridge_target_cache.get(src_xy)
                             or self._enemy_bridge_target_cache.get(bid))
                if target_xy is None:
                    try:
                        tgt = ct.get_bridge_target(bid)
                    except Exception:
                        continue
                    target_xy = (tgt.x, tgt.y)
                    self._enemy_bridge_target_cache[bid] = target_xy
            else:
                direction = self._enemy_dir_cache.get(bid)
                if direction is None:
                    try:
                        direction = ct.get_direction(bid)
                    except Exception:
                        continue
                    self._enemy_dir_cache[bid] = direction
                target_xy = self._tile_in_direction(src_xy, direction)

            if target_xy in self.disrupt_blacklist:
                continue
            if target_xy in self.disrupt_attack_blacklist:
                continue
            tinfo = self.tile_cache.get(target_xy)
            if tinfo is None:
                continue
            t_env, t_bid, t_etype, t_team = tinfo
            if t_env == Environment.WALL:
                continue
            # Empty (no building) or any-team road only.
            if t_bid is not None and t_etype != EntityType.ROAD:
                continue
            # Skip ores — those are blocked via _disruptor_pick_ore_block.
            if t_bid is None and t_env in (Environment.ORE_TITANIUM,
                                           Environment.ORE_AXIONITE):
                continue
            # Don't try to put a sentinel on the enemy core 3x3 footprint.
            if ec is not None:
                if abs(target_xy[0] - ec[0]) <= 1 and abs(target_xy[1] - ec[1]) <= 1:
                    continue
            # Bot blocking — can't build through.
            if target_xy in self.bot_pos_cache:
                continue
            if max(abs(target_xy[0] - my_xy[0]),
                   abs(target_xy[1] - my_xy[1])) > max_path:
                continue

            adj = self._disruptor_pick_adjacent_goal(my_xy, target_xy)
            if adj is None:
                continue
            path = self._disruptor_greedy_path(my_xy, adj, max_nodes=150)
            if path is None:
                continue
            plen = len(path)
            if plen >= best_len:
                continue
            best_len = plen
            best = (target_xy, src_xy)
        return best

    # ------------------------------------------------------------------ #
    #  Splitter-sentinel target                                           #
    # ------------------------------------------------------------------ #
    #  Place a sentinel on an empty / road tile cardinally adjacent to
    #  an enemy SPLITTER (excluding the splitter's back / input tile).
    #  The splitter feeds the sentinel its ammo, so the sentinel's
    #  facing must NOT point at the splitter — pick the 7-direction
    #  facing that hits the most enemy buildings instead.
    # ------------------------------------------------------------------ #

    def _disruptor_pick_splitter_sentinel(self, my_xy, max_path):
        """Find the best sentinel placement adjacent to an enemy
        splitter. Returns (placement_xy, splitter_xy) or None.

        Selection: across all (splitter, candidate-cardinal) pairs,
        pick the one whose best 7-direction sentinel facing covers
        the most enemy buildings (CORE wins outright). Reachability
        is verified by the same Chebyshev prefilter + capped greedy
        path used by the other pickers.
        """
        my_team = self.my_team_cache
        ct = self._ct
        ec = self.enemy_core_pos

        best_target = None      # (placement_xy, splitter_xy)
        best_score = 0          # number of enemies in cone (CORE = sentinel _try result)

        for src_xy in self._last_vision_set:
            info = self.tile_cache.get(src_xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if (bid is None or team == my_team or team is None
                    or etype != EntityType.SPLITTER):
                continue
            if max(abs(src_xy[0] - my_xy[0]),
                   abs(src_xy[1] - my_xy[1])) > max_path + 1:
                continue
            # Splitter direction (use cached; FFI on miss).
            sdir = self._enemy_dir_cache.get(bid)
            if sdir is None:
                try:
                    sdir = ct.get_direction(bid)
                except Exception:
                    continue
                self._enemy_dir_cache[bid] = sdir
            sdx, sdy = DIRECTION_DELTAS.get(sdir, (0, 0))
            # Back of the splitter is the tile in the opposite
            # direction — that's its input feed, sentinel can't sit
            # there without breaking the splitter's own supply chain.
            back_offset = (-sdx, -sdy)

            for cdx, cdy in _CARDINAL_OFFSETS:
                if (cdx, cdy) == back_offset:
                    continue
                cand = (src_xy[0] + cdx, src_xy[1] + cdy)
                if cand == my_xy:
                    continue
                ce = self.tile_cache.get(cand)
                if ce is None:
                    continue
                c_env, c_bid, c_etype, c_team = ce
                if c_env == Environment.WALL:
                    continue
                if c_bid is not None and c_etype != EntityType.ROAD:
                    continue
                if c_bid is None and c_env in (Environment.ORE_TITANIUM,
                                               Environment.ORE_AXIONITE):
                    continue
                if cand in self.bot_pos_cache:
                    continue
                if (ec is not None
                        and abs(cand[0] - ec[0]) <= 1
                        and abs(cand[1] - ec[1]) <= 1):
                    continue
                if cand in self.disrupt_attack_blacklist:
                    continue
                if cand in self.disrupt_blacklist:
                    continue
                if max(abs(cand[0] - my_xy[0]),
                       abs(cand[1] - my_xy[1])) > max_path:
                    continue
                # Reachable from us?
                adj = self._disruptor_pick_adjacent_goal(my_xy, cand)
                if adj is None:
                    continue
                path = self._disruptor_greedy_path(my_xy, adj, max_nodes=150)
                if path is None:
                    continue

                # Score this placement: best 7-direction enemy count.
                score = self._splitter_sentinel_score(ct, cand, src_xy)
                if score > best_score:
                    best_score = score
                    best_target = (cand, src_xy)

        return best_target

    def _splitter_sentinel_score(self, ct, sentinel_xy, splitter_xy):
        """Count of enemy buildings inside the best 7-direction
        sentinel cone from `sentinel_xy`, excluding the direction
        toward `splitter_xy` (its ammo source). The goal is economy
        disruption — CORE is treated like any other building (no
        priority bonus)."""
        sentinel_pos = xy_to_pos(sentinel_xy)
        excluded = direction_between(sentinel_xy, splitter_xy)
        my_team = self.my_team_cache
        best = 0
        for d in _BUGNAV_DIRS:
            if d == excluded:
                continue
            try:
                tiles = ct.get_attackable_tiles_from(
                    sentinel_pos, d, EntityType.SENTINEL)
            except Exception:
                continue
            count = 0
            for t in tiles:
                info = self.tile_cache.get((t.x, t.y))
                if info is None:
                    continue
                bid, btype, bteam = info[1], info[2], info[3]
                if bid is None or bteam == my_team:
                    continue
                if btype in (EntityType.ROAD, EntityType.MARKER,
                             EntityType.HARVESTER, EntityType.FOUNDRY,
                             EntityType.BARRIER):
                    continue
                count += 1
            if count > best:
                best = count
        return best

    def _disruptor_block_splitter_sentinel(self, ct, placement_xy, splitter_xy):
        """Drive the navigate→sentinel-on-tile flow for a splitter-
        sentinel target. Returns 'done', 'in_progress', or 'abandon'.

        Mirrors _disruptor_block_chain_extension but the sentinel
        facing comes from a 7-direction scan that excludes the
        direction toward the splitter (its ammo source).
        """
        info = self.tile_cache.get(placement_xy)
        if info is None:
            return 'abandon'
        env, bid, etype, team = info
        if env == Environment.WALL:
            return 'abandon'
        if bid is not None and etype != EntityType.ROAD:
            return 'abandon'
        # Splitter still there?
        sinfo = self.tile_cache.get(splitter_xy)
        if (sinfo is None or sinfo[1] is None
                or sinfo[2] != EntityType.SPLITTER
                or sinfo[3] == self.my_team_cache):
            return 'abandon'

        my_xy = self._my_xy
        is_enemy_road = (bid is not None and etype == EntityType.ROAD
                         and team != self.my_team_cache)
        is_allied_road = (bid is not None and etype == EntityType.ROAD
                          and team == self.my_team_cache)
        target_pos = xy_to_pos(placement_xy)
        dsq = (my_xy[0] - placement_xy[0]) ** 2 + (my_xy[1] - placement_xy[1]) ** 2

        if is_enemy_road:
            if my_xy != placement_xy:
                result = self._disruptor_navigate(ct, placement_xy)
                if result == 'blocked':
                    return 'abandon'
                return 'in_progress'
            if self._attack_heal_check(ct, bid, placement_xy):
                return 'abandon'
            my_pos = xy_to_pos(my_xy)
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0])
                    and ct.can_fire(my_pos)):
                ct.fire(my_pos)
                self._attack_heal_note_fire()
                self._ds_event(
                    f"[DS] splitter-sent fire enemy-road@{placement_xy}")
            return 'in_progress'

        if dsq > GC.ACTION_RADIUS_SQ:
            adj = self._disruptor_pick_adjacent_goal(my_xy, placement_xy)
            if adj is None:
                return 'abandon'
            result = self._disruptor_navigate(ct, adj)
            if result == 'blocked':
                return 'abandon'
            return 'in_progress'
        if dsq == 0:
            self._disruptor_step_off(ct, my_xy)
            return 'in_progress'

        if is_allied_road:
            if ct.can_destroy(target_pos):
                ct.destroy(target_pos)
                self.tile_cache[placement_xy] = (env, None, None, None)
                self._ds_event(
                    f"[DS] splitter-sent destroy ally-road@{placement_xy}")

        # Pick facing via the 7-direction scan, excluding direction
        # toward the splitter (the sentinel's ammo source).
        sentinel_dir = self._intercept_pick_sentinel_dir(
            ct, placement_xy, splitter_xy)
        if sentinel_dir is None:
            return 'abandon'

        ti, _ax = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0
        sentinel_cost = int(GC.SENTINEL_BASE_COST[0] * scale)
        if (ct.get_action_cooldown() == 0
                and ti >= sentinel_cost
                and ct.can_build_sentinel(target_pos, sentinel_dir)):
            ct.build_sentinel(target_pos, sentinel_dir)
            self.tile_cache[placement_xy] = (
                env, -1, EntityType.SENTINEL, self.my_team_cache)
            self._ds_event(
                f"[DS] splitter-sent sentinel@{placement_xy} "
                f"split={splitter_xy} dir={sentinel_dir.value}")
            return 'done'
        return 'in_progress'

    def _disruptor_block_chain_extension(self, ct, target_xy, source_xy):
        """Drive: navigate to the target, clear the road if any, place a
        sentinel facing the enemy core. Returns 'done', 'in_progress',
        or 'abandon'.

        Empty target → walk adjacent + build sentinel.
        Allied road  → walk adjacent + free destroy + build sentinel.
        Enemy road   → walk onto + fire (builder fire range = 0) until
                       destroyed, then walk adjacent + build sentinel.
        """
        info = self.tile_cache.get(target_xy)
        if info is None:
            return 'abandon'
        env, bid, etype, team = info
        if env == Environment.WALL:
            return 'abandon'
        # Target must still be empty or a road (anything else means
        # someone built on it before we got there).
        if bid is not None and etype != EntityType.ROAD:
            return 'abandon'

        my_xy = self._my_xy
        is_enemy_road = (bid is not None and etype == EntityType.ROAD
                         and team != self.my_team_cache)
        is_allied_road = (bid is not None and etype == EntityType.ROAD
                          and team == self.my_team_cache)
        target_pos = xy_to_pos(target_xy)
        dsq = (my_xy[0] - target_xy[0]) ** 2 + (my_xy[1] - target_xy[1]) ** 2

        # Enemy road — builder bot fire range is 0, so we have to step onto
        # the tile, fire, then step off and rebuild from the adjacent flow.
        if is_enemy_road:
            if my_xy != target_xy:
                result = self._disruptor_navigate(ct, target_xy)
                if result == 'blocked':
                    return 'abandon'
                return 'in_progress'
            if self._attack_heal_check(ct, bid, target_xy):
                return 'abandon'
            my_pos = xy_to_pos(my_xy)
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0])
                    and ct.can_fire(my_pos)):
                ct.fire(my_pos)
                self._attack_heal_note_fire()
                self._ds_event(f"[DS] chain-block fire enemy-road@{target_xy}")
            return 'in_progress'

        # Empty / allied-road: navigate to action range adjacent.
        if dsq > GC.ACTION_RADIUS_SQ:
            adj = self._disruptor_pick_adjacent_goal(my_xy, target_xy)
            if adj is None:
                return 'abandon'
            result = self._disruptor_navigate(ct, adj)
            if result == 'blocked':
                return 'abandon'
            return 'in_progress'
        if dsq == 0:
            # Standing on it (rare with pick_adjacent_goal). Step off so
            # we can build a non-walkable sentinel on the tile.
            self._disruptor_step_off(ct, my_xy)
            return 'in_progress'

        # Allied road: free destroy first (doesn't burn action cooldown,
        # so we can still place the sentinel this turn).
        if is_allied_road:
            if ct.can_destroy(target_pos):
                ct.destroy(target_pos)
                self.tile_cache[target_xy] = (env, None, None, None)
                self._ds_event(f"[DS] chain-block destroy ally-road@{target_xy}")

        # Build a turret facing the enemy core (sighted or guessed) so
        # the attack cone covers the supplying chain that points at us.
        # Prefer a GUNNER if its longer attack cone would reach the
        # enemy core — burns the core directly. Otherwise fall back to
        # SENTINEL which still denies the build site and shoots at the
        # nearby chain.
        # Bypasses the team-wide MIN_TITANIUM reserve gate (`_can_spend`)
        # for this build: chain-block turrets are tactically valuable
        # enough to spend into the reserve.
        ec_known = self.enemy_core_pos
        ec = ec_known or self.enemy_core_guess
        if ec is None:
            return 'abandon'
        sentinel_dir = cardinal_direction_between(target_xy, ec)
        ti, _ax = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0

        # GUNNER preferred if SOME facing puts the enemy core in its
        # cone. Gunners reach 3 tiles in the cardinal direction and 2
        # diagonally, so the right facing depends on geometry — checking
        # only the cardinal-toward-core misses many corner-of-core
        # positions where a different facing reaches the 3x3 footprint.
        # Try the 7 facings that aren't pointing at the source — gunners
        # take ammo from the facing direction, so facing the supplying
        # conveyor / bridge would starve the turret.
        gunner_dir = None
        if ec_known is not None and ct.get_action_cooldown() == 0:
            core_tiles = set(
                (ec_known[0] + dx, ec_known[1] + dy)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            excluded = direction_between(target_xy, source_xy)
            for d in (Direction.NORTH, Direction.EAST,
                      Direction.SOUTH, Direction.WEST,
                      Direction.NORTHEAST, Direction.SOUTHEAST,
                      Direction.SOUTHWEST, Direction.NORTHWEST):
                if d == excluded:
                    continue
                try:
                    tiles = ct.get_attackable_tiles_from(
                        target_pos, d, EntityType.GUNNER)
                except Exception:
                    continue
                for t in tiles:
                    if (t.x, t.y) in core_tiles:
                        gunner_dir = d
                        break
                if gunner_dir is not None:
                    break

        if gunner_dir is not None:
            gunner_cost = int(GC.GUNNER_BASE_COST[0] * scale)
            if (ct.get_action_cooldown() == 0
                    and ti >= gunner_cost
                    and ct.can_build_gunner(target_pos, gunner_dir)):
                ct.build_gunner(target_pos, gunner_dir)
                self.tile_cache[target_xy] = (
                    env, -1, EntityType.GUNNER, self.my_team_cache)
                self._ds_event(
                    f"[DS] chain-block GUNNER@{target_xy} src={source_xy} "
                    f"dir={gunner_dir.value} (reaches core)")
                return 'done'
            return 'in_progress'

        sentinel_cost = int(GC.SENTINEL_BASE_COST[0] * scale)
        if (ct.get_action_cooldown() == 0
                and ti >= sentinel_cost
                and ct.can_build_sentinel(target_pos, sentinel_dir)):
            ct.build_sentinel(target_pos, sentinel_dir)
            self.tile_cache[target_xy] = (
                env, -1, EntityType.SENTINEL, self.my_team_cache)
            self._ds_event(
                f"[DS] chain-block sentinel@{target_xy} src={source_xy} "
                f"dir={sentinel_dir.value}")
            return 'done'
        return 'in_progress'

    def _disruptor_find_feeder(self, hxy):
        """Cardinally-adjacent non-armoured enemy conveyor or bridge."""
        my_team = self.my_team_cache
        for dx, dy in _CARDINAL_OFFSETS:
            nxy = (hxy[0] + dx, hxy[1] + dy)
            info = self.tile_cache.get(nxy)
            if info is None:
                continue
            _, bid, etype, team = info
            if bid is None or team == my_team or team is None:
                continue
            # Ignore ARMOURED_CONVEYOR — builder bots can't destroy it fast enough.
            if etype in (EntityType.CONVEYOR, EntityType.BRIDGE):
                return nxy
        return None

    def _intercept_pick_sentinel_dir(self, ct, sentinel_xy, harvester_xy):
        """7-direction sentinel scan for the intercept placement. Excludes
        the direction pointing at the harvester (a sentinel can't hit its
        own facing tile). Returns the direction that covers the most enemy
        buildings, CORE being an immediate win. Falls back to "away from
        harvester" if every scanned direction is empty — that keeps the
        harvester inside the king-move attack cone.
        """
        sentinel_pos = xy_to_pos(sentinel_xy)
        harvester_dir = cardinal_direction_between(sentinel_xy, harvester_xy)
        my_team = self.my_team_cache
        best_dir = None
        best_count = -1
        for d in _BUGNAV_DIRS:   # 8 dirs, cardinal + diagonal
            if d == harvester_dir:
                continue
            try:
                tiles = ct.get_attackable_tiles_from(
                    sentinel_pos, d, EntityType.SENTINEL)
            except Exception:
                continue
            count = 0
            saw_core = False
            for t in tiles:
                info = self.tile_cache.get((t.x, t.y))
                if info is None:
                    continue
                bid, btype, bteam = info[1], info[2], info[3]
                if bid is None or bteam == my_team:
                    continue
                if btype == EntityType.CORE:
                    saw_core = True
                    break
                # Harvesters/foundries fuel sentinels (ours included);
                # roads, markers, and barriers aren't worth attacking.
                if btype in (EntityType.HARVESTER, EntityType.FOUNDRY,
                             EntityType.ROAD, EntityType.MARKER,
                             EntityType.BARRIER):
                    continue
                count += 1
            if saw_core:
                return d
            if count > best_count:
                best_count = count
                best_dir = d
        if best_dir is None:
            # No enemies in any scanned cone — face AWAY from the harvester
            # so the harvester sits inside the attack cone.
            best_dir = cardinal_direction_between(harvester_xy, sentinel_xy)
        return best_dir

    def _harvester_at_sentinel_cap(self, hxy):
        """True iff harvester `hxy` already has 2+ allied sentinels
        cardinally adjacent. Disruptor intercepts skip a saturated
        harvester rather than oversaturating its defensive ring.
        A sentinel can sit between two harvesters and count for both
        — the cap is per-harvester, not per-sentinel."""
        my_team = self.my_team_cache
        count = 0
        for dx, dy in _CARDINAL_OFFSETS:
            nxy = (hxy[0] + dx, hxy[1] + dy)
            info = self.tile_cache.get(nxy)
            if info is None:
                continue
            bid, etype, team = info[1], info[2], info[3]
            if (bid is not None and team == my_team
                    and etype == EntityType.SENTINEL):
                count += 1
                if count >= 2:
                    return True
        return False

    def _find_intercept_sentinel_tile(self, hxy, my_xy):
        """Priority A: empty cardinal neighbour of the harvester. Return the
        closest-to-me candidate or None. Excludes walls, tiles with
        buildings, and tiles currently occupied by a bot (we can't place
        under a bot)."""
        best = None
        best_d2 = None
        for dx, dy in _CARDINAL_OFFSETS:
            cxy = (hxy[0] + dx, hxy[1] + dy)
            if cxy in self.known_walls:
                continue
            info = self.tile_cache.get(cxy)
            if info is None:
                continue
            env, bid, _, _ = info
            if env == Environment.WALL:
                continue
            if bid is not None:
                continue
            if cxy in self.bot_pos_cache:
                continue
            d2 = (cxy[0] - my_xy[0]) ** 2 + (cxy[1] - my_xy[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best = cxy
                best_d2 = d2
        return best

    def _disruptor_step3_intercept(self, ct):
        """Sub-state machine:
          0 = plan (caller pre-set clean/dirty + target; verify + advance)
          1 = clean case: walk to action range of sentinel_tile, place sentinel
              dirty case: walk onto feeder, fire until destroyed, move off
          2 = dirty case only: we just stepped off destroyed tile; place sentinel
          3 = done — re-check intercepts or return to caller's step
        """
        my_xy = self._my_xy

        if self.intercept_target_harvester is None:
            self._disruptor_intercept_reset_and_resume()
            return

        hxy = self.intercept_target_harvester
        fxy = self.intercept_target_conveyor   # None when clean_case

        # Sanity check — only trust cache for tiles observed THIS turn
        # (scanning clears bids when a tile drops out of vision, which
        # would otherwise false-positive as "gone").
        vision = self._last_vision_set
        hinfo = self.tile_cache.get(hxy)
        harv_gone = False
        if hxy in vision:
            harv_gone = (hinfo is None
                         or hinfo[2] != EntityType.HARVESTER
                         or hinfo[3] == self.my_team_cache)
        feeder_gone = False
        if fxy is not None and fxy in vision:
            finfo = self.tile_cache.get(fxy)
            feeder_gone = (finfo is None
                           or finfo[1] is None
                           or finfo[3] == self.my_team_cache
                           or finfo[2] not in (EntityType.CONVEYOR, EntityType.BRIDGE))
        if harv_gone or (feeder_gone and self.intercept_substep < 2):
            if DEBUG:
                print(f"[DS] intercept aborted (harv_gone={harv_gone} feeder_gone={feeder_gone})",
                      file=sys.stderr)
            self._disruptor_intercept_reset_and_resume()
            return

        if self.intercept_substep == 0:
            # Caller (_disruptor_pick_intercept / step 5 harass) has already
            # chosen clean_case + sentinel_tile or the feeder attack target.
            # If nothing was set (defensive: e.g. the caller raced another
            # unit), re-evaluate from current vision.
            if self.intercept_clean_case and self.intercept_sentinel_tile is None:
                empty = self._find_intercept_sentinel_tile(hxy, my_xy)
                if empty is not None:
                    self.intercept_sentinel_tile = empty
                else:
                    self.intercept_clean_case = False
                    feeder = self._disruptor_find_feeder(hxy)
                    if feeder is None:
                        self._disruptor_intercept_reset_and_resume()
                        return
                    self.intercept_target_conveyor = feeder
                    self.intercept_destroyed_tile = feeder
            elif not self.intercept_clean_case and fxy is None:
                feeder = self._disruptor_find_feeder(hxy)
                if feeder is None:
                    self._disruptor_intercept_reset_and_resume()
                    return
                self.intercept_target_conveyor = feeder
                self.intercept_destroyed_tile = feeder
            self._ds_event(
                f"[DS] intercept "
                f"{'CLEAN' if self.intercept_clean_case else 'DIRTY'} "
                f"target="
                f"{self.intercept_sentinel_tile if self.intercept_clean_case else self.intercept_target_conveyor}")
            self.intercept_substep = 1
            self.path = None
            self.path_index = 0

        if self.intercept_substep == 1:
            if self.intercept_clean_case:
                self._disruptor_intercept_clean(ct, my_xy, hxy)
            else:
                self._disruptor_intercept_dirty_attack(ct, my_xy, hxy, fxy)
            return

        if self.intercept_substep == 2:
            # Dirty case: we destroyed feeder, now place sentinel on that tile
            self._disruptor_intercept_dirty_place(ct, my_xy, hxy)
            return

        if self.intercept_substep == 3:
            self._disruptor_intercept_reset_and_resume()
            return

    def _disruptor_intercept_clean(self, ct, my_xy, hxy):
        """Pathfind to within action range of sentinel_tile, then place sentinel.

        Feeder-present: face the feeder (king-move cone covers feeder +
        harvester). Feeder-absent (the new sentinel-first path): face
        AWAY from the harvester so the harvester sits inside the cone.
        """
        tile = self.intercept_sentinel_tile
        fxy = self.intercept_target_conveyor
        tile_pos = xy_to_pos(tile)

        dsq = (my_xy[0] - tile[0]) ** 2 + (my_xy[1] - tile[1]) ** 2
        if dsq == 0:
            # Standing on the tile — step off so we can build non-walkable here.
            self._disruptor_step_off(ct, my_xy)
            return

        info = self.tile_cache.get(tile)
        if info is not None and info[1] is not None:
            # Someone built here first. Abort this intercept; harass/step2 will
            # re-scan on the next turn.
            if DEBUG:
                print(f"[DS] clean sentinel tile {tile} now occupied; abort",
                      file=sys.stderr)
            self._disruptor_intercept_reset_and_resume()
            return

        if dsq <= GC.ACTION_RADIUS_SQ:
            # Close enough — pick facing via 7-dir enemy-building scan.
            sentinel_dir = self._intercept_pick_sentinel_dir(ct, tile, hxy)
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.SENTINEL_BASE_COST[0])
                    and ct.can_build_sentinel(tile_pos, sentinel_dir)):
                ct.build_sentinel(tile_pos, sentinel_dir)
                self._ds_event(f"[DS] intercept sentinel at {tile} facing {sentinel_dir.value} (CLEAN)")
                self.intercept_substep = 3
            return

        # Not in range — pathfind adjacent (not onto)
        # Short-range A* to an adjacent tile of the sentinel spot.
        goal = self._disruptor_pick_adjacent_goal(my_xy, tile)
        if goal is None:
            if DEBUG:
                print(f"[DS] clean no-adj goal at tile={tile}; blacklist + abandon",
                      file=sys.stderr)
            self.intercept_feeder_blacklist.add(self.intercept_target_conveyor)
            self._disruptor_intercept_reset_and_resume()
            return
        if (self.path is None or not self.path or self.path[-1] != goal):
            path = self._compute_path(my_xy, goal)
            if path is None:
                if DEBUG:
                    print(f"[DS] clean A* fail {my_xy}->{goal}; blacklist + abandon",
                          file=sys.stderr)
                self.intercept_feeder_blacklist.add(self.intercept_target_conveyor)
                self._disruptor_intercept_reset_and_resume()
                return
            self.path = path
            self.path_index = 0
        result = self._follow_path(ct)
        if result == 'blocked':
            if DEBUG:
                print(f"[DS] clean follow_path blocked; clearing path", file=sys.stderr)
            self.path = None
            self.path_index = 0

    def _disruptor_intercept_dirty_attack(self, ct, my_xy, hxy, fxy):
        """Walk onto the feeder and fire at own tile until it's destroyed."""
        if my_xy == fxy:
            # Feeder gone?
            info = self.tile_cache.get(fxy)
            if info is None or info[1] is None:
                # Destroyed — step off and place sentinel
                self.intercept_substep = 2
                self.path = None
                self.path_index = 0
                self._attack_heal_reset()
                return
            # Still there — heal-check then fire.
            if self._attack_heal_check(ct, info[1], fxy):
                self.intercept_feeder_blacklist.add(fxy)
                self._disruptor_intercept_reset_and_resume()
                return
            if ct.get_action_cooldown() != 0:
                return
            mpos = xy_to_pos(my_xy)
            if (self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0])
                    and ct.can_fire(mpos)):
                ct.fire(mpos)
                self._attack_heal_note_fire()
                self._ds_event(f"[DS] intercept fire@{my_xy}")
            return

        # Not on the feeder yet — short-range A* onto it. Conveyors & bridges are walkable.
        if (self.path is None or not self.path or self.path[-1] != fxy):
            path = self._compute_path(my_xy, fxy)
            if path is None:
                if DEBUG:
                    print(f"[DS] dirty A* fail {my_xy}->{fxy}; blacklist + abandon",
                          file=sys.stderr)
                self.intercept_feeder_blacklist.add(fxy)
                self._disruptor_intercept_reset_and_resume()
                return
            self.path = path
            self.path_index = 0
        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0

    def _disruptor_intercept_dirty_place(self, ct, my_xy, hxy):
        """We destroyed the feeder and need to: (1) step off, (2) place a sentinel."""
        tile = self.intercept_destroyed_tile
        if tile is None:
            self._disruptor_intercept_reset_and_resume()
            return

        if my_xy == tile:
            # Still on the destroyed tile — step off
            self._disruptor_step_off(ct, my_xy)
            return

        # Off the tile. If in action range, try to place the sentinel.
        dsq = (my_xy[0] - tile[0]) ** 2 + (my_xy[1] - tile[1]) ** 2
        if dsq > GC.ACTION_RADIUS_SQ:
            # Drifted too far — short-range A* back to an adjacent tile.
            goal = self._disruptor_pick_adjacent_goal(my_xy, tile)
            if goal is None:
                self._disruptor_intercept_reset_and_resume()
                return
            if (self.path is None or not self.path or self.path[-1] != goal):
                path = self._compute_path(my_xy, goal)
                if path is None:
                    self._disruptor_intercept_reset_and_resume()
                    return
                self.path = path
                self.path_index = 0
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
            return

        # In range — run the 7-dir scan + build sentinel
        info = self.tile_cache.get(tile)
        if info is not None and info[1] is not None:
            # Something rebuilt on the tile before us (unlikely) — abandon
            self._disruptor_intercept_reset_and_resume()
            return

        # 7-dir scan — we committed to intercepting, so we always place a
        # sentinel (no threshold gate). Picker returns the best available
        # direction by enemy-building count.
        d = self._intercept_pick_sentinel_dir(ct, tile, hxy)
        tile_pos = xy_to_pos(tile)
        if ct.get_action_cooldown() == 0 and ct.can_build_sentinel(tile_pos, d):
            ct.build_sentinel(tile_pos, d)
            self._ds_event(f"[DS] intercept sentinel at {tile} facing {d.value} (dirty)")
            self.intercept_substep = 3

    def _disruptor_step_off(self, ct, my_xy):
        """Step off to any adjacent walkable tile. Prefer toward enemy core
        (progress) over away from it."""
        target = self.enemy_core_pos or self.enemy_core_guess
        prefer_dir = direction_between(my_xy, target) if target else None
        tried_dirs = []
        if prefer_dir is not None and prefer_dir != Direction.CENTRE:
            tried_dirs.append(prefer_dir)
        for d in DIRECTION_DELTAS:
            if d == prefer_dir or d == Direction.CENTRE:
                continue
            tried_dirs.append(d)
        for d in tried_dirs:
            if ct.can_move(d):
                ct.move(d)
                self.path = None
                self.path_index = 0
                return
        # No movement available this turn — try building a road toward enemy core
        if ct.get_action_cooldown() == 0:
            for d in tried_dirs:
                dx, dy = DIRECTION_DELTAS[d]
                nxy = (my_xy[0] + dx, my_xy[1] + dy)
                if nxy in self.known_walls:
                    continue
                entry = self.tile_cache.get(nxy)
                if entry and entry[0] == Environment.WALL:
                    continue
                if entry and entry[1] is not None:
                    continue
                npos = xy_to_pos(nxy)
                if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                        and ct.can_build_road(npos)):
                    ct.build_road(npos)
                    if ct.can_move(d):
                        ct.move(d)
                    return

    def _disruptor_pick_adjacent_goal(self, my_xy, tile_xy):
        """Closest bot-walkable 8-neighbor of tile_xy — A* goal for approach.

        Filters out walls AND non-walkable buildings (harvesters, barriers,
        turrets, foundries, etc.). Without this filter the picker will
        happily return the harvester itself as "closest neighbour of the
        sentinel tile", A* can't route onto it, and the caller
        (_disruptor_intercept_clean) bounces into reset_and_resume on the
        same turn the pick happened — the bot never physically moves.
        """
        tx, ty = tile_xy
        best = None
        best_d2 = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (tx + dx, ty + dy)
                if nxy in self.known_walls:
                    continue
                entry = self.tile_cache.get(nxy)
                if entry is not None:
                    env, bid, etype, _ = entry
                    if env == Environment.WALL:
                        continue
                    if bid is not None and etype not in _WALKABLE_BUILDINGS:
                        continue
                d2 = (nxy[0] - my_xy[0]) ** 2 + (nxy[1] - my_xy[1]) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = nxy
        return best

    # ------------------------------------------------------------------ #
    #  Core-ring helper (shared with Step 4 Priority 1 and legacy Step 8  #
    #  launcher placement, which Case B still reuses).                    #
    # ------------------------------------------------------------------ #

    def _disruptor_core_ring_tiles(self):
        """The 12 cardinally-adjacent-to-footprint ring tiles. Same geometry
        as `_disruptor_ring_goal` but returns the list instead of picking one.
        """
        ecx, ecy = self.enemy_core_pos
        tiles = []
        for dx in (-2, 2):
            for dy in (-1, 0, 1):
                tiles.append((ecx + dx, ecy + dy))
        for dy in (-2, 2):
            for dx in (-1, 0, 1):
                tiles.append((ecx + dx, ecy + dy))
        return tiles

    def _disruptor_intercept_reset_and_resume(self):
        """Clear intercept state and resume whichever step the intercept
        was launched from (step 2 normally, step 5 in harass mode)."""
        # Latch the "already intercepted this trip" flag only when the
        # reset follows a successful sentinel placement (substep 3 reached)
        # during a Step-2 intercept. Aborts from substep 0/1/2 shouldn't
        # count — the bot didn't actually drop a sentinel.
        placed = (self.intercept_substep == 3)
        if placed and (self.intercept_post_step or 2) == 2:
            self._step2_intercepted = True
        self.intercept_target_harvester = None
        self.intercept_target_conveyor = None
        self.intercept_sentinel_tile = None
        self.intercept_sentinel_dir = None
        self.intercept_destroyed_tile = None
        self.intercept_substep = 0
        self.intercept_clean_case = False
        self.path = None
        self.path_index = 0
        self._attack_heal_reset()
        # Scrub any residual oscillation walls accumulated during the
        # intercept walk/fire cycle so the post-intercept greedy pathfind
        # doesn't route around phantom obstacles.
        if self.oscillation_walls:
            self.oscillation_walls.clear()
        # Return to the step the intercept was launched from.
        self.disruptor_step = self.intercept_post_step or 2

    # ------------------------------------------------------------------ #
    #  Step 8: place launchers on the core-adjacent ring                  #
    # ------------------------------------------------------------------ #

    def _disruptor_step8_place_launchers(self, ct):
        my_xy = self._my_xy

        if self.launcher_place_substep == 0:
            target = self._disruptor_find_launcher_target(my_xy)
            if target is None:
                # Nothing left on the ring to place on — escalate to Step 9.
                self.disruptor_step = 9
                self.launcher_place_substep = 0
                self._ds_event("[DS] step 8: no launcher target -> step 9")
                return
            info = self.tile_cache.get(target)
            if info is None:
                self.launcher_place_attempted.add(target)
                return
            env, bid, etype, team = info
            if bid is None:
                road_type = 'none'
            elif etype == EntityType.ROAD and team == self.my_team_cache:
                road_type = 'allied'
            elif etype == EntityType.ROAD and team != self.my_team_cache:
                road_type = 'enemy'
            else:
                # Shouldn't happen (planner filter), but be safe
                self.launcher_place_attempted.add(target)
                return
            self.launcher_place_target = target
            self.launcher_place_road_type = road_type
            self.launcher_place_substep = 1
            self.path = None
            self.path_index = 0
            self._ds_event(f"[DS] step 8 target={target} road={road_type}")

        target = self.launcher_place_target
        if target is None:
            self.launcher_place_substep = 0
            return

        if self.launcher_place_substep == 1:
            # Approach. Enemy road: walk ONTO it (to attack own tile).
            # Otherwise walk adjacent.
            if self.launcher_place_road_type == 'enemy':
                if my_xy == target:
                    self.launcher_place_substep = 2
                    self.path = None
                    self.path_index = 0
                    return
                result = self._disruptor_navigate(ct, target)
                if result == 'blocked':
                    self.launcher_place_attempted.add(target)
                    self.launcher_place_substep = 0
                    self.launcher_place_target = None
                return

            # 'none' or 'allied': approach any adjacent tile
            dsq = (my_xy[0] - target[0]) ** 2 + (my_xy[1] - target[1]) ** 2
            if dsq <= GC.ACTION_RADIUS_SQ:
                # Close enough — transition to destroy (allied) or place (none)
                self.launcher_place_substep = 2 if self.launcher_place_road_type == 'allied' else 4
                self.path = None
                self.path_index = 0
                return
            # Hybrid nav toward the target itself; we'll stop via the dsq check above.
            result = self._disruptor_navigate(ct, target)
            if result == 'blocked':
                self.launcher_place_attempted.add(target)
                self.launcher_place_substep = 0
                self.launcher_place_target = None
            return

        if self.launcher_place_substep == 2:
            # Destroy the road
            if self.launcher_place_road_type == 'allied':
                tpos = xy_to_pos(target)
                if ct.can_destroy(tpos):
                    ct.destroy(tpos)
                    info = self.tile_cache.get(target)
                    if info:
                        self.tile_cache[target] = (info[0], None, None, None)
                self.launcher_place_substep = 4  # straight to placement
                return
            # enemy: fire on own tile until destroyed
            if my_xy != target:
                # Drifted off — go back to substep 1
                self.launcher_place_substep = 1
                return
            info = self.tile_cache.get(target)
            if info is None or info[1] is None:
                # Destroyed — need to step off
                self.launcher_place_substep = 3
                return
            if ct.get_action_cooldown() == 0:
                mpos = xy_to_pos(my_xy)
                if self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0]) and ct.can_fire(mpos):
                    ct.fire(mpos)
                    self._ds_event(f"[DS] step8 fire@{my_xy}")
            return

        if self.launcher_place_substep == 3:
            # Enemy-road path: step off before placing
            if my_xy == target:
                self._disruptor_step_off(ct, my_xy)
                return
            self.launcher_place_substep = 4

        if self.launcher_place_substep == 4:
            # Place launcher from adjacent
            dsq = (my_xy[0] - target[0]) ** 2 + (my_xy[1] - target[1]) ** 2
            if dsq == 0:
                # Can't build non-walkable on own tile
                self._disruptor_step_off(ct, my_xy)
                return
            if dsq > GC.ACTION_RADIUS_SQ:
                # Bugnav back toward the launcher tile.
                result = self._disruptor_navigate(ct, target)
                if result == 'blocked':
                    self.launcher_place_attempted.add(target)
                    self.launcher_place_substep = 0
                    self.launcher_place_target = None
                return

            # Validate tile is empty
            info = self.tile_cache.get(target)
            if info is not None and info[1] is not None:
                self.launcher_place_attempted.add(target)
                self.launcher_place_substep = 0
                self.launcher_place_target = None
                return
            tpos = xy_to_pos(target)
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.LAUNCHER_BASE_COST[0])
                    and ct.can_build_launcher(tpos)):
                ct.build_launcher(tpos)
                self._ds_event(f"[DS] step8 launcher at {target}")
                self.launcher_place_attempted.add(target)
                self.launcher_place_substep = 0
                self.launcher_place_target = None
                self.path = None
                self.path_index = 0

    def _disruptor_find_launcher_target(self, my_xy):
        """Pick a core-ring tile that is empty, holds an allied road, or
        holds an enemy road — and is reachable within the threshold."""
        if self.enemy_core_pos is None:
            return None
        my_team = self.my_team_cache
        best = None
        best_len = None
        for xy in self._disruptor_core_ring_tiles():
            if xy in self.launcher_place_attempted:
                continue
            info = self.tile_cache.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if env == Environment.WALL:
                continue
            valid = False
            if bid is None:
                valid = True
            elif etype == EntityType.ROAD:
                valid = True
            if not valid:
                continue
            # Pathfinding heuristic goal: for 'none' or 'allied', a tile
            # adjacent to the ring tile; for 'enemy' the ring tile itself.
            if bid is None or team == my_team:
                probe_goal = self._disruptor_pick_adjacent_goal(my_xy, xy)
                if probe_goal is None:
                    continue
            else:
                probe_goal = xy
            path = self._compute_path(my_xy, probe_goal)
            if path is None:
                continue
            plen = len(path)
            if plen > IGNORE_ENEMY_CONVEYOR_THRESHOLD:
                continue
            if best_len is None or plen < best_len:
                best_len = plen
                best = xy
        return best

    # ------------------------------------------------------------------ #
    #  Builder-side launcher protocol                                     #
    # ------------------------------------------------------------------ #
    #
    # Steps:
    #   1. Find a marker tile within OUR action radius AND within the
    #      launcher's vision. Place marker with the encoded request value.
    #   2. Wait 1 turn so the launcher reads the marker next time it runs.
    #   3. Destroy our marker — this promotes the pending request to active.
    #   4. Pathfind adjacent to the launcher.
    #   5. Wait — the launcher's run_launcher will pick us up and throw us
    #      to launch_target_xy. When we detect we've landed, reset state
    #      and enter launch_post_step.
    # ------------------------------------------------------------------ #

    def _run_launch_protocol(self, ct):
        my_xy = self._my_xy
        lxy = self.launch_launcher_xy
        if lxy is None:
            self._launch_protocol_reset(fail=True)
            return
        # Launcher might have been destroyed meanwhile — validate.
        linfo = self.tile_cache.get(lxy)
        if linfo is None or linfo[1] is None or linfo[3] != self.my_team_cache \
                or linfo[2] != EntityType.LAUNCHER:
            if DEBUG:
                print(f"[DS] protocol: launcher {lxy} gone -> abort", file=sys.stderr)
            self._launch_protocol_reset(fail=True)
            return

        tx, ty = self.launch_target_xy

        if self.launch_protocol_step == 1:
            # Place marker
            marker_xy = self._launch_find_marker_tile(ct, lxy)
            if marker_xy is None:
                # Can't place marker — nav closer to the launcher.
                result = self._disruptor_navigate(ct, lxy)
                if result == 'blocked':
                    self._launch_protocol_reset(fail=True)
                return
            bot_id_mod = ct.get_id() % 10000
            value = LAUNCHER_PROTOCOL_PREFIX + bot_id_mod * 10000 + tx * 100 + ty
            mpos = xy_to_pos(marker_xy)
            if ct.can_place_marker(mpos):
                ct.place_marker(mpos, value)
                self.launch_marker_xy = marker_xy
                self.launch_protocol_step = 2
                self._ds_event(f"[DS] protocol: marker@{marker_xy} val={value}")
            return

        if self.launch_protocol_step == 2:
            # Wait one turn for the launcher to read the marker.
            self.launch_protocol_step = 3
            return

        if self.launch_protocol_step == 3:
            # Destroy our marker to activate the request
            mxy = self.launch_marker_xy
            if mxy is None:
                self._launch_protocol_reset(fail=True)
                return
            mpos = xy_to_pos(mxy)
            dsq = (my_xy[0] - mxy[0]) ** 2 + (my_xy[1] - mxy[1]) ** 2
            if dsq > GC.ACTION_RADIUS_SQ:
                # Drifted — nav back to the marker.
                result = self._disruptor_navigate(ct, mxy)
                if result == 'blocked':
                    self._launch_protocol_reset(fail=True)
                return
            if ct.can_destroy(mpos):
                ct.destroy(mpos)
                self._ds_event("[DS] protocol: marker destroyed")
                self.launch_protocol_step = 4
                self.path = None
                self.path_index = 0
            return

        if self.launch_protocol_step == 4:
            # Be adjacent to the launcher (Chebyshev 1).
            cheb = max(abs(my_xy[0] - lxy[0]), abs(my_xy[1] - lxy[1]))
            if cheb <= 1:
                self.launch_protocol_step = 5
                return
            result = self._disruptor_navigate(ct, lxy)
            if result == 'blocked':
                self._launch_protocol_reset(fail=True)
            return

        if self.launch_protocol_step == 5:
            # Wait for the launcher to throw us. Check landing.
            if my_xy == self.launch_target_xy:
                post = self.launch_post_step
                self._ds_event(f"[DS] protocol: landed@{my_xy} -> step {post}")
                self.disruptor_step = post if post is not None else 2
                self._launch_protocol_reset(fail=False)
                # Attack-on-land: don't waste a turn — fire on the enemy
                # structure we just landed on if conditions allow.
                self._try_attack_on_land(ct)
                return
            # Check if we drifted out of adjacency (launcher died mid-protocol
            # is already handled above). Otherwise, just wait one more turn.
            cheb = max(abs(my_xy[0] - lxy[0]), abs(my_xy[1] - lxy[1]))
            if cheb > 1:
                # We've been displaced somehow — restart protocol.
                self.launch_protocol_step = 4
                self.path = None
                self.path_index = 0
            return

    def _launch_find_marker_tile(self, ct, launcher_xy):
        """Pick a tile that is (a) within our action radius, (b) within the
        launcher's vision radius, and (c) empty/placeable.
        Returns (x,y) or None. Markers go on walkable or empty tiles; the
        engine allows markers on most tiles (HP 1, free).
        """
        my_xy = self._my_xy
        lx, ly = launcher_xy
        max_launcher_r2 = GC.LAUNCHER_VISION_RADIUS_SQ
        best = None
        best_d2 = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cxy = (my_xy[0] + dx, my_xy[1] + dy)
                # Within action radius (r²≤2)
                if dx * dx + dy * dy > GC.ACTION_RADIUS_SQ:
                    continue
                # Within launcher vision
                ld2 = (cxy[0] - lx) ** 2 + (cxy[1] - ly) ** 2
                if ld2 > max_launcher_r2:
                    continue
                # Must be placeable: empty (no building) and not a wall
                if cxy in self.known_walls:
                    continue
                info = self.tile_cache.get(cxy)
                if info is None:
                    continue
                env, bid, etype, team = info
                if env == Environment.WALL:
                    continue
                # Placing a marker over an existing marker is allowed ONLY
                # if it's allied; but to keep this simple, only accept
                # empty tiles.
                if bid is not None:
                    continue
                # Prefer tiles close to the launcher
                if best_d2 is None or ld2 < best_d2:
                    best_d2 = ld2
                    best = cxy
        return best

    # ------------------------------------------------------------------ #
    #  Enemy launcher detection + Case A response (Part 3, Commit A)      #
    # ------------------------------------------------------------------ #

    def _disruptor_recompute_danger_zone(self, ct):
        """Rebuild self.dangerous_sentinel_tiles from current vision.

        A tile is dangerous if it sits inside the attackable cone of an
        enemy SENTINEL that's cardinally adjacent to a HARVESTER (any
        team — sentinels accept ammo from either side's harvester, so
        an enemy sentinel sitting next to OUR harvester is just as
        deadly as one next to its own). Recomputed every turn — cheap:
        iterate vision (~70), per enemy sentinel pay one cardinal
        check + one get_attackable_tiles_from FFI.
        """
        zone = set()
        my_team = self.my_team_cache
        tc = self.tile_cache
        for xy in self._last_vision_set:
            info = tc.get(xy)
            if info is None:
                continue
            bid, etype, team = info[1], info[2], info[3]
            if (bid is None or team == my_team or team is None
                    or etype != EntityType.SENTINEL):
                continue
            # Must be cardinally adjacent to a HARVESTER (any team).
            fed = False
            for dx, dy in _CARDINAL_OFFSETS:
                ne = tc.get((xy[0] + dx, xy[1] + dy))
                if (ne is not None and ne[1] is not None
                        and ne[2] == EntityType.HARVESTER):
                    fed = True
                    break
            if not fed:
                continue
            try:
                t_dir = ct.get_direction(bid)
            except Exception:
                continue
            try:
                tiles = ct.get_attackable_tiles_from(
                    xy_to_pos(xy), t_dir, EntityType.SENTINEL)
            except Exception:
                continue
            for t in tiles:
                zone.add((t.x, t.y))
        self.dangerous_sentinel_tiles = zone

    def _disruptor_below_danger_hp(self, ct):
        """True iff this disruptor's HP is below DISRUPTOR_DANGER_HP_FRAC
        of max — the threshold where dangerous sentinel tiles become
        impassable rather than just heuristically penalised."""
        try:
            cur = ct.get_hp()
        except Exception:
            return False
        return cur < int(GC.BUILDER_BOT_MAX_HP * DISRUPTOR_DANGER_HP_FRAC)

    def _disruptor_flee_danger_zone(self, ct, my_xy):
        """We're standing inside an enemy sentinel cone AND below the
        danger HP threshold. Override normal step dispatch and walk
        toward the nearest safe tile (one that's NOT in the danger
        zone). The end-of-turn self-heal in main.py runs unconditionally
        for damaged disruptors, so the heal happens in parallel as we
        walk out. Returns True if a flee step was issued (caller should
        skip normal step dispatch this turn).
        """
        if not self.dangerous_sentinel_tiles:
            return False
        if my_xy not in self.dangerous_sentinel_tiles:
            return False
        # Pick the closest tile in 8-Chebyshev radius that's safe and
        # passable from the greedy's perspective.
        best = None
        best_d2 = None
        danger = self.dangerous_sentinel_tiles
        for r in range(1, 7):
            # Ring scan, smallest radius first — first hit wins.
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    nxy = (my_xy[0] + dx, my_xy[1] + dy)
                    if nxy in danger:
                        continue
                    if nxy in self.known_walls:
                        continue
                    info = self.tile_cache.get(nxy)
                    if info is not None and info[0] == Environment.WALL:
                        continue
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = nxy
            if best is not None:
                break
        if best is None:
            return False
        self._ds_event(f"[DS] FLEE danger@{my_xy} → {best}")
        # _disruptor_navigate respects danger tiles (low-HP is
        # impassable), so the path it builds will route around the
        # cone rather than back into it.
        self._disruptor_navigate(ct, best)
        return True

    def _disruptor_turret_damage_check(self, ct, my_xy):
        """HP drop between turns ⇒ we're under fire. Abandon every
        active attack state unconditionally, then look for visible
        enemy turrets whose attackable cone covers us — for each such
        turret, blacklist every tile in its current-rotation cone so
        future attack pickers route around it.

        `disrupt_prev_hp` is updated at the END of this call so the
        comparison always uses last turn's snapshot. The first turn
        of a disruptor's life has prev_hp=None → no detection,
        just seed the snapshot.

        Abandon happens even if no visible turret is found — a
        sentinel can shoot from up to 6 tiles away while our vision
        only reaches ~4.5, so the source isn't always observable.
        """
        try:
            cur_hp = ct.get_hp()
        except Exception:
            cur_hp = None
        prev_hp = self.disrupt_prev_hp
        self.disrupt_prev_hp = cur_hp
        if prev_hp is None or cur_hp is None or cur_hp >= prev_hp:
            return

        # 1) Abandon every active attack state. The bot was firing
        # while exposed to a turret cone — finishing the attack means
        # taking more damage. Each branch also blacklists its target
        # so the picker doesn't immediately re-select it.
        abandoned = []
        if self.disrupt_target is not None:
            abandoned.append(f"disrupt={self.disrupt_target}")
            self._disrupt_reset(abandon_reason="turret fire")
        if self.disrupt_chain_block_target is not None:
            abandoned.append(
                f"chain-block={self.disrupt_chain_block_target}")
            self.disrupt_blacklist.add(self.disrupt_chain_block_target)
            self.disrupt_chain_block_target = None
            self.disrupt_chain_block_source = None
        if self.disrupt_splitter_target is not None:
            placement = self.disrupt_splitter_target[0]
            abandoned.append(f"splitter-sent={placement}")
            self.disrupt_blacklist.add(placement)
            self.disrupt_splitter_target = None
        # Step 3 intercept: only the dirty (feeder-attack) substep
        # involves firing on a tile — substep 0/2/3 are scan / step-off
        # / sentinel-place and don't expose the bot to repeated turret
        # fire. Abandon only when actively firing.
        if (self.intercept_substep == 1
                and not self.intercept_clean_case
                and self.intercept_target_conveyor is not None):
            abandoned.append(
                f"intercept={self.intercept_target_conveyor}")
            self.intercept_feeder_blacklist.add(
                self.intercept_target_conveyor)
            self._disruptor_intercept_reset_and_resume()
        self._attack_heal_reset()

        # 2) Look for visible turret(s) covering us → blacklist cones.
        my_team = self.my_team_cache
        tc = self.tile_cache
        cones_blacklisted = 0
        for xy in self._last_vision_set:
            info = tc.get(xy)
            if info is None:
                continue
            _env, bid, etype, team = info
            if (bid is None or team == my_team or team is None
                    or etype not in _ENEMY_TURRET_TYPES):
                continue
            try:
                t_dir = ct.get_direction(bid)
            except Exception:
                continue
            try:
                tiles = ct.get_attackable_tiles_from(
                    xy_to_pos(xy), t_dir, etype)
            except Exception:
                continue
            tile_xys = [(t.x, t.y) for t in tiles]
            if my_xy not in tile_xys:
                continue
            cones_blacklisted += 1
            for txy in tile_xys:
                self.disrupt_attack_blacklist.add(txy)

        if abandoned or cones_blacklisted:
            self._ds_event(
                f"[DS] turret-damage hp {prev_hp}->{cur_hp} "
                f"abandon=[{','.join(abandoned) or 'none'}] "
                f"cones={cones_blacklisted} "
                f"blacklist_size={len(self.disrupt_attack_blacklist)}")

    def _detect_enemy_launcher_near(self, prev_xy):
        """Check the 8 neighbors of prev_xy for an enemy LAUNCHER. Returns
        (x,y) or None. Launchers pick up bots within r²≤2 (Chebyshev 1),
        so diagonal-adjacent counts just like cardinal.
        """
        my_team = self.my_team_cache
        px, py = prev_xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (px + dx, py + dy)
                info = self.tile_cache.get(nxy)
                if info is None:
                    continue
                _, bid, etype, team = info
                if bid is None or team == my_team or team is None:
                    continue
                if etype == EntityType.LAUNCHER:
                    return nxy
        return None

    def _handle_enemy_launch(self, ct, launcher_xy):
        """Record the launcher footprint as blocked, abandon the current
        path. Then try attack-on-land / build-on-land. If we were on an
        attack/build task and weren't able to complete it on-land, start a
        Case-B counter (use an allied launcher to reach the original target).
        """
        self.launched_this_turn = True
        self._acted_on_land = False
        self.enemy_launcher_positions.add(launcher_xy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                self.blocked_launcher_tiles.add(
                    (launcher_xy[0] + dx, launcher_xy[1] + dy))
        self.path = None
        self.path_index = 0
        self._ds_event(f"[DS] ENEMY LAUNCH: launcher={launcher_xy} "
                       f"from={self.prev_intended_pos} to={self._my_xy}")

        # --- Attack-on-land / build-on-land ---
        # These only make sense if we were on an attack/build task. Also
        # check if we were running the launch protocol (our own counter)
        # and just landed on the intended target — in that case Step 5 of
        # _run_launch_protocol handles the transition; don't double-act.
        if self.case_b_active or self.launch_protocol_step > 0:
            return

        task = self._current_attack_task_target()
        if task is None:
            return  # Pure pathfinding (Case A) — nothing more to do.
        task_target, task_kind = task

        # Attack-on-land: we landed on the attack target
        if task_kind == 'attack' and self._my_xy == task_target:
            if self._try_attack_on_land(ct):
                return

        # Build-on-land: we landed on or adjacent to the build target
        if task_kind.startswith('build_'):
            dsq = ((self._my_xy[0] - task_target[0]) ** 2
                   + (self._my_xy[1] - task_target[1]) ** 2)
            if dsq <= GC.ACTION_RADIUS_SQ and dsq > 0:
                if self._try_build_on_land(ct, task_target, task_kind):
                    return

        # Couldn't act on-land — start Case B counter.
        self._maybe_start_case_b(ct, task_target, task_kind)

    def _try_attack_on_land(self, ct):
        """Fire at the enemy infrastructure we're standing on. Returns True
        if we spent the action."""
        if ct.get_action_cooldown() != 0:
            return False
        my_xy = self._my_xy
        info = self.tile_cache.get(my_xy)
        if info is None or info[1] is None or info[3] == self.my_team_cache:
            return False
        # Fire only on destroyable enemy structures. Harvesters + foundries
        # feed our downstream sentinels — do not destroy.
        if info[2] not in (EntityType.CONVEYOR, EntityType.BRIDGE,
                           EntityType.SPLITTER, EntityType.ROAD):
            return False
        mpos = xy_to_pos(my_xy)
        if (self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0])
                and ct.can_fire(mpos)):
            ct.fire(mpos)
            self._acted_on_land = True
            self._ds_event(f"[DS] ATTACK-ON-LAND@{my_xy}")
            return True
        return False

    def _try_build_on_land(self, ct, target_xy, task_kind):
        """Place the turret/launcher on target_xy from our (adjacent) position."""
        if ct.get_action_cooldown() != 0:
            return False
        info = self.tile_cache.get(target_xy)
        if info is not None and info[1] is not None:
            return False   # someone built here
        ec = self.enemy_core_pos or self.enemy_core_guess
        tpos = xy_to_pos(target_xy)

        if task_kind == 'build_gunner':
            if ec is None:
                return False
            d = cardinal_direction_between(target_xy, ec)
            if (self._can_spend(ct, GC.GUNNER_BASE_COST[0])
                    and ct.can_build_gunner(tpos, d)):
                ct.build_gunner(tpos, d)
                self._acted_on_land = True
                self._ds_event(f"[DS] BUILD-ON-LAND gunner@{target_xy} face={d.value}")
                return True
        elif task_kind == 'build_sentinel':
            if ec is None:
                return False
            d = cardinal_direction_between(target_xy, ec)
            if (self._can_spend(ct, GC.SENTINEL_BASE_COST[0])
                    and ct.can_build_sentinel(tpos, d)):
                ct.build_sentinel(tpos, d)
                self._acted_on_land = True
                self._ds_event(f"[DS] BUILD-ON-LAND sentinel@{target_xy} face={d.value}")
                return True
        elif task_kind == 'build_launcher':
            if (self._can_spend(ct, GC.LAUNCHER_BASE_COST[0])
                    and ct.can_build_launcher(tpos)):
                ct.build_launcher(tpos)
                self._acted_on_land = True
                self._ds_event(f"[DS] BUILD-ON-LAND launcher@{target_xy}")
                return True
        elif task_kind == 'build_barrier':
            if (self._can_spend(ct, GC.BARRIER_BASE_COST[0])
                    and ct.can_build_barrier(tpos)):
                ct.build_barrier(tpos)
                self._acted_on_land = True
                self._ds_event(f"[DS] BUILD-ON-LAND barrier@{target_xy}")
                return True
        return False

    def _current_attack_task_target(self):
        """Return (target_xy, kind) for the current attack/build task, or None.
        kind is 'attack'|'build_gunner'|'build_sentinel'|'build_launcher'.
        """
        if self.disruptor_step == 3:
            if self.intercept_substep == 1:
                if self.intercept_clean_case and self.intercept_sentinel_tile:
                    return (self.intercept_sentinel_tile, 'build_sentinel')
                if not self.intercept_clean_case and self.intercept_target_conveyor:
                    return (self.intercept_target_conveyor, 'attack')
            if self.intercept_substep == 2 and self.intercept_destroyed_tile:
                return (self.intercept_destroyed_tile, 'build_sentinel')
        if self.disruptor_step == 5 and self.disrupt_target is not None:
            if self.disrupt_target_destroyed:
                return (self.disrupt_target, 'build_barrier')
            return (self.disrupt_target, 'attack')
        if self.disruptor_step == 8:
            if self.launcher_place_substep == 4 and self.launcher_place_target:
                return (self.launcher_place_target, 'build_launcher')
            if (self.launcher_place_substep in (1, 2)
                    and self.launcher_place_road_type == 'enemy'
                    and self.launcher_place_target):
                return (self.launcher_place_target, 'attack')
        return None

    # ------------------------------------------------------------------ #
    #  Case B: use or build an allied launcher to reach the target again  #
    # ------------------------------------------------------------------ #

    def _maybe_start_case_b(self, ct, task_target, task_kind):
        """If we can reach task_target via an allied launcher, enter Case B."""
        # Prefer existing allied launcher in range
        launcher_xy = self._find_allied_launcher_near(task_target)
        if launcher_xy is not None:
            self._start_case_b_use_launcher(launcher_xy, task_target, task_kind)
            return
        # Otherwise find a build site
        build_site = self._find_counter_launcher_site(task_target)
        if build_site is not None:
            self._start_case_b_build_launcher(build_site, task_target, task_kind)
            return
        if DEBUG:
            print(f"[DS] Case B: no launcher option for {task_target}; abandoning",
                  file=sys.stderr)
        self._abandon_current_attack_target()

    def _start_case_b_use_launcher(self, launcher_xy, task_target, task_kind):
        self.case_b_active = True
        self.case_b_target = task_target
        self.case_b_target_kind = task_kind
        self.case_b_original_step = self.disruptor_step
        self.case_b_phase = 'use_launcher'
        self.case_b_build_site = None
        # Configure the launch protocol. After landing, _run_launch_protocol
        # normally sets disruptor_step and re-enters the assault. For Case B
        # we set launch_post_step so the protocol's own landing block does
        # the right thing (see _run_launch_protocol's step-5 branch below).
        self.launch_target_xy = task_target
        self.launch_launcher_xy = launcher_xy
        self.launch_post_step = self.case_b_original_step
        self.launch_protocol_step = 1
        self.path = None
        self.path_index = 0
        self._ds_event(f"[DS] Case B USE launcher={launcher_xy} target={task_target} "
                       f"kind={task_kind}")

    def _start_case_b_build_launcher(self, build_site, task_target, task_kind):
        self.case_b_active = True
        self.case_b_target = task_target
        self.case_b_target_kind = task_kind
        self.case_b_original_step = self.disruptor_step
        self.case_b_phase = 'build_launcher'
        self.case_b_build_site = build_site
        # Seed Step 8 state for reuse. The site is expected to be empty;
        # if the cache disagrees, Step 8's own planner will bail.
        self.launcher_place_target = build_site
        info = self.tile_cache.get(build_site) or (Environment.EMPTY, None, None, None)
        _, bid, etype, team = info
        if bid is None:
            self.launcher_place_road_type = 'none'
        elif etype == EntityType.ROAD and team == self.my_team_cache:
            self.launcher_place_road_type = 'allied'
        elif etype == EntityType.ROAD and team != self.my_team_cache:
            self.launcher_place_road_type = 'enemy'
        else:
            self.launcher_place_road_type = 'none'
        self.launcher_place_substep = 1
        self.path = None
        self.path_index = 0
        self._ds_event(f"[DS] Case B BUILD launcher@{build_site} target={task_target} "
                       f"kind={task_kind}")

    def _run_case_b_counter(self, ct):
        """Dispatch for the active Case B state. Called from _run_disruptor
        BEFORE normal step dispatch when case_b_active is True."""
        if not self.case_b_active:
            return

        # If the protocol succeeded (or failed) and reset itself, we're done.
        if self.case_b_phase == 'use_launcher':
            if self.launch_protocol_step == 0:
                # Done (landed or aborted). If we landed on target, try to
                # attack/build now (same turn as landing). Then finish Case B.
                self._case_b_finish_after_land(ct)
                return
            self._run_launch_protocol(ct)
            return

        if self.case_b_phase == 'build_launcher':
            # Drive Step 8 machinery. When launcher_place_target clears back
            # to None with substep 0, the launcher was built. Transition to
            # use_launcher phase using case_b_build_site as the new launcher.
            self._disruptor_step8_place_launchers(ct)
            if self.launcher_place_target is None and self.launcher_place_substep == 0:
                # Launcher placed — switch to use_launcher phase
                launcher_xy = self.case_b_build_site
                self.case_b_phase = 'use_launcher'
                self.launch_target_xy = self.case_b_target
                self.launch_launcher_xy = launcher_xy
                self.launch_post_step = self.case_b_original_step
                self.launch_protocol_step = 1
                self.path = None
                self.path_index = 0
                self._ds_event(f"[DS] Case B launcher built@{launcher_xy}; protocol start")

    def _case_b_finish_after_land(self, ct):
        """Protocol finished. If we landed on target, try attack/build now.
        Then clear Case B state."""
        my_xy = self._my_xy
        if self.case_b_target is not None and my_xy == self.case_b_target:
            if self.case_b_target_kind == 'attack':
                self._try_attack_on_land(ct)
            # build kinds would need us to be ADJACENT (target is empty
            # tile we want to build ON); if we landed ON the target, that's
            # the wrong spot for a build task, so no build-on-land.
        self.disruptor_step = self.case_b_original_step
        self.case_b_active = False
        self.case_b_target = None
        self.case_b_target_kind = None
        self.case_b_original_step = None
        self.case_b_phase = None
        self.case_b_build_site = None

    def _find_allied_launcher_near(self, target_xy):
        """Allied LAUNCHER with squared-distance to target ≤ LAUNCHER_VISION_RADIUS_SQ.

        Bounded offset scan (|d|≤5 is enough for r²≤26) — avoids iterating
        the entire tile_cache on every call, which cost ~750μs on 50x50 maps.
        """
        max_r2 = GC.LAUNCHER_VISION_RADIUS_SQ
        my_team = self.my_team_cache
        tx, ty = target_xy
        best = None
        best_d2 = None
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                d2 = dx * dx + dy * dy
                if d2 > max_r2:
                    continue
                xy = (tx + dx, ty + dy)
                info = self.tile_cache.get(xy)
                if info is None:
                    continue
                _, bid, etype, team = info
                if bid is None or team != my_team:
                    continue
                if etype != EntityType.LAUNCHER:
                    continue
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = xy
        return best

    def _find_counter_launcher_site(self, target_xy):
        """Empty tile within LAUNCHER_VISION_RADIUS_SQ of target_xy that isn't
        blocked. Picks the tile closest to the disruptor by squared Euclidean
        distance.

        Previously iterated the entire tile_cache and ran A* for every
        candidate — up to 80 A* calls × ~320μs ≈ 25ms per invocation, which
        blew the 2ms budget on ladder whenever a bot got launched during an
        active attack task. A bounded offset scan + distance heuristic is
        <100μs. We rely on the downstream marker-placement / navigate steps
        to bail if the chosen site turns out to be unreachable.
        """
        max_r2 = GC.LAUNCHER_VISION_RADIUS_SQ
        my_xy = self._my_xy
        tx, ty = target_xy
        best = None
        best_d2 = None
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx * dx + dy * dy > max_r2:
                    continue
                xy = (tx + dx, ty + dy)
                if xy in self.blocked_launcher_tiles:
                    continue
                info = self.tile_cache.get(xy)
                if info is None:
                    continue
                env, bid, etype, team = info
                if env == Environment.WALL or bid is not None:
                    continue
                d2 = (xy[0] - my_xy[0]) ** 2 + (xy[1] - my_xy[1]) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best = xy
        return best

    def _disruptor_find_harass_target(self, my_xy):
        """Nearest frontier tile, biased toward the enemy core so we keep
        pushing deeper into enemy territory. Returns (x,y) or None."""
        if not self.frontier_cache:
            return None
        ec = self.enemy_core_pos or self.enemy_core_guess
        best = None
        best_score = None
        for fxy in self.frontier_cache:
            if fxy in self.known_walls:
                continue
            if fxy in self.blocked_launcher_tiles:
                continue
            d_self = euclidean_dist_sq(my_xy, fxy)
            d_enemy = euclidean_dist_sq(fxy, ec) if ec is not None else 0
            # Strong bias toward tiles close to the enemy core.
            score = d_self + 2 * d_enemy
            if best_score is None or score < best_score:
                best_score = score
                best = fxy
        return best

    def _abandon_current_attack_target(self):
        """Clear the current step's target and blacklist so the planner
        re-picks. Returns to the step's planner substep."""
        if self.disruptor_step == 3:
            self.intercept_target_harvester = None
            self.intercept_target_conveyor = None
            self.intercept_substep = 0
            self.disruptor_step = 2
        elif self.disruptor_step == 5:
            # Harass: abandon the current disruption; harass will re-pick
            # after DISRUPT_INTERVAL idle turns.
            self._disrupt_reset(abandon_reason="abandon (launcher)")
        elif self.disruptor_step == 8:
            # Step 8 is only reachable from Case B's build_launcher phase.
            if self.launcher_place_target is not None:
                self.launcher_place_attempted.add(self.launcher_place_target)
            self.launcher_place_target = None
            self.launcher_place_substep = 0
        self.path = None
        self.path_index = 0

    def _launch_protocol_reset(self, fail):
        if DEBUG and fail:
            print(f"[DS] protocol: FAIL reset", file=sys.stderr)
        self.launch_protocol_step = 0
        self.launch_target_xy = None
        self.launch_launcher_xy = None
        self.launch_marker_xy = None
        self.launch_post_step = None
        self.path = None
        self.path_index = 0
        if fail:
            # Return to harass — it will re-pick targets after DISRUPT_INTERVAL.
            self.disruptor_step = 5

    # --- Supply-flow check (reused by Step 5a disruption target selection) -

    def _has_supply_flow(self, ct, target_xy):
        """A target is only valid if at least one enemy conveyor/splitter
        cardinally points at it OR an enemy bridge targets it."""
        my_team = self.my_team_cache
        # Cardinally-adjacent enemy CONVEYOR/SPLITTER whose output is target_xy.
        for dx, dy in _CARDINAL_OFFSETS:
            nxy = (target_xy[0] + dx, target_xy[1] + dy)
            info = self.tile_cache.get(nxy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team == my_team or team is None:
                continue
            if etype not in (EntityType.CONVEYOR, EntityType.SPLITTER,
                             EntityType.ARMOURED_CONVEYOR):
                continue
            direction = self._enemy_dir_cache.get(bid)
            if direction is None:
                try:
                    direction = ct.get_direction(bid)
                except Exception:
                    continue
                self._enemy_dir_cache[bid] = direction
            out = self._tile_in_direction(nxy, direction)
            if out == target_xy:
                return True
        # Enemy bridges targeting target_xy. BRIDGE_TARGET_RADIUS_SQ = 9
        # caps the bridge-to-target distance at sqrt(9) = 3 tiles, so a
        # 7x7 window around target_xy covers every possible source.
        # Iterating the full tile_cache here was the dominant TLE source
        # for `_find_disruption_target` on mature maps with many enemy
        # bridges in vision (~2500 dict lookups per candidate × N
        # candidates per turn).
        tx, ty = target_xy
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nxy = (tx + dx, ty + dy)
                info = self.tile_cache.get(nxy)
                if info is None:
                    continue
                env, bid, etype, team = info
                if bid is None or team == my_team or team is None:
                    continue
                if etype != EntityType.BRIDGE:
                    continue
                tgt_xy = self._enemy_bridge_target_cache.get(bid)
                if tgt_xy is None:
                    try:
                        tgt = ct.get_bridge_target(bid)
                    except Exception:
                        continue
                    tgt_xy = (tgt.x, tgt.y)
                    self._enemy_bridge_target_cache[bid] = tgt_xy
                if tgt_xy == target_xy:
                    return True
        return False

    def _tile_in_direction(self, xy, direction):
        dx, dy = DIRECTION_DELTAS.get(direction, (0, 0))
        return (xy[0] + dx, xy[1] + dy)

    # --- Step 5: harass mode ---------------------------------------------

    def _disruptor_step5_harass(self, ct):
        """Harass mode: hunt harvesters + every DISRUPT_INTERVAL idle turns,
        destroy an enemy conveyor and drop a barrier on the destroyed tile.
        Never leaves the HARASS_RADIUS Euclidean zone around the enemy core.
        """
        my_xy = self._my_xy
        if self.step10_entry_turn is None:
            self.step10_entry_turn = ct.get_current_round()
            self._ds_event(f"[DS] HARASS enter at ({my_xy[0]},{my_xy[1]})")

        # A. Already mid-disruption → drive the attack → barrier flow.
        if self.disrupt_target is not None:
            self._continue_disruption(ct)
            return

        # B. Harvester hunt (no detour limit inside harass mode).
        pick = self._disruptor_pick_intercept_harass(my_xy)
        if pick is not None:
            (self.intercept_target_harvester,
             self.intercept_target_conveyor,
             self.intercept_clean_case,
             self.intercept_sentinel_tile) = pick
            self.intercept_substep = 0
            self.intercept_sentinel_dir = None
            self.intercept_destroyed_tile = (
                self.intercept_target_conveyor
                if not self.intercept_clean_case else None)
            self.intercept_post_step = 5
            self.disruptor_step = 3
            self.path = None
            self.path_index = 0
            self._ds_event(f"[DS] harass intercept {self.intercept_target_harvester}")
            return

        # B2. Chain-extension intercept (enemy conveyor/bridge pointing
        # at empty/road) — same per-target detour budget as Step 2.
        chain_pick = self._disruptor_pick_chain_extension_intercept(
            my_xy, INTERCEPT_DETOUR)
        if chain_pick is not None:
            self.disrupt_chain_block_target = chain_pick[0]
            self.disrupt_chain_block_source = chain_pick[1]
            self._ds_event(
                f"[DS] harass chain-block {chain_pick[0]} src={chain_pick[1]}")
            result = self._disruptor_block_chain_extension(
                ct, chain_pick[0], chain_pick[1])
            if result != 'abandon':
                return
            self.disrupt_chain_block_target = None
            self.disrupt_chain_block_source = None

        # B2b. Splitter-sentinel: place a sentinel adjacent to an enemy
        # splitter (excluding its back tile).
        split_pick = self._disruptor_pick_splitter_sentinel(
            my_xy, INTERCEPT_DETOUR)
        if split_pick is not None:
            self.disrupt_splitter_target = split_pick
            self._ds_event(
                f"[DS] harass splitter-sent placement={split_pick[0]} "
                f"split={split_pick[1]}")
            result = self._disruptor_block_splitter_sentinel(
                ct, split_pick[0], split_pick[1])
            if result != 'abandon':
                return
            self.disrupt_splitter_target = None

        # B3. Opportunistic ore-block: drop a barrier on a nearby empty
        # ore so the opponent can't harvest it. Same per-target detour
        # budget as the Step-2 picker.
        ore_xy = self._disruptor_pick_ore_block(my_xy, INTERCEPT_DETOUR)
        if ore_xy is not None:
            self.disrupt_ore_block_target = ore_xy
            self._ds_event(f"[DS] harass ore-block {ore_xy}")
            if self._disruptor_block_ore(ct, ore_xy) != 'abandon':
                return
            self.disrupt_ore_block_target = None

        # C. 8-turn counter → pick + start a disruption cycle.
        self.disrupt_counter += 1
        if self.disrupt_counter >= DISRUPT_INTERVAL:
            target = self._find_disruption_target(my_xy)
            if target is not None:
                self.disrupt_target = target
                self.disrupt_heal_start_hp = None
                self.disrupt_fires_in_window = 0
                self._ds_event(f"[DS] disrupt target {target}")
                self._continue_disruption(ct)
                return
            # No valid target this cycle — swallow the counter, keep exploring.
            self.disrupt_counter = 0

        # D. Idle → explore inside the harass radius.
        self._harass_explore(ct, my_xy)

    # ------------------------------------------------------------------ #
    #  Step 5 helpers                                                     #
    # ------------------------------------------------------------------ #

    def _disruptor_pick_intercept_harass(self, my_xy):
        """Harvester intercept for harass mode. Differences from the
        Step-2 picker:
          - no INTERCEPT_DETOUR path-length ceiling (accept any reachable),
          - restrict candidates to harvesters inside the harass radius of
            the enemy core.
        Returns (harvester_xy, feeder_or_None, clean_bool, sentinel_tile_or_None).
        """
        ec = self.enemy_core_pos or self.enemy_core_guess
        if ec is None:
            return None
        r2 = HARASS_RADIUS * HARASS_RADIUS
        best = None
        best_len = None
        my_team = self.my_team_cache

        for xy in self._last_vision_set:
            if xy in self.disrupt_attack_blacklist:
                continue
            info = self.tile_cache.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if etype != EntityType.HARVESTER:
                continue
            if team == my_team or team is None:
                continue
            if env != Environment.ORE_TITANIUM:
                continue
            dxr = xy[0] - ec[0]
            dyr = xy[1] - ec[1]
            if dxr * dxr + dyr * dyr > r2:
                continue
            # Skip harvesters whose defensive ring already has 2+
            # allied sentinels — adding more would oversaturate.
            if self._harvester_at_sentinel_cap(xy):
                continue
            empty = self._find_intercept_sentinel_tile(xy, my_xy)
            feeder = None if empty is not None else self._disruptor_find_feeder(xy)
            if empty is None and feeder is None:
                continue
            if feeder is not None and feeder in self.intercept_feeder_blacklist:
                continue
            # Same picker-budget cap as Step-2 intercept. The harass
            # version has no INTERCEPT_DETOUR ceiling, so without a
            # node cap a row of unreachable harvesters in late-game
            # vision can each burn a full 2000-node greedy.
            path = self._disruptor_greedy_path(my_xy, xy, max_nodes=250)
            if path is None:
                continue
            plen = len(path)
            if best_len is not None and plen >= best_len:
                continue
            best_len = plen
            if empty is not None:
                best = (xy, None, True, empty)
            else:
                best = (xy, feeder, False, None)
        return best

    def _enemy_turret_supply_chains(self, ct):
        """Return a set of (x,y) holding enemy CONVEYOR / SPLITTER /
        ARMOURED_CONVEYOR / BRIDGE tiles whose supply lands on one of
        OUR allied turrets (gunner / sentinel / breach), transitively.
        Only considers tiles in current vision — the disruption picker
        can't see past that anyway. Recomputed every call so a chain
        the enemy re-routes away from our turret immediately becomes
        disruptable again.

        Algorithm:
          1. Collect allied turret tiles in vision.
          2. Build forward map src_xy → output_xy for every enemy
             transport in vision.
          3. Mark every src whose output sits on an allied turret as
             "feeding a turret"; reverse-BFS to mark upstream tiles
             that feed those.
        """
        my_team = self.my_team_cache
        tc = self.tile_cache
        vision = self._last_vision_set

        # Step 1: allied turret tiles.
        turret_tiles = set()
        for xy in vision:
            info = tc.get(xy)
            if info is None:
                continue
            bid, etype, team = info[1], info[2], info[3]
            if (bid is not None and team == my_team
                    and etype in (EntityType.GUNNER,
                                  EntityType.SENTINEL,
                                  EntityType.BREACH)):
                turret_tiles.add(xy)
        if not turret_tiles:
            return set()

        # Step 2: forward map of enemy transports.
        forward = {}
        for xy in vision:
            info = tc.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team == my_team or team is None:
                continue
            if etype in (EntityType.CONVEYOR, EntityType.SPLITTER,
                         EntityType.ARMOURED_CONVEYOR):
                d = self._enemy_dir_cache.get(bid)
                if d is None:
                    try:
                        d = ct.get_direction(bid)
                    except Exception:
                        continue
                    self._enemy_dir_cache[bid] = d
                forward[xy] = self._tile_in_direction(xy, d)
            elif etype == EntityType.BRIDGE:
                tgt_xy = self._enemy_bridge_target_cache.get(bid)
                if tgt_xy is None:
                    try:
                        tgt = ct.get_bridge_target(bid)
                    except Exception:
                        continue
                    tgt_xy = (tgt.x, tgt.y)
                    self._enemy_bridge_target_cache[bid] = tgt_xy
                forward[xy] = tgt_xy

        # Step 3: BFS upstream from anyone feeding our turret.
        protected = set()
        # Reverse map: dst → list[src].
        reverse = {}
        for src_xy, dst_xy in forward.items():
            reverse.setdefault(dst_xy, []).append(src_xy)
        queue = []
        for src_xy, dst_xy in forward.items():
            if dst_xy in turret_tiles:
                if src_xy not in protected:
                    protected.add(src_xy)
                    queue.append(src_xy)
        while queue:
            node = queue.pop()
            for upstream in reverse.get(node, ()):
                if upstream not in protected:
                    protected.add(upstream)
                    queue.append(upstream)
        return protected

    def _find_disruption_target(self, my_xy):
        """Closest-by-Euclidean-distance enemy CONVEYOR/BRIDGE/SPLITTER
        inside the harass radius of the enemy core with a supply feeder.
        Pathfinding is then run ONCE on the winning candidate — running
        A*/greedy for every conveyor in vision was TLEing on ladder when
        10+ enemy conveyors clustered near the core. Returns (x,y) or None.

        Excludes any enemy transport that's part of a chain feeding one
        of OUR allied turrets (gunner/sentinel/breach) — destroying it
        would cut off our own ammo supply. Recomputed every call from
        current vision, so as soon as the enemy redirects the chain
        away from our turret, the tiles become valid targets again.
        """
        ec = self.enemy_core_pos or self.enemy_core_guess
        if ec is None:
            return None
        r2 = HARASS_RADIUS * HARASS_RADIUS
        my_team = self.my_team_cache
        protected = self._enemy_turret_supply_chains(self._ct)

        # Step 1: filter + pick the Euclidean-nearest candidate.
        # Only cheap checks here (no pathfinding).
        best_cand = None
        best_d2 = None
        for xy in self._last_vision_set:
            if xy in self.disrupt_blacklist:
                continue
            if xy in self.disrupt_attack_blacklist:
                continue
            if xy in protected:
                continue
            info = self.tile_cache.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team == my_team or team is None:
                continue
            if etype not in (EntityType.CONVEYOR, EntityType.BRIDGE,
                             EntityType.SPLITTER):
                continue
            if etype == EntityType.ARMOURED_CONVEYOR:
                continue
            dxr = xy[0] - ec[0]
            dyr = xy[1] - ec[1]
            if dxr * dxr + dyr * dyr > r2:
                continue
            if not self._has_supply_flow(self._ct, xy):
                continue
            d2 = (xy[0] - my_xy[0]) ** 2 + (xy[1] - my_xy[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_cand = xy

        if best_cand is None:
            return None

        # Step 2: one pathfind. If the winner isn't reachable within the
        # path-length threshold, we drop the cycle — the Euclidean picker
        # already chose the nearest by straight-line distance, so any
        # other reachable candidate would be even further around obstacles.
        # Tight node cap matches the path-length threshold (~6 tiles).
        path = self._disruptor_greedy_path(my_xy, best_cand, max_nodes=150)
        if path is None or len(path) > IGNORE_ENEMY_CONVEYOR_THRESHOLD:
            return None
        return best_cand

    def _disruptor_try_anti_heal_launcher(self, ct, my_xy):
        """Try to plant a launcher next to the enemy builder bot that's
        healing our disrupt target. Returns True if a launcher was
        placed this turn.

        Called from the heal-check path after HEALED_DETECTION_TURNS
        fires failed to lower the target's HP. Geometry:
          1. Find an enemy BUILDER_BOT cardinally/diagonally adjacent
             to my_xy (this is the bot reaching over to heal).
          2. Pick an empty / allied-road tile that is BOTH adjacent to
             that bot AND inside our action radius (dsq ≤ 8 from us).
          3. Destroy the road if needed (free), then build the launcher.
        Bypasses the team-wide MIN_TITANIUM reserve gate — same logic
        as chain-block sentinels: anti-heal launchers are tactically
        valuable enough to spend into the reserve.
        """
        if ct.get_action_cooldown() != 0:
            return False
        my_team = self.my_team_cache
        bpc = self.bot_pos_cache
        tc = self.tile_cache

        # Step 1: nearest enemy builder bot adjacent to us.
        enemy_bot_xy = None
        best_d2 = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (my_xy[0] + dx, my_xy[1] + dy)
                entry = bpc.get(nxy)
                if entry is None:
                    continue
                _uid, team = entry
                if team == my_team or team is None:
                    continue
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    enemy_bot_xy = nxy
        if enemy_bot_xy is None:
            return False

        # Step 2: empty / allied-road tile adjacent to the enemy bot AND
        # within our action radius. Prefer empty (no destroy needed).
        ti, _ax = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0
        launcher_cost = int(GC.LAUNCHER_BASE_COST[0] * scale)
        if ti < launcher_cost:
            return False

        candidates_empty = []
        candidates_road = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                txy = (enemy_bot_xy[0] + dx, enemy_bot_xy[1] + dy)
                if txy == my_xy:
                    continue
                if txy in self.known_walls:
                    continue
                ddx = txy[0] - my_xy[0]
                ddy = txy[1] - my_xy[1]
                if ddx * ddx + ddy * ddy > GC.ACTION_RADIUS_SQ:
                    continue
                if txy in bpc:
                    continue
                info = tc.get(txy)
                if info is None:
                    continue
                env, bid, etype, team = info
                if env == Environment.WALL:
                    continue
                if bid is None:
                    candidates_empty.append(txy)
                elif etype == EntityType.ROAD and team == my_team:
                    candidates_road.append(txy)

        for txy in candidates_empty + candidates_road:
            tpos = xy_to_pos(txy)
            info = tc.get(txy)
            env = info[0] if info else Environment.EMPTY
            bid = info[1] if info else None
            etype = info[2] if info else None
            team = info[3] if info else None
            # Destroy allied road first (free, doesn't burn action cd).
            if (bid is not None and etype == EntityType.ROAD
                    and team == my_team):
                if ct.can_destroy(tpos):
                    ct.destroy(tpos)
                    tc[txy] = (env, None, None, None)
            if ct.can_build_launcher(tpos):
                ct.build_launcher(tpos)
                tc[txy] = (env, -1, EntityType.LAUNCHER, my_team)
                self._ds_event(
                    f"[DS] anti-heal launcher@{txy} vs bot@{enemy_bot_xy}")
                return True
        return False

    def _continue_disruption(self, ct):
        """Drive the attack → step-off → barrier flow on self.disrupt_target.
        Mirrors Step-4a attack + Step-4c place logic but drops a BARRIER
        instead of a gunner/sentinel.
        """
        target = self.disrupt_target
        if target is None:
            return
        my_xy = self._my_xy
        info = self.tile_cache.get(target)
        target_gone = (info is None or info[1] is None
                       or info[3] == self.my_team_cache)

        # Post-destroy barrier flow: once we've marked the target destroyed
        # (we were on it when it went to 0 HP), keep driving toward barrier
        # placement — don't let the "vanished" abandon path fire.
        if self.disrupt_target_destroyed:
            if my_xy == target:
                # Still on the tile — step off so we can build.
                self._disruptor_step_off(ct, my_xy)
                return
            # Someone built here before us — bail.
            if info is not None and info[1] is not None:
                self._disrupt_reset(abandon_reason="post-destroy occupied")
                return
            dsq = (my_xy[0] - target[0]) ** 2 + (my_xy[1] - target[1]) ** 2
            if dsq > GC.ACTION_RADIUS_SQ:
                result = self._disruptor_navigate(ct, target)
                if result == 'blocked':
                    self._disrupt_reset(abandon_reason="post-destroy nav blocked")
                return
            tpos = xy_to_pos(target)
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.BARRIER_BASE_COST[0])
                    and ct.can_build_barrier(tpos)):
                ct.build_barrier(tpos)
                self._ds_event(f"[DS] disrupt barrier@{target}")
                self._disrupt_reset(abandon_reason=None)
            return

        # First observation of destruction while on-tile → flag + step off.
        if my_xy == target and target_gone and my_xy in self._last_vision_set:
            self.disrupt_target_destroyed = True
            self._ds_event(f"[DS] disrupt destroyed@{target}")
            self._disruptor_step_off(ct, my_xy)
            return

        # Target vanished before we arrived → someone else did it; abandon.
        if my_xy != target and target_gone and target in self._last_vision_set:
            self._disrupt_reset(abandon_reason="vanished")
            return

        # Target still alive — approach.
        if my_xy != target:
            result = self._disruptor_navigate(ct, target)
            if result == 'blocked':
                self._disrupt_reset(abandon_reason="nav blocked")
            return

        # On-tile, alive → fire, with heal detection.
        bid = info[1]
        try:
            cur_hp = ct.get_hp(bid)
        except Exception:
            cur_hp = None

        # Heal window: measured in SUCCESSFUL FIRES, not wall-clock turns.
        # Turns where we don't shoot (action cooldown, insufficient Ti)
        # can't cause HP to drop, so including them in the window would
        # let us false-positive "healed" on a target we never hit.
        #
        # The check REPEATS every HEALED_DETECTION_TURNS fires — after
        # each check the window resets and we re-snapshot the start HP,
        # so a target that's constantly healed keeps getting re-tested
        # and eventually abandoned (instead of this being a one-shot
        # check).
        #
        # Abandon only when the enemy heal is keeping pace with our
        # damage. Builder fire = 2 dmg, 5 fires = 10 max damage.
        # Heal action = 4 HP. So:
        #   drop 10 → no heal (perfect)
        #   drop 6-8 → one heal action (still net-positive, keep firing)
        #   drop ≤ 4 → enemy is matching or out-healing us (abandon)
        # Setting MIN_DROP = 4 (= HEAL_AMOUNT) abandons iff the
        # window's net damage is ≤ 4 HP — i.e. enemy healed at least
        # twice. One partial heal is fine, the conveyor still dies.
        MIN_DROP_PER_WINDOW = 4
        if self.disrupt_heal_start_hp is None and cur_hp is not None:
            self.disrupt_heal_start_hp = cur_hp
            self.disrupt_fires_in_window = 0
        elif (self.disrupt_heal_start_hp is not None
              and cur_hp is not None
              and self.disrupt_fires_in_window >= HEALED_DETECTION_TURNS):
            if cur_hp >= self.disrupt_heal_start_hp - MIN_DROP_PER_WINDOW:
                # HP didn't drop meaningfully after HEALED_DETECTION_TURNS
                # fires. Before abandoning, try to plant an anti-heal
                # launcher next to the enemy builder bot that's healing
                # the target. Up to 2 launchers per target — if HP STILL
                # doesn't drop after both, give up.
                if (self.disrupt_anti_heal_launchers < 2
                        and self._disruptor_try_anti_heal_launcher(ct, my_xy)):
                    self.disrupt_anti_heal_launchers += 1
                    # Reset the heal window — keep firing for another
                    # HEALED_DETECTION_TURNS to see if the launcher
                    # cleared the healer.
                    self.disrupt_heal_start_hp = cur_hp
                    self.disrupt_fires_in_window = 0
                    return
                self._ds_event(
                    f"[DS] disrupt healed @{target} "
                    f"hp {self.disrupt_heal_start_hp}->{cur_hp} "
                    f"after {self.disrupt_fires_in_window} fires "
                    f"(launchers={self.disrupt_anti_heal_launchers}); abandon")
                self._disrupt_reset(abandon_reason="healed")
                return
            # Meaningful drop — reset the window and keep firing.
            self.disrupt_heal_start_hp = cur_hp
            self.disrupt_fires_in_window = 0

        if ct.get_action_cooldown() == 0:
            mpos = xy_to_pos(my_xy)
            if (self._can_spend(ct, GC.BUILDER_BOT_ATTACK_COST[0])
                    and ct.can_fire(mpos)):
                ct.fire(mpos)
                self.disrupt_fires_in_window += 1
                self._ds_event(f"[DS] disrupt fire@{my_xy}")

    def _attack_heal_check(self, ct, bid, target_xy):
        """Generic heal detection for "fire on own tile until destroyed"
        attack sites (intercept feeder, chain-extension enemy-road,
        splitter-sentinel enemy-road). Returns True if the caller
        should abandon the attack (target is being healed faster than
        we're damaging it).

        Self-resets when `target_xy` changes, so callers don't need to
        manage reset on transitions. Same threshold model as
        `_continue_disruption`: after HEALED_DETECTION_TURNS successful
        fires, require a near-full damage drop (≥ HEALED_DETECTION_TURNS
        * 2 - 1 HP) — anything less means meaningful healing is happening.
        """
        if self.attack_heal_target != target_xy:
            self.attack_heal_target = target_xy
            self.attack_heal_start_hp = None
            self.attack_heal_fires = 0
        try:
            cur_hp = ct.get_hp(bid)
        except Exception:
            cur_hp = None
        if cur_hp is None:
            return False
        if self.attack_heal_start_hp is None:
            self.attack_heal_start_hp = cur_hp
            self.attack_heal_fires = 0
            return False
        if self.attack_heal_fires < HEALED_DETECTION_TURNS:
            return False
        # Same threshold rationale as _continue_disruption: 5 fires =
        # 10 max damage, heal action = 4 HP. Abandon only when the
        # enemy is matching us (drop ≤ 4 = heal twice). One heal
        # leaves drop ≥ 6 — keep firing, the target still dies.
        MIN_DROP = 4
        if cur_hp >= self.attack_heal_start_hp - MIN_DROP:
            self._ds_event(
                f"[DS] attack healed @{target_xy} "
                f"hp {self.attack_heal_start_hp}->{cur_hp} "
                f"after {self.attack_heal_fires} fires; abandon")
            self._attack_heal_reset()
            return True
        # Meaningful drop — reset window and keep firing.
        self.attack_heal_start_hp = cur_hp
        self.attack_heal_fires = 0
        return False

    def _attack_heal_note_fire(self):
        """Call after a successful ct.fire() at an attack site that uses
        `_attack_heal_check`. Increments the in-window fire counter."""
        self.attack_heal_fires += 1

    def _attack_heal_reset(self):
        """Clear shared attack-heal tracking. Call when abandoning the
        attack site so the next site starts fresh."""
        self.attack_heal_target = None
        self.attack_heal_start_hp = None
        self.attack_heal_fires = 0

    def _disrupt_reset(self, abandon_reason=None):
        """Clear disrupt state and reset the 8-turn idle counter so harass
        mode picks a fresh target after DISRUPT_INTERVAL more turns.

        Every abandoned target is permanently blacklisted so the next
        disruption cycle doesn't immediately re-pick the same tile (the
        Euclidean-nearest picker would otherwise loop on a healed / nav-
        blocked target). Blacklist the tile even on successful completion
        — once we've placed a barrier there, there's nothing left to
        disrupt anyway.
        """
        if self.disrupt_target is not None:
            self.disrupt_blacklist.add(self.disrupt_target)
        if DEBUG and abandon_reason:
            print(f"[DS] disrupt reset ({abandon_reason}) "
                  f"blacklist now={len(self.disrupt_blacklist)}",
                  file=sys.stderr)
        self.disrupt_target = None
        self.disrupt_target_destroyed = False
        self.disrupt_heal_start_hp = None
        self.disrupt_fires_in_window = 0
        self.disrupt_anti_heal_launchers = 0
        self.disrupt_counter = 0
        self.path = None
        self.path_index = 0

    def _harass_explore(self, ct, my_xy):
        """Default harass action: walk a clockwise circle around the enemy
        core. Higher-priority checks (continue_disruption / harvester
        intercept / ore-block / 8-turn disruption counter) all run
        before this and divert the bot when they have something to do —
        so when none of them fire, the circle keeps the bot rotating
        instead of sitting still waiting for a target to walk into view.
        """
        ec = self.enemy_core_pos or self.enemy_core_guess
        if ec is None:
            return

        tgt = self._disruptor_pick_circle_waypoint(my_xy, ec)
        if tgt is None:
            return
        self.step10_wander_target = tgt

        need_path = (self.path is None or not self.path
                     or self.path[-1] != tgt)
        if need_path:
            path = self._compute_path(my_xy, tgt)
            if path is None:
                # Can't route to this waypoint right now (rare — usually
                # walled in by the enemy core footprint plus surrounding
                # buildings). Bump the cursor so we try a different angle
                # next turn instead of looping on the same dead end.
                n = len(self._disruptor_circle_waypoints or [])
                if n:
                    self.disrupt_circle_idx = (self.disrupt_circle_idx + 1) % n
                self.step10_wander_target = None
                return
            self.path = path
            self.path_index = 0

        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0
            self.step10_wander_target = None

    # ------------------------------------------------------------------ #
    #  Circular patrol around the enemy core (idle fallback)              #
    # ------------------------------------------------------------------ #
    #  Mirrors PatrolMixin._patrol_priority_5_circle but centred on the
    #  enemy core. Used by `_harass_explore` when no in-radius frontier
    #  tile remains so a disruptor with nothing actionable in vision
    #  keeps moving and uncovers fresh angles instead of sitting still.
    # ------------------------------------------------------------------ #

    def _disruptor_circle_radius(self):
        """Patrol radius around the enemy core — tunable via constants."""
        return DISRUPTOR_CIRCLE_RADIUS

    def _build_disruptor_circle_waypoints(self, ec):
        """Precompute clockwise waypoints around `ec` at the circle
        radius, ~1-tile spacing. Out-of-bounds waypoints are dropped
        upfront (unlike patrol's clamp-to-bounds, which can collapse
        many slots onto the same edge tile and stall the rotation)."""
        cx, cy = ec
        R = self._disruptor_circle_radius()
        n = max(8, int(math.ceil(2 * math.pi * R)))
        mw = self.map_w
        mh = self.map_h
        waypoints = []
        for i in range(n):
            a = (2 * math.pi * i) / n
            tx = cx + int(round(R * math.cos(a)))
            ty = cy + int(round(R * math.sin(a)))
            if mw is not None and (tx < 0 or ty < 0 or tx >= mw or ty >= mh):
                continue
            xy = (tx, ty)
            if waypoints and waypoints[-1] == xy:
                continue
            waypoints.append(xy)
        if len(waypoints) > 1 and waypoints[0] == waypoints[-1]:
            waypoints.pop()
        return waypoints

    def _disruptor_nearest_circle_idx(self, xy):
        best_i = 0
        best_dsq = None
        for i, wp in enumerate(self._disruptor_circle_waypoints):
            d = (wp[0] - xy[0]) ** 2 + (wp[1] - xy[1]) ** 2
            if best_dsq is None or d < best_dsq:
                best_dsq = d
                best_i = i
        return best_i

    def _disruptor_pick_circle_waypoint(self, my_xy, ec):
        """Return the next clockwise waypoint around the enemy core,
        advancing the cursor when we've reached (or are within action
        range of) the current one. Skips known walls — if a full lap
        is exhausted, returns None so the caller can idle this turn.
        """
        # Lazy build. Rebuild if the enemy_core_pos sighting moved the
        # centre off the guess we used last time.
        if (self._disruptor_circle_waypoints is None
                or self._disruptor_circle_center != ec):
            self._disruptor_circle_waypoints = (
                self._build_disruptor_circle_waypoints(ec))
            self._disruptor_circle_center = ec
            if self._disruptor_circle_waypoints:
                self.disrupt_circle_idx = self._disruptor_nearest_circle_idx(my_xy)

        wps = self._disruptor_circle_waypoints
        if not wps:
            return None
        n = len(wps)

        # Advance if we've arrived (exact or action-radius close).
        cur = wps[self.disrupt_circle_idx]
        cdsq = (my_xy[0] - cur[0]) ** 2 + (my_xy[1] - cur[1]) ** 2
        if cdsq <= GC.ACTION_RADIUS_SQ:
            self.disrupt_circle_idx = (self.disrupt_circle_idx + 1) % n

        # One full lap of skipping walls — if every slot is walled, give
        # up and let the bot idle.
        for _ in range(n):
            wp = wps[self.disrupt_circle_idx]
            if wp not in self.known_walls:
                return wp
            self.disrupt_circle_idx = (self.disrupt_circle_idx + 1) % n
        return None

