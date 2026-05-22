import sys
from cambc import Direction, EntityType, Environment, ResourceType, GameConstants as GC
from constants import (
    OBSERVE_TURNS, MAX_HARVESTERS_PER_BOT, MAX_BRIDGE_RETRY, DEBUG,
    FOUNDRY_MARKER_VALUE, MIN_TITANIUM, ROAD_PROTECT_RANGE,
)
from utils import (
    cardinal_direction_between, direction_between, euclidean_dist_sq,
    DIRECTION_DELTAS, xy_to_pos, neighbors_4,
)
from pathfinding import astar_cached, astar_monotonic, _WALKABLE_BUILDINGS

# Allied transport buildings
_ALLIED_TRANSPORT = frozenset({
    EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
})

# Buildings we can destroy and build over during chain construction.
# Markers are special: any team may build over them, the engine destroys
# the marker as part of the build action. Bots can't walk onto markers,
# and firing at them is wasteful — the correct response to a marker in
# the way is to build on top of it.
#
# Allied barriers are included: chain-to-core destroys an allied barrier
# in its path (free destroy, Ti refund) then builds a conveyor on top.
# Conveyors and bridges may also target an allied barrier tile because
# the chain builder will clear it before the next iteration runs.
_DESTROYABLE = frozenset({
    EntityType.ROAD, EntityType.MARKER, EntityType.BARRIER,
})

# Enemy walkable buildings we can stand on and attack from. Excludes
# markers — bots cannot walk onto markers, so "walk onto and fire"
# doesn't apply. Enemy markers are handled by calling build_*() directly,
# which replaces the marker.
_ENEMY_WALKABLE = frozenset({
    EntityType.ROAD, EntityType.CONVEYOR, EntityType.SPLITTER,
    EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
})


class ChainingMixin:
    """Chain-to-core conveyor building using monotonic A*.

    Note: _allied_barrier_walls() lives on HarvestingMixin and is shared
    via the Player mixin chain.
    """

    # ------------------------------------------------------------------ #
    #  Step: chain_to_core                                                #
    # ------------------------------------------------------------------ #

    def _step_chain_to_core(self, ct):
        """Build conveyor chain from chain_start back to core (or to a
        goal override when `_chain_goal` is set — used by the axionite
        Step 7b chain to target a foundry tile instead of the core)."""
        my_xy = self._my_xy

        # Pause the chain to heal a damaged ally building inside our
        # action radius. We don't move during the heal so we stay on
        # the chain — the build resumes next turn once the building
        # is back to full HP (or out of range / vision).
        if self._economy_check_heal_action_radius(ct, my_xy) != 'none':
            return
        # Short-detour heal divert: walk up to 4 tiles to a damaged
        # chain piece, heal it to full, then resume. Chain state
        # (chain_start, chain_path, _chain_built) is untouched, so
        # the resume is automatic on the first turn this returns
        # False. self.path may get clobbered by the divert; chain
        # logic recomputes it as needed.
        if self._economy_try_heal_chain_divert(ct, my_xy):
            return

        if self.chain_start is None or self.core_pos is None:
            self._chain_complete()
            return
        # Goal for the chain builder: foundry_tile when driving the
        # axionite chain, otherwise the core centre.
        goal = self._chain_goal or self.core_pos

        # One-time init
        if self._chain_built is None:
            self._chain_built = {self.chain_start}
        if self._observe_skip is None:
            self._observe_skip = set()

        # --- Handle pending opportunistic harvester ---
        if self._pending_harvester_xy is not None:
            if self._handle_pending_harvester(ct, my_xy):
                return  # consumed this turn; chain resumes next turn
            # else: pending was cleared (done / invalid / abandoned); continue chain

        # --- Drain pending chain-protect roads ---
        # The conveyor case populates + drains in the main loop, but
        # bridges build out of band: _optimal_bridge populates the
        # queue then returns True without going through the main loop
        # again (chain_path is reset, bridge_walk takes over). Drain
        # here so a freshly-built bridge gets its 8-neighbours paved
        # before the bridge_walk handler steps the bot onto the bridge
        # and teleports it across.
        if self._chain_protect_queue:
            if self._chain_protect_pave_step(ct):
                return

        # --- Handle bridge walk-in-progress ---
        if self._bridge_walk_target is not None:
            if my_xy == self._bridge_walk_target:
                self._bridge_walk_target = None

                # Arrived at bridge target — check what's here
                bt_entry = self.tile_cache.get(my_xy)
                has_existing = (bt_entry and bt_entry[3] == self.my_team_cache
                                and bt_entry[2] in _ALLIED_TRANSPORT)

                if has_existing:
                    # If we landed on a splitter (axionite chain targeting
                    # a splitter) or on the chain goal, the chain is done.
                    # Splitters must NEVER be destroyed or bridged over.
                    is_splitter = bt_entry[2] == EntityType.SPLITTER
                    if my_xy == goal or (is_splitter and self.current_chain_type == 'axionite'):
                        print(f"[{self.corner}] bridge walk reached "
                              f"{'splitter' if is_splitter else 'goal'} "
                              f"({my_xy[0]},{my_xy[1]})")
                        self._chain_reached_end(my_xy, ct=ct)
                        return

                    # Bridge landed on existing infrastructure.
                    # Titanium chains: the bridge already delivers
                    # resources to this network — chain is complete
                    # regardless of capacity (no need to observe).
                    if self.current_chain_type != 'axionite':
                        print(f"[{self.corner}] bridge landed on infra at "
                              f"({my_xy[0]},{my_xy[1]}) — chain complete")
                        self._chain_reached_end(my_xy, ct=ct)
                        return
                    # Axionite chains: observe for capacity, then
                    # follow_chain to find where to place foundry.
                    self.observe_xy = my_xy
                    self.observe_target = bt_entry[1]
                    self.observe_turns_left = OBSERVE_TURNS
                    self.observe_ever_empty = False
                    self.observe_saw_axionite = False
                    self._chain_merge_tile = None
                    self.step = "observe_conveyor"
                    if DEBUG: print(f"[{self.corner}] bridge landed on existing infra at ({my_xy[0]},{my_xy[1]}), observing",
                          file=sys.stderr)
                    return

                # Enemy walkable building — fire at it until destroyed
                if (bt_entry and bt_entry[1] is not None
                        and bt_entry[3] != self.my_team_cache
                        and bt_entry[2] in _ENEMY_WALKABLE):
                    my_pos = xy_to_pos(my_xy)
                    if ct.can_fire(my_pos):
                        ct.fire(my_pos)
                    # Stay on bridge target until enemy is gone
                    self._bridge_walk_target = my_xy
                    if DEBUG: print(f"[{self.corner}] firing at enemy {bt_entry[2]} at ({my_xy[0]},{my_xy[1]})",
                          file=sys.stderr)
                    return

                # Landed on empty/road — replan chain from here
                self.chain_start = my_xy
                self.chain_path = None
                self._chain_built = {my_xy}
                self._chain_start_fixed = False  # Need to build/fix conveyor here
                if DEBUG: print(f"[{self.corner}] bridge landed, replanning from ({my_xy[0]},{my_xy[1]})",
                      file=sys.stderr)
                return

            # Walk to bridge target
            walk_turns = self._bridge_walk_turns + 1
            self._bridge_walk_turns = walk_turns

            # Hard cap: spending more than MAX_BRIDGE_RETRY turns walking
            # to the same bridge target means it's genuinely unreachable.
            # Abandon the entire chain rather than thrashing. For
            # axionite chains call _axionite_abandon so the outer
            # chain_to_foundry handler doesn't misinterpret the reset
            # as "chain reached foundry" and transition to place_foundry.
            if walk_turns > MAX_BRIDGE_RETRY:
                if DEBUG:
                    print(f"[{self.corner}] bridge walk exceeded {MAX_BRIDGE_RETRY} turns, "
                          f"abandoning chain", file=sys.stderr)
                self._bridge_walk_target = None
                self._bridge_walk_turns = 0
                if self.current_chain_type == 'axionite':
                    self._axionite_abandon("bridge walk exceeded")
                else:
                    self._chain_complete()
                return

            # Track if we actually moved this turn
            prev_xy = self._bridge_walk_prev_xy
            if prev_xy == my_xy:
                self._bridge_walk_stuck += 1
            else:
                self._bridge_walk_stuck = 0
            self._bridge_walk_prev_xy = my_xy

            # If stuck for 2+ turns, A* path is bad — clear it and try BugNav
            if self._bridge_walk_stuck >= 2:
                self.path = None

            # Try A* — clear oscillation_walls so they don't block the route
            if not self.path or self.path[-1] != self._bridge_walk_target:
                saved_osc = self.oscillation_walls
                self.oscillation_walls = set()
                self.path = self._compute_path(my_xy, self._bridge_walk_target)
                self.oscillation_walls = saved_osc
                self.path_index = 0
            if self.path:
                result = self._follow_path(ct)
                if result == 'blocked':
                    self.path = None  # Path hit a wall — discard it
                else:
                    return

            # A* failed or blocked — BugNav: try all directions toward target
            target = self._bridge_walk_target
            pref_dir = direction_between(my_xy, target)
            for d in [pref_dir,
                      Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST,
                      Direction.NORTHEAST, Direction.SOUTHEAST,
                      Direction.SOUTHWEST, Direction.NORTHWEST]:
                if ct.can_move(d):
                    ct.move(d)
                    return

            # Can't move at all. After 5 turns, treat as unreachable.
            if walk_turns < 5:
                return

            unreachable = self._bridge_walk_target
            if DEBUG: print(f"[{self.corner}] bridge target ({unreachable[0]},{unreachable[1]}) unreachable after {walk_turns} turns",
                  file=sys.stderr)

            bridge_tile = None
            for bxy in [my_xy] + [(my_xy[0]+dx, my_xy[1]+dy)
                                  for dx, dy in ((0,-1),(1,0),(0,1),(-1,0),
                                                 (1,-1),(1,1),(-1,1),(-1,-1))]:
                be = self.tile_cache.get(bxy)
                if be and be[2] == EntityType.BRIDGE and be[3] == self.my_team_cache:
                    bridge_tile = bxy
                    break

            if bridge_tile and ct.get_action_cooldown() == 0:
                bp = xy_to_pos(bridge_tile)
                if ct.can_destroy(bp):
                    ct.destroy(bp)
                    be = self.tile_cache.get(bridge_tile)
                    self.tile_cache[bridge_tile] = (be[0] if be else Environment.EMPTY, None, None, None)
                    if DEBUG: print(f"[{self.corner}] destroyed bridge at ({bridge_tile[0]},{bridge_tile[1]}), retrying",
                          file=sys.stderr)

                bl = self._bridge_target_blacklist
                bl.add(unreachable)
                self._bridge_target_blacklist = bl
                self._bridge_walk_target = None
                self._bridge_walk_turns = 0
                self.chain_path = None
                self._chain_start_fixed = False
            elif not bridge_tile:
                self._chain_complete()
            return

        # --- Compute chain path (once) ---
        if self.chain_path is None:
            # Skip expensive A* during TLE recovery — wait for recovery to end
            if self.tle_recovery_turns > 0:
                return
            # When routing to a non-core goal (e.g. foundry_splitter_tile
            # during the reroute sub-chain), the 9 real core tiles must
            # be impassable so the chain never replays the "chain reached
            # core tile" intercept. Inject them into extra_walls.
            extra_walls = set(self._allied_barrier_walls())
            if (self._chain_goal is not None
                    and not self._is_core_tile(self._chain_goal)):
                cx, cy = self.core_pos
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        extra_walls.add((cx + dx, cy + dy))
            # The foundry_tile itself must be impassable during the
            # chain-to-X sub-chain — a foundry will be built there, and
            # we must not route conveyors or bridges through / into it.
            # Treat it exactly as if the foundry is already in place.
            # Skip the case where goal IS foundry_tile (bootstrap path
            # with no X), in which case the chain needs to terminate on it.
            if (self.foundry_tile is not None
                    and self._chain_goal != self.foundry_tile):
                extra_walls.add(self.foundry_tile)
            self.chain_path = astar_monotonic(
                self.chain_start, goal,
                self.tile_cache, self.known_walls,
                map_w=self.map_w, map_h=self.map_h,
                extra_walls=extra_walls,
                my_team=self.my_team_cache,
            )
            self.chain_index = 0
            self._chain_start_fixed = False
            if self.chain_path is None:
                # Monotonic A* failed — try a bridge from chain_start
                if self._optimal_bridge(ct, self.chain_start):
                    return
                if DEBUG: print(f"[{self.corner}] monotonic A* failed, abandoning chain",
                      file=sys.stderr)
                self._chain_complete()
                return
            if DEBUG: print(f"[{self.corner}] chain route: {len(self.chain_path)} tiles",
                  file=sys.stderr)

        # --- Fix chain_start conveyor direction (once) ---
        if not self._chain_start_fixed:
            if len(self.chain_path) == 0:
                needed_dir = cardinal_direction_between(self.chain_start, goal)
            else:
                needed_dir = cardinal_direction_between(self.chain_start, self.chain_path[0])

            # Conveyor target check on what chain_start would point at
            ddx, ddy = DIRECTION_DELTAS[needed_dir]
            cs_next = (self.chain_start[0] + ddx, self.chain_start[1] + ddy)
            cs_next_entry = self.tile_cache.get(cs_next)
            cs_is_core = self._is_core_tile(cs_next)

            # If chain_start's conveyor would point at a wall, the enemy
            # core, or a non-buildable tile — bridge instead of building
            # a conveyor. Pointing at the enemy core literally feeds them
            # our resources, so this guard is critical even when the
            # enemy core isn't in tile_cache yet (use _is_enemy_core_tile
            # which falls back to the cached center position).
            if not cs_is_core and (
                    cs_next in self.known_walls
                    or self._is_enemy_core_tile(cs_next)
                    or (cs_next_entry and cs_next_entry[0] == Environment.WALL)
                    or (cs_next_entry and cs_next_entry[1] is not None
                        and cs_next_entry[2] not in _ALLIED_TRANSPORT
                        and cs_next_entry[2] not in _DESTROYABLE
                        and cs_next_entry[2] not in _ENEMY_WALKABLE)):
                # Build bridge at chain_start (bot can be adjacent — action radius)
                if self._optimal_bridge(ct, self.chain_start):
                    return
                # Bridge failed — mark fixed and let the build loop handle it
                self._chain_start_fixed = True
                return

            # If chain_start's conveyor would point at allied transport (not
            # ours) we must do the 4-turn capacity check BEFORE committing
            # to the merge. Bridging is preferred when the foreign network
            # is saturated.
            first_chain = not self._first_chain_done
            if not cs_is_core and cs_next_entry is not None:
                if (cs_next_entry[3] == self.my_team_cache
                        and cs_next_entry[2] in _ALLIED_TRANSPORT
                        and cs_next not in (self._chain_built or set())):
                    if cs_next not in self._observe_skip:
                        self.observe_xy = cs_next
                        self.observe_target = cs_next_entry[1]
                        self.observe_turns_left = OBSERVE_TURNS
                        self.observe_ever_empty = False
                        self.observe_saw_axionite = False
                        self._chain_merge_tile = self.chain_start
                        self.step = "observe_conveyor"
                        if DEBUG:
                            print(f"[{self.corner}] chain_start observing foreign at "
                                  f"({cs_next[0]},{cs_next[1]})", file=sys.stderr)
                        return
                    # Already observed and marked saturated. First chain
                    # gets one more try at bridging over; subsequent
                    # chains accept the merge and fall through to build.
                    if first_chain:
                        if self._optimal_bridge(ct, self.chain_start):
                            return
                    pass  # fall through: build conveyor, merge next iter

            # Check if chain_start already has a correctly-directed conveyor
            cs_entry = self.tile_cache.get(self.chain_start)
            has_conveyor = (cs_entry and cs_entry[1] is not None
                           and cs_entry[3] == self.my_team_cache
                           and cs_entry[2] in (EntityType.CONVEYOR, EntityType.BRIDGE,
                                               EntityType.ARMOURED_CONVEYOR))

            if has_conveyor:
                # Conveyor exists — check if actual direction matches path
                actual_dir = self._chain_conv_dir or cardinal_direction_between(self.chain_start, goal)
                if actual_dir == needed_dir:
                    self._chain_start_fixed = True
                else:
                    # Direction wrong — need to be at chain_start to rebuild
                    if my_xy == self.chain_start:
                        if ct.get_action_cooldown() > 0:
                            return
                        ti, _ = ct.get_global_resources()
                        scale = ct.get_scale_percent() / 100.0
                        cost = int(GC.CONVEYOR_BASE_COST[0] * scale)
                        if ti < cost:
                            self._chain_start_fixed = True
                            return
                        cs_pos = xy_to_pos(self.chain_start)
                        if cs_entry[1] is not None and ct.can_destroy(cs_pos):
                            ct.destroy(cs_pos)
                            self.tile_cache[self.chain_start] = (cs_entry[0], None, None, None)
                        if ct.can_build_conveyor(cs_pos, needed_dir):
                            if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                                return  # Wait for reserve
                            ct.build_conveyor(cs_pos, needed_dir)
                            self.building_cache[self.chain_start] = (EntityType.CONVEYOR, needed_dir)
                        else:
                            race = self._recheck_build_race(ct, self.chain_start)
                            if race == 'chain_complete':
                                self._chain_reached_end(self.chain_start, ct=ct)
                                return
                        self._chain_start_fixed = True
                        return
                    # Not at chain_start — try to move there
                    d = cardinal_direction_between(my_xy, self.chain_start)
                    if ct.can_move(d):
                        ct.move(d)
                        return
                    # Can't move (blocked by another bot or obstacle) —
                    # accept current direction and proceed
                    self._chain_start_fixed = True
            else:
                # No conveyor at chain_start — need to build one
                # If enemy walkable building at chain_start, move onto it and fire first
                if (cs_entry and cs_entry[1] is not None
                        and cs_entry[3] != self.my_team_cache
                        and cs_entry[2] in _ENEMY_WALKABLE):
                    if my_xy == self.chain_start:
                        # On it — fire until destroyed
                        if ct.get_action_cooldown() == 0:
                            cs_pos = xy_to_pos(self.chain_start)
                            if ct.can_fire(cs_pos):
                                ct.fire(cs_pos)
                        return
                    else:
                        # Walk onto it
                        d = cardinal_direction_between(my_xy, self.chain_start)
                        if ct.can_move(d):
                            ct.move(d)
                        return
                if ct.get_action_cooldown() > 0:
                    return
                cs_pos = xy_to_pos(self.chain_start)
                if cs_entry and cs_entry[1] is not None:
                    if cs_entry[2] in _DESTROYABLE and ct.can_destroy(cs_pos):
                        ct.destroy(cs_pos)
                        self.tile_cache[self.chain_start] = (cs_entry[0], None, None, None)
                # Prefer an ARMOURED conveyor for the first chain tile when
                # we can afford both resources — the entry point to every
                # chain is the most valuable tile to protect against turret
                # fire and raiders.
                ti_now, ax_now = ct.get_global_resources()
                scale = ct.get_scale_percent() / 100.0
                arm_ti = int(GC.ARMOURED_CONVEYOR_BASE_COST[0] * scale)
                arm_ax = int(GC.ARMOURED_CONVEYOR_BASE_COST[1] * scale)
                can_armour = (
                    ti_now >= arm_ti + int(MIN_TITANIUM * scale)
                    and ax_now >= arm_ax
                    and ct.can_build_armoured_conveyor(cs_pos, needed_dir)
                )
                if can_armour:
                    ct.build_armoured_conveyor(cs_pos, needed_dir)
                    self.building_cache[self.chain_start] = (
                        EntityType.ARMOURED_CONVEYOR, needed_dir)
                    if DEBUG: print(f"[{self.corner}] built ARMOURED conveyor at chain_start ({self.chain_start[0]},{self.chain_start[1]})",
                          file=sys.stderr)
                    d = cardinal_direction_between(my_xy, self.chain_start)
                    if ct.can_move(d):
                        ct.move(d)
                elif ct.can_build_conveyor(cs_pos, needed_dir):
                    if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                        return  # Wait for reserve
                    ct.build_conveyor(cs_pos, needed_dir)
                    self.building_cache[self.chain_start] = (EntityType.CONVEYOR, needed_dir)
                    if DEBUG: print(f"[{self.corner}] built conveyor at chain_start ({self.chain_start[0]},{self.chain_start[1]})",
                          file=sys.stderr)
                    # Move onto the new conveyor
                    d = cardinal_direction_between(my_xy, self.chain_start)
                    if ct.can_move(d):
                        ct.move(d)
                else:
                    race = self._recheck_build_race(ct, self.chain_start)
                    if race == 'chain_complete':
                        self._chain_reached_end(self.chain_start, ct=ct)
                        return
                self._chain_start_fixed = True
                return

        # --- Chain complete check ---
        if self.chain_index >= len(self.chain_path):
            if DEBUG: print(f"[{self.corner}] chain complete", file=sys.stderr)
            self._chain_reached_end(ct=ct)
            return

        # --- Build next chain tile ---
        target_xy = self.chain_path[self.chain_index]
        is_last = (self.chain_index == len(self.chain_path) - 1)
        build_from = self.chain_start if self.chain_index == 0 else self.chain_path[self.chain_index - 1]
        t_entry = self.tile_cache.get(target_xy)

        if DEBUG:
            is_wall = (target_xy in self.known_walls or (t_entry and t_entry[0] == Environment.WALL))
            print(f"[{self.corner}] chain my={my_xy} idx={self.chain_index}/{len(self.chain_path)} tgt={target_xy} from={build_from} t={t_entry} wall={is_wall} bwalk={self._bridge_walk_target}", file=sys.stderr)

        first_chain = not self._first_chain_done

        # --- Early merge detection ---
        # If target_xy holds another team's allied transport AND our
        # previous chain tile (build_from) is already an allied transport
        # we built, then the merge is in place — our conveyor at
        # build_from outputs into the foreign network at target_xy. The
        # foreign network carries the resources to the core. Mark the
        # chain complete instead of trying to bridge over or walk through
        # (both fail when target_xy is a tile we don't own).
        # Skip when _suppress_chain_merge — the sub-chain must reach its
        # specific goal (foundry/splitter tile), not merge into a random
        # allied network along the way.
        if (not self._suppress_chain_merge
                and not first_chain
                and t_entry is not None
                and t_entry[1] is not None
                and t_entry[3] == self.my_team_cache
                and t_entry[2] in _ALLIED_TRANSPORT
                and target_xy not in self._chain_built):
            bf_entry = self.tile_cache.get(build_from)
            if (bf_entry is not None
                    and bf_entry[1] is not None
                    and bf_entry[3] == self.my_team_cache
                    and bf_entry[2] in _ALLIED_TRANSPORT
                    and build_from in self._chain_built):
                print(f"[{self.corner}] implicit merge: ({build_from[0]},{build_from[1]}) "
                      f"feeds foreign at ({target_xy[0]},{target_xy[1]})")
                self._chain_reached_end(target_xy, ct=ct)
                return

        # --- Determine what conveyor at target_xy would point at ---
        if is_last:
            conv_dir = cardinal_direction_between(target_xy, goal)
            dx, dy = DIRECTION_DELTAS[conv_dir]
            next_tile = (target_xy[0] + dx, target_xy[1] + dy)
        else:
            next_tile = self.chain_path[self.chain_index + 1]

        # --- Per-piece chain padding ---
        # If the bot is within ROAD_PROTECT_RANGE of the core, pave
        # every paveable 8-neighbour of the JUST-BUILT chain piece
        # (build_from for this iteration) with cheap roads before
        # laying the next conveyor / bridge. Excludes the chain's
        # own incoming (upstream) and outgoing (target_xy) tiles so
        # we don't waste a road on a spot the chain itself will
        # fill. `build_from in _chain_built` gates the populate so
        # we never try to pave around an unbuilt anchor.
        if (self.core_pos is not None
                and max(abs(my_xy[0] - self.core_pos[0]),
                        abs(my_xy[1] - self.core_pos[1])) <= ROAD_PROTECT_RANGE
                and self._chain_built is not None
                and build_from in self._chain_built):
            if self.chain_index >= 2:
                upstream = self.chain_path[self.chain_index - 2]
            elif self.chain_index == 1:
                upstream = self.chain_start
            else:
                upstream = None  # chain_start has no chain upstream
            self._chain_protect_populate(build_from, upstream, target_xy)
            if self._chain_protect_pave_step(ct):
                return

        # --- Conveyor target check on next_tile ---
        nt_entry = self.tile_cache.get(next_tile)
        is_core_tile = self._is_core_tile(next_tile)

        # Enemy core tile — treat as a wall, bridge from target_xy
        if self._is_enemy_core_tile(next_tile):
            if self._optimal_bridge(ct, target_xy):
                return
            if self._bridge_has_valid_target(target_xy):
                return
            self.chain_index += 1
            return

        # target_xy must be buildable before we can route through it.
        # If it's a wall or known wall, skip the nt_entry observe — the
        # wall handler further down will bridge from build_from.
        target_buildable = (
            target_xy not in self.known_walls
            and not (t_entry and t_entry[0] == Environment.WALL))

        if not is_core_tile and nt_entry is not None and target_buildable:
            # Allied transport at next_tile (not our chain).
            # Skip observe when next_tile IS the chain goal — we want to
            # build a conveyor pointing at the goal, not observe/bridge it.
            next_is_goal = (next_tile == goal)
            if (not next_is_goal
                    and nt_entry[3] == self.my_team_cache
                    and nt_entry[2] in _ALLIED_TRANSPORT
                    and next_tile not in self._chain_built):
                # Always observe the foreign conveyor for capacity
                # before committing. Bridging is preferred when the
                # network is saturated — for subsequent chains because
                # of capacity, for first_chain because of the
                # never-merge preference.
                if next_tile not in self._observe_skip:
                    self.observe_xy = next_tile
                    self.observe_target = nt_entry[1]
                    self.observe_turns_left = OBSERVE_TURNS
                    self.observe_ever_empty = False
                    self.observe_saw_axionite = False
                    self._chain_merge_tile = target_xy
                    self.step = "observe_conveyor"
                    if DEBUG: print(f"[{self.corner}] observing conv at ({next_tile[0]},{next_tile[1]})",
                          file=sys.stderr)
                    return
                # Already observed (and marked saturated). First chain
                # gets one more attempt at bridging over; otherwise
                # fall through to the build path.
                if first_chain:
                    if self._optimal_bridge(ct, target_xy):
                        return

            # Wall or non-buildable building at next_tile (walls, allied
            # harvesters/sentinels/turrets, enemy non-walkable). Allied
            # roads/markers/barriers are fine — they're destroyable + buildable.
            elif (nt_entry[0] == Environment.WALL
                  or (nt_entry[1] is not None
                      and nt_entry[2] not in _ALLIED_TRANSPORT
                      and nt_entry[2] not in _DESTROYABLE
                      and nt_entry[2] not in _ENEMY_WALKABLE)):
                # Need bridge from target_xy to jump over the wall/obstacle.
                # But first, clear target_xy if it has an enemy or allied building.
                if (t_entry and t_entry[1] is not None
                        and t_entry[3] != self.my_team_cache
                        and t_entry[2] in _ENEMY_WALKABLE):
                    # Walk onto target_xy and fire until cleared
                    if my_xy == target_xy:
                        if ct.get_action_cooldown() == 0:
                            pos = xy_to_pos(target_xy)
                            if ct.can_fire(pos):
                                ct.fire(pos)
                        return
                    d = cardinal_direction_between(my_xy, target_xy)
                    if ct.can_move(d):
                        ct.move(d)
                    return
                # Must be within action radius of target_xy to bridge
                tadx = my_xy[0] - target_xy[0]
                tady = my_xy[1] - target_xy[1]
                if tadx * tadx + tady * tady > GC.ACTION_RADIUS_SQ:
                    # Another bot may have built through our planned bridge
                    # position while we were walking — if so, the chain is
                    # effectively done and we can stop.
                    if self._bridge_target_taken(target_xy):
                        print(f"[{self.corner}] bridge target ({target_xy[0]},"
                              f"{target_xy[1]}) filled by another bot")
                        self._chain_reached_end(target_xy, ct=ct)
                        return
                    # Too far — pathfind to target_xy first
                    if self.path is None or self.path_index >= len(self.path):
                        self.path = self._compute_path(my_xy, target_xy)
                        self.path_index = 0
                    if self.path:
                        self._follow_path(ct)
                    else:
                        # Can't reach target — try bridge from build_from or my_xy
                        if self._optimal_bridge(ct, build_from):
                            return
                        if self._optimal_bridge(ct, my_xy):
                            return
                    return
                if self._optimal_bridge(ct, target_xy):
                    return
                if self._bridge_has_valid_target(target_xy):
                    return  # Wait for resources
                self.chain_index += 1
                return

            # Enemy walkable building at next_tile — will attack when we get there
            elif (nt_entry[1] is not None
                  and nt_entry[3] != self.my_team_cache
                  and nt_entry[2] in _ENEMY_WALKABLE):
                pass  # proceed to build conveyor, attack enemy on next iteration

        elif not is_core_tile and next_tile in self.known_walls:
            # Known wall at next_tile — bridge from target_xy
            tadx = my_xy[0] - target_xy[0]
            tady = my_xy[1] - target_xy[1]
            if tadx * tadx + tady * tady > GC.ACTION_RADIUS_SQ:
                # Another bot may have filled our planned bridge position
                # while we were walking — chain complete.
                if self._bridge_target_taken(target_xy):
                    print(f"[{self.corner}] bridge target ({target_xy[0]},"
                          f"{target_xy[1]}) filled by another bot (wall branch)")
                    self._chain_reached_end(target_xy, ct=ct)
                    return
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(my_xy, target_xy)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                else:
                    if self._optimal_bridge(ct, build_from):
                        return
                    if self._optimal_bridge(ct, my_xy):
                        return
                return
            if self._optimal_bridge(ct, target_xy):
                return
            self.chain_index += 1
            return

        # --- Handle current target_xy ---

        # If target is a core tile, handle per chain type.
        if self._is_core_tile(target_xy):
            # Axionite chains that reach the core without placing a
            # foundry get intercepted: pick a foundry tile + X, destroy
            # the last built conveyor (currently pointing into the
            # core), and reroute toward X with merging suppressed.
            if (self.current_chain_type == 'axionite'
                    and not self.axionite_chain_done
                    and self.foundry_tile is None):
                self._reroute_axionite_to_foundry(ct, build_from)
                return
            print(f"[{self.corner}] chain reached core tile ({target_xy[0]},{target_xy[1]})")
            self._chain_reached_end(target_xy, ct=ct)
            return

        # If target IS the chain goal, we've arrived — don't observe it.
        if target_xy == goal:
            print(f"[{self.corner}] chain reached goal ({target_xy[0]},{target_xy[1]})")
            self._chain_reached_end(target_xy, ct=ct)
            return

        # Axionite chain hit an allied splitter — chain is done.
        # Splitters must NEVER be destroyed or observed-over.
        if (self.current_chain_type == 'axionite'
                and t_entry is not None
                and t_entry[3] == self.my_team_cache
                and t_entry[2] == EntityType.SPLITTER):
            print(f"[{self.corner}] AX chain hit splitter at "
                  f"({target_xy[0]},{target_xy[1]}) → complete")
            self.axionite_chain_done = True
            self.axionite_chain_substep = 'complete'
            self.step = "chain_to_core"
            return

        # If target has existing allied transport we didn't build —
        # observe for capacity before deciding.
        if (t_entry is not None
                and t_entry[3] == self.my_team_cache
                and t_entry[2] in _ALLIED_TRANSPORT
                and target_xy not in self._chain_built):
            if target_xy not in self._observe_skip:
                self.observe_xy = target_xy
                self.observe_target = t_entry[1]
                self.observe_turns_left = OBSERVE_TURNS
                self.observe_ever_empty = False
                self.observe_saw_axionite = False
                # merge_tile is the tile BEFORE target (where we'd
                # build a linking conveyor if capacity is OK).
                self._chain_merge_tile = build_from
                self.step = "observe_conveyor"
                if DEBUG: print(f"[{self.corner}] observing existing transport at ({target_xy[0]},{target_xy[1]})",
                      file=sys.stderr)
                return
            # Already observed and marked saturated. First chain gets
            # one more try at bridging over; otherwise walk through to
            # let the normal build path handle the tile.
            if first_chain:
                if self._optimal_bridge(ct, target_xy):
                    return
            # Walk through (either first_chain after a bridge fail or
            # subsequent chain after a saturated observe).
            d = cardinal_direction_between(my_xy, target_xy)
            if my_xy != build_from:
                d2 = cardinal_direction_between(my_xy, build_from)
                if ct.can_move(d2):
                    ct.move(d2)
                return
            if ct.can_move(d):
                ct.move(d)
                self.chain_index += 1
            return

        # If already on the target tile — build conveyor here if needed
        if my_xy == target_xy:
            t_now = self.tile_cache.get(target_xy)
            if (t_now and t_now[3] == self.my_team_cache
                    and t_now[2] in (EntityType.CONVEYOR, EntityType.BRIDGE,
                                     EntityType.ARMOURED_CONVEYOR)):
                # Already has our conveyor/bridge — check direction
                if is_last:
                    needed = cardinal_direction_between(target_xy, goal)
                else:
                    needed = cardinal_direction_between(
                        target_xy, self.chain_path[self.chain_index + 1])
                bc = self.building_cache.get(target_xy)
                cur_dir = bc[1] if bc else None
                if cur_dir == needed or t_now[2] == EntityType.BRIDGE:
                    self._chain_built.add(target_xy)
                    self.chain_index += 1
                    self._last_build_action = f"skip(already_built@{target_xy})"
                    return
                # Direction wrong — destroy and rebuild
                if ct.get_action_cooldown() > 0:
                    return
                build_pos = xy_to_pos(target_xy)
                if ct.can_destroy(build_pos):
                    ct.destroy(build_pos)
                    self.tile_cache[target_xy] = (t_now[0], None, None, None)
                    self.building_cache.pop(target_xy, None)
                    if DEBUG: print(f"[{self.corner}] chain rebuild: destroyed wrong-dir "
                          f"conv at ({target_xy[0]},{target_xy[1]}) "
                          f"was={cur_dir} need={needed}", file=sys.stderr)
                # Fall through to build with correct direction
            # Need to build conveyor here — but first check what it would point at
            if is_last:
                cdir = cardinal_direction_between(target_xy, goal)
            else:
                cdir = cardinal_direction_between(target_xy, self.chain_path[self.chain_index + 1])
            cdx, cdy = DIRECTION_DELTAS[cdir]
            cnext = (target_xy[0] + cdx, target_xy[1] + cdy)
            cn_entry = self.tile_cache.get(cnext)
            cn_is_core = self._is_core_tile(cnext)
            # Check if conveyor target is invalid (harvester, sentinel, wall, etc.)
            if not cn_is_core and (
                    cnext in self.known_walls
                    or (cn_entry and cn_entry[0] == Environment.WALL)
                    or (cn_entry and cn_entry[1] is not None
                        and cn_entry[2] not in _ALLIED_TRANSPORT
                        and cn_entry[2] not in _DESTROYABLE
                        and cn_entry[2] not in _ENEMY_WALKABLE)):
                # Invalid target — don't build a conveyor pointing into a wall.
                self.chain_index += 1
                self._last_build_action = f"skip(invalid_cnext={cnext} entry={cn_entry})"
                return
            if ct.get_action_cooldown() > 0:
                self._last_build_action = f"wait(cd>0)"
                return
            build_pos = xy_to_pos(target_xy)
            # Destroy allied road/marker/barrier if present
            if t_now and t_now[1] is not None and t_now[2] in _DESTROYABLE:
                if ct.can_destroy(build_pos):
                    ct.destroy(build_pos)
                    self.tile_cache[target_xy] = (t_now[0], None, None, None)
            # Attack enemy walkable building if present — use fire(), not
            # destroy(). Builder fire only works on my_xy; this block runs
            # inside the `my_xy == target_xy` branch so build_pos IS my_xy.
            if (t_now and t_now[1] is not None
                    and t_now[3] != self.my_team_cache
                    and t_now[2] in _ENEMY_WALKABLE):
                if ct.can_fire(build_pos):
                    ct.fire(build_pos)
                    if DEBUG:
                        print(f"[{self.corner}] fire@({target_xy[0]},{target_xy[1]}) "
                              f"enemy {t_now[2]} blocking chain",
                              file=sys.stderr)
                return  # Wait for destruction, build next turn
            if ct.can_build_conveyor(build_pos, cdir):
                if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                    self._last_build_action = f"wait_reserve@{target_xy}"
                    return
                ct.build_conveyor(build_pos, cdir)
                self.building_cache[target_xy] = (EntityType.CONVEYOR, cdir)
                self._chain_built.add(target_xy)
                self.chain_index += 1
                self._last_build_action = f"conv@{target_xy} dir={cdir}"
            else:
                self._last_build_action = f"conv_FAIL@{target_xy} dir={cdir} tile={t_now}"
                race = self._recheck_build_race(ct, target_xy)
                if race == 'chain_complete':
                    print(f"[{self.corner}] conv race complete at {target_xy}")
                    self._chain_reached_end(target_xy, ct=ct)
                    return
                if race == 'enemy_walkable':
                    return
            return

        # Walk to build position — need to be within action radius of target_xy
        adx = my_xy[0] - target_xy[0]
        ady = my_xy[1] - target_xy[1]
        if adx * adx + ady * ady > GC.ACTION_RADIUS_SQ:
            # Too far — try direct move first, then pathfind
            moved = False
            for try_d in [direction_between(my_xy, target_xy),
                          cardinal_direction_between(my_xy, target_xy),
                          direction_between(my_xy, build_from)]:
                if ct.can_move(try_d):
                    ct.move(try_d)
                    moved = True
                    break
            if moved:
                self._walk_stuck_turns = 0
            else:
                self._walk_stuck_turns += 1
                if self._walk_stuck_turns >= 3:
                    # Direct movement failed — try A* pathfinding
                    if self.path is None or self.path_index >= len(self.path):
                        self.path = self._compute_path(my_xy, target_xy)
                        self.path_index = 0
                    if self.path:
                        result = self._follow_path(ct)
                        if result == 'moved':
                            self._walk_stuck_turns = 0
                        elif result == 'blocked':
                            self.path = None
                            self.path_index = 0
                    else:
                        # Can't reach target — bridge from current position
                        self._walk_stuck_turns = 0
                        self.path = None
                        self.path_index = 0
                        # Try bridge from build_from (previous chain tile)
                        if not self._optimal_bridge(ct, build_from):
                            # Try bridge from my_xy as last resort
                            if not self._optimal_bridge(ct, my_xy):
                                if not self._bridge_has_valid_target(my_xy):
                                    self.chain_index += 1
            return

        # Target already has our conveyor (we built it earlier) —
        # check direction before skipping.
        if (t_entry is not None
                and t_entry[3] == self.my_team_cache
                and t_entry[2] in (EntityType.CONVEYOR, EntityType.BRIDGE)):
            bc = self.building_cache.get(target_xy)
            cur_dir = bc[1] if bc else None
            if is_last:
                needed = cardinal_direction_between(target_xy, goal)
            else:
                needed = cardinal_direction_between(
                    target_xy, self.chain_path[self.chain_index + 1])
            if cur_dir == needed or t_entry[2] == EntityType.BRIDGE:
                d = cardinal_direction_between(my_xy, target_xy)
                if ct.can_move(d):
                    ct.move(d)
                self._chain_built.add(target_xy)
                self.chain_index += 1
                return
            # Direction wrong — move onto it to rebuild
            d = cardinal_direction_between(my_xy, target_xy)
            if ct.can_move(d):
                ct.move(d)
            return

        # --- Build conveyor at target ---

        # Target is a wall — need bridge FROM the previous tile to jump over it
        if (target_xy in self.known_walls
                or (t_entry and t_entry[0] == Environment.WALL)):
            # Bridge is placed on build_from (the last chain tile before the wall)
            bridge_src = build_from
            adx = my_xy[0] - bridge_src[0]
            ady = my_xy[1] - bridge_src[1]
            if adx * adx + ady * ady > GC.ACTION_RADIUS_SQ:
                self._last_build_action = f"wall_walk(src={bridge_src} dsq={adx*adx+ady*ady})"
                for try_d in [direction_between(my_xy, bridge_src),
                              cardinal_direction_between(my_xy, bridge_src)]:
                    if ct.can_move(try_d):
                        ct.move(try_d)
                        return
                return  # Can't move — wait (don't skip)
            if ct.get_action_cooldown() > 0:
                self._last_build_action = f"wall_wait(cd>0 src={bridge_src})"
                return
            result = self._optimal_bridge(ct, bridge_src)
            if result:
                self._last_build_action = f"wall_bridge@{bridge_src}"
                return
            has_tgt = self._bridge_has_valid_target(bridge_src)
            if has_tgt:
                self._last_build_action = f"wall_bridge_wait(src={bridge_src})"
                return  # Temporary failure (cost) — wait and retry
            self._last_build_action = f"wall_SKIP(no_valid_tgt src={bridge_src})"
            self.chain_index += 1
            return

        # Enemy building at target — walk onto it, fire until destroyed, then build
        if (t_entry and t_entry[1] is not None
                and t_entry[3] != self.my_team_cache):
            if t_entry[2] in _ENEMY_WALKABLE:
                # Walk onto it first, then fire (handled by my_xy == target_xy block)
                d = cardinal_direction_between(my_xy, target_xy)
                if ct.can_move(d):
                    ct.move(d)
                return
            else:
                # Non-walkable enemy — bridge from build_from to jump over
                if ct.get_action_cooldown() > 0:
                    return
                result = self._optimal_bridge(ct, build_from)
                if result:
                    return
                if self._bridge_has_valid_target(build_from):
                    return  # Wait for resources
                self.chain_index += 1
                return

        if ct.get_action_cooldown() > 0:
            return

        build_pos = xy_to_pos(target_xy)

        # Destroy allied road/barrier/marker first (free)
        if t_entry and t_entry[1] is not None and t_entry[3] == self.my_team_cache:
            if t_entry[2] in _DESTROYABLE and ct.can_destroy(build_pos):
                ct.destroy(build_pos)
                self.tile_cache[target_xy] = (t_entry[0], None, None, None)

        # Final target check before building — catches cases where nt_entry
        # was None earlier (unknown tile that's now visible as a harvester etc.)
        if is_last:
            conv_dir = cardinal_direction_between(target_xy, goal)
        else:
            conv_dir = cardinal_direction_between(target_xy, self.chain_path[self.chain_index + 1])
        fdx, fdy = DIRECTION_DELTAS[conv_dir]
        final_next = (target_xy[0] + fdx, target_xy[1] + fdy)
        fn_entry = self.tile_cache.get(final_next)
        fn_is_core = self._is_core_tile(final_next)
        if not fn_is_core and (
                final_next in self.known_walls
                or (fn_entry and fn_entry[0] == Environment.WALL)
                or (fn_entry and fn_entry[1] is not None
                    and fn_entry[2] not in _ALLIED_TRANSPORT
                    and fn_entry[2] not in _DESTROYABLE
                    and fn_entry[2] not in _ENEMY_WALKABLE)):
            if self._optimal_bridge(ct, target_xy):
                self._last_build_action = f"fnext_bridge@{target_xy}"
                return
            has_tgt = self._bridge_has_valid_target(target_xy)
            if has_tgt:
                self._last_build_action = f"fnext_bridge_wait@{target_xy} fnext={final_next} fn_entry={fn_entry}"
                return
            self._last_build_action = f"fnext_SKIP@{target_xy} fnext={final_next}"
            self.chain_index += 1
            return

        if ct.can_build_conveyor(build_pos, conv_dir):
            if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                self._last_build_action = f"wait_reserve@{target_xy}"
                return
            ct.build_conveyor(build_pos, conv_dir)
            self.building_cache[target_xy] = (EntityType.CONVEYOR, conv_dir)
            self._chain_built.add(target_xy)
            self._scan_opportunistic_harvester(target_xy, final_next)
            d = cardinal_direction_between(my_xy, target_xy)
            if ct.can_move(d):
                ct.move(d)
            self.chain_index += 1
            self._last_build_action = f"conv@{target_xy} dir={conv_dir}"
        else:
            # Build failed — race check: another bot may have placed something here
            race = self._recheck_build_race(ct, target_xy)
            if race == 'chain_complete':
                print(f"[{self.corner}] conv race complete at {target_xy} (block 2)")
                self._chain_reached_end(target_xy, ct=ct)
                return
            if race == 'enemy_walkable':
                return  # next iteration walks onto + fires
            # Build failed — check if tile is genuinely blocked (wall, enemy)
            # or just temporarily occupied (bot standing on it)
            t_now = self.tile_cache.get(target_xy)
            genuinely_blocked = (
                target_xy in self.known_walls
                or (t_now and t_now[0] == Environment.WALL)
                or (t_now and t_now[1] is not None
                    and t_now[3] != self.my_team_cache
                    and t_now[2] not in _ENEMY_WALKABLE)
            )
            if genuinely_blocked:
                if self._optimal_bridge(ct, target_xy):
                    return
                if self._bridge_has_valid_target(target_xy):
                    return  # Wait for resources
                self.chain_index += 1
            # else: temporarily blocked (bot on tile) — retry next turn

    # ------------------------------------------------------------------ #
    #  Step: observe_conveyor                                             #
    # ------------------------------------------------------------------ #

    def _step_observe_conveyor(self, ct):
        """Unified capacity check for both chain types: merge at the first
        empty observation, bridge over after OBSERVE_TURNS always full.
        No resource-type filtering — titanium chains may merge into any
        network (including axionite-carrying ones). The per-type divergence
        happens in _do_merge: titanium completes the chain, axionite enters
        the follow-chain state machine (Step 7b.1) to place a foundry."""
        if self.observe_target is None:
            self.step = "chain_to_core"
            return

        # If observing an allied splitter on an axionite chain, the chain
        # has reached a splitter — complete immediately. Splitters must
        # NEVER be destroyed or bridged over.
        if self.current_chain_type == 'axionite' and self.observe_xy is not None:
            obs_entry = self.tile_cache.get(self.observe_xy)
            if (obs_entry and obs_entry[3] == self.my_team_cache
                    and obs_entry[2] == EntityType.SPLITTER):
                print(f"[{self.corner}] observe hit splitter at "
                      f"({self.observe_xy[0]},{self.observe_xy[1]}) → complete")
                self.axionite_chain_done = True
                self.axionite_chain_substep = 'complete'
                self.observe_xy = None
                self.observe_target = None
                self.step = "chain_to_core"
                return

        # Axionite reroute / direct-placement sub-chain suppresses
        # merging when the observed conveyor IS our X (splitter tile).
        # Merging into X would trigger follow_chain → scenario_1b which
        # destroys X — wrong, we need X retrofitted as a splitter
        # AFTER the foundry is built. Merges into OTHER conveyors are
        # still allowed (they route via follow_chain normally).
        if (self._suppress_chain_merge
                and self.foundry_splitter_tile is not None
                and self.observe_xy == self.foundry_splitter_tile):
            if DEBUG:
                print(f"[{self.corner}] observe suppressed (==X) at "
                      f"({self.observe_xy[0]},{self.observe_xy[1]}) → bridge",
                      file=sys.stderr)
            self._observe_bridge_over(ct)
            return

        if self.tle_recovery_turns > 0:
            self._observe_skip.add(self.observe_xy)
            self.observe_xy = None
            self.observe_target = None
            self.step = "chain_to_core"
            return

        try:
            resource = ct.get_stored_resource(self.observe_target)
        except Exception:
            obs_entry = self.tile_cache.get(self.observe_xy)
            if obs_entry is not None and obs_entry[1] is not None:
                self._observe_skip.add(self.observe_xy)
            self.observe_xy = None
            self.observe_target = None
            self.step = "chain_to_core"
            return

        if resource is None:
            # Capacity available — merge unless this is the bot's first
            # chain (first chains must never merge into foreign networks;
            # bridge over instead so each bot's first harvester gets its
            # own dedicated path to the core).
            if not self._first_chain_done:
                if DEBUG:
                    print(f"[{self.corner}] conv at ({self.observe_xy[0]},{self.observe_xy[1]}) "
                          f"empty but first_chain → bridge",
                          file=sys.stderr)
                self._observe_bridge_over(ct)
                return
            if DEBUG:
                print(f"[{self.corner}] conv at ({self.observe_xy[0]},{self.observe_xy[1]}) "
                      f"empty → merge ({self.current_chain_type})",
                      file=sys.stderr)
            self._do_merge(ct)
            return

        self.observe_turns_left -= 1
        if DEBUG:
            print(f"[{self.corner}] observe ({self.observe_xy[0]},{self.observe_xy[1]}) "
                  f"resource={resource} left={self.observe_turns_left}",
                  file=sys.stderr)
        if self.observe_turns_left > 0:
            return  # keep observing
        if DEBUG:
            print(f"[{self.corner}] conv at ({self.observe_xy[0]},{self.observe_xy[1]}) "
                  f"full {OBSERVE_TURNS}t → bridge", file=sys.stderr)
        self._observe_bridge_over(ct)

    def _observe_bridge_over(self, ct):
        """Shared 'bridge over the observed conveyor' path — used by both
        titanium and axionite observe rules."""
        self._observe_skip.add(self.observe_xy)
        saturated_xy = self.observe_xy
        merge_from = self._chain_merge_tile
        self.observe_xy = None
        self.observe_target = None

        if merge_from:
            if self._try_bridge_over(ct, merge_from, saturated_xy):
                return
        else:
            my_xy = self._my_xy
            self.chain_start = my_xy
            self.chain_path = None
            self._chain_built = {my_xy}
            self._chain_start_fixed = True

        self.step = "chain_to_core"

    # ------------------------------------------------------------------ #
    #  Bridge support                                                     #
    # ------------------------------------------------------------------ #

    def _is_core_tile(self, xy):
        """Check if xy is one of the 9 tiles of the 3×3 core."""
        if self.core_pos is None:
            return False
        cx, cy = self.core_pos
        return abs(xy[0] - cx) <= 1 and abs(xy[1] - cy) <= 1

    def _is_core_adjacent(self, xy):
        """Check if xy is cardinally adjacent to any core tile."""
        if self.core_pos is None:
            return False
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            if self._is_core_tile((xy[0] + ddx, xy[1] + ddy)):
                return True
        return False

    def _splitter_dir_valid(self, tile_xy, direction, foundry_xy=None):
        """A splitter's back (input) tile must never be a core tile or
        the foundry tile — otherwise the splitter is oriented so that
        the core/foundry would be feeding INTO it, which is nonsense
        (the core and foundry are resource sinks, not sources). Only
        relevant when the splitter tile is core-adjacent; away from
        the core there's no risk."""
        if not self._is_core_adjacent(tile_xy):
            return True
        ddx, ddy = DIRECTION_DELTAS[direction]
        back = (tile_xy[0] - ddx, tile_xy[1] - ddy)
        if self._is_core_tile(back):
            return False
        if foundry_xy is not None and back == foundry_xy:
            return False
        return True

    def _pick_splitter_dir(self, tile_xy, preferred_dir, output_target_xy,
                           foundry_xy=None):
        """Return a splitter direction that (a) keeps `output_target_xy`
        as the splitter's front or a side output, and (b) passes
        `_splitter_dir_valid` (back is not core / foundry). Prefers
        `preferred_dir` when valid. `output_target_xy` must be
        cardinally adjacent to `tile_xy`. Returns None if nothing valid
        exists."""
        perp = {
            Direction.NORTH: (Direction.EAST, Direction.WEST),
            Direction.SOUTH: (Direction.EAST, Direction.WEST),
            Direction.EAST:  (Direction.NORTH, Direction.SOUTH),
            Direction.WEST:  (Direction.NORTH, Direction.SOUTH),
        }
        tgt_dir = cardinal_direction_between(tile_xy, output_target_xy)
        if tgt_dir is None:
            # Fall back to preferred_dir validation only.
            if preferred_dir is not None and self._splitter_dir_valid(
                    tile_xy, preferred_dir, foundry_xy):
                return preferred_dir
            return None
        # Directions that keep output_target as front or side output.
        allowed = [tgt_dir] + list(perp[tgt_dir])
        # Try preferred first if it's in the allowed set.
        order = []
        if preferred_dir in allowed:
            order.append(preferred_dir)
        for d in allowed:
            if d not in order:
                order.append(d)
        for d in order:
            if self._splitter_dir_valid(tile_xy, d, foundry_xy):
                return d
        return None

    def _queue_splitter_bridge_fixes(self, splitter_xy, splitter_dir):
        """After placing a splitter, find cardinally adjacent conveyors
        that target the splitter from a non-back side (front or sides).
        These are blocked because splitters only accept input from the
        back. Queue them for bridge replacement. Also stores the splitter
        for the output-existence check that runs after fixes complete.
        Excludes the foundry tile — it will be replaced by a foundry."""
        ddx, ddy = DIRECTION_DELTAS[splitter_dir]
        back = (splitter_xy[0] - ddx, splitter_xy[1] - ddy)
        ftile = self.foundry_tile
        # Also check follow_chain_final_pos for 1B (foundry goes there).
        ftile_1b = self.follow_chain_final_pos
        fixes = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (splitter_xy[0] + dx, splitter_xy[1] + dy)
            if nxy == back:
                continue
            if nxy == ftile or nxy == ftile_1b:
                continue
            cached = self.building_cache.get(nxy)
            if cached is None:
                continue
            if cached[0] not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                continue
            cdir = cached[1]
            if cdir is None:
                continue
            cdx, cdy = DIRECTION_DELTAS[cdir]
            if (nxy[0] + cdx, nxy[1] + cdy) == splitter_xy:
                fixes.append((nxy, splitter_xy))
        self._splitter_bridge_fixes = fixes
        self._splitter_output_check = (splitter_xy, splitter_dir)
        # Log what we found at each neighbor for diagnostics.
        neighbor_info = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (splitter_xy[0] + dx, splitter_xy[1] + dy)
            tag = ""
            if nxy == back:
                tag = "(back)"
            elif nxy == ftile:
                tag = "(ftile)"
            elif nxy == ftile_1b:
                tag = "(ftile_1b)"
            bc = self.building_cache.get(nxy)
            neighbor_info.append(f"{nxy}{tag}={bc}")
        print(f"[{self.corner}] splitter checks at ({splitter_xy[0]},"
              f"{splitter_xy[1]}) dir={splitter_dir.value}: "
              f"{len(fixes)} blocked | neighbors: "
              f"{', '.join(neighbor_info)}")

    def _process_splitter_bridge_fixes(self, ct):
        """Process blocked-conveyor→bridge replacements, then ensure the
        splitter has at least one non-back output (conveyor facing away
        or bridge). If no output exists, build a bridge on a free
        adjacent side targeting the core. Returns True when all done."""
        # Phase 1: replace blocked conveyors with bridges.
        if self._splitter_bridge_fixes:
            conv_xy, splitter_xy = self._splitter_bridge_fixes[0]
            if self._my_xy != splitter_xy:
                # Try diagonal first (8-dir), then cardinal fallback.
                d = direction_between(self._my_xy, splitter_xy)
                if d is not None and ct.can_move(d):
                    ct.move(d)
                    return False
                # Move failed — try pathfinding with road building.
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(self._my_xy, splitter_xy)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                return False
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.BRIDGE_BASE_COST[0]):
                return False
            cpos = xy_to_pos(conv_xy)
            spos = xy_to_pos(splitter_xy)
            c_entry = self.tile_cache.get(conv_xy)
            if c_entry and c_entry[1] is not None and c_entry[3] == self.my_team_cache:
                if ct.can_destroy(cpos):
                    ct.destroy(cpos)
                    self.tile_cache[conv_xy] = (c_entry[0], None, None, None)
                    self.building_cache.pop(conv_xy, None)
            if ct.can_build_bridge(cpos, spos):
                ct.build_bridge(cpos, spos)
                self.building_cache[conv_xy] = (EntityType.BRIDGE, splitter_xy)
                self.bridge_target_cache[conv_xy] = splitter_xy
                self.tile_cache[conv_xy] = (
                    c_entry[0] if c_entry else Environment.EMPTY,
                    -1, EntityType.BRIDGE, self.my_team_cache,
                )
                print(f"[{self.corner}] bridge-fix: ({conv_xy[0]},{conv_xy[1]}) "
                      f"→ splitter ({splitter_xy[0]},{splitter_xy[1]})")
                self._splitter_bridge_fixes.pop(0)
            if self._splitter_bridge_fixes:
                return False

        # Phase 2: ensure the splitter has at least one non-back output.
        return self._ensure_splitter_has_output(ct)

    def _ensure_splitter_has_output(self, ct):
        """Check the most recently placed splitter (stored in
        _splitter_output_check) for a non-back adjacent conveyor/bridge
        that faces away (output). If none exists, build a bridge on a
        free side targeting the core. Returns True when done."""
        check = getattr(self, '_splitter_output_check', None)
        if check is None:
            return True
        sxy, sdir = check
        bc = self.building_cache.get(sxy)
        if bc is None or bc[0] != EntityType.SPLITTER:
            self._splitter_output_check = None
            return True

        ddx, ddy = DIRECTION_DELTAS[sdir]
        back = (sxy[0] - ddx, sxy[1] - ddy)
        ftile = self.foundry_tile
        ftile_1b = self.follow_chain_final_pos

        has_output = False
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (sxy[0] + dx, sxy[1] + dy)
            if nxy == back:
                continue
            if nxy == ftile or nxy == ftile_1b:
                continue
            nc = self.building_cache.get(nxy)
            if nc is None:
                continue
            if nc[0] in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                ndir = nc[1]
                if ndir is None:
                    continue
                ndx, ndy = DIRECTION_DELTAS[ndir]
                if (nxy[0] + ndx, nxy[1] + ndy) != sxy:
                    has_output = True
                    break
            elif nc[0] == EntityType.BRIDGE:
                bt = self.bridge_target_cache.get(nxy)
                if bt != sxy:
                    has_output = True
                    break
            # Foundries are NOT counted as output here: side output into
            # an adjacent foundry still flows axionite, but we still want
            # a conveyor/bridge output that continues the chain past the
            # splitter. ftile / ftile_1b are already skipped above — any
            # OTHER adjacent foundry belongs to a different chain and
            # shouldn't satisfy this splitter's output requirement.

        # Edge case 1: if splitter is core-adjacent and a non-back
        # neighbor IS a core tile, count that as output — the core
        # absorbs resources and prevents clogging.
        if not has_output:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxy = (sxy[0] + dx, sxy[1] + dy)
                if nxy == back:
                    continue
                if self._is_core_tile(nxy):
                    has_output = True
                    break

        if has_output:
            print(f"[{self.corner}] splitter ({sxy[0]},{sxy[1]}) has output — done")
            self._splitter_output_check = None
            return True
        print(f"[{self.corner}] splitter ({sxy[0]},{sxy[1]}) NO output — "
              f"need bridge to core")

        # Find candidate tiles for an output bridge.
        candidates = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (sxy[0] + dx, sxy[1] + dy)
            if nxy == back:
                continue
            if nxy == ftile or nxy == ftile_1b:
                continue
            if self._is_core_tile(nxy):
                continue
            ne = self.tile_cache.get(nxy)
            if ne is None:
                continue
            if ne[0] == Environment.WALL or nxy in self.known_walls:
                continue
            candidates.append(nxy)

        # Edge case 2: no valid candidates at all — skip and continue.
        if not candidates:
            print(f"[{self.corner}] splitter ({sxy[0]},{sxy[1]}) no output "
                  f"candidates — skipping")
            self._splitter_output_check = None
            return True

        if ct.get_action_cooldown() > 0:
            return False
        if not self._can_spend(ct, GC.BRIDGE_BASE_COST[0]):
            return False
        for nxy in candidates:
            ne = self.tile_cache.get(nxy)
            if ne and ne[1] is not None:
                if (ne[3] == self.my_team_cache
                        and ne[2] == EntityType.ROAD
                        and euclidean_dist_sq(self._my_xy, nxy) <= GC.ACTION_RADIUS_SQ):
                    npos = xy_to_pos(nxy)
                    if ct.can_destroy(npos):
                        ct.destroy(npos)
                        self.tile_cache[nxy] = (ne[0], None, None, None)
                        self.building_cache.pop(nxy, None)
                    else:
                        continue
                else:
                    continue
            if self._build_axionite_bridge(ct, nxy):
                print(f"[{self.corner}] splitter output bridge from "
                      f"({nxy[0]},{nxy[1]}) for splitter "
                      f"({sxy[0]},{sxy[1]})")
                self._splitter_output_check = None
                return True
            if self.path is not None and self.path_index > 0:
                return False
        self._splitter_output_check = None
        return True

    def _scan_foundry_redirects(self, foundry_xy, splitter_xy):
        """After placing a foundry, find conveyors/bridges that target
        the foundry and queue them for redirection to the splitter.
        Conveyors: cardinally adjacent tiles whose direction points at
        the foundry. Bridges: any bridge within dsq<=9 whose target is
        the foundry. Excludes the splitter tile itself."""
        targets = []
        fx, fy = foundry_xy
        my_xy = self._my_xy

        # Cardinal conveyors targeting the foundry
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (fx + dx, fy + dy)
            if nxy == splitter_xy:
                continue
            cached = self.building_cache.get(nxy)
            if cached is None:
                continue
            if cached[0] in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                cdir = cached[1]
                if cdir is None:
                    continue
                cdx, cdy = DIRECTION_DELTAS[cdir]
                if (nxy[0] + cdx, nxy[1] + cdy) == foundry_xy:
                    dist = euclidean_dist_sq(my_xy, nxy)
                    targets.append((dist, nxy))

        # Bridges within range targeting the foundry
        for bxy, btarget in self.bridge_target_cache.items():
            if btarget != foundry_xy:
                continue
            if bxy == splitter_xy:
                continue
            be = self.tile_cache.get(bxy)
            if (be and be[1] is not None
                    and be[2] == EntityType.BRIDGE
                    and be[3] == self.my_team_cache):
                dist = euclidean_dist_sq(my_xy, bxy)
                targets.append((dist, bxy))

        if not targets:
            return
        targets.sort()
        self._foundry_redirect_queue = [xy for _, xy in targets]
        self._foundry_redirect_splitter = splitter_xy
        self._foundry_redirect_foundry = foundry_xy
        print(f"[{self.corner}] foundry redirect: {len(targets)} "
              f"tile(s) targeting foundry ({fx},{fy}) → "
              f"redirect to splitter ({splitter_xy[0]},{splitter_xy[1]})")

    def _process_foundry_redirects(self, ct):
        """Process one foundry redirect per call.

        Conveyors: replace with a splitter in the direction of the
        upstream conveyor feeding it. Then perform splitter checks.

        Bridges: try to retarget to an empty tile cardinally adjacent
        to the foundry and build a splitter there facing away from
        core and foundry. If no empty tile, bridge to the original
        splitter instead.

        Returns True when all done (including splitter checks)."""
        if not self._foundry_redirect_queue:
            return True

        # Phase 1: process splitter checks from a previous redirect.
        if self._splitter_bridge_fixes or self._splitter_output_check:
            if not self._process_splitter_bridge_fixes(ct):
                return False
            # Splitter checks done — pop entry and continue.
            self._foundry_redirect_queue.pop(0)
            return not self._foundry_redirect_queue

        # Check for pending splitter build (from bridge retarget).
        pending = getattr(self, '_foundry_redirect_pending_splitter', None)
        if pending is not None:
            empty_tile, sdir = pending
            if euclidean_dist_sq(self._my_xy, empty_tile) > GC.ACTION_RADIUS_SQ:
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(self._my_xy, empty_tile)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                else:
                    d = direction_between(self._my_xy, empty_tile)
                    if d is not None and ct.can_move(d):
                        ct.move(d)
                return False
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                return False
            epos = xy_to_pos(empty_tile)
            if ct.can_build_splitter(epos, sdir):
                ct.build_splitter(epos, sdir)
                self.building_cache[empty_tile] = (EntityType.SPLITTER, sdir)
                self.tile_cache[empty_tile] = (
                    Environment.EMPTY, -1, EntityType.SPLITTER, self.my_team_cache,
                )
                print(f"[{self.corner}] foundry redirect: splitter at "
                      f"({empty_tile[0]},{empty_tile[1]}) facing {sdir.value}")
                self._foundry_redirect_pending_splitter = None
                self._queue_splitter_bridge_fixes(empty_tile, sdir)
                # Splitter checks will run next call (phase 1).
                return False
            # Build failed — pop and continue.
            self._foundry_redirect_pending_splitter = None
            self._foundry_redirect_queue.pop(0)
            return not self._foundry_redirect_queue

        target = self._foundry_redirect_queue[0]
        splitter = getattr(self, '_foundry_redirect_splitter', None)
        foundry = getattr(self, '_foundry_redirect_foundry', None)
        if splitter is None:
            self._foundry_redirect_queue = []
            return True

        # Move within action range of target.
        if euclidean_dist_sq(self._my_xy, target) > GC.ACTION_RADIUS_SQ:
            if self.path is None or self.path_index >= len(self.path):
                self.path = self._compute_path(self._my_xy, target)
                self.path_index = 0
            if self.path:
                self._follow_path(ct)
            else:
                d = direction_between(self._my_xy, target)
                if d is not None and ct.can_move(d):
                    ct.move(d)
            return False

        if ct.get_action_cooldown() > 0:
            return False

        t_entry = self.tile_cache.get(target)
        t_bc = self.building_cache.get(target)
        is_conveyor = (t_bc is not None and t_bc[0] in (
            EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR))
        is_bridge = (t_bc is not None and t_bc[0] == EntityType.BRIDGE)

        # --- Case 1: Conveyor → replace with splitter ---
        if is_conveyor:
            if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                return False
            # Find upstream conveyor direction (the conveyor feeding this one).
            splitter_dir = None
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxy = (target[0] + dx, target[1] + dy)
                nc = self.building_cache.get(nxy)
                if nc is None:
                    continue
                if nc[0] in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    ndir = nc[1]
                    if ndir is None:
                        continue
                    ndx, ndy = DIRECTION_DELTAS[ndir]
                    if (nxy[0] + ndx, nxy[1] + ndy) == target:
                        splitter_dir = ndir
                        break
            if splitter_dir is None:
                # No upstream conveyor — use the original splitter's dir.
                sbc = self.building_cache.get(splitter)
                if sbc and isinstance(sbc[1], Direction):
                    splitter_dir = sbc[1]
                else:
                    splitter_dir = t_bc[1]  # fallback to conveyor's own dir
            tpos = xy_to_pos(target)
            if t_entry and t_entry[1] is not None:
                if ct.can_destroy(tpos):
                    ct.destroy(tpos)
                    self.tile_cache[target] = (t_entry[0], None, None, None)
                    self.building_cache.pop(target, None)
            if ct.can_build_splitter(tpos, splitter_dir):
                ct.build_splitter(tpos, splitter_dir)
                self.building_cache[target] = (EntityType.SPLITTER, splitter_dir)
                self.tile_cache[target] = (
                    t_entry[0] if t_entry else Environment.EMPTY,
                    -1, EntityType.SPLITTER, self.my_team_cache,
                )
                print(f"[{self.corner}] foundry redirect: conv ({target[0]},"
                      f"{target[1]}) → splitter facing {splitter_dir.value}")
                self._queue_splitter_bridge_fixes(target, splitter_dir)
                # Splitter checks will run next call (phase 1).
                return False
            # Build failed — pop and continue.
            self._foundry_redirect_queue.pop(0)
            return not self._foundry_redirect_queue

        # --- Case 2: Bridge → try empty tile adjacent to foundry ---
        if is_bridge and foundry is not None:
            # Look for an empty tile cardinally adjacent to foundry.
            empty_tile = None
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxy = (foundry[0] + dx, foundry[1] + dy)
                if nxy == splitter or nxy == target:
                    continue
                if self._is_core_tile(nxy):
                    continue
                ne = self.tile_cache.get(nxy)
                if ne is None:
                    continue
                if ne[1] is None and ne[0] == Environment.EMPTY:
                    empty_tile = nxy
                    break

            if empty_tile is not None:
                # Retarget bridge to the empty tile.
                if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                    return False
                tpos = xy_to_pos(target)
                epos = xy_to_pos(empty_tile)
                if t_entry and t_entry[1] is not None:
                    if ct.can_destroy(tpos):
                        ct.destroy(tpos)
                        self.tile_cache[target] = (t_entry[0], None, None, None)
                        self.building_cache.pop(target, None)
                        self.bridge_target_cache.pop(target, None)
                if ct.can_build_bridge(tpos, epos):
                    ct.build_bridge(tpos, epos)
                    self.building_cache[target] = (EntityType.BRIDGE, empty_tile)
                    self.bridge_target_cache[target] = empty_tile
                    self.tile_cache[target] = (
                        t_entry[0] if t_entry else Environment.EMPTY,
                        -1, EntityType.BRIDGE, self.my_team_cache,
                    )
                    # Now pathfind to empty_tile and build splitter there.
                    # Pick direction facing away from both core and foundry.
                    sdir = None
                    for d in (Direction.NORTH, Direction.SOUTH,
                              Direction.EAST, Direction.WEST):
                        ddx, ddy = DIRECTION_DELTAS[d]
                        front = (empty_tile[0] + ddx, empty_tile[1] + ddy)
                        back = (empty_tile[0] - ddx, empty_tile[1] - ddy)
                        if self._is_core_tile(front) or front == foundry:
                            continue
                        if self._is_core_tile(back) or back == foundry:
                            continue
                        sdir = d
                        break
                    if sdir is None:
                        # Fallback: any direction where back isn't core/foundry.
                        sdir = self._pick_splitter_dir(
                            empty_tile, Direction.NORTH, foundry, foundry)
                    if sdir is not None:
                        # Store the empty tile + dir for next call to build
                        # the splitter there.
                        self._foundry_redirect_queue[0] = empty_tile
                        self._foundry_redirect_pending_splitter = (empty_tile, sdir)
                        print(f"[{self.corner}] foundry redirect: bridge "
                              f"({target[0]},{target[1]}) retargeted → "
                              f"({empty_tile[0]},{empty_tile[1]}), "
                              f"will build splitter {sdir.value}")
                        return False
                # Failed to retarget — fall through to bridge-to-splitter.

        # --- Fallback: bridge targeting the original splitter ---
        if not self._can_spend(ct, GC.BRIDGE_BASE_COST[0]):
            return False
        tpos = xy_to_pos(target)
        spos = xy_to_pos(splitter)
        if t_entry and t_entry[1] is not None and t_entry[3] == self.my_team_cache:
            if ct.can_destroy(tpos):
                ct.destroy(tpos)
                self.tile_cache[target] = (t_entry[0], None, None, None)
                self.building_cache.pop(target, None)
                self.bridge_target_cache.pop(target, None)
        if ct.can_build_bridge(tpos, spos):
            ct.build_bridge(tpos, spos)
            self.building_cache[target] = (EntityType.BRIDGE, splitter)
            self.bridge_target_cache[target] = splitter
            self.tile_cache[target] = (
                t_entry[0] if t_entry else Environment.EMPTY,
                -1, EntityType.BRIDGE, self.my_team_cache,
            )
            print(f"[{self.corner}] foundry redirect: ({target[0]},{target[1]}) "
                  f"→ bridge to splitter ({splitter[0]},{splitter[1]})")
        self._foundry_redirect_queue.pop(0)
        return not self._foundry_redirect_queue

    def _cardinal_gap_tiles(self, from_xy, to_xy):
        """Return cardinal-walk tiles from `from_xy` (exclusive) toward
        `to_xy` (EXCLUSIVE). Used to manually extend chain_path so the
        last tile is CARDINALLY ADJACENT to the goal — the chain
        builder then fills in the conveyor direction toward the goal
        via its `is_last` logic. Walks x first then y.
        Returns [] if from_xy == to_xy, cardinally adjacent, or the
        path would cross a core tile (can't build on core)."""
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        if abs(dx) + abs(dy) <= 1:
            return []
        fx, fy = from_xy
        tx, ty = to_xy
        gap = []
        # Walk until one step short of to_xy
        while True:
            remaining_dx = tx - fx
            remaining_dy = ty - fy
            if abs(remaining_dx) + abs(remaining_dy) <= 1:
                break
            if remaining_dx != 0:
                fx += 1 if remaining_dx > 0 else -1
            else:
                fy += 1 if remaining_dy > 0 else -1
            if self._is_core_tile((fx, fy)):
                return []
            gap.append((fx, fy))
        return gap

    def _recheck_build_race(self, ct, xy):
        """Called after can_build_{conveyor,bridge} unexpectedly fails on xy.

        Another builder bot may have placed something on xy THIS turn.
        Query the engine for the fresh state and react:
        - Allied conveyor/splitter/bridge/armoured_conveyor/core: chain is
          effectively complete (someone else built what we needed). Update
          tile_cache and return 'chain_complete'.
        - Enemy walkable building: update tile_cache so the caller's next
          iteration walks onto it and fires. Return 'enemy_walkable'.
        - Nothing helpful: return None — caller handles as before.
        """
        try:
            pos = xy_to_pos(xy)
            bid = ct.get_tile_building_id(pos)
        except Exception:
            return None
        if bid is None:
            return None
        try:
            etype = ct.get_entity_type(bid)
            team = ct.get_team(bid)
        except Exception:
            return None
        old = self.tile_cache.get(xy)
        env = old[0] if old else Environment.EMPTY
        self.tile_cache[xy] = (env, bid, etype, team)

        if team == self.my_team_cache and etype in (
                EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE,
                EntityType.ARMOURED_CONVEYOR, EntityType.CORE):
            return 'chain_complete'
        if team != self.my_team_cache and etype in _ENEMY_WALKABLE:
            return 'enemy_walkable'
        return None

    def _bridge_target_taken(self, bridge_at):
        """True if another bot has already placed allied transport at
        bridge_at (our planned bridge position). We must see the tile in
        current vision to avoid false positives from stale tile_cache."""
        if bridge_at in self._chain_built:
            return False
        if bridge_at not in self._last_vision_set:
            return False
        entry = self.tile_cache.get(bridge_at)
        if entry is None or entry[1] is None:
            return False
        return (entry[3] == self.my_team_cache
                and entry[2] in _ALLIED_TRANSPORT)

    def _bridge_has_valid_target(self, bridge_at):
        """Quick check: does a valid bridge target exist from bridge_at?
        Returns True if _optimal_bridge COULD succeed (ignoring resources/cooldown).
        Includes allied transport as last-resort targets."""
        bx, by = bridge_at
        r = 3
        goal = self._chain_goal or self.core_pos
        bridge_core_dsq = euclidean_dist_sq(bridge_at, goal)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                dsq = dx * dx + dy * dy
                if dsq == 0 or dsq > GC.BRIDGE_TARGET_RADIUS_SQ:
                    continue
                xy = (bx + dx, by + dy)
                if xy in self.known_walls:
                    continue
                if self._is_enemy_core_tile(xy):
                    continue
                if self._is_core_tile(xy):
                    return True
                entry = self.tile_cache.get(xy)
                if entry is None:
                    continue
                if entry[0] == Environment.WALL:
                    continue
                dist = euclidean_dist_sq(xy, goal)
                if dist >= bridge_core_dsq:
                    continue
                # Markers are always on empty tiles and can be built over
                # by any team — treat them as empty for bridge validity.
                if entry[1] is None or entry[2] == EntityType.MARKER:
                    if entry[0] in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                        continue
                    return True
                elif entry[3] == self.my_team_cache and entry[2] == EntityType.ROAD:
                    return True
                elif entry[3] != self.my_team_cache and entry[2] in _ENEMY_WALKABLE:
                    return True
                elif (entry[3] == self.my_team_cache
                      and entry[2] == EntityType.BARRIER):
                    return True  # Builder destroys the barrier before use
                elif (entry[3] == self.my_team_cache
                      and entry[2] in _ALLIED_TRANSPORT):
                    return True  # Allied transport as last resort
        return False

    def _is_enemy_core_tile(self, xy):
        """Check if xy is one of the 9 tiles of the enemy 3×3 core."""
        if self.enemy_core_pos is None:
            return False
        cx, cy = self.enemy_core_pos
        return abs(xy[0] - cx) <= 1 and abs(xy[1] - cy) <= 1

    def _optimal_bridge(self, ct, bridge_at):
        """Build a bridge at bridge_at targeting the closest valid tile toward core.

        The bot builds the bridge from its current position (must be within
        action radius² of bridge_at). The bridge is placed at bridge_at,
        then the bot walks onto it and pathfinds to the bridge target.

        Valid targets (in priority order):
        1. Allied core tile, empty (no ore), road, enemy walkable
        2. Allied conveyor/bridge/splitter (last resort — capacity checked
           when the bridge walker arrives via observe_conveyor)
        """
        bx, by = bridge_at
        r = 3  # sqrt(BRIDGE_TARGET_RADIUS_SQ=9)
        best = None
        best_dist = float('inf')
        best_transport = None  # Fallback: allied transport (last resort)
        best_transport_dist = float('inf')
        goal = self._chain_goal or self.core_pos
        bridge_core_dsq = euclidean_dist_sq(bridge_at, goal)
        bl = self._bridge_target_blacklist

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                dsq = dx * dx + dy * dy
                if dsq == 0 or dsq > GC.BRIDGE_TARGET_RADIUS_SQ:
                    continue
                xy = (bx + dx, by + dy)
                if xy in self.known_walls:
                    continue
                if xy in bl:
                    continue
                if self._is_enemy_core_tile(xy):
                    continue
                # The pending foundry tile is treated as if a foundry
                # is already there — don't ever target it with a bridge.
                if (self.foundry_tile is not None
                        and xy == self.foundry_tile
                        and self._chain_goal != self.foundry_tile):
                    continue

                # Core tiles are NOT valid bridge targets: ct.build_bridge
                # rejects them, and targeting the core from the failure-
                # path race check would falsely declare the chain complete
                # (the core is always present). Let the chain hit the core
                # via regular conveyors on core-adjacent tiles instead.
                if self._is_core_tile(xy):
                    continue

                entry = self.tile_cache.get(xy)
                if entry is None:
                    continue
                env, bid, etype, team = entry
                if env == Environment.WALL:
                    continue

                # Allied barriers are OK as bridge targets: the chain
                # builder destroys the barrier before stepping onto it
                # and building the conveyor that replaces the bridge
                # landing tile. Treat them like empty tiles here.
                if (bid is not None and team == self.my_team_cache
                        and etype == EntityType.BARRIER):
                    bid = None

                dist_to_core = euclidean_dist_sq(xy, goal)
                if dist_to_core >= bridge_core_dsq:
                    continue

                # Markers are always on empty tiles and can be built over
                # by any team — treat a marker-bearing tile like an empty
                # tile for bridge target selection.
                if bid is not None and etype == EntityType.MARKER:
                    bid = None

                if bid is None:
                    if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                        continue  # skip ore tiles
                    # Skip enclosed pockets — landing a bridge in a tile
                    # whose entire 8-neighbourhood is unwalkable means the
                    # bot can never step off it to extend the chain.
                    # Counts walls AND non-walkable buildings (harvesters,
                    # barriers, turrets) as blockers.
                    tx, ty = xy
                    enclosed = True
                    for nx, ny in ((tx-1,ty-1),(tx,ty-1),(tx+1,ty-1),
                                   (tx-1,ty),(tx+1,ty),
                                   (tx-1,ty+1),(tx,ty+1),(tx+1,ty+1)):
                        nxy = (nx, ny)
                        if nxy in self.known_walls:
                            continue
                        ne = self.tile_cache.get(nxy)
                        if ne is None:
                            enclosed = False  # unknown tile = assume open
                            break
                        if ne[0] == Environment.WALL:
                            continue
                        n_bid, n_etype = ne[1], ne[2]
                        if n_bid is not None and n_etype not in _WALKABLE_BUILDINGS:
                            continue  # non-walkable building blocks step-off
                        enclosed = False
                        break
                    if enclosed:
                        continue
                    # Standard valid target (empty tile)
                    if dist_to_core < best_dist:
                        best_dist = dist_to_core
                        best = xy
                elif team == self.my_team_cache and etype == EntityType.ROAD:
                    if xy in self.bot_pos_cache:
                        continue
                    # Road on ore = claimed by another bot
                    if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                        continue
                    if dist_to_core < best_dist:
                        best_dist = dist_to_core
                        best = xy
                elif team != self.my_team_cache and etype in _ENEMY_WALKABLE:
                    if dist_to_core < best_dist:
                        best_dist = dist_to_core
                        best = xy
                elif (team == self.my_team_cache
                      and etype in _ALLIED_TRANSPORT
                      and xy not in self._chain_built):
                    # For axionite chains, splitters are primary targets
                    # (the chain's goal IS a splitter tile).
                    if (self.current_chain_type == 'axionite'
                            and etype == EntityType.SPLITTER):
                        if dist_to_core < best_dist:
                            best_dist = dist_to_core
                            best = xy
                    # Other allied transport — last resort fallback
                    elif dist_to_core < best_transport_dist:
                        best_transport_dist = dist_to_core
                        best_transport = xy
                else:
                    continue

        # Prefer standard targets; fall back to allied transport
        if best is None:
            best = best_transport
            best_dist = best_transport_dist

        if best is None:
            if DEBUG: print(f"[{self.corner}] _optimal_bridge({bridge_at}): no target found bl={bl}", file=sys.stderr)
            return False
        if DEBUG: print(f"[{self.corner}] _optimal_bridge({bridge_at}): best={best}", file=sys.stderr)

        my_xy = self._my_xy

        # Bot must be within action radius of bridge_at
        adx = my_xy[0] - bridge_at[0]
        ady = my_xy[1] - bridge_at[1]
        if adx * adx + ady * ady > GC.ACTION_RADIUS_SQ:
            return False

        if ct.get_action_cooldown() > 0:
            return True  # Will retry next turn

        bridge_pos = xy_to_pos(bridge_at)
        target_pos = xy_to_pos(best)

        # Clear bridge_at if it has a building we can remove
        br_entry = self.tile_cache.get(bridge_at)
        if br_entry and br_entry[1] is not None:
            if br_entry[3] == self.my_team_cache:
                if br_entry[2] in _DESTROYABLE:
                    # Allied road/marker/barrier — safe to destroy
                    if ct.can_destroy(bridge_pos):
                        ct.destroy(bridge_pos)
                        self.tile_cache[bridge_at] = (br_entry[0], None, None, None)
                elif bridge_at in self._chain_built:
                    # Our own chain conveyor — safe to replace with bridge
                    if ct.can_destroy(bridge_pos):
                        ct.destroy(bridge_pos)
                        self.tile_cache[bridge_at] = (br_entry[0], None, None, None)
                        self._chain_built.discard(bridge_at)
                        self.building_cache.pop(bridge_at, None)
                elif br_entry[2] in _ALLIED_TRANSPORT:
                    # Another bot already owns this tile — we can't build
                    # on it and we can't declare chain_complete just
                    # because a foreign network is here (their direction
                    # might not even feed our core). Return False so the
                    # caller can back off and try a different approach
                    # (walk-through merge, bridge from build_from, etc.).
                    return False
                else:
                    return False  # Allied non-destroyable (harvester, sentinel, etc.)
            elif br_entry[2] in _ENEMY_WALKABLE:
                return False  # Enemy walkable at bridge source — can't bridge here

        if ct.can_build_bridge(bridge_pos, target_pos):
            if not self._has_economy_reserve(ct, GC.BRIDGE_BASE_COST[0]):
                return True  # Pretend success — wait next turn for reserve
            ct.build_bridge(bridge_pos, target_pos)
            if DEBUG: print(f"[{self.corner}] bridge ({bridge_at[0]},{bridge_at[1]})->({best[0]},{best[1]})",
                  file=sys.stderr)

            # Move onto the bridge, then walk to target and replan
            self._bridge_walk_target = best
            self.chain_path = None
            self.path = None
            self.path_index = 0

            # Per-piece chain padding for the just-built bridge: same
            # rationale as the conveyor case in _step_chain_to_core.
            # Skip the immediate step-onto-bridge if there's anything
            # to pave — once on the bridge the bot teleports to the
            # bridge target and can't easily come back to seal these
            # tiles. The queue drains over the next few turns via the
            # top-of-step drain in _step_chain_to_core, after which
            # the bridge_walk handler picks up and walks across.
            queued = False
            if (self.core_pos is not None
                    and max(abs(my_xy[0] - self.core_pos[0]),
                            abs(my_xy[1] - self.core_pos[1])) <= ROAD_PROTECT_RANGE):
                # Bridges have an explicit `target_pos` (not a cardinal
                # neighbour) — exclude the bridge target itself, but
                # we don't have a meaningful "upstream" exclusion the
                # way conveyors do, so pass None for that.
                queued = self._chain_protect_populate(bridge_at, None, best)

            # Try to step onto the bridge immediately, unless we have
            # roads to pave first.
            if not queued and bridge_at != my_xy:
                d = cardinal_direction_between(my_xy, bridge_at)
                if ct.can_move(d):
                    ct.move(d)

            return True

        # Bridge placement failed — race check
        race = self._recheck_build_race(ct, bridge_at)
        if race == 'chain_complete':
            print(f"[{self.corner}] bridge race complete at {bridge_at}")
            self._chain_reached_end(bridge_at, ct=ct)
            return True
        # Only race-check `best` when it's NOT a core tile. Core tiles
        # always report CORE from the engine, which would trigger a
        # false 'chain_complete' even though the bridge never landed.
        if race is None and not self._is_core_tile(best):
            race = self._recheck_build_race(ct, best)
            if race == 'chain_complete':
                print(f"[{self.corner}] bridge race complete at {best}")
                self._chain_reached_end(best, ct=ct)
                return True
        return False

    def _try_bridge_over(self, ct, from_xy, blocked_xy):
        """Build a bridge from from_xy to jump over blocked_xy toward core."""
        return self._optimal_bridge(ct, from_xy)

    # ------------------------------------------------------------------ #
    #  Merge                                                              #
    # ------------------------------------------------------------------ #

    def _do_merge(self, ct):
        """Merge our chain into the observed conveyor by building a linking conveyor."""
        merge_tile = self._chain_merge_tile
        if merge_tile is None:
            self._chain_complete()
            return

        # Walk within action range of merge_tile if we drifted off
        # during the 4-turn observe window. Without this guard,
        # can_build_conveyor silently returns False and the merge
        # gets wrongly marked as skipped.
        my_xy = self._my_xy
        mdx = my_xy[0] - merge_tile[0]
        mdy = my_xy[1] - merge_tile[1]
        if mdx * mdx + mdy * mdy > GC.ACTION_RADIUS_SQ:
            if self.path is None or self.path_index >= len(self.path):
                self.path = self._compute_path(my_xy, merge_tile)
                self.path_index = 0
            if self.path:
                self._follow_path(ct)
            else:
                # Can't reach merge_tile — drop the observation and
                # let chain_to_core re-evaluate.
                self._observe_skip.add(self.observe_xy)
                self.observe_xy = None
                self.observe_target = None
                self.step = "chain_to_core"
            return

        if ct.get_action_cooldown() > 0:
            return

        merge_pos = xy_to_pos(merge_tile)
        merge_dir = cardinal_direction_between(merge_tile, self.observe_xy)

        t_entry = self.tile_cache.get(merge_tile)
        # If our chain conveyor is already at merge_tile AND actually
        # points at observe_xy, the merge is already in place. A
        # conveyor here pointing somewhere else (e.g. toward the core)
        # does NOT count as merged — flow wouldn't reach observe_xy.
        if (t_entry and t_entry[1] is not None
                and t_entry[3] == self.my_team_cache
                and t_entry[2] in _ALLIED_TRANSPORT):
            bc = self.building_cache.get(merge_tile)
            existing_dir = bc[1] if bc else None
            points_at_observed = False
            if t_entry[2] == EntityType.BRIDGE:
                # Bridges deliver to their target regardless of facing.
                bt = self.bridge_target_cache.get(merge_tile)
                points_at_observed = (bt == self.observe_xy)
            elif existing_dir is not None and isinstance(existing_dir, Direction):
                edx, edy = DIRECTION_DELTAS[existing_dir]
                points_at_observed = (
                    (merge_tile[0] + edx, merge_tile[1] + edy) == self.observe_xy)
            if points_at_observed:
                if DEBUG: print(f"[{self.corner}] merge already in place at ({merge_tile[0]},{merge_tile[1]})",
                      file=sys.stderr)
                self._post_merge_transition()
                return
            # Wrong direction — destroy and rebuild facing observe_xy.
            if ct.get_action_cooldown() > 0:
                return
            if ct.can_destroy(merge_pos):
                ct.destroy(merge_pos)
                self.tile_cache[merge_tile] = (t_entry[0], None, None, None)
                self.building_cache.pop(merge_tile, None)
                t_entry = self.tile_cache.get(merge_tile)
                if DEBUG: print(f"[{self.corner}] merge redirect: destroyed wrong-dir conv at "
                      f"({merge_tile[0]},{merge_tile[1]})", file=sys.stderr)

        if t_entry and t_entry[1] is not None and t_entry[3] == self.my_team_cache:
            if t_entry[2] in _DESTROYABLE and ct.can_destroy(merge_pos):
                ct.destroy(merge_pos)
                self.tile_cache[merge_tile] = (t_entry[0], None, None, None)

        if ct.can_build_conveyor(merge_pos, merge_dir):
            if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                return  # Wait for reserve
            ct.build_conveyor(merge_pos, merge_dir)
            self.building_cache[merge_tile] = (EntityType.CONVEYOR, merge_dir)
            if DEBUG: print(f"[{self.corner}] merged at ({self.observe_xy[0]},{self.observe_xy[1]})",
                  file=sys.stderr)
            self._post_merge_transition()
        else:
            # Race check — someone may have built the linking conveyor already
            race = self._recheck_build_race(ct, merge_tile)
            if race == 'chain_complete':
                self._post_merge_transition()
                return
            self._observe_skip.add(self.observe_xy)
            self.observe_xy = None
            self.observe_target = None
            self.step = "chain_to_core"

    def _post_merge_transition(self):
        """Branch point after a successful merge. Titanium chains complete
        here; axionite chains enter the follow-chain state machine so the
        bot can walk the merged network to its end and place a foundry."""
        if self.current_chain_type == 'axionite':
            # Start following from the observed conveyor tile.
            self.follow_chain_pos = self.observe_xy
            self.follow_chain_prev_type = 'conveyor'
            self.axionite_chain_substep = 'follow_chain'
            self.observe_xy = None
            self.observe_target = None
            # current_chain_type stays 'axionite' so the dispatch keeps
            # calling _step_chain_to_core_axionite.
            self.step = "chain_to_core"
            print(f"[{self.corner}] AX merged → follow_chain from "
                  f"({self.follow_chain_pos[0]},{self.follow_chain_pos[1]})")
            return
        self._chain_complete()

    def _chain_reached_end(self, merged_at=None, ct=None):
        """Chain completion helper — call this INSTEAD of _chain_complete()
        from any site where the chain reached its planned end (merged
        implicitly, bridge target filled by ally, reached a core tile, etc.).

        Titanium: calls _chain_complete() directly — unchanged behaviour.

        Axionite: transitions into the follow-chain state machine so the
        bot can walk the merged network to its end and place a foundry.
        - If `merged_at` is provided and is a core tile, we already know
          the final conveyor is our last built tile — jump straight to
          Scenario 1B.
        - If `merged_at` is provided and is NOT a core tile, start the
          follow-chain loop from there.
        - If `merged_at` is None, infer it from `chain_path[-1]`'s output
          direction. If we can't, fall back to the direct-placement path
          (check_foundry → select_foundry_tile → ...).
        """
        if self.current_chain_type != 'axionite':
            # Before declaring the titanium chain complete, pave any
            # empty 8-neighbours of the final core-targeting conveyor
            # with cheap roads. Roads are walkable for us but occupy
            # the tile so an enemy disruptor can't drop a building
            # adjacent to our core-feeding conveyor without first
            # spending an action to clear them. If there's nothing to
            # pave, _chain_seal_setup falls straight through to
            # _chain_complete().
            anchor = merged_at if merged_at is not None else (
                self.chain_path[-1] if self.chain_path else self.chain_start)
            if not self._chain_seal_setup(anchor):
                self._chain_complete()
            return

        # If we're currently in the chain_to_foundry substep, the chain
        # has reached its goal (X for the reroute path, or an existing
        # foundry tile). Verify the last built building actually
        # delivers into the goal before transitioning. If not, replan
        # so the chain can extend or bridge the final hop.
        #
        # Bypass: if `merged_at` is a foreign allied transport (built
        # by another bot, not in our own _chain_built), our chain
        # delivers via THEIR network — skip the goal-verification path
        # entirely and fall through to the implicit-merge → follow_chain
        # handling further down. The other bot's chain may extend
        # somewhere very different from our planned goal, so we walk
        # it to find a final placement spot.
        is_foreign_merge = False
        if (merged_at is not None
                and self._chain_built is not None
                and merged_at not in self._chain_built):
            mi = self.tile_cache.get(merged_at)
            if (mi is not None and mi[1] is not None
                    and mi[3] == self.my_team_cache
                    and mi[2] in _ALLIED_TRANSPORT):
                is_foreign_merge = True
                print(f"[{self.corner}] AX chain_to_foundry: foreign merge "
                      f"at ({merged_at[0]},{merged_at[1]}) → follow_chain")
        if (self.axionite_chain_substep == 'chain_to_foundry'
                and not is_foreign_merge):
            # Goal: existing foundry, else X (splitter tile), else the
            # bootstrap foundry_tile itself. Matches _chain_goal set by
            # select_foundry_tile / reroute.
            goal = (self.foundry_target
                    or self.foundry_splitter_tile
                    or self.foundry_tile)
            last = None
            if self.chain_path and self.chain_index > 0:
                idx = min(self.chain_index - 1, len(self.chain_path) - 1)
                last = self.chain_path[idx]
            elif self.chain_start is not None:
                last = self.chain_start
            connected = False
            if last is not None and goal is not None:
                if last == goal:
                    connected = True
                else:
                    lb = self.building_cache.get(last)
                    if lb is not None:
                        lbt = lb[0]
                        if (lbt in (EntityType.CONVEYOR,
                                    EntityType.ARMOURED_CONVEYOR)
                                and lb[1] is not None):
                            adjacent = (abs(last[0] - goal[0])
                                        + abs(last[1] - goal[1])) == 1
                            if adjacent:
                                ldx, ldy = DIRECTION_DELTAS[lb[1]]
                                connected = (last[0] + ldx,
                                             last[1] + ldy) == goal
                        elif lbt == EntityType.BRIDGE:
                            connected = self.bridge_target_cache.get(last) == goal
            if not connected:
                # If last is within bridge range of goal, try a direct
                # bridge instead of extending with conveyors — faster
                # and avoids multi-tile gap-fill.
                if (last is not None and goal is not None and ct is not None
                        and euclidean_dist_sq(last, goal) <= GC.BRIDGE_TARGET_RADIUS_SQ
                        and ct.get_action_cooldown() == 0
                        and self._can_spend(ct, GC.BRIDGE_BASE_COST[0])):
                    lpos = xy_to_pos(last)
                    gpos = xy_to_pos(goal)
                    # Destroy existing conveyor at last (free) then bridge
                    # atomically — only destroy if we can build right after.
                    le = self.tile_cache.get(last)
                    if (le and le[1] is not None
                            and le[3] == self.my_team_cache):
                        if ct.can_destroy(lpos):
                            ct.destroy(lpos)
                            self.tile_cache[last] = (le[0], None, None, None)
                            self.building_cache.pop(last, None)
                    if ct.can_build_bridge(lpos, gpos):
                        ct.build_bridge(lpos, gpos)
                        self.building_cache[last] = (EntityType.BRIDGE, goal)
                        self.bridge_target_cache[last] = goal
                        self.tile_cache[last] = (
                            le[0] if le else Environment.EMPTY,
                            -1, EntityType.BRIDGE, self.my_team_cache,
                        )
                        print(f"[{self.corner}] AX chain_reached_end: "
                              f"direct bridge ({last[0]},{last[1]})"
                              f"→({goal[0]},{goal[1]})")
                        # Bridge connects last to goal — chain is done.
                        self.chain_path = None
                        self.chain_index = 0
                        self._chain_start_fixed = False
                        goal_bc = self.building_cache.get(goal)
                        if self.foundry_target is not None:
                            self.axionite_chain_substep = 'complete'
                        elif goal_bc and goal_bc[0] == EntityType.SPLITTER:
                            self.axionite_chain_substep = 'complete'
                        else:
                            self.axionite_chain_substep = 'retrofit_foundry_splitter'
                        self.step = "chain_to_core"
                        return

                # Chain fell short of the goal (astar_monotonic picks a
                # "closest to start" endpoint when the goal isn't a core
                # tile). Manually extend chain_path by cardinal-walking
                # from `last` to `goal` so the chain builder finishes
                # the gap next turn(s).
                if (last is not None and goal is not None
                        and self.chain_path is not None):
                    gap = self._cardinal_gap_tiles(last, goal)
                    if gap:
                        first_gap = gap[0]
                        # The existing conveyor at `last` was built
                        # with its direction pointing at the OLD next
                        # tile in chain_path. After extension, its next
                        # tile is `first_gap`. If `last`'s current
                        # direction doesn't face `first_gap`, destroy
                        # the existing conveyor at `last` and rewind
                        # chain_index so the builder rebuilds it with
                        # the correct direction.
                        rewind_last = False
                        lb = self.building_cache.get(last)
                        if (lb is not None
                                and lb[0] in (EntityType.CONVEYOR,
                                              EntityType.ARMOURED_CONVEYOR)
                                and lb[1] is not None):
                            ldx, ldy = DIRECTION_DELTAS[lb[1]]
                            if (last[0] + ldx, last[1] + ldy) != first_gap:
                                rewind_last = True
                        elif lb is None:
                            # building_cache has no entry — the conveyor
                            # at last may have been destroyed already
                            # (e.g. by the direct bridge shortcut above).
                            # Check tile_cache: if no building, last needs
                            # rebuilding with the correct direction.
                            le = self.tile_cache.get(last)
                            if le is None or le[1] is None:
                                rewind_last = True
                        if rewind_last:
                            if self._chain_built is not None:
                                self._chain_built.discard(last)
                            if ct is not None:
                                lpos = xy_to_pos(last)
                                le = self.tile_cache.get(last)
                                if (le and le[1] is not None
                                        and le[3] == self.my_team_cache
                                        and ct.can_destroy(lpos)):
                                    ct.destroy(lpos)
                                    self.tile_cache[last] = (le[0], None, None, None)
                                    self.building_cache.pop(last, None)
                        print(f"[{self.corner}] AX chain_reached_end: last={last} "
                              f"not connected to goal={goal} — extend by {gap}"
                              + (" (destroy+rewind last)" if rewind_last else ""))
                        old_len = len(self.chain_path)
                        self.chain_path = self.chain_path + gap
                        if rewind_last:
                            # Point chain_index at `last` so the chain
                            # builder rebuilds it with the new direction.
                            # When old_len == 0, `last` is chain_start, not
                            # an entry in chain_path — rewind to index 0
                            # and clear `_chain_start_fixed` so the
                            # chain-start fix block reruns and rebuilds it
                            # facing the new first_gap. Subtracting 1 from
                            # old_len = 0 would yield chain_index = -1 and
                            # explode at the chain_path[idx-1] access next
                            # turn.
                            if old_len == 0:
                                self.chain_index = 0
                                self._chain_start_fixed = False
                            else:
                                self.chain_index = old_len - 1
                        else:
                            self.chain_index = old_len
                        self.step = "chain_to_core"
                        return
                # No gap to extend (last == goal somehow), or no
                # chain_path — accept and proceed.
                print(f"[{self.corner}] AX chain_reached_end: last={last} "
                      f"not connected to goal={goal} — proceed anyway")
            # Connected — proceed.
            self.chain_path = None
            self.chain_index = 0
            self._chain_start_fixed = False
            if self.foundry_target is not None:
                print(f"[{self.corner}] AX chain_reached_end: reached "
                      f"existing foundry ({self.foundry_target[0]},"
                      f"{self.foundry_target[1]})")
                self.axionite_chain_substep = 'complete'
            else:
                # If the goal tile already has a splitter (placed by
                # another bot or a previous chain), the chain is done —
                # no need to retrofit or place a foundry.
                tgt = self.foundry_splitter_tile or self.foundry_tile
                tgt_bc = self.building_cache.get(tgt) if tgt else None
                if tgt_bc and tgt_bc[0] == EntityType.SPLITTER:
                    print(f"[{self.corner}] AX chain_reached_end: "
                          f"goal ({tgt[0]},{tgt[1]}) already a splitter "
                          f"→ complete")
                    self.axionite_chain_substep = 'complete'
                else:
                    tgt_str = (f"X=({tgt[0]},{tgt[1]})"
                               if self.foundry_splitter_tile is not None
                               else f"foundry_tile=({tgt[0]},{tgt[1]})")
                    print(f"[{self.corner}] AX chain_reached_end: reached "
                          f"{tgt_str} → retrofit_foundry_splitter")
                    self.axionite_chain_substep = 'retrofit_foundry_splitter'
            self.step = "chain_to_core"
            return

        # Figure out the implicit merge point if caller didn't supply one.
        if merged_at is None:
            last = None
            if self.chain_path and self.chain_index > 0:
                idx = min(self.chain_index - 1, len(self.chain_path) - 1)
                last = self.chain_path[idx]
            elif self.chain_start is not None:
                last = self.chain_start
            if last is None:
                print(f"[{self.corner}] AX chain_reached_end: no last — abandon")
                self._axionite_abandon("chain_reached_end no last")
                return
            # If `last` is already an allied splitter or foundry, the chain
            # already terminates at a productive endpoint (Scenario 4) —
            # mark done and exit instead of looping through direct-placement.
            last_entry = self.tile_cache.get(last)
            if (last_entry and last_entry[1] is not None
                    and last_entry[3] == self.my_team_cache
                    and last_entry[2] in (EntityType.SPLITTER, EntityType.FOUNDRY)):
                print(f"[{self.corner}] AX chain_reached_end: last=({last[0]},{last[1]}) "
                      f"is {last_entry[2]} → scenario 4 done")
                self.axionite_chain_done = True
                self.axionite_chain_substep = 'complete'
                self.step = "chain_to_core"
                return
            cached = self.building_cache.get(last)
            if (cached is not None
                    and cached[0] in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR)
                    and cached[1] is not None):
                ddx, ddy = DIRECTION_DELTAS[cached[1]]
                merged_at = (last[0] + ddx, last[1] + ddy)
                print(f"[{self.corner}] AX chain_reached_end: inferred merge "
                      f"at ({merged_at[0]},{merged_at[1]}) from last={last}")
                # Validate: the inferred tile must actually be a merge
                # point (core or allied transport). If not, the chain
                # plan fell one (or more) tiles short of a real merge —
                # replan from here instead of entering follow_chain on
                # an empty tile.
                if not self._is_core_tile(merged_at):
                    m_entry = self.tile_cache.get(merged_at)
                    is_allied_transport = (
                        m_entry is not None
                        and m_entry[1] is not None
                        and m_entry[3] == self.my_team_cache
                        and m_entry[2] in (EntityType.CONVEYOR,
                                           EntityType.ARMOURED_CONVEYOR,
                                           EntityType.BRIDGE,
                                           EntityType.SPLITTER,
                                           EntityType.FOUNDRY))
                    # Fall back to building_cache if tile_cache doesn't
                    # have the entry yet (vision timing race).
                    if not is_allied_transport:
                        mb = self.building_cache.get(merged_at)
                        if mb is not None and mb[0] in (
                                EntityType.CONVEYOR,
                                EntityType.ARMOURED_CONVEYOR,
                                EntityType.BRIDGE,
                                EntityType.SPLITTER,
                                EntityType.FOUNDRY):
                            is_allied_transport = True
                    if not is_allied_transport:
                        # Detect replan loops: if we already tried to
                        # replan from this exact `last` tile, A* is just
                        # going to return the same plan again. Fall back
                        # to direct placement with a fresh goal.
                        if self._last_replan_from == last:
                            print(f"[{self.corner}] AX chain_reached_end: "
                                  f"replan loop at ({last[0]},{last[1]}) "
                                  f"→ direct placement fallback")
                            self._last_replan_from = None
                            self.axionite_chain_substep = None
                            self.chain_path = None
                            self._chain_start_fixed = False
                            self.chain_start = self._my_xy
                            self._chain_built = {self._my_xy}
                            self._chain_goal = None
                            self.foundry_tile = None
                            self.foundry_target = None
                            self.step = "chain_to_core"
                            return
                        self._last_replan_from = last
                        print(f"[{self.corner}] AX chain_reached_end: "
                              f"({merged_at[0]},{merged_at[1]}) not a merge "
                              f"point → replan from last=({last[0]},{last[1]})")
                        self.chain_path = None
                        self.chain_start = last
                        self._chain_built = {last}
                        self._chain_start_fixed = False
                        self.axionite_chain_substep = 'chain_to_foundry'
                        self.step = "chain_to_core"
                        return
            else:
                print(f"[{self.corner}] AX chain_reached_end: couldn't infer — "
                      f"fallback to direct placement")
                # Reset chain state to trigger check_foundry next call.
                self.axionite_chain_substep = None
                self.chain_path = None
                self._chain_start_fixed = False
                self.chain_start = self._my_xy
                self._chain_built = {self._my_xy}
                self._chain_goal = None
                self.step = "chain_to_core"
                return

        # Axionite chain reached a core tile → our last built conveyor
        # targets the core. Route to scenario 1A (splitter-on-final,
        # foundry-adjacent) if the building BEFORE final is a bridge;
        # otherwise scenario 1B (foundry-on-final, retrofit upstream
        # conveyors into splitters).
        if self._is_core_tile(merged_at):
            final = None
            if self.chain_path and self.chain_index > 0:
                idx = min(self.chain_index - 1, len(self.chain_path) - 1)
                final = self.chain_path[idx]
            elif self.chain_start is not None:
                final = self.chain_start
            if final is None:
                print(f"[{self.corner}] AX chain reached core but no final — abandon")
                self._axionite_abandon("reached core no final")
                return

            prev_type = 'conveyor'
            # Check chain_path[index-2] first — that's the building we
            # placed right before `final`.
            if self.chain_path and self.chain_index >= 2:
                prior = self.chain_path[self.chain_index - 2]
                pcached = self.building_cache.get(prior)
                if pcached and pcached[0] == EntityType.BRIDGE:
                    prev_type = 'bridge'
            # Fallback: look behind `final` via its conveyor facing
            # (cardinal neighbour case — previous building is directly
            # adjacent and delivers via conveyor/bridge adjacency).
            if prev_type == 'conveyor':
                fcached = self.building_cache.get(final)
                if (fcached
                        and fcached[0] in (EntityType.CONVEYOR,
                                           EntityType.ARMOURED_CONVEYOR)
                        and fcached[1] is not None):
                    fdx, fdy = DIRECTION_DELTAS[fcached[1]]
                    behind = (final[0] - fdx, final[1] - fdy)
                    bcached = self.building_cache.get(behind)
                    if bcached and bcached[0] == EntityType.BRIDGE:
                        prev_type = 'bridge'
            # Fallback 2: scan allied bridges whose TARGET is final_pos.
            # A bridge may sit several tiles away (up to BRIDGE_TARGET
            # radius) and still deliver INTO final — this happens when
            # scenario_2/3 restarted a chain into a slot already fed by
            # a bridge from a prior merged network.
            if prev_type == 'conveyor':
                for bxy, btarget in self.bridge_target_cache.items():
                    if btarget != final:
                        continue
                    be = self.tile_cache.get(bxy)
                    if (be and be[1] is not None
                            and be[2] == EntityType.BRIDGE
                            and be[3] == self.my_team_cache):
                        prev_type = 'bridge'
                        print(f"[{self.corner}] AX chain_reached_end: "
                              f"bridge-scan found bridge at ({bxy[0]},{bxy[1]}) "
                              f"→ final=({final[0]},{final[1]})")
                        break

            self.follow_chain_final_pos = final
            self.follow_chain_prev_type = prev_type
            self._scenario_step = 0
            if prev_type == 'bridge':
                self.axionite_chain_substep = 'scenario_1a'
                scen = '1A'
            else:
                self.axionite_chain_substep = 'scenario_1b'
                scen = '1B'
            self.step = "chain_to_core"
            print(f"[{self.corner}] AX chain reached core at "
                  f"({merged_at[0]},{merged_at[1]}) → scenario {scen} "
                  f"final=({final[0]},{final[1]}) prev={prev_type}")
            return

        # Otherwise: start walking the merged network from merged_at.
        self.follow_chain_pos = merged_at
        self.follow_chain_prev_type = 'conveyor'
        self.axionite_chain_substep = 'follow_chain'
        self.step = "chain_to_core"
        print(f"[{self.corner}] AX implicit merge → follow_chain from "
              f"({merged_at[0]},{merged_at[1]})")

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #

    def _scan_opportunistic_harvester(self, conv_xy, conv_target_xy):
        """After placing a chain conveyor at conv_xy, look at its 4 cardinal
        neighbors (excluding conv_target_xy, the direction it outputs) for
        an unclaimed titanium ore. If found, mark it as pending so the bot
        builds a free harvester feeding into this conveyor next turn."""
        if self._pending_harvester_xy is not None:
            return  # already have one pending
        if self.harvesters_built >= MAX_HARVESTERS_PER_BOT:
            return
        for nxy in neighbors_4(conv_xy[0], conv_xy[1]):
            if nxy == conv_target_xy:
                continue  # where conveyor outputs; can't place harvester here
            if nxy in self._chain_built:
                continue  # already a chain conveyor
            entry = self.tile_cache.get(nxy)
            if entry is None or entry[0] != Environment.ORE_TITANIUM:
                continue
            if self._is_ore_enclosed(nxy):
                continue
            bid, etype, team = entry[1], entry[2], entry[3]
            # Already has an allied harvester — skip
            if bid is not None and team == self.my_team_cache and etype == EntityType.HARVESTER:
                continue
            # Allied non-destroyable (sentinel/turret/etc) — skip
            if (bid is not None and team == self.my_team_cache
                    and etype not in (EntityType.ROAD, EntityType.MARKER)):
                continue
            # Allied bot standing on it — claimed
            if nxy in self.bot_pos_cache and self.bot_pos_cache[nxy][1] == self.my_team_cache:
                continue
            self._pending_harvester_xy = nxy
            return

    def _handle_pending_harvester(self, ct, my_xy):
        """Process a pending opportunistic harvester placement.

        Returns True if this consumed the turn (chain should not advance),
        False if the pending was cleared and chain can resume normally.
        """
        ph_xy = self._pending_harvester_xy
        ph_entry = self.tile_cache.get(ph_xy)

        # Validate: still an ore?
        if ph_entry is None or ph_entry[0] != Environment.ORE_TITANIUM:
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return False

        bid, etype, team = ph_entry[1], ph_entry[2], ph_entry[3]

        # Already has our harvester — done
        if bid is not None and team == self.my_team_cache and etype == EntityType.HARVESTER:
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return False

        # Allied non-walkable that we can't destroy — give up
        if (bid is not None and team == self.my_team_cache
                and etype not in (EntityType.ROAD, EntityType.MARKER)):
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return False

        # Can we afford a harvester?
        scale = ct.get_scale_percent() / 100.0
        harvester_cost = int(GC.HARVESTER_BASE_COST[0] * scale)
        ti, _ = ct.get_global_resources()
        if ti < harvester_cost:
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return False

        ph_pos = xy_to_pos(ph_xy)

        # Case: enemy walkable building on the ore — walk onto it and fire
        if bid is not None and team != self.my_team_cache and etype in _ENEMY_WALKABLE:
            if my_xy != ph_xy:
                # Record where to return to (current chain position) and walk onto ore
                if self._pending_return_xy is None:
                    self._pending_return_xy = my_xy
                d = cardinal_direction_between(my_xy, ph_xy)
                if ct.can_move(d):
                    ct.move(d)
                return True
            # Standing on the enemy — fire
            if ct.get_action_cooldown() == 0 and ct.can_fire(ph_pos):
                ct.fire(ph_pos)
            return True

        # If we walked onto the ore to clear an enemy, and now it's clear,
        # walk back to our chain position before building
        if (self._pending_return_xy is not None
                and my_xy == ph_xy
                and my_xy != self._pending_return_xy):
            d = cardinal_direction_between(my_xy, self._pending_return_xy)
            if ct.can_move(d):
                ct.move(d)
                self._pending_return_stuck = 0
            else:
                self._pending_return_stuck = getattr(self, '_pending_return_stuck', 0) + 1
                if self._pending_return_stuck >= 3:
                    self._pending_harvester_xy = None
                    self._pending_return_xy = None
                    self._pending_return_stuck = 0
                    return False
            return True

        # Need to be within action radius to build
        dsq = euclidean_dist_sq(my_xy, ph_xy)
        if dsq > GC.ACTION_RADIUS_SQ:
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return False

        # Allied road on the ore — destroy (free) then build
        if bid is not None and team == self.my_team_cache and etype == EntityType.ROAD:
            if ct.can_destroy(ph_pos):
                ct.destroy(ph_pos)
                self.tile_cache[ph_xy] = (ph_entry[0], None, None, None)
            # fall through to harvester build

        # Build the harvester
        if ct.get_action_cooldown() != 0:
            return True  # wait
        if ct.can_build_harvester(ph_pos):
            if not self._has_economy_reserve(ct, GC.HARVESTER_BASE_COST[0]):
                return True  # Wait next turn for reserve
            ct.build_harvester(ph_pos)
            self.harvesters_built += 1
            if DEBUG: print(f"[{self.corner}] opportunistic harvester #{self.harvesters_built} at ({ph_xy[0]},{ph_xy[1]})",
                  file=sys.stderr)
            self._pending_harvester_xy = None
            self._pending_return_xy = None
            return True  # consumed cooldown; chain continues next turn
        # Can't build (race / unexpected state) — abandon
        self._pending_harvester_xy = None
        self._pending_return_xy = None
        return False

    def _chain_protect_pave_step(self, ct):
        """Per-piece chain padding: drain `_chain_protect_queue` one
        road at a time. Same paving primitive as `_step_chain_seal_roads`,
        but invoked from inside `_step_chain_to_core` between chain
        pieces. Returns True iff the bot's turn was consumed (caller
        should return immediately); False means the queue is now empty
        and the chain build can proceed.
        """
        my_xy = self._my_xy
        # Drop entries that are no longer paveable (built up since,
        # became a wall, etc.). Keeps the queue clean across turns.
        while self._chain_protect_queue:
            head = self._chain_protect_queue[0]
            if not self._chain_seal_can_pave(head):
                self._chain_protect_queue.pop(0)
                continue
            break
        if not self._chain_protect_queue:
            return False
        target = min(
            self._chain_protect_queue,
            key=lambda xy: euclidean_dist_sq(my_xy, xy))
        d2 = euclidean_dist_sq(my_xy, target)
        if d2 <= GC.ACTION_RADIUS_SQ and ct.get_action_cooldown() == 0:
            if not self._can_spend(ct, GC.ROAD_BASE_COST[0]):
                # Out of titanium for the padding — abandon the queue
                # rather than block chain progress on a nice-to-have.
                self._chain_protect_queue = []
                return False
            tpos = xy_to_pos(target)
            if ct.can_build_road(tpos):
                ct.build_road(tpos)
                entry = self.tile_cache.get(target)
                env = entry[0] if entry else Environment.EMPTY
                self.tile_cache[target] = (env, None, None, None)
                if DEBUG:
                    print(f"[{self.corner}] chain-protect road@({target[0]},{target[1]})",
                          file=sys.stderr)
            self._chain_protect_queue.remove(target)
            return True
        # Walk closer.
        if (self.path is None or not self.path or self.path[-1] != target):
            path = self._compute_path(my_xy, target, max_nodes=80)
            if path is None:
                # Unreachable — drop and try the next.
                self._chain_protect_queue.remove(target)
                self.path = None
                return True
            self.path = path
            self.path_index = 0
        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0
        return True

    def _chain_protect_populate(self, target_xy, build_from, next_tile):
        """Populate `_chain_protect_queue` with paveable 8-neighbours
        of `target_xy`, excluding the chain's own incoming and outgoing
        tiles. Latches `_chain_protect_target` so the queue isn't
        re-populated every turn for the same chain step. Returns True
        if anything got queued.
        """
        if self._chain_protect_target == target_xy:
            return False
        self._chain_protect_target = target_xy
        self._chain_protect_queue = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sxy = (target_xy[0] + dx, target_xy[1] + dy)
                if sxy == build_from or sxy == next_tile:
                    continue
                if self._chain_seal_can_pave(sxy):
                    self._chain_protect_queue.append(sxy)
        return bool(self._chain_protect_queue)

    def _chain_seal_can_pave(self, xy):
        """True iff `xy` is a tile we should pave with a road as part
        of sealing the area around the final core-targeting conveyor.
        Skips walls, ores, the core 3x3 footprint, and any tile that
        already holds a building.
        """
        if xy in self.known_walls:
            return False
        if self.core_pos is not None:
            cx, cy = self.core_pos
            if abs(xy[0] - cx) <= 1 and abs(xy[1] - cy) <= 1:
                return False
        entry = self.tile_cache.get(xy)
        if entry is None:
            return False
        env, bid, _, _ = entry
        if env == Environment.WALL:
            return False
        if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            return False
        if bid is not None:
            return False
        return True

    def _chain_seal_setup(self, anchor_xy):
        """Stage the post-chain road-seal pass. `anchor_xy` is the
        final core-targeting conveyor we just placed (or merged into).
        Collect every empty 8-neighbour and queue them for paving in
        the new `chain_seal_roads` step.

        Returns True iff there's actually work to do; False means the
        caller should fall straight through to `_chain_complete()`.
        """
        if anchor_xy is None:
            return False
        queue = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (anchor_xy[0] + dx, anchor_xy[1] + dy)
                if self._chain_seal_can_pave(nxy):
                    queue.append(nxy)
        if not queue:
            return False
        self._chain_seal_queue = queue
        self._chain_seal_anchor = anchor_xy
        self.step = "chain_seal_roads"
        self.path = None
        self.path_index = 0
        return True

    def _step_chain_seal_roads(self, ct):
        """Pave queued empty 8-neighbours of the final conveyor with
        roads, then call `_chain_complete()`. One road per turn at
        most (action cooldown). Drops a tile from the queue once an
        ally builds something there or A* can't reach it. Aborts the
        seal entirely if we can't afford a road — completion shouldn't
        block forever on a nice-to-have.
        """
        my_xy = self._my_xy

        # Drop tiles that are no longer paveable (built up since we
        # queued them, became a wall, etc.).
        self._chain_seal_queue = [
            xy for xy in self._chain_seal_queue
            if self._chain_seal_can_pave(xy)
        ]
        if not self._chain_seal_queue:
            self._chain_seal_anchor = None
            self._chain_complete()
            return

        # Closest queued tile wins.
        target = min(
            self._chain_seal_queue,
            key=lambda xy: euclidean_dist_sq(my_xy, xy))

        # In action range → build the road directly.
        d2 = euclidean_dist_sq(my_xy, target)
        if d2 <= GC.ACTION_RADIUS_SQ and ct.get_action_cooldown() == 0:
            if not self._can_spend(ct, GC.ROAD_BASE_COST[0]):
                # Can't afford even a single road — abort the seal,
                # finish the chain anyway. Roads here are nice-to-have.
                self._chain_seal_queue = []
                self._chain_seal_anchor = None
                self._chain_complete()
                return
            tpos = xy_to_pos(target)
            if ct.can_build_road(tpos):
                ct.build_road(tpos)
                # Mark the tile as built so the next scan replaces
                # bid=None with the real bid. Matches the cache
                # update in _follow_path's road-build branch.
                entry = self.tile_cache.get(target)
                env = entry[0] if entry else Environment.EMPTY
                self.tile_cache[target] = (env, None, None, None)
                self._chain_seal_queue.remove(target)
                if DEBUG:
                    print(f"[{self.corner}] chain-seal road@({target[0]},{target[1]})",
                          file=sys.stderr)
            else:
                # can_build_road said no (e.g. tile briefly blocked) —
                # drop and try another.
                self._chain_seal_queue.remove(target)
            return

        # Walk closer.
        if (self.path is None or not self.path or self.path[-1] != target):
            path = self._compute_path(my_xy, target, max_nodes=100)
            if path is None:
                # Unreachable — drop and try the next.
                self._chain_seal_queue.remove(target)
                self.path = None
                return
            self.path = path
            self.path_index = 0
        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0

    def _chain_complete(self):
        """Clean up chain state. Go to stored ores or explore.

        Bumps the per-type chain counter on a clean titanium complete.
        The axionite Step 7b.6 flips `axionite_chain_done = True` before
        calling here so the eligibility gate self-enforces.
        """
        if self.current_chain_type == 'titanium':
            self.titanium_chains_completed += 1
            self._first_chain_done = True
        # Reset chain-type dispatch + axionite substep state so the
        # next ore picks fresh.
        self.current_chain_type = None
        self._chain_goal = None
        self.axionite_chain_substep = None
        self.foundry_target = None
        self.foundry_tile = None
        self._foundry_destroyed_etype = None
        self.follow_chain_pos = None
        self.follow_chain_prev_type = None
        self.follow_chain_final_pos = None
        self._follow_chain_seen = set()
        self._scenario_23_target = None
        self._scenario_step = 0
        self.splitter_queue = []
        self._last_replan_from = None
        self.foundry_splitter_tile = None
        self._retrofit_x_dir = None
        self._suppress_chain_merge = False
        self._splitter_bridge_fixes = []
        self._splitter_output_check = None
        self._foundry_redirect_queue = []
        self._foundry_redirect_splitter = None
        self._foundry_redirect_foundry = None
        self._foundry_redirect_pending_splitter = None
        self._1b_first_splitter = None
        self.chain_path = None
        self.chain_index = 0
        self._chain_built = None
        self._chain_start_fixed = False
        self._chain_seal_queue = []
        self._chain_seal_anchor = None
        self._chain_protect_queue = []
        self._chain_protect_target = None
        self._chain_merge_tile = None
        self._observe_skip = None
        self._bridge_walk_target = None
        self._bridge_walk_turns = 0
        self._bridge_target_blacklist = set()
        self._bridge_walk_via = None
        self._pending_harvester_xy = None
        self._pending_return_xy = None
        self._chain_conv_dir = None
        self.observe_xy = None
        self.observe_target = None

        if self.stored_ores and self.harvesters_built < MAX_HARVESTERS_PER_BOT:
            self.target_ore = self.stored_ores.pop(0)
            self._from_stored_ore = True
            self.chain_start = None
            self._chain_dir = None
            self.path = None
            self.path_index = 0
            self.step = "goto_ore"
            if DEBUG: print(f"[{self.corner}] next ore at ({self.target_ore[0]},{self.target_ore[1]})",
                  file=sys.stderr)
        else:
            self.chain_start = None
            self._chain_dir = None
            self.path = None
            self.path_index = 0
            self.step = "explore"
            if DEBUG: print(f"[{self.corner}] chain done, exploring", file=sys.stderr)

    # ====================================================================== #
    #  Step 7b: Axionite chain-to-core (routes to a foundry)                  #
    # ====================================================================== #

    def _step_chain_to_core_axionite(self, ct):
        """Axionite chain driver. Substeps:"""
        # Same heal-in-place guard as the titanium chain. Pauses
        # foundry / splitter / chain work while a damaged ally
        # building is in our action radius.
        if self._economy_check_heal_action_radius(ct, self._my_xy) != 'none':
            return
        if self._economy_try_heal_chain_divert(ct, self._my_xy):
            return
        return self._step_chain_to_core_axionite_inner(ct)

    def _step_chain_to_core_axionite_inner(self, ct):
        """Axionite chain driver. Substeps:

        Merge path (triggered by _post_merge_transition after a merge):
        - follow_chain: walk the merged network to its end
        - scenario_1a: bridge→conveyor→core endpoint — destroy+foundry+splitter+2 bridges
        - scenario_1b: conveyor→conveyor→core endpoint — destroy+foundry+queue splitters
        - scenario_1b_queue: retrofit one conveyor from splitter_queue per turn
        - scenario_2_3: chain doesn't reach core yet — restart axionite chain

        Direct-placement path (triggered when the chain reaches the core
        without merging, or on fresh axionite chain entry):
        - check_foundry: is there an existing foundry adjacent to the core?
        - select_foundry_tile: if not, pick a core-adjacent tile to build on
        - chain_to_foundry: delegate to _step_chain_to_core with a goal override
        - place_foundry: pathfind adjacent, destroy building, build foundry
        - ensure_titanium: retrofit a splitter if the foundry lost its Ti feed
        - complete: flip axionite_chain_done, call _chain_complete
        """
        # TLE recovery: skip the planning substeps; let the shared chain
        # builder handle its own recovery if we're already inside it.
        if (self.tle_recovery_turns > 0
                and self.axionite_chain_substep in (None, 'check_foundry',
                                                    'select_foundry_tile',
                                                    'follow_chain',
                                                    'scenario_1a', 'scenario_1b',
                                                    'scenario_1b_queue',
                                                    'scenario_2_3',
                                                    'retrofit_foundry_splitter')):
            return

        if self.axionite_chain_substep is None:
            # Check if a foundry pair exists. If yes, go straight to
            # select_foundry_tile. If not, build toward the core first —
            # the chain's conveyors will populate the ring, and when
            # the chain reaches the core, _reroute_axionite_to_foundry
            # will pick the newly-laid pair.
            existing = self._find_existing_foundry()
            if existing is not None:
                self.axionite_chain_substep = 'check_foundry'
            elif self._pick_foundry_tile() is not None:
                self.axionite_chain_substep = 'check_foundry'
            else:
                # No pair yet — build toward core to populate the ring.
                self._step_chain_to_core(ct)
                return

        s = self.axionite_chain_substep

        # ---- Merge-path substeps (Step 7b.1 + Scenarios 1-4) ----
        if s == 'follow_chain':
            self._follow_merged_chain_step(ct)
            return

        if s == 'scenario_1a':
            if self._do_scenario_1a(ct):
                self.axionite_chain_substep = 'complete'
            return

        if s == 'scenario_1b':
            if self._do_scenario_1b_splitters(ct):
                self.axionite_chain_substep = 'scenario_1b_queue'
            return

        if s == 'scenario_1b_queue':
            if self._do_scenario_1b_queue(ct):
                self.axionite_chain_substep = 'scenario_1b_foundry'
            return

        if s == 'scenario_1b_foundry':
            if self._do_scenario_1b_place_foundry(ct):
                ftile = self.follow_chain_final_pos
                splitter = self._1b_first_splitter
                if splitter and ftile:
                    self._scan_foundry_redirects(ftile, splitter)
                if self._foundry_redirect_queue:
                    self.axionite_chain_substep = 'redirect_foundry'
                else:
                    self.axionite_chain_substep = 'complete'
            return

        if s == 'scenario_2_3':
            # Walk to the dead-end target (the empty/road tile past the
            # last working conveyor/bridge) and restart the chain from
            # there. Add every tile we followed to `_observe_skip` so
            # the restart won't re-merge back into the same broken
            # network — it has to reach the core on its own.
            target = self._scenario_23_target
            if target is None:
                self._axionite_abandon("scenario_2_3 no target")
                return
            if self._scenario_step == 0:
                if self._my_xy == target:
                    self._scenario_step = 1  # fall through to restart
                else:
                    self.path = self._compute_path(self._my_xy, target)
                    self.path_index = 0
                    if self.path is None:
                        # Can't physically reach the dead-end (e.g. bridge
                        # jumped a wall). Our merge still contributed
                        # capacity to the existing network, so treat this
                        # as Scenario 4 and mark done instead of abandoning.
                        print(f"[{self.corner}] AX scenario 2/3 no path to "
                              f"({target[0]},{target[1]}) → scenario 4 done")
                        self.axionite_chain_done = True
                        self.axionite_chain_substep = 'complete'
                        return
                    self._follow_path(ct)
                    return
            # Step 1: restart. Inject the seen tiles into observe_skip,
            # reset chain state, drop back to chain_to_core.
            if self._observe_skip is None:
                self._observe_skip = set()
            for xy in self._follow_chain_seen:
                self._observe_skip.add(xy)
            self.chain_start = self._my_xy
            self.chain_path = None
            self._chain_built = {self._my_xy}
            self._chain_start_fixed = False
            self._chain_goal = None
            self.foundry_target = None
            self.foundry_tile = None
            # Leave axionite_chain_substep = None so _step_chain_to_core_axionite
            # falls into check_foundry → select_foundry_tile → chain_to_foundry.
            self.axionite_chain_substep = None
            self.follow_chain_pos = None
            self.follow_chain_prev_type = None
            self.follow_chain_final_pos = None
            self._follow_chain_seen = set()
            self._scenario_23_target = None
            print(f"[{self.corner}] AX scenario 2/3 → restart chain from "
                  f"({self._my_xy[0]},{self._my_xy[1]})")
            return

        if s == 'check_foundry':
            existing = self._find_existing_foundry()
            if existing is not None:
                self.foundry_target = existing
                self._chain_goal = existing
                self.axionite_chain_substep = 'chain_to_foundry'
                self._suppress_chain_merge = True
                print(f"[{self.corner}] AX → chain to existing foundry "
                      f"({existing[0]},{existing[1]})")
            else:
                self.axionite_chain_substep = 'select_foundry_tile'
            return

        if s == 'select_foundry_tile':
            # Re-check for an existing foundry in case another bot
            # placed one since check_foundry ran last turn.
            existing = self._find_existing_foundry()
            if existing is not None:
                self.foundry_target = existing
                self.foundry_tile = None
                self.foundry_splitter_tile = None
                self._chain_goal = existing
                self.axionite_chain_substep = 'chain_to_foundry'
                self._suppress_chain_merge = True
                print(f"[{self.corner}] AX → chain to existing foundry "
                      f"({existing[0]},{existing[1]})")
                return
            pick = self._pick_foundry_tile()
            if pick is None:
                # No valid (foundry, X) pair yet — build toward core
                # to populate the ring. Reset substep to None so the
                # next call enters the build-to-core path.
                self.axionite_chain_substep = None
                return
            self.foundry_tile, self.foundry_splitter_tile = pick
            # Route to X (splitter tile) when we have one — the sub-chain
            # delivers axionite onto X which is later retrofit into a
            # splitter. When foundry_splitter_tile is None (bootstrap
            # case, no titanium ring yet), target foundry_tile directly
            # and skip the retrofit step.
            self._chain_goal = self.foundry_splitter_tile or self.foundry_tile
            self.axionite_chain_substep = 'chain_to_foundry'
            # Suppress merging: the chain must land cardinally-adjacent
            # to X and point at X. If we merged into X, the follow_chain
            # scenario 1B would destroy X (wrong — we need X to become
            # a splitter).
            self._suppress_chain_merge = True
            x_str = (f"X=({self.foundry_splitter_tile[0]},{self.foundry_splitter_tile[1]})"
                     if self.foundry_splitter_tile is not None else "X=None(bootstrap)")
            print(f"[{self.corner}] AX → chain to {x_str} foundry_tile="
                  f"({self.foundry_tile[0]},{self.foundry_tile[1]})")
            return

        if s == 'chain_to_foundry':
            # Delegate to the shared chain builder. For axionite,
            # _chain_reached_end now handles the successful-reach path
            # directly (transitions substep to place_foundry / complete
            # without clearing chain_start). If chain_start DOES get
            # cleared during this call it means _chain_complete was
            # invoked via a failure path (bridge walk abort, monotonic
            # A* failure, race) — the chain was NOT successful. Don't
            # restore state; let the abandon propagate.
            self._step_chain_to_core(ct)
            return

        if s == 'retrofit_foundry_splitter':
            done = self._do_retrofit_foundry_splitter(ct)
            if done:
                self._scenario_step = 0
                self.axionite_chain_substep = 'place_foundry'
            return

        if s == 'place_foundry':
            done = self._do_place_foundry(ct)
            if done:
                splitter = self.foundry_splitter_tile
                if splitter and self.foundry_tile:
                    self._scan_foundry_redirects(self.foundry_tile, splitter)
                if self._foundry_redirect_queue:
                    self.axionite_chain_substep = 'redirect_foundry'
                else:
                    self.axionite_chain_substep = 'complete'
            return

        if s == 'redirect_foundry':
            if self._process_foundry_redirects(ct):
                self.axionite_chain_substep = 'complete'
            return

        if s == 'complete':
            self.axionite_chain_done = True
            print(f"[{self.corner}] AX axionite chain done")
            self._chain_complete()
            return

        # Unknown substep — safety net.
        self._axionite_abandon("bad substep")

    # ---- Axionite helpers ---- #

    def _axionite_abandon(self, reason):
        print(f"[{self.corner}] AX axionite chain abandoned: {reason}")
        self.current_chain_type = None
        self._chain_goal = None
        self.axionite_chain_substep = None
        self.foundry_tile = None
        self.foundry_target = None
        self._foundry_destroyed_etype = None
        self.follow_chain_pos = None
        self.follow_chain_prev_type = None
        self.follow_chain_final_pos = None
        self._follow_chain_seen = set()
        self._scenario_23_target = None
        self._scenario_step = 0
        self.splitter_queue = []
        self._last_replan_from = None
        self.foundry_splitter_tile = None
        self._retrofit_x_dir = None
        self._suppress_chain_merge = False
        self._splitter_bridge_fixes = []
        self._splitter_output_check = None
        self._foundry_redirect_queue = []
        self._foundry_redirect_splitter = None
        self._foundry_redirect_foundry = None
        self._foundry_redirect_pending_splitter = None
        self._1b_first_splitter = None
        self._chain_complete()

    def _core_adjacent_tiles(self):
        """Yield the 12 tiles cardinally adjacent to the 3x3 core footprint
        (i.e. the ring of buildable tiles that touch a core tile)."""
        if self.core_pos is None:
            return
        cx, cy = self.core_pos
        # Top edge (y = cy - 2)
        for dx in (-1, 0, 1):
            yield (cx + dx, cy - 2)
        # Bottom edge (y = cy + 2)
        for dx in (-1, 0, 1):
            yield (cx + dx, cy + 2)
        # Left / right edges (x = cx ± 2), 3 tiles each
        for dy in (-1, 0, 1):
            yield (cx - 2, cy + dy)
            yield (cx + 2, cy + dy)

    def _find_existing_foundry(self):
        """Return (x,y) of an allied foundry (or foundry-placeholder marker)
        on one of the 12 core-adjacent tiles, or None. Foundry markers
        are treated as pending foundries — once seen, this bot won't try
        to place its own foundry elsewhere; instead its axionite chain
        routes to the marker tile and completes as if it were a foundry."""
        my_team = self.my_team_cache
        bc = self.building_cache
        for xy in self._core_adjacent_tiles():
            e = self.tile_cache.get(xy)
            if e is not None and e[1] is not None and e[3] == my_team:
                if e[2] == EntityType.FOUNDRY:
                    return xy
                # This bot only ever places foundry markers, so any
                # allied marker on the core-adjacent ring is a pending
                # foundry.
                if e[2] == EntityType.MARKER:
                    return xy
            # Fall back to building_cache for entries outside vision
            # (markers aren't cached there, so only foundries). Iterate
            # the tiny core_adj set, not all of building_cache, so this
            # stays O(12) instead of O(building_cache size).
            cached = bc.get(xy)
            if cached is not None and cached[0] == EntityType.FOUNDRY:
                return xy
        return None

    def _reroute_axionite_to_foundry(self, ct, last_conv_xy):
        """Axionite chain was about to step onto a core tile. If an
        allied foundry already exists adjacent to the core, route to
        it (via chain_to_foundry with foundry_target set). Otherwise
        pick a (foundry_tile, X) pair, destroy `last_conv_xy` (the
        last built conveyor currently pointing into the core), and
        reroute the chain from there toward X with merging suppressed."""
        # Prefer an existing allied foundry to avoid multiple bots all
        # placing redundant foundries at the same tile.
        existing = self._find_existing_foundry()
        if existing is not None:
            self.foundry_target = existing
            self.foundry_tile = None
            self.foundry_splitter_tile = None
            self._chain_goal = existing
            # Destroy the last conveyor pointing at the core so we
            # don't continue pouring axionite into it.
            if last_conv_xy is not None:
                lpos = xy_to_pos(last_conv_xy)
                l_entry = self.tile_cache.get(last_conv_xy)
                if (l_entry and l_entry[1] is not None
                        and l_entry[3] == self.my_team_cache
                        and ct.can_destroy(lpos)):
                    ct.destroy(lpos)
                    self.tile_cache[last_conv_xy] = (l_entry[0], None, None, None)
                    self.building_cache.pop(last_conv_xy, None)
                    if self._chain_built is not None:
                        self._chain_built.discard(last_conv_xy)
            self.chain_start = last_conv_xy
            self.chain_path = None
            self.chain_index = 0
            self._chain_start_fixed = False
            self._chain_built = {last_conv_xy} if last_conv_xy is not None else set()
            self.axionite_chain_substep = 'chain_to_foundry'
            self._suppress_chain_merge = True
            self.step = "chain_to_core"
            print(f"[{self.corner}] AX reroute: existing foundry at "
                  f"({existing[0]},{existing[1]}) — destroy ({last_conv_xy}), "
                  f"chain to existing")
            return
        pick = self._pick_foundry_tile()
        if pick is None:
            # No valid (foundry, X) pair yet — core ring doesn't have
            # enough conveyors. Enter check_foundry to retry each turn;
            # titanium chains will populate the ring.
            self.axionite_chain_substep = 'check_foundry'
            print(f"[{self.corner}] AX reroute: no valid (foundry,X) — "
                  f"waiting for ring to populate")
            return
        self.foundry_tile, self.foundry_splitter_tile = pick
        # Route to X (splitter tile), NOT foundry_tile.
        self._chain_goal = self.foundry_splitter_tile
        # Destroy the last built conveyor — it currently points at the
        # core. Free destroy (allied transport we built).
        if last_conv_xy is not None:
            lpos = xy_to_pos(last_conv_xy)
            l_entry = self.tile_cache.get(last_conv_xy)
            if (l_entry and l_entry[1] is not None
                    and l_entry[3] == self.my_team_cache
                    and ct.can_destroy(lpos)):
                ct.destroy(lpos)
                self.tile_cache[last_conv_xy] = (l_entry[0], None, None, None)
                self.building_cache.pop(last_conv_xy, None)
                if self._chain_built is not None:
                    self._chain_built.discard(last_conv_xy)
        # Restart the chain from the destroyed tile, heading to X.
        self.chain_start = last_conv_xy
        self.chain_path = None
        self.chain_index = 0
        self._chain_start_fixed = False
        self._chain_built = {last_conv_xy} if last_conv_xy is not None else set()
        self.axionite_chain_substep = 'chain_to_foundry'
        # Sub-chain must not merge into existing allied transport —
        # the whole point is to redirect to our own foundry.
        self._suppress_chain_merge = True
        self.step = "chain_to_core"
        print(f"[{self.corner}] AX reroute: destroyed ({last_conv_xy[0] if last_conv_xy else '?'},"
              f"{last_conv_xy[1] if last_conv_xy else '?'}), chaining to X="
              f"({self.foundry_splitter_tile[0]},{self.foundry_splitter_tile[1]}) "
              f"foundry_tile=({self.foundry_tile[0]},{self.foundry_tile[1]})")

    def _pick_foundry_tile(self):
        """Pick (foundry_tile, X) per Strategy.md. Primary rule:
          1. foundry_tile is empty or allied road
          2. X is an allied conveyor cardinally adjacent to foundry_tile
             whose direction targets a core tile
          3. Both tiles are in the 12-tile core-adjacent ring
        Fallback (when no valid primary pair exists): two cardinally-
        adjacent allied conveyors inside the core ring — the first is
        destroyed to place the foundry, the second becomes a splitter
        keeping its original direction. Returns (foundry_tile, X) or None."""
        core_adj = set(self._core_adjacent_tiles())
        my = self.my_team_cache

        def _is_allied_conveyor(xy):
            # Must be a conveyor in BOTH building_cache and tile_cache.
            # Exclude splitters, foundries, and other non-conveyor types.
            c = self.building_cache.get(xy)
            if c is None or c[0] not in (
                    EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                return False
            e = self.tile_cache.get(xy)
            if e is None or e[3] != my:
                return False
            # Double-check tile_cache type — building_cache can be stale.
            if e[2] in (EntityType.SPLITTER, EntityType.FOUNDRY):
                return False
            return True

        def _conv_targets_core(xy):
            c = self.building_cache.get(xy)
            if c is None or c[1] is None:
                return None
            cdx, cdy = DIRECTION_DELTAS[c[1]]
            t = (xy[0] + cdx, xy[1] + cdy)
            return t if self._is_core_tile(t) else None

        # Primary: empty/road foundry tile + core-targeting X conveyor.
        for ftile in core_adj:
            e = self.tile_cache.get(ftile)
            if e is None:
                continue
            env, bid, etype, team = e
            is_empty = bid is None and env == Environment.EMPTY
            is_road  = (bid is not None and team == my
                        and etype == EntityType.ROAD)
            if not (is_empty or is_road):
                continue
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                xxy = (ftile[0] + ddx, ftile[1] + ddy)
                if xxy not in core_adj:
                    continue
                if not _is_allied_conveyor(xxy):
                    continue
                if _conv_targets_core(xxy) is not None:
                    return (ftile, xxy)

        # Fallback 1: two cardinally-adjacent allied conveyors in the core
        # ring. Prefer the pair where X targets the core (so its side
        # output after retrofit keeps flowing into the core); fall back
        # to any adjacent pair.
        best_pair = None
        for a in core_adj:
            if not _is_allied_conveyor(a):
                continue
            for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                b = (a[0] + ddx, a[1] + ddy)
                if b not in core_adj:
                    continue
                if not _is_allied_conveyor(b):
                    continue
                # `a` = foundry_tile (destroyed), `b` = X (retrofit splitter)
                if _conv_targets_core(b) is not None:
                    return (a, b)
                if best_pair is None:
                    best_pair = (a, b)
        if best_pair is not None:
            return best_pair

        return None

    def _do_place_foundry(self, ct):
        """Step off the foundry tile if needed, clear any building there,
        build the foundry. Returns True on success."""
        if self.foundry_tile is None:
            self._axionite_abandon("no foundry_tile in place_foundry")
            return False
        ftile = self.foundry_tile
        my_xy = self._my_xy

        # Step 1: if we're standing ON the foundry tile, step off to an
        # adjacent empty/road tile (foundries are non-walkable — the
        # builder cannot remain on the tile it's about to build on).
        if my_xy == ftile:
            for nx, ny in ((ftile[0]+1, ftile[1]), (ftile[0]-1, ftile[1]),
                           (ftile[0], ftile[1]+1), (ftile[0], ftile[1]-1)):
                ne = self.tile_cache.get((nx, ny))
                if ne is None:
                    continue
                if ne[0] == Environment.WALL:
                    continue
                if ne[1] is not None:
                    # Only step onto walkable buildings.
                    if ne[2] not in (EntityType.ROAD, EntityType.CONVEYOR,
                                     EntityType.SPLITTER, EntityType.BRIDGE,
                                     EntityType.ARMOURED_CONVEYOR, EntityType.CORE):
                        continue
                d = cardinal_direction_between(ftile, (nx, ny))
                if ct.can_move(d):
                    ct.move(d)
                    return False  # moved — build next turn
            # No adjacent walkable tile — try any build path.
            return False

        # Step 2: we must be within action range of ftile to build.
        if euclidean_dist_sq(my_xy, ftile) > GC.ACTION_RADIUS_SQ:
            self.path = self._compute_path(my_xy, ftile)
            self.path_index = 0
            if self.path is None:
                self._axionite_abandon("no path to foundry_tile")
                return False
            self._follow_path(ct)
            return False

        # Resource check. Foundry is a one-time strategic investment so
        # we bypass the MIN_TITANIUM reserve. If we can't afford yet,
        # drop a foundry placeholder marker on the tile (free, no
        # cooldown) so (a) other bots see it as a pending foundry and
        # don't try to build their own, (b) the tile stays non-walkable,
        # and (c) anything that gets built on top gets destroyed and the
        # marker re-placed until we have resources.
        fpos = xy_to_pos(ftile)
        f_entry = self.tile_cache.get(ftile)
        ti, _ = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0
        foundry_cost = int(GC.FOUNDRY_BASE_COST[0] * scale)
        if ti < foundry_cost:
            self._maintain_foundry_marker(ct, ftile, fpos, f_entry)
            return False  # wait for titanium

        if ct.get_action_cooldown() > 0:
            return False

        # Clear any allied building on ftile (free destroy) + build foundry
        # in the same turn. A foundry marker we placed earlier is allied
        # and handled by this same branch (destroy then build).
        if f_entry is not None and f_entry[1] is not None:
            self._foundry_destroyed_etype = f_entry[2]
            if f_entry[3] == self.my_team_cache:
                if ct.can_destroy(fpos):
                    ct.destroy(fpos)
                    self.tile_cache[ftile] = (f_entry[0], None, None, None)
        else:
            self._foundry_destroyed_etype = None
        if ct.can_build_foundry(fpos):
            ct.build_foundry(fpos)
            self.building_cache[ftile] = (EntityType.FOUNDRY, None)
            self.tile_cache[ftile] = (
                f_entry[0] if f_entry else Environment.EMPTY,
                -1, EntityType.FOUNDRY, self.my_team_cache,
            )
            print(f"[{self.corner}] AX foundry at ({ftile[0]},{ftile[1]})")
            return True
        return False

    def _maintain_foundry_marker(self, ct, ftile, fpos, f_entry):
        """Place or refresh the foundry placeholder marker at ftile.

        If the tile already holds our marker with FOUNDRY_MARKER_VALUE,
        do nothing. If something else got built on top, destroy it and
        re-place the marker. If the tile is empty, just place the marker.
        Marker placement is free and does not consume action cooldown,
        but we can place at most one marker per round."""
        if euclidean_dist_sq(self._my_xy, ftile) > GC.ACTION_RADIUS_SQ:
            return  # out of action range — can't place marker
        if f_entry is not None and f_entry[1] is not None:
            etype = f_entry[2]
            team = f_entry[3]
            if etype == EntityType.FOUNDRY and team == self.my_team_cache:
                # Another bot finished the foundry for us.
                self.building_cache[ftile] = (EntityType.FOUNDRY, None)
                return
            if etype == EntityType.MARKER and team == self.my_team_cache:
                bid = f_entry[1]
                val = None
                if bid is not None and bid != -1:
                    try:
                        val = ct.get_marker_value(bid)
                    except Exception:
                        val = None
                if val == FOUNDRY_MARKER_VALUE:
                    return  # marker intact, nothing to do
                # Wrong-value allied marker — overwrite with our value.
                # place_marker overwrites allied markers directly.
            else:
                # Something non-marker (ally road/conveyor, or enemy) sits
                # on our foundry tile. Destroy it so we can re-place the
                # marker. Only destroy if we own it and have cooldown.
                if team == self.my_team_cache and ct.get_action_cooldown() == 0:
                    if ct.can_destroy(fpos):
                        ct.destroy(fpos)
                        self.tile_cache[ftile] = (f_entry[0], None, None, None)
                        self.building_cache.pop(ftile, None)
                # Fall through: attempt marker placement below.
        if ct.can_place_marker(fpos):
            ct.place_marker(fpos, FOUNDRY_MARKER_VALUE)
            env = f_entry[0] if f_entry else Environment.EMPTY
            self.tile_cache[ftile] = (env, -1, EntityType.MARKER,
                                      self.my_team_cache)
            print(f"[{self.corner}] AX foundry marker at "
                  f"({ftile[0]},{ftile[1]})")

    def _do_retrofit_foundry_splitter(self, ct):
        """Retrofit conveyor X (foundry_splitter_tile) into a splitter
        keeping X's original direction. X is the conveyor cardinally
        adjacent to foundry_tile that was identified by _pick_foundry_tile.
        After retrofit the splitter's back receives axionite flow from
        the reroute sub-chain, its forward output continues into the
        core, and its side output (toward foundry_tile) feeds the
        foundry. Mini state machine using self._scenario_step:
          0: pathfind within action range of X, capture X's direction
          1: wait for reserve, destroy X, build splitter atomically
        Returns True when the splitter is built."""
        xtile = self.foundry_splitter_tile
        if xtile is None:
            self._axionite_abandon("retrofit_splitter: no X tile")
            return False
        step = self._scenario_step
        if step == 0:
            # Capture X's direction NOW so we don't lose it after the
            # destroy in step 1 (which clears building_cache[xtile]).
            cached = self.building_cache.get(xtile)
            if cached is None or cached[1] is None:
                # Try to read live via the engine if in vision.
                xe = self.tile_cache.get(xtile)
                if (xe and xe[1] is not None
                        and xe[2] in (EntityType.CONVEYOR,
                                      EntityType.ARMOURED_CONVEYOR)
                        and xe[3] == self.my_team_cache):
                    try:
                        self._retrofit_x_dir = ct.get_direction(xe[1])
                    except Exception:
                        self._retrofit_x_dir = None
                if self._retrofit_x_dir is None:
                    # Not in vision yet — walk closer and retry.
                    if euclidean_dist_sq(self._my_xy, xtile) > GC.BUILDER_BOT_VISION_RADIUS_SQ:
                        self.path = self._compute_path(self._my_xy, xtile)
                        self.path_index = 0
                        if self.path is None:
                            self._axionite_abandon("retrofit_splitter: no path to X")
                            return False
                        self._follow_path(ct)
                    return False
            else:
                self._retrofit_x_dir = cached[1]
            if euclidean_dist_sq(self._my_xy, xtile) > GC.ACTION_RADIUS_SQ:
                self.path = self._compute_path(self._my_xy, xtile)
                self.path_index = 0
                if self.path is None:
                    self._axionite_abandon("retrofit_splitter: no path to X")
                    return False
                self._follow_path(ct)
                return False
            if self._my_xy == xtile:
                self._step_off_tile(ct, xtile)
                return False
            self._scenario_step = 1
            return False
        if step == 1:
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                return False
            if self._retrofit_x_dir is None:
                self._axionite_abandon("retrofit_splitter: no x_dir saved")
                return False
            x_dir = self._retrofit_x_dir
            if not self._splitter_dir_valid(xtile, x_dir, self.foundry_tile):
                self._axionite_abandon(
                    f"retrofit_splitter: x_dir {x_dir.value} invalid (back=core/foundry)")
                return False
            xpos = xy_to_pos(xtile)
            x_entry = self.tile_cache.get(xtile)
            # Destroy X (free action — no cooldown consumed) then build.
            if (x_entry and x_entry[1] is not None
                    and x_entry[3] == self.my_team_cache):
                if ct.can_destroy(xpos):
                    ct.destroy(xpos)
                    self.tile_cache[xtile] = (x_entry[0], None, None, None)
                    self.building_cache.pop(xtile, None)
            if ct.can_build_splitter(xpos, x_dir):
                ct.build_splitter(xpos, x_dir)
                self.building_cache[xtile] = (EntityType.SPLITTER, x_dir)
                self.tile_cache[xtile] = (
                    Environment.EMPTY, -1, EntityType.SPLITTER, self.my_team_cache,
                )
                print(f"[{self.corner}] AX foundry splitter retrofit at "
                      f"({xtile[0]},{xtile[1]}) facing {x_dir.value}")
                self._retrofit_x_dir = None
                self._queue_splitter_bridge_fixes(xtile, x_dir)
                if self._splitter_bridge_fixes or self._splitter_output_check:
                    self._scenario_step = 2
                    return False
                return True
            return False

        if step == 2:
            if self._process_splitter_bridge_fixes(ct):
                return True
            return False

    def _ensure_foundry_titanium_supply(self, ct):
        """Step 7b.5. If the destroyed building was an allied conveyor
        that was targeting the core, the foundry already has titanium
        flow — done. Otherwise find a core-targeting conveyor X adjacent
        to the foundry and replace it with a splitter whose output
        direction is the same as Y (the conveyor feeding X), so the
        splitter alternately feeds the foundry and the core."""
        ftile = self.foundry_tile

        # Happy path: we destroyed a conveyor that was already feeding
        # the core — the existing chain now feeds the foundry's tile.
        if self._foundry_destroyed_etype in (EntityType.CONVEYOR,
                                              EntityType.ARMOURED_CONVEYOR,
                                              EntityType.SPLITTER):
            if DEBUG:
                print(f"[{self.corner}] foundry Ti supply preserved "
                      f"(destroyed {self._foundry_destroyed_etype})",
                      file=sys.stderr)
            return True

        # Find conveyor X adjacent to foundry, targeting the core.
        my_team = self.my_team_cache
        conv_x = None
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (ftile[0] + ddx, ftile[1] + ddy)
            cached = self.building_cache.get(nxy)
            if cached is None or cached[0] != EntityType.CONVEYOR:
                continue
            cdir = cached[1]
            if cdir is None:
                continue
            cdx, cdy = DIRECTION_DELTAS[cdir]
            if self._is_core_tile((nxy[0] + cdx, nxy[1] + cdy)):
                conv_x = (nxy, cdir)
                break

        if conv_x is None:
            # No Ti feed to retrofit — foundry works with axionite only
            # (still useful as the axionite receiver). Log and finish.
            if DEBUG:
                print(f"[{self.corner}] foundry has no Ti feed, continuing",
                      file=sys.stderr)
            return True

        x_xy, x_dir = conv_x

        # Find conveyor Y that targets X — Y's direction determines the
        # splitter facing so the Ti flow continues in the same vector.
        splitter_dir = None
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cand_xy = (x_xy[0] + ddx, x_xy[1] + ddy)
            cached = self.building_cache.get(cand_xy)
            if cached is None or cached[0] != EntityType.CONVEYOR:
                continue
            cdir = cached[1]
            if cdir is None:
                continue
            cdx, cdy = DIRECTION_DELTAS[cdir]
            if (cand_xy[0] + cdx, cand_xy[1] + cdy) == x_xy:
                splitter_dir = cdir
                break
        if splitter_dir is None:
            # Fall back to X's own direction (pointing at the core).
            splitter_dir = x_dir
        # Validate: the splitter's back must not be core or foundry.
        # The foundry must remain a front/side output so Ti flows in.
        validated = self._pick_splitter_dir(
            x_xy, splitter_dir, ftile, ftile)
        if validated is None:
            if DEBUG:
                print(f"[{self.corner}] foundry Ti splitter: no valid dir at "
                      f"({x_xy[0]},{x_xy[1]}), skipping", file=sys.stderr)
            return True  # best-effort; foundry still runs on axionite only
        splitter_dir = validated

        # Walk adjacent to X if we aren't already.
        my_xy = self._my_xy
        if euclidean_dist_sq(my_xy, x_xy) > GC.ACTION_RADIUS_SQ:
            self.path = self._compute_path(my_xy, x_xy)
            self.path_index = 0
            if self.path is None:
                if DEBUG:
                    print(f"[{self.corner}] no path to splitter site, skipping",
                          file=sys.stderr)
                return True  # best-effort; foundry still runs
            self._follow_path(ct)
            return False

        if ct.get_action_cooldown() > 0:
            return False

        # Wait for reserve BEFORE destroying X so we don't end up
        # with an empty tile and no splitter on retry.
        if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
            return False  # wait
        x_pos = xy_to_pos(x_xy)
        if ct.can_destroy(x_pos):
            ct.destroy(x_pos)
            xe = self.tile_cache.get(x_xy)
            self.tile_cache[x_xy] = (xe[0] if xe else Environment.EMPTY,
                                      None, None, None)
        if ct.can_build_splitter(x_pos, splitter_dir):
            ct.build_splitter(x_pos, splitter_dir)
            self.building_cache[x_xy] = (EntityType.SPLITTER, splitter_dir)
            print(f"[{self.corner}] AX splitter retrofit at ({x_xy[0]},{x_xy[1]}) "
                  f"facing {splitter_dir.value}")
            return True
        return False

    # ====================================================================== #
    #  Step 7b.1 — follow the merged chain + Scenario 1 handlers              #
    # ====================================================================== #

    def _follow_merged_chain_step(self, ct):
        """Walk one step along the merged network. Updates
        `follow_chain_pos` / `follow_chain_prev_type`, and transitions
        to a scenario substep when the end of the chain is detected.

        The bot physically walks alongside `pos` each turn — this both
        keeps us within cache range of the next tile to inspect and
        (critically) populates tile_cache with terrain we'll need later
        if the chain ends in a dead-end and scenario_2/3 has to pathfind
        around a wall the bridge hopped over."""
        pos = self.follow_chain_pos
        if pos is None:
            self._axionite_abandon("follow_chain no pos")
            return

        # Remember every building we traverse so Scenario 2/3 can mark
        # them as "skip observe" when it restarts the chain from past
        # the dead end — prevents re-merging into the same network.
        self._follow_chain_seen.add(pos)

        # Walk one A* step toward pos each turn. This expands vision so
        # we can traverse the rest of the network via tile_cache. If a
        # path exists, walk and return. If no foot path exists (bridge
        # hopped a wall), FALL THROUGH to inspect pos from cache — we
        # don't need to physically be on pos to read its entry.
        if euclidean_dist_sq(self._my_xy, pos) > 1:
            path = self._compute_path(self._my_xy, pos)
            if path:
                self.path = path
                self.path_index = 0
                self._follow_path(ct)
                return

        entry = self.tile_cache.get(pos)
        if entry is None or entry[1] is None:
            # Lost the building — retry next turn (we already walked above).
            return
        etype = entry[2]
        team = entry[3]
        if team != self.my_team_cache:
            # Foreign building ate our target — abandon.
            self._axionite_abandon("follow hit foreign")
            return

        # End at splitter / foundry (Scenario 4)
        if etype in (EntityType.SPLITTER, EntityType.FOUNDRY):
            print(f"[{self.corner}] AX follow → scenario 4 at ({pos[0]},{pos[1]}) {etype}")
            self.axionite_chain_done = True
            self.axionite_chain_substep = 'complete'
            return

        # Determine target tile + next prev_type
        if etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
            cached = self.building_cache.get(pos)
            if cached is None or cached[1] is None:
                return  # wait for direction cache
            conv_dir = cached[1]
            ddx, ddy = DIRECTION_DELTAS[conv_dir]
            target_xy = (pos[0] + ddx, pos[1] + ddy)
            next_prev = 'conveyor'
        elif etype == EntityType.BRIDGE:
            target_xy = self.bridge_target_cache.get(pos)
            if target_xy is None:
                return
            next_prev = 'bridge'
        else:
            self._axionite_abandon("follow hit non-transport")
            return

        print(f"[{self.corner}] AX follow pos=({pos[0]},{pos[1]}) "
              f"etype={etype} target=({target_xy[0]},{target_xy[1]}) "
              f"is_core={self._is_core_tile(target_xy)} "
              f"prev={self.follow_chain_prev_type} core={self.core_pos}")

        # End at core (Scenario 1)
        if self._is_core_tile(target_xy):
            self.follow_chain_final_pos = pos
            self._scenario_step = 0
            if self.follow_chain_prev_type == 'bridge':
                self.axionite_chain_substep = 'scenario_1a'
                print(f"[{self.corner}] AX follow → scenario 1A final=({pos[0]},{pos[1]})")
            else:
                self.axionite_chain_substep = 'scenario_1b'
                print(f"[{self.corner}] AX follow → scenario 1B final=({pos[0]},{pos[1]})")
            return

        t_entry = self.tile_cache.get(target_xy)
        # Also check building_cache — it may have data from a previous
        # scan even if tile_cache lost the entry (e.g. bot walked away
        # and back, tile_cache refreshed but building_cache retained).
        t_bc = self.building_cache.get(target_xy)
        if t_entry is not None and t_entry[1] is not None and t_entry[3] == self.my_team_cache:
            t_etype = t_entry[2]
            if t_etype in (EntityType.SPLITTER, EntityType.FOUNDRY):
                print(f"[{self.corner}] AX follow → scenario 4 (target {t_etype})")
                self.axionite_chain_done = True
                self.axionite_chain_substep = 'complete'
                return
            if t_etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR,
                           EntityType.BRIDGE):
                if target_xy in self._follow_chain_seen:
                    print(f"[{self.corner}] AX follow cycle detected at "
                          f"({target_xy[0]},{target_xy[1]}) — abandon")
                    self._axionite_abandon("follow cycle")
                    return
                self.follow_chain_pos = target_xy
                self.follow_chain_prev_type = next_prev
                return
        elif (t_bc is not None
              and t_bc[0] in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR,
                              EntityType.BRIDGE, EntityType.SPLITTER,
                              EntityType.FOUNDRY)):
            # building_cache knows about a building at target_xy but
            # tile_cache doesn't (bot is too far to scan). Advance
            # the follow pointer — the bot will walk closer next turn
            # and tile_cache will refresh.
            if t_bc[0] in (EntityType.SPLITTER, EntityType.FOUNDRY):
                print(f"[{self.corner}] AX follow → scenario 4 "
                      f"(target {t_bc[0]} via building_cache)")
                self.axionite_chain_done = True
                self.axionite_chain_substep = 'complete'
                return
            if target_xy in self._follow_chain_seen:
                self._axionite_abandon("follow cycle (building_cache)")
                return
            self.follow_chain_pos = target_xy
            self.follow_chain_prev_type = next_prev
            return
        elif t_entry is None:
            # Target tile not in tile_cache at all — bot can't see it
            # yet. Keep walking toward pos; next turn's scan will
            # hopefully cover target_xy.
            return

        # Target is empty / road / enemy walkable → Scenario 2 (prev conveyor)
        # or Scenario 3 (prev bridge). Collapse both into scenario_2_3.
        print(f"[{self.corner}] AX follow → scenario 2/3 (target "
              f"({target_xy[0]},{target_xy[1]}))")
        self._scenario_23_target = target_xy
        self._scenario_step = 0
        self.axionite_chain_substep = 'scenario_2_3'

    # ---- Scenario 1A: bridge → conveyor → core ---- #

    def _do_scenario_1a(self, ct):
        """Replace a bridge→conveyor→core endpoint with a splitter + foundry
        + 2 bridges. The final conveyor (`follow_chain_final_pos`) sits on
        the bridge's target tile. Steps (one per action cooldown):

        0. Pathfind adjacent to final_pos.
        1. Destroy the final conveyor (free).
        2. Pick a foundry tile cardinal to final_pos (not a core tile).
        3. Build foundry on that tile.
        4. Build splitter on final_pos facing the foundry.
        5. Build bridge 1 on a non-back side of splitter → core.
        6. Build bridge 2 on a side of foundry → core.
        Returns True when all steps complete or on unrecoverable failure."""
        final_pos = self.follow_chain_final_pos
        if final_pos is None:
            self._axionite_abandon("scenario_1a no final_pos")
            return True
        step = self._scenario_step

        # Step 0: move adjacent
        if step == 0:
            if euclidean_dist_sq(self._my_xy, final_pos) > GC.ACTION_RADIUS_SQ:
                self.path = self._compute_path(self._my_xy, final_pos)
                self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                return False
            self._scenario_step = 1
            return False

        # Step 1: pick foundry tile, then wait for resources before any destroy
        if step == 1:
            ft = self._pick_scenario_1a_foundry_tile(final_pos)
            if ft is None:
                self._axionite_abandon("1A no foundry tile")
                return True
            self.foundry_tile = ft
            self._scenario_step = 2
            return False

        # Step 2: build splitter at final_pos FIRST (before foundry).
        if step == 2:
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                return False
            sdir = self._direction_out(final_pos, self.foundry_tile)
            if sdir is None:
                self._axionite_abandon("1A splitter dir none")
                return True
            sdir = self._pick_splitter_dir(
                final_pos, sdir, self.foundry_tile, self.foundry_tile)
            if sdir is None:
                self._axionite_abandon("1A splitter dir invalid (back=core/foundry)")
                return True
            sp = xy_to_pos(final_pos)
            sp_entry = self.tile_cache.get(final_pos)
            if sp_entry and sp_entry[1] is not None and sp_entry[3] == self.my_team_cache:
                if ct.can_destroy(sp):
                    ct.destroy(sp)
                    self.tile_cache[final_pos] = (sp_entry[0], None, None, None)
                    self.building_cache.pop(final_pos, None)
            if ct.can_build_splitter(sp, sdir):
                ct.build_splitter(sp, sdir)
                self.building_cache[final_pos] = (EntityType.SPLITTER, sdir)
                self.tile_cache[final_pos] = (
                    Environment.EMPTY, -1, EntityType.SPLITTER, self.my_team_cache,
                )
                print(f"[{self.corner}] AX 1A splitter at ({final_pos[0]},{final_pos[1]}) "
                      f"facing {sdir.value}")
                self._queue_splitter_bridge_fixes(final_pos, sdir)
                self._scenario_step = 3
            return False

        # Step 3: fix blocked conveyors adjacent to the splitter.
        if step == 3:
            if self._process_splitter_bridge_fixes(ct):
                self._scenario_step = 4
            return False

        # Step 4: build foundry at ft (splitter already placed).
        if step == 4:
            ft = self.foundry_tile
            if euclidean_dist_sq(self._my_xy, ft) > GC.ACTION_RADIUS_SQ:
                # Walk onto the splitter (final_pos) to get within range.
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(self._my_xy, ft)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                else:
                    d = direction_between(self._my_xy, ft)
                    if d is not None and ct.can_move(d):
                        ct.move(d)
                return False
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.FOUNDRY_BASE_COST[0]):
                return False
            f_pos = xy_to_pos(ft)
            fe = self.tile_cache.get(ft)
            if fe and fe[1] is not None and fe[3] == self.my_team_cache:
                if ct.can_destroy(f_pos):
                    ct.destroy(f_pos)
                    self.tile_cache[ft] = (fe[0], None, None, None)
                    self.building_cache.pop(ft, None)
            if ct.can_build_foundry(f_pos):
                ct.build_foundry(f_pos)
                self.building_cache[ft] = (EntityType.FOUNDRY, None)
                self.tile_cache[ft] = (
                    fe[0] if fe else Environment.EMPTY,
                    -1, EntityType.FOUNDRY, self.my_team_cache,
                )
                print(f"[{self.corner}] AX 1A foundry at ({ft[0]},{ft[1]})")
                self._scan_foundry_redirects(ft, final_pos)
                if self._foundry_redirect_queue:
                    self._scenario_step = 45
                else:
                    self._scenario_step = 5
            return False

        if step == 45:  # redirect conveyors/bridges targeting foundry → splitter
            if self._process_foundry_redirects(ct):
                self._scenario_step = 5
            return False

        # Step 5: bridge 1 — from a splitter side (not back) to core.
        # Skip if the splitter is already cardinally adjacent to the
        # core (its output reaches the core directly).
        if step == 5:
            if self._is_core_adjacent(final_pos):
                print(f"[{self.corner}] AX 1A splitter ({final_pos[0]},{final_pos[1]}) "
                      f"core-adjacent, skip bridge 1")
                self._scenario_step = 6
                return False
            if self._place_scenario_bridge(ct, final_pos,
                                           exclude=self._opposite_xy(final_pos, self.foundry_tile)):
                self._scenario_step = 6
            return False

        # Step 6: bridge 2 — from a foundry side to core. Skip if the
        # foundry is already cardinally adjacent to the core.
        if step == 6:
            if self._is_core_adjacent(self.foundry_tile):
                print(f"[{self.corner}] AX 1A foundry ({self.foundry_tile[0]},"
                      f"{self.foundry_tile[1]}) core-adjacent, skip bridge 2")
                return True
            if self._place_scenario_bridge(ct, self.foundry_tile, exclude=None):
                return True  # Scenario 1A complete
            return False

        # Shouldn't happen
        return True

    # ---- Scenario 1B: conveyor → conveyor → core ---- #

    def _do_scenario_1b_splitters(self, ct):
        """Compute the splitter queue from conveyors feeding into final_pos.
        Splitters are built BEFORE the foundry so we never have a foundry
        without a splitter. Returns True when the queue is ready."""
        final_pos = self.follow_chain_final_pos
        if final_pos is None:
            self._axionite_abandon("scenario_1b no final_pos")
            return True
        # Enqueue adjacent conveyors feeding INTO the foundry tile.
        self.splitter_queue = []
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (final_pos[0] + ddx, final_pos[1] + ddy)
            cached = self.building_cache.get(nxy)
            if cached is None:
                continue
            if cached[0] not in (EntityType.CONVEYOR,
                                 EntityType.ARMOURED_CONVEYOR):
                continue
            cdir = cached[1]
            if cdir is None:
                continue
            cdx, cdy = DIRECTION_DELTAS[cdir]
            if (nxy[0] + cdx, nxy[1] + cdy) == final_pos:
                self.splitter_queue.append(nxy)
        if not self.splitter_queue:
            self._axionite_abandon("1B no conveyors feeding final_pos")
            return False
        print(f"[{self.corner}] AX 1B splitter queue: {self.splitter_queue}")
        self._scenario_step = 0
        return True

    def _do_scenario_1b_place_foundry(self, ct):
        """Build the foundry at final_pos AFTER all splitters are placed.
        Returns True when foundry is built."""
        final_pos = self.follow_chain_final_pos
        if final_pos is None:
            self._axionite_abandon("scenario_1b_foundry no final_pos")
            return True
        step = self._scenario_step

        if step == 0:
            if euclidean_dist_sq(self._my_xy, final_pos) > GC.ACTION_RADIUS_SQ:
                # Try to walk onto an adjacent walkable tile (e.g. the
                # splitter tile) to get within action range of final_pos.
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(self._my_xy, final_pos)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                else:
                    # Direct move toward final_pos as fallback.
                    d = cardinal_direction_between(self._my_xy, final_pos)
                    if d is not None and ct.can_move(d):
                        ct.move(d)
                return False
            if self._my_xy == final_pos:
                self._step_off_tile(ct, final_pos)
                return False
            self._scenario_step = 1
            return False

        if step == 1:
            if ct.get_action_cooldown() > 0:
                return False
            if not self._can_spend(ct, GC.FOUNDRY_BASE_COST[0]):
                return False
            fp = xy_to_pos(final_pos)
            f_entry = self.tile_cache.get(final_pos)
            if f_entry and f_entry[1] is not None and f_entry[3] == self.my_team_cache:
                if ct.can_destroy(fp):
                    ct.destroy(fp)
                    self.tile_cache[final_pos] = (f_entry[0], None, None, None)
                    self.building_cache.pop(final_pos, None)
            if ct.can_build_foundry(fp):
                ct.build_foundry(fp)
                self.building_cache[final_pos] = (EntityType.FOUNDRY, None)
                self.tile_cache[final_pos] = (
                    f_entry[0] if f_entry else Environment.EMPTY,
                    -1, EntityType.FOUNDRY, self.my_team_cache,
                )
                print(f"[{self.corner}] AX 1B foundry at ({final_pos[0]},{final_pos[1]})")
                return True
            return False

        return False

    def _do_scenario_1b_queue(self, ct):
        """Process splitter_queue one entry at a time. For each entry:
        pathfind adjacent → destroy conveyor → build splitter (same dir)
        → if no existing bridge/conveyor facing away from splitter, build
        an output bridge toward core."""
        if not self.splitter_queue:
            return True
        target_xy = self.splitter_queue[0]
        step = self._scenario_step

        if step == 0:  # move adjacent
            if euclidean_dist_sq(self._my_xy, target_xy) > GC.ACTION_RADIUS_SQ:
                if self.path is None or self.path_index >= len(self.path):
                    self.path = self._compute_path(self._my_xy, target_xy)
                    self.path_index = 0
                if self.path:
                    self._follow_path(ct)
                else:
                    d = direction_between(self._my_xy, target_xy)
                    if d is not None and ct.can_move(d):
                        ct.move(d)
                    else:
                        d2 = cardinal_direction_between(self._my_xy, target_xy)
                        if d2 is not None and ct.can_move(d2):
                            ct.move(d2)
                return False
            if self._my_xy == target_xy:
                self._step_off_tile(ct, target_xy)
                return False
            self._scenario_step = 1
            return False

        if step == 1:  # destroy + build splitter
            if ct.get_action_cooldown() > 0:
                return False
            # If we already captured the splitter_dir on a previous
            # attempt (destroyed but couldn't build), reuse it.
            splitter_dir = self._retrofit_x_dir
            if splitter_dir is None:
                cached = self.building_cache.get(target_xy)
                if cached is None or cached[1] is None:
                    # Lost it — drop this entry.
                    self.splitter_queue.pop(0)
                    self._scenario_step = 0
                    return False
                b_dir = cached[1]  # B's original direction (toward foundry)
                # Determine the correct splitter direction:
                # - If upstream A is a CONVEYOR, splitter must face A's
                #   direction so the splitter's back receives A's output.
                # - If upstream A is a BRIDGE, keep B's direction
                #   (bridges deliver to target regardless of facing).
                splitter_dir = b_dir
                foundry_xy = self.follow_chain_final_pos
                for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nxy = (target_xy[0] + ddx, target_xy[1] + ddy)
                    if nxy == foundry_xy:
                        continue
                    n_cached = self.building_cache.get(nxy)
                    if n_cached is None:
                        continue
                    if n_cached[0] in (EntityType.CONVEYOR,
                                       EntityType.ARMOURED_CONVEYOR):
                        ndir = n_cached[1]
                        if ndir is None:
                            continue
                        ndx, ndy = DIRECTION_DELTAS[ndir]
                        if (nxy[0] + ndx, nxy[1] + ndy) == target_xy:
                            splitter_dir = ndir
                            break
                # Ensure the splitter's back tile isn't the core or the
                # foundry (only matters when target_xy is core-adjacent).
                # Foundry must stay a front/side output of the splitter.
                validated = self._pick_splitter_dir(
                    target_xy, splitter_dir, foundry_xy, foundry_xy)
                if validated is None:
                    # No orientation keeps the foundry fed and a valid
                    # back tile — drop this queue entry.
                    print(f"[{self.corner}] AX 1B splitter at ({target_xy[0]},"
                          f"{target_xy[1]}) no valid dir, dropping")
                    self.splitter_queue.pop(0)
                    self._scenario_step = 0
                    self._retrofit_x_dir = None
                    return False
                splitter_dir = validated
                self._retrofit_x_dir = splitter_dir  # survive turn boundary
            if not self._can_spend(ct, GC.SPLITTER_BASE_COST[0]):
                return False
            tpos = xy_to_pos(target_xy)
            t_entry = self.tile_cache.get(target_xy)
            if t_entry and t_entry[1] is not None:
                if ct.can_destroy(tpos):
                    ct.destroy(tpos)
                    self.tile_cache[target_xy] = (t_entry[0], None, None, None)
                    self.building_cache.pop(target_xy, None)
            if ct.can_build_splitter(tpos, splitter_dir):
                ct.build_splitter(tpos, splitter_dir)
                self.building_cache[target_xy] = (EntityType.SPLITTER, splitter_dir)
                self.tile_cache[target_xy] = (
                    Environment.EMPTY, -1, EntityType.SPLITTER, self.my_team_cache,
                )
                print(f"[{self.corner}] AX 1B splitter at ({target_xy[0]},{target_xy[1]}) "
                      f"facing {splitter_dir.value}")
                self._retrofit_x_dir = None
                if self._1b_first_splitter is None:
                    self._1b_first_splitter = target_xy
                self._queue_splitter_bridge_fixes(target_xy, splitter_dir)
                self._scenario_step = 2
            return False

        if step == 2:  # fix blocked conveyors + ensure output exists
            if not self._process_splitter_bridge_fixes(ct):
                return False
            self.splitter_queue.pop(0)
            self._scenario_step = 0
            return False

        return False

    # ---- Scenario 1 helper methods ---- #

    def _pick_scenario_1a_foundry_tile(self, final_pos):
        """Cardinal neighbour of final_pos that is NOT a core tile.
        Preference order:
          1. Empty and core-adjacent (refined output reaches core directly)
          2. Any empty
          3. Allied road / conveyor (will be destroyed — last resort)
        Returns None if nothing valid."""
        best_empty_core_adj = None
        best_empty = None
        last_conv = None
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (final_pos[0] + ddx, final_pos[1] + ddy)
            if self._is_core_tile(nxy):
                continue
            ne = self.tile_cache.get(nxy)
            if ne is None:
                continue
            env, bid, etype, team = ne
            if env == Environment.WALL:
                continue
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                continue
            if bid is None:
                if self._is_core_adjacent(nxy):
                    if best_empty_core_adj is None:
                        best_empty_core_adj = nxy
                elif best_empty is None:
                    best_empty = nxy
            elif (team == self.my_team_cache
                  and etype in (EntityType.ROAD, EntityType.CONVEYOR,
                                EntityType.ARMOURED_CONVEYOR)):
                if last_conv is None:
                    last_conv = nxy
        return best_empty_core_adj or best_empty or last_conv

    def _direction_out(self, frm, to):
        """Cardinal Direction from frm to to (must be cardinal adjacent)."""
        dx = to[0] - frm[0]
        dy = to[1] - frm[1]
        if dx == 1 and dy == 0:
            return Direction.EAST
        if dx == -1 and dy == 0:
            return Direction.WEST
        if dx == 0 and dy == 1:
            return Direction.SOUTH
        if dx == 0 and dy == -1:
            return Direction.NORTH
        return None

    def _opposite_xy(self, centre_xy, facing_xy):
        """Return the tile directly opposite `facing_xy` across `centre_xy`."""
        dx = facing_xy[0] - centre_xy[0]
        dy = facing_xy[1] - centre_xy[1]
        return (centre_xy[0] - dx, centre_xy[1] - dy)

    def _step_off_tile(self, ct, tile_xy):
        """If the bot is standing on tile_xy, move to an adjacent walkable tile."""
        my_xy = self._my_xy
        if my_xy != tile_xy:
            return
        for d in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST,
                  Direction.NORTHEAST, Direction.SOUTHEAST,
                  Direction.SOUTHWEST, Direction.NORTHWEST):
            if ct.can_move(d):
                ct.move(d)
                return

    def _place_scenario_bridge(self, ct, source_xy, exclude=None):
        """Build an optimal-bridge targeting the core from a tile cardinally
        adjacent to `source_xy` (excluding the `exclude` tile if given).
        Prefers empty sides; falls back to destroying allied roads if
        no empty side exists. Moves the bot to within action range
        before building. Returns True when the bridge is built (or
        failure is unrecoverable)."""
        empty_cands = []
        road_cands = []
        for ddx, ddy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxy = (source_xy[0] + ddx, source_xy[1] + ddy)
            if exclude is not None and nxy == exclude:
                continue
            if self._is_core_tile(nxy):
                continue
            ne = self.tile_cache.get(nxy)
            if ne is None:
                continue
            env, bid, etype, team = ne
            if env == Environment.WALL:
                continue
            if bid is None:
                empty_cands.append(nxy)
            elif team == self.my_team_cache and etype == EntityType.ROAD:
                road_cands.append(nxy)
        cands = empty_cands or road_cands
        if not cands:
            print(f"[{self.corner}] AX scenario bridge no side of ({source_xy[0]},{source_xy[1]})")
            return True  # best-effort — give up on this bridge
        bridge_at = cands[0]

        # Move within action range of bridge_at.
        if euclidean_dist_sq(self._my_xy, bridge_at) > GC.ACTION_RADIUS_SQ:
            self.path = self._compute_path(self._my_xy, bridge_at)
            self.path_index = 0
            if self.path:
                self._follow_path(ct)
            return False

        if ct.get_action_cooldown() > 0:
            return False
        # If the chosen side has an allied road, destroy it and then
        # build the bridge in the same turn (ct.destroy does not consume
        # the action cooldown).
        bt_entry = self.tile_cache.get(bridge_at)
        if (bt_entry and bt_entry[1] is not None
                and bt_entry[2] == EntityType.ROAD
                and bt_entry[3] == self.my_team_cache):
            bpos = xy_to_pos(bridge_at)
            if ct.can_destroy(bpos):
                ct.destroy(bpos)
                self.tile_cache[bridge_at] = (bt_entry[0], None, None, None)
                self.building_cache.pop(bridge_at, None)
                print(f"[{self.corner}] AX scenario bridge destroyed road at "
                      f"({bridge_at[0]},{bridge_at[1]})")
        # Try to target the core (or nearest allied transport to core)
        # using the dedicated axionite bridge builder.
        if self._build_axionite_bridge(ct, bridge_at):
            return True
        return False

    def _build_axionite_bridge(self, ct, bridge_at):
        """Build a bridge at bridge_at targeting (in priority order):
          1. A core tile directly (if the engine accepts it)
          2. The nearest allied transport (conveyor/splitter/foundry/
             bridge) to the core within bridge range
          3. The nearest empty tile closer to the core
        Used by scenario 1A/1B side bridges that must deliver refined
        axionite into the core. Moves the bot into action range of
        bridge_at if not already there.
        Returns True on successful placement, False otherwise (including
        the case where we moved this turn and should retry next turn)."""
        if self.core_pos is None:
            return False
        # Move into action range of bridge_at first.
        if euclidean_dist_sq(self._my_xy, bridge_at) > GC.ACTION_RADIUS_SQ:
            self.path = self._compute_path(self._my_xy, bridge_at)
            self.path_index = 0
            if self.path:
                self._follow_path(ct)
            return False
        if ct.get_action_cooldown() > 0:
            return False
        r = 3  # sqrt(BRIDGE_TARGET_RADIUS_SQ = 9)
        bx, by = bridge_at
        bridge_pos = xy_to_pos(bridge_at)
        core_cands = []
        transport_cands = []  # (dist_to_core, xy)
        empty_cands = []      # (dist_to_core, xy)
        bl = self._bridge_target_blacklist
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                dsq = dx * dx + dy * dy
                if dsq == 0 or dsq > GC.BRIDGE_TARGET_RADIUS_SQ:
                    continue
                xy = (bx + dx, by + dy)
                if xy in self.known_walls or xy in bl:
                    continue
                if self._is_enemy_core_tile(xy):
                    continue
                # Pending foundry tile — treat as already-a-foundry; never
                # target it with a bridge (unless the chain is bootstrap-
                # routing directly at foundry_tile as its goal).
                if (self.foundry_tile is not None
                        and xy == self.foundry_tile
                        and self._chain_goal != self.foundry_tile):
                    continue
                if self._is_core_tile(xy):
                    core_cands.append(xy)
                    continue
                ne = self.tile_cache.get(xy)
                if ne is None:
                    continue
                env, bid, etype, team = ne
                if env == Environment.WALL:
                    continue
                dist_core = euclidean_dist_sq(xy, self.core_pos)
                if bid is None:
                    if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                        continue
                    empty_cands.append((dist_core, xy))
                elif team == self.my_team_cache and etype in _ALLIED_TRANSPORT:
                    if xy in self._chain_built:
                        continue
                    transport_cands.append((dist_core, xy))
        # Build priority list: core tiles first, then closest allied
        # transport, then closest empty tile.
        candidates = list(core_cands)
        if transport_cands:
            transport_cands.sort()
            candidates.append(transport_cands[0][1])
        if empty_cands:
            empty_cands.sort()
            candidates.append(empty_cands[0][1])
        if not candidates:
            return False
        for cand in candidates:
            cand_pos = xy_to_pos(cand)
            if not ct.can_build_bridge(bridge_pos, cand_pos):
                continue
            if not self._can_spend(ct, GC.BRIDGE_BASE_COST[0]):
                return False
            ct.build_bridge(bridge_pos, cand_pos)
            self.building_cache[bridge_at] = (EntityType.BRIDGE, cand)
            self.bridge_target_cache[bridge_at] = cand
            self.tile_cache[bridge_at] = (
                Environment.EMPTY, -1, EntityType.BRIDGE, self.my_team_cache,
            )
            print(f"[{self.corner}] AX axionite bridge ({bridge_at[0]},"
                  f"{bridge_at[1]})->({cand[0]},{cand[1]})")
            return True
        return False
