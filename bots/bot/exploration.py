import sys
from cambc import Direction, EntityType, Environment, Position, GameConstants as GC
from constants import (
    INITIAL_EXPLORE_RADIUS, EXPLORE_RADIUS_EXPAND_INTERVAL,
    MAX_EXPLORE_RADIUS, FRONTIER_SCAN_RADIUS, FRONTIER_BIAS_STRENGTH,
    ORE_PATHFIND_GIVE_UP_DIST, DEBUG,
)
from utils import (
    euclidean_dist_sq, direction_between,
    DIRECTION_DELTAS, xy_to_pos,
)

# Markers excluded — bots can't walk onto markers; to repair a tile
# that now holds an enemy marker, we build directly over it.
_ENEMY_WALKABLE_REPAIR = frozenset({
    EntityType.ROAD, EntityType.CONVEYOR, EntityType.SPLITTER,
    EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
})

_CACHE_TYPES_REPAIR = frozenset({
    EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR, EntityType.HARVESTER, EntityType.FOUNDRY,
    EntityType.BARRIER,
})
from pathfinding import astar_cached, _WALKABLE_BUILDINGS  # only for _find_frontier_target fallback

# Corner → preferred direction for leaving core
_CORNER_LEAVE_DIR = {
    "NW": Direction.NORTHWEST,
    "NE": Direction.NORTHEAST,
    "SW": Direction.SOUTHWEST,
    "SE": Direction.SOUTHEAST,
}

# Corner → preferred exploration bias direction (unit vector-ish)
_CORNER_BIAS = {
    "NW": (0, -1),   # north
    "NE": (1, 0),    # east
    "SE": (0, 1),    # south
    "SW": (-1, 0),   # west
}

# Fallback directions for leaving core (try preferred, then rotate out)
_LEAVE_FALLBACKS = {
    "NW": [Direction.NORTHWEST, Direction.NORTH, Direction.WEST,
            Direction.NORTHEAST, Direction.SOUTHWEST],
    "NE": [Direction.NORTHEAST, Direction.NORTH, Direction.EAST,
            Direction.NORTHWEST, Direction.SOUTHEAST],
    "SW": [Direction.SOUTHWEST, Direction.SOUTH, Direction.WEST,
            Direction.NORTHWEST, Direction.SOUTHEAST],
    "SE": [Direction.SOUTHEAST, Direction.SOUTH, Direction.EAST,
            Direction.SOUTHWEST, Direction.NORTHEAST],
}


class ExplorationMixin:
    """Frontier exploration, leaving core, axionite/titanium detection."""

    def _step_leave_core(self, ct):
        """Move off the core immediately."""
        my_xy = self._my_xy

        # Check if already off core
        entry = self.tile_cache.get(my_xy)
        on_core = entry is not None and entry[2] == EntityType.CORE
        if not on_core:
            self.step = "explore"
            return

        # --- Move off core ---
        corner = self.corner if self.corner in _LEAVE_FALLBACKS else "NW"
        fallbacks = _LEAVE_FALLBACKS[corner]

        # Try preferred directions first, then all 8 as fallback
        all_dirs = [Direction.NORTH, Direction.NORTHEAST, Direction.EAST,
                    Direction.SOUTHEAST, Direction.SOUTH, Direction.SOUTHWEST,
                    Direction.WEST, Direction.NORTHWEST]
        move_order = list(fallbacks)
        for d in all_dirs:
            if d not in move_order:
                move_order.append(d)

        for d in move_order:
            if ct.can_move(d):
                ct.move(d)
                self.step = "explore"
                return

        if ct.get_action_cooldown() == 0:
            any_open = False
            for d in move_order:
                dx, dy = DIRECTION_DELTAS[d]
                target_xy = (my_xy[0] + dx, my_xy[1] + dy)
                if target_xy in self.known_walls:
                    continue
                t_entry = self.tile_cache.get(target_xy)
                if t_entry and t_entry[0] == Environment.WALL:
                    continue
                any_open = True
                target_pos = xy_to_pos(target_xy)
                if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                        and ct.can_build_road(target_pos)):
                    ct.build_road(target_pos)
                    if ct.can_move(d):
                        ct.move(d)
                        self.step = "explore"
                    return
            if not any_open:
                # All 8 directions are walls (corner maps) — skip to
                # explore, which can pathfind through other core tiles.
                self.step = "explore"
                return

    def _step_explore(self, ct):
        """Frontier exploration with ore detection and directional bias."""
        my_xy = self._my_xy

        # Heal a damaged ally building if a path of length <
        # _HEAL_VISION_PATH_BUDGET reaches it. Returns True after
        # diverting (move/heal); we yield this turn.
        if self._economy_try_heal_in_vision(ct, my_xy):
            return
        # Otherwise, opportunistic self-heal (only when the next path
        # tile doesn't need a road build — heal uses the action cd).
        self._economy_self_heal_if_safe(ct, my_xy)

        # Initialize / expand explore radius
        if self.explore_radius is None:
            self.explore_radius = INITIAL_EXPLORE_RADIUS
            self._next_radius_expand = EXPLORE_RADIUS_EXPAND_INTERVAL
        self._next_radius_expand -= 1
        if self._next_radius_expand <= 0 and self.explore_radius < MAX_EXPLORE_RADIUS:
            self.explore_radius += 1
            self._next_radius_expand = EXPLORE_RADIUS_EXPAND_INTERVAL

        # --- FAST PATH: if we have a valid path, just follow it ---
        # Scan vision for unclaimed ore or stealable harvester — redirect if found
        if self.path and self.path_index < len(self.path) and self.frontier_target:
            target = self._scan_titanium_target(my_xy) if self.tle_recovery_turns == 0 else None
            if target:
                best_xy, kind = target
                self.frontier_target = None
                self.path = None
                self.path_index = 0
                if kind == 'unclaimed':
                    self.target_ore = best_xy
                    self.step = "goto_ore"
                    if DEBUG:
                        print(f"[{self.corner}] ore at ({best_xy[0]},{best_xy[1]})",
                              file=sys.stderr)
                else:
                    self.steal_target = best_xy
                    self.step = "steal_harvester"
                    if DEBUG:
                        print(f"[{self.corner}] steal harvester at ({best_xy[0]},{best_xy[1]})",
                              file=sys.stderr)
                return
            result = self._follow_path(ct)
            if result == 'done':
                self.frontier_target = None
                self.path = None
                self.path_index = 0
            elif result == 'blocked':
                print(f"[{self.corner}] explore path blocked at {my_xy} "
                      f"target={self.frontier_target} idx={self.path_index}/{len(self.path) if self.path else 0}",
                      file=sys.stderr)
                if self.tle_recovery_turns == 0:
                    self.path = self._compute_path(my_xy, self.frontier_target)
                    self.path_index = 0
                    if self.path is None:
                        self._frontier_blacklist.add(self.frontier_target)
                        self.frontier_target = None
            elif result is None:
                print(f"[{self.corner}] explore _follow_path returned None "
                      f"at {my_xy} target={self.frontier_target} "
                      f"idx={self.path_index}/{len(self.path) if self.path else 0}",
                      file=sys.stderr)
            return

        # --- SLOW PATH: need a new target — full ore scan + frontier search ---
        # During TLE recovery, skip the full scan — just follow existing path
        if self.tle_recovery_turns > 0:
            return

        # Step 3: remember at most one axionite position during exploration.
        # Used by Step 4 as a free candidate after the first titanium chain
        # completes. We never pathfind to it here — just record it.
        tc = self.tile_cache
        if self.stored_axionite_pos is None:
            for xy in self._last_vision_set:
                entry = tc.get(xy)
                if entry and entry[0] == Environment.ORE_AXIONITE:
                    self.stored_axionite_pos = xy
                    break

        target = self._scan_titanium_target(my_xy)
        if target is not None:
            best_xy, kind = target
            self.path = None
            self.path_index = 0
            if kind == 'unclaimed':
                self.target_ore = best_xy
                self.step = "goto_ore"
                if DEBUG:
                    print(f"[{self.corner}] ore at ({best_xy[0]},{best_xy[1]})",
                          file=sys.stderr)
            elif kind == 'unclaimed_axionite':
                self.target_ore = best_xy
                self.step = "goto_ore"
                if DEBUG:
                    print(f"[{self.corner}] ore at ({best_xy[0]},{best_xy[1]}) [axionite]",
                          file=sys.stderr)
            else:
                self.steal_target = best_xy
                self.step = "steal_harvester"
                if DEBUG:
                    print(f"[{self.corner}] steal harvester at ({best_xy[0]},{best_xy[1]})",
                          file=sys.stderr)
            return

        # --- Frontier search ---
        if self.frontier_target is not None:
            # Fulfilled when reached, or when the tile is no longer a
            # frontier (all its previously-unknown neighbors are now known).
            if (my_xy == self.frontier_target
                    or self.frontier_target not in self.frontier_cache):
                self.frontier_target = None
                self.path = None
                self.path_index = 0
                self._frontier_blacklist = set()

        if self.frontier_target is None and self.tle_recovery_turns == 0:
            _blacklist = self._frontier_blacklist
            local_ft = None
            # Fast path: scan only local vision (cheap)
            for _ in range(3):
                ft = self._find_frontier_target(ct, blacklist=_blacklist)
                if ft is None:
                    break
                local_ft = ft
                path = self._compute_path(my_xy, ft)
                if path is not None:
                    self.frontier_target = ft
                    self.path = path
                    self.path_index = 0
                    break
                _blacklist.add(ft)
            # Slow path: no local frontier — use the global perimeter cache
            global_ft = None
            if self.frontier_target is None:
                for _ in range(5):
                    ft = self._find_global_frontier(blacklist=_blacklist)
                    if ft is None:
                        break
                    global_ft = ft
                    path = self._compute_path(my_xy, ft)
                    if path is not None:
                        self.frontier_target = ft
                        self.path = path
                        self.path_index = 0
                        break
                    _blacklist.add(ft)
            self._frontier_blacklist = _blacklist

            if self.frontier_target is None:
                print(f"[{self.corner}] explore: no reachable frontier "
                      f"my={my_xy} local_ft={local_ft} global_ft={global_ft} "
                      f"fc_size={len(self.frontier_cache)} bl={len(_blacklist)}",
                      file=sys.stderr)

        if self.frontier_target is None and self.tle_recovery_turns == 0:
            # Clear blacklist so next turn gets fresh attempts.
            self._frontier_blacklist = set()
            self._fallback_to_core(my_xy)
            if self.frontier_target is None:
                return

        if self.path is not None:
            result = self._follow_path(ct)
            if result == 'done':
                self.frontier_target = None
                self.path = None
                self.path_index = 0
            elif result == 'blocked':
                if self.tle_recovery_turns == 0:
                    self.path = self._compute_path(my_xy, self.frontier_target)
                    self.path_index = 0
                    if self.path is None:
                        self._frontier_blacklist.add(self.frontier_target)
                        self.frontier_target = None

    def _scan_titanium_target(self, my_xy):
        """Scan current vision for the closest ore target.

        Candidates: unclaimed titanium ore, enemy harvester on titanium
        (stealable), and — if eligible — unclaimed axionite ore. All
        candidate types compete for the single closest-Euclidean slot.
        Axionite is eligible only after the bot's first titanium chain
        is complete AND it hasn't already built an axionite chain.
        Returns (xy, kind) or None; kind is 'unclaimed' | 'steal' |
        'unclaimed_axionite'.
        """
        best = None
        best_kind = None
        best_dsq = ORE_PATHFIND_GIVE_UP_DIST * ORE_PATHFIND_GIVE_UP_DIST
        ore_bl = self._ore_blacklist
        my_team = self.my_team_cache
        tc = self.tile_cache
        bpc = self.bot_pos_cache
        # Stealing requires affording a sentinel right now
        ct = getattr(self, '_ct', None)
        can_steal = ct is not None and self._can_afford_sentinel(ct)
        # Axionite eligibility: first titanium chain must be complete,
        # and we must not have already built an axionite chain. Also
        # skip the axionite logic entirely during TLE recovery.
        axionite_eligible = (self.titanium_chains_completed >= 1
                             and not self.axionite_chain_done
                             and self.tle_recovery_turns == 0)

        def _candidate_unclaimed_axionite(xy, entry):
            """Shared axionite candidate filter: not blacklisted, not
            enclosed, no building (allied barriers OK — economy bots
            bust them through), no camping enemy, not claimed by a
            standing allied bot."""
            if xy in ore_bl:
                return False
            if self._is_ore_enclosed(xy):
                return False
            bid, etype, team = entry[1], entry[2], entry[3]
            if bid is not None:
                if not (etype == EntityType.BARRIER and team == my_team):
                    return False
            if xy in bpc and bpc[xy][1] == my_team and xy != my_xy:
                return False
            if self._enemy_camping_ore(xy):
                return False
            return True

        for xy in self._last_vision_set:
            entry = tc.get(xy)
            if entry is None:
                continue
            env = entry[0]

            # --- Axionite branch ---
            if env == Environment.ORE_AXIONITE:
                if not axionite_eligible:
                    continue
                dx = xy[0] - my_xy[0]
                dy = xy[1] - my_xy[1]
                dsq = dx * dx + dy * dy
                if dsq >= best_dsq:
                    continue
                if not _candidate_unclaimed_axionite(xy, entry):
                    continue
                best_dsq = dsq
                best = xy
                best_kind = 'unclaimed_axionite'
                continue

            if env != Environment.ORE_TITANIUM:
                continue
            if xy in ore_bl:
                continue
            if self._is_ore_enclosed(xy):
                continue
            bid, etype, team = entry[1], entry[2], entry[3]

            dx = xy[0] - my_xy[0]
            dy = xy[1] - my_xy[1]
            dsq = dx * dx + dy * dy
            if dsq >= best_dsq:
                continue

            # Enemy harvester on ore — stealable candidate
            if bid is not None and team != my_team and etype == EntityType.HARVESTER:
                if not can_steal:
                    continue
                if self._is_already_stolen(xy):
                    continue
                if not self._has_available_steal_tile(xy):
                    continue
                best_dsq = dsq
                best = xy
                best_kind = 'steal'
                continue

            # Any other building on ore — not an unclaimed candidate.
            # Exception: allied barrier on ore is still claimable —
            # _barrier_bust_step destroys it, builds a road, and the
            # normal claim flow takes over.
            if bid is not None:
                if not (etype == EntityType.BARRIER and team == my_team):
                    continue
            # Allied bot already standing on the ore = claimed
            if xy in bpc and bpc[xy][1] == my_team and xy != my_xy:
                continue
            if self._enemy_camping_ore(xy):
                continue

            best_dsq = dsq
            best = xy
            best_kind = 'unclaimed'

        # Step 4 extension: the stored axionite position from Step 3 is
        # always a valid candidate when eligible, even outside current
        # vision (we remembered it during explore; just pathfind to it).
        # If it turns out to be claimed on arrival, _step_claim_ore will
        # route us back to explore.
        if axionite_eligible and self.stored_axionite_pos is not None:
            sxy = self.stored_axionite_pos
            if sxy not in ore_bl:
                dx = sxy[0] - my_xy[0]
                dy = sxy[1] - my_xy[1]
                dsq = dx * dx + dy * dy
                if dsq < best_dsq:
                    # Validate against tile_cache if we can see it; if
                    # unseen, trust the stored position.
                    e = tc.get(sxy)
                    valid = True
                    if e is not None:
                        if e[0] != Environment.ORE_AXIONITE:
                            valid = False
                        elif not _candidate_unclaimed_axionite(sxy, e):
                            valid = False
                    if valid:
                        best_dsq = dsq
                        best = sxy
                        best_kind = 'unclaimed_axionite'

        if best is None:
            return None
        return (best, best_kind)

    def _quick_ore_check(self, my_xy):
        """Check only the 4 tiles ahead on the path for ore. ~4 dict lookups."""
        if not self.path:
            return None
        ore_bl = self._ore_blacklist
        my_team = self.my_team_cache
        tc = self.tile_cache
        end = min(self.path_index + 4, len(self.path))
        for i in range(self.path_index, end):
            xy = self.path[i]
            entry = tc.get(xy)
            if entry and entry[0] == Environment.ORE_TITANIUM:
                if xy not in ore_bl and not self._is_ore_enclosed(xy):
                    bid, etype, team = entry[1], entry[2], entry[3]
                    if not (bid is not None and team == my_team
                            and etype in (EntityType.HARVESTER, EntityType.ROAD)):
                        return xy
        return None

    def _find_frontier_target(self, ct, blacklist=None):
        """Find best frontier tile in local vision.

        A frontier tile is adjacent to at least one in-bounds unknown tile.
        Applies directional bias based on corner.
        """
        my_xy = self._my_xy

        if self.core_pos is None:
            return None

        explore_r_sq = self.explore_radius * self.explore_radius if self.explore_radius else INITIAL_EXPLORE_RADIUS ** 2
        bias = _CORNER_BIAS.get(self.corner, (0, 0))
        mw = self.map_w
        mh = self.map_h

        # Search local vision only — 4-neighbor frontier check, single dict lookup
        best = None
        best_score = 999999
        scan_r_sq = FRONTIER_SCAN_RADIUS * FRONTIER_SCAN_RADIUS

        mx, my = my_xy
        cx, cy = self.core_pos
        bx, by = bias
        tc = self.tile_cache

        for xy in self._last_vision_set:
            if blacklist and xy in blacklist:
                continue
            x, y = xy
            dx_m = x - mx
            dy_m = y - my
            d_bot = dx_m * dx_m + dy_m * dy_m
            if d_bot > scan_r_sq:
                continue
            dx_c = x - cx
            dy_c = y - cy
            if dx_c * dx_c + dy_c * dy_c > explore_r_sq:
                continue
            entry = tc.get(xy)
            if entry is None or entry[0] == Environment.WALL:
                continue
            # Skip edge-adjacent tiles — frontiers at the map edge
            # only lead to dead ends.
            if mw is not None and (x == 0 or y == 0 or x == mw - 1 or y == mh - 1):
                continue

            # 4-neighbor frontier check — single lookup each (~4 lookups)
            is_frontier = False
            for nx, ny in ((x, y-1), (x+1, y), (x, y+1), (x-1, y)):
                if mw is not None and (nx < 0 or ny < 0 or nx >= mw or ny >= mh):
                    continue
                if (nx, ny) not in tc:
                    is_frontier = True
                    break

            if not is_frontier:
                continue

            # Score: distance from bot, with bias for frontiers in the
            # corner's preferred direction relative to the CORE (not the bot —
            # otherwise the bias rotates as the bot moves around the core).
            score = d_bot
            if bx != 0 or by != 0:
                dot = dx_c * bx + dy_c * by
                if dot > 0:
                    score -= FRONTIER_BIAS_STRENGTH

            if score < best_score:
                best_score = score
                best = xy

        return best

    def _find_global_frontier(self, blacklist=None):
        """Find closest global frontier tile from self.frontier_cache.

        Unlike _find_frontier_target (which only scans local vision),
        this iterates the maintained perimeter of explored space.
        Applies corner directional bias scored relative to core.
        """
        fc = self.frontier_cache
        if not fc or self.core_pos is None:
            return None

        my_xy = self._my_xy
        mx, my_ = my_xy
        cx, cy = self.core_pos
        bx, by = _CORNER_BIAS.get(self.corner, (0, 0))

        mw = self.map_w
        mh = self.map_h
        best = None
        best_score = 999999
        for xy in fc:
            if blacklist and xy in blacklist:
                continue
            x, y = xy
            if mw is not None and (x == 0 or y == 0 or x == mw - 1 or y == mh - 1):
                continue
            dx_m = x - mx
            dy_m = y - my_
            score = dx_m * dx_m + dy_m * dy_m
            if bx != 0 or by != 0:
                dx_c = x - cx
                dy_c = y - cy
                dot = dx_c * bx + dy_c * by
                if dot > 0:
                    score -= FRONTIER_BIAS_STRENGTH
            if score < best_score:
                best_score = score
                best = xy
        return best

    def _fallback_to_core(self, my_xy):
        """No frontier found — expand explore radius and pick a distant
        reachable tile to keep the bot moving toward unexplored territory."""
        if self.core_pos is None:
            return

        # Aggressively expand explore radius when we can't find frontiers
        if self.explore_radius is not None and self.explore_radius < MAX_EXPLORE_RADIUS:
            self.explore_radius = MAX_EXPLORE_RADIUS

        cx, cy = self.core_pos
        tc = self.tile_cache
        near_core = euclidean_dist_sq(my_xy, self.core_pos) <= 16

        candidates = []
        if not near_core:
            for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3),
                           (-2, -2), (2, 2), (2, -2), (-2, 2)]:
                candidates.append((cx + dx, cy + dy))
        else:
            # Near core — pick the farthest visible tile to move away
            best_far = None
            best_far_d = 0
            for xy in self._last_vision_set:
                e = tc.get(xy)
                if e is None or e[0] == Environment.WALL:
                    continue
                d = euclidean_dist_sq(my_xy, xy)
                if d > best_far_d:
                    best_far_d = d
                    best_far = xy
            if best_far:
                candidates.append(best_far)

        for target in candidates:
            if target == my_xy or target in self.known_walls:
                continue
            entry = tc.get(target)
            if entry and entry[0] == Environment.WALL:
                continue
            path = self._compute_path(my_xy, target)
            if path:
                self.frontier_target = target
                self.path = path
                self.path_index = 0
                return

    # ------------------------------------------------------------------ #
    #  Step: repair — rebuild a destroyed allied building from cache      #
    # ------------------------------------------------------------------ #

    def _step_repair(self, ct):
        my_xy = self._my_xy
        rxy = self.repair_target
        if rxy is None:
            self._finish_repair()
            return

        cached = self.building_cache.get(rxy)
        if cached is None:
            self._finish_repair()
            return

        cached_etype, cached_detail = cached
        entry = self.tile_cache.get(rxy)

        # Tile not in vision — pathfind closer
        if entry is None or rxy not in self._last_vision_set:
            if self.path is None:
                self.path = self._compute_path(my_xy, rxy)
                self.path_index = 0
                if self.path is None:
                    self._finish_repair()
                    return
            self._follow_path(ct)
            return

        env, bid, cur_etype, cur_team = entry

        # Already something allied here — don't destroy it to rebuild.
        # If it's tracked transport, update cache. If it's a non-tracked
        # allied building (like a patrol defence sentinel) drop the cache
        # entry so Pass B doesn't re-queue it next turn.
        if cur_team == self.my_team_cache and bid is not None:
            if cur_etype in _CACHE_TYPES_REPAIR:
                if cur_etype in (EntityType.CONVEYOR, EntityType.SPLITTER,
                                 EntityType.ARMOURED_CONVEYOR):
                    # tile_cache may carry a stale bid if the building
                    # was destroyed and replaced between the scan that
                    # populated the entry and now. Treat the stale-id
                    # case as "no longer needs repair" — next scan will
                    # refresh the cache with the real bid (or clear it).
                    try:
                        direction = ct.get_direction(bid)
                    except Exception:
                        self._finish_repair()
                        return
                    self.building_cache[rxy] = (cur_etype, direction)
                elif cur_etype == EntityType.BRIDGE:
                    self.building_cache[rxy] = (cur_etype, self.bridge_target_cache.get(rxy))
                else:
                    self.building_cache[rxy] = (cur_etype, None)
            else:
                self.building_cache.pop(rxy, None)
            self._finish_repair()
            return

        # Enemy non-walkable — can't repair
        if (bid is not None and cur_team != self.my_team_cache
                and cur_etype not in _ENEMY_WALKABLE_REPAIR):
            self._finish_repair()
            return

        # Enemy walkable — walk onto + fire
        if (bid is not None and cur_team != self.my_team_cache
                and cur_etype in _ENEMY_WALKABLE_REPAIR):
            if my_xy == rxy:
                if ct.get_action_cooldown() == 0:
                    pos = xy_to_pos(rxy)
                    if ct.can_fire(pos):
                        ct.fire(pos)
                return
            d = direction_between(my_xy, rxy)
            if ct.can_move(d):
                ct.move(d)
            return

        # Allied road — destroy (free)
        if (bid is not None and cur_team == self.my_team_cache
                and cur_etype == EntityType.ROAD):
            pos = xy_to_pos(rxy)
            if ct.can_destroy(pos):
                ct.destroy(pos)
                self.tile_cache[rxy] = (env, None, None, None)
            # Fall through to build

        # Need to be in action radius
        dsq = euclidean_dist_sq(my_xy, rxy)
        if dsq > GC.ACTION_RADIUS_SQ:
            if self.path is None:
                # Find adjacent walkable tile
                goal = self._repair_find_adjacent(my_xy, rxy)
                if goal is None:
                    self._finish_repair()
                    return
                self.path = self._compute_path(my_xy, goal)
                self.path_index = 0
                if self.path is None:
                    self._finish_repair()
                    return
            self._follow_path(ct)
            return

        # Build the cached building
        if ct.get_action_cooldown() > 0:
            return
        pos = xy_to_pos(rxy)
        if cached_etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
            if cached_detail is not None and ct.can_build_conveyor(pos, cached_detail):
                if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                    return  # Wait for reserve
                ct.build_conveyor(pos, cached_detail)
                if DEBUG:
                    print(f"[{self.corner}] repaired conveyor at ({rxy[0]},{rxy[1]})",
                          file=sys.stderr)
        elif cached_etype == EntityType.SPLITTER:
            if cached_detail is not None and ct.can_build_splitter(pos, cached_detail):
                if not self._has_economy_reserve(ct, GC.SPLITTER_BASE_COST[0]):
                    return  # Wait for reserve
                ct.build_splitter(pos, cached_detail)
                if DEBUG:
                    print(f"[{self.corner}] repaired splitter at ({rxy[0]},{rxy[1]})",
                          file=sys.stderr)
        elif cached_etype == EntityType.BRIDGE:
            if cached_detail is not None:
                target_pos = xy_to_pos(cached_detail)
                if ct.can_build_bridge(pos, target_pos):
                    if not self._has_economy_reserve(ct, GC.BRIDGE_BASE_COST[0]):
                        return  # Wait for reserve
                    ct.build_bridge(pos, target_pos)
                    if DEBUG:
                        print(f"[{self.corner}] repaired bridge at ({rxy[0]},{rxy[1]})",
                              file=sys.stderr)
        elif cached_etype == EntityType.HARVESTER:
            if ct.can_build_harvester(pos):
                if not self._has_economy_reserve(ct, GC.HARVESTER_BASE_COST[0]):
                    return  # Wait for reserve
                ct.build_harvester(pos)
                if DEBUG:
                    print(f"[{self.corner}] repaired harvester at ({rxy[0]},{rxy[1]})",
                          file=sys.stderr)
        elif cached_etype == EntityType.BARRIER:
            if not self._can_spend(ct, GC.BARRIER_BASE_COST[0]):
                return  # Hard titanium floor
            if ct.can_build_barrier(pos):
                ct.build_barrier(pos)
                if DEBUG:
                    print(f"[{self.corner}] repaired barrier at ({rxy[0]},{rxy[1]})",
                          file=sys.stderr)
        self._finish_repair()

    def _finish_repair(self):
        """Clean up repair state and resume previous activity."""
        self.repair_target = None
        self.path = None
        self.path_index = 0
        self.step = self._pre_repair_step or "explore"
        self.target_ore = self._pre_repair_target_ore
        self.frontier_target = self._pre_repair_frontier
        self._pre_repair_step = None
        self._pre_repair_target_ore = None
        self._pre_repair_frontier = None

    def _repair_find_adjacent(self, my_xy, target_xy):
        """Find closest walkable 8-neighbor of target_xy."""
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

    # ------------------------------------------------------------------ #
    #  Economy-bot heal helpers                                           #
    # ------------------------------------------------------------------ #
    #  Used by `_step_chain_to_core`, `_step_build_harvester`, and
    #  `_step_explore`. Roads are excluded — cheap to rebuild and their
    #  HP doesn't matter for combat.
    # ------------------------------------------------------------------ #

    _HEAL_VISION_PATH_BUDGET = 6
    # Tighter budget for in-task divert (chain / harvester / claim_ore).
    # 4 means "less than 5 tiles" per the user's spec — small enough
    # that detouring doesn't seriously delay the chain build.
    _HEAL_TASK_DIVERT_BUDGET = 4
    # Restrict in-task diverts to chain pieces only (the user's spec —
    # "damaged conveyor / bridge"). Splitter and armoured_conveyor are
    # included for parity, since they're the same chain.
    _HEAL_TASK_DIVERT_TYPES = (
        EntityType.CONVEYOR, EntityType.BRIDGE, EntityType.SPLITTER,
        EntityType.ARMOURED_CONVEYOR,
    )

    def _heal_target_pos(self, my_xy, building_xy, building_etype):
        """Pick the actual `target_pos` to pass to `ct.heal`. The 3x3
        core needs the closest footprint tile (otherwise can_heal fails
        when bot is in range of an edge tile but not the centre)."""
        if building_etype == EntityType.CORE:
            cx, cy = building_xy
            closest_x = max(cx - 1, min(cx + 1, my_xy[0]))
            closest_y = max(cy - 1, min(cy + 1, my_xy[1]))
            return xy_to_pos((closest_x, closest_y))
        return xy_to_pos(building_xy)

    def _economy_check_heal_action_radius(self, ct, my_xy):
        """Lock the bot in place to heal a damaged ally building inside
        action radius. Returns one of:
          'healed' — heal action fired this turn
          'wait'   — damaged building in range but action cd not ready;
                     caller MUST also yield (don't move) so the bot
                     stays adjacent for next turn
          'none'   — no damaged building in range, caller proceeds
        """
        my_team = self.my_team_cache
        tc = self.tile_cache
        mhc = self._max_hp_by_type
        best_bid = None
        best_xy = None
        best_etype = None
        best_hp = None
        for xy in self._last_vision_set:
            dx = xy[0] - my_xy[0]
            dy = xy[1] - my_xy[1]
            if dx * dx + dy * dy > GC.ACTION_RADIUS_SQ:
                continue
            info = tc.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team != my_team:
                continue
            if etype == EntityType.ROAD:
                continue
            max_hp = mhc.get(etype)
            if max_hp is None:
                try:
                    max_hp = ct.get_max_hp(bid)
                except Exception:
                    continue
                mhc[etype] = max_hp
            if not max_hp:
                continue
            try:
                hp = ct.get_hp(bid)
            except Exception:
                continue
            if hp >= max_hp:
                continue
            if best_hp is None or hp < best_hp:
                best_hp = hp
                best_bid = bid
                best_xy = xy
                best_etype = etype
        if best_bid is None:
            return 'none'
        if ct.get_action_cooldown() != 0:
            return 'wait'
        target_pos = self._heal_target_pos(my_xy, best_xy, best_etype)
        if ct.can_heal(target_pos):
            ct.heal(target_pos)
            if DEBUG:
                print(f"[{self.corner}] heal-in-place {best_xy} hp={best_hp}",
                      file=sys.stderr)
            return 'healed'
        return 'wait'

    def _economy_try_heal_in_vision(self, ct, my_xy, max_path=None,
                                    target_types=None):
        """While exploring (or mid-task): divert toward a damaged ally
        building if the path to an adjacent tile is ≤ `max_path`. With
        `target_types` (a tuple of EntityType) only those types are
        considered; default is all non-road allied buildings.

        Returns True if heal/move action fired this turn — caller
        should return without running its normal step logic.
        """
        my_team = self.my_team_cache
        tc = self.tile_cache
        mhc = self._max_hp_by_type
        budget = self._HEAL_VISION_PATH_BUDGET if max_path is None else max_path

        # Step 1: Chebyshev prefilter — Chebyshev is a lower bound on
        # path length, so anything past `budget` Chebyshev is also past
        # `budget` along an A* path. Cheap dict lookups, no FFI yet.
        candidates = []  # (cheb, bid, xy, etype)
        for xy in self._last_vision_set:
            cheb = max(abs(xy[0] - my_xy[0]), abs(xy[1] - my_xy[1]))
            if cheb > budget:
                continue
            info = tc.get(xy)
            if info is None:
                continue
            env, bid, etype, team = info
            if bid is None or team != my_team:
                continue
            if target_types is not None:
                if etype not in target_types:
                    continue
            elif etype == EntityType.ROAD:
                continue
            candidates.append((cheb, bid, xy, etype))
        if not candidates:
            return False
        candidates.sort(key=lambda c: c[0])

        # Step 2: HP query closest first; first damaged candidate wins.
        best_bid = None
        best_xy = None
        best_etype = None
        for _cheb, bid, xy, etype in candidates[:8]:
            max_hp = mhc.get(etype)
            if max_hp is None:
                try:
                    max_hp = ct.get_max_hp(bid)
                except Exception:
                    continue
                mhc[etype] = max_hp
            if not max_hp:
                continue
            try:
                hp = ct.get_hp(bid)
            except Exception:
                continue
            if hp >= max_hp:
                continue
            best_bid = bid
            best_xy = xy
            best_etype = etype
            break
        if best_bid is None:
            return False

        # Step 3: in action range → heal in place.
        if best_etype == EntityType.CORE:
            cx, cy = best_xy
            closest_x = max(cx - 1, min(cx + 1, my_xy[0]))
            closest_y = max(cy - 1, min(cy + 1, my_xy[1]))
            ddx = closest_x - my_xy[0]
            ddy = closest_y - my_xy[1]
            range_dsq = ddx * ddx + ddy * ddy
        else:
            range_dsq = (best_xy[0] - my_xy[0]) ** 2 + (best_xy[1] - my_xy[1]) ** 2
        target_pos = self._heal_target_pos(my_xy, best_xy, best_etype)
        if range_dsq <= GC.ACTION_RADIUS_SQ:
            if ct.get_action_cooldown() == 0 and ct.can_heal(target_pos):
                ct.heal(target_pos)
                if DEBUG:
                    print(f"[{self.corner}] heal-in-vision {best_xy}",
                          file=sys.stderr)
                return True
            # In range but cooldown not ready — hold position.
            return True

        # Step 4: walk closer. One A* on the winner only (mirrors the
        # disruptor pickers' single-pathfind pattern). Cap the node
        # budget tight — a heal diversion path goes through vision
        # (~70 tiles), so 100 nodes is more than we'd ever need but
        # stops runaway A* on an unreachable target (e.g. a damaged
        # conveyor behind a wall the rest of which is outside vision)
        # from burning the whole turn budget.
        goal = self._patrol_find_adjacent(my_xy, best_xy) \
            if hasattr(self, '_patrol_find_adjacent') \
            else self._heal_find_adjacent(my_xy, best_xy)
        if goal is None:
            return False
        path = self._compute_path(my_xy, goal, max_nodes=100)
        if path is None or len(path) > budget:
            return False
        # Hijack `self.path` for one move — the next turn's step (or
        # frontier search) will rebuild it.
        self.path = path
        self.path_index = 0
        self.frontier_target = None
        self._follow_path(ct)
        return True

    def _economy_try_heal_chain_divert(self, ct, my_xy):
        """Tighter in-task heal divert (chain / harvester / claim_ore).
        Restricted to chain-piece buildings (CONVEYOR / BRIDGE /
        SPLITTER / ARMOURED_CONVEYOR) and a 4-tile path budget. Returns
        True if a heal / move action fired — caller MUST return so the
        normal step logic doesn't try to do something else this turn.
        Once the diverted-to building is back at full HP this helper
        starts returning False, and the original step resumes naturally
        because we never touched its state (target_ore / chain_start /
        chain_path are all preserved)."""
        return self._economy_try_heal_in_vision(
            ct, my_xy,
            max_path=self._HEAL_TASK_DIVERT_BUDGET,
            target_types=self._HEAL_TASK_DIVERT_TYPES,
        )

    def _heal_find_adjacent(self, my_xy, target_xy):
        """Closest 8-neighbour of `target_xy` that is bot-walkable, used
        as the A* goal when navigating to heal a damaged building.
        Patrol's `_patrol_find_adjacent` does the same job and is reused
        when available; this is the economy-side fallback for any
        future role that doesn't have the patrol mixin loaded."""
        tc = self.tile_cache
        best = None
        best_dsq = None
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
                d2 = (nxy[0] - my_xy[0]) ** 2 + (nxy[1] - my_xy[1]) ** 2
                if best_dsq is None or d2 < best_dsq:
                    best_dsq = d2
                    best = nxy
        return best

    def _economy_self_heal_if_safe(self, ct, my_xy):
        """Self-heal if (a) we're damaged, (b) action cooldown is ready,
        and (c) the next path step doesn't need a road-build (i.e., the
        next tile we'll move onto already holds a walkable building).
        Heal uses the action cooldown that road-build needs; if we
        spend it on healing the bot will then be unable to move past
        an empty tile that requires a road. No-op otherwise."""
        if ct.get_action_cooldown() != 0:
            return
        try:
            cur_hp = ct.get_hp()
        except Exception:
            return
        if cur_hp >= GC.BUILDER_BOT_MAX_HP:
            return
        if self.path and self.path_index < len(self.path):
            next_tile = self.path[self.path_index]
            if next_tile == my_xy and self.path_index + 1 < len(self.path):
                next_tile = self.path[self.path_index + 1]
            info = self.tile_cache.get(next_tile)
            # Conservative: if we don't know, can't confirm walkable.
            if info is None:
                return
            env, bid, etype, _team = info
            if env == Environment.WALL:
                return
            # Empty / ore tile would need a road build → save the action.
            if bid is None:
                return
            if etype not in _WALKABLE_BUILDINGS:
                return
        my_pos = xy_to_pos(my_xy)
        if ct.can_heal(my_pos):
            ct.heal(my_pos)
            if DEBUG:
                print(f"[{self.corner}] self-heal hp={cur_hp}",
                      file=sys.stderr)
