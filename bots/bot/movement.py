import sys
from cambc import Direction, EntityType, Environment, GameConstants as GC
from constants import (
    POSITION_HISTORY_LEN, OSCILLATION_THRESHOLD, ASTAR_MAX_NODES, DEBUG,
)
from utils import (
    direction_between, euclidean_dist_sq, xy_to_pos, DIRECTION_DELTAS,
)
from pathfinding import astar_cached, _WALKABLE_BUILDINGS


# All 8 move directions for sidestep / escape
_ALL_DIRS = [
    Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
    Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST,
]


class MovementMixin:
    """Path following, stuck detection, oscillation breaking."""

    # ------------------------------------------------------------------ #
    #  Centralised A* that auto-includes oscillation / avoid-bot walls    #
    # ------------------------------------------------------------------ #

    def _compute_path(self, start, goal, extra_walls=None, cardinal_only=False,
                      known_only=False, max_nodes=None):
        """A* wrapper that merges in oscillation_walls, avoid_bots, stationary
        bots, and the enemy core 3x3 footprint as impassable walls.
        Dynamically scales max_nodes based on remaining CPU time budget.

        known_only=True treats unseen tiles as impassable. Use for disruptor
        long-range pathing so A* doesn't explore hundreds of unknown tiles.
        """
        # Dynamic node budget: leave 400μs safety margin, ~0.8μs per node
        ct = getattr(self, '_ct', None)
        # Patrol-bot paths are bounded by PATROL_MAX_DISTANCE (≈ 20 tiles
        # from the core) — they never need the full 800-node A* budget
        # of an exploring economy bot. Capping at 400 nodes here turns
        # the worst-case path compute from ~640μs into ~320μs and stops
        # the rare unreachable target (e.g. a damaged building behind a
        # turret cluster) from burning the rest of a 2 ms turn budget
        # on a search that's going to fail anyway.
        cap = ASTAR_MAX_NODES // 2 if self.role == 'patrol' else ASTAR_MAX_NODES
        # Caller-supplied max_nodes is a HARD upper bound on top of the
        # dynamic CPU-time budget. Used by heal sites that should never
        # explore beyond the bot's vision footprint.
        if max_nodes is not None:
            cap = min(cap, max_nodes)
        if ct is not None:
            elapsed = ct.get_cpu_time_elapsed()
            remaining = 1800 - elapsed  # leave 200μs buffer beyond this
            budget = min(cap, max(50, int(remaining / 0.8)))
            if budget < 50:
                return None  # no time for A*
        else:
            budget = cap

        walls = set()
        if self.oscillation_walls:
            walls.update(self.oscillation_walls)
        if self.avoid_bots:
            for xy in self.bot_pos_cache:
                if xy != start:
                    walls.add(xy)
            self.avoid_bots = False
        # Treat bots stationary for 3+ turns as walls
        for xy, count in self._bot_stationary.items():
            if count >= 3 and xy != start and xy != goal:
                walls.add(xy)
        # Treat the entire enemy core 3x3 footprint as walls
        if self.enemy_core_pos is not None:
            ecx, ecy = self.enemy_core_pos
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    walls.add((ecx + dx, ecy + dy))
        # Patrol bots cannot bust through allied barriers — inject all
        # known allied barrier tiles as walls so patrol pathfinding
        # routes around them. Economy bots get cost 3 (bustable) instead.
        # Reads from the maintained set instead of iterating tile_cache:
        # on mature 50x50 maps the full scan was ~1.25ms per call, the
        # dominant patrol-side TLE source on damage-driven repaths.
        if self.role == 'patrol' and self.allied_barriers:
            for xy in self.allied_barriers:
                if xy != start and xy != goal:
                    walls.add(xy)
        # Disruptors avoid known enemy launcher footprints (launcher + 8-adj).
        # Don't block start/goal tiles even if they fall inside the zone —
        # we may need to path OUT of a zone, and Case-B / Step 9 targets can
        # legitimately be inside (we use our own launcher to reach them).
        blocked = getattr(self, 'blocked_launcher_tiles', None)
        if blocked:
            for xy in blocked:
                if xy != start and xy != goal:
                    walls.add(xy)
        if extra_walls:
            walls.update(extra_walls)
        return astar_cached(
            start, goal, self.tile_cache, self.known_walls,
            extra_walls=walls if walls else None,
            cardinal_only=cardinal_only,
            max_nodes=budget,
            map_w=self.map_w, map_h=self.map_h,
            my_team=self.my_team_cache,
            known_only=known_only,
        )

    # ------------------------------------------------------------------ #
    #  Path following                                                     #
    # ------------------------------------------------------------------ #

    def _follow_path(self, ct):
        """Follow self.path step by step. Build roads on empty tiles.

        Returns:
            'done'       — reached end of path
            'moved'      — moved one tile this turn
            'built_road' — built a road, will move next turn
            'blocked'    — can't progress, caller should repath
            None         — no path set
        """
        if not self.path or self.path_index >= len(self.path):
            return 'done'

        my_xy = self._my_xy

        # --- Advance past tiles we're already on ---
        target_xy = self.path[self.path_index]
        if my_xy == target_xy:
            self.path_index += 1
            if self.path_index >= len(self.path):
                return 'done'
            target_xy = self.path[self.path_index]

        direction = direction_between(my_xy, target_xy)

        # --- Try to move ---
        if ct.can_move(direction):
            ct.move(direction)
            self.path_index += 1
            self._stuck_count = 0
            # Clear oscillation walls once we've made progress
            if self.oscillation_walls and target_xy not in self.oscillation_walls:
                self.oscillation_walls = set()
            return 'moved'

        # --- Can't move — diagnose ---
        entry = self.tile_cache.get(target_xy)

        if DEBUG and self.step == 'explore':
            tpos = xy_to_pos(target_xy)
            dx = target_xy[0] - my_xy[0]
            dy = target_xy[1] - my_xy[1]
            diag = abs(dx) == 1 and abs(dy) == 1
            adj_info = ""
            if diag:
                a1 = (my_xy[0] + dx, my_xy[1])
                a2 = (my_xy[0], my_xy[1] + dy)
                e1 = ct.get_tile_env(xy_to_pos(a1))
                e2 = ct.get_tile_env(xy_to_pos(a2))
                adj_info = f" diag_adj=({a1}={e1},  {a2}={e2})"
            print(f"[{self.corner}] can't move {my_xy}→{target_xy} "
                  f"dir={direction} mcd={ct.get_move_cooldown()}{adj_info}",
                  file=sys.stderr)

        # Wall — repath
        if target_xy in self.known_walls:
            return 'blocked'
        if entry and entry[0] == Environment.WALL:
            return 'blocked'

        # Empty / ore tile with no building — build a road to walk on it
        if entry is None or entry[1] is None:
            if ct.get_action_cooldown() == 0:
                target_pos = xy_to_pos(target_xy)
                can_afford = self._can_spend(ct, GC.ROAD_BASE_COST[0])
                can_build = ct.can_build_road(target_pos)
                if DEBUG and self.step == 'explore' and not (can_afford and can_build):
                    ti, _ = ct.get_global_resources()
                    dsq = euclidean_dist_sq(my_xy, target_xy)
                    # Try building with a fresh Position to rule out
                    # stale Position objects. get_tile_env raises if the
                    # tile is outside vision — guard with is_in_vision.
                    from cambc import Position as Pos
                    fresh = Pos(target_xy[0], target_xy[1])
                    fresh_build = ct.can_build_road(fresh)
                    if ct.is_in_vision(fresh):
                        env_at = ct.get_tile_env(fresh)
                        bid_at = ct.get_tile_building_id(fresh)
                    else:
                        env_at = 'OUT_OF_VISION'
                        bid_at = None
                    print(f"[{self.corner}] road fail at {target_xy}: "
                          f"afford={can_afford} build={can_build} "
                          f"fresh_build={fresh_build} env={env_at} "
                          f"bid={bid_at} dsq={dsq} "
                          f"acd={ct.get_action_cooldown()}",
                          file=sys.stderr)
                if can_afford and can_build:
                    ct.build_road(target_pos)
                    env = entry[0] if entry else Environment.EMPTY
                    self.tile_cache[target_xy] = (env, None, None, None)
                    if ct.can_move(direction):
                        ct.move(direction)
                        self.path_index += 1
                        self._stuck_count = 0
                        return 'moved'
                    return 'built_road'
                return 'blocked'
            return 'built_road'

        # Allied barrier — economy bots bust through (destroy → road →
        # move → restore). Patrol injects barriers into walls in
        # _compute_path so they path around instead of getting here.
        if (entry[1] is not None
                and entry[2] == EntityType.BARRIER
                and entry[3] == self.my_team_cache
                and self.role == 'economy'):
            return self._barrier_bust_step(ct, target_xy, direction)

        # Non-walkable building
        if entry[1] is not None and entry[2] not in _WALKABLE_BUILDINGS:
            # Marker (any team) — build a road over it. Any team may
            # build over a marker; the engine auto-destroys the marker
            # as part of the build action.
            if entry[2] == EntityType.MARKER:
                if ct.get_action_cooldown() == 0:
                    target_pos = xy_to_pos(target_xy)
                    if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                            and ct.can_build_road(target_pos)):
                        ct.build_road(target_pos)
                        if ct.can_move(direction):
                            ct.move(direction)
                            self.path_index += 1
                            self._stuck_count = 0
                            return 'moved'
                        return 'built_road'
                return 'built_road'
            return 'blocked'

        # Walkable tile but can_move failed — bot blocking
        if target_xy in self.bot_pos_cache:
            self._stuck_count += 1
            if self._stuck_count >= 2:
                # Try diagonal sidestep to let the other bot pass
                if self._try_sidestep(ct, my_xy, direction):
                    return 'moved'
                # Can't sidestep — repath avoiding bots
                self.avoid_bots = True
                if DEBUG: print(f"[{self.corner}] bot blocking at ({target_xy[0]},{target_xy[1]})",
                      file=sys.stderr)
                return 'blocked'
            return 'built_road'  # Wait one turn

        return 'blocked'

    # ------------------------------------------------------------------ #
    #  Barrier busting (economy bots only)                                #
    # ------------------------------------------------------------------ #

    def _barrier_bust_step(self, ct, target_xy, direction):
        """Replace the allied barrier at target_xy with a road so the bot
        can walk through. Each busted tile is appended to
        pending_barrier_restores and rebuilt by _barrier_bust_restore
        once the bot has moved off that tile (possibly several turns
        later if we're busting consecutive barriers).

        Ore tiles are NEVER queued for restore — if the bot busted a
        barrier off an ore it's almost certainly heading there to mine,
        and rebuilding the barrier on top of a new harvester would fail
        and leak into pending_repairs.

        Returns one of the standard _follow_path return strings.
        """
        target_pos = xy_to_pos(target_xy)

        # Already destroyed + road built from a prior turn — try to move through.
        entry = self.tile_cache.get(target_xy)
        is_ore = entry is not None and entry[0] in (
            Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)
        if entry and entry[2] == EntityType.ROAD and entry[3] == self.my_team_cache:
            if ct.can_move(direction):
                ct.move(direction)
                self.path_index += 1
                self._stuck_count = 0
                if not is_ore and target_xy not in self.pending_barrier_restores:
                    self.pending_barrier_restores.append(target_xy)
                return 'moved'
            return 'built_road'

        # Destroy the barrier (free, no cooldown).
        if ct.can_destroy(target_pos):
            ct.destroy(target_pos)
            env = entry[0] if entry else Environment.EMPTY
            self.tile_cache[target_xy] = (env, None, None, None)
            if DEBUG:
                print(f"[{self.corner}] barrier bust at ({target_xy[0]},{target_xy[1]})",
                      file=sys.stderr)

        # Build a road on the cleared tile (action cooldown).
        if (ct.get_action_cooldown() == 0
                and self._can_spend(ct, GC.ROAD_BASE_COST[0])
                and ct.can_build_road(target_pos)):
            ct.build_road(target_pos)
            # Try to move on the same turn (build + move share the turn).
            if ct.can_move(direction):
                ct.move(direction)
                self.path_index += 1
                self._stuck_count = 0
                if not is_ore and target_xy not in self.pending_barrier_restores:
                    self.pending_barrier_restores.append(target_xy)
                return 'moved'
            if not is_ore and target_xy not in self.pending_barrier_restores:
                self.pending_barrier_restores.append(target_xy)
            return 'built_road'

        # Cooldown not ready — wait. Keep the tile queued so we don't
        # lose it while we wait for the cooldown to tick down.
        if not is_ore and target_xy not in self.pending_barrier_restores:
            self.pending_barrier_restores.append(target_xy)
        return 'built_road'

    def _barrier_bust_restore(self, ct):
        """Rebuild any previously-busted barriers we're in action range
        of. Called once per turn from run_builder after _scan_turn.

        - Skips tiles the bot is standing on (can't build non-walkable
          on own tile — wait for the bot to step off naturally).
        - Fires at most one build per turn (one action cooldown per
          turn), leaving the rest of the queue for later turns.
        - Tiles permanently out of reach (>ACTION_RADIUS_SQ once the
          bot has moved on) are handed to pending_repairs so the
          normal repair pipeline can catch them.
        """
        queue = self.pending_barrier_restores
        if not queue:
            return
        my_xy = self._my_xy
        kept = []
        already_acted = False

        for bxy in queue:
            # Already rebuilt by someone else (patrol, another economy bot).
            existing = self.tile_cache.get(bxy)
            if (existing and existing[1] is not None
                    and existing[2] == EntityType.BARRIER
                    and existing[3] == self.my_team_cache):
                continue  # drop from queue

            dx = my_xy[0] - bxy[0]
            dy = my_xy[1] - bxy[1]
            dsq = dx * dx + dy * dy

            # Can't build non-walkable on own tile — keep waiting.
            if dsq == 0:
                kept.append(bxy)
                continue
            # Out of action range — defer to repair pipeline.
            if dsq > GC.ACTION_RADIUS_SQ:
                if bxy not in self.pending_repairs:
                    self.pending_repairs.append(bxy)
                continue
            # We already spent this turn's cooldown on an earlier tile
            # in the queue. Keep for next turn.
            if already_acted or ct.get_action_cooldown() != 0:
                kept.append(bxy)
                continue

            b_pos = xy_to_pos(bxy)
            entry = self.tile_cache.get(bxy)
            # Tile must be free of buildings (except an allied road we
            # placed during the bust). Free destroys don't consume the
            # action cooldown, so we can still fire the build this turn.
            if entry and entry[1] is not None:
                if entry[3] == self.my_team_cache and entry[2] == EntityType.ROAD:
                    if ct.can_destroy(b_pos):
                        ct.destroy(b_pos)
                        self.tile_cache[bxy] = (entry[0], None, None, None)
                else:
                    # Something unexpected occupied the tile — defer.
                    if bxy not in self.pending_repairs:
                        self.pending_repairs.append(bxy)
                    continue

            if (self._can_spend(ct, GC.BARRIER_BASE_COST[0])
                    and ct.can_build_barrier(b_pos)):
                ct.build_barrier(b_pos)
                already_acted = True
                if DEBUG:
                    print(f"[{self.corner}] barrier restore at ({bxy[0]},{bxy[1]})",
                          file=sys.stderr)
                continue
            # Build failed unexpectedly — keep retrying next turn.
            kept.append(bxy)

        self.pending_barrier_restores = kept

    # ------------------------------------------------------------------ #
    #  Sidestep: dodge out of a narrow corridor to yield                  #
    # ------------------------------------------------------------------ #

    def _try_sidestep(self, ct, my_xy, blocked_dir):
        """Try moving to any adjacent tile to escape a head-on deadlock.
        Prefer diagonals (to escape 1-wide corridors). Avoid backwards."""
        opposite = blocked_dir.opposite()
        for d in _ALL_DIRS:
            if d == opposite:
                continue  # Don't go backwards
            if ct.can_move(d):
                ct.move(d)
                self.path = None  # Force repath from new position
                self.path_index = 0
                self._stuck_count = 0
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Oscillation detection                                              #
    # ------------------------------------------------------------------ #

    def _check_oscillation(self, my_xy):
        """If current pos appears 3+ times in history, break out with temp walls."""
        if self._bridge_walk_target is not None:
            return  # Don't interfere with bridge walks
        if self.role == 'disruptor':
            # Disruptors legitimately stand still for many turns while
            # firing on a feeder/disrupt target. Oscillation walls populated
            # during that idle window would block the tiles they just
            # walked and strand the bot on reset.
            return
        if len(self.pos_history) < OSCILLATION_THRESHOLD:
            return
        count = self.pos_history.count(my_xy)
        if count >= OSCILLATION_THRESHOLD:
            # Accumulate oscillation walls (don't clear previous ones)
            for p in self.pos_history:
                if p != my_xy:
                    self.oscillation_walls.add(p)
            self.path = None
            self.path_index = 0
            self.frontier_target = None

    # ------------------------------------------------------------------ #
    #  Top-level economy stuck detection                                  #
    # ------------------------------------------------------------------ #

    def _check_top_level_stuck(self, ct, my_xy):
        """Position + step unchanged for 2+ turns → force escape + explore."""
        # Patrol, destructor, and disruptor bots handle their own stuck
        # recovery (or are tolerant of being stuck near a target).
        if self.role in ('patrol', 'destructor', 'disruptor'):
            return
        # Skip steps where waiting is expected
        if self.step in ("claim_ore", "build_harvester", "observe_conveyor",
                        "leave_core", "chain_to_core", "steal_harvester") \
                or self._bridge_walk_target is not None:
            self.economy_stuck_turns = 0
            self.economy_stuck_pos = my_xy
            self.economy_stuck_step = self.step
            return

        if my_xy == self.economy_stuck_pos and self.step == self.economy_stuck_step:
            self.economy_stuck_turns += 1
            if self.economy_stuck_turns >= 5:
                if DEBUG: print(f"[{self.corner}] STUCK {self.step} at ({my_xy[0]},{my_xy[1]}) for {self.economy_stuck_turns}t",
                      file=sys.stderr)
                self.economy_stuck_turns = 0
                self.path = None
                self.path_index = 0
                self.frontier_target = None
                self.target_ore = None
                self.avoid_bots = True
                self._frontier_blacklist = set()
                # Add current + all recent positions to oscillation walls
                self.oscillation_walls.add(my_xy)
                for p in self.pos_history:
                    self.oscillation_walls.add(p)
                self.step = "explore"

                # Force escape: prefer directions toward core center
                escape_dirs = sorted(
                    _ALL_DIRS,
                    key=lambda d: (
                        euclidean_dist_sq(
                            (my_xy[0] + DIRECTION_DELTAS[d][0],
                             my_xy[1] + DIRECTION_DELTAS[d][1]),
                            self.core_pos,
                        ) if self.core_pos else 0
                    ),
                )
                for d in escape_dirs:
                    if ct.can_move(d):
                        ct.move(d)
                        break
                else:
                    # Can't move anywhere — try building a road + move
                    if ct.get_action_cooldown() == 0:
                        for d in _ALL_DIRS:
                            dx, dy = DIRECTION_DELTAS[d]
                            esc = (my_xy[0] + dx, my_xy[1] + dy)
                            if esc in self.known_walls:
                                continue
                            esc_entry = self.tile_cache.get(esc)
                            if esc_entry and esc_entry[0] == Environment.WALL:
                                continue
                            esc_pos = xy_to_pos(esc)
                            if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                                    and ct.can_build_road(esc_pos)):
                                ct.build_road(esc_pos)
                                if ct.can_move(d):
                                    ct.move(d)
                                break
        else:
            self.economy_stuck_turns = 0
            self.economy_stuck_pos = my_xy
            self.economy_stuck_step = self.step

    # ------------------------------------------------------------------ #
    #  Debug: draw path as indicator lines                                #
    # ------------------------------------------------------------------ #

    # Color per corner so paths are distinguishable in replay
    _PATH_COLORS = {
        "NW": (255, 50, 50),    # red
        "NE": (50, 50, 255),    # blue
        "SW": (50, 255, 50),    # green
        "SE": (255, 200, 50),   # yellow
        "??": (200, 200, 200),  # grey
    }

    _MAX_PATH_LINES = 6  # Max segments to draw per path

    def _draw_path(self, ct, my_xy):
        """Draw path segments + dot at goal. Capped at _MAX_PATH_LINES segments."""
        r, g, b = self._PATH_COLORS.get(self.corner, (200, 200, 200))
        me = xy_to_pos(my_xy)
        drew = False

        # Movement path
        if self.path and self.path_index < len(self.path):
            remaining = len(self.path) - self.path_index
            n = min(remaining, self._MAX_PATH_LINES)
            prev = me
            for i in range(self.path_index, self.path_index + n):
                cur = xy_to_pos(self.path[i])
                ct.draw_indicator_line(prev, cur, r, g, b)
                prev = cur
            # Dot at the final goal (even if we didn't draw all segments)
            ct.draw_indicator_dot(xy_to_pos(self.path[-1]), r, g, b)
            drew = True

        # Chain path — white
        if self.chain_path and self.chain_index < len(self.chain_path):
            remaining = len(self.chain_path) - self.chain_index
            n = min(remaining, self._MAX_PATH_LINES)
            prev = me
            for i in range(self.chain_index, self.chain_index + n):
                cur = xy_to_pos(self.chain_path[i])
                ct.draw_indicator_line(prev, cur, 255, 255, 255)
                prev = cur
            ct.draw_indicator_dot(xy_to_pos(self.chain_path[-1]), 255, 255, 255)
            drew = True

        # Fallback: line to current goal
        if not drew:
            goal = (self.target_ore or self.frontier_target
                    or self.observe_xy or self.core_pos)
            if goal and goal != my_xy:
                ct.draw_indicator_line(me, xy_to_pos(goal), r, g, b)
                ct.draw_indicator_dot(xy_to_pos(goal), r, g, b)
            else:
                ct.draw_indicator_dot(me, r, g, b)
