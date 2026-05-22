import sys
from cambc import Direction, EntityType, Environment, GameConstants as GC
from constants import (
    ORE_PATHFIND_GIVE_UP_DIST, MAX_HARVESTERS_PER_BOT,
    SENTINEL_ATTACK_THRESHOLD, DEFEND_HARVESTER_RANGE,
    MIN_TITANIUM, HARVEST_AFFORD_MULT, DEBUG, ECONOMY_TEST_MODE, PLACE_GUNNER,
)
from utils import (
    euclidean_dist_sq, cardinal_direction_between, direction_between,
    DIRECTION_DELTAS, xy_to_pos, neighbors_4,
)
from pathfinding import astar_cached, astar_monotonic


# Markers excluded — bots can't walk onto them, and we can't fire at
# a marker from an adjacent tile (fire targets must be walkable).
_ENEMY_WALKABLE_STEAL = frozenset({
    EntityType.ROAD, EntityType.CONVEYOR, EntityType.SPLITTER,
    EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
})

_STEAL_STOLEN_MARKERS = frozenset({
    EntityType.SENTINEL, EntityType.CONVEYOR, EntityType.BRIDGE,
})

# Allied non-walkable defense buildings that also disqualify a
# cardinal-of-ore tile from being used as a chain start. We never want to
# tear down a turret we just paid for to lay a conveyor in its spot.
# BARRIER is intentionally NOT in this set: allied barriers are cheap
# (2 Ti), destroyable by the owning team for a refund, and the
# conveyor/bridge build blocks already destroy them before placing
# (see `cs_entry[2] in (ROAD, BARRIER, MARKER)` paths in
# _step_claim_ore / _step_claim_ore_axionite).
_ALLIED_CHAIN_START_BLOCKERS = frozenset({
    EntityType.GUNNER, EntityType.SENTINEL,
    EntityType.BREACH, EntityType.LAUNCHER, EntityType.FOUNDRY,
    EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR,
    EntityType.BRIDGE, EntityType.SPLITTER,
})

_ALL_8_DIRS = [
    Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
    Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST,
]

# For Step 5a gunner placement: candidates perpendicular to the conveyor
# direction (these tiles defend the chain approach).
_PERPENDICULAR = {
    Direction.NORTH: (Direction.EAST, Direction.WEST),
    Direction.SOUTH: (Direction.EAST, Direction.WEST),
    Direction.EAST:  (Direction.NORTH, Direction.SOUTH),
    Direction.WEST:  (Direction.NORTH, Direction.SOUTH),
}


class HarvestingMixin:
    """Ore detection, claim ore (on-ore positioning), harvester placement."""

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _chain_start_blocked(self, xy, entry):
        """A candidate cardinal-of-ore tile is 'blocked' if we can neither
        walk on it (to clear) nor build a conveyor over it: walls,
        harvesters (allied or enemy), enemy barriers (we can't destroy
        them), our own non-walkable defense buildings (gunners, sentinels,
        barriers, etc.), and claimed ores (ore with a road or builder bot
        on it — another bot is already working that tile)."""
        if xy in self.known_walls:
            return True
        if entry is None:
            return False
        env = entry[0]
        if env == Environment.WALL:
            return True
        # Claimed ore: ore tile with a road/bot signals another bot claimed it
        if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            bid, etype = entry[1], entry[2]
            if (bid is not None and etype == EntityType.ROAD) or xy in self.bot_pos_cache:
                return True
        bid, etype, team = entry[1], entry[2], entry[3]
        if bid is None:
            return False
        if etype == EntityType.HARVESTER:
            return True
        if etype == EntityType.BARRIER and team != self.my_team_cache:
            return True
        if team == self.my_team_cache and etype in _ALLIED_CHAIN_START_BLOCKERS:
            return True
        return False

    def _allied_barrier_walls(self):
        """Set of allied non-walkable defense tiles to inject as walls
        into chain routing — gunners, sentinels, and other turret-class
        buildings. Barriers are NOT included: the chain builder destroys
        any allied barrier in its path (free destroy + Ti refund) then
        builds a conveyor on that tile.

        Reads the incrementally-maintained `self.allied_blockers` set
        (kept in sync by scanning.py). Was a full tile_cache scan
        (~2500 entries × 0.25μs = 625μs per call) on mature 50x50
        maps; now O(1) — caller already pays the cost of iterating
        the set inside _compute_path's wall set construction.
        """
        return self.allied_blockers

    def _in_defend_range(self, xy):
        """True if this tile is within DEFEND_HARVESTER_RANGE (Euclidean)
        of the core center — qualifies for enhanced defense (Step 5a)."""
        if self.core_pos is None:
            return False
        return euclidean_dist_sq(xy, self.core_pos) <= DEFEND_HARVESTER_RANGE * DEFEND_HARVESTER_RANGE

    def _can_spend(self, ct, base_cost_ti):
        """Hard floor on every titanium spend. True iff paying
        `base_cost_ti` (scaled by current scale) would leave the team
        with AT LEAST MIN_TITANIUM (also scaled by current scale)
        titanium remaining. Gates economy builds, defense builds, roads,
        gunner rotations, and bot spawns. The ONLY exempt spend is
        panic-mode patrol spawning in core_logic — everything else falls
        through to its "cannot afford" path (usually waiting) instead of
        draining the reserve.

        MIN_TITANIUM is interpreted as a base value (e.g. "2 builder
        bots = 40 Ti") and rises with the scale factor so the reserve
        always represents the same purchasing power.

        Exemption: economy bots that haven't yet completed their first
        chain bypass the reserve. Until that first harvester is
        delivering ore back to the core, gating every road / conveyor
        / harvester spend on a 50 Ti reserve can stall the bot
        indefinitely (it can't pay for the chain that would replenish
        the reserve). After `_first_chain_done` flips True the normal
        floor applies again."""
        if base_cost_ti <= 0:
            return True
        ti, _ = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0
        cost = int(base_cost_ti * scale)
        if (self.role == 'economy'
                and not getattr(self, '_first_chain_done', False)):
            return ti >= cost
        # Disruptors bypass the MIN_TITANIUM reserve when paying for a
        # sentinel or gunner. These turrets are the disruptor's main
        # tactical output (intercept placement, chain-block, splitter-
        # sentinel) — gating them on the team-wide reserve causes a
        # disruptor to walk all the way to the placement tile and then
        # idle there indefinitely while waiting for Ti to climb above
        # the floor.
        if (self.role == 'disruptor'
                and base_cost_ti in (GC.SENTINEL_BASE_COST[0],
                                     GC.GUNNER_BASE_COST[0])):
            return ti >= cost
        floor = int(MIN_TITANIUM * scale)
        return ti - cost >= floor

    # Legacy alias — kept to minimise churn at existing call sites.
    def _has_economy_reserve(self, ct, base_cost_ti):
        return self._can_spend(ct, base_cost_ti)

    # ------------------------------------------------------------------ #
    #  Resource affordability check                                       #
    # ------------------------------------------------------------------ #

    def _can_afford_harvest(self, ct, include_sentinel=False):
        """Check if we can afford a harvester + a tunable fraction of
        the conveyors needed to chain back. Passive titanium income
        (10 per 4 rounds) covers the rest while the bot is chaining,
        so there's no point waiting for the full chain cost up front —
        if we run out mid-chain, _step_chain_to_core already pauses
        until resources arrive. The fraction is HARVEST_AFFORD_MULT
        (constants.py); lower = start sooner, higher = wait longer."""
        ti, _ = ct.get_global_resources()
        scale = ct.get_scale_percent() / 100.0

        harvester_cost = int(GC.HARVESTER_BASE_COST[0] * scale)
        conveyor_cost = int(GC.CONVEYOR_BASE_COST[0] * scale)

        # Estimate chain length as Manhattan distance from ore to core
        if self.target_ore and self.core_pos:
            dx = abs(self.target_ore[0] - self.core_pos[0])
            dy = abs(self.target_ore[1] - self.core_pos[1])
            est_conveyors = dx + dy
        else:
            est_conveyors = 10  # fallback

        chain_budget = int(est_conveyors * HARVEST_AFFORD_MULT) * conveyor_cost
        total = harvester_cost + chain_budget
        if include_sentinel:
            total += int(GC.SENTINEL_BASE_COST[0] * scale)
        # Keep a MIN_TITANIUM emergency-defense buffer on top of the
        # estimated chain cost — don't even start a new harvester if
        # finishing it would drain us below the (scale-adjusted) reserve.
        return ti - total >= int(MIN_TITANIUM * scale)

    # ------------------------------------------------------------------ #
    #  Step: goto_ore                                                     #
    # ------------------------------------------------------------------ #

    def _step_goto_ore(self, ct):
        """Pathfind ONTO a titanium ore tile, building roads as needed.

        Standing on ore with a road = 'claimed'. Transitions to claim_ore
        when reached, or back to explore if unreachable / too far / claimed.
        """
        my_xy = self._my_xy

        # Pause the trip to heal a damaged ally building inside our
        # action radius — same pattern as chain_to_core. Free (no
        # movement, just an action-cooldown spend on the heal). Then
        # try a short-detour heal divert (≤4 tiles) so a damaged chain
        # piece next to our route gets fixed instead of walked past.
        if self._economy_check_heal_action_radius(ct, my_xy) != 'none':
            return
        if self._economy_try_heal_chain_divert(ct, my_xy):
            return

        if self.target_ore is None:
            self.step = "explore"
            return

        # Already standing on the ore
        if my_xy == self.target_ore:
            self.step = "claim_ore"
            if DEBUG: print(f"[{self.corner}] on ore at ({my_xy[0]},{my_xy[1]}), claiming",
                  file=sys.stderr)
            return

        # Check if ore was claimed by someone else
        if self._is_ore_claimed(self.target_ore, my_xy):
            self._abandon_ore()
            return

        # If an enemy bot has been camping the target ore for 3+ turns,
        # consider it untargetable and give up
        if self._enemy_camping_ore(self.target_ore):
            if DEBUG: print(f"[{self.corner}] enemy camping ore ({self.target_ore[0]},{self.target_ore[1]}), abandoning",
                  file=sys.stderr)
            self._abandon_ore()
            return

        # Check for closer unclaimed ore in VISION only (not full cache)
        best_ore = None
        best_dist_sq = euclidean_dist_sq(my_xy, self.target_ore)
        for xy in self._last_vision_set:
            entry = self.tile_cache.get(xy)
            if entry is None or entry[0] != Environment.ORE_TITANIUM:
                continue
            if xy in self._ore_blacklist:
                continue
            if self._is_ore_enclosed(xy):
                continue
            if self._is_ore_claimed(xy, my_xy):
                continue
            if self._enemy_camping_ore(xy):
                continue
            dsq = euclidean_dist_sq(my_xy, xy)
            if dsq < best_dist_sq:
                best_dist_sq = dsq
                best_ore = xy

        if best_ore is not None:
            self.target_ore = best_ore
            self.path = None
            self.path_index = 0

        # Compute path if needed
        max_dist = None if self._from_stored_ore else ORE_PATHFIND_GIVE_UP_DIST
        if self.path is None:
            if self.tle_recovery_turns > 0:
                return  # Skip A* during TLE recovery
            self.path = self._compute_path(my_xy, self.target_ore)
            self.path_index = 0
            if self.path is None or (max_dist and len(self.path) > max_dist):
                if DEBUG: print(f"[{self.corner}] ore too far/unreachable, back to explore",
                      file=sys.stderr)
                self._abandon_ore()
                return

        # Follow path
        result = self._follow_path(ct)
        if result == 'done':
            self.step = "claim_ore"
            if DEBUG: print(f"[{self.corner}] reached ore at ({self.target_ore[0]},{self.target_ore[1]})",
                  file=sys.stderr)
        elif result == 'blocked':
            if self.tle_recovery_turns == 0:
                self.path = self._compute_path(my_xy, self.target_ore)
                self.path_index = 0
            if self.path is None or (max_dist and len(self.path) > max_dist):
                self._abandon_ore()

    # ------------------------------------------------------------------ #
    #  Step: claim_ore                                                    #
    # ------------------------------------------------------------------ #

    def _step_claim_ore(self, ct):
        """Bot is standing on the ore (on a road). From this single position:
        1. Place conveyor on cardinal-adjacent tile closest to core (chain start)
        2. Conditionally place a sentinel on another cardinal-adjacent tile
        All builds happen from one position — no extra pathfinding.

        Dispatches to _step_claim_ore_axionite when the ore is axionite —
        that path skips sentinel/gunner/barrier entirely.
        """
        my_xy = self._my_xy

        if self.target_ore is None:
            self.step = "explore"
            return

        # Off the ore — common when _build_defense_barrier walked us
        # off to fire at an enemy walkable on a barrier tile. Walk
        # back instead of abandoning, unless the ore tile now holds
        # something that means the claim is lost (enemy harvester on
        # it, or unknown — out of vision).
        if my_xy != self.target_ore:
            ore_entry = self.tile_cache.get(self.target_ore)
            if ore_entry is None:
                self.step = "explore"
                return
            t_bid, t_etype = ore_entry[1], ore_entry[2]
            # Already harvested or replaced — give up.
            if t_bid is not None and t_etype in (
                    EntityType.HARVESTER, EntityType.SENTINEL,
                    EntityType.GUNNER, EntityType.BREACH,
                    EntityType.LAUNCHER, EntityType.FOUNDRY,
                    EntityType.BARRIER, EntityType.CONVEYOR,
                    EntityType.SPLITTER, EntityType.BRIDGE,
                    EntityType.ARMOURED_CONVEYOR):
                self.step = "explore"
                return
            # Bot blocking the ore — give up.
            if self.target_ore in self.bot_pos_cache:
                self.step = "explore"
                return
            # Walk back. Direct cardinal move when possible; otherwise
            # let _follow_path handle path computation next turn.
            d = cardinal_direction_between(my_xy, self.target_ore)
            if ct.can_move(d):
                ct.move(d)
            return

        ore_env_entry = self.tile_cache.get(my_xy)
        if ore_env_entry and ore_env_entry[0] == Environment.ORE_AXIONITE:
            self._step_claim_ore_axionite(ct)
            return

        # Commit to a titanium chain as soon as we're on the ore — the
        # downstream chain dispatch and observe_conveyor branch read
        # current_chain_type to pick rules.
        self.current_chain_type = 'titanium'

        # Re-check enclosed now that we're on the ore (all neighbors visible)
        if self._is_ore_enclosed(self.target_ore):
            if DEBUG: print(f"[{self.corner}] ore ({self.target_ore[0]},{self.target_ore[1]}) enclosed, abandoning",
                  file=sys.stderr)
            self._abandon_ore()
            return

        # Enemy walkable building on ore — fire until destroyed, then build road
        ore_entry = self.tile_cache.get(my_xy)
        if (ore_entry and ore_entry[1] is not None
                and ore_entry[3] != self.my_team_cache
                and ore_entry[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                                     EntityType.SPLITTER, EntityType.BRIDGE,
                                     EntityType.ARMOURED_CONVEYOR)):
            my_pos = xy_to_pos(my_xy)
            if ct.get_action_cooldown() == 0 and ct.can_fire(my_pos):
                ct.fire(my_pos)
            return
        # No building on ore — need a road to claim it
        if ore_entry and ore_entry[1] is None:
            if ct.get_action_cooldown() == 0:
                ore_pos = xy_to_pos(my_xy)
                if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                        and ct.can_build_road(ore_pos)):
                    ct.build_road(ore_pos)
            return

        # Resource check — wait if can't afford
        if not self._can_afford_harvest(ct):
            return

        # Determine chain start tile (once) — closest cardinal to core,
        # falling back to other cardinals if preferred is blocked.
        if self.chain_start is None:
            # Bots launched far from base by enemy launchers can land
            # without ever having seen our core (corner='??'). They
            # have nothing to chain back to; fall back to explore so
            # they find their way home.
            if self.core_pos is None:
                self.step = "explore"
                self.chain_target_ore = None
                return
            pref_dir = cardinal_direction_between(my_xy, self.core_pos)
            pdx, pdy = DIRECTION_DELTAS[pref_dir]
            pref_xy = (my_xy[0] + pdx, my_xy[1] + pdy)
            pref_entry = self.tile_cache.get(pref_xy)
            pref_blocked = self._chain_start_blocked(pref_xy, pref_entry)

            if not pref_blocked:
                self.chain_start = pref_xy
                self._chain_dir = pref_dir
            else:
                # Preferred tile is blocked — try other 3 cardinal directions
                best_alt = None
                best_alt_dist = float('inf')
                for d in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
                    if d == pref_dir:
                        continue
                    adx, ady = DIRECTION_DELTAS[d]
                    alt_xy = (my_xy[0] + adx, my_xy[1] + ady)
                    alt_e = self.tile_cache.get(alt_xy)
                    if self._chain_start_blocked(alt_xy, alt_e):
                        continue
                    dist = euclidean_dist_sq(alt_xy, self.core_pos)
                    if dist < best_alt_dist:
                        best_alt_dist = dist
                        best_alt = alt_xy
                        best_alt_dir = d

                if best_alt is not None:
                    self.chain_start = best_alt
                    self._chain_dir = best_alt_dir
                else:
                    # All 4 cardinal tiles are walls — check for adjacent chain
                    for nx, ny in neighbors_4(my_xy[0], my_xy[1]):
                        adj_e = self.tile_cache.get((nx, ny))
                        if (adj_e and adj_e[3] == self.my_team_cache
                                and adj_e[2] in (EntityType.CONVEYOR, EntityType.BRIDGE,
                                                 EntityType.ARMOURED_CONVEYOR)):
                            # Adjacent chain exists — harvester can output to it directly
                            # Set chain_start to this tile, skip conveyor + chain
                            self.chain_start = (nx, ny)
                            self._chain_dir = cardinal_direction_between(my_xy, (nx, ny))
                            self._sentinel_placed = True
                            # Go straight to build_harvester, skip chain_to_core after
                            self.step = "build_harvester"
                            if DEBUG: print(f"[{self.corner}] all cardinals walled, adjacent chain at ({nx},{ny})",
                                  file=sys.stderr)
                            return
                    # No adjacent chain either — blacklist this ore
                    if DEBUG: print(f"[{self.corner}] all cardinals walled, no adjacent chain, blacklisting ore",
                          file=sys.stderr)
                    self._ore_blacklist.add(self.target_ore)
                    self._abandon_ore()
                    return

        # --- Build conveyor at chain_start if not there yet ---
        cs_entry = self.tile_cache.get(self.chain_start)
        conveyor_exists = (
            cs_entry is not None
            and cs_entry[2] == EntityType.CONVEYOR
            and cs_entry[3] == self.my_team_cache
        )

        if not conveyor_exists and not self._skip_chain_conveyor:
            if ct.get_action_cooldown() > 0:
                return
            cs_pos = xy_to_pos(self.chain_start)

            # Compute monotonic A* path to determine correct conveyor direction
            mono_path = astar_monotonic(
                self.chain_start, self.core_pos,
                self.tile_cache, self.known_walls,
                map_w=self.map_w, map_h=self.map_h,
                extra_walls=self._allied_barrier_walls(),
                my_team=self.my_team_cache,
            )

            if mono_path is None or len(mono_path) == 0:
                # Can't route to core — fall back to core direction
                conv_dir = cardinal_direction_between(self.chain_start, self.core_pos)
            else:
                conv_dir = cardinal_direction_between(self.chain_start, mono_path[0])
                # Don't store path — chain_to_core will recompute with
                # better visibility when it starts

            # Check what the conveyor at chain_start would point at
            cdx, cdy = DIRECTION_DELTAS[conv_dir]
            conv_target = (self.chain_start[0] + cdx, self.chain_start[1] + cdy)
            ct_entry = self.tile_cache.get(conv_target)

            # Bad target: wall, allied non-destroyable (harvester/sentinel/turret),
            # allied transport (would merge without capacity check),
            # or the ore tile itself (harvester will be built there)
            target_invalid = False
            if conv_target == self.target_ore:
                target_invalid = True
            elif conv_target in self.known_walls:
                target_invalid = True
            elif self._is_enemy_core_tile(conv_target):
                target_invalid = True
            elif ct_entry is not None:
                if ct_entry[0] == Environment.WALL:
                    target_invalid = True
                elif ct_entry[1] is not None:
                    etype, team = ct_entry[2], ct_entry[3]
                    if team == self.my_team_cache:
                        # Allied: only roads/markers/barriers are OK (destroyable)
                        if etype not in (EntityType.ROAD, EntityType.MARKER,
                                         EntityType.BARRIER):
                            target_invalid = True
                    else:
                        # Enemy: walkable buildings (we can fire-clear them
                        # when the chain extends through the tile) and
                        # markers (treated as empty — any team can build
                        # over a marker, the engine destroys it on build).
                        if etype not in (EntityType.ROAD, EntityType.CONVEYOR,
                                         EntityType.SPLITTER, EntityType.BRIDGE,
                                         EntityType.ARMOURED_CONVEYOR,
                                         EntityType.MARKER):
                            target_invalid = True

            if target_invalid:
                # Conveyor would point into invalid tile — skip the conveyor
                # build for this ore; chain_to_core will bridge through here
                # instead. Defense still runs: the ore is being harvested
                # regardless of how it chains, and if it's in defend range
                # the kit still applies.
                if DEBUG: print(f"[{self.corner}] chain_start conv target invalid, skip conveyor",
                      file=sys.stderr)
                self._skip_chain_conveyor = True
                # Remember the intended conveyor direction so the defense
                # planner can still face the gunner along it. Without this
                # `_chain_conv_dir` would stay None and the gunner would
                # fall back to the ore→chain_start offset — not pointing
                # at the chain.
                self._chain_conv_dir = conv_dir
                # Fall through to the defense block below — no return, no
                # step transition, no _sentinel_placed flag flip.
            else:
                # Enemy walkable building at chain_start — move onto it and fire.
                # Markers are NOT walkable and fall through to the normal
                # build_conveyor path below; the engine destroys the marker as
                # part of the build action.
                if (cs_entry and cs_entry[1] is not None
                        and cs_entry[3] != self.my_team_cache
                        and cs_entry[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                                            EntityType.SPLITTER, EntityType.BRIDGE,
                                            EntityType.ARMOURED_CONVEYOR)):
                    if my_xy == self.chain_start:
                        if ct.can_fire(cs_pos):
                            ct.fire(cs_pos)
                        return
                    else:
                        d = cardinal_direction_between(my_xy, self.chain_start)
                        if ct.can_move(d):
                            ct.move(d)
                        return

                # Destroy allied road/barrier/marker at chain_start
                if cs_entry and cs_entry[1] is not None and cs_entry[3] == self.my_team_cache:
                    if cs_entry[2] in (EntityType.ROAD, EntityType.BARRIER, EntityType.MARKER):
                        if ct.can_destroy(cs_pos):
                            ct.destroy(cs_pos)
                            self.tile_cache[self.chain_start] = (cs_entry[0], None, None, None)

                if ct.can_build_conveyor(cs_pos, conv_dir):
                    if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                        return  # Wait for reserve to refill
                    ct.build_conveyor(cs_pos, conv_dir)
                    self.building_cache[self.chain_start] = (EntityType.CONVEYOR, conv_dir)
                    self._chain_conv_dir = conv_dir  # Store actual direction for chain_to_core
                    if DEBUG: print(f"[{self.corner}] conveyor at ({self.chain_start[0]},{self.chain_start[1]}) facing {conv_dir.value}",
                          file=sys.stderr)
                else:
                    # Race: another bot may have just built here. If it's allied
                    # transport, abandon this ore (another bot beat us to the spot).
                    race = self._recheck_build_race(ct, self.chain_start)
                    if race == 'chain_complete':
                        if DEBUG: print(f"[{self.corner}] race: allied built at chain_start, abandoning ore",
                              file=sys.stderr)
                        self._abandon_ore()
                        return
                return  # Cooldown consumed — try sentinel next turn

        # --- Step 5a (close to core): enhanced defense kit ---
        # Builds barriers on the remaining cardinals + a gunner or
        # sentinel, then falls through to build_harvester. Each substep
        # consumes one action cooldown so this spans several turns.
        if (not ECONOMY_TEST_MODE and not self._sentinel_placed
                and self._in_defend_range(my_xy)):
            self._claim_ore_defend(ct)
            if not self._sentinel_placed:
                return  # Defend pipeline still in progress

        # --- Conditionally build sentinel on one other cardinal-adjacent tile ---
        if ECONOMY_TEST_MODE:
            self._sentinel_placed = True
        if not self._sentinel_placed:
            if ct.get_action_cooldown() > 0:
                return

            # Pick best tile: prefer empty env over ore, skip walls/buildings
            best_tile = None
            best_score = 999
            for nx, ny in neighbors_4(my_xy[0], my_xy[1]):
                sxy = (nx, ny)
                if sxy == self.chain_start:
                    continue
                if sxy in self.known_walls:
                    continue
                s_entry = self.tile_cache.get(sxy)
                if s_entry and s_entry[0] == Environment.WALL:
                    continue
                # Already has an allied sentinel — done
                if s_entry and s_entry[2] == EntityType.SENTINEL and s_entry[3] == self.my_team_cache:
                    self._sentinel_placed = True
                    best_tile = None
                    break
                # Score: 0=empty, 1=ore, skip if non-destroyable building
                if s_entry and s_entry[1] is not None:
                    if s_entry[3] != self.my_team_cache:
                        continue
                    if s_entry[2] not in (EntityType.ROAD, EntityType.MARKER):
                        continue
                score = 0 if (s_entry is None or s_entry[0] == Environment.EMPTY) else 1
                if score < best_score:
                    best_score = score
                    best_tile = sxy

            if self._sentinel_placed:
                pass  # Already placed, fall through to transition
            elif best_tile is not None and self.tle_recovery_turns == 0:
                # --- Sentinel scan: check if placement is worth the cost ---
                sentinel_pos = xy_to_pos(best_tile)
                harvester_dir = cardinal_direction_between(best_tile, self.target_ore)

                all_dirs = [Direction.NORTH, Direction.NORTHEAST, Direction.EAST,
                            Direction.SOUTHEAST, Direction.SOUTH, Direction.SOUTHWEST,
                            Direction.WEST, Direction.NORTHWEST]

                place_sentinel = False
                chosen_dir = None
                best_enemy_count = 0
                best_enemy_dir = None

                for d in all_dirs:
                    if d == harvester_dir:
                        continue  # Can't face harvester (ammo comes from there)

                    tiles = ct.get_attackable_tiles_from(sentinel_pos, d, EntityType.SENTINEL)

                    # Check attackable tiles using tile_cache (no API calls)
                    found_core = False
                    enemy_count = 0
                    tc = self.tile_cache
                    for t in tiles:
                        txy = (t.x, t.y)
                        entry = tc.get(txy)
                        if entry is None:
                            continue
                        bid, btype, bteam = entry[1], entry[2], entry[3]
                        if bid is not None and bteam != self.my_team_cache:
                            if btype == EntityType.CORE:
                                chosen_dir = d
                                place_sentinel = True
                                found_core = True
                                break
                            # Harvesters / foundries fuel sentinels — don't
                            # count them as valuable destruction targets
                            if btype not in (EntityType.ROAD, EntityType.MARKER,
                                             EntityType.HARVESTER, EntityType.FOUNDRY):
                                enemy_count += 1
                    if found_core:
                        break

                    if enemy_count > best_enemy_count:
                        best_enemy_count = enemy_count
                        best_enemy_dir = d

                if not place_sentinel and best_enemy_count >= SENTINEL_ATTACK_THRESHOLD:
                    chosen_dir = best_enemy_dir
                    place_sentinel = True

                if place_sentinel and chosen_dir is not None:
                    # Destroy road/marker at sentinel tile if needed
                    s_entry = self.tile_cache.get(best_tile)
                    if s_entry and s_entry[1] is not None and s_entry[3] == self.my_team_cache:
                        if s_entry[2] in (EntityType.ROAD, EntityType.MARKER):
                            if ct.can_destroy(sentinel_pos):
                                ct.destroy(sentinel_pos)
                                self.tile_cache[best_tile] = (s_entry[0], None, None, None)
                    if (self._can_spend(ct, GC.SENTINEL_BASE_COST[0])
                            and ct.can_build_sentinel(sentinel_pos, chosen_dir)):
                        ct.build_sentinel(sentinel_pos, chosen_dir)
                        self._sentinel_placed = True
                        if DEBUG: print(f"[{self.corner}] sentinel at ({best_tile[0]},{best_tile[1]}) facing {chosen_dir.value}",
                              file=sys.stderr)
                        return
                # Scan decided not to place, or can't afford — skip
                self._sentinel_placed = True
            else:
                # No valid tile — skip sentinel placement
                self._sentinel_placed = True

        # All builds complete → transition
        self.step = "build_harvester"
        self._harvester_moved = False

    # ------------------------------------------------------------------ #
    #  Step 5c: Claim axionite ore (no defense)                           #
    # ------------------------------------------------------------------ #

    def _step_claim_ore_axionite(self, ct):
        """Axionite variant of claim_ore — no barriers, gunner, or sentinel.
        Only builds the chain-start conveyor, then transitions directly to
        build_harvester. The bot is already standing on the axionite ore.
        """
        my_xy = self._my_xy
        # Commit to an axionite chain for the downstream dispatch.
        self.current_chain_type = 'axionite'

        # Re-check enclosed now that all neighbours are visible.
        if self._is_ore_enclosed(self.target_ore):
            if DEBUG: print(f"[{self.corner}] ax ore ({self.target_ore[0]},{self.target_ore[1]}) enclosed, abandoning",
                  file=sys.stderr)
            self._abandon_ore()
            return

        # Enemy walkable building on ore — fire until destroyed, then road.
        ore_entry = self.tile_cache.get(my_xy)
        if (ore_entry and ore_entry[1] is not None
                and ore_entry[3] != self.my_team_cache
                and ore_entry[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                                     EntityType.SPLITTER, EntityType.BRIDGE,
                                     EntityType.ARMOURED_CONVEYOR)):
            my_pos = xy_to_pos(my_xy)
            if ct.get_action_cooldown() == 0 and ct.can_fire(my_pos):
                ct.fire(my_pos)
            return
        # No building on ore — build a road to claim it.
        if ore_entry and ore_entry[1] is None:
            if ct.get_action_cooldown() == 0:
                ore_pos = xy_to_pos(my_xy)
                if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                        and ct.can_build_road(ore_pos)):
                    ct.build_road(ore_pos)
            return

        # Resource check — wait if we can't afford the harvester + chain.
        if not self._can_afford_harvest(ct):
            return

        # --- Determine chain_start (same logic as titanium claim) ---
        if self.chain_start is None:
            pref_dir = cardinal_direction_between(my_xy, self.core_pos)
            pdx, pdy = DIRECTION_DELTAS[pref_dir]
            pref_xy = (my_xy[0] + pdx, my_xy[1] + pdy)
            pref_entry = self.tile_cache.get(pref_xy)
            pref_blocked = self._chain_start_blocked(pref_xy, pref_entry)

            if not pref_blocked:
                self.chain_start = pref_xy
                self._chain_dir = pref_dir
            else:
                best_alt = None
                best_alt_dist = float('inf')
                best_alt_dir = None
                for d in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
                    if d == pref_dir:
                        continue
                    adx, ady = DIRECTION_DELTAS[d]
                    alt_xy = (my_xy[0] + adx, my_xy[1] + ady)
                    alt_e = self.tile_cache.get(alt_xy)
                    if self._chain_start_blocked(alt_xy, alt_e):
                        continue
                    dist = euclidean_dist_sq(alt_xy, self.core_pos)
                    if dist < best_alt_dist:
                        best_alt_dist = dist
                        best_alt = alt_xy
                        best_alt_dir = d
                if best_alt is not None:
                    self.chain_start = best_alt
                    self._chain_dir = best_alt_dir
                else:
                    # No cardinal chain_start available — abandon this ore.
                    if DEBUG: print(f"[{self.corner}] ax all cardinals walled, blacklisting ore",
                          file=sys.stderr)
                    self._ore_blacklist.add(self.target_ore)
                    self._abandon_ore()
                    return

        # --- Build conveyor at chain_start ---
        cs_entry = self.tile_cache.get(self.chain_start)
        conveyor_exists = (
            cs_entry is not None
            and cs_entry[2] == EntityType.CONVEYOR
            and cs_entry[3] == self.my_team_cache
        )

        if not conveyor_exists:
            if ct.get_action_cooldown() > 0:
                return
            cs_pos = xy_to_pos(self.chain_start)

            mono_path = astar_monotonic(
                self.chain_start, self.core_pos,
                self.tile_cache, self.known_walls,
                map_w=self.map_w, map_h=self.map_h,
                extra_walls=self._allied_barrier_walls(),
                my_team=self.my_team_cache,
            )
            if mono_path is None or len(mono_path) == 0:
                conv_dir = cardinal_direction_between(self.chain_start, self.core_pos)
            else:
                conv_dir = cardinal_direction_between(self.chain_start, mono_path[0])

            # Conveyor target validity check (same as titanium path).
            cdx, cdy = DIRECTION_DELTAS[conv_dir]
            conv_target = (self.chain_start[0] + cdx, self.chain_start[1] + cdy)
            ct_entry = self.tile_cache.get(conv_target)
            target_invalid = False
            if conv_target == self.target_ore:
                target_invalid = True
            elif conv_target in self.known_walls:
                target_invalid = True
            elif self._is_enemy_core_tile(conv_target):
                target_invalid = True
            elif ct_entry is not None:
                if ct_entry[0] == Environment.WALL:
                    target_invalid = True
                elif ct_entry[1] is not None:
                    etype, team = ct_entry[2], ct_entry[3]
                    if team == self.my_team_cache:
                        if etype not in (EntityType.ROAD, EntityType.MARKER,
                                         EntityType.BARRIER):
                            target_invalid = True
                    else:
                        if etype not in (EntityType.ROAD, EntityType.CONVEYOR,
                                         EntityType.SPLITTER, EntityType.BRIDGE,
                                         EntityType.ARMOURED_CONVEYOR,
                                         EntityType.MARKER):
                            target_invalid = True

            if target_invalid:
                if DEBUG: print(f"[{self.corner}] ax chain_start conv target invalid, skipping to chain",
                      file=sys.stderr)
                self.step = "build_harvester"
                self._sentinel_placed = True  # unused for axionite but harmless
                self._harvester_moved = False
                return

            # Enemy walkable building at chain_start — walk onto + fire.
            if (cs_entry and cs_entry[1] is not None
                    and cs_entry[3] != self.my_team_cache
                    and cs_entry[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                                        EntityType.SPLITTER, EntityType.BRIDGE,
                                        EntityType.ARMOURED_CONVEYOR)):
                if my_xy == self.chain_start:
                    if ct.can_fire(cs_pos):
                        ct.fire(cs_pos)
                    return
                else:
                    d = cardinal_direction_between(my_xy, self.chain_start)
                    if ct.can_move(d):
                        ct.move(d)
                    return

            # Destroy allied road/barrier/marker at chain_start.
            if cs_entry and cs_entry[1] is not None and cs_entry[3] == self.my_team_cache:
                if cs_entry[2] in (EntityType.ROAD, EntityType.BARRIER, EntityType.MARKER):
                    if ct.can_destroy(cs_pos):
                        ct.destroy(cs_pos)
                        self.tile_cache[self.chain_start] = (cs_entry[0], None, None, None)

            if ct.can_build_conveyor(cs_pos, conv_dir):
                if not self._has_economy_reserve(ct, GC.CONVEYOR_BASE_COST[0]):
                    return  # wait
                ct.build_conveyor(cs_pos, conv_dir)
                self.building_cache[self.chain_start] = (EntityType.CONVEYOR, conv_dir)
                self._chain_conv_dir = conv_dir
                if DEBUG: print(f"[{self.corner}] ax conveyor at ({self.chain_start[0]},{self.chain_start[1]}) facing {conv_dir.value}",
                      file=sys.stderr)
            else:
                race = self._recheck_build_race(ct, self.chain_start)
                if race == 'chain_complete':
                    if DEBUG: print(f"[{self.corner}] ax race at chain_start, abandoning",
                          file=sys.stderr)
                    self._abandon_ore()
                    return
            return  # cooldown consumed; transition next turn

        # Conveyor is in place — axionite needs NO defense. Transition.
        self._sentinel_placed = True  # skip the defense block on re-entry
        self.step = "build_harvester"
        self._harvester_moved = False

    # ------------------------------------------------------------------ #
    #  Step 5a: Enhanced defense (close to core)                          #
    # ------------------------------------------------------------------ #

    def _claim_ore_defend(self, ct):
        """Build the full defense kit around a close-to-core harvester:
        2 barriers on the remaining cardinals + a gunner (default) or
        sentinel (if the sentinel scan justifies it).

        Bot is standing on the ore. self.chain_start is set and the
        conveyor is already built (caller guarantees both).

        Substeps drive multi-turn execution (each build consumes one
        action cooldown):
            0  plan layout (free)         → 1
            1  build barriers (one/turn)  → 2
            2  build turret               → done
        On completion sets self._sentinel_placed = True so the caller
        falls through to build_harvester.
        """
        my_xy = self._my_xy
        ore_xy = self.target_ore

        # ---- Substep 0: plan layout (no action cooldown consumed) ----
        if self._defend_substep == 0:
            cs = self.chain_start
            if cs is None:
                # Caller invariant violated — fall through to legacy path
                self._sentinel_placed = False
                return

            # Reset the layout buckets in case we re-enter for a fresh ore
            self._defend_barrier_xys = []
            self._defend_turret_xy = None
            self._defend_turret_kind = None
            self._defend_turret_dir = None

            # Cardinal candidates of the ore minus chain_start
            candidates = []
            for nx, ny in neighbors_4(ore_xy[0], ore_xy[1]):
                xy = (nx, ny)
                if xy == cs:
                    continue
                candidates.append(xy)

            def _tile_buildable_for_defense(xy):
                """True if we can place a non-walkable defense building
                (barrier/gunner/sentinel) on this tile after at most a
                free-destroy + walk-fire."""
                if xy in self.known_walls:
                    return False
                e = self.tile_cache.get(xy)
                if e is None:
                    return True  # unknown — assume buildable
                env, bid, etype, team = e
                if env == Environment.WALL:
                    return False
                if bid is None:
                    return True
                if etype == EntityType.MARKER:
                    return True
                if team == self.my_team_cache:
                    # Allied: only roads/markers/barriers are clearable
                    return etype in (EntityType.ROAD, EntityType.MARKER,
                                     EntityType.BARRIER)
                # Enemy walkable — we can clear it by walking on + firing
                return etype in (EntityType.ROAD, EntityType.CONVEYOR,
                                 EntityType.SPLITTER, EntityType.BRIDGE,
                                 EntityType.ARMOURED_CONVEYOR)

            valid_candidates = [xy for xy in candidates
                                if _tile_buildable_for_defense(xy)]

            # ---- Sentinel scan first ----
            sentinel_tile = None
            sentinel_dir = None
            if self.tle_recovery_turns == 0:
                # Try the best candidate (first valid). One scan only —
                # 7-direction get_attackable_tiles_from is expensive.
                for cand in valid_candidates:
                    place, d = self._sentinel_scan_tile(ct, cand, ore_xy)
                    if place:
                        sentinel_tile = cand
                        sentinel_dir = d
                        break

            if sentinel_tile is not None:
                self._defend_turret_xy = sentinel_tile
                self._defend_turret_kind = EntityType.SENTINEL
                self._defend_turret_dir = sentinel_dir
            elif not PLACE_GUNNER:
                # PLACE_GUNNER=False: skip the gunner entirely. No turret
                # tile is reserved, so the perpendicular cardinal that
                # would have been the gunner falls through into the
                # barrier loop below. Sentinel scans are unaffected
                # because they short-circuit the branch above.
                pass
            else:
                # ---- Gunner: pick from the 2 perpendicular cardinals ----
                # Use _chain_dir (ore → chain_start position offset), NOT
                # _chain_conv_dir (conveyor facing, which follows the
                # monotonic path toward core). We want the gunner on
                # either side of where the *conveyor sits relative to
                # the ore*, so it can defend that conveyor tile.
                perp = _PERPENDICULAR.get(self._chain_dir, ())
                gunner_tile = None
                # Prefer non-ore valid perpendiculars first
                ore_fallback = None
                for d in perp:
                    dx, dy = DIRECTION_DELTAS[d]
                    xy = (ore_xy[0] + dx, ore_xy[1] + dy)
                    if xy not in valid_candidates:
                        continue
                    e = self.tile_cache.get(xy)
                    is_ore = e is not None and e[0] in (
                        Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)
                    if is_ore:
                        if ore_fallback is None:
                            ore_fallback = xy
                        continue
                    gunner_tile = xy
                    break
                if gunner_tile is None:
                    gunner_tile = ore_fallback
                if gunner_tile is not None:
                    self._defend_turret_xy = gunner_tile
                    self._defend_turret_kind = EntityType.GUNNER
                    # Face from the gunner tile TOWARD the chain_start
                    # (the conveyor the gunner is defending). Example:
                    # gunner at (11,15), conveyor at (10,16) → face SW.
                    # `direction_between` is 8-directional so the diag
                    # case is natural here.
                    cs = self.chain_start or (
                        gunner_tile[0] + DIRECTION_DELTAS[self._chain_dir][0],
                        gunner_tile[1] + DIRECTION_DELTAS[self._chain_dir][1])
                    self._defend_turret_dir = (
                        direction_between(gunner_tile, cs)
                        or self._chain_conv_dir
                        or self._chain_dir)

            # ---- Barriers: remaining valid cardinals ----
            for xy in valid_candidates:
                if xy == self._defend_turret_xy:
                    continue
                self._defend_barrier_xys.append(xy)

            self._defend_substep = 1
            if DEBUG:
                print(f"[{self.corner}] defend layout: turret="
                      f"{self._defend_turret_xy} kind={self._defend_turret_kind} "
                      f"barriers={self._defend_barrier_xys}",
                      file=sys.stderr)
            return  # No action consumed; resume next call

        # ---- Substep 1: build barriers, one per action cooldown ----
        if self._defend_substep == 1:
            if not self._defend_barrier_xys:
                self._defend_substep = 2
                # Fall through
            else:
                if ct.get_action_cooldown() > 0:
                    return
                bxy = self._defend_barrier_xys[0]
                if not self._build_defense_barrier(ct, bxy):
                    return  # Need another turn (clearing/firing/etc)
                self._defend_barrier_xys.pop(0)
                return

        # ---- Substep 2: build turret ----
        if self._defend_substep == 2:
            if self._defend_turret_xy is None:
                # No turret possible — done
                self._defend_substep = 3
            else:
                if ct.get_action_cooldown() > 0:
                    return
                if not self._build_defense_turret(ct):
                    return
                self._defend_substep = 3
                return

        # ---- Substep 3: done ----
        if self._defend_substep == 3:
            self._sentinel_placed = True
            self._defend_substep = 0
            return

    def _sentinel_scan_tile(self, ct, sentinel_xy, harvester_xy):
        """Run the 7-direction sentinel scan from sentinel_xy. Returns
        (place: bool, direction: Direction or None)."""
        sentinel_pos = xy_to_pos(sentinel_xy)
        harvester_dir = cardinal_direction_between(sentinel_xy, harvester_xy)
        place = False
        chosen = None
        best_count = 0
        best_dir = None
        for d in _ALL_8_DIRS:
            if d == harvester_dir:
                continue
            tiles = ct.get_attackable_tiles_from(
                sentinel_pos, d, EntityType.SENTINEL)
            found_core = False
            count = 0
            tc = self.tile_cache
            for t in tiles:
                txy = (t.x, t.y)
                e = tc.get(txy)
                if e is None:
                    continue
                bid, btype, bteam = e[1], e[2], e[3]
                if bid is not None and bteam != self.my_team_cache:
                    if btype == EntityType.CORE:
                        chosen = d
                        place = True
                        found_core = True
                        break
                    if btype not in (EntityType.ROAD, EntityType.MARKER,
                                     EntityType.HARVESTER, EntityType.FOUNDRY):
                        count += 1
            if found_core:
                break
            if count > best_count:
                best_count = count
                best_dir = d
        if not place and best_count >= SENTINEL_ATTACK_THRESHOLD:
            chosen = best_dir
            place = True
        return (place, chosen)

    def _build_defense_barrier(self, ct, bxy):
        """Place a barrier at bxy from the ore tile. Returns True if the
        action was issued (cooldown consumed) or the tile no longer needs
        a barrier; False if we need another turn (e.g. firing on enemy)."""
        b_pos = xy_to_pos(bxy)
        e = self.tile_cache.get(bxy)
        # Already an allied barrier — done
        if (e is not None and e[1] is not None
                and e[2] == EntityType.BARRIER
                and e[3] == self.my_team_cache):
            return True
        # Enemy walkable — walk onto + fire (multi-turn)
        if (e is not None and e[1] is not None
                and e[3] != self.my_team_cache
                and e[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                             EntityType.SPLITTER, EntityType.BRIDGE,
                             EntityType.ARMOURED_CONVEYOR)):
            my_xy = self._my_xy
            if my_xy == bxy:
                if ct.can_fire(b_pos):
                    ct.fire(b_pos)
                return False
            d = cardinal_direction_between(my_xy, bxy)
            if ct.can_move(d):
                ct.move(d)
            return False
        # Allied road/marker — destroy free, then build barrier
        if (e is not None and e[1] is not None
                and e[3] == self.my_team_cache
                and e[2] in (EntityType.ROAD, EntityType.MARKER)):
            if ct.can_destroy(b_pos):
                ct.destroy(b_pos)
                self.tile_cache[bxy] = (e[0], None, None, None)
        # Build the barrier
        if not self._can_spend(ct, GC.BARRIER_BASE_COST[0]):
            return False  # Hard titanium floor — wait for reserve
        if ct.can_build_barrier(b_pos):
            ct.build_barrier(b_pos)
            if DEBUG:
                print(f"[{self.corner}] defend barrier at ({bxy[0]},{bxy[1]})",
                      file=sys.stderr)
            return True
        return True  # cooldown will tick; treat as consumed to avoid loops

    def _build_defense_turret(self, ct):
        """Place gunner/sentinel at self._defend_turret_xy facing
        self._defend_turret_dir. Returns True on success or skip; False
        if still working on it."""
        txy = self._defend_turret_xy
        kind = self._defend_turret_kind
        tdir = self._defend_turret_dir
        if txy is None or kind is None or tdir is None:
            return True
        t_pos = xy_to_pos(txy)
        e = self.tile_cache.get(txy)
        # Already an allied turret of any kind — done
        if (e is not None and e[1] is not None
                and e[3] == self.my_team_cache
                and e[2] in (EntityType.GUNNER, EntityType.SENTINEL)):
            return True
        # Enemy walkable — walk onto + fire
        if (e is not None and e[1] is not None
                and e[3] != self.my_team_cache
                and e[2] in (EntityType.ROAD, EntityType.CONVEYOR,
                             EntityType.SPLITTER, EntityType.BRIDGE,
                             EntityType.ARMOURED_CONVEYOR)):
            my_xy = self._my_xy
            if my_xy == txy:
                if ct.can_fire(t_pos):
                    ct.fire(t_pos)
                return False
            d = cardinal_direction_between(my_xy, txy)
            if ct.can_move(d):
                ct.move(d)
            return False
        # Allied road/marker/barrier in the way — destroy free
        if (e is not None and e[1] is not None
                and e[3] == self.my_team_cache
                and e[2] in (EntityType.ROAD, EntityType.MARKER,
                             EntityType.BARRIER)):
            if ct.can_destroy(t_pos):
                ct.destroy(t_pos)
                self.tile_cache[txy] = (e[0], None, None, None)
        if kind == EntityType.SENTINEL:
            if not self._can_spend(ct, GC.SENTINEL_BASE_COST[0]):
                return False  # Hard titanium floor
            if ct.can_build_sentinel(t_pos, tdir):
                ct.build_sentinel(t_pos, tdir)
                if DEBUG:
                    print(f"[{self.corner}] defend sentinel at ({txy[0]},{txy[1]}) facing {tdir.value}",
                          file=sys.stderr)
                return True
        else:  # GUNNER
            if not self._can_spend(ct, GC.GUNNER_BASE_COST[0]):
                return False  # Hard titanium floor
            if ct.can_build_gunner(t_pos, tdir):
                ct.build_gunner(t_pos, tdir)
                if DEBUG:
                    print(f"[{self.corner}] defend gunner at ({txy[0]},{txy[1]}) facing {tdir.value}",
                          file=sys.stderr)
                return True
        return True  # treat as consumed to advance

    # ------------------------------------------------------------------ #
    #  Step: build_harvester                                              #
    # ------------------------------------------------------------------ #

    def _step_build_harvester(self, ct):
        """1. Move onto the conveyor (chain_start)
        2. Destroy road on ore (free)
        3. Build harvester on cleared ore
        4. Scan for nearby ores, store positions, transition
        """
        my_xy = self._my_xy

        # Same heal-in-place guard as _step_chain_to_core: don't burn
        # the action cooldown on building a harvester / road if a
        # nearby ally is taking damage and we're in range to heal.
        if self._economy_check_heal_action_radius(ct, my_xy) != 'none':
            return
        # If another bot got to our target ore first while we were
        # busy / diverting to heal, the harvester is already there —
        # no point continuing this build, just resume exploring.
        if self.target_ore is not None:
            te = self.tile_cache.get(self.target_ore)
            if (te and te[1] is not None
                    and te[2] == EntityType.HARVESTER):
                if DEBUG:
                    print(f"[{self.corner}] target ore ({self.target_ore[0]},"
                          f"{self.target_ore[1]}) already has harvester, "
                          f"explore", file=sys.stderr)
                self.target_ore = None
                self.chain_start = None
                self._sentinel_placed = False
                self._harvester_moved = False
                self.step = "explore"
                return
        # Short-detour heal divert (chain pieces only, ≤ 4 tiles).
        if self._economy_try_heal_chain_divert(ct, my_xy):
            return

        if self.chain_start is None or self.target_ore is None:
            self.step = "explore"
            return

        ore_pos = xy_to_pos(self.target_ore)

        # --- Move + destroy road + build harvester in a single turn ---
        # Move cooldown and action cooldown are separate, and destroy is
        # free (no cooldown). So: move (uses move cd) → destroy road on
        # ore (free) → build harvester (uses action cd), all in one turn.
        # This prevents other bots from claiming the ore in a gap.
        if my_xy != self.chain_start:
            if ct.get_action_cooldown() > 0:
                return  # wait so we can build on the same turn we move
            if not self._has_economy_reserve(ct, GC.HARVESTER_BASE_COST[0]):
                return  # wait for reserve so we can build immediately
            direction = cardinal_direction_between(my_xy, self.chain_start)
            if ct.can_move(direction):
                ct.move(direction)
                self._harvester_move_stuck = 0
                # Fall through to destroy + build below (same turn)
            else:
                # Can't move — chain_start blocked (another bot?) or not walkable
                self._harvester_move_stuck = getattr(self, '_harvester_move_stuck', 0) + 1
                if ct.get_action_cooldown() == 0:
                    cs_entry = self.tile_cache.get(self.chain_start)
                    cs_pos = xy_to_pos(self.chain_start)
                    # Allied barrier/road/marker on chain_start — destroy it
                    # so we can build a road and step onto the tile. This
                    # case fires when the chain_start picker chose an
                    # allied barrier (e.g. a defense barrier from a
                    # neighbouring harvester) AND the claim-ore step's
                    # `target_invalid` shortcut skipped the conveyor build,
                    # leaving the barrier in place. Without this branch the
                    # bot loops can_move-False for 10 turns then abandons.
                    if (cs_entry and cs_entry[1] is not None
                            and cs_entry[3] == self.my_team_cache
                            and cs_entry[2] in (EntityType.BARRIER,
                                                EntityType.ROAD,
                                                EntityType.MARKER)):
                        if ct.can_destroy(cs_pos):
                            ct.destroy(cs_pos)
                            self.tile_cache[self.chain_start] = (
                                cs_entry[0], None, None, None)
                            self._harvester_move_stuck = 0
                            cs_entry = self.tile_cache.get(self.chain_start)
                    # Empty tile (originally or post-destroy) — drop a road
                    # so the bot can walk on it next turn.
                    if cs_entry is None or cs_entry[1] is None:
                        if (self._can_spend(ct, GC.ROAD_BASE_COST[0])
                                and ct.can_build_road(cs_pos)):
                            ct.build_road(cs_pos)
                            self._harvester_move_stuck = 0
                if self._harvester_move_stuck >= 10:
                    if DEBUG: print(f"[{self.corner}] stuck moving to chain_start, abandoning ore",
                          file=sys.stderr)
                    self._harvester_move_stuck = 0
                    self._abandon_ore()
                return

        if ct.get_action_cooldown() > 0:
            return  # already on chain_start but action cd not ready

        if not self._has_economy_reserve(ct, GC.HARVESTER_BASE_COST[0]):
            return

        # Destroy road on ore (free) then build harvester (action cd)
        ore_entry = self.tile_cache.get(self.target_ore)
        if ore_entry and ore_entry[1] is not None:
            if ore_entry[3] == self.my_team_cache and ore_entry[2] == EntityType.ROAD:
                if ct.can_destroy(ore_pos):
                    ct.destroy(ore_pos)
                    self.tile_cache[self.target_ore] = (ore_entry[0], None, None, None)

        if ct.can_build_harvester(ore_pos):
            ct.build_harvester(ore_pos)
            self.harvesters_built += 1
            if DEBUG: print(f"[{self.corner}] harvester #{self.harvesters_built} at ({self.target_ore[0]},{self.target_ore[1]})",
                  file=sys.stderr)

            # --- Sub-step 4: scan visible tiles for nearby ores ---
            # Only store ores of the same type as the current chain.
            # Titanium chains look for more titanium; axionite chains
            # look for more axionite.
            want_env = (Environment.ORE_AXIONITE
                        if self.current_chain_type == 'axionite'
                        else Environment.ORE_TITANIUM)
            for xy in self._last_vision_set:
                entry = self.tile_cache.get(xy)
                if entry and entry[0] == want_env and xy != self.target_ore:
                    if not self._is_ore_enclosed(xy) and not self._is_ore_claimed(xy, my_xy):
                        if xy not in self.stored_ores:
                            self.stored_ores.append(xy)

            # Reset claim state, transition to chain_to_core
            ore_xy = self.target_ore
            self.target_ore = None
            self._harvester_moved = False
            self._sentinel_placed = False
            # chain_start stays set — chaining.py reads it
            self.step = "chain_to_core"
            self.path = None
            self.path_index = 0
            if DEBUG: print(f"[{self.corner}] -> chain_to_core from ({self.chain_start[0]},{self.chain_start[1]})",
                  file=sys.stderr)
        else:
            # Can't build — retry once next turn before abandoning
            retries = getattr(self, '_harvester_retries', 0) + 1
            self._harvester_retries = retries
            if retries > 1:
                if DEBUG: print(f"[{self.corner}] can't build harvester at ({self.target_ore[0]},{self.target_ore[1]}), abandoning",
                      file=sys.stderr)
                self._harvester_retries = 0
                self._abandon_ore()

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _is_ore_enclosed(self, ore_xy):
        """Check if ore is completely surrounded by ores OR completely surrounded
        by walls/non-walkable buildings. Mixed surroundings = not enclosed."""
        tc = self.tile_cache
        kw = self.known_walls
        ox, oy = ore_xy
        all_ore = True
        all_wall = True
        for nx, ny in ((ox, oy-1), (ox+1, oy), (ox, oy+1), (ox-1, oy)):
            nxy = (nx, ny)
            ne = tc.get(nxy)
            if ne is None:
                return False  # Unknown — assume accessible
            env = ne[0]
            is_ore = env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE)
            is_wall = (env == Environment.WALL or nxy in kw
                       or (ne[1] is not None and ne[2] not in (
                           EntityType.ROAD, EntityType.CONVEYOR, EntityType.SPLITTER,
                           EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
                           EntityType.MARKER, EntityType.CORE)))
            if not is_ore:
                all_ore = False
            if not is_wall:
                all_wall = False
            if not all_ore and not all_wall:
                return False  # Mixed — not enclosed
        return all_ore or all_wall

    def _is_ore_claimed(self, ore_xy, my_xy):
        """Check if an ore tile is already claimed by another allied bot/building."""
        entry = self.tile_cache.get(ore_xy)
        if entry:
            env, bid, etype, team = entry
            if bid is not None and team == self.my_team_cache:
                # Allied harvester, conveyor, splitter, bridge ON the ore = taken
                if etype in (EntityType.HARVESTER, EntityType.CONVEYOR,
                             EntityType.SPLITTER, EntityType.BRIDGE,
                             EntityType.ARMOURED_CONVEYOR):
                    return True
                # Allied road + another allied bot on it = claimed
                if etype == EntityType.ROAD:
                    if ore_xy in self.bot_pos_cache:
                        bot_uid, bot_team = self.bot_pos_cache[ore_xy]
                        if bot_team == self.my_team_cache and ore_xy != my_xy:
                            return True
        return False

    def _enemy_camping_ore(self, ore_xy):
        """True if an enemy bot has been on ore_xy for 3+ consecutive turns.
        Uses _bot_stationary dict maintained in _scan_turn."""
        if ore_xy not in self.bot_pos_cache:
            return False
        _, team = self.bot_pos_cache[ore_xy]
        if team == self.my_team_cache:
            return False
        return self._bot_stationary.get(ore_xy, 0) >= 3

    def _abandon_ore(self):
        """Clear ore targeting state, return to explore."""
        if self.target_ore:
            self._ore_blacklist.add(self.target_ore)
        self.target_ore = None
        self._from_stored_ore = False
        self.chain_start = None
        self._chain_dir = None
        self.path = None
        self.path_index = 0
        self._harvester_moved = False
        self._sentinel_placed = False
        self._skip_chain_conveyor = False
        self._observe_skip = None
        self._chain_conv_dir = None
        # Reset Step 5a defend pipeline state
        self._defend_substep = 0
        self._defend_barrier_xys = []
        self._defend_turret_xy = None
        self._defend_turret_kind = None
        self._defend_turret_dir = None
        self.step = "explore"

    # ------------------------------------------------------------------ #
    #  Steal enemy harvester — Step 4b                                    #
    # ------------------------------------------------------------------ #

    def _can_afford_sentinel(self, ct):
        """True if we can afford a sentinel without breaking the
        MIN_TITANIUM hard floor at current scale."""
        return self._can_spend(ct, GC.SENTINEL_BASE_COST[0])

    def _is_already_stolen(self, ore_xy):
        """True if the enemy harvester at ore_xy already has an allied
        sentinel / conveyor / bridge cardinally adjacent. Zero API calls."""
        tc = self.tile_cache
        my_team = self.my_team_cache
        ox, oy = ore_xy
        for nx, ny in ((ox, oy - 1), (ox + 1, oy), (ox, oy + 1), (ox - 1, oy)):
            info = tc.get((nx, ny))
            if info is None:
                continue
            if info[3] == my_team and info[2] in _STEAL_STOLEN_MARKERS:
                return True
        return False

    def _has_available_steal_tile(self, ore_xy):
        """True if at least one cardinal neighbor of ore_xy is empty or an
        enemy walkable building (i.e. a tile we could build on after clearing)."""
        return bool(self._get_available_steal_tiles(ore_xy))

    def _get_available_steal_tiles(self, ore_xy):
        """Return cardinally adjacent (x,y) tiles suitable for sentinel
        placement. Empty tiles are listed first (no clearing cost) but
        enemy walkable tiles are included too so the 7-direction scan
        can consider all of them and pick the best firing angle —
        empty tiles remain preferred in tiebreaks by the picker."""
        tc = self.tile_cache
        kw = self.known_walls
        my_team = self.my_team_cache
        empty = []
        enemy_walkable = []
        ox, oy = ore_xy
        for nx, ny in ((ox, oy - 1), (ox + 1, oy), (ox, oy + 1), (ox - 1, oy)):
            nxy = (nx, ny)
            if nxy in kw:
                continue
            info = tc.get(nxy)
            if info is None:
                continue
            env, bid, etype, team = info
            if env == Environment.WALL:
                continue
            if bid is None:
                empty.append(nxy)
                continue
            if team != my_team and etype in _ENEMY_WALKABLE_STEAL:
                enemy_walkable.append(nxy)
        return empty + enemy_walkable

    def _abandon_steal(self, reason="?"):
        """Clear steal state and return to explore."""
        if DEBUG:
            print(f"[{self.corner}] abandon steal {self.steal_target} ({reason})",
                  file=sys.stderr)
        if self.steal_target is not None:
            self._ore_blacklist.add(self.steal_target)
        self.steal_target = None
        self.sentinel_tile = None
        self.sentinel_dir = None
        self.steal_chain_tile = None
        self._steal_build_retries = 0
        self._steal_afford_waits = 0
        self.path = None
        self.path_index = 0
        self.step = "explore"

    def _pick_sentinel_tile_and_dir(self, ct, available, harvester_xy):
        """Run the 7-direction sentinel scan on every available tile and
        return (best_tile, best_dir). Enemy core found → return immediately.

        Tiebreak order: higher enemy-building count → empty tile over
        enemy-walkable (no clearing cost) → closer to the bot."""
        my_xy = self._my_xy
        tc = self.tile_cache
        my_team = self.my_team_cache
        enemy_core = self.enemy_core_pos

        def _is_empty(xy):
            e = tc.get(xy)
            return e is not None and e[1] is None

        best_tile = None
        best_dir = None
        best_count = -1  # allow 0-count wins (we always want to place a sentinel)
        best_is_empty = False

        for tile_xy in available:
            harv_dir = cardinal_direction_between(tile_xy, harvester_xy)
            tile_pos = xy_to_pos(tile_xy)
            tile_best_count = -1
            tile_best_dir = None
            found_core = False

            for d in _ALL_8_DIRS:
                if d == harv_dir:
                    continue  # Can't face harvester (ammo source)

                tiles = ct.get_attackable_tiles_from(tile_pos, d, EntityType.SENTINEL)

                count = 0
                for t in tiles:
                    txy = (t.x, t.y)
                    if enemy_core is not None and txy == enemy_core:
                        tile_best_dir = d
                        tile_best_count = 999
                        found_core = True
                        break
                    info = tc.get(txy)
                    if info is None:
                        continue
                    bid, etype, team = info[1], info[2], info[3]
                    if bid is not None and team != my_team:
                        # Harvesters / foundries fuel sentinels — ignore
                        if etype not in (EntityType.ROAD, EntityType.MARKER,
                                         EntityType.HARVESTER, EntityType.FOUNDRY):
                            count += 1
                if found_core:
                    break
                if count > tile_best_count:
                    tile_best_count = count
                    tile_best_dir = d

            if found_core:
                return (tile_xy, tile_best_dir)

            tile_is_empty = _is_empty(tile_xy)
            take_this = False
            if tile_best_count > best_count:
                take_this = True
            elif tile_best_count == best_count and best_tile is not None:
                # Tie on enemy count — prefer empty tile (no clearing
                # cost), then closer to the bot.
                if tile_is_empty and not best_is_empty:
                    take_this = True
                elif tile_is_empty == best_is_empty:
                    if (euclidean_dist_sq(tile_xy, my_xy)
                            < euclidean_dist_sq(best_tile, my_xy)):
                        take_this = True
            if take_this:
                best_count = tile_best_count
                best_dir = tile_best_dir
                best_tile = tile_xy
                best_is_empty = tile_is_empty

        return (best_tile, best_dir)

    def _step_steal_harvester(self, ct):
        """Place a sentinel adjacent to an enemy harvester on titanium ore,
        then optionally chain the harvester's output to our core."""
        my_xy = self._my_xy

        # --- Post-sentinel walking phase ---
        # After the sentinel is built we pick a cardinally-adjacent
        # chain_tile and stash it on steal_chain_tile. The bot then
        # walks to that exact tile before handing off to chain_to_core;
        # otherwise chain_to_core would try to run chain_start_fix from
        # wherever we're currently standing (possibly 2+ tiles away)
        # and the chain ends up with a gap.
        if self.steal_chain_tile is not None:
            chain_tile = self.steal_chain_tile
            if my_xy == chain_tile:
                self.chain_start = chain_tile
                self._chain_start_fixed = False
                self._chain_built = {chain_tile}
                self._chain_conv_dir = None
                self.chain_path = None
                self.chain_index = 0
                self.steal_chain_tile = None
                self.path = None
                self.path_index = 0
                self.step = "chain_to_core"
                if DEBUG:
                    print(f"[{self.corner}] steal → chain_to_core at ({chain_tile[0]},{chain_tile[1]})",
                          file=sys.stderr)
                return
            if self.path is None or self.path_index >= len(self.path):
                self.path = self._compute_path(my_xy, chain_tile)
                self.path_index = 0
                if self.path is None:
                    if DEBUG:
                        print(f"[{self.corner}] steal: chain walk lost path — back to explore",
                              file=sys.stderr)
                    self.steal_chain_tile = None
                    self.chain_start = None
                    self._chain_built = None
                    self._chain_conv_dir = None
                    self.path = None
                    self.path_index = 0
                    self.step = "explore"
                    return
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
            return

        hxy = self.steal_target

        if hxy is None:
            self._abandon_steal("hxy_none")
            return

        # Validate: harvester still there and still enemy
        entry = self.tile_cache.get(hxy)
        if entry is not None and hxy in self._last_vision_set:
            env, bid, etype, team = entry
            if (env != Environment.ORE_TITANIUM or bid is None
                    or etype != EntityType.HARVESTER
                    or team == self.my_team_cache):
                if DEBUG:
                    print(f"[{self.corner}] steal target ({hxy[0]},{hxy[1]}) invalidated",
                          file=sys.stderr)
                self._abandon_steal("invalidated")
                return
            if self._is_already_stolen(hxy):
                if DEBUG:
                    print(f"[{self.corner}] steal target ({hxy[0]},{hxy[1]}) already stolen",
                          file=sys.stderr)
                self._abandon_steal("already_stolen")
                return

        # --- Substep 1: pick best sentinel tile + direction ---
        if self.sentinel_tile is None:
            if self.tle_recovery_turns > 0:
                return  # Sentinel scan is FFI-heavy — defer
            available = self._get_available_steal_tiles(hxy)
            if not available:
                self._abandon_steal("no_available")
                return
            tile, direction = self._pick_sentinel_tile_and_dir(ct, available, hxy)
            if tile is None or direction is None:
                self._abandon_steal("scan_none")
                return
            self.sentinel_tile = tile
            self.sentinel_dir = direction
            self.path = None
            self.path_index = 0
            if DEBUG:
                print(f"[{self.corner}] steal sentinel tile ({tile[0]},{tile[1]}) dir {direction.value}",
                      file=sys.stderr)

        sxy = self.sentinel_tile
        s_entry = self.tile_cache.get(sxy)

        # Re-validate: sentinel_tile may have been claimed by an allied
        # building (our own chain, another bot) since we picked it. If it
        # now holds a non-destroyable allied building, re-pick next turn.
        if (s_entry is not None and s_entry[1] is not None
                and s_entry[3] == self.my_team_cache
                and s_entry[2] not in (EntityType.ROAD, EntityType.MARKER)):
            # Try re-picking from the currently-available tiles (excluding
            # this one). If none, abandon.
            available = [t for t in self._get_available_steal_tiles(hxy)
                         if t != sxy]
            if not available:
                self._abandon_steal("sentinel_tile_lost")
                return
            tile, direction = self._pick_sentinel_tile_and_dir(ct, available, hxy)
            if tile is None or direction is None:
                self._abandon_steal("repick_none")
                return
            self.sentinel_tile = tile
            self.sentinel_dir = direction
            self.path = None
            self.path_index = 0
            if DEBUG:
                print(f"[{self.corner}] steal RE-pick sentinel ({tile[0]},{tile[1]}) dir {direction.value}",
                      file=sys.stderr)
            sxy = self.sentinel_tile
            s_entry = self.tile_cache.get(sxy)

        # --- Substep 2: clear enemy walkable building on sentinel_tile ---
        if (s_entry is not None and s_entry[1] is not None
                and s_entry[3] != self.my_team_cache
                and s_entry[2] in _ENEMY_WALKABLE_STEAL):
            if my_xy == sxy:
                if ct.get_action_cooldown() == 0:
                    s_pos = xy_to_pos(sxy)
                    if ct.can_fire(s_pos):
                        ct.fire(s_pos)
                return
            # Walk toward sentinel tile (stand on it to attack)
            if self.path is None or self.path_index >= len(self.path):
                self.path = self._compute_path(my_xy, sxy)
                self.path_index = 0
                if self.path is None:
                    self._abandon_steal("clear_nopath")
                    return
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
            return

        # If the sentinel tile now has a building we can't destroy (enemy
        # non-walkable, or any allied building other than road/marker), the
        # tile is no longer usable — abandon. (Allied roads/markers get
        # destroyed for free below.)
        if (s_entry is not None and s_entry[1] is not None
                and not (s_entry[3] == self.my_team_cache
                         and s_entry[2] in (EntityType.ROAD, EntityType.MARKER))):
            self._abandon_steal("blocked_building")
            return

        # --- Substep 3: position adjacent to sentinel_tile and build it ---
        dsq = euclidean_dist_sq(my_xy, sxy)
        if my_xy == sxy or dsq > GC.ACTION_RADIUS_SQ:
            # Need to be within action radius but NOT on the tile itself.
            # Pathfind to any walkable 8-neighbor of sxy.
            if self.path is None or self.path_index >= len(self.path):
                goal = self._steal_find_adjacent(my_xy, sxy)
                if goal is None:
                    self._abandon_steal("no_adj_goal")
                    return
                self.path = self._compute_path(my_xy, goal)
                self.path_index = 0
                if self.path is None:
                    self._abandon_steal("adj_nopath")
                    return
            result = self._follow_path(ct)
            if result == 'blocked':
                self.path = None
                self.path_index = 0
            return

        # In range and not on the tile — build sentinel
        if ct.get_action_cooldown() > 0:
            return
        # Wait for titanium rather than bailing — passive income will catch up
        if not self._can_afford_sentinel(ct):
            waits = getattr(self, '_steal_afford_waits', 0) + 1
            self._steal_afford_waits = waits
            if waits > 20:
                self._steal_afford_waits = 0
                self._abandon_steal("cannot_afford")
            return
        self._steal_afford_waits = 0
        s_pos = xy_to_pos(sxy)
        # If an allied road/marker sits on sxy, destroy first (free)
        if (s_entry is not None and s_entry[1] is not None
                and s_entry[3] == self.my_team_cache
                and s_entry[2] in (EntityType.ROAD, EntityType.MARKER)):
            if ct.can_destroy(s_pos):
                ct.destroy(s_pos)
                self.tile_cache[sxy] = (s_entry[0], None, None, None)
                return  # cooldown consumed, build next turn

        if ct.can_build_sentinel(s_pos, self.sentinel_dir):
            ct.build_sentinel(s_pos, self.sentinel_dir)
            if DEBUG:
                print(f"[{self.corner}] STOLE harvester at ({hxy[0]},{hxy[1]}) sentinel ({sxy[0]},{sxy[1]}) dir {self.sentinel_dir.value}",
                      file=sys.stderr)
            # --- Substep 4: check for chain opportunity ---
            remaining = [t for t in self._get_available_steal_tiles(hxy) if t != sxy]
            # Pick the first candidate (closest-to-core order) we can
            # actually pathfind to from our current position. If none
            # are reachable, drop the chain attempt — the sentinel is
            # already doing the useful work (firing at enemy infra),
            # and trying to walk through an unreachable maze just
            # stalls the bot for the rest of the game.
            if self.core_pos is not None:
                remaining.sort(key=lambda t: euclidean_dist_sq(t, self.core_pos))
            chain_tile = None
            for cand in remaining:
                probe = self._compute_path(my_xy, cand)
                if probe is not None:
                    chain_tile = cand
                    break
            if chain_tile is None:
                if DEBUG:
                    print(f"[{self.corner}] steal: no pathable chain_start — "
                          f"dropping chain, back to explore",
                          file=sys.stderr)
                self.steal_target = None
                self.sentinel_tile = None
                self.sentinel_dir = None
                self.path = None
                self.path_index = 0
                self.step = "explore"
                return
            # Stash chain_tile and let the post-sentinel walking phase
            # at the top of this handler walk the bot to it next turn.
            # Reuse the probe path we just computed so we don't burn
            # another A* call.
            self.steal_chain_tile = chain_tile
            self.steal_target = None
            self.sentinel_tile = None
            self.sentinel_dir = None
            self.path = probe
            self.path_index = 0
            if DEBUG:
                print(f"[{self.corner}] steal walk → chain_tile ({chain_tile[0]},{chain_tile[1]})",
                      file=sys.stderr)
            return
        # Can't build — something changed; try again next turn, bail after retry
        retries = getattr(self, '_steal_build_retries', 0) + 1
        self._steal_build_retries = retries
        if retries > 2:
            self._steal_build_retries = 0
            self._abandon_steal("build_retry")

    def _steal_find_adjacent(self, my_xy, target_xy):
        """Find the closest walkable 8-neighbor of target_xy (not target_xy itself)."""
        tc = self.tile_cache
        kw = self.known_walls
        best = None
        best_dsq = 999999
        tx, ty = target_xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxy = (tx + dx, ty + dy)
                if nxy in kw:
                    continue
                info = tc.get(nxy)
                if info and info[0] == Environment.WALL:
                    continue
                # Skip non-walkable buildings — bots can't stand on markers
                # either, so the steal sentinel tile can't be a marker.
                if info and info[1] is not None:
                    etype = info[2]
                    if etype not in (EntityType.ROAD, EntityType.CONVEYOR,
                                     EntityType.SPLITTER, EntityType.BRIDGE,
                                     EntityType.ARMOURED_CONVEYOR):
                        continue
                dsq = euclidean_dist_sq(my_xy, nxy)
                if dsq < best_dsq:
                    best_dsq = dsq
                    best = nxy
        return best
