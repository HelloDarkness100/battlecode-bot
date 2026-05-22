import sys
from cambc import EntityType, Environment, Position
from utils import mirror_xy
from constants import DEBUG

_SYM_ORDER = ['diag', 'vert', 'horiz']

# Allied non-walkable defensive buildings. Tracked in
# self.allied_blockers so chain planners inject them as walls in
# O(|blockers|) instead of scanning the whole tile_cache each call.
_BLOCKER_TYPES = frozenset({
    EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH,
    EntityType.LAUNCHER, EntityType.HARVESTER, EntityType.FOUNDRY,
})


class ScanningMixin:
    """Populates tile_cache, bot_pos_cache, bridge_target_cache each turn.

    Uses batch queries and only calls get_tile_env() on newly visible tiles.
    """

    def _scan_turn(self, ct):
        """Main per-turn scan. Populates all caches with minimal API calls.
        Sets self._my_xy for use by run_builder (avoids duplicate get_position)."""
        my_pos = ct.get_position()
        my_xy = (my_pos.x, my_pos.y)
        self._my_xy = my_xy  # Shared with run_builder — one get_position() call total
        moved = (my_xy != self._last_pos)

        # --- Step 1: get visible tile positions (reuse if stationary) ---
        if moved or self._last_tiles is None:
            tiles = ct.get_nearby_tiles()
            # Convert to tuples ONCE — avoid repeated .x/.y FFI access
            tile_xys = [(t.x, t.y) for t in tiles]
            self._last_tiles = tile_xys
            self._last_pos = my_xy
        else:
            tile_xys = self._last_tiles

        # --- Step 2: get_tile_env only on NEWLY VISIBLE tiles ---
        if moved:
            current_vision = set(tile_xys)
            newly_visible = current_vision - self._last_vision_set
            self._last_vision_set = current_vision

            sym = self._symmetry
            mw = self.map_w
            mh = self.map_h
            mirror_src = getattr(self, '_mirror_src', set())  # tiles added by mirroring

            for xy in newly_visible:
                # Real-observation walls never change — short-circuit.
                # Mirrored walls fall through so their prediction gets
                # validated against the real env below.
                if xy in self.known_walls and xy not in mirror_src:
                    if xy not in self.tile_cache:
                        self.tile_cache[xy] = (Environment.WALL, None, None, None)
                        self._update_frontier(xy)
                    continue

                cached = self.tile_cache.get(xy)
                if cached is not None and xy not in mirror_src:
                    # Real observation from a previous turn — skip
                    continue

                env = ct.get_tile_env(Position(xy[0], xy[1]))

                # Validate mirrored prediction: any env mismatch (WALL vs
                # EMPTY, EMPTY vs ORE_TITANIUM, ORE_TITANIUM vs ORE_AXIONITE,
                # etc.) is enough to reject the current symmetry.
                if cached is not None and xy in mirror_src:
                    mirror_src.discard(xy)  # Now confirmed by real observation
                    if cached[0] != env:
                        # Symmetry was WRONG — purge all mirrored tiles and retry
                        sym = self._handle_bad_symmetry(mirror_src)
                        mirror_src = getattr(self, '_mirror_src', set())

                if env == Environment.WALL:
                    self.known_walls.add(xy)
                self.tile_cache[xy] = (env, None, None, None)
                self._update_frontier(xy)

                # Mirror: cache the symmetric tile's terrain for free
                if sym is not None:
                    mxy = mirror_xy(xy, sym, mw, mh)
                    if mxy not in self.tile_cache or mxy in mirror_src:
                        if env == Environment.WALL:
                            self.known_walls.add(mxy)
                        self.tile_cache[mxy] = (env, None, None, None)
                        mirror_src.add(mxy)
                        self._update_frontier(mxy)

            self._mirror_src = mirror_src

        # --- Step 3: batch building query ---
        buildings = ct.get_nearby_buildings()
        building_positions = set()

        for bid in buildings:
            bpos = ct.get_position(bid)
            bxy = (bpos.x, bpos.y)
            building_positions.add(bxy)
            btype = ct.get_entity_type(bid)
            bteam = ct.get_team(bid)

            old = self.tile_cache.get(bxy)
            env = old[0] if old else Environment.EMPTY

            self.tile_cache[bxy] = (env, bid, btype, bteam)

            if btype == EntityType.BRIDGE and bteam == self.my_team_cache:
                tgt = ct.get_bridge_target(bid)
                self.bridge_target_cache[bxy] = (tgt.x, tgt.y)

            # Track enemy core center (only the center is in tile_cache —
            # we need to remember it to recognise the full 3x3 footprint)
            if btype == EntityType.CORE and bteam != self.my_team_cache:
                self.enemy_core_pos = bxy

            # --- Pass A: update building_cache for allied infrastructure ---
            if bteam == self.my_team_cache:
                if btype in (EntityType.CONVEYOR, EntityType.SPLITTER,
                             EntityType.ARMOURED_CONVEYOR):
                    cached = self.building_cache.get(bxy)
                    if cached is None or cached[0] != btype:
                        direction = ct.get_direction(bid)
                        self.building_cache[bxy] = (btype, direction)
                elif btype == EntityType.BRIDGE:
                    target = self.bridge_target_cache.get(bxy)
                    self.building_cache[bxy] = (btype, target)
                elif btype in (EntityType.HARVESTER, EntityType.FOUNDRY):
                    self.building_cache[bxy] = (btype, None)
                elif btype == EntityType.BARRIER:
                    self.building_cache[bxy] = (btype, None)
                    self.allied_barriers.add(bxy)
                # Track every allied non-walkable defensive building.
                # Includes the HARVESTER/FOUNDRY/etc. branches above —
                # checking after the elif chain keeps the set in sync
                # without an extra branch in the hot path.
                if btype in _BLOCKER_TYPES:
                    self.allied_blockers.add(bxy)

        # Clear building data only for tiles that were visible LAST turn
        # (as a building) AND are visible THIS turn with no building — i.e.
        # confirmed destroyed. Tiles that simply drifted out of vision must
        # keep their bid — otherwise we'd hallucinate "empty" tiles, and
        # downstream pickers (e.g. disruptor intercept sentinel-tile picker)
        # would retarget them every turn and loop forever on approach.
        prev_bldg = getattr(self, '_prev_building_positions', set())
        current_vision = self._last_vision_set
        for xy in prev_bldg:
            if xy not in building_positions and xy in current_vision:
                old = self.tile_cache.get(xy)
                if old and old[1] is not None:
                    self.tile_cache[xy] = (old[0], None, None, None)
                    self.bridge_target_cache.pop(xy, None)
                    # If a tracked allied barrier / blocker just
                    # disappeared, drop the set entries too.
                    self.allied_barriers.discard(xy)
                    self.allied_blockers.discard(xy)
        self._prev_building_positions = building_positions

        # --- Pass B: detect destroyed buildings ---
        if self.tle_recovery_turns == 0 and moved:
            _CACHE_TYPES = (EntityType.CONVEYOR, EntityType.SPLITTER,
                            EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
                            EntityType.HARVESTER, EntityType.FOUNDRY,
                            EntityType.BARRIER)
            current_vision = self._last_vision_set
            bc = self.building_cache
            tc = self.tile_cache
            my_team = self.my_team_cache
            for xy in list(bc):
                if xy not in current_vision:
                    continue
                cached_etype, cached_detail = bc[xy]
                entry = tc.get(xy)
                if entry is None:
                    continue
                env, bid, cur_etype, cur_team = entry
                # Same allied building still present
                if cur_etype == cached_etype and cur_team == my_team:
                    continue
                # Different allied transport — someone rebuilt, update cache
                if cur_team == my_team and bid is not None and cur_etype in _CACHE_TYPES:
                    if cur_etype in (EntityType.CONVEYOR, EntityType.SPLITTER,
                                     EntityType.ARMOURED_CONVEYOR):
                        direction = ct.get_direction(bid)
                        bc[xy] = (cur_etype, direction)
                    elif cur_etype == EntityType.BRIDGE:
                        bc[xy] = (cur_etype, self.bridge_target_cache.get(xy))
                    else:
                        # HARVESTER, FOUNDRY, BARRIER — no extra detail
                        bc[xy] = (cur_etype, None)
                    # Cached BARRIER → cur_etype != BARRIER means the
                    # barrier is gone; drop the set entry. Conversely a
                    # newly observed BARRIER lands here when the cache
                    # had something else for the tile.
                    if cur_etype == EntityType.BARRIER:
                        self.allied_barriers.add(xy)
                    elif cached_etype == EntityType.BARRIER:
                        self.allied_barriers.discard(xy)
                    # Same logic for the blocker set: HARVESTER /
                    # FOUNDRY / BARRIER are blockers; CONVEYOR /
                    # BRIDGE / SPLITTER / ARMOURED_CONVEYOR are not.
                    if cur_etype in _BLOCKER_TYPES:
                        self.allied_blockers.add(xy)
                    elif cached_etype in _BLOCKER_TYPES:
                        self.allied_blockers.discard(xy)
                    continue
                # Tile now holds an allied building that's NOT tracked
                # economy infrastructure. For sentinels we leave the
                # old economy entry untouched — when the sentinel
                # self-destructs (see sentinel.py) the tile will show
                # as empty on the next pass and drop straight into
                # pending_repairs, rebuilding the original conveyor /
                # bridge / splitter. For other non-economy allied
                # buildings (gunners, breach, launcher, etc.) the
                # replacement is permanent; drop the cache entry so
                # we don't loop trying to repair over them.
                if cur_team == my_team and bid is not None:
                    if cur_etype != EntityType.SENTINEL:
                        if cached_etype == EntityType.BARRIER:
                            self.allied_barriers.discard(xy)
                        del bc[xy]
                    # The new allied building may itself be a blocker
                    # (sentinel/gunner/breach/launcher) — keep the
                    # blocker set in sync.
                    if cur_etype in _BLOCKER_TYPES:
                        self.allied_blockers.add(xy)
                    elif cached_etype in _BLOCKER_TYPES:
                        self.allied_blockers.discard(xy)
                    continue
                # Tile is empty, has enemy building, or has allied road → needs repair
                if cached_etype == EntityType.BARRIER:
                    self.allied_barriers.discard(xy)
                if cached_etype in _BLOCKER_TYPES:
                    self.allied_blockers.discard(xy)
                if xy not in self.pending_repairs:
                    self.pending_repairs.append(xy)

        # --- Step 4: batch unit query ---
        self.bot_pos_cache.clear()
        new_bot_positions = set()
        units = ct.get_nearby_units()
        for uid in units:
            upos = ct.get_position(uid)
            uxy = (upos.x, upos.y)
            uteam = ct.get_team(uid)
            self.bot_pos_cache[uxy] = (uid, uteam)
            if uxy != self._my_xy:
                new_bot_positions.add(uxy)

        # Track stationary bots — increment count if same position, reset if moved
        old_stat = self._bot_stationary
        new_stat = {}
        for xy in new_bot_positions:
            new_stat[xy] = old_stat.get(xy, 0) + 1
        self._bot_stationary = new_stat

    # ------------------------------------------------------------------ #
    #  Frontier cache: incremental maintenance                            #
    # ------------------------------------------------------------------ #
    #  frontier_cache holds known non-wall tiles that have at least one
    #  in-bounds unknown neighbor. This lets exploration find the outer
    #  perimeter of explored space in O(|frontier|) instead of scanning
    #  the entire tile_cache. Out-of-bounds neighbors do NOT count as
    #  "unknown" — this prevents bots chasing the map edge.
    # ------------------------------------------------------------------ #

    def _update_frontier(self, xy):
        """Update frontier status for xy and its 4 cardinal neighbors.
        Called after a tile is added to tile_cache."""
        self._check_frontier(xy)
        x, y = xy
        tc = self.tile_cache
        for nxy in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if nxy in tc:
                self._check_frontier(nxy)

    def _check_frontier(self, xy):
        """Add/remove xy in frontier_cache based on current tile_cache state.
        Mirrored (predicted) tiles don't count as 'known' — only tiles
        confirmed by real observation remove frontier status."""
        entry = self.tile_cache.get(xy)
        if entry is None or entry[0] == Environment.WALL:
            self.frontier_cache.discard(xy)
            return
        # Don't mark mirrored tiles as frontiers — they're predictions.
        mirror_src = getattr(self, '_mirror_src', set())
        if xy in mirror_src:
            self.frontier_cache.discard(xy)
            return
        x, y = xy
        mw = self.map_w
        mh = self.map_h
        tc = self.tile_cache
        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if mw is not None and (nx < 0 or ny < 0 or nx >= mw or ny >= mh):
                continue
            # Mirrored neighbors count as unknown for frontier purposes.
            if (nx, ny) not in tc or (nx, ny) in mirror_src:
                self.frontier_cache.add(xy)
                return
        self.frontier_cache.discard(xy)

    def _handle_bad_symmetry(self, mirror_src):
        """Symmetry prediction was wrong. Purge mirrored tiles, try next type."""
        # Remove all tiles that came from mirroring
        affected = list(mirror_src)
        for mxy in affected:
            if mxy in self.tile_cache:
                env = self.tile_cache[mxy][0]
                del self.tile_cache[mxy]
                if env == Environment.WALL:
                    self.known_walls.discard(mxy)
                self.frontier_cache.discard(mxy)
        # Re-evaluate neighbors of purged tiles (they may have lost/gained frontier status)
        for mxy in affected:
            x, y = mxy
            for nxy in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
                if nxy in self.tile_cache:
                    self._check_frontier(nxy)
        mirror_src.clear()
        self._mirror_src = mirror_src

        # Try next symmetry type
        tried = getattr(self, '_sym_tried', set())
        tried.add(self._symmetry)
        self._sym_tried = tried

        for s in _SYM_ORDER:
            if s not in tried:
                self._symmetry = s
                if DEBUG:
                    print(f"[{getattr(self, 'corner', '??')}] symmetry {list(tried)} wrong, trying {s}",
                          file=sys.stderr)
                return s

        # All three failed — disable mirroring
        self._symmetry = None
        if DEBUG:
            print(f"[{getattr(self, 'corner', '??')}] all symmetries wrong, mirroring disabled",
                  file=sys.stderr)
        return None
