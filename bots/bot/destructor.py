"""Destructor bot — test-only role for simulating enemy attacks on allied
infrastructure. Spawned when REPAIR_TEST=True to validate that patrol and
economy bots detect destroyed buildings and trigger repairs.

The destructor pathfinds to the nearest allied conveyor/bridge/splitter/
armoured_conveyor, destroys it (free action), then repeats. Mimics what an
enemy builder might do to break our chains.
"""
import sys
from cambc import EntityType, GameConstants as GC
from constants import DEBUG
from utils import euclidean_dist_sq, xy_to_pos, direction_between


_DESTRUCTOR_TARGETS = frozenset({
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
})

_DESTRUCTOR_LIFESPAN = 500  # Turns before self-destruct


class DestructorMixin:
    """Test-only role: destroy allied chain infrastructure."""

    def _run_destructor(self, ct):
        my_xy = self._my_xy

        # Track spawn round on first turn, self-destruct after lifespan
        if self._destructor_spawn_round is None:
            self._destructor_spawn_round = ct.get_current_round()
            self._destructor_destroyed = set()
        elif ct.get_current_round() - self._destructor_spawn_round >= _DESTRUCTOR_LIFESPAN:
            if DEBUG:
                print(f"[D] self-destructing after {_DESTRUCTOR_LIFESPAN} turns",
                      file=sys.stderr)
            ct.self_destruct()
            return  # execution terminated anyway

        # Find closest allied conveyor/bridge/splitter target
        target_xy = self._destructor_find_target(my_xy)
        if target_xy is None:
            return  # Nothing to destroy — wait

        # Within action radius? destroy (free)
        dsq = euclidean_dist_sq(my_xy, target_xy)
        if dsq <= GC.ACTION_RADIUS_SQ:
            target_pos = xy_to_pos(target_xy)
            if ct.can_destroy(target_pos):
                ct.destroy(target_pos)
                # Clear from tile_cache so next scan picks a different target
                old = self.tile_cache.get(target_xy)
                if old:
                    self.tile_cache[target_xy] = (old[0], None, None, None)
                # Remember we've already hit this tile — never target again
                self._destructor_destroyed.add(target_xy)
                if DEBUG:
                    print(f"[D] destroyed allied at ({target_xy[0]},{target_xy[1]})",
                          file=sys.stderr)
                # Clear path so next turn picks a new target
                self.path = None
                self.path_index = 0
            return

        # Not in range — pathfind toward it
        if self.path is None or self.path_index >= len(self.path):
            goal = self._destructor_find_adjacent(my_xy, target_xy)
            if goal is None:
                return
            self.path = self._compute_path(my_xy, goal)
            self.path_index = 0
            if self.path is None:
                return

        result = self._follow_path(ct)
        if result == 'blocked':
            self.path = None
            self.path_index = 0

    def _destructor_find_target(self, my_xy):
        """Find closest allied conveyor/bridge/splitter in tile_cache that
        this destructor hasn't already hit."""
        tc = self.tile_cache
        my_team = self.my_team_cache
        done = self._destructor_destroyed
        best = None
        best_dsq = 999999
        for xy, (env, bid, etype, team) in tc.items():
            if bid is None:
                continue
            if team != my_team:
                continue
            if etype not in _DESTRUCTOR_TARGETS:
                continue
            if xy in done:
                continue
            d = euclidean_dist_sq(my_xy, xy)
            if d < best_dsq:
                best_dsq = d
                best = xy
        return best

    def _destructor_find_adjacent(self, my_xy, target_xy):
        """Find closest walkable 8-neighbor of target_xy to pathfind to."""
        from cambc import Environment
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
                d = euclidean_dist_sq(my_xy, nxy)
                if d < best_dsq:
                    best_dsq = d
                    best = nxy
        return best
