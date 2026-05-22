import sys
from cambc import EntityType, Direction, GameConstants as GC
from constants import DEBUG, ECONOMY_TEST_MODE
from sentinel import _TARGET_PRIORITY, _IGNORED_TARGETS
from pathfinding import _WALKABLE_BUILDINGS

_ALL_DIRS = [
    Direction.NORTH, Direction.NORTHEAST, Direction.EAST, Direction.SOUTHEAST,
    Direction.SOUTH, Direction.SOUTHWEST, Direction.WEST, Direction.NORTHWEST,
]

_DELTA_TO_DIR = {
    (0, -1): Direction.NORTH,
    (1, 0):  Direction.EAST,
    (0, 1):  Direction.SOUTH,
    (-1, 0): Direction.WEST,
}

# Enemy entity types that do NOT justify paying 10 Ti + 1 turn to rotate.
# We can still fire at them when they wander into the current attackable cone.
_NO_ROTATE_TARGETS = frozenset({
    EntityType.ROAD,
    EntityType.MARKER,
    EntityType.BUILDER_BOT,
})


class GunnerMixin:
    """Gunner turret logic. Rotations are expensive (10 Ti + 1 turn), so we
    only rotate when a high-value target justifies it — non-road enemy
    buildings, or an allied walkable building that is currently taking
    damage. Enemy bots and roads can still be shot if they wander into the
    current facing cone, but they never trigger a rotation on their own.
    """

    def run_gunner(self, ct):
        if ECONOMY_TEST_MODE:
            return

        my_team = ct.get_team()
        my_pos = ct.get_position()
        mx, my_ = my_pos.x, my_pos.y

        # Snapshot the direction the gunner was built facing. After every
        # rotation priority (cone-fire / threat-rotate / defend-damaged) is
        # exhausted this turn, we rotate back toward this default so the
        # gunner keeps covering the conveyor it was placed to defend.
        if self._gunner_original_dir is None:
            self._gunner_original_dir = ct.get_direction()

        nearby_buildings = list(ct.get_nearby_buildings())
        nearby_units = list(ct.get_nearby_units())

        # ---- Per-turn: which cardinal neighbour holds our feeding
        # harvester? The gunner can't accept ammo from its facing
        # direction, so that direction must be excluded from rotation.
        # Recomputed every turn because adjacent buildings can change
        # (harvesters rebuilt, barriers destroyed, etc.).
        harvester_dir = None
        for bid in nearby_buildings:
            if ct.get_team(bid) != my_team:
                continue
            if ct.get_entity_type(bid) != EntityType.HARVESTER:
                continue
            bp = ct.get_position(bid)
            dx = bp.x - mx
            dy = bp.y - my_
            if abs(dx) + abs(dy) == 1:
                harvester_dir = _DELTA_TO_DIR.get((dx, dy))
                break

        # ---- Single pass over nearby buildings: enemies + damaged allies ----
        all_enemies = []   # everything we can fire at (incl. bots/roads)
        rot_enemies = []   # enemies worth paying a rotation for
        damaged_allied = []  # (x,y) of allied walkable buildings taking damage
        mhc = self._max_hp_by_type

        for bid in nearby_buildings:
            team = ct.get_team(bid)
            et = ct.get_entity_type(bid)
            if team != my_team:
                if et in _IGNORED_TARGETS:
                    continue
                pri = _TARGET_PRIORITY.get(et, 999)
                pos = ct.get_position(bid)
                hp = ct.get_hp(bid)
                all_enemies.append((pri, hp, pos, et))
                if et not in _NO_ROTATE_TARGETS:
                    rot_enemies.append((pri, hp, pos, et))
            else:
                if et not in _WALKABLE_BUILDINGS:
                    continue
                max_hp = mhc.get(et)
                if max_hp is None:
                    try:
                        max_hp = ct.get_max_hp(bid)
                    except Exception:
                        continue
                    mhc[et] = max_hp
                if not max_hp:
                    continue
                try:
                    hp = ct.get_hp(bid)
                except Exception:
                    continue
                if hp < max_hp:
                    bp = ct.get_position(bid)
                    damaged_allied.append((bp.x, bp.y))

        for uid in nearby_units:
            if ct.get_team(uid) == my_team:
                continue
            et = ct.get_entity_type(uid)
            pri = _TARGET_PRIORITY.get(et, 999)
            all_enemies.append((pri, ct.get_hp(uid), ct.get_position(uid), et))
            # Bots intentionally NOT added to rot_enemies: they move too
            # much and rotating to chase them burns titanium.

        if not all_enemies and not damaged_allied:
            self._gunner_rotate_to_default(ct, harvester_dir)
            return

        # Friendly-fire guard + enemy bot positions
        allied_bot_tiles = set()
        enemy_bot_tiles = set()
        for uid in nearby_units:
            if ct.get_entity_type(uid) != EntityType.BUILDER_BOT:
                continue
            up = ct.get_position(uid)
            if ct.get_team(uid) == my_team:
                allied_bot_tiles.add((up.x, up.y))
            else:
                enemy_bot_tiles.add((up.x, up.y))

        all_enemies.sort(key=lambda t: (t[0], t[1]))
        rot_enemies.sort(key=lambda t: (t[0], t[1]))

        # The enemy CORE is a 3x3 footprint but `ct.get_position(bid)`
        # returns only the centre. The attack cone may cover an edge
        # tile of the footprint without containing the centre, so any
        # cone-membership test on the centre alone misses CORE hits.
        # Expand to the 9 footprint tiles on demand.
        def _enemy_tiles(pos, et):
            if et == EntityType.CORE:
                return [(pos.x + dx, pos.y + dy)
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            return [(pos.x, pos.y)]

        # ---- Step 1: fire at anything already in the current cone ----
        current_dir = ct.get_direction()
        current_attackable = set((t.x, t.y) for t in ct.get_attackable_tiles())
        in_current = False
        for pri, hp, pos, et in all_enemies:
            for txy in _enemy_tiles(pos, et):
                if txy in current_attackable and txy not in allied_bot_tiles:
                    in_current = True
                    break
            if in_current:
                break

        if in_current:
            tgt = ct.get_gunner_target()
            if tgt is not None:
                tgt_xy = (tgt.x, tgt.y)
                tgt_bid = ct.get_tile_building_id(tgt)
                # Safe to fire if: no allied building, OR an enemy bot is on
                # top (turret attacks hit ONLY the bot, not the building beneath).
                allied_bldg = tgt_bid is not None and ct.get_team(tgt_bid) == my_team
                if not allied_bldg or tgt_xy in enemy_bot_tiles:
                    if ct.get_action_cooldown() == 0 and ct.can_fire(tgt):
                        ct.fire(tgt)
                    return
            # Allied building in line of fire with no enemy bot on top — fall through

        # ---- Step 2: rotation decision ----
        # Only rotate for (a) non-road enemy buildings, or (b) to defend a
        # damaged allied walkable building.
        best_dir = None
        best_pri = 999

        if rot_enemies:
            for d in _ALL_DIRS:
                if d == harvester_dir:
                    continue
                tiles = ct.get_attackable_tiles_from(my_pos, d, EntityType.GUNNER)
                tile_set = set((t.x, t.y) for t in tiles)
                for pri, hp, pos, et in rot_enemies:
                    if pri >= best_pri:
                        break
                    hit = False
                    for txy in _enemy_tiles(pos, et):
                        if txy in tile_set and txy not in allied_bot_tiles:
                            hit = True
                            break
                    if hit:
                        best_pri = pri
                        best_dir = d
                        break

        if best_dir is None and damaged_allied:
            # Face the closest damaged allied walkable building so any
            # attacker adjacent to it falls into our cone.
            damaged_allied.sort(key=lambda xy: (xy[0] - mx) ** 2 + (xy[1] - my_) ** 2)
            for dmg_xy in damaged_allied:
                for d in _ALL_DIRS:
                    if d == harvester_dir:
                        continue
                    tiles = ct.get_attackable_tiles_from(my_pos, d, EntityType.GUNNER)
                    if any(t.x == dmg_xy[0] and t.y == dmg_xy[1] for t in tiles):
                        best_dir = d
                        break
                if best_dir is not None:
                    break

        if best_dir is None:
            self._gunner_rotate_to_default(ct, harvester_dir)
            return

        # ---- Step 3: rotate ----
        if best_dir != current_dir:
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.GUNNER_ROTATE_COST[0])
                    and ct.can_rotate(best_dir)):
                ct.rotate(best_dir)
                if DEBUG:
                    print(f"[G] rotate -> {best_dir.value}", file=sys.stderr)
            return

        # ---- Step 4: fire ----
        tgt = ct.get_gunner_target()
        if tgt is None:
            return
        tgt_xy = (tgt.x, tgt.y)
        tgt_bid = ct.get_tile_building_id(tgt)
        allied_bldg = tgt_bid is not None and ct.get_team(tgt_bid) == my_team
        if allied_bldg and tgt_xy not in enemy_bot_tiles:
            return
        if ct.get_action_cooldown() == 0 and ct.can_fire(tgt):
            ct.fire(tgt)

    def _gunner_rotate_to_default(self, ct, harvester_dir):
        """Rotate back to the direction the gunner was built facing, if
        we've drifted and can afford the cost. Called only when no
        higher-priority rotation target (enemy in any cone, damaged
        ally) exists this turn. Skip if the default equals the harvester
        feed direction (facing the harvester would starve the gunner).
        """
        original = self._gunner_original_dir
        if original is None:
            return
        if original == harvester_dir:
            return
        current = ct.get_direction()
        if current == original:
            return
        if ct.get_action_cooldown() != 0:
            return
        if not self._can_spend(ct, GC.GUNNER_ROTATE_COST[0]):
            return
        if ct.can_rotate(original):
            ct.rotate(original)
            if DEBUG:
                print(f"[G] rotate back -> {original.value}", file=sys.stderr)

