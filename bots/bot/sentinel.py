import sys
from cambc import EntityType, Direction
from constants import DEBUG, ECONOMY_TEST_MODE, SENTINEL_IDLE_CLEANUP_TURNS
from utils import xy_to_pos

# Target priority: lower = higher priority. Harvesters and foundries are
# intentionally excluded — they feed our sentinels their ammo, destroying
# them is counterproductive.
_TARGET_PRIORITY = {
    # Enemy core — winning condition, always shoot first.
    EntityType.CORE: -1,
    # Enemy turrets — ordered by burn-priority. Breach explodes when
    # destroyed (high burst damage to anything adjacent), so killing it
    # off our infrastructure first matters most. Sentinels next (their
    # cone bypasses cover). Gunners + launchers last; launchers can't
    # shoot turrets so they're least urgent.
    EntityType.BREACH: 0,
    EntityType.SENTINEL: 1,
    EntityType.GUNNER: 2,
    EntityType.LAUNCHER: 3,
    # Enemy builder bots
    EntityType.BUILDER_BOT: 4,
    # Enemy logistics
    EntityType.CONVEYOR: 5,
    EntityType.ARMOURED_CONVEYOR: 5,
    EntityType.BRIDGE: 5,
    EntityType.SPLITTER: 5,
    # Enemy defenses
    EntityType.BARRIER: 6,
    # Enemy infrastructure
    EntityType.ROAD: 7,
    EntityType.MARKER: 8,
}

_IGNORED_TARGETS = frozenset({EntityType.HARVESTER, EntityType.FOUNDRY})


class SentinelMixin:
    """Handles sentinel turret logic — scan, prioritise, fire. No rotation."""

    def run_sentinel(self, ct):
        """Called each turn for a sentinel unit."""
        if ECONOMY_TEST_MODE:
            return
        my_team = ct.get_team()
        my_pos = ct.get_position()
        my_xy = (my_pos.x, my_pos.y)

        # Get fixed attackable tiles (sentinel can't rotate)
        attackable = ct.get_attackable_tiles()
        if not attackable:
            return

        # Convert to set of (x,y) for fast lookup
        attackable_set = set()
        for pos in attackable:
            attackable_set.add((pos.x, pos.y))

        # Per engine rules: if a turret shot hits a tile that contains both
        # a building and a builder bot, ONLY the bot is damaged. Collect
        # allied bot tiles so we never waste a shot pointed at a building
        # our own bot is standing on (e.g. a builder destroying an enemy
        # conveyor it's just walked onto).
        allied_bot_tiles = set()
        for uid in ct.get_nearby_units():
            if ct.get_team(uid) != my_team:
                continue
            if ct.get_entity_type(uid) != EntityType.BUILDER_BOT:
                continue
            upos = ct.get_position(uid)
            allied_bot_tiles.add((upos.x, upos.y))

        # Scan for enemy buildings
        targets = []
        for bid in ct.get_nearby_buildings():
            bteam = ct.get_team(bid)
            if bteam == my_team:
                continue
            btype = ct.get_entity_type(bid)
            # Never target enemy harvesters/foundries — they may be fuelling
            # our own sentinels, and destroying them hurts us more than them.
            if btype in _IGNORED_TARGETS:
                continue
            priority = _TARGET_PRIORITY.get(btype, 999)
            bpos = ct.get_position(bid)
            bxy = (bpos.x, bpos.y)
            if bxy in attackable_set and bxy not in allied_bot_tiles:
                bhp = ct.get_hp(bid)
                targets.append((priority, bhp, bpos))
            elif btype == EntityType.CORE:
                # Core is 3x3 — check if any of its 9 tiles are attackable
                # and not occupied by an allied bot (friendly-fire guard).
                cx, cy = bxy
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cxy = (cx + dx, cy + dy)
                        if (cxy in attackable_set
                                and cxy not in allied_bot_tiles):
                            bhp = ct.get_hp(bid)
                            targets.append((priority, bhp, xy_to_pos(cxy)))
                            break
                    else:
                        continue
                    break

        # Scan for enemy units
        for uid in ct.get_nearby_units():
            uteam = ct.get_team(uid)
            if uteam == my_team:
                continue
            utype = ct.get_entity_type(uid)
            priority = _TARGET_PRIORITY.get(utype, 999)
            upos = ct.get_position(uid)
            if (upos.x, upos.y) in attackable_set:
                uhp = ct.get_hp(uid)
                targets.append((priority, uhp, upos))

        if not targets:
            # Nothing to shoot right now. Chain-supplied sentinels are
            # short-term tools (turret response, supply disruption) and
            # should self-destruct once their job is done so the economy
            # can reclaim the tile. Sentinels cardinally adjacent to an
            # allied harvester are the permanent harvester-defense kit —
            # they never self-destruct.
            supplied_by_harvester = False
            for bid in ct.get_nearby_buildings():
                if ct.get_team(bid) != my_team:
                    continue
                if ct.get_entity_type(bid) != EntityType.HARVESTER:
                    continue
                bpos = ct.get_position(bid)
                if abs(bpos.x - my_xy[0]) + abs(bpos.y - my_xy[1]) == 1:
                    supplied_by_harvester = True
                    break
            if supplied_by_harvester:
                self._idle_turns = 0
                return
            idle = getattr(self, '_idle_turns', 0) + 1
            self._idle_turns = idle
            if idle >= SENTINEL_IDLE_CLEANUP_TURNS:
                if DEBUG:
                    print(f"[S] self-destruct idle at ({my_xy[0]},{my_xy[1]})",
                          file=sys.stderr)
                ct.self_destruct()
            return

        # Sort by priority first, then lowest HP (finish off weak targets)
        targets.sort(key=lambda t: (t[0], t[1]))
        best_pos = targets[0][2]

        self._idle_turns = 0
        if ct.can_fire(best_pos):
            ct.fire(best_pos)
