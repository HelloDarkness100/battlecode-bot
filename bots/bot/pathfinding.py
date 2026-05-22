import heapq
from cambc import EntityType, Environment
from constants import ASTAR_MAX_NODES, STRICT_MONOTONIC, BARRIER_PATH_COST
from utils import neighbors_8, neighbors_4

# Entity types a bot can walk on (cost 1 — already built).
# Markers are *not* walkable — bots can't step onto them. The only way
# to remove a marker is to build over it (the engine destroys the marker
# as part of the build action).
_WALKABLE_BUILDINGS = frozenset({
    EntityType.ROAD,
    EntityType.CONVEYOR,
    EntityType.SPLITTER,
    EntityType.BRIDGE,
    EntityType.ARMOURED_CONVEYOR,
    EntityType.CORE,
})

# Buildings that make an ore tile impassable
_BLOCKING_ON_ORE = frozenset({
    EntityType.HARVESTER,
    EntityType.BARRIER,
    EntityType.FOUNDRY,
    EntityType.GUNNER,
    EntityType.SENTINEL,
    EntityType.BREACH,
    EntityType.LAUNCHER,
})


def _tile_cost(xy, tile_cache, extra_walls, my_team=None, known_only=False):
    """Return movement cost for a tile, or -1 if impassable.

    Reads entirely from tile_cache — zero API calls.
    known_walls entries are already in tile_cache as WALL.

    When known_only=True, unknown (not-yet-seen) tiles are impassable
    instead of the default cost-3. Used by disruptors to avoid exploring
    hundreds of unknown tiles during long-range pathfind.
    """
    if extra_walls and xy in extra_walls:
        return -1

    entry = tile_cache.get(xy)
    if entry is None:
        return -1 if known_only else 3

    env, bid, etype, team = entry

    if env == Environment.WALL:
        return -1

    if (env == Environment.ORE_TITANIUM or env == Environment.ORE_AXIONITE):
        if bid is not None and etype in _BLOCKING_ON_ORE:
            # Allied barrier on ore: economy bots can bust through
            # (destroy → road → step → harvester). Enemy barrier,
            # harvester, etc. stay impassable.
            if (etype == EntityType.BARRIER
                    and my_team is not None and team == my_team):
                return BARRIER_PATH_COST
            return -1

    if bid is not None:
        if etype in _WALKABLE_BUILDINGS:
            return 1
        # Markers: allied markers are our own foundry placeholders —
        # treat as impassable so A* routes around them (we don't want
        # other flows destroying our markers). Enemy markers behave like
        # empty tiles at cost 2 — the bot plans through them and building
        # a road/conveyor on the tile atomically destroys the marker.
        # (Bots physically cannot step onto ANY marker, but the cost-2
        # phantom behavior for enemy markers simulates build-over.)
        if etype == EntityType.MARKER:
            if my_team is not None and team == my_team:
                return -1
            return 2
        # Allied barriers are bustable by economy bots
        # (destroy → road → move → restore). Patrol bots inject these
        # into extra_walls during patrol circling so they path around
        # instead of trying to bust through. Enemy barriers stay
        # impassable — builder bots can't destroy them.
        if etype == EntityType.BARRIER and my_team is not None and team == my_team:
            return BARRIER_PATH_COST
        return -1

    return 2


def astar_cached(start, goal, tile_cache, known_walls,
                 extra_walls=None, cardinal_only=False,
                 max_nodes=ASTAR_MAX_NODES, map_w=None, map_h=None,
                 my_team=None, known_only=False):
    """A* pathfinding using only tile_cache — zero game-engine API calls.

    Args:
        start: (x,y) start position (excluded from result)
        goal: (x,y) goal position (included in result)
        tile_cache: dict[(x,y)] = (env, bid, etype, team)
        known_walls: set of (x,y) confirmed wall positions (unused, kept for API compat)
        extra_walls: optional set of (x,y) temporary impassable positions
        cardinal_only: if True, only expand N/E/S/W (for conveyor routing)
        max_nodes: safety cap on nodes expanded
        map_w: map width (if set, out-of-bounds tiles are impassable)
        map_h: map height

    Returns:
        list[(x,y)] path from start to goal (excluding start, including goal),
        or None if no path found.
    """
    if start == goal:
        return []

    if map_w is not None:
        gx, gy = goal
        if gx < 0 or gy < 0 or gx >= map_w or gy >= map_h:
            return None

    neighbor_fn = neighbors_4 if cardinal_only else neighbors_8

    if cardinal_only:
        def h(a):
            return abs(a[0] - goal[0]) + abs(a[1] - goal[1])
    else:
        def h(a):
            dx = abs(a[0] - goal[0])
            dy = abs(a[1] - goal[1])
            return max(dx, dy)

    counter = 0
    open_heap = [(h(start), counter, start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        nodes_expanded += 1
        if nodes_expanded >= max_nodes:
            return None

        current_g = g_score[current]

        for nx, ny in neighbor_fn(current[0], current[1]):
            if map_w is not None and (nx < 0 or ny < 0 or nx >= map_w or ny >= map_h):
                continue
            nxy = (nx, ny)
            if nxy in closed:
                continue

            cost = _tile_cost(nxy, tile_cache, extra_walls, my_team, known_only)
            if cost < 0:
                continue

            tentative_g = current_g + cost
            if tentative_g < g_score.get(nxy, float('inf')):
                g_score[nxy] = tentative_g
                came_from[nxy] = current
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h(nxy), counter, nxy))

    return None


def _core_adjacent_tiles(core_pos):
    """Return the 12 tiles cardinally adjacent to the 3×3 core."""
    cx, cy = core_pos
    tiles = []
    for x in (cx - 1, cx, cx + 1):
        tiles.append((x, cy - 2))
        tiles.append((x, cy + 2))
    for y in (cy - 1, cy, cy + 1):
        tiles.append((cx - 2, y))
        tiles.append((cx + 2, y))
    return tiles


def astar_monotonic(start, core_pos, tile_cache, known_walls,
                    map_w=None, map_h=None, max_nodes=ASTAR_MAX_NODES,
                    strict=STRICT_MONOTONIC, extra_walls=None,
                    my_team=None):
    """Monotonic-distance A* for chain routing — always moves toward core.

    Cardinal-only. Only expands neighbors that decrease (or with strict=False,
    don't increase) Manhattan distance to the goal. Obstacles (walls,
    buildings) are treated as passable with cost 2 — handled at build-time
    via optimal_bridge.

    The goal is the single closest non-wall core-edge tile to start (from
    the 12 cardinally adjacent + 9 core tiles). This ensures the chain
    heads straight for the nearest edge instead of veering to a farther one.

    Returns list[(x,y)] path (excluding start, including goal), or None.
    """
    if start == core_pos:
        return []

    cx, cy = core_pos
    # Core tiles (3×3) — valid goals but not passable mid-path
    core_3x3 = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            core_3x3.add((cx + dx, cy + dy))

    # All candidate goals: 12 core-adjacent + 9 core tiles
    adj_tiles = _core_adjacent_tiles(core_pos)
    all_goals = set(adj_tiles) | core_3x3

    # Filter out walls, then pick the single closest to start
    valid_goals = set()
    for g in all_goals:
        if g in known_walls:
            continue
        entry = tile_cache.get(g)
        if entry and entry[0] == Environment.WALL:
            continue
        valid_goals.add(g)

    if not valid_goals:
        return None

    # Pick closest core-edge tile to start as the single goal
    sx, sy = start
    best_goal = None
    best_d = float('inf')
    for g in valid_goals:
        d = abs(g[0] - sx) + abs(g[1] - sy)
        if d < best_d:
            best_d = d
            best_goal = g

    def min_manhattan(xy):
        return abs(xy[0] - best_goal[0]) + abs(xy[1] - best_goal[1])

    counter = 0
    open_heap = [(min_manhattan(start), counter, start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current == best_goal:
            # Reconstruct path
            path = []
            node = current
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        if current in closed:
            continue
        closed.add(current)

        nodes_expanded += 1
        if nodes_expanded >= max_nodes:
            return None

        current_g = g_score[current]
        current_dist = min_manhattan(current)
        x, y = current

        for nx, ny in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            if map_w is not None and (nx < 0 or ny < 0 or nx >= map_w or ny >= map_h):
                continue
            nxy = (nx, ny)
            if nxy in closed:
                continue

            neighbor_dist = min_manhattan(nxy)

            # Monotonic constraint
            if strict:
                if neighbor_dist >= current_dist:
                    continue
            else:
                if neighbor_dist > current_dist:
                    continue

            # Core tiles are valid ONLY as the goal, not mid-path
            if nxy in core_3x3 and nxy != best_goal:
                continue

            # Hard-impassable tiles (e.g. allied barriers) — chain routing
            # must never plan a conveyor through these.
            if extra_walls and nxy in extra_walls:
                continue

            # Tile cost — walls/buildings passable (cost 2) for phantom path.
            # Pass my_team so allied markers return -1 (route around)
            # while enemy markers return 2 (plan through, build-over).
            cost = _tile_cost(nxy, tile_cache, None, my_team)
            if cost < 0:
                # Allied markers are foundry placeholders — non-pathfindable.
                # Exception: if the marker IS the goal (axionite chain
                # routing to an allied foundry marker), allow phantom-pass
                # so the chain can terminate there.
                entry = tile_cache.get(nxy)
                if (entry is not None and entry[1] is not None
                        and entry[2] == EntityType.MARKER
                        and nxy != best_goal):
                    continue
                cost = 2  # phantom passable — obstacle handled at build-time

            tentative_g = current_g + cost
            if tentative_g < g_score.get(nxy, float('inf')):
                g_score[nxy] = tentative_g
                came_from[nxy] = current
                counter += 1
                heapq.heappush(open_heap, (tentative_g + min_manhattan(nxy), counter, nxy))

    return None
