# Cambridge Battlecode Bot

## What This Is
A Python bot for the Cambridge Battlecode competition. The game engine creates
one Player instance per unit (core, builder bot, turret) and calls `run(controller)`
once per round with a **2ms time limit**. Maps range from **20×20 to 50×50 tiles**.

## Key Reference Files
- `Strategy.md` — What to build. Read this FIRST before any coding.
- `cambridge_battlecode_docs.md` — Game API, entities, mechanics, costs.

## Code Structure
All bot code goes in `bots/bot/`. The engine loads `bots/bot/main.py` looking for a Player class.

Uses the **mixin pattern** to keep files small and focused. Each mixin defines
a group of related methods that all share `self` on the Player class:

```
bots/bot/
├── main.py          ~50 lines   Player class, __init__, run() dispatch
├── constants.py     ~30 lines   All tunable numbers
├── utils.py         ~55 lines   Distance, direction, encoding helpers
├── pathfinding.py   ~140 lines  Cache-powered A* (8-dir + cardinal-only)
├── scanning.py      ~80 lines   _scan_turn, _get_nearby_cached, cache management
├── core_logic.py    ~60 lines   Core unit spawning logic
├── exploration.py   ~120 lines  Frontier detection, _step_explore, axionite detection
├── harvesting.py    ~150 lines  Ore detection, claim ore (on-ore), build harvester + gunner
├── chaining.py      ~180 lines  Chain to core, observe conveyor, splitter placement
├── movement.py      ~120 lines  _follow_path, stuck detection, barrier busting
└── gunner.py        ~60 lines   GunnerMixin: turret targeting, rotation, firing
```

### Mixin Pattern
```python
# bots/bot/scanning.py
class ScanningMixin:
    def _scan_turn(self, ct): ...
    def _get_nearby_cached(self, ct): ...

# bots/bot/exploration.py
class ExplorationMixin:
    def _step_explore(self, ct): ...
    def _find_frontier_target(self, ct): ...

# bots/bot/main.py
from scanning import ScanningMixin
from exploration import ExplorationMixin
from harvesting import HarvestingMixin
from chaining import ChainingMixin
from movement import MovementMixin
from core_logic import CoreMixin
from gunner import GunnerMixin

class Player(ScanningMixin, ExplorationMixin, HarvestingMixin,
             ChainingMixin, MovementMixin, CoreMixin, GunnerMixin):
    def __init__(self):
        # ALL state variables defined here — shared across all mixins via self
        ...
    def run(self, ct):
        # Dispatch to core or builder
        ...
```

Every method has full `self.` access to all state (tile_cache, step, my_team_cache,
etc.) because they're all on the same class. But each file is small, focused, and
independently editable.

---

## CRITICAL: Position Object Overhead

`Position(x, y)` is NOT a cheap Python constructor — it crosses the FFI boundary
into Rust. Creating hundreds of Position objects per turn WILL cause TLE.

**Rules:**
- A* must use `(x,y)` tuples internally, NEVER Position objects
- Frontier search must use `(x,y)` tuples for candidates, not Positions
- Neighbor generation must use tuple arithmetic `(x+dx, y+dy)`, not `Position.add()`
- `get_nearby_tiles()` returns Position objects — convert to tuples ONCE:
  ```python
  tiles = [(t.x, t.y) for t in ct.get_nearby_tiles()]
  ```
  Then never touch the Position objects again.
- Only create a Position object at the exact moment you need to pass it to
  an engine API call (ct.move, ct.build_road, ct.get_tile_env, etc.)

**Bug Fixing**
 When fixing a bug, try not to apply a bandaid fix, instead, try to diagnose and fix the root cause e.g. If a bot is stuck, don't add a stuck detector and redirect the bot, instead, try to diagnose the root cause of why the bot is stuck, and fix the root cause instead, otherwise it will just keep getting stuck. Same with oscillation, don't add an oscillation detector, figure out why it's oscillating and prevent it from doing so in the first place. Feel free to ask me to test specific things if needed.
 
**Safe pattern:**
```python
# Good — tuple arithmetic in loops
for dx, dy in DIRS:
    nx, ny = cx + dx, cy + dy
    if (nx, ny) in tile_cache: ...

# Bad — Position creation in loops
for d in DIRECTIONS:
    neighbor = pos.add(d)  # FFI call to create Position
```

---

## TLE Diagnosis

If bots are TLEing, add timing to identify the bottleneck:

```python
import time, sys

def run_builder(self, ct):
    t0 = time.perf_counter()
    self._scan_turn(ct)
    t1 = time.perf_counter()
    # ... state machine logic ...
    t2 = time.perf_counter()
    print(f"[{self.corner}] scan={1000*(t1-t0):.1f}ms logic={1000*(t2-t1):.1f}ms", file=sys.stderr)
```

**Common TLE causes (in order of likelihood):**
1. **Position objects in loops** — A*, frontier search, or neighbor gen creating
   hundreds of Positions per turn. Fix: use (x,y) tuples everywhere.
2. **get_tile_env() on all tiles** — newly-visible optimization not working.
   Should only call on ~15 new tiles per move, not all ~70.
3. **Frontier search every turn** — should only recompute when target reached
   or invalidated, not while following a path.
4. **Multiple _scan_turn calls** — must run exactly once per turn.
5. **can_move/can_build checks in loops** — each is an API call. Don't try
   all 8 directions if you already know which one to use.

---

## Server Benchmark Results

We ran a dedicated benchmark bot on the actual competition servers to measure
exact operation costs. These numbers are from the server environment (not local
Mac), which is what matters for TLE.

### Operation Costs (measured on server)

| Operation | Cost | Notes |
|---|---|---|
| Baseline (get_position + get_team) | ~4μs | Minimum per-turn overhead |
| `get_tile_env()` | ~0.9μs | FFI call, cached after first query |
| `Position(x, y)` constructor | ~0.4μs | FFI call — avoid in loops |
| `dict.get()` lookup | ~0.25μs | Pure Python, no FFI |
| `set` membership (`in`) | ~0.23μs | Pure Python, no FFI |
| `heapq` push/pop | ~0.33μs | Pure Python, A* inner loop |
| `draw_indicator_line()` | ~10μs | Expensive — gate behind DEBUG flag |
| Full scan (tiles + buildings + units) | ~200-400μs | Depends on visible entities |
| A* 100 nodes | ~100μs | Pure Python dict/heap work |
| A* 500 nodes | ~400μs | Approaching budget limit |
| A* 2000 nodes | ~1200μs | Leaves almost no budget for anything else |

### Key Takeaway

**The bottleneck is total operation count, not individual call cost.** FFI calls
(~0.9μs) are cheap individually, but pure Python operations (dict lookups, set
checks, heap ops) at ~0.25μs each add up fast. On a 2ms budget:

- 1000 dict lookups = 250μs (12.5% of budget)
- 500 heap operations = 165μs (8% of budget)
- A* with 500 nodes = ~400μs (20% of budget)
- Full scan + A* + frontier search can easily hit 1500μs

**Practical limits per turn:**
- A* should stay under ~500 nodes (keep `ASTAR_MAX_NODES = 800` as safety cap)
- Frontier search should iterate `_last_vision_set` (~70 tiles), not full `tile_cache`
- Only recompute paths/frontiers when the current target is reached or invalidated
- Debug drawing must be gated behind `DEBUG` flag — 10 draw calls = 100μs wasted

---

## Symmetry Mirroring Optimisation

Maps in Battlecode are always symmetrical — diagonal, vertical, or horizontal.
Every `get_tile_env()` call reveals one tile, but symmetry means the mirror tile
has the **same terrain for free**. This effectively doubles our map knowledge
with zero extra API calls.

### How It Works

1. **Detect symmetry type** from core position at game start:
   - Core on the horizontal midline → `horiz` (left-right mirror)
   - Core on the vertical midline → `vert` (top-bottom mirror)
   - Otherwise → `diag` (180° rotation, most common)

2. **Mirror every `get_tile_env()` result** during `_scan_turn()`:
   ```python
   env = ct.get_tile_env(Position(xy[0], xy[1]))
   self.tile_cache[xy] = (env, None, None, None)

   # Free mirror — same terrain, zero API calls
   mxy = mirror_xy(xy, symmetry, map_w, map_h)
   self.tile_cache[mxy] = (env, None, None, None)
   ```

3. **Track mirrored tiles** in `_mirror_src` set so we know which cache entries
   are predictions vs real observations.

### Mirror Functions (pure tuple arithmetic, no FFI)

```python
def mirror_xy(xy, symmetry, map_w, map_h):
    x, y = xy
    if symmetry == 'vert':   return (x, map_h - 1 - y)
    if symmetry == 'horiz':  return (map_w - 1 - x, y)
    return (map_w - 1 - x, map_h - 1 - y)   # diag
```

### Fallback: Wrong Symmetry Prediction

The initial guess can be wrong. When a bot finally observes a mirrored tile and
the real terrain doesn't match the prediction:

1. **Purge** all mirrored tiles from `tile_cache` and `known_walls`
2. **Try the next symmetry type** (diag → vert → horiz)
3. If all three fail, **disable mirroring** entirely

This is tracked in `_handle_bad_symmetry()` in `scanning.py`. The fallback is
safe because mirrored tiles are never used for building decisions — only for
pathfinding and frontier search. A wrong prediction just means A* might pick
a suboptimal path, which self-corrects once the bot gets closer and sees the
real terrain.

### Why This Matters

- On a 50×50 map, a bot sees ~70 tiles per turn but the map has 2,500 tiles.
  Without mirroring, exploring the full map takes hundreds of turns.
- With mirroring, every tile explored reveals its mirror — the bot effectively
  knows ~2× as much of the map at all times.
- Frontier search benefits most: mirrored walls eliminate phantom frontiers on
  the far side of the map, so bots don't waste time exploring already-known areas.
- A* pathfinding to distant targets (like the core) gets better cost estimates
  because more of the map is in `tile_cache`.

---

## THE CRITICAL RULE: Tile Cache Architecture

The #1 cause of TLE (time limit exceeded) is making too many game engine API calls.
Each call (get_tile_env, get_tile_building_id, get_entity_type, get_team, etc.)
crosses a Python→Rust FFI boundary at ~10-20μs per call. On a 2ms budget,
you can afford ~100-150 calls total. Every call must be intentional.

### Cache Design
`self.tile_cache` is a dict mapping `(x,y)` → `(env, bid, etype, team)`.
It is populated once per turn in `_scan_turn()`. EVERY other function —
pathfinding, ore detection, bridge targeting, protect checks, frontier search —
reads from this dict with **zero API calls**.

### Populating the Cache Efficiently (_scan_turn)

Use batch queries to minimize API calls:

```
Step 1: get_nearby_tiles()               → list of visible positions    [1 call]
Step 2: get_tile_env(pos) per NEW tile   → terrain type                 [~15 calls if moved, 0 if stationary]
Step 3: get_nearby_buildings()           → all building IDs at once     [1 call]
Step 4: Per building:
          get_position()                                                [1 call each]
          get_entity_type()                                             [1 call each]
          get_team()                                                    [1 call each]
          get_bridge_target() if bridge                                 [1 extra call]
Step 5: get_nearby_units()               → all unit IDs at once         [1 call]
Step 6: Per unit:
          get_position()                                                [1 call each]
          get_team()                                                    [1 call each]
Step 7: Merge into tile_cache: (x,y) → (env, bid, etype, team)
        Also populate: bot_pos_cache, bridge_target_cache, known_walls
```

### Key Optimizations

**1. Only query `get_tile_env()` on NEWLY VISIBLE tiles.**
Terrain (EMPTY, WALL, ORE) never changes. When a bot moves 1 tile, ~70% of
visible tiles were already visible last turn. Track `last_vision_set` (a set of
`(x,y)` tuples) and only call `get_tile_env()` on tiles in
`current_vision - last_vision_set`. For tiles already in `tile_cache`, reuse the
cached env value. This is the single biggest optimization (~65 fewer calls per
turn when moving).

```python
current_vision = set((t.x, t.y) for t in tiles)
newly_visible = current_vision - self._last_vision_set
self._last_vision_set = current_vision

for xy in newly_visible:
    env = ct.get_tile_env(Position(xy[0], xy[1]))
    # ... cache it

# For tiles in current_vision but NOT newly_visible:
# env is already in tile_cache — skip the API call
```

**2. Skip `get_tile_env()` on known walls.**
Walls never change. If `(x,y)` is already in `known_walls`, skip the API call
entirely. Saves 15-30 calls on wall-heavy maps.

```python
for xy in newly_visible:
    if xy in self.known_walls:
        continue  # Wall forever — skip
    env = ct.get_tile_env(...)
```

**3. Use `get_nearby_buildings()` instead of per-tile `get_tile_building_id()`.**
One batch call replaces ~60-80 individual calls. Then iterate the returned IDs
with `get_position`/`get_entity_type`/`get_team` (3 calls per building, ~20
buildings = ~60 calls). Much cheaper than querying every tile.

**4. Use `get_nearby_units()` for bot positions.**
One batch call replaces checking `get_tile_builder_bot_id()` per walkable tile.
Only need `get_position()` + `get_team()` per unit (~4 bots × 2 = 8 calls).
Store in `self.bot_pos_cache: dict[(x,y)] = bot_id`.

**5. Cache `get_nearby_tiles()` positions when stationary.**
The position list only changes when the bot moves. If the bot hasn't moved since
last turn, reuse the position list. Still re-query buildings/units (those change),
but skip the tile position enumeration.

```python
my_pos = ct.get_position()
if (my_pos.x, my_pos.y) == self._last_pos:
    tiles = self._last_tiles  # Reuse
else:
    tiles = list(ct.get_nearby_tiles())
    self._last_tiles = tiles
    self._last_pos = (my_pos.x, my_pos.y)
```

### Per-Turn API Budget (after all optimizations)

| Scenario | API Calls |
|---|---|
| Bot moved 1 tile | ~65-80 calls |
| Bot stationary | ~40-50 calls |
| Bot stationary, no new buildings | ~5-10 calls |

---

## A* Pathfinding Rules

- `astar_cached()` reads ONLY from `tile_cache` — **zero API calls**
- Uses `(x,y)` int tuples internally, NOT Position objects (avoids FFI overhead
  from Position constructor)
- Only converts to Position at point of use (e.g., when calling `ct.move()`)
- Returns `list[(x,y)]` not `list[Position]`
- Safety cap: `ASTAR_MAX_NODES = 2000` (never hit on maps ≤50×50)
- Ore tiles with buildings on them (harvesters, barriers) = impassable
- For exploration: consider greedy best-first search (heuristic only, no g-cost)
  since we don't need optimal paths to frontiers — just any reachable path.
  Expands far fewer nodes than full A*.

### Two A* variants with DIFFERENT cost models:

**8-directional A* (bot movement):**
Used for pathfinding the bot itself — exploration, going to ore, etc.

| Tile type | Cost |
|---|---|
| Allied walkable (conveyor, road, splitter, core) | 1 |
| Empty / ore / marker (need to build road) | 2 |
| Unknown (not in cache) | 3 |
| Wall / non-walkable building | impassable |

**Cardinal-only A* (conveyor chain routing):**
Used for planning where to build the conveyor chain. Different cost model
because allied conveyors must be traversable — the chain builder needs to
consider routing through existing networks (merge or bridge over).

| Tile type | Cost | Reason |
|---|---|---|
| Empty / road / marker | 2 | Will build conveyor here |
| Allied conveyor / splitter | 4 | Prefer empty, but allow routing through |
| Allied barrier | 6 | Can destroy, but costly |
| Unknown (not in cache) | 3 | Assume traversable |
| Wall / enemy building / harvester | impassable | |

**Why allied conveyors are cost 4, not impassable:** If A* treats conveyors as
walls, it routes around existing networks even when merging would save 15 tiles.
Cost 4 means A* prefers building on empty tiles (cost 2) when routes are similar,
but will route through an existing network when the detour is significantly longer.

When the chain builder follows the cardinal A* path and reaches a tile with an
existing allied conveyor, it enters `observe_conveyor` state instead of building.
If spare capacity → merge. If full → bridge over.

---

## Other Caching Rules

- Cache `ct.get_team()` once per turn in `self.my_team_cache` (set at start of
  `run_builder()`). All team comparisons use this cached value.
- Cache `ct.get_nearby_tiles()` result once per turn (or reuse when stationary).
- Cache bridge targets in `self.bridge_target_cache: dict[(x,y)] = (target_x, target_y)`
  during `_scan_turn` (one `get_bridge_target()` call per allied bridge).
- `known_walls` is a permanent set — walls never change. Persists across turns,
  even for tiles no longer in vision.
- `_find_frontier_target()` only recomputes when the current frontier is reached
  or invalidated — NOT every turn while following a path to an existing frontier.

---

## Conveyor Network Capacity Model

Conveyor chains form networks feeding into splitters at the core. Each network
supports up to **4 harvesters** (1 stack/turn capacity, each harvester outputs
1 stack per 4 turns). The core has 12 cardinally adjacent tiles (3 per side),
allowing up to 12 independent networks.

**When a chain encounters an existing allied conveyor during chain-to-core:**
The bot enters an `observe_conveyor` state and watches the conveyor for up to
4 turns using `get_stored_resource(building_id)`:
- If conveyor is **ever empty** (returns None) → network has spare capacity →
  merge into it (connect your chain, skip building to core)
- If conveyor is **always full** for 4 turns → network is saturated →
  bridge over it and continue your own chain

`get_stored_resource()` is the ONLY API call made during the observe state.
This is a small, bounded cost (1 call per turn for up to 4 turns).

**When reaching the core:** prefer faces without existing splitters. If all
faces have splitters, observe existing splitters the same way to find one
with spare capacity.

**Do NOT use MCMF or complex graph analysis.** The observation check is simpler,
more reliable, and measures actual runtime capacity instead of theoretical capacity.

---

## Testing

```bash
cambc run bot bot              # Run against itself (resolves to bots/bot/)
cambc run bot bot maps/X.map26 # Specific map
cambc watch                       # Watch replay (visual — human only)
```

Test on BOTH small (20×20) and large (50×50) maps. Small maps have different
dynamics — less exploration needed, bots encounter enemies faster, constants like
`INITIAL_EXPLORE_RADIUS` and `FRONTIER_SCAN_RADIUS` should not exceed map size.

---

## Debugging

**Claude Code can see:** terminal output from `cambc run` (winner, errors, TLEs)
and anything printed to stderr.

**Claude Code CANNOT see:** the visual replay (`cambc watch` opens a browser GUI),
stdout prints (captured into replay file only), or debug indicators
(`draw_indicator_line`/`draw_indicator_dot`).

### Logging Rules

**Always use stderr for debug logging:**
```python
import sys
print(f"[{self.corner}] msg", file=sys.stderr)
```

**NEVER use bare `print()` for debugging** — stdout goes to the replay file and
is invisible in the terminal.

### What to Log

Log state transitions, not every turn. Only print when something *changes*:

```python
# Step transitions
if self.step != self._prev_step:
    print(f"[{self.corner}] {self._prev_step} -> {self.step} at ({my_pos.x},{my_pos.y})", file=sys.stderr)
    self._prev_step = self.step

# Stuck detection firing
print(f"[{self.corner}] STUCK in {self.step} at ({my_pos.x},{my_pos.y}) for {self.economy_stuck_turns}t", file=sys.stderr)

# Harvester placement
print(f"[{self.corner}] harvester #{self.harvesters_built} at ({target.x},{target.y})", file=sys.stderr)

# Chain progress
print(f"[{self.corner}] chain at ({self.chain_pos.x},{self.chain_pos.y}) dist={d:.1f}", file=sys.stderr)

# Pathfinding failure
print(f"[{self.corner}] A* failed: ({start.x},{start.y})->({goal.x},{goal.y})", file=sys.stderr)

# Barrier busting
print(f"[{self.corner}] busting barrier at ({bp.x},{bp.y}), detour={len(around)} vs through={len(through)}", file=sys.stderr)

# Oscillation detected
print(f"[{self.corner}] OSCILLATION at ({my_pos.x},{my_pos.y}), history={self.pos_history}", file=sys.stderr)
```

### Debugging Workflow

1. Claude Code runs `cambc run bot bot` and reads stderr output
2. User watches replay with `cambc watch` and describes what they see:
   - "Bot NW is stuck oscillating between (5,3) and (5,4)"
   - "Chain from harvester at (15,8) stops at (12,6), no bridge built"
   - "All 4 bots explore north, nobody goes south"
3. Claude Code correlates user description with stderr logs and fixes the issue
4. Repeat

This gives Claude Code the *state machine data* (what step, what position, what
failed) while the user provides the *spatial context* (what it looks like on the
map). Together they diagnose any bug.

---

## GameConstants — Never Hardcode Game Values

All base costs, radii, HP values, and game limits are available via `GameConstants`.
**Never hardcode these** — they change between patches (e.g., bridge scaling was
nerfed from 5% to 10%).

```python
from cambc import GameConstants as GC

# Base costs are (titanium, axionite) tuples
GC.HARVESTER_BASE_COST        # (20, 0)
GC.CONVEYOR_BASE_COST         # (3, 0)
GC.BRIDGE_BASE_COST           # (20, 0)
GC.SPLITTER_BASE_COST         # (6, 0)
GC.BARRIER_BASE_COST          # (3, 0)
GC.ROAD_BASE_COST             # (1, 0)
GC.BUILDER_BOT_BASE_COST      # (30, 0)

# Radii (squared)
GC.BUILDER_BOT_VISION_RADIUS_SQ  # 20
GC.BRIDGE_TARGET_RADIUS_SQ       # 9
GC.CORE_VISION_RADIUS_SQ         # 36
GC.CORE_ACTION_RADIUS_SQ         # 8
GC.ACTION_RADIUS_SQ              # 2

# Game limits
GC.MAX_TURNS                  # 2000
GC.MAX_TEAM_UNITS             # 50
GC.STACK_SIZE                  # 10
GC.STARTING_TITANIUM           # 500
GC.PASSIVE_TITANIUM_AMOUNT     # 10
GC.PASSIVE_TITANIUM_INTERVAL   # 4
```

Use these in `constants.py` so all game values come from one place.

---

## Code Conventions

- State machine steps are methods prefixed with `_step_`: `_step_explore`,
  `_step_build_bridge`, `_step_chain_to_core`, etc.
- Never call `ct.get_team()` directly — use `self.my_team_cache`
- Never iterate `ct.get_nearby_tiles()` directly — use cached version
- Never call `ct.get_tile_env()` or `ct.get_tile_building_id()` outside
  `_scan_turn()` — read from `self.tile_cache`
- Never create Position objects in tight loops or inside A* — use `(x,y)` tuples
- **Never hardcode base costs, radii, or game limits** — use `GameConstants`
- All tunable numbers go in `constants.py`, never magic numbers in code
- All imports must be top-level (engine requirement — no imports inside `run()`)
- Only Python stdlib available — no numpy, scipy, or external packages