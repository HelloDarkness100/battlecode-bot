# Strategy

## ARCHITECTURE & PERFORMANCE

Maps range from **20×20 to 50×50 tiles**. All quantities and radii should be tuned for the smallest maps — what works on 20×20 will work on 50×50 but not always vice versa.

### Tile Cache System
Every turn, each builder bot performs a vision scan (`_scan_turn`) that populates a persistent `tile_cache` dict: `(x,y)` → `(env, building_id, entity_type, team)`. All subsequent logic — pathfinding, ore detection, bridge targeting, protect checks, etc. — reads from this cache with **zero game-engine API calls**.

#### Cache Population (_scan_turn) — Batch Queries

```
Step 1: get_nearby_tiles()               → visible positions            [1 call, cached when stationary]
Step 2: get_tile_env(pos) per NEW tile   → terrain type                 [~15 calls if moved, 0 if not]
Step 3: get_nearby_buildings()           → all building IDs at once     [1 call]
Step 4: Per building:
          get_position() + get_entity_type() + get_team()               [3 calls each]
          get_bridge_target() for bridges                               [1 extra call]
Step 5: get_nearby_units()               → all unit IDs at once         [1 call]
Step 6: Per unit:
          get_position() + get_team()                                   [2 calls each]
Step 7: Merge into tile_cache
```

#### Key Scan Optimizations

1. **Only `get_tile_env()` on newly visible tiles.** Terrain (EMPTY, WALL, ORE) never changes. Track `last_vision_set` — only query tiles in `new_tiles - last_vision_set`. Tiles already in `tile_cache` keep their cached env. Saves ~65 calls per turn when moving.

2. **Skip known walls entirely.** If `(x,y)` is in `known_walls`, don't call `get_tile_env()` at all. Walls never change. Saves 15-30 calls on wall-heavy maps.

3. **Batch building/unit queries.** `get_nearby_buildings()` replaces 60-80 individual `get_tile_building_id()` calls. `get_nearby_units()` replaces per-tile `get_tile_builder_bot_id()` calls.

4. **Cache tile positions when stationary.** `get_nearby_tiles()` returns the same positions if the bot hasn't moved. Reuse the list, only re-query buildings/units (those change).

#### Per-Turn API Budget

| Scenario | API Calls |
|---|---|
| Bot moved 1 tile | ~65-80 |
| Bot stationary | ~40-50 |
| Bot stationary, no building changes | ~5-10 |

### Pathfinding: Cache-Powered A*
All pathfinding uses `astar_cached()`, which reads entirely from `tile_cache` and `known_walls` with **zero game-engine API calls**.

Key features:
- Uses `(x,y)` int tuples internally — never creates Position objects during search. Only converts to Position at point of use (e.g., `ct.move()`). Returns `list[(x,y)]` not `list[Position]`.
- Safety cap: `ASTAR_MAX_NODES = 2000` — generous for any map up to 50×50
- `known_walls`: persistent set of confirmed wall positions (survives across turns, even outside current vision)
- `extra_walls`: temporary walls injected per-search (bot positions, oscillation walls)
- `allow_positions`: positions treated as passable despite having non-walkable buildings
- Ore tiles with buildings (harvesters, barriers) = impassable

**Two A* variants with different cost models:**

**8-directional A* (bot movement):** For pathfinding the bot itself (exploration, going to ore, etc.)
- Allied walkable (conveyor, road, splitter, core) = cost 1
- Empty / ore / marker (need road) = cost 2
- Allied barrier = cost 3 (economy bots can bust through — destroy, road, move, restore)
- Unknown = cost 3
- Wall / non-walkable building (except allied barrier) = impassable

**Note on barrier busting (economy bots only):** When an economy bot's path hits an allied barrier, it executes a multi-step pass-through: destroy barrier (free) → build road (action cooldown) → move onto road → once past, destroy road (free) → rebuild barrier (action cooldown). Patrol bots treat allied barriers as impassable during patrol mode (they can bust during non-patrol pathfinding like repair or turret response).

**Cardinal-only A* (conveyor chain routing):** For planning where to build conveyor chains. Allied conveyors must be traversable so the chain builder can consider routing through existing networks (merge or bridge over).
- Empty / road / marker (will build conveyor) = cost 2
- Allied conveyor / splitter (prefer empty, but allow routing through) = cost 4
- Allied barrier = impassable (use optimal_bridge to jump over — never build conveyors through barriers)
- Unknown = cost 3
- Wall / enemy building / harvester = impassable

**Barrier rules for chain-to-core:** If a conveyor's target tile is an allied barrier → enter `optimal_bridge` state (same as walls). Bridges must NOT target allied barriers as valid landing tiles.

For **long-range pathfinding into unexplored territory** (disruptors heading to the enemy core): use **greedy best-first search**. Treats unknown tiles as passable (optimistic) — the bot actively explores toward the goal rather than stopping at the known region boundary. Heuristic-only, no g-cost — dramatically fewer node expansions than A*. Based on testing, greedy is more reliable than frontier-guided A* at actually reaching the enemy core on wall-heavy maps. See the "Disruptor Pathfinding" section below.

The path is computed once, followed step by step, and recomputed only when blocked or the goal changes.

### Disruptor Pathfinding (Greedy Best-First Search)

When the goal is far away and the map between the bot and goal is mostly unknown, full A* wastes work. Testing showed that **greedy best-first search** is the most reliable pathfinding approach for disruptors — it finds the enemy core on more maps than frontier-guided A*, even though frontier approaches often produce shorter paths when they succeed.

**Concept:** At each step, expand the tile with the lowest Euclidean² distance to the goal. No g-cost, no path reconstruction over the whole graph — just heuristic guidance.

**Key difference from other uses of greedy in this bot:**
- **Unknown tiles are treated as PASSABLE** (optimistic). This lets the bot actively explore toward the goal rather than stopping at the known region boundary.
- Walls (`known_walls`, `Environment.WALL`) are impassable.
- Non-walkable buildings (allied or enemy) are impassable.
- Allied walkable buildings are passable.

**Reference implementation:** See `bots/path_greedy_bot/common.py` and `bots/path_greedy_bot/main.py` — the greedy implementation there was validated against BFS/A*/frontier in pathfinding tests. The disruptor should port this implementation (adapting only the action loop to handle attack/build tasks).

**Repath triggers:**
1. Bot reaches the end of the current path
2. Path becomes invalid (a tile on the path is now a wall we can see)
3. Goal changes (symmetry advance)

**Failure handling:**
- If greedy returns no path AND all newly revealed information doesn't unlock a route → advance symmetry
- Bot continues counting Step 2 turns; if 300 turns elapse without reaching the core, self-destruct

No frontier set maintenance. No A* fallback needed (greedy with optimistic unknowns handles wall-heavy maps).



### Frontier Caching
`_find_frontier_target()` only recomputes when the current frontier is reached, invalidated, or the bot has no target. It does NOT recompute every turn while following a path — that's wasted computation. The frontier scan uses a local window (`FRONTIER_SCAN_RADIUS`) around the bot, not the full explore radius.

### Stuck Detection

**Path-level stuck detection**: If the bot hasn't moved for 2+ turns while following a path, it sets `avoid_bots = True` and returns 'blocked', triggering a repath that injects all nearby bot positions as temporary walls.

**Bot blocking detection**: When `can_move` fails despite no move cooldown, the bot checks `bot_pos_cache` for a builder bot on the next tile. If found, sets `avoid_bots = True` and repaths.

**Oscillation detection**: A ring buffer of length `POSITION_HISTORY_LEN = 10` records the bot's position each turn. If the current position appears `OSCILLATION_THRESHOLD = 3` or more times in the buffer, all other recent positions are added to `oscillation_walls` as temporary walls. The path is cleared, and the next A* call blocks those tiles. Oscillation walls are cleared after one repath.

**Top-level economy stuck detection**: After each step dispatch, checks if both position and step are unchanged for 2+ turns. If not waiting for resources, forces recovery — resets to `explore` with all stale state cleared.

### TLE Detection & Recovery

Each turn, `self.turn_completed` is set `False` at the very start of `run()` and `True` at the very end. If the engine kills execution mid-turn (2ms exceeded), it stays `False`. Next turn, this is detected as a TLE.

**State variables:**
- `turn_completed: bool` — set False at start, True at end of run()
- `tle_recovery_turns: int` — turns remaining in recovery mode (0 = normal)
- `consecutive_tles: int` — consecutive TLE count (for escalating recovery)

**On TLE detected** (previous `turn_completed` was False):
1. Increment `consecutive_tles`
2. Set `tle_recovery_turns = 3 + consecutive_tles` (escalates: 4, 5, 6... turns)
3. Log to stderr: `[{corner}] TLE detected, recovery for {tle_recovery_turns} turns`

**During recovery mode** (`tle_recovery_turns > 0`):
Skip all expensive optional operations:
- Skip sentinel placement scan (7× `get_attackable_tiles_from` calls)
- Skip frontier recomputation (use existing frontier target or just follow current path)
- Skip full vision ore scanning (don't check every visible tile for unclaimed ore)
- Skip `observe_conveyor` (don't wait 4 turns watching a conveyor — bridge over immediately via `optimal_bridge`)
- Skip monotonic A* recomputation if a path already exists

Core operations that ALWAYS run even in recovery:
- `_scan_turn` (batch queries — the essential ~40-80 API calls)
- `_follow_path` (just `can_move` + `move` — 2 calls)
- Building conveyors/harvesters (the bot's primary job)
- Path following and basic movement
- `optimal_bridge` (still needed to keep chains alive)

**On successful turn** (no TLE):
- Reset `consecutive_tles = 0`
- Decrement `tle_recovery_turns` by 1 (if > 0)

**Implementation pattern:**
```python
def run_builder(self, ct):
    # TLE detection
    if not self.turn_completed:
        self.consecutive_tles += 1
        self.tle_recovery_turns = 3 + self.consecutive_tles
        print(f"[{self.corner}] TLE! recovery={self.tle_recovery_turns}t", file=sys.stderr)
    else:
        self.consecutive_tles = 0
    self.turn_completed = False
    
    # ... normal logic, checking self.tle_recovery_turns > 0 before expensive ops ...
    
    if self.tle_recovery_turns > 0:
        self.tle_recovery_turns -= 1
    self.turn_completed = True
```

### Allied Building Cache (for Repair Detection)

A separate persistent cache tracks every allied conveyor, bridge, harvester, foundry, and splitter that any bot has ever seen. This cache is **never cleared** — our strategy is non-destructive (our own bots never destroy allied infrastructure), so any disappearance means the enemy destroyed it.

**Cache structure:** `self.building_cache: dict[(x,y)] → (entity_type, direction_or_target)`
- For conveyors/splitters: store `(EntityType.CONVEYOR, Direction.SOUTH)` — the facing direction
- For bridges: store `(EntityType.BRIDGE, (target_x, target_y))` — the bridge target position
- For harvesters/foundries: store `(EntityType.HARVESTER, None)` — no direction needed

**Updating the cache:** During `_scan_turn`, after populating `tile_cache`, also update `building_cache`:
- For every allied conveyor/bridge/harvester/foundry/splitter in vision: add or update its entry in `building_cache` with current type and direction/target
- This means the cache grows over time as the bot explores and builds — each bot remembers at minimum every building it personally placed, plus any allied buildings it has ever seen

**Detecting destroyed buildings:** During exploration or patrolling, compare visible tiles to `building_cache`:
- For each `(x, y)` in `building_cache` that is currently in vision, check what `tile_cache` says is there now
- If the tile still has the same allied building type → no action needed
- If the tile has a **different** allied building or different direction → the enemy destroyed the original and an allied bot has already rebuilt something. **Update** `building_cache` with the new info. No repair needed.
- If the tile has an **allied road** → enemy destroyed the building and someone built a road over it. **Needs repair** — destroy the road and rebuild the cached building.
- If the tile is **empty** → enemy destroyed the building. **Needs repair** — build the cached building.
- If the tile has an **enemy walkable building** (road, conveyor, splitter) → enemy definitely destroyed it. **Needs repair** — pathfind onto tile, destroy enemy building, rebuild cached building.
- If the tile has an **enemy non-walkable building** (barrier, harvester, turret) → cannot repair (we can't destroy non-walkable enemy buildings from adjacent). **Skip** — remove from repair queue but keep in `building_cache` in case it's later cleared.

**Repair queue:** `self.pending_repairs: list[(x, y)]` — positions that need repair, detected during scanning. Processed when the bot is in an appropriate state (explore for economy bots, any time for patrol bots).

### Conveyor Network Capacity Model
Each conveyor network is a tree/chain of conveyors feeding into one splitter at the core. A conveyor passes 1 stack per turn. A harvester outputs 1 stack every 4 turns. Therefore **each network supports exactly 4 harvesters** before saturating.

The core is 3×3, giving 12 cardinally adjacent tiles (3 per side). Each can hold a splitter, so the theoretical maximum is 12 networks / 48 harvesters.

**Detecting network capacity — the observation check:**
When building a conveyor chain back to the core and encountering an existing allied conveyor, the bot must determine if that network has spare capacity:

1. Stand adjacent to the existing conveyor
2. Observe it for up to 4 turns using `get_stored_resource(building_id)`
3. If the conveyor is **ever empty** (returns None) during those 4 turns → the network has spare capacity → **merge** your chain into it by connecting your conveyor to it
4. If the conveyor is **always full** for 4 consecutive turns → the network is saturated → **bridge over** it and continue building your own chain to the core

This replaces the MCMF solver with a simple, reliable runtime check that requires zero graph analysis.

**When reaching the core directly (no existing network encountered):**
Pick the best cardinally adjacent core tile for the new splitter. Prefer tiles on sides that don't already have splitters. If all sides have splitters, pick the side whose existing splitter's network has the fewest harvesters (can be estimated by observing fullness).

## MISCELLANEOUS INFO

Any time a quantity is mentioned, assume it is a variable that can be changed, and ensure everything works expectedly when the values are changed.

If you are trying to place a building but are unable to, check if an allied road or barrier is in that tile. If so, simply delete the tile (deleting allied roads/barriers does not cost an action cooldown). After destroying, invalidate the tile in the tile_cache.

Whenever it says "Spawn X bots", it means to spawn them in one at a time at the target tile for their role as soon as that tile is no longer occupied.

Note that all markers, whether enemy or allied, can be built over. If an enemy marker is in the way of building a road, harvester, bridge, etc., simply build on top of it.

All distances given are Euclidean distance.

Enemy Core Position — When pathfinding to the enemy core, the map is always reflected diagonally, vertically or horizontally. Calculate the position of the enemy core by mirroring our core's position. If centred in one plane but not the other, symmetry is determined by which axis is centred. If neither, attempt diagonal first, then vertical, then horizontal.

There are 9 possible places to spawn on the core, and each builder bot derives their role from the tile they are spawned on.

## Phase 1: Economy

Bots spawned on the top left, top right, bottom left and bottom right should be assigned the economy role. For Phase 1, the core should spawn 4 Economy bots on each of the 4 corner tiles.

**There is NO shell-building phase.** Bots spawn on the core and immediately leave to explore. Splitters are placed later, only when a chain actually reaches the core.

Economy bots follow this state machine:

### Step 1: Leave Core
When spawned, the bot immediately moves off the core:
1. Try to move in the direction corresponding to the bot's corner (NW bot moves northwest, etc.)
2. If blocked, try any direction that moves away from the core center
3. Build roads on empty tiles if needed to create an exit. Action cooldown and movement cooldown are separate, so you can place a road and move on the same turn
4. Transition to explore as soon as off the core

### Step 2: Explore (Frontier Exploration)
Explore using greedy nearest frontier detection within a local scan window (`FRONTIER_SCAN_RADIUS = 12` tiles around the bot). The exploration area is bounded by `INITIAL_EXPLORE_RADIUS = 15` tiles from the core, expanding by 1 tile every `EXPLORE_RADIUS_EXPAND_INTERVAL = 50` turns, up to `MAX_EXPLORE_RADIUS = 30`. Tiles beyond this radius are treated as walls.

**Corner directional bias**: To spread bots out and avoid all heading the same direction:
- NW bot → prefers north frontiers
- NE bot → prefers east frontiers
- SE bot → prefers south frontiers
- SW bot → prefers west frontiers

The bias is applied as a score modifier (`bias_strength = 50`) on frontier tiles in the preferred direction. It naturally fades as preferred frontiers are exhausted.

**Repair during exploration:** Each turn during exploration, after `_scan_turn` updates both `tile_cache` and `building_cache`, check `pending_repairs` for any repairs that are now in vision. Priority order during exploration is:
1. **Titanium ore** (Step 4) — always highest priority, ore drives the economy
2. **Repairs** — if no ore target, and `pending_repairs` is non-empty, pathfind to the closest repair and fix it (see repair procedure below)
3. **Frontier exploration** — if no ore and no repairs, continue exploring

Economy bots **never abandon a chain to repair** — if the bot is in `claim_ore`, `build_harvester`, or `chain_to_core` steps, repairs are deferred. The bot still updates `building_cache` and `pending_repairs` from vision in all steps, so repairs accumulate and are handled when the bot returns to exploration.

**Repair procedure** (shared by economy and patrol bots):
1. Pathfind to the repair position
2. Check what is currently on the tile (from `tile_cache`):
   - **Empty tile:** Build the cached building (conveyor with correct direction, bridge with correct target, harvester, etc.)
   - **Allied road:** Destroy the road (free), then build the cached building
   - **Enemy walkable building:** Pathfind onto the tile, attack with `ct.fire(my_pos)` until destroyed, then build the cached building
   - **Enemy non-walkable building (barrier, turret, harvester):** Cannot repair — remove from `pending_repairs`, keep in `building_cache` for future retry
   - **Different allied building:** Another bot already rebuilt something here — update `building_cache` with the new info, remove from `pending_repairs`
3. After repair, return to previous activity (exploration for economy, patrol for patrol bots)

### Step 3: Axionite Detection (During Exploration)
If an axionite ore comes into vision (read from tile_cache, zero API calls):
- **Remember the position** in `self.stored_axionite_pos`
- Continue exploring (do NOT pathfind to it)
- Each economy bot only stores 1 axionite position. After finding one, all future axionite ores are ignored during the first chain.

**Axionite is NOT a target during the first chain.** The bot must complete its first titanium chain before axionite becomes a valid ore target. The stored position is used so the bot can go straight to it after the first chain completes instead of re-exploring.

**Starting from the second chain onwards**, axionite ores are included in the ore scan (Step 4) alongside titanium. Each economy bot can build **1 axionite chain total** (tracked by `self.axionite_chain_done = False`). After completing an axionite chain, the bot ignores all axionite for the rest of the game.

No marker communication. No separate axionite bots. Axionite is fully integrated into economy bot logic.

### Step 4: Ore Detection (During Exploration)
Scan vision for ore targets. There are three types of valid targets — **unclaimed titanium ore**, **unclaimed axionite ore** (only after first chain), and **enemy harvesters on titanium ore** (stealable). All compete for the "closest ore" slot.

**Unclaimed titanium ore candidates:**
- Titanium ore without any harvester on it
- Skip if another allied builder bot is already ON the ore tile (from bot_pos_cache) — it's claimed
- Skip if another allied road is on the ore tile — another bot has likely claimed it
- **Skip enclosed ores**: If all 4 cardinal neighbors of the ore are walls or other ores, skip it — there is no valid tile to place a chain_start conveyor. This check is based on environment only (ignores buildings on top). Unknown neighbors are assumed accessible; the check is repeated at claim time when all neighbors are visible.

**Unclaimed axionite ore candidates** (only eligible after first titanium chain is complete AND `axionite_chain_done == False`):
- Axionite ore without any harvester on it
- Same skip logic as titanium (claimed, enclosed)
- If the bot has a `stored_axionite_pos` from Step 3, that position is included as a candidate even if not currently in vision (pathfind to it; if it's claimed when you arrive, return to explore)

**Enemy harvester candidates (stealable):**
- Titanium ore WITH an **enemy harvester** on it
- Skip if the enemy harvester already has an adjacent allied sentinel, conveyor, or bridge — it's already been stolen
- Skip if there are **zero available tiles**: an available tile is a cardinally adjacent tile that is either empty or has an enemy walkable building (road, conveyor, splitter, bridge). If all 4 cardinal tiles are walls, non-walkable buildings, or allied buildings, the harvester can't be stolen — skip it.

**Target selection:** Pick the closest candidate (any type) by Euclidean distance. ALWAYS prioritise the closest, regardless of type. If a closer target of any type appears in vision while pathfinding, switch to it.

**Full vision scanning**: While following a frontier path, the bot scans its entire vision for unclaimed ore (titanium + axionite if eligible) AND stealable enemy harvesters each turn. If a target is found, it immediately redirects.

If the closest target is **unclaimed titanium ore** → pathfind directly onto the ore tile → Step 5 (Claim Ore — titanium path)
- **Barrier on target ore:** If an allied barrier sits on the ore, the movement A* treats it as cost 3 (traversable). When the bot arrives, destroy the barrier (free), build a road on the ore, and continue to Step 5. Barriers adjacent to the ore that block cardinal tiles are also destroyed (free) during Step 5a.3/5a.4 as needed.

If the closest target is **unclaimed axionite ore** → pathfind directly onto the ore tile → Step 5c (Claim Ore — axionite path)

If the closest target is an **enemy harvester** → Step 4b (Steal Harvester)

**Why pathfind onto unclaimed ore?** Standing on the ore means all 4 cardinal directions are within build range. A road + builder bot on top is a clear "claimed" signal to other bots.

### Step 4b: Steal Enemy Harvester
The bot has identified a stealable enemy harvester on a titanium ore. The goal is to place a sentinel to destroy enemy infrastructure (and protect the stolen harvester), then optionally start a chain back to our core.

**Step 4b.1 — Find best sentinel tile:**
Collect all "available" cardinally adjacent tiles (empty or enemy walkable building). For each available tile, run the 7-direction sentinel scan (same as Step 5 sentinel scan — exclude the direction facing the harvester, scan the other 7 with `get_attackable_tiles_from`). Track:
- Does any direction hit the enemy core? (Highest priority)
- What's the max enemy building count across all 7 directions?

Pick the available tile with the **highest enemy building count** across its best direction. If tied, pick the one closest to the bot. This ensures the sentinel is placed where it can do the most damage.

**Step 4b.2 — Pathfind to sentinel tile:**
Pathfind to the chosen available tile. If it has an enemy walkable building (road, conveyor, splitter, bridge), stand on it and attack with `ct.fire(my_pos)` until destroyed.

**Step 4b.3 — Place sentinel:**
Move to an adjacent tile (any walkable tile next to the sentinel position), then build the sentinel on the cleared tile facing the calculated best direction. The sentinel will begin firing at enemy infrastructure using standard sentinel attack logic. Ammo is provided by the enemy harvester (it outputs to adjacent buildings — our sentinel is adjacent).

**Step 4b.4 — Check for chain opportunity:**
After placing the sentinel, re-check the harvester's cardinal tiles for remaining available tiles (empty or enemy walkable — the sentinel tile is no longer available since we just built on it).
- If **no available tiles remain** → the sentinel is enough. The harvester feeds our sentinel ammo while the sentinel destroys enemy infrastructure. Return to exploration (Step 2).
- If **at least one available tile remains** → we can chain this harvester to our core. Pathfind to an available tile. If it's an enemy walkable building, attack until destroyed. Then begin chain-to-core logic: place a conveyor/bridge facing toward the core and proceed to Step 7 (Chain to Core).

**Key points:**
- Sentinel placement always takes priority over chain start — even if we could chain, place the sentinel first
- The harvester remains an enemy harvester — we don't need to rebuild it. It outputs resources to adjacent buildings regardless of team. Our conveyor/splitter chain will carry those resources to our core.
- If the bot can't reach the sentinel tile (pathfinding gives up at `ORE_PATHFIND_GIVE_UP_DIST`), abandon and return to exploration

### Step 5: Claim Ore (Standing on Ore)
Once standing on the ore tile (with a road built on it from pathfinding):

**Distance check:** Compute Euclidean distance from ore to core center. This determines the defense strategy:
- **Within `DEFEND_HARVESTER_RANGE = 8` tiles** → Enhanced defense (barriers + gunner/sentinel) — see Step 5a
- **Beyond 8 tiles** → Normal defense (sentinel-only check) — see Step 5b

### Step 5a: Enhanced Defense (Close to Core)
The ore is within `DEFEND_HARVESTER_RANGE` of the core. Build full defense: conveyor + barriers + gunner or sentinel.

**Resource check:** Must afford harvester + conveyor chain + gunner (or sentinel if sentinel scan passes) + 2 barriers. Use `GameConstants` with current scale. If insufficient, wait on ore.

**Barrier/building prep:** If any cardinally adjacent tile has an allied barrier on it, destroy it (free) before evaluating tiles. Allied barriers on the ore itself should also be destroyed. This clears space for the defense layout.

**Step 5a.1 — Place conveyor:**
Same as normal — compute monotonic A* path, place conveyor on the cardinally adjacent tile determined by the path direction.

**Step 5a.2 — Determine defense tile (sentinel or gunner):**
After conveyor is placed, up to 3 cardinal tiles remain. Run the sentinel scan first:

**Sentinel check (same 7-direction scan as Step 5b):** Run on all remaining valid tiles. If the sentinel scan passes (enemy core found, or enemy buildings ≥ `SENTINEL_ATTACK_THRESHOLD`), mark that tile for sentinel and skip gunner.

**Gunner check (only if sentinel was NOT placed):** Check the **2 cardinally adjacent tiles perpendicular to the conveyor direction.** These tiles defend the conveyor line:
- Conveyor placed NORTH → check EAST and WEST tiles
- Conveyor placed EAST → check NORTH and SOUTH tiles
- Conveyor placed SOUTH → check EAST and WEST tiles
- Conveyor placed WEST → check NORTH and SOUTH tiles

A valid gunner tile is: empty, enemy walkable building, or allied barrier (destroy first). Use titanium ore tiles only as a **last resort**. Pick the first valid tile from the two candidates.

**Step 5a.3 — Place barriers on remaining tiles:**
After conveyor (1 tile) and sentinel/gunner (1 tile) are designated, check the remaining 2 cardinal tiles. For each:
- **Empty tile:** build barrier directly (bot is standing on the ore, within action radius)
- **Allied road:** destroy road (free), build barrier
- **Marker:** build barrier on top (markers can be overwritten)
- **Enemy walkable building (road, conveyor, splitter, bridge):** pathfind onto the tile, attack with `ct.fire(my_pos)` until destroyed, then move back to the ore tile and build barrier
- **Wall, allied building (non-road), enemy non-walkable:** skip — can't place barrier

**Step 5a.4 — Place gunner/sentinel:**
After barriers are placed, place the gunner or sentinel on its designated tile:
- If the tile has an enemy walkable building: pathfind to it, attack until destroyed, move off, build turret from adjacent
- If the tile has an allied barrier: destroy it (free), then move off and build turret from adjacent
- Gunner initial direction: face the direction of the conveyor (protecting the chain approach). The gunner will rotate as needed during runtime.
- Sentinel direction: determined by the 7-direction scan from Step 5a.2

**Step 5a.5 — Transition:**
Move onto the conveyor tile → destroy road on ore (free) → build harvester → proceed to chain_to_core.

### Step 5b: Normal Defense (Far from Core)
The ore is beyond `DEFEND_HARVESTER_RANGE` of the core. Sentinel-only check, no barriers, no gunner.

**Resource check**: Must afford harvester + conveyor chain. If sentinel scan passes, also include sentinel cost. If insufficient, wait on ore.

Then in a single position:
1. **Compute the monotonic A\* path** from the chain_start tile back to the core. Use the path's first step to determine the correct conveyor direction. If the path says the conveyor would point back at the ore tile, treat it as invalid and skip to bridge handling.
2. **Place the conveyor** on the cardinally adjacent tile determined by the path direction, after performing the standard conveyor target check. If invalid, skip — chain_to_core will handle it with a bridge.
3. **Conditionally place a sentinel** on another cardinally adjacent tile if ALL of the following are true:
   - A valid tile exists (no building, not a wall, prefer empty tiles over ore tiles)
   - There isn't already an allied sentinel cardinally adjacent to this ore
   - The **sentinel scan** determines it's worth placing

   **Sentinel scan — single pass, 7 directions:**
   Exclude the direction facing the harvester (sentinel can't accept ammo from facing direction). For each of the 7 remaining directions, call `get_attackable_tiles_from(sentinel_tile, direction, EntityType.SENTINEL)` and check:
   - **Enemy core on any attackable tile?** → stop immediately, place sentinel facing this direction
   - **Count enemy buildings** (exclude roads/markers). Track direction with highest count.
   
   After scanning: enemy core found → place; best count ≥ `SENTINEL_ATTACK_THRESHOLD` → place; otherwise → skip.

4. Both builds (conveyor + optional sentinel) happen from one position — no additional pathfinding.

After conveyor + defense placed, transition to build_harvester.

### Step 5c: Claim Axionite Ore (Standing on Ore)
Only reached when the bot's target was an axionite ore (from Step 4). Simpler than titanium — **no barriers, no gunners, no sentinels**.

**Resource check:** Must afford harvester + conveyor chain. Use `GameConstants` with current scale. If insufficient, wait.

Then in a single position:
1. **Compute the monotonic A\* path** from the chain_start tile back to the core (same as titanium).
2. **Place the conveyor** on the cardinally adjacent tile determined by the path direction (same target checks as titanium). If invalid, skip — chain_to_core handles it with a bridge.
3. **No defense buildings.** No sentinel scan, no gunner, no barriers.

After conveyor placed, transition to build_harvester (Step 6).

### Step 6: Build Harvester
The bot is still standing on the ore tile:
1. **Destroy the road** on the ore tile (free — destroying allied roads has no action cooldown). This prevents other bots from walking onto the ore while we move to chain_start.
2. **Move onto the conveyor / bridge** tile placed in Step 5/5c. If the bot can't move (e.g. another bot is blocking chain_start), wait up to 10 turns before abandoning the ore.
3. **Build a harvester** on the now-cleared ore tile
4. Scan for other ores in vision without harvesters (excluding enclosed ores), store their positions. **For axionite chains: only store axionite ores, skip titanium. For titanium chains: only store titanium ores, skip axionite.**
5. If a bridge was placed in step 5, pathfind to the target of the bridge and place a conveyor.

After harvester built, transition to chain_to_core:
- If this is a **titanium** harvester → Step 7 (Chain to Core — titanium)
- If this is an **axionite** harvester → Step 7b (Chain to Core — axionite)

### Step 7: Chain to core
If this is a builder bot's first time chaining to the core, it should NEVER merge into another conveyor network, it should ONLY direct itself directly to the core.
1. The first conveyor already exists (placed in Step 5 next to the harvester)
2. We will use a slightly modified version of A* for the path back to the core. This path must only move in cardinally adjacent directions, and the distance from the core must never increase, so each step should be getting closer to the core. If a step were to increase the distance from the core, (e.g. because it wants to follow a wall) it should calculate that next step without any walls / buildings (so it should go into the wall / building) This ensures we have a path that never gets moves away from the core. Before placing each conveyor however, we have to perform a series of checks about the tile the conveyor is targeting; this is how we will ensure we keep the chain alive even if our pathfinding goes into a wall / building.

Before placing the conveyor, consider what tile the conveyor would target if it were to be placed. If it targets:
   - **Empty / road / marker:** build the conveyor as planned, continue following the path
   - **Allied conveyor / splitter:** enter `observe_conveyor` state (see below), unless it is the builder bot's first chain to the core, where we don't want to merge into another conveyor network, enter `optimal_bridge` state instead (see below)
   - **Allied barrier:** Enter `optimal_bridge` state (barriers block resource flow — never build conveyors into them)
   - **Enemy walkable building (road, conveyor, splitter, ect)**: Walk onto tile, and use builder bot attack until the building is destroyed, then build your conveyor
   - **Wall tile, Non walkable buildings:** Enter `optimal_bridge` state (see below) 
Repeat this loop until we have placed the final conveyor facing towards the core, so resources flow to the core. Mark the chain as complete, go to next step
 
**Encountering an existing allied conveyor (observe_conveyor):**
The cardinal A* path may route through an existing conveyor network. When the chain builder reaches such a tile:
1. Get the building id of the targeted conveyor / splitter
2. Each turn, call `get_stored_resource(building_id)` on the conveyor
3. If the conveyor is **ever empty** (returns None) on any turn → **merge** (network has spare capacity). No need to wait all 4 turns — merge at the first empty observation.
4. If the conveyor is **always full** for 4 consecutive turns → **bridge over** (network saturated) → enter `optimal_bridge` state

**What happens after merging depends on the chain type:**
- **Titanium chain:** Mark chain as complete → Step 8
- **Axionite chain:** Do NOT mark complete. Instead, enter `build_foundry` state (Step 7b.1) — the bot must follow the merged network to its end and place a foundry there

**Building an optimal bridge (optimal_bridge):**
In this state, we don't want to build a conveyor for whatever reason (because it's directed into a wall, we don't want to merge into existing chain, ect.) Instead of building the conveyor, we will build a bridge.

The bridge's target must be to a **valid tile.**
    - Valid tiles only include: Allied core, empty tiles (no ores), roads (allied or enemy), or enemy walkable buildings (conveyors / splitters)
    - **NOT valid:** Allied barriers, walls, ores, allied non-walkable buildings, enclosed empty tiles
    - **Skip enclosed pockets**: Empty tiles where all 8 surrounding tiles are walls are not valid bridge targets (unreachable)
The builder bot should choose the valid tile closest to the core to target. Build the bridge with that target. If the target is the allied core, then mark the chain as complete, move to step 8. Otherwise, pathfind (using normal A*) to that target. If the target is an enemy walkable building, attack it using `ct.fire(position)` until it is destroyed, then build the conveyor. Otherwise, perform a conveyor target check like before. Build the conveyor if all is fine, otherwise you may need to go through observe_conveyor and / or optimal_bridge again.

### Step 8: Ores from tile cache
Check tile cache for ores without a harvester, pick the closest one to head towards. Include axionite ores if eligible (first chain complete AND `axionite_chain_done == False`). If none exist, go back to Step 2 (Exploration). Go to step 4 if any ore without harvesters come into vision. If you reach the tile cached ore and it has a harvester already, go back to step 2. If it doesn't, pathfind onto that ore and start from step 5 (titanium) or step 5c (axionite).

### Step 9: Role Transition
After building `MAX_HARVESTERS_PER_BOT = 6` harvesters and chaining them all, the bot can switch roles into a disruptor depending on the round (see Phase 3). Currently defaults to returning to exploration. If you haven't built `MAX_HARVESTERS_PER_BOT = 6`, return to step 2 Explore.

After completing an axionite chain, set `axionite_chain_done = True` and return to Step 2 (Exploration) — all future ore scans ignore axionite.

### Step 7b: Chain to Core — Axionite
This step is used when chaining an **axionite** harvester back toward the core. The chain-building logic is the same as titanium (Step 7 — same monotonic A*, same conveyor target checks) with these key differences:

1. **Axionite chains CAN merge** into any allied network (titanium or axionite) — no first-chain restriction.
2. **After merging:** do NOT mark chain complete. Instead enter `build_foundry` phase (Step 7b.1).
3. **Do NOT target the allied core** with bridges. Exclude all 9 core tiles from bridge valid targets.
4. **If the chain reaches the core WITHOUT merging:** use the direct foundry placement method (Step 7b.6 — same as old logic).

**Note:** "Conveyor" in this section refers to both regular conveyors and armoured conveyors.

---

**Step 7b.1 — Follow merged chain to end (build_foundry phase):**
After merging into an existing allied network, the bot must follow that network to its end to determine where to place the foundry.

**Following the chain:** Starting from the tile where you merged, follow the direction of conveyors (walk in their output direction) and pathfind to bridge targets. At each building:
- **Allied conveyor/armoured conveyor:** check what it targets (the tile in its facing direction). Move to that tile. Continue following.
- **Allied bridge:** get the bridge target from `bridge_target_cache`. Pathfind to the target tile. Continue following.
- **Allied splitter:** this is an end point (splitter already distributes). Go to Scenario 4.
- **Allied foundry:** this is an end point. Go to Scenario 4.

**End of chain detection:** The chain ends when the current conveyor/bridge targets one of:
- **The allied core** → Scenario 1
- **An allied foundry or splitter** → Scenario 4
- **An empty tile, allied road, or enemy walkable building** → Scenario 2 (conveyor) or Scenario 3 (bridge)

---

**Step 7b.2 — Scenario 1: Final building targets the core**

The final building in the chain is a conveyor or bridge whose target is the allied core. We need to intercept this connection and insert a foundry + splitter.

**Case A — Previous building is a bridge** (i.e., the final conveyor is the bridge's target tile):
1. Destroy the final conveyor/bridge (free — allied building)
2. Find an **available tile** cardinally adjacent to the tile where the final building was. Available = empty tile (preferred), or as a last resort an allied conveyor (destroy it to make space). NOT the core.
3. Place a **foundry** on the available tile
4. On the tile where the final building was (now empty), place a **splitter** with its back facing away from the foundry (so the splitter faces toward the foundry, outputting to it)
5. Build **2 bridges**, both targeting the core (or if core is out of bridge range, target the closest allied conveyor to the core):
   - **Bridge 1:** on a cardinally adjacent tile of the splitter (any side except the splitter's back). This ensures titanium continues flowing from the chain through the splitter to the core.
   - **Bridge 2:** on a tile adjacent to the foundry. This sends refined axionite from the foundry to the core.

**Resource flow after Case A:**
```
Chain → Splitter → Bridge 1 → Core (titanium continues)
              ↓
           Foundry → Bridge 2 → Core (refined axionite)
              ↑
        Axionite bridge (from our chain)
```

**Case B — Previous building is a conveyor** (the one before the final conveyor):
1. Destroy the **final conveyor** (free — allied building)
2. Place a **foundry** on the tile where the final conveyor was
3. Go back to the **previous conveyor** (the second-to-last one in the chain)
4. **Splitter logic on all adjacent conveyors targeting the foundry:**
   Scan the 4 cardinally adjacent tiles of the foundry. For EACH tile that has a conveyor targeting the foundry (its output direction points toward the foundry tile):
   a. Pathfind to that conveyor and destroy it (free)
   b. Build a **splitter** on the same tile, facing the **same direction** as the destroyed conveyor
   c. Check the splitter's output sides (all sides except its back). If there is NOT already a bridge or conveyor facing away from the splitter on at least one side:
      - Find an empty/road tile adjacent to the splitter
      - Build a **bridge** targeting the core (or closest conveyor to core if out of range)
   d. This ensures each titanium chain that was feeding into the foundry's position now splits between the foundry and the core
5. After all splitter logic is done, mark chain as complete

**Resource flow after Case B:**
```
Chain → Splitter → Foundry (titanium + axionite from our chain)
              ↓
           Bridge → Core (titanium continues to core)
```

---

**Step 7b.3 — Scenario 2: Final conveyor targets empty/road/enemy walkable**
The chain is broken — the last conveyor points at a non-functional tile. This means we haven't reached the core yet.
1. Start `chain_to_core` again from this position, treating it as an **axionite chain** (same rules — can merge, no core bridge targets)
2. If the chain merges again, re-enter `build_foundry` (Step 7b.1) and follow the new chain
3. If it reaches the core without merging, use direct foundry placement (Step 7b.6)

**Step 7b.4 — Scenario 3: Final bridge targets empty/road/enemy walkable**
Same as Scenario 2, but first pathfind to the bridge's target tile, then restart `chain_to_core` as an axionite chain from there.

**Step 7b.5 — Scenario 4: Final building targets a splitter or foundry**
The chain already terminates at a splitter (which feeds the core) or an existing foundry. No foundry placement needed.
- Mark axionite chain as complete. Set `axionite_chain_done = True`. Return to Step 2 (Exploration).

---

**Step 7b.6 — Direct foundry placement (no merge occurred):**
If the axionite chain reaches the core area without ever merging into an existing network, use this method to place a foundry directly. The chain should NOT connect to the core — it connects to a foundry instead.

**Check for existing foundry:**
Scan `tile_cache` / `building_cache` for any allied foundry adjacent to the core. If one exists:
- Chain toward the foundry (goal = within bridge range dist² ≤ 9)
- Build a bridge targeting the foundry
- Mark chain as complete → set `axionite_chain_done = True` → Step 2

**Select foundry tile (no existing foundry):**
Select a tile from the 12 cardinally adjacent core tiles meeting ALL requirements:
1. The tile must be **empty or an allied road**
2. The tile must be **cardinally adjacent to an allied conveyor that targets the core** (call this conveyor **X**)
3. Both the foundry tile AND conveyor X must be among the 12 core-adjacent tiles

**Example:** Core at (cx, cy). Conveyor X at (cx+2, cy) facing west. Valid foundry tiles: (cx+2, cy-1) or (cx+2, cy+1).

**Fallback:** If no tile meets all 3 requirements, select any empty/road core-adjacent tile.

**Chain to foundry tile:**
- Goal is within bridge range of the foundry tile
- **Never merge** during this sub-phase
- **Bridges must NOT target core tiles**
- Build bridge targeting the foundry tile when in range

**Place foundry:**
1. Pathfind adjacent to the selected tile
2. Destroy whatever is on the tile (road = free, conveyor = free). Place foundry.

**Ensure titanium supply (if tile was empty/road, not a conveyor):**
1. Find conveyor X: cardinally adjacent to foundry, targeting the core
2. Find conveyor Y: the conveyor targeting X. Get Y's direction.
3. Destroy X, place **splitter** on X's tile facing Y's direction
4. If Y doesn't exist (X fed by a bridge), place splitter facing the core
5. Check splitter sides for existing bridge/conveyor away from splitter. If missing, build a bridge targeting the core.

**Complete:** Set `axionite_chain_done = True`. Return to Step 2.

### Sentinel Logic
Sentinels are turret units — each gets its own Player instance. They are conditionally placed by economy bots during ore claiming (Step 5) and attack enemies autonomously.

**Properties:** 30 HP, 15 Ti base cost, +20% scaling, vision radius² = 32, attack radius² = 32. Cannot rotate (fixed direction set at build time). Fires every 3 turns (cooldown 3), consuming 5 resources per shot. When loaded with refined axionite, stuns the target for +2 cooldown. Ammo is fed via conveyors from any direction except the facing direction.

**Ammo:** The sentinel is placed cardinally adjacent to the harvester, so the harvester's resource output automatically feeds ammo to the sentinel. No extra conveyor infrastructure needed.

**Sentinel attack priority** (highest to lowest):
1. Enemy builder bots (mobile threats that can destroy infrastructure)
2. Enemy turrets (gunners, sentinels, breach, launchers)
3. Enemy harvesters, foundries (economic targets)
4. Enemy conveyors, armoured conveyors, bridges, splitters (logistics)
5. Enemy core
6. Enemy barriers
7. Enemy roads
8. Enemy markers

**Sentinel state machine (runs each turn):**
1. **Scan vision** for enemy buildings and units — call `get_nearby_buildings()` and `get_nearby_units()` each turn for a fresh scan
2. **Get attackable tiles** — call `get_attackable_tiles()` once to get the fixed set of tiles this sentinel can hit
3. **Select target** by priority — iterate the priority list above. For each priority level, check if any enemy of that type is on an attackable tile. Pick the first match.
4. **Fire** at the target using `ct.fire(target_pos)` if action cooldown is 0. Sentinel fires every 3 turns.
5. If target is destroyed or moves off attackable tiles, return to step 1.
6. If no enemies on attackable tiles, do nothing (idle).

### Gunner Logic
Gunners are turret units — each gets its own Player instance. They are placed by economy bots during enhanced defense ore claiming (Step 5a) when the sentinel scan doesn't pass.

**Properties:** 40 HP, 10 Ti base cost, +10% scaling, vision radius² = 13. Can target any occupied tile in its facing direction. Can rotate to any direction instantly with `ct.rotate(direction)` — costs 10 Ti from global pool + 1 turn action cooldown. Fires every turn (cooldown 1), consuming 2 resources per shot.

**Ammo:** The gunner is placed cardinally adjacent to the harvester, so the harvester's resource output automatically feeds ammo to the gunner.

**Gunner attack priority** (same as sentinel):
1. Enemy builder bots
2. Enemy turrets (gunners, sentinels, breach, launchers)
3. Enemy harvesters, foundries
4. Enemy conveyors, armoured conveyors, bridges, splitters
5. Enemy core
6. Enemy barriers
7. Enemy roads
8. Enemy markers

**Gunner state machine (runs each turn):**
1. **Scan vision** — call `get_nearby_buildings()` and `get_nearby_units()` for fresh scan
2. **Check current direction first** — call `get_attackable_tiles()` (the gunner's current facing). If a priority target is already in the current attackable tiles, skip to step 5 (no rotation needed).
3. **Virtual 7-direction scan** — for each of 7 directions (exclude direction facing harvester), call `get_attackable_tiles_from(my_pos, direction, EntityType.GUNNER)`. This is a **query only** — no Ti cost, no cooldown. Collect all enemies by priority across all directions.
4. **Select best direction** — pick the direction containing the highest-priority enemy. If the gunner is already facing that direction, no rotation needed. Otherwise, **rotate** with `ct.rotate(direction)` — goes directly to any direction (no 45° stepping). Costs 10 Ti + 1 turn action cooldown. Only rotate when the direction actually needs to change.
5. **Confirm target** — call `get_gunner_target()` to verify the gunner is targeting an enemy, NOT an allied building in the line of fire. Gunners hit the **first occupied tile** in their facing direction — an allied building between the gunner and the enemy will be hit instead. If `get_gunner_target()` returns an allied building, do NOT fire — try the next priority target or direction.
6. **Fire** at the target using `ct.fire(target_pos)` if action cooldown is 0
7. If no enemies in vision or all directions blocked by allied buildings, do nothing (idle)

**Key difference from sentinel:** Gunners can rotate to track threats, but each rotation costs 10 Ti + 1 turn cooldown. The virtual scan (`get_attackable_tiles_from`) is free — only the actual `ct.rotate()` call costs resources. Sentinels are fixed direction but cheaper to maintain.

### Barrier Busting (Economy Bots)
When an economy bot's 8-directional A* path goes through an allied barrier (cost 3 in the movement A* model), it executes a multi-step pass-through:

1. **Destroy barrier** (free — `ct.destroy(pos)`, no action cooldown)
2. **Build road** on the cleared tile (action cooldown)
3. **Move onto the road**
4. Once past the barrier tile (moved to the next tile on the path):
5. **Destroy the road** (free)
6. **Rebuild the barrier** (action cooldown)

Patrol bots treat allied barriers as **impassable** during their patrol circling (Priority 5). However, when executing higher-priority actions (repair, turret response, healing), patrol bots CAN bust through barriers using the same logic.

**Barrier handling in chain-to-core:** If a conveyor's target tile would be an allied barrier → enter `optimal_bridge` state (same as walls). Bridges must NOT target allied barriers as landing tiles. The cardinal chain A* treats barriers as impassable.

**Barrier on target ore:** If the ore tile the bot is pathfinding to has an allied barrier on it, the A* path will route through it (cost 3). When the bot arrives, destroy the barrier (free), build a road on the ore, and continue with Step 5 as normal.

### Builder Bot Self-Heal
At the end of each turn, after all step logic, builder bots check if they are idle (action cooldown still 0) and below half health (< 15 HP). If so, they heal themselves using `ct.heal(my_pos)` for 1 Ti, restoring 4 HP. This ensures damaged bots recover without delaying builds or chain construction.

## Phase 2: Patrol Bots

Patrol bots defend allied infrastructure near the core. They spawn on the **East middle tile** of the core.

### Spawning Conditions (tracked by core_logic)

The core tracks these counters and spawns patrol bots accordingly:

| Trigger | Patrol bots spawned | Notes |
|---|---|---|
| Round 100 reached | 2 | One-time trigger at round 100 |
| New enemy builder bot enters core vision | 2 per new bot | Track seen enemy bot IDs to avoid re-triggering on the same bot |
| Allied entity/building damaged in core vision | 2 per 4 damaged events | Track cumulative damage events. First 2 spawn after the first damage event, then 2 more after every 4th damage event |

**Hard cap: 8 patrol bots total.** The core tracks how many patrol bots have been spawned (not how many are alive — once spawned, they count toward the cap permanently). Use `get_nearby_units()` on the core each turn to detect new enemy builder bots and `get_hp()` on tracked buildings to detect damage.

Patrol bots are spawned one at a time on the East middle tile, waiting until the tile is unoccupied before spawning the next.

### Patrol Bot State Machine

Patrol bots use the same Player class (mixin pattern). In `run()`, when `self.role == 'patrol'`, dispatch to `_run_patrol(ct)`. Patrol bots use the same tile cache, pathfinding, and movement infrastructure as economy bots.

**Priority system:** Each turn, the patrol bot evaluates actions in this priority order. The first matching condition is executed — lower priorities are skipped.

#### Priority 0: Self-Heal (Always First)
At the start of every turn, before any other action:
- If HP is below max and action cooldown is 0: `ct.heal(my_pos)` (costs 1 Ti, heals 4 HP)
- Healing does NOT consume the move cooldown, so the bot can heal AND move in the same turn
- If the bot was damaged last turn, attempt to move **away** from the damage direction (track the direction damage came from by comparing HP between turns)

#### Priority 1: Heal Damaged Allied Buildings
Scan vision (from tile_cache + `get_nearby_buildings()`) for allied buildings with `get_hp(id) < get_max_hp(id)`:
- If multiple damaged buildings found, pick the **closest** one
- Pathfind to be adjacent (chebyshev ≤ 1) to the damaged building
- Once adjacent, `ct.heal(building_pos)` each turn until fully healed (costs 1 Ti per heal, restores 4 HP)
- If the building is destroyed while en route, return to patrol

#### Priority 2: Respond to Enemy Turrets (Supply Disruption)
If an enemy turret (gunner, sentinel, or breach — **NOT launcher**, launchers can't attack us) enters vision, the patrol bot's goal is to **cut the turret's ammo supply first**, then place a sentinel to destroy it.

**Step 2a — Identify ammo source:**
The turret needs ammo fed to it. Check what is supplying it:
- **Conveyors/harvesters:** Check all 4 cardinally adjacent tiles of the turret (from `tile_cache`). Any conveyor facing toward the turret, or any harvester adjacent to the turret, is an ammo source.
- **Bridges:** Check all allied and enemy bridges in vision (from `bridge_target_cache`). If any bridge's target position is the turret's tile, that bridge is an ammo source.

Classify the ammo source:

**If ammo source is a conveyor or bridge:**
The conveyor/bridge was being fed by a chain behind it. Destroying it and placing a sentinel on the same tile means the sentinel **inherits the ammo supply** from the existing chain. Sentinel range² = 32 (~5.6 tiles) is more than enough to reach the turret even from a bridge position (max 3 tiles away).

1. Pathfind to the conveyor/bridge tile
2. **Destroy it:**
   - If allied: `ct.destroy(pos)` — instant, free, no cooldown. The destroyed building is already in `building_cache` so it will be queued for repair later in the cleanup phase.
   - If enemy: pathfind onto the tile (it's walkable) and `ct.fire(my_pos)` until destroyed.
3. **Move off the tile** to any adjacent walkable tile (sentinels are non-walkable, so they can't be built on a tile the bot is standing on — only roads and conveyors can)
4. **Place a sentinel on the now-empty tile** facing the enemy turret. Use `get_attackable_tiles_from()` to find the direction that puts the turret in attack range. The sentinel inherits ammo from whatever was feeding the destroyed conveyor/bridge — the chain behind it is still intact.
5. The sentinel's standard attack logic handles destroying the turret from here.

**If ammo source is a harvester:**
1. Check all 4 cardinally adjacent tiles of the harvester (from `tile_cache`) for a valid sentinel position — an empty tile, or a walkable building that can be destroyed.
2. **If a valid tile exists:**
   - Pathfind to the valid tile. Destroy any walkable building there if needed.
   - **Move off the tile** to any adjacent walkable tile (sentinels are non-walkable — can't build on a tile the bot is standing on).
   - Build a sentinel on the now-empty tile facing the enemy turret (use `get_attackable_tiles_from()` for direction). The sentinel gets ammo from the adjacent harvester.
   - The sentinel's attack logic destroys the turret.
3. **If no valid tile exists:**
   - If the harvester is **allied**: destroy the harvester (`ct.destroy(pos)` — free). This cuts the turret's ammo entirely. The destroyed harvester is in `building_cache` for later repair. Then find another nearby ore source or allied harvester to place a sentinel adjacent to, facing the turret.
   - If the harvester is **enemy**: we can't destroy it (non-walkable enemy building). Find the closest allied harvester or ore within vision to place a sentinel that can reach the turret. If nothing is in range, skip this turret and return to patrol — another patrol bot may handle it from a different angle.

**Step 2b — Track active turret threats:**
Maintain `self.active_turret_threats: set[(x,y)]` — positions of enemy turrets that the patrol bot has responded to (placed a sentinel near). Each turn, check if these turrets are still alive (from `tile_cache`). Remove destroyed ones.

**Step 2c — Cleanup after turrets destroyed:**
When ALL turrets in `active_turret_threats` have been destroyed (the set is empty and we previously had entries):
1. **Destroy allied sentinels** that were placed for turret response — the patrol bot tracks which sentinels it placed (store positions in `self.placed_defense_sentinels: list[(x,y)]`). For each: pathfind adjacent, `ct.destroy(pos)`.
2. **Trigger repairs** — the destroyed conveyors/bridges/harvesters from Step 2a are already in `pending_repairs` via `building_cache`. The patrol bot's Priority 4 (repair) or economy bots in explore mode will handle rebuilding them.
3. Clear `placed_defense_sentinels` and return to normal patrol.

#### Priority 3: Destroy Enemy Infrastructure
If an enemy road, conveyor, bridge, splitter, or other building enters vision:
- Pathfind to the tile containing the enemy building
- Stand on the tile (if walkable — roads, conveyors, splitters) and attack using `ct.fire(my_pos)` until destroyed (costs 2 Ti, deals 2 damage per attack)
- If the enemy building is non-walkable (barrier, harvester, turret), pathfind adjacent and attack from there
- After destroying, build an allied road on the cleared tile (to maintain walkability for future pathing)
- Return to patrol

#### Priority 4: Repair Destroyed Buildings (Building Cache)
Use the `building_cache` and `pending_repairs` system (see "Allied Building Cache" in Architecture section). Each turn during patrol, `_scan_turn` updates both caches. If `pending_repairs` has entries:
1. Pick the **closest** repair from `pending_repairs`
2. Pathfind to the repair position
3. Execute the standard repair procedure:
   - **Empty tile:** Build the cached building
   - **Allied road:** Destroy road (free), build cached building
   - **Enemy walkable:** Pathfind onto tile, attack until destroyed, build cached building
   - **Enemy non-walkable:** Skip — remove from queue
   - **Different allied building:** Update cache, remove from queue
4. After repair, if `pending_repairs` is still non-empty AND the next repair is within `PATROL_MAX_DISTANCE` of core, continue repairing. Otherwise return to patrol.

Unlike economy bots which defer repairs during chain-building, patrol bots repair **immediately** (it's their job). Patrol bots also prefer merging existing chains when reconnecting via monotonic A* (unlike economy first-chain which avoids merging).

#### Priority 5: Default — Patrol
If no higher-priority action is needed:
- **Target:** Circle the core at radius 6 (Euclidean distance from core center)
- **Movement:** Follow a clockwise (or counterclockwise — pick one at spawn and stick with it) path around the core, staying at approximately radius 6
- **Pathfinding:** Compute waypoints at radius 6 around the core. Pick the next waypoint in the circle and pathfind to it. When reached, pick the next.
- **Map edge handling:** If the patrol path reaches a map edge or impassable wall, reverse direction (clockwise ↔ counterclockwise) and continue
- **Distance enforcement:** If the bot is ever more than 8 tiles from the core (e.g., after chasing an enemy or healing), pathfind back to radius 6 before resuming patrol
- **Do NOT leave the patrol radius** unless performing a higher-priority action (Priorities 1-4). After completing that action, always return to patrol radius.

### Patrol Bot Constants

| Constant | Value | Description |
|---|---|---|
| `PATROL_INITIAL_ROUND` | 100 | Round at which first 2 patrol bots spawn |
| `PATROL_MAX_COUNT` | 8 | Hard cap on total patrol bots spawned |
| `PATROL_RADIUS` | 6 | Euclidean distance from core center for patrol circle |
| `PATROL_MAX_DISTANCE` | 8 | Max distance from core before forced return |
| `PATROL_DAMAGE_THRESHOLD` | 4 | Damage events before spawning 2 more patrol bots |

## Phase 3: Disruptor Bots

Disruptor bots are offensive units that pathfind to the enemy core, destroy enemy infrastructure, and set up turrets to siege the core. They spawn on the **south middle tile** of the core (role detection: south middle = disruptor, south east/west = economy).

### Spawn Conditions

Disruptor spawns are only enabled when `ECONOMY_TEST_MODE = False`. When enabled, disruptors spawn based on these conditions:

| Condition | Bots | Priority |
|---|---|---|
| **Start of game:** Turn 1 and Turn 2 | 1 per turn = 2 total | Highest — spawn before economy bots |
| **First titanium ≥ `DISRUPTOR_TITANIUM = 3000`** | 4 disruptors, one per turn | High — pauses 20-turn check until batch complete |
| **Every 20 turns**, if titanium > 3000, total disruptors spawned < `MAX_DISRUPTORS = 25`, and `get_unit_count() < 50` | 1 per eligible turn | Low — only spawn if nothing else needs to spawn |

**Spawn order at game start:**
- Turn 1: Disruptor 1 spawns on south middle (starts pathfinding to enemy core immediately so it clears the tile)
- Turn 2: Disruptor 2 spawns on south middle
- Turn 3-6: 4 economy bots spawn (one per turn on their corner tiles)

**Core spawning tracker state:**
- `total_disruptors_spawned: int` — lifetime count for the `MAX_DISRUPTORS` cap
- `titanium_threshold_met: bool` — True after the 4-bot batch has been spawned
- `pending_disruptor_spawns: int` — queued disruptors still to spawn from a batch
- `last_20turn_check: int` — round of last 20-turn check

### Disruptor State Machine

Disruptor bots use the same Player class (mixin pattern). In `run()`, when `self.role == 'disruptor'`, dispatch to `_run_disruptor(ct)`.

**No resource minimum.** Unlike economy bots that wait for MIN_TITANIUM, disruptors build as soon as they have enough for the specific building they want (gunner = 10 Ti, sentinel = 15 Ti, launcher = 20 Ti at base scale). The goal is maximum disruption speed.

#### Step 1 (pre-init): Calculate enemy core position
On spawn, use symmetry detection to calculate the enemy core position. Try symmetries in order: diagonal → vertical → horizontal. Store the current guess as `self.enemy_core_guess`.

Build a **symmetry tile cache**: as the bot scans terrain, track the mirrored positions of environment tiles (walls, ore types) under the current symmetry assumption. Store as `self.symmetry_cache: dict[(x,y)] → Environment` — the expected environment at mirrored positions.

**Symmetry recalculation triggers:**
- During `_scan_turn`, if any visible tile's environment conflicts with `symmetry_cache` expectation → current symmetry is wrong. Move to the next symmetry type, rebuild the cache.
- If the bot reaches the guessed enemy core position and there's no core there → try the next symmetry type.

#### Step 2: Pathfind to Enemy Core
Use **greedy best-first search** (see "Disruptor Pathfinding" in Architecture section). The reference implementation is in `bots/path_greedy_bot/` — port it into the disruptor.

Each turn:
1. If no current path OR the path is invalidated (a tile on path is now a known wall), recompute greedy path to `enemy_core_guess`
2. Follow the path
3. If greedy returns no path → advance symmetry
4. If the bot reaches the goal area (adjacent to enemy core tile and it's actually a core) OR the bot is within `HARASS_RADIUS = 12` Euclidean tiles of the enemy core position → transition directly to **Step 5 (Harass Mode)**. There is no longer a separate core-assault phase.

**Self-destruct:** Track `self.step2_turns_count` — turns spent in Step 2. Turns in Step 3 (intercepting) do NOT count. If `step2_turns_count >= SELF_DESTRUCT_DISRUPTOR_TURNS = 300`, the opponent's side is unreachable → `ct.destroy(my_pos)`.

#### Step 3: Intercept Enemy Harvesters (Always Active)
Each turn during Step 2 pathfinding, scan `tile_cache` for enemy titanium harvesters in vision. There is **no round gate** — disruptors always check for intercepts during pathfinding.

**Detour check:** For each enemy titanium harvester:
1. Compute A* path from current position to the harvester
2. If path length ≤ `INTERCEPT_DETOUR = 6` → interrupt pathfinding, enter intercept mode

The bot can intercept multiple harvesters per trip — after placing a sentinel, re-run Step 3 scan. If no more harvesters in range, resume Step 2.

**Intercept procedure (sentinel-first, with conveyor fallback):**

**Priority A: Place sentinel on empty tile cardinally adjacent to harvester.**
Look for any empty tile cardinally adjacent to the harvester (doesn't need to also be adjacent to a conveyor). If such a tile exists:
1. Pathfind adjacent to the empty tile (or onto it)
2. Run 7-direction sentinel scan on that empty tile (exclude direction facing harvester)
3. Place sentinel facing the direction with the most enemy buildings
4. The sentinel will destroy the harvester as part of its normal targeting

**Priority B: Attack conveyor fallback.**
If no empty tile is adjacent to the harvester, find a cardinally adjacent enemy conveyor/bridge feeding the harvester (ignore armoured conveyors — builder bots can't destroy them). Then:
1. Pathfind ONTO the enemy conveyor/bridge, attack with `ct.fire(my_pos)` until destroyed
2. Move off the tile (preferably onto an existing allied road — move + action cooldowns are separate; if no road, place one first and sentinel next turn)
3. Place sentinel on the now-empty destroyed tile (7-direction scan)

If neither option is available (harvester fully enclosed by walls, no reachable conveyor), abandon this harvester and resume Step 2.

After any intercept, resume Step 2 (pathfinding to enemy core) OR Step 5 (if already in harass mode).

---

#### Step 5: Harass Mode (Permanent)
Once the disruptor reaches the enemy core area, it enters harass mode **immediately** — there is no more 3-priority core assault. The disruptor's purpose is ongoing disruption around the enemy core.

**12-tile radius constraint:** The disruptor stays within `HARASS_RADIUS = 12` Euclidean tiles of the enemy core position. Pathfinding targets and frontier exploration are bounded by this radius. The disruptor will not leave this zone.

**Harass mode has two activities:**
1. **Harvester hunting (ongoing)** — always check for enemy harvesters within 12 tiles of enemy core, intercept using Step 3 logic (no detour limit once in harass mode — intercept any reachable harvester)
2. **Periodic conveyor disruption (every 8 turns)** — see below

**Frontier exploration within harass radius:**
When idle (no harvester to hunt, not currently disrupting a conveyor), explore unexplored tiles within the 12-tile Euclidean radius of the enemy core. If all tiles within radius are explored and no harvesters exist → wait for the 8-turn counter to expire, then run conveyor disruption.

##### Step 5a: Periodic Conveyor Disruption

**The 8-turn counter (`disrupt_counter`):**
- Starts at 0 when disruptor enters harass mode
- Increments each turn the bot is **exploring / idle** (not already harassing or disrupting)
- **Pauses** while harassing a harvester (Step 3 intercept) or mid-disruption (Step 5a)
- **Resets to 0** when a disruption completes (success or abandoned due to heal)
- When counter reaches `DISRUPT_INTERVAL = 8` → run disruption procedure below

**Disruption procedure:**
1. Scan `tile_cache` (within 12-tile radius of enemy core) for enemy conveyors/bridges/splitters
2. Apply **supply flow check**: only valid if a conveyor or bridge actually targets it (prevents attacking dead infrastructure)
3. Ignore armoured conveyors (builder bots can't destroy them)
4. Pick the closest valid target by path length. If path length > 6 or no valid target exists → skip this cycle, reset counter, resume exploration
5. Pathfind onto the target, attack with `ct.fire(my_pos)` until destroyed
6. **Heal detection** (same logic as before): after `HEALED_DETECTION_TURNS = 5` turns, if HP isn't decreasing → abandon target (don't add to healed set — just move on), reset counter
7. When destroyed:
   - Move off the tile (to any adjacent walkable tile; place a road if needed)
   - Place a **barrier** on the destroyed tile (cheap 2 Ti base, just blocks rebuilding)
8. Reset `disrupt_counter = 0`, resume harass exploration

---

### Enemy Launcher Detection (Always Active for Disruptors)

Every turn, disruptor bots track their **intended position**. This is the tile the bot planned to be on after this turn's movement. If the bot chose not to move, intended = current.

At the start of each turn, compare actual position to last turn's intended position:
- **If they match:** normal turn, no launcher activity
- **If they don't match, AND the intended tile was cardinally adjacent to an enemy launcher:** the bot was launched by the enemy. Enter "launched" state.

**State variables:**
- `self.intended_pos: (x,y) | None` — position the bot intended to be on this turn
- `self.prev_intended_pos: (x,y) | None` — intended position from last turn
- `self.launched_this_turn: bool`
- `self.blocked_launcher_tiles: set[(x,y)]` — launcher + 8 adjacent tiles treated as walls

**Response to being launched:**

**Case A — Was pathfinding to target tile (no immediate attack/build goal):**
Add the enemy launcher position + its 8 adjacent tiles to `blocked_launcher_tiles`. Repath avoiding these tiles. Continue toward original goal.

**Case B — Was pathfinding to attack or place a turret:**
The target is important enough to use our own launcher. Counter with:
1. Check for allied launcher in vision range. If the target tile is within r² ≤ 26 of an allied launcher → use that launcher.
2. If no suitable allied launcher → build one. Pathfind to a tile where the target is within r² ≤ 26, build launcher there.
3. Use the builder bot launcher protocol (below) to get launched onto the target (or adjacent).
4. **Attack-on-land trick:** The launch happens during the bot's turn. Attack cooldown and move cooldown are separate — if the bot was launched AND has action cooldown 0, it can attack the same turn it lands. This lets us attack enemy infrastructure even if the enemy launcher keeps throwing us.
5. **Build-on-land trick:** Same principle. If we launch to a tile adjacent to the target, we can build the turret before the enemy launcher throws us away.

**Note on blocked_launcher_tiles:** Permanent for the bot, but if we decide to attack/build near one of those tiles later, we can still use our own launcher to reach it — those tiles are valid launch *destinations* even if they're walls for pathfinding.

---

### Builder-Side Launcher Protocol

Used by disruptors (Case B above) to get launched by an allied launcher. Also usable by any builder bot that wants to be launched.

**Marker protocol format:**
```
value = LAUNCHER_PROTOCOL_PREFIX + (bot_id % 10000) * 10000 + target_x * 100 + target_y
LAUNCHER_PROTOCOL_PREFIX = 100_000_000
```

**Protocol sequence:**
1. Bot identifies target tile T (must be bot-passable — if T is empty, change target to a road/conveyor tile)
2. Bot pathfinds close to launcher L — needs to place a marker in BOTH L's vision range (r² ≤ 26) AND within the bot's own action radius (r² ≤ 2, i.e. the bot's 8 surrounding tiles + own tile)
3. Bot places marker on an empty or allied road tile (destroy the road first if needed) with the encoded value
4. **Turn N+1:** Launcher reads marker, stores request as pending
5. **Turn N+1 or N+2:** Bot destroys the marker — this activates the request on the launcher's next turn
6. Bot pathfinds to a tile cardinally adjacent to L
7. **Next turn:** Launcher detects the bot is adjacent with an active request and calls `ct.launch(bot_pos, target_pos)`. Bot lands on T.
8. Bot continues whatever it was doing.

**Launch target adjustments:** If the target is an enemy launcher's adjacent tile, we expect to be immediately launched back. That's fine — we'll still get our attack/build in that turn (attack-on-land / build-on-land trick).

---

### Launcher Turret Logic

Launchers are turret units — each gets its own Player instance. They have no facing direction and no ammo requirement.

**Properties:** 30 HP, 20 Ti base cost, +10% scaling, vision/throw range² = 26. Throws adjacent builder bots to any bot-passable tile within range.

**State:**
- `self.pending_requests: dict[bot_id_mod → (target_x, target_y, marker_xy)]`
- `self.active_requests: dict[bot_id_mod → (target_x, target_y)]`
- `self.read_marker_positions: set[(x,y)]` — markers already processed

**Launcher state machine (runs each turn):**

1. **Read request markers:** For each allied marker in vision:
   - Skip if position is in `read_marker_positions`
   - Read value with `get_marker_value(marker_id)`
   - If value ≥ `LAUNCHER_PROTOCOL_PREFIX`: decode (bot_id_mod, target_x, target_y). Store in `pending_requests`. Add position to `read_marker_positions`.

2. **Check for activation:** For each entry in `pending_requests`, check if the marker at its stored position is gone (destroyed). If so, move the request from `pending_requests` to `active_requests`.

3. **Launch allied bots:** For each adjacent (chebyshev ≤ 1) allied builder bot, check if its `id % 10000` is in `active_requests`. If yes:
   - Attempt `ct.launch(bot_pos, target_pos)`
   - **If launch fails** (target has a bot on it, or target not passable): recompute target — find a bot-passable tile close to the original target that does NOT have a bot on it. Try again.
   - Remove from `active_requests` after successful launch

4. **Launch enemy bots away:**
   - For each adjacent enemy builder bot:
     - **If the enemy core is in vision:** pick the enemy core tile **farthest from the launcher's current position** (max distance among the 9 core tiles, that is bot-passable and has no bot on it). Launch there.
     - **If enemy core is NOT in vision, OR all 9 core tiles are full/invalid:** fall back to the farthest bot-passable tile from our core within throw range (no bot on it)
   - `ct.launch(enemy_bot_pos, target)`
   - If launch fails (bot on target): try once more with a recomputed target avoiding occupied tiles

---

### Disruptor Constants

| Constant | Value | Description |
|---|---|---|
| `START_DISRUPTOR_BOTS` | 2 | Disruptors spawned at game start (turns 1-2) |
| `DISRUPTOR_TITANIUM` | 3000 | Titanium threshold for spawning |
| `MAX_DISRUPTORS` | 25 | Hard cap on total disruptors spawned |
| `SELF_DESTRUCT_DISRUPTOR_TURNS` | 300 | Turns in Step 2 without progress before self-destruct |
| `INTERCEPT_DETOUR` | 6 | Max path length (tiles) from current position to a harvester to intercept |
| `IGNORE_ENEMY_CONVEYOR_THRESHOLD` | 6 | Max path length to enemy conveyor before skipping (Step 5a disruption) |
| `HEALED_DETECTION_TURNS` | 5 | Turns of attacking before abandoning a healed target |
| `HARASS_RADIUS` | 12 | Euclidean radius (in tiles) from enemy core within which disruptor operates in harass mode |
| `DISRUPT_INTERVAL` | 8 | Turns between conveyor disruption attacks in harass mode |
| `LAUNCHER_PROTOCOL_PREFIX` | 100_000_000 | Marker value prefix for launcher requests |
| `ECONOMY_TEST_MODE` | False | If True, disables all disruptor spawning |


## TUNABLE CONSTANTS

Note: Maps range from 20×20 to 50×50. Constants should be tuned for the smallest maps — especially `INITIAL_EXPLORE_RADIUS` and `FRONTIER_SCAN_RADIUS` which should not exceed the map dimensions.

### Bot-specific constants (in constants.py)

| Constant | Value | Description |
|---|---|---|
| `ECONOMY_BOTS_COUNT` | 4 | Number of economy bots spawned |
| `INITIAL_EXPLORE_RADIUS` | 15 | Starting exploration radius from core |
| `EXPLORE_RADIUS_EXPAND_INTERVAL` | 50 | Turns between each 1-tile radius expansion |
| `MAX_EXPLORE_RADIUS` | 30 | Maximum exploration radius |
| `ORE_PATHFIND_GIVE_UP_DIST` | 12 | Abandon ore if pathfinding leads this far away |
| `MAX_HARVESTERS_PER_BOT` | 6 | Harvesters before role transition |
| `BRIDGE_DETOUR_THRESHOLD` | 10 | Extra conveyor tiles before resorting to a bridge |
| `ASTAR_MAX_NODES` | 2000 | Safety cap — generous for any map up to 50×50 |
| `FRONTIER_SCAN_RADIUS` | 12 | Local scan window for frontier detection |
| `POSITION_HISTORY_LEN` | 10 | Ring buffer size for oscillation detection |
| `OSCILLATION_THRESHOLD` | 3 | Revisits in history before breaking out |
| `OBSERVE_TURNS` | 4 | Turns to observe a conveyor for capacity check |
| `SENTINEL_ATTACK_THRESHOLD` | 8 | Min enemy buildings in one direction to justify sentinel |
| `DEFEND_HARVESTER_RANGE` | 8 | Max Euclidean distance from core for enhanced defense (barriers + gunner) |
| `TLE_RECOVERY_BASE` | 3 | Base recovery turns after a TLE (+ consecutive count) |

### Game constants (use GameConstants — do NOT hardcode)

All base costs, scaling values, radii, and HP values should be read from `GameConstants` at import time. Never hardcode these — they may change between patches.

```python
from cambc import GameConstants

# Base costs are (titanium, axionite) tuples
GC = GameConstants
HARVESTER_COST = GC.HARVESTER_BASE_COST        # (20, 0)
CONVEYOR_COST = GC.CONVEYOR_BASE_COST          # (3, 0)
BRIDGE_COST = GC.BRIDGE_BASE_COST              # (20, 0)
SPLITTER_COST = GC.SPLITTER_BASE_COST          # (6, 0)
BARRIER_COST = GC.BARRIER_BASE_COST            # (3, 0)
ROAD_COST = GC.ROAD_BASE_COST                  # (1, 0)

# Radii
BUILDER_VISION_SQ = GC.BUILDER_BOT_VISION_RADIUS_SQ  # 20
BRIDGE_RANGE_SQ = GC.BRIDGE_TARGET_RADIUS_SQ          # 9
CORE_VISION_SQ = GC.CORE_VISION_RADIUS_SQ             # 36

# Other
MAX_TURNS = GC.MAX_TURNS                        # 2000
MAX_UNITS = GC.MAX_TEAM_UNITS                   # 50
STACK_SIZE = GC.STACK_SIZE                       # 10
```