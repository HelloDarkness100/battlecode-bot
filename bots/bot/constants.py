from cambc import GameConstants as GC

# --- Debug mode: set False before uploading to server ---
DEBUG = False
ECONOMY_TEST_MODE = False  # Disables sentinel placement + firing for clean economy testing
REPAIR_TEST = False      # Spawn 3 destructor bots at round 300 to destroy allied infrastructure

# Repair test
REPAIR_TEST_ROUND = 300
REPAIR_TEST_COUNT = 3

# --- Bot-specific tunable constants ---

# Economy
# Tiered economy bot count + starting-disruptor cap based on the
# number of titanium ores visible to the core on turn 1. Lots of ore =
# economy ramps fast, so we can afford to skip an economy slot and put
# more pressure on the enemy via disruptors. No ore = grind, prioritise
# economy.
TITANIUM_ORE_THRESHOLD = 5            # ore_count > threshold ⇒ HIGH tier; also the saturation point for the disruptor ore_factor
ECONOMY_BOTS_HIGH_ORE = 2             # ore_count > TITANIUM_ORE_THRESHOLD
ECONOMY_BOTS_MED_ORE = 3              # 1 ≤ ore_count ≤ TITANIUM_ORE_THRESHOLD
ECONOMY_BOTS_LOW_ORE = 4              # ore_count == 0
# Hard ceiling on economy bots. Late-game (Ti ≥ DISRUPTOR_TITANIUM) the
# core tops up from the early-tier target to this number before
# spawning more disruptors.
ECONOMY_BOTS_MAX = 4
# Fraction of the estimated chain-back conveyor cost we require up front
# before starting a new harvester claim. The harvester cost itself is
# always required in full; this only scales the chain estimate. Lower =
# start sooner and rely on passive Ti income to cover the rest while
# the chain is being built; higher = wait for more of the chain budget
# before committing.
HARVEST_AFFORD_MULT = 0.3

# Exploration
INITIAL_EXPLORE_RADIUS = 15
EXPLORE_RADIUS_EXPAND_INTERVAL = 50
MAX_EXPLORE_RADIUS = 30
FRONTIER_SCAN_RADIUS = 12

# Ore / harvesting
ORE_PATHFIND_GIVE_UP_DIST = 12
MAX_HARVESTERS_PER_BOT = 6

# Pathfinding
ASTAR_MAX_NODES = 800
BRIDGE_DETOUR_THRESHOLD = 5
STRICT_MONOTONIC = False  # True = strict decrease, False = allow same distance
# Cost of routing through an allied barrier (destroy → build/walk).
# Reference points: empty tile = 2, existing transport = 1, unknown = 3.
# At BARRIER=4 the chain prefers any cardinal detour of ≤1 extra tile and
# only busts a barrier when the detour would cost more. Raise to favour
# detours over barrier destruction more aggressively.
BARRIER_PATH_COST = 6
# Hard cap on turns spent walking to a single bridge target. Beyond this
# we conclude the bridge target is genuinely unreachable and abandon
# the entire chain rather than thrashing.
MAX_BRIDGE_RETRY = 100

# Stuck / oscillation detection
POSITION_HISTORY_LEN = 10
OSCILLATION_THRESHOLD = 3

# Conveyor observation
OBSERVE_TURNS = 4

# Sentinel placement
SENTINEL_ATTACK_THRESHOLD = 6  # Min enemy buildings in one direction to justify sentinel

# Defense-kit gunner toggle: when True, the full ore-defense kit places a
# gunner on the conveyor-adjacent perpendicular tile (default behaviour).
# When False, that tile becomes a barrier instead — still blocks bots, far
# cheaper, but no active fire. Sentinel placements are unaffected: the
# sentinel-scan branch in _claim_ore_defend still runs when its direction
# scan clears SENTINEL_ATTACK_THRESHOLD.
PLACE_GUNNER = False

# Idle turns before a chain-supplied sentinel self-destructs. Sentinels
# cardinally adjacent to an allied harvester NEVER self-destruct — they
# are part of the harvester defense kit.
SENTINEL_IDLE_CLEANUP_TURNS = 400

# Enhanced harvester defense: ores within this Euclidean distance of the
# core get the full defense kit (conveyor + 2 barriers + gunner/sentinel).
# Beyond this, fall back to the cheap sentinel-only check.
DEFEND_HARVESTER_RANGE = 20

# Within this Chebyshev radius of the core, the chain builder pads
# every conveyor/bridge it places by paving the 8-neighbour ring with
# cheap roads (skipping the chain's own in/out tiles). Roads occupy
# the tile so an enemy disruptor can't drop a sentinel adjacent to a
# core-feeding conveyor without first spending an action to clear
# them. 7 covers everything inside the typical defence kit range.
ROAD_PROTECT_RANGE = 7

# Titanium reserve floor — base value at scale=100%. The actual reserve
# enforced by _can_spend is `int(MIN_TITANIUM * current_scale)`, so the
# buffer keeps the same purchasing power as the game scales costs up.
# Example: MIN_TITANIUM=40 ≈ 2 builder bots at any scale. Every titanium
# spend is gated by this floor; the only exemption is panic-mode patrol
# spawning in core_logic.
MIN_TITANIUM = 20

# Patrol bots upgrade any allied conveyor they pass to ARMOURED_CONVEYOR
# if team axionite is above this floor. Core-adjacent ring upgrades are
# unconditional (armoured ring = always defended). Lower-priority ring
# than circling self-destruct logic, so it never starves higher-priority
# actions.
MIN_AXIONITE_FOR_ARMOURED = 400

# TLE recovery
TLE_RECOVERY_BASE = 3  # Base recovery turns after TLE (+ consecutive count)

# Axionite detection
AXIONITE_FOUND_LIMIT = 1

# Foundry placeholder marker value. When a bot wants to place a foundry
# but can't afford it yet, it drops a marker with this value on the
# chosen foundry tile. The marker is non-walkable (like a foundry) and
# other bots treat any allied core-adjacent marker as a pending foundry
# for axionite chain completion.
FOUNDRY_MARKER_VALUE = 0xF00D0001

# Axionite conversion: core converts axionite → titanium when Ti drops
# below the scaled MIN_TITANIUM floor (1 Ax → 4 Ti).
CONVERT_AXIONITE = True

# Frontier bias
FRONTIER_BIAS_STRENGTH = 20

# Early patrol: damage-triggered patrols that bypass MIN_TITANIUM and
# PATROL_ENEMY_BOT_TRIGGER_ROUND. Hard-capped at EARLY_PATROL total.
EARLY_PATROL = 4

# Temporary patrol: bots spawned after PATROL_TEMP_ROUND self-destruct
# after TEMP_SELF_DESTRUCT_TURNS of only circling (no higher-priority action).
PATROL_TEMP_ROUND = 350
TEMP_SELF_DESTRUCT_TURNS = 40

# Patrol bots
PATROL_INITIAL_ROUND = 300
PATROL_ENEMY_BOT_TRIGGER_ROUND = 40  # Enemy-bot trigger inactive before this round
PATROL_PER_ENEMY_BOT = 1            # Patrol bots spawned per new enemy builder bot seen
PATROL_MAX_COUNT = 25
PATROL_RADIUS = 3
PATROL_MAX_DISTANCE = 20
PATROL_DAMAGE_THRESHOLD = 4
PATROL_GAP_WAIT_ROUNDS = 10  # Rounds to wait after spotting a broken conveyor
                             # target before "fixing" it — gives economy bots
                             # time to extend their own chain.

# Core panic mode: when core HP drops below PANIC_HP_THRESHOLD (fraction of
# max), spawn PANIC_INITIAL_SPAWNS extra patrol bots. For each additional
# PANIC_STEP_PCT drop below that, spawn PANIC_STEP_SPAWNS more. Panic
# spawns bypass PATROL_MAX_COUNT.
PANIC_HP_THRESHOLD = 0.5
PANIC_STEP_PCT = 0.1
PANIC_INITIAL_SPAWNS = 5
PANIC_STEP_SPAWNS = 2

# --- Disruptor bots ---
# Starting disruptors are scaled dynamically from START_DISRUPTOR_BOTS
# (floor) up to START_DISRUPTORS_MAX based on three map features:
#   - area_factor    : smaller maps → more starting disruptors (short trip)
#   - center_factor  : our core closer to map centre → enemy core is also
#                      closer (map symmetry), so more starting disruptors
#   - ore_factor     : titanium ores visible to the core. SLIGHT influence
#                      only — we don't want to let an ore-poor opening hide
#                      ores just outside vision and starve us of disruptors
#                      on small maps. ore_count == 0 → 0, ore_count >=
#                      TITANIUM_ORE_THRESHOLD → 1, linear in between.
# The formula is a weighted AVERAGE of the factors, then scaled into
# the [BOTS, MAX] range:
#   blend = (AREA_W*area + CENTER_W*center + ORE_W*ore) / (sum of weights)
#   count = round(BOTS + blend * (MAX - BOTS))
# `blend` is naturally in [0, 1] so `count` never overshoots MAX —
# weights only control the RELATIVE influence of each factor, not the
# absolute magnitude of the score. Setting all weights to 0 forces
# count = BOTS (floor only).
START_DISRUPTOR_BOTS = 0                   # Floor (minimum count)
START_DISRUPTORS_MAX = 2                   # Ceiling on dynamic count
START_DISRUPTORS_AREA_WEIGHT = 2.0         # Weight on small-map bonus
START_DISRUPTORS_CENTER_WEIGHT = 2.0       # Weight on center-close bonus
START_DISRUPTORS_ORE_WEIGHT = 1.0          # Weight on ore-rich bonus (kept low — slight nudge only)
START_DISRUPTORS_REF_AREA = 2500           # Normalisation baseline (50x50)
DISRUPTOR_TITANIUM = 670                  # Ti threshold for 4-disruptor batch + 20-turn spawns
MAX_DISRUPTORS = 25                        # Hard cap on total disruptors spawned
DISRUPTOR_SPAWN_INTERVAL = 20              # Turns between late-game disruptor spawns
SELF_DESTRUCT_DISRUPTOR_TURNS = 500        # Turns in Step 2 before self-destruct
# Sentinel-zone safety: enemy SENTINELs cardinally adjacent to a
# HARVESTER (any team) have ammo and are dangerous. Tiles inside
# their attackable cone are tracked in self.dangerous_sentinel_tiles.
# - Below DISRUPTOR_DANGER_HP_FRAC of max HP, the disruptor treats
#   those tiles as walls in greedy pathfinding AND interrupts normal
#   step dispatch to flee toward the nearest safe tile while the
#   end-of-turn self-heal kicks in.
# - At or above the threshold, the tiles are still passable but get
#   a heuristic penalty so the greedy steers around them when the
#   detour is cheap.
DISRUPTOR_DANGER_HP_FRAC = 0.6
DISRUPTOR_DANGER_HEURISTIC_PENALTY = 25    # ~5-tile-equivalent push at full HP
INTERCEPT_DETOUR = 6                       # Max A* path length from bot to harvester to intercept
IGNORE_ENEMY_CONVEYOR_THRESHOLD = 6        # Max A* path length to a disruption target (Step 5a)
HEALED_DETECTION_TURNS = 5                 # Turns of attacking before abandoning a healed target
HARASS_RADIUS = 12                         # Euclidean radius (tiles) around enemy core for harass mode
DISRUPTOR_CIRCLE_RADIUS = 4                # Patrol radius around enemy core for the harass-mode default circle
DISRUPT_INTERVAL = 8                       # Idle turns between conveyor disruption attacks in Step 5a
LAUNCHER_PROTOCOL_PREFIX = 100_000_000     # Marker-value prefix for launch requests
