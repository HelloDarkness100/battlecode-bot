import sys
from collections import deque
from cambc import EntityType, Environment, GameConstants as GC
from scanning import ScanningMixin
from exploration import ExplorationMixin
from harvesting import HarvestingMixin
from chaining import ChainingMixin
from movement import MovementMixin
from core_logic import CoreMixin
from sentinel import SentinelMixin
from gunner import GunnerMixin
from patrol import PatrolMixin
from destructor import DestructorMixin
from disruptor import DisruptorMixin
from launcher import LauncherMixin
from constants import POSITION_HISTORY_LEN, EXPLORE_RADIUS_EXPAND_INTERVAL, TLE_RECOVERY_BASE, DEBUG
from utils import detect_symmetry, xy_to_pos


class Player(ScanningMixin, ExplorationMixin, HarvestingMixin,
             ChainingMixin, MovementMixin, CoreMixin, SentinelMixin,
             GunnerMixin, PatrolMixin, DestructorMixin,
             DisruptorMixin, LauncherMixin):

    def __init__(self):
        # --- Identity ---
        self.unit_type = None       # EntityType (CORE or BUILDER_BOT)
        self.corner = None          # str: "NW", "NE", "SW", "SE" (economy bots)
        self.role = None            # str: "economy", "patrol", "disruptor", "axionite"
        self.my_team_cache = None   # Team enum, cached once per turn

        # --- Core state ---
        self.core_pos = None        # (x,y) of our core center
        self.enemy_core_pos = None  # (x,y) of enemy core center (set by _scan_turn when seen)
        self.map_w = None
        self.map_h = None
        self._symmetry = None           # 'diag', 'vert', or 'horiz'
        self.economy_spawned = 0
        self._economy_used_corners = set()   # corner indices already spawned
        self.patrol_spawned = 0
        self.disruptor_spawned = 0
        self.axionite_spawned = 0

        # --- Tile cache (populated by _scan_turn) ---
        self.tile_cache = {}            # (x,y) → (env, bid, etype, team)
        self.frontier_cache = set()     # (x,y) known tiles with ≥1 in-bounds unknown neighbor
        self.known_walls = set()        # permanent set of wall positions
        self.bot_pos_cache = {}         # (x,y) → (unit_id, team)
        self._bot_stationary = {}       # (x,y) → turns a bot has stayed there
        self.bridge_target_cache = {}   # (x,y) → (target_x, target_y)

        # --- Vision tracking (for incremental env queries) ---
        self._last_vision_set = set()   # set of (x,y) visible last turn
        self._last_pos = None           # (x,y) last turn position
        self._my_xy = None              # (x,y) this turn — set by _scan_turn
        self._last_tiles = None         # cached get_nearby_tiles() result

        # --- State machine ---
        self.step = "leave_core"
        self._prev_step = None          # for logging transitions

        # --- Pathfinding ---
        self.path = None                # list[(x,y)] current path
        self.path_index = 0

        # --- Exploration ---
        self.frontier_target = None     # (x,y) current frontier goal
        self._first_chain_done = False  # True after first full chain to core
        self.explore_radius = None      # current explore radius (set on first run)
        self._next_radius_expand = EXPLORE_RADIUS_EXPAND_INTERVAL
        self._frontier_blacklist = set()

        # --- Ore / harvesting ---
        self.target_ore = None          # (x,y) ore being pathfound to
        self._from_stored_ore = False   # True when target_ore came from stored_ores
        self.chain_start = None         # (x,y) first conveyor tile of current chain
        self._chain_dir = None          # Direction the chain flows (toward core)
        self._chain_conv_dir = None     # Actual direction first conveyor was built facing
        self.harvesters_built = 0
        self._harvester_moved = False   # True once on conveyor tile for build_harvester
        self._sentinel_placed = False   # True once sentinel placed at ore (or scan decided not to)
        self._skip_chain_conveyor = False  # True when chain_start conveyor target is invalid —
                                           # skip conveyor build, still run defense, let
                                           # chain_to_core bridge through this tile instead

        # --- Step 5a (close-to-core enhanced defense) state machine ---
        self._defend_substep = 0
        self._defend_barrier_xys = []
        self._defend_turret_xy = None
        self._defend_turret_kind = None  # EntityType.GUNNER or SENTINEL
        self._defend_turret_dir = None

        # --- Barrier busting (economy bots only) ---
        # List of (x,y) tiles that were allied barriers before we walked
        # through them. _barrier_bust_restore rebuilds each one as soon
        # as the bot is within action radius and NOT standing on the
        # tile. Using a list lets the bot safely bust several consecutive
        # barriers without forgetting earlier ones while it waits to
        # step off.
        self.pending_barrier_restores = []
        self.stored_ores = []           # list of (x,y) ores seen during harvester build
        self.axionite_positions = []    # list of (x,y) axionite ores found
        self.axionite_found_count = 0
        self._ore_blacklist = set()     # ores we tried and couldn't reach

        # --- Chaining ---
        self.chain_path = None          # list[(x,y)] conveyor chain route
        self.chain_index = 0
        self._chain_built = None        # set of (x,y) tiles we built chain conveyors on
        self._chain_start_fixed = False # True once chain_start conveyor direction verified
        # After the final core-targeting conveyor is placed, the bot
        # paves any empty 8-neighbours of that tile with cheap roads
        # before _chain_complete fires. Roads occupy the tile so an
        # enemy disruptor can't drop a building there without spending
        # an action to clear it first.
        self._chain_seal_queue = []     # list[(x,y)] of empty tiles to pave
        self._chain_seal_anchor = None  # (x,y) of the final conveyor
        # Per-piece chain padding: when the chain builder is within
        # ROAD_PROTECT_RANGE of the core, each new conveyor/bridge
        # gets its 8-neighbour ring paved (excluding the chain's own
        # in/out tiles) before the bot advances to the next chain
        # piece. _chain_protect_target latches the target_xy so we
        # don't re-populate the queue on every turn.
        self._chain_protect_queue = []
        self._chain_protect_target = None
        self._chain_merge_tile = None   # (x,y) tile where merge conveyor would go
        self._observe_skip = None       # set of (x,y) conveyors found saturated
        self._bridge_walk_target = None # (x,y) bridge destination (where we end up)
        self._bridge_walk_turns = 0
        self._bridge_walk_prev_xy = None
        self._bridge_walk_stuck = 0
        self._bridge_astar_fails = 0
        self._bridge_target_blacklist = set()
        self._chain_dbg_last = None
        self._walk_stuck_turns = 0
        self._pending_harvester_xy = None   # Opportunistic harvester spot after chain conveyor
        self._pending_return_xy = None      # Tile to walk back to after clearing enemy from pending ore

        self.observe_xy = None          # (x,y) of conveyor being observed
        self.observe_target = None      # building id being observed
        self.observe_turns_left = 0
        self.observe_ever_empty = False
        self.observe_saw_axionite = False  # True if raw axionite was seen on
                                            # the observed conveyor during the
                                            # 4-turn window (blocks titanium
                                            # merges to avoid contamination)

        # --- Axionite chain state ---
        self.stored_axionite_pos = None    # (x,y) or None — one axionite ore
                                            # remembered during Step 3 explore
        self.axionite_chain_done = False    # bot-lifetime: axionite chain built
        self.titanium_chains_completed = 0  # counter for eligibility gating
        self.current_chain_type = None      # 'titanium' | 'axionite' | None
        # Optional monotonic-A* goal override for the chain builder. When
        # None, _step_chain_to_core targets self.core_pos. When set, the
        # chain targets this tile instead (used by the axionite chain to
        # aim at a foundry tile rather than the core).
        self._chain_goal = None

        # --- Axionite chain-to-core substeps (Step 7b) ---
        self.axionite_chain_substep = None  # 'check_foundry'|'select_foundry_tile'|
                                             # 'chain_to_foundry'|'place_foundry'|
                                             # 'ensure_titanium'|'complete'
        self.foundry_target = None          # (x,y) existing foundry found
        self.foundry_tile = None            # (x,y) tile we intend to build on
        self._foundry_destroyed_etype = None  # what was at foundry_tile

        # --- Step 7b.1 follow-merged-chain state ---
        self.follow_chain_pos = None         # (x,y) current building inspected
        self.follow_chain_prev_type = None   # 'conveyor' | 'bridge' — the one
                                              # BEFORE the current position (used
                                              # by Scenario 1 A/B split)
        self.follow_chain_final_pos = None   # (x,y) final building (Scenario 1)
        self._scenario_step = 0              # sub-step within a scenario handler
        self.splitter_queue = []             # [(x,y)] for Scenario 1B retrofits
        # Dead-end target for Scenario 2/3: the empty/road tile past the last
        # working conveyor/bridge. The bot walks to this tile, then restarts
        # the axionite chain from there so the restart extends (not re-merges)
        # the old network.
        self._scenario_23_target = None
        # Tiles we followed during the current follow-chain step. Used by the
        # Scenario 2/3 restart to inject these as `_observe_skip` entries so
        # we don't re-merge into the same broken network.
        self._follow_chain_seen = set()
        # _chain_reached_end replan loop guard: if we've already replanned
        # from the same `last` tile once, fall back to direct placement.
        self._last_replan_from = None
        # The allied conveyor X cardinally adjacent to foundry_tile that
        # targets a core tile. Retrofit into a splitter after foundry
        # placement. Populated by _pick_foundry_tile.
        self.foundry_splitter_tile = None
        # X's direction captured before the retrofit destroy, so the
        # splitter rebuild survives a turn boundary.
        self._retrofit_x_dir = None
        # When True, _step_observe_conveyor bypasses merging and goes
        # straight to bridge-over. Used by the axionite reroute
        # sub-chain (Bug 1 fix) so it can't merge back into the network
        # it's about to replace.
        self._suppress_chain_merge = False
        # After placing a splitter, conveyors on non-back sides that
        # pointed at the splitter are blocked. Queue them here to be
        # replaced with bridges targeting the splitter.
        self._splitter_bridge_fixes = []
        self._splitter_output_check = None
        self._foundry_redirect_queue = []
        self._foundry_redirect_splitter = None
        self._foundry_redirect_foundry = None
        self._foundry_redirect_pending_splitter = None
        self._1b_first_splitter = None

        # --- Steal enemy harvester state ---
        self.steal_target = None        # (x,y) of enemy harvester on ore
        self.sentinel_tile = None       # (x,y) where sentinel will be placed
        self.sentinel_dir = None        # Direction for sentinel
        self.steal_chain_tile = None    # (x,y) chain_start candidate after sentinel

        # --- Stuck detection ---
        self.pos_history = deque(maxlen=POSITION_HISTORY_LEN)  # auto-evicts oldest
        self.economy_stuck_turns = 0
        self.economy_stuck_step = None
        self.economy_stuck_pos = None
        self.avoid_bots = False
        self.oscillation_walls = set()
        self._stuck_count = 0           # turns stuck on same path step

        # --- Build action tracking (for stdout status) ---
        self._last_build_action = None  # str describing what we tried to build

        # --- Building cache & repair ---
        # building_cache tracks allied *economy* buildings only
        # (conveyor/splitter/bridge/harvester/foundry/barrier). When a
        # patrol sentinel is placed on top of a cached tile we leave
        # the old entry intact so that after the sentinel self-destructs
        # the repair pipeline automatically rebuilds the original
        # economy building. Non-economy allied buildings (gunners, etc.)
        # are never written here.
        self.building_cache = {}           # (x,y) -> (entity_type, direction_or_target)
        # Subset view of building_cache: tiles holding an allied BARRIER.
        # Maintained alongside building_cache so patrol pathfinding can
        # inject barriers as walls in O(|barriers|) instead of iterating
        # the whole tile_cache (~2500 entries on mature 50x50 maps — was
        # the dominant patrol-side TLE source when damage triggered
        # frequent repaths).
        self.allied_barriers = set()       # {(x,y)} allied barriers
        # Incrementally-maintained set of allied non-walkable defensive
        # buildings (gunner / sentinel / breach / launcher / harvester
        # / foundry). Replaces the per-call full tile_cache scan in
        # _allied_barrier_walls — chain planning was paying ~625μs per
        # call on mature 50x50 maps to scan 2500 entries.
        self.allied_blockers = set()
        self.pending_repairs = []          # [(x,y)] positions needing repair
        self.repair_target = None          # (x,y) currently being repaired
        self._pre_repair_step = None       # step to resume after repair
        self._pre_repair_target_ore = None
        self._pre_repair_frontier = None

        # --- TLE recovery ---
        self.turn_completed = True
        self.tle_recovery_turns = 0
        self.consecutive_tles = 0

        # --- Patrol state ---
        self.patrol_action = None        # 'heal_bldg', 'turret_response', 'destroy_infra', 'repair_chain', 'circle', 'return_to_core', None
        self._patrol_spawn_round = None  # round this patrol bot was assigned
        self._patrol_idle_turns = 0      # consecutive turns spent only circling
        self.patrol_target_xy = None     # (x,y) current action target
        self.patrol_target_bid = None    # building id being targeted/healed
        self.patrol_prev_hp = None       # HP last turn
        self._patrol_waypoints = None    # list[(x,y)] precomputed ring waypoints
        self.patrol_waypoint_idx = 0     # current waypoint index (advances each turn)
        self._patrol_last_clamped = None # last clamped target (x,y) for stuck detection
        self._patrol_clamp_count = 0     # turns clamped target unchanged
        self._max_hp_by_type = {}        # EntityType → max HP (cached FFI)
        # Per-bot lazy caches for enemy building geometry. Populated by
        # the disruptor's chain-extension picker — without these the
        # picker did one FFI per enemy conveyor / bridge in vision
        # *every turn*, which dominated Step-2 disruptor turn cost in
        # late-game dense enemy territory. Conveyor direction and
        # bridge target are fixed for the lifetime of a bid (a
        # destroyed + rebuilt building gets a new bid), so caching by
        # bid is permanent and doesn't go stale.
        self._enemy_dir_cache = {}        # bid → Direction
        self._enemy_bridge_target_cache = {}  # bid → (x, y)
        self._gunner_original_dir = None # Direction this gunner was built facing;
                                         # rotate back to it when no threat warrants
                                         # a different facing (gunner.py only).
        self._patrol_gap_watch = {}      # (x,y) → first_round the gap was seen
        self._patrol_orphan_watch = {}   # (x,y) → first_round the orphaned harvester was seen
                                         # Patrol waits 5 rounds before "fixing"
                                         # a broken conveyor target, in case an
                                         # economy bot is actively chaining there.

        # --- Turret-response state machine (patrol Priority 2) ---
        # Multi-turn process: identify ammo source → cut it → place sentinel
        # that inherits the ammo → sentinel destroys turret. Sentinels are
        # left standing after the turret dies so they can defend against
        # future waves; economy bots rebuild infrastructure around them.
        self.active_turret_threats = set()   # set[(x,y)] — turrets being handled
        self.turret_response_substep = 0     # 0=idle 1=goto 2=destroy 3=vacate 4=build
        self.turret_target = None            # (x,y) — current enemy turret
        self.turret_ammo_source = None       # (x,y) — identified feeder tile
        self.turret_ammo_source_type = None  # 'conveyor' | 'bridge' | 'harvester'
        self.turret_sentinel_tile = None     # (x,y) — where the sentinel will go
        self.turret_sentinel_dir = None      # Direction — sentinel facing
        self._turret_response_bc_stash = None  # (sxy, old_cache_entry) stash
                                                # so the repair cache survives
                                                # the destroy→sentinel dance

        # --- Destructor state (test only) ---
        self._destructor_spawn_round = None  # round when destructor first ran
        self._destructor_destroyed = set()   # tiles this destructor has already hit

        # --- Disruptor state (builder-bot only) ---
        self.disruptor_step = 1                  # 1..10 state machine position
        self.disruptor_step2_turns = 0           # turns spent in Step 2 (self-destruct gate)
        self.enemy_core_guess = None             # (x,y) current pathfinding target
        self.disruptor_sym_when_guessed = None   # the symmetry the current guess was built from
        self._disruptor_last_scanning_sym = None  # last observed value of scanning.py's self._symmetry;
                                                  # we only auto-sync when scanning.py itself cycles,
                                                  # so a local force_next_symmetry override sticks.
        self.disruptor_forced_sym_failures = 0   # bounded local-fallback cycle count
        self._disruptor_step_stub_logged = False  # one-shot log for Part 1 stubs
        self._disruptor_sym_order = None          # distance-sorted symmetry priority
                                                  # (populated on step1_init)
        self._sym_tried_disruptor = set()         # symmetries this bot has already
                                                  # ruled out via arrival-check or
                                                  # scanning-detected mismatch

        # True once this disruptor has placed ANY intercept sentinel during
        # Step 2's pathfind-to-core leg. Suppresses further Step-2 intercept
        # picks so the bot commits to reaching the enemy core instead of
        # chain-intercepting harvester after harvester. Harass (Step 5)
        # uses its own picker and is unaffected.
        self._step2_intercepted = False

        # --- Disruptor Step 3 (intercept enemy harvesters) ---
        self.intercept_target_harvester = None    # (x,y) enemy titanium harvester
        self.intercept_target_conveyor = None     # (x,y) cardinal-adjacent enemy feeder
        self.intercept_substep = 0                # 0=plan, 1=attack-feeder, 2=move-off+place, 3=recheck
        self.intercept_sentinel_tile = None       # (x,y) where the sentinel will go (clean case)
        self.intercept_sentinel_dir = None        # cached direction if chosen ahead of build turn
        self.intercept_destroyed_tile = None      # (x,y) we attacked (for the dirty case)
        self.intercept_clean_case = False         # True = clean (empty tile both-adj), False = dirty (attack-on-tile)

        # --- Disruptor Step 8 (place launchers on core-adjacent ring) ---
        self.launcher_place_target = None         # (x,y) where we want to place the launcher
        self.launcher_place_substep = 0           # 0=plan, 1=approach, 2=destroy-road, 3=step-off, 4=place
        self.launcher_place_road_type = None      # 'none' | 'allied' | 'enemy'
        self.launcher_place_attempted = set()     # (x,y) already tried

        # --- Disruptor Step 9 + builder-side launcher protocol ---
        self.launch_protocol_step = 0             # 0 = inactive, 1..5 = in progress
        self.launch_target_xy = None              # (x,y) the tile we want to be launched ONTO
        self.launch_launcher_xy = None            # (x,y) of the allied launcher we'll use
        self.launch_marker_xy = None              # (x,y) of our placed marker
        self.launch_post_step = None              # disruptor_step to enter after landing

        # --- Disruptor enemy launcher detection (Part 3) ---
        self.prev_intended_pos = None             # (x,y) position at end of our last turn
        self.launched_this_turn = False           # transient: set by detection when a mismatch fires
        self.blocked_launcher_tiles = set()       # (x,y) — enemy launcher + 8-adjacent, permanent
        self.enemy_launcher_positions = set()     # (x,y) — enemy launcher centers we've seen

        # --- Disruptor Case B counter-launcher state (Part 3) ---
        self.case_b_active = False                # True while the counter is running
        self.case_b_target = None                 # (x,y) original attack/build target
        self.case_b_target_kind = None            # 'attack'|'build_gunner'|'build_sentinel'|'build_launcher'
        self.case_b_original_step = None          # disruptor_step to restore after landing
        self.case_b_phase = None                  # 'use_launcher' | 'build_launcher'
        self.case_b_build_site = None             # (x,y) where we'll drop the counter-launcher
        self._acted_on_land = False               # transient: set by attack/build-on-land within same turn

        # --- Disruptor harass mode (Step 5) ---
        self.step10_wander_target = None          # (x,y) current frontier walk goal
        self.step10_entry_turn = None             # round we entered harass
        self.intercept_post_step = 2              # disruptor_step to return to after intercept (2 from Step 2, 5 from harass)
        self.intercept_feeder_blacklist = set()   # (x,y) — feeders we gave up on (A* unreachable)
        self.disrupt_counter = 0                  # idle/explore turns since last disruption
        self.disrupt_target = None                # (x,y) of current disruption target, if any
        self.disrupt_target_destroyed = False     # True once we destroyed disrupt_target — route to barrier
        self.disrupt_heal_start_hp = None         # HP snapshot for heal detection on disrupt target
        self.disrupt_fires_in_window = 0          # count of successful fires since heal window opened
        self.disrupt_anti_heal_launchers = 0      # launchers placed against the current disrupt target's healers (caps at 2)
        # Generic heal-tracking shared by every "fire on own tile until
        # destroyed" attack site (intercept feeder, chain-extension
        # enemy-road, splitter-sentinel enemy-road). Auto-resets when
        # `attack_heal_target` changes — callers don't need to manage
        # the reset themselves.
        self.attack_heal_target = None
        self.attack_heal_start_hp = None
        self.attack_heal_fires = 0
        # Splitter-sentinel target: placement tile cardinally adjacent
        # to an enemy splitter (excluding the splitter's back input
        # tile). The 7-direction scan picks a sentinel facing that
        # excludes the direction toward the splitter (the splitter is
        # the sentinel's ammo source).
        self.disrupt_splitter_target = None      # (placement_xy, splitter_xy)

        # Turret-damage response state.
        # `disrupt_prev_hp` is this bot's HP at the end of the previous
        # turn; an HP drop between turns means something shot us. If a
        # visible enemy turret is within attack range we treat that as
        # the source, abandon the current disrupt target, and add
        # every tile the turret can attack to `disrupt_attack_blacklist`
        # so future attack pickers never pick a target that sits under
        # that turret's cone.
        self.disrupt_prev_hp = None
        self.disrupt_attack_blacklist = set()
        # Tiles inside the attackable cone of any enemy SENTINEL that
        # is cardinally adjacent to a HARVESTER (any team — sentinels
        # accept ammo from either team's harvester). Recomputed every
        # turn from current vision. Below DISRUPTOR_DANGER_HP_FRAC the
        # disruptor avoids these tiles entirely and flees the area;
        # above, they get a heuristic penalty so the greedy prefers
        # safe routes when the detour is cheap.
        self.dangerous_sentinel_tiles = set()
        self._disruptor_low_hp_cached = False
        self.disrupt_blacklist = set()            # (x,y) of disruption targets we've abandoned — never re-pick
        # Opportunistic ore blocker: when a disruptor encounters an
        # enemy/unclaimed titanium or axionite ore within the same
        # detour budget as a harvester intercept, walk adjacent and drop
        # a barrier on the ore. Denies the opponent that ore while still
        # being recoverable later (allied bots destroy our own barriers
        # on chain). Active in Step 2 and Step 5; cleared after the
        # barrier lands or the target becomes invalid.
        self.disrupt_ore_block_target = None      # (x,y) of ore we're walking to block
        # Chain-extension intercept: the disruptor sees an enemy
        # conveyor/splitter/armoured-conveyor whose output is empty/
        # road, or a bridge whose target is empty/road, and walks over
        # to drop a sentinel on the build site so the enemy can't
        # extend their chain. Source tile remembered so the sentinel
        # is faced toward the enemy infrastructure.
        self.disrupt_chain_block_target = None    # (x,y) sentinel placement tile
        self.disrupt_chain_block_source = None    # (x,y) enemy conveyor/bridge feeding it
        # Idle circling: when harass-mode exploration runs out of in-
        # radius frontier tiles, the bot walks a clockwise ring around
        # the enemy core (same waypoint pattern as our own patrol bots
        # use around our core) so it keeps probing fresh angles instead
        # of standing still until something walks into vision.
        # Waypoints are built lazily once `enemy_core_pos` (or the guess)
        # is known and rebuilt if the centre changes.
        self._disruptor_circle_waypoints = None
        self._disruptor_circle_center = None      # (x,y) the waypoints are centred on
        self.disrupt_circle_idx = 0


        # --- Launcher-unit state (this Player instance IS a launcher) ---
        self.launcher_pending_requests = {}      # bot_id_mod -> (tx, ty, marker_xy)
        self.launcher_active_requests = {}       # bot_id_mod -> (tx, ty)
        self.launcher_read_markers = set()       # (x,y) of markers already decoded
        self.launcher_our_core_xy = None         # cached our-team core center
        self.launcher_enemy_core_xy = None       # cached enemy-team core center

    def run(self, ct):
        """Entry point called by game engine each turn."""
        # Detect unit type on first run
        if self.unit_type is None:
            self.unit_type = ct.get_entity_type()

        if self.unit_type == EntityType.CORE:
            self.run_core(ct)
        elif self.unit_type == EntityType.BUILDER_BOT:
            self.run_builder(ct)
        elif self.unit_type == EntityType.SENTINEL:
            self.run_sentinel(ct)
        elif self.unit_type == EntityType.GUNNER:
            self.run_gunner(ct)
        elif self.unit_type == EntityType.LAUNCHER:
            self.run_launcher(ct)

    def run_builder(self, ct):
        """Main builder bot logic — scan then dispatch on step."""
        self._ct = ct  # Store for helpers that need ct without explicit param
        # TLE detection
        if not self.turn_completed:
            self.consecutive_tles += 1
            self.tle_recovery_turns = TLE_RECOVERY_BASE + self.consecutive_tles
            print(f"[{self.corner}] TLE! recovery={self.tle_recovery_turns}t",
                  file=sys.stderr)
        else:
            self.consecutive_tles = 0
        self.turn_completed = False

        # Cache team once per turn
        self.my_team_cache = ct.get_team()

        # Populate tile_cache, bot_pos_cache, etc.
        # _scan_turn also sets self._my_xy — avoids a second get_position() call
        self._scan_turn(ct)
        my_xy = self._my_xy

        # Restore any recently-busted barriers before the step logic runs.
        # Fires at most one rebuild per turn; leftover tiles stay queued.
        if self.pending_barrier_restores:
            self._barrier_bust_restore(ct)

        # Record position history (deque auto-evicts oldest — O(1) append)
        self.pos_history.append(my_xy)

        # Detect corner assignment on first builder run. Re-run on
        # subsequent turns while the bot is still tagged "??" — that
        # tag means our core wasn't in vision on the first turn (e.g.
        # because a launcher threw the bot away from the core before
        # its run() fired). Once the bot wanders back into core
        # vision, _assign_role can finish its job and assign a real
        # corner instead of leaving the bot stuck in fallback mode.
        if (self.corner is None and self.role is None) or self.corner == "??":
            self._assign_role(ct, my_xy)

        # Log step transitions
        if DEBUG and self.step != self._prev_step:
            print(f"[{self.corner}] {self._prev_step} -> {self.step} at ({my_xy[0]},{my_xy[1]})",
                  file=sys.stderr)
            self._prev_step = self.step

        # Repair interruption — higher priority than ore/explore but lower than chain
        if (self.role == 'economy'
                and self.pending_repairs
                and self.repair_target is None
                and self.step in ("explore", "goto_ore", "leave_core")
                and self.tle_recovery_turns == 0):
            self._pre_repair_step = self.step
            self._pre_repair_target_ore = self.target_ore
            self._pre_repair_frontier = self.frontier_target
            best_rxy = min(self.pending_repairs,
                           key=lambda r: (r[0] - my_xy[0]) ** 2 + (r[1] - my_xy[1]) ** 2)
            self.pending_repairs.remove(best_rxy)
            self.repair_target = best_rxy
            self.path = None
            self.path_index = 0
            self.step = "repair"

        # Dispatch by role
        if self.role == 'patrol':
            self._run_patrol(ct)
        elif self.role == 'destructor':
            self._run_destructor(ct)
        elif self.role == 'disruptor':
            self._run_disruptor(ct)
        else:
            step_fn = self._step_dispatch.get(self.step)
            if step_fn:
                step_fn(self, ct)

        # Economy bot status to stdout (visible in replay)
        if self.role not in ('patrol', 'destructor', 'disruptor'):
            self._print_status(ct)
        elif self.role == 'disruptor':
            self._print_disruptor_status(ct)

        # Self-heal if idle (action cooldown unused). Disruptors heal
        # on ANY damage — they spend a lot of time alone in enemy
        # territory without other allies nearby to heal them, and a
        # burning launcher/turret can finish off a 70%-HP disruptor in
        # a couple of shots. Other roles still wait for <50% HP so they
        # don't starve their own builds on trivial chip damage.
        if ct.get_action_cooldown() == 0:
            cur_hp = ct.get_hp()
            max_hp = GC.BUILDER_BOT_MAX_HP
            threshold = max_hp if self.role == 'disruptor' else max_hp // 2
            if cur_hp < threshold:
                my_pos = xy_to_pos(my_xy)
                if ct.can_heal(my_pos):
                    ct.heal(my_pos)

        # Post-dispatch: oscillation + top-level stuck detection
        self._check_oscillation(my_xy)
        self._check_top_level_stuck(ct, my_xy)

        # Debug: draw path + timing warnings
        if DEBUG:
            elapsed = ct.get_cpu_time_elapsed()
            if elapsed < 1000:
                self._draw_path(ct, my_xy)
            elif elapsed > 1800:
                print(f"[{self.corner}] SLOW {self.step} {elapsed}us",
                      file=sys.stderr)

        # Record end-of-turn position for next turn's enemy-launcher detection.
        # Anything that moved us between turns (an enemy launcher picking us
        # up) will make next turn's get_position differ from this value.
        # Disruptors use it for Case A/B response; patrol bots use it to
        # blacklist the launcher footprint for pathfinding.
        if self.role in ('disruptor', 'patrol'):
            end_pos = ct.get_position()
            self.prev_intended_pos = (end_pos.x, end_pos.y)

        # TLE recovery countdown + mark turn complete
        if self.tle_recovery_turns > 0:
            self.tle_recovery_turns -= 1
        self.turn_completed = True

    def _assign_role(self, ct, my_xy):
        """Determine bot role from spawn position relative to core.

        Panic override: if the core HP is below PANIC_HP_THRESHOLD when
        this bot is being assigned, it becomes a patrol regardless of
        which core tile it spawned on. Matches core_logic's panic-mode
        flood spawn.
        """
        from constants import PANIC_HP_THRESHOLD
        # Find our core — use the building scan position (center tile).
        # Once set, don't re-derive from tile_cache iteration (which
        # could pick a corner tile instead of the center).
        core_bid = None
        if self.core_pos is not None:
            # Already know the center — just find the bid for this turn.
            ce = self.tile_cache.get(self.core_pos)
            if ce and ce[2] == EntityType.CORE and ce[3] == self.my_team_cache:
                core_bid = ce[1]
        if core_bid is None:
            for xy, (env, bid, etype, team) in self.tile_cache.items():
                if etype == EntityType.CORE and team == self.my_team_cache:
                    if self.core_pos is None:
                        # First discovery — get_position returned the
                        # center tile during the scan, so this is correct
                        # on the first turn. On later turns, verify it's
                        # the center by checking that all 8 neighbors are
                        # also core tiles in tile_cache.
                        self.core_pos = xy
                    core_bid = bid
                    break

        if self.core_pos is None:
            # Fallback: can't see core yet
            self.corner = "??"
            self.role = "economy"
            return

        cx, cy = self.core_pos
        dx = my_xy[0] - cx
        dy = my_xy[1] - cy

        # Populate all 9 core tiles in tile_cache AND building_cache
        # so _follow_path knows they're walkable (no road needed),
        # splitter checks recognize adjacent core tiles, and building_cache
        # diagnostics show the correct type.
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                cxy = (cx + ddx, cy + ddy)
                ce = self.tile_cache.get(cxy)
                if ce is None or ce[1] is None:
                    self.tile_cache[cxy] = (
                        Environment.EMPTY, core_bid, EntityType.CORE,
                        self.my_team_cache)
                if cxy not in self.building_cache:
                    self.building_cache[cxy] = (EntityType.CORE, None)

        # Initialize map dimensions + symmetry if not set
        if self.map_w is None:
            self.map_w = ct.get_map_width()
            self.map_h = ct.get_map_height()
            self._symmetry = detect_symmetry(self.core_pos, self.map_w, self.map_h)

        # Panic override — core damaged, this bot is a patrol.
        if core_bid is not None:
            try:
                hp = ct.get_hp(core_bid)
                max_hp = ct.get_max_hp(core_bid)
            except Exception:
                hp = max_hp = None
            if (hp is not None and max_hp
                    and hp < max_hp * PANIC_HP_THRESHOLD):
                self.corner = "P"
                self.role = "patrol"
                self.step = "patrol"
                self._patrol_spawn_round = ct.get_current_round()
                return

        # Patrol bots spawn on East middle core tile at (cx+1, cy)
        if dx == 1 and dy == 0:
            self.corner = "P"
            self.role = "patrol"
            self.step = "patrol"
            self._patrol_spawn_round = ct.get_current_round()
            return

        # Destructor bots spawn on West middle core tile at (cx-1, cy)
        if dx == -1 and dy == 0:
            self.corner = "D"
            self.role = "destructor"
            self.step = "destructor"
            return

        # Disruptor bots spawn on South middle core tile at (cx, cy+1)
        if dx == 0 and dy == 1:
            self.corner = "DS"
            self.role = "disruptor"
            self.step = "disruptor"
            return

        if dx <= 0 and dy <= 0:
            self.corner = "NW"
        elif dx >= 0 and dy <= 0:
            self.corner = "NE"
        elif dx <= 0 and dy >= 0:
            self.corner = "SW"
        else:
            self.corner = "SE"

        self.role = "economy"

    # Step dispatch table — maps step name to method
    _step_dispatch = {
        "leave_core":       lambda self, ct: self._step_leave_core(ct),
        "explore":          lambda self, ct: self._step_explore(ct),
        "goto_ore":         lambda self, ct: self._step_goto_ore(ct),
        "claim_ore":        lambda self, ct: self._step_claim_ore(ct),
        "build_harvester":  lambda self, ct: self._step_build_harvester(ct),
        "chain_to_core":    lambda self, ct: (
            self._step_chain_to_core_axionite(ct)
            if self.current_chain_type == 'axionite'
            else self._step_chain_to_core(ct)
        ),
        "chain_seal_roads": lambda self, ct: self._step_chain_seal_roads(ct),
        "steal_harvester":  lambda self, ct: self._step_steal_harvester(ct),
        "observe_conveyor": lambda self, ct: self._step_observe_conveyor(ct),
        "repair":           lambda self, ct: self._step_repair(ct),
    }

    def _print_status(self, ct):
        """Print economy bot status to stdout (visible in replay)."""
        my_xy = self._my_xy
        s = self.step
        c = self.corner or "??"
        parts = [f"[{c}] ({my_xy[0]},{my_xy[1]}) {s}"]
        if s == "explore":
            if self.frontier_target:
                parts.append(f"ft={self.frontier_target}")
            parts.append(f"h={self.harvesters_built}")
        elif s == "goto_ore":
            if self.target_ore:
                parts.append(f"ore={self.target_ore}")
        elif s == "claim_ore":
            if self.target_ore:
                parts.append(f"ore={self.target_ore}")
            parts.append(f"cs={self.chain_start} sent={self._sentinel_placed}")
        elif s == "build_harvester":
            if self.target_ore:
                parts.append(f"ore={self.target_ore}")
            parts.append(f"moved={self._harvester_moved}")
        elif s == "chain_to_core":
            ci = self.chain_index
            cl = len(self.chain_path) if self.chain_path else 0
            parts.append(f"idx={ci}/{cl}")
            if self.chain_path and ci < cl:
                parts.append(f"tgt={self.chain_path[ci]}")
            if self._bridge_walk_target:
                parts.append(f"bwalk={self._bridge_walk_target}")
            if self._pending_harvester_xy:
                parts.append(f"pharvest={self._pending_harvester_xy}")
            ti, _ = ct.get_global_resources()
            parts.append(f"ti={ti}")
        elif s == "observe_conveyor":
            parts.append(f"xy={self.observe_xy} left={self.observe_turns_left}")
        elif s == "repair":
            parts.append(f"rxy={self.repair_target}")
        elif s == "steal_harvester":
            parts.append(f"steal={self.steal_target} st={self.sentinel_tile}")
        if self._last_build_action:
            parts.append(f"BUILD:{self._last_build_action}")
            self._last_build_action = None
        print(" ".join(parts))
