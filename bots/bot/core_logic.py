import sys
from cambc import EntityType, Environment, Position, GameConstants as GC
from constants import (
    DEBUG, ECONOMY_BOTS_MAX,
    TITANIUM_ORE_THRESHOLD,
    ECONOMY_BOTS_HIGH_ORE, ECONOMY_BOTS_MED_ORE, ECONOMY_BOTS_LOW_ORE,
    PATROL_INITIAL_ROUND, PATROL_ENEMY_BOT_TRIGGER_ROUND,
    PATROL_PER_ENEMY_BOT, PATROL_MAX_COUNT, PATROL_DAMAGE_THRESHOLD,
    PANIC_HP_THRESHOLD, PANIC_STEP_PCT,
    PANIC_INITIAL_SPAWNS, PANIC_STEP_SPAWNS,
    REPAIR_TEST, REPAIR_TEST_ROUND, REPAIR_TEST_COUNT,
    CONVERT_AXIONITE, MIN_TITANIUM,
    EARLY_PATROL, ECONOMY_TEST_MODE,
    START_DISRUPTOR_BOTS, DISRUPTOR_TITANIUM, MAX_DISRUPTORS,
    DISRUPTOR_SPAWN_INTERVAL,
    START_DISRUPTORS_MAX, START_DISRUPTORS_AREA_WEIGHT,
    START_DISRUPTORS_CENTER_WEIGHT, START_DISRUPTORS_ORE_WEIGHT,
    START_DISRUPTORS_REF_AREA,
)
from utils import detect_symmetry


# Corner offsets from core center (3x3 core: center ± 1)
_CORNER_OFFSETS = [
    (-1, -1),  # NW  (top-left)     idx=0
    ( 1, -1),  # NE  (top-right)    idx=1
    (-1,  1),  # SW  (bottom-left)  idx=2
    ( 1,  1),  # SE  (bottom-right) idx=3
]

# Enemy turret types that trigger a patrol-spawn response when first
# seen in vision. Launcher is excluded — it can't attack buildings.
_ENEMY_TURRET_TYPES = frozenset({
    EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH,
})

# Patrol bots spawn on the East middle tile (cx+1, cy) in normal play —
# _assign_role detects the role by the spawn-tile offset, and the other
# core tiles are reserved for economy / destructor / future roles.
_PATROL_SPAWN_OFFSET = (1, 0)

# During a panic event (core HP < PANIC_HP_THRESHOLD) the core floods
# out patrols on any of the 9 footprint tiles it can reach, and
# _assign_role forces the spawned bot into the patrol role regardless
# of offset. Normal (non-patrol) role spawns are postponed until panic
# ends. East is tried first so bots on the default patrol tile still
# work like usual.
_PANIC_SPAWN_OFFSETS = [
    ( 1,  0),
    ( 0,  1),
    (-1,  0),
    ( 0, -1),
    ( 1,  1),
    (-1,  1),
    ( 1, -1),
    (-1, -1),
    ( 0,  0),
]


class CoreMixin:
    """Handles core unit spawning logic."""

    def run_core(self, ct):
        """Called each turn for the core unit."""
        if self.core_pos is None:
            pos = ct.get_position()
            self.core_pos = (pos.x, pos.y)
            self.my_team_cache = ct.get_team()
            self.map_w = ct.get_map_width()
            self.map_h = ct.get_map_height()
            self._symmetry = detect_symmetry(self.core_pos, self.map_w, self.map_h)
            # One-time titanium-ore scan in core vision: drives both the
            # economy bot count and the starting-disruptor cap. Cached so
            # `_pick_economy_spawn_corner` doesn't re-scan per spawn.
            ti_count, nearest_ti = self._scan_titanium_ores_in_core_vision(ct)
            self._titanium_ore_count = ti_count
            self._first_titanium_ore = nearest_ti
            # Economy still uses ore tiers — fewer ores means more economy
            # bots are needed to extract what little there is.
            if ti_count > TITANIUM_ORE_THRESHOLD:
                self._economy_bots_target = ECONOMY_BOTS_HIGH_ORE
            elif ti_count >= 1:
                self._economy_bots_target = ECONOMY_BOTS_MED_ORE
            else:
                self._economy_bots_target = ECONOMY_BOTS_LOW_ORE
            # Disruptor count: ore_count is now a SLIGHT influence on the
            # weighted formula instead of a hard cap. This matters on
            # small maps where ores often sit just outside the core's
            # vision radius — under the old hard cap, ti_count==0 would
            # zero out disruptors even though the map was small enough
            # to make the trip cheap.
            self._start_disruptor_target = self._compute_start_disruptor_target(
                ti_count)
            if DEBUG:
                print(f"[CORE] tier ti_ore={ti_count} "
                      f"economy={self._economy_bots_target} "
                      f"disruptor={self._start_disruptor_target}",
                      file=sys.stderr)
            # Patrol tracking state
            self.patrol_bots_spawned = 0
            self.seen_enemy_bot_ids = set()
            self.seen_enemy_turret_ids = set()
            self.damage_events = 0
            # Bids that have already triggered a damage-driven patrol
            # response. Each building only counts once — sustained
            # damage to the same tile shouldn't keep spawning patrols.
            self.damaged_bid_seen = set()
            self.initial_patrol_spawned = False
            self.buildings_hp_cache = {}
            self.patrol_spawn_queue = 0
            self.early_patrol_spawned = 0  # damage-triggered spawns that bypass MIN_TI + round gate
            self.early_patrol_queue = 0    # queued early spawns (bypass _can_spend)
            # Panic mode: extra spawn budget above PATROL_MAX_COUNT when
            # the core takes serious damage.
            self.panic_thresholds_triggered = set()
            self.panic_extra_cap = 0
            self.core_max_hp = None
            # Repair test: destructor tracking
            self.destructor_bots_spawned = 0
            # Disruptor spawning (Phase 3)
            self.pending_disruptor_spawns = 0    # remaining queued normal-tier spawns
            self.titanium_threshold_met = False  # one-shot latch for the 4-spawn batch
            self._last_disruptor_turn_spawn = 0  # round of most recent cadence spawn

        self.my_team_cache = ct.get_team()

        # Convert axionite → titanium when Ti falls below the scaled floor
        if CONVERT_AXIONITE and ECONOMY_TEST_MODE:
            ti, ax = ct.get_global_resources()
            scale = ct.get_scale_percent() / 100.0
            floor = int(MIN_TITANIUM * scale)
            if ti < floor and ax > 0:
                deficit = floor - ti
                # 1 axionite → 4 titanium
                ax_needed = -(-deficit // 4)  # ceiling division
                amount = min(ax_needed, ax)
                ct.convert(amount)
                if DEBUG:
                    print(f"[CORE] converted {amount} Ax → {amount * 4} Ti "
                          f"(ti={ti} floor={floor})", file=sys.stderr)

        # --- Priority 1: opening disruptor spawns ---
        # Highest priority — consumes the core's one-action-per-turn slot
        # before economy corner spawns. If the south-middle tile is
        # currently blocked (previous disruptor hasn't moved off), fall
        # through to economy spawning this round and retry next round.
        if (not ECONOMY_TEST_MODE
                and self.disruptor_spawned < self._start_disruptor_target
                and ct.get_action_cooldown() == 0):
            if self._try_spawn_disruptor(ct):
                return

        # Spawn economy bots on corners immediately (count tiered by
        # how many titanium ores the core saw on turn 1).
        if self.economy_spawned < self._economy_bots_target:
            if (ct.get_action_cooldown() == 0
                    and self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0])):
                idx = self._pick_economy_spawn_corner(ct)
                if idx is not None:
                    offset = _CORNER_OFFSETS[idx]
                    sx = self.core_pos[0] + offset[0]
                    sy = self.core_pos[1] + offset[1]
                    spawn_pos = Position(sx, sy)
                    if ct.can_spawn(spawn_pos):
                        ct.spawn_builder(spawn_pos)
                        self.economy_spawned += 1
                        self._economy_used_corners.add(idx)
                        if DEBUG:
                            print(f"[CORE] spawned economy #{self.economy_spawned} "
                                  f"at ({sx},{sy}) corner_idx={idx}",
                                  file=sys.stderr)
            return  # Prioritize economy spawns first

        # --- Patrol bot triggers ---
        current_round = ct.get_current_round()

        # Effective spawn cap = base max + any panic extras awarded below
        effective_cap = PATROL_MAX_COUNT + self.panic_extra_cap
        # Remaining capacity (cap queue to remaining spawn budget)
        remaining = effective_cap - self.patrol_bots_spawned - self.patrol_spawn_queue

        # Trigger 1: initial spawn at PATROL_INITIAL_ROUND
        if not self.initial_patrol_spawned and current_round >= PATROL_INITIAL_ROUND:
            add = min(2, max(0, remaining))
            self.patrol_spawn_queue += add
            remaining -= add
            self.initial_patrol_spawned = True
            if DEBUG:
                print(f"[CORE] initial patrol trigger at round {current_round}",
                      file=sys.stderr)

        # Trigger 2: new enemy builder bots in vision (inactive before trigger round)
        if current_round >= PATROL_ENEMY_BOT_TRIGGER_ROUND:
            for uid in ct.get_nearby_units():
                if ct.get_team(uid) == self.my_team_cache:
                    continue
                if ct.get_entity_type(uid) != EntityType.BUILDER_BOT:
                    continue
                if uid not in self.seen_enemy_bot_ids:
                    self.seen_enemy_bot_ids.add(uid)
                    add = min(PATROL_PER_ENEMY_BOT, max(0, remaining))
                    self.patrol_spawn_queue += add
                    remaining -= add
                    if DEBUG and add > 0:
                        print(f"[CORE] new enemy bot seen, +{add} patrol",
                              file=sys.stderr)

        # Trigger 2b: new enemy turrets in vision. Fires immediately —
        # turrets are a direct threat to allied buildings and the core,
        # so we always respond without waiting on the bot-trigger round.
        for bid in ct.get_nearby_buildings():
            if ct.get_team(bid) == self.my_team_cache:
                continue
            if ct.get_entity_type(bid) not in _ENEMY_TURRET_TYPES:
                continue
            if bid not in self.seen_enemy_turret_ids:
                self.seen_enemy_turret_ids.add(bid)
                add = min(PATROL_PER_ENEMY_BOT, max(0, remaining))
                self.patrol_spawn_queue += add
                remaining -= add
                if DEBUG and add > 0:
                    print(f"[CORE] new enemy turret seen, +{add} patrol",
                          file=sys.stderr)

        # Trigger 3: damage to allied buildings in vision
        # Damage-triggered spawns can bypass MIN_TITANIUM and the round
        # gate up to EARLY_PATROL total, letting the core respond to
        # early aggression before the economy is established.
        new_hp_cache = {}
        for bid in ct.get_nearby_buildings():
            if ct.get_team(bid) != self.my_team_cache:
                continue
            hp = ct.get_hp(bid)
            new_hp_cache[bid] = hp
            prev_hp = self.buildings_hp_cache.get(bid)
            if (prev_hp is not None and hp < prev_hp
                    and bid not in self.damaged_bid_seen):
                # First damage event observed for this building — count
                # it and remember the bid so subsequent hits on the
                # same building don't keep triggering patrol spawns.
                self.damaged_bid_seen.add(bid)
                self.damage_events += 1
                if self.damage_events == 1 or (self.damage_events - 1) % PATROL_DAMAGE_THRESHOLD == 0:
                    add = min(2, max(0, remaining))
                    # Route through early queue if under the cap
                    early_remaining = EARLY_PATROL - self.early_patrol_spawned - self.early_patrol_queue
                    early = min(add, max(0, early_remaining))
                    normal = add - early
                    self.early_patrol_queue += early
                    self.patrol_spawn_queue += normal
                    remaining -= add
                    if DEBUG and add > 0:
                        print(f"[CORE] damage event #{self.damage_events}, "
                              f"+{early} early +{normal} normal patrol",
                              file=sys.stderr)
        self.buildings_hp_cache = new_hp_cache

        # --- Panic mode: extra patrol spawns when the core is damaged ---
        core_hp = ct.get_hp()
        if self.core_max_hp is None:
            self.core_max_hp = ct.get_max_hp()
        panic_active = False
        if self.core_max_hp and self.core_max_hp > 0:
            hp_frac = core_hp / self.core_max_hp
            panic_active = hp_frac < PANIC_HP_THRESHOLD
            # Thresholds: 0.5, 0.4, 0.3, 0.2, 0.1 (inclusive of initial 0.5)
            threshold = PANIC_HP_THRESHOLD
            step = PANIC_STEP_PCT
            while threshold > 0:
                key = round(threshold, 3)
                if hp_frac < threshold and key not in self.panic_thresholds_triggered:
                    self.panic_thresholds_triggered.add(key)
                    spawns = (PANIC_INITIAL_SPAWNS
                              if key == round(PANIC_HP_THRESHOLD, 3)
                              else PANIC_STEP_SPAWNS)
                    self.panic_extra_cap += spawns
                    self.patrol_spawn_queue += spawns
                    if DEBUG:
                        print(f"[CORE] PANIC @{int(hp_frac * 100)}% HP "
                              f"(threshold {int(threshold * 100)}%) → +{spawns} patrol",
                              file=sys.stderr)
                threshold -= step
            effective_cap = PATROL_MAX_COUNT + self.panic_extra_cap

        # Execute queued patrol spawns.
        #
        # Three tiers with different resource gates:
        #   1. Panic spawns — bypass everything (dying core is the emergency)
        #   2. Early damage spawns — bypass _can_spend + round gate, capped
        #      at EARLY_PATROL total. Lets the core respond to aggression
        #      before the economy is established.
        #   3. Normal spawns — gated by _can_spend (MIN_TITANIUM floor)
        total_queue = self.patrol_spawn_queue + self.early_patrol_queue
        if (total_queue > 0
                and self.patrol_bots_spawned < effective_cap
                and ct.get_action_cooldown() == 0):
            cx, cy = self.core_pos
            if panic_active:
                for offset in _PANIC_SPAWN_OFFSETS:
                    spawn_pos = Position(cx + offset[0], cy + offset[1])
                    if ct.can_spawn(spawn_pos):
                        ct.spawn_builder(spawn_pos)
                        self.patrol_bots_spawned += 1
                        self.patrol_spawn_queue -= 1
                        if DEBUG:
                            print(f"[CORE] PANIC patrol #{self.patrol_bots_spawned} "
                                  f"@({spawn_pos.x},{spawn_pos.y}) queue={self.patrol_spawn_queue}",
                                  file=sys.stderr)
                        break
            elif self.early_patrol_queue > 0:
                # Early damage-triggered spawn: bypass _can_spend
                spawn_pos = Position(cx + _PATROL_SPAWN_OFFSET[0],
                                     cy + _PATROL_SPAWN_OFFSET[1])
                if ct.can_spawn(spawn_pos):
                    ct.spawn_builder(spawn_pos)
                    self.patrol_bots_spawned += 1
                    self.early_patrol_spawned += 1
                    self.early_patrol_queue -= 1
                    if DEBUG:
                        print(f"[CORE] EARLY patrol #{self.early_patrol_spawned} "
                              f"@({spawn_pos.x},{spawn_pos.y}) queue={self.early_patrol_queue}",
                              file=sys.stderr)
            elif self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0]):
                spawn_pos = Position(cx + _PATROL_SPAWN_OFFSET[0],
                                     cy + _PATROL_SPAWN_OFFSET[1])
                if ct.can_spawn(spawn_pos):
                    ct.spawn_builder(spawn_pos)
                    self.patrol_bots_spawned += 1
                    self.patrol_spawn_queue -= 1
                    if DEBUG:
                        print(f"[CORE] patrol #{self.patrol_bots_spawned} "
                              f"@({spawn_pos.x},{spawn_pos.y}) queue={self.patrol_spawn_queue}",
                              file=sys.stderr)

        # --- Priority 3: titanium-threshold batch (one-shot, 4 disruptors) ---
        ti, _ = ct.get_global_resources()
        if (not ECONOMY_TEST_MODE
                and not self.titanium_threshold_met
                and ti >= DISRUPTOR_TITANIUM):
            self.titanium_threshold_met = True
            cap_left = MAX_DISRUPTORS - self.disruptor_spawned
            self.pending_disruptor_spawns = min(4, max(0, cap_left))
            if DEBUG and self.pending_disruptor_spawns > 0:
                print(f"[CORE] Ti>={DISRUPTOR_TITANIUM} disruptor batch queued "
                      f"({self.pending_disruptor_spawns})", file=sys.stderr)

        # Drain the queue one spawn per turn (bypasses the 20-turn cadence).
        if (not ECONOMY_TEST_MODE
                and self.pending_disruptor_spawns > 0
                and ct.get_action_cooldown() == 0
                and ct.get_unit_count() < GC.MAX_TEAM_UNITS
                and self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0])):
            # Top up economy to ECONOMY_BOTS_MAX before spending Ti on
            # disruptors. Skipped corners on the early tier (HIGH/MED
            # ore) get filled in here.
            if self._topup_economy_at_threshold(ct):
                return
            if self._try_spawn_disruptor(ct):
                self.pending_disruptor_spawns -= 1
            return

        # --- Priority 5: 20-turn cadence, strictly below patrol priority ---
        if (not ECONOMY_TEST_MODE
                and not panic_active
                and self.disruptor_spawned < MAX_DISRUPTORS
                and current_round - self._last_disruptor_turn_spawn >= DISRUPTOR_SPAWN_INTERVAL
                and ti > DISRUPTOR_TITANIUM
                and ct.get_action_cooldown() == 0
                and ct.get_unit_count() < GC.MAX_TEAM_UNITS
                and self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0])):
            # Same late-game top-up gate as the queue-drain path.
            if self._topup_economy_at_threshold(ct):
                return
            if self._try_spawn_disruptor(ct):
                self._last_disruptor_turn_spawn = current_round
                return

        # Non-patrol role spawning (destructor etc.) is postponed during
        # panic — the core concentrates its action cooldown on patrols.
        if panic_active:
            return

        # Repair test: spawn destructor bots at REPAIR_TEST_ROUND on West middle tile
        if (REPAIR_TEST
                and current_round >= REPAIR_TEST_ROUND
                and self.destructor_bots_spawned < REPAIR_TEST_COUNT
                and ct.get_action_cooldown() == 0
                and self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0])):
            spawn_pos = Position(self.core_pos[0] - 1, self.core_pos[1])
            if ct.can_spawn(spawn_pos):
                ct.spawn_builder(spawn_pos)
                self.destructor_bots_spawned += 1
                if DEBUG:
                    print(f"[CORE] destructor #{self.destructor_bots_spawned} spawned",
                          file=sys.stderr)

    def _pick_economy_spawn_corner(self, ct):
        """Pick the next economy spawn corner index (0..3 = NW/NE/SW/SE).

        Priority A — first spawn only: pick the corner closest to the
        nearest titanium ore the core saw on turn 1 (axionite excluded
        — economy bots harvest titanium first, axionite is later-game).

        Priority B — every other spawn (or no titanium in vision): pick
        the unused corner closest to the centre of the map. Bots that
        fan out toward the centre cover the largest unexplored area
        first.

        Returns None if every corner is already used.
        """
        used = self._economy_used_corners
        available = [i for i in range(len(_CORNER_OFFSETS)) if i not in used]
        if not available:
            return None
        cx, cy = self.core_pos

        # Priority A: first spawn + titanium ore visible to the core.
        if not used and self._first_titanium_ore is not None:
            ore_xy = self._first_titanium_ore
            best_i = available[0]
            best_d2 = None
            for i in available:
                ox = cx + _CORNER_OFFSETS[i][0]
                oy = cy + _CORNER_OFFSETS[i][1]
                d2 = (ox - ore_xy[0]) ** 2 + (oy - ore_xy[1]) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            return best_i

        # Priority B: closest to map centre.
        if self.map_w is None:
            return available[0]
        mcx = (self.map_w - 1) / 2.0
        mcy = (self.map_h - 1) / 2.0
        best_i = available[0]
        best_d2 = None
        for i in available:
            ox = cx + _CORNER_OFFSETS[i][0]
            oy = cy + _CORNER_OFFSETS[i][1]
            d2 = (ox - mcx) ** 2 + (oy - mcy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _scan_titanium_ores_in_core_vision(self, ct):
        """Single-pass scan over the core's `get_nearby_tiles`. Returns
        (count, nearest_xy) of TITANIUM ore only. Called once on turn 1
        and the result is cached on `self`."""
        cx, cy = self.core_pos
        count = 0
        nearest = None
        nearest_d2 = None
        for tile in ct.get_nearby_tiles():
            if ct.get_tile_env(tile) != Environment.ORE_TITANIUM:
                continue
            count += 1
            d2 = (tile.x - cx) ** 2 + (tile.y - cy) ** 2
            if nearest_d2 is None or d2 < nearest_d2:
                nearest_d2 = d2
                nearest = (tile.x, tile.y)
        return count, nearest

    def _topup_economy_at_threshold(self, ct):
        """Late-game economy top-up: spawn one economy bot in an unused
        corner. Called from the disruptor spawn paths to ensure we
        reach ECONOMY_BOTS_MAX before draining titanium into more
        disruptors. Returns True if a spawn fired this turn."""
        if self.economy_spawned >= ECONOMY_BOTS_MAX:
            return False
        if not self._can_spend(ct, GC.BUILDER_BOT_BASE_COST[0]):
            return False
        idx = self._pick_economy_spawn_corner(ct)
        if idx is None:
            return False
        offset = _CORNER_OFFSETS[idx]
        sx = self.core_pos[0] + offset[0]
        sy = self.core_pos[1] + offset[1]
        spawn_pos = Position(sx, sy)
        if not ct.can_spawn(spawn_pos):
            return False
        ct.spawn_builder(spawn_pos)
        self.economy_spawned += 1
        self._economy_used_corners.add(idx)
        if DEBUG:
            print(f"[CORE] late-game economy #{self.economy_spawned} "
                  f"at ({sx},{sy}) corner_idx={idx}", file=sys.stderr)
        return True

    def _try_spawn_disruptor(self, ct):
        """Spawn a disruptor on the south-middle core tile (cx, cy+1)."""
        cx, cy = self.core_pos
        spawn_pos = Position(cx, cy + 1)
        if not ct.can_spawn(spawn_pos):
            return False
        ct.spawn_builder(spawn_pos)
        self.disruptor_spawned += 1
        if DEBUG:
            print(f"[CORE] disruptor #{self.disruptor_spawned} at ({cx},{cy + 1})",
                  file=sys.stderr)
        return True

    def _compute_start_disruptor_target(self, ore_count):
        """Dynamic count for the Turn-1/2 disruptor burst.

        Uses a weighted AVERAGE of three factors (each in [0, 1])
        scaled linearly into [START_DISRUPTOR_BOTS, START_DISRUPTORS_MAX]:

            blend  = (W_AREA * area + W_CENTER * center + W_ORE * ore)
                   / (W_AREA + W_CENTER + W_ORE)
            raw    = START_DISRUPTOR_BOTS
                   + blend * (START_DISRUPTORS_MAX - START_DISRUPTOR_BOTS)
            count  = round(raw)

        blend is naturally in [0, 1]; raw is naturally in
        [START_DISRUPTOR_BOTS, START_DISRUPTORS_MAX]. Weights control
        the RELATIVE influence of each factor without being able to
        push raw past the ceiling. Setting all weights to 0 falls back
        to the floor count.

        area_factor    : 0 at a full 50x50 map, ~0.84 at the smallest 20x20.
        center_factor  : 0 when our core sits in the corner, ~1 at the centre.
        ore_factor     : 0 when no titanium ore visible, 1 at >= threshold,
                         linear in between. Slight weight only — small maps
                         frequently hide ore just outside core vision and we
                         don't want that to zero out the disruptor count.
        """
        cx = (self.map_w - 1) / 2.0
        cy = (self.map_h - 1) / 2.0
        area = self.map_w * self.map_h
        area_factor = max(0.0, 1.0 - area / START_DISRUPTORS_REF_AREA)

        dx = self.core_pos[0] - cx
        dy = self.core_pos[1] - cy
        core_dist = (dx * dx + dy * dy) ** 0.5
        max_dist = (cx * cx + cy * cy) ** 0.5
        center_factor = (max(0.0, 1.0 - core_dist / max_dist)
                         if max_dist > 0 else 0.0)

        if TITANIUM_ORE_THRESHOLD > 0:
            ore_factor = min(1.0, ore_count / TITANIUM_ORE_THRESHOLD)
        else:
            ore_factor = 1.0 if ore_count > 0 else 0.0

        total_weight = (START_DISRUPTORS_AREA_WEIGHT
                        + START_DISRUPTORS_CENTER_WEIGHT
                        + START_DISRUPTORS_ORE_WEIGHT)
        if total_weight > 0:
            blend = (START_DISRUPTORS_AREA_WEIGHT * area_factor
                     + START_DISRUPTORS_CENTER_WEIGHT * center_factor
                     + START_DISRUPTORS_ORE_WEIGHT * ore_factor) / total_weight
        else:
            blend = 0.0

        raw = (START_DISRUPTOR_BOTS
               + blend * (START_DISRUPTORS_MAX - START_DISRUPTOR_BOTS))
        count = int(round(raw))
        # Defensive clamp — round() can't violate the bounds given valid
        # inputs, but this guards against float drift and negative ranges.
        count = max(START_DISRUPTOR_BOTS, min(START_DISRUPTORS_MAX, count))
        if DEBUG:
            print(f"[CORE] start_disruptor_target={count} "
                  f"(area={area} area_factor={area_factor:.2f} "
                  f"core_dist={core_dist:.1f} center_factor={center_factor:.2f} "
                  f"ore_count={ore_count} ore_factor={ore_factor:.2f} "
                  f"blend={blend:.2f} raw={raw:.2f})", file=sys.stderr)
        return count
