"""Launcher turret AI.

Responsibilities:
  1. Read allied markers carrying LAUNCHER_PROTOCOL requests and stage them.
  2. When the requesting builder destroys its marker, promote the request to active.
  3. On the builder's next adjacent approach, launch it to its target.
  4. Otherwise, launch any adjacent enemy builder bot away from our core / toward
     the enemy core when either is in vision.

Each launcher has its own Player instance — state persists across turns per unit
but is NOT shared across launchers. Markers are the only inter-unit channel.
"""
import sys
from cambc import EntityType, Position, GameConstants as GC
from constants import DEBUG, ECONOMY_TEST_MODE, LAUNCHER_PROTOCOL_PREFIX


# Decoded protocol values occupy [PREFIX, PREFIX + 100_000_000) —
# bot_id_mod < 10000, tx < 100, ty < 100. Bounded range-check so random
# u32 marker values don't get mis-decoded as launch requests.
_PROTOCOL_SPAN = 100_000_000


class LauncherMixin:
    def run_launcher(self, ct):
        if ECONOMY_TEST_MODE:
            return

        my_team = ct.get_team()
        my_pos = ct.get_position()
        my_xy = (my_pos.x, my_pos.y)

        nearby_buildings = list(ct.get_nearby_buildings())
        nearby_units = list(ct.get_nearby_units())

        # --- Refresh our/enemy core cache from vision ---
        for bid in nearby_buildings:
            et = ct.get_entity_type(bid)
            if et != EntityType.CORE:
                continue
            bp = ct.get_position(bid)
            if ct.get_team(bid) == my_team:
                self.launcher_our_core_xy = (bp.x, bp.y)
            else:
                self.launcher_enemy_core_xy = (bp.x, bp.y)

        # --- Step 1: decode allied markers carrying protocol values ---
        for bid in nearby_buildings:
            if ct.get_entity_type(bid) != EntityType.MARKER:
                continue
            if ct.get_team(bid) != my_team:
                continue
            mpos = ct.get_position(bid)
            mxy = (mpos.x, mpos.y)
            if mxy in self.launcher_read_markers:
                continue
            value = ct.get_marker_value(bid)
            if not (LAUNCHER_PROTOCOL_PREFIX
                    <= value < LAUNCHER_PROTOCOL_PREFIX + _PROTOCOL_SPAN):
                continue
            rem = value - LAUNCHER_PROTOCOL_PREFIX
            bot_id_mod = rem // 10000
            tx = (rem % 10000) // 100
            ty = rem % 100
            self.launcher_pending_requests[bot_id_mod] = (tx, ty, mxy)
            self.launcher_read_markers.add(mxy)
            if DEBUG:
                print(f"[L] pending req bid_mod={bot_id_mod} tgt=({tx},{ty}) "
                      f"marker={mxy}", file=sys.stderr)

        # --- Step 2: promote pending -> active when marker is gone ---
        for bot_id_mod in list(self.launcher_pending_requests.keys()):
            tx, ty, mxy = self.launcher_pending_requests[bot_id_mod]
            mbid = ct.get_tile_building_id(Position(mxy[0], mxy[1]))
            still_allied_marker = (mbid is not None
                                   and ct.get_entity_type(mbid) == EntityType.MARKER
                                   and ct.get_team(mbid) == my_team)
            if not still_allied_marker:
                self.launcher_active_requests[bot_id_mod] = (tx, ty)
                del self.launcher_pending_requests[bot_id_mod]
                if DEBUG:
                    print(f"[L] activate req bid_mod={bot_id_mod} tgt=({tx},{ty})",
                          file=sys.stderr)

        # Launcher actions need cooldown == 0 (LAUNCHER_FIRE_COOLDOWN = 1).
        if ct.get_action_cooldown() != 0:
            return

        # --- Step 3: fulfil an active request ---
        for uid in nearby_units:
            if ct.get_team(uid) != my_team:
                continue
            if ct.get_entity_type(uid) != EntityType.BUILDER_BOT:
                continue
            bp = ct.get_position(uid)
            if abs(bp.x - my_xy[0]) > 1 or abs(bp.y - my_xy[1]) > 1:
                continue  # pickup radius² = 2: cardinal OR diagonal adjacent
            bot_id_mod = uid % 10000
            if bot_id_mod not in self.launcher_active_requests:
                continue
            tx, ty = self.launcher_active_requests[bot_id_mod]
            target_pos = Position(tx, ty)
            if ct.can_launch(bp, target_pos):
                ct.launch(bp, target_pos)
                del self.launcher_active_requests[bot_id_mod]
                if DEBUG:
                    print(f"[L] launch ally {bot_id_mod} -> ({tx},{ty})",
                          file=sys.stderr)
                return
            alt = self._launcher_alt_target(ct, (tx, ty))
            if alt is not None:
                alt_pos = Position(alt[0], alt[1])
                if ct.can_launch(bp, alt_pos):
                    ct.launch(bp, alt_pos)
                    del self.launcher_active_requests[bot_id_mod]
                    if DEBUG:
                        print(f"[L] launch ally {bot_id_mod} -> alt ({alt[0]},{alt[1]})",
                              file=sys.stderr)
                    return

        # --- Step 4: launch adjacent enemy builder bots away ---
        for uid in nearby_units:
            if ct.get_team(uid) == my_team:
                continue
            if ct.get_entity_type(uid) != EntityType.BUILDER_BOT:
                continue
            bp = ct.get_position(uid)
            if abs(bp.x - my_xy[0]) > 1 or abs(bp.y - my_xy[1]) > 1:
                continue
            tgt = self._launcher_pick_enemy_target(ct, my_xy, (bp.x, bp.y))
            if tgt is None:
                continue
            target_pos = Position(tgt[0], tgt[1])
            if ct.can_launch(bp, target_pos):
                ct.launch(bp, target_pos)
                if DEBUG:
                    print(f"[L] launch enemy {(bp.x, bp.y)} -> {tgt}",
                          file=sys.stderr)
                return

    # ------------------------------------------------------------------ #
    #  Target selection helpers                                           #
    # ------------------------------------------------------------------ #

    def _launcher_alt_target(self, ct, desired):
        """Nearest bot-passable tile to `desired` within launcher throw range
        that has no bot currently on it."""
        max_r2 = GC.LAUNCHER_VISION_RADIUS_SQ
        best = None
        best_d2 = None
        for t in ct.get_nearby_tiles(max_r2):
            if ct.get_tile_builder_bot_id(t) is not None:
                continue
            if not ct.is_tile_passable(t):
                continue
            d2 = (t.x - desired[0]) ** 2 + (t.y - desired[1]) ** 2
            if best_d2 is None or d2 < best_d2:
                best = (t.x, t.y)
                best_d2 = d2
        return best

    def _launcher_pick_enemy_target(self, ct, my_xy, bot_xy):
        """Pick where to throw an enemy bot. Goal is MAX DISPLACEMENT
        — the enemy should land as far as possible from where it was
        picked up so it has to walk many turns back to reach us.

        Distance is measured from the BOT'S current position (not from
        the launcher or the cores) so the heuristic works regardless
        of whether the launcher sits in our base or right next to the
        enemy core. The previous "closest tile to enemy core" rule
        broke when the launcher was already adjacent to the enemy
        core: it threw the bot 1 tile and the bot walked straight
        back next turn.

        Tie-break: among equally-far tiles, prefer the one closest to
        the enemy core (or, if unknown, farthest from our core) so we
        push the enemy in a strategically useful direction when there
        are multiple max-displacement options.
        Candidate tiles must be bot-passable and not currently occupied.
        """
        max_r2 = GC.LAUNCHER_VISION_RADIUS_SQ
        if self.launcher_enemy_core_xy is not None:
            tiebreak_anchor = self.launcher_enemy_core_xy
            tiebreak_minimize = True
        elif self.launcher_our_core_xy is not None:
            tiebreak_anchor = self.launcher_our_core_xy
            tiebreak_minimize = False
        else:
            tiebreak_anchor = my_xy
            tiebreak_minimize = False

        best = None
        best_displacement = -1
        best_tiebreak = None
        for t in ct.get_nearby_tiles(max_r2):
            if ct.get_tile_builder_bot_id(t) is not None:
                continue
            if not ct.is_tile_passable(t):
                continue
            disp2 = (t.x - bot_xy[0]) ** 2 + (t.y - bot_xy[1]) ** 2
            tb = (t.x - tiebreak_anchor[0]) ** 2 + (t.y - tiebreak_anchor[1]) ** 2
            tb_score = tb if tiebreak_minimize else -tb
            if (disp2 > best_displacement
                    or (disp2 == best_displacement
                        and (best_tiebreak is None
                             or tb_score < best_tiebreak))):
                best = (t.x, t.y)
                best_displacement = disp2
                best_tiebreak = tb_score
        return best
