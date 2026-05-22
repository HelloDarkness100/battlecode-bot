import math
from cambc import Direction, Position

# Pre-computed direction deltas — avoids calling .delta() in tight loops
DIRECTION_DELTAS = {
    Direction.NORTH:     (0, -1),
    Direction.NORTHEAST: (1, -1),
    Direction.EAST:      (1,  0),
    Direction.SOUTHEAST: (1,  1),
    Direction.SOUTH:     (0,  1),
    Direction.SOUTHWEST: (-1, 1),
    Direction.WEST:      (-1, 0),
    Direction.NORTHWEST: (-1, -1),
    Direction.CENTRE:    (0,  0),
}

# Reverse lookup: (dx, dy) → Direction
_DELTA_TO_DIR = {v: k for k, v in DIRECTION_DELTAS.items() if k != Direction.CENTRE}

CARDINAL_DIRS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]


def euclidean_dist_sq(a, b):
    """Squared Euclidean distance between two (x,y) tuples."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def euclidean_dist(a, b):
    """Euclidean distance between two (x,y) tuples."""
    return math.sqrt(euclidean_dist_sq(a, b))


def direction_between(from_xy, to_xy):
    """Return the closest 8-directional Direction from from_xy toward to_xy."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if dx == 0 and dy == 0:
        return Direction.CENTRE
    # Normalize to -1/0/1
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    return _DELTA_TO_DIR.get((sx, sy), Direction.CENTRE)


def cardinal_direction_between(from_xy, to_xy):
    """Return the closest cardinal Direction from from_xy toward to_xy."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) >= abs(dy):
        return Direction.EAST if dx >= 0 else Direction.WEST
    return Direction.SOUTH if dy >= 0 else Direction.NORTH


def neighbors_8(x, y):
    """Yield all 8 neighbor (x,y) tuples."""
    yield (x,     y - 1)
    yield (x + 1, y - 1)
    yield (x + 1, y)
    yield (x + 1, y + 1)
    yield (x,     y + 1)
    yield (x - 1, y + 1)
    yield (x - 1, y)
    yield (x - 1, y - 1)


def neighbors_4(x, y):
    """Yield 4 cardinal neighbor (x,y) tuples."""
    yield (x, y - 1)
    yield (x + 1, y)
    yield (x, y + 1)
    yield (x - 1, y)


def encode_position(x, y):
    """Pack (x, y) into a u32 marker value. Supports coords 0-255."""
    return (x & 0xFF) | ((y & 0xFF) << 8)


def decode_position(val):
    """Unpack a u32 marker value into (x, y)."""
    return (val & 0xFF, (val >> 8) & 0xFF)


def detect_symmetry(core_xy, map_w, map_h):
    """Detect map symmetry type from core position. Returns 'diag', 'vert', or 'horiz'."""
    cx, cy = core_xy
    mid_x = (map_w - 1) / 2.0
    mid_y = (map_h - 1) / 2.0
    centered_x = abs(cx - mid_x) < 1.0
    centered_y = abs(cy - mid_y) < 1.0
    if centered_x and not centered_y:
        return 'vert'
    if centered_y and not centered_x:
        return 'horiz'
    return 'diag'


def mirror_position(core_xy, map_w, map_h):
    """Estimate enemy core position by mirroring our core."""
    sym = detect_symmetry(core_xy, map_w, map_h)
    return mirror_xy(core_xy, sym, map_w, map_h)


def mirror_xy(xy, symmetry, map_w, map_h):
    """Mirror an (x,y) tile based on symmetry type. Pure tuple arithmetic."""
    x, y = xy
    if symmetry == 'vert':
        return (x, map_h - 1 - y)
    if symmetry == 'horiz':
        return (map_w - 1 - x, y)
    return (map_w - 1 - x, map_h - 1 - y)


def xy_to_pos(xy):
    """Convert (x,y) tuple to Position object. Use sparingly — only at API boundary."""
    return Position(xy[0], xy[1])
