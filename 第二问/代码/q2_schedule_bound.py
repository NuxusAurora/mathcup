"""Continuous lower bound using rectangles and every possible first reuse."""
import csv
import math
import sys
from functools import lru_cache
from itertools import permutations

from q2_schedule import (
    EDGE, OBSERVE, OMEGA, OUTPUT, RADIUS, SPEED, STARTS, TARGETS, shadow_polygons,
)


def around(point, radius):
    x, y = point
    return (max(RADIUS, x - radius), min(EDGE, x + radius),
            max(RADIUS, y - radius), min(EDGE, y + radius))


def point_distance(point, box):
    x, y = point
    a, b, c, d = box
    return math.hypot(max(a - x, x - b, 0), max(c - y, y - d, 0))


def box_distance(p, q):
    return math.hypot(max(p[0] - q[1], q[0] - p[1], 0),
                      max(p[2] - q[3], q[2] - p[3], 0))


def width(box):
    return max(box[1] - box[0], box[3] - box[2])


def split(box):
    a, b, c, d = box
    if b - a >= d - c:
        middle = (a + b) / 2
        return (a, middle, c, d), (middle, b, c, d)
    middle = (c + d) / 2
    return (a, b, c, middle), (a, b, middle, d)


def make_turn_bound():
    shadows = [[polygon.tolist() for polygon in task] for task in shadow_polygons()]

    @lru_cache(maxsize=200_000)
    def bound(box, task, require_visibility=True):
        a, b, c, d = box
        x, y, h = TARGETS[task].tolist()
        far_x, far_y = max(abs(x - a), abs(x - b)), max(abs(y - c), abs(y - d))
        pitch = math.atan2(h - 2, math.hypot(far_x, far_y))
        if require_visibility:
            if pitch > math.radians(80) + 1e-12:
                return math.inf
            for polygon in shadows[task]:
                maximum = max(
                    nx * (b if nx >= 0 else a) + ny * (d if ny >= 0 else c) + bias
                    for nx, ny, bias in polygon
                )
                if maximum < -1e-10:
                    return math.inf
        far_east = x - a
        near_y = max(c - y, y - d, 0)
        yaw = math.atan2(near_y if far_east >= 0 else far_y, far_east)
        return max(yaw, pitch) / OMEGA

    return bound


def first_ready_lower(origin, task, target, turn_bound):
    """Certify no direct or indirect path can beat the specified time."""
    stack = [around(origin, SPEED * target)]
    while stack:
        box = stack.pop()
        if point_distance(origin, box) / SPEED + turn_bound(box, task) >= target + 1e-10:
            continue
        if width(box) < 1e-4:
            return False
        stack.extend(split(box))
    return True


def certify(target):
    turn_bound = make_turn_bound()
    ready_lower = {}
    for car, origin in enumerate(STARTS.tolist()):
        for task in range(3):
            low, high = 0.0, target - OBSERVE
            if first_ready_lower(origin, task, high, turn_bound):
                low = high
            else:
                while high - low > 0.0005:
                    middle = (low + high) / 2
                    if first_ready_lower(origin, task, middle, turn_bound):
                        low = middle
                    else:
                        high = middle
            ready_lower[car, task] = low

    domain = (RADIUS, EDGE, RADIUS, EDGE)
    count = 0
    for car, origin in enumerate(STARTS.tolist()):
        for first, second in permutations(range(3), 2):
            other = min(ready_lower[k, first] for k in range(5) if k != car)
            g_min = turn_bound(domain, first)
            h_min = turn_bound(domain, second)
            travel_budget = SPEED * (target - OBSERVE - 2 * g_min - h_min)
            if travel_budget <= 0:
                continue
            first_box = around(origin, travel_budget)
            stack = [(first_box, first_box)]
            target_xy = TARGETS[second, :2].tolist()
            while stack:
                p, q = stack.pop()
                count += 1
                g, h = turn_bound(p, first), turn_bound(q, second)
                first_travel = point_distance(origin, p) / SPEED
                second_travel = box_distance(p, q) / SPEED
                total_travel = point_distance(origin, q) / SPEED
                synced = max(other, first_travel + g)
                second_leg = second_travel + h
                union = (min(p[0], q[0]), max(p[1], q[1]),
                         min(p[2], q[2]), max(p[3], q[3]))
                if point_distance(target_xy, union) >= SPEED / OMEGA + 1e-10:
                    # Across this convex hull the turn-time slope is <= 1/SPEED.
                    second_leg = max(second_leg, turn_bound(p, second, False))
                lower = OBSERVE + max(synced + g + second_leg, total_travel + 2*g + h)
                if lower >= target + 1e-10:
                    continue
                if max(width(p), width(q)) < 1e-4:
                    print(f"Unresolved V{car + 1}: T{first + 1} -> T{second + 1}", p, q)
                    return False
                if width(p) >= width(q):
                    stack.extend((part, q) for part in split(p))
                else:
                    stack.extend((p, part) for part in split(q))
    print(f"Continuous lower bound {target:.6f} s, {count} box pairs.")
    return True


if __name__ == "__main__":
    with (OUTPUT / "第二问第（2）题逐车时间轴.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    upper = max(float(row["观测开始_s"]) for row in rows if row["方案"] == "优化")
    gap = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
    threshold = math.floor((upper - gap) * 1_000_000) / 1_000_000
    if not certify(threshold):
        raise SystemExit("Threshold not certified")
