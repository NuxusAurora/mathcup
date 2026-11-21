"""Solve the three-alert schedule and export both event timelines."""
import csv
import heapq
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "综合" / "代码"))
from check_bounds import RECTANGLES, RADIUS


STARTS = np.array([(15, 14), (88, 10), (14, 86), (88, 68), (50, 93)], dtype=float)
ALERTS = [(30, 40, 45), (150, 45, 35), (285, 38, 55)]
TARGETS = np.array([
    (50 + r * math.cos(math.radians(a)), 50 + r * math.sin(math.radians(a)), h)
    for a, r, h in ALERTS
])
BUILDINGS = np.column_stack((RECTANGLES, [20, 28, 12, 24, 32]))
SPEED, TURN, OBSERVE = 3.0, 30.0, 5.0
EDGE = 100 - RADIUS
STEP = 0.5
PAD = RADIUS + 0.001
OMEGA = math.radians(TURN)
OUTPUT = ROOT / "第二问" / "文档"


def shadow_polygons():
    shadows = []
    for x, y, h in TARGETS:
        target = np.array([x, y])
        polygons = []
        for left, right, bottom, top, roof in BUILDINGS:
            corners = np.array([(a, b) for a in (left, right) for b in (bottom, top)])
            scale = (h - 2) / (h - roof)
            far = target + scale * (corners - target)
            polygons.append(ConvexHull(np.vstack((corners, far))).equations)
        shadows.append(polygons)
    return shadows


def hit(start, end, rectangle, last=1.0):
    """Closed segment/slab intersection, restricted to the parameter [0, last]."""
    start, end = np.asarray(start), np.asarray(end)
    shape = np.broadcast_shapes(start.shape[:-1], end.shape[:-1])
    low, high = np.zeros(shape), np.full(shape, last)
    for axis, (left, right) in enumerate(np.reshape(rectangle, (2, 2))):
        origin = start[..., axis]
        direction = end[..., axis] - origin
        moving = direction != 0
        divisor = np.where(moving, direction, 1)
        a, b = (left - origin) / divisor, (right - origin) / divisor
        low = np.maximum(low, np.where(moving, np.minimum(a, b), -np.inf))
        high = np.minimum(high, np.where(moving, np.maximum(a, b), np.inf))
        high = np.where(~moving & ((origin < left) | (origin > right)), -np.inf, high)
    return low <= high


def road(start, end):
    clear = True
    for left, right, bottom, top in RECTANGLES:
        # Open rectangle slightly larger than the exact rounded obstacle.
        pad = PAD - 1e-7
        clear = clear & ~hit(start, end, (left-pad, right+pad, bottom-pad, top+pad))
    return clear


def angles(points, task):
    x, y, h = TARGETS[task]
    dx, dy = x - points[..., 0], y - points[..., 1]
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(h - 2, np.hypot(dx, dy)))
    return yaw, pitch


def turn_time(points, task):
    yaw, pitch = angles(points, task)
    return np.maximum(np.abs(yaw), pitch) / TURN


def visible(points, task):
    target = TARGETS[task]
    clear = angles(points, task)[1] <= 80
    for *rectangle, roof in BUILDINGS:
        clear &= ~hit(points, target[:2], rectangle, (roof - 2) / (target[2] - 2))
    return clear


class Roads:
    """Finite waypoint graph used by both strategies to price and recover paths."""

    def __init__(self):
        self.corners = np.array([
            (x, y) for left, right, bottom, top in RECTANGLES
            for x in (left - PAD, right + PAD) for y in (bottom - PAD, top + PAD)
        ])
        self.edges = [
            [(j, float(np.linalg.norm(p - q))) for j, q in enumerate(self.corners)
             if i != j and road(p, q)]
            for i, p in enumerate(self.corners)
        ]

    def paths(self, origin, points):
        direct = road(origin, points)
        best = np.where(direct, np.linalg.norm(points - origin, axis=1), np.inf)
        via = np.full(len(points), -1, dtype=int)
        dist = np.where(road(origin, self.corners),
                        np.linalg.norm(self.corners - origin, axis=1), np.inf)
        previous = [-1] * len(dist)
        heap = [(d, i) for i, d in enumerate(dist) if np.isfinite(d)]
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d != dist[i]:
                continue
            for j, length in self.edges[i]:
                if d + length < dist[j]:
                    dist[j], previous[j] = d + length, i
                    heapq.heappush(heap, (dist[j], j))
        for i, corner in enumerate(self.corners):
            trial = dist[i] + np.linalg.norm(points - corner, axis=1)
            better = (trial < best - 1e-10) & road(corner, points)
            best[better], via[better] = trial[better], i
        return best, via, previous

    def route(self, origin, station):
        _, via, previous = self.paths(origin, np.array([station]))
        interior = []
        i = via[0]
        while i >= 0:
            interior.append(self.corners[i])
            i = previous[i]
        return np.array([origin, *reversed(interior), station])


def stations(roads):
    axis = np.arange(RADIUS, EDGE + 1e-8, STEP)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    points = np.unique(np.vstack((grid, STARTS, roads.corners)), axis=0)
    points = points[road(points, points)]
    return [points[visible(points, t)] for t in range(3)]


def pair_compatible(a, b, task):
    u, v = a - TARGETS[task, :2], b - TARGETS[task, :2]
    return np.linalg.norm(a - b, axis=-1) >= 2 * RADIUS and (
        abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]) > 1e-7
    )


def nearest_assignment(roads, candidates):
    """Greedy alert order, then distance to alert and route length to station."""
    current = STARTS.copy()
    plan = []
    for task, points in enumerate(candidates):
        target = TARGETS[task, :2]
        cars = sorted(range(5), key=lambda k: (np.linalg.norm(current[k] - target), k))[:2]
        selected = []
        for k in cars:
            length, _, _ = roads.paths(current[k], points)
            for index in np.argsort(length, kind="stable"):
                station = points[index]
                if not selected or pair_compatible(selected[0][1], station, task):
                    break
            assert np.isfinite(length[index])
            selected.append((k, station))
            current[k] = station
        plan.append((task, selected))
    return plan


@dataclass
class Leg:
    car: int
    task: int
    route: np.ndarray
    depart: float
    arrive: float
    ready: float
    start: float
    end: float
    reset: float


def replay(roads, plan):
    position, available = STARTS.copy(), np.zeros(5)
    result = []
    for task, assignment in plan:
        pending = []
        for k, station in assignment:
            route = roads.route(position[k], station)
            length = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
            turn = turn_time(station, task)
            arrive = available[k] + length / SPEED
            pending.append((k, route, available[k], arrive, turn))
        start = max(arrive + turn for _, _, _, arrive, turn in pending)
        for k, route, depart, arrive, turn in pending:
            reset = start + OBSERVE + turn
            result.append(Leg(k, task, route, depart, arrive, arrive + turn,
                              start, start + OBSERVE, reset))
            position[k], available[k] = route[-1], reset
    return result


def objective(legs):
    return max(leg.start for leg in legs)


def joint_search(roads, candidates, initial_limit):
    ready, best_station, outbound = {}, {}, {}
    for task, points in enumerate(candidates):
        for car, origin in enumerate(STARTS):
            length, _, _ = roads.paths(origin, points)
            time = length / SPEED + turn_time(points, task)
            index = np.argmin(time)
            ready[car, task] = time[index]
            best_station[car, task] = points[index]
            outbound[car, task] = time

    # Any solution better than this incumbent must reach both consecutive
    # sites within SPEED * (limit - OBSERVE) metres of the reused car's origin.
    start_sites = [np.where(np.any(np.all(points[:, None] == STARTS, axis=2), axis=1))[0]
                   for points in candidates]
    limit, winner = initial_limit, None
    for pass_index in range(2):
        for car, origin in enumerate(STARTS):
            others = [k for k in range(5) if k != car]
            for first, second, third in permutations(range(3)):
                limit_distance = SPEED * (limit - OBSERVE)
                p_set = candidates[first]
                q_set = candidates[second]
                p_ids = np.flatnonzero(outbound[car, first] + OBSERVE < limit)
                q_ids = np.flatnonzero(np.linalg.norm(q_set - origin, axis=1) <= limit_distance)
                if pass_index == 0:
                    p_ids = np.intersect1d(p_ids, start_sites[first])
                    q_ids = np.intersect1d(q_ids, start_sites[second])
                if not len(p_ids) or not len(q_ids):
                    continue
                q = q_set[q_ids]
                q_turn = turn_time(q, second)
                p = p_set[p_ids]
                first_ready = outbound[car, first][p_ids]
                p_turn = turn_time(p, first)
                # Direct legs attain their Euclidean lower bound. Detours
                # are unnecessary for the incumbent established below.
                between = np.linalg.norm(p[:, None] - q, axis=2) / SPEED + q_turn
                clear = road(p[:, None], q) & (
                    first_ready + OBSERVE + p_turn < limit
                )[:, None]
                between = np.where(clear, between, np.inf)
                for helper, partner in permutations(others, 2):
                    remaining = [k for k in others if k not in (helper, partner)]
                    third_ready = max(ready[k, third] for k in remaining)
                    if max(third_ready, ready[helper, first] + OBSERVE,
                           ready[partner, second]) >= limit:
                        continue
                    hp, sp = best_station[helper, first], best_station[partner, second]
                    p_ok = np.array([pair_compatible(point, hp, first) for point in p])
                    q_ok = np.array([pair_compatible(point, sp, second) for point in q])
                    if not pair_compatible(*(best_station[k, third] for k in remaining), third):
                        continue
                    first_start = np.maximum(first_ready, ready[helper, first])
                    times = first_start[:, None] + OBSERVE + p_turn[:, None] + between
                    times = np.maximum(times, max(third_ready, ready[partner, second]))
                    times = np.where(p_ok[:, None] & q_ok, times, np.inf)
                    i, j = np.unravel_index(np.argmin(times), times.shape)
                    if times[i, j] < limit:
                        limit = times[i, j]
                        winner = [(first, [(car, p[i]), (helper, hp)]),
                                  (second, [(car, q[j]), (partner, sp)]),
                                  (third, [(k, best_station[k, third]) for k in remaining])]
    assert winner is not None
    return winner


def refine(roads, plan):
    """Refine both reused-car stations while holding the other four fixed."""
    (first, [(car, p), (helper, hp)]), (second, [(_, q), (_, sp)]), _ = plan
    ready = next(leg.ready for leg in replay(roads, plan) if leg.car == helper)
    origin = STARTS[car]
    shadows = shadow_polygons()

    def own_ready(first_site):
        return np.linalg.norm(first_site - origin) / SPEED + turn_time(first_site, first)

    def tail(first_site, second_site):
        drive = np.linalg.norm(second_site - first_site) / SPEED
        return OBSERVE + turn_time(first_site, first) + drive + turn_time(second_site, second)

    def projected_finish(first_site, second_site):
        return max(ready, own_ready(first_site)) + tail(first_site, second_site)

    def site_constraints(raw):
        first_site, second_site = raw[:4].reshape(2, 2)
        start = raw[4]
        values = [start - ready, start - own_ready(first_site)]
        for station, task in ((first_site, first), (second_site, second)):
            for polygon in shadows[task]:
                values.append(np.max(polygon[:, :2] @ station + polygon[:, 2]) - 1e-7)
            values.append(80 - angles(station, task)[1])
        return values

    trial = minimize(lambda raw: raw[4] + tail(*raw[:4].reshape(2, 2)),
                     np.r_[p, q, max(ready, own_ready(p))],
                     bounds=[(RADIUS, EDGE)] * 4 + [(0, None)], method="SLSQP",
                     constraints={"type": "ineq", "fun": site_constraints},
                     options={"ftol": 1e-12, "maxiter": 500})
    a, b = trial.x[:4].round(11).reshape(2, 2)
    if (road(origin, a) and road(a, b) and visible(a, first) and visible(b, second)
            and pair_compatible(a, hp, first) and pair_compatible(b, sp, second)
            and projected_finish(a, b) < projected_finish(p, q)):
        plan[0][1][0] = (car, a)
        plan[1][1][0] = (car, b)
    return plan


def export(solutions):
    rows = []
    for name, legs in solutions.items():
        for leg in sorted(legs, key=lambda leg: (leg.car, leg.depart)):
            station = leg.route[-1]
            yaw, pitch = angles(station, leg.task)
            rows.append({
                "方案": name, "车辆": f"V{leg.car + 1}", "告警": f"T{leg.task + 1}",
                "路径坐标_m": json.dumps(leg.route.tolist(), separators=(",", ":")),
                "行驶_m": f"{(leg.arrive - leg.depart) * SPEED:.9f}",
                "方位角_deg": f"{yaw:.9f}", "俯仰角_deg": f"{pitch:.9f}",
                "出发_s": f"{leg.depart:.9f}", "到站_转动开始_s": f"{leg.arrive:.9f}",
                "指向完成_s": f"{leg.ready:.9f}", "观测开始_s": f"{leg.start:.9f}",
                "观测结束_复位开始_s": f"{leg.end:.9f}", "复位完成_s": f"{leg.reset:.9f}",
            })
    with (OUTPUT / "第二问第（2）题逐车时间轴.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    roads = Roads()
    candidates = stations(roads)
    baseline = replay(roads, nearest_assignment(roads, candidates))
    plan = joint_search(roads, candidates, objective(baseline))
    optimized = replay(roads, refine(roads, plan))
    solutions = {"优化": optimized, "最近指派": baseline}
    export(solutions)
    for name, legs in solutions.items():
        print(name, f"{objective(legs):.9f} s")


if __name__ == "__main__":
    main()
