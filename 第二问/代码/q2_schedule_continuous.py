"""Solve Q2(2) by branching on a repeated duty and its continuous station boxes."""
import argparse
from dataclasses import dataclass, replace
import heapq
from itertools import count
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize, nnls

from q2_schedule import (
    EDGE, OBSERVE, OMEGA, OUTPUT, PAD, RADIUS, RECTANGLES, SPEED, STARTS,
    TARGETS, Roads, angles, export, objective, pair_compatible, replay, road,
    shadow_polygons, turn_time, visible,
)
from q2_schedule_bound import (
    around, box_distance, first_ready_lower, make_turn_bound, point_distance,
    split, width,
)
from q2_schedule_verify import check, read_rows


FIELD = (RADIUS, EDGE, RADIUS, EDGE)
CARS = tuple(range(len(STARTS)))
TASKS = tuple(range(len(TARGETS)))
SHADOW_MARGIN = 1e-7  # m in a normalized shadow-face inequality, used for feasible trials.
BOUND_MARGIN = 2e-9  # s, subtracted for floating-point calculations.


def center(box):
    a, b, c, d = box
    return np.array([(a + b) / 2, (c + d) / 2])


def bounds(box):
    a, b, c, d = box
    return [(a, b), (c, d)]


def direct_ready(origin, site, task):
    return math.dist(origin, site) / SPEED + float(turn_time(site, task))


def checked_rows(legs):
    rows = [(leg.car, leg.task, leg.route, np.array([
        leg.depart, leg.arrive, leg.ready, leg.start, leg.end, leg.reset,
    ])) for leg in legs]
    return check(rows, require_all=False)


def box_support(box, vector, *, maximum=False):
    """Minimize or maximize a linear function over the station rectangle."""
    x, y = vector
    a, b, c, d = box
    if maximum:
        return x * (b if x >= 0 else a) + y * (d if y >= 0 else c)
    return x * (a if x >= 0 else b) + y * (c if y >= 0 else d)


class SlewTangent:
    """Taylor minorant of one active slew axis, which also bounds the max."""
    def __init__(self, site, task):
        self.site = site
        dx, dy, h = TARGETS[task] - np.r_[site, 2]
        self.target = TARGETS[task, :2]
        r2 = dx**2 + dy**2
        r = math.sqrt(r2)
        yaw, pitch = angles(site, task)
        self.yaw = abs(yaw) > pitch
        if self.yaw:
            self.gradient = np.array([abs(dy), -math.copysign(1, dy) * dx]) / (OMEGA * r2)
        else:
            self.gradient = h * np.array([dx, dy]) / (OMEGA * r * (r2 + h**2))

    def remainder(self, box):
        a, b, c, d = box
        x, y = self.site
        union = (min(a, x), max(b, x), min(c, y), max(d, y))
        target_y = self.target[1]
        if self.yaw and union[2] <= target_y <= union[3]:
            return math.inf  # The absolute azimuth can change branch.
        r_min = point_distance(self.target, union)
        if not r_min:
            return math.inf
        r_far = math.hypot(max(abs(x - a), abs(x - b)), max(abs(y - c), abs(y - d)))
        return 0.5 * (r_far / r_min)**2 / OMEGA


class ChainMinorant:
    """Use tangents of travel norms and conservative slew-angle curvature."""
    def __init__(self, chain, helper, p, q, shadows):
        car, first, second = chain
        self.p = SlewTangent(p, first)
        self.q = SlewTangent(q, second)
        initial = p - STARTS[car]
        between = p - q
        drive_p = initial / (SPEED * np.linalg.norm(initial)) if np.any(initial) else initial
        drive_q = between / (SPEED * np.linalg.norm(between)) if np.any(between) else between
        slope_first = drive_p + self.p.gradient
        slope_tail = drive_q + self.p.gradient
        norm2 = float(slope_first @ slope_first)
        self.weight = float(np.clip(-(slope_first @ slope_tail) / norm2, 0, 1)) if norm2 else 0.0
        self.grad_p = slope_tail + self.weight * slope_first
        self.grad_q = self.q.gradient - drive_q
        ready_gap = direct_ready(STARTS[car], p, first) - helper
        self.value = (helper + OBSERVE + float(turn_time(p, first))
                      + direct_ready(p, q, second) + self.weight * ready_gap)

        # Shadow margins near the trial identify potential supporting faces.
        # They enter the bound only where a box implies that face is necessary.
        candidates = []
        for faces in shadows[second]:
            signs = faces[:, :2] @ q + faces[:, 2]
            index = int(signs.argmax())
            if abs(signs[index]) < 0.01:
                candidates.append((faces, index))
        self.supports = []
        if candidates:
            normals = np.column_stack([faces[index, :2] for faces, index in candidates])
            multipliers, _ = nnls(normals, self.grad_q)
            self.supports = [(faces, index, weight) for (faces, index), weight
                             in zip(candidates, multipliers) if weight > 0]

    def lower(self, p, q):
        ep, eq = self.p.remainder(p), self.q.remainder(q)
        if not math.isfinite(ep + eq):
            return -math.inf
        grad_q, constant = self.grad_q.copy(), self.value
        for faces, index, multiplier in self.supports:
            if all(box_support(q, row[:2], maximum=True) + row[2] < -1e-10
                   for i, row in enumerate(faces) if i != index):
                normal, offset = faces[index, :2], faces[index, 2]
                constant -= multiplier * (normal @ self.q.site + offset)
                grad_q -= multiplier * normal
        return (constant + box_support(p, self.grad_p) - self.grad_p @ self.p.site
                + box_support(q, grad_q) - grad_q @ self.q.site
                - (1 + self.weight) * ep - eq)


@dataclass(frozen=True)
class Node:
    cars: tuple = CARS
    first: tuple = TASKS
    second: tuple = TASKS
    p: tuple = FIELD
    q: tuple = FIELD

    def branch(self):
        for name in ("cars", "first", "second"):
            domain = getattr(self, name)
            if len(domain) > 1:
                middle = len(domain) // 2
                return [replace(self, **{name: part}) for part in
                        (domain[:middle], domain[middle:])]
        name = "p" if width(self.p) >= width(self.q) else "q"
        return [replace(self, **{name: half}) for half in split(getattr(self, name))]

    def chain(self):
        if len(self.cars) == len(self.first) == len(self.second) == 1:
            return self.cars[0], self.first[0], self.second[0]
        return None


@dataclass
class Result:
    legs: list
    lower: float
    nodes: int
    status: str


class ContinuousSchedule:
    def __init__(self):
        self.roads = Roads()
        self.shadows = shadow_polygons()
        self.turn_bound = make_turn_bound()
        self.stations = {}
        self.ready = {}
        self.partner = {}
        self.task_floor = 0.0
        self.minorants = {}

    def site_clearance(self, site, task):
        x, y = site
        return [
            80 - angles(site, task)[1],
            *(np.max(faces[:, :2] @ site + faces[:, 2]) - SHADOW_MARGIN
              for faces in self.shadows[task]),
            *(max(left - x, x - right, bottom - y, y - top) - PAD
              for left, right, bottom, top in RECTANGLES),
        ]

    def route_ready(self, origin, site, task):
        if not visible(site, task):
            return math.inf
        route = self.roads.route(origin, site)
        if not np.all(road(route[:-1], route[1:])):
            return math.inf
        length = np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
        return length / SPEED + float(turn_time(site, task))

    def solo_station(self, origin, task):
        """Get a feasible continuous start from a small adaptive box search."""
        ready = self.route_ready(origin, origin, task)
        best = origin if math.isfinite(ready) else None
        queue = [(0.0, FIELD)]
        for _ in range(160):
            if not queue:
                break
            lower, box = heapq.heappop(queue)
            if lower >= ready:
                break
            site = center(box)
            if direct_ready(origin, site, task) < ready:
                arrival = self.route_ready(origin, site, task)
                if arrival < ready:
                    best, ready = site, arrival
            for half in split(box):
                lower = point_distance(origin, half) / SPEED + self.turn_bound(half, task)
                if lower < ready:
                    heapq.heappush(queue, (lower, half))
        if best is None:
            return None, math.inf
        trial = minimize(lambda site: direct_ready(origin, site, task), best,
                         bounds=bounds(FIELD), method="SLSQP",
                         constraints={"type": "ineq", "fun": self.site_clearance,
                                      "args": (task,)},
                         options={"ftol": 1e-11, "maxiter": 100})
        trial_ready = self.route_ready(origin, trial.x, task)
        return (trial.x, trial_ready) if trial_ready < ready else (best, ready)

    def solo_bound(self, origin, task, site, ready):
        if site is not None and np.array_equal(origin, site):
            # Any improvement is within SPEED * ready of origin. Bound the
            # gradients of both slew axes there, including the max and abs.
            x, y, h = TARGETS[task]
            nearest = math.dist(origin, (x, y)) - SPEED * ready
            if min(nearest, h - 2) > SPEED / OMEGA:
                return ready - BOUND_MARGIN
        low, high = 0.0, min(ready, 12.0)
        if first_ready_lower(origin, task, high, self.turn_bound):
            return high
        for _ in range(15):
            mid = (low + high) / 2
            if first_ready_lower(origin, task, mid, self.turn_bound):
                low = mid
            else:
                high = mid
        return low

    def prepare(self):
        solo = np.zeros((len(CARS), len(TASKS)))
        for car in CARS:
            for task in TASKS:
                site, ready = self.solo_station(STARTS[car], task)
                self.stations[car, task] = site
                self.ready[car, task] = ready
                solo[car, task] = self.solo_bound(STARTS[car], task, site, ready)
        self.task_floor = float(np.sort(solo, axis=0)[1].max())
        self.partner = {(car, task): min(solo[k, task] for k in CARS if k != car)
                        for car in CARS for task in TASKS}

    def lower(self, node):
        first, second = node.first, node.second
        if len(first) == len(second) == 1 and first == second:
            return math.inf
        if len(first) == 1:
            second = tuple(j for j in second if j != first[0])
        if len(second) == 1:
            first = tuple(i for i in first if i != second[0])
        p, q = node.p, node.q
        g = min(self.turn_bound(p, i) for i in first)
        h = min(self.turn_bound(q, j) for j in second)
        partner = min(self.partner[k, i] for k in node.cars for i in first)
        initial = min(point_distance(STARTS[k], p) for k in node.cars) / SPEED
        total_drive = min(point_distance(STARTS[k], q) for k in node.cars) / SPEED
        tail = box_distance(p, q) / SPEED + h
        union = (min(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), max(p[3], q[3]))
        if all(point_distance(TARGETS[j, :2], union) > SPEED / OMEGA + 1e-10
               for j in second):
            # Over this rectangle motion cannot reduce dwell plus slew time.
            tail = max(tail, min(self.turn_bound(p, j, False) for j in second))
        chain = OBSERVE + max(max(partner, initial + g) + g + tail,
                              total_drive + 2 * g + h)
        if max(width(p), width(q)) <= 3.1:
            local = self.minorants.get(node.chain())
            if local is not None:
                chain = max(chain, local.lower(p, q))
        return max(self.task_floor, chain) - BOUND_MARGIN

    def refined_sites(self, chain, p, q):
        car, first, second = chain
        origin = STARTS[car]
        helper = self.partner[car, first]

        def tail(a, b):
            return OBSERVE + float(turn_time(a, first)) + direct_ready(a, b, second)

        def constraints(raw):
            a, b = raw[:4].reshape(2, 2)
            start = raw[4]
            return [start - direct_ready(origin, a, first),
                    *self.site_clearance(a, first), *self.site_clearance(b, second)]

        trial = minimize(
            lambda raw: raw[4] + tail(*raw[:4].reshape(2, 2)),
            np.r_[p, q, max(helper, direct_ready(origin, p, first))],
            bounds=bounds(FIELD) * 2 + [(helper, None)],
            constraints={"type": "ineq", "fun": constraints}, method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 150},
        )
        return trial.x[:4].reshape(2, 2)

    def complete(self, chain, p, q):
        """Obtain an incumbent by bottleneck matching the other observation slots."""
        car, first, second = chain
        own = self.route_ready(STARTS[car], p, first)
        tail = OBSERVE + float(turn_time(p, first)) + self.route_ready(p, q, second)
        if not math.isfinite(own + tail):
            return None
        (third,) = set(TASKS) - {first, second}
        cars = [k for k in CARS if k != car]
        slots = (first, second, third, third)
        cost = np.array([[self.ready[k, j] for j in slots] for k in cars])
        for row, k in enumerate(cars):
            for col, station in enumerate((p, q)):
                helper = self.stations[k, slots[col]]
                if helper is None or not pair_compatible(station, helper, slots[col]):
                    cost[row, col] = math.inf
        cost[:, 0] += tail
        for cap in np.unique(cost[np.isfinite(cost)]):
            used, columns = linear_sum_assignment(cost > cap)
            if np.any(cost[used, columns] > cap):
                continue
            teams = {first: [(car, p)], second: [(car, q)], third: []}
            for row, col in zip(used, columns):
                k, j = cars[row], slots[col]
                teams[j].append((k, self.stations[k, j]))
            legs = replay(self.roads, [(j, teams[j]) for j in (first, second, third)])
            try:
                checked_rows(legs)
            except AssertionError:
                return None
            return legs
        return None

    def solve(self, gap, max_nodes):
        self.prepare()
        incumbent, upper, explored = [], math.inf, 0
        attempts = set()
        serial = count()
        root = Node()
        queue = [(self.lower(root), next(serial), root)]
        while queue and explored < max_nodes and queue[0][0] < upper - gap:
            lower, _, node = heapq.heappop(queue)
            explored += 1
            chain = node.chain()
            if chain is not None:
                # Try local continuous optimization once at each spatial scale.
                scale = int(math.floor(math.log2(max(width(node.p), width(node.q)))))
                key = chain, scale
                if key not in attempts:
                    attempts.add(key)
                    sites = self.refined_sites(chain, center(node.p), center(node.q))
                    trial = self.complete(chain, *sites)
                    if trial and objective(trial) < upper:
                        incumbent, upper = trial, objective(trial)
                        self.minorants[chain] = ChainMinorant(
                            chain, self.partner[chain[:2]], *sites, self.shadows
                        )
                        print(f"Incumbent {upper:.9f} s at node {explored}, "
                              f"V{chain[0] + 1}: T{chain[1] + 1} -> T{chain[2] + 1}",
                              flush=True)
            if lower >= upper - gap:
                heapq.heappush(queue, (lower, next(serial), node))
                break
            for child in node.branch():
                estimate = max(lower, self.lower(child))
                if estimate < upper:
                    heapq.heappush(queue, (estimate, next(serial), child))
        lower = min(queue[0][0], upper) if queue else upper
        status = "gap_met" if incumbent and upper - lower <= gap else "node_limit"
        return Result(incumbent, lower, explored, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=float, default=2e-7, help="absolute gap in seconds")
    parser.add_argument("--nodes", type=int, default=500_000, help="node limit")
    parser.add_argument("--output", type=Path,
                        default=OUTPUT / "第二问第（2）题连续分支定界时间轴.csv")
    args = parser.parse_args()
    if not math.isfinite(args.gap) or args.gap <= 0 or args.nodes < 1:
        parser.error("gap and node limit must be positive")
    reference = OUTPUT / "第二问第（2）题逐车时间轴.csv"
    if args.output.resolve() == reference.resolve():
        parser.error("use a separate path for the new solution")
    result = ContinuousSchedule().solve(args.gap, args.nodes)
    print(f"Status {result.status}; lower bound {result.lower:.12f} s; nodes {result.nodes}")
    if not result.legs:
        raise SystemExit("No verified incumbent found")
    upper = objective(result.legs)
    print(f"Upper bound {upper:.12f} s; gap {upper - result.lower:.12g} s")
    path = export({"连续分支定界": result.legs}, args.output)
    for name, legs in {**read_rows(path), **read_rows(reference)}.items():
        clearance = check(legs, require_all=False)
        print(f"{name}: {max(times[3] for *_, times in legs):.9f} s; "
              f"clearances {clearance[:3]}; angles {clearance[3]}")
    print(path)


if __name__ == "__main__":
    main()
