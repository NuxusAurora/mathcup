"""Continuous station search from the Q4 release state."""
import math
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

from q4_timeline import (
    DEADLINE, Duty, observation_starts, snapshot, tracks_for, trajectory_clear,
)
from q2_schedule import BUILDINGS, EDGE, OBSERVE, RADIUS, SPEED, TURN, Roads

CLEARANCE = 1e-7
SIGHT_MARGIN = 1.1e-8
BUILDING_GEOMETRY = tuple(tuple(map(float, rect)) for rect in BUILDINGS)


def intersects(start, end, rectangle, last=1.0):
    first = 0.0
    for a, b, lo, hi in zip(start, end, rectangle[::2], rectangle[1::2]):
        d = b - a
        if d == 0:
            if not lo <= a <= hi:
                return False
            continue
        u, v = sorted(((lo - a) / d, (hi - a) / d))
        first, last = max(first, u), min(last, v)
        if first > last:
            return False
    return True


def road_segment(a, b):
    pad = RADIUS + CLEARANCE
    return not any(intersects(a, b, (l - pad, r + pad, d - pad, u + pad))
                   for l, r, d, u, _ in BUILDING_GEOMETRY)


@lru_cache(maxsize=16384)
def station_aim(point, target):
    if not all(RADIUS <= coordinate <= EDGE for coordinate in point):
        return None
    x, y, h = target
    dx, dy = x - point[0], y - point[1]
    pitch = math.degrees(math.atan2(h - 2, math.hypot(dx, dy)))
    if pitch > 80 - CLEARANCE:
        return None
    if any(intersects(point, target[:2], (l - SIGHT_MARGIN, r + SIGHT_MARGIN,
                                          d - SIGHT_MARGIN, u + SIGHT_MARGIN),
                      (roof - 2 + SIGHT_MARGIN) / (h - 2))
           for l, r, d, u, roof in BUILDING_GEOMETRY):
        return None
    return math.degrees(math.atan2(dy, dx)), pitch


class StationSearch:
    def __init__(self):
        self.roads = Roads()
        self.route = lru_cache(maxsize=8192)(self._route)

    def _route(self, origin, station):
        if road_segment(origin, station):
            return origin, station
        path = tuple(map(tuple, self.roads.route(np.array(origin), np.array(station))))
        if all(road_segment(a, b) for a, b in zip(path, path[1:])):
            return path

    def replay(self, layout, points, states, targets, interiors=None):
        """Layout orders joint tasks and identifies their two vehicles."""
        cars = states.copy()
        duties = []
        for i in range(0, len(points), 2):
            task, first, second = layout[i // 2]
            target = targets[task]
            a, b = points[i:i + 2]
            x, y = a[0] - target[0], a[1] - target[1]
            u, v = b[0] - target[0], b[1] - target[1]
            if math.dist(a, b) < 2 * RADIUS or abs(x * v - y * u) <= 1e-7:
                return None
            pair = []
            for j, (car, point) in enumerate(zip((first, second), (a, b)), i):
                origin, depart = cars[car]
                aim = station_aim(point, target)
                if aim is None:
                    return None
                if interiors is None:
                    path = self.route(origin, point)
                    if path is None:
                        return None
                else:
                    path = (origin, *interiors[j], point)
                    if not all(road_segment(p, q) for p, q in zip(path, path[1:])):
                        return None
                arrive = depart + sum(math.dist(p, q) for p, q in zip(path, path[1:])) / SPEED
                turn = max(map(abs, aim)) / TURN
                pair.append((car, path, aim, depart, arrive, turn))
            start = max(arrive + turn for _, _, _, _, arrive, turn in pair)
            if start > DEADLINE:
                return None
            for car, path, aim, depart, arrive, turn in pair:
                finish = start + OBSERVE
                reset = finish + turn
                duties.append(Duty(car, task, path, aim, depart, arrive,
                                   arrive + turn, start, finish, reset))
                cars[car] = (path[-1], reset)
        return tuple(duties)

    def refine(self, layout, duties, past, states, locked, targets, fault):
        """Refine real-valued positions by bounded line minimization."""
        points = [duty.path[-1] for duty in duties]
        interiors = tuple(duty.path[1:-1] for duty in duties)
        locked_last = max(locked.values(), default=0.0)

        def cost(candidate):
            starts = [duty.start for duty in candidate[::2]]
            return max(locked_last, *starts) + 1e-5 * sum(starts) + 1e-6 * sum(
                duty.ready for duty in candidate
            )

        value = cost(duties)
        for sweep in range(2):
            for index, duty in enumerate(duties):
                x, y = points[index]
                origin = duty.path[-2]
                dx, dy = origin[0] - x, origin[1] - y
                norm = math.hypot(dx, dy)
                directions = [(1, 0), (0, 1)]
                if norm > CLEARANCE:
                    directions.append((dx / norm, dy / norm))
                for vx, vy in directions:
                    x, y = points[index]

                    def evaluate(step):
                        trial = list(points)
                        trial[index] = (x + vx * step, y + vy * step)
                        plan = self.replay(layout, trial, states, targets, interiors)
                        return trial, plan, cost(plan) if plan else 1e6

                    radius = 5.0 if sweep == 0 else 2.0
                    best = value
                    for sign in (-1, 1):
                        end = radius
                        # Shrink toward the known feasible incumbent at a boundary.
                        while end > 0.01 and evaluate(sign * end)[1] is None:
                            end /= 2
                        result = minimize_scalar(lambda t: evaluate(sign * t)[2],
                                                 bounds=(0, end), method="bounded",
                                                 options={"xatol": 0.008, "maxiter": 14})
                        for step in (end, result.x):
                            trial, plan, candidate = evaluate(sign * step)
                            if (candidate < best - 1e-8
                                    and trajectory_clear(tracks_for(past, plan, fault), fault)):
                                best, best_points, best_plan = candidate, trial, plan
                    if best < value:
                        value, points, duties = best, best_points, best_plan
        return duties


def solve(search, base, fault, release, targets, layouts, original_starts):
    """Keep the fixed plan as a candidate and search feasible causal alternatives."""
    best, best_cut = base, None
    best_starts = observation_starts(base)
    key = lambda starts: (-len(starts), max(starts.values(), default=0.0))
    for cut in sorted({release, 35.0, *(t for t in original_starts if t >= release)}):
        if len(best_starts) == len(targets) and cut >= max(best_starts.values()):
            break
        past, states, locked = snapshot(base, fault, cut)
        pending = targets.keys() - locked.keys()
        if not pending:
            continue
        seeds = []
        seen = set()
        for tasks, points in layouts:
            if not pending <= {task for task, _, _ in tasks}:
                continue
            layout = tuple(group for group in tasks if group[0] in pending)
            stations = tuple(point for i, group in enumerate(tasks) if group[0] in pending
                             for point in points[2 * i:2 * i + 2])
            if (layout, stations) in seen:
                continue
            seen.add((layout, stations))
            duties = search.replay(layout, stations, states, targets)
            if duties is None:
                continue
            tracks = tracks_for(past, duties, fault)
            if trajectory_clear(tracks, fault):
                seeds.append((key(observation_starts(tracks)), layout, duties, tracks))
        if not seeds:
            continue
        # Refine the seed with the best score at this release state.
        _, layout, duties, tracks = min(seeds, key=lambda row: row[0])
        if key(observation_starts(tracks)) < key(best_starts):
            best, best_cut, best_starts = tracks, cut, observation_starts(tracks)
        refined = search.refine(layout, duties, past, states, locked, targets, fault)
        tracks = tracks_for(past, refined, fault)
        starts = observation_starts(tracks)
        if key(starts) < key(best_starts):
            best, best_cut, best_starts = tracks, cut, starts
    return best, best_cut
