"""Accuracy bounds and continuous station refinement for Q2(2)."""
import math
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from q2_error_solve import local_error
from q2_schedule import EDGE, OBSERVE, RADIUS, SPEED, STARTS, TARGETS, objective, replay, turn_time
from q2_schedule_bound import point_distance


SIGMA = math.radians(1)
MIN_RANGES = (TARGETS[:, 2] - 2) / math.tan(math.radians(80))
ERROR_FLOOR = float(max(SIGMA * math.sqrt(2) * MIN_RANGES))
# Widen an error test in the lower-bound model and tighten it in the upper-bound model.
ERROR_MARGIN = 2e-8


def rms(a, b, task):
    u, v = a - TARGETS[task, :2], b - TARGETS[task, :2]
    r1, r2 = math.hypot(*u), math.hypot(*v)
    determinant = abs(u[0] * v[1] - u[1] * v[0])
    return SIGMA * math.hypot(r1, r2) * r1 * r2 / determinant if determinant else math.inf


def pair_sites(legs, task):
    return tuple(leg.route[-1] for leg in legs if leg.task == task)


def build_cover(task, limit, turn_bound):
    """Cover every admissible stop, then relax compatibility between boxes."""
    axis = np.linspace(RADIUS, EDGE, 1 + math.ceil((EDGE - RADIUS) / 3))
    target, radius_floor = TARGETS[task, :2], MIN_RANGES[task]
    # Larger values already cover the entire field, and the cap avoids overflow.
    budget = min((limit + ERROR_MARGIN) / SIGMA, 200.0)
    max_range = math.sqrt(max(0.0, budget**2 - radius_floor**2))
    cells, ranges, slews = [], [], []
    for a, b in zip(axis, axis[1:]):
        for c, d in zip(axis, axis[1:]):
            cell = (a, b, c, d)
            distance = max(radius_floor, point_distance(target, cell) - 1e-9)
            slew = turn_bound(cell, task)
            if distance <= max_range and math.isfinite(slew):
                cells.append(cell)
                ranges.append(distance)
                slews.append(slew)
    if not cells:
        return np.empty((0, 4)), np.empty((len(STARTS), 0))
    cells = np.array(cells)
    corners = [cells[:, [x, y]] - target for x, y in ((0, 2), (0, 3), (1, 2), (1, 3))]
    determinant_max = np.zeros((len(cells), len(cells)))
    for u in corners:
        for v in corners:
            # A bilinear determinant reaches its extreme over two rectangles at corners.
            value = np.abs(np.outer(u[:, 0], v[:, 1]) - np.outer(u[:, 1], v[:, 0]))
            np.maximum(determinant_max, value, out=determinant_max)
    radii = np.array(ranges)
    sine_max = np.minimum(1, (determinant_max + 1e-8) / np.outer(radii, radii))
    compatible = SIGMA * np.hypot(radii[:, None], radii) <= (limit + ERROR_MARGIN) * sine_max
    ready = np.array([[point_distance(start, cell) / SPEED for cell in cells]
                      for start in STARTS]) + slews
    pair_ready = np.empty_like(ready)
    for car in range(len(STARTS)):
        other = np.min(np.delete(ready, car, axis=0), axis=0)
        partner = np.min(np.where(compatible, other[None, :], np.inf), axis=1)
        pair_ready[car] = np.maximum(ready[car], partner)
    return cells, pair_ready


class AccuracyLimit:
    def __init__(self, limit):
        self.limit = limit
        self.task_floor = 0.0
        self.cover = ()
        self.pair_floor = lru_cache(maxsize=100_000)(self._pair_floor)

    def prepare(self, turn_bound):
        if self.limit < ERROR_FLOOR - ERROR_MARGIN:
            self.task_floor = math.inf
            return
        self.cover = tuple(build_cover(task, self.limit, turn_bound) for task in range(len(TARGETS)))
        self.task_floor = float(max(ready.min(initial=np.inf) for _, ready in self.cover))

    def _pair_floor(self, car, task, box):
        cells, ready = self.cover[task]
        a, b, c, d = box
        # Boundary overlap is included even when the node touches only one face.
        keep = ((cells[:, 0] <= b) & (cells[:, 1] >= a)
                & (cells[:, 2] <= d) & (cells[:, 3] >= c))
        return float(ready[car, keep].min(initial=np.inf))

    def errors(self, legs):
        return tuple(rms(*pair_sites(legs, task), task) for task in range(len(TARGETS)))

    def accepts(self, legs):
        for task, value in enumerate(self.errors(legs)):
            a, b = pair_sites(legs, task)
            if not (value <= self.limit
                    and local_error(a, b, TARGETS[task]) <= self.limit):
                return False
        return True

    def refine(self, legs, site_clearance, roads, verify):
        """Move all six stations continuously for this candidate assignment."""
        if not legs:
            return None
        starts = [max(leg.start for leg in legs if leg.task == task)
                  for task in range(len(TARGETS))]
        sites = np.array([leg.route[-1] for leg in legs])
        teams, previous = {}, {}
        predecessors = []
        for index, leg in enumerate(legs):
            teams.setdefault(leg.task, []).append(index)
            predecessors.append(previous.get(leg.car))
            previous[leg.car] = index
        slack = self.limit - ERROR_MARGIN

        def constraints(raw):
            points = raw[:sites.size].reshape(sites.shape)
            events, upper = raw[sites.size:-1], raw[-1]
            slews = [float(turn_time(p, leg.task)) for p, leg in zip(points, legs)]
            values = [*(upper - events)]
            for index, (p, leg, previous) in enumerate(zip(points, legs, predecessors)):
                origin, depart = STARTS[leg.car], 0.0
                if previous is not None:
                    origin = points[previous]
                    depart = events[legs[previous].task] + OBSERVE + slews[previous]
                route = np.vstack((origin, leg.route[1:-1], p))
                travel = np.linalg.norm(np.diff(route, axis=0), axis=1).sum() / SPEED
                values.append(events[leg.task] - depart - travel - slews[index])
                values.extend(site_clearance(p, leg.task))
            for task, (first, second) in teams.items():
                a, b = points[first], points[second]
                u, v = a - TARGETS[task, :2], b - TARGETS[task, :2]
                cross = abs(u[0] * v[1] - u[1] * v[0])
                r1, r2 = math.hypot(*u), math.hypot(*v)
                # Equivalent to RMS <= slack when both ranges are positive.
                values.append(cross / max(r1 * r2, 1e-12) - SIGMA * math.hypot(r1, r2) / slack)
                values.append(math.dist(a, b) - 2 * RADIUS)
            return values

        result = minimize(
            lambda raw: raw[-1], np.r_[sites.ravel(), starts, max(starts)],
            bounds=[(RADIUS, EDGE)] * sites.size + [(0, None)] * (len(starts) + 1),
            constraints={"type": "ineq", "fun": constraints},
            method="SLSQP", options={"ftol": 2e-10, "maxiter": 90},
        )
        points = result.x[:sites.size].reshape(sites.shape)
        plan = [(task, [(legs[i].car, points[i]) for i in indices])
                for task, indices in teams.items()]
        trial = replay(roads, plan)
        best = legs if self.accepts(legs) else None
        if best is not None and objective(trial) >= objective(best):
            return best
        try:
            verify(trial)
        except AssertionError:
            return best
        return trial if self.accepts(trial) else best
