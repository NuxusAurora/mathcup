"""Solve and verify the Q2(2) Plan B with 10 cm waypoint and station errors."""
import argparse
import math
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from q2_schedule import (
    OBSERVE, OUTPUT, RADIUS, SPEED, STARTS, TARGETS, TURN, export, replay,
)
from q2_schedule_verify import check, nearest_on_segment, read_rows

OFFSET = 0.10
RMS_LIMIT = 1.0
BEARING_SIGMA = math.radians(1.0)
B_PATH = OUTPUT / "第二问第（2）题B实际方案时间轴.csv"
A_PATH = OUTPUT / "第二问第（2）题连续分支定界时间轴.csv"
SEED = (83.200, 62.429, 38.781, 95.016)


class PlanBRoutes:
    """Route policy for the selected vehicle assignment."""

    def route(self, origin, destination):
        if np.array_equal(origin, STARTS[2]):
            return np.array([origin, (25.501, 77.501), destination])
        return np.array([origin, destination])


def nominal_legs(stations):
    v4, v5 = np.reshape(stations, (2, 2))
    assignments = [
        (0, [(3, v4), (4, v5)]),
        (1, [(0, np.array([13.790, 46.670])), (4, v5)]),
        (2, [(1, np.array([47.990, 11.130])), (2, np.array([31.310, 50.060]))]),
    ]
    return replay(PlanBRoutes(), assignments)


def station_bounds(station, task):
    return station_bounds_at(station, TARGETS[task])


def station_bounds_at(station, target):
    direction = np.asarray(target[:2]) - station
    distance = np.linalg.norm(direction)
    azimuth_error = math.asin(OFFSET / distance)
    pitch = math.atan2(target[2] - 2, distance - OFFSET)
    yaw = min(math.pi, abs(math.atan2(direction[1], direction[0])) + azimuth_error)
    return math.degrees(pitch), math.degrees(max(pitch, yaw)) / TURN


def rms_bound(first, second, task):
    """Bound first-order RMS when the estimator uses nominal station coordinates."""
    return rms_bound_at(first, second, TARGETS[task])


def rms_bound_at(first, second, target):
    rays = np.asarray(target[:2]) - np.array([first, second])
    radii = np.linalg.norm(rays, axis=1)
    cosine = (rays[0] @ rays[1]) / np.prod(radii)
    acute = math.acos(np.clip(abs(cosine), 0, 1))
    min_angle = acute - np.arcsin(OFFSET / radii).sum()
    if min_angle <= 0:
        return math.inf

    # Each actual bearing line misses the target by at most OFFSET when
    # translated to the planned station. The row normals have unit length.
    bias = OFFSET / math.sin(min_angle / 2)
    # Linearize the noisy intersection at the biased, noise-free intersection.
    # Its distance to station i is at most nominal range i plus the bias bound.
    noise = BEARING_SIGMA * np.linalg.norm(radii + bias) / math.sin(min_angle)
    return math.hypot(bias, noise)


def worst_schedule(rows):
    """Propagate readiness bounds. Equal waypoints denote a single parked position."""
    available = np.zeros(len(STARTS))
    start_upper = []
    for task in range(len(TARGETS)):
        pending = []
        for car, duty, route, _ in rows:
            if duty != task:
                continue
            length = np.linalg.norm(np.diff(route, axis=0), axis=1)
            drive = (length.sum() + 2 * OFFSET * np.count_nonzero(length)) / SPEED
            _, turn = station_bounds(route[-1], task)
            pending.append((car, available[car] + drive + turn, turn))
        start = max(ready for _, ready, _ in pending)
        start_upper.append(start)
        for car, _, turn in pending:
            available[car] = start + OBSERVE + turn
    return np.array(start_upper)


def segment_separation(a, b, c, d):
    """Exact distance between two closed planar line segments."""
    ab, cd = b - a, d - c
    offset = c - a
    normal_ab, normal_cd = np.array([-ab[1], ab[0]]), np.array([-cd[1], cd[0]])
    determinant = normal_ab @ cd
    if determinant:
        u = (normal_cd @ offset) / (normal_cd @ ab)
        v = -(normal_ab @ offset) / determinant
        if 0 <= u <= 1 and 0 <= v <= 1:
            return 0.0
    return min(nearest_on_segment(p, q, x, x)
               for p, q, x in ((a, b, c), (a, b, d), (c, d, a), (c, d, b)))


def audit(rows):
    assert sorted((car, task) for car, task, _, _ in rows) == [
        (0, 1), (1, 2), (2, 2), (3, 0), (4, 0), (4, 1),
    ], "Unexpected Plan B assignment"
    first, second = sorted((task, route, times) for car, task, route, times in rows
                           if car == 4)
    assert first[2][0] < second[2][0] and np.all(second[1] == first[1][-1]), (
        "V5 must serve T1 first and stay parked for T2"
    )
    center, sight, _, _ = check(rows)
    by_task = [[row for row in rows if row[1] == task] for task in range(len(TARGETS))]
    start = np.array([group[0][3][3] for group in by_task])
    rms = [rms_bound(first[2][-1], second[2][-1], first[1])
           for first, second in by_task]
    field = min(np.minimum(route, 100 - route).min() for _, _, route, _ in rows)
    pitch = max(station_bounds(route[-1], task)[0] for _, task, route, _ in rows)
    separation = min(
        segment_separation(a, b, c, d)
        for (car, _, route, _), (other, _, path, _) in combinations(rows, 2)
        if car != other
        for a, b in pairwise(route)
        for c, d in pairwise(path)
    )
    margins = (center - RADIUS - OFFSET, sight - OFFSET,
               field - RADIUS - OFFSET, separation - 2 * OFFSET)
    assert min(margins[:3]) > 0 and margins[3] > 2 * RADIUS, margins
    assert pitch <= 80 and max(rms) <= RMS_LIMIT, (pitch, rms)
    return start, worst_schedule(rows), rms, margins, pitch


def check_rows(legs):
    return [(leg.car, leg.task, leg.route, [
        leg.depart, leg.arrive, leg.ready, leg.start, leg.end, leg.reset,
    ]) for leg in legs]


def solve():
    # Optimize the two T1 stations for the conservative latest-start bound.
    # Keep a little slack on pitch and T1/T2 accuracy before rounding to mm.
    def constraints(x):
        v4, v5 = np.reshape(x[:4], (2, 2))
        rows = check_rows(nominal_legs(x[:4]))
        return [
            *(x[4] - worst_schedule(rows)),
            0.995 - rms_bound(v4, v5, 0),
            0.995 - rms_bound(np.array([13.790, 46.670]), v5, 1),
            79.98 - station_bounds(v4, 0)[0],
        ]

    result = minimize(
        lambda x: x[4], [*SEED, 16.3], method="SLSQP",
        bounds=[(80, 87), (60, 64), (37, 45), (90, 98.3), (15, 22)],
        constraints={"type": "ineq", "fun": constraints},
        options={"ftol": 1e-10, "maxiter": 250},
    )
    print(f"Local optimization: {result.message}")
    candidates = []
    for stations in (SEED, np.round(result.x[:4], 3)):
        legs = nominal_legs(stations)
        try:
            _, deadline, _, _, _ = audit(check_rows(legs))
        except AssertionError:
            continue
        candidates.append((max(deadline), legs))
    if not candidates:
        raise ValueError("No verified Plan B candidate")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--solve", type=Path, metavar="CSV",
                        help="solve Plan B and save the nominal event timeline")
    action.add_argument("--verify", type=Path, metavar="CSV",
                        help="verify a Plan B timeline (default: the delivered CSV)")
    args = parser.parse_args()
    if args.solve:
        export({"B实际方案": solve()}, args.solve)

    a_rows, = read_rows(A_PATH).values()
    check(a_rows)
    print(f"A latest nominal start: {max(row[3][3] for row in a_rows):.9f} s")
    rows, = read_rows(args.solve or args.verify or B_PATH).values()
    start, deadline, rms, margins, pitch = audit(rows)
    for j, (nominal, upper, error) in enumerate(zip(start, deadline, rms), 1):
        print(f"T{j}: nominal {nominal:.9f} s, upper {upper:.9f} s, "
              f"RMS upper {error:.9f} m")
    print(f"Max pitch {pitch:.9f} deg")
    print("Minimum lower bounds (body/building, sight/building, body/field, "
          "center/center), m:", ", ".join(f"{m:.9f}" for m in margins))


if __name__ == "__main__":
    main()
