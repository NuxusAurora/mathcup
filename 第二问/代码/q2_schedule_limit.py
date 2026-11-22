"""Locate the continuous infimum and test whether an admissible plan attains it."""
import math
from itertools import product

import numpy as np
from scipy.optimize import root

from q2_schedule import (
    OBSERVE, OMEGA, SPEED, STARTS, TARGETS, shadow_polygons, turn_time,
)
from q2_schedule_bound import (
    certify, make_turn_bound, point_distance, split, t1_partner_lower, width,
)
from q2_schedule_verify import read_rows


FIRST_BOX = (14.94, 15.15, 86.97, 87.18)
SECOND_BOX = (16.66, 16.87, 88.55, 88.77)
LOCALIZATION_TIME = 10.352124
LOWER = 10.35211850
ORIGIN = STARTS[2]


def vertices(box):
    return np.array(list(product(box[:2], box[2:])))


def shadow_corner():
    x, y, h = TARGETS[2]
    # Far north face of B5's shadow and the ray through B3's northeast corner.
    north = y + (h - 2) / (h - 32) * (46 - y)
    east = 24 + (x - 24) / (y - 76) * (north - 76)
    return np.array([east, north])


def verify_local_geometry(q):
    """Check strict margins for the local argument and the fixed partners."""
    shadows = shadow_polygons()
    p_vertices, q_vertices = vertices(FIRST_BOX), vertices(SECOND_BOX)
    assert point_distance(q, SECOND_BOX) == 0
    for task, points in ((0, p_vertices), (2, q_vertices)):
        for building, polygon in enumerate(shadows[task]):
            signs = points @ polygon[:, :2].T + polygon[:, 2]
            if task == 2 and building in (2, 4):
                active = abs(polygon[:, :2] @ q + polygon[:, 2]) < 1e-10
                assert active.sum() == 1 and signs[:, ~active].max() < -0.1
                face = polygon[active][0]
                if building == 2:
                    assert np.all(face[:2] > 0)
                else:
                    assert np.allclose(face[:2], [0, 1], atol=1e-12)
            else:
                assert np.max(signs.min(axis=0)) > 0.1

    # Both paths stay north of all buildings. The two triangulations keep
    # positive separation and the same nonzero determinant across each box.
    assert min(ORIGIN[1], FIRST_BOX[2], SECOND_BOX[2]) > 76 + 1.5
    for box, task, partner in ((FIRST_BOX, 0, STARTS[4]), (SECOND_BOX, 2, STARTS[0])):
        assert point_distance(partner, box) > 3
        v = TARGETS[task, :2] - partner
        u = TARGETS[task, :2] - vertices(box)
        determinant = u[:, 0] * v[1] - u[:, 1] * v[0]
        assert min(determinant) * max(determinant) > 0
        r_min = point_distance(TARGETS[task, :2], box)
        assert math.degrees(math.atan2(TARGETS[task, 2] - 2, r_min)) < 80

    # On this box T3 is southeast and yaw controls the simultaneous turn.
    x, y, h = TARGETS[2]
    a, b, c, d = SECOND_BOX
    assert x > b and y < c
    r_min = point_distance((x, y), SECOND_BOX)
    assert math.atan2(c - y, x - a) > math.atan2(h - 2, r_min)
    k = (x - 24) / (76 - y)
    dx_max, dy_min, dy_max = b - FIRST_BOX[0], c - FIRST_BOX[3], d - FIRST_BOX[2]
    assert a > FIRST_BOX[1] and dy_min > 0
    # For G(q)=travel time+T3 turn, G_x and G_y-k G_x are positive.
    # These are the derivatives along the generators of the feasible wedge.
    drive = (dy_min - k * dx_max) / (SPEED * math.hypot(dx_max, dy_max))
    yaw = math.hypot(1, k) / (OMEGA * r_min)
    assert drive > yaw
    return drive - yaw


def pitch_gradient(p):
    displacement = p - TARGETS[0, :2]
    r = np.linalg.norm(displacement)
    h = TARGETS[0, 2] - 2
    return -h * displacement / (OMEGA * r * (r * r + h * h))


def verify_first_site(q, helper):
    """Bound the remaining two coordinates using box and Taylor minorants."""
    g3 = float(turn_time(q, 2))
    bound_turn = make_turn_bound()

    def components(p, weight):
        g1 = float(turn_time(p, 0))
        a, b = p - ORIGIN, p - q
        s, t = np.linalg.norm(a), np.linalg.norm(b)
        ready_gap = s / SPEED + g1 - helper
        blend = helper + OBSERVE + g1 + t / SPEED + g3 + weight * ready_gap
        gradient = (weight * a / s + b / t) / SPEED + (1 + weight) * pitch_gradient(p)
        return ready_gap, blend, gradient

    def stationary(raw):
        ready_gap, _, gradient = components(raw[:2], raw[2])
        return np.r_[ready_gap, gradient]

    trial = root(stationary, [15.044, 87.076, 0.98], tol=1e-11)
    p, weight = trial.x[:2], trial.x[2]
    assert np.max(abs(stationary(trial.x))) < 1e-12
    assert point_distance(p, FIRST_BOX) == 0 and 0 < weight < 1
    # The pitch angle wins throughout this box, so its smooth Taylor
    # bound applies. The two Euclidean norms have convex tangent bounds.
    x, y, h = TARGETS[0]
    a, b, c, d = FIRST_BOX
    far = math.hypot(max(abs(x - a), abs(x - b)), max(abs(y - c), abs(y - d)))
    yaw_max = math.atan2(max(abs(y - c), abs(y - d)), x - b)
    assert x > b and math.atan2(h - 2, far) > yaw_max
    curvature = (1 + weight) / (OMEGA * point_distance((x, y), FIRST_BOX)**2)
    ready_gap, blend, gradient = components(p, weight)
    stack = [FIRST_BOX]
    count = 0
    while stack:
        box = stack.pop()
        count += 1
        g1 = bound_turn(box, 0, False)
        simple = (max(helper, point_distance(ORIGIN, box) / SPEED + g1)
                  + OBSERVE + g1 + point_distance(q, box) / SPEED + g3)
        displacements = vertices(box) - p
        tangent = blend + min(displacements @ gradient)
        remainder = 0.5 * curvature * max(np.sum(displacements**2, axis=1))
        if max(simple, tangent - remainder) >= LOWER + 1e-10:
            continue
        assert width(box) > 1e-8
        stack.extend(split(box))

    value = blend + max(0, ready_gap) - weight * ready_gap
    print(f"First-site lower bound {LOWER:.11f} s, {count} boxes.")
    print(f"Boundary trial p={p.tolist()}, q={q.tolist()}, time={value:.12f} s.")
    return value


def main():
    plans = read_rows()
    legs = plans["连续分支定界"]
    upper = max(times[3] for _, _, _, times in legs)
    assert upper < LOCALIZATION_TIME
    assert certify(LOCALIZATION_TIME, (FIRST_BOX, SECOND_BOX))
    q = shadow_corner()
    descent_margin = verify_local_geometry(q)
    helper = t1_partner_lower(make_turn_bound())
    value = verify_first_site(q, helper)
    assert value < upper
    # The chosen other cars attain the chain's relaxations. Their positions
    # remain away from the local routes, and the other ready times are earlier.
    for car, task, route, times in legs:
        if car == 2:
            continue
        if (car, task) in ((4, 0), (0, 2)):
            np.testing.assert_array_equal(route, [STARTS[car], STARTS[car]])
        else:
            assert task == 1 and times[3] < LOWER
        if car == 4:
            assert times[2] <= helper + 1e-9
        else:
            assert times[2] < LOWER
        assert min(route[:, 0]) > SECOND_BOX[1] + 3 or max(route[:, 1]) < STARTS[2, 1] - 3
    print(f"Strict local descent margin {descent_margin:.8f} s/m.")
    print(f"Feasible upper bound {upper:.9f} s.")


if __name__ == "__main__":
    main()
