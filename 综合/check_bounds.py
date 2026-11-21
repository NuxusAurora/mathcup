"""Independent geometric and scoring checks. Run with Python 3 and NumPy."""
from itertools import combinations
from math import erfc, hypot, pi, sqrt

import numpy as np


RECTANGLES = [(22,34,16,26), (63,79,59,71), (10,24,65,76),
              (72,86,22,34), (41,51,38,46)]
RADIUS = 1.5


def free_area():
    excluded = 0.0
    for x0,x1,y0,y1 in RECTANGLES:
        assert min(x0,y0,100-x1,100-y1) >= 2*RADIUS
        w,h = x1-x0, y1-y0
        excluded += w*h + 2*RADIUS*(w+h) + pi*RADIUS**2
    for a,b in combinations(RECTANGLES,2):
        dx = max(a[0]-b[1], b[0]-a[1], 0)
        dy = max(a[2]-b[3], b[2]-a[3], 0)
        assert hypot(dx,dy) >= 2*RADIUS
    return (100-2*RADIUS)**2-excluded


def grid_area(step):
    x = np.arange(RADIUS + step/2, 100-RADIUS, step)
    count = 0
    for y in x:
        allowed = np.ones(x.size, dtype=bool)
        for x0,x1,y0,y1 in RECTANGLES:
            dx = np.maximum(np.maximum(x0-x, x-x1), 0)
            dy = max(y0-y, y-y1, 0)
            allowed &= dx**2 + dy**2 >= RADIUS**2
        count += int(allowed.sum())
    return count*step**2


def angle_check():
    # Difference of two independent 1-degree measurement errors.
    rng = np.random.default_rng(261120)
    perturb = rng.normal(size=1_000_000) - rng.normal(size=1_000_000)
    rows = []
    for angle in (1,5,30,90):
        flip = np.mean(angle+perturb <= 0)
        expected = erfc(angle/2)/2
        assert abs(flip-expected) <= 6*sqrt(expected*(1-expected)/len(perturb))+1e-6
        rows.append((angle,expected,float(flip)))
    return rows


def scoring_check():
    # Abstract pairs verify the expectation for any fixed unfinished task.
    for pair in combinations(range(5),2):
        assert sum(failed not in pair for failed in range(5)) == 3
    for q in range(3):
        for n in (3,4):
            expected_bound = (q+(3-q)*3/5)/n
            print(f"before={q}, alerts={n}, mean ceiling={expected_bound:.6f}")


if __name__ == "__main__":
    area = free_area()
    print(f"analytic free area={area:.9f}")
    for step in (0.2,0.1,0.05):
        print(f"step={step}, grid free area={grid_area(step):.6f}")
    assert abs(grid_area(0.05)-area)<0.2
    for row in angle_check():
        print("angle, analytic and simulated sign reversal:",row)
    scoring_check()
