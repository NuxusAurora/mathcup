"""Finite-difference validation of the local bearing error."""
import math

import numpy as np


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def local_error(a, b, target):
    """Numerically differentiate the two-bearing intersection, in metres per degree."""
    bearings = [np.arctan2(p[..., 1] - target[1], p[..., 0] - target[0])
                for p in (a, b)]
    h = 0.00001  # degrees, well within the local regime
    step = math.radians(h)

    def intersection(da, db):
        alpha, beta = bearings[0] + da, bearings[1] + db
        u = np.stack((np.cos(alpha), np.sin(alpha)), axis=-1)
        v = np.stack((np.cos(beta), np.sin(beta)), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return a + (cross(b - a, v) / cross(u, v))[..., None] * u

    first = (intersection(step, 0) - intersection(-step, 0)) / (2 * h)
    second = (intersection(0, step) - intersection(0, -step)) / (2 * h)
    # Each bearing has independent noise with a standard deviation of one degree.
    return np.sqrt(np.sum(first**2 + second**2, axis=-1))
