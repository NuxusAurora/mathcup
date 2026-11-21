"""Reproduce the static measurement experiment with Python 3 and NumPy."""
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


R = 40.0
RAD = math.pi / 180
LIMIT = 5 * RAD
CDF = NormalDist().cdf
N = 1_000_000
SEED = 20261120


def quadrature(grid_size):
    """Marginalize the length of an isotropic normal vector in polar coordinates."""
    phi = np.linspace(-math.pi, math.pi, grid_size, endpoint=False)
    cos_phi = np.cos(phi)
    sin_sq = np.sin(phi) ** 2
    # The omitted wrapped-normal images are at least 535 standard deviations away.
    kernel = np.array([sum(CDF((LIMIT - x + 2*k*math.pi) / RAD)
                           - CDF((-LIMIT - x + 2*k*math.pi) / RAD)
                           for k in (-1, 0, 1)) for x in phi])

    def probability(sigma):
        rho = R / sigma
        z = rho * cos_phi
        cdf_z = np.array([0.5 * math.erfc(-v / math.sqrt(2)) for v in z])
        density = (math.exp(-rho**2 / 2) + math.sqrt(2*math.pi) * z
                   * np.exp(-rho**2 * sin_sq / 2) * cdf_z) / (2*math.pi)
        step = 2*math.pi / grid_size
        assert abs(density.sum()*step - 1) < 1e-9
        return float(density @ kernel * step)
    return probability


def bisect_sigma(probability):
    low, high = 0.3, 4.0
    assert probability(low) > 0.95 > probability(high)
    for _ in range(40):
        mid = (low + high) / 2
        if probability(mid) >= 0.95:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def wilson(k):
    p = k/N
    z = NormalDist().inv_cdf(0.975)
    den = 1 + z*z/N
    center = (p + z*z/(2*N))/den
    half = z/den * math.sqrt(p*(1-p)/N + z*z/(4*N*N))
    return [center-half, center+half]


def measurement():
    fine = quadrature(8192)
    sigma = bisect_sigma(fine)
    coarse = bisect_sigma(quadrature(4096))
    linear = R*math.sqrt((LIMIT/NormalDist().inv_cdf(0.975))**2 - RAD**2)
    rng = np.random.default_rng(SEED)
    x, y, zero, control = rng.normal(size=(4, N))
    threshold = (sigma - 0.3)/0.02
    rows = []
    for s in [0, math.floor(threshold), threshold, math.ceil(threshold), 80]:
        sd = 0.3 + 0.02*s
        angle = np.arctan2(-sd*y, R-sd*x) + (0.8*zero + 0.6*control)*RAD
        wrapped = (angle + math.pi) % (2*math.pi) - math.pi
        k = int(np.sum(abs(wrapped) <= LIMIT))
        truth = fine(sd)
        assert abs(k/N-truth) < 5*math.sqrt(truth*(1-truth)/N)
        rows.append(dict(s_m=s, quadrature=truth, simulation=k/N, wilson95=wilson(k)))
    var = np.array([(sigma/R)**2, (0.8*RAD)**2, (0.6*RAD)**2])
    return dict(seed=SEED,n=N,numpy=np.__version__,sigma_m=sigma,s_m=threshold,
                coarse_fine_sigma_diff_m=abs(coarse-sigma),linear_s_m=(linear-0.3)/0.02,
                first_order_variance_shares=(var/var.sum()).tolist(),coverage=rows)


def rule_checks():
    # Synthetic boundary examples, independent of the contest map.
    def visible_across_roof(z_entry, z_exit, roof):
        return min(z_entry, z_exit) > roof
    assert not visible_across_roof(10, 12, 10)
    assert visible_across_roof(10.001, 12, 10)
    assert not visible_across_roof(11, 8, 10)
    # Compare the closed-form local RMS against explicit inversion of the Jacobian.
    angles = [1, 30, 90, 150, 179]
    rms = []
    for degrees in angles:
        a = degrees * RAD
        G = np.array([[0, 1/R], [-math.sin(a)/R, math.cos(a)/R]])
        value = RAD*math.sqrt(np.trace(np.linalg.inv(G.T @ G)))
        assert math.isclose(value, RAD*R*math.sqrt(2)/abs(math.sin(a)), rel_tol=1e-10)
        rms.append(float(value))
    def score(starts):
        ok = [s for s in starts if s is not None and 0 <= s <= 60]
        return len(ok)/len(starts), max(ok) if len(ok)==len(starts) else 60
    assert score([0, 5, 60]) == (1, 60)
    assert score([0, 5, 60.001]) == (2/3, 60)
    assert score([None]*4) == (0, 60)
    return dict(synthetic_angle_deg=angles, local_rms_m=rms,
                interpretation_B=dict(duration_s=[5,15,30],recovery_s=[5+d for d in (5,15,30)]))


if __name__ == "__main__":
    result = dict(measurement=measurement(), checks=rule_checks())
    (Path(__file__).resolve().parents[1] / "文档" / "measurement_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
