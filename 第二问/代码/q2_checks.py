"""Reproduce the two abstract examples in the question 2 report (Python + NumPy)."""
import csv
from math import cos, radians, sin, sqrt
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "文档"


def save_rows(filename, rows):
    with (OUTPUT / filename).open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def error_examples():
    sigma = radians(1)
    # Common draws make comparisons of the local models less noisy.
    noise = np.random.default_rng(20261121).normal(0, sigma, (200_000, 2))
    rows = []
    for r1, r2, angle in [(40,40,30), (40,40,60), (40,40,90),
                          (40,40,150), (20,60,90)]:
        g = radians(angle)
        jac = np.array([[0, 1/r1], [-sin(g)/r2, cos(g)/r2]])
        covariance = sigma**2 * np.linalg.inv(jac.T @ jac)
        local_rms = sigma * sqrt(r1*r1+r2*r2) / abs(sin(g))
        draws = np.linalg.solve(jac, noise.T).T
        sample_rms = sqrt(np.sum(draws**2)/len(draws))
        assert np.isclose(np.trace(covariance), local_rms**2)
        assert abs(sample_rms/local_rms - 1) < 0.01
        rows.append({"r1_m":r1, "r2_m":r2, "angle_deg":angle,
                     "local_rms_m":round(local_rms,6),
                     "sample_linear_rms_m":round(sample_rms,6),
                     "major_axis_sd_m":round(sqrt(np.linalg.eigvalsh(covariance)[-1]),6)})
    return rows


def timing_examples():
    first_drive = {"A":1, "B":3}
    turn = {"A":1, "B":2}
    partner_drive = {"A":9, "B":1}
    partner_turn = {"A":3, "B":1}
    rows = []
    for order in ("AB", "BA"):
        release = 0
        starts = []
        for index, job in enumerate(order):
            drive = first_drive[job] if index==0 else 4
            arrival = release + drive
            ready = arrival + turn[job]
            other_ready = partner_drive[job] + partner_turn[job]
            start = max(ready, other_ready)
            end = start + 5
            resources = [("R",release,arrival,ready,turn[job]),
                         ("P_"+job,0,partner_drive[job],other_ready,partner_turn[job])]
            for resource, depart, arrive, aimed, reset in resources:
                rows.append({"order":order, "task":job, "resource":resource,
                             "depart_s":depart, "arrive_s":arrive, "ready_s":aimed,
                             "wait_s":start-aimed, "start_s":start, "end_s":end,
                             "reset_done_s":end+reset})
            release = end + turn[job]
            starts.append(start)
        assert starts == ([12,24] if order=="AB" else [5,17])
    return rows


if __name__ == "__main__":
    save_rows("局部精度算例.csv", error_examples())
    save_rows("同步时序算例.csv", timing_examples())
