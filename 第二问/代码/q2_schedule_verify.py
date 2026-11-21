"""Independently check the exported schedules and draw their timelines."""
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from q2_schedule import BUILDINGS, OUTPUT, RADIUS, STARTS, TARGETS


def nearest_on_segment(start, end, low, high):
    """Exact minimum distance from a segment to an axis-aligned box."""
    velocity = end - start
    times = {0.0, 1.0}
    for axis in range(len(start)):
        if velocity[axis] != 0:
            for face in (low[axis], high[axis]):
                t = (face - start[axis]) / velocity[axis]
                if 0 < t < 1:
                    times.add(t)
    times = sorted(times)
    distances = []
    for left, right in zip(times, times[1:]):
        midpoint = start + ((left + right) / 2) * velocity
        active = (midpoint < low) | (midpoint > high)
        residual = (start - np.clip(midpoint, low, high))[active]
        direction = velocity[active]
        slope = np.dot(direction, direction)
        t = np.clip(-np.dot(residual, direction) / slope, left, right) if slope else left
        point = start + t * velocity
        distances.append(np.linalg.norm(point - np.clip(point, low, high)))
    return min(distances)


def read_rows():
    fields = ["出发_s", "到站_转动开始_s", "指向完成_s", "观测开始_s",
              "观测结束_复位开始_s", "复位完成_s"]
    plans = defaultdict(list)
    with (OUTPUT / "第二问第（2）题逐车时间轴.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            car, task = int(row["车辆"][1:]) - 1, int(row["告警"][1:]) - 1
            route = np.array(json.loads(row["路径坐标_m"]))
            times = np.array([float(row[field]) for field in fields])
            plans[row["方案"]].append((car, task, route, times))
    return plans


def check(legs):
    assert len(legs) == 6
    by_car, by_task = defaultdict(list), defaultdict(list)
    road_clearance, sight_clearance = math.inf, math.inf
    for car, task, route, times in legs:
        a, b, ready, start, end, reset = times
        assert np.all(np.diff(times) >= -1e-8)
        assert np.all((RADIUS <= route) & (route <= 100 - RADIUS))
        for left, right, bottom, top, roof in BUILDINGS:
            for p, q in zip(route, route[1:]):
                distance = nearest_on_segment(p, q, [left, bottom], [right, top])
                road_clearance = min(road_clearance, distance)
            distance = nearest_on_segment(
                np.r_[route[-1], 2], TARGETS[task], [left, bottom, 0], [right, top, roof]
            )
            sight_clearance = min(sight_clearance, distance)
        x, y, height = TARGETS[task] - np.r_[route[-1], 2]
        horizontal = math.hypot(x, y)
        pitch = math.degrees(math.atan(height / horizontal))
        bearing = math.degrees(math.acos(x / horizontal))
        assert -10 <= pitch <= 80
        turn = max(bearing, abs(pitch)) / 30
        distance = sum(math.dist(p, q) for p, q in zip(route, route[1:]))
        assert math.isclose(b - a, distance / 3, abs_tol=1e-8)
        assert math.isclose(ready - b, turn, abs_tol=1e-8)
        assert math.isclose(end - start, 5, abs_tol=1e-8)
        assert math.isclose(reset - end, turn, abs_tol=1e-8)
        by_car[car].append((route, times))
        by_task[task].append((car, route[-1], ready, start))
    assert road_clearance >= RADIUS - 1e-8 and sight_clearance > 1e-8
    angles = []
    assert set(by_task) == set(range(3)) and set(by_car) == set(range(5))
    for task, group in sorted(by_task.items()):
        (car_a, a, ready_a, start_a), (car_b, b, ready_b, start_b) = group
        assert car_a != car_b and math.dist(a, b) >= 2 * RADIUS
        assert math.isclose(start_a, start_b, abs_tol=1e-8)
        assert math.isclose(start_a, max(ready_a, ready_b), abs_tol=1e-8)
        u, v = TARGETS[task, :2] - a, TARGETS[task, :2] - b
        cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        angles.append(math.degrees(math.acos(np.clip(cosine, -1, 1))))
    assert all(0 < angle < 180 for angle in angles)

    tracks = {}
    for car, duties in by_car.items():
        previous, ready = STARTS[car], 0.0
        track = [(0.0, previous)]
        for route, times in sorted(duties, key=lambda duty: duty[1][0]):
            assert np.allclose(route[0], previous, rtol=0, atol=1e-8)
            assert math.isclose(ready, times[0], abs_tol=1e-8)
            t = times[0]
            for p, q in zip(route, route[1:]):
                t += math.dist(p, q) / 3
                track.append((t, q))
            ready, previous = times[-1], route[-1]
            track.append((ready, previous))
        t, positions = zip(*track)
        tracks[car] = np.array(t), np.array(positions)

    separation = math.inf
    for a, b in combinations(tracks, 2):
        times = np.unique(np.r_[tracks[a][0], tracks[b][0]])
        relative = np.zeros((len(times), 2))
        for car, sign in ((a, 1), (b, -1)):
            t, positions = tracks[car]
            for axis in range(2):
                relative[:, axis] += sign * np.interp(times, t, positions[:, axis])
        for p, q in zip(relative, relative[1:]):
            separation = min(separation, nearest_on_segment(p, q, [0, 0], [0, 0]))
    assert separation >= 2 * RADIUS - 1e-8
    return road_clearance, sight_clearance, separation, angles


def plot(plans):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=font_path) if font_path.is_file() else None
    colors = ["#32718a", "#c47530", "#c9d1d6", "#306a51", "#9171a2"]
    labels = ["行驶", "云台转动", "等待同组车辆", "联合观测", "云台复位"]
    figure, panels = plt.subplots(2, 1, sharex=True, figsize=(12, 6.2))
    figure.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.185, hspace=0.52)
    for panel, (name, legs) in zip(panels, plans.items()):
        for car, task, _, times in legs:
            for left, right, color in zip(times, times[1:], colors):
                if right - left > 1e-7:
                    panel.barh(car, right - left, left=left, height=0.58,
                               color=color, edgecolor="white", linewidth=0.6)
            panel.text((times[3] + times[4]) / 2, car, f"T{task + 1}",
                       color="white", ha="center", va="center", fontsize=9)
        limit = max(leg[3][3] for leg in legs)
        panel.axvline(limit, color="#232e36", linestyle="--", linewidth=1)
        panel.set_title(f"{name}　｜　最晚观测开始 {limit:.3f} s",
                        fontproperties=font, loc="left", fontsize=13, pad=9)
        panel.set_yticks(range(5), [f"V{k}" for k in range(1, 6)])
        panel.set_xlim(0, 29)
        panel.set_xticks(range(0, 29, 2))
        panel.set_ylim(4.62, -0.62)
        panel.grid(axis="x", color="#dbe1e5", linewidth=0.55)
        panel.set_axisbelow(True)
        panel.spines[["top", "right", "left"]].set_visible(False)
        panel.spines["bottom"].set_color("#a9b5bd")
        panel.tick_params(axis="both", length=0, pad=5, labelsize=9)
    panels[-1].set_xlabel("告警发布后的时间（s）", fontproperties=font, labelpad=6)
    figure.legend([Patch(color=c) for c in colors], labels, ncol=5, frameon=False,
                  loc="lower center", prop=font, bbox_to_anchor=(0.5, 0.017))
    figure.savefig(OUTPUT.parent / "图片" / "第二问第（2）题时间轴.png", dpi=155)
    plt.close(figure)


if __name__ == "__main__":
    plans = read_rows()
    assert set(plans) == {"优化", "最近指派"}
    for name, legs in plans.items():
        road, sight, distance, angles = check(legs)
        print(f"{name}: road={road:.8f} m, sight={sight:.8f} m, vehicle={distance:.8f} m, "
              f"intersection degrees={[round(angle, 4) for angle in angles]}")
    plot(plans)
