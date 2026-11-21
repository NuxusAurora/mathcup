"""Independently check the exported event records and paired scores."""
import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np

from q4_score import DEADLINE, EVENT, RESULT
from q4_timeline import (
    AIM, DRIVE, EAST, FAILED, IDLE, OBSERVE_PHASE, RESERVED, RESET, SCORES, TASKS, TRACKS, WAIT,
)
from q2_schedule import BUILDINGS, RADIUS, STARTS, TARGETS
from q2_schedule_verify import nearest_on_segment

EPS = 1e-8
# Two independently rounded nanosecond endpoints in the question 2 CSV.
ANGLE_EPS = 3.1e-8
CARS = tuple(f"V{i}" for i in range(1, 6))


@dataclass(frozen=True)
class Event:
    kind: str
    task: str
    begin: float
    end: float
    origin: tuple
    destination: tuple
    angle_from: tuple
    angle_to: tuple

    def at(self, time):
        fraction = (time - self.begin) / (self.end - self.begin)
        return tuple((1 - fraction) * a + fraction * b for a, b in zip(
            self.origin + self.angle_from, self.destination + self.angle_to
        ))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_tracks(path):
    plans = defaultdict(lambda: defaultdict(list))
    for row in read_csv(path):
        fields = ("起点x_m", "起点y_m", "终点x_m", "终点y_m",
                  "起始方位_deg", "起始俯仰_deg", "终止方位_deg", "终止俯仰_deg")
        values = tuple(float(row[name]) for name in fields)
        plans[row["时间轴ID"]][row["车辆"]].append(Event(
            row["阶段"], row["告警"], float(row["开始_s"]), float(row["结束_s"]),
            values[:2], values[2:4], values[4:6], values[6:],
        ))
    return plans


def close(a, b):
    return abs(a - b) <= EPS


@lru_cache(maxsize=None)
def road_clearance(p, q):
    return min(nearest_on_segment(np.array(p), np.array(q), [x, y], [u, v])
               for x, u, y, v, _ in BUILDINGS)


@lru_cache(maxsize=None)
def check_view(point, target):
    dx, dy, dz = np.array(target) - np.r_[point, 2.0]
    r = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, r))
    assert -10 <= pitch <= 80
    clearance = min(nearest_on_segment(np.r_[point, 2.0], np.array(target),
                                       [x, y, 0], [u, v, height])
                    for x, u, y, v, height in BUILDINGS)
    assert clearance > 1e-8, (point, target, clearance)
    yaw = math.degrees(math.acos(max(-1.0, min(1.0, dx / r))))
    return (yaw if dy >= 0 else -yaw, pitch), clearance


def spans(a, b):
    i = j = 0
    while i < len(a) and j < len(b):
        left, right = max(a[i].begin, b[j].begin), min(a[i].end, b[j].end)
        if right > left:
            yield left, right, a[i], b[j]
        i, j = i + (a[i].end <= b[j].end), j + (b[j].end <= a[i].end)


def separation(plan, fault):
    closest = math.inf
    for car_a, car_b in combinations(CARS, 2):
        for left, right, a, b in spans(plan[car_a], plan[car_b]):
            if fault in (car_a, car_b):
                right = min(right, EVENT)
            if not left < right:
                break
            x, y, *_ = a.at(left)
            u, v, *_ = b.at(left)
            p, q = x - u, y - v
            x, y, *_ = a.at(right)
            u, v, *_ = b.at(right)
            dx, dy = x - u - p, y - v - q
            stationary = math.hypot(dx, dy) <= EPS
            fraction = 0 if stationary else -(p * dx + q * dy) / (dx * dx + dy * dy)
            t = min(1, max(0, fraction))
            closest = min(closest, math.hypot(p + t * dx, q + t * dy))
    assert closest >= 2 * RADIUS - EPS, closest
    return closest


def check_track(plan, targets, fault):
    assert set(plan) == set(CARS)
    views, roads, observations = [], [], defaultdict(list)
    for car, origin in zip(CARS, STARTS):
        events = plan[car]
        time, point, angles = 0, tuple(origin), EAST
        assert events[-1].end >= DEADLINE
        for i, event in enumerate(events):
            assert event.end > event.begin and close(event.begin, time), (car, event, time)
            assert math.dist(point, event.origin) <= EPS
            assert math.dist(angles, event.angle_from) <= EPS
            assert all(RADIUS - EPS <= p <= 100 - RADIUS + EPS
                       for p in event.origin + event.destination)
            dt = event.end - event.begin
            yaw, pitch = event.angle_from
            assert -180 <= yaw <= 180 and -10 <= pitch <= 80
            yaw, pitch = event.angle_to
            assert -180 <= yaw <= 180 and -10 <= pitch <= 80
            assert max(abs(a - b) for a, b in zip(event.angle_from, event.angle_to)) <= 30 * dt + ANGLE_EPS
            if event.kind == DRIVE:
                assert event.angle_from == event.angle_to == EAST
                assert close(math.dist(event.origin, event.destination), 3 * dt)
                roads.append(road_clearance(event.origin, event.destination))
            else:
                assert event.origin == event.destination
            if event.kind == IDLE:
                assert event.angle_from == event.angle_to == EAST
            if event.kind == FAILED:
                assert car == fault and close(event.begin, EVENT) and i + 1 == len(events)
            elif car == fault:
                assert event.end <= EVENT + EPS
            if event.kind in (WAIT, OBSERVE_PHASE, RESERVED):
                required, clearance = check_view(event.origin, targets[event.task])
                assert math.dist(required, event.angle_from) < 1e-5
                assert event.angle_from == event.angle_to
                views.append(clearance)
            if event.kind == OBSERVE_PHASE:
                assert i and events[i - 1].kind in (WAIT, AIM) and events[i - 1].task == event.task
                if car == fault:
                    assert event.begin < EVENT and close(event.end, EVENT)
                else:
                    assert close(dt, 5) and events[i + 1].kind == RESET
                    assert events[i + 1].task == event.task
                observations[event.task].append((car, event, events[i - 1]))
            if event.kind == AIM:
                assert event.angle_from == EAST
                required, clearance = check_view(event.origin, targets[event.task])
                if math.dist(required, event.angle_to) <= 1e-5:
                    views.append(clearance)
                else:
                    assert i + 1 < len(events) and events[i + 1].kind in (FAILED, RESET)
            if event.kind == RESET and event.angle_to != EAST:
                assert i + 1 < len(events) and events[i + 1].kind == FAILED
            time, point, angles = event.end, event.destination, event.angle_to
        assert (events[-1].kind == FAILED) == (car == fault)

    assert min(roads, default=math.inf) >= RADIUS - EPS
    starts = {}
    for task, entries in observations.items():
        assert len(entries) == 2
        (car_a, a, before_a), (car_b, b, before_b) = entries
        assert car_a != car_b and close(a.begin, b.begin)
        ready_a = before_a.begin if before_a.kind == WAIT else before_a.end
        ready_b = before_b.begin if before_b.kind == WAIT else before_b.end
        assert close(a.begin, max(ready_a, ready_b))
        assert math.dist(a.origin, b.origin) >= 2 * RADIUS - EPS
        u, v = targets[task][:2]
        x, y = a.origin[0] - u, a.origin[1] - v
        p, q = b.origin[0] - u, b.origin[1] - v
        assert abs(x * q - y * p) > 1e-7
        assert 0 <= a.begin <= DEADLINE
        starts[task] = a.begin
    return starts, min(roads, default=math.inf), min(views), separation(plan, fault)


def difference(base, new):
    earliest = math.inf
    for car in CARS:
        for left, right, a, b in spans(base[car], new[car]):
            if right - left <= EPS:
                continue
            if ((a.kind, a.task) != (b.kind, b.task)
                    or math.dist(a.at(left), b.at(left)) > EPS
                    or math.dist(a.at(right), b.at(right)) > EPS):
                earliest = min(left, earliest)
                break
    return earliest


def verify(root):
    plans = read_tracks(root / TRACKS)
    rows = read_csv(root / SCORES)
    originals = dict(zip(TASKS, map(tuple, TARGETS)))
    original_scores = {}
    for row in read_csv(RESULT):
        key = (float(row["方位_deg"]) if row["方位_deg"] else None,
               int(row["水平距离_m"]) if row["水平距离_m"] else None,
               row["故障车辆"], int(row["中断时长_s"]))
        original_scores[key] = row

    expected = {}
    for group, angles, outages in (
        ("主分析", range(0, 360, 45), (0, 5, 15, 25, 30)),
        ("方位平移", [a + 22.5 for a in range(0, 360, 45)], (0, 5, 15, 25, 30)),
        ("恢复时刻A", range(0, 360, 45), (1, 10, 25)),
    ):
        positions = list(product(angles, (30, 40, 50)))
        if group != "方位平移":
            positions.append((None, None))
        expected[group] = {(a, r, f, d) for (a, r), f, d in product(
            positions, ("", *CARS), outages
        ) if a is not None or f or d}

    checked_plans, retasked_targets, ids = {}, {}, set()
    used = set()
    results = Counter()
    limits = [math.inf] * 3
    for row in rows:
        angle = float(row["方位_deg"]) if row["方位_deg"] else None
        radius = int(row["水平距离_m"]) if row["水平距离_m"] else None
        fault, outage = row["故障车辆"], int(row["中断时长_s"])
        group, scene = row["样本组"], row["场景ID"]
        key = angle, radius, fault, outage
        assert scene not in ids
        ids.add(scene)
        expected[group].remove(key)
        labels = ["新增"] if angle is not None else []
        if fault:
            labels.append("故障")
        if outage:
            labels.append("中断")
        assert "＋".join(labels) == row["扰动"]
        targets = dict(originals)
        if angle is not None:
            assert int(row["高度_m"]) == 50
            theta = math.radians(angle)
            targets["T4"] = (50 + radius * math.cos(theta), 50 + radius * math.sin(theta), 50)
        else:
            assert not row["高度_m"]
        assert len(targets) == int(row["N"])
        base_id, new_id = row["固定时间轴ID"], row["重调度时间轴ID"]
        used.update((base_id, new_id))
        assert base_id == ("F" + fault[1:] if fault else "F0")
        release = float(row["首个可改派时刻_s"])
        assert release == EVENT + outage
        change = difference(plans[base_id], plans[new_id])
        if row["策略"] == "沿用原表":
            assert change == math.inf and base_id == new_id
            assert not row["采用改派时刻_s"] and not row["首次执行分歧_s"]
        else:
            cut = float(row["采用改派时刻_s"])
            assert close(float(row["首次执行分歧_s"]), change)
            assert release <= cut + EPS and cut <= change + EPS
            assert (row["策略"] == "即时改派") == close(cut, release)
        paired = []
        for side, timeline in (("固定", base_id), ("重调度", new_id)):
            identity = (timeline, fault)
            if identity not in checked_plans:
                checked_plans[identity] = check_track(plans[timeline], targets, fault)
            starts, *minima = checked_plans[identity]
            if timeline.startswith("R"):
                stored_targets = retasked_targets.setdefault(timeline, targets)
                assert stored_targets == targets
            limits = [min(a, b) for a, b in zip(limits, minima)]
            k, n = len(starts), len(targets)
            last = max(starts.values()) if k == n else DEADLINE
            assert int(row[side + "k"]) == k
            assert close(float(row[side + "完成率"]), k / n)
            assert close(float(row[side + "最晚开始_s"]), last)
            for task in TASKS:
                value = row[f"{side}{task}开始_s"]
                assert close(float(value), starts[task]) if task in starts else not value
            assert "T4" not in starts or starts["T4"] >= release - EPS
            paired.append((k, -last, starts))
        old, new = paired
        if group == "主分析":
            fixed = original_scores.pop(key)
            assert int(fixed["k"]) == old[0] and close(float(fixed["最晚开始_s"]), -old[1])
        for task, start in old[2].items():
            if start < EVENT:
                assert close(new[2][task], start)
        assert new[:2] >= old[:2]
        results[(group, row["扰动"], "胜" if new[:2] > old[:2] else "平")] += 1

    assert not original_scores and all(not remaining for remaining in expected.values())
    assert used == set(plans)
    print(f"{len(rows)} paired scenarios, {len(plans)} independent timelines")
    print(f"Minimum clearances (road/building, sight/building, car pair): {limits}")
    for key, count in sorted(results.items()):
        print(*key, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT.parent)
    verify(parser.parse_args().output)
