"""Search and export paired, causal replays for question 4."""
import argparse
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from q4_score import DEADLINE, EVENT, RESULT, score
from q4_timeline import (
    SCORES, TASKS, TRACKS, Duty, first_change, fixed_tracks, observation_starts, read_duties,
    snapshot, tracks_for, trajectory_clear,
)
from q2_schedule import BUILDINGS, EDGE, OBSERVE, RADIUS, SPEED, STARTS, TARGETS, TURN, Roads, hit, road

BEAM = 12
STATIONS_PER_CAR = 7


def target_at(angle, radius):
    radians = math.radians(angle)
    return (50 + radius * math.cos(radians), 50 + radius * math.sin(radians), 50.0)


@dataclass
class State:
    cars: dict
    starts: dict
    duties: tuple


class StationSearch:
    """Target-keyed visibility and route costs over a finite common station set."""

    def __init__(self, duties, original_tracks, cuts):
        self.roads = Roads()
        axis = np.arange(RADIUS, EDGE + 1e-8, 2.0)
        grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        actual = {point for duty in duties for point in duty.path}
        for cut in cuts:
            for track in original_tracks:
                phase = next(phase for phase in track if phase.begin <= cut < phase.end)
                actual.add(phase.state(cut)[:2])
        points = np.unique(np.vstack((grid, STARTS, self.roads.corners, sorted(actual))), axis=0)
        self.points = points[road(points, points)]

    @lru_cache(maxsize=3072)
    def routes(self, point):
        return self.roads.paths(np.array(point), self.points)

    @lru_cache(maxsize=64)
    def geometry(self, target):
        dx, dy = (np.array(target[:2]) - self.points).T
        yaw = np.degrees(np.arctan2(dy, dx))
        pitch = np.degrees(np.arctan2(target[2] - 2, np.hypot(dx, dy)))
        clear = pitch <= 80
        for *rect, roof in BUILDINGS:
            clear &= ~hit(self.points, target[:2], rect, (roof - 2) / (target[2] - 2))
        return np.stack((yaw, pitch), axis=-1), clear

    @lru_cache(maxsize=32768)
    def choices(self, point, target):
        distance, via, previous = self.routes(point)
        aims, clear = self.geometry(target)
        turn = np.max(np.abs(aims), axis=1) / TURN
        cost = np.where(clear, distance / SPEED + turn, np.inf)
        selected = []
        for i in np.argsort(cost, kind="stable"):
            if not np.isfinite(cost[i]):
                break
            end = tuple(self.points[i])
            if any(math.dist(end, option[0][-1]) < 3.5 for option in selected):
                continue
            interior = []
            corner = via[i]
            while corner >= 0:
                interior.append(tuple(self.roads.corners[corner]))
                corner = previous[corner]
            selected.append(((point, *reversed(interior), end), tuple(aims[i]),
                             float(distance[i] / SPEED), float(turn[i])))
            if len(selected) == STATIONS_PER_CAR:
                break
        return tuple(selected)

    def assignments(self, state, task, target):
        options = {car: (clock, self.choices(point, target))
                   for car, (point, clock) in state.cars.items()}
        for a, b in combinations(options, 2):
            time_a, opts_a = options[a]
            time_b, opts_b = options[b]
            feasible = []
            for i, (path_a, _, drive_a, turn_a) in enumerate(opts_a):
                ready_a = time_a + drive_a + turn_a
                x, y = path_a[-1]
                for j, (path_b, _, drive_b, turn_b) in enumerate(opts_b):
                    u, v = path_b[-1]
                    if (math.hypot(x - u, y - v) < 2 * RADIUS
                            or abs((x - target[0]) * (v - target[1])
                                   - (y - target[1]) * (u - target[0])) <= 1e-7):
                        continue
                    ready_b = time_b + drive_b + turn_b
                    start = max(ready_a, ready_b)
                    if start <= DEADLINE:
                        feasible.append((start, drive_a + turn_a + drive_b + turn_b, i, j))
            for start, _, i, j in sorted(feasible)[:2]:
                cars, starts = dict(state.cars), dict(state.starts)
                duties = []
                for car, depart, (path, aim, drive, turn) in (
                    (a, time_a, opts_a[i]), (b, time_b, opts_b[j])
                ):
                    arrive = depart + drive
                    finish = start + OBSERVE
                    reset = finish + turn
                    cars[car] = (path[-1], reset)
                    duties.append(Duty(car, task, path, aim, depart, arrive,
                                       arrive + turn, start, finish, reset))
                starts[task] = start
                yield State(cars, starts, state.duties + tuple(duties))


def policy(geometry, base, fault, targets, cuts):
    old_starts = observation_starts(base)
    best = score([old_starts.get(task) for task in targets])
    best_tracks, best_cut = base, None

    def key(metrics):
        k, _, last = metrics
        return -k, last

    def rank(state):
        return (max(state.starts.values()), sum(state.starts.values()),
                sum(clock for _, clock in state.cars.values()))

    for cut in cuts:
        if cut >= best[2] and best[0] == best[1]:
            continue
        past, cars, starts = snapshot(base, fault, cut)
        if len(starts) == len(targets):
            continue
        frontier = [State(cars, starts, ())]
        for _ in range(len(targets) - len(starts)):
            candidates = (next_state for state in frontier for task, target in targets.items()
                          if task not in state.starts
                          for next_state in geometry.assignments(state, task, target))
            frontier = sorted(candidates, key=rank)[:BEAM]
            for state in frontier:
                metrics = score([state.starts.get(task) for task in targets])
                if key(metrics) >= key(best):
                    continue
                tracks = tracks_for(past, state.duties, fault)
                if trajectory_clear(tracks, fault):
                    best, best_tracks, best_cut = metrics, tracks, cut
    return best_tracks, best_cut


def scenario_grid():
    main = list(product(range(0, 360, 45), (30, 40, 50)))
    shifted = list(product((a + 22.5 for a in range(0, 360, 45)), (30, 40, 50)))
    faults = (None, *range(5))
    for group, targets, outages in (
        ("主分析", [None, *main], (0, 5, 15, 25, 30)),
        ("方位平移", shifted, (0, 5, 15, 25, 30)),
        ("恢复时刻A", [None, *main], (1, 10, 25)),
    ):
        for target, fault, outage in product(targets, faults, outages):
            if target is not None or fault is not None or outage:
                yield group, target, fault, outage


def format_time(value):
    return "" if value is None else f"{value:.12f}".rstrip("0").rstrip(".")


def generate(score_file, track_file, sample=False):
    duties = read_duties()
    original = fixed_tracks(duties, None)
    locked_times = sorted({duty.start for duty in duties if duty.start > EVENT})
    geometry = StationSearch(duties, original, [EVENT, 6, 10, 15, 20, 30, 35, *locked_times])
    bases = {fault: fixed_tracks(duties, fault) for fault in (None, *range(5))}
    fixed = {fault: f"F{0 if fault is None else fault + 1}" for fault in bases}
    score_writer = csv.writer(score_file, lineterminator="\n")
    track_writer = csv.writer(track_file, lineterminator="\n")
    score_writer.writerow((
        "场景ID", "样本组", "扰动", "方位_deg", "水平距离_m", "高度_m", "故障车辆",
        "中断时长_s", "首个可改派时刻_s", "策略", "采用改派时刻_s", "首次执行分歧_s",
        "固定时间轴ID", "重调度时间轴ID", "固定k", "重调度k", "N", "固定完成率",
        "重调度完成率", "固定最晚开始_s", "重调度最晚开始_s",
        *(f"{side}{task}开始_s" for side in ("固定", "重调度") for task in TASKS),
    ))
    track_writer.writerow((
        "时间轴ID", "车辆", "阶段", "告警", "开始_s", "结束_s", "起点x_m", "起点y_m",
        "终点x_m", "终点y_m", "起始方位_deg", "起始俯仰_deg", "终止方位_deg", "终止俯仰_deg",
    ))

    def write_track(timeline_id, tracks):
        for car, track in enumerate(tracks, 1):
            for p in track:
                values = (p.begin, p.end, *p.p, *p.q, *p.aim0, *p.aim1)
                track_writer.writerow((timeline_id, f"V{car}", p.kind, p.task,
                                       *(format_time(value) for value in values)))

    for fault, base in bases.items():
        write_track(fixed[fault], base)

    @lru_cache(maxsize=None)
    def solve(target, fault, outage):
        release = EVENT + outage
        targets = dict(zip(TASKS, map(tuple, TARGETS)))
        if target is not None:
            targets["T4"] = target_at(*target)
        base = bases[fault]
        # These checkpoints merely keep already issued instructions longer.
        cuts = [release, *(t for t in locked_times if t > release)]
        planned, cut = policy(geometry, base, fault, targets, cuts)
        return planned, cut, first_change(base, planned)

    ids = {}
    count = 0
    for group, target, fault, outage in scenario_grid():
        if sample and not (group == "主分析" and target in (None, (0, 30), (225, 50))):
            continue
        count += 1
        scene = f"S{count:04d}"
        planned, cut, change = solve(target, fault, outage)
        base = bases[fault]
        old, new = observation_starts(base), observation_starts(planned)
        tasks = TASKS if target is not None else TASKS[:3]
        k0, total, last0 = score([old.get(task) for task in tasks])
        k1, _, last1 = score([new.get(task) for task in tasks])
        identity = (target, fault, outage)
        if cut is None:
            timeline_id = fixed[fault]
        elif identity in ids:
            timeline_id = ids[identity]
        else:
            timeline_id = ids[identity] = f"R{count:04d}"
            write_track(timeline_id, planned)
        event_labels = [tag for tag, present in (
            ("新增", target is not None), ("故障", fault is not None), ("中断", bool(outage))
        ) if present]
        angle, radius = target if target is not None else ("", "")
        strategy = "沿用原表" if cut is None else ("即时改派" if cut == EVENT + outage else "先保留原表后改派")
        score_writer.writerow((
            scene, group, "＋".join(event_labels), angle, radius, 50 if target is not None else "",
            f"V{fault + 1}" if fault is not None else "", outage, EVENT + outage, strategy,
            format_time(cut), format_time(change), fixed[fault], timeline_id,
            k0, k1, total, f"{k0 / total:.9f}", f"{k1 / total:.9f}",
            format_time(last0), format_time(last1),
            *(format_time(starts.get(task)) for starts in (old, new) for task in TASKS),
        ))
        if count % 100 == 0:
            print(f"{count} scenarios", flush=True)
    print(f"{count} scenarios, {len(bases) + len(ids)} distinct timelines")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT.parent)
    parser.add_argument("--sample", action="store_true", help="Run a small diagnostic subset")
    args = parser.parse_args()
    if args.sample and args.output.resolve() == RESULT.parent.resolve():
        parser.error("--sample requires a separate --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as temp:
        staged = Path(temp)
        with ((staged / SCORES).open("w", newline="", encoding="utf-8-sig") as scores,
              (staged / TRACKS).open("w", newline="", encoding="utf-8-sig") as tracks):
            generate(scores, tracks, args.sample)
        for name in (SCORES, TRACKS):
            (staged / name).replace(args.output / name)
    for name in (SCORES, TRACKS):
        print(f"已保存：{(args.output / name).resolve()}")


if __name__ == "__main__":
    main()
