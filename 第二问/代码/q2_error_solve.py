"""Resolve the contest's fixed scheduling instance at a chosen local RMS limit."""
import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np

from q2_schedule import (
    EDGE, OBSERVE, RADIUS, SPEED, STARTS, TARGETS, Roads, objective,
    pair_compatible, replay, road, turn_time, visible,
)
from q2_schedule_verify import check


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


def station_points(roads, spacing):
    axis = np.r_[np.arange(RADIUS, EDGE, spacing), EDGE]
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    theta = np.deg2rad(np.arange(0, 360, 10))
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    rings = []
    for x, y, height in TARGETS:
        inner = (height - 2) / math.tan(math.radians(80)) + 1e-7
        radii = inner + spacing * np.array([0, 0.5, 1, 2])
        rings.append([x, y] + (radii[:, None, None] * directions).reshape(-1, 2))
    points = np.unique(np.vstack((STARTS, grid, roads.corners, *rings)), axis=0)
    return points[np.all((RADIUS <= points) & (points <= EDGE), axis=1) & road(points, points)]


@dataclass(frozen=True)
class Task:
    sites: np.ndarray
    left: np.ndarray
    right: np.ndarray
    errors: np.ndarray
    turn: np.ndarray


@dataclass(frozen=True)
class Result:
    legs: list
    station_counts: tuple
    nodes: int
    exhausted: bool


class SearchTimeExpired(Exception):
    pass


class Instance:
    def __init__(self, spacing):
        self.roads = Roads()
        self.points = station_points(self.roads, spacing)
        self.starts = tuple(int(np.flatnonzero(np.all(self.points == p, axis=1))[0])
                            for p in STARTS)
        self.tasks = tuple(self.build_task(task) for task in range(len(TARGETS)))
        self.car_pairs = tuple(combinations(range(len(STARTS)), 2))
        # Per-instance, bounded cache. The geometry is independent of the RMS limit.
        self.travel = lru_cache(maxsize=1024)(self.travel_from)

    def travel_from(self, station):
        return self.roads.paths(self.points[station], self.points)[0] / SPEED

    def build_task(self, task):
        sites = np.flatnonzero(visible(self.points, task))
        local = self.points[sites]
        li, ri = np.triu_indices(len(sites), 1)
        usable = pair_compatible(local[li], local[ri], task)
        li, ri = li[usable], ri[usable]
        result = local_error(local[li], local[ri], TARGETS[task])
        good = np.isfinite(result)
        return Task(sites, li[good], ri[good], result[good], turn_time(local, task))

    def solve(self, rms_limit, seconds):
        eligible = []
        for task in self.tasks:
            valid = task.errors <= rms_limit
            i, j = task.left[valid], task.right[valid]
            if not len(i):
                return Result([], tuple(len(t.sites) for t in self.tasks), 0, True)
            mask = np.zeros((len(task.sites), len(task.sites)), dtype=bool)
            mask[i, j] = mask[j, i] = True
            eligible.append(mask)

        started = monotonic()
        deadline = math.inf if seconds == 0 else started + seconds
        best, winner, nodes = math.inf, [], 0

        def time_check():
            if monotonic() >= deadline:
                raise SearchTimeExpired

        def possible(job, stations, free, previous_start):
            task = self.tasks[job]
            ready = np.array([self.travel(p)[task.sites] + release + task.turn
                              for p, release in zip(stations, free)])
            choices = [np.flatnonzero(row < best + 1e-9) for row in ready]
            batches = []
            for a, b in self.car_pairs:
                i, j = choices[a], choices[b]
                if not len(i) or not len(j):
                    continue
                k, l = np.nonzero(eligible[job][np.ix_(i, j)])
                left, right = i[k], j[l]
                times = np.maximum(ready[a, left], ready[b, right])
                keep = np.isfinite(times) & (times >= previous_start - 1e-9)
                if keep.any():
                    batches.append((a, b, left[keep], right[keep], times[keep]))
            return batches

        def remaining_can_start(order, depth, stations, free):
            # Even after other visits, direct travel from the current position
            # is a lower bound by the Euclidean triangle inequality.
            if not math.isfinite(best):
                return True
            for job in order[depth:]:
                task = self.tasks[job]
                shifts = self.points[task.sites] - self.points[stations, None]
                lower = np.hypot(shifts[..., 0], shifts[..., 1]) / SPEED
                lower += np.asarray(free)[:, None] + task.turn
                choices = [np.flatnonzero(row < best + 1e-9) for row in lower]
                if not any(eligible[job][np.ix_(choices[a], choices[b])].any()
                           for a, b in self.car_pairs):
                    return False
            return True

        def visit(order, depth, stations, free, path, previous_start, width):
            nonlocal best, winner, nodes
            nodes += 1
            time_check()
            if not remaining_can_start(order, depth, stations, free):
                return
            job = order[depth]
            task = self.tasks[job]
            batches = possible(job, stations, free, previous_start)
            if not batches:
                return
            times = np.concatenate([batch[4] for batch in batches])
            ends = np.cumsum([len(batch[4]) for batch in batches])
            ids = np.argsort(times, kind="stable")
            if width:
                ids = ids[:width]
            for index in ids:
                time_check()
                start = times[index]
                if start > best + 1e-9:
                    break
                bidx = int(np.searchsorted(ends, index, side="right"))
                a, b, left, right, _ = batches[bidx]
                offset = index if bidx == 0 else index - ends[bidx - 1]
                i, j = left[offset], right[offset]
                x, y = task.sites[i], task.sites[j]
                entry = (job, [(a, self.points[x]), (b, self.points[y])])
                if depth == len(order) - 1:
                    legs = replay(self.roads, [*path, entry])
                    elapsed = objective(legs)
                    if elapsed >= best:
                        continue
                    checks = [(l.car, l.task, l.route,
                               (l.depart, l.arrive, l.ready, l.start, l.end, l.reset))
                              for l in legs]
                    try:
                        check(checks, require_all=False)
                    except AssertionError:
                        continue
                    best, winner = elapsed, legs
                    continue
                next_stations, next_free = stations.copy(), free.copy()
                next_stations[a], next_stations[b] = x, y
                next_free[a], next_free[b] = start + OBSERVE + task.turn[[i, j]]
                path.append(entry)
                visit(order, depth + 1, next_stations, next_free, path, start, width)
                path.pop()

        orders = list(permutations(range(len(self.tasks))))
        zero = [0.0] * len(self.starts)
        exhausted = False
        try:
            # Establish an upper bound quickly; the following pass has no width cap.
            for width in (3, None):
                for order in orders:
                    visit(order, 0, list(self.starts), zero, [], 0.0, width)
            exhausted = True
        except SearchTimeExpired:
            pass
        return Result(winner, tuple(len(t.sites) for t in self.tasks), nodes, exhausted)


def task_quality(legs, task):
    a, b = sorted((leg.route[-1] for leg in legs if leg.task == task), key=tuple)
    u, v = a - TARGETS[task, :2], b - TARGETS[task, :2]
    angle = math.degrees(math.atan2(abs(cross(u, v)), np.dot(u, v)))
    return float(local_error(a, b, TARGETS[task])), angle


def export(path, legs):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("车辆", "目标", "路线坐标_m", "出发_s", "到站_s", "指向完成_s",
                         "观测开始_s", "观测结束_s", "复位完成_s", "局部RMS_m", "交会角_deg"))
        for leg in sorted(legs, key=lambda l: (l.car, l.depart)):
            error, angle = task_quality(legs, leg.task)
            writer.writerow((f"V{leg.car+1}", f"T{leg.task+1}", json.dumps(leg.route.tolist()),
                             leg.depart, leg.arrive, leg.ready, leg.start, leg.end,
                             leg.reset, error, angle))


def report(instance, limit, seconds, path):
    print(f"\n最大允许局部定位 RMS {limit:g} m，正在重新选站和排程……", flush=True)
    result = instance.solve(limit, seconds)
    print("各目标候选站数 " + " / ".join(str(n) for n in result.station_counts)
          + f"，检查 {result.nodes:,} 个搜索节点。")
    if not result.legs:
        status = ("已穷尽当前离散模型，未找到达标排程。" if result.exhausted
                  else "搜索达到时限，尚未找到达标排程。")
        print(status)
        return False
    legs = result.legs
    status = "离散模型最优" if result.exhausted else "时限内找到的最快可行解，未证明离散最优"
    print(f"{status}。最晚观测开始 {objective(legs):.6f} s，"
          f"全部观测结束 {max(leg.end for leg in legs):.6f} s。")
    for task in sorted(range(len(TARGETS)), key=lambda t: min(l.start for l in legs if l.task == t)):
        pair = [leg for leg in legs if leg.task == task]
        error, angle = task_quality(legs, task)
        stations = "、".join(f"V{l.car+1}@({l.route[-1,0]:.6f},{l.route[-1,1]:.6f}) m"
                             for l in pair)
        print(f"  T{task+1}  {stations}  RMS {error:.6f} m，交会角 {angle:.3f}°，"
              f"开测 {pair[0].start:.6f} s")
    idle = sorted(set(range(len(STARTS))) - {leg.car for leg in legs})
    if idle:
        print("  全程待命 " + "、".join(f"V{car+1}" for car in idle))
    if path:
        export(path, legs)
        print(f"已导出逐车时间轴 {path.resolve()}")
    return True


def nonnegative(raw):
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("请输入有限非负数")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="按误差门限重算 2026 数维杯 B 题第二问第（2）项",
                                     epilog="--floor 进入室内测距模拟，--archive 查询已存方案。")
    parser.add_argument("limits", nargs="*", type=nonnegative, metavar="误差_m")
    parser.add_argument("--grid", type=float, default=3.0, metavar="间距_m",
                        help="候选站方格间距，默认 3 m，另加入初始点、绕障点和近目标圆环")
    parser.add_argument("--seconds", type=nonnegative, default=15, metavar="秒",
                        help="每个门限的搜索时限，默认 15 秒，0 表示不限时")
    parser.add_argument("--csv", type=Path, metavar="文件", help="为单个门限导出逐车时间轴")
    args = parser.parse_args(argv)
    if not math.isfinite(args.grid) or not 2 <= args.grid <= 20:
        parser.error("方格间距须在 2 至 20 m 之间")
    if args.csv and len(args.limits) != 1:
        parser.error("--csv 需要一个命令行误差门限")
    if args.csv and args.csv.exists():
        parser.error(f"输出文件已存在 {args.csv}")
    print(f"原题场地与五辆观测车，位置误差按 1° 独立测角误差作局部数值估计。"
          f"站位方格间距 {args.grid:g} m。", flush=True)
    instance = Instance(args.grid)
    if args.limits:
        try:
            found = [report(instance, limit, args.seconds, args.csv) for limit in args.limits]
        except OSError as exc:
            parser.exit(1, f"导出失败。{exc}\n")
        return 0 if all(found) else 1
    while True:
        try:
            raw = input("最大允许误差（m，回车退出） ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if raw.lower() in ("", "q", "quit", "exit"):
            return 0
        try:
            limit = nonnegative(raw)
        except ValueError as exc:
            print(f"输入无效。{exc}")
            continue
        report(instance, limit, args.seconds, None)


if __name__ == "__main__":
    raise SystemExit(main())
