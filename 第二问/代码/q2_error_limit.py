"""室内地面测点离线模拟：按位置误差门限联合选择测站、测量车和任务顺序。"""
import argparse
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product

import numpy as np


# Each car stays on its own horizontal rail; distances and times are in m and s.
RAIL_Y = (0, 4, 9, 13)
STOPS_X = (2, 9, 16)
INITIAL_X = (1, 15, 4, 17)
SPEED = 1  # metres per second
SETUP = 1
READING = 4
# (true x, true y, known six-metre square's lower-left x and y).
FLOOR_MARKS = ((3.73, 3.31, 0, 0), (9.28, 10.17, 6, 7), (15.46, 5.74, 12, 3))
TRIALS = 256
SEED = 20261121


@dataclass(frozen=True)
class Mode:
    cars: tuple[int, int]
    stops: tuple[int, int]
    rms: float


@dataclass(frozen=True)
class Visit:
    task: int
    mode: Mode
    start: float


def simulate_modes(mark, rng):
    """Fit both noisy ranges on the marked floor square; truth is used only to generate/score trials."""
    x, y, left, bottom = mark
    axis = np.linspace(0, 6, 61)
    gx, gy = np.meshgrid(left + axis, bottom + axis)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    truth = np.array([x, y])
    errors = np.sum((grid - truth) ** 2, axis=1)

    # Use the same observations when a station participates in different pairs.
    stations = list(product(range(len(RAIL_Y)), STOPS_X))
    positions = np.array([(stop, RAIL_Y[car]) for car, stop in stations])
    distances = np.linalg.norm(positions - truth, axis=1)
    observations = np.maximum(0, rng.normal(distances, 0.20 + 0.035 * distances,
                                            size=(TRIALS, len(stations))))
    predicted = np.linalg.norm(positions[:, None, :] - grid, axis=2)
    pairs = [(a, b) for a, b in combinations(range(len(stations)), 2)
             if stations[a][0] != stations[b][0]]
    squared_error = np.zeros(len(pairs))
    for block in np.array_split(observations, 16):
        residuals = (predicted[:, None, :] - block.T[:, :, None]) ** 2
        for k, (a, b) in enumerate(pairs):
            loss = residuals[a] + residuals[b]
            # Score mirror-image fits with equal weight; grid ordering must
            # not decide which side of the baseline receives the estimate.
            tied = loss <= loss.min(axis=1, keepdims=True) + 1e-12
            trial_errors = np.sum(tied * errors, axis=1) / tied.sum(axis=1)
            squared_error[k] += trial_errors.sum()

    modes = []
    for (a, b), mse_sum in zip(pairs, squared_error):
        (i, xi), (j, xj) = stations[a], stations[b]
        modes.append(Mode((i, j), (xi, xj), math.sqrt(mse_sum / TRIALS)))
    return tuple(modes)


@lru_cache(maxsize=1)
def floor_experiment():
    rng = np.random.default_rng(SEED)
    return tuple(simulate_modes(mark, rng) for mark in FLOOR_MARKS)


def solve(modes_by_task, limit):
    """Exhaust the finite model, pruning only when a partial schedule is already slower."""
    eligible = [[m for m in modes if m.rms <= limit] for modes in modes_by_task]
    if not all(eligible):
        return ()
    best_score = (math.inf, math.inf)
    best_visits = ()
    jobs = range(len(eligible))

    def search(order, depth, positions, available, visits, latest, worst_error):
        nonlocal best_score, best_visits
        if depth == len(order):
            if (latest, worst_error) < best_score:
                best_score = (latest, worst_error)
                best_visits = tuple(visits)
            return
        task = order[depth]
        for mode in eligible[task]:
            a, b = mode.cars
            xa, xb = mode.stops
            start = max(available[a] + abs(positions[a] - xa) / SPEED,
                        available[b] + abs(positions[b] - xb) / SPEED) + SETUP
            next_latest = max(latest, start)
            error = max(worst_error, mode.rms)
            if (next_latest, error) >= best_score:
                continue
            next_pos, next_free = positions.copy(), available.copy()
            next_pos[a], next_pos[b] = xa, xb
            next_free[a] = next_free[b] = start + READING
            visits.append(Visit(task, mode, start))
            search(order, depth + 1, next_pos, next_free, visits, next_latest, error)
            visits.pop()

    for order in permutations(jobs):
        search(order, 0, list(INITIAL_X), [0] * len(RAIL_Y), [], 0, 0.0)
    return tuple(sorted(best_visits, key=lambda v: (v.start, v.task)))


def parse_limit(raw):
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("误差门限必须是有限的非负数")
    return value


def report(limit):
    modes = floor_experiment()
    visits = solve(modes, limit)
    print(f"\n位置 RMS 门限 {limit:g} m，每个站对用 {TRIALS} 组模拟双车测距评估。")
    if not visits:
        print("此模拟的离散站位中没有达标排程。")
        for i, choices in enumerate(modes, 1):
            best = min(m.rms for m in choices)
            if best > limit:
                print(f"  地面点 P{i} 的最佳站对 RMS 为 {best:.6f} m")
        return
    latest = max(v.start for v in visits)
    print(f"最短的最晚测量开始时间：{latest:.3f} s；全部测量完成：{latest + READING:.3f} s")
    for visit in visits:
        pair = "、".join(f"C{car+1}@({x},{RAIL_Y[car]})"
                        for car, x in zip(visit.mode.cars, visit.mode.stops))
        print(f"  P{visit.task+1}  {pair} m  RMS {visit.mode.rms:.6f} m  "
              f"开始 {visit.start:.3f} s  结束 {visit.start + READING:.3f} s")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(a in ("--archive", "--reference", "--scores") or a.startswith("--scores=")
           for a in argv):
        from q2_error_archive import main as archive_main
        return archive_main([a for a in argv if a != "--archive"])

    if "--floor" not in argv:
        if argv not in (["-h"], ["--help"]):
            raise SystemExit("连续站址求解请使用 q2_schedule_continuous.py --error 门限；室内实验使用 --floor。")

    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="--archive 可进入旧的论文时间轴查询模式。")
    parser.add_argument("limits", type=parse_limit, nargs="*", metavar="门限_m",
                        help="可指定多个误差门限。省略时进入交互模式。")
    args = parser.parse_args([a for a in argv if a != "--floor"])
    print("室内地面测量模拟 | 3 个地面点，4 辆轨道车，每车 3 个停靠站")
    if args.limits:
        for limit in args.limits:
            report(limit)
        return
    while True:
        try:
            raw = input("\n最大允许位置 RMS（m，回车退出）：").strip()
            if not raw or raw.lower() in ("q", "quit", "exit"):
                return
            report(parse_limit(raw))
        except ValueError as exc:
            print(f"输入无效：{exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == "__main__":
    raise SystemExit(main())
