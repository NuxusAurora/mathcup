"""Compare complete offline Q2(2) candidate schedules at one or more RMS limits."""
import argparse
import csv
import math
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path

from q2_schedule import OBSERVE, SPEED, STARTS, TARGETS, TURN
from q2_schedule_verify import check, read_rows


STD_RAD = math.radians(1)
# Elevation <= 80 degrees gives a lower bound on both horizontal ranges.
ERROR_FLOORS = tuple(STD_RAD * math.sqrt(2) * (h - 2) / math.tan(math.radians(80))
                     for _, _, h in TARGETS)
# The comparison manuscript reports constraint levels and rounded times.
# Individual unrounded errors and complete timelines were not supplied.
REFERENCE = ((0.5, 35.17), (0.75, 30.31), (1.0, 25.43))
# Certified lower endpoint for the same scene without the added accuracy constraint.
UNCONSTRAINED_LOWER_S = 10.352118500


@dataclass(frozen=True)
class Candidate:
    name: str
    errors: tuple[float, ...]
    starts: tuple[float, ...]

    @property
    def required_limit(self):
        return max(self.errors)

    @property
    def latest(self):
        return max(self.starts)


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("请输入有限的非负数")
    return value


def archive_candidates():
    result = []
    for name, legs in read_rows().items():
        check(legs)
        errors, starts = [], []
        for task, target in enumerate(TARGETS):
            observations = [leg for leg in legs if leg[1] == task]
            a, b = [target[:2] - leg[2][-1] for leg in observations]
            r1, r2 = math.hypot(*a), math.hypot(*b)
            sine = abs(a[0] * b[1] - a[1] * b[0]) / (r1 * r2)
            errors.append(STD_RAD * math.hypot(r1, r2) / sine)
            starts.append(max(leg[3][3] for leg in observations))
        result.append(Candidate(name, tuple(errors), tuple(starts)))
    return result


def read_scores(path):
    """Each row is one complete schedule, never three independently selected tasks."""
    error_fields = [f"T{i}误差_m" for i in (1, 2, 3)]
    start_fields = [f"T{i}开始_s" for i in (1, 2, 3)]
    fields = ["方案", *error_fields, *start_fields]
    result = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(fields) <= set(reader.fieldnames or []):
            raise ValueError(f"{path} 的 CSV 必须含有列 {','.join(fields)}")
        for row in reader:
            try:
                name = (row["方案"] or "").strip()
                if not name:
                    raise ValueError("方案名为空")
                errors = tuple(nonnegative(row[field]) for field in error_fields)
                starts = tuple(nonnegative(row[field]) for field in start_fields)
                if any(err < floor for err, floor in zip(errors, ERROR_FLOORS)):
                    raise ValueError("目标误差低于俯仰角限制给出的理论下界，请检查单位和原始精度")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} 第 {reader.line_num} 行数据无效：{exc}") from exc
            result.append(Candidate(f"{path.name} / {name}", errors, starts))
    if not result:
        raise ValueError(f"{path} 没有方案记录")
    return result


def best_candidate(candidates, limit):
    return min((c for c in candidates if c.required_limit <= limit),
               key=lambda c: (c.latest, c.required_limit, c.name), default=None)


def time_lower_bound(limit):
    """Solve a relaxed makespan problem; only its objective bounds the original."""
    ranges, slews = [], []
    for _, _, height in TARGETS:
        # E >= sigma*hypot(r1,r2) and both ranges >= r_min.
        min_range = (height - 2) / math.tan(math.radians(80))
        scale = limit / STD_RAD
        max_range = scale * math.sqrt(1 - (min_range / scale) ** 2)
        ranges.append(max_range)
        slews.append(math.degrees(math.atan2(height - 2, max_range)) / TURN)

    jobs = range(len(TARGETS))
    initial = [[max(0.0, math.dist(p, TARGETS[j, :2]) - ranges[j]) / SPEED + slews[j]
                for j in jobs] for p in STARTS]
    transfer = [[max(0.0, math.dist(TARGETS[i, :2], TARGETS[j, :2]) - ranges[i] - ranges[j])
                 / SPEED + slews[j] for j in jobs] for i in jobs]
    pairs = list(combinations(range(len(STARTS)), 2))
    best = math.inf
    for order in permutations(jobs):
        for assignment in product(pairs, repeat=len(order)):
            previous = [None] * len(STARTS)
            available = [0.0] * len(STARTS)
            latest = 0.0
            for job, pair in zip(order, assignment):
                start = max(initial[car][job] if previous[car] is None else
                            available[car] + transfer[previous[car]][job] for car in pair)
                latest = max(start, latest)
                if latest >= best:
                    break
                for car in pair:
                    available[car] = start + OBSERVE + slews[job]
                    previous[car] = job
            else:
                best = latest
    return max(best, UNCONSTRAINED_LOWER_S)


def report(candidates, limit, reference):
    print(f"\n误差门限 {limit:g} m（各目标局部水平定位均方根）")
    if limit < max(ERROR_FLOORS):
        i = ERROR_FLOORS.index(max(ERROR_FLOORS))
        print(f"按题设几何，T{i+1} 的误差至少为 {ERROR_FLOORS[i]:.9f} m，该门限不可行。")
        return
    print(f"连续模型的最晚开始时刻下界：{time_lower_bound(limit):.9f} s")
    winner = best_candidate(candidates, limit)
    if winner:
        print(f"候选集内最短的最晚观测开始时刻：{winner.latest:.9f} s（{winner.name}）")
        for task, (err, start) in enumerate(zip(winner.errors, winner.starts), 1):
            print(f"  T{task}  误差 {err:.8f} m，观测开始 {start:.9f} s")
    else:
        print("当前候选集中没有三个目标均达标的方案。")

    if reference:
        reported = [t for bound, t in REFERENCE if bound <= limit]
        if reported:
            print(f"对照稿另报达标方案约 {min(reported):.2f} s（仅汇总数据，未复核）。")
        else:
            print("对照稿没有报告此门限下的达标方案。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("limits", metavar="误差_m", type=nonnegative, nargs="*",
                        help="共同误差上限，单位 m。留空可反复输入。")
    parser.add_argument("--scores", action="append", type=Path, default=[],
                        help="追加同一题设下的完整方案成绩 CSV，可重复指定。")
    parser.add_argument("--reference", action="store_true",
                        help="另列对照稿的取整结果，单独显示。")
    args = parser.parse_args()
    try:
        candidates = archive_candidates()
        for path in args.scores:
            candidates.extend(read_scores(path))
    except (OSError, ValueError, AssertionError) as exc:
        parser.exit(1, f"读取或复核方案失败：{exc or '时间轴未通过可行性检查'}\n")
    print(f"已载入 {len(candidates)} 套完整方案。按三个目标共同达标筛选，再比较最晚开始时刻。")
    if args.limits:
        for limit in args.limits:
            report(candidates, limit, args.reference)
        return
    while True:
        try:
            raw = input("\n最大允许误差（m，回车退出）：").strip()
            if not raw or raw.lower() in ("q", "quit", "exit"):
                return
            report(candidates, nonnegative(raw), args.reference)
        except ValueError as exc:
            print(f"输入无效：{exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            return


if __name__ == "__main__":
    main()
