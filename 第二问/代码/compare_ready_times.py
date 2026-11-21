"""Compare two supplied sets of readiness timestamps. All times are in seconds."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path


def read_starts(path, plan_names):
    pairs = {plan: {} for plan in plan_names}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"order", "task", "resource", "ready_s"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("CSV 缺少必需列：" + ", ".join(sorted(required)))
        for row in reader:
            plan = (row["order"] or "").strip()
            if plan not in pairs:
                continue
            task = (row["task"] or "").strip()
            resource = (row["resource"] or "").strip()
            try:
                time = Decimal((row["ready_s"] or "").strip())
            except InvalidOperation:
                raise ValueError(f"第 {reader.line_num} 行就绪时刻无效") from None
            if not task or not resource or not time.is_finite() or time < 0:
                raise ValueError(f"第 {reader.line_num} 行任务、资源或就绪时刻无效")
            members = pairs[plan].setdefault(task, {})
            if resource in members:
                raise ValueError(f"{plan}/{task} 的资源重复：{resource}")
            members[resource] = time

    starts = {}
    for plan, tasks in pairs.items():
        if not tasks:
            raise ValueError(f"缺少方案 {plan}")
        starts[plan] = {}
        for task, members in tasks.items():
            if len(members) != 2:
                raise ValueError(f"{plan}/{task} 需要两个不同资源的就绪记录")
            starts[plan][task] = max(members.values())
    return starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    if args.baseline == args.candidate or len(set(args.tasks)) != len(args.tasks):
        parser.error("两个方案名须不同，任务名须唯一")
    try:
        starts = read_starts(args.csv, [args.baseline, args.candidate])
        for name, tasks in starts.items():
            if set(tasks) != set(args.tasks):
                raise ValueError(f"方案 {name} 的任务集合与 --tasks 不一致")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")

    baseline, candidate = starts[args.baseline], starts[args.candidate]
    print("任务\t基准开始(s)\t候选开始(s)")
    for task in args.tasks:
        print(f"{task}\t{baseline[task]}\t{candidate[task]}")
    for name, tasks in starts.items():
        latest = max(tasks.values())
        critical = "、".join(t for t in args.tasks if tasks[t] == latest)
        print(f"{name} 最晚开始 {latest} s（{critical}）")
    before, after = max(baseline.values()), max(candidate.values())
    print(f"提前量 {before-after} s")
    if before > 0:
        print(f"改善比例 {(before-after)/before*100:.2f}%")


if __name__ == "__main__":
    main()
