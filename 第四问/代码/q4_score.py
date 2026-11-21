"""Replay the fixed timetable and write the scores for the question 4 scenarios."""
from collections import defaultdict
import csv
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TIMELINE = ROOT / "第二问/文档/第二问第（2）题逐车时间轴.csv"
RESULT = ROOT / "第四问/文档/固定计划逐场景成绩.csv"
EVENT, DEADLINE = 5, 60


def read_plan():
    pairs = {}
    with TIMELINE.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["方案"] == "优化":
                pairs.setdefault(row["告警"], []).append(
                    (row["车辆"], float(row["观测开始_s"]), float(row["指向完成_s"]))
                )
    if set(pairs) != {"T1", "T2", "T3"}:
        raise ValueError("基准时间轴须含优化方案的三个原告警")
    plan = {}
    for alarm, rows in pairs.items():
        vehicles = {row[0] for row in rows}
        start = rows[0][1]
        if (len(rows) != 2 or len(vehicles) != 2
                or not all(row[1] == start and 0 <= row[2] <= start for row in rows)
                or not 0 <= start <= DEADLINE):
            raise ValueError(f"{alarm} 的双车开始记录不一致、未就绪或超出基准窗口")
        plan[alarm] = (start, vehicles)
    return plan


def score(starts):
    """One entry per required alarm; None means no joint start was recorded."""
    valid = [t for t in starts if t is not None and 0 <= t <= DEADLINE]
    last = max(valid) if len(valid) == len(starts) else DEADLINE
    return len(valid), len(starts), last


def scenarios(plan):
    # A survivor follows the original clock, including reserved waiting time.
    targets = [None, *product(range(0, 360, 45), (30, 40, 50))]
    vehicles = sorted(set.union(*(pair for _, pair in plan.values())))
    if len(vehicles) != 5:
        raise ValueError("本次基准回放要求五种等概率故障身份")
    for target, fault, outage in product(targets, [None, *vehicles], (0, 5, 15, 25, 30)):
        labels = [name for name, present in (
            ("新增", target is not None), ("故障", fault is not None), ("中断", outage > 0)
        ) if present]
        if not labels:
            continue
        starts = [t if t < EVENT or fault not in pair else None for t, pair in plan.values()]
        if target is not None:
            starts.append(None)
        k, total, last = score(starts)
        angle, radius = target if target is not None else ("", "")
        yield ("＋".join(labels), angle, radius, 50 if target is not None else "",
               fault or "", outage, EVENT + outage if outage else "",
               k, total, f"{k / total:.9f}", f"{last:.9f}")


def main():
    rows = list(scenarios(read_plan()))
    with RESULT.open("w", newline="", encoding="utf-8-sig") as dest:
        writer = csv.writer(dest, lineterminator="\n")
        writer.writerow(("扰动", "方位_deg", "水平距离_m", "高度_m", "故障车辆",
                         "中断时长_s", "恢复时刻_s", "k", "N", "完成率", "最晚开始_s"))
        writer.writerows(rows)
    print(f"{len(rows)} scenarios")
    rates = defaultdict(list)
    for label, *_, completed, total, rate_text, last_start in rows:
        rates[label].append(completed / total)
    for label, values in sorted(rates.items()):
        print(f"{label}（{len(values)} 个场景） 平均完成率 {sum(values) / len(values):.2%}")
    print(f"逐场景成绩：{RESULT}")


if __name__ == "__main__":
    main()
