"""Recompute the 599 fixed-plan scores and certificate-based completion counts."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ATTACHMENTS = ROOT / "第四问/文档/A019复核附件"
NOMINAL = ROOT / "第二问/文档/第二问第（2）题连续分支定界时间轴.csv"
LIMITED = ROOT / "第二问/文档/第二问第（2）题连续分支定界_1m时间轴.csv"
LABEL = "连续分支定界"


def read_reference(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["方案"] == LABEL]


def reference(path, mode):
    certificate = ATTACHMENTS / f"q4_{mode}_certificate.json"
    bundle = json.loads(certificate.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (bundle["mode"] != mode or ROOT / bundle["source"] != path
            or bundle["source_sha256"] != digest):
        raise ValueError(f"证书与输入不符：{path}")
    duties = read_reference(path)
    identity = {(row["车辆"], row["告警"]): row for row in duties}
    if len(identity) != 6 or len(duties) != 6 or len(bundle["reference"]) != 6:
        raise ValueError("初始时间轴须含六个车辆任务")
    for duty in bundle["reference"]:
        car = f"V{duty['car'] + 1}"
        row = identity[car, duty["task"]]
        if (abs(float(row["观测开始_s"]) - duty["start"]) > 1e-8
                or json.loads(row["路径坐标_m"]) != duty["path"]):
            raise ValueError(f"{car}/{duty['task']} 的时间轴和证书基准不一致")
    print(mode, path.name, LABEL, "SHA-256", digest)
    return duties


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="逐场输出固定成绩与证书完成数")
    args = parser.parse_args()
    if args.output and args.output.suffix.lower() != ".csv":
        parser.error("逐场成绩须输出为 CSV 文件")
    if args.output and args.output.resolve() in {NOMINAL.resolve(), LIMITED.resolve()}:
        parser.error("成绩输出路径与初始时间轴冲突")
    duties = reference(NOMINAL, "nominal")
    reference(LIMITED, "1m")
    for mode in ("nominal", "1m"):
        subprocess.run([sys.executable, "-B", "-X", "utf8", str(ATTACHMENTS / "verify_q4.py"),
                        str(ATTACHMENTS / f"q4_{mode}_certificate.json")], check=True)
    locations = [None, *product(range(0, 360, 45), (30, 40, 50))]
    faults = [None, *(f"V{i}" for i in range(1, 6))]
    groups = defaultdict(Counter)
    results = []
    digest = hashlib.sha256(NOMINAL.read_bytes()).hexdigest()
    latest = max(float(row["观测开始_s"]) for row in duties)
    for location, fault, outage in product(locations, faults, (0, 5, 15, 25)):
        flags = (location is not None, fault is not None, outage > 0)
        if not any(flags):
            continue
        lost = {r["告警"] for r in duties
                if r["车辆"] == fault and float(r["观测开始_s"]) >= 5}
        missing = len(lost) + int(location is not None)
        groups[flags][missing] += 1
        n = 3 + int(location is not None)
        results.append((f"N{len(results)+1:04d}", *(location or ("", "")), fault or "",
                        outage, n, n - missing, 60 if missing else latest, n,
                        "证书保证满分" if missing else "沿用原表", digest))
    counts = sum(groups.values(), Counter())
    if sum(counts.values()) != 599:
        raise ValueError("场景数不符")
    print("新增 故障 中断 | 场景数 | 原表漏项数为 0 / 1 / 2 的场数")
    for flags, group in sorted(groups.items()):
        print(*(int(x) for x in flags), "|", sum(group.values()), "|",
              group[0], group[1], group[2])
    print("全部场景", sum(counts.values()), counts[0], counts[1], counts[2])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temp:
            staged = Path(temp) / "scores.csv"
            with staged.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(("场景ID", "方位_deg", "距离_m", "故障车", "中断时长_s",
                                 "总告警数", "固定开测数", "固定最晚评分_s", "保证开测数",
                                 "计分依据", "无门限时间轴_SHA256"))
                writer.writerows(results)
            staged.replace(args.output)


if __name__ == "__main__":
    main()
