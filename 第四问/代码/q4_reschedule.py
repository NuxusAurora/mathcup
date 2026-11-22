"""Solve and independently verify the Q4 paired scenario timelines."""
import argparse
import csv
import hashlib
from collections import Counter
from itertools import permutations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from q4_continuous import StationSearch, solve
from q4_score import DEADLINE, EVENT, full_score_lower_bound, read_plan, score_for
from q4_table5a import ATTACHMENTS, NOMINAL, reference, scenarios
from q4_timeline import (
    SCORES, TASKS, TRACKS, first_change, fixed_tracks, num, observation_starts, read_duties,
    targets_for,
)
from q4_verify import check_track, close, difference, read_csv, read_tracks

CERTIFICATE = ATTACHMENTS / "q4_nominal_certificate.json"
TRACK_FIELDS = (
    "时间轴ID", "车辆", "阶段", "告警", "开始_s", "结束_s", "起点x_m", "起点y_m",
    "终点x_m", "终点y_m", "起始方位_deg", "起始俯仰_deg", "终止方位_deg", "终止俯仰_deg",
)
SCORE_FIELDS = (
    "场景ID", "样本组", "扰动", "方位_deg", "水平距离_m", "高度_m", "故障车辆",
    "中断时长_s", "首个可改派时刻_s", "策略", "采用改派时刻_s", "首次执行分歧_s",
    "固定时间轴ID", "重调度时间轴ID", "固定k", "重调度k", "N", "固定完成率",
    "重调度完成率", "固定最晚开始_s", "重调度最晚开始_s",
    *(f"{side}{task}开始_s" for side in ("固定", "重调度") for task in TASKS),
    "满分最晚开测下界_s", "时间上下界差_s", "无门限时间轴_SHA256", "连续证书_SHA256",
)


def layout_candidates(group, targets, originals):
    layouts = set()
    # A template certified elsewhere can still be feasible at this target.
    for template in group["plans"].values():
        task_pairs = {task: pair for task, pair in originals.items() if task in targets}
        from_cert = {}
        for duty in template:
            task = duty["task"]
            if task in targets:
                from_cert.setdefault(task, []).append((duty["car"], tuple(duty["path"][-1])))
        task_pairs.update(from_cert)
        for order in permutations(sorted(task_pairs)):
            teams = tuple((task, *(car for car, _ in task_pairs[task])) for task in order)
            stations = tuple(point for task in order for _, point in task_pairs[task])
            layouts.add((teams, stations))
    return sorted(layouts)


def track_rows(name, tracks):
    for i, track in enumerate(tracks, 1):
        for p in track:
            yield (name, f"V{i}", p.kind, p.task,
                   *(num(value) for value in (p.begin, p.end, *p.p, *p.q, *p.aim0, *p.aim1)))


def generate(score_writer, track_writer, selected):
    duties = read_duties(reference(NOMINAL, "nominal"))
    originals = {}
    for duty in duties:
        originals.setdefault(duty.task, []).append((duty.car, duty.path[-1]))
    ref_hash = hashlib.sha256(NOMINAL.read_bytes()).hexdigest()
    raw_cert = CERTIFICATE.read_bytes()
    cert_hash = hashlib.sha256(raw_cert).hexdigest()
    groups = {None if g["fault"] == -1 else g["fault"]: g for g in json.loads(raw_cert)["groups"]}
    bases = {fault: fixed_tracks(duties, fault) for fault in groups}
    search = StationSearch()
    written = set()
    count = 0
    for scene, location, fault, outage in selected:
        targets = targets_for(location)
        base = bases[fault]
        old = observation_starts(base)
        original = {task: pair for task, pair in originals.items()
                    if all(car != fault for car, _ in pair) or task in old}
        release = EVENT + outage
        planned, cut = solve(search, base, fault, release, targets,
                             layout_candidates(groups[fault], targets, original), old.values())
        change = first_change(base, planned)
        new = observation_starts(planned)
        if len(new) != len(targets):
            raise ValueError(f"{scene} 仅能开测 {len(new)}/{len(targets)} 项")
        baseline_id = f"F{0 if fault is None else fault + 1}"
        if baseline_id not in written:
            track_writer.writerows(track_rows(baseline_id, base))
            written.add(baseline_id)
        plan_id = baseline_id if cut is None else f"R{scene[1:]}"
        if cut is not None:
            track_writer.writerows(track_rows(plan_id, planned))
        angle, radius = ("", "") if location is None else location
        car = "" if fault is None else f"V{fault + 1}"
        labels = "＋".join(name for name, present in (
            ("新增", location is not None), ("故障", car), ("中断", outage)) if present)
        strategy = "沿用原表" if cut is None else ("即时改派" if cut == release else "先保留原表后改派")
        k0, total, last0 = score_for(old, targets)
        k1, _, last1 = score_for(new, targets)
        lower = full_score_lower_bound(old, targets, release)
        score_writer.writerow((
            scene, "主分析", labels, angle, radius, 50 if location is not None else "", car,
            outage, release, strategy, num(cut), num(change), baseline_id, plan_id,
            k0, k1, total, num(k0 / total), num(k1 / total), num(last0), num(last1),
            *(num(starts.get(task)) for starts in (old, new) for task in TASKS),
            num(lower), num(last1 - lower), ref_hash, cert_hash,
        ))
        count += 1
        if count % 100 == 0:
            print(f"已求解 {count} 场", flush=True)


def verify_output(directory, expected):
    if not __debug__:
        raise RuntimeError("复核依赖 Python 的断言，请退出 -O 模式")
    tracks = read_tracks(directory / TRACKS)
    rows = read_csv(directory / SCORES)
    if len(rows) != len(expected):
        raise ValueError("导出场景数错误")
    ref_hash = hashlib.sha256(NOMINAL.read_bytes()).hexdigest()
    cert_hash = hashlib.sha256(CERTIFICATE.read_bytes()).hexdigest()
    original_plan = read_plan()
    checked = {}
    outcomes = Counter()
    worst = 0
    for row, (scene, location, fault, outage) in zip(rows, expected):
        targets = targets_for(location)
        assert row["场景ID"] == scene
        assert row["无门限时间轴_SHA256"] == ref_hash and row["连续证书_SHA256"] == cert_hash
        car = None if fault is None else f"V{fault + 1}"
        assert row["故障车辆"] == (car or "") and int(row["中断时长_s"]) == outage
        actual_location = (float(row["方位_deg"]), float(row["水平距离_m"])) if row["方位_deg"] else None
        assert actual_location == location
        ids = row["固定时间轴ID"], row["重调度时间轴ID"]
        assert ids[0] == ("F0" if car is None else "F" + car[1:])
        paired = []
        for side, track_id in zip(("固定", "重调度"), ids):
            if track_id not in checked:
                checked[track_id] = check_track(tracks[track_id], targets, car)[0]
            starts = checked[track_id]
            k, total, last = score_for(starts, targets)
            assert k == int(row[side + "k"]) and total == int(row["N"])
            assert close(k / total, float(row[side + "完成率"]))
            assert close(last, float(row[side + "最晚开始_s"]))
            for task in TASKS:
                field = row[f"{side}{task}开始_s"]
                assert close(starts[task], float(field)) if task in starts else not field
            paired.append((k, -last))
        expected_fixed = {task: t for task, (t, cars) in original_plan.items()
                          if t < EVENT or car not in cars}
        assert checked[ids[0]].keys() == expected_fixed.keys()
        assert all(close(checked[ids[0]][task], t) for task, t in expected_fixed.items())
        release = EVENT + outage
        assert close(release, float(row["首个可改派时刻_s"]))
        lower = full_score_lower_bound(expected_fixed, targets, release)
        assert lower <= -paired[1][1] + 1e-8
        assert close(float(row["满分最晚开测下界_s"]), lower)
        assert close(float(row["时间上下界差_s"]), -paired[1][1] - lower)
        assert checked[ids[1]].get("T4", DEADLINE) >= release - 1e-8
        for task, start in checked[ids[0]].items():
            if start < EVENT:
                assert close(checked[ids[1]][task], start)
        if ids[0] == ids[1]:
            assert not row["首次执行分歧_s"] and not row["采用改派时刻_s"]
        else:
            change = difference(tracks[ids[0]], tracks[ids[1]])
            cut = float(row["采用改派时刻_s"])
            assert release <= cut + 1e-8 <= change + 2e-8
            assert close(float(row["首次执行分歧_s"]), change)
        assert paired[1][0] == len(targets) and paired[1] >= paired[0]
        outcomes[paired[1][0] - paired[0][0] if paired[1][0] != paired[0][0]
                 else ("提速" if paired[1] > paired[0] else "平")] += 1
        worst = max(worst, -paired[1][1])
    assert set(checked) == set(tracks)
    print(f"独立复核 {len(rows)} 场、{len(tracks)} 套时间轴；最慢可行开测 {worst:.6f} s")
    print(dict(outcomes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新建的结果目录")
    parser.add_argument("--scene", action="append", help="只解指定编号，例如 N0107，可重复")
    parser.add_argument("--verify", action="store_true", help="独立复核指定结果目录")
    args = parser.parse_args()
    selected = list(scenarios())
    if args.scene:
        selected = [item for item in selected if item[0] in args.scene]
        if {row[0] for row in selected} != set(args.scene):
            parser.error("代表场景编号为 N0001 至 N0599")
    if args.verify:
        verify_output(args.output, selected)
        return
    if args.output.exists():
        parser.error("输出须为尚未存在的目录")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "verified"
        stage.mkdir()
        with ((stage / SCORES).open("w", newline="", encoding="utf-8-sig") as scores,
              (stage / TRACKS).open("w", newline="", encoding="utf-8-sig") as tracks):
            writers = [csv.writer(file, lineterminator="\n") for file in (scores, tracks)]
            for writer, fields in zip(writers, (SCORE_FIELDS, TRACK_FIELDS)):
                writer.writerow(fields)
            generate(*writers, selected)
        verify_output(stage, selected)
        stage.rename(args.output)
    print(f"已保存到 {args.output.resolve()}")


if __name__ == "__main__":
    main()
