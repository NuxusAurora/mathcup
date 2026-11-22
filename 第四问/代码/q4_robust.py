"""Continuous rescheduling with the Plan B horizontal waypoint error."""
import argparse
import csv
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, pairwise, permutations, product
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize_scalar

from q4_continuous import BUILDING_GEOMETRY, intersects
from q4_score import full_score_lower_bound, score_for
from q4_table5a import ATTACHMENTS, scenarios
from q4_timeline import DEADLINE, EVENT, RESERVED, TASKS, num, targets_for
from q4_verify import check_view, road_clearance
from q2_schedule import EDGE, OBSERVE, RADIUS, SPEED, STARTS, Roads
from q2_schedule_robust import (
    B_PATH, OFFSET, RMS_LIMIT, audit, rms_bound_at, segment_separation, station_bounds_at,
)
from q2_schedule_verify import read_rows

TEMPLATES = ATTACHMENTS / "q4_1m_certificate.json"
RESULTS = "B重调度逐场景对照.csv"
DUTIES = "B保证逐车时间轴.csv"
PADDING = RADIUS + OFFSET + 0.001
OBSERVE_FIXED = "原表观测"
OBSERVE_NEW = "改派观测"
EPS = 1e-8

FIELDS = (
    "场景ID", "方位_deg", "距离_m", "故障车", "中断时长_s", "固定时间轴ID", "重调度时间轴ID",
    "最早可改派_s", "安全改派_s", "总告警数", "固定开测数", "重调度开测数",
    "固定最晚评分_s", "重调度最晚开测上界_s", "开测下界_s", "重调度最大RMS上界_m",
    *(f"{side}{task}开测_s" for side in ("固定", "重调度") for task in TASKS),
    "B时间轴_SHA256", "初始化模板_SHA256",
)
DUTY_FIELDS = ("时间轴ID", "车辆", "告警", "状态", "路线_m", "出发_s", "到站上界_s",
               "就绪上界_s", "开测_s", "结束_s", "复位上界_s")


@dataclass(frozen=True)
class Duty:
    car: int
    task: str
    path: tuple
    depart: float
    arrive: float
    ready: float
    start: float
    end: float
    reset: float
    status: str = OBSERVE_NEW

    def slots(self):
        clock = self.depart
        for a, b in pairwise(self.path):
            end = clock + drive(a, b)
            if end > clock:
                yield clock, end, a, b
            clock = end
        p = self.path[-1]
        yield clock, self.reset, p, p


def drive(a, b):
    return (math.dist(a, b) + 2 * OFFSET) / SPEED if a != b else 0.0


def pair_duties(task, launches, target, status=OBSERVE_NEW):
    timings = []
    for car, depart, path in launches:
        arrive = depart + sum(drive(a, b) for a, b in pairwise(path))
        _, turn = station_bounds_at(path[-1], target)
        timings.append((car, path, depart, arrive, turn))
    at = max(arrive + turn for _, _, _, arrive, turn in timings)
    return tuple(Duty(car, task, path, depart, arrive, arrive + turn, at, at + OBSERVE,
                      at + OBSERVE + turn, status)
                 for car, path, depart, arrive, turn in timings)


def initial_duties():
    plan = read_rows(B_PATH)["B实际方案"]
    _, upper, _, _, _ = audit(plan)
    clocks = [0.0] * len(STARTS)
    duties = []
    targets = targets_for(None)
    for task, target in targets.items():
        legs = [(car, clocks[car], tuple(map(tuple, route))) for car, t, route, _ in plan
                if t == int(task[1:]) - 1]
        pair = pair_duties(task, legs, target, OBSERVE_FIXED)
        assert math.isclose(upper[int(task[1:]) - 1], pair[0].start, abs_tol=EPS)
        for d in pair:
            clocks[d.car] = d.reset
        duties.extend(pair)
    return tuple(duties)


def fixed_duties(base, fault):
    failed_tasks = {d.task for d in base if d.car == fault and d.start >= EVENT}
    return tuple(replace(d, status=RESERVED) if d.task in failed_tasks else d
                 for d in base if d.car != fault)


def starts_of(duties):
    return {d.task: d.start for d in duties if d.status != RESERVED}


def release_states(duties, release):
    return {d.car: (d.path[-1], max(release, d.reset)) for d in duties}


@lru_cache(maxsize=32768)
def robust_station(point, target):
    if not all(RADIUS + OFFSET <= value <= EDGE - OFFSET for value in point):
        return False
    if math.dist(point, target[:2]) <= OFFSET:
        return False
    pitch, _ = station_bounds_at(point, target)
    if pitch > 80 - EPS:
        return False
    return not any(intersects(point, target[:2], (l - OFFSET - EPS, r + OFFSET + EPS,
                                                  d - OFFSET - EPS, u + OFFSET + EPS),
                              (roof - 2 + OFFSET + EPS) / (target[2] - 2))
                   for l, r, d, u, roof in BUILDING_GEOMETRY)


@lru_cache(maxsize=8192)
def robust_pair(a, b, target):
    return (math.dist(a, b) >= 2 * (RADIUS + OFFSET) + EPS
            and robust_station(a, target) and robust_station(b, target)
            and rms_bound_at(a, b, target) <= RMS_LIMIT - EPS)


@lru_cache(maxsize=65536)
def separated(a, b, c, d):
    limit = 2 * (RADIUS + OFFSET)
    for i in (0, 1):
        if max(a[i], b[i]) + limit + EPS < min(c[i], d[i]):
            return True
        if max(c[i], d[i]) + limit + EPS < min(a[i], b[i]):
            return True
    return segment_separation(*(np.asarray(point) for point in (a, b, c, d))) > limit + EPS


def spacing(duties, cars):
    tracks = {car: [] for car in cars}
    for car in cars:
        time, origin = 0.0, tuple(STARTS[car])
        for duty in (d for d in duties if d.car == car):
            if duty.depart > time:
                tracks[car].append((time, duty.depart, origin, origin))
            tracks[car].extend(duty.slots())
            time, origin = duty.reset, duty.path[-1]
        tracks[car].append((time, math.inf, origin, origin))
    for first, second in combinations(tracks.values(), 2):
        i = j = 0
        while i < len(first) and j < len(second):
            s, t, a, b = first[i]
            u, v, c, d = second[j]
            if max(s, u) < min(t, v) and not separated(a, b, c, d):
                return False
            i, j = i + (t <= v), j + (v <= t)
    return True


class Search:
    def __init__(self):
        self.roads = Roads(PADDING)
        self.route = lru_cache(maxsize=16384)(self._route)

    def _route(self, a, b):
        if not any(intersects(a, b, (l - PADDING, r + PADDING, d - PADDING, u + PADDING))
                   for l, r, d, u, _ in BUILDING_GEOMETRY):
            return a, b
        path = tuple(map(tuple, self.roads.route(np.asarray(a), np.asarray(b))))
        if all(road_clearance(p, q) > RADIUS + OFFSET + EPS for p, q in pairwise(path)):
            return path

    def add(self, task, pair, targets, states):
        target = targets[task]
        if not robust_pair(pair[0][1], pair[1][1], target):
            return None
        launches = []
        for car, point in pair:
            a, depart = states[car]
            path = self.route(a, point)
            if path is None:
                return None
            launches.append((car, depart, path))
        duties = pair_duties(task, launches, target)
        if duties[0].start <= DEADLINE:
            return duties

    def initial(self, pending, pool, initial, prefix, targets, diverse=False):
        choices = {}
        for task in pending:
            found = [(pair, plan) for pair in pool[task]
                     if (plan := self.add(task, pair, targets, initial))]
            found.sort(key=lambda x: x[1][0].start)
            if diverse:
                by_team = {}
                for pair, _ in found:
                    team = tuple(sorted(car for car, _ in pair))
                    options = by_team.setdefault(team, [])
                    if len(options) < 6:
                        options.append(pair)
                choices[task] = [pair for options in by_team.values() for pair in options]
            else:
                choices[task] = [pair for pair, _ in found[:24]]
        best = (None, ())
        for order in permutations(pending):
            frontier = [((), initial)]
            for task in order:
                extended = []
                for duties, state in frontier:
                    for pair in choices[task]:
                        plan = self.add(task, pair, targets, state)
                        if plan and spacing(prefix + duties + plan, initial):
                            next_state = state | {d.car: (d.path[-1], d.reset) for d in plan}
                            extended.append((duties + plan, next_state))
                if not extended:
                    break
                extended.sort(key=lambda x: max(d.start for d in x[0]))
                frontier = extended[:12 if diverse else 8]
                candidate = frontier[0][0]
                key = (-len(candidate), max(d.start for d in candidate))
                if best[0] is None or key < best[0]:
                    best = key, candidate
        return best[1]

    def reassign_pairs(self, pool, initial, targets):
        choices = {}
        for task, pairs in pool.items():
            target = targets[task]
            stations = {tuple(sorted(point for _, point in pair)) for pair in pairs}
            choices[task] = [((a, p), (b, q))
                             for p, q in sorted(stations) if robust_pair(p, q, target)
                             for a, b in permutations(initial, 2)
                             if max(initial[car][1] + math.dist(initial[car][0], point) / SPEED
                                    for car, point in ((a, p), (b, q))) <= DEADLINE]
        return choices

    def refine(self, duties, initial, prefix, targets):
        if not duties:
            return duties
        points = [d.path[-1] for d in duties]
        pairs = [(d.task, d.car, duties[i + 1].car) for i, d in enumerate(duties)
                 if i % 2 == 0]
        best = max(d.start for d in duties)

        def candidate(stations):
            state = initial.copy()
            new_duties = ()
            for i, (task, a, b) in enumerate(pairs):
                plan = self.add(task, ((a, stations[2*i]), (b, stations[2*i+1])), targets, state)
                if not plan:
                    return ()
                new_duties += plan
                state.update({d.car: (d.path[-1], d.reset) for d in plan})
            return new_duties if spacing(prefix + new_duties, initial) else ()

        for i in range(len(points)):
            for axis, direction in product(range(2), (-1, 1)):
                def trial(step):
                    moved = list(points)
                    point = list(points[i])
                    point[axis] += direction * step
                    moved[i] = tuple(point)
                    return moved, candidate(moved)

                def objective(step):
                    _, plan = trial(step)
                    return max(d.start for d in plan) if plan else 1e6

                step = minimize_scalar(objective, bounds=(0, 2), method="bounded",
                                       options={"maxiter": 11, "xatol": 0.015}).x
                moved, plan = trial(step)
                if plan and max(d.start for d in plan) < best - EPS:
                    duties, points, best = plan, moved, max(d.start for d in plan)
        return duties


def station_pool(raw, base):
    pool = {task: set() for task in TASKS}
    for group in raw["groups"]:
        for plan in group["plans"].values():
            rows = {}
            for item in plan:
                rows.setdefault(item["task"], []).append((item["car"], tuple(item["path"][-1])))
            for task, pair in rows.items():
                pool[task].add(tuple(sorted(pair)))
    for task in TASKS[:-1]:
        pool[task].add(tuple(sorted((d.car, d.path[-1]) for d in base if d.task == task)))
    return {task: sorted(pairs) for task, pairs in pool.items()}


def rows_for(key, duties):
    for d in duties:
        yield (key, f"V{d.car + 1}", d.task, d.status,
               json.dumps(d.path, separators=(",", ":")),
               *(num(t) for t in (d.depart, d.arrive, d.ready, d.start, d.end, d.reset)))


def runs(selected, score_writer, duty_writer):
    raw = TEMPLATES.read_bytes()
    template_hash = hashlib.sha256(raw).hexdigest()
    input_hash = hashlib.sha256(B_PATH.read_bytes()).hexdigest()
    base = initial_duties()
    pool = station_pool(json.loads(raw), base)
    fixed = {car: fixed_duties(base, car) for car in (None, *range(len(STARTS)))}
    search = Search()
    cache, written = {}, set()
    for count, (scene, location, fault, outage) in enumerate(selected, 1):
        targets = targets_for(location)
        old = fixed[fault]
        old_starts = starts_of(old)
        release = EVENT + outage
        initial = release_states(old, release)
        pending = tuple(task for task in targets if task not in old_starts)
        identity = location, fault, tuple(initial.values())
        if identity not in cache:
            seeds = {task: pool[task] for task in pending}
            available = {task: [pair for pair in seeds[task]
                                if all(car in initial for car, _ in pair)] for task in pending}
            new = search.initial(pending, available, initial, old, targets)
            candidates = [(new, initial)]
            if len(new) < 2 * len(pending):
                cut = max(clock for _, clock in initial.values())
                delayed = {car: (point, cut) for car, (point, _) in initial.items()}
                candidates.append((search.initial(pending, available, delayed, old, targets),
                                   delayed))
                alternates = search.reassign_pairs(seeds, initial, targets)
                candidates.append((search.initial(pending, alternates, initial, old, targets,
                                                  diverse=True), initial))
            best, states = min(candidates, key=lambda item: (
                -len(item[0]), max((d.start for d in item[0]), default=0)))
            cache[identity] = search.refine(best, states, old, targets)
        suffix = cache[identity]
        new = old + suffix
        k0, total, last0 = score_for(old_starts, targets)
        new_starts = starts_of(new)
        k1, _, last1 = score_for(new_starts, targets)
        fixed_id = f"F{0 if fault is None else fault + 1}"
        revised_id = f"R{scene[1:]}" if suffix else fixed_id
        if fixed_id not in written:
            duty_writer.writerows(rows_for(fixed_id, old))
            written.add(fixed_id)
        if suffix:
            duty_writer.writerows(rows_for(revised_id, new))
        lower = full_score_lower_bound(old_starts, targets, release)
        worst_rms = max(rms_for(new, task, target) for task, target in targets.items()
                        if task in new_starts)
        angle, radius = ("", "") if location is None else location
        score_writer.writerow((
            scene, angle, radius, "" if fault is None else f"V{fault+1}", outage,
            fixed_id, revised_id, num(release), num(min(d.depart for d in suffix)) if suffix else "",
            total, k0, k1, num(last0), num(last1), num(lower), num(worst_rms),
            *(num(starts.get(task)) for starts in (old_starts, new_starts)
              for task in TASKS), input_hash, template_hash,
        ))
        if count % 100 == 0:
            print(f"B 已求解 {count} 场", flush=True)


def rms_for(duties, task, target):
    pair = [d.path[-1] for d in duties if d.task == task and d.status != RESERVED]
    return rms_bound_at(*pair, target)


def close(a, b):
    return abs(a - b) < EPS


def verify(directory, selected):
    if not __debug__:
        raise RuntimeError("复核依赖 Python 断言")
    with (directory / DUTIES).open(encoding="utf-8-sig", newline="") as f:
        tracks = {}
        for row in csv.DictReader(f):
            values = (float(row[key]) for key in DUTY_FIELDS[5:])
            tracks.setdefault(row["时间轴ID"], []).append(Duty(
                int(row["车辆"][1:]) - 1, row["告警"], tuple(map(tuple, json.loads(row["路线_m"]))),
                *values, row["状态"],
            ))
    with (directory / RESULTS).open(encoding="utf-8-sig", newline="") as f:
        score_rows = list(csv.DictReader(f))
    assert len(score_rows) == len(selected)
    base = initial_duties()
    input_hash = hashlib.sha256(B_PATH.read_bytes()).hexdigest()
    template_hash = hashlib.sha256(TEMPLATES.read_bytes()).hexdigest()
    full, latest, worst = 0, 0.0, 0.0
    for row, (scene, location, fault, outage) in zip(score_rows, selected):
        assert row["场景ID"] == scene and int(row["中断时长_s"]) == outage
        assert (float(row["方位_deg"]), float(row["距离_m"])) == location if location else not row["方位_deg"]
        assert row["故障车"] == (f"V{fault + 1}" if fault is not None else "")
        assert row["B时间轴_SHA256"] == input_hash and row["初始化模板_SHA256"] == template_hash
        reference = fixed_duties(base, fault)
        old, new = (tracks[row[key]] for key in ("固定时间轴ID", "重调度时间轴ID"))
        assert rows_match(reference, old) and rows_match(reference, new[:len(old)])
        suffix = new[len(old):]
        targets = targets_for(location)
        release = EVENT + outage
        assert close(float(row["最早可改派_s"]), release)
        assert close(float(row["开测下界_s"]),
                     full_score_lower_bound(starts_of(old), targets, release))
        if suffix:
            cut = float(row["安全改派_s"])
            assert close(cut, min(d.depart for d in suffix))
            initial = release_states(old, EVENT + outage)
            clocks = {car: clock for car, (_, clock) in initial.items()}
            places = {car: point for car, (point, _) in initial.items()}
            assert len(suffix) % 2 == 0
            seen = set(starts_of(old))
            for a, b in zip(suffix[::2], suffix[1::2]):
                assert a.task == b.task and a.task in targets and a.task not in seen
                seen.add(a.task)
                assert a.car != b.car and a.car != fault and b.car != fault
                assert a.status == b.status == OBSERVE_NEW
                target = targets[a.task]
                for duty in (a, b):
                    assert duty.depart + EPS >= clocks[duty.car] and places[duty.car] == duty.path[0]
                    assert all(RADIUS + OFFSET <= x <= EDGE - OFFSET for p in duty.path for x in p)
                    assert all(road_clearance(p, q) >= RADIUS + OFFSET for p, q in pairwise(duty.path))
                    _, sight = check_view(duty.path[-1], target)
                    assert sight > OFFSET and station_bounds_at(duty.path[-1], target)[0] <= 80
                    clocks[duty.car], places[duty.car] = duty.reset, duty.path[-1]
                assert rows_match(pair_duties(a.task, [(d.car, d.depart, d.path) for d in (a, b)],
                                              target), (a, b))
                assert a.start <= DEADLINE
            assert spacing(tuple(old) + tuple(suffix), initial)
        else:
            assert not row["安全改派_s"] and old == new
        for plan, side in ((old, "固定"), (new, "重调度")):
            actual = starts_of(plan)
            k, n, last = score_for(actual, targets)
            assert int(row[side + "开测数"]) == k and int(row["总告警数"]) == n
            last_field = "固定最晚评分_s" if side == "固定" else "重调度最晚开测上界_s"
            assert close(float(row[last_field]), last)
            for task in TASKS:
                field = row[f"{side}{task}开测_s"]
                assert close(float(field), actual[task]) if task in actual else not field
        rms = max(rms_for(new, task, target) for task, target in targets.items() if task in actual)
        assert rms <= RMS_LIMIT and close(float(row["重调度最大RMS上界_m"]), rms)
        assert k >= int(row["固定开测数"])
        worst = max(worst, rms)
        if k == n:
            latest = max(latest, last)
            full += 1
    print(f"B 复核 {len(selected)} 场；满分 {full} 场；满分组最慢开测上界 {latest:.9f} s；最大 RMS 上界 {worst:.9f} m")


def rows_match(expected, actual):
    return len(expected) == len(actual) and all(
        (a.car, a.task, a.path, a.status) == (b.car, b.task, b.path, b.status)
        and all(close(getattr(a, key), getattr(b, key))
                for key in ("depart", "arrive", "ready", "start", "end", "reset"))
        for a, b in zip(expected, actual)
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--scene", action="append")
    cli.add_argument("--verify", action="store_true")
    args = cli.parse_args()
    selected = [row for row in scenarios() if not args.scene or row[0] in args.scene]
    if args.scene and {row[0] for row in selected} != set(args.scene):
        cli.error("场景编号为 N0001 至 N0599")
    if args.verify:
        verify(args.output, selected)
        return
    if args.output.exists():
        cli.error("求解输出须为新目录")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "verified"
        stage.mkdir()
        with ((stage / RESULTS).open("w", newline="", encoding="utf-8-sig") as scores,
              (stage / DUTIES).open("w", newline="", encoding="utf-8-sig") as tracks):
            writers = [csv.writer(file, lineterminator="\n") for file in (scores, tracks)]
            for writer, fields in zip(writers, (FIELDS, DUTY_FIELDS)):
                writer.writerow(fields)
            runs(selected, *writers)
        verify(stage, selected)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
