"""Vehicle events and replay of the jointly issued question 2 timetable."""
import csv
import json
import math
import sys
from dataclasses import dataclass, replace
from itertools import combinations

from q4_score import DEADLINE, EVENT, ROOT, TIMELINE

sys.path.insert(0, str(ROOT / "第二问" / "代码"))
from q2_schedule import OBSERVE, RADIUS, SPEED, STARTS, TURN

EAST = (0.0, 0.0)
TOL = 1e-8
TASKS = ("T1", "T2", "T3", "T4")
SCORES, TRACKS = "重调度逐场景对照.csv", "第四问逐车时间轴.csv"
DRIVE, AIM, WAIT, OBSERVE_PHASE, RESET, IDLE, RESERVED, FAILED = (
    "行驶", "指向", "等搭档", "联合观测", "复位", "待命", "原表保留时段", "故障退出"
)


@dataclass(frozen=True)
class Phase:
    kind: str
    task: str
    begin: float
    end: float
    p: tuple
    q: tuple
    aim0: tuple = EAST
    aim1: tuple = EAST

    def state(self, t):
        fraction = (t - self.begin) / (self.end - self.begin)
        return tuple(a + fraction * (b - a) for a, b in zip(
            self.p + self.aim0, self.q + self.aim1
        ))

    def until(self, t):
        x, y, yaw, pitch = self.state(t)
        return replace(self, end=t, q=(x, y), aim1=(yaw, pitch))


@dataclass(frozen=True)
class Duty:
    car: int
    task: str
    path: tuple
    aim: tuple
    depart: float
    arrive: float
    ready: float
    start: float
    finish: float
    reset: float

    def phases(self, success=True):
        phases = []
        time = self.depart
        for p, q in zip(self.path, self.path[1:]):
            end = time + math.dist(p, q) / SPEED
            if end > time:
                phases.append(Phase(DRIVE, self.task, time, end, p, q))
            time = end
        station = self.path[-1]
        phases.extend((
            Phase(AIM, self.task, self.arrive, self.ready, station, station, EAST, self.aim),
            Phase(WAIT, self.task, self.ready, self.start, station, station, self.aim, self.aim),
            Phase(OBSERVE_PHASE if success else RESERVED, self.task, self.start, self.finish,
                  station, station, self.aim, self.aim),
            Phase(RESET, self.task, self.finish, self.reset, station, station, self.aim, EAST),
        ))
        return tuple(phase for phase in phases if phase.end > phase.begin + TOL)


def read_duties():
    duties = []
    fields = ("出发_s", "到站_转动开始_s", "指向完成_s", "观测开始_s",
              "观测结束_复位开始_s", "复位完成_s")
    with TIMELINE.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["方案"] == "优化":
                duties.append(Duty(int(row["车辆"][1:]) - 1, row["告警"],
                                   tuple(map(tuple, json.loads(row["路径坐标_m"]))),
                                   (float(row["方位角_deg"]), float(row["俯仰角_deg"])),
                                   *(float(row[field]) for field in fields)))
    return duties


def fixed_tracks(duties, fault):
    lost = {duty.task for duty in duties if duty.car == fault and duty.start >= EVENT}
    tracks = []
    for car, origin in enumerate(STARTS):
        phases = fill(tuple(phase for duty in duties if duty.car == car
                            for phase in duty.phases(duty.task not in lost)), car)
        if fault == car:
            phases = prefix(phases, EVENT)
            p = phases[-1].q if phases else tuple(origin)
            aim = phases[-1].aim1 if phases else EAST
            phases += (Phase(FAILED, "", EVENT, DEADLINE, p, p, aim, aim),)
        tracks.append(phases)
    return tuple(tracks)


def prefix(track, time):
    return tuple(phase if phase.end <= time else phase.until(time)
                 for phase in track if phase.begin < time - TOL)


def fill(phases, car):
    """Fill unassigned time with stationary east-facing standby."""
    previous = 0.0
    position = tuple(STARTS[car])
    track = []
    for phase in phases:
        if phase.begin > previous + TOL:
            track.append(Phase(IDLE, "", previous, phase.begin, position, position))
        track.append(phase)
        previous, position = phase.end, phase.q
    if previous < DEADLINE:
        track.append(Phase(IDLE, "", previous, DEADLINE, position, position))
    return tuple(track)


def observation_starts(tracks):
    """A failed car can contribute only to a start before its exit."""
    return {phase.task: phase.begin for track in tracks for phase in track
            if phase.kind == OBSERVE_PHASE}


def snapshot(tracks, fault, cut):
    """Lock ongoing valid observation/reset chains and resume from real car state."""
    past, states = [], {}
    locked = {}
    for car, track in enumerate(tracks):
        if car == fault:
            past.append(track)
            continue
        active = next(phase for phase in track if phase.begin <= cut + TOL < phase.end)
        chain = prefix(track, cut)
        ready = cut
        if active.kind == OBSERVE_PHASE:
            # At this optional cutoff the controller chooses to retain this start.
            chain = tuple(phase for phase in track if phase.begin < active.end - TOL
                          or (phase.kind == RESET and phase.begin == active.end))
            ready = chain[-1].end
        elif active.kind == RESET:
            chain = tuple(phase for phase in track if phase.begin < active.end - TOL)
            ready = active.end
        x, y, yaw, pitch = chain[-1].q + chain[-1].aim1
        point, aim = (x, y), (yaw, pitch)
        duration = max(abs(yaw), abs(pitch)) / TURN
        if duration > TOL:
            chain += (Phase(RESET, active.task, ready, ready + duration, point, point, aim),)
            ready += duration
        past.append(chain)
        states[car] = (point, ready)
    locked = observation_starts(past)
    return tuple(past), states, locked


def tracks_for(past, duties, fault):
    return tuple(track if car == fault else fill(
        track + tuple(phase for duty in duties if duty.car == car for phase in duty.phases()),
        car,
    ) for car, track in enumerate(past))


def trajectory_clear(tracks, fault):
    for i, j in combinations(range(len(tracks)), 2):
        end = EVENT if fault in (i, j) else math.inf
        a, b = tracks[i], tracks[j]
        m = n = 0
        while m < len(a) and n < len(b):
            left, right = max(a[m].begin, b[n].begin), min(a[m].end, b[n].end, end)
            if left >= end:
                break
            if right > left:
                x, y, *_ = a[m].state(left)
                u, v, *_ = b[n].state(left)
                p = (x - u, y - v)
                x, y, *_ = a[m].state(right)
                u, v, *_ = b[n].state(right)
                dx, dy = x - u - p[0], y - v - p[1]
                length2 = dx * dx + dy * dy
                t = max(0.0, min(1.0, -(p[0] * dx + p[1] * dy) / length2)) if length2 else 0
                if math.hypot(p[0] + t * dx, p[1] + t * dy) < 2 * RADIUS - TOL:
                    return False
            m, n = m + (a[m].end <= b[n].end), n + (b[n].end <= a[m].end)
    return True


def first_change(reference, planned):
    answer = math.inf
    for a, b in zip(reference, planned):
        i = j = 0
        while i < len(a) and j < len(b):
            left, right = max(a[i].begin, b[j].begin), min(a[i].end, b[j].end)
            if right > left + TOL:
                t = (left + right) / 2
                if ((a[i].kind, a[i].task) != (b[j].kind, b[j].task)
                        or math.dist(a[i].state(t), b[j].state(t)) > TOL):
                    answer = min(answer, left)
                    break
            i, j = i + (a[i].end <= b[j].end), j + (b[j].end <= a[i].end)
    return None if math.isinf(answer) else answer


if __name__ == "__main__":
    tracks = fixed_tracks(read_duties(), fault=None)
    starts = observation_starts(tracks)
    print(f"基准时间轴：{len(tracks)} 辆车，{len(starts)} 个观测任务")
    for task, start in sorted(starts.items()):
        print(f"{task} 观测开始：{start:.9f} s")
