"""Verify the reoptimized Q4 continuous coverage certificates with rational bounds.

Run: py -3 -B verify_q4.py q4_1m_certificate.json
Only the Python standard library is required.
"""
import argparse
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, product
import json
import math
from pathlib import Path

BUILDINGS = (
    (22, 34, 16, 26, 20), (63, 79, 59, 71, 28),
    (10, 24, 65, 76, 12), (72, 86, 22, 34, 24), (41, 51, 38, 46, 32),
)
STARTS = ((15, 14), (88, 10), (14, 86), (88, 68), (50, 93))
PI = F("3.141592653589793")
TRIG_ERROR = F(1, 10**12)
STEP = 10**7
# Pi itself lies between PI and PI + 10^-15.
SIGMA_UPPER = (PI + F(1, 10**15)) / 180
PI_TIGHT = F("3.1415926535897932384626433832795028841971693993751")
TRIG_TIGHT = F(1, 10**45)


@lru_cache(maxsize=None)
def trig(degrees, tight=False):
    """Return rational Taylor values with certified absolute error."""
    angle = ((degrees + 180) % 360 - 180) * (PI_TIGHT if tight else PI) / 180
    sin_term, cos_term = angle, F(1)
    sine, cosine = sin_term, cos_term
    for n in range(1, 37 if tight else 16):
        sin_term *= -angle**2 / ((2*n) * (2*n + 1))
        cos_term *= -angle**2 / ((2*n - 1) * (2*n))
        sine += sin_term
        cosine += cos_term
    return sine, cosine


BASE_TARGETS = {}
for name, angle, distance, height in (
    ("T1", 30, 40, 45), ("T2", 150, 45, 35), ("T3", 285, 38, 55),
):
    s, c = trig(F(angle), tight=True)
    scale = 10**36
    point = tuple(F(round(z*scale),scale) for z in (50 + distance*c, 50 + distance*s))
    BASE_TARGETS[name] = (*point, F(1, 10**35), height)


def subtract(p, q):
    return tuple(a-b for a, b in zip(p, q))


def dot(p, q):
    return sum(a*b for a, b in zip(p, q))


def determinant(a, b):
    return a[0]*b[1]-a[1]*b[0]


def hull_points(x, y, h):
    return tuple(product((x-h, x+h), (y-h, y+h)))


def distance_lower(value, scale=10**12):
    value = F(value)
    return F(math.isqrt(value.numerator * scale**2 // value.denominator), scale)


def duration_upper(p, q):
    d = subtract(p, q)
    squared = dot(d, d)/F(9)
    lower = distance_lower(squared, STEP)
    return lower if lower**2 == squared else lower + F(1, STEP)


def angular_bound(x, y, degrees, tight=False):
    if degrees == 180:
        return True
    s, c = trig(degrees,tight)
    return s*x-c*y >= (TRIG_TIGHT if tight else TRIG_ERROR)*(abs(x)+abs(y))


def angle_upper(x, y):
    # A floating-point estimate only proposes a bound. The inequality below
    # proves it using rational Taylor enclosures.
    n = min(180000000, math.ceil(math.degrees(math.atan2(float(y), float(x)))*10**6)+1)
    while not angular_bound(x, y, F(n, 10**6)):
        n += 1
    return F(n, 10**6)


def separated(left, right, normal):
    a = [dot(normal, p) for p in left]
    b = [dot(normal, p) for p in right]
    return max(a) < min(b) or max(b) < min(a)


def sight_clear(station, x, y, vertices, elevation):
    for a, b, c, d, roof in BUILDINGS:
        scale = F(roof-2, elevation)
        truncated = (station, *(tuple(p + scale*(q-p) for p,q in zip(station,v))
                                for v in vertices))
        rectangle = tuple(product((a,b),(c,d)))
        normals = ((1,0), (0,1), (station[1]-y, x-station[0]),
                   *((station[1]-v[1], v[0]-station[0]) for v in vertices))
        if not any(separated(truncated, rectangle, n) for n in normals):
            return False
    return True


@lru_cache(maxsize=30000)
def station_bounds(station, target):
    x, y, h, height = target
    sx, sy = station
    vertices = hull_points(x, y, h)
    dx, dy = abs(x-sx), abs(y-sy)
    far = (dx+h)**2+(dy+h)**2
    near = (max(dx-h,0))**2+(max(dy-h,0))**2
    lower = distance_lower(near)
    if not angular_bound(lower, height-2, F(80)):
        lower = distance_lower(near,10**32)
        assert angular_bound(lower, height-2, F(80), tight=True), ("pitch", station, target)
    assert sight_clear(station, x, y, vertices, height-2), ("sight", station, target)
    yaw = F(180) if x-h<sx and abs(y-sy)<=h else max(
        angle_upper(xx-sx, abs(yy-sy)) for xx,yy in vertices
    )
    pitch = angle_upper(lower, height-2)
    return far, max(yaw,pitch)/30


def pair_bounds(a, b, target, precision):
    u, first = station_bounds(a, target)
    v, second = station_bounds(b, target)
    cross = [determinant(subtract(b,a), subtract(p,a)) for p in hull_points(*target[:3])]
    lower = max(min(cross), -max(cross))
    assert lower > 0
    squared_rms = SIGMA_UPPER**2 * (u+v)*u*v/lower**2
    if precision:
        assert squared_rms <= 1, ("RMS", a, b, target, float(squared_rms))
    return squared_rms, (first, second)


def clear_road(start, end):
    """Check against conservatively enlarged rectangular building footprints."""
    for left, right, bottom, top, _ in BUILDINGS:
        lo, hi = F(0), F(1)
        intersect = True
        for p, q, low, high in zip(start, end, (left,bottom), (right,top)):
            low -= F(3,2)
            high += F(3,2)
            if p == q:
                if p < low or p > high:
                    intersect = False
                    break
            else:
                a,b = sorted(((low-p)/(q-p), (high-p)/(q-p)))
                lo,hi = max(a,lo),min(b,hi)
        if intersect and lo <= hi:
            return False
    return True


def position(phase, time):
    t,u,p,q = phase
    ratio = (time-t)/(u-t)
    return tuple(a+ratio*(b-a) for a,b in zip(p,q))


def segments_separated(first, second, clearance=3):
    # If the full segments are separated, arbitrary within-phase arrival
    # time errors preserve clearance, including exact tangency.
    diffs = [subtract(p, q) for p in first[2:] for q in second[2:]]
    normals = diffs.copy()
    for seg in (first, second):
        x,y = subtract(seg[3],seg[2])
        normals.extend(((y,-x),(-y,x)))
    for n in normals:
        sq = dot(n,n)
        if sq:
            lower = min(dot(n,d) for d in diffs)
            if lower>=0 and lower**2>=clearance**2*sq:
                return True
    return False


def check_distances(tracks, fault=None, failure_time=F(5), tolerance=F(6, STEP)):
    for i,j in combinations(tracks,2):
        a,b = tracks[i], tracks[j]
        ai = bi = 0
        cutoff = failure_time if fault in (i,j) else 1000
        while ai<len(a) and bi<len(b):
            first,second = a[ai],b[bi]
            lo,hi = max(first[0],second[0]),min(first[1],second[1],cutoff)
            if lo>cutoff:
                break
            if lo<=hi:
                u = subtract(position(first,lo), position(second,lo))
                v = subtract(position(first,hi), position(second,hi))
                d = subtract(v,u)
                length = dot(d,d)
                s = min(1,max(0,-dot(u,d)/length)) if length else 0
                nearest = tuple(p+s*q for p,q in zip(u,d))
                if dot(nearest,nearest) < (3+tolerance)**2:
                    assert segments_separated(first,second), ("spacing", i, j, lo, hi,first,second)
            ai += first[1] <= second[1]
            bi += second[1] <= first[1]


def replay(plan, origins, release, targets, precision):
    tracks = {k:[] for k in origins}
    clocks = {k:F(release) for k in origins}
    where = origins.copy()
    starts = {}
    worst = F(0)
    for task, assignments in plan:
        assert task not in starts and len(assignments)==2
        assert len({car for car, _ in assignments})==2
        error, turns = pair_bounds(*(path[-1] for _,path in assignments), targets[task],precision)
        worst = max(worst,error)
        for car,path in assignments:
            assert path[0]==where[car]
            for p,q in zip(path,path[1:]):
                assert all(F(3,2)<=a<=F(197,2) for a in (*p,*q))
                assert clear_road(p,q)
                t = clocks[car]
                u = t+duration_upper(p,q)
                if t<u:
                    tracks[car].append((t,u,p,q))
                clocks[car]=u
            where[car]=path[-1]
        obs=max(clocks[car]+turn for (car,_),turn in zip(assignments, turns))
        starts[task]=obs
        for (car,_),turn in zip(assignments,turns):
            t,u=clocks[car],obs+5+turn
            p=where[car]
            tracks[car].append((t,u,p,p))
            clocks[car]=u
    for car in tracks:
        tracks[car].append((clocks[car],F(100),where[car],where[car]))
    return tracks, starts, where, worst


def schedules(raw):
    tasks = {}
    for duty in sorted(raw,key=lambda d:d["start"]):
        tasks.setdefault(duty["task"], []).append((
            duty["car"], tuple(tuple(F(x) for x in p) for p in duty["path"])
        ))
    return tuple(tasks.items())


def check_partition(boxes):
    book={(F(x),F(y),F(h)):(idx,F(error)) for x,y,h,idx,error in boxes}
    assert len(book)==len(boxes)
    def visit(x,y,h,depth=0):
        a,b=abs(x-50),abs(y-50)
        if max(a-h,0)**2+max(b-h,0)**2>2500 or (a+h)**2+(b+h)**2<900:
            return
        if book.pop((x,y,h),None) is not None:
            return
        assert depth<11, ("uncovered",x,y,h)
        for i,j in product((-1,1),repeat=2):
            visit(x+i*h/2,y+j*h/2,h/2,depth+1)
    visit(F(50),F(50),F(50))
    assert not book


def verify(file):
    bundle=json.loads(file.read_text(encoding="utf-8"),parse_float=F)
    data=bundle["groups"]
    precision=bundle["mode"]=="1m"
    assert bundle["mode"] in ("1m","nominal")
    base=schedules(bundle["reference"])
    reference_starts={d["task"]:F(d["start"]) for d in bundle["reference"]}
    assert {d["task"] for d in bundle["reference"]}==set(BASE_TARGETS)
    assert {d["car"] for d in bundle["reference"]}==set(range(5))
    assert {d["fault"] for d in data}==set(range(-1,5)) and len(data)==6
    init={i:tuple(map(F,p)) for i,p in enumerate(STARTS)}
    baseline, first_starts, origins, reference_error = replay(base,init,0,BASE_TARGETS,precision)
    # The rational shadow timeline and the published minimal-wait prefix
    # differ by microseconds, including at their intermediate waypoints.
    check_distances(baseline,tolerance=F(1,1000))
    assert all(0<=first_starts[t]-reference_starts[t]+F(1,10**8)<F(1,10**5)
               for t in reference_starts)
    assert all(abs(t-5)>F(1,100) for t in first_starts.values())
    assert all(baseline[i][-1][0]<35 for i in baseline)
    print(bundle["mode"], bundle["source"], bundle["source_sha256"], flush=True)
    for group in data:
        fault=group["fault"]
        assert not group["missing"]
        check_partition(group["covered"])
        lost={task for task,assignments in base
              if first_starts[task]>=5 and fault in {d[0] for d in assignments}}
        fixed={i:p for i,p in origins.items() if i!=fault}
        templates={idx:schedules(raw) for idx,raw in group["plans"].items()}
        worst_start=max((t for q,t in first_starts.items() if q not in lost),default=F(0))
        worst_error=reference_error
        for x,y,h,idx,reported in group["covered"]:
            targets={**BASE_TARGETS,"T4":(F(x),F(y),F(h),50)}
            paths, starts, _, error=replay(templates[str(idx)],fixed,35,targets,precision)
            assert set(starts)==lost|{"T4"}
            check_distances(paths)
            worst_start=max(worst_start,*starts.values())
            worst_error=max(worst_error,error)
        assert worst_start<=60
        upper_time=F(math.ceil(worst_start*1000000),1000000)
        upper_rms=distance_lower(worst_error,10**9)+F(1,10**9)
        print(f"{'healthy' if fault<0 else f'V{fault+1}'}"
              f"  {len(group['covered'])} squares  {len(templates)} schedules"
              f"  fixed {3-len(lost)}/4   time <= {float(upper_time):.6f} s"
              + (f"   RMS <= {float(upper_rms):.9f} m" if precision else ""), flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file",type=Path)
    verify(ap.parse_args().file)
