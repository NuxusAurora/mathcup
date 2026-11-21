# -*- coding: utf-8 -*-
"""
问题四：扰动下的重调度与方案比较
- 基线：问题二(2) 最优调度（读 问题二_2_最优调度时间轴.csv）
- 扰动：新增告警(θ,R,50) / 随机车辆故障 / 通信中断 [5,t_rec]
- 重调度 vs 保持原方案；评价 (η, t_last)，窗口 [0,60] s
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOLVE_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _SOLVE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import *
from _geometry import *
from itertools import permutations, product, combinations

import matplotlib.colors as mcolors

HORIZON = 60.0
G_LO, G_HI = 40.0, 140.0
NEW_H = 50.0

ORIG = ['T1', 'T2', 'T3']
ALERT_COLOR = {'T1': '#d62728', 'T2': '#1f77b4', 'T3': '#2ca02c', 'NEW': '#9467bd'}

# ---------- 读基线调度 ----------
BASE_CSV = os.path.join(_SOLVE_DIR, '问题二', '结果', '问题二_2_最优调度时间轴.csv')
base_df = pd.read_csv(BASE_CSV)
tasks = {v: [] for v in VEHICLE_ORDER}
for _, row in base_df.iterrows():
    v = row['车辆']; al = row['告警']
    if al == '-':
        continue
    s = row['观测站'].strip('()').split(',')
    st = np.array([float(s[0]), float(s[1])])
    tasks[v].append({
        'alert': al, 'station': st,
        'depart': float(row['出发时刻(s)']), 'arrive': float(row['到站时刻(s)']),
        'rot': float(row['云台转动(s)']), 'start': float(row['观测开始(s)']),
        'end': float(row['观测结束(s)']), 'reset': float(row['复位完成(s)']),
    })
baseline_start = {}
for v in VEHICLE_ORDER:
    for tk in tasks[v]:
        baseline_start[tk['alert']] = tk['start']
print('基线观测开始:', {k: round(v, 2) for k, v in baseline_start.items()})
print('基线任务序列:', {v: [tk['alert'] for tk in tasks[v]] for v in VEHICLE_ORDER})

# ---------- 可观测候选点 ----------
def obs_pts(target, gs=2.0):
    xx = np.arange(0.0, 100.0 + gs, gs)
    X, Y = np.meshgrid(xx, xx)
    disk = (X - VEH_R >= 0) & (X + VEH_R <= 100) & (Y - VEH_R >= 0) & (Y + VEH_R <= 100)
    for b in BUILDINGS:
        dx = np.maximum(np.maximum(b[0] - X, 0.0), X - b[1])
        dy = np.maximum(np.maximum(b[2] - Y, 0.0), Y - b[3])
        disk &= np.hypot(dx, dy) >= VEH_R
    tx, ty, H = target
    r = np.hypot(tx - X, ty - Y)
    phi = np.degrees(np.arctan2(H - CAM_H, r))
    pitch = (phi >= PITCH_MIN) & (phi <= PITCH_MAX)
    blocked = np.zeros(X.shape, dtype=bool)
    Xf = X.ravel(); Yf = Y.ravel()
    for b in BUILDINGS:
        xmin, xmax, ymin, ymax, h = b[0], b[1], b[2], b[3], b[4]
        ok = np.ones(Xf.size, dtype=bool); tmin = np.zeros_like(Xf); tmax = np.ones_like(Xf)
        for p0, p1, lo, hi in [(Xf, tx, xmin, xmax), (Yf, ty, ymin, ymax)]:
            same = (p1 == p0)
            ok = ok & (~same | ((p0 >= lo) & (p0 <= hi)))
            diff = ~same
            if diff.any():
                denom = np.where(diff, p1 - p0, 1.0)
                t1 = (lo - p0) / denom; t2 = (hi - p0) / denom
                tlo = np.minimum(t1, t2); thi = np.maximum(t1, t2)
                tmin = np.maximum(tmin, np.where(diff, tlo, tmin))
                tmax = np.minimum(tmax, np.where(diff, thi, tmax))
        ok = ok & (tmin <= tmax)
        t1z = (0 - CAM_H) / (H - CAM_H); t2z = (h - CAM_H) / (H - CAM_H)
        tloz = min(t1z, t2z); thiz = max(t1z, t2z)
        ok = ok & (tmin <= thiz) & (tmax >= tloz)
        blocked |= ok.reshape(X.shape)
    obs = disk & pitch & (~blocked)
    idx = np.argwhere(obs)
    pts = []
    for i, j in idx:
        x, y = X[i, j], Y[i, j]
        pts.append((x, y, azimuth_from_station(x, y, target), r[i, j]))
    return np.array(pts)

TARG = {al: TARGETS[al] for al in ORIG}
PTS = {al: obs_pts(TARG[al]) for al in ORIG}


def rot_of(st, al, TARGD):
    return rotate_time(azimuth_from_station(st[0], st[1], TARGD[al]), pitch_deg(st[0], st[1], TARGD[al]))


# ---------- 状态快照 ----------
def snapshot(t_trig, failed):
    states = {}
    done = set()
    for v in VEHICLE_ORDER:
        if v == failed:
            continue
        pos = VEHICLES[v].copy()
        avail = 0.0
        prev = VEHICLES[v].copy()
        for tk in tasks[v]:
            if t_trig >= tk['reset']:
                pos = tk['station'].copy(); avail = tk['reset']; prev = pos
                done.add(tk['alert'])
                continue
            if t_trig >= tk['start']:
                # 观测已开始（尚未结束/复位）→ 不可重排，计为已开始
                done.add(tk['alert'])
                pos = tk['station'].copy(); avail = tk['reset']; prev = pos
                break
            # 已派出未开始观测 → 可重指派，位置插值
            if t_trig < tk['depart']:
                pos = prev.copy()
            elif t_trig < tk['arrive']:
                frac = (t_trig - tk['depart']) / max(tk['arrive'] - tk['depart'], 1e-9)
                pos = prev + frac * (tk['station'] - prev)
            else:
                pos = tk['station'].copy()
            avail = t_trig
            break
        states[v] = (pos, avail)
    return states, done


# ---------- 调度 ----------
def alert_options(alert, vehicles, states, TARGD, PTS):
    pts = PTS[alert]
    cand = {}
    for v in vehicles:
        px, py = states[v][0]
        d = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
        cand[v] = pts[np.argsort(d)[:6]]
    options = []
    for a in range(len(vehicles)):
        for b in range(a + 1, len(vehicles)):
            va, vb = vehicles[a], vehicles[b]
            best = None
            for sa in cand[va]:
                for sb in cand[vb]:
                    g = inter_angle(sa[2], sb[2])
                    if not (G_LO <= g <= G_HI):
                        continue
                    ra = states[va][1] + np.hypot(sa[0] - states[va][0][0], sa[1] - states[va][0][1]) / SPEED + rot_of(sa, alert, TARGD)
                    rb = states[vb][1] + np.hypot(sb[0] - states[vb][0][0], sb[1] - states[vb][0][1]) / SPEED + rot_of(sb, alert, TARGD)
                    mx = max(ra, rb)
                    if best is None or mx < best[0] - 1e-9:
                        best = (mx, g, (sa[0], sa[1]), (sb[0], sb[1]), va, vb)
            if best is not None:
                options.append(best)
    return options


def simulate_states(routes, st_of, alerts, states, TARGD):
    pos = {v: np.array(states[v][0], float) for v in states}
    starts = {al: 0.0 for al in alerts}
    for _ in range(200):
        new = {}
        for v in states:
            tv = states[v][1]
            p = pos[v].copy()
            for al in routes.get(v, []):
                st = st_of[(v, al)]
                drive = np.hypot(st[0] - p[0], st[1] - p[1]) / SPEED
                arrive = tv + drive
                rot = rot_of(st, al, TARGD)
                ready = arrive + rot
                new[al] = max(new.get(al, -np.inf), ready)
                tv = starts[al] + OBS_TIME + rot
                p = st
        if max(abs(new[al] - starts[al]) for al in alerts) < 1e-9:
            break
        starts = new
    return starts


def reschedule(remaining, vehicles, states, TARGD, PTS):
    if len(remaining) == 0:
        return (0, 0.0), {v: [] for v in vehicles}, {}, {}
    opts = {al: alert_options(al, vehicles, states, TARGD, PTS) for al in remaining}
    if any(len(opts[al]) == 0 for al in remaining):
        return None, None, None, None
    best = None
    for combo in product(*[opts[al] for al in remaining]):
        routes = {v: [] for v in vehicles}
        st_of = {}
        for al, opt in zip(remaining, combo):
            mx, g, sa, sb, va, vb = opt
            routes[va].append(al); routes[vb].append(al)
            st_of[(va, al)] = np.array(sa); st_of[(vb, al)] = np.array(sb)
        order_opts = [list(permutations(routes[v])) for v in vehicles]
        for ordc in product(*order_opts):
            routes2 = {vehicles[k]: list(ordc[k]) for k in range(len(vehicles))}
            starts = simulate_states(routes2, st_of, remaining, states, TARGD)
            # lexicographic: max eta, then min t_last
            k = sum(1 for al in remaining if starts[al] <= HORIZON)
            t_last = HORIZON if k < len(remaining) else max(starts.values())
            key = (-k, t_last)
            if best is None or key < best[0]:
                best = (key, routes2, st_of, starts)
    return best[0], best[1], best[2], best[3]


def evaluate(remaining, done, starts_remaining, all_alerts):
    N = len(all_alerts)
    k = 0
    t_last = 0.0
    for al in all_alerts:
        if al in done:
            s = baseline_start[al]
            k += 1
            t_last = max(t_last, s)
        else:
            s = starts_remaining.get(al, 1e9)
            if s <= HORIZON:
                k += 1
                t_last = max(t_last, s)
    if k < N:
        t_last = HORIZON
    return k / N, t_last


# ---------- 保持原方案 ----------
def keep_original(scenario, TARGD, PTS):
    new = scenario.get('new'); failed = scenario.get('failed'); t_rec = scenario.get('t_rec')
    all_alerts = ORIG + (['NEW'] if new else [])
    done = set(al for al in ORIG if baseline_start[al] < 5.0)
    # 原始方案中因故障丢失车辆的告警
    lost = set()
    if failed:
        for al in ORIG:
            serving = [v for v in VEHICLE_ORDER if any(tk['alert'] == al for tk in tasks[v])]
            if failed in serving and al not in done:
                lost.add(al)
    starts_remaining = {}
    for al in ORIG:
        if al not in done and al not in lost:
            starts_remaining[al] = baseline_start[al]
        elif al in lost:
            starts_remaining[al] = 1e9
    # 新增告警：恢复后指派给最早空闲的2台非故障车
    if new:
        trig = t_rec if t_rec else 5.0
        vehicles = [v for v in VEHICLE_ORDER if v != failed]
        free = {}
        lastpos = {}
        for v in vehicles:
            f = 0.0
            lp = VEHICLES[v].copy()
            for tk in tasks[v]:
                f = max(f, tk['reset'])
                lp = tk['station'].copy()
            free[v] = max(f, trig)
            lastpos[v] = lp
        states = {v: (lastpos[v], free[v]) for v in vehicles}
        opts = alert_options('NEW', vehicles, states, TARGD, PTS)
        if opts:
            best = min(opts, key=lambda o: o[0])
            starts_remaining['NEW'] = best[0]
        else:
            starts_remaining['NEW'] = 1e9
    eta, t_last = evaluate([], done, starts_remaining, all_alerts)
    return eta, t_last, starts_remaining, done, lost


# ---------- 场景 ----------
def run_scenario(sc, TARGD, PTSd):
    new = sc.get('new'); failed = sc.get('failed'); t_rec = sc.get('t_rec')
    t_trig = t_rec if t_rec else 5.0
    all_alerts = ORIG + (['NEW'] if new else [])
    states, done = snapshot(t_trig, failed)
    vehicles = [v for v in VEHICLE_ORDER if v != failed]
    remaining = [al for al in ORIG if al not in done] + (['NEW'] if new else [])
    if new:
        TARGD['NEW'] = target_from_alert(new[0], new[1], NEW_H)
        PTSd['NEW'] = obs_pts(TARGD['NEW'])
    else:
        TARGD.pop('NEW', None); PTSd.pop('NEW', None)
    key, routes, st_of, starts = reschedule(remaining, vehicles, states, TARGD, PTSd)
    if starts is None:
        eta_r, tlast_r = 0.0, HORIZON
        starts_r = {al: 1e9 for al in remaining}
    else:
        eta_r, tlast_r = evaluate(remaining, done, starts, all_alerts)
        starts_r = starts
    eta_k, tlast_k, starts_k, done_k, lost_k = keep_original(sc, TARGD, PTSd)
    return {
        'scenario': sc, 'N': len(all_alerts), 'done': done,
        'eta_r': eta_r, 'tlast_r': tlast_r, 'starts_r': starts_r,
        'routes_r': routes, 'st_of_r': st_of, 'states': states, 'vehicles': vehicles, 'remaining': remaining,
        'eta_k': eta_k, 'tlast_k': tlast_k, 'starts_k': starts_k, 'lost': lost_k, 'done_k': done_k,
        'TARGD': dict(TARGD),
    }


# 场景集
scenarios = []
for th in [0, 45, 90, 135, 180, 225, 270, 315]:
    scenarios.append(dict(new=(th, 40)))
for v in VEHICLE_ORDER:
    scenarios.append(dict(failed=v))
for tr in [8, 15, 25]:
    scenarios.append(dict(t_rec=tr))
scenarios.append(dict(new=(135, 40), failed='V2'))
scenarios.append(dict(new=(270, 40), t_rec=15))
scenarios.append(dict(failed='V3', t_rec=10))
scenarios.append(dict(new=(0, 45), failed='V5', t_rec=20))

results = []
for sc in scenarios:
    TARGD = {al: TARGETS[al] for al in ORIG}
    PTSd = {al: PTS[al] for al in ORIG}
    r = run_scenario(sc, TARGD, PTSd)
    results.append(r)
    label = '新告警' if 'new' in sc else ('故障' if 'failed' in sc else '通信')
    if 'new' in sc: label += f' θ={sc["new"][0]}°,R={sc["new"][1]}'
    if 'failed' in sc: label += f' {sc["failed"]}'
    if 't_rec' in sc: label += f' t_rec={sc["t_rec"]}'
    print(f'{label}: N={r["N"]}, 重调度(η={r["eta_r"]:.2f},t_last={r["tlast_r"]:.1f}) | 原方案(η={r["eta_k"]:.2f},t_last={r["tlast_k"]:.1f})')

# ================= 绘图 =================
CMAP4 = 'magma'

# 图4-1 基线调度甘特（标注 t=5）
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ypos = {v: len(VEHICLE_ORDER) - 1 - i for i, v in enumerate(VEHICLE_ORDER)}
kind_color = {'drive': '#d9d9d9', 'rotate': '#e6e6e6', 'wait': '#f5f5f5', 'reset': '#c8c8c8'}
for v in VEHICLE_ORDER:
    y = ypos[v]
    for tk in tasks[v]:
        ax.barh(y, tk['arrive'] - tk['depart'], left=tk['depart'], height=0.5, color='#d9d9d9', edgecolor='black', linewidth=0.4)
        ax.barh(y, tk['rot'], left=tk['arrive'], height=0.5, color='#e6e6e6', edgecolor='black', linewidth=0.4)
        ax.barh(y, tk['start'] - (tk['arrive'] + tk['rot']), left=tk['arrive'] + tk['rot'], height=0.5, color='#f5f5f5', edgecolor='black', linewidth=0.4)
        ax.barh(y, OBS_TIME, left=tk['start'], height=0.5, color=ALERT_COLOR[tk['alert']], edgecolor='black', linewidth=0.4)
        ax.barh(y, tk['reset'] - tk['end'], left=tk['end'], height=0.5, color='#c8c8c8', edgecolor='black', linewidth=0.4)
ax.axvline(5.0, color='#d62728', linewidth=1.4, linestyle='--')
ax.text(5.1, len(VEHICLE_ORDER) - 0.4, 't=5 s 扰动时刻', fontsize=9, color='#d62728')
for al in ORIG:
    ax.barh(0, 0, left=0, color=ALERT_COLOR[al], label=f'{al}观测')
ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()), fontsize=9)
ax.set_xlabel('时间 (s)', fontsize=10); ax.set_ylabel('车辆', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.set_xlim(0, 22)
top_legend(fig, [ax])
save_fig(fig, '图4-1_基线调度甘特')

# 图4-2 新增告警方向场景（8方向位置示意）
fig, ax = plt.subplots(figsize=(7.0, 4.8))
for b, nm in zip(BUILDINGS, BUILDING_NAMES):
    ax.add_patch(Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], facecolor='#6b6b6b', edgecolor='black', linewidth=0.9))
    ax.text((b[0]+b[1])/2, (b[3]+b[2])/2, nm, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax.plot([0,100,100,0,0],[0,0,100,100,0], color='black', linewidth=1.2)
ax.plot(*CENTER, marker='o', color='black', markersize=5)
for th in [0,45,90,135,180,225,270,315]:
    t = target_from_alert(th, 40, NEW_H)
    ax.plot(t[0], t[1], marker='^', color='#9467bd', markersize=9)
    ax.annotate(f'{th}°', xy=(t[0], t[1]), xytext=(t[0]+1.5, t[1]+1.5), fontsize=8, color='#9467bd')
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图4-2_新增告警方向场景')

# 分组结果提取
new_res = [r for r in results if 'new' in r['scenario'] and 'failed' not in r['scenario'] and 't_rec' not in r['scenario']]
fail_res = [r for r in results if 'failed' in r['scenario'] and 'new' not in r['scenario'] and 't_rec' not in r['scenario']]
comm_res = [r for r in results if 't_rec' in r['scenario'] and 'new' not in r['scenario'] and 'failed' not in r['scenario']]
comb_res = [r for r in results if r not in new_res and r not in fail_res and r not in comm_res]

# 图4-3 全部场景 η 对比（重调度 vs 原方案）
fig, ax = plt.subplots(figsize=(10.0, 4.6))
n = len(results)
x = np.arange(n)
ax.bar(x - 0.2, [r['eta_r'] for r in results], width=0.38, color='#1f9e89', edgecolor='black', linewidth=0.6, label='重调度')
ax.bar(x + 0.2, [r['eta_k'] for r in results], width=0.38, color='#fdae61', edgecolor='black', linewidth=0.6, label='保持原方案')
ax.set_xticks(x); ax.set_xticklabels([f'S{i+1}' for i in range(n)], fontsize=8, rotation=45)
ax.set_ylabel('完成率 η', fontsize=10)
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-3_各场景完成率对比')

# 图4-4 新增告警方向 η 曲线
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ths = [r['scenario']['new'][0] for r in new_res]
ax.plot(ths, [r['eta_r'] for r in new_res], marker='o', color='#1f9e89', linewidth=1.6, label='重调度')
ax.plot(ths, [r['eta_k'] for r in new_res], marker='s', color='#fdae61', linewidth=1.6, label='保持原方案')
ax.set_xlabel('新增告警方位角 θ (deg)', fontsize=10); ax.set_ylabel('完成率 η', fontsize=10)
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-4_新增告警方向影响')

# 图4-5 故障车辆影响
fig, ax = plt.subplots(figsize=(7.0, 4.4))
fvs = [r['scenario']['failed'] for r in fail_res]
ax.bar(np.arange(5) - 0.2, [r['eta_r'] for r in fail_res], width=0.38, color='#1f9e89', edgecolor='black', linewidth=0.6, label='重调度')
ax.bar(np.arange(5) + 0.2, [r['eta_k'] for r in fail_res], width=0.38, color='#fdae61', edgecolor='black', linewidth=0.6, label='保持原方案')
ax.set_xticks(np.arange(5)); ax.set_xticklabels(fvs, fontsize=10)
ax.set_ylabel('完成率 η', fontsize=10); ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-5_故障车辆影响')

# 图4-6 通信中断影响
fig, ax = plt.subplots(figsize=(7.0, 4.4))
trs = [r['scenario']['t_rec'] for r in comm_res]
ax.plot(trs, [r['eta_r'] for r in comm_res], marker='o', color='#1f9e89', linewidth=1.6, label='重调度')
ax.plot(trs, [r['eta_k'] for r in comm_res], marker='s', color='#fdae61', linewidth=1.6, label='保持原方案')
ax.set_xlabel('通信恢复时刻 t_rec (s)', fontsize=10); ax.set_ylabel('完成率 η', fontsize=10)
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-6_通信中断影响')

# 图4-7 组合扰动 η/t_last 对比
fig, ax = plt.subplots(figsize=(7.0, 4.4))
cn = len(comb_res)
x = np.arange(cn)
ax.bar(x - 0.2, [r['eta_r'] for r in comb_res], width=0.38, color='#1f9e89', edgecolor='black', linewidth=0.6, label='重调度')
ax.bar(x + 0.2, [r['eta_k'] for r in comb_res], width=0.38, color='#fdae61', edgecolor='black', linewidth=0.6, label='保持原方案')
ax.set_xticks(x); ax.set_xticklabels([f'C{i+1}' for i in range(cn)], fontsize=10)
ax.set_ylabel('完成率 η', fontsize=10); ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-7_组合扰动完成率对比')

# 图4-8 某场景重调度时间轴（新告警 θ=0,R=40）
def draw_scenario_gantt(r, fname):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    vlist = r['vehicles']
    ypos = {v: len(vlist) - 1 - i for i, v in enumerate(vlist)}
    # 用模拟产生事件
    routes = r.get('routes_r'); st_of = r.get('st_of_r'); states = r['states']
    if routes is None:
        save_fig(fig, fname + '_empty'); return
    remaining = r['remaining']
    # 事件重建
    starts = r['starts_r']
    for v in vlist:
        y = ypos[v]
        tv = states[v][1]; p = np.array(states[v][0], float)
        for al in routes[v]:
            st = st_of[(v, al)]
            drive = np.hypot(st[0]-p[0], st[1]-p[1]) / SPEED
            ax.barh(y, drive, left=tv, height=0.5, color='#d9d9d9', edgecolor='black', linewidth=0.4)
            arrive = tv + drive
            rot = rot_of(st, al, r['TARGD'])
            ax.barh(y, rot, left=arrive, height=0.5, color='#e6e6e6', edgecolor='black', linewidth=0.4)
            ready = arrive + rot
            si = starts[al]
            if si > ready + 1e-9:
                ax.barh(y, si - ready, left=ready, height=0.5, color='#f5f5f5', edgecolor='black', linewidth=0.4)
            ax.barh(y, OBS_TIME, left=si, height=0.5, color=ALERT_COLOR[al], edgecolor='black', linewidth=0.4)
            tv = si + OBS_TIME + rot
            p = st
    ax.axvline(5.0, color='#d62728', linewidth=1.2, linestyle='--')
    for al in ORIG + ['NEW']:
        ax.barh(0, 0, left=0, color=ALERT_COLOR[al], label=f'{al}观测')
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()), fontsize=9)
    ax.set_xlabel('时间 (s)', fontsize=10); ax.set_ylabel('车辆', fontsize=10)
    ax.tick_params(labelsize=9, direction='in'); despine(ax)
    ax.set_xlim(0, HORIZON)
    top_legend(fig, [ax])
    save_fig(fig, fname)

r_new0 = results[0]  # 新告警 θ=0,R=40
if r_new0['routes_r'] is not None:
    draw_scenario_gantt(r_new0, '图4-8_重调度时间轴示例')

# 图4-9 原方案时间轴（θ=0 场景）
def draw_keep_gantt(r, fname):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    vlist = [v for v in VEHICLE_ORDER if v != r['scenario'].get('failed')]
    ypos = {v: len(vlist)-1-i for i, v in enumerate(vlist)}
    for v in vlist:
        y = ypos[v]
        for tk in tasks[v]:
            if tk['alert'] in r['done'] or tk['alert'] in r.get('lost', set()):
                continue
            if tk['start'] >= 5.0:
                ax.barh(y, tk['arrive']-tk['depart'], left=tk['depart'], height=0.5, color='#d9d9d9', edgecolor='black', linewidth=0.4)
                ax.barh(y, tk['rot'], left=tk['arrive'], height=0.5, color='#e6e6e6', edgecolor='black', linewidth=0.4)
                ax.barh(y, OBS_TIME, left=tk['start'], height=0.5, color=ALERT_COLOR[tk['alert']], edgecolor='black', linewidth=0.4)
    ax.axvline(5.0, color='#d62728', linewidth=1.2, linestyle='--')
    for al in ORIG:
        ax.barh(0, 0, left=0, color=ALERT_COLOR[al], label=f'{al}观测')
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()), fontsize=9)
    ax.set_xlabel('时间 (s)', fontsize=10); ax.set_ylabel('车辆', fontsize=10)
    ax.tick_params(labelsize=9, direction='in'); despine(ax)
    ax.set_xlim(0, HORIZON)
    top_legend(fig, [ax])
    save_fig(fig, fname)

draw_keep_gantt(r_new0, '图4-9_原方案时间轴示例')

# 图4-10 新增告警可观测区（θ=0,R=40）
fig, ax = plt.subplots(figsize=(6.8, 6.0))
t0 = target_from_alert(0, 40, NEW_H)
Xp = PTS['NEW'] if 'NEW' in PTS else obs_pts(t0)
for b, nm in zip(BUILDINGS, BUILDING_NAMES):
    ax.add_patch(Rectangle((b[0], b[2]), b[1]-b[0], b[3]-b[2], facecolor='#6b6b6b', edgecolor='black', linewidth=0.9))
    ax.text((b[0]+b[1])/2, (b[3]+b[2])/2, nm, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax.plot([0,100,100,0,0],[0,0,100,100,0], color='black', linewidth=1.2)
if len(Xp):
    ax.scatter(Xp[:,0], Xp[:,1], c=Xp[:,3], s=3, cmap=CMAP4, alpha=0.6, vmin=0, vmax=50)
ax.plot(t0[0], t0[1], marker='^', color='#9467bd', markersize=11)
ax.text(t0[0]+1.5, t0[1]+1.5, 'NEW', fontsize=10, color='#9467bd', fontweight='bold')
ax.set_xlim(0,100); ax.set_ylim(0,100); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图4-10_新增告警可观测区')

# 图4-11 t_last 对比
fig, ax = plt.subplots(figsize=(10.0, 4.6))
n = len(results)
x = np.arange(n)
ax.scatter(x, [r['tlast_r'] for r in results], color='#1f9e89', marker='o', s=45, label='重调度', zorder=3)
ax.scatter(x, [r['tlast_k'] for r in results], color='#fdae61', marker='s', s=45, label='保持原方案', zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f'S{i+1}' for i in range(n)], fontsize=8, rotation=45)
ax.set_ylabel('最晚观测开始时间 t_last (s)', fontsize=10)
ax.set_ylim(0, HORIZON + 3)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-11_各场景t_last对比')

# 图4-12 收益场景统计
fig, ax = plt.subplots(figsize=(7.0, 4.4))
gain = [r['eta_r'] - r['eta_k'] for r in results]
bins = np.arange(-0.05, 1.05, 0.05)
ax.hist(gain, bins=20, color='#1f9e89', edgecolor='black', linewidth=0.6, alpha=0.85)
ax.set_xlabel('Δη = η_重调度 − η_原方案', fontsize=10); ax.set_ylabel('场景数', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图4-12_重调度收益分布')

# 图4-13 η 随 R 影响（θ=0 方向，R=30/40/50）
fig, ax = plt.subplots(figsize=(7.0, 4.4))
Rs = [30, 40, 50]
eta_r_r = []; eta_k_r = []
for Rv in Rs:
    TARGD = {al: TARGETS[al] for al in ORIG}
    PTSd = {al: PTS[al] for al in ORIG}
    r = run_scenario(dict(new=(0, Rv)), TARGD, PTSd)
    eta_r_r.append(r['eta_r']); eta_k_r.append(r['eta_k'])
ax.plot(Rs, eta_r_r, marker='o', color='#1f9e89', linewidth=1.6, label='重调度')
ax.plot(Rs, eta_k_r, marker='s', color='#fdae61', linewidth=1.6, label='保持原方案')
ax.set_xlabel('新增告警水平距离 R (m)', fontsize=10); ax.set_ylabel('完成率 η', fontsize=10)
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-13_新增告警距离影响')

# 图4-14 故障车辆影响矩阵（故障车 × 告警损失）
fig, ax = plt.subplots(figsize=(7.0, 3.6))
M = np.zeros((len(ORIG), len(VEHICLE_ORDER)))
for v in VEHICLE_ORDER:
    serving = [al for al in ORIG if any(tk['alert'] == al for tk in tasks[v])]
    for al in serving:
        M[ORIG.index(al), VEHICLE_ORDER.index(v)] = 1
mesh = safe_heatmap(ax, M, row_labels=ORIG, col_labels=VEHICLE_ORDER, cmap='YlGnBu', linewidth=0.5, fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j+0.5, i+0.5, f'{int(M[i,j])}', ha='center', va='center', fontsize=11)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('是否服务该告警', fontsize=9)
save_fig(fig, '图4-14_车辆告警服务矩阵')

# 图4-15 组合扰动 t_last 对比
fig, ax = plt.subplots(figsize=(7.0, 4.4))
x = np.arange(len(comb_res))
ax.bar(x-0.2, [r['tlast_r'] for r in comb_res], width=0.38, color='#1f9e89', edgecolor='black', linewidth=0.6, label='重调度')
ax.bar(x+0.2, [r['tlast_k'] for r in comb_res], width=0.38, color='#fdae61', edgecolor='black', linewidth=0.6, label='保持原方案')
ax.set_xticks(x); ax.set_xticklabels([f'C{i+1}' for i in range(len(comb_res))], fontsize=10)
ax.set_ylabel('t_last (s)', fontsize=10); ax.set_ylim(0, HORIZON+3)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-15_组合扰动t_last对比')

# 图4-16 全部场景 η 与 t_last 散点
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.scatter([r['eta_r'] for r in results], [r['tlast_r'] for r in results], color='#1f9e89', s=50, label='重调度', edgecolors='black', linewidth=0.5)
ax.scatter([r['eta_k'] for r in results], [r['tlast_k'] for r in results], color='#fdae61', s=50, marker='s', label='保持原方案', edgecolors='black', linewidth=0.5)
ax.set_xlabel('完成率 η', fontsize=10); ax.set_ylabel('t_last (s)', fontsize=10)
ax.set_xlim(0.3, 1.05); ax.set_ylim(0, HORIZON+3)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图4-16_η与t_last散点')

# 图4-17 场景汇总表图（热力：场景×指标）
fig, ax = plt.subplots(figsize=(8.6, 5.2))
n = len(results)
M = np.zeros((n, 4))
for i, r in enumerate(results):
    M[i, 0] = r['N']; M[i, 1] = r['eta_r']; M[i, 2] = r['eta_k']; M[i, 3] = r['tlast_r']
data = M
mesh = safe_heatmap(ax, data, row_labels=[f'S{i+1}' for i in range(n)],
                    col_labels=['N', 'η_重调度', 'η_原方案', 't_last重调度'], cmap='YlGnBu', linewidth=0.4, fontsize=8)
for i in range(n):
    for j in range(4):
        ax.text(j+0.5, i+0.5, f'{data[i,j]:.2f}' if j < 4 else '', ha='center', va='center', fontsize=7)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('数值', fontsize=9)
save_fig(fig, '图4-17_场景汇总热力')

# 图4-18 重调度收益与 t_rec
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for r in comm_res:
    ax.scatter([r['scenario']['t_rec']], [r['eta_r']], color='#1f9e89', s=60, zorder=3)
    ax.scatter([r['scenario']['t_rec']], [r['eta_k']], color='#fdae61', s=60, marker='s', zorder=3)
ax.set_xlabel('t_rec (s)', fontsize=10); ax.set_ylabel('完成率 η', fontsize=10)
ax.set_ylim(0, 1.15)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图4-18_通信恢复时刻散点')

# 图4-19 重调度车辆复用（某组合场景 C4）
if comb_res:
    rc = comb_res[-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if rc['routes_r'] is not None:
        M = np.zeros((len(rc['remaining']), len(rc['vehicles'])))
        vix = {v: i for i, v in enumerate(rc['vehicles'])}
        for v in rc['vehicles']:
            for al in rc['routes_r'].get(v, []):
                M[rc['remaining'].index(al), vix[v]] += 1
        mesh = safe_heatmap(ax, M, row_labels=rc['remaining'], col_labels=rc['vehicles'], cmap='YlGnBu', linewidth=0.5, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j+0.5, i+0.5, f'{int(M[i,j])}', ha='center', va='center', fontsize=11)
        cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('任务数', fontsize=9)
    save_fig(fig, '图4-19_组合场景车辆复用')

# 图4-20 重调度策略对比汇总（η改善条形）
fig, ax = plt.subplots(figsize=(10.0, 4.4))
x = np.arange(len(results))
deta = [r['eta_r'] - r['eta_k'] for r in results]
colors = ['#1f9e89' if d >= 0 else '#d62728' for d in deta]
bars = ax.bar(x, deta, width=0.55, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'S{i+1}' for i in range(n)], fontsize=8, rotation=45)
ax.set_ylabel('Δη（重调度−原方案）', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图4-20_重调度收益汇总')

# ---------- 结果输出 ----------
rows = []
for i, r in enumerate(results):
    sc = r['scenario']
    label = ''
    if 'new' in sc: label += f'新增(θ={sc["new"][0]},R={sc["new"][1]}) '
    if 'failed' in sc: label += f'故障{sc["failed"]} '
    if 't_rec' in sc: label += f'通信t_rec={sc["t_rec"]} '
    rows.append([i+1, label.strip(), r['N'], f'{r["eta_r"]:.3f}', f'{r["tlast_r"]:.2f}',
                 f'{r["eta_k"]:.3f}', f'{r["tlast_k"]:.2f}', f'{r["eta_r"]-r["eta_k"]:+.3f}'])
sc_df = pd.DataFrame(rows, columns=['场景', '扰动', 'N', 'η_重调度', 't_last_重调度(s)', 'η_原方案', 't_last_原方案(s)', 'Δη'])
save_csv(sc_df, '问题四_场景结果.csv')

print('\n问题四求解完成。')
