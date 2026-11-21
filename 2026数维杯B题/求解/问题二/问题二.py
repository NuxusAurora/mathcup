# -*- coding: utf-8 -*-
"""
问题二：两站交会定位与多目标车辆指派
 (1) 两站定位精度模型 + T2 最优站对（行驶 ≤30 s）
 (2) 并发告警 T1/T2/T3 车辆复用与路径规划（min 最大观测开始时间）+ 最近指派启发式基准
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOLVE_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _SOLVE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import *
from _geometry import *
from itertools import permutations, product

import matplotlib.colors as mcolors

# ---------- 常数 ----------
SIG_DEG = 1.0
SIG_RAD = np.radians(SIG_DEG)
TRAVEL_MAX = 30.0
DIST_MAX = TRAVEL_MAX * SPEED

ALERT_NAMES = ['T1', 'T2', 'T3']
ALERT_COLOR = {'T1': '#d62728', 'T2': '#1f77b4', 'T3': '#2ca02c'}


def rmin_of(target):
    return (target[2] - CAM_H) / np.tan(np.radians(PITCH_MAX))


# ================= 可观测区（向量化） =================
def obs_grid(target, gs=1.0):
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
    return X, Y, disk, pitch, blocked, obs, r, phi


def draw_overlay(ax, target=None):
    for b, nm in zip(BUILDINGS, BUILDING_NAMES):
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                               facecolor='#6b6b6b', edgecolor='black', linewidth=0.9, zorder=5))
        ax.text((b[0] + b[1]) / 2, (b[3] + b[2]) / 2, nm, ha='center', va='center',
                color='white', fontsize=9, zorder=6, fontweight='bold')
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='black', linewidth=1.2, zorder=4)
    ax.plot(*CENTER, marker='o', color='black', markersize=5, zorder=7)
    ax.text(CENTER[0] + 1.5, CENTER[1] + 1.5, 'O', fontsize=10, zorder=7)
    if target is not None:
        ax.plot(target[0], target[1], marker='^', color='#d62728', markersize=9, zorder=8)
        ax.text(target[0] + 1.5, target[1] + 1.5, 'T', fontsize=10, color='#d62728', zorder=8, fontweight='bold')
    for k, v in VEHICLES.items():
        ax.plot(v[0], v[1], marker='s', color='#1f77b4', markersize=6, zorder=8)
        ax.text(v[0] + 1.2, v[1] + 1.2, k, fontsize=8, color='#1f77b4', zorder=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
    ax.tick_params(labelsize=8, direction='in')
    despine(ax)


# ================= 问题 2(1)：交会定位精度 =================
T2 = TARGETS['T2']
X2, Y2, disk2, pitch2, blocked2, obs2, r2, phi2 = obs_grid(T2, gs=1.0)
print(f'[Q2(1)] T2 可观测区面积 ≈ {obs2.sum():.0f} m²（1m网格）')

r2min = rmin_of(T2)
S_W = np.array([T2[0] - r2min, T2[1]])
S_N = np.array([T2[0], T2[1] + r2min])
print(f'[Q2(1)] 解析 r_min(T2) = {r2min:.4f} m')
print(f'[Q2(1)] 西站 {np.round(S_W,3)} 可观测={is_observable(S_W[0], S_W[1], T2)}, 北站 {np.round(S_N,3)} 可观测={is_observable(S_N[0], S_N[1], T2)}')


def precision_of(pa, pb, target):
    r1 = np.hypot(target[0] - pa[0], target[1] - pa[1])
    r2 = np.hypot(target[0] - pb[0], target[1] - pb[1])
    a1 = azimuth_from_station(pa[0], pa[1], target)
    a2 = azimuth_from_station(pb[0], pb[1], target)
    g = inter_angle(a1, a2)
    P = abs(np.sin(np.radians(g))) / np.sqrt(r1 ** 2 + r2 ** 2)
    return P, g, r1, r2, a1, a2


P_opt, g_opt, r1_opt, r2_opt, az1_opt, az2_opt = precision_of(S_W, S_N, T2)
base_W = np.hypot(S_W[0] - S_N[0], S_W[1] - S_N[1])
sig_p = SIG_RAD * np.sqrt(r1_opt ** 2 + r2_opt ** 2) / abs(np.sin(np.radians(g_opt)))
print(f'[Q2(1)] 最优站对: 西站({S_W[0]:.3f},{S_W[1]:.3f},az={az1_opt:.1f}°), 北站({S_N[0]:.3f},{S_N[1]:.3f},az={az2_opt:.1f}°)')
print(f'[Q2(1)] 基线 B={base_W:.3f} m, 交会角 γ={g_opt:.1f}°, r1={r1_opt:.3f}, r2={r2_opt:.3f}')
print(f'[Q2(1)] 精度指标 P={P_opt:.5f}, σ_p={sig_p*100:.3f} cm')

idx = np.argwhere(obs2 & (r2 <= 25.0))
cand_xy = np.column_stack([X2[idx[:, 0], idx[:, 1]], Y2[idx[:, 0], idx[:, 1]]])
cand_r = r2[idx[:, 0], idx[:, 1]]
cand_az = np.array([azimuth_from_station(c[0], c[1], T2) for c in cand_xy])
n = len(cand_xy)
daz = np.abs(cand_az[:, None] - cand_az[None, :]) % 360.0
daz = np.minimum(daz, 360.0 - daz)
Pmat = np.abs(np.sin(np.radians(daz))) / np.sqrt(cand_r[:, None] ** 2 + cand_r[None, :] ** 2)
iu = np.triu_indices(n, 1)
Pvals = Pmat[iu]; gvals = daz[iu]
mask = (gvals >= 30) & (gvals <= 150)
valid_P = Pvals[mask]; valid_g = gvals[mask]
print(f'[Q2(1)] 网格枚举: 候选站数={n}, 最优 P={valid_P.max():.5f} (解析 {P_opt:.5f})')


def pair_travel(pa, pb):
    best = (np.inf, None, None)
    keys = VEHICLE_ORDER
    da = [np.hypot(pa[0] - VEHICLES[v][0], pa[1] - VEHICLES[v][1]) for v in keys]
    db = [np.hypot(pb[0] - VEHICLES[v][0], pb[1] - VEHICLES[v][1]) for v in keys]
    for vi in range(5):
        for vj in range(5):
            if vi == vj:
                continue
            if da[vi] > DIST_MAX or db[vj] > DIST_MAX:
                continue
            mx = max(da[vi], db[vj]) / SPEED
            if mx < best[0]:
                best = (mx, keys[vi], keys[vj])
    if best[1] is None:
        return None
    return best


trv_opt = pair_travel(S_W, S_N)
print(f'[Q2(1)] 最优站对车辆指派: {trv_opt[1]}→西站, {trv_opt[2]}→北站, 最大行驶时间={trv_opt[0]:.2f} s')

# ================= 问题 2(2)：调度 =================
def obs_points(target, gs=1.0):
    X, Y, _, _, _, obs, r, _ = obs_grid(target, gs)
    idx = np.argwhere(obs)
    pts = []
    for i, j in idx:
        x, y = X[i, j], Y[i, j]
        pts.append((x, y, azimuth_from_station(x, y, target), r[i, j]))
    return np.array(pts)


def build_options(alert_name, K=60, g_lo=40.0, g_hi=140.0, r_cap=None):
    """为告警生成 (va,vb,sa,sb,max行驶时间,γ) 候选选项：每车辆就近选站，站对γ∈[g_lo,g_hi]。"""
    T = TARGETS[alert_name]
    pts = obs_points(T)
    if r_cap is not None:
        pts = pts[pts[:, 3] <= r_cap]
    cand_by_v = {}
    for v in VEHICLE_ORDER:
        d = np.hypot(pts[:, 0] - VEHICLES[v][0], pts[:, 1] - VEHICLES[v][1])
        order = np.argsort(d)[:K]
        cand_by_v[v] = pts[order]
    options = []
    for a in range(5):
        for b in range(a + 1, 5):
            va, vb = VEHICLE_ORDER[a], VEHICLE_ORDER[b]
            best = None
            for sa in cand_by_v[va]:
                for sb in cand_by_v[vb]:
                    g = inter_angle(sa[2], sb[2])
                    if not (g_lo <= g <= g_hi):
                        continue
                    da = np.hypot(sa[0] - VEHICLES[va][0], sa[1] - VEHICLES[va][1])
                    db = np.hypot(sb[0] - VEHICLES[vb][0], sb[1] - VEHICLES[vb][1])
                    mx = max(da, db) / SPEED
                    if best is None or mx < best[0] - 1e-9:
                        best = (mx, g, sa[:2], sb[:2])
            if best is not None:
                options.append((va, vb, best[2], best[3], best[0], best[1]))
    return options


def simulate(routes, st_of, alerts):
    pos = {v: VEHICLES[v].copy() for v in VEHICLE_ORDER}
    starts = {a: 0.0 for a in alerts}
    for _ in range(100):
        new = {}
        for v in VEHICLE_ORDER:
            t = 0.0
            p = VEHICLES[v].copy()
            for al in routes[v]:
                st = st_of[(v, al)]
                drive = np.hypot(st[0] - p[0], st[1] - p[1]) / SPEED
                arrive = t + drive
                rot = rotate_time(azimuth_from_station(st[0], st[1], TARGETS[al]),
                                  pitch_deg(st[0], st[1], TARGETS[al]))
                ready = arrive + rot
                new[al] = max(new.get(al, -np.inf), ready)
                t = starts[al] + OBS_TIME + rot
                p = st
        if max(abs(new[a] - starts[a]) for a in alerts) < 1e-9:
            break
        starts = new
    events = {v: [] for v in VEHICLE_ORDER}
    for v in VEHICLE_ORDER:
        t = 0.0
        p = VEHICLES[v].copy()
        for al in routes[v]:
            st = st_of[(v, al)]
            drive = np.hypot(st[0] - p[0], st[1] - p[1]) / SPEED
            events[v].append(('drive', t, t + drive, al))
            arrive = t + drive
            rot = rotate_time(azimuth_from_station(st[0], st[1], TARGETS[al]),
                              pitch_deg(st[0], st[1], TARGETS[al]))
            events[v].append(('rotate', arrive, arrive + rot, al))
            ready = arrive + rot
            si = starts[al]
            if si > ready + 1e-9:
                events[v].append(('wait', ready, si, al))
            events[v].append(('observe', si, si + OBS_TIME, al))
            events[v].append(('reset', si + OBS_TIME, si + OBS_TIME + rot, al))
            t = si + OBS_TIME + rot
            p = st
    return starts, events


def schedule_optimal(alerts, g_lo=40.0, g_hi=140.0, r_cap=None):
    opts = {al: build_options(al, g_lo=g_lo, g_hi=g_hi, r_cap=r_cap) for al in alerts}
    veh = VEHICLE_ORDER
    best = (np.inf, None, None, None)
    for combo in product(*[opts[al] for al in alerts]):
        routes = {v: [] for v in veh}
        st_of = {}
        for al, opt in zip(alerts, combo):
            va, vb, sa, sb, tr, g = opt
            routes[va].append(al); routes[vb].append(al)
            st_of[(va, al)] = np.array(sa); st_of[(vb, al)] = np.array(sb)
        order_opts = [list(permutations(routes[v])) for v in veh]
        for ordc in product(*order_opts):
            routes2 = {veh[k]: [t for t in ordc[k]] for k in range(len(veh))}
            starts, _ = simulate(routes2, st_of, alerts)
            mk = max(starts.values())
            if mk < best[0]:
                best = (mk, routes2, st_of, starts)
    return best, opts


def nearest_heuristic(alerts, g_lo=40.0, g_hi=140.0, r_cap=None):
    opts = {al: build_options(al, g_lo=g_lo, g_hi=g_hi, r_cap=r_cap) for al in alerts}
    veh = VEHICLE_ORDER
    pos = {v: VEHICLES[v].copy() for v in veh}
    t_avail = {v: 0.0 for v in veh}
    routes = {v: [] for v in veh}
    st_of = {}
    starts = {}
    for al in alerts:
        best_opt = None; best_start = np.inf
        for opt in opts[al]:
            va, vb, sa, sb, tr, g = opt
            rva = t_avail[va] + np.hypot(sa[0] - pos[va][0], sa[1] - pos[va][1]) / SPEED \
                  + rotate_time(azimuth_from_station(sa[0], sa[1], TARGETS[al]), pitch_deg(sa[0], sa[1], TARGETS[al]))
            rvb = t_avail[vb] + np.hypot(sb[0] - pos[vb][0], sb[1] - pos[vb][1]) / SPEED \
                  + rotate_time(azimuth_from_station(sb[0], sb[1], TARGETS[al]), pitch_deg(sb[0], sb[1], TARGETS[al]))
            s = max(rva, rvb)
            if s < best_start:
                best_start = s; best_opt = opt
        va, vb, sa, sb, tr, g = best_opt
        starts[al] = best_start
        for v, st in [(va, sa), (vb, sb)]:
            routes[v].append(al)
            st_of[(v, al)] = np.array(st)
            rot = rotate_time(azimuth_from_station(st[0], st[1], TARGETS[al]), pitch_deg(st[0], st[1], TARGETS[al]))
            pos[v] = np.array(st)
            t_avail[v] = best_start + OBS_TIME + rot
    _, events = simulate(routes, st_of, alerts)
    return max(starts.values()), starts, routes, st_of, events


(mk_opt, routes_opt, st_opt, starts_opt), opts_opt = schedule_optimal(ALERT_NAMES)
mk_heur, starts_heur, routes_heur, st_heur, events_heur = nearest_heuristic(ALERT_NAMES)
_, events_opt = simulate(routes_opt, st_opt, ALERT_NAMES)

print(f'\n[Q2(2)] 最优 makespan = {mk_opt:.3f} s')
for v in VEHICLE_ORDER:
    if routes_opt[v]:
        for al in routes_opt[v]:
            s = st_opt[(v, al)]
            print(f'  车辆 {v} → 告警 {al}: 站({s[0]:.2f},{s[1]:.2f})')
for al in ALERT_NAMES:
    print(f'  告警 {al}: 观测开始 = {starts_opt[al]:.3f} s')
print(f'[Q2(2)] 最近指派启发式 makespan = {mk_heur:.3f} s')


def serving(routes, al):
    return [v for v in VEHICLE_ORDER if al in routes[v]]


# ================= 绘图 =================
CMAP2 = 'Spectral_r'

# 图2-1 T2 可观测区域
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X2, Y2, np.where(obs2, 1.0, 0.0), cmap=mcolors.ListedColormap(['#e9e9e9', '#1f9e89']), shading='auto')
draw_overlay(ax, T2)
save_fig(fig, '图2-1_T2可观测区域掩膜')

# 图2-2 交会定位几何示意
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.arrow(T2[0], T2[1], -8, 0, head_width=0.5, head_length=0.6, fc='#1f77b4', ec='#1f77b4', linewidth=1.4)
ax.arrow(T2[0], T2[1], 0, 8, head_width=0.5, head_length=0.6, fc='#d62728', ec='#d62728', linewidth=1.4)
ax.plot(T2[0], T2[1], marker='^', color='#d62728', markersize=11)
ax.plot(S_W[0], S_W[1], marker='s', color='#1f77b4', markersize=8)
ax.plot(S_N[0], S_N[1], marker='s', color='#d62728', markersize=8)
ax.plot([S_W[0], S_N[0]], [S_W[1], S_N[1]], color='#444444', linewidth=1.2)
ax.text((S_W[0] + S_N[0]) / 2 + 0.4, (S_W[1] + S_N[1]) / 2 + 0.4, f'基线 B={base_W:.2f} m', fontsize=9)
ax.annotate(f'γ={g_opt:.0f}°', xy=(T2[0] - 3.0, T2[1] + 3.2), fontsize=10, color='#444444')
ax.text(S_W[0] - 0.4, S_W[1] - 1.4, '西站(az=0°)', fontsize=8, color='#1f77b4', ha='right')
ax.text(S_N[0] + 0.5, S_N[1] + 0.6, '北站(az=270°)', fontsize=8, color='#d62728')
ax.set_xlim(0, 22); ax.set_ylim(65, 86)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-2_两站交会定位几何示意')

# 图2-3 P 随交会角变化
fig, ax = plt.subplots(figsize=(7.0, 4.4))
gg = np.linspace(5, 175, 400)
for rv, c in zip([5.82, 8.0, 12.0, 20.0], ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']):
    PP = np.abs(np.sin(np.radians(gg))) / np.sqrt(2 * rv ** 2)
    ax.plot(gg, PP, color=c, linewidth=1.6, label=f'r={rv} m')
ax.axvline(90, color='#888888', linewidth=0.8, linestyle=':')
ax.set_xlabel('交会角 γ (deg)', fontsize=10); ax.set_ylabel('精度指标 P (1/m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图2-3_精度指标随交会角变化')

# 图2-4 P 随距离变化
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rr2 = np.linspace(5, 60, 400)
for gv, c in zip([30.0, 60.0, 90.0], ['#2ca02c', '#1f77b4', '#d62728']):
    PP = np.abs(np.sin(np.radians(gv))) / np.sqrt(2 * rr2 ** 2)
    ax.plot(rr2, PP, color=c, linewidth=1.6, label=f'γ={gv:.0f}°')
ax.set_xlabel('站点距离 r (m)', fontsize=10); ax.set_ylabel('精度指标 P (1/m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图2-4_精度指标随距离变化')

# 图2-5 站对精度热力
fig, ax = plt.subplots(figsize=(7.0, 4.4))
gs_ = np.linspace(10, 170, 81); rs_ = np.linspace(5, 40, 71)
GG, RR = np.meshgrid(gs_, rs_)
PP = np.abs(np.sin(np.radians(GG))) / np.sqrt(2 * RR ** 2)
data_hm = PP.T if PP.shape[0] > PP.shape[1] else PP
rlab = rs_ if data_hm.shape[0] == len(rs_) else gs_
clab = gs_ if data_hm.shape[0] == len(rs_) else rs_
mesh = safe_heatmap(ax, data_hm, row_labels=[f'{v:.0f}' for v in rlab],
                    col_labels=[f'{v:.0f}' for v in clab], cmap=CMAP2, linewidth=0.3, fontsize=8)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('P (1/m)', fontsize=9)
save_fig(fig, '图2-5_站对精度热力图')

# 图2-6 最优站对几何标注
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X2, Y2, np.where(obs2, 1.0, 0.0), cmap=mcolors.ListedColormap(['#e9e9e9', '#1f9e89']), shading='auto')
draw_overlay(ax, T2)
ax.plot(S_W[0], S_W[1], marker='s', color='#1f77b4', markersize=9, zorder=9)
ax.plot(S_N[0], S_N[1], marker='s', color='#d62728', markersize=9, zorder=9)
ax.plot([S_W[0], S_N[0]], [S_W[1], S_N[1]], color='#444444', linewidth=1.4, zorder=8)
ax.plot([S_W[0], T2[0]], [S_W[1], T2[1]], color='#1f77b4', linewidth=1.1, linestyle='--', zorder=8)
ax.plot([S_N[0], T2[0]], [S_N[1], T2[1]], color='#d62728', linewidth=1.1, linestyle='--', zorder=8)
ax.text(S_W[0] + 0.5, S_W[1] - 1.6, f'S1({S_W[0]:.1f},{S_W[1]:.1f})', fontsize=8, color='#1f77b4')
ax.text(S_N[0] + 0.5, S_N[1] + 1.0, f'S2({S_N[0]:.1f},{S_N[1]:.1f})', fontsize=8, color='#d62728')
save_fig(fig, '图2-6_最优站对几何标注')

# 图2-7 σ_p 随 γ 退化
fig, ax = plt.subplots(figsize=(7.0, 4.4))
gg3 = np.linspace(1, 179, 500)
sig_deg = SIG_RAD * np.sqrt(2 * 5.82 ** 2) / np.abs(np.sin(np.radians(gg3)))
ax.semilogy(gg3, sig_deg, color='#d62728', linewidth=1.6)
ax.axvline(90, color='#888888', linewidth=0.8, linestyle=':')
ax.set_xlabel('交会角 γ (deg)', fontsize=10); ax.set_ylabel('位置误差 σ_p (m，对数轴)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-7_位置误差随交会角退化')

# 图2-8 基线长度与定位误差
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rr4 = np.linspace(5, 50, 400)
BB = np.sqrt(2) * rr4
sig4 = SIG_RAD * np.sqrt(2 * rr4 ** 2) / 1.0
ax.plot(BB, sig4 * 100, color='#1f77b4', linewidth=1.6)
ax.scatter([base_W], [sig_p * 100], color='#d62728', s=60, zorder=5)
ax.set_xlabel('基线长度 B (m)', fontsize=10); ax.set_ylabel('位置误差 σ_p (cm)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-8_基线长度与定位误差')

# 图2-9 T2 俯仰角分布
fig, ax = plt.subplots(figsize=(6.8, 6.0))
mesh = ax.pcolormesh(X2, Y2, np.ma.masked_invalid(np.where(obs2, phi2, np.nan)), cmap='RdYlBu_r', vmin=0, vmax=80, shading='auto')
draw_overlay(ax, T2)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('俯仰角 (deg)', fontsize=9)
save_fig(fig, '图2-9_T2可观测区俯仰角分布')

# 图2-10 T2 可观测区 r 分布
fig, ax = plt.subplots(figsize=(6.8, 6.0))
mesh = ax.pcolormesh(X2, Y2, np.ma.masked_invalid(np.where(obs2, r2, np.nan)), cmap=CMAP2, vmin=0, vmax=40, shading='auto')
draw_overlay(ax, T2)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('水平距离 r (m)', fontsize=9)
save_fig(fig, '图2-10_T2可观测区距离分布')

# 图2-11 三告警站点与车辆指派总览
fig, ax = plt.subplots(figsize=(7.6, 6.4))
for b, nm in zip(BUILDINGS, BUILDING_NAMES):
    ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], facecolor='#6b6b6b', edgecolor='black', linewidth=0.9, zorder=2))
    ax.text((b[0] + b[1]) / 2, (b[3] + b[2]) / 2, nm, ha='center', va='center', color='white', fontsize=9, zorder=3, fontweight='bold')
ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='black', linewidth=1.2, zorder=2)
ax.plot(*CENTER, marker='o', color='black', markersize=5, zorder=4)
for al in ALERT_NAMES:
    T = TARGETS[al]
    ax.plot(T[0], T[1], marker='^', color=ALERT_COLOR[al], markersize=11, zorder=5)
    ax.text(T[0] + 1.5, T[1] + 1.5, al, fontsize=10, color=ALERT_COLOR[al], zorder=5, fontweight='bold')
    for v in serving(routes_opt, al):
        s = st_opt[(v, al)]
        ax.plot(s[0], s[1], marker='o', color=ALERT_COLOR[al], markersize=7, zorder=6)
for v in VEHICLE_ORDER:
    ax.plot(VEHICLES[v][0], VEHICLES[v][1], marker='s', color='#444444', markersize=6, zorder=7)
    ax.text(VEHICLES[v][0] + 1.0, VEHICLES[v][1] + 1.0, v, fontsize=8, color='#444444', zorder=7)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-11_三告警站点与车辆总览')


def draw_gantt(events, fname):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ypos = {v: len(VEHICLE_ORDER) - 1 - i for i, v in enumerate(VEHICLE_ORDER)}
    kind_color = {'drive': '#d9d9d9', 'rotate': '#e6e6e6', 'wait': '#f5f5f5', 'observe': None, 'reset': '#c8c8c8'}
    for v in VEHICLE_ORDER:
        y = ypos[v]
        for (kind, t0, t1, al) in events[v]:
            c = ALERT_COLOR[al] if kind == 'observe' else kind_color[kind]
            ax.barh(y, t1 - t0, left=t0, height=0.55, color=c, edgecolor='black', linewidth=0.5)
    for al in ALERT_NAMES:
        ax.barh(0, 0, left=0, color=ALERT_COLOR[al], label=f'{al}观测')
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(list(ypos.keys()), fontsize=9)
    ax.set_xlabel('时间 (s)', fontsize=10); ax.set_ylabel('车辆', fontsize=10)
    ax.tick_params(labelsize=9, direction='in'); despine(ax)
    ax.set_xlim(0, max(max(t1 for _, t0, t1, _ in events[v]) for v in VEHICLE_ORDER) * 1.06)
    top_legend(fig, [ax])
    save_fig(fig, fname)


draw_gantt(events_opt, '图2-12_最优调度甘特时间轴')
draw_gantt(events_heur, '图2-17_启发式调度甘特时间轴')

# 图2-13 各车路径
fig, ax = plt.subplots(figsize=(7.6, 6.4))
for b, nm in zip(BUILDINGS, BUILDING_NAMES):
    ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], facecolor='#6b6b6b', edgecolor='black', linewidth=0.9))
    ax.text((b[0] + b[1]) / 2, (b[3] + b[2]) / 2, nm, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='black', linewidth=1.2)
vcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for k, v in enumerate(VEHICLE_ORDER):
    path = [VEHICLES[v].copy()]
    for al in routes_opt[v]:
        path.append(st_opt[(v, al)])
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color=vcolors[k], linewidth=1.6, marker='s', markersize=5, label=v)
for al in ALERT_NAMES:
    T = TARGETS[al]
    ax.plot(T[0], T[1], marker='^', color=ALERT_COLOR[al], markersize=9)
    ax.text(T[0] + 1.2, T[1] + 1.2, al, fontsize=9, color=ALERT_COLOR[al], fontweight='bold')
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图2-13_各车辆行驶路径')

# 图2-14 各车时间轴分解
fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
kind_color = {'drive': '#d9d9d9', 'rotate': '#e6e6e6', 'wait': '#f5f5f5', 'observe': None, 'reset': '#c8c8c8'}
for k, v in enumerate(VEHICLE_ORDER):
    ax = axes[k]
    for (kind, t0, t1, al) in events_opt[v]:
        c = ALERT_COLOR[al] if kind == 'observe' else kind_color[kind]
        ax.barh(0, t1 - t0, left=t0, height=0.5, color=c, edgecolor='black', linewidth=0.4)
    ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
    ax.set_xlabel('时间 (s)', fontsize=8)
    ax.text(0.5, -0.98, v, ha='center', va='top', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8, direction='in'); despine(ax)
save_fig(fig, '图2-14_各车辆时间轴分解')

# 图2-15 观测开始时间对比
fig, ax = plt.subplots(figsize=(7.0, 4.4))
x = np.arange(3); w = 0.35
ax.bar(x - w / 2, [starts_opt[a] for a in ALERT_NAMES], width=w, color='#1f9e89', edgecolor='black', linewidth=0.8, label='优化方案')
ax.bar(x + w / 2, [starts_heur[a] for a in ALERT_NAMES], width=w, color='#fdae61', edgecolor='black', linewidth=0.8, label='最近指派启发式')
for i, a in enumerate(ALERT_NAMES):
    ax.text(i - w / 2, starts_opt[a] + 0.3, f'{starts_opt[a]:.1f}', ha='center', fontsize=8)
    ax.text(i + w / 2, starts_heur[a] + 0.3, f'{starts_heur[a]:.1f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(ALERT_NAMES, fontsize=10)
ax.set_ylabel('观测开始时间 (s)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.6)
top_legend(fig, [ax])
save_fig(fig, '图2-15_观测开始时间对比')

# 图2-16 车辆复用矩阵（行=告警, 列=车辆）
fig, ax = plt.subplots(figsize=(7.0, 3.6))
M = np.zeros((len(ALERT_NAMES), len(VEHICLE_ORDER)))
for v in VEHICLE_ORDER:
    for al in routes_opt[v]:
        M[ALERT_NAMES.index(al), VEHICLE_ORDER.index(v)] += 1
mesh = safe_heatmap(ax, M, row_labels=ALERT_NAMES, col_labels=VEHICLE_ORDER, cmap='YlGnBu', linewidth=0.5, fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j + 0.5, i + 0.5, f'{int(M[i, j])}', ha='center', va='center', fontsize=11, color='black')
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('任务数', fontsize=9)
save_fig(fig, '图2-16_车辆复用矩阵')

# 图2-18 行驶距离对比
def total_drive(routes, st_of):
    td = {}
    for v in VEHICLE_ORDER:
        p = VEHICLES[v].copy(); d = 0.0
        for al in routes[v]:
            st = st_of[(v, al)]
            d += np.hypot(st[0] - p[0], st[1] - p[1]); p = st
        td[v] = d
    return td


d_opt = total_drive(routes_opt, st_opt)
d_heur = total_drive(routes_heur, st_heur)
fig, ax = plt.subplots(figsize=(7.0, 4.4))
x = np.arange(5); w = 0.35
ax.bar(x - w / 2, [d_opt[v] for v in VEHICLE_ORDER], width=w, color='#1f9e89', edgecolor='black', linewidth=0.8, label='优化方案')
ax.bar(x + w / 2, [d_heur[v] for v in VEHICLE_ORDER], width=w, color='#fdae61', edgecolor='black', linewidth=0.8, label='最近指派启发式')
ax.set_xticks(x); ax.set_xticklabels(VEHICLE_ORDER, fontsize=10)
ax.set_ylabel('累计行驶距离 (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图2-18_各车行驶距离对比')

# 图2-19 等待时间对比
def total_wait(events):
    tw = {}
    for v in VEHICLE_ORDER:
        tw[v] = sum(t1 - t0 for kind, t0, t1, _ in events[v] if kind == 'wait')
    return tw


w_opt = total_wait(events_opt); w_heur = total_wait(events_heur)
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.bar(x - w / 2, [w_opt[v] for v in VEHICLE_ORDER], width=w, color='#1f9e89', edgecolor='black', linewidth=0.8, label='优化方案')
ax.bar(x + w / 2, [w_heur[v] for v in VEHICLE_ORDER], width=w, color='#fdae61', edgecolor='black', linewidth=0.8, label='最近指派启发式')
ax.set_xticks(x); ax.set_xticklabels(VEHICLE_ORDER, fontsize=10)
ax.set_ylabel('等待时间 (s)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图2-19_各车等待时间对比')

# 图2-20 站点可观测性与视线
fig, ax = plt.subplots(figsize=(7.6, 6.4))
for b, nm in zip(BUILDINGS, BUILDING_NAMES):
    ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], facecolor='#6b6b6b', edgecolor='black', linewidth=0.9))
    ax.text((b[0] + b[1]) / 2, (b[3] + b[2]) / 2, nm, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='black', linewidth=1.2)
for al in ALERT_NAMES:
    T = TARGETS[al]
    ax.plot(T[0], T[1], marker='^', color=ALERT_COLOR[al], markersize=11)
    ax.text(T[0] + 1.5, T[1] + 1.5, al, fontsize=10, color=ALERT_COLOR[al], fontweight='bold')
    for v in serving(routes_opt, al):
        s = st_opt[(v, al)]
        ax.plot(s[0], s[1], marker='o', color=ALERT_COLOR[al], markersize=7)
        ax.plot([s[0], T[0]], [s[1], T[1]], color=ALERT_COLOR[al], linewidth=0.9, linestyle='--', alpha=0.7)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-20_站点可观测性与视线')

# 图2-21 makespan 与 γ 阈值灵敏度
fig, ax = plt.subplots(figsize=(7.0, 4.4))
glo_list = [20, 30, 40, 50, 60]
mks = []
for gl in glo_list:
    (mk, _, _, _), _ = schedule_optimal(ALERT_NAMES, g_lo=gl, g_hi=180 - gl)
    mks.append(mk)
ax.plot(glo_list, mks, marker='o', color='#1f77b4', linewidth=1.6)
ax.set_xlabel('交会角下限 γ_min (deg)', fontsize=10); ax.set_ylabel('最优 makespan (s)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-21_makespan与交会角阈值灵敏度')

# 图2-22 车辆时间占用构成
fig, ax = plt.subplots(figsize=(7.0, 4.4))
kinds = ['drive', 'rotate', 'wait', 'observe', 'reset']
tot_kind = {k: 0.0 for k in kinds}
for v in VEHICLE_ORDER:
    for kind, t0, t1, _ in events_opt[v]:
        tot_kind[kind] += t1 - t0
kind_labels = ['行驶', '云台转动', '等待', '联合观测', '云台复位']
kind_colors = ['#d9d9d9', '#e6e6e6', '#f5f5f5', '#1f9e89', '#c8c8c8']
vals = [tot_kind[k] for k in kinds]
bars = ax.bar(range(5), vals, width=0.45, color=kind_colors, edgecolor='black', linewidth=0.8)
for b_, v_ in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v_ + 0.3, f'{v_:.1f}', ha='center', fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(kind_labels, fontsize=9)
ax.set_ylabel('五车累计时长 (s)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.6)
save_fig(fig, '图2-22_车辆时间占用构成')

# 图2-23 速度-精度权衡：makespan 随站距上限 r_cap 变化
rcaps = [15, 20, 25, 30, 40, 50, 60, None]
mk_cap = []
for rc in rcaps:
    (mk, _, _, _), _ = schedule_optimal(ALERT_NAMES, r_cap=rc)
    mk_cap.append(mk)
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xlab = ['15', '20', '25', '30', '40', '50', '60', '∞']
ax.plot(xlab, mk_cap, marker='o', color='#1f77b4', linewidth=1.6)
ax.axhline(mk_opt, color='#d62728', linewidth=1.0, linestyle='--')
ax.set_xlabel('观测站最大水平距离 r_cap (m)', fontsize=10)
ax.set_ylabel('最优 makespan (s)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图2-23_makespan随站距上限权衡')

# r_cap=40 的精度约束调度（供论文对比）
mk_cap40, routes_cap40, st_cap40, starts_cap40, _ = None, None, None, None, None
(mk_cap40, routes_cap40, st_cap40, starts_cap40), _ = schedule_optimal(ALERT_NAMES, r_cap=40.0)
print(f'[Q2(2)] r≤40m 精度约束调度 makespan = {mk_cap40:.3f} s')
for v in VEHICLE_ORDER:
    for al in routes_cap40[v]:
        s = st_cap40[(v, al)]
        print(f'  车辆 {v} → 告警 {al}: 站({s[0]:.2f},{s[1]:.2f})')

# 保存 r≤40m 精度约束调度时间轴（问题四基线）
rows_cap = []
for v in VEHICLE_ORDER:
    p = VEHICLES[v].copy(); t = 0.0
    for al in routes_cap40[v]:
        st = st_cap40[(v, al)]
        drive = np.hypot(st[0] - p[0], st[1] - p[1]) / SPEED
        arrive = t + drive
        rot = rotate_time(azimuth_from_station(st[0], st[1], TARGETS[al]), pitch_deg(st[0], st[1], TARGETS[al]))
        si = starts_cap40[al]
        rows_cap.append([v, al, f'({st[0]:.2f},{st[1]:.2f})', f'{t:.3f}', f'{arrive:.3f}', f'{rot:.3f}',
                         f'{si:.3f}', f'{si + OBS_TIME:.3f}', f'{si + OBS_TIME + rot:.3f}'])
        t = si + OBS_TIME + rot; p = st
    if not routes_cap40[v]:
        rows_cap.append([v, '-', '-', '-', '-', '-', '-', '-', '-'])
cap_df = pd.DataFrame(rows_cap, columns=['车辆', '告警', '观测站', '出发时刻(s)', '到站时刻(s)', '云台转动(s)', '观测开始(s)', '观测结束(s)', '复位完成(s)'])
save_csv(cap_df, '问题二_2_精度约束调度时间轴.csv')

# ================= 结果输出 =================
q1_df = pd.DataFrame([
    ['最优站 S1(西)', f'({S_W[0]:.3f}, {S_W[1]:.3f})', f'方位角 {az1_opt:.1f}°'],
    ['最优站 S2(北)', f'({S_N[0]:.3f}, {S_N[1]:.3f})', f'方位角 {az2_opt:.1f}°'],
    ['基线长度 B (m)', f'{base_W:.3f}', ''],
    ['交会角 γ (deg)', f'{g_opt:.1f}', ''],
    ['站点距离 r1/r2 (m)', f'{r1_opt:.3f} / {r2_opt:.3f}', ''],
    ['精度指标 P (1/m)', f'{P_opt:.5f}', ''],
    ['定位误差 σ_p (cm)', f'{sig_p * 100:.3f}', ''],
    ['车辆指派', f'{trv_opt[1]}→S1, {trv_opt[2]}→S2', f'最大行驶 {trv_opt[0]:.2f} s'],
], columns=['指标', '值', '备注'])
save_csv(q1_df, '问题二_1_T2最优站对.csv')

rows = []
for v in VEHICLE_ORDER:
    p = VEHICLES[v].copy(); t = 0.0
    for al in routes_opt[v]:
        st = st_opt[(v, al)]
        drive = np.hypot(st[0] - p[0], st[1] - p[1]) / SPEED
        arrive = t + drive
        rot = rotate_time(azimuth_from_station(st[0], st[1], TARGETS[al]), pitch_deg(st[0], st[1], TARGETS[al]))
        si = starts_opt[al]
        rows.append([v, al, f'({st[0]:.2f},{st[1]:.2f})', f'{t:.2f}', f'{arrive:.2f}', f'{rot:.2f}',
                     f'{si:.2f}', f'{si + OBS_TIME:.2f}', f'{si + OBS_TIME + rot:.2f}'])
        t = si + OBS_TIME + rot; p = st
    if not routes_opt[v]:
        rows.append([v, '-', '-', '-', '-', '-', '-', '-', '-'])
tl_df = pd.DataFrame(rows, columns=['车辆', '告警', '观测站', '出发时刻(s)', '到站时刻(s)', '云台转动(s)', '观测开始(s)', '观测结束(s)', '复位完成(s)'])
save_csv(tl_df, '问题二_2_最优调度时间轴.csv')

cmp_df = pd.DataFrame([
    ['优化方案 makespan (s)', f'{mk_opt:.3f}'],
    ['最近指派启发式 makespan (s)', f'{mk_heur:.3f}'],
    ['makespan 改进 (s)', f'{mk_heur - mk_opt:.3f}'],
    ['优化方案观测开始', ' / '.join([f'{a}={starts_opt[a]:.2f}' for a in ALERT_NAMES])],
    ['启发式观测开始', ' / '.join([f'{a}={starts_heur[a]:.2f}' for a in ALERT_NAMES])],
    ['复用车辆', ' / '.join([v for v in VEHICLE_ORDER if len(routes_opt[v]) > 1])],
    ['r≤40m精度约束 makespan (s)', f'{mk_cap40:.3f}'],
], columns=['指标', '值'])
save_csv(cmp_df, '问题二_2_方案对比.csv')

print('\n问题二求解完成。')
