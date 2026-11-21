# -*- coding: utf-8 -*-
"""
问题一：单目标 T1 的可观测区域与最优观测站
三约束：车辆几何 / 视线遮挡 / 云台俯仰；目标 max Q=1/r（等价 min r）。
输出：可观测区域图、约束分解图、最优观测站（含并列取舍规则）、CSV。
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOLVE_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _SOLVE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import *
from _geometry import *

import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection

# ---------- 参数 ----------
T = TARGETS['T1']
tx, ty, H = T
R_ANALYTIC = (H - CAM_H) / np.tan(np.radians(PITCH_MAX))   # 43/tan80 ≈ 7.583

# ---------- 网格 ----------
GS = 0.5
xx = np.arange(0.0, 100.0 + GS, GS)
X, Y = np.meshgrid(xx, xx)

# 几何约束
disk = (X - VEH_R >= 0) & (X + VEH_R <= 100) & (Y - VEH_R >= 0) & (Y + VEH_R <= 100)
for b in BUILDINGS:
    dx = np.maximum(np.maximum(b[0] - X, 0.0), X - b[1])
    dy = np.maximum(np.maximum(b[2] - Y, 0.0), Y - b[3])
    disk &= np.hypot(dx, dy) >= VEH_R

# 俯仰约束
r = np.hypot(tx - X, ty - Y)
phi = np.degrees(np.arctan2(H - CAM_H, r))
pitch = (phi >= PITCH_MIN) & (phi <= PITCH_MAX)

# 遮挡约束（向量化 slab）
def los_blocked_grid(X, Y, target):
    tx_, ty_, H_ = target[0], target[1], target[2]
    Xf = X.ravel(); Yf = Y.ravel()
    blocked = np.zeros(Xf.size, dtype=bool)
    z0, z1 = CAM_H, H_
    for b in BUILDINGS:
        xmin, xmax, ymin, ymax, h = b[0], b[1], b[2], b[3], b[4]
        zmin, zmax = 0.0, h
        ok = np.ones(Xf.size, dtype=bool)
        tmin = np.zeros_like(Xf); tmax = np.ones_like(Xf)
        for p0, p1, lo, hi in [(Xf, tx_, xmin, xmax), (Yf, ty_, ymin, ymax)]:
            same = (p1 == p0)
            ok = ok & (~same | ((p0 >= lo) & (p0 <= hi)))
            diff = ~same
            if diff.any():
                denom = np.where(diff, p1 - p0, 1.0)
                t1 = (lo - p0) / denom
                t2 = (hi - p0) / denom
                tlo = np.minimum(t1, t2); thi = np.maximum(t1, t2)
                tmin = np.maximum(tmin, np.where(diff, tlo, tmin))
                tmax = np.minimum(tmax, np.where(diff, thi, tmax))
        ok = ok & (tmin <= tmax)
        # z 轴（标量）
        t1z = (zmin - z0) / (z1 - z0); t2z = (zmax - z0) / (z1 - z0)
        tloz = min(t1z, t2z); thiz = max(t1z, t2z)
        ok = ok & (tmin <= thiz) & (tmax >= tloz)
        blocked |= ok
    return blocked.reshape(X.shape)

blocked = los_blocked_grid(X, Y, T)
obs = disk & pitch & (~blocked)

# 净距（到建筑）
clear = np.full(X.shape, 100.0)
for b in BUILDINGS:
    dx = np.maximum(np.maximum(b[0] - X, 0.0), X - b[1])
    dy = np.maximum(np.maximum(b[2] - Y, 0.0), Y - b[3])
    clear = np.minimum(clear, np.hypot(dx, dy))

Q = np.where(obs, 1.0 / np.where(r == 0, np.nan, r), np.nan)

# ---------- 最优站 ----------
r_obs = np.where(obs, r, np.inf)
rmin = r_obs.min()
tol = 1e-6
opt_idx = np.argwhere((r_obs - rmin) <= tol)
print(f'解析 r* = {R_ANALYTIC:.4f} m, 数值(网格0.5m) r_min = {rmin:.4f} m, 网格等值点数 = {opt_idx.shape[0]}')

# 精确最优站集合 = 以 T1 为圆心、R_ANALYTIC 为半径的可行圆弧。
# 并列取舍：在该圆弧上取离某车辆初始位置最近者（最小行驶距离），再按建筑净距最大。
angles = np.linspace(0, 360, 14400, endpoint=False)
best_pt = None; best_d = np.inf; best_c = -np.inf; best_a = None
n_circle_feasible = 0
for a in angles:
    px = tx + R_ANALYTIC * np.cos(np.radians(a))
    py = ty + R_ANALYTIC * np.sin(np.radians(a))
    if not is_observable(px, py, T):
        continue
    n_circle_feasible += 1
    dmin = min(np.hypot(px - v[0], py - v[1]) for v in VEHICLES.values())
    cmin = min(np.hypot(max(b[0] - px, 0.0, px - b[1]), max(b[2] - py, 0.0, py - b[3])) for b in BUILDINGS)
    if dmin < best_d - 1e-9 or (abs(dmin - best_d) <= 1e-9 and cmin > best_c):
        best_d = dmin; best_c = cmin; best_pt = (px, py); best_a = a
ox, oy = best_pt
r_star = R_ANALYTIC; phi_star = PITCH_MAX
az_star = azimuth_from_station(ox, oy, T)
near_v = min(VEHICLES.items(), key=lambda kv: np.hypot(ox - kv[1][0], oy - kv[1][1]))
print(f'最优站(圆弧搜索): ({ox:.3f}, {oy:.3f}), r*={r_star:.4f} m, phi*={phi_star:.2f} deg, 方位角={az_star:.2f} deg')
print(f'就近车辆={near_v[0]}, 行驶距离={best_d:.3f} m, 行驶时间={best_d/SPEED:.3f} s, 建筑净距={best_c:.3f} m')

# ---------- 绘图风格 ----------
CMAP = 'turbo'
FEAS_C = '#1f9e89'; INFEAS_C = '#e9e9e9'; BLD_C = '#6b6b6b'

def draw_overlay(ax):
    """画区域边界、建筑、调度中心、目标、车辆。"""
    for b, nm in zip(BUILDINGS, BUILDING_NAMES):
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2],
                               facecolor=BLD_C, edgecolor='black', linewidth=0.9, zorder=5))
        ax.text((b[0] + b[1]) / 2, (b[3] + b[2]) / 2, nm, ha='center', va='center',
                color='white', fontsize=9, zorder=6, fontweight='bold')
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color='black', linewidth=1.2, zorder=4)
    ax.plot(*CENTER, marker='o', color='black', markersize=5, zorder=7)
    ax.text(CENTER[0] + 1.5, CENTER[1] + 1.5, 'O', fontsize=10, zorder=7)
    ax.plot(tx, ty, marker='^', color='#d62728', markersize=9, zorder=8)
    ax.text(tx + 1.5, ty + 1.5, 'T1', fontsize=10, color='#d62728', zorder=8, fontweight='bold')
    for k, v in VEHICLES.items():
        ax.plot(v[0], v[1], marker='s', color='#1f77b4', markersize=6, zorder=8)
        ax.text(v[0] + 1.2, v[1] + 1.2, k, fontsize=8, color='#1f77b4', zorder=8)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
    ax.tick_params(labelsize=8, direction='in')
    despine(ax)

def map_fig(field, Xm, Ym, vmin=None, vmax=None, cmap=CMAP, nan_white=True):
    """返回画好 field 的 ax（区域底图）。"""
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    if nan_white:
        field = np.ma.masked_invalid(field)
    mesh = ax.pcolormesh(Xm, Ym, field, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    draw_overlay(ax)
    return fig, ax, mesh

# ========== 图 1-1 区域总览 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='#f6f6f4', edgecolor='black', linewidth=1.2, zorder=0))
draw_overlay(ax)
save_fig(fig, '图1-1_作业区域总览')

# ========== 图 1-2 可观测区域掩膜 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
field = np.where(obs, 1.0, 0.0)
ax.pcolormesh(X, Y, field, cmap=mcolors.ListedColormap([INFEAS_C, FEAS_C]), shading='auto')
draw_overlay(ax)
save_fig(fig, '图1-2_T1可观测区域掩膜')

# ========== 图 1-3 几何约束可行域 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X, Y, np.where(disk, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, '#74add1']), shading='auto')
draw_overlay(ax)
save_fig(fig, '图1-3_车辆几何约束可行域')

# ========== 图 1-4 视线遮挡约束 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X, Y, np.where(~blocked, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, '#fdae61']), shading='auto')
draw_overlay(ax)
save_fig(fig, '图1-4_视线遮挡约束可见域')

# ========== 图 1-5 俯仰约束可行域 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X, Y, np.where(pitch, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, '#d8b5e8']), shading='auto')
draw_overlay(ax)
save_fig(fig, '图1-5_云台俯仰约束可行域')

# ========== 图 1-6 三约束分解对比（1×3） ==========
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
masks = [disk, ~blocked, pitch]
titles = ['车辆几何约束', '视线遮挡约束', '云台俯仰约束']
colors = ['#74add1', '#fdae61', '#d8b5e8']
for ax_, mk, cl, tl in zip(axes, masks, colors, titles):
    ax_.pcolormesh(X, Y, np.where(mk, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, cl]), shading='auto')
    draw_overlay(ax_)
    ax_.text(50, -6.5, tl, ha='center', va='top', fontsize=10, fontweight='bold')
save_fig(fig, '图1-6_三约束可行域分解对比')

# ========== 图 1-7 水平距离 r 分布 ==========
fig, ax, mesh = map_fig(np.where(obs, r, np.nan), X, Y, vmin=rmin, vmax=80)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('水平距离 r (m)', fontsize=9)
save_fig(fig, '图1-7_可观测区水平距离分布')

# ========== 图 1-8 观测质量 Q=1/r 分布 ==========
fig, ax, mesh = map_fig(Q, X, Y, vmin=0, vmax=1.0 / rmin)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('观测质量 Q=1/r (1/m)', fontsize=9)
save_fig(fig, '图1-8_可观测区观测质量分布')

# ========== 图 1-9 俯仰角分布 ==========
fig, ax, mesh = map_fig(np.where(obs, phi, np.nan), X, Y, vmin=0, vmax=80)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('视线俯仰角 (deg)', fontsize=9)
save_fig(fig, '图1-9_可观测区俯仰角分布')

# ========== 图 1-10 最优观测站标注 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X, Y, np.where(obs, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, FEAS_C]), shading='auto')
draw_overlay(ax)
ax.plot(ox, oy, marker='*', color='#d62728', markersize=16, zorder=9)
ax.plot([ox, tx], [oy, ty], color='#d62728', linewidth=1.4, linestyle='--', zorder=8)
ax.text(ox + 1.5, oy - 2.5, f'最优站({ox:.1f},{oy:.1f})', fontsize=9, color='#d62728', zorder=9)
save_fig(fig, '图1-10_最优观测站标注')

# ========== 图 1-11 等值最优站位圆弧 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.pcolormesh(X, Y, np.where(obs, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, FEAS_C]), shading='auto')
draw_overlay(ax)
theta_c = np.linspace(0, 2 * np.pi, 400)
ax.plot(tx + r_star * np.cos(theta_c), ty + r_star * np.sin(theta_c),
        color='#d62728', linewidth=1.6, linestyle='--', zorder=8)
ax.plot(ox, oy, marker='*', color='#d62728', markersize=16, zorder=9)
save_fig(fig, '图1-11_等值最优站位圆弧')

# ========== 图 1-12 r 与俯仰角关系散点 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
mask = obs & (r < 80)
rs = r[mask].ravel(); phis = phi[mask].ravel()
ax.scatter(rs, phis, s=5, c=rs, cmap=CMAP, alpha=0.5, edgecolors='none')
ax.axhline(PITCH_MAX, color='#d62728', linewidth=1.2, linestyle='--')
ax.axvline(r_star, color='#d62728', linewidth=1.2, linestyle='--')
ax.plot(r_star, phi_star, marker='*', color='#d62728', markersize=16)
ax.set_xlabel('水平距离 r (m)', fontsize=10); ax.set_ylabel('视线俯仰角 (deg)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.set_xlim(0, 80); ax.set_ylim(0, 85)
save_fig(fig, '图1-12_距离与俯仰角散点关系')

# ========== 图 1-13 可观测区 r 分布直方图 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rs_all = r[obs].ravel()
ax.hist(rs_all, bins=40, color='#1f9e89', edgecolor='black', linewidth=0.6, alpha=0.85)
ax.axvline(r_star, color='#d62728', linewidth=1.4, linestyle='--')
ax.set_xlabel('水平距离 r (m)', fontsize=10); ax.set_ylabel('频数', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图1-13_可观测区距离分布直方图')

# ========== 图 1-14 建筑净距分布 ==========
fig, ax, mesh = map_fig(np.where(obs, clear, np.nan), X, Y, vmin=0, vmax=30, cmap='RdYlGn')
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('到最近建筑净距 (m)', fontsize=9)
save_fig(fig, '图1-14_可观测区建筑净距分布')

# ========== 图 1-15 / 1-16 局部放大 ==========
for zoom, tag in [(5.0, '图1-15_目标局部放大5m'), (15.0, '图1-16_目标局部放大15m')]:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.pcolormesh(X, Y, np.where(obs, 1.0, 0.0), cmap=mcolors.ListedColormap([INFEAS_C, FEAS_C]), shading='auto')
    draw_overlay(ax)
    ax.plot(ox, oy, marker='*', color='#d62728', markersize=14, zorder=9)
    ax.set_xlim(tx - zoom, tx + zoom); ax.set_ylim(ty - zoom, ty + zoom)
    save_fig(fig, tag)

# ========== 图 1-17 3D 视线与建筑示意 ==========
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(figsize=(7.4, 6.0))
ax = fig.add_subplot(111, projection='3d')
for b in BUILDINGS:
    x0, x1, y0, y1, h = b
    verts = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h)]
    faces = [[verts[0], verts[1], verts[2], verts[3]], [verts[4], verts[5], verts[6], verts[7]],
             [verts[0], verts[1], verts[5], verts[4]], [verts[2], verts[3], verts[7], verts[6]],
             [verts[1], verts[2], verts[6], verts[5]], [verts[3], verts[0], verts[4], verts[7]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor='#b8b8b8', edgecolor='black', linewidth=0.5, alpha=0.75))
ax.plot([ox, tx], [oy, ty], [CAM_H, H], color='#d62728', linewidth=2)
ax.plot([ox], [oy], [CAM_H], marker='s', color='#1f77b4', markersize=8)
ax.plot([tx], [ty], [H], marker='^', color='#d62728', markersize=10)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 50)
ax.view_init(elev=28, azim=-55)
ax.xaxis.pane.set_facecolor('white'); ax.yaxis.pane.set_facecolor('white'); ax.zaxis.pane.set_facecolor('white')
ax.tick_params(labelsize=8)
save_fig(fig, '图1-17_3D视线与建筑示意')

# ========== 图 1-18 俯仰角随距离变化曲线 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rr = np.linspace(3, 100, 400)
pp = np.degrees(np.arctan2(H - CAM_H, rr))
ax.plot(rr, pp, color='#1f77b4', linewidth=1.6)
ax.axhline(PITCH_MAX, color='#d62728', linewidth=1.2, linestyle='--')
ax.axvline(R_ANALYTIC, color='#d62728', linewidth=1.2, linestyle='--')
ax.fill_between(rr, PITCH_MIN, PITCH_MAX, where=(pp <= PITCH_MAX), color='#1f9e89', alpha=0.12)
ax.set_xlabel('水平距离 r (m)', fontsize=10); ax.set_ylabel('视线俯仰角 (deg)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.set_xlim(0, 100); ax.set_ylim(0, 85)
save_fig(fig, '图1-18_俯仰角随距离变化')

# ========== 图 1-19 可观测区面积随俯仰上限灵敏度 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
caps = np.arange(30, 81, 2.5)
areas = []
for c in caps:
    pitch_c = (phi >= PITCH_MIN) & (phi <= c)
    areas.append(((disk & pitch_c & (~blocked)).sum() * GS * GS))
ax.plot(caps, areas, marker='o', color='#1f9e89', linewidth=1.6, markersize=4)
ax.axvline(PITCH_MAX, color='#d62728', linewidth=1.2, linestyle='--')
ax.set_xlabel('俯仰角上限 (deg)', fontsize=10); ax.set_ylabel('可观测区面积 ($m^2$)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图1-19_面积随俯仰上限灵敏度')

# ========== 图 1-20 最优站视线垂直剖面 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
s_dist = np.linspace(0, r_star, 200)
z_los = CAM_H + (H - CAM_H) * (s_dist / r_star)
ax.plot(s_dist, z_los, color='#d62728', linewidth=1.8, label='视线')
ax.plot(0, CAM_H, marker='s', color='#1f77b4', markersize=8)
ax.plot(r_star, H, marker='^', color='#d62728', markersize=10)
ax.axhline(CAM_H, color='#999999', linewidth=0.8, linestyle=':')
ax.annotate(f'俯仰角={phi_star:.1f}°', xy=(r_star * 0.55, (H - CAM_H) * 0.55), fontsize=9, color='#d62728')
ax.set_xlabel('沿视线水平距离 (m)', fontsize=10); ax.set_ylabel('高度 z (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.set_xlim(0, r_star * 1.05); ax.set_ylim(0, H * 1.15)
top_legend(fig, [ax])
save_fig(fig, '图1-20_最优站视线垂直剖面')

# ========== 图 1-21 各约束排除区域对比 ==========
fig, ax = plt.subplots(figsize=(7.0, 4.4))
cat_names = ['同时满足\n(可观测)', '仅几何\n不满足', '仅俯仰\n不满足', '仅遮挡\n不满足', '多约束\n同时不满足']
tot = X.size
n_feas = obs.sum()
n_geo = (~disk & pitch & ~blocked).sum()
n_pit = (disk & ~pitch & ~blocked).sum()
n_los = (disk & pitch & blocked).sum()
n_multi = (~disk & ~pitch) | (~disk & blocked) | (~pitch & blocked) | (~disk & ~pitch & blocked)
n_multi = n_multi.sum()
vals = [n_feas, n_geo, n_pit, n_los, n_multi]
colors = ['#1f9e89', '#74add1', '#d8b5e8', '#fdae61', '#c0c0c0']
bars = ax.bar(range(len(cat_names)), vals, width=0.45, color=colors, edgecolor='black', linewidth=0.8)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + tot * 0.01, f'{v}', ha='center', fontsize=9)
ax.set_xticks(range(len(cat_names))); ax.set_xticklabels(cat_names, fontsize=9)
ax.set_ylabel('网格点数', fontsize=10)
ax.set_ylim(0, tot * 0.72)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.6)
save_fig(fig, '图1-21_各约束排除区域统计')

# ========== 图 1-22 可观测点按约束着色 ==========
fig, ax = plt.subplots(figsize=(6.8, 6.0))
# 仅取可观测点，按俯仰角着色；叠加遮挡临界点（可观测区边缘）
obs_pts = obs.ravel()
sc = ax.scatter(X.ravel()[obs_pts], Y.ravel()[obs_pts], c=phi.ravel()[obs_pts], s=3,
                cmap=CMAP, vmin=0, vmax=80, alpha=0.7)
draw_overlay(ax)
ax.plot(ox, oy, marker='*', color='#d62728', markersize=16, zorder=9)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label('视线俯仰角 (deg)', fontsize=9)
save_fig(fig, '图1-22_可观测点俯仰着色')

# ---------- 结果输出 ----------
print(f'\n可观测区网格点数 = {obs.sum()}，面积 ≈ {obs.sum() * GS * GS:.1f} m²')
print(f'总网格点 = {tot}')

# CSV1：网格采样（1 m 简化）
xs_ = np.arange(0, 101, 1.0)
Xs, Ys = np.meshgrid(xs_, xs_)
disk_s = (Xs - VEH_R >= 0) & (Xs + VEH_R <= 100) & (Ys - VEH_R >= 0) & (Ys + VEH_R <= 100)
for b in BUILDINGS:
    dx = np.maximum(np.maximum(b[0] - Xs, 0.0), Xs - b[1])
    dy = np.maximum(np.maximum(b[2] - Ys, 0.0), Ys - b[3])
    disk_s &= np.hypot(dx, dy) >= VEH_R
r_s = np.hypot(tx - Xs, ty - Ys)
phi_s = np.degrees(np.arctan2(H - CAM_H, r_s))
pitch_s = (phi_s >= PITCH_MIN) & (phi_s <= PITCH_MAX)
blocked_s = los_blocked_grid(Xs, Ys, T)
obs_s = disk_s & pitch_s & (~blocked_s)
grid_df = pd.DataFrame({
    'x': Xs.ravel(), 'y': Ys.ravel(),
    '几何可行': disk_s.ravel().astype(int),
    '俯仰可行': pitch_s.ravel().astype(int),
    '视线遮挡': blocked_s.ravel().astype(int),
    '可观测': obs_s.ravel().astype(int),
    '水平距离r_m': np.round(r_s.ravel(), 3),
    '俯仰角_deg': np.round(phi_s.ravel(), 3),
})
save_csv(grid_df, '问题一_可观测区域网格.csv')

summary_df = pd.DataFrame([
    ['目标T1坐标', f'({tx:.3f}, {ty:.3f}, {H:.0f})'],
    ['解析最优水平距离r*', f'{R_ANALYTIC:.4f} m'],
    ['数值最优水平距离r*', f'{r_star:.4f} m'],
    ['最优站坐标', f'({ox:.3f}, {oy:.3f})'],
    ['视线俯仰角', f'{phi_star:.3f} deg'],
    ['观测质量Q=1/r', f'{1 / r_star:.5f}'],
    ['就近车辆', near_v[0]],
    ['就近车辆行驶距离', f'{best_d:.3f} m'],
    ['就近车辆行驶时间', f'{best_d / SPEED:.3f} s'],
    ['可观测区面积', f'{obs.sum() * GS * GS:.1f} m²'],
    ['等值最优站数量', f'{n_circle_feasible}（r*圆弧可行采样点）'],
    ['并列取舍规则', '就近车辆行驶距离最小，再取建筑净距最大'],
], columns=['指标', '值'])
save_csv(summary_df, '问题一_最优观测站参数.csv')

print('\n问题一求解完成。')
