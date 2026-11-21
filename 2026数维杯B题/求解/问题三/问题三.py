# -*- coding: utf-8 -*-
"""
问题三：定位误差与指向误差对水平方位精度的影响
- 误差传播模型：Δψ_pos ≈ (Y·δx − X·δy)/r²，σ_ψ_pos = σ_pos/r
- 总指向偏差 = Δψ_pos + 零位偏差(0.8°) + 控制误差(0.6°)
- 求 P(|Δψ|≤5°)≥0.95 的最大行驶距离 s_max；s=80 时 σ(s)=a+bs 的参数约束
- 蒙特卡洛验证 + 三误差源方差分解
"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SOLVE_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _SOLVE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import *
from _geometry import *
from scipy.stats import norm

import matplotlib.colors as mcolors

# ---------- 参数 ----------
R0 = 40.0                 # 水平距离 (m)
SIG_Z = 0.8               # 零位偏差 (deg)
SIG_C = 0.6               # 控制误差 (deg)
A0, B0 = 0.3, 0.02        # σ_pos(s) = A0 + B0*s
THR = 5.0                 # 精度要求 (deg)
CONF = 0.95
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

K_PSI = R2D / R0          # 1 m 定位误差 → 方位误差(deg) 转换系数 = 180/(40π)

# ---------- 解析 ----------
k = norm.ppf((1 + CONF) / 2)      # 1.96
sig_total_limit = THR / k          # σ_total 上限 (deg)
sig_pos_limit = np.sqrt((sig_total_limit ** 2 - SIG_Z ** 2 - SIG_C ** 2) / (K_PSI ** 2))
s_max = (sig_pos_limit - A0) / B0
print(f'k={k:.4f}, σ_total上限={sig_total_limit:.4f}°, σ_pos上限={sig_pos_limit:.4f} m, s_max={s_max:.3f} m')
print(f'换算系数 K_ψ={K_PSI:.4f} °/m')

# s=80 时 σ(80)=a+80b 约束
s80 = 80.0
const_ab = sig_pos_limit
print(f'σ(80)=a+80b ≤ {const_ab:.4f} m（a,b≥0），b≤{(const_ab-A0)/80:.5f}（若a=0.3）')

# 仅定位误差时的上界（供对比）
sig_pos_only = sig_total_limit / K_PSI
s_max_only = (sig_pos_only - A0) / B0
print(f'仅考虑定位误差时 s_max={s_max_only:.3f} m')


def sigma_pos(s, a=A0, b=B0):
    return a + b * s


def sigma_psi_pos_deg(s):
    return K_PSI * sigma_pos(s)


def sigma_total_deg(s):
    return np.sqrt(sigma_psi_pos_deg(s) ** 2 + SIG_Z ** 2 + SIG_C ** 2)


def prob_ok(s):
    return 2 * norm.cdf(THR / sigma_total_deg(s)) - 1


# ---------- 蒙特卡洛 ----------
rng = np.random.default_rng(20261121)
NM = 200000


def mc_prob(s, a=A0, b=B0):
    sp = sigma_pos(s, a, b)
    dx = rng.normal(0, sp, NM)
    dy = rng.normal(0, sp, NM)
    dz = rng.normal(0, SIG_Z, NM)
    dc = rng.normal(0, SIG_C, NM)
    # 目标在 (40,0)，车辆估计位置 (dx,dy)
    psi = np.degrees(np.arctan2(0 - dy, R0 - dx))   # 指令方位角与真实(0°)之差（定位引起）
    tot = psi + dz + dc
    tot = (tot + 180.0) % 360.0 - 180.0              # 折算到 [-180,180)
    return (np.abs(tot) <= THR).mean(), (np.abs(psi) <= THR).mean(), tot, psi


ss = np.array([0, 10, 20, 30, 40, 50, 60, 66.92, 70, 80, 90, 100])
p_mc = []; p_mc_pos = []
for s in ss:
    p, pp, tot, psi = mc_prob(s)
    p_mc.append(p); p_mc_pos.append(pp)
    print(f's={s:6.2f} m: P(|Δψ_total|≤5°) 解析={prob_ok(s):.4f}, MC={p:.4f}, P(|Δψ_pos|≤5°) MC={pp:.4f}')

# ---------- 绘图 ----------
CMAP3 = 'YlOrRd'

# 图3-1 误差传播几何示意
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot([0, R0], [0, 0], color='#1f77b4', linewidth=1.6, label='真实视线')
ax.plot([0, R0], [0, 0], color='#1f77b4', linewidth=1.6)
# 估计位置误差圆
ax.add_patch(Circle((0, 0), 1.2, fill=False, edgecolor='#888888', linewidth=0.8, linestyle=':'))
ax.scatter([0.8], [0.9], color='#d62728', s=40, zorder=5, label='估计位置 P̂')
ax.plot([0.8, R0], [0.9, 0], color='#d62728', linewidth=1.4, linestyle='--', label='指令视线')
ax.annotate('', xy=(R0, 0), xytext=(0.8, 0.9),
            arrowprops=dict(arrowstyle='->', color='#d62728', linewidth=1.2))
ax.annotate('Δψ', xy=(R0 * 0.62, 0.42), fontsize=11, color='#d62728')
ax.plot([0], [0], marker='s', color='#1f77b4', markersize=8, label='真实位置 P')
ax.plot([R0], [0], marker='^', color='#2ca02c', markersize=11, label='目标 T')
ax.set_xlim(-2, R0 + 3); ax.set_ylim(-1.5, 2.6)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)', fontsize=10); ax.set_ylabel('y (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-1_误差传播几何示意')

# 图3-2 σ_pos(s)
fig, ax = plt.subplots(figsize=(7.0, 4.4))
sx = np.linspace(0, 100, 400)
ax.plot(sx, sigma_pos(sx), color='#1f77b4', linewidth=1.8, label='σ(s)=0.3+0.02s')
ax.axhline(sig_pos_limit, color='#d62728', linewidth=1.2, linestyle='--', label=f'上限 {sig_pos_limit:.2f} m')
ax.axvline(s_max, color='#d62728', linewidth=1.0, linestyle=':')
ax.scatter([s_max], [sig_pos_limit], color='#d62728', s=60, zorder=5)
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('定位标准差 σ_pos (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-2_定位标准差随行驶距离')

# 图3-3 σ_ψ 随 s
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(sx, sigma_psi_pos_deg(sx), color='#1f77b4', linewidth=1.8, label='定位引起的方位偏差 σ_ψ')
ax.plot(sx, sigma_total_deg(sx), color='#d62728', linewidth=1.8, label='总方位偏差 σ_total')
ax.axhline(THR, color='#888888', linewidth=1.0, linestyle='--')
ax.axhline(sig_total_limit, color='#2ca02c', linewidth=1.0, linestyle='--', label='95%阈值')
ax.axvline(s_max, color='#d62728', linewidth=1.0, linestyle=':')
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('方位偏差标准差 (deg)', fontsize=10)
ax.set_ylim(0, 6)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-3_方位偏差标准差随行驶距离')

# 图3-4 三误差源标准差对比
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(sx, sigma_psi_pos_deg(sx), color='#1f77b4', linewidth=1.8, label='定位误差贡献')
ax.axhline(SIG_Z, color='#2ca02c', linewidth=1.8, label='云台零位偏差 0.8°')
ax.axhline(SIG_C, color='#ff7f0e', linewidth=1.8, label='云台控制误差 0.6°')
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('各误差源标准差 (deg)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-4_三误差源标准差对比')

# 图3-5 P(|Δψ|≤5°) 随 s（解析+MC）
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(sx, prob_ok(sx), color='#1f77b4', linewidth=1.8, label='解析')
ax.scatter(ss, p_mc, color='#d62728', s=45, zorder=5, label='蒙特卡洛', edgecolors='black', linewidth=0.5)
ax.axhline(CONF, color='#888888', linewidth=1.0, linestyle='--', label='要求 0.95')
ax.axvline(s_max, color='#d62728', linewidth=1.0, linestyle=':')
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('P(|Δψ|≤5°)', fontsize=10)
ax.set_ylim(0.5, 1.02)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-5_概率随行驶距离')

# 图3-6 MC 样本散点
fig, ax = plt.subplots(figsize=(6.8, 5.2))
sp = sigma_pos(66.92)
dx = rng.normal(0, sp, 8000); dy = rng.normal(0, sp, 8000)
psi = np.degrees(np.arctan2(-dy, R0 - dx))
sc = ax.scatter(dx, dy, c=psi, cmap=CMAP3, s=8, alpha=0.6, vmin=-8, vmax=8)
ax.add_patch(Circle((0, 0), sp, fill=False, edgecolor='#888888', linewidth=0.9, linestyle=':'))
ax.add_patch(Circle((0, 0), 2 * sp, fill=False, edgecolor='#aaaaaa', linewidth=0.7, linestyle=':'))
ax.set_xlabel('δx (m)', fontsize=10); ax.set_ylabel('δy (m)', fontsize=10)
ax.set_aspect('equal', adjustable='box')
ax.tick_params(labelsize=9, direction='in'); despine(ax)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Δψ_pos (deg)', fontsize=9)
save_fig(fig, '图3-6_定位误差样本与方位偏差')

# 图3-7 Δψ 分布直方图
fig, ax = plt.subplots(figsize=(7.0, 4.4))
p, pp, tot, psi_all = mc_prob(66.92)
ax.hist(tot, bins=60, density=True, color='#1f9e89', edgecolor='black', linewidth=0.4, alpha=0.8)
xg = np.linspace(-15, 15, 400)
ax.plot(xg, norm.pdf(xg, 0, sigma_total_deg(66.92)), color='#d62728', linewidth=1.6, label='正态拟合')
ax.axvline(-THR, color='#888888', linewidth=1.0, linestyle='--'); ax.axvline(THR, color='#888888', linewidth=1.0, linestyle='--')
ax.set_xlim(-15, 15)
ax.set_xlabel('总方位偏差 Δψ_total (deg)', fontsize=10); ax.set_ylabel('概率密度', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-7_总方位偏差分布直方图')

# 图3-8 小角度近似误差
fig, ax = plt.subplots(figsize=(7.0, 4.4))
sps = np.linspace(0.1, 3.0, 60)
err_mean = []; err_std = []
for spv in sps:
    ddx = rng.normal(0, spv, 30000); ddy = rng.normal(0, spv, 30000)
    exact = np.degrees(np.arctan2(-ddy, R0 - ddx))
    approx = np.degrees(-ddy / R0)
    err_mean.append(np.abs((exact - approx)).mean()); err_std.append((exact - approx).std())
ax.plot(sps, err_mean, color='#1f77b4', linewidth=1.6, label='近似误差均值')
ax.plot(sps, err_std, color='#d62728', linewidth=1.6, label='近似误差标准差')
ax.set_xlabel('定位标准差 σ_pos (m)', fontsize=10); ax.set_ylabel('近似误差 (deg)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-8_小角度近似误差')

# 图3-9 三误差源方差占比堆叠（随 s）
fig, ax = plt.subplots(figsize=(7.0, 4.4))
sx2 = np.linspace(0, 100, 200)
v_pos = sigma_psi_pos_deg(sx2) ** 2
v_z = np.full_like(sx2, SIG_Z ** 2)
v_c = np.full_like(sx2, SIG_C ** 2)
tot_v = v_pos + v_z + v_c
ax.stackplot(sx2, v_pos / tot_v * 100, v_z / tot_v * 100, v_c / tot_v * 100,
             colors=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.85,
             edgecolor='black', linewidth=0.5, labels=['定位误差', '零位偏差', '控制误差'])
ax.axvline(s_max, color='#d62728', linewidth=1.0, linestyle='--')
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('方差占比 (%)', fontsize=10)
ax.set_ylim(0, 100)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-9_三误差源方差占比堆叠')

# 图3-10 方差占比（s=66.9 与 s=20 对比，1×2）
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax_, sv in zip(axes, [20.0, s_max]):
    vp = sigma_psi_pos_deg(sv) ** 2; vz = SIG_Z ** 2; vc = SIG_C ** 2
    vals = [vp, vz, vc]; labels = ['定位', '零位', '控制']
    cols = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = ax_.bar(range(3), vals, width=0.45, color=cols, edgecolor='black', linewidth=0.8)
    for b_, v_ in zip(bars, vals):
        ax_.text(b_.get_x() + b_.get_width() / 2, v_ + 0.05, f'{v_ / sum(vals) * 100:.1f}%', ha='center', fontsize=9)
    ax_.set_xticks(range(3)); ax_.set_xticklabels(labels, fontsize=9)
    ax_.set_ylabel('方差 (deg²)', fontsize=10)
    ax_.text(0.5, -0.18, f's={sv:.1f} m', ha='center', va='top', fontsize=10, fontweight='bold')
    ax_.tick_params(labelsize=9, direction='in'); despine(ax_)
save_fig(fig, '图3-10_方差占比对比')

# 图3-11 s_max 随精度要求
fig, ax = plt.subplots(figsize=(7.0, 4.4))
thrs = np.linspace(2, 10, 100)
smaxs = []
for t in thrs:
    lim = t / k
    spl = np.sqrt((lim ** 2 - SIG_Z ** 2 - SIG_C ** 2) / K_PSI ** 2)
    smaxs.append((spl - A0) / B0 if lim ** 2 > SIG_Z ** 2 + SIG_C ** 2 else 0.0)
ax.plot(thrs, smaxs, color='#1f77b4', linewidth=1.8)
ax.scatter([THR], [s_max], color='#d62728', s=60, zorder=5)
ax.set_xlabel('精度要求 |Δψ| 阈值 (deg)', fontsize=10); ax.set_ylabel('最大行驶距离 s_max (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图3-11_最大行驶距离随精度要求')

# 图3-12 a-b 可行域
fig, ax = plt.subplots(figsize=(7.0, 4.6))
bmax = const_ab / s80
bb = np.linspace(0, bmax, 200)
aa = const_ab - s80 * bb
ax.fill_between(bb, 0, aa, color='#1f9e89', alpha=0.35)
ax.plot(bb, aa, color='#d62728', linewidth=1.8, label=f'a+80b={const_ab:.4f}')
ax.scatter([B0], [A0], color='#1f77b4', s=70, zorder=5, label='当前 (a=0.3,b=0.02)')
ax.annotate('可行域', xy=(0.012, 0.35), fontsize=11, color='#1f9e89')
ax.set_xlim(0, bmax * 1.08); ax.set_ylim(0, const_ab * 1.15)
ax.set_xlabel('参数 b (m/m)', fontsize=10); ax.set_ylabel('参数 a (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-12_参数ab可行域')

# 图3-13 a-b 可行域热力（σ_pos(80) 值）
fig, ax = plt.subplots(figsize=(6.6, 5.4))
bbg = np.linspace(0, 0.03, 120); aag = np.linspace(0, 2.0, 120)
BBG, AAG = np.meshgrid(bbg, aag)
SP = AAG + s80 * BBG
feas = SP <= const_ab
SPm = np.ma.masked_where(~feas, SP)
mesh = ax.pcolormesh(BBG, AAG, SPm, cmap=CMAP3, vmin=0, vmax=2.5, shading='auto')
ax.contour(BBG, AAG, SP, levels=[const_ab], colors='#d62728', linewidths=1.8)
ax.scatter([B0], [A0], color='#1f77b4', s=70, zorder=5, edgecolors='black', linewidth=0.6)
ax.set_xlabel('参数 b (m/m)', fontsize=10); ax.set_ylabel('参数 a (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('σ(80)=a+80b (m)', fontsize=9)
save_fig(fig, '图3-13_参数ab热力图')

# 图3-14 MC 收敛
fig, ax = plt.subplots(figsize=(7.0, 4.4))
Ns = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
est = []
spv = sigma_pos(66.92)
for Nv in Ns:
    ddx = rng.normal(0, spv, Nv); ddy = rng.normal(0, spv, Nv)
    ddz = rng.normal(0, SIG_Z, Nv); ddc = rng.normal(0, SIG_C, Nv)
    psi = np.degrees(np.arctan2(-ddy, R0 - ddx))
    tot = (psi + ddz + ddc + 180.0) % 360.0 - 180.0
    est.append((np.abs(tot) <= THR).mean())
ax.axhline(prob_ok(66.92), color='#1f77b4', linewidth=1.2, label='解析值')
ax.plot(Ns, est, marker='o', color='#d62728', linewidth=1.4, markersize=4)
ax.set_xscale('log')
ax.set_xlabel('蒙特卡洛样本量 N', fontsize=10); ax.set_ylabel('P(|Δψ|≤5°) 估计', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-14_蒙特卡洛收敛')

# 图3-15 不同 s 的分位数曲线
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for sv in [20.0, 40.0, 60.0, 80.0]:
    qt = np.linspace(0.5, 0.999, 100)
    z = sigma_total_deg(sv) * norm.ppf((1 + qt) / 2) * 2
    ax.plot(qt, z, linewidth=1.6, label=f's={sv:.0f} m')
ax.axhline(THR, color='#888888', linewidth=1.0, linestyle='--')
ax.axvline(CONF, color='#d62728', linewidth=1.0, linestyle=':')
ax.set_xlabel('置信水平', fontsize=10); ax.set_ylabel('|Δψ| 双侧分位宽度 (deg)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-15_分位数曲线')

# 图3-16 Δψ 对 δx/δy 敏感性
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
dvals = np.linspace(-1.6, 1.6, 400)
for ax_, which in zip(axes, ['x', 'y']):
    if which == 'x':
        psi_x = np.degrees(np.arctan2(0, R0 - dvals))
        ax_.plot(dvals, psi_x, color='#1f77b4', linewidth=1.8)
        ax_.set_xlabel('δx (m，δy=0)', fontsize=10)
    else:
        psi_y = np.degrees(np.arctan2(-dvals, R0))
        ax_.plot(dvals, psi_y, color='#d62728', linewidth=1.8)
        ax_.set_xlabel('δy (m，δx=0)', fontsize=10)
    ax_.set_ylabel('Δψ_pos (deg)', fontsize=10)
    ax_.tick_params(labelsize=9, direction='in'); despine(ax_)
    ax_.text(0.5, -0.18, '沿视线(δx)' if which == 'x' else '垂直视线(δy)', ha='center', va='top', fontsize=10, fontweight='bold')
save_fig(fig, '图3-16_方位偏差对定位误差敏感性')

# 图3-17 总误差分解瀑布
fig, ax = plt.subplots(figsize=(7.0, 4.4))
spv = sigma_pos(66.92)
parts = [sigma_psi_pos_deg(66.92), SIG_Z, SIG_C]
names = ['定位误差', '零位偏差', '控制误差']
cols = ['#1f77b4', '#2ca02c', '#ff7f0e']
cum = 0.0
bars = []
for p_, n_, c_ in zip(parts, names, cols):
    ax.bar(0, p_ ** 2, bottom=cum, color=c_, edgecolor='black', linewidth=0.8, width=0.45)
    ax.text(0.24, cum + p_ ** 2 / 2, f'{n_}\n{p_:.2f}°', va='center', fontsize=9)
    cum += p_ ** 2
ax.text(0.24, cum + 0.05, f'总方差={cum:.2f} deg²\nσ_total={np.sqrt(cum):.2f}°', fontsize=10)
ax.set_xlim(-0.5, 1.2); ax.set_xticks([])
ax.set_ylabel('方差累加 (deg²)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图3-17_总误差方差分解')

# 图3-18 s_max 随目标距离 r
fig, ax = plt.subplots(figsize=(7.0, 4.4))
rs = np.linspace(20, 100, 400)
smax_r = []
for rv in rs:
    kk = R2D / rv
    spl = np.sqrt((sig_total_limit ** 2 - SIG_Z ** 2 - SIG_C ** 2) / kk ** 2)
    smax_r.append((spl - A0) / B0)
ax.plot(rs, smax_r, color='#1f77b4', linewidth=1.8)
ax.scatter([R0], [s_max], color='#d62728', s=60, zorder=5)
ax.set_xlabel('目标水平距离 r (m)', fontsize=10); ax.set_ylabel('最大行驶距离 s_max (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图3-18_最大行驶距离随目标距离')

# 图3-19 s_max 随置信水平
fig, ax = plt.subplots(figsize=(7.0, 4.4))
confs = np.linspace(0.5, 0.999, 400)
smax_c = []
for cf in confs:
    kk = norm.ppf((1 + cf) / 2)
    spl = np.sqrt(((THR / kk) ** 2 - SIG_Z ** 2 - SIG_C ** 2) / K_PSI ** 2)
    smax_c.append((spl - A0) / B0 if spl > 0 else 0.0)
ax.plot(confs, smax_c, color='#1f77b4', linewidth=1.8)
ax.scatter([CONF], [s_max], color='#d62728', s=60, zorder=5)
ax.set_xlabel('置信水平', fontsize=10); ax.set_ylabel('最大行驶距离 s_max (m)', fontsize=10)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
save_fig(fig, '图3-19_最大行驶距离随置信水平')

# 图3-20 三误差源相对贡献柱状
fig, ax = plt.subplots(figsize=(7.0, 4.4))
vp = sigma_psi_pos_deg(s_max) ** 2; vz = SIG_Z ** 2; vc = SIG_C ** 2
vals = [vp, vz, vc]
bars = ax.bar(range(3), [v / sum(vals) * 100 for v in vals], width=0.45,
              color=['#1f77b4', '#2ca02c', '#ff7f0e'], edgecolor='black', linewidth=0.8)
for b_, v_ in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v_ / sum(vals) * 100 + 1, f'{v_ / sum(vals) * 100:.1f}%', ha='center', fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(['定位误差', '零位偏差', '控制误差'], fontsize=10)
ax.set_ylabel('方差占比 (%)', fontsize=10)
ax.set_ylim(0, 100)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
ax.grid(axis='y', color='#cccccc', linewidth=0.5, alpha=0.6)
save_fig(fig, '图3-20_三误差源相对贡献')

# 图3-21 σ_total 随 s 与 r 热力
fig, ax = plt.subplots(figsize=(7.0, 4.4))
sr = np.linspace(20, 100, 81); sg = np.linspace(0, 100, 71)
SR, SG = np.meshgrid(sr, sg)
SIG = np.sqrt((R2D / SR * (A0 + B0 * SG)) ** 2 + SIG_Z ** 2 + SIG_C ** 2)
data = SIG.T if SIG.shape[0] > SIG.shape[1] else SIG
mesh = safe_heatmap(ax, data, row_labels=[f'{v:.0f}' for v in (sr if data.shape[0] == len(sr) else sg)],
                    col_labels=[f'{v:.0f}' for v in (sg if data.shape[0] == len(sr) else sr)],
                    cmap=CMAP3, linewidth=0.3, fontsize=8)
cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04); cb.set_label('σ_total (deg)', fontsize=9)
save_fig(fig, '图3-21_总误差随距离与行驶距离')

# 图3-22 部署建议示意（随 s 的精度余量）
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.plot(sx, sigma_total_deg(sx), color='#d62728', linewidth=1.8, label='σ_total')
ax.fill_between(sx, 0, sigma_total_deg(sx), where=(sx <= s_max), color='#1f9e89', alpha=0.25)
ax.axhline(sig_total_limit, color='#2ca02c', linewidth=1.2, linestyle='--', label=f'95%阈值 {sig_total_limit:.2f}°')
ax.axvline(s_max, color='#d62728', linewidth=1.0, linestyle=':')
ax.text(s_max + 1, 0.3, f's_max={s_max:.1f} m', fontsize=10, color='#d62728')
ax.set_xlabel('累计行驶距离 s (m)', fontsize=10); ax.set_ylabel('总方位偏差标准差 (deg)', fontsize=10)
ax.set_ylim(0, 5.5)
ax.tick_params(labelsize=9, direction='in'); despine(ax)
top_legend(fig, [ax])
save_fig(fig, '图3-22_精度余量与部署建议')

# ---------- 结果输出 ----------
res_df = pd.DataFrame([
    ['转换系数 K_ψ (°/m)', f'{K_PSI:.4f}'],
    ['95% 分位系数 k', f'{k:.4f}'],
    ['σ_total 上限 (deg)', f'{sig_total_limit:.4f}'],
    ['σ_pos 上限 (m)', f'{sig_pos_limit:.4f}'],
    ['最大允许行驶距离 s_max (m)', f'{s_max:.3f}'],
    ['仅定位误差时 s_max (m)', f'{s_max_only:.3f}'],
    ['s=80 参数约束', f'a + 80b ≤ {const_ab:.4f} m（a,b≥0）'],
    ['当前 σ(80) 值 (m)', f'{sigma_pos(80):.3f}'],
    ['定位误差方差占比@s_max', f'{vp / (vp + vz + vc) * 100:.2f}%'],
    ['零位偏差方差占比@s_max', f'{vz / (vp + vz + vc) * 100:.2f}%'],
    ['控制误差方差占比@s_max', f'{vc / (vp + vz + vc) * 100:.2f}%'],
], columns=['指标', '值'])
save_csv(res_df, '问题三_误差传播结果.csv')

mc_df = pd.DataFrame({'s_m': ss, 'P_total_解析': [prob_ok(s) for s in ss],
                      'P_total_MC': p_mc, 'P_pos_MC': p_mc_pos})
save_csv(mc_df, '问题三_蒙特卡洛验证.csv')

print('\n问题三求解完成。')
