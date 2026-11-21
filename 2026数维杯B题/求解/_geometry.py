# -*- coding: utf-8 -*-
"""
公共几何模块（正式求解工具）—— 建筑/车辆/告警参数 + 可观测性判定。
供 问题一~问题四 各脚本 `from _geometry import *` 复用，保证几何判定一致。
"""
import numpy as np

# ---------- 常量 ----------
XMIN_REG, XMAX_REG = 0.0, 100.0
YMIN_REG, YMAX_REG = 0.0, 100.0
CENTER = (50.0, 50.0)
CAM_H = 2.0        # 相机离地高度 (m)
VEH_R = 1.5        # 车辆等效圆盘半径 (m)
SPEED = 3.0        # 行驶速度 (m/s)
OMEGA = 30.0       # 云台最大角速度 (deg/s)
PITCH_MIN, PITCH_MAX = -10.0, 80.0   # 俯仰角范围 (deg)
OBS_TIME = 5.0     # 联合观测时长 (s)

# 建筑物：xmin, xmax, ymin, ymax, height
BUILDINGS = np.array([
    [22, 34, 16, 26, 20],   # B1
    [63, 79, 59, 71, 28],   # B2
    [10, 24, 65, 76, 12],   # B3
    [72, 86, 22, 34, 24],   # B4
    [41, 51, 38, 46, 32],   # B5
], dtype=float)
BUILDING_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5']

# 车辆初始位置
VEHICLES = {
    'V1': np.array([15.0, 14.0]),
    'V2': np.array([88.0, 10.0]),
    'V3': np.array([14.0, 86.0]),
    'V4': np.array([88.0, 68.0]),
    'V5': np.array([50.0, 93.0]),
}
VEHICLE_ORDER = ['V1', 'V2', 'V3', 'V4', 'V5']


def target_from_alert(theta_deg, R, H):
    """由告警 (theta 度, R, H) 计算目标 (x, y, H)。"""
    th = np.radians(theta_deg)
    return np.array([50.0 + R * np.cos(th), 50.0 + R * np.sin(th), H])


def make_targets():
    """三个基础告警目标。"""
    return {
        'T1': target_from_alert(30, 40, 45),
        'T2': target_from_alert(150, 45, 35),
        'T3': target_from_alert(285, 38, 55),
    }


TARGETS = make_targets()


# ---------- 几何判定 ----------
def _point_rect_dist(x, y, b):
    """点 (x,y) 到矩形 [xmin,xmax]x[ymin,ymax] 的欧氏距离（点在内部为 0）。"""
    dx = max(b[0] - x, 0.0, x - b[1])
    dy = max(b[2] - y, 0.0, y - b[3])
    return np.hypot(dx, dy)


def disk_in_free(x, y, veh_r=VEH_R):
    """车辆几何约束：圆盘中心 (x,y)，圆盘在区域内且不侵入建筑。"""
    if x - veh_r < XMIN_REG or x + veh_r > XMAX_REG:
        return False
    if y - veh_r < YMIN_REG or y + veh_r > YMAX_REG:
        return False
    for b in BUILDINGS:
        if _point_rect_dist(x, y, b) < veh_r:
            return False
    return True


def seg_box_hit(x0, y0, z0, x1, y1, z1, b):
    """3D 线段 (p0,z0)->(p1,z1) 与轴对齐盒 [xmin,xmax]x[ymin,ymax]x[zmin,zmax] 是否相交（含接触）。"""
    xmin, xmax, ymin, ymax, h = b[0], b[1], b[2], b[3], b[4]
    zmin, zmax = 0.0, h
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    tmin, tmax = 0.0, 1.0
    for p0, p1, lo, hi in [(x0, x1, xmin, xmax), (y0, y1, ymin, ymax), (z0, z1, zmin, zmax)]:
        if p1 == p0:
            if p0 < lo or p0 > hi:
                return False
        else:
            t1 = (lo - p0) / (p1 - p0)
            t2 = (hi - p0) / (p1 - p0)
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def los_blocked(x, y, target):
    """视线遮挡约束：线段 (x,y,CAM_H)->(target[0],target[1],target[2]) 是否被任一建筑遮挡。"""
    tx, ty, H = target[0], target[1], target[2]
    for b in BUILDINGS:
        if seg_box_hit(x, y, CAM_H, tx, ty, H, b):
            return True
    return False


def pitch_deg(x, y, target):
    """视线俯仰角 (deg)。"""
    r = np.hypot(target[0] - x, target[1] - y)
    return np.degrees(np.arctan2(target[2] - CAM_H, r))


def pitch_ok(x, y, target):
    """俯仰角是否落在 [-10,80] 内。"""
    p = pitch_deg(x, y, target)
    return PITCH_MIN - 1e-12 <= p <= PITCH_MAX + 1e-12


def is_observable(x, y, target):
    """三约束同时满足。"""
    if not disk_in_free(x, y):
        return False
    if not pitch_ok(x, y, target):
        return False
    if los_blocked(x, y, target):
        return False
    return True


def hor_dist(x, y, target):
    return np.hypot(target[0] - x, target[1] - y)


def azimuth_from_station(x, y, target):
    """从观测站 (x,y) 到目标水平投影的方位角（自正东逆时针，[0,360)）。"""
    a = np.degrees(np.arctan2(target[1] - y, target[0] - x))
    return a % 360.0


def inter_angle(a1, a2):
    """两方位线交会角 gamma（取较小夹角，[0,180]）。"""
    d = abs(a1 - a2) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def rotate_time(az_deg, pitch_deg):
    """云台从正东水平(0,0)转至 (az, pitch) 的用时 = max(方位转角,俯仰转角)/30。
    方位转角取最短路径 [0,180]。"""
    az = az_deg % 360.0
    az_turn = min(az, 360.0 - az)
    return max(az_turn, abs(pitch_deg)) / OMEGA


def travel_time(p0, p1):
    return np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / SPEED
