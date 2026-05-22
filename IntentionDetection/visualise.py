#!/usr/bin/env python

import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "icm20948-python-main"))

from icm20948 import ICM20948

BETA = 0.1

imu = ICM20948()

# --- Initialise quaternion from first sensor reading (ENU frame) ---
ax, ay, az, _, _, _ = imu.read_accelerometer_gyro_data()
mx, my, mz = imu.read_magnetometer_data()

roll0  = math.atan2(ay, math.sqrt(ax**2 + az**2))
pitch0 = math.atan2(-ax, az)
yaw0   = math.atan2(
    mx * math.cos(pitch0) + mz * math.sin(pitch0),
    mx * math.sin(pitch0) * math.sin(roll0) + my * math.cos(roll0) - mz * math.cos(pitch0) * math.sin(roll0)
)

cr, sr = math.cos(roll0 / 2), math.sin(roll0 / 2)
cp, sp = math.cos(pitch0 / 2), math.sin(pitch0 / 2)
cy, sy = math.cos(yaw0 / 2), math.sin(yaw0 / 2)

qw = cr*cp*cy + sr*sp*sy
qx = sr*cp*cy - cr*sp*sy
qy = cr*sp*cy + sr*cp*sy
qz = cr*cp*sy - sr*sp*cy

last_time = time.time()

# Box vertices shaped like a PCB (flat rectangular board)
half = np.array([1.5, 1.0, 0.15])
_v = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
]) * half

FACES = [
    ([0, 1, 2, 3], 'royalblue'),   # bottom
    ([4, 5, 6, 7], 'tomato'),       # top (red = "up" face)
    ([0, 1, 5, 4], 'lightgray'),
    ([2, 3, 7, 6], 'lightgray'),
    ([0, 3, 7, 4], 'lightgray'),
    ([1, 2, 6, 5], 'lightgray'),
]

# Axis arrows in body frame: X=red, Y=green, Z=blue
ARROWS = [
    (np.array([1.8, 0, 0]), 'red',   'X'),
    (np.array([0, 1.2, 0]), 'green', 'Y'),
    (np.array([0, 0, 0.5]), 'blue',  'Z'),
]


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),       2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx*qx + qy*qy)],
    ])


def madgwick_update(ax, ay, az, gx, gy, gz, mx, my, mz, dt):
    global qw, qx, qy, qz

    gx = math.radians(gx)
    gy = math.radians(gy)
    gz = math.radians(gz)

    qdot_w = 0.5 * (-qx*gx - qy*gy - qz*gz)
    qdot_x = 0.5 * ( qw*gx + qy*gz - qz*gy)
    qdot_y = 0.5 * ( qw*gy - qx*gz + qz*gx)
    qdot_z = 0.5 * ( qw*gz + qx*gy - qy*gx)

    a_norm = math.sqrt(ax*ax + ay*ay + az*az)
    if a_norm == 0:
        return
    ax /= a_norm; ay /= a_norm; az /= a_norm

    m_norm = math.sqrt(mx*mx + my*my + mz*mz)
    if m_norm == 0:
        return
    mx /= m_norm; my /= m_norm; mz /= m_norm

    hx = 2*mx*(0.5-qy*qy-qz*qz) + 2*my*(qx*qy-qw*qz) + 2*mz*(qx*qz+qw*qy)
    hy = 2*mx*(qx*qy+qw*qz) + 2*my*(0.5-qx*qx-qz*qz) + 2*mz*(qy*qz-qw*qx)
    hz = 2*mx*(qx*qz-qw*qy) + 2*my*(qy*qz+qw*qx) + 2*mz*(0.5-qx*qx-qy*qy)
    bx = math.sqrt(hx*hx + hy*hy)
    by = 0.0
    bz = hz

    f1 = 2*(qx*qz - qw*qy)       - ax
    f2 = 2*(qw*qx + qy*qz)       - ay
    f3 = 2*(0.5 - qx*qx - qy*qy) - az

    f4 = 2*bx*(0.5-qy*qy-qz*qz) + 2*by*(qw*qz+qx*qy) + 2*bz*(qx*qz-qw*qy) - mx
    f5 = 2*bx*(qx*qy-qw*qz) + 2*by*(0.5-qx*qx-qz*qz) + 2*bz*(qw*qx+qy*qz) - my
    f6 = 2*bx*(qw*qy+qx*qz) + 2*by*(qy*qz-qw*qx) + 2*bz*(0.5-qx*qx-qy*qy) - mz

    grad_w = (-2*qy*f1 + 2*qx*f2 + (2*by*qz-2*bz*qy)*f4 + (-2*bx*qz+2*bz*qx)*f5 + (2*bx*qy-2*by*qx)*f6)
    grad_x = (2*qz*f1 + 2*qw*f2 - 4*qx*f3 + (2*by*qy+2*bz*qz)*f4 + (2*bx*qy-4*by*qx+2*bz*qw)*f5 + (2*bx*qz-2*by*qw-4*bz*qx)*f6)
    grad_y = (-2*qw*f1 + 2*qz*f2 - 4*qy*f3 + (-4*bx*qy+2*by*qx-2*bz*qw)*f4 + (2*bx*qx+2*bz*qz)*f5 + (2*bx*qw+2*by*qz-4*bz*qy)*f6)
    grad_z = (2*qx*f1 + 2*qy*f2 + (-4*bx*qz+2*by*qw+2*bz*qx)*f4 + (-2*bx*qw-4*by*qz+2*bz*qy)*f5 + (2*bx*qx+2*by*qy)*f6)

    g_norm = math.sqrt(grad_w**2 + grad_x**2 + grad_y**2 + grad_z**2)
    if g_norm > 0:
        grad_w /= g_norm; grad_x /= g_norm; grad_y /= g_norm; grad_z /= g_norm
        qdot_w -= BETA * grad_w
        qdot_x -= BETA * grad_x
        qdot_y -= BETA * grad_y
        qdot_z -= BETA * grad_z

    qw += qdot_w * dt; qx += qdot_x * dt
    qy += qdot_y * dt; qz += qdot_z * dt

    q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw /= q_norm; qx /= q_norm; qy /= q_norm; qz /= q_norm


fig = plt.figure(figsize=(8, 7))
ax3d = fig.add_subplot(111, projection='3d')


def update(_frame):
    global last_time

    a_x, a_y, a_z, g_x, g_y, g_z = imu.read_accelerometer_gyro_data()
    m_x, m_y, m_z = imu.read_magnetometer_data()

    now = time.time()
    dt = now - last_time
    last_time = now

    madgwick_update(a_x, a_y, a_z, g_x, g_y, g_z, m_x, m_y, m_z, dt)

    R = quat_to_rot(qw, qx, qy, qz)
    rotated = (R @ _v.T).T

    ax3d.cla()
    ax3d.set_xlim(-2, 2); ax3d.set_ylim(-2, 2); ax3d.set_zlim(-2, 2)
    ax3d.set_xlabel('X (East)'); ax3d.set_ylabel('Y (North)'); ax3d.set_zlabel('Z (Up)')
    ax3d.set_box_aspect([1, 1, 1])

    polys = [[rotated[i] for i in face] for face, _ in FACES]
    colors = [c for _, c in FACES]
    collection = Poly3DCollection(polys, alpha=0.85, facecolor=colors, edgecolor='k', linewidth=0.4)
    ax3d.add_collection3d(collection)

    # Body-frame axis arrows
    origin = np.zeros(3)
    for tip, color, label in ARROWS:
        rotated_tip = R @ tip
        ax3d.quiver(*origin, *rotated_tip, color=color, linewidth=2, arrow_length_ratio=0.2)
        ax3d.text(*(rotated_tip * 1.1), label, color=color, fontsize=9, weight='bold')

    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))

    ax3d.set_title(
        "Roll: {:6.1f}°   Pitch: {:6.1f}°   Yaw: {:6.1f}°".format(roll, pitch, yaw),
        fontsize=12
    )


ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)
plt.tight_layout()
plt.show()
