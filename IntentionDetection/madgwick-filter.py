#!/usr/bin/env python

import os
import sys
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "icm20948-python-main"))

from icm20948 import ICM20948

BETA = 0.1  # gradient descent gain: higher = faster correction, more noise sensitivity

imu = ICM20948()

# --- Initialise quaternion from first sensor reading (NED frame) ---
ax, ay, az, _, _, _ = imu.read_accelerometer_gyro_data()
mx, my, mz = imu.read_magnetometer_data()

roll0  = math.atan2(ay, math.sqrt(ax**2 + az**2))
pitch0 = math.atan2(-ax, az)
yaw0   = math.atan2(
    mx * math.cos(pitch0) + mz * math.sin(pitch0),
    mx * math.sin(pitch0) * math.sin(roll0) + my * math.cos(roll0) - mz * math.cos(pitch0) * math.sin(roll0)
)

# Euler (ZYX) to quaternion
cr, sr = math.cos(roll0 / 2),  math.sin(roll0 / 2)
cp, sp = math.cos(pitch0 / 2), math.sin(pitch0 / 2)
cy, sy = math.cos(yaw0 / 2),   math.sin(yaw0 / 2)

qw = cr*cp*cy + sr*sp*sy
qx = sr*cp*cy - cr*sp*sy
qy = cr*sp*cy + sr*cp*sy
qz = cr*cp*sy - sr*sp*cy

last_time = time.time()

print("Madgwick filter (9-axis) running. Press Ctrl+C to exit.\n")
print("Initial angles (deg): Roll={:.2f}  Pitch={:.2f}  Yaw={:.2f}\n".format(
    math.degrees(roll0), math.degrees(pitch0), math.degrees(yaw0)))

while True:
    ax, ay, az, gx, gy, gz = imu.read_accelerometer_gyro_data()
    mx, my, mz = imu.read_magnetometer_data()

    now = time.time()
    dt = now - last_time
    last_time = now

    # Gyroscope deg/s → rad/s
    gx = math.radians(gx)
    gy = math.radians(gy)
    gz = math.radians(gz)

    # Quaternion rate of change from gyroscope: 0.5 * q ⊗ [0, gx, gy, gz]
    qdot_w = 0.5 * (-qx*gx - qy*gy - qz*gz)
    qdot_x = 0.5 * ( qw*gx + qy*gz - qz*gy)
    qdot_y = 0.5 * ( qw*gy - qx*gz + qz*gx)
    qdot_z = 0.5 * ( qw*gz + qx*gy - qy*gx)

    # Normalise accelerometer; skip frame if invalid
    a_norm = math.sqrt(ax*ax + ay*ay + az*az)
    if a_norm == 0:
        continue
    ax /= a_norm; ay /= a_norm; az /= a_norm

    # Normalise magnetometer; skip frame if invalid
    m_norm = math.sqrt(mx*mx + my*my + mz*mz)
    if m_norm == 0:
        continue
    mx /= m_norm; my /= m_norm; mz /= m_norm

    # Rotate magnetometer to earth frame to get reference field direction
    hx = 2*mx*(0.5 - qy*qy - qz*qz) + 2*my*(qx*qy - qw*qz) + 2*mz*(qx*qz + qw*qy)
    hy = 2*mx*(qx*qy + qw*qz) + 2*my*(0.5 - qx*qx - qz*qz) + 2*mz*(qy*qz - qw*qx)
    hz = 2*mx*(qx*qz - qw*qy) + 2*my*(qy*qz + qw*qx) + 2*mz*(0.5 - qx*qx - qy*qy)
    bx = math.sqrt(hx*hx + hy*hy)  # project onto xz plane → by = 0
    by = 0.0
    bz = hz

    # --- Objective functions ---
    # 6-axis (accel): expected gravity direction vs measured
    f1 = 2*(qx*qz - qw*qy)       - ax
    f2 = 2*(qw*qx + qy*qz)       - ay
    f3 = 2*(0.5 - qx*qx - qy*qy) - az

    # 9-axis extension (mag): expected magnetic field direction vs measured
    f4 = 2*bx*(0.5 - qy*qy - qz*qz) + 2*by*(qw*qz + qx*qy) + 2*bz*(qx*qz - qw*qy) - mx
    f5 = 2*bx*(qx*qy - qw*qz) + 2*by*(0.5 - qx*qx - qz*qz) + 2*bz*(qw*qx + qy*qz) - my
    f6 = 2*bx*(qw*qy + qx*qz) + 2*by*(qy*qz - qw*qx) + 2*bz*(0.5 - qx*qx - qy*qy) - mz

    # --- Gradient: J^T * f ---
    # 6-axis Jacobian:
    #   J[0] = [-2qy,  2qz, -2qw,  2qx]  (wrt f1)
    #   J[1] = [ 2qx,  2qw,  2qz,  2qy]  (wrt f2)
    #   J[2] = [   0, -4qx, -4qy,    0]  (wrt f3)
    # 9-axis Jacobian:
    #   J[3] = [ 2by*qz-2bz*qy,   2by*qy+2bz*qz,  -4bx*qy+2by*qx-2bz*qw,  -4bx*qz+2by*qw+2bz*qx]  (wrt f4)
    #   J[4] = [-2bx*qz+2bz*qx,   2bx*qy-4by*qx+2bz*qw,  2bx*qx+2bz*qz,  -2bx*qw-4by*qz+2bz*qy]  (wrt f5)
    #   J[5] = [ 2bx*qy-2by*qx,   2bx*qz-2by*qw-4bz*qx,  2bx*qw+2by*qz-4bz*qy,  2bx*qx+2by*qy]   (wrt f6)
    
    # computes jacobian implicitly (instead of doing matrix multiplication)
    grad_w = (-2*qy*f1 + 2*qx*f2
              + (2*by*qz - 2*bz*qy)*f4
              + (-2*bx*qz + 2*bz*qx)*f5
              + (2*bx*qy - 2*by*qx)*f6)

    grad_x = (2*qz*f1 + 2*qw*f2 - 4*qx*f3
              + (2*by*qy + 2*bz*qz)*f4
              + (2*bx*qy - 4*by*qx + 2*bz*qw)*f5
              + (2*bx*qz - 2*by*qw - 4*bz*qx)*f6)

    grad_y = (-2*qw*f1 + 2*qz*f2 - 4*qy*f3
              + (-4*bx*qy + 2*by*qx - 2*bz*qw)*f4
              + (2*bx*qx + 2*bz*qz)*f5
              + (2*bx*qw + 2*by*qz - 4*bz*qy)*f6)

    grad_z = (2*qx*f1 + 2*qy*f2
              + (-4*bx*qz + 2*by*qw + 2*bz*qx)*f4
              + (-2*bx*qw - 4*by*qz + 2*bz*qy)*f5
              + (2*bx*qx + 2*by*qy)*f6)

    # Normalise gradient
    g_norm = math.sqrt(grad_w**2 + grad_x**2 + grad_y**2 + grad_z**2)
    grad_w /= g_norm; grad_x /= g_norm; grad_y /= g_norm; grad_z /= g_norm

    # Apply gradient correction to quaternion rate then integrate
    qdot_w -= BETA * grad_w
    qdot_x -= BETA * grad_x
    qdot_y -= BETA * grad_y
    qdot_z -= BETA * grad_z

    qw += qdot_w * dt
    qx += qdot_x * dt
    qy += qdot_y * dt
    qz += qdot_z * dt

    # Normalise quaternion
    q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw /= q_norm; qx /= q_norm; qy /= q_norm; qz /= q_norm

    # Quaternion to Euler angles (degrees)
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))

    print("Roll: {:7.2f} deg   Pitch: {:7.2f} deg   Yaw: {:7.2f} deg".format(roll, pitch, yaw))

    time.sleep(0.01)
