#!/usr/bin/env python

import os
import sys
import time
import math
import csv
import serial
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "icm20948-python-main"))

from icm20948 import ICM20948

# --- Config ---
BETA = 0.1

SERIAL_PORT           = "/dev/ttyUSB0"
BAUD_RATE             = 9600

PITCH_THRESHOLD       = 15.0   # deg   — rule 1: forward lean
PITCH_VEL_THRESHOLD   = 20.0   # deg/s — rule 2: actively moving into the lean
# VERT_ACCEL_THRESHOLD = 0.15   # g    — rule 3: vertical accel (reserved for later)
DURATION_THRESHOLD    = 0.300  # s    — rule 4: sustained for 300ms
PITCH_ACCEL_WINDOW    = 10     # samples (~100ms at 100Hz) — rule 5: smoothed accel
COOLDOWN              = 5.0    # s    — minimum gap between motor on commands

# --- IMU & serial setup ---
imu = ICM20948()
# ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

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

# --- Intent detection state ---
last_motor_on_time = -COOLDOWN   # allows immediate first trigger
pitch_above_since  = None        # timestamp when pitch first exceeded threshold
pitch_accel_buf    = deque(maxlen=PITCH_ACCEL_WINDOW)
prev_pitch_vel     = 0.0

start_time = time.time()
last_time  = start_time

csv_path = os.path.join(os.path.dirname(__file__), "detections.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["elapsed_s", "pitch_deg", "pitch_vel_degs", "avg_pitch_accel_degs2"])

print("Intent detection running. Press Ctrl+C to exit.")
print("Logging detections to:", csv_path, "\n")

while True:
    ax, ay, az, gx, gy, gz = imu.read_accelerometer_gyro_data()
    mx, my, mz = imu.read_magnetometer_data()

    now = time.time()
    dt  = now - last_time
    last_time = now

    # Gyroscope deg/s → rad/s for Madgwick update
    gx_r = math.radians(gx)
    gy_r = math.radians(gy)
    gz_r = math.radians(gz)

    # Quaternion rate of change from gyroscope: 0.5 * q ⊗ [0, gx, gy, gz]
    qdot_w = 0.5 * (-qx*gx_r - qy*gy_r - qz*gz_r)
    qdot_x = 0.5 * ( qw*gx_r + qy*gz_r - qz*gy_r)
    qdot_y = 0.5 * ( qw*gy_r - qx*gz_r + qz*gx_r)
    qdot_z = 0.5 * ( qw*gz_r + qx*gy_r - qy*gx_r)

    a_norm = math.sqrt(ax*ax + ay*ay + az*az)
    if a_norm == 0:
        continue
    ax /= a_norm; ay /= a_norm; az /= a_norm

    m_norm = math.sqrt(mx*mx + my*my + mz*mz)
    if m_norm == 0:
        continue
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

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))

    # gy (deg/s, unconverted) ≈ pitch angular velocity
    pitch_vel = gy

    # Rule 5: sliding window average of pitch acceleration
    pitch_accel = (pitch_vel - prev_pitch_vel) / dt if dt > 0 else 0.0
    pitch_accel_buf.append(pitch_accel)
    prev_pitch_vel = pitch_vel
    avg_pitch_accel = sum(pitch_accel_buf) / len(pitch_accel_buf)

    # --- Intent detection ---
    if (pitch > PITCH_THRESHOLD             # rule 1
            and pitch_vel > PITCH_VEL_THRESHOLD  # rule 2
            and avg_pitch_accel > 0):            # rule 5
        if pitch_above_since is None:
            pitch_above_since = now
        if (now - pitch_above_since) >= DURATION_THRESHOLD:  # rule 4
            if (now - last_motor_on_time) >= COOLDOWN:
                # ser.write(b"motor on\n")
                last_motor_on_time = now
                elapsed = now - start_time
                csv_writer.writerow(["{:.3f}".format(elapsed),
                                     "{:.2f}".format(pitch),
                                     "{:.2f}".format(pitch_vel),
                                     "{:.2f}".format(avg_pitch_accel)])
                csv_file.flush()
                print("Intent detected — motor on  (t={:.3f}s)".format(elapsed))
    else:
        pitch_above_since = None

    # --- Status line ---
    cooldown_remaining = max(0.0, COOLDOWN - (now - last_motor_on_time))
    status = "COOLDOWN {:.1f}s".format(cooldown_remaining) if cooldown_remaining > 0 else "ready"
    print("Pitch: {:6.2f} deg  PitchVel: {:6.2f} deg/s  AvgAccel: {:6.2f} deg/s²  [{}]".format(
        pitch, pitch_vel, avg_pitch_accel, status))

    time.sleep(0.01)
