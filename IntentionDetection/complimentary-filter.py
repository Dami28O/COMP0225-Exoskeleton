#!/usr/bin/env python

import os
import sys
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "icm20948-python-main"))

from icm20948 import ICM20948

# Complementary filter coefficient: higher = trust gyro more, lower = trust accel more
ALPHA = 0.98

imu = ICM20948()

# Initialise angles from accelerometer on first read
ax, ay, az, gx, gy, gz = imu.read_accelerometer_gyro_data()
roll  = math.degrees(math.atan2(ay, az))
pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

last_time = time.time()

print("Complementary filter running. Press Ctrl+C to exit.\n")
print("Initial angles from accelerometer:")
print("  Roll:  {:7.2f} deg".format(roll))
print("  Pitch: {:7.2f} deg\n")

while True:
    ax, ay, az, gx, gy, gz = imu.read_accelerometer_gyro_data()

    now = time.time()
    dt = now - last_time
    last_time = now

    # Accelerometer-derived angles (absolute, noisy)
    accel_roll  = math.degrees(math.atan2(ay, az))
    accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

    # Complementary filter: blend gyro integration with accel estimate
    roll  = ALPHA * (roll  + gx * dt) + (1 - ALPHA) * accel_roll
    pitch = ALPHA * (pitch + gy * dt) + (1 - ALPHA) * accel_pitch

    print("Roll: {:7.2f} deg   Pitch: {:7.2f} deg".format(roll, pitch))

    time.sleep(0.01)
