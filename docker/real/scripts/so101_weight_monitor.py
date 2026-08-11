#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Combined robot + scale weight monitor.

Tails the M5Stack Tab5's JSON weight stream (see docker/real/scripts's sibling
Tab5 main.py) and, if the SO101 follower arm is connected, its live joint
state, printing both to the terminal. Robot connection is optional - if the
arm isn't plugged in, weight-only monitoring still works.

Usage:
    python so101_weight_monitor.py \\
        --robot.port=$ROBOT_PORT \\
        --robot.id=$ROBOT_ID
"""

import argparse
import sys
import time

from lerobot.robots.so101_follower import SO101Follower as SOFollower
from lerobot.robots.so101_follower import SO101FollowerConfig as SOFollowerRobotConfig
from sim_to_real_so101.utils.scale_reader import ScaleReader, find_scale_port


def main():
    parser = argparse.ArgumentParser(description="Combined robot + scale weight monitor")
    parser.add_argument("--robot.port", type=str, default=None, help="Robot serial port")
    parser.add_argument("--robot.id", type=str, default=None, help="Robot ID for calibration lookup")
    parser.add_argument(
        "--scale-port", type=str, default=None, help="Tab5 serial port (auto-detected if omitted)"
    )
    args = parser.parse_args()

    robot_port = getattr(args, "robot.port")
    robot_id = getattr(args, "robot.id")

    scale_port = args.scale_port or find_scale_port()
    if scale_port is None:
        print("ERROR: Tab5 scale device not found (no /dev/serial/by-id/usb-Espressif_* entry).")
        sys.exit(1)
    print(f"Scale: {scale_port}")
    scale = ScaleReader(scale_port)
    scale.start()

    robot = None
    if robot_port and robot_id:
        try:
            robot = SOFollower(SOFollowerRobotConfig(id=robot_id, port=robot_port))
            robot.connect()
            print(f"Robot: connected on {robot_port} (id={robot_id})")
        except Exception as e:
            print(f"Robot: not connected ({e}) - continuing with weight-only monitoring")
            robot = None
    else:
        print("Robot: no --robot.port/--robot.id given - weight-only monitoring")

    try:
        while True:
            grams = scale.grams
            grams_str = f"{grams:7.2f} g" if grams is not None else "  ---.-- g"

            if robot is not None and robot.is_connected:
                obs = robot.get_observation()
                joints = " ".join(
                    f"{k.replace('.pos', '')}={v:7.2f}" for k, v in obs.items() if k.endswith(".pos")
                )
                print(f"\rweight: {grams_str} | {joints}", end="", flush=True)
            else:
                print(f"\rweight: {grams_str} | robot: not connected", end="", flush=True)

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        scale.stop()
        if robot is not None and robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
