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
Real-hardware teleop data collection for the scooping task.

Teleoperates the follower arm via the leader arm, recording camera images,
robot joint state, the M5Stack Tab5 scale reading, and a per-episode task
instruction (e.g. "take 10g banana and 20g vanilla") into a LeRobot dataset.

Usage:
    python so101_record_scoop.py \\
        --leader.type=so101_leader --leader.port=$TELEOP_PORT --leader.id=$TELEOP_ID \\
        --robot.type=so101_follower --robot.port=$ROBOT_PORT --robot.id=$ROBOT_ID \\
        --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_EXTERNAL, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $CAMERA_GRIPPER, width: 640, height: 480, fps: 30}}" \\
        --repo_id=local/scoop_powder --root=./datasets/scoop_powder

At the "task>" prompt, type the instruction for the next episode (e.g.
"take 10g banana") and press Enter to start recording. While recording:
    s        stop and save the episode
    c        cancel and discard the episode
    q        stop, save, and quit the whole session
Ctrl-C also ends the session (does not save a partial episode).

Deliberately include corrective demonstrations (intentional under/over-scoop,
check the scale, scoop again to correct) - this is imitation learning, so the
policy can only learn closed-loop correction behavior it actually sees
demonstrated.
"""

from dataclasses import asdict, dataclass
import logging
from pprint import pformat
import select
import sys
import termios
import time
import tty

import numpy as np
import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, so101_follower  # noqa: F401
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.teleoperators.so101_leader import SO101LeaderConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

from sim_to_real_so101.utils.scale_reader import ScaleReader, find_scale_port


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


@dataclass
class RecordConfig:
    leader: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    repo_id: str = "local/scoop_powder"
    root: str = "./datasets/scoop_powder"
    fps: int = 30
    scale_port: str | None = None  # M5Stack Tab5 scale serial port; auto-detected if unset
    disable_scale: bool = False  # skip scale entirely, even if one is detected


def _check_key(fd: int) -> str | None:
    """Non-blocking single-char stdin read. Returns None if nothing is pending."""
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        return sys.stdin.read(1)
    return None


def build_features(cameras: dict, fps: int) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "fps": fps,
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "fps": fps,
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "observation.scale_grams": {
            "dtype": "float32",
            "fps": fps,
            "shape": (1,),
            "names": ["grams"],
        },
    }
    for name, cam_cfg in (cameras or {}).items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (cam_cfg.height, cam_cfg.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def record_episode(
    leader: Teleoperator,
    robot: Robot,
    scale: ScaleReader | None,
    dataset: LeRobotDataset,
    task: str,
    fps: int,
    camera_names: list[str],
) -> str:
    """Runs one recording loop until the operator stops/cancels it.

    Puts the terminal in cbreak mode for the duration of the loop so single
    keypresses (no Enter needed) can be read without blocking the control
    loop - restored to normal (cooked) mode on return so the next task>
    input() prompt behaves normally.

    Returns "saved", "cancelled", or "quit".
    """
    dt = 1.0 / fps
    print(f"[RECORDING] task={task!r} -- press 's' to save, 'c' to cancel, 'q' to save+quit")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    frame_count = 0
    episode_start = time.time()
    last_heartbeat = episode_start
    try:
        while True:
            tic = time.time()

            if tic - last_heartbeat > 1.0:
                elapsed = tic - episode_start
                print(
                    f"\r[RECORDING] {frame_count} frames, "
                    f"{frame_count / elapsed:.1f} fps avg   ",
                    end="",
                    flush=True,
                )
                last_heartbeat = tic

            action = leader.get_action()
            robot.send_action(action)
            obs = robot.get_observation()

            state = np.array([obs[k] for k in JOINT_NAMES], dtype=np.float32)
            action_arr = np.array([action[k] for k in JOINT_NAMES], dtype=np.float32)
            grams = scale.grams if scale is not None else None

            frame = {
                "observation.state": state,
                "action": action_arr,
                "observation.scale_grams": np.array(
                    [grams if grams is not None else 0.0], dtype=np.float32
                ),
                "task": task,
            }
            for cam in camera_names:
                frame[f"observation.images.{cam}"] = obs[cam]

            dataset.add_frame(frame)
            frame_count += 1

            key = _check_key(fd)
            if key == "c":
                dataset.clear_episode_buffer()
                print(f"\n[CANCELLED] discarded {frame_count} frames")
                return "cancelled"
            if key == "s":
                dataset.save_episode()
                print(f"\n[SAVED] episode with {frame_count} frames")
                return "saved"
            if key == "q":
                dataset.save_episode()
                print(f"\n[SAVED] episode with {frame_count} frames")
                return "quit"

            toc = time.time()
            if toc - tic < dt:
                time.sleep(dt - (toc - tic))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@draccus.wrap()
def main(cfg: RecordConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    leader = make_teleoperator_from_config(cfg.leader)
    robot = make_robot_from_config(cfg.robot)

    leader.connect()
    robot.connect()

    scale = None
    if not cfg.disable_scale:
        scale_port = cfg.scale_port or find_scale_port()
        if scale_port:
            scale = ScaleReader(scale_port)
            scale.start()
            logging.info(f"Scale connected on {scale_port}")
        else:
            logging.info("No scale detected - recording without weight readings.")

    camera_names = list(cfg.robot.cameras.keys()) if cfg.robot.cameras else []
    features = build_features(cfg.robot.cameras, cfg.fps)

    dataset = LeRobotDataset.create(
        cfg.repo_id,
        fps=cfg.fps,
        features=features,
        root=cfg.root,
        robot_type="so101_follower",
    )

    try:
        while True:
            try:
                task = input("\ntask> ").strip()
            except EOFError:
                break
            if not task or task.lower() == "q":
                break

            result = record_episode(leader, robot, scale, dataset, task, cfg.fps, camera_names)
            if result == "quit":
                break
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    finally:
        dataset.finalize()
        if scale is not None:
            scale.stop()
        # Hardware disconnect can fail on its own (e.g. a servo "Overload
        # error" if the arm was under load at the moment torque is
        # disabled) - don't let that mask the fact that the dataset above
        # was already safely finalized.
        for name, dev in (("leader", leader), ("robot", robot)):
            try:
                dev.disconnect()
            except Exception as e:
                logging.warning(f"{name}.disconnect() failed: {e}")
        logging.info(f"Session done. {dataset.num_episodes} episodes saved to {cfg.root}")


if __name__ == "__main__":
    main()
