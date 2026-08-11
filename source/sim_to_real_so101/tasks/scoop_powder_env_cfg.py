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
"""Scoop-powder task: pick up a scoop, dip it into a flavor container, tilt it
over a cup to "pour", repeat until the target amount is deposited.

There is no granular/powder physics in Isaac Lab, so the "amount deposited"
signal is a deliberately crude analytic proxy (see
``sim_to_real_so101.mdp.terms.scoop_deposit_grams``), not a physically
accurate simulation of scooping. It exists to teach the coarse motion
(reach the right container, carry, tilt over the cup, repeat) and to expose
a scale-shaped observation under the same key the real Tab5 channel uses -
real gram accuracy has to come from real data (see docker/real/scripts/
so101_record_scoop.py), not this sim.

No custom USD assets exist yet for the containers/scoop/cup, so this scene
uses simple procedural primitives (cylinders) as placeholders - swap in real
assets later without changing the task logic.
"""
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from sim_to_real_so101.assets.so101 import S0101_CONTACT_GRASP_CFG
from sim_to_real_so101.mdp import (
    reset_scoop_powder_objects,
    scoop_grasped,
    scoop_deposit_grams,
    scoop_target_reached_termination,
    time_out,
)

from .task_env_cfg import (
    SO101TaskSceneCfg,
    SO101TaskEnvCfg,
    TaskEventCfg,
    TaskObservationsCfg,
)

# Mat surface sits at roughly this world Z (matches VIAL_SPAWN_Z in
# vials_to_rack_env_cfg.py - same mat, same resting height convention).
MAT_SURFACE_Z = 0.05

CONTAINER_RADIUS = 0.03
CONTAINER_HEIGHT = 0.06
CUP_RADIUS = 0.045
CUP_HEIGHT = 0.05
SCOOP_RADIUS = 0.008
SCOOP_LENGTH = 0.09

container_base = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Container",
    spawn=sim_utils.CylinderCfg(
        radius=CONTAINER_RADIUS,
        height=CONTAINER_HEIGHT,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.23, -0.12, MAT_SURFACE_Z + CONTAINER_HEIGHT / 2),
    ),
)

cup = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cup",
    spawn=sim_utils.CylinderCfg(
        radius=CUP_RADIUS,
        height=CUP_HEIGHT,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.20, 0.10, MAT_SURFACE_Z + CUP_HEIGHT / 2),
    ),
)

scoop = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Scoop",
    spawn=sim_utils.CylinderCfg(
        radius=SCOOP_RADIUS,
        height=SCOOP_LENGTH,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.75)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.23, 0.02, MAT_SURFACE_Z + SCOOP_RADIUS),
        # lying on its side (long axis horizontal) so the gripper can grasp
        # it the same way it grasps a vial lying on the mat
        rot=euler_angles_to_quat(np.array([0, 90, 0]), degrees=True),
    ),
)


@configclass
class ScoopPowderSceneCfg(SO101TaskSceneCfg):
    # Override robot with contact sensors enabled (needed for grasp detection)
    robot: ArticulationCfg = S0101_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    container_1 = container_base.replace()
    container_1.prim_path = "{ENV_REGEX_NS}/Container_1"
    container_1.init_state.pos = (0.23, -0.12, MAT_SURFACE_Z + CONTAINER_HEIGHT / 2)

    container_2 = container_base.replace()
    container_2.prim_path = "{ENV_REGEX_NS}/Container_2"
    container_2.init_state.pos = (0.23, -0.04, MAT_SURFACE_Z + CONTAINER_HEIGHT / 2)
    container_2.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.85))

    cup = cup.replace()
    cup.prim_path = "{ENV_REGEX_NS}/Cup"

    scoop = scoop.replace()
    scoop.prim_path = "{ENV_REGEX_NS}/Scoop"

    # Contact sensor on gripper jaw to detect scoop grasping
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Scoop",
        ],
    )


@configclass
class ScoopPowderEventCfg(TaskEventCfg):
    """Configuration for events."""

    reset_scoop_scene = EventTerm(
        func=reset_scoop_powder_objects,
        mode="reset",
        params={
            "objects": {
                "container_1": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "yaw": (-0.3, 0.3)},
                "container_2": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "yaw": (-0.3, 0.3)},
                "cup": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "yaw": (-0.3, 0.3)},
                "scoop": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "roll": (-0.2, 0.2), "yaw": (-0.3, 0.3)},
            },
            "fixed_z": {
                "container_1": MAT_SURFACE_Z + CONTAINER_HEIGHT / 2,
                "container_2": MAT_SURFACE_Z + CONTAINER_HEIGHT / 2,
                "cup": MAT_SURFACE_Z + CUP_HEIGHT / 2,
                "scoop": MAT_SURFACE_Z + SCOOP_RADIUS,
            },
        },
    )


@configclass
class ScoopPowderObservationsCfg(TaskObservationsCfg):
    """Configuration for observations."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        scoop_grasped_obs = ObsTerm(
            func=scoop_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "scoop_name": "scoop",
                "min_height": MAT_SURFACE_Z + 0.01,
                "warmup_steps": 30,
                "force_threshold": 2,  # N
            },
        )

        # Synthetic scale reading - same "grams" concept the real Tab5 channel
        # exposes (see docker/real/scripts/so101_control.py's scale.grams),
        # so sim and real data share one observation schema.
        scale_grams_obs = ObsTerm(
            func=scoop_deposit_grams,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "scoop_name": "scoop",
                "containers": ["container_1", "container_2"],
                "cup_name": "cup",
                "container_radius": CONTAINER_RADIUS + 0.015,
                "container_z_range": (0.0, CONTAINER_HEIGHT + 0.03),
                "cup_radius": CUP_RADIUS + 0.015,
                "cup_z_range": (0.0, CUP_HEIGHT + 0.05),
                "warmup_steps": 30,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class ScoopPowderTerminationsCfg:
    """Termination terms for the scoop-powder evaluation task."""

    time_out = DoneTerm(
        func=time_out,
        time_out=True,
    )

    # Fixed target for now - real per-instruction targets ("10g banana") are
    # a language-conditioning concern at the GR00T/data level, not something
    # this Isaac Lab termination needs to parse.
    success = DoneTerm(
        func=scoop_target_reached_termination,
        time_out=False,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
            "target_grams": 10.0,
            "tolerance_grams": 1.0,
            "scoop_name": "scoop",
            "containers": ["container_1", "container_2"],
            "cup_name": "cup",
            "confirm_steps": 25,
            "container_radius": CONTAINER_RADIUS + 0.015,
            "container_z_range": (0.0, CONTAINER_HEIGHT + 0.03),
            "cup_radius": CUP_RADIUS + 0.015,
            "cup_z_range": (0.0, CUP_HEIGHT + 0.05),
            "warmup_steps": 30,
        },
    )


@configclass
class ScoopPowderEnvCfg(SO101TaskEnvCfg):
    """Base config."""

    scene: ScoopPowderSceneCfg = ScoopPowderSceneCfg()
    events: ScoopPowderEventCfg = ScoopPowderEventCfg()
    observations: ScoopPowderObservationsCfg = ScoopPowderObservationsCfg()


@configclass
class ScoopPowderEvalEnvCfg(ScoopPowderEnvCfg):
    """Eval config - adds the (fixed-target) success termination."""

    terminations: ScoopPowderTerminationsCfg = ScoopPowderTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 450 / 60.0
