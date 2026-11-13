"""Bimanual YAM embodiment for RoboEval.

The YAM (Yet-Another-Manipulator) bimanual arm, ported from allenai/molmoact2's
`sim_eval` (ManiSkill/SAPIEN) into RoboEval's MuJoCo robot abstraction so it can
be swapped in for the Panda in RoboCasa (and other) scenes.

Model source
------------
MuJoCo MJCF + meshes come from the HF dataset
`TreeePlanter/molmoact2-sim-eval-assets` (`assets/yam/yam_mujoco/`), downloaded
into `roboeval/envs/xmls/yam/yam_mujoco/`. The whole bimanual robot (base + two
6-DoF arms + two parallel grippers) is one *flattened* MJCF, so this config uses
RoboEval's **single-model** robot path (`RobotConfig.model` set), NOT the
per-arm composition path the Panda uses.

`bimanual_yam_linear_flattened_cams.xml` is the flattened model with the three
YAM cameras baked in (regenerate via `task_planning/bake_yam_cameras.py`).

Camera viewpoints (ported from molmoact2)
-----------------------------------------
molmoact2 defines three RealSense-style cameras as (pos, quat, hfov) on robot
links. Two conversions are applied (see `bake_yam_cameras.py`):
  * fov: molmoact2 values are RealSense HORIZONTAL fov (top≈D435 color 69.4°,
    wrist≈D435 depth 87°); MuJoCo uses VERTICAL fovy → converted at aspect 640/360.
  * frame: SAPIEN camera looks +x (up +z); MuJoCo looks -z (up +y) → change-of-basis.
The values below are the MuJoCo-converted poses (also baked into the model).

Integration status
-------------------
Verified: model + meshes load in MuJoCo (nq=16); the three cameras render correct
top / left-wrist / right-wrist views. The config encodes arms, grippers, IK roots
and joint limits from the real model. Driving it under RoboEval's controller/IK in
a full RoboCasa scene still needs the single-model robot path
(`Robot.__init__` else-branch / `_on_loaded`) to register arms+grippers — see
`HANDOFF` notes. The replay demos are recorded Panda trajectories, so YAM applies
to the **authored `RobocasaBakedScene` / fresh-rollout** setups, not Panda replay.
"""
from __future__ import annotations

import numpy as np

from roboeval.const import ASSETS_PATH, HandSide
from roboeval.robots.config import (
    ArmConfig,
    FloatingBaseConfig,
    GripperConfig,
    RobotConfig,
    RobotIKConfig,
)
from roboeval.robots.robot import Robot

# Hardware/API constants used by the YAM hardware adapter. The simulation model
# below comes from the imported RoboEval YAM embodiment, while these values keep
# the local hardware bridge import-compatible with the rest of this workspace.
YAM_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
YAM_POSITION_LIMITS_RAD = (
    (-2.61799, 3.05433),
    (0.0, 3.66519),
    (0.0, 3.66519),
    (-1.5708, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)
YAM_SDK_POSITION_MARGIN_RAD = 0.15
YAM_HARDWARE_POSITION_LIMITS_RAD = tuple(
    (lower + YAM_SDK_POSITION_MARGIN_RAD, upper - YAM_SDK_POSITION_MARGIN_RAD)
    for lower, upper in YAM_POSITION_LIMITS_RAD
)
YAM_VELOCITY_LIMITS_RAD_S = (11.0, 11.0, 11.0, 33.0, 33.0, 33.0)
YAM_TORQUE_LIMITS_NM = (28.0, 28.0, 28.0, 10.0, 10.0, 10.0)
YAM_KP = (80.0, 80.0, 80.0, 10.0, 10.0, 10.0)
YAM_KD = (5.0, 5.0, 5.0, 1.5, 1.5, 1.5)
YAM_GRAVITY_COMP_FACTOR = (1.0, 1.1, 1.1, 1.2, 1.0, 1.0)
YAM_GRAV_COMP_KD = (0.1, 0.1, 0.1, 0.3, 0.05, 0.05)
YAM_COULOMB_FRICTION_NM = (0.3, 0.3, 0.3, 0.06, 0.06, 0.06)
YAM_USE_COULOMB_FRICTION = False
YAM_GRIPPER_RANGE = (0.0, 1.0)
YAM_GRIPPER_FORCE_LIMIT_N = 50.0
YAM_GRIPPER_KP = 20.0
YAM_GRIPPER_KD = 0.5
YAM_GRIPPER_TRAVEL_M = 0.0475
YAM_POLICY_CONTROL_FREQUENCY_HZ = 30.0
YAM_SIM_GRIPPER_FORCE_LIMIT_N = YAM_GRIPPER_FORCE_LIMIT_N
YAM_SIM_GRIPPER_KP = 2000.0
YAM_SIM_GRIPPER_KD = 40.0
YAM_SIM_PAD_FRICTION = (3.0, 0.2, 0.01)
YAM_SIM_PAD_CONDIM = 6
YAM_SIM_PAD_SOLREF = (0.002, 0.5)
YAM_SIM_PAD_SOLIMP = (0.9, 0.95, 0.001)
YAM_SIM_ARM_GAINS = (40.0, 40.0, 40.0, 20.0, 10.0, 10.0)
YAM_SIM_ARM_DAMPING = (6.0, 6.0, 6.0, 3.0, 3.0, 3.0)
YAM_SIM_ARM_VELOCITY_LIMITS = {
    f"{side}_joint{joint_idx}": limit
    for side in ("left", "right")
    for joint_idx, limit in enumerate(YAM_VELOCITY_LIMITS_RAD_S, start=1)
}

# --------------------------------------------------------------------------- #
# Model path
# --------------------------------------------------------------------------- #
YAM_MODEL = ASSETS_PATH / "yam" / "yam_mujoco" / "bimanual_yam_linear_flattened_cams.xml"

# --------------------------------------------------------------------------- #
# Cameras — molmoact2 viewpoints, converted to MuJoCo (baked into YAM_MODEL).
# Kept here too for ObservationConfig lookup / re-baking. fov = vertical fovy(deg).
# --------------------------------------------------------------------------- #
YAM_CAMERA_CONFIG = {
    "top_cam": {
        "manual": True,
        "parent": "bimanual_base",
        # First robot camera must match Bimanual Panda's cameraid=1/head view.
        # YAM reuses Panda's task preset pose, then shifts 20 cm toward the
        # workspace/shelf. This local pose preserves Panda's cameraid=1 world view.
        "position": (-0.6, 0.0, 0.84),
        "quaternion": (0.6533, 0.2707, -0.2707, -0.6533),
        "fov": 45.0,
        "source": {"matches": "Bimanual Panda head camera"},
    },
    "left_cam": {
        "manual": True,
        "parent": "left_link_6",
        "position": (0.0, 0.09, 0.06),
        "quaternion": (0.0, 0.0, 0.9659, 0.2588),
        "fov": 56.2,                                       # 87° HFOV -> fovy @16:9
        "source": {"sapien_quat": (0.6124, -0.3536, -0.3536, -0.6124), "hfov_deg": 87.0},
    },
    "right_cam": {
        "manual": True,
        "parent": "right_link_6",
        "position": (0.0, 0.09, 0.06),
        "quaternion": (0.0, 0.0, 0.9659, 0.2588),
        "fov": 56.2,
        "source": {"sapien_quat": (0.6124, -0.3536, -0.3536, -0.6124), "hfov_deg": 87.0},
    },
}
YAM_CAMERAS = list(YAM_CAMERA_CONFIG.keys())

# --------------------------------------------------------------------------- #
# Arms — both 6-DoF, defined against names inside the flattened model.
# --------------------------------------------------------------------------- #
YAM_LEFT_ARM = ArmConfig(
    site="left_tcp_site",
    body="left_arm",          # single-model: wrap the pre-loaded arm root
    links=["left_link_1", "left_link_2", "left_link_3",
           "left_link_4", "left_link_5", "left_link_6"],  # descendants of body
    joints=["left_joint1", "left_joint2", "left_joint3",
            "left_joint4", "left_joint5", "left_joint6"],
    actuators=["left_joint1", "left_joint2", "left_joint3",
               "left_joint4", "left_joint5", "left_joint6"],
)

YAM_RIGHT_ARM = ArmConfig(
    site="right_tcp_site",
    body="right_arm",         # single-model: wrap the pre-loaded arm root
    links=["right_link_1", "right_link_2", "right_link_3",
           "right_link_4", "right_link_5", "right_link_6"],  # descendants of body
    joints=["right_joint1", "right_joint2", "right_joint3",
            "right_joint4", "right_joint5", "right_joint6"],
    actuators=["right_joint1", "right_joint2", "right_joint3",
               "right_joint4", "right_joint5", "right_joint6"],
)

YAM_ACTUATORS = {
    "left_joint1": True, "left_joint2": True, "left_joint3": True,
    "left_joint4": True, "left_joint5": True, "left_joint6": True,
    "right_joint1": True, "right_joint2": True, "right_joint3": True,
    "right_joint4": True, "right_joint5": True, "right_joint6": True,
}

# --------------------------------------------------------------------------- #
# Grippers — parallel-jaw, two slide tips per side. Native actuator ctrlrange is
# [-0.0475, 0], where the negative end is open. RoboEval actions stay in the
# local convention: 0=open, 1=closed.
# --------------------------------------------------------------------------- #
YAM_GRIPPER = GripperConfig(
    body="left_link_6",                       # embedded gripper (no standalone model)
    actuators=["left_gripper_left_tip", "left_gripper_right_tip"],
    range=np.array([0.0, 1.0]),
    pad_bodies=["left_link_left_finger", "left_link_right_finger"],
    # The armor box encloses the whole finger link, so it overlaps objects the
    # open hand is merely hovering over — only the tip colliders grip.
    pad_geom_exclude=["_finger_armor"],
    reverse_control=False,                    # RoboEval convention: 0=open, 1=closed
    discrete=False,                           # molmoact2 emits continuous [0,1]
)

YAM_RIGHT_GRIPPER = GripperConfig(
    body="right_link_6",
    actuators=["right_gripper_left_tip", "right_gripper_right_tip"],
    range=np.array([0.0, 1.0]),               # see YAM_GRIPPER
    pad_bodies=["right_link_left_finger", "right_link_right_finger"],
    pad_geom_exclude=["_finger_armor"],       # see YAM_GRIPPER
    reverse_control=False,
    discrete=False,
)

# Fixed base — YAM is a tabletop bimanual rig; no floating DoFs by default.
# Shift 20 cm toward the task workspace/shelf from Panda's preset base position,
# and mount high enough for the solved YAM home pose to match Panda's wrist pose.
YAM_FIXED_BASE = FloatingBaseConfig(
    dofs={},
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    offset_position=np.array([0.2, 0.0, 0.96]),
)

# Base/stand geom welded under `bimanual_base` (which has no geometry of its own).
# Keep a full-width bimanual stand under both YAM arms. The stand is expressed in
# the pelvis frame, so the center offset is half the mount height: bottom at floor,
# top at the arm mount.
YAM_STAND_HEIGHT = 0.96
YAM_BASE_MOUNT_OFFSET = YAM_STAND_HEIGHT / 2.0
YAM_BASE_GEOM = {
    "size": [0.2, 0.6, YAM_STAND_HEIGHT / 2.0],
    "pos": [-0.15, 0.0, -YAM_BASE_MOUNT_OFFSET],
    "color": [0.50, 0.51, 0.55, 1.0],
}

# --------------------------------------------------------------------------- #
# Joint limits (radians, read from the model).
# --------------------------------------------------------------------------- #
_YAM_LIMITS = {
    "joint1": (-2.6180, 3.0543),
    "joint2": (0.0000, 3.6652),
    "joint3": (0.0000, 3.6652),
    "joint4": (-1.5708, 1.5708),
    "joint5": (-1.5708, 1.5708),
    "joint6": (-2.0944, 2.0944),
}

# Keyframes ported verbatim from molmoact2 `bimanual_yam.py`.
# Per-arm "rest": [j1..j6] = [0, pi/4, pi/2, 0, 0, 0]; fingers slightly open.
YAM_ARM_REST = [0.0, np.pi / 4, np.pi / 2, 0.0, 0.0, 0.0]   # one arm, 6 DoF
YAM_ARM_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]               # one arm, 6 DoF
YAM_PANDA_LIKE_HOME = {
    HandSide.LEFT: [0.16790, 1.91036, 1.75242, -1.30, 0.0, 0.0],
    HandSide.RIGHT: [-0.16761, 1.91107, 1.75375, -1.30, 0.00001, 0.0],
}
YAM_GRIPPER_OPEN = -0.0475                                  # molmoact2 open value
YAM_GRIPPER_CLOSED = 0.0
YAM_FINGER_REST = -0.02                                     # molmoact2 rest fingers


def _yam_joint_limits() -> dict:
    out = {}
    for side in ("left", "right"):
        for j, lim in _YAM_LIMITS.items():
            out[f"{side}_{j}"] = lim
    return out


BIMANUAL_YAM_CONFIG = RobotConfig(
    delta_range=(-0.1, 0.1),
    position_kp=4500,
    teleop_pos_step=0.005,
    teleop_rot_step=0.005,
    teleop_max_position_lead=0.05,
    pelvis_body="bimanual_base",   # single-model: real base body (not an invisible_base)
    floating_base=YAM_FIXED_BASE,
    gripper={HandSide.LEFT: YAM_GRIPPER, HandSide.RIGHT: YAM_RIGHT_GRIPPER},
    arms={HandSide.LEFT: YAM_LEFT_ARM, HandSide.RIGHT: YAM_RIGHT_ARM},
    arm_offset={HandSide.LEFT: np.zeros(3), HandSide.RIGHT: np.zeros(3)},
    arm_offset_euler={HandSide.LEFT: np.zeros(3), HandSide.RIGHT: np.zeros(3)},
    actuators=YAM_ACTUATORS,
    cameras=list(YAM_CAMERA_CONFIG.keys()),
    namespaces_to_remove=[],
    model=YAM_MODEL,                          # single-model path
    model_name="Bimanual YAM",
    base_geom=YAM_BASE_GEOM,                  # visible stand welded to bimanual_base
    arm_velocity_limits=YAM_SIM_ARM_VELOCITY_LIMITS,
)


# When the single flattened model loads into a scene, dm_control namespaces every
# element under the model name from `<mujoco model="bimanual_yam_linear">`.
#   * the worldbody prefix-filter in base_ik sees the *compiled* name → forward slash
#   * after base_ik re-parses with escape_separators=True, lookups use *backslash*
#     (same convention as the Panda IK config, e.g. "panda nohand_left\\link0")
YAM_NS = "bimanual_yam_linear/"        # robot_prefix (worldbody filter)
_NS = "bimanual_yam_linear\\"          # safe_find lookups in the IK model


def create_bimanual_yam_config() -> RobotIKConfig:
    """IK config for the bimanual YAM (roots/sites/limits from the flattened model)."""
    return RobotIKConfig(
        robot_prefix=YAM_NS,
        root_body_name=f"{_NS}bimanual_base",
        torso_name=f"{_NS}bimanual_base",
        arm_roots=[f"{_NS}left_arm", f"{_NS}right_arm"],
        arm_sites=[f"{_NS}left_tcp_site", f"{_NS}right_tcp_site"],
        joint_limits={f"{_NS}{k}": v for k, v in _yam_joint_limits().items()},
        end_effector_exclude_word="finger",
    )


def create_bimanual_yam_ik_config() -> RobotIKConfig:
    """Compatibility alias used by the existing robot registry."""
    return create_bimanual_yam_config()


class BimanualYAM(Robot):
    """Bimanual YAM robot (two 6-DoF arms + parallel grippers), single MJCF."""

    preset_model_names = ("Bimanual Panda",)

    def get_keyframes(self, name_identifier="home"):
        """Start from a solved pose matching Panda's initial wrist pose."""
        self.keyframes = {
            side: list(joints) for side, joints in YAM_PANDA_LIKE_HOME.items()
        }

    def _on_loaded(self, model):
        self._dampen_sim_arm_actuators(model)
        self._strengthen_sim_grippers(model)
        super()._on_loaded(model)
        self.get_keyframes()

    def _dampen_sim_arm_actuators(self, model):
        for side in ("left", "right"):
            for joint_idx, (gain, damping) in enumerate(
                zip(YAM_SIM_ARM_GAINS, YAM_SIM_ARM_DAMPING, strict=True),
                start=1,
            ):
                actuator = model.find("actuator", f"{side}_joint{joint_idx}")
                if actuator is None:
                    continue
                actuator.gainprm = (gain,)
                actuator.biasprm = (0.0, -gain, -damping)

    def _strengthen_sim_grippers(self, model):
        for actuator_name in (
            "left_gripper_left_tip",
            "left_gripper_right_tip",
            "right_gripper_left_tip",
            "right_gripper_right_tip",
        ):
            actuator = model.find("actuator", actuator_name)
            if actuator is None:
                continue
            actuator.forcerange = (
                -YAM_SIM_GRIPPER_FORCE_LIMIT_N,
                YAM_SIM_GRIPPER_FORCE_LIMIT_N,
            )
            actuator.gainprm = (YAM_SIM_GRIPPER_KP,)
            actuator.biasprm = (0.0, -YAM_SIM_GRIPPER_KP, -YAM_SIM_GRIPPER_KD)

        for geom in model.find_all("geom"):
            geom_name = geom.name or ""
            if "_tip_collider_" not in geom_name:
                continue
            geom.friction = YAM_SIM_PAD_FRICTION
            geom.condim = YAM_SIM_PAD_CONDIM
            geom.priority = 1
            geom.solimp = YAM_SIM_PAD_SOLIMP
            geom.solref = YAM_SIM_PAD_SOLREF

    def reset_pose(self):
        super().reset_pose()
        for gripper in self.grippers.values():
            gripper.set_control(0.0)
            for actuator in gripper.actuators:
                ctrlrange = np.asarray(actuator.ctrlrange, dtype=np.float64)
                self._mojo.physics.bind(actuator).ctrl = ctrlrange[0]
                if actuator.joint:
                    self._mojo.physics.bind(actuator.joint).qpos = ctrlrange[0]
        self._mojo.physics.forward()

    @property
    def ik_config(self) -> RobotIKConfig:
        return create_bimanual_yam_ik_config()

    @property
    def config(self) -> RobotConfig:
        return BIMANUAL_YAM_CONFIG
