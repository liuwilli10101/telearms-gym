"""H1 Robot."""
import numpy as np
from mojo.elements.consts import JointType

from roboeval.action_modes import PelvisDof
from roboeval.const import ASSETS_PATH, HandSide
from roboeval.robots.config import ArmConfig, FloatingBaseConfig, RobotConfig, RobotIKConfig
from roboeval.robots.configs.robotiq import ROBOTIQ_2F85, ROBOTIQ_2F85_FINE_MANIPULATION
from roboeval.robots.robot import Robot
from roboeval.utils.dof import Dof

# from ik.base_ik import RobotIKConfig

H1_WRIST_DOF = Dof(
    joint_type=JointType.HINGE,
    axis=(1, 0, 0),
    joint_range=(-1.5708, 1.5708),
    action_range=(-18, 18),
)
H1_LEFT_ARM = ArmConfig(
    site="left_end_effector",
    links=[
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
    ],
    wrist_dof=H1_WRIST_DOF,
    offset_euler=np.array([np.pi / 2, np.pi / 2, 0]),
)
H1_RIGHT_ARM = ArmConfig(
    site="right_end_effector",
    links=[
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ],
    wrist_dof=H1_WRIST_DOF,
    offset_euler=np.array([np.pi / 2, np.pi / 2, 0]),
)
H1_ACTUATORS = {
    "left_hip_yaw": False,
    "left_hip_roll": False,
    "left_hip_pitch": False,
    "left_knee": False,
    "left_ankle": False,
    "right_hip_yaw": False,
    "right_hip_roll": False,
    "right_hip_pitch": False,
    "right_knee": False,
    "right_ankle": False,
    "torso": False,
    "left_shoulder_pitch": True,
    "left_shoulder_roll": True,
    "left_shoulder_yaw": True,
    "left_elbow": True,
    "left_wrist": True,
    "right_shoulder_pitch": True,
    "right_shoulder_roll": True,
    "right_shoulder_yaw": True,
    "right_elbow": True,
    "right_wrist": True,
}
STIFFNESS_XY = 1e4
STIFFNESS_Z = 1e6
RANGE_DOF_Z = (0.4, 1.0)
H1_FLOATING_BASE = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(
            joint_type=JointType.SLIDE,
            axis=(1, 0, 0),
            stiffness=STIFFNESS_XY,
        ),
        PelvisDof.Y: Dof(
            joint_type=JointType.SLIDE,
            axis=(0, 1, 0),
            stiffness=STIFFNESS_XY,
        ),
        PelvisDof.Z: Dof(
            joint_type=JointType.SLIDE,
            axis=(0, 0, 1),
            joint_range=RANGE_DOF_Z,
            action_range=RANGE_DOF_Z,
            stiffness=STIFFNESS_Z,
        ),
        PelvisDof.RZ: Dof(
            joint_type=JointType.HINGE,
            axis=(0, 0, 1),
            stiffness=STIFFNESS_XY,
        ),
    },
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    offset_position=np.array([0, 0, 1]),
)
H1_CONFIG = RobotConfig(
    model=ASSETS_PATH / "h1/h1.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="pelvis",
    floating_base=H1_FLOATING_BASE,
    gripper=ROBOTIQ_2F85,
    arms={HandSide.LEFT: H1_LEFT_ARM, HandSide.RIGHT: H1_RIGHT_ARM},
    actuators=H1_ACTUATORS,
    cameras=["head", "left_wrist", "right_wrist"],
    namespaces_to_remove=["light"],
    model_name="h1",
)
H1_FINE_MANIPULATION_CONFIG = RobotConfig(
    model=ASSETS_PATH / "h1/h1.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="pelvis",
    floating_base=H1_FLOATING_BASE,
    gripper=ROBOTIQ_2F85_FINE_MANIPULATION,
    arms={HandSide.LEFT: H1_LEFT_ARM, HandSide.RIGHT: H1_RIGHT_ARM},
    actuators=H1_ACTUATORS,
    cameras=["head", "left_wrist", "right_wrist"],
    namespaces_to_remove=["light"],
    model_name="h1",
)


class H1(Robot):
    """H1 Robot."""

    @property
    def ik_config(self) -> RobotIKConfig:
        """Get robot IK config."""
        return create_h1_config()

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return H1_CONFIG


class H1FineManipulation(Robot):
    """H1 Robot with Robotiq gripper for fine manipulations."""

    @property
    def ik_config(self) -> RobotIKConfig:
        """Get robot IK config."""
        return create_h1_config()

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return H1_FINE_MANIPULATION_CONFIG
    

def create_h1_config() -> RobotIKConfig:
    """Create config for H1 robot."""
    return RobotIKConfig(
        robot_prefix="h1",
        root_body_name="h1\\pelvis",
        torso_name="h1\\torso_link",
        arm_roots=[
            f"h1\\{H1_CONFIG.arms[HandSide.LEFT].links[0]}",
            f"h1\\{H1_CONFIG.arms[HandSide.RIGHT].links[0]}",
        ],
        arm_sites=[
            f"h1\\{H1_CONFIG.arms[HandSide.LEFT].site}",
            f"h1\\{H1_CONFIG.arms[HandSide.RIGHT].site}",
        ],
        joint_limits={
            "h1\\left_elbow": (-1.25, np.pi / 2),
            "h1\\right_elbow": (-1.25, np.pi / 2),
        },
        end_effector_exclude_word="robotiq"
    )
