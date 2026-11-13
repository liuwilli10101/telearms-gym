"""Hello Robotics Stretch Robot."""
import numpy as np
from mojo.elements.consts import JointType

from roboeval.action_modes import PelvisDof
from roboeval.const import ASSETS_PATH, HandSide
from roboeval.robots.config import (
    ArmConfig,
    FloatingBaseConfig,
    GripperConfig,
    RobotConfig,
)
from roboeval.robots.robot import Robot
from roboeval.utils.dof import Dof

STRETCH_ACTUATORS = {
    "forward": False,
    "turn": False,
    "lift": True,
    "arm_extend": True,
    "wrist_yaw": True,
    "grip": True,
    "head_pan": True,
    "head_tilt": True,
}
STRETCH_HAND = ArmConfig(
    site="wrist",
    links=[
        "link_lift",
        "link_arm_l4",
        "link_arm_l3",
        "link_arm_l2",
        "link_arm_l1",
        "link_arm_l0",
        "link_wrist_yaw",
    ],
)
STIFFNESS = 1e4
STRETCH_FLOATING_BASE = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(
            joint_type=JointType.SLIDE,
            axis=(1, 0, 0),
            stiffness=STIFFNESS,
        ),
        PelvisDof.Y: Dof(
            joint_type=JointType.SLIDE,
            axis=(0, 1, 0),
            stiffness=STIFFNESS,
        ),
        PelvisDof.RZ: Dof(
            joint_type=JointType.HINGE,
            axis=(0, 0, 1),
            stiffness=STIFFNESS,
        ),
    },
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
)
STRETCH_GRIPPER = GripperConfig(
    body="stretch/link_gripper_slider",
    actuators=["grip"],
    pad_bodies=["rubber_tip_left", "rubber_tip_right"],
    range=np.array([0, 1]),
)
STRETCH_ROBOT = RobotConfig(
    model=ASSETS_PATH / "hello_robot_stretch/stretch.xml",
    delta_range=(-0.1, 0.1),
    position_kp=300,
    pelvis_body="base_link",
    floating_base=STRETCH_FLOATING_BASE,
    gripper=STRETCH_GRIPPER,
    arms={HandSide.RIGHT: STRETCH_HAND},
    actuators=STRETCH_ACTUATORS,
    namespaces_to_remove=["light"],
)


class StretchRobot(Robot):
    """Hello Robotics Stretch Robot."""

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return STRETCH_ROBOT
