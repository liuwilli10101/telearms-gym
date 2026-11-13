"""RobotiQ Grippers."""
import numpy as np

from roboeval.const import ASSETS_PATH
from roboeval.robots.config import GripperConfig

ROBOTIQ_2F85 = GripperConfig(
    model=ASSETS_PATH / "robotiq_2f85/2f85.xml",
    actuators=["fingers_actuator"],
    range=np.array([0, 1]),
    pinch_site="pinch",
    pad_bodies=["left_pad", "right_pad"],
)
ROBOTIQ_2F85_FINE_MANIPULATION = GripperConfig(
    model=ASSETS_PATH / "robotiq_2f85/2f85_fine_manipulation.xml",
    actuators=["fingers_actuator"],
    range=np.array([0, 1]),
    pinch_site="pinch",
    pad_bodies=["left_pad", "right_pad"],
)
