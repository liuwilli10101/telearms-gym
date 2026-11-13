"""RBY1 (Rainbow Robotics) bimanual robot, ported from upstream RoboEval-Rainbow.

Single combined-model robot: instead of stitching component models together at
runtime (rby1_body.xml + rby1_arm.xml x2 + rby1_hand.xml), it loads ONE
self-contained ``rby1.xml`` (thirdparty/mujoco_menagerie/rby1/). That file
already bakes in the full composition: torso/head base, both 7-DoF arms mounted
at the shoulders (right arm rotated 180 deg about Z with its Y-axis joints
mirrored), and a 2-finger hand on each wrist.

When mojo attaches this file to the world it namespaces every element with the
model name ("rby1"), and the authored "/" separators in names become the
escaped "\\" separator. So the runtime element names are:

  body   : rby1/invisible_base, rby1/rby1_arm_left\\link_arm_0, ...
  joint  : rby1/rby1_arm_left\\arm_0, ...
  actuator (local .name): rby1_arm_left\\arm_1_act, ...
  site   (local .name)  : rby1_arm_left\\wrist_site, ...

Helpers below build these names so the config stays readable.
"""
import numpy as np
from dm_control import mjcf
from mojo.elements.consts import JointType

from roboeval.const import THIRD_PARTY_PATH, HandSide
from roboeval.robots.config import ArmConfig, FloatingBaseConfig, RobotConfig, GripperConfig, RobotIKConfig
from roboeval.robots.robot import Robot
from roboeval.robots.arm import Arm
from roboeval.robots.gripper import Gripper
from roboeval.utils.dof import Dof
from roboeval.action_modes import PelvisDof

# Combined model file (single source of truth for the Rainbow robot).
RBY1_COMBINED_MODEL = THIRD_PARTY_PATH / "mujoco_menagerie/rby1/rby1.xml"

# Model namespace mojo assigns from <mujoco model="rby1">.
_MODEL_NS = "rby1"
_ARM_PREFIX = {HandSide.LEFT: "rby1_arm_left", HandSide.RIGHT: "rby1_arm_right"}
_HAND_PREFIX = {HandSide.LEFT: "rby1_hand_left", HandSide.RIGHT: "rby1_hand_right"}


def _local(prefix: str, name: str) -> str:
    """Local (within-model) element name, e.g. 'rby1_arm_left\\arm_0'.

    Used wherever code matches against the mjcf element's local ``.name``
    (config.actuators keys, ArmConfig.links/joints/actuators, ArmConfig.site).
    """
    return f"{prefix}\\{name}"


def _full(local_name: str) -> str:
    """Fully-qualified name with the model namespace, e.g. 'rby1/rby1_arm_left\\link_arm_0'.

    Used wherever code resolves via Body.get / Site.get against the world root.
    """
    return f"{_MODEL_NS}/{local_name}"


def _make_arm_config(side: HandSide) -> ArmConfig:
    """Arm config that points at the arm already present in the combined model."""
    p = _ARM_PREFIX[side]
    return ArmConfig(
        site=_local(p, "wrist_site"),          # resolved relative to the arm body
        body=_full(_local(p, "link_arm_0")),   # arm root, resolved on the world root
        links=[_local(p, f"link_arm_{i}") for i in range(7)],
        joints=[_local(p, f"arm_{i}") for i in range(7)],
        actuators=[_local(p, f"arm_{i}_act") for i in range(1, 8)],
        wrist_dof=None,
        model=None,
    )


def _make_gripper_config(side: HandSide) -> GripperConfig:
    """Gripper config that points at the hand already present in the combined model."""
    h = _HAND_PREFIX[side]
    return GripperConfig(
        model=None,
        body=_full(_local(h, "rby1_hand")),
        actuators=[_local(h, "finger_act")],
        range=np.array([0, 1]),
        pad_bodies=[_local(h, "ee_finger_1"), _local(h, "ee_finger_2")],
        reverse_control=False,
    )


# Per-side arm / gripper configs.
RBY1_ARMS = {HandSide.LEFT: _make_arm_config(HandSide.LEFT),
             HandSide.RIGHT: _make_arm_config(HandSide.RIGHT)}
RBY1_GRIPPERS = {HandSide.LEFT: _make_gripper_config(HandSide.LEFT),
                 HandSide.RIGHT: _make_gripper_config(HandSide.RIGHT)}

# Limb actuators (both arms) keyed by local .name; gripper actuators are handled
# by the Gripper objects and deliberately excluded here.
RBY1_ACTUATORS = {
    _local(_ARM_PREFIX[side], f"arm_{i}_act"): True
    for side in (HandSide.LEFT, HandSide.RIGHT)
    for i in range(1, 8)
}

# Cameras (added in this fork; upstream ships none). Torso and head are rigid
# in the combined model (only wheels/arms/fingers have joints), so the head
# camera is fixed relative to the base. Quats are lookat-computed in the
# parent body frame (MuJoCo camera looks along -z, +y up; up-hint +x world):
# head at world (0.18, 0, 1.58) looking at the workspace center (0.55, 0,
# 0.98); wrist cams sit 9 cm to the local +y side of link_arm_6 looking down
# the tool axis (local -z) past the fingertips.
RBY1_CAMERA_CONFIG = {
    "head": {
        "manual": True,
        # Panda-equivalent oblique, from the FRONT: world (1.15, 0, 1.75)
        # looking at (0.45, 0, 0.95) at ~41 deg, fov 45. The literal Panda
        # pose (-0.4, 0, 1.8) is unusable on this robot: RBY1's own torso and
        # head fill the frame center and hide the workspace from EVERY
        # behind/above/over-shoulder position (render-verified). NOTE the
        # front view is laterally mirrored (robot's left arm on image right).
        # Parented to rby1/invisible_base (pelvis frame at the +0.18 stance).
        "parent": "rby1/invisible_base",
        "position": (0.97, 0.0, 1.75),
        "quaternion": (0.6619, 0.2487, 0.2487, 0.6619),
        "fov": 45,
    },
    # Wrist cams mount on the HAND body (not link_arm_6: from there the view
    # is blocked by the wrist/hand structure). Offsets/quats are expressed in
    # the hand frame at the HOME pose (get_keyframes below): camera 9 cm
    # outboard (world y) and 3 cm up, looking at a point 10 cm ahead and
    # 22 cm below the hand (view tilted toward the workspace, image-up =
    # world +x at home).
    "left_wrist": {
        "manual": True,
        "parent": "rby1/rby1_hand_left\\rby1_hand",
        "position": (-0.0152, -0.0887, 0.0299),
        "quaternion": (0.7213, 0.2445, -0.0156, 0.6479),
        "fov": 60,
    },
    "right_wrist": {
        "manual": True,
        "parent": "rby1/rby1_hand_right\\rby1_hand",
        "position": (0.0796, 0.0417, 0.0305),
        "quaternion": (0.9486, 0.1323, 0.204, 0.2025),
        "fov": 60,
    },
}

# Floating base configuration.
#
# The DOFs below stay dormant unless the run enables them
# (``enable_floating_base=true`` -> [X, Y, RZ]); with no DOFs requested the
# base is fixed exactly as before. When enabled, the demo recorder can drive
# the pelvis (forward/turn) and the two wheels spin to match.
_BASE_STIFFNESS = 1e4
RBY1_FLOATING_BASE = FloatingBaseConfig(
    dofs={
        PelvisDof.X: Dof(joint_type=JointType.SLIDE, axis=(1, 0, 0), stiffness=_BASE_STIFFNESS),
        PelvisDof.Y: Dof(joint_type=JointType.SLIDE, axis=(0, 1, 0), stiffness=_BASE_STIFFNESS),
        PelvisDof.RZ: Dof(joint_type=JointType.HINGE, axis=(0, 0, 1), stiffness=_BASE_STIFFNESS),
    },
    delta_range_position=(-0.01, 0.01),
    delta_range_rotation=(-0.05, 0.05),
    # Stand as close to the table as the geometry allows. The tabletop slab
    # spans x [0.30, 0.90] (z 0.89-0.95) and the torso's front face sits at
    # stance + 0.111 in that height band, so any stance beyond 0.18 visually
    # embeds the torso in the table's back edge (the torso has no colliders
    # vs the table, so nothing stops it physically -- it just looks stuck).
    # Reach at 0.18~0.20 (IK grid, top-down targets, z 1.05): each arm 19/25;
    # combining arms only the far x=0.75 row is out of reach. +0.30 would
    # cover 24/25 per arm but clips the table by 11 cm. (Reset re-places the
    # pelvis through this offset, so it must live HERE, not in _setup_pelvis.)
    offset_position=np.array([0.18, 0.0, 0.0]),
)

# Bimanual Rainbow Robot Configuration.
# ``model`` stays None so the base Robot uses the compositional control flow;
# the single combined file is loaded inside ``_setup_pelvis`` below.
RBY1_CONFIG = RobotConfig(
    model=None,
    delta_range=(-0.1, 0.1),
    position_kp=1000,
    pelvis_body=_full("invisible_base"),
    floating_base=RBY1_FLOATING_BASE,
    gripper=RBY1_GRIPPERS[HandSide.LEFT],  # default; per-side built in _get_grippers
    arms=RBY1_ARMS,
    arm_offset={
        HandSide.LEFT: np.array([0, 0, 0]),
        HandSide.RIGHT: np.array([0, 0, 0]),
    },
    arm_offset_euler={
        HandSide.LEFT: np.array([0, 0, 0]),
        HandSide.RIGHT: np.array([0, 0, 0]),
    },
    actuators=RBY1_ACTUATORS,
    cameras=RBY1_CAMERA_CONFIG,
    namespaces_to_remove=[],
    model_name="Rainbow",
)


def _ik_arm(prefix: str, name: str) -> str:
    """Name of an arm element in the IK solver's re-flattened model.

    The IK rebuilds the model with ``to_xml`` -> ``from_xml_string(escape_separators=True)``,
    which escapes the model separator "/" *again*. So a runtime name like
    ``rby1/rby1_arm_left\\arm_0`` becomes ``rby1\\rby1_arm_left\\\\arm_0``
    (outer single backslash, inner double).
    """
    return f"{_MODEL_NS}\\{prefix}\\\\{name}"


def _ik_root(name: str) -> str:
    """Name of a model-root-level body in the IK solver's re-flattened model."""
    return f"{_MODEL_NS}\\{name}"


def create_rby1_ik_config() -> RobotIKConfig:
    """Create IK config for the bimanual RBY1 (Rainbow) robot.

    Joint limits are taken verbatim from the combined model (right arm
    arm_3/arm_5 are already mirrored in the file).

    Note: upstream RoboEval-Rainbow passes ``exclude_last_n_joints=0`` here;
    this fork's RobotIKConfig has no such field and its IK always uses every
    reduced-model joint, which is the same behaviour (all 7 arm joints).
    """
    left_ranges = {
        "arm_0": (-2.35619, 2.35619),
        "arm_1": (-0.05, 3.14159),
        "arm_2": (-2.0944, 2.0944),
        "arm_3": (-2.61799, 0.01),
        "arm_4": (-6.28319, 6.28319),
        "arm_5": (-1.74533, 2.00713),
        "arm_6": (-2.96706, 2.96706),
    }
    # Right arm: arm_3/arm_5 mirrored to match the 180-deg Z mounting.
    right_ranges = {
        "arm_0": (-2.35619, 2.35619),
        "arm_1": (-0.05, 3.14159),
        "arm_2": (-2.0944, 2.0944),
        "arm_3": (-0.01, 2.61799),
        "arm_4": (-6.28319, 6.28319),
        "arm_5": (-2.00713, 1.74533),
        "arm_6": (-2.96706, 2.96706),
    }
    joint_limits = {}
    for j, r in left_ranges.items():
        joint_limits[_ik_arm(_ARM_PREFIX[HandSide.LEFT], j)] = r
    for j, r in right_ranges.items():
        joint_limits[_ik_arm(_ARM_PREFIX[HandSide.RIGHT], j)] = r

    # The robot root in the IK model is the attachment-frame wrapper body
    # ("rby1\\"), which carries the free joint used to pose the robot.
    root_body = f"{_MODEL_NS}\\"

    return RobotIKConfig(
        robot_prefix="",
        root_body_name=root_body,
        torso_name=root_body,
        arm_roots=[
            _ik_arm(_ARM_PREFIX[HandSide.LEFT], "link_arm_0"),
            _ik_arm(_ARM_PREFIX[HandSide.RIGHT], "link_arm_0"),
        ],
        arm_sites=[
            _ik_arm(_ARM_PREFIX[HandSide.LEFT], "wrist_site"),
            _ik_arm(_ARM_PREFIX[HandSide.RIGHT], "wrist_site"),
        ],
        joint_limits=joint_limits,
        end_effector_exclude_word="finger",
    )


class Rainbow(Robot):
    """RBY1 (Rainbow) bimanual robot loaded from a single combined model file.

    Fully compatible with BimanualPanda environments. The whole robot
    (base + both arms + both hands) lives in ``rby1.xml``; this class wires the
    base ``Robot`` machinery up to the elements already present in that file
    instead of attaching separate component models at runtime.
    """

    @property
    def ik_config(self) -> RobotIKConfig:
        """Get robot IK config."""
        return create_rby1_ik_config()

    @property
    def config(self) -> RobotConfig:
        """Get robot config."""
        return RBY1_CONFIG

    def get_keyframes(self, name_identifier='home'):
        """Grasp-ready teleop start pose (replaces upstream's wide 'T' pose,
        which parks both hands OUTSIDE the table at y +-0.53).

        FK-searched 2026-07-15 (random search + anneal, elbow-down bias,
        limits tightened 12% off the edges, at the +0.30 m base stance from
        RBY1_FLOATING_BASE.offset_position): hands hover at
        (0.55, +-0.25, 1.12) -- over the workspace, ~17 cm above the table --
        tools pointing straight down, elbows low, zero contacts.
        """
        home_left = [-0.387, 1.176, -0.614, -2.31, 1.168, 1.642, -1.932]
        home_right = [1.057, 1.189, 1.391, 1.864, -1.388, -1.174, 0.126]
        self.keyframes = {
            HandSide.LEFT: home_left,
            HandSide.RIGHT: home_right,
        }

    def _setup_pelvis(self, model: mjcf.RootElement):
        """Load the combined RBY1 model and use its attachment frame as the pelvis.

        ``load_model`` wraps the attached model in a top-level frame body
        (the parent of ``invisible_base``). That top-level body is the pelvis:
        a free joint for posing the robot can only live on a top-level body.
        """
        # Attach the whole combined robot (base + arms + hands) to the world.
        # (The 0.20 m forward stance lives in RBY1_FLOATING_BASE.offset_position;
        # reset re-places the pelvis through that path, so setting the position
        # here would be silently undone.)
        self._pelvis = self._mojo.load_model(str(RBY1_COMBINED_MODEL))
        self._mojo.mark_dirty()

        if self._pelvis.is_kinematic():
            self._pelvis.set_kinematic(False)
        return self._pelvis.mjcf

    def load_arms(self, parent=None):
        """Bind the two arms that already exist inside the combined model.

        Right-arm joint limits / ctrlranges are already mirrored in the file
        (build_rby1_combined.py does it), so nothing is mirrored here.
        """
        self._arms = {}
        for side, arm_config in self.config.arms.items():
            if arm_config is None:
                continue
            # base_site is unused for body-mode arms (they are already placed
            # at the shoulder inside the combined model).
            self._arms[side] = Arm(
                side=side, base_site=None, arm_config=arm_config, mojo=self._mojo
            )

        # Force contact solver options on the root model (the combined file sets
        # these too, but keep parity with the upstream Rainbow behaviour).
        root_mjcf = self._mojo.root_element.mjcf
        root_mjcf.option.impratio = 10
        root_mjcf.option.noslip_iterations = 3
        self._mojo.mark_dirty()

    def set_up_grippers(self):
        """Wrist sites come straight from each arm's end-effector site.

        The hands are already mounted at the wrist in the combined model, so no
        wrist body needs to be created.
        """
        self._wrist_sites = {}
        for side, arm in self._arms.items():
            self._wrist_sites[side] = arm.end_effector_site

    def _get_grippers(self):
        """Bind the per-side hands present in the combined model.

        The link_arm_5 <-> hand contact exclude is already baked into the file,
        so it is not re-added here.
        """
        grippers = {}
        for side, arm in self._arms.items():
            grippers[side] = Gripper(
                side,
                self._wrist_sites[side],
                self.config.arms[side],
                RBY1_GRIPPERS[side],
                self._mojo,
            )
        self._mojo.mark_dirty()
        return grippers
