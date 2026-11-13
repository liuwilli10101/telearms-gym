from __future__ import annotations
from abc import ABC
import numpy as np
from typing import Dict
import re
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
import string
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import GoalPatch, ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubeBridgeStackingTask(RoboEvalEnv, MetricRolloutEval):
    """Build a bridge with two cubes as the base and one cube spanning across them on top.

Success Criteria:
Any two cubes may form the base and any cube may be the span. The two base cubes rest level on the table with a gap of at most 3 cm between them; the third sits on both, clear of the table, overlapping each by at least 15%, centred on their midpoint within 3 cm and upright within 10 degrees; all released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Geometry constants (ColoredCube is 4cm) ---
    _CUBE_SIZE = 0.04
    _HALF = _CUBE_SIZE / 2.0
    _TABLE_Z = 0.97
    _STACK_Z = _TABLE_Z + _CUBE_SIZE  # 1.01

    # --- Goal tolerances from spec ---
    _MAX_BASE_GAP = 0.03
    _SAME_HEIGHT_TOL = 0.01
    _MAX_CENTER_OFFSET = 0.03
    _MIN_CONTACT_RATIO = 0.15
    _MAX_TILT_DEG = 10.0

    _NUM_SUBGOALS = 8

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY allowed asset: ColoredCube)
        self.cube_base_left = ColoredCube(self._mojo)
        self.cube_base_right = ColoredCube(self._mojo)
        self.cube_bridge = ColoredCube(self._mojo)

        # Track list for convenience/metrics
        self.cubes = [self.cube_base_left, self.cube_base_right, self.cube_bridge]

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_base_left.set_color("red")
        self.cube_base_right.set_color("blue")
        self.cube_bridge.set_color("green")

        # Nominal goal positions (side-by-side along Y; bridge centered above)
        # Choose base center spacing s=0.06 -> edge gap = 0.02 (<= 0.03) and stable overlap.
        self._base_x = 0.55
        self._base_y_left = +0.03
        self._base_y_right = -0.03
        self._goal_pos_left = np.array([self._base_x, self._base_y_left, self._TABLE_Z])
        self._goal_pos_right = np.array([self._base_x, self._base_y_right, self._TABLE_Z])
        self._goal_pos_bridge = np.array([self._base_x, 0.0, self._STACK_Z])
        self._goal_quat_upright = np.array([1.0, 0.0, 0.0, 0.0])

        # ⚠️ CRITICAL: Initialize metrics with FULL signature
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _on_reset(self):
        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_base_left.set_color("red")
        self.cube_base_right.set_color("blue")
        self.cube_bridge.set_color("green")

        # 1. Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2. Deterministic RNG (kept for pipeline consistency; do not randomize goal here)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False after reset)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._TABLE_Z)

        # The bases must start apart -- "side-by-side" has to be false at reset -- but
        # sampling each one independently across the whole workspace could put them
        # 0.50 m apart in Y and 0.30 m in X, and the reported difficulty of this task
        # was almost entirely the long haul of dragging them together. Sample the pair
        # around a shared centre with a bounded gap instead: still clearly separated,
        # never across the table.
        _BASE_GAP_Y = (0.16, 0.26)   # start apart, but within one comfortable reach
        _BASE_OFFSET_X = 0.07        # max X stagger between the two bases

        gap_y = float(rng.uniform(*_BASE_GAP_Y))
        centre_x = float(rng.uniform(x_min + _BASE_OFFSET_X, x_max - _BASE_OFFSET_X))
        centre_y = float(rng.uniform(y_min + gap_y / 2.0, y_max - gap_y / 2.0))

        pL = np.array(
            [np.clip(centre_x + rng.uniform(-_BASE_OFFSET_X, _BASE_OFFSET_X), x_min, x_max),
             centre_y + gap_y / 2.0, z],
            dtype=float,
        )
        pR = np.array(
            [np.clip(centre_x + rng.uniform(-_BASE_OFFSET_X, _BASE_OFFSET_X), x_min, x_max),
             centre_y - gap_y / 2.0, z],
            dtype=float,
        )

        # Place bridge on the table, away from both bases (no contact, not centered, not stacked).
        # Rejection sample until it's sufficiently far from both base cubes in XY.
        pB = None
        for _ in range(100):
            cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float)
            dL = float(np.linalg.norm(cand[:2] - pL[:2]))
            dR = float(np.linalg.norm(cand[:2] - pR[:2]))
            if dL > 0.12 and dR > 0.12:
                pB = cand
                break
        if pB is None:
            # Fallback: corner placement (still within bounds and far from most samples)
            pB = np.array([x_min, y_min, z], dtype=float)

        # Random yaw about Z (still upright) for variety
        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements

        self.cube_base_left.set_pose(position=pL, quat=yaw_quat())
        self.cube_base_right.set_pose(position=pR, quat=yaw_quat())
        self._mojo.physics.forward()

        self.cube_bridge.set_pose(position=pB, quat=yaw_quat())
        self._mojo.physics.forward()

        # 4. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # --- Helper geometry checks ---
    def _tilt_deg(self, quat_xyzw: np.ndarray) -> float:
        """Return tilt from world-up in degrees, based on cube's local +Z axis."""
        try:
            # NOTE: body.get_quaternion() returns wxyz; read it as wxyz (it is NOT xyzw).
            q = Quaternion(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])  # w,x,y,z
            local_up_world = q.rotate([0.0, 0.0, 1.0])
            local_up_world = np.array(local_up_world, dtype=float)
            local_up_world /= (np.linalg.norm(local_up_world) + 1e-9)
            dot = float(np.clip(np.dot(local_up_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
            return float(np.degrees(np.arccos(dot)))
        except Exception:
            return 180.0

    def _contact_area_ratio_xy(self, a_pos: np.ndarray, b_pos: np.ndarray) -> float:
        """
        Approximate contact area ratio between two upright cubes via XY overlap of their top/bottom faces.
        Ratio is overlap_area / cube_face_area.
        """
        dx = abs(float(a_pos[0] - b_pos[0]))
        dy = abs(float(a_pos[1] - b_pos[1]))
        overlap_x = max(0.0, self._CUBE_SIZE - dx)
        overlap_y = max(0.0, self._CUBE_SIZE - dy)
        overlap_area = overlap_x * overlap_y
        face_area = self._CUBE_SIZE * self._CUBE_SIZE
        return float(overlap_area / (face_area + 1e-9))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: the instruction never says which cube goes on top, so the
        # highest one IS the bridge and the other two are the bases. Binding these to
        # self.cube_bridge (green) failed a correctly built bridge that happened to
        # use a different colour for the span.
        by_height = sorted(self.cubes, key=lambda c: float(c.body.get_position()[2]))
        left, right, bridge = by_height[0], by_height[1], by_height[2]

        pL = left.body.get_position()
        pR = right.body.get_position()
        pB = bridge.body.get_position()
        qB = bridge.body.get_quaternion()

        # --- Goal condition 1: base cubes side-by-side with small gap, same height, on table ---
        # Axis-agnostic: "side by side" was read as "separated along Y with X within
        # 2 cm", but the instruction names no axis and a pair laid out along X -- or
        # diagonally -- is just as much a base for a bridge. What matters is that the
        # two are level, on the table, and close enough for a cube to span them; the
        # contact-ratio check below already enforces that the span actually lands.
        centre_dist = float(np.linalg.norm(pL[:2] - pR[:2]))
        z_dist = abs(float(pL[2] - pR[2]))

        # Edge gap between the two bases (centre distance minus cube size)
        edge_gap = centre_dist - self._CUBE_SIZE

        bases_on_table = left.is_colliding(self.table.table_top) and right.is_colliding(self.table.table_top)
        base_side_by_side = (
            (z_dist <= self._SAME_HEIGHT_TOL)
            and (edge_gap >= -0.002)  # allow tiny penetration/tolerance
            and (edge_gap <= self._MAX_BASE_GAP)
        )
        base_ok = bool(bases_on_table and base_side_by_side)

        # --- Goal conditions 2 & 3: bridge cube rests on top of both base cubes (meaningful contact) ---
        # Use both collision and an overlap-based proxy for minimum contact area ratio.
        bridge_collides_left = bridge.is_colliding(left)
        bridge_collides_right = bridge.is_colliding(right)

        ratio_left = self._contact_area_ratio_xy(pB, pL)
        ratio_right = self._contact_area_ratio_xy(pB, pR)
        contact_ok = bool(
            bridge_collides_left
            and bridge_collides_right
            and (ratio_left >= self._MIN_CONTACT_RATIO)
            and (ratio_right >= self._MIN_CONTACT_RATIO)
        )

        # Bridge should not be on table if it is on top of bases
        bridge_not_on_table = not bridge.is_colliding(self.table.table_top)

        # --- Goal condition 4: centered over pair (in table plane) ---
        midpoint_xy = 0.5 * (pL[:2] + pR[:2])
        center_offset = float(np.linalg.norm(pB[:2] - midpoint_xy))
        centered_ok = center_offset <= self._MAX_CENTER_OFFSET

        # --- Goal condition 5: bridge approximately upright ---
        tilt_deg = self._tilt_deg(np.array(qB, dtype=float))
        upright_ok = tilt_deg <= self._MAX_TILT_DEG

        positions_ok = bool(base_ok and contact_ok and bridge_not_on_table and centered_ok and upright_ok)

        # No semantic constraints (ColoredCube has no labels); keep for template compatibility
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics (coarse mapping: if final arrangement satisfied, all subgoals satisfied)
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "bridge_center_offset_xy_m": center_offset,
                "base_edge_gap_m": float(edge_gap),
                "bridge_to_left_contact_ratio": float(ratio_left),
                "bridge_to_right_contact_ratio": float(ratio_right),
            },
            pose_error={
                "base_centre_distance_m": float(centre_dist),
                "base_z_mismatch_m": float(z_dist),
                "bridge_tilt_deg": float(tilt_deg),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})