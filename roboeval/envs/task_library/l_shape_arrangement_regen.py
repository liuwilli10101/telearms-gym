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

class LCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes on the table into an L-shape using the workspace convention
    where X is front/back and Y is left/right. The L-shape is defined by three cubes aligned
    along +Y (leftward) and one additional cube extending along +X (backward) from the
    corner cube.

    Color assignment is FIXED (success checks specific colors at specific positions):
      - The +Y leg, from leftmost to the corner, is red -> blue -> green
        (red is the far-left end, blue the middle, green the corner).
      - yellow extends backward (+X) from the green corner cube.

    Sequential Sub-Goals:
        0. Grasp cube_1 with either gripper.
        1. Move cube_1 toward its intended position in the L-shape.
        2. Place cube_1 down on the table.
        3. Grasp cube_2 with either gripper.
        4. Move cube_2 to form the middle of the vertical leg of the L (Y-axis leg).
        5. Place cube_2 down on the table.
        6. Grasp cube_3 with either gripper.
        7. Move cube_3 to become the corner of the L-shape.
        8. Place cube_3 down on the table.
        9. Grasp cube_4 with either gripper.
        10. Move cube_4 behind (+X) the corner cube to form the short leg of the L.
        11. Place cube_4 down on the table.
        12. Nudge cubes to ensure the three-cube column is along +Y with consistent spacing,
            and the fourth cube extends along +X from the corner.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # ---- Task parameters (from spec) ----
    _ALIGN_TOL_M = 0.03
    _MIN_SEP_M = 0.045
    _REL_TOL_M = 0.02

    _CUBE_Z = 0.97  # ColoredCube resting Z on table

    _NUM_SUBGOALS = 13

    def generate_goal_conditions(self):
        """Structured goal conditions mirroring the task specification."""
        return [
            {
                "type": "arrangement",
                "subject": "cube_1",
                "target": "cube_2",
                "parameters": {"relation": "aligned", "axis": "y", "tolerance_m": 0.03},
                "description": "cube_1 and cube_2 are aligned along the Y axis (same x within tolerance).",
            },
            {
                "type": "arrangement",
                "subject": "cube_2",
                "target": "cube_3",
                "parameters": {"relation": "aligned", "axis": "y", "tolerance_m": 0.03},
                "description": "cube_2 and cube_3 are aligned along the Y axis (same x within tolerance).",
            },
            {
                "type": "relative",
                "subject": "cube_1",
                "target": "cube_2",
                "parameters": {
                    "relation": "left_of",
                    "axis_index": 1,
                    "min_separation_m": 0.045,
                    "tolerance_m": 0.02,
                },
                "description": "cube_1 is left of cube_2 (higher y; +Y is left).",
            },
            {
                "type": "relative",
                "subject": "cube_2",
                "target": "cube_3",
                "parameters": {
                    "relation": "left_of",
                    "axis_index": 1,
                    "min_separation_m": 0.045,
                    "tolerance_m": 0.02,
                },
                "description": "cube_2 is left of cube_3 (higher y; +Y is left).",
            },
            {
                "type": "arrangement",
                "subject": "cube_3",
                "target": "cube_4",
                "parameters": {"relation": "aligned", "axis": "x", "tolerance_m": 0.03},
                "description": "cube_3 and cube_4 are aligned along the X axis (same y within tolerance).",
            },
            {
                "type": "relative",
                "subject": "cube_4",
                "target": "cube_3",
                "parameters": {
                    "relation": "back_of",
                    "axis_index": 0,
                    "min_separation_m": 0.045,
                    "tolerance_m": 0.02,
                },
                "description": "cube_4 is behind cube_3 (higher x; +X is back/away from robot).",
            },
        ]

    def _initialize_env(self):
        # Table exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create four colored cubes (ONLY allowed asset)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)

        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4]

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Define a deterministic goal L-shape in workspace bounds
        # Three cubes along +Y (left): cube_3 is the corner (rightmost / lowest y),
        # then cube_2, then cube_1 at highest y.
        x0 = 0.55
        y_corner = -0.06
        dy = 0.06
        dx = 0.06

        self._goal_positions = {
            "cube_3": np.array([x0, y_corner, self._CUBE_Z]),               # corner (rightmost)
            "cube_2": np.array([x0, y_corner + dy, self._CUBE_Z]),          # middle
            "cube_1": np.array([x0, y_corner + 2 * dy, self._CUBE_Z]),      # leftmost
            "cube_4": np.array([x0 + dx, y_corner, self._CUBE_Z]),          # behind corner (+X)
        }

        # Metrics (FULL signature required)
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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. RANDOM UNSOLVED placements within workspace bounds (must NOT satisfy _success())
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._CUBE_Z

        min_pairwise_sep = 0.06  # conservative to avoid initial overlaps

        def random_quat_z():
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            return np.array([0.0, 0.0, np.sin(theta / 2.0), np.cos(theta / 2.0)])

        def sample_positions():
            poses = {}
            placed = []
            for name in ["cube_1", "cube_2", "cube_3", "cube_4"]:
                for _ in range(2000):
                    p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)
                    if all(np.linalg.norm(p[:2] - q[:2]) >= min_pairwise_sep for q in placed):
                        poses[name] = p
                        placed.append(p)
                        break
                else:
                    # Fallback (should be very rare): relax constraint slightly
                    p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)
                    poses[name] = p
                    placed.append(p)
            return poses

        def positions_would_be_success(poses: Dict[str, np.ndarray]) -> bool:
            # Recreate the exact spatial checks used by _success(), without touching metrics.
            p1, p2, p3, p4 = poses["cube_1"], poses["cube_2"], poses["cube_3"], poses["cube_4"]

            aligned_12 = abs(p1[0] - p2[0]) <= self._ALIGN_TOL_M
            aligned_23 = abs(p2[0] - p3[0]) <= self._ALIGN_TOL_M
            left_12 = (p1[1] - p2[1]) >= (self._MIN_SEP_M - self._REL_TOL_M)
            left_23 = (p2[1] - p3[1]) >= (self._MIN_SEP_M - self._REL_TOL_M)
            aligned_34 = abs(p3[1] - p4[1]) <= self._ALIGN_TOL_M
            back_43 = (p4[0] - p3[0]) >= (self._MIN_SEP_M - self._REL_TOL_M)

            positions_ok = bool(aligned_12 and aligned_23 and left_12 and left_23 and aligned_34 and back_43)
            return positions_ok

        def far_from_goal_enough(poses: Dict[str, np.ndarray]) -> bool:
            # Ensure at least one cube is not near its goal (avoids accidental stage completion).
            thresh = 0.05
            d1 = float(np.linalg.norm(poses["cube_1"][:2] - self._goal_positions["cube_1"][:2]))
            d2 = float(np.linalg.norm(poses["cube_2"][:2] - self._goal_positions["cube_2"][:2]))
            d3 = float(np.linalg.norm(poses["cube_3"][:2] - self._goal_positions["cube_3"][:2]))
            d4 = float(np.linalg.norm(poses["cube_4"][:2] - self._goal_positions["cube_4"][:2]))
            return max(d1, d2, d3, d4) >= thresh

        poses = None
        for _ in range(500):
            candidate = sample_positions()
            if (not positions_would_be_success(candidate)) and far_from_goal_enough(candidate):
                poses = candidate
                break
        if poses is None:
            # Deterministic unsolved fallback: keep bounds, but break L relations explicitly
            poses = sample_positions()
            poses["cube_1"] = np.array([0.42, -0.22, z], dtype=float)
            poses["cube_2"] = np.array([0.68,  0.22, z], dtype=float)
            poses["cube_3"] = np.array([0.42,  0.22, z], dtype=float)
            poses["cube_4"] = np.array([0.68, -0.22, z], dtype=float)

        self.cube_1.set_pose(position=poses["cube_1"], quat=random_quat_z())
        self.cube_2.set_pose(position=poses["cube_2"], quat=random_quat_z())
        self.cube_3.set_pose(position=poses["cube_3"], quat=random_quat_z())
        self.cube_4.set_pose(position=poses["cube_4"], quat=random_quat_z())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ----------------- Helpers -----------------
    def _pos(self, cube: ColoredCube) -> np.ndarray:
        return np.array(cube.body.get_position())

    def _aligned(self, a: ColoredCube, b: ColoredCube, axis: str, tol: float) -> bool:
        pa, pb = self._pos(a), self._pos(b)
        if axis == "y":
            # "Aligned along Y axis" => same X within tolerance
            return abs(pa[0] - pb[0]) <= tol
        if axis == "x":
            # "Aligned along X axis" => same Y within tolerance
            return abs(pa[1] - pb[1]) <= tol
        return False

    def _left_of(self, a: ColoredCube, b: ColoredCube, min_sep: float, tol: float) -> bool:
        pa, pb = self._pos(a), self._pos(b)
        # +Y is left. Require a_y > b_y with minimum separation (with tolerance slack).
        return (pa[1] - pb[1]) >= (min_sep - tol)

    def _back_of(self, a: ColoredCube, b: ColoredCube, min_sep: float, tol: float) -> bool:
        pa, pb = self._pos(a), self._pos(b)
        # +X is back/away. Require a_x > b_x with minimum separation (with tolerance slack).
        return (pa[0] - pb[0]) >= (min_sep - tol)

    def _dist_to_goal(self, cube: ColoredCube, goal_pos: np.ndarray) -> float:
        return float(np.linalg.norm(self._pos(cube)[:2] - goal_pos[:2]))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_positions"):
            return False

        # Goal condition checks (match task specification)
        aligned_12 = self._aligned(self.cube_1, self.cube_2, axis="y", tol=self._ALIGN_TOL_M)
        aligned_23 = self._aligned(self.cube_2, self.cube_3, axis="y", tol=self._ALIGN_TOL_M)
        left_12 = self._left_of(self.cube_1, self.cube_2, min_sep=self._MIN_SEP_M, tol=self._REL_TOL_M)
        left_23 = self._left_of(self.cube_2, self.cube_3, min_sep=self._MIN_SEP_M, tol=self._REL_TOL_M)
        aligned_34 = self._aligned(self.cube_3, self.cube_4, axis="x", tol=self._ALIGN_TOL_M)
        back_43 = self._back_of(self.cube_4, self.cube_3, min_sep=self._MIN_SEP_M, tol=self._REL_TOL_M)

        positions_ok = bool(aligned_12 and aligned_23 and left_12 and left_23 and aligned_34 and back_43)

        # Ensure cubes are released (not held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # No semantic constraints for ColoredCube beyond spatial arrangement
        semantic_ok = True

        # Subgoal approximation via "cube is near its goal pose and released"
        near_1 = self._dist_to_goal(self.cube_1, self._goal_positions["cube_1"]) <= 0.03
        near_2 = self._dist_to_goal(self.cube_2, self._goal_positions["cube_2"]) <= 0.03
        near_3 = self._dist_to_goal(self.cube_3, self._goal_positions["cube_3"]) <= 0.03
        near_4 = self._dist_to_goal(self.cube_4, self._goal_positions["cube_4"]) <= 0.03

        # Stages 0-2: cube_1 done
        if near_1 and released:
            for i in [0, 1, 2]:
                self._metric_stage(i, True)

        # Stages 3-5: cube_2 done
        if near_2 and released:
            for i in [3, 4, 5]:
                self._metric_stage(i, True)

        # Stages 6-8: cube_3 done
        if near_3 and released:
            for i in [6, 7, 8]:
                self._metric_stage(i, True)

        # Stages 9-11: cube_4 done
        if near_4 and released:
            for i in [9, 10, 11]:
                self._metric_stage(i, True)

        # Stage 12: alignment/overall arrangement
        if positions_ok and released:
            self._metric_stage(12, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_1_xy_to_goal_m": self._dist_to_goal(self.cube_1, self._goal_positions["cube_1"]),
                "cube_2_xy_to_goal_m": self._dist_to_goal(self.cube_2, self._goal_positions["cube_2"]),
                "cube_3_xy_to_goal_m": self._dist_to_goal(self.cube_3, self._goal_positions["cube_3"]),
                "cube_4_xy_to_goal_m": self._dist_to_goal(self.cube_4, self._goal_positions["cube_4"]),
            },
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})