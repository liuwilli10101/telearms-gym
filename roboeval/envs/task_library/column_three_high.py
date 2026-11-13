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

class ColoredCubeStackingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Stack three colored cubes vertically to form a single stable tower. The final configuration
    is a one-column stack with two cubes resting on top of another cube in sequence.

    Sequential Sub-Goals:
        1. Grasp any one cube to start building the tower.
        2. Move the grasped cube to become the bottom cube position on the table.
        3. Release the bottom cube in a stable pose on the table.
        4. Grasp a second cube to place as the middle cube.
        5. Lift the second cube above the bottom cube height to prepare for stacking.
        6. Align the second cube over the bottom cube for a centered placement.
        7. Place the second cube on top of the bottom cube to form a 2-cube stack.
        8. Grasp the third cube to place as the top cube.
        9. Lift the third cube above the current stack height.
        10. Align the third cube over the middle cube for a centered placement.
        11. Place the third cube on top of the middle cube to complete a 3-cube vertical tower.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants (ColoredCube geometry) ---
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _CUBE_DZ = 0.04

    # Success tolerances
    _XY_ALIGN_TOL = 0.014  # meters (task constraint)
    _Z_STACK_TOL = 0.02    # meters (lenient, collision does most of the work)

    _NUM_SUBGOALS = 11

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly three ColoredCube assets (and only this asset type)
        self.cube_bottom = ColoredCube(self._mojo)
        self.cube_middle = ColoredCube(self._mojo)
        self.cube_top = ColoredCube(self._mojo)

        self.cubes = [self.cube_bottom, self.cube_middle, self.cube_top]

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "green", "blue"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Define canonical goal stack positions (single vertical column)
        base_xy = np.array([0.55, 0.00], dtype=float)
        self._goal_positions = {
            "bottom": np.array([base_xy[0], base_xy[1], self._CUBE_REST_Z], dtype=float),
            "middle": np.array([base_xy[0], base_xy[1], self._CUBE_REST_Z + self._CUBE_DZ], dtype=float),
            "top": np.array([base_xy[0], base_xy[1], self._CUBE_REST_Z + 2 * self._CUBE_DZ], dtype=float),
        }
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

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

        # 2. Deterministic RNG (pipeline may apply later perturbations)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        # (Keep deterministic color assignment)
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False after reset)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z_base = self._CUBE_REST_Z

        # Keep cubes well-separated on the table so they are not stacked/colliding
        min_xy_sep = 0.07  # conservative separation to avoid accidental collisions/stacking
        goal_xy = self._goal_positions["bottom"][:2].copy()
        min_goal_xy_sep = 0.06  # avoid spawning right at the goal column

        positions = []
        max_tries = 500

        def sample_xy():
            return np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=float)

        for _ in range(3):
            for _try in range(max_tries):
                xy = sample_xy()
                if np.linalg.norm(xy - goal_xy) < min_goal_xy_sep:
                    continue
                if any(np.linalg.norm(xy - p) < min_xy_sep for p in positions):
                    continue
                positions.append(xy)
                break
            else:
                # Fallback: deterministic spread if sampling fails
                positions = [
                    np.array([0.45, -0.18], dtype=float),
                    np.array([0.62, 0.18], dtype=float),
                    np.array([0.68, -0.02], dtype=float),
                ]
                break

        # Random yaw-only orientations (keep cubes upright)
        def yaw_quat(yaw: float) -> np.ndarray:
            q = Quaternion(axis=[0.0, 0.0, 1.0], radians=float(yaw))
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        quats = [yaw_quat(rng.uniform(-np.pi, np.pi)) for _ in range(3)]

        # Place all cubes on the table (same base Z), not stacked
        for cube, xy, quat in zip(self.cubes, positions, quats):
            cube.set_pose(position=np.array([xy[0], xy[1], z_base], dtype=float), quat=quat)
            self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        bottom = self.cube_bottom
        middle = self.cube_middle
        top = self.cube_top

        # --- Spatial checks: relative "on" + column alignment ---
        # Collision-based "on" checks (directionally robust in this benchmark)
        bottom_on_table = bottom.is_colliding(self.table.table_top)
        middle_on_bottom = middle.is_colliding(bottom) and (not middle.is_colliding(self.table.table_top))
        top_on_middle = top.is_colliding(middle) and (not top.is_colliding(self.table.table_top))

        # Position-based alignment checks (approximate x-y column)
        pb = bottom.body.get_position()
        pm = middle.body.get_position()
        pt = top.body.get_position()

        xy_bm = np.linalg.norm((pm - pb)[:2])
        xy_mt = np.linalg.norm((pt - pm)[:2])
        aligned_xy = (xy_bm < self._XY_ALIGN_TOL) and (xy_mt < self._XY_ALIGN_TOL)

        # Optional vertical spacing sanity (avoid side-contact false positives)
        dz_bm = pm[2] - pb[2]
        dz_mt = pt[2] - pm[2]
        good_vertical_spacing = (abs(dz_bm - self._CUBE_DZ) < self._Z_STACK_TOL) and (
            abs(dz_mt - self._CUBE_DZ) < self._Z_STACK_TOL
        )

        positions_ok = bool(
            bottom_on_table and middle_on_bottom and top_on_middle and aligned_xy and good_vertical_spacing
        )
        semantic_ok = True  # No semantic/color constraints in this task definition

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoal stages: mark all complete once full tower is achieved (simple, robust)
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics payloads
        target_distance = {
            "middle_to_bottom_center_dist": float(distance(middle.body, bottom.body)),
            "top_to_middle_center_dist": float(distance(top.body, middle.body)),
            "xy_align_bottom_middle": float(xy_bm),
            "xy_align_middle_top": float(xy_mt),
        }
        pose_error = {
            "bottom_pos_err": float(np.linalg.norm(pb - self._goal_positions["bottom"])),
            "middle_pos_err": float(np.linalg.norm(pm - self._goal_positions["middle"])),
            "top_pos_err": float(np.linalg.norm(pt - self._goal_positions["top"])),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})