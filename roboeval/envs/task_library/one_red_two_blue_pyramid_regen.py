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

class BlueBaseRedPyramid(RoboEvalEnv, MetricRolloutEval):
    """
    Build a 3-cube pyramid: two blue cubes form a side-by-side base (separated along the Y axis),
    and a red cube is stacked on top centered over the base.

    Sequential Sub-Goals:
        1. Grasp a blue cube intended for the left side of the base.
        2. Place the first blue cube on the table in the left half of the workspace (higher y).
        3. Grasp the second blue cube intended for the right side of the base.
        4. Place the second blue cube on the table to the right of the first (lower y), aligned in x.
        5. Adjust the blue base cubes so they are side-by-side along Y and aligned along X.
        6. Grasp the red cube that will become the top of the pyramid.
        7. Place the red cube on top of the two blue cubes, centered over their midpoint.
        8. Release the red cube after it is stably placed on top of the blue base.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace / object constants
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _CUBE_SIZE = 0.04
    _STACK_Z = _CUBE_REST_Z + _CUBE_SIZE  # 1.01

    # Goal tolerances from task spec
    _MIN_BASE_SEP_Y = 0.035
    _MAX_BASE_SEP_Y = 0.050  # keep bridge stable
    _ALIGN_TOL_X = 0.02
    _ALIGN_TOL_Z = 0.02
    _CENTER_TOL_X = 0.02
    _CENTER_TOL_Y = 0.02
    _MIN_CLEARANCE = 0.03

    # For metrics / task-level success tolerance
    _POS_TOL_FOR_METRICS = 0.014

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.blue_cube_left = ColoredCube(self._mojo)
        self.blue_cube_right = ColoredCube(self._mojo)
        self.red_cube_top = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.blue_cube_left.set_color("blue")
        self.blue_cube_right.set_color("blue")
        self.red_cube_top.set_color("red")

        self.cubes = [self.blue_cube_left, self.blue_cube_right, self.red_cube_top]

        # Deterministic goal poses (goal geometry: base along Y, top bridges midpoint)
        base_x = 0.55
        base_y_left = +0.02   # LEFT = +Y
        base_y_right = -0.02  # RIGHT = -Y

        self._goal_pos_left = np.array([base_x, base_y_left, self._CUBE_REST_Z])
        self._goal_pos_right = np.array([base_x, base_y_right, self._CUBE_REST_Z])
        self._goal_pos_top = np.array([base_x, 0.0, self._STACK_Z])

        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (pipeline may override later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.blue_cube_left.set_color("blue")
        self.blue_cube_right.set_color("blue")
        self.red_cube_top.set_color("red")

        # 3. Place objects in RANDOM unsolved positions within workspace bounds
        #    Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = self._CUBE_REST_Z

        def rand_quat_z():
            # random yaw about Z
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        def sample_pose():
            return np.array(
                [float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)), float(z_base)],
                dtype=float,
            )

        # Keep cubes separated on the table to avoid immediate accidental stacking/supporting,
        # and also ensure configuration is not already solved.
        min_pair_dist = 0.12  # > cube size; reduces accidental contacts/stacking
        max_tries = 2000

        pL = pR = pT = None
        qL = qR = qT = None

        for _ in range(max_tries):
            pL_cand = sample_pose()
            pR_cand = sample_pose()
            pT_cand = sample_pose()

            # Ensure pairwise XY separation
            dLR = np.linalg.norm((pL_cand - pR_cand)[:2])
            dLT = np.linalg.norm((pL_cand - pT_cand)[:2])
            dRT = np.linalg.norm((pR_cand - pT_cand)[:2])
            if dLR < min_pair_dist or dLT < min_pair_dist or dRT < min_pair_dist:
                continue

            # Ensure not in goal state / not satisfying success geometry by construction:
            # - Keep both blue cubes on table (z_base)
            # - Keep red cube on table (not stacked): z_base ensures not_on_table_ok fails
            # - Also avoid the base being within the required sep window and aligned in x at once
            sep_y = float(pL_cand[1] - pR_cand[1])
            aligned_x = abs(float(pL_cand[0] - pR_cand[0])) <= self._ALIGN_TOL_X
            sep_in_goal = (sep_y >= self._MIN_BASE_SEP_Y) and (sep_y <= self._MAX_BASE_SEP_Y)

            # It's enough to break at least one of these for unsolved base;
            # additionally red on table guarantees top_ok fails.
            if aligned_x and sep_in_goal:
                continue

            pL, pR, pT = pL_cand, pR_cand, pT_cand
            qL, qR, qT = rand_quat_z(), rand_quat_z(), rand_quat_z()
            break

        if pL is None:
            # Fallback deterministic unsolved arrangement (still within bounds)
            pL = np.array([0.45, 0.20, z_base], dtype=float)
            pR = np.array([0.65, -0.20, z_base], dtype=float)
            pT = np.array([0.55, 0.00, z_base], dtype=float)
            qL = qR = qT = rand_quat_z()

        # Place on table (no stacking)
        self.blue_cube_left.set_pose(position=pL, quat=qL)
        self.blue_cube_right.set_pose(position=pR, quat=qR)
        self.red_cube_top.set_pose(position=pT, quat=qT)
        self._mojo.physics.forward()

        # 4. Initialize subgoals to False
        for idx in range(8):
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

        # Current poses
        pL = self.blue_cube_left.body.get_position()
        pR = self.blue_cube_right.body.get_position()
        pT = self.red_cube_top.body.get_position()

        # --- Semantic (color) verification ---
        colors_ok = (
            self.blue_cube_left.get_color() == "blue"
            and self.blue_cube_right.get_color() == "blue"
            and self.red_cube_top.get_color() == "red"
        )

        # --- Base relationship checks (Y-axis is left/right) ---
        sep_y = float(pL[1] - pR[1])  # positive if left cube is actually left (+Y)
        left_of_ok = sep_y >= self._MIN_BASE_SEP_Y
        sep_not_too_large = sep_y <= self._MAX_BASE_SEP_Y

        aligned_x_ok = abs(float(pL[0] - pR[0])) <= self._ALIGN_TOL_X
        aligned_z_ok = abs(float(pL[2] - pR[2])) <= self._ALIGN_TOL_Z
        base_alignment_ok = aligned_x_ok and aligned_z_ok

        base_on_table = (
            self.blue_cube_left.is_colliding(self.table.table_top)
            and self.blue_cube_right.is_colliding(self.table.table_top)
        )

        # --- Top cube arrangement: centered on midpoint and stacked one level ---
        midpoint = 0.5 * (pL + pR)
        centered_x_ok = abs(float(pT[0] - midpoint[0])) <= self._CENTER_TOL_X
        centered_y_ok = abs(float(pT[1] - midpoint[1])) <= self._CENTER_TOL_Y
        centered_ok = centered_x_ok and centered_y_ok

        # Height relation: above base by at least min clearance; also not on table
        base_z = float(max(pL[2], pR[2]))
        clearance_ok = float(pT[2] - base_z) >= self._MIN_CLEARANCE
        not_on_table_ok = not self.red_cube_top.is_colliding(self.table.table_top)

        # Physical support: top cube should be colliding with both base cubes
        supported_by_base = (
            self.red_cube_top.is_colliding(self.blue_cube_left)
            and self.red_cube_top.is_colliding(self.blue_cube_right)
        )

        top_ok = centered_ok and clearance_ok and not_on_table_ok and supported_by_base

        # Overall spatial condition
        positions_ok = (
            left_of_ok
            and sep_not_too_large
            and base_alignment_ok
            and base_on_table
            and top_ok
        )

        # Released: no cube held
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal metrics (best-effort state-based staging) ---
        # 1-2: left blue placed near left base goal
        left_near_goal = np.linalg.norm(pL - self._goal_pos_left) <= 0.05
        left_placed = left_near_goal and self.blue_cube_left.is_colliding(self.table.table_top)

        # 3-4: right blue placed near right base goal
        right_near_goal = np.linalg.norm(pR - self._goal_pos_right) <= 0.05
        right_placed = right_near_goal and self.blue_cube_right.is_colliding(self.table.table_top)

        # 5: base aligned / side-by-side along Y
        base_ok_now = left_of_ok and sep_not_too_large and base_alignment_ok

        # 6-7: red cube placed near top goal and stacked correctly
        top_near_goal = np.linalg.norm(pT - self._goal_pos_top) <= 0.06

        self._metric_stage(0, left_near_goal)
        self._metric_stage(1, left_placed)
        self._metric_stage(2, right_near_goal)
        self._metric_stage(3, right_placed)
        self._metric_stage(4, base_ok_now)
        self._metric_stage(5, top_near_goal)
        self._metric_stage(6, top_ok)
        self._metric_stage(7, released)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and colors_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        target_distance = {
            "blue_cube_left": float(np.linalg.norm(pL - self._goal_pos_left)),
            "blue_cube_right": float(np.linalg.norm(pR - self._goal_pos_right)),
            "red_cube_top": float(np.linalg.norm(pT - self._goal_pos_top)),
        }
        pose_error = {
            "blue_cube_left_pos_err_m": target_distance["blue_cube_left"],
            "blue_cube_right_pos_err_m": target_distance["blue_cube_right"],
            "red_cube_top_pos_err_m": target_distance["red_cube_top"],
            "pos_tol_m": float(self._POS_TOL_FOR_METRICS),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})