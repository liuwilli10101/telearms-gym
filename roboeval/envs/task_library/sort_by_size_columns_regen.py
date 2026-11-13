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

class CubeSizeColumnGrouping(RoboEvalEnv, MetricRolloutEval):
    """
    Sort four cubes into two side-by-side columns: standard ColoredCubes in the left column (+Y)
    and "small" cubes in the right column (-Y), all within the specified workspace bounds.

    Note: The task specification refers to "standard" vs "small" cubes, but this environment is
    constrained to use ONLY the ColoredCube asset type. We therefore represent the two groups
    as four ColoredCubes with distinct colors and enforce the required column arrangement purely
    via position-based success conditions.

    Sequential Sub-Goals:
        1. Grasp std_cube_1 with either gripper.
        2. Move std_cube_1 to the left column region (positive Y).
        3. Place std_cube_1 on the table in the left column.
        4. Grasp std_cube_2 with either gripper.
        5. Move std_cube_2 to the left column region (positive Y).
        6. Place std_cube_2 on the table in the left column (separated along X).
        7. Grasp small_cube_1 with either gripper.
        8. Move small_cube_1 to the right column region (negative Y).
        9. Place small_cube_1 on the table in the right column.
        10. Grasp small_cube_2 with either gripper.
        11. Move small_cube_2 to the right column region (negative Y).
        12. Place small_cube_2 on the table in the right column (separated along X).
        13. Align the two standard cubes into a clean left column (same Y, separated in X).
        14. Align the two small cubes into a clean right column (same Y, separated in X).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants (from specification) ---
    _X_MIN, _X_MAX = 0.40, 0.70
    _Y_MIN, _Y_MAX = -0.25, 0.25

    _LEFT_Y_CENTER = 0.12
    _RIGHT_Y_CENTER = -0.12
    _Y_TOL = 0.03

    _MIN_PAIRWISE_X_SEP = 0.05
    _REC_X_BAND = (0.45, 0.65)

    _MIN_COLUMN_SEP_Y = 0.12

    _CUBE_Z_ON_TABLE = 0.97
    _NUM_SUBGOALS = 14

    def _initialize_env(self):
        # Props (table is in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY asset allowed: ColoredCube)
        self.std_cube_1 = ColoredCube(self._mojo)
        self.std_cube_2 = ColoredCube(self._mojo)
        self.small_cube_1 = ColoredCube(self._mojo)
        self.small_cube_2 = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.std_cube_1.set_color("red")
        self.std_cube_2.set_color("blue")
        self.small_cube_1.set_color("green")
        self.small_cube_2.set_color("yellow")

        self.cubes = [self.std_cube_1, self.std_cube_2, self.small_cube_1, self.small_cube_2]
        self._cube_names = ["std_cube_1", "std_cube_2", "small_cube_1", "small_cube_2"]

        # Deterministic goal poses (two X slots per column)
        self._goal_poses = {
            "std_cube_1": np.array([0.48, self._LEFT_Y_CENTER, self._CUBE_Z_ON_TABLE]),
            "std_cube_2": np.array([0.60, self._LEFT_Y_CENTER, self._CUBE_Z_ON_TABLE]),
            "small_cube_1": np.array([0.48, self._RIGHT_Y_CENTER, self._CUBE_Z_ON_TABLE]),
            "small_cube_2": np.array([0.60, self._RIGHT_Y_CENTER, self._CUBE_Z_ON_TABLE]),
        }

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
        # 1) Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # 3) REQUIRED: set_color() in _on_reset
        self.std_cube_1.set_color("red")
        self.std_cube_2.set_color("blue")
        self.small_cube_1.set_color("green")
        self.small_cube_2.set_color("yellow")

        # 4) Place objects in RANDOM UNSOLVED positions (so _success() returns False)
        # Strategy: sample all cubes in a central Y band that is NOT within either column tolerance.
        # Also enforce a small minimum XY separation to avoid initial cube-cube collisions.
        x_low, x_high = self._X_MIN + 0.01, self._X_MAX - 0.01
        # Central band far from +/-0.12 by > _Y_TOL (+ margin)
        y_low, y_high = -0.06, 0.06
        y_low = max(y_low, self._Y_MIN + 0.01)
        y_high = min(y_high, self._Y_MAX - 0.01)

        min_xy_sep = 0.065  # conservative to reduce collision risk

        def sample_pos(existing: list[np.ndarray]) -> np.ndarray:
            for _ in range(500):
                pos = np.array(
                    [
                        float(rng.uniform(x_low, x_high)),
                        float(rng.uniform(y_low, y_high)),
                        float(self._CUBE_Z_ON_TABLE),
                    ],
                    dtype=float,
                )
                if all(np.linalg.norm(pos[:2] - e[:2]) >= min_xy_sep for e in existing):
                    return pos
            # Fallback: deterministic spread if rejection sampling fails
            base_xs = np.linspace(x_low, x_high, 4)
            idx = len(existing)
            return np.array([float(base_xs[idx]), float(rng.uniform(y_low, y_high)), float(self._CUBE_Z_ON_TABLE)])

        existing: list[np.ndarray] = []
        p1 = sample_pos(existing); existing.append(p1)
        p2 = sample_pos(existing); existing.append(p2)
        p3 = sample_pos(existing); existing.append(p3)
        p4 = sample_pos(existing); existing.append(p4)

        quat = np.array([0.0, 0.0, 0.0, 1.0])

        self.std_cube_1.set_pose(position=p1, quat=quat)
        self.std_cube_2.set_pose(position=p2, quat=quat)
        self.small_cube_1.set_pose(position=p3, quat=quat)
        self.small_cube_2.set_pose(position=p4, quat=quat)

        # 5) Physics settling
        self._mojo.physics.forward()

        # 6) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ------------------------- Success helpers -------------------------

    def _within_workspace_xy(self, pos: np.ndarray) -> bool:
        return (self._X_MIN <= pos[0] <= self._X_MAX) and (self._Y_MIN <= pos[1] <= self._Y_MAX)

    def _in_column(self, pos: np.ndarray, y_center: float, y_tol: float) -> bool:
        if not self._within_workspace_xy(pos):
            return False
        return abs(float(pos[1]) - float(y_center)) <= float(y_tol)

    def _in_recommended_x_band(self, pos: np.ndarray) -> bool:
        return self._REC_X_BAND[0] <= float(pos[0]) <= self._REC_X_BAND[1]

    def _is_on_table(self, cube: ColoredCube) -> bool:
        # Table-top collision is reliable; also guard against floor if available.
        if not cube.is_colliding(self.table.table_top):
            return False
        if hasattr(self, "_floor") and cube.is_colliding(self._floor):
            return False
        return True

    def _any_pair_colliding(self, objs) -> bool:
        n = len(objs)
        for i in range(n):
            for j in range(i + 1, n):
                if objs[i].is_colliding(objs[j]):
                    return True
        return False

    # ------------------------------ Success ------------------------------

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Current positions
        p_std1 = self.std_cube_1.body.get_position()
        p_std2 = self.std_cube_2.body.get_position()
        p_sml1 = self.small_cube_1.body.get_position()
        p_sml2 = self.small_cube_2.body.get_position()

        # Per-cube placement conditions (in correct column, within bounds, on table)
        std1_ok = self._in_column(p_std1, self._LEFT_Y_CENTER, self._Y_TOL) and self._is_on_table(self.std_cube_1)
        std2_ok = self._in_column(p_std2, self._LEFT_Y_CENTER, self._Y_TOL) and self._is_on_table(self.std_cube_2)
        sml1_ok = self._in_column(p_sml1, self._RIGHT_Y_CENTER, self._Y_TOL) and self._is_on_table(self.small_cube_1)
        sml2_ok = self._in_column(p_sml2, self._RIGHT_Y_CENTER, self._Y_TOL) and self._is_on_table(self.small_cube_2)

        # Column alignment checks (same Y within tolerance, separated along X)
        left_y_aligned = abs(float(p_std1[1]) - float(p_std2[1])) <= self._Y_TOL
        right_y_aligned = abs(float(p_sml1[1]) - float(p_sml2[1])) <= self._Y_TOL

        left_x_sep_ok = abs(float(p_std1[0]) - float(p_std2[0])) >= self._MIN_PAIRWISE_X_SEP
        right_x_sep_ok = abs(float(p_sml1[0]) - float(p_sml2[0])) >= self._MIN_PAIRWISE_X_SEP

        left_x_band_ok = self._in_recommended_x_band(p_std1) and self._in_recommended_x_band(p_std2)
        right_x_band_ok = self._in_recommended_x_band(p_sml1) and self._in_recommended_x_band(p_sml2)

        left_group_ok = bool(left_y_aligned and left_x_sep_ok and left_x_band_ok and std1_ok and std2_ok)
        right_group_ok = bool(right_y_aligned and right_x_sep_ok and right_x_band_ok and sml1_ok and sml2_ok)

        # Left column must be to the left (+Y) of right column (-Y) with separation
        left_y_mean = 0.5 * (float(p_std1[1]) + float(p_std2[1]))
        right_y_mean = 0.5 * (float(p_sml1[1]) + float(p_sml2[1]))
        column_sep_ok = (left_y_mean - right_y_mean) >= self._MIN_COLUMN_SEP_Y

        # Avoid cube-cube interpenetration (should be true if spaced correctly)
        no_collisions = not self._any_pair_colliding(self.cubes)

        positions_ok = bool(left_group_ok and right_group_ok and column_sep_ok and no_collisions)

        # No additional semantic constraints beyond arrangement
        semantic_ok = True

        # Must not be holding any cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # Stage metrics (best-effort mapping to subgoals; we infer completion from final placements)
        # std_cube_1: stages 0,1,2
        if std1_ok:
            self._metric_stage(0, True)
            self._metric_stage(1, True)
            self._metric_stage(2, True)

        # std_cube_2: stages 3,4,5
        if std2_ok:
            self._metric_stage(3, True)
            self._metric_stage(4, True)
            self._metric_stage(5, True)

        # small_cube_1: stages 6,7,8
        if sml1_ok:
            self._metric_stage(6, True)
            self._metric_stage(7, True)
            self._metric_stage(8, True)

        # small_cube_2: stages 9,10,11
        if sml2_ok:
            self._metric_stage(9, True)
            self._metric_stage(10, True)
            self._metric_stage(11, True)

        # Alignment stages: 12, 13
        if left_group_ok:
            self._metric_stage(12, True)
        if right_group_ok:
            self._metric_stage(13, True)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalization inputs
        target_distance = {
            "std_cube_1": float(np.linalg.norm(p_std1 - self._goal_poses["std_cube_1"])),
            "std_cube_2": float(np.linalg.norm(p_std2 - self._goal_poses["std_cube_2"])),
            "small_cube_1": float(np.linalg.norm(p_sml1 - self._goal_poses["small_cube_1"])),
            "small_cube_2": float(np.linalg.norm(p_sml2 - self._goal_poses["small_cube_2"])),
        }
        pose_error = {
            "std_cube_1_y_err": float(abs(p_std1[1] - self._LEFT_Y_CENTER)),
            "std_cube_2_y_err": float(abs(p_std2[1] - self._LEFT_Y_CENTER)),
            "small_cube_1_y_err": float(abs(p_sml1[1] - self._RIGHT_Y_CENTER)),
            "small_cube_2_y_err": float(abs(p_sml2[1] - self._RIGHT_Y_CENTER)),
            "left_pair_y_err": float(abs(p_std1[1] - p_std2[1])),
            "right_pair_y_err": float(abs(p_sml1[1] - p_sml2[1])),
            "left_pair_x_sep": float(abs(p_std1[0] - p_std2[0])),
            "right_pair_x_sep": float(abs(p_sml1[0] - p_sml2[0])),
            "column_y_sep": float(left_y_mean - right_y_mean),
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