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

class AlignedCubeRowPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a single left-to-right row (along the Y axis)
    and rotate them so all cubes share the same yaw angle with faces aligned.

    Sequential Sub-Goals:
        1. Order cubes along Y: blue (left, +Y) -> green -> red (right, -Y), with minimum spacing.
        2. Align cubes in X (front/back) so the row runs along Y (not X).
        3. Align all cube yaws (rotation about Z) so faces are aligned.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal tolerances (from task spec)
    _MIN_Y_SEP = 0.04
    _X_ALIGN_TOL = 0.015
    _YAW_TOL_DEG = 8.0

    # Deterministic goal layout (meets min separation, centered in workspace)
    _GOAL_X = 0.55
    _GOAL_YS = {
        "cube_blue": +0.06,   # left (+Y)
        "cube_green": 0.00,   # middle
        "cube_red": -0.06,    # right (-Y)
    }
    _GOAL_Z = 0.97
    _GOAL_YAW = 0.0  # radians (all cubes share the same yaw)

    _NUM_SUBGOALS = 3

    @staticmethod
    def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
        """Return quaternion (x,y,z,w) for a pure yaw rotation about +Z."""
        return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64)

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
        """Extract yaw (about Z) from quaternion (x,y,z,w)."""
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # yaw (Z) from quaternion (assuming standard ROS / xyzw convention)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _compute_goal_conditions(self):
        """Programmatic representation of the task goal conditions."""
        return [
            {
                "type": "arrangement",
                "subject": "cube_red",
                "target": "cube_green",
                "parameters": {"relation": "right_of", "axis_index": 1, "direction_sign": -1, "min_delta": self._MIN_Y_SEP},
            },
            {
                "type": "arrangement",
                "subject": "cube_green",
                "target": "cube_blue",
                "parameters": {"relation": "right_of", "axis_index": 1, "direction_sign": -1, "min_delta": self._MIN_Y_SEP},
            },
            {
                "type": "arrangement",
                "subject": "cube_red",
                "target": "cube_blue",
                "parameters": {"relation": "aligned", "axis_index": 0, "tolerance": self._X_ALIGN_TOL},
            },
            {
                "type": "orientation",
                "subject": "cube_red",
                "target": "cube_green",
                "parameters": {"relation": "same_yaw", "tolerance_degrees": self._YAW_TOL_DEG},
            },
            {
                "type": "orientation",
                "subject": "cube_green",
                "target": "cube_blue",
                "parameters": {"relation": "same_yaw", "tolerance_degrees": self._YAW_TOL_DEG},
            },
        ]

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube allowed)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]
        self._cube_name = {self.cube_red: "cube_red", self.cube_green: "cube_green", self.cube_blue: "cube_blue"}
        self._expected_color = {"cube_red": "red", "cube_green": "green", "cube_blue": "blue"}

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (even if unused here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 3. Place objects in RANDOM (unsolved) positions within workspace bounds.
        #    We force failure by (a) breaking X alignment and (b) breaking yaw alignment.
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        Z0 = self._GOAL_Z

        # Ensure X spread > _X_ALIGN_TOL by construction
        x_base = float(rng.uniform(X_MIN + 0.05, X_MAX - 0.05))
        dx_min = float(self._X_ALIGN_TOL + 0.03)  # strictly larger than tolerance
        dx1 = float(rng.uniform(dx_min, 0.10))
        dx2 = float(rng.uniform(dx_min, 0.10))

        xs = np.clip(
            np.array([x_base, x_base + dx1, x_base - dx2], dtype=np.float64),
            X_MIN,
            X_MAX,
        )

        # Random Y positions (not necessarily ordered)
        ys = rng.uniform(Y_MIN, Y_MAX, size=3).astype(np.float64)

        # Random yaws; ensure at least one pair differs by more than yaw tolerance
        yaw_tol = float(np.deg2rad(self._YAW_TOL_DEG))
        yaw0 = float(rng.uniform(-np.pi, np.pi))
        yaw1 = float(self._wrap_to_pi(yaw0 + float(rng.choice([-1.0, 1.0])) * (yaw_tol + float(rng.uniform(0.2, 1.0)))))
        yaw2 = float(rng.uniform(-np.pi, np.pi))

        q0 = self._yaw_to_quat_xyzw(yaw0)
        q1 = self._yaw_to_quat_xyzw(yaw1)
        q2 = self._yaw_to_quat_xyzw(yaw2)

        self.cube_red.set_pose(
            position=np.array([xs[0], ys[0], Z0], dtype=np.float64),
            quat=q0,
        )
        self.cube_green.set_pose(
            position=np.array([xs[1], ys[1], Z0], dtype=np.float64),
            quat=q1,
        )
        self.cube_blue.set_pose(
            position=np.array([xs[2], ys[2], Z0], dtype=np.float64),
            quat=q2,
        )

        # 4. Physics settling
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

        # Fetch poses
        pos_r = self.cube_red.body.get_position()
        pos_g = self.cube_green.body.get_position()
        pos_b = self.cube_blue.body.get_position()

        qr = self.cube_red.body.get_quaternion()
        qg = self.cube_green.body.get_quaternion()
        qb = self.cube_blue.body.get_quaternion()

        # --- Arrangement checks (Y-axis ordering and spacing) ---
        # "right_of" along Y with direction_sign = -1 means subject has LOWER y than target
        red_right_of_green = (pos_g[1] - pos_r[1]) >= self._MIN_Y_SEP
        green_right_of_blue = (pos_b[1] - pos_g[1]) >= self._MIN_Y_SEP
        ordering_ok = bool(red_right_of_green and green_right_of_blue)

        # --- X alignment check (row runs along Y; keep X aligned) ---
        xs = np.array([pos_r[0], pos_g[0], pos_b[0]], dtype=np.float64)
        x_spread = float(np.max(xs) - np.min(xs))
        x_aligned_ok = x_spread <= self._X_ALIGN_TOL

        # --- On-table sanity (helps prevent success while held / floating) ---
        on_table = (
            self.cube_red.is_colliding(self.table.table_top)
            and self.cube_green.is_colliding(self.table.table_top)
            and self.cube_blue.is_colliding(self.table.table_top)
            and abs(float(pos_r[2]) - self._GOAL_Z) < 0.03
            and abs(float(pos_g[2]) - self._GOAL_Z) < 0.03
            and abs(float(pos_b[2]) - self._GOAL_Z) < 0.03
        )

        # --- Orientation checks (same yaw within tolerance) ---
        yaw_r = self._yaw_from_quat_xyzw(qr)
        yaw_g = self._yaw_from_quat_xyzw(qg)
        yaw_b = self._yaw_from_quat_xyzw(qb)

        yaw_tol = np.deg2rad(self._YAW_TOL_DEG)
        def _yaw_diff_mod_90(a: float, b: float) -> float:
            d = (a - b) % (np.pi / 2.0)
            return min(d, np.pi / 2.0 - d)

        rg_ok = _yaw_diff_mod_90(yaw_r, yaw_g) <= yaw_tol
        gb_ok = _yaw_diff_mod_90(yaw_g, yaw_b) <= yaw_tol

        yaw_aligned_ok = bool(rg_ok and gb_ok)

        # --- Semantic/content checks (colors should remain correct) ---
        color_ok = (
            self.cube_red.get_color() == "red"
            and self.cube_green.get_color() == "green"
            and self.cube_blue.get_color() == "blue"
        )
        semantic_ok = bool(color_ok)

        # --- Released check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage tracking
        if ordering_ok:
            self._metric_stage(0, True)
        if ordering_ok and x_aligned_ok:
            self._metric_stage(1, True)
        if ordering_ok and x_aligned_ok and yaw_aligned_ok:
            self._metric_stage(2, True)

        positions_ok = bool(ordering_ok and x_aligned_ok and yaw_aligned_ok and on_table)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "y_sep_green_to_red_minus_min": float((pos_g[1] - pos_r[1]) - self._MIN_Y_SEP),
                "y_sep_blue_to_green_minus_min": float((pos_b[1] - pos_g[1]) - self._MIN_Y_SEP),
                "x_spread": float(x_spread),
            },
            pose_error={
                "yaw_rg_deg": float(np.rad2deg(abs(self._wrap_to_pi(yaw_r - yaw_g)))),
                "yaw_gb_deg": float(np.rad2deg(abs(self._wrap_to_pi(yaw_g - yaw_b)))),
                "yaw_rb_deg": float(np.rad2deg(abs(self._wrap_to_pi(yaw_r - yaw_b)))),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})