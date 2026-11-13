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

class CubeRowSpacingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes into a single horizontal row on the table with exactly
    5 cm gaps between each adjacent pair. The cubes should be aligned and evenly spaced.

    Sequential Sub-Goals:
        1. Align all cubes into a straight row along the table Y axis (small X deviation).
        2. Achieve a 5 cm gap between the leftmost two adjacent cubes (±5 mm).
        3. Achieve a 5 cm gap between the middle adjacent cubes (±5 mm).
        4. Achieve a 5 cm gap between the rightmost two adjacent cubes (±5 mm).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry / tolerances from task spec
    _CUBE_SIZE_M = 0.04
    _TARGET_GAP_M = 0.05
    _GAP_TOL_M = 0.005
    _ROW_LAT_TOL_M = 0.01  # lateral deviation (Y) tolerance

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly four ColoredCubes (ONLY allowed asset)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4]

        # REQUIRED: set_color() in _initialize_env
        self._cube_colors = ["red", "blue", "green", "yellow"]
        for cube, color in zip(self.cubes, self._cube_colors):
            cube.set_color(color)

        # Pre-compute a canonical goal row (used for metrics; success is permutation-invariant)
        center_x, center_y = 0.55, 0.0
        center_to_center = self._CUBE_SIZE_M + self._TARGET_GAP_M  # 0.09m
        xs = [center_x + (i - 1.5) * center_to_center for i in range(4)]
        self._goal_positions = [np.array([x, center_y, 0.97], dtype=float) for x in xs]

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

        # 2. Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        for cube, color in zip(self.cubes, self._cube_colors):
            cube.set_color(color)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False after reset)
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z_base = 0.97

        # Rejection sample non-overlapping positions and enforce NOT row-aligned
        # (row_aligned requires max |y - mean_y| <= 0.01, so enforce > 0.02)
        min_center_dist_xy = 0.065  # > cube size (0.04) to avoid overlaps
        max_tries = 500

        def _ok_positions_xy(pts_xy: np.ndarray) -> bool:
            # Non-overlap in XY
            for i in range(len(pts_xy)):
                for j in range(i + 1, len(pts_xy)):
                    if float(np.linalg.norm(pts_xy[i] - pts_xy[j])) < min_center_dist_xy:
                        return False
            # Ensure NOT row-aligned: the row is along Y now, so force the cubes to span X
            # (row_aligned requires max |x - mean_x| <= 0.01, so enforce > 0.02).
            xs_ = pts_xy[:, 0]
            x_mean = float(np.mean(xs_))
            if float(np.max(np.abs(xs_ - x_mean))) <= 0.02:
                return False
            return True

        pts_xy = None
        for _ in range(max_tries):
            xs = rng.uniform(x_low, x_high, size=(4,))
            ys = rng.uniform(y_low, y_high, size=(4,))
            cand = np.stack([xs, ys], axis=1)
            if _ok_positions_xy(cand):
                pts_xy = cand
                break

        # Fallback (should be rare): deterministic unsolved spread in Y within bounds
        if pts_xy is None:
            xs = rng.uniform(x_low, x_high, size=(4,))
            ys = np.array([-0.12, 0.12, -0.18, 0.18], dtype=float)
            pts_xy = np.stack([xs, ys], axis=1)

        # Apply poses (random yaw; keep upright)
        for cube, (x, y) in zip(self.cubes, pts_xy):
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
            cube.set_pose(position=np.array([float(x), float(y), z_base], dtype=float), quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(4):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _gap_between_cubes(self, a: ColoredCube, b: ColoredCube) -> float:
        """Empty-space gap along Y between two cube bodies, assuming cube size is 0.04m."""
        pa = a.body.get_position()
        pb = b.body.get_position()
        return float(abs(pb[1] - pa[1]) - self._CUBE_SIZE_M)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Released check (single- or multi-gripper compatible)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Basic table contact / height sanity (spatial task; keep it light but safe)
        on_table = True
        for c in self.cubes:
            pos = c.body.get_position()
            if abs(float(pos[2]) - 0.97) > 0.02:
                on_table = False
                break
            # Prefer explicit table-top collision as a robust indicator it's resting on the table
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break

        # Permutation-invariant evaluation:
        # Sort cubes by Y so "adjacent" is defined left-to-right along table Y (the row runs
        # along Y, the screen-horizontal direction; cubes share the same X depth).
        cubes_sorted = sorted(self.cubes, key=lambda cc: float(cc.body.get_position()[1]))
        xs = np.array([c.body.get_position()[0] for c in cubes_sorted], dtype=float)
        ys = np.array([c.body.get_position()[1] for c in cubes_sorted], dtype=float)

        # Row alignment: all X nearly equal (within 1 cm depth deviation)
        x_mean = float(np.mean(xs))
        row_aligned = bool(np.max(np.abs(xs - x_mean)) <= self._ROW_LAT_TOL_M)

        # Adjacent gaps: compute empty-space gap (center distance - cube_size)
        gaps = [self._gap_between_cubes(cubes_sorted[i], cubes_sorted[i + 1]) for i in range(3)]
        gap_ok = [abs(g - self._TARGET_GAP_M) <= self._GAP_TOL_M for g in gaps]

        # Subgoal staging
        if row_aligned and on_table:
            self._metric_stage(0, True)
        if gap_ok[0] and row_aligned and on_table:
            self._metric_stage(1, True)
        if gap_ok[1] and row_aligned and on_table:
            self._metric_stage(2, True)
        if gap_ok[2] and row_aligned and on_table:
            self._metric_stage(3, True)

        positions_ok = bool(row_aligned and all(gap_ok) and on_table)
        semantic_ok = True  # no semantic constraints in this task

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        gap_errors = [abs(g - self._TARGET_GAP_M) for g in gaps]
        pose_error = {
            "row_lateral_max_abs_x_error_m": float(np.max(np.abs(xs - x_mean))) if len(xs) else 1.0,
            "gap_12_abs_error_m": float(gap_errors[0]) if len(gap_errors) > 0 else 1.0,
            "gap_23_abs_error_m": float(gap_errors[1]) if len(gap_errors) > 1 else 1.0,
            "gap_34_abs_error_m": float(gap_errors[2]) if len(gap_errors) > 2 else 1.0,
        }

        # A simple "target_distance" metric: mean absolute gap error (0 is perfect)
        target_distance = {
            "mean_abs_gap_error_m": float(np.mean(gap_errors)) if len(gap_errors) else 1.0
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})