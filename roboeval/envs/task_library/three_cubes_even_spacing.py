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
    Arrange three colored cubes into a single horizontal row on the table with equal
    spacing of 8 cm between adjacent cubes. Color order does not matter.

    Sequential Sub-Goals:
        1. Pick any cube.
        2. Move cubes toward a common row line.
        3. Place cubes down on the table (not stacked).
        4. Align cubes to be collinear (horizontal row).
        5. Adjust spacing so adjacent cubes are 8 cm apart.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from specification)
    _N_CUBES = 3
    _PRIMARY_AXIS = "y"  # the row runs along Y (screen-horizontal); cubes share the same X
    _SPACING_M = 0.08
    _SPACING_TOL_M = 0.01
    _COLLINEAR_TOL_M = 0.01
    _HEIGHT_TOL_M = 0.01
    _TABLE_Z_CUBE = 0.97

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY asset allowed)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._N_CUBES)]

        # REQUIRED: set_color() in _initialize_env()
        # (Keep fixed distinct colors; order doesn't matter for success.)
        init_colors = ["red", "green", "blue"]
        for cube, col in zip(self.cubes, init_colors):
            cube.set_color(col)

        # Nominal goal row (used for metrics only; success is relative geometry-based)
        cx, cy, cz = 0.55, 0.0, self._TABLE_Z_CUBE
        half_span = self._SPACING_M
        self._goal_positions = [
            np.array([cx - half_span, cy, cz]),
            np.array([cx, cy, cz]),
            np.array([cx + half_span, cy, cz]),
        ]

        # Initialize metrics (FULL signature REQUIRED)
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

        # 2. Deterministic RNG (REQUIRED pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        reset_colors = ["red", "green", "blue"]
        for cube, col in zip(self.cubes, reset_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM unsolved configuration within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._TABLE_Z_CUBE

        quat = np.array([0.0, 0.0, 0.0, 1.0])
        min_xy_sep = 0.06  # simple overlap-avoidance margin

        def sample_positions():
            poses = []
            for _ in range(self._N_CUBES):
                for _try in range(200):
                    x = float(rng.uniform(x_min, x_max))
                    y = float(rng.uniform(y_min, y_max))
                    p = np.array([x, y, z], dtype=float)
                    if all(np.linalg.norm(p[:2] - q[:2]) >= min_xy_sep for q in poses):
                        poses.append(p)
                        break
                else:
                    # Fallback: accept even if closer (should be rare)
                    poses.append(p)
            return poses

        # Rejection sampling: avoid starting in a solved state
        for _ in range(200):
            poses = sample_positions()
            for cube, pos in zip(self.cubes, poses):
                cube.set_pose(position=pos, quat=quat)

            self._mojo.physics.forward()

            arrangement_ok, _ = self._row_spacing_and_collinearity_ok()
            on_table_ok = self._all_on_table()

            # We want reset to be UNSOLVED: break if not satisfying success geometry/table constraints
            if not (arrangement_ok and on_table_ok):
                break

            # If we accidentally sampled a solved state, force-break collinearity by moving one cube in X.
            p0 = self.cubes[0].body.get_position().copy()
            delta_x = float(rng.choice([-0.05, 0.05]))
            p0[0] = float(np.clip(p0[0] + delta_x, x_min, x_max))
            self.cubes[0].set_pose(position=p0, quat=quat)
            self._mojo.physics.forward()

            arrangement_ok2, _ = self._row_spacing_and_collinearity_ok()
            on_table_ok2 = self._all_on_table()
            if not (arrangement_ok2 and on_table_ok2):
                break

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _row_spacing_and_collinearity_ok(self):
        """
        Returns:
            positions_ok (bool), diagnostics (dict)
        """
        poses = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in poses], dtype=float)
        ys = np.array([p[1] for p in poses], dtype=float)
        zs = np.array([p[2] for p in poses], dtype=float)

        # Collinearity for a horizontal row along Y: X-values should match closely.
        x_span = float(np.max(xs) - np.min(xs))
        collinear_ok = x_span <= self._COLLINEAR_TOL_M

        # Same table height (not stacked)
        z_span = float(np.max(zs) - np.min(zs))
        same_height_ok = z_span <= self._HEIGHT_TOL_M

        # Sort cubes by Y, then check adjacent Y spacings.
        y_sorted = np.sort(ys)
        dy1 = float(y_sorted[1] - y_sorted[0])
        dy2 = float(y_sorted[2] - y_sorted[1])
        spacing_err1 = abs(dy1 - self._SPACING_M)
        spacing_err2 = abs(dy2 - self._SPACING_M)
        spacing_ok = (spacing_err1 <= self._SPACING_TOL_M) and (spacing_err2 <= self._SPACING_TOL_M)

        diagnostics = {
            "x_span_m": x_span,
            "z_span_m": z_span,
            "dy1_m": dy1,
            "dy2_m": dy2,
            "spacing_err1_m": float(spacing_err1),
            "spacing_err2_m": float(spacing_err2),
        }
        return bool(collinear_ok and same_height_ok and spacing_ok), diagnostics

    def _all_on_table(self) -> bool:
        # Table contact is a robust proxy for "resting on table" and "not stacked".
        return all(c.is_colliding(self.table.table_top) for c in self.cubes)

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        arrangement_ok, diag = self._row_spacing_and_collinearity_ok()
        on_table_ok = self._all_on_table()

        positions_ok = bool(arrangement_ok and on_table_ok)
        semantic_ok = True  # No semantic/color-order constraints for this task

        # Stage completion (no grasp-state required; mark stages when final geometry is satisfied)
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        # Provide task-relevant scalar errors.
        mean_pos = np.mean(np.stack([c.body.get_position() for c in self.cubes], axis=0), axis=0)
        target_center = np.array([0.55, 0.0, self._TABLE_Z_CUBE], dtype=float)
        center_error = float(np.linalg.norm(mean_pos - target_center))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "row_center_error_m": center_error,
                "spacing_err1_m": diag.get("spacing_err1_m", None),
                "spacing_err2_m": diag.get("spacing_err2_m", None),
                "x_span_m": diag.get("x_span_m", None),
                "z_span_m": diag.get("z_span_m", None),
            },
            pose_error={
                "dy1_m": diag.get("dy1_m", None),
                "dy2_m": diag.get("dy2_m", None),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})