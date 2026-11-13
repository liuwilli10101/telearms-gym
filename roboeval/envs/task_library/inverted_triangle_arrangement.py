# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

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
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class InvertedTriangleCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes on the table to form an inverted triangle, with the
    single apex cube closest to the robot and the other two cubes forming a base row
    farther away.

    Sequential Sub-Goals:
        1. Grasp any one cube.
        2. Transport it toward an intended vertex of the inverted triangle.
        3. Place and release it on the table.
        4. Repeat grasp/transport/place for the remaining cubes (sequentially).
        5. Align/nudge so the base pair is level (same depth) and the apex is centered
           and closer to the robot.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task geometry parameters (from spec) ---
    _BASE_DEPTH_TOL_M = 0.03
    _BASE_MIN_LAT_SEP_M = 0.08
    _BASE_MAX_LAT_SEP_M = 0.18
    _APEX_CLOSER_BY_M = 0.06
    _APEX_CENTER_TOL_M = 0.03

    # Deterministic "ideal" goal vertex locations (all on table)
    # NOTE: "Closer to robot" corresponds to larger X in this benchmark coordinate setup.
    _GOAL_BASE_X = 0.52
    _GOAL_APEX_X = 0.59  # add slack vs _APEX_CLOSER_BY_M to avoid borderline failures
    _GOAL_BASE_Y_OFF = 0.06  # base separation = 0.12m (within [0.08, 0.18])
    _GOAL_APEX_Y = 0.0
    _CUBE_Z = 0.97

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ColoredCube only)
        self.cubes = [ColoredCube(self._mojo) for _ in range(3)]

        # REQUIRED: Call set_color() in _initialize_env()
        self._cube_colors = ["red", "green", "blue"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Cache goal vertices (apex, base-left, base-right) for metric calculations
        self._goal_vertices = [
            np.array([self._GOAL_APEX_X, self._GOAL_APEX_Y, self._CUBE_Z]),
            np.array([self._GOAL_BASE_X, +self._GOAL_BASE_Y_OFF, self._CUBE_Z]),
            np.array([self._GOAL_BASE_X, -self._GOAL_BASE_Y_OFF, self._CUBE_Z]),
        ]

        # Initialize metrics (FULL signature)
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
        _ = rng.random()  # (intentionally unused; ensures deterministic path)

        # REQUIRED: Call set_color() in _on_reset()
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # 3. Place objects in RANDOM unsolved positions within workspace bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z = 0.97
        # We force an UNSOLVED configuration by making all cubes share (approximately) the same X,
        # which guarantees apex_depth_ok fails (apex_x - base_x_mean == 0 < _APEX_CLOSER_BY_M).
        x_shared = float(rng.uniform(0.40, 0.70))
        min_y_sep = 0.05  # simple anti-overlap heuristic

        ys = []
        for _i in range(3):
            for _attempt in range(200):
                y = float(rng.uniform(-0.25, 0.25))
                if all(abs(y - yj) >= min_y_sep for yj in ys):
                    ys.append(y)
                    break
            else:
                # Fallback deterministic spread if sampling fails
                ys = [-0.12, 0.0, 0.12]
                break

        # Small x jitter (kept tiny so apex_depth_ok still fails robustly)
        x_jitter = rng.uniform(-0.002, 0.002, size=3)

        for c, y, dx in zip(self.cubes, ys, x_jitter):
            x = float(np.clip(x_shared + float(dx), 0.40, 0.70))
            c.set_pose(
                position=np.array([x, float(np.clip(y, -0.25, 0.25)), self._CUBE_Z], dtype=float),
                quat=np.array([0, 0, 0, 1], dtype=float),
            )

        # 4. Physics settling
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        # (We mark all as True only on full success; no grasp-state success criteria.)
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _euclid(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _min_assignment_error(self, positions, targets):
        """Brute-force minimal sum of distances between 3 cube positions and 3 target vertices."""
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        best = float("inf")
        best_perm = perms[0]
        for p in perms:
            s = (
                self._euclid(positions[0], targets[p[0]])
                + self._euclid(positions[1], targets[p[1]])
                + self._euclid(positions[2], targets[p[2]])
            )
            if s < best:
                best = s
                best_perm = p
        return best, best_perm

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Current cube positions
        pos = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in pos], dtype=float)
        ys = np.array([p[1] for p in pos], dtype=float)

        # Identify apex as cube closest to robot: maximum X
        apex_idx = int(np.argmax(xs))
        base_indices = [i for i in range(3) if i != apex_idx]
        b0, b1 = base_indices[0], base_indices[1]

        apex_x, apex_y = float(xs[apex_idx]), float(ys[apex_idx])
        b0_x, b0_y = float(xs[b0]), float(ys[b0])
        b1_x, b1_y = float(xs[b1]), float(ys[b1])

        # Constraints
        # 1) All cubes on table (collision with table top)
        # Collision can be flaky right after reset; allow a Z-based fallback near calibrated rest height.
        table_collide = all(c.is_colliding(self.table.table_top) for c in self.cubes)
        z_ok = all(abs(float(p[2]) - float(self._CUBE_Z)) <= 0.03 for p in pos)
        all_on_table = bool(table_collide or z_ok)

        # 2) Base pair at similar depth (same X within tolerance)
        base_depth_ok = abs(b0_x - b1_x) <= self._BASE_DEPTH_TOL_M

        # 3) Base lateral separation along Y within bounds
        base_sep = abs(b0_y - b1_y)
        base_sep_ok = (base_sep >= self._BASE_MIN_LAT_SEP_M) and (base_sep <= self._BASE_MAX_LAT_SEP_M)

        # 4) Apex closer to robot than base by at least specified margin (X axis)
        base_x_mean = 0.5 * (b0_x + b1_x)
        apex_depth_ok = (apex_x - base_x_mean) >= self._APEX_CLOSER_BY_M

        # 5) Apex centered between base cubes along Y
        base_y_mean = 0.5 * (b0_y + b1_y)
        apex_center_ok = abs(apex_y - base_y_mean) <= self._APEX_CENTER_TOL_M

        positions_ok = bool(all_on_table and base_depth_ok and base_sep_ok and apex_depth_ok and apex_center_ok)

        # No semantic requirements (color permutation allowed)
        semantic_ok = True

        # Must be released
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoal staging (coarse: all true only when fully successful)
        if positions_ok and released:
            for i in range(6):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # - target_distance: arrangement constraint residuals
        # - pose_error: best assignment error to canonical goal vertices (order-free)
        pos_np = [np.array(p, dtype=float) for p in pos]
        min_assign_sum, _ = self._min_assignment_error(pos_np, self._goal_vertices)

        target_distance = {
            "base_depth_diff_m": abs(b0_x - b1_x),
            "base_lateral_sep_m": float(base_sep),
            "apex_minus_base_depth_m": float(apex_x - base_x_mean),
            "apex_centering_y_err_m": abs(apex_y - base_y_mean),
        }
        pose_error = {
            "min_sum_distance_to_goal_vertices_m": float(min_assign_sum),
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