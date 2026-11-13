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


class RedBlueSideSorting(RoboEvalEnv, MetricRolloutEval):
    """Sort red cubes to the left side and blue cubes to the right side of the workspace.

Success Criteria:
- Both red cubes are inside the left-side workspace region and both blue cubes are inside the right-side region.
- All four cubes retain their expected colors.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task workspace constraints (narrower than global workspace; used for success)
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _Z_MIN = 0.00  # region spec minimum; we additionally check near table height

    # Region definitions (left = +Y, right = -Y)
    _LEFT_REGION = {"x": (0.40, 0.70), "y": (0.00, 0.25), "z_min": 0.00}
    _RIGHT_REGION = {"x": (0.40, 0.70), "y": (-0.25, 0.00), "z_min": 0.00}

    _CUBE_REST_Z = 0.97
    _TABLE_Z = 0.95

    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # Create required assets (ONLY ColoredCube)
        self.red_cube_1 = ColoredCube(self._mojo)
        self.red_cube_2 = ColoredCube(self._mojo)
        self.blue_cube_1 = ColoredCube(self._mojo)
        self.blue_cube_2 = ColoredCube(self._mojo)

        self.cubes = [self.red_cube_1, self.red_cube_2, self.blue_cube_1, self.blue_cube_2]

        # REQUIRED: set_color() in _initialize_env
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

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

        # 2. Deterministic RNG (required pattern, even if goal-state is deterministic)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

        # 3. Randomize object poses in UNSOLVED positions (within workspace bounds)
        # Keep at least one cube guaranteed in the wrong region so _success() is False.
        x_lo, x_hi = self._X_BOUNDS
        y_lo, y_hi = self._Y_BOUNDS
        z = self._CUBE_REST_Z

        # Keep cubes away from exact boundaries used by _in_region (tol=0.014), plus a small margin.
        margin = 0.03
        x_lo_m, x_hi_m = x_lo + margin, x_hi - margin
        y_lo_m, y_hi_m = y_lo + margin, y_hi - margin

        # Simple rejection sampling for non-overlapping placements in XY.
        min_xy_sep = 0.06  # conservative spacing to avoid initial contacts
        placed_xy = []

        def sample_xy_in_rect(x0, x1, y0, y1):
            return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=float)

        def far_enough(xy):
            for p in placed_xy:
                if float(np.linalg.norm(xy - p)) < min_xy_sep:
                    return False
            return True

        def sample_pose_in_rect(x0, x1, y0, y1):
            for _ in range(500):
                xy = sample_xy_in_rect(x0, x1, y0, y1)
                if far_enough(xy):
                    placed_xy.append(xy)
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # (w,x,y,z)
                    quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)  # -> (x,y,z,w)
                    return np.array([xy[0], xy[1], z], dtype=float), quat
            # Fallback (should be extremely rare)
            xy = sample_xy_in_rect(x0, x1, y0, y1)
            placed_xy.append(xy)
            return np.array([xy[0], xy[1], z], dtype=float), np.array([0, 0, 0, 1], dtype=float)

        # Guarantee UNSOLVED: force one red cube to the RIGHT region (wrong side),
        # and sample all others randomly anywhere in the workspace.
        # Right region y in [-0.25, 0.00] (use margin away from 0 boundary).
        rr_y0, rr_y1 = self._RIGHT_REGION["y"]
        right_y0 = rr_y0 + margin
        right_y1 = rr_y1 - margin  # negative

        # Place red_cube_1 explicitly in wrong region (right side)
        pos, quat = sample_pose_in_rect(x_lo_m, x_hi_m, right_y0, right_y1)
        self.red_cube_1.set_pose(position=pos, quat=quat)

        # Place remaining cubes randomly in full workspace (still respecting spacing)
        pos, quat = sample_pose_in_rect(x_lo_m, x_hi_m, y_lo_m, y_hi_m)
        self.red_cube_2.set_pose(position=pos, quat=quat)

        pos, quat = sample_pose_in_rect(x_lo_m, x_hi_m, y_lo_m, y_hi_m)
        self.blue_cube_1.set_pose(position=pos, quat=quat)

        pos, quat = sample_pose_in_rect(x_lo_m, x_hi_m, y_lo_m, y_hi_m)
        self.blue_cube_2.set_pose(position=pos, quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _point_to_aabb_xy_distance(p_xy: np.ndarray, x_bounds, y_bounds) -> float:
        """Distance in XY plane from point to axis-aligned rectangle (0 if inside)."""
        px, py = float(p_xy[0]), float(p_xy[1])
        x0, x1 = float(x_bounds[0]), float(x_bounds[1])
        y0, y1 = float(y_bounds[0]), float(y_bounds[1])

        dx = 0.0
        if px < x0:
            dx = x0 - px
        elif px > x1:
            dx = px - x1

        dy = 0.0
        if py < y0:
            dy = y0 - py
        elif py > y1:
            dy = py - y1

        return float(np.sqrt(dx * dx + dy * dy))

    def _in_region(self, cube: ColoredCube, region: Dict, tol: float = 0.014) -> bool:
        pos = cube.body.get_position()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        in_x = (region["x"][0] - tol) <= x <= (region["x"][1] + tol)
        in_y = (region["y"][0] - tol) <= y <= (region["y"][1] + tol)
        above_min = z >= (region.get("z_min", 0.0) - tol)

        # Encourage "on the table" (not stacked): near rest height and colliding with table top
        near_table_height = abs(z - self._CUBE_REST_Z) <= 0.03
        on_table = cube.is_colliding(self.table.table_top)

        return bool(in_x and in_y and above_min and near_table_height and on_table)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Semantic checks (colors must match cube identities)
        semantic_ok = (
            self.red_cube_1.get_color() == "red"
            and self.red_cube_2.get_color() == "red"
            and self.blue_cube_1.get_color() == "blue"
            and self.blue_cube_2.get_color() == "blue"
        )

        # Position checks (region membership)
        red1_ok = self._in_region(self.red_cube_1, self._LEFT_REGION)
        red2_ok = self._in_region(self.red_cube_2, self._LEFT_REGION)
        blue1_ok = self._in_region(self.blue_cube_1, self._RIGHT_REGION)
        blue2_ok = self._in_region(self.blue_cube_2, self._RIGHT_REGION)
        positions_ok = bool(red1_ok and red2_ok and blue1_ok and blue2_ok)

        # Released check (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoal metrics: each cube correctly placed AND not held
        per_cube_released = [
            not any(self.robot.is_gripper_holding_object(obj, s) for s in self.robot.grippers)
            for obj in [self.red_cube_1, self.red_cube_2, self.blue_cube_1, self.blue_cube_2]
        ]
        self._metric_stage(0, bool(red1_ok and per_cube_released[0]))
        self._metric_stage(1, bool(red2_ok and per_cube_released[1]))
        self._metric_stage(2, bool(blue1_ok and per_cube_released[2]))
        self._metric_stage(3, bool(blue2_ok and per_cube_released[3]))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: region distances (0 if inside), plus z error to table rest height
        def region_dist(cube: ColoredCube, region: Dict) -> float:
            p = cube.body.get_position()
            return self._point_to_aabb_xy_distance(p[:2], region["x"], region["y"])

        target_distance = {
            "red_cube_1_to_left_region_xy": region_dist(self.red_cube_1, self._LEFT_REGION),
            "red_cube_2_to_left_region_xy": region_dist(self.red_cube_2, self._LEFT_REGION),
            "blue_cube_1_to_right_region_xy": region_dist(self.blue_cube_1, self._RIGHT_REGION),
            "blue_cube_2_to_right_region_xy": region_dist(self.blue_cube_2, self._RIGHT_REGION),
        }
        pose_error = {
            "red_cube_1_z_abs_err": abs(float(self.red_cube_1.body.get_position()[2]) - self._CUBE_REST_Z),
            "red_cube_2_z_abs_err": abs(float(self.red_cube_2.body.get_position()[2]) - self._CUBE_REST_Z),
            "blue_cube_1_z_abs_err": abs(float(self.blue_cube_1.body.get_position()[2]) - self._CUBE_REST_Z),
            "blue_cube_2_z_abs_err": abs(float(self.blue_cube_2.body.get_position()[2]) - self._CUBE_REST_Z),
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