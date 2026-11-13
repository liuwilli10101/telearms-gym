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

class GreenBlueRedCubeStacking(RoboEvalEnv, MetricRolloutEval):
    """Build a three-cube tower ordered green, blue, red from bottom to top.

Success Criteria:
- A vertical tower is built with green on the table, blue on green, and red on blue.
- All cube colors match green, blue, red, and the tower XY centers are aligned within tolerance.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace constraints (documented; randomization handled by pipeline)
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    # Geometry constants for ColoredCube (4cm)
    _BASE_Z = 0.97
    _DZ = 0.04  # stacking increment
    _POS_TOL = 0.014  # success tolerance reference

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY allowed asset: ColoredCube)
        self.green_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.red_cube = ColoredCube(self._mojo)

        # REQUIRED: call set_color() in _initialize_env
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

        self.cubes = [self.green_cube, self.blue_cube, self.red_cube]

        # Nominal goal poses (single vertical column)
        self._goal_xy = np.array([0.55, 0.00])
        self._goal_positions = {
            "green": np.array([self._goal_xy[0], self._goal_xy[1], self._BASE_Z]),
            "blue": np.array([self._goal_xy[0], self._goal_xy[1], self._BASE_Z + self._DZ]),
            "red": np.array([self._goal_xy[0], self._goal_xy[1], self._BASE_Z + 2 * self._DZ]),
        }
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
        # REQUIRED: call set_color() in _on_reset
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization object (pipeline may use seed later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM unsolved positions within required workspace bounds.
        #    Keep Z base fixed at 0.97; use random yaw to avoid perfect alignment.
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._BASE_Z

        # Ensure cubes are not stacked/aligned at reset (so _success() is False)
        # by enforcing a minimum XY separation greater than the success tolerance.
        min_sep = max(0.06, 4.0 * self._POS_TOL)  # > xy_tol, conservative

        def sample_xy(existing: list[np.ndarray]) -> np.ndarray:
            for _ in range(200):
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                if all(float(np.linalg.norm(p - q)) >= min_sep for q in existing):
                    return p
            # Fallback: deterministic spaced points (still within bounds)
            base = np.array([0.45, -0.15], dtype=float)
            offsets = [np.array([0.0, 0.0]), np.array([0.12, 0.0]), np.array([0.0, 0.12])]
            for off in offsets:
                p = base + off
                p[0] = float(np.clip(p[0], x_lo, x_hi))
                p[1] = float(np.clip(p[1], y_lo, y_hi))
                if all(float(np.linalg.norm(p - q)) >= min_sep for q in existing):
                    return p
            return base

        xys: list[np.ndarray] = []
        g_xy = sample_xy(xys); xys.append(g_xy)
        b_xy = sample_xy(xys); xys.append(b_xy)
        r_xy = sample_xy(xys); xys.append(r_xy)

        def yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        self.green_cube.set_pose(position=np.array([g_xy[0], g_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        self.blue_cube.set_pose(position=np.array([b_xy[0], b_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        self.red_cube.set_pose(position=np.array([r_xy[0], r_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        g = self.green_cube
        b = self.blue_cube
        r = self.red_cube

        # Color verification (task is color-conditioned stacking)
        colors_ok = (g.get_color() == "green" and b.get_color() == "blue" and r.get_color() == "red")

        # Positions / relations
        gpos = g.body.get_position()
        bpos = b.body.get_position()
        rpos = r.body.get_position()

        # "On-top-of" via collision + table-contact exclusion for upper cubes
        blue_on_green = (
            b.is_colliding(g)
            and (not b.is_colliding(self.table.table_top))
        )
        red_on_blue = (
            r.is_colliding(b)
            and (not r.is_colliding(self.table.table_top))
        )

        # Base should be on the table
        green_on_table = g.is_colliding(self.table.table_top)

        # "Aligned vertical column": same (x,y) within tolerance
        # Use a slightly relaxed XY tolerance while keeping success strict enough.
        xy_tol = self._POS_TOL
        aligned_xy = (
            self._xy_dist(gpos, bpos) < xy_tol
            and self._xy_dist(bpos, rpos) < xy_tol
            and self._xy_dist(gpos, rpos) < xy_tol
        )

        positions_ok = bool(green_on_table and blue_on_green and red_on_blue and aligned_xy)

        # Released gripper(s)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics (mark all as complete only when full goal is met)
        if colors_ok and positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(colors_ok and positions_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        target_distance = {
            "green_to_goal": float(np.linalg.norm(gpos - self._goal_positions["green"])),
            "blue_to_goal": float(np.linalg.norm(bpos - self._goal_positions["blue"])),
            "red_to_goal": float(np.linalg.norm(rpos - self._goal_positions["red"])),
            "blue_to_green_xy": self._xy_dist(bpos, gpos),
            "red_to_blue_xy": self._xy_dist(rpos, bpos),
        }
        pose_error = {
            "green_pos_err": target_distance["green_to_goal"],
            "blue_pos_err": target_distance["blue_to_goal"],
            "red_pos_err": target_distance["red_to_goal"],
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})