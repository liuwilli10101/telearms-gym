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

class YellowBlueRedTowerStacking(RoboEvalEnv, MetricRolloutEval):
    """Build a yellow-blue-red tower from bottom to top.

Success Criteria:
- A vertical tower is built with yellow on the table, blue on yellow, and red on blue.
- Blue and red are not touching the table, each layer has valid XY and Z alignment, and colors match yellow/blue/red.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task tolerances (meters)
    _XY_TOL = 0.014
    _Z_TOL = 0.014
    _LAYER_DZ = 0.04
    _BASE_Z = 0.97

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset type)
        self.yellow_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.red_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color in _initialize_env()
        self.yellow_cube.set_color("yellow")
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

        self.cubes = [self.yellow_cube, self.blue_cube, self.red_cube]

        # Metrics (full signature REQUIRED)
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

        # 2) Deterministic RNG handle (pipeline may add perturbations later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color in _on_reset()
        self.yellow_cube.set_color("yellow")
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

        # 3) Randomize UNSOLVED initial placements within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._BASE_Z)

        # Keep cubes sufficiently separated so they are not accidentally stacked/aligned
        # (make sure _success() conditions fail after reset)
        min_xy_sep = 0.08  # comfortably > _XY_TOL and typical cube contact region

        def sample_xy(existing: list[np.ndarray]) -> np.ndarray:
            for _ in range(500):
                xy = np.array(
                    [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
                )
                if all(float(np.linalg.norm(xy - e)) >= min_xy_sep for e in existing):
                    return xy
            # Fallback: deterministic spaced points (still within bounds)
            base = np.array([0.45, -0.15], dtype=float)
            step = np.array([0.10, 0.15], dtype=float)
            for i in range(3):
                cand = base + i * step
                cand[0] = float(np.clip(cand[0], x_min, x_max))
                cand[1] = float(np.clip(cand[1], y_min, y_max))
                if all(float(np.linalg.norm(cand - e)) >= min_xy_sep for e in existing):
                    return cand
            return np.array([0.45, 0.0], dtype=float)

        xys: list[np.ndarray] = []
        y_xy = sample_xy(xys); xys.append(y_xy)
        b_xy = sample_xy(xys); xys.append(b_xy)
        r_xy = sample_xy(xys); xys.append(r_xy)

        def yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        self.yellow_cube.set_pose(position=np.array([y_xy[0], y_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        self.blue_cube.set_pose(position=np.array([b_xy[0], b_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        self.red_cube.set_pose(position=np.array([r_xy[0], r_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(10):
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
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        y_pos = self.yellow_cube.body.get_position()
        b_pos = self.blue_cube.body.get_position()
        r_pos = self.red_cube.body.get_position()

        # Spatial checks for "on" relations:
        # - collision indicates contact
        # - XY alignment and Z offset ensure "stacked" rather than lateral contact
        yellow_on_table = self.yellow_cube.is_colliding(self.table.table_top)

        blue_collide_yellow = self.blue_cube.is_colliding(self.yellow_cube)
        red_collide_blue = self.red_cube.is_colliding(self.blue_cube)

        blue_not_on_table = not self.blue_cube.is_colliding(self.table.table_top)
        red_not_on_table = not self.red_cube.is_colliding(self.table.table_top)

        blue_xy_ok = self._xy_dist(b_pos, y_pos) <= self._XY_TOL
        red_xy_ok = self._xy_dist(r_pos, b_pos) <= self._XY_TOL

        blue_z_ok = abs((b_pos[2] - y_pos[2]) - self._LAYER_DZ) <= self._Z_TOL
        red_z_ok = abs((r_pos[2] - b_pos[2]) - self._LAYER_DZ) <= self._Z_TOL

        positions_ok = bool(
            yellow_on_table
            and blue_collide_yellow
            and red_collide_blue
            and blue_not_on_table
            and red_not_on_table
            and blue_xy_ok
            and red_xy_ok
            and blue_z_ok
            and red_z_ok
        )

        # No semantic content beyond color; colors are fixed via set_color()
        semantic_ok = bool(
            self.yellow_cube.get_color() == "yellow"
            and self.blue_cube.get_color() == "blue"
            and self.red_cube.get_color() == "red"
        )

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (coarse mapping to observable conditions)
        if yellow_on_table:
            for i in (0, 1, 2):
                self._metric_stage(i, True)
        if yellow_on_table and blue_collide_yellow and blue_not_on_table and blue_xy_ok and blue_z_ok:
            for i in (3, 4, 5, 6):
                self._metric_stage(i, True)
        if positions_ok:
            for i in (7, 8, 9):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "blue_to_yellow_3d": float(distance(self.blue_cube.body, self.yellow_cube.body)),
                "red_to_blue_3d": float(distance(self.red_cube.body, self.blue_cube.body)),
                "blue_to_yellow_xy": float(self._xy_dist(b_pos, y_pos)),
                "red_to_blue_xy": float(self._xy_dist(r_pos, b_pos)),
            },
            pose_error={
                "blue_minus_yellow_dz_err": float(abs((b_pos[2] - y_pos[2]) - self._LAYER_DZ)),
                "red_minus_blue_dz_err": float(abs((r_pos[2] - b_pos[2]) - self._LAYER_DZ)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})