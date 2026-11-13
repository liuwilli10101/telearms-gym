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

class ThreeCubePyramidStacking(RoboEvalEnv, MetricRolloutEval):
    """
    Build a small 3-cube pyramid: place the green and blue cubes side-by-side as the base,
    then stack the red cube centered on top of them (bridging both base cubes).

    Sequential Sub-Goals:
        1. Grasp one base cube (green or blue).
        2. Move the grasped base cube to a chosen base location on the table.
        3. Place and release the first base cube on the table.
        4. Grasp the other base cube (blue or green).
        5. Align the second base cube adjacent to the first so they touch and form a stable base.
        6. Place and release the second base cube adjacent to the first.
        7. Grasp the red cube for the top placement.
        8. Lift the red cube above the base height to clear the two base cubes.
        9. Center the red cube over the seam between the green and blue cubes to bridge them.
        10. Place and release the red cube on top, forming the pyramid.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry constants for ColoredCube (4cm)
    _CUBE_SIZE = 0.04
    _BASE_Z = 0.97
    _TOP_Z = 1.01  # BASE_Z + 0.04
    _ADJ_CENTER_DIST = 0.04  # touching, center-to-center
    _TOL = 0.014  # success tolerance guideline

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY allowed asset: ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.green_cube, self.blue_cube]

        # Metrics (FULL signature REQUIRED)
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

        # 2. Deterministic RNG (may be used for deterministic variants that still satisfy goal)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        # 3. Place objects in RANDOM (unsolved) configuration within workspace bounds.
        #    Keep all cubes on the table (Z = BASE_Z) so top condition fails and _success() is False.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._BASE_Z

        # Sample 3 non-overlapping-ish XY points (avoid accidental stacking/bridging)
        min_xy_sep = 2.2 * self._CUBE_SIZE  # ~0.088m
        pts = []
        for _ in range(2000):
            cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if all(np.linalg.norm(cand - p) >= min_xy_sep for p in pts):
                pts.append(cand)
            if len(pts) == 3:
                break
        if len(pts) < 3:
            # Fallback: fixed spread positions (still unsolved), within bounds
            pts = [
                np.array([0.45, -0.18], dtype=float),
                np.array([0.62, 0.15], dtype=float),
                np.array([0.55, 0.00], dtype=float),
            ]

        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            # return as (x, y, z, w)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        # Assign random points to cubes
        r_xy, g_xy, b_xy = pts[0], pts[1], pts[2]
        r_pos = np.array([r_xy[0], r_xy[1], z], dtype=float)
        g_pos = np.array([g_xy[0], g_xy[1], z], dtype=float)
        b_pos = np.array([b_xy[0], b_xy[1], z], dtype=float)

        # Place (all on table, random yaw); layer-by-layer for stability
        self.green_cube.set_pose(position=g_pos, quat=yaw_quat())
        self.blue_cube.set_pose(position=b_pos, quat=yaw_quat())
        self._mojo.physics.forward()

        self.red_cube.set_pose(position=r_pos, quat=yaw_quat())
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Semantic/color verification (ColoredCube) ---
        colors_ok = (
            self.red_cube.get_color() == "red"
            and self.green_cube.get_color() == "green"
            and self.blue_cube.get_color() == "blue"
        )

        # --- Spatial checks (relative arrangement) ---
        r_pos = self.red_cube.body.get_position()
        g_pos = self.green_cube.body.get_position()
        b_pos = self.blue_cube.body.get_position()

        # Base: adjacent/touching on the table
        gb_xy_dist = float(np.linalg.norm((g_pos - b_pos)[:2]))
        gb_touching = self.green_cube.is_colliding(self.blue_cube) or (
            abs(gb_xy_dist - self._ADJ_CENTER_DIST) <= self._TOL
        )

        base_on_table = self.green_cube.is_colliding(self.table.table_top) and self.blue_cube.is_colliding(
            self.table.table_top
        )
        base_height_ok = (abs(float(g_pos[2]) - self._BASE_Z) <= self._TOL) and (
            abs(float(b_pos[2]) - self._BASE_Z) <= self._TOL
        )

        base_ok = bool(gb_touching and base_on_table and base_height_ok)

        # Top: red bridges both green and blue, centered over the seam, not touching table
        seam_xy = 0.5 * (g_pos[:2] + b_pos[:2])
        red_centered = float(np.linalg.norm(r_pos[:2] - seam_xy)) <= self._TOL

        red_on_green = self.red_cube.is_colliding(self.green_cube)
        red_on_blue = self.red_cube.is_colliding(self.blue_cube)
        red_not_on_table = not self.red_cube.is_colliding(self.table.table_top)

        # Height heuristic: red should be above base layer by about one cube height
        expected_top_z = float(max(g_pos[2], b_pos[2]) + self._CUBE_SIZE)
        red_height_ok = abs(float(r_pos[2]) - expected_top_z) <= self._TOL
        red_above_base = float(r_pos[2]) >= float(max(g_pos[2], b_pos[2]) + 0.5 * self._CUBE_SIZE)

        top_ok = bool(red_centered and red_on_green and red_on_blue and red_not_on_table and red_height_ok and red_above_base)

        positions_ok = bool(base_ok and top_ok)

        # Released condition
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoal staging (state-based approximations)
        if base_ok:
            # Base-related subgoals (1-6) considered achieved once base is formed and released
            for i in range(0, 6):
                self._metric_stage(i, True)
            self._metric_stage(6, True)  # "grasp red" proxy unlocked by having base

        if top_ok:
            for i in range(7, self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and colors_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        target_distance = {
            "base_center_distance_error": abs(gb_xy_dist - self._ADJ_CENTER_DIST),
            "red_to_seam_xy": float(np.linalg.norm(r_pos[:2] - seam_xy)),
        }
        pose_error = {
            "green_z_error": abs(float(g_pos[2]) - self._BASE_Z),
            "blue_z_error": abs(float(b_pos[2]) - self._BASE_Z),
            "red_z_error": abs(float(r_pos[2]) - expected_top_z),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})