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

class TriangleCubeArrangementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Reposition three colored cubes on the tabletop so their centers form an equilateral triangle
    of a specified side length. The task is successful when all three cubes lie on the table
    surface and satisfy the triangle distance constraints within tolerance.

    Sequential Sub-Goals:
        1. Grasp any one cube to begin forming the triangle.
        2. Move a grasped cube toward a planned triangle vertex location.
        3. Place the cube back on the table at the intended vertex.
        4. Grasp another cube to complete additional triangle vertices.
        5. Move a grasped cube toward a planned triangle vertex location.
        6. Place the cube back on the table at the intended vertex.
        7. Fine-adjust cube positions so all three pairwise distances match the equilateral triangle constraint.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from spec)
    _SIDE_LEN_M = 0.12
    _SIDE_TOL_M = 0.02
    _ON_TABLE_Z_TOL_M = 0.015
    _CUBE_REST_Z = 0.97

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY allowed asset: ColoredCube)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]

        # REQUIRED: set_color() in _initialize_env
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # Precompute a deterministic goal triangle (centroid at workspace center)
        cx, cy = 0.55, 0.0
        s = float(self._SIDE_LEN_M)
        h = s / np.sqrt(3.0)  # centroid-to-vertex distance

        # Vertices (equilateral triangle in XY plane)
        self._goal_positions = {
            "cube_red": np.array([cx, cy + h, self._CUBE_REST_Z]),
            "cube_green": np.array([cx + 0.5 * s, cy - 0.5 * h, self._CUBE_REST_Z]),
            "cube_blue": np.array([cx - 0.5 * s, cy - 0.5 * h, self._CUBE_REST_Z]),
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

        # 2. Deterministic RNG (required pattern; randomization added by validation pipeline)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 4. Place objects in RANDOM UNSOLVED positions within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_REST_Z)

        s = float(self._SIDE_LEN_M)
        tol = float(self._SIDE_TOL_M)

        def _sample_xy() -> np.ndarray:
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def _yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        def _tri_ok(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
            d12 = float(np.linalg.norm(p1 - p2))
            d23 = float(np.linalg.norm(p2 - p3))
            d31 = float(np.linalg.norm(p3 - p1))
            return (abs(d12 - s) <= tol) and (abs(d23 - s) <= tol) and (abs(d31 - s) <= tol)

        # Rejection sample: keep cubes separated and explicitly NOT in goal triangle distances
        min_sep = 0.06  # avoid overlaps/stacking/near-collisions
        for _ in range(5000):
            pr_xy = _sample_xy()
            pg_xy = _sample_xy()
            pb_xy = _sample_xy()

            if (
                np.linalg.norm(pr_xy - pg_xy) < min_sep
                or np.linalg.norm(pg_xy - pb_xy) < min_sep
                or np.linalg.norm(pb_xy - pr_xy) < min_sep
            ):
                continue

            if _tri_ok(pr_xy, pg_xy, pb_xy):
                continue

            pr = np.array([pr_xy[0], pr_xy[1], z], dtype=float)
            pg = np.array([pg_xy[0], pg_xy[1], z], dtype=float)
            pb = np.array([pb_xy[0], pb_xy[1], z], dtype=float)
            break
        else:
            # Fallback: guaranteed-unsolved (collinear-ish) placement within bounds
            pr = np.array([0.44, -0.20, z], dtype=float)
            pg = np.array([0.66, -0.20, z], dtype=float)
            pb = np.array([0.55, 0.22, z], dtype=float)

        self.cube_red.set_pose(position=pr, quat=_yaw_quat())
        self.cube_green.set_pose(position=pg, quat=_yaw_quat())
        self.cube_blue.set_pose(position=pb, quat=_yaw_quat())

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(p: np.ndarray, q: np.ndarray) -> float:
        d = p[:2] - q[:2]
        return float(np.linalg.norm(d))

    def _is_on_table(self, cube: ColoredCube) -> bool:
        # Collision with table top + z-height close to rest height
        p = cube.body.get_position()
        if abs(float(p[2]) - float(self._CUBE_REST_Z)) > float(self._ON_TABLE_Z_TOL_M):
            return False
        return bool(cube.is_colliding(self.table.table_top))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Released check (supports single- or multi-gripper robots)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # All cubes resting on table (not stacked)
        on_table_ok = (
            self._is_on_table(self.cube_red)
            and self._is_on_table(self.cube_green)
            and self._is_on_table(self.cube_blue)
        )

        # Pairwise equilateral constraints in the table plane (XY)
        pr = self.cube_red.body.get_position()
        pg = self.cube_green.body.get_position()
        pb = self.cube_blue.body.get_position()

        d_rg = self._xy_dist(pr, pg)
        d_gb = self._xy_dist(pg, pb)
        d_br = self._xy_dist(pb, pr)

        s = float(self._SIDE_LEN_M)
        tol = float(self._SIDE_TOL_M)
        tri_ok = (
            abs(d_rg - s) <= tol
            and abs(d_gb - s) <= tol
            and abs(d_br - s) <= tol
        )

        positions_ok = bool(on_table_ok and tri_ok)
        semantic_ok = True  # no semantic constraints beyond color initialization

        # Subgoals: mark all complete once final arrangement is correct and released
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "side_length_m": s,
                "tol_m": tol,
            },
            pose_error={
                "d_rg_err_m": float(d_rg - s),
                "d_gb_err_m": float(d_gb - s),
                "d_br_err_m": float(d_br - s),
                "red_z_err_m": float(pr[2] - self._CUBE_REST_Z),
                "green_z_err_m": float(pg[2] - self._CUBE_REST_Z),
                "blue_z_err_m": float(pb[2] - self._CUBE_REST_Z),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})