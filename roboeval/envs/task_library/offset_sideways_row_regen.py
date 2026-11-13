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


class DiagonalCubeRowArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a diagonal row such that each successive cube is
    placed farther back (+X) and 4 cm to the left (+Y) of the previous cube.

    Sequential Sub-Goals:
        1. Grasp the first cube.
        2. Place cube 1 as the front/right-most anchor.
        3. Place cube 2 behind (+X) and left (+Y) of cube 1.
        4. Place cube 3 behind (+X) and left (+Y) of cube 2.
        5. Fine-adjust spacing into an evenly stepped diagonal row.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task geometry (meters) ---
    _REST_Z = 0.97
    _STEP_XY = np.array([0.06, 0.04], dtype=float)  # +X back, +Y left
    _TOL_XYZ = np.array([0.02, 0.01, 0.03], dtype=float)

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube is allowed)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3]

        # REQUIRED: set_color() in _initialize_env
        self.cube_1.set_color("red")
        self.cube_2.set_color("green")
        self.cube_3.set_color("blue")

        # Cache goal poses (used by _success metrics and optional samplers)
        anchor = np.array([0.50, -0.08, self._REST_Z], dtype=float)
        self._goal_positions = {
            "cube_1": anchor,
            "cube_2": anchor + np.array([self._STEP_XY[0], self._STEP_XY[1], 0.0]),
            "cube_3": anchor + np.array([2 * self._STEP_XY[0], 2 * self._STEP_XY[1], 0.0]),
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

    def sample_initial_object_poses(self):
        """
        Returns randomized, non-goal initial poses for external randomization pipelines.
        NOTE: Per spec, _on_reset() places objects in the goal state (solved) deterministically.
        """
        rng = np.random.default_rng(self.seed)

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        goal_ps = np.stack([self._goal_positions["cube_1"],
                            self._goal_positions["cube_2"],
                            self._goal_positions["cube_3"]], axis=0)[:, :2]

        poses = []
        min_sep = 0.07  # keep cubes from starting in collision
        min_goal_clear = 0.08

        attempts = 0
        while len(poses) < 3 and attempts < 500:
            attempts += 1
            p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

            # Keep away from goal positions
            if np.min(np.linalg.norm(goal_ps - p[None, :], axis=1)) < min_goal_clear:
                continue

            # Keep separated from previously sampled
            if any(np.linalg.norm(p - q[:2]) < min_sep for q in poses):
                continue

            poses.append(np.array([p[0], p[1], self._REST_Z], dtype=float))

        # Fallback (rare): spread them deterministically
        if len(poses) < 3:
            poses = [
                np.array([0.62, -0.18, self._REST_Z], dtype=float),
                np.array([0.60, 0.00, self._REST_Z], dtype=float),
                np.array([0.44, 0.18, self._REST_Z], dtype=float),
            ]

        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return [(poses[i], quat.copy()) for i in range(3)]

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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        self.cube_1.set_color("red")
        self.cube_2.set_color("green")
        self.cube_3.set_color("blue")

        # 3. Place objects in RANDOM UNSOLVED positions (within workspace bounds)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._REST_Z)

        goal_ps_xy = np.stack(
            [self._goal_positions["cube_1"], self._goal_positions["cube_2"], self._goal_positions["cube_3"]],
            axis=0,
        )[:, :2]

        exp_xy = self._STEP_XY.astype(float).copy()
        tol_xy = self._TOL_XYZ[:2].astype(float).copy()

        def _is_solved_xy(p1, p2, p3) -> bool:
            d21 = (p2 - p1)
            d32 = (p3 - p2)
            ok21 = (abs(d21[0] - exp_xy[0]) <= tol_xy[0]) and (abs(d21[1] - exp_xy[1]) <= tol_xy[1])
            ok32 = (abs(d32[0] - exp_xy[0]) <= tol_xy[0]) and (abs(d32[1] - exp_xy[1]) <= tol_xy[1])
            return bool(ok21 and ok32)

        poses_xy = []
        min_sep = 0.07          # reduce initial collisions
        min_goal_clear = 0.08   # keep away from goal positions

        # Rejection sample until arrangement is NOT solved
        for _ in range(800):
            poses_xy.clear()
            attempts = 0
            while len(poses_xy) < 3 and attempts < 500:
                attempts += 1
                p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

                if np.min(np.linalg.norm(goal_ps_xy - p[None, :], axis=1)) < min_goal_clear:
                    continue
                if any(np.linalg.norm(p - q) < min_sep for q in poses_xy):
                    continue

                poses_xy.append(p)

            if len(poses_xy) < 3:
                # Fallback
                poses_xy = [
                    np.array([0.62, -0.18], dtype=float),
                    np.array([0.60, 0.00], dtype=float),
                    np.array([0.44, 0.18], dtype=float),
                ]

            if not _is_solved_xy(poses_xy[0], poses_xy[1], poses_xy[2]):
                break

        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.cube_1.set_pose(position=np.array([poses_xy[0][0], poses_xy[0][1], z], dtype=float), quat=quat)
        self.cube_2.set_pose(position=np.array([poses_xy[1][0], poses_xy[1][1], z], dtype=float), quat=quat)
        self.cube_3.set_pose(position=np.array([poses_xy[2][0], poses_xy[2][1], z], dtype=float), quat=quat)

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

        p1 = self.cube_1.body.get_position()
        p2 = self.cube_2.body.get_position()
        p3 = self.cube_3.body.get_position()

        d21 = p2 - p1
        d32 = p3 - p2

        exp = np.array([self._STEP_XY[0], self._STEP_XY[1], 0.0], dtype=float)
        tol = self._TOL_XYZ

        def _offset_ok(d):
            return (
                abs(d[0] - exp[0]) <= tol[0]
                and abs(d[1] - exp[1]) <= tol[1]
                and abs(d[2] - exp[2]) <= tol[2]
            )

        # Relative offset constraints (cube2 from cube1, cube3 from cube2)
        rel_ok = _offset_ok(d21) and _offset_ok(d32)

        # Arrangement consistency: same XY step twice (also rejects swapped ordering)
        step12_xy = d21[:2]
        step23_xy = d32[:2]
        arr_ok = (
            abs(step12_xy[0] - self._STEP_XY[0]) <= tol[0]
            and abs(step12_xy[1] - self._STEP_XY[1]) <= tol[1]
            and abs(step23_xy[0] - self._STEP_XY[0]) <= tol[0]
            and abs(step23_xy[1] - self._STEP_XY[1]) <= tol[1]
        )

        # On-table (not stacked / not floating): collide with table top and near rest Z
        on_table = True
        for c in self.cubes:
            pos = c.body.get_position()
            if abs(pos[2] - self._REST_Z) > tol[2]:
                on_table = False
                break
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break

        positions_ok = bool(rel_ok and arr_ok and on_table)

        # No semantic checks for this spatial task
        semantic_ok = True

        # Released gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics (mark all stages when arrangement is correct)
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # Distances of achieved relative offsets to targets (XYZ)
        err21 = np.linalg.norm((d21 - exp) * np.array([1.0, 1.0, 1.0]))
        err32 = np.linalg.norm((d32 - exp) * np.array([1.0, 1.0, 1.0]))
        # XY-only step consistency error between the two steps
        step_consistency_xy = np.linalg.norm((step12_xy - step23_xy))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube2_from_cube1_offset_error_xyz": float(err21),
                "cube3_from_cube2_offset_error_xyz": float(err32),
                "step_consistency_error_xy": float(step_consistency_xy),
            },
            pose_error={
                "cube1_z_error": float(abs(p1[2] - self._REST_Z)),
                "cube2_z_error": float(abs(p2[2] - self._REST_Z)),
                "cube3_z_error": float(abs(p3[2] - self._REST_Z)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})