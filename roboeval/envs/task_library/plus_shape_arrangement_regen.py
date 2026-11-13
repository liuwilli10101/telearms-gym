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

class PlusSignCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange five colored cubes on the table into a plus (+) sign: one center cube with four cubes
    placed front (-X), back (+X), left (+Y), and right (-Y) of it at equal spacing.

    Sequential Sub-Goals:
        0. Grasp the cube designated as the center cube.
        1. Place the center cube at a stable reference location within the workspace.
        2. Move a cube to the FRONT (-X / decreasing x) position relative to the center.
        3. Place the front cube aligned with the center in y, offset by one spacing in -X.
        4. Move a cube to the BACK (+X / increasing x) position relative to the center.
        5. Place the back cube aligned with the center in y, offset by one spacing in +X.
        6. Move a cube to the LEFT (+Y / increasing y) position relative to the center.
        7. Place the left cube aligned with the center in x, offset by one spacing in +Y.
        8. Move a cube to the RIGHT (-Y / decreasing y) position relative to the center.
        9. Place the right cube aligned with the center in x, offset by one spacing in -Y.
        10. Fine-align the four neighbor cubes so they form a clean plus (+) with equal spacing.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task geometry (per spec)
    _CENTER_POS = np.array([0.55, 0.00, 0.97])
    _SPACING_M = 0.06

    # Use the stricter tolerance noted in task constraints for success checking.
    _TOL_M = 0.014

    _NUM_SUBGOALS = 11

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.cube_center = ColoredCube(self._mojo)
        self.cube_front = ColoredCube(self._mojo)
        self.cube_back = ColoredCube(self._mojo)
        self.cube_left = ColoredCube(self._mojo)
        self.cube_right = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_center,
            self.cube_front,
            self.cube_back,
            self.cube_left,
            self.cube_right,
        ]

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Precompute goal poses (quat identity; orientation irrelevant for this task)
        cx, cy, cz = self._CENTER_POS
        s = self._SPACING_M
        self._goal_positions = {
            "center": np.array([cx, cy, cz]),
            "front": np.array([cx - s, cy, cz]),   # -X
            "back": np.array([cx + s, cy, cz]),    # +X
            "left": np.array([cx, cy + s, cz]),    # +Y
            "right": np.array([cx, cy - s, cz]),   # -Y
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

    def _sample_objects(self):
        """
        Kept for compatibility with pipelines that may call it.
        This task's reset places objects in the goal state (the validation pipeline
        will apply randomization later).
        """
        return

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

        # 3. Set colors (required)
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # 4. Randomize initial poses (UNSOLVED) within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97
        quat = self._goal_quat

        tol = float(self._TOL_M)
        min_sep = 0.045  # simple anti-overlap heuristic for ~5cm cubes

        def sample_point():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z_base], dtype=float)

        def far_enough(p, existing):
            for q in existing:
                if np.linalg.norm(p[:2] - q[:2]) < min_sep:
                    return False
            return True

        def plus_solved(center, front, back, left, right):
            s = float(self._SPACING_M)
            def ok(p, dx, dy):
                d = p - center
                return (abs(d[0] - dx) <= tol) and (abs(d[1] - dy) <= tol)
            return (
                ok(front, -s, 0.0)
                and ok(back, +s, 0.0)
                and ok(left, 0.0, +s)
                and ok(right, 0.0, -s)
            )

        # Rejection-sample until NOT in solved plus configuration
        for _ in range(500):
            pts = []
            while len(pts) < 5:
                p = sample_point()
                if far_enough(p, pts):
                    pts.append(p)

            p_center, p_front, p_back, p_left, p_right = pts
            if not plus_solved(p_center, p_front, p_back, p_left, p_right):
                break
        else:
            # Fallback: force a violation by moving one cube away from the required offset
            p_center = np.array([0.55, 0.00, z_base], dtype=float)
            p_front = np.array([0.48, 0.18, z_base], dtype=float)
            p_back = np.array([0.66, -0.18, z_base], dtype=float)
            p_left = np.array([0.60, 0.22, z_base], dtype=float)
            p_right = np.array([0.44, -0.22, z_base], dtype=float)

        self.cube_center.set_pose(position=p_center, quat=quat)
        self.cube_front.set_pose(position=p_front, quat=quat)
        self.cube_back.set_pose(position=p_back, quat=quat)
        self.cube_left.set_pose(position=p_left, quat=quat)
        self.cube_right.set_pose(position=p_right, quat=quat)

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_on_table(self, cube: ColoredCube) -> bool:
        # Table collision direction is reliable using table.table_top
        if not hasattr(self, "table") or self.table is None:
            return False
        return cube.is_colliding(self.table.table_top)

    def _offset_ok(self, subject: ColoredCube, target: ColoredCube, dx: float, dy: float, tol: float) -> bool:
        ps = subject.body.get_position()
        pt = target.body.get_position()
        d = ps - pt
        return (abs(d[0] - dx) <= tol) and (abs(d[1] - dy) <= tol)

    def _compute_reward(self) -> float:
        # Dense shaping: negative sum of L2 position errors to their designated goal locations.
        # Success bonus handled by _success().
        if not hasattr(self, "cubes"):
            return 0.0

        goals = [
            self._goal_positions["center"],
            self._goal_positions["front"],
            self._goal_positions["back"],
            self._goal_positions["left"],
            self._goal_positions["right"],
        ]
        err = 0.0
        for cube, g in zip(self.cubes, goals):
            p = cube.body.get_position()
            err += float(np.linalg.norm(p - g))

        # Small penalty for not being on the table
        table_pen = 0.0
        for cube in self.cubes:
            if not self._is_on_table(cube):
                table_pen += 0.1

        # Convert to reward (higher is better)
        return -(err + table_pen)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        tol = float(self._TOL_M)

        # Table contact required for all cubes
        table_ok = all(self._is_on_table(c) for c in self.cubes)

        # Relative offset checks (task semantics: -X=front, +X=back, +Y=left, -Y=right)
        front_ok = self._offset_ok(self.cube_front, self.cube_center, dx=-self._SPACING_M, dy=0.0, tol=tol)
        back_ok = self._offset_ok(self.cube_back, self.cube_center, dx=+self._SPACING_M, dy=0.0, tol=tol)
        left_ok = self._offset_ok(self.cube_left, self.cube_center, dx=0.0, dy=+self._SPACING_M, tol=tol)
        right_ok = self._offset_ok(self.cube_right, self.cube_center, dx=0.0, dy=-self._SPACING_M, tol=tol)

        # Plus-pattern existence (compositional): all four neighbors must match the center offsets
        positions_ok = bool(table_ok and front_ok and back_ok and left_ok and right_ok)

        # No semantic constraints beyond spatial arrangement
        semantic_ok = True

        # Must be released (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Update subgoal metrics (best-effort mapping to subgoals)
        # 0 grasp center (proxy): center is at correct location OR the whole arrangement is correct
        self._metric_stage(0, bool(np.linalg.norm(self.cube_center.body.get_position() - self._goal_positions["center"]) <= 2 * tol))
        # 1 place center
        self._metric_stage(1, bool(self._is_on_table(self.cube_center) and np.linalg.norm(self.cube_center.body.get_position() - self._goal_positions["center"]) <= tol))

        # 2-3 front transport/place (proxy)
        self._metric_stage(2, bool(front_ok))
        self._metric_stage(3, bool(front_ok and self._is_on_table(self.cube_front)))

        # 4-5 back transport/place (proxy)
        self._metric_stage(4, bool(back_ok))
        self._metric_stage(5, bool(back_ok and self._is_on_table(self.cube_back)))

        # 6-7 left transport/place (proxy)
        self._metric_stage(6, bool(left_ok))
        self._metric_stage(7, bool(left_ok and self._is_on_table(self.cube_left)))

        # 8-9 right transport/place (proxy)
        self._metric_stage(8, bool(right_ok))
        self._metric_stage(9, bool(right_ok and self._is_on_table(self.cube_right)))

        # 10 final alignment
        self._metric_stage(10, bool(positions_ok))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        target_distance = {
            "center": float(np.linalg.norm(self.cube_center.body.get_position() - self._goal_positions["center"])),
            "front": float(np.linalg.norm(self.cube_front.body.get_position() - self._goal_positions["front"])),
            "back": float(np.linalg.norm(self.cube_back.body.get_position() - self._goal_positions["back"])),
            "left": float(np.linalg.norm(self.cube_left.body.get_position() - self._goal_positions["left"])),
            "right": float(np.linalg.norm(self.cube_right.body.get_position() - self._goal_positions["right"])),
        }
        pose_error = {
            # Orientation is not task-relevant; report 0.0 (identity target).
            "center_quat_err": 0.0,
            "front_quat_err": 0.0,
            "back_quat_err": 0.0,
            "left_quat_err": 0.0,
            "right_quat_err": 0.0,
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})