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


class TCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes on the tabletop into a T-shape:
    three cubes form a horizontal row (left-to-right along +Y to -Y),
    and one cube is placed centered behind the middle cube (toward +X).

    Sequential Sub-Goals:
        1. (Optional) Grasp any cube to begin arranging.
        2. Move a cube to serve as the center (intersection) of the T.
        3. Place the center cube on the table.
        4. Move a cube to the left of the center (+Y).
        5. Place the left cube aligned in X with the center.
        6. Move a cube to the right of the center (-Y).
        7. Place the right cube aligned in X with the center.
        8. Move the final cube behind the center (+X), centered in Y.
        9. Place the stem cube to complete the T.
        10. Fine-adjust all cubes to satisfy tolerances.
        11. Release all cubes.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace bounds required by task spec
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _Z_TABLE_CUBE = 0.97

    # Goal tolerances/thresholds from task spec
    _ALIGN_TOL_X = 0.03
    _ALIGN_TOL_Y = 0.03
    _LEFT_MIN_DY = 0.03
    _RIGHT_MIN_DY = 0.03
    _STEM_MIN_DX = 0.05

    _NUM_SUBGOALS = 11

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly four ColoredCubes (ONLY allowed asset)
        self.cube_left = ColoredCube(self._mojo)
        self.cube_center = ColoredCube(self._mojo)
        self.cube_right = ColoredCube(self._mojo)
        self.cube_stem = ColoredCube(self._mojo)

        self.cubes = [self.cube_left, self.cube_center, self.cube_right, self.cube_stem]

        # REQUIRED: set_color() in _initialize_env()
        self.cube_left.set_color("red")
        self.cube_center.set_color("blue")
        self.cube_right.set_color("green")
        self.cube_stem.set_color("yellow")

        # Deterministic goal state (T-shape)
        cx, cy = 0.55, 0.00
        arm_dy = 0.06  # > 0.03
        stem_dx = 0.07  # > 0.05

        self._goal_pos = {
            "cube_center": np.array([cx, cy, self._Z_TABLE_CUBE]),
            "cube_left": np.array([cx, cy + arm_dy, self._Z_TABLE_CUBE]),   # +Y (left)
            "cube_right": np.array([cx, cy - arm_dy, self._Z_TABLE_CUBE]),  # -Y (right)
            "cube_stem": np.array([cx + stem_dx, cy, self._Z_TABLE_CUBE]),  # +X (behind)
        }

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

    def _sample_initial_object_poses(self, rng: np.random.Generator):
        """
        Sample non-colliding tabletop XY poses within the required workspace.
        (Not used for goal-state reset; provided for downstream randomization pipelines.)
        """
        xmin, xmax = self._X_BOUNDS
        ymin, ymax = self._Y_BOUNDS

        # Conservative spacing to avoid initial contact
        min_center_dist = 0.08

        poses_xy = []
        for _ in range(5000):
            if len(poses_xy) >= 4:
                break
            x = rng.uniform(xmin + 0.03, xmax - 0.03)
            y = rng.uniform(ymin + 0.03, ymax - 0.03)
            p = np.array([x, y], dtype=float)
            if all(np.linalg.norm(p - q) >= min_center_dist for q in poses_xy):
                poses_xy.append(p)

        # Fallback: deterministic spread if sampling failed
        if len(poses_xy) < 4:
            poses_xy = [
                np.array([0.50, 0.18]),
                np.array([0.62, 0.15]),
                np.array([0.60, -0.18]),
                np.array([0.48, -0.15]),
            ]

        return [np.array([p[0], p[1], self._Z_TABLE_CUBE]) for p in poses_xy[:4]]

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

        # 3. REQUIRED: set_color() in _on_reset()
        self.cube_left.set_color("red")
        self.cube_center.set_color("blue")
        self.cube_right.set_color("green")
        self.cube_stem.set_color("yellow")

        # 4. Place objects in RANDOM *unsolved* positions (guarantee _success() is False)
        def random_yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        def positions_ok(p_left, p_center, p_right, p_stem) -> bool:
            in_bounds = all(self._within_workspace(p) for p in [p_left, p_center, p_right, p_stem])
            left_of = (float(p_left[1] - p_center[1]) >= self._LEFT_MIN_DY)
            right_of = (float(p_right[1] - p_center[1]) <= -self._RIGHT_MIN_DY)
            x_align_left = (abs(float(p_left[0] - p_center[0])) <= self._ALIGN_TOL_X)
            x_align_right = (abs(float(p_right[0] - p_center[0])) <= self._ALIGN_TOL_X)
            stem_behind = (float(p_stem[0] - p_center[0]) >= self._STEM_MIN_DX)
            stem_centered_y = (abs(float(p_stem[1] - p_center[1])) <= self._ALIGN_TOL_Y)
            return bool(
                in_bounds
                and left_of
                and right_of
                and x_align_left
                and x_align_right
                and stem_behind
                and stem_centered_y
            )

        poses = None
        for _ in range(200):
            sampled = self._sample_initial_object_poses(rng)
            # Assign in canonical order: left, center, right, stem
            p_left, p_center, p_right, p_stem = sampled[0], sampled[1], sampled[2], sampled[3]
            if not positions_ok(p_left, p_center, p_right, p_stem):
                poses = (p_left, p_center, p_right, p_stem)
                break

        # Hard fallback: force an obviously unsolved configuration (violates left/right/stem relations)
        if poses is None:
            cx = float(rng.uniform(self._X_BOUNDS[0] + 0.05, self._X_BOUNDS[1] - 0.05))
            cy = float(rng.uniform(self._Y_BOUNDS[0] + 0.05, self._Y_BOUNDS[1] - 0.05))
            p_center = np.array([cx, cy, self._Z_TABLE_CUBE], dtype=float)
            # Keep all other cubes near the same Y so left_of/right_of fail
            p_left = np.array([cx - 0.06, cy + 0.005, self._Z_TABLE_CUBE], dtype=float)
            p_right = np.array([cx + 0.06, cy - 0.005, self._Z_TABLE_CUBE], dtype=float)
            p_stem = np.array([cx - 0.02, cy + 0.01, self._Z_TABLE_CUBE], dtype=float)  # not behind (+X) enough
            poses = (p_left, p_center, p_right, p_stem)

        p_left, p_center, p_right, p_stem = poses

        self.cube_center.set_pose(position=p_center.copy(), quat=random_yaw_quat())
        self.cube_left.set_pose(position=p_left.copy(), quat=random_yaw_quat())
        self.cube_right.set_pose(position=p_right.copy(), quat=random_yaw_quat())
        self.cube_stem.set_pose(position=p_stem.copy(), quat=random_yaw_quat())

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _within_workspace(self, pos: np.ndarray) -> bool:
        return (
            (self._X_BOUNDS[0] <= float(pos[0]) <= self._X_BOUNDS[1])
            and (self._Y_BOUNDS[0] <= float(pos[1]) <= self._Y_BOUNDS[1])
        )

    def _success(self) -> bool:
        # CRITICAL: reset success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_pos"):
            return False

        # Get positions
        p_left = self.cube_left.body.get_position()
        p_center = self.cube_center.body.get_position()
        p_right = self.cube_right.body.get_position()
        p_stem = self.cube_stem.body.get_position()

        # Workspace bounds (task spec)
        in_bounds = all(self._within_workspace(p) for p in [p_left, p_center, p_right, p_stem])

        # Table contact (ensure cubes are on table, not floating/fallen)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # T-shape relations:
        # - left/right defined along Y: +Y is left, -Y is right
        # - behind defined along X: +X is behind
        left_of = (float(p_left[1] - p_center[1]) >= self._LEFT_MIN_DY)
        right_of = (float(p_right[1] - p_center[1]) <= -self._RIGHT_MIN_DY)

        x_align_left = (abs(float(p_left[0] - p_center[0])) <= self._ALIGN_TOL_X)
        x_align_right = (abs(float(p_right[0] - p_center[0])) <= self._ALIGN_TOL_X)

        stem_behind = (float(p_stem[0] - p_center[0]) >= self._STEM_MIN_DX)
        stem_centered_y = (abs(float(p_stem[1] - p_center[1])) <= self._ALIGN_TOL_Y)

        positions_ok = bool(
            in_bounds
            and on_table
            and left_of
            and right_of
            and x_align_left
            and x_align_right
            and stem_behind
            and stem_centered_y
        )

        # Semantic/content constraints: none for this task
        semantic_ok = True

        # Released: no cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Mark stages when final arrangement is satisfied
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "cube_left": float(np.linalg.norm(p_left - self._goal_pos["cube_left"])),
            "cube_center": float(np.linalg.norm(p_center - self._goal_pos["cube_center"])),
            "cube_right": float(np.linalg.norm(p_right - self._goal_pos["cube_right"])),
            "cube_stem": float(np.linalg.norm(p_stem - self._goal_pos["cube_stem"])),
        }
        pose_error = {
            # Orientation not constrained; report position error as pose proxy
            "cube_left_pos_err": target_distance["cube_left"],
            "cube_center_pos_err": target_distance["cube_center"],
            "cube_right_pos_err": target_distance["cube_right"],
            "cube_stem_pos_err": target_distance["cube_stem"],
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})