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


class SideBySideCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place the red and blue cubes onto the same green goal patch, arranging them side by side
    along the Y-axis. Ensure the left-right order follows the workspace convention (+Y is left).

    Sequential Sub-Goals:
        1. Place the red cube on the green goal patch.
        2. Place the blue cube on the green goal patch.
        3. Ensure red is left of blue along +Y (red_y > blue_y).
        4. Align the cubes along X (same front-back position within tolerance).
        5. Release both cubes (no gripper holding either cube).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants ---
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    _X_ALIGN_TOL = 0.03
    _LEFT_OF_MARGIN = 0.01  # meters; require a positive Y separation for "left_of"
    _Y_SEPARATION = 0.05    # meters; safe side-by-side spacing for 4cm cubes on a 10cm patch

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY those allowed) ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # Required: set_color in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cubes = [self.red_cube, self.blue_cube]

        # Initialize metrics with full signature (required)
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
        # 1) Re-initialize metrics (required on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # Required: set_color in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3) Place objects in RANDOM *unsolved* state within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Randomize patch position within bounds (keep some margin from edges)
        patch_margin = 0.06
        patch_x = float(rng.uniform(x_min + patch_margin, x_max - patch_margin))
        patch_y = float(rng.uniform(y_min + patch_margin, y_max - patch_margin))
        patch_pos = np.array([patch_x, patch_y, self._PATCH_Z], dtype=float)
        self.green_goal_patch.body.set_position(patch_pos)

        # Sample cube positions so at least one cube is definitely NOT on the patch,
        # and ensure cubes are not overlapping each other.
        min_cube_sep = 0.07          # > ~cube size to avoid overlap at reset
        not_on_patch_dist = 0.14     # > patch half-diagonal (~0.0707) to ensure off-patch in XY

        def sample_xy():
            return np.array(
                [float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))],
                dtype=float,
            )

        patch_xy = patch_pos[:2].copy()

        for _ in range(500):
            red_xy = sample_xy()
            blue_xy = sample_xy()

            # separate cubes
            if float(np.linalg.norm(red_xy - blue_xy)) < min_cube_sep:
                continue

            # ensure at least one cube is far from patch (so success cannot be true)
            red_far = float(np.linalg.norm(red_xy - patch_xy)) >= not_on_patch_dist
            blue_far = float(np.linalg.norm(blue_xy - patch_xy)) >= not_on_patch_dist
            if not (red_far or blue_far):
                continue

            break
        else:
            # Fallback deterministic unsolved placement (still within bounds)
            red_xy = np.array([0.40, 0.25], dtype=float)
            blue_xy = np.array([0.70, -0.25], dtype=float)

        # Random yaw for each cube
        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        red_pos = np.array([red_xy[0], red_xy[1], self._CUBE_Z], dtype=float)
        blue_pos = np.array([blue_xy[0], blue_xy[1], self._CUBE_Z], dtype=float)

        self.red_cube.set_pose(position=red_pos, quat=yaw_quat())
        self.blue_cube.set_pose(position=blue_pos, quat=yaw_quat())

        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """
        Reliable 'on patch' check using the GoalPatch.collider direction.
        """
        if not hasattr(target, "collider"):
            return False
        if not block.is_colliding(target.collider):
            return False
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False
        return True

    def _success(self) -> bool:
        # Reset success flag at start (required)
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "green_goal_patch") or not hasattr(self, "red_cube") or not hasattr(self, "blue_cube"):
            return False

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        # Goal condition 1/2: both cubes on the green goal patch
        red_on_patch = self._is_block_on_target(self.red_cube, self.green_goal_patch)
        blue_on_patch = self._is_block_on_target(self.blue_cube, self.green_goal_patch)

        # Goal condition 3: red left of blue along Y (+Y is left)
        left_right_ok = (red_pos[1] - blue_pos[1]) > self._LEFT_OF_MARGIN

        # Goal condition 4: aligned along X (front-back)
        x_aligned = abs(float(red_pos[0] - blue_pos[0])) <= self._X_ALIGN_TOL

        positions_ok = bool(red_on_patch and blue_on_patch and left_right_ok and x_aligned)
        semantic_ok = True  # no semantic constraints in this task

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging
        self._metric_stage(0, bool(red_on_patch))
        self._metric_stage(1, bool(blue_on_patch))
        self._metric_stage(2, bool(left_right_ok))
        self._metric_stage(3, bool(x_aligned))
        self._metric_stage(4, bool(released))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (must be last metric call before return)
        red_xy = np.array([red_pos[0], red_pos[1]], dtype=float)
        blue_xy = np.array([blue_pos[0], blue_pos[1]], dtype=float)
        patch_xy = np.array([patch_pos[0], patch_pos[1]], dtype=float)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_patch_xy_m": float(np.linalg.norm(red_xy - patch_xy)),
                "blue_to_patch_xy_m": float(np.linalg.norm(blue_xy - patch_xy)),
            },
            pose_error={
                "x_alignment_abs_m": float(abs(red_pos[0] - blue_pos[[0]])),
                "y_order_margin_m": float(red_pos[1] - blue_pos[1]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})