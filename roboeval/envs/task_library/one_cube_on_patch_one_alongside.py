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


class CubeAndRightPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place one colored cube on the green goal patch and place the second cube
    immediately to the right of the patch so the cube touches the patch's right edge.

    Sequential Sub-Goals:
        1. Grasp one cube (intended for placement on the patch).
        2. Move the grasped cube above the green goal patch.
        3. Place the cube onto the green goal patch.
        4. Release the cube after it is stably on the patch.
        5. Grasp the second cube (intended for right-of-patch placement).
        6. Move the second cube to the right side of the green goal patch.
        7. Align the cube so its left face is flush with the patch's right side.
        8. Place the cube on the table immediately to the right of the patch.
        9. Release the cube after it is stably placed touching the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951
    _PATCH_SIZE = (0.1, 0.1)  # (x, y) size on table

    _CUBE_HALF = 0.02
    _PATCH_HALF_Y = 0.05

    # Spatial tolerances (task registry suggests 0.014m)
    _POS_TOL = 0.014
    _Z_TOL = 0.02

    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Required assets only: ColoredCube, GoalPatch
        self.cube_on_patch = ColoredCube(self._mojo)
        self.cube_right_of_patch = ColoredCube(self._mojo)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_on_patch.set_color("red")
        self.cube_right_of_patch.set_color("blue")

        self.goal_patch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="target")

        self.cubes = [self.cube_on_patch, self.cube_right_of_patch]

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

    def _physics_forward_if_available(self):
        # Some backends expose physics.forward(); be defensive.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _desired_right_cube_position(self) -> np.ndarray:
        """
        "Right of patch" in this coordinate system is -Y direction.
        Touching condition: cube's left face touches the patch's right edge.

        Patch right edge y = patch_y - patch_half_y
        Cube left face y = cube_y + cube_half
        Touch => cube_y + cube_half = patch_y - patch_half_y
              => cube_y = patch_y - (patch_half_y + cube_half)
        """
        patch_pos = self.goal_patch.body.get_position()
        desired = patch_pos.copy()
        desired[2] = self._CUBE_Z
        desired[1] = patch_pos[1] - (self._PATCH_HALF_Y + self._CUBE_HALF)
        return desired

    def _is_block_on_patch(self, block: ColoredCube, patch: GoalPatch) -> bool:
        # Prefer GoalPatch.collider when available; fall back to patch itself.
        coll_target = patch.collider if hasattr(patch, "collider") and patch.collider is not None else patch
        collision_ok = bool(block.is_colliding(coll_target))

        # Geometric fallback (collision can be flaky for thin patches)
        block_pos = block.body.get_position()
        patch_pos = patch.body.get_position()
        half_x = float(self._PATCH_SIZE[0]) * 0.5
        half_y = float(self._PATCH_SIZE[1]) * 0.5
        within_x = abs(block_pos[0] - patch_pos[0]) <= (half_x - self._CUBE_HALF + self._POS_TOL)
        within_y = abs(block_pos[1] - patch_pos[1]) <= (half_y - self._CUBE_HALF + self._POS_TOL)
        geom_ok = bool(within_x and within_y)

        # Ensure it's resting on the table height (not floating)
        z = block_pos[2]
        if abs(z - self._CUBE_Z) > self._Z_TOL:
            return False

        return bool(collision_ok or geom_ok)

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

        # 2. Deterministic RNG (kept for framework consistency)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_on_patch.set_color("red")
        self.cube_right_of_patch.set_color("blue")

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        def sample_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        # 3. Randomize objects into UNSOLVED positions (ensure _success() is False)
        # Randomize patch position
        patch_xy = sample_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_Z], dtype=float)
        self.goal_patch.body.set_position(patch_pos)
        self._physics_forward_if_available()

        # Helper predicates to avoid accidentally sampling a solved configuration
        def cube_b_is_in_goal_region(cube_b_pos: np.ndarray) -> bool:
            desired_b = self._desired_right_cube_position()
            x_aligned = abs(cube_b_pos[0] - patch_pos[0]) <= self._POS_TOL
            y_touching = abs(cube_b_pos[1] - desired_b[1]) <= self._POS_TOL
            z_on_table = abs(cube_b_pos[2] - self._CUBE_Z) <= self._Z_TOL
            return bool(x_aligned and y_touching and z_on_table)

        # Sample Cube A (red) NOT on patch
        for _ in range(200):
            a_xy = sample_xy()
            cube_a_pos = np.array([a_xy[0], a_xy[1], self._CUBE_Z], dtype=float)
            self.cube_on_patch.set_pose(position=cube_a_pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
            self._physics_forward_if_available()
            if not self._is_block_on_patch(self.cube_on_patch, self.goal_patch):
                break
        else:
            # Fallback: force it off-patch deterministically
            cube_a_pos = np.array([x_min, y_max, self._CUBE_Z], dtype=float)
            self.cube_on_patch.set_pose(position=cube_a_pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
            self._physics_forward_if_available()

        # Sample Cube B (blue) NOT in right-of-patch touching pose
        for _ in range(200):
            b_xy = sample_xy()
            cube_b_pos = np.array([b_xy[0], b_xy[1], self._CUBE_Z], dtype=float)

            # Also keep cubes from starting interpenetrating/stacked
            if np.linalg.norm((cube_b_pos - cube_a_pos)[:2]) < (2.5 * self._CUBE_HALF):
                continue

            self.cube_right_of_patch.set_pose(position=cube_b_pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
            self._physics_forward_if_available()

            if not cube_b_is_in_goal_region(cube_b_pos):
                break
        else:
            # Fallback: place it far from the desired right-of-patch position
            cube_b_pos = np.array([x_max, y_min, self._CUBE_Z], dtype=float)
            self.cube_right_of_patch.set_pose(position=cube_b_pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
            self._physics_forward_if_available()

        self._physics_forward_if_available()

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
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False

        # Condition 1: Cube A is on the green goal patch
        on_patch = self._is_block_on_patch(self.cube_on_patch, self.goal_patch)

        # Condition 2: Cube B is directly to the right of the patch and touching its right edge
        patch_pos = self.goal_patch.body.get_position()
        cube_b_pos = self.cube_right_of_patch.body.get_position()
        desired_b = self._desired_right_cube_position()

        x_aligned = abs(cube_b_pos[0] - patch_pos[0]) <= self._POS_TOL
        y_touching = abs(cube_b_pos[1] - desired_b[1]) <= self._POS_TOL
        z_on_table = abs(cube_b_pos[2] - self._CUBE_Z) <= self._Z_TOL

        # Ensure cube B is on the table surface (prefer collision, but allow geometric Z fallback)
        if hasattr(self.table, "table_top"):
            on_table_b = bool(self.cube_right_of_patch.is_colliding(self.table.table_top))
        else:
            on_table_b = False
        on_table_b = bool(on_table_b or z_on_table)

        right_touch_ok = bool(x_aligned and y_touching and z_on_table and on_table_b)

        positions_ok = bool(on_patch and right_touch_ok)

        # No semantic constraints beyond colors (colors are fixed and not evaluated)
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage bookkeeping (simple: mark all stages when arrangement is achieved)
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize
        cube_a_to_patch_xy = float(np.linalg.norm(self.cube_on_patch.body.get_position()[:2] - patch_pos[:2]))
        cube_b_to_goal_xy = float(np.linalg.norm(cube_b_pos[:2] - desired_b[:2]))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_on_patch_to_patch_xy": cube_a_to_patch_xy,
                "cube_right_of_patch_to_goal_xy": cube_b_to_goal_xy,
            },
            pose_error={
                "cube_right_of_patch_xy_error": cube_b_to_goal_xy,
                "cube_on_patch_xy_error": cube_a_to_patch_xy,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})