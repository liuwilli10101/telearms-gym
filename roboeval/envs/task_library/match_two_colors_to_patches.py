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


class ColorMatchedCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place each of two colored cubes on its matching target patch.

Success Criteria:
- The red cube is on the red patch and the blue cube is on the blue patch by patch collision and XY tolerance checks.
- The cube colors are verified as red and blue.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    # Tight-ish XY tolerance in addition to collider contact (helps with thin patch geometry)
    _XY_TOL = 0.03

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY allowed assets) ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.red_goal_patch = GoalPatch(
            self._mojo, color="red", size=(0.1, 0.1), name="red_goal_patch"
        )
        self.blue_goal_patch = GoalPatch(
            self._mojo, color="blue", size=(0.1, 0.1), name="blue_goal_patch"
        )

        self.cubes = [self.red_cube, self.blue_cube]
        self.patches = [self.red_goal_patch, self.blue_goal_patch]

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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3. Place objects in RANDOM unsolved positions (within workspace bounds)
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25

        def rand_xy():
            return np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)

        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        # Randomize patch locations with separation to avoid overlap
        min_patch_sep = 0.18
        red_patch_xy = rand_xy()
        for _ in range(200):
            blue_patch_xy = rand_xy()
            if float(np.linalg.norm(blue_patch_xy - red_patch_xy)) >= min_patch_sep:
                break
        else:
            blue_patch_xy = np.array([0.55, -0.15], dtype=float)

        red_patch_pos = np.array([red_patch_xy[0], red_patch_xy[1], self._PATCH_Z], dtype=float)
        blue_patch_pos = np.array([blue_patch_xy[0], blue_patch_xy[1], self._PATCH_Z], dtype=float)

        self.red_goal_patch.body.set_position(red_patch_pos)
        self.blue_goal_patch.body.set_position(blue_patch_pos)

        self._mojo.physics.forward()

        # Randomize cube locations ensuring NOT on their matching patches (so _success() is False)
        # Keep cubes separated a bit to reduce initial collisions.
        min_not_on_patch = 0.10  # > patch half-size (0.05) and > _XY_TOL, avoids contact
        min_cube_sep = 0.08

        def sample_cube_xy(avoid_xy_list):
            for _ in range(400):
                xy = rand_xy()
                ok = True
                # Avoid being on either patch (especially matching) to prevent accidental success
                if float(np.linalg.norm(xy - red_patch_xy)) < min_not_on_patch:
                    ok = False
                if float(np.linalg.norm(xy - blue_patch_xy)) < min_not_on_patch:
                    ok = False
                for axy in avoid_xy_list:
                    if float(np.linalg.norm(xy - axy)) < min_cube_sep:
                        ok = False
                        break
                if ok:
                    return xy
            return rand_xy()

        red_cube_xy = sample_cube_xy([])
        blue_cube_xy = sample_cube_xy([red_cube_xy])

        red_cube_pos = np.array([red_cube_xy[0], red_cube_xy[1], self._CUBE_Z], dtype=float)
        blue_cube_pos = np.array([blue_cube_xy[0], blue_cube_xy[1], self._CUBE_Z], dtype=float)

        self.red_cube.set_pose(position=red_cube_pos, quat=yaw_quat())
        self.blue_cube.set_pose(position=blue_cube_pos, quat=yaw_quat())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(2):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_block_on_patch(self, block: ColoredCube, patch: GoalPatch) -> bool:
        """
        Reliable "on patch" check:
          1) Collision from the BLOCK side with patch.collider (direction matters)
          2) XY proximity to patch center (guards against grazing contacts)
        """
        if not hasattr(patch, "collider"):
            return False

        if not block.is_colliding(patch.collider):
            return False

        bpos = block.body.get_position()
        ppos = patch.body.get_position()
        xy_dist = float(np.linalg.norm((bpos - ppos)[:2]))
        if xy_dist > self._XY_TOL:
            return False

        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Goal conditions: each cube on its matching patch
        red_on_red = self._is_block_on_patch(self.red_cube, self.red_goal_patch)
        blue_on_blue = self._is_block_on_patch(self.blue_cube, self.blue_goal_patch)
        positions_ok = bool(red_on_red and blue_on_blue)

        # Semantic condition: colors match expectations (defensive)
        semantic_ok = bool(self.red_cube.get_color() == "red" and self.blue_cube.get_color() == "blue")

        # Released: no gripper holding any cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging
        if red_on_red and released:
            self._metric_stage(0, True)
        if blue_on_blue and released:
            self._metric_stage(1, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Distances for metrics
        red_xy = float(np.linalg.norm((self.red_cube.body.get_position() - self.red_goal_patch.body.get_position())[:2]))
        blue_xy = float(np.linalg.norm((self.blue_cube.body.get_position() - self.blue_goal_patch.body.get_position())[:2]))

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_cube_to_red_patch_xy": red_xy,
                "blue_cube_to_blue_patch_xy": blue_xy,
            },
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})