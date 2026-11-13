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
from roboeval.utils.guide_patch import cube_on_patch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class ColorMatchedCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place red on red, blue on blue, and green on green.

Success Criteria:
Each cube center may be anywhere on its matching square patch with a 2 cm inset from the edges. Small contact gaps are tolerated, but the cubes must be resting near the surface and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _NUM_STAGES = 3
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    # GoalPatch sizes are HALF-extents, so size=(0.1, 0.1) draws a 20 cm square.
    # Success used to demand the cube centre land within 1.4 cm of the patch CENTRE,
    # i.e. a 2.8 cm bullseye inside a 20 cm target, which is why operators reported
    # visibly-correct placements failing. The cube now counts as placed when it sits
    # on the patch, with a small inset so it must be properly on rather than clipping
    # a corner.
    _PATCH_HALF_EXTENT = 0.1
    _PATCH_INSET = 0.02
    _XY_TOL = _PATCH_HALF_EXTENT - _PATCH_INSET  # 0.08 m per axis from patch centre

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Goal patches (fixed targets) ---
        self.red_goal_patch = GoalPatch(self._mojo, color="red", size=(self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT), name="red_goal_patch")
        self.blue_goal_patch = GoalPatch(self._mojo, color="blue", size=(self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT), name="blue_goal_patch")
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT), name="green_goal_patch")

        # --- Colored cubes ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube]
        self.patches = [self.red_goal_patch, self.blue_goal_patch, self.green_goal_patch]

        # Canonical goal layout (within workspace bounds; spaced along Y)
        self._goal_x = 0.55
        self._goal_y_map = {
            "red": 0.18,
            "blue": 0.00,
            "green": -0.18,
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

        # 2. Deterministic RNG (pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        # Place patches at canonical goal locations (targets remain fixed)
        red_patch_pos = np.array([self._goal_x, self._goal_y_map["red"], self._PATCH_REST_Z])
        blue_patch_pos = np.array([self._goal_x, self._goal_y_map["blue"], self._PATCH_REST_Z])
        green_patch_pos = np.array([self._goal_x, self._goal_y_map["green"], self._PATCH_REST_Z])

        self.red_goal_patch.body.set_position(red_patch_pos)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.blue_goal_patch.body.set_position(blue_patch_pos)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.green_goal_patch.body.set_position(green_patch_pos)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # Randomize cube poses in workspace bounds, ensuring UNSOLVED configuration
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        patch_xy = np.stack([red_patch_pos[:2], blue_patch_pos[:2], green_patch_pos[:2]], axis=0)
        # Start clear of every patch so the reset is never already solved. This is
        # a function of the patch size, not a multiple of the success tolerance --
        # scaling it with the tolerance would demand 28 cm of clearance and starve
        # the rejection sampler.
        min_patch_clearance = self._PATCH_HALF_EXTENT + 0.03
        min_cube_separation = 0.06

        def sample_xy(existing_xy: list[np.ndarray]) -> np.ndarray:
            for _ in range(500):
                xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

                # Not on/near any patch (prevents accidental collision+XY pass)
                if np.min(np.linalg.norm(patch_xy - xy[None, :], axis=1)) < min_patch_clearance:
                    continue

                # Keep cubes separated
                if existing_xy:
                    if np.min(np.linalg.norm(np.stack(existing_xy, axis=0) - xy[None, :], axis=1)) < min_cube_separation:
                        continue

                # Also ensure not accidentally on its own matching patch specifically
                # (redundant but explicit for requirement 2)
                return xy

            # Fallback (still within bounds); pick a corner-ish spot away from patches
            return np.array([x_min, y_max], dtype=float)

        placed_xy: list[np.ndarray] = []
        for cube in [self.red_cube, self.blue_cube, self.green_cube]:
            xy = sample_xy(placed_xy)
            placed_xy.append(xy)

            yaw = rng.uniform(-np.pi, np.pi)
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)

            cube.set_pose(position=np.array([xy[0], xy[1], z], dtype=float), quat=quat)
            if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
                self._mojo.physics.forward()

        # Physics settling
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_STAGES):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
        d = a_xyz[:2] - b_xyz[:2]
        return float(np.linalg.norm(d))

    def _is_cube_on_patch(self, cube: ColoredCube, patch: GoalPatch) -> bool:
        return cube_on_patch(
            cube, patch, (self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT), self._PATCH_INSET
        )

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red_ok = self._is_cube_on_patch(self.red_cube, self.red_goal_patch)
        blue_ok = self._is_cube_on_patch(self.blue_cube, self.blue_goal_patch)
        green_ok = self._is_cube_on_patch(self.green_cube, self.green_goal_patch)
        positions_ok = bool(red_ok and blue_ok and green_ok)

        # Semantic check (color binding) — verify cube colors match intended mapping
        semantic_ok = (
            self.red_cube.get_color() == "red"
            and self.blue_cube.get_color() == "blue"
            and self.green_cube.get_color() == "green"
        )

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics
        if red_ok:
            self._metric_stage(0, True)
        if blue_ok:
            self._metric_stage(1, True)
        if green_ok:
            self._metric_stage(2, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Distances for metrics
        target_distance = {
            "red_cube_to_red_patch_xy": self._xy_dist(
                self.red_cube.body.get_position(), self.red_goal_patch.body.get_position()
            ),
            "blue_cube_to_blue_patch_xy": self._xy_dist(
                self.blue_cube.body.get_position(), self.blue_goal_patch.body.get_position()
            ),
            "green_cube_to_green_patch_xy": self._xy_dist(
                self.green_cube.body.get_position(), self.green_goal_patch.body.get_position()
            ),
        }

        pose_error = {
            # Orientation not constrained in this task; report 0 for completeness.
            "red_cube_quat_error": 0.0,
            "blue_cube_quat_error": 0.0,
            "green_cube_quat_error": 0.0,
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})