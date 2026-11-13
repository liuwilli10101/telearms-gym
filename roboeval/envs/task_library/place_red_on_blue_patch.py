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


class RedCubeTargetPlacement(RoboEvalEnv, MetricRolloutEval):
    """Place the red cube on the blue target patch.

Success Criteria:
- The red cube is on the blue goal patch according to the target collision check.
- No gripper is holding the red cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Assets (MUST use ONLY these)
        self.red_cube = ColoredCube(self._mojo)
        self.red_cube.set_color("red")  # REQUIRED in _initialize_env

        self.blue_goal_patch = GoalPatch(
            self._mojo, color="blue", size=(0.1, 0.1), name="target"
        )

        # Convenience list for metrics API (expects a list)
        self.cubes = [self.red_cube]

        # Initialize metrics (FULL signature required)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _maybe_forward_physics(self):
        # Some builds require physics.forward() after teleports.
        physics = getattr(self._mojo, "physics", None)
        if physics is not None and hasattr(physics, "forward"):
            physics.forward()

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """
        Reliable 'on patch' check: query collision from the BLOCK side.
        Prefer target.collider when available, but fall back to target itself.
        """
        patch_collider = getattr(target, "collider", None)
        collider = patch_collider if patch_collider is not None else target

        if not block.is_colliding(collider):
            return False

        # If a floor geom exists in some presets, reject floor contact.
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False

        return True

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

        # 2) Deterministic RNG (required pattern; may be used by downstream randomizers)
        rng = np.random.default_rng(self.seed)

        # 3) Place objects in RANDOM UNSOLVED positions (ensure _success() is False)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Randomize patch pose within bounds
        patch_xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
        patch_pos = np.array([patch_xy[0], patch_xy[1], 0.951], dtype=float)
        patch_yaw = float(rng.uniform(-np.pi, np.pi))
        patch_quat = np.array([0.0, 0.0, np.sin(patch_yaw / 2.0), np.cos(patch_yaw / 2.0)], dtype=float)

        if hasattr(self.blue_goal_patch, "set_pose"):
            self.blue_goal_patch.set_pose(position=patch_pos, quat=patch_quat)
        else:
            self.blue_goal_patch.body.set_position(patch_pos)
        self._maybe_forward_physics()

        # Randomize cube pose, but keep it clearly OFF the patch so collision-based "on patch" is False
        self.red_cube.set_color("red")  # REQUIRED in _on_reset

        # Rejection sample until cube is far enough in XY from patch center
        min_xy_sep = 0.12  # > patch half-size (~0.05) + margin to avoid accidental overlap/contact
        for _ in range(200):
            cube_xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if float(np.linalg.norm(cube_xy - patch_xy)) >= min_xy_sep:
                break
        else:
            # Fallback: deterministic offset within bounds
            cube_xy = np.clip(patch_xy + np.array([min_xy_sep, 0.0]), [x_min, y_min], [x_max, y_max])

        cube_pos = np.array([cube_xy[0], cube_xy[1], 0.97], dtype=float)
        cube_yaw = float(rng.uniform(-np.pi, np.pi))
        cube_quat = np.array([0.0, 0.0, np.sin(cube_yaw / 2.0), np.cos(cube_yaw / 2.0)], dtype=float)
        self.red_cube.set_pose(position=cube_pos, quat=cube_quat)
        self._maybe_forward_physics()

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        _ = rng

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "red_cube") or not hasattr(self, "blue_goal_patch") or not hasattr(self, "table"):
            return False

        # Goal condition: red cube "on" blue goal patch
        on_patch = self._is_block_on_target(self.red_cube, self.blue_goal_patch)

        # Released: no gripper holding the cube
        released = not any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        )

        # Subgoal staging (best-effort, using available signals)
        # 1) Grasp
        grasped = not released
        self._metric_stage(0, bool(grasped))

        # 2) Lift
        cube_pos = self.red_cube.body.get_position()
        lifted = (cube_pos[2] > 0.99) and (not self.red_cube.is_colliding(self.table.table_top))
        self._metric_stage(1, bool(lifted))

        # 3) Transport (above/near target in XY)
        patch_pos = self.blue_goal_patch.body.get_position()
        xy_dist = float(np.linalg.norm(cube_pos[:2] - patch_pos[:2]))
        transported = lifted and (xy_dist < 0.08)
        self._metric_stage(2, bool(transported))

        # 4) Align (tighter XY)
        aligned = lifted and (xy_dist < 0.02)
        self._metric_stage(3, bool(aligned))

        # 5) Place (on target)
        placed = bool(on_patch)
        self._metric_stage(4, bool(placed))

        # 6) Release (placed + released)
        self._metric_stage(5, bool(placed and released))

        # Final success
        self._success_check = bool(on_patch and released)

        # Metrics finalize (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy": xy_dist,
                "cube_to_patch_3d": float(distance(self.red_cube.body, self.blue_goal_patch.body)),
            },
            pose_error={
                "cube_xy_error": xy_dist,
                "cube_z_error": float(abs(cube_pos[2] - 0.97)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})