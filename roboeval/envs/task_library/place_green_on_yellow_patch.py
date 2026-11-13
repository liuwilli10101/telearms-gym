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

class GreenCubeOnYellowTargetTask(RoboEvalEnv, MetricRolloutEval):
    """Place the green cube on the yellow target patch.

Success Criteria:
- The green cube is on the yellow patch, supported on the table, and within the expected rest-height tolerance.
- The cube color is green and no gripper is holding it."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY allowed assets) ---
        self.green_cube = ColoredCube(self._mojo)
        self.green_cube.set_color("green")  # REQUIRED in _initialize_env

        self.yellow_target_patch = GoalPatch(
            self._mojo, color="yellow", size=(0.1, 0.1), name="target"
        )

        # Standard list handle for metrics + generic loops
        self.cubes = [self.green_cube]

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

    def _physics_forward(self):
        """Best-effort physics forward after discrete placements."""
        try:
            if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
                self._mojo.physics.forward()
        except Exception:
            pass

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """
        Reliable 'on patch' check: collision from the BLOCK side against target.collider.
        """
        collider = getattr(target, "collider", target)
        try:
            if not block.is_colliding(collider):
                return False
        except Exception:
            return False
        # If a floor geom exists, reject floor-contact (defensive)
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

        # 2) Deterministic RNG handle (required pattern)
        rng = np.random.default_rng(self.seed)  # noqa: F841

        # 3) Random UNSOLVED placement (patch on table; cube elsewhere on table)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25]
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Sample patch position
        patch_xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_REST_Z], dtype=float)
        self.yellow_target_patch.body.set_position(patch_pos)
        self._physics_forward()
        self._mojo.physics.forward()

        # REQUIRED: set_color() in _on_reset as well
        self.green_cube.set_color("green")

        # Sample cube position far enough from patch so success is impossible right after reset
        # (_success allows on_patch via dist_xy < 0.04 even without collision)
        min_dist = 0.12  # comfortably > 0.04 and also > "over_patch" threshold 0.08
        for _ in range(200):
            cube_xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if float(np.linalg.norm(cube_xy - patch_xy)) >= min_dist:
                break
        else:
            # Fallback: place cube at opposite corner within bounds
            cube_xy = np.array([x_max, y_max], dtype=float)
            if float(np.linalg.norm(cube_xy - patch_xy)) < min_dist:
                cube_xy = np.array([x_min, y_min], dtype=float)

        cube_pos = np.array([cube_xy[0], cube_xy[1], self._CUBE_REST_Z], dtype=float)

        # Random yaw (stable on table), keep quat normalized
        yaw = float(rng.uniform(-np.pi, np.pi))
        # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)

        self.green_cube.set_pose(position=cube_pos, quat=quat)
        self._physics_forward()
        self._mojo.physics.forward()

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        cube = self.green_cube
        patch = self.yellow_target_patch

        cube_pos = cube.body.get_position()
        patch_pos = patch.body.get_position()

        dist_xy = float(np.linalg.norm((cube_pos - patch_pos)[:2]))

        # Goal condition: cube is on the patch and supported by the table
        on_patch_collision = self._is_block_on_target(cube, patch)
        on_patch = bool(
            on_patch_collision
            or (dist_xy < 0.04 and abs(float(cube_pos[2]) - self._CUBE_REST_Z) < 0.03)
        )

        try:
            on_table_collision = bool(cube.is_colliding(self.table.table_top))
        except Exception:
            on_table_collision = False
        on_table = bool(
            on_table_collision or abs(float(cube_pos[2]) - self._CUBE_REST_Z) < 0.03
        )

        # Extra stability guard: cube should be near its rest Z when placed
        z_ok = abs(float(cube_pos[2]) - self._CUBE_REST_Z) < 0.03

        positions_ok = bool(on_patch and on_table and z_ok)

        # Color verification (task-specific semantic constraint for ColoredCube)
        semantic_ok = bool(cube.get_color() == "green")

        # Released condition
        released = not any(
            self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers
        )

        # Subgoal staging (best-effort inference)
        held = not released
        lifted = (not on_table) and (float(cube_pos[2]) > self._CUBE_REST_Z + 0.02)

        over_patch = dist_xy < 0.08
        aligned = dist_xy < 0.02
        placed = bool(on_patch and on_table)

        self._metric_stage(0, bool(held))                  # grasp
        self._metric_stage(1, bool(lifted))                # lift
        self._metric_stage(2, bool(over_patch))            # transport
        self._metric_stage(3, bool(aligned))               # align
        self._metric_stage(4, bool(placed))                # place
        self._metric_stage(5, bool(placed and released))   # release

        # Success = all goal constraints + released
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy": dist_xy,
                "cube_to_patch_xyz": float(np.linalg.norm(cube_pos - patch_pos)),
            },
            pose_error={
                "cube_rest_z_abs_error": abs(float(cube_pos[2]) - self._CUBE_REST_Z),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})