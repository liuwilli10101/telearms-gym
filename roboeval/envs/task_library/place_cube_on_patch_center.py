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

class RedCubeCenterPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Cover the small green patch with the red cube.

Success Criteria:
The red cube center must land inside the drawn green square, which is exactly the accepted region, resting at table height and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    # This task asks for centering, not just "somewhere on the patch", so the fix is
    # to shrink the drawing rather than widen the check: the patch used to be drawn
    # as a 20 cm square (GoalPatch sizes are HALF-extents) while success demanded the
    # cube centre land within 2 cm of its middle, an invisible target inside a target.
    # The square is now the accepted region itself -- 6 cm across, so a 4 cm cube
    # sitting on it with a rim showing on every side is exactly what passes.
    _PATCH_HALF_EXTENT = 0.03
    _XY_TOL_M = _PATCH_HALF_EXTENT

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Assets (MUST be exactly: ColoredCube, GoalPatch)
        self.red_cube = ColoredCube(self._mojo)
        self.red_cube.set_color("red")  # REQUIRED in _initialize_env

        self.green_goal_patch = GoalPatch(
            self._mojo,
            color="green",
            size=(self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT),
            name="target",
        )

        # Convenience list for metrics/slip tracking
        self.cubes = [self.red_cube]

        # Place patch at a fixed, reachable location (within workspace bounds)
        patch_pos = np.array([0.62, 0.00, self._PATCH_REST_Z])
        self.green_goal_patch.body.set_position(patch_pos)

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

        # 2. Deterministic RNG (kept for pipeline compatibility)
        rng = np.random.default_rng(self.seed)

        # 3. RANDOM UNSOLVED STATE: cube placed away from patch center and (typically) off the patch
        self.red_cube.set_color("red")  # REQUIRED in _on_reset

        patch_pos = self.green_goal_patch.body.get_position()

        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Sample clear of the drawn patch so the reset is never already solved.
        # Derived from the patch half-extent, not guessed at: reading size=(0.1, 0.1)
        # as a full width is what made the old margin land the cube on the target.
        min_xy_dist = self._PATCH_HALF_EXTENT + 0.06

        for _ in range(200):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            xy_err = float(np.linalg.norm(np.array([x - patch_pos[0], y - patch_pos[1]])))
            if xy_err >= min_xy_dist:
                break
        else:
            # Fallback: push along +Y within bounds if sampling fails (still unsolved)
            x = float(np.clip(patch_pos[0], x_min, x_max))
            y = float(np.clip(patch_pos[1] + min_xy_dist, y_min, y_max))

        # Random yaw about Z for variety. set_pose feeds this to Quaternion(), which
        # reads [w, x, y, z]; re-ordering to xyzw turned the yaw into a 180 deg flip
        # about a tilted axis and reset the cube lying over.
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # (w, x, y, z)

        self.red_cube.set_pose(position=np.array([x, y, z], dtype=float), quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_distance(self, a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
        d = a_xyz[:2] - b_xyz[:2]
        return float(np.linalg.norm(d))

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        # No inset: the patch is drawn at the tolerance, so its edge is the promise.
        return cube_on_patch(
            block,
            target,
            (self._PATCH_HALF_EXTENT, self._PATCH_HALF_EXTENT),
            inset=0.0,
            rest_z=self._CUBE_REST_Z,
        )

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        cube_pos = self.red_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        xy_err = self._xy_distance(cube_pos, patch_pos)
        # Per-axis, so the accepted region is the square that is drawn rather than
        # the circle inscribed in it -- the corners of the patch have to count too.
        centered = bool(np.all(np.abs((cube_pos - patch_pos)[:2]) <= self._XY_TOL_M))
        on_patch = self._is_block_on_target(self.red_cube, self.green_goal_patch)

        holding = any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        )
        released = not holding

        # Stage subgoals (heuristic signals)
        # 1) grasp
        grasped = holding
        # 2) lift (clear of table)
        lifted = grasped and (cube_pos[2] > (self._CUBE_REST_Z + 0.02))
        # 3) transport (above/near patch while lifted)
        transported = lifted and (xy_err < 0.10)
        # 4) align (centered in XY while lifted)
        aligned = lifted and centered
        # 5) place (on patch and roughly at rest height)
        placed = on_patch and centered and (cube_pos[2] < (self._CUBE_REST_Z + 0.02))
        # 6) release
        released_ok = placed and released

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(aligned))
        self._metric_stage(4, bool(placed))
        self._metric_stage(5, bool(released_ok))

        positions_ok = bool(centered and on_patch)
        semantic_ok = True  # No semantic constraints beyond color; cube is explicitly set to red.

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy_m": float(xy_err),
                "xy_tolerance_m": float(self._XY_TOL_M),
            },
            pose_error={
                "cube_xy_error_m": float(xy_err),
                "cube_z_error_m": float(abs(float(cube_pos[2] - self._CUBE_REST_Z))),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})