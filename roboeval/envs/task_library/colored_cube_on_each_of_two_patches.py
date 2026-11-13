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

class CubePatchPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place the red cube onto the green patch and the blue cube onto the yellow patch.
    Placing a cube on the wrong-colored patch does not count, even if every patch ends up holding a cube.

    Each cube must be centered on its target patch (resting on the patch, not merely
    near it or hanging over the edge), and both grippers must be released at the end.
    The task succeeds only when both cubes are simultaneously centered on their matching
    patches with nothing held.

    Sequential Sub-Goals:
        1. Grasp the red cube.
        2. Place the red cube centered on the green patch.
        3. Release the red cube.
        4. Grasp the blue cube.
        5. Place the blue cube centered on the yellow patch.
        6. Release the blue cube.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace constants / tolerances
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951
    # Patch half-extent is 0.1m and cube half-extent is 0.02m, so a cube whose center is
    # within 0.06m of the patch center is fully on the patch with no overhang. This matches
    # the "centered, not hanging over the edge" success criterion without demanding pixel-
    # perfect centering on a large patch.
    _ON_TARGET_XY_TOL = 0.06  # meters

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create required assets ONLY: ColoredCube, GoalPatch ---
        self.colored_cube_1 = ColoredCube(self._mojo)
        self.colored_cube_2 = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.colored_cube_1.set_color("red")
        self.colored_cube_2.set_color("blue")

        # Two fixed goal patches
        self.goal_patch_1 = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="goal_patch_1")
        self.goal_patch_2 = GoalPatch(self._mojo, color="yellow", size=(0.1, 0.1), name="goal_patch_2")

        self.cubes = [self.colored_cube_1, self.colored_cube_2]
        self.patches = [self.goal_patch_1, self.goal_patch_2]

        # Cache goal patch centers (XY), within workspace bounds
        self._patch_pos_1 = np.array([0.55, 0.15, self._PATCH_Z])
        self._patch_pos_2 = np.array([0.55, -0.15, self._PATCH_Z])

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

    def _maybe_forward_physics(self):
        # Some builds expose physics.forward(); fall back to stepping.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        else:
            self._mojo.step()

    def _on_reset(self):
        # REQUIRED: set_color() in _on_reset()
        self.colored_cube_1.set_color("red")
        self.colored_cube_2.set_color("blue")

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

        # 2. Deterministic RNG (pipeline may add further randomization later)
        rng = np.random.default_rng(self.seed)

        # 3. Place goal patches at fixed locations
        self.goal_patch_1.body.set_position(self._patch_pos_1.copy())
        self._maybe_forward_physics()

        self.goal_patch_2.body.set_position(self._patch_pos_2.copy())
        self._maybe_forward_physics()

        # 4. Randomize cube initial poses in UNSOLVED positions (within workspace bounds)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z

        p1_xy = self._patch_pos_1[:2].copy()
        p2_xy = self._patch_pos_2[:2].copy()

        # Ensure cubes start off both patches (and not within the on-target tolerance).
        # Use a margin to avoid borderline cases due to contact/collision jitter.
        min_patch_clearance = self._ON_TARGET_XY_TOL + 0.05
        min_cube_separation = 0.06

        def sample_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def far_from_patches(xy: np.ndarray) -> bool:
            return (np.linalg.norm(xy - p1_xy) > min_patch_clearance) and (np.linalg.norm(xy - p2_xy) > min_patch_clearance)

        # Sample cube 1
        for _ in range(500):
            c1_xy = sample_xy()
            if far_from_patches(c1_xy):
                break
        else:
            # Fallback deterministic unsolved point (still within bounds, far in Y)
            c1_xy = np.array([0.45, 0.25], dtype=float)

        # Sample cube 2 with separation from cube 1
        for _ in range(500):
            c2_xy = sample_xy()
            if far_from_patches(c2_xy) and (np.linalg.norm(c2_xy - c1_xy) > min_cube_separation):
                break
        else:
            c2_xy = np.array([0.65, -0.25], dtype=float)

        cube1_pos = np.array([c1_xy[0], c1_xy[1], z], dtype=float)
        cube2_pos = np.array([c2_xy[0], c2_xy[1], z], dtype=float)

        # Slight random yaw for variety; keep upright
        def yaw_quat(yaw: float) -> np.ndarray:
            q = Quaternion(axis=[0, 0, 1], angle=float(yaw))
            # Return (x, y, z, w) to match existing usage
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        self.colored_cube_1.set_pose(position=cube1_pos, quat=yaw_quat(rng.uniform(-np.pi, np.pi)))
        self._maybe_forward_physics()

        self.colored_cube_2.set_pose(position=cube2_pos, quat=yaw_quat(rng.uniform(-np.pi, np.pi)))
        self._maybe_forward_physics()

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # Keep rng "used" deterministically (avoid lint / keep pattern stable if extended)
        _ = rng.random()

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _target_collider(self, patch: GoalPatch):
        # Prefer .collider when present (more reliable directionality).
        return patch.collider if hasattr(patch, "collider") else patch

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _is_cube_on_patch(self, cube: ColoredCube, patch: GoalPatch) -> bool:
        # Reliable pattern: cube.is_colliding(patch.collider) when available
        if not cube.is_colliding(self._target_collider(patch)):
            return False

        cube_pos = cube.body.get_position()
        patch_pos = patch.body.get_position() if hasattr(patch, "body") else patch.get_position()

        # Enforce precise placement tolerance (XY)
        if self._xy_dist(cube_pos, patch_pos) > self._ON_TARGET_XY_TOL:
            return False

        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "goal_patch_1") or not hasattr(self, "goal_patch_2"):
            return False

        on_1 = self._is_cube_on_patch(self.colored_cube_1, self.goal_patch_1)
        on_2 = self._is_cube_on_patch(self.colored_cube_2, self.goal_patch_2)
        positions_ok = bool(on_1 and on_2)

        # No semantic conditions beyond position for this task
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging (simple, state-based)
        if on_1:
            self._metric_stage(1, True)  # cube1 placed
        if on_2:
            self._metric_stage(4, True)  # cube2 placed
        if released and on_1:
            self._metric_stage(2, True)  # released cube1 after placement
        if released and on_2:
            self._metric_stage(5, True)  # released cube2 after placement

        # If fully solved, mark all stages achieved
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distances to targets (XY) for debugging/scoring
        c1_pos = self.colored_cube_1.body.get_position()
        c2_pos = self.colored_cube_2.body.get_position()
        p1_pos = self.goal_patch_1.body.get_position()
        p2_pos = self.goal_patch_2.body.get_position()

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube1_to_patch1_xy": self._xy_dist(c1_pos, p1_pos),
                "cube2_to_patch2_xy": self._xy_dist(c2_pos, p2_pos),
            },
            pose_error={
                "cube1_patch1_xy_error": self._xy_dist(c1_pos, p1_pos),
                "cube2_patch2_xy_error": self._xy_dist(c2_pos, p2_pos),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})