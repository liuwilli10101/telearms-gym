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


class DualPatchCubePairPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Put two cubes on each of the two patches.

Success Criteria:
Every cube center must be inside one of the drawn squares with a 2 cm inset, two per patch, all flat at table height and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace / object constants
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    # GoalPatch sizes are HALF-extents. The old (0.10, 0.10) drew two 20 cm squares
    # and was then halved again to (0.05, 0.05), so the code described a patch a
    # quarter of the drawn area: the reset scattered cubes onto the visible targets
    # while believing it had kept them off, and the XY check disagreed with what the
    # operator could see. Two 4 cm cubes need nowhere near 20 cm, and squares that
    # wide left the rejection sampler almost no free table, so the patch is now
    # drawn at the size the task actually needs.
    _PATCH_SIZE = (0.06, 0.06)  # half-extents -> 12 cm x 12 cm drawn
    _PATCH_HALF = _PATCH_SIZE

    # A cube counts as on a patch once its centre is inside the drawn square, with a
    # small inset so clipping a corner does not count.
    _ON_PATCH_INSET = 0.02
    _ON_TABLE_TOL_Z = 0.03

    _NUM_SUBGOALS = 4

    def _forward_physics(self):
        # Some backends expose physics.forward(); keep it defensive.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Goal patches (fixed, separated)
        self.goal_patch_1 = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="goal_patch_1")
        self.goal_patch_2 = GoalPatch(self._mojo, color="blue", size=self._PATCH_SIZE, name="goal_patch_2")
        self.goal_patches = [self.goal_patch_1, self.goal_patch_2]

        # Cubes
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4]

        # REQUIRED: call set_color() in _initialize_env()
        self._cube_colors = ["red", "blue", "green", "yellow"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Nominal patch poses (kept within workspace bounds)
        self._patch_1_pos = np.array([0.55, 0.18, self._PATCH_REST_Z])
        self._patch_2_pos = np.array([0.55, -0.18, self._PATCH_REST_Z])
        self._patch_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Pre-defined "solved" cube offsets on each patch (two per patch, no overlap)
        self._solved_offsets = [
            np.array([0.0, +0.03, 0.0]),
            np.array([0.0, -0.03, 0.0]),
        ]

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

        # REQUIRED: call set_color() in _on_reset()
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # 3. Set goal patches to nominal fixed poses
        # GoalPatch does NOT support set_pose(); use body.set_position().
        self.goal_patch_1.body.set_position(self._patch_1_pos.copy())
        self._forward_physics()
        self.goal_patch_2.body.set_position(self._patch_2_pos.copy())
        self._forward_physics()

        # 4. Randomize cubes to UNSOLVED poses within workspace bounds and NOT on patches
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        patch_xy = [
            self.goal_patch_1.body.get_position()[:2].copy(),
            self.goal_patch_2.body.get_position()[:2].copy(),
        ]
        # Require cube center to be outside the drawn patch extents by a margin.
        avoid_dx = self._PATCH_HALF[0] + 0.04
        avoid_dy = self._PATCH_HALF[1] + 0.04

        # Simple rejection sampling with separation to reduce initial overlaps
        placed_xy = []
        min_sep = 0.055  # conservative separation in XY

        for cube in self.cubes:
            for _ in range(500):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                xy = np.array([x, y], dtype=float)

                # Must NOT be on either patch (outside axis-aligned extents)
                on_any_patch = False
                for pxy in patch_xy:
                    if (abs(xy[0] - pxy[0]) <= avoid_dx) and (abs(xy[1] - pxy[1]) <= avoid_dy):
                        on_any_patch = True
                        break
                if on_any_patch:
                    continue

                # Keep cubes separated
                if any(float(np.linalg.norm(xy - prev)) < min_sep for prev in placed_xy):
                    continue

                placed_xy.append(xy)
                yaw = float(rng.uniform(-np.pi, np.pi))
                # set_pose feeds this to Quaternion(), which reads [w, x, y, z].
                # Re-ordering to xyzw turned a yaw into a 180 deg flip about a tilted
                # axis, so cubes reset lying over instead of upright.
                quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
                cube.set_pose(position=np.array([x, y, z], dtype=float), quat=quat.elements)
                self._forward_physics()
                break
            else:
                # Fallback: place far from patches deterministically (still unsolved).
                # Step along the free front strip so repeated fallbacks do not stack
                # every starved cube on the same spot.
                fallback_xy = np.array([0.42, -0.09 + 0.06 * len(placed_xy)], dtype=float)
                placed_xy.append(fallback_xy)
                cube.set_pose(position=np.array([fallback_xy[0], fallback_xy[1], z], dtype=float), quat=np.array([1.0, 0.0, 0.0, 0.0]))  # identity, [w, x, y, z]
                self._forward_physics()

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_cube_on_patch(self, cube: ColoredCube, patch: GoalPatch) -> bool:
        cube_pos = cube.body.get_position()

        # If cube is not on/near table height, treat as not "on the patch" (prevents counting stacked cubes).
        if abs(cube_pos[2] - self._CUBE_REST_Z) > self._ON_TABLE_TOL_Z:
            return False

        # Exclude floor collisions if present
        if hasattr(self, "_floor") and cube.is_colliding(self._floor):
            return False

        # XY inclusion against the square that is actually drawn. A bare collision
        # with the collider is NOT enough on its own: a cube whose centre sits just
        # outside the edge still touches it, and counting that would put a cube on a
        # patch the operator can see it is not on.
        return cube_on_patch(
            cube, patch, self._PATCH_HALF, self._ON_PATCH_INSET, rest_z=self._CUBE_REST_Z
        )

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "goal_patches") or not hasattr(self, "table"):
            return False

        # Determine which patch (if any) each cube is on
        cube_on_patch = []
        patch_counts = {0: 0, 1: 0}
        for cube in self.cubes:
            on0 = self._is_cube_on_patch(cube, self.goal_patches[0])
            on1 = self._is_cube_on_patch(cube, self.goal_patches[1])

            # If overlapping both (rare), treat as ambiguous and fail for that cube
            if on0 and on1:
                cube_on_patch.append(None)
            elif on0:
                cube_on_patch.append(0)
                patch_counts[0] += 1
            elif on1:
                cube_on_patch.append(1)
                patch_counts[1] += 1
            else:
                cube_on_patch.append(None)

        each_cube_on_some_patch = all(p is not None for p in cube_on_patch)
        counts_ok = (patch_counts[0] == 2) and (patch_counts[1] == 2)
        positions_ok = bool(each_cube_on_some_patch and counts_ok)

        # No additional semantic constraints in this task
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal stages: each cube is on some patch
        for i, p_idx in enumerate(cube_on_patch):
            self._metric_stage(i, bool(p_idx is not None))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distance to nearest patch center (XY) and count errors
        target_distance = {}
        for i, cube in enumerate(self.cubes):
            cpos = cube.body.get_position()
            p0 = self.goal_patches[0].body.get_position()
            p1 = self.goal_patches[1].body.get_position()
            d0 = float(np.linalg.norm((cpos[:2] - p0[:2])))
            d1 = float(np.linalg.norm((cpos[:2] - p1[:2])))
            target_distance[f"cube_{i+1}_to_nearest_patch_xy"] = min(d0, d1)

        pose_error = {
            "goal_patch_1_count_error": float(abs(patch_counts[0] - 2)),
            "goal_patch_2_count_error": float(abs(patch_counts[1] - 2)),
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