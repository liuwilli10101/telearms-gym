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


class CubeStackAndPlace(RoboEvalEnv, MetricRolloutEval):
    """Build a two-cube stack on the green patch and place a third cube to the patch's right.

Success Criteria:
- The bottom cube is on the green patch within the patch XY tolerance.
- The top cube is stacked on the bottom cube with contact, no table contact, and correct XY/Z alignment.
- The side cube is right of the patch, X-aligned with it, on the table, not on the patch, and no gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants (from spec) ---
    _PATCH_SIZE = (0.1, 0.1)

    _BOTTOM_ON_PATCH_XY_TOL = 0.03
    _TOP_ON_BOTTOM_XY_TOL = 0.02

    # "Right of patch" in this environment means NEGATIVE Y direction.
    _SIDE_RIGHT_DIST_RANGE = (0.06, 0.14)  # meters (patch_y - cube_y)
    _SIDE_X_ALIGNMENT_TOL = 0.06  # meters (|cube_x - patch_x|)

    _N_SUBGOALS = 10

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Objects (ONLY allowed assets)
        self.goal_patch_green = GoalPatch(
            self._mojo, color="green", size=self._PATCH_SIZE, name="target"
        )

        self.cube_bottom = ColoredCube(self._mojo)
        self.cube_top = ColoredCube(self._mojo)
        self.cube_side = ColoredCube(self._mojo)
        self.cubes = [self.cube_bottom, self.cube_top, self.cube_side]

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        # (Initial colors will be deterministically re-set on reset as well.)
        self.cube_bottom.set_color("red")
        self.cube_top.set_color("blue")
        self.cube_side.set_color("yellow")

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        # Use seed-deterministic distinct colors for the three cubes.
        palette = np.array(["red", "blue", "yellow", "purple", "orange"], dtype=object)
        chosen = rng.choice(palette, size=3, replace=False).tolist()
        self.cube_bottom.set_color(chosen[0])
        self.cube_top.set_color(chosen[1])
        self.cube_side.set_color(chosen[2])

        # 4. Place objects in RANDOM unsolved positions (ensure _success() is False)
        # Workspace bounds
        X_LO, X_HI = 0.40, 0.70
        Y_LO, Y_HI = -0.25, 0.25
        Z_BASE = 0.97

        def sample_xy():
            return np.array([rng.uniform(X_LO, X_HI), rng.uniform(Y_LO, Y_HI)], dtype=float)

        def far_enough(xy, others, min_dist):
            return all(np.linalg.norm(xy - o) >= min_dist for o in others)

        # Randomize patch on table (fixed z)
        patch_xy = sample_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], 0.951], dtype=float)
        self.goal_patch_green.body.set_position(patch_pos)
        self._physics_forward_if_available()

        # Sample cube positions (on table, not stacked, not on patch, not in side-goal region)
        min_sep = 0.08  # keep cubes from colliding/stacking at reset
        margin_from_patch = self._BOTTOM_ON_PATCH_XY_TOL + 0.06  # keep "bottom_on_patch" false robustly

        # Bottom cube: ensure not on patch
        for _ in range(200):
            bottom_xy = sample_xy()
            if np.linalg.norm(bottom_xy - patch_xy) > margin_from_patch:
                break
        else:
            bottom_xy = np.clip(patch_xy + np.array([0.12, 0.12]), [X_LO, Y_LO], [X_HI, Y_HI])

        used = [bottom_xy]

        # Top cube: ensure not colliding/aligned with bottom cube (no stack)
        for _ in range(200):
            top_xy = sample_xy()
            if far_enough(top_xy, used, min_sep) and (np.linalg.norm(top_xy - bottom_xy) > (self._TOP_ON_BOTTOM_XY_TOL + 0.06)):
                break
        else:
            top_xy = np.clip(bottom_xy + np.array([0.12, -0.12]), [X_LO, Y_LO], [X_HI, Y_HI])
        used.append(top_xy)

        # Side cube: explicitly avoid satisfying "right of patch" condition
        # Avoid both: dy_right in range AND x aligned within tolerance.
        for _ in range(400):
            side_xy = sample_xy()
            if not far_enough(side_xy, used, min_sep):
                continue
            dy_right = float(patch_xy[1] - side_xy[1])
            x_aligned = abs(float(side_xy[0] - patch_xy[0])) <= self._SIDE_X_ALIGNMENT_TOL
            in_right_dist = (dy_right >= self._SIDE_RIGHT_DIST_RANGE[0]) and (dy_right <= self._SIDE_RIGHT_DIST_RANGE[1])
            if not (x_aligned and in_right_dist):
                # also keep it off the patch robustly
                if np.linalg.norm(side_xy - patch_xy) > (self._BOTTOM_ON_PATCH_XY_TOL + 0.05):
                    break
        else:
            # Force violation: x misaligned strongly
            side_xy = np.array([np.clip(patch_xy[0] + 0.15, X_LO, X_HI), patch_xy[1]], dtype=float)

        # Apply poses (all on table, no stack)
        q = np.array([0, 0, 0, 1], dtype=float)
        self.cube_bottom.set_pose(position=np.array([bottom_xy[0], bottom_xy[1], Z_BASE]), quat=q)
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        self.cube_top.set_pose(position=np.array([top_xy[0], top_xy[1], Z_BASE]), quat=q)
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        self.cube_side.set_pose(position=np.array([side_xy[0], side_xy[1], Z_BASE]), quat=q)
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._N_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch_green"):
            return False

        patch_xy = self.goal_patch_green.body.get_position()[:2]

        # Identity-free: the task asks for a stack on the patch and a third cube
        # beside it, and never says which cube plays which part -- the three colours
        # are even re-drawn at random on every reset, so the roles were literally
        # unknowable. Read the roles off the scene instead: the raised cube is the
        # top of the stack, the cube under it is the base, and the remaining one is
        # the side cube.
        by_height = sorted(self.cubes, key=lambda c: float(c.body.get_position()[2]))
        cube_top = by_height[-1]
        on_table = by_height[:-1]
        cube_bottom = next(
            (c for c in on_table if cube_top.is_colliding(c)),
            min(on_table, key=lambda c: float(
                np.linalg.norm(c.body.get_position()[:2] - cube_top.body.get_position()[:2]))),
        )
        cube_side = next(c for c in on_table if c is not cube_bottom)

        bottom_xy = cube_bottom.body.get_position()[:2]
        top_xy = cube_top.body.get_position()[:2]
        side_xy = cube_side.body.get_position()[:2]

        bottom_on_patch = self._is_block_on_target(cube_bottom, self.goal_patch_green) and (
            np.linalg.norm(bottom_xy - patch_xy) <= self._BOTTOM_ON_PATCH_XY_TOL
        )

        top_on_bottom = (
            cube_top.is_colliding(cube_bottom)
            and (np.linalg.norm(top_xy - bottom_xy) <= self._TOP_ON_BOTTOM_XY_TOL)
            and (cube_top.body.get_position()[2] > cube_bottom.body.get_position()[2])
            and (not cube_top.is_colliding(self.table.table_top))
        )

        # Right-of check: patch_y - side_y in range and x aligned to patch.
        dy_right = float(patch_xy[1] - side_xy[1])  # positive if cube is to the RIGHT (negative Y)
        side_right_of = (dy_right >= self._SIDE_RIGHT_DIST_RANGE[0]) and (dy_right <= self._SIDE_RIGHT_DIST_RANGE[1])
        side_x_aligned = (abs(float(side_xy[0] - patch_xy[0])) <= self._SIDE_X_ALIGNMENT_TOL)
        side_on_table = cube_side.is_colliding(self.table.table_top)
        side_not_on_patch = not self._is_block_on_target(cube_side, self.goal_patch_green)

        side_ok = bool(side_right_of and side_x_aligned and side_on_table and side_not_on_patch)

        positions_ok = bool(bottom_on_patch and top_on_bottom and side_ok)

        # No semantic constraints for ColoredCube beyond spatial arrangement
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (coarse but monotonic)
        if bottom_on_patch:
            for i in range(0, 3):  # subgoals 1-3
                self._metric_stage(i, True)
        if bottom_on_patch and top_on_bottom:
            for i in range(3, 7):  # subgoals 4-7
                self._metric_stage(i, True)
        if bottom_on_patch and top_on_bottom and side_ok:
            for i in range(7, 10):  # subgoals 8-10
                self._metric_stage(i, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "bottom_to_patch_xy": float(np.linalg.norm(bottom_xy - patch_xy)),
                "top_to_bottom_xy": float(np.linalg.norm(top_xy - bottom_xy)),
                "side_right_distance": float(dy_right),
                "side_x_alignment": float(abs(float(side_xy[0] - patch_xy[0]))),
            },
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # --- Helpers ---
    def _physics_forward_if_available(self):
        # Some backends expose an explicit forward() call; keep it defensive.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _is_block_on_target(self, block, target_patch) -> bool:
        """
        Reliable 'on patch' check:
        GoalPatch collision direction can be unreliable; use target_patch.collider from the BLOCK side.
        """
        if not hasattr(target_patch, "collider"):
            return False
        if not block.is_colliding(target_patch.collider):
            return False
        return True