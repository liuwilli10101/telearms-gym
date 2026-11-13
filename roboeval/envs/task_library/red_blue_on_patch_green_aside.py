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

class DualCubeOnTargetTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place the red and blue cubes together on the green goal patch, and place the green cube
    next to (but not on) the patch.

    Sequential Sub-Goals:
        1. Place the red cube on the green goal patch.
        2. Place the blue cube on the green goal patch.
        3. Ensure red and blue cubes are adjacent while both are on the patch.
        4. Place the green cube adjacent to (but not on) the patch.
        5. Release all cubes.
        6. (Optional stabilization) Keep red cube stably on the patch.
        7. (Optional stabilization) Keep blue cube stably on the patch.
        8. (Optional stabilization) Keep red-blue adjacency stable.
        9. (Optional stabilization) Keep green cube adjacent but not on patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task thresholds (from spec) ---
    _RB_ADJ_MAX_DIST = 0.04   # meters (center-to-center in XY)
    _G_ADJ_PATCH_MAX_DIST = 0.06  # meters (distance from cube center to patch rectangle, XY)
    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Required assets only: ColoredCube, GoalPatch ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube]

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
        # Some backends require an explicit forward after teleporting props.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    @staticmethod
    def _point_to_aabb_distance_2d(p_xy: np.ndarray, center_xy: np.ndarray, half_extents_xy: np.ndarray) -> float:
        """
        Distance from point to an axis-aligned rectangle (AABB) in 2D.
        Returns 0 if point is inside/on rectangle.
        """
        dx = abs(float(p_xy[0] - center_xy[0])) - float(half_extents_xy[0])
        dy = abs(float(p_xy[1] - center_xy[1])) - float(half_extents_xy[1])
        dx = max(dx, 0.0)
        dy = max(dy, 0.0)
        return float(np.hypot(dx, dy))

    def _is_block_on_patch(self, block: ColoredCube, patch: GoalPatch) -> bool:
        """
        Reliable on-target check using collider direction from the BLOCK side.
        Additionally require the block center to be within the patch AABB in XY to
        avoid counting edge-touching adjacency as "on patch".
        """
        if not hasattr(patch, "collider"):
            return False
        if not block.is_colliding(patch.collider):
            return False

        p = block.body.get_position()
        if p is None or len(p) < 3:
            return False
        z = float(p[2])
        if z < 0.955 or z > 1.02:
            return False

        patch_p = patch.body.get_position()
        if patch_p is None or len(patch_p) < 2:
            return False
        patch_xy = patch_p[:2]
        patch_half = np.array([0.05, 0.05])  # size=(0.1,0.1)
        if self._point_to_aabb_distance_2d(p[:2], patch_xy, patch_half) > 1e-6:
            return False

        return True

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

        # 2. Deterministic RNG handle (even if goal placement is deterministic)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        # Workspace bounds
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        cube_z = 0.97
        patch_z = 0.951
        patch_half = np.array([0.05, 0.05])  # size=(0.1, 0.1)

        def rand_yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        def sample_xy() -> np.ndarray:
            return np.array([float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))], dtype=float)

        def far_from_patch(xy: np.ndarray, patch_xy: np.ndarray, margin: float) -> bool:
            # Require cube center to be outside patch AABB by at least `margin` in 2D distance-to-rect sense.
            return self._point_to_aabb_distance_2d(xy, patch_xy, patch_half) > margin

        # 3. Parking pattern: move cubes out of the way before table placement
        park_pos = np.array([0.55, 0.0, 1.8])
        for c in self.cubes:
            c.set_pose(position=park_pos, quat=np.array([0, 0, 0, 1]))
        self._maybe_forward_physics()
        self._mojo.physics.forward()

        # 4. Place patch randomly within workspace and settle
        patch_xy = sample_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], patch_z], dtype=float)
        self.green_goal_patch.body.set_position(patch_pos)
        self._maybe_forward_physics()
        self._mojo.physics.forward()

        # 5. Place cubes in RANDOM, UNSOLVED poses (ensure red/blue NOT on patch; also avoid trivial adjacency)
        #    We enforce: red and blue far from patch so red_on/blue_on are False after reset.
        #    Additionally separate cubes from each other a bit to avoid accidental red-blue adjacency.
        min_cube_sep = 0.06
        far_margin = 0.08  # > 0.06 adjacency-to-rect threshold, ensures NOT adjacent to patch either

        cube_xys = []
        for _ in range(3):
            for _attempt in range(200):
                xy = sample_xy()
                if not far_from_patch(xy, patch_xy, far_margin):
                    continue
                if any(np.linalg.norm(xy - other) < min_cube_sep for other in cube_xys):
                    continue
                cube_xys.append(xy)
                break
            else:
                # Fallback (still within bounds): push away from patch center directionally
                direction = rng.normal(size=2)
                n = np.linalg.norm(direction) + 1e-9
                direction = direction / n
                xy = patch_xy + direction * 0.18
                xy[0] = float(np.clip(xy[0], x_low, x_high))
                xy[1] = float(np.clip(xy[1], y_low, y_high))
                cube_xys.append(xy)

        red_pos = np.array([cube_xys[0][0], cube_xys[0][1], cube_z], dtype=float)
        blue_pos = np.array([cube_xys[1][0], cube_xys[1][1], cube_z], dtype=float)
        green_pos = np.array([cube_xys[2][0], cube_xys[2][1], cube_z], dtype=float)

        self.red_cube.set_pose(position=red_pos, quat=rand_yaw_quat())
        self._maybe_forward_physics()
        self._mojo.physics.forward()

        self.blue_cube.set_pose(position=blue_pos, quat=rand_yaw_quat())
        self._maybe_forward_physics()
        self._mojo.physics.forward()

        self.green_cube.set_pose(position=green_pos, quat=rand_yaw_quat())
        self._maybe_forward_physics()
        self._mojo.physics.forward()

        # 8. Initialize subgoals to False
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
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "green_goal_patch"):
            return False

        patch_xy = self.green_goal_patch.body.get_position()[:2]
        patch_half = np.array([0.05, 0.05])  # since size=(0.1,0.1) and we keep quat identity

        red_p = self.red_cube.body.get_position()
        blue_p = self.blue_cube.body.get_position()
        green_p = self.green_cube.body.get_position()
        patch_p = self.green_goal_patch.body.get_position()

        # Goal conditions:
        red_on = self._is_block_on_patch(self.red_cube, self.green_goal_patch)
        blue_on = self._is_block_on_patch(self.blue_cube, self.green_goal_patch)

        rb_adj = False
        if red_on and blue_on:
            rb_adj = (self._xy_dist(red_p, blue_p) <= self._RB_ADJ_MAX_DIST)

        # Green cube adjacent to patch (but not on patch)
        green_on_patch = self._is_block_on_patch(self.green_cube, self.green_goal_patch)
        green_to_patch_rect = self._point_to_aabb_distance_2d(green_p[:2], patch_xy, patch_half)
        green_adj_not_on = (not green_on_patch) and (green_to_patch_rect <= self._G_ADJ_PATCH_MAX_DIST)

        positions_ok = bool(red_on and blue_on and rb_adj and green_adj_not_on)

        # No extra semantic checks (ColoredCube only)
        semantic_ok = True

        # Released check
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (best-effort from observable predicates only)
        self._metric_stage(0, bool(red_on))
        self._metric_stage(1, bool(blue_on))
        self._metric_stage(2, bool(rb_adj))
        self._metric_stage(3, bool(green_adj_not_on))
        self._metric_stage(4, bool(released))
        self._metric_stage(5, bool(red_on and released))
        self._metric_stage(6, bool(blue_on and released))
        self._metric_stage(7, bool(rb_adj and released))
        self._metric_stage(8, bool(green_adj_not_on and released))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # Target distance: report distances relevant to goal conditions
        target_distance = {
            "red_to_patch_center_xy": self._xy_dist(red_p, patch_p),
            "blue_to_patch_center_xy": self._xy_dist(blue_p, patch_p),
            "red_blue_xy": self._xy_dist(red_p, blue_p),
            "green_to_patch_rect_xy": float(green_to_patch_rect),
        }
        pose_error = {
            "patch_quat_error": 0.0,  # patch kept fixed, identity
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