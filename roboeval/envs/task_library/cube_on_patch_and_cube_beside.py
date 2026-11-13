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


class PatchStackAndPlace(RoboEvalEnv, MetricRolloutEval):
    """Place red on the green patch and blue beside it.

Success Criteria:
Red’s center may be anywhere inside the patch with a 2 cm inset from its edges. Place blue outside the patch, within 6 cm of an edge. Keep both cubes resting on the table and release them."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (from spec) ---
    # _PATCH_SIZE is passed straight to GoalPatch, which reads it as HALF-extents,
    # so the drawn patch is 20 cm across. Requiring the red cube within 3 cm of the
    # patch CENTRE therefore rejected cubes sitting plainly on the patch. Accept
    # anywhere on it, inset a little so a corner-clipping cube still fails.
    _PATCH_INSET = 0.02
    _PATCH_SIZE = (0.10, 0.10)  # HALF-extents in (x, y); the patch is 0.20 m across

    # A reasonable "adjacent" band for "beside" (spec doesn't provide an edge distance limit)
    _BLUE_MAX_EDGE_DISTANCE = 0.06  # meters from cube center to nearest patch edge in XY

    # "Beside" means outside the patch but within the edge band above. A flat centre
    # -distance cap of 0.12 m left only a 2 cm ring outside a 0.10 m half-extent
    # patch, with nothing valid near the corners (a corner alone is 0.141 m out).
    # Derive the cap from the patch diagonal plus the band so the whole ring works.
    _BLUE_MAX_CENTER_DIST = float(np.hypot(0.10, 0.10) + 0.06)  # ~0.20 m

    _NUM_SUBGOALS = 8

    def _initialize_env(self):
        # Props from preset
        self.table = self._preset.get_props(Table)[0]

        # --- Required assets only: ColoredCube, GoalPatch ---
        self.goal_patch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="target")

        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_blue]

        # Metrics init (FULL signature required)
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
        # 1) Re-init metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG (required pattern; randomization may be applied by pipeline)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")

        # 3) Place objects in RANDOM (unsolved) state within workspace bounds
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        Z_BASE = 0.97

        def sample_xy():
            return np.array([rng.uniform(X_MIN, X_MAX), rng.uniform(Y_MIN, Y_MAX)], dtype=float)

        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return Quaternion(axis=[0, 0, 1], angle=yaw).elements

        # Randomize patch pose (position only; GoalPatch has no set_pose)
        patch_xy = sample_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], 0.951], dtype=float)
        self.goal_patch.body.set_position(patch_pos)
        self._forward_physics()

        # Sample cube positions that are guaranteed NOT to satisfy success conditions:
        # - Red: place far enough from patch so it cannot be centered/on patch.
        # - Blue: place far enough from patch so it cannot be "beside" (fails center-distance <= 0.12).
        def sample_far_from_patch(min_dist_xy: float) -> np.ndarray:
            for _ in range(200):
                xy = sample_xy()
                if float(np.linalg.norm(xy - patch_xy)) >= float(min_dist_xy):
                    return xy
            # Fallback: deterministic corner far from patch
            corners = np.array(
                [[X_MIN, Y_MIN], [X_MIN, Y_MAX], [X_MAX, Y_MIN], [X_MAX, Y_MAX]], dtype=float
            )
            d = np.linalg.norm(corners - patch_xy[None, :], axis=1)
            return corners[int(np.argmax(d))]

        red_xy = sample_far_from_patch(min_dist_xy=0.15)
        blue_xy = sample_far_from_patch(min_dist_xy=0.20)

        red_pos = np.array([red_xy[0], red_xy[1], Z_BASE], dtype=float)
        blue_pos = np.array([blue_xy[0], blue_xy[1], Z_BASE], dtype=float)

        self.cube_red.set_pose(position=red_pos, quat=yaw_quat())
        self._forward_physics()

        self.cube_blue.set_pose(position=blue_pos, quat=yaw_quat())
        self._forward_physics()

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ------------------------
    # Helpers
    # ------------------------
    def _forward_physics(self):
        """Best-effort physics forward (requested by task constraints)."""
        try:
            if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
                self._mojo.physics.forward()
        except Exception:
            pass

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _rect_point_distance_xy(self, p_xy: np.ndarray, rect_center_xy: np.ndarray, half_extents: np.ndarray) -> float:
        """Euclidean distance from point to axis-aligned rectangle in XY (0 if inside)."""
        d = np.abs(p_xy - rect_center_xy) - half_extents
        d = np.maximum(d, 0.0)
        return float(np.linalg.norm(d))

    def _is_on_patch(self, cube: ColoredCube) -> bool:
        return cube_on_patch(cube, self.goal_patch, self._PATCH_SIZE, self._PATCH_INSET)

    def _is_centered_on_patch(self, cube: ColoredCube, xy_tol: float) -> bool:
        patch_pos = self.goal_patch.body.get_position()
        cube_pos = cube.body.get_position()
        return bool(np.all(np.abs(cube_pos[:2] - patch_pos[:2]) <= float(xy_tol) + 1e-9))

    def _is_beside_patch(self, cube: ColoredCube) -> bool:
        patch_pos = self.goal_patch.body.get_position()
        cube_pos = cube.body.get_position()

        # Must be on the table (collision can be flaky if solver leaves a tiny gap; add Z fallback)
        on_table_collision = bool(cube.is_colliding(self.table.table_top))
        on_table_z = abs(float(cube_pos[2]) - 0.97) <= 0.03  # ColoredCube rest_z ± tolerance
        on_table = bool(on_table_collision or on_table_z)

        # Must NOT be on the patch:
        # Use a robust geometric "outside" check plus the task's on-patch predicate (which includes centering).
        hx = self._PATCH_SIZE[0]
        hy = self._PATCH_SIZE[1]
        dx = float(cube_pos[0] - patch_pos[0])
        dy = float(cube_pos[1] - patch_pos[1])

        inside_patch_xy = (abs(dx) <= hx - 1e-3) and (abs(dy) <= hy - 1e-3)
        not_on_patch = (not inside_patch_xy) and (not self._is_on_patch(cube))

        # Must be immediately beside: near the patch boundary and within max center distance
        center_dist = self._xy_dist(cube_pos, patch_pos)
        within_center_dist = center_dist <= float(self._BLUE_MAX_CENTER_DIST)

        edge_dist = self._rect_point_distance_xy(
            p_xy=cube_pos[:2],
            rect_center_xy=patch_pos[:2],
            half_extents=np.array([hx, hy]),
        )
        adjacent = edge_dist <= float(self._BLUE_MAX_EDGE_DISTANCE)

        return bool(on_table and not_on_patch and within_center_dist and adjacent)

    # ------------------------
    # Success + subgoal staging
    # ------------------------
    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False

        patch_pos = self.goal_patch.body.get_position()
        red_pos = self.cube_red.body.get_position()
        blue_pos = self.cube_blue.body.get_position()

        # Goal conditions
        red_on_patch = self._is_on_patch(self.cube_red)
        red_centered = self._is_centered_on_patch(
            self.cube_red, self._PATCH_SIZE[0] - self._PATCH_INSET
        )
        red_ok = bool(red_on_patch and red_centered)

        blue_ok = self._is_beside_patch(self.cube_blue)

        # Released gripper condition (no cube held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(red_ok and blue_ok)
        semantic_ok = True  # ColoredCube task; color is enforced by set_color() at reset.

        # --- Subgoal staging (best-effort, based on observable states) ---
        holding_red = any(self.robot.is_gripper_holding_object(self.cube_red, s) for s in self.robot.grippers)
        holding_blue = any(self.robot.is_gripper_holding_object(self.cube_blue, s) for s in self.robot.grippers)

        red_above_patch = (self._xy_dist(red_pos, patch_pos) <= 0.05) and (float(red_pos[2]) >= 1.02)
        blue_near_patch = (self._xy_dist(blue_pos, patch_pos) <= self._BLUE_MAX_CENTER_DIST) and (float(blue_pos[2]) >= 1.02)

        # 1-4 (red cube)
        if holding_red:
            self._metric_stage(0, True)
        if red_above_patch:
            self._metric_stage(1, True)
        if red_ok:
            self._metric_stage(2, True)
        if red_ok and (not holding_red):
            self._metric_stage(3, True)

        # 5-8 (blue cube)
        if holding_blue:
            self._metric_stage(4, True)
        if blue_near_patch:
            self._metric_stage(5, True)
        if blue_ok:
            self._metric_stage(6, True)
        if blue_ok and (not holding_blue):
            self._metric_stage(7, True)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call)
        hx = self._PATCH_SIZE[0]
        hy = self._PATCH_SIZE[1]
        blue_edge_dist = self._rect_point_distance_xy(
            p_xy=blue_pos[:2],
            rect_center_xy=patch_pos[:2],
            half_extents=np.array([hx, hy]),
        )

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_patch_xy": float(np.linalg.norm((red_pos - patch_pos)[:2])),
                "blue_to_patch_center_xy": float(np.linalg.norm((blue_pos - patch_pos)[:2])),
                "blue_to_patch_edge_xy": float(blue_edge_dist),
            },
            pose_error={
                "red_centering_error_xy": float(np.linalg.norm((red_pos - patch_pos)[:2])),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})