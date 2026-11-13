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


class ColoredCubeRowPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes onto a long green target patch and arrange them into a single straight row.
    The row must be aligned along the length (long axis) of the patch, and all cubes must rest fully
    within the patch boundaries.

    Sequential Sub-Goals:
        1. Move all three cubes onto the green goal patch.
        2. Align the three cubes into a straight line along the patch long axis.
        3. Adjust spacing so the cubes are roughly evenly spaced.
        4. Release all cubes (no cube held by any gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal-condition tolerances from task spec
    _COLLINEARITY_TOL_M = 0.02
    _SPACING_TOL_M = 0.03
    _WITHIN_PATCH_MARGIN_M = 0.01

    # Asset geometry
    _CUBE_HALF = 0.02  # 4cm cube
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Long green goal patch (only allowed asset type: GoalPatch)
        # Make it "long" by using a rectangular size; we treat the larger dimension as the long axis.
        self.goal_patch = GoalPatch(
            self._mojo, color="green", size=(0.08, 0.28), name="goal_patch_green_long"
        )
        self._patch_size = (0.08, 0.28)  # (size_x, size_y) in patch-local frame

        # Three colored cubes (only allowed asset type: ColoredCube)
        self.cubes = [ColoredCube(self._mojo) for _ in range(3)]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "yellow"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

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

        # 2) Deterministic RNG (pipeline may add randomization later)
        rng = np.random.default_rng(self.seed)  # noqa: F841

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3) Place patch deterministically, but randomize cubes into UNSOLVED poses in workspace bounds
        patch_pos = np.array([0.55, 0.0, self._PATCH_Z])
        self.goal_patch.body.set_position(patch_pos)
        self._physics_forward()

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z

        # Keep cubes off the patch so _success() is False after reset
        # Conservative "keep-out" radius around patch center in XY to avoid being on/within patch.
        half_x = self._patch_size[0] / 2.0
        half_y = self._patch_size[1] / 2.0
        keepout_r = float(np.hypot(half_x, half_y) + 0.06)  # > patch half-diagonal + buffer

        cube_positions = []
        for i, cube in enumerate(self.cubes):
            # Rejection sample positions within workspace and away from patch + other cubes
            for _ in range(200):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p_xy = np.array([x, y], dtype=float)

                if np.linalg.norm(p_xy - patch_pos[:2]) < keepout_r:
                    continue

                ok = True
                for prev in cube_positions:
                    if np.linalg.norm(p_xy - prev[:2]) < 0.06:
                        ok = False
                        break
                if not ok:
                    continue

                pos = np.array([x, y, z], dtype=float)
                cube_positions.append(pos)
                break
            else:
                # Fallback (still off-patch and inside workspace)
                pos = np.array([x_min + 0.01 + 0.03 * i, y_max - 0.01 - 0.03 * i, z], dtype=float)
                cube_positions.append(pos)

            # Random yaw about Z
            yaw = float(rng.uniform(-np.pi, np.pi))
            q_xyzw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

            cube.set_pose(position=cube_positions[i], quat=q_xyzw)
            self._physics_forward()

        # Store goal positions (kept for any external logging; not used by _success())
        self._goal_cube_positions = [p.copy() for p in cube_positions]

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
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False

        patch_pos = self.goal_patch.body.get_position()
        patch_quat_xyzw = self.goal_patch.body.get_quaternion()

        # Determine patch-local coordinates and long axis
        long_is_y = self._patch_size[1] >= self._patch_size[0]

        # 1) Each cube must be on the patch (reliable collider direction) and within patch bounds
        on_patch_flags = []
        within_bounds_flags = []
        local_long = []
        local_short = []

        for cube in self.cubes:
            on_patch = self._is_cube_on_patch(cube, self.goal_patch)
            on_patch_flags.append(on_patch)

            # Local coordinates in patch frame for boundary/alignment checks
            p = cube.body.get_position()
            local = self._world_to_patch_local(p, patch_pos, patch_quat_xyzw)
            lx, ly = float(local[0]), float(local[1])

            long_coord = ly if long_is_y else lx
            short_coord = lx if long_is_y else ly
            local_long.append(long_coord)
            local_short.append(short_coord)

            within_bounds_flags.append(self._is_within_patch_bounds_local(lx, ly))

        all_on_patch = all(on_patch_flags)
        all_within_bounds = all(within_bounds_flags)

        # 2) Aligned along patch long axis (collinearity): short-axis deviation small
        local_short = np.array(local_short, dtype=float)
        short_dev = float(np.max(np.abs(local_short - np.mean(local_short)))) if len(local_short) else 1e9
        aligned = short_dev <= self._COLLINEARITY_TOL_M

        # 3) Roughly even spacing along long axis
        local_long = np.sort(np.array(local_long, dtype=float))
        even_spacing = False
        spacing_err = 1e9
        if len(local_long) == 3:
            g1 = float(local_long[1] - local_long[0])
            g2 = float(local_long[2] - local_long[1])
            spacing_err = abs(g1 - g2)
            # Also enforce minimum practical separation between cube centers
            even_spacing = (spacing_err <= self._SPACING_TOL_M) and (g1 >= 0.05) and (g2 >= 0.05)

        # 4) Released (no cube held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage metrics
        if all_on_patch and all_within_bounds:
            self._metric_stage(0, True)
        if all_on_patch and all_within_bounds and aligned:
            self._metric_stage(1, True)
        if all_on_patch and all_within_bounds and aligned and even_spacing:
            self._metric_stage(2, True)
        if released:
            self._metric_stage(3, True)

        positions_ok = bool(all_on_patch and all_within_bounds)
        arrangement_ok = bool(aligned and even_spacing)
        self._success_check = bool(positions_ok and arrangement_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        target_distance = {
            f"cube_{i+1}_to_patch_center_xy_m": float(
                np.linalg.norm(self.cubes[i].body.get_position()[:2] - patch_pos[:2])
            )
            for i in range(3)
        }
        pose_error = {
            "row_short_axis_deviation_m": float(short_dev),
            "row_spacing_diff_m": float(spacing_err),
        }
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # -----------------------
    # Helpers
    # -----------------------
    def _physics_forward(self):
        # Some builds expose physics.forward(); fall back to a single step.
        try:
            if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
                self._mojo.physics.forward()
            else:
                self._mojo.step()
        except Exception:
            self._mojo.step()

    def _quat_xyzw_to_quaternion(self, q_xyzw: np.ndarray) -> Quaternion:
        # NOTE: only caller passes body.get_quaternion() which is wxyz; read it as wxyz.
        q_wxyz = np.asarray(q_xyzw, dtype=float)
        return Quaternion(q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3])

    def _world_to_patch_local(
        self, world_pos: np.ndarray, patch_pos: np.ndarray, patch_quat_xyzw: np.ndarray
    ) -> np.ndarray:
        q = self._quat_xyzw_to_quaternion(patch_quat_xyzw)
        v = np.asarray(world_pos, dtype=float) - np.asarray(patch_pos, dtype=float)
        # Patch local frame: rotate world vector by inverse patch rotation
        return np.array(q.inverse.rotate(v), dtype=float)

    def _is_within_patch_bounds_local(self, lx: float, ly: float) -> bool:
        half_x = self._patch_size[0] / 2.0
        half_y = self._patch_size[1] / 2.0
        # Require cube footprint fully inside patch with margin
        lim_x = half_x - self._WITHIN_PATCH_MARGIN_M - self._CUBE_HALF
        lim_y = half_y - self._WITHIN_PATCH_MARGIN_M - self._CUBE_HALF
        return (abs(lx) <= lim_x) and (abs(ly) <= lim_y)

    def _is_cube_on_patch(self, cube: ColoredCube, patch: GoalPatch) -> bool:
        # Reliable collision direction: cube against patch.collider
        if not hasattr(patch, "collider"):
            # If collider missing (unexpected), fall back to patch.is_colliding
            return bool(patch.is_colliding(cube))
        return bool(cube.is_colliding(patch.collider))