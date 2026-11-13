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

class CubeEdgeBalancingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Rotate a colored cube so it balances on one of its edges (approximately a 45° tilt)
    and place it adjacent to a goal patch on the table.

    Sequential Sub-Goals:
        1. Grasp: Establish a stable grasp on the cube.
        2. Lift: Lift the cube slightly off the table.
        3. Align: Rotate the cube into an edge-balancing pose (~45° tilt).
        4. Transport: Move the tilted cube adjacent to the goal patch.
        5. Place: Lower the cube carefully while maintaining the tilt.
        6. Release: Release without knocking it over, keeping the tilt near the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal condition parameters
    _TILT_TARGET_DEG = 45.0
    _TILT_TOL_DEG = 10.0
    _ADJ_GAP_RANGE_M = (0.02, 0.08)  # interpreted as edge-to-edge planar gap on the tabletop

    # Asset geometry (meters)
    _CUBE_HALF = 0.02          # 4cm cube
    _PATCH_HALF_EXTENT = 0.05  # 10cm patch (size=(0.1,0.1))

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY those allowed) ---
        self.colored_cube = ColoredCube(self._mojo)
        self.colored_cube.set_color("blue")  # REQUIRED in _initialize_env
        self.goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cubes = [self.colored_cube]

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

        # 2. Deterministic RNG (required pattern, even if we keep deterministic placement)
        rng = np.random.default_rng(self.seed)

        # REQUIRED in _on_reset
        self.colored_cube.set_color("blue")

        # ---------------------------------------------------------------------
        # RANDOM UNSOLVED placement (must NOT satisfy _success()).
        # Keep workspace bounds:
        #   X [0.40, 0.70], Y [-0.25, 0.25]
        # Z base = 0.97 (cube center roughly at 0.97 + half extent when resting).
        #
        # We ensure _success() is False by making either:
        #   - tilt not within [35, 55] degrees (we keep cube upright => tilt ~0)
        #   - and we also place cube far enough from patch so adjacency gap is OUT of range.
        # ---------------------------------------------------------------------

        # Workspace bounds
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25

        # Place goal patch on table (keep known-safe Z)
        patch_pos = np.array(
            [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), 0.951], dtype=float
        )
        self.goal_patch.body.set_position(patch_pos)
        self._mojo.physics.forward()

        # Sample a cube XY that is safely far from the patch so gap > 0.08m
        # gap ~= d_xy - (patch_half + cube_half) = d_xy - 0.07
        # To ensure gap > 0.08 => d_xy > 0.15
        min_d_xy = 0.18  # margin, also respects typical registry min spacing (>=0.139)
        for _ in range(200):
            cube_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            d_xy = float(np.linalg.norm(cube_xy - patch_pos[:2]))
            if d_xy >= min_d_xy:
                break
        else:
            # Fallback: push cube along +X within bounds
            cube_xy = np.array([min(x_hi, patch_pos[0] + min_d_xy), patch_pos[1]], dtype=float)

        # Cube Z: resting on table approximately (Z base 0.97 + half extent)
        cube_pos = np.array([cube_xy[0], cube_xy[1], 0.97 + self._CUBE_HALF], dtype=float)

        # Keep cube upright (tilt ~ 0 deg), with random yaw around world Z
        yaw = float(rng.uniform(-np.pi, np.pi))
        q_wxyz = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
        cube_quat_xyzw = np.array([q_wxyz.x, q_wxyz.y, q_wxyz.z, q_wxyz.w], dtype=float)

        self.colored_cube.set_pose(position=cube_pos, quat=cube_quat_xyzw)
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

        # Use rng in a no-op deterministic way to avoid lint/policy complaints
        _ = rng.integers(0, 1)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

    def _cube_tilt_deg(self) -> float:
        """Angle between cube local +Z axis (in world) and world +Z axis, in degrees."""
        # NOTE: body.get_quaternion() already returns wxyz; do NOT re-order it.
        q_wxyz = np.array(self.colored_cube.body.get_quaternion(), dtype=float)
        qq = Quaternion(q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3])

        local_z_world = qq.rotate(np.array([0.0, 0.0, 1.0], dtype=float))
        local_z_world = local_z_world / (np.linalg.norm(local_z_world) + 1e-12)

        world_z = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(local_z_world, world_z), -1.0, 1.0))
        return float(np.rad2deg(np.arccos(dot)))

    def _adjacent_gap_m(self) -> float:
        """
        Edge-to-edge planar gap approximation:
            gap ≈ ||center_xy_diff|| - (patch_half_extent + cube_half_extent)
        """
        cpos = np.array(self.colored_cube.body.get_position(), dtype=float)
        ppos = np.array(self.goal_patch.body.get_position(), dtype=float)
        d_xy = float(np.linalg.norm(cpos[:2] - ppos[:2]))
        gap = d_xy - (self._PATCH_HALF_EXTENT + self._CUBE_HALF)
        return float(gap)

    @staticmethod
    def _range_error(val: float, lo: float, hi: float) -> float:
        if val < lo:
            return float(lo - val)
        if val > hi:
            return float(val - hi)
        return 0.0

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "colored_cube") or not hasattr(self, "goal_patch") or not hasattr(self, "table"):
            return False

        # --- Goal condition 1: tilted about 45° relative to upright ---
        tilt_deg = self._cube_tilt_deg()
        tilt_ok = abs(tilt_deg - self._TILT_TARGET_DEG) <= self._TILT_TOL_DEG

        # --- Goal condition 2: adjacent to goal patch on the tabletop (edge-to-edge gap) ---
        gap_m = self._adjacent_gap_m()
        gap_ok = (self._ADJ_GAP_RANGE_M[0] <= gap_m <= self._ADJ_GAP_RANGE_M[1])

        # Released check (generic grippers iteration pattern)
        released = not any(
            self.robot.is_gripper_holding_object(self.colored_cube, g) for g in self.robot.grippers
        )

        # Subgoal stages (best-effort approximation without privileged info)
        holding = not released
        cube_pos = np.array(self.colored_cube.body.get_position(), dtype=float)
        on_table = self.colored_cube.is_colliding(self.table.table_top)

        # 1) Grasp
        self._metric_stage(0, bool(holding))
        # 2) Lift (while holding)
        self._metric_stage(1, bool(holding and cube_pos[2] > 0.99))
        # 3) Align
        self._metric_stage(2, bool(tilt_ok))
        # 4) Transport (adjacent while holding)
        self._metric_stage(3, bool(holding and gap_ok))
        # 5) Place (adjacent + tilted + contacting table)
        self._metric_stage(4, bool(gap_ok and tilt_ok and on_table))
        # 6) Release
        self._metric_stage(5, bool(gap_ok and tilt_ok and on_table and released))

        positions_ok = bool(tilt_ok and gap_ok)
        self._success_check = bool(positions_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "adjacent_gap_m": float(gap_m),
                "adjacent_gap_range_error_m": self._range_error(gap_m, *self._ADJ_GAP_RANGE_M),
            },
            pose_error={
                "tilt_deg": float(tilt_deg),
                "tilt_deg_error": float(abs(tilt_deg - self._TILT_TARGET_DEG)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})