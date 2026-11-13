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


class AlternatingRedBlueCubeRow(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes into a single straight row along the +Y axis (left direction),
    such that the color pattern alternates red, blue, red, blue when sorted by Y from
    right (-Y) to left (+Y).

    Sequential Sub-Goals:
        1. Grasp any cube to begin arranging the row.
        2. Place a cube to establish one endpoint of the intended +Y row within the workspace.
        3. Adjust cubes so their X coordinates are approximately aligned (row extends along Y).
        4. Reorder cubes along Y so that sorting by Y gives red, blue, red, blue (right to left).
        5. Fine-tune placements so neighboring cubes have roughly even spacing along Y without stacking.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (from specification) ---
    _WORKSPACE_X = (0.40, 0.70)
    _WORKSPACE_Y = (-0.25, 0.25)

    _ROW_X_MAX_DEVIATION_M = 0.03
    _NEIGHBOR_DY_M = 0.06
    _NEIGHBOR_DY_TOL_M = 0.03

    _TABLE_Z_CUBE = 0.97  # ColoredCube resting Z on table
    _EXPECTED_SEQUENCE = ["red", "blue", "red", "blue"]  # by sorted Y ascending (right->left)

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create four colored cubes (ONLY asset allowed)
        self.cubes = [ColoredCube(self._mojo) for _ in range(4)]

        # Predefine goal row positions (ordered by Y ascending: right (-Y) -> left (+Y))
        # Spacing ~0.06m, centered near y=0, all at fixed x for a straight +Y row.
        self._goal_x = 0.55
        self._goal_ys = [-0.09, -0.03, 0.03, 0.09]  # ascending
        self._goal_positions = [
            np.array([self._goal_x, y, self._TABLE_Z_CUBE], dtype=float) for y in self._goal_ys
        ]
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # Assign colors to cube *instances* (must also be done in _on_reset)
        # (Cube identity doesn't matter for success; success checks by Y-sorted colors.)
        for cube, col in zip(self.cubes, self._EXPECTED_SEQUENCE):
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

        # 2. Deterministic randomization
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions within workspace bounds on the table.
        #    We guarantee unsolved by forcing X misalignment beyond _ROW_X_MAX_DEVIATION_M.
        quat = self._goal_quat
        z = self._TABLE_Z_CUBE

        # Two cubes on the low-X band, two on the high-X band -> x_aligned will be False.
        x_low = float(rng.uniform(self._WORKSPACE_X[0], 0.50))
        x_high = float(rng.uniform(0.60, self._WORKSPACE_X[1]))
        xs = [x_low, x_high, x_low, x_high]
        rng.shuffle(xs)

        # Sample Y positions with mild separation to reduce initial overlaps.
        ys = []
        min_sep = 0.05
        for _ in range(4):
            for _attempt in range(200):
                y = float(rng.uniform(self._WORKSPACE_Y[0], self._WORKSPACE_Y[1]))
                if all(abs(y - y_prev) >= min_sep for y_prev in ys):
                    ys.append(y)
                    break
            else:
                # Fallback: deterministic spread if rejection sampling fails
                ys = list(np.linspace(self._WORKSPACE_Y[0] + 0.04, self._WORKSPACE_Y[1] - 0.04, 4))
                break

        for cube, col, x, y in zip(self.cubes, self._EXPECTED_SEQUENCE, xs, ys):
            cube.set_color(col)
            cube.set_pose(position=np.array([x, y, z], dtype=float), quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
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
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Gather state ---
        positions = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)
        zs = np.array([p[2] for p in positions], dtype=float)

        # Workspace bounds (tight bounds from task spec)
        in_bounds = bool(
            np.all((xs >= self._WORKSPACE_X[0]) & (xs <= self._WORKSPACE_X[1]))
            and np.all((ys >= self._WORKSPACE_Y[0]) & (ys <= self._WORKSPACE_Y[1]))
        )

        # On table & not stacked (approx): colliding with table top and near correct Z
        # (Table collision is reliable via self.table.table_top)
        on_table = True
        for c, z in zip(self.cubes, zs):
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break
            if abs(float(z) - self._TABLE_Z_CUBE) > 0.03:
                on_table = False
                break

        # Row constraint: aligned along Y, so X coordinates should be similar
        x_mean = float(np.mean(xs)) if len(xs) else 0.0
        x_aligned = bool(np.max(np.abs(xs - x_mean)) <= self._ROW_X_MAX_DEVIATION_M)

        # Sort by Y ascending (right -> left)
        order = list(np.argsort(ys))
        sorted_cubes = [self.cubes[i] for i in order]
        sorted_ys = ys[order]

        # Spacing constraint along Y (neighbor distances)
        min_dy = self._NEIGHBOR_DY_M - self._NEIGHBOR_DY_TOL_M
        max_dy = self._NEIGHBOR_DY_M + self._NEIGHBOR_DY_TOL_M
        spacing_ok = True
        neighbor_dys = []
        for i in range(len(sorted_ys) - 1):
            dy = float(sorted_ys[i + 1] - sorted_ys[i])
            neighbor_dys.append(dy)
            if not (min_dy <= dy <= max_dy):
                spacing_ok = False
                break

        # Semantic check: alternating color pattern in Y-ascending order
        color_seq = [c.get_color() for c in sorted_cubes]
        semantic_ok = bool(color_seq == self._EXPECTED_SEQUENCE)

        # Released gripper check
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Overall position/arrangement check (excluding semantic)
        positions_ok = bool(in_bounds and on_table and x_aligned and spacing_ok)

        # --- Subgoal staging (coarse, current-state based) ---
        grasped_any = any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )
        # Endpoint placed: any cube close to either endpoint Y (within 4cm) and on table
        endpoint_ys = (self._goal_ys[0], self._goal_ys[-1])
        endpoint_placed = False
        if on_table:
            for y in ys:
                if min(abs(float(y) - endpoint_ys[0]), abs(float(y) - endpoint_ys[1])) <= 0.04:
                    endpoint_placed = True
                    break

        self._metric_stage(0, bool(grasped_any or positions_ok or semantic_ok))
        self._metric_stage(1, bool(endpoint_placed or positions_ok))
        self._metric_stage(2, bool(x_aligned))
        self._metric_stage(3, bool(semantic_ok))
        self._metric_stage(4, bool(spacing_ok and on_table))

        # Final success condition
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize ---
        # Target distances: compare Y-sorted cubes to Y-ordered goal positions
        target_distance = {}
        pose_error = {}
        for rank, cube in enumerate(sorted_cubes):
            goal_pos = self._goal_positions[rank]
            cur_pos = cube.body.get_position()
            target_distance[f"cube_sorted_rank_{rank}"] = float(np.linalg.norm(np.array(cur_pos) - goal_pos))
            # Orientation is irrelevant here; report position error magnitude as pose_error proxy.
            pose_error[f"cube_sorted_rank_{rank}"] = float(np.linalg.norm(np.array(cur_pos) - goal_pos))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})