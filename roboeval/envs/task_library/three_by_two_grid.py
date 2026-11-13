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
from roboeval.envs.props.items import GoalPatch, ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubeGridArrangementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange six colored cubes on the table into a flat 3x2 grid pattern.
    All cubes should be upright and aligned into two rows and three columns
    with consistent spacing.

    Sequential Sub-Goals:
        1. Grasp any one cube to begin placing it into the grid.
        2. Grasp another cube (if applicable) to enable faster placement.
        3. Transport a grasped cube to an intended grid cell location on the table.
        4. Place the cube down upright in its grid cell.
        5. Nudge/realign cubes so rows and columns are straight with consistent spacing.
        6. Release the gripper(s) after final alignment adjustments are complete.
        7. Release the gripper(s) after final alignment adjustments are complete.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from spec)
    _ROWS = 2
    _COLS = 3
    _Z_TABLE_CUBE = 0.97

    _ROW_SPACING_RANGE = (0.04, 0.09)
    _COL_SPACING_RANGE = (0.04, 0.09)
    _ALIGN_TOL = 0.02  # meters

    # Additional robust tolerances
    _Z_TOL = 0.02  # meters: allow slight lift while still "on table"
    _UPRIGHT_MAX_TILT_RAD = np.deg2rad(12.0)  # allow yaw, restrict roll/pitch

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube allowed)
        self.cubes = [ColoredCube(self._mojo) for _ in range(6)]

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
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

    def _make_goal_grid(self, center_xy=np.array([0.55, 0.0]), row_spacing=0.06, col_spacing=0.06):
        """
        Row-major grid positions with:
          - rows along X (back->front),
          - cols along Y (left(+Y)->right(-Y)).
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])

        xs = [cx + (r - (self._ROWS - 1) / 2.0) * row_spacing for r in range(self._ROWS)]
        ys = [cy + ((self._COLS - 1) / 2.0 - c) * col_spacing for c in range(self._COLS)]

        goal_positions = []
        for r in range(self._ROWS):
            for c in range(self._COLS):
                goal_positions.append(np.array([xs[r], ys[c], self._Z_TABLE_CUBE], dtype=float))
        return goal_positions, np.array(xs, dtype=float), np.array(ys, dtype=float)

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

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Keep goal definition (used by some eval pipelines / logging), but DO NOT place cubes in goal.
        row_spacing = float(rng.uniform(self._ROW_SPACING_RANGE[0] + 0.01, self._ROW_SPACING_RANGE[1] - 0.01))
        col_spacing = float(rng.uniform(self._COL_SPACING_RANGE[0] + 0.01, self._COL_SPACING_RANGE[1] - 0.01))
        self._goal_positions, self._goal_row_xs, self._goal_col_ys = self._make_goal_grid(
            center_xy=np.array([0.55, 0.0]), row_spacing=row_spacing, col_spacing=col_spacing
        )

        # Place objects in RANDOM UNSOLVED positions within workspace bounds.
        # To guarantee _success() == False after reset, constrain cubes to (nearly) a single Y-line so
        # inferred column spacing is ~0 and fails the spacing checks.
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        z = float(self._Z_TABLE_CUBE)

        min_pairwise_dist = 0.055  # conservative to avoid overlaps
        y_line_halfwidth = 0.008   # keep all cubes near same Y -> unsolved with high certainty

        positions = []
        for _ in range(2000):
            positions.clear()
            for _i in range(len(self.cubes)):
                x = float(rng.uniform(X_MIN, X_MAX))
                y = float(np.clip(rng.uniform(-y_line_halfwidth, y_line_halfwidth), Y_MIN, Y_MAX))
                p = np.array([x, y, z], dtype=float)

                if all(np.linalg.norm(p[:2] - q[:2]) >= min_pairwise_dist for q in positions):
                    positions.append(p)

            if len(positions) == len(self.cubes):
                break

        if len(positions) != len(self.cubes):
            # Fallback (still unsolved): spread along X, near single Y line
            xs = np.linspace(X_MIN + 0.03, X_MAX - 0.03, len(self.cubes))
            ys = rng.uniform(-y_line_halfwidth, y_line_halfwidth, size=len(self.cubes))
            positions = [np.array([float(x), float(y), z], dtype=float) for x, y in zip(xs, ys)]

        # Random upright yaw for each cube
        for cube, pos in zip(self.cubes, positions):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(position=pos, quat=np.array([q.x, q.y, q.z, q.w], dtype=float))

        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_upright(self, quat_xyzw: np.ndarray) -> bool:
        """
        Upright = cube local +Z axis approximately aligned with world +Z.
        Yaw is unconstrained.
        """
        # NOTE: body.get_quaternion() returns wxyz; read it as wxyz (it is NOT xyzw).
        q = Quaternion(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])  # w,x,y,z
        R = q.rotation_matrix
        local_z_in_world = R[:, 2]
        cos_tilt = float(np.dot(local_z_in_world, np.array([0.0, 0.0, 1.0])))
        cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
        tilt = float(np.arccos(cos_tilt))
        return tilt <= self._UPRIGHT_MAX_TILT_RAD

    def _fit_rows_cols_from_positions(self, positions_xy: np.ndarray):
        """
        Robustly estimate row X-centers (2) and col Y-centers (3) by sorting.
        Returns (row_means_sorted_low_to_high_x, col_means_sorted_high_to_low_y).
        """
        xs = positions_xy[:, 0]
        ys = positions_xy[:, 1]

        # Rows: 2 groups of 3 via x sort
        x_order = np.argsort(xs)
        row0 = x_order[:3]
        row1 = x_order[3:]
        row_means = np.array([np.mean(xs[row0]), np.mean(xs[row1])], dtype=float)
        row_means.sort()  # low X (back) -> high X (front)

        # Cols: 3 groups of 2 via y sort (left is +Y)
        y_order = np.argsort(ys)[::-1]  # high Y (left) -> low Y (right)
        col0 = y_order[0:2]
        col1 = y_order[2:4]
        col2 = y_order[4:6]
        col_means = np.array([np.mean(ys[col0]), np.mean(ys[col1]), np.mean(ys[col2])], dtype=float)

        # Ensure left->right ordering (descending Y)
        col_means = np.sort(col_means)[::-1]
        return row_means, col_means

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Gather poses
        positions = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        quats = np.array([c.body.get_quaternion() for c in self.cubes], dtype=float)
        positions_xy = positions[:, :2]

        # Stage 0/1: any grasped (single-arm robot may only have one gripper; we keep both stages for metric parity)
        any_grasped = any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )
        if any_grasped:
            self._metric_stage(0, True)
            self._metric_stage(1, True)

        # Fit grid lines from current positions (identity-free arrangement)
        row_xs, col_ys = self._fit_rows_cols_from_positions(positions_xy)

        # Spacing checks
        row_spacing = float(abs(row_xs[1] - row_xs[0]))
        col_s1 = float(abs(col_ys[0] - col_ys[1]))
        col_s2 = float(abs(col_ys[1] - col_ys[2]))

        spacing_ok = (
            (self._ROW_SPACING_RANGE[0] <= row_spacing <= self._ROW_SPACING_RANGE[1])
            and (self._COL_SPACING_RANGE[0] <= col_s1 <= self._COL_SPACING_RANGE[1])
            and (self._COL_SPACING_RANGE[0] <= col_s2 <= self._COL_SPACING_RANGE[1])
            and (abs(col_s1 - col_s2) <= self._ALIGN_TOL)  # "consistent spacing"
        )

        # Assign each cube to nearest (row, col) cell; require bijection (6 unique cells)
        cell_to_cube = {}
        per_cube_cell = []
        for i in range(len(self.cubes)):
            x, y, z = positions[i]
            r = int(np.argmin(np.abs(row_xs - x)))
            c = int(np.argmin(np.abs(col_ys - y)))
            key = (r, c)
            per_cube_cell.append(key)
            if key in cell_to_cube:
                # Two cubes mapped to same cell -> not a valid 3x2 grid
                cell_to_cube = None
                break
            cell_to_cube[key] = i

        grid_filled = (cell_to_cube is not None) and (len(cell_to_cube) == self._ROWS * self._COLS)

        # Alignment + table contact + upright checks
        alignment_ok = True
        on_table_ok = True
        upright_ok = True
        for (r, c), idx in (cell_to_cube or {}).items():
            x, y, z = positions[idx]
            q = quats[idx]

            if abs(x - row_xs[r]) > self._ALIGN_TOL or abs(y - col_ys[c]) > self._ALIGN_TOL:
                alignment_ok = False
                break

            # Flat on table: collide with table top and be near table height
            if not self.cubes[idx].is_colliding(self.table.table_top):
                on_table_ok = False
                break
            if abs(z - self._Z_TABLE_CUBE) > self._Z_TOL:
                on_table_ok = False
                break

            if not self._is_upright(q):
                upright_ok = False
                break

        positions_ok = bool(spacing_ok and grid_filled and alignment_ok and on_table_ok and upright_ok)

        # Stage 2/3/4/5 based on partial completion heuristics
        # Stage 2: any cube close to any inferred cell center (transport)
        inferred_centers = np.array(
            [[row_xs[r], col_ys[c]] for r in range(self._ROWS) for c in range(self._COLS)], dtype=float
        )
        dists_to_any_center = []
        for i in range(len(self.cubes)):
            d = np.min(np.linalg.norm(inferred_centers - positions_xy[i], axis=1))
            dists_to_any_center.append(float(d))
        if np.min(dists_to_any_center) < 0.05:
            self._metric_stage(2, True)

        # Stage 3: at least one cube well-aligned, on table, upright
        one_placed = False
        if cell_to_cube is not None:
            for (r, c), idx in cell_to_cube.items():
                if (
                    abs(positions[idx][0] - row_xs[r]) <= self._ALIGN_TOL
                    and abs(positions[idx][1] - col_ys[c]) <= self._ALIGN_TOL
                    and self.cubes[idx].is_colliding(self.table.table_top)
                    and self._is_upright(quats[idx])
                ):
                    one_placed = True
                    break
        if one_placed:
            self._metric_stage(3, True)

        # Stage 4: majority of cells filled and aligned
        if cell_to_cube is not None:
            aligned_count = 0
            for (r, c), idx in cell_to_cube.items():
                if abs(positions[idx][0] - row_xs[r]) <= self._ALIGN_TOL and abs(positions[idx][1] - col_ys[c]) <= self._ALIGN_TOL:
                    aligned_count += 1
            if aligned_count >= 4:
                self._metric_stage(4, True)

        # Stage 5: final alignment
        if positions_ok:
            self._metric_stage(5, True)

        # Released check (must not be holding any cube)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )
        if positions_ok and released:
            self._metric_stage(6, True)

        self._success_check = bool(positions_ok and released)

        # Simple diagnostic metrics
        mean_dist_to_center = float(np.mean(dists_to_any_center)) if len(dists_to_any_center) else 0.0
        mean_z_err = float(np.mean(np.abs(positions[:, 2] - self._Z_TABLE_CUBE)))

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_xy_dist_to_nearest_inferred_cell_m": mean_dist_to_center,
                "row_spacing_m": row_spacing,
                "col_spacing_0_m": col_s1,
                "col_spacing_1_m": col_s2,
            },
            pose_error={
                "mean_abs_z_error_m": mean_z_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})