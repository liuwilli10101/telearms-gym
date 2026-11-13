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
class CubeGridArrangementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange six colored cubes flat on the table to form a neat 2x3 rectangular grid.
    All cubes should be aligned in two rows and three columns with consistent spacing.

    Sequential Sub-Goals:
        1. Grasp cube 1 with either gripper.
        2. Move cube 1 toward its intended grid cell location.
        3. Place cube 1 flat on the table at its grid cell.
        4. Grasp cube 2 with either gripper.
        5. Move cube 2 toward its intended grid cell location.
        6. Place cube 2 flat on the table at its grid cell.
        7. Grasp cube 3 with either gripper.
        8. Move cube 3 toward its intended grid cell location.
        9. Place cube 3 flat on the table at its grid cell.
        10. Grasp cube 4 with either gripper.
        11. Move cube 4 toward its intended grid cell location.
        12. Place cube 4 flat on the table at its grid cell.
        13. Grasp cube 5 with either gripper.
        14. Move cube 5 toward its intended grid cell location.
        15. Place cube 5 flat on the table at its grid cell.
        16. Grasp cube 6 with either gripper.
        17. Move cube 6 toward its intended grid cell location.
        18. Place cube 6 flat on the table at its grid cell.
        19. Nudge and align cubes so rows/columns are straight and spacing is consistent.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from spec)
    _ROWS = 2
    _COLS = 3
    _POSITION_TOL_M = 0.03
    _ROW_ALIGN_TOL_M = 0.02
    _COL_ALIGN_TOL_M = 0.02
    _MIN_GAP_M = 0.02
    _MAX_GAP_M = 0.06
    _ROLL_PITCH_TOL_DEG = 10.0

    # Asset physicals
    _CUBE_SIZE_M = 0.04
    _CUBE_REST_Z = 0.97

    _NUM_CUBES = 6
    _NUM_SUBGOALS = 19

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ColoredCube only)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._NUM_CUBES)]

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Precompute a canonical goal grid (used for reset + as initial guess in success metrics)
        self._goal_positions = self._compute_goal_grid_positions(
            center_x=0.55,
            center_y=0.0,
            row_center_spacing=self._CUBE_SIZE_M + 0.04,  # -> gap 0.04 (within [0.02, 0.06])
            col_center_spacing=self._CUBE_SIZE_M + 0.04,
            z=self._CUBE_REST_Z,
        )

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (even if not used here)
        rng = np.random.default_rng(self.seed)

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # 3. Randomize initial cube poses (unsolved) within workspace bounds.
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Enforce non-overlap (in XY) and also enforce "not already solved".
        min_center_dist = self._CUBE_SIZE_M + self._MIN_GAP_M + 1e-3

        goal_xy = np.array([p[:2] for p in self._goal_positions], dtype=float)

        def _sample_positions():
            pts = []
            max_tries = 5000
            for _ in range(max_tries):
                if len(pts) >= self._NUM_CUBES:
                    break
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)
                if all(np.linalg.norm(p - q) >= min_center_dist for q in pts):
                    pts.append(p)
            if len(pts) != self._NUM_CUBES:
                # Fallback: spread along X if rejection got stuck
                xs = np.linspace(x_min + 0.02, x_max - 0.02, self._NUM_CUBES)
                ys = rng.uniform(y_min + 0.02, y_max - 0.02, size=self._NUM_CUBES)
                pts = [np.array([float(xs[i]), float(ys[i])], dtype=float) for i in range(self._NUM_CUBES)]
            return np.array(pts, dtype=float)

        # Resample until arrangement is NOT a valid 2x3 grid and not near the goal grid.
        for _ in range(200):
            xy = _sample_positions()

            # Ensure at least one cube is clearly away from any goal cell center in XY
            # (prevents accidental success due to a rare perfect sample).
            min_d_to_goal = np.min(np.linalg.norm(xy[:, None, :] - goal_xy[None, :, :], axis=2), axis=1)
            far_from_goal = bool(np.max(min_d_to_goal) > (self._POSITION_TOL_M * 1.5))

            # Set poses (upright with random yaw)
            for cube, pxy in zip(self.cubes, xy):
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat_upright_yaw_xyzw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
                cube.set_pose(position=np.array([pxy[0], pxy[1], z], dtype=float), quat=quat_upright_yaw_xyzw)

            # Physics settling (so collision queries / positions are consistent)
            self._mojo.physics.forward()

            grid_ok, _ = self._grid_arrangement_ok()
            if (not grid_ok) and far_from_goal:
                break

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # Keep rng "used" deterministically (no effect on placements)
        _ = rng.integers(0, 1)

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
        if len(self.cubes) != self._NUM_CUBES:
            return False

        # Released check
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Upright + on-table checks
        upright_ok = True
        on_table_ok = True
        roll_pitch_angles = []
        for cube in self.cubes:
            q_xyzw = np.array(cube.body.get_quaternion(), dtype=float)
            ang = self._upright_angle_deg_from_quat_xyzw(q_xyzw)
            roll_pitch_angles.append(ang)
            if ang > self._ROLL_PITCH_TOL_DEG:
                upright_ok = False
                break
            # Prefer collision with table top; also accept close Z as a fallback.
            pos = np.array(cube.body.get_position(), dtype=float)
            if not cube.is_colliding(self.table.table_top) and abs(pos[2] - self._CUBE_REST_Z) > 0.02:
                on_table_ok = False
                break

        # Arrangement check: infer a best-fit 2x3 grid from the current cube positions
        grid_ok, assign_info = self._grid_arrangement_ok()

        positions_ok = bool(grid_ok and upright_ok and on_table_ok)

        # Mark subgoals:
        # - For each cube: grasp/transport/place are marked true if the cube is "placed" in some grid cell.
        # - Final align subgoal is marked true if the full grid constraints are satisfied.
        # assign_info["cell_distances"] is per-cube distance to its assigned cell (in XY).
        placed_mask = []
        if assign_info is not None and "cell_distances" in assign_info:
            placed_mask = [d <= self._POSITION_TOL_M for d in assign_info["cell_distances"]]
        else:
            placed_mask = [False] * self._NUM_CUBES

        for i in range(self._NUM_CUBES):
            base = 3 * i
            placed_i = bool(placed_mask[i] and upright_ok and on_table_ok)
            # Grasp, transport, place (coarsely inferred from placement state)
            self._metric_stage(base + 0, placed_i)
            self._metric_stage(base + 1, placed_i)
            self._metric_stage(base + 2, placed_i)

        # Align stage
        self._metric_stage(self._NUM_SUBGOALS - 1, positions_ok)

        # Final success requires arrangement + upright + released
        self._success_check = bool(positions_ok and released)

        # Metrics
        # NOTE: _metric_finalize() cannot receive None values (it casts to float).
        mean_cell_dist = float("nan")
        max_cell_dist = float("nan")
        pose_upright_mean = float(np.mean(roll_pitch_angles)) if roll_pitch_angles else float("nan")

        if assign_info is not None and "cell_distances" in assign_info:
            ds = np.array(assign_info["cell_distances"], dtype=float)
            mean_cell_dist = float(np.mean(ds))
            max_cell_dist = float(np.max(ds))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_cell_distance_xy_m": mean_cell_dist,
                "max_cell_distance_xy_m": max_cell_dist,
            },
            pose_error={
                "mean_upright_angle_deg": pose_upright_mean,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # --------------------------
    # Helpers
    # --------------------------
    def _compute_goal_grid_positions(
        self,
        center_x: float,
        center_y: float,
        row_center_spacing: float,
        col_center_spacing: float,
        z: float,
    ):
        """
        Returns row-major positions: (row 0, col 0..2), (row 1, col 0..2)
        Rows vary in X; columns vary in Y, with col 0 at left (+Y) and col 2 at right (-Y).
        """
        xs = [
            center_x + (r - (self._ROWS - 1) / 2.0) * row_center_spacing
            for r in range(self._ROWS)
        ]
        ys = [
            center_y + ((self._COLS - 1) / 2.0 - c) * col_center_spacing
            for c in range(self._COLS)
        ]
        poses = []
        for r in range(self._ROWS):
            for c in range(self._COLS):
                poses.append(np.array([xs[r], ys[c], z], dtype=float))
        return poses

    def _upright_angle_deg_from_quat_xyzw(self, q_xyzw: np.ndarray) -> float:
        """
        Returns angle (deg) between cube's local +Z axis and world +Z axis.
        For an upright cube resting on a face, this should be near 0 (yaw is free).
        NOTE: body.get_quaternion() returns wxyz, so the input is wxyz (despite the name).
        """
        q = np.array(q_xyzw, dtype=float)
        if q.shape[0] != 4:
            return 180.0
        w, x, y, z = q
        n = np.linalg.norm(q)
        if n < 1e-9:
            return 180.0
        x, y, z, w = x / n, y / n, z / n, w / n

        # Rotation matrix from quaternion (x,y,z,w)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

        # Local +Z axis in world coordinates is the 3rd column of R
        z_axis_world = R[:, 2]
        z_axis_world /= (np.linalg.norm(z_axis_world) + 1e-12)
        cosang = float(np.clip(np.dot(z_axis_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        return ang

    def _grid_arrangement_ok(self):
        """
        Checks whether cubes form a 2x3 grid with:
          - two aligned rows (X) and three aligned columns (Y)
          - consistent spacing (edge gaps within [_MIN_GAP_M, _MAX_GAP_M])
          - no cubes touching (pairwise center distance implies edge gap >= _MIN_GAP_M)
          - each cube within _POSITION_TOL_M of an inferred grid cell center

        Returns: (ok: bool, info: dict)
          info includes:
            - row_centers (2,), col_centers (3,)
            - cell_centers (6,2)
            - assignment (list length 6): assigned cell index for each cube index
            - cell_distances (list length 6): per-cube XY distance to its cell
        """
        positions = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        xs = positions[:, 0]
        ys = positions[:, 1]

        # Quick workspace sanity (not a success requirement per se, but prevents nonsense fits)
        if np.any(xs < 0.35) or np.any(xs > 0.75) or np.any(ys < -0.30) or np.any(ys > 0.30):
            return False, None

        # Ensure cubes are not colliding/touching: enforce minimum center distance in XY
        # Edge gap ~= center_dist - cube_size (conservative).
        min_center_dist = self._CUBE_SIZE_M + self._MIN_GAP_M - 1e-3
        for i in range(self._NUM_CUBES):
            for j in range(i + 1, self._NUM_CUBES):
                dij = float(np.linalg.norm(positions[i, :2] - positions[j, :2]))
                if dij < min_center_dist:
                    return False, None
                # Also reject explicit collision if available
                if self.cubes[i].is_colliding(self.cubes[j]):
                    return False, None

        # --- Find best row split (2 groups of 3) by brute force over combinations ---
        import itertools

        idxs = list(range(self._NUM_CUBES))
        best = None  # (score, row0_idxs, row1_idxs, x0, x1)

        for row0 in itertools.combinations(idxs, 3):
            row0 = set(row0)
            row1 = [i for i in idxs if i not in row0]

            x0 = float(np.mean(xs[list(row0)]))
            x1 = float(np.mean(xs[row1]))
            # enforce ordering: row 0 = back (lower X), row 1 = front (higher X)
            if x0 > x1:
                x0, x1 = x1, x0
                row0, row1 = set(row1), list(row0)

            # Alignment in each row
            row0_err = float(np.max(np.abs(xs[list(row0)] - x0)))
            row1_err = float(np.max(np.abs(xs[row1] - x1)))
            if row0_err > self._ROW_ALIGN_TOL_M or row1_err > self._ROW_ALIGN_TOL_M:
                continue

            # Row spacing constraint (edge gap between rows)
            row_center_delta = abs(x1 - x0)
            row_edge_gap = row_center_delta - self._CUBE_SIZE_M
            if not (self._MIN_GAP_M <= row_edge_gap <= self._MAX_GAP_M):
                continue

            score = row0_err + row1_err
            if best is None or score < best[0]:
                best = (score, list(row0), list(row1), x0, x1)

        if best is None:
            return False, None

        _, row0_idxs, row1_idxs, x0, x1 = best
        row_centers = np.array([x0, x1], dtype=float)

        # --- Column grouping by Y: sort and group into 3 pairs (left/mid/right) ---
        y_order = list(np.argsort(-ys))  # descending: +Y (left) -> -Y (right)
        col0 = y_order[0:2]
        col1 = y_order[2:4]
        col2 = y_order[4:6]
        col_groups = [col0, col1, col2]

        col_centers = []
        for g in col_groups:
            yc = float(np.mean(ys[g]))
            # Alignment in each column
            col_err = float(np.max(np.abs(ys[g] - yc)))
            if col_err > self._COL_ALIGN_TOL_M:
                return False, None
            col_centers.append(yc)

        # Ensure proper left->right order: col 0 highest Y, col 2 lowest Y
        y0, y1, y2 = col_centers
        if not (y0 > y1 > y2):
            # Re-sort centers if needed
            col_centers = sorted(col_centers, reverse=True)
            y0, y1, y2 = col_centers

        col_centers = np.array([y0, y1, y2], dtype=float)

        # Column spacing constraints (edge gaps between adjacent columns)
        d01 = abs(y0 - y1)
        d12 = abs(y1 - y2)
        gap01 = d01 - self._CUBE_SIZE_M
        gap12 = d12 - self._CUBE_SIZE_M
        if not (self._MIN_GAP_M <= gap01 <= self._MAX_GAP_M):
            return False, None
        if not (self._MIN_GAP_M <= gap12 <= self._MAX_GAP_M):
            return False, None

        # "Consistent spacing": adjacent gaps should be roughly similar
        if abs(d01 - d12) > 0.02:
            return False, None

        # Build inferred cell centers (row-major) in XY
        cell_centers_xy = []
        for r in range(self._ROWS):
            for c in range(self._COLS):
                cell_centers_xy.append(np.array([row_centers[r], col_centers[c]], dtype=float))
        cell_centers_xy = np.array(cell_centers_xy, dtype=float)  # (6,2)

        # Assign cubes to cells by brute-force permutation (6! = 720)
        cube_xy = positions[:, :2]
        best_perm = None
        best_max = None
        best_mean = None
        best_ds = None

        for perm in itertools.permutations(range(6), 6):
            # perm[cube_idx] = cell_idx
            ds = np.linalg.norm(cube_xy - cell_centers_xy[list(perm)], axis=1)
            maxd = float(np.max(ds))
            meand = float(np.mean(ds))
            if best_perm is None or maxd < best_max or (maxd == best_max and meand < best_mean):
                best_perm = list(perm)
                best_max = maxd
                best_mean = meand
                best_ds = ds

        if best_perm is None:
            return False, None

        # Position tolerance per cube to its assigned cell
        if best_max > self._POSITION_TOL_M:
            return False, None

        info = {
            "row_centers": row_centers,
            "col_centers": col_centers,
            "cell_centers_xy": cell_centers_xy,
            "assignment": best_perm,
            "cell_distances": [float(d) for d in best_ds],
        }
        return True, info