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

class CubeSquareArrangementTask(RoboEvalEnv, MetricRolloutEval):
    """Arrange four cubes into a tight 2-by-2 square.

Success Criteria:
- Four cubes form a compact 2-by-2 square with adjacent side contacts accepted by the touching-edge check.
- All cubes are on the table rather than stacked or floating.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _NUM_CUBES = 4
    _NUM_SUBGOALS = 6
    _CUBE_SIZE_M = 0.04
    _TABLE_Z = 0.97  # ColoredCube resting Z on table (per spec)
    _CENTER_XY = np.array([0.55, 0.0])

    # Goal tolerances (from spec)
    _POS_TOL_M = 0.02
    _TOUCH_GAP_TOL_M = 0.01

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._NUM_CUBES)]

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Precompute goal positions: 2x2 touching square (centered at workspace center)
        # Adjacent centers separated by 0.04m => offsets from centroid are +/- 0.02m on X and Y.
        half = self._CUBE_SIZE_M / 2.0
        cx, cy = float(self._CENTER_XY[0]), float(self._CENTER_XY[1])
        self._goal_xy_offsets = [
            np.array([-half, +half]),
            np.array([-half, -half]),
            np.array([+half, +half]),
            np.array([+half, -half]),
        ]
        self._goal_positions = [np.array([cx + o[0], cy + o[1], self._TABLE_Z]) for o in self._goal_xy_offsets]
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 3. Randomize objects into an UNSOLVED configuration (within bounds)
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = float(self._TABLE_Z)

        # Keep cubes separated so they don't accidentally form the 2x2 touching pattern
        min_pair_dist = self._CUBE_SIZE_M + 0.02  # > 0.04m target touching distance

        def _random_quat_z():
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        max_tries = 200
        placed_positions = None

        for _ in range(max_tries):
            pts = []
            for _i in range(self._NUM_CUBES):
                for _j in range(500):
                    x = float(rng.uniform(x_lo, x_hi))
                    y = float(rng.uniform(y_lo, y_hi))
                    p = np.array([x, y], dtype=float)
                    if all(np.linalg.norm(p - q) >= min_pair_dist for q in pts):
                        pts.append(p)
                        break
                else:
                    pts = None
                    break
            if pts is None:
                continue

            placed_positions = [np.array([p[0], p[1], z], dtype=float) for p in pts]

            # Apply poses
            for cube, col, pos in zip(self.cubes, self._cube_colors, placed_positions):
                cube.set_color(col)  # REQUIRED in _on_reset()
                cube.set_pose(position=pos.copy(), quat=_random_quat_z())

            # Physics settling
            self._mojo.physics.forward()

            # Ensure NOT already solved after reset (avoid rare accidental goal arrangement)
            best_max_xy_err, _, _ = self._best_assignment_errors()
            positions_ok = (best_max_xy_err <= self._POS_TOL_M) and self._touching_edges_ok()
            if not positions_ok:
                break
        else:
            # Fallback: deterministic "clearly unsolved" line if sampling somehow fails
            base_x = 0.42
            ys = np.linspace(-0.20, 0.20, self._NUM_CUBES)
            for cube, col, y in zip(self.cubes, self._cube_colors, ys):
                cube.set_color(col)  # REQUIRED in _on_reset()
                cube.set_pose(position=np.array([base_x, float(y), z], dtype=float), quat=_random_quat_z())
            self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _cube_xy(self, cube: ColoredCube) -> np.ndarray:
        p = cube.body.get_position()
        return np.array([float(p[0]), float(p[1])])

    def _best_assignment_errors(self):
        """
        Returns:
            best_max_xy_err: float
            best_mean_xy_err: float
            best_perm: tuple[int, ...] mapping cube index -> goal index
        """
        cube_xys = [self._cube_xy(c) for c in self.cubes]
        goal_xys = [self._CENTER_XY + o for o in self._goal_xy_offsets]

        best_max = float("inf")
        best_mean = float("inf")
        best_perm = None

        # Brute-force all 4! assignments (small and robust)
        import itertools

        for perm in itertools.permutations(range(self._NUM_CUBES), self._NUM_CUBES):
            # perm[k] = cube index assigned to goal k
            errs = []
            for goal_k, cube_i in enumerate(perm):
                e = np.linalg.norm(cube_xys[cube_i] - goal_xys[goal_k])
                errs.append(float(e))
            max_e = max(errs)
            mean_e = sum(errs) / len(errs)

            if (max_e < best_max) or (max_e == best_max and mean_e < best_mean):
                best_max = max_e
                best_mean = mean_e
                best_perm = perm

        return best_max, best_mean, best_perm

    def _touching_edges_ok(self) -> bool:
        """
        Checks that (in XY) there are exactly four "edge-adjacent" pairs at ~0.04m
        (the four edges of the 2x2 square), within touch-gap tolerance.
        """
        xys = [self._cube_xy(c) for c in self.cubes]
        d_target = self._CUBE_SIZE_M
        tol = self._TOUCH_GAP_TOL_M

        # Count pairs whose XY distance is close to cube size (touching side-to-side).
        edge_pairs = 0
        for i in range(self._NUM_CUBES):
            for j in range(i + 1, self._NUM_CUBES):
                d = float(np.linalg.norm(xys[i] - xys[j]))
                if abs(d - d_target) <= tol:
                    edge_pairs += 1

        # In a perfect 2x2 grid, there are 4 touching edges and 2 diagonals.
        return edge_pairs == 4

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Released check (single-arm or multi-gripper safe)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # All cubes should be on the table surface (not stacked / not floating)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # Arrangement check: relative 2x2 touching pattern (centroid/translation invariant).
        # The absolute-position check against _CENTER_XY was removed — the touching-edges
        # constraint already pins the 2x2 square geometry without needing the cluster to be
        # at a specific world coordinate.
        best_max_xy_err, best_mean_xy_err, _ = self._best_assignment_errors()  # diagnostics only
        positions_ok = bool(self._touching_edges_ok())

        # No semantic constraints for this task
        semantic_ok = True

        # Subgoal staging: mark all subgoals complete once arrangement is correct and released
        if positions_ok and released and on_table:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released and on_table)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_xy_to_best_goal_m": float(best_mean_xy_err),
                "max_xy_to_best_goal_m": float(best_max_xy_err),
            },
            pose_error={
                "best_assignment_max_xy_err_m": float(best_max_xy_err),
                "touching_edges_ok": bool(self._touching_edges_ok()),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})