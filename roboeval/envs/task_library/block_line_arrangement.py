"""Arrange colored cubes into a single horizontal row on the table."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class BlockLineArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange the colored cubes into a single horizontal row on the table. The cubes should be
    aligned along the table's left-to-right axis with roughly even spacing and no stacking.

    Sequential Sub-Goals:
        1. Grasp any cube with the left gripper.
        2. Grasp any cube with the right gripper.
        3. Lift a grasped cube clear of the table surface to enable repositioning.
        4. Move a cube toward the intended row location.
        5. Place a cube down onto the table near the row, avoiding stacking.
        6. Adjust placements so all cubes share a common y coordinate and have roughly even x spacing.
        7. Release all cubes after final alignment.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False

    _success_check = True
    _final_metrics: Dict = {}

    # Tolerance on the shared row axis (X — cubes should share front/back depth).
    _SHARED_X_TOL_M = 0.03
    _ON_TABLE_Z_TOL_M = 0.02
    _NO_STACKING_MIN_DELTA_Z_M = 0.04
    # Adjacent-cube spacing along the row (Y axis: left-to-right).
    _ADJ_SPACING_RANGE_M = (0.05, 0.12)

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self._colors = ["red", "blue", "green", "yellow", "purple"]
        self.cubes = [ColoredCube(self._mojo) for _ in range(len(self._colors))]

        for cube, col in zip(self.cubes, self._colors):
            cube.set_color(col)

        # Row runs left-to-right along Y; cubes share front/back depth (X).
        self._goal_x = 0.55
        self._goal_z = 0.97
        self._goal_spacing = 0.07
        n = len(self.cubes)
        # Centred around y=0: e.g. for 5 cubes at 0.07 spacing -> [-0.14, -0.07, 0, +0.07, +0.14].
        start_y = -((n - 1) / 2.0) * self._goal_spacing
        self._goal_ys = [start_y + i * self._goal_spacing for i in range(n)]
        self._goal_positions = [np.array([self._goal_x, y, self._goal_z], dtype=float) for y in self._goal_ys]

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
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        for cube, col in zip(self.cubes, self._colors):
            cube.set_color(col)

        # Randomize cubes into UNSOLVED poses within workspace bounds.
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = float(self._goal_z)

        # Rejection-sample non-overlapping placements, and enforce "not solved" by
        # spreading X beyond the shared-X tolerance (success requires shared X).
        min_xy_sep = 0.065  # conservative to reduce initial collisions
        positions = []

        # Force at least two cubes to be separated in X beyond the success tolerance.
        forced_x = np.array(
            [
                rng.uniform(x_lo, 0.46),
                rng.uniform(0.64, x_hi),
            ],
            dtype=float,
        )

        for i, cube in enumerate(self.cubes):
            placed = False
            for _ in range(2000):
                if i < 2:
                    x = float(forced_x[i])
                else:
                    x = float(rng.uniform(x_lo, x_hi))
                y = float(rng.uniform(y_lo, y_hi))
                cand = np.array([x, y, z_base], dtype=float)

                if all(np.linalg.norm((cand - p)[:2]) >= min_xy_sep for p in positions):
                    positions.append(cand)
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
                    cube.set_pose(cand, np.array([quat.w, quat.x, quat.y, quat.z], dtype=float))
                    placed = True
                    break

            if not placed:
                # Fallback: spread along y with alternating x to guarantee unsolved state.
                y = float(y_lo + (y_hi - y_lo) * (i + 1) / (len(self.cubes) + 1))
                x = float((0.42 if (i % 2 == 0) else 0.68))
                cand = np.array([x, y, z_base], dtype=float)
                positions.append(cand)
                cube.set_pose(cand, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

        for _ in range(50):
            self._mojo.step()

        self._num_subgoals = 7
        for idx in range(self._num_subgoals):
            self._metric_stage(idx, False)

        # To visualize the solved configuration, replace the randomized placement above with:
        # --- GOAL STATE ---
        # fixed_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # for cube, pos in zip(self.cubes, self._goal_positions):
        #     cube.set_pose(pos, fixed_quat)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        n = len(self.cubes)
        if n < 2:
            return False

        positions = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)
        zs = np.array([p[2] for p in positions], dtype=float)

        on_table = True
        for c, z in zip(self.cubes, zs):
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break
            if abs(z - self._goal_z) > self._ON_TABLE_Z_TOL_M:
                on_table = False
                break

        no_stacking = True
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = self.cubes[i], self.cubes[j]
                if ci.is_colliding(cj):
                    no_stacking = False
                    break
                dxy = float(np.linalg.norm((positions[i] - positions[j])[:2]))
                dz = float(abs(positions[i][2] - positions[j][2]))
                if dxy < 0.03 and dz >= self._NO_STACKING_MIN_DELTA_Z_M:
                    no_stacking = False
                    break
            if not no_stacking:
                break

        # Row runs along Y (left-to-right): cubes should share X, vary Y.
        x_mean = float(np.mean(xs))
        shared_x = bool(np.max(np.abs(xs - x_mean)) <= self._SHARED_X_TOL_M)

        order = np.argsort(ys)
        ys_sorted = ys[order]
        adj = np.diff(ys_sorted)
        spacing_in_range = bool(
            np.all((adj >= self._ADJ_SPACING_RANGE_M[0]) & (adj <= self._ADJ_SPACING_RANGE_M[1]))
        )
        spacing_even = bool((np.max(adj) - np.min(adj)) <= 0.03) if len(adj) > 0 else False

        positions_ok = bool(on_table and no_stacking and shared_x and spacing_in_range and spacing_even)

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        if positions_ok:
            for i in range(getattr(self, "_num_subgoals", 7)):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and released)

        goal_ys = np.array(self._goal_ys, dtype=float)
        if len(goal_ys) == n:
            goal_positions = np.stack(
                [np.full(n, self._goal_x), goal_ys, np.full(n, self._goal_z)], axis=1
            )
            cur_positions_sorted = np.stack([xs[order], ys_sorted, zs[order]], axis=1)
            dists = np.linalg.norm(cur_positions_sorted - goal_positions, axis=1)
            mean_dist = float(np.mean(dists))
        else:
            mean_dist = float("nan")

        spacing_std = float(np.std(adj)) if len(adj) > 0 else float("nan")
        x_std = float(np.std(xs))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_cube_to_goal_m": mean_dist,
            },
            pose_error={
                "x_std_m": x_std,
                "adjacent_spacing_std_m": spacing_std,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
