from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class ThreeStepStaircaseStackSixCubes(RoboEvalEnv, MetricRolloutEval):
    """
    Build a 3-step staircase from six colored cubes: three cubes form a straight base row
    on the table, a second row of two cubes is stacked on the first two base cubes, and
    a final cube is stacked on the leftmost second-row cube to create column heights of
    3, 2, and 1 along +X.

    Sequential Sub-Goals:
        1. Grasp any cube with the left gripper.
        2. Grasp any cube with the right gripper.
        3. Lift a grasped cube clear of the table surface to enable placement.
        4. Move a cube to its intended column/stack location.
        5. Place cubes to form the base row and the two higher steps (3-2-1 columns).
        6. Align columns and spacing so the staircase is straight, evenly spaced, and stable.
        7. Release all cubes after the staircase is completed.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants (per proposal)
    _N_CUBES = 6
    _BASE_Z = 0.97
    _DZ = 0.04

    _BASE_ROW_Y_ALIGNMENT_TOL = 0.025
    _BASE_ROW_X_SPACING_TOL = 0.03
    _XY_STACK_ALIGNMENT_TOL = 0.025
    _CONTACT_Z_TOL = 0.015
    _ON_TABLE_Z_TOL = 0.02

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Props (table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (cube_1 .. cube_6)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._N_CUBES)]

        # Required: set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Goal geometry: 3 columns along +X with heights 3,2,1
        # Column 0 (x_index=0): cube_1 (base), cube_4 (middle), cube_6 (top)
        # Column 1 (x_index=1): cube_2 (base), cube_5 (middle)
        # Column 2 (x_index=2): cube_3 (base)
        self._x0 = 0.48
        self._dx = 0.06
        self._y0 = 0.00

        self._goal_positions = {
            "cube_1": np.array([self._x0 + 0 * self._dx, self._y0, self._BASE_Z]),
            "cube_2": np.array([self._x0 + 1 * self._dx, self._y0, self._BASE_Z]),
            "cube_3": np.array([self._x0 + 2 * self._dx, self._y0, self._BASE_Z]),
            "cube_4": np.array([self._x0 + 0 * self._dx, self._y0, self._BASE_Z + 1 * self._DZ]),
            "cube_5": np.array([self._x0 + 1 * self._dx, self._y0, self._BASE_Z + 1 * self._DZ]),
            "cube_6": np.array([self._x0 + 0 * self._dx, self._y0, self._BASE_Z + 2 * self._DZ]),
        }
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Metrics (full signature required)
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

        # 2. Deterministic RNG available if needed (do not use for goal placement)
        rng = np.random.default_rng(self.seed)

        # Required: set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM, UNSOLVED positions (ensure _success() is False)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Keep cubes well-separated so they don't start stacked/aligned as a staircase.
        # (Cube is ~4cm; use a conservative clearance.)
        min_xy_sep = 0.085

        positions = []
        for _i in range(self._N_CUBES):
            for _attempt in range(2000):
                x = float(rng.uniform(x_min + 0.02, x_max - 0.02))
                y = float(rng.uniform(y_min + 0.02, y_max - 0.02))
                p = np.array([x, y, self._BASE_Z], dtype=float)
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_xy_sep for q in positions):
                    positions.append(p)
                    break
            else:
                # Fallback: spread along X deterministically (still unsolved and within bounds)
                k = len(positions)
                x = float(x_min + 0.03 + (k % 3) * 0.09)
                y = float(y_min + 0.03 + (k // 3) * 0.12)
                positions.append(np.array([x, y, self._BASE_Z], dtype=float))

        # Random yaw rotations (upright), keep quat as [w, x, y, z]
        for cube, pos in zip(self.cubes, positions):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(position=pos, quat=np.array([q.w, q.x, q.y, q.z], dtype=float))

        # Let physics settle
        for _ in range(80):
            self._mojo.step()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # Goal: 3-step staircase with 6 cubes — 3 on base layer, 2 on middle, 1 on top.
        # Any 3 cubes at BASE_Z form the bottom row (Y-aligned, evenly spaced along X).
        # Any 2 cubes at BASE_Z + DZ stack atop 2 bottom cubes.
        # Any 1 cube at BASE_Z + 2*DZ stacks atop a middle cube.
        # Layer-by-layer placement for physics stability:
        #   cube_1, cube_2, cube_3 = bottom layer cubes (sorted by X)
        #   cube_4, cube_5 = middle layer cubes (sorted by X)
        #   cube_6 = top layer cube
        #   q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        #   cube_1.set_pose(position=np.array([0.48, 0.0, 0.97]), quat=q_identity)
        #   cube_2.set_pose(position=np.array([0.54, 0.0, 0.97]), quat=q_identity)
        #   cube_3.set_pose(position=np.array([0.60, 0.0, 0.97]), quat=q_identity)
        #   for _ in range(50): self._mojo.step()
        #   cube_4.set_pose(position=np.array([0.48, 0.0, 1.01]), quat=q_identity)
        #   cube_5.set_pose(position=np.array([0.54, 0.0, 1.01]), quat=q_identity)
        #   for _ in range(50): self._mojo.step()
        #   cube_6.set_pose(position=np.array([0.48, 0.0, 1.05]), quat=q_identity)
        #   for _ in range(50): self._mojo.step()
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _success(self) -> bool:
        # Reset success flag at start
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or len(getattr(self, "cubes", [])) != 6:
            return False

        # a. Get positions and group by Z-height layer.
        # NOTE: Axis assumption inherited from SimGen. Verify with live sim if success checks fail.
        positions = {c: c.body.get_position() for c in self.cubes}
        z_base = self._BASE_Z
        dz = self._DZ
        tol = self._ON_TABLE_Z_TOL

        bottom_layer = [c for c in self.cubes if abs(positions[c][2] - z_base) <= tol]
        middle_layer = [c for c in self.cubes if abs(positions[c][2] - (z_base + dz)) <= tol]
        top_layer = [c for c in self.cubes if abs(positions[c][2] - (z_base + 2 * dz)) <= tol]

        # b. Count check: 3 bottom, 2 middle, 1 top
        counts_ok = len(bottom_layer) == 3 and len(middle_layer) == 2 and len(top_layer) == 1

        base_row_ok = False
        middle_ok = False
        top_ok = False

        if counts_ok:
            # c. Bottom row structure
            # Sort bottom layer cubes by X (generated code uses X-axis ordering)
            bottom_sorted = sorted(bottom_layer, key=lambda c: float(positions[c][0]))
            bottom_xs = [float(positions[c][0]) for c in bottom_sorted]
            bottom_ys = [float(positions[c][1]) for c in bottom_sorted]

            # Y alignment: all 3 cubes share approximately the same Y coordinate
            y_mean = float(np.mean(bottom_ys))
            y_aligned = all(abs(y - y_mean) <= self._BASE_ROW_Y_ALIGNMENT_TOL for y in bottom_ys)

            # Even spacing: X differences between adjacent cubes are approximately equal
            spacing = np.diff(bottom_xs)
            spacing_ok = (
                all(abs(s - spacing[0]) <= self._BASE_ROW_X_SPACING_TOL for s in spacing)
                if len(spacing) > 0
                else False
            )

            # On-table: all 3 bottom cubes colliding with table_top
            on_table_ok = all(c.is_colliding(self.table.table_top) for c in bottom_layer)

            base_row_ok = bool(y_aligned and spacing_ok and on_table_ok)

            # d. Middle layer: each middle cube must be colliding with (stacked on) a bottom cube
            middle_ok = all(
                any(mc.is_colliding(bc) for bc in bottom_layer)
                for mc in middle_layer
            )

            # e. Top layer: 1 cube on top of a middle cube
            top_cube = top_layer[0]
            top_ok = any(top_cube.is_colliding(mc) for mc in middle_layer)

        # f. Released: no gripper holding any cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # g. Combine
        positions_ok = bool(counts_ok and base_row_ok and middle_ok and top_ok)

        # Stage metrics: mark all when full structure achieved
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and released)

        # h. Wire _metric_finalize with success metrics
        # Compute per-cube distances to nearest goal position
        goal_pos_list = list(self._goal_positions.values())
        dist_list = []
        for cube in self.cubes:
            pos = positions[cube]
            min_dist = float(min(np.linalg.norm(pos - gp) for gp in goal_pos_list))
            dist_list.append(min_dist)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={f"cube_{i+1}_nearest_goal": dist_list[i] for i in range(self._N_CUBES)},
            pose_error={f"cube_{i+1}_dist": dist_list[i] for i in range(self._N_CUBES)},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
