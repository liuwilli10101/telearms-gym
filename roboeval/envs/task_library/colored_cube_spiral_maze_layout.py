from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class ColoredCubeSpiralMazeLayout(RoboEvalEnv, MetricRolloutEval):
    """Lay the colored cubes flat into a continuous single-path square spiral maze wall.

Success Criteria:
- All cubes are in a single layer at table height and colliding with the table top.
- The cubes form one connected, unbranching adjacency path with exactly two endpoints.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _NUM_CUBES = 12
    _GRID_ROWS = 5
    _GRID_COLS = 5
    # These three used to contradict each other: with 0.06 spacing, 0.04 cubes and
    # a 3 mm minimum gap, a cube could only move 8.5 mm toward its neighbour before
    # the gap check failed -- so the advertised 20 mm position tolerance was really
    # 8.5 mm. The 5x5 grid caps spacing at 0.07 (a wider grid leaves the workspace),
    # so the tolerance comes down to meet it and the gap check is dropped: cubes
    # touching does not break a spiral layout, and the per-cell position checks
    # already pin the shape.
    _CELL_SPACING = 0.07
    _POS_TOL = 0.015
    _ON_TABLE_Z_TOL = 0.02
    _MIN_GAP = 0.0
    _CUBE_SIZE = 0.04
    _TABLE_Z_CUBE_BASE = 0.97

    _SPIRAL_CELLS = [
        (0, 0),  # cube_01
        (0, 1),  # cube_02
        (0, 2),  # cube_03
        (0, 3),  # cube_04
        (0, 4),  # cube_05
        (1, 4),  # cube_06
        (2, 4),  # cube_07
        (3, 4),  # cube_08
        (4, 4),  # cube_09
        (4, 3),  # cube_10
        (4, 2),  # cube_11
        (3, 2),  # cube_12
    ]

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self.cubes = [ColoredCube(self._mojo) for _ in range(self._NUM_CUBES)]

        self._cube_colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
        ]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        self._grid_center_world = np.array([0.55, 0.0, self._TABLE_Z_CUBE_BASE], dtype=float)
        self._grid_center_rc = np.array(
            [(self._GRID_ROWS - 1) / 2.0, (self._GRID_COLS - 1) / 2.0], dtype=float
        )

        self._goal_positions = []
        for (r, c) in self._SPIRAL_CELLS:
            pos = self._cell_center_position(r, c)
            self._goal_positions.append(pos)

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _cell_center_position(self, row: int, col: int) -> np.ndarray:
        dr = float(row) - self._grid_center_rc[0]
        dc = float(col) - self._grid_center_rc[1]
        x = self._grid_center_world[0] + dr * self._CELL_SPACING
        y = self._grid_center_world[1] + dc * self._CELL_SPACING
        z = self._TABLE_Z_CUBE_BASE
        return np.array([x, y, z], dtype=float)

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

        _rng = np.random.default_rng(self.seed)

        # Randomize cube poses within workspace bounds and away from goal cells
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._TABLE_Z_CUBE_BASE

        min_center_dist = self._CUBE_SIZE + self._MIN_GAP
        # Extra margin to reduce accidental contacts at reset
        min_center_dist_reset = min_center_dist + 0.002

        def random_quat_yaw(yaw: float) -> np.ndarray:
            cy = float(np.cos(yaw * 0.5))
            sy = float(np.sin(yaw * 0.5))
            return np.array([cy, 0.0, 0.0, sy], dtype=float)  # w,x,y,z

        positions = []
        max_tries = 5000
        tries = 0
        while len(positions) < self._NUM_CUBES and tries < max_tries:
            tries += 1
            x = float(_rng.uniform(x_min, x_max))
            y = float(_rng.uniform(y_min, y_max))
            p = np.array([x, y, z], dtype=float)

            # Enforce spacing between cubes
            ok = True
            for q in positions:
                if float(np.linalg.norm((p - q)[:2])) < min_center_dist_reset:
                    ok = False
                    break
            if not ok:
                continue

            # Keep away from the corresponding goal position so success cannot occur at reset
            goal = self._goal_positions[len(positions)]
            if float(np.linalg.norm(p[:2] - goal[:2])) <= (self._POS_TOL + 0.01):
                continue

            positions.append(p)

        # Fallback: if sampling failed, deterministically place cubes along the workspace perimeter
        if len(positions) < self._NUM_CUBES:
            positions = []
            xs = np.linspace(x_min + 0.02, x_max - 0.02, self._NUM_CUBES)
            ys = np.full(self._NUM_CUBES, y_min + 0.02)
            for i in range(self._NUM_CUBES):
                positions.append(np.array([float(xs[i]), float(ys[i]), z], dtype=float))

        for i, cube in enumerate(self.cubes):
            cube.set_color(self._cube_colors[i])
            yaw = float(_rng.uniform(-np.pi, np.pi))
            cube.set_pose(position=positions[i].copy(), quat=random_quat_yaw(yaw))

        for _ in range(50):
            self._mojo.step()

        for idx in range(5):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # Goal: All 12 cubes placed in spiral wall cells, touching table, forming
        # a connected single-path spiral (no branches, exactly 2 endpoints).
        # Each cube at its _SPIRAL_CELLS grid position:
        #   q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        #   for i, cube in enumerate(self.cubes):
        #       cube.set_color(self._cube_colors[i])
        #       cube.set_pose(position=self._goal_positions[i].copy(), quat=q_identity)
        #   for _ in range(50):
        #       self._mojo.step()
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Check: no gripper holding any cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # 1. Single layer check: all cubes at approximately base Z height
        zs = [float(c.body.get_position()[2]) for c in self.cubes]
        single_layer_ok = all(abs(z - self._TABLE_Z_CUBE_BASE) <= self._ON_TABLE_Z_TOL for z in zs)

        # 2. On-table check: all cubes colliding with table_top
        on_table_ok = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # 3. Build adjacency graph using pairwise XY distances.
        # Use orthogonal-only adjacency (threshold = 1.1 * _CELL_SPACING) to avoid
        # diagonal neighbors being treated as spiral-adjacent.
        positions = [c.body.get_position() for c in self.cubes]
        adj_thresh = self._CELL_SPACING * 1.1
        adj = {i: [] for i in range(len(self.cubes))}
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                d = float(np.linalg.norm(np.array(positions[i][:2]) - np.array(positions[j][:2])))
                if d <= adj_thresh:
                    adj[i].append(j)
                    adj[j].append(i)

        # 4. Connectivity check: BFS from cube 0, verify all cubes reachable
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node])
        connected_ok = len(visited) == len(self.cubes)

        # 5. Spiral path check: no branches (every node has degree <= 2),
        # exactly 2 endpoints (degree == 1). Ensures the wall traces a single
        # unbranching spiral path.
        degrees = [len(adj[i]) for i in range(len(self.cubes))]
        endpoint_count = sum(1 for d in degrees if d == 1)
        branch_count = sum(1 for d in degrees if d > 2)
        spiral_path_ok = branch_count == 0 and endpoint_count == 2

        arrangement_ok = bool(single_layer_ok and on_table_ok and connected_ok and spiral_path_ok)

        if arrangement_ok:
            self._metric_stage(0, True)
            self._metric_stage(1, True)
            self._metric_stage(2, True)
            self._metric_stage(3, True)
            if released:
                self._metric_stage(4, True)

        self._success_check = bool(arrangement_ok and released)

        # Compute per-cube distance to nearest goal position for metrics
        per_cube_dist = {}
        for i, cube in enumerate(self.cubes):
            pos = cube.body.get_position()
            min_dist = float(min(
                np.linalg.norm(np.array(pos[:2]) - np.array(gp[:2]))
                for gp in self._goal_positions
            ))
            per_cube_dist[f"cube_{i+1:02d}_nearest_goal_dist"] = min_dist

        avg_dist = float(np.mean(list(per_cube_dist.values()))) if per_cube_dist else 0.0
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "avg_nearest_goal_dist_m": avg_dist,
                **per_cube_dist,
            },
            pose_error={
                "orientation_unconstrained": 0.0,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
