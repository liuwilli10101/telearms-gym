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
class AlternatingColorCubeRow(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four cubes into a single straight row with alternating colors.
    Final left-to-right order (increasing X) must be: green, yellow, green, yellow.

    Sequential Sub-Goals:
        1. Grasp a green cube to place it as the leftmost cube in the row.
        2. Place the first green cube at the leftmost row position.
        3. Grasp a yellow cube to place it immediately to the right of the leftmost green cube.
        4. Place the first yellow cube as the second cube in the row.
        5. Grasp the remaining green cube to place it as the third cube in the row.
        6. Place the second green cube as the third cube in the row.
        7. Grasp the remaining yellow cube to place it as the rightmost cube in the row.
        8. Place the second yellow cube as the fourth/rightmost cube in the row.
        9. Align cube positions so they form a straight row with consistent spacing and correct alternation.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal parameters (from task specification)
    _EXPECTED_ORDER = ["green", "yellow", "green", "yellow"]
    _SAME_ROW_TOL_Y = 0.03
    _SAME_HEIGHT_TOL_Z = 0.02
    _MIN_DX = 0.04
    _MAX_DX = 0.10

    # Task constants
    _N_CUBES = 4
    _N_SUBGOALS = 9
    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ColoredCube only)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._N_CUBES)]

        # REQUIRED: set_color() in _initialize_env()
        for cube, col in zip(self.cubes, self._EXPECTED_ORDER):
            cube.set_color(col)

        # Goal row poses (row axis = x, constant y)
        center_x, center_y = 0.55, 0.0
        spacing = 0.06  # within [0.04, 0.10]
        start_x = center_x - 1.5 * spacing
        self._goal_positions = [
            np.array([start_x + i * spacing, center_y, self._CUBE_REST_Z]) for i in range(self._N_CUBES)
        ]

        # Initialize metrics (FULL signature REQUIRED)
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

        # 2. Deterministic RNG (required pattern; validation pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM (unsolved) positions within workspace bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Use a conservative min separation to reduce initial interpenetration
        min_sep = 0.05

        # Sample random positions until we are confidently NOT in a solved configuration.
        # We force a violation of the "same_row" condition by ensuring sufficient Y spread.
        for _attempt in range(200):
            # sample XYZ points using the deterministic rng (replay-safe)
            pts = np.array(
                get_random_points_on_plane(
                    self._N_CUBES,
                    np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z], dtype=float),
                    np.array([(x_max - x_min) * 0.5, (y_max - y_min) * 0.5, 0.0], dtype=float),
                    min_sep,
                    rng=rng,
                ),
                dtype=float,
            )

            # ensure not "same_row" (max deviation from mean Y > tol)
            ys = pts[:, 1].astype(float)
            y_mean = float(np.mean(ys))
            if float(np.max(np.abs(ys - y_mean))) <= (self._SAME_ROW_TOL_Y + 0.01):
                continue

            # also avoid exact goal placement (very unlikely, but keep deterministic guarantee)
            if np.allclose(pts, np.array(self._goal_positions), atol=1e-6):
                continue

            # assign positions to cubes in a non-identity shuffled mapping (further reduces chance of solved order)
            perm = rng.permutation(self._N_CUBES)
            if np.all(perm == np.arange(self._N_CUBES)):
                perm = np.roll(perm, 1)

            for cube_i, pos_i in enumerate(perm.tolist()):
                cube = self.cubes[cube_i]
                col = self._EXPECTED_ORDER[cube_i]
                cube.set_color(col)  # REQUIRED: set_color() in _on_reset()
                cube.set_pose(position=pts[pos_i], quat=np.array([0.0, 0.0, 0.0, 1.0]))
            break
        else:
            # Fallback: deterministic unsolved placement (still within bounds), guarantees same_row violation
            fallback_positions = np.array(
                [
                    [0.42, -0.22, z],
                    [0.68, 0.22, z],
                    [0.50, -0.10, z],
                    [0.60, 0.10, z],
                ],
                dtype=float,
            )
            perm = np.array([1, 0, 3, 2], dtype=int)
            for cube_i, pos_i in enumerate(perm.tolist()):
                cube = self.cubes[cube_i]
                col = self._EXPECTED_ORDER[cube_i]
                cube.set_color(col)  # REQUIRED: set_color() in _on_reset()
                cube.set_pose(position=fallback_positions[pos_i], quat=np.array([0.0, 0.0, 0.0, 1.0]))

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._N_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        if len(self.cubes) != self._N_CUBES:
            return False

        # Gather cube states
        positions = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)
        zs = np.array([p[2] for p in positions], dtype=float)

        # All cubes should be on the table surface (collision direction is reliable here)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # Row constraints: approximately same Y, same height Z
        y_mean = float(np.mean(ys))
        z_mean = float(np.mean(zs))
        same_row = bool(np.max(np.abs(ys - y_mean)) <= self._SAME_ROW_TOL_Y)
        same_height = bool(np.max(np.abs(zs - z_mean)) <= self._SAME_HEIGHT_TOL_Z)

        # Order and spacing along X
        sort_idx = np.argsort(xs)  # increasing X = left-to-right per spec (row_axis='x')
        sorted_xs = xs[sort_idx]
        dx = np.diff(sorted_xs) if len(sorted_xs) >= 2 else np.array([], dtype=float)
        spacing_ok = bool((dx.size == (self._N_CUBES - 1)) and np.all(dx >= self._MIN_DX) and np.all(dx <= self._MAX_DX))

        # Color alternation/order check in sorted-by-X order
        color_order_ok = True
        for rank, cube_i in enumerate(sort_idx.tolist()):
            if self.cubes[cube_i].get_color() != self._EXPECTED_ORDER[rank]:
                color_order_ok = False
                break

        positions_ok = bool(on_table and same_row and same_height and spacing_ok)
        semantic_ok = bool(color_order_ok)

        # Released gripper condition
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage metrics (mark all stages complete once final condition is met)
        if positions_ok and semantic_ok:
            for i in range(self._N_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: per-slot distances after sorting by X against goal positions
        target_distance = {}
        pose_error = {}
        for slot, cube_i in enumerate(sort_idx.tolist()):
            d = float(np.linalg.norm(self.cubes[cube_i].body.get_position() - self._goal_positions[slot]))
            target_distance[f"slot_{slot}_dist"] = d
        pose_error["row_y_max_dev"] = float(np.max(np.abs(ys - y_mean)))
        pose_error["row_z_max_dev"] = float(np.max(np.abs(zs - z_mean)))
        pose_error["spacing_min_dx"] = float(np.min(dx)) if dx.size else 0.0
        pose_error["spacing_max_dx"] = float(np.max(dx)) if dx.size else 0.0

        # Finalize metrics at END
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})