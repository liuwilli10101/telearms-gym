"""Arrange four ColoredCubes into a flat 2x2 Microsoft-logo grid (R G / B Y) on the table."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class MicrosoftLogo2x2SemanticCubeGrid(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes flat on the table into a tight 2x2 grid that
    reproduces the Microsoft logo color layout. From the validation camera
    (top of image = +X, left of image = +Y) the layout is:

        red    green        (top row,    +X)
        blue   yellow       (bottom row, -X)

    Sequential Sub-Goals:
        1. Grasp any cube with the left gripper.
        2. Grasp any cube with the right gripper.
        3. Move cubes near their intended quadrant positions in the 2x2 logo grid.
        4. Place the cubes flat on the table to form a tight 2x2 grid.
        5. Align spacing so the 2x2 grid is squared up and evenly spaced.
        6. Verify each color occupies the correct logo quadrant.
        7. Release all cubes after the logo grid is formed.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _BASE_Z = 0.97
    # Expected center-to-center cell spacing. Cubes are 0.04m, so 0.06 means
    # ~2cm gap between adjacent cubes; the offset tolerance below covers tight
    # packing (offset=0.02) up to looser placements (offset=0.055).
    _CELL_SPACING = 0.06
    _ON_TABLE_Z_TOL = 0.02
    # Allowed pairwise overlap: cubes may interpenetrate slightly because of
    # physics jitter. Negative `_MIN_GAP` accepts touching/near-touching cubes.
    _MIN_GAP = -0.008
    _CUBE_SIZE = 0.04

    _NUM_SUBGOALS = 7

    # Translation-invariant 2x2 layout. Quadrant signs are relative to the
    # cubes' own centroid. Convention (from the validation camera):
    # top of image = +X, left of image = +Y. Microsoft logo:
    #   red    (top-left)    -> (+x, +y)
    #   green  (top-right)   -> (+x, -y)
    #   blue   (bottom-left) -> (-x, +y)
    #   yellow (bottom-right)-> (-x, -y)
    _QUADRANT_SIGNS = {
        "red":    (+1, +1),
        "green":  (+1, -1),
        "blue":   (-1, +1),
        "yellow": (-1, -1),
    }
    _GRID_OFFSET_TOL = 0.025  # |offset - cell_spacing/2| per axis (m)
    _MIN_QUADRANT_OFFSET = 0.010  # each |offset| must exceed this to clearly lie in a quadrant

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue, self.cube_yellow]
        self._target_colors = ["red", "green", "blue", "yellow"]

        # Goal poses anchored at workspace centre (used for diagnostic metrics only;
        # the success check is centroid-anchored / translation-invariant).
        anchor_x, anchor_y = 0.55, 0.0
        half = self._CELL_SPACING / 2.0
        self._goal_positions = {
            "red":    np.array([anchor_x + half, anchor_y + half, self._BASE_Z]),
            "green":  np.array([anchor_x + half, anchor_y - half, self._BASE_Z]),
            "blue":   np.array([anchor_x - half, anchor_y + half, self._BASE_Z]),
            "yellow": np.array([anchor_x - half, anchor_y - half, self._BASE_Z]),
        }

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

        # REQUIRED: set_color() in _on_reset() too.
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        # Workspace bounds
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._BASE_Z

        # Keep cubes well-separated and away from goal positions so the reset is unsolved.
        min_center_dist = self._CUBE_SIZE + self._MIN_GAP + 0.01
        goal_exclusion = 0.05  # avoid spawning right at any goal quadrant

        placed_xy = []

        for cube in self.cubes:
            for _ in range(2000):
                x = float(rng.uniform(x_lo, x_hi))
                y = float(rng.uniform(y_lo, y_hi))
                xy = np.array([x, y], dtype=float)

                if any(float(np.linalg.norm(xy - g[:2])) < goal_exclusion
                       for g in self._goal_positions.values()):
                    continue

                if any(float(np.linalg.norm(xy - prev)) < min_center_dist for prev in placed_xy):
                    continue

                placed_xy.append(xy)
                pos = np.array([x, y, z], dtype=float)
                break
            else:
                # Fallback (still in-bounds): deterministic offsets if sampling fails
                x = float(np.clip(0.42 + 0.06 * len(placed_xy), x_lo, x_hi))
                y = float(np.clip(-0.20 + 0.12 * len(placed_xy), y_lo, y_hi))
                pos = np.array([x, y, z], dtype=float)
                placed_xy.append(pos[:2].copy())

            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
            cube.set_pose(position=pos, quat=q.elements)

        for _ in range(50):
            self._mojo.step()

        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not self.cubes or len(self.cubes) < 4:
            return False

        positions_ok = True
        target_distance: Dict[str, float] = {}
        pose_error: Dict[str, float] = {}

        cube_positions = [np.array(c.body.get_position(), dtype=float) for c in self.cubes]

        # Translation-invariant: anchor the expected 2x2 grid on the cubes' own centroid.
        centroid_xy = np.mean([p[:2] for p in cube_positions], axis=0)
        half_cell = self._CELL_SPACING / 2.0

        # Identity-aware slot assignment: classify each cube by its quadrant
        # around the centroid, then check that quadrant matches the cube's color.
        sign_to_color = {(sx, sy): col for col, (sx, sy) in self._QUADRANT_SIGNS.items()}

        slot_assignments = []  # list of (cube, quadrant_color, p, dx, dy)
        for cube, p in zip(self.cubes, cube_positions):
            dx = float(p[0] - centroid_xy[0])
            dy = float(p[1] - centroid_xy[1])

            # Quadrant offsets must be non-trivial.
            if abs(dx) < self._MIN_QUADRANT_OFFSET or abs(dy) < self._MIN_QUADRANT_OFFSET:
                positions_ok = False

            # Grid tightness: offsets should match expected half_cell in each axis.
            if abs(abs(dx) - half_cell) > self._GRID_OFFSET_TOL:
                positions_ok = False
            if abs(abs(dy) - half_cell) > self._GRID_OFFSET_TOL:
                positions_ok = False

            # On-table via contact (no absolute Z check).
            if not cube.is_colliding(self.table.table_top):
                positions_ok = False

            sx = int(np.sign(dx)) if dx != 0 else 0
            sy = int(np.sign(dy)) if dy != 0 else 0
            quadrant_color = sign_to_color.get((sx, sy), None)
            slot_assignments.append((cube, quadrant_color, p, dx, dy))

        # Each of the four expected quadrants must be occupied exactly once.
        occupied = [c for (_, c, _, _, _) in slot_assignments if c is not None]
        if len(set(occupied)) != 4 or len(occupied) != len(self.cubes):
            positions_ok = False

        # Color-quadrant matching: the cube currently in each quadrant must
        # actually have that quadrant's color.
        color_layout_ok = True
        for cube, quadrant_color, p, dx, dy in slot_assignments:
            if quadrant_color is None:
                color_layout_ok = False
                continue
            if cube.get_color() != quadrant_color:
                color_layout_ok = False

            sx, sy = self._QUADRANT_SIGNS.get(quadrant_color, (0, 0))
            expected = np.array(
                [centroid_xy[0] + sx * half_cell, centroid_xy[1] + sy * half_cell],
                dtype=float,
            )
            target_distance[quadrant_color] = float(np.linalg.norm(p[:2] - expected))
            pose_error[quadrant_color] = float(abs(p[2] - self._BASE_Z))

        # Pairwise minimum spacing (avoid overlapping cubes).
        min_center_dist = self._CUBE_SIZE + self._MIN_GAP
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                dxy = float(np.linalg.norm(cube_positions[i][:2] - cube_positions[j][:2]))
                if dxy < min_center_dist:
                    positions_ok = False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        if positions_ok and color_layout_ok:
            for idx in range(6):
                self._metric_stage(idx, True)
        if released:
            self._metric_stage(6, True)

        self._success_check = bool(positions_ok and color_layout_ok and released)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
