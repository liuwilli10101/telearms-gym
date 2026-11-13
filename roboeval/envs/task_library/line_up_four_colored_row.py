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

class ColoredCubeRowArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes into a single horizontal row on the table (the row runs
    left-to-right across the table, i.e. along the Y axis, with all cubes at the same X depth).
    The cubes must be ordered red, blue, green, yellow along the row (either left-to-right or
    right-to-left is accepted).

    Sequential Sub-Goals:
        1. Grasp any one cube to begin arranging the row.
        2. Grasp another cube (bimanual handling encouraged).
        3. Move the red cube toward the intended leftmost position of the row.
        4. Place the red cube on the table as the leftmost cube of the row.
        5. Move the blue cube to the immediate right of the red cube.
        6. Place the blue cube to the right of the red cube with small spacing.
        7. Move the green cube to the immediate right of the blue cube.
        8. Place the green cube to the right of the blue cube with small spacing.
        9. Move the yellow cube to the immediate right of the green cube.
        10. Nudge cubes as needed so the four cubes are aligned in a straight horizontal row.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (from spec) ---
    _ROW_ALIGN_TOL = 0.03
    _MIN_SEP = 0.02
    _SPACING = 0.06
    _SPACING_TOL = 0.03
    _CUBE_Z = 0.97
    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Props (table exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create required assets ONLY: ColoredCube
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)
        self.yellow_cube = ColoredCube(self._mojo)

        # REQUIRED: call set_color() in _initialize_env
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube, self.yellow_cube]

        # Initialize metrics (FULL signature)
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
        # 1) Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG (even if not used for goal placements)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: call set_color() in _on_reset
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

        # 3) Place objects in RANDOM *unsolved* positions (within workspace bounds)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z

        goal_center_x, goal_center_y = 0.55, 0.00
        goal_start_x = goal_center_x - 1.5 * self._SPACING
        goal_positions = [
            np.array([goal_start_x + 0 * self._SPACING, goal_center_y, z]),  # red
            np.array([goal_start_x + 1 * self._SPACING, goal_center_y, z]),  # blue
            np.array([goal_start_x + 2 * self._SPACING, goal_center_y, z]),  # green
            np.array([goal_start_x + 3 * self._SPACING, goal_center_y, z]),  # yellow
        ]

        def _pairwise_min_dist_ok(pts_xy: np.ndarray, min_dist: float) -> bool:
            n = pts_xy.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(pts_xy[i] - pts_xy[j]) < min_dist:
                        return False
            return True

        def _would_be_success_like(pts: np.ndarray) -> bool:
            # Approximate the same geometric checks used in _success() (row along Y).
            red_p, blue_p, green_p, yellow_p = pts
            xs = np.array([red_p[0], blue_p[0], green_p[0], yellow_p[0]])
            row_aligned = (xs.max() - xs.min()) <= self._ROW_ALIGN_TOL

            deltas = np.array(
                [
                    blue_p[1] - red_p[1],
                    green_p[1] - blue_p[1],
                    yellow_p[1] - green_p[1],
                ]
            )
            order_ok = bool(np.all(deltas >= self._MIN_SEP) or np.all(deltas <= -self._MIN_SEP))
            spacing_ok = bool(np.all(np.abs(np.abs(deltas) - self._SPACING) <= self._SPACING_TOL))
            return bool(row_aligned and order_ok and spacing_ok)

        # Sample until not already solved (and avoid overlaps)
        pts = None
        for _ in range(200):
            xs = rng.uniform(x_min, x_max, size=4)
            ys = rng.uniform(y_min, y_max, size=4)
            candidate = np.stack([xs, ys, np.full(4, z)], axis=1)

            # Keep cubes reasonably separated on the plane
            if not _pairwise_min_dist_ok(candidate[:, :2], min_dist=0.07):
                continue

            # Avoid trivially being exactly at goal poses
            if np.min([np.linalg.norm(candidate[i] - goal_positions[i]) for i in range(4)]) < 1e-6:
                continue

            # Ensure not in (or extremely near) a success arrangement
            if _would_be_success_like(candidate):
                continue

            pts = candidate
            break

        if pts is None:
            # Deterministic fallback that is guaranteed unsolved but within bounds
            pts = np.array(
                [
                    [0.42, -0.20, z],
                    [0.68, 0.20, z],
                    [0.60, -0.05, z],
                    [0.50, 0.10, z],
                ],
                dtype=float,
            )

        for cube, pos in zip(self.cubes, pts):
            cube.set_pose(position=pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # Silence lint about rng (kept for deterministic-pattern compliance)
        _ = rng

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Semantic (color) verification ---
        semantic_ok = (
            self.red_cube.get_color() == "red"
            and self.blue_cube.get_color() == "blue"
            and self.green_cube.get_color() == "green"
            and self.yellow_cube.get_color() == "yellow"
        )

        # --- Spatial checks ---
        # The row runs along the Y axis (the screen-horizontal "left-to-right" direction);
        # the four cubes share the same X depth.
        red_p = self.red_cube.body.get_position()
        blue_p = self.blue_cube.body.get_position()
        green_p = self.green_cube.body.get_position()
        yellow_p = self.yellow_cube.body.get_position()

        # 1) All four cubes on the table top (reliable collision geom)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # 2) Row aligned => all cubes at nearly the same X depth
        xs = np.array([red_p[0], blue_p[0], green_p[0], yellow_p[0]])
        row_aligned = (xs.max() - xs.min()) <= self._ROW_ALIGN_TOL

        # 3) Color order red->blue->green->yellow monotonic along Y, with a minimum gap.
        #    Accept either handedness (left->right or right->left), since which screen side
        #    is "+Y" depends on the camera.
        deltas = np.array([blue_p[1] - red_p[1], green_p[1] - blue_p[1], yellow_p[1] - green_p[1]])
        order_ok = bool(np.all(deltas >= self._MIN_SEP) or np.all(deltas <= -self._MIN_SEP))

        # 4) Even spacing (no clustering): consecutive gaps close to desired spacing
        spacing_ok = bool(np.all(np.abs(np.abs(deltas) - self._SPACING) <= self._SPACING_TOL))

        positions_ok = bool(on_table and row_aligned and order_ok and spacing_ok)

        # Released: no cube is held in any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Metric stages: mark all subgoals complete when final arrangement is correct
        if positions_ok and semantic_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        goal_positions = {
            "red": np.array([0.55, +1.5 * self._SPACING, self._CUBE_Z]),
            "blue": np.array([0.55, +0.5 * self._SPACING, self._CUBE_Z]),
            "green": np.array([0.55, -0.5 * self._SPACING, self._CUBE_Z]),
            "yellow": np.array([0.55, -1.5 * self._SPACING, self._CUBE_Z]),
        }
        target_distance = {
            "red_to_goal": float(np.linalg.norm(red_p - goal_positions["red"])),
            "blue_to_goal": float(np.linalg.norm(blue_p - goal_positions["blue"])),
            "green_to_goal": float(np.linalg.norm(green_p - goal_positions["green"])),
            "yellow_to_goal": float(np.linalg.norm(yellow_p - goal_positions["yellow"])),
            "row_x_spread": float(xs.max() - xs.min()),
            "spacing_error_mean": float(np.mean(np.abs(np.abs(deltas) - self._SPACING))),
        }
        pose_error = {
            "red_pos_err": float(np.linalg.norm(red_p - goal_positions["red"])),
            "blue_pos_err": float(np.linalg.norm(blue_p - goal_positions["blue"])),
            "green_pos_err": float(np.linalg.norm(green_p - goal_positions["green"])),
            "yellow_pos_err": float(np.linalg.norm(yellow_p - goal_positions["yellow"])),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})