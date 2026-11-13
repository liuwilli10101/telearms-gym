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


class ColoredCubeRowAlignment(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a single left-to-right row along the Y-axis.
    The row must be evenly spaced with red on the left (+Y), blue in the middle,
    and green on the right (-Y).

    Sequential Sub-Goals:
        1. Grasp the blue cube.
        2. Place the blue cube at the intended middle position.
        3. Grasp the red cube.
        4. Place the red cube to the left of the blue cube (+Y) with proper spacing.
        5. Grasp the green cube.
        6. Place the green cube to the right of the blue cube (-Y) with proper spacing.
        7. Fine-tune alignment: same X and even spacing along Y in red-blue-green order.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace / goal parameters
    _X_TOL = 0.02
    _SPACING_TOL = 0.02
    _MIN_SEP = 0.04
    _Z_TABLE = 0.97

    def _initialize_env(self):
        # Table exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube]
        self._expected_colors = {
            "red_cube": "red",
            "blue_cube": "blue",
            "green_cube": "green",
        }

        # Goal: evenly spaced along Y, constant X
        self._goal_x = 0.55
        self._goal_spacing = 0.05  # >= 0.04 min separation, stable and within workspace
        self._goal_positions = {
            "red_cube": np.array([self._goal_x, +self._goal_spacing, self._Z_TABLE]),
            "blue_cube": np.array([self._goal_x, 0.0, self._Z_TABLE]),
            "green_cube": np.array([self._goal_x, -self._goal_spacing, self._Z_TABLE]),
        }
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization handle (pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() returns False)
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z = self._Z_TABLE

        def _sample_positions():
            # Sample distinct Y positions to reduce interpenetration; keep within bounds.
            ys = rng.uniform(y_low, y_high, size=3)
            for _ in range(50):
                if np.min(np.abs(ys[:, None] - ys[None, :]) + np.eye(3) * 1e9) > 0.03:
                    break
                ys = rng.uniform(y_low, y_high, size=3)

            xs = rng.uniform(x_low, x_high, size=3)
            pr = np.array([xs[0], ys[0], z])
            pb = np.array([xs[1], ys[1], z])
            pg = np.array([xs[2], ys[2], z])
            return pr, pb, pg

        pr, pb, pg = _sample_positions()

        # Reject rare chance of landing in solved arrangement; force a guaranteed failure if needed.
        for _ in range(100):
            xs = np.array([pr[0], pb[0], pg[0]])
            order_ok = (pr[1] - pb[1] > self._MIN_SEP) and (pb[1] - pg[1] > self._MIN_SEP)
            x_align_ok = (np.max(xs) - np.min(xs)) <= self._X_TOL
            d_rb = abs(pr[1] - pb[1])
            d_bg = abs(pb[1] - pg[1])
            spacing_ok = abs(d_rb - d_bg) <= self._SPACING_TOL

            if not (order_ok and x_align_ok and spacing_ok):
                break
            pr, pb, pg = _sample_positions()

        # Hard guarantee: violate min-separation/order while staying in workspace bounds.
        xs = np.array([pr[0], pb[0], pg[0]])
        order_ok = (pr[1] - pb[1] > self._MIN_SEP) and (pb[1] - pg[1] > self._MIN_SEP)
        x_align_ok = (np.max(xs) - np.min(xs)) <= self._X_TOL
        d_rb = abs(pr[1] - pb[1])
        d_bg = abs(pb[1] - pg[1])
        spacing_ok = abs(d_rb - d_bg) <= self._SPACING_TOL
        if order_ok and x_align_ok and spacing_ok:
            x0 = float(rng.uniform(x_low, x_high))
            pr = np.array([x0, 0.00, z])
            pb = np.array([x0, 0.01, z])   # too close (violates _MIN_SEP and order)
            pg = np.array([x0, -0.01, z])  # too close (violates _MIN_SEP)

        self.red_cube.set_pose(position=pr, quat=self._goal_quat)
        self.blue_cube.set_pose(position=pb, quat=self._goal_quat)
        self.green_cube.set_pose(position=pg, quat=self._goal_quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        self._num_subgoals = 7
        for idx in range(self._num_subgoals):
            self._metric_stage(idx, False)

        # Keep rng "used" deterministically (avoid lint / deterministic divergence across tasks)
        _ = rng.random()

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

        # Current positions
        pr = self.red_cube.body.get_position()
        pb = self.blue_cube.body.get_position()
        pg = self.green_cube.body.get_position()

        # Semantic/content check (colors must match identity)
        semantic_ok = (
            self.red_cube.get_color() == self._expected_colors["red_cube"]
            and self.blue_cube.get_color() == self._expected_colors["blue_cube"]
            and self.green_cube.get_color() == self._expected_colors["green_cube"]
        )

        # Released check (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Table contact / height sanity (on table, not stacked)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)
        z_ok = all(abs(c.body.get_position()[2] - self._Z_TABLE) < 0.02 for c in self.cubes)

        # Order constraints along Y: left-to-right is +Y to -Y
        order_ok = (pr[1] - pb[1] > self._MIN_SEP) and (pb[1] - pg[1] > self._MIN_SEP)

        # Collinear along Y with approximately same X
        xs = np.array([pr[0], pb[0], pg[0]])
        x_align_ok = (np.max(xs) - np.min(xs)) <= self._X_TOL

        # Even spacing along Y: |(red-blue) - (blue-green)| <= tol
        d_rb = abs(pr[1] - pb[1])
        d_bg = abs(pb[1] - pg[1])
        spacing_ok = abs(d_rb - d_bg) <= self._SPACING_TOL

        positions_ok = bool(on_table and z_ok and order_ok and x_align_ok and spacing_ok)

        # Subgoal staging (coarse: mark all complete when final arrangement is correct)
        if positions_ok and semantic_ok and released:
            for i in range(self._num_subgoals):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        target_distance = {
            "red_to_goal": float(np.linalg.norm(pr - self._goal_positions["red_cube"])),
            "blue_to_goal": float(np.linalg.norm(pb - self._goal_positions["blue_cube"])),
            "green_to_goal": float(np.linalg.norm(pg - self._goal_positions["green_cube"])),
        }
        pose_error = {
            "x_span": float(np.max(xs) - np.min(xs)),
            "spacing_error": float(abs(d_rb - d_bg)),
            "min_sep_violation_rb": float(max(0.0, self._MIN_SEP - (pr[1] - pb[1]))),
            "min_sep_violation_bg": float(max(0.0, self._MIN_SEP - (pb[1] - pg[1]))),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})