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

class ColorPairRowSorting(RoboEvalEnv, MetricRolloutEval):
    """Sort six cubes into red, green, and blue pairs arranged as a left-to-right row.

Success Criteria:
- Six cubes form three same-color pairs: red, green, and blue, with each pair arranged front-to-back along X.
- The three pairs are placed in a left-to-right Y row with the front cubes aligned in X and all cubes on the table.
- All colors match their expected identities and no gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # ---- Goal parameters (from task specification) ----
    _REST_Z = 0.97
    _PAIR_DX_RANGE = (0.04, 0.08)
    _PAIR_TOL_Y = 0.03
    _PAIR_TOL_Z = 0.03

    _SLOT_TOL_XY = (0.05, 0.05)
    _SLOT_X = 0.55
    _SLOT_Y_RED = 0.18     # left (+Y)
    _SLOT_Y_GREEN = 0.0    # middle
    _SLOT_Y_BLUE = -0.18   # right (-Y)

    _ROW_MIN_DELTA_Y = 0.10  # for left_of along +Y

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create 6 colored cubes (ONLY allowed asset)
        self.red_cube_1 = ColoredCube(self._mojo)
        self.red_cube_2 = ColoredCube(self._mojo)
        self.green_cube_1 = ColoredCube(self._mojo)
        self.green_cube_2 = ColoredCube(self._mojo)
        self.blue_cube_1 = ColoredCube(self._mojo)
        self.blue_cube_2 = ColoredCube(self._mojo)

        self.cubes = [
            self.red_cube_1, self.red_cube_2,
            self.green_cube_1, self.green_cube_2,
            self.blue_cube_1, self.blue_cube_2,
        ]

        # REQUIRED: set_color() in BOTH _initialize_env AND _on_reset
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.green_cube_1.set_color("green")
        self.green_cube_2.set_color("green")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

        # Precompute goal poses (pair = front cube at slot, back cube behind along +X)
        self._goal_xy = {
            "red_cube_1": np.array([self._SLOT_X, self._SLOT_Y_RED]),
            "green_cube_1": np.array([self._SLOT_X, self._SLOT_Y_GREEN]),
            "blue_cube_1": np.array([self._SLOT_X, self._SLOT_Y_BLUE]),
        }
        self._goal_dx = 0.06  # within [0.04, 0.08]
        self._goal_pos = {
            "red_cube_1": np.array([self._SLOT_X, self._SLOT_Y_RED, self._REST_Z]),
            "red_cube_2": np.array([self._SLOT_X + self._goal_dx, self._SLOT_Y_RED, self._REST_Z]),
            "green_cube_1": np.array([self._SLOT_X, self._SLOT_Y_GREEN, self._REST_Z]),
            "green_cube_2": np.array([self._SLOT_X + self._goal_dx, self._SLOT_Y_GREEN, self._REST_Z]),
            "blue_cube_1": np.array([self._SLOT_X, self._SLOT_Y_BLUE, self._REST_Z]),
            "blue_cube_2": np.array([self._SLOT_X + self._goal_dx, self._SLOT_Y_BLUE, self._REST_Z]),
        }

        # Initialize metrics (FULL signature required)
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

        # 2) Deterministic RNG (required pattern; may be used by the validation randomizer)
        rng = np.random.default_rng(self.seed)  # noqa: F841

        # 3) REQUIRED: set_color() in BOTH _initialize_env AND _on_reset
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.green_cube_1.set_color("green")
        self.green_cube_2.set_color("green")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

        # 4) Place objects in RANDOM unsolved positions (ensure _success() == False)
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._REST_Z

        objs = [
            self.red_cube_1, self.red_cube_2,
            self.green_cube_1, self.green_cube_2,
            self.blue_cube_1, self.blue_cube_2,
        ]

        def _yaw_quat(yaw: float) -> np.ndarray:
            # Quaternion for rotation about Z (x,y,z,w)
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        def _sample_positions(n: int, min_dist: float = 0.055, max_tries: int = 2000) -> np.ndarray:
            pts = []
            tries = 0
            while len(pts) < n and tries < max_tries:
                tries += 1
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                    pts.append(p)
            if len(pts) < n:
                # Fallback (still within bounds)
                pts = [np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float) for _ in range(n)]
            return np.stack(pts, axis=0)

        max_resets = 100
        for _ in range(max_resets):
            xy = _sample_positions(len(objs))
            yaws = rng.uniform(-np.pi, np.pi, size=len(objs))

            for i, obj in enumerate(objs):
                pos = np.array([xy[i, 0], xy[i, 1], z], dtype=float)
                obj.set_pose(position=pos, quat=_yaw_quat(float(yaws[i])))

            # Let physics/collisions update before checking "unsolved"
            self._mojo.physics.forward()

            # Replicate the spatial portion of _success() without calling it (avoid metric finalization)
            red_pair_ok = self._check_back_of(
                self.red_cube_2, self.red_cube_1,
                delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
            )
            green_pair_ok = self._check_back_of(
                self.green_cube_2, self.green_cube_1,
                delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
            )
            blue_pair_ok = self._check_back_of(
                self.blue_cube_2, self.blue_cube_1,
                delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
            )

            blue_slot_ok = self._check_xy_near(self.blue_cube_1, self._goal_xy["blue_cube_1"], self._SLOT_TOL_XY)
            green_slot_ok = self._check_xy_near(self.green_cube_1, self._goal_xy["green_cube_1"], self._SLOT_TOL_XY)
            red_slot_ok = self._check_xy_near(self.red_cube_1, self._goal_xy["red_cube_1"], self._SLOT_TOL_XY)

            green_left_of_blue = self._check_left_of(self.green_cube_1, self.blue_cube_1, self._ROW_MIN_DELTA_Y)
            red_left_of_green = self._check_left_of(self.red_cube_1, self.green_cube_1, self._ROW_MIN_DELTA_Y)

            on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

            positions_ok = bool(
                red_pair_ok and green_pair_ok and blue_pair_ok
                and red_slot_ok and green_slot_ok and blue_slot_ok
                and red_left_of_green and green_left_of_blue
                and on_table
            )

            if not positions_ok:
                break

        # 5) Physics settling
        self._mojo.physics.forward()

        # 6) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ---- Helper checks ----
    def _p(self, obj) -> np.ndarray:
        return obj.body.get_position()

    def _check_back_of(self, subject, target, delta_range, tol_y, tol_z) -> bool:
        ps = self._p(subject)
        pt = self._p(target)
        dx = ps[0] - pt[0]  # +X is "back"
        return (delta_range[0] <= dx <= delta_range[1]) and (abs(ps[1] - pt[1]) <= tol_y) and (abs(ps[2] - pt[2]) <= tol_z)

    def _check_xy_near(self, obj, target_xy, tol_xy) -> bool:
        p = self._p(obj)
        return (abs(p[0] - target_xy[0]) <= tol_xy[0]) and (abs(p[1] - target_xy[1]) <= tol_xy[1])

    def _check_left_of(self, subject, target, min_delta_y) -> bool:
        # "left" is +Y
        ps = self._p(subject)
        pt = self._p(target)
        return (ps[1] - pt[1]) >= min_delta_y

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Pair relations (back_of along +X / axis 0)
        red_pair_ok = self._check_back_of(
            self.red_cube_2, self.red_cube_1,
            delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
        )
        green_pair_ok = self._check_back_of(
            self.green_cube_2, self.green_cube_1,
            delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
        )
        blue_pair_ok = self._check_back_of(
            self.blue_cube_2, self.blue_cube_1,
            delta_range=self._PAIR_DX_RANGE, tol_y=self._PAIR_TOL_Y, tol_z=self._PAIR_TOL_Z
        )

        # Row ordering along Y (left_of uses axis 1 / +Y) — relative ordering, not slots
        green_left_of_blue = self._check_left_of(self.green_cube_1, self.blue_cube_1, self._ROW_MIN_DELTA_Y)
        red_left_of_green = self._check_left_of(self.red_cube_1, self.green_cube_1, self._ROW_MIN_DELTA_Y)

        # Row alignment: front cubes of all three pairs aligned in X (same front-line),
        # so the three pairs sit in a single row across Y rather than scattered.
        front_xs = [
            float(self._p(self.red_cube_1)[0]),
            float(self._p(self.green_cube_1)[0]),
            float(self._p(self.blue_cube_1)[0]),
        ]
        row_x_aligned = (max(front_xs) - min(front_xs)) <= 0.05

        # Basic sanity: cubes should be supported by the table surface
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        positions_ok = bool(
            red_pair_ok and green_pair_ok and blue_pair_ok
            and red_left_of_green and green_left_of_blue
            and row_x_aligned
            and on_table
        )

        # Diagnostic-only slot proximity flags (not gating success)
        blue_slot_ok = self._check_xy_near(self.blue_cube_1, self._goal_xy["blue_cube_1"], self._SLOT_TOL_XY)
        green_slot_ok = self._check_xy_near(self.green_cube_1, self._goal_xy["green_cube_1"], self._SLOT_TOL_XY)
        red_slot_ok = self._check_xy_near(self.red_cube_1, self._goal_xy["red_cube_1"], self._SLOT_TOL_XY)

        # Color semantics are ensured by construction, but still verify
        semantic_ok = (
            self.red_cube_1.get_color() == "red"
            and self.red_cube_2.get_color() == "red"
            and self.green_cube_1.get_color() == "green"
            and self.green_cube_2.get_color() == "green"
            and self.blue_cube_1.get_color() == "blue"
            and self.blue_cube_2.get_color() == "blue"
        )

        # Must be released (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging (spatial proxies)
        if red_pair_ok:
            self._metric_stage(0, True)
        if red_pair_ok and red_slot_ok:
            self._metric_stage(1, True)
        if green_pair_ok:
            self._metric_stage(2, True)
        if green_pair_ok and green_slot_ok:
            self._metric_stage(3, True)
        if blue_pair_ok:
            self._metric_stage(4, True)
        if blue_pair_ok and blue_slot_ok:
            self._metric_stage(5, True)
        if red_left_of_green and green_left_of_blue:
            self._metric_stage(6, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "red_cube_1_xy": float(np.linalg.norm(self._p(self.red_cube_1)[:2] - self._goal_xy["red_cube_1"])),
            "green_cube_1_xy": float(np.linalg.norm(self._p(self.green_cube_1)[:2] - self._goal_xy["green_cube_1"])),
            "blue_cube_1_xy": float(np.linalg.norm(self._p(self.blue_cube_1)[:2] - self._goal_xy["blue_cube_1"])),
        }
        pose_error = {
            "red_pair_dx_error": float(
                abs((self._p(self.red_cube_2)[0] - self._p(self.red_cube_1)[0]) - self._goal_dx)
            ),
            "green_pair_dx_error": float(
                abs((self._p(self.green_cube_2)[0] - self._p(self.green_cube_1)[0]) - self._goal_dx)
            ),
            "blue_pair_dx_error": float(
                abs((self._p(self.blue_cube_2)[0] - self._p(self.blue_cube_1)[0]) - self._goal_dx)
            ),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})