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


class CubeZigzagArrangement(RoboEvalEnv, MetricRolloutEval):
    """Arrange red, green, and blue into a left-to-right zigzag: front, back, front.

Success Criteria:
- Red is left of green and green is left of blue by the target Y separation.
- Green is behind red and blue is in front of green by the target X separation.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal condition parameters.
    # The zigzag is defined by ORDER plus a minimum offset, not by exact spacing.
    # Requiring four independent separations to each land inside a +/-2 cm window
    # (Y in [0.08, 0.12] twice, X in [0.03, 0.07] twice) made the task effectively
    # unpassable by teleoperation: operators reported building the correct shape
    # and still failing. These are now one-sided minimums with a loose upper bound
    # that only rules out cubes flung across the table.
    _MIN_SEP_Y = 0.06
    _MAX_SEP_Y = 0.22
    _MIN_SEP_X = 0.03
    _MAX_SEP_X = 0.16

    # Used for per-cube "at goal pose" staging (not part of success definition)
    _POSE_EPS = 0.03

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Objects (ONLY ColoredCube allowed) ---
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]

        # --- Goal (left-to-right along Y, zigzag depth along X) ---
        # Left = +Y, Right = -Y
        # Front (toward robot) = smaller X (toward -X)
        self._goal_pos_red = np.array([0.50, 0.10, 0.97])   # left-front
        self._goal_pos_green = np.array([0.57, 0.00, 0.97]) # middle-back (behind red)
        self._goal_pos_blue = np.array([0.50, -0.10, 0.97]) # right-front (in front of green)
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

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

        # 2. Deterministic RNG (required pattern; validation may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 3. Place objects in RANDOM *unsolved* positions within workspace bounds
        # Workspace: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z0 = 0.97

        def _rand_pose():
            pos = np.array(
                [
                    rng.uniform(x_low, x_high),
                    rng.uniform(y_low, y_high),
                    z0,
                ],
                dtype=float,
            )
            # small random yaw
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements
            return pos, quat

        def _pairwise_separated(ps: np.ndarray, min_dist: float = 0.06) -> bool:
            for i in range(3):
                for j in range(i + 1, 3):
                    if float(np.linalg.norm(ps[i] - ps[j])) < min_dist:
                        return False
            return True

        def _between(val: float, low: float, high: float) -> bool:
            return (val >= low) and (val <= high)

        # Resample until NOT in solved zigzag relation (so _success() is False after reset)
        # Also keep cubes reasonably separated to avoid immediate interpenetration.
        for _ in range(200):
            pr, qr = _rand_pose()
            pg, qg = _rand_pose()
            pb, qb = _rand_pose()

            ps = np.stack([pr, pg, pb], axis=0)
            if not _pairwise_separated(ps, min_dist=0.06):
                continue

            dy_rg = float(pr[1] - pg[1])
            dy_gb = float(pg[1] - pb[1])
            dx_gr = float(pg[0] - pr[0])
            dx_bg = float(pg[0] - pb[0])

            red_left_of_green = _between(
                dy_rg, self._MIN_SEP_Y, self._MAX_SEP_Y
            )
            green_left_of_blue = _between(
                dy_gb, self._MIN_SEP_Y, self._MAX_SEP_Y
            )
            green_behind_red = _between(
                dx_gr, self._MIN_SEP_X, self._MAX_SEP_X
            )
            blue_in_front_of_green = _between(
                dx_bg, self._MIN_SEP_X, self._MAX_SEP_X
            )

            positions_ok = bool(
                red_left_of_green
                and green_left_of_blue
                and green_behind_red
                and blue_in_front_of_green
            )

            if positions_ok:
                continue  # avoid starting solved

            # Found an unsolved configuration
            self.cube_red.set_pose(position=pr, quat=qr)
            self.cube_green.set_pose(position=pg, quat=qg)
            self.cube_blue.set_pose(position=pb, quat=qb)
            break
        else:
            # Fallback (guaranteed unsolved): align all Y equal (breaks left/right relations)
            pr = np.array([0.45, 0.00, z0], dtype=float)
            pg = np.array([0.55, 0.00, z0], dtype=float)
            pb = np.array([0.65, 0.00, z0], dtype=float)
            self.cube_red.set_pose(position=pr, quat=self._goal_quat.copy())
            self.cube_green.set_pose(position=pg, quat=self._goal_quat.copy())
            self.cube_blue.set_pose(position=pb, quat=self._goal_quat.copy())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        # (We cannot directly detect "grasp/transport" events; stages are inferred from placement/relations.)
        for idx in range(10):
            self._metric_stage(idx, False)

        _ = rng  # keep rng referenced (lint-friendly)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _between(val: float, low: float, high: float) -> bool:
        return (val >= low) and (val <= high)

    def _at_goal(self, cube: ColoredCube, goal_pos: np.ndarray) -> bool:
        p = cube.body.get_position()
        return float(np.linalg.norm(p - goal_pos)) <= float(self._POSE_EPS)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        pr = self.cube_red.body.get_position()
        pg = self.cube_green.body.get_position()
        pb = self.cube_blue.body.get_position()

        # --- Goal conditions (relative) ---
        # 1) Red left of Green (along +Y direction)
        dy_rg = float(pr[1] - pg[1])  # positive if red is left of green
        red_left_of_green = self._between(
            dy_rg,
            self._MIN_SEP_Y,
            self._MAX_SEP_Y,
        )

        # 2) Green left of Blue
        dy_gb = float(pg[1] - pb[1])  # positive if green is left of blue
        green_left_of_blue = self._between(
            dy_gb,
            self._MIN_SEP_Y,
            self._MAX_SEP_Y,
        )

        # 3) Green behind Red (behind = +X larger)
        dx_gr = float(pg[0] - pr[0])  # positive if green is behind red
        green_behind_red = self._between(
            dx_gr,
            self._MIN_SEP_X,
            self._MAX_SEP_X,
        )

        # 4) Blue in front of Green (front = -X smaller)
        dx_bg = float(pg[0] - pb[0])  # positive if blue is in front of green
        blue_in_front_of_green = self._between(
            dx_bg,
            self._MIN_SEP_X,
            self._MAX_SEP_X,
        )

        positions_ok = bool(
            red_left_of_green
            and green_left_of_blue
            and green_behind_red
            and blue_in_front_of_green
        )

        # No semantic constraints beyond colors (already fixed via set_color)
        semantic_ok = True

        # Released: no cube held in any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging (inferred from placement/relations) ---
        red_at = self._at_goal(self.cube_red, self._goal_pos_red)
        green_at = self._at_goal(self.cube_green, self._goal_pos_green)
        blue_at = self._at_goal(self.cube_blue, self._goal_pos_blue)

        # 1-3: red placed
        if red_at:
            self._metric_stage(0, True)
            self._metric_stage(1, True)
            self._metric_stage(2, True)

        # 4-6: green placed
        if green_at:
            self._metric_stage(3, True)
            self._metric_stage(4, True)
            self._metric_stage(5, True)

        # 7-9: blue placed
        if blue_at:
            self._metric_stage(6, True)
            self._metric_stage(7, True)
            self._metric_stage(8, True)

        # 10: final alignment (all relations satisfied + released)
        if positions_ok and released:
            self._metric_stage(9, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize (MUST be last metric call before return) ---
        target_distance = {
            "dy_red_left_of_green_err": abs(dy_rg - self._MIN_SEP_Y),
            "dy_green_left_of_blue_err": abs(dy_gb - self._MIN_SEP_Y),
            "dx_green_behind_red_err": abs(dx_gr - self._MIN_SEP_X),
            "dx_blue_in_front_of_green_err": abs(dx_bg - self._MIN_SEP_X),
        }
        pose_error = {
            "red_to_goal": float(np.linalg.norm(pr - self._goal_pos_red)),
            "green_to_goal": float(np.linalg.norm(pg - self._goal_pos_green)),
            "blue_to_goal": float(np.linalg.norm(pb - self._goal_pos_blue)),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})