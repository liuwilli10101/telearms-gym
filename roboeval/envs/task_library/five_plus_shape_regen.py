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

class PlusSignCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange five colored cubes on the tabletop into a plus-sign shape: the red cube is the
    center, and the other four cubes (blue, orange, green, purple) form the four arms
    (front, back, left, right) around it.

    Sequential Sub-Goals:
        1. Grasp any cube to begin arranging the plus sign.
        2. Keep the red cube on the table as the center of the plus sign.
        3. Place one arm cube on any arm of the plus.
        4. Place a second arm cube on another arm.
        5. Place a third arm cube on another arm.
        6. Place the fourth arm cube on the last remaining arm.
        7. Refine alignment so the four arm cubes are axis-aligned with the center.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace bounds requested by task spec (stricter than global bounds)
    _X_MIN, _X_MAX = 0.40, 0.70
    _Y_MIN, _Y_MAX = -0.25, 0.25

    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97  # REQUIRED resting Z for ColoredCube set_pose

    _ARM_DIST = 0.06
    _ARM_TOL = 0.02
    _ALIGN_TOL = 0.02
    _MAX_CENTER_HEIGHT_ABOVE_TABLE = 0.03

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create task objects (MUST use ONLY ColoredCube) ---
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_orange = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_purple = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_orange.set_color("orange")
        self.cube_green.set_color("green")
        self.cube_purple.set_color("purple")

        self.cubes = [
            self.cube_red,
            self.cube_blue,
            self.cube_orange,
            self.cube_green,
            self.cube_purple,
        ]

        # Precompute goal poses
        self.init_object_poses()

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

    def init_object_poses(self):
        """
        Define the deterministic GOAL positions (plus-sign around the red center).
        """
        center = np.array([0.55, 0.00, self._CUBE_REST_Z])

        # Axis convention: front = -X, back = +X, left = +Y, right = -Y
        self._goal_pos = {
            "red": center,
            "blue": center + np.array([-self._ARM_DIST, 0.0, 0.0]),     # front (-X)
            "orange": center + np.array([+self._ARM_DIST, 0.0, 0.0]),   # back (+X)
            "green": center + np.array([0.0, +self._ARM_DIST, 0.0]),    # left (+Y)
            "purple": center + np.array([0.0, -self._ARM_DIST, 0.0]),   # right (-Y)
        }

        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 2) Deterministic RNG handle (required pattern; not used for goal placement)
        rng = np.random.default_rng(self.seed)

        # 3) REQUIRED: set_color() in _on_reset
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_orange.set_color("orange")
        self.cube_green.set_color("green")
        self.cube_purple.set_color("purple")

        # --- Randomize initial poses in UNSOLVED configuration (must make _success() False) ---
        z = float(self._CUBE_REST_Z)
        min_sep = 0.055  # keep cubes from starting overlapped (approx cube size + margin)

        def rand_quat_yaw() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        def sample_xy() -> np.ndarray:
            x = float(rng.uniform(self._X_MIN, self._X_MAX))
            y = float(rng.uniform(self._Y_MIN, self._Y_MAX))
            return np.array([x, y], dtype=float)

        def far_enough(pxy: np.ndarray, others_xy: list[np.ndarray]) -> bool:
            return all(float(np.linalg.norm(pxy - o)) >= min_sep for o in others_xy)

        # Rejection sample a configuration that is NOT a plus-sign (within tolerances)
        max_tries = 500
        for _ in range(max_tries):
            xy_list: list[np.ndarray] = []
            # Sample 5 non-overlapping XY points
            for _k in range(5):
                ok = False
                for _j in range(300):
                    pxy = sample_xy()
                    if far_enough(pxy, xy_list):
                        xy_list.append(pxy)
                        ok = True
                        break
                if not ok:
                    break
            if len(xy_list) != 5:
                continue

            red_p = np.array([xy_list[0][0], xy_list[0][1], z], dtype=float)
            blue_p = np.array([xy_list[1][0], xy_list[1][1], z], dtype=float)
            orange_p = np.array([xy_list[2][0], xy_list[2][1], z], dtype=float)
            green_p = np.array([xy_list[3][0], xy_list[3][1], z], dtype=float)
            purple_p = np.array([xy_list[4][0], xy_list[4][1], z], dtype=float)

            # Ensure NOT already a complete plus (order-free over the four arm cubes)
            _covered, _all_on_arm = self._plus_arms_status(
                red_p, [blue_p, orange_p, green_p, purple_p]
            )
            if _all_on_arm and len(_covered) == 4:
                continue  # would be solved; resample

            # Commit sampled poses
            self.cube_red.set_pose(position=red_p, quat=rand_quat_yaw())
            self.cube_blue.set_pose(position=blue_p, quat=rand_quat_yaw())
            self.cube_orange.set_pose(position=orange_p, quat=rand_quat_yaw())
            self.cube_green.set_pose(position=green_p, quat=rand_quat_yaw())
            self.cube_purple.set_pose(position=purple_p, quat=rand_quat_yaw())
            break
        else:
            # Fallback: deterministic unsolved spread (still within workspace)
            base = np.array([0.45, -0.20, z], dtype=float)
            self.cube_red.set_pose(position=base + np.array([0.00, 0.00, 0.0]), quat=rand_quat_yaw())
            self.cube_blue.set_pose(position=base + np.array([0.20, 0.00, 0.0]), quat=rand_quat_yaw())
            self.cube_orange.set_pose(position=base + np.array([0.00, 0.20, 0.0]), quat=rand_quat_yaw())
            self.cube_green.set_pose(position=base + np.array([0.20, 0.20, 0.0]), quat=rand_quat_yaw())
            self.cube_purple.set_pose(position=base + np.array([0.10, 0.10, 0.0]), quat=rand_quat_yaw())

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _in_workspace_xy(self, p: np.ndarray) -> bool:
        return (self._X_MIN <= float(p[0]) <= self._X_MAX) and (self._Y_MIN <= float(p[1]) <= self._Y_MAX)

    def _on_table_not_stacked(self, cube: ColoredCube) -> bool:
        """
        On tabletop (collision with table top) and not lifted/stacked high.
        Uses task's max height above table constraint.
        """
        p = cube.body.get_position()
        if not self._in_workspace_xy(p):
            return False

        # Must be on the table (table collision is reliable)
        if not cube.is_colliding(self.table.table_top):
            return False

        # Height constraint: cube center should be close to table surface
        # (table surface is z=0.95; cube rest center z=0.97)
        if float(p[2]) > self._TABLE_Z + self._MAX_CENTER_HEIGHT_ABOVE_TABLE:
            return False

        # If a floor exists, ensure it's not on the floor
        if hasattr(self, "_floor") and cube.is_colliding(self._floor):
            return False

        return True

    def _check_relative(self, center_p: np.ndarray, other_p: np.ndarray,
                        axis: int, direction_sign: int,
                        distance_m: float, tolerance_m: float,
                        aligned_axis: int, aligned_tolerance_m: float) -> bool:
        """
        Check that other is at center + direction_sign * distance along `axis`,
        and aligned (close) along `aligned_axis`.
        """
        delta = other_p - center_p
        desired = float(direction_sign) * float(distance_m)

        axis_ok = abs(float(delta[axis]) - desired) <= float(tolerance_m)
        align_ok = abs(float(delta[aligned_axis])) <= float(aligned_tolerance_m)
        return bool(axis_ok and align_ok)

    # The four plus-sign arms, each as (axis, direction_sign, aligned_axis):
    #   front (-X), back (+X), left (+Y), right (-Y).
    _ARM_SPECS = ((0, -1, 1), (0, +1, 1), (1, +1, 0), (1, -1, 0))

    def _plus_arms_status(self, center_p: np.ndarray, arm_positions: list[np.ndarray]):
        """
        Order-free arm check. For the given arm cubes (any color order), return:
          - covered: the set of distinct arm indices (0..3) that are occupied
          - all_on_arm: True if EVERY arm cube sits on some valid arm

        The arm tolerances are small relative to the arm distance, so a cube can match
        at most one arm; a complete plus is `all_on_arm and len(covered) == 4`.
        """
        covered: set[int] = set()
        all_on_arm = True
        for cp in arm_positions:
            hit = None
            for si, (axis, sign, aligned_axis) in enumerate(self._ARM_SPECS):
                if self._check_relative(
                    center_p=center_p, other_p=cp,
                    axis=axis, direction_sign=sign,
                    distance_m=self._ARM_DIST, tolerance_m=self._ARM_TOL,
                    aligned_axis=aligned_axis, aligned_tolerance_m=self._ALIGN_TOL,
                ):
                    hit = si
                    break
            if hit is None:
                all_on_arm = False
            else:
                covered.add(hit)
        return covered, all_on_arm

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_pos"):
            return False

        red_p = self.cube_red.body.get_position()
        blue_p = self.cube_blue.body.get_position()
        orange_p = self.cube_orange.body.get_position()
        green_p = self.cube_green.body.get_position()
        purple_p = self.cube_purple.body.get_position()

        # Arrangement/on-table constraints
        on_table_ok = all(self._on_table_not_stacked(c) for c in self.cubes)

        # Order-free arm constraint: red is the center, and the four other cubes must each
        # occupy a distinct arm (+X / -X / +Y / -Y). Which colored cube lands on which arm
        # does NOT matter — the operator never has to reason about axis signs by color.
        arms_covered, all_on_arm = self._plus_arms_status(
            red_p, [blue_p, orange_p, green_p, purple_p]
        )
        num_arms = len(arms_covered)
        arms_ok = bool(all_on_arm and num_arms == 4)

        positions_ok = bool(on_table_ok and arms_ok)

        # Released gripper constraint
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging ---
        # 0: grasp any cube
        grasp_any = any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )
        self._metric_stage(0, bool(grasp_any))

        # 1: red established as the center (on the table; the plus is defined relative to it)
        self._metric_stage(1, self._on_table_not_stacked(self.cube_red))

        # 2-5: progressively more arms covered (order-free)
        self._metric_stage(2, num_arms >= 1)
        self._metric_stage(3, num_arms >= 2)
        self._metric_stage(4, num_arms >= 3)
        self._metric_stage(5, num_arms >= 4)

        # 6: alignment/refinement (all conditions)
        self._metric_stage(6, bool(positions_ok))

        # Final success
        self._success_check = bool(positions_ok and released)

        # Metrics: distance to goal for each cube (XY + Z)
        target_distance = {
            "cube_red": float(np.linalg.norm(self.cube_red.body.get_position() - self._goal_pos["red"])),
            "cube_blue": float(np.linalg.norm(self.cube_blue.body.get_position() - self._goal_pos["blue"])),
            "cube_orange": float(np.linalg.norm(self.cube_orange.body.get_position() - self._goal_pos["orange"])),
            "cube_green": float(np.linalg.norm(self.cube_green.body.get_position() - self._goal_pos["green"])),
            "cube_purple": float(np.linalg.norm(self.cube_purple.body.get_position() - self._goal_pos["purple"])),
        }

        pose_error = {
            "center_xy_error_m": float(np.linalg.norm(red_p[:2] - self._goal_pos["red"][:2])),
            "blue_xy_error_m": float(np.linalg.norm(blue_p[:2] - self._goal_pos["blue"][:2])),
            "orange_xy_error_m": float(np.linalg.norm(orange_p[:2] - self._goal_pos["orange"][:2])),
            "green_xy_error_m": float(np.linalg.norm(green_p[:2] - self._goal_pos["green"][:2])),
            "purple_xy_error_m": float(np.linalg.norm(purple_p[:2] - self._goal_pos["purple"][:2])),
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