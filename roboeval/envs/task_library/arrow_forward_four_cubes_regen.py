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

class ForwardArrowCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Rearrange four colored cubes on the table to form an arrow shape pointing forward
    (toward the robot/camera), i.e., in the -X direction.

    The arrow consists of:
      - Tip cube (frontmost, smallest X)
      - Two wing cubes behind the tip (+X) aligned in X and split left/right in Y
        (left = +Y, right = -Y)
      - Tail cube furthest back (largest X) aligned in Y with the tip

    Sequential Sub-Goals:
        1. Grasp the cube designated as the arrow tip.
        2. Place the tip cube at the front of the arrangement (smaller x than the others).
        3. Grasp the cube designated as the left wing.
        4. Place the left wing behind the tip and to the left (+Y).
        5. Grasp the cube designated as the right wing.
        6. Place the right wing behind the tip and to the right (-Y).
        7. Grasp the cube designated as the arrow tail.
        8. Place the tail behind the wings and centered in Y with the tip.
        9. Fine-align positions so the arrow is symmetric in Y and ordered along X.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal condition parameters (from task spec)
    _MIN_X_SEP = 0.03
    _X_ALIGN_TOL = 0.02
    _Y_ALIGN_TOL = 0.02
    _MIN_Y_WING_SEP = 0.04
    _Y_SYM_TOL = 0.02

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube is allowed)
        self.cube_tip = ColoredCube(self._mojo)
        self.cube_left_wing = ColoredCube(self._mojo)
        self.cube_right_wing = ColoredCube(self._mojo)
        self.cube_tail = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_tip,
            self.cube_left_wing,
            self.cube_right_wing,
            self.cube_tail,
        ]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_tip.set_color("red")
        self.cube_left_wing.set_color("blue")
        self.cube_right_wing.set_color("green")
        self.cube_tail.set_color("yellow")

        # Define deterministic goal poses (workspace-aligned)
        # Arrow points "forward" = -X, so tip has smallest X.
        self._goal_pos = {
            "cube_tip": np.array([0.47, 0.00, 0.97]),
            "cube_left_wing": np.array([0.53, +0.06, 0.97]),   # left = +Y
            "cube_right_wing": np.array([0.53, -0.06, 0.97]),  # right = -Y
            "cube_tail": np.array([0.59, 0.00, 0.97]),
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (even if unused here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_tip.set_color("red")
        self.cube_left_wing.set_color("blue")
        self.cube_right_wing.set_color("green")
        self.cube_tail.set_color("yellow")

        # --- Random UNSOLVED placement within workspace bounds ---
        # Workspace bounds
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Separation to avoid overlaps
        min_sep_xy = 0.06

        def _rand_y():
            return float(rng.uniform(y_lo, y_hi))

        def _rand_yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        def _too_close(p, pts):
            for q in pts:
                if np.linalg.norm((p[:2] - q[:2])) < min_sep_xy:
                    return True
            return False

        # Force an unsolved configuration by ensuring the "tip" is NOT the frontmost cube:
        # Make tip have a large X, while other cubes have smaller X (violates c1/c2).
        max_tries = 200
        placed = False
        for _ in range(max_tries):
            pts = []

            # Sample three non-tip cubes in the lower X range
            def _sample_non_tip():
                for __ in range(200):
                    p = np.array(
                        [rng.uniform(x_lo, 0.62), _rand_y(), z_base],
                        dtype=float,
                    )
                    if not _too_close(p, pts):
                        pts.append(p)
                        return p
                return None

            p_lw = _sample_non_tip()
            p_rw = _sample_non_tip()
            p_tail = _sample_non_tip()
            if p_lw is None or p_rw is None or p_tail is None:
                continue

            # Sample tip cube in the higher X range and not too close to others
            p_tip = None
            for __ in range(200):
                p = np.array(
                    [rng.uniform(0.63, x_hi), _rand_y(), z_base],
                    dtype=float,
                )
                if (not _too_close(p, pts)) and (p[0] >= max(p_lw[0], p_rw[0]) + self._MIN_X_SEP):
                    p_tip = p
                    break
            if p_tip is None:
                continue

            # Additional guard: avoid accidentally meeting the "tail behind wings" ordering too cleanly
            # by ensuring tail isn't guaranteed to be the farthest back.
            # (Still random, but this reduces rare chance of satisfying all predicates.)
            if p_tail[0] > max(p_lw[0], p_rw[0]) + 0.10:
                continue

            # Apply randomized yaw orientations (not required by success, but allowed)
            self.cube_tip.set_pose(position=p_tip, quat=_rand_yaw_quat())
            self.cube_left_wing.set_pose(position=p_lw, quat=_rand_yaw_quat())
            self.cube_right_wing.set_pose(position=p_rw, quat=_rand_yaw_quat())
            self.cube_tail.set_pose(position=p_tail, quat=_rand_yaw_quat())

            placed = True
            break

        if not placed:
            # Fallback (still unsolved): fixed non-goal arrangement within bounds
            self.cube_tip.set_pose(position=np.array([0.68, 0.10, z_base]), quat=self._goal_quat)
            self.cube_left_wing.set_pose(position=np.array([0.50, -0.10, z_base]), quat=self._goal_quat)
            self.cube_right_wing.set_pose(position=np.array([0.52, 0.00, z_base]), quat=self._goal_quat)
            self.cube_tail.set_pose(position=np.array([0.48, 0.18, z_base]), quat=self._goal_quat)

        # Small settle
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(9):
            self._metric_stage(idx, False)

        # Keep rng referenced to avoid lint/unused confusion in some pipelines
        _ = rng

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _axis_index_from_name(axis: str) -> int:
        if axis.lower() == "x":
            return 0
        if axis.lower() == "y":
            return 1
        if axis.lower() == "z":
            return 2
        raise ValueError(f"Unknown axis: {axis}")

    def _in_front_of(self, subject_pos: np.ndarray, target_pos: np.ndarray,
                     axis_idx: int, direction_sign: str, min_sep: float) -> bool:
        # direction_sign '-' means "front" is decreasing axis value.
        if direction_sign == "-":
            return (target_pos[axis_idx] - subject_pos[axis_idx]) >= min_sep
        return (subject_pos[axis_idx] - target_pos[axis_idx]) >= min_sep

    def _left_of(self, subject_pos: np.ndarray, target_pos: np.ndarray,
                 axis_idx: int, direction_sign: str, min_sep: float) -> bool:
        # For this task, left/right is ONLY Y-axis. direction_sign '+' means left is higher Y.
        if direction_sign == "+":
            return (subject_pos[axis_idx] - target_pos[axis_idx]) >= min_sep
        return (target_pos[axis_idx] - subject_pos[axis_idx]) >= min_sep

    def _aligned(self, a: np.ndarray, b: np.ndarray, axis_idx: int, tol: float) -> bool:
        return abs(float(a[axis_idx] - b[axis_idx])) <= tol

    def _symmetric_about(self, left_pos: np.ndarray, right_pos: np.ndarray,
                         center_pos: np.ndarray, axis_idx: int, tol: float) -> bool:
        # Symmetry about center on the given axis:
        # midpoint(left,right) ~= center, and equal offsets
        midpoint = 0.5 * (left_pos[axis_idx] + right_pos[axis_idx])
        offsets_equal = abs(float((left_pos[axis_idx] - center_pos[axis_idx]) -
                                  (center_pos[axis_idx] - right_pos[axis_idx]))) <= tol
        midpoint_ok = abs(float(midpoint - center_pos[axis_idx])) <= tol
        return bool(midpoint_ok and offsets_equal)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        tip_p = self.cube_tip.body.get_position()
        lw_p = self.cube_left_wing.body.get_position()
        rw_p = self.cube_right_wing.body.get_position()
        tail_p = self.cube_tail.body.get_position()

        # Table contact sanity (avoid fallen cubes)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # Goal conditions (from spec)
        x_idx = self._axis_index_from_name("x")
        y_idx = self._axis_index_from_name("y")

        c1 = self._in_front_of(tip_p, lw_p, axis_idx=x_idx, direction_sign="-", min_sep=self._MIN_X_SEP)
        c2 = self._in_front_of(tip_p, rw_p, axis_idx=x_idx, direction_sign="-", min_sep=self._MIN_X_SEP)
        c3 = self._in_front_of(lw_p, tail_p, axis_idx=x_idx, direction_sign="-", min_sep=self._MIN_X_SEP)
        c4 = self._in_front_of(rw_p, tail_p, axis_idx=x_idx, direction_sign="-", min_sep=self._MIN_X_SEP)

        c5 = self._aligned(lw_p, rw_p, axis_idx=x_idx, tol=self._X_ALIGN_TOL)
        c6 = self._left_of(lw_p, rw_p, axis_idx=y_idx, direction_sign="+", min_sep=self._MIN_Y_WING_SEP)
        c7 = self._aligned(tip_p, tail_p, axis_idx=y_idx, tol=self._Y_ALIGN_TOL)
        c8 = self._symmetric_about(lw_p, rw_p, tip_p, axis_idx=y_idx, tol=self._Y_SYM_TOL)

        positions_ok = bool(on_table and c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8)

        # Spatial task: no semantic content check needed
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (best-effort using spatial predicates)
        # 0 grasp tip (not directly observable) -> mark when tip is correctly "front" vs wings
        self._metric_stage(0, bool(c1 and c2))
        # 1 place tip at front
        self._metric_stage(1, bool(c1 and c2))
        # 2 grasp left wing (not observable) -> mark when left/right ordering achieved
        self._metric_stage(2, bool(c6))
        # 3 place left wing
        self._metric_stage(3, bool(c1 and c3 and c6))
        # 4 grasp right wing (not observable)
        self._metric_stage(4, bool(c6 and c5))
        # 5 place right wing
        self._metric_stage(5, bool(c2 and c4 and c5 and c6))
        # 6 grasp tail (not observable)
        self._metric_stage(6, bool(c3 and c4))
        # 7 place tail
        self._metric_stage(7, bool(c3 and c4 and c7))
        # 8 align all / finalize symmetry
        self._metric_stage(8, bool(c5 and c7 and c8 and c1 and c2 and c3 and c4))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distances to goal positions (per cube)
        def _pos_err(obj, goal):
            p = obj.body.get_position()
            return float(np.linalg.norm(p - goal))

        target_distance = {
            "cube_tip_to_goal": _pos_err(self.cube_tip, self._goal_pos["cube_tip"]),
            "cube_left_wing_to_goal": _pos_err(self.cube_left_wing, self._goal_pos["cube_left_wing"]),
            "cube_right_wing_to_goal": _pos_err(self.cube_right_wing, self._goal_pos["cube_right_wing"]),
            "cube_tail_to_goal": _pos_err(self.cube_tail, self._goal_pos["cube_tail"]),
        }

        pose_error = {
            "tip_pos_err": target_distance["cube_tip_to_goal"],
            "left_wing_pos_err": target_distance["cube_left_wing_to_goal"],
            "right_wing_pos_err": target_distance["cube_right_wing_to_goal"],
            "tail_pos_err": target_distance["cube_tail_to_goal"],
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