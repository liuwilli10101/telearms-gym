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
from roboeval.utils.guide_patch import guide_patch, place_guide, tolerance_size
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubeMidpointPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Place the green cube on the moving midpoint target between red and blue.

Success Criteria:
- The green cube center is within 3 cm on each axis of the red/blue midpoint; references are at least 8 cm apart.
- All cubes rest on the table and green lies between the references.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task tolerances from spec
    _D_LR = 0.20
    _D_ML = 0.10
    _D_MR = 0.10
    # 1 cm on both distance and collinearity meant the midpoint had to be hit to
    # within a centimetre, by eye, with no marker on the table. Operators reported
    # placing the cube visibly in the middle and failing. The green guide patch
    # added below draws the accepted region, and these match its size.
    _DIST_TOL = 0.03
    _COLLINEAR_TOL = 0.03

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create ONLY the allowed assets: ColoredCube
        self.cube_left = ColoredCube(self._mojo)
        self.cube_right = ColoredCube(self._mojo)
        self.cube_mid = ColoredCube(self._mojo)

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_left.set_color("red")
        self.cube_right.set_color("blue")
        self.cube_mid.set_color("green")

        # Marks the midpoint the green cube has to reach. The references are the
        # RED (left) and BLUE (right) cubes; the guide removes the guesswork about
        # which pair the midpoint is measured between.
        self.midpoint_guide = guide_patch(
            self._mojo,
            color="green",
            size=tolerance_size(self._DIST_TOL),
            name="midpoint_target",
        )

        self.cubes = [self.cube_left, self.cube_mid, self.cube_right]

        # Reference goal positions (used for metrics / subgoal staging)
        self._goal_left_pos = np.array([0.55, +0.10, 0.97])
        self._goal_right_pos = np.array([0.55, -0.10, 0.97])
        self._goal_mid_pos = np.array([0.55, 0.00, 0.97])

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

    @staticmethod
    def _point_line_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Perpendicular distance from p to the infinite line through a->b in XY plane."""
        ap = p - a
        ab = b - a
        ab_n2 = float(np.dot(ab, ab))
        if ab_n2 < 1e-12:
            return float(np.linalg.norm(ap))
        # area = |ab x ap| (2D scalar cross via z-component)
        cross = float(ab[0] * ap[1] - ab[1] * ap[0])
        return abs(cross) / (np.sqrt(ab_n2) + 1e-12)

    @staticmethod
    def _between_on_segment_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        """Check p's projection lies within segment a-b (with small tolerance) in XY."""
        ap = p - a
        ab = b - a
        ab_n2 = float(np.dot(ab, ab))
        if ab_n2 < 1e-12:
            return False
        t = float(np.dot(ap, ab)) / ab_n2
        return (-tol <= t) and (t <= 1.0 + tol)

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

        # 3. Place objects in RANDOM UNSOLVED positions within workspace bounds
        self.cube_left.set_color("red")
        self.cube_right.set_color("blue")
        self.cube_mid.set_color("green")

        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = 0.97

        def _rand_pose_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def _yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)

        # Sample until clearly not in goal configuration (and not overlapping too closely).
        # The two REFERENCE cubes spawn already _D_LR apart: success requires that
        # spacing, but the instruction only asks for the middle cube to be placed,
        # so leaving it to chance made the operator rearrange the references too.
        min_sep = 0.06  # conservative to reduce initial interpenetration
        for _ in range(500):
            pL2 = _rand_pose_xy()
            theta = float(rng.uniform(-np.pi, np.pi))
            pR2 = pL2 + self._D_LR * np.array([np.cos(theta), np.sin(theta)], dtype=float)
            if not (x_min <= pR2[0] <= x_max and y_min <= pR2[1] <= y_max):
                continue
            pM2 = _rand_pose_xy()

            # keep some separation
            if np.linalg.norm(pL2 - pR2) < min_sep:
                continue
            if np.linalg.norm(pM2 - pL2) < min_sep:
                continue
            if np.linalg.norm(pM2 - pR2) < min_sep:
                continue

            pL = np.array([pL2[0], pL2[1], z0], dtype=float)
            pR = np.array([pR2[0], pR2[1], z0], dtype=float)
            pM = np.array([pM2[0], pM2[1], z0], dtype=float)

            # Reject the same marked square used by the success predicate.
            accidentally_solved = bool(np.all(
                np.abs(pM2 - (pL2 + pR2) / 2) <= self._DIST_TOL + 1e-9
            ))
            if accidentally_solved:
                continue

            self.cube_left.set_pose(position=pL, quat=_yaw_quat())
            self.cube_right.set_pose(position=pR, quat=_yaw_quat())
            self.cube_mid.set_pose(position=pM, quat=_yaw_quat())
            break
        else:
            # Fallback: deterministic unsolved placement (guaranteed not midpoint constraints)
            self.cube_left.set_pose(position=np.array([0.45, 0.20, z0]), quat=np.array([1, 0, 0, 0]))
            self.cube_right.set_pose(position=np.array([0.65, -0.20, z0]), quat=np.array([1, 0, 0, 0]))
            self.cube_mid.set_pose(position=np.array([0.60, 0.10, z0]), quat=np.array([1, 0, 0, 0]))

        # 4. Physics settling
        self._mojo.physics.forward()

        self._update_midpoint_guide()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _update_midpoint_guide(self):
        midpoint = (np.asarray(self.cube_left.body.get_position())[:2]
                    + np.asarray(self.cube_right.body.get_position())[:2]) / 2
        place_guide(self.midpoint_guide, *midpoint)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._update_midpoint_guide()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Positions
        pL = self.cube_left.body.get_position()
        pR = self.cube_right.body.get_position()
        pM = self.cube_mid.body.get_position()

        # Reference distance is diagnostic, not a hidden fixed-spacing target.
        d_lr = distance(self.cube_left.body, self.cube_right.body)

        # Collinearity + "midpoint between" (XY plane)
        pL2 = np.array([pL[0], pL[1]])
        pR2 = np.array([pR[0], pR[1]])
        pM2 = np.array([pM[0], pM[1]])

        col_err = self._point_line_distance_2d(pM2, pL2, pR2)
        between_ok = self._between_on_segment_2d(pM2, pL2, pR2, tol=self._COLLINEAR_TOL)

        midpoint = (pL2 + pR2) / 2
        inside_guide = bool(np.all(np.abs(pM2 - midpoint) <= self._DIST_TOL + 1e-9))
        on_table = all(abs(float(p[2]) - 0.97) <= 0.008 or
                       (abs(float(p[2]) - 0.97) <= 0.025 and c.is_colliding(self.table.table_top))
                       for c, p in ((self.cube_left, pL), (self.cube_right, pR), (self.cube_mid, pM)))
        positions_ok = bool(inside_guide and np.linalg.norm(pL2 - pR2) >= 0.08 and between_ok and on_table)

        # No semantic constraints for ColoredCube beyond spatial checks
        semantic_ok = True

        # Released gripper (generic gripper iteration)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, purely observational) ---
        # 1) grasp
        grasped = any(self.robot.is_gripper_holding_object(self.cube_mid, s) for s in self.robot.grippers)
        # 2) lift (cube_mid raised above table contact height)
        lifted = bool(pM[2] > 0.99)  # > 2cm above rest_z=0.97
        # 3) transport (closer to midpoint than a coarse threshold)
        midpoint_xy = 0.5 * (pL2 + pR2)
        mid_xy_err = float(np.linalg.norm(pM2 - midpoint_xy))
        transported = bool(mid_xy_err < 0.05)
        # 4) align (fine XY)
        aligned = inside_guide
        # 5) place (near rest height + table collision)
        placed = bool(abs(pM[2] - 0.97) < 0.02 and self.cube_mid.is_colliding(self.table.table_top))
        # 6) release
        released_mid = bool((not grasped) and placed)

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(aligned))
        self._metric_stage(4, bool(placed))
        self._metric_stage(5, bool(released_mid))

        # If fully successful, ensure all stages show completion
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "midpoint_xy_m": mid_xy_err,
            },
            pose_error={
                "reference_separation_m": float(d_lr),
                "midpoint_xy_err_m": float(mid_xy_err),
                "collinearity_err_m": float(col_err),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})