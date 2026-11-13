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

class FrontLeftToBackRightCubeDiagonal(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a straight diagonal line that runs from the
    front-left of the workspace to the back-right, following the specified axis
    convention (front = lower X, left = higher Y).

    Sequential Sub-Goals:
        1. Grasp any one of the cubes with the left gripper.
        2. Move the grasped cube toward its intended diagonal position.
        3. Place the cube down on the table near the diagonal line.
        4. Release the cube after placement.
        5. Grasp another cube with the right gripper.
        6. Move the grasped cube to refine the diagonal.
        7. Nudge/adjust cubes so they are collinear in XY and ordered from front-left to back-right.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal constraints from specification
    _BOUNDS_X = (0.40, 0.70)
    _BOUNDS_Y = (-0.25, 0.25)

    _MAX_PERP_ERROR_M = 0.02
    _MIN_ADJ_DIST_M = 0.06
    _MAX_ADJ_DIST_M = 0.18
    _MAX_Z_DIFF_M = 0.02

    _REL_TOL_M = 0.02  # ordering tolerance

    _CUBE_Z_ON_TABLE = 0.97
    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly three ColoredCube objects (and ONLY this asset type)
        self.cube_front_left = ColoredCube(self._mojo)
        self.cube_middle = ColoredCube(self._mojo)
        self.cube_back_right = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.cube_front_left.set_color("red")
        self.cube_middle.set_color("green")
        self.cube_back_right.set_color("blue")

        self.cubes = [self.cube_front_left, self.cube_middle, self.cube_back_right]

        # Canonical goal poses (used both for reset goal-state and for metrics)
        # Diagonal direction: X increases while Y decreases.
        self._goal_positions = {
            "front_left": np.array([0.47, 0.10, self._CUBE_Z_ON_TABLE]),
            "middle": np.array([0.55, 0.02, self._CUBE_Z_ON_TABLE]),
            "back_right": np.array([0.63, -0.06, self._CUBE_Z_ON_TABLE]),
        }
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 2. Deterministic RNG (required pattern; used for randomized placement)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset
        self.cube_front_left.set_color("red")
        self.cube_middle.set_color("green")
        self.cube_back_right.set_color("blue")

        # 4. Place objects in RANDOM UNSOLVED positions (guarantee ordering violation)
        #    Success requires: fl.x < mid.x - tol and fl.y > mid.y + tol (and similarly mid vs br).
        #    We randomize within bounds but enforce the opposite relations so _success() is False.
        tries = 0
        while True:
            tries += 1

            # Random middle pose
            mid_x = float(rng.uniform(self._BOUNDS_X[0] + 0.05, self._BOUNDS_X[1] - 0.05))
            mid_y = float(rng.uniform(self._BOUNDS_Y[0] + 0.05, self._BOUNDS_Y[1] - 0.05))

            # Enforce ordering violation in X: make front_left have *larger* X than middle,
            # and back_right have *smaller* X than middle (opposite of goal).
            dx1 = float(rng.uniform(self._REL_TOL_M + 0.01, 0.15))
            dx2 = float(rng.uniform(self._REL_TOL_M + 0.01, 0.15))
            fl_x = float(np.clip(mid_x + dx1, self._BOUNDS_X[0], self._BOUNDS_X[1]))
            br_x = float(np.clip(mid_x - dx2, self._BOUNDS_X[0], self._BOUNDS_X[1]))

            # Also enforce ordering violation in Y: make front_left have *smaller* Y than middle,
            # and back_right have *larger* Y than middle (opposite of goal).
            dy1 = float(rng.uniform(self._REL_TOL_M + 0.01, 0.18))
            dy2 = float(rng.uniform(self._REL_TOL_M + 0.01, 0.18))
            fl_y = float(np.clip(mid_y - dy1, self._BOUNDS_Y[0], self._BOUNDS_Y[1]))
            br_y = float(np.clip(mid_y + dy2, self._BOUNDS_Y[0], self._BOUNDS_Y[1]))

            # If clipping accidentally removed the violation, retry.
            if not (fl_x > mid_x + self._REL_TOL_M and br_x < mid_x - self._REL_TOL_M):
                if tries < 100:
                    continue
            if not (fl_y < mid_y - self._REL_TOL_M and br_y > mid_y + self._REL_TOL_M):
                if tries < 100:
                    continue

            p_fl = np.array([fl_x, fl_y, self._CUBE_Z_ON_TABLE], dtype=np.float64)
            p_mid = np.array([mid_x, mid_y, self._CUBE_Z_ON_TABLE], dtype=np.float64)
            p_br = np.array([br_x, br_y, self._CUBE_Z_ON_TABLE], dtype=np.float64)

            self.cube_front_left.set_pose(position=p_fl, quat=self._goal_quat.copy())
            self.cube_middle.set_pose(position=p_mid, quat=self._goal_quat.copy())
            self.cube_back_right.set_pose(position=p_br, quat=self._goal_quat.copy())
            break

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy(pos: np.ndarray) -> np.ndarray:
        return np.array([float(pos[0]), float(pos[1])], dtype=np.float64)

    def _in_bounds(self, pos: np.ndarray) -> bool:
        x, y = float(pos[0]), float(pos[1])
        return (self._BOUNDS_X[0] <= x <= self._BOUNDS_X[1]) and (self._BOUNDS_Y[0] <= y <= self._BOUNDS_Y[1])

    @staticmethod
    def _perp_distance_point_to_line_xy(p_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
        # Distance from p to infinite line through a->b in XY
        v = b_xy - a_xy
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return float("inf")
        v_hat = v / v_norm
        n_hat = np.array([-v_hat[1], v_hat[0]], dtype=np.float64)  # perpendicular unit normal
        return float(abs(np.dot(p_xy - a_xy, n_hat)))

    def _success(self) -> bool:
        # CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Get positions
        p_fl = self.cube_front_left.body.get_position()
        p_mid = self.cube_middle.body.get_position()
        p_br = self.cube_back_right.body.get_position()

        # Basic bounds + table contact (prevents stacking)
        in_bounds = self._in_bounds(p_fl) and self._in_bounds(p_mid) and self._in_bounds(p_br)
        on_table = (
            self.cube_front_left.is_colliding(self.table.table_top)
            and self.cube_middle.is_colliding(self.table.table_top)
            and self.cube_back_right.is_colliding(self.table.table_top)
        )

        # Height consistency
        zs = np.array([float(p_fl[2]), float(p_mid[2]), float(p_br[2])], dtype=np.float64)
        z_consistent = float(np.max(zs) - np.min(zs)) <= self._MAX_Z_DIFF_M

        # Ordering: front-left -> middle -> back-right means X increases and Y decreases
        # Using tolerance: require clear separation beyond tolerance
        ordering_ok = (
            (p_fl[0] < p_mid[0] - self._REL_TOL_M)
            and (p_fl[1] > p_mid[1] + self._REL_TOL_M)
            and (p_mid[0] < p_br[0] - self._REL_TOL_M)
            and (p_mid[1] > p_br[1] + self._REL_TOL_M)
        )

        # Collinearity in XY: check middle point perpendicular error to the line endpoints
        fl_xy, mid_xy, br_xy = self._xy(p_fl), self._xy(p_mid), self._xy(p_br)
        perp_mid = self._perp_distance_point_to_line_xy(mid_xy, fl_xy, br_xy)
        collinear_ok = perp_mid <= self._MAX_PERP_ERROR_M

        # Ensure middle is between endpoints along the line direction (in projection)
        v = br_xy - fl_xy
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            between_ok = False
        else:
            v_hat = v / v_norm
            t_mid = float(np.dot(mid_xy - fl_xy, v_hat))
            t_br = float(np.dot(br_xy - fl_xy, v_hat))
            between_ok = (0.0 < t_mid < t_br)

        # Spacing constraints (adjacent distances in XY)
        d_fl_mid = float(np.linalg.norm(mid_xy - fl_xy))
        d_mid_br = float(np.linalg.norm(br_xy - mid_xy))
        spacing_ok = (
            (self._MIN_ADJ_DIST_M <= d_fl_mid <= self._MAX_ADJ_DIST_M)
            and (self._MIN_ADJ_DIST_M <= d_mid_br <= self._MAX_ADJ_DIST_M)
        )

        positions_ok = bool(in_bounds and on_table and z_consistent and ordering_ok and collinear_ok and between_ok and spacing_ok)

        # Released gripper(s)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoals: mark all as achieved once final arrangement is correct (simple staging)
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Set success flag
        self._success_check = bool(positions_ok and released)

        # Metrics: distances to canonical goal points (informational; goal is primarily relational)
        goal_fl = self._goal_positions["front_left"]
        goal_mid = self._goal_positions["middle"]
        goal_br = self._goal_positions["back_right"]

        target_distance = {
            "cube_front_left_to_goal_m": float(np.linalg.norm(p_fl - goal_fl)),
            "cube_middle_to_goal_m": float(np.linalg.norm(p_mid - goal_mid)),
            "cube_back_right_to_goal_m": float(np.linalg.norm(p_br - goal_br)),
            "adjacent_dist_fl_mid_xy_m": d_fl_mid,
            "adjacent_dist_mid_br_xy_m": d_mid_br,
        }
        pose_error = {
            "perpendicular_error_middle_to_line_xy_m": float(perp_mid),
            "z_span_m": float(np.max(zs) - np.min(zs)),
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