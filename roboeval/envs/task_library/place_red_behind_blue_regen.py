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

class RedBehindBluePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place the red cube directly behind the blue cube.

Success Criteria:
- The red cube is behind the blue cube by at least the configured positive-X separation.
- The red and blue cubes are aligned in Y, stable on the table, and within stability tolerances.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace/task parameters
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    _MIN_DELTA_X = 0.05
    _MAX_ABS_DELTA_Y = 0.05
    _MAX_ABS_TILT_RAD = 0.35

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create task objects (ONLY allowed asset: ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.cubes = [self.red_cube, self.blue_cube]

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

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

        # 2) Deterministic RNG (may be used by external pipelines)
        rng = np.random.default_rng(self.seed)
        _ = rng  # keep deterministic pattern without relying on global RNG

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3) Place objects in RANDOM UNSOLVED positions (ensure _success() is False)
        # Required workspace bounds for this reset:
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Make red cube *not behind* blue cube by placing it in front (dx < _MIN_DELTA_X).
        # Also keep a reasonable separation to avoid immediate interpenetration.
        min_sep_xy = 0.06

        for _attempt in range(100):
            blue_x = float(rng.uniform(0.55, x_hi))  # leave room to place red in front within bounds
            blue_y = float(rng.uniform(y_lo, y_hi))

            front_offset = float(rng.uniform(0.06, 0.18))  # red in front of blue
            red_x = float(np.clip(blue_x - front_offset, x_lo, x_hi))
            red_y = float(rng.uniform(y_lo, y_hi))

            # Ensure some XY separation
            if np.linalg.norm(np.array([red_x - blue_x, red_y - blue_y])) >= min_sep_xy:
                blue_pos = np.array([blue_x, blue_y, z_base], dtype=float)
                red_pos = np.array([red_x, red_y, z_base], dtype=float)
                break
        else:
            # Fallback deterministic unsolved placement (still within required bounds)
            blue_pos = np.array([0.62, 0.00, z_base], dtype=float)
            red_pos = np.array([0.48, 0.12, z_base], dtype=float)  # in front + misaligned

        self.blue_cube.set_pose(position=blue_pos, quat=np.array([1, 0, 0, 0]))
        self.red_cube.set_pose(position=red_pos, quat=np.array([1, 0, 0, 0]))

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _tilt_from_quat(quat_xyzw: np.ndarray) -> float:
        """
        Returns how far the cube is from resting flat on ANY of its faces (radians).

        These are plain colour cubes with no up-face requirement, so a cube that
        settled onto a different face is still correctly placed. Measuring only the
        local +Z axis reported 90 degrees for such a cube and failed it; taking the
        best-aligned face normal measures what the task actually cares about.
        NOTE: body.get_quaternion() returns wxyz, so the input is wxyz (despite the name).
        """
        q = Quaternion(quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3])  # w, x, y, z
        world_z = np.array([0.0, 0.0, 1.0])
        best = np.pi
        for axis in (np.array([1.0, 0.0, 0.0]),
                     np.array([0.0, 1.0, 0.0]),
                     np.array([0.0, 0.0, 1.0])):
            axis_in_world = q.rotate(axis)
            cosang = float(np.clip(abs(np.dot(axis_in_world, world_z)), -1.0, 1.0))
            best = min(best, float(np.arccos(cosang)))
        return best

    def _is_on_table_and_stable(self, cube: ColoredCube) -> bool:
        # Table collision is the recommended way to verify "on table"
        if not cube.is_colliding(self.table.table_top):
            return False
        quat = cube.body.get_quaternion()
        tilt = self._tilt_from_quat(np.array(quat))
        return bool(abs(tilt) <= self._MAX_ABS_TILT_RAD)

    def _success(self) -> bool:
        # CRITICAL: reset success flag at start
        self._success_check = False

        # Defensive checks (called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red_pos = np.array(self.red_cube.body.get_position())
        blue_pos = np.array(self.blue_cube.body.get_position())

        dx = float(red_pos[0] - blue_pos[0])
        dy = float(red_pos[1] - blue_pos[1])

        behind_ok = dx >= self._MIN_DELTA_X
        aligned_y_ok = abs(dy) <= self._MAX_ABS_DELTA_Y

        red_on_table_ok = self._is_on_table_and_stable(self.red_cube)
        blue_on_table_ok = self._is_on_table_and_stable(self.blue_cube)

        positions_ok = bool(behind_ok and aligned_y_ok and red_on_table_ok and blue_on_table_ok)
        semantic_ok = True  # no semantic constraints beyond colors, which are set explicitly

        # Released condition (generic across robot configurations)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (lightweight heuristics) ---
        red_held = any(
            self.robot.is_gripper_holding_object(self.red_cube, s) for s in self.robot.grippers
        )
        red_z = float(red_pos[2])

        # 1) grasp
        if red_held:
            self._metric_stage(0, True)

        # 2) lift (lifted above resting height while held)
        if red_held and (red_z > 1.01):
            self._metric_stage(1, True)

        # 3) transport (moved generally behind blue)
        if dx > 0.02:
            self._metric_stage(2, True)

        # 4) align in Y
        if abs(dy) <= self._MAX_ABS_DELTA_Y:
            self._metric_stage(3, True)

        # 5) place (on table + satisfies relative constraints)
        if positions_ok:
            self._metric_stage(4, True)

        # 6) release
        if positions_ok and released:
            self._metric_stage(5, True)

        # Set overall success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distance-to-goal expressed relative to current blue cube pose
        desired_red_xy = np.array([blue_pos[0] + self._MIN_DELTA_X, blue_pos[1]])
        red_xy = red_pos[:2]
        pose_err = float(np.linalg.norm(red_xy - desired_red_xy))

        behind_delta_x_error = float(max(0.0, (blue_pos[0] + self._MIN_DELTA_X) - red_pos[0]))
        y_align_error = float(abs(dy))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "behind_delta_x_error": behind_delta_x_error,
                "y_align_error": y_align_error,
            },
            pose_error={
                "red_xy_to_desired_xy_l2": pose_err,
                "red_minus_blue_dx": dx,
                "red_minus_blue_dy": dy,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})