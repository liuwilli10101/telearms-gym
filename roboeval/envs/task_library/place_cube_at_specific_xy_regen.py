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

class CubeForwardRightPlacement(RoboEvalEnv, MetricRolloutEval):
    """Place the cube at the forward-right target point.

Success Criteria:
- The cube XY position is within tolerance of the configured forward-right target coordinate.
- The cube is colliding with the table top.
- No gripper is holding the cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (from spec) ---
    _TARGET_XY = np.array([0.50, -0.10], dtype=float)
    _TOL_XY = 0.03

    # Workspace bounds for sampling (from spec; within global workspace constraints)
    _X_MIN, _X_MAX = 0.40, 0.70
    _Y_MIN, _Y_MAX = -0.25, 0.25

    _CUBE_Z_ON_TABLE = 0.97
    _TABLE_Z = 0.95

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # The goal is a bare coordinate with nothing drawn at it, so operators had
        # no way to tell where "forward right" is. Mark the accepted region.
        self.target_guide = guide_patch(
            self._mojo,
            color="white",
            size=tolerance_size(self._TOL_XY),
            name="forward_right_target",
        )

        # Single required asset: ColoredCube
        self.colored_cube_1 = ColoredCube(self._mojo)
        self.colored_cube_1.set_color("blue")  # REQUIRED in _initialize_env

        # Metric interface expects a list
        self.cubes = [self.colored_cube_1]

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

    def _sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample a reachable, non-goal XY position on the table within workspace bounds.

        NOTE: The validation pipeline may apply additional randomization; this helper enforces:
          - axis convention: X=position[0], Y=position[1]
          - avoid sampling inside the goal tolerance region (non-goal start)
        """
        for _ in range(200):
            x = float(rng.uniform(self._X_MIN, self._X_MAX))
            y = float(rng.uniform(self._Y_MIN, self._Y_MAX))
            xy = np.array([x, y], dtype=float)

            # Exclude near-goal region (keep initial state unsolved when used)
            if np.linalg.norm(xy - self._TARGET_XY) > (self._TOL_XY + 0.02):
                return xy

        # Fallback (still within bounds)
        return np.array([0.62, 0.12], dtype=float)

    def _on_reset(self):
        # Draw the goal region every reset (props reset to default poses).
        place_guide(self.target_guide, self._TARGET_XY[0], self._TARGET_XY[1])

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

        # 2. Deterministic RNG (may be used by downstream randomization/bench harness)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False after reset)
        self.colored_cube_1.set_color("blue")  # REQUIRED in _on_reset

        start_xy = self._sample_initial_state(rng)
        start_pos = np.array([start_xy[0], start_xy[1], self._CUBE_Z_ON_TABLE], dtype=float)

        # Random yaw about Z; keep cube upright
        yaw = float(rng.uniform(-np.pi, np.pi))
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
        start_quat = np.array([qw, 0.0, 0.0, qz], dtype=float)

        self.colored_cube_1.set_pose(position=start_pos, quat=start_quat)

        # Keep a sampled (unsolved) initial state handy for external randomizers
        self._sampled_start_xy = start_xy

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "colored_cube_1"):
            return False

        cube = self.colored_cube_1

        # --- Position check (axis convention enforced explicitly) ---
        pos = cube.body.get_position()
        xy = np.array([pos[0], pos[1]], dtype=float)  # X=position[0], Y=position[1]
        xy_err = xy - self._TARGET_XY
        xy_dist = float(np.linalg.norm(xy_err))

        within_xy = (abs(xy_err[0]) <= self._TOL_XY) and (abs(xy_err[1]) <= self._TOL_XY)

        # Table contact (use table.table_top; reliable collision direction)
        on_table = cube.is_colliding(self.table.table_top)

        positions_ok = bool(within_xy and on_table)

        # No semantic constraints for ColoredCube beyond being placeable
        semantic_ok = True

        # Released: must not be held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers
        )

        # --- Subgoal staging (best-effort from state; finalized on full success) ---
        # (0) grasp: inferred if currently held
        grasped = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
        self._metric_stage(0, bool(grasped or released))  # allow released-at-goal to count

        # (1) lift: inferred if not colliding with table top (i.e., off table)
        lifted = not on_table
        self._metric_stage(1, bool(lifted or released))

        # (2) transport: inferred by being near target in XY (even if held)
        near_target = xy_dist <= (self._TOL_XY + 0.01)
        self._metric_stage(2, bool(near_target))

        # (3) place: on table and near target
        placed = bool(on_table and near_target)
        self._metric_stage(3, bool(placed))

        # (4) release: released (and ideally placed)
        self._metric_stage(4, bool(released and placed))

        # Overall success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # If success, mark all stages as complete
        if self._success_check:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "colored_cube_1_xy": xy_dist,
            },
            pose_error={
                "colored_cube_1_x_err": float(xy_err[0]),
                "colored_cube_1_y_err": float(xy_err[1]),
                "colored_cube_1_xy_dist": xy_dist,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})