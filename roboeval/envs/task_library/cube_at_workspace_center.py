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

class CubeCenterPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Move the blue cube onto the marked centre of the workspace.

Success Criteria:
The cube center must land inside the white square drawn at the workspace centre, resting on the table and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace definition (given by task constraints)
    _WORKSPACE_CENTER_XY = np.array([0.55, 0.0], dtype=float)
    _CUBE_REST_Z = 0.97

    # "Workspace centre" is a bare coordinate: there is nothing on the table at
    # (0.55, 0) for an operator to aim at, so the accepted region is drawn as a guide
    # patch below and this is its half-width. One tolerance for the success check,
    # the reset margin and the drawing -- the previous 0.014 m survived in the reset
    # margin and in the align subgoal after the check itself had been relaxed to
    # 0.06, which let a reset spawn an already-solved cube and made subgoal 3
    # unreachable on runs that succeeded.
    _XY_TOL_M = 0.06

    # Heuristics for subgoal staging
    _LIFT_Z_DELTA_M = 0.02
    _TRANSPORT_RADIUS_M = 0.10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Draws the accepted region so the target is not invisible.
        self.center_guide = guide_patch(
            self._mojo,
            color="white",
            size=tolerance_size(self._XY_TOL_M),
            name="workspace_center_target",
        )

        # Create props (ONLY allowed asset: ColoredCube)
        self.colored_cube = ColoredCube(self._mojo)
        self.colored_cube.set_color("blue")  # REQUIRED in _initialize_env

        self.cubes = [self.colored_cube]  # For metrics API compatibility

        # Metrics (FULL signature REQUIRED)
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
        # Draw the goal region every reset (props reset to default poses).
        place_guide(self.center_guide, *self._WORKSPACE_CENTER_XY)

        # Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG (kept for framework consistency; pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset as well
        self.colored_cube.set_color("blue")

        # Place cube in RANDOM UNSOLVED state within workspace bounds:
        # X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Ensure it's not already within success tolerance of the workspace center.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Clear of the accepted region, so the reset is never already solved.
        min_unsolved_dist = self._XY_TOL_M + 0.04
        for _ in range(1000):
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            if float(np.linalg.norm(np.array([x, y]) - self._WORKSPACE_CENTER_XY)) > min_unsolved_dist:
                break

        # Random yaw, keep cube upright. set_pose feeds this to Quaternion(), which
        # reads [w, x, y, z]; re-ordering to xyzw turned the yaw into a 180 deg flip
        # about a tilted axis and reset the cube lying over.
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)

        start_pos = np.array([x, y, self._CUBE_REST_Z], dtype=float)
        self.colored_cube.set_pose(position=start_pos, quat=quat.elements)

        # Let physics settle
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "colored_cube") or not hasattr(self, "table"):
            return False

        cube_pos = self.colored_cube.body.get_position()
        cube_xy = np.array([cube_pos[0], cube_pos[1]], dtype=float)
        xy_err = float(np.linalg.norm(cube_xy - self._WORKSPACE_CENTER_XY))

        # Check "resting on table"
        # Prefer table-top collision if available; fall back to Z proximity.
        on_table = False
        if hasattr(self.table, "table_top"):
            on_table = bool(self.colored_cube.is_colliding(self.table.table_top))
        z_close = abs(float(cube_pos[2]) - self._CUBE_REST_Z) <= 0.02
        on_table = bool(on_table or z_close)

        # Region check: cube inside the drawn central region.
        # Workspace bounds X[0.40, 0.70], Y[-0.25, 0.25]; center ~(0.55, 0.0).
        in_center_region = xy_err <= self._XY_TOL_M
        positions_ok = bool(in_center_region and on_table)
        semantic_ok = True  # No semantic conditions for this task

        held = any(
            self.robot.is_gripper_holding_object(self.colored_cube, g)
            for g in getattr(self.robot, "grippers", [])
        )
        released = not held

        # Subgoal staging (heuristic, monotonic via current state)
        lifted = float(cube_pos[2]) >= (self._CUBE_REST_Z + self._LIFT_Z_DELTA_M)
        near_center = xy_err <= self._TRANSPORT_RADIUS_M
        aligned_in_air = (xy_err <= self._XY_TOL_M) and lifted
        placed = positions_ok

        self._metric_stage(0, bool(held))             # grasp
        self._metric_stage(1, bool(lifted))           # lift
        self._metric_stage(2, bool(lifted and near_center))  # transport
        self._metric_stage(3, bool(aligned_in_air))   # align
        self._metric_stage(4, bool(placed))           # place
        self._metric_stage(5, bool(placed and released))  # release

        # Overall success: centered on table AND released
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_xy_to_workspace_center_m": xy_err,
            },
            pose_error={
                "xy_error_m": xy_err,
                "z_error_m": float(abs(float(cube_pos[2]) - self._CUBE_REST_Z)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})