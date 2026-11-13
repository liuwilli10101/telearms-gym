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


class OppositeCornerCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place two cubes at opposite workspace corners: red cube 1 front-left and green cube 2 back-right.

The visible red and green patches mark the two success regions. In this workspace,
front is smaller X, back is larger X, left is larger Y, and right is smaller Y.

Success Criteria:
- red cube_1 is in the red front-left workspace patch and green cube_2 is in the green back-right workspace patch, within the configured position tolerance.
- Both cubes are resting on the table at the expected cube height and colliding with the table top.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace bounds (task proposal bounds for random initial sampling)
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)

    # Goal regions (front-left and back-right) per task specification
    _REGION_FL = {"x_range": (0.40, 0.48), "y_range": (0.12, 0.25)}
    _REGION_BR = {"x_range": (0.62, 0.70), "y_range": (-0.25, -0.12)}
    _PATCH_Z = 0.951

    # Cube/table constants
    _CUBE_REST_Z = 0.97
    _ON_TABLE_Z_TOL = 0.03  # accept small variation while still "on table"
    _MIN_CUBE_SPACING = 0.05  # registry guidance

    # Region success tolerance (in addition to being inside the region bounds)
    # The patch is drawn at the accepted region including this tolerance, so a
    # cube that looks inside the patch is inside. Alpha was also raised from
    # 0.35 to 0.65: at 0.35 the targets read as barely-there against the table.
    _POS_TOL = 0.025

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly the required assets: two ColoredCubes
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("green")

        self.front_left_patch = self._create_success_patch(
            self._REGION_FL, color=np.array([1.0, 0.0, 0.0, 0.65]), name="front_left_red_target"
        )
        self.back_right_patch = self._create_success_patch(
            self._REGION_BR, color=np.array([0.0, 0.75, 0.0, 0.65]), name="back_right_green_target"
        )
        self.patches = [self.front_left_patch, self.back_right_patch]
        self._place_goal_patches()

        # Nominal goal poses (used for metrics / reference)
        self._goal_pose_cube_1 = np.array([0.44, 0.20, self._CUBE_REST_Z])  # inside front-left
        self._goal_pose_cube_2 = np.array([0.66, -0.20, self._CUBE_REST_Z])  # inside back-right

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

        # 2. Deterministic RNG (kept for downstream pipelines; do not remove)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("green")
        self._place_goal_patches()

        # 3. Place objects in RANDOM UNSOLVED positions (avoid goal regions so _success() is False)
        poses = self._sample_initial_object_poses(rng)

        # Randomize yaw on table (upright)
        yaw1 = float(rng.uniform(-np.pi, np.pi))
        yaw2 = float(rng.uniform(-np.pi, np.pi))
        # set_pose feeds this to Quaternion(), which reads [w, x, y, z]. Re-ordering
        # to xyzw turned the yaw into a 180 deg flip about a tilted axis, so cubes
        # reset lying over instead of upright.
        quat1 = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw1).elements
        quat2 = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw2).elements

        self.cube_1.set_pose(position=poses[0].copy(), quat=quat1)
        self.cube_2.set_pose(position=poses[1].copy(), quat=quat2)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(8):
            self._metric_stage(idx, False)

    def _success_ranges(self, region: Dict) -> Dict:
        return {
            "x_range": (
                region["x_range"][0] - self._POS_TOL,
                region["x_range"][1] + self._POS_TOL,
            ),
            "y_range": (
                region["y_range"][0] - self._POS_TOL,
                region["y_range"][1] + self._POS_TOL,
            ),
        }

    def _create_success_patch(self, region: Dict, color: np.ndarray, name: str) -> GoalPatch:
        region = self._success_ranges(region)
        x0, x1 = region["x_range"]
        y0, y1 = region["y_range"]
        half_extents = ((x1 - x0) / 2.0, (y1 - y0) / 2.0)
        return GoalPatch(
            self._mojo,
            color=color,
            size=half_extents,
            name=name,
            collision_enabled=False,
        )

    def _region_center(self, region: Dict) -> np.ndarray:
        region = self._success_ranges(region)
        x0, x1 = region["x_range"]
        y0, y1 = region["y_range"]
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, self._PATCH_Z], dtype=float)

    def _place_goal_patches(self):
        self.front_left_patch.set_position(self._region_center(self._REGION_FL))
        self.back_right_patch.set_position(self._region_center(self._REGION_BR))

    def _get_goal_region_parameters(self) -> Dict:
        return {
            "frame": "workspace",
            "axis_interpretation": {
                "front": "-X",
                "back": "+X",
                "left": "+Y",
                "right": "-Y",
            },
            "targets": {
                "cube_1": {
                    "cube_color": "red",
                    "target_patch_color": "red",
                    "region_name": "front-left",
                    "x_range": [float(v) for v in self._REGION_FL["x_range"]],
                    "y_range": [float(v) for v in self._REGION_FL["y_range"]],
                    "accepted_x_range": [float(v) for v in self._success_ranges(self._REGION_FL)["x_range"]],
                    "accepted_y_range": [float(v) for v in self._success_ranges(self._REGION_FL)["y_range"]],
                    "center_xyz": self._region_center(self._REGION_FL).tolist(),
                },
                "cube_2": {
                    "cube_color": "green",
                    "target_patch_color": "green",
                    "region_name": "back-right",
                    "x_range": [float(v) for v in self._REGION_BR["x_range"]],
                    "y_range": [float(v) for v in self._REGION_BR["y_range"]],
                    "accepted_x_range": [float(v) for v in self._success_ranges(self._REGION_BR)["x_range"]],
                    "accepted_y_range": [float(v) for v in self._success_ranges(self._REGION_BR)["y_range"]],
                    "center_xyz": self._region_center(self._REGION_BR).tolist(),
                },
            },
            "position_tolerance": float(self._POS_TOL),
            "require_on_table": True,
            "visual_patches": True,
        }

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _inside_region_xy(self, pos: np.ndarray, x_range, y_range, tol: float = 0.0) -> bool:
        return (
            (x_range[0] - tol) <= float(pos[0]) <= (x_range[1] + tol)
            and (y_range[0] - tol) <= float(pos[1]) <= (y_range[1] + tol)
        )

    def _is_on_table(self, cube: ColoredCube) -> bool:
        pos = cube.body.get_position()
        z_ok = abs(float(pos[2]) - self._CUBE_REST_Z) <= self._ON_TABLE_Z_TOL
        # Table collision is reliable using table_top
        on_table_contact = cube.is_colliding(self.table.table_top)
        return bool(z_ok and on_table_contact)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        p1 = self.cube_1.body.get_position()
        p2 = self.cube_2.body.get_position()

        cube1_in_fl = self._inside_region_xy(
            p1, self._REGION_FL["x_range"], self._REGION_FL["y_range"], tol=self._POS_TOL
        )
        cube2_in_br = self._inside_region_xy(
            p2, self._REGION_BR["x_range"], self._REGION_BR["y_range"], tol=self._POS_TOL
        )

        cube1_on_table = self._is_on_table(self.cube_1)
        cube2_on_table = self._is_on_table(self.cube_2)

        positions_ok = bool(cube1_in_fl and cube2_in_br and cube1_on_table and cube2_on_table)
        semantic_ok = True  # no semantic constraints beyond spatial in this task

        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging (approximate via observable end conditions)
        if cube1_in_fl and cube1_on_table:
            for idx in range(4):  # subgoals 1-4 for cube_1
                self._metric_stage(idx, True)
        if cube2_in_br and cube2_on_table:
            for idx in range(4, 8):  # subgoals 5-8 for cube_2
                self._metric_stage(idx, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: report distances to nominal goal pose centers
        d1 = float(np.linalg.norm(p1 - self._goal_pose_cube_1))
        d2 = float(np.linalg.norm(p2 - self._goal_pose_cube_2))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"cube_1_to_goal": d1, "cube_2_to_goal": d2},
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # -------------------------------------------------------------------------
    # Optional hook for evaluation pipelines that randomize initial states.
    # (Not called by _on_reset(), which must place the goal state per spec.)
    # -------------------------------------------------------------------------
    def _sample_initial_object_poses(self, rng: np.random.Generator):
        """
        Sample two non-goal initial poses within x∈[0.40,0.70], y∈[-0.25,0.25],
        avoiding the goal regions and maintaining minimum inter-cube spacing.
        """

        def in_goal_region(pos_xy):
            x, y = float(pos_xy[0]), float(pos_xy[1])
            in_fl = (self._REGION_FL["x_range"][0] <= x <= self._REGION_FL["x_range"][1]) and (
                self._REGION_FL["y_range"][0] <= y <= self._REGION_FL["y_range"][1]
            )
            in_br = (self._REGION_BR["x_range"][0] <= x <= self._REGION_BR["x_range"][1]) and (
                self._REGION_BR["y_range"][0] <= y <= self._REGION_BR["y_range"][1]
            )
            return in_fl or in_br

        max_tries = 200
        poses = []
        for _ in range(2):
            for _t in range(max_tries):
                x = rng.uniform(self._X_BOUNDS[0], self._X_BOUNDS[1])
                y = rng.uniform(self._Y_BOUNDS[0], self._Y_BOUNDS[1])
                candidate = np.array([x, y], dtype=float)

                if in_goal_region(candidate):
                    continue

                if poses:
                    prev = poses[0][:2]
                    if float(np.linalg.norm(candidate - prev)) < self._MIN_CUBE_SPACING:
                        continue

                poses.append(np.array([x, y, self._CUBE_REST_Z], dtype=float))
                break

        if len(poses) != 2:
            # Fallback deterministic non-goal positions (still within bounds)
            poses = [
                np.array([0.58, 0.00, self._CUBE_REST_Z], dtype=float),
                np.array([0.52, -0.08, self._CUBE_REST_Z], dtype=float),
            ]

        return poses
