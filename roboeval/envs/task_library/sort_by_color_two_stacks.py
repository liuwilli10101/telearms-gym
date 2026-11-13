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


class ColorStackSortingTask(RoboEvalEnv, MetricRolloutEval):
    """Sort two red and two blue cubes into separate same-color stacks.

Success Criteria:
- The two red cubes form one stack and the two blue cubes form a separate stack.
- Each stack has its base cube on the table and its top cube on the matching-color base, with no mixed bases.
- All four cube colors match expectations and no gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace/table constants
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _CUBE_SIZE = 0.04
    _STACK_DZ = 0.04

    # Success tolerances (task registry mentions 0.014m)
    _XY_TOL = 0.014
    _Z_TOL = 0.02

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY asset allowed)
        self.red_cube_1 = ColoredCube(self._mojo)
        self.red_cube_2 = ColoredCube(self._mojo)
        self.blue_cube_1 = ColoredCube(self._mojo)
        self.blue_cube_2 = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

        self.cubes = [self.red_cube_1, self.red_cube_2, self.blue_cube_1, self.blue_cube_2]

        # Metrics init (FULL signature required)
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
        self.red_cube_1.set_color("red")
        self.red_cube_2.set_color("red")
        self.blue_cube_1.set_color("blue")
        self.blue_cube_2.set_color("blue")

        # 3. Place objects in RANDOM (UNSOLVED) positions.
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Ensure not stacked: keep all cubes at the same rest Z and separated in XY.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        min_xy_sep = 2.2 * self._CUBE_SIZE  # keep cubes well separated to avoid accidental stacking/collisions

        points = []
        max_tries = 2000
        for _ in range(max_tries):
            candidate = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])
            if all(np.linalg.norm(candidate - p) >= min_xy_sep for p in points):
                points.append(candidate)
            if len(points) == 4:
                break

        if len(points) < 4:
            # Fallback deterministic layout within bounds (still unsolved: no stacking, all on table)
            points = [
                np.array([0.44, -0.18]),
                np.array([0.44, 0.18]),
                np.array([0.66, -0.18]),
                np.array([0.66, 0.18]),
            ]

        positions = [np.array([p[0], p[1], z]) for p in points]

        # Optional random yaw; keep upright
        def yaw_quat(yaw: float) -> np.ndarray:
            qq = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)  # elements are [w, x, y, z]
            w, x, y, z_ = qq.elements
            return np.array([x, y, z_, w], dtype=float)

        quats = [yaw_quat(rng.uniform(-np.pi, np.pi)) for _ in range(4)]

        # Place sequentially for physics stability
        self.red_cube_1.set_pose(position=positions[0], quat=quats[0])
        self._mojo.physics.forward()
        self.red_cube_2.set_pose(position=positions[1], quat=quats[1])
        self._mojo.physics.forward()
        self.blue_cube_1.set_pose(position=positions[2], quat=quats[2])
        self._mojo.physics.forward()
        self.blue_cube_2.set_pose(position=positions[3], quat=quats[3])
        self._mojo.physics.forward()

        # 4. Extra settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # (Avoid unused-variable lint; rng may be used by downstream patching)
        _ = rng

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_on(self, top: ColoredCube, bottom: ColoredCube) -> bool:
        """Robust-ish 'on' check using collision + relative pose constraints."""
        if not top.is_colliding(bottom):
            return False

        top_p = top.body.get_position()
        bot_p = bottom.body.get_position()

        # Top cube should not be on the table
        if top.is_colliding(self.table.table_top):
            return False

        # Enforce expected relative placement (stacking)
        xy_dist = np.linalg.norm((top_p - bot_p)[:2])
        z_err = abs((top_p[2] - bot_p[2]) - self._STACK_DZ)

        if xy_dist > self._XY_TOL:
            return False
        if z_err > self._Z_TOL:
            return False
        if top_p[2] <= bot_p[2]:
            return False

        return True

    def _is_on_table(self, cube: ColoredCube) -> bool:
        if not cube.is_colliding(self.table.table_top):
            return False
        p = cube.body.get_position()
        return abs(p[2] - self._CUBE_REST_Z) <= 0.03

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Goal conditions:
        # 1) red_cube_2 on red_cube_1
        # 2) blue_cube_2 on blue_cube_1
        red_stacked = self._is_on(self.red_cube_2, self.red_cube_1)
        blue_stacked = self._is_on(self.blue_cube_2, self.blue_cube_1)

        # Ensure bases are on table (prevents "floating" success)
        red_base_on_table = self._is_on_table(self.red_cube_1)
        blue_base_on_table = self._is_on_table(self.blue_cube_1)

        # 3) red_cube_1 not on blue_cube_1
        # 4) blue_cube_1 not on red_cube_1
        no_mixed_bases = (not self._is_on(self.red_cube_1, self.blue_cube_1)) and (
            not self._is_on(self.blue_cube_1, self.red_cube_1)
        )

        positions_ok = bool(red_stacked and blue_stacked and red_base_on_table and blue_base_on_table and no_mixed_bases)

        # Semantic/content check: verify colors are correct for each named cube
        semantic_ok = (
            self.red_cube_1.get_color() == "red"
            and self.red_cube_2.get_color() == "red"
            and self.blue_cube_1.get_color() == "blue"
            and self.blue_cube_2.get_color() == "blue"
        )

        # Must be released (no cube held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage marking (coarse mapping to the provided 10 subgoals)
        # Mark all as complete only when final configuration is achieved.
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        # Provide a couple of interpretable distances/errors for the two stacks.
        red_p_top = self.red_cube_2.body.get_position()
        red_p_bot = self.red_cube_1.body.get_position()
        blue_p_top = self.blue_cube_2.body.get_position()
        blue_p_bot = self.blue_cube_1.body.get_position()

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_stack_xy": float(np.linalg.norm((red_p_top - red_p_bot)[:2])),
                "blue_stack_xy": float(np.linalg.norm((blue_p_top - blue_p_bot)[:2])),
                "red_blue_base_xy": float(np.linalg.norm((red_p_bot - blue_p_bot)[:2])),
            },
            pose_error={
                "red_stack_dz_err": float(abs((red_p_top[2] - red_p_bot[2]) - self._STACK_DZ)),
                "blue_stack_dz_err": float(abs((blue_p_top[2] - blue_p_bot[2]) - self._STACK_DZ)),
                "red_base_z_err": float(abs(red_p_bot[2] - self._CUBE_REST_Z)),
                "blue_base_z_err": float(abs(blue_p_bot[2] - self._CUBE_REST_Z)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})