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

class CubeForwardMoveTask(RoboEvalEnv, MetricRolloutEval):
    """
    Move a single colored cube 10 cm forward toward the robot from its initial position on the table.
    The task succeeds when the cube’s final position is displaced by the specified amount along the
    toward-robot direction.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the colored cube.
        2. Lift the cube slightly to clear the table for transport.
        3. Move the cube toward the robot by approximately 10 cm.
        4. Place the cube back down on the table at the new position.
        5. Release the cube after placement.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal parameters (from task spec)
    _DISPLACEMENT_M = 0.10
    _TOLERANCE_M = 0.02

    def _compute_toward_robot_direction(self) -> np.ndarray:
        """
        Coordinate convention in this benchmark:
        - Table is generally at +X relative to the robot/camera.
        - "Toward robot" corresponds to decreasing X.
        """
        v = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        return v / (np.linalg.norm(v) + 1e-12)

    def _spawn_colored_cube(self):
        # Asset constraint: MUST use this exact instantiation pattern
        self.colored_cube_0 = ColoredCube(self._mojo)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create task object(s) (ONLY allowed asset: ColoredCube)
        self._spawn_colored_cube()
        self.colored_cube_0.set_color("blue")  # REQUIRED in _initialize_env

        # Keep a canonical list for metrics APIs
        self.cubes = [self.colored_cube_0]

        # Reference pose storage (initial_pose reference is captured/defined via these)
        self._reference_initial_pos = None
        self._target_displacement_dir = self._compute_toward_robot_direction()

        # ⚠️ CRITICAL: Initialize metrics with FULL signature
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True
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
            track_collisions=True
        )

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3. Randomize an UNSOLVED initial pose within workspace bounds
        init_x = float(rng.uniform(0.40, 0.70))
        init_y = float(rng.uniform(-0.25, 0.25))
        init_pos = np.array([init_x, init_y, 0.97], dtype=np.float64)

        # Set reference initial pose to the randomized start pose so success requires a +0.10m
        # displacement along toward-robot direction (i.e., -X). At reset, displacement is ~0 => unsolved.
        self._target_displacement_dir = self._compute_toward_robot_direction()
        self._reference_initial_pos = init_pos.copy()

        # REQUIRED in _on_reset
        self.colored_cube_0.set_color("blue")

        # Random yaw about Z for variety (keeps cube on table)
        yaw = float(rng.uniform(-np.pi, np.pi))
        q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        # Place cube at randomized initial pose
        self.colored_cube_0.set_pose(position=init_pos, quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "colored_cube_0") or self._reference_initial_pos is None:
            return False

        cur_pos = np.array(self.colored_cube_0.body.get_position(), dtype=np.float64)
        displacement = cur_pos - self._reference_initial_pos

        # Displacement along toward-robot direction
        proj = float(np.dot(displacement, self._target_displacement_dir))
        proj_err = abs(proj - self._DISPLACEMENT_M)

        positions_ok = (proj_err <= self._TOLERANCE_M)

        # Subgoal staging: only mark achieved when the overall goal condition is met
        if positions_ok:
            for i in range(5):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok)

        # Finalize metrics (MUST be last metric call before return)
        orth = displacement - proj * self._target_displacement_dir
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "colored_cube_0_proj_error_m": float(proj_err),
                "colored_cube_0_orth_error_m": float(np.linalg.norm(orth)),
            },
            pose_error={
                "colored_cube_0": float(proj_err),
            }
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})