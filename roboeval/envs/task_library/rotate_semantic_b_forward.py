# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

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
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube
class RotateBFaceForwardTask(RoboEvalEnv, MetricRolloutEval):
    """Rotate the semantic cube so B faces the robot.

Success Criteria:
- The semantic cube is inside the workspace, on the table, and near expected table height.
- The B face is readable in the configured forward-facing direction.
- No gripper is holding the cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # In this benchmark coordinate/camera convention, "toward the robot (forward)" corresponds to the
    # cube face with outward normal pointing along -X being visible/readable to the robot/camera.
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Workspace bounds (per task constraints)
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)
    _Z_ON_TABLE = 0.97  # SemanticCube base Z on table surface

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create exactly one SemanticCube using ensure_cubes() workflow
        cubes_info = ensure_cubes("B", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["B"]
        self.semantic_cube = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
        self.cubes = [self.semantic_cube]

        # Cache goal pose (filled in _on_reset)
        self._goal_pos = np.array([0.55, 0.0, self._Z_ON_TABLE], dtype=float)
        self._goal_quat = None  # pyquaternion.Quaternion

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

        # 2. Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # 3. Randomize initial cube position within required workspace bounds (UNSOLVED start)
        x_bounds = (0.40, 0.70)
        y_bounds = (-0.25, 0.25)
        pos = np.array(
            [
                float(rng.uniform(*x_bounds)),
                float(rng.uniform(*y_bounds)),
                float(self._Z_ON_TABLE),
            ],
            dtype=float,
        )

        # Keep a fixed nominal goal pose for metrics/orientation error (do NOT overwrite with start pose)
        self._goal_pos = np.array([0.55, 0.0, self._Z_ON_TABLE], dtype=float)

        # Compute the goal quaternion (B face forward) for metric pose error, but do NOT set it as initial pose
        face_idx = self.semantic_cube._labels.index("B")
        q_goal = self.semantic_cube.get_orientation_for_face_and_glyph(
            face_idx=face_idx,
            face_dir_world=self._FACE_DIR,
        )
        self._goal_quat = q_goal

        # Randomize initial orientation, rejecting any that would already satisfy the semantic goal
        # (ensures _success() returns False immediately after reset)
        q_init = None
        for _ in range(200):
            # Deterministic random quaternion (pyquaternion.Quaternion.random has no rng kwarg in this runtime)
            q_arr = rng.normal(size=4).astype(float)
            n = float(np.linalg.norm(q_arr))
            if n < 1e-12:
                q_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                q_arr = q_arr / n
            q_candidate = Quaternion(q_arr)
            self.semantic_cube.set_pose(position=pos, quat=q_candidate.elements)
            # Ensure B is NOT readable facing robot-forward at reset
            if not bool(self.semantic_cube.is_face_readable("B", face_dir=self._FACE_DIR)):
                q_init = q_candidate
                break
        if q_init is None:
            # Deterministic fallback: rotate the goal by 90 degrees about world-Z to force misalignment
            q_init = Quaternion(axis=[0.0, 0.0, 1.0], angle=np.pi / 2.0) * q_goal
            self.semantic_cube.set_pose(position=pos, quat=q_init.elements)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False (3 subgoals)
        for idx in range(3):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or self._goal_quat is None:
            return False

        cube = self.semantic_cube
        pos = cube.body.get_position()

        # --- Spatial checks (lightweight; primary goal is orientation) ---
        in_workspace = (
            (self._X_BOUNDS[0] <= pos[0] <= self._X_BOUNDS[1])
            and (self._Y_BOUNDS[0] <= pos[1] <= self._Y_BOUNDS[1])
        )
        on_table = cube.is_colliding(self.table.table_top)
        z_ok = abs(float(pos[2]) - self._Z_ON_TABLE) < 0.03  # allow small settling tolerance
        positions_ok = bool(in_workspace and on_table and z_ok)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        semantic_ok = bool(cube.is_face_readable("B", face_dir=self._FACE_DIR))

        # --- Release check ---
        released = not any(
            self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort) ---
        grasped = not released
        if grasped:
            self._metric_stage(0, True)  # grasp
        if semantic_ok:
            self._metric_stage(1, True)  # align
        if semantic_ok and released:
            self._metric_stage(2, True)  # release

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize (must be last metric call before return) ---
        # Distance to nominal goal position (XY primarily meaningful)
        d_xy = float(np.linalg.norm((pos - self._goal_pos)[:2]))

        # Orientation error relative to goal quaternion (radians)
        q_curr = Quaternion(cube.body.get_quaternion())
        q_err = (self._goal_quat.inverse * q_curr).normalised
        w = float(np.clip(abs(q_err.w), -1.0, 1.0))
        ang_err = float(2.0 * np.arccos(w))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_xy_to_goal_m": d_xy,
            },
            pose_error={
                "b_face_forward_angle_err_rad": ang_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})