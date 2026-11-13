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

class RotateAToTopTask(RoboEvalEnv, MetricRolloutEval):
    """Rotate the semantic cube so A is on top.

Success Criteria:
- The semantic cube is inside the workspace, on the table, and near the expected rest height.
- The A face is readable in the upward face direction.
- No gripper is holding the cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For this task, "top face" means +Z in the world frame.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Workspace bounds (per spec)
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # --- SemanticCube creation (MUST use ensure_cubes workflow; ONLY asset allowed) ---
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]
        self.semantic_cube_1 = SemanticCube(
            self._mojo,
            xml_path=str(info["xml"]),
            labels=info["labels"],
        )
        self.cubes = [self.semantic_cube_1]
        self._target_labels = ["A"]

        # Goal reference pose (used for metrics; success does not require exact XY)
        self._goal_pos = np.array([0.55, 0.00, self._CUBE_REST_Z])

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

        # 2. Deterministic RNG (validation may randomize later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM (UNSOLVED) state within required bounds
        cube = self.semantic_cube_1

        # Required tighter workspace for initial placement
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = self._CUBE_REST_Z

        def _random_unit_quat() -> np.ndarray:
            # Uniform random unit quaternion (Shoemake)
            u1, u2, u3 = rng.random(3)
            q = np.array(
                [
                    np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
                    np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
                    np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
                    np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
                ],
                dtype=float,
            )
            # pyquaternion Quaternion expects [w, x, y, z] in constructor,
            # but cube.set_pose expects [x, y, z, w] elements (as used elsewhere via q.elements).
            # We'll construct with (w, x, y, z) then pass q.elements ([x,y,z,w]).
            qq = Quaternion(q[3], q[0], q[1], q[2])
            return qq.elements

        placed = False
        for _ in range(200):
            pos = np.array(
                [
                    float(rng.uniform(x_min, x_max)),
                    float(rng.uniform(y_min, y_max)),
                    float(z0),
                ],
                dtype=float,
            )

            quat_xyzw = _random_unit_quat()
            cube.set_pose(position=pos, quat=quat_xyzw)

            # Let physics settle minimally and verify it's unsolved and on table
            self._mojo.physics.forward()

            # Ensure not already solved: 'A' must NOT be readable on +Z after reset
            if cube.is_face_readable("A", face_dir=self._FACE_DIR):
                continue

            # Also try to keep it on the table at reset (so failure is due to orientation)
            if not cube.is_colliding(self.table.table_top):
                continue

            placed = True
            break

        # Fallback: if we somehow couldn't satisfy checks, at least force unsolved orientation
        if not placed:
            pos = np.array(
                [float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), float(z0)],
                dtype=float,
            )
            face_idx = cube._labels.index("A")
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            q_unsolved = (Quaternion(q_goal) * Quaternion(axis=[1.0, 0.0, 0.0], angle=np.pi / 2.0)).elements
            cube.set_pose(position=pos, quat=q_unsolved)
            self._mojo.physics.forward()

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        cube = self.semantic_cube_1
        pos = cube.body.get_position()

        # --- Spatial / state checks (minimal for this orientation-only task) ---
        in_workspace = (
            (self._X_BOUNDS[0] <= float(pos[0]) <= self._X_BOUNDS[1])
            and (self._Y_BOUNDS[0] <= float(pos[1]) <= self._Y_BOUNDS[1])
        )
        on_table = cube.is_colliding(self.table.table_top)
        near_table_height = abs(float(pos[2]) - self._CUBE_REST_Z) <= 0.05
        positions_ok = bool(in_workspace and on_table and near_table_height)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        semantic_ok = cube.is_face_readable("A", face_dir=self._FACE_DIR)

        # --- Release check ---
        released = not any(
            self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, works for single- or multi-gripper robots) ---
        held = any(self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers)
        lifted = float(pos[2]) > (self._CUBE_REST_Z + 0.02)
        placed = bool(on_table and near_table_height)

        # 0. Optional assist grasp (right) -> treat as "held" for generic grippers
        self._metric_stage(0, bool(held))
        # 1. Stable grasp (left) -> also "held" (robot may only have one gripper)
        self._metric_stage(1, bool(held))
        # 2. Lift
        self._metric_stage(2, bool(lifted))
        # 3. Align (A on top)
        self._metric_stage(3, bool(semantic_ok))
        # 4. Place (on table while aligned)
        self._metric_stage(4, bool(placed and semantic_ok))
        # 5. Release (released, on table, aligned)
        self._metric_stage(5, bool(released and placed and semantic_ok))

        # --- Final success condition ---
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Finalize metrics (MUST be last metric call before return) ---
        xy_dist = float(np.linalg.norm(np.array([pos[0], pos[1]]) - self._goal_pos[:2]))
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_xy_to_center": xy_dist,
            },
            pose_error={
                # Orientation success is semantic (A readable on +Z), yaw-invariant.
                # Use a simple binary proxy for pose error.
                "A_on_top_error": 0.0 if semantic_ok else 1.0,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})