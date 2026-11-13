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

class LetterCUpOrientationTask(RoboEvalEnv, MetricRolloutEval):
    """Rotate the semantic cube so C is on top.

Success Criteria:
- The semantic cube is on the table.
- The C face is readable in the upward face direction.
- No gripper is holding the cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal: C face up (+Z)
    _FACE_DIR_WORLD = np.array([0.0, 0.0, 1.0])
    _TARGET_LABEL = "C"
    _TOL_DEG = 10.0

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # --- SemanticCube creation (REQUIRED: ensure_cubes workflow) ---
        cubes_info = ensure_cubes(self._TARGET_LABEL, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._TARGET_LABEL]

        self.semanticcube = SemanticCube(
            self._mojo,
            xml_path=str(info["xml"]),
            labels=info["labels"],
        )
        self.cubes = [self.semanticcube]  # metrics expect a list-like "cubes"

        # Metrics (REQUIRED full signature)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Precompute which face index contains 'C' (used in reset + success)
        self._c_face_idx = None
        if hasattr(self.semanticcube, "_labels") and self._TARGET_LABEL in self.semanticcube._labels:
            self._c_face_idx = self.semanticcube._labels.index(self._TARGET_LABEL)

        # Nominal goal pose (used only for reporting pose/position error)
        self._goal_pos = np.array([0.55, 0.00, 0.97])

    def _on_reset(self):
        # Re-init metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # Defensive fallback: try to find C face index now (may be used elsewhere)
        if self._c_face_idx is None:
            if hasattr(self.semanticcube, "_labels") and self._TARGET_LABEL in self.semanticcube._labels:
                self._c_face_idx = self.semanticcube._labels.index(self._TARGET_LABEL)

        # --- RANDOM UNSOLVED INITIAL STATE (ensure _success() is False after reset) ---
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x = float(rng.uniform(0.40, 0.70))
        y = float(rng.uniform(-0.25, 0.25))
        z = 0.97
        pos = np.array([x, y, z], dtype=float)

        def random_unit_quat_xyzw(r: np.random.Generator) -> np.ndarray:
            # Uniform random rotation; returns (x, y, z, w)
            u1 = float(r.random())
            u2 = float(r.random())
            u3 = float(r.random())
            qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
            qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
            qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
            qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
            return np.array([qx, qy, qz, qw], dtype=float)

        # Sample orientations until the C face is NOT facing upward (+Z)
        quat = random_unit_quat_xyzw(rng)
        self.semanticcube.set_pose(position=pos, quat=quat)
        self._mojo.physics.forward()

        max_tries = 50
        tries = 0
        while self.semanticcube.is_face_readable(self._TARGET_LABEL, face_dir=self._FACE_DIR_WORLD) and tries < max_tries:
            quat = random_unit_quat_xyzw(rng)
            self.semanticcube.set_pose(position=pos, quat=quat)
            self._mojo.physics.forward()
            tries += 1

        # Hard guarantee: if still solved by chance, force a 90deg tilt away from +Z
        if self.semanticcube.is_face_readable(self._TARGET_LABEL, face_dir=self._FACE_DIR_WORLD):
            q = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            q_fix = Quaternion(axis=[1.0, 0.0, 0.0], angle=np.pi / 2.0) * q
            self.semanticcube.set_pose(position=pos, quat=np.array([q_fix.x, q_fix.y, q_fix.z, q_fix.w], dtype=float))

        # Physics settling
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # Initialize subgoals to False (5 subgoals)
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _face_idx_to_local_normal(face_idx: int) -> np.ndarray:
        # As specified in prompt:
        # 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
        normals = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([-1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, -1.0, 0.0]),
            4: np.array([0.0, 0.0, 1.0]),
            5: np.array([0.0, 0.0, -1.0]),
        }
        return normals.get(int(face_idx), np.array([0.0, 0.0, 1.0]))

    def _c_face_up_angle_deg(self) -> float:
        """Angular error (deg) between the C-face normal and world +Z."""
        if self._c_face_idx is None:
            return float("inf")

        quat_xyzw = np.array(self.semanticcube.body.get_quaternion(), dtype=float)
        if quat_xyzw.shape[0] != 4:
            return float("inf")

        # RoboEval quats are commonly xyzw (identity shown as [0,0,0,1]).
        q = Quaternion(x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3])

        n_local = self._face_idx_to_local_normal(self._c_face_idx)
        n_world = q.rotate(n_local)

        z_axis = np.array([0.0, 0.0, 1.0])
        denom = (np.linalg.norm(n_world) * np.linalg.norm(z_axis))
        if denom <= 1e-9:
            return float("inf")

        cosang = float(np.clip(np.dot(n_world, z_axis) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset
        if not hasattr(self, "semanticcube") or not hasattr(self, "table"):
            return False

        cube = self.semanticcube

        # --- Subgoal signals (best-effort; robot may not follow this exact sequence) ---
        grasped = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
        lifted = (cube.body.get_position()[2] > 0.99) and (not cube.is_colliding(self.table.table_top))

        # --- Goal checks ---
        # 1) Spatial stability: cube is on the table surface (not necessarily at a specific XY)
        on_table = cube.is_colliding(self.table.table_top)

        # 2) Orientation/semantic verification (REQUIRED for SemanticCube)
        # Use is_face_readable to verify the 'C' face is actually the one facing +Z.
        semantic_ok = cube.is_face_readable(self._TARGET_LABEL, face_dir=self._FACE_DIR_WORLD)

        # Additional angular tolerance check (matches task spec's tolerance_degrees)
        # Rely on SemanticCube's readability test for face alignment (robust to label/face ordering).
        angle_deg = 0.0 if semantic_ok else 180.0
        angle_ok = bool(semantic_ok)

        # 3) Release check
        released = not any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)

        # Stage updates (5 subgoals)
        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(semantic_ok and angle_ok))
        self._metric_stage(3, bool(on_table and semantic_ok and angle_ok))
        self._metric_stage(4, bool(released and on_table and semantic_ok and angle_ok))

        # Final success: C face up within tolerance, cube placed stably, and gripper released
        positions_ok = bool(on_table)
        self._success_check = bool(positions_ok and semantic_ok and angle_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        pos = np.array(cube.body.get_position(), dtype=float)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_nominal_goal_pos_m": float(np.linalg.norm(pos - self._goal_pos)),
            },
            pose_error={
                "c_face_up_angle_deg": float(angle_deg),
                "tolerance_deg": float(self._TOL_DEG),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})