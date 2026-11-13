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

class LetterFaceAlignmentTask(RoboEvalEnv, MetricRolloutEval):
    """
    Pick up a semantic cube and rotate it so that a specified letter face points
    directly toward the robot (world -X / front). The cube should end oriented
    with that target letter facing the robot/camera direction.

    Sequential Sub-Goals:
        1. Grasp the semantic cube.
        2. Lift the cube slightly to allow free rotation.
        3. Align the target letter face to point toward world -X (toward robot).
        4. Place the cube back on the table while maintaining orientation.
        5. Release the cube without disturbing the final orientation.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Axis convention: front/toward robot is world -X
    _FACE_DIR_WORLD = np.array([-1.0, 0.0, 0.0])

    # Task parameters
    _TARGET_LETTER = "A"
    _ANG_TOL_DEG = 15.0

    # Goal workspace bounds (tighter than global workspace)
    _X_MIN, _X_MAX = 0.40, 0.70
    _Y_MIN, _Y_MAX = -0.25, 0.25

    # SemanticCube table resting height (base Z)
    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create exactly ONE SemanticCube using ensure_cubes() workflow (REQUIRED) ---
        cubes_info = ensure_cubes(self._TARGET_LETTER, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._TARGET_LETTER]

        # Mandatory instantiation pattern (supports either key name from factory)
        xml_key = "xml" if "xml" in info else "xml_path"
        self.semantic_cube_1 = SemanticCube(
            self._mojo,
            xml_path=str(info[xml_key]),
            labels=info["labels"],
        )

        self.cubes = [self.semantic_cube_1]
        self._target_labels = [self._TARGET_LETTER]

        # Precompute a canonical goal pose (used for pose error + reset goal state)
        self._goal_pos = np.array([0.55, 0.0, self._CUBE_REST_Z])

        face_idx = self.semantic_cube_1._labels.index(self._TARGET_LETTER)
        self._goal_quat = self.semantic_cube_1.get_orientation_for_face_and_glyph(
            face_idx=face_idx,
            face_dir_world=self._FACE_DIR_WORLD,
        )

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

    def _on_reset(self):
        # Re-initialize metrics on every reset (REQUIRED)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG handle (required pattern)
        _rng = np.random.default_rng(self.seed)

        # --- Place objects in RANDOM, UNSOLVED poses within workspace bounds ---
        # Random XY in bounds; Z at table rest height
        x = float(_rng.uniform(self._X_MIN, self._X_MAX))
        y = float(_rng.uniform(self._Y_MIN, self._Y_MAX))
        pos = np.array([x, y, self._CUBE_REST_Z], dtype=float)

        # Random orientation, with rejection to ensure NOT in goal orientation tolerance
        def _random_unit_quat_elements(rng: np.random.Generator) -> np.ndarray:
            # Shoemake method (uniform random unit quaternion)
            u1, u2, u3 = rng.random(3)
            qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
            qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
            qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
            qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
            # pyquaternion uses (w, x, y, z) ordering in .elements
            return np.array([qw, qx, qy, qz], dtype=float)

        quat_elements = None
        for _ in range(200):
            cand = _random_unit_quat_elements(_rng)
            try:
                q_cand = Quaternion(cand)
                ang_err_deg = float(np.degrees((self._goal_quat.inverse * q_cand).angle))
            except Exception:
                ang_err_deg = 180.0

            # Ensure outside tolerance so _success() is False after reset
            if ang_err_deg > (self._ANG_TOL_DEG + 5.0):
                quat_elements = cand
                break

        if quat_elements is None:
            # Fallback: guaranteed 180-deg rotation about a fixed axis (not within 15 deg)
            q_fallback = Quaternion(axis=[0.0, 0.0, 1.0], angle=np.pi) * self._goal_quat
            quat_elements = q_fallback.elements

        self.semantic_cube_1.set_pose(
            position=pos,
            quat=quat_elements,
        )

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START (REQUIRED)
        self._success_check = False

        # Defensive checks (_success can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        cube = self.semantic_cube_1

        # --- Spatial: keep cube within workspace bounds & on the table ---
        pos = cube.body.get_position()
        within_bounds = (
            (self._X_MIN <= float(pos[0]) <= self._X_MAX) and
            (self._Y_MIN <= float(pos[1]) <= self._Y_MAX)
        )
        on_table = cube.is_colliding(self.table.table_top)
        positions_ok = bool(within_bounds and on_table)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        semantic_ok = bool(cube.is_face_readable(self._TARGET_LETTER, face_dir=self._FACE_DIR_WORLD))

        # --- Orientation tolerance (additional explicit angular tolerance check) ---
        # Note: Quaternion ordering is assumed consistent with set_pose(q.elements) usage.
        try:
            q_cur = Quaternion(cube.body.get_quaternion())
            ang_err_deg = float(np.degrees((self._goal_quat.inverse * q_cur).angle))
        except Exception:
            # If quaternion parsing fails for any reason, fall back to semantic readability only.
            ang_err_deg = 180.0
        orientation_ok = bool(ang_err_deg <= (self._ANG_TOL_DEG + 1e-6))

        # --- Released gripper check ---
        holding = any(self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers)
        released = not holding

        # --- Subgoal staging (best-effort inference from contacts/holding) ---
        # 1) grasp
        self._metric_stage(0, bool(holding))
        # 2) lift (slightly above rest height and not on table)
        self._metric_stage(1, bool(holding and (float(pos[2]) > self._CUBE_REST_Z + 0.01) and (not on_table)))
        # 3) align (while holding)
        self._metric_stage(2, bool(holding and semantic_ok and orientation_ok))
        # 4) place (aligned and on table)
        self._metric_stage(3, bool(semantic_ok and orientation_ok and on_table))
        # 5) release (final)
        self._metric_stage(4, bool(semantic_ok and orientation_ok and positions_ok and released))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and orientation_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        center_xy = np.array([0.55, 0.0], dtype=float)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_xy_to_center": float(np.linalg.norm(np.array(pos[:2], dtype=float) - center_xy)),
            },
            pose_error={
                "angular_deg_to_goal": float(ang_err_deg),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})