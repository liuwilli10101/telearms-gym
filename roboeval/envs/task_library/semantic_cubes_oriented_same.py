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

class AlignedLetterCubeRow(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic cubes into a single straight row running front-to-back along
    the table's X axis (the cubes line up one behind another, NOT side by side), then
    rotate every cube so its 'A' face points straight up on the top face.

    Success requires: the three cubes are ordered front-to-back along X with an even gap
    between neighbors (each ~6-12 cm apart in X) and almost no left-right (Y) drift across
    the row; every cube rests on the table and shows 'A' on its top face; and no cube is
    held. The cubes are interchangeable — only the front-to-back ordering by position
    matters, not which cube is which.

    Sequential Sub-Goals:
        1. Place one cube as the front cube of the row (smallest X).
        2. Place a second cube behind it (+X) with reasonable spacing.
        3. Place the third cube behind the second (+X) with reasonable spacing.
        4. Align all three into a straight front-to-back line (same Y, even X spacing).
        5. Rotate every cube so its 'A' face is on the top face.
        6. Release all cubes (no gripper holding).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For a flat tabletop arrangement, the top face (+Z) is the intended readable face.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Goal condition parameters (from task spec)
    _MIN_SPACING_X = 0.06
    _MAX_SPACING_X = 0.12
    _MAX_LATERAL_DEVIATION_Y = 0.02

    # SemanticCube resting Z (must be on table surface without clipping)
    _TABLE_CUBE_Z = 0.97

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly three SemanticCubes (ONLY allowed asset).
        # Use ensure_cubes() workflow (required) and instantiate via required pattern.
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]
        xml_path = info.get("xml", info.get("xml_path"))
        labels = info["labels"]

        self.cubes = [
            SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels),
            SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels),
            SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels),
        ]

        # We will orient all cubes to show this letter on the top face in the goal state.
        self._goal_letter = "A"

        # Goal row positions: aligned along X axis, constant Y (straight row), within workspace.
        # Spacing is 0.06m (within [0.06, 0.12]).
        self._goal_positions = [
            np.array([0.49, 0.00, self._TABLE_CUBE_Z]),
            np.array([0.55, 0.00, self._TABLE_CUBE_Z]),
            np.array([0.61, 0.00, self._TABLE_CUBE_Z]),
        ]

        # Metrics (must be initialized with full signature)
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

        # Deterministic RNG (required pattern).
        rng = np.random.default_rng(self.seed)

        # --- Randomize cubes into UNSOLVED poses within workspace bounds ---
        # Workspace bounds required by spec
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._TABLE_CUBE_Z

        # Helper: sample positions separated enough to avoid immediate row success by chance
        min_pairwise_dist = 0.09  # ensure not trivially within spacing bounds/row alignment
        positions = []
        for _ in range(200):
            candidate = np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float
            )
            if all(np.linalg.norm(candidate[:2] - p[:2]) >= min_pairwise_dist for p in positions):
                positions.append(candidate)
            if len(positions) == 3:
                break
        if len(positions) < 3:
            # Fallback (still within bounds) with fixed spread
            positions = [
                np.array([0.42, -0.20, z], dtype=float),
                np.array([0.60, 0.18, z], dtype=float),
                np.array([0.68, -0.05, z], dtype=float),
            ]

        # Random orientations: choose random top-face labels independently
        # and apply random yaw so readability alignment is very unlikely.
        for cube, pos in zip(self.cubes, positions):
            face_idx = int(rng.integers(0, len(cube._labels)))
            q_face = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            q_yaw = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            q = q_yaw * q_face
            cube.set_pose(position=pos, quat=q.elements)

        # Ensure reset is not accidentally solved (extremely unlikely, but enforced)
        # If solved, perturb cube3 slightly within bounds.
        self._mojo.physics.forward()
        self._mojo.physics.forward()
        if self._success():
            p = self.cubes[2].body.get_position().copy()
            p[0] = float(np.clip(p[0] + 0.03, x_min, x_max))
            p[1] = float(np.clip(p[1] + 0.03, y_min, y_max))
            # Change orientation too
            face_idx = int(rng.integers(0, len(self.cubes[2]._labels)))
            q_face = self.cubes[2].get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            q_yaw = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            q = q_yaw * q_face
            self.cubes[2].set_pose(position=p, quat=q.elements)
            self._mojo.physics.forward()
            self._mojo.physics.forward()

        # Initialize subgoals to False
        self._num_subgoals = 6
        for idx in range(self._num_subgoals):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _quat_angle_error(q_current: np.ndarray, q_target: np.ndarray) -> float:
        """Angular distance between two quaternions (radians)."""
        qc = Quaternion(q_current)
        qt = Quaternion(q_target)
        dq = qt.inverse * qc
        return float(abs(dq.angle))

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if len(self.cubes) != 3:
            return False

        # Identity-free: sort cubes by X ascending so slot[0] is leftmost.
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][0])
        c1, c2, c3 = [cp[0] for cp in cubes_sorted]
        p1, p2, p3 = [cp[1] for cp in cubes_sorted]

        # --- Spatial / arrangement checks ---
        # Right-of relations along +X with spacing bounds.
        dx21 = float(p2[0] - p1[0])
        dx32 = float(p3[0] - p2[0])

        eps = 1e-3
        right_of_2 = (dx21 >= (self._MIN_SPACING_X - eps)) and (dx21 <= (self._MAX_SPACING_X + eps))
        right_of_3 = (dx32 >= (self._MIN_SPACING_X - eps)) and (dx32 <= (self._MAX_SPACING_X + eps))

        # Straight row aligned along X: small Y deviation across all cubes.
        y_mean = float((p1[1] + p2[1] + p3[1]) / 3.0)
        aligned_row = (
            abs(float(p1[1] - y_mean)) <= self._MAX_LATERAL_DEVIATION_Y
            and abs(float(p2[1] - y_mean)) <= self._MAX_LATERAL_DEVIATION_Y
            and abs(float(p3[1] - y_mean)) <= self._MAX_LATERAL_DEVIATION_Y
        )

        # Optional stability: ensure cubes are on the table surface.
        on_table = (
            c1.is_colliding(self.table.table_top)
            and c2.is_colliding(self.table.table_top)
            and c3.is_colliding(self.table.table_top)
        )

        positions_ok = bool(right_of_2 and right_of_3 and aligned_row and on_table)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        # Determine the top label of cube 1, then require cubes 2 and 3 to show the same on TOP.
        try:
            top_label_1 = c1.get_label(side="top")[0]
        except Exception:
            top_label_1 = None

        semantic_ok = False
        if isinstance(top_label_1, str) and len(top_label_1) == 1:
            semantic_ok = all(
                cube.is_face_readable(top_label_1, face_dir=self._FACE_DIR)
                for cube in self.cubes
            )

        # --- Released check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Metric stages (subgoals) ---
        # 0: cube1 leftmost by X ordering
        leftmost_ok = bool(p1[0] < p2[0] < p3[0])
        self._metric_stage(0, leftmost_ok)

        # 1: cube2 right-of cube1 spacing ok
        self._metric_stage(1, bool(right_of_2))

        # 2: cube3 right-of cube2 spacing ok
        self._metric_stage(2, bool(right_of_3))

        # 3: row aligned
        self._metric_stage(3, bool(aligned_row))

        # 4: semantic top-face consistency
        self._metric_stage(4, bool(semantic_ok))

        # 5: released
        self._metric_stage(5, bool(released))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize ---
        # Distances to goal positions
        target_distance = {
            "semantic_cube_1_to_goal": float(np.linalg.norm(p1 - self._goal_positions[0])),
            "semantic_cube_2_to_goal": float(np.linalg.norm(p2 - self._goal_positions[1])),
            "semantic_cube_3_to_goal": float(np.linalg.norm(p3 - self._goal_positions[2])),
        }

        # Pose error: angular error to the orientation that would put cube1's top label on TOP (+Z)
        pose_error = {}
        if isinstance(top_label_1, str) and len(top_label_1) == 1 and top_label_1 in c1._labels:
            for i, cube in enumerate(self.cubes, start=1):
                face_idx = cube._labels.index(top_label_1)
                q_goal = cube.get_orientation_for_face_and_glyph(
                    face_idx=face_idx,
                    face_dir_world=self._FACE_DIR,
                ).elements
                q_cur = cube.body.get_quaternion()
                pose_error[f"semantic_cube_{i}_angle_err_rad"] = self._quat_angle_error(q_cur, q_goal)
        else:
            pose_error = {
                "semantic_cube_1_angle_err_rad": float("nan"),
                "semantic_cube_2_angle_err_rad": float("nan"),
                "semantic_cube_3_angle_err_rad": float("nan"),
            }

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})