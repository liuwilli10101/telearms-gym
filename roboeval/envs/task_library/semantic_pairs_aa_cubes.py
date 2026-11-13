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

class AlignedLetterCubesPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Place two A semantic cubes side by side with matching orientation.

Success Criteria:
- Two semantic cubes are side by side on the table, separated by an allowed Y edge gap, X-aligned, and not overlapping.
- Both target A labels are readable in the required face direction and the cubes have the same facing yaw within tolerance.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Back face (-X) is the robot/camera-visible face for semantic readability in this setup
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Goal tolerances (from task specification)
    _MAX_LATERAL_GAP_M = 0.08
    _MAX_ROW_OFFSET_M = 0.03
    _YAW_TOL_DEG = 10.0

    # Cube geometry (SemanticCube is 4cm)
    _CUBE_SIZE_M = 0.04
    _CUBE_REST_Z = 0.97

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly two SemanticCubes; both must be able to show letter 'A'
        self._target_labels = ["A", "A"]
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]

        self.cubes = [
            SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"]),
            SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"]),
        ]
        self.semantic_cube_1, self.semantic_cube_2 = self.cubes

        # Initialize metrics (full required signature)
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
        # Re-initialize metrics (required every reset)
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

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Place objects in RANDOM *unsolved* positions.
        # Guarantee failure of the "side-by-side same row" condition by forcing dx > _MAX_ROW_OFFSET_M.
        margin = 0.08  # comfortably above row tolerance
        x1 = float(rng.uniform(x_min, x_max - (self._MAX_ROW_OFFSET_M + margin)))
        x2 = float(rng.uniform(x1 + (self._MAX_ROW_OFFSET_M + margin), x_max))

        # Sample Y positions and ensure they are not overlapping in the plane.
        # (Overlap check uses collisions, but we also ensure a minimum planar separation here.)
        min_center_sep = self._CUBE_SIZE_M * 1.25  # > cube size to avoid immediate overlap
        for _ in range(100):
            y1 = float(rng.uniform(y_min, y_max))
            y2 = float(rng.uniform(y_min, y_max))
            if np.hypot(x1 - x2, y1 - y2) >= min_center_sep:
                break
        else:
            # Fallback: deterministic separation in Y if repeated samples fail
            y1, y2 = -0.20, 0.20

        positions = [
            np.array([x1, y1, z], dtype=float),
            np.array([x2, y2, z], dtype=float),
        ]

        # Randomize orientations (do NOT align 'A' to the target face direction).
        for cube, pos in zip(self.cubes, positions):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
            cube.set_pose(position=pos, quat=q.elements)

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
        """Return world yaw (rotation about +Z) from quaternion in (w, x, y, z) format."""
        q = Quaternion(quat_wxyz)
        # pyquaternion provides yaw_pitch_roll as (yaw, pitch, roll)
        yaw, _, _ = q.yaw_pitch_roll
        return float(yaw)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _success(self) -> bool:
        # Reset success flag at START (required)
        self._success_check = False

        # Defensive checks (_success can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        c1, c2 = self.cubes
        p1 = c1.body.get_position()
        p2 = c2.body.get_position()

        dx = float(abs(p1[0] - p2[0]))
        dy = float(abs(p1[1] - p2[1]))

        # Side-by-side definition:
        # - roughly same row in X (row offset)
        # - small edge gap in Y (lateral), within tolerance
        edge_gap_y = dy - self._CUBE_SIZE_M
        row_aligned = dx <= self._MAX_ROW_OFFSET_M
        gap_ok = (edge_gap_y >= 0.0) and (edge_gap_y <= self._MAX_LATERAL_GAP_M)

        # Ensure both are on the table (stable placement) and not overlapping
        on_table = c1.is_colliding(self.table.table_top) and c2.is_colliding(self.table.table_top)
        not_overlapping = not c1.is_colliding(c2)

        side_by_side_ok = bool(row_aligned and gap_ok and on_table and not_overlapping)

        # Semantic verification (required): 'A' must be readable on the intended face direction
        semantic_ok = True
        for cube, target in zip(self.cubes, self._target_labels):
            if not cube.is_face_readable(target, face_dir=self._FACE_DIR):
                semantic_ok = False
                break

        # Same facing direction (yaw aligned)
        q1 = np.array(c1.body.get_quaternion(), dtype=float)
        q2 = np.array(c2.body.get_quaternion(), dtype=float)
        yaw1 = self._yaw_from_quat_wxyz(q1)
        yaw2 = self._yaw_from_quat_wxyz(q2)
        yaw_diff = abs(self._wrap_to_pi(yaw1 - yaw2))
        same_facing_ok = yaw_diff <= np.deg2rad(self._YAW_TOL_DEG)

        # Released (no gripper holding either cube)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics
        self._metric_stage(0, bool(side_by_side_ok))
        self._metric_stage(1, bool(c1.is_face_readable("A", face_dir=self._FACE_DIR)))
        self._metric_stage(2, bool(c2.is_face_readable("A", face_dir=self._FACE_DIR)))
        self._metric_stage(3, bool(same_facing_ok))
        self._metric_stage(4, bool(released))

        positions_ok = bool(side_by_side_ok and same_facing_ok)
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "row_offset_x_m": dx,
                "center_separation_y_m": dy,
                "edge_gap_y_m": float(edge_gap_y),
            },
            pose_error={
                "yaw_diff_deg": float(np.rad2deg(yaw_diff)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})