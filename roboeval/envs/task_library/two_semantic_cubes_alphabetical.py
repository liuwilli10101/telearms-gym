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


class AlphabeticalCubeOrdering(RoboEvalEnv, MetricRolloutEval):
    """Arrange A on the left (larger Y) and B on the right (smaller Y).

Success Criteria:
Keep the cubes X-aligned, separated, on the table, and released, with A/B readable in their slots."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Flat, side-by-side arrangement: verify letters on the TOP face (+Z).
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Goal tolerances (from task spec)
    _MIN_LATERAL_SEP_M = 0.03
    _MAX_DEPTH_OFFSET_M = 0.03

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create EXACTLY the required assets: SemanticCube only
        self._target_labels = ["A", "B"]
        cubes_info = ensure_cubes("".join(self._target_labels), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        info_a = cubes_info["A"]
        info_b = cubes_info["B"]

        # Mandatory instantiation pattern
        self.cube_a = SemanticCube(self._mojo, xml_path=str(info_a["xml"]), labels=info_a["labels"])
        self.cube_b = SemanticCube(self._mojo, xml_path=str(info_b["xml"]), labels=info_b["labels"])
        self.cubes = [self.cube_a, self.cube_b]

        # Goal poses (A is leftmost in camera view => higher +Y; B rightmost => lower -Y)
        self._goal_pos_a = np.array([0.55, 0.10, 0.97])
        self._goal_pos_b = np.array([0.55, -0.10, 0.97])

        # Metrics (full required signature)
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
        # Re-initialize metrics (required on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        # Sample RANDOM, UNSOLVED placements:
        # Guarantee failure of the "A left of B" check by enforcing y_A <= y_B
        # (A starts on the right, so the operator must swap them).
        # Also randomize top face so semantic check is very unlikely to pass.
        for _ in range(100):
            x_a = float(rng.uniform(x_min, x_max))
            x_b = float(rng.uniform(x_min, x_max))

            y_a = float(rng.uniform(y_min, y_max - 0.06))
            y_b = float(y_a + rng.uniform(0.06, 0.18))  # keep separated to avoid collision
            if y_b <= y_max:
                pos_a = np.array([x_a, y_a, z_base], dtype=float)
                pos_b = np.array([x_b, y_b, z_base], dtype=float)
                break
        else:
            # Fallback (still unsolved): A not left of B
            pos_a = np.array([0.55, -0.10, z_base], dtype=float)
            pos_b = np.array([0.55, 0.10, z_base], dtype=float)

        for cube, target_label, pos in [
            (self.cube_a, "A", pos_a),
            (self.cube_b, "B", pos_b),
        ]:
            # Put a NON-target label face on top to avoid semantic success.
            target_idx = cube._labels.index(target_label)
            other_indices = [i for i in range(len(cube._labels)) if i != target_idx]
            face_idx = int(rng.choice(other_indices))
            q = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR)
            cube.set_pose(position=pos, quat=q.elements)

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        self._num_subgoals = 6
        for idx in range(self._num_subgoals):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # Reset success flag at start
        self._success_check = False

        # Defensive checks (_success can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: whichever cube sits in the left slot must read "A".
        # Left is +Y here (matching _goal_pos_a at y=+0.10), so sort by DESCENDING Y
        # and map slot[0] to "A". Sorting ascending assigned "A" to the right-hand
        # slot, which made the accepted arrangement read "B A" left-to-right.
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_A_cube, pos_a = cubes_sorted[0]  # larger Y  => left
        slot_B_cube, pos_b = cubes_sorted[1]  # smaller Y => right

        # --- Spatial checks (goal conditions) ---
        # "A left of B" in camera view => along Y axis: left is +Y, right is -Y.
        # The sort already orders them; this enforces the separation itself.
        lateral_sep = float(pos_a[1] - pos_b[1])
        left_of_ok = lateral_sep >= self._MIN_LATERAL_SEP_M

        # "Aligned in depth" => similar X (depth) coordinate
        depth_offset = float(abs(pos_a[0] - pos_b[0]))
        aligned_ok = depth_offset <= self._MAX_DEPTH_OFFSET_M

        # On table (not strictly required by spec, but helps avoid mid-air success)
        on_table_ok = slot_A_cube.is_colliding(self.table.table_top) and slot_B_cube.is_colliding(
            self.table.table_top
        )

        # Avoid degenerate overlaps/collisions
        not_colliding_each_other = not slot_A_cube.is_colliding(slot_B_cube)

        positions_ok = bool(left_of_ok and aligned_ok and on_table_ok and not_colliding_each_other)

        # --- Semantic verification: cube currently in each slot must show that slot's target letter ---
        semantic_ok = all(
            cube.is_face_readable(target_label, face_dir=self._FACE_DIR)
            for cube, target_label in [(slot_A_cube, "A"), (slot_B_cube, "B")]
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Mark stages if fully successful (simple mapping)
        if positions_ok and semantic_ok and released:
            for i in range(self._num_subgoals):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (must be last metric call before return)
        # Provide distances/errors relevant to this task
        target_distance = {
            "lateral_sep_y": max(0.0, self._MIN_LATERAL_SEP_M - lateral_sep),
            "depth_offset_x": max(0.0, depth_offset - self._MAX_DEPTH_OFFSET_M),
        }
        pose_error = {
            "cube_a_to_goal_m": float(np.linalg.norm(pos_a - self._goal_pos_a)),
            "cube_b_to_goal_m": float(np.linalg.norm(pos_b - self._goal_pos_b)),
            "cube_a_b_distance_m": float(distance(slot_A_cube.body, slot_B_cube.body)),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})