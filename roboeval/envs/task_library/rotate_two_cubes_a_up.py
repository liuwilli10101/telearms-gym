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


class SideBySideUprightCubes(RoboEvalEnv, MetricRolloutEval):
    """Place two semantic cubes side by side with A facing up on both.

Success Criteria:
- The two semantic cubes are side by side along Y with small edge gap, aligned in X, and on the table at expected height.
- Both cubes have A readable on the upward face direction.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For a flat, on-table arrangement we want the TOP face readable.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Semantic goal
    _TARGET_LABEL = "A"

    # Geometry constants
    _CUBE_SIZE_M = 0.04
    _CUBE_REST_Z = 0.97

    # Goal tolerances from task spec
    _MAX_EDGE_GAP_M = 0.03           # max gap between cube edges
    _MAX_X_OFFSET_M = 0.03           # "roughly aligned laterally" (use X alignment)
    _Z_TOL_M = 0.02                  # keep on-table height reasonably close

    # Nominal goal placement
    _GOAL_X = 0.55
    _GOAL_Y1 = +0.03
    _GOAL_Y2 = -0.03  # center-to-center dy = 0.06 => edge gap = 0.02

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create two SemanticCubes. Both must support having an 'A' face.
        cubes_info = ensure_cubes(self._TARGET_LABEL, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._TARGET_LABEL]

        # ⚠️ REQUIRED instantiation pattern
        self.cube_1 = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
        self.cube_2 = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
        self.cubes = [self.cube_1, self.cube_2]
        self._target_labels = [self._TARGET_LABEL, self._TARGET_LABEL]

        # Cache goal poses
        self._goal_positions = [
            np.array([self._GOAL_X, self._GOAL_Y1, self._CUBE_REST_Z], dtype=float),
            np.array([self._GOAL_X, self._GOAL_Y2, self._CUBE_REST_Z], dtype=float),
        ]

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

        # 2. Deterministic RNG handle
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM unsolved positions (within workspace bounds)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Ensure not already solved by making them either too far apart in Y or misaligned in X.
        # Also randomize orientation so the 'A' face is (very likely) not readable upward.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_REST_Z)

        def _rand_quat():
            q = rng.normal(size=4)
            q /= np.linalg.norm(q) + 1e-12
            return q

        # Choose a configuration that violates side-by-side constraints robustly:
        # enforce large Y separation (dy > cube_size + max_edge_gap) or large X offset (dx > max_x_offset).
        # We'll do large Y separation with random Xs as well.
        attempts = 0
        while True:
            attempts += 1

            # Random X, Y for cube 1
            x1 = float(rng.uniform(x_min, x_max))
            y1 = float(rng.uniform(y_min, y_max))

            # Random X for cube 2, and Y far enough from cube 1
            x2 = float(rng.uniform(x_min, x_max))

            # Enforce y-separation well beyond "side-by-side" allowable range
            # side-by-side requires: 0 <= (dy - cube_size) <= max_edge_gap  => dy <= cube_size + max_edge_gap
            min_dy = float(self._CUBE_SIZE_M + self._MAX_EDGE_GAP_M + 0.05)  # margin
            # Sample y2 from the two disjoint intervals
            lower_hi = y1 - min_dy
            upper_lo = y1 + min_dy
            intervals = []
            if lower_hi >= y_min:
                intervals.append((y_min, min(lower_hi, y_max)))
            if upper_lo <= y_max:
                intervals.append((max(upper_lo, y_min), y_max))

            if not intervals:
                # Fallback: if y cannot be separated due to bounds, force large x-offset
                y2 = float(rng.uniform(y_min, y_max))
                # force x-offset
                x2 = float(np.clip(x1 + np.sign(rng.uniform(-1, 1)) * (self._MAX_X_OFFSET_M + 0.08), x_min, x_max))
            else:
                a, b = intervals[rng.integers(0, len(intervals))]
                y2 = float(rng.uniform(a, b))

            p1 = np.array([x1, y1, z], dtype=float)
            p2 = np.array([x2, y2, z], dtype=float)

            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            edge_gap = dy - self._CUBE_SIZE_M
            side_by_side_ok = (dx <= self._MAX_X_OFFSET_M) and (0.0 <= edge_gap <= self._MAX_EDGE_GAP_M)

            # Also avoid exact goal positions (rare but possible)
            near_goal = (
                np.linalg.norm(p1 - self._goal_positions[0]) < 1e-6
                and np.linalg.norm(p2 - self._goal_positions[1]) < 1e-6
            )

            if (not side_by_side_ok) and (not near_goal):
                break

            if attempts > 200:
                # Deterministic fallback: break side-by-side by setting far-apart Y explicitly
                p1 = np.array([0.45, -0.20, z], dtype=float)
                p2 = np.array([0.65, 0.20, z], dtype=float)
                break

        # Set random orientations (do NOT align 'A' upward)
        q1 = _rand_quat()
        q2 = _rand_quat()
        self.cube_1.set_pose(position=p1, quat=q1)
        self.cube_2.set_pose(position=p2, quat=q2)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
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

        p1 = self.cube_1.body.get_position()
        p2 = self.cube_2.body.get_position()

        # --- Spatial arrangement: "side-by-side" ---
        dx = float(abs(p1[0] - p2[0]))
        dy = float(abs(p1[1] - p2[1]))

        # Treat "side-by-side" as separated along Y with small edge gap, and aligned in X.
        edge_gap = dy - self._CUBE_SIZE_M  # gap between cube faces (approx)
        side_by_side_ok = (dx <= self._MAX_X_OFFSET_M) and (0.0 <= edge_gap <= self._MAX_EDGE_GAP_M)

        # On-table sanity (avoid floating / falling)
        on_table_ok = True
        for cube in self.cubes:
            pos = cube.body.get_position()
            if abs(float(pos[2]) - self._CUBE_REST_Z) > self._Z_TOL_M:
                on_table_ok = False
                break
            # Prefer reliable table collision if available
            if hasattr(self.table, "table_top") and not cube.is_colliding(self.table.table_top):
                on_table_ok = False
                break

        positions_ok = bool(side_by_side_ok and on_table_ok)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        semantic_ok = True
        for cube, target_label in zip(self.cubes, self._target_labels):
            # NOTE: is_face_readable uses face_dir (no _world)
            if not cube.is_face_readable(target_label, face_dir=self._FACE_DIR):
                semantic_ok = False
                break

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoals (coarse, since this is an end-state checker)
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics summary
        d1 = float(np.linalg.norm(self.cube_1.body.get_position() - self._goal_positions[0]))
        d2 = float(np.linalg.norm(self.cube_2.body.get_position() - self._goal_positions[1]))
        # Provide interpretable scalars
        target_distance = {
            "cube_1_to_goal_m": d1,
            "cube_2_to_goal_m": d2,
            "x_alignment_m": dx,
            "edge_gap_m": edge_gap,
        }
        pose_error = {
            "cube_1_A_face_up_ok": float(semantic_ok and self.cube_1.is_face_readable(self._TARGET_LABEL, face_dir=self._FACE_DIR)),
            "cube_2_A_face_up_ok": float(semantic_ok and self.cube_2.is_face_readable(self._TARGET_LABEL, face_dir=self._FACE_DIR)),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})