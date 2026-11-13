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

class XyzCubeRowAlignment(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic letter cubes into a single horizontal row along the Y axis
    with X, Y, Z ordered from left to right (+Y to -Y). All letters must face the robot
    (letter visible on the face oriented toward -X).

    Sequential Sub-Goals:
        1. Securely grasp cube X with either gripper.
        2. Rotate cube X so its letter face points toward -X (toward the robot).
        3. Place cube X on the left side of the row (+Y direction).
        4. Securely grasp cube Y with either gripper.
        5. Rotate cube Y so its letter face points toward -X (toward the robot).
        6. Place cube Y in the middle of the row (between X and Z along Y).
        7. Securely grasp cube Z with either gripper.
        8. Rotate cube Z so its letter face points toward -X (toward the robot).
        9. Place cube Z on the right side of the row (-Y direction).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Letters must face the robot: toward -X (camera-visible back face).
    _FACE_DIR_WORLD = np.array([-1.0, 0.0, 0.0])

    # Goal layout: a single row along Y, aligned in X and Z.
    _GOAL_X = 0.55
    _GOAL_Z = 0.97
    _GOAL_YS = [0.10, 0.00, -0.10]  # X (left), Y (middle), Z (right)

    # Success thresholds from task spec / constraints
    _MIN_Y_SEP = 0.06
    _MAX_X_SPREAD = 0.03
    _MAX_Z_SPREAD = 0.02

    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # Create SemanticCubes via ensure_cubes workflow (MANDATORY)
        self._target_labels = ["X", "Y", "Z"]
        cubes_info = ensure_cubes("".join(self._target_labels), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.cubes = []
        for ch in self._target_labels:
            info = cubes_info[ch]
            cube = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
            self.cubes.append(cube)

        # Convenience references (do not imply any ordering logic beyond labels)
        self.cube_X, self.cube_Y, self.cube_Z = self.cubes

        # Precompute goal positions (used in metrics and success)
        self._goal_positions = [
            np.array([self._GOAL_X, self._GOAL_YS[0], self._GOAL_Z]),
            np.array([self._GOAL_X, self._GOAL_YS[1], self._GOAL_Z]),
            np.array([self._GOAL_X, self._GOAL_YS[2], self._GOAL_Z]),
        ]
        self._goal_quats_wxyz = [None, None, None]  # filled in _on_reset()

        # Initialize metrics (FULL signature required)
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

        # Deterministic RNG (required pattern; may be used by downstream randomization)
        rng = np.random.default_rng(self.seed)

        # Always compute/store canonical GOAL quaternions for metrics (but do NOT spawn in goal)
        for i, (cube, target_letter) in enumerate(zip(self.cubes, self._target_labels)):
            face_idx = cube._labels.index(target_letter)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR_WORLD,
            )
            self._goal_quats_wxyz[i] = np.array(q_goal.elements, dtype=float)

        # --- RANDOM unsolved initialization (within required bounds) ---
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Guarantee NOT successful after reset by enforcing large X spread (> _MAX_X_SPREAD).
        xs = np.array([0.40, 0.70, float(rng.uniform(0.48, 0.62))], dtype=float)  # spread >= 0.30

        # Random Y positions with modest separation to reduce initial interpenetration.
        ys = []
        while len(ys) < 3:
            y = float(rng.uniform(-0.22, 0.22))
            if all(abs(y - y_prev) >= 0.10 for y_prev in ys):
                ys.append(y)
        ys = np.array(ys, dtype=float)

        z = float(self._GOAL_Z)

        for cube, x, y in zip(self.cubes, xs, ys):
            # Random orientation (not necessarily readable toward -X)
            q_rand = (
                Quaternion(axis=[0.0, 0.0, 1.0], angle=float(rng.uniform(0.0, 2.0 * np.pi)))
                * Quaternion(axis=[0.0, 1.0, 0.0], angle=float(rng.uniform(-np.pi, np.pi)))
                * Quaternion(axis=[1.0, 0.0, 0.0], angle=float(rng.uniform(-np.pi, np.pi)))
            )
            cube.set_pose(position=np.array([x, y, z], dtype=float), quat=np.array(q_rand.elements, dtype=float))

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _compute_goal_conditions(self):
        # Lightweight structured description (overwritten logic block per steering meta)
        return {
            "order_along_axis": {
                "axis": "y",
                "left_is_positive": True,
                "sequence": ["X", "Y", "Z"],
                "min_separation_m": self._MIN_Y_SEP,
            },
            "row_alignment": {
                "row_axis": "y",
                "alignment_axes": ["x", "z"],
                "max_x_spread_m": self._MAX_X_SPREAD,
                "max_z_spread_m": self._MAX_Z_SPREAD,
            },
            "orientation": {
                "face_toward_direction": "-X",
                "face_dir_world": self._FACE_DIR_WORLD.tolist(),
            },
        }

    def _success(self) -> bool:
        # RESET success flag at START (critical pattern)
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if any(c is None for c in getattr(self, "cubes", [])):
            return False

        # --- Spatial checks ---
        # Identity-free positional/semantic check.
        # Sort cubes by Y descending so that +Y -> -Y corresponds to slots [X, Y, Z].
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]
        slot_targets = list(self._target_labels)  # ["X","Y","Z"] from +Y to -Y

        # Ordering: +Y is left. Must have slot[0] left of slot[1] left of slot[2].
        ordered_ok = bool(
            (slot_positions[0][1] - slot_positions[1][1] >= self._MIN_Y_SEP)
            and (slot_positions[1][1] - slot_positions[2][1] >= self._MIN_Y_SEP)
        )

        # Row aligned: small spread in X and Z across all three cubes
        xs = np.array([slot_positions[0][0], slot_positions[1][0], slot_positions[2][0]])
        zs = np.array([slot_positions[0][2], slot_positions[1][2], slot_positions[2][2]])
        aligned_ok = bool((float(xs.max() - xs.min()) <= self._MAX_X_SPREAD) and (float(zs.max() - zs.min()) <= self._MAX_Z_SPREAD))

        # On table (helps avoid "held in the air" false positives)
        on_table_ok = all(cube.is_colliding(self.table.table_top) for cube in self.cubes)

        positions_ok = bool(ordered_ok and aligned_ok and on_table_ok)

        # --- Semantic verification (REQUIRED for SemanticCube) ---
        # The cube occupying each slot must show the slot's target letter readable toward the robot.
        semantic_ok = all(
            cube.is_face_readable(letter, face_dir=self._FACE_DIR_WORLD)
            for cube, letter in zip(slot_cubes, slot_targets)
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Mark subgoals complete if overall task complete (simple, robust)
        if positions_ok and semantic_ok and released:
            for idx in range(self._NUM_SUBGOALS):
                self._metric_stage(idx, True)

        self._success_check = bool(positions_ok and semantic_ok and released)


        # Metrics: distances to canonical goal positions + orientation error to goal quats
        target_distance = {}
        pose_error = {}

        for cube, label, goal_pos, goal_quat in zip(self.cubes, self._target_labels, self._goal_positions, self._goal_quats_wxyz):
            pos = cube.body.get_position()
            target_distance[label] = float(np.linalg.norm(np.array(pos) - np.array(goal_pos)))

            # Orientation error (degrees) w.r.t. goal quaternion (best-effort)
            try:
                q_curr = Quaternion(cube.body.get_quaternion())
                q_goal = Quaternion(goal_quat)
                dq = q_goal.inverse * q_curr
                pose_error[label] = float(np.degrees(dq.angle))
            except Exception:
                pose_error[label] = float("nan")

        # Finalize metrics at END (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})