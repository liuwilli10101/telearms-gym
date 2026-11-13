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

class AlphabetCubeRowArrange(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic cubes to form a left-to-right row reading A, B, C from the robot's viewpoint,
    with each letter facing the robot (visible on the back (-X) face).

    Sequential Sub-Goals:
        1. Ensure all cubes are resting on the table (not stacked) and released.
        2. Order cubes left-to-right as A, then B, then C (along +Y to -Y).
        3. Align cubes into a straight row (same table-X within tolerance) with small gaps.
        4. Orient each cube so its target letter is readable on the back (-X) face toward the robot.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Letters should face the robot camera on the back (-X) face
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Task parameters
    _TARGET_WORD = "ABC"
    _MIN_Y_SEP = 0.04
    _X_ALIGN_TOL = 0.03
    _TABLE_Z = 0.97
    _MAX_Z_ERR = 0.03

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Create SemanticCubes (ONLY allowed asset) ---
        cubes_info = ensure_cubes(self._TARGET_WORD, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self._target_labels = list(self._TARGET_WORD)

        # Instantiate in A,B,C identity order (cube_A, cube_B, cube_C)
        self.cube_A = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["A"]["xml"]),
            labels=cubes_info["A"]["labels"],
        )
        self.cube_B = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["B"]["xml"]),
            labels=cubes_info["B"]["labels"],
        )
        self.cube_C = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["C"]["xml"]),
            labels=cubes_info["C"]["labels"],
        )
        self.cubes = [self.cube_A, self.cube_B, self.cube_C]

        # Goal poses (centered at X=0.55; left-to-right is +Y -> -Y)
        self._goal_positions = {
            "A": np.array([0.55, 0.08, self._TABLE_Z]),
            "B": np.array([0.55, 0.00, self._TABLE_Z]),
            "C": np.array([0.55, -0.08, self._TABLE_Z]),
        }
        self._goal_quats = {}  # filled in _on_reset

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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3. Compute/record GOAL quaternions for metrics (but DO NOT place cubes at goal)
        self._goal_quats = {}
        for cube, letter in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(letter)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            self._goal_quats[letter] = q_goal

        # 4. Place objects in RANDOM unsolved positions within workspace bounds
        #    Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = float(self._TABLE_Z)

        # Keep cubes separated to avoid initial collisions; cube edge ~0.05 => use conservative sep
        min_sep = 0.08

        # Sample positions with rejection to ensure separation
        positions = []
        for _i in range(len(self.cubes)):
            for _attempt in range(500):
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_sep for q in positions):
                    positions.append(p)
                    break
            else:
                # Fallback (should be rare): deterministic spread
                base_x = 0.50 + 0.05 * (_i - 1)
                base_y = 0.18 - 0.18 * _i
                positions.append(np.array([np.clip(base_x, x_lo, x_hi), np.clip(base_y, y_lo, y_hi), z], dtype=float))

        # Force at least one cube to be semantically incorrect by rotating away from the readable goal orientation
        # (ensures _success() is False immediately after reset)
        rot_bad = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(np.pi / 2.0))

        for cube, letter, pos in zip(self.cubes, self._target_labels, positions):
            q_goal = self._goal_quats[letter]
            if letter == "A":
                q_set = q_goal * rot_bad  # guaranteed not readable as goal
            else:
                # Random stable yaw about Z (keeps cube resting on table)
                yaw = float(rng.uniform(-np.pi, np.pi))
                q_set = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(position=pos, quat=q_set.elements)

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        self._num_subgoals = 4
        for idx in range(self._num_subgoals):
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

        # --- Read current poses ---
        pA = self.cube_A.body.get_position()
        pB = self.cube_B.body.get_position()
        pC = self.cube_C.body.get_position()

        # --- On-table checks (not stacked): require contact with table top and near nominal Z ---
        on_table_A = self.cube_A.is_colliding(self.table.table_top) and (abs(pA[2] - self._TABLE_Z) <= self._MAX_Z_ERR)
        on_table_B = self.cube_B.is_colliding(self.table.table_top) and (abs(pB[2] - self._TABLE_Z) <= self._MAX_Z_ERR)
        on_table_C = self.cube_C.is_colliding(self.table.table_top) and (abs(pC[2] - self._TABLE_Z) <= self._MAX_Z_ERR)
        on_table_ok = on_table_A and on_table_B and on_table_C

        # --- Identity-free positional/semantic check ---
        # Sort cubes by Y descending so that left (+Y) -> right (-Y) from robot viewpoint
        # corresponds to slots [A, B, C] regardless of which cube body is which.
        cubes_with_pos = [(self.cube_A, pA), (self.cube_B, pB), (self.cube_C, pC)]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]

        # Relative ordering along table_y with required separation
        order_AB = (slot_positions[0][1] - slot_positions[1][1]) >= self._MIN_Y_SEP
        order_BC = (slot_positions[1][1] - slot_positions[2][1]) >= self._MIN_Y_SEP
        order_ok = order_AB and order_BC

        # Straight row alignment (same table-x within tolerance) across all three slots
        sx0, sx1, sx2 = slot_positions[0][0], slot_positions[1][0], slot_positions[2][0]
        align_ok = (abs(sx0 - sx1) <= self._X_ALIGN_TOL) and (abs(sx0 - sx2) <= self._X_ALIGN_TOL)

        # Avoid overlaps/collisions between any pair of cubes
        no_cube_collision = (
            (not self.cube_A.is_colliding(self.cube_B))
            and (not self.cube_B.is_colliding(self.cube_C))
            and (not self.cube_A.is_colliding(self.cube_C))
        )

        positions_ok = bool(on_table_ok and order_ok and align_ok and no_cube_collision)

        # Semantic verification: the cube occupying each slot must show the slot's target
        # letter readable toward the robot. Identity (which cube body) does not matter.
        slot_targets = ["A", "B", "C"]
        semantic_ok = all(
            cube.is_face_readable(letter, face_dir=self._FACE_DIR)
            for cube, letter in zip(slot_cubes, slot_targets)
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging ---
        # 0: all cubes on table and released
        self._metric_stage(0, bool(on_table_ok and released))
        # 1: ordering achieved
        self._metric_stage(1, bool(on_table_ok and order_ok))
        # 2: row alignment + gaps (no collisions)
        self._metric_stage(2, bool(on_table_ok and order_ok and align_ok and no_cube_collision))
        # 3: semantic readability achieved
        self._metric_stage(3, bool(semantic_ok))

        # --- Final success ---
        self._success_check = bool(positions_ok and semantic_ok and released)

        slot_readable = [
            cube.is_face_readable(letter, face_dir=self._FACE_DIR)
            for cube, letter in zip(slot_cubes, slot_targets)
        ]


        # --- Metrics finalize ---
        # Target distances to nominal goal positions
        goalA = self._goal_positions["A"]
        goalB = self._goal_positions["B"]
        goalC = self._goal_positions["C"]

        dA = float(np.linalg.norm(pA - goalA))
        dB = float(np.linalg.norm(pB - goalB))
        dC = float(np.linalg.norm(pC - goalC))

        # Orientation error (deg) relative to goal quaternions from _on_reset()
        def _quat_angle_deg(q_now, q_goal: Quaternion) -> float:
            qn = Quaternion(q_now)
            qe = q_goal.inverse * qn
            w = np.clip(abs(qe.w), -1.0, 1.0)
            return float(np.degrees(2.0 * np.arccos(w)))

        qA_now = self.cube_A.body.get_quaternion()
        qB_now = self.cube_B.body.get_quaternion()
        qC_now = self.cube_C.body.get_quaternion()

        angA = _quat_angle_deg(qA_now, self._goal_quats.get("A", Quaternion()))
        angB = _quat_angle_deg(qB_now, self._goal_quats.get("B", Quaternion()))
        angC = _quat_angle_deg(qC_now, self._goal_quats.get("C", Quaternion()))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"cube_A": dA, "cube_B": dB, "cube_C": dC},
            pose_error={"cube_A_deg": angA, "cube_B_deg": angB, "cube_C_deg": angC},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})