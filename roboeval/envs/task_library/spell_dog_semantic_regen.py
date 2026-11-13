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

class DogCubeRowArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic letter cubes into a left-to-right row that spells DOG.
    Ensure the D, O, and G faces are oriented toward the robot (toward -X).

    Sequential Sub-Goals:
        1. Grasp the D cube with either gripper.
        2. Move the D cube to the LEFT position in the row (higher y; +Y) at a consistent x with the row.
        3. Rotate the D cube so the letter face points toward the robot (toward -X).
        4. Release the D cube in its final leftmost position.
        5. Grasp the O cube with either gripper.
        6. Move the O cube to the middle position between D and G along the Y axis.
        7. Rotate the O cube so the letter face points toward the robot (toward -X).
        8. Release the O cube in its final middle position.
        9. Grasp the G cube with either gripper.
        10. Move the G cube to the RIGHT position in the row (lower y; -Y) at a consistent x with the row.
        11. Rotate the G cube so the letter face points toward the robot (toward -X).
        12. Release the G cube in its final rightmost position.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task requires letter faces toward robot: -X
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Goal tolerances from spec
    _LEFT_OF_TOL_M = 0.02
    _ALIGN_X_TOL_M = 0.03
    _MAX_Y_SPAN_M = 0.35

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create SemanticCubes for D, O, G (ONLY allowed asset type)
        self._word = "DOG"
        self._target_labels = list(self._word)

        cubes_info = ensure_cubes(self._word, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.cube_D = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["D"]["xml"]),
            labels=cubes_info["D"]["labels"],
        )
        self.cube_O = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["O"]["xml"]),
            labels=cubes_info["O"]["labels"],
        )
        self.cube_G = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["G"]["xml"]),
            labels=cubes_info["G"]["labels"],
        )

        self.cubes = [self.cube_D, self.cube_O, self.cube_G]

        # Fixed "canonical" goal poses used for metrics (success is relational, not absolute)
        x = 0.55
        spacing = 0.06  # >= 0.05m recommended minimum spacing
        self._goal_positions = {
            "D": np.array([x, +spacing, 0.97]),
            "O": np.array([x, 0.0, 0.97]),
            "G": np.array([x, -spacing, 0.97]),
        }
        self._goal_quats = {}  # populated in _on_reset()

        # Metrics (FULL required signature)
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

        # Deterministic RNG (required pattern; pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # --- Populate canonical goal quats for metrics (do NOT place cubes in goal) ---
        self._goal_quats = {}
        for cube, target_letter in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(target_letter)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            self._goal_quats[target_letter] = q_goal.elements

        # --- Randomize cube poses into an UNSOLVED state within workspace bounds ---
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Keep some minimum separation to avoid interpenetration
        min_xy_sep = 0.06

        positions = []
        for _i in range(len(self.cubes)):
            for _try in range(500):
                p = np.array(
                    [
                        rng.uniform(x_lo, x_hi),
                        rng.uniform(y_lo, y_hi),
                        z_base,
                    ],
                    dtype=float,
                )
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_xy_sep for q in positions):
                    positions.append(p)
                    break
            else:
                # Fallback (still within bounds) if sampling fails
                positions.append(
                    np.array(
                        [
                            rng.uniform(x_lo, x_hi),
                            rng.uniform(y_lo, y_hi),
                            z_base,
                        ],
                        dtype=float,
                    )
                )

        # Force semantic mismatch by orienting each target letter AWAY from robot (+X)
        away_from_robot = -self._FACE_DIR  # +X

        for cube, target_letter, pos in zip(self.cubes, self._target_labels, positions):
            face_idx = cube._labels.index(target_letter)
            q_away = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=away_from_robot,
            )
            cube.set_pose(position=pos, quat=q_away.elements)

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(12):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free positional/semantic check.
        # Sort cubes by Y desc so slot[0]=left target "D", slot[1]="O", slot[2]="G".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]
        slot_targets = ["D", "O", "G"]

        # Slot aliases (preserve downstream variable names)
        cube_D_slot, cube_O_slot, cube_G_slot = slot_cubes[0], slot_cubes[1], slot_cubes[2]
        pD, pO, pG = slot_positions[0], slot_positions[1], slot_positions[2]

        # --- Spatial checks (axis mapping is non-negotiable: left/right is Y) ---
        d_left_of_o = (pD[1] - pO[1]) > self._LEFT_OF_TOL_M
        o_left_of_g = (pO[1] - pG[1]) > self._LEFT_OF_TOL_M

        xs = np.array([pD[0], pO[0], pG[0]])
        ys = np.array([pD[1], pO[1], pG[1]])

        x_aligned = (np.max(xs) - np.min(xs)) <= self._ALIGN_X_TOL_M
        y_span_ok = (np.max(ys) - np.min(ys)) <= self._MAX_Y_SPAN_M

        # Keep cubes on the table surface (helps reject lifted solutions)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        positions_ok = bool(d_left_of_o and o_left_of_g and x_aligned and y_span_ok and on_table)

        # --- Semantic verification: cube currently in each slot must show that slot's target letter ---
        semantic_ok = all(
            c.is_face_readable(t, face_dir=self._FACE_DIR)
            for c, t in zip(slot_cubes, slot_targets)
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal metrics (approximated via state predicates) ---
        # 0-3: D slot (on table, left of O slot, oriented, released)
        self._metric_stage(0, bool(cube_D_slot.is_colliding(self.table.table_top)))
        self._metric_stage(1, bool(d_left_of_o))
        self._metric_stage(2, bool(cube_D_slot.is_face_readable("D", face_dir=self._FACE_DIR)))
        self._metric_stage(
            3,
            bool(
                not any(
                    self.robot.is_gripper_holding_object(cube_D_slot, s)
                    for s in self.robot.grippers
                )
            ),
        )

        # 4-7: O slot (on table, between D and G along Y, oriented, released)
        o_between = (pD[1] - pO[1]) > self._LEFT_OF_TOL_M and (pO[1] - pG[1]) > self._LEFT_OF_TOL_M
        self._metric_stage(4, bool(cube_O_slot.is_colliding(self.table.table_top)))
        self._metric_stage(5, bool(o_between))
        self._metric_stage(6, bool(cube_O_slot.is_face_readable("O", face_dir=self._FACE_DIR)))
        self._metric_stage(
            7,
            bool(
                not any(
                    self.robot.is_gripper_holding_object(cube_O_slot, s)
                    for s in self.robot.grippers
                )
            ),
        )

        # 8-11: G slot (on table, O left of G, oriented, released)
        self._metric_stage(8, bool(cube_G_slot.is_colliding(self.table.table_top)))
        self._metric_stage(9, bool(o_left_of_g))
        self._metric_stage(10, bool(cube_G_slot.is_face_readable("G", face_dir=self._FACE_DIR)))
        self._metric_stage(
            11,
            bool(
                not any(
                    self.robot.is_gripper_holding_object(cube_G_slot, s)
                    for s in self.robot.grippers
                )
            ),
        )

        # Overall success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Finalize metrics (MUST be last metric call before return) ---
        # Target distance: to canonical goal poses (for logging/analysis)
        goalD = self._goal_positions["D"]
        goalO = self._goal_positions["O"]
        goalG = self._goal_positions["G"]

        def _ang_err_deg(cube, letter: str) -> float:
            if letter not in self._goal_quats:
                return float("nan")
            q_goal = Quaternion(self._goal_quats[letter])
            q_act = Quaternion(cube.body.get_quaternion())
            dq = q_goal.inverse * q_act
            return float(np.degrees(dq.angle))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_D_to_goal_m": float(np.linalg.norm(pD - goalD)),
                "cube_O_to_goal_m": float(np.linalg.norm(pO - goalO)),
                "cube_G_to_goal_m": float(np.linalg.norm(pG - goalG)),
                "row_x_span_m": float(np.max(xs) - np.min(xs)),
                "row_y_span_m": float(np.max(ys) - np.min(ys)),
            },
            pose_error={
                "cube_D_ang_err_deg": _ang_err_deg(cube_D_slot, "D"),
                "cube_O_ang_err_deg": _ang_err_deg(cube_O_slot, "O"),
                "cube_G_ang_err_deg": _ang_err_deg(cube_G_slot, "G"),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})