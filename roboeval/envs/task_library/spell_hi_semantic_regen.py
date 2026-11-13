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

class SpellHiCubesTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place two semantic letter cubes side by side to spell 'HI', with both letters facing
    the robot/camera. The 'H' cube must be on the left (+Y) and the 'I' cube on the
    right (-Y), aligned in front/back (X).

    Sequential Sub-Goals:
        1. Place cubes at the target positions (H at left, I at right).
        2. Ensure H is to the left of I along +Y (min separation).
        3. Align the cubes in X (front/back alignment).
        4. Orient H so its visible -X face shows 'H' toward the robot/camera.
        5. Orient I so its visible -X face shows 'I' toward the robot/camera.
        6. Release both cubes.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task requires letters to face the robot/camera on the visible -X face
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Computed exact goal positions (line of 2 cubes along Y; left is +Y).
    # Use 0.05 m offsets so the 0.04 m cubes have a comfortable gap that
    # comfortably exceeds _MIN_Y_SEP (success threshold).
    _GOAL_POSITIONS = {
        "H": np.array([0.55, 0.05, 0.97]),
        "I": np.array([0.55, -0.05, 0.97]),
    }

    # Goal tolerances / constraints
    _POSE_TOL = 0.014          # success tolerance (auto-generated)
    _X_ALIGN_TOL = 0.03        # alignment tolerance from spec
    _MIN_Y_SEP = 0.05          # min left-of separation from spec

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create SemanticCubes (ONLY allowed asset)
        cubes_info = ensure_cubes("HI", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        info_h = cubes_info["H"]
        info_i = cubes_info["I"]

        self.cube_H = SemanticCube(self._mojo, xml_path=str(info_h["xml"]), labels=info_h["labels"])
        self.cube_I = SemanticCube(self._mojo, xml_path=str(info_i["xml"]), labels=info_i["labels"])

        self.cubes = [self.cube_H, self.cube_I]
        self._target_labels = ["H", "I"]

        # Initialize metrics (full signature required)
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
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Random UNSOLVED placement: ensure not at goal and not semantically correct
        placed_positions = []
        min_obj_sep = 0.06  # conservative separation to avoid initial overlap

        for cube, target_letter in zip(self.cubes, self._target_labels):
            # Sample a random position in-bounds and away from goal tolerance
            goal_xy = self._GOAL_POSITIONS[target_letter][:2]
            for _ in range(200):
                pos = np.array(
                    [
                        rng.uniform(x_lo, x_hi),
                        rng.uniform(y_lo, y_hi),
                        z_base,
                    ],
                    dtype=float,
                )

                # Reject if too close to its goal in XY
                if np.linalg.norm((pos[:2] - goal_xy)) <= (self._POSE_TOL * 2.0):
                    continue

                # Reject if too close to already placed cube(s)
                if any(np.linalg.norm(pos[:2] - p[:2]) < min_obj_sep for p in placed_positions):
                    continue

                placed_positions.append(pos)
                break
            else:
                # Fallback: deterministic unsolved offset (still in bounds)
                pos = self._GOAL_POSITIONS[target_letter].copy()
                pos[0] = float(np.clip(pos[0] + 0.10, x_lo, x_hi))
                pos[1] = float(np.clip(pos[1] + (0.12 if target_letter == "H" else -0.12), y_lo, y_hi))
                pos[2] = z_base
                placed_positions.append(pos)

            # Force a WRONG glyph to face the camera so the semantic check fails
            # deterministically. Pick the wrong label from the rng so reset
            # diversity is preserved across seeds (not always the same face).
            wrong_candidates = [lbl for lbl in cube._labels if lbl != target_letter]
            wrong_label = wrong_candidates[int(rng.integers(0, len(wrong_candidates)))]
            face_idx_wrong = cube._labels.index(wrong_label)
            q = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx_wrong,
                face_dir_world=self._FACE_DIR,  # NOTE: face_dir_world parameter name
            )
            # Add a random extra yaw to vary in-plane glyph rotation between resets.
            extra_yaw = float(rng.uniform(-np.pi, np.pi))
            q_extra = Quaternion(axis=[0.0, 0.0, 1.0], angle=extra_yaw)
            q_final = q_extra * q
            cube.set_pose(position=pos, quat=q_final.elements)

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
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

        # Identity-free: sort cubes by Y desc so slot[0]=left target "H", slot[1]=right target "I".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_H_cube, pos_h = cubes_sorted[0]
        slot_I_cube, pos_i = cubes_sorted[1]

        # --- Spatial checks (relative only) ---
        goal_h = self._GOAL_POSITIONS["H"]
        goal_i = self._GOAL_POSITIONS["I"]

        # On-table check (cubes resting on table top, near nominal Z)
        on_table_h = slot_H_cube.is_colliding(self.table.table_top) and (abs(pos_h[2] - 0.97) <= 0.03)
        on_table_i = slot_I_cube.is_colliding(self.table.table_top) and (abs(pos_i[2] - 0.97) <= 0.03)
        on_table_ok = bool(on_table_h and on_table_i)

        # Relative: H must be left of I along +Y with minimum separation
        left_of_ok = bool((pos_h[1] - pos_i[1]) >= self._MIN_Y_SEP)

        # Alignment: similar X (front/back)
        x_aligned_ok = bool(abs(pos_h[0] - pos_i[0]) <= self._X_ALIGN_TOL)

        positions_ok = bool(on_table_ok and left_of_ok and x_aligned_ok)

        # --- Semantic verification: cube in left slot must show "H", in right slot must show "I" ---
        semantic_ok = True
        if not slot_H_cube.is_face_readable("H", face_dir=self._FACE_DIR):
            semantic_ok = False
        if not slot_I_cube.is_face_readable("I", face_dir=self._FACE_DIR):
            semantic_ok = False

        # --- Release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging ---
        # 0: cubes on table
        self._metric_stage(0, on_table_ok)
        # 1: left-of relation
        self._metric_stage(1, left_of_ok)
        # 2: x alignment
        self._metric_stage(2, x_aligned_ok)
        # 3: H readable (left slot)
        self._metric_stage(3, slot_H_cube.is_face_readable("H", face_dir=self._FACE_DIR))
        # 4: I readable (right slot)
        self._metric_stage(4, slot_I_cube.is_face_readable("I", face_dir=self._FACE_DIR))
        # 5: released
        self._metric_stage(5, released)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        readable_H = slot_H_cube.is_face_readable("H", face_dir=self._FACE_DIR)
        readable_I = slot_I_cube.is_face_readable("I", face_dir=self._FACE_DIR)
        print(
            f"[spell_hi] on_table={on_table_ok} "
            f"left_of={left_of_ok} x_aligned={x_aligned_ok} "
            f"readable=H:{readable_H} I:{readable_I} released={released}"
        )

        # Finalize metrics (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "H_xy_to_goal": float(np.linalg.norm((pos_h - goal_h)[:2])),
                "I_xy_to_goal": float(np.linalg.norm((pos_i - goal_i)[:2])),
                "H_to_I": float(distance(slot_H_cube.body, slot_I_cube.body)),
            },
            pose_error={
                "y_left_of_margin": float((pos_h[1] - pos_i[1]) - self._MIN_Y_SEP),
                "x_alignment_error": float(abs(pos_h[0] - pos_i[0])),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})