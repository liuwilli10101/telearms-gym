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

class CatWordCubeRow(RoboEvalEnv, MetricRolloutEval):
    """Spell CAT with three semantic cubes in a left-to-right row.

Success Criteria:
- After sorting by descending Y, the row slots read C, A, T with minimum Y separations.
- The cubes are X-aligned, near table height, and the C/A/T labels are readable in the required face direction.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _TARGET_WORD = "CAT"
    _NUM_SUBGOALS = 7

    # Face direction required by task spec ("facing the robot" toward -X)
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Goal layout (must read left-to-right as +Y -> -Y)
    _GOAL_POSITIONS = {
        "C": np.array([0.55, 0.05, 0.97]),
        "A": np.array([0.55, 0.00, 0.97]),
        "T": np.array([0.55, -0.05, 0.97]),
    }

    # Tolerances from task spec
    _X_ALIGN_TOL = 0.03
    _Z_TOL = 0.02
    _MIN_Y_SEP = 0.05

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create SemanticCubes via ensure_cubes workflow (ONLY asset allowed)
        cubes_info = ensure_cubes(self._TARGET_WORD, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.cubes = []
        self._cube_by_letter = {}
        for ch in self._TARGET_WORD:
            info = cubes_info[ch]
            xml_path = info.get("xml", info.get("xml_path"))
            cube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=info["labels"])
            self.cubes.append(cube)
            self._cube_by_letter[ch] = cube

        # Metrics initialization (full signature)
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

        # Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # --- Randomize objects into an UNSOLVED state within workspace bounds ---
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        z0 = 0.97

        # Guarantee failure of _success(): break X alignment strongly and keep Y separations small
        # (so y_order_ok and/or x_aligned_ok will be False).
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Choose a center point then place cubes with large X offsets (violates x_aligned_ok)
        x_center = float(rng.uniform(x_min + 0.12, x_max - 0.12))
        y_center = float(rng.uniform(y_min + 0.03, y_max - 0.03))

        # Fixed offsets to ensure x-alignment check fails (max deviation > 0.03)
        x_offsets = {
            "C": -0.08,
            "A": 0.00,
            "T": 0.08,
        }

        # Keep Y very close so required separations (>= 0.05) are violated
        y_offsets = {
            "C": float(rng.uniform(-0.01, 0.01)),
            "A": float(rng.uniform(-0.01, 0.01)),
            "T": float(rng.uniform(-0.01, 0.01)),
        }

        for ch in self._TARGET_WORD:
            cube = self._cube_by_letter[ch]

            x = float(np.clip(x_center + x_offsets[ch], x_min, x_max))
            y = float(np.clip(y_center + y_offsets[ch], y_min, y_max))
            pos = np.array([x, y, z0], dtype=float)

            # Random orientation (not guaranteed to be readable)
            roll = float(rng.uniform(-np.pi, np.pi))
            pitch = float(rng.uniform(-np.pi, np.pi))
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw) * Quaternion(axis=[0, 1, 0], angle=pitch) * Quaternion(
                axis=[1, 0, 0], angle=roll
            )
            cube.set_pose(position=pos, quat=q.elements)

        # Physics settling
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_cube_by_letter"):
            return False

        # Identity-free positional/semantic check.
        # Sort cubes by Y desc so slot[0]=left target "C", slot[1]="A", slot[2]="T".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]
        slot_targets = list(self._TARGET_WORD)  # "C","A","T" left to right

        cube_C, cube_A, cube_T = slot_cubes[0], slot_cubes[1], slot_cubes[2]
        pC, pA, pT = slot_positions[0], slot_positions[1], slot_positions[2]

        # --- Spatial checks ---
        # 1) Left-to-right reading along Y (+Y to -Y): slot[0] left of slot[1], slot[1] left of slot[2]
        # Use a small epsilon to avoid float/settling edge cases at exactly MIN_Y_SEP.
        eps = 1e-3
        y_order_ok = (pC[1] - pA[1] >= (self._MIN_Y_SEP - eps)) and (pA[1] - pT[1] >= (self._MIN_Y_SEP - eps))

        # 2) Single row: approximately same X and resting on table (Z)
        mean_x = float((pC[0] + pA[0] + pT[0]) / 3.0)
        x_aligned_ok = (abs(pC[0] - mean_x) <= self._X_ALIGN_TOL and
                        abs(pA[0] - mean_x) <= self._X_ALIGN_TOL and
                        abs(pT[0] - mean_x) <= self._X_ALIGN_TOL)

        z_on_table_ok = (abs(pC[2] - 0.97) <= self._Z_TOL and
                         abs(pA[2] - 0.97) <= self._Z_TOL and
                         abs(pT[2] - 0.97) <= self._Z_TOL)

        positions_ok = bool(y_order_ok and x_aligned_ok and z_on_table_ok)

        # --- Semantic verification: cube currently in each slot must show that slot's target letter ---
        semantic_ok = all(
            cube.is_face_readable(ch, face_dir=self._FACE_DIR)
            for cube, ch in zip(slot_cubes, slot_targets)
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging (coarse, but consistent and monotonic) ---
        # 0: grasp (cannot be reliably inferred at success time; leave False)
        # 1-3: arrangement
        self._metric_stage(1, bool(x_aligned_ok and z_on_table_ok))
        self._metric_stage(2, bool(pC[1] - pA[1] >= (self._MIN_Y_SEP - eps)))
        self._metric_stage(3, bool(pA[1] - pT[1] >= (self._MIN_Y_SEP - eps)))
        # 4-6: orientation + release
        self._metric_stage(4, bool(cube_C.is_face_readable("C", face_dir=self._FACE_DIR)))
        self._metric_stage(5, bool(cube_A.is_face_readable("A", face_dir=self._FACE_DIR)))
        self._metric_stage(6, bool(cube_T.is_face_readable("T", face_dir=self._FACE_DIR) and released))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "CA_y_sep_error": float(abs((pC[1] - pA[1]) - self._MIN_Y_SEP)),
            "AT_y_sep_error": float(abs((pA[1] - pT[1]) - self._MIN_Y_SEP)),
        }
        pose_error = {
            "x_alignment_max_error": float(max(abs(pC[0] - mean_x), abs(pA[0] - mean_x), abs(pT[0] - mean_x))),
            "z_height_max_error": float(max(abs(pC[2] - 0.97), abs(pA[2] - 0.97), abs(pT[2] - 0.97))),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})