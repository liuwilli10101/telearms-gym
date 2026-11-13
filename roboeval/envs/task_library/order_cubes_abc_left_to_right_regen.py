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
    """Arrange semantic cubes into a left-to-right ABC row with letters on top.

Success Criteria:
- After sorting by descending Y, the left, middle, and right slots contain A, B, and C with minimum Y separation.
- The three cubes are X-aligned, within workspace bounds, and their A/B/C labels are readable on top.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For this flat horizontal arrangement, letters must be readable on the TOP face (+Z).
    _FACE_DIR_TOP = np.array([0.0, 0.0, 1.0])

    # Goal tolerances / task parameters
    _MIN_Y_SEP = 0.03
    _X_ALIGN_TOL = 0.03

    # Workspace bounds requested by proposal (stricter than global bounds)
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _Z_ON_TABLE = 0.97

    # Metrics stages
    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # Create three SemanticCubes using ensure_cubes() workflow (ONLY allowed asset).
        self._target_labels = ["A", "B", "C"]
        cubes_info = ensure_cubes("ABC", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def _xml_key(info: Dict) -> str:
            # Registry text is inconsistent ("xml" vs "xml_path"); support both.
            if "xml" in info:
                return info["xml"]
            return info["xml_path"]

        info_a = cubes_info["A"]
        info_b = cubes_info["B"]
        info_c = cubes_info["C"]

        self.semantic_cube_A = SemanticCube(
            self._mojo, xml_path=str(_xml_key(info_a)), labels=info_a["labels"]
        )
        self.semantic_cube_B = SemanticCube(
            self._mojo, xml_path=str(_xml_key(info_b)), labels=info_b["labels"]
        )
        self.semantic_cube_C = SemanticCube(
            self._mojo, xml_path=str(_xml_key(info_c)), labels=info_c["labels"]
        )

        self.cubes = [self.semantic_cube_A, self.semantic_cube_B, self.semantic_cube_C]

        # Predefine goal positions (single row along Y; +Y is "left", -Y is "right")
        x_goal = float(np.clip(0.55, *self._X_BOUNDS))
        self._goal_positions = {
            "A": np.array([x_goal, 0.10, self._Z_ON_TABLE]),
            "B": np.array([x_goal, 0.00, self._Z_ON_TABLE]),
            "C": np.array([x_goal, -0.10, self._Z_ON_TABLE]),
        }

        # Initialize metrics (FULL signature) after objects exist
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

        rng = np.random.default_rng(self.seed)

        # --- Randomize into UNSOLVED state (guarantee _success() == False) ---
        # Sample 3 non-overlapping positions within bounds on the table.
        min_sep = 0.06  # conservative to avoid interpenetration/contact at reset
        positions = []
        max_tries = 500

        for _ in range(3):
            for _try in range(max_tries):
                x = float(rng.uniform(self._X_BOUNDS[0], self._X_BOUNDS[1]))
                y = float(rng.uniform(self._Y_BOUNDS[0], self._Y_BOUNDS[1]))
                p = np.array([x, y, self._Z_ON_TABLE], dtype=float)
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_sep for q in positions):
                    positions.append(p)
                    break
            else:
                # Fallback: spread along Y within bounds if rejection sampling fails
                base_x = float(np.clip(0.55, *self._X_BOUNDS))
                ys = np.linspace(self._Y_BOUNDS[0] + 0.05, self._Y_BOUNDS[1] - 0.05, 3)
                rng.shuffle(ys)
                positions = [np.array([base_x, float(ys[i]), self._Z_ON_TABLE], dtype=float) for i in range(3)]
                break

        # Force semantic failure by placing a WRONG glyph on TOP for every cube.
        # (Even if positions accidentally satisfy ordering/alignment, semantic_ok will be False.)
        self._goal_quats = {}
        for cube, letter, pos in zip(self.cubes, self._target_labels, positions):
            labels = list(cube._labels)
            wrong_choices = [l for l in labels if l != letter]
            wrong_letter = str(rng.choice(wrong_choices))
            wrong_face_idx = labels.index(wrong_letter)

            q = cube.get_orientation_for_face_and_glyph(
                face_idx=wrong_face_idx,
                face_dir_world=self._FACE_DIR_TOP,
            )

            # Keep stored goal quats dict for downstream metric code compatibility
            # (not used for reset pose any more).
            face_idx_goal = labels.index(letter)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx_goal,
                face_dir_world=self._FACE_DIR_TOP,
            )
            self._goal_quats[letter] = q_goal

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

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free positional/semantic check.
        # Sort cubes by Y desc so slot[0]=leftmost->target "A", slot[1]="B", slot[2]="C".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]
        slot_targets = list(self._target_labels)  # ["A","B","C"] from +Y to -Y

        pA, pB, pC = slot_positions[0], slot_positions[1], slot_positions[2]

        # --- Spatial checks ---
        # Left-to-right ordering is +Y (left) to -Y (right)
        y_order_ok = (pA[1] >= pB[1] + self._MIN_Y_SEP) and (pB[1] >= pC[1] + self._MIN_Y_SEP)

        # X alignment (same row) within tolerance
        xs = np.array([pA[0], pB[0], pC[0]])
        x_align_ok = (xs.max() - xs.min()) <= self._X_ALIGN_TOL

        # (Optional sanity) Keep within proposal workspace bounds
        within_bounds = True
        for p in (pA, pB, pC):
            within_bounds = within_bounds and (self._X_BOUNDS[0] <= p[0] <= self._X_BOUNDS[1])
            within_bounds = within_bounds and (self._Y_BOUNDS[0] <= p[1] <= self._Y_BOUNDS[1])

        positions_ok = bool(y_order_ok and x_align_ok and within_bounds)

        # --- SEMANTIC verification (REQUIRED) ---
        # The cube occupying each slot must show the slot's target letter readable on top.
        semantic_ok = all(
            c.is_face_readable(t, face_dir=self._FACE_DIR_TOP)
            for c, t in zip(slot_cubes, slot_targets)
        )

        # Slot-aliased cubes for downstream stage/metric reporting.
        cube_A, cube_B, cube_C = slot_cubes[0], slot_cubes[1], slot_cubes[2]

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Stage metrics (simple, position/orientation based) ---
        # 0: A correct (pose+semantic)
        self._metric_stage(
            0,
            bool(
                np.linalg.norm(pA - self._goal_positions["A"]) <= 0.05
                and cube_A.is_face_readable("A", face_dir=self._FACE_DIR_TOP)
            ),
        )
        # 1: B correct
        self._metric_stage(
            1,
            bool(
                np.linalg.norm(pB - self._goal_positions["B"]) <= 0.05
                and cube_B.is_face_readable("B", face_dir=self._FACE_DIR_TOP)
            ),
        )
        # 2: C correct
        self._metric_stage(
            2,
            bool(
                np.linalg.norm(pC - self._goal_positions["C"]) <= 0.05
                and cube_C.is_face_readable("C", face_dir=self._FACE_DIR_TOP)
            ),
        )
        # 3: A left of B
        self._metric_stage(3, bool(pA[1] >= pB[1] + self._MIN_Y_SEP))
        # 4: B left of C
        self._metric_stage(4, bool(pB[1] >= pC[1] + self._MIN_Y_SEP))
        # 5: X aligned
        self._metric_stage(5, bool(x_align_ok))
        # 6: released
        self._metric_stage(6, bool(released))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Final metrics ---
        # Target distances to desired goal positions
        target_distance = {
            "A_to_goal_m": float(np.linalg.norm(pA - self._goal_positions["A"])),
            "B_to_goal_m": float(np.linalg.norm(pB - self._goal_positions["B"])),
            "C_to_goal_m": float(np.linalg.norm(pC - self._goal_positions["C"])),
            "min_y_sep_AB_m": float(pA[1] - pB[1]),
            "min_y_sep_BC_m": float(pB[1] - pC[1]),
            "x_spread_m": float(xs.max() - xs.min()),
        }

        # Orientation error (degrees) vs desired "letter on top" quaternion
        pose_error = {}
        for cube, letter in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(letter)
            desired_q = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx, face_dir_world=self._FACE_DIR_TOP
            )
            current_q = cube.body.get_quaternion()

            # Quaternion angular distance (robust to sign flip)
            q_cur = Quaternion(current_q)
            q_des = Quaternion(desired_q.elements)
            dq = q_des.inverse * q_cur
            angle_rad = 2.0 * np.arccos(np.clip(abs(dq.w), 0.0, 1.0))
            pose_error[f"{letter}_angle_err_deg"] = float(np.degrees(angle_rad))

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})