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


class SemanticCubePyramidBuild(RoboEvalEnv, MetricRolloutEval):
    """
    Build a 3-cube pyramid: place two semantic cubes as a left-right base and stack
    the A cube centered on top. The top cube must have the letter A facing forward
    toward the robot/camera (-X).

    Sequential Sub-Goals:
        1. Grasp one base cube (B or C).
        2. Place cube B on the table on the LEFT (+Y).
        3. Grasp the other base cube (C or B).
        4. Place cube C on the table on the RIGHT (-Y).
        5. Grasp cube A.
        6. Rotate cube A so the A face points forward (-X).
        7. Place cube A centered above the two base cubes.
        8. Release cube A once stable.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Forward toward robot/camera for this task is -X (use SAME vector for reset + success)
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # SemanticCube constants
    _Z_TABLE = 0.97
    _CUBE_SIZE = 0.04
    _Z_STACK = 0.04

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create EXACTLY the required assets: SemanticCube only
        self._target_labels = ["A", "B", "C"]
        cubes_info = ensure_cubes("".join(self._target_labels), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.cubes = []
        for ch in self._target_labels:
            info = cubes_info[ch]
            cube = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
            self.cubes.append(cube)

        self.cube_A, self.cube_B, self.cube_C = self.cubes

        # Goal geometry (stable pyramid / bridge-on-two)
        cx, cy = 0.55, 0.0
        base_half_sep = 0.02  # centers separated by 0.04m (touching)
        self._goal_pos_B = np.array([cx, cy + base_half_sep, self._Z_TABLE])  # left (+Y)
        self._goal_pos_C = np.array([cx, cy - base_half_sep, self._Z_TABLE])  # right (-Y)
        self._goal_pos_A = np.array([cx, cy, self._Z_TABLE + self._Z_STACK])  # centered on seam

        self._goal_positions = {
            "A": self._goal_pos_A,
            "B": self._goal_pos_B,
            "C": self._goal_pos_C,
        }
        self._goal_quats = {"A": None, "B": None, "C": None}

        # Metrics (FULL signature required)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _sample_initial_state(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """
        Deterministic (seeded) sampling helper. Not used to randomize _on_reset() because
        validation expects a solvable goal-state reset, but provided for compatibility.
        """
        # Sample three non-overlapping-ish table positions (no random yaw; keep readable)
        xs = rng.uniform(0.42, 0.68, size=3)
        ys = rng.uniform(-0.20, 0.20, size=3)
        return {
            "A": np.array([xs[0], ys[0], self._Z_TABLE]),
            "B": np.array([xs[1], ys[1], self._Z_TABLE]),
            "C": np.array([xs[2], ys[2], self._Z_TABLE]),
        }

    def _on_reset(self):
        # 1) Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3) Place objects in RANDOM UNSOLVED configuration (all on table, separated, random yaw)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._Z_TABLE

        # Keep cubes from interpenetrating; also avoid accidentally "touching both" patterns.
        min_center_dist = 0.06  # > cube size (0.04), gives clearance

        def sample_positions() -> Dict[str, np.ndarray]:
            letters = ["A", "B", "C"]
            pts = {}
            for L in letters:
                for _ in range(500):
                    p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float)
                    if all(np.linalg.norm(p[:2] - pts[LL][:2]) >= min_center_dist for LL in pts):
                        pts[L] = p
                        break
                else:
                    # Fallback (very unlikely): spread along Y deterministically within bounds
                    base_x = float(rng.uniform(x_min + 0.05, x_max - 0.05))
                    pts = {
                        "A": np.array([base_x, 0.18, z], dtype=float),
                        "B": np.array([base_x, 0.0, z], dtype=float),
                        "C": np.array([base_x, -0.18, z], dtype=float),
                    }
                    break
            return pts

        init_pos = sample_positions()

        # Random upright yaw for each cube to randomize semantics/orientation
        for cube, letter in [(self.cube_A, "A"), (self.cube_B, "B"), (self.cube_C, "C")]:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(position=init_pos[letter], quat=q.elements)

        self._mojo.physics.forward()

        # Keep goal quats for metric reporting (unchanged goal definition)
        # Base layer: B and C goal orientations
        for cube, target_letter in [(self.cube_B, "B"), (self.cube_C, "C")]:
            face_idx = cube._labels.index(target_letter)
            q_goal = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR)
            self._goal_quats[target_letter] = q_goal
        # Top layer: A goal orientation
        face_idx_A = self.cube_A._labels.index("A")
        qA_goal = self.cube_A.get_orientation_for_face_and_glyph(face_idx=face_idx_A, face_dir_world=self._FACE_DIR)
        self._goal_quats["A"] = qA_goal

        # 4) Initialize subgoals to False
        for idx in range(8):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Identity-free positional/semantic check ---
        # Sort cubes by Z asc to identify base (lower 2) vs top (upper 1).
        # Within base, sort by Y desc so slot[0]=left target "B", slot[1]=right target "C".
        # Top slot target is "A".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_by_z = sorted(cubes_with_pos, key=lambda cp: cp[1][2])
        base_cp = cubes_by_z[0:2]
        top_cp = cubes_by_z[2:3]
        base_sorted = sorted(base_cp, key=lambda cp: cp[1][1], reverse=True)

        slot_left_cube, slot_left_pos = base_sorted[0]
        slot_right_cube, slot_right_pos = base_sorted[1]
        slot_top_cube, slot_top_pos = top_cp[0]

        # Use slot-based aliases for the spatial checks (preserve original names).
        posB = slot_left_pos
        posC = slot_right_pos
        posA = slot_top_pos

        # --- Spatial / arrangement checks ---
        # On-table (base cubes) (collision can be flaky right after reset; use Z-height as primary signal)
        on_table_B = abs(posB[2] - self._Z_TABLE) <= 0.03
        on_table_C = abs(posC[2] - self._Z_TABLE) <= 0.03

        # Left-of along Y (+Y is left), with min separation
        left_of = (posB[1] - posC[1]) >= 0.035

        # Aligned along X (front/back)
        aligned_x = abs(posB[0] - posC[0]) <= 0.03

        # A above both (stacked)
        above_B = (posA[2] - posB[2]) >= 0.03
        above_C = (posA[2] - posC[2]) >= 0.03

        # A centered between B and C along Y
        mid_y = 0.5 * (posB[1] + posC[1])
        between_y = abs(posA[1] - mid_y) <= 0.03

        # Touching (collision) for support
        touching_B = slot_top_cube.is_colliding(slot_left_cube)
        touching_C = slot_top_cube.is_colliding(slot_right_cube)

        positions_ok = bool(
            on_table_B
            and on_table_C
            and left_of
            and aligned_x
            and above_B
            and above_C
            and between_y
            and touching_B
            and touching_C
        )

        # --- SEMANTIC VERIFICATION (REQUIRED for SemanticCube) ---
        # Verify the cube occupying each slot has the slot's target letter readable toward -X.
        slot_pairs = [
            (slot_left_cube, "B"),
            (slot_right_cube, "C"),
            (slot_top_cube, "A"),
        ]
        semantic_ok = all(c.is_face_readable(t, face_dir=self._FACE_DIR) for c, t in slot_pairs)

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort; monotonic flags handled by metric logger) ---
        # 1-4: base built
        self._metric_stage(1, bool(on_table_B and left_of))
        self._metric_stage(3, bool(on_table_C and left_of))
        # 5-7: top placed and aligned
        self._metric_stage(6, bool(above_B and above_C and between_y and touching_B and touching_C))
        # 6: A face forward (top-slot cube must show "A" readable toward -X)
        self._metric_stage(5, bool(slot_top_cube.is_face_readable("A", face_dir=self._FACE_DIR)))
        # 8: release
        self._metric_stage(7, bool(released))

        # Overall success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize (MUST be last metric call before return) ---
        target_distance = {
            "A": float(np.linalg.norm(posA - self._goal_positions["A"])),
            "B": float(np.linalg.norm(posB - self._goal_positions["B"])),
            "C": float(np.linalg.norm(posC - self._goal_positions["C"])),
        }

        # Orientation error (degrees): report A (key requirement), plus B/C for completeness
        pose_error = {}
        for cube, letter in [(self.cube_A, "A"), (self.cube_B, "B"), (self.cube_C, "C")]:
            q_goal = self._goal_quats.get(letter, None)
            if q_goal is None:
                pose_error[f"{letter}_rot_deg"] = float("nan")
                continue
            q_cur = Quaternion(cube.body.get_quaternion())
            # Quaternion angular distance (0..pi)
            dq = (q_goal.conjugate * q_cur).normalised
            ang = 2.0 * np.arccos(np.clip(abs(dq.w), -1.0, 1.0))
            pose_error[f"{letter}_rot_deg"] = float(np.degrees(ang))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})