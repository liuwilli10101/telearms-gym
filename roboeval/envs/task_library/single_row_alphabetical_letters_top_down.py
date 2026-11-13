"""Arrange three semantic letter cubes in alphabetical order top-to-bottom in a single row."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class SingleRowAlphabeticalLettersTopDown(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic letter cubes in a single straight row so that the
    letters visible on the back faces (-X) read in **strictly ascending
    alphabetical order** from top to bottom along +X. Any letters work — the
    user is free to use whichever face of each cube is most convenient (each
    cube has 6 faces, one with its primary letter and 5 with random extras),
    so e.g. A → H → T is valid because A < H < T alphabetically.

    The `letters="ABC"` constructor argument controls which 3 SemanticCube
    assets are spawned (their "primary" letters). Default "ABC". This is just
    asset configuration — success only checks that the three back-face letters
    are ascending alphabetical.

    Sequential Sub-Goals:
        1. Ensure the robot is not holding any letter cube.
        2. Form a straight row of three cubes along +X with proper spacing.
        3. Make the top cube show some readable letter on its back face.
        4. (placed as part of the row)
        5. Make the middle cube show some readable letter on its back face.
        6. (placed as part of the row)
        7. Make the bottom cube show some readable letter on its back face.
        8. Verify the three back-face letters are strictly ascending alphabetical.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _FACE_DIR_WORLD = np.array([-1.0, 0.0, 0.0])
    _FACE_DIR_NAME = "back"

    _TABLE_Z_CUBE = 0.97
    _ROW_Y = 0.00
    _X_START = 0.48
    _X_SPACING = 0.06
    _ALIGN_Y_TOL = 0.03
    # NOTE: X-axis ordering = top-to-bottom in bimanual workspace; Y-axis = left-to-right.
    _ORDER_MIN_SEP_X = 0.03

    _ON_TABLE_Z_TOL = 0.02

    _NUM_SUBGOALS = 8

    def __init__(self, *, letters: str = "ABC", **kwargs):
        letters_up = "".join(letters).upper()
        if len(letters_up) != 3:
            raise ValueError(
                f"`letters` must be exactly 3 characters; got {letters!r} (len={len(letters_up)})"
            )
        if len(set(letters_up)) != 3:
            raise ValueError(f"`letters` must contain 3 DISTINCT characters; got {letters!r}")
        if not all(ch.isalpha() for ch in letters_up):
            raise ValueError(f"`letters` must be alphabetic; got {letters!r}")
        self._letters = letters_up                          # e.g. "DGZ"
        self._sorted_letters = "".join(sorted(letters_up))  # e.g. "DGZ" -> "DGZ"
        super().__init__(**kwargs)

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self._target_labels = list(self._letters)
        cubes_info = ensure_cubes(self._letters, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.letter_cubes = []
        for ch in self._target_labels:
            info = cubes_info[ch]
            xml_path = info.get("xml_path", info.get("xml"))
            labels = info.get("labels", None)
            cube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels)
            self.letter_cubes.append(cube)

        # Map letter -> cube body (handy for orientation checks / metrics).
        self.cube_by_letter: Dict[str, SemanticCube] = {
            ch: cube for ch, cube in zip(self._target_labels, self.letter_cubes)
        }
        self.cubes = self.letter_cubes

        self.distractor = ColoredCube(self._mojo)
        self.distractor.set_color("blue")

        # Goal slots are keyed by alphabetical rank (0 = top), so the per-letter
        # goal position is the slot for that letter's alphabetical position.
        sorted_letters = list(self._sorted_letters)  # e.g. ["D","G","Z"]
        self._goal_positions: Dict[str, np.ndarray] = {
            sorted_letters[i]: np.array(
                [self._X_START + i * self._X_SPACING, self._ROW_Y, self._TABLE_Z_CUBE]
            )
            for i in range(3)
        }
        self._goal_quats: Dict[str, np.ndarray] = {}

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

        # Pre-compute and store goal quaternions (used for pose_error metrics),
        # but do NOT place cubes in the goal state.
        self._goal_quats = {}
        for cube, ch in zip(self.letter_cubes, self._target_labels):
            face_idx = cube._labels.index(ch)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR_WORLD,
            )
            self._goal_quats[ch] = np.array(q_goal.elements, dtype=float)

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._TABLE_Z_CUBE

        def _rand_quat() -> np.ndarray:
            v = rng.normal(size=4)
            v = v / (np.linalg.norm(v) + 1e-12)
            return v.astype(float)

        def _sample_non_overlapping_positions(n: int, min_dist: float = 0.07, max_tries: int = 2000):
            pts = []
            for _ in range(max_tries):
                if len(pts) >= n:
                    break
                p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float)
                if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                    pts.append(p)
            if len(pts) < n:
                # Fallback: deterministic spread within bounds (still unsolved via orientation below)
                xs = np.linspace(x_min + 0.02, x_max - 0.02, n)
                ys = np.linspace(y_min + 0.02, y_max - 0.02, n)
                pts = [np.array([xs[i], ys[-(i + 1)], z], dtype=float) for i in range(n)]
            return pts

        # Randomize distractor pose within bounds too
        self.distractor.set_color("blue")
        self.distractor.set_pose(
            position=np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

        # Place letter cubes in random (unsolved) poses within bounds
        positions = _sample_non_overlapping_positions(3, min_dist=0.07)
        rng.shuffle(positions)  # extra randomization

        for cube, pos in zip(self.letter_cubes, positions):
            cube.set_pose(position=pos, quat=_rand_quat())

        # Ensure unsolved: force the alphabetically-first cube far from its own
        # goal slot (prevents accidental success even if random placement aligns).
        first_letter = self._sorted_letters[0]
        first_cube = self.cube_by_letter[first_letter]
        first_pos = first_cube.body.get_position()
        if float(np.linalg.norm(first_pos - self._goal_positions[first_letter])) <= 0.08:
            new_pos = _sample_non_overlapping_positions(1, min_dist=0.01)[0]
            first_cube.set_pose(position=new_pos, quat=_rand_quat())

        for _ in range(50):
            self._mojo.step()

        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _readable_back_letter(self, cube) -> "str | None":
        """Return whichever of the cube's six labels is currently readable on
        its back face (-X), or None if no label currently passes the check."""
        for label in cube._labels:
            if cube.is_face_readable(label, face_dir=self._FACE_DIR_NAME):
                return label
        return None

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "letter_cubes") or len(self.letter_cubes) != 3:
            return False

        face_dir_name = self._FACE_DIR_NAME

        # Sort cubes by X ascending so slot[0] = top (back along -X is the
        # SMALLEST X if camera is at high X; in this codebase A goes at the
        # smallest X end -> top of the row).
        cubes_with_pos = [(c, c.body.get_position()) for c in self.letter_cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][0])
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]

        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        on_table_ok = True
        z_ok = True
        for cube, pos in zip(slot_cubes, slot_positions):
            if not cube.is_colliding(self.table.table_top):
                on_table_ok = False
            if abs(float(pos[2]) - self._TABLE_Z_CUBE) > self._ON_TABLE_Z_TOL:
                z_ok = False

        # Row alignment: same Y across all three slots.
        ys = [float(p[1]) for p in slot_positions]
        row_alignment_ok = (
            (abs(ys[0] - ys[1]) <= self._ALIGN_Y_TOL)
            and (abs(ys[1] - ys[2]) <= self._ALIGN_Y_TOL)
        )

        # Strictly ordered along +X with required separation.
        xs = [float(p[0]) for p in slot_positions]
        order_ok = (xs[1] - xs[0] >= self._ORDER_MIN_SEP_X) and (
            xs[2] - xs[1] >= self._ORDER_MIN_SEP_X
        )

        # Pairwise center distances (avoid overlap).
        sep_ok = True
        if distance(slot_cubes[0].body, slot_cubes[1].body) < 0.045:
            sep_ok = False
        if distance(slot_cubes[1].body, slot_cubes[2].body) < 0.045:
            sep_ok = False
        if distance(slot_cubes[0].body, slot_cubes[2].body) < 0.045:
            sep_ok = False

        positions_ok = bool(on_table_ok and z_ok and row_alignment_ok and order_ok and sep_ok)

        # Generic semantic check: each cube's back face must show SOME readable
        # letter (any of its 6 labels), and the letters across the 3 slots must
        # be in strictly ascending alphabetical order top -> bottom.
        slot_letters = [self._readable_back_letter(c) for c in slot_cubes]
        all_have_readable_letter = all(l is not None for l in slot_letters)
        ascending_alphabetical = (
            all_have_readable_letter
            and slot_letters[0] < slot_letters[1] < slot_letters[2]
        )
        semantic_ok = bool(ascending_alphabetical)

        # --- Subgoal staging ---
        self._metric_stage(0, bool(released))
        # 1/2: top slot placed AND has a readable back-face letter
        self._metric_stage(1, bool(positions_ok))
        self._metric_stage(2, bool(slot_letters[0] is not None))
        # 3/4: middle slot
        self._metric_stage(3, bool(positions_ok))
        self._metric_stage(4, bool(slot_letters[1] is not None))
        # 5/6: bottom slot
        self._metric_stage(5, bool(positions_ok))
        self._metric_stage(6, bool(slot_letters[2] is not None))
        self._metric_stage(7, bool(positions_ok and semantic_ok))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics ---
        target_distance = {
            ch: float(np.linalg.norm(self.cube_by_letter[ch].body.get_position() - self._goal_positions[ch]))
            for ch in self._target_labels
        }

        pose_error: Dict[str, float] = {}
        for ch in self._target_labels:
            cur_q = np.array(self.cube_by_letter[ch].body.get_quaternion(), dtype=float)
            goal_q = np.array(
                self._goal_quats.get(ch, np.array([1.0, 0.0, 0.0, 0.0])), dtype=float
            )
            q_err = Quaternion(cur_q) * Quaternion(goal_q).inverse
            pose_error[ch] = float(q_err.angle)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
