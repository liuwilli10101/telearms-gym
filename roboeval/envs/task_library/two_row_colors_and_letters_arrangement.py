"""Arrange two parallel rows: front row colored cubes (R, G, B), back row letter cubes (A, B, C)."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class TwoRowColorsAndLettersArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange two parallel rows on the table (2 rows, 1 layer):
    - Row 1 (front): colored cubes only, ordered left-to-right as Red, Green, Blue.
    - Row 2 (behind): semantic letter cubes only, ordered left-to-right as A, B, C,
      with each letter readable on the visible BACK face (-X).

    Sequential Sub-Goals:
        1. Grasp a cube with the left gripper.
        2. Grasp a cube with the right gripper.
        3. Move the red cube into the left position of the colored row.
        4. Place the red cube on the table as part of the colored row.
        5. Move the green cube into the middle position of the colored row.
        6. Place the green cube on the table as part of the colored row.
        7. Move the blue cube into the right position of the colored row.
        8. Place the blue cube on the table as part of the colored row.
        9. Move the A cube into the left position of the letter row (behind the colored row).
        10. Place the A cube on the table as part of the letter row.
        11. Move the B cube into the middle position of the letter row.
        12. Place the B cube on the table as part of the letter row.
        13. Move the C cube into the right position of the letter row.
        14. Adjust the letter cubes so A/B/C are visible on their BACK faces.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Camera-reliable visible face for this task: BACK (-X)
    _FACE_DIR_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)
    _FACE_DIR_NAME = "back"

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create ColoredCubes (REQUIRED asset) ---
        self.color_red = ColoredCube(self._mojo)
        self.color_green = ColoredCube(self._mojo)
        self.color_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.color_red.set_color("red")
        self.color_green.set_color("green")
        self.color_blue.set_color("blue")

        self._colored_cubes = [self.color_red, self.color_green, self.color_blue]
        self._colored_names = ["color_red", "color_green", "color_blue"]
        self._colored_expected = ["red", "green", "blue"]

        # --- Create SemanticCubes (REQUIRED asset) using ensure_cubes() workflow ---
        self._target_labels = ["A", "B", "C"]
        cubes_info = ensure_cubes("ABC", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.letter_A = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["A"]["xml"]),
            labels=cubes_info["A"]["labels"],
        )
        self.letter_B = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["B"]["xml"]),
            labels=cubes_info["B"]["labels"],
        )
        self.letter_C = SemanticCube(
            self._mojo,
            xml_path=str(cubes_info["C"]["xml"]),
            labels=cubes_info["C"]["labels"],
        )

        self._letter_cubes = [self.letter_A, self.letter_B, self.letter_C]
        self._letter_names = ["letter_A", "letter_B", "letter_C"]

        # Unified list (used by metrics and "released" check)
        self.cubes = self._colored_cubes + self._letter_cubes
        self._cube_names = self._colored_names + self._letter_names

        # Predefine goal positions for consistent checking/metrics
        # NOTE: "Left-to-right" in camera view corresponds to +Y (left) -> -Y (right).
        self._z_table = 0.97
        self._row_spacing_y = 0.07  # >= 0.05m minimum spacing
        self._row_start_y = self._row_spacing_y  # for 3 cubes: +s, 0, -s

        # Row 1 (colored) placed "in front", Row 2 (letters) placed "behind" along +X
        self._x_colored = 0.55
        self._x_letters = 0.63  # behind colored row

        self._goal_pos_colored = [
            np.array([self._x_colored, self._row_start_y - 0 * self._row_spacing_y, self._z_table], dtype=float),
            np.array([self._x_colored, self._row_start_y - 1 * self._row_spacing_y, self._z_table], dtype=float),
            np.array([self._x_colored, self._row_start_y - 2 * self._row_spacing_y, self._z_table], dtype=float),
        ]
        self._goal_pos_letters = [
            np.array([self._x_letters, self._row_start_y - 0 * self._row_spacing_y, self._z_table], dtype=float),
            np.array([self._x_letters, self._row_start_y - 1 * self._row_spacing_y, self._z_table], dtype=float),
            np.array([self._x_letters, self._row_start_y - 2 * self._row_spacing_y, self._z_table], dtype=float),
        ]

        # Store desired quaternions for letters (computed on reset since cube types are fixed here)
        self._goal_quat_letters = [None, None, None]

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

        # 2. Deterministic RNG (required pattern)
        _rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset() for ColoredCubes
        self.color_red.set_color("red")
        self.color_green.set_color("green")
        self.color_blue.set_color("blue")

        # 4. Place objects in RANDOM (unsolved) positions within workspace bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # To guarantee _success() == False after reset, place LETTERS in FRONT of COLORED (violates behind_ok),
        # and randomize letter orientations (likely violates semantic_ok).
        z = float(self._z_table)

        def _rand_quat() -> np.ndarray:
            axis = _rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            angle = float(_rng.uniform(0.0, 2.0 * np.pi))
            return Quaternion(axis=axis, angle=angle).elements

        # Keep a safe minimum separation to avoid overlap
        min_sep = 0.07

        # Sample Y positions without getting too close (simple rejection sampling)
        def _sample_positions(n: int, x_range, y_range):
            pts = []
            for _ in range(2000):
                if len(pts) >= n:
                    break
                x = float(_rng.uniform(*x_range))
                y = float(_rng.uniform(*y_range))
                p = np.array([x, y, z], dtype=float)
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_sep for q in pts):
                    pts.append(p)
            if len(pts) < n:
                # Fallback: deterministic spread (still within bounds)
                xs = np.linspace(x_range[0], x_range[1], n)
                ys = np.linspace(y_range[0], y_range[1], n)
                pts = [np.array([float(xs[i]), float(ys[i]), z], dtype=float) for i in range(n)]
            return pts

        # Colored cubes: place more towards back (larger X)
        colored_positions = _sample_positions(
            n=3,
            x_range=(0.56, 0.70),
            y_range=(-0.22, 0.22),
        )

        # Letter cubes: place more towards front (smaller X) to violate behind_ok
        letter_positions = _sample_positions(
            n=3,
            x_range=(0.40, 0.52),
            y_range=(-0.22, 0.22),
        )

        # Place colored cubes (identity quat)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for cube, pos in zip(self._colored_cubes, colored_positions):
            cube.set_pose(position=pos, quat=q_identity)

        # Place letter cubes with random orientations
        for cube, pos in zip(self._letter_cubes, letter_positions):
            cube.set_pose(position=pos, quat=_rand_quat())

        # Precompute the "goal" quaternions (for pose_error metric) but do not set them
        for i, (cube, target_letter) in enumerate(zip(self._letter_cubes, self._target_labels)):
            face_idx = cube._labels.index(target_letter)
            q_goal = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR_WORLD)
            self._goal_quat_letters[i] = q_goal

        # Physics settle
        for _ in range(100):
            self._mojo.step()

        # 5. Initialize subgoals to False
        self._num_subgoals = 14
        for idx in range(self._num_subgoals):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # _rng = np.random.default_rng(self.seed)
        #
        # # 3. REQUIRED: set_color() in _on_reset() for ColoredCubes
        # self.color_red.set_color("red")
        # self.color_green.set_color("green")
        # self.color_blue.set_color("blue")
        #
        # # Colored row (keep stable identity quaternion)
        # q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # for cube, pos in zip(self._colored_cubes, self._goal_pos_colored):
        #     cube.set_pose(position=pos, quat=q_identity)
        #
        # # Physics settle
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Letter row: orient so A/B/C are readable on BACK face (-X)
        # for i, (cube, target_letter, pos) in enumerate(
        #     zip(self._letter_cubes, self._target_labels, self._goal_pos_letters)
        # ):
        #     face_idx = cube._labels.index(target_letter)
        #     q = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR_WORLD)
        #     self._goal_quat_letters[i] = q
        #     cube.set_pose(position=pos, quat=q.elements)
        #
        # # Physics settle
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # 5. Initialize subgoals to False
        # self._num_subgoals = 14
        # for idx in range(self._num_subgoals):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Parameters from task spec (tolerances) ---
        height_tol = 0.02
        row_x_tol = 0.03
        behind_min_sep = 0.05
        y_align_tol = 0.03

        # --- Helper: on-table check (collision + height) ---
        def _on_table(obj) -> bool:
            z = float(obj.body.get_position()[2])
            return (abs(z - self._z_table) <= height_tol) and obj.is_colliding(self.table.table_top)

        # --- Verify colored cube colors (semantic check for ColoredCube) ---
        colors_ok = True
        for cube, expected in zip(self._colored_cubes, self._colored_expected):
            if cube.get_color() != expected:
                colors_ok = False
                break

        # --- Verify all cubes are on table (1 layer) ---
        on_table_ok = all(_on_table(c) for c in self.cubes)

        # --- Row alignment check: each row should have near-constant X; ordered by Y left->right ---
        def _row_ok(row_cubes, expected_order_names) -> Dict:
            xs = [float(c.body.get_position()[0]) for c in row_cubes]
            x_ok = (max(xs) - min(xs)) <= row_x_tol

            ys = [float(c.body.get_position()[1]) for c in row_cubes]
            sort_idx = list(np.argsort(ys)[::-1])  # left-to-right: +Y to -Y

            ordered_names = []
            for j in sort_idx:
                if row_cubes[j] in self._colored_cubes:
                    ordered_names.append(self._colored_names[self._colored_cubes.index(row_cubes[j])])
                else:
                    ordered_names.append(self._letter_names[self._letter_cubes.index(row_cubes[j])])
            order_ok = ordered_names == expected_order_names

            return {"ok": bool(x_ok and order_ok), "x_ok": bool(x_ok), "order_ok": bool(order_ok), "sort_idx": sort_idx}

        colored_row_check = _row_ok(
            self._colored_cubes,
            expected_order_names=["color_red", "color_green", "color_blue"],
        )
        # Letter row: order is verified via slot-based semantic readability below
        # (sorted_letter_cubes_for_semantic). Here we only check geometric row-ness
        # (X alignment) without binding cube identity to slot.
        letter_xs = [float(c.body.get_position()[0]) for c in self._letter_cubes]
        letter_x_ok = (max(letter_xs) - min(letter_xs)) <= row_x_tol
        letter_row_check = {"ok": bool(letter_x_ok), "x_ok": bool(letter_x_ok), "order_ok": True, "sort_idx": []}

        # --- Rows relative: ALL letter cubes must be behind ALL color cubes along +X ---
        # Stricter than mean-based check: guarantees no single outlier can fool the condition.
        letter_x_values = [float(c.body.get_position()[0]) for c in self._letter_cubes]
        color_x_values = [float(c.body.get_position()[0]) for c in self._colored_cubes]
        behind_ok = (min(letter_x_values) - max(color_x_values)) >= behind_min_sep

        # --- Column-wise alignment (A behind Red, B behind Green, C behind Blue) along Y ---
        # Sort both rows by Y-position (descending) before pairing to avoid index-order dependency.
        sorted_colored = sorted(self._colored_cubes, key=lambda c: c.body.get_position()[1], reverse=True)
        sorted_letters = sorted(self._letter_cubes, key=lambda c: c.body.get_position()[1], reverse=True)
        y_align_ok = True
        for color_cube, letter_cube in zip(sorted_colored, sorted_letters):
            y_c = float(color_cube.body.get_position()[1])
            y_l = float(letter_cube.body.get_position()[1])
            if abs(y_c - y_l) > y_align_tol:
                y_align_ok = False
                break

        # --- Semantic verification (REQUIRED for SemanticCube): labels readable on BACK face ---
        # Sort letter cubes by Y (descending) to match left-to-right label order before semantic check.
        sorted_letter_cubes_for_semantic = sorted(
            self._letter_cubes, key=lambda c: c.body.get_position()[1], reverse=True
        )
        semantic_ok = True
        for cube, target_letter in zip(sorted_letter_cubes_for_semantic, self._target_labels):
            if not cube.is_face_readable(target_letter, face_dir=self._FACE_DIR_NAME):
                semantic_ok = False
                break

        # --- Released check (REQUIRED pattern) ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(
            colors_ok
            and on_table_ok
            and colored_row_check["ok"]
            and letter_row_check["ok"]
            and behind_ok
            and y_align_ok
        )

        # --- Stage metrics (coarse mapping to the provided subgoals) ---
        if positions_ok:
            for idx in range(2, 7):
                self._metric_stage(idx, True)
            for idx in range(7, 13):
                self._metric_stage(idx, True)
        if semantic_ok:
            self._metric_stage(13, True)

        # Set success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics: target distances and pose errors ---
        target_distance = {}
        pose_error = {}

        for name, cube, goal_pos in zip(self._colored_names, self._colored_cubes, self._goal_pos_colored):
            cur = np.array(cube.body.get_position(), dtype=float)
            target_distance[name] = float(np.linalg.norm(cur - goal_pos))

        for i, (name, cube, goal_pos) in enumerate(zip(self._letter_names, self._letter_cubes, self._goal_pos_letters)):
            cur = np.array(cube.body.get_position(), dtype=float)
            target_distance[name] = float(np.linalg.norm(cur - goal_pos))

            try:
                q_cur = Quaternion(cube.body.get_quaternion())
                q_goal = self._goal_quat_letters[i] if self._goal_quat_letters[i] is not None else Quaternion()
                q_err = q_goal.conjugate * q_cur
                w = float(np.clip(abs(q_err.w), -1.0, 1.0))
                pose_error[name] = float(2.0 * np.arccos(w))
            except Exception:
                pose_error[name] = float("nan")

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
