"""Arrange five SemanticCubes in a 1x5 line to spell HELLO with back-face letters readable."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class HelloSemanticCubes1x5Line(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange five SemanticCubes into a single-layer 1x5 straight line on the table to spell "HELLO"
    from left to right, with each cube rotated so its BACK face (-X, visible to the camera) shows
    the correct letter.

    Sequential Sub-Goals:
        1. Place H in the leftmost slot (and readable on back face)
        2. Place E next to H
        3. Place first L next to E
        4. Place second L next to first L
        5. Place O in the rightmost slot
        6. All cubes correct and released (no gripper holding any cube)
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False

    _FACE_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)

    _WORD = "HELLO"
    _N_CUBES = 5

    _CUBE_BASE_Z = 0.97
    _ORIGIN_XYZ = np.array([0.55, 0.0, _CUBE_BASE_Z], dtype=float)  # reset-only anchor for non-solved start
    _POS_TOL = 0.015
    _SLOT_SPACING = 0.06

    # Structural tolerances for translation-invariant success check.
    _X_ALIGN_TOL = 0.03   # ptp(x) across the row must be <= this (metres)
    _GAP_STD_TOL = 0.015  # std of adjacent Y gaps (metres); enforces ~uniform spacing
    _MIN_GAP = 0.035      # adjacent cubes must be at least this far apart in Y
    _UPRIGHT_COS_TOL = 0.98

    _success_check = True
    _final_metrics: Dict = {}

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        cubes_info = ensure_cubes(self._WORD, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self._target_labels = list(self._WORD)

        self.cubes = []
        for ch in self._target_labels:
            info = cubes_info[ch]
            xml_path = info.get("xml_path", info.get("xml"))
            labels = info.get("labels", info.get("label"))
            cube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels)
            self.cubes.append(cube)

        half = (self._N_CUBES - 1) / 2.0
        self._slot_positions = []
        for i in range(self._N_CUBES):
            y = (half - i) * self._SLOT_SPACING
            pos = self._ORIGIN_XYZ + np.array([0.0, y, 0.0], dtype=float)
            self._slot_positions.append(pos)

        self._target_quats_wxyz = []
        for cube, target_letter in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(target_letter)
            q = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR)
            self._target_quats_wxyz.append(np.array(q.elements, dtype=float))

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
        rng = np.random.default_rng(self.seed)

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Randomize cube poses within workspace bounds, ensuring an unsolved start.
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_BASE_Z)

        # Sample positions with simple rejection to reduce overlap and avoid goal slots.
        min_sep = 0.055  # conservative spacing to reduce initial interpenetration
        away_from_goal = max(self._POS_TOL * 2.5, 0.04)

        sampled_positions = []
        for i in range(self._N_CUBES):
            goal_pos_i = np.array(self._slot_positions[i], dtype=float)
            for _attempt in range(200):
                p = np.array(
                    [
                        float(rng.uniform(x_min, x_max)),
                        float(rng.uniform(y_min, y_max)),
                        z,
                    ],
                    dtype=float,
                )

                # Keep away from this cube's goal slot so at least one condition fails.
                if float(np.linalg.norm(p - goal_pos_i)) <= away_from_goal:
                    continue

                ok = True
                for pj in sampled_positions:
                    if float(np.linalg.norm(p - pj)) < min_sep:
                        ok = False
                        break
                if ok:
                    sampled_positions.append(p)
                    break
            else:
                # Fallback: place deterministically far from goal line if sampling fails
                p = np.array([x_min + 0.02 * i, y_max - 0.02 * i, z], dtype=float)
                sampled_positions.append(p)

        # Random yaw (upright), and for at least one cube force a non-target yaw to break semantic success.
        wrong_idx = int(rng.integers(0, self._N_CUBES))
        for i, (cube, pos) in enumerate(zip(self.cubes, sampled_positions)):
            yaw = float(rng.uniform(-np.pi, np.pi))
            if i == wrong_idx:
                # Ensure it's not accidentally at/near the target orientation.
                yaw += float(np.pi / 2.0)
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)  # wxyz
            cube.set_pose(position=np.array(pos, dtype=float), quat=np.array(q.elements, dtype=float))

        for _ in range(50):
            self._mojo.step()

        for idx in range(6):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # _ = rng
        # for cube, pos, quat_wxyz in zip(self.cubes, self._slot_positions, self._target_quats_wxyz):
        #     cube.set_pose(position=np.array(pos, dtype=float), quat=np.array(quat_wxyz, dtype=float))
        # for _ in range(50):
        #     self._mojo.step()
        # for idx in range(6):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _quat_angle_error_rad(q_current_wxyz: np.ndarray, q_target_wxyz: np.ndarray) -> float:
        q1 = np.array(q_current_wxyz, dtype=float)
        q2 = np.array(q_target_wxyz, dtype=float)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        q2 = q2 / (np.linalg.norm(q2) + 1e-12)
        w2, x2, y2, z2 = q2
        q2_inv = np.array([w2, -x2, -y2, -z2], dtype=float)
        w1, x1, y1, z1 = q1
        wI, xI, yI, zI = q2_inv
        q_err = np.array(
            [
                wI * w1 - xI * x1 - yI * y1 - zI * z1,
                wI * x1 + xI * w1 + yI * z1 - zI * y1,
                wI * y1 - xI * z1 + yI * w1 + zI * x1,
                wI * z1 + xI * y1 - yI * x1 + zI * w1,
            ],
            dtype=float,
        )
        qw = float(np.clip(abs(q_err[0]), -1.0, 1.0))
        return float(2.0 * np.arccos(qw))

    @staticmethod
    def _quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in q_wxyz]
        n = (w * w + x * x + y * y + z * z) ** 0.5 + 1e-12
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if len(getattr(self, "cubes", [])) != self._N_CUBES:
            return False

        # Translation-invariant line check: sort cubes by Y-descending so that the
        # highest-Y cube is "leftmost" (should be 'H'); index-to-letter mapping then
        # becomes independent of any world-frame anchor.
        sorted_cubes = sorted(self.cubes, key=lambda c: c.body.get_position()[1], reverse=True)
        positions = np.array([c.body.get_position() for c in sorted_cubes], dtype=float)
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]

        # --- structural line geometry ---
        x_ptp = float(np.ptp(x_vals))
        x_align_ok = x_ptp <= self._X_ALIGN_TOL

        y_diffs = np.diff(y_vals)  # with reverse=True sort, these are negative
        gap_mags = -y_diffs        # positive gap magnitudes between adjacent cubes
        min_gap_ok = bool(np.all(gap_mags >= self._MIN_GAP))
        gap_std_ok = bool(float(np.std(gap_mags)) <= self._GAP_STD_TOL)

        # --- orientation: each cube upright (local +Z ~= world +Z) ---
        upright_ok = True
        for cube in sorted_cubes:
            q = np.array(cube.body.get_quaternion(), dtype=float)
            R = self._quat_to_rotmat(q)
            local_z_in_world = R[:, 2]
            if float(abs(np.dot(local_z_in_world, np.array([0.0, 0.0, 1.0])))) < self._UPRIGHT_COS_TOL:
                upright_ok = False
                break

        # --- on-table via contact (no absolute Z) ---
        on_table_ok = all(c.is_colliding(self.table.table_top) for c in sorted_cubes)

        positions_ok = bool(x_align_ok and min_gap_ok and gap_std_ok and upright_ok and on_table_ok)

        # --- semantic: each cube's target letter reads toward -X, in order H,E,L,L,O ---
        semantic_ok = True
        per_letter_ok = []
        for cube, target_letter in zip(sorted_cubes, self._target_labels):
            ok_i = bool(cube.is_face_readable(target_letter, face_dir=self._FACE_DIR))
            per_letter_ok.append(ok_i)
            if not ok_i:
                semantic_ok = False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- per-cube stage bookkeeping: a letter "stage" fires when that letter is
        #     readable in the correct Y-sorted slot AND the row itself is geometrically valid.
        for i in range(5):
            self._metric_stage(i, bool(per_letter_ok[i] and positions_ok))
        all_done = bool(positions_ok and semantic_ok and released)
        self._metric_stage(5, all_done)

        self._success_check = all_done

        # --- metrics: centroid-anchored goal line for reporting only ---
        centroid_xy = positions[:, :2].mean(axis=0)
        half = (self._N_CUBES - 1) / 2.0
        anchored_goals = []
        for i in range(self._N_CUBES):
            y = (half - i) * self._SLOT_SPACING
            anchored_goals.append(
                np.array([centroid_xy[0], centroid_xy[1] + y, self._CUBE_BASE_Z], dtype=float)
            )

        dists = [
            float(np.linalg.norm(np.array(c.body.get_position(), dtype=float) - g))
            for c, g in zip(sorted_cubes, anchored_goals)
        ]

        ang_errs = []
        for cube, q_goal in zip(sorted_cubes, self._target_quats_wxyz):
            q_cur = np.array(cube.body.get_quaternion(), dtype=float)
            ang_errs.append(self._quat_angle_error_rad(q_cur, q_goal))

        target_distance = {f"cube_{i+1:02d}_m": float(dists[i]) for i in range(self._N_CUBES)}
        target_distance.update(
            {
                "mean_m": float(np.mean(dists)) if dists else 0.0,
                "max_m": float(np.max(dists)) if dists else 0.0,
                "tolerance_m": float(self._POS_TOL),
                "x_ptp": x_ptp,
                "gap_std": float(np.std(gap_mags)) if gap_mags.size else 0.0,
            }
        )

        pose_error = {f"cube_{i+1:02d}_rad": float(ang_errs[i]) for i in range(self._N_CUBES)}
        pose_error.update(
            {
                "mean_rad": float(np.mean(ang_errs)) if ang_errs else 0.0,
                "max_rad": float(np.max(ang_errs)) if ang_errs else 0.0,
            }
        )

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
