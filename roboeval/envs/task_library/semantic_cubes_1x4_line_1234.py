"""Arrange four SemanticCubes in a 1x4 line to read '1234' left-to-right with back-face digits."""
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


class SemanticCubes1x4Line1234(RoboEvalEnv, MetricRolloutEval):
    """Arrange four SemanticCubes into a single-layer 1x4 straight line on the table to read "1234"
    from left to right (camera view), with each cube rotated so its BACK face (-X, visible to
    the camera) shows the correct digit.

    Sequential Sub-Goals:
        1. Place cube "1" in the leftmost slot (highest +Y).
        2. Place cube "2" in the next slot.
        3. Place cube "3" in the next slot.
        4. Place cube "4" in the rightmost slot (lowest -Y).
        5. Ensure final arrangement is correct and all cubes are released.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics = {}

    _FACE_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)
    _BASE_Z = 0.97
    _X_CENTER = 0.55                    # reset-only anchor (not used for success)
    _POS_TOL = 0.015
    _Z_TOL = 0.02
    _SLOT_YS = [0.09, 0.03, -0.03, -0.09]
    _TARGET_LABELS = ["1", "2", "3", "4"]
    _NUM_SUBGOALS = 5

    # Structural tolerances for translation-invariant success check.
    _X_ALIGN_TOL = 0.03       # ptp(x) across the row (metres)
    _GAP_STD_TOL = 0.015      # std of adjacent Y gaps (metres)
    _MIN_GAP = 0.035          # minimum Y gap between adjacent cubes
    _UPRIGHT_COS_TOL = 0.98

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        cubes_info = ensure_cubes("".join(self._TARGET_LABELS), universe="0123456789")

        self.cubes = []
        for digit_label in self._TARGET_LABELS:
            info = cubes_info[digit_label]
            xml_path = info.get("xml", info.get("xml_path"))
            labels = info.get("labels")
            cube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=labels)
            self.cubes.append(cube)

        self._goal_positions = [
            np.array([self._X_CENTER, y, self._BASE_Z], dtype=float) for y in self._SLOT_YS
        ]

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

        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z_height = float(self._BASE_Z)

        min_intercube_dist = 0.055
        min_goal_clearance = max(0.05, 3.0 * float(self._POS_TOL))
        placed_xy = []

        for cube_idx, cube in enumerate(self.cubes):
            goal_xy = self._goal_positions[cube_idx][:2]

            chosen_x = None
            chosen_y = None
            for _ in range(1000):
                cand_x = float(rng.uniform(x_low, x_high))
                cand_y = float(rng.uniform(y_low, y_high))
                cand_xy = np.array([cand_x, cand_y], dtype=float)

                if float(np.linalg.norm(cand_xy - goal_xy)) < min_goal_clearance:
                    continue

                ok = True
                for prev_xy in placed_xy:
                    if float(np.linalg.norm(cand_xy - prev_xy)) < min_intercube_dist:
                        ok = False
                        break
                if not ok:
                    continue

                chosen_x, chosen_y = cand_x, cand_y
                placed_xy.append(cand_xy)
                break

            if chosen_x is None or chosen_y is None:
                chosen_x = float(np.clip(self._X_CENTER + 0.12 * (cube_idx - 1.5), x_low, x_high))
                chosen_y = float(np.clip(0.22 - 0.12 * cube_idx, y_low, y_high))
                placed_xy.append(np.array([chosen_x, chosen_y], dtype=float))

            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            quat_obj = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(
                position=np.array([chosen_x, chosen_y, z_height], dtype=float),
                quat=quat_obj.elements,
            )

        for _ in range(50):
            self._mojo.step()

        for stage_idx in range(self._NUM_SUBGOALS):
            self._metric_stage(stage_idx, False)

        # --- GOAL STATE ---
        # _ = np.random.default_rng(self.seed)
        # for cube, target_digit, pos in zip(self.cubes, self._TARGET_LABELS, self._goal_positions):
        #     face_idx = cube._labels.index(target_digit)
        #     q = cube.get_orientation_for_face_and_glyph(
        #         face_idx=face_idx,
        #         face_dir_world=self._FACE_DIR,
        #     )
        #     cube.set_pose(position=pos, quat=q.elements)
        # for _ in range(50):
        #     self._mojo.step()
        # for idx in range(self._NUM_SUBGOALS):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super(type(self), self)._on_step()
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)

        # Translation-invariant line check: sort cubes by Y-descending so the highest-Y
        # cube is "leftmost" (should be '1'); index-to-digit mapping is anchor-free.
        sorted_cubes = sorted(self.cubes, key=lambda c: c.body.get_position()[1], reverse=True)
        positions = np.array([c.body.get_position() for c in sorted_cubes], dtype=float)
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]

        # --- structural line geometry ---
        x_ptp = float(np.ptp(x_vals))
        x_align_ok = x_ptp <= self._X_ALIGN_TOL

        y_diffs = np.diff(y_vals)       # negative with reverse=True sort
        gap_mags = -y_diffs
        min_gap_ok = bool(np.all(gap_mags >= self._MIN_GAP))
        gap_std_ok = bool(float(np.std(gap_mags)) <= self._GAP_STD_TOL)

        # --- per-cube pose / semantic ---
        per_cube_pose_ok = []
        per_cube_sem_ok = []
        for cube, target_digit in zip(sorted_cubes, self._TARGET_LABELS):
            quat_obj = Quaternion(cube.body.get_quaternion())
            up_vec = quat_obj.rotate(world_up)
            upright_ok = float(np.dot(up_vec, world_up)) >= self._UPRIGHT_COS_TOL
            per_cube_pose_ok.append(bool(upright_ok))
            per_cube_sem_ok.append(bool(cube.is_face_readable(target_digit, face_dir=self._FACE_DIR)))

        # --- on-table via contact (no absolute Z) ---
        on_table_ok = all(c.is_colliding(self.table.table_top) for c in sorted_cubes)

        positions_ok = bool(
            x_align_ok and min_gap_ok and gap_std_ok and on_table_ok and all(per_cube_pose_ok)
        )
        semantic_ok = bool(all(per_cube_sem_ok))

        released = not any(
            self.robot.is_gripper_holding_object(obj, side)
            for obj in self.cubes
            for side in self.robot.grippers
        )

        can_write_stages = hasattr(self, "_stage_flags") and bool(getattr(self, "_stage_flags", {}))
        if can_write_stages:
            for stage_idx in range(4):
                self._metric_stage(
                    stage_idx,
                    bool(positions_ok and per_cube_sem_ok[stage_idx] and per_cube_pose_ok[stage_idx]),
                )
            self._metric_stage(4, bool(positions_ok and semantic_ok and released))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- metrics: centroid-anchored line for reporting only ---
        centroid_xy = positions[:, :2].mean(axis=0)
        anchored_goals = [
            np.array([centroid_xy[0], centroid_xy[1] + dy, self._BASE_Z], dtype=float)
            for dy in self._SLOT_YS
        ]

        target_distance: Dict[str, float] = {}
        pose_error: Dict[str, float] = {}

        face_normals_local = {
            0: np.array([1.0, 0.0, 0.0], dtype=float),
            1: np.array([-1.0, 0.0, 0.0], dtype=float),
            2: np.array([0.0, 1.0, 0.0], dtype=float),
            3: np.array([0.0, -1.0, 0.0], dtype=float),
            4: np.array([0.0, 0.0, 1.0], dtype=float),
            5: np.array([0.0, 0.0, -1.0], dtype=float),
        }

        for cube_idx, (cube, target_digit) in enumerate(zip(sorted_cubes, self._TARGET_LABELS)):
            metric_name = f"cube_{cube_idx+1:02d}"
            pos = cube.body.get_position()
            quat_obj = Quaternion(cube.body.get_quaternion())

            goal_pos = anchored_goals[cube_idx]
            target_distance[metric_name] = float(np.linalg.norm(pos[:2] - goal_pos[:2]))

            face_idx = None
            if hasattr(cube, "_labels") and isinstance(getattr(cube, "_labels", None), (list, tuple)):
                try:
                    face_idx = cube._labels.index(target_digit)
                except Exception:
                    face_idx = None

            if face_idx is None:
                n_local = np.array([-1.0, 0.0, 0.0], dtype=float)
            else:
                n_local = face_normals_local.get(int(face_idx), np.array([-1.0, 0.0, 0.0], dtype=float))

            n_world = quat_obj.rotate(n_local)
            denom = float(np.linalg.norm(n_world)) + 1e-9
            dotv = float(np.clip(float(np.dot(n_world, self._FACE_DIR)) / denom, -1.0, 1.0))
            pose_error[metric_name] = float(np.arccos(dotv))

        target_distance["x_ptp"] = x_ptp
        target_distance["gap_std"] = float(np.std(gap_mags)) if gap_mags.size else 0.0

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
