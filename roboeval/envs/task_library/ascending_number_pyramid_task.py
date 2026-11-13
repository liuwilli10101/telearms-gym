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


class AscendingNumberPyramidTask(RoboEvalEnv, MetricRolloutEval):
    """
    Build a 3-tier number pyramid using numbered semantic cubes so the numbers increase
    as you go downward. Final structure:
      - Top tier:        1
      - Middle tier:     2   3
      - Bottom tier:     4   5   6   (left-to-right in camera view: +Y to -Y)

    All cubes must be oriented so their correct number is readable on the visible BACK face (-X).

    Sequential Sub-Goals:
        1. Place cubes 4, 5, 6 as an aligned bottom row (left-to-right: 4, then 5, then 6).
        2. Stack cube 2 on cube 4; stack cube 3 on cube 5.
        3. Stack cube 1 on cube 2.
        4. Ensure all cubes show their own number on the BACK (-X) face.
        5. Ensure robot releases all cubes.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _FACE_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)

    _CUBE_SIZE = 0.04
    _BASE_Z = 0.97
    _STACK_DZ = 0.04

    _LEFT_OF_MIN_DY = 0.03
    _ROW_ALIGN_TOL_X = 0.03
    _ON_TABLE_TOL_Z = 0.02
    _ON_TOL_Z = 0.02
    _ON_TOL_XY = 0.02

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self._target_labels = list("123456")
        cubes_info = ensure_cubes("".join(self._target_labels), universe="0123456789")

        self.cubes = []
        for ch in self._target_labels:
            info = cubes_info[ch]
            cube = SemanticCube(
                self._mojo,
                xml_path=str(info["xml"]),
                labels=info["labels"],
            )
            self.cubes.append(cube)

        self.cube_1, self.cube_2, self.cube_3, self.cube_4, self.cube_5, self.cube_6 = self.cubes

        x0 = 0.55
        y_left, y_mid, y_right = 0.06, 0.00, -0.06

        self._goal_pos = {
            "cube_4": np.array([x0, y_left, self._BASE_Z], dtype=float),
            "cube_5": np.array([x0, y_mid, self._BASE_Z], dtype=float),
            "cube_6": np.array([x0, y_right, self._BASE_Z], dtype=float),
            "cube_2": np.array([x0, y_left, self._BASE_Z + self._STACK_DZ], dtype=float),
            "cube_3": np.array([x0, y_mid, self._BASE_Z + self._STACK_DZ], dtype=float),
            "cube_1": np.array([x0, y_left, self._BASE_Z + 2 * self._STACK_DZ], dtype=float),
        }

        self._goal_quat = {}
        for cube, label in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(label)
            q = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            self._goal_quat[label] = q

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

        # Random unsolved placement: put ALL cubes on the table (base Z), scattered in-bounds,
        # and with random yaw so the goal orientation/stacking is not already satisfied.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._BASE_Z)

        min_sep = 0.06  # > cube size to reduce interpenetration/accidental contacts
        pts = []
        max_tries = 2000
        for _ in range(max_tries):
            if len(pts) >= len(self.cubes):
                break
            p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if all(np.linalg.norm(p - q) >= min_sep for q in pts):
                pts.append(p)
        if len(pts) < len(self.cubes):
            # Fallback: allow smaller separation if space is tight
            pts = []
            for _ in range(max_tries):
                if len(pts) >= len(self.cubes):
                    break
                p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
                if all(np.linalg.norm(p - q) >= 0.045 for q in pts):
                    pts.append(p)

        def _rand_quat_yaw() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return q.elements

        for cube, pxy in zip(self.cubes, pts):
            pos = np.array([float(pxy[0]), float(pxy[1]), z], dtype=float)
            cube.set_pose(position=pos, quat=_rand_quat_yaw())

        for _ in range(80):
            self._mojo.step()

        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # rng = np.random.default_rng(self.seed)
        # _ = rng
        #
        # def _set_cube_pose(cube: SemanticCube, label: str, pos: np.ndarray):
        #     q = self._goal_quat[label]
        #     cube.set_pose(position=pos, quat=q.elements)
        #
        # _set_cube_pose(self.cube_4, "4", self._goal_pos["cube_4"])
        # _set_cube_pose(self.cube_5, "5", self._goal_pos["cube_5"])
        # _set_cube_pose(self.cube_6, "6", self._goal_pos["cube_6"])
        # for _ in range(50):
        #     self._mojo.step()
        #
        # _set_cube_pose(self.cube_2, "2", self._goal_pos["cube_2"])
        # _set_cube_pose(self.cube_3, "3", self._goal_pos["cube_3"])
        # for _ in range(50):
        #     self._mojo.step()
        #
        # _set_cube_pose(self.cube_1, "1", self._goal_pos["cube_1"])
        # for _ in range(50):
        #     self._mojo.step()
        #
        # for idx in range(self._NUM_SUBGOALS):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_on_table(self, cube: SemanticCube) -> bool:
        pos = cube.body.get_position()
        z_ok = abs(float(pos[2]) - self._BASE_Z) <= self._ON_TABLE_TOL_Z
        coll_ok = cube.is_colliding(self.table.table_top)
        return bool(z_ok and coll_ok)

    def _is_stacked_on(self, top: SemanticCube, bottom: SemanticCube) -> bool:
        p_top = top.body.get_position()
        p_bot = bottom.body.get_position()

        dz = float(p_top[2] - p_bot[2])
        z_ok = abs(dz - self._STACK_DZ) <= self._ON_TOL_Z

        dxy = np.linalg.norm((p_top[:2] - p_bot[:2]).astype(float))
        xy_ok = dxy <= self._ON_TOL_XY

        contact_ok = top.is_colliding(bottom)
        not_on_table = not top.is_colliding(self.table.table_top)

        return bool(z_ok and xy_ok and contact_ok and not_on_table)

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        # Guard named cube attributes
        for attr in ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5", "cube_6"]:
            if not hasattr(self, attr):
                return False

        # Identity-free positional/semantic check.
        # Group cubes by tier (Z) and within each tier sort by Y desc to get slots
        # left-to-right. Slots map to label targets:
        #   bottom (3 cubes): ["4","5","6"]
        #   middle (2 cubes): ["2","3"]
        #   top    (1 cube ): ["1"]
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        # Sort by Z ascending so we can split into tiers
        cubes_by_z = sorted(cubes_with_pos, key=lambda cp: cp[1][2])

        bottom_cp = cubes_by_z[0:3]
        middle_cp = cubes_by_z[3:5]
        top_cp = cubes_by_z[5:6]

        # Within each tier, sort by Y desc (+Y -> -Y = left -> right)
        bottom_sorted = sorted(bottom_cp, key=lambda cp: cp[1][1], reverse=True)
        middle_sorted = sorted(middle_cp, key=lambda cp: cp[1][1], reverse=True)
        top_sorted = top_cp  # only one

        bottom_cubes = [cp[0] for cp in bottom_sorted]
        bottom_positions = [cp[1] for cp in bottom_sorted]
        middle_cubes = [cp[0] for cp in middle_sorted]
        middle_positions = [cp[1] for cp in middle_sorted]
        top_cubes = [cp[0] for cp in top_sorted]

        p4, p5, p6 = bottom_positions[0], bottom_positions[1], bottom_positions[2]

        # Y-descending = left-to-right in the bimanual workspace.
        # y_left=0.06, y_mid=0.00, y_right=-0.06 matches left-to-right ordering (4, 5, 6).
        left_order_ok = bool(
            (p4[1] - p5[1]) > self._LEFT_OF_MIN_DY and (p5[1] - p6[1]) > self._LEFT_OF_MIN_DY
        )

        on_table_ok = all(self._is_on_table(c) for c in bottom_cubes)

        xs = np.array([p4[0], p5[0], p6[0]], dtype=float)
        row_align_ok = bool((xs.max() - xs.min()) <= self._ROW_ALIGN_TOL_X)

        # Middle slot 0 (left, target "2") stacks on bottom slot 0 (left, target "4")
        # Middle slot 1 (right, target "3") stacks on bottom slot 1 (middle, target "5")
        # Top stacks on middle slot 0 (target "2")
        stack_2_on_4 = self._is_stacked_on(middle_cubes[0], bottom_cubes[0]) if len(middle_cubes) >= 1 else False
        stack_3_on_5 = self._is_stacked_on(middle_cubes[1], bottom_cubes[1]) if len(middle_cubes) >= 2 else False
        stack_1_on_2 = self._is_stacked_on(top_cubes[0], middle_cubes[0]) if (len(top_cubes) >= 1 and len(middle_cubes) >= 1) else False

        positions_ok = bool(left_order_ok and on_table_ok and row_align_ok and stack_2_on_4 and stack_3_on_5 and stack_1_on_2)

        # Slot-based semantic check: each slot's occupying cube must show that slot's target label.
        slot_cube_target_pairs = [
            (bottom_cubes[0], "4"),
            (bottom_cubes[1], "5"),
            (bottom_cubes[2], "6"),
            (middle_cubes[0], "2"),
            (middle_cubes[1], "3"),
            (top_cubes[0], "1"),
        ]
        semantic_ok = all(
            c.is_face_readable(t, face_dir=self._FACE_DIR) for c, t in slot_cube_target_pairs
        )

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        bottom_row_ok = bool(left_order_ok and on_table_ok and row_align_ok)
        middle_ok = bool(stack_2_on_4 and stack_3_on_5)
        top_ok = bool(stack_1_on_2)
        orientation_ok = bool(semantic_ok)
        released_ok = bool(released)

        self._metric_stage(0, bottom_row_ok)
        self._metric_stage(1, middle_ok)
        self._metric_stage(2, top_ok)
        self._metric_stage(3, orientation_ok)
        self._metric_stage(4, released_ok)

        self._success_check = bool(positions_ok and semantic_ok and released)

        target_distance = {}
        pose_error = {}

        cube_map = {
            "cube_1": (self.cube_1, "1"),
            "cube_2": (self.cube_2, "2"),
            "cube_3": (self.cube_3, "3"),
            "cube_4": (self.cube_4, "4"),
            "cube_5": (self.cube_5, "5"),
            "cube_6": (self.cube_6, "6"),
        }

        for name, (cube, label) in cube_map.items():
            cur_pos = cube.body.get_position().astype(float)
            goal_pos = self._goal_pos.get(name, None)
            if goal_pos is not None:
                target_distance[name] = float(np.linalg.norm(cur_pos - goal_pos))
            else:
                target_distance[name] = float("inf")

            cur_q = Quaternion(cube.body.get_quaternion())
            goal_q = self._goal_quat.get(label, None)
            if goal_q is not None:
                dq = goal_q.conjugate * cur_q
                pose_error[name] = float(abs(dq.angle))
            else:
                pose_error[name] = float("inf")

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
