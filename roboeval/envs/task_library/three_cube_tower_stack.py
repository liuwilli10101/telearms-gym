from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class ThreeCubeTowerStack(RoboEvalEnv, MetricRolloutEval):
    """
    Stack three colored cubes into a single vertical tower with a 1x1 base (Rows: 1, Cols: 1).
    All cubes should be stably stacked (no gaps, no sideways offset) with no other arrangements required.

    Sequential Sub-Goals:
        1. Grasp any cube with the left gripper.
        2. Grasp any cube with the right gripper.
        3. Lift a grasped cube clear of the table surface to enable stacking.
        4. Move a cube above another cube to prepare for stacking.
        5. Place a cube on top of another cube to form the tower.
        6. Adjust the stack so cubes are vertically aligned (minimal sideways offset) and stable.
        7. Release all cubes after the tower is completed.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_SIZE_M = 0.04
    _BASE_Z = 0.97
    _STACK_DZ = 0.04

    _ON_TABLE_Z_TOL = 0.02
    _XY_ALIGN_TOL = 0.025
    _CONTACT_Z_TOL = 0.015

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3]

        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("green")

        self._stage_done = [False] * self._NUM_SUBGOALS

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

        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("green")

        # Randomize cubes into an unsolved configuration: all on table, separated in XY, random yaw.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._BASE_Z

        # Keep cubes well separated so they don't start stacked/colliding.
        min_xy_sep = 0.08  # > cube size to avoid contact and accidental stacking
        points = []
        for _ in range(1000):
            if len(points) >= 3:
                break
            cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if all(float(np.linalg.norm(cand - p)) >= min_xy_sep for p in points):
                points.append(cand)

        # Fallback (should be rare): deterministic spread if rejection sampling fails.
        if len(points) < 3:
            points = [
                np.array([0.45, -0.15], dtype=float),
                np.array([0.60, 0.00], dtype=float),
                np.array([0.65, 0.15], dtype=float),
            ]

        cubes = [self.cube_1, self.cube_2, self.cube_3]
        rng.shuffle(cubes)

        for cube, xy in zip(cubes, points[:3]):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=float)
            pos = np.array([float(xy[0]), float(xy[1]), z], dtype=float)
            cube.set_pose(position=pos, quat=quat)
            for _ in range(25):
                self._mojo.step()

        self._stage_done = [False] * self._NUM_SUBGOALS
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # rng = np.random.default_rng(self.seed)
        # _ = rng
        #
        # self.cube_1.set_color("red")
        # self.cube_2.set_color("blue")
        # self.cube_3.set_color("green")
        #
        # x0, y0 = 0.55, 0.0
        #
        # p1 = np.array([x0, y0, self._BASE_Z])
        # p2 = np.array([x0, y0, self._BASE_Z + self._STACK_DZ])
        # p3 = np.array([x0, y0, self._BASE_Z + 2 * self._STACK_DZ])
        #
        # q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        #
        # self.cube_1.set_pose(position=p1, quat=q)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # self.cube_2.set_pose(position=p2, quat=q)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # self.cube_3.set_pose(position=p3, quat=q)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # self._stage_done = [False] * self._NUM_SUBGOALS
        # for idx in range(self._NUM_SUBGOALS):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _update_stage(self, idx: int, cond: bool):
        if cond and not self._stage_done[idx]:
            self._stage_done[idx] = True
            self._metric_stage(idx, True)

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Sort cubes by Z position to dynamically assign bottom/middle/top roles.
        # This avoids false failures when any ordering of the 3 cubes forms a valid tower.
        cubes_by_z = sorted(self.cubes, key=lambda c: c.body.get_position()[2])
        bottom, middle, top = cubes_by_z[0], cubes_by_z[1], cubes_by_z[2]

        p_bottom = bottom.body.get_position()
        p_middle = middle.body.get_position()
        p_top = top.body.get_position()

        on_table_z_ok = abs(float(p_bottom[2]) - self._BASE_Z) <= self._ON_TABLE_Z_TOL
        bottom_on_table = bottom.is_colliding(self.table.table_top) and on_table_z_ok

        upper_not_on_table = (
            not middle.is_colliding(self.table.table_top)
            and not top.is_colliding(self.table.table_top)
        )

        contact_bm = middle.is_colliding(bottom)
        contact_mt = top.is_colliding(middle)

        dz_bm = float(p_middle[2] - p_bottom[2])
        dz_mt = float(p_top[2] - p_middle[2])
        z_contact_ok = (
            abs(dz_bm - self._STACK_DZ) <= self._CONTACT_Z_TOL
            and abs(dz_mt - self._STACK_DZ) <= self._CONTACT_Z_TOL
        )

        xy_bm = self._xy_dist(p_middle, p_bottom)
        xy_mt = self._xy_dist(p_top, p_middle)
        xy_align_ok = xy_bm <= self._XY_ALIGN_TOL and xy_mt <= self._XY_ALIGN_TOL

        positions_ok = bool(
            bottom_on_table and upper_not_on_table
            and contact_bm and contact_mt
            and z_contact_ok and xy_align_ok
        )

        released = not any(
            self.robot.is_gripper_holding_object(obj, side)
            for obj in self.cubes
            for side in self.robot.grippers
        )

        gripper_sides = list(self.robot.grippers)

        if len(gripper_sides) >= 1:
            g0 = gripper_sides[0]
            g0_grasping = any(self.robot.is_gripper_holding_object(obj, g0) for obj in self.cubes)
            self._update_stage(0, g0_grasping)

        if len(gripper_sides) >= 2:
            g1 = gripper_sides[1]
            g1_grasping = any(self.robot.is_gripper_holding_object(obj, g1) for obj in self.cubes)
            self._update_stage(1, g1_grasping)

        lifted_any = any(
            (float(c.body.get_position()[2]) > (self._BASE_Z + 0.03))
            and not c.is_colliding(self.table.table_top)
            for c in self.cubes
        )
        self._update_stage(2, lifted_any)

        transported_any = (
            (float(p_middle[2]) > float(p_bottom[2]) + 0.03 and self._xy_dist(p_middle, p_bottom) < 0.06)
            or (float(p_top[2]) > float(p_middle[2]) + 0.03 and self._xy_dist(p_top, p_middle) < 0.06)
        )
        self._update_stage(3, transported_any)

        placed_ok = bool(
            bottom_on_table and upper_not_on_table
            and contact_bm and contact_mt and z_contact_ok
        )
        self._update_stage(4, placed_ok)

        aligned_ok = bool(placed_ok and xy_align_ok)
        self._update_stage(5, aligned_ok)

        self._update_stage(6, bool(aligned_ok and released))

        self._success_check = bool(positions_ok and released)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "xy_align_bm_m": float(xy_bm),
                "xy_align_mt_m": float(xy_mt),
                "z_gap_err_bm_m": float(abs(dz_bm - self._STACK_DZ)),
                "z_gap_err_mt_m": float(abs(dz_mt - self._STACK_DZ)),
            },
            pose_error={
                "bottom_z_err_m": float(abs(float(p_bottom[2]) - self._BASE_Z)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
