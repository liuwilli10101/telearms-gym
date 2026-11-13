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

class RedCubeStackingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Stack three red colored cubes into a single vertical column (tower) on the table.
    The final arrangement is three red cubes with one on top of another.

    Sequential Sub-Goals:
        1. Securely grasp one red cube to serve as the base cube.
        2. Place the base cube flat on the table in a stable position.
        3. Grasp a second red cube for stacking.
        4. Align the second cube above the base cube before lowering.
        5. Place the second cube on top of the base cube.
        6. Grasp the third red cube for the top of the tower.
        7. Align the third cube above the second cube before lowering.
        8. Place the third cube on top of the second cube to complete the tower.
        9. Release the cubes without toppling the stack.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry constants for ColoredCube (4cm)
    _CUBE_SIZE = 0.04
    _Z0 = 0.97
    _DZ = 0.04

    # Tolerances
    _XY_ALIGN_TOL = 0.014
    _Z_STEP_TOL = 0.02  # slightly looser than XY due to settling variance

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (assets must be ONLY ColoredCube)
        self.red_cube_1 = ColoredCube(self._mojo)
        self.red_cube_2 = ColoredCube(self._mojo)
        self.red_cube_3 = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        for c in (self.red_cube_1, self.red_cube_2, self.red_cube_3):
            c.set_color("red")

        self.cubes = [self.red_cube_1, self.red_cube_2, self.red_cube_3]

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

        # 2) Deterministic RNG (pipeline may add randomization later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        for c in self.cubes:
            c.set_color("red")

        # 3) RANDOM unsolved placements within workspace bounds
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Keep cubes separated so they don't start stacked (ensures _success() is False)
        min_xy_sep = 2.2 * self._CUBE_SIZE  # conservative separation (> cube width)
        goal_xy = np.array([0.55, 0.00], dtype=float)

        def sample_xy(existing: list[np.ndarray]) -> np.ndarray:
            for _ in range(500):
                xy = np.array(
                    [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
                )
                # Avoid being too close to goal column
                if np.linalg.norm(xy - goal_xy) < 0.06:
                    continue
                # Enforce separation
                if all(np.linalg.norm(xy - e) >= min_xy_sep for e in existing):
                    return xy
            # Fallback (still bounded)
            return np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
            )

        xys: list[np.ndarray] = []
        for _ in range(3):
            xys.append(sample_xy(xys))

        # Small random yaw only; keep cubes upright (prevents accidental stable stacking)
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        q1 = yaw_quat(rng.uniform(-np.pi, np.pi))
        q2 = yaw_quat(rng.uniform(-np.pi, np.pi))
        q3 = yaw_quat(rng.uniform(-np.pi, np.pi))

        # Place all on table (unsolved: only one should be base in success, others must not touch table)
        self.red_cube_1.set_pose(position=np.array([xys[0][0], xys[0][1], self._Z0]), quat=q1)
        self._mojo.physics.forward()

        self.red_cube_2.set_pose(position=np.array([xys[1][0], xys[1][1], self._Z0]), quat=q2)
        self._mojo.physics.forward()

        self.red_cube_3.set_pose(position=np.array([xys[2][0], xys[2][1], self._Z0]), quat=q3)
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(9):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        c1, c2, c3 = self.red_cube_1, self.red_cube_2, self.red_cube_3

        # Semantic (color) verification
        semantic_ok = all(c.get_color() == "red" for c in (c1, c2, c3))

        # Positions / relations
        p1 = c1.body.get_position()
        p2 = c2.body.get_position()
        p3 = c3.body.get_position()

        # Base on table; upper cubes not touching table
        base_on_table = c1.is_colliding(self.table.table_top)
        c2_not_on_table = not c2.is_colliding(self.table.table_top)
        c3_not_on_table = not c3.is_colliding(self.table.table_top)

        # "On" relations via reliable collision + height ordering
        c2_on_c1 = c2.is_colliding(c1) and (p2[2] > p1[2] + 0.5 * self._CUBE_SIZE)
        c3_on_c2 = c3.is_colliding(c2) and (p3[2] > p2[2] + 0.5 * self._CUBE_SIZE)

        # Vertical column alignment (XY coincidence)
        align_12 = self._xy_dist(p1, p2) <= self._XY_ALIGN_TOL
        align_23 = self._xy_dist(p2, p3) <= self._XY_ALIGN_TOL

        # Approximate Z stepping (robust to small settling)
        zstep_12 = abs((p2[2] - p1[2]) - self._DZ) <= self._Z_STEP_TOL
        zstep_23 = abs((p3[2] - p2[2]) - self._DZ) <= self._Z_STEP_TOL

        positions_ok = bool(
            base_on_table
            and c2_not_on_table
            and c3_not_on_table
            and c2_on_c1
            and c3_on_c2
            and align_12
            and align_23
            and zstep_12
            and zstep_23
        )

        # Released gripper condition (generic over grippers)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in (c1, c2, c3)
            for s in self.robot.grippers
        )

        # Subgoal staging (proxy signals derived from final relations)
        # 0: grasp base (proxy: base placed stably)
        # 1: place base
        if base_on_table:
            self._metric_stage(0, True)
            self._metric_stage(1, True)

        # 2-4: second cube grasp/align/place (proxy: on + aligned)
        if c2_on_c1:
            self._metric_stage(2, True)
            self._metric_stage(4, True)
        if c2_on_c1 and align_12:
            self._metric_stage(3, True)

        # 5-7: third cube grasp/align/place (proxy: on + aligned)
        if c3_on_c2:
            self._metric_stage(5, True)
            self._metric_stage(7, True)
        if c3_on_c2 and align_23:
            self._metric_stage(6, True)

        # 8: release without toppling (proxy: full stack + released)
        if positions_ok and released and semantic_ok:
            self._metric_stage(8, True)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "xy_align_12": float(self._xy_dist(p1, p2)),
                "xy_align_23": float(self._xy_dist(p2, p3)),
                "z_step_12_err": float(abs((p2[2] - p1[2]) - self._DZ)),
                "z_step_23_err": float(abs((p3[2] - p2[2]) - self._DZ)),
            },
            pose_error={
                "c1_to_goal": float(
                    np.linalg.norm(p1 - np.array([0.55, 0.00, self._Z0]))
                ),
                "c2_to_goal": float(
                    np.linalg.norm(p2 - np.array([0.55, 0.00, self._Z0 + self._DZ]))
                ),
                "c3_to_goal": float(
                    np.linalg.norm(p3 - np.array([0.55, 0.00, self._Z0 + 2 * self._DZ]))
                ),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})