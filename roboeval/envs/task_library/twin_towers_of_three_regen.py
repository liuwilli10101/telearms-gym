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

class ParallelCubeStacks(RoboEvalEnv, MetricRolloutEval):
    """
    Build two vertical stacks of three cubes each. The two stacks should be parallel and
    separated left-to-right by 10 cm (along the Y axis).

    Sequential Sub-Goals:
        1. Place cube_L1 on the table (start left stack base).
        2. Stack cube_L2 on cube_L1.
        3. Stack cube_L3 on cube_L2 (left stack complete).
        4. Place cube_R1 on the table (start right stack base).
        5. Stack cube_R2 on cube_R1.
        6. Stack cube_R3 on cube_R2 (right stack complete).
        7. Ensure bases are aligned in X and separated by +0.10 m in Y (cube_L1 left of cube_R1).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal tolerances from task spec
    _STACK_POS_TOL = 0.02
    _BASE_Y_SEP_M = 0.10
    _BASE_Y_TOL = 0.02
    _BASE_X_TOL = 0.02

    # ColoredCube geometry
    _CUBE_Z0 = 0.97
    _DZ = 0.04

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create exactly and only ColoredCube assets
        self.cube_L1 = ColoredCube(self._mojo)
        self.cube_L2 = ColoredCube(self._mojo)
        self.cube_L3 = ColoredCube(self._mojo)
        self.cube_R1 = ColoredCube(self._mojo)
        self.cube_R2 = ColoredCube(self._mojo)
        self.cube_R3 = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_L1, self.cube_L2, self.cube_L3,
            self.cube_R1, self.cube_R2, self.cube_R3,
        ]

        # REQUIRED: set_color() in _initialize_env()
        # (Colors are not part of success criteria; just to disambiguate visually.)
        self.cube_L1.set_color("red")
        self.cube_L2.set_color("orange")
        self.cube_L3.set_color("yellow")
        self.cube_R1.set_color("blue")
        self.cube_R2.set_color("purple")
        self.cube_R3.set_color("green")

        # Precompute goal poses (two stacks, +Y is "left", -Y is "right")
        self._goal_x = 0.55
        self._goal_y_L = 0.05
        self._goal_y_R = -0.05  # y_L - y_R = +0.10

        self._goal_pos = {
            "cube_L1": np.array([self._goal_x, self._goal_y_L, self._CUBE_Z0]),
            "cube_L2": np.array([self._goal_x, self._goal_y_L, self._CUBE_Z0 + 1 * self._DZ]),
            "cube_L3": np.array([self._goal_x, self._goal_y_L, self._CUBE_Z0 + 2 * self._DZ]),
            "cube_R1": np.array([self._goal_x, self._goal_y_R, self._CUBE_Z0]),
            "cube_R2": np.array([self._goal_x, self._goal_y_R, self._CUBE_Z0 + 1 * self._DZ]),
            "cube_R3": np.array([self._goal_x, self._goal_y_R, self._CUBE_Z0 + 2 * self._DZ]),
        }

        # ⚠️ CRITICAL: Initialize metrics with FULL signature
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _sample_initial_state(self):
        """
        The validation pipeline may apply additional randomization externally.
        Keep this deterministic and compatible with the required goal-state reset.
        """
        return {}

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

        # 3) REQUIRED: set_color() in _on_reset()
        self.cube_L1.set_color("red")
        self.cube_L2.set_color("orange")
        self.cube_L3.set_color("yellow")
        self.cube_R1.set_color("blue")
        self.cube_R2.set_color("purple")
        self.cube_R3.set_color("green")

        q = np.array([0.0, 0.0, 0.0, 1.0])

        # 4) Place objects in RANDOM UNSOLVED positions within workspace bounds
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = float(self._CUBE_Z0)

        # Keep cubes separated so they don't accidentally stack/collide at reset
        min_xy_sep = 0.065  # > cube size, avoids overlap/stacking at t=0

        # Also keep at least one cube clearly away from its goal so pose_ok cannot pass
        goal_positions_xy = np.array([self._goal_pos[k][:2] for k in ["cube_L1", "cube_L2", "cube_L3", "cube_R1", "cube_R2", "cube_R3"]])
        min_goal_avoid = 0.05

        placed_xy = []

        def sample_xy():
            for _ in range(2000):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)

                # keep away from previously placed cubes
                if placed_xy:
                    dmin = float(np.min(np.linalg.norm(np.asarray(placed_xy) - p[None, :], axis=1)))
                    if dmin < min_xy_sep:
                        continue

                # avoid being too close to ANY goal XY (prevents accidental success)
                if float(np.min(np.linalg.norm(goal_positions_xy - p[None, :], axis=1))) < min_goal_avoid:
                    continue

                return p

            # Fallback: deterministic spread (still unsolved)
            idx = len(placed_xy)
            x = x_min + (idx + 1) * (x_max - x_min) / 7.0
            y = y_min + (idx + 1) * (y_max - y_min) / 7.0
            return np.array([float(x), float(y)], dtype=float)

        cubes_in_order = [
            ("cube_L1", self.cube_L1),
            ("cube_L2", self.cube_L2),
            ("cube_L3", self.cube_L3),
            ("cube_R1", self.cube_R1),
            ("cube_R2", self.cube_R2),
            ("cube_R3", self.cube_R3),
        ]

        # Place all cubes flat on the table (no initial stacks)
        for _, cube in cubes_in_order:
            xy = sample_xy()
            placed_xy.append(xy)
            cube.set_pose(position=np.array([xy[0], xy[1], z0], dtype=float), quat=q)

        # Let physics settle a bit
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _pos(self, cube: ColoredCube) -> np.ndarray:
        return np.array(cube.body.get_position())

    def _close_to(self, cube: ColoredCube, goal_pos: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(self._pos(cube) - goal_pos)) <= float(tol)

    def _is_stacked_on(self, top: ColoredCube, bottom: ColoredCube, tol_xy: float) -> bool:
        """
        Conservative 'on' check combining collision and approximate pose.
        """
        pt = self._pos(top)
        pb = self._pos(bottom)

        # Expected vertical offset for adjacent cubes in a stack
        ok_z = abs((pt[2] - pb[2]) - self._DZ) <= 0.02
        ok_xy = np.linalg.norm(pt[:2] - pb[:2]) <= tol_xy

        colliding = top.is_colliding(bottom)
        not_on_table = not top.is_colliding(self.table.table_top)

        return bool(ok_z and ok_xy and colliding and not_on_table)

    def _compute_reward(self) -> float:
        """
        Dense shaping: negative sum of distances to goal poses, plus base-separation error.
        """
        if not hasattr(self, "cubes") or not hasattr(self, "_goal_pos"):
            return 0.0

        goals = [
            ("cube_L1", self.cube_L1),
            ("cube_L2", self.cube_L2),
            ("cube_L3", self.cube_L3),
            ("cube_R1", self.cube_R1),
            ("cube_R2", self.cube_R2),
            ("cube_R3", self.cube_R3),
        ]

        dist_sum = 0.0
        for name, cube in goals:
            dist_sum += float(np.linalg.norm(self._pos(cube) - self._goal_pos[name]))

        # Base separation/alignment shaping
        pL = self._pos(self.cube_L1)
        pR = self._pos(self.cube_R1)
        y_err = abs((pL[1] - pR[1]) - self._BASE_Y_SEP_M)
        x_err = abs(pL[0] - pR[0])

        reward = -(dist_sum + 2.0 * y_err + 1.0 * x_err)

        # Small success bonus
        if getattr(self, "_success_check", False):
            reward += 5.0
        return float(reward)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_pos"):
            return False

        # Released constraint (single-arm safe pattern: iterate over grippers)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # NOTE: per-cube absolute pose closeness against _goal_pos was removed —
        # the stack relations + base y-separation + base x-alignment below already
        # pin the geometry without requiring exact world coordinates.

        # Stacking relations
        left_stack_ok = (
            self.cube_L1.is_colliding(self.table.table_top) and
            self._is_stacked_on(self.cube_L2, self.cube_L1, tol_xy=0.02) and
            self._is_stacked_on(self.cube_L3, self.cube_L2, tol_xy=0.02)
        )
        right_stack_ok = (
            self.cube_R1.is_colliding(self.table.table_top) and
            self._is_stacked_on(self.cube_R2, self.cube_R1, tol_xy=0.02) and
            self._is_stacked_on(self.cube_R3, self.cube_R2, tol_xy=0.02)
        )

        # Base separation (+Y is left): y_L1 - y_R1 ≈ +0.10, and x aligned
        pL1 = self._pos(self.cube_L1)
        pR1 = self._pos(self.cube_R1)
        y_sep_ok = abs((pL1[1] - pR1[1]) - self._BASE_Y_SEP_M) <= self._BASE_Y_TOL
        x_align_ok = abs(pL1[0] - pR1[0]) <= self._BASE_X_TOL

        positions_ok = bool(left_stack_ok and right_stack_ok and y_sep_ok and x_align_ok)
        semantic_ok = True  # no semantic constraints in this task

        # Subgoal staging (best-effort; grasp is not directly observable here)
        sg0 = self.cube_L1.is_colliding(self.table.table_top)
        sg1 = self._is_stacked_on(self.cube_L2, self.cube_L1, tol_xy=0.03)
        sg2 = self._is_stacked_on(self.cube_L3, self.cube_L2, tol_xy=0.03)
        sg3 = self.cube_R1.is_colliding(self.table.table_top)
        sg4 = self._is_stacked_on(self.cube_R2, self.cube_R1, tol_xy=0.03)
        sg5 = self._is_stacked_on(self.cube_R3, self.cube_R2, tol_xy=0.03)
        sg6 = bool(y_sep_ok and x_align_ok)

        subgoals = [sg0, sg1, sg2, sg3, sg4, sg5, sg6]
        for i, ok in enumerate(subgoals):
            if ok:
                self._metric_stage(i, True)

        # Set success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        dists = {
            "cube_L1": float(np.linalg.norm(self._pos(self.cube_L1) - self._goal_pos["cube_L1"])),
            "cube_L2": float(np.linalg.norm(self._pos(self.cube_L2) - self._goal_pos["cube_L2"])),
            "cube_L3": float(np.linalg.norm(self._pos(self.cube_L3) - self._goal_pos["cube_L3"])),
            "cube_R1": float(np.linalg.norm(self._pos(self.cube_R1) - self._goal_pos["cube_R1"])),
            "cube_R2": float(np.linalg.norm(self._pos(self.cube_R2) - self._goal_pos["cube_R2"])),
            "cube_R3": float(np.linalg.norm(self._pos(self.cube_R3) - self._goal_pos["cube_R3"])),
            "base_y_sep_error": float(abs((pL1[1] - pR1[1]) - self._BASE_Y_SEP_M)),
            "base_x_align_error": float(abs(pL1[0] - pR1[0])),
        }
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dists,
            pose_error=dists,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})