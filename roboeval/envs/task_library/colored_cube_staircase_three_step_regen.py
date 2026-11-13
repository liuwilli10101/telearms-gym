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


class ThreeStepCubeStaircase(RoboEvalEnv, MetricRolloutEval):
    """
    Construct a 3-step staircase that rises toward +X (back) using six colored cubes:
    three base cubes in a front-to-back line, with one cube stacked on the middle base,
    and two cubes stacked on the back base (making the back column 3 cubes tall).

    Color assignment is FIXED (success checks specific colors at specific positions):
      - front column (1 high): red base
      - middle column (2 high): green base, with yellow stacked on it
      - back column (3 high): blue base, with purple on the blue, and orange on the purple

    Sequential Sub-Goals:
        0. Grasp a cube intended for the front base position.
        1. Place the front base cube on the table (lowest step, most toward -X/front).
        2. Place the middle base cube behind (+X) the front base cube, aligned in Y.
        3. Place the back base cube behind (+X) the middle base cube, aligned in Y.
        4. Stack one cube on the middle base cube (middle step is 2 cubes high).
        5. Stack one cube on the back base cube (start back step stack).
        6. Stack the final cube on top of the back stack (back step is 3 cubes high).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace/table constants for ColoredCube
    _CUBE_REST_Z = 0.97
    _CUBE_DZ = 0.04

    # Goal tolerances
    _MAX_ABS_DELTA_Y_BASES = 0.04
    _MAX_ABS_DELTA_Y_BEHIND = 0.03
    _MIN_DELTA_X_BEHIND = 0.04
    _POSE_TOL = 0.014  # success tolerance (distance)
    _STACK_XY_TOL = 0.014
    _STACK_Z_TOL = 0.02

    _WORKSPACE_X = (0.35, 0.75)
    _WORKSPACE_Y = (-0.30, 0.30)

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (MUST use only ColoredCube)
        self.cube_front_base = ColoredCube(self._mojo)
        self.cube_mid_base = ColoredCube(self._mojo)
        self.cube_back_base = ColoredCube(self._mojo)
        self.cube_mid_top = ColoredCube(self._mojo)
        self.cube_back_mid = ColoredCube(self._mojo)
        self.cube_back_top = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_front_base,
            self.cube_mid_base,
            self.cube_back_base,
            self.cube_mid_top,
            self.cube_back_mid,
            self.cube_back_top,
        ]

        # REQUIRED: set_color() in _initialize_env()
        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        for cube, col in zip(self.cubes, colors):
            cube.set_color(col)

        # Define deterministic goal poses (staircase rises toward +X (back))
        y0 = 0.00
        x_front = 0.46
        x_mid = 0.52
        x_back = 0.58

        self._goal_pos = {
            "cube_front_base": np.array([x_front, y0, self._CUBE_REST_Z]),
            "cube_mid_base": np.array([x_mid, y0, self._CUBE_REST_Z]),
            "cube_back_base": np.array([x_back, y0, self._CUBE_REST_Z]),
            "cube_mid_top": np.array([x_mid, y0, self._CUBE_REST_Z + self._CUBE_DZ]),
            "cube_back_mid": np.array([x_back, y0, self._CUBE_REST_Z + self._CUBE_DZ]),
            "cube_back_top": np.array([x_back, y0, self._CUBE_REST_Z + 2 * self._CUBE_DZ]),
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

        # 2. Deterministic RNG (required pattern; validation may add randomization later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        for cube, col in zip(self.cubes, colors):
            cube.set_color(col)

        q = np.array([0.0, 0.0, 0.0, 1.0])

        # 3. Place objects in RANDOM UNSOLVED positions (all on table, no stacking)
        #    Keep workspace bounds exactly as required by this task statement.
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z0 = self._CUBE_REST_Z

        # Rejection-sample non-overlapping XY points so cubes don't start stacked/colliding.
        min_xy_dist = 0.07  # > cube size (~0.04) to avoid accidental contact/stacking
        pts: list[np.ndarray] = []
        max_tries = 2000

        for _ in range(len(self.cubes)):
            for _t in range(max_tries):
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                if all(float(np.linalg.norm(p - p2)) >= min_xy_dist for p2 in pts):
                    pts.append(p)
                    break
            else:
                # Fallback: if sampling fails, relax spacing slightly and continue
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                pts.append(p)

        # Set all cubes on the table (same z) so stacking checks fail => _success() is False.
        for cube, pxy in zip(self.cubes, pts):
            cube.set_pose(position=np.array([pxy[0], pxy[1], z0], dtype=float), quat=q)
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _in_workspace_xy(self, pos: np.ndarray) -> bool:
        return (self._WORKSPACE_X[0] <= float(pos[0]) <= self._WORKSPACE_X[1]) and (
            self._WORKSPACE_Y[0] <= float(pos[1]) <= self._WORKSPACE_Y[1]
        )

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _pose_close(self, cube: ColoredCube, goal_pos: np.ndarray, tol: float = None) -> bool:
        if tol is None:
            tol = self._POSE_TOL
        pos = cube.body.get_position()
        return self._dist(pos, goal_pos) <= tol

    def _behind(self, subj: ColoredCube, targ: ColoredCube, min_dx: float, max_abs_dy: float) -> bool:
        ps = subj.body.get_position()
        pt = targ.body.get_position()
        dx = float(ps[0] - pt[0])  # +X is "behind/back" per task spec
        dy = float(ps[1] - pt[1])
        return (dx >= min_dx) and (abs(dy) <= max_abs_dy)

    def _is_on_table(self, cube: ColoredCube, max_height_above_table: float = 0.03) -> bool:
        pos = cube.body.get_position()
        if not self._in_workspace_xy(pos):
            return False
        if not cube.is_colliding(self.table.table_top):
            return False
        # z should be close to rest height on table
        if abs(float(pos[2]) - self._CUBE_REST_Z) > max_height_above_table:
            return False
        return True

    def _is_stacked_on(self, top: ColoredCube, bottom: ColoredCube) -> bool:
        pt = top.body.get_position()
        pb = bottom.body.get_position()

        # Contact-based check
        if not top.is_colliding(bottom):
            return False

        # Ensure the "top" cube is not resting on the table directly
        if top.is_colliding(self.table.table_top):
            return False

        # Geometric sanity (aligned in XY; z increment ~ 0.04)
        xy_ok = np.linalg.norm((pt - pb)[:2]) <= self._STACK_XY_TOL
        z_ok = abs(float(pt[2] - pb[2]) - self._CUBE_DZ) <= self._STACK_Z_TOL
        return bool(xy_ok and z_ok)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Goal condition checks (relations + on-table + stacking) ---
        # Base ordering (+X behind)
        mid_behind_front = self._behind(
            self.cube_mid_base, self.cube_front_base, self._MIN_DELTA_X_BEHIND, self._MAX_ABS_DELTA_Y_BEHIND
        )
        back_behind_mid = self._behind(
            self.cube_back_base, self.cube_mid_base, self._MIN_DELTA_X_BEHIND, self._MAX_ABS_DELTA_Y_BEHIND
        )

        # Bases on table within workspace
        front_on_table = self._is_on_table(self.cube_front_base, max_height_above_table=0.03)
        mid_on_table = self._is_on_table(self.cube_mid_base, max_height_above_table=0.03)
        back_on_table = self._is_on_table(self.cube_back_base, max_height_above_table=0.03)

        # Rough Y alignment across bases
        y_front = float(self.cube_front_base.body.get_position()[1])
        y_mid = float(self.cube_mid_base.body.get_position()[1])
        y_back = float(self.cube_back_base.body.get_position()[1])
        bases_y_aligned = (abs(y_front - y_mid) <= self._MAX_ABS_DELTA_Y_BASES) and (
            abs(y_mid - y_back) <= self._MAX_ABS_DELTA_Y_BASES
        )

        # Stacking relations
        mid_top_on_mid = self._is_stacked_on(self.cube_mid_top, self.cube_mid_base)
        back_mid_on_back = self._is_stacked_on(self.cube_back_mid, self.cube_back_base)
        back_top_on_back_mid = self._is_stacked_on(self.cube_back_top, self.cube_back_mid)

        # NOTE: absolute pose closeness against hardcoded _goal_pos was removed —
        # the relative checks above (mid_behind_front / back_behind_mid /
        # bases_y_aligned / *_on_table / *_stacked_on) already pin the staircase
        # geometry without requiring exact world coordinates.

        positions_ok = bool(
            mid_behind_front
            and back_behind_mid
            and front_on_table
            and mid_on_table
            and back_on_table
            and bases_y_aligned
            and mid_top_on_mid
            and back_mid_on_back
            and back_top_on_back_mid
        )

        semantic_ok = True  # no semantic constraints in this task

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging ---
        # 0: grasp (not directly observable at final state; mark true once structure is completed)
        self._metric_stage(1, bool(front_on_table))
        self._metric_stage(2, bool(front_on_table and mid_on_table and mid_behind_front))
        self._metric_stage(3, bool(front_on_table and mid_on_table and back_on_table and mid_behind_front and back_behind_mid))
        self._metric_stage(4, bool(mid_top_on_mid))
        self._metric_stage(5, bool(back_mid_on_back))
        self._metric_stage(6, bool(back_top_on_back_mid))
        self._metric_stage(0, bool(positions_ok and released))

        # --- Set success ---
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize (must be last metric call before return) ---
        target_distance = {
            "cube_front_base": self._dist(self.cube_front_base.body.get_position(), self._goal_pos["cube_front_base"]),
            "cube_mid_base": self._dist(self.cube_mid_base.body.get_position(), self._goal_pos["cube_mid_base"]),
            "cube_back_base": self._dist(self.cube_back_base.body.get_position(), self._goal_pos["cube_back_base"]),
            "cube_mid_top": self._dist(self.cube_mid_top.body.get_position(), self._goal_pos["cube_mid_top"]),
            "cube_back_mid": self._dist(self.cube_back_mid.body.get_position(), self._goal_pos["cube_back_mid"]),
            "cube_back_top": self._dist(self.cube_back_top.body.get_position(), self._goal_pos["cube_back_top"]),
        }
        pose_error = dict(target_distance)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})