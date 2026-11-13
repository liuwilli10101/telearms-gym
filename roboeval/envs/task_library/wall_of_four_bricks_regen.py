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

class ColoredCubeWallBuilding(RoboEvalEnv, MetricRolloutEval):
    """Build a 2-by-2 wall from four colored cubes.

Success Criteria:
Any colour may go in any slot. Two cubes sit side by side on the table, close enough to touch or nearly so; the other two are stacked one on each of them; both rows line up front to back and every cube is released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry / tolerances
    _BASE_Z = 0.97
    _CUBE_DZ = 0.04
    _BOTTOM_Z = _BASE_Z
    _TOP_Z = _BASE_Z + _CUBE_DZ

    _XY_TOL = 0.03
    _X_ALIGN_TOL = 0.03
    _LEFT_OF_MIN_DY = 0.02  # enforce correct ordering along +Y (left) vs -Y (right)
    # Only the ordering was ever checked, so two separate two-cube towers built a
    # metre apart passed as a "2-by-2 wall". The columns have to be adjacent for the
    # arrangement to read as one wall: cubes are 4 cm wide, so this allows up to a
    # 3 cm gap between the bottom pair and nothing wider.
    _LEFT_OF_MAX_DY = 0.07

    # Goal layout (stable side-by-side spacing)
    _CENTER_X = 0.55
    _CENTER_Y = 0.0
    _Y_SPACING = 0.05  # centers separated along Y

    _NUM_SUBGOALS = 8

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create required assets: ONLY ColoredCube (x4)
        self.cube_bottom_left = ColoredCube(self._mojo)
        self.cube_bottom_right = ColoredCube(self._mojo)
        self.cube_top_left = ColoredCube(self._mojo)
        self.cube_top_right = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_bottom_left,
            self.cube_bottom_right,
            self.cube_top_left,
            self.cube_top_right,
        ]

        # REQUIRED: set_color() in _initialize_env()
        self._cube_colors = ["red", "blue", "green", "yellow"]
        for cube, color in zip(self.cubes, self._cube_colors):
            cube.set_color(color)

        # Pre-compute goal poses (positions only; quats set at placement time)
        y_left = self._CENTER_Y + self._Y_SPACING / 2.0
        y_right = self._CENTER_Y - self._Y_SPACING / 2.0

        self._goal_pos = {
            "cube_bottom_left": np.array([self._CENTER_X, y_left, self._BOTTOM_Z]),
            "cube_bottom_right": np.array([self._CENTER_X, y_right, self._BOTTOM_Z]),
            "cube_top_left": np.array([self._CENTER_X, y_left, self._TOP_Z]),
            "cube_top_right": np.array([self._CENTER_X, y_right, self._TOP_Z]),
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

    def reset_object_placements(self, rng: np.random.Generator):
        """
        Place objects in a goal-satisfying configuration (the validation pipeline may
        apply additional randomization externally).
        """
        q = np.array([0.0, 0.0, 0.0, 1.0])

        # Layer-by-layer placement for stability: bottom layer first
        self.cube_bottom_left.set_pose(position=self._goal_pos["cube_bottom_left"], quat=q)
        self.cube_bottom_right.set_pose(position=self._goal_pos["cube_bottom_right"], quat=q)
        for _ in range(50):
            self._mojo.step()

        # Then top layer
        self.cube_top_left.set_pose(position=self._goal_pos["cube_top_left"], quat=q)
        self.cube_top_right.set_pose(position=self._goal_pos["cube_top_right"], quat=q)
        for _ in range(50):
            self._mojo.step()

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
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        for cube, color in zip(self.cubes, self._cube_colors):
            cube.set_color(color)

        # 3. Place objects in RANDOM *unsolved* placements within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._BASE_Z

        # Keep cubes separated and away from goal locations to avoid accidental success.
        min_pair_dist = 0.07
        min_goal_dist = 0.06

        goal_positions = np.array(list(self._goal_pos.values()))[:, :2]

        pts = []
        max_tries = 2000
        for _ in range(max_tries):
            if len(pts) == 4:
                break
            cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

            # not too close to any existing cube
            if pts and np.min([np.linalg.norm(cand - p) for p in pts]) < min_pair_dist:
                continue

            # not too close to any goal XY
            if np.min(np.linalg.norm(goal_positions - cand[None, :], axis=1)) < min_goal_dist:
                continue

            pts.append(cand)

        # Fallback (still unsolved): spread along Y at same Z (no stacking)
        if len(pts) < 4:
            ys = np.linspace(y_min + 0.05, y_max - 0.05, 4)
            rng.shuffle(ys)
            pts = [np.array([rng.uniform(x_min, x_max), float(ys[i])]) for i in range(4)]

        # Random yaw orientations; keep all on table (no stacking)
        for cube, p in zip(self.cubes, pts):
            yaw = rng.uniform(-np.pi, np.pi)
            q = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # [w,x,y,z]
            q = q  # to [x,y,z,w]
            cube.set_pose(position=np.array([p[0], p[1], z]), quat=q)

        # Let physics settle
        self._mojo.physics.forward()

        # 4. Extra settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_released(self) -> bool:
        return not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

    def _is_stacked_on(
        self,
        top: ColoredCube,
        bottom: ColoredCube,
        tol_xy: float = 0.03,
    ) -> bool:
        top_p = top.body.get_position()
        bot_p = bottom.body.get_position()

        xy_ok = np.linalg.norm(top_p[:2] - bot_p[:2]) <= tol_xy
        z_ok = abs((top_p[2] - bot_p[2]) - self._CUBE_DZ) <= 0.03

        # Collision-based sanity checks (avoid "stacked while still on table")
        colliding_ok = top.is_colliding(bottom)
        not_on_table = not top.is_colliding(self.table.table_top)

        return bool(xy_ok and z_ok and colliding_ok and not_on_table)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: the task says "build a 2-by-2 wall from four colored cubes"
        # and never says which colour belongs where. Binding the roles to
        # cube_bottom_left/…_top_right fixed the answer to red and blue on the
        # bottom with green on red and yellow on blue, so a wall built with a
        # different colour in each slot was failed for the colours alone.
        # The two cubes left low are the bases, the two raised are the tops, and
        # left is simply whichever of a pair sits further left.
        by_height = sorted(self.cubes, key=lambda c: float(c.body.get_position()[2]))
        bases = sorted(by_height[:2], key=lambda c: float(c.body.get_position()[1]), reverse=True)
        tops = sorted(by_height[2:], key=lambda c: float(c.body.get_position()[1]), reverse=True)
        base_left, base_right = bases
        top_left, top_right = tops

        bl = base_left.body.get_position()
        br = base_right.body.get_position()
        tl = top_left.body.get_position()
        tr = top_right.body.get_position()

        # Goal conditions (per task spec)
        # +Y is LEFT. Adjacent, not merely ordered.
        bottom_dy = float(bl[1] - br[1])
        left_of_ok = self._LEFT_OF_MIN_DY < bottom_dy <= self._LEFT_OF_MAX_DY
        bottom_x_aligned_ok = abs(bl[0] - br[0]) <= self._X_ALIGN_TOL

        # Each raised cube must rest on a base, and on a DIFFERENT one, so a single
        # three-high tower plus a stray cube cannot pass as a wall.
        straight = (self._is_stacked_on(top_left, base_left, tol_xy=self._XY_TOL)
                    and self._is_stacked_on(top_right, base_right, tol_xy=self._XY_TOL))
        crossed = (self._is_stacked_on(top_left, base_right, tol_xy=self._XY_TOL)
                   and self._is_stacked_on(top_right, base_left, tol_xy=self._XY_TOL))
        top_left_on_ok = top_right_on_ok = bool(straight or crossed)

        top_x_aligned_ok = abs(tl[0] - tr[0]) <= self._X_ALIGN_TOL

        positions_ok = bool(
            left_of_ok
            and bottom_x_aligned_ok
            and top_left_on_ok
            and top_right_on_ok
            and top_x_aligned_ok
        )

        # No semantic constraints for this task (colors are incidental)
        semantic_ok = True

        released = self._is_released()

        # Subgoal staging (approximate: treat "grasp" as implied by successful placement)
        # Indices 0..7 correspond to orders 1..8 in the spec.
        bottom_left_placed = base_left.is_colliding(self.table.table_top)
        bottom_right_placed = base_right.is_colliding(self.table.table_top) and left_of_ok and bottom_x_aligned_ok
        top_left_stacked = top_left_on_ok
        top_right_stacked = top_right_on_ok

        if bottom_left_placed:
            self._metric_stage(0, True)  # grasp (implied)
            self._metric_stage(1, True)  # place
        if bottom_right_placed:
            self._metric_stage(2, True)
            self._metric_stage(3, True)
        if top_left_stacked:
            self._metric_stage(4, True)
            self._metric_stage(5, True)
        if top_right_stacked:
            self._metric_stage(6, True)
            self._metric_stage(7, True)

        # Success requires goal configuration + released
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "cube_bottom_left": float(np.linalg.norm(bl - self._goal_pos["cube_bottom_left"])),
            "cube_bottom_right": float(np.linalg.norm(br - self._goal_pos["cube_bottom_right"])),
            "cube_top_left": float(np.linalg.norm(tl - self._goal_pos["cube_top_left"])),
            "cube_top_right": float(np.linalg.norm(tr - self._goal_pos["cube_top_right"])),
        }
        pose_error = {
            # Position-only proxy (no specific orientation requirement here)
            k: v for k, v in target_distance.items()
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})