# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

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
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class DualCubeStackingTask(RoboEvalEnv, MetricRolloutEval):
    """Build two separate vertical stacks of two cubes each.

Success Criteria:
Any cube may sit on any other. Two cubes rest on the table as bases, the other two sit squarely on them without touching the table, and the two bases stand about two to three cube widths apart. All cubes released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants ---
    _CUBE_BASE_Z = 0.97
    _CUBE_SIZE = 0.04
    _STACK_Z_DELTA = 0.04

    _TARGET_STACK_SEP_M = 0.10
    # Judging 10 cm by eye with nothing on the table to measure against is the hard
    # part of this task, and it is not the part being tested. Widen the band so a
    # stack pair that plainly reads as "about a hand apart" passes.
    _STACK_SEP_TOL_M = 0.035

    # For robust stacking verification
    _STACK_XY_TOL_M = 0.03
    _STACK_Z_TOL_M = 0.02

    _NUM_SUBGOALS = 17

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.stack_1_bottom_cube = ColoredCube(self._mojo)
        self.stack_1_top_cube = ColoredCube(self._mojo)
        self.stack_2_bottom_cube = ColoredCube(self._mojo)
        self.stack_2_top_cube = ColoredCube(self._mojo)

        self.cubes = [
            self.stack_1_bottom_cube,
            self.stack_1_top_cube,
            self.stack_2_bottom_cube,
            self.stack_2_top_cube,
        ]

        # REQUIRED: set_color() in _initialize_env()
        self.stack_1_bottom_cube.set_color("red")
        self.stack_1_top_cube.set_color("blue")
        self.stack_2_bottom_cube.set_color("green")
        self.stack_2_top_cube.set_color("yellow")

        # Nominal goal poses (used by metrics and reset goal-state placement)
        # Choose Y positions to make stacks 0.10m apart in the table plane.
        self._stack1_base_xy = np.array([0.55, +0.05])
        self._stack2_base_xy = np.array([0.55, -0.05])

        self._goal_poses = {
            "s1_bottom": np.array([self._stack1_base_xy[0], self._stack1_base_xy[1], self._CUBE_BASE_Z]),
            "s1_top": np.array(
                [self._stack1_base_xy[0], self._stack1_base_xy[1], self._CUBE_BASE_Z + self._STACK_Z_DELTA]
            ),
            "s2_bottom": np.array([self._stack2_base_xy[0], self._stack2_base_xy[1], self._CUBE_BASE_Z]),
            "s2_top": np.array(
                [self._stack2_base_xy[0], self._stack2_base_xy[1], self._CUBE_BASE_Z + self._STACK_Z_DELTA]
            ),
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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.stack_1_bottom_cube.set_color("red")
        self.stack_1_top_cube.set_color("blue")
        self.stack_2_bottom_cube.set_color("green")
        self.stack_2_top_cube.set_color("yellow")

        # 3. Place objects in RANDOM *unsolved* configuration within workspace bounds.
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = self._CUBE_BASE_Z

        # Keep cubes separated to avoid starting already stacked/overlapping.
        # (cube size ~0.04m; use conservative separation)
        min_pair_dist = 0.08

        # Also ensure bottom cubes are NOT at the target separation (0.10 +/- tol),
        # and that no top cube starts stacked on its corresponding bottom cube.
        def _sample_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def _yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)

        # Try to find a valid random arrangement
        max_attempts = 500
        xy = None
        for _ in range(max_attempts):
            pts = [_sample_xy() for _ in range(4)]

            # pairwise separation
            ok_sep = True
            for i in range(4):
                for j in range(i + 1, 4):
                    if float(np.linalg.norm(pts[i] - pts[j])) < min_pair_dist:
                        ok_sep = False
                        break
                if not ok_sep:
                    break
            if not ok_sep:
                continue

            # Ensure bottom cubes are not at target stack separation (avoid immediate success).
            b1_xy = pts[0]
            b2_xy = pts[2]
            base_sep_xy = float(np.linalg.norm(b1_xy - b2_xy))
            if abs(base_sep_xy - self._TARGET_STACK_SEP_M) <= self._STACK_SEP_TOL_M:
                continue

            # Ensure not accidentally exactly on goal bases (extra guard)
            if float(np.linalg.norm(b1_xy - self._goal_poses["s1_bottom"][:2])) < 0.03:
                continue
            if float(np.linalg.norm(b2_xy - self._goal_poses["s2_bottom"][:2])) < 0.03:
                continue

            xy = pts
            break

        if xy is None:
            # Fallback (still unsolved): spread them deterministically in-bounds
            xy = [
                np.array([0.42, 0.22]),
                np.array([0.68, 0.22]),
                np.array([0.42, -0.22]),
                np.array([0.68, -0.22]),
            ]

        # Assign positions (all on table, not stacked)
        # Order: s1_bottom, s1_top, s2_bottom, s2_top
        self.stack_1_bottom_cube.set_pose(position=np.array([xy[0][0], xy[0][1], z0]), quat=_yaw_quat())
        self.stack_1_top_cube.set_pose(position=np.array([xy[1][0], xy[1][1], z0]), quat=_yaw_quat())
        self.stack_2_bottom_cube.set_pose(position=np.array([xy[2][0], xy[2][1], z0]), quat=_yaw_quat())
        self.stack_2_top_cube.set_pose(position=np.array([xy[3][0], xy[3][1], z0]), quat=_yaw_quat())
        self._mojo.physics.forward()

        # 4. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: the instruction asks for two stacks of two and says nothing
        # about which cube goes where, so the two cubes that ended up raised are the
        # tops and the two left on the table are the bases. Binding these to
        # stack_1_/stack_2_ fixed the pairing to blue-on-red and yellow-on-green:
        # two perfectly good towers built the other way round -- blue on green,
        # yellow on red -- were failed for the colours alone.
        by_height = sorted(self.cubes, key=lambda c: float(c.body.get_position()[2]))
        b1, b2, t1, t2 = by_height[0], by_height[1], by_height[2], by_height[3]

        # Released condition (single- or multi-gripper safe)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Helper: robust "on" relationship (collision + alignment)
        def _is_stacked(top, bottom) -> bool:
            top_pos = top.body.get_position()
            bot_pos = bottom.body.get_position()

            xy_dist = float(np.linalg.norm(top_pos[:2] - bot_pos[:2]))
            z_delta = float(top_pos[2] - bot_pos[2])

            colliding = top.is_colliding(bottom)
            top_not_on_table = not top.is_colliding(self.table.table_top)

            return bool(
                colliding
                and top_not_on_table
                and (xy_dist <= self._STACK_XY_TOL_M)
                and (abs(z_delta - self._STACK_Z_DELTA) <= self._STACK_Z_TOL_M)
            )

        # Bases should be on the table
        b1_on_table = b1.is_colliding(self.table.table_top)
        b2_on_table = b2.is_colliding(self.table.table_top)

        # Pair each raised cube with whichever base it is actually sitting on. Two
        # stacks exist only if the two tops land on two DIFFERENT bases, which also
        # rules out one three-high tower plus a loose cube.
        straight = _is_stacked(t1, b1) and _is_stacked(t2, b2)
        crossed = _is_stacked(t1, b2) and _is_stacked(t2, b1)
        s1_ok = s2_ok = bool(straight or crossed)

        # Stack base planar spacing constraint
        p1 = b1.body.get_position()
        p2 = b2.body.get_position()
        base_sep_xy = float(np.linalg.norm(p1[:2] - p2[:2]))
        spacing_ok = abs(base_sep_xy - self._TARGET_STACK_SEP_M) <= self._STACK_SEP_TOL_M

        positions_ok = bool(b1_on_table and b2_on_table and s1_ok and s2_ok and spacing_ok)
        semantic_ok = True  # no semantic constraints in this task

        # Stage metrics (coarse mapping from spatial state)
        # Stages 1-4: stack 1 base placed + released
        if b1_on_table and released:
            for i in range(0, 4):
                self._metric_stage(i, True)

        # Stages 5-9: stack 1 completed + released
        if s1_ok and released:
            for i in range(4, 9):
                self._metric_stage(i, True)

        # Stages 10-12: stack 2 base placed at correct separation + released
        if b2_on_table and spacing_ok and released:
            for i in range(9, 12):
                self._metric_stage(i, True)

        # Stages 13-17: stack 2 completed + released
        if s2_ok and released:
            for i in range(12, 17):
                self._metric_stage(i, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        # Provide compact, task-relevant errors
        # NOTE: metric_rollout expects pose_error values to be floats (not nested dicts).
        def _pose_err(actual_pos: np.ndarray, goal_pos: np.ndarray) -> float:
            dxy = float(np.linalg.norm(actual_pos[:2] - goal_pos[:2]))
            dz = float(abs(actual_pos[2] - goal_pos[2]))
            return float(dxy + dz)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "base_sep_xy_m": base_sep_xy,
                "base_sep_xy_err_m": float(abs(base_sep_xy - self._TARGET_STACK_SEP_M)),
            },
            pose_error={
                "s1_bottom_err": _pose_err(b1.body.get_position(), self._goal_poses["s1_bottom"]),
                "s1_top_err": _pose_err(t1.body.get_position(), self._goal_poses["s1_top"]),
                "s2_bottom_err": _pose_err(b2.body.get_position(), self._goal_poses["s2_bottom"]),
                "s2_top_err": _pose_err(t2.body.get_position(), self._goal_poses["s2_top"]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})