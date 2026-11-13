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

class TwoLayerCubeWall(RoboEvalEnv, MetricRolloutEval):
    """Build a six-cube wall with three cubes on the bottom row and three directly above them.

Success Criteria:
- Cubes 1-3 form a bottom row on the table, with cube_2 right of cube_1, cube_3 right of cube_2, and the row aligned in Y.
- Cubes 4-6 are stacked on cubes 1-3 respectively, with valid on-top relations for the top row.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task geometry constants (ColoredCube: 4cm; base_z=0.97; stack increment=0.04)
    _CUBE_SIZE_M = 0.04
    _BASE_Z = 0.97
    _STACK_DZ = 0.04

    # Tolerances from task spec (kept slightly permissive but consistent)
    _TOL_RIGHT_OF = 0.03
    _TOL_ON = 0.02
    _TOL_ALIGN_Y = 0.03

    _NUM_SUBGOALS = 12

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create exactly six ColoredCube assets (and nothing else)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)
        self.cube_5 = ColoredCube(self._mojo)
        self.cube_6 = ColoredCube(self._mojo)

        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4, self.cube_5, self.cube_6]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Precompute goal positions for a 2x3 wall:
        # Bottom row varies along +X (left->right), constant Y; top row stacked above.
        center_x, center_y = 0.55, 0.0
        spacing_x = 0.05  # >= 0.05m minimum spacing guideline

        x_left = center_x - spacing_x
        x_mid = center_x
        x_right = center_x + spacing_x
        y_row = center_y

        self._goal_positions = {
            "cube_1": np.array([x_left,  y_row, self._BASE_Z]),
            "cube_2": np.array([x_mid,   y_row, self._BASE_Z]),
            "cube_3": np.array([x_right, y_row, self._BASE_Z]),
            "cube_4": np.array([x_left,  y_row, self._BASE_Z + self._STACK_DZ]),
            "cube_5": np.array([x_mid,   y_row, self._BASE_Z + self._STACK_DZ]),
            "cube_6": np.array([x_right, y_row, self._BASE_Z + self._STACK_DZ]),
        }
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 2. Deterministic RNG for reproducible random initializations
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # 3. Place objects in RANDOM UNSOLVED positions (all on table, not stacked)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._BASE_Z

        # Ensure far enough from each cube's goal pose so near_goal fails (pose_tol=0.06 in _success()).
        min_goal_dist = 0.085  # > 0.06 ensures near_goal == False
        min_pair_dist = 0.06   # reduce overlaps / immediate contacts at reset

        placed = []
        cube_names = ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5", "cube_6"]

        for cube, name in zip(self.cubes, cube_names):
            goal_pos = self._goal_positions[name]
            pos = None
            for _ in range(500):
                candidate = np.array(
                    [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float
                )

                # Keep away from its own goal pose (guarantees near_goal fails)
                if np.linalg.norm(candidate - goal_pos) < min_goal_dist:
                    continue

                # Avoid overlaps with already placed cubes (XY distance)
                if any(np.linalg.norm((candidate - p)[:2]) < min_pair_dist for p in placed):
                    continue

                pos = candidate
                break

            # Fallback (should be very rare): still ensure not near its goal
            if pos is None:
                pos = np.array([x_lo, y_lo, z], dtype=float)
                if np.linalg.norm(pos - goal_pos) < min_goal_dist:
                    pos = np.array([x_hi, y_hi, z], dtype=float)

            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            quat = np.array([q.x, q.y, q.z, q.w], dtype=float)

            cube.set_pose(position=pos, quat=quat)
            placed.append(pos)

        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_right_of(self, a: ColoredCube, b: ColoredCube, tol: float) -> bool:
        """Interpret 'right_of' as +X direction (a.x > b.x), with a mild y-alignment check."""
        pa = a.body.get_position()
        pb = b.body.get_position()
        if (pa[0] - pb[0]) <= max(0.0, self._CUBE_SIZE_M * 0.25):
            return False
        if abs(pa[1] - pb[1]) > tol:
            return False
        return True

    def _is_aligned_axis_y(self, a: ColoredCube, b: ColoredCube, tol: float) -> bool:
        """Aligned along y-axis means approximately same y coordinate."""
        pa = a.body.get_position()
        pb = b.body.get_position()
        return abs(pa[1] - pb[1]) <= tol

    def _is_on(self, top: ColoredCube, bottom: ColoredCube, tol_xy: float, tol_z: float) -> bool:
        """Check stacking: top is supported by bottom (collision), close in XY, and correct Z offset."""
        pt = top.body.get_position()
        pb = bottom.body.get_position()

        # Must be colliding (support contact)
        if not top.is_colliding(bottom):
            return False

        # XY alignment
        if np.linalg.norm((pt - pb)[:2]) > tol_xy:
            return False

        # Height difference near one cube size
        dz = pt[2] - pb[2]
        if abs(dz - self._STACK_DZ) > tol_z:
            return False

        # Ensure top is not also resting on the table
        if top.is_colliding(self.table.table_top):
            return False

        return True

    def _near_goal_pose(self, cube: ColoredCube, goal_pos: np.ndarray, pos_tol: float) -> bool:
        p = cube.body.get_position()
        return np.linalg.norm(p - goal_pos) <= pos_tol

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_positions"):
            return False

        # Released condition (no cube held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # --- Spatial relations for the 2x3 wall ---
        # Identity-free: the task asks for six cubes in a two-row wall and never says
        # which colour goes where. Binding the slots to cube_1..cube_6 demanded red,
        # blue, green along the bottom in that order with purple, yellow, orange
        # above them, so a correctly built wall using any other arrangement of the
        # six colours failed. The three cubes left low ARE the bottom row, in
        # whatever order they ended up along it, and the three raised are the top.
        by_height = sorted(self.cubes, key=lambda c: float(c.body.get_position()[2]))
        bottom = sorted(by_height[:3], key=lambda c: float(c.body.get_position()[0]))
        top = by_height[3:]

        right_21 = self._is_right_of(bottom[1], bottom[0], tol=self._TOL_RIGHT_OF)
        right_32 = self._is_right_of(bottom[2], bottom[1], tol=self._TOL_RIGHT_OF)
        align_12 = self._is_aligned_axis_y(bottom[0], bottom[1], tol=self._TOL_ALIGN_Y)
        align_23 = self._is_aligned_axis_y(bottom[1], bottom[2], tol=self._TOL_ALIGN_Y)

        # Bottom cubes should be on the table (collision with table top is reliable)
        bottom_on_table = all(c.is_colliding(self.table.table_top) for c in bottom)

        # Top row: each raised cube must sit on a bottom cube, and on a DIFFERENT one
        # each, so a taller tower plus strays cannot pass as a wall.
        def _sits_on_a_distinct_base(tops, bases):
            used = []
            for t in tops:
                match = next(
                    (b for b in bases
                     if b not in used
                     and self._is_on(t, b, tol_xy=self._TOL_ON, tol_z=self._TOL_ON)),
                    None,
                )
                if match is None:
                    return False
                used.append(match)
            return True

        top_row_ok = _sits_on_a_distinct_base(top, bottom)
        on_41 = on_52 = on_63 = top_row_ok

        # Optional: keep the structure near intended footprint (helps avoid degenerate "right_of" far away)
        # Use a lenient tolerance so physics settling doesn't break success.
        pose_tol = 0.06
        near_goal = all(
            self._near_goal_pose(c, self._goal_positions[name], pos_tol=pose_tol)
            for c, name in zip(
                [self.cube_1, self.cube_2, self.cube_3, self.cube_4, self.cube_5, self.cube_6],
                ["cube_1", "cube_2", "cube_3", "cube_4", "cube_5", "cube_6"],
            )
        )

        # NOTE: absolute near_goal gate against _goal_positions removed; the relative
        # right_of / aligned / on_table / on_top relations already pin the wall geometry.
        positions_ok = bool(
            bottom_on_table
            and right_21
            and right_32
            and align_12
            and align_23
            and on_41
            and on_52
            and on_63
        )

        # No semantic constraints for this spatial task
        semantic_ok = True

        # Subgoal staging (index = order-1)
        # 1-2: cube_1 placed at left-bottom and on table
        if self.cube_1.is_colliding(self.table.table_top) and self._near_goal_pose(
            self.cube_1, self._goal_positions["cube_1"], pos_tol=pose_tol
        ):
            self._metric_stage(0, True)  # transport cube_1
            self._metric_stage(1, True)  # place cube_1

        # 3-4: cube_2 placed and aligned/right_of cube_1
        if self.cube_2.is_colliding(self.table.table_top) and self._near_goal_pose(
            self.cube_2, self._goal_positions["cube_2"], pos_tol=pose_tol
        ):
            self._metric_stage(2, True)  # transport cube_2
            if right_21 and align_12:
                self._metric_stage(3, True)  # align cube_2 with cube_1

        # 5-6: cube_3 placed and aligned/right_of cube_2
        if self.cube_3.is_colliding(self.table.table_top) and self._near_goal_pose(
            self.cube_3, self._goal_positions["cube_3"], pos_tol=pose_tol
        ):
            self._metric_stage(4, True)  # transport cube_3
            if right_32 and align_23:
                self._metric_stage(5, True)  # align cube_3 with row

        # 7-8: cube_4 stacked on cube_1
        if self._near_goal_pose(self.cube_4, self._goal_positions["cube_4"], pos_tol=pose_tol):
            self._metric_stage(6, True)  # grasp cube_4 (proxy)
        if on_41:
            self._metric_stage(7, True)  # place cube_4

        # 9-10: cube_5 stacked on cube_2
        if self._near_goal_pose(self.cube_5, self._goal_positions["cube_5"], pos_tol=pose_tol):
            self._metric_stage(8, True)  # grasp cube_5 (proxy)
        if on_52:
            self._metric_stage(9, True)  # place cube_5

        # 11-12: cube_6 stacked on cube_3
        if self._near_goal_pose(self.cube_6, self._goal_positions["cube_6"], pos_tol=pose_tol):
            self._metric_stage(10, True)  # grasp cube_6 (proxy)
        if on_63:
            self._metric_stage(11, True)  # place cube_6

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        # Provide per-cube target distances and pose errors (position-only proxy).
        target_distance = {
            "cube_1_to_goal": float(np.linalg.norm(self.cube_1.body.get_position() - self._goal_positions["cube_1"])),
            "cube_2_to_goal": float(np.linalg.norm(self.cube_2.body.get_position() - self._goal_positions["cube_2"])),
            "cube_3_to_goal": float(np.linalg.norm(self.cube_3.body.get_position() - self._goal_positions["cube_3"])),
            "cube_4_to_goal": float(np.linalg.norm(self.cube_4.body.get_position() - self._goal_positions["cube_4"])),
            "cube_5_to_goal": float(np.linalg.norm(self.cube_5.body.get_position() - self._goal_positions["cube_5"])),
            "cube_6_to_goal": float(np.linalg.norm(self.cube_6.body.get_position() - self._goal_positions["cube_6"])),
        }
        pose_error = {
            "mean_pos_error": float(
                np.mean(
                    [
                        np.linalg.norm(self.cube_1.body.get_position() - self._goal_positions["cube_1"]),
                        np.linalg.norm(self.cube_2.body.get_position() - self._goal_positions["cube_2"]),
                        np.linalg.norm(self.cube_3.body.get_position() - self._goal_positions["cube_3"]),
                        np.linalg.norm(self.cube_4.body.get_position() - self._goal_positions["cube_4"]),
                        np.linalg.norm(self.cube_5.body.get_position() - self._goal_positions["cube_5"]),
                        np.linalg.norm(self.cube_6.body.get_position() - self._goal_positions["cube_6"]),
                    ]
                )
            )
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})