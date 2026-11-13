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
class ThreeCubeWallRow(RoboEvalEnv, MetricRolloutEval):
    """Arrange red, green, and blue into a front-to-back wall row.

Success Criteria:
Red in front of green in front of blue along X with at least 3.5 cm between them, the three cubes within 5 cm of each other in Y, all flat on the table and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task tolerances / geometry (meters)
    _TABLE_Z_CUBE = 0.97
    # Y spread across ALL THREE cubes, not a pairwise offset: 2 cm meant the row had
    # to be straighter than the sibling two-cube tasks demand of a single pair, with
    # nothing on the table to line the cubes up against. A row this wide still plainly
    # reads as a wall.
    _TOL_Y = 0.05
    _TOL_Z = 0.02  # cubes are 4 cm tall, so this still rejects any stacking
    _MIN_DX = 0.035

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create exactly the required assets: 3 ColoredCubes
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]

        # Nominal goal poses (also used for metric reporting)
        # NOTE: Goal condition defines "in_front_of" along x as "lower x than".
        center_x, center_y = 0.55, 0.0
        spacing_x = 0.05  # stable separation (>= ~cube size) while remaining "side-by-side"
        self._goal_positions = {
            "cube_red": np.array([center_x - spacing_x, center_y, self._TABLE_Z_CUBE]),
            "cube_green": np.array([center_x, center_y, self._TABLE_Z_CUBE]),
            "cube_blue": np.array([center_x + spacing_x, center_y, self._TABLE_Z_CUBE]),
        }
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

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

    def _sample_initial_state(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """
        Sample an UNSOLVED initial state (used by downstream randomization pipelines).
        Positions are on the table within workspace bounds, and separated to avoid overlap.
        """
        # Workspace constraints
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25

        # Use utility sampler for well-separated planar points
        # NOTE: get_random_points_on_plane() does not accept an rng kwarg in this codebase.
        origin = np.array([(x_low + x_high) * 0.5, (y_low + y_high) * 0.5, self._TABLE_Z_CUBE])
        extents = np.array([(x_high - x_low) * 0.5, (y_high - y_low) * 0.5, 0.0])
        pts = get_random_points_on_plane(
            3,
            origin,
            extents,
            0.08,  # spacing (min separation)
        )

        return {
            "cube_red": np.array([pts[0][0], pts[0][1], self._TABLE_Z_CUBE]),
            "cube_green": np.array([pts[1][0], pts[1][1], self._TABLE_Z_CUBE]),
            "cube_blue": np.array([pts[2][0], pts[2][1], self._TABLE_Z_CUBE]),
        }

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

        # 2) Deterministic RNG handle (required pattern; used by external validators)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 3) Place objects in RANDOM, UNSOLVED positions (ensure _success() is False after reset)
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25

        # Resample until configuration is not accidentally in the solved row arrangement
        for _ in range(50):
            init = self._sample_initial_state(rng)

            r = init["cube_red"].copy()
            g = init["cube_green"].copy()
            b = init["cube_blue"].copy()

            # Compute (potential) success-critical predicates (quat ignored by _success)
            ys = np.array([r[1], g[1], b[1]])
            y_aligned = (np.max(ys) - np.min(ys)) <= self._TOL_Y
            order_ok = (r[0] < g[0] - self._MIN_DX) and (g[0] < b[0] - self._MIN_DX)

            # Force "unsolved": if it accidentally matches both row ordering and Y alignment,
            # break the ordering deterministically by swapping red/blue x,y (still within bounds).
            if order_ok and y_aligned:
                r, b = b, r
                order_ok = (r[0] < g[0] - self._MIN_DX) and (g[0] < b[0] - self._MIN_DX)

            # Additionally, make Y spread exceed tolerance if it is too small (keeps within bounds).
            ys = np.array([r[1], g[1], b[1]])
            if (np.max(ys) - np.min(ys)) <= self._TOL_Y:
                # Push one cube in Y by a safe offset, clipped to workspace bounds.
                idx = int(rng.integers(0, 3))
                offset = float(rng.choice([-1.0, 1.0]) * max(0.03, self._TOL_Y * 2.0))
                if idx == 0:
                    r[1] = float(np.clip(r[1] + offset, y_low, y_high))
                elif idx == 1:
                    g[1] = float(np.clip(g[1] + offset, y_low, y_high))
                else:
                    b[1] = float(np.clip(b[1] + offset, y_low, y_high))

            # Final quick check: avoid the exact solved conjunction
            ys = np.array([r[1], g[1], b[1]])
            y_aligned = (np.max(ys) - np.min(ys)) <= self._TOL_Y
            order_ok = (r[0] < g[0] - self._MIN_DX) and (g[0] < b[0] - self._MIN_DX)
            if not (order_ok and y_aligned):
                init["cube_red"] = r
                init["cube_green"] = g
                init["cube_blue"] = b
                break

        self.cube_red.set_pose(position=init["cube_red"], quat=self._goal_quat)
        self.cube_green.set_pose(position=init["cube_green"], quat=self._goal_quat)
        self.cube_blue.set_pose(position=init["cube_blue"], quat=self._goal_quat)

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks (_success can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red = self.cube_red.body.get_position()
        green = self.cube_green.body.get_position()
        blue = self.cube_blue.body.get_position()

        # 1) On-table (no stacking): all cubes should be supported by the table top
        # NOTE: use self.table.table_top for reliable table collision
        on_table = (
            self.cube_red.is_colliding(self.table.table_top)
            and self.cube_green.is_colliding(self.table.table_top)
            and self.cube_blue.is_colliding(self.table.table_top)
        )

        # 2) X-order constraint ("in_front_of" defined as lower x than target)
        order_ok = (red[0] < green[0] - self._MIN_DX) and (green[0] < blue[0] - self._MIN_DX)

        # 3) Collinearity along X: nearly same Y and Z for all cubes (no stacking / no skew)
        ys = np.array([red[1], green[1], blue[1]])
        zs = np.array([red[2], green[2], blue[2]])
        y_aligned = (np.max(ys) - np.min(ys)) <= self._TOL_Y
        z_aligned = (np.max(zs) - np.min(zs)) <= self._TOL_Z

        positions_ok = bool(on_table and order_ok and y_aligned and z_aligned)

        # No semantic constraints beyond fixed cube colors (already set)
        semantic_ok = True

        # 4) Released: no cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (simple, monotonic): only mark all true when final arrangement is correct
        if positions_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        goal_r = self._goal_positions["cube_red"]
        goal_g = self._goal_positions["cube_green"]
        goal_b = self._goal_positions["cube_blue"]

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "dx_red_green": float(abs((green[0] - red[0]) - (goal_g[0] - goal_r[0]))),
                "dx_green_blue": float(abs((blue[0] - green[0]) - (goal_b[0] - goal_g[0]))),
                "y_spread": float(np.max(ys) - np.min(ys)),
                "z_spread": float(np.max(zs) - np.min(zs)),
            },
            pose_error={
                "red_to_goal": float(np.linalg.norm(red - goal_r)),
                "green_to_goal": float(np.linalg.norm(green - goal_g)),
                "blue_to_goal": float(np.linalg.norm(blue - goal_b)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})