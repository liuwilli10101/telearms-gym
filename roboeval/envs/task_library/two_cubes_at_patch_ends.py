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


class DualCubeSidePlacement(RoboEvalEnv, MetricRolloutEval):
    """Put red against the left edge of the green patch and blue against the right edge.

Success Criteria:
Each cube centre must be within 3 cm of the drawn edge on its own side -- inside, on, or just outside it all count -- and anywhere along that edge, flat on the table and released."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry (meters)
    # GoalPatch reads size as HALF-extents, so this draws a 0.20 m square and the
    # visible Y edge is at 0.10, not 0.05. Halving it again put the goal 5 cm inside
    # the patch: a cube flush against the edge the operator can see sat 5 cm past
    # the accepted band and failed. _PATCH_SIZE is already the half-extent.
    _PATCH_SIZE = (0.1, 0.1)
    _PATCH_HALF_X = _PATCH_SIZE[0]  # 0.10, half the length of each side edge
    _PATCH_HALF_Y = _PATCH_SIZE[1]  # 0.10, the drawn edge
    _CUBE_SIZE = 0.04
    _CUBE_HALF = 0.5 * _CUBE_SIZE  # 0.02

    # Target the DRAWN EDGE, not a particular side of it. A cube "beside the left
    # edge" can be flush on the inside (centre at patch_half - cube_half = 0.08),
    # sitting on the line (0.10) or flush on the outside (0.12). The edge is a line
    # with no thickness, so those three read identically to the operator; aiming the
    # band at 0.12 accepted only the outside one and rejected a cube placed neatly
    # against the edge from within the patch by a single centimetre.
    _EDGE_OFFSET_Y = _PATCH_HALF_Y  # 0.10, the drawn edge itself

    # Success tolerances
    # Wide enough to span flush-inside (0.08) through flush-outside (0.12) with a
    # centimetre of teleoperation slop at each end.
    _EDGE_TOL = 0.03
    # Anywhere ALONG the edge counts. The side edges are 20 cm long, and the task is
    # "put the cube against this edge" -- not "against the middle of it". Requiring
    # the cube within 5 cm of the patch centre in X quietly threw away the outer half
    # of both edges and failed cubes sitting neatly against the far end of one.
    _X_SPAN_TOL = _PATCH_HALF_X  # 0.10: the cube centre lies somewhere on the edge

    _NUM_SUBGOALS = 8

    def _physics_forward(self):
        # Some backends expose physics.forward(); fall back gracefully.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Required assets only: GoalPatch + ColoredCube
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="target")

        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Initialize metrics with FULL signature (REQUIRED)
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

        # 2. Deterministic RNG (required pattern; may be used by validation randomization)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() returns False)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        cube_z = 0.97
        patch_z = 0.951

        # Random patch pose (within bounds, with a small margin to keep cubes in-bounds)
        patch_x = float(rng.uniform(0.45, 0.65))
        patch_y = float(rng.uniform(-0.10, 0.10))
        self.green_goal_patch.body.set_position(np.array([patch_x, patch_y, patch_z]))
        self._physics_forward()

        # Random cube poses, but intentionally on the WRONG sides:
        #   red cube on RIGHT (-Y) of patch, blue cube on LEFT (+Y) of patch
        # This guarantees the relative-side checks in _success() fail.
        for _ in range(200):
            dy = float(rng.uniform(0.06, 0.20))  # separated from patch center

            red_y = patch_y - dy
            blue_y = patch_y + dy
            if not (y_min <= red_y <= y_max and y_min <= blue_y <= y_max):
                continue

            red_x = float(np.clip(patch_x + rng.uniform(-0.10, 0.10), x_min, x_max))
            blue_x = float(np.clip(patch_x + rng.uniform(-0.10, 0.10), x_min, x_max))

            # Keep cubes separated from each other
            if np.hypot(red_x - blue_x, red_y - blue_y) < 0.06:
                continue

            self.red_cube.set_color("red")
            self.red_cube.set_pose(position=np.array([red_x, red_y, cube_z]), quat=np.array([1.0, 0.0, 0.0, 0.0]))
            self._physics_forward()

            self.blue_cube.set_color("blue")
            self.blue_cube.set_pose(position=np.array([blue_x, blue_y, cube_z]), quat=np.array([1.0, 0.0, 0.0, 0.0]))
            self._physics_forward()
            break

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _edge_touch_errors(self):
        """Return (red_edge_err, blue_edge_err, red_x_err, blue_x_err)."""
        patch_pos = self.green_goal_patch.body.get_position()
        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        # Desired offsets from patch centre along Y: the drawn edge on each side.
        desired_red_dy = +self._EDGE_OFFSET_Y
        desired_blue_dy = -self._EDGE_OFFSET_Y

        red_edge_err = abs((red_pos[1] - patch_pos[1]) - desired_red_dy)
        blue_edge_err = abs((blue_pos[1] - patch_pos[1]) - desired_blue_dy)

        red_x_err = abs(red_pos[0] - patch_pos[0])
        blue_x_err = abs(blue_pos[0] - patch_pos[0])

        return red_edge_err, blue_edge_err, red_x_err, blue_x_err

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "green_goal_patch"):
            return False

        patch_pos = self.green_goal_patch.body.get_position()
        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        red_edge_err, blue_edge_err, red_x_err, blue_x_err = self._edge_touch_errors()

        # Relative side checks (left = +Y, right = -Y)
        red_left_of = (red_pos[1] > patch_pos[1])
        blue_right_of = (blue_pos[1] < patch_pos[1])

        # Flush contact checks (geometric)
        red_touch_left_edge = (red_edge_err <= self._EDGE_TOL) and (red_x_err <= self._X_SPAN_TOL)
        blue_touch_right_edge = (blue_edge_err <= self._EDGE_TOL) and (blue_x_err <= self._X_SPAN_TOL)

        # Ensure cubes are on the table (stable placement)
        # Collision checks can be flaky immediately after reset, so allow a Z-height fallback.
        red_on_table = self.red_cube.is_colliding(self.table.table_top)
        blue_on_table = self.blue_cube.is_colliding(self.table.table_top)
        if not red_on_table:
            red_on_table = abs(float(red_pos[2]) - 0.97) <= 0.03
        if not blue_on_table:
            blue_on_table = abs(float(blue_pos[2]) - 0.97) <= 0.03
        on_table = bool(red_on_table and blue_on_table)

        positions_ok = bool(red_left_of and blue_right_of and red_touch_left_edge and blue_touch_right_edge and on_table)

        # No semantic content beyond color (colors are fixed, but we still ensure they are correct)
        semantic_ok = (self.red_cube.get_color() == "red") and (self.blue_cube.get_color() == "blue")

        # Released condition: neither cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_edge_error_m": float(red_edge_err),
                "blue_edge_error_m": float(blue_edge_err),
            },
            pose_error={
                "red_x_align_error_m": float(red_x_err),
                "blue_x_align_error_m": float(blue_x_err),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})