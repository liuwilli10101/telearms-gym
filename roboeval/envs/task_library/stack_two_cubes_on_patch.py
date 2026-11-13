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


class RedOnBlueStackingTask(RoboEvalEnv, MetricRolloutEval):
    """Build a red-on-blue stack with the blue base on the green patch.

Success Criteria:
- The blue cube is on and near the center of the green patch and supported on the table.
- The red cube is stacked on blue with contact, XY alignment, expected Z gap, and no table contact.
- The cube colors are red and blue, and no gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_Z_ON_TABLE = 0.97
    _PATCH_Z_ON_TABLE = 0.951
    _CUBE_SIZE = 0.04
    _STACK_DZ = 0.04

    _XY_ON_TARGET_TOL = 0.03
    _XY_STACK_TOL = 0.02
    _Z_STACK_TOL = 0.02

    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Props from preset
        self.table = self._preset.get_props(Table)[0]

        # Task objects (ONLY allowed assets)
        self.blue_cube = ColoredCube(self._mojo)
        self.red_cube = ColoredCube(self._mojo)
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

        self.cubes = [self.blue_cube, self.red_cube]

        # Metrics (FULL required signature)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _physics_forward_if_available(self):
        # Some backends require an explicit forward() after teleports.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        # Prefer collision direction: BLOCK vs PATCH.collider
        if hasattr(target, "collider") and (target.collider is not None):
            if bool(block.is_colliding(target.collider)):
                return True

        # Geometric fallback (GoalPatch collision can be unreliable due to thin geom)
        if hasattr(target, "body") and hasattr(block, "body"):
            tpos = target.body.get_position()
            bpos = block.body.get_position()
            xy_err = np.linalg.norm((bpos - tpos)[:2])
            z_err = abs(bpos[2] - self._CUBE_Z_ON_TABLE)
            if (xy_err <= max(self._XY_ON_TARGET_TOL, 0.055)) and (z_err <= 0.03):
                return True

        # Last-resort fallback
        return bool(block.is_colliding(target))

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

        # 2. Deterministic RNG (pipeline may apply additional randomization later)
        rng = np.random.default_rng(self.seed)  # noqa: F841

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.blue_cube.set_color("blue")
        self.red_cube.set_color("red")

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        def _rand_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

        # Randomize patch within bounds
        patch_xy = _rand_xy()
        if hasattr(self.green_goal_patch, "body"):
            self.green_goal_patch.body.set_position(
                np.array([patch_xy[0], patch_xy[1], self._PATCH_Z_ON_TABLE])
            )
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        # Randomize blue cube away from patch so blue_on_patch/near_patch are False
        # Use a margin comfortably larger than both tol and patch half-size.
        min_blue_patch_xy_dist = 0.10
        for _ in range(200):
            blue_xy = _rand_xy()
            if np.linalg.norm(blue_xy - patch_xy) >= min_blue_patch_xy_dist:
                break
        else:
            blue_xy = np.clip(patch_xy + np.array([min_blue_patch_xy_dist, 0.0]), [x_min, y_min], [x_max, y_max])

        self.blue_cube.set_pose(
            position=np.array([blue_xy[0], blue_xy[1], self._CUBE_Z_ON_TABLE]),
            quat=np.array([0, 0, 0, 1]),
        )
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        # Randomize red cube on the table, not stacked, and not aligned with blue
        min_red_blue_xy_dist = 0.10
        for _ in range(200):
            red_xy = _rand_xy()
            if np.linalg.norm(red_xy - blue_xy) >= min_red_blue_xy_dist:
                break
        else:
            red_xy = np.clip(blue_xy + np.array([0.0, min_red_blue_xy_dist]), [x_min, y_min], [x_max, y_max])

        # Keep red on table (not stacked) to guarantee stacked=False
        self.red_cube.set_pose(
            position=np.array([red_xy[0], red_xy[1], self._CUBE_Z_ON_TABLE]),
            quat=np.array([0, 0, 0, 1]),
        )
        self._physics_forward_if_available()
        self._mojo.physics.forward()

        # Let contacts settle
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks (_success() can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Positions
        blue_pos = self.blue_cube.body.get_position()
        red_pos = self.red_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        blue_xy_err = np.linalg.norm((blue_pos - patch_pos)[:2])
        red_blue_xy_err = np.linalg.norm((red_pos - blue_pos)[:2])
        stack_dz_err = abs((red_pos[2] - blue_pos[2]) - self._STACK_DZ)

        # Goal condition 1: blue on green goal patch
        blue_on_patch = self._is_block_on_target(self.blue_cube, self.green_goal_patch)
        blue_near_patch_xy = blue_xy_err <= self._XY_ON_TARGET_TOL
        blue_on_table_col = (
            bool(self.blue_cube.is_colliding(self.table.table_top)) if hasattr(self.table, "table_top") else False
        )
        blue_on_table_z = abs(blue_pos[2] - self._CUBE_Z_ON_TABLE) <= 0.03
        blue_on_table = bool(blue_on_table_col or blue_on_table_z)

        # Goal condition 2: red on blue (stacked)
        red_collides_blue = bool(self.red_cube.is_colliding(self.blue_cube))
        red_not_on_table = not bool(self.red_cube.is_colliding(self.table.table_top))
        stacked = bool(
            red_collides_blue
            and red_not_on_table
            and (red_blue_xy_err <= self._XY_STACK_TOL)
            and (stack_dz_err <= self._Z_STACK_TOL)
        )

        # Ensure correct cube identities via color (robustness)
        color_ok = (self.blue_cube.get_color() == "blue") and (self.red_cube.get_color() == "red")

        # Released (not a success criterion in spec, but required in template)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(blue_on_patch and blue_near_patch_xy and blue_on_table and stacked)
        semantic_ok = bool(color_ok)

        # Subgoal staging (metrics only; not used as success criteria)
        holding_blue = any(
            self.robot.is_gripper_holding_object(self.blue_cube, s) for s in self.robot.grippers
        )
        holding_red = any(
            self.robot.is_gripper_holding_object(self.red_cube, s) for s in self.robot.grippers
        )

        # 1) grasp blue
        if holding_blue:
            self._metric_stage(0, True)

        # 2) transport blue near patch
        if blue_xy_err <= 0.08:
            self._metric_stage(1, True)

        # 3) place blue on patch
        if blue_on_patch and blue_on_table:
            self._metric_stage(2, True)

        # 4) release blue
        if (blue_on_patch and blue_on_table) and (not holding_blue):
            self._metric_stage(3, True)

        # 5) grasp red
        if holding_red:
            self._metric_stage(4, True)

        # 6) lift red above blue top face
        if red_pos[2] >= blue_pos[2] + self._STACK_DZ + 0.02:
            self._metric_stage(5, True)

        # 7) align red above blue
        if red_blue_xy_err <= self._XY_STACK_TOL:
            self._metric_stage(6, True)

        # 8) place red on blue
        if stacked:
            self._metric_stage(7, True)

        # 9) release red after stacking
        if stacked and (not holding_red):
            self._metric_stage(8, True)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "blue_to_patch_xy": float(blue_xy_err),
                "red_to_blue_xy": float(red_blue_xy_err),
                "red_to_blue_stack_dz_err": float(stack_dz_err),
            },
            pose_error={
                "blue_z_err": float(abs(blue_pos[2] - self._CUBE_Z_ON_TABLE)),
                "red_z_err_vs_stack": float(stack_dz_err),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})