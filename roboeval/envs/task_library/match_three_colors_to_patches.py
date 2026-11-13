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


class ColorMatchedCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes (red, blue, green) onto the matching goal patches of the same color.
    The task is complete when each cube is positioned on its corresponding colored target zone.

    Sequential Sub-Goals:
        1. Place the red cube on the red goal patch.
        2. Place the blue cube on the blue goal patch.
        3. Place the green cube on the green goal patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_Z = 0.97
    _PATCH_Z = 0.951
    _DIST_TOL = 0.014  # tolerance hint from task constraints

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create cubes (REQUIRED assets only) ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube]

        # --- Create goal patches (REQUIRED assets only) ---
        self.red_goal_patch = GoalPatch(self._mojo, color="red", size=(0.1, 0.1), name="red_target")
        self.blue_goal_patch = GoalPatch(self._mojo, color="blue", size=(0.1, 0.1), name="blue_target")
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="green_target")
        self.patches = [self.red_goal_patch, self.blue_goal_patch, self.green_goal_patch]

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

        # 2. Deterministic RNG (required pattern; may be used by external randomization)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: Call set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")

        # 4. Place patches (fixed, within workspace)
        x_patch = 0.62
        patch_ys = [0.18, 0.0, -0.18]  # red, blue, green (within Y bounds)

        self.red_goal_patch.body.set_position(np.array([x_patch, patch_ys[0], self._PATCH_Z]))
        self.blue_goal_patch.body.set_position(np.array([x_patch, patch_ys[1], self._PATCH_Z]))
        self.green_goal_patch.body.set_position(np.array([x_patch, patch_ys[2], self._PATCH_Z]))

        # 5. Randomize cubes into UNSOLVED poses within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z

        patch_positions = [
            self.red_goal_patch.body.get_position().copy(),
            self.blue_goal_patch.body.get_position().copy(),
            self.green_goal_patch.body.get_position().copy(),
        ]
        for p in patch_positions:
            p[2] = z

        # Keep cubes away from their targets and from each other to avoid accidental success/collisions.
        min_to_any_patch_xy = 0.06
        min_cube_separation_xy = 0.06

        def sample_pose(existing_xy: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            for _ in range(500):
                pos = np.array(
                    [
                        rng.uniform(x_min, x_max),
                        rng.uniform(y_min, y_max),
                        z,
                    ],
                    dtype=float,
                )

                # far from ALL patches (stronger than just its own) => ensures _success() is False
                ok = True
                for pp in patch_positions:
                    if np.linalg.norm(pos[:2] - pp[:2]) <= min_to_any_patch_xy:
                        ok = False
                        break
                if not ok:
                    continue

                # separate cubes
                for ex in existing_xy:
                    if np.linalg.norm(pos[:2] - ex[:2]) <= min_cube_separation_xy:
                        ok = False
                        break
                if not ok:
                    continue

                # random yaw
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw).elements  # (w,x,y,z)
                quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)  # to (x,y,z,w)
                return pos, quat

            # Fallback (still unsolved): fixed corners
            pos = np.array([0.45, 0.22, z], dtype=float)
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            return pos, quat

        placed_xy: list[np.ndarray] = []
        red_pos, red_q = sample_pose(placed_xy)
        placed_xy.append(red_pos.copy())
        blue_pos, blue_q = sample_pose(placed_xy)
        placed_xy.append(blue_pos.copy())
        green_pos, green_q = sample_pose(placed_xy)
        placed_xy.append(green_pos.copy())

        self.red_cube.set_pose(position=red_pos, quat=red_q)
        self.blue_cube.set_pose(position=blue_pos, quat=blue_q)
        self.green_cube.set_pose(position=green_pos, quat=green_q)

        # 6. Physics settling
        self._mojo.physics.forward()

        # 7. Initialize subgoals to False (3 color-matched placements)
        for idx in range(3):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_distance(self, a_pos: np.ndarray, b_pos: np.ndarray) -> float:
        d = a_pos[:2] - b_pos[:2]
        return float(np.linalg.norm(d))

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """
        Reliable "on patch" check:
          - collision must be evaluated from the block side against target.collider
          - also require XY closeness as a guardrail
        """
        if not hasattr(target, "collider"):
            return False

        if not block.is_colliding(target.collider):
            return False

        bpos = block.body.get_position()
        tpos = target.body.get_position()
        if self._xy_distance(bpos, tpos) > self._DIST_TOL:
            return False

        # If floor exists, ensure the block isn't fallen off the table
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False

        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "red_goal_patch"):
            return False

        # --- Goal checks (positional/rotational only; no grasp-based success) ---
        on_red = self._is_block_on_target(self.red_cube, self.red_goal_patch)
        on_blue = self._is_block_on_target(self.blue_cube, self.blue_goal_patch)
        on_green = self._is_block_on_target(self.green_cube, self.green_goal_patch)
        positions_ok = bool(on_red and on_blue and on_green)

        # Semantic verification: ensure each specific cube still has the intended color
        semantic_ok = (
            self.red_cube.get_color() == "red"
            and self.blue_cube.get_color() == "blue"
            and self.green_cube.get_color() == "green"
        )

        # Released: none of the cubes are currently held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Metric staging (3 subgoals)
        if on_red and (self.red_cube.get_color() == "red") and not any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(0, True)
        if on_blue and (self.blue_cube.get_color() == "blue") and not any(
            self.robot.is_gripper_holding_object(self.blue_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(1, True)
        if on_green and (self.green_cube.get_color() == "green") and not any(
            self.robot.is_gripper_holding_object(self.green_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(2, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Distances for metrics
        d_red = self._xy_distance(self.red_cube.body.get_position(), self.red_goal_patch.body.get_position())
        d_blue = self._xy_distance(self.blue_cube.body.get_position(), self.blue_goal_patch.body.get_position())
        d_green = self._xy_distance(self.green_cube.body.get_position(), self.green_goal_patch.body.get_position())

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_cube_to_red_patch_xy": d_red,
                "blue_cube_to_blue_patch_xy": d_blue,
                "green_cube_to_green_patch_xy": d_green,
            },
            pose_error={
                "red_cube_xy_error": d_red,
                "blue_cube_xy_error": d_blue,
                "green_cube_xy_error": d_green,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})