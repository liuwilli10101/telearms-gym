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


class YellowCubeOnGreenTargetTask(RoboEvalEnv, MetricRolloutEval):
    """
    Pick up the yellow colored cube and place it onto the green target patch on the table.
    The task is successful when the cube is positioned on the marked green placement area.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the yellow cube (either hand).
        2. Lift the yellow cube off the table surface.
        3. Move the yellow cube over the green goal patch.
        4. Lower the yellow cube onto the green target patch.
        5. Release the cube while keeping it positioned on the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False

    _success_check = True
    _final_metrics: Dict = {}

    # Geometry constants (meters)
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    _CUBE_HALF = 0.02  # 4cm cube
    _PATCH_HALF_X = 0.05  # 10cm patch
    _PATCH_HALF_Y = 0.05
    _XY_TOL = 0.014  # task tolerance
    _LIFT_Z_THRESH = 0.99

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY allowed assets) ---
        self.yellow_cube = ColoredCube(self._mojo)
        self.yellow_cube.set_color("yellow")  # REQUIRED in _initialize_env

        self.green_goal_patch = GoalPatch(
            self._mojo, color="green", size=(0.1, 0.1), name="target"
        )

        # Keep a canonical list for metric plumbing / generic loops
        self.cubes = [self.yellow_cube]

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

    def _maybe_forward_physics(self):
        """Best-effort 'forward' call as required by some pipelines; fallback to stepping."""
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        else:
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization handle (even if unused here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED in _on_reset
        self.yellow_cube.set_color("yellow")

        # 3. Randomize objects into UNSOLVED positions within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Random patch pose (fixed z)
        patch_xy = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
        )
        self.green_goal_patch.body.set_position(
            np.array([patch_xy[0], patch_xy[1], self._PATCH_REST_Z], dtype=float)
        )
        self._maybe_forward_physics()

        # Random cube pose (fixed z), guaranteed NOT on patch footprint
        # Ensure cube center is outside the "within patch" bounds used by _cube_within_patch_footprint()
        sep_x = (self._PATCH_HALF_X - self._CUBE_HALF + self._XY_TOL) + 0.02
        sep_y = (self._PATCH_HALF_Y - self._CUBE_HALF + self._XY_TOL) + 0.02

        cube_xy = None
        for _ in range(200):
            cand = np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
            )
            dx = float(cand[0] - patch_xy[0])
            dy = float(cand[1] - patch_xy[1])
            if (abs(dx) > sep_x) or (abs(dy) > sep_y):
                cube_xy = cand
                break
        if cube_xy is None:
            # Fallback: force an offset while clamping to bounds
            cube_xy = patch_xy + np.array([np.sign(rng.standard_normal()) * sep_x, 0.0])
            cube_xy[0] = float(np.clip(cube_xy[0], x_min, x_max))
            cube_xy[1] = float(np.clip(cube_xy[1], y_min, y_max))

        # Random yaw about z
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # (w, x, y, z)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)  # to (x,y,z,w)

        self.yellow_cube.set_pose(
            position=np.array([cube_xy[0], cube_xy[1], self._CUBE_REST_Z], dtype=float),
            quat=quat,
        )
        self._maybe_forward_physics()

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _cube_within_patch_footprint(self) -> bool:
        """Geometric footprint containment check (robust vs thin patch collisions)."""
        cube_pos = self.yellow_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        dx = float(cube_pos[0] - patch_pos[0])
        dy = float(cube_pos[1] - patch_pos[1])

        # Cube must fit entirely within patch bounds (allow small tolerance)
        x_ok = abs(dx) <= (self._PATCH_HALF_X - self._CUBE_HALF + self._XY_TOL)
        y_ok = abs(dy) <= (self._PATCH_HALF_Y - self._CUBE_HALF + self._XY_TOL)

        # Resting on/near table height (avoid "on patch" while floating)
        z_ok = abs(float(cube_pos[2]) - self._CUBE_REST_Z) <= 0.03

        return bool(x_ok and y_ok and z_ok)

    def _is_cube_on_patch(self) -> bool:
        """Use the reliable collider-direction pattern when available, plus footprint check."""
        on_patch_by_geom = self._cube_within_patch_footprint()

        on_patch_by_collision = False
        if hasattr(self.green_goal_patch, "collider"):
            on_patch_by_collision = bool(
                self.yellow_cube.is_colliding(self.green_goal_patch.collider)
            )

        # Accept either (geometry is the primary; collision is an additional confirmation)
        return bool(on_patch_by_geom or on_patch_by_collision)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "yellow_cube") or not hasattr(self, "green_goal_patch"):
            return False

        # Semantic/color verification (required for ColoredCube tasks involving color)
        color_ok = (self.yellow_cube.get_color() == "yellow")

        # Released gripper (generic over grippers; works for single/dual-arm setups)
        released = not any(
            self.robot.is_gripper_holding_object(self.yellow_cube, g)
            for g in self.robot.grippers
        )
        holding = not released

        cube_pos = self.yellow_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        lifted = bool(float(cube_pos[2]) > self._LIFT_Z_THRESH)
        over_patch_xy = bool(
            abs(float(cube_pos[0] - patch_pos[0])) <= (self._PATCH_HALF_X + 0.02)
            and abs(float(cube_pos[1] - patch_pos[1])) <= (self._PATCH_HALF_Y + 0.02)
        )

        on_patch = self._is_cube_on_patch()

        # Subgoals (0..4)
        self._metric_stage(0, bool(holding))  # grasp
        self._metric_stage(1, bool(holding and lifted))  # lift
        self._metric_stage(2, bool(holding and lifted and over_patch_xy))  # transport
        self._metric_stage(3, bool(on_patch))  # place (contact/pose)
        self._metric_stage(4, bool(on_patch and released))  # release

        positions_ok = bool(on_patch)
        semantic_ok = bool(color_ok)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # Target distance: XY distance to patch center (primary)
        dxy = float(np.linalg.norm((cube_pos[:2] - patch_pos[:2]), ord=2))
        dz = float(abs(cube_pos[2] - self._CUBE_REST_Z))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy": dxy,
            },
            pose_error={
                "xy_center_error": dxy,
                "z_error": dz,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})