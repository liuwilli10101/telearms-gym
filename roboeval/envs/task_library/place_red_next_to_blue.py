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

class RedCubeAdjacentPlacement(RoboEvalEnv, MetricRolloutEval):
    """Place the red cube directly adjacent to the blue cube with side faces touching.

Success Criteria:
- The red and blue cubes are upright on the table at expected height.
- They are side-adjacent by contact or tight side-touch distance along one axis, with the other axis aligned and Z values matching.
- No gripper is holding either cube."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Geometric constants for ColoredCube ---
    _CUBE_SIZE = 0.04
    _TABLE_Z = 0.97  # ColoredCube resting Z on table (given)
    _ADJ_TOL = 0.014  # task tolerance (given)
    _ALIGN_TOL = 0.012  # tighter tolerance for the orthogonal axis alignment
    _Z_TOL = 0.01
    _UPRIGHT_ANGLE_TOL_RAD = np.deg2rad(12.0)

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY allowed asset: ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

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

        # 2. Deterministic RNG (required pattern; validation pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() == False after reset)
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._TABLE_Z

        def rand_pose_xy():
            pos = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            quat = np.array([q.x, q.y, q.z, q.w], dtype=float)  # x,y,z,w
            return pos, quat

        # Sample until not in (or near) the goal side-touch configuration and not overlapping
        max_tries = 2000
        blue_pos, blue_quat = rand_pose_xy()
        red_pos, red_quat = rand_pose_xy()

        for _ in range(max_tries):
            blue_pos, blue_quat = rand_pose_xy()
            red_pos, red_quat = rand_pose_xy()

            dx = float(red_pos[0] - blue_pos[0])
            dy = float(red_pos[1] - blue_pos[1])
            dist_xy = float(np.linalg.norm([dx, dy]))

            # Reject overlaps / near-overlaps
            if dist_xy < (self._CUBE_SIZE * 0.9):
                continue

            # Reject goal-like adjacency (matches _success() tolerances)
            x_side_touch = (abs(abs(dx) - self._CUBE_SIZE) <= self._ADJ_TOL) and (abs(dy) <= self._ALIGN_TOL)
            y_side_touch = (abs(abs(dy) - self._CUBE_SIZE) <= self._ADJ_TOL) and (abs(dx) <= self._ALIGN_TOL)
            if x_side_touch or y_side_touch:
                continue

            break
        else:
            # Fallback: force a clearly unsolved configuration within bounds
            blue_pos = np.array([0.55, 0.00, z], dtype=float)
            red_pos = np.array([0.55, 0.20, z], dtype=float)
            blue_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            red_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        # REQUIRED: set_color() in _on_reset
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.blue_cube.set_pose(position=blue_pos, quat=blue_quat)
        self.red_cube.set_pose(position=red_pos, quat=red_quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

        # (Optional) use rng to avoid "unused variable" lint in some pipelines
        _ = rng.random()

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_upright(self, cube: ColoredCube) -> bool:
        """Upright = cube's local +Z axis aligned with world +Z (yaw allowed)."""
        q = Quaternion(cube.body.get_quaternion())
        R = q.rotation_matrix
        local_z_world = R[:, 2]  # third column is local +Z expressed in world
        cosang = float(np.clip(np.dot(local_z_world, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        return ang <= self._UPRIGHT_ANGLE_TOL_RAD

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red_p = self.red_cube.body.get_position()
        blue_p = self.blue_cube.body.get_position()

        dx = float(red_p[0] - blue_p[0])
        dy = float(red_p[1] - blue_p[1])
        dz = float(red_p[2] - blue_p[2])

        # --- Goal checks: "side_touching_adjacent", "any_side_face", both upright on table ---
        # On-table (contact + reasonable height)
        red_on_table = self.red_cube.is_colliding(self.table.table_top) and abs(float(red_p[2] - self._TABLE_Z)) <= self._Z_TOL
        blue_on_table = self.blue_cube.is_colliding(self.table.table_top) and abs(float(blue_p[2] - self._TABLE_Z)) <= self._Z_TOL

        upright_ok = self._is_upright(self.red_cube) and self._is_upright(self.blue_cube)

        # Adjacent with side faces touching: one axis offset ~ cube size, orthogonal axis ~ 0, heights match
        dz_ok = abs(dz) <= self._Z_TOL
        x_side_touch = (abs(abs(dx) - self._CUBE_SIZE) <= self._ADJ_TOL) and (abs(dy) <= self._ALIGN_TOL)
        y_side_touch = (abs(abs(dy) - self._CUBE_SIZE) <= self._ADJ_TOL) and (abs(dx) <= self._ALIGN_TOL)

        # Require physical contact OR sufficiently close (some engines can be slightly non-colliding at tolerance edge)
        colliding = self.red_cube.is_colliding(self.blue_cube)
        close_enough_for_contact = (x_side_touch or y_side_touch) and dz_ok

        positions_ok = bool((colliding or close_enough_for_contact) and (x_side_touch or y_side_touch) and dz_ok and red_on_table and blue_on_table and upright_ok)

        # No grasp-state in goal; but require released for evaluation stability
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in [self.red_cube, self.blue_cube]
            for s in self.robot.grippers
        )

        # Subgoal staging (mark all when final arrangement achieved)
        if positions_ok:
            for i in range(6):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and released)

        # Metrics
        center_dist_xy = float(np.linalg.norm((red_p - blue_p)[:2]))
        # "touch error" = min deviation from target side-touch configuration
        touch_err = float(
            min(
                abs(abs(dx) - self._CUBE_SIZE) + abs(dy),
                abs(abs(dy) - self._CUBE_SIZE) + abs(dx),
            )
        )
        pose_error = {
            "red_blue_center_dist_xy": center_dist_xy,
            "side_touch_alignment_error": touch_err,
            "dz_abs": abs(dz),
        }
        target_distance = {
            "target_center_offset": self._CUBE_SIZE,
            "actual_center_dist_xy": center_dist_xy,
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})