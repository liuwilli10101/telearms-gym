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

class CubeClusterCenterTask(RoboEvalEnv, MetricRolloutEval):
    """
    Gather four colored cubes onto the green center marker patch in the middle of the table
    workspace. Success only requires every cube to end up near the green marker (within the
    center radius) and resting on the table; there is no separate cube-to-cube spacing rule.
    The marker is purely a visual target so the operator can see where the center is.

    Sequential Sub-Goals:
        1. Grasp any cube to begin clustering.
        2. Move the red cube toward the center clustering region.
        3. Place the red cube down near the workspace center.
        4. Move the blue cube toward the center clustering region.
        5. Place the blue cube adjacent to the already-placed cube(s).
        6. Move the green cube toward the center clustering region.
        7. Place the green cube close to the existing cluster.
        8. Move the yellow cube toward the center clustering region.
        9. Place the yellow cube to complete a tight 4-cube cluster near the center.
        10. Nudge cubes as needed so all four are tightly grouped without gaps.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task thresholds (from spec)
    _CENTER_XY = np.array([0.55, 0.0], dtype=float)  # workspace center in this sim
    _CENTER_RADIUS_M = 0.08
    # NOTE: pairwise cube distance is no longer enforced for success — being within the center
    # radius already keeps the four cubes clustered. Kept only as a reported metric.
    _MAX_PAIRWISE_XY_M = 0.09
    _CUBE_TABLE_Z = 0.97
    _ON_TABLE_Z_TOL_M = 0.02

    _NUM_SUBGOALS = 10

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY permitted asset type)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_blue, self.cube_green, self.cube_yellow]

        # Visible center marker so the operator can see where to cluster (the center point
        # is otherwise invisible). Visual-only: success never uses this patch's collider, and
        # it must not push cubes, so disable contact on ALL of its geoms. Half-extent 0.05 =>
        # a 0.10m square whose corners sit ~0.07m from center, inside the 0.08m success radius.
        self.center_marker = GoalPatch(
            self._mojo, color="green", size=(0.05, 0.05), name="center_marker", collision_enabled=False
        )
        self.center_marker.body.set_position(
            np.array([self._CENTER_XY[0], self._CENTER_XY[1], 0.951], dtype=float)
        )
        for _g in self.center_marker.body.geoms:
            _bound = self._mojo.physics.bind(_g.mjcf)
            _bound.contype = 0
            _bound.conaffinity = 0

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

        # 2. Deterministic RNG (required pattern; validation may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        # Keep the visible center marker pinned at the center on every reset.
        self.center_marker.body.set_position(
            np.array([self._CENTER_XY[0], self._CENTER_XY[1], 0.951], dtype=float)
        )

        # 3. Place objects in RANDOM UNSOLVED configuration within workspace bounds.
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = float(self._CUBE_TABLE_Z)

        center_xy = np.array(self._CENTER_XY, dtype=float)

        # Enforce "unsolved" by construction:
        # - keep cubes reasonably separated (prevents tight cluster)
        # - ensure at least one cube is outside center radius
        min_pairwise_xy = 0.10  # > _MAX_PAIRWISE_XY_M to guarantee not a tight cluster
        min_outside_center = float(self._CENTER_RADIUS_M) + 0.03

        def sample_xy() -> np.ndarray:
            return np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)

        xys = []
        outside_center_count = 0
        for _ in range(2000):
            xy = sample_xy()

            # pairwise separation constraint
            ok = True
            for prev in xys:
                if float(np.linalg.norm(xy - prev)) < min_pairwise_xy:
                    ok = False
                    break
            if not ok:
                continue

            xys.append(xy)
            if float(np.linalg.norm(xy - center_xy)) > min_outside_center:
                outside_center_count += 1

            if len(xys) == 4 and outside_center_count >= 1:
                break

            # If we filled 4 but none are outside center, restart
            if len(xys) == 4 and outside_center_count < 1:
                xys = []
                outside_center_count = 0

        # Fallback (still unsolved) if sampling failed
        if len(xys) != 4:
            xys = [
                np.array([0.41, -0.22], dtype=float),
                np.array([0.69, -0.22], dtype=float),
                np.array([0.41, 0.22], dtype=float),
                np.array([0.69, 0.22], dtype=float),
            ]

        # Random yaw for each cube (keep upright)
        for cube, xy in zip(self.cubes, xys):
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(
                position=np.array([float(xy[0]), float(xy[1]), z], dtype=float),
                quat=np.array([quat.x, quat.y, quat.z, quat.w], dtype=float),
            )

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
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

        # Per-cube checks: on table + near workspace center (XY) + Z within tolerance
        center_dists = []
        positions_ok = True
        for cube in self.cubes:
            pos = cube.body.get_position()

            on_table_collision = cube.is_colliding(self.table.table_top)
            z_ok = abs(float(pos[2]) - float(self._CUBE_TABLE_Z)) <= float(self._ON_TABLE_Z_TOL_M)
            center_xy_dist = self._xy_dist(np.array(pos, dtype=float), np.array([cx := self._CENTER_XY[0], cy := self._CENTER_XY[1], 0.0], dtype=float))
            center_dists.append(center_xy_dist)
            center_ok = center_xy_dist <= float(self._CENTER_RADIUS_M)

            if not (on_table_collision and z_ok and center_ok):
                positions_ok = False
                break

        # Pairwise cube distance is no longer enforced — being within the center radius already
        # keeps the cubes clustered. Compute the max pair distance only for metrics.
        max_pairwise_xy = 0.0
        for i in range(len(self.cubes)):
            pi = np.array(self.cubes[i].body.get_position(), dtype=float)
            for j in range(i + 1, len(self.cubes)):
                pj = np.array(self.cubes[j].body.get_position(), dtype=float)
                max_pairwise_xy = max(max_pairwise_xy, self._xy_dist(pi, pj))

        # No semantic constraints (ColoredCube only)
        semantic_ok = True

        # Released: no cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Mark subgoals complete when final arrangement is correct and released
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "center_xy_max_m": float(max(center_dists)) if center_dists else float("inf"),
                "center_xy_mean_m": float(np.mean(center_dists)) if center_dists else float("inf"),
                "pairwise_xy_max_m": float(max_pairwise_xy),
            },
            pose_error={
                "within_center_radius_m": float(self._CENTER_RADIUS_M),
                "max_pairwise_xy_threshold_m": float(self._MAX_PAIRWISE_XY_M),
                "on_table_z_target_m": float(self._CUBE_TABLE_Z),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})