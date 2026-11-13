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

class CubeRingArrangementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Reposition five colored cubes to form an evenly spaced circular ring centered on the
    workspace center, creating a regular pentagon pattern on the tabletop.

    Sequential Sub-Goals:
        1. Grasp a cube to begin relocating it to the ring.
        2. Lift the grasped cube clear of other objects/table for safe transport.
        3. Move the cube toward an available position on the circular ring around the workspace center.
        4. Place the cube down at a ring vertex location to help form an evenly spaced circle.
        5. Release the cube after placement and prepare to adjust the next cube.
        6. Fine-adjust cube positions so all cubes share the same radius from center and equal spacing.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from specification)
    _CENTER_XY = np.array([0.55, 0.0])
    _N_CUBES = 5
    _NOMINAL_RADIUS_M = 0.12
    _RADIAL_TOL_M = 0.015
    _ANGULAR_SPACING_DEG = 72.0
    _ANGULAR_TOL_DEG = 10.0
    _MAX_HEIGHT_ABOVE_TABLE_M = 0.03
    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._N_CUBES)]

        # Assign fixed, distinct colors (required: call set_color() in _initialize_env)
        self._cube_colors = ["red", "blue", "green", "yellow", "purple"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Precompute goal poses (regular pentagon ring about workspace center)
        self._goal_positions = self._compute_ring_positions(
            center_xy=self._CENTER_XY,
            radius=self._NOMINAL_RADIUS_M,
            n=self._N_CUBES,
            start_angle_rad=0.0,
            z=self._CUBE_REST_Z,
        )

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

        # 2. Deterministic RNG (required)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset as well
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM UNSOLVED positions within workspace bounds.
        #    Make them *not* satisfy the ring: keep radii far from nominal by clustering near center.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = self._CUBE_REST_Z

        center_xy = self._CENTER_XY.astype(float).copy()
        cluster_radius = 0.04  # ensures |r - 0.12| >= ~0.08 > radial tol, so success will be False

        # Simple rejection sampling to avoid severe overlaps while staying in-bounds.
        placed_positions = []
        min_sep = 0.045
        max_tries = 500

        for cube in self.cubes:
            pos = None
            for _ in range(max_tries):
                # sample in a small disk around center
                ang = rng.uniform(0.0, 2.0 * np.pi)
                rad = cluster_radius * np.sqrt(rng.uniform(0.0, 1.0))
                x = float(center_xy[0] + rad * np.cos(ang))
                y = float(center_xy[1] + rad * np.sin(ang))

                # enforce workspace bounds
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    continue

                candidate = np.array([x, y, z0], dtype=float)

                # enforce minimum separation from already placed cubes
                ok = True
                for p in placed_positions:
                    if float(np.linalg.norm(candidate[:2] - p[:2])) < min_sep:
                        ok = False
                        break
                if not ok:
                    continue

                pos = candidate
                break

            # Fallback (still unsolved) if rejection fails
            if pos is None:
                x = float(np.clip(center_xy[0] + rng.uniform(-cluster_radius, cluster_radius), x_min, x_max))
                y = float(np.clip(center_xy[1] + rng.uniform(-cluster_radius, cluster_radius), y_min, y_max))
                pos = np.array([x, y, z0], dtype=float)

            placed_positions.append(pos)

            # random yaw about z; quaternion [x,y,z,w]
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
            cube.set_pose(position=pos.copy(), quat=quat)

        # Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _compute_ring_positions(center_xy: np.ndarray, radius: float, n: int, start_angle_rad: float, z: float):
        positions = []
        for i in range(n):
            theta = start_angle_rad + i * (2.0 * np.pi / n)
            x = float(center_xy[0] + radius * np.cos(theta))
            y = float(center_xy[1] + radius * np.sin(theta))
            positions.append(np.array([x, y, z], dtype=float))
        return positions

    @staticmethod
    def _angle_wrap_0_2pi(a: np.ndarray) -> np.ndarray:
        return np.mod(a, 2.0 * np.pi)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Collect positions ---
        center_xy = self._CENTER_XY
        positions = [c.body.get_position() for c in self.cubes]

        # --- Table / height constraints ---
        # Must be on/near table (colliding table top) and not lifted above threshold.
        on_table = True
        height_ok = True
        max_height_above = 0.0
        for p, c in zip(positions, self.cubes):
            # Reliable table contact check: use table.table_top
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break
            height_above = float(p[2] - self._CUBE_REST_Z)
            max_height_above = max(max_height_above, height_above)
            if height_above > self._MAX_HEIGHT_ABOVE_TABLE_M:
                height_ok = False
                break

        # --- Radial ring constraint ---
        radial_errors = []
        radial_ok = True
        angles = []
        for p in positions:
            dx, dy = float(p[0] - center_xy[0]), float(p[1] - center_xy[1])
            r = float(np.sqrt(dx * dx + dy * dy))
            radial_errors.append(abs(r - self._NOMINAL_RADIUS_M))
            if abs(r - self._NOMINAL_RADIUS_M) > self._RADIAL_TOL_M:
                radial_ok = False
            angles.append(np.arctan2(dy, dx))

        # --- Angular spacing constraint (regular pentagon, rotation-invariant) ---
        angles = self._angle_wrap_0_2pi(np.array(angles, dtype=float))
        angles_sorted = np.sort(angles)
        angles_aug = np.concatenate([angles_sorted, np.array([angles_sorted[0] + 2.0 * np.pi])])
        diffs = np.diff(angles_aug)  # radians
        diffs_deg = diffs * (180.0 / np.pi)
        spacing_errors_deg = np.abs(diffs_deg - self._ANGULAR_SPACING_DEG)

        angular_ok = bool(np.all(spacing_errors_deg <= self._ANGULAR_TOL_DEG))

        # --- Released constraint ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        positions_ok = bool(on_table and height_ok and radial_ok and angular_ok)

        # No semantic constraints in this task (colors do not affect goal)
        semantic_ok = True

        # Subgoals: treat success as completion of all stages (simple non-temporal staging)
        if positions_ok:
            for i in range(6):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "radial_mean_abs_error_m": float(np.mean(radial_errors)) if len(radial_errors) else 0.0,
                "radial_max_abs_error_m": float(np.max(radial_errors)) if len(radial_errors) else 0.0,
                "spacing_mean_abs_error_deg": float(np.mean(spacing_errors_deg)) if len(spacing_errors_deg) else 0.0,
                "spacing_max_abs_error_deg": float(np.max(spacing_errors_deg)) if len(spacing_errors_deg) else 0.0,
                "max_height_above_table_m": float(max_height_above),
            },
            pose_error={
                # Orientation is irrelevant for this task; report 0 for completeness.
                "mean_orientation_error_rad": 0.0
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})