"""Arrange yellow cubes into a single-layer ring with an empty center."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class LargeFlatYellowCubeRingEmptyCenter(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange many yellow colored cubes into a single-layer large flat circle (ring)
    on the table with an empty middle: cubes form a closed loop around a clear
    central hole (Layers: 1).

    Sequential Sub-Goals:
        1. Grasp a yellow cube to start forming the ring.
        2. Place cubes around the circle perimeter to form a large single-layer ring.
        3. Adjust cube positions to close gaps so the ring reads as a continuous loop.
        4. Clear the middle so the center remains empty (no cubes placed inside the hole).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _N_CUBES = 16
    _CUBE_COLOR = "yellow"

    _BASE_Z = 0.97
    _TABLE_Z = 0.95

    _RING_CENTER_XY = np.array([0.55, 0.0], dtype=float)

    # Use a larger radius so adjacent cubes do not overlap (prevents settling explosions)
    _RING_RADIUS = 0.145

    # Success tolerances (looser than slot-matching to account for settling drift)
    _RADIAL_TOL = 0.03
    _NO_CUBE_WITHIN_RADIUS = 0.095
    _UPRIGHT_MAX_TILT_DEG = 15.0
    _Z_TOL = 0.03

    # Ring continuity via angular gap constraints (closed loop, no big missing arc)
    _MAX_ANG_GAP = 0.70
    _MIN_ANG_GAP = 0.10

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self.cubes = []
        for _ in range(self._N_CUBES):
            c = ColoredCube(self._mojo)
            c.set_color(self._CUBE_COLOR)
            self.cubes.append(c)

        self._slot_positions = self._compute_slot_positions()

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _compute_slot_positions(self):
        center_xy = self._RING_CENTER_XY
        slots_xyz = []
        for k in range(self._N_CUBES):
            ang = 2.0 * np.pi * float(k) / float(self._N_CUBES)
            xy = center_xy + self._RING_RADIUS * np.array([np.cos(ang), np.sin(ang)], dtype=float)
            slots_xyz.append(np.array([float(xy[0]), float(xy[1]), float(self._BASE_Z)], dtype=float))
        return slots_xyz

    def _is_upright(self, quat_wxyz: np.ndarray) -> bool:
        try:
            q = Quaternion(quat_wxyz)
        except Exception:
            return False
        up_world = q.rotate(np.array([0.0, 0.0, 1.0], dtype=float))
        up_world = up_world / (np.linalg.norm(up_world) + 1e-9)
        cosang = float(np.clip(np.dot(up_world, np.array([0.0, 0.0, 1.0], dtype=float)), -1.0, 1.0))
        max_tilt_rad = np.deg2rad(self._UPRIGHT_MAX_TILT_DEG)
        return cosang >= float(np.cos(max_tilt_rad))

    def _on_reset(self):
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        for cube in self.cubes:
            cube.set_color(self._CUBE_COLOR)

        # Random UNSOLVED initialization within workspace bounds.
        # Guarantee failure by placing at least one cube inside the "hole" radius.
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z = float(self._BASE_Z)

        placed_xy = []

        def _sample_xy_in_bounds():
            return np.array([float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))], dtype=float)

        def _too_close(xy, others, min_dist):
            for o in others:
                if float(np.linalg.norm(xy - o)) < float(min_dist):
                    return True
            return False

        # Place one cube definitely inside the center hole (so _success() is False after reset).
        # Keep it within bounds and not exactly at center to reduce stacking/overlap.
        center_xy = self._RING_CENTER_XY.copy()
        for _ in range(200):
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            rad = float(rng.uniform(0.0, 0.02))  # well within _NO_CUBE_WITHIN_RADIUS
            xy0 = center_xy + rad * np.array([np.cos(ang), np.sin(ang)], dtype=float)
            if x_low <= xy0[0] <= x_high and y_low <= xy0[1] <= y_high:
                placed_xy.append(xy0)
                break
        else:
            # Fallback (should not happen): clamp center into bounds
            xy0 = np.array(
                [float(np.clip(center_xy[0], x_low, x_high)), float(np.clip(center_xy[1], y_low, y_high))], dtype=float
            )
            placed_xy.append(xy0)

        # Place remaining cubes randomly with a modest separation to avoid heavy overlap.
        min_sep = 0.06
        while len(placed_xy) < self._N_CUBES:
            xy = _sample_xy_in_bounds()
            # Avoid placing too close to other cubes; also avoid accidentally forming the ring
            # by discouraging points near the target radius.
            radial = float(np.linalg.norm(xy - center_xy))
            near_ring = abs(radial - float(self._RING_RADIUS)) <= float(self._RADIAL_TOL * 0.75)
            if near_ring:
                continue
            if _too_close(xy, placed_xy, min_sep):
                continue
            placed_xy.append(xy)

        # Apply poses with random yaw (upright orientation).
        for cube, xy in zip(self.cubes, placed_xy):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(position=np.array([float(xy[0]), float(xy[1]), z], dtype=float), quat=q.elements)

        for _ in range(50):
            self._mojo.step()

        for idx in range(4):
            self._metric_stage(idx, False)

        # To visualize the solved configuration, replace the randomized placement above with:
        # --- GOAL STATE ---
        # for cube, pos in zip(self.cubes, self._slot_positions):
        #     yaw = float(rng.uniform(-np.pi, np.pi))
        #     q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
        #     cube.set_pose(position=pos.copy(), quat=q.elements)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "_slot_positions") or self._slot_positions is None:
            return False
        if not self.cubes or len(self.cubes) != self._N_CUBES:
            return False

        color_ok = all(c.get_color() == self._CUBE_COLOR for c in self.cubes)

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        quats = np.array([c.body.get_quaternion() for c in self.cubes], dtype=float)

        upright_ok = all(self._is_upright(q) for q in quats)

        z = positions[:, 2]
        z_ok = bool(np.all(np.abs(z - self._BASE_Z) <= self._Z_TOL))
        z_range_ok = bool((float(np.max(z)) - float(np.min(z))) <= (self._Z_TOL * 2.0))
        table_contact_ok = all(c.is_colliding(self.table.table_top) for c in self.cubes)
        single_layer_ok = bool(z_ok and z_range_ok and (table_contact_ok or bool(np.all(z >= 0.94))))

        center_xy = self._RING_CENTER_XY
        rel_xy = positions[:, :2] - center_xy[None, :]
        radial = np.linalg.norm(rel_xy, axis=1)

        hole_clear_ok = bool(np.all(radial >= self._NO_CUBE_WITHIN_RADIUS))

        radial_err = np.abs(radial - float(self._RING_RADIUS))
        radial_ok_count = int(np.sum(radial_err <= self._RADIAL_TOL))
        radial_ok = radial_ok_count == self._N_CUBES
        mean_radial_ok = bool(abs(float(np.mean(radial)) - float(self._RING_RADIUS)) <= self._RADIAL_TOL)

        angles = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
        angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
        angles_sorted = np.sort(angles)
        if len(angles_sorted) >= 2:
            gaps = np.diff(angles_sorted)
            gaps = np.concatenate([gaps, np.array([angles_sorted[0] + 2.0 * np.pi - angles_sorted[-1]], dtype=float)])
            max_gap = float(np.max(gaps))
            min_gap = float(np.min(gaps))
        else:
            max_gap = float("inf")
            min_gap = 0.0

        ring_continuous_ok = bool((max_gap <= self._MAX_ANG_GAP) and (min_gap >= self._MIN_ANG_GAP))

        self._metric_stage(0, radial_ok_count >= 1)
        self._metric_stage(1, radial_ok_count >= (self._N_CUBES // 2))
        self._metric_stage(2, bool(radial_ok and mean_radial_ok and ring_continuous_ok))
        self._metric_stage(3, bool(radial_ok and mean_radial_ok and ring_continuous_ok and hole_clear_ok))

        self._success_check = bool(
            color_ok
            and released
            and upright_ok
            and single_layer_ok
            and radial_ok
            and mean_radial_ok
            and ring_continuous_ok
            and hole_clear_ok
        )

        mean_radial_error = float(np.mean(radial_err)) if len(radial_err) > 0 else float("inf")
        max_radial_error = float(np.max(radial_err)) if len(radial_err) > 0 else float("inf")
        min_hole_clearance = float(np.min(radial) - self._NO_CUBE_WITHIN_RADIUS) if len(radial) > 0 else float("-inf")

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_radial_error_m": mean_radial_error,
                "max_radial_error_m": max_radial_error,
                "max_angular_gap_rad": float(max_gap),
                "min_hole_clearance_m": float(min_hole_clearance),
            },
            pose_error={
                "mean_radial_error_m": mean_radial_error,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
