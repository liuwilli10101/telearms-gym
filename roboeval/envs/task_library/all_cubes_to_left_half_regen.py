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


class LeftHalfCubeRelocation(RoboEvalEnv, MetricRolloutEval):
    """
    Translate all colored cubes into the left half of the workspace (+Y direction)
    while preserving their relative spacing (i.e., the group moves together without
    changing its internal layout).

    Sequential Sub-Goals:
        1. Move all cubes into the left half: y >= 0.0 (axis index 1; +Y is LEFT).
        2. Preserve relative spacing: pairwise XY offsets remain (rigid translation only).
        3. Release all cubes (no cube held by any gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters
    _N_CUBES = 5
    _CUBE_Z = 0.97

    # Left-half definition along +Y
    _LEFT_Y_MIN = 0.0
    _LEFT_Y_MAX = 0.25

    # Rigid-translation preservation tolerance (meters) for pairwise XY offsets
    _PAIRWISE_XY_TOL = 0.02

    # Workspace bounds (for reset randomization helpers)
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    # For stable non-colliding layouts
    _MIN_CENTER_SPACING = 0.05

    # Metric stages
    _NUM_STAGES = 3

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset)
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._N_CUBES)]

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "green", "yellow", "purple"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # A simple "goal" layout entirely in left half (used for the solved pose on _on_reset)
        # (This does NOT define the task's success target uniquely; the task allows any global translation,
        #  but _on_reset must place a configuration that is already successful.)
        x0 = 0.55
        ys = [0.20, 0.15, 0.10, 0.05, 0.00]  # all within [0.0, 0.25]
        self._goal_positions = [np.array([x0, y, self._CUBE_Z], dtype=float) for y in ys]

        # Reference XY positions (captured from the episode's initial state)
        self._reference_xy = None
        self._reset_i = 0

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

        # 2. Deterministic RNG (vary across resets)
        rng = np.random.default_rng(int(self.seed) + int(getattr(self, "_reset_i", 0)))

        # 3. Place objects in RANDOM UNSOLVED positions within required bounds
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z = float(self._CUBE_Z)

        # MUST call set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Sample non-colliding XY positions; force at least one cube to start in right half (y < 0)
        xy = np.zeros((self._N_CUBES, 2), dtype=float)
        placed = 0
        max_tries = 2000

        # First cube: force unsolved by placing in right half (y < 0)
        xy[0, 0] = rng.uniform(x_low, x_high)
        xy[0, 1] = rng.uniform(y_low, min(-1e-3, y_high))

        placed = 1
        tries = 0
        while placed < self._N_CUBES and tries < max_tries:
            tries += 1
            cand = np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=float)
            d = np.linalg.norm(xy[:placed] - cand[None, :], axis=1)
            if np.all(d >= self._MIN_CENTER_SPACING):
                xy[placed] = cand
                placed += 1

        # Fallback if spacing sampling fails: deterministic spread that still keeps at least one y<0
        if placed < self._N_CUBES:
            base_x = 0.55
            base_y = -0.08
            rel = np.array(
                [
                    [0.00, 0.00],
                    [0.03, 0.06],
                    [-0.03, 0.06],
                    [0.03, 0.12],
                    [-0.03, 0.12],
                ],
                dtype=float,
            )
            xy = rel + np.array([base_x, base_y], dtype=float)
            xy[:, 0] = np.clip(xy[:, 0], x_low, x_high)
            xy[:, 1] = np.clip(xy[:, 1], y_low, y_high)
            # Ensure at least one cube remains in right half
            xy[0, 1] = min(xy[0, 1], -0.01)

        # Apply poses (single layer on table)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        for i, cube in enumerate(self.cubes):
            pos = np.array([xy[i, 0], xy[i, 1], z], dtype=float)
            cube.set_pose(position=pos, quat=quat)

        # Let physics settle
        self._mojo.physics.forward()

        # Reset reference so arrangement check is not trivially satisfied during reset-time _success()
        self._reference_xy = None
        self._reset_i = int(getattr(self, "_reset_i", 0)) + 1

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_STAGES):
            self._metric_stage(idx, False)

    def reset_object_poses(self):
        """
        Place cubes into an UNSOLVED but rigid-translation-consistent layout:
        - All cubes share a fixed internal arrangement (pairwise XY offsets fixed).
        - At least one cube starts in the right half (y < 0.0).
        - Positions remain within workspace bounds.

        This method is intentionally deterministic given self.seed.
        """
        rng = np.random.default_rng(self.seed)

        # Internal layout (relative offsets in XY): a straight line along Y with safe spacing.
        # This defines the "layout" that must be preserved via rigid translation.
        rel_xy = np.array(
            [
                [0.0, +0.10],
                [0.0, +0.05],
                [0.0, +0.00],
                [0.0, -0.05],
                [0.0, -0.10],
            ],
            dtype=float,
        )

        # Try sampling a global translation that ensures at least one cube is at y<0.
        # Also keep within the "initial-state" suggested bounds: x∈[0.40,0.70], y∈[-0.25,0.25].
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25

        found = False
        for _ in range(200):
            # Keep x centered; allow some variability
            x0 = rng.uniform(x_low + 0.05, x_high - 0.05)

            # Bias y so the group straddles the y=0 boundary (unsolved requirement)
            y0 = rng.uniform(-0.12, 0.08)

            xy = rel_xy + np.array([x0, y0], dtype=float)

            # Bounds check
            if np.any(xy[:, 0] < x_low) or np.any(xy[:, 0] > x_high):
                continue
            if np.any(xy[:, 1] < y_low) or np.any(xy[:, 1] > y_high):
                continue

            # Must have at least one cube in the right half (y < 0)
            if not np.any(xy[:, 1] < self._LEFT_Y_MIN):
                continue

            found = True
            break

        if not found:
            # Fallback: a simple straddling placement
            x0, y0 = 0.55, -0.05
            xy = rel_xy + np.array([x0, y0], dtype=float)

        # Apply poses (single layer on table)
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        for i, cube in enumerate(self.cubes):
            pos = np.array([xy[i, 0], xy[i, 1], self._CUBE_Z], dtype=float)
            cube.set_pose(position=pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))

        for _ in range(50):
            self._mojo.step()

        # Capture reference layout for rigid-translation success checking
        self._reference_xy = np.array([c.body.get_position()[:2] for c in self.cubes], dtype=float)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()

        # If reference wasn't captured by reset_object_poses(), capture it on the first step.
        if self._reference_xy is None and hasattr(self, "cubes"):
            self._reference_xy = np.array([c.body.get_position()[:2] for c in self.cubes], dtype=float)

        self._metric_step()

    @staticmethod
    def _max_pairwise_xy_offset_error(cur_xy: np.ndarray, ref_xy: np.ndarray) -> float:
        """Max over pairs of ||(cur_i-cur_j) - (ref_i-ref_j)||_2 in XY."""
        n = cur_xy.shape[0]
        max_err = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cur_off = cur_xy[i] - cur_xy[j]
                ref_off = ref_xy[i] - ref_xy[j]
                err = float(np.linalg.norm(cur_off - ref_off))
                if err > max_err:
                    max_err = err
        return max_err

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if len(self.cubes) == 0:
            return False

        # Current positions
        cur_pos = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        cur_xy = cur_pos[:, :2]
        cur_y = cur_pos[:, 1]

        # Goal condition 1: all cubes in left half (+Y), using Y axis ONLY
        eps = 1e-6
        in_left_half = bool(np.all(cur_y >= (self._LEFT_Y_MIN - eps)) and np.all(cur_y <= (self._LEFT_Y_MAX + eps)))

        # Goal condition 2: preserve pairwise XY offsets (rigid translation; no rotation)
        arrangement_ok = False
        max_pairwise_err = float("inf")
        if self._reference_xy is not None and self._reference_xy.shape == cur_xy.shape:
            max_pairwise_err = self._max_pairwise_xy_offset_error(cur_xy, self._reference_xy)
            arrangement_ok = bool(max_pairwise_err <= self._PAIRWISE_XY_TOL)

        # Released (no cube held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Metric stages
        if in_left_half:
            self._metric_stage(0, True)
        if arrangement_ok:
            self._metric_stage(1, True)
        if released:
            self._metric_stage(2, True)

        # Success
        self._success_check = bool(in_left_half and arrangement_ok and released)

        # Metrics
        # Y-violation: how far cubes are outside [0.0, 0.25]
        y_below = float(np.maximum(0.0, self._LEFT_Y_MIN - np.min(cur_y)))
        y_above = float(np.maximum(0.0, np.max(cur_y) - self._LEFT_Y_MAX))
        y_violation = y_below + y_above

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "y_violation_m": y_violation,
            },
            pose_error={
                "max_pairwise_xy_offset_error_m": (0.0 if np.isinf(max_pairwise_err) else float(max_pairwise_err)),
            },
        )
        return self._success_check

    def task_reward(self) -> float:
        # Simple shaping: encourage left-half satisfaction and layout preservation.
        if not hasattr(self, "cubes") or len(getattr(self, "cubes", [])) == 0:
            return 0.0

        if self._success():
            return 1.0

        cur_pos = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        cur_xy = cur_pos[:, :2]
        cur_y = cur_pos[:, 1]

        y_below = float(np.maximum(0.0, self._LEFT_Y_MIN - np.min(cur_y)))
        y_above = float(np.maximum(0.0, np.max(cur_y) - self._LEFT_Y_MAX))
        y_violation = y_below + y_above

        pairwise_err = 0.0
        if self._reference_xy is not None and self._reference_xy.shape == cur_xy.shape:
            pairwise_err = self._max_pairwise_xy_offset_error(cur_xy, self._reference_xy)

        # Penalize holding slightly to prefer placing/releasing
        holding = any(self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers)
        hold_pen = 0.05 if holding else 0.0

        return float(-(2.0 * y_violation + 1.0 * pairwise_err + hold_pen))

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})