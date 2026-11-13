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


class CubeRingPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes in an even ring around a green target patch.
    Each cube should be placed at approximately the same radius from the patch
    center with roughly 90° spacing between neighbors.

    Sequential Sub-Goals:
        1. Place at least 1 cube on the ring (correct radius).
        2. Place at least 2 cubes on the ring (correct radius + separated).
        3. Place at least 3 cubes on the ring (correct radius + separated).
        4. Place all 4 cubes on the ring (correct radius + separated).
        5. Adjust to even spacing (≈90° between neighbors).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from spec)
    _RADIUS_M = 0.10
    _RADIUS_TOL_M = 0.02
    _EVEN_SPACING_DEG = 90.0
    _SPACING_TOL_DEG = 20.0
    _MIN_INTER_CUBE_DIST_M = 0.05

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Target patch (fixed, green). Visual-only: success uses geometric ring metrics
        # and table contact, never the patch collider. Disabling collision prevents cubes
        # spawned at/near the patch center from being violently ejected off the table by
        # the thin patch collision box.
        self.goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target", collision_enabled=False)

        # GoalPatch's "visual" geom keeps MuJoCo's default collision flags (contype=1),
        # so even with collision_enabled=False the thin 20cm patch box still collides and
        # ejects cubes spawned on/near it. Force contact OFF on every patch geom — the
        # patch is a visual-only target in this task.
        for _g in self.goal_patch.body.geoms:
            _bound = self._mojo.physics.bind(_g.mjcf)
            _bound.contype = 0
            _bound.conaffinity = 0

        # Four colored cubes (ONLY allowed asset type)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "yellow", "purple"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Place patch at a canonical workspace center (position set again in _on_reset)
        self._patch_xy = np.array([0.55, 0.0], dtype=float)
        self.goal_patch.body.set_position(np.array([self._patch_xy[0], self._patch_xy[1], 0.951], dtype=float))

        # Precompute canonical goal ring positions (used for deterministic goal-state reset)
        # Angles: 0, 90, 180, 270 degrees
        angles = np.deg2rad([0.0, 90.0, 180.0, 270.0])
        self._goal_ring_positions = []
        for a in angles:
            x = self._patch_xy[0] + self._RADIUS_M * np.cos(a)
            y = self._patch_xy[1] + self._RADIUS_M * np.sin(a)
            self._goal_ring_positions.append(np.array([x, y, 0.97], dtype=float))

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

        # 2. Deterministic RNG (required pattern, even if not used heavily here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        Z_CUBE = 0.97
        Z_PATCH = 0.951

        # Randomize patch within bounds (keep a small margin)
        patch_x = float(rng.uniform(X_MIN + 0.05, X_MAX - 0.05))
        patch_y = float(rng.uniform(Y_MIN + 0.05, Y_MAX - 0.05))
        self._patch_xy = np.array([patch_x, patch_y], dtype=float)
        self.goal_patch.body.set_position(np.array([patch_x, patch_y, Z_PATCH], dtype=float))

        # Force at least one cube to violate the ring radius tolerance:
        # put cube_1 near the patch center => radius ~ 0, error ~ RADIUS_M (> tol)
        c1_xy = self._patch_xy + rng.uniform(-0.005, 0.005, size=(2,))
        c1_xy[0] = float(np.clip(c1_xy[0], X_MIN, X_MAX))
        c1_xy[1] = float(np.clip(c1_xy[1], Y_MIN, Y_MAX))
        self.cube_1.set_pose(position=np.array([c1_xy[0], c1_xy[1], Z_CUBE], dtype=float), quat=np.array([0.0, 0.0, 0.0, 1.0]))

        # Place remaining cubes randomly with mild separation to avoid heavy interpenetration
        placed_xy = [c1_xy.copy()]
        # Cube is 0.04m wide, so its diagonal footprint is ~0.057m. Require >= 0.06m center
        # separation so cubes never spawn interpenetrating (which causes physics blow-ups).
        min_sep = 0.06
        for cube in [self.cube_2, self.cube_3, self.cube_4]:
            for _ in range(200):
                xy = np.array(
                    [rng.uniform(X_MIN, X_MAX), rng.uniform(Y_MIN, Y_MAX)],
                    dtype=float,
                )
                if all(float(np.linalg.norm(xy - p)) >= min_sep for p in placed_xy):
                    placed_xy.append(xy)
                    cube.set_pose(position=np.array([xy[0], xy[1], Z_CUBE], dtype=float), quat=np.array([0.0, 0.0, 0.0, 1.0]))
                    break
            else:
                # Fallback (should be rare): place with deterministic offsets inside bounds
                xy = np.array(
                    [np.clip(self._patch_xy[0] + rng.uniform(-0.15, 0.15), X_MIN, X_MAX),
                     np.clip(self._patch_xy[1] + rng.uniform(-0.15, 0.15), Y_MIN, Y_MAX)],
                    dtype=float,
                )
                placed_xy.append(xy)
                cube.set_pose(position=np.array([xy[0], xy[1], Z_CUBE], dtype=float), quat=np.array([0.0, 0.0, 0.0, 1.0]))

        # 4. Physics settling: forward kinematics, then step so any residual contact
        # resolves gently inside reset (before recording/teleop starts) instead of
        # producing a sudden impulse on the first sim step.
        self._mojo.physics.forward()
        for _ in range(25):
            self._mojo.step()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _wrap_to_2pi(a: np.ndarray) -> np.ndarray:
        return (a + 2.0 * np.pi) % (2.0 * np.pi)

    def _compute_ring_metrics(self):
        """
        Returns:
            radial_errors (np.ndarray): shape (4,), |r_i - R|
            radii (np.ndarray): shape (4,)
            angles (np.ndarray): shape (4,) in [0, 2pi)
            min_pairwise_dist (float): minimum XY distance between cube centers
            spacing_diffs_deg (np.ndarray): shape (4,), cyclic diffs between sorted angles in degrees
        """
        patch_xy = self.goal_patch.body.get_position()[:2]
        cube_xys = np.stack([c.body.get_position()[:2] for c in self.cubes], axis=0)

        vecs = cube_xys - patch_xy[None, :]
        radii = np.linalg.norm(vecs, axis=1)
        radial_errors = np.abs(radii - self._RADIUS_M)

        angles = self._wrap_to_2pi(np.arctan2(vecs[:, 1], vecs[:, 0]))

        # Pairwise distances (XY)
        min_d = float("inf")
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                d = float(np.linalg.norm(cube_xys[i] - cube_xys[j]))
                if d < min_d:
                    min_d = d

        # Angular spacing (cyclic)
        idx = np.argsort(angles)
        ang_sorted = angles[idx]
        diffs = np.diff(ang_sorted)
        diffs = np.concatenate([diffs, np.array([(ang_sorted[0] + 2.0 * np.pi) - ang_sorted[-1]])])
        spacing_diffs_deg = np.rad2deg(diffs)

        return radial_errors, radii, angles, min_d, spacing_diffs_deg

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False

        radial_errors, radii, angles, min_pairwise_dist, spacing_diffs_deg = self._compute_ring_metrics()

        # Must be on the table surface (avoid stacked/airborne solutions)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # Radius condition for all cubes
        radius_ok_each = radial_errors <= self._RADIUS_TOL_M
        radius_ok_all = bool(np.all(radius_ok_each))

        # Separation condition
        separation_ok = bool(min_pairwise_dist >= self._MIN_INTER_CUBE_DIST_M)

        # Even spacing condition (invariant to global rotation)
        spacing_errors = np.abs(spacing_diffs_deg - self._EVEN_SPACING_DEG)
        spacing_ok = bool(np.all(spacing_errors <= self._SPACING_TOL_DEG))

        positions_ok = bool(on_table and radius_ok_all and separation_ok and spacing_ok)

        # No semantic predicates (colors are not constrained in the goal), but keep a placeholder.
        semantic_ok = True

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage progress (order-free; based on how many cubes satisfy radius + on-table, and final spacing)
        num_on_ring = int(np.sum(radius_ok_each)) if on_table else 0
        if num_on_ring >= 1:
            self._metric_stage(0, True)
        if num_on_ring >= 2 and separation_ok:
            self._metric_stage(1, True)
        if num_on_ring >= 3 and separation_ok:
            self._metric_stage(2, True)
        if num_on_ring >= 4 and separation_ok:
            self._metric_stage(3, True)
        if num_on_ring >= 4 and separation_ok and spacing_ok:
            self._metric_stage(4, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_radial_error_m": float(np.mean(np.abs(radii - self._RADIUS_M))),
                "min_pairwise_dist_m": float(min_pairwise_dist),
                "mean_spacing_error_deg": float(np.mean(np.abs(spacing_diffs_deg - self._EVEN_SPACING_DEG))),
            },
            pose_error={
                "max_radial_error_m": float(np.max(np.abs(radii - self._RADIUS_M))),
                "max_spacing_error_deg": float(np.max(np.abs(spacing_diffs_deg - self._EVEN_SPACING_DEG))),
            },
        )
        return self._success_check

    def compute_dense_reward(self) -> float:
        # Dense reward based purely on spatial arrangement (no grasp/state predicates).
        if not hasattr(self, "cubes") or not hasattr(self, "goal_patch"):
            return 0.0

        radial_errors, radii, angles, min_pairwise_dist, spacing_diffs_deg = self._compute_ring_metrics()

        # Normalize errors into [0, 1] style components, then combine
        # Radial: 1 at zero error, 0 at >= tolerance
        radial_score = 1.0 - float(np.clip(np.mean(radial_errors) / max(self._RADIUS_TOL_M, 1e-6), 0.0, 1.0))

        # Spacing: 1 at perfect 90deg, 0 at >= tolerance
        spacing_err = np.mean(np.abs(spacing_diffs_deg - self._EVEN_SPACING_DEG))
        spacing_score = 1.0 - float(np.clip(spacing_err / max(self._SPACING_TOL_DEG, 1e-6), 0.0, 1.0))

        # Separation: 1 if >= min, else linear penalty
        sep_score = float(np.clip(min_pairwise_dist / max(self._MIN_INTER_CUBE_DIST_M, 1e-6), 0.0, 1.0))

        # On-table bonus (keeps reward low if cubes are lifted/stacked)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes) if hasattr(self, "table") else False
        on_table_score = 1.0 if on_table else 0.0

        # Combine
        reward = 0.45 * radial_score + 0.45 * spacing_score + 0.10 * sep_score
        reward *= (0.5 + 0.5 * on_table_score)
        return float(np.clip(reward, 0.0, 1.0))

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})