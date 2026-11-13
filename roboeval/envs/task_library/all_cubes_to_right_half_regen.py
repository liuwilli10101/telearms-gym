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
from roboeval.utils.guide_patch import guide_patch, place_guide
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubesShiftRight(RoboEvalEnv, MetricRolloutEval):
    """Move all four cubes into the translucent right-half region (negative Y), preserving their formation.

Success Criteria:
Keep each pair’s relative XY offset within 5 cm of its starting value. Place all cubes on the table and release them."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_REST_Z = 0.97
    _Y_RIGHT_HALF_MAX = 0.0
    _PAIRWISE_TOL_XY = 0.05
    _Y_EPS = 1e-4

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        self.right_guide = guide_patch(self._mojo, color="white", size=(0.15, 0.125), name="right_half")

        # Create cubes
        self.red_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.yellow_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")
        self.yellow_cube.set_color("yellow")

        self.cubes = [self.red_cube, self.green_cube, self.blue_cube, self.yellow_cube]
        self._cube_names = ["red_cube", "green_cube", "blue_cube", "yellow_cube"]

        # Reference formation (pairwise XY offsets) captured per-episode
        self._pair_order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self._ref_pairwise_xy = None
        self._need_recapture_ref = True

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
        place_guide(self.right_guide, 0.55, -0.125)
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization object
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM unsolved positions within workspace bounds
        #    Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        #    Ensure at least one cube starts in the LEFT half (Y > 0) so _success() is False.
        x_low, x_high = 0.40, 0.70
        y_low, y_high = 0.03, 0.23  # formation fits inside the marked 25 cm half
        min_sep = 0.06  # keep stable spacing

        def sample_xy(require_left_half: bool = False) -> np.ndarray:
            for _ in range(2000):
                x = float(rng.uniform(x_low, x_high))
                if require_left_half:
                    y = float(rng.uniform(max(0.05, y_low), y_high))  # force Y > 0
                else:
                    y = float(rng.uniform(y_low, y_high))
                return np.array([x, y], dtype=float)
            # Fallback (should not happen)
            return np.array([0.55, 0.10 if require_left_half else -0.10], dtype=float)

        xy_points = []
        # Force one cube to be in left half to guarantee unsolved reset
        forced_left_idx = int(rng.integers(0, len(self.cubes)))

        for i in range(4):
            require_left = (i == forced_left_idx)
            for _ in range(5000):
                cand = sample_xy(require_left_half=require_left)
                if all(np.linalg.norm(cand - p) >= min_sep for p in xy_points):
                    xy_points.append(cand)
                    break
            else:
                # If rejection sampling fails, place with relaxed separation but still in-bounds
                xy_points.append(sample_xy(require_left_half=require_left))

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")
        self.yellow_cube.set_color("yellow")

        # Place cubes (single layer)
        for cube, xy in zip(self.cubes, xy_points):
            cube.set_pose(
                position=np.array([xy[0], xy[1], self._CUBE_REST_Z], dtype=float),
                quat=np.array([1, 0, 0, 0], dtype=float),
            )

        # 4. Physics settling
        self._mojo.physics.forward()

        # Capture reference formation for this episode (pairwise offsets at start)
        self._capture_reference_pairwise_xy()
        self._need_recapture_ref = True

        # 5. Initialize subgoals to False
        # Subgoals: 0..3 right-half checks, 4 formation preserved
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()

        # Allow recapturing the "initial formation" after potential external randomization.
        # Do it once, early in the episode, and only when cubes are released.
        if getattr(self, "_need_recapture_ref", False):
            released = not any(
                self.robot.is_gripper_holding_object(obj, s)
                for obj in self.cubes
                for s in self.robot.grippers
            )
            if released:
                self._capture_reference_pairwise_xy()
                self._need_recapture_ref = False

        self._metric_step()

    def _capture_reference_pairwise_xy(self):
        xy = self._get_xy_positions()
        ref = []
        for i, j in self._pair_order:
            ref.append(xy[i] - xy[j])
        self._ref_pairwise_xy = np.asarray(ref, dtype=float)

    def _get_xy_positions(self) -> np.ndarray:
        xy = []
        for c in self.cubes:
            p = c.body.get_position()
            xy.append([float(p[0]), float(p[1])])
        return np.asarray(xy, dtype=float)

    def _pairwise_xy_error(self) -> float:
        if self._ref_pairwise_xy is None:
            return float("inf")
        xy = self._get_xy_positions()
        cur = []
        for i, j in self._pair_order:
            cur.append(xy[i] - xy[j])
        cur = np.asarray(cur, dtype=float)
        diffs = cur - self._ref_pairwise_xy
        return float(np.max(np.linalg.norm(diffs, axis=1)))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Individual right-half checks
        ys = [float(c.body.get_position()[1]) for c in self.cubes]
        right_half_flags = [y <= (self._Y_RIGHT_HALF_MAX + self._Y_EPS) for y in ys]
        red_ok, green_ok, blue_ok, yellow_ok = right_half_flags

        # Pattern preservation (pairwise XY offsets invariant under translation)
        pairwise_err = self._pairwise_xy_error()
        formation_ok = pairwise_err <= self._PAIRWISE_TOL_XY

        # Also require cubes are placed on the tabletop (not being held / not floating)
        # (Table collision is reliable; keep a small Z sanity check.)
        on_table = True
        for c in self.cubes:
            pz = float(c.body.get_position()[2])
            if abs(pz - self._CUBE_REST_Z) > 0.04:
                on_table = False
                break
            if not c.is_colliding(self.table.table_top):
                on_table = False
                break

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        positions_ok = bool(all(right_half_flags) and on_table)
        semantic_ok = True  # No semantic constraints beyond color existence

        # Subgoal metrics
        if red_ok:
            self._metric_stage(0, True)
        if green_ok:
            self._metric_stage(1, True)
        if blue_ok:
            self._metric_stage(2, True)
        if yellow_ok:
            self._metric_stage(3, True)
        if positions_ok and formation_ok:
            self._metric_stage(4, True)

        self._success_check = bool(positions_ok and formation_ok and semantic_ok and released)

        # Finalize metrics
        target_distance = {
            name: float(max(0.0, y - self._Y_RIGHT_HALF_MAX))
            for name, y in zip(self._cube_names, ys)
        }
        pose_error = {
            "pairwise_xy_max_error_m": float(pairwise_err),
            "max_y_overage_m": float(max(target_distance.values())) if len(target_distance) else float("inf"),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})