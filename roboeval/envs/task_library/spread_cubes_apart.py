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

class CubeSeparationTask(RoboEvalEnv, MetricRolloutEval):
    """Separate three clustered cubes so every pair is at least 10 cm apart.

Success Criteria:
- Every pair of cubes is at least 10 cm apart.
- All cubes are on the table.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _MIN_PAIRWISE_DISTANCE_M = 0.10
    _CUBE_REST_Z = 0.97
    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Assets (MANDATORY: ONLY ColoredCube)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3]

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("green")

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (even if selecting among goal layouts)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("green")

        # 3. Place objects in RANDOM *UNSOLVED* positions (clustered so at least one pair < 10cm)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Keep a small margin so offsets don't exit bounds
        margin = 0.08
        cx = float(rng.uniform(x_min + margin, x_max - margin))
        cy = float(rng.uniform(y_min + margin, y_max - margin))
        center = np.array([cx, cy], dtype=float)

        # Create a tight cluster with pairwise distances in [min_sep, max_sep] < 0.10
        # (min_sep avoids heavy interpenetration; max_sep guarantees failure of separation constraint)
        min_sep = 0.05
        max_sep = 0.09

        pts_xy = None
        for _ in range(500):
            cand = []
            for _k in range(3):
                r = float(rng.uniform(0.00, 0.06))
                theta = float(rng.uniform(-np.pi, np.pi))
                p = center + r * np.array([np.cos(theta), np.sin(theta)], dtype=float)
                # Clip to bounds (with tiny buffer)
                p[0] = float(np.clip(p[0], x_min + 0.01, x_max - 0.01))
                p[1] = float(np.clip(p[1], y_min + 0.01, y_max - 0.01))
                cand.append(p)

            d01 = float(np.linalg.norm(cand[0] - cand[1]))
            d02 = float(np.linalg.norm(cand[0] - cand[2]))
            d12 = float(np.linalg.norm(cand[1] - cand[2]))

            # Ensure "unsolved": at least one pair is too close (< 0.10), and avoid severe overlap
            if (min(d01, d02, d12) < self._MIN_PAIRWISE_DISTANCE_M) and (
                d01 >= min_sep and d02 >= min_sep and d12 >= min_sep
            ) and (d01 <= max_sep and d02 <= max_sep and d12 <= max_sep):
                pts_xy = cand
                break

        # Fallback: deterministic tight line cluster (still unsolved) if sampling fails
        if pts_xy is None:
            base = center
            pts_xy = [
                base + np.array([0.00, 0.00]),
                base + np.array([0.06, 0.00]),
                base + np.array([0.03, 0.05]),
            ]

        pos1 = np.array([pts_xy[0][0], pts_xy[0][1], z], dtype=float)
        pos2 = np.array([pts_xy[1][0], pts_xy[1][1], z], dtype=float)
        pos3 = np.array([pts_xy[2][0], pts_xy[2][1], z], dtype=float)

        # Random yaw rotations (upright)
        def yaw_quat(yaw: float) -> np.ndarray:
            half = 0.5 * yaw
            return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)

        self.cube_1.set_pose(position=pos1, quat=yaw_quat(float(rng.uniform(-np.pi, np.pi))))
        self.cube_2.set_pose(position=pos2, quat=yaw_quat(float(rng.uniform(-np.pi, np.pi))))
        self.cube_3.set_pose(position=pos3, quat=yaw_quat(float(rng.uniform(-np.pi, np.pi))))

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _pairwise_distances(self) -> Dict[str, float]:
        d12 = float(distance(self.cube_1.body, self.cube_2.body))
        d13 = float(distance(self.cube_1.body, self.cube_3.body))
        d23 = float(distance(self.cube_2.body, self.cube_3.body))
        return {"d12": d12, "d13": d13, "d23": d23}

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Must be released at success time
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Must be on the table (task says "on the table")
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)

        # Pairwise separation constraints
        ds = self._pairwise_distances()
        positions_ok = (
            ds["d12"] >= self._MIN_PAIRWISE_DISTANCE_M
            and ds["d13"] >= self._MIN_PAIRWISE_DISTANCE_M
            and ds["d23"] >= self._MIN_PAIRWISE_DISTANCE_M
        )

        # Subgoals (best-effort, compatible with single- or multi-gripper robots)
        # 0: "left gripper grasps a cube" -> any grasp by any available gripper counts
        any_grasp = any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )
        if any_grasp:
            self._metric_stage(0, True)

        # 1: "right gripper grasps a cube" -> if multiple grippers exist, require 2 distinct cubes held
        held_pairs = []
        for s in self.robot.grippers:
            for obj in self.cubes:
                if self.robot.is_gripper_holding_object(obj, s):
                    held_pairs.append((s, obj))
        held_unique_objs = {obj for _, obj in held_pairs}
        if len(self.robot.grippers) >= 2:
            if len(held_unique_objs) >= 2:
                self._metric_stage(1, True)
        else:
            # Single-arm: mark as achieved once any grasp occurs (cannot be truly bimanual)
            if any_grasp:
                self._metric_stage(1, True)

        # 2-4: each cube is sufficiently far from the other two
        if min(ds["d12"], ds["d13"]) >= self._MIN_PAIRWISE_DISTANCE_M:
            self._metric_stage(2, True)  # cube_1 transported away
        if min(ds["d12"], ds["d23"]) >= self._MIN_PAIRWISE_DISTANCE_M:
            self._metric_stage(3, True)  # cube_2 transported away
        if min(ds["d13"], ds["d23"]) >= self._MIN_PAIRWISE_DISTANCE_M:
            self._metric_stage(4, True)  # cube_3 transported away

        # 5-6: placement/release (both satisfied when nothing is held)
        if released:
            self._metric_stage(5, True)
            self._metric_stage(6, True)

        semantic_ok = True  # no semantic constraints in this task

        self._success_check = bool(positions_ok and on_table and semantic_ok and released)

        # Metrics
        target_distance = {
            "min_pairwise_distance_m": self._MIN_PAIRWISE_DISTANCE_M,
            "d12_m": ds["d12"],
            "d13_m": ds["d13"],
            "d23_m": ds["d23"],
        }
        pose_error = {
            # Report how much each constraint is violated (0 if satisfied)
            "d12_shortfall_m": float(max(0.0, self._MIN_PAIRWISE_DISTANCE_M - ds["d12"])),
            "d13_shortfall_m": float(max(0.0, self._MIN_PAIRWISE_DISTANCE_M - ds["d13"])),
            "d23_shortfall_m": float(max(0.0, self._MIN_PAIRWISE_DISTANCE_M - ds["d23"])),
            "on_table": float(1.0 if on_table else 0.0),
            "released": float(1.0 if released else 0.0),
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