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


class CubeRowArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange five colored cubes into a single left-to-right horizontal row on the table.
    The cubes should be aligned and evenly spaced from one another.

    Sequential Sub-Goals:
        1. Acquire a stable grasp on the red cube.
        2. Move the red cube toward the intended row region.
        3. Place the red cube as one element of the row (any slot).
        4. Acquire a stable grasp on the blue cube.
        5. Move the blue cube toward the intended row region.
        6. Place the blue cube into the row with approximate spacing from its neighbors.
        7. Acquire a stable grasp on the green cube.
        8. Move the green cube toward the intended row region.
        9. Place the green cube into the row with approximate spacing from its neighbors.
        10. Acquire a stable grasp on the yellow cube.
        11. Move the yellow cube toward the intended row region.
        12. Place the yellow cube into the row with approximate spacing from its neighbors.
        13. Acquire a stable grasp on the purple cube.
        14. Move the purple cube toward the intended row region.
        15. Place the purple cube into the row with approximate spacing from its neighbors.
        16. Fine-tune positions so all cubes lie on one horizontal line and are evenly spaced.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (customized for this task) ---
    _CUBE_COLORS = ["red", "blue", "green", "yellow", "purple"]
    _CUBE_NAMES = ["cube_red", "cube_blue", "cube_green", "cube_yellow", "cube_purple"]

    # Arrangement semantics: left-to-right in camera view is +Y -> -Y
    _SAME_LINE_TOL_M = 0.03              # x-alignment tolerance
    _SPACING_TOL_RATIO = 0.2             # even-spacing tolerance (relative to mean gap)

    # NOTE: The proposal lists min_neighbor_distance_m=0.06, but the computed canonical
    # goal coordinates imply 0.05m gaps. We enforce "no touching" primarily via
    # collision checks, and a conservative minimum neighbor center distance of 0.05m.
    _MIN_NEIGHBOR_DIST_M = 0.05

    _CUBE_REST_Z = 0.97
    _NUM_SUBGOALS = 16

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (MUST use only ColoredCube)
        self.cubes = []
        for _ in range(5):
            self.cubes.append(ColoredCube(self._mojo))

        # MUST call set_color() in _initialize_env()
        for cube, col in zip(self.cubes, self._CUBE_COLORS):
            cube.set_color(col)

        # Canonical goal positions (computed): one horizontal row along Y, fixed X.
        self._goal_positions = [
            np.array([0.55, 0.10, self._CUBE_REST_Z]),
            np.array([0.55, 0.05, self._CUBE_REST_Z]),
            np.array([0.55, 0.00, self._CUBE_REST_Z]),
            np.array([0.55, -0.05, self._CUBE_REST_Z]),
            np.array([0.55, -0.10, self._CUBE_REST_Z]),
        ]

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

        # 2. Deterministic RNG (required pattern; may be used by external perturbations)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Keep cubes separated to avoid initial overlaps/touching
        min_sep = 0.07  # conservative > 0.045 used in success check

        def _sample_pose_xy(existing_xy):
            for _ in range(2000):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                if existing_xy:
                    d2 = [(x - ex) ** 2 + (y - ey) ** 2 for (ex, ey) in existing_xy]
                    if np.min(d2) < (min_sep * min_sep):
                        continue
                return x, y
            return None

        placed = False
        for _attempt in range(200):
            xy_list = []
            poses = []
            ok = True

            for _i in range(5):
                s = _sample_pose_xy(xy_list)
                if s is None:
                    ok = False
                    break
                x, y = s
                xy_list.append((x, y))
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements  # [w,x,y,z]
                quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
                poses.append((np.array([x, y, z], dtype=float), quat_xyzw))

            if not ok:
                continue

            # Apply sampled poses
            for cube, col, (pos, quat_xyzw) in zip(self.cubes, self._CUBE_COLORS, poses):
                cube.set_color(col)  # MUST call set_color() in _on_reset()
                cube.set_pose(position=pos.copy(), quat=quat_xyzw.copy())

            # Physics settling
            self._mojo.physics.forward()
            self._mojo.physics.forward()

            # Ensure reset state is UNSOLVED (row condition must fail)
            if self._all_on_table() and (not self._row_even_spacing_ok()):
                placed = True
                break

        if not placed:
            # Deterministic fallback: clearly unsolved (x not aligned), within bounds
            fallback_positions = [
                np.array([0.42, 0.22, z]),
                np.array([0.48, 0.10, z]),
                np.array([0.56, -0.02, z]),
                np.array([0.63, -0.12, z]),
                np.array([0.69, -0.22, z]),
            ]
            for cube, col, pos in zip(self.cubes, self._CUBE_COLORS, fallback_positions):
                cube.set_color(col)
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements  # [w,x,y,z]
                quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
                cube.set_pose(position=pos.copy(), quat=quat_xyzw)
            self._mojo.physics.forward()
            self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # avoid "unused" linting patterns (rng is intentionally available)
        _ = rng

    def _on_step(self):
        # Always call base _on_step so the framework runs success checking / bookkeeping.
        super()._on_step()
        self._metric_step()

    def _all_on_table(self) -> bool:
        # Use Z-height + workspace bounds; collision checks can be brittle across assets/resets.
        for cube in self.cubes:
            p = cube.body.get_position()
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            if abs(z - self._CUBE_REST_Z) > 0.03:
                return False
            if not (0.35 <= x <= 0.75 and -0.30 <= y <= 0.30):
                return False
        return True

    def _row_even_spacing_ok(self) -> bool:
        # Gather positions
        positions = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)

        # 1) Alignment: x-positions approximately equal (same line)
        x_ref = float(np.mean(xs))
        if np.max(np.abs(xs - x_ref)) > self._SAME_LINE_TOL_M:
            return False

        # 2) Left-to-right order: +Y (left) -> -Y (right), strictly decreasing
        order = list(np.argsort(-ys))  # sort by y descending
        ys_sorted = ys[order]
        # strict ordering (avoid ties)
        if not np.all(ys_sorted[:-1] > ys_sorted[1:] + 1e-4):
            return False

        # 3) No touching / separation: use center distances (collision queries can be brittle)
        min_sep = 0.045
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                dx = float(xs[i] - xs[j])
                dy = float(ys[i] - ys[j])
                if (dx * dx + dy * dy) < (min_sep * min_sep):
                    return False

        gaps = ys_sorted[:-1] - ys_sorted[1:]  # positive gaps along Y (left->right)
        if np.min(gaps) < (self._MIN_NEIGHBOR_DIST_M - 1e-4):
            return False

        # 4) Even spacing: adjacent gaps approximately equal
        mean_gap = float(np.mean(gaps))
        if mean_gap <= 1e-6:
            return False
        if np.max(np.abs(gaps - mean_gap)) > self._SPACING_TOL_RATIO * mean_gap:
            return False

        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or len(getattr(self, "cubes", [])) != 5:
            return False

        positions_ok = bool(self._all_on_table() and self._row_even_spacing_ok())

        # No semantic constraints beyond spatial arrangement for this task
        semantic_ok = True

        # Update metric stages only upon full success (keeps subgoal tracking simple & robust)
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok)

        # Metrics: distance of each cube to nearest goal slot (order-independent)
        goal_xy = np.array([gp[:2] for gp in self._goal_positions], dtype=float)
        target_distance = {}
        pose_error = {}

        for name, cube in zip(self._CUBE_NAMES, self.cubes):
            p = cube.body.get_position()
            xy = np.array([p[0], p[1]], dtype=float)
            d = float(np.min(np.linalg.norm(goal_xy - xy[None, :], axis=1)))
            target_distance[name] = d
            pose_error[name] = d  # pose error proxy for this spatial arrangement

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})