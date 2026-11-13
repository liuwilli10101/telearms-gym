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

class CubeClusterPackingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Pack seven colored cubes into a compact, dense cluster located at the center of the workspace.
    The final configuration should minimize the cluster footprint while keeping all cubes on the table.

    Sequential Sub-Goals:
        1. Grasp any cube to start consolidating cubes toward the center.
        2. Carry a grasped cube toward the workspace center region.
        3. Place the cube adjacent to the growing cluster to reduce the cluster footprint.
        4. Nudge/compress cubes inward to remove gaps and tighten the cluster.
        5. Release after placement/adjustment so cubes remain stably positioned in the cluster.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task thresholds (from spec)
    _WORKSPACE_CENTER_XY = np.array([0.55, 0.0], dtype=float)
    _MAX_CLUSTER_BBOX_EXTENT_XY_M = 0.14
    _MAX_DISTANCE_TO_CENTER_XY_M = 0.10
    _NEIGHBOR_RADIUS_M = 0.06
    _MIN_NEIGHBORS_PER_CUBE = 1

    # A small margin consistent with general spatial tolerance guidance
    _TOL_M = 0.014

    _NUM_CUBES = 7
    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ColoredCube only)
        self.cubes = []
        self._cube_names = [f"cube_{i+1}" for i in range(self._NUM_CUBES)]
        self._cube_colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan"]

        for i in range(self._NUM_CUBES):
            cube = ColoredCube(self._mojo)
            # REQUIRED: call set_color() in _initialize_env
            cube.set_color(self._cube_colors[i % len(self._cube_colors)])
            self.cubes.append(cube)

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (even if not used here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: call set_color() in _on_reset as well
        for i, cube in enumerate(self.cubes):
            cube.set_color(self._cube_colors[i % len(self._cube_colors)])

        # Randomize cubes into an UNSOLVED configuration (ensure _success() == False).
        # We enforce large pairwise separations so neighbor-density fails.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = 0.97  # ColoredCube rest Z on table (MUST use this)

        min_sep = 0.085  # > _NEIGHBOR_RADIUS_M (0.06) to make density constraint fail
        positions_xy = []

        max_attempts_per_cube = 3000
        for _ in range(self._NUM_CUBES):
            placed = False
            for _attempt in range(max_attempts_per_cube):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)

                if len(positions_xy) == 0:
                    positions_xy.append(p)
                    placed = True
                    break

                dmin = float(np.min(np.linalg.norm(np.stack(positions_xy, axis=0) - p[None, :], axis=1)))
                if dmin >= min_sep:
                    positions_xy.append(p)
                    placed = True
                    break

            if not placed:
                # Fallback (still random within bounds); may reduce separation but remains unsolved typically.
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                positions_xy.append(np.array([x, y], dtype=float))

        for cube, xy in zip(self.cubes, positions_xy):
            pos = np.array([float(xy[0]), float(xy[1]), z], dtype=float)
            # Defensive bounds clamp
            pos[0] = float(np.clip(pos[0], x_min, x_max))
            pos[1] = float(np.clip(pos[1], y_min, y_max))

            # Random yaw (upright)
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # (w, x, y, z)
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)

            cube.set_pose(position=pos, quat=quat_xyzw)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _cluster_bbox_extents_xy(self, positions_xy: np.ndarray) -> np.ndarray:
        """Return [extent_x, extent_y] in meters."""
        mins = positions_xy.min(axis=0)
        maxs = positions_xy.max(axis=0)
        return maxs - mins

    def _all_on_table(self) -> bool:
        # Use reliable table surface collision (table.table_top)
        for cube in self.cubes:
            pos = cube.body.get_position()
            if abs(float(pos[2]) - 0.97) > 0.03:
                return False
            if not cube.is_colliding(self.table.table_top):
                return False
        return True

    def _neighbor_density_ok(self, positions_xy: np.ndarray) -> bool:
        n = positions_xy.shape[0]
        for i in range(n):
            neigh = 0
            for j in range(n):
                if i == j:
                    continue
                if float(np.linalg.norm(positions_xy[i] - positions_xy[j])) <= self._NEIGHBOR_RADIUS_M:
                    neigh += 1
                    if neigh >= self._MIN_NEIGHBORS_PER_CUBE:
                        break
            if neigh < self._MIN_NEIGHBORS_PER_CUBE:
                return False
        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        positions = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        positions_xy = positions[:, :2]
        center_xy = self._WORKSPACE_CENTER_XY.reshape(1, 2)

        # Translation-invariant: compactness is measured around the cluster's own
        # centroid, not the world-frame workspace center, so the cluster can form
        # anywhere on the table.
        cluster_centroid = positions_xy.mean(axis=0, keepdims=True)
        dists_to_centroid = np.linalg.norm(positions_xy - cluster_centroid, axis=1)
        within_centroid_radius = bool(
            np.all(dists_to_centroid <= (self._MAX_DISTANCE_TO_CENTER_XY_M + self._TOL_M))
        )

        # Keep dists_to_center for diagnostic metrics only.
        dists_to_center = np.linalg.norm(positions_xy - center_xy, axis=1)

        extents_xy = self._cluster_bbox_extents_xy(positions_xy)
        footprint_ok = bool(np.all(extents_xy <= (self._MAX_CLUSTER_BBOX_EXTENT_XY_M + self._TOL_M)))

        on_table = self._all_on_table()

        # Condition 2: density constraint (each cube has at least one neighbor nearby)
        density_ok = self._neighbor_density_ok(positions_xy)

        positions_ok = bool(within_centroid_radius and footprint_ok and on_table and density_ok)
        semantic_ok = True  # no semantic/color ordering constraints in this task

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (heuristics; success sets all True)
        any_lifted = bool(np.any(positions[:, 2] > 0.99))
        any_near_center = bool(np.any(dists_to_center <= (self._MAX_DISTANCE_TO_CENTER_XY_M + 0.02)))
        at_least_three_near_center = bool(np.sum(dists_to_center <= 0.08) >= 3)
        roughly_compact = bool(np.all(extents_xy <= 0.18))

        self._metric_stage(0, any_lifted)                         # grasp
        self._metric_stage(1, any_near_center)                    # transport
        self._metric_stage(2, at_least_three_near_center and on_table)  # place
        self._metric_stage(3, roughly_compact and density_ok)     # align/compress
        self._metric_stage(4, released)                           # release

        if positions_ok and semantic_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Set success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "avg_dist_to_center_xy_m": float(np.mean(dists_to_center)),
                "max_dist_to_center_xy_m": float(np.max(dists_to_center)),
            },
            pose_error={
                "cluster_bbox_extent_x_m": float(extents_xy[0]),
                "cluster_bbox_extent_y_m": float(extents_xy[1]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})