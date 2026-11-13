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


class TriangleCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes onto a single green goal patch and arrange them into a triangular pattern on the patch.

    Sequential Sub-Goals:
        1. Grasp the red cube with either gripper.
        2. Move the red cube to the green goal patch area.
        3. Place and release the red cube on the green goal patch.
        4. Grasp the blue cube with either gripper.
        5. Move the blue cube to the green goal patch area.
        6. Place and release the blue cube on the green goal patch.
        7. Grasp the yellow cube with either gripper.
        8. Move the yellow cube to the green goal patch area.
        9. Place and release the yellow cube on the green goal patch.
        10. Adjust cube positions on the patch so the three cubes form a triangle pattern.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal-condition parameters (from task specification)
    _TOLERANCE_M = 0.03
    _MIN_PAIRWISE_DISTANCE_M = 0.05
    _MAX_PAIRWISE_DISTANCE_M = 0.14
    _COLLINEARITY_MIN_ALTITUDE_M = 0.02  # "not a straight line" threshold (perpendicular distance)

    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY those allowed) ---
        self.goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # Required: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_blue, self.cube_yellow]

        # Fixed goal patch pose (within workspace bounds)
        self._patch_goal_pos = np.array([0.55, 0.0, self._PATCH_REST_Z])
        self.goal_patch.body.set_position(self._patch_goal_pos)

        # Define a compact, non-collinear triangle that fits on a 10cm x 10cm patch.
        # Use side length ~0.051m (within [0.05, 0.14]) and keep centers within +/-0.03m of patch center.
        s = 0.051
        h = s * np.sqrt(3.0) / 2.0
        # Centered equilateral triangle in (dx, dy)
        tri_offsets = [
            np.array([0.0, 2.0 * h / 3.0]),  # vertex A
            np.array([-s / 2.0, -h / 3.0]),  # vertex B
            np.array([s / 2.0, -h / 3.0]),  # vertex C
        ]
        self._triangle_goal_positions = [
            np.array([self._patch_goal_pos[0] + o[0], self._patch_goal_pos[1] + o[1], self._CUBE_REST_Z])
            for o in tri_offsets
        ]

        # Initialize metrics (FULL signature)
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
        # 1) Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3) Keep goal patch fixed
        self.goal_patch.body.set_position(self._patch_goal_pos)

        # Required: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        # 4) Randomize cubes into UNSOLVED poses within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Ensure cubes start away from the patch so they're not "on patch" at reset
        # (Patch is 10cm x 10cm; enforce a generous margin.)
        min_dist_to_patch_xy = 0.12
        min_pairwise_xy = 0.08

        placed = []
        for cube in self.cubes:
            for _attempt in range(500):
                pos = np.array(
                    [
                        float(rng.uniform(x_min, x_max)),
                        float(rng.uniform(y_min, y_max)),
                        float(z),
                    ],
                    dtype=float,
                )

                # Far from patch center in XY so _is_cube_on_patch() should be False
                if float(np.linalg.norm(pos[:2] - self._patch_goal_pos[:2])) < min_dist_to_patch_xy:
                    continue

                # Keep cubes separated to reduce initial interpenetration
                if any(float(np.linalg.norm(pos[:2] - p[:2])) < min_pairwise_xy for p in placed):
                    continue

                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements  # (w,x,y,z)
                quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)  # to (x,y,z,w)

                cube.set_pose(position=pos, quat=quat)
                self._physics_forward()
                placed.append(pos)
                break
            else:
                # Fallback: deterministic corner positions (still away from patch), if sampling fails
                fallback_positions = [
                    np.array([0.41, -0.22, z], dtype=float),
                    np.array([0.69, 0.22, z], dtype=float),
                    np.array([0.41, 0.22, z], dtype=float),
                ]
                idx = len(placed)
                pos = fallback_positions[idx]
                cube.set_pose(position=pos, quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float))
                self._physics_forward()
                placed.append(pos)

        # 5) Physics settling
        self._mojo.physics.forward()

        # 6) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _physics_forward(self):
        # Some backends provide a forward() call; fall back safely.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        elif hasattr(self._mojo, "forward"):
            self._mojo.forward()
        else:
            # As a last resort, step once (advances time, but stabilizes placement).
            self._mojo.step()

    def _is_cube_on_patch(self, cube: ColoredCube, patch: GoalPatch) -> bool:
        # Reliable direction: cube.is_colliding(patch.collider)
        if not hasattr(patch, "collider"):
            return False
        if not cube.is_colliding(patch.collider):
            return False
        return True

    @staticmethod
    def _xy(p: np.ndarray) -> np.ndarray:
        return np.array([float(p[0]), float(p[1])], dtype=float)

    @staticmethod
    def _point_to_line_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        # Perpendicular distance from point p to infinite line through a-b in 2D
        ap = p - a
        ab = b - a
        denom = float(np.linalg.norm(ab))
        if denom < 1e-9:
            return float(np.linalg.norm(ap))
        # magnitude of 2D cross product divided by |ab|
        cross = float(abs(ab[0] * ap[1] - ab[1] * ap[0]))
        return cross / denom

    def _triangle_formation_ok(self, cube_positions_xy) -> bool:
        a, b, c = cube_positions_xy

        # Pairwise distances
        dab = float(np.linalg.norm(a - b))
        dbc = float(np.linalg.norm(b - c))
        dca = float(np.linalg.norm(c - a))

        dists = [dab, dbc, dca]
        if any(d < self._MIN_PAIRWISE_DISTANCE_M for d in dists):
            return False
        if any(d > self._MAX_PAIRWISE_DISTANCE_M for d in dists):
            return False

        # Non-collinearity check via "altitudes" (distance of each vertex to opposite side line)
        h_a = self._point_to_line_distance_2d(a, b, c)
        h_b = self._point_to_line_distance_2d(b, a, c)
        h_c = self._point_to_line_distance_2d(c, a, b)
        max_altitude = max(h_a, h_b, h_c)

        # Require sufficiently non-collinear
        if max_altitude < self._COLLINEARITY_MIN_ALTITUDE_M:
            return False

        return True

    def _best_vertex_assignment_distances(self, cube_positions_xy):
        # Compute per-cube distance to nearest triangle vertex with best overall assignment (brute-force 3! = 6)
        verts = [self._xy(p) for p in self._triangle_goal_positions]
        cubes = cube_positions_xy

        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        best = None
        best_cost = 1e9
        for perm in perms:
            d0 = float(np.linalg.norm(cubes[0] - verts[perm[0]]))
            d1 = float(np.linalg.norm(cubes[1] - verts[perm[1]]))
            d2 = float(np.linalg.norm(cubes[2] - verts[perm[2]]))
            cost = d0 + d1 + d2
            if cost < best_cost:
                best_cost = cost
                best = (d0, d1, d2)
        return best_cost, best

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "goal_patch") or not hasattr(self, "table"):
            return False

        # --- Goal conditions: each cube on patch ---
        on_patch = [self._is_cube_on_patch(c, self.goal_patch) for c in self.cubes]
        positions_ok = bool(all(on_patch))

        # --- Arrangement: triangle formation (not collinear) with spacing constraints ---
        cube_positions = [c.body.get_position() for c in self.cubes]
        cube_positions_xy = [self._xy(p) for p in cube_positions]
        arrangement_ok = positions_ok and self._triangle_formation_ok(cube_positions_xy)

        # --- Released ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # --- Subgoal metrics (no grasp-state checks; infer from goal-relevant spatial conditions only) ---
        # Red on patch -> stages 1-3; Blue -> 4-6; Yellow -> 7-9; Triangle arrangement -> stage 10
        # (We mark the corresponding stages as achieved when their spatial condition is met.)
        if on_patch[0]:
            for s in [0, 1, 2]:
                self._metric_stage(s, True)
        if on_patch[1]:
            for s in [3, 4, 5]:
                self._metric_stage(s, True)
        if on_patch[2]:
            for s in [6, 7, 8]:
                self._metric_stage(s, True)
        if arrangement_ok:
            self._metric_stage(9, True)

        # Final success
        self._success_check = bool(positions_ok and arrangement_ok and released)

        # Metrics: distances to nearest goal triangle vertices (best assignment)
        assign_cost, per_cube_d = self._best_vertex_assignment_distances(cube_positions_xy)
        if per_cube_d is None:
            per_cube_d = (float("inf"), float("inf"), float("inf"))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "sum_cube_to_triangle_vertices_xy_m": float(assign_cost),
                "cube_red_to_nearest_vertex_xy_m": float(per_cube_d[0]),
                "cube_blue_to_nearest_vertex_xy_m": float(per_cube_d[1]),
                "cube_yellow_to_nearest_vertex_xy_m": float(per_cube_d[2]),
            },
            pose_error={
                "triangle_assignment_sum_xy_m": float(assign_cost),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})