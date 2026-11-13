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

class TriangularCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes on the table around a green goal patch so they form
    a triangular pattern surrounding the patch. The final layout should approximate
    an equilateral triangle centered on the target area.

    Sequential Sub-Goals:
        1. Grasp ColoredCube_1 (any gripper).
        2. Move ColoredCube_1 to a vertex location around the green goal patch.
        3. Place/release ColoredCube_1.
        4. Grasp ColoredCube_2 (any gripper).
        5. Move ColoredCube_2 to another vertex around the green goal patch.
        6. Place/release ColoredCube_2.
        7. Grasp ColoredCube_3 (any gripper).
        8. Move ColoredCube_3 to complete the triangular arrangement.
        9. Place/release ColoredCube_3.
        10. Align all three cubes so they are roughly 120° apart around the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task geometry (from spec; also respect registry constraint cube<->patch min distance)
    _ANNULUS_R_MIN = 0.08
    _ANNULUS_R_MAX = 0.16
    _CUBE_PATCH_MIN_DIST = 0.139  # mandatory registry constraint (XY center-to-center)

    _MEAN_RADIUS = 0.12
    _RADIUS_TOL = 0.04
    _ANG_SEP_DEG = 120.0
    _ANG_TOL_DEG = 25.0
    _SIDE_LEN_TOL_RATIO = 0.25
    _CENTER_TOL = 0.03  # "center_is_target": centroid close to patch center

    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (MUST use ONLY ColoredCube and GoalPatch) ---
        self.goalpatch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cube1 = ColoredCube(self._mojo)
        self.cube2 = ColoredCube(self._mojo)
        self.cube3 = ColoredCube(self._mojo)
        self.cubes = [self.cube1, self.cube2, self.cube3]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube1.set_color("red")
        self.cube2.set_color("blue")
        self.cube3.set_color("yellow")

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

        # 2. Deterministic RNG (may be used by the validation pipeline)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube1.set_color("red")
        self.cube2.set_color("blue")
        self.cube3.set_color("yellow")

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() is False)

        # Workspace bounds (REQUIRED)
        X_LO, X_HI = 0.40, 0.70
        Y_LO, Y_HI = -0.25, 0.25

        # Randomize patch position (still within workspace; centered-ish to keep reachable)
        patch_xy = np.array(
            [float(rng.uniform(0.48, 0.62)), float(rng.uniform(-0.08, 0.08))],
            dtype=float,
        )
        self.goalpatch.body.set_position(np.array([patch_xy[0], patch_xy[1], self._PATCH_Z], dtype=float))

        # Helper: quick "solved" check mirroring _success arrangement + annulus checks (XY-only)
        def _is_solved_xy(cube_xy_list):
            r_min_eff = max(self._ANNULUS_R_MIN, self._CUBE_PATCH_MIN_DIST)
            r_max_eff = self._ANNULUS_R_MAX

            cube_xy = [np.array(xy, dtype=float) for xy in cube_xy_list]
            r_arr = np.array([float(np.linalg.norm(xy - patch_xy)) for xy in cube_xy], dtype=float)

            # Per-cube annulus
            if not np.all((r_arr >= r_min_eff) & (r_arr <= r_max_eff)):
                return False

            # Radii near mean
            radius_ok = bool(np.all(np.abs(r_arr - self._MEAN_RADIUS) <= self._RADIUS_TOL))

            # Centroid near target
            centroid_xy = np.mean(np.stack(cube_xy, axis=0), axis=0)
            centroid_err = float(np.linalg.norm(centroid_xy - patch_xy))
            center_ok = centroid_err <= self._CENTER_TOL

            # Angular separation ~120
            angles = np.array([np.arctan2(xy[1] - patch_xy[1], xy[0] - patch_xy[0]) for xy in cube_xy], dtype=float)
            angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
            angles_sorted = np.sort(angles)
            deltas = np.diff(np.r_[angles_sorted, angles_sorted[0] + 2.0 * np.pi])
            deltas_deg = deltas * (180.0 / np.pi)
            ang_ok = bool(np.all(np.abs(deltas_deg - self._ANG_SEP_DEG) <= self._ANG_TOL_DEG))

            # Side lengths roughly equal
            Rm = float(np.mean(r_arr))
            expected_side = np.sqrt(3.0) * Rm
            sides = []
            for i in range(3):
                for j in range(i + 1, 3):
                    sides.append(float(np.linalg.norm(cube_xy[i] - cube_xy[j])))
            sides = np.array(sides, dtype=float)
            side_ok = bool(np.all(np.abs(sides - expected_side) <= (self._SIDE_LEN_TOL_RATIO * expected_side)))

            return bool(radius_ok and center_ok and ang_ok and side_ok)

        # Sample random cube positions; keep within workspace; avoid overlaps; reject solved layouts
        min_pair_dist = 0.055
        max_tries = 200
        cube_positions_xy = None

        for _ in range(max_tries):
            pts = []
            ok = True
            for _k in range(3):
                # Uniform over workspace (unsolved with extremely high probability)
                xy = np.array([float(rng.uniform(X_LO, X_HI)), float(rng.uniform(Y_LO, Y_HI))], dtype=float)
                # Simple pairwise separation
                if any(float(np.linalg.norm(xy - p)) < min_pair_dist for p in pts):
                    ok = False
                    break
                pts.append(xy)

            if not ok:
                continue

            # Ensure not in solved configuration
            if _is_solved_xy(pts):
                continue

            cube_positions_xy = pts
            break

        # Fallback: force unsolved by placing all cubes near the patch (violates annulus -> not success)
        if cube_positions_xy is None:
            cube_positions_xy = [
                np.array(
                    [float(np.clip(patch_xy[0] + rng.uniform(-0.03, 0.03), X_LO, X_HI)),
                     float(np.clip(patch_xy[1] + rng.uniform(-0.03, 0.03), Y_LO, Y_HI))],
                    dtype=float,
                )
                for _ in range(3)
            ]

        # Place cubes with random yaw
        for cube, xy in zip(self.cubes, cube_positions_xy):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            cube.set_pose(
                position=np.array([float(xy[0]), float(xy[1]), self._CUBE_Z], dtype=float),
                quat=np.array([q.x, q.y, q.z, q.w], dtype=float),
            )
            self._mojo.physics.forward()

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
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goalpatch"):
            return False

        patch_pos = self.goalpatch.body.get_position()
        patch_xy = np.array([patch_pos[0], patch_pos[1]], dtype=float)

        # --- Per-cube annulus checks ---
        r_min_eff = max(self._ANNULUS_R_MIN, self._CUBE_PATCH_MIN_DIST)
        r_max_eff = self._ANNULUS_R_MAX

        cube_xy = []
        cube_r = []
        cube_ok = []
        z_ok = []
        on_table = []
        for cube in self.cubes:
            p = cube.body.get_position()
            xy = np.array([p[0], p[1]], dtype=float)
            cube_xy.append(xy)

            r_i = float(np.linalg.norm(xy - patch_xy))
            cube_r.append(r_i)

            # Height/table contact (defensive; goal is on table)
            z_ok_i = abs(float(p[2]) - self._CUBE_Z) < 0.03
            z_ok.append(z_ok_i)

            on_table_i = cube.is_colliding(self.table.table_top)
            on_table.append(bool(on_table_i))

            annulus_ok = (r_i >= r_min_eff) and (r_i <= r_max_eff)
            cube_ok.append(bool(annulus_ok and z_ok_i and on_table_i))

        # --- Triangle arrangement check around patch ---
        # 1) Radii near mean (as in spec)
        r_arr = np.array(cube_r, dtype=float)
        radius_ok = bool(np.all(np.abs(r_arr - self._MEAN_RADIUS) <= self._RADIUS_TOL))

        # 2) Centroid near target center (center_is_target)
        centroid_xy = np.mean(np.stack(cube_xy, axis=0), axis=0)
        centroid_err = float(np.linalg.norm(centroid_xy - patch_xy))
        center_ok = centroid_err <= self._CENTER_TOL

        # 3) Angular separation ~120° (+/- tolerance)
        angles = np.array([np.arctan2(xy[1] - patch_xy[1], xy[0] - patch_xy[0]) for xy in cube_xy], dtype=float)
        angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
        angles_sorted = np.sort(angles)
        deltas = np.diff(np.r_[angles_sorted, angles_sorted[0] + 2.0 * np.pi])  # three gaps sum to 2pi
        deltas_deg = deltas * (180.0 / np.pi)
        ang_ok = bool(np.all(np.abs(deltas_deg - self._ANG_SEP_DEG) <= self._ANG_TOL_DEG))

        # 4) Side lengths roughly equal (equilateral-ish), using measured mean radius
        # Expected equilateral side length for circumradius R: s = sqrt(3) * R
        Rm = float(np.mean(r_arr))
        expected_side = np.sqrt(3.0) * Rm
        sides = []
        for i in range(3):
            for j in range(i + 1, 3):
                sides.append(float(np.linalg.norm(cube_xy[i] - cube_xy[j])))
        sides = np.array(sides, dtype=float)
        side_ok = bool(
            np.all(np.abs(sides - expected_side) <= (self._SIDE_LEN_TOL_RATIO * expected_side))
        )

        arrangement_ok = bool(radius_ok and center_ok and ang_ok and side_ok)

        # Released gripper (no grasp-state success conditions, but we must ensure not held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(all(cube_ok) and arrangement_ok)
        semantic_ok = True  # no semantic conditions for ColoredCube/GoalPatch

        # --- Subgoal staging (approximate using measurable conditions) ---
        # Stage 1-3: cube1 in annulus (proxy for "placed"), etc.
        # Stage 10: overall alignment (arrangement_ok).
        if cube_ok[0]:
            self._metric_stage(0, True)   # grasp/transport/place proxy group start (cube1)
            self._metric_stage(1, True)
            self._metric_stage(2, True)
        if cube_ok[1]:
            self._metric_stage(3, True)
            self._metric_stage(4, True)
            self._metric_stage(5, True)
        if cube_ok[2]:
            self._metric_stage(6, True)
            self._metric_stage(7, True)
            self._metric_stage(8, True)
        if arrangement_ok:
            self._metric_stage(9, True)   # align

        # Set success flag based on all conditions
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # Report radial errors and centroid error for easy debugging/evaluation.
        target_distance = {
            "cube1_radius_err_m": float(abs(r_arr[0] - self._MEAN_RADIUS)),
            "cube2_radius_err_m": float(abs(r_arr[1] - self._MEAN_RADIUS)),
            "cube3_radius_err_m": float(abs(r_arr[2] - self._MEAN_RADIUS)),
            "centroid_err_m": float(centroid_err),
            "mean_side_len_err_m": float(abs(float(np.mean(sides)) - expected_side)),
        }
        pose_error = {}

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})