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

class CubeRadialPlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes on the table so they form an evenly spaced ring around
    the center of a green target patch. The cubes should be placed at the same radius
    from the patch center and separated by 120 degrees.

    Sequential Sub-Goals:
        1. All cubes are on the target ring radius (within tolerance).
        2. Cubes are evenly spaced angularly (~120° separation, within tolerance).
        3. Cubes maintain a minimum pairwise separation (avoid clustering).
        4. All cubes are released (not held by any gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters from specification
    _TARGET_CENTER_XY = np.array([0.55, 0.0])
    _RADIUS_M = 0.14
    _RADIUS_TOL_M = 0.02
    _EXPECTED_SEP_DEG = 120.0
    _SEP_TOL_DEG = 15.0
    _MIN_PAIRWISE_DIST_M = 0.06

    # Asset constraints (auto-generated): keep cube centers far enough from GoalPatch center in XY
    # We satisfy this by placing cubes at radius 0.14m (still within radius tolerance about 0.12±0.02).
    _SAFE_GOAL_RADIUS_M = 0.14

    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Required assets only: GoalPatch + ColoredCube
        self.goal_patch_green = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3]

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("yellow")

        # Initialize metrics with FULL signature (required)
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

        # 2) Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() again on reset
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")
        self.cube_3.set_color("yellow")

        # Place goal patch at fixed center (on-table Z for GoalPatch is 0.951)
        patch_pos = np.array([self._TARGET_CENTER_XY[0], self._TARGET_CENTER_XY[1], 0.951])
        self.goal_patch_green.body.set_position(patch_pos)
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # RANDOM UNSOLVED STATE:
        # Keep cubes within workspace bounds and on-table Z=0.97, but ensure the arrangement is NOT solved.
        # We keep the (safe) radius fixed at 0.14m, and intentionally break angular spacing by clustering
        # two cubes close together, which also violates the minimum pairwise distance.
        phi0 = float(rng.uniform(0.0, 2.0 * np.pi))
        delta = float(rng.uniform(np.deg2rad(5.0), np.deg2rad(20.0)))  # close pair -> unsolved
        thetas = [phi0, phi0 + delta, phi0 + 2.0 * np.pi / 3.0]  # not evenly spaced (not 120/120/120)

        r = self._SAFE_GOAL_RADIUS_M
        positions = []
        for th in thetas:
            x = float(self._TARGET_CENTER_XY[0] + r * np.cos(th))
            y = float(self._TARGET_CENTER_XY[1] + r * np.sin(th))

            # Enforce workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25]
            x = float(np.clip(x, 0.40, 0.70))
            y = float(np.clip(y, -0.25, 0.25))

            positions.append(np.array([x, y, 0.97], dtype=float))

        # Place cubes sequentially for stability with random yaw orientations
        for cube, pos in zip(self.cubes, positions):
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw).elements  # (w,x,y,z)
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
            cube.set_pose(position=pos, quat=quat_xyzw)
            self._mojo.physics.forward()
            self._mojo.physics.forward()

        # Physics settling
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _wrap_to_2pi(a: float) -> float:
        a = float(a) % (2.0 * np.pi)
        return a

    def _angles_and_seps_deg(self, center_xy: np.ndarray, xy_points: np.ndarray):
        # xy_points: (N,2)
        angles = []
        for i in range(xy_points.shape[0]):
            dx, dy = xy_points[i, 0] - center_xy[0], xy_points[i, 1] - center_xy[1]
            angles.append(self._wrap_to_2pi(np.arctan2(dy, dx)))
        angles = np.array(angles, dtype=float)

        # sort angles for neighbor separations with wraparound
        order = np.argsort(angles)
        angles_sorted = angles[order]
        diffs = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 2.0 * np.pi]))
        seps_deg = np.degrees(diffs)
        return angles, seps_deg, order

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch_green"):
            return False

        # Positions
        center_xy = self.goal_patch_green.body.get_position()[:2].copy()
        cube_pos = np.array([c.body.get_position() for c in self.cubes], dtype=float)
        cube_xy = cube_pos[:, :2]
        cube_z = cube_pos[:, 2]

        # (A) On-table (approx) check: keep it lenient; primary constraints are XY ring constraints.
        on_table_z = np.all((cube_z > 0.955) & (cube_z < 1.00))

        # (B) Radius-from-center checks (task spec: 0.12 ± 0.02 in table XY)
        dists = np.linalg.norm(cube_xy - center_xy[None, :], axis=1)
        radial_errors = np.abs(dists - self._RADIUS_M)
        radial_ok = bool(np.all(radial_errors <= self._RADIUS_TOL_M))

        # (C) Even angular spacing around center (120° ± 15°, wraparound)
        _, seps_deg, _ = self._angles_and_seps_deg(center_xy, cube_xy)
        sep_errors = np.abs(seps_deg - self._EXPECTED_SEP_DEG)
        angular_ok = bool(np.all(sep_errors <= self._SEP_TOL_DEG))

        # (D) Pairwise min distance (in XY plane)
        min_pairwise = float("inf")
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                dij = float(np.linalg.norm(cube_xy[i] - cube_xy[j]))
                if dij < min_pairwise:
                    min_pairwise = dij
        pairwise_ok = bool(min_pairwise >= self._MIN_PAIRWISE_DIST_M)

        # Released (robot not holding any cube) - implemented in generic gripper-safe way
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage metrics
        if on_table_z and radial_ok:
            self._metric_stage(0, True)
        if on_table_z and radial_ok and angular_ok:
            self._metric_stage(1, True)
        if on_table_z and radial_ok and angular_ok and pairwise_ok:
            self._metric_stage(2, True)
        if on_table_z and radial_ok and angular_ok and pairwise_ok and released:
            self._metric_stage(3, True)

        positions_ok = bool(on_table_z and radial_ok and angular_ok and pairwise_ok)

        # Final success: arrangement + released
        self._success_check = bool(positions_ok and released)

        # Finalize metrics (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_radial_error_m": float(np.mean(radial_errors)),
                "max_radial_error_m": float(np.max(radial_errors)),
                "min_pairwise_dist_m": float(min_pairwise),
                "mean_sep_error_deg": float(np.mean(sep_errors)),
                "max_sep_error_deg": float(np.max(sep_errors)),
            },
            pose_error={
                "cube_1_radial_error_m": float(radial_errors[0]),
                "cube_2_radial_error_m": float(radial_errors[1]),
                "cube_3_radial_error_m": float(radial_errors[2]),
                "mean_sep_error_deg": float(np.mean(sep_errors)),
                "max_sep_error_deg": float(np.max(sep_errors)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})