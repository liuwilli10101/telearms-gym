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

class CubeCrossPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes into a cross pattern on the table:
    one cube north (+Y), one south (-Y), one east (+X), and one west (-X)
    of a shared center point.

    Sequential Sub-Goals:
        1. Grasp the cube intended for the north (+Y) position.
        2. Place the north cube to the +Y side of the cross center.
        3. Grasp the cube intended for the south (-Y) position.
        4. Place the south cube to the -Y side of the cross center.
        5. Grasp the cube intended for the east (+X) position.
        6. Place the east cube to the +X side of the cross center.
        7. Grasp the cube intended for the west (-X) position.
        8. Place the west cube to the -X side of the cross center.
        9. Align positions so both pairs share the same cross center.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace/task constants
    _CENTER_XY = np.array([0.55, 0.00])
    _ARM_OFFSET = 0.05  # 5cm from center; yields 10cm separation between opposite arms
    _CUBE_Z = 0.97

    # Goal tolerances/constraints (from spec)
    _MIN_DELTA = 0.08
    _ALIGN_TOL = 0.02
    _CENTER_TOL = 0.02

    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Props from preset
        self.table = self._preset.get_props(Table)[0]

        # Create required assets (ONLY ColoredCube)
        self.cube_north = ColoredCube(self._mojo)
        self.cube_south = ColoredCube(self._mojo)
        self.cube_east = ColoredCube(self._mojo)
        self.cube_west = ColoredCube(self._mojo)

        self.cubes = [self.cube_north, self.cube_south, self.cube_east, self.cube_west]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = {
            "north": "red",
            "south": "blue",
            "east": "green",
            "west": "yellow",
        }
        self.cube_north.set_color(self._cube_colors["north"])
        self.cube_south.set_color(self._cube_colors["south"])
        self.cube_east.set_color(self._cube_colors["east"])
        self.cube_west.set_color(self._cube_colors["west"])

        # Deterministic goal poses (also used for metrics/pose_error)
        cx, cy = float(self._CENTER_XY[0]), float(self._CENTER_XY[1])
        d = float(self._ARM_OFFSET)
        self._goal_pos = {
            "north": np.array([cx, cy + d, self._CUBE_Z]),
            "south": np.array([cx, cy - d, self._CUBE_Z]),
            "east":  np.array([cx + d, cy, self._CUBE_Z]),
            "west":  np.array([cx - d, cy, self._CUBE_Z]),
        }

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization handle (even if not used here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_north.set_color(self._cube_colors["north"])
        self.cube_south.set_color(self._cube_colors["south"])
        self.cube_east.set_color(self._cube_colors["east"])
        self.cube_west.set_color(self._cube_colors["west"])

        # 3. Randomize objects into UNSOLVED poses within workspace bounds
        # Workspace bounds (spec)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_Z)

        # Keep cubes reasonably separated to avoid initial interpenetration
        min_pair_dist = 0.06

        def sample_xy() -> np.ndarray:
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def pairwise_ok(xys) -> bool:
            for i in range(len(xys)):
                for j in range(i + 1, len(xys)):
                    if float(np.linalg.norm(xys[i] - xys[j])) < float(min_pair_dist):
                        return False
            return True

        def would_be_success(pn, ps, pe, pw) -> bool:
            # Replicate _success() geometric conditions (without side effects / physics queries)
            north_above_south = (pn[1] - ps[1]) >= self._MIN_DELTA
            ns_x_aligned = abs(pn[0] - ps[0]) <= self._ALIGN_TOL

            east_right_of_west = (pe[0] - pw[0]) >= self._MIN_DELTA
            ew_y_aligned = abs(pe[1] - pw[1]) <= self._ALIGN_TOL

            mid_ns = 0.5 * (pn + ps)
            mid_ew = 0.5 * (pe + pw)
            center_consistent = (
                abs(mid_ns[0] - mid_ew[0]) <= self._CENTER_TOL
                and abs(mid_ns[1] - mid_ew[1]) <= self._CENTER_TOL
            )
            return bool(
                north_above_south
                and ns_x_aligned
                and east_right_of_west
                and ew_y_aligned
                and center_consistent
            )

        def too_close_to_goal(xy: np.ndarray, key: str, tol: float = 0.03) -> bool:
            return float(np.linalg.norm(xy - self._goal_pos[key][:2])) <= float(tol)

        # Rejection sample until definitely unsolved
        pn = ps = pe = pw = None
        for _ in range(500):
            xys = [sample_xy() for _ in range(4)]
            if not pairwise_ok(xys):
                continue

            pn, ps, pe, pw = xys  # assign in cube order: north, south, east, west

            # Ensure not trivially already at intended goals
            if (
                too_close_to_goal(pn, "north")
                and too_close_to_goal(ps, "south")
                and too_close_to_goal(pe, "east")
                and too_close_to_goal(pw, "west")
            ):
                continue

            # Ensure the geometric success conditions are NOT satisfied
            if would_be_success(pn, ps, pe, pw):
                continue

            break

        # Fallback (still guaranteed unsolved): collapse N/S x-alignment by forcing mismatch
        if pn is None:
            pn = np.array([0.45, 0.20], dtype=float)
            ps = np.array([0.65, 0.20], dtype=float)  # same y => fails north_above_south
            pe = np.array([0.50, -0.10], dtype=float)
            pw = np.array([0.60, -0.15], dtype=float)

        # Random yaw orientation (optional; does not affect success)
        def random_quat_z() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            qz = Quaternion(axis=[0, 0, 1], angle=yaw)
            return np.array([qz.x, qz.y, qz.z, qz.w], dtype=float)

        self.cube_north.set_pose(position=np.array([pn[0], pn[1], z], dtype=float), quat=random_quat_z())
        self.cube_south.set_pose(position=np.array([ps[0], ps[1], z], dtype=float), quat=random_quat_z())
        self.cube_east.set_pose(position=np.array([pe[0], pe[1], z], dtype=float), quat=random_quat_z())
        self.cube_west.set_pose(position=np.array([pw[0], pw[1], z], dtype=float), quat=random_quat_z())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy(obj) -> np.ndarray:
        p = obj.body.get_position()
        return np.array([float(p[0]), float(p[1])])

    @staticmethod
    def _pos(obj) -> np.ndarray:
        p = obj.body.get_position()
        return np.array([float(p[0]), float(p[1]), float(p[2])])

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        pn = self._pos(self.cube_north)
        ps = self._pos(self.cube_south)
        pe = self._pos(self.cube_east)
        pw = self._pos(self.cube_west)

        # Basic table contact / height sanity (spatial task)
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)
        z_ok = all(abs(self._pos(c)[2] - self._CUBE_Z) <= 0.03 for c in self.cubes)

        # Goal conditions (from spec)
        north_above_south = (pn[1] - ps[1]) >= self._MIN_DELTA
        ns_x_aligned = abs(pn[0] - ps[0]) <= self._ALIGN_TOL

        east_right_of_west = (pe[0] - pw[0]) >= self._MIN_DELTA
        ew_y_aligned = abs(pe[1] - pw[1]) <= self._ALIGN_TOL

        mid_ns = 0.5 * (pn[:2] + ps[:2])
        mid_ew = 0.5 * (pe[:2] + pw[:2])
        center_consistent = (
            abs(mid_ns[0] - mid_ew[0]) <= self._CENTER_TOL
            and abs(mid_ns[1] - mid_ew[1]) <= self._CENTER_TOL
        )

        positions_ok = bool(
            on_table
            and z_ok
            and north_above_south
            and ns_x_aligned
            and east_right_of_west
            and ew_y_aligned
            and center_consistent
        )

        semantic_ok = True  # no semantic constraints for ColoredCube beyond placement

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Metric subgoals (coarse proxies for progress)
        # 0-7: each cube at/near its intended arm position; 8: final cross consistency
        def near_xy(cube, key, tol=0.03) -> bool:
            return float(np.linalg.norm(self._xy(cube) - self._goal_pos[key][:2])) <= float(tol)

        self._metric_stage(0, near_xy(self.cube_north, "north"))
        self._metric_stage(1, near_xy(self.cube_north, "north"))
        self._metric_stage(2, near_xy(self.cube_south, "south"))
        self._metric_stage(3, near_xy(self.cube_south, "south"))
        self._metric_stage(4, near_xy(self.cube_east, "east"))
        self._metric_stage(5, near_xy(self.cube_east, "east"))
        self._metric_stage(6, near_xy(self.cube_west, "west"))
        self._metric_stage(7, near_xy(self.cube_west, "west"))
        self._metric_stage(8, center_consistent and ns_x_aligned and ew_y_aligned)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        target_distance = {
            "north_xy": float(np.linalg.norm(self._xy(self.cube_north) - self._goal_pos["north"][:2])),
            "south_xy": float(np.linalg.norm(self._xy(self.cube_south) - self._goal_pos["south"][:2])),
            "east_xy": float(np.linalg.norm(self._xy(self.cube_east) - self._goal_pos["east"][:2])),
            "west_xy": float(np.linalg.norm(self._xy(self.cube_west) - self._goal_pos["west"][:2])),
            "center_midpoint_ns_to_ew": float(np.linalg.norm(mid_ns - mid_ew)),
        }
        pose_error = {
            "north_pos_err": float(np.linalg.norm(pn - self._goal_pos["north"])),
            "south_pos_err": float(np.linalg.norm(ps - self._goal_pos["south"])),
            "east_pos_err": float(np.linalg.norm(pe - self._goal_pos["east"])),
            "west_pos_err": float(np.linalg.norm(pw - self._goal_pos["west"])),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})