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

class BlueCubesToBack(RoboEvalEnv, MetricRolloutEval):
    """
    Relocate every blue colored cube to the back of the workspace, defined as higher x values
    (away from the robot/camera). Non-blue cubes are irrelevant and may remain anywhere.

    Sequential Sub-Goals:
        1. Grasp any blue cube with either gripper.
        2. Lift the grasped blue cube off the table.
        3. Transport the blue cube toward the back (increase x).
        4. Place the blue cube within the back region (x >= 0.62).
        5. Release the blue cube after placing it stably in the back region.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"

    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Region definition (task goal)
    _BACK_REGION_BOUNDS = {
        "x_min": 0.62,
        "x_max": 0.70,
        "y_min": -0.25,
        "y_max": 0.25,
    }

    # Workspace (hard constraints)
    _WORKSPACE_X = (0.35, 0.75)
    _WORKSPACE_Y = (-0.30, 0.30)
    _CUBE_Z_ON_TABLE = 0.97

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (MUST use only ColoredCube)
        # Ensure at least two blue cubes exist
        self.cubes = [ColoredCube(self._mojo) for _ in range(6)]

        # Assign deterministic colors (MUST call set_color here)
        self._cube_colors = ["blue", "blue", "red", "green", "yellow", "purple"]
        for cube, c in zip(self.cubes, self._cube_colors):
            cube.set_color(c)

        # Cache blue subset indices for evaluation
        self._blue_indices = [i for i, c in enumerate(self._cube_colors) if c == "blue"]

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

        # 2. Deterministic RNG (required pattern; may be used by downstream randomizers)
        rng = np.random.default_rng(self.seed)

        # 3. MUST call set_color() in _on_reset as well
        for cube, c in zip(self.cubes, self._cube_colors):
            cube.set_color(c)

        # 4. Randomize objects into an UNSOLVED configuration within required workspace bounds.
        #    Requirement workspace bounds (override defensively here):
        #      X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        wsx = (0.40, 0.70)
        wsy = (-0.25, 0.25)
        z0 = self._CUBE_Z_ON_TABLE

        # Ensure _success() is False after reset by forcing ALL blue cubes to start NOT in back region.
        bounds = self._BACK_REGION_BOUNDS
        blue_x_max_unsolved = min(bounds["x_min"] - 0.03, wsx[1])  # strictly before back region

        # Simple collision-avoidance via rejection sampling in XY
        placed_xy = []
        min_sep = 0.055  # conservative for cubes

        def sample_xy(x_rng, y_rng, max_tries=500):
            for _ in range(max_tries):
                x = float(rng.uniform(x_rng[0], x_rng[1]))
                y = float(rng.uniform(y_rng[0], y_rng[1]))
                ok = True
                for (px, py) in placed_xy:
                    if (x - px) * (x - px) + (y - py) * (y - py) < (min_sep * min_sep):
                        ok = False
                        break
                if ok:
                    placed_xy.append((x, y))
                    return x, y
            # Fallback: accept last sample (still clipped to bounds)
            x = float(rng.uniform(x_rng[0], x_rng[1]))
            y = float(rng.uniform(y_rng[0], y_rng[1]))
            placed_xy.append((x, y))
            return x, y

        for i, cube in enumerate(self.cubes):
            if i in self._blue_indices:
                # Force blue cubes into the FRONT/MID region (unsolved): x < back x_min
                x_rng = (wsx[0], max(wsx[0], blue_x_max_unsolved))
                y_rng = wsy
            else:
                # Non-blue cubes anywhere in workspace
                x_rng = wsx
                y_rng = wsy

            x, y = sample_xy(x_rng, y_rng)

            pos = np.array(
                [
                    float(np.clip(x, wsx[0], wsx[1])),
                    float(np.clip(y, wsy[0], wsy[1])),
                    z0,
                ],
                dtype=float,
            )

            # Random yaw for variety (keeps stable on table)
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # [w, x, y, z]
            quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)  # to [x,y,z,w]

            cube.set_pose(position=pos, quat=quat)

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _point_to_rect_distance_xy(p_xy: np.ndarray, bounds: Dict) -> float:
        """Euclidean distance in XY to the closest point inside the axis-aligned rectangle."""
        x, y = float(p_xy[0]), float(p_xy[1])
        cx = min(max(x, bounds["x_min"]), bounds["x_max"])
        cy = min(max(y, bounds["y_min"]), bounds["y_max"])
        dx = x - cx
        dy = y - cy
        return float(np.sqrt(dx * dx + dy * dy))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_blue_indices"):
            return False
        if len(self._blue_indices) == 0:
            return False

        bounds = self._BACK_REGION_BOUNDS

        # Position check for ALL blue cubes
        positions_ok = True
        per_blue_dist = {}
        for bi in self._blue_indices:
            cube = self.cubes[bi]
            pos = cube.body.get_position()

            in_x = (bounds["x_min"] <= pos[0] <= bounds["x_max"])
            in_y = (bounds["y_min"] <= pos[1] <= bounds["y_max"])

            d = self._point_to_rect_distance_xy(pos[:2], bounds)
            per_blue_dist[f"blue_cube_{bi}"] = d

            if not (in_x and in_y):
                positions_ok = False

        # Semantic/content check (not applicable beyond color subset selection)
        semantic_ok = True
        if any(self.cubes[i].get_color() != "blue" for i in self._blue_indices):
            semantic_ok = False

        # Released: ensure no BLUE cube is currently held
        released = not any(
            self.robot.is_gripper_holding_object(self.cubes[i], g)
            for i in self._blue_indices
            for g in self.robot.grippers
        )

        # Subgoal staging (best-effort inference from observable state)
        # 0: grasp (any blue held)
        any_blue_held = any(
            self.robot.is_gripper_holding_object(self.cubes[i], g)
            for i in self._blue_indices
            for g in self.robot.grippers
        )
        if any_blue_held:
            self._metric_stage(0, True)

        # 1: lift (any held blue above table height by margin)
        lifted = False
        if any_blue_held:
            for i in self._blue_indices:
                z = float(self.cubes[i].body.get_position()[2])
                if z > (self._CUBE_Z_ON_TABLE + 0.02):
                    lifted = True
                    break
        if lifted:
            self._metric_stage(1, True)

        # 2: transport (any held blue at/behind x_min)
        transported = False
        if any_blue_held:
            for i in self._blue_indices:
                x = float(self.cubes[i].body.get_position()[0])
                if x >= bounds["x_min"]:
                    transported = True
                    break
        if transported:
            self._metric_stage(2, True)

        # 3: place (any blue in region AND on/near table surface)
        placed_any = False
        for i in self._blue_indices:
            pos = self.cubes[i].body.get_position()
            if (bounds["x_min"] <= pos[0] <= bounds["x_max"]) and (bounds["y_min"] <= pos[1] <= bounds["y_max"]):
                if abs(float(pos[2]) - self._CUBE_Z_ON_TABLE) < 0.03:
                    placed_any = True
                    break
        if placed_any:
            self._metric_stage(3, True)

        # 4: release (final): all blue cubes in region and no blue cube held
        if positions_ok and semantic_ok and released:
            self._metric_stage(4, True)

        # Success requires all blue cubes in back region and not held
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=per_blue_dist,
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})