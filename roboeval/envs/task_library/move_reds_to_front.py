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

class RedCubeFrontRelocationTask(RoboEvalEnv, MetricRolloutEval):
    """
    Relocate all red colored cubes from their initial scattered positions to the front region
    of the workspace (closer to the robot) while leaving non-red cubes unconstrained.

    Sequential Sub-Goals:
        1. Acquire a stable grasp on a red cube (repeat as needed across cubes).
        2. Move the grasped red cube toward the front region of the workspace.
        3. Place the red cube down in the front region.
        4. Release the red cube after placing it in the front region.
        5. (Optional) Repeat with the other gripper in a bimanual setup.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Region definition: "front_of_workspace" per task spec uses an X-threshold check.
    # Success requires: x <= _X_FRONT_THRESHOLD (with tolerance).
    _X_FRONT_THRESHOLD = 0.50
    _X_TOL = 0.014  # task constraint tolerance

    # We keep 8 stages to mirror the provided subgoals (left/right), even if the robot is single-arm.
    _NUM_SUBGOALS = 8

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Create cubes (ColoredCube only) ---
        # Multiple red cubes + other colors. Non-red cubes are unconstrained by the goal.
        self.cubes = []
        for _ in range(6):
            self.cubes.append(ColoredCube(self._mojo))

        # Define which indices are "red cubes" for the task (do not infer at runtime).
        self._red_indices = [0, 1, 2]
        self._nonred_indices = [3, 4, 5]

        # Assign colors (REQUIRED: call set_color() in _initialize_env)
        for i in self._red_indices:
            self.cubes[i].set_color("red")
        nonred_colors = ["blue", "green", "yellow"]
        for i, c in zip(self._nonred_indices, nonred_colors):
            self.cubes[i].set_color(c)

        # Store a deterministic "goal arrangement" (front region) used for pose-error metrics only.
        # Note: the *actual* goal condition is region membership (x <= threshold), not exact poses.
        self._red_goal_positions = {
            0: np.array([0.42, -0.12, 0.97]),
            1: np.array([0.45,  0.00, 0.97]),
            2: np.array([0.48,  0.12, 0.97]),
        }

        # ⚠️ CRITICAL: Initialize metrics with FULL signature
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True
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
            track_collisions=True
        )

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # Re-apply colors (REQUIRED: call set_color() in _on_reset)
        for i in self._red_indices:
            self.cubes[i].set_color("red")
        nonred_colors = ["blue", "green", "yellow"]
        for i, c in zip(self._nonred_indices, nonred_colors):
            self.cubes[i].set_color(c)

        # 3. Place objects in RANDOM UNSOLVED positions within workspace bounds.
        #    Ensure at least one red cube is outside the "front_of_workspace" region:
        #    require x > (_X_FRONT_THRESHOLD + _X_TOL) so _success() returns False after reset.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        # Simple rejection sampling to keep cubes reasonably separated on the table.
        min_xy_sep = 0.07

        def sample_xy(existing_xy: list[np.ndarray], x_low: float, x_high: float) -> np.ndarray:
            for _ in range(2000):
                x = float(rng.uniform(x_low, x_high))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)
                if all(np.linalg.norm(p - q) >= min_xy_sep for q in existing_xy):
                    return p
            # Fallback (should be rare): accept without separation constraint.
            return np.array([float(rng.uniform(x_low, x_high)), float(rng.uniform(y_min, y_max))], dtype=float)

        placed_xy: list[np.ndarray] = []

        # Force one red cube to start "not in front region" (unsolved).
        # Keep it comfortably beyond tolerance.
        violating_red = int(rng.choice(self._red_indices))
        for i in self._red_indices:
            if i == violating_red:
                xy = sample_xy(placed_xy, max(self._X_FRONT_THRESHOLD + self._X_TOL + 0.01, x_min), x_max)
            else:
                # Other red cubes random anywhere (may or may not be in front); task still unsolved due to violator.
                xy = sample_xy(placed_xy, x_min, x_max)
            placed_xy.append(xy)
            pos = np.array([xy[0], xy[1], z_base], dtype=float)
            self.cubes[i].set_pose(position=pos, quat=np.array([0, 0, 0, 1]))

        # Place non-red cubes randomly anywhere within bounds too.
        for i in self._nonred_indices:
            xy = sample_xy(placed_xy, x_min, x_max)
            placed_xy.append(xy)
            pos = np.array([xy[0], xy[1], z_base], dtype=float)
            self.cubes[i].set_pose(position=pos, quat=np.array([0, 0, 0, 1]))

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
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identify task-relevant cubes by fixed indices; also verify they are still red.
        red_cubes = [self.cubes[i] for i in self._red_indices]

        # Semantic verification: the designated red cubes must actually be red.
        semantic_ok = all(c.get_color() == "red" for c in red_cubes)

        # Positional goal: every red cube must be in the "front_of_workspace" region:
        # x <= threshold (with tolerance).
        positions_ok = True
        x_violations = {}
        for k, c in enumerate(red_cubes):
            x = float(c.body.get_position()[0])
            violation = max(0.0, x - (self._X_FRONT_THRESHOLD + self._X_TOL))
            x_violations[f"red_cube_{k}_x_violation"] = float(violation)
            if x > (self._X_FRONT_THRESHOLD + self._X_TOL):
                positions_ok = False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Call metric stages when goal is achieved (single-shot marking; stages are repeated subgoals).
        if positions_ok and semantic_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: region-distance style target (how far past the front threshold each red cube is).
        # Pose-error: same scalar here, since goal is region membership (not exact target poses).
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dict(x_violations),
            pose_error=dict(x_violations),
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})