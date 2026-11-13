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

class RedCubeInFrontOfBlueCube(RoboEvalEnv, MetricRolloutEval):
    """Place red directly in front of blue.

Success Criteria:
- The red cube is in front of the blue cube by at least the configured X separation.
- The red and blue cubes are aligned in Y and both collide with the table top.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal condition parameters (from task spec)
    _FRONT_IS_NEGATIVE_X = True
    _MIN_X_SEPARATION_M = 0.04
    _Y_ALIGN_TOL_M = 0.05

    # A convenient "nominal" solved relative offset (not required, used for pose_error metrics)
    _NOMINAL_DX_M = 0.06  # blue_x - red_x
    _NOMINAL_DY_M = 0.00

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # --- Props: MUST use ONLY ColoredCube ---
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Park them safely within workspace (will be set properly on reset)
        self.red_cube.set_pose(
            position=np.array([0.55, 0.08, 0.97]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.blue_cube.set_pose(
            position=np.array([0.55, -0.08, 0.97]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )

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

    def sample_initial_state(self, rng: np.random.Generator) -> Dict:
        """
        Produce an UNSOLVED randomized initial state (used by external randomization pipelines).

        Workspace bounds: X [0.35, 0.75], Y [-0.30, 0.30]
        """
        def _rand_xy():
            x = rng.uniform(0.40, 0.70)
            y = rng.uniform(-0.25, 0.25)
            return np.array([x, y, 0.97], dtype=float)

        # Sample until not already solved and not overlapping too closely
        for _ in range(200):
            red_pos = _rand_xy()
            blue_pos = _rand_xy()

            # Keep some minimum spacing between centers to avoid near-overlap at spawn
            if np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.06:
                continue

            dx = blue_pos[0] - red_pos[0]  # positive means red is "in front" when front = -X
            dy = red_pos[1] - blue_pos[1]

            solved = (dx >= self._MIN_X_SEPARATION_M) and (abs(dy) <= self._Y_ALIGN_TOL_M)
            if solved:
                continue

            return {"red_pos": red_pos, "blue_pos": blue_pos}

        # Fallback (still unsolved)
        return {
            "red_pos": np.array([0.62, 0.18, 0.97]),
            "blue_pos": np.array([0.50, -0.18, 0.97]),
        }

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization object (even if not used here)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3. Place objects in RANDOM UNSOLVED positions within bounds
        def _rand_xy():
            x = rng.uniform(0.40, 0.70)
            y = rng.uniform(-0.25, 0.25)
            return np.array([x, y, 0.97], dtype=float)

        red_pos = None
        blue_pos = None
        for _ in range(400):
            rp = _rand_xy()
            bp = _rand_xy()

            # avoid near-overlap at spawn
            if np.linalg.norm(rp[:2] - bp[:2]) < 0.06:
                continue

            dx = float(bp[0] - rp[0])
            dy = float(rp[1] - bp[1])
            solved = (dx >= self._MIN_X_SEPARATION_M) and (abs(dy) <= self._Y_ALIGN_TOL_M)
            if solved:
                continue

            red_pos, blue_pos = rp, bp
            break

        # Deterministic fallback if sampling fails (still unsolved and within bounds)
        if red_pos is None or blue_pos is None:
            red_pos = np.array([0.66, 0.20, 0.97], dtype=float)
            blue_pos = np.array([0.44, -0.20, 0.97], dtype=float)

        # Layer-by-layer placement for stability
        self.blue_cube.set_pose(position=blue_pos, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        self._mojo.physics.forward()

        self.red_cube.set_pose(position=red_pos, quat=np.array([1.0, 0.0, 0.0, 0.0]))
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
        if not hasattr(self, "red_cube") or not hasattr(self, "blue_cube"):
            return False

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        # Goal condition 1: red is in front of blue along X, with front = -X
        # => red_x smaller than blue_x by at least min separation
        dx = float(blue_pos[0] - red_pos[0])
        in_front_ok = dx >= self._MIN_X_SEPARATION_M

        # Goal condition 2: laterally aligned along Y within tolerance
        dy = float(red_pos[1] - blue_pos[1])
        y_aligned_ok = abs(dy) <= self._Y_ALIGN_TOL_M

        # Optional sanity: both cubes are on the table (table collision is reliable)
        on_table_ok = bool(
            self.red_cube.is_colliding(self.table.table_top)
            and self.blue_cube.is_colliding(self.table.table_top)
        )

        positions_ok = bool(in_front_ok and y_aligned_ok and on_table_ok)

        # No semantic constraints beyond color (colors are fixed at reset)
        semantic_ok = True

        # Must be released (no gripper holding either cube)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Mark stages complete only when the final arrangement is achieved
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        red_to_blue_dist = float(distance(self.red_cube.body, self.blue_cube.body))
        pose_error = {
            "x_front_gap_error_m": float(max(0.0, self._MIN_X_SEPARATION_M - dx)),
            "y_alignment_error_m": float(abs(dy)),
            "dx_from_nominal_m": float(abs(dx - self._NOMINAL_DX_M)),
            "dy_from_nominal_m": float(abs(dy - self._NOMINAL_DY_M)),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_blue_distance_m": red_to_blue_dist,
            },
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})