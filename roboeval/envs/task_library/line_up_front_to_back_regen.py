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

class FrontToBackCubeLine(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a straight line along the front-to-back direction,
    with one cube closest to the robot (front) and one farthest (back). Color order is
    FIXED: red is the front cube (closest to the robot, lowest X), green is in the middle,
    and blue is the back cube (farthest, highest X). The three cubes must share the same Y,
    be ordered red -> green -> blue along +X with a gap between each, and not touch.

    Workspace convention enforced:
      - Front/Back axis is X (index 0): front = lower X, back = higher X
      - Left/Right axis is Y (index 1): alignment requires same Y

    Sequential Sub-Goals:
        1. Grasp any one cube to begin arranging the line.
        2. Move a cube to its approximate front/middle/back X position while keeping Y near the shared line.
        3. Place the cube on the table at the intended location (no stacking).
        4. Adjust positions so all cubes share approximately the same Y and are ordered by X from front to back.
        5. Release cubes after final placement and alignment.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal condition parameters from task specification
    _Y_ALIGN_TOL = 0.03
    _MIN_X_SEP = 0.05

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets: MUST use ONLY ColoredCube ---
        self.red_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.green_cube, self.blue_cube]

        # Cache goal poses (set in _on_reset)
        self._goal_positions = {}

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

    def goal_specification(self) -> Dict:
        """Structured goal description for evaluators/UI."""
        return {
            "task_name": "front-to-back-cube-line",
            "axis_conventions": {
                "front_back_axis": "x",
                "front_is_lower_x": True,
                "left_right_axis": "y",
            },
            "objects": [
                {"name": "red_cube", "color": "red"},
                {"name": "green_cube", "color": "green"},
                {"name": "blue_cube", "color": "blue"},
            ],
            "goal_conditions": [
                {
                    "type": "arrangement",
                    "relation": "aligned",
                    "axis": "y",
                    "tolerance": self._Y_ALIGN_TOL,
                    "objects": ["red_cube", "green_cube", "blue_cube"],
                },
                {
                    "type": "relative",
                    "relation": "in_front_of",
                    "axis": "x",
                    "subject": "red_cube",
                    "target": "green_cube",
                    "required_sign": "<",
                    "min_separation": self._MIN_X_SEP,
                },
                {
                    "type": "relative",
                    "relation": "in_front_of",
                    "axis": "x",
                    "subject": "green_cube",
                    "target": "blue_cube",
                    "required_sign": "<",
                    "min_separation": self._MIN_X_SEP,
                },
            ],
        }

    def sample_initial_state(self) -> Dict:
        """
        Declares the (randomized) initial-state intent. The actual evaluation pipeline
        may apply additional randomization/perturbations.
        """
        return {
            "state_type": "randomized",
            "workspace_bounds": {"x": [0.40, 0.70], "y": [-0.25, 0.25], "z_base": 0.97},
            "objects": ["red_cube", "green_cube", "blue_cube"],
            "notes": "Cubes start separated enough to be individually graspable.",
        }

    def _on_reset(self):
        # REQUIRED: Re-initialize metrics (on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # REQUIRED: Deterministic RNG pattern (even if goal placement is deterministic)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        # --- Define GOAL poses (for metrics), but DO NOT start in goal configuration ---
        # Keep within workspace bounds: X[0.40,0.70], Y[-0.25,0.25], Z base=0.97.
        y_goal = 0.00
        x_front_goal, x_mid_goal, x_back_goal = 0.45, 0.55, 0.65
        self._goal_positions = {
            "red_cube": np.array([x_front_goal, y_goal, 0.97]),
            "green_cube": np.array([x_mid_goal, y_goal, 0.97]),
            "blue_cube": np.array([x_back_goal, y_goal, 0.97]),
        }

        # --- Randomize UNSOLVED initial cube poses within bounds ---
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = 0.97

        # Conservative planar spacing to avoid overlaps/collisions.
        # (Larger than _MIN_X_SEP to reduce accidental contacts.)
        min_planar_sep = 0.07

        def _positions_ok_and_unsolved(rp, gp, bp) -> bool:
            # Ensure within bounds
            for p in (rp, gp, bp):
                if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and abs(p[2] - z) < 1e-9):
                    return False

            # Ensure separated
            def d2(a, b):
                v = a[:2] - b[:2]
                return float(v @ v)

            if d2(rp, gp) < min_planar_sep**2 or d2(gp, bp) < min_planar_sep**2 or d2(rp, bp) < min_planar_sep**2:
                return False

            # Ensure NOT in success arrangement
            y_ok = (
                abs(rp[1] - gp[1]) <= self._Y_ALIGN_TOL
                and abs(gp[1] - bp[1]) <= self._Y_ALIGN_TOL
                and abs(rp[1] - bp[1]) <= self._Y_ALIGN_TOL
            )
            x_order_ok = (rp[0] + self._MIN_X_SEP) < gp[0] and (gp[0] + self._MIN_X_SEP) < bp[0]
            positions_ok = bool(y_ok and x_order_ok)
            return not positions_ok

        # Sample until we get an unsolved, separated configuration
        rp = gp = bp = None
        for _ in range(500):
            rp = np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), z])
            gp = np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), z])
            bp = np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), z])
            if _positions_ok_and_unsolved(rp, gp, bp):
                break
        else:
            # Deterministic fallback that is guaranteed unsolved (violates Y alignment).
            rp = np.array([0.45, -0.18, z])
            gp = np.array([0.55, 0.02, z])
            bp = np.array([0.65, 0.18, z])

        # Small random yaw (still on table), deterministic via rng
        def _yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            # z-axis yaw quaternion: [x, y, z, w]
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        self.red_cube.set_pose(position=rp, quat=_yaw_quat())
        self.green_cube.set_pose(position=gp, quat=_yaw_quat())
        self.blue_cube.set_pose(position=bp, quat=_yaw_quat())

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        NUM_SUBGOALS = 5
        for idx in range(NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_positions"):
            return False

        # Fetch positions
        r = self.red_cube.body.get_position()
        g = self.green_cube.body.get_position()
        b = self.blue_cube.body.get_position()

        # Condition 1: lateral alignment along Y (index 1)
        y_ok = (
            abs(r[1] - g[1]) <= self._Y_ALIGN_TOL
            and abs(g[1] - b[1]) <= self._Y_ALIGN_TOL
            and abs(r[1] - b[1]) <= self._Y_ALIGN_TOL
        )

        # Condition 2 & 3: front-to-back order along X (index 0), front = lower X
        x_order_ok = (r[0] + self._MIN_X_SEP) < g[0] and (g[0] + self._MIN_X_SEP) < b[0]

        # Additional safety: ensure cubes are not colliding with each other (not stacked/overlapping)
        no_pair_collision = (
            (not self.red_cube.is_colliding(self.green_cube))
            and (not self.green_cube.is_colliding(self.blue_cube))
            and (not self.red_cube.is_colliding(self.blue_cube))
        )

        # Released: no cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(y_ok and x_order_ok and no_pair_collision)

        # No semantic constraints besides fixed colors; verify colors defensively
        semantic_ok = (
            self.red_cube.get_color() == "red"
            and self.green_cube.get_color() == "green"
            and self.blue_cube.get_color() == "blue"
        )

        # Subgoal staging (best-effort based on observable state)
        # 0-2 (grasp/transport/place) are not directly inferable; mark them upon achieving arrangement.
        # 3: align (observable)
        # 4: release (observable)
        if positions_ok:
            for i in range(4):
                self._metric_stage(i, True)
        if released:
            self._metric_stage(4, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distance to goal poses (per cube) and pose/alignment errors
        goal_r = self._goal_positions.get("red_cube", r)
        goal_g = self._goal_positions.get("green_cube", g)
        goal_b = self._goal_positions.get("blue_cube", b)

        target_distance = {
            "red_to_goal": float(np.linalg.norm(r - goal_r)),
            "green_to_goal": float(np.linalg.norm(g - goal_g)),
            "blue_to_goal": float(np.linalg.norm(b - goal_b)),
            "mean_to_goal": float((np.linalg.norm(r - goal_r) + np.linalg.norm(g - goal_g) + np.linalg.norm(b - goal_b)) / 3.0),
        }

        y_spread = float(max(abs(r[1] - g[1]), abs(g[1] - b[1]), abs(r[1] - b[1])))
        x_margin_rg = float((g[0] - r[0]) - self._MIN_X_SEP)
        x_margin_gb = float((b[0] - g[0]) - self._MIN_X_SEP)

        pose_error = {
            "y_alignment_spread": y_spread,
            "x_separation_margin_rg": x_margin_rg,
            "x_separation_margin_gb": x_margin_gb,
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