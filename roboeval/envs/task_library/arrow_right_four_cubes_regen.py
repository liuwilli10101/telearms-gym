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
class RightArrowCubeArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes on the table into a 2D arrow shape that points to the right
    (the -Y direction). The arrow must have a clear tip on the right and a tail extending
    leftward (+Y).

    Sequential Sub-Goals:
        1. Grasp any one cube to begin forming the arrow.
        2. Move the designated tip cube to the rightmost (-Y) position of the arrow.
        3. Place one wing cube one step left (+Y) of the tip and offset along X.
        4. Place the second wing cube one step left (+Y) of the tip and offset along X
           (opposite side) to complete the arrowhead.
        5. Place the tail cube two steps left (+Y) of the tip and aligned in X to form the shaft.
        6. Fine-tune cube positions to satisfy arrowhead separation along X and overall geometry.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace / geometry constants
    _CUBE_Z = 0.97
    _CENTER_X = 0.55

    # Arrow geometry (meters)
    _STEP_Y = 0.06          # "one cube-step" in the task spec
    _TAIL_OFFSET_Y = 0.12   # "two cube-steps"
    _WING_MIN_SEP_X = 0.05

    # Tolerances
    _ALIGN_TOL_X = 0.02
    _ALIGN_TOL_Y = 0.02
    _OFFSET_TOL = 0.02

    # Subgoals
    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY permitted asset)
        self.cube_tip = ColoredCube(self._mojo)
        self.cube_wing_front = ColoredCube(self._mojo)
        self.cube_wing_back = ColoredCube(self._mojo)
        self.cube_tail = ColoredCube(self._mojo)

        self.cubes = [
            self.cube_tip,
            self.cube_wing_front,
            self.cube_wing_back,
            self.cube_tail,
        ]

        # REQUIRED: set_color() in _initialize_env()
        # (Colors act as stable identifiers for tip/wing/tail roles.)
        self.cube_tip.set_color("red")
        self.cube_wing_front.set_color("blue")
        self.cube_wing_back.set_color("green")
        self.cube_tail.set_color("yellow")

        # Pre-compute a canonical goal configuration (used for metrics and pose error)
        self._goal_poses = self.compute_goal_configuration()

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

    def compute_goal_configuration(self) -> Dict[str, np.ndarray]:
        """
        Deterministic canonical arrow configuration.

        Axis convention reminder:
          +Y = left, -Y = right  (arrow points to -Y)
        """
        x_tip = self._CENTER_X
        y_tip = -0.10

        # Wings are one step left (+Y) of the tip, split along X for arrowhead spread
        y_wing = y_tip + self._STEP_Y
        x_wing_front = x_tip + 0.03
        x_wing_back = x_tip - 0.03

        # Tail is two steps left (+Y) of the tip, aligned in X with the tip
        x_tail = x_tip
        y_tail = y_tip + self._TAIL_OFFSET_Y

        return {
            "tip": np.array([x_tip, y_tip, self._CUBE_Z]),
            "wing_front": np.array([x_wing_front, y_wing, self._CUBE_Z]),
            "wing_back": np.array([x_wing_back, y_wing, self._CUBE_Z]),
            "tail": np.array([x_tail, y_tail, self._CUBE_Z]),
        }

    def sample_initial_state(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """
        Provided for compatibility with downstream randomization pipelines.

        This environment's own _on_reset() must place the cubes in the SUCCESS (goal) state
        deterministically; external evaluators may override with randomized states.
        """
        return self.compute_goal_configuration()

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

        # 2. ⚠️ CRITICAL: Use deterministic RNG (even if not used here)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset()
        self.cube_tip.set_color("red")
        self.cube_wing_front.set_color("blue")
        self.cube_wing_back.set_color("green")
        self.cube_tail.set_color("yellow")

        # Keep canonical goal (used for metrics), but DO NOT place cubes at goal.
        self._goal_poses = self.compute_goal_configuration()

        # 4. Place objects in RANDOM UNSOLVED positions within workspace bounds.
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        xmin, xmax = 0.40, 0.70
        ymin, ymax = -0.25, 0.25
        z = self._CUBE_Z

        def rand_quat_yaw() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)  # x,y,z,w
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        # Sample non-overlapping points and ensure configuration is not already successful.
        # (Success is strict; we still defensively reject any accidental goal-satisfying sample.)
        min_sep = 0.06
        max_tries = 200
        for _ in range(max_tries):
            # Deterministic sampling using the episode RNG (env_utils.get_random_points_on_plane has no rng kwarg)
            pts_xy = []
            min_sep2 = float(min_sep * min_sep)
            for _attempt in range(2000):
                x = float(rng.uniform(xmin, xmax))
                y = float(rng.uniform(ymin, ymax))
                if all(((x - px) ** 2 + (y - py) ** 2) >= min_sep2 for (px, py) in pts_xy):
                    pts_xy.append((x, y))
                    if len(pts_xy) >= 4:
                        break
            if len(pts_xy) < 4:
                continue
            pts = [np.array([p[0], p[1], z], dtype=float) for p in pts_xy]

            self.cube_tip.set_pose(position=pts[0], quat=rand_quat_yaw())
            self.cube_wing_front.set_pose(position=pts[1], quat=rand_quat_yaw())
            self.cube_wing_back.set_pose(position=pts[2], quat=rand_quat_yaw())
            self.cube_tail.set_pose(position=pts[3], quat=rand_quat_yaw())

            # Physics settling
            self._mojo.physics.forward()

            # Ensure unsolved after reset
            cond = self.check_success_conditions()
            positions_ok = (
                cond["c1_tip_right_of_tail"]
                and cond["c2_wf_offset_ok"]
                and cond["c3_wb_offset_ok"]
                and cond["c4_wings_ok"]
                and cond["c5_tail_ok"]
                and cond["c_tip_is_rightmost"]
                and cond["z_ok"]
            )
            if not positions_ok:
                break
        else:
            # Fallback: force a clear failure by placing tip far left in Y
            self.cube_tip.set_pose(position=np.array([0.55, 0.22, z], dtype=float), quat=rand_quat_yaw())
            self.cube_wing_front.set_pose(position=np.array([0.50, -0.10, z], dtype=float), quat=rand_quat_yaw())
            self.cube_wing_back.set_pose(position=np.array([0.60, -0.05, z], dtype=float), quat=rand_quat_yaw())
            self.cube_tail.set_pose(position=np.array([0.55, -0.20, z], dtype=float), quat=rand_quat_yaw())
            self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _pos(self, obj: ColoredCube) -> np.ndarray:
        return np.array(obj.body.get_position())

    def _dist_to_goal(self, obj: ColoredCube, goal_key: str) -> float:
        return float(np.linalg.norm(self._pos(obj) - self._goal_poses[goal_key]))

    def check_success_conditions(self) -> Dict[str, bool]:
        """
        Returns per-condition booleans matching the task's goal-conditions.
        """
        tip = self._pos(self.cube_tip)
        wf = self._pos(self.cube_wing_front)
        wb = self._pos(self.cube_wing_back)
        tail = self._pos(self.cube_tail)

        # Condition 1: tip right_of tail along Y (tip more negative Y), min separation
        c1_tip_right_of_tail = (tail[1] - tip[1]) >= 0.06

        # Condition 2 & 3: wings one step left (+Y) of tip (offset along Y)
        wf_offset = wf[1] - tip[1]
        wb_offset = wb[1] - tip[1]
        c2_wf_offset_ok = abs(wf_offset - self._STEP_Y) <= self._OFFSET_TOL
        c3_wb_offset_ok = abs(wb_offset - self._STEP_Y) <= self._OFFSET_TOL

        # Condition 4: wings separated along X, aligned in Y
        c4_wings_sep_x = abs(wf[0] - wb[0]) >= self._WING_MIN_SEP_X
        c4_wings_align_y = abs(wf[1] - wb[1]) <= self._ALIGN_TOL_Y
        c4_wings_ok = c4_wings_sep_x and c4_wings_align_y

        # Condition 5: tail aligned in X with tip, and two steps left (+Y) of tip
        tail_offset = tail[1] - tip[1]
        c5_tail_align_x = abs(tail[0] - tip[0]) <= self._ALIGN_TOL_X
        c5_tail_offset_ok = abs(tail_offset - self._TAIL_OFFSET_Y) <= 0.03  # slightly looser for robustness
        c5_tail_ok = c5_tail_align_x and c5_tail_offset_ok

        # Extra: enforce tip is the rightmost cube in Y (smallest Y), to avoid role swapping
        ys = [tip[1], wf[1], wb[1], tail[1]]
        c_tip_is_rightmost = tip[1] <= (min(ys[1:]) - 0.02)

        # Extra: ensure cubes are on/near the table plane (not stacked)
        z_ok = all(abs(self._pos(c)[2] - self._CUBE_Z) <= 0.02 for c in self.cubes)

        return {
            "c1_tip_right_of_tail": bool(c1_tip_right_of_tail),
            "c2_wf_offset_ok": bool(c2_wf_offset_ok),
            "c3_wb_offset_ok": bool(c3_wb_offset_ok),
            "c4_wings_ok": bool(c4_wings_ok),
            "c5_tail_ok": bool(c5_tail_ok),
            "c_tip_is_rightmost": bool(c_tip_is_rightmost),
            "z_ok": bool(z_ok),
        }

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_poses"):
            return False

        cond = self.check_success_conditions()

        positions_ok = (
            cond["c1_tip_right_of_tail"]
            and cond["c2_wf_offset_ok"]
            and cond["c3_wb_offset_ok"]
            and cond["c4_wings_ok"]
            and cond["c5_tail_ok"]
            and cond["c_tip_is_rightmost"]
            and cond["z_ok"]
        )

        # No semantic constraints beyond role identity; keep a simple role-color check.
        semantic_ok = (
            self.cube_tip.get_color() == "red"
            and self.cube_wing_front.get_color() == "blue"
            and self.cube_wing_back.get_color() == "green"
            and self.cube_tail.get_color() == "yellow"
        )

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging (best-effort, sequential-friendly)
        any_holding = any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )
        self._metric_stage(0, bool(any_holding))
        self._metric_stage(1, bool(cond["c_tip_is_rightmost"] and cond["c1_tip_right_of_tail"]))
        self._metric_stage(2, bool(cond["c2_wf_offset_ok"]))
        self._metric_stage(3, bool(cond["c3_wb_offset_ok"] and cond["c4_wings_sep_x"] if "c4_wings_sep_x" in cond else cond["c4_wings_ok"]))
        self._metric_stage(4, bool(cond["c5_tail_ok"]))
        self._metric_stage(5, bool(positions_ok))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics (pose error to canonical goal positions)
        target_distance = {
            "tip_to_goal": self._dist_to_goal(self.cube_tip, "tip"),
            "wing_front_to_goal": self._dist_to_goal(self.cube_wing_front, "wing_front"),
            "wing_back_to_goal": self._dist_to_goal(self.cube_wing_back, "wing_back"),
            "tail_to_goal": self._dist_to_goal(self.cube_tail, "tail"),
        }
        pose_error = {
            "tip_pos_err": target_distance["tip_to_goal"],
            "wing_front_pos_err": target_distance["wing_front_to_goal"],
            "wing_back_pos_err": target_distance["wing_back_to_goal"],
            "tail_pos_err": target_distance["tail_to_goal"],
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