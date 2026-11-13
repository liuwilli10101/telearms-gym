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
class WindmillCubeSpiralArrange(RoboEvalEnv, MetricRolloutEval):
    """Arrange four cubes around the patch in a windmill-like spiral with increasing radius.

Success Criteria:
- Around the goal patch, red is right of the patch, blue is behind it, green is left of it, and yellow is in front of it within the configured offset ranges.
- The distances from the patch increase in red, blue, green, yellow order and all cubes are at table height within workspace bounds.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace bounds (task spec)
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    # Z constants (asset registry)
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    # Goal condition tolerances
    _ORTHO_TOL = 0.04  # meters
    _RADIUS_ORDER_TOL = 0.02  # meters (used as an order margin)

    # Spiral arms, as (relation, min_offset, max_offset) from the patch centre.
    # The windows used to overlap (red 0.06-0.10, blue 0.07-0.11, green 0.10-0.14,
    # yellow 0.11-0.15) while success ALSO demanded strictly increasing radii. Red
    # at 0.10 with blue at 0.07 satisfied every window and still failed the order
    # check, so a correctly built windmill could not pass. The windows are now
    # disjoint, which makes the ordering a consequence of them rather than an
    # extra condition that can contradict them.
    _ARM_SPECS = {
        "red": ("right_of", 0.05, 0.10),
        "blue": ("back_of", 0.08, 0.13),
        "green": ("left_of", 0.11, 0.16),
        "yellow": ("front_of", 0.13, 0.18),
    }
    # The windows still overlap (they have to, or the arms do not fit the table),
    # so the radius order is checked with slack instead of strictly: a windmill that
    # looks right passes even if two arms come out nearly the same length.
    _ORDER_SLACK_M = 0.03

    _NUM_SUBGOALS = 12

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Goal patch
        self.goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # Colored cubes (ONLY allowed asset type)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_cube = ColoredCube(self._mojo)
        self.yellow_cube = ColoredCube(self._mojo)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

        self.cubes = [self.red_cube, self.blue_cube, self.green_cube, self.yellow_cube]

        # Targets are defined relative to the goal patch (computed in _on_reset)
        self._target_patch_xy = np.array([0.55, 0.0], dtype=float)
        self._target_xy = {}

        # Track whether each cube has been held at least once (for grasp subgoals)
        self._held_once = {k: False for k in ["red", "blue", "green", "yellow"]}

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

    def _safe_forward(self):
        # Some stacks expose physics.forward(); others only step(). Guard both.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _clip_workspace_xy(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).copy()
        xy[0] = float(np.clip(xy[0], self._X_BOUNDS[0], self._X_BOUNDS[1]))
        xy[1] = float(np.clip(xy[1], self._Y_BOUNDS[0], self._Y_BOUNDS[1]))
        return xy

    def _set_goal_targets_from_patch(self, patch_xy: np.ndarray):
        """
        Define one concrete goal instance that satisfies the required offset bands
        AND increasing-radius spiral order:
          red (closest) -> right (-Y)
          blue          -> back (+X)
          green         -> left (+Y)
          yellow        -> front (-X) farthest
        """
        px, py = float(patch_xy[0]), float(patch_xy[1])

        # Choose offsets within the allowed ranges.
        # Keep orthogonal offsets near 0 to satisfy orthogonal tolerance.
        self._target_xy = {
            "red": np.array([px, py - 0.075], dtype=float),     # right (-Y), window 0.05-0.10
            "blue": np.array([px + 0.105, py], dtype=float),    # back (+X),  window 0.08-0.13
            "green": np.array([px, py + 0.135], dtype=float),   # left (+Y),  window 0.11-0.16
            "yellow": np.array([px - 0.155, py], dtype=float),  # front (-X), window 0.13-0.18
        }

        # Enforce workspace bounds defensively.
        for k, xy in list(self._target_xy.items()):
            self._target_xy[k] = self._clip_workspace_xy(xy)

    def _place_pose(self, obj, pos_xyz: np.ndarray):
        pos = np.asarray(pos_xyz, dtype=float)
        # GoalPatch uses body/set_position API; cubes use set_pose(position=..., quat=...).
        if obj.__class__.__name__ == "GoalPatch":
            obj.body.set_position(pos)
        elif hasattr(obj, "set_pose"):
            obj.set_pose(position=pos, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        elif hasattr(obj, "set_position") and callable(getattr(obj, "set_position")):
            obj.set_position(pos)
        else:
            obj.body.set_position(pos)

        self._safe_forward()
        for _ in range(50):
            self._mojo.step()

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

        # Reset held-once tracking
        self._held_once = {k: False for k in ["red", "blue", "green", "yellow"]}

        # --- Randomize initial UNSOLVED poses (must make _success() False) ---
        # Task-required workspace bounds for reset placement
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # The patch must leave room for every arm, or the goal poses get clipped to
        # the workspace edge and the task becomes unsatisfiable. Sample the patch
        # from the sub-box where all four arms fit at their maximum offset.
        max_back = self._ARM_SPECS["blue"][2]
        max_front = self._ARM_SPECS["yellow"][2]
        max_left = self._ARM_SPECS["green"][2]
        max_right = self._ARM_SPECS["red"][2]
        patch_x_lo = max(x_min, self._X_BOUNDS[0] + max_front)
        patch_x_hi = min(x_max, self._X_BOUNDS[1] - max_back)
        patch_y_lo = max(y_min, self._Y_BOUNDS[0] + max_right)
        patch_y_hi = min(y_max, self._Y_BOUNDS[1] - max_left)

        # Randomize patch location within the feasible sub-box
        patch_xy = np.array(
            [rng.uniform(patch_x_lo, patch_x_hi), rng.uniform(patch_y_lo, patch_y_hi)],
            dtype=float,
        )
        self._target_patch_xy = patch_xy
        self._set_goal_targets_from_patch(patch_xy)

        # Place patch first
        self._place_pose(self.goal_patch, np.array([patch_xy[0], patch_xy[1], self._PATCH_Z], dtype=float))
        self._mojo.physics.forward()

        # Place cubes in a stable 2x2 pattern near the patch but separated to prevent collisions
        # that can push cubes out of workspace during settling. Still UNSOLVED (wrong relations/order).
        pad = 0.03
        sx_min, sx_max = x_min + pad, x_max - pad
        sy_min, sy_max = y_min + pad, y_max - pad

        spawn_center = np.array(
            [float(np.clip(patch_xy[0], sx_min, sx_max)), float(np.clip(patch_xy[1], sy_min, sy_max))],
            dtype=float,
        )

        offsets = [
            np.array([-0.06, -0.06], dtype=float),
            np.array([-0.06,  0.06], dtype=float),
            np.array([ 0.06, -0.06], dtype=float),
            np.array([ 0.06,  0.06], dtype=float),
        ]

        cube_xys = []
        for off in offsets:
            cand = spawn_center + off
            cand[0] = float(np.clip(cand[0], sx_min, sx_max))
            cand[1] = float(np.clip(cand[1], sy_min, sy_max))
            cube_xys.append(cand)

        # Sequential placement for stability
        self._place_pose(self.red_cube, np.array([cube_xys[0][0], cube_xys[0][1], self._CUBE_Z], dtype=float))
        self._mojo.physics.forward()

        self._place_pose(self.blue_cube, np.array([cube_xys[1][0], cube_xys[1][1], self._CUBE_Z], dtype=float))
        self._mojo.physics.forward()

        self._place_pose(self.green_cube, np.array([cube_xys[2][0], cube_xys[2][1], self._CUBE_Z], dtype=float))
        self._mojo.physics.forward()

        self._place_pose(self.yellow_cube, np.array([cube_xys[3][0], cube_xys[3][1], self._CUBE_Z], dtype=float))
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()

        # Update "held once" for grasp subgoals (works for single or dual grippers).
        hold_states = {}
        for name, cube in [("red", self.red_cube), ("blue", self.blue_cube), ("green", self.green_cube), ("yellow", self.yellow_cube)]:
            hold_states[name] = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
            if hold_states[name]:
                self._held_once[name] = True

        # Stage progress (best-effort; final success still checked in _success()).
        # Grasp stages:
        if self._held_once["red"]:
            self._metric_stage(0, True)
        if self._held_once["blue"]:
            self._metric_stage(3, True)
        if self._held_once["green"]:
            self._metric_stage(6, True)
        if self._held_once["yellow"]:
            self._metric_stage(9, True)

        self._metric_step()

    def _xy_dist(self, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
        d = np.asarray(a_xy, dtype=float) - np.asarray(b_xy, dtype=float)
        return float(np.linalg.norm(d[:2]))

    def _cube_xy(self, cube) -> np.ndarray:
        p = cube.body.get_position()
        return np.array([float(p[0]), float(p[1])], dtype=float)

    @staticmethod
    def _rel_offset(subj_xy: np.ndarray, targ_xy: np.ndarray, relation: str) -> float:
        """Signed offset along the arm direction (positive = correct side)."""
        if relation == "right_of":
            return float(targ_xy[1] - subj_xy[1])
        if relation == "left_of":
            return float(subj_xy[1] - targ_xy[1])
        if relation == "back_of":
            return float(subj_xy[0] - targ_xy[0])
        if relation == "front_of":
            return float(targ_xy[0] - subj_xy[0])
        return float("nan")

    def _rel_ok(self, subj_xy: np.ndarray, targ_xy: np.ndarray, relation: str,
                min_off: float, max_off: float, ortho_tol: float) -> bool:
        dx = float(subj_xy[0] - targ_xy[0])
        dy = float(subj_xy[1] - targ_xy[1])

        if relation == "right_of":
            # right = -Y => offset = targ_y - subj_y
            off = float(targ_xy[1] - subj_xy[1])
            return (min_off <= off <= max_off) and (abs(dx) <= ortho_tol)
        if relation == "left_of":
            # left = +Y => offset = subj_y - targ_y
            off = float(subj_xy[1] - targ_xy[1])
            return (min_off <= off <= max_off) and (abs(dx) <= ortho_tol)
        if relation == "back_of":
            # back = +X => offset = subj_x - targ_x
            off = float(subj_xy[0] - targ_xy[0])
            return (min_off <= off <= max_off) and (abs(dy) <= ortho_tol)
        if relation == "front_of":
            # front = -X => offset = targ_x - subj_x
            off = float(targ_xy[0] - subj_xy[0])
            return (min_off <= off <= max_off) and (abs(dy) <= ortho_tol)

        return False

    def _in_target_xy(self, cube, key: str, tol_xy: float = 0.03) -> bool:
        cxy = self._cube_xy(cube)
        txy = self._target_xy[key]
        return self._xy_dist(cxy, txy) <= tol_xy

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False
        if not hasattr(self, "_target_xy") or not self._target_xy:
            return False

        patch_xy = np.array(self.goal_patch.body.get_position()[:2], dtype=float)

        red_xy = self._cube_xy(self.red_cube)
        blue_xy = self._cube_xy(self.blue_cube)
        green_xy = self._cube_xy(self.green_cube)
        yellow_xy = self._cube_xy(self.yellow_cube)

        # Relative placement checks (task goal-conditions)
        arm_xy = {"red": red_xy, "blue": blue_xy, "green": green_xy, "yellow": yellow_xy}
        arm_offsets = {
            name: self._rel_offset(arm_xy[name], patch_xy, spec[0])
            for name, spec in self._ARM_SPECS.items()
        }
        red_ok = self._rel_ok(red_xy, patch_xy, *self._ARM_SPECS["red"], self._ORTHO_TOL)
        blue_ok = self._rel_ok(blue_xy, patch_xy, *self._ARM_SPECS["blue"], self._ORTHO_TOL)
        green_ok = self._rel_ok(green_xy, patch_xy, *self._ARM_SPECS["green"], self._ORTHO_TOL)
        yellow_ok = self._rel_ok(yellow_xy, patch_xy, *self._ARM_SPECS["yellow"], self._ORTHO_TOL)

        # Spiral radius order: distances from patch strictly increasing in order
        d_red = self._xy_dist(red_xy, patch_xy)
        d_blue = self._xy_dist(blue_xy, patch_xy)
        d_green = self._xy_dist(green_xy, patch_xy)
        d_yellow = self._xy_dist(yellow_xy, patch_xy)

        # Order on the along-axis offsets, which are the quantities the arm windows
        # actually constrain. Euclidean radius mixes in the orthogonal slop and can
        # invert the order for a placement that satisfies every window.
        slack = self._ORDER_SLACK_M
        radius_order_ok = (
            arm_offsets["red"] <= arm_offsets["blue"] + slack and
            arm_offsets["blue"] <= arm_offsets["green"] + slack and
            arm_offsets["green"] <= arm_offsets["yellow"] + slack
        )

        positions_ok = bool(red_ok and blue_ok and green_ok and yellow_ok and radius_order_ok)

        # Optional physical sanity: cubes should be at approximately table rest height
        cube_z_ok = True
        for c in self.cubes:
            if not hasattr(c, "body"):
                cube_z_ok = False
                break
            cz = c.body.get_position()
            if cz is None or len(cz) < 3:
                cube_z_ok = False
                break
            if abs(float(cz[2]) - float(self._CUBE_Z)) > 0.03:
                cube_z_ok = False
                break
        positions_ok = bool(positions_ok and cube_z_ok)

        # No semantic content beyond color (colors are set deterministically)
        semantic_ok = True

        # Released gripper check
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging (best-effort, based on observed grasp + placement)
        # Red stages
        if self._held_once.get("red", False):
            self._metric_stage(0, True)
        if self._in_target_xy(self.red_cube, "red", tol_xy=0.035):
            self._metric_stage(1, True)
        if self._in_target_xy(self.red_cube, "red", tol_xy=0.025) and not any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(2, True)

        # Blue stages
        if self._held_once.get("blue", False):
            self._metric_stage(3, True)
        if self._in_target_xy(self.blue_cube, "blue", tol_xy=0.035):
            self._metric_stage(4, True)
        if self._in_target_xy(self.blue_cube, "blue", tol_xy=0.025) and not any(
            self.robot.is_gripper_holding_object(self.blue_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(5, True)

        # Green stages
        if self._held_once.get("green", False):
            self._metric_stage(6, True)
        if self._in_target_xy(self.green_cube, "green", tol_xy=0.035):
            self._metric_stage(7, True)
        if self._in_target_xy(self.green_cube, "green", tol_xy=0.025) and not any(
            self.robot.is_gripper_holding_object(self.green_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(8, True)

        # Yellow stages
        if self._held_once.get("yellow", False):
            self._metric_stage(9, True)
        if self._in_target_xy(self.yellow_cube, "yellow", tol_xy=0.035):
            self._metric_stage(10, True)
        if self._in_target_xy(self.yellow_cube, "yellow", tol_xy=0.025) and not any(
            self.robot.is_gripper_holding_object(self.yellow_cube, g) for g in self.robot.grippers
        ):
            self._metric_stage(11, True)

        # If fully successful, mark all stages as achieved.
        if bool(positions_ok and semantic_ok and released):
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "red_to_target_xy": self._xy_dist(red_xy, self._target_xy["red"]),
            "blue_to_target_xy": self._xy_dist(blue_xy, self._target_xy["blue"]),
            "green_to_target_xy": self._xy_dist(green_xy, self._target_xy["green"]),
            "yellow_to_target_xy": self._xy_dist(yellow_xy, self._target_xy["yellow"]),
            "red_to_patch_xy": d_red,
            "blue_to_patch_xy": d_blue,
            "green_to_patch_xy": d_green,
            "yellow_to_patch_xy": d_yellow,
        }
        pose_error = {
            "patch_xy_error_from_goal": self._xy_dist(patch_xy, self._target_patch_xy),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})