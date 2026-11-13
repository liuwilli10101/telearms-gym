# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

from __future__ import annotations
from abc import ABC
import numpy as np
from itertools import permutations
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


class GreenPatchCornerGuarding(RoboEvalEnv, MetricRolloutEval):
    """Place one cube near each visible corner of the green patch, in any color order.

Success Criteria:
Each cube center may be within 5 cm in X and Y of its corner. Occupy all four distinct corners and release the cubes."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _NUM_SUBGOALS = 12
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    # Corner offset goals (relative to patch center), per task spec
    # northwest: (-X,+Y), northeast: (-X,-Y), southwest: (+X,+Y), southeast: (+X,-Y)
    # GoalPatch takes HALF-extents, so size=(0.1, 0.1) draws a 20 cm square whose
    # visible corners sit at +/-0.10 from the centre. The goals used to be hardcoded
    # at +/-0.06 -- inside the patch, not on its corners -- so a cube placed on the
    # corner the operator can actually see landed 4 cm away and failed the +/-0.03
    # window. Corners are now derived from the patch, and the tolerance covers a
    # cube resting anywhere around its corner.
    _PATCH_HALF_EXTENT_M = 0.10
    _CORNER_TOL_XYZ = np.array([0.05, 0.05, 0.05])
    _CORNER_SPECS = {
        "cube_1": {"corner_name": "northwest", "sign_xy": np.array([-1.0, 1.0])},
        "cube_2": {"corner_name": "northeast", "sign_xy": np.array([-1.0, -1.0])},
        "cube_3": {"corner_name": "southwest", "sign_xy": np.array([1.0, 1.0])},
        "cube_4": {"corner_name": "southeast", "sign_xy": np.array([1.0, -1.0])},
    }

    _INSTRUCTION_VARIANTS = [
        "Guard the four corners of the green target patch by placing a cube at each corner (NW, NE, SW, SE).",
        "Put four cubes around the green square so each cube sits near a different corner: (-X,+Y), (-X,-Y), (+X,+Y), (+X,-Y).",
        "Place the cubes at the patch corners: front-left, front-right, back-left, and back-right relative to the robot.",
    ]

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Required assets ONLY: GoalPatch + ColoredCube
        self.goal_patch_green = GoalPatch(
            self._mojo,
            color="green",
            size=(self._PATCH_HALF_EXTENT_M, self._PATCH_HALF_EXTENT_M),
            name="target",
        )

        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cube_4 = ColoredCube(self._mojo)

        # Keep a canonical list for iteration/metrics
        self.cubes = [self.cube_1, self.cube_2, self.cube_3, self.cube_4]
        self._cube_names = ["cube_1", "cube_2", "cube_3", "cube_4"]

        # MUST call set_color() in BOTH _initialize_env AND _on_reset
        self._cube_colors = ["red", "blue", "yellow", "purple"]
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

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

    def get_task_instruction(self) -> str:
        # Deterministic selection for reproducibility
        idx = int(self.seed) % len(self._INSTRUCTION_VARIANTS) if hasattr(self, "seed") else 0
        return self._INSTRUCTION_VARIANTS[idx]

    def _corner_goal_pos(self, patch_pos: np.ndarray, cube_name: str) -> np.ndarray:
        spec = self._CORNER_SPECS[cube_name]
        sign = spec["sign_xy"]
        half = self._PATCH_HALF_EXTENT_M
        pos = patch_pos.copy()
        pos[0] = patch_pos[0] + sign[0] * half
        pos[1] = patch_pos[1] + sign[1] * half
        pos[2] = self._CUBE_Z
        return pos

    def _sample_object_poses(self) -> Dict[str, np.ndarray]:
        """
        Provide randomized, UNSOLVED poses for external randomization pipelines.

        Notes:
          - Keeps cubes away from the patch to avoid immediate goal satisfaction.
          - Uses deterministic RNG (self.seed).
          - Stays within workspace bounds.
        """
        rng = np.random.default_rng(self.seed)

        # Sample patch position (keep comfortably inside workspace)
        patch_x = float(rng.uniform(0.48, 0.62))
        patch_y = float(rng.uniform(-0.10, 0.10))
        patch_pos = np.array([patch_x, patch_y, self._PATCH_Z], dtype=float)

        # Sample cube positions away from patch and away from corner goal regions
        cube_positions = []
        max_tries = 500

        def _in_workspace(xy: np.ndarray) -> bool:
            return (0.35 <= xy[0] <= 0.75) and (-0.30 <= xy[1] <= 0.30)

        # Corner target positions (XY) to avoid spawning in/near success
        corner_xys = [self._corner_goal_pos(patch_pos, n)[:2] for n in self._cube_names]

        for _ in range(4):
            placed = False
            for _t in range(max_tries):
                xy = np.array(
                    [
                        rng.uniform(0.40, 0.72),
                        rng.uniform(-0.26, 0.26),
                    ],
                    dtype=float,
                )
                if not _in_workspace(xy):
                    continue

                # Keep cubes away from patch center (avoid patch overlap / trivial near-corner)
                if np.linalg.norm(xy - patch_pos[:2]) < 0.18:
                    continue

                # Keep cubes away from any corner goal region (avoid immediate success)
                if any(np.linalg.norm(xy - cxy) < 0.09 for cxy in corner_xys):
                    continue

                # Keep cubes apart (>= 5cm)
                if any(np.linalg.norm(xy - p[:2]) < 0.06 for p in cube_positions):
                    continue

                cube_positions.append(np.array([xy[0], xy[1], self._CUBE_Z], dtype=float))
                placed = True
                break

            if not placed:
                # Fallback: deterministic spread (still unsolved, away from patch)
                base_x = 0.70
                base_y = 0.24
                k = len(cube_positions)
                cube_positions.append(np.array([base_x, base_y - 0.08 * k, self._CUBE_Z], dtype=float))

        return {
            "goal_patch_green": patch_pos,
            "cube_1": cube_positions[0],
            "cube_2": cube_positions[1],
            "cube_3": cube_positions[2],
            "cube_4": cube_positions[3],
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

        # 2. Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # 3. Randomize object poses (UNSOLVED) within required workspace bounds
        # Workspace bounds required by task:
        #   X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97 (cubes)
        def _rand_yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return Quaternion(axis=[0, 0, 1], angle=yaw).elements

        # Patch pose (random, within bounds; keep corners inside bounds by margin)
        patch_margin = 0.08  # >= corner offsets (0.06) plus some clearance
        patch_x = float(rng.uniform(0.40 + patch_margin, 0.70 - patch_margin))
        patch_y = float(rng.uniform(-0.25 + patch_margin, 0.25 - patch_margin))
        patch_pos = np.array([patch_x, patch_y, self._PATCH_Z], dtype=float)
        self.goal_patch_green.body.set_position(patch_pos)

        # MUST call set_color() in BOTH _initialize_env AND _on_reset
        for c, col in zip(self.cubes, self._cube_colors):
            c.set_color(col)

        # Corner target positions (XY) to avoid spawning in/near success
        corner_xys = [self._corner_goal_pos(patch_pos, n)[:2] for n in self._cube_names]

        # Sample cube positions away from patch center and away from corner target regions
        cube_positions = []
        max_tries = 1000

        def _in_required_workspace(xy: np.ndarray) -> bool:
            return (0.40 <= float(xy[0]) <= 0.70) and (-0.25 <= float(xy[1]) <= 0.25)

        for i, (name, cube) in enumerate(zip(self._cube_names, self.cubes)):
            placed = False
            for _t in range(max_tries):
                xy = np.array(
                    [
                        rng.uniform(0.40, 0.70),
                        rng.uniform(-0.25, 0.25),
                    ],
                    dtype=float,
                )
                if not _in_required_workspace(xy):
                    continue

                # Keep cubes away from patch center
                if float(np.linalg.norm(xy - patch_pos[:2])) < 0.18:
                    continue

                # Keep cubes away from any corner goal region (avoid immediate success)
                if any(float(np.linalg.norm(xy - cxy)) < 0.10 for cxy in corner_xys):
                    continue

                # Keep cubes apart
                if any(float(np.linalg.norm(xy - p[:2])) < 0.06 for p in cube_positions):
                    continue

                pos = np.array([xy[0], xy[1], self._CUBE_Z], dtype=float)
                cube_positions.append(pos)
                cube.set_pose(position=pos, quat=_rand_yaw_quat())
                self._mojo.physics.forward()
                placed = True
                break

            if not placed:
                # Deterministic fallback still within bounds and still unsolved (far from patch/corners)
                fallback_xs = [0.68, 0.66, 0.64, 0.62]
                fallback_ys = [0.22, 0.10, -0.02, -0.14]
                xy = np.array([fallback_xs[i], fallback_ys[i]], dtype=float)
                pos = np.array([xy[0], xy[1], self._CUBE_Z], dtype=float)
                cube_positions.append(pos)
                cube.set_pose(position=pos, quat=np.array([1, 0, 0, 0], dtype=float))
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
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch_green"):
            return False

        patch_pos = self.goal_patch_green.body.get_position()

        goals = [self._corner_goal_pos(patch_pos, name) for name in self._cube_names]
        positions = [np.asarray(c.body.get_position()) for c in self.cubes]
        # Select a bijection, so one cube cannot satisfy multiple corners.
        assignment = min(permutations(range(4)), key=lambda order: max(
            np.max(np.abs(positions[i] - goals[j]) / self._CORNER_TOL_XYZ)
            for i, j in enumerate(order)))
        placed_flags = {}
        target_distance = {}
        pose_error = {}
        for i, j in enumerate(assignment):
            name = self._cube_names[i]
            err = positions[i] - goals[j]
            placed_flags[name] = bool(np.all(np.abs(err) <= self._CORNER_TOL_XYZ + 1e-9))
            target_distance[name] = float(np.linalg.norm(err[:2]))
            pose_error[name] = err
        positions_ok = all(placed_flags.values())

        # Arrangement condition: cubes must occupy distinct corners (min pairwise distance in XY)
        distinct_ok = True
        min_pairwise = 0.05
        for i in range(len(self.cubes)):
            pi = self.cubes[i].body.get_position()[:2]
            for j in range(i + 1, len(self.cubes)):
                pj = self.cubes[j].body.get_position()[:2]
                if float(np.linalg.norm(pi - pj)) < min_pairwise:
                    distinct_ok = False
                    break
            if not distinct_ok:
                break

        # Released gripper condition
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # No additional semantic constraints (colors are not part of goal), but keep hook for extensibility
        semantic_ok = True

        # Update subgoal stages: mark the 3 stages per cube as complete if that cube is correctly placed
        # (We do not attempt to infer grasp/transport separately.)
        for ci, name in enumerate(self._cube_names):
            if placed_flags.get(name, False):
                base_stage = ci * 3
                self._metric_stage(base_stage + 0, True)
                self._metric_stage(base_stage + 1, True)
                self._metric_stage(base_stage + 2, True)

        self._success_check = bool(positions_ok and distinct_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error={k: float(np.linalg.norm(v)) for k, v in pose_error.items()},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})