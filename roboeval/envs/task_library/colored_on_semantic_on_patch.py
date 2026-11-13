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

class SemanticCubeGreenTargetRedStack(RoboEvalEnv, MetricRolloutEval):
    """
    Place the semantic cube onto the green target patch, then stack the red colored cube
    on top of the semantic cube to form a two-cube tower on the target.

    The semantic cube must be oriented so its 'A' face faces the robot;

    Sequential Sub-Goals:
        1. Grasp the semantic cube with either gripper.
        2. Move the semantic cube over the green goal patch.
        3. Place the semantic cube onto the green goal patch with its 'A' face facing the robot.
        4. Release the semantic cube after it is placed on the patch.
        5. Grasp the red colored cube with either gripper.
        6. Align the red cube above the semantic cube for stable stacking.
        7. Place the red colored cube on top of the semantic cube.
        8. Release the red cube after it is stacked on the semantic cube.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Orient the semantic label toward the robot / the fixed "external" camera so it is
    # readable. That camera sits on the -X side (pos x=0.37) looking toward +X, so the
    # camera/robot-facing horizontal direction is -X.
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Geometry constants
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    _CUBE_SIZE = 0.04
    _STACK_DZ = 0.04

    # Tolerances for stacking the red cube on the semantic cube. Loosened from the
    # original 0.014m so that "roughly aligned on top" counts as a successful stack
    # instead of demanding near-perfect centering during manual teleop.
    _XY_TOL = 0.025
    _Z_TOL = 0.02

    _NUM_SUBGOALS = 8

    def _initialize_env(self):
        # Props from preset
        self.table = self._preset.get_props(Table)[0]

        # ── Assets (ONLY: SemanticCube, ColoredCube, GoalPatch) ──────────────────
        # SemanticCube: REQUIRED ensure_cubes() workflow (use the recommended pattern).
        cubes_info = ensure_cubes("ABCDEFGHIJ", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self._target_label = "A"
        info = cubes_info[self._target_label]

        xml_path = info.get("xml", info.get("xml_path"))
        self.semantic_cube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=info["labels"])

        # ColoredCube: REQUIRED instantiation + set_color in _initialize_env AND _on_reset
        self.red_cube = ColoredCube(self._mojo)
        self.red_cube.set_color("red")

        # GoalPatch: REQUIRED (green target)
        self.goal_patch_green = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # Convenience list for metrics / released checks
        self.cubes = [self.semantic_cube, self.red_cube]

        # Metrics (FULL signature required)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _physics_forward(self):
        # Advisory: call physics.forward() after placement when available
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _settle(self, n: int = 50):
        for _ in range(n):
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

        # 2. Deterministic RNG
        _rng = np.random.default_rng(self.seed)

        # Workspace bounds
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25

        def _sample_xy():
            return np.array([_rng.uniform(x_low, x_high), _rng.uniform(y_low, y_high)], dtype=float)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() == False)
        # Randomize patch in workspace
        patch_xy = _sample_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_REST_Z], dtype=float)
        self.goal_patch_green.body.set_position(patch_pos)
        self._physics_forward()
        self._settle(10)

        # Randomize semantic cube away from patch
        min_patch_dist = 0.14  # > half patch size + tol to avoid being "on patch"
        for _ in range(200):
            sem_xy = _sample_xy()
            if np.linalg.norm(sem_xy - patch_xy) >= min_patch_dist:
                break
        sem_pos = np.array([sem_xy[0], sem_xy[1], self._CUBE_REST_Z], dtype=float)

        # Force semantic label to be NOT readable from _FACE_DIR by orienting a different face
        face_idx_target = self.semantic_cube._labels.index(self._target_label)
        other_face_idx = (face_idx_target + 1) % len(self.semantic_cube._labels)
        q = self.semantic_cube.get_orientation_for_face_and_glyph(
            face_idx=other_face_idx, face_dir_world=self._FACE_DIR
        )
        self.semantic_cube.set_pose(position=sem_pos, quat=q.elements)
        self._physics_forward()
        self._settle(25)

        # Randomize red cube away from semantic cube (and not stacked)
        self.red_cube.set_color("red")
        min_sem_dist = 0.12
        for _ in range(200):
            red_xy = _sample_xy()
            if np.linalg.norm(red_xy - sem_xy) >= min_sem_dist:
                break
        red_pos = np.array([red_xy[0], red_xy[1], self._CUBE_REST_Z], dtype=float)
        yaw = _rng.uniform(-np.pi, np.pi)
        red_quat = Quaternion(axis=[0, 0, 1], angle=yaw).elements  # [w,x,y,z]
        self.red_cube.set_pose(position=red_pos, quat=red_quat)
        self._physics_forward()
        self._settle(50)

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_block_on_patch(self, block, patch) -> bool:
        # GoalPatch collision is most reliable from the BLOCK side using patch.collider
        if not hasattr(patch, "collider"):
            return False
        return bool(block.is_colliding(patch.collider))

    def _success(self) -> bool:
        # ⚠️ RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        sem_p = self.semantic_cube.body.get_position()
        red_p = self.red_cube.body.get_position()

        # ── Goal condition 1: semantic cube on green goal patch ──────────────────
        sem_on_patch = self._is_block_on_patch(self.semantic_cube, self.goal_patch_green)

        # Extra geometric sanity (helps when collision is flaky).
        # Patch half-extent is 0.10m and the cube half-extent is 0.5*_CUBE_SIZE, so any
        # center within (patch_half - cube_half) keeps the cube fully on the patch. We only
        # require it to be on the patch (no overhang), not pixel-perfect on the center.
        patch_p = self.goal_patch_green.body.get_position()
        _patch_half_extent = 0.10
        sem_xy_ok = np.linalg.norm((sem_p - patch_p)[:2]) <= (_patch_half_extent - 0.5 * self._CUBE_SIZE)

        goal1_ok = bool(sem_on_patch and sem_xy_ok)

        # ── Required SEMANTIC verification for SemanticCube ─────────────────────
        semantic_ok = bool(self.semantic_cube.is_face_readable(self._target_label, face_dir=self._FACE_DIR))

        # ── Goal condition 2: red cube on semantic cube (stack) ─────────────────
        xy_close = np.linalg.norm((red_p - sem_p)[:2]) <= self._XY_TOL
        z_close = abs((red_p[2] - sem_p[2]) - self._STACK_DZ) <= self._Z_TOL

        stacked_contact = bool(self.red_cube.is_colliding(self.semantic_cube))
        red_not_on_table = not bool(self.red_cube.is_colliding(self.table.table_top))

        # Also verify color is actually red
        red_color_ok = (self.red_cube.get_color() == "red")

        goal2_ok = bool(xy_close and z_close and stacked_contact and red_not_on_table and red_color_ok)

        # ── Released gripper check ─────────────────────────────────────────────
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(goal1_ok and goal2_ok)

        # Subgoal staging (best-effort based on observable end-state)
        if goal1_ok and semantic_ok:
            for i in range(4):  # semantic cube placed & released stages
                self._metric_stage(i, True)
        if positions_ok and semantic_ok:
            for i in range(4, self._NUM_SUBGOALS):  # red cube stacked & released stages
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "semantic_to_patch_xy": float(np.linalg.norm((sem_p - patch_p)[:2])),
                "red_to_semantic_xy": float(np.linalg.norm((red_p - sem_p)[:2])),
                "stack_dz_error": float(abs((red_p[2] - sem_p[2]) - self._STACK_DZ)),
            },
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})