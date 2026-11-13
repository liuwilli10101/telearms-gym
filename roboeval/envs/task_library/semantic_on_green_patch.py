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

class SemanticCubeATargetPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place the semantic cube onto the green goal patch and orient it so the face with
    the letter 'A' is pointing upward (+Z). The task is successful when the cube is
    on the target patch (in contact, within XY tolerance) and the 'A' face is on top.

    Sequential Sub-Goals:
        1. Securely grasp the semantic cube.
        2. Lift the cube off the table.
        3. Move the cube above the green goal patch area.
        4. Rotate the cube so the 'A' face is oriented upward.
        5. Lower the cube onto the green goal patch.
        6. Release the cube while keeping it on the patch with the 'A' face up.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # We require the 'A' face to be on the TOP (+Z).
    _FACE_DIR_WORLD = np.array([0.0, 0.0, 1.0])

    # Goal tolerances
    _XY_TOL_M = 0.03
    _A_UP_TOL_DEG = 12.0

    # Z placement constants (MANDATED)
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    # Subgoal count
    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Goal patch (green)
        self.goal_patch_green = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # Semantic cube: must use ensure_cubes() workflow
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]
        xml_path = info["xml"] if "xml" in info else info["xml_path"]

        self.semantic_cube_1 = SemanticCube(self._mojo, xml_path=str(xml_path), labels=info["labels"])
        self.cubes = [self.semantic_cube_1]
        self._target_labels = ["A"]

        # Precompute goal orientation quaternion for "A up"
        face_idx = self.semantic_cube_1._labels.index("A")
        self._goal_quat_A_up = self.semantic_cube_1.get_orientation_for_face_and_glyph(
            face_idx=face_idx,
            face_dir_world=self._FACE_DIR_WORLD,
        )

        # Initialize metrics (FULL signature required)
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
        # Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG (pipeline may add perturbations later)
        rng = np.random.default_rng(self.seed)

        # ---------------------------------------------------------------------
        # RANDOM UNSOLVED placement (within workspace bounds)
        # ---------------------------------------------------------------------
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Randomize patch position
        patch_pos = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), self._PATCH_REST_Z],
            dtype=float,
        )
        self.goal_patch_green.body.set_position(patch_pos)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # Randomize cube position far enough from patch to guarantee failure of XY tolerance
        min_xy_sep = max(0.12, self._XY_TOL_M * 3.0)
        for _ in range(200):
            cube_xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
            if float(np.linalg.norm(cube_xy - patch_pos[:2])) >= min_xy_sep:
                break
        cube_pos = np.array([cube_xy[0], cube_xy[1], self._CUBE_REST_Z], dtype=float)

        # Randomize cube orientation (not required for guaranteed failure, but unsolved/random)
        q_rand = Quaternion.random()
        self.semantic_cube_1.set_pose(position=cube_pos, quat=q_rand.elements)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks (_success can be called during reset)
        if not hasattr(self, "semantic_cube_1") or not hasattr(self, "goal_patch_green") or not hasattr(self, "table"):
            return False

        cube = self.semantic_cube_1
        patch = self.goal_patch_green

        cube_pos = cube.body.get_position()
        patch_pos = patch.body.get_position()

        # --- Position condition: on patch (XY tolerance + contact) ---
        xy_dist = float(np.linalg.norm((cube_pos - patch_pos)[:2]))
        within_xy = xy_dist <= self._XY_TOL_M

        # GoalPatch collision is unreliable from patch side; use cube -> patch.collider
        in_contact = False
        if hasattr(patch, "collider"):
            in_contact = bool(cube.is_colliding(patch.collider))
        else:
            # Fallback (should not be needed, but keeps evaluation robust)
            in_contact = bool(cube.is_colliding(patch))

        positions_ok = bool(within_xy and in_contact)

        # --- Orientation/semantic condition: 'A' face readable on TOP (+Z) ---
        semantic_ok = bool(cube.is_face_readable("A", face_dir=self._FACE_DIR_WORLD))

        # --- Released check ---
        released = not any(
            self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers
        )

        # ------------------------------------------------------------------
        # Subgoal staging (best-effort, single-arm compatible)
        # ------------------------------------------------------------------
        grasped = not released
        lifted = (not cube.is_colliding(self.table.table_top)) or (cube_pos[2] > (self._CUBE_REST_Z + 0.01))
        transported = xy_dist <= 0.08  # coarse "near target" threshold
        aligned = semantic_ok
        placed = positions_ok
        done_release = placed and aligned and released

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(aligned))
        self._metric_stage(4, bool(placed))
        self._metric_stage(5, bool(done_release))

        # ------------------------------------------------------------------
        # Final success condition
        # ------------------------------------------------------------------
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Pose error (orientation): compare to stored goal quaternion for "A up"
        q_cur = Quaternion(cube.body.get_quaternion())
        q_goal = Quaternion(self._goal_quat_A_up.elements)
        dot = abs(float(np.dot(q_cur.elements, q_goal.elements)))
        dot = float(np.clip(dot, -1.0, 1.0))
        ang_err_deg = float(np.degrees(2.0 * np.arccos(dot)))

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy_m": xy_dist,
            },
            pose_error={
                "cube_to_patch_xy_m": xy_dist,
                "A_up_angle_error_deg": ang_err_deg,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})