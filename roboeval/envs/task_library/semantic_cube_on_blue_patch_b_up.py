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


class SemanticCubeBlueBPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Pick up the semantic cube and place it onto the blue target patch.
    Finish with the cube oriented so the face showing the letter 'B' is pointing upward.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the semantic cube (right gripper may assist).
        2. Lift the semantic cube clear of the table surface.
        3. Move the semantic cube above the blue goal patch.
        4. Rotate/reorient the cube so the 'B' face is on top before placement.
        5. Lower the cube onto the blue goal patch while maintaining the desired orientation.
        6. Release the cube after it is stably placed on the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal requires B pointing upward (+Z). Use SAME vector for orientation+verification.
    _FACE_DIR_WORLD = np.array([0.0, 0.0, 1.0])

    # Deterministic initial orientation: show 'B' on back face (-X), i.e., not face-up.
    _INIT_FACE_DIR_WORLD = np.array([1.0, 0.0, 0.0])

    # Z-heights (MANDATORY)
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    # Workspace / spacing constraints
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _MIN_CUBE_PATCH_SPACING_XY = 0.139  # center-to-center XY at reset
    _TRANSPORT_XY_THRESH = 0.09         # "above patch" heuristic for subgoal staging
    _LIFT_Z_THRESH = 1.00               # "lifted" heuristic for subgoal staging

    def _initialize_env(self):
        # Props (table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Goal patch (blue) at a fixed reachable location
        self.blue_goal_patch = GoalPatch(self._mojo, color="blue", size=(0.1, 0.1), name="target")

        # Semantic cube: create type that contains 'B' and instantiate exactly one cube
        cubes_info = ensure_cubes("B", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["B"]
        self.semantic_cube = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])
        self.cubes = [self.semantic_cube]
        self._target_labels = ["B"]

        # Metrics init (FULL signature required)
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

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # Enforced workspace bounds per requirements (do not rely on class constants here)
        x_bounds = (0.40, 0.70)
        y_bounds = (-0.25, 0.25)

        # 3. Place patch at a random pose (reachable, within bounds)
        patch_xy = np.array(
            [
                float(rng.uniform(*x_bounds)),
                float(rng.uniform(*y_bounds)),
            ],
            dtype=float,
        )
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_REST_Z], dtype=float)
        self.blue_goal_patch.body.set_position(patch_pos)

        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # 4. Place cube at a random, UNSOLVED pose (not on patch + not B-up)
        #    Rejection sample to ensure sufficient XY separation from patch.
        min_sep = float(max(self._MIN_CUBE_PATCH_SPACING_XY, self._TRANSPORT_XY_THRESH + 0.03))
        for _ in range(200):
            cube_xy = np.array(
                [
                    float(rng.uniform(*x_bounds)),
                    float(rng.uniform(*y_bounds)),
                ],
                dtype=float,
            )
            if float(np.linalg.norm(cube_xy - patch_xy)) >= min_sep:
                break
        else:
            # Fallback: push along x within bounds if sampling fails (still unsolved orientation below)
            cube_xy = patch_xy.copy()
            cube_xy[0] = float(np.clip(cube_xy[0] - min_sep, *x_bounds))

        cube_pos = np.array([cube_xy[0], cube_xy[1], self._CUBE_REST_Z], dtype=float)

        # 5. Set cube to an UNSOLVED orientation: B NOT readable on top (+Z)
        face_idx_b = self.semantic_cube._labels.index("B")
        init_quat = self.semantic_cube.get_orientation_for_face_and_glyph(
            face_idx=face_idx_b,
            face_dir_world=self._INIT_FACE_DIR_WORLD,
        )
        self.semantic_cube.set_pose(position=cube_pos, quat=init_quat.elements)

        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # 6. Physics settling (REQUIRED)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        self._mojo.physics.forward()

        # 7. Initialize subgoals to False (6 subgoals)
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_cube_on_patch(self) -> bool:
        # Reliable GoalPatch collision pattern: cube -> patch.collider
        if not hasattr(self, "blue_goal_patch") or not hasattr(self.blue_goal_patch, "collider"):
            return False
        if not self.semantic_cube.is_colliding(self.blue_goal_patch.collider):
            return False
        return True

    def _success(self) -> bool:
        # Reset success flag at START
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "blue_goal_patch"):
            return False

        cube = self.semantic_cube
        patch = self.blue_goal_patch

        cube_pos = cube.body.get_position()
        patch_pos = patch.body.get_position()
        xy_dist = float(np.linalg.norm(cube_pos[:2] - patch_pos[:2]))

        # Subgoal 1: grasp
        grasped = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
        self._metric_stage(0, bool(grasped))

        # Subgoal 2: lift
        lifted = (cube_pos[2] > self._LIFT_Z_THRESH) and (not cube.is_colliding(self.table.table_top))
        self._metric_stage(1, bool(lifted))

        # Subgoal 3: transport above patch (heuristic)
        transported = xy_dist < self._TRANSPORT_XY_THRESH
        self._metric_stage(2, bool(transported))

        # Goal condition 2: orientation (semantic verification REQUIRED)
        semantic_ok = bool(cube.is_face_readable("B", face_dir=self._FACE_DIR_WORLD))
        self._metric_stage(3, bool(semantic_ok))

        # Goal condition 1: on patch (table_top collision can be false when resting on patch)
        on_patch = self._is_cube_on_patch()
        floor_ok = True
        if hasattr(self, "_floor"):
            floor_ok = not bool(cube.is_colliding(self._floor))
        placed = bool(on_patch and floor_ok and (abs(float(cube_pos[2]) - self._CUBE_REST_Z) < 0.03))
        self._metric_stage(4, bool(placed))

        # Released (required for final success)
        released = not any(
            self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers
        )
        self._metric_stage(5, bool(placed and released))

        positions_ok = placed

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Pose error metric: angular distance to the exact "B-up readable" target orientation
        face_idx_b = cube._labels.index("B")
        target_q = cube.get_orientation_for_face_and_glyph(face_idx=face_idx_b, face_dir_world=self._FACE_DIR_WORLD)
        current_q = Quaternion(cube.body.get_quaternion())
        dq = Quaternion(target_q.elements) * current_q.inverse
        angle_deg = float(np.degrees(dq.angle))

        # Finalize metrics at END (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_to_patch_xy_m": xy_dist,
            },
            pose_error={
                "B_up_orientation_error_deg": angle_deg,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})