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


class RedOnSemanticCubeStacking(RoboEvalEnv, MetricRolloutEval):
    """
    Pick up the red colored cube and stack it on top of the semantic cube. The semantic
    cube must also be oriented so its 'A' face faces the robot (toward -X, upright and
    readable). The task succeeds when the red cube is stably resting on the semantic cube
    AND the semantic cube's 'A' face is turned toward the robot, with nothing held.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the red colored cube with either gripper.
        2. Lift the red colored cube clear of the table surface.
        3. Move the red colored cube above the semantic cube while maintaining clearance.
        4. Align the red cube over the semantic cube's top face for a stable stack.
        5. Lower the red cube onto the semantic cube's top face.
        6. Release the red cube after it is stably supported by the semantic cube.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For a vertical stack, the camera best sees the back (-X) face of the lower cube.
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Stacking geometry (4cm cube)
    _CUBE_SIZE = 0.04
    _TABLE_Z_CUBE_REST = 0.97
    _STACK_DZ = 0.04

    # Spatial tolerances / thresholds
    _XY_ALIGN_TOL = 0.014  # per task constraint
    _Z_STACK_TOL = 0.018

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY these two) ---
        self.colored_cube_red = ColoredCube(self._mojo)
        self.colored_cube_red.set_color("red")  # REQUIRED in _initialize_env

        # Create exactly one SemanticCube type and instance (semantic content must be verifiable)
        # Choose a deterministic target label (single character) to verify readability.
        self._target_label = "A"
        cubes_info = ensure_cubes(self._target_label, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._target_label]
        self.semantic_cube_1 = SemanticCube(self._mojo, xml_path=str(info["xml"]), labels=info["labels"])

        # Keep a canonical list for metrics and release checks
        self.cubes = [self.colored_cube_red, self.semantic_cube_1]

        # Initialize metrics (FULL signature)
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

        # REQUIRED: set color in _on_reset as well
        self.colored_cube_red.set_color("red")

        # Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # --- Place objects in RANDOM *unsolved* configuration within workspace bounds ---
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Enforce non-stacked placement: keep XY separation > _XY_ALIGN_TOL and avoid red-on-top dz.
        min_xy_sep = max(self._XY_ALIGN_TOL + 0.04, 0.06)

        def sample_xy():
            return float(rng.uniform(0.40, 0.70)), float(rng.uniform(-0.25, 0.25))

        sem_x, sem_y = sample_xy()
        # Resample red until far enough in XY to guarantee xy_ok is False
        for _ in range(200):
            red_x, red_y = sample_xy()
            if float(np.linalg.norm(np.array([red_x - sem_x, red_y - sem_y], dtype=float))) > min_xy_sep:
                break
        else:
            # Fallback: place at opposite side in X within bounds
            red_x = float(np.clip(sem_x + 0.20, 0.40, 0.70))
            red_y = float(np.clip(sem_y + 0.20, -0.25, 0.25))

        sem_pos = np.array([sem_x, sem_y, self._TABLE_Z_CUBE_REST], dtype=float)
        red_pos = np.array([red_x, red_y, self._TABLE_Z_CUBE_REST], dtype=float)

        # Orient semantic cube so its target label is readable on the chosen face direction.
        face_idx = self.semantic_cube_1._labels.index(self._target_label)
        q = self.semantic_cube_1.get_orientation_for_face_and_glyph(
            face_idx=face_idx,
            face_dir_world=self._FACE_DIR,
        )

        # Parking + place with settling (keeps reset stable, avoids accidental stacking/contact)
        self.colored_cube_red.set_pose(
            position=np.array([red_x, red_y, 1.8], dtype=float),
            quat=np.array([0, 0, 0, 1], dtype=float),
        )
        self._mojo.physics.forward()

        self.semantic_cube_1.set_pose(position=sem_pos, quat=q.elements)
        self._mojo.physics.forward()

        # Place red on table away from semantic cube (unsolved by construction)
        self.colored_cube_red.set_pose(position=red_pos, quat=np.array([0, 0, 0, 1], dtype=float))
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks (called during reset in some pipelines)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "colored_cube_red") or not hasattr(self, "semantic_cube_1"):
            return False

        red = self.colored_cube_red
        sem = self.semantic_cube_1

        red_pos = red.body.get_position()
        sem_pos = sem.body.get_position()

        xy_dist = float(np.linalg.norm(red_pos[:2] - sem_pos[:2]))
        z_delta = float(red_pos[2] - sem_pos[2])

        # --- Goal condition: red cube is on top of semantic cube ---
        contact_ok = bool(red.is_colliding(sem))
        off_table = bool(not red.is_colliding(self.table.table_top))
        xy_ok = bool(xy_dist < self._XY_ALIGN_TOL)
        z_ok = bool(abs(z_delta - self._STACK_DZ) < self._Z_STACK_TOL)

        positions_ok = bool(contact_ok and off_table and xy_ok and z_ok)

        # --- REQUIRED: Semantic verification for SemanticCube ---
        semantic_ok = bool(sem.is_face_readable(self._target_label, face_dir=self._FACE_DIR))

        # --- Released gripper check (iterate over all grippers) ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in [red, sem] for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, sequential semantics) ---
        grasped = any(self.robot.is_gripper_holding_object(red, s) for s in self.robot.grippers)
        lifted = bool(grasped and (not red.is_colliding(self.table.table_top)) and (red_pos[2] > self._TABLE_Z_CUBE_REST + 0.02))
        transported = bool(lifted and (xy_dist < 0.08) and (red_pos[2] > sem_pos[2] + 0.06))
        aligned = bool(transported and (xy_dist < 0.02))
        placed = bool(contact_ok and off_table and z_ok and (xy_dist < 0.02))
        released_after_place = bool(placed and released)

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(aligned))
        self._metric_stage(4, bool(placed))
        self._metric_stage(5, bool(released_after_place))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_semantic_xy": xy_dist,
                "red_to_semantic_dz_err": float(abs(z_delta - self._STACK_DZ)),
            },
            pose_error={},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})