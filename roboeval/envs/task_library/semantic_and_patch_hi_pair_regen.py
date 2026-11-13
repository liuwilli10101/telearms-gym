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


class HiCubePlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Place the two semantic cubes labeled 'H' and 'I' onto the single green goal patch so they read 'HI'
    from the robot's viewpoint. Both letters must face toward the robot (toward -X) and be arranged
    left-to-right using the +Y (left) / -Y (right) convention.

    Sequential Sub-Goals:
        1. Place both cubes on the green goal patch.
        2. Arrange cubes to read left-to-right: H on the left (+Y), I on the right (-Y), aligned in X.
        3. Ensure both letters face toward the robot (-X) and release all objects.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task requires letters to face the robot/camera: toward -X
    _FACE_DIR_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    # Goal tolerances/thresholds from spec
    _ON_PATCH_TOL_XY = 0.03
    _MIN_Y_SEP = 0.04
    _MAX_X_DELTA = 0.03
    _MIN_COS_SIM = 0.9

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Assets: SemanticCube (H, I) ---
        self._target_labels = ["H", "I"]
        cubes_info = ensure_cubes("".join(self._target_labels), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def _xml_key(info: Dict) -> str:
            return str(info["xml_path"]) if "xml_path" in info else str(info["xml"])

        info_h = cubes_info["H"]
        info_i = cubes_info["I"]
        self.cube_H = SemanticCube(self._mojo, xml_path=_xml_key(info_h), labels=info_h["labels"])
        self.cube_I = SemanticCube(self._mojo, xml_path=_xml_key(info_i), labels=info_i["labels"])
        self.cubes = [self.cube_H, self.cube_I]

        # --- Asset: GoalPatch (green) ---
        self.goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # Goal-state poses (must make _success() True)
        self._goal_patch_pos = np.array([0.55, 0.0, 0.951], dtype=np.float32)
        # Keep both cubes within 3cm of patch center while maintaining >=4cm separation in Y
        self._goal_pos_H = np.array([0.55, +0.025, 0.97], dtype=np.float32)  # left (+Y)
        self._goal_pos_I = np.array([0.55, -0.025, 0.97], dtype=np.float32)  # right (-Y)

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

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        # 3. Place GoalPatch (kept at nominal goal location)
        self.goal_patch.body.set_position(self._goal_patch_pos)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        elif hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "step"):
            self._mojo.physics.step()
        else:
            self._mojo.physics.forward()

        patch_xy = self._goal_patch_pos[:2].copy()

        # 4. Randomize cube initial poses in UNSOLVED state within workspace bounds
        #    Guarantee failure by (a) reversing left-right order and (b) facing +X (not -X),
        #    and keep cubes away from the patch so on-patch check is not accidentally satisfied.
        min_patch_clearance = 0.09  # > on-patch tol, conservative vs patch size

        # Sample a shared x near each other (doesn't matter), and pick Y so H is on right, I on left.
        for _ in range(200):
            x0 = float(rng.uniform(x_min, x_max))
            y_center = float(rng.uniform(y_min + 0.06, y_max - 0.06))
            y_offset = float(rng.uniform(0.01, 0.03))  # keep them relatively close; order will be reversed

            pos_H = np.array([x0, y_center - y_offset, z_base], dtype=np.float32)  # H on -Y (right) -> wrong
            pos_I = np.array([x0 + float(rng.uniform(-0.01, 0.01)), y_center + y_offset, z_base], dtype=np.float32)

            dH = float(np.linalg.norm(pos_H[:2] - patch_xy))
            dI = float(np.linalg.norm(pos_I[:2] - patch_xy))
            if dH >= min_patch_clearance and dI >= min_patch_clearance:
                break
        else:
            # Fallback (still within bounds and clearly off-patch / wrong order)
            pos_H = np.array([0.45, -0.20, z_base], dtype=np.float32)
            pos_I = np.array([0.46, +0.20, z_base], dtype=np.float32)

        # Force letters to face +X (opposite of required -X) to ensure semantic_ok is False after reset
        wrong_face_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        face_idx_H = self.cube_H._labels.index("H")
        face_idx_I = self.cube_I._labels.index("I")
        quat_H = self.cube_H.get_orientation_for_face_and_glyph(face_idx=face_idx_H, face_dir_world=wrong_face_dir)
        quat_I = self.cube_I.get_orientation_for_face_and_glyph(face_idx=face_idx_I, face_dir_world=wrong_face_dir)

        self.cube_H.set_pose(position=pos_H, quat=quat_H.elements)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        elif hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "step"):
            self._mojo.physics.step()
        else:
            self._mojo.physics.forward()

        self.cube_I.set_pose(position=pos_I, quat=quat_I.elements)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        elif hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "step"):
            self._mojo.physics.step()
        else:
            self._mojo.physics.forward()

        # 5. Physics settling
        for _ in range(50):
            if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "step"):
                self._mojo.physics.step()
            else:
                self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(3):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "goal_patch") or not hasattr(self, "table"):
            return False

        patch_pos = self.goal_patch.body.get_position()

        # --- Identity-free positional/semantic check ---
        # Sort cubes by Y desc so slot[0]=left target "H", slot[1]=right target "I".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_left_cube, pos_H = cubes_sorted[0]
        slot_right_cube, pos_I = cubes_sorted[1]

        # --- On-patch checks (use collider-direction pattern + XY tolerance) ---
        # GoalPatch collision can be unreliable; use cube.is_colliding(patch.collider)
        on_patch_H = False
        on_patch_I = False
        if hasattr(self.goal_patch, "collider"):
            on_patch_H = slot_left_cube.is_colliding(self.goal_patch.collider)
            on_patch_I = slot_right_cube.is_colliding(self.goal_patch.collider)

        dH_xy = float(np.linalg.norm((pos_H[:2] - patch_pos[:2])))
        dI_xy = float(np.linalg.norm((pos_I[:2] - patch_pos[:2])))
        close_H = dH_xy <= self._ON_PATCH_TOL_XY
        close_I = dI_xy <= self._ON_PATCH_TOL_XY

        # Also ensure cubes are resting on table height band (SemanticCube base z=0.97)
        z_ok_H = abs(float(pos_H[2]) - 0.97) <= 0.03
        z_ok_I = abs(float(pos_I[2]) - 0.97) <= 0.03

        positions_ok = bool((on_patch_H or close_H) and (on_patch_I or close_I) and z_ok_H and z_ok_I)

        # --- Relative arrangement: H left of I (+Y), aligned in X ---
        left_of_ok = (float(pos_H[1]) - float(pos_I[1])) >= self._MIN_Y_SEP
        aligned_x_ok = abs(float(pos_H[0]) - float(pos_I[0])) <= self._MAX_X_DELTA
        arrangement_ok = bool(left_of_ok and aligned_x_ok)

        # --- Semantic verification: cube in left slot must show "H", cube in right slot must show "I" ---
        semantic_ok = True
        if not slot_left_cube.is_face_readable("H", face_dir=self._FACE_DIR_WORLD):
            semantic_ok = False
        if not slot_right_cube.is_face_readable("I", face_dir=self._FACE_DIR_WORLD):
            semantic_ok = False

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal metrics
        if positions_ok:
            self._metric_stage(0, True)
        if positions_ok and arrangement_ok:
            self._metric_stage(1, True)
        if positions_ok and arrangement_ok and semantic_ok and released:
            self._metric_stage(2, True)

        self._success_check = bool(positions_ok and arrangement_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_H_to_patch_xy": dH_xy,
                "cube_I_to_patch_xy": dI_xy,
                "H_minus_I_y_sep_err": float(max(0.0, self._MIN_Y_SEP - (float(pos_H[1]) - float(pos_I[1])))),
            },
            pose_error={
                "x_alignment_abs_err": abs(float(pos_H[0]) - float(pos_I[0])),
                "H_readable": float(semantic_ok and slot_left_cube.is_face_readable("H", face_dir=self._FACE_DIR_WORLD)),
                "I_readable": float(semantic_ok and slot_right_cube.is_face_readable("I", face_dir=self._FACE_DIR_WORLD)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})