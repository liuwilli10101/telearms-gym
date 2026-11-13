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


class DualCubeSideBySidePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place a semantic cube and a red cube side by side on one green patch.

Success Criteria:
- The semantic cube and red cube are both on and fully within the green patch.
- The two cubes are side by side at the allowed center distance and aligned along X or Y rather than diagonal.
- The semantic cube's target label is readable, the colored cube is red, and no gripper is holding either cube."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For this flat placement task, keep the semantic glyph readable on the TOP face.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Task parameters (from specification)
    _PATCH_SIZE = (0.10, 0.10)  # (width_x, depth_y)
    _CONTAIN_TOL = 0.02
    _SIDE_BY_SIDE_MIN = 0.04
    _SIDE_BY_SIDE_MAX = 0.075
    _ALIGN_MAX_MISALIGN = 0.02
    _CUBE_HALF = 0.02  # 4cm cube

    def _initialize_env(self):
        # Table already exists in the preset
        self.table = self._preset.get_props(Table)[0]

        # --- Goal patch (green) ---
        self.goalpatch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="target")

        # --- Semantic cube (single) ---
        # Use ensure_cubes() workflow (required)
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]
        xml_path = info.get("xml", info.get("xml_path"))
        self.semanticcube = SemanticCube(self._mojo, xml_path=str(xml_path), labels=info["labels"])
        self._target_labels = ["A"]

        # --- Colored cube (red) ---
        self.red_cube = ColoredCube(self._mojo)
        self.red_cube.set_color("red")  # REQUIRED in _initialize_env

        # Convenience list for metrics / release checks
        self.cubes = [self.semanticcube, self.red_cube]

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
        # Re-init metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG (pipeline may add extra randomization later)
        rng = np.random.default_rng(self.seed)

        # Ensure correct cube color on every reset (REQUIRED)
        self.red_cube.set_color("red")

        # --- Keep goal patch fixed in workspace ---
        patch_pos = np.array([0.55, 0.00, 0.951])
        self.goalpatch.body.set_position(patch_pos)
        self.goalpatch.body.set_quaternion(np.array([0.0, 0.0, 0.0, 1.0]))
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # --- Randomize cubes in UNSOLVED poses within workspace bounds ---
        # Workspace bounds
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        Z_BASE = 0.97

        # Patch "fully within" extents used by success checks
        half_w = self._PATCH_SIZE[0] / 2.0
        half_d = self._PATCH_SIZE[1] / 2.0
        max_dx = (half_w - self._CUBE_HALF) + self._CONTAIN_TOL
        max_dy = (half_d - self._CUBE_HALF) + self._CONTAIN_TOL

        # Ensure cubes start outside the patch region to guarantee _success() == False
        # Add margin so "on patch" / "within" won't be satisfied accidentally.
        outside_margin = 0.02

        def sample_pos_outside_patch():
            for _ in range(1000):
                x = rng.uniform(X_MIN, X_MAX)
                y = rng.uniform(Y_MIN, Y_MAX)
                # Outside the fully-within region (and typically outside contact region too)
                if (abs(x - patch_pos[0]) > (max_dx + outside_margin)) or (abs(y - patch_pos[1]) > (max_dy + outside_margin)):
                    return np.array([x, y, Z_BASE])
            # Fallback: place clearly outside along X within bounds
            return np.array([X_MIN + 0.02, Y_MAX - 0.02, Z_BASE])

        sem_pos = sample_pos_outside_patch()

        # Ensure red cube is not side-by-side adjacent to semantic cube (avoid accidental stage 2)
        for _ in range(1000):
            red_pos = sample_pos_outside_patch()
            dist_xy = float(np.linalg.norm((red_pos - sem_pos)[:2]))
            if dist_xy > (self._SIDE_BY_SIDE_MAX + 0.03):
                break
        else:
            red_pos = np.array([X_MAX - 0.02, Y_MIN + 0.02, Z_BASE])

        # Randomize orientations (not required for failure, but increases unsolved diversity)
        def random_unit_quat():
            q = rng.normal(size=4)
            q = q / (np.linalg.norm(q) + 1e-12)
            # return in (x,y,z,w)
            return q.astype(float)

        self.semanticcube.set_pose(position=sem_pos, quat=random_unit_quat())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.red_cube.set_pose(position=red_pos, quat=random_unit_quat())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(4):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_block_on_patch(self, block, patch) -> bool:
        # GoalPatch collision is more reliable via patch.collider and checking from the BLOCK side
        if not hasattr(patch, "collider"):
            return False
        return bool(block.is_colliding(patch.collider))

    def _is_fully_within_patch_xy(self, block, patch, tol: float) -> bool:
        p = patch.body.get_position()
        b = block.body.get_position()

        half_w = self._PATCH_SIZE[0] / 2.0
        half_d = self._PATCH_SIZE[1] / 2.0

        # Cube must be fully inside patch boundaries (with tolerance)
        max_dx = (half_w - self._CUBE_HALF) + tol
        max_dy = (half_d - self._CUBE_HALF) + tol
        return (abs(b[0] - p[0]) <= max_dx) and (abs(b[1] - p[1]) <= max_dy)

    def _success(self) -> bool:
        # RESET success flag at START (required)
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goalpatch"):
            return False

        sem = self.semanticcube
        red = self.red_cube
        patch = self.goalpatch

        # --- Patch / containment checks ---
        sem_on = self._is_block_on_patch(sem, patch)
        red_on = self._is_block_on_patch(red, patch)

        sem_within = self._is_fully_within_patch_xy(sem, patch, tol=self._CONTAIN_TOL)
        red_within = self._is_fully_within_patch_xy(red, patch, tol=self._CONTAIN_TOL)

        # --- Relative arrangement checks (side-by-side + aligned) ---
        ps = sem.body.get_position()
        pr = red.body.get_position()
        dxy = pr[:2] - ps[:2]
        dist_xy = float(np.linalg.norm(dxy))
        side_by_side_ok = (self._SIDE_BY_SIDE_MIN <= dist_xy <= self._SIDE_BY_SIDE_MAX)

        # "Aligned along x or y" => small offset on at least one axis (not diagonal)
        aligned_ok = (abs(dxy[0]) <= self._ALIGN_MAX_MISALIGN) or (abs(dxy[1]) <= self._ALIGN_MAX_MISALIGN)

        # --- Semantic + color verification (required) ---
        # NOTE: is_face_readable uses parameter name 'face_dir' (no _world)
        semantic_ok = bool(sem.is_face_readable(self._target_labels[0], face_dir=self._FACE_DIR))
        color_ok = bool(red.get_color() == "red")

        # --- Released check (required pattern: iterate all grippers) ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in [sem, red] for s in self.robot.grippers
        )

        # Stage updates
        if sem_on and sem_within and semantic_ok:
            self._metric_stage(0, True)
        if red_on and red_within and color_ok:
            self._metric_stage(1, True)
        if side_by_side_ok and aligned_ok:
            self._metric_stage(2, True)
        if released:
            self._metric_stage(3, True)

        positions_ok = bool(sem_on and red_on and sem_within and red_within and side_by_side_ok and aligned_ok)
        content_ok = bool(semantic_ok and color_ok)

        self._success_check = bool(positions_ok and content_ok and released)

        # Metrics finalize (must be last metric call before return)
        patch_pos = patch.body.get_position()
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "semantic_to_patch_center_xy": float(np.linalg.norm((ps - patch_pos)[:2])),
                "red_to_patch_center_xy": float(np.linalg.norm((pr - patch_pos)[:2])),
                "cube_pair_center_distance_xy": dist_xy,
            },
            pose_error={
                "pair_axis_misalignment_m": float(min(abs(dxy[0]), abs(dxy[1]))),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})