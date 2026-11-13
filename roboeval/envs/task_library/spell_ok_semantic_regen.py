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

class SpellOkCubeAlignmentTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange two semantic letter cubes side by side to spell "OK" as seen by the robot,
    with the letters facing the robot/camera. The "O" must be on the left (+Y) and
    the "K" on the right (-Y), aligned at similar front/back position (X).

    Sequential Sub-Goals:
        1. Bring the two cubes into the correct left/right ordering (O left of K).
        2. Align the cubes front/back (X alignment).
        3. Rotate both cubes so their target letters face the robot (toward -X).
        4. Release both cubes (no gripper holding).
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task requires letters to face the robot/camera on the BACK face (-X direction).
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Goal thresholds (from task specification)
    _MIN_Y_DELTA = 0.06          # O.y > K.y by at least 6 cm
    _X_ALIGN_TOL = 0.03          # |O.x - K.x| <= 3 cm

    # Workspace-appropriate goal positions (must match provided computed coordinates)
    _GOAL_POS_O = np.array([0.55, 0.03, 0.97])
    _GOAL_POS_K = np.array([0.55, -0.03, 0.97])

    _TARGET_LABELS = ["O", "K"]
    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create the required SemanticCubes (ONLY allowed asset for this task)
        cubes_info = ensure_cubes("".join(self._TARGET_LABELS), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def _xml_key(info: Dict):
            # Support either key name seen in different toolchains.
            return str(info["xml"]) if "xml" in info else str(info["xml_path"])

        info_o = cubes_info["O"]
        info_k = cubes_info["K"]

        self.semantic_cube_O = SemanticCube(
            self._mojo,
            xml_path=_xml_key(info_o),
            labels=info_o["labels"],
        )
        self.semantic_cube_K = SemanticCube(
            self._mojo,
            xml_path=_xml_key(info_k),
            labels=info_k["labels"],
        )

        self.cubes = [self.semantic_cube_O, self.semantic_cube_K]

        # Metrics (REQUIRED full signature)
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

        # Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # Workspace bounds (requirement)
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Sample positions that are intentionally UNSOLVED:
        # - break X alignment by enforcing |dx| > _X_ALIGN_TOL + margin
        # - break left/right ordering by enforcing (O.y - K.y) < _MIN_Y_DELTA
        margin_x = 0.02
        min_dx = self._X_ALIGN_TOL + margin_x

        # Choose Xs far enough apart
        x_o = float(rng.uniform(x_lo, x_hi))
        # Ensure |x_k - x_o| >= min_dx while staying in bounds
        for _ in range(50):
            x_k = float(rng.uniform(x_lo, x_hi))
            if abs(x_k - x_o) >= min_dx:
                break
        else:
            # Fallback: push to bounds deterministically
            x_k = float(np.clip(x_o + np.sign(rng.standard_normal()) * min_dx, x_lo, x_hi))

        # Choose Ys such that y_delta < _MIN_Y_DELTA (unsolved)
        y_k = float(rng.uniform(y_lo, y_hi))
        # Force O not sufficiently to the left (+Y) of K (could be equal/right of K)
        y_o_max = min(y_hi, y_k + self._MIN_Y_DELTA - 0.01)
        y_o = float(rng.uniform(y_lo, y_o_max)) if y_o_max > y_lo else float(y_lo)

        pos_o = np.array([x_o, y_o, z_base], dtype=float)
        pos_k = np.array([x_k, y_k, z_base], dtype=float)

        # Randomize orientations to further prevent semantic success (letters not necessarily facing -X)
        def _random_quat() -> Quaternion:
            # Uniform-ish random orientation via random axis-angle (sufficient for making readable unlikely)
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            return Quaternion(axis=axis, angle=angle)

        self.semantic_cube_O.set_pose(position=pos_o, quat=_random_quat().elements)
        self.semantic_cube_K.set_pose(position=pos_k, quat=_random_quat().elements)

        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # Reset success flag at START
        self._success_check = False

        # Defensive checks (called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: sort cubes by Y desc so slot[0]=left target "O", slot[1]=right target "K".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_O_cube, o_pos = cubes_sorted[0]
        slot_K_cube, k_pos = cubes_sorted[1]

        # Spatial checks
        y_delta = float(o_pos[1] - k_pos[1])
        x_err = float(abs(o_pos[0] - k_pos[0]))

        left_right_ok = y_delta >= self._MIN_Y_DELTA
        x_aligned_ok = x_err <= self._X_ALIGN_TOL

        # Keep them on the table (light sanity check)
        on_table_ok = (
            slot_O_cube.is_colliding(self.table.table_top)
            and slot_K_cube.is_colliding(self.table.table_top)
        )

        positions_ok = bool(left_right_ok and x_aligned_ok and on_table_ok)

        # SEMANTIC VERIFICATION: cube currently in each slot must show that slot's target letter.
        semantic_ok = (
            slot_O_cube.is_face_readable("O", face_dir=self._FACE_DIR)
            and slot_K_cube.is_face_readable("K", face_dir=self._FACE_DIR)
        )

        # Gripper release check (generic over grippers)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Subgoal staging
        if left_right_ok:
            self._metric_stage(0, True)
        if x_aligned_ok:
            self._metric_stage(1, True)
        if semantic_ok:
            self._metric_stage(2, True)
        if released:
            self._metric_stage(3, True)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "y_order_margin(O_minus_K_minus_min)": float(y_delta - self._MIN_Y_DELTA),
                "x_alignment_error_abs": float(x_err),
            },
            pose_error={
                "O_face_readable": float(1.0 if slot_O_cube.is_face_readable("O", face_dir=self._FACE_DIR) else 0.0),
                "K_face_readable": float(1.0 if slot_K_cube.is_face_readable("K", face_dir=self._FACE_DIR) else 0.0),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})