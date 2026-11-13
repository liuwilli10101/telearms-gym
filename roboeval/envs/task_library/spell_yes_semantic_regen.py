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

class YesCubeRowArrangement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three semantic cubes into a single left-to-right row that spells YES
    as seen by the robot/camera, with Y on the left (+Y), E in the middle, and
    S on the right (-Y). Ensure the letter faces are oriented toward the robot/
    camera direction (toward -X).

    Sequential Sub-Goals:
        0. Place Y to the LEFT of E (higher Y by at least 0.04m).
        1. Place E to the LEFT of S (higher Y by at least 0.04m).
        2. Align all three cubes into a single row along Y with consistent spacing,
           similar X, and resting on the table.
        3. Orient cubes so Y/E/S faces are readable toward -X.
        4. Release all cubes (no gripper holding).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Face direction for semantic readability (toward robot/camera): -X
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Goal tolerances/thresholds (from task spec)
    _MIN_SEP = 0.04
    _X_TOL = 0.03
    _Z_TOL = 0.02
    _SPACING_RANGE = (0.04, 0.12)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Target word/cubes (content-based semantic cubes)
        self._target_labels = ["Y", "E", "S"]

        cubes_info = ensure_cubes("YES", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def _xml_key(info: Dict) -> str:
            # ensure_cubes() historically returned either "xml" or "xml_path"
            return str(info["xml"] if "xml" in info else info["xml_path"])

        info_y = cubes_info["Y"]
        info_e = cubes_info["E"]
        info_s = cubes_info["S"]

        self.cube_Y = SemanticCube(self._mojo, xml_path=_xml_key(info_y), labels=info_y["labels"])
        self.cube_E = SemanticCube(self._mojo, xml_path=_xml_key(info_e), labels=info_e["labels"])
        self.cube_S = SemanticCube(self._mojo, xml_path=_xml_key(info_s), labels=info_s["labels"])

        self.cubes = [self.cube_Y, self.cube_E, self.cube_S]

        # Fixed goal positions: a row along Y (left-to-right is +Y to -Y)
        # Keep X constant (front/back), Z on table at 0.97.
        self._goal_positions = {
            "Y": np.array([0.55, 0.08, 0.97]),
            "E": np.array([0.55, 0.00, 0.97]),
            "S": np.array([0.55, -0.08, 0.97]),
        }
        self._goal_quats = {k: None for k in ["Y", "E", "S"]}

        # Metrics (must be initialized with full signature)
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
        # Re-initialize metrics (required on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # --- Store goal quaternions (for metrics), but DO NOT place cubes in goal state ---
        for cube, target_label in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(target_label)
            q_goal = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=self._FACE_DIR,
            )
            self._goal_quats[target_label] = np.array(q_goal.elements, dtype=float)

        # --- Randomize initial poses in UNSOLVED configuration within workspace bounds ---
        # Workspace bounds
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25
        Z0 = 0.97

        # Sample three non-overlapping positions; force incorrect Y ordering (Y not left of E, E not left of S)
        min_xy_sep = 0.065

        def _sample_positions_unsolved():
            # Choose three distinct Y locations then assign labels in reverse order to violate stage0/stage1
            ys = np.sort(rng.uniform(Y_MIN, Y_MAX, size=3))  # ys[0] < ys[1] < ys[2]
            # Assign: Y gets smallest, E middle, S largest -> violates required Y > E > S
            y_map = {"Y": ys[0], "E": ys[1], "S": ys[2]}

            # Sample X independently to also tend to violate x-alignment
            x_map = {k: float(rng.uniform(X_MIN, X_MAX)) for k in ["Y", "E", "S"]}

            pos_map = {k: np.array([x_map[k], y_map[k], Z0], dtype=float) for k in ["Y", "E", "S"]}

            # Reject if too close (to avoid interpenetration)
            keys = ["Y", "E", "S"]
            for i in range(3):
                for j in range(i + 1, 3):
                    pi = pos_map[keys[i]][:2]
                    pj = pos_map[keys[j]][:2]
                    if float(np.linalg.norm(pi - pj)) < min_xy_sep:
                        return None
            return pos_map

        pos_map = None
        for _ in range(200):
            pos_map = _sample_positions_unsolved()
            if pos_map is not None:
                break
        if pos_map is None:
            # Fallback: deterministic separated but UNSOLVED layout (still within bounds)
            pos_map = {
                "Y": np.array([0.45, -0.12, Z0], dtype=float),
                "E": np.array([0.60, 0.00, Z0], dtype=float),
                "S": np.array([0.50, 0.12, Z0], dtype=float),
            }

        # Set orientations so target faces are NOT readable toward -X (use opposite face direction +X)
        wrong_face_dir = -self._FACE_DIR  # +X
        for cube, target_label in zip(self.cubes, self._target_labels):
            face_idx = cube._labels.index(target_label)
            q_wrong = cube.get_orientation_for_face_and_glyph(
                face_idx=face_idx,
                face_dir_world=wrong_face_dir,
            )
            cube.set_pose(position=pos_map[target_label], quat=q_wrong.elements)

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # Reset success flag at start
        self._success_check = False

        # Defensive checks (_success() can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: sort cubes by Y desc so slot[0]=left target "Y", slot[1]="E", slot[2]="S".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]
        slot_targets = list(self._target_labels)  # ["Y","E","S"]

        slot_Y_cube, slot_E_cube, slot_S_cube = slot_cubes[0], slot_cubes[1], slot_cubes[2]
        pY, pE, pS = slot_positions[0], slot_positions[1], slot_positions[2]

        # --- Spatial checks (ordering and row alignment) ---
        # Left = +Y, Right = -Y
        stage0 = (pY[1] - pE[1]) >= self._MIN_SEP
        stage1 = (pE[1] - pS[1]) >= self._MIN_SEP

        # Row alignment: similar X, near table Z, and spacing within range
        xs = np.array([pY[0], pE[0], pS[0]])
        zs = np.array([pY[2], pE[2], pS[2]])
        x_aligned = (np.max(xs) - np.min(xs)) <= self._X_TOL
        z_ok = np.all(np.abs(zs - 0.97) <= self._Z_TOL)

        dy_ye = pY[1] - pE[1]
        dy_es = pE[1] - pS[1]
        spacing_ok = (self._SPACING_RANGE[0] <= dy_ye <= self._SPACING_RANGE[1]) and (
            self._SPACING_RANGE[0] <= dy_es <= self._SPACING_RANGE[1]
        )

        # Ensure cubes are resting on the table and not interpenetrating each other
        on_table = all(c.is_colliding(self.table.table_top) for c in self.cubes)
        no_overlap = not (
            slot_Y_cube.is_colliding(slot_E_cube)
            or slot_E_cube.is_colliding(slot_S_cube)
            or slot_Y_cube.is_colliding(slot_S_cube)
        )

        stage2 = bool(x_aligned and z_ok and spacing_ok and on_table and no_overlap)
        positions_ok = bool(stage0 and stage1 and stage2)

        # --- Semantic verification: cube currently in each slot must show that slot's target letter ---
        semantic_ok = all(
            c.is_face_readable(t, face_dir=self._FACE_DIR)
            for c, t in zip(slot_cubes, slot_targets)
        )
        stage3 = bool(semantic_ok)

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )
        stage4 = bool(released)

        # Metrics stages
        self._metric_stage(0, stage0)
        self._metric_stage(1, stage1)
        self._metric_stage(2, stage2)
        self._metric_stage(3, stage3)
        self._metric_stage(4, stage4)

        # Final success condition
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (must be last metric call before return)
        def _pos_err(cube, label: str) -> float:
            return float(np.linalg.norm(cube.body.get_position() - self._goal_positions[label]))

        def _ori_err_deg(cube, label: str) -> float:
            goal_q = self._goal_quats[label]
            if goal_q is None:
                return 180.0
            q_cur = Quaternion(cube.body.get_quaternion())
            q_goal = Quaternion(goal_q)
            q_rel = q_goal.inverse * q_cur
            return float(abs(q_rel.angle) * 180.0 / np.pi)

        target_distance = {
            "cube_Y_to_goal": _pos_err(slot_Y_cube, "Y"),
            "cube_E_to_goal": _pos_err(slot_E_cube, "E"),
            "cube_S_to_goal": _pos_err(slot_S_cube, "S"),
        }
        pose_error = {
            "cube_Y_ori_deg": _ori_err_deg(slot_Y_cube, "Y"),
            "cube_E_ori_deg": _ori_err_deg(slot_E_cube, "E"),
            "cube_S_ori_deg": _ori_err_deg(slot_S_cube, "S"),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})