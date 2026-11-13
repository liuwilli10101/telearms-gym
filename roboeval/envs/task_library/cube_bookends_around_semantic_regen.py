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
class CenteredSemanticCubeRow(RoboEvalEnv, MetricRolloutEval):
    """Make a three-cube row with the semantic cube centered between two colored cubes.

Success Criteria:
- The semantic cube is centered between two colored cubes along Y, with the colored cubes on left and right sides at the required separation.
- All three cubes are aligned in X.
- The semantic cube's target label is readable on the required face direction, and no gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For a flat horizontal arrangement, the top face is the best readable face.
    _FACE_DIR_WORLD = np.array([0.0, 0.0, 1.0])

    # Goal geometry (must respect min pairwise XY spacing >= 0.0966m)
    _CENTER_X = 0.55
    _CENTER_Y = 0.00
    _ROW_DY = 0.10  # >= 0.0966 to satisfy mandatory min distance constraints
    _Z_TABLE = 0.97

    # Tolerances and constraints from task spec
    _CENTER_TOL_X = 0.03
    _CENTER_TOL_Y = 0.03
    _ALIGN_TOL_X = 0.03
    _MIN_SEP_Y = 0.06

    # Subgoal count for metrics
    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Create cubes (ONLY allowed assets: ColoredCube, SemanticCube) ---
        self.colored_cube_1 = ColoredCube(self._mojo)
        self.colored_cube_2 = ColoredCube(self._mojo)

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.colored_cube_1.set_color("red")
        self.colored_cube_2.set_color("blue")

        # Semantic cube: MUST use ensure_cubes() workflow
        self._target_label = "A"
        cubes_info = ensure_cubes(self._target_label, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._target_label]
        xml_path = str(info["xml_path"] if "xml_path" in info else info["xml"])
        self.semantic_cube_1 = SemanticCube(self._mojo, xml_path=xml_path, labels=info["labels"])

        # Convenience list for metrics / release checks
        self.cubes = [self.semantic_cube_1, self.colored_cube_1, self.colored_cube_2]

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

        # 2. Deterministic RNG (validation may randomize later)
        rng = np.random.default_rng(self.seed)

        # 3. Set colors again (REQUIRED)
        self.colored_cube_1.set_color("red")
        self.colored_cube_2.set_color("blue")

        # 4. Place objects in RANDOM UNSOLVED positions within workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._Z_TABLE

        # Ensure semantic cube is NOT centered (guarantees _success() == False after reset)
        # Centered window:
        cx0, cx1 = self._CENTER_X - self._CENTER_TOL_X, self._CENTER_X + self._CENTER_TOL_X
        cy0, cy1 = self._CENTER_Y - self._CENTER_TOL_Y, self._CENTER_Y + self._CENTER_TOL_Y

        def sample_xy_not_centered():
            for _ in range(2000):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                centered_ok = (cx0 <= x <= cx1) and (cy0 <= y <= cy1)
                if not centered_ok:
                    return np.array([x, y])
            # Fallback: deterministic non-centered point
            return np.array([x_min, y_max])

        def sample_xy():
            return np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))])

        def far_enough(p, others, min_dist=0.08):
            for o in others:
                if np.linalg.norm(p - o) < min_dist:
                    return False
            return True

        # Sample semantic cube XY first (not centered), then other cubes with simple separation
        sem_xy = sample_xy_not_centered()
        others = [sem_xy]

        for _ in range(2000):
            c1_xy = sample_xy()
            if far_enough(c1_xy, others):
                others.append(c1_xy)
                break
        else:
            c1_xy = np.array([x_max, y_min])
            others.append(c1_xy)

        for _ in range(2000):
            c2_xy = sample_xy()
            if far_enough(c2_xy, others):
                break
        else:
            c2_xy = np.array([x_max, y_max])

        pos_sem = np.array([sem_xy[0], sem_xy[1], z])
        pos_c1 = np.array([c1_xy[0], c1_xy[1], z])
        pos_c2 = np.array([c2_xy[0], c2_xy[1], z])

        # Randomize orientations (also helps avoid semantic readability being satisfied)
        q_sem = rng.normal(size=4)
        q_sem = q_sem / (np.linalg.norm(q_sem) + 1e-12)
        q_c1 = rng.normal(size=4)
        q_c1 = q_c1 / (np.linalg.norm(q_c1) + 1e-12)
        q_c2 = rng.normal(size=4)
        q_c2 = q_c2 / (np.linalg.norm(q_c2) + 1e-12)

        self.semantic_cube_1.set_pose(position=pos_sem, quat=q_sem)
        self._mojo.physics.forward()

        self.colored_cube_1.set_pose(position=pos_c1, quat=q_c1)
        self._mojo.physics.forward()

        self.colored_cube_2.set_pose(position=pos_c2, quat=q_c2)
        self._mojo.physics.forward()

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
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
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if any(obj is None for obj in [self.semantic_cube_1, self.colored_cube_1, self.colored_cube_2]):
            return False

        # Identity-free among the two colored cubes: the task says "one colored cube
        # to the left and one to the right" and never says which colour goes which
        # side, but the check required the RED one on the left and the BLUE one on
        # the right. Whichever sits further left simply is the left one.
        left_cube, right_cube = sorted(
            [self.colored_cube_1, self.colored_cube_2],
            key=lambda c: float(c.body.get_position()[1]),
            reverse=True,
        )

        sem_p = self.semantic_cube_1.body.get_position()
        c1_p = left_cube.body.get_position()
        c2_p = right_cube.body.get_position()

        # --- Goal condition 1: semantic cube centered between the two colored cubes ---
        # Translation-invariant: compare to the midpoint of the two flanking cubes,
        # not to a fixed world-frame position.
        midpoint_y = 0.5 * (c1_p[1] + c2_p[1])
        centered_ok = abs(sem_p[1] - midpoint_y) <= self._CENTER_TOL_Y

        # --- Goal condition 2 & 3: left/right relations along Y with min separation ---
        left_ok = (c1_p[1] - sem_p[1]) >= self._MIN_SEP_Y
        right_ok = (sem_p[1] - c2_p[1]) >= self._MIN_SEP_Y

        # --- Goal conditions 4-6: X alignment across all three cubes ---
        align_c1_c2_x = abs(c1_p[0] - c2_p[0]) <= self._ALIGN_TOL_X
        align_c1_sem_x = abs(c1_p[0] - sem_p[0]) <= self._ALIGN_TOL_X
        align_c2_sem_x = abs(c2_p[0] - sem_p[0]) <= self._ALIGN_TOL_X
        x_aligned_ok = align_c1_c2_x and align_c1_sem_x and align_c2_sem_x

        positions_ok = bool(centered_ok and left_ok and right_ok and x_aligned_ok)

        # --- SEMANTIC VERIFICATION (REQUIRED): target label must be readable ---
        semantic_ok = bool(self.semantic_cube_1.is_face_readable(self._target_label, face_dir=self._FACE_DIR_WORLD))

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal metrics ---
        if centered_ok:
            self._metric_stage(0, True)
        if left_ok and align_c1_sem_x:
            self._metric_stage(1, True)
        if right_ok and align_c2_sem_x:
            self._metric_stage(2, True)
        if x_aligned_ok:
            self._metric_stage(3, True)
        if semantic_ok:
            self._metric_stage(4, True)
        if released:
            self._metric_stage(5, True)

        self._success_check = bool(positions_ok and semantic_ok and released)
        if self._success_check:
            self._metric_stage(6, True)

        # --- Metrics finalize (MUST be last metric call before return) ---
        goal_sem = np.array([self._CENTER_X, self._CENTER_Y, self._Z_TABLE])
        goal_c1 = np.array([self._CENTER_X, self._CENTER_Y + self._ROW_DY, self._Z_TABLE])
        goal_c2 = np.array([self._CENTER_X, self._CENTER_Y - self._ROW_DY, self._Z_TABLE])

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "semantic_cube_1": float(np.linalg.norm(sem_p[:2] - goal_sem[:2])),
                "colored_cube_1": float(np.linalg.norm(c1_p[:2] - goal_c1[:2])),
                "colored_cube_2": float(np.linalg.norm(c2_p[:2] - goal_c2[:2])),
            },
            pose_error={
                # Colored cubes: pose/orientation not required for success in this task
                "semantic_cube_readable_error": float(0.0 if semantic_ok else 1.0),
                "x_alignment_error_max": float(
                    max(abs(c1_p[0] - sem_p[0]), abs(c2_p[0] - sem_p[0]), abs(c1_p[0] - c2_p[0]))
                ),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})