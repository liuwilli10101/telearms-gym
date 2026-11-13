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

class SideBySideCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Pick up two colored cubes and place them next to each other on a single red target patch.
    The final configuration requires both cubes to be on the patch and arranged side by side.

    Sequential Sub-Goals:
        1. Place cube_1 on the red goal patch.
        2. Place cube_2 on the red goal patch.
        3. Arrange cubes side-by-side (small gap) aligned along the y-axis (same x).
        4. Release both cubes (no gripper holding either cube).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometry constants (meters)
    _CUBE_SIZE = 0.04
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    # Goal tolerances
    _ALIGN_X_TOL = 0.014  # alignment tolerance for "aligned_axis": 'y' => same x
    _MAX_GAP_M = 0.03     # maximum allowed edge-to-edge gap for side-by-side

    # Metric stages
    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Objects (ONLY allowed assets)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2]

        # Target patch (red)
        self.goal_patch_red = GoalPatch(self._mojo, color="red", size=(0.1, 0.1), name="target")

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("blue")
        self.cube_2.set_color("green")

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

    def _physics_forward(self):
        """Best-effort physics forward call (some backends expose physics.forward())."""
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def _settle(self, n: int = 50):
        for _ in range(n):
            self._mojo.step()

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """
        Reliable on-patch check using the GoalPatch .collider pattern.
        """
        if not hasattr(target, "collider"):
            return False
        if not block.is_colliding(target.collider):
            return False
        return True

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

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("blue")
        self.cube_2.set_color("green")

        # Workspace bounds (meters)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Place patch randomly within bounds
        patch_pos = np.array(
            [
                float(rng.uniform(x_min + 0.05, x_max - 0.05)),
                float(rng.uniform(y_min + 0.05, y_max - 0.05)),
                self._PATCH_REST_Z,
            ],
            dtype=float,
        )
        self.goal_patch_red.body.set_position(patch_pos)
        self._physics_forward()
        self._settle()

        # Place cubes in random UNSOLVED poses (ensure not on patch, not side-by-side success)
        def sample_cube_xy(existing: list[np.ndarray]) -> np.ndarray:
            # Keep cubes clearly off the patch: outside a conservative square around it.
            # Patch size is 0.10m => half-extent 0.05; keep extra margin.
            off_patch_margin = 0.09
            min_sep = 0.07  # avoid overlaps / immediate contacts

            for _ in range(200):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))

                # Not on/near patch
                if abs(x - float(patch_pos[0])) <= off_patch_margin and abs(y - float(patch_pos[1])) <= off_patch_margin:
                    continue

                p = np.array([x, y], dtype=float)

                # Not too close to other cube(s)
                if any(float(np.linalg.norm(p - q)) < min_sep for q in existing):
                    continue

                return p

            # Fallback to corners if sampling fails (still within bounds)
            return np.array([x_min, y_max], dtype=float)

        # Sample positions
        p1_xy = sample_cube_xy([])
        p2_xy = sample_cube_xy([p1_xy])

        cube1_pos = np.array([p1_xy[0], p1_xy[1], self._CUBE_REST_Z], dtype=float)
        cube2_pos = np.array([p2_xy[0], p2_xy[1], self._CUBE_REST_Z], dtype=float)

        # Random yaw for variety (quat format: x,y,z,w; yaw about z)
        def yaw_quat(yaw: float) -> np.ndarray:
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw))
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        self.cube_1.set_pose(position=cube1_pos, quat=yaw_quat(rng.uniform(-np.pi, np.pi)))
        self._physics_forward()
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        self._settle()

        self.cube_2.set_pose(position=cube2_pos, quat=yaw_quat(rng.uniform(-np.pi, np.pi)))
        self._physics_forward()
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        self._settle()

        # Extra settling
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()
        self._settle()

        # Safety: if somehow solved (shouldn't be), force cube_2 far in Y (still in bounds)
        if self._success():
            forced_y = float(y_min if cube1_pos[1] > 0 else y_max)
            forced_x = float(np.clip(cube1_pos[0] + 0.08, x_min, x_max))
            self.cube_2.set_pose(
                position=np.array([forced_x, forced_y, self._CUBE_REST_Z], dtype=float),
                quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            )
            self._physics_forward()
            self._settle()

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
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch_red"):
            return False

        # Goal condition 1 & 2: both cubes on the red patch
        on_patch_1 = self._is_block_on_target(self.cube_1, self.goal_patch_red)
        on_patch_2 = self._is_block_on_target(self.cube_2, self.goal_patch_red)
        positions_ok = bool(on_patch_1 and on_patch_2)

        # Goal condition 3: side-by-side with small gap, aligned along y-axis (same x)
        p1 = self.cube_1.body.get_position()
        p2 = self.cube_2.body.get_position()

        x_aligned = abs(float(p1[0] - p2[0])) <= self._ALIGN_X_TOL

        center_dy = abs(float(p1[1] - p2[1]))
        edge_gap = center_dy - self._CUBE_SIZE  # gap between cube faces along Y
        gap_ok = (edge_gap >= -0.005) and (edge_gap <= self._MAX_GAP_M)  # allow tiny penetration tolerance

        arrangement_ok = bool(x_aligned and gap_ok)

        # Released: no gripper holding either cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics
        if on_patch_1:
            self._metric_stage(0, True)
        if on_patch_2:
            self._metric_stage(1, True)
        if positions_ok and arrangement_ok:
            self._metric_stage(2, True)
        if released:
            self._metric_stage(3, True)

        # Final success
        semantic_ok = True  # no semantic constraints beyond spatial arrangement in this task
        self._success_check = bool(positions_ok and arrangement_ok and semantic_ok and released)

        # Metrics finalize
        patch_xy = self.goal_patch_red.body.get_position()[:2]
        d1 = float(np.linalg.norm(self.cube_1.body.get_position()[:2] - patch_xy))
        d2 = float(np.linalg.norm(self.cube_2.body.get_position()[:2] - patch_xy))
        align_err_x = float(abs(p1[0] - p2[0]))
        gap_err = float(max(0.0, edge_gap - self._MAX_GAP_M))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_1_to_patch_xy": d1,
                "cube_2_to_patch_xy": d2,
                "cube_pair_x_alignment_error": align_err_x,
            },
            pose_error={
                "side_by_side_edge_gap_m": float(edge_gap),
                "side_by_side_gap_excess_m": gap_err,
            },
        )
        return self._success_check

    def get_goal_conditions(self):
        return [
            {
                "type": "position",
                "subject": "cube_1",
                "target": "goal_patch_red",
                "parameters": {"relation": "on"},
                "description": "Cube 1 is placed on the red goal patch.",
            },
            {
                "type": "position",
                "subject": "cube_2",
                "target": "goal_patch_red",
                "parameters": {"relation": "on"},
                "description": "Cube 2 is placed on the red goal patch.",
            },
            {
                "type": "arrangement",
                "subject": "cube_1",
                "target": "cube_2",
                "parameters": {
                    "relation": "side_by_side",
                    "max_gap_m": self._MAX_GAP_M,
                    "aligned_axis": "y",
                },
                "description": "The two cubes are side by side with a small gap and aligned to form a neat pair on the patch.",
            },
        ]

    def get_task_instruction(self):
        return "Place the two colored cubes side by side on the red target patch."

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})