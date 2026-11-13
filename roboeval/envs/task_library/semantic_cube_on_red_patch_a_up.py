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
    Pick up the semantic cube and place it onto the red target patch. Finish with the cube
    oriented so the face labeled 'A' is pointing upward.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the semantic cube.
        2. Lift the cube clear of the table.
        3. Transport the cube above the red goal patch.
        4. Rotate/align the cube so the 'A' face is oriented upward.
        5. Lower the cube onto the red target patch.
        6. Release the cube, leaving it stably on the patch with the 'A' face up.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951
    _FACE_DIR_UP = np.array([0.0, 0.0, 1.0])  # 'A' must face +Z
    _ON_PATCH_XY_TOL = 0.014  # meters
    _A_UP_TOL_DEG = 15.0

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create required assets (ONLY SemanticCube + GoalPatch) ---
        # Goal patch (red)
        self.goal_patch_red_1 = GoalPatch(
            self._mojo, color="red", size=(0.1, 0.1), name="goal_patch_red_1"
        )

        # Semantic cube: must use ensure_cubes() workflow
        cubes_info = ensure_cubes("A", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info["A"]
        self.semantic_cube_1 = SemanticCube(
            self._mojo, xml_path=str(info["xml"]), labels=info["labels"]
        )

        # Keep a list for metric/slip tracking (template expects self.cubes)
        self.cubes = [self.semantic_cube_1]

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
        # Requirement: call physics.forward() after each placement (best-effort).
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

    def compute_pose_errors(self):
        """Compute task-relevant pose errors for reporting."""
        cube = self.semantic_cube_1
        patch = self.goal_patch_red_1

        cube_pos = np.array(cube.body.get_position())
        patch_pos = np.array(patch.body.get_position())

        xy_dist = float(np.linalg.norm(cube_pos[:2] - patch_pos[:2]))

        # Angle between the world +Z and the world direction of the cube face that carries 'A'
        a_face_up_deg = float(self._compute_face_up_angle_deg(cube, label="A"))

        return {"xy_dist_to_patch_m": xy_dist, "a_face_up_deg": a_face_up_deg}

    def _compute_face_up_angle_deg(self, cube: SemanticCube, label: str) -> float:
        # Determine which face index contains the label, then compute that face's world normal.
        try:
            face_idx = cube._labels.index(label)
        except Exception:
            return 180.0

        # Local face normals by face index (per provided mapping)
        local_normals = {
            0: np.array([1.0, 0.0, 0.0]),   # +X
            1: np.array([-1.0, 0.0, 0.0]),  # -X
            2: np.array([0.0, 1.0, 0.0]),   # +Y
            3: np.array([0.0, -1.0, 0.0]),  # -Y
            4: np.array([0.0, 0.0, 1.0]),   # +Z (top)
            5: np.array([0.0, 0.0, -1.0]),  # -Z (bottom)
        }
        n_local = local_normals.get(face_idx, np.array([0.0, 0.0, 1.0]))

        q = Quaternion(cube.body.get_quaternion())
        n_world = np.array(q.rotate(n_local), dtype=float)
        n_world /= (np.linalg.norm(n_world) + 1e-12)

        z = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(n_world, z), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def check_on_patch_relation(self, cube: SemanticCube, patch: GoalPatch) -> bool:
        """
        Reliable 'on patch' check using the patch collider direction:
          cube.is_colliding(patch.collider)
        Also reject floor collisions if available.
        """
        if not hasattr(patch, "collider"):
            return False
        if not cube.is_colliding(patch.collider):
            return False
        if hasattr(self, "_floor") and cube.is_colliding(self._floor):
            return False
        return True

    def check_face_up_orientation(self, cube: SemanticCube, label: str = "A") -> bool:
        """
        Semantic verification (required) + geometric tolerance:
          - label must be readable on the upward face direction
          - the labeled face normal must be within tolerance of +Z
        """
        semantic_ok = cube.is_face_readable(label, face_dir=self._FACE_DIR_UP)
        if not semantic_ok:
            return False
        ang = self._compute_face_up_angle_deg(cube, label=label)
        return bool(ang <= self._A_UP_TOL_DEG)

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

        # 2. Deterministic RNG (per-task seed)
        rng = np.random.default_rng(getattr(self, "seed", 0))

        # 3. Randomize objects into UNSOLVED configuration within workspace bounds
        # Workspace: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25

        # Random patch position
        patch_xy = np.array(
            [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), self._PATCH_REST_Z],
            dtype=float,
        )

        # Random cube position, kept away from patch in XY so it's not already "on patch"
        # Ensure distance > tolerance + margin
        min_sep = float(self._ON_PATCH_XY_TOL + 0.06)
        for _ in range(200):
            cube_xy = np.array(
                [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), self._CUBE_REST_Z],
                dtype=float,
            )
            if float(np.linalg.norm(cube_xy[:2] - patch_xy[:2])) >= min_sep:
                break

        # Place patch (GoalPatch uses body API; no set_pose)
        self.goal_patch_red_1.body.set_position(patch_xy)
        self._physics_forward()
        self._mojo.physics.forward()

        # Place cube near table height with a random orientation that is NOT 'A' face up
        cube = self.semantic_cube_1
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        for _ in range(200):
            # Deterministic random quaternion using this episode RNG (pyquaternion.Quaternion.random
            # does not accept an RNG positional argument in this environment).
            q_raw = rng.normal(size=4).astype(float)
            q_raw /= (np.linalg.norm(q_raw) + 1e-12)
            quat = np.array([q_raw[0], q_raw[1], q_raw[2], q_raw[3]], dtype=float)
            try:
                cube.set_pose(position=cube_xy, quat=quat)
                self._physics_forward()
                self._mojo.physics.forward()
                if not self.check_face_up_orientation(cube, label="A"):
                    break
            except Exception:
                # Fallback: still set pose if possible; if not, we'll leave default quat
                try:
                    cube.set_pose(position=cube_xy, quat=quat)
                except Exception:
                    pass
                break

        # 4. Physics settling (required)
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # Reset success flag at start
        self._success_check = False

        # Defensive checks - _success() may be called during reset
        if not hasattr(self, "semantic_cube_1") or not hasattr(self, "goal_patch_red_1"):
            return False
        if not hasattr(self, "table") or not hasattr(self, "robot"):
            return False

        cube = self.semantic_cube_1
        patch = self.goal_patch_red_1

        # --- Goal condition 1: cube on red patch ---
        on_patch = self.check_on_patch_relation(cube, patch)

        cube_pos = np.array(cube.body.get_position())
        patch_pos = np.array(patch.body.get_position())
        xy_dist = float(np.linalg.norm(cube_pos[:2] - patch_pos[:2]))
        positions_ok = bool(on_patch and (xy_dist <= self._ON_PATCH_XY_TOL))

        # --- Goal condition 2: 'A' face up (semantic + geometric tolerance) ---
        orientation_ok = self.check_face_up_orientation(cube, label="A")

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers
        )

        # Stage metrics (best-effort; final stage requires full success)
        holding = not released
        lifted = bool(cube_pos[2] > (self._CUBE_REST_Z + 0.02))
        near_patch = bool(xy_dist < 0.08)

        self._metric_stage(0, bool(holding))                      # grasp
        self._metric_stage(1, bool(lifted))                       # lift
        self._metric_stage(2, bool(near_patch and (holding or lifted)))  # transport
        self._metric_stage(3, bool(orientation_ok and (holding or lifted)))  # align
        self._metric_stage(4, bool(positions_ok))                 # place
        self._metric_stage(5, bool(positions_ok and orientation_ok and released))  # release

        # Final success
        self._success_check = bool(positions_ok and orientation_ok and released)

        pose_err = self.compute_pose_errors()
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"cube_to_patch_xy_m": pose_err["xy_dist_to_patch_m"]},
            pose_error={"a_face_up_deg": pose_err["a_face_up_deg"]},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})