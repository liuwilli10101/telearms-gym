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

class TopLetterCubeStackingTask(RoboEvalEnv, MetricRolloutEval):
    """
    Build a vertical stack of three semantic cubes so that P is on the bottom, O is in the middle,
    and T is on top. Rotate the cubes so the letters T-O-P are visible to the robot on the
    robot-facing (back) faces.

    Sequential Sub-Goals:
        1. Grasp the cube that will be placed at the bottom (P).
        2. Move the P cube to the stacking location.
        3. Rotate/align the P cube so 'P' is on the robot-facing back (-X) face.
        4. Place the P cube on the table as the bottom of the stack.
        5. Grasp the cube that will be placed in the middle (O).
        6. Rotate/align the O cube so 'O' is on the robot-facing back (-X) face.
        7. Place the O cube on top of the P cube.
        8. Grasp the cube that will be placed on top (T).
        9. Rotate/align the T cube so 'T' is on the robot-facing back (-X) face.
        10. Place the T cube on top of the O cube to complete the vertical stack.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For vertical stacks, the robot/camera primarily sees the BACK face (-X).
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Workspace/geometry constants
    _STACK_XY = np.array([0.55, 0.00])
    _Z_BASE = 0.97
    _DZ = 0.04
    _XY_TOL = 0.014

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create ONLY SemanticCube assets (required)
        cubes_info = ensure_cubes("TOP", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        # Instantiate cubes (use required instantiation pattern)
        info_p = cubes_info["P"]
        info_o = cubes_info["O"]
        info_t = cubes_info["T"]

        self.semantic_cube_P = SemanticCube(
            self._mojo, xml_path=str(info_p["xml"]), labels=info_p["labels"]
        )
        self.semantic_cube_O = SemanticCube(
            self._mojo, xml_path=str(info_o["xml"]), labels=info_o["labels"]
        )
        self.semantic_cube_T = SemanticCube(
            self._mojo, xml_path=str(info_t["xml"]), labels=info_t["labels"]
        )

        # Bottom-to-top ordering for checks
        self.cubes = [self.semantic_cube_P, self.semantic_cube_O, self.semantic_cube_T]
        self._target_labels = ["P", "O", "T"]

        # Metrics (full signature required)
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
        # Re-initialize metrics on every reset (required)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG (required pattern; validation may add perturbations later)
        rng = np.random.default_rng(self.seed)

        # RANDOM UNSOLVED STATE: place cubes separated on the table within workspace bounds.
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._Z_BASE)

        # Ensure cubes are not stacked/aligned: enforce minimum XY separation.
        min_sep = 0.08  # comfortably > stacking tol and typical cube footprint margin

        def sample_xy(existing):
            for _ in range(500):
                xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
                if all(np.linalg.norm(xy - e) >= min_sep for e in existing):
                    return xy
            # Fallback: deterministic spaced placement if repeated sampling fails
            base = np.array([0.45, -0.15], dtype=float)
            offsets = [
                np.array([0.00, 0.00]),
                np.array([0.18, 0.00]),
                np.array([0.09, 0.24]),
            ]
            for off in offsets:
                xy2 = np.clip(base + off, [x_min, y_min], [x_max, y_max])
                if all(np.linalg.norm(xy2 - e) >= min_sep for e in existing):
                    return xy2
            return np.clip(base, [x_min, y_min], [x_max, y_max])

        xys = []
        for _ in range(3):
            xys.append(sample_xy(xys))

        # Random orientations (likely not matching required readable face); also avoid identical yaws.
        yaws = rng.uniform(-np.pi, np.pi, size=3)
        if abs(yaws[1] - yaws[0]) < 0.5:
            yaws[1] += 1.0
        if abs(yaws[2] - yaws[0]) < 0.5:
            yaws[2] -= 1.0

        cubes = [self.semantic_cube_P, self.semantic_cube_O, self.semantic_cube_T]
        for cube, xy, yaw in zip(cubes, xys, yaws):
            pos = np.array([xy[0], xy[1], z], dtype=float)
            quat = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw))
            cube.set_pose(position=pos, quat=quat.elements)
            self._mojo.physics.forward()

        # Initialize subgoal stages to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # Reset success flag at start (required)
        self._success_check = False

        # Defensive checks (success can be called during reset)
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Identity-free: sort cubes by Z ascending so slot[0]=bottom target "P",
        # slot[1]=middle target "O", slot[2]=top target "T".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][2])
        p, o, t = [cp[0] for cp in cubes_sorted]
        p_pos, o_pos, t_pos = [cp[1] for cp in cubes_sorted]
        slot_targets = list(self._target_labels)  # ["P","O","T"] bottom to top

        # --- Spatial checks (stack relations + alignment) ---
        # Bottom cube must be on the table
        p_on_table = p.is_colliding(self.table.table_top)

        # O on P, T on O (collision-based)
        o_on_p = o.is_colliding(p) and (not o.is_colliding(self.table.table_top))
        t_on_o = t.is_colliding(o) and (not t.is_colliding(self.table.table_top))

        # XY alignment to ensure vertical stack (within tolerance)
        po_xy_ok = np.linalg.norm((o_pos - p_pos)[:2]) <= self._XY_TOL
        ot_xy_ok = np.linalg.norm((t_pos - o_pos)[:2]) <= self._XY_TOL

        # Basic Z ordering (avoid inverted stack)
        z_order_ok = (t_pos[2] > o_pos[2]) and (o_pos[2] > p_pos[2])

        positions_ok = bool(p_on_table and o_on_p and t_on_o and po_xy_ok and ot_xy_ok and z_order_ok)

        # --- Semantic verification: cube in each stack slot must show that slot's target letter ---
        semantic_ok = all(
            cube.is_face_readable(letter, face_dir=self._FACE_DIR)
            for cube, letter in zip([p, o, t], slot_targets)
        )

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, observable-only) ---
        # Stage indices correspond to the provided subgoals (1..10).
        p_oriented = p.is_face_readable("P", face_dir=self._FACE_DIR)
        o_oriented = o.is_face_readable("O", face_dir=self._FACE_DIR)
        t_oriented = t.is_face_readable("T", face_dir=self._FACE_DIR)

        # 1-4: P grasp/transport/align/place (we can only verify align+place reliably)
        self._metric_stage(0, False)  # grasp P (not directly observable)
        self._metric_stage(1, False)  # transport P (not directly observable)
        self._metric_stage(2, bool(p_oriented))
        self._metric_stage(3, bool(p_on_table and p_oriented))

        # 5-7: O grasp/align/place-on-P
        self._metric_stage(4, False)  # grasp O (not directly observable)
        self._metric_stage(5, bool(o_oriented))
        self._metric_stage(6, bool(o_on_p and o_oriented and p_on_table))

        # 8-10: T grasp/align/place-on-O
        self._metric_stage(7, False)  # grasp T (not directly observable)
        self._metric_stage(8, bool(t_oriented))
        self._metric_stage(9, bool(t_on_o and t_oriented and o_on_p and p_on_table))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        stack_center = np.array([self._STACK_XY[0], self._STACK_XY[1], self._Z_BASE])
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "P_to_stack_center_xy": float(np.linalg.norm((p_pos - stack_center)[:2])),
                "O_to_P_xy": float(np.linalg.norm((o_pos - p_pos)[:2])),
                "T_to_O_xy": float(np.linalg.norm((t_pos - o_pos)[:2])),
            },
            pose_error={
                # 0.0 if readable on the required face, else 1.0
                "P_back_face_readable_error": 0.0 if p_oriented else 1.0,
                "O_back_face_readable_error": 0.0 if o_oriented else 1.0,
                "T_back_face_readable_error": 0.0 if t_oriented else 1.0,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})