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
class SemanticCubeAbcStacking(RoboEvalEnv, MetricRolloutEval):
    """Stack semantic cubes into a C-B-A tower from bottom to top with readable labels.

Success Criteria:
- After sorting by height, the bottom, middle, and top stack slots correspond to C, B, and A.
- C is on the table, B contacts C and not the table, A contacts B and not the table, with valid XY and Z stack alignment.
- The C/B/A labels are readable in the required face direction and no gripper is holding any cube."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For vertical stacks, the camera-visible semantic face should be the BACK face (-X).
    _FACE_DIR = np.array([-1.0, 0.0, 0.0])

    # Tolerances (task constraints)
    _XY_TOL = 0.014
    _Z_TOL = 0.02
    _STACK_DZ = 0.04

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create SemanticCubes (ONLY allowed asset)
        cubes_info = ensure_cubes("ABC", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        def _xml_from(info: Dict) -> str:
            return str(info.get("xml", info.get("xml_path")))

        info_a = cubes_info["A"]
        info_b = cubes_info["B"]
        info_c = cubes_info["C"]

        self.cube_c = SemanticCube(self._mojo, xml_path=_xml_from(info_c), labels=info_c["labels"])
        self.cube_b = SemanticCube(self._mojo, xml_path=_xml_from(info_b), labels=info_b["labels"])
        self.cube_a = SemanticCube(self._mojo, xml_path=_xml_from(info_a), labels=info_a["labels"])

        # Ordering here matches the desired stack bottom->top (C, B, A)
        self.cubes = [self.cube_c, self.cube_b, self.cube_a]
        self._target_labels = ["C", "B", "A"]

        # Pre-compute a canonical goal stack center (not enforced as absolute in success)
        self._goal_xy = np.array([0.55, 0.00])
        self._goal_positions = [
            np.array([self._goal_xy[0], self._goal_xy[1], 0.97]),               # C (table)
            np.array([self._goal_xy[0], self._goal_xy[1], 0.97 + self._STACK_DZ]),  # B
            np.array([self._goal_xy[0], self._goal_xy[1], 0.97 + 2 * self._STACK_DZ]),  # A
        ]

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
        # Re-initialize metrics (required every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        # Sample 3 well-separated XY positions on the table to ensure UNSOLVED (not stacked)
        min_sep = 0.08  # keep cubes apart to avoid accidental stacking/collisions
        xys = []
        for _ in range(1000):
            cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])
            if all(np.linalg.norm(cand - p) >= min_sep for p in xys):
                xys.append(cand)
                if len(xys) == 3:
                    break
        if len(xys) < 3:
            # Fallback deterministic spread if sampling ever fails
            xys = [
                np.array([0.45, -0.18]),
                np.array([0.62, 0.00]),
                np.array([0.52, 0.18]),
            ]

        # Place cubes randomly on table (all at same base Z), and guarantee semantic failure by
        # orienting at least one target face away from the required facing direction.
        for i, (cube, target_label) in enumerate(zip(self.cubes, self._target_labels)):
            pos = np.array([xys[i][0], xys[i][1], z_base])

            if cube is self.cube_c:
                # Force "C" to face AWAY from robot so semantic check fails deterministically.
                face_idx = cube._labels.index("C")
                q = cube.get_orientation_for_face_and_glyph(
                    face_idx=face_idx,
                    face_dir_world=-self._FACE_DIR,  # opposite required direction
                )
                quat = q.elements
            else:
                # Random orientation for other cubes (deterministic from rng; pyquaternion Quaternion.random has no rng kw)
                u1 = float(rng.random())
                u2 = float(rng.random())
                u3 = float(rng.random())
                qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
                qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
                qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
                qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
                quat = np.array([qw, qx, qy, qz], dtype=float)

            cube.set_pose(position=pos, quat=quat)

        # Settle physics
        self._mojo.physics.forward()

        # Initialize metric subgoals to False
        self._num_subgoals = 5
        for idx in range(self._num_subgoals):
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

        # Identity-free: sort cubes by Z asc so slot[0]=bottom target "C", slot[1]="B", slot[2]="A".
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][2])
        cube_c, cube_b, cube_a = cubes_sorted[0][0], cubes_sorted[1][0], cubes_sorted[2][0]
        pos_c, pos_b, pos_a = cubes_sorted[0][1], cubes_sorted[1][1], cubes_sorted[2][1]

        def _xy_close(p, q, tol=self._XY_TOL) -> bool:
            return float(np.linalg.norm(np.array(p[:2]) - np.array(q[:2]))) <= tol

        def _z_close_delta(p, q, dz, tol=self._Z_TOL) -> bool:
            return abs(float((p[2] - q[2]) - dz)) <= tol

        # Spatial: stacking relations
        c_on_table = cube_c.is_colliding(self.table.table_top)
        b_on_c = cube_b.is_colliding(cube_c) and (not cube_b.is_colliding(self.table.table_top))
        a_on_b = cube_a.is_colliding(cube_b) and (not cube_a.is_colliding(self.table.table_top))

        aligned_xy = _xy_close(pos_b, pos_c) and _xy_close(pos_a, pos_b)
        aligned_z = _z_close_delta(pos_b, pos_c, self._STACK_DZ) and _z_close_delta(pos_a, pos_b, self._STACK_DZ)

        positions_ok = bool(c_on_table and b_on_c and a_on_b and aligned_xy and aligned_z)

        # Semantic verification: cube currently in each stack slot must show that slot's target letter.
        slot_pairs = [(cube_c, "C"), (cube_b, "B"), (cube_a, "A")]
        semantic_ok = all(c.is_face_readable(t, face_dir=self._FACE_DIR) for c, t in slot_pairs)

        # Gripper release check
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics (coarse-grained)
        # 0: C is base on table
        self._metric_stage(0, bool(c_on_table))
        # 1: C readable toward robot
        self._metric_stage(1, bool(cube_c.is_face_readable("C", face_dir=self._FACE_DIR)))
        # 2: B stacked on C and readable
        self._metric_stage(
            2,
            bool(b_on_c and _xy_close(pos_b, pos_c) and cube_b.is_face_readable("B", face_dir=self._FACE_DIR)),
        )
        # 3: A stacked on B and readable
        self._metric_stage(
            3,
            bool(a_on_b and _xy_close(pos_a, pos_b) and cube_a.is_face_readable("A", face_dir=self._FACE_DIR)),
        )
        # 4: released
        self._metric_stage(4, bool(released))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "dist_A_to_B": float(distance(cube_a.body, cube_b.body)),
                "dist_B_to_C": float(distance(cube_b.body, cube_c.body)),
            },
            pose_error={
                "xy_err_A_to_B": float(np.linalg.norm(pos_a[:2] - pos_b[:2])),
                "xy_err_B_to_C": float(np.linalg.norm(pos_b[:2] - pos_c[:2])),
                "z_err_A_to_B": float(abs((pos_a[2] - pos_b[2]) - self._STACK_DZ)),
                "z_err_B_to_C": float(abs((pos_b[2] - pos_c[2]) - self._STACK_DZ)),
                "semantic_ok": float(1.0 if semantic_ok else 0.0),
                "released": float(1.0 if released else 0.0),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})