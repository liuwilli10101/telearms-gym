from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube


class FourVerticalStacksColorGradientNumberOrderWord(RoboEvalEnv, MetricRolloutEval):
    """Build four stacks of two: blue–green, 1–2, A–T, and red–red, bottom to top.

Success Criteria:
Each base rests on the table and its upper cube rests on it. Exact centering and uprightness are not required. Keep stacks separated, semantic labels readable, and release all cubes."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # SemanticCube API expects a string face_dir, not a vector.
    _FACE_DIR = "back"

    # Geometry constants
    _CUBE_SIZE = 0.04
    _Z_BOTTOM = 0.97
    _Z_STEP = 0.04

    # Tolerances (from task spec)
    _BASE_XY_TOL = 0.03
    _STACK_XY_TOL = 0.03
    _Z_TOL = 0.03
    _UPRIGHT_DEG_TOL = 20.0

    # Minimum XY separation between different stacks' centroids (translation-invariant).
    _STACK_SEPARATION = 0.10

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Colored cubes (4) ---
        self.cube_01 = ColoredCube(self._mojo)  # blue
        self.cube_02 = ColoredCube(self._mojo)  # green
        self.cube_03 = ColoredCube(self._mojo)  # red
        self.cube_04 = ColoredCube(self._mojo)  # red

        # REQUIRED: set_color() in _initialize_env
        self.cube_01.set_color("blue")
        self.cube_02.set_color("green")
        self.cube_03.set_color("red")
        self.cube_04.set_color("red")

        # --- Semantic cubes (4): digits 1,2 and letters A,T ---
        # Two per stack instead of three. The third cube always had to go onto an
        # already-tall tower, which was the reported difficulty and is not the
        # ordering skill this task tests. The word stack drops from CAT to AT.
        digits_info = ensure_cubes("12", universe="0123456789")
        letters_info = ensure_cubes("AT", universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self.cube_07 = SemanticCube(self._mojo, xml_path=str(digits_info["1"]["xml"]), labels=digits_info["1"]["labels"])
        self.cube_08 = SemanticCube(self._mojo, xml_path=str(digits_info["2"]["xml"]), labels=digits_info["2"]["labels"])

        self.cube_09 = SemanticCube(self._mojo, xml_path=str(letters_info["A"]["xml"]), labels=letters_info["A"]["labels"])
        self.cube_10 = SemanticCube(self._mojo, xml_path=str(letters_info["T"]["xml"]), labels=letters_info["T"]["labels"])

        # Convenience lists
        self.colored_cubes = [self.cube_01, self.cube_02, self.cube_03, self.cube_04]
        self.semantic_cubes = [self.cube_07, self.cube_08, self.cube_09, self.cube_10]
        self.cubes = self.colored_cubes + self.semantic_cubes

        # Stack definitions (bottom->top)
        self._stacks = [
            {
                "stack_id": "stack_color_gradient",
                "base_xy": np.array([0.55 - 0.12, -0.12]),
                "objs": [self.cube_01, self.cube_02],
                "kind": "colored",
                "expected_colors": ["blue", "green"],
            },
            {
                "stack_id": "stack_numbers_ascending",
                "base_xy": np.array([0.55 - 0.12, 0.12]),
                "objs": [self.cube_07, self.cube_08],
                "kind": "semantic",
                "expected_labels": ["1", "2"],
            },
            {
                "stack_id": "stack_word",
                "base_xy": np.array([0.55 + 0.12, -0.12]),
                "objs": [self.cube_09, self.cube_10],
                "kind": "semantic",
                "expected_labels": ["A", "T"],
            },
            {
                "stack_id": "stack_single_color",
                "base_xy": np.array([0.55 + 0.12, 0.12]),
                "objs": [self.cube_03, self.cube_04],
                "kind": "colored",
                "expected_colors": ["red", "red"],
            },
        ]

        # Precompute goal poses
        self._goal_pos = {}
        for stack in self._stacks:
            base_xy = stack["base_xy"]
            for level, obj in enumerate(stack["objs"]):
                pos = np.array([base_xy[0], base_xy[1], self._Z_BOTTOM + level * self._Z_STEP], dtype=float)
                self._goal_pos[obj] = pos

        # Initialize metrics (FULL signature required)
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

        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset for all ColoredCubes
        self.cube_01.set_color("blue")
        self.cube_02.set_color("green")
        self.cube_03.set_color("red")
        self.cube_04.set_color("red")

        # Random UNSOLVED placement within bounds:
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = self._Z_BOTTOM

        # Sample non-overlapping XY points for all cubes using the correct API signature:
        # get_random_points_on_plane(amount, origin, extents, spacing)
        min_dist = 2.25 * self._CUBE_SIZE
        origin = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, float(z_base)], dtype=float)
        extents = np.array([(x_max - x_min) * 0.5, (y_max - y_min) * 0.5, 0.0], dtype=float)
        pts = get_random_points_on_plane(len(self.cubes), origin, extents, float(min_dist))

        def _rand_upright_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.w, q.x, q.y, q.z], dtype=float)

        def _semantic_back_readable_quat(sem_cube: SemanticCube, glyph: str):
            face_dir_world = [-1.0, 0.0, 0.0]
            face_idx = sem_cube._labels.index(glyph) if hasattr(sem_cube, "_labels") and glyph in sem_cube._labels else 0
            q = sem_cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=face_dir_world)
            return np.array(q.elements, dtype=float)

        # Place all cubes at random XY, fixed Z base
        for obj, pt in zip(self.cubes, pts):
            x = float(pt[0])
            y = float(pt[1])
            pos = np.array([x, y, float(z_base)], dtype=float)

            if obj in self.semantic_cubes:
                glyph = obj.get_label()
                quat = _semantic_back_readable_quat(obj, glyph)
            else:
                quat = _rand_upright_quat()

            obj.set_pose(position=pos, quat=quat)

        # Let physics settle (required pattern)
        for _ in range(50):
            self._mojo.step()

        # Hard guarantee: ensure not all cubes are at their goal positions
        all_close = True
        for obj in self.cubes:
            gp = self._goal_pos.get(obj, None)
            if gp is None:
                continue
            p = obj.body.get_position()
            if float(np.linalg.norm(p - gp)) > 0.01:
                all_close = False
                break

        if all_close and len(self.cubes) > 0:
            obj = self.cubes[0]
            p = obj.body.get_position()
            new_x = float(np.clip(p[0] + 0.06, x_min, x_max))
            new_y = float(np.clip(p[1] + 0.06, y_min, y_max))
            if obj in self.semantic_cubes:
                glyph = obj.get_label()
                quat = _semantic_back_readable_quat(obj, glyph)
            else:
                quat = _rand_upright_quat()
            obj.set_pose(position=np.array([new_x, new_y, float(z_base)], dtype=float), quat=quat)
            for _ in range(50):
                self._mojo.step()

        # Initialize subgoals to False
        for i in range(self._NUM_SUBGOALS):
            self._metric_stage(i, False)

        # --- GOAL STATE ---
        # rng = np.random.default_rng(self.seed)
        #
        # # REQUIRED: set_color() in _on_reset for all ColoredCubes
        # self.cube_01.set_color("blue")
        # self.cube_02.set_color("green")
        # self.cube_03.set_color("yellow")
        # self.cube_04.set_color("red")
        # self.cube_05.set_color("red")
        # self.cube_06.set_color("red")
        #
        # identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # _face_dir_world = np.array([-1.0, 0.0, 0.0], dtype=float)
        #
        # # Bottom layer placement
        # for stack in self._stacks:
        #     obj = stack["objs"][0]
        #     pos = self._goal_pos[obj]
        #     if stack["kind"] == "semantic":
        #         target = stack["expected_labels"][0]
        #         face_idx = obj._labels.index(target)
        #         q = obj.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=_face_dir_world)
        #         obj.set_pose(position=pos, quat=q.elements)
        #     else:
        #         obj.set_pose(position=pos, quat=identity)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Middle layer placement
        # for stack in self._stacks:
        #     obj = stack["objs"][1]
        #     pos = self._goal_pos[obj]
        #     if stack["kind"] == "semantic":
        #         target = stack["expected_labels"][1]
        #         face_idx = obj._labels.index(target)
        #         q = obj.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=_face_dir_world)
        #         obj.set_pose(position=pos, quat=q.elements)
        #     else:
        #         obj.set_pose(position=pos, quat=identity)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Top layer placement
        # for stack in self._stacks:
        #     obj = stack["objs"][2]
        #     pos = self._goal_pos[obj]
        #     if stack["kind"] == "semantic":
        #         target = stack["expected_labels"][2]
        #         face_idx = obj._labels.index(target)
        #         q = obj.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=_face_dir_world)
        #         obj.set_pose(position=pos, quat=q.elements)
        #     else:
        #         obj.set_pose(position=pos, quat=identity)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Initialize subgoals to False
        # for i in range(self._NUM_SUBGOALS):
        #     self._metric_stage(i, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _close_xyz(self, obj, target_pos: np.ndarray, xy_tol: float, z_tol: float) -> bool:
        p = obj.body.get_position()
        dx = float(p[0] - target_pos[0])
        dy = float(p[1] - target_pos[1])
        dz = float(p[2] - target_pos[2])
        return (abs(dx) <= xy_tol) and (abs(dy) <= xy_tol) and (abs(dz) <= z_tol)

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_stacks"):
            return False

        positions_ok = True
        colored_ok = True
        semantic_ok = True

        pose_error: Dict[str, float] = {}
        target_distance: Dict[str, float] = {}

        # Per-stack translation-invariant checks: each stack is defined by the identity
        # of its two cubes. We verify support contacts, height, and ordering
        # without referencing a fixed world base_xy.
        stack_centroids = []  # XY centroid of each stack (for mutual-separation check)

        for stack in self._stacks:
            stack_objs_by_z = sorted(stack["objs"], key=lambda o: o.body.get_position()[2])
            positions = [np.array(o.body.get_position(), dtype=float) for o in stack_objs_by_z]

            stack_centroids.append(np.mean([p[:2] for p in positions], axis=0))

            # Bottom cube must rest on the table (contact-based, no absolute Z).
            if not stack_objs_by_z[0].is_colliding(self.table.table_top):
                positions_ok = False

            # Middle / top cubes must not touch the table and must contact the cube below.
            for level in range(1, len(stack_objs_by_z)):
                upper = stack_objs_by_z[level]
                lower = stack_objs_by_z[level - 1]
                if upper.is_colliding(self.table.table_top):
                    positions_ok = False
                if not upper.is_colliding(lower):
                    positions_ok = False

            # Z monotonicity with approximately one cube-size step between levels.
            for level in range(1, len(positions)):
                dz = float(positions[level][2] - positions[level - 1][2])
                if dz < (self._Z_STEP - self._Z_TOL) or dz > (self._Z_STEP + self._Z_TOL):
                    positions_ok = False

            # Content ordering (bottom -> top).
            if stack["kind"] == "colored":
                for obj, exp_c in zip(stack_objs_by_z, stack["expected_colors"]):
                    if obj.get_color() != exp_c:
                        colored_ok = False
                        break
            else:
                for obj, exp_l in zip(stack_objs_by_z, stack["expected_labels"]):
                    if not obj.is_face_readable(exp_l, face_dir=self._FACE_DIR):
                        semantic_ok = False
                        break

        # Mutual separation between stacks.
        for i in range(len(stack_centroids)):
            for j in range(i + 1, len(stack_centroids)):
                dxy = float(np.linalg.norm(stack_centroids[i] - stack_centroids[j]))
                target_distance[
                    f"{self._stacks[i]['stack_id']}__to__{self._stacks[j]['stack_id']}_xy"
                ] = dxy
                if dxy < self._STACK_SEPARATION:
                    positions_ok = False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        all_ok = bool(positions_ok and colored_ok and semantic_ok)
        if all_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(all_ok and released)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
