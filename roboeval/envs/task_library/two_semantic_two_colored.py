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


class AlternatingCubeRowArrange(RoboEvalEnv, MetricRolloutEval):
    """Make a straight four-cube row alternating semantic, colored, semantic, colored.

Success Criteria:
- The four cubes form a straight row with nearly equal X positions and left-to-right Y order semantic, colored, semantic, colored.
- Adjacent row gaps are within the configured spacing range, all cubes are near table height, and no cubes overlap.
- The semantic slots show A and B readable on the required face direction, and no gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Flat horizontal arrangement: use TOP face readability for SemanticCube.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Geometry / tolerances from task spec + mandatory min spacing constraints.
    _CENTER_X = 0.55
    _CENTER_Y = 0.0
    _Z_ON_TABLE = 0.97

    _MAX_LATERAL_DEVIATION_M = 0.03  # deviation from row axis (X scatter)
    # The mandatory minimum pairwise distance (0.0966) is ABOVE the old target of
    # 0.07, so clamping the lower bound to it collapsed the intended +/-3 cm window
    # into [0.0966, 0.10] -- 3.4 mm wide, and three adjacent gaps had to land in it
    # at once. Target a spacing that clears the mandatory minimum on its own so the
    # tolerance is the real tolerance.
    _TARGET_SPACING_M = 0.12
    _SPACING_TOLERANCE_M = 0.03
    _SPACING_MIN_M = max(_TARGET_SPACING_M - _SPACING_TOLERANCE_M, 0.0966)
    _SPACING_MAX_M = _TARGET_SPACING_M + _SPACING_TOLERANCE_M
    _MIN_CLEARANCE_M = 0.01

    def _initialize_env(self):
        # Props (Table exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Semantic cubes: MUST use ensure_cubes() workflow ---
        # Pick two deterministic labels (content must be verifiable).
        self._target_labels = ["A", "B"]
        cubes_info = ensure_cubes("".join(self._target_labels), universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        info_a = cubes_info[self._target_labels[0]]
        info_b = cubes_info[self._target_labels[1]]

        self.semantic_cube_1 = SemanticCube(self._mojo, xml_path=str(info_a["xml"]), labels=info_a["labels"])
        self.semantic_cube_2 = SemanticCube(self._mojo, xml_path=str(info_b["xml"]), labels=info_b["labels"])
        self.semantic_cubes = [self.semantic_cube_1, self.semantic_cube_2]

        # --- Colored cubes ---
        self.colored_cube_1 = ColoredCube(self._mojo)
        self.colored_cube_2 = ColoredCube(self._mojo)
        self._colored_colors = ["red", "blue"]

        # REQUIRED: call set_color() in BOTH _initialize_env() AND _on_reset()
        self.colored_cube_1.set_color(self._colored_colors[0])
        self.colored_cube_2.set_color(self._colored_colors[1])

        # Canonical object order (left->right target pattern):
        # semantic, colored, semantic, colored
        self.cubes = [self.semantic_cube_1, self.colored_cube_1, self.semantic_cube_2, self.colored_cube_2]

        # Goal row (along Y axis; left-to-right is +Y -> -Y in camera view).
        # Choose spacing within [0.0966, 0.10] to satisfy both the task tolerance and mandatory spacing.
        self._goal_spacing = 0.099
        start_y = self._CENTER_Y + 1.5 * self._goal_spacing
        ys = [start_y - i * self._goal_spacing for i in range(4)]
        self._goal_positions = [
            np.array([self._CENTER_X, ys[0], self._Z_ON_TABLE]),
            np.array([self._CENTER_X, ys[1], self._Z_ON_TABLE]),
            np.array([self._CENTER_X, ys[2], self._Z_ON_TABLE]),
            np.array([self._CENTER_X, ys[3], self._Z_ON_TABLE]),
        ]

        # Store target quaternions for pose error metrics (set in _on_reset()).
        self._goal_quats = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(4)]

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

        # 2. Deterministic RNG (pipeline may add randomization later)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() again on reset
        self.colored_cube_1.set_color(self._colored_colors[0])
        self.colored_cube_2.set_color(self._colored_colors[1])

        # 4. Set GOAL quaternions (targets) for pose error metrics (do NOT place objects at goal)
        # Semantic cube targets: readable target labels on TOP face.
        for sem_i, (cube, target_label) in enumerate(zip(self.semantic_cubes, self._target_labels)):
            face_idx = cube._labels.index(target_label)
            q_goal = cube.get_orientation_for_face_and_glyph(face_idx=face_idx, face_dir_world=self._FACE_DIR)
            self._goal_quats[0 if sem_i == 0 else 2] = np.array(q_goal.elements, dtype=float)

        # Colored cube targets: identity, in the [w, x, y, z] order that
        # body.get_quaternion() returns and that the semantic goals above already use.
        # Written as [0, 0, 0, 1] this was a 180 deg turn about Z, so the angular
        # pose_error for a perfectly placed colored cube came out at pi rad.
        self._goal_quats[1] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._goal_quats[3] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # 5. Place objects in RANDOM, UNSOLVED positions within workspace bounds
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._Z_ON_TABLE

        # Ensure cubes are well-separated and not near their respective goal positions
        min_pairwise = 0.11  # >= mandatory 0.0966 plus margin
        min_from_own_goal = 0.08  # ensures per-object close check (<=0.03) fails
        max_tries = 2000

        positions = []
        for i in range(4):
            for _ in range(max_tries):
                p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)

                # far enough from its own goal location
                if np.linalg.norm(p - self._goal_positions[i]) < min_from_own_goal:
                    continue

                # pairwise separation
                ok = True
                for pj in positions:
                    if np.linalg.norm(p - pj) < min_pairwise:
                        ok = False
                        break
                if not ok:
                    continue

                positions.append(p)
                break
            else:
                # Fallback (should be rare): spread along Y away from goal row, still within bounds.
                base_y = y_hi - 0.02 - i * 0.08
                base_y = float(np.clip(base_y, y_lo + 0.02, y_hi - 0.02))
                p = np.array([rng.uniform(x_lo, x_hi), base_y, z], dtype=float)
                positions.append(p)

        # Random orientations (intentionally NOT matching goal, and for semantic cubes tilt off-top to break readability)
        def random_quat_tilted() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            tilt = float(rng.uniform(np.pi / 4, 3 * np.pi / 4))  # strong tilt so top-face readability likely fails
            q = (Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw) * Quaternion(axis=[1.0, 0.0, 0.0], angle=tilt)).elements
            q = np.array(q, dtype=float)
            q = q / (np.linalg.norm(q) + 1e-12)
            return q

        def random_quat_any() -> np.ndarray:
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            ang = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=axis, angle=ang).elements
            q = np.array(q, dtype=float)
            q = q / (np.linalg.norm(q) + 1e-12)
            return q

        # Apply poses
        self.semantic_cube_1.set_pose(position=positions[0], quat=random_quat_tilted())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.colored_cube_1.set_pose(position=positions[1], quat=random_quat_any())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.semantic_cube_2.set_pose(position=positions[2], quat=random_quat_tilted())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.colored_cube_2.set_pose(position=positions[3], quat=random_quat_any())
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # 6. Physics settling
        self._mojo.physics.forward()

        # 7. Initialize subgoals to False (8 subgoals)
        for idx in range(8):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or len(getattr(self, "cubes", [])) != 4:
            return False

        # Current poses
        positions = [c.body.get_position() for c in self.cubes]
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        zs = np.array([p[2] for p in positions])

        # --- Row straightness (aligned along table_y; i.e., same X, varying Y) ---
        mean_x = float(np.mean(xs))
        lateral_ok = bool(np.all(np.abs(xs - mean_x) <= self._MAX_LATERAL_DEVIATION_M))

        # --- Identity-free row check ---
        # Sort all cubes by Y desc to get slot[0..3] in left-to-right order.
        # Expected pattern: semantic(A), colored(red), semantic(B), colored(blue).
        # Distinguish semantic vs colored by membership in self.semantic_cubes.
        sem_set = set(id(c) for c in self.semantic_cubes)
        cubes_with_pos = [(c, c.body.get_position()) for c in self.cubes]
        cubes_sorted = sorted(cubes_with_pos, key=lambda cp: cp[1][1], reverse=True)
        slot_cubes = [cp[0] for cp in cubes_sorted]
        slot_positions = [cp[1] for cp in cubes_sorted]

        # Slot-based type pattern check (semantic, colored, semantic, colored).
        slot_is_semantic = [id(c) in sem_set for c in slot_cubes]
        type_pattern_ok = slot_is_semantic == [True, False, True, False]

        y_s1 = float(slot_positions[0][1])
        y_c1 = float(slot_positions[1][1])
        y_s2 = float(slot_positions[2][1])
        y_c2 = float(slot_positions[3][1])

        order_ok = bool(
            (y_s1 >= y_c1 + self._MIN_CLEARANCE_M)
            and (y_c1 >= y_s2 + self._MIN_CLEARANCE_M)
            and (y_s2 >= y_c2 + self._MIN_CLEARANCE_M)
        )

        # --- Spacing (approximately even) between adjacent cubes in the intended order ---
        d1 = y_s1 - y_c1
        d2 = y_c1 - y_s2
        d3 = y_s2 - y_c2
        spacing_min = self._SPACING_MIN_M
        spacing_max = self._SPACING_MAX_M + 1e-6
        spacing_ok = bool(
            (d1 >= spacing_min and d1 <= spacing_max)
            and (d2 >= spacing_min and d2 <= spacing_max)
            and (d3 >= spacing_min and d3 <= spacing_max)
        )

        # --- Table contact sanity (must be on table height-ish) ---
        z_ok = bool(np.all(np.abs(zs - self._Z_ON_TABLE) <= 0.03))

        # --- No overlaps / collisions among cubes ---
        no_overlap = True
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                if self.cubes[i].is_colliding(self.cubes[j]):
                    no_overlap = False
                    break
            if not no_overlap:
                break

        positions_ok = bool(lateral_ok and order_ok and spacing_ok and z_ok and no_overlap and type_pattern_ok)

        # --- Semantic verification: leftmost semantic slot shows "A", second semantic slot shows "B" ---
        semantic_ok = True
        if type_pattern_ok:
            if not slot_cubes[0].is_face_readable(self._target_labels[0], face_dir=self._FACE_DIR):
                semantic_ok = False
            if not slot_cubes[2].is_face_readable(self._target_labels[1], face_dir=self._FACE_DIR):
                semantic_ok = False
        else:
            semantic_ok = False

        # --- Gripper release check ---
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (place-based proxy for grasp/place sequence) ---
        # We treat each "place" subgoal as achieved when the cube is close to its target pose.
        goal_pos = self._goal_positions
        per_obj_close = []
        for i, cube in enumerate(self.cubes):
            p = cube.body.get_position()
            per_obj_close.append(bool(np.linalg.norm(p - goal_pos[i]) <= 0.03))

        # Map: (grasp, place) for each object -> mark both when placed (proxy)
        # semantic1: stages 0,1; colored1: 2,3; semantic2: 4,5; colored2: 6,7
        stage_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for obj_i, (a, b) in enumerate(stage_pairs):
            if per_obj_close[obj_i]:
                self._metric_stage(a, True)
                self._metric_stage(b, True)

        # Success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize ---
        target_distance = {}
        pose_error = {}

        for i, cube in enumerate(self.cubes):
            p = cube.body.get_position()
            q = cube.body.get_quaternion()
            td = float(np.linalg.norm(p - self._goal_positions[i]))
            target_distance[f"cube_{i}"] = td

            # Quaternion angular distance (robust to q/-q)
            q = np.array(q, dtype=float)
            qt = np.array(self._goal_quats[i], dtype=float)
            q = q / (np.linalg.norm(q) + 1e-12)
            qt = qt / (np.linalg.norm(qt) + 1e-12)
            dot = float(np.clip(abs(np.dot(q, qt)), -1.0, 1.0))
            ang = float(2.0 * np.arccos(dot))
            pose_error[f"cube_{i}"] = ang

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})