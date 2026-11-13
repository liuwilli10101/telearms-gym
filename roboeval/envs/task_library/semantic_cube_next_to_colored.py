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

class AdjacentRedCubePlacementTask(RoboEvalEnv, MetricRolloutEval):
    """
    Move the semantic cube so it is placed directly next to the red colored cube with one
    pair of side faces touching (no overlap). The final arrangement should have the two
    cubes adjacent without overlap. The semantic cube must also end resting flat with its
    'A' face pointing UP (readable from above); success requires both the side-face
    adjacency AND the 'A' face being up, with the red cube actually red and nothing held.

    Sequential Sub-Goals:
        1. Securely grasp the semantic cube with either gripper.
        2. Lift the semantic cube off the table to allow collision-free transport.
        3. Move the semantic cube near the red colored cube while avoiding collisions.
        4. Align the semantic cube so a side face can make flush contact with a side face of the red cube.
        5. Place the semantic cube on the table adjacent to the red cube so their sides touch.
        6. Release the semantic cube while keeping the cubes in touching adjacency.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # For a flat (tabletop) arrangement, the top face is reliably visible.
    _FACE_DIR = np.array([0.0, 0.0, 1.0])

    # Geometry/thresholds (meters)
    _CUBE_SIZE = 0.04
    _MAX_PENETRATION = 0.002
    _SUCCESS_TOL = 0.014
    _MAX_CENTER_OFFSET_ORTHO = 0.02
    _MAX_Z_OFFSET = 0.02

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY those allowed) ---
        # ColoredCube (red)
        self.colored_cube_red = ColoredCube(self._mojo)
        self.colored_cube_red.set_color("red")  # REQUIRED in _initialize_env

        # SemanticCube (one cube). We require semantic verification (label readability).
        self._target_label = "A"
        cubes_info = ensure_cubes(self._target_label, universe="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        info = cubes_info[self._target_label]
        self.semantic_cube_1 = SemanticCube(
            self._mojo,
            xml_path=str(info["xml"]),
            labels=info["labels"],
        )

        # Convenience list for metrics and generic loops
        self.cubes = [self.semantic_cube_1, self.colored_cube_red]

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

        # REQUIRED: set_color() in _on_reset for ColoredCube
        self.colored_cube_red.set_color("red")

        # ---------------------------------------------------------------------
        # Place objects in RANDOM UNSOLVED configuration (ensure _success() False)
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # ---------------------------------------------------------------------
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z_base = 0.97

        # Sample positions such that cubes are not adjacent-touching by a side face.
        # (Keep them well separated in XY to robustly fail adjacency checks.)
        min_xy_sep = 0.12  # > 0.10 "near" threshold and far from 0.04 adjacency target

        red_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
        for _ in range(200):
            sem_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if float(np.linalg.norm(sem_xy - red_xy)) >= min_xy_sep:
                break
        else:
            # Fallback deterministic separation if sampling is unlucky
            sem_xy = np.array([np.clip(red_xy[0] + 0.15, x_lo, x_hi), np.clip(red_xy[1] + 0.15, y_lo, y_hi)])

        red_pos = np.array([red_xy[0], red_xy[1], z_base], dtype=float)
        sem_pos = np.array([sem_xy[0], sem_xy[1], z_base], dtype=float)

        # Orient semantic cube so target label is readable on the TOP face.
        face_idx = self.semantic_cube_1._labels.index(self._target_label)
        q = self.semantic_cube_1.get_orientation_for_face_and_glyph(
            face_idx=face_idx,
            face_dir_world=self._FACE_DIR,
        )

        # Apply poses (and forward kinematics if available)
        self.colored_cube_red.set_pose(position=red_pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        self.semantic_cube_1.set_pose(position=sem_pos, quat=q.elements)
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

        # Physics settling
        self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _adjacent_touching_side_face(self, a_pos: np.ndarray, b_pos: np.ndarray):
        """
        Returns (ok, touch_axis, orth_offset, z_offset, axis_sep).
          - touch_axis: 'x' or 'y' (which axis is used for side-face contact)
          - orth_offset: offset orthogonal to touch axis (XY plane)
          - z_offset: absolute z difference
          - axis_sep: absolute separation along touch axis
        """
        d = a_pos - b_pos
        dx, dy, dz = float(d[0]), float(d[1]), float(d[2])

        ax, ay, az = abs(dx), abs(dy), abs(dz)

        # Side-face touching occurs if either X or Y separation is ~ cube_size,
        # while the orthogonal separation is small, and z levels are similar.
        def _check(axis_sep: float, orth_sep: float):
            # No overlap: separation along touch axis should not be less than (size - max_penetration)
            no_overlap = axis_sep >= (self._CUBE_SIZE - self._MAX_PENETRATION)
            close_to_face_contact = abs(axis_sep - self._CUBE_SIZE) <= self._SUCCESS_TOL
            orth_ok = orth_sep <= self._MAX_CENTER_OFFSET_ORTHO
            z_ok = az <= self._MAX_Z_OFFSET
            ok = bool(no_overlap and close_to_face_contact and orth_ok and z_ok)
            return ok

        ok_x = _check(ax, ay)
        ok_y = _check(ay, ax)

        if ok_x and not ok_y:
            return True, "x", ay, az, ax
        if ok_y and not ok_x:
            return True, "y", ax, az, ay
        if ok_x and ok_y:
            # If both pass (rare), prefer the axis closer to exact 0.04
            if abs(ax - self._CUBE_SIZE) <= abs(ay - self._CUBE_SIZE):
                return True, "x", ay, az, ax
            return True, "y", ax, az, ay

        # Not adjacent-touching by side face
        # Return best-effort diagnostics
        if ax >= ay:
            return False, "x", ay, az, ax
        return False, "y", ax, az, ay

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        sem = self.semantic_cube_1
        red = self.colored_cube_red

        sem_pos = sem.body.get_position()
        red_pos = red.body.get_position()

        # Spatial goal: adjacent_touching (side-face), no overlap, small orth offset
        positions_ok, touch_axis, orth_offset, z_offset, axis_sep = self._adjacent_touching_side_face(
            sem_pos, red_pos
        )

        # Verify red cube is actually red (task refers to "red colored cube")
        color_ok = (hasattr(red, "get_color") and red.get_color() == "red")

        # REQUIRED: Semantic verification for SemanticCube
        semantic_ok = bool(sem.is_face_readable(self._target_label, face_dir=self._FACE_DIR))

        # Released check (generic grippers iteration)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort state-based) ---
        held_sem = any(self.robot.is_gripper_holding_object(sem, s) for s in self.robot.grippers)
        lifted_sem = (float(sem_pos[2]) > 0.99) and (not sem.is_colliding(self.table.table_top))
        near_red = float(np.linalg.norm((sem_pos - red_pos)[:2])) < 0.10
        aligned = bool((orth_offset <= self._MAX_CENTER_OFFSET_ORTHO) and (z_offset <= self._MAX_Z_OFFSET))
        placed_adjacent = bool(positions_ok and sem.is_colliding(self.table.table_top))
        released_after_place = bool(placed_adjacent and released)

        self._metric_stage(0, bool(held_sem))
        self._metric_stage(1, bool(lifted_sem))
        self._metric_stage(2, bool(near_red))
        self._metric_stage(3, bool(aligned))
        self._metric_stage(4, bool(placed_adjacent))
        self._metric_stage(5, bool(released_after_place))

        self._success_check = bool(positions_ok and color_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        target_distance = {
            "axis_face_contact_error": float(abs(axis_sep - self._CUBE_SIZE)),
            "orthogonal_offset": float(orth_offset),
            "z_offset": float(z_offset),
        }
        pose_error = {
            "touch_axis": 0.0 if positions_ok else 1.0,  # simple scalar placeholder
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})