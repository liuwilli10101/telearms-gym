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

class DiagonalCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes onto the square green goal patch and arrange them
    in a straight diagonal line from one corner of the patch to the opposite corner.

    Sequential Sub-Goals:
        1. Grasp any one cube with the left gripper.
        2. Grasp any one cube with the right gripper.
        3. Move the red cube over the goal patch region.
        4. Place the red cube onto the goal patch (approximate diagonal endpoint).
        5. Move the blue cube over the goal patch region.
        6. Place the blue cube onto the goal patch (approximate middle of diagonal).
        7. Move the yellow cube over the goal patch region.
        8. Place the yellow cube onto the goal patch (approximate diagonal endpoint).
        9. Adjust cube positions so the three cubes form a straight diagonal line across the patch.
        10. Release any held cube after placing/alignment adjustments (left gripper).
        11. Release any held cube after placing/alignment adjustments (right gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal tolerances from task spec
    _MAX_PERP_DEV_M = 0.02
    _MIN_ADJ_SPACING_M = 0.03
    _MAX_CORNER_DIST_M = 0.04

    # Goal patch specification (must match required instantiation pattern)
    _PATCH_SIZE = (0.1, 0.1)

    # Resting heights (MANDATORY)
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    _NUM_SUBGOALS = 11

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Assets (ONLY allowed assets: ColoredCube, GoalPatch)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)
        self.cubes = [self.cube_red, self.cube_blue, self.cube_yellow]

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        self.goal_patch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE, name="target")

        # Cache goal patch half extents (axis-aligned patch)
        self._patch_half = np.array([self._PATCH_SIZE[0] / 2.0, self._PATCH_SIZE[1] / 2.0], dtype=float)

        # Storage for a nominal goal pose set at reset (used for metrics)
        self._goal_positions = {
            "cube_red": None,
            "cube_blue": None,
            "cube_yellow": None,
        }
        self._goal_diag_dir = None  # unit vector in XY for chosen diagonal

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
        # Some builds expose physics.forward(); keep it defensive.
        if hasattr(self._mojo, "physics") and hasattr(self._mojo.physics, "forward"):
            self._mojo.physics.forward()

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # 3. Place goal patch at a fixed, valid location (cubes will be randomized unsolved)
        patch_center = np.array([0.55, 0.0, self._PATCH_Z], dtype=float)
        self.goal_patch.body.set_position(patch_center)
        self._mojo.physics.forward()
        self._physics_forward()
        self._mojo.physics.forward()

        # REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        # Randomize cube poses in unsolved positions within workspace bounds.
        # Ensure they start OFF the patch so _success() is False immediately after reset.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_Z)

        # Keep a margin away from the patch footprint so they do not collide with the patch collider.
        # Patch half-extent is 0.05; add extra margin.
        off_patch_margin = float(max(self._patch_half) + 0.06)  # >= 0.11m away in XY from center

        # Avoid cube-cube overlap at start
        min_xy_sep = 0.06

        def sample_off_patch_xy(existing_xy: list[np.ndarray]) -> np.ndarray:
            for _ in range(2000):
                xy = np.array(
                    [float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))],
                    dtype=float,
                )
                # Off-patch constraint (outside an inflated patch AABB)
                dx = abs(float(xy[0] - patch_center[0]))
                dy = abs(float(xy[1] - patch_center[1]))
                if (dx <= off_patch_margin) and (dy <= off_patch_margin):
                    continue
                # Separation constraint
                if any(np.linalg.norm(xy - e) < min_xy_sep for e in existing_xy):
                    continue
                return xy
            # Fallback: place far in workspace corners (still within bounds)
            corners = [
                np.array([x_min, y_min], dtype=float),
                np.array([x_min, y_max], dtype=float),
                np.array([x_max, y_min], dtype=float),
                np.array([x_max, y_max], dtype=float),
            ]
            for c in corners:
                dx = abs(float(c[0] - patch_center[0]))
                dy = abs(float(c[1] - patch_center[1]))
                if (dx > off_patch_margin) or (dy > off_patch_margin):
                    if all(np.linalg.norm(c - e) >= min_xy_sep for e in existing_xy):
                        return c
            return np.array([x_min, y_min], dtype=float)

        existing: list[np.ndarray] = []
        red_xy = sample_off_patch_xy(existing)
        existing.append(red_xy)
        blue_xy = sample_off_patch_xy(existing)
        existing.append(blue_xy)
        yellow_xy = sample_off_patch_xy(existing)
        existing.append(yellow_xy)

        def rand_quat_z() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0, 0, 1], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        red_pos = np.array([red_xy[0], red_xy[1], z], dtype=float)
        blue_pos = np.array([blue_xy[0], blue_xy[1], z], dtype=float)
        yellow_pos = np.array([yellow_xy[0], yellow_xy[1], z], dtype=float)

        self.cube_red.set_pose(position=red_pos, quat=rand_quat_z())
        self._physics_forward()
        self._mojo.physics.forward()

        self.cube_blue.set_pose(position=blue_pos, quat=rand_quat_z())
        self._physics_forward()
        self._mojo.physics.forward()

        self.cube_yellow.set_pose(position=yellow_pos, quat=rand_quat_z())
        self._physics_forward()
        self._mojo.physics.forward()

        # Store nominal "goal" positions for metrics (not used as success targets here)
        self._goal_positions["cube_red"] = red_pos.copy()
        self._goal_positions["cube_blue"] = blue_pos.copy()
        self._goal_positions["cube_yellow"] = yellow_pos.copy()
        self._goal_diag_dir = None

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_block_on_patch(self, block: ColoredCube, patch: GoalPatch) -> bool:
        # Prefer the thin collision geometry via patch.collider when available; fall back to patch itself.
        target = patch.collider if hasattr(patch, "collider") else patch
        if not block.is_colliding(target):
            return False
        # If an explicit floor exists, avoid counting floor contact as "on patch"
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False
        return True

    def _diagonal_line_check(self, cube_xy: np.ndarray, patch_xy: np.ndarray) -> Dict:
        """
        Evaluate whether all cube XY positions lie on a single patch diagonal, with:
          - max perpendicular deviation <= _MAX_PERP_DEV_M
          - min adjacent spacing along the diagonal >= _MIN_ADJ_SPACING_M
          - endpoints near opposite corners (<= _MAX_CORNER_DIST_M)
        Returns dict with fields: ok (bool), best_diag (int), metrics (dict).
        """
        # Two candidate diagonals through patch center:
        diags = [
            (np.array([1.0, 1.0], dtype=float) / np.sqrt(2.0), 0),
            (np.array([1.0, -1.0], dtype=float) / np.sqrt(2.0), 1),
        ]

        best = None
        for d, diag_id in diags:
            rel = cube_xy - patch_xy[None, :]
            t = rel @ d  # projection scalars along diagonal
            perp = rel - t[:, None] * d[None, :]
            perp_dists = np.linalg.norm(perp, axis=1)
            max_perp = float(np.max(perp_dists))

            # Spacing along diagonal: sort by projection coordinate
            order = np.argsort(t)
            t_sorted = t[order]
            spacings = np.diff(t_sorted)
            min_spacing = float(np.min(spacings)) if spacings.size > 0 else 0.0

            # Determine endpoint cubes and corner anchoring
            # Patch corners for each diagonal (square, axis-aligned):
            hx, hy = float(self._patch_half[0]), float(self._patch_half[1])
            if diag_id == 0:
                c1 = patch_xy + np.array([hx, hy], dtype=float)
                c2 = patch_xy + np.array([-hx, -hy], dtype=float)
            else:
                c1 = patch_xy + np.array([hx, -hy], dtype=float)
                c2 = patch_xy + np.array([-hx, hy], dtype=float)

            end_lo = cube_xy[order[0]]
            end_hi = cube_xy[order[-1]]

            # Any corner can be endpoint 1; need endpoints near opposite corners.
            # Check both assignments and take the better one.
            assign1 = max(float(np.linalg.norm(end_lo - c1)), float(np.linalg.norm(end_hi - c2)))
            assign2 = max(float(np.linalg.norm(end_lo - c2)), float(np.linalg.norm(end_hi - c1)))
            endpoint_corner_maxdist = min(assign1, assign2)

            # Middle cube should be between endpoints (always true by ordering), track its centeredness
            middle_t = float(t_sorted[1]) if t_sorted.size == 3 else float(np.median(t_sorted))
            span = float(t_sorted[-1] - t_sorted[0]) if t_sorted.size > 1 else 0.0
            middle_offset_from_center = abs(middle_t - (t_sorted[0] + 0.5 * span)) if span > 1e-9 else 0.0

            ok = (
                (max_perp <= self._MAX_PERP_DEV_M)
                and (min_spacing >= self._MIN_ADJ_SPACING_M)
                and (endpoint_corner_maxdist <= self._MAX_CORNER_DIST_M)
            )

            metrics = {
                "diag_id": diag_id,
                "max_perp_dev_m": max_perp,
                "min_adj_spacing_m": min_spacing,
                "endpoint_corner_maxdist_m": endpoint_corner_maxdist,
                "middle_offset_from_span_center_m": float(middle_offset_from_center),
            }

            candidate = {"ok": bool(ok), "best_diag": diag_id, "metrics": metrics}
            if best is None:
                best = candidate
            else:
                # Prefer an "ok" diagonal; otherwise prefer smaller max perpendicular deviation
                if (not best["ok"] and candidate["ok"]) or (
                    (best["ok"] == candidate["ok"])
                    and (candidate["metrics"]["max_perp_dev_m"] < best["metrics"]["max_perp_dev_m"])
                ):
                    best = candidate

        return best if best is not None else {"ok": False, "best_diag": -1, "metrics": {}}

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch"):
            return False

        # 1) All three cubes are on the green goal patch
        on_patch = [self._is_block_on_patch(c, self.goal_patch) for c in self.cubes]
        on_patch_all = bool(all(on_patch))

        # 2) Diagonal arrangement across the patch (order-free collinearity)
        patch_xy = self.goal_patch.body.get_position()[:2]
        cube_xy = np.stack([c.body.get_position()[:2] for c in self.cubes], axis=0)
        diag_eval = self._diagonal_line_check(cube_xy=cube_xy, patch_xy=patch_xy)
        arrangement_ok = bool(diag_eval.get("ok", False))

        positions_ok = bool(on_patch_all and arrangement_ok)
        semantic_ok = True  # no semantic labels in this task

        # 3) Released (not holding any cube)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # Subgoal staging (coarse): mark all done only when fully successful
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        # Provide per-cube distances to patch center and best-fit diagonal stats.
        patch_pos = self.goal_patch.body.get_position()
        target_distance = {
            "cube_red_to_patch_center_m": float(np.linalg.norm(self.cube_red.body.get_position() - patch_pos)),
            "cube_blue_to_patch_center_m": float(np.linalg.norm(self.cube_blue.body.get_position() - patch_pos)),
            "cube_yellow_to_patch_center_m": float(np.linalg.norm(self.cube_yellow.body.get_position() - patch_pos)),
        }
        pose_error = {
            "max_perp_dev_m": float(diag_eval.get("metrics", {}).get("max_perp_dev_m", 1e9)),
            "min_adj_spacing_m": float(diag_eval.get("metrics", {}).get("min_adj_spacing_m", 0.0)),
            "endpoint_corner_maxdist_m": float(diag_eval.get("metrics", {}).get("endpoint_corner_maxdist_m", 1e9)),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})