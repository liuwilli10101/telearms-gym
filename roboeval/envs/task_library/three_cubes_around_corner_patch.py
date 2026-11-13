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

class TargetTrianglePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes around the green target patch at the front-left corner
    of the workspace so the cubes form a clear triangular arrangement around the patch.

    Sequential Sub-Goals:
        1. Grasp the red cube.
        2. Place the red cube near the green goal patch as one vertex of the triangle.
        3. Grasp the blue cube.
        4. Place the blue cube near the green goal patch as another vertex of the triangle.
        5. Grasp the yellow cube.
        6. Place the yellow cube near the green goal patch as the third vertex of the triangle.
        7. Nudge/reposition cubes so they are well-spaced and clearly form a triangle around the patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task parameters (from spec) ---
    _ANNULUS_INNER_R = 0.06
    _ANNULUS_OUTER_R = 0.12

    _MIN_PAIRWISE_DIST = 0.07
    _MIN_TRI_AREA = 0.001
    _MIN_ANG_SEP_DEG = 40.0

    # Stable, clearly-triangular goal geometry (centered on patch)
    _GOAL_RADIUS = 0.10
    _GOAL_ANGLES_DEG = (20.0, 160.0, 280.0)  # red, blue, yellow

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (ONLY allowed assets: ColoredCube, GoalPatch) ---
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_blue, self.cube_yellow]

        self.goal_patch_green = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # Fixed goal patch pose (front-left corner region of workspace, but safely within bounds)
        self._patch_pos = np.array([0.62, 0.22, 0.951])
        self.goal_patch_green.body.set_position(self._patch_pos)

        # Precompute ideal goal XY positions around patch (red, blue, yellow)
        self._goal_xy = []
        for a_deg in self._GOAL_ANGLES_DEG:
            a = np.deg2rad(a_deg)
            self._goal_xy.append(
                self._patch_pos[:2] + self._GOAL_RADIUS * np.array([np.cos(a), np.sin(a)])
            )
        self._goal_xy = np.asarray(self._goal_xy)

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # 3. Reset patch pose (fixed)
        self.goal_patch_green.body.set_position(self._patch_pos)
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_yellow.set_color("yellow")

        # 4. Place cubes in RANDOM *UNSOLVED* positions within bounds.
        #    Guarantee _success() == False by forcing each cube to violate the annulus constraint:
        #    place each cube at least (outer_r + margin) away from the patch in XY.
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        patch_xy = self._patch_pos[:2].astype(float)
        min_dist_from_patch = float(self._ANNULUS_OUTER_R + 0.05)  # ensure outside annulus
        min_pairwise = 0.05  # light anti-overlap to reduce instability

        placed_xy = []

        for cube in self.cubes:
            xy = None
            for _ in range(500):
                cand = np.array(
                    [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
                )
                if float(np.linalg.norm(cand - patch_xy)) < min_dist_from_patch:
                    continue
                if any(float(np.linalg.norm(cand - p)) < min_pairwise for p in placed_xy):
                    continue
                xy = cand
                break

            if xy is None:
                # Fallback: deterministic corner positions (still outside annulus) within bounds
                fallback = [
                    np.array([0.40, -0.25], dtype=float),
                    np.array([0.70, -0.25], dtype=float),
                    np.array([0.70, 0.25], dtype=float),
                ]
                for cand in fallback:
                    if float(np.linalg.norm(cand - patch_xy)) >= min_dist_from_patch and all(
                        float(np.linalg.norm(cand - p)) >= min_pairwise for p in placed_xy
                    ):
                        xy = cand
                        break
                if xy is None:
                    # Last-resort: just pick a bound-safe point and rely on annulus violation
                    xy = np.array([0.70, -0.25], dtype=float)

            placed_xy.append(xy)

            cube.set_pose(
                position=np.array([float(xy[0]), float(xy[1]), float(z_base)]),
                quat=np.array([0, 0, 0, 1]),
            )
            self._mojo.physics.forward()
            self._mojo.physics.forward()

        # 5. Physics settling
        self._mojo.physics.forward()
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ----------------- Geometry helpers -----------------
    @staticmethod
    def _xy(pos3: np.ndarray) -> np.ndarray:
        return np.asarray(pos3, dtype=float)[:2]

    @staticmethod
    def _tri_area_xy(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        # area = 0.5 * |(b-a) x (c-a)| in 2D (z-component of cross)
        return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def _point_in_triangle_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        # Barycentric sign method in 2D
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    @staticmethod
    def _min_angular_separation_deg(angles_deg: np.ndarray) -> float:
        ang = np.mod(angles_deg, 360.0)
        ang_sorted = np.sort(ang)
        diffs = np.diff(np.r_[ang_sorted, ang_sorted[0] + 360.0])
        return float(np.min(diffs))

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "goal_patch_green"):
            return False

        patch_xy = self._xy(self.goal_patch_green.body.get_position())

        # --- Condition 1: each cube within annulus around patch (and not on patch collider) ---
        annulus_ok = True
        radial_dists = {}
        not_on_patch = True
        cube_xys = []

        for cube, name in [
            (self.cube_red, "cube_red"),
            (self.cube_blue, "cube_blue"),
            (self.cube_yellow, "cube_yellow"),
        ]:
            cxy = self._xy(cube.body.get_position())
            cube_xys.append(cxy)

            d = float(np.linalg.norm(cxy - patch_xy))
            radial_dists[name] = d

            if not (self._ANNULUS_INNER_R <= d <= self._ANNULUS_OUTER_R):
                annulus_ok = False

            # Ensure cube is not on top of patch (collision with patch collider)
            # (GoalPatch collision direction matters: check from cube side)
            if hasattr(self.goal_patch_green, "collider") and cube.is_colliding(self.goal_patch_green.collider):
                # Be robust to overly-broad/unstable thin-collider contacts by also requiring
                # the cube center to be within the patch footprint (AABB in XY).
                delta = np.abs(cxy - patch_xy)
                if float(delta[0]) <= 0.072 and float(delta[1]) <= 0.072:
                    not_on_patch = False

        cube_xys = np.asarray(cube_xys)

        # --- Condition 2: triangle arrangement around patch ---
        # Pairwise distance check
        pairwise_ok = True
        pairwise_d = {}
        pairs = [("red", 0, "blue", 1), ("red", 0, "yellow", 2), ("blue", 1, "yellow", 2)]
        for n1, i1, n2, i2 in pairs:
            dij = float(np.linalg.norm(cube_xys[i1] - cube_xys[i2]))
            pairwise_d[f"{n1}-{n2}"] = dij
            if dij < self._MIN_PAIRWISE_DIST:
                pairwise_ok = False

        # Triangle area check
        tri_area = self._tri_area_xy(cube_xys[0], cube_xys[1], cube_xys[2])
        area_ok = tri_area >= self._MIN_TRI_AREA

        # Angular separation around the patch
        vecs = cube_xys - patch_xy[None, :]
        angles_deg = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
        min_ang_sep = self._min_angular_separation_deg(angles_deg)
        ang_ok = min_ang_sep >= self._MIN_ANG_SEP_DEG

        # Patch should be "inside" the triangle to be "around" it (robust surround test)
        inside_ok = self._point_in_triangle_xy(patch_xy, cube_xys[0], cube_xys[1], cube_xys[2])

        arrangement_ok = bool(pairwise_ok and area_ok and ang_ok and inside_ok)

        positions_ok = bool(annulus_ok and not_on_patch and arrangement_ok)

        # NOTE: Per active constraints, do NOT require grasp/release state for success.
        semantic_ok = True

        # Subgoal staging (we cannot reliably detect "grasp" without using grasp-state success checks)
        # Stage "place" subgoals based on annulus satisfaction; final stage on arrangement.
        # Indices:
        # 0 grasp_red, 1 place_red, 2 grasp_blue, 3 place_blue, 4 grasp_yellow, 5 place_yellow, 6 adjust_triangle
        self._metric_stage(1, bool(self._ANNULUS_INNER_R <= radial_dists["cube_red"] <= self._ANNULUS_OUTER_R))
        self._metric_stage(3, bool(self._ANNULUS_INNER_R <= radial_dists["cube_blue"] <= self._ANNULUS_OUTER_R))
        self._metric_stage(5, bool(self._ANNULUS_INNER_R <= radial_dists["cube_yellow"] <= self._ANNULUS_OUTER_R))
        self._metric_stage(6, bool(arrangement_ok))

        # If fully successful, mark all stages complete for reporting consistency
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_red_to_patch_xy_m": float(radial_dists.get("cube_red", np.nan)),
                "cube_blue_to_patch_xy_m": float(radial_dists.get("cube_blue", np.nan)),
                "cube_yellow_to_patch_xy_m": float(radial_dists.get("cube_yellow", np.nan)),
                "min_pairwise_xy_m": float(min(pairwise_d.values())) if len(pairwise_d) else float(np.nan),
                "pair_red_blue_xy_m": float(pairwise_d.get("red-blue", np.nan)),
                "pair_red_yellow_xy_m": float(pairwise_d.get("red-yellow", np.nan)),
                "pair_blue_yellow_xy_m": float(pairwise_d.get("blue-yellow", np.nan)),
            },
            pose_error={
                "triangle_area_m2": float(tri_area),
                "triangle_area_shortfall_m2": float(max(0.0, self._MIN_TRI_AREA - tri_area)),
                "min_angular_separation_deg": float(min_ang_sep),
                "annulus_ok": float(bool(annulus_ok)),
                "not_on_patch": float(bool(not_on_patch)),
                "patch_inside_triangle": float(bool(inside_ok)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})