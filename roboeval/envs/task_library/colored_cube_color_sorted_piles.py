from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class ColoredCubeColorSortedPiles(RoboEvalEnv, MetricRolloutEval):
    """Sort eight cubes into four same-color stacks of two: red, blue, green, and yellow.

Success Criteria:
Put one same-color cube on each base, keep stacks at least 12 cm apart, and release all cubes. Stack alignment tolerance is 3 cm; uprightness checks still apply."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_SIZE = 0.04
    _BASE_Z = 0.97
    _STACK_DZ = 0.04

    _BASE_POS_TOL = 0.03
    _ON_TOP_XY_TOL = 0.03
    _ON_TOP_Z_TOL = 0.02
    _MIN_INTER_PILE_DIST = 0.12
    _UPRIGHT_TOL_DEG = 15.0

    _NUM_SUBGOALS = 5
    _EXPECTED_COLORS = ("red", "blue", "green", "yellow")
    # Three-high piles were reported as too hard: the third cube has to be placed
    # on an already-tall, already-wobbly tower. Two per pile keeps the sorting
    # skill the task is about and drops the balancing act.
    _CUBES_PER_PILE = 2

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        # Derived from the colours and pile height so the two stay in step.
        n_cubes = len(self._EXPECTED_COLORS) * self._CUBES_PER_PILE
        self._cube_ids = [f"cube_{i:02d}" for i in range(1, n_cubes + 1)]
        self._cube_color_map = {
            cube_id: self._EXPECTED_COLORS[idx // self._CUBES_PER_PILE]
            for idx, cube_id in enumerate(self._cube_ids)
        }

        self._ref_xy = np.array([0.55, 0.0], dtype=float)

        self._piles = [
            {
                "color": "red",
                "base_xy_offset": np.array([-0.12, -0.12], dtype=float),
            },
            {
                "color": "blue",
                "base_xy_offset": np.array([-0.12, 0.12], dtype=float),
            },
            {
                "color": "green",
                "base_xy_offset": np.array([0.12, -0.12], dtype=float),
            },
            {
                "color": "yellow",
                "base_xy_offset": np.array([0.12, 0.12], dtype=float),
            },
        ]

        # Compute expected base XY for each color pile (for tolerance check)
        self._pile_base_xy = {
            pile["color"]: self._ref_xy + pile["base_xy_offset"]
            for pile in self._piles
        }

        self.cubes = []
        for cube_id in self._cube_ids:
            cube = ColoredCube(self._mojo)
            cube.set_color(self._cube_color_map[cube_id])
            self.cubes.append(cube)

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

        # Re-apply cube colors (ensures colors are correct after each reset)
        for cube_id, cube in zip(self._cube_ids, self.cubes):
            cube.set_color(self._cube_color_map[cube_id])

        # Randomize all cubes into unsolved tabletop positions (no stacks).
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._BASE_Z)

        # Keep cubes separated to avoid accidental stacking/collisions that could satisfy success.
        min_sep = 1.6 * float(self._CUBE_SIZE)  # conservative spacing (> cube size)

        positions_xy = []
        max_tries = 5000
        for _ in range(len(self._cube_ids)):
            placed = False
            for _try in range(max_tries):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)
                if all(float(np.linalg.norm(p - q)) >= min_sep for q in positions_xy):
                    positions_xy.append(p)
                    placed = True
                    break
            if not placed:
                # Fallback: accept even if closer (still unsolved since z is base and not stacked).
                positions_xy.append(
                    np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max))], dtype=float)
                )

        # Apply poses with random yaw (upright), all on the table base height.
        for cube, pxy in zip(self.cubes, positions_xy):
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            cube.set_pose(np.array([pxy[0], pxy[1], z], dtype=float), quat=np.array(q.elements, dtype=float))

        # Let physics settle.
        for _ in range(100):
            self._mojo.step()

        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        _ = rng.random()

        # --- GOAL STATE ---
        # rng = np.random.default_rng(self.seed)
        #
        # # Re-apply cube colors
        # for cube_id, cube in zip(self._cube_ids, self.cubes):
        #     cube.set_color(self._cube_color_map[cube_id])
        #
        # q_upright = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        #
        # # Place bottom cubes first (one per pile)
        # for pile in self._piles:
        #     color = pile["color"]
        #     base_xy = self._pile_base_xy[color]
        #     # Get all cubes of this color
        #     color_cubes = [c for c in self.cubes if c.get_color() == color]
        #     color_cubes[0].set_pose(np.array([base_xy[0], base_xy[1], self._BASE_Z], dtype=float), quat=q_upright)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Place middle cubes
        # for pile in self._piles:
        #     color = pile["color"]
        #     base_xy = self._pile_base_xy[color]
        #     color_cubes = [c for c in self.cubes if c.get_color() == color]
        #     color_cubes[1].set_pose(np.array([base_xy[0], base_xy[1], self._BASE_Z + self._STACK_DZ], dtype=float), quat=q_upright)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Place top cubes
        # for pile in self._piles:
        #     color = pile["color"]
        #     base_xy = self._pile_base_xy[color]
        #     color_cubes = [c for c in self.cubes if c.get_color() == color]
        #     color_cubes[2].set_pose(np.array([base_xy[0], base_xy[1], self._BASE_Z + 2 * self._STACK_DZ], dtype=float), quat=q_upright)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # for idx in range(self._NUM_SUBGOALS):
        #     self._metric_stage(idx, False)
        #
        # _ = rng.random()
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_upright(self, quat: np.ndarray, tol_deg: float) -> bool:
        try:
            q = Quaternion(quat)
            z_axis = np.array(q.rotate([0.0, 0.0, 1.0]), dtype=float)
            z_axis /= (np.linalg.norm(z_axis) + 1e-9)
            cos_ang = float(np.clip(np.dot(z_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
            ang_deg = float(np.degrees(np.arccos(cos_ang)))
            return ang_deg <= tol_deg
        except Exception:
            return False

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Group cubes by their current color at runtime (property-based, not ID-based).
        color_groups: Dict = {}
        for cube in self.cubes:
            col = cube.get_color()
            color_groups.setdefault(col, []).append(cube)

        # Verify exactly 4 color groups with exactly 3 cubes each.
        expected_colors = set(self._EXPECTED_COLORS)
        if set(color_groups.keys()) != expected_colors:
            return False
        if any(len(group) != self._CUBES_PER_PILE for group in color_groups.values()):
            return False

        positions_ok = True
        upright_ok = True
        inter_pile_ok = True

        base_xys = []

        for color in self._EXPECTED_COLORS:
            group = color_groups[color]
            # Sort cubes in this group by Z to get bottom/middle/top.
            # Level 0 is the base; every level above must sit on the one below.
            # Written as a loop so _CUBES_PER_PILE is the single source of truth.
            group_by_z = sorted(group, key=lambda c: c.body.get_position()[2])
            bottom = group_by_z[0]
            bottom_pos = bottom.body.get_position()
            base_xys.append(np.array([bottom_pos[0], bottom_pos[1]], dtype=float))

            # Bottom cube: must rest on the table (contact-based, no absolute XY anchor).
            if not bottom.is_colliding(self.table.table_top):
                positions_ok = False
            if not self._is_upright(bottom.body.get_quaternion(), self._UPRIGHT_TOL_DEG):
                upright_ok = False

            for level in range(1, self._CUBES_PER_PILE):
                upper = group_by_z[level]
                lower = group_by_z[level - 1]
                upper_pos = upper.body.get_position()
                lower_pos = lower.body.get_position()

                # XY colocated with the BASE, resting on the cube below, off the table.
                xy_err = float(np.linalg.norm(upper_pos[:2] - bottom_pos[:2]))
                dz = float(upper_pos[2] - lower_pos[2])
                if xy_err > self._ON_TOP_XY_TOL:
                    positions_ok = False
                if abs(dz - self._STACK_DZ) > self._ON_TOP_Z_TOL:
                    positions_ok = False
                if not upper.is_colliding(lower):
                    positions_ok = False
                if upper.is_colliding(self.table.table_top):
                    positions_ok = False
                if not self._is_upright(upper.body.get_quaternion(), self._UPRIGHT_TOL_DEG):
                    upright_ok = False

        # Inter-pile separation: each pile's bottom cube must be spatially apart from others.
        for i in range(len(base_xys)):
            for j in range(i + 1, len(base_xys)):
                d = float(np.linalg.norm(base_xys[i] - base_xys[j]))
                if d < self._MIN_INTER_PILE_DIST:
                    inter_pile_ok = False
                    break

        if positions_ok and upright_ok and inter_pile_ok:
            for idx in range(self._NUM_SUBGOALS):
                self._metric_stage(idx, True)

        self._success_check = bool(positions_ok and upright_ok and inter_pile_ok and released)

        # Compute per-cube distance metrics (centroid-anchored: anchor each pile on its
        # own bottom-cube XY so metrics remain translation-invariant).
        target_distance: Dict[str, float] = {}
        pose_error: Dict[str, float] = {}
        for color in self._EXPECTED_COLORS:
            group = color_groups[color]
            group_by_z = sorted(group, key=lambda c: c.body.get_position()[2])
            bottom_xy = group_by_z[0].body.get_position()[:2]
            bottom_z = float(group_by_z[0].body.get_position()[2])
            for level, cube in enumerate(group_by_z):
                key = f"{color}_level{level}"
                pos = cube.body.get_position()
                goal = np.array([bottom_xy[0], bottom_xy[1], bottom_z + level * self._STACK_DZ])
                target_distance[key] = float(np.linalg.norm(pos - goal))
                try:
                    q = Quaternion(cube.body.get_quaternion())
                    z_axis = np.array(q.rotate([0.0, 0.0, 1.0]), dtype=float)
                    z_axis /= (np.linalg.norm(z_axis) + 1e-9)
                    cos_ang = float(np.clip(np.dot(z_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
                    pose_error[key] = float(np.degrees(np.arccos(cos_ang)))
                except Exception:
                    pose_error[key] = float("inf")

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
