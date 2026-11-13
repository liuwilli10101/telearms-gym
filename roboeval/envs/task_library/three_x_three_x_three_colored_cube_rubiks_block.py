"""Assemble 27 colored cubes into a 3x3x3 Rubik's-cube-like block with one sorted face."""
from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval


class ThreeXThreeXThreeColoredCubeRubiksBlockWithOneSortedFace(RoboEvalEnv, MetricRolloutEval):
    """
    Assemble 27 colored cubes into a 3x3x3 Rubik's-cube-like block.
    Ensure the completed cube is a clean 3-layer structure (3 rows x 3 cols per layer),
    and make one outer face plane (the +X face) correctly color-sorted so that all 9
    cubes on that face are red.

    Sequential Sub-Goals:
        1. Grasp a cube to begin assembling the 3D block.
        2. Place cubes to complete the bottom 3x3 layer in a tight grid.
        3. Stack cubes to complete the middle 3x3 layer aligned with the bottom layer.
        4. Stack cubes to complete the top 3x3 layer aligned with the lower layers.
        5. Ensure the +X face plane of the assembled cube is entirely red (all 9 face positions).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _ROWS = 3
    _COLS = 3
    _LAYERS = 3
    _TOTAL = 27

    _GRID_SPACING = 0.055
    _BASE_Z = 0.97
    _Z_INC = 0.04
    _ORIGIN_XY = np.array([0.55, 0.0])

    _POS_TOL = 0.02
    _UPRIGHT_DEG = 15.0
    _AXIS_ALIGN_DEG = 10.0

    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]

        self.cubes = []
        self._cube_ids = [f"cube_{i:02d}" for i in range(1, self._TOTAL + 1)]

        self._color_map = {
            "cube_01": "red",
            "cube_02": "red",
            "cube_03": "red",
            "cube_04": "red",
            "cube_05": "red",
            "cube_06": "red",
            "cube_07": "red",
            "cube_08": "red",
            "cube_09": "red",
            "cube_10": "blue",
            "cube_11": "blue",
            "cube_12": "blue",
            "cube_13": "green",
            "cube_14": "green",
            "cube_15": "green",
            "cube_16": "yellow",
            "cube_17": "yellow",
            "cube_18": "yellow",
            "cube_19": "orange",
            "cube_20": "orange",
            "cube_21": "orange",
            "cube_22": "white",
            "cube_23": "white",
            "cube_24": "white",
            "cube_25": "purple",
            "cube_26": "purple",
            "cube_27": "purple",
        }

        self._id_to_cube = {}
        # Scatter cubes immediately so physics.reset() (called before _on_reset)
        # doesn't see 27 cubes overlapping at the worldbody origin — that triggers
        # mjWARN_CNSTRFULL because every overlapping pair generates contacts.
        # _on_reset() will re-randomize positions; this is just to keep the
        # initial state physically valid.
        scatter_cols = 5
        scatter_spacing = 0.06
        scatter_origin = np.array([0.41, -0.18, self._BASE_Z], dtype=float)
        for idx, cube_id in enumerate(self._cube_ids):
            c = ColoredCube(self._mojo)
            c.set_color(self._color_map[cube_id])
            row = idx // scatter_cols
            col = idx % scatter_cols
            init_pos = scatter_origin + np.array(
                [col * scatter_spacing, row * scatter_spacing, 0.0], dtype=float
            )
            c.set_pose(
                position=init_pos,
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            self.cubes.append(c)
            self._id_to_cube[cube_id] = c

        self._goal_cells = []
        self._goal_positions = {}
        self._goal_positions_list = []
        self._compute_goal_grid()

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _compute_goal_grid(self):
        self._goal_cells.clear()
        self._goal_positions.clear()
        self._goal_positions_list.clear()

        for z_idx in range(self._LAYERS):
            z = self._BASE_Z + z_idx * self._Z_INC
            for y_idx in range(self._ROWS):
                y = self._ORIGIN_XY[1] + (y_idx - 1) * self._GRID_SPACING
                for x_idx in range(self._COLS):
                    x = self._ORIGIN_XY[0] + (x_idx - 1) * self._GRID_SPACING
                    key = (x_idx, y_idx, z_idx)
                    pos = np.array([x, y, z], dtype=float)
                    self._goal_cells.append(key)
                    self._goal_positions[key] = pos
                    self._goal_positions_list.append(pos)

        self._goal_positions_list = np.asarray(self._goal_positions_list, dtype=float)

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

        for cube_id, cube in self._id_to_cube.items():
            cube.set_color(self._color_map[cube_id])

        # Random UNSOLVED scatter on the table within workspace bounds.
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = float(self._BASE_Z)

        goal_xy = self._goal_positions_list[:, :2]
        min_sep_cubes = 0.035
        min_sep_goal = 0.040

        placed_xy = []
        for cube_id in self._cube_ids:
            cube = self._id_to_cube[cube_id]

            xy = None
            for _ in range(20000):
                cand = np.array(
                    [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)],
                    dtype=float,
                )

                if np.min(np.linalg.norm(goal_xy - cand[None, :], axis=1)) < min_sep_goal:
                    continue

                if placed_xy:
                    prev = np.asarray(placed_xy, dtype=float)
                    if np.min(np.linalg.norm(prev - cand[None, :], axis=1)) < min_sep_cubes:
                        continue

                xy = cand
                break

            if xy is None:
                # Fallback (still random, but may violate separation constraints in rare cases)
                xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)

            placed_xy.append(xy)

            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=float)

            cube.set_pose(position=np.array([xy[0], xy[1], z], dtype=float), quat=quat_wxyz)

        for _ in range(50):
            self._mojo.step()

        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # To visualize the solved configuration, replace the randomized placement above with:
        # --- GOAL STATE ---
        # red_cubes = [self._id_to_cube[f"cube_{i:02d}"] for i in range(1, 10)]
        # other_cubes = [self._id_to_cube[f"cube_{i:02d}"] for i in range(10, 28)]
        # red_i, other_i = 0, 0
        # q_identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        # for z_idx in range(self._LAYERS):
        #     for y_idx in range(self._ROWS):
        #         for x_idx in range(self._COLS):
        #             key = (x_idx, y_idx, z_idx)
        #             pos = self._goal_positions[key].copy()
        #             if x_idx == (self._COLS - 1):
        #                 cube = red_cubes[red_i]
        #                 red_i += 1
        #             else:
        #                 cube = other_cubes[other_i]
        #                 other_i += 1
        #             cube.set_pose(position=pos, quat=q_identity)
        #     for _ in range(50):
        #         self._mojo.step()
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _build_centroid_anchored_grid(self, cube_positions: np.ndarray):
        """Build a 3x3x3 goal grid centred on the cubes' own XY centroid and lowest Z.

        Returns (cells, goal_positions, goal_positions_list) where cells is the list
        of (x_idx, y_idx, z_idx) tuples and goal_positions_list is the Nx3 array in
        matching order. This makes the success check translation-invariant.
        """
        centroid_xy = cube_positions[:, :2].mean(axis=0)
        base_z = float(cube_positions[:, 2].min())

        cells = []
        goal_positions: Dict = {}
        goal_list = []
        for z_idx in range(self._LAYERS):
            z = base_z + z_idx * self._Z_INC
            for y_idx in range(self._ROWS):
                y = float(centroid_xy[1]) + (y_idx - 1) * self._GRID_SPACING
                for x_idx in range(self._COLS):
                    x = float(centroid_xy[0]) + (x_idx - 1) * self._GRID_SPACING
                    key = (x_idx, y_idx, z_idx)
                    pos = np.array([x, y, z], dtype=float)
                    cells.append(key)
                    goal_positions[key] = pos
                    goal_list.append(pos)

        return cells, goal_positions, np.asarray(goal_list, dtype=float)

    def _greedy_assign_cubes_to_cells(self):
        n_c = len(self.cubes)
        n_p = self._TOTAL
        if n_c != n_p:
            return {}, {}, float("inf"), float("inf"), {}

        cube_pos = np.asarray([c.body.get_position() for c in self.cubes], dtype=float)
        cells, goal_positions, goal_list = self._build_centroid_anchored_grid(cube_pos)

        diffs = cube_pos[:, None, :] - goal_list[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)

        pairs = []
        for i in range(n_c):
            for j in range(n_p):
                pairs.append((dmat[i, j], i, j))
        pairs.sort(key=lambda t: t[0])

        used_cubes = set()
        used_cells = set()
        assigned = {}
        cube_to_cell = {}

        for d, i, j in pairs:
            if i in used_cubes or j in used_cells:
                continue
            cell_key = cells[j]
            assigned[cell_key] = i
            cube_to_cell[i] = cell_key
            used_cubes.add(i)
            used_cells.add(j)
            if len(used_cubes) == n_c:
                break

        if len(assigned) != n_p:
            return {}, {}, float("inf"), float("inf"), goal_positions

        errs = []
        for cell_key, cube_i in assigned.items():
            pos = self.cubes[cube_i].body.get_position()
            goal = goal_positions[cell_key]
            errs.append(float(np.linalg.norm(pos - goal)))
        max_err = float(np.max(errs)) if errs else float("inf")
        mean_err = float(np.mean(errs)) if errs else float("inf")
        return assigned, cube_to_cell, max_err, mean_err, goal_positions

    def _is_upright_and_axis_aligned(self, quat_xyzw_or_wxyz):
        qarr = np.asarray(quat_xyzw_or_wxyz, dtype=float).reshape(-1)
        if qarr.shape[0] != 4:
            return False, False

        q = Quaternion(qarr[0], qarr[1], qarr[2], qarr[3])

        ex_w = np.asarray(q.rotate([1.0, 0.0, 0.0]), dtype=float)
        ey_w = np.asarray(q.rotate([0.0, 1.0, 0.0]), dtype=float)
        ez_w = np.asarray(q.rotate([0.0, 0.0, 1.0]), dtype=float)

        upright_thr = np.cos(np.deg2rad(self._UPRIGHT_DEG))
        align_thr = np.cos(np.deg2rad(self._AXIS_ALIGN_DEG))

        upright = bool(ez_w[2] >= upright_thr)
        axis_aligned = bool(
            (np.max(np.abs(ex_w)) >= align_thr)
            and (np.max(np.abs(ey_w)) >= align_thr)
            and (np.max(np.abs(ez_w)) >= align_thr)
        )
        return upright, axis_aligned

    def _check_single_connected_component(self, positions: np.ndarray) -> bool:
        """BFS over cube adjacency graph. Returns True iff all 27 cubes form one connected component."""
        n = len(positions)
        if n == 0:
            return False
        # Adjacency threshold: cubes within 1.5 * grid spacing are considered connected.
        adj_threshold = self._GRID_SPACING * 1.5
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(positions[i] - positions[j])) <= adj_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == n

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if len(self.cubes) != self._TOTAL:
            return False

        assigned, cube_to_cell, max_err, mean_err, _ = self._greedy_assign_cubes_to_cells()
        grid_filled = len(assigned) == self._TOTAL

        positions_ok = bool(grid_filled and (max_err <= self._POS_TOL))

        upright_count = 0
        axis_aligned_count = 0
        for c in self.cubes:
            quat = c.body.get_quaternion()
            upright, axis_aligned = self._is_upright_and_axis_aligned(quat)
            upright_count += int(upright)
            axis_aligned_count += int(axis_aligned)

        upright_frac = upright_count / float(self._TOTAL)
        axis_aligned_frac = axis_aligned_count / float(self._TOTAL)

        # Require ALL 27 cubes upright (100% threshold, not 95%).
        upright_ok = upright_count == self._TOTAL
        axis_aligned_ok = axis_aligned_frac >= 0.95

        face_ok = False
        red_on_face = 0
        if grid_filled:
            # NOTE: Axis assumption inherited from SimGen. Verify with live sim if success checks fail.
            face_cells = [(2, y, z) for z in range(self._LAYERS) for y in range(self._ROWS)]
            face_ok = True
            for cell_key in face_cells:
                cube_i = assigned.get(cell_key, None)
                if cube_i is None:
                    face_ok = False
                    continue
                if self.cubes[cube_i].get_color() != "red":
                    face_ok = False
                else:
                    red_on_face += 1
        face_red_fraction = red_on_face / 9.0

        adjacency_ok = False
        connected_ok = False
        if grid_filled:
            positions = np.asarray([c.body.get_position() for c in self.cubes], dtype=float)
            neighbor_thr = self._GRID_SPACING * 1.20
            ok_cnt = 0
            for i in range(self._TOTAL):
                di = np.linalg.norm(positions[None, i, :] - positions, axis=-1)
                di[i] = 1e9
                if float(np.min(di)) <= neighbor_thr:
                    ok_cnt += 1
            adjacency_ok = ok_cnt >= self._TOTAL
            # All 27 cubes must form a single connected component.
            connected_ok = self._check_single_connected_component(positions)

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        any_stacked = any(c.body.get_position()[2] > (self._BASE_Z + 0.01) for c in self.cubes)
        self._metric_stage(0, bool(any_stacked and released))

        bottom_ok = False
        middle_ok = False
        top_ok = False
        if grid_filled:
            bottom_ok = all((x, y, 0) in assigned for x in range(3) for y in range(3))
            middle_ok = all((x, y, 1) in assigned for x in range(3) for y in range(3))
            top_ok = all((x, y, 2) in assigned for x in range(3) for y in range(3))
        self._metric_stage(1, bool(bottom_ok and positions_ok))
        self._metric_stage(2, bool(middle_ok and positions_ok))
        self._metric_stage(3, bool(top_ok and positions_ok))

        self._metric_stage(4, bool(positions_ok and face_ok))

        self._success_check = bool(
            positions_ok
            and face_ok
            and upright_ok
            and axis_aligned_ok
            and adjacency_ok
            and connected_ok
            and released
        )

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "mean_cell_error_m": float(mean_err) if np.isfinite(mean_err) else 1.0,
                "max_cell_error_m": float(max_err) if np.isfinite(max_err) else 1.0,
                "pos_tol_m": float(self._POS_TOL),
                "face_red_fraction": float(face_red_fraction),
            },
            pose_error={
                "upright_fraction": float(upright_frac),
                "axis_aligned_fraction": float(axis_aligned_frac),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
