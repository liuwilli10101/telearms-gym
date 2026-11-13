from __future__ import annotations

import numpy as np
from typing import Dict
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class YellowCubePyramidStack321(RoboEvalEnv, MetricRolloutEval):
    """
    Build a small 3-layer pyramid using six yellow colored cubes: a 3-cube base row on the table,
    a 2-cube middle row centered on the base, and 1 cube centered on top (3-2-1).

    Sequential Sub-Goals:
        0. Grasp any yellow cube.
        1. Place three yellow cubes as the base row on the table.
        2. Place two yellow cubes as the centered middle row (bridging the base seams).
        3. Place the final yellow cube centered on top to complete the pyramid.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Geometry / tolerances ---
    _N_CUBES = 6
    _CUBE_SIZE_M = 0.04
    _BASE_Z = 0.97
    _DZ = 0.04

    # Task spec tolerance (keep as given)
    _POS_TOL = 0.02

    # For physical stability/contact, use touching spacing (still within tolerance of spec's 0.055m)
    _SPACING_Y = 0.04  # base row center-to-center spacing (reset-only anchor)

    # Orientation tolerances
    _UPRIGHT_COS_TOL = float(np.cos(np.deg2rad(12.0)))
    _AXIS_ALIGN_COS_TOL = float(np.cos(np.deg2rad(12.0)))

    # Structural (translation-invariant) pyramid tolerances.
    _BASE_COLLINEAR_TOL = 0.015    # max perpendicular deviation of a base cube from the fit line (m)
    _BASE_SPACING_STD_TOL = 0.010  # std of adjacent base-cube spacing (m)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Create cubes (ONLY ColoredCube) ---
        self.cubes = []
        self._cube_ids = [f"cube_{i:02d}" for i in range(1, self._N_CUBES + 1)]
        for _ in range(self._N_CUBES):
            c = ColoredCube(self._mojo)
            c.set_color("yellow")  # REQUIRED in _initialize_env()
            self.cubes.append(c)

        # --- Build goal slot positions for a centered 3-2-1 pyramid ---
        # Place at table center-ish with requested offset in X.
        origin_x = 0.55
        origin_y = 0.00

        # Base: y = [-s, 0, +s] at z = base_z
        y0 = [-self._SPACING_Y, 0.0, self._SPACING_Y]
        z_base = self._BASE_Z
        z_mid = self._BASE_Z + self._DZ
        z_top = self._BASE_Z + 2 * self._DZ

        # Middle row bridges seams: y = [-s/2, +s/2]
        y1 = [-0.5 * self._SPACING_Y, 0.5 * self._SPACING_Y]

        # Top centered: y = 0
        self._goal_positions = [
            np.array([origin_x, origin_y + y0[0], z_base]),  # slot 0: base left
            np.array([origin_x, origin_y + y0[1], z_base]),  # slot 1: base mid
            np.array([origin_x, origin_y + y0[2], z_base]),  # slot 2: base right
            np.array([origin_x, origin_y + y1[0], z_mid]),   # slot 3: middle left
            np.array([origin_x, origin_y + y1[1], z_mid]),   # slot 4: middle right
            np.array([origin_x, origin_y + 0.0, z_top]),     # slot 5: top
        ]

        # Slot groupings (by layer)
        self._base_slots = [0, 1, 2]
        self._mid_slots = [3, 4]
        self._top_slots = [5]

        # Contact requirements (slot index pairs)
        self._adjacent_pairs = [
            (0, 1),
            (1, 2),  # base row touching
            (3, 4),  # middle row touching
            (3, 0),
            (3, 1),  # middle-left supported by base-left + base-mid
            (4, 1),
            (4, 2),  # middle-right supported by base-mid + base-right
            (5, 3),
            (5, 4),  # top supported by both middle cubes
        ]

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

        # 2. Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # 3. Set cube colors (REQUIRED in _on_reset())
        for c in self.cubes:
            c.set_color("yellow")

        # 4. Place objects in RANDOM unsolved positions within bounds
        # Bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._BASE_Z
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Sample well-separated XY points so they don't start stacked/in-contact.
        # Also keep them away from the pyramid goal region so slot assignment cannot succeed.
        # Goal region is near (0.55, 0.0); avoid a radius around it and also avoid being within
        # _POS_TOL of any individual goal slot.
        goal_center = np.array([0.55, 0.0])
        avoid_center_r = 0.14  # larger than any slot's radial extent from center
        min_sep = 0.09  # prevent accidental contacts (contact thresh ~0.052)

        placed_xy = []
        for cube in self.cubes:
            xy = None
            for _ in range(2000):
                cand = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

                # Keep away from goal center (prevents matching all slots)
                if float(np.linalg.norm(cand - goal_center)) < avoid_center_r:
                    continue

                # Keep away from each goal slot by more than position tolerance
                too_close_to_slot = False
                for gp in self._goal_positions:
                    if float(np.linalg.norm(cand - gp[:2])) <= (self._POS_TOL + 0.02):
                        too_close_to_slot = True
                        break
                if too_close_to_slot:
                    continue

                # Keep cubes separated to avoid accidental stacking/contact structure
                if any(float(np.linalg.norm(cand - p)) < min_sep for p in placed_xy):
                    continue

                xy = cand
                break

            # Fallback (should be rare): place on a simple line far from goal center
            if xy is None:
                idx = len(placed_xy)
                xy = np.array([x_min + 0.02 * idx, y_max - 0.02 * idx])

            placed_xy.append(xy)
            cube.set_pose(position=np.array([xy[0], xy[1], z]), quat=identity_quat)

        for _ in range(100):
            self._mojo.step()

        # 5. Initialize subgoals to False (4 subgoals)
        for idx in range(4):
            self._metric_stage(idx, False)

        # --- GOAL STATE ---
        # rng = np.random.default_rng(self.seed)
        #
        # # Set cube colors
        # for c in self.cubes:
        #     c.set_color("yellow")
        #
        # # Place objects in GOAL STATE positions (validation pipeline will randomize later)
        # # Permute which physical cube goes to which slot (still solved, but consistent with "any cube any slot").
        # perm = list(rng.permutation(self._N_CUBES))
        # identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        #
        # # Layer-by-layer placement for physics stability
        # # Base layer
        # for k, slot_idx in enumerate(self._base_slots):
        #     cube = self.cubes[perm[k]]
        #     cube.set_pose(position=self._goal_positions[slot_idx], quat=identity_quat)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Middle layer
        # for k, slot_idx in enumerate(self._mid_slots, start=3):
        #     cube = self.cubes[perm[k]]
        #     cube.set_pose(position=self._goal_positions[slot_idx], quat=identity_quat)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Top cube
        # cube = self.cubes[perm[5]]
        # cube.set_pose(position=self._goal_positions[self._top_slots[0]], quat=identity_quat)
        # for _ in range(50):
        #     self._mojo.step()
        #
        # # Initialize subgoals to False (4 subgoals)
        # for idx in range(4):
        #     self._metric_stage(idx, False)
        # --- END GOAL STATE ---

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_upright_and_axis_aligned(self, cube: ColoredCube) -> bool:
        q_arr = cube.body.get_quaternion()
        try:
            q = Quaternion(q_arr)
        except Exception:
            return False

        # Upright: local +Z aligned with world +Z (allow 180deg about Z)
        local_z_world = np.array(q.rotate([0.0, 0.0, 1.0]))
        if abs(float(np.dot(local_z_world, np.array([0.0, 0.0, 1.0])))) < self._UPRIGHT_COS_TOL:
            return False

        # Axis-aligned yaw: local +X projected onto XY aligned with world X or Y axes
        local_x_world = np.array(q.rotate([1.0, 0.0, 0.0]))
        v = local_x_world.copy()
        v[2] = 0.0
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return False
        v /= n

        ax = abs(float(np.dot(v, np.array([1.0, 0.0, 0.0]))))
        ay = abs(float(np.dot(v, np.array([0.0, 1.0, 0.0]))))
        if max(ax, ay) < self._AXIS_ALIGN_COS_TOL:
            return False
        return True

    def _in_contact(self, a: ColoredCube, b: ColoredCube, thresh: float = 0.052) -> bool:
        try:
            if a.is_colliding(b):
                return True
        except Exception:
            pass
        return distance(a.body, b.body) < thresh

    def _partition_layers(self):
        """Partition cubes by physical layer using contact with the table and with each other.

        Returns (base_idxs, mid_idxs, top_idxs) or None if the partition is not a clean 3-2-1.
        """
        n = len(self.cubes)
        base_idxs = []
        above_idxs = []
        for i, c in enumerate(self.cubes):
            if c.is_colliding(self.table.table_top):
                base_idxs.append(i)
            else:
                above_idxs.append(i)
        if len(base_idxs) != 3 or len(above_idxs) != 3:
            return None

        # A middle-layer cube is supported by >= 2 base cubes.
        mid_idxs, top_idxs = [], []
        for i in above_idxs:
            c = self.cubes[i]
            base_contacts = sum(
                1 for j in base_idxs if self._in_contact(c, self.cubes[j])
            )
            if base_contacts >= 2:
                mid_idxs.append(i)
            else:
                top_idxs.append(i)
        if len(mid_idxs) != 2 or len(top_idxs) != 1:
            return None

        return base_idxs, mid_idxs, top_idxs

    def _base_row_geometry_ok(self, base_idxs) -> bool:
        """Base cubes must be roughly collinear with approximately equal adjacent spacing."""
        pts = np.array([self.cubes[i].body.get_position()[:2] for i in base_idxs], dtype=float)

        # Direction = longest-axis unit vector from the 3-point XY cluster.
        centre = pts.mean(axis=0)
        centred = pts - centre
        # SVD-based best-fit line direction (principal axis).
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        normal = np.array([-direction[1], direction[0]])

        # Perpendicular deviations of each cube from the line through the centroid.
        perp = np.abs(centred @ normal)
        if float(np.max(perp)) > self._BASE_COLLINEAR_TOL:
            return False

        # Ordered projections along the direction -> gaps must be ~equal.
        projs = centred @ direction
        order = np.argsort(projs)
        sorted_projs = projs[order]
        gaps = np.diff(sorted_projs)
        if float(np.std(gaps)) > self._BASE_SPACING_STD_TOL:
            return False

        # Adjacent pairs must actually touch.
        order_cube_idxs = [base_idxs[k] for k in order]
        for a, b in zip(order_cube_idxs[:-1], order_cube_idxs[1:]):
            if not self._in_contact(self.cubes[a], self.cubes[b]):
                return False

        return True

    def _success(self) -> bool:
        self._success_check = False

        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        any_held = any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )
        # Any cube raised above its starting base layer (relative: above the lowest cube's Z).
        z_vals = [float(c.body.get_position()[2]) for c in self.cubes]
        any_lifted = (max(z_vals) - min(z_vals)) > 0.02
        self._metric_stage(0, bool(any_lifted or any_held))

        # --- Structural partition into layers (3-2-1) ---
        partition = self._partition_layers()

        color_ok = all(c.get_color() == "yellow" for c in self.cubes)
        upright_ok = all(self._is_upright_and_axis_aligned(c) for c in self.cubes)

        base_ok = False
        mid_ok = False
        top_ok = False
        contact_ok = False

        if partition is not None:
            base_idxs, mid_idxs, top_idxs = partition

            base_ok = self._base_row_geometry_ok(base_idxs)

            # Each middle cube contacts 2 different base cubes.
            mid_base_contacts = []
            for i in mid_idxs:
                n_contacts = sum(
                    1 for j in base_idxs if self._in_contact(self.cubes[i], self.cubes[j])
                )
                mid_base_contacts.append(n_contacts)
            mid_ok = all(n >= 2 for n in mid_base_contacts)

            # Top cube contacts both middle cubes.
            top_cube = self.cubes[top_idxs[0]]
            top_mid_contacts = sum(
                1 for j in mid_idxs if self._in_contact(top_cube, self.cubes[j])
            )
            top_ok = top_mid_contacts >= 2

            contact_ok = mid_ok and top_ok

        positions_ok = bool(base_ok and mid_ok and top_ok and color_ok and upright_ok)

        self._metric_stage(1, bool(base_ok and color_ok))
        self._metric_stage(2, bool(base_ok and mid_ok and color_ok))
        self._metric_stage(3, bool(positions_ok and contact_ok))

        self._success_check = bool(positions_ok and contact_ok and released)

        # Metrics: centroid-anchored pyramid error (for reporting only).
        if partition is not None:
            base_xy = np.array([self.cubes[i].body.get_position()[:2] for i in partition[0]])
            centre_xy = base_xy.mean(axis=0)
            base_z = float(np.mean([self.cubes[i].body.get_position()[2] for i in partition[0]]))
            anchored = [
                *[self.cubes[i].body.get_position() - np.array([*centre_xy, base_z])
                  for i in partition[0]],
                *[self.cubes[i].body.get_position() - np.array([*centre_xy, base_z + self._DZ])
                  for i in partition[1]],
                *[self.cubes[i].body.get_position() - np.array([*centre_xy, base_z + 2 * self._DZ])
                  for i in partition[2]],
            ]
            errs = [float(np.linalg.norm(v)) for v in anchored]
            mean_pos_err = float(np.mean(errs)) if errs else 1.0
            max_pos_err = float(np.max(errs)) if errs else 1.0
        else:
            mean_pos_err = 1.0
            max_pos_err = 1.0

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "pyramid_mean_pos_err_m": mean_pos_err,
                "pyramid_max_pos_err_m": max_pos_err,
            },
            pose_error={
                "pyramid_mean_pos_err_m": mean_pos_err,
                "pyramid_max_pos_err_m": max_pos_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
