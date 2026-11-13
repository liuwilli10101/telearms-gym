"""SetTable: Arrange a place setting with utensils in correct relative positions."""

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.kitchen_items import Plate, Fork, Knife, Spoon, Mug
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class SetTableEnv(RoboEvalEnv, MetricRolloutEval):
    """Arrange a place setting: fork left, knife right, spoon above, mug top-right of plate."""

    _PRESET_PATH = PRESETS_PATH / "kitchen" / "counter_workspace.yaml"

    _success_check = True
    _final_metrics = {}

    # Workspace bounds from counter_workspace.yaml
    _WORKSPACE_CENTER = np.array([0.55, 0.0])
    _WORKSPACE_HALF = np.array([0.275, 0.275])

    # Expected XY offsets from plate center for each utensil
    _FORK_OFFSET = np.array([-0.15, 0.0])     # Left of plate
    _KNIFE_OFFSET = np.array([0.15, 0.0])      # Right of plate
    _SPOON_OFFSET = np.array([0.0, 0.12])      # Above plate
    _MUG_OFFSET = np.array([0.12, 0.12])        # Top-right of plate

    _POSITION_TOLERANCE = 0.05  # +-5cm

    def _initialize_env(self):
        self.plate = Plate(self._mojo)
        self.fork = Fork(self._mojo)
        self.knife = Knife(self._mojo)
        self.spoon = Spoon(self._mojo)
        self.mug = Mug(self._mojo)
        self.table = self._preset.get_props(Table)[0]

        self._all_props = [self.plate, self.fork, self.knife, self.spoon, self.mug]

        self._metric_init(
            track_slippage=True,
            slip_objects=self._all_props,
            robot=self.robot,
            track_collisions=True,
        )

    def _on_reset(self):
        super()._on_reset()
        self._metric_init(
            track_slippage=True,
            slip_objects=self._all_props,
            robot=self.robot,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Place plate at center of workspace with small perturbation
        plate_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.plate.body.set_position(np.array([0.55, 0.0, 1.00]) + plate_offset)
        plate_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.plate.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=plate_angle).elements
        )

        # Scatter utensils at various positions (not at target positions)
        scatter_positions = [
            np.array([0.45, -0.15, 0.955]),   # Fork
            np.array([0.65, -0.15, 0.955]),    # Knife
            np.array([0.45, 0.15, 0.96]),      # Spoon
            np.array([0.65, 0.15, 1.00]),      # Mug
        ]
        for prop, base_pos in zip(
            [self.fork, self.knife, self.spoon, self.mug], scatter_positions
        ):
            offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
            prop.body.set_position(base_pos + offset)
            angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
            prop.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )

        # Initialize metric stages (one per utensil placed correctly)
        for idx in range(1, 5):
            self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    @staticmethod
    def _check_relative_position(item_pos, plate_pos, expected_offset_xy, tolerance=0.05):
        """Check if item is at expected XY offset from plate center.

        This is a local math helper for relative position checking --
        computes actual XY offset and checks against expected offset.
        """
        actual_offset = item_pos[:2] - plate_pos[:2]
        return float(np.linalg.norm(actual_offset - expected_offset_xy)) < tolerance

    def _success(self) -> bool:
        self._success_check = True

        plate_pos = self.plate.body.get_position()
        fork_pos = self.fork.body.get_position()
        knife_pos = self.knife.body.get_position()
        spoon_pos = self.spoon.body.get_position()
        mug_pos = self.mug.body.get_position()

        # Check relative positions with +-5cm tolerance
        utensil_checks = {
            "fork": (fork_pos, self._FORK_OFFSET),
            "knife": (knife_pos, self._KNIFE_OFFSET),
            "spoon": (spoon_pos, self._SPOON_OFFSET),
            "mug": (mug_pos, self._MUG_OFFSET),
        }

        dist_metrics = {}
        for idx, (name, (item_pos, expected_offset)) in enumerate(
            utensil_checks.items(), start=1
        ):
            actual_offset = item_pos[:2] - plate_pos[:2]
            offset_dist = float(np.linalg.norm(actual_offset - expected_offset))
            dist_metrics[f"{name}-target_offset"] = offset_dist

            if not self._check_relative_position(
                item_pos, plate_pos, expected_offset, self._POSITION_TOLERANCE
            ):
                self._success_check = False
            else:
                self._metric_stage(idx)

        # All items must be upright (inline check_upright logic)
        # See gensim/domains/kitchen/predicates.py::check_upright
        for prop in self._all_props:
            quat = prop.body.get_quaternion()
            q = Quaternion(quat)
            z_axis_local = q.rotate(np.array([0, 0, 1]))
            tilt_deg = np.degrees(np.arccos(np.clip(abs(z_axis_local[2]), 0, 1)))
            if tilt_deg > 15.0:
                self._success_check = False

        # Robot must not be holding any prop
        for prop in self._all_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    self._success_check = False

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_metrics,
        )
        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for prop in self._all_props:
            if prop.is_colliding(self.floor):
                return True
            if self._is_out_of_bounds(prop):
                return True
        return False

    def _is_out_of_bounds(self, prop) -> bool:
        """Check if a prop has left the counter workspace (with margin)."""
        pos = prop.body.get_position()
        xy = pos[:2]
        return bool(
            np.any(np.abs(xy - self._WORKSPACE_CENTER) > self._WORKSPACE_HALF + 0.1)
        )

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
