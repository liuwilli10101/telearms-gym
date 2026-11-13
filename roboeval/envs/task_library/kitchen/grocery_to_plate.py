"""GroceryToPlate: Pick a grocery item and place it on a plate on the counter."""

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.kitchen_items import Apple, Plate
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class GroceryToPlateEnv(RoboEvalEnv, MetricRolloutEval):
    """Pick a grocery item (apple) and place it on a plate on the kitchen counter."""

    _PRESET_PATH = PRESETS_PATH / "kitchen" / "counter_workspace.yaml"

    _success_check = True
    _final_metrics = {}

    # Workspace bounds from counter_workspace.yaml
    _WORKSPACE_CENTER = np.array([0.55, 0.0])
    _WORKSPACE_HALF = np.array([0.275, 0.275])

    def _initialize_env(self):
        self.apple = Apple(self._mojo)
        self.plate = Plate(self._mojo)
        self.table = self._preset.get_props(Table)[0]

        self._metric_init(
            track_slippage=True,
            slip_objects=[self.apple],
            robot=self.robot,
            track_collisions=True,
        )

    def _on_reset(self):
        super()._on_reset()
        self._metric_init(
            track_slippage=True,
            slip_objects=[self.apple],
            robot=self.robot,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Place plate on counter with small XY perturbation
        plate_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.plate.body.set_position(np.array([0.55, 0.1, 1.00]) + plate_offset)
        plate_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.plate.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=plate_angle).elements
        )

        # Place apple on counter away from plate with XY + rotation perturbation
        apple_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.apple.body.set_position(np.array([0.55, -0.15, 0.985]) + apple_offset)
        apple_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.apple.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=apple_angle).elements
        )

        # Initialize metric stages
        for idx in range(1, 3):
            self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True

        apple_pos = self.apple.body.get_position()
        plate_pos = self.plate.body.get_position()

        # Inline check_on_plate logic (see gensim/domains/kitchen/predicates.py)
        # Apple must be within XY tolerance of plate center and above the plate
        xy_dist = np.linalg.norm(apple_pos[:2] - plate_pos[:2])
        if not (xy_dist < 0.05 and apple_pos[2] > plate_pos[2]):
            self._success_check = False

        # Robot must not be holding the apple
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.apple, side):
                self._success_check = False

        # Stage tracking
        grasping_apple = any(
            self.robot.is_gripper_holding_object(self.apple, side)
            for side in self.robot.grippers
        )
        if grasping_apple:
            self._metric_stage(1)
        if xy_dist < 0.05 and apple_pos[2] > plate_pos[2]:
            self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"apple-plate": distance(self.apple.body, self.plate.body)},
        )
        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        # Floor collision check
        if self.apple.is_colliding(self.floor):
            return True
        if self.plate.is_colliding(self.floor):
            return True
        # Workspace bounds check
        if self._is_out_of_bounds(self.apple) or self._is_out_of_bounds(self.plate):
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
