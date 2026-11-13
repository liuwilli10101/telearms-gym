"""PrepareBreakfastTray: Sequentially load a breakfast tray and move it to a target counter."""

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.kitchen_items import Plate, Mug, Apple
from roboeval.envs.props.items import BreakfastTray
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class PrepareBreakfastTrayEnv(RoboEvalEnv, MetricRolloutEval):
    """Sequentially load a breakfast tray: plate on tray, mug on tray, apple on plate, then move tray to target counter."""

    _PRESET_PATH = PRESETS_PATH / "kitchen" / "counter_with_tray.yaml"

    # Enable temporal success checking for sequential subgoals
    USE_TEMPORAL_SUCCESS = True

    _success_check = True
    _final_metrics = {}

    # Workspace bounds (broader for two-counter layout)
    _WORKSPACE_CENTER = np.array([0.55, 0.0])
    _WORKSPACE_HALF = np.array([0.275, 0.35])

    def _initialize_env(self):
        self.plate = Plate(self._mojo)
        self.mug = Mug(self._mojo)
        self.apple = Apple(self._mojo)
        self.breakfast_tray = BreakfastTray(self._mojo)

        # Get counters from preset: first = main counter, second = target counter
        tables = self._preset.get_props(Table)
        self.main_counter = tables[0]
        self.target_counter = tables[1]

        self._metric_init(
            track_slippage=True,
            slip_objects=[self.plate, self.mug, self.apple],
            robot=self.robot,
            track_collisions=True,
        )

        # Store target position for tray delivery check
        self._target_pos = None

    def _initialize_temporal_goals(self):
        """Define 4 sequential subgoals for breakfast tray preparation."""
        # Subgoal 1: Place plate on tray
        self._add_temporal_goal(
            name="plate_on_tray",
            check_fn=lambda: self.plate.is_colliding(self.breakfast_tray),
            description="Place plate on the breakfast tray",
        )
        # Subgoal 2: Place mug on tray
        self._add_temporal_goal(
            name="mug_on_tray",
            check_fn=lambda: self.mug.is_colliding(self.breakfast_tray),
            description="Place mug on the breakfast tray",
        )
        # Subgoal 3: Place apple on plate (must be centered)
        self._add_temporal_goal(
            name="apple_on_plate",
            check_fn=lambda: self._check_on_plate(self.apple, self.plate),
            description="Place apple centered on the plate",
        )
        # Subgoal 4: Move tray to target counter
        self._add_temporal_goal(
            name="tray_to_target",
            check_fn=lambda: self._check_tray_at_target(),
            description="Move loaded tray to the target counter",
        )

    def _on_reset(self):
        super()._on_reset()
        self._metric_init(
            track_slippage=True,
            slip_objects=[self.plate, self.mug, self.apple],
            robot=self.robot,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Place plate on main counter with perturbation
        plate_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.plate.body.set_position(np.array([0.55, -0.1, 1.00]) + plate_offset)
        plate_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.plate.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=plate_angle).elements
        )

        # Place mug on main counter with perturbation
        mug_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.mug.body.set_position(np.array([0.55, 0.15, 1.00]) + mug_offset)
        mug_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.mug.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=mug_angle).elements
        )

        # Place apple on main counter with perturbation
        apple_offset = rng.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])
        self.apple.body.set_position(np.array([0.45, 0.0, 0.985]) + apple_offset)
        apple_angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        self.apple.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=apple_angle).elements
        )

        # Place breakfast tray on main counter
        self.breakfast_tray.body.set_position(np.array([0.55, -0.05, 1.00]))

        # Store target counter position for tray delivery check
        self._target_pos = self.target_counter.body.get_position()

        # Initialize metric stages (one per subgoal)
        for idx in range(1, 5):
            self._metric_stage(idx, False)

    def _on_step(self):
        # CRITICAL: Call super()._on_step() FIRST to update TemporalSuccessChecker
        super()._on_step()
        self._metric_step()

    def _check_on_plate(self, item_prop, plate_prop):
        """Check if item is centered on plate (inline check_on_plate math)."""
        item_pos = item_prop.body.get_position()
        plate_pos = plate_prop.body.get_position()
        xy_dist = np.linalg.norm(item_pos[:2] - plate_pos[:2])
        return bool(xy_dist < 0.05 and item_pos[2] > plate_pos[2])

    def _check_tray_at_target(self):
        """Check if tray XY is near target counter position."""
        if self._target_pos is None:
            return False
        tray_pos = self.breakfast_tray.body.get_position()
        return bool(
            np.linalg.norm(tray_pos[:2] - self._target_pos[:2]) < 0.15
        )

    def _is_out_of_bounds(self, prop) -> bool:
        """Check if a prop has left the counter workspace (with margin)."""
        pos = prop.body.get_position()
        xy = pos[:2]
        return bool(
            np.any(np.abs(xy - self._WORKSPACE_CENTER) > self._WORKSPACE_HALF + 0.1)
        )

    def _success(self) -> bool:
        self._success_check = True

        # Check temporal sequence is complete
        temporal_complete = (
            self._temporal_checker is not None
            and self._temporal_checker.is_complete()
        )
        if not temporal_complete:
            self._success_check = False

        # Robot must not be holding any prop
        all_props = [self.plate, self.mug, self.apple, self.breakfast_tray]
        for prop in all_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    self._success_check = False

        # Stage tracking based on temporal progress
        completed, _ = self.get_temporal_progress()
        for idx in range(1, completed + 1):
            self._metric_stage(idx)

        dist_metrics = {
            "apple-plate": distance(self.apple.body, self.plate.body),
        }
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_metrics,
        )
        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        # Floor collision check
        all_props = [self.plate, self.mug, self.apple, self.breakfast_tray]
        for prop in all_props:
            if prop.is_colliding(self.floor):
                return True
            if self._is_out_of_bounds(prop):
                return True
        return False

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
