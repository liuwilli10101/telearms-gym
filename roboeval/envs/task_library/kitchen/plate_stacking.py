"""PlateStacking: Stack plates in a neat pile on the kitchen counter."""

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.kitchen_items import Plate
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class PlateStackingEnv(RoboEvalEnv, MetricRolloutEval):
    """Stack three plates in a neat pile on the kitchen counter."""

    _PRESET_PATH = PRESETS_PATH / "kitchen" / "counter_workspace.yaml"

    _NUM_PLATES = 3
    _success_check = True
    _final_metrics = {}

    # Workspace bounds from counter_workspace.yaml
    _WORKSPACE_CENTER = np.array([0.55, 0.0])
    _WORKSPACE_HALF = np.array([0.275, 0.275])

    def _initialize_env(self):
        self.plates = [Plate(self._mojo) for _ in range(self._NUM_PLATES)]
        self.table = self._preset.get_props(Table)[0]

        self._metric_init(
            track_slippage=True,
            slip_objects=self.plates,
            robot=self.robot,
            track_collisions=True,
        )

    def _on_reset(self):
        super()._on_reset()
        self._metric_init(
            track_slippage=True,
            slip_objects=self.plates,
            robot=self.robot,
            track_collisions=True,
        )

        rng = np.random.default_rng(self.seed)

        # Spread plates across the counter surface with perturbation
        # Plates are r=0.114 discs: centers must stay > 0.228 apart or a
        # falling plate lands on a neighbour's rim and tips onto its edge.
        # Spacing 0.335 with +-0.02 jitter keeps the worst case at 0.278,
        # and every rim stays on the tabletop (x 0.30-0.90, y +-0.60).
        base_positions = [
            np.array([0.45, -0.30, 1.00]),
            np.array([0.60, 0.0, 1.00]),
            np.array([0.75, 0.30, 1.00]),
        ]
        for plate, base_pos in zip(self.plates, base_positions):
            offset = rng.uniform([-0.02, -0.02, 0], [0.02, 0.02, 0])
            plate.body.set_position(base_pos + offset)
            angle = rng.uniform(-np.deg2rad(30), np.deg2rad(30))
            plate.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )

        # Initialize metric stages (one per plate placed in stack)
        for idx in range(1, self._NUM_PLATES + 1):
            self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True

        # Sort plates by Z height (bottom to top)
        plates_sorted = sorted(
            self.plates, key=lambda p: p.body.get_position()[2]
        )

        # Bottom plate must be on the counter surface
        if not plates_sorted[0].is_colliding(self.table.table_top):
            self._success_check = False

        # Each upper plate must be colliding with the plate below it
        for i in range(1, len(plates_sorted)):
            if not plates_sorted[i].is_colliding(plates_sorted[i - 1]):
                self._success_check = False

        # All plates must be upright (inline check_upright logic)
        # See gensim/domains/kitchen/predicates.py::check_upright
        for plate in plates_sorted:
            quat = plate.body.get_quaternion()
            q = Quaternion(quat)
            # Tilt angle: angle between plate's Z-axis and world Z-axis
            z_axis_local = q.rotate(np.array([0, 0, 1]))
            tilt_deg = np.degrees(np.arccos(np.clip(abs(z_axis_local[2]), 0, 1)))
            if tilt_deg > 15.0:
                self._success_check = False

        # Robot must not be holding any plate
        for plate in self.plates:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    self._success_check = False

        # Stage tracking: count how many plates are stacked
        if plates_sorted[0].is_colliding(self.table.table_top):
            self._metric_stage(1)
        if len(plates_sorted) > 1 and plates_sorted[1].is_colliding(plates_sorted[0]):
            self._metric_stage(2)
        if len(plates_sorted) > 2 and plates_sorted[2].is_colliding(plates_sorted[1]):
            self._metric_stage(3)

        # Distance metrics
        dist_metrics = {}
        for i, plate in enumerate(self.plates):
            for side in self.robot.grippers:
                dist_metrics[f"plate_{i}-gripper_{side.value}"] = distance(
                    self.robot.grippers[side].body, plate.body
                )

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_metrics,
        )
        return self._success_check

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for plate in self.plates:
            if plate.is_colliding(self.floor):
                return True
            if self._is_out_of_bounds(plate):
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
