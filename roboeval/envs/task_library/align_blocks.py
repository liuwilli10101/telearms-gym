# envs/task_library/align_blocks.py
# ---------------------------------
"""
Align N cubic blocks in a single straight horizontal line.
All block must (i) rest on the table, (ii) have their XY–centres aligned
within ±_ALIGN_TOL in the Y–direction, and (iii) be released.
"""

from __future__ import annotations
import numpy as np
from typing import Dict

from pyquaternion import Quaternion

from roboeval.roboeval_env           import RoboEvalEnv
from roboeval.const               import PRESETS_PATH, HandSide
from roboeval.envs.props.items    import Cube
from roboeval.envs.props.tables   import Table
from roboeval.utils.env_utils     import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


# --------------------------------------------------------------------------- #
class AlignBlocksEnv(RoboEvalEnv, MetricRolloutEval):
    """Place M blocks so that they form a straight line."""

    _PRESET_PATH     = PRESETS_PATH / "bimanual_setup.yaml"
    _NUM_BLOCKS      = 3                         # you can change this
    _SCATTER_CENTRE  = np.array([0.55, 0.0, 1.0])
    _SCATTER_EXTENTS = np.array([0.12, 0.20])    # initial X/Y spread
    _SCATTER_STEP    = 0.10                      # min spacing at reset

    _ROT_BOUNDS      = np.deg2rad([0, 0, 180])   # random yaw @ reset
    _ALIGN_TOL       = 0.03                      # ≤ 3 cm Y-deviation

    _success_check   = True
    _final_metrics: Dict[str, float] = {}

    # --------------------------------------------------------------------- #
    # INITIALISATION
    # --------------------------------------------------------------------- #
    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.table  = self._preset.get_props(Table)[0]

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

    # --------------------------------------------------------------------- #
    # RESET
    # --------------------------------------------------------------------- #
    def _on_reset(self):
        rng = np.random.default_rng(self.seed)
        pts = get_random_points_on_plane(
            self._NUM_BLOCKS,
            self._SCATTER_CENTRE,
            self._SCATTER_EXTENTS,
            self._SCATTER_STEP,
            rng=rng,
        )

        for block, p in zip(self.blocks, pts):
            block.body.set_position(p)
            yaw = rng.uniform(-self._ROT_BOUNDS[2], self._ROT_BOUNDS[2])
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=yaw).elements
            )

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)
        # Stage 1: first block placed, Stage 2: all placed, Stage 3: aligned
        for s in range(1, 3):
            self._metric_stage(s, False)

    def _on_step(self):
        self._metric_step()

    # --------------------------------------------------------------------- #
    # SUCCESS + METRICS
    # --------------------------------------------------------------------- #
    def _success(self) -> bool:
        self._success_check = True
        dist_metrics: Dict[str, float] = {}

        # --- table contact ------------------------------------------------
        on_table = [blk.is_colliding(self.table) for blk in self.blocks]
        all_on_table = all(on_table)

        # --- x-alignment --------------------------------------------------
        x_vals = np.array([blk.body.get_position()[0] for blk in self.blocks])
        x_range = float(np.ptp(x_vals))           # peak-to-peak spread
        aligned = x_range < self._ALIGN_TOL
        dist_metrics["x_ptp"] = x_range
        
        # --- none grasped -------------------------------------------------
        held = any(
            self.robot.is_gripper_holding_object(blk, side)
            for blk in self.blocks for side in self.robot.grippers
        )
        
        if held: self._metric_stage(1)  # block(s) held by gripper(s)
        if aligned: self._metric_stage(2)  # blocks horizontally aligned

        self._success_check = aligned and not held and all_on_table

        # --- fine-grained metrics ----------------------------------------
        # distance from each block to each gripper  (optional but useful)
        for i, blk in enumerate(self.blocks):
            for side in self.robot.grippers:
                dist_metrics[f"block_{i}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, blk.body)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_metrics,
        )
        return self._success_check

    # --------------------------------------------------------------------- #
    def _get_task_info(self):
        """Expose metrics every step or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
    # --------------------------------------------------------------------- #
    def _on_step(self):
        self._metric_step()

    # --------------------------------------------------------------------- #
    def _fail(self) -> bool:
        """Fail if any block falls off the table/floor (optional)."""
        if super()._fail():
            return True
        return False
