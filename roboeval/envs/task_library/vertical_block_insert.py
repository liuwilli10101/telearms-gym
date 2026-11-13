# envs/task_library/vertical_insert.py
# ------------------------------------
import numpy as np
from pyquaternion import Quaternion

from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import Cube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils  import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.roboeval_env import RoboEvalEnv

# --------------------------------------------------------------------------- #
class _TwoBlockBaseEnv(RoboEvalEnv):
    """Common two-block set-up (side-by-side start)."""
    _PRESET_PATH      = PRESETS_PATH / "bimanual_setup.yaml"
    _NUM_BLOCKS        = 2
    _BLOCKS_CENTRE     = np.array([0.55, 0.0, 1.0])
    _BLOCKS_EXTENTS    = np.array([0.12, 0.20])      # x/y scattering
    _BLOCKS_SEPARATION = 0.12                        # ≥ 12 cm apart initially

    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.table = self._preset.get_props(Table)[0]

    def _on_reset(self):
        # deterministic RNG so demos can be replayed
        rng = np.random.default_rng(self.seed)
        pts = get_random_points_on_plane(
            self._NUM_BLOCKS,
            self._BLOCKS_CENTRE,
            self._BLOCKS_EXTENTS,
            self._BLOCKS_SEPARATION,
            rng=rng,
        )
        for block, p in zip(self.blocks, pts):
            block.set_pose(p)                        # axis-aligned start


# --------------------------------------------------------------------------- #
class VerticalInsertTwoBlocksEnv(_TwoBlockBaseEnv, MetricRolloutEval):
    """
    Task: lift one block, slide the second block underneath, then lower the
    first so that it ends up stacked.  (Vertical insertion)

    Stage-1  – any block lifted by ≥ _LIFT_THRESH
    Stage-2  – other block is positioned underneath (table contact, x-y overlap)
    Stage-3  – lifted block placed on top (stack satisfied, no block grasped)
    """

    _LIFT_THRESH   = 0.05     # ≥ 5 cm counts as “lifted”
    _XY_TOL        = 0.03     # centres must coincide within 3 cm
    _success_check = True
    _final_metrics = {}

    # ------------- metric buffers -------------------------------------- #
    def _initialize_env(self):
        super()._initialize_env()
        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

    def _on_reset(self):
        super()._on_reset()
        self._init_z = [c.body.get_position()[2] for c in self.blocks]
        # Roles are filled in by _success() once exactly one block is lifted.
        self.lifted_block = None
        self.ground_block = None
        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)
        for s in range(1, 4):
            self._metric_stage(s, False)

    def _on_step(self):
        self._metric_step()

    # ------------- helper utilities ------------------------------------ #
    @staticmethod
    def _block_height(block):
        return block.body.get_position()[2]

    @staticmethod
    def _xy_dist(b1, b2):
        p1 = b1.body.get_position()[:2]
        p2 = b2.body.get_position()[:2]
        return float(np.linalg.norm(p1 - p2))

    # ------------- success logic --------------------------------------- #
    def _success(self) -> bool:
        self._success_check = True
        b0, b1 = self.blocks
        z0, z1 = self._block_height(b0), self._block_height(b1)

        grasped = any(self.robot.is_gripper_holding_object(block, side)
                      for side in self.robot.grippers
                      for block in self.blocks)

        # --- identify which block is being lifted -----------------------
        lift0 = z0 - self._init_z[0] > self._LIFT_THRESH
        lift1 = z1 - self._init_z[1] > self._LIFT_THRESH

        # Exactly one block must be lifted; otherwise the role assignment is
        # ambiguous and intermediate stages cannot fire.
        exactly_one_lifted = lift0 ^ lift1
        if exactly_one_lifted:
            self.lifted_block = b0 if lift0 else b1
            self.ground_block = b1 if lift0 else b0
        else:
            self.lifted_block = None
            self.ground_block = None

        # --- check ground block slid under ------------------------------
        under_ok = (
            exactly_one_lifted
            and self.ground_block.is_colliding(self.table)
            and self._xy_dist(self.lifted_block, self.ground_block) < self._XY_TOL
        )

        # --- final stack ------------------------------------------------
        # Final state: one block resting on the other, no gripper engaged.
        # Identify top/bottom by Z so we don't depend on the (now-cleared)
        # lifted/ground assignment from the in-progress stage.
        top, bottom = (b0, b1) if z0 > z1 else (b1, b0)
        stacked_ok = (
            top.is_colliding(bottom)
            and not top.is_colliding(self.table)
            and bottom.is_colliding(self.table)
            and not any(self.robot.is_gripper_holding_object(block, side)
                        for side in self.robot.grippers
                        for block in self.blocks)
        )

        if grasped: self._metric_stage(1)
        if under_ok: self._metric_stage(2)
        if stacked_ok: self._metric_stage(3)

        # --- fine-grained metrics --------------------------------------
        dist_metrics = {
            "lift dist block 0": z0 - self._init_z[0],
            "lift dist block 1": z1 - self._init_z[1],
            "xy dist": self._xy_dist(b0, b1),
        }
        self._success_check = stacked_ok
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_metrics,
        )
        return self._success_check

    # expose metrics every step (optional):
    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})
    
    def _fail(self) -> bool:
        if super()._fail():
            return True
        # ground_block is only assigned while exactly one block is lifted; before
        # that point there is no "ground" to fall off the table.
        if self.ground_block is not None and not self.ground_block.is_colliding(self.table):
            return True
        return False
