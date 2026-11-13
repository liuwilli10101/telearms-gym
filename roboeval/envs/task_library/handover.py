from abc import ABC

import numpy as np
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import Cube, LargeCube, Rod
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance

class CubeHandover(RoboEvalEnv, MetricRolloutEval):
    """Hand over a Rod from the left gripper to the right gripper."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    _BLOCKS_POS_BOUNDS = np.array([0.0, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCKS_STEP = 0.15
    _LIFT_DIST = 0.1


    _BLOCKS_POS = np.array([0.5, 0, 1])
    
    success_check = False
    _final_metrics = {}
    left_target_distance = 0
    right_target_distance = 0
    
    def _initialize_env(self):
        """
        Called once when the environment is constructed.
        Create the block(s) here and store references.
        """
        # Create a single block or 'Prop' (assuming you have a block or Prop class).
        # If you need to add geometry, you can do so here. For example:
        self.block = Rod(self._mojo)
        self.initial_gripper = None
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.block,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )


    def _on_reset(self):
        """
        We want to place the block in the left gripper at the start of each episode.
        """       
        self.initial_gripper = None
        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        
        for point in points:
            self.block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.block,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True     
        )
        
        self._init_z_pos = self.block.body.get_position()[2] - 0.0301
        for idx in range(1, 3): self._metric_stage(idx, False)


    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        """
        We succeed if the right gripper is holding the block, and the left is no longer holding it.
        """
        
        """Succeed only if the block is handed from one gripper to the other."""
        self.success_check = False
        right_holding = self.robot.is_gripper_holding_object(self.block, HandSide.RIGHT)
        left_holding = self.robot.is_gripper_holding_object(self.block, HandSide.LEFT)

        # Track the initial gripper if not yet recorded
        if self.initial_gripper is None:
            if left_holding:
                self.initial_gripper = HandSide.LEFT
            elif right_holding:
                self.initial_gripper = HandSide.RIGHT

        # Validate success if block is transferred to the opposite gripper
        if self.initial_gripper == HandSide.LEFT and right_holding and not left_holding:
            self.success_check = True
        elif self.initial_gripper == HandSide.RIGHT and left_holding and not right_holding:
            self.success_check = True

        # –––––––––––––– distance checking –––––––––––––––––––
        self.left_target_distance = distance(self.block.body, self.robot.grippers[HandSide.LEFT].body)
        self.right_target_distance = distance(self.block.body, self.robot.grippers[HandSide.RIGHT].body)
        self.lift_distance = self.block.body.get_position()[2] - self._init_z_pos
        
        # ––––––––––––––– stage monitoring –––––––––––––––––––––
        grasping = right_holding or left_holding
        block_transfer = self.initial_gripper == HandSide.LEFT and right_holding or self.initial_gripper == HandSide.RIGHT and left_holding
        
        if grasping: self._metric_stage(1)
        if block_transfer: self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "block to left gripper": self.left_target_distance,
                "block to right gripper": self.right_target_distance,
                "lift distance": self.lift_distance
            }
        )

        return self.success_check

    def _fail(self) -> bool:
        """
        Failure conditions. For instance, if the block is dropped to the floor or 
        if the episode times out. 
        """
        # check if the parent class has any default fail conditions
        if super()._fail():
            return True
        
        # check if the block is colliding with it (i.e., dropped).
        if self.block.is_colliding(self._floor):
            return True
        
        return False
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    