"""Manipulation tasks."""
from abc import ABC

import numpy as np
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import Cube
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


class StackTwoBlocksEnv(RoboEvalEnv, MetricRolloutEval):
    """Stack two blocks on top of each other in random positions"""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    
    _NUM_BLOCKS = 2
    _BLOCKS_POS = np.array([0.5, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])

    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_ROT_BOUNDS_SINGLE = np.deg2rad(180)

    _success_check = True
    _final_metrics = {}
    _SUCCESS_DIST = 0.1
    _BLOCKS_STEP = 0.05

    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.table = self._preset.get_props(Table)[0]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True       
        )
    def _on_step(self):
        self._metric_step()
        
    def _on_reset(self):
        super()._on_reset()
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True       
        )
        
        rng = np.random.default_rng(self.seed)
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
            rng=rng,
        )
        for block, point in zip(self.blocks, points):
            block.body.set_position(point)
            angle = rng.uniform(-self._BLOCKS_ROT_BOUNDS_SINGLE, self._BLOCKS_ROT_BOUNDS_SINGLE)
        
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )
        
        self._init_z_1_pos = self.blocks[0].body.get_position()[2] - 0.0301
        self._init_z_2_pos = self.blocks[1].body.get_position()[2] - 0.0301
        for idx in range(1, 4): self._metric_stage(idx, False)

        
    def _success(self) -> bool:
        self._success_check = True
        blocks_sorted = sorted(self.blocks, key=lambda b: b.body.get_position()[2])
        dist_blocks = {}
        
        if not blocks_sorted[0].is_colliding(self.table.table_top):
            self._success_check = False
        if not blocks_sorted[1].is_colliding(blocks_sorted[0]) or blocks_sorted[1].is_colliding(self.table.table_top):
            self._success_check = False

        for idx, block in enumerate(self.blocks):
            for side in self.robot.grippers:
                dist_blocks[f"block_{idx}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, block.body)
                if self.robot.is_gripper_holding_object(block, side):
                    self._success_check = False
                    
         # –––––––––––distance checking –––––––––––––
        block_1_lift_distance = self.blocks[0].body.get_position()[2] - self._init_z_1_pos
        block_2_lift_distance = self.blocks[1].body.get_position()[2] - self._init_z_2_pos
        
        dist_blocks.update(
            {
                "block 1 lift distance": block_1_lift_distance,
                "block 2 lift distance": block_2_lift_distance
            }
        )
        
        # ––––––––––– stage monitoring ––––––––––––––
        grasping_block_1 = any(self.robot.is_gripper_holding_object(self.blocks[0], side) for side in self.robot.grippers)
        grasping_block_2 = any(self.robot.is_gripper_holding_object(self.blocks[1], side) for side in self.robot.grippers)
        block_stack = blocks_sorted[0].is_colliding(self.table) and blocks_sorted[1].is_colliding(blocks_sorted[0])
        block_table_check = blocks_sorted[1].is_colliding(self.table)

        if grasping_block_1 or grasping_block_2: self._metric_stage(1)
        # Stage 2: both blocks held simultaneously (the second gripper has
        # picked up the partner of the first-grasped block).
        if grasping_block_1 and grasping_block_2: self._metric_stage(2)
        if block_stack and not block_table_check: self._metric_stage(3)
                    
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=dist_blocks
        )
        return self._success_check  

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for block in self.blocks:
            if block.is_colliding(self.floor):
                return True
        return False

    def _get_task_info(self):   
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
        