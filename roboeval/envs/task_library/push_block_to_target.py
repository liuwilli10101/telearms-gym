from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.tables import Table
from roboeval.envs.props.items import Cube, GoalPatch
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.const import HandSide


class PushBlockToTarget(RoboEvalEnv, MetricRolloutEval, ABC): 
    """
    Robot must push blocks to a target position
    """
    
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    _success_check = True
    _final_metrics = {}

    _BLOCK_COUNT = 3
    _BLOCK_POS = np.array([0.7, 0.0, 1])
    _BLOCK_ROT = np.deg2rad(180)
    _BLOCK_STEP = 0.15
    _BLOCK_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCK_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _BLOCK_ROT_BOUNDS = np.deg2rad(30)

    _TARGET_SIZE = (0.07, 0.07)  # GoalPatch uses (width, depth) tuple
    _TARGET_COLOR = "green"
    _TARGET_POS = np.array([0.5, 0, 0.95])
    # Y bound kept within the workspace [-0.25, 0.25] to avoid the patch
    # spawning off the table edge.
    _TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])
    _TARGET_ROT_BOUNDS = np.deg2rad(30)  # Target orientation randomization
    
    _TARGET_POS_THRESHOLD = 0.05  # Distance threshold for target position success
    _TARGET_ROT_THRESHOLD = np.deg2rad(10)  # Rotation threshold for target orientation success
    
    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        self.blocks = [Cube(self._mojo) for _ in range(self._BLOCK_COUNT)]
        
        # Create target using GoalPatch procedural asset
        self.target = GoalPatch(
            self._mojo,
            color=self._TARGET_COLOR,
            size=self._TARGET_SIZE,
            name="target"
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,  # watch block slips
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )
        
        # Target pose storage
        self.target_position = None
        self.target_orientation = None

    def _on_reset(self):
        rng = np.random.default_rng(self.seed)
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCK_POS,
            self._BLOCK_POS_EXTENTS,
            self._BLOCK_STEP,
            self._BLOCK_POS_BOUNDS,
            rng=rng,
        )
        for block, point in zip(self.blocks, points):
            block.body.set_position(point)
            angle = rng.uniform(-self._BLOCK_ROT_BOUNDS, self._BLOCK_ROT_BOUNDS)
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._BLOCK_ROT + angle).elements
            )

        offset = rng.uniform(-self._TARGET_POS_BOUNDS, self._TARGET_POS_BOUNDS)
        self.target_position = self._TARGET_POS + offset
        self.target.set_position(self._TARGET_POS + offset)

        target_angle = rng.uniform(-self._TARGET_ROT_BOUNDS, self._TARGET_ROT_BOUNDS)
        self.target_orientation = Quaternion(axis=[0, 0, 1], angle=target_angle)

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,  # watch block slips
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)

        
    def _on_step(self):
        self._metric_step()
    
    def _success(self):
        self._success_check = True
        obj_dist = {}
        obj_angle = {}
        for idx, block in enumerate(self.blocks):
            if not block.is_colliding(self.target):
                self._success_check = False
            # –––––––––– distance checks –––––
            obj_dist[f"block_{idx} to target"] = distance(block.body, self.target.body) 
            obj_angle[f"block_{idx} to target"] = Quaternion.absolute_distance(Quaternion(block.body.get_quaternion()), self.target_orientation)

        # —————————— stage monitoring ————————————
        grasping = any(self.robot.is_gripper_holding_object(block, side) for side in self.robot.grippers)
        target_collision = sum(block.is_colliding(self.target) for block in self.blocks)
        all_block_target_collision = all(block.is_colliding(self.target) for block in self.blocks)
        target_pos_check = any(
            np.linalg.norm(block.body.get_position() - self.target_position) < self._TARGET_POS_THRESHOLD
            for block in self.blocks
        )
        target_rot_check = any(
            Quaternion.absolute_distance(Quaternion(block.body.get_quaternion()), self.target_orientation) < self._TARGET_ROT_THRESHOLD
            for block in self.blocks
        )

        if grasping: self._metric_stage(1)
        if target_collision >= 1: self._metric_stage(2) # check whether a single block is on target
        if all_block_target_collision: self._metric_stage(3) # all blocks on target
        if target_pos_check: self._metric_stage(4) # target position check
        if target_rot_check: self._metric_stage(5) # target orientation check
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=obj_dist,
            pose_error=obj_angle,
        )
        return self._success_check
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    