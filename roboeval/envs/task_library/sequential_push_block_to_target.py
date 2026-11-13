from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.tables import Table
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.utils.metric_rollout import MetricRolloutEval


class SequentialPushBlockToTarget(RoboEvalEnv, MetricRolloutEval, ABC): 
    """
    Robot must sequentially push red block to target, then blue block to target
    
    Sequential Sub-Goals (Temporal):
        1. Push red block to red target
        2. Push blue block to blue target
    """
    
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    
    # Enable temporal success checking
    USE_TEMPORAL_SUCCESS = True
    
    _success_check = True
    _final_metrics = {}

    # Block spawn configuration
    _BLOCK_POS = np.array([0.7, 0.0, 1])
    _BLOCK_ROT = np.deg2rad(180)
    _BLOCK_STEP = 0.15
    _BLOCK_POS_EXTENTS = np.array([0.1, 0.25])
    _BLOCK_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _BLOCK_ROT_BOUNDS = np.deg2rad(30)

    # Red target configuration
    _RED_TARGET_SIZE = np.array([0.07, 0.07, 0.001])
    _RED_TARGET_COLOR = np.array([0.8, 0.3, 0.3, 1.0])  # Red
    _RED_TARGET_POS = np.array([0.5, 0.2, 0.95])
    _RED_TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])
    _RED_TARGET_ROT_BOUNDS = np.deg2rad(30)
    
    # Blue target configuration
    _BLUE_TARGET_SIZE = np.array([0.07, 0.07, 0.001])
    _BLUE_TARGET_COLOR = np.array([0.3, 0.3, 0.8, 1.0])  # Blue
    _BLUE_TARGET_POS = np.array([0.5, -0.2, 0.95])
    _BLUE_TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])
    _BLUE_TARGET_ROT_BOUNDS = np.deg2rad(30)
    
    _TARGET_POS_THRESHOLD = 0.05  # Distance threshold for target position success
    _TARGET_ROT_THRESHOLD = np.deg2rad(10)  # Rotation threshold for target orientation success
    
    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        
        # Create colored blocks
        self.red_block = ColoredCube(self._mojo)
        self.red_block.set_color("red")
        
        self.blue_block = ColoredCube(self._mojo)
        self.blue_block.set_color("blue")
        
        self.blocks = [self.red_block, self.blue_block]

        # Create red target using GoalPatch
        self.red_target = GoalPatch(
            self._mojo,
            color="red",
            size=(self._RED_TARGET_SIZE[0], self._RED_TARGET_SIZE[1]),
            name="red_target"
        )
        
        # Create blue target using GoalPatch
        self.blue_target = GoalPatch(
            self._mojo,
            color="blue",
            size=(self._BLUE_TARGET_SIZE[0], self._BLUE_TARGET_SIZE[1]),
            name="blue_target"
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )
        
        # Target pose storage
        self.red_target_position = None
        self.red_target_orientation = None
        self.blue_target_position = None
        self.blue_target_orientation = None

    def _initialize_temporal_goals(self):
        """Define temporal sequence: red block first, then blue block."""
        # Stage 1: Push red block to red target
        self._add_temporal_goal(
            name="push_red_to_target",
            check_fn=lambda: self._is_block_on_target(self.red_block, self.red_target),
            description="Push red block to red target"   
        )
        
        # Stage 2: Push blue block to blue target
        self._add_temporal_goal(
            name="push_blue_to_target",
            check_fn=lambda: self._is_block_on_target(self.blue_block, self.blue_target),
            description="Push blue block to blue target"
        )

    def _on_reset(self):
        rng = np.random.default_rng(self.seed)
        # Spawn blocks at random positions
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCK_POS,
            self._BLOCK_POS_EXTENTS,
            self._BLOCK_STEP,
            self._BLOCK_POS_BOUNDS,
            rng=rng,
        )

        # Set colors of the blocks
        self.red_block.set_color("red")
        self.blue_block.set_color("blue")

        for block, point in zip(self.blocks, points):
            block.body.set_position(point)
            angle = rng.uniform(-self._BLOCK_ROT_BOUNDS, self._BLOCK_ROT_BOUNDS)
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._BLOCK_ROT + angle).elements
            )

        # Set red target position
        offset = rng.uniform(-self._RED_TARGET_POS_BOUNDS, self._RED_TARGET_POS_BOUNDS)
        self.red_target_position = self._RED_TARGET_POS + offset
        self.red_target.set_position(self.red_target_position)

        target_angle = rng.uniform(-self._RED_TARGET_ROT_BOUNDS, self._RED_TARGET_ROT_BOUNDS)
        self.red_target_orientation = Quaternion(axis=[0, 0, 1], angle=target_angle)
        self.red_target.body.set_quaternion(self.red_target_orientation.elements)

        # Set blue target position
        offset = rng.uniform(-self._BLUE_TARGET_POS_BOUNDS, self._BLUE_TARGET_POS_BOUNDS)
        self.blue_target_position = self._BLUE_TARGET_POS + offset
        self.blue_target.set_position(self.blue_target_position)

        target_angle = rng.uniform(-self._BLUE_TARGET_ROT_BOUNDS, self._BLUE_TARGET_ROT_BOUNDS)
        self.blue_target_orientation = Quaternion(axis=[0, 0, 1], angle=target_angle)
        self.blue_target.body.set_quaternion(self.blue_target_orientation.elements)

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )
    
        for idx in range(1, 6): 
            self._metric_stage(idx, False)

        
    def _on_step(self):
        # ✅ CRITICAL: Call super() to update temporal checker
        super()._on_step()
        
        # Update per-step metrics
        self._metric_step()

        self._temporal_checker.print_status()
        
        # Track stages for analytics (independent of temporal)
        if self.red_block.is_colliding(self.red_target.collider):
            self._metric_stage(1)  # Red block touching red target
            
        if self._is_block_on_target(self.red_block, self.red_target):
            self._metric_stage(2)  # Red block on red target (temporal goal 1)
            
        if self.blue_block.is_colliding(self.blue_target.collider):
            self._metric_stage(3)  # Blue block touching blue target
            
        if self._is_block_on_target(self.blue_block, self.blue_target):
            self._metric_stage(4)  # Blue block on blue target (temporal goal 2)
    
    def _success(self):
        """
        Success requires temporal sequence complete AND both blocks on targets.
        
        This demonstrates modular composition:
        - Temporal: Enforces red-first, then-blue sequence
        - Standard: Verifies both blocks are properly positioned
        """
        # Check temporal sequence (updated in super()._on_step())
        temporal_complete = (
            self._temporal_checker is not None 
            and self._temporal_checker.is_complete()
        )
        
        # Standard success checks
        self._success_check = True
        obj_dist = {}
        obj_angle = {}
        
        # Check red block
        if not self.red_block.is_colliding(self.red_target.collider):
            self._success_check = False
        obj_dist["red_block to red_target"] = distance(self.red_block.body, self.red_target.body)
        obj_angle["red_block to red_target"] = Quaternion.absolute_distance(
            Quaternion(self.red_block.body.get_quaternion()),
            self.red_target_orientation
        )

        # Check blue block
        if not self.blue_block.is_colliding(self.blue_target.collider):
            self._success_check = False
        obj_dist["blue_block to blue_target"] = distance(self.blue_block.body, self.blue_target.body)
        obj_angle["blue_block to blue_target"] = Quaternion.absolute_distance(
            Quaternion(self.blue_block.body.get_quaternion()), 
            self.blue_target_orientation
        )
        
        # Additional checks for precise positioning
        red_pos_check = np.linalg.norm(
            self.red_block.body.get_position() - self.red_target_position
        ) < self._TARGET_POS_THRESHOLD
        
        blue_pos_check = np.linalg.norm(
            self.blue_block.body.get_position() - self.blue_target_position
        ) < self._TARGET_POS_THRESHOLD
        
        if red_pos_check and blue_pos_check:
            self._metric_stage(5)  # Both blocks precisely positioned

        # Success requires both temporal sequence AND standard checks. When
        # USE_TEMPORAL_SUCCESS is off, the gate degrades to the standard check.
        if self.USE_TEMPORAL_SUCCESS and not temporal_complete:
            self._success_check = False

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=obj_dist,
            pose_error=obj_angle,
        )

        # Explicitly add temporal success to final metrics
        if self.USE_TEMPORAL_SUCCESS:
            self._final_metrics["temporal_success"] = float(temporal_complete)

        return self._success_check
    
    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        """Check if a block is properly positioned on its target."""
        # Block must be colliding with target
        if not block.is_colliding(target.collider):
            return False
        
        # Block must not be on floor
        if block.is_colliding(self._floor):
            return False
        
        return True
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        info = getattr(self, "_final_metrics", {})
        
        # Add temporal progress
        if self.USE_TEMPORAL_SUCCESS and self._temporal_checker:
            completed, total = self.get_temporal_progress()
            info["temporal_progress"] = f"{completed}/{total}"
            info["temporal_complete"] = self._temporal_checker.is_complete()
        
        return info
