from abc import ABC
import numpy as np
from pyquaternion import Quaternion

from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.living_room import SmallTable, NightStand, Broom, WireBrush, Dustpan
from roboeval.envs.props.tables import Table
from roboeval.envs.props.items import Cube, SmallCube
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.envs.props.prop import Prop
from roboeval.utils.metric_rollout import MetricRolloutEval

class _SweepEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base environment class for sweeping debris tasks."""

    _PRESET_PATH = PRESETS_PATH / "sweep.yaml"
    _success_check = True
    _final_metrics = {}
    
    def _initialize_env(self):
        """Initialize environment with basic elements."""
        self.table = self._preset.get_props(Table)[0]
        
    def _get_task_info(self):
        """Return metrics for the task.
        
        Returns:
            dict: Task metrics collected during the episode.
        """
        return getattr(self, "_final_metrics", {})
        
class SweepFloor(RoboEvalEnv, ABC, MetricRolloutEval): 
    """Environment where robot must pick up a broom and sweep cubes into a dustpan."""
    
    # Cube configuration
    _CUBE_COUNT = 3
    _CUBE_POS = np.array([0.7, 0, 0.0])
    _CUBE_ROT = np.deg2rad(180)
    _CUBE_STEP = 0.15
    _CUBE_POS_EXTENTS = np.array([0.1, 0.25])
    _CUBE_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _CUBE_ROT_BOUNDS = np.deg2rad(30)
    
    # Tool positions
    _BROOM_POS = np.array([0.5, 0.2, 0.3])
    _DUSTPAN_POS = np.array([0.5, -0.2, 0.3])
    
    _success_check = True
    _final_metrics = {}
    
    def _initialize_env(self):
        """Initialize environment with broom, dustpan and cubes."""
        super()._initialize_env()
        self.broom = Broom(self._mojo)
        self.dustpan = Dustpan(self._mojo)
        self.cubes = [SmallCube(self._mojo) for _ in range(self._CUBE_COUNT)]
        
        self._setup_metrics()

    def _setup_metrics(self):
        """Set up metrics tracking for the environment."""
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.broom, self.dustpan],
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )

    def _on_reset(self):
        """Reset environment state for a new episode."""
        # Place cubes at random positions
        points = get_random_points_on_plane(
            len(self.cubes),
            self._CUBE_POS,
            self._CUBE_POS_EXTENTS,
            self._CUBE_STEP,
            self._CUBE_POS_BOUNDS,
        )
        for cube, point in zip(self.cubes, points):
            cube.body.set_position(point)
            angle = np.random.uniform(-self._CUBE_ROT_BOUNDS, self._CUBE_ROT_BOUNDS)
            angle = 0
            cube.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._CUBE_ROT + angle).elements
            )
        
        # Set tool positions
        self.broom.body.set_position(self._BROOM_POS)
        self.broom.body.set_quaternion(
            Quaternion(axis=[1, 0, 0], angle=np.deg2rad(90)).elements
        )
        self.dustpan.body.set_position(self._DUSTPAN_POS)
        
        self._setup_metrics()
        
    def _on_step(self):
        """Update metrics on each environment step."""
        self._metric_step()
    
    def _success(self):
        """Check if the task is successfully completed.
        
        Returns:
            bool: True if all success criteria are met, False otherwise.
        """
        self._success_check = True
        
        # Check if all cubes are in the dustpan and not on the floor
        for cube in self.cubes:         
            if not cube.is_colliding(self.dustpan) or cube.is_colliding(self._floor):
                self._success_check = False

        # Check if robot is holding the broom
        if not any(self.robot.is_gripper_holding_object(self.broom, side) for side in self.robot.grippers):
            self._success_check = False
            
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check
        )
        return self._success_check
    
    def _get_task_info(self):
        """Return metrics for the task.
        
        Returns:
            dict: Task metrics collected during the episode.
        """
        return getattr(self, "_final_metrics", {})
    
class SweepTable(_SweepEnv): 
    """Environment where robot must use wire brushes to sweep cubes into a target area on a table."""
    
    # Cube configuration
    _CUBE_COUNT = 3
    _CUBE_POS = np.array([0.7, 0.0, 1])
    _CUBE_ROT = np.deg2rad(180)
    _CUBE_STEP = 0.15
    _CUBE_POS_EXTENTS = np.array([0.1, 0.25])
    _CUBE_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _CUBE_ROT_BOUNDS = np.deg2rad(30)
    
    # Tool positions
    _BRUSH_POS_R = np.array([0.65, -0.25, 1])
    _BRUSH_POS_L = np.array([0.65, 0.25, 1])
    _DUSTPAN_POS = np.array([0.6, 0.35, 1])
    _DUSTPAN_ROT = np.deg2rad(-90)

    # Target area configuration
    _TARGET_SIZE = np.array([0.07, 0.07, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([0.7, 0, 0.95])
    _TARGET_POS_BOUNDS = np.array([0.05, 0.4, 0.0])
    
    def _initialize_env(self):
        """Initialize environment with brushes, cubes and target area."""
        super()._initialize_env()
        self.wire_brush_r = WireBrush(self._mojo)
        self.wire_brush_l = WireBrush(self._mojo)
        self.cubes = [SmallCube(self._mojo) for _ in range(self._CUBE_COUNT)]
        
        # Create target area
        self.target = Body.create(self._mojo)
        self.target_collider = Geom.create(
            self._mojo,
            parent=self.target,
            geom_type=GeomType.BOX,
            size=self._TARGET_SIZE,
            color=self._TARGET_COLOR,
        )
        
        self._setup_metrics()

    def _setup_metrics(self):
        """Set up metrics tracking for the environment."""
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.wire_brush_r, self.wire_brush_l],
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True      
        )

    def _on_reset(self):
        """Reset environment state for a new episode."""
        # Place cubes at random positions
        points = get_random_points_on_plane(
            len(self.cubes),
            self._CUBE_POS,
            self._CUBE_POS_EXTENTS,
            self._CUBE_STEP,
            self._CUBE_POS_BOUNDS,
        )
        for cube, point in zip(self.cubes, points):
            cube.body.set_position(point)
            angle = np.random.uniform(-self._CUBE_ROT_BOUNDS, self._CUBE_ROT_BOUNDS)
            angle = 0
            cube.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._CUBE_ROT + angle).elements
            )
        
        # Set tool positions
        self.wire_brush_r.body.set_position(self._BRUSH_POS_R)
        self.wire_brush_l.body.set_position(self._BRUSH_POS_L)
        self.target.set_position(self._TARGET_POS)
        
        self._setup_metrics()
        
        # Initialize all stages as not completed
        for idx in range(1, 5):
            self._metric_stage(idx, False)
        
    def _on_step(self):
        """Update metrics on each environment step."""
        self._metric_step()
    
    def _success(self):
        """Check if the task is successfully completed.
        
        Returns:
            bool: True if all success criteria are met, False otherwise.
        """
        self._success_check = True
        obj_dist = {}
        
        # Check if all cubes are on the target and not on the floor
        for idx, cube in enumerate(self.cubes):
            if not cube.is_colliding(self.target_collider) or cube.is_colliding(self._floor):
                self._success_check = False
            obj_dist[f"cube_{idx}"] = distance(cube.body, self.target)
        
        # Track distances between grippers and brushes
        obj_dist["gripper-left brush"] = distance(self.wire_brush_l.body, self.robot.grippers[HandSide.LEFT].body)
        obj_dist["gripper-right brush"] = distance(self.wire_brush_r.body, self.robot.grippers[HandSide.RIGHT].body)

        # Track task completion stages
        grasping_left = self.robot.is_gripper_holding_object(self.wire_brush_l, HandSide.LEFT)
        grasping_right = self.robot.is_gripper_holding_object(self.wire_brush_r, HandSide.RIGHT)
        target_collision = sum(cube.is_colliding(self.target_collider) for cube in self.cubes)
        all_cube_target_collision = all(cube.is_colliding(self.target_collider) for cube in self.cubes)

        if grasping_left:
            self._metric_stage(1)
        if grasping_right:
            self._metric_stage(2)
        if target_collision >= 2:
            self._metric_stage(3)
        if all_cube_target_collision:
            self._metric_stage(4)
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=obj_dist
        )
        return self._success_check
