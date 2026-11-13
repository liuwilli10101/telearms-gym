from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.package import Package
from roboeval.envs.props.items import Cube
from roboeval.envs.props.cabintets import BaseCabinet
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.envs.props.ycb import YcbProp
from roboeval.utils.metric_rollout import MetricRolloutEval


class _PackingCubesEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base env for packing/unpacking cubes"""

    _PRESET_PATH = PRESETS_PATH / "packaging.yaml"


    _CUBE_COUNT = 3

    # _CUBE_POS = np.array([1.2, 0, 1.7])
    _TOLERANCE = .1
    _CUBE_ROT = np.deg2rad(180)
    _CUBE_STEP = 0.15
    _CUBE_POS_EXTENTS = np.array([0.1, 0.25])
    _CUBE_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _CUBE_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.05
    _BOX_POS = np.array([0.7, 0, 1.05])
    
    _success_check = True
    _final_metrics = {}


    def _initialize_env(self):
        self.packing_box = self._preset.get_props(Package)[0]
        self.table = self._preset.get_props(BaseCabinet)[0]
        self.cubes = [Cube(self._mojo) for _ in range(self._CUBE_COUNT)]
        
        objs = [self.packing_box]
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=objs.extend(self.cubes),
            track_collisions=True,
            robot=self.robot,         
        )
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})  

 
class UnpackCubes(_PackingCubesEnv):
    """
    Given cubes inside the packing box, robot effectors must move the cubes to the counter
    """
    _CUBE_POS = np.array([0.6, 0.3, 1.1])
    _CUBE_POS_EXTENTS = np.array([0.1, 0.25]) # TODO: fix the XML combined mesh for base is does not outline a concave geometry

    def _on_reset(self):
        self.packing_box.set_state(state=0.2)
        self.packing_box.set_state(target_state={ 
            "packaging_box/joint_0": 0.2,
            "packaging_box/joint_1": 0.2,
        }) # Start at fully closed joints

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
            
        slip_objs = [self.packing_box] + self.cubes

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=slip_objs,
            track_collisions=True,
            robot=self.robot,         
        )
    
    def _success(self) -> bool:
        self._success_check = True
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for cube in self.cubes:
            if not cube.is_colliding(self.table.counter):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3)
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cube, side):
                    self._success_check = False
                    self._metric_stage(1)
        if not np.allclose(self.packing_box.get_state(), 1, atol=self._TOLERANCE): # Check if the box is fully opened
            self._success_check = False
            self._metric_stage(2, False)
        else:
            self._metric_stage(2)
            
        self._final_metrics = self._metric_finalize(self._success_check)
        return self._success_check
    
    def _on_step(self):
        self._metric_step()
    
class PackCubes(_PackingCubesEnv):
    """
    Given cubes outside the packing box, robot effectors must move the cubes inside the packing box
    """
    _CUBE_POS = np.array([0.5, -.3, 1])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

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
            
        objs = [self.packing_box]
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=objs.extend(self.cubes),
            track_collisions=True,
            robot=self.robot,         
        )
    
    def _success(self) -> bool:
        self._success_check = True
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for cube in self.cubes:         
            # might want to add a site that encompasses the inside of the box
            if not cube.is_colliding(self.packing_box.base):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3)
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cube, side):
                    self._success_check = False
                    self._metric_stage(2)
        
        # Close all lid joints of the box 
        if not np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE):
            self._success_check = False
            self._metric_stage(1, False)
        else:
            self._metric_stage(1)
            
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check
        )
            
        return self._success_check
    
    def _on_step(self):
        self._metric_step()

class PackBox(RoboEvalEnv, ABC, MetricRolloutEval):
    """
    Base pack box environment (static)
    Success: Robot must use effectors to close box lid
    """
    _PRESET_PATH = PRESETS_PATH / "packaging_box.yaml"

    _BOX_POS = np.array([0.6, 0.3, 1.05])
    _BOX_POS_EXTENTS = np.array([0.1, 0.14])
    _BOX_POS_BOUNDS = np.array([0.001, 0.001, 0])
    _BOX_ROT_BOUNDS = np.deg2rad(30)
    _BOX_ROT = np.deg2rad(90)
    _YCB_STEP = 0.02

    _TOLERANCE = .1
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.packing_box = Package(self._mojo)
        self.table = self._preset.get_props(BaseCabinet)[0]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )

    def _on_step(self):
        self._metric_step()

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        self.packing_box.body.set_position(self._BOX_POS)
        self.packing_box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT).elements)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)

    def _success(self) -> bool:
        self.success_check = True
        
        # Close all lid joints of the box 
        if not np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE):
            self.success_check = False
            
        # —————————— distance check ——————————————————
        right_gripper_right_flap = distance(self.robot.grippers[HandSide.RIGHT].body, self.packing_box.flap_1)
        left_gripper_left_flap = distance(self.robot.grippers[HandSide.LEFT].body, self.packing_box.flap_2)

        
        # ——————————— stage monitoring ———————————————
        grasping_right_flap = self.robot.is_gripper_holding_object(self.packing_box.flap_1, HandSide.RIGHT)
        grasping_left_flap = self.robot.is_gripper_holding_object(self.packing_box.flap_2, HandSide.LEFT)
        close_box = np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE)
        box_state = self.packing_box.get_state()
        close_right_flap = abs(box_state[0]) < self._TOLERANCE
        close_left_flap = abs(box_state[1]) < self._TOLERANCE

        if grasping_left_flap: self._metric_stage(1) 
        if grasping_right_flap: self._metric_stage(2)
        if close_right_flap: self._metric_stage(3)
        if close_left_flap: self._metric_stage(4)
        if close_box: self._metric_stage(5) # both flaps are closed 
        
        self._final_metrics = self._metric_finalize(
            success_flag=self.success_check,
            target_distance={
                "right gripper-right flap distance": right_gripper_right_flap,
                "left gripper-left flap distance": left_gripper_left_flap
            }
            )

        return self.success_check
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})      
    
class PackBoxPositionAndOrientation(PackBox):
    """Pack box with position and orientation randomization"""

    _BOX_POS = np.array([0.7, 0.25, 1.05])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        points = get_random_points_on_plane(
            1,
            self._BOX_POS,
            self._BOX_POS_EXTENTS,
            self._YCB_STEP,
            self._BOX_POS_BOUNDS,
        )

        point = points[0]
        self.packing_box.body.set_position(point)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.packing_box.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT + angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)

class PackBoxOrientation(PackBox):
    """Pack box with orientation randomization"""
    _PRESET_PATH = PRESETS_PATH / "packaging_box_rotation.yaml"

    _BOX_POS = np.array([0.72, 0, 1.05])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        self.packing_box.body.set_position(self._BOX_POS)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.packing_box.body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT + angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)
        

class PackBoxPosition(PackBox):
    """Pack box with position randomization"""
    _BOX_POS = np.array([0.7, 0.3, 1.05])
    
    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        points = get_random_points_on_plane(
            1,
            self._BOX_POS,
            self._BOX_POS_EXTENTS,
            self._YCB_STEP,
            self._BOX_POS_BOUNDS,
        )

        point = points[0]
        self.packing_box.body.set_position(point)
        self.packing_box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=self._BOX_ROT).elements)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.packing_box,
            track_collisions=True,
            robot=self.robot,         
        )
        
        for idx in range(1, 6): self._metric_stage(idx , False)
class _PackingYCBEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base env for packing/unpacking ycb objects"""

    _PRESET_PATH = PRESETS_PATH / "packaging.yaml"
    _YCB_COUNT = 3

    # _CUBE_POS = np.array([1.2, 0, 1.7])
    _TOLERANCE = .1
    _YCB_ROT = np.deg2rad(180)
    _YCB_STEP = 0.15
    _YCB_POS_EXTENTS = np.array([0.1, 0.25])
    _YCB_POS_BOUNDS = np.array([0.005, 0.005, 0])
    _YCB_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.05
    _BOX_POS = np.array([0.7, 0, 1.05])
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.packing_box = self._preset.get_props(Package)[0]
        self.table = self._preset.get_props(BaseCabinet)[0]
        self.master_chef_can = YcbProp("007_tuna_fish_can", self._mojo)
        self.cracker_box = YcbProp("011_banana", self._mojo)
        self.apple = YcbProp("013_apple", self._mojo)
        self.ycb = [self.master_chef_can, self.cracker_box, self.apple]
        
        objs = [self.packing_box]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.packing_box, self.master_chef_can, self.cracker_box, self.apple],
            track_collisions=True,
            robot=self.robot,         
        )
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})  
    

class UnpackYCB(_PackingYCBEnv):
    """
    Given ycb objects inside the packing box, robot effectors must move the ycb objects to the counter
    """
    _YCB_POS = np.array([0.7, 0.3, 1.1])

    def _on_reset(self):
        self.packing_box.set_state(state=0.2)
        self.packing_box.set_state(target_state={ 
            "packaging_box/joint_0": 0.2,
            "packaging_box/joint_1": 0.2,
        }) # Start at fully closed joints

        points = get_random_points_on_plane(
            len(self.ycb),
            self._YCB_POS,
            self._YCB_POS_EXTENTS,
            self._YCB_STEP,
            self._YCB_POS_BOUNDS,
        )
        for ycb_obj, point in zip(self.ycb, points):
            ycb_obj.body.set_position(point)
            angle = np.random.uniform(-self._YCB_ROT_BOUNDS, self._YCB_ROT_BOUNDS)
            angle = 0
            ycb_obj.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._YCB_ROT + angle).elements
            )
        objs = [self.packing_box]
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=objs.extend(self.ycb),
            track_collisions=True,
            robot=self.robot,         
        )
    
    def _success(self) -> bool:
        self._success_check = True
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for ycb_obj in self.ycb:
            if not ycb_obj.is_colliding(self.table.counter):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3)
                
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(ycb_obj, side):
                    self._success_check = False
                    self._metric_stage(2)
                    
        if not np.allclose(self.packing_box.get_state(), 1, atol=self._TOLERANCE): # Check if the box is fully opened
            self._success_check =False
            self._metric_stage(1, False)
        else: 
            self._metric_stage(1)
            
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check
        )
        return self._success_check
    
    def _on_step(self):
        self._metric_step()
    
class PackYCB(_PackingYCBEnv):
    """
    Given ycb objects outside the packing box, robot effectors must move the ycb objects inside the packing box
    """
    _YCB_POS = np.array([0.7, -.3, 1])

    def _on_reset(self):
        self.packing_box.set_state(state=0)
        self.packing_box.set_state(target_state={
            "packaging_box/joint_0": 1,
            "packaging_box/joint_1": 1,
        })

        points = get_random_points_on_plane(
            len(self.ycb),
            self._YCB_POS,
            self._YCB_POS_EXTENTS,
            self._YCB_STEP,
            self._YCB_POS_BOUNDS,
        )
        for ycb_obj, point in zip(self.ycb, points):
            ycb_obj.body.set_position(point)
            angle = np.random.uniform(-self._YCB_ROT_BOUNDS, self._YCB_ROT_BOUNDS)
            angle = 0
            ycb_obj.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._YCB_ROT + angle).elements
            )
            
        objs = [self.packing_box]
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=[self.packing_box, self.master_chef_can, self.cracker_box, self.apple],
            track_collisions=True,
            robot=self.robot,         
        )
    
    def _success(self) -> bool:
        self._success_check = True
        for ycb_obj in self.ycb:         
            # might want to add a site that encompasses the inside of the box
            if not ycb_obj.is_colliding(self.packing_box.base):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3, True)
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(ycb_obj, side):
                    self._success_check = False
                    self._metric_stage(2)
        
        # Close all lid joints of the box 
        if not np.allclose(self.packing_box.get_state(), 0, atol=self._TOLERANCE):
            self._success_check = False
            self._metric_stage(1, False)
        else:
            self._metric_stage(1)
            
        self._final_metrics = self._metric_finalize(success_flag=self._success_check)
        return self._success_check 
    
    def _on_step(self):
        self._metric_step()