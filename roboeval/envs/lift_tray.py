from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import BreakfastTray, LighterBreakfastTray
from roboeval.envs.props.items import Cube, Box, Sandwich
from roboeval.envs.props.cabintets import BaseCabinet
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from roboeval.envs.props.ycb import YcbProp
from roboeval.envs.props.prop import Prop
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance

class _TrayEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base env for lifting and moving tray"""
    _PRESET_PATH = PRESETS_PATH / "lift_tray.yaml"
    _success_check = True
    _final_metrics = {}
    

    def _initialize_env(self):
        self.breakfast_tray = self._preset.get_props(BreakfastTray)[0]
        self.table = self._preset.get_props(BaseCabinet)[0]
        self.target_table = self._preset.get_props(BaseCabinet)[-1]  
    

# ----------------------------------- Lift Tray Tasks -----------------------------------
class LiftTray(_TrayEnv):
    """
    Lift the tray with both effectors (static)
    """

    _TOLERANCE = .1
    _OBJ_ROT = np.array([0.5 ,0.5 ,0.5 ,0.5])
    _OBJ_STEP = 0.15
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])
    _OBJ_POS_BOUNDS = np.array([0.005, 0.005, 0])
    _OBJ_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.1

    _OBJ_POS = np.array([0.65, -0.325, 1.0])
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])  

    

    def _initialize_env(self):
        super()._initialize_env()
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True        
        )

    def _on_reset(self):
        self.breakfast_tray.body.set_quaternion(
            self._OBJ_ROT
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True        
        )
        
        self.init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.02694
        for idx in range(1, 4): self._metric_stage(idx, False) # set all stages to false
        
    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        
        for side in self.robot.grippers:
            if not self.robot.is_gripper_holding_object(self.breakfast_tray, side):
                self._success_check = False
            
        if self.breakfast_tray.is_colliding(self.table):
            self._success_check = False
        
        # –––––––––––––– Distance Checking –––––––––––––––––
        tray_table_distance = distance(self.breakfast_tray.body, self.table.body)
        tray_lift_distance = self.breakfast_tray.body.get_position()[2] - self.init_z_pos
        right_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.RIGHT].body)
        left_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.LEFT].body)

        # –––––––––––––– Stage Monitoring ––––––––––––––––––
        colliding = self.breakfast_tray.is_colliding(self.table)
        floor_collision = self.breakfast_tray.is_colliding(self._floor)
        holding_obj = all(self.robot.is_gripper_holding_object(self.breakfast_tray, side) for side in self.robot.grippers)
        grasping_left = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.LEFT) 
        grasping_right = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.RIGHT) 
        
        if grasping_left: self._metric_stage(1) # split into left and right grasp check
        if grasping_right: self._metric_stage(2)
        if holding_obj and not colliding and not floor_collision: self._metric_stage(3) # Check if tray is not colliding with the table or floor

        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "lift_distance": tray_lift_distance,
                "tray-table distance": tray_table_distance,
                "right gripper-tray distance":right_gripper_tray_dist,
                "left gripper-tray distance": left_gripper_tray_dist
            }
        )
        return self._success_check

    def _fail(self) -> bool:
        return super()._fail()
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class LiftTrayPosition(LiftTray):
    """
    Lift the tray with both effectors and randomization on tray's position
    """

    def _on_reset(self):
        super()._on_reset()
        points = get_random_points_on_plane(
            1,
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )

        # Set new tray location for tray
        self.init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.127
        self.breakfast_tray.body.set_position(points[0])


class LiftTrayOrientation(LiftTray):
    """
    Lift the tray with both effectors and randomization on tray's orientation
    """

    _OBJ_ROT_BOUNDS = np.deg2rad(20)

    _OBJ_POS = np.array([0.65, -0.325, 0.9])
    
    _success_check = True
    _final_metrics = {}

    def _on_reset(self):
        super()._on_reset()
        angle = np.random.uniform(-self._OBJ_ROT_BOUNDS, self._OBJ_ROT_BOUNDS)
        new_rot = Quaternion(axis=[0, 0, 1], angle=angle) * Quaternion(self._OBJ_ROT)
        self.breakfast_tray.body.set_quaternion(new_rot.elements)
        
        self.init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.027
    

class LiftTrayPositionAndOrientation(LiftTray):
    """
    Lift the tray with both effectors and randomization of position AND orientation
    """


    def _on_reset(self):
        super()._on_reset()
        points = get_random_points_on_plane(
            1,
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )

        # Set new tray location for tray
        self.breakfast_tray.body.set_position(points[0])
        angle = np.random.uniform(-self._OBJ_ROT_BOUNDS, self._OBJ_ROT_BOUNDS)
        new_rot = Quaternion(axis=[0, 0, 1], angle=angle) * Quaternion(self._OBJ_ROT)
        self.breakfast_tray.body.set_quaternion(new_rot.elements)
        
        self.init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.127
        


# ----------------------------------- Move Tray Tasks -----------------------------------
class MoveTray(_TrayEnv):
    """
    Place objects onto a tray and place tray on the other counter 
    """
    
    _OBJ_COUNT = 2

    _TOLERANCE = .1
    _OBJ_ROT = np.deg2rad(180)
    _OBJ_STEP = 0.15
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])
    _OBJ_POS_BOUNDS = np.array([0.005, 0.005, 0])
    _OBJ_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.1
    
    _OBJ_POS = np.array([0.65, -0.325, 1.1])
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25]) # TODO: fix the XML combined mesh for base is does not outline a concave geometry
    _success_check = True
    _final_metrics = {}
    
    def _initialize_env(self):
        super()._initialize_env()
        self.apple = YcbProp("013_apple", self._mojo)
        self.pudding_box = YcbProp("008_pudding_box", self._mojo)
        self.tray_item = [self.apple, self.pudding_box]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,  
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True   
        )
        

    def _on_reset(self):

        points = get_random_points_on_plane(
            len(self.tray_item),
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )
        for tray_item_obj, point in zip(self.tray_item, points):
            tray_item_obj.body.set_position(point)
            tray_item_obj.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._OBJ_ROT).elements
            )
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,  
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True   
        )
        
        self._init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.027
        for idx in range(1, 5): self._metric_stage(idx, False) # set all stages to false
        
    def _on_step(self):
        self._metric_step()
    
    def _success(self) -> bool:
        self._success_check = True

        for tray_item_obj in self.tray_item:
            if not tray_item_obj.is_colliding(self.breakfast_tray.body):
                self._success_check = False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(tray_item_obj, side):
                    self._success_check = False
        if not self.breakfast_tray.is_colliding(self.target_table.hob):
            self._success_check = False
            
        # –––––––––––––– Distance Checking –––––––––––––––––
        tray_table_distance = distance(self.breakfast_tray.body, self.target_table.body)
        tray_lift_distance = self.breakfast_tray.body.get_position()[2] -  self._init_z_pos
        right_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.RIGHT].body)
        left_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.LEFT].body)
        
        # –––––––––––––– Stage Monitoring ––––––––––––––––––
        colliding = self.breakfast_tray.is_colliding(self.table)
        floor_collision = self.breakfast_tray.is_colliding(self._floor)
        holding_obj = all(self.robot.is_gripper_holding_object(self.breakfast_tray, side) for side in self.robot.grippers)
        target_collision = self.breakfast_tray.is_colliding(self.target_table.hob)
        grasping_left = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.LEFT) 
        grasping_right = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.RIGHT) 
        
        if grasping_left: self._metric_stage(1) # split grasp into left and right check
        if grasping_right: self._metric_stage(2)
        if holding_obj and not colliding and not floor_collision: self._metric_stage(3) # Check if tray is not colliding with the table or floor WHILE holding the object
        if target_collision: self._metric_stage(4) #  keep 
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "lift_distance": tray_lift_distance,
                "tray-table distance": tray_table_distance,
                "right gripper-tray distance":right_gripper_tray_dist,
                "left gripper-tray distance": left_gripper_tray_dist
            }
        )
        return self._success_check
    
    def _fail(self) -> bool:
        if super()._fail():
            return True
        for tray_item_obj in self.tray_item:
            if not tray_item_obj.is_colliding(self.breakfast_tray.body):
                return True
        return False
    
    def _get_task_info(self):   
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class MoveStackedOBJLiftTray(MoveTray):
    """
    Place stacked objects onto a tray and place tray on the other counter
    """
    _TOLERANCE = .1
    _SUCCESSFUL_DIST = 0.05
    
    _PLATE_QUAT = Quaternion(axis=[0, 0, 1], degrees=-90)
    _PLATE_POS_BOUNDS = np.array([1 , 0.1, .93])
    _PLATE_ROT_BOUNDS = np.deg2rad([0, 0, 30])
    _PLATE_ROT = np.deg2rad(90)
    
    _OBJ_ROT = np.deg2rad(90)
    _OBJ_STEP = 0.01
    _OBJ_POS_EXTENTS = np.array([0.01, 0.025])
    _OBJ_POS_BOUNDS = np.array([0.0, 0.1, 0])
    _OBJ_ROT_BOUNDS = np.deg2rad(30)
    _OBJ_POS = np.array([0.65, -0.325, .93])

    _TOLERANCE = np.deg2rad(10)
    _TOASTED = False
    
    _SANDWICH_OFFSET = np.array([0, 0, 0.02])
    _SANDWICH_POS_BOUNDS = np.array([1, 0.1, 0.98])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _ROUNDED = True

    def _initialize_env(self):
        super()._initialize_env()
        
        self.plate = YcbProp("029_plate", self._mojo)
        self.sandwich = Sandwich(
            self._mojo, toasted=self._TOASTED, rounded_collider=self._ROUNDED
        )

    def _on_reset(self):   
        points = get_random_points_on_plane(
            1,
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )     
        self.plate.body.set_position(points[0])
        self.sandwich.body.set_position(self.plate.body.get_position() + np.array([0, 0, .05]))
    
    def _success(self) -> bool:

        if not self.plate.is_colliding(self.breakfast_tray):
            return False
        if not self.sandwich.is_colliding(self.plate):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.breakfast_tray, side):
                return False
        if not self.breakfast_tray.is_colliding(self.target_table.hob):
            return False
        return True
    
    def _fail(self) -> bool:
        if super()._fail():
            return True
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.sandwich, side):
                return True
            
        if not self.plate.is_colliding(self.breakfast_tray):
            return True
        if not self.sandwich.is_colliding(self.plate):
            return True
        return False
    
    @property
    def _sandwich_anchor(self) -> Prop:
        return self.plate
    
class BalanceAndMoveTray(_TrayEnv):
    """
    Lift tray and ensure that objects are in upright orientation
    """
    _OBJ_COUNT = 2

    _TOLERANCE = .1
    _OBJ_ROT = np.deg2rad(0)
    _OBJ_STEP = 0.1
    _OBJ_POS_EXTENTS = np.array([0.1, 0.25])
    _OBJ_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _OBJ_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.05
    
    _PLATE_POS_BOUNDS = np.array([1 , 0.1, .93])
    _SANDWICH_POS_BOUNDS = np.array([1, 0.1, 0.98])
    _MUSTARD_POS = np.array([0.95, -0.15, 0.95])
    _MUG_POS = np.array([1.05, -0.15, .98])
    _MUG_ROT = np.deg2rad(0)

    _OBJ_POS = np.array([0.65, -0.325, .93])
    
    _SUCCESS_ROT = np.deg2rad(0)
    _SUCCESS_ROT_TOLERANCE = .1
    
    _TOASTED = False
    _ROUNDED = True

    
    def _initialize_env(self):
        super()._initialize_env()
        self.mustard = YcbProp("006_mustard_bottle", self._mojo)
        self.mug = YcbProp("025_mug", self._mojo)
        self.tray_item = [self.mug, self.mustard]

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.tray_item),
            self._OBJ_POS,
            self._OBJ_POS_EXTENTS,
            self._OBJ_STEP,
            self._OBJ_POS_BOUNDS,
        )
        for tray_item_obj, point in zip(self.tray_item, points):
            tray_item_obj.body.set_position(point)
        self.mug.body.set_quaternion(
            Quaternion(axis=[0, 1, 0], angle=self._MUG_ROT).elements
        )
            
    def _success(self) -> bool:
        z_axis = np.array([0, 0, 1]) # z-axis
        
        
        for tray_item_obj in self.tray_item:
            obj_up = Quaternion(tray_item_obj.body.get_quaternion()).rotate(z_axis)
            angle = np.arccos(np.clip(np.dot(obj_up, z_axis), -1.0, 1.0))
            print(np.rad2deg(angle))
            if abs(angle - self._SUCCESS_ROT) > self._SUCCESS_ROT_TOLERANCE:
                print("object angle is not upright")
                return False
            else:
                print("obj angle is upright")
            if not tray_item_obj.is_colliding(self.breakfast_tray.body):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(tray_item_obj, side):
                    return False
        if not self.breakfast_tray.is_colliding(self.target_table.hob):
            return False
        
        if not self.plate.is_colliding(self.breakfast_tray):
            return False
        if not self.sandwich.is_colliding(self.plate):
            return False
        if not self.breakfast_tray.is_colliding(self.target_table.hob):
            return False
        
        return True
    
    def _fail(self) -> bool:
        if super()._fail():
            return True
        for tray_item_obj in self.tray_item:
            if not tray_item_obj.is_colliding(self.breakfast_tray.body):
                return True
            
        if not self.plate.is_colliding(self.breakfast_tray):
            return True
        if not self.sandwich.is_colliding(self.plate):
            return True
            
        return False
    
    @property
    def _sandwich_anchor(self) -> Prop:
        return self.plate


# ----------------------------------- Other Tray Tasks -----------------------------------
class DragOverAndLiftTray(_TrayEnv):
    """
    Tray begins on adjacent table
    Drag tray to target table and raise above target table
    """

    _PRESET_PATH = PRESETS_PATH / "drag_over_and_lift_tray.yaml"
    _OBJ_POS = np.array([0.65, 0.325, 1.0])


    def _initialize_env(self):
        self.breakfast_tray = self._preset.get_props(LighterBreakfastTray)[0]
        self.table = self._preset.get_props(BaseCabinet)[0]
        self.target_table = self._preset.get_props(BaseCabinet)[-1]
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True        
        )  

    def _on_step(self):
        self._metric_step()
        
    def _on_reset(self):
        self.breakfast_tray.body.set_position(self._OBJ_POS)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.breakfast_tray,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True        
        )  
        self._init_z_pos = self.breakfast_tray.body.get_position()[2] - 0.12694
        for idx in range(1, 5): self._metric_stage(idx, False) # set all stages to false

    def _success(self) -> bool:
        self._success_check = True
        
        # –––––––––– success check ––––––––––––––––––
        for side in self.robot.grippers:
            if not self.robot.is_gripper_holding_object(self.breakfast_tray, side):
                self._success_check = False
        if self.breakfast_tray.is_colliding(self.table):
            self._success_check = False
        if self.breakfast_tray.is_colliding(self.target_table):
            self._success_check = False
        if self.breakfast_tray.body.get_position()[1] > 0:
            self._success_check = False
            
        # ––––––––––– distance check –––––––––––––––––
        tray_table_distance = distance(self.breakfast_tray.body, self.table.body)
        tray_lift_distance = self.breakfast_tray.body.get_position()[2] -  self._init_z_pos
        right_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.RIGHT].body)
        left_gripper_tray_dist = distance(self.breakfast_tray.body, self.robot.grippers[HandSide.LEFT].body)
        
         # –––––––––––––– Stage Monitoring ––––––––––––––––––
        colliding = self.breakfast_tray.is_colliding(self.table) or self.breakfast_tray.is_colliding(self.target_table)
        floor_collision = self.breakfast_tray.is_colliding(self._floor)
        holding_obj = all(self.robot.is_gripper_holding_object(self.breakfast_tray, side) for side in self.robot.grippers)
        target_collision = self.breakfast_tray.is_colliding(self.table.hob)
        grasping_left = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.LEFT) 
        grasping_right = self.robot.is_gripper_holding_object(self.breakfast_tray, HandSide.RIGHT)
        above_table = self.breakfast_tray.body.get_position()[1] <= 0
        
        if grasping_left: self._metric_stage(1) # left effector grasp tray
        if grasping_right: self._metric_stage(2) # right effector grasp tray
        if target_collision: self._metric_stage(3) #  collide with table 
        if holding_obj and not colliding and not floor_collision and above_table: self._metric_stage(4) # Check if tray is not colliding with the table or floor WHILE holding the object and hovering over target table
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "lift_distance": tray_lift_distance,
                "tray-table distance": tray_table_distance,
                "right gripper-tray distance":right_gripper_tray_dist,
                "left gripper-tray distance": left_gripper_tray_dist
            }
        )
        return self._success_check
    
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})