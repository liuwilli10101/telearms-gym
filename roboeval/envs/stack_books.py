from abc import ABC

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.book_shelf import BookShelf
from roboeval.envs.props.items import Book
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
import quaternion
from scipy.spatial.transform import Rotation
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.const import HandSide  # LEFT / RIGHT enums

class _StackBooksEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base Environment for Stacking Books"""

    _PRESET_PATH = PRESETS_PATH / "place_books.yaml"


    _BOOK_COUNT = 2

    _BOOK_POS = np.array([0.4, 0, 0.72])
    _BOOK_ROT = np.deg2rad(180)
    _BOOK_STEP = 0.15
    _BOOK_POS_EXTENTS = np.array([0.1, 0.25])
    _BOOK_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _BOOK_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.005
    
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.books = [Book(self._mojo) for _ in range(self._BOOK_COUNT)]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        self.book_init_height = self.books[0].body.get_position()[2]
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
        
class StackSingleBookShelf(_StackBooksEnv):
    """
    Simple pick and place task: Robot transports books from counter to a shelf and checks if the books are colliding with either the top or lower shelf
    """

    _BOOK_COUNT = 1
    SUCCESS_HEIGHT = 0.1
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        self.books[0].body.set_position(self._BOOK_POS)
        self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=np.deg2rad(90)).elements)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)


    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True

        for book in self.books:
            #  We check if the book is colliding with the mesh of either the top or bottom shelf
            if not book.is_colliding(self.book_shelf.lower_shelf_body) and not book.is_colliding(self.book_shelf.upper_shelf_body):
                self._success_check = False
  
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False   
         
        # ––––––– define stage boundaries ––––––––––––––––––––           
        grasped  = any(
            self.robot.is_gripper_holding_object(self.books[0], side)
            for side in self.robot.grippers
        )
        on_shelf = (
            self.books[0].is_colliding(self.book_shelf.lower_shelf_body)
            or self.books[0].is_colliding(self.book_shelf.upper_shelf_body)
        )
        lifted   = (self.books[0].body.get_position()[2] - self.book_init_height) >= self.SUCCESS_HEIGHT
        released = on_shelf and not grasped
        
        # ─────── monitor task progression ───────────────────── 
        if grasped:   self._metric_stage(1)   # holding
        if lifted:    self._metric_stage(2)   # lifted: ——–––– KEEP
        if on_shelf:  self._metric_stage(3)   # touching shelf
        if released:  self._metric_stage(4)   # placed & released
                    
        book_lift_dist = self.books[0].body.get_position()[2] - self.book_init_height
        upper_shelf_dist = distance(self.books[0].body, self.book_shelf.upper_shelf_body)
        lower_shelf_dist = distance(self.books[0].body, self.book_shelf.lower_shelf_body)
        

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "book lift distance": book_lift_dist,
                "book-upper shelf": upper_shelf_dist,
                "book-lower shelf": lower_shelf_dist
            }
            )
        return self._success_check
    
class StackSingleBookShelfPosition(StackSingleBookShelf):
    """
    Stack single book on shelf with position randomization
    """
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })


        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=np.deg2rad(90)).elements)
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)

class StackSingleBookShelfPositionAndOrientation(StackSingleBookShelf):
    """
    Stack single book on shelf with position and orientation randomization
    """
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)

            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            self.books[0].body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 5): self._metric_stage(idx, False)


class StackBooksRightSide(_StackBooksEnv):
    """
    Transpot book from left side of the table to the right side of the lower shelf 
    """
    _BOOK_COUNT = 1
    _TARGET_SIZE = np.array([0.07, 0.1, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([0.6, -0.35, 1.025])
    _BOOK_ROT = np.deg2rad(90)
    _BOOK_POS = np.array([0.6, 0.45, 0.79])
    _SUCCESS_ROT_TOLERANCE = 15
    _SUCCESS_ROT = 90
    SUCCESS_HEIGHT = 0.1
    
    def _initialize_env(self):
        self.book = Book(self._mojo)
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.target: Body = Body.create(self._mojo)
        self.target_collider = Geom.create(
            self._mojo,
            parent=self.target,
            geom_type=GeomType.BOX,
            size=self._TARGET_SIZE,
            color=self._TARGET_COLOR,
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.book,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        self.book.body.set_position(self._BOOK_POS)
        self.book.body.set_quaternion(
                Quaternion(axis=[1, 0, 0], angle=self._BOOK_ROT).elements
            )
        
        self.target.set_position(self._TARGET_POS)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.book,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.book.body.get_position()[2]
        for idx in range(1, 6): self._metric_stage(idx, False) 
        
    def _on_step(self):
        self._metric_step()
        
    def _success(self):
        self._success_check = True
                        
        if not self.book.is_colliding(self.target_collider):
            print("Book is not colliding with target")
            self._success_check = False
        else:
            print("Book is colliding with target")

        if not self.book.is_colliding(self.book_shelf.lower_shelf_body):
            self._success_check = False

        # check book angle
        mj_q = self.book.body.get_quaternion()  # [w, x, y, z]
        quat_scipy_order = [mj_q[1], mj_q[2], mj_q[3], mj_q[0]]

        rot = Rotation.from_quat(quat_scipy_order)

        # convert to euler angles in  degrees
        rot_euler_deg = rot.as_euler('xyz', degrees=True)
        roll_deg, pitch_deg, yaw_deg = rot_euler_deg

        # check angle error
        angle_error = abs(roll_deg - self._SUCCESS_ROT)

        print(f'Book Orientation: {roll_deg}, angle_error: {angle_error}')

        if (angle_error > self._SUCCESS_ROT_TOLERANCE):
            self._success_check = False     
            
            
        # —— stage monitoring definitions  -----------------------------------
        grasped = any(
            self.robot.is_gripper_holding_object(self.book, side)
            for side in self.robot.grippers
        )
        angle_check = angle_error <= self._SUCCESS_ROT_TOLERANCE
        touches_target = self.book.is_colliding(self.target_collider)
        released = not grasped and touches_target
        lifted   = (self.book.body.get_position()[2] - self.book_init_height) >= self.SUCCESS_HEIGHT


        # ––––––––––– track and set stages ––––––––––––––––––––––––––––––––
        if grasped: self._metric_stage(1)     # picked up
        if lifted: self._metric_stage(2) # lifted 
        if touches_target: self._metric_stage(3) # book touches target area
        if angle_check: self._metric_stage(4) # book is in upright position
        if released: self._metric_stage(5) # if book is right side of lower sehfl

            
        distance_to_target = distance(self.book.body, self.target)   
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            pose_error=angle_error,
            target_distance=distance_to_target
        )
        return self._success_check
    
class StackBooksUpright(_StackBooksEnv):
    """
    Pick and Place Task: Robot transports books to the lower shelf and checks if the books are in an upright orientation
    """
    _SUCCESS_ROT = 90
    _SUCCESS_ROT_TOLERANCE = 15
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            angle = np.deg2rad(-90)
            book.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._BOOK_ROT + angle).elements
            )
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        for book in self.books:       
            # check book angle 
            mj_q = book.body.get_quaternion()  # [w, x, y, z]
            quat_scipy_order = [mj_q[1], mj_q[2], mj_q[3], mj_q[0]]
            
            rot = Rotation.from_quat(quat_scipy_order)
            
            # convert to euler angles in  degrees 
            rot_euler_deg = rot.as_euler('xyz', degrees=True)
            roll_deg, pitch_deg, yaw_deg = rot_euler_deg

            # check angle error
            angle_error = abs(roll_deg - self._SUCCESS_ROT)

            print(f'Book Orientation: {roll_deg}, angle_error: {angle_error}')

            if (angle_error > self._SUCCESS_ROT_TOLERANCE):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3, True)
            
            if not book.is_colliding(self.book_shelf.lower_shelf_body):
                self._success_check = False 
                self._metric_stage(2, False)  
            else:
                self._metric_stage(2, True)
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False   
                else:
                    self._metric_stage(1)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            pose_error=angle_error
        ) 
        return self._success_check
    
class _PickBooksEnv(RoboEvalEnv, ABC, MetricRolloutEval):
    """Base initializations for picking books off the shelf"""

    _PRESET_PATH = PRESETS_PATH / "place_books.yaml"


    _BOOK_COUNT = 3

    _BOOK_SHELF_POS = np.array([0.6, -0.425, 1.1]) # Upper shelf placement 
    _BOOK_TABLE_POS = np.array([0.4, 0, 0.72])
    _BOOK_ROT = np.deg2rad(90)
    _BOOK_STEP = 0.15
    _BOOK_POS_EXTENTS = np.array([0.1, 0.25])
    _BOOK_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _BOOK_ROT_BOUNDS = np.deg2rad(30)
    _SUCCESSFUL_DIST = 0.005
    
    _success_check = True
    _final_metrics = {}



    def _initialize_env(self):
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.books = [Book(self._mojo) for _ in range(self._BOOK_COUNT)]

        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})

class PickSingleBookFromTable(_PickBooksEnv):
    """Pick a single book up from the table (static)"""
    _BOOK_COUNT = 1
    _LIFT_SUCCESS = 0.1

    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        self.books[0].body.set_position(self._BOOK_TABLE_POS)
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOOK_ROT).elements
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    def _success(self):
        self._success_check = True

        book = self.books[0]
        if book.body.get_position()[2] < 0.77 or book.is_colliding(self.book_shelf.counter) or book.is_colliding(self._floor):
            self._success_check = False

        if not any(self.robot.is_gripper_holding_object(self.books[0], side) for side in self.robot.grippers): # At least one effector must be gripping the book
            self._success_check = False
            
        distance_from_left_gripper = distance(book.body, self.robot.grippers[HandSide.LEFT].body)
        distance_from_right_gripper = distance(book.body, self.robot.grippers[HandSide.RIGHT].body)
        book_lift_dist = self.books[0].body.get_position()[2] - self.book_init_height
        
        # –––––––– stage monitoring –––––––––––––––––––––––
        grasped = any(
            self.robot.is_gripper_holding_object(book, side)
            for side in self.robot.grippers
        )
        
        if grasped: self._metric_stage(1) # hold book or in touch with book
        if grasped and not book.is_colliding(self.book_shelf.counter) and not book.is_colliding(self._floor): self._metric_stage(2)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "book lift distance": book_lift_dist,
                "left gripper-book distance": distance_from_left_gripper,
                "right gripper-book distance": distance_from_right_gripper
            }
            )
        return self._success_check

# Pick Single Book with position randomization
class PickSingleBookFromTablePosition(PickSingleBookFromTable):
    """Pick single book from table with position randomization"""
    _BOOK_COUNT = 1
    _LIFT_SUCCESS = 0.1

    def _on_reset(self):
        # super()._on_reset()

        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })

        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_TABLE_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
        
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=self._BOOK_ROT).elements
        )
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )
        
        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)

    
class PickSingleBookFromTableOrientation(PickSingleBookFromTable):
    """Pick single book from table with orientation randomization"""
    _BOOK_ROT_BOUNDS = np.deg2rad(180)

    def _on_reset(self):
        # super()._on_reset()
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        self.books[0].body.set_position(self._BOOK_TABLE_POS)

        angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
        self.books[0].body.set_quaternion(
            Quaternion(axis=[0, 0, 1], angle=angle).elements
        )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )

        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)
        


class PickSingleBookFromTablePositionAndOrientation(PickSingleBookFromTable):
    """Pick single book from table with position and orientation randomization"""
    _BOOK_ROT_BOUNDS = np.deg2rad(180)

    def _on_reset(self):

        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        points = get_random_points_on_plane(
            len(self.books),
            self._BOOK_TABLE_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            book.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )

        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot,    
        )

        self.book_init_height = self.books[0].body.get_position()[2] - .005
        for idx in range(1, 3): self._metric_stage(idx, False)

    
class PickSingleBookFromShelf(_PickBooksEnv):
    """
    Given a single book placed upright on the lower shelf, pick and place the book from the shelf to the counter 
    """
    
    _BOOK_COUNT = 1
    
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        book = self.books[0]
        point = self._BOOK_SHELF_POS
        book.body.set_position(point)
        book.body.set_quaternion(
            Quaternion(axis=[1, 0, 0], angle=self._BOOK_ROT).elements
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        self.book_init_height = self.books[0].body.get_position()[2]
        for idx in range(1, 3): self._metric_stage(idx, False)


    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        for book in self.books:
            if not book.is_colliding(self.book_shelf.counter):
                self._success_check = False
            else:
                self._metric_stage(2, True)
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False
                    self._metric_stage(1)
                    
        distance_from_left_gripper = distance(book.body, self.robot.grippers[HandSide.LEFT].body)
        distance_from_right_gripper = distance(book.body, self.robot.grippers[HandSide.RIGHT].body)
        book_lift_dist = self.books[0].body.get_position()[2] - self.book_init_height

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "book lift distance": book_lift_dist,
                "left gripper-book distance": distance_from_left_gripper,
                "right gripper-book distance": distance_from_right_gripper
            }
            )
        return self._success_check
    
class PickBookPlaceLeftSideEnv(PickSingleBookFromShelf):
    _TARGET_SIZE = np.array([0.07, 0.1, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([0.6, 0.45, 0.69])
    
    def _initialize_env(self):
        super()._initialize_env()
        self.target: Body = Body.create(self._mojo)
        self.target_collider = Geom.create(
            self._mojo,
            parent=self.target,
            geom_type=GeomType.BOX,
            size=self._TARGET_SIZE,
            color=self._TARGET_COLOR,
        )
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )

    def _on_step(self):
        self._metric_step()
        
    def _on_reset(self):
        super()._on_reset()
        self.target.set_position(self._TARGET_POS)
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books[0],
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
    def _success(self):
        self._success_check = True
        for book in self.books:            
            if not book.is_colliding(self.target_collider):
                self._success_check = False
                self._metric_stage(2, False)
            else:
                self._metric_stage(2, True)
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False
                    self._metric_stage(1)
                    
        target_dist = distance(self.books[0].body, self.target)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"book-target distance:": target_dist}
            )   
        return self._success_check

class PickBookFromStack(_PickBooksEnv):
    """
    Given a stack of books on the upper shelf, pick and place each book from the shelf to the counter.
    """
    _BOOK_SPACING = 0.05 # spacing value
    
    def _initialize_env(self):
        super()._initialize_env()
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
    def _on_step(self):
        self._metric_step()
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        # Define an incremental translation to each book to spawn books side by side
        x, y, z = 0.6, -0.425, 1.1
        points = [self._BOOK_SHELF_POS]
        for _ in  range(self._BOOK_COUNT):
            y += self._BOOK_SPACING 
            point = np.array([x, y , z])
            points.append(point)
        
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            book.body.set_quaternion(
            Quaternion(axis=[1, 0, 0], angle=self._BOOK_ROT).elements
        )
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )

    
    def _success(self) -> bool:
        self._success_check = True
        for book in self.books:
            if not book.is_colliding(self.book_shelf.counter):
                self._success_check = False
                self._metric_stage(2, False)
            else:
                self._metric_stage(2, True)
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False 
                    self._metric_stage(1)
        self._final_metrics = self._metric_finalize(success_flag=self._success_check)   
        return self._success_check
    
class InsertBookInStack(_PickBooksEnv):
    """
    Insertion Task: We start with a book on the counter and we check the following: 
    1. Book is on the upper shelf with the other books in the stack
    2. Book is colliding with other books 
    3. All books are upright
    """
    
    _SUCCESS_ROT = 90
    _SUCCESS_ROT_TOLERANCE = 15
    
    _BOOK_SPACING = 0.05 # spacing value
    _TARGET_BOOK_COUNT = 2
    _TARGET_BOOK_POS = np.array([0.365, 0, 0.72])
    _TARGET_BOOK_ROT = np.deg2rad(180)
    _SUCCESSFUL_DIST = 0.1
    
    def _initialize_env(self):
        self.book_shelf=  self._preset.get_props(BookShelf)[0]
        self.books = [Book(self._mojo) for _ in range(self._BOOK_COUNT)]
        self.target_books = [Book(self._mojo) for _ in range(self._TARGET_BOOK_COUNT)]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.target_books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
    
    def _on_reset(self):
        self.book_shelf.set_state(state=0)
        self.book_shelf.set_state(target_state={
            "bookshelf_v2/joint_0": 0,
            "bookshelf_v2/joint_1": 0,
            "bookshelf_v2/joint_2": 0
        })
        
        # Define an incremental translation to each book to spawn books side by side
        x, y, z = 0.6, -0.425, 1.1
        points = [self._BOOK_SHELF_POS]
        for _ in  range(self._BOOK_COUNT):
            y += self._BOOK_SPACING 
            point = np.array([x, y ,z ])
            points.append(point)
        
        for book, point in zip(self.books, points):
            book.body.set_position(point)
            book.body.set_quaternion(
            Quaternion(axis=[1, 0, 0], angle=self._BOOK_ROT).elements
        )
            
        target_points = get_random_points_on_plane(
            len(self.target_books),
            self._TARGET_BOOK_POS,
            self._BOOK_POS_EXTENTS,
            self._BOOK_STEP,
            self._BOOK_POS_BOUNDS,
        )
        
        for target_book, point in zip(self.target_books, target_points):
            target_book.body.set_position(point)
            angle = np.random.uniform(-self._BOOK_ROT_BOUNDS, self._BOOK_ROT_BOUNDS)
            angle = np.deg2rad(-90)
            target_book.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._TARGET_BOOK_ROT + angle).elements
            )
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.target_books,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
    def _on_step(self):
        self._metric_step()
             
    def _success(self) -> bool:
        self._success_check = True
        angle_errors:   dict[str, float] = {}
        distance_errors: dict[str, float] = {}
        for idx, book in enumerate(self.books):                        
            # check book angle 
            mj_q = book.body.get_quaternion()  # [w, x, y, z]
            quat_scipy_order = [mj_q[1], mj_q[2], mj_q[3], mj_q[0]]
            
            rot = Rotation.from_quat(quat_scipy_order)
            
            # convert to euler angles in  degrees 
            rot_euler_deg = rot.as_euler('xyz', degrees=True)
            roll_deg, pitch_deg, yaw_deg = rot_euler_deg

            # check angle error
            angle_error = abs(roll_deg - self._SUCCESS_ROT)
            book_name = f"book {idx}"
            angle_errors[book_name] = angle_error

            print(f'Book Orientation: {roll_deg}, angle_error: {angle_error}')

            if (angle_error > self._SUCCESS_ROT_TOLERANCE):
                print("book angle is not upright")
                self._success_check = False
                self._metric_stage(2, False)
            else:
                print("book angle is upright")
                self._metric_stage(2, True)
                
            if not book.is_colliding(self.book_shelf.lower_shelf_body):
                self._success_check = False
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(book, side):
                    self._success_check = False  
                    self._metric_stage(1)
        
        for idx, target_book in enumerate(self.target_books):
            if not target_book.is_colliding(self.book_shelf.lower_shelf_body):
                self._success_check = False
                self._metric_stage(3, False)
            else:
                self._metric_stage(3, True)
            
            # check book angle 
            mj_q = target_book.body.get_quaternion()  # [w, x, y, z]
            quat_scipy_order = [mj_q[1], mj_q[2], mj_q[3], mj_q[0]]
            
            rot = Rotation.from_quat(quat_scipy_order)
            
            # convert to euler angles in  degrees 
            rot_euler_deg = rot.as_euler('xyz', degrees=True)
            roll_deg, pitch_deg, yaw_deg = rot_euler_deg

            # check angle error
            angle_error = abs(roll_deg - self._SUCCESS_ROT)
            book_name = f"target_book_{idx}"
            angle_errors[book_name] = angle_error

            print(f'Book Orientation: {roll_deg}, angle_error: {angle_error}')

            if (angle_error > self._SUCCESS_ROT_TOLERANCE):
                print("target book angle is not upright")
                self._success_check = False
            else:
                print("target book angle is upright")
                
            # book_up = Quaternion(target_book.body.get_quaternion()).rotate(pitch)
            # angle = np.arccos(np.clip(np.dot(book_up, pitch), -1.0, 1.0))
            # print(f'Target Book Orientation: {np.rad2deg(angle)}')
            # if abs(np.rad2deg(angle) - self._SUCCESS_ROT) > self._SUCCESS_ROT_TOLERANCE:
            #     return False
            
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(target_book, side):
                    self._success_check = False    
                
            # if not any(target_book.is_colliding(book) for book in self.books): # Logic is that there must be that the target book must be touching at least one book form the stack
            #     self._success_check = False
            
            count = 0
            for idx, book in enumerate(self.books):
                # check if the target book is colliding with the stack of books
                print(f"Distance: {distance(target_book.body, book.body)}")
                distance_errors[book_name] = distance(target_book.body, book.body)
                if distance(target_book.body, book.body) < self._SUCCESSFUL_DIST:
                    count += 1
            if count < 2:
                print("target book is not inserted in between the stack of books")
                self._success_check = False
                
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            pose_error=angle_errors,
            target_distance=distance_errors,
        )
        return self._success_check