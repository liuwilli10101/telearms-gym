from abc import ABC
from typing import Any

import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.custom import Toast
from roboeval.envs.props.partnet import Toaster
from roboeval.utils.geometry import check_sites_intersection_mojo
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance


TOASTER_INIT_POS = np.array([0.85, -0.1, 0.65])
TOAST_PIECES_POSITIONS = [
    np.array([0.85, 0.25, 1.0]),
    np.array([0.85, 0.35, 1.0]),
    np.array([0.85, 0.45, 1.0]),
    np.array([0.85, 0.55, 1.0]),
]
TOAST_MAX_NUM = len(TOAST_PIECES_POSITIONS)


class BreadInToaster(RoboEvalEnv, ABC, MetricRolloutEval):
    """Place bread in toaster base environment (static)"""

    _TOASTER_ROT_BOUNDS = np.deg2rad(90)
    _TOAST_ROT_BOUNDS = np.deg2rad(90)

    _PRESET_PATH = PRESETS_PATH / "bread_in_toaster.yaml"
    
    _success_check = True
    _final_metrics = {}

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self._num_toast = kwargs.get("num_toast", 1)
        self._use_stack = kwargs.get("use_stack", False)

        self._num_toast = max(1, min(self._num_toast, 4))

    def _initialize_env(self):
        self.toaster = Toaster(self._mojo)
        self.toasts = [Toast(self._mojo) for _ in range(self._num_toast)]
        
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.toasts,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )

    def _on_reset(self):
        self.toaster.set_pose(position=TOASTER_INIT_POS)
        for i, toast in enumerate(self.toasts):
            toast.set_pose(
                position=TOAST_PIECES_POSITIONS[i],
                quat=Quaternion().elements,
            )
            
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.toasts,
            slip_sample_window=20,
            track_collisions=True,
            robot=self.robot
        )
        
        for idx in range(1,4): self._metric_stage(idx, False)


    def _success(self) -> bool:
        obj_dist = {}
        for site in self.toaster.sites:
            if site.mjcf.name == "boundary":
                continue
            self._mojo.physics.bind(site.mjcf).rgba = [0, 1, 0, 0.25]

        num_success = 0
        for idx_toast, toast in enumerate(self.toasts):
            for idx_site, site in enumerate(self.toaster.sites):
                if site.mjcf.name == "boundary":
                    continue
                if check_sites_intersection_mojo(self._mojo, toast.aabb, site):
                    num_success += 1
                    self._mojo.physics.bind(site.mjcf).rgba = [1, 0, 0, 0.25]
                    
                # —————————— distance tracking —————————————
                obj_dist[f"toast {idx_toast}-site{idx_site}"] = distance(site, toast.body)
                obj_dist[f"toast {idx_toast}-right gripper"] = distance(toast.body, self.robot.grippers[HandSide.RIGHT].body)
                obj_dist[f"toast {idx_toast}-left gripper"] = distance(toast.body, self.robot.grippers[HandSide.LEFT].body)
                
                # —————————— stage monitoring —————————————
                toast_site_check = check_sites_intersection_mojo(self._mojo, toast.aabb, site)
                grasping = any(self.robot.is_gripper_holding_object(toast, side) for side in self.robot.grippers)
                lift_dist = toast.body.get_position()[2] - 0.914
                obj_dist[f"toast {idx_toast} lift distance"] = lift_dist
                
                if grasping: self._metric_stage(1)
                if lift_dist > 0.1 and grasping: self._metric_stage(2)
                if toast_site_check: self._metric_stage(3)

                    
        self._success_check = num_success >= len(self.toasts)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,  
            target_distance=obj_dist
        )
        return num_success >= len(self.toasts)

    def _fail(self) -> bool:
        if super()._fail():
            return True
        if self.toaster.is_colliding(self.floor):
            return True
        for toast in self.toasts:
            if toast.is_colliding(self.floor):
                return True

        return False
    
    def _on_step(self):
        self._metric_step()

    def _get_task_info(self):   
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})
    
class BreadInToasterPosition(BreadInToaster):
    """Bread in toaster with randomized position of toast pieces and toaster"""
    def _on_reset(self):
        super()._on_reset()
        spawn = self.get_spawn_boundary("workspace")
        if spawn:
            spawn.clear()
            for toast in self.toasts:
                spawn.add_no_sample(toast)
            spawn.sample(self.toaster, ignore_collisions=False, min_distance=0.1)

    
class BreadInToasterOrientation(BreadInToaster):
    """Bread in toaster with randomized orientation of toast pieces and toaster"""
    def _on_reset(self):
        super()._on_reset()
        toaster_angle = np.random.uniform(-self._TOASTER_ROT_BOUNDS, self._TOASTER_ROT_BOUNDS)
        self.toaster.set_pose(
            position=TOASTER_INIT_POS,
            quat=Quaternion(axis=[0, 0, 1], angle=toaster_angle).elements,
        )
        for i, toast in enumerate(self.toasts):
            toast_angle = np.random.uniform(-self._TOAST_ROT_BOUNDS, self._TOAST_ROT_BOUNDS)
            toast.set_pose(
                position=TOAST_PIECES_POSITIONS[i],quat=Quaternion(axis=[0, 0, 1], angle=toast_angle).elements,
            )

class BreadInToasterPositionAndOrientation(BreadInToaster):
    """Bread in toaster with randomized position AND orientation of bread and toaster"""
    def _on_reset(self):
        super()._on_reset()

        spawn = self.get_spawn_boundary("workspace")
        if spawn:
            spawn.clear()
            for toast in self.toasts:
                spawn.add_no_sample(toast)
            spawn.sample(self.toaster, ignore_collisions=False, min_distance=0.1)
            
        toaster_angle = np.random.uniform(-self._TOASTER_ROT_BOUNDS, self._TOASTER_ROT_BOUNDS)
        self.toaster.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=toaster_angle).elements)
        for toast in self.toasts:
            toast_angle = np.random.uniform(-self._TOAST_ROT_BOUNDS, self._TOAST_ROT_BOUNDS)
            toast.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=toast_angle).elements)
            
    
    
