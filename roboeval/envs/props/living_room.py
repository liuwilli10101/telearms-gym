"""Different pickable items."""
from pathlib import Path

from dm_control import mjcf
from mujoco_utils import mjcf_utils

from roboeval.const import ASSETS_PATH
from abc import ABC
from typing import Any, Dict, List, Optional
import numpy as np
from mojo.elements import Body, MujocoElement, Joint, Site

from roboeval.envs.props.prop import CollidableProp, KinematicProp
from roboeval.utils.physics_utils import set_joint_position, get_joint_position



class Cushion(KinematicProp):
    """Pillow Cushion."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cushion/cushion.xml"


class Sofa(KinematicProp):
    """Sofa."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/sofa/sofa.xml"
    

class Broom(KinematicProp):
    """Broom."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/broom/broom.xml"
    
class WireBrush(KinematicProp):
    """Wire Brush."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/wire_brush/wire_brush.xml"
    
class Dustpan(KinematicProp):
    """Dustpan."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/dustpan/dustpan.xml"
    
class Vase(KinematicProp):
    """Vase."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/vase/vase.xml"
    
class NightStand(KinematicProp):
    """Night stand."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/night_stand/night_stand.xml"
    
    
class SmallTable(KinematicProp):
    """small table."""
 
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/small_table/small_table.xml"

class Lamp(KinematicProp, ABC):
    '''
    Define an articulated lamp object
    '''
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / self.asset_path


    def _post_init(self):
        self._joints = self.body.joints

    def set_state(self, state: Optional[float] = None, target_state: Optional[Dict[str, float]] = None):
        """
        Set the state of the joints.

        Args:
            state (Optional[float]): A single normalized value to set all joints to. If None, `target_state` must be provided.
            target_state (Optional[Dict[str, float]]): A dictionary mapping joint names to their desired normalized values.
        """
        if not self._joints:
            if state is not None or (target_state and len(target_state) > 0):
                print("[WARNING] set_state called but no joints available.")
            return

        if target_state:
            for joint_name, value in target_state.items():
                # Fetch the joint reference
                joint = (Joint.get(self._mojo, joint_name, self.body))
                print(f'{joint_name} is being loaded')
                if joint and hasattr(joint.mjcf, 'range'):
                    set_joint_position(joint, value, True)
                else:
                    print(f"[WARNING] Joint '{joint_name}' not found in {self.asset_path}")
        elif state is not None:
            for joint in self.all_joints:
                if joint.range is None:
                    continue

                parent = joint.root.model
                print(joint.root.model)
                joint_name = f'{parent}/{joint.name}'

                joint_obj = (Joint.get(self._mojo, joint_name, self.body))
                if joint_obj and hasattr(joint_obj.mjcf, 'range') and len(joint.range) > 0:
                    print(joint.range)
                    print(f'{joint_name} is being loaded')
                    set_joint_position(joint_obj, state, True)
        else:
            print("[WARNING] set_state called with neither 'state' nor 'target_state'. Nothing to do.")

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        if not self._joints:
            return np.array([])
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.asset_path =  kwargs.get("asset_path", "props/lamp/lamp.xml") # specify the customizable asset path as input
        self.bodies_to_disable: list[str] = kwargs.get("bodies_to_disable", [])
        self.target_joints: list[str] = kwargs.get("target_joints", [])

    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
          # Disable (remove) specific bodies if requested
        for body_name in self.bodies_to_disable:
            body_elem = model.find("body", body_name)
            if body_elem is not None:
                body_elem.remove()

        # Collect references to *all* bodies, joints, and sites
        self.all_bodies = model.find_all("body")
        self.all_joints = model.find_all("joint")
        self.all_sites  = model.find_all("site")
        
class CoffeeKettle(KinematicProp, ABC):
    '''
    Define an articulated coffee kettle object
    '''
    
    _SPOUT_SITE = "spout"
    
    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / self.asset_path

    def _post_init(self):
        self._joints = self.body.joints

    def set_state(self, state: Optional[float] = None, target_state: Optional[Dict[str, float]] = None):
        """
        Set the state of the joints.

        Args:
            state (Optional[float]): A single normalized value to set all joints to. If None, `target_state` must be provided.
            target_state (Optional[Dict[str, float]]): A dictionary mapping joint names to their desired normalized values.
        """
        if not self._joints:
            if state is not None or (target_state and len(target_state) > 0):
                print("[WARNING] set_state called but no joints available.")
            return

        if target_state:
            for joint_name, value in target_state.items():
                # Fetch the joint reference
                joint = (Joint.get(self._mojo, joint_name, self.body))
                print(f'{joint_name} is being loaded')
                if joint and hasattr(joint.mjcf, 'range'):
                    set_joint_position(joint, value, True)
                else:
                    print(f"[WARNING] Joint '{joint_name}' not found in {self.asset_path}")
        elif state is not None:
            for joint in self.all_joints:
                if joint.range is None:
                    continue

                parent = joint.root.model
                print(joint.root.model)
                joint_name = f'{parent}/{joint.name}'

                joint_obj = (Joint.get(self._mojo, joint_name, self.body))
                if joint_obj and hasattr(joint_obj.mjcf, 'range') and len(joint.range) > 0:
                    print(joint.range)
                    print(f'{joint_name} is being loaded')
                    set_joint_position(joint_obj, state, True)
        else:
            print("[WARNING] set_state called with neither 'state' nor 'target_state'. Nothing to do.")

    def get_state(self) -> np.ndarray[float]:
        """Get normalized state of joints."""
        if not self._joints:
            return np.array([])
        return np.array([get_joint_position(joint, True) for joint in self._joints])

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.asset_path =  kwargs.get("asset_path", "props/coffee_kettle/coffee_kettle.xml") # specify the customizable asset path as input
        self.bodies_to_disable: list[str] = kwargs.get("bodies_to_disable", [])
        self.target_joints: list[str] = kwargs.get("target_joints", [])

    def _on_loaded(self, model): 
        object = MujocoElement(self._mojo, model)   
        self.spout = Site.get(self._mojo, self._SPOUT_SITE, object)
        
        # Disable (remove) specific bodies if requested
        for body_name in self.bodies_to_disable:
            body_elem = model.find("body", body_name)
            if body_elem is not None:
                body_elem.remove()

        # Collect references to *all* bodies, joints, and sites
        self.all_bodies = model.find_all("body")
        self.all_joints = model.find_all("joint")
        self.all_sites  = model.find_all("site")
        
    
