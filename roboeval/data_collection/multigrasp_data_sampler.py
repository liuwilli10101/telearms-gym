# Multi-Grasp Sampler with Placement Propagation
# This script samples N grasping poses from TSRs, validates them, and propagates them forward
# based on desired object placement poses.
# 
# Workflow:
# 1. Sample N grasp poses for the object and validate them with motion planning
# 2. Sample M valid placement poses for the object itself (where the object should be placed)
# 3. For each valid grasp, calculate corresponding gripper placement poses by:
#    - Computing relative transform from original object pose to each placement pose
#    - Applying same transform to grasp pose to get required gripper placement pose
# 4. Validate gripper placement poses with motion planning
# 5. Retain only grasps with at least one valid placement and execute the best one

# Imports
try:
    from moviepy.editor import VideoClip
    import pygame  # noqa: F401
except ImportError:
    raise ImportError(
        "Please install moviepy and pygame for this example. "
        "i.e. `pip install moviepy pygame`"
    )

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../thirdparty/curobo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../thirdparty/tsr/src/tsr'))  # Note: added /src

from roboeval.envs.manipulation import StackTwoBlocks, StackNBlocks
from roboeval.envs.task_library.move_cube import PushBlockToTarget
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda, SinglePanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
import numpy as np
from scipy.spatial.transform import Rotation as R
# Third Party
import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from tsr import TSR, TSRChain

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from functools import reduce
import builtins
builtins.reduce = reduce

# Define utility functions (copied from curobo_plan.py to avoid import issues)

def xmat_to_quat(data):
    """
    Convert a body's xmat (flattened 3x3 rotation matrix) from mjData to a MuJoCo-style quaternion [w, x, y, z].
    """
    R_flat = data.xmat       # shape (9,)
    R_mat = np.reshape(R_flat, (3, 3))      # shape (3, 3)
    quat_xyzw = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
    quat_wxyz = np.concatenate(([quat_xyzw[3]], quat_xyzw[:3]))  # [w, x, y, z]
    return quat_wxyz

def get_mesh_dict(env):
    mesh_dict = {}
    for prop in env._preset.get_props():
        prop_path = prop._model_path.parent
        for geom in prop._geoms:
            geom_filename = geom.mjcf.mesh.file.get_vfs_filename(filename_with_hash=False)
            if geom_filename is not None:
                mesh_dict[geom.mjcf.full_identifier] = str(prop_path / "assets" / geom_filename)
    return mesh_dict

def generate_world_config(mesh_dict, physics, geom_group=3, mesh_test=True, keywords_to_omit=[]):
    world_config = {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}}
    
    collision_geom_idx = np.where(physics.model.geom_group == geom_group)[0]
    
    for i in collision_geom_idx:
        geom = physics.model.geom(i)
        skip_geom = False
        for keyword in keywords_to_omit:
            if keyword in geom.name:
                skip_geom = True
                break
        if skip_geom:
            continue
        geom_data = physics.data.geom(i)
        quaternion = xmat_to_quat(geom_data)

        if geom.type[0] == 3:  # Capsule
            world_config["capsule"][geom.name] = {
                "name": geom.name,
                "radius": geom.size[0],
                "height": geom.size[1],
                "pose": np.concatenate([geom_data.xpos, quaternion])
            }
        elif geom.type[0] == 5:  # Cylinder
            world_config["cylinder"][geom.name] = {
                "name": geom.name,
                "radius": geom.size[0],
                "height": geom.size[1],
                "pose": np.concatenate([geom_data.xpos, quaternion])
            }
        elif geom.type[0] == 6:  # Cuboid

            # Multiply dims by 2 to match mujoco's size convention
            cuboid_dims = np.array(geom.size)*2
            cuboid_dims_list = list(cuboid_dims)
            world_config["cuboid"][geom.name] = {
                "dims": cuboid_dims_list,
                "pose": np.concatenate([geom_data.xpos, quaternion])
            }

            # Shift the pose of cuboid down to match the bottom face
            world_config["cuboid"][geom.name]["pose"][2] -= cuboid_dims[2] / 2.0
        elif geom.type[0] == 7:  # Mesh
            mesh_file = mesh_dict.get(geom.name, None)
            print(f"Mesh geom size: {geom.size}")
            world_config["mesh"][geom.name] = {
                "file_path": mesh_file,
                "pose": np.concatenate([geom_data.xpos, quaternion])
            }

    if mesh_test:
        for key, value in world_config["mesh"].items():
            print(f"{key}: {value['file_path']}")
    
    return world_config

def get_robot_transform(physics, robot_name):
    t = physics.named.data.xpos[robot_name]
    R_flat = physics.named.data.xmat[robot_name]  # shape (9,)
    R = np.reshape(R_flat, (3, 3))  # shape (3, 3)
    return {"R": R, "t": -R @ t}

def transform_objects_to_robot_frame(objects, robot_transform):
    R = robot_transform["R"]
    t = robot_transform["t"]
    result = objects.copy()
    
    for key, v in objects["cuboid"].items():
        new_pose = R @ v["pose"][:3] + t
        result["cuboid"][key]["pose"][:3] = new_pose

    for key, v in objects["mesh"].items():
        new_pose = R @ v["pose"][:3] + t
        result["mesh"][key]["pose"][:3] = new_pose

    return result

def transform_target_to_robot_frame(target, robot_transform):
    """Transform a target pose to the robot frame."""
    rotation = robot_transform["R"]
    translation = robot_transform["t"]

    target = np.array(target)
    if target.shape != (7,):
        raise ValueError("Target must be a 7-element array: [x, y, z, qw, qx, qy, qz]")
    
    new_pose = rotation @ target[:3] + translation
    new_quat = R.from_quat(target[3:], scalar_first=True).as_quat(scalar_first=True)
    
    return np.concatenate([new_pose, new_quat])

def get_object_pose(env, object_name, idx=None):
    """Function to get the pose of an object in the mujoco environment."""
    prop = getattr(env, object_name)

    if idx is not None:
        prop = prop[idx] if isinstance(prop, list) else prop
    if prop is None:
        raise ValueError(f"Object {object_name} not found in the environment.")

    prop_pose = prop.get_pose()
    return prop_pose

# TSR generation functions
DEFAULT_PLACEMENT_SPEC = {
    "relative_position": ["above", "below", "left", "right", "front", "behind"],
    "upright_face": ["top", "bottom", "left", "right", "front", "back"],
    "in_plane_rotation": [0, 90, 180, 270],
    "allow_rotation_about_up": False,
    "pose_tolerance": {
        "position": 0.0,
        "rotation": 0.0,
    }
}

def deep_merge_spec(default, override):
    spec = default.copy()
    if override is None:
        return spec
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            spec[k] = {**spec[k], **v}
        else:
            spec[k] = v
    return spec

def get_relative_offsets(positions, cube_size):
    direction_map = {
        "above": [0, 0, cube_size],
        "below": [0, 0, -cube_size],
        "left": [-cube_size, 0, 0],
        "right": [cube_size, 0, 0],
        "front": [0, -cube_size, 0],
        "behind": [0, cube_size, 0],
    }
    if not positions:
        return [np.zeros(3)]
    return [np.array(direction_map[p]) for p in positions]

def get_upright_rotations(faces):
    if not faces:
        return [np.eye(3)]
    
    face_to_z = {
        "top": np.eye(3),
        "bottom": R.from_euler("x", 180, degrees=True).as_matrix(),
        "left": R.from_euler("y", -90, degrees=True).as_matrix(),
        "right": R.from_euler("y", 90, degrees=True).as_matrix(),
        "front": R.from_euler("x", -90, degrees=True).as_matrix(),
        "back": R.from_euler("x", 90, degrees=True).as_matrix(),
    }
    return [face_to_z[face] for face in faces]

def get_inplane_rotations(rotations, allow_any=False):
    if allow_any or rotations is None:
        return [R.from_euler("z", deg, degrees=True).as_matrix() for deg in [0, 90, 180, 270]]
    return [R.from_euler("z", deg, degrees=True).as_matrix() for deg in rotations]

# def generate_placement_tsrs(base_pose, cube_size, spec_overrides=None):
#     """Generate TSRChain for placing a cube relative to another cube."""
#     spec = deep_merge_spec(DEFAULT_PLACEMENT_SPEC, spec_overrides)

#     offsets = get_relative_offsets(spec["relative_position"], cube_size)
#     upright_rotations = get_upright_rotations(spec["upright_face"])
#     inplane_rotations = get_inplane_rotations(spec["in_plane_rotation"], spec["allow_rotation_about_up"])

#     pos_tol = spec["pose_tolerance"].get("position", 0.0)
#     rot_tol = np.radians(spec["pose_tolerance"].get("rotation", 0.0))

#     tsrs = []

#     for offset in offsets:
#         T0_w = np.eye(4)
#         T0_w[:3, 3] = base_pose[:3, 3] + base_pose[:3, :3] @ offset

#         for R_up in upright_rotations:
#             for R_inplane in inplane_rotations:
#                 R_total = R_up @ R_inplane

#                 Tw_e = np.eye(4)
#                 Tw_e[:3, :3] = R_total

#                 Bw = np.zeros((6, 2))
#                 Bw[:3, 0] = -pos_tol
#                 Bw[:3, 1] = pos_tol
#                 Bw[3:, 0] = -rot_tol
#                 Bw[3:, 1] = rot_tol

#                 tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
#                 tsrs.append(TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr))
    
#     return tsrs

def box_grasp(T0_w, length, width, height, lateral_offset=0.0, lateral_tolerance=0.02, rotation_tolerance=0.2, **kwargs):
    """Generate a list of TSRChain objects for grasping a box."""
    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError('Box dimensions must be > 0')

    if lateral_tolerance < 0.0:
        raise ValueError('lateral_tolerance must be >= 0.0')

    chain_list = []

    def make_chain(Tw_e, Bw):
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        return TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr)

    # Top face (approach from +z, palm toward -z)
    Tw_e_top = np.array([[1.,  0.,  0., 0.],
                         [0., -1.,  0., 0.],
                         [0.,  0., -1., height + lateral_offset],
                         [0.,  0.,  0., 1.]])
    Bw_top = np.zeros((6, 2))
    Bw_top[0, :] = [-lateral_tolerance, lateral_tolerance]
    Bw_top[4, :] = [-rotation_tolerance, rotation_tolerance]
    chain_list.append(make_chain(Tw_e_top, Bw_top))

    # Bottom face (approach from -z, palm toward +z)
    Tw_e_bottom = np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., -lateral_offset],
                            [0., 0., 0., 1.]])
    Bw_bottom = np.zeros((6, 2))
    Bw_bottom[0, :] = [-lateral_tolerance, lateral_tolerance]
    chain_list.append(make_chain(Tw_e_bottom, Bw_bottom))

    # Front face (+x direction, z into box x)
    Tw_e_front = np.array([[0., -1.,  0., 0.5 * length + lateral_offset],
                           [0.,  0., -1., 0.5 * height],
                           [1.,  0.,  0., 0.],
                           [0.,  0.,  0., 1.]])
    Bw_front = np.zeros((6, 2))
    Bw_front[2, :] = [-lateral_tolerance, lateral_tolerance]
    chain_list.append(make_chain(Tw_e_front, Bw_front))

    # Back face (-x direction)
    Tw_e_back = np.array([[0.,  1.,  0., -0.5 * length - lateral_offset],
                          [0.,  0., -1., 0.5 * height],
                          [-1., 0.,  0., 0.],
                          [0.,  0.,  0., 1.]])
    Bw_back = np.zeros((6, 2))
    Bw_back[2, :] = [-lateral_tolerance, lateral_tolerance]
    chain_list.append(make_chain(Tw_e_back, Bw_back))

    # Left face (+y direction)
    Tw_e_left = np.array([[1.,  0.,  0., 0.],
                          [0.,  0., -1., 0.5 * width + lateral_offset],
                          [0.,  1.,  0., 0.5 * height],
                          [0.,  0.,  0., 1.]])
    Bw_left = np.zeros((6, 2))
    Bw_left[0, :] = [-lateral_tolerance, lateral_tolerance]
    chain_list.append(make_chain(Tw_e_left, Bw_left))

    # Right face (-y direction)
    Tw_e_right = np.array([[1.,  0.,  0., 0.],
                           [0.,  0.,  1., -0.5 * width - lateral_offset],
                           [0., -1.,  0., 0.5 * height],
                           [0.,  0.,  0., 1.]])
    Bw_right = np.zeros((6, 2))
    Bw_right[0, :] = [-lateral_tolerance, lateral_tolerance]
    chain_list.append(make_chain(Tw_e_right, Bw_right))

    # Add 180° z-rotated versions of each TSR
    rotated_chain_list = []
    R_z180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0., 0.],
                       [np.sin(np.pi),  np.cos(np.pi), 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])
    for c in chain_list:
        tsr = c.TSRs[0]
        Tw_e_rot = np.dot(tsr.Tw_e, R_z180)
        tsr_rot = TSR(T0_w=tsr.T0_w, Tw_e=Tw_e_rot, Bw=tsr.Bw)
        rotated_chain_list.append(TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr_rot))

    return chain_list + rotated_chain_list

@dataclass
class GraspPoseSample:
    """Container for a grasp pose sample with its metadata."""
    pose_matrix: np.ndarray  # 4x4 transformation matrix
    pose_vector: np.ndarray  # [x, y, z, qw, qx, qy, qz]
    transformed_pose: np.ndarray  # pose in robot frame
    tsr_chain_idx: int  # which TSR chain this came from
    is_valid: bool = False
    valid_placements: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.valid_placements is None:
            self.valid_placements = []

@dataclass
class PlacementPoseSample:
    """Container for a placement pose sample with its metadata."""
    pose_matrix: np.ndarray  # 4x4 transformation matrix
    pose_vector: np.ndarray  # [x, y, z, qw, qx, qy, qz]
    transformed_pose: np.ndarray  # pose in robot frame

class MultiGraspSampler:
    """
    Samples multiple grasping poses from TSRs, validates them using motion planning,
    and propagates them forward based on placement poses.
    """
    
    def __init__(self, env, motion_gen, robot_transform, num_grasp_samples=10, num_placement_samples=5):
        self.env = env
        self.motion_gen = motion_gen
        self.robot_transform = robot_transform
        self.num_grasp_samples = num_grasp_samples
        self.num_placement_samples = num_placement_samples
        
    def sample_grasp_poses(self, object_pose: np.ndarray, object_dims: Tuple[float, float, float]) -> List[GraspPoseSample]:
        """
        Sample N grasp poses from TSR chains for the given object.
        
        Args:
            object_pose: [x, y, z, qw, qx, qy, qz] pose of the object
            object_dims: (length, width, height) dimensions of the object
            
        Returns:
            List of GraspPoseSample objects
        """
        print(f"Sampling {self.num_grasp_samples} grasp poses...")
        
        # Create object transformation matrix
        rotation = R.from_quat(object_pose[3:], scalar_first=True).as_matrix()
        box_transform = np.eye(4)
        box_transform[:3, :3] = rotation
        box_transform[:3, 3] = object_pose[:3]
        
        # Generate TSR chains for grasping
        box_tsr_chains = box_grasp(
            T0_w=box_transform,
            length=object_dims[0],
            width=object_dims[1],
            height=object_dims[2],
            lateral_offset=0.05,
            lateral_tolerance=0.00
        )
        
        grasp_samples = []
        samples_per_chain = max(1, self.num_grasp_samples // len(box_tsr_chains))
        
        for chain_idx, tsr_chain in enumerate(box_tsr_chains):
            for sample_idx in range(samples_per_chain):
                try:
                    # Sample a pose from this TSR chain
                    pose_matrix = tsr_chain.sample()
                    
                    # Convert to pose vector
                    pose_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat(scalar_first=True)
                    pose_vec = pose_matrix[:3, 3]
                    pose_vector = np.concatenate([pose_vec, pose_quat])
                    
                    # Transform to robot frame
                    transformed_pose = transform_target_to_robot_frame(pose_vector, self.robot_transform)
                    
                    # Create grasp sample
                    grasp_sample = GraspPoseSample(
                        pose_matrix=pose_matrix,
                        pose_vector=pose_vector,
                        transformed_pose=transformed_pose,
                        tsr_chain_idx=chain_idx
                    )
                    
                    grasp_samples.append(grasp_sample)
                    
                except Exception as e:
                    print(f"Failed to sample grasp pose from chain {chain_idx}, sample {sample_idx}: {e}")
                    continue
                    
        # If we didn't get enough samples, fill with random selections
        while len(grasp_samples) < self.num_grasp_samples:
            try:
                chain_idx = random.randint(0, len(box_tsr_chains) - 1)
                tsr_chain = box_tsr_chains[chain_idx]
                
                pose_matrix = tsr_chain.sample()
                pose_quat = R.from_matrix(pose_matrix[:3, :3]).as_quat(scalar_first=True)
                pose_vec = pose_matrix[:3, 3]
                pose_vector = np.concatenate([pose_vec, pose_quat])
                transformed_pose = transform_target_to_robot_frame(pose_vector, self.robot_transform)
                
                grasp_sample = GraspPoseSample(
                    pose_matrix=pose_matrix,
                    pose_vector=pose_vector,
                    transformed_pose=transformed_pose,
                    tsr_chain_idx=chain_idx
                )
                
                grasp_samples.append(grasp_sample)
                
            except Exception as e:
                print(f"Failed to sample additional grasp pose: {e}")
                break
                
        print(f"Successfully sampled {len(grasp_samples)} grasp poses")
        return grasp_samples[:self.num_grasp_samples]
    
    def validate_grasp_poses(self, grasp_samples: List[GraspPoseSample], start_state: JointState) -> List[GraspPoseSample]:
        """
        Validate grasp poses by attempting to plan motion to them.
        
        Args:
            grasp_samples: List of grasp poses to validate
            start_state: Starting robot configuration
            
        Returns:
            List of valid grasp poses
        """
        print(f"Validating {len(grasp_samples)} grasp poses...")
        
        valid_grasps = []
        
        for idx, grasp_sample in enumerate(grasp_samples):
            try:
                # Create goal pose
                goal_pose = Pose.from_list(grasp_sample.transformed_pose.tolist())
                
                # Plan motion to grasp pose
                result = self.motion_gen.plan_single(
                    start_state, 
                    goal_pose, 
                    MotionGenPlanConfig(max_attempts=3)
                )
                
                if result.success:
                    grasp_sample.is_valid = True
                    valid_grasps.append(grasp_sample)
                    print(f"Grasp {idx+1}/{len(grasp_samples)}: VALID")
                else:
                    print(f"Grasp {idx+1}/{len(grasp_samples)}: INVALID - motion planning failed, {result.status}")
                    
            except Exception as e:
                print(f"Grasp {idx+1}/{len(grasp_samples)}: INVALID - {e}")
                continue
                
        print(f"Found {len(valid_grasps)} valid grasp poses out of {len(grasp_samples)}")
        return valid_grasps
    
    def sample_object_placement_poses(self, target_object_pose: np.ndarray, target_object_dims: Tuple[float, float, float],
                                     placed_object_dims: Tuple[float, float, float], placement_spec: Dict = None) -> List[PlacementPoseSample]:
        """
        Sample placement poses for the object itself relative to a target object.
        
        Args:
            target_object_pose: [x, y, z, qw, qx, qy, qz] pose of the target object
            target_object_dims: (length, width, height) of the target object
            placed_object_dims: (length, width, height) of the object being placed
            placement_spec: Dict specifying placement constraints (relative_position, upright_face, etc.)
            
        Returns:
            List of PlacementPoseSample objects representing where the object should be placed
        """
        print(f"Sampling {self.num_placement_samples} object placement poses...")
        
        # Use default placement spec if none provided
        if placement_spec is None:
            placement_spec = {
                "relative_position": ["above"],
                "upright_face": ["top"],
                "allow_rotation_about_up": True,
                "pose_tolerance": {
                    "position": 0.01,
                    "rotation": 5.0
                }
            }
        
        # Get the relative offsets and orientations based on the spec
        offsets = get_relative_offsets(placement_spec.get("relative_position", ["above"]), placed_object_dims[0])
        upright_rotations = get_upright_rotations(placement_spec.get("upright_face", ["top"]))
        allow_rotation_about_up = placement_spec.get("allow_rotation_about_up", False)
        
        # Get position and rotation tolerances
        pos_tolerance = placement_spec.get("pose_tolerance", {}).get("position", 0.01)
        rot_tolerance = placement_spec.get("pose_tolerance", {}).get("rotation", 5.0)
        
        # Create target object transformation matrix
        target_rotation = R.from_quat(target_object_pose[3:], scalar_first=True).as_matrix()
        target_transform = np.eye(4)
        target_transform[:3, :3] = target_rotation
        target_transform[:3, 3] = target_object_pose[:3]
        
        placement_samples = []
        
        # Generate samples by combining offsets and orientations
        samples_per_combination = max(1, self.num_placement_samples // (len(offsets) * len(upright_rotations)))
        
        for offset in offsets:
            for upright_rotation in upright_rotations:
                for _ in range(samples_per_combination):
                    try:
                        # Calculate base position: target position + offset in target's frame
                        base_position = target_object_pose[:3] + target_rotation @ offset
                        
                        # Add random position noise within tolerance
                        if pos_tolerance > 0:
                            position_noise = np.random.uniform(-pos_tolerance, pos_tolerance, 3)
                            base_position += position_noise
                        
                        # Start with the upright rotation
                        final_rotation = upright_rotation.copy()
                        
                        # Add random in-plane rotation if allowed
                        if allow_rotation_about_up:
                            if rot_tolerance > 0:
                                # Random rotation around z-axis within tolerance
                                z_rotation = np.random.uniform(-rot_tolerance, rot_tolerance)
                            else:
                                # Random rotation from 0-360 degrees
                                z_rotation = np.random.uniform(0, 360)
                            
                            in_plane_rot = R.from_euler("z", z_rotation, degrees=True).as_matrix()
                            final_rotation = final_rotation @ in_plane_rot
                        
                        # Convert rotation to quaternion
                        rotation_quat = R.from_matrix(final_rotation).as_quat(scalar_first=True)  # [w, x, y, z]
                        
                        # Create pose vector
                        pose_vector = np.concatenate([base_position, rotation_quat])
                        
                        # Create transformation matrix for consistency
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = final_rotation
                        pose_matrix[:3, 3] = base_position
                        
                        # Transform to robot frame
                        transformed_pose = transform_target_to_robot_frame(pose_vector, self.robot_transform)
                        
                        # Create placement sample
                        placement_sample = PlacementPoseSample(
                            pose_matrix=pose_matrix,
                            pose_vector=pose_vector,
                            transformed_pose=transformed_pose,
                        )
                        
                        placement_samples.append(placement_sample)
                        
                        # Stop if we have enough samples
                        if len(placement_samples) >= self.num_placement_samples:
                            break
                            
                    except Exception as e:
                        print(f"Failed to sample placement pose: {e}")
                        continue
                        
                if len(placement_samples) >= self.num_placement_samples:
                    break
            if len(placement_samples) >= self.num_placement_samples:
                break
                
        print(f"Successfully sampled {len(placement_samples)} object placement poses")
        return placement_samples[:self.num_placement_samples]
    
    def propagate_grasps_to_placements(self, valid_grasps: List[GraspPoseSample], 
                                     original_object_pose: np.ndarray,
                                     object_placement_samples: List[PlacementPoseSample]) -> List[GraspPoseSample]:
        """
        For each valid grasp, calculate the corresponding gripper placement poses based on 
        object placement poses and check which ones are reachable.
        
        Args:
            valid_grasps: List of validated grasp poses
            original_object_pose: [x, y, z, qw, qx, qy, qz] original pose of the object being grasped
            object_placement_samples: List of desired object placement poses
            
        Returns:
            List of grasp poses that have at least one valid placement
        """
        print(f"Propagating {len(valid_grasps)} grasps to {len(object_placement_samples)} object placements...")
        
        grasps_with_placements = []
        
        for grasp_idx, grasp_sample in enumerate(valid_grasps):
            print(f"Testing grasp {grasp_idx+1}/{len(valid_grasps)}...")
            
            # First, plan motion to this grasp pose to get the joint state after grasping
            robot_pose = self.env._robot.qpos_actuated[:-1]
            start_angles = robot_pose.tolist()
            start_state = JointState.from_position(
                torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                           "panda_joint5", "panda_joint6","panda_joint7"]
            )
            
            goal_pose = Pose.from_list(grasp_sample.transformed_pose.tolist())
            grasp_result = self.motion_gen.plan_single(
                start_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=1)
            )
            
            if not grasp_result.success:
                print(f"  Grasp motion planning failed, skipping...")
                continue
                
            # Get the final joint state after grasping
            grasp_traj = grasp_result.get_interpolated_plan()
            grasp_end_state = JointState.from_position(torch.tensor(grasp_traj[-1].position.cpu().numpy().tolist(), dtype=torch.float32).cuda().reshape(1, -1),
                                                      joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                                                                   "panda_joint5", "panda_joint6","panda_joint7"])

            # Calculate corresponding gripper placement poses for each object placement
            gripper_placement_poses = self.calculate_gripper_placement_poses(
                grasp_sample, original_object_pose, object_placement_samples
            )
            
            # Test each gripper placement pose
            valid_placement_count = 0
            for placement_idx, gripper_placement_pose in enumerate(gripper_placement_poses):
                try:
                    # Plan motion from grasp end state to gripper placement pose
                    placement_goal_pose = Pose.from_list(gripper_placement_pose.tolist())
                    
                    placement_result = self.motion_gen.plan_single(
                        grasp_end_state,
                        placement_goal_pose,
                        MotionGenPlanConfig(max_attempts=1)
                    )
                    
                    if placement_result.success:
                        grasp_sample.valid_placements.append(gripper_placement_pose)
                        valid_placement_count += 1
                        
                except Exception as e:
                    print(f"    Placement {placement_idx+1} failed: {e}")
                    continue
                    
            print(f"  Found {valid_placement_count} valid placements for this grasp")
            
            # Only keep grasps that have at least one valid placement
            if valid_placement_count > 0:
                grasps_with_placements.append(grasp_sample)
                
        print(f"Retained {len(grasps_with_placements)} grasps with valid placements")
        return grasps_with_placements
    
    def execute_best_grasp_and_placement(self, final_grasps: List[GraspPoseSample], 
                                       camera_views: List[str] = ["external", "head"]) -> Dict:
        """
        Execute the best grasp and one of its valid placements.
        
        Args:
            final_grasps: List of grasps with valid placements
            camera_views: List of camera views to record
            
        Returns:
            Dictionary containing execution results and recorded observations
        """
        if not final_grasps:
            print("No valid grasps with placements found!")
            return {"success": False, "observations": {}}
            
        # Select the grasp with the most valid placements
        best_grasp = max(final_grasps, key=lambda g: len(g.valid_placements))
        print(f"Executing best grasp with {len(best_grasp.valid_placements)} valid placements")
        
        # Set up observation recording
        recorded_observations = {}
        for camera_view in camera_views:
            recorded_observations[camera_view] = []
            
        try:
            # Execute grasp motion
            robot_pose = self.env._robot.qpos_actuated[:-1]
            start_angles = robot_pose.tolist()
            start_state = JointState.from_position(
                torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                           "panda_joint5", "panda_joint6","panda_joint7"]
            )
            
            goal_pose = Pose.from_list(best_grasp.transformed_pose.tolist())
            grasp_result = self.motion_gen.plan_single(
                start_state, 
                goal_pose, 
                MotionGenPlanConfig(max_attempts=1)
            )
            
            if not grasp_result.success:
                print("Failed to plan grasp motion during execution")
                return {"success": False, "observations": recorded_observations}
                
            # Execute grasp trajectory
            grasp_traj = grasp_result.get_interpolated_plan()
            for i in range(grasp_traj.shape[0]):
                traj_step = grasp_traj[i].position.cpu().numpy()
                traj_step = np.concatenate([traj_step, np.zeros(1)])  # Open gripper
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Close gripper
            traj_step[-1] = 1  # Close the gripper
            for i in range(20):
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Attach object to robot (disable collision checking)
            self.motion_gen.attach_objects_to_robot(
                joint_state=grasp_traj[-1],
                object_names=["cube/collision"],
            )
            
            # Execute placement motion using the first valid placement
            placement_pose = best_grasp.valid_placements[0]
            placement_goal_pose = Pose.from_list(placement_pose.tolist())
            
            start_pose = JointState.from_position(
                grasp_traj[-1].position.cuda().reshape(1, -1),
                joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                             "panda_joint5", "panda_joint6","panda_joint7"]
            )
            placement_result = self.motion_gen.plan_single(
                start_pose,
                placement_goal_pose,
                MotionGenPlanConfig(max_attempts=60)
            )
            
            if not placement_result.success:
                print("Failed to plan placement motion during execution")
                return {"success": False, "observations": recorded_observations}
                
            # Execute placement trajectory
            placement_traj = placement_result.get_interpolated_plan()
            for i in range(placement_traj.shape[0]):
                traj_step = placement_traj[i].position.cpu().numpy()
                traj_step = np.concatenate([traj_step, np.ones(1)])  # Keep gripper closed
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Open gripper to release object
            traj_step[-1] = 0  # Open the gripper
            for i in range(20):
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Detach object from robot
            self.motion_gen.detach_object_from_robot()
            
            print("Successfully executed grasp and placement!")
            return {"success": True, "observations": recorded_observations}
            
        except Exception as e:
            print(f"Execution failed: {e}")
            return {"success": False, "observations": recorded_observations}

    def calculate_gripper_placement_poses(self, grasp_sample: GraspPoseSample, 
                                         original_object_pose: np.ndarray,
                                         object_placement_poses: List[PlacementPoseSample]) -> List[np.ndarray]:
        """
        Calculate where the gripper needs to be for each object placement pose.
        
        This works by:
        1. Computing the relative transform from the original object pose to each placement pose
        2. Applying that same transform to the grasp pose to get the gripper placement pose
        
        Args:
            grasp_sample: The grasp pose relative to the original object
            original_object_pose: [x, y, z, qw, qx, qy, qz] original pose of the object
            object_placement_poses: List of desired object placement poses
            
        Returns:
            List of gripper poses in robot frame for each object placement
        """
        gripper_placement_poses = []
        
        # Convert original object pose to transformation matrix
        orig_rot = R.from_quat(original_object_pose[3:], scalar_first=True).as_matrix()
        orig_transform = np.eye(4)
        orig_transform[:3, :3] = orig_rot
        orig_transform[:3, 3] = original_object_pose[:3]
        
        # Convert grasp pose to transformation matrix
        grasp_rot = R.from_quat(grasp_sample.pose_vector[3:], scalar_first=True).as_matrix()
        grasp_transform = np.eye(4)
        grasp_transform[:3, :3] = grasp_rot
        grasp_transform[:3, 3] = grasp_sample.pose_vector[:3]
        
        # Calculate relative transform from original object pose to grasp pose
        # T_object_to_gripper = T_object^-1 * T_grasp
        object_to_gripper_transform = np.linalg.inv(orig_transform) @ grasp_transform
        
        for placement_sample in object_placement_poses:
            try:
                # Convert placement pose to transformation matrix
                place_rot = R.from_quat(placement_sample.pose_vector[3:], scalar_first=True).as_matrix()
                place_transform = np.eye(4)
                place_transform[:3, :3] = place_rot
                place_transform[:3, 3] = placement_sample.pose_vector[:3]
                
                # Calculate where the gripper should be for this object placement
                # T_gripper_placement = T_object_placement * T_object_to_gripper
                gripper_transform = place_transform @ object_to_gripper_transform
                
                # Convert back to pose vector
                gripper_rot_quat = R.from_matrix(gripper_transform[:3, :3]).as_quat(scalar_first=True)
                gripper_pose_vector = np.concatenate([gripper_transform[:3, 3], gripper_rot_quat])
                
                # Transform to robot frame
                gripper_pose_robot_frame = transform_target_to_robot_frame(gripper_pose_vector, self.robot_transform)
                gripper_placement_poses.append(gripper_pose_robot_frame)
                
            except Exception as e:
                print(f"Failed to calculate gripper placement pose: {e}")
                continue
                
        return gripper_placement_poses

    def stack_n_blocks(self, block_indices: List[int], object_dims: Tuple[float, float, float], 
                      camera_views: List[str] = ["external", "head"]) -> Dict:
        """
        Stack N blocks by sequentially grasping and placing them.
        
        Args:
            block_indices: List of block indices to stack in order (first is base, rest are stacked on top)
            object_dims: (length, width, height) dimensions of the blocks
            camera_views: List of camera views to record
            
        Returns:
            Dictionary containing execution results and recorded observations
        """
        if len(block_indices) < 2:
            print("Need at least 2 blocks to stack!")
            return {"success": False, "observations": {}}
            
        print(f"Stacking {len(block_indices)} blocks in sequence: {block_indices}")
        
        # Set up observation recording
        recorded_observations = {}
        for camera_view in camera_views:
            recorded_observations[camera_view] = []
            
        all_success = True
        
        # Stack blocks sequentially: blocks[1] on blocks[0], blocks[2] on blocks[1], etc.
        for stack_level in range(1, len(block_indices)):
            source_block_idx = block_indices[stack_level]  # Block to move
            target_block_idx = block_indices[stack_level - 1]  # Block to place on
            
            print(f"\n=== STACK LEVEL {stack_level}: Moving block {source_block_idx} onto block {target_block_idx} ===")
            
            try:
                # Get current poses (they may have changed from previous stacking)
                source_pose = get_object_pose(self.env, "blocks", source_block_idx)
                target_pose = get_object_pose(self.env, "blocks", target_block_idx)
                
                print(f"Source block {source_block_idx} pose: {source_pose}")
                print(f"Target block {target_block_idx} pose: {target_pose}")
                
                # Step 1: Sample grasp poses for the source block
                grasp_samples = self.sample_grasp_poses(source_pose, object_dims)
                
                # Step 2: Validate grasp poses
                robot_pose = self.env._robot.qpos_actuated[:-1]
                start_angles = robot_pose.tolist()
                start_state = JointState.from_position(
                    torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
                    joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                               "panda_joint5", "panda_joint6","panda_joint7"]
                )
                
                valid_grasps = self.validate_grasp_poses(grasp_samples, start_state)
                
                if not valid_grasps:
                    print(f"No valid grasp poses found for block {source_block_idx}! Stopping.")
                    all_success = False
                    break
                
                # Step 3: Sample placement poses (placing source block on target block)
                placement_spec = {
                    "relative_position": ["above"],
                    "upright_face": ["top"],
                    "allow_rotation_about_up": True,
                    "pose_tolerance": {
                        "position": 0.00,
                        "rotation": 5.0
                    }
                }
                object_placement_samples = self.sample_object_placement_poses(
                    target_pose, object_dims, object_dims, placement_spec
                )
                
                # Step 4: Propagate grasps to placements
                final_grasps = self.propagate_grasps_to_placements(
                    valid_grasps, source_pose, object_placement_samples
                )
                
                if not final_grasps:
                    print(f"No grasps with valid placements found for block {source_block_idx}! Stopping.")
                    all_success = False
                    break
                
                # Step 5: Execute the best grasp and placement for this block
                level_result = self.execute_single_stack_level(
                    final_grasps, source_block_idx, camera_views, recorded_observations
                )
                
                if not level_result["success"]:
                    print(f"Failed to stack block {source_block_idx} onto block {target_block_idx}! Stopping.")
                    all_success = False
                    break
                    
                print(f"Successfully stacked block {source_block_idx} onto block {target_block_idx}")
                
                # Let physics settle between stacking operations
                for _ in range(20):
                    obs, _, _, _, _ = self.env.step()
                    for camera_view in camera_views:
                        if f"rgb_{camera_view}" in obs:
                            recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                
                # Update the world configuration after stacking
                self.update_world_config_after_stacking()
                
            except Exception as e:
                print(f"Error during stack level {stack_level}: {e}")
                all_success = False
                break
        
        return {"success": all_success, "observations": recorded_observations}
    
    def execute_single_stack_level(self, final_grasps: List[GraspPoseSample], 
                                  block_idx: int, camera_views: List[str],
                                  recorded_observations: Dict) -> Dict:
        """
        Execute grasp and placement for a single block in the stack.
        
        Args:
            final_grasps: List of grasps with valid placements
            block_idx: Index of the block being moved
            camera_views: List of camera views to record
            recorded_observations: Dictionary to append observations to
            
        Returns:
            Dictionary containing execution results
        """
        if not final_grasps:
            print(f"No valid grasps with placements found for block {block_idx}!")
            return {"success": False}
            
        # Select the grasp with the most valid placements
        best_grasp = max(final_grasps, key=lambda g: len(g.valid_placements))
        print(f"Executing best grasp for block {block_idx} with {len(best_grasp.valid_placements)} valid placements")
        
        try:
            # Execute grasp motion
            robot_pose = self.env._robot.qpos_actuated[:-1]
            start_angles = robot_pose.tolist()
            start_state = JointState.from_position(
                torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                           "panda_joint5", "panda_joint6","panda_joint7"]
            )
            
            goal_pose = Pose.from_list(best_grasp.transformed_pose.tolist())

            # Derive the pre-grasp pose
            pre_grasp_pose = goal_pose
            pre_grasp_pose.position[0][2] += 0.1  # Move up by 10 cm

            grasp_result = self.motion_gen.plan_single(
                start_state, 
                pre_grasp_pose, 
                MotionGenPlanConfig(max_attempts=3)
            )
            
            if not grasp_result.success:
                print(f"Failed to plan grasp motion for block {block_idx}")
                return {"success": False}
                
            # Execute grasp trajectory
            grasp_traj = grasp_result.get_interpolated_plan()
            for i in range(grasp_traj.shape[0]):
                traj_step = grasp_traj[i].position.cpu().numpy()
                traj_step = np.concatenate([traj_step, np.zeros(1)])  # Open gripper
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Close gripper
            traj_step[-1] = 1  # Close the gripper
            for i in range(20):
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            # Attach object to robot (use block-specific collision name)
            # For StackNBlocks environment, blocks are typically named as "cube{i}/collision"
            # But let's use the simpler name first and handle errors gracefully
            try:
                if block_idx == 0:
                    block_collision_name = "cube/collision"  # First block might just be "cube"
                else:
                    block_collision_name = f"cube_{block_idx}/collision"
                    
                self.motion_gen.attach_objects_to_robot(
                    joint_state=grasp_traj[-1],
                    object_names=[block_collision_name],
                )
            except Exception as e:
                print(f"Warning: Failed to attach block {block_idx} with name '{block_collision_name}': {e}")
                # Try alternative naming schemes
                alternative_names = [
                    f"blocks{block_idx}/collision",
                    f"block{block_idx}/collision", 
                    f"cube_{block_idx}/collision",
                    "cube/collision"  # fallback
                ]
                attached = False
                for alt_name in alternative_names:
                    try:
                        self.motion_gen.attach_objects_to_robot(
                            joint_state=grasp_traj[-1],
                            object_names=[alt_name],
                        )
                        print(f"Successfully attached with name: {alt_name}")
                        attached = True
                        break
                    except:
                        continue
                        
                if not attached:
                    print(f"Warning: Could not attach block {block_idx} - continuing without attachment")
            
            # Execute placement motion using the first valid placement
            placement_pose = best_grasp.valid_placements[0]
            placement_goal_pose = Pose.from_list(placement_pose.tolist())
            
            start_pose = JointState.from_position(
                grasp_traj[-1].position.cuda().reshape(1, -1),
                joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                             "panda_joint5", "panda_joint6","panda_joint7"]
            )
            placement_result = self.motion_gen.plan_single(
                start_pose,
                placement_goal_pose,
                MotionGenPlanConfig(max_attempts=60)
            )
            
            if not placement_result.success:
                print(f"Failed to plan placement motion for block {block_idx}, the reason is: {placement_result.status}")
                # Still detach the object before returning
                self.motion_gen.detach_object_from_robot()
                return {"success": False}
                
            # Execute placement trajectory
            placement_traj = placement_result.get_interpolated_plan()
            for i in range(placement_traj.shape[0]):
                traj_step = placement_traj[i].position.cpu().numpy()
                traj_step = np.concatenate([traj_step, np.ones(1)])  # Keep gripper closed
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

            # Detach object from robot
            self.motion_gen.detach_object_from_robot()
                        
            # Open gripper to release object
            traj_step[-1] = 0  # Open the gripper
            for i in range(20):
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

            # Retract gripper to above the placed block
            retract_pose = placement_goal_pose
            retract_pose.position[0][2] += 0.1  # Move up by 10

            retract_result = self.motion_gen.plan_single(
                JointState.from_position(placement_traj[-1].position.cuda().reshape(1, -1),
                                         joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", 
                                                      "panda_joint5", "panda_joint6","panda_joint7"]),
                retract_pose,
                MotionGenPlanConfig(max_attempts=10)
            )

            if not retract_result.success:
                print(f"Failed to retract gripper after placement for block {block_idx}")
                # Still detach object before returning
                self.motion_gen.detach_object_from_robot()
                return {"success": False}
            
            # Execute retract trajectory
            retract_traj = retract_result.get_interpolated_plan()
            for i in range(retract_traj.shape[0]):
                traj_step = retract_traj[i].position.cpu().numpy()
                traj_step = np.concatenate([traj_step, np.zeros(1)])
                obs, reward, terminated, truncated, info = self.env.step(traj_step)
                for camera_view in camera_views:
                    if f"rgb_{camera_view}" in obs:
                        recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
                        
            


            print(f"Successfully executed grasp and placement for block {block_idx}!")
            return {"success": True}
            
        except Exception as e:
            print(f"Execution failed for block {block_idx}: {e}")
            # Try to detach object in case of error
            try:
                self.motion_gen.detach_object_from_robot()
            except:
                pass
            return {"success": False}

    def update_world_config_after_stacking(self):
        """Update the world configuration after stacking to reflect new block positions."""
        try:
            print("Updating world configuration after block movement...")
            
            # Generate new world config with current block positions
            mesh_dict = get_mesh_dict(self.env)
            world_config_dict = generate_world_config(
                mesh_dict, 
                self.env._mojo.physics, 
                mesh_test=False, 
                keywords_to_omit=["panda"]
            )

            print(f"Generated world configuration for blocks: {world_config_dict}")  # print metadata only to reduce spam
            
            # Transform to robot frame
            world_config_robot_frame = transform_objects_to_robot_frame(
                world_config_dict, 
                self.robot_transform
            )

            print(f"Updated world configuration for blocks: {world_config_robot_frame}")  # print metadata only to reduce spam

            # Convert to WorldConfig object
            world_config = WorldConfig.from_dict(world_config_robot_frame)
            
            # Update motion generator
            self.motion_gen.clear_world_cache()
            self.motion_gen.update_world(world_config)
            
            print("World configuration updated successfully")
            
        except Exception as e:
            print(f"Warning: Failed to update world configuration: {e}")
            print("Continuing without world update - this may affect collision detection")

def debug_collision_names(env):
    """Debug function to print all collision object names in the environment."""
    print("\n=== DEBUG: Available collision objects ===")
    mesh_dict = get_mesh_dict(env)
    world_config = generate_world_config(mesh_dict, env._mojo.physics, mesh_test=False, keywords_to_omit=["panda"])
    
    for obj_type, objects in world_config.items():
        if objects:  # Only print non-empty categories
            print(f"{obj_type.upper()}:")
            for name, details in objects.items():
                print(f"  - {name}")
    print("=" * 45)

def main(num_grasp_samples=15, num_placement_samples=8, num_blocks_to_stack=3, camera_views=["external", "head"]):
    """Main function to run the multi-grasp sampling and execution for N blocks.
    
    Args:
        num_grasp_samples: Number of grasp poses to sample for each block
        num_placement_samples: Number of placement poses to sample for each block
        num_blocks_to_stack: Total number of blocks to stack (including the base)
        camera_views: List of camera views to record
    """
    
    print(f"Starting multi-grasp sampler with {num_grasp_samples} grasp samples, {num_placement_samples} placement samples, and {num_blocks_to_stack} blocks to stack")
    
    # Set up the camera configs
    camera_configs = []
    for camera_view in camera_views:
        camera_configs.append(CameraConfig(name=camera_view, rgb=True, depth=False, resolution=(480, 480)))
    # Set up the environment
    # env = StackNBlocks(
    #     num_blocks=max(3, num_blocks_to_stack),  # Ensure we have at least as many blocks as we need
    #     blocks_step=0.15,
    #     action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[], block_until_reached=True),
    #     render_mode=None,
    #     robot_cls=SinglePanda,
    #     observation_config=ObservationConfig(
    #         cameras=camera_configs,
    #     ),
    # )

    env = PushBlockToTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[], block_until_reached=True),
        render_mode=None,
        robot_cls=SinglePanda,
        observation_config=ObservationConfig(
            cameras=camera_configs,
        ),
    )

    env.reset()

    # Step the environment a few times to let the physics settle
    for _ in range(100):
        env.step()

    # Set up cuRobo environment
    mesh_dict = get_mesh_dict(env)
    world_config = generate_world_config(mesh_dict, env._mojo.physics, mesh_test=False, keywords_to_omit=["panda"])

    print(f"Generated world configuration: {world_config}")  # print metadata only to reduce spam

    # Debug: Print available collision objects
    debug_collision_names(env)

    # Convert world config to robot frame
    panda_transform = get_robot_transform(env._mojo.physics, "panda nohand_left/")
    panda_world_config = transform_objects_to_robot_frame(world_config, panda_transform)

    print("Panda World Config: ", panda_world_config)
    
    # Initialize motion generation
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        panda_world_config,
        interpolation_dt=0.002,
        collision_activation_distance=0.00
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    # Initialize the multi-grasp sampler
    sampler = MultiGraspSampler(
        env=env,
        motion_gen=motion_gen,
        robot_transform=panda_transform,
        num_grasp_samples=num_grasp_samples,
        num_placement_samples=num_placement_samples
    )

    # Define which blocks to stack (first block is the base, rest are stacked on top)
    if num_blocks_to_stack < 2:
        print("Need at least 2 blocks to stack! Setting to 2.")
        num_blocks_to_stack = 2
    
    block_indices = list(range(num_blocks_to_stack))  # e.g., [0, 1, 2, 3] for 4 blocks
    object_dims = (0.04, 0.04, 0.04)  # Cube dimensions
    
    print(f"Will stack blocks in order: {block_indices}")
    print(f"Block dimensions: {object_dims}")
    
    # Execute the N-block stacking
    result = sampler.stack_n_blocks(block_indices, object_dims, camera_views)
    
    if result["success"]:
        print("Multi-grasp N-block stacking completed successfully!")
        
        # Save videos for each camera view
        fps = 30
        for camera_view, frames in result["observations"].items():
            if frames:
                frames_array = np.moveaxis(np.array(frames), 1, -1)
                video_clip = VideoClip(
                    make_frame=lambda t, frames=frames_array: frames[min(int(t * fps), len(frames) - 1)],
                    duration=len(frames_array) / fps
                )
                
                output_filename = f"multi_grasp_{num_blocks_to_stack}blocks_{camera_view}.mp4"
                video_clip.write_videofile(output_filename, fps=fps, codec="libx264")
                print(f"Saved video for camera view: {output_filename}")
    else:
        print("Multi-grasp N-block stacking execution failed!")
    
    # Print summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Number of blocks to stack: {num_blocks_to_stack}")
    print(f"Block indices: {block_indices}")
    print(f"Execution success: {result['success']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-grasp sampler for stacking N blocks")
    parser.add_argument("--num_grasp_samples", type=int, default=15, help="Number of grasp poses to sample")
    parser.add_argument("--num_placement_samples", type=int, default=8, help="Number of placement poses to sample")
    parser.add_argument("--num_blocks_to_stack", type=int, default=3, help="Number of blocks to stack")
    parser.add_argument("--camera_views", nargs="+", default=["external", "head"], help="Camera views to record")
    
    args = parser.parse_args()
    
    main(
        num_grasp_samples=args.num_grasp_samples,
        num_placement_samples=args.num_placement_samples,
        num_blocks_to_stack=args.num_blocks_to_stack,
        camera_views=args.camera_views
    )