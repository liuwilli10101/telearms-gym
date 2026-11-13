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
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda, SinglePanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
import numpy as np
from scipy.spatial.transform import Rotation as R
# Third Party
import torch
from roboeval.integrations.world_config_builder import build_world_config, transform_pose_world_to_base

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from tsr import TSR, TSRChain

import random

import numpy as np

from functools import reduce
import builtins
builtins.reduce = reduce

# def place_object(obj_transform, ee_transform, obj, pose_tsr_chain, manip_idx, 
#                      **kwargs):
#     """
#     Generates end-effector poses for placing an object.
#     This function assumes the object is grasped when called
    
#     @param robot The robot grasping the object
#     @param bowl The grasped object
#     @param pose_tsr_chain The tsr chain for sampling placement poses for the object
#     @param manip_idx The index of the manipulator to perform the grasp
#     """

#     # Can this work without importing anything?
#     # manip = robot.GetManipulators()[manip_idx]

#     # if not manip.IsGrabbing(obj):
#     #     raise Exception('manip %s is not grabbing %s' % (manip.GetName(), obj.GetName()))

#     ee_in_obj = np.dot(np.linalg.inv(obj_transform), 
#                           ee_transform)
#     Bw = np.zeros((6,2)) 

#     for tsr in pose_tsr_chain.TSRs:
#         if tsr.manipindex != manip_idx:
#             raise Exception('pose_tsr_chain defined for a different manipulator.')

#     grasp_tsr = TSR(Tw_e = ee_in_obj, Bw = Bw, manipindex = manip_idx)
#     all_tsrs = list(pose_tsr_chain.TSRs) + [grasp_tsr]
#     place_chain = TSRChain(sample_start = False, sample_goal = True, constrain = False,
#                            TSRs = all_tsrs)

#     return  [ place_chain ]

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

# ----------------------------
# Deep Merge for Spec Overrides
# ----------------------------

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

# ----------------------------
# Main TSR Generator
# ----------------------------

def generate_placement_tsrs(base_pose, cube_size, spec_overrides=None):
    """
    Generate TSRChain for placing a cube relative to another cube.
    
    Args:
        base_pose: (4,4) np.array - pose of the reference cube
        cube_size: float - edge length of the cube
        spec_overrides: dict - constraint overrides
    Returns:
        TSRChain
    """
    spec = deep_merge_spec(DEFAULT_PLACEMENT_SPEC, spec_overrides)

    offsets = get_relative_offsets(spec["relative_position"], cube_size)
    upright_rotations = get_upright_rotations(spec["upright_face"])
    inplane_rotations = get_inplane_rotations(spec["in_plane_rotation"], spec["allow_rotation_about_up"])

    pos_tol = spec["pose_tolerance"].get("position", 0.0)
    rot_tol = np.radians(spec["pose_tolerance"].get("rotation", 0.0))

    tsrs = []

    for offset in offsets:
        T0_w = np.eye(4)
        T0_w[:3, 3] = base_pose[:3, 3] + base_pose[:3, :3] @ offset

        for R_up in upright_rotations:
            for R_inplane in inplane_rotations:
                R_total = R_up @ R_inplane

                Tw_e = np.eye(4)
                Tw_e[:3, :3] = R_total

                Bw = np.zeros((6, 2))
                Bw[:3, 0] = -pos_tol
                Bw[:3, 1] = pos_tol
                Bw[3:, 0] = -rot_tol
                Bw[3:, 1] = rot_tol

                tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
                tsrs.append(TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr))
    
    return tsrs
# ----------------------------
# Helper Functions
# ----------------------------

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

def box_on_box_place_tsr(T_support_box_world,
                         support_dims,  # (length, width, height)
                         new_box_dims,  # (length, width, height)
                         lateral_tolerance=0.01,
                         face_options=None,
                         lateral_offset=0.15):
    """
    Generates TSRs for placing a box on another box, aligned with its top (or other faces).

    @param T_support_box_world: 4x4 transform of the lower box in world frame
    @param support_dims: (length, width, height) of the support box
    @param new_box_dims: (length, width, height) of the box being placed
    @param lateral_tolerance: allowed XY shift when placing
    @param face_options: list of face names to place onto (e.g., ['top', 'left'])

    @return: list of TSRChains
    """
    support_length, support_width, support_height = support_dims
    new_length, new_width, new_height = new_box_dims

    chain_list = []

    def make_chain(Tw_e, Bw):
        tsr = TSR(T0_w=T_support_box_world, Tw_e=Tw_e, Bw=Bw)
        return TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr)

    face_orientations = {
        'top': np.array([[1.,  0.,  0., 0.],
                         [0.,  -1.,  0., 0.],
                         [0.,  0.,  -1., 0.5 * support_height + 0.5 * new_height + lateral_offset],
                         [0.,  0.,  0., 1.]]),

        'bottom': np.array([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., -1., -0.5 * support_height - 0.5 * new_height - lateral_offset],
                            [0., 0., 0., 1.]]),

        'front': np.array([[1., 0., 0., 0.5 * support_length + 0.5 * new_length],
                           [0., 0., 1., 0.],
                           [0., -1., 0., 0.],
                           [0., 0., 0., 1.]]),

        'back': np.array([[ -1., 0., 0., -0.5 * support_length - 0.5 * new_length],
                          [  0., 0., 1., 0.],
                          [  0., 1., 0., 0.],
                          [  0., 0., 0., 1.]]),

        'left': np.array([[0., -1., 0., 0.],
                          [1.,  0., 0., 0.5 * support_width + 0.5 * new_width],
                          [0.,  0., 1., 0.],
                          [0.,  0., 0., 1.]]),

        'right': np.array([[0., 1., 0., 0.],
                           [-1., 0., 0., -0.5 * support_width - 0.5 * new_width],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    }

    if face_options is None:
        face_options = ['top']

    for face in face_options:
        Tw_e = face_orientations[face]
        Bw = np.zeros((6, 2))
        Bw[0:2, :] = [-lateral_tolerance, lateral_tolerance]
        chain_list.append(make_chain(Tw_e, Bw))

    return chain_list

def box_grasp(T0_w, length, width, height,
              lateral_offset=0.0,
              lateral_tolerance=0.02,
              rotation_tolerance=0.7,
              **kwargs):
    """
    Generate a list of TSRChain objects for grasping a box.

    Assumptions:
      1. End-effector z-axis is out of the palm (toward the object during grasp)
      2. Box frame is centered at the bottom of the box
    """
    import numpy as np

    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError('Box dimensions must be > 0')

    if lateral_tolerance < 0.0:
        raise ValueError('lateral_tolerance must be >= 0.0')

    chain_list = []

    def make_chain(Tw_e, Bw):
        tsr = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        return TSRChain(sample_start=False, sample_goal=True, constrain=False, TSR=tsr)

    # Top face (approach from +z, palm toward -z)
    # Top face (approach from +z, palm toward -z), using x-180 rotation
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



def xmat_to_quat(data):
    """
    Convert a body's xmat (flattened 3x3 rotation matrix) from mjData to a MuJoCo-style quaternion [w, x, y, z].

    Args:
        data: mjData object from MuJoCo simulation.

    Returns:
        A numpy array representing the quaternion [w, x, y, z].
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

def generate_world_config(mesh_dict, physics, geom_group = 3, mesh_test = False, keywords_to_omit = []):
    world_config = {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}}
    
    collision_geom_idx = np.where(physics.model.geom_group == geom_group)[0]
    
    ### Looping over Robot Meshes for world_config ####
    for i in collision_geom_idx:
        
        # Access the geom model and data
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


        ## Check for different geom types ##
        if geom.type[0] ==  3: #Capsule       
            world_config["capsule"][geom.name] = {  "name": geom.name, 
                                                    "radius": geom.size[0],
                                                    "height": geom.size[1],
                                                    "pose": np.concatenate([geom_data.xpos, 
                                                                            quaternion])
                                                }

        if geom.type[0] == 5: #Cylinder
            world_config["cylinder"][geom.name] = { "name":geom.name, 
                                                    "radius": geom.size[0],
                                                    "height": geom.size[1],
                                                    "pose" : np.concatenate([geom_data.xpos, 
                                                                            quaternion])
                                                    }


        
        if geom.type[0] ==  6: #Cuboid
            # print(i)
            world_config["cuboid"][geom.name] = {
                                                "dims": list(geom.size),
                                                "pose": np.concatenate([geom_data.xpos, 
                                                                        quaternion])
                                                }

        if geom.type[0] ==  7: #Mesh

            #Find the key for the mesh file
            mesh_file = mesh_dict.get(geom.name, None)
            world_config["mesh"][geom.name] = {
                                               "file_path": mesh_file,
                                               "pose": np.concatenate([geom_data.xpos, quaternion])
                                            }

    ### Testing Meshes Work ###
    if mesh_test:
        for key, value in world_config["mesh"].items():
            print(value["name"], ":", value["file_path"])
    
    return world_config

##Panda rotated about z-axis from Mujoco World by 180 degrees
def get_robot_transform(physics, robot_name):
    t = physics.named.data.xpos[robot_name]

    R_flat = physics.named.data.xmat[robot_name]  # shape (9,)
    R = np.reshape(R_flat, (3, 3))  # shape (3, 3)

    return {"R":R,"t":-R@t}

def transform_objects_to_robot_frame(objects:dict, robot_transform):

    R = robot_transform["R"] 
    t = robot_transform["t"]
    result = objects.copy()
    for key, v in objects["cuboid"].items():
        new_pose = R@v["pose"][:3] + t
        result["cuboid"][key]["pose"][:3] = new_pose

    for key, v in objects["mesh"].items():
        new_pose = R@v["pose"][:3] + t
        result["mesh"][key]["pose"][:3] = new_pose

    print(f"Old Objects: {objects} to New Objects: {result}")

    return result

def transform_target_to_robot_frame(target, robot_transform):
    """
    Transform a target pose to the robot frame. First three elements are position (x, y, z) and the last four are quaternion (qw, qx, qy, qz).
    """
    rotation = robot_transform["R"]
    translation = robot_transform["t"]

    target = np.array(target)  # Ensure target is a numpy array
    if target.shape != (7,):
        raise ValueError("Target must be a 7-element array: [x, y, z, qw, qx, qy, qz]")

    # Position transform
    new_pos = rotation @ target[:3] + translation

    # Orientation transform: q_robot = q_Rwr * q_world
    q_world = R.from_quat(target[3:], scalar_first=True)
    q_r_wr = R.from_matrix(rotation)
    q_robot = (q_r_wr * q_world).as_quat(scalar_first=True)

    return np.concatenate([new_pos, q_robot])  # [x, y, z, qw, qx, qy, qz]

def get_object_pose(env, object_name, idx = None):
    """ Function to get the pose of an object in the mujoco environment.
    Args:
        object_name (str): The name of the object whose pose is to be retrieved.
    Returns:
        Pose: The pose of the object in the mujoco environment.
    Raises:
        ValueError: If the object is not found in the environment.
    """

    # Get the pose of the object corresponding to the object name from mujoco env
    # Get the variable from the mujoco environment, the variable is the object name
    prop = getattr(env, object_name)

    if idx is not None:
        prop = prop[idx] if isinstance(prop, list) else prop
    if prop is None:
        raise ValueError(f"Object {object_name} not found in the environment.")

    prop_pose = prop.get_pose()

    return prop_pose


def sample_gripper_action(object, object_pose, gripper_offset=0.1):
    """
    Sample a gripper action for the given object and its pose.

    Args:
        object: The object to interact with.
        object_pose: The pose of the object.
        gripper_offset: The offset to apply to the gripper pose.

    Returns:
        A dictionary representing the gripper action.
    """
    gripper_pose = object_pose.copy()
    gripper_pose[2] += gripper_offset  # Move the gripper above the object

    return {
        "object": object,
        "gripper_pose": gripper_pose
    }


def main(camera_views = ["external", "head"]):

    """Main function to run the evaluation."""

    # Set up the camera configs
    camera_configs = []
    for camera_view in camera_views:
        camera_config = CameraConfig(
            name=f"{camera_view}",
            rgb=True,
            depth=False,
            resolution=(480, 480),
            # pos=np.array([0.0, 10.0, 10.0]).tolist(),
        )

        camera_configs.append(camera_config)
    
    # Set up the environment
    env = StackNBlocks(
        num_blocks=5,
        blocks_step=0.1,  # Step size for the number of blocks  
        action_mode=JointPositionActionMode(floating_base=True, absolute=True, floating_dofs=[], block_until_reached=True),
        render_mode=None,
        robot_cls=SinglePanda,
        observation_config=ObservationConfig(
            cameras=camera_configs,
        ),
    )

    env.reset()

    # Step the environment a few times to let the physics settle
    for _ in range(10):
        env.step()

    # Set up cuRobo environment
    world_config_dict, robot_tf = build_world_config(env, robot_name="panda nohand_left/", geom_group=3, target_frame="robot_base", keywords_to_omit=["panda"])
    panda_world_config = world_config_dict  # already in robot base frame

    print("Panda World Config: ", panda_world_config["_metadata"])  # print metadata only to reduce spam
    # Initialize motion generation
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        panda_world_config,
        interpolation_dt=0.002,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    # Given the task, get the list of actions to perform
    pose = get_object_pose(env, "blocks", 0) # x, y, z, qw, qx, qy, qz

    rotation = R.from_quat(pose[3:], scalar_first=True).as_matrix()
    box_transform = np.eye(4)
    box_transform[:3, :3] = rotation
    box_transform[:3, 3] = pose[:3]

    box_tsr_chain = box_grasp(
    T0_w=box_transform,
    length=0.04,
    width=0.04,
    height=0.04,
    lateral_offset=0.05,
    lateral_tolerance=0.01)

    # pick_pose = box_tsr_chain[random.randint(0, len(box_tsr_chain)-1)].sample()
    pick_pose = box_tsr_chain[0].sample()

    # Convert the homogeneous transformation matrix to a x,y,z,w,qx,qy,qz representation
    pick_pose_quat = R.from_matrix(pick_pose[:3, :3]).as_quat(scalar_first=True)  # [qw, qx, qy, qz]
    pick_pose_vec = pick_pose[:3, 3]
    offseted_pose = np.concatenate([pick_pose_vec, pick_pose_quat])

    transformed_pose = transform_pose_world_to_base(offseted_pose.tolist(), robot_tf)

    # Get the current robot pose
    robot_pose = env._robot.qpos_actuated[:-1]

    # Define the goal pose for the robot to reach
    # goal_pose = Pose.from_list([ 0.0286, -0.6152,  0.5602, 0.4477, -0.1715, -0.6593, -0.5792])

    goal_pose = Pose.from_list(transformed_pose.tolist())

    print("Goal Pose: ", goal_pose)

    # Define the start state of the robot
    start_angles = robot_pose.tolist()
    # start_angles = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    torch_start_angles = torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1)
    start_state = JointState.from_position(
        torch_start_angles.cuda(),
        joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
        "panda_joint6","panda_joint7"]
    )

    result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=60))
    traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
    print("Trajectory Generated: ", result.success)

    # # Guard: abort if pick planning failed
    # if (traj is None) or (not result.success):
    #     print("Pick planning failed; exiting early.")
    #     return

    # Set up instances of the recorded observations
    recorded_observations = {}
    for camera_view in camera_views:
        recorded_observations[camera_view] = []
    for i in range(traj.shape[0]):
        traj_step = traj[i].position.cpu().numpy() 
        # Add gripper action of 0
        traj_step = np.concatenate([traj_step, np.zeros(1)])  # Assuming 2 gripper joints
        obs, reward, terminated, truncated, info = env.step(traj_step)
        for camera_view in camera_views:
            # Record the RGB observation for each camera view
            if f"rgb_{camera_view}" in obs:
                if camera_view not in recorded_observations:
                    recorded_observations[camera_view] = []
                recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())
        # if i % 1000 == 0:
        #     env.reset()

    # Close the gripper
    traj_step[-1] = 1 # Close the gripper

    for i in range(20):
        # Step the environment with the final action
        # This is to ensure the gripper closes properly and the object is held
        obs, reward, terminated, truncated, info = env.step(traj_step)
        for camera_view in camera_views:
            recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

    # Disable the first cube from collision checking
    try:
        motion_gen.attach_objects_to_robot(
            joint_state=traj[-1],
            object_names=["cube/collision"],
        )
    except Exception as e:
        print(f"attach_objects_to_robot failed: {e}")

    # # Get the placement pose of the object
    placement_pose = get_object_pose(env, "blocks", 1)  # Assuming the second block is the placement target
    placement_rotation = R.from_quat(placement_pose[3:], scalar_first=True).as_matrix()
    placement_transform = np.eye(4)
    placement_transform[:3, :3] = placement_rotation
    placement_transform[:3, 3] = placement_pose[:3]

    

    # Generate placement TSRs
    # placement_tsr_chain = box_on_box_place_tsr(
    #     T_support_box_world=placement_transform,
    #     support_dims=(0.04, 0.04, 0.04),  # Dimensions of the support box
    #     new_box_dims=(0.04, 0.04, 0.04),  # Dimensions of the box being placed
    #     lateral_tolerance=0.01,
    #     face_options=['top']  # Specify which faces to use for placement
    # )

    base_pose = placement_transform
    # np.eye(4)
    cube_size = 0.04

    spec = {
        "relative_position": ["above"],
        "upright_face": ["top"],
        "allow_rotation_about_up": True,
        "pose_tolerance": {
            "position": 0.01,
            "rotation": 5.0
        }
    }

    placement_tsr_chain = generate_placement_tsrs(base_pose, cube_size, spec)

    # Define the start state for placement (from end of pick trajectory)
    start_angles = traj[-1].position.cpu().numpy().tolist()
    start_state = JointState.from_position(
        torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
        joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
        "panda_joint6","panda_joint7"]
    )

    # Try multiple TSR samples and increased attempts for robust planning
    plan_config = MotionGenPlanConfig(max_attempts=60)
    placement_traj = None

    import random as _random
    tsr_candidates = placement_tsr_chain.copy()
    _random.shuffle(tsr_candidates)
    tsr_candidates = tsr_candidates[:min(12, len(tsr_candidates))]

    for tsr in tsr_candidates:
        for _ in range(3):  # resample a few times per TSR
            placement_pose = tsr.sample()
            placement_pose_quat = R.from_matrix(placement_pose[:3, :3]).as_quat(scalar_first=True)
            placement_pose_vec = placement_pose[:3, 3]
            placement_offseted_pose = np.concatenate([placement_pose_vec, placement_pose_quat])

            transformed_placement_pose = transform_pose_world_to_base(placement_offseted_pose.tolist(), robot_tf)
            placement_goal_pose = Pose.from_list(transformed_placement_pose.tolist())

            placement_result = motion_gen.plan_single(
                start_state=start_state,
                goal_pose=placement_goal_pose,
                plan_config=plan_config,
            )
            if placement_result.success:
                placement_traj = placement_result.get_interpolated_plan()
                if placement_traj is not None:
                    print("Placement Trajectory Generated: ", True)
                    break
        if placement_traj is not None:
            break

    if placement_traj is None:
        print("Placement planning failed after retries; skipping placement execution.")
    else:
        # Execute the placement trajectory
        for i in range(placement_traj.shape[0]):
            placement_traj_step = placement_traj[i].position.cpu().numpy()
            # Add gripper action of 1 (hold closed during approach)
            placement_traj_step = np.concatenate([placement_traj_step, np.ones(1)])
            obs, reward, terminated, truncated, info = env.step(placement_traj_step)
            for camera_view in camera_views:
                recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

        # Open the gripper to release the object
        placement_traj_step[-1] = 0
        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(placement_traj_step)
            for camera_view in camera_views:
                recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())


    # # Pick up the third block (index 2)
    # pick_pose = get_object_pose(env, "blocks", 2)  # x, y, z, qw, qx, qy, qz
    # rotation = R.from_quat(pick_pose[3:], scalar_first=True).as_matrix()
    # block_transform = np.eye(4)
    # block_transform[:3, :3] = rotation
    # block_transform[:3, 3] = pick_pose[:3] 

    # block_tsr_chain = box_grasp(
    #     T0_w=block_transform,
    #     length=0.04,
    #     width=0.04,
    #     height=0.04,
    #     lateral_offset=0.3,
    #     lateral_tolerance=0.01
    # )

    # # Sample a pick pose
    # pick_pose = block_tsr_chain[0].sample()

    # # Convert the homogeneous transformation matrix to a x,y,z,w,qx,qy,qz representation
    # pick_pose_quat = R.from_matrix(pick_pose[:3, :3]).as_quat(scalar_first=True)  # [qw, qx, qy, qz]
    # pick_pose_vec = pick_pose[:3, 3]
    # offseted_pose = np.concatenate([pick_pose_vec, pick_pose_quat])
    # # offseted_pose = pick_pose.copy()
    # # offseted_pose[:3] += np.array([0, 0, 0.1])  # Adjust the z-coordinate for the gripper offset

    # transformed_pose = transform_target_to_robot_frame(offseted_pose, panda_transform)
    # # Define the goal pose for the robot to reach
    # goal_pose = Pose.from_list(transformed_pose.tolist())   
    # print("Block Pick Goal Pose: ", goal_pose)
    # # Define the start state of the robot
    # start_angles = placement_traj[-1].position.cpu().numpy().tolist()
    # start_state = JointState.from_position(
    #     torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
    #     joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
    #     "panda_joint6","panda_joint7"]
    # )

    # # Clear the collision cache
    # world_config = generate_world_config(mesh_dict, env._mojo.physics, mesh_test = False, keywords_to_omit=["panda"])
    # world_config = WorldConfig.from_dict(world_config)
    # motion_gen.clear_world_cache()
    # motion_gen.update_world(world_config)

    # # Plan the pick motion
    # pick_result = motion_gen.plan_single(
    #     start_state=start_state,
    #     goal_pose=goal_pose,
    #     plan_config=MotionGenPlanConfig(max_attempts=60)
    # )
    # pick_traj = pick_result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
    # print("Pick Trajectory Generated: ", pick_result.success)
    # # Execute the pick trajectory
    # for i in range(pick_traj.shape[0]):
    #     pick_traj_step = pick_traj[i].position.cpu().numpy()
    #     # Add gripper action of 0
    #     pick_traj_step = np.concatenate([pick_traj_step, np.zeros(1)])  # Assuming 2 gripper joints
    #     obs, reward, terminated, truncated, info = env.step(pick_traj_step)
    #     for camera_view in camera_views:
    #         recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

    # # Close the gripper to grasp the block
    # pick_traj_step[-1] = 1  # Close the gripper
    # for i in range(20):
    #     # Step the environment with the final action
    #     # This is to ensure the gripper closes properly and the block is held
    #     obs, reward, terminated, truncated, info = env.step(pick_traj_step)
    #     for camera_view in camera_views:
    #         recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

    # # Disable the second cube from collision checking
    # motion_gen.attach_objects_to_robot(
    #     joint_state=pick_traj[-1],
    #     object_names=["cube_2/collision"],
    # )
    # # Get the placement pose of the object
    # placement_pose = get_object_pose(env, "blocks", 0)  # Assuming the first block is the placement target
    # placement_rotation = R.from_quat(placement_pose[3:], scalar_first=True).as_matrix()
    # placement_transform = np.eye(4)
    # placement_transform[:3, :3] = placement_rotation
    # placement_transform[:3, 3] = placement_pose[:3]
    # # Generate placement TSRs
    # placement_tsr_chain = box_on_box_place_tsr(
    #     T_support_box_world=placement_transform,
    #     support_dims=(0.04, 0.04, 0.04),  # Dimensions of the support box
    #     new_box_dims=(0.04, 0.04, 0.04),  # Dimensions of the box being placed
    #     lateral_tolerance=0.01,
    #     face_options=['top']  # Specify which faces to use for placement
    # )
    # # Sample a placement pose
    # placement_pose = placement_tsr_chain[0].sample()

    # placement_pose_quat = R.from_matrix(placement_pose[:3, :3]).as_quat(scalar_first=True)  # [qw, qx, qy, qz]
    # placement_pose_vec = placement_pose[:3, 3]
    # placement_offseted_pose = np.concatenate([placement_pose_vec, placement_pose_quat])
    # # placement_offseted_pose = placement_pose.copy()
    # # placement_offseted_pose[:3] += np.array([0, 0, 0.1])  # Adjust the z-coordinate for the gripper offset
    # transformed_placement_pose = transform_target_to_robot_frame(placement_offseted_pose, panda_transform)
    # # Define the goal pose for the robot to reach for placement
    # placement_goal_pose = Pose.from_list(transformed_placement_pose.tolist())
    # print("Placement Goal Pose for Block: ", placement_goal_pose)
    # start_angles = pick_traj[-1].position.cpu().numpy().tolist()
    # start_state = JointState.from_position(
    #     torch.tensor(start_angles, dtype=torch.float32).cuda().reshape(1, -1),
    #     joint_names=["panda_joint1","panda_joint2","panda_joint3","panda_joint4", "panda_joint5",
    #     "panda_joint6","panda_joint7"]
    # )

    # # Plan the placement motion for the block
    # placement_result = motion_gen.plan_single(
    #     start_state=start_state,
    #     goal_pose=placement_goal_pose,
    #     plan_config=MotionGenPlanConfig(max_attempts=1)
    # )
    # placement_traj = placement_result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
    # print("Placement Trajectory for Block Generated: ", placement_result.success)

    # # Execute the placement trajectory for the block
    # for i in range(placement_traj.shape[0]):
    #     placement_traj_step = placement_traj[i].position.cpu().numpy()
    #     # Add gripper action of 0
    #     placement_traj_step = np.concatenate([placement_traj_step, np.ones(1)])  # Assuming 2 gripper joints
    #     obs, reward, terminated, truncated, info = env.step(placement_traj_step)
    #     for camera_view in camera_views:
    #         recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

    # # Open the gripper to release the block
    # placement_traj_step[-1] = 0  # Open the gripper
    # for i in range(20):
    #     # Step the environment with the final action
    #     # This is to ensure the gripper opens properly and the block is released
    #     obs, reward, terminated, truncated, info = env.step(placement_traj_step)
    #     for camera_view in camera_views:
    #         recorded_observations[camera_view].append(obs[f"rgb_{camera_view}"].copy())

    # # Add back the collision checking for the second cube
    # motion_gen.detach_objects_from_robot(
    #     joint_state=placement_traj[-1],
    #     object_names=["cube_2/collision"],
    # )

    # Save the recorded observations as a video

    # frames = np.moveaxis(np.array(recorded_observations), 1, -1)
    # fps = 30
    # video_clip = VideoClip(
    #     make_frame=lambda t: frames[int(t * fps)], duration=int(len(frames) / fps)
    # )
    # # video_clip.preview()
    # video_clip.write_videofile("reach_target_pixels.mp4", fps=fps, codec="libx264")

    # Save each camera view as a separate video
    fps = 30
    for camera_view, frames in recorded_observations.items():
        # Convert frames to numpy array
        frames_array = np.moveaxis(np.array(frames), 1, -1)

        # Create video clip for this camera view
        video_clip = VideoClip(
            make_frame=lambda t, frames=frames_array: frames[min(int(t * fps), len(frames) - 1)],
            duration=len(frames_array) / fps
        )
        
        # Save the video with the camera view name
        output_filename = f"{camera_view}.mp4"
        video_clip.write_videofile(output_filename, fps=fps, codec="libx264")
        print(f"Saved video for camera view: {output_filename}")

if __name__ == "__main__":
    main()