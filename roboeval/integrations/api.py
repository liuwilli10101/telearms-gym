# A thin API layer that talks to cuRobo using your existing helpers in data_collection/curobo_plan.py
from typing import Dict, List, Tuple, Optional
import numpy as np

# Reuse helpers and MotionGen pipeline from roboeval example
from roboeval.data_collection.curobo_plan import (
    box_grasp,
    get_object_pose,
    get_mesh_dict,
    generate_world_config,
    get_robot_transform,
    transform_objects_to_robot_frame,
    transform_target_to_robot_frame,
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    Pose,
    JointState,
    R,
)


class CuRoboContext:
    def __init__(self, env):
        # Assuming env is a StackTwoBlocks instance, already reset
        self.env = env
        # Build world config from the running MuJoCo env
        mesh_dict = get_mesh_dict(self.env)
        world_config = generate_world_config(mesh_dict, self.env._mojo.physics, mesh_test=False, keywords_to_omit=["panda"])
        # Transform world to robot frame
        self.robot_transform = get_robot_transform(self.env._mojo.physics, "panda nohand_left/")
        self.panda_world_config = transform_objects_to_robot_frame(world_config, self.robot_transform)
        # Configure and warm up cuRobo
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            "franka.yml", self.panda_world_config, interpolation_dt=0.002
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        # Runtime memory
        self.last_ee_quat = None
        self.attached_names = []
        # Recording buffer
        self.frames: List[np.ndarray] = []
        # Persisted gripper control (0=open, 1=closed)
        self.gripper_ctrl: float = 0.0

    def current_joints(self):
        q = self.env._robot.qpos_actuated[:-1]
        return q

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            if isinstance(obs, dict) and "rgb_external" in obs:
                self.frames.append(obs["rgb_external"].copy())
        except Exception:
            pass
        return obs, reward, terminated, truncated, info


def Move(ctx: CuRoboContext, target_pose_wxyz: List[float], time_dilation: Optional[float] = None) -> bool:
    # Move the EE to a Cartesian pose in robot frame
    goal_pose = Pose.from_list(target_pose_wxyz)
    start = ctx.current_joints().tolist()
    torch_start = np.array(start, dtype=np.float32).reshape(1, -1)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_start = torch.tensor(torch_start, dtype=torch.float32, device=device)
    start_state = JointState.from_position(
        torch_start,
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
    )
    plan_cfg = MotionGenPlanConfig(max_attempts=60)
    if time_dilation is not None:
        plan_cfg.time_dilation_factor = time_dilation
    result = ctx.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    if not result.success:
        print("[Move] MotionGen planning failed for target:", target_pose_wxyz)
        return False
    traj = result.get_interpolated_plan()
    for i in range(traj.shape[0]):
        q = traj[i].position.cpu().numpy()
        # Append persisted gripper command (mirror curobo_plan behavior)
        q = np.concatenate([q, np.array([ctx.gripper_ctrl], dtype=q.dtype)])
        ctx.step(q)
    return True


def Pick(ctx: CuRoboContext, obj_name: str) -> bool:
    obj_idx= int(obj_name.split("_")[-1]) - 1 if "_" in obj_name else None
    obj_name = obj_name.split("_")[0] if "_" in obj_name else obj_name
    # Get object pose in world, create a TSR to approach/grasp, plan with cuRobo
    print(f"[Pick] Picking {obj_name}_{obj_idx + 1} via blocks[{obj_idx}]")
    pose = get_object_pose(ctx.env, obj_name, obj_idx) 
    rot = R.from_quat(pose[3:], scalar_first=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pose[:3]
    tsrs = box_grasp(
        T0_w=T, length=0.04, width=0.04, height=0.03, lateral_offset=0.07, lateral_tolerance=0.01
    )
    pick_T = tsrs[0].sample()
    pick_quat = R.from_matrix(pick_T[:3, :3]).as_quat(scalar_first=True)
    pick_vec = pick_T[:3, 3]
    target_w = np.concatenate([pick_vec, pick_quat])
    # Transform to robot frame and move to grasp pose
    target = transform_target_to_robot_frame(target_w, ctx.robot_transform)
    if not Move(ctx, target.tolist()):
        return False
    # Close gripper and dwell for stability
    ctx.gripper_ctrl = 1.0
    q = ctx.env._robot.qpos_actuated.copy()
    q = np.concatenate([q[:-1], np.array([ctx.gripper_ctrl])])
    for _ in range(50):  # dwell frames like curobo_plan does (ensures firm grasp)
        ctx.step(q)
    # Remember grasp orientation for downstream place
    ctx.last_ee_quat = pick_quat.copy()
    # Lift up to a safe height before moving elsewhere
    lift_w = target_w.copy()
    lift_w[2] += 0.2
    lift_r = transform_target_to_robot_frame(lift_w, ctx.robot_transform)
    if not Move(ctx, lift_r.tolist()):
        return False
    # Attach the picked object to the robot to avoid treating it as an external obstacle
    try:
        # Identify nearest cuboid entry in the collision world to the grasped object's pose
        grasp_pos_r = transform_target_to_robot_frame(np.concatenate([pose[:3], R.from_quat(pose[3:], scalar_first=True).as_quat(scalar_first=True)]), ctx.robot_transform)[:3]
        nearest_name = None
        nearest_dist = float("inf")
        for name, entry in ctx.panda_world_config.get("cuboid", {}).items():
            p = entry.get("pose", None)
            if p is None:
                continue
            d = np.linalg.norm(np.array(p[:3]) - grasp_pos_r)
            if d < nearest_dist:
                nearest_dist = d
                nearest_name = name
        if nearest_name is not None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            q_now = np.array(ctx.current_joints(), dtype=np.float32).reshape(1, -1)
            q_now = torch.tensor(q_now, dtype=torch.float32, device=device)
            js = JointState.from_position(q_now, joint_names=[
                "panda_joint1","panda_joint2","panda_joint3","panda_joint4","panda_joint5","panda_joint6","panda_joint7"
            ])
            ctx.motion_gen.attach_objects_to_robot(joint_state=js, object_names=[nearest_name])
            ctx.attached_names = [nearest_name]
    except Exception as e:
        print("[Pick] attach_objects_to_robot skipped:", e)
    return True


def Place(ctx: CuRoboContext, src_obj: str, tgt_obj: str, pre_h: float = 0.28, final_h: float = 0.03) -> bool:
    # Compute EE pose using TSR-style top-down orientation relative to target block
    obj_idx= int(tgt_obj.split("_")[-1]) - 1 if "_" in tgt_obj else None
    tgt_obj = tgt_obj.split("_")[0] if "_" in tgt_obj else tgt_obj
    # Get object pose in world, create a TSR to approach/grasp, plan with cuRobo
    print(f"[Place] Placing {src_obj} on {tgt_obj} with pre_h={pre_h}, final_h={final_h} via blocks[{obj_idx}]")
    pose = get_object_pose(ctx.env, tgt_obj, obj_idx) 
    rot_tgt = R.from_quat(pose[3:], scalar_first=True).as_matrix()
    T0_w = np.eye(4)
    T0_w[:3, :3] = rot_tgt
    T0_w[:3, 3] = pose[:3]

    def ee_pose_world(z_offset: float) -> np.ndarray:
        Tw_e_top = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, z_offset],
            [0.0, 0.0, 0.0, 1.0],
        ])
        T_w_e = T0_w @ Tw_e_top
        return T_w_e

    # Pre-place: high above the target (keep gripper closed until release)
    T_pre = ee_pose_world(0.5 * 0.04 + pre_h)
    pre_pos = T_pre[:3, 3]
    pre_quat = R.from_matrix(T_pre[:3, :3]).as_quat(scalar_first=True)
    pre_w = np.concatenate([pre_pos, pre_quat])
    pre_r = transform_target_to_robot_frame(pre_w, ctx.robot_transform)
    if not Move(ctx, pre_r.tolist()):
        return False

    # Staged descent to near-contact (smaller steps for smoother approach)
    target_offsets = []
    h = pre_h
    step = 0.02
    while h - step > final_h:
        h -= step
        # switch to finer steps in last 3 cm
        if h <= final_h + 0.03:
            step = 0.01
        target_offsets.append(h)
    target_offsets.append(final_h)

    lowest_success_h = None
    for h_try in target_offsets:
        T_step = ee_pose_world(0.5 * 0.04 + h_try)
        step_pos = T_step[:3, 3]
        step_quat = R.from_matrix(T_step[:3, :3]).as_quat(scalar_first=True)
        step_w = np.concatenate([step_pos, step_quat])
        step_r = transform_target_to_robot_frame(step_w, ctx.robot_transform)
        # slow down only very near contact
        td = 0.7 if (h_try <= final_h + 0.015) else None
        if Move(ctx, step_r.tolist(), time_dilation=td):
            lowest_success_h = h_try
        else:
            continue

    if lowest_success_h is None:
        return False

    # Closed settle before release to bleed residual motion
    try:
        q_hold = ctx.env._robot.qpos_actuated.copy()
        q_hold = np.concatenate([q_hold[:-1], np.array([1.0])])  # ensure closed during settle
        for _ in range(20):
            ctx.step(q_hold)
    except Exception:
        pass

    # # Smoothly open the gripper to avoid lateral impulse
    # try:
    #     q_base = ctx.env._robot.qpos_actuated.copy()
    #     for val in np.linspace(1.0, 0.0, num=15):
    #         q_base[-1] = val
    #         ctx.step(q_base.copy())
    # except Exception:
    #     pass


    # Ensure gripper stays open and let block settle
    ctx.gripper_ctrl = 0.0
    q = ctx.env._robot.qpos_actuated.copy()
    q = np.concatenate([q[:-1], np.array([ctx.gripper_ctrl])])
    for _ in range(50):  # dwell frames like curobo_plan does (ensures firm grasp)
        ctx.step(q)

    # Detach in planner BEFORE retreat so start state is not colliding
    try:
        ctx.motion_gen.detach_object_from_robot()
        ctx.attached_names = []
    except Exception as e:
        print("[Place] detach_object_from_robot skipped:", e)

    # Ensure gripper stays open and let block settle
    ctx.gripper_ctrl = 0.0
    q = ctx.env._robot.qpos_actuated.copy()
    q = np.concatenate([q[:-1], np.array([ctx.gripper_ctrl])])
    for _ in range(50):  # dwell frames like curobo_plan does (ensures firm grasp)
        ctx.step(q)

    # Retreat upwards to clear the stack (slightly less aggressive, with mild slowdown)
    T_retreat = ee_pose_world(0.5 * 0.04 + max(pre_h, lowest_success_h + 0.18))
    ret_pos = T_retreat[:3, 3]
    ret_quat = R.from_matrix(T_retreat[:3, :3]).as_quat(scalar_first=True)
    ret_w = np.concatenate([ret_pos, ret_quat])
    ret_r = transform_target_to_robot_frame(ret_w, ctx.robot_transform)
    _ = Move(ctx, ret_r.tolist(), time_dilation=0.8)

    return True
