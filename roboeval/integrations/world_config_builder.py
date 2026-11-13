import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R

"""Utilities to build a cuRobo-compatible WorldConfig with a single canonical frame.

Contract:
- Canonical frame can be either 'robot_base' or 'sim_world'.
- We collect raw geom poses in sim world frame.
- If target_frame == 'robot_base', we apply base_T_world (R,t) to every object exactly once.
- We annotate the returned dict with metadata so consumers can detect the frame and avoid double transforms.

Returned:
  world_config: dict  (structure matching MotionGen WorldConfig.from_dict)
  robot_transform: dict with keys R (3x3), t (3,), frame tags

Assumptions:
- physics.model.geom(i) API (MuJoCo) similar to what existing script uses.
- geom.group filtering via geom_group.
- geom.type codes follow MuJoCo enums: 3=capsule, 5=cylinder, 6=box, 7=mesh.
"""

# Geom type constants (correct MuJoCo enums)
GEOM_CAPSULE = 3
GEOM_CYLINDER = 5
GEOM_BOX = 6
GEOM_MESH = 7


def xmat_to_quat(R_flat: np.ndarray) -> np.ndarray:
    R_mat = np.reshape(R_flat, (3, 3))
    quat_xyzw = R.from_matrix(R_mat).as_quat()  # [x,y,z,w]
    return np.array([quat_xyzw[3], *quat_xyzw[:3]])  # [w,x,y,z]


def get_robot_transform(physics, robot_name: str) -> Dict[str, np.ndarray]:
    t_world_base = physics.named.data.xpos[robot_name]
    R_flat = physics.named.data.xmat[robot_name]
    R_world_base = np.reshape(R_flat, (3, 3))
    # For transforming world->base: x_base = R_world_base^T (x_world - t_world_base)
    R_base_world = R_world_base.T
    t_base_world = -R_base_world @ t_world_base
    return {
        "R_world_base": R_world_base,
        "t_world_base": t_world_base,
        "R": R_base_world,  # world->base rotation
        "t": t_base_world,  # world->base translation
        "from": "world",
        "to": "robot_base",
    }


def _collect_geoms(physics, geom_group: int, keywords_to_omit: Optional[List[str]] = None) -> List[int]:
    if keywords_to_omit is None:
        keywords_to_omit = []
    idx = np.where(physics.model.geom_group == geom_group)[0]
    kept = []
    for i in idx:
        name = physics.model.geom(i).name.decode() if hasattr(physics.model.geom(i).name, 'decode') else physics.model.geom(i).name
        if any(k in name for k in keywords_to_omit):
            continue
        kept.append(i)
    return kept


def _empty_world_config() -> Dict:
    return {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}, "_metadata": {}}


def _apply_transform_in_place(world_config: Dict, Rm: np.ndarray, tm: np.ndarray):
    for cat in ["cylinder", "cuboid", "mesh", "capsule"]:
        for obj_name, entry in world_config[cat].items():
            p = np.array(entry["pose"][0:3])
            p_new = Rm @ p + tm
            entry["pose"][0:3] = p_new.tolist()


def build_world_config(env, robot_name: str, geom_group: int = 3, target_frame: str = "robot_base", keywords_to_omit: Optional[List[str]] = None) -> Tuple[Dict, Dict[str, np.ndarray]]:
    physics = env._mojo.physics  # existing access pattern
    world_config = _empty_world_config()
    geom_indices = _collect_geoms(physics, geom_group, keywords_to_omit)

    for i in geom_indices:
        geom = physics.model.geom(i)
        data = physics.data.geom(i)
        name = geom.name.decode() if hasattr(geom.name, 'decode') else geom.name
        quat = xmat_to_quat(data.xmat)
        pose = data.xpos.tolist() + quat.tolist()

        gtype = geom.type[0]
        if gtype == GEOM_BOX:
            # Box: geom.size gives half-extents (sx, sy, sz)
            sx, sy, sz = geom.size
            world_config["cuboid"][name] = {
                "pose": pose,
                "dims": [2 * sx, 2 * sy, 2 * sz],
            }
        elif gtype == GEOM_CYLINDER:
            # Cylinder: size[0]=radius, size[1]=half-height
            radius, half_h = geom.size[0], geom.size[1]
            world_config["cylinder"][name] = {
                "pose": pose,
                "radius": float(radius),
                "height": float(2 * half_h),
            }
        elif gtype == GEOM_CAPSULE:
            radius, half_h = geom.size[0], geom.size[1]
            world_config["capsule"][name] = {
                "pose": pose,
                "radius": float(radius),
                "height": float(2 * half_h),
            }
        elif gtype == GEOM_MESH:
            # Skip mesh inclusion until robust mesh path+faces pipeline added
            continue
        else:
            continue

    # Ensure mesh category empty to avoid warp loader errors
    world_config["mesh"] = {}

    robot_transform = get_robot_transform(physics, robot_name)

    if target_frame == "robot_base":
        _apply_transform_in_place(world_config, robot_transform["R"], robot_transform["t"])
        world_config["_metadata"].update({
            "frame": "robot_base",
            "applied_transform": True,
            "robot_name": robot_name,
        })
    elif target_frame == "sim_world":
        world_config["_metadata"].update({
            "frame": "sim_world",
            "applied_transform": False,
            "robot_name": robot_name,
        })
    else:
        raise ValueError(f"Unsupported target_frame: {target_frame}")

    world_config["_metadata"]["robot_transform"] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in robot_transform.items()
        if k in ("R_world_base", "t_world_base")
    }

    # Debug: warn if no obstacles collected (causes MotionGen primitive collision error)
    total_primitives = sum(len(world_config[c]) for c in ("cuboid", "cylinder", "capsule"))
    if total_primitives == 0:
        print("[world_config_builder] WARNING: No primitive obstacles collected. Check geom_group or filtering keywords.")

    return world_config, robot_transform


def transform_pose_world_to_base(pose_w: List[float], robot_transform: Dict[str, np.ndarray]) -> List[float]:
    pose_w = np.array(pose_w)
    p_w = pose_w[:3]
    q_w = pose_w[3:]
    p_b = robot_transform["R"] @ p_w + robot_transform["t"]
    q_world = R.from_quat(q_w[1:].tolist() + [q_w[0]])  # convert to xyzw
    q_r_wr = R.from_matrix(robot_transform["R"])  # world->base
    q_b = (q_r_wr * q_world).as_quat()
    q_b_wxyz = np.array([q_b[3], q_b[0], q_b[1], q_b[2]])
    return np.concatenate([p_b, q_b_wxyz]).tolist()
