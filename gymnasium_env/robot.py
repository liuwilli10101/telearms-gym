"""The bimanual YAM as the policy sees it, and the messages RemoteTeleArmsEnv exchanges with
its simulator process (from the training repo's training/prepare_dataset.py)."""
from __future__ import annotations

import pickle
import socket
import struct
from dataclasses import dataclass

import numpy as np

# The robot's qpos and qvel come first in the simulator's state; the free objects follow.
YAM_SIM_ROBOT_QPOS_DIM = 16
YAM_SIM_ROBOT_QVEL_DIM = 16


@dataclass(frozen=True)
class RobotLayout:
    robot_type: str
    state_names: tuple[str, ...]
    camera_model_names: dict[str, str]
    camera_dataset_keys: dict[str, str]
    policy_from_sim_order: tuple[int, ...]
    policy_gripper_indices: tuple[int, ...]
    invert_policy_grippers: bool = False

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    def _check(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[-1] != self.state_dim:
            raise ValueError(f"{self.robot_type} expects {self.state_dim} values, got {values.shape}")
        return values

    def policy_from_sim(self, values: np.ndarray) -> np.ndarray:
        converted = self._check(values)[..., self.policy_from_sim_order].copy()
        if self.invert_policy_grippers:
            converted[..., self.policy_gripper_indices] = 1.0 - converted[..., self.policy_gripper_indices]
        return converted

    def sim_action_from_policy(self, values: np.ndarray) -> np.ndarray:
        converted = self._check(values).copy()
        if self.invert_policy_grippers:
            converted[..., self.policy_gripper_indices] = 1.0 - converted[..., self.policy_gripper_indices]
        simulation = np.empty_like(converted)
        simulation[..., self.policy_from_sim_order] = converted
        return simulation


YAM_LAYOUT = RobotLayout(
    "bimanual_yam",
    tuple([f"left_joint{i}" for i in range(1, 7)] + ["left_gripper"]
          + [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"]),
    {"front": "top_cam", "left": "left_cam", "right": "right_cam"},
    {"front": "camera_front", "left": "camera_left", "right": "camera_right"},
    tuple([*range(6), 12, *range(6, 12), 13]),
    (6, 13),
    True,
)


def resolve_camera_model_names(env) -> dict[str, str]:
    """{"front" | "left" | "right": the MuJoCo camera name}, matching namespaced names too."""
    available = {env._mojo.model.camera(i).name for i in range(env._mojo.model.ncam)}
    resolved: dict[str, str] = {}
    for key, wanted in YAM_LAYOUT.camera_model_names.items():
        if wanted in available:
            resolved[key] = wanted
            continue
        suffix_matches = sorted(name for name in available if name.endswith(f"/{wanted}"))
        if len(suffix_matches) != 1:
            raise ValueError(f"missing camera {wanted}; available: {sorted(available)}")
        resolved[key] = suffix_matches[0]
    return resolved


def send_message(conn: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_exact(conn: socket.socket, length: int) -> bytes | None:
    data = bytearray()
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_message(conn: socket.socket):
    header = _recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    payload = _recv_exact(conn, length)
    return pickle.loads(payload) if payload is not None else None
