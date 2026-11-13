import os
import json
import torch
import copy
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor
)

import argparse

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from roboeval.action_modes import JointPositionActionMode
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.robots.configs.panda import BimanualPanda, SinglePanda
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

from roboeval.demonstrations.demo_store import DemoStore
from roboeval.demonstrations.utils import Metadata
from roboeval.demonstrations.demo_converter import DemoConverter

try:
    from moviepy.editor import VideoClip
except ImportError:
    raise ImportError("Install moviepy for video preview: pip install moviepy pygame")

from typing import Type, Union, Callable, Optional, Dict


from roboeval.roboeval_env import RoboEvalEnv



from roboeval.envs.task_library.stack_blocks import StackTwoBlocksEnv
from roboeval.envs.task_library.push_block_to_target import PushBlockToTarget
from roboeval.envs.task_library.rotate_cube import RotateLetterBlockEnv
from roboeval.envs.task_library.align_blocks import AlignBlocksEnv


ENVIRONMENTS: dict[str, Type[RoboEvalEnv]] = {
    "Stack Two Blocks": StackTwoBlocksEnv,
    "Push Block to Target": PushBlockToTarget,
    "Rotate Letter Block": RotateLetterBlockEnv,
    "Align Blocks": AlignBlocksEnv,
}


def register_openvla():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_vla_model(ckpt_path: str, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # with open(os.path.join(ckpt_path, "dataset_statistics.json"), "r") as f:
    #     model.norm_stats = json.load(f)

    return processor, model




def setup_reach_target_dual_env(downsample_rate: int = 25):
    return StackTwoBlocksEnv(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, block_until_reached=False, ee=True, floating_dofs=[]
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=False, resolution=(256, 256)),
                CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(256, 256)),
            ]
        ),
        render_mode='rgb_array',
        robot_cls=SinglePanda,
        control_frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )


def get_successful_demos(env, dataset_path: str, downsample_rate: int, num_demos: int = 1):
    demo_store = DemoStore()
    dataset_path = Path(dataset_path)
    demos = demo_store.get_demos_from_folder(
        dataset_path,
        Metadata.from_env(env),
        amount=num_demos,
        frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )
    return [
        copy.deepcopy(demo)
        for demo in demos
        if sum(step.reward for step in demo.timesteps) > 0.25
    ]


def run_inference_loop(
    env,
    processor,
    model,
    instruction,
    use_demos=False,
    demo=None,
    device="cuda:0",
    max_steps=1000,
    unnorm_key: str | None = None,
):
    """Runs the agent in a closed loop using either demo or OpenVLA predictions."""
    obs = env.reset(seed=demo.seed if (use_demos and demo) else None)
    images_env, images, actions, infos, observations = [], [], [], [], []

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    for step_idx in range(min(max_steps, len(demo.timesteps) if demo else max_steps)):
        image = obs[0]["rgb_head"] if step_idx == 0 else obs["rgb_head"]
        images.append(copy.deepcopy(image))
        images_env.append(copy.deepcopy(env.render()))
        observations.append(copy.deepcopy(obs[0]) if step_idx == 0 else copy.deepcopy(obs))

        if use_demos:
            action = demo.timesteps[step_idx].executed_action
        else:
            pil_image = Image.fromarray(np.moveaxis(image, 0, -1))
            inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, do_sample=False, unnorm_key="bridge_orig")

        # Ensure action shape matches environment expectations (e.g., add gripper dim if missing)
        # action_array = np.array(action, dtype=np.float32).reshape(-1)
        # expected_dims = int(np.prod(env.action_space.shape)) if hasattr(env, "action_space") else action_array.shape[0]
        # if action_array.shape[0] != expected_dims:
        #     if action_array.shape[0] + 1 == expected_dims:
        #         action_array = np.concatenate([action_array, np.array([0.0], dtype=np.float32)])
        #     else:
        #         print(
        #             f"Warning: Adjusting action dim from {action_array.shape[0]} to {expected_dims} by padding zeros."
        #         )
        #         pad_width = expected_dims - action_array.shape[0]
        #         if pad_width > 0:
        #             action_array = np.concatenate([action_array, np.zeros(pad_width, dtype=np.float32)])
        #         else:
        #             action_array = action_array[:expected_dims]

        actions.append(action.tolist())
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append(copy.deepcopy(info))

        if terminated or truncated:
            break

    return images_env, images, actions, observations, infos

def visualize_frames(frames, fps=25):
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.preview()

def save_frames_to_video(frames, video_path, filename="rollout.mp4", fps=25):
    """Save rollout frames to a video file in the checkpoint directory."""
    output_path = os.path.join(video_path, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if 'env' in filename:
        frames = np.array(frames)
        # frames = np.moveaxis(frames, 0, -1)  # (T, C, H, W) → (T, H, W, C)
    else:
        frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264")
    print(f"Saved rollout video to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA Inference")
    parser.add_argument("--ckpt_path", type=str, required=True, default="", help="Path to the checkpoint directory")
    parser.add_argument("--env_name", type=str, required=True, default="StackTwoBlocks", help="Environment name")
    parser.add_argument("--use_demos", action="store_true", help="Use demos for inference")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset for demos")
    parser.add_argument("--instruction", type=str, required=True, default="Stack the blocks", help="Instruction for the model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--downsample_rate", type=int, default=25, help="Downsample rate for the environment")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps for each episode")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--unnorm_key", type=str, help="Dataset statistics key to unnormalize actions (optional)")

    return parser.parse_args()

def setup_env(downsample_rate: int = 25, env_name: str = "ReachTargetDual"):
    """Set up the environment based on the provided environment name."""

    env = ENVIRONMENTS.get(env_name)

    if env is None:
        raise ValueError(f"Unsupported environment name: {env_name}")
    

    return env(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=False, block_until_reached=False, ee=True, floating_dofs=[]
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=False, resolution=(256, 256)),
                CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(256, 256)),
            ]
        ),
        render_mode='rgb_array',
        robot_cls=SinglePanda,
        control_frequency=CONTROL_FREQUENCY_MAX // downsample_rate,
    )
def save_json_data(data, filepath):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)  # fallback: convert unknown types to string

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=convert)

def filter_dict(d, keys_to_filter_out):
    """Filter out keys from a dictionary."""
    return {k: v for k, v in d.items() if k not in keys_to_filter_out}

def main():
    # === Configs ===
    # ckpt_path = "/media/mnt1/projects/bimanual/openvla/reach_ckpt"

    # Set up input args using argparse
    args = parse_args()
    ckpt_path = args.ckpt_path
    env_name = args.env_name
    use_demos = args.use_demos
    dataset_path = args.dataset_path
    instruction = args.instruction
    device = args.device
    downsample_rate = args.downsample_rate
    max_steps = args.max_steps
    num_episodes = args.num_episodes
    unnorm_key = args.unnorm_key

    print(f"Loading model for {env_name} from {ckpt_path}")

    # dataset_path = "/home/helen/Documents/projects/bimanual/roboeval/collected_demos_oculus/Reach_Target_Dual/Bimanual Panda"
    use_demos = False  # Toggle this!
    # instruction = "reach the red sphere with the left hand and the green sphere with the right hand"
    # device = "cuda:0"
    # downsample_rate = 25
    # max_steps = 200
    # num_episodes = 5  # 🔁 Run multiple episodes

    # === Load VLA ===
    register_openvla()
    processor, model = load_vla_model(ckpt_path, device)

    if unnorm_key is None:
        # Try environment-specific default, otherwise fall back to first key available
        default_unnorm_map = {
            "Stack Two Blocks": "bridge_orig",
            "Push Block to Target": "bridge_orig",
            "Rotate Letter Block": "bridge_orig",
            "Align Blocks": "bridge_orig",
        }
        suggested_key = default_unnorm_map.get(env_name)
        if suggested_key and suggested_key in getattr(model, "norm_stats", {}):
            unnorm_key = suggested_key
        else:
            norm_stats_keys = list(getattr(model, "norm_stats", {}).keys())
            if norm_stats_keys:
                unnorm_key = norm_stats_keys[0]
                print(
                    "Info: No unnorm_key provided; defaulting to first available statistics key "
                    f"'{unnorm_key}'."
                )
            else:
                unnorm_key = None

    if unnorm_key and unnorm_key not in getattr(model, "norm_stats", {}):
        print(
            f"Warning: Requested unnorm_key '{unnorm_key}' not found in model statistics. "
            "Falling back to first available key."
        )
        norm_stats_keys = list(getattr(model, "norm_stats", {}).keys())
        unnorm_key = norm_stats_keys[0] if norm_stats_keys else None

    if unnorm_key:
        print(f"Using unnorm_key: {unnorm_key}")

    # === Setup Env ===
    env = setup_env(downsample_rate=downsample_rate, env_name=env_name)

    # === Get Demos if Needed ===
    demos = get_successful_demos(env, dataset_path, downsample_rate, num_demos=num_episodes) if use_demos else []

    for ep in range(num_episodes):
        print(f"\nRunning episode {ep + 1}/{num_episodes} ({'demo' if use_demos else 'inference'})")
        demo = DemoConverter.joint_to_ee(demos[ep]) if use_demos else None

        images_env, frames, actions, observations, infos = run_inference_loop(
            env,
            processor,
            model,
            instruction,
            use_demos,
            demo,
            device,
            max_steps,
            unnorm_key=unnorm_key,
        )

        # === Save video ===
        mode = "demo" if use_demos else "inference"
        filename = f"rollout_{mode}_ep{ep}.mp4"
        filename_env = f"rollout_{mode}_env_ep{ep}.mp4"
        save_frames_to_video(frames, ckpt_path, filename=filename, fps=5)
        save_frames_to_video(images_env, ckpt_path, filename=filename_env, fps=5)

        # === Save JSON ===
        observations = [
            filter_dict(obs, ["rgb_head", "rgb_left_wrist"])
            for obs in observations
        ]
        data_dir = os.path.join(ckpt_path, "rollout_data", f"{mode}_ep{ep}")
        save_json_data(actions, os.path.join(data_dir, "actions.json"))
        save_json_data(observations, os.path.join(data_dir, "observations.json"))
        save_json_data(infos, os.path.join(data_dir, "infos.json"))


    env.close()

def run_policy(env, ckpt_path="", video_path="/home/carter/RoboEval/examples/openvla_rollouts", env_name="StackTwoBlocks", use_demos=False, dataset_path="", instruction="Stack the blocks", device="cuda:0", downsample_rate=25, max_steps=500, num_episodes=2, unnorm_key="bridge_orig"):
    ckpt_path = ckpt_path
    video_path = video_path
    env_name = env_name
    use_demos = use_demos
    dataset_path = dataset_path
    instruction = instruction
    device = device
    downsample_rate = downsample_rate
    max_steps = max_steps
    num_episodes = num_episodes
    unnorm_key = "bridge_orig"

    print(f"Loading model for {env_name} from {ckpt_path}")

    # dataset_path = "/home/helen/Documents/projects/bimanual/roboeval/collected_demos_oculus/Reach_Target_Dual/Bimanual Panda"
    use_demos = False  # Toggle this!
    # instruction = "reach the red sphere with the left hand and the green sphere with the right hand"
    # device = "cuda:0"
    # downsample_rate = 25
    # max_steps = 200
    # num_episodes = 5  # 🔁 Run multiple episodes

    # === Load VLA ===
    register_openvla()
    processor, model = load_vla_model(ckpt_path, device)


    if unnorm_key and unnorm_key not in getattr(model, "norm_stats", {}):
        print(
            f"Warning: Requested unnorm_key '{unnorm_key}' not found in model statistics. "
            "Falling back to first available key."
        )
        norm_stats_keys = list(getattr(model, "norm_stats", {}).keys())
        unnorm_key = norm_stats_keys[0] if norm_stats_keys else None

    if unnorm_key:
        print(f"Using unnorm_key: {unnorm_key}")


    # === Get Demos if Needed ===
    demos = get_successful_demos(env, dataset_path, downsample_rate, num_demos=num_episodes) if use_demos else []

    for ep in range(num_episodes):
        print(f"\nRunning episode {ep + 1}/{num_episodes} ({'demo' if use_demos else 'inference'})")
        demo = DemoConverter.joint_to_ee(demos[ep]) if use_demos else None

        images_env, frames, actions, observations, infos = run_inference_loop(
            env,
            processor,
            model,
            instruction,
            use_demos,
            demo,
            device,
            max_steps,
            unnorm_key=unnorm_key,
        )

        # === Save video ===
        mode = "demo" if use_demos else "inference"
        filename = f"rollout_{mode}_ep{ep}.mp4"
        filename_env = f"rollout_{mode}_env_ep{ep}.mp4"
        save_frames_to_video(frames, video_path, filename=filename, fps=5)
        save_frames_to_video(images_env, video_path, filename=filename_env, fps=5)

        # === Save JSON ===
        observations = [
            filter_dict(obs, ["rgb_head", "rgb_left_wrist"])
            for obs in observations
        ]
        data_dir = os.path.join(video_path, "rollout_data", f"{mode}_ep{ep}")
        save_json_data(actions, os.path.join(data_dir, "actions.json"))
        save_json_data(observations, os.path.join(data_dir, "observations.json"))
        save_json_data(infos, os.path.join(data_dir, "infos.json"))


    env.close()
    return os.path.join(video_path, filename)


if __name__ == "__main__":
    main()