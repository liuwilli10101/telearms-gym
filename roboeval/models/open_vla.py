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

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from roboeval.action_modes import JointPositionActionMode
from roboeval.utils.observation_config import ObservationConfig, CameraConfig
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX

from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata
from demonstrations.demo_converter import DemoConverter

try:
    from moviepy.editor import VideoClip
except ImportError:
    raise ImportError("Install moviepy for video preview: pip install moviepy pygame")


def register_openvla():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_vla_model(ckpt_path: str, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    with open(os.path.join(ckpt_path, "dataset_statistics.json"), "r") as f:
        model.norm_stats = json.load(f)

    return processor, model


def setup_reach_target_dual_env(downsample_rate: int = 25):
    return ReachTargetDual(
        action_mode=JointPositionActionMode(
            floating_base=True, absolute=True, block_until_reached=False, ee=True, floating_dofs=[]
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=False, resolution=(256, 256)),
                CameraConfig(name="left_wrist", rgb=True, depth=False, resolution=(256, 256)),
                CameraConfig(name="right_wrist", rgb=True, depth=False, resolution=(256, 256)),
            ]
        ),
        render_mode=None,
        robot_cls=BimanualPanda,
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


def run_inference_loop(env, processor, model, instruction, use_demos=False, demo=None, device="cuda:0", max_steps=200):
    """Runs the agent in a closed loop using either demo or OpenVLA predictions."""
    obs = env.reset(seed=demo.seed if (use_demos and demo) else None)
    images = []
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    for step_idx in range(min(max_steps, len(demo.timesteps) if demo else max_steps)):
        if step_idx == 0:
            images.append(copy.deepcopy(obs[0]["rgb_head"]))
        else:
            images.append(obs["rgb_head"])

        if use_demos:
            action = demo.timesteps[step_idx].executed_action
        else:
            image = Image.fromarray(np.moveaxis(images[-1], 0, -1))
            inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
            action = model.predict_action(**inputs, do_sample=False)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return images


def visualize_frames(frames, fps=25):
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.preview()

def save_frames_to_video(frames, ckpt_path, filename="rollout.mp4", fps=25):
    """Save rollout frames to a video file in the checkpoint directory."""
    output_path = os.path.join(ckpt_path, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = np.moveaxis(np.array(frames), 1, -1)  # (T, C, H, W) → (T, H, W, C)
    duration = len(frames) / fps
    clip = VideoClip(make_frame=lambda t: frames[min(int(t * fps), len(frames) - 1)], duration=duration)
    clip.write_videofile(output_path, fps=fps, codec="libx264")
    print(f"Saved rollout video to: {output_path}")



def main():
    # === Configs ===
    # ckpt_path = "/media/mnt1/projects/bimanual/openvla/reach_ckpt"
    ckpt_path = "/media/mnt1/projects/bimanual/openvla/bimauanl_data_abs"
    dataset_path = "/home/helen/Documents/projects/bimanual/roboeval/collected_demos_oculus/Reach_Target_Dual/Bimanual Panda"
    use_demos = False  # Toggle this!
    instruction = "reach the red sphere with the left hand and the green sphere with the right hand"
    device = "cuda:0"
    downsample_rate = 25
    max_steps = 200
    num_episodes = 5  # 🔁 Run multiple episodes

    # === Load VLA ===
    register_openvla()
    processor, model = load_vla_model(ckpt_path, device)

    # === Setup Env ===
    env = setup_reach_target_dual_env(downsample_rate=downsample_rate)

    # === Get Demos if Needed ===
    demos = get_successful_demos(env, dataset_path, downsample_rate, num_demos=num_episodes) if use_demos else []

    for ep in range(num_episodes):
        print(f"\nRunning episode {ep + 1}/{num_episodes} ({'demo' if use_demos else 'inference'})")
        demo = DemoConverter.joint_to_ee(demos[ep]) if use_demos else None

        frames = run_inference_loop(
            env, processor, model, instruction, use_demos, demo, device, max_steps
        )

        # Save video
        mode = "demo" if use_demos else "inference"
        filename = f"rollout_{mode}_ep{ep}.mp4"
        save_frames_to_video(frames, ckpt_path, filename=filename, fps=5)

    env.close()



if __name__ == "__main__":
    main()