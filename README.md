# telearms-gym

Gymnasium environments for the 100 TeleArms tasks on the RoboEval simulator: the 69 tasks people
teleoperated with the bimanual YAM robot, and 31 colour or letter variants of them (the same
scene and success check in other colours or letters).

## Install

Python 3.10 or newer. The repository is private, so install it with a GitHub account that can
read it:

```bash
pip install "git+ssh://git@github.com/liuwilli10101/telearms-gym.git"
# or, from a checkout:  pip install -e .
```

The simulator pins numpy 1.26 and gymnasium 0.29 (the stepjam/Gymnasium fork), so give it its own
environment. `camera_resolution` and `augmentation` also need torch
(`pip install "telearms-gym[augment] @ git+ssh://git@github.com/liuwilli10101/telearms-gym.git"`).

## Use

```python
import gymnasium as gym
import gymnasium_env.telearms_env  # registers telearms/<Task> for all 100 tasks

env = gym.make("telearms/CubePatchPlacement", obs_mode="privileged", reward_type="dense")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["instruction"])
```

`gym.make("gymnasium_env.telearms_env:telearms/CubePatchPlacement")` imports the module itself.
An id is `telearms/` plus the task name without spaces; `TeleArmsEnv.tasks()` lists the names.

| Argument | Default | Meaning |
|---|---|---|
| `obs_mode` | `"images"` | `"images"`: front, left and right 224×224 RGB cameras plus the 14 joint and gripper values. `"privileged"`: one vector of those 14 values, each cube's position, orientation and velocity, and each goal patch's position and orientation (its length per task is in `telearms_tasks.json` `privileged_dims`). |
| `reward_type` | `"sparse"` | `"sparse"`: 1 on the step that first succeeds. `"dense"`: also 1/N the first time each of the task's N stages (grasp, lift, place, ...) is reached; at most 2 per episode. |
| `augmentation` | `None` | `"mild"`, `"perspective"` or `"appearance"`: the v5 dataset's domain randomization of cameras, lights, textures and colours. |
| `camera_resolution` | `None` | `(640, 360)`: render like the real 16:9 cameras, letterboxed to 224×224. |
| `initial_pos_range` | `None` | `(min_x, max_x, min_y, max_y)`: place the cubes uniformly in that table area. |
| `max_steps` | `5400` | Truncate after 3 minutes at 30 Hz. |
| `render_mode` | `None` | `"human"` opens a viewer; `"rgb_array"` returns the three cameras side by side. |

Actions are 14 absolute joint targets at 30 Hz: for each arm its 6 joints, then its gripper.
An episode terminates on its first success.

Rendering uses `MUJOCO_GL=egl` by default. On a machine without EGL, use `obs_mode="privileged"`
and set `MUJOCO_GL=disable` before importing: privileged observations render nothing.

A policy that needs numpy 2 cannot share an interpreter with the simulator. Set `ROBOEVAL_PY` to
a Python that has telearms-gym installed, and `TeleArmsEnv`, created in an interpreter without
mujoco, runs the simulator in a subprocess of it with the same API.

## Datasets

- [mliu48/telearms](https://huggingface.co/datasets/mliu48/telearms): 30 Minari offline RL
  datasets with privileged observations (`telearms/<Task>/<subset>-v0`);
  `dataset.recover_environment()` builds this env.
- The `offline-rl` branch of `mliu48/telearms-yam-cube-v5-exp-{random,elo,score}-1500`: the same
  demos with images, as LeRobot datasets.
- [mliu48/telearms-yam-cube-v5-realistic](https://huggingface.co/datasets/mliu48/telearms-yam-cube-v5-realistic):
  the pi0.5 training set.

## Source

`roboeval/` is `roboeval/` of frodobots-org/telearms-roboplayground at commit 594af61 (branch
`telearms-main`), unchanged, under the Apache-2.0 license in `LICENSE`. `gymnasium_env/` is the
environment from the TeleArms training code.
