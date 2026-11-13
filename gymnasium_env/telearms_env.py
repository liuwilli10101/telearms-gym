"""A gymnasium env for one TeleArms task, in the policy's own interface.

    env = TeleArmsEnv("Hi Cube Placement")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The TeleArms v5 dataset's 69 tasks and 31 of its colour/letter variants (telearms_tasks.json)
are registered with gymnasium, 100 in all, by name without spaces:

    import gymnasium_env.telearms_env
    env = gym.make("telearms/FrontToBackCubeLine", initial_pos_range=(0.40, 0.70, -0.25, 0.25),
                   augmentation="mild")

A variant is its base task shown in other colours or letters (variants.py): the same scene and
success check, with the instruction rewritten to match; info["instruction"] gives each task's
instruction, one short LIBERO-style sentence ("Put the red cube behind the blue cube.").

initial_pos_range=(min_x, max_x, min_y, max_y) places every free object (the cubes) uniformly
in that table area at reset, instead of where the task puts them. augmentation renders like
the v5 dataset: the real cameras and cell under one of its domain randomizations, drawn
once per episode ("mild" = its -d1 realistic, "perspective" = -d2 viewpoint,
"appearance" = -d3 appearance).

Every reset holds the robot still for a few steps so the cubes rest, and draws again (from a
stream the seed fixes) while the task already counts as done.

Observations and actions match what pi0.5 sees: three 224x224 RGB cameras plus
the 14-dim policy state; actions are the 14-dim absolute policy action. Reward
is 1.0 on the first step the task succeeds, else 0.0 (reward_type="sparse").
reward_type="dense" adds 1/N on the first step each of the task's own stages (RoboEval's
stage flags: grasp, lift, place, ...) is reached, N being the stages the episode can still
earn: not reached while the cubes settle at reset, and passed by at least half of the task's
successful human demos (the rest are telearms_tasks.json "unrewarded_stages"). An
episode's dense return is at most 2.

obs_mode="privileged" replaces the cameras with the simulator's object state, as the offline RL
datasets' state-based version records it (privileged_obs): a flat float32 vector of the 14-dim
policy state, every free object's qpos (position, wxyz quaternion) and qvel, and each goal
patch's position and wxyz quaternion (patches have no qpos; the task places them at reset).
Its length depends on the task (telearms_tasks.json "privileged_dims"). Nothing is rendered;
on a machine without EGL, set MUJOCO_GL=disable before importing this module.

The simulator needs numpy 1.26 and gymnasium 0.29. In an interpreter without mujoco (say, a
policy's, with numpy 2), TeleArmsEnv runs it in a subprocess of the Python that ROBOEVAL_PY
names, which must have this package installed: the same API, over a local socket.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np

# mujoco and the RoboEval env stack are imported in reset(), so this module also imports where
# only RemoteTeleArmsEnv can run.
from .contracts import CAMERA_PROFILE, apply_camera_profile, set_policy_control_rate, step_policy
from .robot import YAM_LAYOUT, YAM_SIM_ROBOT_QPOS_DIM, YAM_SIM_ROBOT_QVEL_DIM, resolve_camera_model_names

# The v5 dataset's tasks, each with its env class, and the env settings all their demos share.
TASKS = json.loads((Path(__file__).with_name("telearms_tasks.json")).read_text())
# augmentation -> sim_render.DR_PROFILES, as the v5 dataset's -d1/-d2/-d3 renders use them
AUGMENTATIONS = {"mild": "realistic", "perspective": "viewpoint", "appearance": "appearance"}
CUBE_SEPARATION = 0.07  # m between free objects placed by initial_pos_range (cubes are 4 cm)
SETTLE_STEPS = 10       # policy steps the robot holds still at reset, so the cubes rest on the table
RESET_TRIES = 100       # draws before giving up on a start the task does not already count as done


def goal_patches(env) -> list:
    """The task's goal patches and targets (GoalPatch props of a RoboEval env), in the order the
    task creates them."""
    from roboeval.envs.props.items import GoalPatch
    found = {}
    for value in vars(env).values():
        for prop in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(prop, GoalPatch):
                found.setdefault(id(prop), prop)
    return list(found.values())


def privileged_obs(env, layout=YAM_LAYOUT) -> np.ndarray:
    """obs_mode="privileged" of a RoboEval env: the policy state, the free objects' qpos and
    qvel, then each goal patch's world position and wxyz quaternion."""
    data = env._mojo.data
    state = layout.policy_from_sim(np.asarray(env.robot.qpos_actuated, np.float32))  # as _obs reads it
    ids = [p.body.id for p in goal_patches(env)]
    return np.concatenate([state, data.qpos[YAM_SIM_ROBOT_QPOS_DIM:], data.qvel[YAM_SIM_ROBOT_QVEL_DIM:],
                           np.concatenate([data.xpos[ids], data.xquat[ids]], axis=1).ravel()]).astype(np.float32)


def observation_space(base_task: str, size: int, obs_mode: str) -> gym.spaces.Space:
    if obs_mode == "privileged":
        return gym.spaces.Box(-np.inf, np.inf, (TASKS["privileged_dims"][base_task],), np.float32)
    img = gym.spaces.Box(0, 255, (size, size, 3), np.uint8)
    return gym.spaces.Dict({
        "images": gym.spaces.Dict({c: img for c in ("front", "left", "right")}),
        "state": gym.spaces.Box(-np.inf, np.inf, (14,), np.float32)})


class TeleArmsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __new__(cls, *args, **kwargs):
        # One class for both kinds of interpreter: with mujoco the simulator runs in this
        # process; without it the same API is served from a ROBOEVAL_PY subprocess.
        import importlib.util
        if cls is TeleArmsEnv and importlib.util.find_spec("mujoco") is None:
            return RemoteTeleArmsEnv(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, task: str, fps: int = 30, size: int = 224, max_steps: int = 5400,
                 render_mode: str | None = None, camera_resolution: tuple[int, int] | None = None,
                 initial_pos_range: tuple[float, float, float, float] | None = None,
                 augmentation: str | None = None, reward_type: str = "sparse", obs_mode: str = "images"):
        """camera_resolution=(640, 360) renders each camera like a real 16:9 one and
        letterboxes it to `size`, as the v5 dataset's frames are made.
        initial_pos_range, augmentation and obs_mode: see the module docstring."""
        if task not in TASKS["instructions"]:
            raise ValueError(f"unknown task {task!r}; TeleArmsEnv.tasks() lists the 100 tasks")
        if obs_mode not in ("images", "privileged"):
            raise ValueError(f"obs_mode must be 'images' or 'privileged', not {obs_mode!r}")
        if obs_mode == "privileged" and (augmentation or camera_resolution or render_mode == "rgb_array"):
            raise ValueError("obs_mode='privileged' renders no cameras: leave augmentation, camera_resolution "
                             "and render_mode='rgb_array' unset")
        if augmentation is not None and augmentation not in AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {sorted(AUGMENTATIONS)} or None, not {augmentation!r}")
        if initial_pos_range is not None:
            min_x, max_x, min_y, max_y = initial_pos_range
            if not (min_x < max_x and min_y < max_y):
                raise ValueError(f"initial_pos_range is (min_x, max_x, min_y, max_y), got {initial_pos_range}")
        if augmentation is not None and size != 224:
            raise ValueError("augmentation renders the dataset's 224x224 letterbox; leave size at 224")
        if reward_type not in ("sparse", "dense"):
            raise ValueError(f"reward_type must be 'sparse' or 'dense', not {reward_type!r}")
        self.reward_type, self.obs_mode = reward_type, obs_mode
        self.render_mode = render_mode
        self.camera_resolution = camera_resolution
        self.initial_pos_range, self.augmentation = initial_pos_range, augmentation
        self.variant = TASKS["variants"].get(task)
        self.base_task = self.variant["base"] if self.variant else task
        self.instruction = TASKS["instructions"][task]
        self.task, self.fps, self.size, self.max_steps, self.env = task, fps, size, max_steps, None
        self.layout = YAM_LAYOUT
        self.action_space = gym.spaces.Box(-np.pi, np.pi, (14,), np.float32)
        self.observation_space = observation_space(self.base_task, size, obs_mode)

    @staticmethod
    def tasks() -> list[str]:
        """The 69 tasks, then the 31 variants."""
        return [*TASKS["tasks"], *TASKS["variants"]]

    def _metadata(self):
        from roboeval.demonstrations.utils import EnvData, Metadata, ObservationMode
        data = EnvData.from_safetensors_metadata(dict(TASKS["environment_data"], env_name=TASKS["tasks"][self.base_task]))
        return Metadata(observation_mode=ObservationMode(TASKS["observation_mode"]), environment_data=data,
                        seed=0, package_versions={})

    def reset(self, *, seed=None, options=None):
        import mujoco

        super().reset(seed=seed)
        if self.env is not None:
            self.close()
        self.env = self._metadata().get_env(self.fps, render_mode=None)
        # One stream for this episode's draws: the seed's, or gym's own.
        rng = np.random.default_rng(seed if seed is not None else int(self.np_random.integers(2**63)))
        # Some tasks' own resets are sometimes already solved (Align Blocks, Rotate Letter Block:
        # about 1 seed in 3), and so are some placements: hold still until the cubes rest, which
        # success checks need, and draw again from this stream while the task counts as done.
        for attempt in range(RESET_TRIES):
            self.env.reset(seed=seed if attempt == 0 else int(rng.integers(2**31)))
            set_policy_control_rate(self.env, self.fps)
            apply_camera_profile(self.env, CAMERA_PROFILE)
            if self.initial_pos_range is not None:
                self._place_free_objects(rng)
            hold = self.layout.sim_action_from_policy(
                self.layout.policy_from_sim(np.asarray(self.env.robot.qpos_actuated, np.float64)))
            settled = set()  # stages done at any point while the cubes settle
            for _ in range(SETTLE_STEPS):
                step_policy(self.env, hold)
                settled |= {k for k, done in self.env._stage_flags.items() if done}
            if not self.env.success:
                break
        else:
            raise RuntimeError(f"{self.task!r} started solved in {RESET_TRIES} draws")
        # Dense reward: stages done while settling, or most human demos skip, earn nothing.
        self.paid_stages = settled | set(TASKS["unrewarded_stages"].get(self.base_task, []))
        self.stage_reward = 1 / max(1, len(set(self.env._stage_flags) - self.paid_stages))
        if self.variant:  # after the task's reset, which sets its colours; before any renderer exists
            from .variants import apply_colors, apply_letters, letter_bijection
            apply_colors(self.env, self.variant["colors"])
            if self.variant["letters"]:
                apply_letters(self.env, letter_bijection(self.variant["letters"], self.task))
        self.cams = resolve_camera_model_names(self.env)
        self.letterbox = self.aug = self.dr = self.renderer = None
        if self.augmentation is not None:
            self._randomize(rng)
        elif self.obs_mode == "images":
            width = height = self.size
            if self.camera_resolution:
                from .sim_render import SHADOW_SIZE, Letterbox, apply_real_cameras
                width, height = self.camera_resolution
                self.letterbox = Letterbox(width, height, self.size)
                model = self.env._mojo.model
                apply_real_cameras(model, self.cams)
                model.vis.quality.shadowsize = SHADOW_SIZE
                model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
                model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self.renderer = mujoco.Renderer(self.env._mojo.model, height=height, width=width)
        if self.render_mode == "human":
            from mujoco import viewer  # an interactive window, like robosuite's has_renderer
            self.viewer = viewer.launch_passive(self.env._mojo.model, self.env._mojo.data)
        self.steps, self.succeeded = 0, False
        return self._obs(), self._info()

    def _place_free_objects(self, rng: np.random.Generator) -> None:
        """Put every free body uniformly in initial_pos_range, CUBE_SEPARATION apart, keeping its
        height and orientation."""
        import mujoco
        model, data = self.env._mojo.model, self.env._mojo.data
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        min_x, max_x, min_y, max_y = self.initial_pos_range
        placed = []
        for j in free:
            for _ in range(1000):
                p = rng.uniform((min_x, min_y), (max_x, max_y))
                if all(np.linalg.norm(p - q) >= CUBE_SEPARATION for q in placed):
                    break
            else:
                raise ValueError(f"cannot fit {len(free)} objects {CUBE_SEPARATION} m apart in {self.initial_pos_range}")
            placed.append(p)
            a, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
            data.qpos[a:a + 2] = p
            data.qvel[v:v + 6] = 0.0
        self.env.mojo.physics.forward()
        self.env._step_cache.clean()

    def _randomize(self, rng: np.random.Generator) -> None:
        """Render like the v5 dataset's DR renders: the real cameras and cell, the episode's lights,
        textures and camera poses, then the sensor response."""
        from .sim_render import DR_PROFILES, ImageAug, VisualRandomizer, WideRenderer, apply_real_cameras
        from .variants import colored_geom_ids
        model = self.env._mojo.model
        apply_real_cameras(model, self.cams)
        randomizer = VisualRandomizer(model)  # sets the shadow map size before the GL context exists
        self.renderer = WideRenderer(model, self.cams)
        profile = AUGMENTATIONS[self.augmentation]
        tinted = colored_geom_ids(self.env) if DR_PROFILES[profile]["object_color"] else None
        self.dr = randomizer.randomize(rng, self.cams, profile=profile, colored=tinted)
        level = DR_PROFILES[profile]["image"]
        if level:
            self.aug = ImageAug(level, rng, list(self.cams), self.renderer.letterbox)
            self.dr["image"] = {"level": level, **self.aug.params}
        self.renderer.upload_textures()

    def step(self, action):
        step_policy(self.env, self.layout.sim_action_from_policy(np.asarray(action, np.float64)))
        self.steps += 1
        newly = bool(self.env.success) and not self.succeeded
        self.succeeded |= newly
        reward = float(newly)
        if self.reward_type == "dense":  # stage flags can turn off again; each stage pays once
            new = {k for k, done in self.env._stage_flags.items() if done} - self.paid_stages
            self.paid_stages |= new
            reward += self.stage_reward * len(new)
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, self.succeeded, self.steps >= self.max_steps, self._info()

    def _obs(self):
        if self.obs_mode == "privileged":
            return privileged_obs(self.env, self.layout)
        if self.augmentation is not None:
            frames = self.renderer.render(self.env._mojo.data)  # already letterboxed
            images = {cam: frames[cam] if self.aug is None else self.aug(cam, frames[cam]) for cam in self.cams}
        else:
            images = {}
            for cam, name in self.cams.items():
                self.renderer.update_scene(self.env._mojo.data, camera=name)
                pixels = self.renderer.render()
                images[cam] = pixels.copy() if self.letterbox is None else self.letterbox(pixels)
        state = self.layout.policy_from_sim(np.asarray(self.env.robot.qpos_actuated, np.float32))
        self._last_images = images
        return {"images": images, "state": state.astype(np.float32)}

    def render(self):
        """human: refresh the viewer window. rgb_array: front | left | right cameras."""
        if self.render_mode == "human" and self.viewer.is_running():
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            return np.concatenate([self._last_images[c] for c in ("front", "left", "right")], axis=1)

    def _info(self):
        return {"task": self.task, "instruction": self.instruction, "success": self.succeeded,
                "task_info": self.env.get_info()}

    def close(self):
        if self.env is not None:
            if self.render_mode == "human":
                self.viewer.close()
            if self.renderer is not None:
                self.renderer.close()
            self.env.close()
            self.env = None


# ---------------------------------------------------------------------------
# A policy often needs numpy 2 while the simulator needs numpy 1.26, and the two cannot share
# an interpreter. RemoteTeleArmsEnv is the same gym API with the simulator in a subprocess of
# ROBOEVAL_PY, speaking a length-prefixed pickle protocol (robot.send_message).

ROBOEVAL_PY = os.environ.get("ROBOEVAL_PY")
# A source checkout (rather than an installed package) goes on the subprocess's path.
REPO = Path(__file__).resolve().parents[1]


class RemoteTeleArmsEnv(gym.Env):
    metadata = TeleArmsEnv.metadata
    tasks = staticmethod(TeleArmsEnv.tasks)

    def __init__(self, task: str, **kwargs):
        import socket
        import subprocess
        import time

        if not ROBOEVAL_PY:
            raise RuntimeError("this interpreter has no mujoco: set ROBOEVAL_PY to a Python that has "
                               "telearms-gym installed (numpy 1.26), and it runs the simulator")
        self.render_mode = kwargs.get("render_mode")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(os.environ)
        if (REPO / "pyproject.toml").is_file():
            env["PYTHONPATH"] = str(REPO)
        # The simulator prints a line for every out-of-bounds action; its errors come
        # back over the socket, so its stdout is only noise here.
        self.proc = subprocess.Popen([ROBOEVAL_PY, "-m", "gymnasium_env.telearms_env", "--serve", str(port)],
                                     env=env, stdout=subprocess.DEVNULL)
        for _ in range(240):
            try:
                self.conn = socket.create_connection(("127.0.0.1", port), timeout=1)
                break
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError("env server exited during startup")
                time.sleep(0.5)
        else:
            raise RuntimeError("could not connect to the env server")
        self.conn.settimeout(None)
        self.action_space = gym.spaces.Box(-np.pi, np.pi, (14,), np.float32)
        variant = TASKS["variants"].get(task)
        self.observation_space = observation_space(variant["base"] if variant else task, kwargs.get("size", 224),
                                                   kwargs.get("obs_mode", "images"))
        self._call("make", task, **kwargs)

    def _call(self, cmd, *args, **kwargs):
        from .robot import recv_message, send_message
        send_message(self.conn, (cmd, args, kwargs))
        status, value = recv_message(self.conn)
        if status != "ok":
            raise RuntimeError(f"env server: {value}")
        return value

    def reset(self, *, seed=None, options=None):
        return self._call("reset", seed=seed)

    def step(self, action):
        return self._call("step", np.asarray(action, np.float64))

    def render(self):
        return self._call("render")

    def close(self):
        if getattr(self, "proc", None) is None:
            return
        try:
            self._call("close")
        except Exception:
            pass
        self.conn.close()
        self.proc.wait(timeout=30)
        self.proc = None


def _serve(port: int) -> None:
    import socket
    import traceback
    from .robot import recv_message, send_message

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    conn, _ = server.accept()
    env = None
    try:
        while (message := recv_message(conn)) is not None:
            cmd, args, kwargs = message
            try:
                if cmd == "make":
                    env, value = TeleArmsEnv(*args, **kwargs), None
                else:
                    value = getattr(env, cmd)(*args, **kwargs)
                send_message(conn, ("ok", value))
            except Exception:
                send_message(conn, ("err", traceback.format_exc()))
            if cmd == "close":
                break
    except (ConnectionError, KeyboardInterrupt):
        pass  # the client went away, or Ctrl+C reached the whole process group
    finally:
        if env is not None:
            env.close()  # before interpreter exit, which otherwise trips over the EGL context
        conn.close()
        server.close()


def gym_id(task: str) -> str:
    """'Front To Back Cube Line' -> 'telearms/FrontToBackCubeLine'"""
    return "telearms/" + "".join(word[:1].upper() + word[1:] for word in task.split())


# A string entry point, so a spec can be saved and rebuilt (minari's recover_environment).
for _task in TeleArmsEnv.tasks():
    if gym_id(_task) not in gym.registry:
        gym.register(gym_id(_task), entry_point="gymnasium_env.telearms_env:TeleArmsEnv", kwargs={"task": _task},
                     disable_env_checker=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--serve"]:
        _serve(int(sys.argv[2]))
