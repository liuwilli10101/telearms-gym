"""Core RoboEval environment functionality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type
from yaml import safe_load

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mojo import Mojo
from mojo.elements import Geom, Camera, Site
from mojo.elements.consts import SiteType
from scipy.spatial.transform import Rotation

from roboeval.action_modes import ActionMode
from roboeval.const import WORLD_MODEL
from roboeval.envs.arenas.arena import Arena
from roboeval.envs.props.preset import Preset
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.robots.robot import Robot
from roboeval.roboeval_renderer import RoboEvalRenderer
from roboeval.utils.callables_cache import CallablesCache
from roboeval.utils.env_health import EnvHealth
from roboeval.utils.observation_config import ObservationConfig
from roboeval.utils.spawn_boundary import SpawnBoundary
from roboeval.utils.temporal_success_checker import TemporalSuccessChecker

from roboeval.ik.base_ik import GenericUpperBodyIK

CONTROL_FREQUENCY_MAX = 500
CONTROL_FREQUENCY_MIN = 20

PHYSICS_DT = 0.002

MAX_DISTANCE_FROM_ORIGIN = 10
SPARSE_REWARD_FACTOR = 1


class RoboEvalEnv(gym.Env):
    """Core RoboEval environment which loads in common robot across all tasks.
    
    This environment provides a standardized interface for robot control tasks
    with support for various observation modes, action modes, and rendering options.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 1 / PHYSICS_DT,
    }

    _ENV_CAMERAS = ["external"]

    _MODEL_PATH: Path = WORLD_MODEL
    _PRESET_PATH: Optional[Path] = None
    _FLOOR = "floor"
    _YAM_BOOK_BODY_MASS = 0.8
    _YAM_BOOK_FRICTION = (0.8, 0.03, 0.001)

    DEFAULT_ROBOT = BimanualPanda
    
    SUCCESS_EVAL_METRICS = False
    USE_TEMPORAL_SUCCESS = False  # Enable temporal success checking

    RESET_ROBOT_POS = np.array([0, 0, 0])
    RESET_ROBOT_QUAT = np.array([1, 0, 0, 0])

    def __init__(
        self,
        action_mode: ActionMode,
        observation_config: ObservationConfig = ObservationConfig(),
        render_mode: Optional[str] = None,
        start_seed: Optional[int] = None,
        control_frequency: int = CONTROL_FREQUENCY_MAX,
        robot_cls: Optional[Type[Robot]] = None,
        **kwargs,
    ):
        """Initialize the environment.

        :param action_mode: The action mode of the robot. Use this to configure how
            you plan to control the robot. E.g. joint position, delta ee pose, etc.
        :param observation_config: Observations configuration. Use this to configure
            collected data.
        :param render_mode: The render mode for mujoco. Options are
            "human", "rgb_array" or "depth_array". If None, the default render mode
            will be used.
        :param start_seed: The seed to start the environment with. If None, a random
            seed will be used.
        :param control_frequency: Control loop frequency, 500 Hz by default.
        :param robot_cls: Environment robot class override.
        """
        self._parse_kwargs(kwargs or {})
        # Tracks physics simulation stability
        self._env_health = EnvHealth()
        # Caches results valid for one environment step
        self._step_cache = CallablesCache()

        # Frame counter + overlay cache for the composite render used by the
        # WebRTC/Agora stream. Overlays are the two wrist cameras drawn small
        # in the bottom corners; they only refresh every Nth frame to keep
        # cost down. See RoboEval@telearms for the original implementation.
        self._frame_count = 0
        self._cached_overlay1 = None
        self._cached_overlay2 = None

        self._observation_config = observation_config
        self.action_mode = action_mode

        if start_seed is None:
            start_seed = np.random.randint(2**32)
        if not isinstance(start_seed, int):
            raise ValueError("Expected start_seed to be an integer.")
        self._next_seed = start_seed
        self._current_seed = None

        assert CONTROL_FREQUENCY_MIN <= control_frequency <= CONTROL_FREQUENCY_MAX, (
            f"Control frequency must be in "
            f"{CONTROL_FREQUENCY_MIN}-{CONTROL_FREQUENCY_MAX} range."
        )
        self._control_frequency = control_frequency
        self._sub_steps_count = int(
            np.round(CONTROL_FREQUENCY_MAX / self._control_frequency)
        )

        self._spawns_info = []
        self._robots_info = {}

        if self._PRESET_PATH is not None:
            with open(self._PRESET_PATH) as fhandle:
                config = safe_load(fhandle)
                # Check if the robot position is given in the presets YAML file
                if "robot" in config:
                    for robot in config["robot"]:
                        self._robots_info[robot["name"]] = dict(
                            position=robot.get("position", [0.0, 0.0, 0.0]),
                            euler=robot.get("euler", [0.0, 0.0, 0.0]),
                        )
                # Check if any spawn regions are defined in the YAML file
                if "spawns" in config:
                    for spawn in config["spawns"]:
                        self._spawns_info.append(
                            dict(
                                name=spawn["name"],
                                size=spawn["size"],
                                position=spawn["position"],
                                euler=spawn["euler"],
                            )
                        )

        self._mojo = Mojo(str(self._MODEL_PATH), timestep=PHYSICS_DT)

        self._mojo.root_element.mjcf.size.memory = str(50_000_000)  # 50 MB

        self._robot = (robot_cls or self.DEFAULT_ROBOT)(self.action_mode, self._mojo)
        self._preset = Preset(self._mojo, self._PRESET_PATH)
        self._arena = Arena(self._mojo, self._PRESET_PATH)

        robot_model_names = [self._robot.config.model_name]
        robot_model_names.extend(getattr(self._robot, "preset_model_names", ()))
        robot_info = None
        for robot_model_name in robot_model_names:
            if robot_model_name in self._robots_info:
                robot_info = self._robots_info[robot_model_name]
                break

        if robot_info is not None:
            self.RESET_ROBOT_POS = np.array(robot_info["position"])
            self.RESET_ROBOT_QUAT = Rotation.from_euler(
                "xyz", robot_info["euler"], degrees=True
            ).as_quat(scalar_first=True)

        self._spawns: Dict[str, Optional[SpawnBoundary]] = {}
        for spawn_info in self._spawns_info:
            # Create the site for this spawn area
            spawn_site = Site.create(
                self._mojo,
                parent=None,
                size=np.array(spawn_info["size"]),
                position=np.array(spawn_info["position"]),
                quaternion=Rotation.from_euler(
                    "xyz", spawn_info["euler"], degrees=True
                ).as_quat(scalar_first=True),
                color=np.array([0, 1, 1, 0.25]),
                site_type=SiteType.BOX,
                group=3,
            )
            # Build the spawn boundary object
            spawn_boundary = SpawnBoundary(self._mojo, spawn_site)
            self._spawns[spawn_info["name"]] = spawn_boundary

        self._initialize_env()
        self._prepare_robot_operating_surfaces()
        self._floor = Geom.get(self._mojo, self._FLOOR)

        self.action_space: spaces.Box = self.action_mode.action_space(
            action_scale=self._sub_steps_count, seed=self._next_seed
        )
        self._action: np.ndarray = np.zeros_like(self.action_space.low)

        self.observation_space: spaces.Space = self.get_observation_space()

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]

        self.render_mode = render_mode

        # Validate cameras configuration
        if type(self._robot.config.cameras) is dict:
            available_cameras = set(self._ENV_CAMERAS + list(self._robot.config.cameras.keys()))
        elif type(self._robot.config.cameras) is list:
            available_cameras = set(self._ENV_CAMERAS + self._robot.config.cameras)
        else:
            raise ValueError("Invalid cameras configuration.")
        
        for camera_config in self._observation_config.cameras:
            assert camera_config.name in available_cameras

        # Mapping original camera names to full identifiers
        self._cameras_map = self._initialize_cameras()

        self.mujoco_renderer: Optional[RoboEvalRenderer] = None
        self.obs_renderers: Optional[dict[tuple[int, int], mujoco.Renderer]] = {}
        self._initialize_renderers()

        self.inverse_kinematics = GenericUpperBodyIK(self, self._robot.ik_config)

        # Initialize temporal success checker if enabled
        self._temporal_checker: Optional[TemporalSuccessChecker] = None
        if self.USE_TEMPORAL_SUCCESS:
            self._temporal_checker = TemporalSuccessChecker()
            self._initialize_temporal_goals()

        # Latches True once the task has succeeded at any point in the current
        # episode. Reset on env reset.
        self._ever_succeeded: bool = False

    def _parse_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Parse additional keyword arguments.
        
        This method can be overridden by subclasses to handle custom arguments.
        
        Args:
            kwargs: Dictionary of additional arguments
        """
        pass

    def _prepare_robot_operating_surfaces(self) -> None:
        """Adjust task fixture collisions for imported bimanual robots."""
        robot_model_name = self._robot.config.model_name
        if robot_model_name != "Bimanual YAM":
            return

        self._prepare_yam_book_physics()

    def _prepare_yam_book_physics(self) -> None:
        """Use realistic book physics for YAM shelf extraction tasks."""
        for body_id in range(self._mojo.model.nbody):
            body_name = mujoco.mj_id2name(
                self._mojo.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if not body_name or body_name.split("/")[-1] != "book":
                continue
            current_mass = float(self._mojo.model.body_mass[body_id])
            if current_mass <= 0.0:
                continue
            inertia_scale = self._YAM_BOOK_BODY_MASS / current_mass
            self._mojo.model.body_mass[body_id] = self._YAM_BOOK_BODY_MASS
            self._mojo.model.body_inertia[body_id] *= inertia_scale

        for geom_id in range(self._mojo.model.ngeom):
            geom_name = mujoco.mj_id2name(
                self._mojo.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if not geom_name or geom_name.split("/")[-1] != "book":
                continue
            self._mojo.model.geom_friction[geom_id] = self._YAM_BOOK_FRICTION
        self._mojo.physics.forward()

    @property
    def task_name(self) -> str:
        """Return the class name of the environment."""
        return self.__class__.__name__

    @property
    def _use_pixels(self) -> bool:
        """Return True if the environment uses pixel observations."""
        return len(self.observation_config.cameras) > 0

    @property
    def seed(self) -> Optional[int]:
        """Return the initial seed of the environment."""
        return self._current_seed

    @property
    def success(self) -> bool | Any:
        """Check if current step is successful."""
        if not self.SUCCESS_EVAL_METRICS:
            return bool(self._step_cache.get(self._success))
        else:
            return self._step_cache.get(self._success)

    @property
    def fail(self) -> bool:
        """Check if current step has failed."""
        return bool(self._step_cache.get(self._fail))

    @property
    def reward(self) -> float:
        """Get current step reward."""
        return float(self._step_cache.get(self._reward))

    @property
    def terminate(self) -> bool:
        """Get current step termination condition."""
        return bool(self.success or self.fail)

    @property
    def truncate(self) -> bool:
        """Get current step truncation condition."""
        return bool(not self.is_healthy)

    @property
    def is_healthy(self) -> bool:
        """Check if the simulation is currently healthy."""
        return bool(self._env_health.is_healthy)

    @property
    def observation_config(self) -> ObservationConfig:
        """Get the observation configuration."""
        return self._observation_config

    @property
    def control_frequency(self) -> int:
        """Return the control frequency of the environment."""
        return self._control_frequency

    @property
    def robot(self) -> Robot:
        """Get the robot instance."""
        return self._robot

    @property
    def floor(self) -> Geom:
        """Get environment floor geometry."""
        return self._floor

    @property
    def mojo(self) -> Mojo:
        """Get the Mojo instance."""
        return self._mojo

    @property
    def action(self) -> np.ndarray:
        """Get the last executed action."""
        return self._action.copy()

    def get_spawn_boundary(self, name: str) -> Optional[SpawnBoundary]:
        """Get the spawn boundary with the given name.
        
        Args:
            name: The name of the spawn boundary
            
        Returns:
            The spawn boundary object or None if not found
        """
        return self._spawns.get(name, None)

    def _initialize_renderers(self) -> None:
        """Initialize rendering objects for the environment."""
        self._close_renderers()
        self.mujoco_renderer: RoboEvalRenderer = RoboEvalRenderer(self._mojo)
        for camera_config in self._observation_config.cameras:
            resolution = camera_config.resolution
            if resolution in self.obs_renderers:
                continue
            self.obs_renderers[resolution] = mujoco.Renderer(
                self._mojo.model, resolution[0], resolution[1]
            )

    def _initialize_cameras(self) -> dict[str, tuple[int, Camera]]:
        """Initialize cameras for the environment.
        
        Returns:
            Dictionary mapping camera names to their IDs and Camera objects
        """
        cameras_map: dict[str, tuple[int, Camera]] = {}
        for camera_name in self._ENV_CAMERAS:
            camera: Camera = Camera.get(
                self._mojo, camera_name, self._mojo.root_element
            )
            cameras_map[camera_name] = (camera.id, camera)
        for robot_camera in self._robot.cameras:
            cameras_map[robot_camera.mjcf.name] = (
                robot_camera.id,
                robot_camera,
            )

        for camera_config in self._observation_config.cameras:
            _, camera = cameras_map[camera_config.name]
            if camera_config.pos is not None:
                camera.set_position(np.array(camera_config.pos))
            if camera_config.quat is not None:
                camera.set_quaternion(np.array(camera_config.quat))
        return cameras_map

    def _safe_close_renderer(self, renderer) -> None:
        """Safely close a renderer object, handling possible exceptions.
        
        Args:
            renderer: The renderer object to close
        """
        if renderer is not None:
            try:
                # In some cases renderer.close() will itself fail because underlying C resources are dead
                close_fn = getattr(renderer, 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                print(f"Warning: Renderer close failed: {e}")

    def _close_renderers(self) -> None:
        """Close all renderers used by the environment."""
        self._safe_close_renderer(self.mujoco_renderer)
        for renderer in self.obs_renderers.values():
            self._safe_close_renderer(renderer)
        self.mujoco_renderer = None
        self.obs_renderers.clear()

    def _initialize_env(self) -> None:
        """Initialize environment-specific components.
        
        This method can be overridden by subclasses to add task-specific items.
        """
        pass

    def get_observation_space(self) -> spaces.Space:
        """Get the observation space for the environment.
        
        Returns:
            A gymnasium Space object defining the observation space
        """
        obs_dict = {}
        if self._observation_config.proprioception:
            obs_dict = {
                "proprioception": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self._robot.qpos) + len(self._robot.qvel),),
                    dtype=np.float32,
                ),
                "proprioception_grippers": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.robot.qpos_grippers),),
                    dtype=np.float32,
                ),
            }
            if self.robot.floating_base:
                obs_dict.update(
                    {
                        "proprioception_floating_base": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(len(self.robot.floating_base.qpos),),
                            dtype=np.float32,
                        ),
                        "proprioception_floating_base_actions": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                len(self.robot.floating_base.get_accumulated_actions),
                            ),
                            dtype=np.float32,
                        ),
                    }
                )
        if self._use_pixels:
            for camera in self.observation_config.cameras:
                if camera.rgb:
                    obs_dict[f"rgb_{camera.name}"] = spaces.Box(
                        low=0, high=255, shape=(3, *camera.resolution), dtype=np.uint8
                    )
                if camera.depth:
                    obs_dict[f"depth_{camera.name}"] = spaces.Box(
                        low=0,
                        high=1,
                        shape=camera.resolution,
                        dtype=np.float32,
                    )
        if self._observation_config.privileged_information:
            obs_dict.update(self._get_task_privileged_obs_space())
        return spaces.Dict(obs_dict)

    def _get_task_privileged_obs_space(self) -> dict[str, Any]:
        """Get the task-specific privileged observation space.
        
        This method should be overridden by subclasses to define task-specific
        privileged observation spaces.
        
        Returns:
            Dictionary of privileged observation spaces
        """
        return {}

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get the current observation.
        
        Returns:
            Dictionary containing all observation components
        """
        obs = {}
        if self._observation_config.proprioception:
            obs |= self._get_proprioception_obs()
        if self._use_pixels:
            obs |= self._get_visual_obs()
        if self._observation_config.privileged_information:
            obs |= self._get_task_privileged_obs()
        return obs

    def _get_task_info(self) -> dict[str, Any]:
        """Get the task-specific info dictionary.
        
        This method should be overridden by subclasses to provide task-specific info.
        
        Returns:
            Dictionary of task-specific info
        """
        return {}

    def get_info(self) -> dict[str, Any]:
        """Get the info dictionary.
        
        Returns:
            Dictionary containing information about the current state
        """
        info = self._get_task_info()
        if self.USE_TEMPORAL_SUCCESS and self._temporal_checker is not None:
            info.update(self.get_temporal_summary())
        info.update({"task_success": float(self._ever_succeeded)})
        return info

    def _get_proprioception_obs(self) -> dict[str, Any]:
        """Get proprioception observations.
        
        Returns:
            Dictionary of proprioception observations
        """
        obs = {
            "proprioception": np.concatenate(
                [self._robot.qpos, self._robot.qvel]
            ).astype(np.float32),
            "proprioception_grippers": np.array(self.robot.qpos_grippers).astype(
                np.float32
            ),
        }
        if self.robot.floating_base:
            obs["proprioception_floating_base"] = np.array(
                self.robot.floating_base.qpos
            ).astype(np.float32)
            obs["proprioception_floating_base_actions"] = np.array(
                self.robot.floating_base.get_accumulated_actions
            ).astype(np.float32)
        return obs

    def _get_visual_obs(self) -> dict[str, Any]:
        """Get visual observations from cameras.
        
        Returns:
            Dictionary of visual observations
        """
        obs = {}
        for camera_config in self._observation_config.cameras:
            obs_renderer = self.obs_renderers[camera_config.resolution]
            obs_renderer.update_scene(
                self._mojo.data, self._cameras_map[camera_config.name][0]
            )
            if camera_config.rgb:
                rgb = obs_renderer.render()
                obs[f"rgb_{camera_config.name}"] = np.moveaxis(rgb, -1, 0)
            if camera_config.depth:
                obs_renderer.enable_depth_rendering()
                obs[f"depth_{camera_config.name}"] = obs_renderer.render()
                obs_renderer.disable_depth_rendering()
        return obs

    def _get_task_privileged_obs(self) -> dict[str, Any]:
        """Get task-specific privileged observations.
        
        This method should be overridden by subclasses to provide task-specific
        privileged observations.
        
        Returns:
            Dictionary of privileged observations
        """
        return {}

    def _update_seed(self, override_seed=None) -> None:
        """Update the seed for the environment.

        Args:
            override_seed: If not None, the next seed will be set to this value.
        """
        if override_seed is not None:
            if not isinstance(override_seed, int):
                logging.warning(
                    "Expected override_seed to be an integer. Casting to int."
                )
                override_seed = int(override_seed)
            self._next_seed = override_seed
            self.action_space = self.action_mode.action_space(
                action_scale=self._sub_steps_count, seed=override_seed
            )
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        self._next_seed = np.random.randint(2**32)
        np.random.seed(self._current_seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment.

        Args:
            seed: If not None, the environment will be reset with this seed
            options: Additional information to specify how the environment is reset
        
        Returns:
            Tuple of (observation, info)
        """
        self._env_health.reset()
        self._update_seed(override_seed=seed)
        self._mojo.physics.reset()
        self._action = np.zeros_like(self._action)
        self._robot.set_pose(self.RESET_ROBOT_POS, self.RESET_ROBOT_QUAT)
        self._robot.reset_pose()
        self._ever_succeeded = False

        # Reset temporal checker if enabled
        if self.USE_TEMPORAL_SUCCESS and self._temporal_checker is not None:
            self._temporal_checker.reset()

        self._on_reset()
        return self.get_observation(), self.get_info()

    def _on_reset(self) -> None:
        """Custom environment reset behavior.
        
        This method should be overridden by subclasses to provide task-specific
        reset behavior.
        """
        pass

    def _on_step(self) -> None:
        """Custom environment behavior after stepping.

        This method should be overridden by subclasses to provide task-specific
        step behavior.

        Note: The temporal success checker is automatically updated here if enabled.
        Subclasses that override this method should call super()._on_step().
        """
        if self.USE_TEMPORAL_SUCCESS and self._temporal_checker is not None:
            self._temporal_checker.update()

    @property
    def stream_camera_id(self) -> int:
        """Camera id used as the main frame of the streamed composite.

        This is the third-person POV the operator solves the task through, and
        the camera view-mode commands are expected to move. Exposed as a single
        accessor so the renderer, the camera controller and the camera-track
        recorder can never disagree about which camera that is.

        Robot cameras are registered under ``robot_camera.mjcf.name``, which
        carries the MJCF attachment namespace (e.g. "panda nohand_left/..."),
        so this exact "top_cam" lookup only hits for robots whose overhead cam
        is registered unnamespaced; otherwise it falls back to world.xml's
        "external". Either way the result is a scene camera, never a wrist cam.
        """
        return self._cameras_map.get(
            "top_cam", self._cameras_map.get("external", (0, None))
        )[0]

    def render(self):
        """Render the external view with the two wrist-camera overlays.

        Mirrors RoboEval@telearms's composite shape, but uses the names the
        playground scene actually exposes: main = "external" (the world-fixed
        operator camera from world.xml, equivalent to RoboEval's head frame),
        overlays = the left/right wrist cameras from the Panda robot. The
        overlays are refreshed every 5 frames to cap the extra render cost.

        The composite only makes sense for the rgb_array stream used by the
        Telearms/WebRTC path. For the interactive human viewer (keyboard/VR
        teleop) human-mode render() returns None, so the composite path would
        throw every frame and re-render the window 2-4x with camera switches —
        rendering it nearly unusable. Fall back to a single plain render.
        """
        if self.render_mode != "rgb_array":
            return self.mujoco_renderer.render(self.render_mode)

        self._frame_count += 1
        update_overlays = (
            self._frame_count % 5 == 0 or self._cached_overlay1 is None
        )

        main_id = self.stream_camera_id
        # Pick any camera whose registered name ends with "left_wrist" /
        # "right_wrist" — the Panda robot registers them namespaced
        # (e.g. "panda nohand_left/left_wrist").
        def _find(*suffixes):
            for name, (cid, _) in self._cameras_map.items():
                if name.endswith(suffixes):
                    return cid
            return None
        overlay1_id = _find("left_wrist", "left_cam")
        overlay2_id = _find("right_wrist", "right_cam")

        try:
            main_frame = self.mujoco_renderer.render(self.render_mode, camera_id=main_id)

            if update_overlays and overlay1_id is not None and overlay2_id is not None:
                overlay1 = self.mujoco_renderer.render(self.render_mode, camera_id=overlay1_id)
                overlay2 = self.mujoco_renderer.render(self.render_mode, camera_id=overlay2_id)
                self._cached_overlay1 = self._resize_frame(overlay1, width=320, height=180)
                self._cached_overlay2 = self._resize_frame(overlay2, width=320, height=180)

            if self._cached_overlay1 is None or self._cached_overlay2 is None:
                return main_frame

            main_h, main_w, _ = main_frame.shape
            composite = main_frame.copy()
            y1, y2 = main_h - 180 - 10, main_h - 10
            composite[y1:y2, 10:10 + 320] = self._cached_overlay1
            composite[y1:y2, main_w - 320 - 10:main_w - 10] = self._cached_overlay2
            return composite
        except Exception as e:
            logging.warning("Composite render failed, falling back to main cam: %s", e)
            return self.mujoco_renderer.render(self.render_mode, camera_id=main_id)

    @staticmethod
    def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize an RGB frame without importing optional image libraries at startup."""
        src_h, src_w = frame.shape[:2]
        y_idx = np.linspace(0, src_h - 1, height).astype(np.intp)
        x_idx = np.linspace(0, src_w - 1, width).astype(np.intp)
        return np.asarray(frame)[y_idx][:, x_idx]

    def step(
        self, action: Optional[np.ndarray] = None, fast: bool = False
    ) -> tuple[Any, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action: Action to take. If None, only physics simulation will be applied
                without any robot actions (useful for letting objects fall due to gravity).
            fast: If True, perform the environment step without processing observations
                and return default values. Useful when performance is crucial,
                but observations are not required, e.g., demo collection in VR.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if self.mujoco_renderer:
            if hasattr(self.mujoco_renderer.viewer, "request_reset"):
                if self.mujoco_renderer.viewer.request_reset:
                    self.mujoco_renderer.viewer.request_reset = False
                    self.reset()
            if hasattr(self.mujoco_renderer.viewer, "task_success"):
                self.mujoco_renderer.viewer.task_success = self._ever_succeeded
            if hasattr(self.mujoco_renderer.viewer, "task_fail"):
                self.mujoco_renderer.viewer.task_fail = self.fail
        self._step_cache.clean()
        
        if action is None:
            # Physics-only step (no robot action)
            with self._env_health.track():
                for _ in range(self._sub_steps_count):
                    self._mojo.step()
                    mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)
            self._action = np.zeros_like(self.action_space.low)
        else:
            # Normal step with robot action
            self._step_mujoco_simulation(action)
            self._action = action
            
        with self._env_health.track():
            self._on_step()

        # Latch success so a transient YES is preserved for the rest of
        # the episode (info["task_success"] / viewer overlay).
        if not self._ever_succeeded and bool(self.success):
            self._ever_succeeded = True

        if fast:
            return {}, 0, False, False, {}
        else:
            return (
                self.get_observation(),
                self.reward,
                self.terminate,
                self.truncate,
                self.get_info(),
            )

    def _step_mujoco_simulation(self, action: np.ndarray) -> None:
        """Step the MuJoCo simulation with the given action.
        
        Args:
            action: Action to apply
            
        Raises:
            ValueError: If action shape doesn't match the action space
        """
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape mismatch: "
                f"expected {self.action_space.shape}, but got {action.shape}."
            )
        if np.any(action < self.action_space.low) or np.any(
            action > self.action_space.high
        ):
            clipped_action = np.clip(
                action, self.action_space.low, self.action_space.high
            )
            print(
                f"Action {action} is out of the action space bounds. "
                f"Overhead: {action - clipped_action}"
            )

        with self._env_health.track():
            for i in range(self._sub_steps_count):
                if i == 0:
                    self.action_mode.step(action, self.inverse_kinematics)
                else:
                    self.action_mode._step_physics()
                mujoco.mj_rnePostConstraint(self._mojo.model, self._mojo.data)

    def _success(self):
        """Check if the episode is successful.
        
        This method should be overridden by subclasses to define success criteria.
        
        Returns:
            True if successful, False otherwise
        """
        return False

    def _initialize_temporal_goals(self) -> None:
        """Initialize temporal sub-goals for sequential task checking.

        This method should be overridden by subclasses that use temporal success
        checking. Use self._add_temporal_goal() to define the sequence.

        Example:
            def _initialize_temporal_goals(self):
                self._add_temporal_goal("pick_blue", lambda: self.blue_cube.is_grasped())
                self._add_temporal_goal("pick_red", lambda: self.red_cube.is_grasped())
                self._add_temporal_goal("stack", lambda: self.is_stacked(red, blue))
        """
        pass

    def _add_temporal_goal(
        self,
        name: str,
        check_fn,
        description: str = "",
        is_optional: bool = False,
        timeout_steps: Optional[int] = None,
        **metadata
    ) -> None:
        """Add a sub-goal to the temporal success checker.

        Args:
            name: Unique identifier for the sub-goal
            check_fn: Callable that returns True when sub-goal is satisfied
            description: Human-readable description
            is_optional: Whether this sub-goal can be skipped
            timeout_steps: Maximum steps allowed for completion
            **metadata: Additional tracking information
        """
        if self._temporal_checker is not None:
            self._temporal_checker.add_sub_goal(
                name=name,
                check_fn=check_fn,
                description=description,
                is_optional=is_optional,
                timeout_steps=timeout_steps,
                **metadata
            )

    def get_temporal_progress(self) -> tuple[int, int]:
        """Get temporal success progress (completed, total)."""
        if self._temporal_checker is not None:
            return self._temporal_checker.get_progress()
        return (0, 0)

    def get_temporal_metrics(self) -> dict:
        """Get detailed temporal execution metrics."""
        if self._temporal_checker is not None:
            return self._temporal_checker.get_metrics()
        return {}

    def print_temporal_status(self) -> None:
        """Print current temporal success checker status."""
        if self._temporal_checker is not None:
            self._temporal_checker.print_status()

    def get_temporal_summary(self) -> dict:
        """Get a summary of temporal execution state."""
        if self._temporal_checker is not None:
            return self._temporal_checker.get_summary()
        return {}

    def _fail(self) -> bool:
        """Check if the episode has failed.

        Returns:
            True if failed, False otherwise
        """
        try:
            return (
                np.linalg.norm(self._robot.pelvis.get_position()) > MAX_DISTANCE_FROM_ORIGIN
            )
        except:
            return True

    def _reward(self) -> float:
        """Get current episode reward.
        
        Returns:
            Reward value
        """
        return float(self.success) * SPARSE_REWARD_FACTOR

    def close(self):
        """Close environment and free resources."""
        self._close_renderers()
