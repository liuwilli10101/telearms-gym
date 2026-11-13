"""Telearms teleoperation driver.

Bridges the Agora media plane (video out + RTM input) and the JSON-RPC
WebSocket control plane (TeleArmsWSClient) to RoboEval's simulation.

Scope: start/stop/reset commands, H.264 video stream over Agora,
keyboard + gamepad input from the frontend over Agora RTM, and S3
telemetry upload of the recorded demo + metrics JSON. Joy-Con IMU input
and temporal success checking remain out of scope.
"""

# Standard library
import asyncio
import atexit
import json
import os
import queue as pyqueue
import signal
import threading
import time
import traceback
import uuid
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type, Union

# Third-party
import mujoco
import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from gymnasium.core import ActType
from pyquaternion import Quaternion

load_dotenv()

# GStreamer — imported after load_dotenv so DISPLAY/MUJOCO_GL env is set
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402

Gst.init(None)

# teleop_sdk is the Agora RTC/RTM wrapper shipped as a wheel under telearms/
import teleop_sdk  # noqa: E402

# Local
from roboeval.data_collection.camera_controller import CameraController, WristCameraController  # noqa: E402
from roboeval.data_collection.camera_controller import (  # noqa: E402
    STEP_AZIM as CAM_STEP_AZIM,
    STEP_DIST as CAM_STEP_DIST,
    STEP_ELEV as CAM_STEP_ELEV,
    camera_frame_for_id,
    camera_name_for_id,
    camera_pose_snapshot,
    stream_camera_id,
)
from roboeval.data_collection.camera_track import (  # noqa: E402
    CameraTrackRecorder,
    move_count as camera_move_count,
)
from roboeval.data_collection.keyboard_input import (  # noqa: E402
    Countdown,
    KeyboardTeleop,
)
from roboeval.data_collection.webrtc_config import (  # noqa: E402
    TaskConfig,
    get_random_task,
    get_task_by_name,
)
from roboeval.demonstrations.demo import TERMINATION_STEPS  # noqa: E402
from roboeval.demonstrations.const import STATE_QPOS_KEY, STATE_QVEL_KEY  # noqa: E402
from roboeval.roboeval_env import RoboEvalEnv  # noqa: E402
from telearms.fixed_rate_pacer import FixedRatePacer  # noqa: E402
from telearms.main_thread_dispatcher import MainThreadDispatcher  # noqa: E402
from telearms.ws_client import InstanceStatus, TeleArmsWSClient  # noqa: E402
from tools.shared.utils import ENVIRONMENTS  # noqa: E402

# S3 upload for telemetry — keep optional so a missing boto3 doesn't block
# startup; _upload_telemetry_to_s3 short-circuits with a warning instead.
try:
    import boto3  # noqa: E402
    from botocore.exceptions import ClientError  # noqa: E402,F401
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not installed, S3 uploads will be disabled")


# ==================== Environment config ====================
WS_SERVER_URL = os.environ.get("WS_SERVER_URL", "ws://localhost:8080")
TELEARMS_BACKEND_URL = os.environ.get("TELEARMS_BACKEND_URL", "http://localhost:3000")
TELEARMS_BACKEND_API_KEY = os.environ.get("TELEARMS_BACKEND_API_KEY", "")

# S3 telemetry upload — mirrors RoboEval. Bucket/region/toggle come from env;
# credentials come from the EC2 instance role or AWS_*/IAM env vars (no
# ~/.aws config is required on staging).
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

# A recording is written to local disk *before* it is uploaded, so a full disk
# destroys the episode and its upload together, leaving no trace anywhere. This
# is what caused the 2026-08-29 scoring outage (BIT-2456): ~2,200 tasks were
# lost across four days with nothing but a swallowed print to show for it.
MIN_FREE_MB_WARN = int(os.environ.get("MIN_FREE_MB_WARN", "5000"))


def _free_mb(path: str) -> Optional[int]:
    """Free megabytes on the filesystem holding `path`, or None if unknown."""
    try:
        st = os.statvfs(path)
        return int(st.f_bavail * st.f_frsize / (1024 * 1024))
    except Exception:
        return None
# NOTE: this default is the *staging* bucket while the backend reads `telearms`.
# An unset value therefore sends telemetry somewhere nothing will ever score it.
# update_and_run.sh refuses to start when it is unset; this warns for any other
# entrypoint.
TELEARMS_S3_BUCKET = os.environ.get("TELEARMS_AWS_S3_BUCKET", "telearms-staging")
if not os.environ.get("TELEARMS_AWS_S3_BUCKET"):
    print(
        "⚠️ TELEARMS_AWS_S3_BUCKET is unset — defaulting to "
        f"'{TELEARMS_S3_BUCKET}'. Telemetry will NOT be scored if the backend "
        "reads a different bucket."
    )
S3_UPLOAD_ENABLED = os.environ.get("S3_UPLOAD_ENABLED", "true").lower() == "true"

WIDTH, HEIGHT = 1280, 720
VIDEO_FPS = 30
CONTROL_HZ = 30
RAW_FRAME_BYTES = WIDTH * HEIGHT * 3

PIPELINE = (
    f'appsrc name=source is-live=true block=false format=TIME '
    f'max-buffers=1 max-bytes={RAW_FRAME_BYTES} leaky-type=downstream '
    f'caps=video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate={VIDEO_FPS}/1 ! '
    f'videoconvert ! '
    f'x264enc speed-preset=ultrafast bframes=0 tune=zerolatency byte-stream=true key-int-max=30 ! '
    f'video/x-h264,stream-format=byte-stream ! '
    f'appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true'
)

# The frontend advertises a different key scheme than KeyboardTeleop's
# (z/x/n/m for grippers, escape/delete for reset). Recording lifecycle is
# owned by backend start/stop commands, not keys, so no frontend keys need
# to be filtered out here.
_IGNORED_FRONTEND_KEYS: set = set()

# Live-editable camera override. Edit this YAML while the sim is running
# and TelearmsTeleop picks up the change on the next frame (see
# _maybe_apply_cam_override). Schema (all fields optional):
#   pos:    [x, y, z]            world-space camera position
#   target: [x, y, z]            point the camera should look at
#                                (derives quat from pos + target)
#   quat:   [w, x, y, z]         direct quaternion override (alt to target)
#   fovy:   <degrees>            vertical FOV
_REPO_ROOT = Path(__file__).resolve().parents[2]
CAM_OVERRIDE_PATH = _REPO_ROOT / "telearms" / "cam_override.yaml"

# App version stamped into the telemetry JSON. Bumped by hand on each release
# so demos on S3 can be traced back to the playground build that produced them.
APP_VERSION = "1.0.0"

# Module-level ref so atexit/signal handlers can tear the instance down.
_active_teleop_instance: Optional["TelearmsTeleop"] = None

# Busy flag for the co-resident verification worker (verification/): touched
# while a task is live so the worker pauses claiming and never contends with
# gameplay. Best-effort on both sides — a stat/unlink hiccup must never affect
# the task itself (the worker also treats very old flags as stale).
BUSY_FLAG_PATH = Path(os.environ.get("PLAYGROUND_BUSY_FLAG", "/tmp/playground-busy"))


def _set_busy_flag() -> None:
    try:
        BUSY_FLAG_PATH.touch(exist_ok=True)
    except OSError as e:
        print(f"[telearms] busy-flag touch failed ({BUSY_FLAG_PATH}): {e}")


def _clear_busy_flag() -> None:
    try:
        BUSY_FLAG_PATH.unlink(missing_ok=True)
    except OSError as e:
        print(f"[telearms] busy-flag remove failed ({BUSY_FLAG_PATH}): {e}")


@dataclass
class SimulationMetrics:
    """Real-time simulation metrics from RoboEval MetricRolloutEval.

    Mirror of RoboEval's SimulationMetrics so the telemetry JSON shape on S3
    stays identical. Defaults are zero/empty; _get_simulation_metrics fills
    whichever fields the running env exposes (everything is hasattr-guarded).
    """
    # Task progress
    task_name: str = ""
    task_success: bool = False
    task_stages_reached: dict = None

    # Timing
    elapsed_seconds: float = 0.0
    frame_count: int = 0

    # Bimanual coordination
    arm_velocity_sync: float = 0.0
    gripper_vertical_sync: float = 0.0

    # Safety metrics
    slip_count: int = 0
    env_collision_count: int = 0
    self_collision_count: int = 0

    # Motion quality (lower is smoother)
    avg_cartesian_jerk_left: float = 0.0
    avg_cartesian_jerk_right: float = 0.0
    avg_joint_jerk_left: float = 0.0
    avg_joint_jerk_right: float = 0.0

    # Path efficiency
    cartesian_path_length_left: float = 0.0
    cartesian_path_length_right: float = 0.0
    orientation_path_length_left: float = 0.0
    orientation_path_length_right: float = 0.0

    # Target-specific
    target_distances: dict = None
    pose_error: float = 0.0

    def __post_init__(self):
        if self.task_stages_reached is None:
            self.task_stages_reached = {}
        if self.target_distances is None:
            self.target_distances = {}

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "task_success": self.task_success,
            "task_stages_reached": self.task_stages_reached,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "frame_count": self.frame_count,
            "arm_velocity_sync": round(self.arm_velocity_sync, 4),
            "gripper_vertical_sync": round(self.gripper_vertical_sync, 4),
            "slip_count": self.slip_count,
            "env_collision_count": self.env_collision_count,
            "self_collision_count": self.self_collision_count,
            "avg_cartesian_jerk": {
                "left": round(self.avg_cartesian_jerk_left, 4),
                "right": round(self.avg_cartesian_jerk_right, 4),
            },
            "avg_joint_jerk": {
                "left": round(self.avg_joint_jerk_left, 4),
                "right": round(self.avg_joint_jerk_right, 4),
            },
            "cartesian_path_length": {
                "left": round(self.cartesian_path_length_left, 4),
                "right": round(self.cartesian_path_length_right, 4),
            },
            "orientation_path_length": {
                "left": round(self.orientation_path_length_left, 4),
                "right": round(self.orientation_path_length_right, 4),
            },
            "target_distances": {k: round(v, 4) for k, v in self.target_distances.items()},
            "pose_error": round(self.pose_error, 4),
        }


@dataclass
class TelearmsTeleopStats:
    is_recording: bool = False
    time: float = 0.0
    reward: float = 0.0
    demos_counter: int = 0


class TelearmsTeleop(KeyboardTeleop):
    """Telearms (Agora + WS) teleoperation driver for RoboEval envs.

    Reuses KeyboardTeleop's arm/gripper/IK logic via inheritance; replaces
    the local pynput listener with RTM-based keyboard/gamepad events and
    streams frames over Agora RTC instead of rendering to a local window.
    """

    # Physics steps applied per control action. Previously inherited from
    # KeyboardTeleop (=3); commit 707af9f removed that class constant from the
    # base while TelearmsTeleop.run() still references self.STEPS_COUNT_FACTOR,
    # so it is redeclared here to preserve the prior control cadence.
    STEPS_COUNT_FACTOR = 3

    # Gamepad axes/buttons come straight from the browser Gamepad API.
    GAMEPAD_MOVEMENT_SCALE = 0.008
    GAMEPAD_ROTATION_SCALE = 0.008
    GAMEPAD_DEADZONE = 0.15
    ENVIRONMENT_SETUP_TIMEOUT_SECONDS = 180

    # Keyboard layout the frontend UI advertises (Keyboard Controls panel).
    # Overrides KeyboardTeleop's older WASD/z-c + v/b + mode-toggle scheme.
    LEFT_ARM_POSITION_KEYS = {
        'forward': 'w', 'back': 's',
        'left': 'a', 'right': 'd',
        'up': 'q', 'down': 'e',
    }
    RIGHT_ARM_POSITION_KEYS = {
        'forward': 'i', 'back': 'k',
        'left': 'j', 'right': 'l',
        'up': 'u', 'down': 'o',
    }
    # Yaw + pitch + roll — matches the UI's Rotate Arm row.
    # Left: 1/4 yaw, 2/3 pitch, R/F roll. Right: 7/0 yaw, 8/9 pitch, T/G roll.
    LEFT_ARM_ORIENTATION_KEYS = {
        'yaw_left': '4', 'yaw_right': '1',
        'pitch_up': '2', 'pitch_down': '3',
        'roll_cw': 'r', 'roll_ccw': 'f',
    }
    RIGHT_ARM_ORIENTATION_KEYS = {
        'yaw_left': '0', 'yaw_right': '7',
        'pitch_up': '8', 'pitch_down': '9',
        'roll_cw': 't', 'roll_ccw': 'g',
    }
    # Split open/close so the UI's Z/X and N/M buttons both map cleanly.
    GRIPPER_KEYS = {
        'left_open': 'z', 'left_close': 'x',
        'right_open': 'n', 'right_close': 'm',
    }
    # pynput sends 'esc'; browsers send 'escape'. Accept both (same for
    # delete/backspace) so the UI's Reset Arm buttons work either way.
    LEFT_RESET_KEYS = {'escape', 'esc'}
    RIGHT_RESET_KEYS = {'delete', 'backspace'}

    def __init__(
        self,
        env_cls: Optional[Type[RoboEvalEnv]],
        action_mode,
        resolution: tuple[int, int] = (WIDTH, HEIGHT),
        demo_directory: Optional[Union[str, Path]] = None,
        robot_cls: Optional[Type[Any]] = None,
        config: Optional[dict] = None,
        port: int = 8080,
    ):
        # NOTE: intentionally do NOT call super().__init__() — the parent
        # starts a pynput listener and opens a human-visible renderer. We
        # replicate the parts we need below and skip those two.
        from roboeval.demonstrations.demo_recorder import DemoRecorder

        self.config = config
        self._width, self._height = resolution
        self._demo_recorder = DemoRecorder(demo_directory)
        self.port = port
        self._env_lock = threading.RLock()
        # GStreamer sends encoded frames on its streaming thread while WS
        # worker threads start, stop, and rebuild the native Agora SDK. Native
        # calls must never overlap: send_video() racing deinit() segfaults.
        self._sdk_lock = threading.Lock()
        # TelearmsWSClient invokes start/preload handlers via asyncio.to_thread.
        # Renderer destruction/recreation is thread-affine, so those handlers
        # dispatch environment changes back to the simulation thread.
        self._main_thread_dispatcher = MainThreadDispatcher()

        self._action_mode = action_mode
        self._robot_cls = robot_cls

        # Resolve initial task
        self._current_task = self._resolve_initial_task(env_cls)
        env_cls = ENVIRONMENTS.get(self._current_task.name)
        print(f"[telearms] Loading environment: {self._current_task.name}")

        # Build env with WebRTC wrapper (rgb_array so render returns frames)
        webrtc_env_cls = self._create_webrtc_env(env_cls)
        self._env = webrtc_env_cls(
            render_mode="rgb_array", action_mode=action_mode, robot_cls=robot_cls
        )
        self._env.mojo.model.vis.global_.offwidth = self._width
        self._env.mojo.model.vis.global_.offheight = self._height
        self._env.reset()

        # State
        self._stats = TelearmsTeleopStats()
        self._stop_countdown: Optional[Countdown] = None
        self._current_action = np.zeros(self._env.action_space.shape)
        self._ik = self._env.inverse_kinematics

        # Input state — no pynput listener; keys come via RTM
        self.pressed_keys: set = set()

        # Gamepad state
        self.gamepad_axes = [0.0, 0.0, 0.0, 0.0]
        self.gamepad_buttons = [0.0] * 17
        self._prev_gamepad_buttons = [0.0] * 17
        self._gamepad_active = False
        # Input methods that actually drove motion during the current
        # recording (e.g. {"keyboard", "gamepad"}). Reset on each
        # _start_recording_for_task; stamped into the telemetry JSON.
        self._input_methods_used: set = set()

        # Control settings
        self.control_mode = self.MODE_POSITION
        self.toggle_gripper_mode = True

        self._setup_grippers()
        self._init_arm_state()
        self._save_initial_positions()

        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        # Edge-detect state for every key whose handler uses "triggered once
        # per press" semantics (grippers + reset keys).
        self._prev_key_state = {
            self.GRIPPER_KEYS["left_open"]: False,
            self.GRIPPER_KEYS["left_close"]: False,
            self.GRIPPER_KEYS["right_open"]: False,
            self.GRIPPER_KEYS["right_close"]: False,
        }
        for key in self.LEFT_RESET_KEYS | self.RIGHT_RESET_KEYS:
            self._prev_key_state[key] = False

        # Scene-control channel (reset_scene / change_task) coming from RTM
        self.scene_control_queue: pyqueue.Queue = pyqueue.Queue()
        self.scene_control_handlers = {
            "reset_scene": self._handle_reset_scene,
            "change_task": self._handle_change_task,
        }

        # Task lifecycle state
        self._env_start_recording_event = threading.Event()
        self._task_start_time: Optional[float] = None
        self._frames_recorded = 0
        self._shutdown_done = False
        # A mission start supplies the fresh Agora tokens that activate video.
        self._video_paused = True

        # Live camera override (file watched; 0 means "never applied yet")
        self._cam_override_mtime: float = 0.0

        # View-mode camera state. The controller holds spherical coords for
        # the 'external' cam; _on_message_received queues camera_view RTM
        # commands here and _drain_camera_view_queue applies them under the
        # env lock on the next tick. In-memory only — restart returns to
        # the world.xml baked pose.
        self._cam_controller = CameraController()
        self.camera_view_queue: pyqueue.Queue = pyqueue.Queue()

        # Records every POV camera change during a task so replay can retrace
        # it. Armed in _start_recording_for_task, drained into the demo metadata
        # and the telemetry JSON in _save_recording.
        self._camera_track_recorder = CameraTrackRecorder()

        # Per-side wrist-cam offset state for the gripper-debug panel.
        # Same drain-on-tick pattern as camera_view; offsets reset on
        # env rebuild so a mission switch doesn't leave them attached
        # to a different baked pose.
        self._wrist_cam_controller = WristCameraController()
        self.wrist_camera_view_queue: pyqueue.Queue = pyqueue.Queue()

        # Agora init — callback pair is registered now; tokens are set later
        # in _handle_start_command when the backend tells us which channel.
        teleop_sdk.init(
            message_received_callback=self._on_message_received,
            on_event_callback=self._on_event_cb,
        )

        # GStreamer pipeline
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsrc = self.pipeline.get_by_name("source")
        # Let GStreamer stamp PTS on every pushed buffer from the pipeline's
        # running clock. Without this, x264enc sees buffers with pts=NONE/0
        # and can emit NAL units the decoder rejects as "no reference frame",
        # which most decoders render as a solid green field.
        self.appsrc.set_property("do-timestamp", True)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.connect("new-sample", self._on_new_sample)

        # WS client
        self._ws_client: Optional[TeleArmsWSClient] = None
        self._init_ws_client()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _resolve_initial_task(self, env_cls: Optional[Type[RoboEvalEnv]]) -> TaskConfig:
        """Pick the TaskConfig to boot with, preferring the user-specified env_cls."""
        if env_cls is None:
            return get_random_task()

        for name, cls in ENVIRONMENTS.items():
            if cls == env_cls:
                task_config = get_task_by_name(name)
                if task_config is None:
                    # Env exists but no TaskConfig registered — synthesize one
                    task_config = TaskConfig(name=name, description=f"{name} task", enabled=True)
                return task_config

        print(f"[telearms] Warning: env_cls {env_cls} not in ENVIRONMENTS, falling back to random task")
        return get_random_task()

    def _save_initial_positions(self):
        """Capture pelvis-relative arm targets for the reset-arm controls.

        ``_calculate_control`` adds the pelvis position to every arm target, so
        reset targets must use the same pelvis-relative frame as
        ``_init_arm_state``. Saving ``gripper.wrist_position`` here used world
        coordinates; the countdown's repeated arm-reset commands then added the
        YAM pelvis transform a second time and sent both hands far from home.
        """
        self.left_arm_initial_position = (
            self.left_arm_target.copy() if self.gripper_l else np.zeros(3)
        )
        self.right_arm_initial_position = (
            self.right_arm_target.copy() if self.gripper_r else np.zeros(3)
        )
        self.left_arm_initial_orientation = (
            Quaternion(self.gripper_l.wrist_orientation) if self.gripper_l else Quaternion()
        )
        self.right_arm_initial_orientation = (
            Quaternion(self.gripper_r.wrist_orientation) if self.gripper_r else Quaternion()
        )

    def _reset_arms_to_initial(self):
        """Restore both arms to the pose captured at boot."""
        if self.gripper_l:
            self.left_arm_target = self.left_arm_initial_position.copy()
            self.left_arm_orientation = Quaternion(self.left_arm_initial_orientation)
        if self.gripper_r:
            self.right_arm_target = self.right_arm_initial_position.copy()
            self.right_arm_orientation = Quaternion(self.right_arm_initial_orientation)

    def _init_ws_client(self):
        """Wire up TeleArmsWSClient and register command handlers."""
        instance_id = os.environ.get("INSTANCE_ID", f"roboplayground-{uuid.uuid4().hex[:8]}")
        self._ws_client = TeleArmsWSClient(
            ws_url=WS_SERVER_URL,
            instance_id=instance_id,
            capabilities=["simulation", "recording", "teleop"],
            on_connected=self._on_ws_connected,
            on_disconnected=self._on_ws_disconnected,
        )
        self._ws_client.register_command_handler("start", self._handle_start_command)
        self._ws_client.register_command_handler("stop", self._handle_stop_command)
        self._ws_client.register_command_handler("reset", self._handle_reset_command)
        self._ws_client.register_command_handler("preload", self._handle_preload_command)

        self._ws_client.start()
        if self._ws_client.wait_for_connection(timeout=10.0):
            print(f"[telearms] Connected to WS server as {instance_id}")
        else:
            print("[telearms] Warning: could not connect to WS server (will keep retrying)")

        global _active_teleop_instance
        _active_teleop_instance = self
        atexit.register(self._cleanup_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ------------------------------------------------------------------
    # WebRTC env wrapper
    # ------------------------------------------------------------------
    def _create_webrtc_env(self, env_cls: Type[RoboEvalEnv]) -> Type[RoboEvalEnv]:
        """Subclass env_cls so step() logs to demo_recorder and render() pushes H.264."""
        get_demo_recorder = lambda: self._demo_recorder
        get_on_render = lambda: self._on_render

        class TelearmsRoboEvalEnv(env_cls, ABC):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._demo_recorder = get_demo_recorder()
                self._on_render = get_on_render()

            def step(self, action: ActType, fast: bool = True):
                super().step(action, fast)
                # Capture the full MuJoCo state (robot + object freejoints) so
                # the demo can be replayed by restoring state instead of
                # re-applying actions open-loop (which does not reproduce
                # contact-rich grasps). Cheap array copies; keeps fast=True.
                info = {
                    STATE_QPOS_KEY: self.mojo.data.qpos.copy(),
                    STATE_QVEL_KEY: self.mojo.data.qvel.copy(),
                }
                timestep = ({}, self.reward, False, False, info)
                self._demo_recorder.add_timestep(timestep, action)
                return timestep

            @property
            def task_name(self) -> str:
                return self.__class__.__base__.__name__

        return TelearmsRoboEvalEnv

    # ------------------------------------------------------------------
    # WS command handlers
    # ------------------------------------------------------------------
    def _fetch_agora_token(self, task_id: str) -> Optional[dict]:
        url = f"{TELEARMS_BACKEND_URL}/api/v1/internal/tasks/{task_id}/agora_token"
        headers = {
            "Content-Type": "application/json",
            "X-Internal-Api-Key": TELEARMS_BACKEND_API_KEY,
        }
        try:
            print(f"[telearms] Fetching Agora token: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            print(f"[telearms] Token fetch failed: HTTP {resp.status_code} — {resp.text}")
            return None
        except requests.RequestException as e:
            print(f"[telearms] Token fetch error: {e}")
            return None

    def _handle_start_command(self, params: dict) -> dict:
        task_id = params.get("task_id")
        config = params.get("config", {})
        # Accept the mission name under any of the historical key names the
        # backend has used; the canonical one in RoboEval is config.mission_name
        # but we've seen config.mission / config.env / config.env_name in logs.
        mission_name = (
            config.get("mission_name")
            or config.get("mission")
            or config.get("env")
            or config.get("env_name")
            or params.get("mission_name")
            or params.get("mission")
        )
        print(
            f"[telearms] start command: task_id={task_id} mission_name={mission_name} "
            f"param_keys={list(params.keys())} config_keys={list(config.keys())}"
        )

        if not task_id:
            print("[telearms] start rejected: missing task_id")
            return {"accepted": False, "error": "task_id is required"}
        if not mission_name:
            print(f"[telearms] start rejected: no mission in params/config → params={params}")
            return {"accepted": False, "error": "mission_name is required"}

        # Guard against double-start drift
        if (
            self._ws_client
            and self._ws_client.current_task_id
            and self._ws_client.current_task_id != task_id
        ):
            return {
                "accepted": False,
                "error": f"busy_with_other_task: current={self._ws_client.current_task_id}",
            }

        # (Re)load environment. We always recreate on start to avoid MuJoCo
        # stale-pointer segfaults between sessions. The WS handler itself runs
        # in a worker thread; GLFW/X11 renderer lifecycle must stay on the
        # simulation thread that owns the render loop.
        task_config = self._current_task
        if mission_name != self._current_task.name:
            env_cls = ENVIRONMENTS.get(mission_name)
            if env_cls is None:
                available = ", ".join(sorted(ENVIRONMENTS.keys()))
                print(
                    f"[telearms] start rejected: unknown mission '{mission_name}'. "
                    f"Available missions in this playground build:\n  {available}"
                )
                return {"accepted": False, "error": f"Unknown mission_name: {mission_name}"}
            task_config = get_task_by_name(mission_name) or TaskConfig(
                name=mission_name, description=f"{mission_name} task", enabled=True
            )

        try:
            self._request_environment_setup(task_config)
        except Exception as e:
            print(f"[telearms] start rejected: setup_environment failed: {e}")
            traceback.print_exc()
            return {"accepted": False, "error": f"setup_environment failed: {e}"}

        # Clear input state carried over from previous session
        self.pressed_keys.clear()
        self._gamepad_active = False

        # Agora handshake
        creds = self._fetch_agora_token(task_id)
        if not creds:
            return {"accepted": False, "error": "failed to fetch Agora credentials"}

        agora_channel = creds.get("agora_channel")
        agora_uid = creds.get("agora_uid")
        rtc_token = creds.get("agora_rtc_token")
        rtm_token = creds.get("agora_rtm_token")

        if not (agora_channel and rtc_token and rtm_token):
            return {"accepted": False, "error": "incomplete Agora credentials"}

        try:
            rtm_user_id = f"{agora_channel}-{agora_uid}"
            # Wait for any in-flight frame send, then keep the full native SDK
            # teardown/reinitialization under the same lock. This closes the
            # check-then-act race that previously produced status-139 crashes.
            self._video_paused = True
            with self._sdk_lock:
                try:
                    teleop_sdk.disconnect()
                except Exception:
                    pass
                try:
                    teleop_sdk.deinit()
                except Exception:
                    pass

                teleop_sdk.init(
                    message_received_callback=self._on_message_received,
                    on_event_callback=self._on_event_cb,
                )
                teleop_sdk.set_token(
                    agora_channel,
                    agora_uid,
                    rtc_token,
                    rtm_user_id,
                    rtm_token,
                )
                teleop_sdk.connect()
                self._video_paused = False
            print(f"[telearms] teleop_sdk connected → channel={agora_channel}, rtm_user={rtm_user_id}")
        except Exception as e:
            with self._sdk_lock:
                self._video_paused = True
            traceback.print_exc()
            return {"accepted": False, "error": f"teleop_sdk init failed: {e}"}

        # Do NOT start recording here. "start" fires at join/preview (the
        # operator can still reposition things). The .safetensors must begin
        # at Start Mission — the last reset, where the backend snaps the scene
        # to the start scenario and the real mission begins. That arrives as a
        # separate "reset" command; recording is armed there
        # (see _handle_reset_command).
        _set_busy_flag()
        return {"accepted": True, "task_id": task_id}

    def _handle_stop_command(self, params: dict) -> dict:
        task_id = params.get("task_id")
        reason = params.get("reason", "stop_requested")

        if (
            self._ws_client
            and self._ws_client.current_task_id
            and task_id != self._ws_client.current_task_id
        ):
            return {"accepted": False, "error": "stale_stop_command"}

        print(f"[telearms] Stop task={task_id} reason={reason}")

        # Cleared first so the verification worker can resume claiming even if
        # some teardown step below throws.
        _clear_busy_flag()

        self.pressed_keys.clear()
        self._gamepad_active = False

        # Stop producing frames before recording/S3 I/O and disconnect only
        # after any in-flight send_video() has returned.
        self._video_paused = True
        with self._sdk_lock:
            try:
                teleop_sdk.disconnect()
            except Exception as e:
                print(f"[telearms] Agora disconnect warning: {e}")

        # Save in-flight recording — pass task_id explicitly so the S3 layout
        # uses the right ID even if send_quit below has already cleared
        # ws_client.current_task_id.
        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording(task_id=task_id)
                print(f"[telearms] Recording saved (frames={self._frames_recorded})")
            except Exception as e:
                print(f"[telearms] Failed to save recording: {e}")

        # Notify backend
        terminal_reasons = {"user_disconnect", "timeout", "error", "cancelled"}
        if self._ws_client and task_id and reason in terminal_reasons:
            self._ws_client.send_quit(task_id=task_id, reason=reason)

        self._task_start_time = None
        self._frames_recorded = 0

        try:
            with self._env_lock:
                self._env.reset()
            self._init_arm_state()
            self._save_initial_positions()
            self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        except Exception as e:
            print(f"[telearms] Env reset warning: {e}")

        return {"accepted": True}

    def _handle_reset_command(self, params: dict) -> dict:
        with self._env_lock:
            self._env.reset()
        self._init_arm_state()
        self._save_initial_positions()
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        # Start Mission: the backend sends "reset" as the last reset before the
        # real mission (snapping the scene to the start scenario). This is where
        # the .safetensors must begin — arm the recorder so the main loop's
        # _start_recording_for_task captures from exactly this reset. Reset the
        # session timers here so duration/frame_count measure the mission only.
        self._task_start_time = time.time()
        self._frames_recorded = 0
        self._env_start_recording_event.set()
        return {"accepted": True}

    def _handle_preload_command(self, params: dict) -> dict:
        mission_name = params.get("mission_name")
        if not mission_name:
            return {"accepted": False, "error": "mission_name is required"}
        if mission_name == self._current_task.name:
            return {"accepted": True, "already_loaded": True}
        if self._ws_client and self._ws_client.current_task_id:
            return {"accepted": False, "error": "busy_with_active_task"}

        env_cls = ENVIRONMENTS.get(mission_name)
        if env_cls is None:
            return {"accepted": False, "error": f"Unknown mission_name: {mission_name}"}

        task_config = get_task_by_name(mission_name) or TaskConfig(
            name=mission_name, description=f"{mission_name} task", enabled=True
        )
        try:
            self._request_environment_setup(task_config)
            return {"accepted": True}
        except Exception as e:
            return {"accepted": False, "error": f"preload failed: {e}"}

    def _request_environment_setup(self, task_config: TaskConfig) -> None:
        """Rebuild an environment on the simulation/render thread."""
        self._main_thread_dispatcher.call(
            lambda: self._setup_environment_for_task(task_config),
            timeout=self.ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
        )

    def _setup_environment_for_task(self, task_config: TaskConfig) -> None:
        """Switch tasks while keeping renderer lifecycle on the calling thread."""
        previous_task = self._current_task
        self._current_task = task_config
        print(
            "[telearms] Rebuilding environment on simulation thread: "
            f"{task_config.name}"
        )
        try:
            self._setup_environment()
        except Exception:
            self._current_task = previous_task
            if previous_task.name != task_config.name:
                try:
                    self._setup_environment()
                except Exception:
                    print("[telearms] Failed to restore previous environment")
                    traceback.print_exc()
            raise

    def _setup_environment(self):
        """Rebuild the env for the currently-selected task."""
        env_cls = ENVIRONMENTS.get(self._current_task.name)
        if env_cls is None:
            raise ValueError(f"Unknown env: {self._current_task.name}")

        with self._env_lock:
            # Close old env before replacing it — frees MuJoCo resources.
            try:
                if hasattr(self._env, "close"):
                    self._env.close()
            except Exception:
                pass

            webrtc_env_cls = self._create_webrtc_env(env_cls)
            self._env = webrtc_env_cls(
                render_mode="rgb_array",
                action_mode=self._action_mode,
                robot_cls=self._robot_cls,
            )
            self._env.mojo.model.vis.global_.offwidth = self._width
            self._env.mojo.model.vis.global_.offheight = self._height
            self._env.reset()

            # Wrist-cam offsets are body-local against THIS env's baked
            # cam_pos; drop the cached baked + offsets so the new env
            # captures fresh on first apply.
            self._wrist_cam_controller.invalidate()

        self._stats = TelearmsTeleopStats()
        self._stop_countdown = None
        self._current_action = np.zeros(self._env.action_space.shape)
        self._ik = self._env.inverse_kinematics
        self._setup_grippers()
        self._init_arm_state()
        self._save_initial_positions()
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {
            self.GRIPPER_KEYS["left_open"]: False,
            self.GRIPPER_KEYS["left_close"]: False,
            self.GRIPPER_KEYS["right_open"]: False,
            self.GRIPPER_KEYS["right_close"]: False,
        }
        for key in self.LEFT_RESET_KEYS | self.RIGHT_RESET_KEYS:
            self._prev_key_state[key] = False

    # ------------------------------------------------------------------
    # Scene-control RTM handlers
    # ------------------------------------------------------------------
    def _handle_reset_scene(self, data: dict):
        with self._env_lock:
            self._env.reset()
        self._init_arm_state()
        self._save_initial_positions()

    def _handle_change_task(self, data: dict):
        name = data.get("mission_name")
        if not name:
            return
        if self._ws_client and self._ws_client.current_task_id:
            print(f"[telearms] Ignoring change_task while {self._ws_client.current_task_id} active")
            return
        task_config = get_task_by_name(name)
        if task_config is None:
            print(f"[telearms] change_task: unknown {name}")
            return
        self._setup_environment_for_task(task_config)

    # ------------------------------------------------------------------
    # GStreamer / Agora video out
    # ------------------------------------------------------------------
    def _on_ws_connected(self):
        # Keep the pipeline ready, but do not feed it until a mission start has
        # connected Agora with fresh credentials.
        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_ws_disconnected(self):
        self._video_paused = True
        with self._sdk_lock:
            try:
                teleop_sdk.disconnect()
            except Exception as e:
                print(f"[telearms] Agora disconnect warning: {e}")

        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording()
                print("[telearms] Recording auto-saved on disconnect")
            except Exception as e:
                print(f"[telearms] Auto-save on disconnect failed: {e}")
        # The control plane is gone, so no stop command will arrive to clear
        # the busy flag — treat disconnect as end-of-task for the flag too.
        _clear_busy_flag()

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                data = mapinfo.data
                # First few frames: log the raw NAL prefix to verify the
                # start-code layout we're stripping is actually 6 bytes.
                if self._frames_recorded < 3:
                    head = data[:12].hex() if len(data) >= 12 else data.hex()
                    print(f"[telearms] x264 out #{self._frames_recorded}: "
                          f"len={len(data)} prefix={head}")
                # Skip the 6-byte Annex-B start code header that GStreamer
                # prepends; teleop_sdk.send_video expects raw NAL units.
                # Re-check the active state under the SDK lock. A start/stop
                # command can change it after this callback begins, but cannot
                # deinit the SDK while this native send is in progress.
                with self._sdk_lock:
                    if not self._video_paused:
                        teleop_sdk.send_video(data[6:])
                        self._frames_recorded += 1
                        if self._frames_recorded % 150 == 0:
                            print(
                                f"[telearms] pushed {self._frames_recorded} "
                                "frames to Agora"
                            )
            finally:
                buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def _on_render(self, frame):
        if self._video_paused:
            return

        if not hasattr(self, "_render_push_count"):
            self._render_push_count = 0

        # The pipeline's appsrc was created with caps saying 1280x720 RGB.
        # If the MuJoCo renderer produces a different size (gym's renderer
        # caches width/height at construction time and ignores later
        # model.vis.global_ tweaks) the x264enc reads the raw bytes with the
        # wrong stride and the browser decodes only green. Resize the frame
        # to match the pipeline caps before pushing.
        import numpy as _np
        expected_h, expected_w = HEIGHT, WIDTH
        if frame.shape[:2] != (expected_h, expected_w):
            if self._render_push_count < 3:
                print(f"[telearms] frame shape {frame.shape} != "
                      f"pipeline {expected_h}x{expected_w}, resizing")
            try:
                from PIL import Image as _Image
                pil = _Image.fromarray(frame)
                pil = pil.resize((expected_w, expected_h), _Image.BILINEAR)
                frame = _np.asarray(pil)
            except Exception as e:
                print(f"[telearms] resize failed: {e}; skipping frame")
                return

        if not frame.flags["C_CONTIGUOUS"]:
            frame = _np.ascontiguousarray(frame)

        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        ret = self.appsrc.emit("push-buffer", buf)
        self._render_push_count += 1
        if self._render_push_count <= 3 or self._render_push_count % 150 == 0:
            print(f"[telearms] pushed render #{self._render_push_count} "
                  f"shape={frame.shape} bytes={len(data)} appsrc={ret}")

    def _on_event_cb(self, event, payload):
        print(f"[telearms] teleop_sdk event: {event} {payload}")

    # ------------------------------------------------------------------
    # Agora RTM input
    # ------------------------------------------------------------------
    def _on_message_received(self, msg: str):
        """Callback from teleop_sdk — every RTM message from the frontend."""
        # Ignore stale messages that arrive after task end but before UI disconnect
        if not self._ws_client or not self._ws_client.current_task_id:
            return
        try:
            action = json.loads(msg)
        except json.JSONDecodeError:
            return

        t = action.get("type")
        data = action.get("data", {})
        if t == "keyboard":
            self._gamepad_active = False
            new_pressed = {k for k, v in data.items() if v and k not in _IGNORED_FRONTEND_KEYS}
            backend_keys = self.pressed_keys & _IGNORED_FRONTEND_KEYS
            self.pressed_keys = new_pressed | backend_keys
            if new_pressed:
                self._input_methods_used.add("keyboard")
        elif t == "gamepad":
            active = False
            axes = data.get("axes") or []
            for i, v in enumerate(axes[:4]):
                self.gamepad_axes[i] = float(v)
                if abs(self.gamepad_axes[i]) > 0.01:
                    active = True
            buttons = data.get("buttons") or []
            for i, v in enumerate(buttons):
                if i >= len(self.gamepad_buttons):
                    break
                self.gamepad_buttons[i] = float(v)
                if self.gamepad_buttons[i] > 0.01:
                    active = True
            self._gamepad_active = active
            if active:
                self._input_methods_used.add("gamepad")
        elif t == "scene_control":
            # frontend command channel (reset_scene / change_task)
            self.scene_control_queue.put((data.get("action"), data))
        elif t == "camera_view":
            # View-mode (UI 'V' toggle). Apply happens on the main loop
            # under the env lock; here we only queue.
            self.camera_view_queue.put(data)
        elif t == "wrist_camera_view":
            # Gripper-debug panel — body-local offsets for the wrist cams.
            # Applied on the main loop under the env lock so we don't
            # write cam_pos while a render is reading it.
            self.wrist_camera_view_queue.put(data)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, exit_event: Optional[threading.Event] = None,
            on_running_event: Optional[threading.Event] = None):
        if exit_event is None:
            exit_event = threading.Event()
        if on_running_event is None:
            on_running_event = threading.Event()
        on_running_event.set()
        pacer = FixedRatePacer(CONTROL_HZ)

        try:
            while not exit_event.is_set():
                # Start/preload commands arrive on WS worker threads. Run their
                # thread-affine renderer work before the next physics/render tick.
                self._main_thread_dispatcher.run_pending()
                with self._env_lock:
                    if self._env is None:
                        pass  # mid-swap; sleep below
                    else:
                        self._handle_input()
                        self._apply_gamepad_deltas()
                        action = self._get_action()
                        for _ in range(self.STEPS_COUNT_FACTOR):
                            self._env.step(action, fast=True)
                        # In telearms mode the backend owns the recording
                        # lifecycle, so we do NOT run the reward-triggered
                        # termination countdown (which saves local demos).

                if self._env is None:
                    pacer.wait(exit_event)
                    continue

                self._render_frame()
                pacer.wait(exit_event)
        finally:
            self.shutdown()

    def _handle_input(self):
        """Drain scene-control queue, honor recording start, process resets."""
        if self._env_start_recording_event.is_set():
            self._start_recording_for_task()
            self._env_start_recording_event.clear()

        while not self.scene_control_queue.empty():
            try:
                action_name, data = self.scene_control_queue.get_nowait()
                handler = self.scene_control_handlers.get(action_name)
                if handler:
                    try:
                        handler(data)
                    except Exception as e:
                        print(f"[telearms] scene_control handler error ({action_name}): {e}")
                self.scene_control_queue.task_done()
            except pyqueue.Empty:
                break

        self._process_reset_keys()
        self._drain_camera_view_queue()
        self._drain_wrist_camera_view_queue()
        self._maybe_apply_cam_override()

    def _drain_camera_view_queue(self):
        """Apply any view-mode commands queued since the last tick.

        Coalesces multiple queued ops into a single env write. Unknown
        actions and malformed payloads are dropped silently — this is a
        best-effort input channel, not a control plane.
        """
        if self.camera_view_queue.empty():
            return
        # Labels for the ops coalesced into this tick's single env write, so the
        # recorded track keeps "what the operator pressed" without carrying the
        # full raw payloads (which replay never needs — the pose supersedes them).
        ops: list = []
        while not self.camera_view_queue.empty():
            try:
                data = self.camera_view_queue.get_nowait()
            except pyqueue.Empty:
                break
            action = data.get("action")
            if action == "preset":
                name = data.get("name")
                if isinstance(name, str) and self._cam_controller.load_preset(name):
                    ops.append(f"preset:{name}")
            elif action == "reset":
                self._cam_controller.load_preset("default")
                ops.append("reset")
            elif action == "orbit":
                try:
                    dx = float(data.get("dx", 0))
                    dy = float(data.get("dy", 0))
                except (TypeError, ValueError):
                    continue
                if dx or dy:
                    self._cam_controller.orbit(
                        d_az=CAM_STEP_AZIM * dx,
                        d_el=CAM_STEP_ELEV * dy,
                    )
                    ops.append("orbit")
            elif action == "zoom":
                try:
                    delta = float(data.get("delta", 0))
                except (TypeError, ValueError):
                    continue
                if delta:
                    self._cam_controller.zoom(CAM_STEP_DIST * delta)
                    ops.append("zoom")

        if ops and self._env is not None:
            if self._cam_controller.apply(self._env):
                self._capture_camera_event(
                    "view", ops, sph=dict(self._cam_controller.state)
                )
                self._input_methods_used.add("view_mode")

    def _drain_wrist_camera_view_queue(self):
        """Apply wrist-cam offset commands queued since the last tick.

        Supported actions:
          nudge:  data = {side, axis ∈ {x,y,z}, delta ∈ ±1}
          reset:  data = {side?}  — clears one or both sides
        Anything else is dropped silently.
        """
        if self.wrist_camera_view_queue.empty():
            return
        applied = False
        while not self.wrist_camera_view_queue.empty():
            try:
                data = self.wrist_camera_view_queue.get_nowait()
            except pyqueue.Empty:
                break
            action = data.get("action")
            if action == "nudge":
                side = data.get("side")
                axis = data.get("axis")
                try:
                    delta = float(data.get("delta", 0))
                except (TypeError, ValueError):
                    continue
                if delta and self._wrist_cam_controller.nudge(side, axis, delta):
                    applied = True
            elif action == "reset":
                self._wrist_cam_controller.reset(data.get("side"))
                applied = True

        if applied and self._env is not None:
            self._wrist_cam_controller.apply(self._env)

    def _maybe_apply_cam_override(self):
        """Hot-reload the external camera pose from CAM_OVERRIDE_PATH.

        Polls the file's mtime; applies only when it changes. Safe to call
        every tick — when the file is absent or unchanged this is a single
        stat() call. Malformed YAML is logged and ignored so a typo doesn't
        kill the stream.
        """
        try:
            mtime = CAM_OVERRIDE_PATH.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._cam_override_mtime:
            return
        self._cam_override_mtime = mtime

        try:
            data = yaml.safe_load(CAM_OVERRIDE_PATH.read_text())
        except Exception as e:
            print(f"[telearms] cam_override YAML parse error: {e}")
            return
        if not isinstance(data, dict):
            return

        env = self._env
        if env is None:
            return
        try:
            ext_id = env._cameras_map["external"][0]
        except (KeyError, IndexError, AttributeError):
            return
        m = env._mojo.model
        d = env._mojo.data

        if "pos" in data:
            m.cam_pos[ext_id] = np.asarray(data["pos"], dtype=float)
        if "target" in data:
            m.cam_quat[ext_id] = self._look_at_quat(
                np.asarray(m.cam_pos[ext_id], dtype=float),
                np.asarray(data["target"], dtype=float),
            )
        elif "quat" in data:
            m.cam_quat[ext_id] = np.asarray(data["quat"], dtype=float)
        if "fovy" in data:
            m.cam_fovy[ext_id] = float(data["fovy"])

        mujoco.mj_forward(m, d)
        print(f"[telearms] cam_override applied: {data}")
        # Recorded too: this path bypasses _cam_controller entirely, so without
        # it the track would silently disagree with what was actually rendered.
        self._capture_camera_event("override", [], sph=None)

    @staticmethod
    def _look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Camera quat looking from pos to target; cam +X locked to world -Y.

        Matches tools/calibrate_external.py so Agora frames preserve the
        "left arm stays on the left" convention.
        """
        look = target - pos
        look /= np.linalg.norm(look)
        z_axis = -look
        x_axis = np.array([0.0, -1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis]).flatten()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot)
        return quat

    # ------------------------------------------------------------------
    # Action generation — overrides KeyboardTeleop's mode-toggle pipeline
    # with independent position + orientation control (matches the UI).
    # ------------------------------------------------------------------
    def get_next_action(self, advance_target: bool = True) -> ActType:
        # `advance_target` is accepted to match KeyboardTeleop's signature
        # (KeyboardTeleop._get_action calls get_next_action(advance_target=...)).
        # Teleop recomputes control from live input every call, so there is no
        # target to advance; the flag is intentionally unused here.
        grip_left, grip_right = self._process_gripper_input()
        self._handle_position_control()
        self._handle_orientation_control()
        return self._calculate_control(grip_left, grip_right)

    def _process_gripper_input(self):
        """Edge-triggered toggle: z/n open, x/m close (UI Grip Hand row)."""
        left_open = self.GRIPPER_KEYS["left_open"]
        left_close = self.GRIPPER_KEYS["left_close"]
        right_open = self.GRIPPER_KEYS["right_open"]
        right_close = self.GRIPPER_KEYS["right_close"]

        if left_open in self.pressed_keys and not self._prev_key_state[left_open]:
            self._gripper_toggle_state["left"] = 1.0
        if left_close in self.pressed_keys and not self._prev_key_state[left_close]:
            self._gripper_toggle_state["left"] = 0.0
        self._prev_key_state[left_open] = left_open in self.pressed_keys
        self._prev_key_state[left_close] = left_close in self.pressed_keys

        if right_open in self.pressed_keys and not self._prev_key_state[right_open]:
            self._gripper_toggle_state["right"] = 1.0
        if right_close in self.pressed_keys and not self._prev_key_state[right_close]:
            self._gripper_toggle_state["right"] = 0.0
        self._prev_key_state[right_open] = right_open in self.pressed_keys
        self._prev_key_state[right_close] = right_close in self.pressed_keys

        return self._gripper_toggle_state["left"], self._gripper_toggle_state["right"]

    # Cap on how far a position target may lead the measured wrist (m). The
    # targets below are pure integrators: every held frame adds a step whether
    # or not the arm managed to follow. Once the arm stops -- on YAM that is
    # joint4 saturating with the wrist still ~55 cm out, well short of the arm's
    # 0.55 m -- the target keeps running while the IK can only return the
    # nearest pose it can hold, and that nearest pose slides along the boundary.
    # Measured pressing "forward" on the right arm: lead grew to 110 mm and the
    # tool went sideways (lateral/forward 0.01 -> 1.32) and then backwards.
    # 30 mm is several times any lead normal motion produces, so this only
    # engages once the arm is genuinely stuck.
    ARM_POSITION_LEAD = 0.03

    def _handle_position_control(self):
        """WASD/IJKL move in XY, Q/E and U/O in Z (UI Move + Raise/Lower)."""
        step = self.ARM_MOVEMENT_STEP

        if self.gripper_l:
            keys = self.LEFT_ARM_POSITION_KEYS
            moving = False
            if keys['forward'] in self.pressed_keys: self.left_arm_target[0] += step; moving = True
            if keys['back'] in self.pressed_keys:    self.left_arm_target[0] -= step; moving = True
            if keys['left'] in self.pressed_keys:    self.left_arm_target[1] += step; moving = True
            if keys['right'] in self.pressed_keys:   self.left_arm_target[1] -= step; moving = True
            if keys['up'] in self.pressed_keys:      self.left_arm_target[2] += step; moving = True
            if keys['down'] in self.pressed_keys:    self.left_arm_target[2] -= step; moving = True
            self.left_arm_target = self._clamp_position_lead(
                self.left_arm_target, self.gripper_l, moving
            )

        if self.gripper_r:
            keys = self.RIGHT_ARM_POSITION_KEYS
            moving = False
            if keys['forward'] in self.pressed_keys: self.right_arm_target[0] += step; moving = True
            if keys['back'] in self.pressed_keys:    self.right_arm_target[0] -= step; moving = True
            if keys['left'] in self.pressed_keys:    self.right_arm_target[1] += step; moving = True
            if keys['right'] in self.pressed_keys:   self.right_arm_target[1] -= step; moving = True
            if keys['up'] in self.pressed_keys:      self.right_arm_target[2] += step; moving = True
            if keys['down'] in self.pressed_keys:    self.right_arm_target[2] -= step; moving = True
            self.right_arm_target = self._clamp_position_lead(
                self.right_arm_target, self.gripper_r, moving
            )

    def _clamp_position_lead(self, target, gripper, is_moving):
        """Bound how far a position target may lead the measured wrist."""
        # Only while this arm is actually being driven. Clamping an idle arm
        # would re-anchor its target on wherever the wrist currently sits, so
        # any sag would drag the target down after it and the arm would walk
        # away on its own -- including the arm the operator never touched.
        if self.ARM_POSITION_LEAD is None or not is_moving:
            return target
        # Targets are stored pelvis-relative; measure the wrist the same way.
        actual = np.asarray(gripper.wrist_position) - np.asarray(
            self._env.robot.pelvis.get_position()
        )
        lead = np.asarray(target) - actual
        distance = float(np.linalg.norm(lead))
        if distance <= self.ARM_POSITION_LEAD:
            return target
        return actual + lead * (self.ARM_POSITION_LEAD / distance)

    def _handle_orientation_control(self):
        """Yaw (1/4, 7/0) + pitch (2/3, 8/9) + roll (R/F, T/G) — UI Rotate Arm row."""
        step = self.ARM_ROTATION_STEP

        if self.gripper_l:
            keys = self.LEFT_ARM_ORIENTATION_KEYS
            if keys['yaw_left'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[0, 0, 1], angle=step)
            if keys['yaw_right'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[0, 0, 1], angle=-step)
            if keys['pitch_up'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[0, 1, 0], angle=step)
            if keys['pitch_down'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[0, 1, 0], angle=-step)
            if keys['roll_cw'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[1, 0, 0], angle=step)
            if keys['roll_ccw'] in self.pressed_keys:
                self.left_arm_orientation *= Quaternion(axis=[1, 0, 0], angle=-step)
            self.left_arm_orientation = self.left_arm_orientation.normalised

        if self.gripper_r:
            keys = self.RIGHT_ARM_ORIENTATION_KEYS
            if keys['yaw_left'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[0, 0, 1], angle=step)
            if keys['yaw_right'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[0, 0, 1], angle=-step)
            if keys['pitch_up'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[0, 1, 0], angle=step)
            if keys['pitch_down'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[0, 1, 0], angle=-step)
            if keys['roll_cw'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[1, 0, 0], angle=step)
            if keys['roll_ccw'] in self.pressed_keys:
                self.right_arm_orientation *= Quaternion(axis=[1, 0, 0], angle=-step)
            self.right_arm_orientation = self.right_arm_orientation.normalised

    def _process_reset_keys(self):
        """Edge-triggered: ESC resets left arm, DEL resets right (UI Reset Arm)."""
        left_pressed = any(k in self.pressed_keys for k in self.LEFT_RESET_KEYS)
        left_prev = any(self._prev_key_state.get(k, False) for k in self.LEFT_RESET_KEYS)
        if left_pressed and not left_prev:
            self._reset_left_arm()

        right_pressed = any(k in self.pressed_keys for k in self.RIGHT_RESET_KEYS)
        right_prev = any(self._prev_key_state.get(k, False) for k in self.RIGHT_RESET_KEYS)
        if right_pressed and not right_prev:
            self._reset_right_arm()

        for k in self.LEFT_RESET_KEYS | self.RIGHT_RESET_KEYS:
            self._prev_key_state[k] = k in self.pressed_keys

    def _reset_left_arm(self):
        if self.gripper_l:
            self.left_arm_target = self.left_arm_initial_position.copy()
            self.left_arm_orientation = Quaternion(self.left_arm_initial_orientation)

    def _reset_right_arm(self):
        if self.gripper_r:
            self.right_arm_target = self.right_arm_initial_position.copy()
            self.right_arm_orientation = Quaternion(self.right_arm_initial_orientation)

    def _render_frame(self):
        """Render to offscreen buffer and push through the GStreamer pipeline.

        Held under _env_lock for the render call so we don't read mujoco_renderer
        on an env mid-swap. _setup_environment closes the old env (which sets its
        mujoco_renderer to None) before publishing the replacement, so a lock-
        free render() races into a NoneType.render() crash whenever the start
        command brings a different mission than the one the launcher booted.
        Pushing the frame out (Agora/GStreamer) stays outside the lock so a slow
        encoder doesn't block start/stop handlers.
        """
        self._update_stats()
        if self._video_paused:
            return
        with self._env_lock:
            if self._env is None:
                return
            frame = self._env.render()
        if frame is not None:
            try:
                self._on_render(frame)
            except Exception as e:
                print(f"[telearms] render push failed: {e}")

    def _apply_gamepad_deltas(self):
        """Turn stick axes into pressed_keys deltas so the IK pipeline sees them."""
        if not self._gamepad_active:
            return

        # Map axes → virtual keys using the same layout the UI advertises:
        # left stick → left arm forward/back + left/right, LT/LB → z up/down,
        # same mirrored for right. Triggers (6/7) are shared vertical controls.
        dz = self.GAMEPAD_DEADZONE
        virtual = set()

        lx, ly = self.gamepad_axes[0], self.gamepad_axes[1]
        rx, ry = self.gamepad_axes[2], self.gamepad_axes[3]

        # Left stick → left arm XY (forward/back on Y, left/right on X)
        if ly < -dz:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["forward"])
        elif ly > dz:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["back"])
        if lx < -dz:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["left"])
        elif lx > dz:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["right"])

        # Right stick → right arm XY
        if ry < -dz:
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["forward"])
        elif ry > dz:
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["back"])
        if rx < -dz:
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["left"])
        elif rx > dz:
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["right"])

        # L/R triggers (buttons 6/7) drive both arms up/down together so
        # the trigger pair works as a shared elevator.
        if self.gamepad_buttons[6] > 0.5:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["down"])
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["down"])
        if self.gamepad_buttons[7] > 0.5:
            virtual.add(self.LEFT_ARM_POSITION_KEYS["up"])
            virtual.add(self.RIGHT_ARM_POSITION_KEYS["up"])

        # Shoulder buttons LB/RB (4/5) → close the gripper (edge trigger).
        # Split open/close means we only simulate the close press here; open
        # stays as a keyboard affordance.
        for idx, close_key in ((4, self.GRIPPER_KEYS["left_close"]),
                                (5, self.GRIPPER_KEYS["right_close"])):
            was_down = self._prev_gamepad_buttons[idx] > 0.5
            is_down = self.gamepad_buttons[idx] > 0.5
            if is_down and not was_down:
                virtual.add(close_key)

        self._prev_gamepad_buttons = list(self.gamepad_buttons)
        self.pressed_keys |= virtual

    # ------------------------------------------------------------------
    # Recording hooks
    # ------------------------------------------------------------------
    def _start_recording_for_task(self):
        """Reset the env, then start recording, so the demo's initial state matches.

        A LightweightDemo only stores the seed (utils.py:Metadata.from_env →
        seed=env.seed); on replay demo_player does env.reset(seed=demo.seed) to
        rebuild element positions and then applies the recorded actions. For the
        replay to line up, the first recorded timestep must be captured from the
        exact reset(seed) state. _setup_environment() reset the env earlier, but
        the Agora handshake plus the main loop's continuous _env.step() let the
        scene settle/drift before we get here, so recording without a fresh reset
        bakes a drifted start — and elements show up in the wrong place on replay.
        Reset immediately before record() (mirrors keyboard_input._start_recording).
        The main loop already holds _env_lock when this runs; RLock makes the
        re-acquire below safe and keeps the intent explicit.
        """
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()
        with self._env_lock:
            # Re-reset to the SAME seed Start Mission used (set by
            # _setup_environment's reset). This anchors the recording's first
            # frame to the exact Start Mission layout the operator saw, while
            # still clearing the drift the Agora handshake + main-loop stepping
            # introduced before recording begins. A bare reset() would draw a
            # NEW random seed, so the demo would start from a different layout
            # than Start Mission.
            self._env.reset(seed=self._env.seed)
        self._init_arm_state()
        self._save_initial_positions()
        self._input_methods_used = set()
        self._stop_countdown = None
        self._gripper_toggle_state = {"left": 0.0, "right": 0.0}
        self._prev_key_state = {
            self.GRIPPER_KEYS["left_open"]: False,
            self.GRIPPER_KEYS["left_close"]: False,
            self.GRIPPER_KEYS["right_open"]: False,
            self.GRIPPER_KEYS["right_close"]: False,
        }
        for key in self.LEFT_RESET_KEYS | self.RIGHT_RESET_KEYS:
            self._prev_key_state[key] = False
        self._demo_recorder.record(self._env, lightweight_demo=True)

        # Arm the camera track after record() (which creates the demo's
        # Metadata) so the baseline event is captured from the model as it
        # stands at step 0, immediately after the reset above.
        cam_id = stream_camera_id(self._env)
        if cam_id is None:
            print("[telearms] no POV camera resolved; camera track disabled")
        else:
            cam_name = camera_name_for_id(self._env, cam_id)
            self._camera_track_recorder.begin(
                camera_name=cam_name,
                camera_frame=camera_frame_for_id(self._env, cam_id),
                control_frequency=self._env.control_frequency,
                baseline_pose=camera_pose_snapshot(self._env, cam_id),
            )
            # Logged because which camera is streamed depends on how the robot's
            # cameras are registered; this line makes it unambiguous per run.
            print(f"[telearms] camera track armed on POV camera {cam_name!r} (id={cam_id})")

    def _capture_camera_event(
        self, src: str, ops: list, sph: Optional[dict] = None
    ) -> None:
        """Record the POV camera pose after something moved it.

        The step index is correct because _handle_input() (which drains the
        camera queues) runs *before* this tick's STEPS_COUNT_FACTOR env.step()
        calls: a count of N means steps 0..N-1 ran under the previous pose, so
        the first step under the new one is exactly N. Matches the track's
        half-open [k, next_k) semantics.
        """
        if not self._demo_recorder.is_recording:
            return
        if self._env is None:
            return
        cam_id = stream_camera_id(self._env)
        if cam_id is None:
            return
        self._camera_track_recorder.record(
            step=self._demo_recorder.recorded_step_count,
            t=time.time() - self._task_start_time if self._task_start_time else 0.0,
            src=src,
            ops=ops,
            pose=camera_pose_snapshot(self._env, cam_id),
            sph=sph,
        )

    def _save_recording(self, task_id: Optional[str] = None):
        """Save the current recording and metrics JSON, then upload to S3.

        Mirrors RoboEval's flow: stop recorder → save .safetensors with the
        task_id as filename → write metrics .json sidecar → push both to S3.
        S3 upload is best-effort; failures don't break the save.
        """
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()

        # Use explicit task_id if provided, otherwise fall back to the ws_client's
        # current task. Backend completion/quit notifications clear current_task_id
        # before this runs, which is why callers thread task_id through explicitly.
        if not task_id and self._ws_client and self._ws_client.current_task_id:
            task_id = self._ws_client.current_task_id

        custom_filename = f"{task_id}.safetensors" if task_id else None

        # Stamp the camera track onto the demo's metadata BEFORE save_demo(),
        # which writes the file and then clears self._demo.
        camera_track = self._camera_track_recorder.finish()
        demo = self._demo_recorder.demo
        if camera_track and demo is not None:
            demo.metadata.camera_track = camera_track
            print(
                f"[telearms] camera track: "
                f"{camera_move_count(camera_track)} POV moves on "
                f"{camera_track['camera_name']}"
            )

        free_mb = _free_mb(str(self._demo_recorder._demo_dir))
        if free_mb is not None and free_mb < MIN_FREE_MB_WARN:
            print(
                f"⚠️ LOW DISK: only {free_mb}MB free — this recording may fail to "
                "save with ENOSPC and the task will never be scored. Check "
                "/etc/cron.daily/telearms-recording-retention"
            )

        saved_path = self._demo_recorder.save_demo(filename=custom_filename)

        if saved_path:
            self._stats.demos_counter += 1
            print(f"📁 Recording saved: {saved_path}")

            metrics_path = self._save_metrics_json(
                saved_path, task_id, camera_track=camera_track
            )

            if metrics_path is None:
                # Without the sidecar the backend cannot score the task, even
                # though the .safetensors may upload fine. Used to be silent.
                print(
                    f"❌ NO SIDECAR for task {task_id}: metrics JSON was not written, "
                    f"so this task cannot be scored (free={free_mb}MB)"
                )

            if task_id:
                self._upload_telemetry_to_s3(saved_path, metrics_path, task_id)
            else:
                # Auto-save on disconnect/shutdown has no task_id, so nothing is
                # uploaded and the episode is invisible to the backend.
                print(
                    "⚠️ Recording saved without a task_id — not uploaded, "
                    "so it will never be scored"
                )
        else:
            print(
                f"❌ RECORDING LOST for task {task_id}: save_demo() returned no path "
                f"(free={free_mb}MB). Nothing was uploaded; this task cannot be scored."
            )

    def _get_simulation_metrics(self) -> SimulationMetrics:
        """Extract current simulation metrics from the environment.

        Every env attribute access is hasattr-guarded so envs that don't
        inherit MetricRolloutEval just leave the corresponding fields at
        their defaults instead of raising.
        """
        metrics = SimulationMetrics()

        try:
            with self._env_lock:
                metrics.task_name = self._current_task.name if self._current_task else "Unknown"
                # Read env.success directly at metric time — telearms-roboplayground
                # doesn't run the per-step success poll RoboEval has, so there's no
                # _cached_task_success to read.
                metrics.task_success = bool(getattr(self._env, "success", False)) if self._env else False
                metrics.elapsed_seconds = (
                    time.time() - self._task_start_time if self._task_start_time else 0
                )
                metrics.frame_count = self._frames_recorded

                env = self._env
                if env is None:
                    return metrics

                if hasattr(env, "_metrics"):
                    if hasattr(env, "_stage_flags"):
                        metrics.task_stages_reached = dict(env._stage_flags)

                    if hasattr(env, "_sum_vel_err") and hasattr(env, "_n_vel") and env._n_vel > 0:
                        metrics.arm_velocity_sync = env._sum_vel_err / env._n_vel
                    if hasattr(env, "_sum_z_err") and hasattr(env, "_n_z") and env._n_z > 0:
                        metrics.gripper_vertical_sync = env._sum_z_err / env._n_z

                    if hasattr(env, "_metrics_slip_count"):
                        metrics.slip_count = env._metrics_slip_count
                    if hasattr(env, "_env_collision_cnt"):
                        metrics.env_collision_count = env._env_collision_cnt
                    if hasattr(env, "_self_collision_cnt"):
                        metrics.self_collision_count = env._self_collision_cnt

                    if hasattr(env, "_cartesian_path_length"):
                        from roboeval.const import HandSide
                        if HandSide.LEFT in env._cartesian_path_length:
                            metrics.cartesian_path_length_left = env._cartesian_path_length[HandSide.LEFT]
                        if HandSide.RIGHT in env._cartesian_path_length:
                            metrics.cartesian_path_length_right = env._cartesian_path_length[HandSide.RIGHT]

                    if hasattr(env, "_orientation_path_length"):
                        from roboeval.const import HandSide
                        if HandSide.LEFT in env._orientation_path_length:
                            metrics.orientation_path_length_left = env._orientation_path_length[HandSide.LEFT]
                        if HandSide.RIGHT in env._orientation_path_length:
                            metrics.orientation_path_length_right = env._orientation_path_length[HandSide.RIGHT]

                    if hasattr(env, "_cartesian_jerk_sum") and hasattr(env, "_jerk_sample_count"):
                        from roboeval.const import HandSide
                        for side, jerk_sum in env._cartesian_jerk_sum.items():
                            count = env._jerk_sample_count.get(side, 1) or 1
                            if side == HandSide.LEFT:
                                metrics.avg_cartesian_jerk_left = jerk_sum / count
                            elif side == HandSide.RIGHT:
                                metrics.avg_cartesian_jerk_right = jerk_sum / count

                    if hasattr(env, "_joint_jerk_sum") and hasattr(env, "_jerk_sample_count"):
                        from roboeval.const import HandSide
                        for side, jerk_sum in env._joint_jerk_sum.items():
                            count = env._jerk_sample_count.get(side, 1) or 1
                            if side == HandSide.LEFT:
                                metrics.avg_joint_jerk_left = jerk_sum / count
                            elif side == HandSide.RIGHT:
                                metrics.avg_joint_jerk_right = jerk_sum / count

                if hasattr(env, "_final_metrics") and env._final_metrics:
                    fm = env._final_metrics
                    if "target_distance" in fm and isinstance(fm["target_distance"], dict):
                        metrics.target_distances = fm["target_distance"]
                    if "object_pose_error" in fm:
                        metrics.pose_error = (
                            fm["object_pose_error"]
                            if isinstance(fm["object_pose_error"], (int, float))
                            else 0.0
                        )

        except Exception as e:
            print(f"[MetricsError] Failed to extract metrics: {e}")

        return metrics

    def get_metrics_for_backend(self) -> dict:
        """Compute metrics + composite scores for the backend / telemetry JSON.

        Score formula matches RoboEval exactly: 30% coordination, 40% safety,
        30% smoothness, +20 bonus on success (capped at 100).
        """
        metrics = self._get_simulation_metrics()

        coordination_score = 1.0 / (1.0 + metrics.arm_velocity_sync + metrics.gripper_vertical_sync)
        safety_score = 1.0 / (
            1.0 + metrics.slip_count + metrics.env_collision_count + metrics.self_collision_count
        )

        total_jerk = (
            metrics.avg_cartesian_jerk_left
            + metrics.avg_cartesian_jerk_right
            + metrics.avg_joint_jerk_left
            + metrics.avg_joint_jerk_right
        )
        smoothness_score = 1.0 / (1.0 + total_jerk / 1000)

        composite_score = int(
            100 * (coordination_score * 0.3 + safety_score * 0.4 + smoothness_score * 0.3)
        )
        if metrics.task_success:
            composite_score = min(100, composite_score + 20)

        return {
            "simulation_metrics": metrics.to_dict(),
            "computed_scores": {
                "coordination_score": round(coordination_score, 4),
                "safety_score": round(safety_score, 4),
                "smoothness_score": round(smoothness_score, 4),
                "composite_score": composite_score,
            },
            "task_success": metrics.task_success,
            "duration_seconds": metrics.elapsed_seconds,
            "frames_recorded": metrics.frame_count,
        }

    def _save_metrics_json(
        self,
        safetensors_path: Path,
        task_id: Optional[str] = None,
        camera_track: Optional[dict] = None,
    ) -> Optional[Path]:
        """Save metrics as a JSON sidecar alongside the safetensors recording."""
        try:
            metrics_path = safetensors_path.with_suffix(".json")

            backend_metrics = self.get_metrics_for_backend()

            metrics_data = {
                "task_id": task_id,
                "task_name": self._current_task.name if self._current_task else "Unknown",
                "task_description": getattr(self._current_task, "description", "") if self._current_task else "",
                "robot_name": self._robot_cls.__name__ if self._robot_cls else "Unknown",
                "app_version": APP_VERSION,
                # Mark the new recording format: this build captures the full
                # MuJoCo state (info_qpos/info_qvel) per step, so the demo can
                # be replayed by restoring state and reproduces the recorded
                # success. Lets consumers tell faithful (v2) demos from old
                # action-only (lightweight) ones without opening the tensors.
                "demo_format": "v2",
                "has_state": True,
                # Deliberately NOT bumped to v3 alongside camera_track:
                # pi05_training/pipeline.py gates the training export on
                # demo_format == "v2" with strict equality, so bumping it would
                # silently drop every new recording from the dataset. The track
                # carries its own "version" field instead.
                "has_camera_track": camera_track is not None,
                "camera_moves": camera_move_count(camera_track),
                "camera_track": camera_track,
                "input_methods": sorted(self._input_methods_used),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_unix": time.time(),
                "safetensors_file": safetensors_path.name,
                **backend_metrics,
                "raw_final_metrics": None,
            }

            if hasattr(self._env, "_final_metrics") and self._env._final_metrics:
                raw_metrics = {}
                for k, v in self._env._final_metrics.items():
                    if hasattr(v, "tolist"):
                        raw_metrics[k] = v.tolist()
                    elif isinstance(v, dict):
                        raw_metrics[k] = {
                            str(kk): vv.tolist() if hasattr(vv, "tolist") else vv
                            for kk, vv in v.items()
                        }
                    else:
                        raw_metrics[k] = v
                metrics_data["raw_final_metrics"] = raw_metrics

            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)

            print(f"📊 Metrics saved: {metrics_path}")
            return metrics_path

        except Exception as e:
            print(f"⚠️ Failed to save metrics JSON: {e}")
            return None

    def _upload_telemetry_to_s3(
        self,
        safetensors_path: Path,
        metrics_path: Optional[Path],
        task_id: str,
    ) -> None:
        """Upload telemetry files (.safetensors and .json) to S3.

        S3 layout: telemetry/{INSTANCE_ID}/tasks/{task_id}/{filename}
        Failures are logged but never raised — we don't want a flaky network
        run to block the local save.
        """
        if not S3_AVAILABLE:
            print("⚠️ S3 upload skipped: boto3 not installed")
            return

        if not S3_UPLOAD_ENABLED:
            print("⚠️ S3 upload disabled via S3_UPLOAD_ENABLED=false")
            return

        try:
            instance_id = os.environ.get("INSTANCE_ID", "unknown")
            instance_id_clean = instance_id.replace("-", "").lower()

            s3_client = boto3.client("s3", region_name=AWS_REGION)

            s3_prefix = f"telemetry/{instance_id_clean}/tasks/{task_id}"

            safetensors_key = f"{s3_prefix}/{task_id}.safetensors"
            print(f"☁️ Uploading to s3://{TELEARMS_S3_BUCKET}/{safetensors_key}...")
            s3_client.upload_file(
                str(safetensors_path),
                TELEARMS_S3_BUCKET,
                safetensors_key,
            )
            print(f"✅ Uploaded: s3://{TELEARMS_S3_BUCKET}/{safetensors_key}")

            if metrics_path and metrics_path.exists():
                json_key = f"{s3_prefix}/{task_id}.json"
                s3_client.upload_file(
                    str(metrics_path),
                    TELEARMS_S3_BUCKET,
                    json_key,
                )
                print(f"✅ Uploaded: s3://{TELEARMS_S3_BUCKET}/{json_key}")

        except Exception as e:
            # Still non-fatal — a flaky network must not kill the episode — but
            # it is no longer quiet. A sustained streak means every task on this
            # instance is silently unscoreable, which is exactly how BIT-2456
            # ran for four days unnoticed.
            self._s3_upload_failures = getattr(self, "_s3_upload_failures", 0) + 1
            free_mb = _free_mb(str(safetensors_path.parent))
            print(
                f"❌ S3 upload failed (consecutive failures: "
                f"{self._s3_upload_failures}) task={task_id} free={free_mb}MB: "
                f"{type(e).__name__}: {e}"
            )
            if self._s3_upload_failures in (1, 5) or self._s3_upload_failures % 25 == 0:
                print(
                    f"🚨 TELEMETRY UPLOAD BROKEN on {os.environ.get('INSTANCE_ID', 'unknown')}: "
                    f"{self._s3_upload_failures} consecutive failures — every task since "
                    "the first failure is unscoreable and earns no points."
                )
        else:
            self._s3_upload_failures = 0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        print(f"[telearms] Signal {signum}, shutting down")
        self.shutdown()
        os._exit(0)

    def _cleanup_on_exit(self):
        self.shutdown()

    def shutdown(self):
        if self._shutdown_done:
            return

        # Prevent new frame sends before waiting for any in-flight native call.
        self._video_paused = True

        # Save any open recording first so we don't lose data.
        if self._demo_recorder and (
            self._demo_recorder.is_recording or self._demo_recorder._demo is not None
        ):
            try:
                self._save_recording()
            except Exception as e:
                print(f"[telearms] save-on-exit failed: {e}")

        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception:
                pass

        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

        with self._sdk_lock:
            try:
                teleop_sdk.disconnect()
            except Exception:
                pass
            try:
                teleop_sdk.deinit()
            except Exception:
                pass

        _clear_busy_flag()
        self._shutdown_done = True
        print("[telearms] Shutdown complete")
