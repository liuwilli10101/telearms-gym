"""External-camera controller driven by RTM view-mode commands.

Mirrors the spherical-coords state that tools/cam_cli.py keeps locally,
but lives inside the running sim so view-mode keypresses from the UI
move the streamed POV camera in real time. State is in-memory only — no
cam_override.yaml write, so a sim restart returns to world.xml's baked
pose.

This module also owns the pose read/write helpers and the replay-side
applier used by the camera-track recorder, so recording and playback
cannot drift apart in how they touch the model.
"""
import logging
import math

import mujoco
import numpy as np

from roboeval.data_collection.camera_track import (
    CameraTrack,
    poses_equal,
    round_pose,
)


# Per-tick step sizes. Client sends a dx/dy/delta in {-1, 0, +1}; the
# controller scales by these. Matches tools/cam_cli.py so poses tuned
# there transfer directly to view-mode presets.
STEP_AZIM = 5.0     # degrees
STEP_ELEV = 3.0     # degrees
STEP_DIST = 0.10    # meters
STEP_FOVY = 2.0     # degrees

# Clamps match cam_cli.py. MIN_DIST prevents the camera from passing
# through the look-at target; elevation is clamped off the poles to keep
# the look-at math stable.
MIN_ELEV = -85.0
MAX_ELEV = 85.0
MIN_DIST = 0.30
MIN_FOVY = 20.0
MAX_FOVY = 90.0


def stream_camera_id(env) -> int | None:
    """Id of the streamed POV camera, or None if it can't be resolved."""
    try:
        return env.stream_camera_id
    except AttributeError:
        # Env predates the accessor — fall back to the same resolution order.
        try:
            entry = env._cameras_map.get(
                "top_cam", env._cameras_map.get("external")
            )
        except AttributeError:
            return None
        return None if entry is None else entry[0]


def camera_name_for_id(env, cam_id: int) -> str | None:
    """Compiled MuJoCo name of a camera id (namespaced, as the model sees it)."""
    try:
        return mujoco.mj_id2name(env.mojo.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    except Exception:
        return None


def camera_frame_for_id(env, cam_id: int) -> str:
    """Parent body name of a camera, or "world" when parented to worldbody.

    Cameras baked into a robot model are body-local (YAM's overhead cam hangs
    off ``bimanual_base``), while world.xml's "external" is world-space. Poses
    are recorded and restored as raw model fields, which is frame-agnostic, but
    the frame is stored alongside so consumers can interpret the numbers.
    """
    try:
        model = env.mojo.model
        body_id = int(model.cam_bodyid[cam_id])
        if body_id == 0:
            return "world"
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
    except Exception:
        return "world"


def resolve_camera_id(env, name: str) -> int | None:
    """Look a camera up by compiled name; None when this model lacks it."""
    try:
        cam_id = mujoco.mj_name2id(env.mojo.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    except Exception:
        return None
    return None if cam_id is None or cam_id < 0 else int(cam_id)


def camera_pose_snapshot(env, cam_id: int) -> dict:
    """Read ground-truth {pos, quat, fovy} for a camera out of the model.

    Read back from the model rather than derived from CameraController state so
    the record stays correct no matter which path moved the camera — view-mode
    commands, the cam_override.yaml hot-reload, or a stale controller after an
    env rebuild.
    """
    model = env.mojo.model
    return round_pose({
        "pos": model.cam_pos[cam_id],
        "quat": model.cam_quat[cam_id],
        "fovy": model.cam_fovy[cam_id],
    })


def apply_camera_pose(env, cam_id: int, pose: dict, forward: bool = False) -> None:
    """Write a recorded pose back into the model.

    ``forward`` runs mj_forward to refresh d.cam_xpos/cam_xmat (what the
    renderer reads). Callers already running physics.forward() afterwards
    should leave it False and let that call do the work.
    """
    model = env.mojo.model
    model.cam_pos[cam_id] = np.asarray(pose["pos"], dtype=float)
    model.cam_quat[cam_id] = np.asarray(pose["quat"], dtype=float)
    model.cam_fovy[cam_id] = float(pose["fovy"])
    if forward:
        mujoco.mj_forward(model, env.mojo.data)


class CameraTrackApplier:
    """Drives a live env's POV camera from a recorded track.

    Thin mujoco-facing wrapper over CameraTrack, which owns the (pure)
    step -> pose lookup. Degrades to a no-op (``active is False``) for demos
    with no track, so replaying an older recording behaves exactly as before.

    ``index_scale`` maps replay indices back into recorded-step space; it is
    needed because DemoConverter.decimate hands the decimated demo the SAME
    metadata object, so a track survives decimation carrying indices that are
    ``decimation_rate`` times too large. ``index_offset`` selects a position
    within each scaled block; post-step renders use the final position.
    """

    def __init__(
        self,
        track: dict | None,
        env,
        index_scale: int = 1,
        index_offset: int = 0,
    ):
        self._env = env
        self._track = CameraTrack(
            track,
            index_scale=index_scale,
            index_offset=index_offset,
        )
        self._cam_id = None
        self._applied = None

        if not len(self._track):
            return

        name = self._track.camera_name
        cam_id = resolve_camera_id(env, name) if name else None
        if cam_id is None:
            # Recorded on a different robot/model. Fall back to whatever this
            # env streams rather than crashing, and say so once.
            cam_id = stream_camera_id(env)
            if cam_id is None:
                logging.warning(
                    "[camera-track] camera %r not in this model and no streamed "
                    "camera to fall back to; track will not be applied.", name,
                )
                return
            logging.warning(
                "[camera-track] camera %r not in this model; applying the track "
                "to the streamed camera (id %d) instead.", name, cam_id,
            )
        self._cam_id = cam_id

    @property
    def active(self) -> bool:
        """Whether there is anything to apply."""
        return self._cam_id is not None and len(self._track) > 0

    def pose_at(self, step: int) -> dict | None:
        """Event in effect at a replay index, or None when inactive."""
        return self._track.pose_at(step) if self.active else None

    def apply_for_step(self, step: int) -> None:
        """Write the pose for this step into the model, skipping no-ops."""
        event = self.pose_at(step)
        if event is None or poses_equal(event, self._applied):
            return
        apply_camera_pose(self._env, self._cam_id, event)
        self._applied = event


# Five named poses exposed to the UI as keys 1-5. `default` mirrors the
# over-the-shoulder baked into world.xml (see commits 5760001/fc6e2e9);
# the rest are first-pass values meant to be tuned live with cam_cli.py
# and then baked back in here.
PRESETS: dict[str, dict] = {
    "default": {
        "target": [0.50, 0.00, 1.35],
        "azimuth": 180.0,
        "elevation": 34.0,
        "distance": 0.30,
        "fovy": 90.0,
    },
    "front": {
        "target": [0.50, 0.00, 1.35],
        "azimuth": 180.0,
        "elevation": -11.0,
        "distance": 0.70,
        "fovy": 90.0,
    },
    "left": {
        "target": [0.50, 0.00, 1.35],
        "azimuth": 115.0,
        "elevation": 1.0,
        "distance": 1.00,
        "fovy": 90.0,
    },
    "right": {
        "target": [0.50, 0.00, 1.35],
        "azimuth": 235.0,
        "elevation": 1.0,
        "distance": 1.00,
        "fovy": 90.0,
    },
    "top": {
        "target": [0.50, 0.00, 1.00],
        "azimuth": 180.0,
        "elevation": 85.0,
        "distance": 1.00,
        "fovy": 60.0,
    },
}


class CameraController:
    """Spherical-coords state for the 'external' MuJoCo camera.

    Holds target/azimuth/elevation/distance/fovy and applies the derived
    pose to a RoboEval env on demand. Thread-unsafe: the caller owns
    synchronization (see TelearmsTeleop._drain_camera_view_queue, which
    runs under the env lock).
    """

    def __init__(self, preset: str = "default"):
        self._state = self._preset_state(preset)

    @staticmethod
    def _preset_state(name: str) -> dict:
        base = PRESETS.get(name, PRESETS["default"])
        return {
            "target": list(base["target"]),
            "azimuth": float(base["azimuth"]),
            "elevation": float(base["elevation"]),
            "distance": float(base["distance"]),
            "fovy": float(base["fovy"]),
        }

    @property
    def state(self) -> dict:
        return self._state

    def load_preset(self, name: str) -> bool:
        if name not in PRESETS:
            return False
        self._state = self._preset_state(name)
        return True

    def orbit(self, d_az: float = 0.0, d_el: float = 0.0) -> None:
        """Orbit around the target. +d_az spins CCW, +d_el tilts up."""
        self._state["azimuth"] = (self._state["azimuth"] + d_az) % 360.0
        self._state["elevation"] = max(
            MIN_ELEV, min(MAX_ELEV, self._state["elevation"] + d_el)
        )

    def zoom(self, delta: float) -> None:
        """+delta pushes the camera AWAY (zoom out); -delta zooms in."""
        self._state["distance"] = max(MIN_DIST, self._state["distance"] + delta)

    def compute_pos(self) -> list[float]:
        """World-space cam position derived from spherical state."""
        tx, ty, tz = self._state["target"]
        r = self._state["distance"]
        az = math.radians(self._state["azimuth"])
        el = math.radians(self._state["elevation"])
        px = tx + r * math.cos(el) * math.cos(az)
        py = ty + r * math.cos(el) * math.sin(az)
        pz = tz + r * math.sin(el)
        return [px, py, pz]

    def apply(self, env) -> bool:
        """Write the pose to the streamed POV ("eagle") camera.

        Targets RoboEvalEnv.stream_camera_id rather than "external" directly, so
        the camera being moved is always the one being rendered. On robots whose
        overhead cam is not registered under a bare "top_cam" key that resolves
        to world.xml's "external" — identical to the previous behavior.

        The spherical state is world-space, but a camera baked into a robot model
        is parented to a body and its cam_pos/cam_quat are in that body's frame,
        so the solution is converted into the target camera's parent frame before
        being written. Worldbody-parented cameras take the identity path.

        Returns False when no POV camera can be resolved.
        """
        cam_id = stream_camera_id(env)
        if cam_id is None:
            return False
        m = env._mojo.model
        d = env._mojo.data
        pos = np.asarray(self.compute_pos(), dtype=float)
        rot = self._look_at_mat(pos, np.asarray(self._state["target"], dtype=float))

        body_id = int(m.cam_bodyid[cam_id])
        if body_id != 0:
            # Express the world-space solution in the parent body's frame.
            body_pos = np.asarray(d.xpos[body_id], dtype=float)
            body_rot = np.asarray(d.xmat[body_id], dtype=float).reshape(3, 3)
            pos = body_rot.T @ (pos - body_pos)
            rot = body_rot.T @ rot

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.flatten())
        m.cam_pos[cam_id] = pos
        m.cam_quat[cam_id] = quat
        m.cam_fovy[cam_id] = float(self._state["fovy"])
        mujoco.mj_forward(m, d)
        return True

    @classmethod
    def _look_at_quat(cls, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """World-space quat for a cam at pos looking at target."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, cls._look_at_mat(pos, target).flatten())
        return quat

    @staticmethod
    def _look_at_mat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """3x3 world rotation for a cam at pos looking at target, world +Z up.

        Diverges from TelearmsTeleop._look_at_quat / tools/calibrate_external.py
        which lock cam +X to world -Y. That convention is stable only near
        azimuth=180°; orbiting past 90°/270° in view mode flipped the basis
        (cross product degenerated as look_dir aligned with -Y), producing a
        sudden roll. Elevation clamps (±85°) keep this formulation away from
        the gimbal-lock poles.
        """
        look = target - pos
        look = look / np.linalg.norm(look)
        z_axis = -look
        world_up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(world_up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        return np.column_stack([x_axis, y_axis, z_axis])


# Per-nudge step for wrist-camera offsets. 5 mm matches the resolution
# you'd want for sub-cm alignment of an end-effector cam without making
# every press a noticeable visual jump.
WRIST_STEP_POS = 0.005  # meters


class WristCameraController:
    """Per-side body-local position offsets for the wrist cameras.

    The wrist cams are <camera mode="fixed"> in h1.xml — cam_pos[id] is in
    the parent (wrist) body's frame. This controller captures the baked
    pose at first apply() and writes (baked + offset) on every subsequent
    nudge so the operator can fine-tune alignment in view mode and watch
    the wrist-overlay views update live.

    Quat is intentionally untouched (rotation tuning is out of scope for
    this debug pass — operators measure XYZ first, bake in h1.xml, then
    we'll add rotation if needed).

    Offsets reset on env rebuild via invalidate(); call that from
    TelearmsTeleop._setup_environment so a mission switch doesn't keep
    stale offsets relative to a different baked pose.
    """

    SIDES = ("left", "right")
    AXES = {"x": 0, "y": 1, "z": 2}

    def __init__(self):
        self._baked: dict[str, np.ndarray] = {}
        self._offsets: dict[str, np.ndarray] = {
            side: np.zeros(3) for side in self.SIDES
        }

    @staticmethod
    def _find_ids(env) -> dict[str, int]:
        """Resolve "left_wrist"/"right_wrist" cam ids from env.

        Cameras are registered with namespace prefixes (e.g.
        "panda nohand_left/left_wrist") so we match by suffix — same
        convention as roboeval_env.py:643.
        """
        ids: dict[str, int] = {}
        try:
            cam_map = env._cameras_map
        except AttributeError:
            return ids
        for name, entry in cam_map.items():
            try:
                cid = entry[0]
            except (TypeError, IndexError):
                continue
            if name.endswith("left_wrist"):
                ids["left"] = cid
            elif name.endswith("right_wrist"):
                ids["right"] = cid
        return ids

    def invalidate(self) -> None:
        """Drop baked cache and zero offsets. Call on env rebuild."""
        self._baked.clear()
        self._offsets = {side: np.zeros(3) for side in self.SIDES}

    def nudge(self, side: str, axis: str, delta: float) -> bool:
        """Bump side's offset on axis by delta * WRIST_STEP_POS."""
        if side not in self._offsets:
            return False
        idx = self.AXES.get(axis)
        if idx is None:
            return False
        self._offsets[side][idx] += float(delta) * WRIST_STEP_POS
        return True

    def reset(self, side: str | None = None) -> None:
        sides = (side,) if side else self.SIDES
        for s in sides:
            if s in self._offsets:
                self._offsets[s] = np.zeros(3)

    def offsets_snapshot(self) -> dict[str, list[float]]:
        return {s: list(map(float, v)) for s, v in self._offsets.items()}

    def apply(self, env) -> bool:
        """Write (baked + offset) to env's wrist cams.

        First call captures baked cam_pos so subsequent calls are additive
        instead of accumulating their own modifications.
        """
        ids = self._find_ids(env)
        if not ids:
            return False
        m = env._mojo.model
        for side, cid in ids.items():
            if side not in self._baked:
                self._baked[side] = np.asarray(m.cam_pos[cid], dtype=float).copy()
            m.cam_pos[cid] = self._baked[side] + self._offsets[side]
        return True
