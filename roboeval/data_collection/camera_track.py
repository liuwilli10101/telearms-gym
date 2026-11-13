"""The operator POV ("eagle") camera track: recording and lookup.

Every time view mode moves the streamed camera during a task, one event is
appended here carrying the trajectory step it takes effect at, the elapsed
wall-clock time, a short label for what the operator pressed, and the resulting
camera pose. The finished document is embedded in the demo's safetensors
metadata and copied into the telemetry JSON sidecar, so a replay can retrace
the exact camera motion.

Semantics are piecewise-constant and half-open: the event at step ``k``
describes the camera for steps ``[k, next_k)``, and there is always a synthetic
``baseline`` event at step 0.

Deliberately free of any mujoco / GStreamer / teleop_sdk import so it can be
unit-tested in CI. Callers read poses out of the model (see
camera_controller.camera_pose_snapshot) and hand plain dicts in.
"""
from __future__ import annotations

import logging
from typing import Optional


# Schema version of the camera-track document embedded in demo metadata.
CAMERA_TRACK_VERSION = 1

# Cap on recorded camera events per demo. Orbit/zoom repeat at 10 Hz (keyboard)
# or 20 Hz (gamepad) while held and missions run up to 900 s, so an uncapped
# track could reach ~18k events. On overflow the recorder halves what it has
# rather than truncating, so a long pan keeps its tail.
MAX_CAMERA_EVENTS = 4000

# Poses closer than this in every component count as unchanged. Needed because
# a clamped orbit or zoom still reports as "applied", so holding a key at a
# limit would otherwise log an identical pose at the full message rate.
POSE_EPS = 1e-6

# Decimals kept when serializing. Keeps the artifact diffable and guarantees
# plain Python floats reach json.dumps (numpy scalars raise).
POSE_ROUND = 6


def round_pose(pose: dict) -> dict:
    """Normalize a {pos, quat, fovy} pose to rounded plain floats."""
    return {
        "pos": [round(float(v), POSE_ROUND) for v in pose["pos"]],
        "quat": [round(float(v), POSE_ROUND) for v in pose["quat"]],
        "fovy": round(float(pose["fovy"]), POSE_ROUND),
    }


def round_sph(sph: Optional[dict]) -> Optional[dict]:
    """Normalize the camera controller's spherical state, or pass through None."""
    if not sph:
        return None
    return {
        "target": [round(float(v), POSE_ROUND) for v in sph["target"]],
        "azimuth": round(float(sph["azimuth"]), 3),
        "elevation": round(float(sph["elevation"]), 3),
        "distance": round(float(sph["distance"]), 3),
        "fovy": round(float(sph["fovy"]), 3),
    }


def poses_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    """True when two poses match within POSE_EPS on every component."""
    if a is None or b is None:
        return False
    if abs(float(a["fovy"]) - float(b["fovy"])) > POSE_EPS:
        return False
    for key in ("pos", "quat"):
        for lhs, rhs in zip(a[key], b[key]):
            if abs(float(lhs) - float(rhs)) > POSE_EPS:
                return False
    return True


def move_count(track: Optional[dict]) -> int:
    """Operator-driven camera changes in a track, excluding the baseline."""
    if not track:
        return 0
    if track.get("total_moves") is not None:
        return max(0, int(track["total_moves"]))
    # Compatibility with tracks written before total_moves was embedded. Each
    # dropped event represents a real pose change removed by thinning.
    retained = max(0, len(track.get("events") or []) - 1)
    return retained + max(0, int(track.get("dropped_events") or 0))


def frame_sample_indices(
    step_count: int, step_frequency: int, output_fps: int
) -> set[int]:
    """Evenly sample post-step states without changing playback duration."""
    step_count = max(0, int(step_count))
    step_frequency = int(step_frequency)
    output_fps = int(output_fps)
    if step_count == 0:
        return set()
    if step_frequency <= 0 or output_fps <= 0:
        raise ValueError("step_frequency and output_fps must be positive")

    frame_count = min(
        step_count,
        max(1, round(step_count / step_frequency * output_fps)),
    )
    if frame_count == 1:
        return {0}
    last = step_count - 1
    return {
        round(frame_index * last / (frame_count - 1))
        for frame_index in range(frame_count)
    }


class CameraTrack:
    """Read-side view of a recorded track: step -> pose lookup.

    ``index_scale`` maps replay indices into recorded-step space. It exists
    because DemoConverter.decimate hands the decimated demo the SAME metadata
    object, so a track survives decimation carrying indices that are
    ``decimation_rate`` times too large. ``index_offset`` selects a position
    within each scaled block; post-step renders use the final position.
    """

    def __init__(
        self,
        track: Optional[dict],
        index_scale: int = 1,
        index_offset: int = 0,
    ):
        self._track = track or {}
        self._events: list[dict] = list(self._track.get("events") or [])
        self._index_scale = max(1, int(index_scale))
        self._index_offset = max(0, int(index_offset))
        self._cursor = 0

    @property
    def camera_name(self) -> Optional[str]:
        return self._track.get("camera_name")

    @property
    def events(self) -> list[dict]:
        return self._events

    def __len__(self) -> int:
        return len(self._events)

    def pose_at(self, step: int) -> Optional[dict]:
        """Event in effect at a replay index, or None for an empty track.

        Walks a monotonic cursor, so sequential replay is O(1) amortized; a
        caller that jumps backwards restarts the walk rather than misreporting.
        """
        if not self._events:
            return None
        target = int(step) * self._index_scale + self._index_offset
        if self._events[self._cursor]["step"] > target:
            self._cursor = 0
        while (
            self._cursor + 1 < len(self._events)
            and self._events[self._cursor + 1]["step"] <= target
        ):
            self._cursor += 1
        return self._events[self._cursor]


class CameraTrackRecorder:
    """Accumulates POV camera changes for one recording."""

    def __init__(self, max_events: int = MAX_CAMERA_EVENTS):
        # Floor of 2 so a pathological cap can't discard the baseline.
        self._max_events = max(2, int(max_events))
        self._reset()

    def _reset(self) -> None:
        self._events: list[dict] = []
        self._camera_name: Optional[str] = None
        self._camera_frame: str = "world"
        self._control_frequency: int = 0
        self._decimation: int = 1
        self._dropped: int = 0
        self._started: bool = False
        # Stride sampling once the cap is hit: accept every Nth change and
        # remember the most recent one so finish() can always end the track on
        # the true final pose.
        self._seen: int = 0
        self._pending: Optional[dict] = None
        # Dedupe against the last pose actually observed, not the last sampled
        # event. Those differ after thinning begins.
        self._last_pose: Optional[dict] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def begin(
        self,
        camera_name: Optional[str],
        camera_frame: str,
        control_frequency: int,
        baseline_pose: dict,
    ) -> None:
        """Arm the recorder and record the baseline pose at step 0.

        ``baseline_pose`` must be read from the live model rather than assumed:
        ``env.reset()`` resets data but not the model (so a camera pose survives
        every reset) and ``_setup_environment`` never re-seeds the camera
        controller, meaning the pose in effect when recording starts can be a
        leftover from an earlier mission rather than world.xml's baked value.
        """
        self._reset()
        self._camera_name = camera_name
        self._camera_frame = camera_frame
        self._control_frequency = int(control_frequency)
        self._started = True
        baseline_pose = round_pose(baseline_pose)
        self._append(0, 0.0, "baseline", [], baseline_pose, None)
        self._last_pose = baseline_pose

    @property
    def is_recording(self) -> bool:
        return self._started

    @property
    def event_count(self) -> int:
        """Events recorded so far, including the baseline."""
        return len(self._events)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def record(self, step: int, t: float, src: str, ops: list,
               pose: dict, sph: Optional[dict] = None) -> None:
        """Append one camera change, skipping poses that did not actually move.

        Once the cap has been hit, only every ``_decimation``-th change is kept
        (the rest are counted in ``dropped_events``). The most recent change is
        always retained as ``_pending`` so ``finish()`` can close the track on
        the true final pose regardless of where the stride landed.
        """
        if not self._started:
            return
        pose = round_pose(pose)
        if poses_equal(pose, self._last_pose):
            return
        self._last_pose = pose

        event = self._build(step, t, src, ops, pose, sph)
        self._seen += 1
        if self._decimation > 1 and self._seen % self._decimation:
            self._dropped += 1
            self._pending = event
            return
        self._pending = None
        self._push(event)

    @staticmethod
    def _build(step: int, t: float, src: str, ops: list,
               pose: dict, sph: Optional[dict]) -> dict:
        return {
            "step": int(step),
            "t": round(float(t), 3),
            "src": src,
            "ops": list(ops),
            **round_pose(pose),
            "sph": round_sph(sph),
        }

    def _append(self, step: int, t: float, src: str, ops: list,
                pose: dict, sph: Optional[dict]) -> None:
        self._push(self._build(step, t, src, ops, pose, sph))

    def _push(self, event: dict) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._halve()

    def _halve(self) -> None:
        """Thin the track in place when it outgrows the cap.

        Keeps the baseline and every second event after it, then doubles the
        acceptance stride so subsequent changes arrive at the new resolution.
        Doubling the stride is what keeps this rare — each halve also doubles
        the headroom, so the number of halvings is logarithmic in the total
        number of changes rather than one per few appends.

        Chosen over truncation because truncation drops the *end* of a long pan,
        losing exactly the motion the track exists to reproduce, whereas this
        degrades resolution evenly and stays bounded at 2x the cap.
        """
        kept = [self._events[0]] + self._events[1::2]
        self._dropped += len(self._events) - len(kept)
        self._events = kept
        self._decimation *= 2
        logging.info(
            "[camera-track] event cap reached; thinned to %d events "
            "(decimation=%d, dropped=%d)",
            len(self._events), self._decimation, self._dropped,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def finish(self) -> Optional[dict]:
        """The track document, or None if recording never started.

        Flushes any stride-dropped final change first, so the track always ends
        on the pose the operator actually left the camera at — otherwise a
        thinned track could end mid-pan and replay would hold the wrong POV
        through the end of the episode.
        """
        if not self._started:
            return None
        if self._pending is not None:
            self._events.append(self._pending)
            self._dropped = max(0, self._dropped - 1)
            self._pending = None
        return {
            "version": CAMERA_TRACK_VERSION,
            "camera_name": self._camera_name,
            "camera_frame": self._camera_frame,
            "control_frequency": self._control_frequency,
            "events": self._events,
            "total_moves": self._seen,
            "event_decimation": self._decimation,
            "dropped_events": self._dropped,
        }
