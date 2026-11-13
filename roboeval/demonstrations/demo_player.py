"""Tool to replay demonstrations."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from roboeval.roboeval_env import CONTROL_FREQUENCY_MAX, RoboEvalEnv
from roboeval.data_collection.camera_controller import CameraTrackApplier
from roboeval.demonstrations.demo import Demo, DemoStep
from roboeval.demonstrations.demo_converter import DemoConverter

import copy
from roboeval.demonstrations.layout import reset_env_for_demo


class DemoPlayer:
    """Tool to replay demonstrations."""

    @staticmethod
    def replay(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
        render_mode: Optional[str] = None,
    ):
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency, render_mode=render_mode)
        DemoPlayer.replay_in_env(demo, env, demo_frequency)

    @staticmethod
    def replay_in_env(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ):
        """Replay demonstration in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        reset_env_for_demo(env, demo)
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.render_mode:
                env.render()
        # env.close()

    @staticmethod
    def replay_state_in_env(
        demo: Demo,
        env: RoboEvalEnv,
        render: bool = False,
        camera: str = "head",
        apply_camera_track: bool = True,
        frame_sink: Optional[Callable[[np.ndarray], None]] = None,
        composite: bool = False,
        render_indices: Optional[set[int]] = None,
    ) -> tuple[bool, list]:
        """Reproduce a demo by restoring full physics state per recorded step.

        Unlike action replay, this restores ``env.mojo.data.qpos``/``qvel`` from
        the recorded state each step, so contact-rich outcomes (e.g. grasps) are
        reproduced exactly — open-loop action replay does not. Every step must
        carry ``info[qpos]``/``info[qvel]`` (see ``DemoStep.state``); otherwise a
        ``ValueError`` is raised so callers can fall back to action replay.

        No frequency decimation is applied: each recorded state is independently
        exact. ``success`` is the final-step ``env.success`` — the same metric
        the recorder writes to the sidecar JSON (instantaneous success at the
        last recorded state), so replay agrees with the recorded outcome.

        When the demo carries a camera track, the operator's POV camera is moved
        to its recorded pose for each step, so the render retraces the camera
        motion the player actually made. Demos without a track render exactly as
        before.

        Args:
            frame_sink: Called with each rendered frame instead of accumulating
                them. Strongly preferred for long demos: a 2-minute recording is
                ~10.8k steps, so collecting 1280x720x3 frames in a list needs
                tens of GB. ``frames`` is only populated when this is None.
            composite: Render via ``env.render()`` (the POV plus the two wrist
                insets — the frame the operator actually saw) rather than a
                single named camera.
            render_indices: Optional subset of state indices to render. State
                restoration and success evaluation still run for every step.

        Returns:
            (success, frames) where frames is a list of rendered RGB arrays when
            ``render`` is True and ``frame_sink`` is None (empty otherwise).
        """
        reset_env_for_demo(env, demo)
        # Built after reset so the applier resolves cameras against the live model.
        applier = (
            CameraTrackApplier(demo.camera_track, env)
            if (apply_camera_track and render)
            else None
        )
        is_temporal = (
            getattr(env, "USE_TEMPORAL_SUCCESS", False)
            and getattr(env, "_temporal_checker", None) is not None
        )
        frames: list = []
        success = False
        for index, step in enumerate(demo.timesteps):
            state = step.state
            if state is None:
                raise ValueError(
                    "Demo has no recorded physics state; use action replay."
                )
            qpos, qvel = state
            if (
                qpos.shape != env.mojo.data.qpos.shape
                or qvel.shape != env.mojo.data.qvel.shape
            ):
                raise ValueError(
                    f"Recorded state shape {qpos.shape}/{qvel.shape} does not "
                    f"match env {env.mojo.data.qpos.shape}/"
                    f"{env.mojo.data.qvel.shape}; model changed since recording."
                )
            env.mojo.data.qpos[:] = np.nan_to_num(qpos)
            env.mojo.data.qvel[:] = np.nan_to_num(qvel)
            # Move the POV camera to its recorded pose BEFORE forward(), so the
            # single forward() below also refreshes d.cam_xpos/cam_xmat (what the
            # renderer reads) — no extra mj_forward needed.
            if applier is not None:
                applier.apply_for_step(index)
            # Recompute contacts / body & site poses / derived data from the
            # restored state without integrating time, so success predicates and
            # the renderer see a consistent world.
            env.mojo.physics.forward()
            # Force lazily-cached success/reward to recompute against the state.
            env._step_cache.clean()
            # Temporal-success tasks need the ordered checker driven per step.
            if is_temporal:
                env._on_step()
            # Final-step success is the recorder's metric (see docstring).
            success = bool(env.success)
            if render and (render_indices is None or index in render_indices):
                frame = (
                    env.render()
                    if composite
                    else env.mujoco_renderer.render("rgb_array", camera_name=camera)
                )
                if frame_sink is not None:
                    frame_sink(frame)
                else:
                    frames.append(frame)
        return success, frames

    @staticmethod
    def replay_in_env_save_position_orientation(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ):
        """Replay demonstration in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        reset_env_for_demo(env, demo)
        position_lefts = []
        position_rights = []
        orientations_left = []
        orientations_right = []
        
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.render_mode:
                env.render()


            

            for hand_side, site_obj in env.robot._wrist_sites.items():
                site_id = site_obj.id
                site_name = site_obj.mjcf.name

                site_position = env.mojo.data.site_xpos[site_id]
                site_orientation = env.mojo.data.site_xmat[site_id]
                
                if 'left' in site_name:
                    position_lefts.append(copy.deepcopy(site_position))
                    orientations_left.append(copy.deepcopy(site_orientation))
                elif 'right' in site_name:
                    position_rights.append(copy.deepcopy(site_position))
                    orientations_right.append(copy.deepcopy(site_orientation))
                else:
                    raise ValueError(f"Unknown wrist site name: {site_name}")
                
        return position_lefts, position_rights, orientations_left, orientations_right

    @staticmethod
    def validate(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
    ) -> bool:
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency)
        return DemoPlayer.validate_in_env(demo, env, demo_frequency)

    @staticmethod
    def validate_in_env(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> bool:
        """Check if demonstration is successful in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        reset_env_for_demo(env, demo)
        is_successful = False
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.reward > 0:
                is_successful = True
                break
        env.close()
        return is_successful

    @staticmethod
    def _get_timesteps_for_replay(
        demo: Demo, env: RoboEvalEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> list[DemoStep]:
        if env.control_frequency != demo_frequency:
            timesteps = DemoConverter.decimate(
                demo,
                target_freq=env.control_frequency,
                original_freq=demo_frequency,
                robot=env.robot,
            ).timesteps
        else:
            timesteps = demo.timesteps
        return timesteps
