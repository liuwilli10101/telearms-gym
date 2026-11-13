"""The simulator settings the TeleArms datasets were rendered and replayed with: the policy's
front-camera view, and a control rate that advances exactly 1/fps simulated seconds per action
(from the training repo's roboeval/training_contracts.py)."""
from __future__ import annotations

CAMERA_PROFILE = 'workspace_v2'


def apply_camera_profile(env, profile: str) -> None:
    """A fixed workspace view shared by dataset rendering and deployment.

    This changes only the policy's front camera, not the teleoperator cameras.
    Target-independent framing avoids leaking ground-truth object positions.
    """
    if profile == 'recorded':
        return
    if profile not in (CAMERA_PROFILE, 'workspace_v1'):
        raise ValueError(f'Unknown policy camera profile: {profile}')
    import numpy as np
    import mujoco

    model, data = env._mojo.model, env._mojo.data
    names = [model.camera(i).name for i in range(model.ncam)]
    matches = [i for i, name in enumerate(names) if name == 'top_cam' or name.endswith('/top_cam')]
    if len(matches) != 1:
        raise ValueError('workspace camera profile requires exactly one YAM top_cam')
    cid = matches[0]
    mujoco.mj_forward(model, data)
    if profile == 'workspace_v1':
        # Retained for the initial diagnostic pilot only. Its narrow crop did
        # not cover the full corpus, so new exports use workspace_v2.
        model.cam_fovy[cid] = 25.0
    else:
        # A more overhead view makes top-face letters less foreshortened.
        # Keep the full workspace, including objects moved beyond reset bounds.
        # These are fixed world coordinates, independent of task/object state.
        position = np.array([0.15, 0.0, 1.90])
        target = np.array([0.53, 0.0, 0.99])
        back = position - target
        back /= np.linalg.norm(back)
        right = np.cross([0., 0., 1.], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        rotation = np.column_stack((right, up, back))
        body = model.cam_bodyid[cid]
        body_rotation = data.xmat[body].reshape(3, 3)
        model.cam_pos[cid] = body_rotation.T @ (position - data.xpos[body])
        mujoco.mju_mat2Quat(model.cam_quat[cid], (body_rotation.T @ rotation).reshape(9))
        model.cam_fovy[cid] = 55.0
    # Maintain scene lighting while lifting the very dark robot-camera images.
    model.vis.headlight.ambient[:] = np.maximum(model.vis.headlight.ambient, 0.35)
    model.vis.headlight.diffuse[:] = np.maximum(model.vis.headlight.diffuse, 0.65)
    mujoco.mj_forward(model, data)


def set_policy_control_rate(env, fps: int) -> None:
    """Make one policy action advance exactly 1/fps simulated seconds.

    RoboEval's default rounds 500/fps substeps (30 Hz otherwise becomes
    29.41 Hz). A phase accumulator alternates 16/17 steps at 30 Hz.
    """
    if fps <= 0 or fps > 500:
        raise ValueError('policy fps must be in (0, 500]')
    env._policy_rate_phase = 0
    env._policy_rate_fps = fps


def step_policy(env, action):
    fps = env._policy_rate_fps
    env._policy_rate_phase += 500
    env._sub_steps_count, env._policy_rate_phase = divmod(env._policy_rate_phase, fps)
    return env.step(action, fast=True)
