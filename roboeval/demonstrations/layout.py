"""Start-layout imposition for Play Local demos.

Cloud demos rebuild their start state with ``env.reset(seed=demo.seed)``: the
env's seeded RNG re-derives every object pose. Local (browser) demos cannot —
the layout is issued by the backend (an ordered candidate list, see
``tasks.reset_layout``) and the browser applies it verbatim. So the browser
records the applied layout in the demo metadata (``Metadata.reset_layout``,
also mirrored in the sidecar's ``raw_final_metrics.reset_layout``) and replay
tooling imposes it right after the seeded reset.

This matters beyond free bodies: goal patches have no joint, so their pose is
not in qpos and state playback alone would leave them wherever the seed put
them — the predicate would then be evaluated against the wrong target.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:  # mujoco is only needed to classify joints; keep import-light modules importable.
    import mujoco  # type: ignore
except Exception:  # pragma: no cover
    mujoco = None


def resolve_object_path(env: Any, path: str) -> Any:
    """``"cube_A"`` / ``"cubes[2]"`` -> the prop object on the env (the spec
    names props by the env's Python attribute; compiled body names are not
    stable — dm_control leaves some unnamed)."""
    if path.endswith("]") and "[" in path:
        base, index = path[:-1].split("[", 1)
        return getattr(env, base)[int(index)]
    return getattr(env, path)


def free_joint_addresses(model: Any, body_id: int) -> Optional[tuple[int, int]]:
    """(qpos_adr, dof_adr) of the body's free joint, or None when it has none."""
    if mujoco is None:
        return None
    start = int(model.body_jntadr[body_id])
    for joint in range(start, start + int(model.body_jntnum[body_id])):
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    return None


def layout_placements(layout: Any) -> list[dict]:
    """The concrete poses out of a ``reset_layout`` document (metadata or
    sidecar form). Empty when the layout carries none (legacy demos)."""
    if not isinstance(layout, dict):
        return []
    placements = layout.get("placements")
    if not isinstance(placements, list):
        return []
    out = []
    for placement in placements:
        if not isinstance(placement, dict) or not isinstance(placement.get("name"), str):
            continue
        pos = placement.get("position")
        quat = placement.get("quaternion")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            continue
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            continue
        out.append({"name": placement["name"], "position": [float(v) for v in pos], "quaternion": [float(v) for v in quat]})
    return out


def impose_reset_layout(env: Any, layout: Any) -> int:
    """Put every object named in ``layout`` at its recorded start pose.

    Free bodies get qpos written (and qvel zeroed); joint-less bodies (goal
    patches) get the MODEL body pose rewritten — what ``Body.set_position``
    does on the Python side and what the browser does too. Ends with a
    forward pass. Returns the number of objects placed; 0 for cloud demos
    (no layout) so callers can use it unconditionally.
    """
    placements = layout_placements(layout)
    if not placements:
        return 0
    model = env.mojo.model
    data = env.mojo.data
    placed = 0
    for placement in placements:
        try:
            prop = resolve_object_path(env, placement["name"])
            body_id = int(prop.body.id)
        except (AttributeError, IndexError, ValueError, TypeError):
            continue
        free = free_joint_addresses(model, body_id)
        if free is not None:
            qpos_adr, dof_adr = free
            data.qpos[qpos_adr:qpos_adr + 3] = placement["position"]
            data.qpos[qpos_adr + 3:qpos_adr + 7] = placement["quaternion"]
            data.qvel[dof_adr:dof_adr + 6] = 0.0
        else:
            model.body_pos[body_id] = np.asarray(placement["position"], dtype=np.float64)
            model.body_quat[body_id] = np.asarray(placement["quaternion"], dtype=np.float64)
        placed += 1
    env.mojo.physics.forward()
    cache = getattr(env, "_step_cache", None)
    if cache is not None:
        cache.clean()
    return placed


def reset_env_for_demo(env: Any, demo: Any) -> None:
    """``env.reset(seed=demo.seed)`` plus the demo's recorded start layout, if
    it has one. Drop-in for every replay path so local and cloud demos rebuild
    the same start state."""
    env.reset(seed=demo.seed)
    metadata = getattr(demo, "metadata", None)
    layout = getattr(metadata, "reset_layout", None)
    impose_reset_layout(env, layout)
