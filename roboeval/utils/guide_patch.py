"""Translucent, non-colliding floor markers that show a human where a task wants things.

Several cube tasks ask for a placement the operator cannot see: "10 cm apart",
"the front-left corner", "the midpoint between the two cubes". Playtesting showed
people fail those purely because the target is invisible, not because the motion
is hard. A guide patch draws the accepted region on the table without taking part
in physics, so it cannot change how a task plays out.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors

from roboeval.envs.props.items import GoalPatch

# Table surface; patches sit a hair above it so they do not z-fight with the top.
TABLE_SURFACE_Z = 0.95
GUIDE_Z_OFFSET = 0.0015
DEFAULT_ALPHA = 0.35


def guide_patch(
    mojo,
    color: str = "white",
    size: tuple[float, float] = (0.04, 0.04),
    alpha: float = DEFAULT_ALPHA,
    name: str | None = None,
) -> GoalPatch:
    """Create a translucent marker. `size` is half-extents, matching GoalPatch."""
    rgb = np.array(mcolors.to_rgb(color))
    rgba = np.append(rgb, float(alpha))
    return GoalPatch(
        mojo,
        color=rgba,
        size=size,
        thickness=0.0008,
        name=name,
        collision_enabled=False,  # never influences physics or is_colliding checks
    )


def place_guide(patch: GoalPatch, x: float, y: float) -> None:
    """Put a guide patch flat on the table at (x, y)."""
    patch.set_position(np.array([float(x), float(y), TABLE_SURFACE_Z + GUIDE_Z_OFFSET]))


def hide_guide(patch: GoalPatch) -> None:
    """Park a guide far under the table when a variant does not use it."""
    patch.set_position(np.array([0.0, 0.0, -1.0]))


def tolerance_size(tol_m: float, floor: float = 0.02) -> tuple[float, float]:
    """Half-extents that draw the actual accepted region for a +/- tol_m window."""
    half = max(float(tol_m), float(floor))
    return (half, half)


def cube_on_patch(cube, patch, half_extents, inset=0.02, rest_z=0.97):
    """Accept the inset square footprint, including tiny solver contact gaps."""
    position = np.asarray(cube.body.get_position(), dtype=float)
    center = np.asarray(patch.body.get_position(), dtype=float)
    inside = np.all(np.abs(position[:2] - center[:2]) <= np.asarray(half_extents) - inset + 1e-9)
    near_surface = abs(float(position[2]) - rest_z) <= 0.025
    contact = hasattr(patch, "collider") and cube.is_colliding(patch.collider)
    resting = contact or abs(float(position[2]) - rest_z) <= 0.008
    return bool(inside and near_surface and resting)
