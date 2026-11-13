"""Colour and letter variants: a task's scene with its cubes and patches recoloured or its letter
cubes relettered (from the training repo's training/relabel.py, which made the datasets'
synthetic variants)."""
from __future__ import annotations

import functools
import random
from pathlib import Path

import numpy as np

TEXTURES = Path(__import__("roboeval").__file__).resolve().parent / "envs/xmls/assets/textures"
# Colours that have a texture asset. White (one target patch) is never remapped.
PALETTE = ("red", "blue", "green", "yellow", "purple", "orange")
# The datasets draw cubes and patches with matplotlib's named colours, so the five they
# already use keep those values. Orange is new: matplotlib's (255, 165, 0) looks yellow on a
# lit face, so it takes the orange texture's (255, 139, 41).
COLOR_RGB = {
    "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 128 / 255, 0.0),
    "yellow": (1.0, 1.0, 0.0), "purple": (128 / 255, 0.0, 128 / 255), "orange": (1.0, 139 / 255, 41 / 255),
}
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def scene_props(env) -> dict[str, list]:
    """The ColoredCube, GoalPatch and SemanticCube objects a task holds, in attribute order."""
    from roboeval.envs.props.items import ColoredCube, GoalPatch
    from roboeval.envs.props.semantic_cube import SemanticCube

    kinds = {"cubes": ColoredCube, "patches": GoalPatch, "letter_cubes": SemanticCube}
    found: dict[str, list] = {kind: [] for kind in kinds}
    seen: set[int] = set()

    def visit(value) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        else:
            for kind, cls in kinds.items():
                if isinstance(value, cls) and id(value) not in seen:
                    seen.add(id(value))
                    found[kind].append(value)

    for value in vars(env).values():
        visit(value)
    return found


def apply_colors(env, mapping: dict[str, str]) -> None:
    """Show each cube and patch whose colour is in `mapping` in its mapped colour.

    Writes only the rendered RGBA, as ColoredCube/GoalPatch.set_color do, and leaves
    their colour names alone. Call after env.reset(): tasks set colours in _on_reset.
    """
    props = scene_props(env)
    for cube in props["cubes"]:
        new = mapping.get(cube.get_color())
        if new:
            rgba = np.append(COLOR_RGB[new], 1.0)
            for geom in cube.body.geoms:
                cube._mojo.physics.bind(geom.mjcf).rgba[:] = rgba
    for patch in props["patches"]:
        new = mapping.get(color_name(patch))
        if new:
            # Keep the patch's own alpha: some targets are translucent.
            rgba = np.append(COLOR_RGB[new], patch._color[3])
            for geom in (patch.visual, getattr(patch, "collider", None)):
                if geom is not None:
                    patch._mojo.physics.bind(geom.mjcf).rgba[:] = rgba


def color_name(patch) -> str | None:
    """A patch's palette colour; patches built from raw RGBA match the nearest one."""
    name = patch.get_color()
    if name in PALETTE:
        return name
    rgb = np.asarray(patch._color[:3], dtype=float)
    distance, nearest = min((float(np.linalg.norm(rgb - np.asarray(COLOR_RGB[c]))), c) for c in PALETTE)
    return nearest if distance < 0.3 else None  # white and other off-palette targets stay as they are


def colored_geom_ids(env) -> list[list[int]]:
    """The geom ids of each palette-coloured cube and patch, which appearance randomization tints."""
    props = scene_props(env)
    ids = [[cube._mojo.physics.bind(g.mjcf).element_id for g in cube.body.geoms] for cube in props["cubes"]]
    ids += [[patch._mojo.physics.bind(g.mjcf).element_id
             for g in (patch.visual, getattr(patch, "collider", None)) if g is not None]
            for patch in props["patches"]]
    return ids


@functools.lru_cache(maxsize=None)
def letter_texture(letter: str) -> np.ndarray:
    """A letter's pixels exactly as MuJoCo stores them for a cube face."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><texture name="t" file="{TEXTURES / f"{letter}.png"}"/></asset></mujoco>')
    size = model.tex_height[0] * model.tex_width[0] * model.tex_nchannel[0]
    return model.tex_data[model.tex_adr[0]:model.tex_adr[0] + size].copy()


def apply_letters(env, mapping: dict[str, str]) -> None:
    """Print mapping[label] on every letter-cube face labelled `label`.

    `mapping` should be a bijection over the letters in the scene so no cube shows
    a letter twice. Changes model textures, so create renderers afterwards.
    """
    import mujoco

    model = env._mojo.model
    rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
    for cube in scene_props(env)["letter_cubes"]:
        for label, (geom, _normal, _up) in cube._faces.items():
            new = mapping.get(label, label)
            if new == label:
                continue
            geom_id = cube._mojo.physics.bind(geom.mjcf).element_id
            texture = model.mat_texid[model.geom_matid[geom_id]][rgb]
            pixels = letter_texture(new)
            start = model.tex_adr[texture]
            if model.tex_height[texture] * model.tex_width[texture] * model.tex_nchannel[texture] != pixels.size:
                raise ValueError(f"texture size differs for face {label} -> {new}")
            model.tex_data[start:start + pixels.size] = pixels


def letter_bijection(targets: dict[str, str], seed: str) -> dict[str, str]:
    """Extend the target letters to a bijection over A-Z, so no cube shows a letter twice."""
    rest_src = [c for c in LETTERS if c not in targets]
    rest_dst = [c for c in LETTERS if c not in targets.values()]
    random.Random(seed).shuffle(rest_dst)
    return {**targets, **dict(zip(rest_src, rest_dst))}
