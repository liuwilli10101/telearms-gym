#!/usr/bin/env python3
"""
make_cube_asset.py
──────────────────
Generate an MJCF cube where every face owns a distinct texture (PNG).

Example
-------
>>> make_cube(
...     out_file="cube_P.xml",
...     faces=["P.png", "Q.png", "R.png", "S.png", "T.png", "U.png"]
... )
"""

import math
import os
import pathlib
import random
import string
import textwrap
from typing import List, Union

# ──────────────────────────────────────────────────────────────────────
# Dynamic path resolution - find RoboEval root directory
# ──────────────────────────────────────────────────────────────────────
_THIS_FILE = pathlib.Path(__file__).resolve()
ROBOEVAL_ROOT = _THIS_FILE.parent.parent.parent  # roboeval/utils/generate_ordered_cubes.py -> RoboEval/

# Default directories relative to RoboEval root
# Texture dir is written into the XML as a relative path (from the cube XML's
# location) so the generated assets stay portable across machines/checkouts.
DEFAULT_TEXTURE_DIR = "../../../../assets/textures"
DEFAULT_OUTPUT_DIR = str(ROBOEVAL_ROOT / "roboeval/envs/xmls/props/cube/ordered_cubes")

# ──────────────────────────────────────────────────────────────────────
# 1. MJCF template  (angles already declared as *radian*)
# ──────────────────────────────────────────────────────────────────────
TEMPLATE = textwrap.dedent("""\
<?xml version="1.0"?>
<mujoco model="cube">
  <compiler angle="radian"
            texturedir="{texture_dir}"
            meshdir="{mesh_dir}"/>
  <!-- Visual-only face geoms (no collision) -->
  <default>
    <default class="side">
      <geom type="box"
            size="{half} {half} {thick}"
            contype="0"
            conaffinity="0"
            density="0"
            mass="0"/>
    </default>
  </default>

  <!-- six textures + six materials -->
  <asset>
        {textures_block}
        {materials_block}
  </asset>

  <!-- six independent geoms, one per face -->
  <worldbody>
    <body name="cube_root">
{geoms_block}
    </body>
  </worldbody>
</mujoco>
""")

# ──────────────────────────────────────────────────────────────────────
# 2. Per-face offsets (now **radians**)
#    Order: +X, −X, +Y, −Y, +Z, −Z
#            nx  ny  nz     rx     ry     rz
# ──────────────────────────────────────────────────────────────────────
deg2rad = math.radians
OFFSETS = [
    ("+x",  1,  0,  0,  deg2rad( 90),  deg2rad( 90),  deg2rad(0)),   # right
    ("-x", -1,  0,  0,  deg2rad( 90),  deg2rad(-90),  deg2rad(0)),   # left
    ("+y",  0,  1,  0,  deg2rad( 90),  deg2rad(  0),  deg2rad(0)),   # front
    ("-y",  0, -1,  0,  deg2rad( 90),  deg2rad(  0),  deg2rad(0)),   # back
    ("+z",  0,  0,  1,  deg2rad(  0),  deg2rad(  0),  deg2rad(0)),   # top
    ("-z",  0,  0, -1,  deg2rad(180),  deg2rad(  0),  deg2rad(0)),   # bottom
]

# ──────────────────────────────────────────────────────────────────────
def make_cube(
    out_file: str,
    faces: List[str],
    *,
    size: float = 0.04,
    thickness: float = 0.001,
    texture_dir: str = None,
    mesh_dir: str = ""
) -> None:
    """
    Create an MJCF cube with per-face textures.

    Parameters
    ----------
    out_file     : destination .xml path
    faces        : list of **exactly six** PNG files in the order
                   [+X, −X, +Y, −Y, +Z, −Z]; every entry must be unique
    size         : cube edge length (m)
    thickness    : thickness of each face geom
    texture_dir  : path passed to <compiler texturedir="…"> (None = use default)
    mesh_dir     : path passed to <compiler meshdir="…">
    """
    assert len(faces) == 6,  "Provide exactly six texture filenames."
    assert len(set(faces)) == 6, "Textures must be unique per face."
    
    # Use default texture directory if not specified
    if texture_dir is None:
        texture_dir = DEFAULT_TEXTURE_DIR

    half  = size / 2
    thick = thickness

    # ── 1) <texture> + <material> blocks ─────────────────────────────
    tex_tpl = '    <texture name="T{idx}" file="{fname}"/>'
    mat_tpl = ('    <material name="M{idx}" reflectance="0.3" shininess="0.6" '
               'texture="T{idx}"/>')

    textures_block  = "\n".join(tex_tpl.format(idx=i, fname=f)
                                for i, f in enumerate(faces))
    materials_block = "\n".join(mat_tpl.format(idx=i) for i in range(6))

    # ── 2) collision geom + six face geoms (radian Euler) ────────────
    # Collision geom (group 3)
    collision_geom = (
        '      <geom name="collision" group="3" type="box" '
        f'size="{half:.4f} {half:.4f} {half:.4f}" '
        'density="1500" friction="1.5 0.1 0.01" '
        'solimp="0.95 0.99 0.001" solref="0.005 1" condim="3" '
        'rgba="0 0 0 0" contype="1" conaffinity="1"/>'
    )
    
    # Visual face geoms (group 2)
    geom_tpl = (
        '      <geom name="{name}" class="side" group="2" material="M{idx}" '
        'pos="{px:.4f} {py:.4f} {pz:.4f}" '
        'euler="{rx:.4f} {ry:.4f} {rz:.4f}"/>'
    )
    face_geoms = "\n".join(
        geom_tpl.format(
            name=f"{tag}_face", idx=i,
            px=sx*half, py=sy*half, pz=sz*half,
            rx=rx, ry=ry, rz=rz
        )
        for i, (tag, sx, sy, sz, rx, ry, rz) in enumerate(OFFSETS)
    )
    
    # Combine collision + face geoms
    geoms_block = collision_geom + "\n" + face_geoms

    # ── 3) fill & write template ─────────────────────────────────────
    xml_txt = TEMPLATE.format(
        model_name=pathlib.Path(out_file).stem,
        texture_dir=texture_dir,
        mesh_dir=mesh_dir,
        half=half,
        thick=thick,
        textures_block=textures_block,
        materials_block=materials_block,
        geoms_block=geoms_block,
    )

    out_path = pathlib.Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_txt)
    print(f"✔  wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Helper: batch-generate cubes spelling a word
# ──────────────────────────────────────────────────────────────────────
def ordered_cubes(
    word: str = "PIG",
    *,
    universe: Union[str, List[str]] = string.ascii_uppercase,
    tex_dir: str = None,
    output_dir: str = None
) -> None:
    """
    For every character in `word`, make one cube whose +X face shows that
    character, with the five remaining faces filled by random *other* symbols.
    
    Parameters
    ----------
    word       : word to spell with cubes (each character gets one cube)
    universe   : set of characters to use for non-primary faces
    tex_dir    : texture directory path (None = use default)
    output_dir : output directory for XML files (None = use default)
    """
    # Use defaults if not specified
    if tex_dir is None:
        tex_dir = DEFAULT_TEXTURE_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    for ch in word:
        others = [c for c in universe if c != ch]
        random.shuffle(others)
        faces = [f"{ch}.png"] + [f"{c}.png" for c in others[:5]]
        out_xml = f"{output_dir}/{word}/cube_{ch}.xml"
        make_cube(out_xml, faces, texture_dir=tex_dir)


# ──────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ordered_cubes("PIG")                        # alphabet demo
    ordered_cubes("123", universe="0123456789") # numeric demo
