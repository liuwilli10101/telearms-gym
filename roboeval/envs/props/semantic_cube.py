# semantic_cube.py
# ----------------
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import warnings

import numpy as np
from dm_control import mjcf                         # <- for type hint only
from pyquaternion import Quaternion                # <- for quaternion math

from roboeval.envs.props.prop import KinematicProp     # or CollidableProp
from mojo.elements import Body, Geom, MujocoElement

def quat_apply(quat_wxyz: np.ndarray | List[float],
               vec_xyz:  np.ndarray | List[float]) -> np.ndarray:
    """
    Rotate `vec_xyz` by quaternion `quat_wxyz` (w, x, y, z) ➜ new 3-vector.
    """
    q = Quaternion(w=quat_wxyz[0], x=quat_wxyz[1],
                   y=quat_wxyz[2], z=quat_wxyz[3])
    return q.rotate(vec_xyz)


# ═══════════════════════════════════════════════════════════════════════════════
#                    STANDARD FACE/GLYPH ORIENTATION CONVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# These are FIXED conventions that ensure consistent readability across all tasks.
# The glyph_dir is automatically determined from face_dir - DO NOT OVERRIDE.

FACE_GLYPH_CONVENTIONS = {
    # face_dir tuple -> (glyph_dir, view_name, description)
    (0, 0, 1): {
        "glyph_dir": np.array([1, 0, 0]),  # Glyph points toward +X (front)
        "view_name": "top",
        "description": "Top view - labels face up, readable from above",
        "camera": "rgb_head",
        "use_cases": ["horizontal arrangements", "pyramids", "flat patterns"]
    },
    (0, 0, -1): {
        "glyph_dir": np.array([1, 0, 0]),  # Glyph points toward +X 
        "view_name": "bottom",
        "description": "Bottom view - labels face down",
        "camera": "rgb_external",
        "use_cases": ["rarely used"]
    },
    (1, 0, 0): {
        "glyph_dir": np.array([0, 0, 1]),  # Glyph points up (+Z)
        "view_name": "front",
        "description": "Front view - labels face forward, readable from front",
        "camera": "rgb_external",
        "use_cases": ["forward-facing displays"]
    },
    (-1, 0, 0): {
        "glyph_dir": np.array([0, 0, 1]),  # Glyph points up (+Z)
        "view_name": "back",
        "description": "Back view - labels face back, readable from behind robot",
        "camera": "rgb_external",
        "use_cases": ["vertical stacks", "towers"]
    },
    (0, 1, 0): {
        "glyph_dir": np.array([0, 0, 1]),  # Glyph points up (+Z)
        "view_name": "left",
        "description": "Left view - labels face left side",
        "camera": "rgb_external",
        "use_cases": ["side-facing displays"]
    },
    (0, -1, 0): {
        "glyph_dir": np.array([0, 0, 1]),  # Glyph points up (+Z)
        "view_name": "right",
        "description": "Right view - labels face right side (toward camera)",
        "camera": "rgb_external",
        "use_cases": ["camera-facing labels"]
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#                    FACE DIRECTION STRING ALIASES
# ═══════════════════════════════════════════════════════════════════════════════
# These allow LLM-generated code to use string names like 'up', 'back', etc.
# instead of numeric vectors. This is a defensive programming pattern that
# catches common LLM mistakes gracefully.

FACE_DIR_ALIASES = {
    # Primary names
    'up': np.array([0.0, 0.0, 1.0]),
    'down': np.array([0.0, 0.0, -1.0]),
    'front': np.array([1.0, 0.0, 0.0]),
    'back': np.array([-1.0, 0.0, 0.0]),
    'left': np.array([0.0, 1.0, 0.0]),
    'right': np.array([0.0, -1.0, 0.0]),
    # Alternative names
    'top': np.array([0.0, 0.0, 1.0]),
    'bottom': np.array([0.0, 0.0, -1.0]),
    'forward': np.array([1.0, 0.0, 0.0]),
    'backward': np.array([-1.0, 0.0, 0.0]),
    # Axis names
    '+z': np.array([0.0, 0.0, 1.0]),
    '-z': np.array([0.0, 0.0, -1.0]),
    '+x': np.array([1.0, 0.0, 0.0]),
    '-x': np.array([-1.0, 0.0, 0.0]),
    '+y': np.array([0.0, 1.0, 0.0]),
    '-y': np.array([0.0, -1.0, 0.0]),
}


def normalize_face_dir(face_dir, default=None):
    """
    Convert face_dir to a normalized numpy array, handling string aliases.

    This is a defensive programming pattern that catches common LLM mistakes
    where string names like 'up', 'back', 'top' are used instead of numeric vectors.

    Args:
        face_dir: Can be:
            - None: returns default (or [0,0,1] if default is None)
            - str: 'up', 'down', 'front', 'back', 'left', 'right', 'top', 'bottom'
            - list/tuple/ndarray: numeric vector [x, y, z]
        default: Default value if face_dir is None

    Returns:
        Normalized numpy array [x, y, z]

    Raises:
        ValueError: If string is not a recognized alias
    """
    if face_dir is None:
        if default is not None:
            return np.asarray(default, dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    if isinstance(face_dir, str):
        key = face_dir.lower().strip()
        if key in FACE_DIR_ALIASES:
            return FACE_DIR_ALIASES[key].copy()
        raise ValueError(
            f"Unknown face_dir string '{face_dir}'. "
            f"Use one of: {list(FACE_DIR_ALIASES.keys())} "
            f"or a numeric vector like [0,0,1] or np.array([-1,0,0])"
        )

    # Assume it's a numeric array-like
    arr = np.asarray(face_dir, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > 1e-9:
        arr = arr / norm
    return arr


def get_standard_glyph_dir(face_dir) -> np.ndarray:
    """
    Get the standard glyph direction for a given face direction.

    This ensures consistent orientation conventions:
    - Top face (+Z): glyph points +X (toward robot front)
    - Front/Back/Left/Right faces: glyph points +Z (upright text)

    Args:
        face_dir: The face normal direction. Can be:
            - numpy array [x, y, z]
            - string alias: 'up', 'down', 'front', 'back', 'left', 'right', etc.

    Returns:
        The standard glyph direction for this face orientation
    """
    # Use normalize_face_dir to handle string aliases gracefully
    face_dir = normalize_face_dir(face_dir)
    
    # Snap to nearest axis-aligned direction
    abs_face = np.abs(face_dir)
    max_idx = np.argmax(abs_face)
    snapped = np.zeros(3)
    snapped[max_idx] = np.sign(face_dir[max_idx])
    
    # Look up in convention table
    key = tuple(int(x) for x in snapped)
    if key in FACE_GLYPH_CONVENTIONS:
        return FACE_GLYPH_CONVENTIONS[key]["glyph_dir"].copy()
    
    # Fallback for non-axis-aligned (should not happen in practice)
    # Default: glyph points up if face is horizontal, forward if face is vertical
    if abs(face_dir[2]) > 0.9:  # Face is roughly horizontal
        return np.array([1, 0, 0])  # Glyph toward +X
    else:  # Face is roughly vertical
        return np.array([0, 0, 1])  # Glyph up


def get_view_info(face_dir) -> dict:
    """
    Get complete view information for a face direction.

    Args:
        face_dir: The face normal direction. Can be:
            - numpy array [x, y, z]
            - string alias: 'up', 'down', 'front', 'back', 'left', 'right', etc.

    Returns:
        Dict with glyph_dir, view_name, description, camera, use_cases
    """
    # Use normalize_face_dir to handle string aliases gracefully
    face_dir = normalize_face_dir(face_dir)
    
    # Snap to nearest axis
    abs_face = np.abs(face_dir)
    max_idx = np.argmax(abs_face)
    snapped = np.zeros(3)
    snapped[max_idx] = np.sign(face_dir[max_idx])
    
    key = tuple(int(x) for x in snapped)
    if key in FACE_GLYPH_CONVENTIONS:
        return FACE_GLYPH_CONVENTIONS[key].copy()
    
    return {
        "glyph_dir": get_standard_glyph_dir(face_dir),
        "view_name": "custom",
        "description": "Non-standard face direction",
        "camera": "rgb_external",
        "use_cases": []
    }

class SemanticCube(KinematicProp):
    """
    Cube where each of the six faces carries a *unique* semantic value
    (letter, digit, colour, …) supplied via `labels`.

    Parameters expected via **kwargs when instantiating:
    ----------------------------------------------------
      xml_path : str            # absolute or relative path to the cube MJCF
      labels   : List[str]      # six strings in geom-definition order
                                #   [+X, -X, +Y, -Y, +Z, -Z]
    """

    # --------------- geom names as created by make_cube_asset.py -------
    _BASE_BODY   = "cube_root"
    _G_NAMES     = ["+x_face", "-x_face", "+y_face",
                    "-y_face", "+z_face", "-z_face"]
    _G_NORMALS   = [np.array([+1, 0, 0]), np.array([-1, 0, 0]),
                    np.array([ 0,+1, 0]), np.array([ 0,-1, 0]),
                    np.array([ 0, 0,+1]), np.array([ 0, 0,-1])]
    
    # (outward normal, glyph-up) in CUBE coordinates
    # This MUST match the orientation of textures generated by `generate_ordered_cubes.py`
    _FACE_AXES = [
        (np.array([+1, 0, 0]), np.array([0, 0,+1])),   # +x
        (np.array([-1, 0, 0]), np.array([0, 0,+1])),   # -x
        (np.array([ 0,+1, 0]), np.array([0, 0,+1])),   # +y
        (np.array([ 0,-1, 0]), np.array([0, 0,+1])),   # -y
        (np.array([ 0, 0,+1]), np.array([0,+1, 0])),   # +z
        (np.array([ 0, 0,-1]), np.array([0,-1, 0])),   # -z
    ]

    # ------------------------------------------------------------------
    # Construction-time kwargs
    # ------------------------------------------------------------------
    def _parse_kwargs(self, kwargs: Dict[str, Any]):
        self._xml_path: str      = kwargs["xml_path"]
        self._labels:  List[str] = list(kwargs["labels"])
        assert len(self._labels) == 6, "`labels` must have length 6"

    # ------------------------------------------------------------------
    @property
    def _model_path(self) -> Path:
        return Path(self._xml_path)

    
     # ------------------------------------------------------------------
    def _on_loaded(self, model: mjcf.RootElement):
        """Grab one Geom per face and remember its semantic label."""
        wrapper = MujocoElement(self._mojo, model)
        body    = Body.get(self._mojo, self._BASE_BODY, wrapper)

        # map: label  ->  (Geom-handle , local_normal)
        self._faces: Dict[str, Tuple[Geom, np.ndarray, np.ndarray]] = {}
        for g_name, lbl, (n_loc, t_loc) in zip(self._G_NAMES, self._labels, self._FACE_AXES):
            geom = Geom.get(self._mojo, g_name, body)
            self._faces[lbl] = (geom, n_loc, t_loc)

    # ------------------------------------------------------------------
    # ---------- Helper utilities exposed to the Environment -----------
    # ------------------------------------------------------------------    
    def _world_up():
        return np.array([0.0, 0.0, 1.0])

    def _cube_up_label(self, dir_vec=np.array([0, 0, 1])):
        """Return (label, cosine) of the face most aligned with +Z."""
        up_vec = dir_vec if dir_vec is not None else self._world_up()
        best, lbl = -1, None
        for n_world, tag in self.face_normals_world():   # see SemanticCube helper
            cos = np.dot(n_world / np.linalg.norm(n_world), up_vec)
            if cos > best:
                best, lbl = cos, tag
        return lbl, best
    
    # Labels whose printed glyph has 180° rotational symmetry: an upside-down
    # texture is visually indistinguishable from the upright one, so the
    # readability check treats either orientation as valid for these.
    # Asymmetric labels (F, G, R, A, B, C, ...) MUST be in the correct
    # orientation — flipped glyphs are rejected.
    _ROTATIONAL_180_SYMMETRIC = frozenset({"H", "I", "N", "O", "S", "X", "Z", "0", "1", "8"})

    def is_face_readable(
        self,
        target_label: str,
        up_tol:  float = 0.97,          # cos(14°)  – face must be ≥14° from upright
        rot_tol: float = 0.94,          # cos(20°)  – text within ±20° of expected direction
        face_dir: np.ndarray = None,    # expected face direction (default: +Z up)
        glyph_dir: np.ndarray = None,   # DEPRECATED: auto-computed from face_dir
        allow_flip: "bool | None" = None,  # None=auto from label symmetry, True/False=force
    ) -> bool:

        """
        True  ↔  `target_label` face

        1) faces the expected direction (default: +Z up)
        2) its glyph reads toward expected direction (auto-computed from face_dir)

        Args:
            target_label: The semantic label to check
            up_tol: Cosine tolerance for face direction alignment
            rot_tol: Cosine tolerance for glyph rotation alignment
            face_dir: Expected face direction (default [0,0,1] = up).
                      For back view use [-1,0,0].
            glyph_dir: DEPRECATED - ignored if provided. Glyph direction is now
                      auto-computed from face_dir using standard conventions:
                      - top view: glyph_dir=[1,0,0]
                      - front/back/left/right: glyph_dir=[0,0,1]
            allow_flip: If None (default), automatically allow 180° glyph flip
                        only for rotationally symmetric labels (H, I, O, S, X, Z, ...).
                        Pass True/False to force the behaviour for any label.
        """
        # Normalize face_dir, handling string aliases like 'up', 'back', etc.
        # This is defensive programming to catch LLM-generated code that uses
        # string names instead of numeric vectors.
        face_dir = normalize_face_dir(face_dir, default=[0.0, 0.0, 1.0])
        
        # Auto-compute glyph_dir from face_dir using standard conventions
        # (ignore any user-provided glyph_dir to prevent incorrect orientations)
        glyph_dir = get_standard_glyph_dir(face_dir)
        
        # ---------- find outward normal & glyph-top dir in WORLD -----------
        n_world = t_world = None
        for n_w, lbl in self.face_normals_world():
            if lbl == target_label:
                n_world = n_w / np.linalg.norm(n_w)
                break
        for t_w, lbl in self.face_dirs_world():
            if lbl == target_label:
                t_world = t_w / np.linalg.norm(t_w)
                break
        if n_world is None or t_world is None:
            return False   # label not found

        # ---------- 1) face must point in expected direction ----------------
        if float(np.dot(n_world, face_dir)) < up_tol:
            return False

        # ---------- 2) glyph must point in expected direction ---------------
        # project t_world into the plane perpendicular to face_dir
        t_proj = t_world - np.dot(t_world, face_dir) * face_dir
        norm = np.linalg.norm(t_proj)
        if norm < 1e-6:              # text points along face normal
            return False
        t_proj /= norm

        # project glyph_dir into same plane for comparison
        g_proj = glyph_dir - np.dot(glyph_dir, face_dir) * face_dir
        g_norm = np.linalg.norm(g_proj)
        if g_norm < 1e-6:
            return False
        g_proj /= g_norm

        cos_glyph = float(np.dot(t_proj, g_proj))
        # Auto-decide flip allowance from the label's rotational symmetry
        # unless the caller explicitly forces it.
        if allow_flip is None:
            flip_ok = str(target_label).upper() in self._ROTATIONAL_180_SYMMETRIC
        else:
            flip_ok = bool(allow_flip)
        if flip_ok:
            cos_glyph = abs(cos_glyph)
        return cos_glyph >= rot_tol

    def get_upward_label(self) -> str:
        """Convenience: label that currently faces +Z in world frame."""
        return self.get_label([0, 0, 1])
    
    def get_label(self, side="top"):
        if isinstance(side, str):
            # RoboEval coordinate system (from robot's POV):
            #   X: front (+) / back (-)
            #   Y: left (+) / right (-)  
            #   Z: up (+) / down (-)
            side_map = {
                "top": [0, 0, 1], "+z": [0, 0, 1],
                "bottom": [0, 0, -1], "-z": [0, 0, -1],
                "front": [1, 0, 0], "+x": [1, 0, 0],
                "back": [-1, 0, 0], "-x": [-1, 0, 0],
                "left": [0, 1, 0], "+y": [0, 1, 0],
                "right": [0, -1, 0], "-y": [0, -1, 0]
            }
            dir_vec = side_map.get(side.lower())
            if dir_vec is None:
                return None, 0.0
            return self._cube_up_label(dir_vec=np.array(dir_vec))
        elif isinstance(side, (np.ndarray, list, tuple)):
            return self._cube_up_label(dir_vec=np.array(side))
        return None, 0.0
            
            
    def get_orientation_for_face_and_glyph(
        self,
        face_idx: int,
        face_dir_world: np.ndarray,
        glyph_dir_world: np.ndarray = None,  # DEPRECATED: auto-computed from face_dir
    ) -> Quaternion:
        """
        Calculate the quaternion to orient a specific face and its glyph using a
        more robust two-step rotation.

        Args:
            face_idx: The index of the face to orient (0-5).
            face_dir_world: The desired world-space direction for the face's normal.
            glyph_dir_world: DEPRECATED - ignored if provided. Glyph direction is now
                           auto-computed from face_dir_world using standard conventions:
                           - top view: glyph_dir=[1,0,0]
                           - front/back/left/right: glyph_dir=[0,0,1]

        Returns:
            A pyquaternion.Quaternion that performs the required rotation.
        """
        # Normalize face_dir_world, handling string aliases like 'up', 'back', etc.
        # This is defensive programming to catch LLM-generated code that uses
        # string names instead of numeric vectors.
        face_dir_world = normalize_face_dir(face_dir_world)
        
        # Auto-compute glyph_dir from face_dir using standard conventions
        # (ignore any user-provided glyph_dir to prevent incorrect orientations)
        glyph_dir_world = get_standard_glyph_dir(face_dir_world)

        # 1. Get local vectors from _FACE_AXES
        n_local, t_local = self._FACE_AXES[face_idx]

        # 2. First rotation: Align the local normal (n_local) with the target world direction (face_dir_world)
        dot = np.dot(n_local, face_dir_world)
        if np.isclose(dot, -1.0):
            # Handle 180-degree rotation case to avoid numerical instability
            # Find an arbitrary axis perpendicular to n_local to rotate around
            axis = np.cross(n_local, [0, 0, 1])
            if np.linalg.norm(axis) < 1e-6:  # If n_local is parallel to Z axis
                axis = np.cross(n_local, [0, 1, 0])
            q1 = Quaternion(axis=axis, angle=np.pi)
        else:
            # Standard case for non-180-degree rotations
            w = 1 + dot
            xyz = np.cross(n_local, face_dir_world)
            q1 = Quaternion(w, xyz[0], xyz[1], xyz[2]).normalised

        # 3. Find where the glyph-up vector (t_local) ends up after the first rotation
        t_intermediate = q1.rotate(t_local)

        # 4. Second rotation: Rotate around face_dir_world to align t_intermediate with glyph_dir_world
        # We project the vectors onto the plane perpendicular to face_dir_world to get the angle
        proj_t_intermediate = t_intermediate - np.dot(t_intermediate, face_dir_world) * face_dir_world
        proj_t_intermediate /= np.linalg.norm(proj_t_intermediate)

        proj_glyph_dir_world = glyph_dir_world - np.dot(glyph_dir_world, face_dir_world) * face_dir_world
        proj_glyph_dir_world /= np.linalg.norm(proj_glyph_dir_world)
        
        # Angle between the projected vectors
        dot_product = np.dot(proj_t_intermediate, proj_glyph_dir_world)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Determine the sign of the angle
        cross_product = np.cross(proj_t_intermediate, proj_glyph_dir_world)
        if np.dot(face_dir_world, cross_product) < 0:
            angle = -angle
            
        q2 = Quaternion(axis=face_dir_world, angle=angle)

        # 5. Final rotation is the combination of both rotations
        return q2 * q1

    def face_normals_world(self):
        q = self.body.get_quaternion()
        for lbl, (_, n_loc, _) in self._faces.items():
            yield quat_apply(q, n_loc), lbl
            
    def face_dirs_world(self):
        """
        Yield (glyph_top_dir_world, label) for all six faces.
        """
        q = self.body.get_quaternion()
        for lbl, (_, _, t_loc) in self._faces.items():
            yield quat_apply(q, t_loc), lbl
