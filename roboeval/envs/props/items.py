"""Different pickable items."""
from pathlib import Path
from typing import Any

from dm_control import mjcf
from mujoco_utils import mjcf_utils

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import KinematicProp
from mojo.elements import Body, MujocoElement, Joint, Site
import numpy as np
import matplotlib.colors as mcolors


class Box(KinematicProp):
    """Box."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/box/box.xml"


class Cube(KinematicProp):
    """Cube."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/cube.xml"
    
class UniqueColoredCube(KinematicProp):
    """Colored Cube."""

    _CUBE_BODY = "cube_root"
    _TOP = "+z_purple"
    _BOTTOM = "-z_orange"
    _FRONT = "+y_blue"
    _BACK = "-y_yellow"
    _LEFT = "-x_green"
    _RIGHT = "+x_red"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/unique_colored_cube.xml"
    
    def _on_loaded(self, model: mjcf.RootElement):
        """On loaded callback to set the body."""
        colored_cube = MujocoElement(self._mojo, model)
        self.body = Body.get(self._mojo, "colored_cube", colored_cube)
        self.top = self.body.geoms[self._TOP]
        self.bottom = self.body.geoms[self._BOTTOM]
        self.front = self.body.geoms[self._FRONT]
        self.back = self.body.geoms[self._BACK]
        self.left = self.body.geoms[self._LEFT]
        self.right = self.body.geoms[self._RIGHT]
        
        self.faces = {
            "top_purple": self.top,
            "bottom_orange": self.bottom,
            "front_blue": self.front,
            "back_yellow": self.back,
            "left_green": self.left,
            "right_red": self.right
        }
        
class ColoredCube(KinematicProp):
    """Colored Cube."""

    color = "red"  # Default color, can be overridden

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/cube.xml"
    
    def get_color(self) -> str:
        """
        Get the color of the cube.

        Returns:
            str: The color of the cube.
        """
        return self.color

    def set_color(self, color: str) -> None:
        """
        Set the color of the cube.

        Args:
            color (str): The color to set for the cube.
        """
        # Convert color name to RGB values
        self.color = color
        rgb = np.array(mcolors.to_rgb(color))
        # Set the color for all geoms in the cube body
        for g in self.body.geoms:
            rgba = np.append(rgb, 1.0)
            # (a) set the mjcf template value
            g.rgba = rgba
            # (b) update runtime (model + data) – bind to physics and overwrite
            self._mojo.physics.bind(g.mjcf).rgba[:] = rgba
    
class Rod(KinematicProp):
    """Rod."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/rod.xml"

class LargeCube(KinematicProp):
    """Large Cube for Two Hand Effector Manipulation."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/large_cube.xml"
    
class SmallCube(KinematicProp):
    """Small Cube to mimic dust-like particles."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cube/small_cube.xml"
    
class BreakfastTray(KinematicProp):
    """Breakfast Tray."""
    _BODY = "wood_tray"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/breakfast_tray/wood_tray.xml"
    
    def _on_loaded(self, model): 
        breakfast_tray = MujocoElement(self._mojo, model)   
        self.body = Body.get(self._mojo, self._BODY, breakfast_tray).geoms[-1]

class LighterBreakfastTray(KinematicProp):
    _BODY = "wood_tray"

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/breakfast_tray/lighter_wood_tray.xml"
    
    def _on_loaded(self, model): 
        breakfast_tray = MujocoElement(self._mojo, model)   
        self.body = Body.get(self._mojo, self._BODY, breakfast_tray).geoms[-1]
    
class Book(KinematicProp):
    """Book"""

    @property 
    def _model_path(self) -> Path: 
        return ASSETS_PATH / "props/book/book.xml"


class Sandwich(KinematicProp):
    """Sandwich."""

    _COLLIDER = "collider"
    _COLLIDER_ROUNDED = "collider_rounded"

    _NORMAL_MODEL = "sandwich.xml"
    _TOASTED_MODEL = "sandwich_toasted.xml"

    @property
    def _model_path(self) -> Path:
        model = self._TOASTED_MODEL if self.toasted else self._NORMAL_MODEL
        return ASSETS_PATH / "props/sandwich" / model

    def _parse_kwargs(self, kwargs: dict[str, Any]):
        self.toasted = kwargs.get("toasted", False)
        self.rounded_collider = kwargs.get("rounded_collider", False)

    def _on_loaded(self, model: mjcf.RootElement):
        collider_to_remove = (
            self._COLLIDER if self.rounded_collider else self._COLLIDER_ROUNDED
        )
        mjcf_utils.safe_find(model, "geom", collider_to_remove).remove()


class Wine(KinematicProp):
    """Wine."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/wine/wine.xml"


class Detergent(KinematicProp):
    """Detergent."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/detergent/detergent.xml"


class Soap(KinematicProp):
    """Soap."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/soap/soap.xml"


class Beer(KinematicProp):
    """Beer."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/beer.xml"


class Cereals(KinematicProp):
    """Cereals."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/cereal.xml"


class Crisps(KinematicProp):
    """Crisps."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/crisps.xml"


class Ketchup(KinematicProp):
    """Ketchup."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/ketchup.xml"


class Mustard(KinematicProp):
    """Mustard."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/mustard.xml"


class Soda(KinematicProp):
    """Soda."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/groceries/soda.xml"


class GoalPatch:
    """
    Procedural goal patch/target zone asset.
    
    Creates a thin flat patch on the table surface using Body/Geom primitives.
    This is NOT a KinematicProp - it's a static visual/collision marker.
    
    Usage:
        patch = GoalPatch(mojo, color="red", size=(0.07, 0.07))
        patch.set_position(np.array([0.5, 0.2, 0.95]))  # Table surface
        
        # Check if block is on patch
        if block.is_colliding(patch):
            ...
    
    Parameters:
        mojo: The Mojo instance
        color: Color name (str) or RGBA array. Default "red"
        size: (half_x, half_y) HALF-EXTENTS in meters, passed straight to the BOX
            geom. The patch you see is TWICE this in each direction: the default
            (0.07, 0.07) draws a 14cm x 14cm square, and (0.1, 0.1) draws 20cm x
            20cm. Several tasks read this as a full width, halved it again, and
            ended up checking a region a quarter the area of the one drawn -- a
            cube placed on the visible patch then failed. When you write a success
            check against a patch, compare to `patch.half_extents`, not to `size`
            scaled by anything.
        thickness: Height of patch in meters. Default 0.001 (1mm)
        name: Optional name for the patch body/geom
        collision_enabled: Whether patch should detect collisions. Default True
    """
    
    # Default patch dimensions (HALF-extents: this draws 14cm x 14cm)
    DEFAULT_SIZE = (0.07, 0.07)
    DEFAULT_THICKNESS = 0.001    # 1mm thin
    
    # Table surface Z coordinate
    TABLE_SURFACE_Z = 0.95
    
    # Workspace bounds (same as code generation constraints)
    WORKSPACE_X_MIN, WORKSPACE_X_MAX = 0.35, 0.75
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX = -0.30, 0.30
    
    def __init__(
        self,
        mojo,
        color: str | np.ndarray = "red",
        size: tuple[float, float] = None,
        thickness: float = None,
        name: str = None,
        collision_enabled: bool = True,
    ):
        from mojo.elements import Body, Geom
        from mojo.elements.consts import GeomType
        
        self._mojo = mojo
        self._name = name or f"goal_patch_{id(self)}"
        
        # Parse color
        if isinstance(color, str):
            self._color_name = color
            rgb = np.array(mcolors.to_rgb(color))
            self._color = np.append(rgb, 1.0)  # RGBA
        else:
            self._color_name = "custom"
            self._color = np.array(color) if len(color) == 4 else np.append(color, 1.0)
        
        # Parse size
        size = size or self.DEFAULT_SIZE
        thickness = thickness or self.DEFAULT_THICKNESS
        self._size = np.array([size[0], size[1], thickness])  # half-extents for BOX
        # Exposed so success checks can ask the patch how big it is instead of
        # re-deriving it from the constructor argument and getting the factor wrong.
        self.half_extents = np.array([size[0], size[1]], dtype=float)
        
        # Create the body
        self.body: Body = Body.create(self._mojo)
        
        # Create collision geom (group=3 for collision detection)
        if collision_enabled:
            self.collider: Geom = Geom.create(
                self._mojo,
                parent=self.body,
                geom_type=GeomType.BOX,
                size=self._size,
                color=self._color,
                group=3,  # Collision group
            )
            self.collider.mjcf.name = f"{self._name}_collider"
        else:
            self.collider = None
        
        # Create visual geom (group=2 for rendering)
        self.visual: Geom = Geom.create(
            self._mojo,
            parent=self.body,
            geom_type=GeomType.BOX,
            size=self._size,
            color=self._color,
            group=2,  # Visual group
        )
        self.visual.mjcf.name = f"{self._name}_visual"
        # Render groups do not disable MuJoCo contacts. Only the optional
        # collider should support objects; visual-only targets must not lift
        # cubes off the table and invalidate table-contact success checks.
        self.visual.mjcf.contype = 0
        self.visual.mjcf.conaffinity = 0
        
        # Cache colliders for is_colliding compatibility
        self.colliders = [self.collider] if self.collider else []
        
        # Default position (will be set by user)
        self._position = np.array([0.55, 0.0, self.TABLE_SURFACE_Z])
    
    def set_position(self, position: np.ndarray):
        """
        Set the position of the goal patch.
        
        Args:
            position: [x, y, z] position. Z should be ~0.95 for table surface.
        """
        self._position = np.array(position)
        self.body.set_position(self._position)
    
    def get_position(self) -> np.ndarray:
        """Get the current position of the goal patch."""
        return np.array(self.body.get_position())
    
    def set_color(self, color: str | np.ndarray):
        """
        Change the color of the goal patch.
        
        Args:
            color: Color name (str) or RGBA array
        """
        if isinstance(color, str):
            self._color_name = color
            rgb = np.array(mcolors.to_rgb(color))
            self._color = np.append(rgb, 1.0)
        else:
            self._color_name = "custom"
            self._color = np.array(color) if len(color) == 4 else np.append(color, 1.0)
        
        # Update visual geom
        self.visual.rgba = self._color
        self._mojo.physics.bind(self.visual.mjcf).rgba[:] = self._color
        
        # Update collider if exists
        if self.collider:
            self.collider.rgba = self._color
            self._mojo.physics.bind(self.collider.mjcf).rgba[:] = self._color
    
    def get_color(self) -> str:
        """Get the color name of the patch."""
        return self._color_name
    
    def is_colliding(self, other) -> bool:
        """
        Check if another object is colliding with this patch.
        
        Args:
            other: A Prop, Body, Geom, or object with colliders
            
        Returns:
            bool: True if collision detected
        """
        from roboeval.utils.physics_utils import has_collided_collections, get_colliders
        
        if not self.collider:
            return False
        
        other_colliders = get_colliders(other)
        return has_collided_collections(
            self._mojo.physics, self.colliders, other_colliders
        )
    
    def in_workspace(self) -> bool:
        """Check if patch position is within workspace bounds."""
        pos = self._position
        return (
            self.WORKSPACE_X_MIN <= pos[0] <= self.WORKSPACE_X_MAX and
            self.WORKSPACE_Y_MIN <= pos[1] <= self.WORKSPACE_Y_MAX
        )
    
    @classmethod
    def create_colored_patches(
        cls,
        mojo,
        colors: list[str],
        positions: list[np.ndarray] = None,
        size: tuple[float, float] = None,
    ) -> list["GoalPatch"]:
        """
        Factory method to create multiple colored patches.
        
        Args:
            mojo: The Mojo instance
            colors: List of color names (e.g., ["red", "blue", "green"])
            positions: Optional list of positions. If None, patches are arranged in a row.
            size: Optional size tuple (width, depth)
            
        Returns:
            List of GoalPatch instances
        """
        patches = []
        
        # Default positions: arrange in a row centered at workspace
        if positions is None:
            n = len(colors)
            spacing = 0.12  # 12cm between patch centers
            start_y = (n - 1) * spacing / 2
            positions = [
                np.array([0.55, start_y - i * spacing, cls.TABLE_SURFACE_Z])
                for i in range(n)
            ]
        
        for i, color in enumerate(colors):
            patch = cls(mojo, color=color, size=size, name=f"{color}_target")
            if i < len(positions):
                patch.set_position(positions[i])
            patches.append(patch)
        
        return patches
