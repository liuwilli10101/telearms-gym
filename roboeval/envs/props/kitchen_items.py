"""Kitchen prop classes for pick-place, stacking, and arrangement tasks."""
from pathlib import Path

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import KinematicProp


class Plate(KinematicProp):
    """Plate."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/plate/plate.xml"


class Mug(KinematicProp):
    """Mug."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/mug/mug.xml"


class Pan(KinematicProp):
    """Pan."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/pan/pan.xml"


class Pot(KinematicProp):
    """Pot (saucepan model)."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/saucepan/saucepan.xml"


class Spatula(KinematicProp):
    """Spatula."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/spatula/spatula.xml"


class Fork(KinematicProp):
    """Fork."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/fork/fork.xml"


class Knife(KinematicProp):
    """Knife."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/knife/knife.xml"


class Spoon(KinematicProp):
    """Spoon."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/cutlery/spoon/spoon.xml"


class Apple(KinematicProp):
    """Apple."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/apple/apple.xml"
