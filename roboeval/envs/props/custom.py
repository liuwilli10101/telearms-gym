from pathlib import Path

from roboeval.const import ASSETS_PATH
from roboeval.envs.props.prop import KinematicProp

CUSTOM_ASSETS_PATH = ASSETS_PATH / "custom"


class Toast(KinematicProp):
    _CACHE_SITES = True

    @property
    def _model_path(self) -> Path:
        return CUSTOM_ASSETS_PATH / "toast" / "model.xml"
