
"""
ArrangeShapeEnv
===============

Spawn N identical block props on the table.  
Episode succeeds once the (x,y) positions of the block centres approximate a
designer-specified 2-D shape within a tolerance `tol`.

Supported shapes
----------------
'line'     – least-squares straight line (max orthogonal dev < tol)  
'circle'   – best-fit circle (peak-to-peak radius spread   < tol)  
'pyramid'  – triangular footprint with rows N, N-1, …, 1  (spacing dev < tol)

Add your own shape by writing a new `_myshape_ok` function and adding it to
`_SHAPE_CHECKS`.
"""
from __future__ import annotations
from abc import ABC

import numpy as np
import math
from itertools import combinations
from typing import Callable, Dict, List

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import Cube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval


# --------------------------------------------------------------------------- #
class ArrangeShapeEnv(RoboEvalEnv, MetricRolloutEval, ABC):
    """
    Arrange the spawned blocks so their XY centers form a target geometric shape
    (configured by the `shape` argument, e.g. 'line', 'circle', 'pyramid', ...), within
    the geometric tolerance `tol`. Success is purely the shape match of the block centers
    for the selected shape; no specific colors, ordering, or release state are required.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"    # contains table & robot
    _success_check: bool = True       # updated every step
    _final_metrics:  dict = {}        # buffer exposed via _get_task_info()

    # --------------------------------------------------------------------- #
    def __init__(self,
                 *,
                 num_blocks: int = 5,
                 shape: str = "line",
                 tol: float = 0.02,
                 **gym_kwargs):
        """
        Parameters
        ----------
        num_blocks : int      how many block props to spawn
        shape     : str      one of 'line', 'circle', 'pyramid', …
        tol       : float    geometric tolerance (metres)
        """
        self._n_blocks   = num_blocks
        self._shape     = shape.lower()
        self._tol       = tol
        super().__init__(**gym_kwargs)

    # --------------------------------------------------------------------- #
    # Environment construction                                              #
    # --------------------------------------------------------------------- #
    def _initialize_env(self):
        # 1) spawn blocks ----------------------------------------------------
        self.blocks = [Cube(self._mojo) for _ in range(self._n_blocks)]

        # 2) grab table from YAML preset (one instance expected)
        self.table = self._preset.get_props(Table)[0]

        # 3) init fine-grained metric buffers ------------------------------
        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

    # --------------------------------------------------------------------- #
    # Episode reset                                                         #
    # --------------------------------------------------------------------- #
    def _on_reset(self):
        # scatter blocks on table top (deterministic via self.seed)
        rng = np.random.default_rng(self.seed)
        points = get_random_points_on_plane(
            self._n_blocks,
            np.array([0.55, 0.0, 1.0]),            # centre pos
            np.array([0.12, 0.25]),                # extents
            0.10,
            rng=rng,
        )
        for block, p in zip(self.blocks, points):
            block.set_pose(p)

        # re-initialise metric buffers
        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

        # two high-level stages: (1) any block placed, (2) shape achieved
        for idx in range(1, 3): self._metric_stage(idx, False)

    def _on_step(self):
        self._metric_step()

    # --------------------------------------------------------------------- #
    # Success check                                                         #
    # --------------------------------------------------------------------- #
    def _success(self) -> bool:
        self._success_check = True  # reset flag
        # stage 1 → at least one block not grasped & resting on table
        grasped = any(self.robot.is_gripper_holding_object(c, side) for c in self.blocks for side in self.robot.grippers)

        # geometric goal ---------------------------------------------------
        centres = np.array([c.body.get_position()[:2] for c in self.blocks])  # (N,2)
        shape_ok = self._SHAPE_CHECKS[self._shape](centres, self._tol)

        if grasped: self._metric_stage(1)
        if shape_ok: self._metric_stage(2)

        self._success_check = shape_ok
        self._final_metrics = self._metric_finalize(
            success_flag=shape_ok,
            target_distance={"shape_deviation": float(shape_ok == False)},
        )
        return shape_ok

    # --------------------------------------------------------------------- #
    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # --------------------------------------------------------------------- #
    # ----------  Shape-specific helper functions  ------------------------ #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _line_ok(pts: np.ndarray, tol: float) -> bool:
        """
        “line” means all blocks share (approximately) the same x
        coordinate or the same y coordinate; we allow variations along **y or x** (left–right or back-front).
        """
        return float(np.ptp(pts[:, 0])) < tol or float(np.ptp(pts[:, 1])) < tol

    @staticmethod
    def _circle_ok(pts: np.ndarray, tol: float) -> bool:
        """Fit Taubin circle; require radial peak-to-peak spread < tol."""
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([2*x, 2*y, np.ones_like(x)])
        c, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
        xc, yc, c0 = c
        r_est = math.sqrt(c0 + xc**2 + yc**2)
        radii = np.sqrt((x - xc)**2 + (y - yc)**2)
        return float(radii.max() - radii.min()) < tol

    @staticmethod
    def _pyramid_ok(pts: np.ndarray, tol: float) -> bool:
        """
        Footprint = isosceles triangle:
        row counts N, N-1, …, 1  (N determined by total #blocks)
        """
        N = len(pts)
        rows = int((math.sqrt(8*N + 1) - 1) // 2)
        if rows * (rows + 1) // 2 != N:
            return False

        # ── cluster rows by Y coordinate ---------------------------------
        ys = np.sort(pts[:, 1])
        gaps = np.diff(ys)
        thresh = gaps.mean() + 2*gaps.std() if len(gaps) else tol
        splits = np.where(gaps > thresh)[0] + 1
        row_vals = np.split(ys, splits)
        if len(row_vals) != rows:
            return False
        if sorted([len(r) for r in row_vals], reverse=True) != list(range(rows, 0, -1)):
            return False

        # ── inside each row: X equally spaced & centred ------------------
        ok = True
        for y_row in row_vals:
            row_pts = pts[np.isclose(pts[:, 1], y_row.mean(), atol=tol)]
            if len(row_pts) <= 1:
                continue
            xs = np.sort(row_pts[:, 0])
            step = np.diff(xs)
            ok &= float(step.max() - step.min()) < tol
        return ok

    # --------------------------------------------------------------------- #
    # Registry – extend with new shapes here                                #
    # --------------------------------------------------------------------- #
    _SHAPE_CHECKS: Dict[str, Callable[[np.ndarray, float], bool]] = {
        "line":    _line_ok.__func__,
        "circle":  _circle_ok.__func__,
        "pyramid": _pyramid_ok.__func__,
    }
