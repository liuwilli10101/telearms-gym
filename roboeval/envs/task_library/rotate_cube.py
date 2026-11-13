# envs/task_library/rotate_block.py
# --------------------------------
"""
Rotate a single semantic block so that a target letter faces +Z (upward).

Stage-1  – block is grasped by any gripper  
Stage-2  – target letter is facing up (≥ _UP_COS_THR) but block is still grasped  
Stage-3  – target letter is up, block rests on the table, **no gripper holding it**
"""

from __future__ import annotations
import numpy as np
from typing import List
import string

from roboeval.roboeval_env            import RoboEvalEnv
from roboeval.const                import PRESETS_PATH, HandSide
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.envs.props.tables    import Table
from roboeval.utils.env_utils      import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils  import distance
from roboeval.utils.cube_factory import ensure_cubes
from pyquaternion import Quaternion


# --------------------------------------------------------------------------- #
class RotateLetterBlockEnv(RoboEvalEnv, MetricRolloutEval):
    """Rotate a block so that target_letter: 'E' is on the top face."""

    _PRESET_PATH  = PRESETS_PATH / "bimanual_setup.yaml"      # table & robot set-up
    _UP_COS_THR   = 0.97      # ≥ ~14° accuracy
    _success_check = True
    _final_metrics: dict = {}

    _BLOCKS_POS = np.array([0.5, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.1, 0.25])

    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])
    _BLOCKS_ROT_BOUNDS_SINGLE = np.deg2rad(180)
    _BLOCKS_STEP = 0.05

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        target_letter: str = "E",
        labels: str = "EPCGDV",   # +X −X +Y −Y +Z −Z
        **kwargs,
    ):
        self._target_letter = target_letter.upper()
        self._labels        = labels
        super().__init__(**kwargs)

    # --------------------------------------------------------------------- #
    # Initialisation & reset
    # --------------------------------------------------------------------- #
    def _initialize_env(self):
        blocks_info = ensure_cubes(self._labels, universe=string.ascii_uppercase)
        # semantic block with known label order
        info = blocks_info[self._target_letter]
        self.blocks  = SemanticCube(self._mojo,
                             xml_path=str(info["xml"]),
                             labels=info["labels"])
        self.table = self._preset.get_props(Table)[0]

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=[self.blocks],
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

    def _on_reset(self):
        rng = np.random.default_rng(self.seed)
        # randomise pose & orientation on every reset
        points = get_random_points_on_plane(
            1,
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
            rng=rng,
        )
        self.blocks.set_pose(
            points[0],
            quat=Quaternion(axis=[0, 1, 0], angle=np.deg2rad(-90)).elements
            )

        q_x = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +X to +Z
        q_z = Quaternion(axis=[0, 0, 1], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +Z to +Y
        q_y = Quaternion(axis=[0, 1, 0], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +Y to +X

        # quaternion multiplication → combined rotation
        q_total = q_z * q_y * q_x
        self.blocks.body.set_quaternion(q_total.elements)

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=[self.blocks],
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)
        for s in range(1, 4):
            self._metric_stage(s, False)

    def _on_step(self):
        self._metric_step()

    # --------------------------------------------------------------------- #
    # Helper: upward-facing label & cosine
    # --------------------------------------------------------------------- #
    @staticmethod
    def _world_up():
        return np.array([0.0, 0.0, 1.0])

    def _up_label_and_cos(self):
        up = self._world_up()
        best_cos, best_lbl = -1.0, None
        for n_world, lbl in self.blocks.face_normals_world():
            c = float(np.dot(n_world / np.linalg.norm(n_world), up))
            if c > best_cos:
                best_cos, best_lbl = c, lbl
        return best_lbl, best_cos

    # --------------------------------------------------------------------- #
    # Success logic
    # --------------------------------------------------------------------- #
    def _success(self) -> bool:
        self._success_check = True

        # ---------- status flags -----------------------------------------
        lbl_up, cos_up  = self._up_label_and_cos()
        correct_letter  = lbl_up == self._target_letter
        orient_ok       = cos_up >= self._UP_COS_THR
        on_table        = self.blocks.is_colliding(self.table)
        grasped         = any(
            self.robot.is_gripper_holding_object(self.blocks, side)
            for side in self.robot.grippers
        )
        
        # ---------- staged progress --------------------------------------
        if grasped: self._metric_stage(1)                         # block lifted/grasped
        if correct_letter and orient_ok: self._metric_stage(2)    # oriented correctly
        if on_table and not grasped: self._metric_stage(3)       # released & settled

        # ---------- distance / misc metrics ------------------------------
        metrics = {"up_cos": cos_up}
        for side in self.robot.grippers:
            metrics[f"block-gripper_{side.value}"] = \
                distance(self.robot.grippers[side].body, self.blocks.body)

        self._success_check = (correct_letter and orient_ok and on_table
                               and not grasped)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=metrics,
        )
        return self._success_check

    # expose per-step info (optional)
    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})

    # optional failure – block fell off:
    def _fail(self) -> bool:
        if super()._fail():
            return True
        return False
