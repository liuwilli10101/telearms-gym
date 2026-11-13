# envs/task_library/arrange_variants.py
# -------------------------------------
"""
Extra “Arrange-*” manipulation tasks that reuse the generic ArrangeWordEnv
infrastructure but enforce different ordering semantics.
"""
from __future__ import annotations
from abc import ABC

import numpy as np
from typing import Dict
import re
import string

from pyquaternion import Quaternion

from roboeval.roboeval_env           import RoboEvalEnv
from roboeval.const               import PRESETS_PATH, HandSide
from roboeval.envs.props.items    import Cube
from roboeval.envs.props.tables   import Table
from roboeval.utils.env_utils     import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

# --------------------------------------------------------------------------- #
#  ArrangeAlphabeticalOrderEnv  – alphabetical A-B-C… left→right.
# --------------------------------------------------------------------------- #
class ArrangeAlphabeticalOrderEnv(RoboEvalEnv, MetricRolloutEval, ABC):
    """Arrange blocks so that their up-faces read A,B,C… in natural order."""
    
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"

    _X_ALIGN_TOL     = 0.03      # metres   (horizontal alignment)
    _UP_COS_THRESH   = 0.95      # top face within ~18°
    _success_check   = True      # flag for success check
    _final_metrics   = {}        # metrics to expose at the end of the episode


   # ------------------------------------------------------------------
    def __init__(self, *, letters: str = "ABC", **kwargs):
        self._letters = letters.upper()
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # 1)  build / load blocks
    # ------------------------------------------------------------------
    def _initialize_env(self):
        blocks_info = ensure_cubes(self._letters, universe=string.ascii_uppercase)

        self.blocks = []
        for ch in self._letters:
            info = blocks_info[ch]
            self.blocks.append(
                SemanticCube(self._mojo,
                             xml_path=str(info["xml"]),
                             labels=info["labels"])
            )

        self.table = self._preset.get_props(Table)[0]

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True,  slip_objects=self.blocks,
                          robot=self.robot,     slip_sample_window=20,
                          track_collisions=True)

    # ------------------------------------------------------------------
    # 2)  reset: scatter randomly on the table
    # ------------------------------------------------------------------
    def _on_reset(self):
        rng = np.random.default_rng(self.seed)
        pts = get_random_points_on_plane(
            len(self.blocks), np.array([0.5, 0, 1]),
            np.array([0.12, 0.25]), 0.12, rng=rng,
        )

        for block, p in zip(self.blocks, pts):
            block.set_pose(p)
            q_x = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +X to +Z
            q_z = Quaternion(axis=[0, 0, 1], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +Z to +Y
            q_y = Quaternion(axis=[0, 1, 0], angle=np.deg2rad(rng.uniform(0, 360)))  # rotate +Y to +X
  
            # quaternion multiplication → combined rotation
            q_total = q_z * q_y * q_x
            block.body.set_quaternion(q_total.elements)

        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True,  slip_objects=self.blocks,
                          robot=self.robot,     slip_sample_window=20,
                          track_collisions=True)
        for s in range(1, 6): self._metric_stage(s, False)

    # ------------------------------------------------------------------
    # 4)  SUCCESS logic + metrics
    # ------------------------------------------------------------------
    def _success(self) -> bool:
        self._success_check = True

        # ---- (a) order blocks by **Y** (left→right) -------------------
        blocks_y_sorted = sorted(self.blocks,
                                key=lambda c: c.body.get_position()[1], reverse=True)
        y_sorted_ok = np.all(np.diff([c.body.get_position()[1]
                                      for c in blocks_y_sorted]) < 0)

        # ---- (b) horizontal alignment  (small X spread) --------------
        x_vals = [c.body.get_position()[0] for c in blocks_y_sorted]
        x_ptp  = float(np.ptp(x_vals))
        x_align_ok = x_ptp < self._X_ALIGN_TOL

        # ---- (c) upward labels & orientation quality -----------------
        up_labels, cosines, readable_flags = [], [], []
        for block in blocks_y_sorted:
            lbl, cos = block.get_label(side="top")
            up_labels.append(lbl)
            cosines.append(cos)
            readable_flags.append(block.is_face_readable(lbl))
            print(f"block {lbl}: up cosine = {cos:.3f}, readable = {readable_flags[-1]}")
        up_ok     = min(cosines) >= self._UP_COS_THRESH

        # ––––– print debug info –––––––––––––––––––––––––––––––––––––––
        current_letter = "".join(up_labels)
        sorted_letters = "".join(sorted(up_labels))
        print(f"Current order: {current_letter}, sorted: {sorted_letters}")

        alphabetical   = "".join(up_labels) == "".join(sorted(up_labels))
        readable_ok = all(readable_flags)
        single_block_grasped = any(
            self.robot.is_gripper_holding_object(block, side)
            for side in self.robot.grippers
            for block in blocks_y_sorted
        )

        # ––––– stage bookkeeping –––––––––––
        if single_block_grasped: self._metric_stage(1) # at least one block grasped
        if y_sorted_ok: self._metric_stage(2) # no overlapping blocks in the y-axes (back-to-front)
        if x_align_ok:   self._metric_stage(3) # horizontal alignment (all blocks in a line left-to-right)
        if alphabetical: self._metric_stage(4) # alphabetical order (A,B,C…)
        if readable_ok: self._metric_stage(5) # all blocks readable (glyphs facing +X, upright)

        # ––––– distance tracking ––––––––––
        distances = {}  
        for idx, block in enumerate(blocks_y_sorted):
            for side in self.robot.grippers:
                distances[f"block_{idx}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, block.body)
                if self.robot.is_gripper_holding_object(block, side):
                    self._success_check = False
        distances.update({
            "horizontal alignment": x_ptp,
        })

        # ---- overall success ----------------------------------------
        grasp_free = self._success_check  # True iff no gripper is holding any block
        print(f"[diag] alpha={alphabetical} x_align={x_align_ok} (ptp={x_ptp:.3f}) "
              f"y_sort={y_sorted_ok} readable={readable_ok} grasp_free={grasp_free}")
        self._success_check &= alphabetical and x_align_ok and y_sorted_ok and readable_ok

        # ---- fine-grained metrics -----------------------------------
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=distances,
        )
        return self._success_check

    def _on_step(self):
        self._metric_step()
        
    def _get_task_info(self):
        """Expose metrics every step (optional) or only at episode end."""
        return getattr(self, "_final_metrics", {})

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------
    @staticmethod
    def _world_up():
        return np.array([0.0, 0.0, 1.0])

    def _block_up_label(self, block):
        """Return (label, cosine) of the face most aligned with +Z."""
        up_vec = self._world_up()
        best, lbl = -1, None
        for n_world, tag in block.face_normals_world():   # see Semanticblock helper
            cos = np.dot(n_world / np.linalg.norm(n_world), up_vec)
            if cos > best:
                best, lbl = cos, tag
        return lbl, best
    
    @staticmethod
    def is_face_readable(
        block,
        target_label: str,
        up_tol:  float = 0.97,          # cos(14°)  – face must be ≥14° from upright
        rot_tol: float = 0.94,          # cos(20°)  – text within ±20° of +X
    ) -> bool:
        """
        True  ↔  `target_label` face

        1) faces (almost) straight up  (+Z, tolerance `up_tol`)
        2) its glyph reads toward +X   (in-plane, tolerance `rot_tol`)
        """

        # ---------- find outward normal & glyph-top dir in WORLD -----------
        n_world = t_world = None
        for n_w, lbl in block.face_normals_world():
            if lbl == target_label:
                n_world = n_w / np.linalg.norm(n_w)
                break
        for t_w, lbl in block.face_dirs_world():
            if lbl == target_label:
                t_world = t_w / np.linalg.norm(t_w)
                break
        if n_world is None or t_world is None:
            return False   # label not found  (shouldn’t happen)

        # ---------- 1) face must look upward -------------------------------
        if float(np.dot(n_world, [0, 0, 1])) < up_tol:
            return False

        # ---------- 2) glyph must read toward +X ---------------------------
        # project t_world into the horizontal plane (remove Z component)
        t_xy = t_world - np.dot(t_world, [0, 0, 1]) * np.array([0, 0, 1])
        norm = np.linalg.norm(t_xy)
        if norm < 1e-6:              # text points straight up/down → unreadable
            return False
        t_xy /= norm                 # normalise  (now purely XY)

        return float(np.dot(t_xy, [1, 0, 0])) >= rot_tol
