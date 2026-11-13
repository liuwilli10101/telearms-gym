"""
Extra “Arrange-*” manipulation tasks that reuse the generic ArrangeWordEnv
infrastructure but enforce different ordering semantics.
"""

from __future__ import annotations
import numpy as np
import re
from roboeval.envs.task_library.arrange_word import ArrangeWordEnv
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.envs.props.tables import Table
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.utils.metric_rollout import MetricRolloutEval 
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.physics_utils import distance
from pyquaternion import Quaternion

# --------------------------------------------------------------------------- #
#   ArrangeMathEquationEnv  – create a *valid* 3-term equation,
#                               e.g.  "1+2=3"  or  "4-1=3".
#    • expects exactly 5 blocks  [digit] [op] [digit] [=] [digit]
# --------------------------------------------------------------------------- #
class ArrangeMathEquationEnv(RoboEvalEnv, MetricRolloutEval):
    """Arrange blocks into a correct (single-digit) math equation left→right."""
    _UP_COS_THRESH = 0.97 
    _ALIGN_TOL = 0.05  # x-alignment tolerance (±5 cm)
    
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"

    _TOL = 0.03                   # y-alignment ±3 cm
    _success_check = True         # flag for success check
    _final_metrics = {}           # metrics to expose at the end of the episode

    def __init__(self, *, equation: str = "1+2=3", **kwargs):
        assert re.fullmatch(r"\d[+\-*/]\d=\d", equation.replace(" ", ""))
        self._target_word = equation.replace(" ", "")
        super().__init__(**kwargs)
        
    def _initialize_env(self):
        # 1) make / load the blocks ------------------------------------------------
        blocks_info = ensure_cubes(self._target_word,
        universe=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "÷", "="])     # <- utility above

        self.blocks = []
        for letter in self._target_word:
            info = blocks_info[letter]
            self.blocks.append(
                SemanticCube(self._mojo,
                             xml_path=str(info["xml"]),
                             labels=info["labels"])
            )

        # 2) static props from preset (table, etc.)
        self.table = self._preset.get_props(Table)[0]

        # 3) metric buffers
        self._metric_init(track_vel_sync=True, track_vertical_sync=True,
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)

    # helper: evaluate string like "1+2=3"
    @staticmethod
    def _is_valid(eq: str) -> bool:
        try:
            lhs, rhs = eq.split("=")
            return eval(lhs) == int(rhs)        # noqa: S307  (safe: single-digit)
        except Exception:
            return False
        
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
                          track_slippage=True, slip_objects=self.blocks,
                          robot=self.robot, slip_sample_window=20,
                          track_collisions=True)
        for s in range(1, 6):
            self._metric_stage(s, False)

    def _on_step(self):
        self._metric_step()

    def _success(self) -> bool:
        self._success_check = True
        blocks_sorted = sorted(self.blocks,
                              key=lambda c: c.body.get_position()[1], reverse=True)
        y_sorted_ok = np.all(np.diff([c.body.get_position()[1] for c in blocks_sorted]) < 0)
        
         # --- x-alignment --------------------------------------------------
        x_vals = np.array([blk.body.get_position()[0] for blk in blocks_sorted])
        x_range = float(np.ptp(x_vals))           # peak-to-peak spread
        aligned = x_range < self._ALIGN_TOL
        
        labels_cos = [c.get_label(side="top") for c in blocks_sorted]
        up_ok = min(cos for _, cos in labels_cos) >= self._UP_COS_THRESH

        eq = "".join(lbl for lbl, _ in labels_cos)
        print(f"Equation: {eq}, valid: {self._is_valid(eq)}")
        self._success_check &= up_ok and self._is_valid(eq)
        readable_flags = [
            self.is_face_readable(c, lbl) for c, (lbl, _) in zip(blocks_sorted, labels_cos)
        ]
        print("Readability flags:", readable_flags)
        is_readable = all(readable_flags)
        single_block_grasped = any(
            self.robot.is_gripper_holding_object(block, side)
            for side in self.robot.grippers
            for block in blocks_sorted
        )
        
        distances = {}  # distance to each block from each gripper
        # –––––––– distance checks –––––––––––
        for idx, block in enumerate(blocks_sorted):
            for side in self.robot.grippers:
                distances[f"block_{idx}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, block.body)
                if self.robot.is_gripper_holding_object(block, side):
                    self._success_check = False
        distances.update({
            "horizontal alignment": x_range,
        })

        # –––––––– stage metrics –––––––––––––
        if single_block_grasped: self._metric_stage(1) # any block grasped
        if aligned: self._metric_stage(2) # x-horizontal alignment
        if y_sorted_ok: self._metric_stage(3) # y spaced -- none overlap
        if self._is_valid(eq): self._metric_stage(4) # valid equation
        if is_readable: self._metric_stage(5) # all faces readable
        
        self._success_check &= aligned and is_readable and self._is_valid(eq)
        
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=distances,
        )
        return self._success_check
    
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
