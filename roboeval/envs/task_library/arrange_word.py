from abc import ABC

import numpy as np
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from pyquaternion import Quaternion
import string

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import Cube, LargeCube, Rod
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.envs.props.tables import Table
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes

class ArrangeWordEnv(RoboEvalEnv, MetricRolloutEval, ABC):
    """Arrange blocks so that they spell an arbitrary word (top faces up & aligned)."""
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"

    _TOL = 0.03                   # y-alignment ±3 cm
    _UP_COS_THRESH = 0.95         # top face within 18°
    _success_check = True         # flag for success check
    _final_metrics = {}           # metrics to expose at the end of the episode

    # -----------------------------------------------------------------
    def __init__(self, *, target_word: str = "PIG", **kwargs):
        self._target_word = target_word.upper()
        super().__init__(**kwargs)    

    # -----------------------------------------------------------------
    def _initialize_env(self):
        # 1) make / load the blocks ------------------------------------------------
        blocks_info = ensure_cubes(self._target_word, universe=string.ascii_uppercase)     # <- utility above

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

    # -----------------------------------------------------------------
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
        for s in range(1, 5):
            self._metric_stage(s, False)

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
        for n_world, tag in block.face_normals_world():   # see SemanticCube helper
            cos = np.dot(n_world / np.linalg.norm(n_world), up_vec)
            if cos > best:
                best, lbl = cos, tag
        return lbl, best

    # -----------------------------------------------------------------
    # SUCCESS + METRICS
    # -----------------------------------------------------------------
    def _success(self) -> bool:
        self._success_check = True  # reset success check flag

        # 1) sort blocks by X coordinate  → stage 1 when all blocks unmoved/grasped?
        blocks_sorted = sorted(self.blocks,
                              key=lambda c: c.body.get_position()[1], reverse=True)
        y_sorted_ok = all(np.diff([b.body.get_position()[1] for b in blocks_sorted]) < 0)

        # 2) check upward labels & top-facing quality
        up_labels, cosines = [], []
        for block in blocks_sorted:
            lbl, cos = block.get_label(side="top")
            up_labels.append(lbl)
            cosines.append(cos)
        word_ok   = "".join(up_labels) == self._target_word

        # 3) horizontal (X) alignment
        x_vals = [c.body.get_position()[0] for c in blocks_sorted]
        x_ptp  = float(np.ptp(x_vals))            # peak-to-peak spread
        x_ok   = x_ptp < self._TOL

       # 4) readbility of faces
        up_labels, cosines, readable_flags = [], [], []
        for block in blocks_sorted:
            lbl, cos = block.get_label(side="top")
            up_labels.append(lbl)
            cosines.append(cos)
            readable_flags.append(self.is_face_readable(block, lbl))
            print(f"block {lbl}: up cosine = {cos:.3f}, readable = {readable_flags[-1]}")
            
        readable_ok = all(readable_flags)
        single_block_grasped = any(
            self.robot.is_gripper_holding_object(block, side)
            for side in self.robot.grippers
            for block in blocks_sorted
        )
        
        # ––––– distance checks –––––––––––––––––––––––––––––––––––
        distances = {}  # distance to each block from each gripper
        for idx, block in enumerate(blocks_sorted):
            for side in self.robot.grippers:
                distances[f"block_{idx}-gripper_{side.value}"] = distance(self.robot.grippers[side].body, block.body)
                if self.robot.is_gripper_holding_object(block, side):
                    self._success_check = False
        distances.update({
            "horizontal alignment": x_ptp,
        })
        

        # ------------------- stage logic ------------------------------ #
        if single_block_grasped: self._metric_stage(1)  # at least one block grasped
        if x_ok: self._metric_stage(2)   # first block placed in X-order
        if y_sorted_ok: self._metric_stage(3)  # all blocks placed & tops correct
        if word_ok: self._metric_stage(4) # word spelled correctly

        self._success_check = word_ok and x_ok and y_sorted_ok and readable_ok and not single_block_grasped

        # ------------------- distance / misc metrics ------------------ #
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=distances,
        )
        return self._success_check
    
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

        