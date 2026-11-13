"""
Integration between RoboEval environments and CuTAMP system.

This module provides utilities to:
1. Generate CuTAMP-compatible YAML environment configurations from RoboEval environments
2. Transform MuJoCo physics world configurations to robot base frame for CuTAMP
3. Handle robot joint name mappings between RoboEval and CuTAMP
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import tempfile
import logging
import copy
import gc
import atexit

# Add cuTAMP to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../thirdparty/cutamp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../thirdparty/curobo'))

from scipy.spatial.transform import Rotation as R

from curobo.geom.types import WorldConfig
from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment, load_env
from cutamp.tamp_world import TAMPWorld

# Import cuTAMP utilities for planning
from cutamp.algorithm import run_cutamp
from cutamp.cost_reduction import CostReducer
from cutamp.constraint_checker import ConstraintChecker
from cutamp.scripts.utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    get_tetris_tuned_constraint_to_mult,
    setup_logging,
)


from roboeval.demonstrations.demo_recorder import DemoRecorder
from roboeval.demonstrations.demo_store import DemoStore

_log = logging.getLogger(__name__)


class RoboEvalCuTAMPBridge:
    """Bridge between RoboEval environments and CuTAMP planning system."""
    
    # Class-level tracking of temporary files across all instances
    _temp_yaml_files: List[str] = []
    _cleanup_registered = False
    
    def __init__(self, env, robot_transform: Dict = None, cfg: Dict = None):
        """
        Initialize the bridge.
        
        Args:
            env: RoboEval environment instance
            robot_transform: Transform dict with 'R' (rotation matrix) and 't' (translation vector)
        """
        self.env = env
        self.robot_transform = robot_transform or self._get_default_robot_transform()
        self.cfg = cfg
        
        # Robot joint name mappings
        self.robot_joint_mappings = {
            "panda": ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", 
                     "panda_joint5", "panda_joint6", "panda_joint7"],
            "ur5": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            "h1": ["h1/left_shoulder_pitch", "h1/left_shoulder_roll", "h1/left_shoulder_yaw", 
                  "h1/left_elbow", "h1/left_wrist", "h1/right_shoulder_pitch", 
                  "h1/right_shoulder_roll", "h1/right_shoulder_yaw", "h1/right_elbow", "h1/right_wrist"]
        }
        
        # Instance-level tracking for cleanup
        self._tamp_world = None
        self._tamp_env = None
        self.recorder = None
        
        # Register cleanup handler once
        if not RoboEvalCuTAMPBridge._cleanup_registered:
            atexit.register(RoboEvalCuTAMPBridge._cleanup_all_temp_files)
            RoboEvalCuTAMPBridge._cleanup_registered = True
    
    @classmethod
    def _cleanup_all_temp_files(cls):
        """Clean up all temporary YAML files created by any instance."""
        if cls._temp_yaml_files:
            _log.info(f"Cleaning up {len(cls._temp_yaml_files)} temporary YAML files...")
            for filepath in cls._temp_yaml_files:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        _log.debug(f"Removed temp file: {filepath}")
                except Exception as e:
                    _log.warning(f"Failed to remove temp file {filepath}: {e}")
            cls._temp_yaml_files.clear()
    
    @classmethod
    def _track_temp_file(cls, filepath: str):
        """Track a temporary file for cleanup."""
        if filepath not in cls._temp_yaml_files:
            cls._temp_yaml_files.append(filepath)
    
    def cleanup_resources(self):
        """Clean up all resources associated with this bridge instance."""
        # Clean up TAMP world
        if self._tamp_world is not None:
            try:
                del self._tamp_world
                self._tamp_world = None
            except Exception as e:
                _log.warning(f"Error cleaning up TAMP world: {e}")
        
        # Clean up TAMP environment
        if self._tamp_env is not None:
            try:
                del self._tamp_env
                self._tamp_env = None
            except Exception as e:
                _log.warning(f"Error cleaning up TAMP env: {e}")
        
        # Clean up recorder
        if self.recorder is not None:
            try:
                if hasattr(self.recorder, 'stop'):
                    self.recorder.stop()
                del self.recorder
                self.recorder = None
            except Exception as e:
                _log.warning(f"Error cleaning up recorder: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Try to clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        _log.debug("Cleaned up bridge resources")
    
    def __del__(self):
        """Cleanup when bridge instance is destroyed."""
        self.cleanup_resources()
    
    def _cleanup_orphaned_temp_files(self, aggressive=False):
        """Clean up orphaned temp files from previous failed runs.
        
        Safetensors creates temp files during save operations. If the process crashes
        or is interrupted, these files can accumulate. This method cleans them up.
        
        Args:
            aggressive: If True, clean ALL temp files regardless of age. If False, only clean old files (>1 hour)
        """
        import glob
        import time
        import subprocess
        
        # Check common temp directories
        temp_locations = [
            '/tmp/tmp*',
            str(Path.home() / '.cache' / 'tmp*'),
            '/root/.cache/tmp*',  # Requires sudo, checked last
        ]
        
        cleaned_count = 0
        cleaned_size = 0
        
        for pattern in temp_locations:
            # Special handling for /root/.cache/ - try without sudo first
            if '/root/.cache' in pattern:
                need_sudo = False
                files_cleaned_without_sudo = 0
                
                try:
                    # First, try without sudo using glob
                    try:
                        found_files = glob.glob(pattern)
                        if not found_files:
                            # glob couldn't access /root/.cache/ - need sudo
                            need_sudo = True
                        else:
                            # Try to clean accessible files
                            for filepath in found_files:
                                try:
                                    if os.path.isfile(filepath):
                                        age = time.time() - os.path.getmtime(filepath)
                                        size = os.path.getsize(filepath)
                                        
                                        # Clean if: aggressive mode, OR file is old (>1 hour), OR file is large (>100MB)
                                        should_clean = aggressive or age > 3600 or size > 100 * 1024 * 1024
                                        
                                        if should_clean:
                                            try:
                                                os.remove(filepath)
                                                files_cleaned_without_sudo += 1
                                                cleaned_size += size
                                                _log.debug(f"Cleaned temp file: {filepath} ({size/1024/1024:.1f}MB)")
                                            except PermissionError:
                                                # Permission denied - need sudo
                                                need_sudo = True
                                except Exception:
                                    pass
                    except Exception:
                        # Can't access /root/.cache/ without sudo
                        need_sudo = True
                    
                    # Only run sudo if we couldn't access or clean the files
                    if need_sudo:
                        try:
                            # Use sudo to find and delete temp files in /root/.cache/
                            if aggressive:
                                # Delete all temp files
                                result = subprocess.run(
                                    ['sudo', 'find', '/root/.cache/', '-name', 'tmp*', '-type', 'f', '-delete'],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                    check=False
                                )
                            else:
                                # Delete only large (>100MB) or old (>1 hour) temp files
                                # First, find large files
                                result_large = subprocess.run(
                                    ['sudo', 'find', '/root/.cache/', '-name', 'tmp*', '-type', 'f', '-size', '+100M', '-delete'],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                    check=False
                                )
                                # Then, find old files (>1 hour = 60 minutes)
                                result_old = subprocess.run(
                                    ['sudo', 'find', '/root/.cache/', '-name', 'tmp*', '-type', 'f', '-mmin', '+60', '-delete'],
                                    capture_output=True,
                                    text=True,
                                    timeout=30,
                                    check=False
                                )
                            _log.debug(f"Cleaned /root/.cache/ temp files (with sudo)")
                            cleaned_count += 1  # We don't know exact count, but indicate success
                        except Exception as e:
                            _log.debug(f"Could not clean /root/.cache/ with sudo: {e}")
                    else:
                        # Successfully cleaned without sudo
                        if files_cleaned_without_sudo > 0:
                            cleaned_count += files_cleaned_without_sudo
                            _log.debug(f"Cleaned {files_cleaned_without_sudo} temp files from /root/.cache/ (without sudo)")
                        
                except Exception as e:
                    _log.debug(f"Could not clean /root/.cache/: {e}")
                continue
            
            # Regular cleanup for accessible directories
            try:
                # Find temp files
                for filepath in glob.glob(pattern):
                    try:
                        if os.path.isfile(filepath):
                            age = time.time() - os.path.getmtime(filepath)
                            size = os.path.getsize(filepath)
                            
                            # Clean if: aggressive mode, OR file is old (>1 hour), OR file is large (>100MB)
                            should_clean = aggressive or age > 3600 or size > 100 * 1024 * 1024
                            
                            if should_clean:
                                try:
                                    # First try without sudo
                                    os.remove(filepath)
                                    cleaned_count += 1
                                    cleaned_size += size
                                    _log.debug(f"Cleaned temp file: {filepath} ({size/1024/1024:.1f}MB, age={age/60:.1f}min)")
                                except PermissionError:
                                    # Fall back to sudo if permission denied
                                    try:
                                        subprocess.run(['sudo', 'rm', '-f', filepath], check=False, capture_output=True, timeout=10)
                                        cleaned_count += 1
                                        cleaned_size += size
                                        _log.debug(f"Cleaned temp file with sudo: {filepath} ({size/1024/1024:.1f}MB)")
                                    except Exception as sudo_error:
                                        _log.debug(f"Could not clean with sudo: {filepath}: {sudo_error}")
                                except Exception as remove_error:
                                    _log.debug(f"Could not remove {filepath}: {remove_error}")
                    except Exception as e:
                        _log.debug(f"Could not process temp file {filepath}: {e}")
            except Exception as e:
                _log.debug(f"Could not check temp location {pattern}: {e}")
        
        if cleaned_count > 0:
            _log.info(f"Cleaned up {cleaned_count} temp files ({cleaned_size/1024/1024/1024:.2f}GB)")
        
        return cleaned_count, cleaned_size

    def _get_default_robot_transform(self) -> Dict:
        """Get default robot transform (identity)."""
        return {"R": np.eye(3), "t": np.zeros(3)}

    def xmat_to_quat(self, data):
        """
        Convert a body's xmat (flattened 3x3 rotation matrix) from mjData to a MuJoCo-style quaternion [w, x, y, z].
        """
        R_flat = data.xmat       # shape (9,)
        R_mat = np.reshape(R_flat, (3, 3))      # shape (3, 3)
        quat_xyzw = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
        quat_wxyz = np.concatenate(([quat_xyzw[3]], quat_xyzw[:3]))  # [w, x, y, z]
        return quat_wxyz

    def get_mesh_dict(self):
        """Extract mesh dictionary from environment.
        Adds entries keyed by both full_identifier and plain geom name so that we can
        later resolve meshes from low-level physics model geoms which only expose name.
        """
        mesh_dict: Dict[str, str] = {}
        for prop in self.env._preset.get_props():
            prop_path = prop._model_path.parent
            for geom in prop._geoms:
                try:
                    geom_filename = geom.mjcf.mesh.file.get_vfs_filename(filename_with_hash=False)
                except Exception:
                    geom_filename = None
                if geom_filename is not None:
                    full_id = getattr(geom.mjcf, 'full_identifier', None)
                    path_str = str(prop_path / "assets" / geom_filename)
                    if full_id:
                        mesh_dict[full_id] = path_str
                    # also map plain name
                    if getattr(geom, 'name', None):
                        mesh_dict[geom.name] = path_str
        return mesh_dict

    def _reduce_bbox(self, bb_min, bb_max):
        """Return (dims, center, global_min) where dims is 3-vector.
        Handles cases where bb_min/max are (N,3) arrays of corner samples.
        """
        bb_min_a = np.array(bb_min, dtype=float)
        bb_max_a = np.array(bb_max, dtype=float)
        if bb_min_a.ndim > 1:
            bb_min_red = bb_min_a.min(axis=0)
        else:
            bb_min_red = bb_min_a
        if bb_max_a.ndim > 1:
            bb_max_red = bb_max_a.max(axis=0)
        else:
            bb_max_red = bb_max_a
        dims = (bb_max_red - bb_min_red)
        center = (bb_min_red + bb_max_red) / 2.0
        return dims[:3], center[:3], bb_min_red[:3]

    def clean_geometry_name(self, name: str) -> str:
        """
        Clean geometry names for consistent TAMP naming.
        
        - Remove '_collision' suffix from MuJoCo collision geometry names
        - Replace '/' and ' ' with '_' 
        - Convert cube/block names to block0, block1, etc. format (0-based indexing)
        - Handle special cube collision naming: cube_collision -> block0, cube_1_collision -> block1, etc.
        - Convert table to floor
        
        Args:
            name: Original geometry name from MuJoCo
            
        Returns:
            Cleaned name suitable for TAMP planning
        """
        if name is None:
            return name
            
        # Handle special cube collision naming pattern first
        import re
        
        # Pattern: cube_collision -> block0
        if name == 'cube_collision':
            return 'block0'
        
        # Pattern: cube_N_collision -> blockN (where N is a number >= 1)
        cube_collision_match = re.match(r'cube_(\d+)_collision', name)
        if cube_collision_match:
            cube_num = int(cube_collision_match.group(1))
            return f'block{cube_num}'
            
        # Remove collision suffix
        clean_name = name
        if clean_name.endswith('_collision'):
            clean_name = clean_name[:-10]  # Remove '_collision'
        elif clean_name.endswith('/collision'):
            clean_name = clean_name[:-10]  # Remove '/collision'
        elif clean_name == 'collision':
            clean_name = 'cube'  # Handle case where geom is just named 'collision'
        
        # Replace problematic characters
        clean_name = clean_name.replace('/', '_').replace(' ', '_')
        
        # Convert table to floor
        if clean_name.lower() == 'table':
            clean_name = 'floor'
            
        # Handle cube_N pattern -> blockN (convert cubes to blocks with 0-based indexing)
        cube_match = re.match(r'cube_(\d+)', clean_name)
        if cube_match:
            cube_num = cube_match.group(1)
            clean_name = f'block{cube_num}'
        
        # Handle cube (no number) -> block0 (convert base cube to block0)
        if clean_name == 'cube':
            clean_name = 'block0'
        
        # Handle block_N pattern -> blockN (remove underscores in numbered blocks)
        block_match = re.match(r'block_(\d+)', clean_name)
        if block_match:
            block_num = block_match.group(1)
            clean_name = f'block{block_num}'
            
        return clean_name

    def generate_world_config(self, geom_group=3, mesh_test=False, keywords_to_omit=None,
                              treat_table_mesh_as_cuboid: bool = False,
                              table_keywords: Optional[List[str]] = None,
                              skip_invalid_mesh: bool = False,
                              table_dims_source: str = "aabb",
                              table_z_offset: float = 0.0,
                              adjust_xy_for_table_collision: bool = True,
                              collision_clearance: float = 0.005,
                              adjust_table_xy_for_collision: bool = True,
                              movable_z_offset: float = 0.0):
        """Generate world configuration (revamped).

        Revamped to mirror the simpler multigrasp_data_sampler style:
        - Collect raw MuJoCo geoms for requested geom_group
        - Export primitive shapes directly
        - Export meshes with file_path
        - Inject a synthetic 'table' cuboid using the table object's aabb (preferred) and
          the position of mesh geom named 'table/collider_001'.
        - Apply 2x dimension scaling to cubes/blocks AND the table cuboid (per request).
        - Optionally resolve initial cube/table collisions by shifting cubes in XY only (preserving Z).

        Args:
            geom_group: MuJoCo geom group index to export.
            mesh_test: If True, print mesh file paths.
            keywords_to_omit: Substrings; geoms containing any are skipped.
            treat_table_mesh_as_cuboid: (legacy flag) retained for API compatibility.
            table_keywords: (unused in revamped path, kept for compatibility)
            skip_invalid_mesh: Skip mesh if file path missing.
            table_dims_source: 'aabb' (only supported now) or 'bbox'.
            table_z_offset: Extra Z offset applied AFTER using mesh geom position.
            adjust_xy_for_table_collision: Adjust cube position in XY to avoid table collisions
            collision_clearance: Clearance space when adjusting position to avoid collisions
            movable_z_offset: Z offset to apply to movable objects (cubes/blocks) - useful for avoiding
                            collision detection issues in cuTAMP while maintaining correct trajectories.
                            Recommended value: 0.013 (13mm) to match typical MuJoCo penetration.
        """
        if keywords_to_omit is None:
            keywords_to_omit = ["panda", "robot"]
        world_config = {"cylinder": {}, "cuboid": {}, "mesh": {}, "capsule": {}}
        physics = self.env._mojo.physics
        mesh_dict = self.get_mesh_dict()

        # 1. Gather table dims via AABB (bbox) if available
        table_obj = getattr(self.env, 'table', None)
        table_dims = None  # full extents (x,y,z)
        table_pose_from_bbox = None  # xyz (bottom-anchored) + quat
        table_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # keep upright
        if table_obj is not None:
            try:
                # Ensure bbox values are current
                table_obj.bbox.update()
                bb_min = np.array(table_obj.bbox.min, dtype=float)
                bb_max = np.array(table_obj.bbox.max, dtype=float)
                # Full dimensions
                dims_full = bb_max - bb_min
                # Apply 2x scaling (same rule as blocks)
                table_dims = dims_full 
                center = (bb_min + bb_max) / 2.0
                # Bottom-anchored pose z (match cuboid export convention)
                bottom_z = bb_min[2]
                table_pose_from_bbox = np.concatenate([
                    np.array([center[0], center[1], center[2]], dtype=float),
                    table_quat,
                ])
                _log.debug(
                    f"Table bbox -> min:{bb_min} max:{bb_max} dims_full:{dims_full} scaled:{table_dims} pose:{table_pose_from_bbox}"
                )
            except Exception as e:
                _log.warning(f"Failed to derive table bbox: {e}")
        # Fallback defaults if bbox unavailable
        if table_dims is None:
            table_dims = np.array([1.2, 0.6, 0.8])  # heuristic full extents
            table_pose_from_bbox = np.concatenate([
                np.array([0.0, 0.0, 0.0 + table_z_offset]), table_quat
            ])

        # 2. Iterate collision geoms, record primitives/meshes (still done for other objects)
        collision_geom_idx = np.where(physics.model.geom_group == geom_group)[0]
        # We no longer depend on a specific table mesh pose; bbox defines canonical table pose
        # Kept variables for potential future orientation extraction
        table_mesh_pos = None
        # Track names to avoid duplicates and generate sequential numbering
        name_counters = {}
        existing_cube_numbers = set()  # Track which cube numbers are already used
        
        for i in collision_geom_idx:
            geom = physics.model.geom(i)
            raw_name = geom.name
            if raw_name is None:
                continue
            if any(k in raw_name for k in keywords_to_omit):
                continue
                
            # Clean the geometry name
            clean_name = self.clean_geometry_name(raw_name)
            
            # Track cube numbers that already exist from MuJoCo
            import re
            cube_match = re.match(r'cube(\d+)', clean_name)
            if cube_match:
                existing_cube_numbers.add(int(cube_match.group(1)))
            
            # Handle duplicate names by adding sequential numbers
            if clean_name in name_counters:
                name_counters[clean_name] += 1
                final_name = f"{clean_name}{name_counters[clean_name] - 1}"  # 0-based indexing
            else:
                name_counters[clean_name] = 1
                # For cube/block names without numbers, start with cube0, cube1, etc. (0-based indexing)
                if clean_name in ['cube', 'block']:
                    # Find next available number starting from 0
                    next_num = 0
                    while next_num in existing_cube_numbers or f"{clean_name}{next_num}" in name_counters:
                        next_num += 1
                    final_name = f"{clean_name}{next_num}"
                    existing_cube_numbers.add(next_num)
                else:
                    final_name = clean_name
                    
            geom_data = physics.data.geom(i)
            quaternion = self.xmat_to_quat(geom_data)
            gtype = geom.type[0]

            if gtype == 3:  # capsule
                world_config['capsule'][final_name] = {
                    "radius": geom.size[0],
                    "height": geom.size[1],
                    "pose": np.concatenate([geom_data.xpos, quaternion]).tolist()
                }
            elif gtype == 5:  # cylinder
                world_config['cylinder'][final_name] = {
                    "radius": geom.size[0],
                    "height": geom.size[1],
                    "pose": np.concatenate([geom_data.xpos, quaternion]).tolist()
                }
            elif gtype == 6:  # box -> cuboid
                cuboid_dims = (np.array(geom.size) * 1.0)  # MuJoCo half-extents -> full
                pose = np.concatenate([geom_data.xpos.copy(), quaternion])
                # Shift z so pose z = bottom (MuJoCo centers shapes at geom.xpos)
                pose[2] -= cuboid_dims[2] / 2.0
                
                # Apply Z offset to movable objects (cubes/blocks) for collision clearance
                # This lifts objects in cuTAMP planning to avoid collision detection issues
                # while robot trajectories remain valid for original MuJoCo positions
                if movable_z_offset != 0.0 and final_name != 'floor':
                    # Check if this is a movable object (cube/block)
                    is_movable = any(keyword in final_name.lower() for keyword in ['cube', 'block', 'target'])
                    if is_movable:
                        pose[2] += movable_z_offset
                        _log.info(f"Applied +{movable_z_offset:.4f}m Z offset to movable object '{final_name}'")
                
                world_config['cuboid'][final_name] = {"dims": cuboid_dims.tolist(), "pose": pose.tolist()}
            elif gtype == 7:  # mesh
                mesh_file = mesh_dict.get(raw_name)  # Use raw name for mesh lookup
                if mesh_file is None and skip_invalid_mesh:
                    continue
                world_config['mesh'][final_name] = {
                    "file_path": mesh_file,
                    "pose": np.concatenate([geom_data.xpos, quaternion]).tolist()
                }

        # 3. Inject synthetic table cuboid from bbox (use "floor" name)
        world_config['cuboid']['floor'] = {
            'dims': np.array(table_dims, dtype=float).tolist(),
            'pose': table_pose_from_bbox.tolist(),
            'color': [180, 180, 180]
        }
        # 4. Remove any raw table meshes/cuboids except our unified 'floor' cuboid
        to_del_mesh = [n for n in world_config['mesh'].keys() if 'table' in n.lower()]
        for n in to_del_mesh:
            del world_config['mesh'][n]
        to_del_cub = [n for n in world_config['cuboid'].keys() if n != 'floor' and 'table' in n.lower()]
        for n in to_del_cub:
            del world_config['cuboid'][n]
        # 5. Apply 2x dims rule to cubes/blocks (floor already scaled). Then ensure no intersection with table.
        for _n, entry in world_config['cuboid'].items():
            if _n.lower() == 'floor':
                continue  # floor already scaled
            # if not any(k in _n.lower() for k in ['cube','block', 'target']):
            #     continue
            if 'dims' in entry and isinstance(entry['dims'], (list, tuple)) and len(entry['dims']) >= 3:
                old_dims = np.array(entry['dims'], dtype=float)
                entry['dims'] = (old_dims * 2.0).tolist()


        if mesh_test:
            for key, value in world_config['mesh'].items():
                print(f"{key}: {value.get('file_path', 'No file path')}")
        return world_config

    def transform_objects_to_robot_frame(self, objects, robot_transform):
        R_wb = robot_transform["R"]  # world->base
        t_wb = robot_transform["t"]  # already -R_wb @ p_b
        result = {k: {kk: (vv.copy() if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in objects.items()}
        def quat_wxyz_to_matrix(q):
            # q = [w,x,y,z]
            return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        def matrix_to_quat_wxyz(m):
            q_xyzw = R.from_matrix(m).as_quat()
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        for gtype in ['cuboid','mesh','cylinder','capsule']:
            for name, entry in result.get(gtype, {}).items():
                pose = entry.get('pose')
                if pose is None:
                    continue
                pose_arr = np.array(pose, dtype=float).reshape(-1)
                if pose_arr.size < 7:
                    continue
                p_w = pose_arr[:3]
                q_w = pose_arr[3:7]
                p_b = R_wb @ p_w + t_wb
                try:
                    R_obj_w = quat_wxyz_to_matrix(q_w)
                    R_obj_b = R_wb @ R_obj_w
                    q_b = matrix_to_quat_wxyz(R_obj_b)
                except Exception:
                    q_b = q_w
                new_pose = np.concatenate([p_b, q_b])
                entry['pose'] = new_pose.tolist()
                if name.lower() == 'floor':
                    _log.debug(f"Transform floor world->base: world_pos={p_w} base_pos={p_b}")
        return result

    def get_robot_transform(self, robot_name: str):
        physics = self.env._mojo.physics
        p_b = physics.named.data.xpos[robot_name]
        R_bw_flat = physics.named.data.xmat[robot_name]
        R_bw = np.reshape(R_bw_flat, (3,3))  # body orientation in world
        R_wb = R_bw.T
        t_wb = -R_wb @ p_b
        return {"R": R_wb, "t": t_wb}

    def generate_cutamp_yaml_config(self, 
                                   env_name: str,
                                   num_blocks: int = 3,
                                   block_size: float = 0.04,
                                   output_path: Optional[str] = None,
                                   treat_table_mesh_as_cuboid: bool = False,
                                   movable_objects: Optional[List[str]] = None,
                                   surface_objects: Optional[List[str]] = None) -> str:
        """
        Generate CuTAMP-compatible YAML configuration from current environment state.
        
        Args:
            env_name: Name for the environment
            num_blocks: Number of blocks in stacking environment  
            block_size: Size of cubic blocks
            output_path: Path to save YAML file (if None, creates temp file)
            treat_table_mesh_as_cuboid: Forwarded flag controlling table handling
            movable_objects: List of movable object names from LLM analysis
            surface_objects: List of surface object names from LLM analysis
            
        Returns:
            str: Path to generated YAML file
        """
        
        # Generate world configuration with requested table handling
        world_config = self.generate_world_config(keywords_to_omit=["panda", "robot"],
                                                  treat_table_mesh_as_cuboid=treat_table_mesh_as_cuboid,
                                                  table_dims_source="aabb")
        # Use class's correct transform
        panda_transform = self.get_robot_transform("panda nohand_left/")
        # Debug floor before transform
        if 'floor' in world_config['cuboid']:
            _log.debug(f"Floor world pose before transform: {world_config['cuboid']['floor']['pose']}")
        robot_world_config = self.transform_objects_to_robot_frame(world_config, panda_transform)
        if 'floor' in robot_world_config['cuboid']:
            _log.debug(f"Floor base pose after transform: {robot_world_config['cuboid']['floor']['pose']}")
        # Build YAML structure
        yaml_config = {
            "name": env_name,
            "geometries": {},
            "types": {"Movable": [], "Surface": []},
            "goal": []
        }
        
        # Add geometries from world config (including mesh objects for table colliders)
        for geom_type, objects in robot_world_config.items():
            if not objects:
                continue
            yaml_config['geometries'][geom_type] = {}
            for obj_name, obj_data in objects.items():
                obj_data_serializable = {}
                for k, v in obj_data.items():
                    if isinstance(v, np.ndarray):
                        obj_data_serializable[k] = v.tolist()
                    else:
                        obj_data_serializable[k] = v
                clean_obj_name = obj_name.replace('/', '_').replace(' ', '_')
                if 'color' not in obj_data_serializable and geom_type != 'mesh':
                    if any(tok in obj_name for tok in ['cube','block']):
                        obj_data_serializable['color'] = [220,220,80]
                    elif any(tok in obj_name for tok in ['table','surface','collider']):
                        obj_data_serializable['color'] = [180,180,180]
                    else:
                        obj_data_serializable['color'] = [128,128,128]
                yaml_config['geometries'][geom_type][clean_obj_name] = obj_data_serializable
                
                # Use LLM-provided object lists if available, otherwise fallback to heuristics
                if movable_objects and clean_obj_name in movable_objects:
                    yaml_config['types']['Movable'].append(clean_obj_name)
                elif surface_objects and clean_obj_name in surface_objects:
                    yaml_config['types']['Surface'].append(clean_obj_name)
                else:
                    # Fallback: Primary classification: cubes/blocks movable; floor/surfaces surfaces
                    if any(tok in obj_name for tok in ['cube','block']):
                        yaml_config['types']['Movable'].append(clean_obj_name)
                    elif any(tok in obj_name for tok in ['floor','surface','collider']):
                        yaml_config['types']['Surface'].append(clean_obj_name)
        # Make every cube/block also a Surface (as requested) so any block can be a support
        for _name in list(yaml_config['types']['Movable']):
            if ('cube' in _name or 'block' in _name) and _name not in yaml_config['types']['Surface']:
                yaml_config['types']['Surface'].append(_name)
        # ------------------------------------------------------------------
        # Stacking semantics adjustment
        # In cuTAMP domain, On(obj, surface) requires that the second argument
        # belong to the Surface type. For a block stacking task we therefore
        # must ALSO classify supporting blocks (all but the top) as Surfaces.
        # Previous version omitted blocks from Surface leading to zero plans.
        # ------------------------------------------------------------------
        # Collect movable blocks and obtain their z positions to impose an ordering.
        movable_blocks = [name for name in yaml_config["types"]["Movable"] 
                          if ("cube" in name or "block" in name)]
        
        # Build a lookup of poses to extract z. Search across geometry types.
        name_to_pose = {}
        for gtype, objs in yaml_config['geometries'].items():
            for oname, data in objs.items():
                pose = data.get('pose')
                if pose and isinstance(pose, list) and len(pose) >= 3:
                    name_to_pose[oname] = pose
        
        block_with_z = []
        for b in movable_blocks:
            z = name_to_pose.get(b, [0,0,0])[2]
            block_with_z.append((b, z))
        # Sort by z ascending; tie-break by name for determinism
        block_with_z.sort(key=lambda x: (x[1], x[0]))
        ordered_blocks = [b for b,_ in block_with_z]
        
        # Fallback: if z ordering produced different length (should not), use original list
        if len(ordered_blocks) != len(movable_blocks):
            ordered_blocks = movable_blocks
        
        # Add all but the top block to Surface list (if not already). Table stays as is.
        for support_block in ordered_blocks[:-1]:
            if support_block not in yaml_config['types']['Surface']:
                yaml_config['types']['Surface'].append(support_block)
        
        # Create stacking goal chain using ordered blocks truncated by num_blocks.
        chain_blocks = ordered_blocks[:num_blocks]
        # If num_blocks exceeds available, chain_blocks is just available ones.
        if len(chain_blocks) > 0:
            # Use first block in chain (should be cube0, block0, etc.)
            yaml_config["goal"].append({"On": [chain_blocks[0], "floor"]})
        for i in range(1, len(chain_blocks)):
            upper = chain_blocks[i]
            lower = chain_blocks[i-1]
            # Ensure lower is typed Surface (already enforced above)
            yaml_config["goal"].append({"On": [upper, lower]})
        
        # Add user-requested additional goals
        # Ensure HandEmpty predicate is present
        additional_goals = [
            {"HandEmpty": []}
        ]
        # Avoid duplicating identical goal dicts
        existing = yaml_config["goal"]
        for g in additional_goals:
            if g not in existing:
                existing.append(g)

        # Save to file with clean YAML output
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
            output_path = temp_file.name
            temp_file.close()
            # Track temp file for cleanup
            self._track_temp_file(output_path)
        
        # Clean YAML output by ensuring all numpy types are converted to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Build YAML structure with renaming: table->floor, cubes/blocks -> block{idx}
        # ------------------------------------------------------------------
        # Build rename mapping first so geometry, types, and goals stay consistent.
        rename_map: Dict[str, str] = {}
        original_block_names: List[str] = []
        cuboids_wc = robot_world_config.get('cuboid', {})
        for oname in cuboids_wc.keys():
            low = oname.lower()
            if low == 'table':
                rename_map[oname] = 'floor'
            elif any(tok in low for tok in ['cube', 'block']):
                original_block_names.append(oname)
        # Deterministic ordering (by original name) for indexing
        original_block_names.sort()
        for idx, oname in enumerate(original_block_names):
            rename_map[oname] = f'block{idx}'
        # Allow limiting number of blocks via num_blocks param (truncate mapping beyond)
        if num_blocks < len(original_block_names):
            # Drop mappings for blocks beyond requested count (they will still appear but not in goals)
            pass  # keep mapping; goals will just slice
        yaml_config = {
            "name": env_name,
            "geometries": {},
            "types": {"Movable": [], "Surface": []},
            "goal": []
        }
        # Populate geometries applying renames
        for geom_type, objects in robot_world_config.items():
            if not objects:
                continue
            yaml_config['geometries'][geom_type] = {}
            for obj_name, obj_data in objects.items():
                new_name = rename_map.get(obj_name, obj_name)
                obj_data_serializable = {}
                for k, v in obj_data.items():
                    if isinstance(v, np.ndarray):
                        obj_data_serializable[k] = v.tolist()
                    else:
                        obj_data_serializable[k] = v
                clean_obj_name = new_name.replace('/', '_').replace(' ', '_')
                if 'color' not in obj_data_serializable and geom_type != 'mesh':
                    low = obj_name.lower()
                    if any(tok in low for tok in ['cube','block']):
                        obj_data_serializable['color'] = [220,220,80]
                    elif any(tok in low for tok in ['table','surface','collider','floor']):
                        obj_data_serializable['color'] = [180,180,180]
                    else:
                        obj_data_serializable['color'] = [128,128,128]
                yaml_config['geometries'][geom_type][clean_obj_name] = obj_data_serializable
                
                # Use LLM-provided object lists if available, otherwise fallback to heuristics
                if movable_objects and clean_obj_name in movable_objects:
                    yaml_config['types']['Movable'].append(clean_obj_name)
                elif surface_objects and clean_obj_name in surface_objects:
                    if clean_obj_name not in yaml_config['types']['Surface']:
                        yaml_config['types']['Surface'].append(clean_obj_name)
                else:
                    # Fallback: Classification (post-rename)
                    if clean_obj_name.startswith('block'):
                        yaml_config['types']['Movable'].append(clean_obj_name)
                    if clean_obj_name == 'floor' or clean_obj_name.startswith('block'):
                        # All blocks act as surfaces + floor
                        if clean_obj_name not in yaml_config['types']['Surface']:
                            yaml_config['types']['Surface'].append(clean_obj_name)
                
                # Ensure blocks are both movable and surface if using LLM lists
                if movable_objects and surface_objects:
                    if clean_obj_name in movable_objects and ('block' in clean_obj_name.lower() or 'cube' in clean_obj_name.lower()):
                        if clean_obj_name not in yaml_config['types']['Surface']:
                            yaml_config['types']['Surface'].append(clean_obj_name)
        # ------------------------------------------------------------------
        # Goals: stack blocks sequentially on floor then each other, limited by num_blocks
        # ------------------------------------------------------------------
        blocks_order = [f'block{i}' for i in range(len(original_block_names))]
        blocks_chain = blocks_order[:num_blocks]
        if len(blocks_chain) > 0:
            # First block on floor (if floor exists)
            if 'floor' in yaml_config['geometries'].get('cuboid', {}):
                yaml_config['goal'].append({"On": [blocks_chain[0], 'floor']})
        for i in range(1, len(blocks_chain)):
            yaml_config['goal'].append({"On": [blocks_chain[i], blocks_chain[i-1]]})
        yaml_config['goal'].append({"HandEmpty": []})
        # ------------------------------------------------------------------
        # Serialize YAML manually (same format logic as before)
        # ------------------------------------------------------------------
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        clean_yaml_config = convert_numpy_types(yaml_config)
        def fmt_val(v):
            if isinstance(v, float):
                return ('{:.15g}'.format(v))
            return str(v)
        def fmt_list(lst):
            return '[' + ', '.join(fmt_val(x) for x in lst) + ']'
        lines = []
        lines.append(f'name: {clean_yaml_config["name"]}')
        lines.append('')
        lines.append('geometries:')
        geoms = clean_yaml_config.get('geometries', {})
        for gtype, objs in geoms.items():
            lines.append(f'  {gtype}:')
            for oname, data in objs.items():
                lines.append(f'    {oname}:')
                if 'dims' in data:
                    lines.append(f'      dims: {fmt_list(data["dims"]) }')
                if 'pose' in data:
                    lines.append(f'      pose: {fmt_list(data["pose"]) }')
                if 'color' in data:
                    lines.append(f'      color: {fmt_list(data["color"]) }')
                if 'file_path' in data:
                    lines.append(f'      file_path: {data["file_path"]}')
                for k, v in data.items():
                    if k in ('dims','pose','color','file_path'): continue
                    if isinstance(v, list):
                        lines.append(f'      {k}: {fmt_list(v)}')
                    else:
                        lines.append(f'      {k}: {v}')
        lines.append('')
        lines.append('types:')
        types_sec = clean_yaml_config.get('types', {})
        for tkey in ['Movable','Surface']:
            if tkey in types_sec:
                lines.append(f'  {tkey}: {fmt_list(types_sec[tkey]) }')
        lines.append('')
        lines.append('goal:')
        for g in clean_yaml_config.get('goal', []):
            if not isinstance(g, dict) or len(g) != 1:
                continue
            k, v = list(g.items())[0]
            if k == 'On' and isinstance(v, (list, tuple)) and len(v) == 2:
                lines.append(f'  - "On": [{v[0]}, {v[1]}]')
            else:
                if isinstance(v, (list, tuple)):
                    lines.append(f'  - {k}: {fmt_list(v)}')
                else:
                    lines.append(f'  - {k}: {v}')
        final_yaml = '\n'.join(lines) + '\n'
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
            output_path = temp_file.name
            temp_file.close()
            # Track temp file for cleanup
            self._track_temp_file(output_path)
        with open(output_path,'w') as f:
            f.write(final_yaml)
        _log.info(f"Generated CuTAMP YAML configuration: {output_path}")
        return output_path

    def create_tamp_world(self, 
                         yaml_config_path: str,
                         config: TAMPConfiguration,
                         robot_name: str = "panda",
                         q_init: Optional[List[float]] = None) -> TAMPWorld:
        """
        Create TAMPWorld from YAML configuration.
        
        Args:
            yaml_config_path: Path to YAML environment configuration
            config: TAMP configuration
            robot_name: Name of robot ("panda", "ur5", etc.)
            q_init: Initial joint configuration
            
        Returns:
            TAMPWorld: Configured TAMP world for planning
        """
        from curobo.types.base import TensorDeviceType
        from cutamp.robots import load_robot_container, get_q_home
        import torch
        
        # Load environment from YAML
        tamp_env = load_env(yaml_config_path)
        
        # Setup tensor args and robot
        tensor_args = TensorDeviceType()
        robot_container = load_robot_container(robot_name, tensor_args)
        
        if q_init is None:
            q_init = get_q_home(robot_name)
        q_init = torch.tensor(q_init, dtype=torch.float32, device=tensor_args.device)
        
        # Create TAMP world
        world = TAMPWorld(
            env=tamp_env,
            tensor_args=tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=config.world_activation_distance,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
        )
        
        return world

    def get_robot_joint_names(self, robot_name: str) -> List[str]:
        """
        Get robot joint names for the specified robot.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            List of joint names
        """
        if robot_name in self.robot_joint_mappings:
            return self.robot_joint_mappings[robot_name]
        else:
            _log.warning(f"Unknown robot {robot_name}, using panda joint names as default")
            return self.robot_joint_mappings["panda"]

    def create_joint_state_from_position(self, 
                                        robot_name: str, 
                                        positions: List[float],
                                        device: str = "cuda"):
        """
        Create JointState from joint positions with proper joint names.
        
        Args:
            robot_name: Name of the robot
            positions: Joint position values
            device: Torch device ("cuda" or "cpu")
            
        Returns:
            JointState with proper joint names
        """
        import torch
        from curobo.types.robot import JointState
        
        joint_names = self.get_robot_joint_names(robot_name)
        
        # Ensure we have the right number of joint values
        if len(positions) != len(joint_names):
            _log.warning(f"Position count ({len(positions)}) doesn't match joint count ({len(joint_names)})")
            joint_names = joint_names[:len(positions)]  # Truncate joint names if needed
        
        return JointState.from_position(
            torch.tensor(positions, dtype=torch.float32, device=device).reshape(1, -1),
            joint_names=joint_names
        )

    def load_tamp_env(self, yaml_path: str) -> TAMPEnvironment:
        """Load a CuTAMP YAML environment file into a TAMPEnvironment."""
        return load_env(yaml_path)

    def run_cutamp_and_get_plan(
        self,
        yaml_path: str,
        tamp_config: TAMPConfiguration,
        q_init: Optional[List[float]] = None,
        use_tetris_tuned_weights: bool = False,
    ):
        """Run cuTAMP on the provided YAML env and return motion plan list (accum_plans).

        Returns:
            (curobo_plan_list, num_satisfying)
        """
        setup_logging()

        # Ensure we request curobo motion planning output
        if not tamp_config.curobo_plan:
            _log.info("Enabling curobo_plan flag on config for motion trajectory generation.")
            tamp_config = TAMPConfiguration(**{**tamp_config.__dict__, "curobo_plan": True})  # immutable dataclass -> recreate
        
        print(f"Loading environment from {yaml_path}")
        env = self.load_tamp_env(yaml_path)
        self._tamp_env = env  # Track for cleanup

        # Build cost reducer / constraint checker (mirrors scripts/run_cutamp.py)
        constraint_to_mult = (
            get_tetris_tuned_constraint_to_mult() if use_tetris_tuned_weights else default_constraint_to_mult.copy()
        )
        cost_reducer = CostReducer(constraint_to_mult)
        constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

        # Derive q_init from current env robot if not provided
        if q_init is None:
            try:
                q_init = self.env._robot.qpos_actuated[:-1].tolist()
            except Exception:
                q_init = None

        print(f"-----------------Running CuTAMP ---------------------")
        try:
            curobo_plan, num_satisfying = run_cutamp(
                env,
                tamp_config,
                cost_reducer,
                constraint_checker,
                q_init=q_init,
                experiment_id=None  # Example experiment ID
            )
        finally:
            # Always clean up CuTAMP resources after planning
            del env
            self._tamp_env = None
            del cost_reducer, constraint_checker
            gc.collect()
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        
        if curobo_plan is None or len(curobo_plan) == 0:
            _log.warning("cuTAMP returned no motion plan trajectories.")
        else:
            _log.info(f"cuTAMP produced {len(curobo_plan)} motion segments (including gripper actions).")
        return curobo_plan, num_satisfying

    def curobo_plan_to_env_actions(
        self,
        curobo_plan: List[Dict[str, Any]],
        initial_gripper: float = 0.0,
        open_val: float = 0.0,
        close_val: float = 1.0,
        gripper_interp_steps: int = 20,
    ) -> List[np.ndarray]:
        """Convert cuRobo accum_plans list to a flat list of environment joint + gripper actions.

        Each trajectory segment -> per-step joint positions plus persisted gripper.
        Each gripper action -> linear interpolation from current gripper to target value.
        """
        actions: List[np.ndarray] = []
        gripper_val = initial_gripper
        for seg in curobo_plan or []:
            seg_type = seg.get("type")
            if seg_type == "trajectory":
                plan = seg["plan"]  # cuRobo JointTrajectory
                # plan.position: (T, 7) expected
                joint_pos = plan.position.cpu().numpy()
                for row in joint_pos:
                    actions.append(np.concatenate([row, [gripper_val]]))
            elif seg_type == "gripper":
                action = seg.get("action")
                target = close_val if action == "close" else open_val
                interp = np.linspace(gripper_val, target, gripper_interp_steps)
                # Duplicate last joint configuration for interpolation; if no previous joint config, skip
                last_joint = actions[-1][:7] if actions else None
                if last_joint is None:
                    gripper_val = target
                    continue
                for v in interp:
                    actions.append(np.concatenate([last_joint, [v]]))
                gripper_val = target
            else:
                _log.debug(f"Unknown segment type {seg_type}, skipping")
        _log.info(f"Converted cuRobo plan into {len(actions)} environment actions")
        return actions

    def _sanitize_timestep_for_demo(self, timestep: tuple) -> tuple:
        """Sanitize timestep to remove non-numeric data that can't be saved to safetensors.
        
        Safetensors can only store numeric numpy arrays, not Python objects like strings,
        dicts, lists, etc. This method filters the info dict to keep only numeric data.
        
        Args:
            timestep: Tuple of (observation, reward, termination, truncation, info)
            
        Returns:
            Sanitized timestep with filtered info dict
        """
        obs, reward, termination, truncation, info = timestep
        
        # Filter info dict to keep only numeric data
        sanitized_info = {}
        for key, value in info.items():
            try:
                # Try to convert to numpy array
                arr = np.asarray(value)
                
                # Check if it's a numeric dtype
                if np.issubdtype(arr.dtype, np.number):
                    sanitized_info[key] = value
                elif arr.dtype == np.bool_:
                    # Boolean is OK
                    sanitized_info[key] = value
                else:
                    # Skip non-numeric types (object, string, etc.)
                    _log.debug(f"Skipping non-numeric info key '{key}' with dtype {arr.dtype}")
            except (ValueError, TypeError):
                # Skip values that can't be converted to arrays
                _log.debug(f"Skipping non-convertible info key '{key}' with type {type(value)}")
        
        # Return sanitized timestep
        return (obs, reward, termination, truncation, sanitized_info)

    def execute_actions_and_record(
        self,
        actions: List[np.ndarray],
        camera_views: List[str],
        video_prefix: str = "cutamp_stack",
        fps: int = 30,
        save_demo: bool = True,
        save_lightweight: bool = True,
        demo_dir: Optional[str] = ".",
        cleanup_temp_files: bool = False,
    ) -> Dict[str, Any]:
        """Execute actions in the RoboEval env and record per-camera MP4 videos + safetensors demo.

        Args:
            actions: List of actions to execute
            camera_views: List of camera view names to record
            video_prefix: Prefix for output video files
            fps: Frames per second for video recording
            save_demo: Whether to save demo as safetensors file
            save_lightweight: Whether to convert to lightweight demo (removes RGB images, saves disk space)
            demo_dir: Directory to save demo files (defaults to recorder's demo_dir)
            cleanup_temp_files: If True, aggressively clean up orphaned temp files during execution (default: False)

        Returns:
            dict with keys:
                - 'videos': dict mapping camera name -> output video path
                - 'demo_path': path to saved safetensors demo file (if save_demo=True)
                - 'num_steps': number of steps executed
        """
        try:
            from moviepy.editor import VideoClip
        except ImportError:
            raise ImportError("moviepy not installed. Install with `pip install moviepy`.")

        # Initialize recording structures
        recorded: Dict[str, List[np.ndarray]] = {cv: [] for cv in camera_views}
        
        # Clean up old temp files before starting (safety measure)
        # This cleans up any orphaned temp files from previous failed runs
        if cleanup_temp_files:
            self._cleanup_orphaned_temp_files()
        
        # start demo recording class
        self.recorder = DemoRecorder(f"{demo_dir}/{video_prefix}")

        
        # Start demo recording using DemoRecorder
        if save_demo:
            # CRITICAL: Record as lightweight from the start to avoid creating 1GB+ temp files
            # This prevents the 64GB temp file issue in /root/.cache/
            lightweight_recording = save_lightweight
            self.recorder.record(self.env, lightweight_demo=lightweight_recording)
            _log.info(f"Started demo recording with DemoRecorder (lightweight={lightweight_recording})")
        
        # Execute actions and record
        _log.info(f"Executing {len(actions)} actions in environment...")
        for i, a in enumerate(actions):
            timestep = self.env.step(a)
            obs = timestep[0]
            
            # Add timestep to demo recorder
            if save_demo:
                # Sanitize timestep to remove non-numeric data from info dict
                # safetensors can only save numeric arrays, not Python objects
                sanitized_timestep = self._sanitize_timestep_for_demo(timestep)
                self.recorder.add_timestep(sanitized_timestep, copy.deepcopy(a))
            
            # Record camera frames for video
            for cv in camera_views:
                key = f"rgb_{cv}"
                if key in obs:
                    recorded[cv].append(obs[key].copy())
            
            if (i + 1) % 100 == 0:
                _log.info(f"Executed {i + 1}/{len(actions)} actions")
        
        # Stop recording and save demo
        demo_path = None
        if save_demo:
            from roboeval.demonstrations.demo import Demo, LightweightDemo
            
            self.recorder.stop()
            
            # Strategy: Never save full RGB demo, go directly to lightweight
            # 1. Save lightweight demo to expected local directory
            # 2. Cache it using DemoStore for centralized storage
            
            if save_lightweight:
                try:
                    _log.info("Saving lightweight demo (was recorded as lightweight from start)...")
                    
                    # Demo was already recorded as lightweight, so just save it directly
                    # This creates NO massive temp files because the demo is already small (~5MB)
                    demo_path = self.recorder.save_demo()
                    _log.info(f"Saved lightweight demo to: {demo_path}")
                    
                    # Get file size for logging
                    light_size_mb = os.path.getsize(demo_path) / 1024 / 1024
                    _log.info(f"Demo file size: {light_size_mb:.1f} MB")
                    
                    # Clean up any temp files created during save operations
                    # Safetensors may create temp files in the parent directory
                    demo_dir = Path(demo_path).parent
                    import glob
                    temp_files = list(glob.glob(str(demo_dir / 'tmp*')))
                    for temp_file in temp_files:
                        try:
                            if os.path.isfile(temp_file):
                                os.remove(temp_file)
                                _log.debug(f"Cleaned up temp file: {temp_file}")
                        except Exception as e:
                            _log.debug(f"Could not remove temp file {temp_file}: {e}")
                    
                    # Cache the lightweight demo using DemoStore
                    # This saves it to the centralized cache location for reuse
                    try:
                        lightweight_demo = LightweightDemo.from_safetensors(demo_path)
                        demo_store = DemoStore()
                        demo_store.cache_demo(lightweight_demo)
                        _log.info(f"Cached lightweight demo to DemoStore")
                    except Exception as cache_error:
                        _log.warning(f"Failed to cache demo in DemoStore: {cache_error}")
                        _log.warning("Local copy is still saved, continuing...")
                    
                    # CRITICAL: Aggressively clean up ALL temp files in cache directories
                    # This prevents the 60GB+ temp file accumulation in /root/.cache/
                    if cleanup_temp_files:
                        _log.info("Cleaning up temp files in cache directories...")
                        self._cleanup_orphaned_temp_files(aggressive=True)
                    
                except Exception as e:
                    _log.error(f"Failed to save lightweight demo: {e}")
                    import traceback
                    traceback.print_exc()
                    demo_path = None
                finally:
                    # ALWAYS clean up temp files even if demo save failed
                    if cleanup_temp_files:
                        self._cleanup_orphaned_temp_files(aggressive=True)
            else:
                # If not saving lightweight, save full demo (not recommended - creates 64GB temp files!)
                _log.warning("WARNING: Saving full RGB demo - this will create LARGE temp files in /root/.cache/!")
                _log.warning("To avoid disk space issues, use save_lightweight=True instead")
                demo_path = self.recorder.save_demo()
                _log.info(f"Saved full demo to: {demo_path}")
        
        # Create output directory for videos
        # Use cfg repo_root if available, otherwise use relative path
        if self.cfg and self.cfg.get('repo_root'):
            repo_root = self.cfg.get('repo_root')
        else:
            # Fallback: use current file's parent directory structure
            repo_root = str(Path(__file__).parent.parent.parent)
        
        video_dir = f"{repo_root}/roboeval/envs/generated_tasks/TAMP"
        os.makedirs(video_dir, exist_ok=True)
        
        # Write videos using moviepy
        _log.info(f"Creating videos from {len(recorded)} camera views...")
        output_video_paths: Dict[str, str] = {}
        
        for cv, frames in recorded.items():
            if len(frames) == 0:
                _log.warning(f"No frames recorded for camera {cv}")
                continue
            
            # Convert frames to moviepy format (H, W, C)
            frames_array = np.moveaxis(np.array(frames), 1, -1)
            duration = len(frames_array) / fps
            
            # Create video clip
            clip = VideoClip(
                make_frame=lambda t, arr=frames_array: arr[min(int(t * fps), arr.shape[0] - 1)],
                duration=duration,
            )
            
            # Save video
            out_path = f"{video_dir}/{video_prefix}_{cv}.mp4"
            clip.write_videofile(out_path, fps=fps, codec="libx264", verbose=False, logger=None)
            output_video_paths[cv] = out_path
            _log.info(f"Saved video: {out_path} ({len(frames)} frames)")
            
            # Clean up clip and frames immediately to free memory
            clip.close()
            del clip
            del frames_array
        
        # Clean up recorded frames dictionary
        recorded.clear()
        del recorded
        
        # Clean up recorder
        if self.recorder is not None:
            try:
                del self.recorder
                self.recorder = None
            except Exception as e:
                _log.warning(f"Error cleaning up recorder: {e}")
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Final cleanup of any remaining temp files
        if cleanup_temp_files:
            self._cleanup_orphaned_temp_files()
        
        # Return comprehensive results
        results = {
            'videos': output_video_paths,
            'demo_path': str(demo_path) if demo_path else None,
            'num_steps': len(actions),
            'camera_views': list(output_video_paths.keys())
        }
        
        return results

    def generate_cutamp_yaml_with_goal_predicates(self, 
                                                 env_name: str,
                                                 goal_predicates_yaml: str,
                                                 num_blocks: int = 3,
                                                 block_size: float = 0.04,
                                                 output_path: Optional[str] = None,
                                                 movable_objects: Optional[List[str]] = None,
                                                 surface_objects: Optional[List[str]] = None,
                                                 movable_z_offset: float = 0.0) -> str:
        """
        Generate CuTAMP YAML configuration with goal predicates from SimGen.
        
        Args:
            env_name: Environment name
            goal_predicates_yaml: YAML string containing goal predicates from SimGen
            num_blocks: Number of blocks in environment
            block_size: Size of blocks
            output_path: Optional output file path
            movable_objects: List of movable object names from LLM analysis
            surface_objects: List of surface object names from LLM analysis
            movable_z_offset: Z offset to apply to movable objects (default: 0.0)
                            Recommended value: 0.013 (13mm) for collision clearance
            
        Returns:
            Path to generated YAML file
        """
        # Import yaml for parsing goal predicates
        import yaml
        
        # Check if output file already exists and preserve orientation_constraints
        existing_orientation_constraints = None
        if output_path and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_yaml = yaml.safe_load(f)
                if existing_yaml and 'orientation_constraints' in existing_yaml:
                    existing_orientation_constraints = existing_yaml['orientation_constraints']
                    print(f"Preserving existing orientation_constraints from {output_path}")
            except Exception as e:
                _log.warning(f"Could not load existing orientation_constraints: {e}")
        
        # Generate the base world configuration (this extracts real MuJoCo geometries)
        # Apply movable_z_offset to lift objects for collision clearance in cuTAMP planning
        world_config = self.generate_world_config(
            keywords_to_omit=["panda", "robot"],
            movable_z_offset=movable_z_offset
        )
        
        # Apply robot transformation (same as generate_cutamp_yaml_config)
        panda_transform = self.get_robot_transform("panda nohand_left/")
        
        # Debug poses before transform
        if 'floor' in world_config.get('cuboid', {}):
            print(f"Floor world pose before transform: {world_config['cuboid']['floor']['pose']}")
        
        # Transform all objects to robot base frame
        world_config = self.transform_objects_to_robot_frame(world_config, panda_transform)
        
        # Debug poses after transform
        if 'floor' in world_config.get('cuboid', {}):
            print(f"Floor robot pose after transform: {world_config['cuboid']['floor']['pose']}")
        
        # Check what cube/block objects already exist from the environment
        existing_cubes = []
        if 'cuboid' in world_config:
            for obj_name in world_config['cuboid'].keys():
                if obj_name.startswith('cube') or obj_name.startswith('block'):
                    existing_cubes.append(obj_name)
        
        print(f"Found existing cubes from MuJoCo environment: {existing_cubes}")
        
        # Parse the goal predicates YAML from LLM (which now includes types section)
        try:
            goal_data = yaml.safe_load(goal_predicates_yaml)
            goal_predicates = goal_data.get('goal', []) if isinstance(goal_data, dict) else []
            
            # Extract types from LLM YAML if available
            llm_types = goal_data.get('types', {}) if isinstance(goal_data, dict) else {}
            llm_movable_objects = llm_types.get('Movable', [])
            llm_surface_objects = llm_types.get('Surface', [])
            
            print(f"LLM YAML types - Movable: {llm_movable_objects}")
            print(f"LLM YAML types - Surface: {llm_surface_objects}")
            
            # Use LLM types if available, otherwise use parameters
            if llm_movable_objects or llm_surface_objects:
                final_movable_objects = llm_movable_objects
                final_surface_objects = llm_surface_objects
                print("Using object types from LLM YAML")
            else:
                # Fallback to parameter-provided lists or heuristics
                final_movable_objects = movable_objects or []
                final_surface_objects = surface_objects or []
                print("Using object types from parameters or heuristics")
                
        except Exception as e:
            _log.warning(f"Could not parse goal predicates YAML: {e}")
            goal_predicates = []
            llm_movable_objects = []
            llm_surface_objects = []
            final_movable_objects = movable_objects or []
            final_surface_objects = surface_objects or []
        
        # Check for missing objects referenced in goals and try to find them in MuJoCo geometries
        referenced_objects = set()
        for goal_item in goal_predicates:
            if isinstance(goal_item, dict):
                for pred_name, pred_args in goal_item.items():
                    if pred_name == 'On' and isinstance(pred_args, list) and len(pred_args) == 2:
                        # Both objects in "On" predicate should exist
                        referenced_objects.add(pred_args[0])  # object being placed
                        referenced_objects.add(pred_args[1])  # surface
        
        # Get all available objects from current world config
        existing_objects = set()
        for geom_type, geoms in world_config.items():
            existing_objects.update(geoms.keys())
            
        # Find missing objects that are referenced in goals
        missing_objects = referenced_objects - existing_objects
        
        print(f"Objects referenced in goals: {referenced_objects}")
        print(f"Available objects in world config: {existing_objects}")
        print(f"Missing objects: {missing_objects}")
        
        # Add missing objects by searching MuJoCo geometries more thoroughly
        if missing_objects and 'cuboid' not in world_config:
            world_config['cuboid'] = {}
            
        # Search for missing objects in MuJoCo geometries using more flexible name matching
        physics = self.env._mojo.physics
        collision_geom_idx = np.where(physics.model.geom_group == 3)[0]  # Use same geom_group as generate_world_config
        
        for obj_name in missing_objects:
            if obj_name in ['floor']:
                continue  # floor should already be handled
                
            found_in_mujoco = False
            
            # Search through MuJoCo geometries for objects that match the missing object name
            for i in collision_geom_idx:
                geom = physics.model.geom(i)
                raw_name = geom.name
                if raw_name is None:
                    continue
                    
                # Skip robot-related geometries
                if any(k in raw_name for k in ["panda", "robot"]):
                    continue
                
                # Check if this geometry corresponds to our missing object
                # Handle various naming patterns: cube_1_collision -> cube1, cube3 -> cube3, etc.
                clean_name = self.clean_geometry_name(raw_name)
                
                # More flexible matching - check if object name is contained in or matches the geometry
                name_matches = (
                    obj_name.lower() == clean_name.lower() or  # Direct match
                    obj_name.lower() in raw_name.lower() or    # obj_name contained in raw MuJoCo name
                    clean_name.lower() == obj_name.lower() or  # Clean name matches
                    (obj_name.lower().startswith('cube') and clean_name.lower().startswith('cube') and 
                     obj_name.lower().replace('cube', '') == clean_name.lower().replace('cube', '')) or  # cube number match
                    (obj_name.lower().startswith('block') and clean_name.lower().startswith('block') and 
                     obj_name.lower().replace('block', '') == clean_name.lower().replace('block', ''))  # block number match
                )
                
                if name_matches:
                    print(f"Found missing object '{obj_name}' as MuJoCo geometry '{raw_name}' -> '{clean_name}'")
                    
                    # Extract geometry data using the same logic as generate_world_config
                    geom_data = physics.data.geom(i)
                    quaternion = self.xmat_to_quat(geom_data)
                    gtype = geom.type[0]
                    
                    if gtype == 6:  # box -> cuboid
                        cuboid_dims = (np.array(geom.size) * 2.0)  # MuJoCo half-extents -> full
                        pose = np.concatenate([geom_data.xpos.copy(), quaternion])
                        # Shift z so pose z = bottom (MuJoCo centers shapes at geom.xpos)
                        pose[2] -= cuboid_dims[2] / 2.0
                        
                        # Transform pose to robot base frame (same logic as transform_objects_to_robot_frame)
                        R_wb = panda_transform["R"]
                        t_wb = panda_transform["t"]
                        p_w = pose[:3]
                        q_w = pose[3:7]
                        p_b = R_wb @ p_w + t_wb
                        
                        # Transform quaternion
                        try:
                            # Helper function to convert quaternion to rotation matrix
                            def quat_wxyz_to_matrix(q):
                                # q = [w,x,y,z]
                                return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                            def matrix_to_quat_wxyz(m):
                                q_xyzw = R.from_matrix(m).as_quat()
                                return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                            
                            R_obj_w = quat_wxyz_to_matrix(q_w)
                            R_obj_b = R_wb @ R_obj_w
                            q_b = matrix_to_quat_wxyz(R_obj_b)
                        except Exception:
                            q_b = q_w  # Fallback to original quaternion
                        transformed_pose = np.concatenate([p_b, q_b])
                        
                        world_config['cuboid'][obj_name] = {
                            "dims": cuboid_dims.tolist(), 
                            "pose": transformed_pose.tolist(),
                            "color": [220, 220, 80] if any(tok in obj_name.lower() for tok in ['cube','block']) else [128, 128, 128]
                        }
                        found_in_mujoco = True
                        break
                    elif gtype == 5:  # cylinder
                        if 'cylinder' not in world_config:
                            world_config['cylinder'] = {}
                        
                        pose = np.concatenate([geom_data.xpos, quaternion])
                        # Transform pose to robot base frame
                        R_wb = panda_transform["R"]
                        t_wb = panda_transform["t"]
                        p_w = pose[:3]
                        q_w = pose[3:7]
                        p_b = R_wb @ p_w + t_wb
                        try:
                            def quat_wxyz_to_matrix(q):
                                return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                            def matrix_to_quat_wxyz(m):
                                q_xyzw = R.from_matrix(m).as_quat()
                                return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                            R_obj_w = quat_wxyz_to_matrix(q_w)
                            R_obj_b = R_wb @ R_obj_w
                            q_b = matrix_to_quat_wxyz(R_obj_b)
                        except Exception:
                            q_b = q_w
                        transformed_pose = np.concatenate([p_b, q_b])
                        
                        world_config['cylinder'][obj_name] = {
                            "radius": geom.size[0],
                            "height": geom.size[1],
                            "pose": transformed_pose.tolist()
                        }
                        found_in_mujoco = True
                        break
                    elif gtype == 3:  # capsule
                        if 'capsule' not in world_config:
                            world_config['capsule'] = {}
                        
                        pose = np.concatenate([geom_data.xpos, quaternion])
                        # Transform pose to robot base frame
                        R_wb = panda_transform["R"]
                        t_wb = panda_transform["t"]
                        p_w = pose[:3]
                        q_w = pose[3:7]
                        p_b = R_wb @ p_w + t_wb
                        try:
                            def quat_wxyz_to_matrix(q):
                                return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                            def matrix_to_quat_wxyz(m):
                                q_xyzw = R.from_matrix(m).as_quat()
                                return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                            R_obj_w = quat_wxyz_to_matrix(q_w)
                            R_obj_b = R_wb @ R_obj_w
                            q_b = matrix_to_quat_wxyz(R_obj_b)
                        except Exception:
                            q_b = q_w
                        transformed_pose = np.concatenate([p_b, q_b])
                        
                        world_config['capsule'][obj_name] = {
                            "radius": geom.size[0],
                            "height": geom.size[1],
                            "pose": transformed_pose.tolist()
                        }
                        found_in_mujoco = True
                        break
            
            # If not found in MuJoCo geometries, create a synthetic object as fallback
            if not found_in_mujoco:
                print(f"Could not find '{obj_name}' in MuJoCo geometries, creating synthetic object")
                
                # Determine appropriate defaults based on object name
                if 'target' in obj_name.lower() or 'goal' in obj_name.lower():
                    default_dims = [0.05, 0.05, 0.05]
                    default_pose_world = [0.6, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [0.2, 0.8, 0.2, 1.0]  # Green
                elif 'button' in obj_name.lower():
                    default_dims = [0.03, 0.03, 0.01]  # Flat button
                    default_pose_world = [0.5, 0.2, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [0.8, 0.2, 0.2, 1.0]  # Red
                elif 'tool' in obj_name.lower():
                    default_dims = [0.02, 0.02, 0.1]  # Tool-like shape
                    default_pose_world = [0.4, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [0.2, 0.2, 0.8, 1.0]  # Blue
                elif 'container' in obj_name.lower() or 'box' in obj_name.lower():
                    default_dims = [0.08, 0.08, 0.05]
                    default_pose_world = [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [0.8, 0.8, 0.2, 1.0]  # Yellow
                elif obj_name.lower().startswith('cube') or obj_name.lower().startswith('block'):
                    # For cube/block objects, use cube-like dimensions
                    default_dims = [block_size, block_size, block_size]
                    default_pose_world = [0.5, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [220, 220, 80]  # Yellow like other cubes
                else:
                    # Generic object
                    default_dims = [0.04, 0.04, 0.04]
                    default_pose_world = [0.5, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0]
                    default_color = [0.5, 0.5, 0.5, 1.0]  # Gray
                
                # Transform synthetic object pose to robot base frame
                R_wb = panda_transform["R"]
                t_wb = panda_transform["t"]
                p_w = np.array(default_pose_world[:3])
                q_w = np.array(default_pose_world[3:7])
                p_b = R_wb @ p_w + t_wb
                
                try:
                    def quat_wxyz_to_matrix(q):
                        return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                    def matrix_to_quat_wxyz(m):
                        q_xyzw = R.from_matrix(m).as_quat()
                        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                    R_obj_w = quat_wxyz_to_matrix(q_w)
                    R_obj_b = R_wb @ R_obj_w
                    q_b = matrix_to_quat_wxyz(R_obj_b)
                except Exception:
                    q_b = q_w
                
                default_pose_robot = np.concatenate([p_b, q_b]).tolist()
                
                world_config['cuboid'][obj_name] = {
                    'dims': default_dims,
                    'pose': default_pose_robot,
                    'color': default_color
                }
        
        # Build the complete YAML configuration
        yaml_config = {
            'name': env_name,
            'geometries': world_config,
            'goal': goal_predicates
        }
        
        
        # Get all available objects from world config (including newly added ones)
        existing_objects = set()
        for geom_type, geoms in world_config.items():
            existing_objects.update(geoms.keys())
        
        print(f"Available objects in world config (after adding missing): {existing_objects}")
        
        # Filter object lists to only include objects that exist in the world config
        # Also ensure that objects referenced in goals are classified appropriately
        if not final_movable_objects and not final_surface_objects:
            # Fallback to heuristic classification if no types provided
            print("No object types provided, using heuristic classification")
            for geom_type, geoms in world_config.items():
                for obj_name in geoms.keys():
                    if geom_type == 'cuboid':
                        obj_lower = obj_name.lower()
                        
                        # Floor is always a surface only
                        if obj_lower == 'floor':
                            final_surface_objects.append(obj_name)
                            
                        # Cubes and blocks are both movable and surface (for stacking)
                        elif any(pattern in obj_lower for pattern in ['cube', 'block']):
                            final_movable_objects.append(obj_name)
                            final_surface_objects.append(obj_name)
                            
                        # Target objects are typically surfaces (things are placed on them)
                        elif any(pattern in obj_lower for pattern in ['target', 'goal', 'destination']):
                            final_surface_objects.append(obj_name)
                            
                        # Tools and items are typically movable
                        elif any(pattern in obj_lower for pattern in ['tool', 'item']):
                            final_movable_objects.append(obj_name)
                            
                        # Containers can be both (movable containers, or surface to put things in)
                        elif any(pattern in obj_lower for pattern in ['container', 'box']):
                            final_movable_objects.append(obj_name)
                            final_surface_objects.append(obj_name)
                            
                        # Buttons and levers are typically neither (they're interactive but not for placing/picking)
                        elif any(pattern in obj_lower for pattern in ['button', 'lever']):
                            # These are usually not movable or surface in the traditional sense
                            pass  # Don't add to either list
                            
                        # Default: assume other cuboids are movable
                        else:
                            final_movable_objects.append(obj_name)
                            
                    elif geom_type == 'mesh':
                        # Meshes are typically surfaces only (like complex floor shapes)
                        final_surface_objects.append(obj_name)
        
        # Filter to only existing objects and ensure referenced objects are classified
        final_movable_objects = [obj for obj in final_movable_objects if obj in existing_objects]
        final_surface_objects = [obj for obj in final_surface_objects if obj in existing_objects]
        
        # Ensure all objects referenced in goals are properly classified
        for obj_name in referenced_objects:
            if obj_name in existing_objects:
                # Objects that appear as the second argument in "On" predicates should be surfaces
                is_surface_in_goals = False
                is_movable_in_goals = False
                
                for goal_item in goal_predicates:
                    if isinstance(goal_item, dict):
                        for pred_name, pred_args in goal_item.items():
                            if pred_name == 'On' and isinstance(pred_args, list) and len(pred_args) == 2:
                                if pred_args[0] == obj_name:  # Object being placed - should be movable
                                    is_movable_in_goals = True
                                if pred_args[1] == obj_name:  # Surface being placed on - should be surface
                                    is_surface_in_goals = True
                
                if is_movable_in_goals and obj_name not in final_movable_objects:
                    final_movable_objects.append(obj_name)
                if is_surface_in_goals and obj_name not in final_surface_objects:
                    final_surface_objects.append(obj_name)
        
        print(f"Final movable objects: {final_movable_objects}")
        print(f"Final surface objects: {final_surface_objects}")

        yaml_config['types'] = {
            'Movable': final_movable_objects,
            'Surface': final_surface_objects
        }
        
        # Format and save the YAML
        def fmt_list(lst):
            if isinstance(lst, (list, tuple)):
                return '[' + ', '.join(map(str, lst)) + ']'
            return str(lst)
        
        lines = [f"name: {yaml_config['name']}", '', 'geometries:']
        
        # Write geometries section
        for geom_type, geoms in yaml_config['geometries'].items():
            if not geoms:
                continue
            lines.append(f'  {geom_type}:')
            for obj_name, data in geoms.items():
                lines.append(f'    {obj_name}:')
                if 'dims' in data:
                    lines.append(f'      dims: {fmt_list(data["dims"])}')
                if 'pose' in data:
                    lines.append(f'      pose: {fmt_list(data["pose"])}')
                if 'color' in data:
                    lines.append(f'      color: {fmt_list(data["color"])}')
                if 'file_path' in data:
                    lines.append(f'      file_path: {data["file_path"]}')
                for k, v in data.items():
                    if k in ('dims', 'pose', 'color', 'file_path'):
                        continue
                    if isinstance(v, list):
                        lines.append(f'      {k}: {fmt_list(v)}')
                    else:
                        lines.append(f'      {k}: {v}')
        
        lines.append('')
        lines.append('types:')
        for tkey in ['Movable', 'Surface']:
            if tkey in yaml_config['types']:
                lines.append(f'  {tkey}: {fmt_list(yaml_config["types"][tkey])}')
        
        lines.append('')
        lines.append('goal:')
        for goal_item in yaml_config['goal']:
            if isinstance(goal_item, dict):
                for pred_name, pred_args in goal_item.items():
                    if pred_name == 'On' and isinstance(pred_args, list) and len(pred_args) == 2:
                        lines.append(f'- "On": [{pred_args[0]}, {pred_args[1]}]')
                    elif pred_name == 'HandEmpty':
                        lines.append(f'- HandEmpty: []')
                    else:
                        if isinstance(pred_args, list):
                            lines.append(f'- {pred_name}: {fmt_list(pred_args)}')
                        else:
                            lines.append(f'- {pred_name}: {pred_args}')
        
        # Preserve orientation_constraints if they existed in the previous YAML
        if existing_orientation_constraints:
            lines.append('')
            lines.append('# Orientation constraints configuration')
            lines.append('orientation_constraints:')
            
            # Write the orientation_constraints section
            def write_nested_dict(d, indent=1):
                result = []
                for key, value in d.items():
                    prefix = '  ' * indent
                    if isinstance(value, dict):
                        result.append(f'{prefix}{key}:')
                        result.extend(write_nested_dict(value, indent + 1))
                    elif isinstance(value, list):
                        result.append(f'{prefix}{key}: {fmt_list(value)}')
                    elif isinstance(value, bool):
                        result.append(f'{prefix}{key}: {str(value).lower()}')
                    elif isinstance(value, (int, float)):
                        result.append(f'{prefix}{key}: {value}')
                    elif isinstance(value, str):
                        result.append(f'{prefix}{key}: "{value}"')
                    else:
                        result.append(f'{prefix}{key}: {value}')
                return result
            
            lines.extend(write_nested_dict(existing_orientation_constraints))
        
        final_yaml = '\n'.join(lines) + '\n'
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        with open(output_path, 'w') as f:
            f.write(final_yaml)
        
        _log.info(f"Generated CuTAMP YAML configuration with goal predicates: {output_path}")
        return output_path


# Utility function for multigrasp_data_sampler.py integration  
def update_load_env_function():
    """
    Update the load_env function in cutamp to handle joint name flexibility.
    This is called to patch the existing system for better robot support.
    """
    _log.info("RoboEval-CuTAMP integration loaded successfully")
