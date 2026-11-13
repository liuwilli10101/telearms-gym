# RoboEval Task Generation Agent

You are an AI agent specialized in generating custom manipulation tasks for the RoboEval robotics simulation framework. Your goal is to create well-structured Python files in `roboeval/envs/generated_tasks/` that define new manipulation environments.

---

## System Overview

RoboEval is a MuJoCo-based robotics simulation framework for manipulation tasks. Tasks inherit from `RoboEvalEnv` and optionally `MetricRolloutEval` for standardized success tracking.

---

## File Tree Structure

```
/home/carter/RoboEval/
├── roboeval/                              # Main package
│   ├── roboeval_env.py                    # Base RoboEvalEnv class
│   ├── const.py                           # ASSETS_PATH, PRESETS_PATH constants
│   ├── action_modes.py                    # JointPositionActionMode, TorqueActionMode, etc.
│   │
│   ├── envs/
│   │   ├── task_library/                  # Hand-crafted reference tasks (EXPLORE THESE)
│   │   │   ├── push_block_to_target.py
│   │   │   ├── stack_blocks.py
│   │   │   ├── rotate_cube.py
│   │   │   ├── arrange_word.py
│   │   │   ├── align_blocks.py
│   │   │   ├── handover.py
│   │   │   └── ...
│   │   │
│   │   ├── generated_tasks/               # YOUR OUTPUT DIRECTORY
│   │   │   └── [your_new_task.py]
│   │   │
│   │   ├── props/                         # Prop classes (EXPLORE THESE)
│   │   │   ├── items.py                   # Cube, ColoredCube, GoalPatch, etc.
│   │   │   ├── semantic_cube.py           # SemanticCube with labeled faces
│   │   │   ├── prop.py                    # KinematicProp base class
│   │   │   ├── preset.py                  # Preset loader
│   │   │   └── tables.py                  # Table class
│   │   │
│   │   ├── presets/                       # YAML environment configs
│   │   │   ├── bimanual_setup.yaml
│   │   │   └── ...
│   │   │
│   │   └── xmls/                          # MuJoCo XML assets
│   │       ├── world.xml                  # Base world
│   │       └── props/
│   │           ├── cube/                  # Cube XML definitions
│   │           │   ├── cube.xml
│   │           │   ├── large_cube.xml
│   │           │   ├── small_cube.xml
│   │           │   ├── unique_colored_cube.xml
│   │           │   └── ordered_cubes/     # SemanticCube XMLs per word
│   │           ├── table/
│   │           └── ...
│   │
│   ├── robots/
│   │   └── configs/
│   │       ├── panda.py                   # SinglePanda, BimanualPanda
│   │       └── ...
│   │
│   └── utils/
│       ├── metric_rollout.py              # MetricRolloutEval mixin
│       ├── physics_utils.py               # distance(), has_collided_collections()
│       ├── env_utils.py                   # get_random_points_on_plane()
│       └── cube_factory.py                # ensure_cubes() for SemanticCubes
│
└── examples/
    ├── launch_reach_target_state.py       # Example launcher script
    ├── launch_reach_target_pixels.py
    └── record_and_replay_demo.py
```

---

## Key Locations

| Component | Path |
|-----------|------|
| **Base Environment** | `roboeval/roboeval_env.py` |
| **Props (Cubes, GoalPatch)** | `roboeval/envs/props/items.py` |
| **SemanticCube** | `roboeval/envs/props/semantic_cube.py` |
| **Task Examples** | `roboeval/envs/task_library/` |
| **Generated Tasks Output** | `roboeval/envs/generated_tasks/` |
| **XML Assets** | `roboeval/envs/xmls/props/` |
| **Presets** | `roboeval/envs/presets/` |
| **Metrics Tracking** | `roboeval/utils/metric_rollout.py` |

---

## Before Generating a Task

**IMPORTANT:** Explore these directories to understand the code patterns:

1. **`roboeval/envs/task_library/`** - Read 2-3 example tasks to understand:
   - Class structure and inheritance
   - How `_initialize_env()`, `_on_reset()`, `_success()` work
   - How props are instantiated and positioned

2. **`roboeval/envs/props/items.py`** - Understand available props:
   - `Cube`, `LargeCube`, `SmallCube` - Basic cubes
   - `ColoredCube` - Runtime color-changeable cube
   - `GoalPatch` - Target zone indicator
   - Other props (Box, Rod, etc.)

3. **`roboeval/envs/props/semantic_cube.py`** - For labeled cubes:
   - `SemanticCube` - Cube with readable labels on faces
   - Face direction conventions
   - `is_face_readable()` method

---

## Cube Props Reference

### Basic Cube (from `items.py`)

```python
from roboeval.envs.props.items import Cube

cube = Cube(self._mojo)
cube.body.set_position(np.array([0.5, 0.0, 1.0]))
cube.body.set_quaternion(Quaternion(axis=[0,0,1], angle=np.pi/4).elements)
```

### ColoredCube (from `items.py`)

Runtime color-changeable cube:

```python
from roboeval.envs.props.items import ColoredCube

cube = ColoredCube(self._mojo)
cube.set_color("red")      # Set color by name
color = cube.get_color()   # Get current color

# Available colors: "red", "green", "blue", "yellow", "orange", "purple", etc.
```

### SemanticCube (from `semantic_cube.py`)

Cube with labeled faces (letters, numbers, symbols):

```python
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.utils.cube_factory import ensure_cubes

# Generate cube XMLs for a word (creates assets if needed)
cube_info = ensure_cubes("HELLO")

# Create semantic cubes
cubes = []
for char, info in cube_info.items():
    cube = SemanticCube(
        self._mojo,
        xml_path=info['xml'],
        labels=info['labels']  # 6-element list: [+X, -X, +Y, -Y, +Z, -Z]
    )
    cubes.append(cube)

# Check if a face label is readable (facing up and correctly oriented)
is_readable = cube.is_face_readable("H", up_tol=0.97, rot_tol=0.94)

# Get face normals in world frame
for normal, label in cube.face_normals_world():
    print(f"Face '{label}' pointing {normal}")
```

**Face Direction Conventions for SemanticCube:**
```python
# Face normals to semantic names:
# [0, 0, 1]  -> "top" / "up" / "+z"
# [0, 0, -1] -> "bottom" / "down" / "-z"
# [1, 0, 0]  -> "front" / "forward" / "+x"
# [-1, 0, 0] -> "back" / "backward" / "-x"
# [0, 1, 0]  -> "left" / "+y"
# [0, -1, 0] -> "right" / "-y"
```

### GoalPatch (from `items.py`)

Visual target zone on surfaces:

```python
from roboeval.envs.props.items import GoalPatch

target = GoalPatch(
    self._mojo,
    color="red",
    size=(0.07, 0.07),
    thickness=0.001,
    collision_enabled=True,
    name="target_zone"
)

target.set_position(np.array([0.5, 0.0, 0.95]))
target.set_color("green")  # Change color
is_touching = target.is_colliding(cube)
```

---

## Task Template

```python
"""
[Task Name]: [Brief description of what the task does]

Success Criteria:
- [Criterion 1]
- [Criterion 2]
"""

from abc import ABC
import numpy as np
from pyquaternion import Quaternion

from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH
from roboeval.envs.props.items import Cube, ColoredCube, GoalPatch
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes


class MyCustomTaskEnv(RoboEvalEnv, MetricRolloutEval, ABC):
    """
    Task Description.

    The robot must [describe objective].
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    _success_check = True
    _final_metrics = {}

    def _initialize_env(self):
        """Called once when environment is created. Set up persistent objects."""
        # Get table from preset
        self.table = self._preset.get_props(Table)[0]

        # Create task objects
        self.blocks = [Cube(self._mojo) for _ in range(3)]

        # Create target zone
        self.target = GoalPatch(
            self._mojo,
            color="green",
            size=(0.08, 0.08),
            name="goal"
        )

        # Initialize metric tracking
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True
        )

    def _on_reset(self):
        """Called on every env.reset(). Randomize positions, reset state."""
        # Generate random positions on table surface
        table_center = np.array([0.5, 0.0, 0.95])
        table_extents = np.array([0.15, 0.25])

        points = get_random_points_on_plane(
            n=len(self.blocks),
            center=table_center,
            extents=table_extents,
            step=0.06
        )

        # Position blocks with random rotations
        for block, point in zip(self.blocks, points):
            block.body.set_position(point)
            angle = np.random.uniform(-np.pi, np.pi)
            block.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=angle).elements
            )

        # Position target
        self.target.set_position(np.array([0.55, 0.15, 0.95]))

        # Reset success state
        self._success_check = True

        # Reinitialize metrics for new episode
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.blocks,
            slip_sample_window=20,
            robot=self.robot,
            track_collisions=True
        )

        # Reset stage tracking
        for stage in range(1, 4):
            self._metric_stage(stage, False)

    def _on_step(self):
        """Called every env.step(). Update metrics."""
        self._metric_step()

    def _success(self) -> bool:
        """Check if task is complete. Return True if successful."""
        self._success_check = True

        # Example: Check if all blocks are on target
        for block in self.blocks:
            dist = distance(block.body, self.target.body)
            if dist > 0.05:
                self._success_check = False

        # Track progress stages for partial credit
        # Stage 1: Robot grasped an object
        for block in self.blocks:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(block, side):
                    self._metric_stage(1, True)
                    break

        # Stage 2: At least one block on target
        blocks_on_target = sum(
            1 for b in self.blocks
            if distance(b.body, self.target.body) < 0.05
        )
        if blocks_on_target >= 1:
            self._metric_stage(2, True)

        # Stage 3: All blocks on target (success)
        if self._success_check:
            self._metric_stage(3, True)

        # Finalize metrics
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            blocks_on_target=blocks_on_target,
            target_distances={
                f"block_{i}": distance(block.body, self.target.body)
                for i, block in enumerate(self.blocks)
            }
        )

        return self._success_check

    def _fail(self) -> bool:
        """Check for failure conditions (optional)."""
        # Example: Block fell off table
        for block in self.blocks:
            if block.body.get_position()[2] < 0.5:
                return True
        return False

    def _get_task_info(self) -> dict:
        """Return metrics dictionary for logging."""
        return getattr(self, "_final_metrics", {})
```

---

## Launching a Simulation

Reference: `examples/launch_reach_target_state.py`

```python
import numpy as np

from roboeval.robots.configs.panda import SinglePanda
from roboeval.action_modes import JointPositionActionMode
from roboeval.roboeval_env import ObservationConfig, CameraConfig

# Import your task
from roboeval.envs.generated_tasks.my_custom_task import MyCustomTaskEnv


def main():
    # Configure action mode
    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[],
        block_until_reached=False
    )

    # Configure observations (optional cameras)
    obs_config = ObservationConfig(
        cameras=[
            CameraConfig(
                name="external",
                rgb=True,
                depth=False,
                width=640,
                height=480
            )
        ]
    )

    # Create environment
    env = MyCustomTaskEnv(
        action_mode=action_mode,
        render_mode="human",  # "human" for visualization, "rgb_array" for headless
        robot_cls=SinglePanda,
        observation_config=obs_config,
    )

    # Run simulation
    obs, info = env.reset()

    for step in range(10000):
        # Random actions for testing (replace with policy)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
```

---

## Quick Launch Script

For testing generated tasks, use the provided launcher scripts:

### Bash Script (Recommended)

From the RoboEval root directory:

```bash
# List all available tasks
./launch_generated_task.sh --list

# Launch a specific task
./launch_generated_task.sh my_custom_task

# Launch with options
./launch_generated_task.sh my_custom_task --steps 5000 --robot bimanual

# Run headless (no visualization)
./launch_generated_task.sh my_custom_task --headless --quiet
```

### Python Script

```bash
# From the generated_tasks directory
python launch_task.py my_custom_task

# With options
python launch_task.py my_custom_task --steps 5000 --robot bimanual --headless
```

### Launcher Options

| Option | Description |
|--------|-------------|
| `--steps, -s <N>` | Number of simulation steps (default: 10000) |
| `--robot, -r <type>` | Robot: `single` or `bimanual` (default: single) |
| `--headless` | Run without visualization |
| `--reset-interval, -i <N>` | Steps between resets (default: 500, 0=disable) |
| `--random-seed <N>` | Set random seed for reproducibility |
| `--list, -l` | List all available tasks |
| `--quiet, -q` | Suppress step-by-step output |

---

## Available Robots

| Robot | Import | Description |
|-------|--------|-------------|
| `SinglePanda` | `from roboeval.robots.configs.panda import SinglePanda` | Single 7-DOF Franka Panda arm |
| `BimanualPanda` | `from roboeval.robots.configs.panda import BimanualPanda` | Two Panda arms |

---

## Key Methods on Props

### Body Methods (all props)
```python
prop.body.set_position(np.array([x, y, z]))
prop.body.get_position() -> np.ndarray
prop.body.set_quaternion(np.array([w, x, y, z]))
prop.body.get_quaternion() -> np.ndarray
```

### Collision Detection
```python
from roboeval.utils.physics_utils import distance, has_collided_collections, get_colliders

# Distance between objects
dist = distance(obj1.body, obj2.body)

# Check collision
colliders_1 = get_colliders(obj1)
colliders_2 = get_colliders(obj2)
collided = has_collided_collections(self._mojo.physics, colliders_1, colliders_2)
```

### Robot Gripper
```python
# Check if robot is holding an object
from roboeval.robots.hand_side import HandSide

is_holding = self.robot.is_gripper_holding_object(block, HandSide.LEFT)
# or HandSide.RIGHT for bimanual robots
```

---

## Utility Functions

### Random Position Generation
```python
from roboeval.utils.env_utils import get_random_points_on_plane

points = get_random_points_on_plane(
    n=5,                              # Number of points
    center=np.array([0.5, 0, 1.0]),   # Center of region
    extents=np.array([0.2, 0.3]),     # Half-width in X and Y
    step=0.05,                        # Minimum spacing
    bounds=None                       # Optional bounds
)
```

### SemanticCube Factory
```python
from roboeval.utils.cube_factory import ensure_cubes

# Creates cube XMLs for each character if not exists
cube_info = ensure_cubes("STACK")
# Returns: {'S': {'xml': Path, 'labels': [...]}, 'T': {...}, ...}
```

---

## Generation Guidelines

1. **File naming**: Use lowercase with underscores (e.g., `stack_colored_blocks.py`)

2. **Class naming**: Use PascalCase ending in `Env` (e.g., `StackColoredBlocksEnv`)

3. **Always include**:
   - Docstring describing the task
   - Success criteria in docstring
   - `_PRESET_PATH` pointing to appropriate preset
   - All five key methods: `_initialize_env`, `_on_reset`, `_on_step`, `_success`, `_get_task_info`

4. **Metrics**: Use `MetricRolloutEval` mixin for standardized tracking:
   - Call `_metric_init()` in `_initialize_env` and `_on_reset`
   - Call `_metric_step()` in `_on_step`
   - Use `_metric_stage(idx, flag)` to track progress
   - Call `_metric_finalize()` in `_success`

5. **Randomization**: Always randomize object positions in `_on_reset` for robust evaluation

6. **Output location**: Save generated tasks to `roboeval/envs/generated_tasks/`

---

## Quick Reference Imports

```python
# Core
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, ASSETS_PATH

# Props
from roboeval.envs.props.items import (
    Cube, LargeCube, SmallCube,
    ColoredCube, GoalPatch, Box, Rod
)
from roboeval.envs.props.semantic_cube import SemanticCube
from roboeval.envs.props.tables import Table

# Utils
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance, get_colliders, has_collided_collections
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.cube_factory import ensure_cubes

# Robots
from roboeval.robots.configs.panda import SinglePanda, BimanualPanda
from roboeval.robots.hand_side import HandSide

# Action modes
from roboeval.action_modes import JointPositionActionMode, TorqueActionMode

# Standard libraries
from abc import ABC
import numpy as np
from pyquaternion import Quaternion
```
