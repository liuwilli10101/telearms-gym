# Task Generation with Cursor / GitHub Copilot

Generate RoboEval manipulation tasks by injecting `Agent.md` as context into your AI coding assistant.

---

## Step 1: Inject Agent.md as Context

### Cursor

Open the chat panel (`Cmd+L` / `Ctrl+L`) and reference the Agent.md file:

```
@file:roboeval/envs/generated_tasks/Agent.md

[Your task description here]
```

Or use Composer (`Cmd+I` / `Ctrl+I`) for inline generation with the same `@file:` syntax.

### GitHub Copilot

In VS Code with Copilot Chat (`Cmd+I` / `Ctrl+I`):

```
#file:roboeval/envs/generated_tasks/Agent.md

[Your task description here]
```

Alternatively, open `Agent.md` in a split pane so Copilot sees it as context.

### Manual Injection

Copy the entire contents of `Agent.md` and paste it at the start of your prompt, followed by your task description.

---

## Step 2: Prompt for Task Generation

After injecting Agent.md, describe your task. Examples:

**Simple Task:**
```
Create a task where the robot pushes a red ColoredCube onto a green GoalPatch.
Success when the cube is within 5cm of the patch center.
```

**Multi-Object Task:**
```
Create a task where the robot stacks 3 ColoredCubes (red, blue, yellow)
into a tower. Red on bottom, blue in middle, yellow on top.
Success when all cubes are stacked and aligned within 3cm.
```

**SemanticCube Task:**
```
Create a task where the robot arranges SemanticCubes to spell "CAT"
from left to right. Success when letters are in order and facing upward.
```

**Full Prompt Example:**
```
@file:roboeval/envs/generated_tasks/Agent.md

Create a task called ColorSortingTaskEnv where:
- 3 ColoredCubes (red, green, blue) start in random positions
- 3 GoalPatches (red, green, blue) are in fixed positions
- Robot must place each cube on its matching colored patch
- Success when all cubes are on their matching patches (within 5cm)

Save as color_sorting_task.py
```

---

## Step 3: Save the Generated File

Save the generated code to:
```
roboeval/envs/generated_tasks/<your_task_name>.py
```

Use lowercase with underscores for the filename (e.g., `color_sorting_task.py`).

---

## Step 4: Test with the Launcher

### List Available Tasks
```bash
./launch_generated_task.sh --list
```

### Run Your Task
```bash
./launch_generated_task.sh <task_name>
```

### Launcher Options

| Option | Description |
|--------|-------------|
| `--steps, -s <N>` | Simulation steps (default: 10000) |
| `--robot, -r <type>` | `single` or `bimanual` (default: single) |
| `--headless` | No visualization window |
| `--reset-interval, -i <N>` | Steps between resets (default: 500) |
| `--quiet, -q` | Suppress output |
| `--help` | Show all options |

### Examples

```bash
# Quick test (1000 steps)
./launch_generated_task.sh color_sorting_task --steps 1000

# Use bimanual robot
./launch_generated_task.sh color_sorting_task --robot bimanual

# Headless mode for servers
./launch_generated_task.sh color_sorting_task --headless --quiet

# Longer run with less frequent resets
./launch_generated_task.sh color_sorting_task --steps 10000 --reset-interval 1000
```

---

## Quick Workflow Summary

```
1. Open Cursor/Copilot Chat
2. Type: @file:roboeval/envs/generated_tasks/Agent.md
3. Describe your task
4. Save output to roboeval/envs/generated_tasks/<name>.py
5. Run: ./launch_generated_task.sh <name>
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Ensure file is in `generated_tasks/` directory |
| Task not found | Check filename matches (no `.py` needed in launcher) |
| Simulation crashes | Check the generated code has all 5 required methods |
| Cubes overlap/explode | Ask AI to increase spacing in `_on_reset()` |

If the generated task has issues, tell the AI:
```
The task has [describe issue]. Fix it following the patterns in Agent.md.
```
