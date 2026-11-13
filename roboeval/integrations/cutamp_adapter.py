"""Minimal cuTAMP symbolic adapter that uses cuTAMP's domain definitions
and executes plans via the motion primitives in `roboeval.integrations.api`.

Key points:
- Uses cuTAMP's built-in domain fluents and operators from `tamp_domain.py`.
- Filters to only Pick/Place operations (no MoveFree, MoveHolding, Push, etc.).
- Does NOT invoke the full cuTAMP trajectory optimizer (relies on existing
  cuRobo primitives already validated in `api.py`).
- Supports arbitrary number of blocks present in the RoboEval environment.
- Builds a simple goal: stack the first `stack_height` blocks in order.
- Records frames from CuRoboContext and saves mp4 via moviepy.

Public API:
    build_domain(block_names) -> (fluents_dict, operators_list)
    build_problem(num_blocks, stack_height) -> (block_names, initial_state, goal_state, operators)
    enumerate_plan_skeletons(initial, goal, operators, max_skeletons)
    execute_skeleton(ctx, skeleton) -> bool
    plan_and_execute(env, stack_height=2, max_skeletons=20, video_path='cutamp_result.mp4', fps=30)

Typical usage (see demo script in examples/):
    from roboeval.integrations.cutamp_adapter import plan_and_execute
    env = make_env_with_external_camera()
    success, ctx = plan_and_execute(env, stack_height=2)
"""
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional
import numpy as np

from cutamp.task_planning import Parameter, State, task_plan_generator
from cutamp.task_planning.tamp_structs import PlanSkeleton, TAMPOperator

# Import cuTAMP domain components
from cutamp.tamp_domain import (
    On, HandEmpty, Holding, IsMovable, IsSurface, HasNotPickedUp,
    At, CanMove, JustMoved, HoldingWithGrasp,
    Pick, Place, MoveFree, MoveHolding, all_tamp_operators, get_initial_state
)

from roboeval.integrations.api import CuRoboContext, Pick as APIPick, Place as APIPlace


# ---------------------------------------------------------------------------
# Domain Construction (using cuTAMP domain)
# ---------------------------------------------------------------------------


def build_domain(block_names: List[str]):
    """Use cuTAMP's domain fluents and operators, filtering for pick/place and movement only."""
    # Filter operators to include MoveFree, MoveHolding, Pick, and Place
    # (MoveFree/MoveHolding are needed to set JustMoved() precondition for Pick/Place)
    required_operators = [op for op in all_tamp_operators if op.name in ['MoveFree', 'MoveHolding', 'Pick', 'Place']]
    
    # Return the cuTAMP fluents we need
    fluents_dict = {
        'on': On,
        'handempty': HandEmpty, 
        'holding': Holding,
        'holdingwithgrasp': HoldingWithGrasp,
        'ismovable': IsMovable,
        'issurface': IsSurface,
        'hasnotpickedup': HasNotPickedUp,
        'at': At,
        'canmove': CanMove,
        'justmoved': JustMoved,
    }
    
    return fluents_dict, required_operators


# ---------------------------------------------------------------------------
# Problem Builder
# ---------------------------------------------------------------------------


def build_problem(num_blocks: int, stack_height: int) -> Tuple[List[str], State, State, List[TAMPOperator]]:
    if num_blocks < 1:
        raise ValueError("Environment must contain at least one block")
    stack_height = max(1, min(stack_height, num_blocks))
    block_names = [f"blocks_{i + 1}" for i in range(num_blocks)]
    table = "table"
    
    fluents, operators = build_domain(block_names + [table])
    
    # Use cuTAMP's initial state builder with our objects
    # Treat blocks as both movables and surfaces (can stack on each other)
    initial = get_initial_state(
        movables=block_names,
        surfaces=[table] + block_names,  # table and blocks can be surfaces
        sticks=[],  # no sticks in our domain
        buttons=[]  # no buttons in our domain
    )
    
    # Add initial placement: all blocks start on table
    initial_atoms = set(initial)
    for block in block_names:
        initial_atoms.add(On.ground(block, table))
    initial = frozenset(initial_atoms)
    
    # Goal: stack blocks in order (block_2 on block_1, block_3 on block_2, etc.)
    goal_atoms = [HandEmpty.ground()]
    if stack_height >= 2:
        for i in range(1, stack_height):
            goal_atoms.append(On.ground(block_names[i], block_names[i-1]))
    goal: State = frozenset(goal_atoms)
    
    print(f"[Debug] Initial state ({len(initial)} atoms):")
    for atom in sorted(str(atom) for atom in initial):
        print(f"  {atom}")
    print(f"[Debug] Goal state ({len(goal)} atoms):")
    for atom in sorted(str(atom) for atom in goal):
        print(f"  {atom}")
    
    return block_names, initial, goal, operators


# ---------------------------------------------------------------------------
# Plan Enumeration
# ---------------------------------------------------------------------------


def enumerate_plan_skeletons(initial: State, goal: State, operators: List[TAMPOperator], max_skeletons: int) -> Iterable[PlanSkeleton]:
    print(f"[Debug] Starting search with {len(operators)} operators")
    print(f"[Debug] Initial state has: {sorted([str(atom) for atom in initial])}")
    print(f"[Debug] Goal state has: {sorted([str(atom) for atom in goal])}")
    
    count = 0
    for skeleton in task_plan_generator(initial, goal, operators, max_plan_skeletons=max_skeletons):
        count += 1
        print(f"[Debug] Generated skeleton {count}: {[op.name for op in skeleton]}")
        yield skeleton
        
    if count == 0:
        print("[Debug] No skeletons generated - likely precondition issues")
        # Let's check what the first step should be
        print("[Debug] Checking available first actions...")
        from cutamp.task_planning.search import get_valid_ground_operators, _Node
        initial_node = _Node(state=initial, parent=None, operator=None, depth=0)
        valid_ops = get_valid_ground_operators(initial_node, operators, verbose=True)
        print(f"[Debug] Valid operators from initial state: {len(valid_ops)}")
        for op in valid_ops[:5]:  # Show first 5
            print(f"  {op.name}")
    else:
        print(f"[Debug] Total skeletons generated: {count}")


# ---------------------------------------------------------------------------
# Execution Mapping (Symbolic -> API)
# ---------------------------------------------------------------------------


def execute_skeleton(ctx: CuRoboContext, skeleton: PlanSkeleton) -> bool:
    held: Optional[str] = None
    for op in skeleton:
        base_name = getattr(getattr(op, 'operator', None), 'name', None)
        if base_name is None:
            base_name = op.name.split('(')[0]
        base_name = base_name.lower()
        
        if base_name.startswith('movefree'):
            # MoveFree operators are handled internally by cuRobo motion planning
            print(f'[Execute] MoveFree: {op.name} (handled internally by cuRobo)')
            continue
            
        elif base_name.startswith('moveholding'):
            # MoveHolding operators are handled internally by cuRobo motion planning
            print(f'[Execute] MoveHolding: {op.name} (handled internally by cuRobo)')
            continue
        
        elif base_name.startswith('pick'):
            # cuTAMP Pick operator has parameters: [obj, grasp, q]
            # We only need the object name for our API
            obj_name = op.values[0]  # First parameter is the object
            print(f'[Execute] Pick: {obj_name}')
            if held is not None:
                print('[Execute] Already holding a block – invalid pick sequence')
                return False
            if not APIPick(ctx, obj_name):
                print(f'[Execute] Pick failed: {obj_name}')
                return False
            held = obj_name
            
        elif base_name.startswith('place'):
            # cuTAMP Place operator has parameters: [obj, grasp, placement, surface, q]
            # We need object and target surface
            obj_name = op.values[0]   # Object being placed
            surface_name = op.values[3]  # Target surface
            print(f'[Execute] Place: {obj_name} -> {surface_name}')
            
            success = False
            for pre_h, final_h in [(0.4, 0.033), (0.28, 0.035), (0.4, 0.08)]:
                if held != obj_name:
                    print(f'[Execute] Mismatch holding {held} but trying to place {obj_name}')
                if APIPlace(ctx, obj_name, surface_name, pre_h=pre_h, final_h=final_h):
                    print(f'[Execute] Place success: {obj_name} -> {surface_name}')
                    held = None
                    success = True
                    break
                    
            if not success:
                print(f'[Execute] Place failed: {obj_name} -> {surface_name}')
                return False

        else:
            print(f'[Execute] Unsupported operator: {op.name}')
            return False
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def plan_and_execute(env, stack_height: int = 2, max_skeletons: int = 20, video_path: str = 'cutamp_result.mp4', fps: int = 30):
    """Symbolically plan a stacking task and execute the first feasible skeleton.

    Returns (success, ctx).
    """
    num_blocks = len(getattr(env, 'blocks', []))
    block_names, initial, goal, operators = build_problem(num_blocks, stack_height)
    print('[Plan] Blocks:', block_names)
    print('[Plan] Goal atoms:', list(goal))

    ctx = CuRoboContext(env)
    success = False
    skeleton_count = 0
    
    print(f'[Plan] Starting skeleton enumeration with {len(operators)} operators: {[op.name for op in operators]}')
    
    for i, skel in enumerate(enumerate_plan_skeletons(initial, goal, operators, max_skeletons=max_skeletons)):
        skeleton_count += 1
        print(f'[Plan] Trying skeleton {i}: {[op.name for op in skel]}')
        if execute_skeleton(ctx, skel):
            print('[Plan] Skeleton executed successfully')
            success = True
            break
    
    if skeleton_count == 0:
        print('[Plan] No skeletons generated by task planner')
    elif not success:
        print(f'[Plan] No feasible skeleton executed (tried {skeleton_count} skeletons)')

    # Video export
    if ctx.frames:
        try:
            from moviepy.editor import VideoClip
            frames = np.moveaxis(np.array(ctx.frames), 1, -1)
            clip = VideoClip(lambda t: frames[int(min(len(frames)-1, t*fps))], duration=len(frames)/fps)
            clip.write_videofile(video_path, fps=fps, codec='libx264')
            print(f'[Video] Saved {video_path}')
        except Exception as e:
            print(f'[Video] Failed: {e}')
    else:
        print('[Video] No frames captured')

    return success, ctx


__all__ = [
    'build_domain',
    'build_problem',
    'enumerate_plan_skeletons',
    'execute_skeleton',
    'plan_and_execute',
]
