"""
Utility functions for translating RoboEval task goals to CuTAMP predicates.
"""

import re
import yaml
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any


def extract_success_method(task_code: str) -> str:
    """
    Extract the _success() method from a RoboEval task class.
    
    Args:
        task_code: Complete task class code as string
        
    Returns:
        Just the _success() method code
    """
    # Find the _success method
    pattern = r'def\s+_success\(self[^)]*\):(.*?)(?=def\s+|\nclass\s+|\Z)'
    match = re.search(pattern, task_code, re.DOTALL)
    
    if match:
        success_code = f"def _success(self):{match.group(1)}"
        return success_code.strip()
    else:
        return task_code  # Return full code if can't find method


def translate_task_to_predicates(task_code: str, task_name: str = "unknown_task") -> str:
    """
    Translate RoboEval task code to TAMP goal predicates YAML.
    
    This function uses the prompt template to generate appropriate goal predicates.
    In a real implementation, this would call an LLM. For now, it provides
    example translations for common patterns.
    
    Args:
        task_code: RoboEval task code string
        task_name: Name of the task
        
    Returns:
        YAML string containing goal predicates
    """
    
    # Extract just the success method for cleaner analysis
    success_method = extract_success_method(task_code)
    
    # Pattern-based translation for common task types
    # In practice, this would be done by an LLM using the prompt template
    
    if "stack" in task_name.lower() or "stack" in success_method.lower():
        # Stacking task - typically involves On predicates and HandEmpty
        if "three" in task_name.lower() or "3" in task_name:
            return """goal:
- "On": [cube_0, table_collider_001]
- "On": [cube_1, cube_0] 
- "On": [cube_2, cube_1]
- HandEmpty: []"""
        elif "two" in task_name.lower() or "2" in task_name:
            return """goal:
- "On": [cube_0, table_collider_001]
- "On": [cube_1, cube_0]
- HandEmpty: []"""
    
    elif "move" in task_name.lower() or "place" in task_name.lower():
        # Moving/placing task - objects to target locations
        return """goal:
- "On": [cube_0, target_surface]
- HandEmpty: []"""
    
    elif "arrange" in task_name.lower():
        # Arrangement task - multiple objects in specific positions
        return """goal:
- "On": [cube_0, table_collider_001]
- "On": [cube_1, table_collider_001] 
- "On": [cube_2, table_collider_001]
- HandEmpty: []"""
    
    # Default fallback
    return """goal:
- "On": [cube_0, table_collider_001]
- HandEmpty: []"""


def create_cutamp_yaml_from_task(task_code: str, 
                                task_name: str,
                                output_path: Optional[str] = None) -> str:
    """
    Create a complete CuTAMP YAML configuration from RoboEval task code.
    
    Args:
        task_code: RoboEval task Python code
        task_name: Name for the task
        output_path: Optional output file path
        
    Returns:
        Path to generated YAML file
    """
    
    # Translate task to goal predicates
    goal_yaml = translate_task_to_predicates(task_code, task_name)
    
    # Parse the goal predicates
    try:
        goal_data = yaml.safe_load(goal_yaml)
        goal_predicates = goal_data.get('goal', [])
    except Exception as e:
        print(f"Could not parse goal YAML: {e}")
        goal_predicates = []
    
    # Create basic world configuration
    # This is a simplified version - real implementation would extract from environment
    world_config = {
        'cuboid': {
            'cube_0': {
                'dims': [0.04, 0.04, 0.04],
                'pose': [0.6, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                'color': [220, 80, 220]
            },
            'cube_1': {
                'dims': [0.04, 0.04, 0.04], 
                'pose': [0.6, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0],
                'color': [80, 220, 80]
            },
            'cube_2': {
                'dims': [0.04, 0.04, 0.04],
                'pose': [0.6, -0.1, 1.0, 1.0, 0.0, 0.0, 0.0], 
                'color': [80, 160, 220]
            }
        },
        'mesh': {
            'table_collider_001': {
                'file_path': '/path/to/table/mesh.obj',
                'pose': [0.6, 0.0, 0.92, 1.0, 0.0, 0.0, 0.0]
            }
        }
    }
    
    # Build complete YAML configuration
    yaml_config = {
        'name': task_name,
        'geometries': world_config,
        'types': {
            'Movable': ['cube_0', 'cube_1', 'cube_2'],
            'Surface': ['cube_0', 'cube_1', 'cube_2', 'table_collider_001']
        },
        'goal': goal_predicates
    }
    
    # Format as YAML text
    def fmt_list(lst):
        if isinstance(lst, (list, tuple)):
            return '[' + ', '.join(map(str, lst)) + ']'
        return str(lst)
    
    lines = [f"name: {yaml_config['name']}", '', 'geometries:']
    
    # Write geometries
    for geom_type, geoms in yaml_config['geometries'].items():
        if not geoms:
            continue
        lines.append(f'  {geom_type}:')
        for obj_name, data in geoms.items():
            lines.append(f'    {obj_name}:')
            for key, value in data.items():
                if isinstance(value, list):
                    lines.append(f'      {key}: {fmt_list(value)}')
                else:
                    lines.append(f'      {key}: {value}')
    
    # Write types
    lines.append('')
    lines.append('types:')
    for type_name, objects in yaml_config['types'].items():
        lines.append(f'  {type_name}: {fmt_list(objects)}')
    
    # Write goal
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
                    lines.append(f'- {pred_name}: {pred_args}')
    
    final_yaml = '\n'.join(lines) + '\n'
    
    # Save to file
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    with open(output_path, 'w') as f:
        f.write(final_yaml)
    
    print(f"Generated CuTAMP YAML: {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    # Example task code
    example_task = '''
class StackThreeBlocks(RoboEvalEnv):
    def _success(self):
        # Check cube_0 on table
        if not self.cubes[0].is_colliding(self.table.collider):
            return False
        # Check cube_1 on cube_0
        if not self.cubes[1].is_colliding(self.cubes[0].body):
            return False  
        # Check cube_2 on cube_1
        if not self.cubes[2].is_colliding(self.cubes[1].body):
            return False
        # Check hand empty
        if self.robot.is_gripper_holding_any_object():
            return False
        return True
'''
    
    # Generate YAML
    yaml_path = create_cutamp_yaml_from_task(
        example_task, 
        "stack_three_blocks",
        "/tmp/example_stack_task.yml"
    )
    
    # Show result
    with open(yaml_path, 'r') as f:
        print("Generated YAML:")
        print(f.read())
