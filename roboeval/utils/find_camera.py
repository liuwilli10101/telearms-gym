"""
Interactive camera position finder for MuJoCo.
Move the camera in the viewer, then check terminal for computed camera XML parameters.
"""
import numpy as np
import mujoco
import time
from roboeval.envs.task_library.stack_blocks import StackTwoBlocksEnv
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import SinglePanda
from roboeval.utils.observation_config import ObservationConfig, CameraConfig


def compute_camera_params_from_viewer(viewer, model, data):
    """
    Extract camera pos and xyaxes directly from MuJoCo's camera state.
    
    Uses mjv_cameraInRoom to get the exact camera position, forward, and up
    vectors that MuJoCo uses for rendering.
    
    In MuJoCo:
    - Camera looks along its local -Z axis
    - Camera +X is "right" in pixel space
    - Camera +Y is "up" in pixel space
    - xyaxes = [cam_x_world (3), cam_y_world (3)]
    """
    cam = viewer.cam
    
    # Get the raw free camera values for reference
    lookat = np.array([cam.lookat[0], cam.lookat[1], cam.lookat[2]])
    distance = cam.distance
    azimuth_deg = cam.azimuth
    elevation_deg = cam.elevation
    
    # Create a scene and update it with current camera
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    
    # Use mjv_cameraInRoom to get the exact camera transform
    headpos = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    mujoco.mjv_cameraInRoom(headpos, forward, up, scene)
    
    # Camera frame:
    # forward = view direction (this is camera -Z)
    # up = camera Y axis
    # right = forward × up (camera X axis)
    cam_y = up / np.linalg.norm(up)
    cam_x = np.cross(forward, up)
    cam_x = cam_x / np.linalg.norm(cam_x)
    
    pos = headpos
    
    # Print debug info
    print("\n" + "="*70)
    print("FREE CAMERA STATE (via mjv_cameraInRoom)")
    print("="*70)
    print(f"Lookat:      [{lookat[0]:.3f}, {lookat[1]:.3f}, {lookat[2]:.3f}]")
    print(f"Distance:    {distance:.3f}")
    print(f"Azimuth:     {azimuth_deg:.1f}°")
    print(f"Elevation:   {elevation_deg:.1f}°")
    print()
    print(f"Camera pos (headpos): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"Forward (view dir):   [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")
    print(f"Up (from MuJoCo):     [{up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f}]")
    print()
    print(f"Camera X (right): [{cam_x[0]:.3f}, {cam_x[1]:.3f}, {cam_x[2]:.3f}]")
    print(f"Camera Y (up):    [{cam_y[0]:.3f}, {cam_y[1]:.3f}, {cam_y[2]:.3f}]")
    print()
    
    pos_str = f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}"
    xyaxes_str = f"{cam_x[0]:.3f} {cam_x[1]:.3f} {cam_x[2]:.3f} {cam_y[0]:.3f} {cam_y[1]:.3f} {cam_y[2]:.3f}"
    
    print(">>> FOR world.xml:")
    print(f'<camera name="external" mode="fixed" pos="{pos_str}" xyaxes="{xyaxes_str}"/>')
    print("="*70 + "\n")
    
    return pos_str, xyaxes_str


def print_camera_info(env):
    """Print current free camera info from environment."""
    renderer = env.mujoco_renderer
    if renderer is None:
        print("No renderer")
        return
    if not hasattr(renderer, 'viewer') or renderer.viewer is None:
        print("No viewer")
        return
    
    viewer = renderer.viewer
    model = env._mojo.model
    data = env._mojo.data
    compute_camera_params_from_viewer(viewer, model, data)


# Create environment
print("Creating environment with human render mode...")
env = StackTwoBlocksEnv(
    action_mode=JointPositionActionMode(
        floating_base=True,
        absolute=False,
        floating_dofs=[],
    ),
    robot_cls=SinglePanda,
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig(name="external", resolution=(480, 640), rgb=True, depth=False),
        ]
    ),
    render_mode="human",
)

obs, info = env.reset()
env.render()  # Open the OpenGL viewer

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Move the camera in the viewer to your desired position")
print("2. Press SPACE to print camera parameters")
print("3. Press Q to quit")
print("="*70 + "\n")

# Main loop
try:
    last_space = False
    while True:
        # Step the environment
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # Render each frame
        
        # Check keyboard
        renderer = env.mujoco_renderer
        if hasattr(renderer, 'viewer') and renderer.viewer is not None:
            viewer = renderer.viewer
            # Check for space key press
            import glfw
            if glfw.get_key(viewer.window, glfw.KEY_SPACE) == glfw.PRESS:
                if not last_space:
                    print_camera_info(env)
                    last_space = True
            else:
                last_space = False
            
            # Check for Q to quit
            if glfw.get_key(viewer.window, glfw.KEY_Q) == glfw.PRESS:
                break
        
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\nExiting...")

# Final print
print("\nFinal camera state:")
print_camera_info(env)

env.close()
