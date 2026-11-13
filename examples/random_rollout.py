import gymnasium as gym
import gymnasium_env.telearms_env  # registers the 100 tasks
import imageio

env = gym.make("telearms/HiCubePlacement",
               initial_pos_range=(0.45, 0.60, -0.15, 0.15),
               augmentation="appearance", render_mode='human')

obs, info = env.reset(seed=0)

for t in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    #video.append_data(env.render())
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
env.close()
