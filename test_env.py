import gymnasium as gym
import highway_env

env = gym.make("highway-v0", render_mode="human")
obs, info = env.reset()

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("환경 세팅 성공!")