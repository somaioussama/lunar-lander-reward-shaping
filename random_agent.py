import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset(seed=0)
while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    print(obs.round(2), round(reward, 2))
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
