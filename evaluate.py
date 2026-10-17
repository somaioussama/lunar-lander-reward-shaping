import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

name = sys.argv[1] if len(sys.argv) > 1 else "ppo_lander_v2"
model = PPO.load(name)

env = gym.make("LunarLander-v3")
mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"{name}: mean={mean:.1f} +/- {std:.1f}")

env = RecordVideo(
    gym.make("LunarLander-v3", render_mode="rgb_array"),
    "videos_" + name,
    episode_trigger=lambda ep: True,
)
for _ in range(3):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        done = term or trunc
env.close()
