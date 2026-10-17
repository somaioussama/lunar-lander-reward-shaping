from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("LunarLander-v3", n_envs=16)
model = PPO(
    "MlpPolicy", env,
    n_steps=1024, batch_size=64, n_epochs=4,
    gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    seed=1, verbose=1,
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lander_v2")
