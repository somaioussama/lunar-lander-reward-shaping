import gymnasium as gym

class TimeFee(gym.Wrapper):
    """Small penalty every frame, so waiting costs something."""
    def __init__(self, env, fee=0.02):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info