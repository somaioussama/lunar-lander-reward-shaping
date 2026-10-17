import gymnasium as gym

class TimeFee(gym.Wrapper):
    """Small penalty every frame, so waiting costs something."""
    def __init__(self, env, fee=0.02):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info
class HoverBonus(gym.Wrapper):
    """+1 per frame while upright and slow. Invites hovering."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, spin = obs[:6]
        if abs(angle) < 0.2 and abs(vx) < 0.3 and abs(vy) < 0.3:
            reward += 1.0
        return obs, reward, terminated, truncated, info
