# Lunar Lander: PPO and Reward Shaping

Training a PPO agent to land a rocket in Gymnasium's `LunarLander-v3`, then changing the reward to see how the agent's behavior changes. Built with Stable-Baselines3 on a laptop CPU.

The project asks one question: **what happens when you change what the agent is paid for?**

## Results

All scores are measured on the **plain environment** (no custom reward wrapper), over 20 episodes with deterministic actions, so my own bonuses can't inflate them. Lunar Lander counts an average of 200 or more as solved.

| Variant | Model | Mean | Std | Behavior |
|---|---|---|---|---|
| Default PPO, 1M steps | (not kept) | 125.7 | 107.9 | Inconsistent: some landings, many crashes |
| **Baseline (tuned PPO)** | `ppo_lander_v2` | **251.8** | 22.1 | Lands reliably |
| TimeFee (-0.02 per frame) | `ppo_timefee` | 214.2 | 79.4 | Lands, but less consistently |
| HoverBonus (+1 per frame) | `ppo_hover` | -13.3 | 46.6 | Slow, drifting descent; rarely lands well |

Clips for each model are in `videos_ppo_lander_v2/`, `videos_ppo_timefee/` and `videos_ppo_hover/`.

> Each variant was trained with a single seed (`seed=1`), so small differences between variants, such as baseline vs. TimeFee, could be training noise rather than a real effect.

## What I did

**1. Baseline.** Default PPO settings reached only about 126 after 1M steps. Switching to settings commonly used for this environment (16 parallel environments, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`) raised the score to 251.8 with a much lower spread, which means landings became reliable instead of lucky.

**2. TimeFee.** A small penalty every frame, so waiting costs something. The rocket still learned to land (214.2), but with a higher spread across episodes than the baseline.

**3. HoverBonus (the reward-hacking experiment).** A +1 bonus for every frame the rocket is upright and slow. The idea is that a reward for *sitting in a good state* can be worth more than finishing the task, because landing ends the episode and the bonus stops.

What happened: the agent's training score (which includes the bonus) reached about 100 to 125, while its score on the real task fell to -13.3. The rocket learned to descend slowly to collect bonus, and since the bonus ignores horizontal position, it drifted away from the pad. A +1 bonus was not strong enough to produce a full hover, so this was a **partial** exploit. The gap between the wrapped training score and the plain score is the key finding: the agent optimized the reward I wrote, not the task I meant.

## Planned

- **Stronger exploit:** a larger bonus (for example +3 per frame) to test whether the size of the bonus decides whether the hack becomes a full hover. Check episode lengths near 1000 to confirm.
- **The fix:** `ImprovementBonus` pays for the *change* in how good the state is between frames, plus a small per-frame fee, so hovering earns nothing. The code is already in `wrappers.py`, and the training run is still to do.
- Multiple seeds per variant, with mean and std across seeds.

## Setup

Requires Python 3.10 to 3.12 (developed on 3.11).

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install swig
pip install "gymnasium[box2d]" stable-baselines3 moviepy
```

If the Box2D install fails with a `swig` error, install `swig` first and retry.

## Usage

```bash
python random_agent.py              # watch an untrained rocket crash
python train_v2.py                  # train the baseline -> ppo_lander_v2.zip
python train_timefee.py             # train with the time fee -> ppo_timefee.zip
python train_hover.py               # train with the hover bonus -> ppo_hover.zip
python evaluate.py ppo_lander_v2    # score on the plain env + record 3 clips
```

`evaluate.py` takes a model name, prints the mean and std over 20 episodes, and saves clips to `videos_<model name>/`.

## Files

| File | Purpose |
|---|---|
| `train_v2.py` | Baseline PPO training with tuned settings |
| `train_timefee.py`, `train_hover.py` | Same settings, different reward wrapper |
| `wrappers.py` | Custom reward wrappers: `TimeFee`, `HoverBonus`, `ImprovementBonus` |
| `evaluate.py` | Honest scoring on the unwrapped environment, plus video recording |
| `train.py`, `random_agent.py` | First-run scripts: default PPO and random actions |

## Takeaways

- The agent learns exactly what you reward, not what you intend.
- A model's training score and its real score can disagree, so always evaluate on the plain environment and watch the clips, not just the numbers.
- Settings matter: the same algorithm went from 126 to 252 with tuned hyperparameters.
- Training is random, so one run is not enough to call a reward good or bad.

## Credits

Based on a project guide by Hari (@harixganesh). Built with [Gymnasium](https://gymnasium.farama.org/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
