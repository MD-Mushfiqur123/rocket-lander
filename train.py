"""Train a PPO agent to land a 3D rocket with domain randomization.

Usage:
    python train.py                     # full training (~1M steps, laptop OK)
    python train.py --steps 100000      # quicker run
    python train.py --no-randomization  # fixed physics curriculum
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket_lander_3d.env import RocketLander3DEnv


class ProgressCallback(BaseCallback):
    def __init__(self, total_steps, check_freq=10000):
        super().__init__()
        self.total_steps = total_steps
        self.check_freq = check_freq

    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            recent = self.locals.get("infos", [])
            landings = sum(1 for i in recent if i.get("outcome") == "landed")
            crashes = sum(1 for i in recent if i.get("outcome") == "crashed")
            print(
                f"[{self.num_timesteps}/{self.total_steps}] "
                f"landings={landings} crashes={crashes}"
            )
        return True


def evaluate_randomized(model, episodes=20):
    env = RocketLander3DEnv(domain_randomization=True, seed=12345)
    results = {"landed": 0, "crashed": 0, "timeout": 0}
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results[info["outcome"] or "timeout"] += 1
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--model-name", default="rocket_lander_3d_ppo")
    args = parser.parse_args()

    gym.register(
        id="RocketLander3D-v0",
        entry_point="rocket_lander_3d.env:RocketLander3DEnv",
    )

    env = DummyVecEnv(
        [lambda: Monitor(RocketLander3DEnv(
            domain_randomization=not args.no_randomization
        ))]
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )
    model.learn(
        total_timesteps=args.steps,
        callback=ProgressCallback(args.steps),
        progress_bar=False,
    )
    model.save(args.model_name)
    print(f"Model saved to {args.model_name}.zip")

    print("\nEvaluating on 20 randomized episodes (unseen physics):")
    results = evaluate_randomized(model)
    total = sum(results.values())
    print(f"  landed:  {results['landed']}/{total}")
    print(f"  crashed: {results['crashed']}/{total}")
    print(f"  timeout: {results['timeout']}/{total}")
    env.close()


if __name__ == "__main__":
    main()
