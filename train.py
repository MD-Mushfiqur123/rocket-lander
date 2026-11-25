"""Train a PPO agent to land a 3D rocket with domain randomization.

Domain randomization strength ramps up automatically (curriculum): training
starts on textbook physics and randomization gradually reaches full strength,
so the policy learns a general landing skill instead of failing on day one.

Usage:
    python train.py                     # full training (~1M steps, laptop OK)
    python train.py --steps 100000      # quicker run
    python train.py --no-randomization  # curriculum disabled
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket_lander_3d.env import RocketLander3DEnv


class CurriculumCallback(BaseCallback):
    def __init__(self, total_steps, ramp_fraction=0.6, eval_steps=10):
        super().__init__()
        self.total_steps = total_steps
        self.ramp_fraction = ramp_fraction
        self.eval_steps = eval_steps

    def _on_step(self):
        strength = min(
            1.0, self.num_timesteps / max(1.0, self.ramp_fraction * self.total_steps)
        )
        for wrapped in self.training_env.envs:
            inner = wrapped.env
            if hasattr(inner, "set_randomization_strength"):
                inner.set_randomization_strength(strength)
        return True


def evaluate_randomized(model, episodes=10):
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

    env = DummyVecEnv([
        lambda: Monitor(
            RocketLander3DEnv(domain_randomization=not args.no_randomization)
        )
    ])
    if not args.no_randomization:
        for wrapped in env.envs:
            wrapped.env.set_randomization_strength(0.0)

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

    total = args.steps
    chunk_size = 100_000
    done_steps = 0
    while done_steps < total:
        n = min(chunk_size, total - done_steps)
        model.learn(total_timesteps=n, reset_num_timesteps=False)
        done_steps += n
        if not args.no_randomization:
            for wrapped in env.envs:
                wrapped.env.set_randomization_strength(
                    min(1.0, done_steps / (0.6 * total))
                )
        model.save(args.model_name)
        results = evaluate_randomized(
            model, episodes=5 if done_steps < total else 15
        )
        label = "full randomization" if done_steps / (0.6 * total) >= 1 else "curriculum"
        print(
            f"[{done_steps:,}/{total:,} steps | {label}] "
            f"landed={results['landed']} crashed={results['crashed']} "
            f"timeout={results['timeout']}"
        )

    print(f"\nFinal model saved to {args.model_name}.zip")
    env.close()


if __name__ == "__main__":
    main()
