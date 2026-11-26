"""Train a PPO agent on the full-mission environment.

Missions (via --preset):
    earth_falcon9 : Falcon-9-style full mission (TVC gimbal, fuel-optimal
                    booster landing; start at any mission phase)
    mars_edl      : Mars Entry-Descent-Landing with parachute deploy timing

Phases (via --start-phase):
    landing   : booster descending toward the pad (easiest)
    reentry   : entering the atmosphere at speed
    boostback : flip-and-burn back toward the pad (hardest)
    ascent    : full launch from the pad (earth_falcon9 only)

Usage:
    python full_mission/train.py --preset earth_falcon9 --start-phase landing
    python full_mission/train.py --preset mars_edl --steps 1000000
    python full_mission/train.py --no-randomization   # curriculum off
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from full_mission.env import FullMissionEnv


class CurriculumCallback:
    """Ramps domain-randomization strength 0 -> 1 over the first 60% of steps."""

    def __init__(self, total_steps, ramp_fraction=0.6):
        self.total_steps = total_steps
        self.ramp_fraction = ramp_fraction

    def apply(self, vec_env, done_steps):
        strength = min(1.0, done_steps / max(1.0, self.ramp_fraction * self.total_steps))
        vec_env.env_method("set_randomization_strength", strength)


def evaluate(model, preset, start_phase, episodes=10):
    env = FullMissionEnv(preset, start_phase=start_phase, seed=12345)
    results = {"landed": 0, "crashed": 0, "timeout": 0}
    fuel = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results[info["outcome"] or "timeout"] += 1
        fuel.append(info["fuel_frac"])
    env.close()
    return results, sum(fuel) / max(len(fuel), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="earth_falcon9",
                        choices=["earth_falcon9", "mars_edl"])
    parser.add_argument("--start-phase", default=None,
                        choices=["landing", "reentry", "boostback", "ascent"])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="parallel envs (8 good for a laptop/Colab T4)")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    start_phase = args.start_phase
    if start_phase is None:
        start_phase = "ascent" if args.preset == "earth_falcon9" else "reentry"
    model_name = args.model_name or f"{args.preset}_{start_phase}_ppo"

    env = (SubprocVecEnv if args.n_envs > 1 else DummyVecEnv)([
        lambda: Monitor(
            FullMissionEnv(args.preset, start_phase=start_phase,
                           domain_randomization=not args.no_randomization)
        )
        for _ in range(args.n_envs)
    ])
    if not args.no_randomization:
        env.env_method("set_randomization_strength", 0.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=1024,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        verbose=1,
        device="auto",
    )

    curriculum = CurriculumCallback(args.steps)
    total = args.steps
    chunk_size = 100_000
    done_steps = 0
    while done_steps < total:
        n = min(chunk_size, total - done_steps)
        model.learn(total_timesteps=n, reset_num_timesteps=False)
        done_steps += n
        if not args.no_randomization:
            curriculum.apply(env, done_steps)
        model.save(model_name)
        results, avg_fuel = evaluate(
            model, args.preset, start_phase, episodes=5 if done_steps < total else 15
        )
        frac = done_steps / (0.6 * total) if not args.no_randomization else 0.0
        label = "full randomization" if frac >= 1 else (
            f"curriculum {frac:.0%}"
        )
        print(
            f"[{done_steps:,}/{total:,} steps | {label}] "
            f"landed={results['landed']} crashed={results['crashed']} "
            f"timeout={results['timeout']} avg_fuel_left={avg_fuel:.2f}"
        )

    print(f"\nFinal model saved to {model_name}.zip")
    env.close()


if __name__ == "__main__":
    main()
