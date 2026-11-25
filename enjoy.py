"""Watch a trained agent land the 3D rocket.

Usage:
    python enjoy.py                         # randomized physics, GUI window
    python enjoy.py --fixed-physics         # textbook physics
    python enjoy.py --model rocket_lander_3d_ppo --episodes 5
"""

import argparse

from stable_baselines3 import PPO

from rocket_lander_3d.env import RocketLander3DEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rocket_lander_3d_ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fixed-physics", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = RocketLander3DEnv(
        render_mode="human",
        domain_randomization=not args.fixed_physics,
    )

    results = {"landed": 0, "crashed": 0, "timeout": 0}
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        outcome = info["outcome"] or "timeout"
        results[outcome] += 1
        print(f"Episode {episode + 1}: {outcome} "
              f"(speed={info['speed']:.2f}, tilt={info['tilt']:.1f}°)")

    print(f"\nResults: {results}")
    env.close()


if __name__ == "__main__":
    main()
