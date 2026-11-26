"""Smoke test: verify the environment runs with a random agent."""

import numpy as np

from rocket_lander_3d.env import RocketLander3DEnv


def main():
    for randomized in (True, False):
        env = RocketLander3DEnv(domain_randomization=randomized, seed=0)
        obs, info = env.reset()
        assert obs.shape == (14,), f"bad obs shape {obs.shape}"
        steps = 0
        total_reward = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            total_reward += reward
            done = terminated or truncated
            steps += 1
        print(f"randomized={randomized}: steps={steps} "
              f"reward={total_reward:.1f} outcome={info['outcome']}")
        env.reset()
        for _ in range(50):
            env.step(np.zeros(3, dtype=np.float32))
        env.close()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
