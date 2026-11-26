"""Verify a scripted controller can actually reach the 'landed' outcome."""

import numpy as np

from rocket_lander_3d.env import MAX_SIDE_THRUST, MAX_THRUST, RocketLander3DEnv

HOVER_THRUST_N = 1.0 * 9.8 + 4 * 0.05 * 9.8


def scripted_action(obs):
    pos = obs[0:3]
    vel = obs[3:6]
    roll, pitch = obs[6], obs[7]
    rel = obs[12:14]

    vz_target = -2.2 if pos[2] > 2.0 else -0.35
    accel_z = 0.8 * (vz_target - vel[2])
    accel_x = np.clip(-0.6 * vel[0] - 1.2 * rel[0], -3, 3)
    accel_y = np.clip(-0.6 * vel[1] - 1.2 * rel[1], -3, 3)

    mass = 1.2
    main = np.clip((HOVER_THRUST_N + mass * accel_z) / MAX_THRUST, 0, 1)
    force_x = mass * accel_x + 0.4 * roll
    force_y = mass * accel_y + 0.4 * pitch
    net = np.array([force_x, force_y]) / MAX_SIDE_THRUST
    left = float(np.clip(net[0], 0, 1) + np.clip(net[1], 0, 1))
    right = float(np.clip(-net[0], 0, 1) + np.clip(-net[1], 0, 1))
    return np.array([main, min(left, 1.0), min(right, 1.0)], dtype=np.float32)


def main():
    env = RocketLander3DEnv(domain_randomization=False, seed=7)
    outcomes = {"landed": 0, "crashed": 0, "timeout": 0, None: 0}
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action = scripted_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
    env.close()
    print("scripted test, fixed physics:", outcomes)
    if outcomes["landed"] >= 5:
        print("LANDED OUTCOME REACHABLE — TEST PASSED")
    else:
        print("TEST FAILED: landing not reachable")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
