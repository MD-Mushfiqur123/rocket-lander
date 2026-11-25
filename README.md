# 3D Rocket Lander — RL with Domain Randomization 🚀

Train an AI to land a 3D rocket on a landing pad using Reinforcement Learning —
the same core technique Tesla uses to train its robots (**domain randomization**),
running on your laptop with only free, open-source tools.

Built with: **PyBullet** (3D physics), **Stable-Baselines3** (PPO), **Gymnasium**.

## How it works

You never tell the AI how to fly. You give it:

1. **A brain** — a neural network policy (PPO) that reads the rocket's position,
   velocity, tilt and angular velocity, and decides how hard to fire each of
   the 3 thrusters (main engine + 2 side thrusters).
2. **A reward function** — scored every step:
   - ➕ get closer to the pad (laterally)
   - ➖ moving too fast, tilted, burning fuel
   - ➖ small time penalty
   - ➕ **+100** for a safe landing (touch the pad, speed < 1.5, tilt < 15°)
   - ➖ **−100** for crashing (touch the ground or tip over)
3. **Domain randomization** — every episode randomizes:
   - gravity (4.9 ~ 24.5 m/s²), rocket mass (0.8 ~ 1.6), thrust (±20%),
   - wind (±15 m/s), engine latency (3~10 steps), sensor noise,
   - start position/velocity and pad location.

Because the physics is never the same twice, the policy can't memorize one
scenario — it must learn a *general* landing skill (sim-to-real readiness).

## Quick start

```bash
pip install -r requirements.txt

python train.py                  # full training (~1M steps, ~1-2h on a laptop)
python train.py --steps 200000   # quicker run
python enjoy.py                  # watch the trained agent land (GUI window)
```

## Files

| File | Purpose |
|---|---|
| `rocket_lander_3d/env.py` | The 3D environment: physics, rocket, pad, rewards, domain randomization |
| `train.py` | Trains a PPO agent and evaluates it on 20 unseen randomized episodes |
| `enjoy.py` | Replays a trained model in a visual window |
| `smoke_test.py` | Runs a random agent through the env to verify it works |
| `scripted_test.py` | Proves landing is achievable with a hand-tuned PD controller |

## Verifying the environment

```bash
python smoke_test.py     # basic check
python scripted_test.py  # physics is landable (controller lands 10/10)
```

## Tuning the difficulty

Pass `domain_randomization=False` for textbook physics, or edit the ranges in
`_randomize_physics()` in `rocket_lander_3d/env.py` to make training harder or
easier. Train without randomization first, then fine-tune with it on, if the
task proves too hard at first.
