# 3D Rocket Lander — RL with Domain Randomization 🚀

Train an AI to land a 3D rocket on a landing pad using Reinforcement Learning —
the same core technique Tesla uses to train its robots (**domain randomization**),
running on your laptop with only free, open-source tools.

Built with: **PyBullet** (3D physics), **Stable-Baselines3** (PPO), **Gymnasium**.

# Rocket Lander — 3D + Full-Mission 🔥

Le Lex Fridman–style rocket-landing RL project, extended with two tracks:

| Track | What | Where |
|---|---|---|
| 3D domain-randomized lander | PPO lands a 3D rocket (PyBullet-free, NumPy physics) with curriculum-scaled domain randomization | `rocket_lander_3d/`, `train.py`, `rocket_lander_3d_colab.ipynb` |
| **Full mission** (new) | Falcon-9-style planar 3-DOF mission simulator with a real **TVC gimbal**, engine-out failures, and **Mars EDL** with parachute physics | `full_mission/` |

## Full-mission track (`full_mission/`)

One continuous mission env — `full_mission/env.py` — pure NumPy, no heavy deps:

- **TVC (Thrust Vector Control)**: the engine itself gimbals (±6° Falcon, ±8° Mars) and produces torque through a 15 m lever arm. Attitude is *not* a direct action — the agent must balance through the gimbal, exactly like a real F9.
- **Fuel-optimal landing**: finite tank (250 t on Earth, 10 t on Mars), and the terminal bonus pays for landing with fuel left (`+100 × fuel fraction`), so the policy discovers its own suicide-burn schedule.
- **Slanted atmosphere**: exponential density profile, drag with real Cd × area; climbing out burns the fuel that landing needs.
- **Engine-out failure** (earth_falcon9): observation dim 15 = `engine_failed` flag; mid-flight single-engine failures are curriculum-scaled via domain-randomization strength.
- **Mars EDL preset**: parachute with an irreversible deploy action — deploy above the dynamic-pressure limit (`max_chute_q = 8 kPa`) and the chute **rips** (−80 extra). Drop timing is a real learned skill.

Presets: `earth_falcon9` (full launch → boostback → reentry → landing) and `mars_edl`.
Start phases: `landing`, `reentry`, `boostback`, `ascent`.

### Physics validation (scripted controllers)

`full_mission/scripted_test.py` validates the simulator with hand-tuned
state-feedback controllers — a suicide-burn landing controller, a
gravity-turn ascent controller (MECO + coast back through 110 km alive),
and a chute-first Mars EDL controller (deploy at q < 8 kPa, powered descent
from 5 km with drag feed-forward):

```bash
PYTHONPATH=. python3 full_mission/scripted_test.py
# landing    : 5/5 landed, ~4-6 m/s touchdown
# full miss. : gravity turn to ~110 km, MECO, coast
# mars edl   : 5/5 landed, 64% fuel remaining
```

### Train

```bash
PYTHONPATH=. python3 full_mission/train.py --preset earth_falcon9 --start-phase landing
PYTHONPATH=. python3 full_mission/train.py --preset earth_falcon9 --start-phase ascent   # full mission
PYTHONPATH=. python3 full_mission/train.py --preset mars_edl --steps 1000000
```

Domain randomization ramps automatically (curriculum: randomization strength
0 → 1 over the first 60 % of steps, landing entry speed 0.5× → 1×). Use
`--no-randomization` for fixed physics. 1–3 M steps recommended per phase
(Colab T4 ≈ 30–60 min / M steps).

## 3D lander track

See the original README section below and `rocket_lander_3d_colab.ipynb` for
the one-click Colab training flow.

---

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
3. **Domain randomization with a curriculum** — training starts on textbook
   physics; randomization strength ramps 0 → 1 over the second half of
   training. Every episode randomizes (at full strength):
   - gravity (9.8 ± 7 m/s²), rocket mass (1.0 +20/−30%), thrust (±15%),
   - wind (±8 m/s), engine latency, sensor noise, attitude noise,
   - start position/velocity and pad location.
4. **Engine failure (single-engine-out)** — in 50% of randomized episodes the
   main engine fails mid-flight (between 20–60% of the episode), losing up to
   45% of thrust — the Falcon 9 single-engine-out scenario. The agent sees an
   `engine_health` flag in its 15-dim observation and must salvage the landing.

Because the physics is never the same twice, the policy can't memorize one
scenario — it must learn a *general* landing skill (sim-to-real readiness).

Also note: the observation includes the rocket's offset from the pad — the
agent can see where the pad is; otherwise landing on a randomized pad is
impossible.

## Quick start

```bash
pip install -r requirements.txt

python train.py                  # full training (~1M steps, ~20-40 min on a laptop)
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
