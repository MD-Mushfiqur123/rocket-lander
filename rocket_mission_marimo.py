# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy>=1.24,<2.0",
#     "gymnasium>=0.29",
#     "stable-baselines3>=2.0",
#     "matplotlib>=3.7",
#     "pillow>=10.0",
# ]
# ///
"""Full-Mission Rocket Lander — Falcon 9 mission + TVC + Mars EDL, one molab notebook.

A planar 3-DOF mission simulator (pure NumPy — zero compiled dependencies):
- earth_falcon9: launch -> boostback -> reentry -> landing (choose a phase)
- mars_edl:      Entry-Descent-Landing with parachute deploy timing
- real Thrust Vector Control: the engine gimbals and produces torque
- fuel-optimal landing: the landing bonus pays for fuel left in the tank

Run on https://molab.marimo.io (free) or locally:
    uv run rocket_mission_marimo.py
"""

import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import base64
    import io
    import math

    import marimo as mo
    import numpy as np
    from gymnasium import Env, spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return (
        DummyVecEnv,
        Env,
        Monitor,
        PPO,
        base64,
        io,
        math,
        mo,
        np,
        spaces,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🚀 Full-Mission Rocket Lander — Falcon 9 + Mars EDL

        An AI learns a **complete rocket mission** in a pure-NumPy planar
        physics simulator. You never tell it how to fly — you only score every
        step:

        | Event | Points |
        |---|---|
        | Descend along the reference profile, null lateral offset | ➕ |
        | Burn fuel / tumble away from velocity-vector / take long | ➖ |
        | Safe landing (on pad, speed < 6-8, tilt < 10-15°) | ➕ **+150 + 100 × fuel left** |
        | Crash (miss the pad, too fast, tip over) | ➖ **−150** |
        | Deploy the Mars chute above 8 kPa dynamic pressure | ➖ **rip (−230)** |

        **Two missions:**
        - 🌍 `earth_falcon9` — launch → gravity turn → MECO → coast → landing,
          with a real **TVC gimbal**: the engine swivels ±6° and steers through
          torque on a 15 m lever arm. Attitude is a balance act, not a switch.
        - 🔴 `mars_edl` — enter at ~600 m/s, deploy the parachute at the right
          moment (too early at high dynamic pressure and it **rips**), then a
          powered retro-propulsion landing on TVC.

        The **landing bonus pays for fuel left in the tank** — so the policy
        discovers its own fuel-optimal suicide-burn schedule, just like SpaceX.

        Pick a mission + mission phase below, press train, and watch it fly.
        """
    )
    return


@app.cell
def _(mo):
    preset_ui = mo.ui.dropdown(
        options=["earth_falcon9", "mars_edl"],
        value="earth_falcon9",
        label="Mission",
    )
    phase_ui = mo.ui.dropdown(
        options=["landing", "reentry", "boostback", "ascent"],
        value="landing",
        label="Mission phase (earth_falcon9 only)",
    )
    steps_ui = mo.ui.slider(
        start=200_000, stop=5_000_000, step=100_000, value=2_000_000,
        label="Training steps",
    )
    batch_ui = mo.ui.dropdown(
        options=[1024, 4096, 16384, 65536],
        value=16384,
        label="Batch size (GPU: 16384-65536)",
    )
    extreme_ui = mo.ui.switch(
        label="🌍 Extreme dataset (randomization 0→2)", value=False,
    )
    train_button = mo.ui.run_button(label="🚀 Train the rocket")
    mo.hstack([preset_ui, phase_ui, steps_ui, batch_ui, extreme_ui, train_button], justify="start")
    return (preset_ui, phase_ui, steps_ui, batch_ui, extreme_ui, train_button)


@app.cell
def _(Env, math, np, spaces):
    """Full-mission environment (embedded verbatim from full_mission/env.py)."""

    PRESETS = {
        "earth_falcon9": {
            "gravity": 9.8, "planet_radius": 6_371_000.0,
            "rho0": 1.2, "scale_height": 8500.0, "cd": 1.0,
            "body_area": 10.75, "chute_area": 0.0, "chute_cd": 0.0,
            "dry_mass": 24_000.0, "fuel_mass": 250_000.0,
            "max_thrust": 7.0e6, "isp": 300.0, "gimbal_deg": 6.0,
            "lever_arm": 15.0, "inertia": 3.65e7, "body_length": 40.0,
            "meco_altitude": 55_000.0, "pad_half_width": 30.0,
            "speed_tol": 6.0, "tilt_tol": 10.0,
            "has_chute": False, "max_chute_q": None,
            "wind_speed": 8.0, "start_phase": "landing",
            "entry_altitude": 60_000.0, "entry_vz": -1000.0, "entry_vx": 250.0,
            "entry_x_jitter": 3000.0,
        },
        "mars_edl": {
            "gravity": 3.71, "planet_radius": 3_389_500.0,
            "rho0": 0.020, "scale_height": 11_100.0, "cd": 1.4,
            "body_area": 15.9, "chute_area": 500.0, "chute_cd": 1.5,
            "dry_mass": 2_400.0, "fuel_mass": 10_000.0,
            "max_thrust": 500_000.0, "isp": 250.0, "gimbal_deg": 8.0,
            "lever_arm": 6.0, "inertia": 2.0e5, "body_length": 12.0,
            "meco_altitude": 25_000.0, "pad_half_width": 30.0,
            "speed_tol": 8.0, "tilt_tol": 15.0,
            "has_chute": True, "max_chute_q": 8000.0,
            "wind_speed": 0.0, "start_phase": "reentry",
            "entry_altitude": 20_000.0, "entry_vz": -600.0, "entry_vx": -120.0,
            "entry_x_jitter": 1200.0,
        },
    }

    DT = 0.2
    SUBSTEPS = 2
    SUCCESS_OMEGA = 0.25
    HOVER_POINT_ALT = 2500.0
    TOUCHDOWN_MARGIN = 0.2

    class FullMissionEnv(Env):
        """Planar 3-DOF full-mission environment with TVC gimbal and fuel.

        Actions (Box 0..1):
            earth_falcon9: [gimbal command, throttle, unused]
            mars_edl:      [gimbal command, throttle, chute deploy]

        Observation (normalized): [x offset, z, vx, vz, pitch, pitch-rate,
        fuel fraction, (chute deployed), phase one-hot (ascent/descent)].
        """

        metadata = {"render_modes": [], "render_fps": 5}

        def __init__(
            self,
            preset="earth_falcon9",
            start_phase=None,
            domain_randomization=True,
            max_episode_steps=3500,
            seed=None,
        ):
            super().__init__()
            if preset not in PRESETS:
                raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
            self.preset_name = preset
            self.cfg = dict(PRESETS[preset])
            self.start_phase = start_phase or self.cfg["start_phase"]
            self.domain_randomization = domain_randomization
            self.max_episode_steps = max_episode_steps
            self.rng = np.random.default_rng(seed)
            self.rand_strength = 1.0
            self.engine_failed = False

            n_act = 4 if self.cfg["has_chute"] else 3
            self.action_space = spaces.Box(0.0, 1.0, shape=(n_act,), dtype=np.float32)

            n_obs = 7 + (1 if self.cfg["has_chute"] else 0) + 2 + 1
            obs_high = np.full(n_obs, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

            self._p = {}
            self.state = None
            self.stage = "ascent"
            self.chute_deployed = False

        def set_randomization_strength(self, strength):
            """0 = textbook physics, 1 = full standard randomization, 2 = extreme
            big-dataset mode (extra mass/drag/inertia/offset jitters on top)."""
            self.rand_strength = float(np.clip(strength, 0.0, 2.0))

        def _randomize(self):
            s = self.rand_strength if self.domain_randomization else 0.0
            sx = min(s, 1.0)
            se = max(0.0, s - 1.0)
            cfg = self.cfg

            def j(u_low, u_high):
                return (1.0 - sx) + sx * self.rng.uniform(u_low, u_high)

            phase = self.start_phase
            if phase == "landing":
                x0 = (self.rng.uniform(-200, 200) * (1.0 - sx)
                      + self.rng.uniform(-1500, 1500) * j(0.5, 2.0) * sx)
                z0 = HOVER_POINT_ALT * (0.1 + 0.9 * sx) * j(0.9, 1.15)
                vx = (self.rng.uniform(-15, 15) * (1.0 - sx)
                      + self.rng.uniform(-40, 40) * j(0.5, 2.0) * sx
                      + self.rng.uniform(-20, 20) * se)
                vz = -j(0.7, 1.3) * (self.rng.uniform(40, 60)
                                     + self.rng.uniform(150, 260) * sx)
                theta = self.rng.uniform(-8, 8) * math.pi / 180 * j(0.5, 2.0)
            elif phase == "reentry":
                jit = cfg.get("entry_x_jitter", 3000.0)
                x0 = (self.rng.uniform(-jit, jit) * j(0.5, 2.0)
                      + self.rng.uniform(-jit, jit) * se)
                z0 = cfg["entry_altitude"] * j(0.9, 1.1)
                vx = cfg["entry_vx"] * self.rng.uniform(-1.2, 1.2) * j(0.6, 1.4)
                vz = cfg["entry_vz"] * self.rng.uniform(0.8, 1.2) * j(0.8, 1.2)
                theta = self.rng.uniform(-10, 10) * math.pi / 180
            elif phase == "boostback":
                x0 = self.rng.uniform(8_000, 20_000) * j(0.6, 1.4)
                z0 = cfg["entry_altitude"] * j(0.9, 1.15)
                vx = self.rng.uniform(250, 500) * j(0.6, 1.4)
                vz = self.rng.uniform(-50, 150)
                theta = self.rng.uniform(-15, 15) * math.pi / 180
            elif phase == "ascent":
                x0, z0 = 0.0, 0.0
                vx = vz = 0.0
                theta = 0.0
            else:
                raise ValueError(f"unknown start_phase {phase!r}")

            self._p = {
                "gravity": cfg["gravity"] * (1.0 + sx * self.rng.uniform(-0.1, 0.1)),
                "mass_dry": cfg["dry_mass"],
                "fuel0": cfg["fuel_mass"] * j(0.95, 1.05),
                "thrust_scale": j(0.9, 1.1),
                "wind": self.rng.uniform(-cfg["wind_speed"], cfg["wind_speed"]) * sx,
                "pad_x": (self.rng.uniform(-3_000, 3_000) * sx
                          + self.rng.uniform(-2_000, 2_000) * se
                          if phase != "ascent" else 0.0),
                "x0": x0, "z0": z0, "vx0": vx, "vz0": vz, "theta0": theta,
            }
            if se > 0.0:
                self._p["mass_dry"] = cfg["dry_mass"] * (
                    1.0 + se * self.rng.uniform(-0.3, 0.3))
                self._p["cd"] = cfg["cd"] * (1.0 + se * self.rng.uniform(-0.3, 0.3))
                self._p["inertia"] = cfg["inertia"] * (
                    1.0 + se * self.rng.uniform(-0.3, 0.3))
                if phase in ("landing", "reentry") and self.rng.random() < 0.5:
                    self._p["engine_failure_hazard"] = self.rng.uniform(0.002, 0.01)
                    self._p["failure_zone_z"] = min(self._p["z0"] * 0.35, 900.0)

        @property
        def _fuel_frac(self):
            return float(self.state["fuel"]) / max(self._p["fuel0"], 1.0)

        def _gravity_at(self, z):
            r = self.cfg["planet_radius"]
            return self._p["gravity"] * (r / (r + max(z, 0.0))) ** 2

        def _atmo_density(self, z):
            return self.cfg["rho0"] * np.exp(-max(z, 0.0) / self.cfg["scale_height"])

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self._randomize()
            self.state = {
                "pos": np.array([self._p["x0"], self._p["z0"]], dtype=np.float64),
                "vel": np.array([self._p["vx0"], self._p["vz0"]], dtype=np.float64),
                "theta": self._p["theta0"],
                "omega": 0.0,
                "fuel": self._p["fuel0"],
            }
            self.chute_deployed = False
            self.engine_failed = False
            self.step_count = 0
            if self.start_phase == "ascent":
                self.stage = "ascent"
            else:
                self.stage = "descent"
            return self._get_obs(), {}

        def _get_obs(self):
            st = self.state
            cfg = self.cfg
            chute = 1.0 if self.chute_deployed else 0.0
            ascent = 1.0 if self.stage == "ascent" else 0.0
            obs = np.array(
                [
                    (st["pos"][0] - self._p["pad_x"]) / 1000.0,
                    st["pos"][1] / 1000.0,
                    st["vel"][0] / 100.0,
                    st["vel"][1] / 100.0,
                    st["theta"],
                    st["omega"],
                    self._fuel_frac,
                ]
                + ([chute] if cfg["has_chute"] else [])
                + [ascent, 1.0 - ascent]
                + [1.0 if self.engine_failed else 0.0],
                dtype=np.float32,
            )
            return obs

        def _thrust_and_torque(self, action):
            st = self.state
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            throttle = float(a[1])
            throttle = 0.0 if throttle < 0.05 else throttle
            if st["fuel"] <= 0.0:
                throttle = 0.0
            thrust = throttle * self.cfg["max_thrust"] * self._p["thrust_scale"]
            gimbal = (2.0 * float(a[0]) - 1.0) * math.radians(self.cfg["gimbal_deg"])
            n = np.array([-math.sin(st["theta"]), math.cos(st["theta"])])
            c, s_ = math.cos(gimbal), math.sin(gimbal)
            t_dir = np.array([n[0] * c - n[1] * s_, n[0] * s_ + n[1] * c])
            torque = -self.cfg["lever_arm"] * thrust * math.sin(gimbal)
            mdot = thrust / (self.cfg["isp"] * 9.80665)
            return thrust, t_dir, torque, mdot

        def _substep(self, action, dt):
            st = self.state
            cfg = self.cfg
            thrust, t_dir, torque, mdot = self._thrust_and_torque(action)
            mass = self._p.get("mass_dry", cfg["dry_mass"]) + st["fuel"]

            a_force = np.zeros(2)
            a_force[1] -= self._gravity_at(st["pos"][1])
            a_force += t_dir * (thrust / mass)

            v_rel = st["vel"] - np.array([self._p["wind"], 0.0])
            v_rel_mag = float(np.linalg.norm(v_rel))
            rho = self._atmo_density(st["pos"][1])
            area = cfg["body_area"] + (
                cfg["chute_area"] if self.chute_deployed else 0.0
            )
            cd_eff = self._p.get("cd", cfg["cd"]) + (
                cfg["chute_cd"] if self.chute_deployed else 0.0)
            q = 0.5 * rho * v_rel_mag ** 2
            if v_rel_mag > 1e-6:
                drag = q * cd_eff * area / mass
                a_force -= drag * (v_rel / v_rel_mag)

            st["vel"] += a_force * dt
            st["pos"] += st["vel"] * dt

            aero_torque = -2.5 * max(q, 0.0) * st["omega"] * 1e-6
            alpha = (torque + aero_torque) / self._p.get("inertia", cfg["inertia"])
            st["omega"] += alpha * dt
            st["theta"] += st["omega"] * dt
            if thrust > 0:
                st["fuel"] = max(0.0, st["fuel"] - mdot * dt)

        def step(self, action):
            self.step_count += 1
            if (not self.engine_failed
                    and self.state["pos"][1] < self._p.get("failure_zone_z", -1.0)
                    and self._p.get("engine_failure_hazard") is not None
                    and self.rng.random() < self._p["engine_failure_hazard"]):
                self.engine_failed = True
                self._p["thrust_scale"] *= 0.55  # one of the landing engines flames out
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)

            if self.cfg["has_chute"] and not self.chute_deployed and a[2] > 0.5:
                v_rel = self.state["vel"] - np.array([self._p["wind"], 0.0])
                v_rel_mag = float(np.linalg.norm(v_rel))
                rho = self._atmo_density(self.state["pos"][1])
                q = 0.5 * rho * v_rel_mag ** 2
                self.chute_deployed = True
                if self.cfg["max_chute_q"] is not None and q > self.cfg["max_chute_q"]:
                    return self._finish("crashed", chute_rip=True)

            prev = {
                "pos": self.state["pos"].copy(),
                "vel": self.state["vel"].copy(),
            }
            fuel_before = self.state["fuel"]

            for _ in range(SUBSTEPS):
                self._substep(action, DT / SUBSTEPS)
            self.state["theta"] = (self.state["theta"] + math.pi) % (2 * math.pi) - math.pi

            if self.stage == "ascent" and (
                self.state["pos"][1] >= self.cfg["meco_altitude"]
                or self.state["vel"][1] <= 0
            ):
                self.stage = "descent"

            outcome = self._check_terminal()
            reward = self._shaped_reward(prev, fuel_before, a, outcome)

            terminated = outcome in ("landed", "crashed")
            truncated = (not terminated) and self.step_count >= self.max_episode_steps
            info = {
                "outcome": outcome,
                "stage": self.stage,
                "engine_failed": self.engine_failed,
                "fuel_frac": self._fuel_frac,
                "speed": float(np.linalg.norm(self.state["vel"])),
                "tilt": math.degrees(abs(self.state["theta"])),
                "chute_aborted": outcome == "crashed" and self.chute_deployed,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        def _check_terminal(self):
            st = self.state
            if st["pos"][1] <= TOUCHDOWN_MARGIN:
                on_pad = abs(st["pos"][0] - self._p["pad_x"]) <= self.cfg["pad_half_width"]
                speed = float(np.linalg.norm(st["vel"]))
                upright = abs(st["theta"]) < math.radians(self.cfg["tilt_tol"])
                calm = abs(st["omega"]) < SUCCESS_OMEGA
                if on_pad and speed < self.cfg["speed_tol"] and upright and calm:
                    return "landed"
                return "crashed"
            if st["pos"][1] > self.cfg["entry_altitude"] * 6:
                return "timeout"
            return None

        def _shaped_reward(self, prev, fuel_before, action, outcome):
            st = self.state
            cfg = self.cfg
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            reward = -0.1

            reward -= 0.05 * float(a[1])

            if st["fuel"] <= 0 and self.stage == "descent" and st["pos"][1] > 0:
                reward -= 0.5

            if self.stage == "ascent":
                goal = np.array(
                    [self._p["pad_x"] + 0.35 * cfg["meco_altitude"], cfg["meco_altitude"]]
                )
                d_prev = np.linalg.norm(prev["pos"] - goal)
                d_now = np.linalg.norm(st["pos"] - goal)
                reward += 2e-4 * (d_prev - d_now)
                reward -= 0.02 * abs(math.degrees(st["theta"]) / 45.0)
            else:
                goal_pos = np.array([self._p["pad_x"], HOVER_POINT_ALT])
                goal_vel = np.zeros(2)
                d_prev = np.linalg.norm(prev["pos"] - goal_pos) / 1_000.0 + \
                    np.linalg.norm(prev["vel"] - goal_vel) / 150.0
                d_now = np.linalg.norm(st["pos"] - goal_pos) / 1_000.0 + \
                    np.linalg.norm(st["vel"] - goal_vel) / 150.0
                reward += 3.0 * (d_prev - d_now)
                z = st["pos"][1]
                if z < 3000.0:
                    vz_ref = -np.clip(0.06 * z, 2.0, 90.0)
                    track = min(abs(st["vel"][1] - vz_ref) / 40.0, 3.0)
                    reward -= 2.0 * track
                retro = -math.atan2(-st["vel"][0], -st["vel"][1]) if \
                    np.linalg.norm(st["vel"]) > 30 else 0.0
                err = abs((st["theta"] - retro + math.pi) % (2 * math.pi) - math.pi)
                reward -= 0.05 * err

            if outcome == "landed":
                reward += 150.0 + 100.0 * self._fuel_frac
            elif outcome == "crashed":
                reward -= 150.0
            return reward

        def _finish(self, outcome, chute_rip=False):
            reward = 150.0 if outcome == "landed" else -150.0 - (80.0 if chute_rip else 0.0)
            info = {
                "outcome": outcome, "stage": self.stage, "engine_failed": self.engine_failed,
                "fuel_frac": self._fuel_frac,
                "speed": float(np.linalg.norm(self.state["vel"])),
                "tilt": math.degrees(abs(self.state["theta"])),
                "chute_aborted": chute_rip,
            }
            return self._get_obs(), float(reward), True, False, info

        def render(self):
            return None

        def close(self):
            return None

    return (FullMissionEnv,)


@app.cell
def _(math, np):
    """Scripted reference controllers (from full_mission/scripted_test.py) used
    by the physics self-check below — proof the simulator is landable."""


    class TvcAttitude:
        """Second-order TVC attitude law: picks a gimbal deflection so the
        commanded angular acceleration drives pitch error and rate to zero."""

        def __init__(self, wn=0.5, zeta=1.0):
            self.wn = wn
            self.zeta = zeta

        def command(self, env, theta, omega, theta_target, throttle):
            cfg = env.cfg
            max_gimbal = math.radians(cfg["gimbal_deg"])
            err = (theta - theta_target + math.pi) % (2 * math.pi) - math.pi
            alpha_des = self.wn ** 2 * err + 2 * self.zeta * self.wn * omega
            thrust = throttle * cfg["max_thrust"] * env._p["thrust_scale"]
            if thrust < 1.0:
                return 0.5
            required = alpha_des * cfg["inertia"] / (cfg["lever_arm"] * thrust)
            d = float(np.clip(required, -max_gimbal, max_gimbal))
            return 0.5 + 0.5 * (d / max_gimbal)


    class LandingController:
        """Stateful suicide-burn controller for the earth_falcon9 landing phase."""

        def __init__(self, eta=1.8):
            self.eta = eta
            self.burn_started = False
            self.att = TvcAttitude()

        def __call__(self, obs, env):
            x_off, z = obs[0] * 1000.0, obs[1] * 1000.0
            vx, vz = obs[2] * 100.0, obs[3] * 100.0
            mass = env.cfg["dry_mass"] + env.state["fuel"]
            if env.state["fuel"] <= 0:
                return np.zeros(env.action_space.shape[0])
            t_max = env.cfg["max_thrust"] * env._p.get("thrust_scale", 1.0)
            g = env._gravity_at(z)
            decel = max(t_max / mass - g, 1e-6)
            need = (vz ** 2) / (2 * decel) * self.eta

            if not self.burn_started and vz < -1 and need >= max(z, 0.0) - 5.0:
                self.burn_started = True

            if self.burn_started and z > 1.0:
                vz_cap = float(np.clip(0.06 * abs(x_off) + 0.8 * abs(vx), 2.5, 40.0))
                vz_des = -np.clip(0.08 * z, 2.0, vz_cap)
                hover_thr = g * mass / t_max
                throttle = float(np.clip(hover_thr - 0.09 * (vz - vz_des), 0.0, 1.0))
            else:
                throttle = 0.0

            theta_target = float(np.clip(
                (0.02 * x_off + 0.4 * vx) * mass / t_max, -0.18, 0.18
            ))
            gimbal = self.att.command(env, obs[4], obs[5], theta_target, throttle)
            return np.array([gimbal, throttle])


    class MarsEdlController:
        """Chute-first EDL: drag-decelerate under parachute, then powered descent
        from ~5 km with TVC nulling lateral offset and a soft velocity-profile
        final descent (drag feed-forward + P on vertical speed)."""

        def __init__(self, powered_alt=5000.0, kp=0.06, kd=0.9, cap=0.18,
                     ka=0.4, v_cap=25.0):
            self.powered_alt = powered_alt
            self.kp = kp
            self.kd = kd
            self.cap = cap
            self.ka = ka
            self.v_cap = v_cap
            self.att = TvcAttitude(wn=0.5, zeta=1.0)
            self.deployed = False

        def __call__(self, obs, env):
            x_off, z = obs[0] * 1000.0, obs[1] * 1000.0
            vx, vz = obs[2] * 100.0, obs[3] * 100.0
            theta, omega = obs[4], obs[5]
            cfg = env.cfg
            mass = cfg["dry_mass"] + env.state["fuel"]
            t_max = cfg["max_thrust"] * env._p.get("thrust_scale", 1.0)
            zero = np.zeros(env.action_space.shape[0])

            v_rel = max(float(np.hypot(vx, vz)), 1e-6)
            rho = env._atmo_density(z)
            q = 0.5 * rho * v_rel ** 2

            if not self.deployed:
                if q < 0.9 * cfg["max_chute_q"] and v_rel > 50:
                    self.deployed = True
                    return np.array([0.5, 0.0, 1.0])
                return zero
            if env.state["fuel"] <= 0:
                return zero

            if z > self.powered_alt:
                return np.array([0.5, 0.0, 0.0])

            v_cap_eff = float(np.clip(0.15 * z, 3.0, self.v_cap))
            vz_des = -np.clip(0.08 * z, 2.5, v_cap_eff)
            a_cmd = self.ka * (vz_des - vz)
            cd_eff = cfg["cd"] + cfg["chute_cd"]
            area_eff = cfg["body_area"] + cfg["chute_area"]
            drag_a = 0.5 * rho * v_rel ** 2 * cd_eff * area_eff / mass
            a_drag_z = drag_a * vz / v_rel
            t_req = mass * (env._gravity_at(z) + a_cmd - a_drag_z)
            thr = float(np.clip(t_req / t_max, 0.0, 1.0))
            if thr < 0.02:
                thr = 0.0

            theta_target = float(np.clip(self.kp * x_off + self.kd * vx,
                                         -self.cap, self.cap))
            gimbal = self.att.command(env, theta, omega, theta_target, max(thr, 0.05))
            return np.array([gimbal, thr, 0.0])

    return (LandingController, MarsEdlController, TvcAttitude)


@app.cell
def _(FullMissionEnv, LandingController, MarsEdlController, mo, np):
    """Physics self-check — runs before training (pure scripted controllers)."""
    def _run_scripted(preset, phase, ctrl_cls, seeds):
        outcomes = []
        for _s in seeds:
            env = FullMissionEnv(preset, start_phase=phase, domain_randomization=False)
            ctrl = ctrl_cls()
            obs, _ = env.reset(seed=_s)
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(ctrl(obs, env))
                done = term or trunc
            env.close()
            outcomes.append(info["outcome"])
        return outcomes

    earth_out = _run_scripted("earth_falcon9", "landing", LandingController, [100, 101, 102])
    mars_out = _run_scripted("mars_edl", "reentry", MarsEdlController, [100, 101])

    def _tick(outs):
        return "✅" * outs.count("landed") + "💥" * (len(outs) - outs.count("landed"))

    mo.md(
        f"""
        ### 🧪 Physics self-check (scripted controllers — no AI, just the simulator)

        | Mission | Scripted result |
        |---|---|
        | 🌍 Falcon 9 landing (suicide burn) | {earth_out} {_tick(earth_out)} |
        | 🔴 Mars EDL (chute + retro-propulsion) | {mars_out} {_tick(mars_out)} |

        If these are ✅, the simulator is physically landable — the AI's job is to
        rediscover this schedule on its own.
        """
    )
    return


@app.cell
def _(
    DummyVecEnv,
    FullMissionEnv,
    Monitor,
    PPO,
    mo,
    np,
    batch_ui,
    extreme_ui,
    phase_ui,
    preset_ui,
    steps_ui,
    train_button,
):
    rocket_model = None

    mo.stop(
        not train_button.value,
        mo.md("\u2b06\ufe0f Pick a mission + phase, set the slider, press **\ud83d\ude80 Train the rocket**."),
    )

    _preset = preset_ui.value
    _phase = phase_ui.value if _preset == "earth_falcon9" else "reentry"
    _total = steps_ui.value

    _vec_env = DummyVecEnv([
        lambda: Monitor(
            FullMissionEnv(_preset, start_phase=_phase,
                           domain_randomization=True)
        )
        for _ in range(4)
    ])
    _vec_env.env_method("set_randomization_strength", 0.0)

    _nsteps = max(4096, batch_ui.value)
    rocket_model = PPO(
        "MlpPolicy",
        _vec_env,
        learning_rate=3e-4,
        n_steps=_nsteps,
        batch_size=batch_ui.value,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 512]),
        verbose=0,
        device="auto",
    )

    _ramp = max(1, int(0.5 * _total))
    _chunk = _nsteps * 4
    import time as _time
    _t0 = _time.time()
    with mo.status.progress_bar(
        total=_total,
        title="Training PPO \u2014 randomization ramps 0\u21921 (then \u21922 extreme) over 50% of training",
    ) as _bar:
        for _c0 in range(0, _total, _chunk):
            _strength = min(1.0, _c0 / _ramp)
            if extreme_ui.value:
                _strength += max(0.0, min(1.0, (_c0 - _ramp) / _ramp))
            _vec_env.env_method("set_randomization_strength", _strength)
            rocket_model.learn(
                total_timesteps=min(_chunk, _total - _c0),
                reset_num_timesteps=False,
            )
            _bar.update(increment=min(_chunk, _total - _c0))

    rocket_model.save("full_mission_ppo")

    _eval_env = FullMissionEnv(_preset, start_phase=_phase, seed=424242)
    _outs = []
    for _ in range(5):
        _obs, _ = _eval_env.reset()
        _done = False
        while not _done:
            _act, _ = rocket_model.predict(_obs, deterministic=True)
            _obs, _r, _term, _trunc, _info = _eval_env.step(_act)
            _done = _term or _trunc
        _outs.append(_info["outcome"] or "timeout")
    _eval_env.close()
    _nok = _outs.count("landed")
    _secs = _time.time() - _t0

    _summary = mo.md(
        f"""
        ### \u2705 Training complete
        - **Mission:** `{_preset}` / `{_phase}` \u2014 **{_total:,} steps** ({_secs:.0f} s)
        - Model saved as `full_mission_ppo.zip`
        - **Held-out eval (unseen randomized starts): {_nok}/5 landed** ({_outs})

        Tip: landings usually appear first on the `landing` phase; for full
        missions (`ascent`, `boostback`) train 1.5M+ steps \u2014 press the button
        again with a bigger slider to keep training the same policy.
        """
    )
    _summary
    return (rocket_model,)


@app.cell
def _(FullMissionEnv, base64, io, math, mo, np, phase_ui, preset_ui,
      rocket_model, train_button):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.figure
    import matplotlib.patches as mpatches
    from PIL import Image

    mo.stop(
        rocket_model is None or not train_button.value,
        mo.md("\U0001f512 Train first \u2014 then this cell films the mission on camera."),
    )

    _preset = preset_ui.value
    _phase = phase_ui.value if _preset == "earth_falcon9" else "reentry"

    def _render(env, trail, thr, width=420, height=300):
        st = env.state
        pad_x = env._p["pad_x"]
        x, z = st["pos"][0], max(st["pos"][1], 0.0)

        half = max(abs(x - pad_x) / 2 + 250.0, z * 1.25, 600.0)
        fig = matplotlib.figure.Figure(figsize=(width / 80, height / 80), dpi=80)
        fig.patch.set_facecolor("#0b0e1a")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#0b0e1a")

        pad_hw = env.cfg["pad_half_width"]
        ax.plot([-half, half], [0, 0], color="#2a2f45", lw=2)
        ax.add_patch(mpatches.Rectangle(
            (pad_x - pad_hw, 0), 2 * pad_hw, max(0.025 * half, 8.0),
            facecolor="#e6c700", edgecolor="none", zorder=2))

        if trail:
            tx, tz = zip(*trail)
            ax.plot(tx, tz, color="#4da6ff", lw=0.9, alpha=0.55)

        L = max(0.07 * half, 25.0)
        W = L * 0.22
        c, s = math.cos(st["theta"]), math.sin(st["theta"])

        def _w(px, pz):
            return (x + px * c - pz * s, z + px * s + pz * c)

        body = [_w(-W, -L / 2), _w(W, -L / 2), _w(W, L / 2), _w(-W, L / 2)]
        ax.add_patch(mpatches.Polygon(
            body, closed=True, facecolor="#cc3333", edgecolor="#ff8080",
            lw=0.8, zorder=5))

        if thr > 0.05:
            pl = L * (0.5 + 0.9 * thr)
            flame = [_w(-W * 0.7, -L / 2), _w(W * 0.7, -L / 2), _w(0.0, -L / 2 - pl)]
            ax.add_patch(mpatches.Polygon(
                flame, closed=True, facecolor="#ffb347", edgecolor="none",
                alpha=0.9, zorder=4))

        if env.chute_deployed:
            cr = L * 1.6
            top = _w(0.0, L)
            ax.add_patch(mpatches.Arc(
                top, 2 * cr, 2 * cr, theta1=20, theta2=160,
                color="#dddddd", lw=2.5, zorder=3))

        hud = (f"alt {z:,.0f} m   speed "
               f"{float(np.hypot(st['vel'][0], st['vel'][1])):.1f} m/s   "
               f"fuel {env._fuel_frac:.0%}")
        ax.text(0.02, 0.02, hud, transform=ax.transAxes, color="#9fb3c8", fontsize=6)
        ax.set_xlim(-half, half)
        ax.set_ylim(-0.06 * half, 2 * half)
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)
        return img

    _hists, _stats = [], []
    for _ep in range(3):
        _env = FullMissionEnv(_preset, start_phase=_phase, seed=777_000 + _ep)
        _obs, _ = _env.reset()
        _done, _ep_r, _hist = False, 0.0, []
        while not _done:
            _act, _ = rocket_model.predict(_obs, deterministic=True)
            _obs, _r, _term, _trunc, _info = _env.step(_act)
            _ep_r += _r
            _done = _term or _trunc
            _st = _env.state
            _hist.append((float(_st["pos"][0]), float(_st["pos"][1]), _st["theta"],
                          float(np.asarray(_act).ravel()[1]), _env.chute_deployed))
        _env.close()
        _hists.append(_hist)
        _stats.append((_info["outcome"] or "timeout", _ep_r, _info["speed"],
                       _info["tilt"]))

    _frames = []
    for _hist in _hists:
        _k = max(1, len(_hist) // 80)
        _skel = FullMissionEnv(_preset, start_phase=_phase, seed=0)
        _skel.reset(seed=0)
        _trail = []
        for _i in range(0, len(_hist), _k):
            _px, _pz, _th, _thr, _chute = _hist[_i]
            _trail.append((_px, _pz))
            _skel.state = {
                "pos": np.array([_px, _pz]), "vel": np.zeros(2),
                "theta": _th, "omega": 0.0, "fuel": _skel._p["fuel0"],
            }
            _skel.chute_deployed = _chute
            _frames.append(_render(_skel, _trail, _thr))

    _buf = io.BytesIO()
    Image.fromarray(_frames[0]).save(
        _buf, format="GIF", save_all=True,
        append_images=[Image.fromarray(f) for f in _frames[1:]],
        duration=90, loop=0,
    )
    _gif = ('<img src="data:image/gif;base64,'
            + base64.b64encode(_buf.getvalue()).decode()
            + '" width="420">')

    _rows = "\n".join(
        f"| {_i + 1} | {_st[0]} | {_st[1]:.1f} | {_st[2]:.2f} | {_st[3]:.1f}\u00b0 |"
        for _i, _st in enumerate(_stats)
    )
    mo.vstack([
        mo.md(f"""
        ### \U0001f3ac Mission replays (randomized episodes the agent never trained on)

        {_gif}

        | # | Outcome | Reward | Touchdown speed | Final tilt |
        |---|---|---|---|---|
        {_rows}
        """),
        mo.download(
            data=_buf.getvalue(),
            filename="full_mission_replay.gif",
            mimetype="image/gif",
            label="\u2b07\ufe0f Download GIF",
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### 📌 Notes

        - **Training time:** ~5–8 min per 500k steps on molab's free sandbox
          (pure-NumPy physics, 4 parallel envs). Slider default (1M) is a solid
          first run; **1.5M–3M** for full missions (`ascent` / `boostback`).
        - expected learning arc: first it learns to track the reference descent
          and hover, then to land (first landings near 1M on textbook physics),
          while the domain-randomization strength ramps 0 → 1 over the first
          60% — wind, gravity, thrust, mass, engine-out and entry conditions
          are all reshuffled every episode.
        - **Mars EDL:** the chute deploy is a one-way door — the agent must
          learn to trigger it under 8 kPa dynamic pressure, then glide to a
          powered landing.
        - **Download:** this file (`rocket_mission_marimo.py`) **is** the whole
          project. molab download button (top-right) → then run locally:
          `uv run rocket_mission_marimo.py`
          (or `pip install marimo && marimo edit rocket_mission_marimo.py`).
        - The repo version with the same code split into modules:
          `full_mission/env.py`, `full_mission/train.py`, `full_mission/scripted_test.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
