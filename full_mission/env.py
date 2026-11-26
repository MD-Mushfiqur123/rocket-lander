"""Full reusable-rocket mission environment (pure NumPy, planar 3-DOF).

One continuous mission: engines-first landing, reentry with atmosphere drag,
boostback, and a gravity-turn ascent with apogee flip back to the pad. The
rocket steers with a real Thrust Vector Control gimbal (the engine itself
gimbals and produces torque), burns fuel with a finite tank, and scores bonus
points for landing with fuel to spare (fuel-optimal suicide burn).

Two presets:
    earth_falcon9 : full launch -> boostback -> reentry -> landing mission
    mars_edl      : Mars Entry-Descent-Landing with parachute deploy timing
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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


class FullMissionEnv(gym.Env):
    """Gymnasium-compatible full-mission environment with TVC and fuel.

    Actions (Box 0..1):
        earth_falcon9: [gimbal command, throttle, unused]
        mars_edl:      [gimbal command, throttle, chute deploy (>0.5 deploys,
                        irreversible; deploying at too-high dynamic pressure
                        rips the chute and destroys the vehicle)]

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

        n_act = 4 if self.cfg["has_chute"] else 3
        self.action_space = spaces.Box(0.0, 1.0, shape=(n_act,), dtype=np.float32)

        n_obs = 7 + (1 if self.cfg["has_chute"] else 0) + 2
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
            + [ascent, 1.0 - ascent],
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

        v_mag = float(np.linalg.norm(st["vel"]))
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
            "engine_failed": False,
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
            "outcome": outcome, "stage": self.stage, "engine_failed": False,
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
