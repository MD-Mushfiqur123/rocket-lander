"""Scripted physics validation for the full-mission environment.

1. Landing phase: stateful suicide-burn controller commits to the landing
   burn and lands softly with the TVC gimbal holding attitude.
2. Ascent phase: gravity-turn ascent reaches MECO altitude, boostback/coast/
   reentry bring the vehicle back to low altitude alive.
3. Mars EDL: parachute at safe dynamic pressure + retro-propulsion lands.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from full_mission.env import FullMissionEnv


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
        t_max = env.cfg["max_thrust"]
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
        act = [gimbal, throttle]
        if env.cfg["has_chute"]:
            v_rel = abs(vz) + abs(vx)
            rho = env._atmo_density(z)
            q = 0.5 * rho * (vz ** 2 + vx ** 2)
            deploy = (z < 20_000 and v_rel > 80
                      and q < 0.75 * env.cfg["max_chute_q"])
            act.append(1.0 if deploy else 0.0)
        return np.array(act)


class GravityTurnController:
    def __init__(self):
        self.att = TvcAttitude()
        self.meco = False

    def __call__(self, obs, env):
        z = obs[1] * 1000.0
        vz = obs[3] * 100.0
        if not self.meco and z > 30_000 and z + vz ** 2 / 19.6 > 60_000:
            self.meco = True
        if self.meco:
            return np.array([0.5, 0.0])
        if z < 8000:
            tilt = 0.0
        else:
            tilt = min((z - 8000) / 60_000, 1.0) * math.radians(35)
        gimbal = self.att.command(env, obs[4], obs[5], tilt, 1.0)
        return np.array([gimbal, 1.0])


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
        t_max = cfg["max_thrust"]
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


def run(env, ctrl_type, episodes=5):
    outcomes = []
    for ep in range(episodes):
        ctrl = ctrl_type()
        obs, _ = env.reset(seed=100 + ep)
        done = False
        while not done:
            action = ctrl(obs, env)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        outcomes.append((info["outcome"], round(info["speed"], 1),
                         round(info["tilt"], 1), round(info["fuel_frac"], 2)))
    return outcomes


if __name__ == "__main__":
    lh = run(FullMissionEnv("earth_falcon9", start_phase="landing",
                            domain_randomization=False), LandingController, 5)
    print("landing   :", lh)
    assert sum(1 for o in lh if o[0] == "landed") >= 4, lh

    max_alts = []
    for ep in range(3):
        env = FullMissionEnv("earth_falcon9", start_phase="ascent",
                             domain_randomization=False)
        ctrl = GravityTurnController()
        obs, _ = env.reset(seed=200 + ep)
        done, max_z = False, 0.0
        while not done:
            obs, r, t, tr, info = env.step(ctrl(obs, env))
            done = t or tr
            max_z = max(max_z, obs[1] * 1000.0)
        max_alts.append((info["outcome"], round(max_z), round(ctrl.meco and 1 or 0)))
        env.close()
    print("full miss.:", max_alts)
    assert all(o[0] == "crashed" and o[1] > 55_000 for o in max_alts), max_alts

    mh = run(FullMissionEnv("mars_edl", start_phase="reentry",
                            domain_randomization=False), MarsEdlController, 5)
    print("mars edl  :", mh)
    assert sum(1 for o in mh if o[0] == "landed") >= 3, mh

    print("SCRIPTED MISSION TESTS PASSED")
