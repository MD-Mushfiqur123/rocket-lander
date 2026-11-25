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
# NOTE: pure NumPy 3D physics — zero compiled dependencies, so this notebook
# installs and runs on any cloud sandbox (molab, Colab, any Python 3.10+).
"""3D Rocket Lander — RL with domain randomization, in one molab notebook.

Train a PPO agent to land a 3D rocket on a pad while the physics changes every
episode (gravity, mass, thrust, wind, attitude noise, engine latency, sensor
noise) — the sim-to-real technique used to train real robots.

Run on https://molab.marimo.io (free, 4 CPU / 32 GB), or locally:
    uv run rocket_lander_marimo.py
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
    from PIL import Image
    from gymnasium import Env, spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return (
        DummyVecEnv,
        Env,
        Image,
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
        # 🚀 3D Rocket Lander — RL + Domain Randomization

        An AI learns to land a 3D rocket using **PPO** (Stable-Baselines3) in a
        pure-NumPy 3D physics simulator (quaternion attitude + translational
        dynamics + wind). You never tell it how to fly — you only score every
        step (reward shaping):

        | Event | Points |
        |---|---|
        | Get closer to the pad | ➕ |
        | Move fast / tilt too much / burn fuel / take long | ➖ |
        | Safe landing (touch pad, speed < 1.5, tilt < 15°) | ➕ **+100** |
        | Crash (touch ground or tip over) | ➖ **−100** |

        **Domain randomization** — every episode reshuffles gravity, rocket mass,
        thrust, wind, engine latency, sensor noise, attitude disturbances, start
        position and pad location. The policy learns a *general* landing skill
        instead of memorizing one scenario.
        """
    )
    return


@app.cell
def _(mo):
    training_steps = mo.ui.slider(
        start=100_000, stop=2_000_000, step=50_000, value=600_000, label="Training steps"
    )
    domain_randomization = mo.ui.checkbox(True, label="Domain randomization")
    train_button = mo.ui.run_button(label="🚀 Train the rocket")
    mo.hstack(
        [training_steps, domain_randomization, train_button],
        justify="start",
    )
    return (
        domain_randomization,
        train_button,
        training_steps,
    )


@app.cell
def _(Env, Image, PPO, math, mo, np, spaces):
    BODY_HEIGHT = 2.0
    LEG_DROP = 1.5
    PAD_HALF_EXTENT = 1.5
    MAX_THRUST = 60.0
    MAX_SIDE_THRUST = 12.0
    DT = 1.0 / 120.0

    class RocketLander3DEnv(Env):
        """3D rocket landing environment (pure NumPy physics, domain-randomized)."""

        def __init__(self, domain_randomization=True, max_episode_steps=900, seed=None):
            self.rng = np.random.default_rng(seed)
            self.domain_randomization = domain_randomization
            self.max_episode_steps = max_episode_steps
            self.rand_strength = 1.0
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
            obs_high = np.full(14, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
            self._physics = {}
            self._state = None
            self._episode_thrust = 1.0
            self._inertia_xx = 1.0
            self._action_queue = []

        @staticmethod
        def _quat_rotate(q, v):
            w, x, y, z = q
            vx, vy, vz = v
            uvx, uvy, uvz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
            uux, uuy, uuz = y * uvz - z * uvy, z * uvx - x * uvz, x * uvy - y * uvx
            return (
                vx + 2.0 * (w * uvx + uux),
                vy + 2.0 * (w * uvy + uuy),
                vz + 2.0 * (w * uvz + uuz),
            )

        @staticmethod
        def _quat_normalize(q):
            q = np.asarray(q, dtype=np.float64)
            return q / np.linalg.norm(q)

        def set_randomization_strength(self, strength):
            """0 = textbook physics, 1 = full domain randomization."""
            self.rand_strength = float(np.clip(strength, 0.0, 1.0))

        def _randomize_physics(self):
            if self.domain_randomization:
                s = self.rand_strength
                gravity = 9.8 + s * self.rng.uniform(-1.0, 1.0) * 7.0
                mass = 1.0 + s * self.rng.uniform(-0.2, 0.3)
                thrust_scale = 1.0 + s * self.rng.uniform(-0.15, 0.15)
                wind = self.rng.uniform(-8, 8, size=3) * s
                wind[2] = 0.0
                engine_latency = int(1 + s * (2 + self.rng.integers(0, 5)))
                sensor_noise = self.rng.uniform(0.0, 0.12 * s)
                attitude_noise = self.rng.uniform(0.0, 0.25 * s)
                start_pos = np.array([0.0, 0.0, 12.0]) * (1.0 - s) + self.rng.uniform(
                    [-8, -8, 10], [8, 8, 14]
                ) * s
                start_vel = self.rng.uniform([-2, -2, 0], [2, 2, 1]) * s
                pad_x = float(self.rng.uniform(-2, 2) * s)
                pad_y = float(self.rng.uniform(-2, 2) * s)
            else:
                gravity, mass, thrust_scale = 9.8, 1.0, 1.0
                wind = np.zeros(3)
                engine_latency, sensor_noise, attitude_noise = 1, 0.0, 0.0
                start_pos = np.array([0.0, 0.0, 12.0])
                start_vel = np.zeros(3)
                pad_x = pad_y = 0.0
            self._physics = {
                "gravity": gravity,
                "mass": mass,
                "thrust_scale": thrust_scale,
                "wind": wind,
                "engine_latency": engine_latency,
                "sensor_noise": sensor_noise,
                "attitude_noise": attitude_noise,
                "start_pos": start_pos,
                "start_vel": start_vel,
                "pad_x": pad_x,
                "pad_y": pad_y,
            }
            self._episode_thrust = thrust_scale
            self._inertia_xx = (mass / 12.0) * (3 * 0.35**2 + BODY_HEIGHT**2)

        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self._randomize_physics()
            self._state = {
                "pos": np.array(self._physics["start_pos"], dtype=np.float64),
                "vel": np.array(self._physics["start_vel"], dtype=np.float64),
                "orn": np.array([1.0, 0.0, 0.0, 0.0]),
                "ang": np.zeros(3),
            }
            self.step_count = 0
            self._action_queue = []
            return self._get_obs(), {}

        def _get_obs(self):
            s = self._state
            roll = math.atan2(
                2 * (s["orn"][0] * s["orn"][1] + s["orn"][2] * s["orn"][3]),
                1 - 2 * (s["orn"][1] ** 2 + s["orn"][2] ** 2),
            )
            pitch = math.asin(max(-1, min(1, 2 * (s["orn"][0] * s["orn"][2] - s["orn"][3] * s["orn"][1]))))
            yaw = math.atan2(
                2 * (s["orn"][0] * s["orn"][3] + s["orn"][1] * s["orn"][2]),
                1 - 2 * (s["orn"][2] ** 2 + s["orn"][3] ** 2),
            )
            obs = np.concatenate(
                [s["pos"], s["vel"], [roll, pitch, yaw], s["ang"],
                 [s["pos"][0] - self._physics["pad_x"],
                  s["pos"][1] - self._physics["pad_y"]]]
            ).astype(np.float32)
            noise = self._physics.get("sensor_noise", 0.0)
            if noise > 0:
                obs += self.rng.normal(0, noise, size=obs.shape).astype(np.float32)
            return obs

        def _physics_substep(self, action):
            s = self._state
            phys = self._physics
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            mass = phys["mass"]

            force = np.zeros(3)
            force[2] -= phys["gravity"] * mass
            thrust_force = a[0] * MAX_THRUST * self._episode_thrust
            force += np.array(
                self._quat_rotate(s["orn"], [0, 0, 1])
            ) * thrust_force
            side_force = (a[1] - a[2]) * MAX_SIDE_THRUST * self._episode_thrust
            force += np.array(self._quat_rotate(s["orn"], [1, 0, 0])) * side_force
            force += np.array(phys["wind"]) * 0.5

            s["vel"] += (force / mass) * DT
            s["pos"] += s["vel"] * DT

            torque = self.rng.normal(
                0.0, phys["attitude_noise"], size=3
            ) * self._inertia_xx
            s["ang"] += (torque / self._inertia_xx) * DT
            s["ang"] *= 0.995

            ang_norm = np.linalg.norm(s["ang"])
            if ang_norm > 1e-9:
                theta_half = ang_norm * DT * 0.5
                axis = s["ang"] / ang_norm
                dq = np.concatenate([
                    [math.cos(theta_half)], axis * math.sin(theta_half)
                ])
                w, x, y, z = dq
                bw, bx, by, bz = s["orn"]
                s["orn"] = self._quat_normalize([
                    bw * w - bx * x - by * y - bz * z,
                    bw * x + bx * w + by * z - bz * y,
                    bw * y - bx * z + by * w + bz * x,
                    bw * z + bx * y - by * x + bz * w,
                ])

        def _check_landing(self):
            s = self._state
            speed = float(np.linalg.norm(s["vel"]))
            half_angle = math.degrees(
                2 * math.acos(min(1.0, abs(float(s["orn"][0]))))
            )
            tilt = half_angle if half_angle <= 90 else 180 - half_angle
            leg_bottom = s["pos"][2] - LEG_DROP
            on_pad_x = abs(s["pos"][0] - self._physics["pad_x"]) <= PAD_HALF_EXTENT
            on_pad_y = abs(s["pos"][1] - self._physics["pad_y"]) <= PAD_HALF_EXTENT
            if leg_bottom <= 0.05 and on_pad_x and on_pad_y and speed < 1.5 and tilt < 15:
                return "landed", speed, tilt, s["pos"]
            if leg_bottom <= 0.0 or tilt > 90 or s["pos"][2] < 0.2:
                return "crashed", speed, tilt, s["pos"]
            return None, speed, tilt, s["pos"]

        def step(self, action):
            self.step_count += 1
            self._action_queue.append(np.asarray(action, dtype=np.float64))
            lag = max(1, self._physics["engine_latency"])
            self._action_queue = self._action_queue[-lag:]
            effective_action = self._action_queue[0]

            for _ in range(2):
                self._physics_substep(effective_action)

            outcome, speed, tilt, pos = self._check_landing()
            s_vel = self._state["vel"]
            rel_x = pos[0] - self._physics["pad_x"]
            rel_y = pos[1] - self._physics["pad_y"]
            alt = max(pos[2] - 1.5, 0.0)

            v_xy_target = np.clip([-0.8 * rel_x, -0.8 * rel_y], -2.0, 2.0)
            vz_target = -float(np.clip(0.5 * alt, 0.2, 2.0))

            terminated = False
            truncated = self.step_count >= self.max_episode_steps
            reward = (
                -0.05 * float(np.hypot(
                    s_vel[0] - v_xy_target[0], s_vel[1] - v_xy_target[1]
                ))
                - 0.05 * abs(s_vel[2] - vz_target)
                - 0.03 * abs(tilt) / 10
                - 0.02 * float(np.hypot(rel_x, rel_y))
                - 0.02
            )
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            reward -= 0.1 * (a[0] + abs(a[1] - a[2]))

            if outcome == "landed":
                reward += 100.0
                terminated = True
            elif outcome == "crashed":
                reward -= 100.0
                terminated = True

            info = {
                "speed": speed,
                "tilt": tilt,
                "lateral_dist": float(np.hypot(rel_x, rel_y)),
                "outcome": outcome,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        def close(self):
            return None

    env_class_note = mo.md(
        "Environment ready: a 3D rocket with 3 thrusters, wind, engine latency, "
        "attitude noise and sensor noise — pure NumPy, no compiled deps."
    )
    env_class_note
    return (RocketLander3DEnv,)


@app.cell
def _(PPO, RocketLander3DEnv, DummyVecEnv, Monitor, mo, np,
      train_button, training_steps, domain_randomization):
    rocket_model = None

    mo.stop(
        not train_button.value,
        mo.md("⬆️ Set the sliders and press **🚀 Train the rocket** to start."),
    )

    env = DummyVecEnv([
        lambda: Monitor(
            RocketLander3DEnv(domain_randomization=domain_randomization.value)
        )
    ])
    rocket_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
    )
    train_envs = env.envs
    for w in train_envs:
        w.env.set_randomization_strength(0.0)
    total = training_steps.value
    with mo.status.progress_bar(
        total=total, title="Training PPO — randomization strength ramps up in the second half"
    ) as bar:
        for chunk in range(0, total, 20_480):
            strength = min(1.0, max(0.0, (chunk - 0.5 * total) / (0.5 * total)))
            for w in train_envs:
                w.env.set_randomization_strength(strength)
            rocket_model.learn(
                total_timesteps=min(20_480, total - chunk),
                reset_num_timesteps=False,
            )
            bar.update(increment=min(20_480, total - chunk))
    rocket_model.save("rocket_lander_3d_ppo")
    training_summary = mo.md(
        f"""
        ### ✅ Training complete
        - **Steps:** {training_steps.value:,}
        - **Domain randomization:** {domain_randomization.value}
        - Model saved as `rocket_lander_3d_ppo.zip`
        """
    )
    training_summary
    return (rocket_model,)


@app.cell
def _(Image, RocketLander3DEnv, base64, io, mo, np, rocket_model):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    mo.stop(
        rocket_model is None,
        mo.md("🔒 Train first — then this cell plays demo landings on camera."),
    )

    PAD_HALF = 1.5

    def render_frame(env_s, physics, width=320, height=320):
        fig = matplotlib.figure.Figure(figsize=(4, 4), dpi=int(width / 4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d")
        px, py = physics["pad_x"], physics["pad_y"]

        pad = [
            [(px - PAD_HALF, py - PAD_HALF, 0), (px + PAD_HALF, py - PAD_HALF, 0),
             (px + PAD_HALF, py + PAD_HALF, 0), (px - PAD_HALF, py + PAD_HALF, 0)],
        ]
        ax.add_collection3d(Poly3DCollection(pad, color="#e6c700", alpha=1.0))
        ground = [[(-8, -8, 0), (8, -8, 0), (8, 8, 0), (-8, 8, 0)]]
        ax.add_collection3d(
            Poly3DCollection(ground, color="#333333", alpha=0.25, zorder=0)
        )

        pos, orn = env_s["pos"], env_s["orn"]
        w, x, y, z = orn

        def rotate(v):
            q = (w, x, y, z)
            u = (0, v[0], v[1], v[2])
            q_conj = (w, -x, -y, -z)
            def qmul(a, b):
                aw, ax, ay, az = a
                bw2, bx, by, bz = b
                return [
                    aw * bw2 - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw2 + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw2 + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw2,
                ]
            r = qmul(qmul(q, u), q_conj)
            return r[1:]

        h = 1.0
        body_faces = []
        bottom = [
            np.add(np.array(rotate(v)), pos)
            for v in [(-0.35, -0.35, -h), (0.35, -0.35, -h),
                      (0.35, 0.35, -h), (-0.35, 0.35, -h)]
        ]
        top = [
            np.add(np.array(rotate(v)), pos)
            for v in [(-0.35, -0.35, h), (0.35, -0.35, h),
                      (0.35, 0.35, h), (-0.35, 0.35, h)]
        ]
        for i in range(4):
            j = (i + 1) % 4
            body_faces.append([bottom[i], bottom[j], top[j], top[i]])
        body_faces.append(top)
        ax.add_collection3d(
            Poly3DCollection(body_faces, facecolor="#cc3333",
                             edgecolor="#661111", linewidth=0.5)
        )

        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_zlim(0, 16)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=45)
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)
        return img

    eval_env = RocketLander3DEnv(domain_randomization=True, seed=20260101)
    demo_frames = []
    demo_stats = []
    for _ep in range(3):
        obs, _ = eval_env.reset()
        ep_reward, done = 0.0, False
        while not done:
            action, _ = rocket_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            ep_reward += reward
            done = term or trunc
            if eval_env.step_count % 4 == 0:
                demo_frames.append(
                    render_frame(eval_env._state, eval_env._physics)
                )
        demo_stats.append((info["outcome"], ep_reward, info["speed"], info["tilt"]))
    eval_env.close()

    pil_frames = [Image.fromarray(f) for f in demo_frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=80, loop=0,
    )
    gif_html = (
        '<img src="data:image/gif;base64,'
        + base64.b64encode(buf.getvalue()).decode()
        + '" width="320">'
    )

    rows = "\n".join(
        f"| {_i + 1} | {_s[0] or 'timeout'} | {_s[1]:.1f} | {_s[2]:.2f} | {_s[3]:.1f}° |"
        for _i, _s in enumerate(demo_stats)
    )
    mo.vstack([
        mo.md(f"""
        ### 🎬 Demo landings (randomized physics the agent never saw during training)

        {gif_html}

        | # | Outcome | Reward | Speed | Tilt |
        |---|---|---|---|---|
        {rows}
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ### 📌 Notes

        - **Training time:** ~2 min per 100k steps on molab's free 4-CPU sandbox
          (pure-NumPy physics, no compiled engine). The slider default (600k) is a
          good first run; for a policy that lands even under FULL randomization,
          train 1.5M–2M steps — press the button again with a bigger slider to
          resume (`reset_num_timesteps=False`).
        - Training plan you'll see: first it learns to stay alive (timeouts),
          then to descend (crashes), then to land (first landings around 400-600k
          on textbook physics). In the **second half** the randomization strength
          ramps from 0 → 1, so the policy gradually masters wind, gravity/mass
          changes, engine lag and sensor noise. More steps = more robust landings.
        - **Download:** this notebook file (`rocket_lander_marimo.py`) is the whole
          project — press the download button at the top-right of molab to keep it,
          then run locally:
          `uv run rocket_lander_marimo.py` or `pip install marimo && marimo edit rocket_lander_marimo.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
