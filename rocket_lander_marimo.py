# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy>=1.24",
#     "pybullet>=3.2.5",
#     "gymnasium>=0.29",
#     "stable-baselines3>=2.0",
#     "matplotlib>=3.7",
#     "pillow>=10.0",
# ]
# ///
"""3D Rocket Lander — RL with domain randomization, in one molab notebook.

Train a PPO agent to land a 3D rocket on a pad while the physics changes every
episode (gravity, mass, thrust, wind, engine latency, sensor noise) — the
sim-to-real technique used to train real robots.

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
    import pybullet as p
    import pybullet_utils.bullet_client as bc
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
        bc,
        math,
        mo,
        np,
        p,
        spaces,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # 🚀 3D Rocket Lander — RL + Domain Randomization

        An AI learns to land a 3D rocket using **PPO** (Stable-Baselines3) in a
        **PyBullet** physics simulator. You never tell it how to fly — you only
        score every step (reward shaping):

        | Event | Points |
        |---|---|
        | Get closer to the pad | ➕ |
        | Move fast / tilt too much / burn fuel / take long | ➖ |
        | Safe landing (touch pad, speed < 1.5, tilt < 15°) | ➕ **+100** |
        | Crash (touch ground or tip over) | ➖ **−100** |

        **Domain randomization** — every episode reshuffles gravity, rocket mass,
        thrust strength, wind, engine latency, sensor noise, start position and
        pad location. The policy therefore learns a *general* landing skill
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
def _(Env, Image, PPO, bc, math, mo, np, p, spaces):
    BODY_HEIGHT = 2.0
    BODY_RADIUS = 0.35
    PAD_HALF_EXTENT = 1.5
    MAX_THRUST = 60.0
    MAX_SIDE_THRUST = 12.0

    class RocketLander3DEnv(Env):
        """3D rocket landing environment on PyBullet with domain randomization."""

        def __init__(
            self,
            domain_randomization=True,
            max_episode_steps=900,
            seed=None,
            client=None,
        ):
            self.rng = np.random.default_rng(seed)
            self.renderer = client if client is not None else bc.BulletClient(
                connection_mode=p.DIRECT
            )
            self.domain_randomization = domain_randomization
            self.max_episode_steps = max_episode_steps
            self.rocket_id = None
            self.step_count = 0
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
            obs_high = np.full(12, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(
                -obs_high, obs_high, dtype=np.float32
            )
            self._physics = {}
            self._episode_thrust = 1.0
            self._wind = np.zeros(3)
            self._engine_latency = 1
            self._action_queue = []

        def _randomize_physics(self):
            if self.domain_randomization:
                gravity = self.rng.uniform(4.9, 24.5)
                mass = self.rng.uniform(0.8, 1.6)
                thrust_scale = self.rng.uniform(0.8, 1.2)
                wind = self.rng.uniform(-15, 15, size=3)
                wind[2] = 0.0
                engine_latency = int(self.rng.integers(3, 10))
                sensor_noise = self.rng.uniform(0.0, 0.20)
                start_pos = self.rng.uniform([-8, -8, 10], [8, 8, 14])
                start_vel = self.rng.uniform([-2, -2, 0], [2, 2, 1])
                pad_x = float(self.rng.uniform(-2, 2))
                pad_y = float(self.rng.uniform(-2, 2))
            else:
                gravity, mass, thrust_scale = 9.8, 1.0, 1.0
                wind = np.zeros(3)
                engine_latency, sensor_noise = 1, 0.0
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
                "start_pos": start_pos,
                "start_vel": start_vel,
                "pad_x": pad_x,
                "pad_y": pad_y,
            }

        def _build_scene(self):
            cid = self.renderer
            cid.resetSimulation()
            cid.setGravity(0, 0, -self._physics["gravity"])
            cid.setTimeStep(1.0 / 120.0)

            plane_shape = cid.createCollisionShape(cid.GEOM_PLANE)
            self.ground_id = cid.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=plane_shape
            )

            pad_shape = cid.createCollisionShape(
                cid.GEOM_BOX, halfExtents=[PAD_HALF_EXTENT, PAD_HALF_EXTENT, 0.05]
            )
            pad_visual = cid.createVisualShape(
                cid.GEOM_BOX, halfExtents=[PAD_HALF_EXTENT, PAD_HALF_EXTENT, 0.05],
                rgbaColor=[0.9, 0.9, 0.1, 1],
            )
            self.pad_id = cid.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=pad_shape,
                baseVisualShapeIndex=pad_visual,
                basePosition=[self._physics["pad_x"], self._physics["pad_y"], 0.025],
            )

            body_shape = cid.createCollisionShape(
                cid.GEOM_CYLINDER, radius=BODY_RADIUS, height=BODY_HEIGHT
            )
            body_visual = cid.createVisualShape(
                cid.GEOM_CYLINDER, radius=BODY_RADIUS, length=BODY_HEIGHT,
                rgbaColor=[0.8, 0.2, 0.2, 1],
            )
            leg_shape = cid.createCollisionShape(
                cid.GEOM_BOX, halfExtents=[0.08, 0.08, 0.25]
            )
            leg_visual = cid.createVisualShape(
                cid.GEOM_BOX, halfExtents=[0.08, 0.08, 0.25],
                rgbaColor=[0.3, 0.3, 0.3, 1],
            )

            x, y, z = self._physics["start_pos"]
            origin = [0, 0, 0, 1]
            self.rocket_id = cid.createMultiBody(
                baseMass=self._physics["mass"],
                baseCollisionShapeIndex=body_shape,
                baseVisualShapeIndex=body_visual,
                basePosition=[x, y, z],
                baseOrientation=origin,
            )
            cid.changeDynamics(self.rocket_id, -1, lateralFriction=0.5)

            self.leg_ids = []
            for dx, dy in [(0.45, 0.45), (-0.45, 0.45), (0.45, -0.45), (-0.45, -0.45)]:
                leg_id = cid.createMultiBody(
                    baseMass=0.05,
                    baseCollisionShapeIndex=leg_shape,
                    baseVisualShapeIndex=leg_visual,
                    basePosition=[x + dx, y + dy, z - (BODY_HEIGHT / 2 + 0.25)],
                    baseOrientation=origin,
                )
                cid.createConstraint(
                    self.rocket_id, -1, leg_id, -1, cid.JOINT_FIXED,
                    [0, 0, 0], [dx, dy, -(BODY_HEIGHT / 2 + 0.25)], [0, 0, 0],
                )
                self.leg_ids.append(leg_id)

            self._episode_thrust = self._physics["thrust_scale"]
            self._wind = np.asarray(self._physics["wind"], dtype=np.float64)
            self._engine_latency = self._physics["engine_latency"]
            self._action_queue = []

        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self._randomize_physics()
            self._build_scene()
            pos = self._physics["start_pos"]
            vel = self._physics["start_vel"]
            self.renderer.resetBasePositionAndOrientation(
                self.rocket_id, list(pos), [0, 0, 0, 1]
            )
            self.renderer.resetBaseVelocity(
                self.rocket_id, linearVelocity=list(vel), angularVelocity=[0, 0, 0]
            )
            self.step_count = 0
            return self._get_obs(), {}

        def _get_obs(self):
            cid = self.renderer
            pos, orn = cid.getBasePositionAndOrientation(self.rocket_id)
            vel, ang = cid.getBaseVelocity(self.rocket_id)
            euler = cid.getEulerFromQuaternion(orn)
            obs = np.array(
                list(pos) + list(vel) + [euler[0], euler[1], euler[2]] + list(ang),
                dtype=np.float32,
            )
            noise = self._physics.get("sensor_noise", 0.0)
            if noise > 0:
                obs += self.rng.normal(0, noise, size=obs.shape).astype(np.float32)
            return obs

        def _apply_thrust(self, action):
            cid = self.renderer
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            main, left, right = a
            pos, orn = cid.getBasePositionAndOrientation(self.rocket_id)

            thrust_force = float(main) * MAX_THRUST * self._episode_thrust
            up_world = cid.rotateVector(orn, [0, 0, 1])
            cid.applyExternalForce(
                self.rocket_id, -1,
                [up_world[0] * thrust_force, up_world[1] * thrust_force,
                 up_world[2] * thrust_force],
                pos, cid.WORLD_FRAME,
            )

            side_force = float(left - right) * MAX_SIDE_THRUST * self._episode_thrust
            side_dir = cid.rotateVector(orn, [1, 0, 0])
            cid.applyExternalForce(
                self.rocket_id, -1,
                [side_dir[0] * side_force, side_dir[1] * side_force,
                 side_dir[2] * side_force],
                pos, cid.WORLD_FRAME,
            )

            if np.linalg.norm(self._wind) > 0:
                cid.applyExternalForce(
                    self.rocket_id, -1, self._wind * 0.5, pos, cid.WORLD_FRAME
                )

        def _check_landing(self):
            cid = self.renderer
            pos, orn = cid.getBasePositionAndOrientation(self.rocket_id)
            vel, _ = cid.getBaseVelocity(self.rocket_id)
            euler = cid.getEulerFromQuaternion(orn)
            speed = float(np.linalg.norm(vel))
            tilt = math.degrees(max(abs(euler[0]), abs(euler[1])))

            touching_pad = False
            touching_ground = False
            for body_id in [self.rocket_id] + self.leg_ids:
                for cp in cid.getContactPoints(bodyA=body_id, bodyB=self.pad_id):
                    touching_pad = True
                for cp in cid.getContactPoints(
                    bodyA=body_id, bodyB=self.ground_id
                ):
                    touching_ground = True

            if touching_pad and speed < 1.5 and tilt < 15:
                return "landed", speed, tilt, pos
            if touching_ground or tilt > 90 or pos[2] < 0.2:
                return "crashed", speed, tilt, pos
            return None, speed, tilt, pos

        def step(self, action):
            self.step_count += 1
            self._action_queue.append(np.asarray(action, dtype=np.float64))
            lag = max(1, self._engine_latency)
            self._action_queue = self._action_queue[-lag:]
            effective_action = self._action_queue[0]

            for _ in range(2):
                self._apply_thrust(effective_action)
                self.renderer.stepSimulation()

            outcome, speed, tilt, pos = self._check_landing()
            pad_x, pad_y = self._physics["pad_x"], self._physics["pad_y"]
            lateral_dist = float(np.hypot(pos[0] - pad_x, pos[1] - pad_y))

            terminated = False
            truncated = self.step_count >= self.max_episode_steps
            reward = 0.0
            if pos[2] > 2:
                reward += (4.0 - lateral_dist) * 0.5
            reward -= 0.05 * speed
            reward -= 0.05 * abs(tilt) / 10
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            reward -= 0.1 * (a[0] + abs(a[1] - a[2]))
            reward -= 0.02

            if outcome == "landed":
                reward += 100.0
                terminated = True
            elif outcome == "crashed":
                reward -= 100.0
                terminated = True

            info = {
                "speed": speed,
                "tilt": tilt,
                "lateral_dist": lateral_dist,
                "outcome": outcome,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

        def close(self):
            self.renderer.disconnect()

    env_class_note = mo.md(
        "Environment ready: a 3D rocket with 3 thrusters, dynamic legs, wind, "
        "latency and sensor noise."
    )
    env_class_note
    return (RocketLander3DEnv,)


@app.cell
def _(PPO, RocketLander3DEnv, DummyVecEnv, Monitor, mo,
      np, train_button, training_steps, domain_randomization):
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
    with mo.status.progress_bar(
        total=training_steps.value, title="Training PPO on randomized physics"
    ) as bar:
        for chunk in range(0, training_steps.value, 20_480):
            rocket_model.learn(
                total_timesteps=min(20_480, training_steps.value - chunk),
                reset_num_timesteps=False,
            )
            bar.update(increment=min(20_480, training_steps.value - chunk))
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
    mo.stop(
        rocket_model is None,
        mo.md("🔒 Train first — then this cell plays demo landings on camera."),
    )

    def capture_screen(lander_cid, rocket_id, width=320, height=320):
        pos, _ = lander_cid.getBasePositionAndOrientation(rocket_id)
        view = lander_cid.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[pos[0], pos[1], max(pos[2] * 0.7, 1.0)],
            distance=8, yaw=45, pitch=-25, roll=0, upAxisIndex=2,
        )
        proj = lander_cid.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=50
        )
        img = lander_cid.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj)[2]
        return np.reshape(img, (height, width, 4))[:, :, :3]

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
                    capture_screen(eval_env.renderer, eval_env.rocket_id)
                )
        demo_stats.append((info["outcome"], ep_reward, info["speed"], info["tilt"]))
    eval_env.close()

    pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in demo_frames]
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

        - **Training time:** ~15–25 min per 500k steps on molab's free 4-CPU sandbox.
          For the full experience train ≥ 1M steps in one session (molab allows 12 h).
        - If landings are still rare at 600k steps, increase the slider and press the
          training button again — it resumes (`reset_num_timesteps=False`).
        - **Download:** this notebook file (`rocket_lander_marimo.py`) is the whole
          project — press the download button at the top-right of molab to keep it,
          then run locally:
          `uv run rocket_lander_marimo.py` or `pip install marimo && marimo edit rocket_lander_marimo.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
