"""A 3D rocket landing environment built on PyBullet with domain randomization.

The agent controls three thrusters (main engine + two side thrusters) and must
land the rocket upright on a pad. Every episode randomizes gravity, mass,
thrust strength, wind, engine latency, and sensor noise so the policy learns a
general landing skill (sim-to-real style domain randomization).
"""

import math

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from gymnasium import spaces

BODY_HEIGHT = 2.0
BODY_RADIUS = 0.35
PAD_HALF_EXTENT = 1.5

MAX_THRUST = 60.0
MAX_SIDE_THRUST = 12.0


class RocketLander3DEnv(gym.Env):
    """Gymnasium-compatible 3D rocket landing environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        seed=None,
        domain_randomization=True,
        max_episode_steps=900,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)
        self.renderer = p.DIRECT if render_mode != "human" else p.GUI
        self.client_id = None
        self.rocket_id = None
        self.pad_id = None
        self.step_count = 0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        obs_high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._physics = {}
        self._episode_thrust = 1.0
        self._wind = np.zeros(3)
        self._engine_latency = 1
        self._action_queue = []

    def _open_client(self):
        if self.client_id is None:
            self.client_id = bc.BulletClient(connection_mode=self.renderer)
            if self.renderer == p.GUI:
                self.client_id.resetDebugVisualizerCamera(
                    cameraDistance=8, cameraYaw=40, cameraPitch=-20,
                    cameraTargetPosition=[0, 0, 1],
                )

    def _build_scene(self):
        cid = self.client_id
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
        else:
            gravity, mass, thrust_scale = 9.8, 1.0, 1.0
            wind = np.zeros(3)
            engine_latency, sensor_noise = 1, 0.0
            start_pos = np.array([0.0, 0.0, 12.0])
            start_vel = np.zeros(3)

        self._physics = {
            "gravity": gravity,
            "mass": mass,
            "thrust_scale": thrust_scale,
            "wind": wind,
            "engine_latency": engine_latency,
            "sensor_noise": sensor_noise,
            "start_pos": start_pos,
            "start_vel": start_vel,
            "pad_x": float(self.rng.uniform(-2, 2)) if self.domain_randomization else 0.0,
            "pad_y": float(self.rng.uniform(-2, 2)) if self.domain_randomization else 0.0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_physics()
        self._open_client()
        self._build_scene()
        pos = self._physics["start_pos"]
        vel = self._physics["start_vel"]
        self.client_id.resetBasePositionAndOrientation(
            self.rocket_id, list(pos), [0, 0, 0, 1]
        )
        self.client_id.resetBaseVelocity(
            self.rocket_id, linearVelocity=list(vel), angularVelocity=[0, 0, 0]
        )
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        cid = self.client_id
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
        cid = self.client_id
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
        cid = self.client_id
        pos, orn = cid.getBasePositionAndOrientation(self.rocket_id)
        vel, _ = cid.getBaseVelocity(self.rocket_id)
        euler = cid.getEulerFromQuaternion(orn)
        speed = float(np.linalg.norm(vel))
        tilt = math.degrees(max(abs(euler[0]), abs(euler[1])))

        touching_pad = False
        touching_ground = False
        for body_id in [self.rocket_id] + getattr(self, "leg_ids", []):
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
            self.client_id.stepSimulation()

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

    def render(self):
        return None

    def close(self):
        if self.client_id is not None:
            self.client_id.disconnect()
            self.client_id = None
