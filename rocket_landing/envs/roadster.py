"""A documented planar ramp, thruster, and landing control task.

This is an original educational simulator. It does not model Tesla hardware or
its controller. The ramp approach is a deterministic fixture; PPO controls the
vehicle after ramp exit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ENV_ID = "RoadsterRampLanding-v0"
SIMULATOR_VERSION = 3
PHYSICS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "roadster_v3.json"
OBSERVATION_FIELDS = (
    "x_norm", "z_norm", "vx_norm", "vz_norm", "sin_heading", "cos_heading",
    "omega_norm", "gas_fraction", "flight_phase", "time_fraction",
)
ACTION_FIELDS = ("horizontal_jet", "vertical_jet", "attitude_jet")


@dataclass(frozen=True)
class RoadsterPhysics:
    """All physical values use SI units and are simulation assumptions."""

    version: int = SIMULATOR_VERSION
    dt_seconds: float = 0.04
    max_episode_steps: int = 300
    timeout_penalty: float = -100.0
    ramp_steps: int = 60
    ramp_radius_m: float = 10.0
    car_center_height_m: float = 0.65
    gravity_mps2: float = 9.81
    launch_speed_mps: float = 10.0
    launch_speed_jitter_mps: float = 1.0
    launch_vertical_speed_mps: float = 1.2
    launch_vertical_jitter_mps: float = 0.8
    launch_omega_radps: float = -0.15
    launch_omega_jitter_radps: float = 0.25
    landing_center_x_m: float = -20.0
    landing_center_jitter_m: float = 1.5
    landing_pad_half_width_m: float = 4.5
    max_horizontal_accel_mps2: float = 18.0
    max_vertical_accel_mps2: float = 25.0
    max_attitude_accel_radps2: float = 18.0
    angular_damping_per_second: float = 0.28
    linear_drag_per_second: float = 0.035
    initial_gas_fraction: float = 1.0
    horizontal_gas_per_second: float = 0.14
    vertical_gas_per_second: float = 0.20
    attitude_gas_per_second: float = 0.045
    safe_max_horizontal_speed_mps: float = 4.5
    safe_max_vertical_speed_mps: float = 3.2
    safe_max_attitude_rad: float = 0.18
    safe_max_spin_radps: float = 1.0
    world_limit_x_m: float = 80.0
    world_limit_z_m: float = 80.0
    training_gamma: float = 0.99
    flawed_upright_angle_rad: float = 0.18
    flawed_slow_speed_mps: float = 2.0
    flawed_bonus_per_step: float = 0.35
    shaping_scale: float = 0.75
    potential_distance_weight: float = 1.0
    potential_speed_weight: float = 0.35
    potential_attitude_weight: float = 0.55
    potential_spin_weight: float = 0.12
    potential_gas_weight: float = 0.08

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, path: Path | None = None) -> "RoadsterPhysics":
        config_path = path or PHYSICS_CONFIG_PATH
        values = json.loads(config_path.read_text())
        if values.get("version") != SIMULATOR_VERSION:
            raise ValueError(
                f"Physics config {config_path} has version {values.get('version')}; "
                f"expected {SIMULATOR_VERSION}."
            )
        return cls(**values)


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class RoadsterLandingEnv(gym.Env[np.ndarray, np.ndarray]):
    """A scripted ramp run-up followed by PPO-controlled booster flight.

    Coordinates are x-horizontal, z-up. The 3D frontend adds illustrative
    depth only. Actions are normalized to [-1, 1] and represent horizontal
    impulse, vertical impulse, and pitch torque. Gas use is tracked separately
    from task reward. The ramp run-up is a scripted fixture in v1; fan and
    suction forces are not modeled as separate dynamics.
    """

    metadata = {"render_modes": []}

    def __init__(self, physics: RoadsterPhysics | None = None):
        super().__init__()
        self.physics = physics or RoadsterPhysics.from_config()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.asarray([-3, -1, -3, -3, -1, -1, -3, 0, 0, 0], dtype=np.float32),
            high=np.asarray([3, 3, 3, 3, 1, 1, 3, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self._step_count = 0
        self._ramp_step = 0
        self._flight_steps = 0
        self._phase = "ramp_scripted"
        self._x = self._z = self._vx = self._vz = 0.0
        self._heading = self._omega = 0.0
        self._gas = self.physics.initial_gas_fraction
        self._landing_center = self.physics.landing_center_x_m
        self._landing_x_error: float | None = None
        self._touchdown_speed: float | None = None
        self._touchdown_angle_error: float | None = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._outcome = "In progress"

    @property
    def config(self) -> RoadsterPhysics:
        return self.physics

    def _ramp_pose(self, progress: float) -> tuple[float, float, float, float, float]:
        """Return x, z, heading, vx, vz along a semicircular demo ramp."""
        q = np.clip(progress, 0.0, 1.0) * np.pi
        radius = self.physics.ramp_radius_m
        # Position runs from ground level, over the crest, to an inverted exit.
        x = radius * np.sin(q)
        z = self.physics.car_center_height_m + radius * (1.0 - np.cos(q))
        speed = radius * np.pi / (self.physics.ramp_steps * self.physics.dt_seconds)
        vx = speed * np.cos(q)
        vz = speed * np.sin(q)
        return float(x), float(z), float(q), float(vx), float(vz)

    def _observation(self) -> np.ndarray:
        p = self.physics
        phase = 1.0 if self._phase == "flight" else 0.0
        return np.asarray(
            [
                np.clip((self._x - self._landing_center) / 30.0, -3.0, 3.0),
                np.clip(self._z / 30.0, -1.0, 3.0),
                np.clip(self._vx / 20.0, -3.0, 3.0),
                np.clip(self._vz / 20.0, -3.0, 3.0),
                np.sin(self._heading),
                np.cos(self._heading),
                np.clip(self._omega / 12.0, -3.0, 3.0),
                np.clip(self._gas, 0.0, 1.0),
                phase,
                np.clip(self._step_count / p.max_episode_steps, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def telemetry(self) -> dict[str, float | str]:
        p = self.physics
        return {
            "step": float(self._step_count), "x": float(self._x), "z": float(self._z),
            "vx": float(self._vx), "vz": float(self._vz),
            "heading": float(wrap_angle(self._heading)), "omega": float(self._omega),
            "gas": float(self._gas), "phase": self._phase,
            "horizontal_jet": float(self._last_action[0]),
            "vertical_jet": float(self._last_action[1]),
            "attitude_jet": float(self._last_action[2]),
            "landing_center_x": float(self._landing_center),
            "ramp_radius": float(p.ramp_radius_m),
            "flight_seconds": float(self._flight_steps * p.dt_seconds),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "telemetry": self.telemetry(), "phase": self._phase,
            "landing_center_x": float(self._landing_center),
            "landing_x_error": self._landing_x_error,
            "touchdown_speed": self._touchdown_speed,
            "touchdown_angle_error": self._touchdown_angle_error,
            "gas_remaining": float(self._gas), "outcome": self._outcome,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        p = self.physics
        self._step_count = self._ramp_step = self._flight_steps = 0
        self._phase = "ramp_scripted"
        self._gas = p.initial_gas_fraction
        self._landing_center = p.landing_center_x_m + float(
            self.np_random.uniform(-p.landing_center_jitter_m, p.landing_center_jitter_m)
        )
        self._landing_x_error = self._touchdown_speed = self._touchdown_angle_error = None
        self._last_action = np.zeros(3, dtype=np.float32)
        self._outcome = "In progress"
        self._x, self._z, self._heading, self._vx, self._vz = self._ramp_pose(0.0)
        self._omega = 0.0
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        p = self.physics
        control = np.asarray(action, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(control)):
            raise ValueError("Roadster action must contain only finite values.")
        control = np.clip(control, -1.0, 1.0)
        previous_distance = float(np.hypot(self._x - self._landing_center, self._z - p.car_center_height_m))
        fuel_used = 0.0
        applied_action = control.copy() if self._phase == "flight" else np.zeros(3, dtype=np.float32)
        self._last_action = applied_action
        self._step_count += 1

        if self._phase == "ramp_scripted":
            self._ramp_step += 1
            progress = self._ramp_step / p.ramp_steps
            self._x, self._z, self._heading, self._vx, self._vz = self._ramp_pose(progress)
            if self._ramp_step >= p.ramp_steps:
                self._phase = "flight"
                self._x, self._z, self._heading, _, _ = self._ramp_pose(1.0)
                self._vx = -max(1.0, p.launch_speed_mps + float(self.np_random.uniform(-p.launch_speed_jitter_mps, p.launch_speed_jitter_mps)))
                self._vz = p.launch_vertical_speed_mps + float(self.np_random.uniform(-p.launch_vertical_jitter_mps, p.launch_vertical_jitter_mps))
                self._omega = p.launch_omega_radps + float(self.np_random.uniform(-p.launch_omega_jitter_radps, p.launch_omega_jitter_radps))
            base_reward = 0.0
        else:
            self._flight_steps += 1
            horizontal, vertical_command, attitude = (float(v) for v in control)
            # Map the signed PPO command smoothly onto the one-sided upward jet.
            # A zero policy output now means half thrust instead of a flat dead zone.
            vertical = 0.5 * (vertical_command + 1.0)
            applied_action[1] = vertical
            self._last_action = applied_action
            ax = horizontal * p.max_horizontal_accel_mps2 - p.linear_drag_per_second * self._vx
            az = vertical * p.max_vertical_accel_mps2 - p.gravity_mps2 - p.linear_drag_per_second * self._vz
            angular_accel = attitude * p.max_attitude_accel_radps2 - p.angular_damping_per_second * self._omega
            # Semi-implicit Euler is stable for this fixed, coarse educational model.
            self._vx += ax * p.dt_seconds
            self._vz += az * p.dt_seconds
            self._omega += angular_accel * p.dt_seconds
            self._x += self._vx * p.dt_seconds
            self._z += self._vz * p.dt_seconds
            self._heading = wrap_angle(self._heading + self._omega * p.dt_seconds)
            fuel_used = p.dt_seconds * (
                abs(horizontal) * p.horizontal_gas_per_second
                + vertical * p.vertical_gas_per_second
                + abs(attitude) * p.attitude_gas_per_second
            )
            self._gas = max(0.0, self._gas - fuel_used)
            distance = float(np.hypot(self._x - self._landing_center, self._z - p.car_center_height_m))
            speed = float(np.hypot(self._vx, self._vz))
            heading_error = abs(wrap_angle(self._heading))
            base_reward = (
                0.25 * (previous_distance - distance)
                - 0.008 * speed
                - 0.025 * heading_error
                - 0.008 * abs(self._omega)
                - 0.12 * fuel_used
            )

        terminated = False
        truncated = False
        if self._phase == "flight":
            heading_error = abs(wrap_angle(self._heading))
            touched_ground = self._z <= p.car_center_height_m
            if touched_ground:
                self._z = p.car_center_height_m
                self._landing_x_error = abs(self._x - self._landing_center)
                self._touchdown_speed = float(np.hypot(self._vx, self._vz))
                self._touchdown_angle_error = heading_error
                safe = (
                    self._landing_x_error <= p.landing_pad_half_width_m
                    and abs(self._vx) <= p.safe_max_horizontal_speed_mps
                    and abs(self._vz) <= p.safe_max_vertical_speed_mps
                    and heading_error <= p.safe_max_attitude_rad
                    and abs(self._omega) <= p.safe_max_spin_radps
                )
                self._outcome = "Safe landing" if safe else "Crash"
                base_reward += 100.0 if safe else -100.0
                terminated = True
            elif abs(self._x) > p.world_limit_x_m or self._z > p.world_limit_z_m or self._z < -2.0:
                self._outcome = "Crash"
                base_reward -= 100.0
                terminated = True

        if self._step_count >= p.max_episode_steps and not terminated:
            self._outcome = "Timeout"
            truncated = True
            base_reward += p.timeout_penalty

        info = self._info()
        info.update({
            "base_reward": float(base_reward), "fuel_used": float(fuel_used),
            "safe_landing": self._outcome == "Safe landing",
            "crash": self._outcome == "Crash", "truncated": truncated,
            "action_applied": applied_action.copy(),
        })
        return self._observation(), float(base_reward), terminated, truncated, info


def rule_based_action(env: RoadsterLandingEnv) -> np.ndarray:
    """Simple PD reference controller for flight; ramp playback remains scripted."""
    if env._phase != "flight":
        return np.zeros(3, dtype=np.float32)
    p = env.physics
    angle_error = wrap_angle(-env._heading)
    attitude = np.clip(3.8 * angle_error - 2.4 * env._omega, -1.0, 1.0)
    horizontal = np.clip(0.11 * (env._landing_center - env._x) - 0.40 * env._vx, -1.0, 1.0)
    desired_vz = np.clip(-0.50 * (env._z - p.car_center_height_m), -5.0, -0.5)
    vertical = np.clip(
        (p.gravity_mps2 + 1.3 * (desired_vz - env._vz)) / p.max_vertical_accel_mps2,
        0.0,
        1.0,
    )
    return np.asarray([horizontal, 2.0 * vertical - 1.0, attitude], dtype=np.float32)


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="rocket_landing.envs.roadster:RoadsterLandingEnv")
