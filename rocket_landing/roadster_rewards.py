"""Reward variants for the Roadster landing task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rocket_landing.envs.roadster import RoadsterLandingEnv, wrap_angle

ROADSTER_VARIANTS = ("baseline", "upright_bonus", "potential_shaping")
ROADSTER_LABELS = {
    "baseline": "Task reward",
    "upright_bonus": "Upright dwell probe",
    "potential_shaping": "Potential progress shaping",
}


class RoadsterRewardWrapper(gym.Wrapper):
    """Apply experimental reward shaping while preserving native task reward."""

    def __init__(self, env: RoadsterLandingEnv, variant: str):
        super().__init__(env)
        if variant not in ROADSTER_VARIANTS:
            raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
        self.variant = variant
        self._potential = 0.0

    def _phi(self) -> float:
        base: RoadsterLandingEnv = self.env.unwrapped
        p = base.physics
        distance = np.hypot(base._x - base._landing_center, base._z - p.car_center_height_m)
        speed = np.hypot(base._vx, base._vz)
        attitude = abs(wrap_angle(base._heading))
        return float(
            -p.potential_distance_weight * distance / 30.0
            -p.potential_speed_weight * speed / 20.0
            -p.potential_attitude_weight * attitude / np.pi
            -p.potential_spin_weight * abs(base._omega) / 12.0
            +p.potential_gas_weight * base._gas
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._potential = self._phi()
        return observation, info

    def step(self, action):
        # The ramp fixture consumes environment steps but does not expose
        # controllable actions. Only reward policy-controlled flight.
        policy_controlled = self.env.unwrapped._phase == "flight"
        observation, native_reward, terminated, truncated, info = self.env.step(action)
        p = self.env.unwrapped.physics
        addition = 0.0
        next_phi = 0.0 if terminated or truncated else self._phi()
        if self.variant == "upright_bonus" and policy_controlled and not (terminated or truncated):
            base: RoadsterLandingEnv = self.env.unwrapped
            speed = float(np.hypot(base._vx, base._vz))
            if abs(wrap_angle(base._heading)) < p.flawed_upright_angle_rad and speed < p.flawed_slow_speed_mps:
                addition = p.flawed_bonus_per_step
        elif self.variant == "potential_shaping" and policy_controlled:
            addition = p.shaping_scale * (p.training_gamma * next_phi - self._potential)
        # Track the changing scripted state without assigning its progress to PPO.
        self._potential = next_phi if policy_controlled else self._phi()
        shaped_reward = float(native_reward + addition)
        info = dict(info)
        info["base_reward"] = float(native_reward)
        info["shaping_addition"] = float(addition)
        info["shaped_reward"] = shaped_reward
        return observation, shaped_reward, terminated, truncated, info


def apply_roadster_reward(env: RoadsterLandingEnv, variant: str) -> RoadsterRewardWrapper:
    return RoadsterRewardWrapper(env, variant)
