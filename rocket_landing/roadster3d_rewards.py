"""Reward variants for the Rapier-based Roadster simulator v6."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rocket_landing.envs.roadster3d import (
    RoadsterLanding3DEnv,
    _tilt_from_quaternion,
)
from rocket_landing.roadster_rewards import ROADSTER_LABELS, ROADSTER_VARIANTS


class RoadsterReward3DWrapper(gym.Wrapper):
    """Add the v6 experimental reward while retaining original task reward."""

    def __init__(self, env: RoadsterLanding3DEnv, variant: str):
        super().__init__(env)
        if variant not in ROADSTER_VARIANTS:
            raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
        self.variant = variant
        self._potential = 0.0

    def _phi(self) -> float:
        base: RoadsterLanding3DEnv = self.env.unwrapped
        p = base.physics
        position, velocity, angular, quaternion, target = base._state()
        distance = float(np.linalg.norm(target - position))
        speed = float(np.linalg.norm(velocity))
        tilt = _tilt_from_quaternion(quaternion)
        spin = float(np.linalg.norm(angular))
        return float(
            -p.potential_distance_weight * distance / 45.0
            -p.potential_speed_weight * speed / 35.0
            -p.potential_tilt_weight * tilt / np.pi
            -p.potential_spin_weight * spin / 8.0
            +p.potential_fuel_weight * base._fuel
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._potential = self._phi()
        return observation, info

    def step(self, action):
        base: RoadsterLanding3DEnv = self.env.unwrapped
        # The policy controls launch throttle as well as the six flight jets.
        policy_controlled = True
        observation, native_reward, terminated, truncated, info = self.env.step(action)
        p = base.physics
        addition = 0.0
        next_phi = 0.0 if terminated or truncated else self._phi()
        if self.variant == "upright_bonus" and policy_controlled and not (terminated or truncated):
            position, velocity, _, quaternion, _ = base._state()
            if (_tilt_from_quaternion(quaternion) < p.flawed_upright_tilt_rad
                    and float(np.linalg.norm(velocity)) < p.flawed_slow_speed_mps):
                addition = p.flawed_bonus_per_step
        elif self.variant == "potential_shaping" and policy_controlled:
            addition = p.shaping_scale * (p.training_gamma * next_phi - self._potential)
            addition -= p.time_cost_per_step
        self._potential = next_phi if policy_controlled else self._phi()
        shaped_reward = float(native_reward + addition)
        info = dict(info)
        info["base_reward"] = float(native_reward)
        info["shaping_addition"] = float(addition)
        info["shaped_reward"] = shaped_reward
        return observation, shaped_reward, terminated, truncated, info


def apply_roadster3d_reward(env: RoadsterLanding3DEnv, variant: str) -> RoadsterReward3DWrapper:
    return RoadsterReward3DWrapper(env, variant)


ROADSTER3D_LABELS = ROADSTER_LABELS
