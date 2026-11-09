"""Reward variants for the Rapier-based Roadster 3D simulators."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rocket_landing.envs.roadster3d_v7 import _tilt_from_quaternion
from rocket_landing.roadster_rewards import ROADSTER_LABELS, ROADSTER_VARIANTS

GUIDED_LANDING_VARIANT = "landing_guidance"
ROADSTER3D_V7_VARIANTS = (*ROADSTER_VARIANTS, GUIDED_LANDING_VARIANT)
ROADSTER3D_V7_LABELS = {**ROADSTER_LABELS, GUIDED_LANDING_VARIANT: "Touchdown-aware potential"}


class RoadsterReward3DWrapper(gym.Wrapper):
    """Add an experimental reward while retaining the environment reward."""

    def __init__(self, env: gym.Env, variant: str):
        super().__init__(env)
        if variant not in ROADSTER3D_V7_VARIANTS:
            raise ValueError(f"variant must be one of {ROADSTER3D_V7_VARIANTS}.")
        self.variant = variant
        self._potential = 0.0

    def _phi(self) -> float:
        base = self.env.unwrapped
        p = base.physics
        position, velocity, angular, quaternion, target = base._state()
        distance = float(np.linalg.norm(target - position))
        if self.variant == GUIDED_LANDING_VARIANT:
            height = max(0.0, float(position[1]))
            target_descent_speed = -min(1.2, max(0.25, 0.35 * height))
            horizontal_speed = float(np.linalg.norm(velocity[[0, 2]]))
            vertical_error = abs(float(velocity[1]) - target_descent_speed)
            tilt = _tilt_from_quaternion(quaternion)
            spin = float(np.linalg.norm(angular))
            # Dense, bounded progress toward low lateral speed, a controlled
            # descent, level attitude and low spin, while retaining a gas
            # reserve. These weights define a new experimental objective.
            return float(
                -distance / 45.0
                -0.35 * horizontal_speed / 8.0
                -0.7 * vertical_error / 8.0
                -0.7 * tilt / np.pi
                -0.2 * spin / 8.0
                +0.35 * base._fuel
            )
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
        base = self.env.unwrapped
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
        elif self.variant in {"potential_shaping", GUIDED_LANDING_VARIANT} and policy_controlled:
            addition = p.shaping_scale * (p.training_gamma * next_phi - self._potential)
            addition -= p.time_cost_per_step
        self._potential = next_phi if policy_controlled else self._phi()
        shaped_reward = float(native_reward + addition)
        info = dict(info)
        info["base_reward"] = float(native_reward)
        info["shaping_addition"] = float(addition)
        info["shaped_reward"] = shaped_reward
        return observation, shaped_reward, terminated, truncated, info


def apply_roadster3d_reward(env: gym.Env, variant: str) -> RoadsterReward3DWrapper:
    return RoadsterReward3DWrapper(env, variant)


ROADSTER3D_LABELS = ROADSTER_LABELS
