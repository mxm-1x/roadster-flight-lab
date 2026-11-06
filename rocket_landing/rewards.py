"""Reward wrappers and the transparent shaping potential."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from rocket_landing.config import REWARD_CONFIG, RewardConfig


def shaping_potential(observation: Sequence[float], config: RewardConfig = REWARD_CONFIG) -> float:
    """Compute the documented potential from LunarLander's eight-value observation.

    Distance combines horizontal displacement from the pad and a lightly weighted
    height term. Negative magnitudes favor low speed, small tilt/spin, and contact.
    The potential is a diagnostic reward component; it does not replace the base reward.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (8,):
        raise ValueError(f"Expected an observation with shape (8,), got {obs.shape}.")
    if not np.isfinite(obs).all():
        raise ValueError("Observation values must be finite.")

    x, height, vx, vy, angle, angular_velocity, left_leg, right_leg = obs
    distance = np.hypot(x, config.potential_height_weight * height)
    contacts = np.clip(left_leg, 0.0, 1.0) + np.clip(right_leg, 0.0, 1.0)
    return float(
        -config.potential_horizontal_distance_weight * distance
        - config.potential_horizontal_speed_weight * abs(vx)
        - config.potential_vertical_speed_weight * abs(vy)
        - config.potential_tilt_weight * abs(angle)
        - config.potential_spin_weight * abs(angular_velocity)
        + config.potential_leg_contact_bonus * contacts
    )


class UprightSlowBonus(gym.RewardWrapper):
    """Deliberately flawed reward: +1 per qualifying nonterminal frame."""

    def __init__(self, env: gym.Env, config: RewardConfig = REWARD_CONFIG) -> None:
        super().__init__(env)
        self.config = config
        self._last_bonus = 0.0

    def reset(self, **kwargs):
        self._last_bonus = 0.0
        return self.env.reset(**kwargs)

    def reward(self, reward: float) -> float:
        return float(reward) + self._last_bonus

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        speed = float(np.hypot(observation[2], observation[3]))
        qualifies = (
            abs(float(observation[4])) < self.config.upright_angle_threshold
            and speed < self.config.slow_speed_threshold
            and not (terminated or truncated)
        )
        self._last_bonus = self.config.flawed_bonus_per_step if qualifies else 0.0
        info = dict(info)
        info["base_reward"] = float(reward)
        info["shaping_component"] = self._last_bonus
        return observation, self.reward(reward), terminated, truncated, info


class PotentialProgressReward(gym.Wrapper):
    """Add discounted potential progress and a small alive-step cost."""

    def __init__(self, env: gym.Env, config: RewardConfig = REWARD_CONFIG) -> None:
        super().__init__(env)
        self.config = config
        self._previous_potential = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_potential = shaping_potential(observation, self.config)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        next_potential = 0.0 if done else shaping_potential(observation, self.config)
        progress = self.config.shaping_gamma * next_potential - self._previous_potential
        time_cost = 0.0 if done else self.config.time_cost_per_step
        shaped_reward = (
            float(reward)
            + self.config.shaping_scale * progress
            - time_cost
        )
        self._previous_potential = next_potential
        info = dict(info)
        info["base_reward"] = float(reward)
        info["shaping_component"] = shaped_reward - float(reward)
        return observation, shaped_reward, terminated, truncated, info


def apply_reward_variant(env: gym.Env, variant: str, config: RewardConfig = REWARD_CONFIG) -> gym.Env:
    """Wrap an environment with a named reward variant."""
    if variant == "baseline":
        return env
    if variant == "flawed_bonus":
        return UprightSlowBonus(env, config)
    if variant == "potential_shaping":
        return PotentialProgressReward(env, config)
    raise ValueError(f"Unknown reward variant {variant!r}.")
