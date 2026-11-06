"""Record complete training flights without changing the environment dynamics."""

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_W


class TrainingRecorder(gym.Wrapper):
    """Write a compact trajectory for every completed training episode."""

    def __init__(self, env: gym.Env, directory: Path):
        super().__init__(env)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "episodes.csv"
        if not self.index_path.exists():
            with self.index_path.open("w", newline="") as file:
                csv.writer(file).writerow(["episode", "outcome", "native_return", "shaped_return", "length"])
        self.episode = 0
        self.states = []
        self.actions = []
        self.native_rewards = []
        self.shaped_rewards = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.states = [np.asarray(observation, dtype=np.float32).copy()]
        self.actions, self.native_rewards, self.shaped_rewards = [], [], []
        center = VIEWPORT_W / SCALE / 2
        base = self.env.unwrapped
        self.pad = (float(base.helipad_x1 - center), float(base.helipad_x2 - center))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        native = float(info.get("base_reward", reward))
        self.states.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.native_rewards.append(native)
        self.shaped_rewards.append(float(reward))
        if terminated or truncated:
            outcome = "Timeout" if truncated else "Safe landing" if native > 0 else "Crash"
            self.episode += 1
            np.savez_compressed(
                self.directory / f"episode_{self.episode:07d}.npz",
                states=np.asarray(self.states, dtype=np.float32),
                actions=np.asarray(self.actions, dtype=np.int8),
                native_rewards=np.asarray(self.native_rewards, dtype=np.float64),
                shaped_rewards=np.asarray(self.shaped_rewards, dtype=np.float64),
                pad=np.asarray(self.pad, dtype=np.float32),
                outcome=np.asarray(outcome),
            )
            with self.index_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    self.episode, outcome, sum(self.native_rewards),
                    sum(self.shaped_rewards), len(self.actions),
                ])
            info = dict(info)
            info["outcome"] = outcome
            info["native_return"] = float(sum(self.native_rewards))
        return observation, reward, terminated, truncated, info
