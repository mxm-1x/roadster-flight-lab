"""Reproducible PPO training and evaluation for Roadster Flight Lab."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rocket_landing import __version__
from rocket_landing.config import DEFAULT_OUTPUT_DIR
from rocket_landing.envs.roadster import (
    ACTION_FIELDS,
    ENV_ID,
    PHYSICS_CONFIG_PATH,
    OBSERVATION_FIELDS,
    SIMULATOR_VERSION,
    RoadsterLandingEnv,
    RoadsterPhysics,
    rule_based_action,
)
from rocket_landing.roadster_rewards import ROADSTER_LABELS, ROADSTER_VARIANTS, apply_roadster_reward

ROADSTER_SEEDS = (0, 1, 2, 3, 4)
ROADSTER_TIMESTEPS = 200_000
ROADSTER_EVAL_EPISODES = 100
ROADSTER_EVAL_SEED_BASE = 20_000
ROADSTER_CHECKPOINT_INTERVAL = 25_000
ROADSTER_BEHAVIOR_CLONING = {
    "expert": "rule_based_pd_v1",
    "episodes": 64,
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "seed_base": 100_000,
    "dagger_rounds": 3,
    "dagger_episodes_per_round": 16,
    "dagger_epochs_per_round": 10,
    "dagger_seed_base": 200_000,
}
ROADSTER_PPO = {
    "policy": "MlpPolicy",
    "policy_kwargs": {"net_arch": [128, 128], "log_std_init": -2.0},
    "learning_rate": 1e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "target_kl": 0.01,
    "device": "cpu",
}
_T_95_CRITICAL = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                  7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def roadster_root(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve() / "roadster"


def roadster_run_dir(output_dir: Path | str, variant: str, seed: int) -> Path:
    if variant not in ROADSTER_VARIANTS:
        raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return roadster_root(output_dir) / variant / f"seed_{seed}"


class RoadsterRecorder(gym.Wrapper):
    """Persist exact state/action/reward rows for completed training episodes."""

    def __init__(self, env: gym.Env, directory: Path):
        super().__init__(env)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = directory / "episodes.csv"
        if not self.index_path.exists():
            with self.index_path.open("w", newline="") as file:
                csv.writer(file).writerow(["episode", "outcome", "native_return", "shaped_return", "length"])
        self.episode = 0
        self.rows: list[dict[str, Any]] = []
        self.actions: list[np.ndarray] = []
        self.native_rewards: list[float] = []
        self.shaped_rewards: list[float] = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.rows = [dict(info["telemetry"])]
        self.actions, self.native_rewards, self.shaped_rewards = [], [], []
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rows.append(dict(info["telemetry"]))
        self.actions.append(np.asarray(info["action_applied"], dtype=np.float32).copy())
        self.native_rewards.append(float(info["base_reward"]))
        self.shaped_rewards.append(float(reward))
        if terminated or truncated:
            self.episode += 1
            np.savez_compressed(
                self.directory / f"episode_{self.episode:07d}.npz",
                telemetry=np.asarray(
                    [[float(row.get(key, 0.0)) if not isinstance(row.get(key), str) else 0.0
                      for key in ROADSTER_TELEMETRY_NUMERIC] for row in self.rows],
                    dtype=np.float32,
                ),
                phases=np.asarray([str(row["phase"]) for row in self.rows], dtype="U24"),
                actions=np.asarray(self.actions, dtype=np.float32),
                native_rewards=np.asarray(self.native_rewards, dtype=np.float64),
                shaped_rewards=np.asarray(self.shaped_rewards, dtype=np.float64),
                outcome=np.asarray(info["outcome"]),
            )
            with self.index_path.open("a", newline="") as file:
                csv.writer(file).writerow([
                    self.episode, info["outcome"], sum(self.native_rewards),
                    sum(self.shaped_rewards), len(self.actions),
                ])
            info = dict(info)
            info["native_return"] = float(sum(self.native_rewards))
        return observation, reward, terminated, truncated, info


ROADSTER_TELEMETRY_NUMERIC = (
    "step", "x", "z", "vx", "vz", "heading", "omega", "gas",
    "horizontal_jet", "vertical_jet", "attitude_jet", "landing_center_x",
    "ramp_radius", "flight_seconds",
)


def make_roadster_env(variant: str, trajectory_dir: Path | None = None, monitor_path: Path | None = None):
    env: gym.Env = apply_roadster_reward(RoadsterLandingEnv(), variant)
    if trajectory_dir is not None:
        env = RoadsterRecorder(env, trajectory_dir)
    if monitor_path is not None:
        env = Monitor(
            env,
            str(monitor_path),
            info_keywords=("outcome", "native_return", "safe_landing", "crash", "truncated"),
        )
    return env


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "gymnasium": gym.__version__, "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__, "numpy": np.__version__,
    }


def collect_rule_controller_data(seed: int, episodes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Collect identical PD-controller demonstrations for reward variants."""
    count = episodes or ROADSTER_BEHAVIOR_CLONING["episodes"]
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    env = RoadsterLandingEnv()
    try:
        for episode in range(count):
            observation, _ = env.reset(seed=ROADSTER_BEHAVIOR_CLONING["seed_base"] + seed * count + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                action = rule_based_action(env)
                if env._phase == "flight":
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(np.asarray(action, dtype=np.float32).copy())
                observation, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def collect_dagger_data(model: PPO, seed: int, round_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Query the PD expert on states visited by the current learner policy."""
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    env = RoadsterLandingEnv()
    count = ROADSTER_BEHAVIOR_CLONING["dagger_episodes_per_round"]
    try:
        for episode in range(count):
            observation, _ = env.reset(seed=ROADSTER_BEHAVIOR_CLONING["dagger_seed_base"] + seed * count * 4 + round_index * count + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                if env._phase == "flight":
                    expert_action = rule_based_action(env)
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    expert_actions.append(np.asarray(expert_action, dtype=np.float32).copy())
                    learner_action, _ = model.predict(observation, deterministic=True)
                    applied_action = np.asarray(learner_action, dtype=np.float32).reshape(3)
                else:
                    expert_action = np.zeros(3, dtype=np.float32)
                    applied_action = expert_action
                observation, _, terminated, truncated, _ = env.step(applied_action)
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(expert_actions, dtype=np.float32)


def behavior_clone_policy(model: PPO, seed: int, demonstration_path: Path | None = None) -> dict[str, Any]:
    """Warm-start PPO's actor with PD demos and learner-state DAgger labels."""
    observations, actions = collect_rule_controller_data(seed)
    policy = model.policy
    device = policy.device
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=ROADSTER_BEHAVIOR_CLONING["learning_rate"]
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = ROADSTER_BEHAVIOR_CLONING["batch_size"]
    losses: list[float] = []

    def fit_epochs(epoch_count: int) -> None:
        inputs = torch.as_tensor(observations, dtype=torch.float32, device=device)
        targets = torch.as_tensor(actions, dtype=torch.float32, device=device)
        for _ in range(epoch_count):
            permutation = torch.randperm(len(inputs), generator=generator)
            epoch_losses: list[float] = []
            for start in range(0, len(inputs), batch_size):
                batch = permutation[start : start + batch_size].to(device)
                features = policy.extract_features(inputs[batch])
                latent_policy, _ = policy.mlp_extractor(features)
                predicted_actions = policy.action_net(latent_policy)
                loss = F.mse_loss(predicted_actions, targets[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)))

    fit_epochs(ROADSTER_BEHAVIOR_CLONING["epochs"])
    initial_samples = len(observations)
    for round_index in range(ROADSTER_BEHAVIOR_CLONING["dagger_rounds"]):
        new_observations, new_actions = collect_dagger_data(model, seed, round_index)
        observations = np.concatenate((observations, new_observations), axis=0)
        actions = np.concatenate((actions, new_actions), axis=0)
        fit_epochs(ROADSTER_BEHAVIOR_CLONING["dagger_epochs_per_round"])
    if demonstration_path is not None:
        np.savez_compressed(demonstration_path, observations=observations, actions=actions)
    return {
        **ROADSTER_BEHAVIOR_CLONING,
        "seed": seed,
        "training_transitions": int(len(observations)),
        "initial_expert_transitions": int(initial_samples),
        "initial_mse": losses[0] if losses else None,
        "final_mse": losses[-1] if losses else None,
        "epoch_losses": losses,
    }


def train_roadster_run(
    variant: str,
    seed: int,
    timesteps: int = ROADSTER_TIMESTEPS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_interval: int = ROADSTER_CHECKPOINT_INTERVAL,
) -> Path:
    if variant not in ROADSTER_VARIANTS:
        raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
    if seed < 0 or timesteps <= 0 or checkpoint_interval <= 0:
        raise ValueError("seed must be nonnegative and step counts must be positive.")
    run_dir = roadster_run_dir(output_dir, variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    recording_dir = run_dir / "training_trajectories" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    status_path = run_dir / "training_status.json"
    status: dict[str, Any] = {
        "environment": ENV_ID, "variant": variant, "seed": seed,
        "timesteps_requested": timesteps, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
    }

    def save_status() -> None:
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    save_status()
    env = None
    try:
        monitor_path = run_dir / "training_monitor.csv"
        env = make_roadster_env(variant, recording_dir, monitor_path)
        env.reset(seed=seed)
        torch.set_num_threads(1)
        model = PPO(
            "MlpPolicy", env, seed=seed, verbose=1,
            policy_kwargs=ROADSTER_PPO["policy_kwargs"],
            learning_rate=ROADSTER_PPO["learning_rate"], n_steps=ROADSTER_PPO["n_steps"],
            batch_size=ROADSTER_PPO["batch_size"], n_epochs=ROADSTER_PPO["n_epochs"],
            gamma=ROADSTER_PPO["gamma"], gae_lambda=ROADSTER_PPO["gae_lambda"],
            clip_range=ROADSTER_PPO["clip_range"], ent_coef=ROADSTER_PPO["ent_coef"],
            target_kl=ROADSTER_PPO["target_kl"], device="cpu",
        )
        status["stage"] = "behavior_cloning_warm_start"
        save_status()
        demonstration_path = run_dir / "behavior_cloning_demonstrations.npz"
        behavior_cloning = behavior_clone_policy(model, seed, demonstration_path)
        (run_dir / "behavior_cloning.json").write_text(json.dumps(behavior_cloning, indent=2) + "\n")
        model.save(run_dir / "model_behavior_cloned.zip")
        status["stage"] = "ppo_fine_tuning"
        save_status()
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
        callback = CheckpointCallback(
            save_freq=checkpoint_interval, save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_roadster",
        )
        model.learn(total_timesteps=timesteps, callback=callback)
        model_path = run_dir / "model.zip"
        model.save(model_path)
        metadata = {
            "project_version": __version__, "environment": ENV_ID,
            "simulator_version": SIMULATOR_VERSION, "variant": variant,
            "variant_label": ROADSTER_LABELS[variant], "seed": seed,
            "observation_fields": list(OBSERVATION_FIELDS), "action_fields": list(ACTION_FIELDS),
            "physics_config_path": "rocket_landing/configs/roadster_v3.json",
            "physics_config_sha256": hashlib.sha256(PHYSICS_CONFIG_PATH.read_bytes()).hexdigest(),
            "physics_config": RoadsterPhysics.from_config().to_dict(), "ppo_config": ROADSTER_PPO,
            "behavior_cloning": {key: value for key, value in behavior_cloning.items() if key != "epoch_losses"},
            "action_mapping": {
                "horizontal_jet": "PPO action clipped to [-1, 1]",
                "vertical_jet": "(PPO action + 1) / 2, giving a smooth normalized [0, 1] upward thrust",
                "attitude_jet": "PPO action clipped to [-1, 1]",
            },
            "timesteps_requested": timesteps, "timesteps_completed": model.num_timesteps,
            "device": "cpu", "versions": runtime_versions(), "model_path": "model.zip",
            "behavior_cloned_model_path": "model_behavior_cloned.zip",
            "behavior_cloning_demonstrations_path": "behavior_cloning_demonstrations.npz",
            "ramp_controller": "scripted_fixture_v1", "training_trajectory_dir": str(recording_dir.relative_to(run_dir)),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status.update({
            "state": "complete", "timesteps_completed": model.num_timesteps,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save_status()
        return model_path
    except BaseException as exc:
        status.update({
            "state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        save_status()
        raise
    finally:
        if env is not None:
            env.close()


def confidence_interval(values: pd.Series, *, binary: bool = False) -> tuple[float | None, float | None]:
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numbers):
        return None, None
    mean = float(numbers.mean())
    if binary:
        z = 1.96
        denominator = 1.0 + z * z / len(numbers)
        center = (mean + z * z / (2.0 * len(numbers))) / denominator
        half = z * np.sqrt(mean * (1.0 - mean) / len(numbers) + z * z / (4.0 * len(numbers) ** 2)) / denominator
        return max(0.0, center - half), min(1.0, center + half)
    if len(numbers) < 2:
        return mean, mean
    margin = 1.96 * float(numbers.std(ddof=1)) / np.sqrt(len(numbers))
    return mean - margin, mean + margin


def seed_mean_confidence_interval(
    values: pd.Series, *, bounded: bool = False
) -> tuple[float | None, float | None]:
    """Student-t interval for run-level means across independent training seeds."""
    numbers = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numbers) < 2:
        return None, None
    mean = float(numbers.mean())
    standard_error = float(numbers.std(ddof=1)) / np.sqrt(len(numbers))
    critical = _T_95_CRITICAL.get(len(numbers), 1.96)
    low, high = mean - critical * standard_error, mean + critical * standard_error
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return low, high


def _evaluation_episode(model: PPO, seed: int, variant: str, save_path: Path) -> dict[str, Any]:
    env = make_roadster_env(variant)
    telemetry: list[dict[str, Any]] = []
    actions: list[list[float]] = []
    native_rewards: list[float] = []
    shaped_rewards: list[float] = []
    native_return = shaped_return = 0.0
    try:
        observation, info = env.reset(seed=seed)
        telemetry.append(dict(info["telemetry"]))
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(np.asarray(info["action_applied"], dtype=np.float32).reshape(3).tolist())
            native = float(info["base_reward"])
            native_return += native
            shaped_return += float(reward)
            native_rewards.append(native)
            shaped_rewards.append(float(reward))
            telemetry.append(dict(info["telemetry"]))
        final = info
        data = np.asarray(
            [[float(row.get(key, 0.0)) for key in ROADSTER_TELEMETRY_NUMERIC] for row in telemetry],
            dtype=np.float32,
        )
        phases = np.asarray([str(row["phase"]) for row in telemetry], dtype="U24")
        np.savez_compressed(
            save_path, telemetry=data, phases=phases,
            actions=np.asarray(actions, dtype=np.float32),
            native_rewards=np.asarray(native_rewards, dtype=np.float64),
            shaped_rewards=np.asarray(shaped_rewards, dtype=np.float64),
            outcome=np.asarray(final["outcome"]),
        )
        p = env.unwrapped.physics
        return {
            "eval_seed": seed, "outcome": final["outcome"],
            "safe_landing": bool(final["safe_landing"]), "crash": bool(final["crash"]),
            "truncated": bool(final["truncated"]),
            "native_return": native_return, "shaped_return": shaped_return,
            "episode_length": len(actions), "flight_seconds": float(final["telemetry"]["flight_seconds"]),
            "landing_x_error": final["landing_x_error"],
            "touchdown_speed": final["touchdown_speed"],
            "touchdown_angle_error": final["touchdown_angle_error"],
            "gas_remaining": float(final["gas_remaining"]),
            "pad_half_width_m": p.landing_pad_half_width_m,
        }
    finally:
        env.close()


def evaluate_roadster_run(
    model_path: Path | str,
    variant: str,
    seed: int,
    episodes: int = ROADSTER_EVAL_EPISODES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variant not in ROADSTER_VARIANTS or episodes <= 0:
        raise ValueError("Use a supported reward variant and a positive episode count.")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(model_path, device="cpu")
    run_dir = roadster_run_dir(output_dir, variant, seed)
    flight_dir = run_dir / "evaluation_flights"
    flight_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _evaluation_episode(model, ROADSTER_EVAL_SEED_BASE + episode, variant,
                           flight_dir / f"episode_{episode:04d}.npz")
        for episode in range(episodes)
    ]
    frame = pd.DataFrame(rows)
    frame.insert(0, "episode", np.arange(episodes, dtype=int))
    frame.to_csv(run_dir / "evaluation_episodes.csv", index=False)
    metrics = [
        "native_return", "shaped_return", "safe_landing", "crash", "truncated",
        "landing_x_error", "touchdown_speed", "touchdown_angle_error",
        "flight_seconds", "gas_remaining",
    ]
    summary: dict[str, Any] = {
        "environment": ENV_ID, "simulator_version": SIMULATOR_VERSION,
        "variant": variant, "variant_label": ROADSTER_LABELS[variant],
        "seed": seed, "episodes": episodes,
        "mean_reward_gap": float((frame["shaped_return"] - frame["native_return"]).mean()),
        "evaluation_seed_base": ROADSTER_EVAL_SEED_BASE,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    for metric in metrics:
        series = pd.to_numeric(frame[metric], errors="coerce")
        low, high = confidence_interval(series, binary=metric in {"safe_landing", "crash", "truncated"})
        summary[f"mean_{metric}"] = float(series.mean()) if series.notna().any() else None
        summary[f"std_{metric}"] = float(series.std(ddof=1)) if episodes > 1 and series.notna().sum() > 1 else (0.0 if series.notna().any() else None)
        summary[f"ci95_{metric}_low"] = low
        summary[f"ci95_{metric}_high"] = high
    summary["safe_landing_rate"] = summary["mean_safe_landing"]
    summary["crash_rate"] = summary["mean_crash"]
    summary["timeout_rate"] = summary["mean_truncated"]
    summary["mean_native_return"] = summary["mean_native_return"]
    summary["mean_shaped_return"] = summary["mean_shaped_return"]
    (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return frame, summary


def run_roadster_experiments(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timesteps: int = ROADSTER_TIMESTEPS,
    episodes: int = ROADSTER_EVAL_EPISODES,
    seeds: tuple[int, ...] = ROADSTER_SEEDS,
    variants: tuple[str, ...] = ROADSTER_VARIANTS,
    force: bool = False,
) -> pd.DataFrame:
    root = roadster_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if variant not in ROADSTER_VARIANTS:
            raise ValueError(f"Unknown Roadster reward variant: {variant}")
        for seed in seeds:
            run_dir = roadster_run_dir(output_dir, variant, seed)
            status_path = run_dir / "training_status.json"
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "evaluation_summary.json"
            evaluation_path = run_dir / "evaluation_episodes.csv"
            model_path = run_dir / "model.zip"
            reusable = all(path.is_file() for path in (status_path, metadata_path, summary_path, evaluation_path, model_path))
            if reusable:
                status = json.loads(status_path.read_text())
                metadata = json.loads(metadata_path.read_text())
                summary = json.loads(summary_path.read_text())
                reusable = (
                    status.get("state") == "complete"
                    and metadata.get("simulator_version") == SIMULATOR_VERSION
                    and metadata.get("variant") == variant
                    and metadata.get("seed") == seed
                    and metadata.get("timesteps_completed", 0) >= timesteps
                    and metadata.get("ppo_config") == ROADSTER_PPO
                    and all(metadata.get("behavior_cloning", {}).get(key) == value for key, value in ROADSTER_BEHAVIOR_CLONING.items())
                    and summary.get("simulator_version") == SIMULATOR_VERSION
                    and summary.get("episodes", 0) >= episodes
                    and summary.get("evaluation_seed_base") == ROADSTER_EVAL_SEED_BASE
                    and len(pd.read_csv(evaluation_path)) >= episodes
                )
            if reusable and not force:
                print(f"Reusing complete {variant} seed {seed} run at {metadata['timesteps_completed']} steps.")
                continue
            model_path = train_roadster_run(variant, seed, timesteps, output_dir)
            evaluate_roadster_run(model_path, variant, seed, episodes, output_dir)
    return write_roadster_comparison(output_dir)


def write_roadster_comparison(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Rebuild aggregate reports from completed, current-version run summaries."""
    root = roadster_root(output_dir)
    summaries: list[dict[str, Any]] = []
    for variant in ROADSTER_VARIANTS:
        for summary_path in sorted((root / variant).glob("seed_*/evaluation_summary.json")):
            metadata_path = summary_path.parent / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            matching_protocol = (
                metadata.get("physics_config") == RoadsterPhysics.from_config().to_dict()
                and metadata.get("ppo_config") == ROADSTER_PPO
                and all(metadata.get("behavior_cloning", {}).get(key) == value for key, value in ROADSTER_BEHAVIOR_CLONING.items())
            )
            if (metadata.get("simulator_version") == SIMULATOR_VERSION
                    and summary.get("simulator_version") == SIMULATOR_VERSION
                    and matching_protocol):
                summaries.append(summary)
    per_run = pd.DataFrame(summaries)
    if per_run.empty:
        return per_run
    per_run.to_csv(root / "comparison_per_run.csv", index=False)
    aggregate = per_run.groupby(["variant", "variant_label"], as_index=False).agg(
        training_seeds=("seed", "nunique"),
        mean_safe_landing_rate=("safe_landing_rate", "mean"),
        std_safe_landing_rate=("safe_landing_rate", "std"),
        mean_crash_rate=("crash_rate", "mean"),
        mean_timeout_rate=("timeout_rate", "mean"),
        mean_native_return=("mean_native_return", "mean"),
        std_native_return_across_seeds=("mean_native_return", "std"),
        mean_touchdown_speed=("mean_touchdown_speed", "mean"),
        mean_landing_x_error=("mean_landing_x_error", "mean"),
        mean_gas_remaining=("mean_gas_remaining", "mean"),
        mean_reward_gap=("mean_reward_gap", "mean"),
    )
    for aggregate_metric, run_metric in {
        "mean_safe_landing_rate": "safe_landing_rate",
        "mean_crash_rate": "crash_rate",
        "mean_timeout_rate": "timeout_rate",
        "mean_native_return": "mean_native_return",
        "mean_touchdown_speed": "mean_touchdown_speed",
        "mean_landing_x_error": "mean_landing_x_error",
        "mean_gas_remaining": "mean_gas_remaining",
        "mean_reward_gap": "mean_reward_gap",
    }.items():
        low_values: list[float | None] = []
        high_values: list[float | None] = []
        for _, row in aggregate.iterrows():
            subset = per_run.loc[per_run["variant"] == row["variant"], run_metric]
            if subset.notna().sum() < 2:
                low, high = None, None
            else:
                # These are per-seed estimates, not individual binary episodes.
                low, high = seed_mean_confidence_interval(
                    subset, bounded=run_metric.endswith("_rate")
                )
            low_values.append(low)
            high_values.append(high)
        aggregate[f"ci95_{aggregate_metric}_low"] = low_values
        aggregate[f"ci95_{aggregate_metric}_high"] = high_values
    aggregate.to_csv(root / "comparison_summary.csv", index=False)
    return per_run


def export_rule_based_preview(output_dir: Path | str = DEFAULT_OUTPUT_DIR, seed: int = 20_000) -> Path:
    """Write one clearly labeled non-ML controller trajectory for the website."""
    root = roadster_root(output_dir) / "reference_controller"
    root.mkdir(parents=True, exist_ok=True)
    env = RoadsterLandingEnv()
    observation, info = env.reset(seed=seed)
    rows = [dict(info["telemetry"])]
    actions, rewards = [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = rule_based_action(env)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append(dict(info["telemetry"]))
        actions.append(info["action_applied"].tolist())
        rewards.append(float(reward))
    data = np.asarray([[float(row.get(k, 0.0)) for k in ROADSTER_TELEMETRY_NUMERIC] for row in rows], dtype=np.float32)
    phases = np.asarray([str(row["phase"]) for row in rows], dtype="U24")
    path = root / "reference_episode.npz"
    np.savez_compressed(path, telemetry=data, phases=phases,
                        actions=np.asarray(actions, dtype=np.float32),
                        native_rewards=np.asarray(rewards), shaped_rewards=np.asarray(rewards),
                        outcome=np.asarray(info["outcome"]))
    env.close()
    return path


def evaluate_rule_controller(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    episodes: int = ROADSTER_EVAL_EPISODES,
) -> dict[str, Any]:
    """Measure the non-ML PD reference on the shared held-out launch sequence."""
    root = roadster_root(output_dir) / "reference_controller"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    env = RoadsterLandingEnv()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=ROADSTER_EVAL_SEED_BASE + episode)
            native_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = rule_based_action(env)
                observation, reward, terminated, truncated, info = env.step(action)
                native_return += float(reward)
            rows.append({
                "episode": episode, "eval_seed": ROADSTER_EVAL_SEED_BASE + episode,
                "outcome": info["outcome"], "safe_landing": bool(info["safe_landing"]),
                "crash": bool(info["crash"]), "truncated": bool(info["truncated"]),
                "native_return": native_return, "landing_x_error": info["landing_x_error"],
                "touchdown_speed": info["touchdown_speed"],
                "touchdown_angle_error": info["touchdown_angle_error"],
                "gas_remaining": info["gas_remaining"],
                "flight_seconds": float(info["telemetry"]["flight_seconds"]),
            })
    finally:
        env.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "evaluation_episodes.csv", index=False)
    summary: dict[str, Any] = {
        "environment": ENV_ID, "simulator_version": SIMULATOR_VERSION,
        "controller": "rule_based_pd_v1", "episodes": episodes,
        "evaluation_seed_base": ROADSTER_EVAL_SEED_BASE,
        "safe_landings": int(frame["safe_landing"].sum()),
        "safe_landing_rate": float(frame["safe_landing"].mean()),
        "crash_rate": float(frame["crash"].mean()),
        "timeout_rate": float(frame["truncated"].mean()),
        "mean_native_return": float(frame["native_return"].mean()),
    }
    for metric in (
        "safe_landing", "crash", "truncated", "native_return", "landing_x_error",
        "touchdown_speed", "touchdown_angle_error", "gas_remaining", "flight_seconds",
    ):
        values = pd.to_numeric(frame[metric], errors="coerce")
        low, high = confidence_interval(values, binary=metric in {"safe_landing", "crash", "truncated"})
        summary[f"ci95_{metric}_low"] = low
        summary[f"ci95_{metric}_high"] = high
    (root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
