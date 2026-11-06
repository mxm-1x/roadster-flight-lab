"""PPO training and held-out evaluation for the Rapier 3D Roadster task."""

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
import rapier3d
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rocket_landing import __version__
from rocket_landing.config import DEFAULT_OUTPUT_DIR
from rocket_landing.envs.roadster3d import (
    ACTION_FIELDS_3D,
    ENV_ID_3D,
    OBSERVATION_FIELDS_3D,
    PHYSICS_CONFIG_3D_PATH,
    SIMULATOR_VERSION_3D,
    TELEMETRY_FIELDS_3D,
    RoadsterLanding3DEnv,
    RoadsterPhysics3D,
)
from rocket_landing.roadster_rewards import ROADSTER_LABELS, ROADSTER_VARIANTS
from rocket_landing.roadster3d_rewards import apply_roadster3d_reward


ROADSTER3D_SEEDS = (0, 1, 2, 3, 4)
ROADSTER3D_TIMESTEPS = 300_000
ROADSTER3D_EVAL_EPISODES = 100
ROADSTER3D_EVAL_SEED_BASE = 40_000
ROADSTER3D_CHECKPOINT_INTERVAL = 50_000
ROADSTER3D_PPO = {
    "policy": "MlpPolicy",
    "policy_kwargs": {"net_arch": [128, 128], "log_std_init": -1.0},
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "device": "cpu",
}


def roadster3d_root(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve() / "roadster3d_v6"


def roadster3d_run_dir(output_dir: Path | str, variant: str, seed: int) -> Path:
    if variant not in ROADSTER_VARIANTS:
        raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return roadster3d_root(output_dir) / variant / f"seed_{seed}"


class Roadster3DRecorder(gym.Wrapper):
    """Save every training attempt, including launch failures and crashes."""

    def __init__(self, env: gym.Env, directory: Path):
        super().__init__(env)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = directory / "episodes.csv"
        if not self.index_path.exists():
            with self.index_path.open("w", newline="") as file:
                csv.writer(file).writerow(
                    ["episode", "outcome", "native_return", "shaped_return", "length"]
                )
        self.episode = 0
        self.rows: list[dict[str, Any]] = []
        self.phases: list[str] = []
        self.actions: list[np.ndarray] = []
        self.native_rewards: list[float] = []
        self.shaped_rewards: list[float] = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.rows = [dict(info["telemetry"])]
        self.phases = [str(info["phase"])]
        self.actions, self.native_rewards, self.shaped_rewards = [], [], []
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rows.append(dict(info["telemetry"]))
        self.phases.append(str(info["phase"]))
        self.actions.append(np.asarray(info["action_applied"], dtype=np.float32).copy())
        self.native_rewards.append(float(info["base_reward"]))
        self.shaped_rewards.append(float(reward))
        if terminated or truncated:
            self.episode += 1
            np.savez_compressed(
                self.directory / f"episode_{self.episode:07d}.npz",
                telemetry=np.asarray(
                    [[float(row.get(key, 0.0)) for key in TELEMETRY_FIELDS_3D] for row in self.rows],
                    dtype=np.float32,
                ),
                phases=np.asarray(self.phases, dtype="U16"),
                actions=np.asarray(self.actions, dtype=np.float32),
                native_rewards=np.asarray(self.native_rewards, dtype=np.float64),
                shaped_rewards=np.asarray(self.shaped_rewards, dtype=np.float64),
                outcome=np.asarray(info["outcome"]),
            )
            native_return = float(sum(self.native_rewards))
            shaped_return = float(sum(self.shaped_rewards))
            with self.index_path.open("a", newline="") as file:
                csv.writer(file).writerow(
                    [self.episode, info["outcome"], native_return, shaped_return, len(self.actions)]
                )
            info = dict(info)
            info["native_return"] = native_return
        return observation, reward, terminated, truncated, info


def make_roadster3d_env(
    variant: str, trajectory_dir: Path | None = None, monitor_path: Path | None = None
):
    env: gym.Env = apply_roadster3d_reward(RoadsterLanding3DEnv(), variant)
    if trajectory_dir is not None:
        env = Roadster3DRecorder(env, trajectory_dir)
    if monitor_path is not None:
        env = Monitor(
            env,
            str(monitor_path),
            info_keywords=("outcome", "native_return", "safe_landing", "crash", "truncated"),
        )
    return env


def runtime_versions_3d() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gymnasium": gym.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rapier3d": getattr(rapier3d, "__version__", "0.36.1"),
    }


def train_roadster3d_run(
    variant: str,
    seed: int,
    timesteps: int = ROADSTER3D_TIMESTEPS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_interval: int = ROADSTER3D_CHECKPOINT_INTERVAL,
) -> Path:
    if variant not in ROADSTER_VARIANTS:
        raise ValueError(f"variant must be one of {ROADSTER_VARIANTS}.")
    if seed < 0 or timesteps <= 0 or checkpoint_interval <= 0:
        raise ValueError("seed must be nonnegative and step counts must be positive.")
    run_dir = roadster3d_run_dir(output_dir, variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    recording_dir = run_dir / "training_trajectories" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    status_path = run_dir / "training_status.json"
    status: dict[str, Any] = {
        "environment": ENV_ID_3D,
        "simulator_version": SIMULATOR_VERSION_3D,
        "variant": variant,
        "seed": seed,
        "timesteps_requested": timesteps,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "running",
    }

    def save_status() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    save_status()
    env = None
    try:
        env = make_roadster3d_env(
            variant, recording_dir, run_dir / "training_monitor.csv"
        )
        env.reset(seed=seed)
        torch.set_num_threads(1)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            policy_kwargs=ROADSTER3D_PPO["policy_kwargs"],
            learning_rate=ROADSTER3D_PPO["learning_rate"],
            n_steps=ROADSTER3D_PPO["n_steps"],
            batch_size=ROADSTER3D_PPO["batch_size"],
            n_epochs=ROADSTER3D_PPO["n_epochs"],
            gamma=ROADSTER3D_PPO["gamma"],
            gae_lambda=ROADSTER3D_PPO["gae_lambda"],
            clip_range=ROADSTER3D_PPO["clip_range"],
            ent_coef=ROADSTER3D_PPO["ent_coef"],
            device="cpu",
        )
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
        callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_roadster3d",
        )
        model.learn(total_timesteps=timesteps, callback=callback)
        model_path = run_dir / "model.zip"
        model.save(model_path)
        physics = RoadsterPhysics3D.from_config()
        metadata = {
            "project_version": __version__,
            "environment": ENV_ID_3D,
            "simulator_version": SIMULATOR_VERSION_3D,
            "physics_engine": "Rapier3D",
            "variant": variant,
            "variant_label": ROADSTER_LABELS[variant],
            "seed": seed,
            "observation_fields": list(OBSERVATION_FIELDS_3D),
            "action_fields": list(ACTION_FIELDS_3D),
            "physics_config_path": "rocket_landing/configs/roadster_v6.json",
            "physics_config_sha256": hashlib.sha256(PHYSICS_CONFIG_3D_PATH.read_bytes()).hexdigest(),
            "physics_config": physics.to_dict(),
            "ppo_config": ROADSTER3D_PPO,
            "training_protocol": "PPO from scratch; no legacy simulator policy transfer or behavior-cloning warm start.",
            "timesteps_requested": timesteps,
            "timesteps_completed": model.num_timesteps,
            "device": "cpu",
            "versions": runtime_versions_3d(),
            "model_path": "model.zip",
            "training_trajectory_dir": str(recording_dir.relative_to(run_dir)),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status.update({
            "state": "complete",
            "timesteps_completed": model.num_timesteps,
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


def _evaluation_episode(model: PPO, seed: int, variant: str, path: Path) -> dict[str, Any]:
    env = make_roadster3d_env(variant)
    rows: list[dict[str, Any]] = []
    phases: list[str] = []
    actions: list[list[float]] = []
    native_rewards: list[float] = []
    shaped_rewards: list[float] = []
    native_return = shaped_return = 0.0
    try:
        observation, info = env.reset(seed=seed)
        rows.append(dict(info["telemetry"]))
        phases.append(str(info["phase"]))
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(dict(info["telemetry"]))
            phases.append(str(info["phase"]))
            actions.append(np.asarray(info["action_applied"], dtype=np.float32).reshape(7).tolist())
            native = float(info["base_reward"])
            native_return += native
            shaped_return += float(reward)
            native_rewards.append(native)
            shaped_rewards.append(float(reward))
        data = np.asarray(
            [[float(row.get(key, 0.0)) for key in TELEMETRY_FIELDS_3D] for row in rows],
            dtype=np.float32,
        )
        np.savez_compressed(
            path,
            telemetry=data,
            phases=np.asarray(phases, dtype="U16"),
            actions=np.asarray(actions, dtype=np.float32),
            native_rewards=np.asarray(native_rewards, dtype=np.float64),
            shaped_rewards=np.asarray(shaped_rewards, dtype=np.float64),
            outcome=np.asarray(info["outcome"]),
        )
        p = env.unwrapped.physics
        return {
            "eval_seed": seed,
            "outcome": info["outcome"],
            "safe_landing": bool(info["safe_landing"]),
            "crash": bool(info["crash"]),
            "truncated": bool(info["truncated"]),
            "native_return": native_return,
            "shaped_return": shaped_return,
            "episode_length": len(actions),
            "elapsed_seconds": float(info["telemetry"]["elapsed_seconds"]),
            "landing_x_error": info["landing_x_error"],
            "landing_z_error": info["landing_z_error"],
            "touchdown_speed": info["touchdown_speed"],
            "touchdown_tilt": info["touchdown_tilt"],
            "gas_remaining": float(info["gas_remaining"]),
            "pad_half_length_m": p.landing_pad_half_length_m,
            "pad_half_width_m": p.landing_pad_half_width_m,
        }
    finally:
        env.close()


def evaluate_roadster3d_run(
    model_path: Path | str,
    variant: str,
    seed: int,
    episodes: int = ROADSTER3D_EVAL_EPISODES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variant not in ROADSTER_VARIANTS or episodes <= 0:
        raise ValueError("Use a supported reward variant and a positive episode count.")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("The model's run metadata is required for evaluation.")
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("simulator_version") != SIMULATOR_VERSION_3D
            or metadata.get("variant") != variant or metadata.get("seed") != seed
            or metadata.get("physics_config") != RoadsterPhysics3D.from_config().to_dict()):
        raise ValueError("Model metadata does not match the requested v6 experiment.")
    model = PPO.load(model_path, device="cpu")
    run_dir = roadster3d_run_dir(output_dir, variant, seed)
    flight_dir = run_dir / "evaluation_flights"
    flight_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _evaluation_episode(
            model,
            ROADSTER3D_EVAL_SEED_BASE + episode,
            variant,
            flight_dir / f"episode_{episode:04d}.npz",
        )
        for episode in range(episodes)
    ]
    frame = pd.DataFrame(rows)
    frame.insert(0, "episode", np.arange(episodes, dtype=int))
    frame.to_csv(run_dir / "evaluation_episodes.csv", index=False)
    metrics = (
        "native_return", "shaped_return", "safe_landing", "crash", "truncated",
        "landing_x_error", "landing_z_error", "touchdown_speed", "touchdown_tilt",
        "elapsed_seconds", "gas_remaining",
    )
    summary: dict[str, Any] = {
        "environment": ENV_ID_3D,
        "simulator_version": SIMULATOR_VERSION_3D,
        "physics_engine": "Rapier3D",
        "variant": variant,
        "variant_label": ROADSTER_LABELS[variant],
        "seed": seed,
        "episodes": episodes,
        "mean_reward_gap": float((frame["shaped_return"] - frame["native_return"]).mean()),
        "evaluation_seed_base": ROADSTER3D_EVAL_SEED_BASE,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="coerce")
        summary[f"mean_{metric}"] = float(values.mean()) if values.notna().any() else None
        summary[f"std_{metric}"] = (
            float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        )
    summary["safe_landing_rate"] = summary["mean_safe_landing"]
    summary["crash_rate"] = summary["mean_crash"]
    summary["timeout_rate"] = summary["mean_truncated"]
    (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return frame, summary


def run_roadster3d_experiments(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timesteps: int = ROADSTER3D_TIMESTEPS,
    episodes: int = ROADSTER3D_EVAL_EPISODES,
    seeds: tuple[int, ...] = ROADSTER3D_SEEDS,
    variants: tuple[str, ...] = ROADSTER_VARIANTS,
    force: bool = False,
) -> pd.DataFrame:
    root = roadster3d_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if variant not in ROADSTER_VARIANTS:
            raise ValueError(f"Unknown Roadster reward variant: {variant}")
        for seed in seeds:
            run_dir = roadster3d_run_dir(output_dir, variant, seed)
            status_path = run_dir / "training_status.json"
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "evaluation_summary.json"
            evaluation_path = run_dir / "evaluation_episodes.csv"
            model_path = run_dir / "model.zip"
            reusable = all(
                path.is_file()
                for path in (status_path, metadata_path, summary_path, evaluation_path, model_path)
            )
            if reusable:
                status = json.loads(status_path.read_text())
                metadata = json.loads(metadata_path.read_text())
                summary = json.loads(summary_path.read_text())
                reusable = (
                    status.get("state") == "complete"
                    and metadata.get("simulator_version") == SIMULATOR_VERSION_3D
                    and metadata.get("variant") == variant
                    and metadata.get("seed") == seed
                    and metadata.get("timesteps_completed", 0) >= timesteps
                    and metadata.get("ppo_config") == ROADSTER3D_PPO
                    and summary.get("simulator_version") == SIMULATOR_VERSION_3D
                    and summary.get("episodes", 0) >= episodes
                    and len(pd.read_csv(evaluation_path)) >= episodes
                )
            if reusable and not force:
                print(f"Reusing complete v6 {variant} seed {seed} run.")
                continue
            model_path = train_roadster3d_run(variant, seed, timesteps, output_dir)
            evaluate_roadster3d_run(model_path, variant, seed, episodes, output_dir)
    return write_roadster3d_comparison(output_dir)


def write_roadster3d_comparison(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    root = roadster3d_root(output_dir)
    summaries: list[dict[str, Any]] = []
    for variant in ROADSTER_VARIANTS:
        for summary_path in sorted((root / variant).glob("seed_*/evaluation_summary.json")):
            metadata_path = summary_path.parent / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            if (
                metadata.get("simulator_version") == SIMULATOR_VERSION_3D
                and metadata.get("physics_config") == RoadsterPhysics3D.from_config().to_dict()
                and metadata.get("ppo_config") == ROADSTER3D_PPO
                and summary.get("simulator_version") == SIMULATOR_VERSION_3D
            ):
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
        mean_landing_z_error=("mean_landing_z_error", "mean"),
        mean_reward_gap=("mean_reward_gap", "mean"),
    )
    aggregate.to_csv(root / "comparison_summary.csv", index=False)
    return aggregate
