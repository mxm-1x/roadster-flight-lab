"""Experiment orchestration, model training, and evaluation."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from rocket_landing import __version__
from rocket_landing.config import (
    CHECKPOINT_INTERVAL,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_SEEDS,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    EVAL_SEED_BASE,
    DEFAULT_OUTPUT_DIR,
    REWARD_CONFIG,
    VARIANTS,
    VIDEO_FRAME_STRIDE,
    ppo_config,
    resolve_output_dir,
)
from rocket_landing.rewards import apply_reward_variant
from rocket_landing.recording import TrainingRecorder


def make_env(variant: str = "baseline", render_mode: str | None = None, monitor_path=None, trajectory_dir=None):
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = apply_reward_variant(env, variant)
    if trajectory_dir is not None:
        env = TrainingRecorder(env, Path(trajectory_dir))
    if monitor_path is not None:
        env = Monitor(env, str(monitor_path), info_keywords=("outcome", "native_return") if trajectory_dir else ())
    return env


def _runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gymnasium": gym.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _run_dir(output_dir: Path, variant: str, seed: int) -> Path:
    return output_dir / variant / f"seed_{seed}"


def train_run(
    variant: str,
    seed: int,
    timesteps: int = DEFAULT_TIMESTEPS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}.")
    if timesteps <= 0 or checkpoint_interval <= 0:
        raise ValueError("timesteps and checkpoint_interval must be positive.")

    run_dir = _run_dir(resolve_output_dir(output_dir), variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = run_dir / "training_monitor.csv"
    recording_dir = run_dir / "training_trajectories" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    status_file = run_dir / "training_status.json"
    status = {"variant": variant, "seed": seed, "timesteps_requested": timesteps,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "state": "running"}

    def save_status():
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_file)

    save_status()
    env = None
    try:
        env = make_env(variant, monitor_path=monitor_file, trajectory_dir=recording_dir)
        env.reset(seed=seed)
        torch.set_num_threads(1)
        model = PPO(
            "MlpPolicy", env, seed=seed, verbose=1, policy_kwargs={"net_arch": [64, 64]},
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99,
            gae_lambda=0.98, clip_range=0.2, ent_coef=0.01, device="cpu",
        )
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
        callback = CheckpointCallback(
            save_freq=max(checkpoint_interval, 1), save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_lunar_lander", save_replay_buffer=False, save_vecnormalize=False,
        )
        model.learn(total_timesteps=timesteps, callback=callback)
        model_path = run_dir / "model.zip"
        model.save(model_path)
        metadata = {
            "project_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": ENV_ID, "variant": variant, "seed": seed,
            "timesteps_requested": timesteps, "timesteps_completed": model.num_timesteps,
            "reward_config": REWARD_CONFIG.to_dict(), "ppo_config": ppo_config(), "device": "cpu",
            "versions": _runtime_versions(), "model_path": "model.zip",
            "training_trajectory_dir": str(recording_dir.relative_to(run_dir)),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status["state"] = "complete"
        status["timesteps_completed"] = model.num_timesteps
        save_status()
        return model_path
    except BaseException as exc:
        status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        save_status()
        raise
    finally:
        if env is not None:
            env.close()


def _classify_episode(
    last_step_reward: float,
    terminated: bool,
    truncated: bool,
) -> tuple[bool, bool, bool]:
    """Return (safe_landing, crash, truncation) using LunarLander terminal reward."""
    safe_landing = bool(terminated and not truncated and last_step_reward > 0)
    crash = bool(terminated and not safe_landing)
    return safe_landing, crash, bool(truncated)


def evaluate_run(
    model_path: Path | str,
    variant: str,
    seed: int,
    episodes: int = DEFAULT_EVAL_EPISODES,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    capture_video: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(model_path, device="cpu")
    # Reward wrappers are diagnostic only here: they do not affect observations,
    # transitions, or chosen actions. Their info fields preserve the native reward.
    env = make_env(variant, render_mode="rgb_array" if capture_video else None)
    run_dir = _run_dir(resolve_output_dir(output_dir), variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    representative_frames: list[np.ndarray] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=EVAL_SEED_BASE + episode)
        terminated = truncated = False
        env_return = 0.0
        shaped_return = 0.0
        length = 0
        last_step_reward = 0.0
        frames = []
        record_episode = capture_video and episode == 0
        while not (terminated or truncated):
            if record_episode and length % VIDEO_FRAME_STRIDE == 0:
                frames.append(env.render()[::2, ::2])
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            native_reward = float(info.get("base_reward", reward))
            env_return += native_reward
            shaped_return += float(reward)
            last_step_reward = native_reward
            length += 1
        if record_episode and (length % VIDEO_FRAME_STRIDE != 0 or not frames):
            frames.append(env.render()[::2, ::2])
        safe_landing, crash, was_truncated = _classify_episode(last_step_reward, terminated, truncated)
        rows.append(
            {
                "episode": episode,
                "eval_seed": EVAL_SEED_BASE + episode,
                "environment_return": env_return,
                "shaped_return": shaped_return,
                "episode_length": length,
                "safe_landing": safe_landing,
                "crash": crash,
                "truncated": was_truncated,
            }
        )
        if record_episode:
            representative_frames = frames
    env.close()

    # Native reward is recovered from wrapper diagnostics; the policy still receives
    # the same observations and transitions as it would in the original environment.
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "evaluation_episodes.csv", index=False)
    if representative_frames:
        iio.imwrite(
            run_dir / "evaluation_episode.gif",
            representative_frames,
            duration=100,
            loop=0,
        )
    summary = {
        "variant": variant,
        "seed": seed,
        "episodes": episodes,
        "mean_return": float(frame["environment_return"].mean()),
        "std_return": float(frame["environment_return"].std(ddof=1)) if episodes > 1 else 0.0,
        "safe_landing_rate": float(frame["safe_landing"].mean()),
        "crash_rate": float(frame["crash"].mean()),
        "truncation_rate": float(frame["truncated"].mean()),
        "mean_episode_length": float(frame["episode_length"].mean()),
        "mean_environment_return": float(frame["environment_return"].mean()),
        "mean_shaped_return": float(frame["shaped_return"].mean()),
        "mean_reward_gap": float((frame["shaped_return"] - frame["environment_return"]).mean()),
        "solved_200_mean_return": bool(frame["environment_return"].mean() >= 200.0),
        "model_path": str(model_path),
        "video_path": "evaluation_episode.gif" if capture_video else None,
    }
    (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return frame, summary


def run_experiments(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    timesteps: int = DEFAULT_TIMESTEPS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    output_dir = resolve_output_dir(output_dir)
    summaries = []
    for variant in VARIANTS:
        for seed in seeds:
            run_dir = _run_dir(output_dir, variant, seed)
            model_path = run_dir / "model.zip"
            metadata_path = run_dir / "metadata.json"
            model_is_current = False
            if model_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                model_is_current = (
                    metadata.get("variant") == variant
                    and metadata.get("seed") == seed
                    and metadata.get("timesteps_requested") == timesteps
                    and metadata.get("reward_config") == REWARD_CONFIG.to_dict()
                    and metadata.get("ppo_config") == ppo_config()
                )
            if not model_is_current:
                model_path = train_run(variant, seed, timesteps, output_dir)

            summary_path = run_dir / "evaluation_summary.json"
            evaluation_is_current = False
            if (
                model_is_current
                and summary_path.is_file()
                and (run_dir / "evaluation_episodes.csv").is_file()
                and (run_dir / "evaluation_episode.gif").is_file()
                and summary_path.stat().st_mtime >= model_path.stat().st_mtime
                and (run_dir / "evaluation_episodes.csv").stat().st_mtime >= model_path.stat().st_mtime
                and (run_dir / "evaluation_episode.gif").stat().st_mtime >= model_path.stat().st_mtime
            ):
                saved_summary = json.loads(summary_path.read_text())
                evaluation_is_current = saved_summary.get("episodes") == episodes
            if evaluation_is_current:
                summary = saved_summary
            else:
                _, summary = evaluate_run(
                    model_path,
                    variant,
                    seed,
                    episodes,
                    output_dir,
                    capture_video=True,
                )
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(output_dir / "comparison_per_run.csv", index=False)
    per_run = pd.DataFrame(summaries)
    aggregate = (
        per_run.groupby("variant", as_index=False)
        .agg(
            training_seeds=("seed", "count"),
            mean_return=("mean_return", "mean"),
            between_seed_std=("mean_return", "std"),
            safe_landing_rate=("safe_landing_rate", "mean"),
            crash_rate=("crash_rate", "mean"),
            truncation_rate=("truncation_rate", "mean"),
            mean_episode_length=("mean_episode_length", "mean"),
            mean_shaped_return=("mean_shaped_return", "mean"),
            mean_reward_gap=("mean_reward_gap", "mean"),
        )
    )
    aggregate["between_seed_std"] = aggregate["between_seed_std"].fillna(0.0)
    aggregate.to_csv(output_dir / "comparison_summary.csv", index=False)
    return aggregate
