"""Shared experiment settings and reward definitions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENV_ID = "LunarLander-v3"
VARIANTS = ("baseline", "flawed_bonus", "potential_shaping")
VARIANT_LABELS = {
    "baseline": "Default reward",
    "flawed_bonus": "Upright + slow bonus",
    "potential_shaping": "Potential progress + time cost",
}
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_EVAL_EPISODES = 100
EVAL_SEED_BASE = 10_000
CHECKPOINT_INTERVAL = 100_000
VIDEO_FRAME_STRIDE = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR_ENV = "ROCKET_LANDING_OUTPUT_DIR"


def resolve_output_dir(output_dir: Path | str | None = None) -> Path:
    """Resolve artifact paths consistently for CLI and Streamlit.

    Relative paths are interpreted from the project root so launch location does
    not decide where models are written or where the demo looks for them.
    """
    configured = output_dir if output_dir is not None else os.getenv(OUTPUT_DIR_ENV)
    path = Path(configured).expanduser() if configured is not None else PROJECT_ROOT / "outputs"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DEFAULT_OUTPUT_DIR = resolve_output_dir()


@dataclass(frozen=True)
class RewardConfig:
    """Explicit reward constants, serialized into every run's metadata."""

    upright_angle_threshold: float = 0.1
    slow_speed_threshold: float = 0.1
    flawed_bonus_per_step: float = 1.0
    shaping_scale: float = 0.5
    time_cost_per_step: float = 0.01
    shaping_gamma: float = 0.99
    potential_horizontal_distance_weight: float = 0.5
    potential_height_weight: float = 0.1
    potential_horizontal_speed_weight: float = 0.25
    potential_vertical_speed_weight: float = 0.25
    potential_tilt_weight: float = 0.25
    potential_spin_weight: float = 0.1
    potential_leg_contact_bonus: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REWARD_CONFIG = RewardConfig()


def ppo_config() -> dict[str, Any]:
    """Return the PPO configuration used for every variant."""
    return {
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": [64, 64]},
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "device": "cpu",
    }
