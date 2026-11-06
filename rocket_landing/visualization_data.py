"""Prepare saved training and evaluation data for visual exploration."""

from pathlib import Path

import pandas as pd


def read_training_episodes(run_dir: Path) -> pd.DataFrame:
    """Read every completed training episode recorded by SB3's Monitor."""
    path = run_dir / "training_monitor.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path, comment="#")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    if not {"r", "l"}.issubset(data.columns):
        return pd.DataFrame()
    data = data.rename(columns={"r": "Reward", "l": "Length"})
    data["Reward"] = pd.to_numeric(data["Reward"], errors="coerce")
    data["Length"] = pd.to_numeric(data["Length"], errors="coerce")
    data = data.dropna(subset=["Reward", "Length"]).reset_index(drop=True)
    if data.empty:
        return data
    data["Episode"] = range(1, len(data) + 1)
    data["Timesteps"] = data["Length"].cumsum()
    data["Average reward (100 episodes)"] = data["Reward"].rolling(100, min_periods=1).mean()
    return data


def read_training_progress(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "sb3_logs" / "progress.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def evaluation_outcomes(data: pd.DataFrame) -> pd.Series:
    """Label outcomes without treating time limits as crashes."""
    def outcome(row):
        if bool(row["truncated"]):
            return "Timeout"
        if bool(row["safe_landing"]):
            return "Safe landing"
        return "Crash" if bool(row["crash"]) else "Unknown"

    return data.apply(outcome, axis=1)
