"""Explore training, individual flights and learned policies locally."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rocket_landing.config import EVAL_SEED_BASE, VARIANTS, VARIANT_LABELS, resolve_output_dir
from rocket_landing.experiment import evaluate_run
from rocket_landing.replay import flight_figure, load_training_episode, replay_episode
from rocket_landing.visualization_data import evaluation_outcomes, read_training_episodes, read_training_progress

OUTPUTS = resolve_output_dir()
ACTION_NAMES = {-1: "Start", 0: "No engine", 1: "Left engine", 2: "Main engine", 3: "Right engine"}


def _available_runs() -> list[tuple[str, int, Path]]:
    runs = []
    if not OUTPUTS.exists():
        return runs
    for variant in VARIANTS:
        variant_dir = OUTPUTS / variant
        if not variant_dir.exists():
            continue
        for run_dir in sorted(variant_dir.glob("seed_*")):
            model = run_dir / "model.zip"
            try:
                seed = int(run_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            if model.is_file() or (run_dir / "training_monitor.csv").is_file():
                runs.append((variant, seed, model))
    return runs


@st.cache_data(ttl=5, max_entries=24, show_spinner=False)
def _training_data(path: str, monitor_mtime: float, progress_mtime: float):
    run_dir = Path(path)
    return read_training_episodes(run_dir), read_training_progress(run_dir)


@st.cache_data(max_entries=16, show_spinner=False)
def _replay_model(path: str, modified: float, variant: str, seed: int):
    return replay_episode(path, variant, seed)


def _training_view(run_dir: Path):
    st.header("Training progress")
    monitor = run_dir / "training_monitor.csv"
    progress = run_dir / "sb3_logs" / "progress.csv"
    episodes, updates = _training_data(
        str(run_dir), monitor.stat().st_mtime if monitor.exists() else 0,
        progress.stat().st_mtime if progress.exists() else 0,
    )
    if episodes.empty:
        st.info("No completed training episodes are available for this run yet.")
        return

    left, middle, right = st.columns(3)
    left.metric("Training episodes", f"{len(episodes):,}")
    middle.metric("Logged timesteps", f"{int(episodes['Timesteps'].iloc[-1]):,}")
    right.metric("Last 100 episode reward", f"{episodes['Reward'].tail(100).mean():.1f}")
    st.caption("Training reward includes the selected reward rule. Scores across different reward rules are not directly comparable.")
    st.subheader("Reward across training attempts")
    st.line_chart(
        episodes.set_index("Episode")[["Reward", "Average reward (100 episodes)"]],
        x_label="Training episode", y_label="Training reward",
        alt="Training reward for every completed episode and its 100-episode moving average",
    )
    st.subheader("Episode length")
    st.line_chart(episodes.set_index("Episode")[["Length"]], x_label="Training episode", y_label="Steps",
                  alt="Number of steps in each training episode")

    if {"outcome", "native_return"}.issubset(episodes.columns):
        st.subheader("Training outcomes")
        counts = episodes["outcome"].value_counts().rename_axis("Outcome").reset_index(name="Episodes")
        st.bar_chart(counts, x="Outcome", y="Episodes", alt="Training episodes by landing, crash, and timeout")
        st.line_chart(episodes.set_index("Episode")[["Reward", "native_return"]],
                      alt="Training reward compared with original environment return")
    else:
        st.info("This older training run saved episode rewards and lengths, but did not record individual training flight states or outcomes. New training runs record them.")

    if not updates.empty and "time/total_timesteps" in updates:
        available = {"Policy loss": "train/policy_gradient_loss", "Value loss": "train/value_loss",
                     "Policy change (KL)": "train/approx_kl", "Explained variance": "train/explained_variance"}
        cols = [v for v in available.values() if v in updates and updates[v].notna().any()]
        if cols:
            st.subheader("PPO learning signals")
            names = {v: k for k, v in available.items()}
            chosen = st.selectbox("Learning signal", cols, format_func=lambda c: names[c])
            chart_data = updates[["time/total_timesteps", chosen]].dropna().rename(
                columns={"time/total_timesteps": "Timesteps", chosen: names[chosen]})
            st.line_chart(chart_data.set_index("Timesteps"), alt=f"{names[chosen]} throughout training")
    with st.expander("View all training episodes"):
        columns = [c for c in ("Episode", "Timesteps", "Reward", "Length", "outcome", "native_return") if c in episodes]
        st.dataframe(episodes[columns], hide_index=True, alt="Training episodes and recorded outcomes")


@st.fragment(run_every=5)
def _live_training_view(run_dir: Path):
    try:
        current_status = json.loads((run_dir / "training_status.json").read_text()).get("state")
        if current_status != "running":
            st.rerun()
    except (OSError, json.JSONDecodeError):
        pass
    st.caption("Training charts refresh every five seconds while this page is open.")
    _training_view(run_dir)


def _flight_view(run_dir: Path, variant: str, model_path: Path, episodes: pd.DataFrame):
    st.header("Explore flights in 3D")
    st.caption("The rocket is a 3D illustration driven by the real 2D LunarLander state. Depth and terrain are illustrative; the simulator's physics remains 2D.")
    st.button("Refresh available attempts", help="Find attempts recorded since the page last loaded.")
    recordings_root = run_dir / "training_trajectories"
    recording_groups = sorted(p for p in recordings_root.glob("*") if p.is_dir() and any(p.glob("episode_*.npz"))) if recordings_root.exists() else []
    checkpoints = sorted((run_dir / "checkpoints").glob("ppo_lunar_lander_*_steps.zip"),
                         key=lambda p: int(p.stem.rsplit("_", 2)[1]))
    modes = []
    if not episodes.empty:
        modes.append("Evaluation attempts")
    if recording_groups:
        modes.append("Recorded training attempts")
    if checkpoints:
        modes.append("Training checkpoints")
    if not modes:
        st.info("Evaluate this policy or train a new model to explore flights.")
        return
    mode = st.selectbox("Flight source", modes)
    selection = None
    if mode == "Evaluation attempts":
        data = episodes.copy()
        data["Outcome"] = evaluation_outcomes(data)
        outcome = st.selectbox("Show attempts", ["All", "Safe landing", "Crash", "Timeout"])
        filtered = data if outcome == "All" else data[data["Outcome"] == outcome]
        if filtered.empty:
            st.info("There are no evaluation attempts with this outcome.")
            return
        rows = {int(row["episode"]): row for _, row in filtered.iterrows()}
        chosen = st.selectbox("Evaluation attempt", list(rows),
                              format_func=lambda i: f"Attempt {i + 1} · {rows[i]['Outcome']} · return {rows[i]['environment_return']:.1f}")
        episode_seed = int(rows[chosen]["eval_seed"])
        selection = (str(model_path), model_path.stat().st_mtime, variant, episode_seed)
        st.caption(f"Replays the saved policy deterministically with evaluation seed {episode_seed}.")
    elif mode == "Training checkpoints":
        checkpoint = st.selectbox("Policy checkpoint", checkpoints,
                                  format_func=lambda p: f"{int(p.stem.rsplit('_', 2)[1]):,} training steps")
        episode_seed = st.number_input("Replay seed", min_value=0, max_value=2_147_483_647, value=EVAL_SEED_BASE, step=1)
        selection = (str(checkpoint), checkpoint.stat().st_mtime, variant, int(episode_seed))
        st.caption("Runs a fresh episode with the selected checkpoint. It is not a recording of a past training attempt.")
    else:
        group = st.selectbox("Training session", recording_groups, format_func=lambda p: p.name)
        recordings = sorted(group.glob("episode_*.npz"))
        index_file = group / "episodes.csv"
        try:
            index = pd.read_csv(index_file) if index_file.exists() else pd.DataFrame()
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            index = pd.DataFrame()
        outcome_by_episode = dict(zip(index["episode"], index["outcome"])) if {"episode", "outcome"}.issubset(index.columns) else {}
        filter_outcome = st.selectbox("Show training attempts", ["All", "Safe landing", "Crash", "Timeout"])
        if filter_outcome != "All":
            recordings = [p for p in recordings if outcome_by_episode.get(int(p.stem.split("_")[-1])) == filter_outcome]
        if not recordings:
            st.info("There are no recorded training attempts with this outcome.")
            return
        labels = {}
        for path in recordings:
            number = int(path.stem.split("_")[-1])
            label = f"Training attempt {number:,}"
            if number in outcome_by_episode:
                label += f" · {outcome_by_episode[number]}"
            labels[path] = label
        recording = st.selectbox("Recorded training attempt", recordings, format_func=lambda p: labels[p])
        selection = (str(recording), recording.stat().st_mtime, "recorded")
        st.caption("This is the exact state and actions captured during training.")

    if st.button("Load 3D flight", type="primary"):
        try:
            with st.spinner("Preparing the flight…"):
                if mode == "Recorded training attempts":
                    flight, pad = load_training_episode(selection[0])
                else:
                    flight, pad = _replay_model(*selection)
                st.session_state["flight_result"] = (selection, flight, pad)
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"Could not load the flight: {exc}")
            return

    result = st.session_state.get("flight_result")
    if not result or result[0] != selection:
        st.info("Choose an attempt and select **Load 3D flight** to explore it.")
        return
    _, flight, pad = result
    a, b, c = st.columns(3)
    a.metric("Outcome", flight.attrs.get("outcome", "Unknown"))
    b.metric("Steps", f"{len(flight) - 1:,}")
    c.metric("Native return", f"{flight['native_total'].iloc[-1]:.1f}")
    st.plotly_chart(flight_figure(flight, pad), width="stretch", theme=None,
                    alt="Animated 3D illustration of the lander's position, tilt, and engine actions")
    st.caption("Use Play and the timeline under the scene. Drag to rotate the camera and scroll to zoom.")
    st.subheader("What happened during this flight")
    st.line_chart(flight.set_index("step")[["native_total", "shaped_total"]], x_label="Step", y_label="Cumulative reward",
                  alt="Original and shaped reward accumulated during the flight")
    st.line_chart(flight.set_index("step")[["vx", "vy"]], x_label="Step", y_label="Normalized velocity",
                  alt="Horizontal and vertical velocity throughout the flight")
    st.line_chart(flight.set_index("step")[["angle", "spin"]], x_label="Step", y_label="Tilt and spin",
                  alt="Rocket tilt and angular velocity throughout the flight")
    actions = flight.loc[flight["action"] >= 0, "action"].map(ACTION_NAMES).value_counts().rename_axis("Action").reset_index(name="Steps")
    st.bar_chart(actions, x="Action", y="Steps", alt="Engine actions chosen during the flight")


def _evaluation_view(run_dir: Path, variant: str, seed: int, model_path: Path, episodes: pd.DataFrame):
    st.header("Evaluation results")
    if not model_path.is_file():
        st.info("A finished model is needed for evaluation. You can explore training logs and saved checkpoints while training continues.")
        return
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists() and not episodes.empty:
        summary = json.loads(summary_path.read_text())
        left, middle, right, extra = st.columns(4)
        left.metric("Mean native return", f"{summary['mean_return']:.1f}", f"± {summary['std_return']:.1f} episode SD")
        middle.metric("Safe landings", f"{summary['safe_landing_rate']:.0%}")
        right.metric("Crashes / timeouts", f"{summary['crash_rate']:.0%} / {summary['truncation_rate']:.0%}")
        extra.metric("Shaped minus native", f"{summary['mean_reward_gap']:.2f}")
        st.line_chart(episodes.set_index("episode")[["environment_return", "shaped_return"]],
                      x_label="Evaluation episode", y_label="Return", alt="Native and shaped return for every evaluation attempt")
        displayed = episodes.copy()
        displayed["Outcome"] = evaluation_outcomes(displayed)
        counts = displayed["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Episodes")
        st.bar_chart(counts, x="Outcome", y="Episodes", alt="Evaluation outcomes by landing, crash, and timeout")
        st.dataframe(displayed, hide_index=True, alt="All evaluation attempts with outcomes and rewards")
    else:
        st.info("This model has not been evaluated yet.")

    comparison = OUTPUTS / "comparison_summary.csv"
    partial = OUTPUTS / "comparison_per_run.csv"
    if comparison.exists():
        st.subheader("Comparison across training seeds")
        st.dataframe(pd.read_csv(comparison), hide_index=True, alt="Aggregate results by reward variant")
    elif partial.exists():
        st.subheader("Completed runs so far")
        st.caption("The full comparison appears when all requested runs finish.")
        st.dataframe(pd.read_csv(partial), hide_index=True, alt="Results from completed training runs")

    st.subheader("Run a new evaluation")
    count = st.slider("Evaluation episodes", min_value=1, max_value=20, value=1)
    if st.button("Run evaluation"):
        with st.spinner("Evaluating the saved policy…"):
            _, result = evaluate_run(model_path, variant, seed, episodes=count,
                                     output_dir=OUTPUTS / "demo_evaluations", capture_video=True)
        st.success(f"Mean native return across {count} episode(s): {result['mean_return']:.1f}")


def main() -> None:
    st.set_page_config(page_title="Rocket Landing RL", page_icon="🚀", layout="wide")
    st.title("🚀 Rocket Landing with Reward Shaping")
    st.caption("Explore training, individual flights and the effects of reward design.")
    runs = _available_runs()
    if not runs:
        st.info(f"No trained models were found under `{OUTPUTS}`.", title="Train the experiment first")
        st.markdown("From the project directory, run the full experiment:")
        st.code("python -m rocket_landing.cli run-all", language="bash")
        st.caption("For a quick preview, train one policy. Set `ROCKET_LANDING_OUTPUT_DIR` in both processes to use another output directory.")
        st.code("python -m rocket_landing.cli train --variant baseline --seed 0 --timesteps 10000\n"
                "python -m rocket_landing.cli evaluate --model outputs/baseline/seed_0/model.zip --variant baseline --seed 0 --episodes 10", language="bash")
        return
    options = {f"{VARIANT_LABELS[variant]} · seed {seed}" + (" · training log" if not model.is_file() else ""): (variant, seed, model) for variant, seed, model in runs}
    chosen = st.selectbox("Choose a trained policy", list(options))
    variant, seed, model_path = options[chosen]
    run_dir = model_path.parent
    status_file = run_dir / "training_status.json"
    try:
        status = json.loads(status_file.read_text()).get("state") if status_file.exists() else None
    except (OSError, json.JSONDecodeError):
        status = None
    if status == "running":
        st.caption("Training is running. New completed attempts become available as they are recorded.")
        if model_path.is_file():
            st.caption("The saved policy belongs to the previous completed run until this run finishes.")
    elif status in ("interrupted", "failed"):
        st.caption(f"The most recent training run was {status}; completed attempts and checkpoints remain available.")
    episode_file = run_dir / "evaluation_episodes.csv"
    summary_file = run_dir / "evaluation_summary.json"
    current_evaluation = (
        model_path.is_file() and episode_file.exists() and summary_file.exists()
        and episode_file.stat().st_mtime >= model_path.stat().st_mtime
        and summary_file.stat().st_mtime >= model_path.stat().st_mtime
    )
    episodes = pd.read_csv(episode_file) if current_evaluation else pd.DataFrame()
    page = st.segmented_control("Explore", ["Training", "Flights & 3D", "Evaluation"], default="Training")
    if page == "Training":
        if status == "running":
            _live_training_view(run_dir)
        else:
            _training_view(run_dir)
    elif page == "Flights & 3D":
        _flight_view(run_dir, variant, model_path, episodes)
    elif page == "Evaluation":
        _evaluation_view(run_dir, variant, seed, model_path, episodes)


if __name__ == "__main__":
    main()
