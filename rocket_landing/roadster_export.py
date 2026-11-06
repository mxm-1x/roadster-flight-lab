"""Export Roadster experiment artifacts for a static, inference-free website."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rocket_landing.config import DEFAULT_OUTPUT_DIR
from rocket_landing.envs.roadster import ACTION_FIELDS, OBSERVATION_FIELDS, SIMULATOR_VERSION, RoadsterPhysics
from rocket_landing.roadster_experiment import (
    ROADSTER_LABELS,
    ROADSTER_BEHAVIOR_CLONING,
    ROADSTER_PPO,
    ROADSTER_TELEMETRY_NUMERIC,
    evaluate_rule_controller,
    export_rule_based_preview,
    roadster_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC = PROJECT_ROOT / "frontend" / "public"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def compact_episode(path: Path, *, kind: str, variant: str, seed: int, episode: int | None = None) -> dict[str, Any]:
    """Convert a saved NPZ episode into a compact browser replay."""
    with np.load(path, allow_pickle=False) as archive:
        telemetry = archive["telemetry"].astype(np.float64)
        phases = archive["phases"].astype(str).tolist()
        actions = archive["actions"].astype(np.float64)
        native_rewards = archive["native_rewards"].astype(np.float64)
        shaped_rewards = archive["shaped_rewards"].astype(np.float64)
        outcome = str(archive["outcome"].item())
    if telemetry.ndim != 2 or telemetry.shape[1] != len(ROADSTER_TELEMETRY_NUMERIC):
        raise ValueError(f"Unsupported Roadster trajectory shape in {path}.")
    steps = telemetry.shape[0] - 1
    if len(phases) != steps + 1 or actions.shape != (steps, len(ACTION_FIELDS)):
        raise ValueError(f"Roadster state/action lengths do not line up in {path}.")
    if len(native_rewards) != steps or len(shaped_rewards) != steps:
        raise ValueError(f"Roadster reward arrays do not line up in {path}.")

    # Action and cumulative reward at row i describe the transition into state i.
    action_rows = np.vstack([np.zeros((1, len(ACTION_FIELDS))), actions])
    native_total = np.r_[0.0, np.cumsum(native_rewards)]
    shaped_total = np.r_[0.0, np.cumsum(shaped_rewards)]
    rows = np.column_stack([telemetry, action_rows, native_total, shaped_total])
    center_index = ROADSTER_TELEMETRY_NUMERIC.index("landing_center_x")
    ramp_index = ROADSTER_TELEMETRY_NUMERIC.index("ramp_radius")
    center = float(telemetry[-1, center_index])
    radius = float(telemetry[0, ramp_index])
    p = RoadsterPhysics.from_config()
    final = telemetry[-1]
    vx = float(final[ROADSTER_TELEMETRY_NUMERIC.index("vx")])
    vz = float(final[ROADSTER_TELEMETRY_NUMERIC.index("vz")])
    angle = float(final[ROADSTER_TELEMETRY_NUMERIC.index("heading")])
    z = float(final[ROADSTER_TELEMETRY_NUMERIC.index("z")])
    x = float(final[ROADSTER_TELEMETRY_NUMERIC.index("x")])
    has_touchdown = outcome in ("Crash", "Safe landing") and abs(z - p.car_center_height_m) < 0.05
    metrics = {
        "landing_x_error": abs(x - center) if has_touchdown else None,
        "touchdown_speed": float(np.hypot(vx, vz)) if has_touchdown else None,
        "touchdown_angle_error": abs(angle) if has_touchdown else None,
        "gas_remaining": float(final[ROADSTER_TELEMETRY_NUMERIC.index("gas")]),
        "flight_seconds": float(final[ROADSTER_TELEMETRY_NUMERIC.index("flight_seconds")]),
        "landing_zone_x": [center - p.landing_pad_half_width_m, center + p.landing_pad_half_width_m],
        "final_x": x,
        "final_z": z,
    }
    source = {"kind": kind, "variant": variant, "seed": seed, "episode": episode,
             "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "schema": 1, "simulator_version": SIMULATOR_VERSION,
        "columns": [*ROADSTER_TELEMETRY_NUMERIC, *ACTION_FIELDS, "native_total", "shaped_total"],
        "rows": rows.round(6).tolist(), "phases": phases,
        "ramp_radius_m": radius,
        "landing_zone_x": metrics["landing_zone_x"],
        "outcome": outcome,
        "native_return": float(native_total[-1]),
        "shaped_return": float(shaped_total[-1]),
        "steps": steps, "metrics": metrics, "source": source,
        "ramp_controller": "scripted_fixture_v1",
    }


def _run_index(root: Path, public: Path, variant: str, seed: int) -> dict[str, Any] | None:
    run_dir = root / variant / f"seed_{seed}"
    model_path = run_dir / "model.zip"
    summary_path = run_dir / "evaluation_summary.json"
    episodes_path = run_dir / "evaluation_episodes.csv"
    metadata_path = run_dir / "metadata.json"
    if not all(path.is_file() for path in (model_path, summary_path, episodes_path, metadata_path)):
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("simulator_version") != SIMULATOR_VERSION:
        return None
    if metadata.get("physics_config") != RoadsterPhysics.from_config().to_dict():
        return None
    if metadata.get("ppo_config") != ROADSTER_PPO:
        return None
    if not all(metadata.get("behavior_cloning", {}).get(key) == value for key, value in ROADSTER_BEHAVIOR_CLONING.items()):
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("simulator_version") != SIMULATOR_VERSION:
        return None
    cloning_path = run_dir / "behavior_cloning.json"
    behavior_cloning = json.loads(cloning_path.read_text()) if cloning_path.is_file() else metadata.get("behavior_cloning", {})
    evaluations = pd.read_csv(episodes_path).replace([np.inf, -np.inf], np.nan)
    curve_url = None
    progress_path = run_dir / "sb3_logs" / "progress.csv"
    if progress_path.is_file():
        progress = pd.read_csv(progress_path).replace([np.inf, -np.inf], np.nan)
        curve_columns = {
            "timesteps": "time/total_timesteps",
            "mean_episode_return": "rollout/ep_rew_mean",
            "mean_episode_length": "rollout/ep_len_mean",
        }
        if all(column in progress for column in curve_columns.values()):
            curve_rows = []
            for source_row in progress.to_dict("records"):
                curve_rows.append({
                    key: None if pd.isna(source_row[column]) else float(source_row[column])
                    for key, column in curve_columns.items()
                })
            relative_curve = Path("data") / "roadster" / "curves" / variant / str(seed) / "training.json"
            write_json(public / relative_curve, {"columns": list(curve_columns), "rows": curve_rows})
            curve_url = "/" + relative_curve.as_posix()
    flight_paths = []
    for episode_index, row in enumerate(evaluations.to_dict("records")):
        row = {
            key: None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            for key, value in row.items()
        }
        episode = int(row.get("episode", episode_index))
        source = run_dir / "evaluation_flights" / f"episode_{episode:04d}.npz"
        if not source.is_file():
            continue
        relative = Path("data") / "roadster" / "flights" / variant / str(seed) / f"evaluation_{episode:04d}.json"
        replay = compact_episode(source, kind="evaluation", variant=variant, seed=seed, episode=episode)
        write_json(public / relative, replay)
        flight_paths.append({
            **row,
            "episode": episode,
            "flight_url": "/" + relative.as_posix(),
            "outcome": str(row["outcome"]),
            "safe_landing": bool(row["safe_landing"]),
            "crash": bool(row["crash"]),
            "truncated": bool(row["truncated"]),
        })

    training_paths = []
    training_bundle_url = None
    training_dir_value = metadata.get("training_trajectory_dir")
    training_index = run_dir / training_dir_value / "episodes.csv" if training_dir_value else None
    if training_index and training_index.is_file():
        try:
            history = pd.read_csv(training_index)
        except (OSError, pd.errors.EmptyDataError):
            history = pd.DataFrame()
        session = training_index.parent.name
        relative_bundle = Path("data") / "roadster" / "training_bundles" / variant / str(seed) / f"{session}.bin"
        bundle_path = public / relative_bundle
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bundle_path.with_suffix(".tmp")
        bundle_index = 0
        with temporary.open("wb") as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="wb", compresslevel=6, mtime=0) as bundle:
                bundle.write(b'{"flights":[')
                for episode in history.get("episode", pd.Series(dtype=int)).astype(int).tolist():
                    source = training_index.parent / f"episode_{episode:07d}.npz"
                    if not source.is_file():
                        continue
                    replay = compact_episode(source, kind="training", variant=variant, seed=seed, episode=episode)
                    if bundle_index:
                        bundle.write(b",")
                    bundle.write(json.dumps(replay, allow_nan=False, separators=(",", ":")).encode())
                    training_paths.append({
                        "episode": episode, "outcome": replay["outcome"],
                        "bundle_index": bundle_index,
                        "native_return": replay["native_return"],
                        "shaped_return": replay["shaped_return"],
                    })
                    bundle_index += 1
                bundle.write(b"]}")
        if bundle_index:
            temporary.replace(bundle_path)
            training_bundle_url = "/" + relative_bundle.as_posix()
        else:
            temporary.unlink(missing_ok=True)

    return {
        "variant": variant, "label": ROADSTER_LABELS[variant], "seed": seed,
        "summary": summary, "evaluations": flight_paths,
        "training_flights": training_paths,
        "training_bundle_url": training_bundle_url,
        "timesteps_completed": metadata.get("timesteps_completed"),
        "training_episodes": int(len(pd.read_csv(training_index))) if training_index and training_index.is_file() else 0,
        "behavior_cloning": behavior_cloning,
        "training_curve_url": curve_url,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }


def export_roadster(output_dir: Path | str = DEFAULT_OUTPUT_DIR, public_dir: Path | str = DEFAULT_PUBLIC) -> dict[str, Any]:
    root = roadster_root(output_dir)
    public = Path(public_dir).resolve()
    roadster_public = public / "data" / "roadster"
    if roadster_public.exists():
        shutil.rmtree(roadster_public)
    roadster_public.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in ROADSTER_LABELS:
        for run_dir in sorted((root / variant).glob("seed_*")) if (root / variant).is_dir() else []:
            try:
                seed = int(run_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            run = _run_index(root, public, variant, seed)
            if run:
                runs.append(run)

    preview_path = export_rule_based_preview(output_dir)
    reference_evaluation = evaluate_rule_controller(output_dir)
    preview = compact_episode(
        preview_path, kind="reference_controller", variant="reference_controller", seed=20_000
    )
    preview_file = Path("data") / "roadster" / "reference_controller.json"
    write_json(public / preview_file, preview)
    comparison_path = root / "comparison_summary.csv"
    comparison = []
    if comparison_path.is_file():
        table = pd.read_csv(comparison_path).replace([np.inf, -np.inf], np.nan)
        comparison = [
            {key: None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
             for key, value in row.items()}
            for row in table.to_dict("records")
        ]
    index = {
        "schema": 1,
        "simulator_version": SIMULATOR_VERSION,
        "physics": RoadsterPhysics.from_config().to_dict(),
        "observation_fields": list(OBSERVATION_FIELDS),
        "action_fields": list(ACTION_FIELDS),
        "runs": runs,
        "comparison": comparison,
        "reference_evaluation": reference_evaluation,
        "preview": {
            "label": "Rule-based reference controller",
            "kind": "reference_controller",
            "flight_url": "/" + preview_file.as_posix(),
            "outcome": preview["outcome"],
            "native_return": preview["native_return"],
            "ramp_controller": "scripted_fixture_v1",
        },
        "note": "Original planar educational simulation. Ramp playback is scripted; PPO controls flight when a learned run is selected. No Tesla model or controller is represented.",
    }
    write_json(roadster_public / "index.json", index)
    print(f"Exported {len(runs)} Roadster runs and a rule-based reference flight.")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC)
    args = parser.parse_args()
    export_roadster(args.output_dir, args.public_dir)


if __name__ == "__main__":
    main()
