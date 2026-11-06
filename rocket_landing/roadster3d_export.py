"""Export recorded Rapier 3D PPO runs for the static results viewer."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rocket_landing.config import DEFAULT_OUTPUT_DIR
from rocket_landing.envs.roadster3d import SIMULATOR_VERSION_3D, TELEMETRY_FIELDS_3D, RoadsterPhysics3D
from rocket_landing.roadster3d_experiment import roadster3d_root
from rocket_landing.envs import roadster3d_v7
from rocket_landing.roadster3d_experiment_v7 import roadster3d_root as roadster3d_v7_root

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compact_3d_episode(path: Path, *, variant: str, seed: int, episode: int, source: str, simulator_version: int = 6,
                       stage: str = "full", evaluation_seed: int | None = None) -> dict[str, Any]:
    if simulator_version == 6:
        fields, p = TELEMETRY_FIELDS_3D, RoadsterPhysics3D.from_config()
    elif simulator_version == 7:
        fields, p = roadster3d_v7.TELEMETRY_FIELDS_3D, roadster3d_v7.RoadsterPhysics3D.from_config()
    else:
        raise ValueError(f"Unsupported simulator version {simulator_version}")
    with np.load(path, allow_pickle=False) as data:
        states = np.asarray(data["telemetry"], dtype=np.float64)
        phases = data["phases"].astype(str).tolist()
        native = np.asarray(data["native_rewards"], dtype=np.float64)
        shaped = np.asarray(data["shaped_rewards"], dtype=np.float64)
        actions = np.asarray(data["actions"])
        outcome = str(data["outcome"].item())
    steps = len(states) - 1
    if states.ndim != 2 or states.shape[1] != len(fields) or steps < 1:
        raise ValueError(f"Invalid 3D telemetry in {path}")
    if len(phases) != steps + 1 or len(native) != steps or len(shaped) != steps or actions.shape != (steps, 7):
        raise ValueError(f"Inconsistent 3D trajectory lengths in {path}")
    if not all(np.isfinite(x).all() for x in (states, native, shaped, actions)):
        raise ValueError(f"Non-finite 3D trajectory in {path}")
    lookup = {name: i for i, name in enumerate(fields)}
    first, last = states[0], states[-1]
    x, z = float(last[lookup["landing_center_x"]]), float(last[lookup["landing_center_z"]])
    zone_x = [x - p.landing_pad_half_length_m, x + p.landing_pad_half_length_m]
    zone_z = [z - p.landing_pad_half_width_m, z + p.landing_pad_half_width_m]
    rows = np.column_stack((states, np.r_[0., np.cumsum(native)], np.r_[0., np.cumsum(shaped)]))
    return {
        "schema": 4, "simulator_version": simulator_version,
        "columns": [*fields, "native_total", "shaped_total"],
        "rows": rows.round(6).tolist(), "phases": phases, "steps": steps,
        "ramp_radius_m": float(first[lookup["ramp_radius_m"]]),
        "ramp_start_x": float(first[lookup["ramp_start_x"]]),
        "ramp_end_x": float(first[lookup["ramp_end_x"]]),
        "ramp_top_y": float(first[lookup["ramp_top_y"]]),
        "ramp_vertical_stem_length_m": p.ramp_vertical_stem_length_m,
        "ramp_top_curve_radius_m": p.ramp_top_curve_radius_m,
        "ramp_top_curve_angle_deg": p.ramp_top_curve_angle_deg,
        "landing_zone_x": zone_x, "landing_zone_z": zone_z, "outcome": outcome,
        "native_return": float(native.sum()), "shaped_return": float(shaped.sum()),
        "metrics": {"gas_remaining": float(last[lookup["fuel"]]), "elapsed_seconds": float(last[lookup["elapsed_seconds"]])},
        "source": {"kind": source, "variant": variant, "seed": seed, "episode": episode, "stage": stage,
                   "evaluation_seed": evaluation_seed,
                   "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def export_3d(output_dir: Path = DEFAULT_OUTPUT_DIR, public_dir: Path = ROOT / "frontend" / "public") -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    paths = [(6, path) for path in roadster3d_root(output_dir).glob("*/seed_*/metadata.json")]
    paths += [(7, path) for path in roadster3d_v7_root(output_dir).glob("*/*/seed_*/metadata.json")]
    for version, meta_path in sorted(paths, key=lambda item: (-item[0], str(item[1]))):
        run_dir = meta_path.parent
        summary_path, episodes_path = run_dir / "evaluation_summary.json", run_dir / "evaluation_episodes.csv"
        if not summary_path.is_file() or not episodes_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text())
        summary = json.loads(summary_path.read_text())
        artifact_name = metadata.get("artifact_path", "model.zip")
        if Path(artifact_name).name != artifact_name:
            raise ValueError(f"Invalid model artifact path: {run_dir}")
        artifact = run_dir / artifact_name
        if not artifact.is_file():
            raise ValueError(f"Missing evaluated artifact: {artifact}")
        if metadata.get("simulator_version") != version or summary.get("simulator_version") != version:
            continue
        physics = RoadsterPhysics3D.from_config() if version == 6 else roadster3d_v7.RoadsterPhysics3D.from_config()
        if metadata.get("physics_config") != physics.to_dict():
            continue
        if summary.get("model_sha256") != hashlib.sha256(artifact.read_bytes()).hexdigest():
            raise ValueError(f"Model checksum differs from evaluation: {run_dir}")
        variant, seed, stage = metadata["variant"], int(metadata["seed"]), metadata.get("stage", "full")
        if summary.get("stage", "full") != stage:
            raise ValueError(f"Training and evaluation stages differ: {run_dir}")
        attempts = []
        evaluation_records = pd.read_csv(episodes_path).to_dict("records")
        if len(evaluation_records) != summary.get("episodes"):
            raise ValueError(f"Evaluation count differs from summary: {run_dir}")
        for record in evaluation_records:
            episode = int(record["episode"])
            file = run_dir / "evaluation_flights" / f"episode_{episode:04d}.npz"
            if not file.is_file():
                raise ValueError(f"Evaluation trajectory missing: {file}")
            relative = Path("data/roadster3d/flights") / f"v{version}" / stage / variant / str(seed) / f"evaluation_{episode:04d}.json"
            replay = compact_3d_episode(file, variant=variant, seed=seed, episode=episode, source="evaluation", simulator_version=version,
                                        stage=stage, evaluation_seed=int(record["eval_seed"]))
            _json(public_dir / relative, replay)
            attempts.append({**_clean(record), "flight_url": "/" + relative.as_posix()})
        training = []
        recorded = metadata.get("training_trajectory_dir")
        training_dir = run_dir / recorded if recorded else None
        if training_dir and (training_dir / "episodes.csv").is_file():
            records = pd.read_csv(training_dir / "episodes.csv").to_dict("records")
            for offset in range(0, len(records), 50):
                relative = Path("data/roadster3d/bundles") / f"v{version}" / stage / variant / str(seed) / f"training_{offset // 50:04d}.bin"
                bundle_path = public_dir / relative
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                flights = []
                for record in records[offset:offset + 50]:
                    episode = int(record["episode"])
                    file = training_dir / f"episode_{episode:07d}.npz"
                    if not file.is_file():
                        raise ValueError(f"Training trajectory missing: {file}")
                    flights.append(compact_3d_episode(file, variant=variant, seed=seed, episode=episode, source="training", simulator_version=version, stage=stage))
                    training.append({**_clean(record), "bundle_url": "/" + relative.as_posix(), "bundle_index": len(flights) - 1})
                with bundle_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                    out.write(json.dumps({"flights": flights}, allow_nan=False, separators=(",", ":")).encode())
        progress_path = run_dir / "sb3_logs/progress.csv"
        curve = []
        if progress_path.is_file():
            for row in pd.read_csv(progress_path).to_dict("records"):
                curve.append(_clean({"timesteps": row.get("time/total_timesteps"), "return": row.get("rollout/ep_rew_mean"), "length": row.get("rollout/ep_len_mean")}))
        runs.append({"variant": variant, "label": metadata.get("variant_label", variant), "seed": seed, "simulator_version": version, "stage": stage, "policy_type": metadata.get("policy_type", "ppo"),
                     "timesteps": metadata.get("timesteps_completed"), "model_sha256": summary["model_sha256"],
                     "summary": _clean(summary), "evaluations": attempts, "training_flights": training, "training_curve": curve})
    index = {"schema": 1, "simulator_versions": [6, 7], "runs": runs}
    _json(public_dir / "data/roadster3d/index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-dir", type=Path, default=ROOT / "frontend" / "public")
    args = parser.parse_args()
    result = export_3d(args.output_dir, args.public_dir)
    print(f"Exported {len(result['runs'])} evaluated 3D runs")


if __name__ == "__main__":
    main()
