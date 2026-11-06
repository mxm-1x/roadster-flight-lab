"""Evaluate an explicitly privileged, rule-based v7 flight controller.

This reference reads simulator state and contact directly. It is not a trained
policy and its flight curriculum does not test the ramp launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rocket_landing.config import DEFAULT_OUTPUT_DIR
from rocket_landing.envs.roadster3d_v7 import TELEMETRY_FIELDS_3D, RoadsterPhysics3D
from rocket_landing.roadster3d_experiment_v7 import ROADSTER3D_EVAL_SEED_BASE, make_roadster3d_env, roadster3d_root


def reference_action(env) -> np.ndarray:
    """PD flight reference with privileged mass, pressure and contact state."""
    p = env.physics
    position, velocity, angular, q, target = env._state()
    mass = env._vehicle_mass + env._gas.gas_kg
    desired_v = -min(1.2, max(0.3, (position[1] - 0.8) * 0.65))
    accel = float(np.clip(4 * (desired_v - velocity[1]), -4, 7))
    pressure = max(0.01, (env._gas.pressure_pa - p.gas_ambient_pressure_pa) /
                   (p.gas_initial_pressure_pa - p.gas_ambient_pressure_pa))
    lift = float(np.clip(mass * (p.gravity_mps2 + accel) / (p.vertical_thrust_max_n * pressure), 0, 1))
    if env._has_contact(env._ground_collider):
        lift = 0.0
    forward = np.clip((target[0] - position[0] - 2 * velocity[0]) * .25, -1, 1)
    lateral = np.clip((target[2] - position[2] - 2 * velocity[2]) * .5, -1, 1)
    pitch = np.clip(-2 * q[2] - 1.5 * angular[2], -1, 1)
    roll = np.clip(2 * q[0] + 1.5 * angular[0], -1, 1)
    yaw = np.clip(-2 * q[1] - 1.5 * angular[1], -1, 1)
    return np.asarray((-1, forward, 2 * lift - 1, lateral, pitch, roll, yaw), dtype=np.float32)


def evaluate_reference(output_dir: Path = DEFAULT_OUTPUT_DIR, episodes: int = 20) -> Path:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    run = roadster3d_root(output_dir) / "reference" / "flight" / "seed_0"
    flights = run / "evaluation_flights"
    flights.mkdir(parents=True, exist_ok=True)
    controller_path = run / "controller.py"
    controller_path.write_bytes(Path(__file__).read_bytes())
    controller_hash = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    results = []
    for episode in range(episodes):
        env = make_roadster3d_env("baseline", stage="flight")
        try:
            _, info = env.reset(seed=ROADSTER3D_EVAL_SEED_BASE + episode)
            rows, phases, actions, native_rewards, shaped_rewards = [dict(info["telemetry"])], [str(info["phase"])], [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                action = reference_action(env.unwrapped)
                _, reward, terminated, truncated, info = env.step(action)
                rows.append(dict(info["telemetry"]))
                phases.append(str(info["phase"]))
                actions.append(np.asarray(info["action_applied"], dtype=np.float32))
                native_rewards.append(float(info["base_reward"]))
                shaped_rewards.append(float(reward))
            np.savez_compressed(flights / f"episode_{episode:04d}.npz",
                telemetry=np.asarray([[float(row.get(key, 0.0)) for key in TELEMETRY_FIELDS_3D] for row in rows], dtype=np.float32),
                phases=np.asarray(phases, dtype="U16"), actions=np.asarray(actions, dtype=np.float32),
                native_rewards=np.asarray(native_rewards), shaped_rewards=np.asarray(shaped_rewards), outcome=np.asarray(info["outcome"]))
            results.append({"episode": episode, "eval_seed": ROADSTER3D_EVAL_SEED_BASE + episode,
                "outcome": info["outcome"], "safe_landing": info["safe_landing"], "crash": info["crash"],
                "truncated": info["truncated"], "native_return": sum(native_rewards), "shaped_return": sum(shaped_rewards),
                "episode_length": len(actions), "touchdown_speed": info["touchdown_speed"], "gas_remaining": info["gas_remaining"]})
        finally:
            env.close()
    frame = pd.DataFrame(results)
    frame.to_csv(run / "evaluation_episodes.csv", index=False)
    summary = {"simulator_version": 7, "stage": "flight", "variant": "reference", "seed": 0,
               "episodes": episodes, "model_sha256": controller_hash,
               "safe_landing_rate": float(frame.safe_landing.mean()), "crash_rate": float(frame.crash.mean()),
               "timeout_rate": float(frame.truncated.mean()), "mean_native_return": float(frame.native_return.mean())}
    metadata = {"simulator_version": 7, "stage": "flight", "variant": "reference", "variant_label": "Privileged PD reference",
                "seed": 0, "timesteps_completed": 0, "policy_type": "rule_based", "artifact_path": "controller.py",
                "physics_config": RoadsterPhysics3D.from_config().to_dict(),
                "training_protocol": "Hand-written controller with direct simulator state and contact access; no learning."}
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Reference: {int(frame.safe_landing.sum())}/{episodes} safe landings")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(evaluate_reference(args.output_dir, args.episodes))


if __name__ == "__main__":
    main()
