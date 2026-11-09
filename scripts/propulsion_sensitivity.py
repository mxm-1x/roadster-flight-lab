"""Compare privileged flight control under explicit propellant mass assumptions."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rocket_landing.envs.roadster3d_v7 import RoadsterLanding3DEnv, RoadsterPhysics3D
from rocket_landing.roadster3d_experiment_v7 import FlightStart
from rocket_landing.roadster3d_rewards import apply_roadster3d_reward
from rocket_landing.reference_3d_v7 import reference_action


def run_sensitivity(
    gas_masses: tuple[float, ...] = (80.0, 160.0, 240.0),
    seeds: range = range(40_000, 40_020),
    output: Path = Path("outputs/propulsion_sensitivity.csv"),
) -> pd.DataFrame:
    base = RoadsterPhysics3D.from_config()
    rows = []
    for gas_mass in gas_masses:
        for seed in seeds:
            physics = replace(base, gas_initial_mass_kg=gas_mass)
            env = FlightStart(
                apply_roadster3d_reward(RoadsterLanding3DEnv(physics), "baseline"),
                profile=0,
            )
            try:
                _, info = env.reset(seed=seed)
                terminated = truncated = False
                while not (terminated or truncated):
                    _, _, terminated, truncated, info = env.step(reference_action(env.unwrapped))
                rows.append({
                    "gas_initial_mass_kg": gas_mass,
                    "seed": seed,
                    "outcome": info["outcome"],
                    "safe_landing": bool(info["safe_landing"]),
                    "touchdown_speed_mps": info["touchdown_speed"],
                    "touchdown_tilt_rad": info["touchdown_tilt"],
                    "gas_remaining_fraction": info["gas_remaining"],
                })
            finally:
                env.close()
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    for mass, group in frame.groupby("gas_initial_mass_kg"):
        counts = Counter(group["outcome"])
        print(f"{mass:g} kg: {counts.get('Safe landing', 0)}/{len(group)} safe")
    print(f"Wrote episode results to {output}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=40_000)
    parser.add_argument("--output", type=Path, default=Path("outputs/propulsion_sensitivity.csv"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    run_sensitivity(seeds=range(args.seed_base, args.seed_base + args.episodes), output=args.output)


if __name__ == "__main__":
    main()
