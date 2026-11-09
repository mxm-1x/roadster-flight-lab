"""Command-line entry point for experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from rocket_landing.config import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TIMESTEPS,
    VARIANTS,
)
from rocket_landing.experiment import evaluate_run, run_experiments, train_run
from rocket_landing.roadster_experiment import (
    ROADSTER_EVAL_EPISODES,
    ROADSTER_SEEDS,
    ROADSTER_TIMESTEPS,
    ROADSTER_VARIANTS,
    evaluate_roadster_run,
    export_rule_based_preview,
    run_roadster_experiments,
    train_roadster_run,
    write_roadster_comparison,
)
from rocket_landing.roadster3d_experiment import (
    ROADSTER3D_EVAL_EPISODES,
    ROADSTER3D_SEEDS,
    ROADSTER3D_TIMESTEPS,
    evaluate_roadster3d_run,
    run_roadster3d_experiments,
    train_roadster3d_run,
    write_roadster3d_comparison,
)
from rocket_landing import roadster3d_experiment_v7 as roadster_v7
from rocket_landing.roadster3d_rewards import ROADSTER3D_V7_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Landing Lab PPO experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one variant/seed model.")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on the native reward.")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--variant", choices=VARIANTS, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    evaluate.add_argument("--no-video", action="store_true")

    all_runs = subparsers.add_parser("run-all", help="Train all variants/seeds and produce reports/videos.")
    all_runs.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    all_runs.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    all_runs.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    all_runs.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster_train = subparsers.add_parser("roadster-train", help="Train one Roadster flight policy.")
    roadster_train.add_argument("--variant", choices=ROADSTER_VARIANTS, required=True)
    roadster_train.add_argument("--seed", type=int, required=True)
    roadster_train.add_argument("--timesteps", type=int, default=ROADSTER_TIMESTEPS)
    roadster_train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster_evaluate = subparsers.add_parser("roadster-evaluate", help="Evaluate one Roadster model.")
    roadster_evaluate.add_argument("--model", type=Path, required=True)
    roadster_evaluate.add_argument("--variant", choices=ROADSTER_VARIANTS, required=True)
    roadster_evaluate.add_argument("--seed", type=int, required=True)
    roadster_evaluate.add_argument("--episodes", type=int, default=ROADSTER_EVAL_EPISODES)
    roadster_evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster_all = subparsers.add_parser("roadster-run-all", help="Train and evaluate Roadster reward variants.")
    roadster_all.add_argument("--timesteps", type=int, default=ROADSTER_TIMESTEPS)
    roadster_all.add_argument("--episodes", type=int, default=ROADSTER_EVAL_EPISODES)
    roadster_all.add_argument("--seeds", type=int, nargs="+", default=list(ROADSTER_SEEDS))
    roadster_all.add_argument("--variants", choices=ROADSTER_VARIANTS, nargs="+", default=list(ROADSTER_VARIANTS))
    roadster_all.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    roadster_all.add_argument("--force", action="store_true", help="Retrain even when matching completed runs exist.")

    preview = subparsers.add_parser("roadster-preview", help="Save a rule-based Roadster preview episode.")
    preview.add_argument("--seed", type=int, default=20_000)
    preview.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster_report = subparsers.add_parser("roadster-report", help="Rebuild Roadster aggregate CSVs from saved evaluations.")
    roadster_report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster3d_train = subparsers.add_parser("roadster3d-train", help="Train one v6 Rapier 3D Roadster policy.")
    roadster3d_train.add_argument("--variant", choices=ROADSTER_VARIANTS, required=True)
    roadster3d_train.add_argument("--seed", type=int, required=True)
    roadster3d_train.add_argument("--timesteps", type=int, default=ROADSTER3D_TIMESTEPS)
    roadster3d_train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster3d_evaluate = subparsers.add_parser("roadster3d-evaluate", help="Evaluate a v6 Rapier 3D model.")
    roadster3d_evaluate.add_argument("--model", type=Path, required=True)
    roadster3d_evaluate.add_argument("--variant", choices=ROADSTER_VARIANTS, required=True)
    roadster3d_evaluate.add_argument("--seed", type=int, required=True)
    roadster3d_evaluate.add_argument("--episodes", type=int, default=ROADSTER3D_EVAL_EPISODES)
    roadster3d_evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    roadster3d_all = subparsers.add_parser("roadster3d-run-all", help="Train and evaluate the v6 3D reward study.")
    roadster3d_all.add_argument("--timesteps", type=int, default=ROADSTER3D_TIMESTEPS)
    roadster3d_all.add_argument("--episodes", type=int, default=ROADSTER3D_EVAL_EPISODES)
    roadster3d_all.add_argument("--seeds", type=int, nargs="+", default=list(ROADSTER3D_SEEDS))
    roadster3d_all.add_argument("--variants", choices=ROADSTER_VARIANTS, nargs="+", default=list(ROADSTER_VARIANTS))
    roadster3d_all.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    roadster3d_all.add_argument("--force", action="store_true", help="Retrain v6 runs even when complete.")

    roadster3d_report = subparsers.add_parser("roadster3d-report", help="Rebuild v6 3D aggregate metrics.")
    roadster3d_report.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    v7_train = subparsers.add_parser("roadster3d-v7-train", help="Train v7 articulated 3D PPO.")
    v7_train.add_argument("--variant", choices=ROADSTER3D_V7_VARIANTS, required=True)
    v7_train.add_argument("--seed", type=int, required=True)
    v7_train.add_argument("--stage", choices=("full", "flight", "flight_curriculum", "flight_profile_0", "flight_profile_1", "flight_profile_2"), default="full")
    v7_train.add_argument("--timesteps", type=int, default=ROADSTER3D_TIMESTEPS)
    v7_train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    v7_train.add_argument("--init-model", type=Path)

    v7_evaluate = subparsers.add_parser("roadster3d-v7-evaluate", help="Evaluate a v7 3D checkpoint.")
    v7_evaluate.add_argument("--model", type=Path, required=True)
    v7_evaluate.add_argument("--variant", choices=ROADSTER3D_V7_VARIANTS, required=True)
    v7_evaluate.add_argument("--seed", type=int, required=True)
    v7_evaluate.add_argument("--stage", choices=("full", "flight", "flight_curriculum", "flight_profile_0", "flight_profile_1", "flight_profile_2"), default="full")
    v7_evaluate.add_argument("--profile", type=int, choices=(0, 1, 2), default=2, help="Fixed difficulty profile for curriculum evaluation.")
    v7_evaluate.add_argument("--episodes", type=int, default=ROADSTER3D_EVAL_EPISODES)
    v7_evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        print(train_run(args.variant, args.seed, args.timesteps, args.output_dir))
    elif args.command == "evaluate":
        _, summary = evaluate_run(
            args.model,
            args.variant,
            args.seed,
            args.episodes,
            args.output_dir,
            capture_video=not args.no_video,
        )
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "run-all":
        print(run_experiments(args.output_dir, args.timesteps, args.episodes, tuple(args.seeds)).to_string(index=False))
    elif args.command == "roadster-train":
        print(train_roadster_run(args.variant, args.seed, args.timesteps, args.output_dir))
    elif args.command == "roadster-evaluate":
        _, summary = evaluate_roadster_run(args.model, args.variant, args.seed, args.episodes, args.output_dir)
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "roadster-run-all":
        print(run_roadster_experiments(
            args.output_dir, args.timesteps, args.episodes, tuple(args.seeds), tuple(args.variants), args.force
        ).to_string(index=False))
    elif args.command == "roadster-preview":
        print(export_rule_based_preview(args.output_dir, args.seed))
    elif args.command == "roadster-report":
        print(write_roadster_comparison(args.output_dir).to_string(index=False))
    elif args.command == "roadster3d-train":
        print(train_roadster3d_run(args.variant, args.seed, args.timesteps, args.output_dir))
    elif args.command == "roadster3d-evaluate":
        _, summary = evaluate_roadster3d_run(
            args.model, args.variant, args.seed, args.episodes, args.output_dir
        )
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "roadster3d-run-all":
        print(run_roadster3d_experiments(
            args.output_dir, args.timesteps, args.episodes,
            tuple(args.seeds), tuple(args.variants), args.force
        ).to_string(index=False))
    elif args.command == "roadster3d-report":
        print(write_roadster3d_comparison(args.output_dir).to_string(index=False))
    elif args.command == "roadster3d-v7-train":
        print(roadster_v7.train_roadster3d_run(args.variant, args.seed, args.timesteps, args.output_dir, stage=args.stage, init_model=args.init_model))
    elif args.command == "roadster3d-v7-evaluate":
        _, summary = roadster_v7.evaluate_roadster3d_run(args.model, args.variant, args.seed, args.episodes, args.output_dir, stage=args.stage, profile=args.profile)
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
