"""Copy completed experiment artifacts into a portable website snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export_showcase(source: Path, target: Path) -> int:
    """Export real models and only evaluations that belong to those models."""
    count = 0
    for model in sorted(source.glob('*/seed_*/model.zip')):
        run = model.parent
        dest = target / run.relative_to(source)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('model.zip', 'metadata.json', 'training_monitor.csv', 'training_status.json'):
            if (run / name).is_file():
                shutil.copy2(run / name, dest / name)
        for folder in ('sb3_logs', 'checkpoints'):
            if (run / folder).is_dir():
                shutil.copytree(run / folder, dest / folder, dirs_exist_ok=True)

        summary = run / 'evaluation_summary.json'
        episodes = run / 'evaluation_episodes.csv'
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        evaluation = json.loads(summary.read_text()) if summary.is_file() else {}
        current = summary.is_file() and episodes.is_file() and (
            evaluation.get('model_sha256') == digest if evaluation.get('model_sha256')
            else min(summary.stat().st_mtime, episodes.stat().st_mtime) >= model.stat().st_mtime
        )
        if current:
            evaluation.pop('model_path', None)
            evaluation['model_sha256'] = digest
            shutil.copy2(episodes, dest / episodes.name)
            (dest / summary.name).write_text(json.dumps(evaluation, indent=2) + '\n')
        else:
            # Remove old derived results if a refreshed model has no current evaluation.
            for name in ('evaluation_summary.json', 'evaluation_episodes.csv'):
                (dest / name).unlink(missing_ok=True)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--target', type=Path, default=PROJECT_ROOT / 'showcase_data')
    args = parser.parse_args()
    count = export_showcase(args.source.resolve(), args.target.resolve())
    print(f'Exported {count} completed model runs to {args.target}')


if __name__ == '__main__':
    main()
