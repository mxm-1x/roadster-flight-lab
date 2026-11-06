"""Export the saved Roadster experiment for the car-only static website."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import plotly

from rocket_landing.config import DEFAULT_OUTPUT_DIR, resolve_output_dir
from rocket_landing.roadster_export import export_roadster

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--public-dir', type=Path, default=PROJECT_ROOT / 'frontend' / 'public')
    args = parser.parse_args()

    source = resolve_output_dir(args.output_dir)
    public = args.public_dir.resolve()
    data = public / 'data'
    # Refresh only the planar archive; saved 3D runs have their own exporter.
    archive = data / 'roadster'
    if archive.exists():
        shutil.rmtree(archive)
    data.mkdir(parents=True, exist_ok=True)

    result = export_roadster(source, public)
    plotly_js = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    shutil.copy2(plotly_js, public / 'plotly.min.js')
    print(
        f"Exported {len(result['runs'])} Roadster policies and their saved trajectories. "
        'Build the frontend with npm run build.',
        flush=True,
    )


if __name__ == '__main__':
    main()
