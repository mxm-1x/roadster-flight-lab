"""Package the built website for free static hosting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_space(root: Path, output: Path) -> dict:
    built = root / 'rocket_landing' / 'web_assets'
    if not (built / 'index.html').is_file() or not (built / 'data' / 'roadster' / 'index.json').is_file():
        raise ValueError('Export the saved experiment data and build the frontend first.')
    readme = (root / 'README.md').read_text()
    if 'sdk: static' not in readme:
        raise ValueError('Publishing is restricted to the free static SDK.')
    files = [(path.relative_to(built).as_posix(), path) for path in sorted(built.rglob('*')) if path.is_file()]
    files.append(('README.md', root / 'README.md'))
    manifest = {'format': 1, 'target': 'huggingface-static', 'files': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Invalid static asset: {name}')
            content = path.read_bytes()
            archive.writestr(name, content)
            manifest['files'][name] = hashlib.sha256(content).hexdigest()
        archive.writestr('SPACE_FILES.json', json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'dist' / 'landing-lab-static.zip')
    args = parser.parse_args()
    manifest = prepare_space(PROJECT_ROOT, args.output)
    print(f'Prepared {len(manifest["files"])} static files: {args.output}')


if __name__ == '__main__':
    main()
