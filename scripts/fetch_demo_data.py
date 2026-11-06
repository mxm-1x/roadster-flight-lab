"""Fetch checksum-verified public replay data from the deployed static Space."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / 'frontend' / 'public'
BASE = 'https://huggingface.co/spaces/msmxii/landing-lab/resolve/main/'


def main():
    with urlopen(BASE + 'SPACE_FILES.json', timeout=60) as response:
        manifest = json.load(response)
    if manifest.get('target') != 'huggingface-static' or manifest.get('format') != 1:
        raise ValueError('Unexpected static data manifest')
    items = [(name, digest) for name, digest in manifest['files'].items()
             if name.startswith('data/') or name == 'plotly.min.js']

    def fetch(item):
        name, digest = item
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Invalid manifest path')
        target = ROOT / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            return
        with urlopen(BASE + quote(name, safe='/'), timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch for {name}; retry after deployment completes')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(fetch, items):
            pass
    print(f'Verified {len(items)} public replay assets in {ROOT}')


if __name__ == '__main__':
    main()
