"""Publish the saved-results website using Hugging Face's free Static SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


def extract_verified_bundle(bundle: Path, directory: Path) -> int:
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('SPACE_FILES.json'))
        if manifest.get('format') != 1 or manifest.get('target') != 'huggingface-static':
            raise ValueError('This is not a prepared Landing Lab Space bundle.')
        expected = manifest['files']
        if set(archive.namelist()) != set(expected) | {'SPACE_FILES.json'}:
            raise ValueError('Archive contents differ from the reviewed manifest.')
        for name, digest in expected.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('Invalid archive path.')
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f'Checksum mismatch: {name}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (directory / 'SPACE_FILES.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(expected)


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='msmxii/landing-lab')
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1] / 'dist' / 'landing-lab-static.zip')
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--wait-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.repo.count('/') != 1:
        raise ValueError('Use an account/space-name repository ID.')
    owner = args.repo.split('/')[0]
    api = HfApi()
    account = api.whoami()['name']
    if account.lower() != owner.lower():
        raise ValueError(f'Authenticated as {account}; the target belongs to {owner}. No files were uploaded.')
    print(f'Authenticated account: {account}', flush=True)

    if not args.status_only:
        with tempfile.TemporaryDirectory(prefix='landing-lab-space-') as temporary:
            folder = Path(temporary) / 'complete'
            folder.mkdir()
            count = extract_verified_bundle(args.bundle, folder)
            print(f'Bundle verified: {count} source/artifact files', flush=True)
            local_manifest = json.loads((folder / 'SPACE_FILES.json').read_text())
            remote_files = {}
            try:
                info = api.space_info(args.repo)
            except RepositoryNotFoundError:
                api.create_repo(args.repo, repo_type='space', space_sdk='static', private=False)
                print(f'Created free Static Space: {args.repo}', flush=True)
            else:
                try:
                    readme = Path(hf_hub_download(args.repo, 'README.md', repo_type='space')).read_text()
                except EntryNotFoundError:
                    readme = ''
                if info.sdk != 'static':
                    raise ValueError('Publishing is restricted to a Static Space; compute hosting is disabled.')
                if not any(title in readme for title in ('title: Landing Lab', 'title: Roadster Flight Lab')):
                    raise ValueError('An existing Space at this target does not identify itself as Landing Lab. No files were overwritten.')
                if info.private:
                    raise ValueError('This existing Space is private. Resolve its intended audience before publishing updates.')
                try:
                    remote_manifest_path = hf_hub_download(args.repo, 'SPACE_FILES.json', repo_type='space')
                    remote_manifest = json.loads(Path(remote_manifest_path).read_text())
                    remote_files = remote_manifest.get('files', {})
                except EntryNotFoundError:
                    pass
                # An interrupted upload can update the checksum manifest
                # before the static entry point. Compare index.html itself so
                # a stale page can never be mistaken for a complete deploy.
                try:
                    remote_index = hf_hub_download(args.repo, 'index.html', repo_type='space', force_download=True)
                    remote_files['index.html'] = hashlib.sha256(Path(remote_index).read_bytes()).hexdigest()
                except EntryNotFoundError:
                    pass

            changed = [name for name, digest in local_manifest['files'].items() if remote_files.get(name) != digest]
            if not remote_files:
                changed = list(local_manifest['files'])
            upload_folder = Path(temporary) / 'changed'
            upload_folder.mkdir()
            for name in changed:
                source = folder / name
                destination = upload_folder / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            (upload_folder / 'SPACE_FILES.json').write_text(json.dumps(local_manifest, indent=2) + '\n')
            total_bytes = sum((folder / name).stat().st_size for name in changed)
            print(f'Uploading {len(changed)} changed files ({total_bytes / (1024 * 1024):.1f} MiB)', flush=True)
            commit = api.upload_folder(repo_id=args.repo, repo_type='space', folder_path=upload_folder,
                                       commit_message='Publish car-only Roadster Flight Lab with saved PPO results and replays')
            print(f'Uploaded source commit: {commit.oid}', flush=True)
    url = f'https://huggingface.co/spaces/{args.repo}'
    print(f'Space URL: {url}', flush=True)
    deadline = time.monotonic() + max(0, args.wait_seconds)
    previous = None
    while True:
        runtime = api.get_space_runtime(args.repo)
        stage = str(runtime.stage)
        if stage != previous:
            print(f'Space stage: {stage}', flush=True)
            previous = stage
        if stage == 'RUNNING':
            print(f'Online: {url}', flush=True)
            return
        if stage in ('BUILD_ERROR', 'RUNTIME_ERROR', 'CONFIG_ERROR', 'NO_APP_FILE'):
            raise RuntimeError(f'Hugging Face reported {stage}. Inspect the Space build/runtime logs.')
        if time.monotonic() >= deadline:
            print('Build still in progress. Re-run with --status-only to continue checking without uploading again.', flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
