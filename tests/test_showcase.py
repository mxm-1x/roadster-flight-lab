"""Ensure published snapshots never attach old scores to new models."""
import json
import os
import zipfile
from pathlib import Path

from scripts.export_showcase import export_showcase
from scripts.prepare_space import prepare_space


def make_run(root: Path):
    run = root / 'baseline' / 'seed_0'
    run.mkdir(parents=True)
    (run / 'model.zip').write_bytes(b'checkpoint-v1')
    (run / 'metadata.json').write_text('{"seed":0}')
    (run / 'evaluation_summary.json').write_text(json.dumps({'mean_return': 25, 'model_path': '/private/local/model.zip'}))
    (run / 'evaluation_episodes.csv').write_text('episode,env_return\n0,25\n')
    return run


def test_export_has_metadata_and_portable_evaluation_identity(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    make_run(source)
    assert export_showcase(source, target) == 1
    result = target / 'baseline' / 'seed_0'
    summary = json.loads((result / 'evaluation_summary.json').read_text())
    assert len(summary['model_sha256']) == 64
    assert 'model_path' not in summary
    assert json.loads((result / 'metadata.json').read_text())['seed'] == 0


def test_export_drops_scores_after_model_is_replaced(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    run = make_run(source)
    export_showcase(source, target)
    (run / 'model.zip').write_bytes(b'checkpoint-v2')
    timestamp = (run / 'evaluation_summary.json').stat().st_mtime + 10
    os.utime(run / 'model.zip', (timestamp, timestamp))
    export_showcase(source, target)
    result = target / 'baseline' / 'seed_0'
    assert (result / 'model.zip').read_bytes() == b'checkpoint-v2'
    assert not (result / 'evaluation_summary.json').exists()
    assert not (result / 'evaluation_episodes.csv').exists()


def test_static_bundle_excludes_servers_models_and_credentials(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'README.md').write_text('---\nsdk: static\napp_file: index.html\n---\n')
    assets = root / 'rocket_landing' / 'web_assets'
    (assets / 'data' / 'roadster').mkdir(parents=True)
    (assets / 'index.html').write_text('built UI')
    (assets / 'data' / 'roadster' / 'index.json').write_text('{"runs":[]}')
    for name in ('.env', '.venv/token', 'frontend/node_modules/private.js', 'outputs/model.zip', 'Dockerfile'):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('private')
    archive = tmp_path / 'static.zip'
    manifest = prepare_space(root, archive)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'index.html', 'data/roadster/index.json', 'README.md', 'SPACE_FILES.json'}
        assert manifest['target'] == 'huggingface-static'
        assert manifest == json.loads(z.read('SPACE_FILES.json'))


def test_static_publisher_rejects_compute_hosting_configuration(tmp_path):
    import pytest
    root = tmp_path / 'project'
    assets = root / 'rocket_landing' / 'web_assets'
    (assets / 'data' / 'roadster').mkdir(parents=True)
    (assets / 'index.html').write_text('UI')
    (assets / 'data' / 'roadster' / 'index.json').write_text('{"runs":[]}')
    (root / 'README.md').write_text('sdk: docker')
    with pytest.raises(ValueError, match='free static SDK'):
        prepare_space(root, tmp_path / 'static.zip')
