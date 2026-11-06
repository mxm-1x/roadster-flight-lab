"""Protect runtime experiments when bundled showcase data initializes a new host."""
from rocket_landing.web import bootstrap_showcase


def test_bootstrap_copies_complete_runs_to_empty_storage(tmp_path):
    source = tmp_path / 'bundled'
    run = source / 'baseline' / 'seed_0'
    run.mkdir(parents=True)
    (run / 'model.zip').write_bytes(b'bundled-model')
    (run / 'metadata.json').write_text('{"seed":0}')
    target = tmp_path / 'persistent'
    assert bootstrap_showcase(source, target) == 1
    copied = target / 'baseline' / 'seed_0'
    assert (copied / 'model.zip').read_bytes() == b'bundled-model'
    assert (copied / 'metadata.json').is_file()
    assert not list(target.rglob('.showcase-*'))


def test_bootstrap_preserves_existing_models_and_online_runs(tmp_path):
    source, target = tmp_path / 'bundled', tmp_path / 'persistent'
    for root, content in ((source, b'bundled'), (target, b'newer-local-model')):
        run = root / 'baseline' / 'seed_0'
        run.mkdir(parents=True)
        (run / 'model.zip').write_bytes(content)
    online = target / 'flawed_bonus' / 'seed_1000000001'
    online.mkdir(parents=True)
    (online / 'model.zip').write_bytes(b'visitor-trained-model')
    assert bootstrap_showcase(source, target) == 0
    assert (target / 'baseline' / 'seed_0' / 'model.zip').read_bytes() == b'newer-local-model'
    assert (online / 'model.zip').read_bytes() == b'visitor-trained-model'
    assert bootstrap_showcase(source, source) == 0
