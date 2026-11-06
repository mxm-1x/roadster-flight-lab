"""The planar exporter must not erase separately generated 3D evidence."""
import sys

from rocket_landing import static_export


def test_planar_export_preserves_3d_data(tmp_path, monkeypatch):
    public = tmp_path / 'public'
    saved = public / 'data' / 'roadster3d' / 'index.json'
    saved.parent.mkdir(parents=True)
    saved.write_text('{"runs":[]}')
    old = public / 'data' / 'roadster' / 'old.json'
    old.parent.mkdir(parents=True)
    old.write_text('obsolete')
    monkeypatch.setattr(sys, 'argv', ['export', '--public-dir', str(public)])
    monkeypatch.setattr(static_export, 'export_roadster', lambda *_: {'runs': []})
    static_export.main()
    assert saved.read_text() == '{"runs":[]}'
    assert not old.exists()
