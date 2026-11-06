from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from rocket_landing import demo  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_available_runs_discovers_only_saved_models(monkeypatch, tmp_path):
    model = tmp_path / "baseline" / "seed_3" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model placeholder")
    (tmp_path / "flawed_bonus" / "seed_4").mkdir(parents=True)
    monkeypatch.setattr(demo, "OUTPUTS", tmp_path)

    assert demo._available_runs() == [("baseline", 3, model)]


def test_empty_state_explains_training_and_output_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCKET_LANDING_OUTPUT_DIR", str(tmp_path))
    app_file = Path(__file__).resolve().parents[1] / "rocket_landing" / "demo.py"
    app = AppTest.from_file(app_file, default_timeout=20).run()

    assert not app.exception
    assert app.info
    assert str(tmp_path) in app.info[0].value
    assert any("run-all" in code.value for code in app.get("code"))


@pytest.mark.parametrize("state", ["running", "interrupted"])
def test_incomplete_training_is_visible_without_a_saved_model(monkeypatch, tmp_path, state):
    run_dir = tmp_path / "baseline" / "seed_1"
    run_dir.mkdir(parents=True)
    (run_dir / "training_monitor.csv").write_text('#{"t_start":0}\nr,l,t\n-20,40,1\n')
    (run_dir / "training_status.json").write_text(f'{{"state":"{state}"}}')
    monkeypatch.setenv("ROCKET_LANDING_OUTPUT_DIR", str(tmp_path))
    app_file = Path(__file__).resolve().parents[1] / "rocket_landing" / "demo.py"
    app = AppTest.from_file(app_file, default_timeout=20).run()
    assert not app.exception
    assert "training log" in app.selectbox[0].options[0]
    assert any(metric.label == "Training episodes" and metric.value == "1" for metric in app.metric)
