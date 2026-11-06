from pathlib import Path

from rocket_landing.config import PROJECT_ROOT, resolve_output_dir


def test_default_output_directory_is_project_level(monkeypatch):
    monkeypatch.delenv("ROCKET_LANDING_OUTPUT_DIR", raising=False)
    assert resolve_output_dir() == PROJECT_ROOT / "outputs"


def test_relative_output_directory_is_resolved_from_project_root():
    assert resolve_output_dir("artifacts") == PROJECT_ROOT / "artifacts"


def test_environment_variable_controls_shared_output_directory(monkeypatch, tmp_path):
    external_output = tmp_path / "rl-artifacts"
    monkeypatch.setenv("ROCKET_LANDING_OUTPUT_DIR", str(external_output))
    assert resolve_output_dir() == external_output
