import numpy as np
import gymnasium as gym
from pathlib import Path
from streamlit.testing.v1 import AppTest

from rocket_landing.recording import TrainingRecorder
from rocket_landing.replay import flight_figure, load_training_episode


class TwoStepLander(gym.Env):
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(-10, 10, shape=(8,), dtype=np.float32)
    helipad_x1 = 8.0
    helipad_x2 = 12.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_number = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.step_number += 1
        state = np.zeros(8, dtype=np.float32)
        state[0] = self.step_number * 0.1
        state[4] = self.step_number * 0.2
        done = self.step_number == 2
        return state, -100.0 if done else 1.0, done, False, {}


def test_recorded_training_flight_preserves_actions_states_and_outcome(tmp_path):
    env = TrainingRecorder(TwoStepLander(), tmp_path)
    env.reset(seed=123)
    env.step(2)
    assert not list(tmp_path.glob("*.npz"))
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["outcome"] == "Crash"
    assert info["native_return"] == -99.0
    assert "1,Crash,-99.0,-99.0,2" in (tmp_path / "episodes.csv").read_text()
    path = next(tmp_path.glob("*.npz"))
    flight, pad = load_training_episode(path)
    assert flight["action"].tolist() == [-1, 2, 0]
    np.testing.assert_allclose(flight["x"], [0, 0.1, 0.2])
    assert flight.attrs["outcome"] == "Crash"
    assert flight["native_total"].iloc[-1] == -99.0
    assert pad == (-2.0, 2.0)

    figure = flight_figure(flight, pad)
    assert len(figure.frames) == 3
    assert figure.frames[-1].name == "2"
    assert figure.frames[0].data[0].x[0] != figure.frames[-1].data[0].x[0]


def test_app_filters_and_renders_a_recorded_crash(tmp_path, monkeypatch):
    run = tmp_path / "baseline" / "seed_5"
    recordings = run / "training_trajectories" / "session"
    env = TrainingRecorder(TwoStepLander(), recordings)
    env.reset()
    env.step(2)
    env.step(0)
    (run / "model.zip").write_bytes(b"placeholder")
    monkeypatch.setenv("ROCKET_LANDING_OUTPUT_DIR", str(tmp_path))
    app_file = Path(__file__).resolve().parents[1] / "rocket_landing" / "demo.py"
    app = AppTest.from_file(app_file, default_timeout=30).run()
    app.get("segmented_control")[0].set_value("Flights & 3D").run()
    next(s for s in app.selectbox if s.label == "Show training attempts").select("Crash").run()
    next(b for b in app.button if b.label == "Load 3D flight").click().run()
    assert not app.exception
    assert any(metric.label == "Outcome" and metric.value == "Crash" for metric in app.metric)
    assert len(app.get("plotly_chart")) == 1
