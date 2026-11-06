import pytest

pytest.importorskip("gymnasium")

from rocket_landing.experiment import _classify_episode


@pytest.mark.parametrize(
    ("last_reward", "terminated", "truncated", "expected"),
    [
        (100.0, True, False, (True, False, False)),
        (-100.0, True, False, (False, True, False)),
        (0.0, False, True, (False, False, True)),
        (0.0, False, False, (False, False, False)),
    ],
)
def test_episode_outcome_classification(last_reward, terminated, truncated, expected):
    assert _classify_episode(last_reward, terminated, truncated) == expected


def test_lunar_lander_observation_and_action_api_and_seeded_reset():
    import gymnasium as gym

    env = gym.make("LunarLander-v3")
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    assert first.shape == (8,)
    assert env.action_space.n == 4
    assert (first == second).all()
    observation, reward, terminated, truncated, _ = env.step(0)
    assert observation.shape == (8,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()
