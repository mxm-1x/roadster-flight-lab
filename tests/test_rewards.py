import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rocket_landing.rewards import (  # noqa: E402
    PotentialProgressReward,
    UprightSlowBonus,
    shaping_potential,
)


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, observation, reward=2.0, terminated=False, truncated=False):
        self.observation = np.asarray(observation, dtype=np.float32)
        self.reward_value = reward
        self.terminated_value = terminated
        self.truncated_value = truncated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        return (
            self.observation.copy(),
            self.reward_value,
            self.terminated_value,
            self.truncated_value,
            {},
        )


def test_potential_is_finite_and_rewards_leg_contact():
    airborne = np.zeros(8)
    landed = np.zeros(8)
    landed[6:] = 1
    assert np.isfinite(shaping_potential(airborne))
    assert shaping_potential(landed) > shaping_potential(airborne)


def test_potential_rejects_bad_observations():
    with pytest.raises(ValueError):
        shaping_potential([0, 0])
    with pytest.raises(ValueError):
        shaping_potential([0, 0, 0, 0, 0, 0, 0, float("nan")])


def test_upright_slow_bonus_thresholds_and_nonterminal_only():
    qualifying = np.zeros(8)
    env = UprightSlowBonus(OneStepEnv(qualifying, reward=2.0))
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(3.0)
    assert info["base_reward"] == pytest.approx(2.0)
    assert info["shaping_component"] == pytest.approx(1.0)
    assert not terminated and not truncated

    too_fast = qualifying.copy()
    too_fast[2] = 0.1
    env = UprightSlowBonus(OneStepEnv(too_fast, reward=2.0))
    env.reset()
    assert env.step(0)[1] == pytest.approx(2.0)

    terminal = UprightSlowBonus(OneStepEnv(qualifying, reward=2.0, terminated=True))
    terminal.reset()
    assert terminal.step(0)[1] == pytest.approx(2.0)


def test_potential_shaping_matches_formula_and_zeros_terminal_potential():
    observation = np.zeros(8, dtype=np.float32)
    env = PotentialProgressReward(OneStepEnv(observation, reward=2.0))
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == pytest.approx(2.0 - 0.01)
    assert not terminated and not truncated

    terminal = PotentialProgressReward(
        OneStepEnv(observation, reward=2.0, terminated=True)
    )
    terminal.reset()
    previous_potential = terminal._previous_potential
    _, terminal_reward, terminated, truncated, _ = terminal.step(0)
    expected = 2.0 + 0.5 * (0.0 - previous_potential)
    assert terminated and not truncated
    assert np.isfinite(terminal_reward)
    assert terminal_reward == pytest.approx(expected)


@pytest.mark.parametrize("variant", ["baseline", "flawed_bonus", "potential_shaping"])
def test_real_environment_reward_wrapper_preserves_transition(variant):
    from rocket_landing.rewards import apply_reward_variant

    original = gym.make("LunarLander-v3")
    wrapped = apply_reward_variant(gym.make("LunarLander-v3"), variant)
    original_obs, _ = original.reset(seed=45)
    wrapped_obs, _ = wrapped.reset(seed=45)
    np.testing.assert_array_equal(original_obs, wrapped_obs)

    for action in (0, 2, 1, 3):
        original_obs, original_reward, original_terminated, original_truncated, _ = original.step(action)
        wrapped_obs, wrapped_reward, wrapped_terminated, wrapped_truncated, info = wrapped.step(action)
        np.testing.assert_array_equal(original_obs, wrapped_obs)
        assert original_terminated == wrapped_terminated
        assert original_truncated == wrapped_truncated
        assert np.isfinite(wrapped_reward)
        if variant != "baseline":
            assert info["base_reward"] == pytest.approx(original_reward)
    original.close()
    wrapped.close()
