import numpy as np
import pytest
from dataclasses import replace

from rocket_landing.envs.roadster3d_v7 import RoadsterLanding3DEnv, RoadsterPhysics3D, _vec


def test_v7_has_articulated_wheels_and_repeatable_state():
    env = RoadsterLanding3DEnv()
    try:
        first, info = env.reset(seed=7)
        assert first.shape == (30,)
        assert len(env._wheel_joints) == len(env._wheel_colliders) == 4
        assert info['telemetry']['gas_kg'] == env.physics.gas_initial_mass_kg
        second, _ = env.reset(seed=7)
        np.testing.assert_allclose(first, second, atol=1e-6)
        for _ in range(30):
            observation, reward, terminated, truncated, _ = env.step(np.array([1, 0, -1, 0, 0, 0, 0], dtype=np.float32))
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert not terminated and not truncated
        assert abs(observation[22:26]).max() > 0.01
    finally:
        env.close()


def test_v7_thruster_consumes_gas_and_reduces_carried_mass():
    env = RoadsterLanding3DEnv()
    try:
        env.reset(seed=1)
        initial_mass = env._vehicle.mass
        position, _, _, quaternion, _ = env._state()
        used = env._apply_thrusters(np.array([0, 1, 0, 0, 0, 0]), position, quaternion)
        assert used > 0
        assert env._gas.gas_kg < env.physics.gas_initial_mass_kg
        # Rapier applies the mass update on the next world step.
        env._world.step()
        assert env._vehicle.mass < initial_mass
    finally:
        env.close()


def test_flight_curriculum_records_3d_wheels_and_uses_ground_contact():
    env = RoadsterLanding3DEnv()
    try:
        observation, info = env.reset(seed=19, options={'phase': 'flight'})
        assert observation.shape == (30,) and info['phase'] == 'flight'
        assert info['telemetry']['ground_contact'] == 0
        assert 'wheel_0_qw' in info['telemetry']
        # With all thrusters off, this initial height results in an unsafe impact.
        action = np.array([-1, 0, -1, 0, 0, 0, 0], dtype=np.float32)
        for _ in range(300):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info['outcome'] == 'Crash'
        assert info['telemetry']['ground_contact'] == 1
        assert info['touchdown_speed'] > 0
    finally:
        env.close()


def test_free_fall_agrees_with_gravity_before_contact():
    physics = replace(RoadsterPhysics3D.from_config(), wind_force_max_n=0, drag_coefficient_area_m2=0)
    env = RoadsterLanding3DEnv(physics)
    try:
        env.reset(seed=10, options={'phase': 'flight'})
        initial_vy = float(env._vehicle.linvel.y)
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(np.array([-1, 0, -1, 0, 0, 0, 0], dtype=np.float32))
            assert not terminated and not truncated and not info['telemetry']['ground_contact']
        assert env._vehicle.linvel.y == pytest.approx(initial_vy - physics.gravity_mps2 * 10 * physics.dt_seconds, abs=.12)
    finally:
        env.close()


def test_height_alone_does_not_trigger_touchdown():
    env = RoadsterLanding3DEnv()
    try:
        env.reset(seed=4, options={'phase': 'flight'})
        offset = np.array([0., 0.9 - env._vehicle.translation.y, 0.])
        wheels = [env._world.impulse_joints[h].body2 for h in env._wheel_joints]
        for handle in [env._vehicle_handle, *env._carrier_handles, *wheels]:
            body = env._world.rigid_bodies[handle]
            body.translation = tuple(_vec(body.translation) + offset)
            body.linvel = (0., 0., 0.)
        _, _, terminated, truncated, info = env.step(np.array([-1, 0, -1, 0, 0, 0, 0], dtype=np.float32))
        assert not terminated and not truncated
        assert not info['telemetry']['ground_contact']
    finally:
        env.close()
