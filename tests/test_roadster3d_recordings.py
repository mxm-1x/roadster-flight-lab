import numpy as np
import pytest

from rocket_landing.envs.roadster3d import TELEMETRY_FIELDS_3D
from rocket_landing.roadster3d_export import compact_3d_episode


def make_recording(path, rewards=(1., -2.)):
    states = np.zeros((3, len(TELEMETRY_FIELDS_3D)), dtype=np.float32)
    index = {name: i for i, name in enumerate(TELEMETRY_FIELDS_3D)}
    states[:, index['x']] = [-16, -15, -14]
    states[:, index['y']] = [1, 2, 3]
    states[:, index['z']] = [0, 0.5, 1]
    states[:, index['rotation_qw']] = 1
    states[:, index['ramp_radius_m']] = 7.5
    states[:, index['landing_center_x']] = -30
    np.savez_compressed(path, telemetry=states, phases=np.array(['launch'] * 3),
                        actions=np.zeros((2, 7)), native_rewards=np.array(rewards),
                        shaped_rewards=np.array([2., -2.]), outcome=np.array('Crash'))


def test_export_preserves_recorded_3d_positions_and_returns(tmp_path):
    path = tmp_path / 'episode.npz'
    make_recording(path)
    replay = compact_3d_episode(path, variant='baseline', seed=17, episode=2, source='evaluation')
    assert replay['simulator_version'] == 6
    assert replay['source']['seed'] == 17
    assert replay['source']['episode'] == 2
    assert replay['rows'][-1][replay['columns'].index('z')] == 1
    assert replay['rows'][-1][replay['columns'].index('native_total')] == -1
    assert replay['native_return'] == -1
    assert replay['shaped_return'] == 0


def test_export_rejects_inconsistent_episode(tmp_path):
    path = tmp_path / 'episode.npz'
    make_recording(path, rewards=(1.,))
    with pytest.raises(ValueError, match='lengths'):
        compact_3d_episode(path, variant='baseline', seed=0, episode=1, source='evaluation')
