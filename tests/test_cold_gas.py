import numpy as np
import pytest

from rocket_landing.cold_gas import ColdGasBank


def bank(mass=2.0):
    return ColdGasBank(mass, 20e6, 101325, 500, 0.05, np.array([1000., 2000.]))


def test_impulse_matches_expelled_gas_and_pressure_decreases():
    unit = bank()
    old_pressure = unit.pressure_pa
    force, used = unit.step(np.array([1., -1.]), 0.1)
    assert force[0] > 0 > force[1]
    assert used == pytest.approx(np.abs(force).sum() * 0.1 / 500)
    assert unit.gas_kg == pytest.approx(2.0 - used)
    assert unit.pressure_pa < old_pressure


def test_thrust_cannot_consume_more_than_remaining_gas():
    unit = bank(mass=0.001)
    force, used = unit.step(np.ones(2), 1.0)
    assert used == pytest.approx(0.001)
    assert unit.gas_kg == 0
    assert np.abs(force).sum() == pytest.approx(used * 500)
    next_force, next_used = unit.step(np.ones(2), 0.1)
    assert next_used == 0
    assert np.all(next_force == 0)


def test_valve_lag_and_reset():
    unit = bank()
    force, _ = unit.step(np.array([1., 0.]), 0.01)
    assert 0 < force[0] < 1000
    unit.reset()
    assert unit.gas_kg == 2
    assert np.all(unit.valves == 0)
