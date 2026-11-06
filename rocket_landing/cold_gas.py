"""A bounded, simplified cold-gas thruster bank in SI units.

The tank follows an isothermal ideal-gas blowdown approximation. Parameters are
research assumptions, not specifications of a particular production vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ColdGasBank:
    initial_gas_kg: float
    initial_pressure_pa: float
    ambient_pressure_pa: float
    effective_exhaust_velocity_mps: float
    valve_time_constant_s: float
    max_thrust_n: np.ndarray
    gas_kg: float = field(init=False)
    valves: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.max_thrust_n = np.asarray(self.max_thrust_n, dtype=np.float64).copy()
        if (self.initial_gas_kg <= 0 or self.initial_pressure_pa <= self.ambient_pressure_pa
                or self.ambient_pressure_pa < 0 or self.effective_exhaust_velocity_mps <= 0
                or self.valve_time_constant_s <= 0 or self.max_thrust_n.ndim != 1
                or not np.isfinite(self.max_thrust_n).all() or np.any(self.max_thrust_n < 0)):
            raise ValueError("Invalid cold-gas bank parameters")
        self.reset()

    def reset(self) -> None:
        self.gas_kg = float(self.initial_gas_kg)
        self.valves = np.zeros_like(self.max_thrust_n)

    @property
    def pressure_pa(self) -> float:
        return self.initial_pressure_pa * self.gas_kg / self.initial_gas_kg

    def step(self, commands: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """Return signed nozzle forces and expelled mass for one timestep.

        Negative commands select the opposite nozzle in a paired axis. Pressure
        and valve dynamics reduce actual thrust; total impulse is limited by gas.
        """
        commands = np.asarray(commands, dtype=np.float64)
        if commands.shape != self.max_thrust_n.shape or not np.isfinite(commands).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Commands and timestep must be finite and valid")
        commands = np.clip(commands, -1, 1)
        alpha = -np.expm1(-dt / self.valve_time_constant_s)
        self.valves += alpha * (commands - self.valves)
        pressure_ratio = max(0., (self.pressure_pa - self.ambient_pressure_pa)
                             / (self.initial_pressure_pa - self.ambient_pressure_pa))
        force = self.max_thrust_n * self.valves * pressure_ratio
        requested_mass = float(np.abs(force).sum() * dt / self.effective_exhaust_velocity_mps)
        if requested_mass > self.gas_kg and requested_mass > 0:
            force *= self.gas_kg / requested_mass
            requested_mass = self.gas_kg
        self.gas_kg = max(0., self.gas_kg - requested_mass)
        return force, requested_mass
