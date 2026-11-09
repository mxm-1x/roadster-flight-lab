"""Rapier-backed 3D rigid-body Roadster landing task (simulator v7).

This is a deliberately simplified, reproducible vehicle proxy. Rapier owns
gravity, rigid-body integration, contact resolution, ramp/ground collisions,
and the forces that produce the flight. The wheel motors, suspension, fan and
cold-gas tank are simplified models with assumed parameters. It is not a Tesla
vehicle model or a validated digital twin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import rapier3d as rp
from gymnasium import spaces
from rocket_landing.cold_gas import ColdGasBank


ENV_ID_3D = "RoadsterRampLanding3D-v7"
SIMULATOR_VERSION_3D = 7
PHYSICS_CONFIG_3D_PATH = Path(__file__).resolve().parents[1] / "configs" / "roadster_v7.json"
OBSERVATION_FIELDS_3D = (
    "target_dx_norm", "target_dy_norm", "target_dz_norm",
    "vx_norm", "vy_norm", "vz_norm",
    "rotation_qx", "rotation_qy", "rotation_qz", "rotation_qw",
    "omega_x_norm", "omega_y_norm", "omega_z_norm",
    "fuel_fraction", "flight_phase", "time_fraction",
    "valve_forward", "valve_lift", "valve_lateral", "valve_pitch", "valve_roll", "valve_yaw",
    "wheel_spin_0", "wheel_spin_1", "wheel_spin_2", "wheel_spin_3",
    "suspension_0", "suspension_1", "suspension_2", "suspension_3",
)
ACTION_FIELDS_3D = (
    "ramp_drive", "forward_thrust", "vertical_thrust", "lateral_thrust",
    "pitch_thrusters", "roll_thrusters", "yaw_thrusters",
)
TELEMETRY_FIELDS_3D = (
    "step", "x", "y", "z", "vx", "vy", "vz",
    "rotation_qx", "rotation_qy", "rotation_qz", "rotation_qw",
    "omega_x", "omega_y", "omega_z", "fuel", *ACTION_FIELDS_3D,
    "landing_center_x", "landing_center_z", "ramp_radius_m",
    "ramp_start_x", "ramp_end_x", "ramp_top_y", "elapsed_seconds",
    "gas_kg", "tank_pressure_pa", "ground_contact", "ramp_contact", "contact_impulse_ns",
    "actual_forward_n", "actual_lift_n", "actual_lateral_n", "actual_pitch_n", "actual_roll_n", "actual_yaw_n",
    *(f"wheel_{i}_{field}" for i in range(4) for field in ("x", "y", "z", "qx", "qy", "qz", "qw")),
)


@dataclass(frozen=True)
class RoadsterPhysics3D:
    """Versioned SI-unit assumptions for the new rigid-body environment."""

    version: int = SIMULATOR_VERSION_3D
    engine: str = "Rapier3D"
    python_engine_version: str = "0.36.1"
    javascript_engine_version: str = "0.21.0"
    dt_seconds: float = 1.0 / 60.0
    max_episode_steps: int = 1800
    max_launch_steps: int = 720
    gravity_mps2: float = 9.81
    vehicle_mass_kg: float = 1750.0
    vehicle_mass_jitter_kg: float = 150.0
    chassis_half_length_m: float = 2.05
    chassis_half_height_m: float = 0.34
    chassis_half_width_m: float = 0.92
    wheel_radius_m: float = 0.36
    wheel_x_offset_m: float = 1.35
    wheel_y_offset_m: float = -0.38
    wheel_z_offset_m: float = 0.82
    initial_vehicle_x_m: float = -16.0
    ramp_start_x_m: float = 0.0
    ramp_radius_m: float = 7.5
    ramp_radius_jitter_m: float = 0.3
    ramp_vehicle_clearance_m: float = 0.78
    ramp_vertical_stem_length_m: float = 8.0
    ramp_top_curve_radius_m: float = 5.0
    ramp_top_curve_angle_deg: float = 40.0
    ramp_segments: int = 128
    ramp_half_width_m: float = 4.5
    ramp_thickness_m: float = 0.55
    ramp_friction: float = 0.85
    ramp_friction_jitter: float = 0.15
    ground_friction: float = 0.9
    tire_friction: float = 0.95
    air_density_kg_m3: float = 1.225
    drag_coefficient_area_m2: float = 1.65
    launch_force_n: float = 12000.0
    launch_force_jitter_n: float = 1200.0
    wind_force_max_n: float = 350.0
    landing_center_x_m: float = 27.0
    landing_center_x_jitter_m: float = 2.0
    landing_center_z_jitter_m: float = 1.5
    landing_pad_half_length_m: float = 5.0
    landing_pad_half_width_m: float = 3.5
    forward_thrust_max_n: float = 10000.0
    vertical_thrust_max_n: float = 30000.0
    lateral_thrust_max_n: float = 5000.0
    pitch_thruster_max_n: float = 3000.0
    roll_thruster_max_n: float = 1800.0
    yaw_thruster_max_n: float = 1800.0
    nozzle_arm_m: float = 1.6
    initial_fuel_fraction: float = 1.0
    fuel_forward_per_second: float = 0.025
    fuel_vertical_per_second: float = 0.04
    fuel_lateral_per_second: float = 0.02
    fuel_attitude_per_second: float = 0.012
    safe_max_horizontal_speed_mps: float = 4.0
    safe_max_vertical_speed_mps: float = 2.5
    safe_max_tilt_rad: float = 0.22
    safe_max_spin_radps: float = 0.8
    touchdown_center_height_m: float = 0.92
    world_limit_x_m: float = 100.0
    world_limit_y_m: float = 100.0
    world_limit_z_m: float = 25.0
    training_gamma: float = 0.99
    timeout_penalty: float = -100.0
    crash_penalty: float = -100.0
    safe_landing_bonus: float = 150.0
    flawed_upright_tilt_rad: float = 0.1
    flawed_slow_speed_mps: float = 0.1
    flawed_bonus_per_step: float = 1.0
    shaping_scale: float = 0.5
    time_cost_per_step: float = 0.01
    potential_distance_weight: float = 1.0
    potential_speed_weight: float = 0.35
    potential_tilt_weight: float = 0.55
    potential_spin_weight: float = 0.12
    potential_fuel_weight: float = 0.08
    gas_initial_mass_kg: float = 80.0
    gas_initial_pressure_pa: float = 30_000_000.0
    gas_ambient_pressure_pa: float = 101_325.0
    gas_exhaust_velocity_mps: float = 500.0
    gas_valve_time_constant_s: float = 0.08
    wheel_motor_target_radps: float = 80.0
    wheel_motor_max_torque_nm: float = 9000.0
    suspension_stiffness_nm: float = 50000.0
    suspension_damping_nsm: float = 6000.0
    suspension_travel_m: float = 0.15
    fan_downforce_n: float = 25000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_config(cls, path: Path | None = None) -> "RoadsterPhysics3D":
        config_path = path or PHYSICS_CONFIG_3D_PATH
        values = json.loads(config_path.read_text())
        if values.get("version") != SIMULATOR_VERSION_3D:
            raise ValueError(
                f"Physics config {config_path} has version {values.get('version')}; "
                f"expected {SIMULATOR_VERSION_3D}."
            )
        return cls(**values)


def _vec(value: Any) -> np.ndarray:
    return np.asarray((float(value.x), float(value.y), float(value.z)), dtype=np.float64)


def _rotation_quaternion(rotation: Any) -> np.ndarray:
    """Return xyzw from Rapier's rotation vector without relying on binding internals."""
    axis_angle = _vec(rotation.scaled_axis)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-9:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    xyz = axis_angle * (np.sin(0.5 * angle) / angle)
    return np.asarray((*xyz, np.cos(0.5 * angle)), dtype=np.float64)


def _rotate(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    w = float(quaternion[3])
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _tilt_from_quaternion(quaternion: np.ndarray) -> float:
    up = _rotate(np.asarray((0.0, 1.0, 0.0)), quaternion)
    return float(np.arccos(np.clip(up[1], -1.0, 1.0)))


def _curved_ramp_mesh(
    radius_m: float,
    start_x_m: float,
    center_clearance_m: float,
    vertical_length_m: float,
    top_radius_m: float,
    top_angle_rad: float,
    half_width_m: float,
    thickness_m: float,
    segments: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build the Tesla-style ramp: a broad foot turn, tall straight stem and
    short curved shoulder near the top.

    The sampled reference path describes the car's approximate centerline. A
    triangle-mesh road surface is offset down from it by the ride clearance and
    thickened away from the vehicle. Rapier simulates contact and launch; this
    function defines only the fixed ramp geometry.
    """
    if segments < 32:
        raise ValueError("The ramp requires at least 32 mesh segments.")
    if radius_m <= 0 or center_clearance_m < 0 or half_width_m <= 0 or thickness_m <= 0:
        raise ValueError("Ramp dimensions must be positive (clearance may be zero).")
    if vertical_length_m < 0 or top_radius_m <= 0 or not 0 < top_angle_rad < np.pi / 2:
        raise ValueError("The ramp stem and upper curve are outside the supported range.")

    base_length = radius_m * np.pi / 2
    shoulder_length = top_radius_m * top_angle_rad
    total_length = base_length + vertical_length_m + shoulder_length
    base_segments = max(12, round(segments * base_length / total_length))
    stem_segments = max(4, round(segments * vertical_length_m / total_length))
    shoulder_segments = max(8, segments - base_segments - stem_segments)
    # Keep the mesh dense and deterministic after rounding the three sections.
    actual_segments = base_segments + stem_segments + shoulder_segments

    path: list[tuple[float, float, float]] = []
    for index in range(base_segments + 1):
        angle = (np.pi / 2) * index / base_segments
        path.append((
            radius_m * float(np.sin(angle)),
            radius_m + center_clearance_m - radius_m * float(np.cos(angle)),
            angle,
        ))
    for index in range(1, stem_segments + 1):
        fraction = index / stem_segments
        path.append((
            radius_m,
            radius_m + center_clearance_m + vertical_length_m * fraction,
            np.pi / 2,
        ))
    for index in range(1, shoulder_segments + 1):
        angle = (np.pi / 2) + top_angle_rad * index / shoulder_segments
        path.append((
            radius_m + top_radius_m * (float(np.sin(angle)) - 1.0),
            radius_m + center_clearance_m + vertical_length_m - top_radius_m * float(np.cos(angle)),
            angle,
        ))

    vertices: list[tuple[float, float, float]] = []
    for center_x, center_y, tangent_angle in path:
        # Left normal points from the road surface toward the vehicle center.
        normal_x, normal_y = -float(np.sin(tangent_angle)), float(np.cos(tangent_angle))
        center_x += start_x_m
        inner_x = center_x - center_clearance_m * normal_x
        inner_y = center_y - center_clearance_m * normal_y
        outer_x = inner_x - thickness_m * normal_x
        outer_y = inner_y - thickness_m * normal_y
        vertices.extend((
            (inner_x, inner_y, -half_width_m),
            (inner_x, inner_y, half_width_m),
            (outer_x, outer_y, -half_width_m),
            (outer_x, outer_y, half_width_m),
        ))

    faces: list[tuple[int, int, int]] = []
    for index in range(actual_segments):
        a = 4 * index
        b = a + 4
        # The first surface follows the car-side profile; the second closes
        # the outer shell. The matching segment edges remain collision-safe.
        faces.extend(((a, a + 1, b + 1), (a, b + 1, b)))
        # Outer surface and the two long edges close the solid ramp shell.
        faces.extend(((a + 2, b + 2, b + 3), (a + 2, b + 3, a + 3)))
        faces.extend(((a + 1, b + 3, b + 1), (a + 1, a + 3, b + 3)))
        faces.extend(((a, b, b + 2), (a, b + 2, a + 2)))
    last = 4 * actual_segments
    faces.extend(((0, 2, 3), (0, 3, 1)))
    faces.extend(((last, last + 1, last + 3), (last, last + 3, last + 2)))
    top_tangent = path[-1][2]
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
        float(path[-1][0] + start_x_m + center_clearance_m * np.sin(top_tangent)),
        float(path[-1][1] - center_clearance_m * np.cos(top_tangent)),
    )


class RoadsterLanding3DEnv(gym.Env[np.ndarray, np.ndarray]):
    """A full 3D, contact-driven Tesla-style ramp and booster landing task.

    The vehicle starts at rest on a real collider surface. PPO controls launch
    drive over a collidable rising ramp with a short curved shoulder; rigid-body
    contacts, gravity, mass and thrust determine takeoff. PPO commands six groups in
    flight. PPO owns the launch throttle and six flight-control channels. The
    browser demo uses Rapier's JS/WASM bindings and the same versioned scenario
    parameters.
    """

    metadata = {"render_modes": []}

    def __init__(self, physics: RoadsterPhysics3D | None = None):
        super().__init__()
        self.physics = physics or RoadsterPhysics3D.from_config()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(30,), dtype=np.float32)
        self._world: Any | None = None
        self._vehicle_handle: Any | None = None
        self._vehicle: Any | None = None
        self._step_count = 0
        self._launch_steps = 0
        self._phase = "launch"
        self._fuel = self.physics.initial_fuel_fraction
        self._target = np.asarray((self.physics.landing_center_x_m, 0.0, 0.0))
        self._wind = np.zeros(3)
        self._launch_force = self.physics.launch_force_n
        self._ramp_friction = self.physics.ramp_friction
        self._ramp_radius = self.physics.ramp_radius_m
        self._vehicle_mass = self.physics.vehicle_mass_kg
        self._ramp_stem_length = self.physics.ramp_vertical_stem_length_m
        self._ramp_top_radius = self.physics.ramp_top_curve_radius_m
        self._ramp_top_angle = float(np.deg2rad(self.physics.ramp_top_curve_angle_deg))
        self._ramp_start_x = self._ramp_end_x = self._ramp_top_y = 0.0
        self._landing_x_error: float | None = None
        self._landing_z_error: float | None = None
        self._touchdown_speed: float | None = None
        self._touchdown_tilt: float | None = None
        self._last_action = np.zeros(7, dtype=np.float32)
        self._outcome = "In progress"
        self._gas: ColdGasBank | None = None
        self._ground_collider: Any = None
        self._ramp_collider: Any = None
        self._chassis_collider: Any = None
        self._wheel_colliders: list[Any] = []
        self._wheel_joints: list[Any] = []
        self._carrier_handles: list[Any] = []
        self._ramp_seen = False
        self._highest_ramp_y = 0.0
        self._settle_steps = 0
        self._contact_impulse = 0.0
        self._actual_forces = np.zeros(6)

    @property
    def config(self) -> RoadsterPhysics3D:
        return self.physics

    def _add_static(self, collider: Any) -> Any:
        assert self._world is not None
        parent = self._world.add_body(rp.RigidBody.fixed())
        return self._world.add_collider(collider, parent=parent)

    def _build_world(self, seed: int) -> None:
        p = self.physics
        rng = self.np_random
        if self._world is None:
            self._world = rp.PhysicsWorld(gravity=(0.0, -p.gravity_mps2, 0.0))
            integration = self._world.integration_parameters
            integration.dt = p.dt_seconds
            integration.num_solver_iterations = 12
            integration.max_ccd_substeps = 4
        else:
            # Reuse the engine's thread pool across episodes while rebuilding
            # the randomized scenario geometry and bodies.
            self._world.clear()

        pad_x = p.landing_center_x_m + float(rng.uniform(-p.landing_center_x_jitter_m, p.landing_center_x_jitter_m))
        pad_z = float(rng.uniform(-p.landing_center_z_jitter_m, p.landing_center_z_jitter_m))
        self._target = np.asarray((pad_x, 0.0, pad_z), dtype=np.float64)
        self._wind = np.asarray(
            (rng.uniform(-p.wind_force_max_n, p.wind_force_max_n), 0.0,
             rng.uniform(-p.wind_force_max_n, p.wind_force_max_n)), dtype=np.float64
        )
        self._launch_force = p.launch_force_n + float(rng.uniform(-p.launch_force_jitter_n, p.launch_force_jitter_n))
        self._ramp_radius = p.ramp_radius_m + float(
            rng.uniform(-p.ramp_radius_jitter_m, p.ramp_radius_jitter_m)
        )
        self._ramp_friction = p.ramp_friction + float(rng.uniform(-p.ramp_friction_jitter, p.ramp_friction_jitter))

        # A single ground collision surface. The marked landing zone is a
        # region on it, avoiding overlapping static colliders at touchdown.
        ground_groups = rp.InteractionGroups(rp.Group.GROUP_0, rp.Group.GROUP_1)
        vehicle_groups = rp.InteractionGroups(rp.Group.GROUP_1, rp.Group.GROUP_0)
        self._ground_collider = self._add_static(
            rp.Collider.cuboid(120.0, 0.5, 30.0)
            .translation((0.0, -0.5, 0.0))
            .friction(p.ground_friction)
            .restitution(0.0)
            .collision_groups(ground_groups)
            .build()
        )

        vertices, indices, self._ramp_end_x, self._ramp_top_y = _curved_ramp_mesh(
            self._ramp_radius,
            p.ramp_start_x_m,
            p.ramp_vehicle_clearance_m,
            p.ramp_vertical_stem_length_m,
            p.ramp_top_curve_radius_m,
            float(np.deg2rad(p.ramp_top_curve_angle_deg)),
            p.ramp_half_width_m,
            p.ramp_thickness_m,
            p.ramp_segments,
        )
        ramp_body = self._world.add_body(rp.RigidBody.fixed())
        ramp_shape = (
            rp.Collider.trimesh(vertices, indices, flags=rp.TriMeshFlags.FIX_INTERNAL_EDGES)
            .friction(self._ramp_friction)
            .restitution(0.0)
            .collision_groups(ground_groups)
            .build()
        )
        self._ramp_collider = self._world.add_collider(ramp_shape, parent=ramp_body)
        self._ramp_start_x = p.ramp_start_x_m

        mass = p.vehicle_mass_kg + float(rng.uniform(-p.vehicle_mass_jitter_kg, p.vehicle_mass_jitter_kg))
        self._vehicle_mass = mass
        self._vehicle_handle = self._world.add_body(
            rp.RigidBody.dynamic()
            .translation((p.initial_vehicle_x_m, 0.78, float(rng.uniform(-0.25, 0.25))))
            .linvel((0.0, 0.0, 0.0))
            .angvel((0.0, 0.0, 0.0))
            .linear_damping(0.015)
            .angular_damping(0.08)
            .additional_mass(p.gas_initial_mass_kg)
            .can_sleep(False)
            .ccd_enabled(True)
        )
        chassis = (
            rp.Collider.cuboid(p.chassis_half_length_m, p.chassis_half_height_m, p.chassis_half_width_m)
            .mass(mass - 80.0)
            .friction(0.65)
            .restitution(0.05)
            .collision_groups(vehicle_groups)
            .build()
        )
        self._chassis_collider = self._world.add_collider(chassis, parent=self._vehicle_handle)
        self._wheel_colliders = []
        self._wheel_joints = []
        self._carrier_handles = []
        # Each wheel has a suspended carrier and an independently rotating axle.
        for x in (-p.wheel_x_offset_m, p.wheel_x_offset_m):
            for z in (-p.wheel_z_offset_m, p.wheel_z_offset_m):
                wheel_pos = (p.initial_vehicle_x_m + x, 0.78 + p.wheel_y_offset_m, z + float(self._world.rigid_bodies[self._vehicle_handle].translation.z))
                carrier = self._world.add_body(rp.RigidBody.dynamic().translation(wheel_pos).can_sleep(False))
                self._carrier_handles.append(carrier)
                self._world.add_collider(rp.Collider.ball(0.08).mass(5.0).collision_groups(rp.InteractionGroups.none()).build(), parent=carrier)
                suspension = (rp.PrismaticJointBuilder((0.0, 1.0, 0.0))
                    .local_anchor1((x, p.wheel_y_offset_m, z)).local_anchor2((0.0, 0.0, 0.0))
                    .limits(-p.suspension_travel_m, p.suspension_travel_m)
                    .motor_position(0.0, p.suspension_stiffness_nm, p.suspension_damping_nsm)
                    .contacts_enabled(False).build())
                self._world.impulse_joints.insert(self._vehicle_handle, carrier, suspension)
                wheel = self._world.add_body(rp.RigidBody.dynamic().translation(wheel_pos).can_sleep(False).ccd_enabled(True))
                tire = (
                    rp.Collider.cylinder(0.14, p.wheel_radius_m)
                    .rotation((np.pi / 2, 0.0, 0.0))
                    .mass(15.0)
                    .friction(p.tire_friction)
                    .restitution(0.02)
                    .collision_groups(vehicle_groups)
                    .build()
                )
                self._wheel_colliders.append(self._world.add_collider(tire, parent=wheel))
                axle = (rp.RevoluteJointBuilder((0.0, 0.0, 1.0))
                    .motor_velocity(0.0, 100.0).motor_max_force(p.wheel_motor_max_torque_nm)
                    .motor_model(rp.MotorModel.FORCE_BASED)
                    .contacts_enabled(False).build())
                self._wheel_joints.append(self._world.impulse_joints.insert(carrier, wheel, axle))
        self._vehicle = self._world.rigid_bodies[self._vehicle_handle]
        self._vehicle.recompute_mass_properties_from_colliders(self._world.colliders)
        self._gas = ColdGasBank(p.gas_initial_mass_kg, p.gas_initial_pressure_pa, p.gas_ambient_pressure_pa,
                                p.gas_exhaust_velocity_mps, p.gas_valve_time_constant_s,
                                np.asarray((p.forward_thrust_max_n, p.vertical_thrust_max_n, p.lateral_thrust_max_n,
                                            p.pitch_thruster_max_n, p.roll_thruster_max_n, p.yaw_thruster_max_n)))

    def close(self) -> None:
        if self._world is not None:
            self._world.clear()
            self._world = None
            self._vehicle_handle = None
            self._vehicle = None
        super().close()

    def _state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._vehicle is not None
        position = _vec(self._vehicle.translation)
        velocity = _vec(self._vehicle.linvel)
        angular = _vec(self._vehicle.angvel)
        quaternion = _rotation_quaternion(self._vehicle.rotation)
        return position, velocity, angular, quaternion, np.asarray(self._target)

    def _observation(self) -> np.ndarray:
        p = self.physics
        position, velocity, angular, quaternion, target = self._state()
        elapsed = self._step_count * p.dt_seconds
        assert self._world is not None and self._gas is not None
        inverse_q = np.asarray((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]))
        wheel_speeds = []
        suspension = []
        for joint_handle, carrier_handle in zip(self._wheel_joints, self._carrier_handles):
            joint = self._world.impulse_joints[joint_handle]
            wheel = self._world.rigid_bodies[joint.body2]
            wheel_speeds.append(np.clip(_rotate(_vec(wheel.angvel), inverse_q)[2] / p.wheel_motor_target_radps, -1, 1))
            relative = _rotate(_vec(self._world.rigid_bodies[carrier_handle].translation) - position, inverse_q)
            suspension.append(np.clip((relative[1] - p.wheel_y_offset_m) / p.suspension_travel_m, -1, 1))
        return np.asarray(
            [
                np.clip((target[0] - position[0]) / 45.0, -1, 1),
                np.clip((target[1] - position[1]) / 20.0, -1, 1),
                np.clip((target[2] - position[2]) / 20.0, -1, 1),
                np.clip(velocity[0] / 35.0, -1, 1),
                np.clip(velocity[1] / 35.0, -1, 1),
                np.clip(velocity[2] / 35.0, -1, 1),
                *quaternion,
                np.clip(angular[0] / 8.0, -1, 1),
                np.clip(angular[1] / 8.0, -1, 1),
                np.clip(angular[2] / 8.0, -1, 1),
                np.clip(self._fuel, 0.0, 1.0),
                1.0 if self._phase == "flight" else 0.0,
                np.clip(elapsed / (p.max_episode_steps * p.dt_seconds), 0.0, 1.0),
                *self._gas.valves, *wheel_speeds, *suspension,
            ], dtype=np.float32,
        )

    def telemetry(self) -> dict[str, float | str]:
        position, velocity, angular, quaternion, _ = self._state()
        physics = self.physics
        result: dict[str, float | str] = {
            "step": float(self._step_count), "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
            "vx": float(velocity[0]), "vy": float(velocity[1]), "vz": float(velocity[2]),
            "rotation_qx": float(quaternion[0]), "rotation_qy": float(quaternion[1]),
            "rotation_qz": float(quaternion[2]), "rotation_qw": float(quaternion[3]),
            "omega_x": float(angular[0]), "omega_y": float(angular[1]), "omega_z": float(angular[2]),
            "fuel": float(self._fuel), "phase": self._phase,
            **{name: float(value) for name, value in zip(ACTION_FIELDS_3D, self._last_action)},
            "landing_center_x": float(self._target[0]), "landing_center_z": float(self._target[2]),
            "ramp_radius_m": float(self._ramp_radius), "ramp_start_x": float(self._ramp_start_x),
            "ramp_end_x": float(self._ramp_end_x), "ramp_top_y": float(self._ramp_top_y),
            "elapsed_seconds": float(self._step_count * physics.dt_seconds),
            "gas_kg": float(self._gas.gas_kg if self._gas else 0.0),
            "tank_pressure_pa": float(self._gas.pressure_pa if self._gas else 0.0),
            "ground_contact": float(self._has_contact(self._ground_collider)),
            "ramp_contact": float(self._has_contact(self._ramp_collider)),
            "contact_impulse_ns": float(self._contact_impulse),
        }
        result.update({key: float(value) for key, value in zip(
            ("actual_forward_n", "actual_lift_n", "actual_lateral_n", "actual_pitch_n", "actual_roll_n", "actual_yaw_n"),
            self._actual_forces,
        )})
        assert self._world is not None
        for index, joint_handle in enumerate(self._wheel_joints):
            wheel = self._world.rigid_bodies[self._world.impulse_joints[joint_handle].body2]
            xyz, q = _vec(wheel.translation), _rotation_quaternion(wheel.rotation)
            for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), (*xyz, *q)):
                result[f"wheel_{index}_{key}"] = float(value)
        return result

    def _info(self) -> dict[str, Any]:
        return {
            "telemetry": self.telemetry(), "phase": self._phase,
            "landing_center_x": float(self._target[0]), "landing_center_z": float(self._target[2]),
            "landing_x_error": self._landing_x_error, "landing_z_error": self._landing_z_error,
            "touchdown_speed": self._touchdown_speed, "touchdown_tilt": self._touchdown_tilt,
            "gas_remaining": float(self._fuel), "outcome": self._outcome,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._step_count = self._launch_steps = 0
        self._phase = "launch"
        self._fuel = self.physics.initial_fuel_fraction
        self._landing_x_error = self._landing_z_error = None
        self._touchdown_speed = self._touchdown_tilt = None
        self._last_action = np.zeros(7, dtype=np.float32)
        self._outcome = "In progress"
        self._ramp_seen = False
        self._highest_ramp_y = 0.0
        self._settle_steps = 0
        self._contact_impulse = 0.0
        self._actual_forces = np.zeros(6)
        self._build_world(seed or 0)
        if options and options.get("phase") == "flight":
            assert self._world is not None and self._vehicle is not None and self._vehicle_handle is not None
            target = self._target
            profile = int(options.get("flight_profile", 2))
            if profile == 0:
                height_range, down_range, horizontal_jitter, position_jitter = (4.0, 4.5), (-0.05, 0.05), 0.0, 0.0
            elif profile == 1:
                height_range, down_range, horizontal_jitter, position_jitter = (3.5, 4.5), (-0.3, -0.1), 0.2, 0.5
            elif profile == 2:
                height_range, down_range, horizontal_jitter, position_jitter = (2.5, 4.0), (-1.0, -0.2), 0.5, 1.5
            else:
                raise ValueError("flight_profile must be 0 (easy), 1 (medium), or 2 (original).")
            spawn = np.asarray((target[0] + self.np_random.uniform(-position_jitter, position_jitter),
                                self.np_random.uniform(*height_range),
                                target[2] + self.np_random.uniform(-position_jitter, position_jitter)))
            shift = spawn - _vec(self._vehicle.translation)
            initial_velocity = (float(self.np_random.uniform(-horizontal_jitter, horizontal_jitter)),
                                float(self.np_random.uniform(*down_range)),
                                float(self.np_random.uniform(-horizontal_jitter, horizontal_jitter)))
            wheel_handles = [self._world.impulse_joints[h].body2 for h in self._wheel_joints]
            for handle in [self._vehicle_handle, *self._carrier_handles, *wheel_handles]:
                body = self._world.rigid_bodies[handle]
                body.translation = tuple(float(v) for v in _vec(body.translation) + shift)
                body.linvel = initial_velocity
            self._phase = "flight"
        elif options and options.get("phase") not in (None, "launch"):
            raise ValueError("phase must be 'launch' or 'flight'")
        return self._observation(), self._info()

    def _world_point(self, local: np.ndarray, position: np.ndarray, quaternion: np.ndarray) -> tuple[float, float, float]:
        return tuple(float(v) for v in position + _rotate(local, quaternion))

    def _has_contact(self, surface: Any, *, chassis_only: bool = False) -> bool:
        if self._world is None or surface is None or self._chassis_collider is None:
            return False
        bodies = [self._chassis_collider] if chassis_only else [self._chassis_collider, *self._wheel_colliders]
        for collider in bodies:
            pair = self._world.narrow_phase.contact_pair(collider, surface)
            if pair is not None and pair.has_any_active_contact:
                return True
        return False

    def _surface_impulse(self, surface: Any) -> float:
        if self._world is None or surface is None:
            return 0.0
        result = 0.0
        for collider in [self._chassis_collider, *self._wheel_colliders]:
            pair = self._world.narrow_phase.contact_pair(collider, surface)
            if pair is not None and pair.has_any_active_contact:
                result += float(pair.total_impulse_magnitude())
        return result

    def _apply_thrusters(self, control: np.ndarray, position: np.ndarray, quaternion: np.ndarray) -> float:
        """Apply bounded nozzle forces at physical attachment points."""
        p = self.physics
        assert self._vehicle is not None
        forward, vertical_command, lateral, pitch, roll, yaw = (float(v) for v in control)
        lift = 0.5 * (vertical_command + 1.0)
        assert self._gas is not None
        force_values, gas_used = self._gas.step(np.asarray((forward, lift, lateral, pitch, roll, yaw)), p.dt_seconds)
        self._actual_forces = force_values.copy()
        forward_f, vertical_f, lateral_f, pitch_f, roll_f, yaw_f = force_values
        self._vehicle.reset_forces()
        self._vehicle.reset_torques()

        local_forces = (
            (np.asarray((forward_f, 0.0, 0.0)), np.zeros(3)),
            (np.asarray((0.0, vertical_f, 0.0)), np.zeros(3)),
            (np.asarray((0.0, 0.0, lateral_f)), np.zeros(3)),
            (np.asarray((0.0, pitch_f / 2, 0.0)), np.asarray((p.nozzle_arm_m, 0.0, 0.0))),
            (np.asarray((0.0, -pitch_f / 2, 0.0)), np.asarray((-p.nozzle_arm_m, 0.0, 0.0))),
            (np.asarray((0.0, roll_f / 2, 0.0)), np.asarray((0.0, 0.0, p.nozzle_arm_m))),
            (np.asarray((0.0, -roll_f / 2, 0.0)), np.asarray((0.0, 0.0, -p.nozzle_arm_m))),
            (np.asarray((yaw_f / 2, 0.0, 0.0)), np.asarray((0.0, 0.0, p.nozzle_arm_m))),
            (np.asarray((-yaw_f / 2, 0.0, 0.0)), np.asarray((0.0, 0.0, -p.nozzle_arm_m))),
        )
        for local_force, local_point in local_forces:
            force = _rotate(local_force, quaternion)
            point = self._world_point(local_point, position, quaternion)
            self._vehicle.add_force_at_point(tuple(float(v) for v in force), point)
        self._vehicle.set_additional_mass(self._gas.gas_kg, True)
        self._fuel = self._gas.gas_kg / p.gas_initial_mass_kg
        return gas_used / p.gas_initial_mass_kg

    def step(self, action: np.ndarray):
        p = self.physics
        control = np.asarray(action, dtype=np.float32).reshape(7)
        if not np.all(np.isfinite(control)):
            raise ValueError("Roadster action must contain only finite values.")
        control = np.clip(control, -1.0, 1.0)
        assert self._world is not None and self._vehicle is not None

        previous_position, previous_velocity, _, previous_q, target = self._state()
        previous_distance = float(np.linalg.norm(target - previous_position))
        applied = np.zeros(7, dtype=np.float32)
        fuel_used = 0.0
        self._step_count += 1

        if self._phase == "launch":
            self._launch_steps += 1
            self._actual_forces.fill(0.0)
            # Electric wheel motors supply drive torque through real contacts.
            launch_throttle = 0.5 * (float(control[0]) + 1.0)
            applied[0] = control[0]
            self._vehicle.reset_forces()
            self._vehicle.reset_torques()
            for handle in self._wheel_joints:
                joint = self._world.impulse_joints[handle]
                joint_data = joint.data
                joint_data.set_motor_velocity(rp.JointAxis.ANG_X, -p.wheel_motor_target_radps, 100.0)
                joint_data.set_motor_max_force(rp.JointAxis.ANG_X, p.wheel_motor_max_torque_nm * launch_throttle)
                joint.data = joint_data
            # A simplified fan applies normal downforce only while touching a
            # surface. Fan flow, ground seal and electrical energy are not modeled.
            if self._has_contact(self._ramp_collider) or self._has_contact(self._ground_collider):
                fan = _rotate(np.asarray((0.0, -p.fan_downforce_n, 0.0)), previous_q)
                self._vehicle.add_force(tuple(float(v) for v in fan))
            self._vehicle.add_force(tuple(float(v) for v in self._wind))
        else:
            applied[0] = -1.0
            applied[1:] = control[1:]
            fuel_used = self._apply_thrusters(control[1:], previous_position, previous_q)
            self._vehicle.add_force(tuple(float(v) for v in self._wind))
            velocity = _vec(self._vehicle.linvel)
            speed = float(np.linalg.norm(velocity))
            drag = -0.5 * p.air_density_kg_m3 * p.drag_coefficient_area_m2 * speed * velocity
            self._vehicle.add_force(tuple(float(v) for v in drag))

        self._world.step()
        position, velocity, angular, quaternion, target = self._state()
        ground_contact = self._has_contact(self._ground_collider)
        ramp_contact = self._has_contact(self._ramp_collider)
        chassis_ground = self._has_contact(self._ground_collider, chassis_only=True)
        self._contact_impulse = self._surface_impulse(self._ground_collider) + self._surface_impulse(self._ramp_collider)
        distance = float(np.linalg.norm(target - position))
        speed = float(np.linalg.norm(velocity))
        tilt = _tilt_from_quaternion(quaternion)
        reward = (
            0.08 * (previous_distance - distance)
            - 0.003 * speed
            - 0.008 * tilt
            - 0.003 * float(np.linalg.norm(angular))
            - 0.08 * fuel_used
        )

        terminated = truncated = False
        if self._phase == "launch":
            self._ramp_seen = self._ramp_seen or ramp_contact
            if ramp_contact:
                self._highest_ramp_y = max(self._highest_ramp_y, float(position[1]))
            launched = (
                self._ramp_seen and not ramp_contact and not ground_contact
                and self._highest_ramp_y >= self._ramp_top_y - 4.0
                and position[1] >= self._ramp_top_y - 2.5
                and position[0] <= self._ramp_end_x + 1.5
                and velocity[0] < -0.5 and velocity[1] > 0.0
            )
            launch_failed = self._launch_steps >= p.max_launch_steps
            if (self._has_contact(self._ramp_collider, chassis_only=True)
                    and float(np.linalg.norm(previous_velocity)) > p.safe_max_horizontal_speed_mps):
                self._outcome = "Crash"
                reward = p.crash_penalty
                terminated = True
            elif launched:
                self._phase = "flight"
                # The policy switches from ramp drive to six flight controls.
            elif ground_contact and self._highest_ramp_y > 3.0 and position[1] < 2.0 and previous_velocity[1] < -2.0:
                self._outcome = "Crash"
                reward = p.crash_penalty
                terminated = True
            elif launch_failed:
                self._outcome = "Launch failure"
                reward = p.crash_penalty
                terminated = True
        else:
            if ramp_contact:
                self._outcome = "Crash"
                reward += p.crash_penalty
                terminated = True
            elif ground_contact:
                self._landing_x_error = abs(float(position[0] - target[0]))
                self._landing_z_error = abs(float(position[2] - target[2]))
                if self._touchdown_speed is None:
                    self._touchdown_speed = float(np.linalg.norm(previous_velocity))
                    self._touchdown_tilt = _tilt_from_quaternion(previous_q)
                safe = (
                    not chassis_ground
                    and
                    self._landing_x_error <= p.landing_pad_half_length_m
                    and self._landing_z_error <= p.landing_pad_half_width_m
                    and float(np.linalg.norm(previous_velocity[[0, 2]])) <= p.safe_max_horizontal_speed_mps
                    and abs(float(previous_velocity[1])) <= p.safe_max_vertical_speed_mps
                    and self._touchdown_tilt <= p.safe_max_tilt_rad
                    and float(np.linalg.norm(angular)) <= p.safe_max_spin_radps
                )
                if not safe:
                    self._outcome = "Crash"
                    reward += p.crash_penalty
                    terminated = True
                else:
                    self._settle_steps += 1
                    if self._settle_steps >= 30:
                        self._outcome = "Safe landing"
                        reward += p.safe_landing_bonus
                        terminated = True
            else:
                self._settle_steps = 0

        outside = (
            abs(position[0]) > p.world_limit_x_m or position[1] < -10.0
            or position[1] > p.world_limit_y_m or abs(position[2]) > p.world_limit_z_m
        )
        if outside and not (terminated or truncated):
            self._outcome = "Crash"
            reward += p.crash_penalty
            terminated = True
        if self._step_count >= p.max_episode_steps and not (terminated or truncated):
            self._outcome = "Timeout"
            reward += p.timeout_penalty
            truncated = True

        self._last_action = applied
        info = self._info()
        info.update({
            "base_reward": float(reward), "fuel_used": float(fuel_used),
            "safe_landing": self._outcome == "Safe landing",
            "crash": self._outcome in {"Crash", "Launch failure"},
            "truncated": truncated, "action_applied": applied.copy(),
            "tilt_rad": tilt, "wind_force_n": self._wind.tolist(),
            "vehicle_mass_kg": float(self._vehicle_mass + (self._gas.gas_kg if self._gas else 0.0)),
        })
        return self._observation(), float(reward), terminated, truncated, info


if ENV_ID_3D not in gym.registry:
    gym.register(id=ENV_ID_3D, entry_point="rocket_landing.envs.roadster3d_v7:RoadsterLanding3DEnv")
