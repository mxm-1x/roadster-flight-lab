"""Replay saved policies and render a 3D view of the 2D simulator's state."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
from stable_baselines3 import PPO

from rocket_landing.experiment import _classify_episode, make_env


def load_training_episode(path: Path | str) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Load the exact states and actions recorded during a training attempt."""
    with np.load(path, allow_pickle=False) as archive:
        states = archive["states"]
        actions = archive["actions"]
        native = archive["native_rewards"]
        shaped = archive["shaped_rewards"]
        if states.shape != (len(actions) + 1, 8) or len(native) != len(actions) or len(shaped) != len(actions):
            raise ValueError("Invalid training trajectory: state and action counts do not match.")
        flight = pd.DataFrame(states, columns=["x", "height", "vx", "vy", "angle", "spin", "left_leg", "right_leg"])
        flight["step"] = np.arange(len(states))
        flight["action"] = np.r_[-1, actions]
        flight["native_total"] = np.r_[0, np.cumsum(native, dtype=np.float64)]
        flight["shaped_total"] = np.r_[0, np.cumsum(shaped, dtype=np.float64)]
        flight.attrs["outcome"] = str(archive["outcome"].item())
        pad = tuple(float(v) for v in archive["pad"])
        return flight, pad


def replay_episode(model_path: Path | str, variant: str, episode_seed: int) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Re-run one seeded episode without changing any saved evaluation files."""
    model = PPO.load(Path(model_path), device="cpu")
    env = make_env(variant)
    try:
        observation, _ = env.reset(seed=episode_seed)
        base = env.unwrapped
        center = VIEWPORT_W / SCALE / 2
        pad = (float(base.helipad_x1 - center), float(base.helipad_x2 - center))
        rows = []
        native_total = shaped_total = 0.0
        terminated = truncated = False
        action = -1
        last_native_reward = 0.0
        step = 0
        while True:
            rows.append({
                "step": step, "x": float(observation[0]), "height": float(observation[1]),
                "vx": float(observation[2]), "vy": float(observation[3]),
                "angle": float(observation[4]), "spin": float(observation[5]),
                "left_leg": float(observation[6]), "right_leg": float(observation[7]),
                "action": action, "native_total": native_total, "shaped_total": shaped_total,
            })
            if terminated or truncated:
                break
            chosen, _ = model.predict(observation, deterministic=True)
            action = int(np.asarray(chosen).item())
            observation, reward, terminated, truncated, info = env.step(action)
            last_native_reward = float(info.get("base_reward", reward))
            native_total += last_native_reward
            shaped_total += float(reward)
            step += 1
        safe, crash, timeout = _classify_episode(last_native_reward, terminated, truncated)
        result = pd.DataFrame(rows)
        result.attrs["outcome"] = "Safe landing" if safe else "Crash" if crash else "Timeout" if timeout else "Unknown"
        return result, pad
    finally:
        env.close()


def _world_positions(flight: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = flight["x"].to_numpy() * (VIEWPORT_W / SCALE / 2)
    z = flight["height"].to_numpy() * (VIEWPORT_H / SCALE / 2) + LEG_DOWN / SCALE
    return x, z


def _rocket_traces(x: float, z: float, angle: float, action: int) -> list:
    """Make an illustrative rocket whose pose follows the simulator observation."""
    radius, bottom, shoulder, nose = 0.43, -0.30, 0.34, 0.63
    ring = np.linspace(0, 2 * np.pi, 9)[:-1]
    local_x = np.r_[radius * np.cos(ring), radius * np.cos(ring), 0.0]
    local_y = np.r_[radius * np.sin(ring), radius * np.sin(ring), 0.0]
    local_z = np.r_[np.full(8, bottom), np.full(8, shoulder), nose]
    c, s = np.cos(angle), np.sin(angle)
    world_x = x + local_x * c - local_z * s
    world_z = z + local_x * s + local_z * c
    i, j, k = [], [], []
    for n in range(8):
        nxt = (n + 1) % 8
        i.extend((n, n, 8 + n, 0))
        j.extend((nxt, 8 + nxt, 8 + nxt, nxt))
        k.extend((8 + nxt, 8 + n, 16, n))
    body = go.Mesh3d(x=world_x, y=local_y, z=world_z, i=i, j=j, k=k,
                     color="#ececec", flatshading=True, name="Rocket",
                     lighting=dict(ambient=0.65, diffuse=0.7, specular=0.5))

    def transform(points):
        p = np.asarray(points, dtype=float)
        return x + p[:, 0] * c - p[:, 2] * s, p[:, 1], z + p[:, 0] * s + p[:, 2] * c

    lx, ly, lz = transform([(-0.3, 0, -0.2), (-0.65, 0, -0.6), (0.3, 0, -0.2), (0.65, 0, -0.6)])
    legs = go.Scatter3d(x=[lx[0], lx[1], None, lx[2], lx[3]], y=[ly[0], ly[1], None, ly[2], ly[3]],
                        z=[lz[0], lz[1], None, lz[2], lz[3]], mode="lines", line=dict(color="#c1c1c1", width=6),
                        name="Landing legs", showlegend=False)
    if action == 2:
        fx, fy, fz = transform([(0, 0, -0.3), (0, 0, -0.9)])
    elif action in (1, 3):
        direction = -1 if action == 1 else 1
        fx, fy, fz = transform([(direction * 0.4, 0, 0), (direction * 0.85, 0, 0)])
    else:
        fx, fy, fz = [], [], []
    flame = go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="#a2a2a2", width=10),
                         name="Engine", showlegend=False)
    marker = go.Scatter3d(x=[x], y=[0], z=[z], mode="markers", marker=dict(size=2, color="#ffffff"), showlegend=False)
    return [body, legs, flame, marker]


def flight_figure(flight: pd.DataFrame, pad: tuple[float, float], max_frames: int = 120) -> go.Figure:
    """Animate a 3D illustration of actual x, height, angle and engine actions."""
    if flight.empty:
        raise ValueError("A flight must contain at least one state.")
    x, z = _world_positions(flight)
    positions = np.unique(np.linspace(0, len(flight) - 1, min(max_frames, len(flight)), dtype=int))
    def rocket_at(index):
        row = flight.iloc[index]
        return _rocket_traces(float(x[index]), float(z[index]), float(row["angle"]), int(row["action"]))

    ground = go.Mesh3d(x=[pad[0], pad[1], pad[1], pad[0]], y=[-0.8, -0.8, 0.8, 0.8], z=[0, 0, 0, 0],
                       i=[0, 0], j=[1, 2], k=[2, 3], color="#a5a5a5", opacity=0.9, name="Landing pad")
    trail = go.Scatter3d(x=x, y=np.zeros_like(x), z=z, mode="lines", line=dict(color="#999999", width=3), name="Flight path")
    frames = [go.Frame(data=rocket_at(int(index)), traces=[1, 2, 3, 4], name=str(int(index))) for index in positions]
    fig = go.Figure(data=[ground, *rocket_at(0), trail], frames=frames)
    duration = max(40, int(1000 * max(len(flight) - 1, 1) / 50 / max(len(positions) - 1, 1)))
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="Horizontal position", yaxis_title="Depth (illustration)", zaxis_title="Height above pad",
                   xaxis=dict(range=[min(-10, float(x.min()) - 1), max(10, float(x.max()) + 1)]),
                   yaxis=dict(range=[-2, 2]), zaxis=dict(range=[min(-1, float(z.min()) - 1), max(11, float(z.max()) + 1)]),
                   aspectmode="data", camera=dict(eye=dict(x=1.4, y=-2.0, z=1.0))),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=0,
                          buttons=[dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": duration, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]),
                                   dict(label="❚❚ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
        sliders=[dict(active=0, x=0.13, len=0.85, y=0,
                      currentvalue=dict(prefix="Step "),
                      steps=[dict(label=str(int(index)), method="animate", args=[[str(int(index))], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]) for index in positions])],
    )
    return fig
