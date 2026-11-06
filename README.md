---
title: Roadster Flight Lab
emoji: 🚘
colorFrom: gray
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
---

# Roadster Flight Lab

[Live interactive demo](https://msmxii-landing-lab.static.hf.space/index.html?build=v7-research)
· [Longer-training diagnosis](reports/training-diagnosis.md)
· [Episode-level results](reports/results)

To preview the saved experiments locally after installing the Python dependencies:

```bash
python scripts/fetch_demo_data.py
cd frontend
npm ci
npm run dev
```

Large generated replay assets are downloaded from the public demo and verified
against its SHA-256 manifest. Raw local training outputs and checkpoints are
excluded from this repository. The committed reports include the longer
307,200-step experiment; its replay recordings have not been published to the
demo yet.

## Current 3D research

The latest Python experiment is **v7**, an experimental Rapier rigid-body
vehicle with four separately rotating wheels, suspension joints, a contact
driven ramp, and cold-gas thrusters. A simplified pressure and valve model
reduces thrust as gas is consumed; its values are assumptions, not Tesla
specifications. The fan applies an approximate downforce during contact and
does not simulate airflow, sealing or electrical energy. Tire deformation,
battery use, detailed aerodynamics and real-world calibration are not modeled.
The replay draws a visual approximation of the vehicle from recorded 3D poses.
This is a research simulator, not a validated Tesla digital twin.

The saved v7 **flight curriculum** starts above the pad. Its results do not
demonstrate a full ramp launch. A hand-written reference uses privileged state
and is separate from the PPO neural network. The existing browser trainer uses
the earlier v6 simulator; the archived v3 experiments are planar and are not
comparable. Small pilot runs are not evidence of a solved task; inspect the
held-out failures and run-level data in the site.

The initial v7 flight-only PPO pilots landed safely in 0/20 and 0/100
held-out episodes (8,192 and 102,400 training steps respectively). The longer
checkpoint, followed by 8,192 steps of full-ramp fine-tuning, reached flight
in 1/20 held-out ramp attempts and landed safely in 0/20. A privileged
hand-written reference landed safely in 4/20 flight-start episodes. These are
small exploratory runs with different budgets, not a controlled reward
comparison. More simulator calibration and independent training runs are
needed before claiming reliable performance.

```bash
# Train and evaluate a flight-start pilot using the Python v7 simulator.
python -m rocket_landing.cli roadster3d-v7-train \
  --variant potential_shaping --seed 0 --stage flight --timesteps 102400
python -m rocket_landing.cli roadster3d-v7-evaluate \
  --model outputs/roadster3d_v7/flight/potential_shaping/seed_0/model.zip \
  --variant potential_shaping --seed 0 --stage flight --episodes 100

# Optional full task, initialized from the flight curriculum checkpoint.
python -m rocket_landing.cli roadster3d-v7-train \
  --variant potential_shaping --seed 0 --stage full --timesteps 300000 \
  --init-model outputs/roadster3d_v7/flight/potential_shaping/seed_0/model.zip
python -m rocket_landing.cli roadster3d-v7-evaluate \
  --model outputs/roadster3d_v7/full/potential_shaping/seed_0/model.zip \
  --variant potential_shaping --seed 0 --stage full --episodes 100
python -m rocket_landing.roadster3d_export
cd frontend && npm ci && npm run build
```

Runs record their configuration, checkpoints, individual training and held-out
trajectories, model hash and episode-level outcomes. Repeat across independent
training seeds before drawing comparisons; different training budgets are not
equivalent experiments. `python -m rocket_landing.reference_3d_v7` evaluates
the privileged hand-written flight reference. Tests can be run with `pytest`.

## Earlier v6 simulator

An educational reinforcement-learning project about ramp-assisted vehicle
flight. The earlier simulator is v6: a 3D rigid-body environment built with
[Rapier](https://rapier.rs/docs/). Python PPO training and the optional
browser-local trainer use Rapier bindings, with shared scenario settings in
`rocket_landing/configs/roadster_v6.json`.

The model is deliberately scoped as a rigid car proxy. Rapier simulates gravity,
mass, forces, air drag, ground and ramp contacts, and collisions. The ramp is a
faceted triangle-mesh collider with a broad curved foot, tall straight stem,
and short curved shoulder at the top, matching the supplied Tesla ramp reference.
The four tire shapes are fixed colliders: the model does not include suspension,
independent wheel rotation, tire deformation, or Tesla hardware. It is not a
validated vehicle digital twin.

## Setup

Python 3.11–3.13 is supported. Install the pinned project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The Python simulator uses the official `rapier3d` package. The static site uses
Rapier's official JavaScript compatibility package and runs without a server
training process.

## Train the 3D simulator

Start with one seed and a short run while tuning a new experiment:

```bash
python -m rocket_landing.cli roadster3d-train \
  --variant potential_shaping --seed 0 --timesteps 50000
python -m rocket_landing.cli roadster3d-evaluate \
  --model outputs/roadster3d_v6/potential_shaping/seed_0/model.zip \
  --variant potential_shaping --seed 0 --episodes 100
```

The standard protocol trains each reward design with five independent seeds,
then evaluates each saved policy on 100 held-out scenarios:

```bash
python -m rocket_landing.cli roadster3d-run-all \
  --timesteps 300000 --episodes 100 --seeds 0 1 2 3 4
python -m rocket_landing.cli roadster3d-report
```

Each run saves PPO checkpoints and logs, versioned settings, episode-level
training trajectories (including launch failures and crashes), held-out
metrics, and evaluation trajectories. The policies are trained from scratch;
neither v3, v4 nor the v5 half-loop policies transfer to v6.
An initial Python/SB3 v6 pilot is included in the saved 3D results.
Browser experiments run locally in a visitor's tab and are not included in the
Python/SB3 artifacts.

## Physics and control

World coordinates are SI units: `x` is downrange, `y` is height, and `z` is
lateral distance. Gravity is `(0, -9.81, 0) m/s²`. A rigid car begins at rest.
PPO controls the powered run-up from rest, then six flight controls: forward,
vertical, lateral, pitch, roll, and yaw. Rapier determines the position,
rotation, and launch velocity from gravity, thrust, and contact with the ramp.
Attitude control comes from nozzle forces applied away from the center of mass.
Wind, vehicle mass, ramp radius and friction, and pad position vary by seeded
episode.

The v6 ramp has three profile sections. For the curved foot, `theta` spans zero
to pi/2 and the approximate car centerline is:

```text
vehicle center x = R * sin(theta)
vehicle center y = R + h - R * cos(theta)
```

Here `R` is the 7.5 m foot radius and `h` is the 0.78 m vehicle-center
clearance. The profile then rises vertically for 8 m and bends through a 40°
upper shoulder with a 5 m radius. Rapier collides against a 128-segment mesh
surface offset from this reference path; the actual car trajectory is produced
by physics. The shoulder sends the car up and back toward the landing pad
without completing the old 180° upside-down loop. Tire friction uses a low
minimum contact coefficient because wheel spin and suspension are outside this
proxy model.

For the upper shoulder, with `a` increasing from zero to 40° and `L` the
vertical stem length, the vehicle centerline continues as:

```text
vehicle center x = R + r_top * (sin(pi/2 + a) - 1)
vehicle center y = R + h + L - r_top * cos(pi/2 + a)
```

The simulator is versioned at `RoadsterRampLanding3D-v6`. The half-loop v5,
straight-ramp v4 prototype, and older `RoadsterRampLanding-v0` planar environment have
different dynamics. Saved v3 results remain available as a historical
benchmark; their 3D depth was illustrative and none of these versions should
be pooled together.

## Reward designs

All policies share the same native task reward:

```text
r_task = 0.08 * (previous_distance - distance)
         - 0.003 * speed
         - 0.008 * tilt
         - 0.003 * angular_speed
         - 0.08 * fuel_used

safe touchdown: +150
crash or failed launch: -100
timeout: -100
```

The comparison adds one of these training objectives:

- **Task reward:** uses the native reward unchanged.
- **Upright dwell probe:** adds `+1` each live control frame when tilt is under
  `0.1 rad` and speed is under `0.1 m/s`, including on the launch ramp. This
  deliberately lets the policy earn reward by remaining still; the separate
  reward-gap metric exposes that behavior.
- **Potential progress:** adds
  `0.5 * (0.99 * Phi(next_state) - Phi(state)) - 0.01` per controlled frame,
  with terminal next potential set to zero:

  ```text
  Phi = -1.00 * distance / 45
        -0.35 * speed / 35
        -0.55 * tilt / pi
        -0.12 * angular_speed / 8
        +0.08 * fuel_fraction
  ```

Every evaluation reports return from the native task reward, as well as the
selected shaped reward and the reward gap. The per-frame time cost changes the
training objective; it is not part of the evaluation reward.

## Website and browser training

The public site keeps the old saved v3 benchmark available as an archive. Its
“Train v6 in your browser” workspace uses Rapier3D WebAssembly and a small
educational JavaScript PPO implementation. Each visitor's attempts, learning
curves, failures and 3D replays stay in that browser tab. This is separate from
Python/SB3 model training and evaluation; do not pool the two result sets.

Build the site locally:

```bash
cd frontend
npm ci
npm run build
cd ..
python -m http.server 8766 --directory rocket_landing/web_assets
```

Open [http://localhost:8766](http://localhost:8766). To package the free static
site for Hugging Face Spaces, run:

```bash
python scripts/prepare_space.py
python scripts/publish_space.py --repo msmxii/landing-lab
```

This deployment is static. It does not start a server-side training job or
require paid compute.

## Project files

- `rocket_landing/envs/roadster3d.py` — Rapier-backed 3D Gymnasium environment.
- `rocket_landing/configs/roadster_v6.json` — shared, versioned 3D settings.
- `rocket_landing/roadster3d_rewards.py` — v6 reward variants.
- `rocket_landing/roadster3d_experiment.py` — v6 training, evaluation and reports.
- `rocket_landing/envs/roadster.py` — retained planar v3 environment.
- `frontend/src/lib/browser-ppo.ts` — Rapier3D browser simulator and compact PPO.
- `frontend/src/lib/roadster-figure.ts` — v3 archive and v4/v5/v6 3D replay geometry.
- `ROADSTER_RL_PLAN.md` — experiment design and extension plan.

Python v6 run artifacts are saved under `outputs/roadster3d_v6/`; historical v3
runs remain under `outputs/roadster/`.
