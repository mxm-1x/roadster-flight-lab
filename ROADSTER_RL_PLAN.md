# Roadster Flight Lab — ML Engineering Plan

## Goal

Extend Landing Lab with an original reinforcement-learning task inspired by the Roadster ramp and booster demonstration: a car follows a curved ramp, becomes airborne, uses short thruster bursts to control its attitude, and attempts a controlled landing.

This is a research simulation, not a recreation of Tesla's vehicle or software. Tesla's proprietary vehicle dynamics, full telemetry, and controller are not public. The simulation must publish its own equations, units, assumptions, and reward definitions, and describe the 3D view as a rendering of the simulation.

## Architecture decision

Add Roadster Flight Lab beside the existing LunarLander experiment. Do not replace or relabel LunarLander runs. Its observation vector, action meanings, simulator, trained policies, reward metrics, and saved flight format are specific to LunarLander; showing those trajectories as Roadsters would misrepresent what the existing neural network learned.

Keep the existing product constraints:

- Local CPU training from the Python CLI.
- A free, static website that plays exported artifacts; no hosted inference or live-training service.
- The current black-and-white visual design.
- Reproducible seeds, saved checkpoints, episode data, and clear separation between training reward and task-performance metrics.

## Proposed first environment

Create a custom Gymnasium environment for a **planar rigid-body car**. It is intentionally a 2D control problem rendered in 3D, with depth clearly marked as illustrative. Use SI units and one documented simulation time step. Put all physics values and success thresholds in a versioned configuration file, rather than scattering constants through the environment and UI.

### Staged task

1. **Ramp launch fixture (v1):** replay a deterministic curved-ramp trajectory to provide reproducible exit states. This is kinematic fixture playback; it does not simulate ramp contact forces, fan suction, or adhesion. The learned portion is airborne attitude control and landing.
2. **Booster-controlled flight (v1):** PPO fine-tunes normalized horizontal thrust, upward thrust, and attitude torque after the scripted ramp exit. The model tracks position, velocity, heading (as sine and cosine), angular velocity, flight phase, remaining gas, and time. Use clipped continuous actions and PPO with an `MlpPolicy`.
3. **End-to-end curriculum (future):** once flight control is stable, expose the approach and ramp controls to the policy and train with a curriculum from easy ramp exits to the full route. The website must identify whether a selected run used a scripted ramp fixture or a learned ramp controller.

Each reward variant uses a common, seed-matched warm start from the documented PD reference controller: 64 supervised demonstrations followed by three DAgger rounds of 16 learner-visited episodes each, then PPO fine-tuning. This was added after the from-scratch pilot failed to coordinate attitude and thrust; the cloned model, demonstration data, counts, and loss history are saved so the ablation remains auditable.

The flight physics uses a fixed integration step and deterministic reset seeds. Ground contact, booster impulse, angular torque, fuel use, termination, and time limits are explicit. The v1 scripted ramp is not a contact or fan-adhesion model. Clip or reject non-finite states and fail with useful diagnostics rather than silently producing invalid episodes.

### Rewards and evaluation

Preserve the project's reward-shaping research question with three configurable treatments:

- **Task reward:** stable reference objective for landing inside the target zone, controlling touchdown speed and attitude, and conserving gas.
- **Upright dwell probe:** a deliberately flawed per-step reward for an upright, slow car. Use it to measure whether the policy loiters or spends gas to optimize the proxy.
- **Potential-difference shaping:** add a documented discounted potential difference based on distance to the pad, speed, attitude error, angular rate, and remaining gas. Set terminal next-state potential to zero; avoid introducing an unreported objective change.

Every policy is evaluated against the same unshaped task objective on common held-out seeds. Report episode-level results and per-run mean, standard deviation, and confidence intervals for safe-landing rate, crash rate, timeout rate, landing position error, touchdown speed, attitude error, flight time, gas remaining, and native versus shaped return. Classify safe landings from explicit configurable physical thresholds, not from a reward sign.

Start with short CPU pilot runs and a small seed set to tune the environment and PPO stability. Move to the full run only after learning curves show the task is trainable. Target at least five independent training seeds per reward treatment for the final comparison; evaluate each model over 100 shared held-out initial conditions. Do not report a result as solved until those measurements exist.

## Code changes

- Add `rocket_landing/envs/roadster.py` and a versioned physics configuration.
- Add Roadster reward wrappers without changing existing LunarLander wrappers.
- Initialize every reward treatment from the same PD demonstration protocol, then compare PPO fine-tuning under each reward. Save the pretraining model and loss history.
- Add Roadster training/evaluation commands and outputs under a separate `outputs/roadster/` tree. Save the config, simulator version, PPO parameters, seeds, library versions, start/end status, checkpoints, episode metrics, and full state/action trajectories.
- Add a separate replay schema for Roadster trajectories. Do not reinterpret the existing LunarLander export schema.
- Extend the static exporter to package compact Roadster results, every completed training attempt, and every evaluation flight. Training attempts may be packed into compressed per-run bundles to keep static hosting manageable; the site continues to load saved files only.
- Add a distinct **Roadster Lab** view with a low-poly electric sports car, curved ramp, visible thruster plumes, flight trail, synchronized timeline, and telemetry panels. Keep existing LunarLander views labeled and available.
- Document reproducible local commands and the difference between the simulation and the real Tesla demo. Do not publish new claims about Tesla performance or imply the policy is Tesla's controller.

## Milestones

1. **Physics contract:** freeze observation/action schemas, coordinate conventions, reset distribution, physical parameter ranges, and safe-landing thresholds. Document all assumptions.
2. **Deterministic environment:** implement the ramp-exit fixture and airborne rigid-body dynamics with fuel and contact events.
3. **Controller baselines:** add a simple rule-based controller and validate that success/failure cases are physically sensible before expensive PPO runs.
4. **Reproducible RL:** implement reward variants, checkpointed PPO training, paired held-out evaluation, episode-level reports, and recorded failures.
5. **Static showcase:** export artifacts, add Roadster-specific 3D replay and comparisons, preserve the current LunarLander experience, then rebuild the free static site.
6. **End-to-end curriculum:** let PPO control the ramp approach after the airborne controller and evaluation pipeline are established.

## Acceptance criteria

- Identical seeds reproduce the same reset and transition sequence.
- The environment has finite bounded observations/rewards, well-defined terminal states, and auditable gas accounting.
- A rule-based controller can produce both successful and deliberately failing episodes under the documented parameter settings.
- Training and evaluation artifacts identify the environment/config version and preserve episode-level state, action, reward, and outcome.
- Evaluation uses common held-out seeds and reports physical outcomes independently from shaped return.
- The website replays the exact exported trajectory, including failed attempts, and visibly distinguishes scripted ramp setup from policy-controlled behavior.
- The original LunarLander models, results, and replay behavior remain unchanged.
- No claim that PPO learned to fly a real car, and no claim of Tesla-equivalent physics or performance.

## Main risks and mitigations

- **Sparse or unstable learning:** begin with ramp-exit stabilization, use a validated controller baseline, then add curriculum stages.
- **Reward hacking:** select policies with native success metrics and inspect failure clips; retain shaped-return gaps in the report.
- **Overconfident physics:** publish parameters and sensitivity ranges; label unknown Tesla specifications and illustrative rendering clearly.
- **Showcase looks more capable than the model:** display which stages are scripted and which are policy-controlled next to the replay.
- **Large static exports:** export selected checkpoint flights and all final evaluation outcomes; keep full training trajectories in local outputs.

## Resume project framing

After the full experiment runs, describe the measurable work: custom Gymnasium rigid-body environment, behavior-cloned PPO control, reward-ablation methodology, multi-seed evaluation, reproducible trajectory capture, and a static interactive 3D results site. Write the final resume bullet from measured success rates and variability; do not promise performance before training.
