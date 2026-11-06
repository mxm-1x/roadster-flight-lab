# Longer flight-only PPO pilot

The same v7 potential-shaping configuration and training seed (1) were trained
from scratch for 307,200 steps, three times the previous 102,400-step budget.
Both checkpoints were evaluated on the same 100 held-out flight-start seeds
(40000–40099). These are simulated starts over the pad, not ramp launches.

| Metric | 102,400 steps | 307,200 steps |
| --- | ---: | ---: |
| Safe landings | 0/100 | 0/100 |
| Mean touchdown speed | 5.27 m/s | 5.24 m/s |
| Mean touchdown tilt | 46.1° | 19.1° |
| Mean remaining gas | 27.8% | 19.4% |

The longer run recorded 3,261 completed training episodes, all crashes. In its
held-out evaluation, 100/100 episodes exceeded the 2.5 m/s vertical-speed
threshold before contact and 99/100 exceeded the 0.22 rad tilt threshold.
The 20-episode privileged, hand-written flight reference landed safely 4 times;
it uses information beyond the PPO observations and is not a learned policy.

The reward has a large success bonus but neither pilot encountered a safe
training landing. This sparse signal is a plausible learning bottleneck, not a
proven sole cause. The assumed gas model also reduces maximum thrust as pressure
falls. Near 28% gas, the seed-40000 vehicle has at most about 8.3 kN vertical
thrust versus about 17 kN weight. The parameters are assumptions, not Tesla
specifications. More time on the current setup alone has not solved the task.

The results and checkpoints are in
`roadster3d_v7/flight/potential_shaping/seed_1/`. The site's saved results have
not been replaced with this local experiment.
