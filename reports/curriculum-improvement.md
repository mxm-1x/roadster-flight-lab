# Flight landing curriculum pilot

This pilot adds three flight-start profiles, a gate that advances after at
least 70 safe landings in 100 attempts, and a separate touchdown-aware
potential reward. Existing v7 runs remain unchanged. Stage 0 starts 4–4.5 m
above the pad with zero lateral offset and near-zero initial velocity; stages
1 and 2 add offsets and recover the original start distribution.

The guided policy trained for 102,400 steps (seed 0). It recorded 1,147
training attempts, all crashes, so the success gate did not advance. On 100
held-out stage-0 scenarios it had 0 safe landings, mean impact speed 7.56 m/s,
mean tilt 0.71 rad, and mean remaining gas 17.1%.

A same-budget PPO potential-shaping baseline (seed 0) trained on the same stage
and evaluated on the same 100 seeds also had 0 safe landings. It averaged 7.05
m/s impact speed, 0.31 rad tilt, and 20.9% remaining gas. Thus this curriculum
and reward pilot did not improve the learned policy.

## Control-authority check

The privileged PD reference was first evaluated on 20 shared scenarios using
the 80 kg gas assumption. It landed safely in 0/20 starts from each of the two
new profiles, and 4/20 starts from the original flight distribution. At stage
0, mean gas remaining was about 27% and mean impact speed was 4.23 m/s.

To test whether finite gas capacity limits the task, I held the same simulator,
reference controller, and 20 stage-0 scenarios fixed while changing only the
assumed gas mass:

| Assumed gas mass | Safe landings | Mean impact speed | Mean gas remaining |
| ---: | ---: | ---: | ---: |
| 80 kg | 0/20 | 4.23 m/s | 27% |
| 160 kg | 9/20 | 2.51 m/s | 43% |
| 240 kg | 20/20 | 1.64 m/s | 50% |

The episode-level sensitivity data is in
[`propulsion-sensitivity.csv`](results/propulsion-sensitivity.csv). Reproduce
the table from the repository root with:

```bash
python scripts/propulsion_sensitivity.py \
  --episodes 20 --seed-base 40000 \
  --output reports/results/propulsion-sensitivity.csv
```

This is a sensitivity experiment, not evidence about Tesla hardware. The gas
mass values are uncalibrated assumptions. Still, the strong change under a
single propulsion parameter indicates that policy changes alone cannot yet
establish a credible solution. The next experiment should version a revised
propulsion model, document its assumptions, and compare PPO against the
reference across multiple seeds before returning to ramp training.

Episode-level PPO results are in `results/guided-curriculum-profile-0/` and
`results/profile-0-baseline/`.
