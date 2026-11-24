# Probability-aware occlusion MPC: an exploratory 2D simulation

An **exploratory side project**, not a paper and not a replication. It asks one question in a deliberately small
simulation:

> Occlusion-aware MPC (OA-MPC) assumes that every region the robot cannot see might hide a fast-moving obstacle.
> That is safe but cautious. Could the robot relax that assumption where hidden space is *probably harmless*, using
> (a) a scene-risk label from a language/vision model and (b) a noisy audio cue, while still slowing down where
> hidden space is probably dangerous, and speeding up once it has "peeked" and seen the space is empty?

Everything here is a **toy 2D simulation**. Nothing has been run on a robot, with real sound, or against a live
foundation-model API (details in sections 7 and 8).

**Quick links**

| What | Where |
|---|---|
| Video: all five methods on the same trial, one pedestrian | [`videos/compare_1agent_seed11.mp4`](videos/compare_1agent_seed11.mp4) |
| Video: same, two pedestrians | [`videos/compare_2agents_seed17.mp4`](videos/compare_2agents_seed17.mp4) |
| Videos: each method on its own | [`videos/`](videos/) |
| **Main results table** (one pedestrian, delay compensated) | [`results/1agent_delay_compensated/summary.csv`](results/1agent_delay_compensated/summary.csv) |
| Every individual trial behind that table | [`results/1agent_delay_compensated/trials.csv`](results/1agent_delay_compensated/trials.csv) |
| Other settings | [1 pedestrian, uncompensated](results/1agent_delay_uncompensated/summary.csv) · [2 pedestrians, compensated](results/2agents_delay_compensated/summary.csv) · [2 pedestrians, uncompensated](results/2agents_delay_uncompensated/summary.csv) |
| Wrong-label test | [`results/fm_error_test.csv`](results/fm_error_test.csv) |
| Cached risk labels | [`results/risk_cache.json`](results/risk_cache.json) |

---

## 1. Summary

**What was built.** A closed-loop 2D simulator of a unicycle robot driving down a corridor past two hidden regions
(a harmless alcove and a blind side corridor where pedestrians walk), a nonlinear MPC that avoids visible
pedestrians, a simplified re-implementation of the OA-MPC idea, and a belief layer that scales the worst-case
assumption using a per-region risk prior, a "peek" rule, and a simulated audio cue. Five planners were compared on
paired random trials.

**What was found, in this simulator only:**

1. **Hidden space is dangerous to a planner that ignores it, but how much depends on a modelling choice.** The
   baseline MPC collided with a hidden pedestrian in 58% of trials when its planner did not correct for a 0.3 s sensor
   delay, and in 7% when it did. The more robust difference is the **safety margin**: OA-MPC roughly doubled the
   mean closest approach to the pedestrian (0.66 m to about 1.2 m) in both settings.
2. **A scene-risk prior moved the caution without buying time.** It kept OA-MPC's margin, ran at full speed past
   the harmless alcove and slower at the junction, with the same total time (10.0 s).
3. **Audio gave mixed results.** Variants using audio were faster than worst-case OA-MPC but had smaller margins
   and, in the harsher setting, more collisions. Method 4 is confounded (section 5.2).
4. **With two pedestrians, none of the occlusion-aware variants helped meaningfully.** That failure comes from my
   simple handling of *visible* pedestrians, so it says little about OA-MPC itself.
5. **Wrong labels hurt asymmetrically.** Labelling the dangerous junction "low risk" erased the safety benefit
   (closest approach 1.16 m back to 0.66 m); labelling the harmless alcove "high risk" only cost 0.4 s.

**What this does not show:** that any of this works on a real robot, that a foundation model labels scenes
reliably, that the approach is safe, or that it beats OA-MPC. See sections 8 and 9.

---

## 2. Literature rebiew

* **OA-MPC** (Firoozi et al.): grows reachable regions from the boundary of what the robot cannot see,
  and requires the robot to keep clear of them and be able to stop. The core idea reimplemented here.
* **Occlusion-Aware Consistent MPC** (Zheng et al., arXiv 2503.04563, a preprint): plans several hypotheses about hidden
  obstacles that share a first segment of trajectory. It motivated the idea that worst-case planning is
  conservative; I did not implement it.
* The probability layer, the peek rule, the audio cue and the label-error test are **my additions**, not from either paper.

---

## 3. The simulated scenario

```
            y=9  +--+
                 |  |   <- side corridor ("junction" zone), x in [6, 8.5], hidden until the robot is almost level with it
                 |  |
      +---+      |  |
      |alc|      |  |
 y=2.5+-+ +------+  +------------+
      robot -->  .  .  .  .  .  .   goal (x=13)    main corridor, y in [0, 2.5], 14 m long
 y=0  +----------------------------+
```

| Element | Setting (all in `config.py`) |
|---|---|
| Robot | unicycle with a speed state; radius 0.3 m; max speed 1.6 m/s; max acceleration/braking 1.0 m/s^2; follows y = 1.25 |
| Control rate / horizon | dt = 0.2 s (5 Hz), N = 5 steps (1.0 s). The OA-MPC paper uses 10 Hz and N = 5 |
| Alcove (x 2 to 3) | a 1 m niche with no exit. It creates a hidden region, but **no pedestrian is ever in it** |
| Junction (x 6 to 8.5) | pedestrians walk straight down it, cross the lane, and exit through an imaginary door in the south wall |
| Pedestrians | radius 0.3 m, speed uniform in [0.9, 1.4] m/s, position x uniform in [6.45, 8.05]. Timing is aimed to cross where a naive full-speed robot would be, plus or minus 0.7 s. The second pedestrian (if used) crosses 1.2 to 2.5 s after the first. They do not react to the robot |
| Perception | ray-cast line of sight against the walls, range 12 m. Visible pedestrians are reported with a **0.3 s delay** (position where they were 0.3 s ago; visibility itself is tested at the current position, a simplification) |
| Trials | each seed fixes the pedestrian(s), so all five methods face identical trials |

**Important consequence of the design:** the only dangerous hidden region is the junction, and I generate the
pedestrians to come from there. So a prior that says "junction = high risk, alcove = low risk" matches the simulator
by construction. This is a favourable test of the prior, not a neutral one.

---

## 4. The five methods

All five share the same MPC and the same handling of **visible** pedestrians. They differ only in how hidden space is treated.

| # | Name | Hidden-space handling |
|---|---|---|
| 1 | Baseline MPC | none: hidden space is treated as empty |
| 2 | OA-MPC (worst case) | every hidden-region boundary point is assumed to hide a pedestrian at the planner's maximum assumed speed |
| 3 | OA + scene prior | method 2, but the assumption is scaled per region by a belief that starts from a risk label |
| 4 | OA + audio | same as 3, but the belief starts at a uniform 0.20 in both regions and is updated by the audio cue |
| 5 | OA + prior + audio | prior from the risk labels, updated by the audio cue |

### 4.1 How each part works

**MPC (`mpc.py`).** State [x, y, heading, speed], controls [acceleration, turn rate]. Built once with CasADi and
solved each step with IPOPT (a few milliseconds per solve in my runs). The cost tracks a straight reference
line plus a speed target, and penalises acceleration, turning and heading error. **Walls, occlusion discs and visible
pedestrians are all soft penalties (squared hinge terms), not hard constraints**, so the optimiser always returns an
answer. A soft penalty on terminal speed approximates the paper's "be able to stop by the end of the horizon". If
the solver fails, the robot brakes.

**Visible pedestrians (`sim.py`).** Constant-velocity prediction from two consecutive observations. On first sighting
the velocity is unknown, so a growing disc is used. By default the prediction is extrapolated over the known 0.3 s
delay (`compensate_latency=True`); `--no-compensate` turns this off.

**Hidden space, OA-MPC style (`world.py`).** Free space is a 0.25 m grid. A cell is visible if the straight line
from the robot crosses no wall. A *frontier* cell is a hidden cell touching a visible cell (where an unseen person
could step out). Frontier cells within 5 m and not more than 0.3 m behind the robot are thinned to at most 14 points,
0.45 m apart. At planning step k, each point gets a disc of radius `r_robot + r_agent + v_agent_max * t_k`
(`v_agent_max` = 1.4 m/s) that the MPC is penalised for entering. The terminal-stop penalty is applied while any
such discs exist.

**Belief (`risk.py`).** One probability p per region (alcove, junction) that "an unseen agent is about to come out of
here":

* *Prior:* low 0.03, medium 0.20, high 0.50 (**my assumed numbers**, not data).
* *Peek rule:* once the robot can see every cell of a region that lies ahead of it, p is set to 0.01. If the region
  becomes hidden again, p returns to its prior. A region is also treated as clear once the robot has driven past it.
* *Leak:* each step, p is pulled 2% back toward its prior.
* *Audio:* each step the robot "hears" or not. A Bayes update uses the robot's own simplified model (P(hear | agent)
  = 0.15, P(hear | none) = 0.03). The simulated microphone detects a *hidden, moving* pedestrian with probability
  `0.25 * exp(-distance / 4 m)` (never below the 3% false-alarm rate), through walls.
* *Silence floor:* silence alone can never push p below 40% of the prior; only seeing the region can clear it. (Without
  this, early versions let a few quiet seconds drive p to near zero and the robot collided.)

**From belief to behaviour (`sim.py`).** With s = clip(p / 0.5, 0, 1):

* disc growth rate is multiplied by s (s = 1 gives the full worst case);
* speed target = 0.9 + (1.6 - 0.9) * (1 - s), using only regions within 2.5 m of the robot;
* the stop penalty is switched on only when a nearby region has p >= 0.15.

Methods 3, 4 and 5 all include the peek rule, the leak, and the silence floor.

### 4.2 Confounds you should know about

* **Method 4 is not "audio alone"**: it differs from method 2 in both the uniform 0.20 prior *and* the audio cue, so
  its results cannot be attributed to audio.
* **The peek rule was never ablated**: there is no run with the belief layer but without peeking.
* Methods 3 to 5 share many hand-set constants; there was no sensitivity analysis.

---

## 5. Results

Per setting: 60 trials with pedestrians and 20 without, seeds 0 to 59 and 0 to 19, identical across methods. CSVs
are in `results/`. **Collision** = centres closer than 0.6 m. **Closest approach** = mean minimum distance to a
pedestrian (0.6 m means contact). **Near miss** = closest approach under 0.9 m. Brackets are 95% Wilson intervals.
No significance tests were run, and the differences between methods 2 to 5 on collision counts are mostly within
the intervals.

### 5.1 One hidden pedestrian

| Method | Collisions, delay compensated | Collisions, NOT compensated | Closest approach, comp. | Near miss, comp. | Time, no pedestrian | Speed at alcove | Speed on approach to junction |
|---|---|---|---|---|---|---|---|
| 1 Baseline | 4/60 (3 to 16%) | 35/60 (46 to 70%) | 0.66 m | 100% | 8.8 s | 1.51 | 1.60 |
| 2 OA-MPC | 0/60 (0 to 6%) | 0/60 (0 to 6%) | 1.20 m | 37% | 10.0 s | 1.20 | 1.18 |
| 3 OA + prior | 0/60 (0 to 6%) | 0/60 (0 to 6%) | 1.16 m | 28% | 10.0 s | 1.50 | 0.92 |
| 4 OA + audio | 1/60 (0 to 9%) | 15/60 (16 to 37%) | 0.88 m | 68% | 9.2 s | 1.42 | 1.38 |
| 5 OA + prior + audio | 0/60 (0 to 6%) | 4/60 (3 to 16%) | 1.01 m | 45% | 9.6 s | 1.50 | 1.08 |

"Speed at alcove" is the mean speed for x in [1, 3] and "approach to junction" for x in [4, 6.5], both from the
no-pedestrian trials. Closest approach without compensation: 0.66, 1.24, 1.19, 0.91 and 1.03 m for methods 1 to 5.

![Speed profile with no pedestrian: who slows down where](results/1agent_delay_compensated/speed_profile.png)

The plot shows mean speed along the corridor in the trials with no pedestrian: method 2 (blue) is slow almost everywhere,
methods 3 and 5 slow only on the approach to the junction, and the baseline never slows.

![Collision rate, near-miss rate and time to goal](results/1agent_delay_compensated/summary.png)

### 5.2 Two hidden pedestrians

| Method | Collisions, compensated | Collisions, NOT compensated | Closest approach, comp. |
|---|---|---|---|
| 1 Baseline | 5/60 | 43/60 (59 to 81%) | 0.65 m |
| 2 OA-MPC | 2/60 | 29/60 (36 to 61%) | 0.72 m |
| 3 OA + prior | 7/60 | 29/60 | 0.72 m |
| 4 OA + audio | 4/60 | 34/60 | 0.71 m |
| 5 OA + prior + audio | 4/60 | 31/60 | 0.71 m |

### 5.3 How to read these

* **Supported (in this simulator):** OA-MPC raised the closest approach to a single hidden pedestrian from about 0.66 m to about
  1.2 m, in both delay settings.
* **Setting-dependent:** the baseline's collision rate (7% vs 58%) depends heavily on whether the planner corrects for
  sensor delay. My first headline number (57%) came from leaving the delay uncorrected; see section 10.
* **Observed, not established:** the scene prior redistributed speed (1.50 vs 1.20 at the alcove, 0.92 vs 1.18 near the
  junction) with no change in total time or measured margin. Audio variants were faster with smaller margins.
* **Not supported:** that any variant helps with two pedestrians. The second pedestrian is usually already visible when the
  robot reaches the junction, so occlusion handling has switched off, and the collisions come from my weak
  visible-pedestrian handling (soft penalties, constant-velocity prediction, 1 m/s^2 braking). That is a limitation
  of this planner, not evidence about OA-MPC.

---

## 6. The foundation-model part: what was and was not done

**Done**

* `risk.py` contains a function that sends a text description of a hidden region to an Anthropic model and expects
  `low | medium | high`, with a keyword fallback and a JSON cache. The code is guarded so that failures fall back
  rather than crash.
* The cache (`results/risk_cache.json`) holds two labels (alcove: low, junction: high). **They were produced by
  Claude in a chat conversation, not by an API call.** The cache was added *after* the experiments and videos in
  this repository were produced. Those runs used the keyword fallback, which returns the same two labels, so
  re-running with the cache present gives the same priors.
* `fm_error_test.py` sets the labels wrongly *by hand* and measures the consequence (method 3, 60 trials, one pedestrian,
  delay compensated):

| Labels | Collisions | Closest approach | Time, no pedestrian |
|---|---|---|---|
| Correct (alcove low, junction high) | 0/60 | 1.16 m | 10.0 s |
| Junction wrongly "low" | 2/60 | 0.66 m | 8.8 s |
| Junction wrongly "medium" | 0/60 | 0.90 m | 9.4 s |
| Alcove wrongly "high" | 0/60 | 1.51 m | 10.4 s |

**Not done**

* **The API code path was never executed.** I had no API key, so it is untested; a key is needed to run
  `--use-llm`.
* **No vision-language model was used**, and no images were sent to any model. The "scene description" is a sentence I wrote.
* **No measurement of how accurate a language or vision model is at labelling scenes.** Two sentences, labelled once.
* **Nothing here shows a foundation model improves planning.** The test shows only that the planner can consume a label,
  and that wrong labels hurt asymmetrically in this simulator. Because the labels agree with how I built the
  simulator, the prior is helped by construction.
* The wrong-label test uses hand-chosen mislabels, not errors sampled from a real model.

**Ideas not implemented** (only ideas): send a camera image to a vision-language model instead of text; have the model
estimate pedestrian type and speed per region (clipped to a hard cap); keep a log of near-misses and let a model revise
priors; replace the simulated microphone with a pretrained audio-event model; run a small local model instead of an API.

---

## 7. What was NOT done (summary)

* No real robot, no ROS or Gazebo, no hardware, no real sound or audio data, no real-world scenes.
* No comparison against the original OA-MPC code or its published results; **my numbers are not comparable to theirs**
  (different scenario, robot model, horizon, speeds).
* No hard constraints, no proof of recursive feasibility, **no safety guarantee**.
* No live LLM or VLM call; the cached labels came from a chat, and the reported runs used the keyword fallback (section 7).
* No ablation of the peek rule; no isolated "audio only" comparison (method 4 is confounded).
* No sensitivity analysis over the hand-set constants (prior values, audio model, weights, thresholds, delay).
* No significance tests; no held-out scenarios or maps; one map, one pedestrian path, pedestrians that do not react.
* No fix for the weak visible-pedestrian handling revealed by the two-pedestrian runs.
* No peer review.

---

## 8. Differences from the OA-MPC paper

| OA-MPC (as published) | This repository |
|---|---|
| Hard constraints, recursive feasibility argument | soft penalties; no feasibility argument |
| Reachable capsules from the occlusion boundary | discs at thinned frontier points, same growth idea |
| Hard terminal stop | soft terminal-speed penalty |
| Their corner scenario, hardware robot | T-junction in a straight corridor, simulated unicycle |
| Worst case everywhere | worst case scaled by a belief in methods 3 to 5 (my addition) |
| (not in paper) | forward filter: frontier cells more than 0.3 m behind the robot are ignored |

---

## 9. Development notes (changes made after seeing results)

All tuning was done on the same simulator the results come from, with no held-out test. In order:

1. A symmetric four-way crossing made worst-case OA-MPC freeze, so I switched to a T-junction.
2. The baseline initially avoided the pedestrian easily, so I made the problem harder: higher speed, weaker braking, and a
   0.3 s perception delay. **This was tuning toward a demonstration where the baseline fails.**
3. My first headline (baseline collides 57%) was driven by the uncompensated delay. I added delay compensation, made
   it the default, and now report both settings.
4. My first OA-MPC baseline kept the stop penalty on everywhere, making it about 1.4 s slower than necessary. I changed it
   to apply only while something is hidden, which removed the apparent speed advantage of the prior.
5. Early belief versions let silence drive the belief to near zero and collided; I added the silence floor. A low speed
   floor made method 3 deadlock at the junction; I raised the minimum cruise speed from 0.4 to 0.9 m/s.
6. Cost weights (for example the disc weight, 120 to 300) were adjusted by hand.
7. The two-pedestrian runs exposed the weak visible-pedestrian handling; I did not fix it.

---

## 10. Reproducing

Tested with Python 3.12.3, numpy 2.5.4, pandas 3.0.6, matplotlib 3.11.2, casadi 3.8.2, shapely 2.2.0, pillow 13.0.0,
imageio-ffmpeg 0.6.0.

```bash
pip install -r requirements.txt
python run_experiments.py --quick                  # ~20 s smoke test
python run_experiments.py --trials 60              # one pedestrian, delay compensated (a few minutes)
python run_experiments.py --no-compensate          # same, planner ignores the sensor delay
python run_experiments.py --agents 2               # two pedestrians (also with --no-compensate)
python make_videos.py                              # side-by-side video, auto-picks a seed
python make_videos.py --agents 2
python fm_error_test.py --trials 60                # wrong-label test
```

By default `run_experiments.py` writes to `results/latest/` (git-ignored scratch space), so a quick test cannot be
mistaken for the committed results. The committed folders were produced with:

```bash
python run_experiments.py --trials 60 --clean-trials 20 --out results/1agent_delay_compensated
python run_experiments.py --trials 60 --clean-trials 20 --no-compensate --out results/1agent_delay_uncompensated
python run_experiments.py --trials 60 --clean-trials 20 --agents 2 --out results/2agents_delay_compensated
python run_experiments.py --trials 60 --clean-trials 20 --agents 2 --no-compensate --out results/2agents_delay_uncompensated
```

**What has and has not been reproduced.** Runs are seeded.

| Check | Result |
|---|---|
| `fm_error_test.py` run twice in the original environment | identical numbers |
| `--quick` run on a second machine (mine) | all metrics identical (only solver timing differed) |
| Full run, one pedestrian, delay compensated, on the second machine | all reported metrics identical to the committed results |
| `make_videos.py` on the second machine | same outcomes for seed 11 |
| Other three settings; wrong-label test on the second machine | **not re-run there** |

Reproducing the numbers shows the code does what this document says. It does not show the findings hold outside this
simulator.

`--use-llm` needs `ANTHROPIC_API_KEY` and is untested (section 7). `results/risk_cache.json` is read before any call.

**Videos** ([`videos/`](videos/); start with [`compare_1agent_seed11.mp4`](videos/compare_1agent_seed11.mp4) and [`compare_2agents_seed17.mp4`](videos/compare_2agents_seed17.mp4)): the *same* trial for all five methods. **The seeds were chosen on purpose**: seed 11
(one pedestrian) and seed 17 (two) are cases where the baseline collides and OA-MPC does not, so they show a
representative failure, not a typical run. Dark = hidden from the robot, yellow = visible, orange circles = where an
unseen person could be, faded red = a pedestrian the robot cannot see. The "belief" readout is the highest belief among
hidden regions within 2.5 m of the robot (0.00 when none). Even in seed 11, closest approaches were only 0.67 to 0.78 m.

---

## 12. Files

| File | Purpose |
|---|---|
| `config.py` | every parameter |
| `world.py` | map, walls, visibility, frontier extraction, pedestrians |
| `mpc.py` | CasADi/IPOPT MPC |
| `risk.py` | label classifier + cache, per-region belief, simulated microphone |
| `sim.py` | closed-loop trial and the five method definitions |
| `run_experiments.py` | paired trials, CSVs, summary tables, plots |
| `render.py`, `make_videos.py` | videos |
| `fm_error_test.py` | wrong-label test |
| `results/` | CSVs and plots for the four settings, `fm_error_test.csv`, `risk_cache.json` |
| `videos/` | MP4s for one and two pedestrians |

---

## 13. Open questions (feedback welcome)

1. Is relaxing the worst case with probabilistic information sensible at all, or does it throw away the property
   that makes OA-MPC valuable? Is there a way to bound a learned prior's influence and keep a guarantee?
2. Is "closest approach" a fair way to compare planners when collisions are rare?
3. What would a meaningful, non-circular test of a language-model risk prior look like?

---

## 14. References

* R. Firoozi, A. Mir, G. S. Camps, M. Schwager. *OA-MPC: Occlusion-Aware MPC for Guaranteed Safe Robot Navigation
  with Unseen Dynamic Obstacles.* IEEE Transactions on Control Systems Technology 33(3), 940 to 951. arXiv:2211.09156.
* Zheng et al. *Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense
  Environments.* arXiv:2503.04563 (preprint).
* R. Firoozi et al. *Foundation Models in Robotics: Applications, Challenges, and the Future.* International
  Journal of Robotics Research 44(5), 701 to 739.
* J. Andersson, J. Gillis, G. Horn, J. Rawlings, M. Diehl. *CasADi: a software framework for nonlinear optimization
  and optimal control.* Mathematical Programming Computation 11, 2019.
* A. Wachter, L. Biegler. *On the implementation of an interior-point filter line-search algorithm for large-scale
  nonlinear programming.* Mathematical Programming 106, 2006.