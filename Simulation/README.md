# TSA Exoskeleton — Sit-to-Stand Simulation

MuJoCo/MyoSuite simulation of a Twisted String Actuator (TSA) knee exoskeleton assisting a musculoskeletal model through sit-to-stand (STS). Includes a reflex-based biological controller, TSA physics, and three PPO training pipelines.

---

## Project Structure

```
TSAExoskeletonSTS/
│
├── base_code.py                    # Run one STS episode manually (useful for debugging)
├── train_ppo.py                    # PPO training — torque-tracking reward
├── train_ppo_min_control.py        # PPO training — min-control / baseline-tracking reward
├── train_ppo_feedback.py           # PPO training — multi-step feedback controller
├── train_ppo_feedback_resume.py    # Resume a feedback PPO run from a checkpoint
├── run_optimal.py                  # Replay best PPO policy deterministically
├── plot_tsa_log.py                 # Plot TSA actuator state logs (CSV → PNG)
├── plot_diag_p4.py                 # Plot Phase 3→4 stability diagnostics
│
├── ctrl_optim/
│   ├── sts_ctrl.py                 # 4-phase STS reflex controller (core sim loop)
│   ├── sts_ctrl_ari.py             # Reference controller variant (read-only)
│   ├── tsa_integration_full.py     # 4-motor-per-leg TSA physics wrapper
│   ├── tsa_integration.py          # Single-motor-per-leg wrapper (earlier version)
│   ├── ppo_wrapper.py              # Gymnasium env — torque-tracking PPO
│   ├── ppo_wrapper_min_control.py  # Gymnasium env — min-control PPO
│   ├── ppo_feedback_wrapper.py     # Gymnasium env — multi-step feedback PPO
│   ├── ppo_config.py               # Shared config dataclass for all PPO runs
│   ├── grid_search.py              # Grid search over TSA params (pre-PPO baseline)
│   ├── eval_best_min_control.py    # Replay best sampled θ from a min-control run
│   ├── extract_muscle_log.py       # Compare baseline vs TSA muscle activation
│   └── ppo_evaluation/
│       ├── compare_baseline_assisted.py  # Side-by-side plots + effort summary
│       ├── plot_ppo_eval.py              # Per-episode reward curve plots
│       └── plot_reward_curve.py          # Reward curve from training CSV
│
├── tsa_modelling/
│   ├── model_v2.py                 # TSASimulator — string geometry + speed-torque curve
│   └── actuator.py                 # TSAActuator — RK4 motor ODE integrator
│
├── myoassist/                      # MyoSuite fork with TorsoLegs environment
├── observations/                   # Notes on observations, stability, TSA signals
├── support_docs/                   # Handoff notes, model notes, design docs
└── logs/                           # Training outputs (git-ignored)
```

---

## Simulation Environment

The model is `TorsoLegs`, registered by MyoSuite. It's a lower-body musculoskeletal model with ~80 Hill-type muscles, driven by activation signals in [0, 1]. The root is defined by three joints: `root_x` (slide), `root_z` (slide), `root_pitch` (hinge) — not a free-joint quaternion.

```bash
from myosuite.utils import gym
env = gym.make("TorsoLegs")
```

All scripts require `mjpython` (MuJoCo's bundled Python runtime) rather than the system Python:

```bash
source myoassist/.myo-venv/bin/activate
which mjpython
```

---

## TSA Physics Model

### String geometry (`tsa_modelling/model_v2.py`)

Contraction as a function of motor angle θ:

```
X(θ) = L − sqrt(L² − (θ·r)²)
```

`L` is the untwisted string length (the key PPO-optimised parameter). `J = dX/dθ` maps motor speed to cable speed.

### Motor ODE (`tsa_modelling/actuator.py`)

`TSAActuator.step()` runs RK4 at each sim timestep. Three branches:

| Branch | Condition | Behaviour |
|--------|-----------|-----------|
| `at_wall` | X ≥ X_max | String fully wound; motor stalls |
| `load_stalled` | τ_desired > τ_available | Motor stalls against load |
| `advancing` | Otherwise | Full EOM integration: I·θ̈ = τ_cmd − τ_load |

Returns `tension`, `torque`, `X`, `theta`, `theta_dot`, `at_wall`, `load_stalled`.

Key motor parameters (set in `tsa_integration_full.py → _build_tsa_sim`):

| Parameter | Value |
|-----------|-------|
| `radius` | 4 mm |
| `max_motor_torque` | 0.1668 N·m (1.7 kg·cm) |
| `no_load_speed` | 800 RPM |
| `pretension_theta` | 20π rad (10 turns) |
| `max_contraction_ratio` | 0.30 |

---

## TSA Integration (`ctrl_optim/tsa_integration_full.py`)

**4-motor layout per leg.** Two central motors (offset 0°), one at +8°, one at −8°. Sagittal torque per motor: `τ_i = T_i × d × cos(α_i)`. Symmetric ±α pairs cancel frontal moments.

**Staggered activation.** Each motor has an `activation_time` [s]; it's inactive until `t >= activation_time`. PPO optimises these times.

**Cable slack model.** A motor delivers zero torque until its contraction `X` exceeds the geometric slack threshold:
```
X_geom = X₀ + MOMENT_ARM × max(0, knee_initial − knee_current)
```

**Torque injection into MuJoCo:**
```python
sim.data.qfrc_applied[knee_dadr] -= result['torque']
```
Applied directly to `qfrc_applied` at the knee DOF — no muscle signals are modified.

**Control mode:** `full_power` is the default. Each active motor is commanded at `T_des = 1e6 N` (clamped internally to τ_stall at current speed). `demand_share` mode exists for comparison but under-delivers during early wind-up.

`tsa_integration.py` is the single-motor-per-leg predecessor — still used by `ppo_wrapper.py` but superseded by the full 4-motor version.

---

## STS Reflex Controller (`ctrl_optim/sts_ctrl.py`)

Phase-gated reflex controller. Each muscle group gets a stimulation signal proportional to sensory error (joint deviation, pelvis velocity, etc.) gated by the current phase.

| Group | Role |
|-------|------|
| `GLU` | Hip extension |
| `VAS` | Knee extension |
| `SOL` | Ankle plantarflexion |
| `TA` | Dorsiflexion |
| `HFL` | Hip flexion |
| `HAM` | Hamstrings |
| `TORSO_EXT / FLEX` | Spinal stabilisation |

### Phases

| Phase | Description | Key entry condition |
|-------|-------------|---------------------|
| 1 | Lean forward | Start; root and feet held |
| 2 | Seat-off | `trunk_lean_rel ≤ p1_lean_target` |
| 3 | Extension/rise | Pelvis rising, lean safe, torso not collapsing |
| 4 | Stabilisation | Pelvis height, contacts, lean all within tolerance |

All phase timings and thresholds live in `STSReflexParams`. `leg_scale` can be passed to `apply_leg_control_reduction()` to simulate reduced lower-limb control capacity.

---

## PPO Optimisation

Three separate pipelines, all using SB3 PPO. They share `ppo_config.py` for hyperparameters.

### 1. Torque-tracking (`train_ppo.py` + `ppo_wrapper.py`)

**What's being optimised:** θ = [L, t0, t1, t2, t3] — string length and 4 staggered motor activation times. Symmetric across legs (5-D action). `t0 ≤ t1 ≤ t2 ≤ t3` enforced by sorting.

**Reward:**
```
R = w_torque · R_torque + w_muscle · R_muscle + w_time · R_time
```
`R_torque` is a one-sided integral: only penalises under-delivery (delivered torque < α × knee demand). This prevents reward-hacking by late/no activation.

**Architecture:** One PPO timestep = one full STS rollout. `terminated=True` always so SB3 immediately starts a new episode.

```bash
mjpython train_ppo.py
mjpython train_ppo.py --timesteps 2000 --out logs/ppo_quick
mjpython train_ppo.py --config path/to/config.json
```

---

### 2. Min-control / baseline-tracking (`train_ppo_min_control.py` + `ppo_wrapper_min_control.py`)

Same one-step structure (θ = [L, t0..t3]), different reward. A reference trajectory is generated once from the working baseline controller (no TSA). Each rollout is rewarded for:

- Staying close to the baseline kinematics (position + velocity tracking)
- Reducing VAS (quadriceps) activation
- Not using excessive TSA torque
- Completing the STS and matching baseline duration

**Reward:**
```
R = w_success · R_success
  + w_track_pos · R_track_pos
  + w_track_vel · R_track_vel
  + w_muscle · R_muscle
  + w_tsa · R_tsa
  + w_stability · R_stability
  + w_time · R_time
```

Default weights are set inline in `_run_episode` — adjust there first. They can be moved to `ppo_config.py` once stable.

```bash
mjpython train_ppo_min_control.py
mjpython train_ppo_min_control.py --timesteps 5000 --out logs/ppo_min_control_run1
```

---

### 3. Multi-step feedback PPO (`train_ppo_feedback.py` + `ppo_feedback_wrapper.py`)

Different architecture from 1 and 2. PPO is a real timestep-level feedback controller:

- **Observation:** normalised skeleton state (22 features — phase, heights, velocities, joint angles, contacts)
- **Action:** `[assist_r, assist_l]` ∈ [0, 1] — idealised knee-extension torque commands at each step
- **Reward:** per-step composite (progress, posture, velocity, muscle, assist effort, contact, symmetry)

The biological STS controller is still running; PPO adds torque on top. Leg control can be scaled down (`--leg-scale`) to simulate impaired users.

```bash
mjpython train_ppo_feedback.py --timesteps 100000 --out logs/ppo_feedback_100k
mjpython train_ppo_feedback.py --timesteps 20000 --tau-max 40 --leg-scale 0.7
mjpython train_ppo_feedback_resume.py --run logs/ppo_feedback_100k --timesteps 50000
```

---

## Configuration (`ctrl_optim/ppo_config.py`)

`TSAOptimConfig` is a nested dataclass used by all three pipelines. Pass as JSON:

```json
{
  "total_timesteps": 10000,
  "env_params": { "L_min": 0.30, "L_max": 0.60, "t_max": 0.55 },
  "reward_params": { "w_muscle": 2.0, "support_fraction": 0.2 },
  "ppo_params": { "learning_rate": 1e-4, "n_steps": 128 }
}
```

Key tuning knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `w_torque` | 0.0 | Torque-tracking weight |
| `w_muscle` | 1.0 | Quadriceps relief weight |
| `support_fraction` | 0.1 | Target exo share of knee demand (α) |
| `t_max` | 0.65 s | Cap on motor activation times — above ~0.6 s motors can miss the STS window |
| `L_min / L_max` | 0.25–0.65 m | String length search bounds |

---

## Training Outputs

Each run creates a directory under `logs/`:

```
logs/ppo_min_control_<timestamp>/
├── config.json                  # Exact config used
├── min_control_components.csv   # Per-episode: reward, R_* components, L, t0..t3
├── best_sampled_theta.json      # Best θ seen during training
├── tensorboard/                 # TensorBoard event files
└── checkpoints/                 # Model checkpoints every 200 episodes
```

---

## Evaluation

### Replay best sampled θ

```bash
mjpython ctrl_optim/eval_best_min_control.py --run logs/ppo_min_control_<timestamp>
```

Loads `best_sampled_theta.json`, runs one rollout with `debug=True` and `log_to_csv=True`, prints reward components and final state.

### Baseline vs assisted comparison

Run two rollouts (one without TSA, one with the optimised θ) and save CSVs with `log_to_csv=True, log_tag="..."` in `base_code.py`. Then:

```bash
python ctrl_optim/ppo_evaluation/compare_baseline_assisted.py \
    --baseline logs/baseline_diag.csv \
    --assisted logs/ppo_best_eval_diag.csv \
    --tsa      logs/ppo_best_eval_tsa.csv \
    --out      logs/comparison_plots/
```

Outputs 5 comparison plots (height, joints, posture, leg stimulation, torso stimulation) and a `muscle_effort_summary.csv` with per-muscle AUC reduction percentages.

### Muscle signal extraction

```bash
mjpython ctrl_optim/extract_muscle_log.py
```

Runs baseline and TSA-enabled rollouts back-to-back, logs VAS/knee demand/delivered torque per step, and prints a per-phase summary table.

### Reward curves

```bash
python ctrl_optim/ppo_evaluation/plot_reward_curve.py --csv logs/ppo_min_control_<timestamp>/min_control_components.csv
python ctrl_optim/ppo_evaluation/plot_ppo_eval.py --csv logs/ppo_min_control_<timestamp>/min_control_components.csv
```

---

## Plotting

### TSA actuator state (`plot_tsa_log.py`)

```bash
mjpython plot_tsa_log.py                       # reads logs/tsa_log.csv
mjpython plot_tsa_log.py path/to/tsa_log.csv
```

Plots knee angle, torque demand vs delivered, number of active motors, per-motor contraction (with X_geom slack threshold), tension, and joint resistance. Phase boundaries are marked.

### Phase 3→4 diagnostics (`plot_diag_p4.py`)

```bash
mjpython plot_diag_p4.py
```

Reads `logs/diag_p4.csv` (written when `debug=True`). Shows pelvis height, root pitch, lean, knee/hip/ankle angles, muscle activations, and pitch velocity. Use this to diagnose why Phase 4 was never reached.

---

## Quick Start

```bash
# 1. Activate environment
source myoassist/.myo-venv/bin/activate

# 2. Test one episode manually
mjpython base_code.py

# 3. Run min-control PPO training
mjpython train_ppo_min_control.py --timesteps 5000 --out logs/test_run

# 4. Evaluate the best parameters found
mjpython ctrl_optim/eval_best_min_control.py --run logs/test_run
```
