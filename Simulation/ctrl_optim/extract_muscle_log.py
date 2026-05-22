"""
Reward signal extraction and verification script.

Runs two STS rollouts (baseline vs TSA-enabled) and logs per-step:
  time, phase, vasti_r, vasti_l, rectfem_r, rectfem_l,
  tau_demand_r, tau_demand_l, tau_del_r, tau_del_l

Output:
  logs/signal_extraction_log.csv  — full per-step log (both conditions)
  Console summary table           — per-phase means for each signal
  Quick comparison                — TSA vs baseline activation in phases 2-3

Run with:  mjpython ctrl_optim/extract_muscle_log.py
"""

import os
import signal
import sys
import csv
from pathlib import Path

import numpy as np

REPO_ROOT     = Path(__file__).resolve().parent.parent
MYOASSIST_DIR = REPO_ROOT / "myoassist"
CTRL_OPTIM_DIR = REPO_ROOT / "ctrl_optim"

for _d in [str(MYOASSIST_DIR), str(CTRL_OPTIM_DIR)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym
from sts_ctrl import SitToStandSim


def set_seated_pose(sim):
    model = sim.model
    data  = sim.data
    name2jid = {model.joint(i).name: i for i in range(model.njnt)}

    def set_joint(name, value):
        qadr = model.jnt_qposadr[name2jid[name]]
        data.qpos[qadr] = value

    set_joint("root_x",       0.0)
    set_joint("root_z",       0.0)
    set_joint("root_pitch",   0.0)
    set_joint("hip_flexion_r", 1.57)
    set_joint("hip_flexion_l", 1.57)
    set_joint("knee_angle_r",  1.75)
    set_joint("knee_angle_l",  1.75)
    set_joint("ankle_angle_r", -0.15)
    set_joint("ankle_angle_l", -0.15)
    data.qvel[:] = 0.0
    sim.forward()


TIMEOUT_S       = 5.0    # abort rollout if STS not achieved within this many sim-seconds
ACTIVATION_GAIN = 10.0   # alpha: scales activation penalty in reward (Option 1)


def run_rollout(env, use_tsa_full: bool, max_steps: int = 6000) -> list[dict]:
    """Run one STS episode and return a list of per-step signal dicts."""
    env.reset(seed=0)
    set_seated_pose(env.sim)

    sts = SitToStandSim(env.sim, env, debug=False, use_tsa_full=use_tsa_full)
    sts.reset_filters()
    sts.get_observation()
    sts.capture_phase1_hold_pose()
    sts.reset_phase(1)
    if use_tsa_full and sts.tsa is not None:
        sts.tsa.reset()

    sim   = env.sim
    model = sim.model

    # Quad actuator IDs in myotorso_exosuit model (myolegs_muscle.xml)
    # VL = vaslat, VM = vasmed, VI = vasint, RF = recfem (not rectfem)
    vaslat_r_id = model.actuator_name2id("vaslat_r")
    vaslat_l_id = model.actuator_name2id("vaslat_l")
    vasmed_r_id = model.actuator_name2id("vasmed_r")
    vasmed_l_id = model.actuator_name2id("vasmed_l")
    vasint_r_id = model.actuator_name2id("vasint_r")
    vasint_l_id = model.actuator_name2id("vasint_l")
    recfem_r_id = model.actuator_name2id("recfem_r")
    recfem_l_id = model.actuator_name2id("recfem_l")

    rows = []
    for _ in range(max_steps):
        obs   = sts.get_observation()
        phase = sts.get_phase()

        sts.step(None, phase)

        t = float(sim.data.time)

        vaslat_r = float(sim.data.act[vaslat_r_id])
        vaslat_l = float(sim.data.act[vaslat_l_id])
        vasmed_r = float(sim.data.act[vasmed_r_id])
        vasmed_l = float(sim.data.act[vasmed_l_id])
        vasint_r = float(sim.data.act[vasint_r_id])
        vasint_l = float(sim.data.act[vasint_l_id])
        recfem_r = float(sim.data.act[recfem_r_id])
        recfem_l = float(sim.data.act[recfem_l_id])

        if use_tsa_full and sts.tsa is not None:
            s         = sts.tsa.last_state
            tau_del_r = float(s.get('r', {}).get('torque', 0.0))
            tau_del_l = float(s.get('l', {}).get('torque', 0.0))
            tau_dem_r = float(getattr(sts, '_last_tau_r', 0.0))
            tau_dem_l = float(getattr(sts, '_last_tau_l', 0.0))
        else:
            tau_del_r = tau_del_l = tau_dem_r = tau_dem_l = 0.0

        rows.append({
            'time':         round(t, 4),
            'phase':        phase,
            'vaslat_r':     round(vaslat_r, 5),   # VL right (2 motors assigned)
            'vaslat_l':     round(vaslat_l, 5),
            'vasmed_r':     round(vasmed_r, 5),   # VM right (1 motor assigned)
            'vasmed_l':     round(vasmed_l, 5),
            'vasint_r':     round(vasint_r, 5),   # VI right (no motor, logged for context)
            'vasint_l':     round(vasint_l, 5),
            'recfem_r':     round(recfem_r, 5),   # RF right (1 motor assigned)
            'recfem_l':     round(recfem_l, 5),
            'tau_demand_r': round(tau_dem_r, 4),
            'tau_demand_l': round(tau_dem_l, 4),
            'tau_del_r':    round(tau_del_r, 4),
            'tau_del_l':    round(tau_del_l, 4),
        })

        if sts.get_phase() >= 4 or t >= TIMEOUT_S:
            break

    sts.close()
    return rows


def _phase_mean(rows: list[dict], phase_num: int, key: str) -> float:
    vals = [r[key] for r in rows if r['phase'] == phase_num]
    return float(np.mean(vals)) if vals else float('nan')


def summarise(rows: list[dict], label: str) -> None:
    phases  = sorted({r['phase'] for r in rows})
    # Show the 3 motor-assigned muscles per side + torque delivery
    signals = ['vaslat_r', 'vasmed_r', 'recfem_r', 'vaslat_l', 'vasmed_l', 'recfem_l',
               'tau_del_r', 'tau_del_l']

    print(f"\n{'='*90}")
    print(f"  {label}  ({len(rows)} steps,  final time {rows[-1]['time']:.2f} s)")
    print(f"{'='*90}")

    col_w = 10
    header = f"  {'Ph':>3}  " + "  ".join(f"{s:>{col_w}}" for s in signals)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for p in phases:
        means = {s: _phase_mean(rows, p, s) for s in signals}
        row_str = f"  {p:>3}  " + "  ".join(f"{means[s]:>{col_w}.4f}" for s in signals)
        print(row_str)

    dt = 0.01
    act_integral = sum(
        (r['vaslat_r'] + r['vasmed_r'] + r['recfem_r'] +
         r['vaslat_l'] + r['vasmed_l'] + r['recfem_l']) * dt
        for r in rows
    )
    tracking_integral = sum(
        (min(r['tau_del_r'], r['tau_demand_r']) + min(r['tau_del_l'], r['tau_demand_l'])) * dt
        for r in rows
    )
    print(f"\n  Activation integral  (VL+VM+RF both legs, Σ×dt):         {act_integral:.3f}")
    print(f"  Activation reward    (×{ACTIVATION_GAIN:.0f} gain, raw signal for PPO):  "
          f"{-ACTIVATION_GAIN * act_integral:.3f}")
    print(f"  Torque tracking      (Σ min(del,dem) × dt):               {tracking_integral:.3f} Nm·s")


def print_comparison(rows_base: list[dict], rows_tsa: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  Extension phase comparison (phases 2 + 3)")
    print(f"{'='*72}")

    muscles = [
        ('vaslat', 'VL (2 motors)'),
        ('vasmed', 'VM (1 motor) '),
        ('recfem', 'RF (1 motor) '),
    ]

    for p in [2, 3]:
        print(f"\n  Phase {p}:")
        for key, label in muscles:
            base = _phase_mean(rows_base, p, f'{key}_r') + _phase_mean(rows_base, p, f'{key}_l')
            tsa  = _phase_mean(rows_tsa,  p, f'{key}_r') + _phase_mean(rows_tsa,  p, f'{key}_l')
            diff = tsa - base
            direction = 'reduced' if diff < 0 else 'INCREASED'
            print(f"    {label}  base={base:.4f}  tsa={tsa:.4f}  "
                  f"diff={diff:+.4f}  scaled={-ACTIVATION_GAIN * diff:+.4f}  ({direction})")

    base_t = rows_base[-1]['time']
    tsa_t  = rows_tsa[-1]['time']
    print(f"\n  Total time:  baseline={base_t:.2f}s  tsa={tsa_t:.2f}s  diff={tsa_t - base_t:+.2f}s")


def main() -> None:
    signal.signal(signal.SIGINT,  lambda _s, _f: os._exit(0))
    signal.signal(signal.SIGTERM, lambda _s, _f: os._exit(0))

    env = gym.make('TorsoLegs')

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    print("Running baseline rollout (no TSA)...")
    rows_base = run_rollout(env, use_tsa_full=False)
    timed_out = rows_base[-1]['time'] >= TIMEOUT_S and rows_base[-1]['phase'] < 4
    print(f"  Done — {len(rows_base)} steps, reached phase {rows_base[-1]['phase']}"
          + ("  [TIMED OUT]" if timed_out else ""))

    print("Running TSA rollout...")
    rows_tsa = run_rollout(env, use_tsa_full=True)
    timed_out = rows_tsa[-1]['time'] >= TIMEOUT_S and rows_tsa[-1]['phase'] < 4
    print(f"  Done — {len(rows_tsa)} steps, reached phase {rows_tsa[-1]['phase']}"
          + ("  [TIMED OUT]" if timed_out else ""))

    csv_path = logs_dir / "signal_extraction_log.csv"
    fieldnames = list(rows_base[0].keys()) + ['condition']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_base:
            w.writerow({**r, 'condition': 'baseline'})
        for r in rows_tsa:
            w.writerow({**r, 'condition': 'tsa'})

    print(f"\nCSV saved → {csv_path}")

    summarise(rows_base, "BASELINE (no TSA)")
    summarise(rows_tsa,  "TSA ENABLED")
    print_comparison(rows_base, rows_tsa)


if __name__ == '__main__':
    main()
