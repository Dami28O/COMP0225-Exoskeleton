import csv
import os
import signal
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("agg")          # mjpython owns the macOS display thread
import matplotlib.pyplot as plt

MYOASSIST_DIR = Path(__file__).resolve().parent / "myoassist"
if str(MYOASSIST_DIR) not in sys.path:
    sys.path.insert(0, str(MYOASSIST_DIR))

CTRL_OPTIM_DIR = Path(__file__).resolve().parent / "ctrl_optim"
if str(CTRL_OPTIM_DIR) not in sys.path:
    sys.path.insert(0, str(CTRL_OPTIM_DIR))

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym
from sts_ctrl import SitToStandSim, STSReflexParams

env = gym.make('TorsoLegs')
env.reset(seed=0)

max_steps = 50000

# -- Knee DOF addresses for torque demand readout --
_name2jid   = {env.sim.model.joint(i).name: i for i in range(env.sim.model.njnt)}
_knee_dadr  = {
    "r": int(env.sim.model.jnt_dofadr[_name2jid["knee_angle_r"]]),
    "l": int(env.sim.model.jnt_dofadr[_name2jid["knee_angle_l"]]),
}

# Cable geometry constant (matches tsa_integration_full.py CABLE_PATH_ARM)
_CABLE_PATH_ARM = 0.025  # m

# Per-step log and initial knee angle reference
_log: list = []
_knee_initial: dict | None = None


def set_seated_pose(sim):
    model = sim.model
    data  = sim.data
    name2jid = {model.joint(i).name: i for i in range(model.njnt)}

    def set_joint(name, val):
        jid  = name2jid[name]
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = val

    set_joint("root_x",        0.0)
    set_joint("root_z",        0.0)
    set_joint("root_pitch",    0.0)
    set_joint("hip_flexion_r", 1.57)
    set_joint("hip_flexion_l", 1.57)
    set_joint("knee_angle_r",  1.75)
    set_joint("knee_angle_l",  1.75)
    set_joint("ankle_angle_r", -0.15)
    set_joint("ankle_angle_l", -0.15)

    data.qvel[:] = 0.0
    sim.forward()

params = STSReflexParams(tsa_string_length=0.25)   # default is 0.50
sts = SitToStandSim(env.sim, env, debug=True, use_tsa_full=False, use_tsa=False, params=params, log_to_csv=False)
# sts = SitToStandSim(env.sim, env, debug=True)

# set_seated_pose(env.sim)
sts.reset_filters()
sts.get_observation()
sts.capture_phase1_hold_pose()
sts.reset_phase(1)
# if sts.tsa is not None:
#     sts.tsa.reset()


def _save_and_plot() -> None:
    if not _log:
        return

    logs_dir = Path(__file__).resolve().parent / "logs" / "baseline"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / "baseline_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_log[0].keys()))
        w.writeheader()
        w.writerows(_log)
    print(f"Baseline CSV  → {csv_path}")

    t          = np.array([r["time"]  for r in _log])
    phase_arr  = np.array([r["phase"] for r in _log])

    COLORS = {"r": "#e07b39", "l": "#3974e0"}
    STYLES = {"r": "-",       "l": "--"}
    LABELS = {"r": "Right",   "l": "Left"}

    def _phase_lines(ax):
        seen = set()
        for p in np.unique(phase_arr):
            idx = np.where(phase_arr == p)[0][0]
            ax.axvline(t[idx], color="grey", lw=0.8, ls=":", alpha=0.7)
            if p not in seen:
                ax.text(t[idx] + 0.02, ax.get_ylim()[1] * 0.97,
                        f"P{int(p)}", fontsize=7, color="grey", va="top")
                seen.add(p)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Baseline STS — no TSA", fontsize=13, fontweight="bold")

    # Panel 1 — Joint angles
    ax = axes[0]
    for side in ("r", "l"):
        ax.plot(t, [r[f"knee_{side}_rad"]  for r in _log],
                color=COLORS[side], ls=STYLES[side], lw=1.4, label=f"Knee {LABELS[side]}")
        ax.plot(t, [r[f"hip_{side}_rad"]   for r in _log],
                color=COLORS[side], ls=":",           lw=1.0, label=f"Hip {LABELS[side]}")
        ax.plot(t, [r[f"ankle_{side}_rad"] for r in _log],
                color=COLORS[side], ls="-.",           lw=1.0, label=f"Ankle {LABELS[side]}")
    _phase_lines(ax)
    ax.set_ylabel("Joint angle (rad)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2 — Knee torque demand
    ax = axes[1]
    for side in ("r", "l"):
        ax.plot(t, [r[f"tau_demand_{side}_Nm"] for r in _log],
                color=COLORS[side], ls=STYLES[side], lw=1.4, label=LABELS[side])
    _phase_lines(ax)
    ax.set_ylabel("Knee torque demand (N·m)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3 — Geometric contraction
    ax = axes[2]
    for side in ("r", "l"):
        ax.plot(t, [r[f"X_geom_{side}_mm"] for r in _log],
                color=COLORS[side], ls=STYLES[side], lw=1.4, label=LABELS[side])
    _phase_lines(ax)
    ax.set_ylabel("Geometric contraction X_geom (mm)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()
    plot_path = logs_dir / "baseline_plot.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline plot → {plot_path}")


def _shutdown(_sig=None, _frame=None):
    try:
        sts.close()
    except Exception:
        pass
    _save_and_plot()
    # os._exit bypasses mjpython's event loop — sys.exit raises SystemExit
    # which mjpython catches and uses to restart the script instead of quitting.
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

try:
    phase = 1
    for step_i in range(max_steps):
        env.mj_render()
        obs   = sts.get_observation()
        phase = sts.get_phase()

        # Latch initial knee angles on first valid observation
        if _knee_initial is None and obs.get("knee_r") is not None:
            _knee_initial = {"r": float(obs["knee_r"]), "l": float(obs["knee_l"])}

        # Geometric cable contraction threshold: grows as knee extends from seated
        if _knee_initial is not None:
            xg_r = _CABLE_PATH_ARM * max(0.0, _knee_initial["r"] - float(obs.get("knee_r", 0.0))) * 1e3
            xg_l = _CABLE_PATH_ARM * max(0.0, _knee_initial["l"] - float(obs.get("knee_l", 0.0))) * 1e3
        else:
            xg_r = xg_l = 0.0

        # Knee extension demand from gravity + Coriolis (same source as tsa_integration_full)
        tau_r = max(0.0, float(env.sim.data.qfrc_bias[_knee_dadr["r"]]))
        tau_l = max(0.0, float(env.sim.data.qfrc_bias[_knee_dadr["l"]]))

        _log.append({
            "time":             float(env.sim.data.time),
            "phase":            int(phase),
            "knee_r_rad":       float(obs.get("knee_r",   float("nan"))),
            "knee_l_rad":       float(obs.get("knee_l",   float("nan"))),
            "hip_r_rad":        float(obs.get("hip_r",    float("nan"))),
            "hip_l_rad":        float(obs.get("hip_l",    float("nan"))),
            "ankle_r_rad":      float(obs.get("ankle_r",  float("nan"))),
            "ankle_l_rad":      float(obs.get("ankle_l",  float("nan"))),
            "tau_demand_r_Nm":  tau_r,
            "tau_demand_l_Nm":  tau_l,
            "X_geom_r_mm":      xg_r,
            "X_geom_l_mm":      xg_l,
        })

        sts.step(None, phase)

except (KeyboardInterrupt, SystemExit):
    pass

_shutdown()
