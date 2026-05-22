"""
Plot TSA actuator log.

Usage
-----
    python plot_tsa_log.py                          # reads logs/tsa_log.csv
    python plot_tsa_log.py logs/full/tsa_log_full.csv

Image is saved next to the CSV with the same stem + _plot.png.
    logs/tsa_log.csv          → logs/plots/tsa_log_plot.png
    logs/full/tsa_log_full.csv → logs/full/plots/tsa_log_full_plot.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    CSV_PATH = Path(sys.argv[1])
    if not CSV_PATH.is_absolute():
        CSV_PATH = ROOT / CSV_PATH
else:
    CSV_PATH = ROOT / "logs" / "tsa_log.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Log file not found: {CSV_PATH}")

OUT_DIR = CSV_PATH.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / (CSV_PATH.stem + "_plot.png")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv(CSV_PATH)
cols_present = set(df.columns)

IS_FULL = "N_active_r" in cols_present   # full 4-motor log

t     = df["time"].to_numpy()
phase = df["phase"].to_numpy()

phase_changes = np.where(np.diff(phase) != 0)[0]
phase_times   = t[phase_changes + 1]
phase_edges   = np.concatenate([[t[0]], phase_times, [t[-1]]])
unique_phases = [phase[0]] + list(phase[phase_changes + 1])

COLORS = {"r": "#1f77b4", "l": "#ff7f0e"}
STYLES = {"r": "-",       "l": "--"}
LABELS = {"r": "Right",   "l": "Left"}

MOTOR_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]   # M0–M3
MOTOR_STYLES = ["-", "--", "-.", ":"]

# ---------------------------------------------------------------------------
# Helper: robust y-limits
# ---------------------------------------------------------------------------

def _ylim(ax, *arrays):
    all_vals = np.concatenate([a.ravel() for a in arrays])
    finite   = all_vals[np.isfinite(all_vals)]
    if len(finite) == 0:
        return
    lo = np.percentile(finite, 1)
    hi = np.percentile(finite, 99)
    pad = 0.12 * max(hi - lo, 1e-6)
    ax.set_ylim(max(lo - pad, 0), hi + pad)


def _phase_lines(ax):
    for pt in phase_times:
        ax.axvline(pt, color="grey", lw=0.8, ls=":", alpha=0.7)


def _phase_labels(ax):
    for i, p in enumerate(unique_phases):
        mid = (phase_edges[i] + phase_edges[i + 1]) / 2
        ax.text(mid, 1.02, f"P{p}", ha="center", va="bottom",
                fontsize=8, color="grey",
                transform=ax.get_xaxis_transform())


def _finish_ax(ax, title, unit):
    ax.set_ylabel(f"{title}\n({unit})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _phase_lines(ax)


# ---------------------------------------------------------------------------
# Full 4-motor plot
# ---------------------------------------------------------------------------

if IS_FULL:
    fig, axes = plt.subplots(8, 1, figsize=(13, 28), sharex=True)
    fig.suptitle(f"TSA Log (4-motor) — {CSV_PATH.name}",
                 fontsize=13, fontweight="bold", y=0.997)

    ax = axes[0]
    for side in ("r", "l"):
        ax.plot(t, df[f"knee_{side}_rad"], color=COLORS[side],
                ls=STYLES[side], lw=1.4, label=LABELS[side])
    _finish_ax(ax, "Knee angle", "rad")

    ax = axes[1]
    for side in ("r", "l"):
        ax.plot(t, df[f"tau_demand_{side}_Nm"], color=COLORS[side],
                ls=STYLES[side], lw=1.4, label=f"Demand {LABELS[side]}")
        ax.plot(t, df[f"total_torque_{side}_Nm"], color=COLORS[side],
                ls=STYLES[side], lw=2.0, alpha=0.5,
                label=f"Delivered {LABELS[side]}")
    _finish_ax(ax, "Torque", "N·m")

    ax = axes[2]
    for side in ("r", "l"):
        ax.step(t, df[f"N_active_{side}"], color=COLORS[side],
                ls=STYLES[side], lw=1.6, where="post", label=LABELS[side])
    ax.set_yticks(range(5))
    _finish_ax(ax, "Motors active", "#")

    # -- Contraction (right & left): shared y-limits so both panels are comparable
    _X_all = np.concatenate([
        df[c].to_numpy()
        for c in ([f"r_m{mi}_X_mm" for mi in range(4)]
                  + [f"l_m{mi}_X_mm" for mi in range(4)]
                  + ["X_geom_r_mm", "X_geom_l_mm"])
        if c in cols_present
    ])

    ax = axes[3]
    for mi in range(4):
        col = f"r_m{mi}_X_mm"
        if col in cols_present:
            ax.plot(t, df[col], color=MOTOR_COLORS[mi], ls=MOTOR_STYLES[mi],
                    lw=1.3, label=f"M{mi}")
    if "X_geom_r_mm" in cols_present:
        ax.plot(t, df["X_geom_r_mm"], color="black", ls="--", lw=1.6,
                label="X_geom (taut threshold)", alpha=0.75)
    _ylim(ax, _X_all)
    _finish_ax(ax, "Contraction (right)", "mm")

    ax = axes[4]
    for mi in range(4):
        col = f"l_m{mi}_X_mm"
        if col in cols_present:
            ax.plot(t, df[col], color=MOTOR_COLORS[mi], ls=MOTOR_STYLES[mi],
                    lw=1.3, label=f"M{mi}")
    if "X_geom_l_mm" in cols_present:
        ax.plot(t, df["X_geom_l_mm"], color="black", ls="--", lw=1.6,
                label="X_geom (taut threshold)", alpha=0.75)
    _ylim(ax, _X_all)
    _finish_ax(ax, "Contraction (left)", "mm")

    # -- Tension (right & left): shared y-limits
    _T_all = np.concatenate([
        df[c].to_numpy()
        for c in ([f"r_m{mi}_tension_N" for mi in range(4)]
                  + [f"l_m{mi}_tension_N" for mi in range(4)])
        if c in cols_present
    ]) if any(f"r_m{mi}_tension_N" in cols_present for mi in range(4)) else np.array([0.0])

    ax = axes[5]
    for mi in range(4):
        col = f"r_m{mi}_tension_N"
        if col in cols_present:
            ax.plot(t, df[col], color=MOTOR_COLORS[mi], ls=MOTOR_STYLES[mi],
                    lw=1.3, label=f"M{mi}")
    _ylim(ax, _T_all)
    _finish_ax(ax, "Tension (right)", "N")

    ax = axes[6]
    for mi in range(4):
        col = f"l_m{mi}_tension_N"
        if col in cols_present:
            ax.plot(t, df[col], color=MOTOR_COLORS[mi], ls=MOTOR_STYLES[mi],
                    lw=1.3, label=f"M{mi}")
    _ylim(ax, _T_all)
    _finish_ax(ax, "Tension (left)", "N")

    ax = axes[7]
    for side in ("r", "l"):
        ax.plot(t, df[f"F_resist_{side}_N"], color=COLORS[side],
                ls=STYLES[side], lw=1.4, label=LABELS[side])
    _finish_ax(ax, "Joint resistance", "N")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    _phase_labels(axes[0])

# ---------------------------------------------------------------------------
# Single-motor plot (original layout)
# ---------------------------------------------------------------------------

else:
    VARS = [
        ("Knee joint angle",      "rad",    ["knee_r_rad",          "knee_l_rad"]),
        ("Desired torque",         "N·m",   ["tau_demand_r_Nm",     "tau_demand_l_Nm"]),
        ("Cable tension",          "N",     ["tension_r_N",         "tension_l_N"]),
        ("Torque delivered",       "N·m",   ["torque_r_Nm",         "torque_l_Nm"]),
        ("Contraction",            "mm",    ["X_r_mm",              "X_l_mm"]),
        ("Motor angle",            "rad",   ["theta_r_rad",         "theta_l_rad"]),
        ("Motor speed",            "rad/s", ["theta_dot_r_rads",    "theta_dot_l_rads"]),
        ("Effective payload mass", "kg",    ["payload_mass_r_kg",   "payload_mass_l_kg"]),
        ("Joint resistance force", "N",     ["F_resist_r_N",        "F_resist_l_N"]),
    ]

    fig, axes = plt.subplots(len(VARS), 1, figsize=(12, 24), sharex=True)
    fig.suptitle(f"TSA Actuator Log — {CSV_PATH.name}",
                 fontsize=14, fontweight="bold", y=0.995)

    for ax, (title, unit, plot_cols) in zip(axes, VARS):
        for col, side in zip(plot_cols, ("r", "l")):
            if col in cols_present:
                ax.plot(t, df[col].to_numpy(),
                        color=COLORS[side], ls=STYLES[side],
                        lw=1.4, label=LABELS[side])

        all_vals = np.concatenate([df[c].to_numpy() for c in plot_cols if c in cols_present])
        _ylim(ax, all_vals)
        _finish_ax(ax, title, unit)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    _phase_labels(axes[0])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()
