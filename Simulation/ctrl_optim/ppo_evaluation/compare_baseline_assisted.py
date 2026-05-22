from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auc_time(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    t = df["time"].to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    return float(np.trapz(y, t))


def interp_to_common_time(df: pd.DataFrame, col: str, t_common: np.ndarray) -> np.ndarray:
    t = df["time"].to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    return np.interp(t_common, t, y)


def plot_compare(base: pd.DataFrame, assist: pd.DataFrame, cols, title, ylabel, out_path: Path):
    t_end = min(float(base["time"].max()), float(assist["time"].max()))
    t_common = np.linspace(0.0, t_end, 600)

    plt.figure(figsize=(9, 5))

    for col in cols:
        if col not in base.columns or col not in assist.columns:
            print(f"[warn] missing {col}; skipping")
            continue

        y_base = interp_to_common_time(base, col, t_common)
        y_assist = interp_to_common_time(assist, col, t_common)

        plt.plot(t_common, y_base, linestyle="--", label=f"{col} baseline")
        plt.plot(t_common, y_assist, label=f"{col} assisted")

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsa(tsa_path: Path, out_dir: Path):
    if not tsa_path.exists():
        print(f"[warn] TSA file not found: {tsa_path}")
        return

    tsa = pd.read_csv(tsa_path)
    t = tsa["time"]

    plt.figure(figsize=(9, 5))
    if "total_torque_r_Nm" in tsa.columns:
        plt.plot(t, tsa["total_torque_r_Nm"], label="right TSA torque")
    if "total_torque_l_Nm" in tsa.columns:
        plt.plot(t, tsa["total_torque_l_Nm"], label="left TSA torque")

    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("TSA assistance torque")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tsa_torque.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    if "N_active_r" in tsa.columns:
        plt.step(t, tsa["N_active_r"], where="post", label="right active motors")
    if "N_active_l" in tsa.columns:
        plt.step(t, tsa["N_active_l"], where="post", label="left active motors")

    plt.xlabel("Time (s)")
    plt.ylabel("Number active")
    plt.title("Active TSA motors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tsa_active_motors.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="No-TSA baseline diag CSV")
    parser.add_argument("--assisted", required=True, help="PPO/TSA assisted diag CSV")
    parser.add_argument("--tsa", default=None, help="PPO/TSA assisted TSA CSV")
    parser.add_argument("--out", default="logs/baseline_vs_assisted_plots")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    assisted_path = Path(args.assisted)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(baseline_path)
    assist = pd.read_csv(assisted_path)

    # Main visual comparisons
    plot_compare(
        base, assist,
        cols=["pelvis_z", "torso_z"],
        title="Body height: baseline vs assisted",
        ylabel="Height (m)",
        out_path=out_dir / "height_baseline_vs_assisted.png",
    )

    plot_compare(
        base, assist,
        cols=["hip", "knee_avg", "ankle"],
        title="Joint angles: baseline vs assisted",
        ylabel="Joint angle (rad)",
        out_path=out_dir / "joints_baseline_vs_assisted.png",
    )

    plot_compare(
        base, assist,
        cols=["root_pitch", "lean"],
        title="Posture: baseline vs assisted",
        ylabel="Angle / lean metric",
        out_path=out_dir / "posture_baseline_vs_assisted.png",
    )

    plot_compare(
        base, assist,
        cols=["stim_VAS", "stim_GLU", "stim_SOL"],
        title="Leg stimulation: baseline vs assisted",
        ylabel="Stimulation",
        out_path=out_dir / "leg_stim_baseline_vs_assisted.png",
    )

    plot_compare(
        base, assist,
        cols=["stim_TORSO_EXT", "stim_TORSO_FLEX"],
        title="Torso stimulation: baseline vs assisted",
        ylabel="Stimulation",
        out_path=out_dir / "torso_stim_baseline_vs_assisted.png",
    )

    # Numerical effort comparison
    muscle_cols = [
        "stim_VAS",
        "stim_GLU",
        "stim_SOL",
        "stim_TA",
        "stim_HAM",
        "stim_HFL",
        "stim_TORSO_EXT",
        "stim_TORSO_FLEX",
    ]

    rows = []
    for col in muscle_cols:
        if col not in base.columns or col not in assist.columns:
            continue

        e_base = auc_time(base, col)
        e_assist = auc_time(assist, col)

        if abs(e_base) > 1e-9:
            reduction = 100.0 * (e_base - e_assist) / e_base
        else:
            reduction = float("nan")

        rows.append({
            "signal": col,
            "baseline_integral": e_base,
            "assisted_integral": e_assist,
            "percent_reduction": reduction,
        })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "muscle_effort_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n[MUSCLE EFFORT SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")

    if args.tsa is not None:
        plot_tsa(Path(args.tsa), out_dir)

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()