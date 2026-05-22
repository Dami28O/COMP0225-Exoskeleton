from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_diag(diag_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(diag_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = df["time"]

    # 1) Heights
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["pelvis_z"], label="pelvis_z")
    plt.plot(t, df["torso_z"] if "torso_z" in df.columns else df["pelvis_z"], label="torso_z")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Sit-to-stand body height")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "height_profile.png", dpi=150)
    plt.close()

    # 2) Root pitch / lean
    plt.figure(figsize=(9, 5))
    if "root_pitch" in df.columns:
        plt.plot(t, df["root_pitch"], label="root_pitch")
    if "lean" in df.columns:
        plt.plot(t, df["lean"], label="trunk_lean_rel")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle / lean metric")
    plt.title("Torso/root posture")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "torso_posture.png", dpi=150)
    plt.close()

    # 3) Hip/knee/ankle
    plt.figure(figsize=(9, 5))
    if "hip" in df.columns:
        plt.plot(t, df["hip"], label="hip_avg")
    if "knee_r" in df.columns:
        plt.plot(t, df["knee_r"], label="knee_r")
    if "knee_l" in df.columns:
        plt.plot(t, df["knee_l"], label="knee_l")
    if "ankle" in df.columns:
        plt.plot(t, df["ankle"], label="ankle_avg")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title("Lower-limb joint angles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "joint_angles.png", dpi=150)
    plt.close()

    # 4) Muscle stim
    stim_cols = [
        "stim_GLU",
        "stim_VAS",
        "stim_HAM",
        "stim_HFL",
        "stim_TORSO_EXT",
        "stim_TORSO_FLEX",
    ]
    existing = [c for c in stim_cols if c in df.columns]

    if existing:
        plt.figure(figsize=(9, 5))
        for c in existing:
            plt.plot(t, df[c], label=c)
        plt.xlabel("Time (s)")
        plt.ylabel("Stimulation")
        plt.title("Muscle stimulation during evaluation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "muscle_stimulation.png", dpi=150)
        plt.close()

    # 5) Phase
    if "phase" in df.columns:
        plt.figure(figsize=(9, 3))
        plt.step(t, df["phase"], where="post")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase")
        plt.title("STS phase progression")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "phase_progression.png", dpi=150)
        plt.close()


def plot_tsa(tsa_csv: Path, out_dir: Path) -> None:
    if not tsa_csv.exists():
        print(f"[warn] TSA CSV not found: {tsa_csv}")
        return

    df = pd.read_csv(tsa_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = df["time"]

    # Total TSA torque
    plt.figure(figsize=(9, 5))
    if "total_torque_r_Nm" in df.columns:
        plt.plot(t, df["total_torque_r_Nm"], label="right TSA torque")
    if "total_torque_l_Nm" in df.columns:
        plt.plot(t, df["total_torque_l_Nm"], label="left TSA torque")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("TSA knee assistance torque")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tsa_torque.png", dpi=150)
    plt.close()

    # Active motors
    plt.figure(figsize=(9, 4))
    if "N_active_r" in df.columns:
        plt.step(t, df["N_active_r"], where="post", label="right active motors")
    if "N_active_l" in df.columns:
        plt.step(t, df["N_active_l"], where="post", label="left active motors")
    plt.xlabel("Time (s)")
    plt.ylabel("N active")
    plt.title("Number of active TSA motors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tsa_active_motors.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", required=True, help="Path to *_diag.csv")
    parser.add_argument("--tsa", default=None, help="Path to *_tsa_log_full.csv")
    parser.add_argument("--out", default="logs/ppo_eval_plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    plot_diag(Path(args.diag), out_dir)

    if args.tsa is not None:
        plot_tsa(Path(args.tsa), out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()