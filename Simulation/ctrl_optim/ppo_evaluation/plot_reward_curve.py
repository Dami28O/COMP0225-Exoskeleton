from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average with same length as input."""
    if len(values) < 2:
        return values

    window = max(1, min(window, len(values)))
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def find_existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def plot_reward_breakdown(
    csv_path: str | Path,
    out_path: str | Path,
    max_episodes: int | None = None,
    smooth_window: int = 20,
) -> None:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "episode" not in df.columns:
        df["episode"] = np.arange(1, len(df) + 1)

    if max_episodes is not None:
        df = df.iloc[:max_episodes].copy()

    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")

    # These are the components from your actual minimal-control PPO reward:
    # R = w_success R_success
    #   + w_track_pos R_track_pos
    #   + w_track_vel R_track_vel
    #   + w_muscle R_muscle
    #   + w_tsa R_tsa
    #   + w_stability R_stability
    #   + w_time R_time
    reward_cols = find_existing_columns(
        df,
        [
            "reward",
            "R_success",
            "R_track_pos",
            "R_track_vel",
            "R_muscle",
            "R_tsa",
            "R_stability",
            "R_time",
            "t_final",
        ],
    )

    if not reward_cols:
        raise ValueError(
            f"No recognised reward columns found. CSV columns are:\n{list(df.columns)}"
        )

    episodes = df["episode"].to_numpy(dtype=float)

    n = len(reward_cols)
    fig_height = max(2.2 * n, 8)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(11, fig_height),
        sharex=True,
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    fig.suptitle(
        "PPO Training — Reward Components per Episode",
        fontsize=10,
        y=1.005,
    )

    for ax, col in zip(axes, reward_cols):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(episodes) & np.isfinite(values)
        ep = episodes[valid]
        vals = values[valid]

        if len(vals) == 0:
            continue

        smoothed = smooth(vals, smooth_window)

        # Raw values faint, smoothed values bold.
        ax.plot(
            ep,
            vals,
            linewidth=0.7,
            alpha=0.22,
            label=f"Raw {col}",
        )
        ax.plot(
            ep,
            smoothed,
            linewidth=2.0,
            label=f"Smoothed {col}",
        )

        # Highlight best total reward episode only on total reward plot.
        if col == "reward":
            best_idx = int(np.nanargmax(vals))
            ax.scatter(
                [ep[best_idx]],
                [vals[best_idx]],
                s=70,
                zorder=5,
                label=f"Best episode {int(ep[best_idx])}",
            )

        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Episode")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reward breakdown plot to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to min_control_components.csv or feedback training CSV.",
    )
    parser.add_argument(
        "--out",
        default="logs/reward_breakdown/reward_components_breakdown.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional number of episodes to plot, e.g. 64.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Moving-average smoothing window.",
    )

    args = parser.parse_args()

    plot_reward_breakdown(
        csv_path=args.csv,
        out_path=args.out,
        max_episodes=args.max_episodes,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()