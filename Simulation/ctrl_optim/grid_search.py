"""
5 × 5 grid search over (L, t_stagger) to validate the reward landscape
before committing to PPO training.

Symmetric activation: M0 at 0 s, M1 at t_stagger, M2 at 2·t_stagger,
                      M3 at 3·t_stagger.

Run:
    mjpython grid_search.py

Output: logs/grid_search_results.csv
        Columns: L, t_stagger, reward, R_torque, R_muscle, t0, t1, t2, t3
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
_CTRL_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CTRL_DIR.parent

for _p in [str(_CTRL_DIR), str(_ROOT_DIR / "tsa_modelling"), str(_ROOT_DIR / "myoassist")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import myosuite.envs.myo.myobase  # noqa: F401  — registers 'TorsoLegs'
from myosuite.utils import gym

from ppo_config import TSAOptimConfig
from ppo_wrapper import TSAOptimEnv

# ── grid definition ─────────────────────────────────────────────────────────
L_VALUES       = [0.30, 0.40, 0.50, 0.60, 0.70]
T_STAGGER_VALUES = [0.0,  0.25, 0.50, 0.75, 1.0]

# ── output ───────────────────────────────────────────────────────────────────
OUT_PATH = _ROOT_DIR / "logs" / "grid_search_results.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mj_env = gym.make('TorsoLegs')

    config = TSAOptimConfig()
    # Enable muscle reward for grid search so we get both signals.
    config.reward_params.w_torque = 1.0
    config.reward_params.w_muscle = 0.5
    config.reward_params.w_time   = 0.0

    env = TSAOptimEnv(mj_env, config)

    results = []
    total = len(L_VALUES) * len(T_STAGGER_VALUES)
    idx = 0

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["L", "t_stagger", "reward", "R_torque", "R_muscle", "t0", "t1", "t2", "t3"])

        for L in L_VALUES:
            for t_s in T_STAGGER_VALUES:
                idx += 1
                t0, t1, t2, t3 = 0.0, t_s, 2 * t_s, 3 * t_s
                theta = np.array([L, t0, t1, t2, t3], dtype=np.float32)

                env.reset()
                _, reward, _, _, info = env.step(theta)

                R_torque = info.get("R_torque", 0.0)
                R_muscle = info.get("R_muscle", 0.0)

                row = [L, t_s, round(reward, 6), round(R_torque, 6), round(R_muscle, 6), t0, t1, t2, t3]
                writer.writerow(row)
                f.flush()
                results.append(row)

                print(
                    f"[{idx:2d}/{total}]  L={L:.2f}  t_s={t_s:.2f}  "
                    f"→ reward={reward:.4f}  "
                    f"(R_torque={R_torque:.4f}  R_muscle={R_muscle:.4f})"
                )

    print(f"\nResults saved to {OUT_PATH}")

    # Print summary tables for each component.
    for label, col in [("reward", 2), ("R_torque", 3), ("R_muscle", 4)]:
        print(f"\n{label} landscape (rows=L, cols=t_stagger):")
        header = "     L\\t_s |" + "".join(f" {t_s:5.2f}" for t_s in T_STAGGER_VALUES)
        print(header)
        print("-" * len(header))
        for i, L in enumerate(L_VALUES):
            row_vals = [results[i * len(T_STAGGER_VALUES) + j][col] for j in range(len(T_STAGGER_VALUES))]
            print(f"  L={L:.2f}  |" + "".join(f" {v:+.3f}" for v in row_vals))


if __name__ == "__main__":
    main()
