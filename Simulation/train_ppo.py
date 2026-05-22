"""
PPO training for TSA hardware parameter optimisation.

Optimises θ = [L, t0, t1, t2, t3] (symmetric 5-D) to maximise:
    R = w_torque · R_torque  +  w_muscle · R_muscle

Run:
    mjpython train_ppo.py
    mjpython train_ppo.py --config ctrl_optim/configs/ppo_experiment1.json
    mjpython train_ppo.py --timesteps 2000 --out logs/ppo_quick

Checkpoints are saved to logs/ppo_checkpoints/ every 200 episodes.
Final model is saved to <out_dir>/ppo_tsa_final.zip.
TensorBoard logs go to <out_dir>/tensorboard/.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_ROOT_DIR = Path(__file__).resolve().parent
_CTRL_DIR = _ROOT_DIR / "ctrl_optim"
_TSA_DIR  = _ROOT_DIR / "tsa_modelling"
_MYO_DIR  = _ROOT_DIR / "myoassist"

for _p in [str(_CTRL_DIR), str(_TSA_DIR), str(_MYO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ppo_config import TSAOptimConfig
from ppo_wrapper import TSAOptimEnv


# ---------------------------------------------------------------------------
# Reward logging callback
# ---------------------------------------------------------------------------

class RewardComponentCallback(BaseCallback):
    """
    Logs per-episode reward components to TensorBoard and a CSV.

    Since each env step is one complete STS episode (terminated=True always),
    every on_step fires at episode end, so we capture info unconditionally.
    """

    def __init__(self, csv_path: Path, episode_offset: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path       = csv_path
        self._csv_file      = None
        self._csv_writer    = None
        self._episode       = episode_offset   # non-zero when resuming

    def _on_training_start(self) -> None:
        append = self.csv_path.exists() and self._episode > 0
        self._csv_file   = open(self.csv_path, "a" if append else "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not append:
            self._csv_writer.writerow(
                ["episode", "reward", "R_torque", "R_muscle", "R_time", "R_completion", "t_final"]
            )

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self._episode += 1
            r      = info.get("reward",       float("nan"))
            r_torq = info.get("R_torque",     float("nan"))
            r_musc = info.get("R_muscle",     float("nan"))
            r_time = info.get("R_time",       float("nan"))
            r_comp = info.get("R_completion", float("nan"))
            t_fin  = info.get("t_final",      float("nan"))

            self.logger.record("reward/total",        r)
            self.logger.record("reward/R_torque",     r_torq)
            self.logger.record("reward/R_muscle",     r_musc)
            self.logger.record("reward/R_time",       r_time)
            self.logger.record("reward/R_completion", r_comp)
            self.logger.record("reward/t_final",      t_fin)

            if self._csv_writer is not None:
                self._csv_writer.writerow(
                    [self._episode, r, r_torq, r_musc, r_time, r_comp, t_fin]
                )
                self._csv_file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None


# ---------------------------------------------------------------------------
# Reward component plot
# ---------------------------------------------------------------------------

def _plot_reward_components(csv_path: Path, out_dir: Path) -> None:
    """Read the per-episode CSV and save a reward-component breakdown figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping reward plot")
        return

    episodes, rewards, r_torques, r_muscles, r_times, t_finals = [], [], [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            r_torques.append(float(row["R_torque"]))
            r_muscles.append(float(row["R_muscle"]))
            r_times.append(float(row["R_time"]))
            t_finals.append(float(row["t_final"]))

    if not episodes:
        return

    def _smooth(vals, w=20):
        if len(vals) < w:
            return list(vals)
        return list(np.convolve(vals, np.ones(w) / w, mode="valid"))

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("PPO Training — Reward Components per Episode", fontsize=13)

    pairs = [
        (rewards,   "Total reward",  "black"),
        (r_torques, "R_torque",      "tab:blue"),
        (r_muscles, "R_muscle",      "tab:orange"),
        (r_times,   "R_time",        "tab:green"),
        (t_finals,  "t_final (s)",   "tab:red"),
    ]
    for ax, (vals, label, color) in zip(axes, pairs):
        ax.plot(episodes, vals, alpha=0.25, color=color, linewidth=0.7)
        smoothed = _smooth(vals)
        ax.plot(episodes[len(episodes) - len(smoothed):], smoothed,
                color=color, linewidth=1.8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode")
    plt.tight_layout()
    plot_path = out_dir / "reward_components.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Reward component plot saved → {plot_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | None) -> TSAOptimConfig:
    if path is None:
        return TSAOptimConfig()
    with open(path) as f:
        d = json.load(f)
    cfg = TSAOptimConfig()
    # Shallow merge of top-level keys and sub-dicts.
    for key, val in d.items():
        if key in ("env_params", "reward_params", "ppo_params"):
            sub = getattr(cfg, key)
            for sub_key, sub_val in val.items():
                if hasattr(sub, sub_key):
                    setattr(sub, sub_key, sub_val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO TSA parameter optimiser")
    parser.add_argument("--config",     default=None,           help="Path to JSON config")
    parser.add_argument("--timesteps",  type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--out",        default=None,           help="Output directory (default: logs/ppo_<timestamp>)")
    parser.add_argument("--resume",     default=None,           help="Path to a checkpoint .zip to resume from")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps

    # Output directories.
    import time as _time
    out_tag = args.out or f"logs/ppo_{int(_time.time())}"
    out_dir = _ROOT_DIR / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir  = out_dir / "tensorboard"
    ck_dir  = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)

    # Save the config used for this run.
    with open(out_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    # Build environment.
    mj_env = gym.make('TorsoLegs')
    env    = TSAOptimEnv(mj_env, config)

    try:
        import tensorboard 
        _tb_log = str(tb_dir)
    except ImportError:
        _tb_log = None
        print("[warn] tensorboard not installed — TensorBoard logging disabled; CSV log still active")

    p = config.ppo_params

    # Resume from checkpoint or start fresh.
    steps_done = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        # Parse completed steps from filename: ppo_tsa_1000_steps.zip → 1000
        try:
            steps_done = int(ckpt_path.stem.rsplit("_steps", 1)[0].rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            steps_done = 0
        model = PPO.load(str(ckpt_path), env=env, device=p.device,
                         tensorboard_log=_tb_log)
        print(f"Resumed from {ckpt_path.name}  ({steps_done} steps done)")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate   = p.learning_rate,
            n_steps         = p.n_steps,
            batch_size      = p.batch_size,
            n_epochs        = p.n_epochs,
            gamma           = p.gamma,
            gae_lambda      = p.gae_lambda,
            ent_coef        = p.ent_coef,
            clip_range      = p.clip_range,
            vf_coef         = p.vf_coef,
            max_grad_norm   = p.max_grad_norm,
            verbose         = 1,
            device          = p.device,
            tensorboard_log = _tb_log,
        )

    remaining = max(config.total_timesteps - steps_done, 0)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 200,
        save_path   = str(ck_dir),
        name_prefix = "ppo_tsa",
    )
    reward_csv = out_dir / "reward_components.csv"
    reward_cb  = RewardComponentCallback(csv_path=reward_csv, episode_offset=steps_done)

    print(f"\nStarting PPO training — {remaining} episodes remaining (total {config.total_timesteps})")
    print(f"Action space: {env.action_space}")
    print(f"Output dir:   {out_dir}\n")

    model.learn(
        total_timesteps = remaining,
        callback        = [checkpoint_cb, reward_cb],
        progress_bar    = True,
    )

    _plot_reward_components(reward_csv, out_dir)

    final_path = out_dir / "ppo_tsa_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}.zip")

    # Quick evaluation: run the greedy policy once and report reward.
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)
    L, *t_vals = action.tolist()
    print(f"\nBest θ found:")
    print(f"  L        = {L:.4f} m")
    for i, t in enumerate(t_vals):
        print(f"  t{i}       = {t:.4f} s")
    print(f"  Reward   = {reward:.4f}")
    print(f"  R_torque = {info.get('R_torque', float('nan')):.4f}")
    print(f"  R_muscle = {info.get('R_muscle', float('nan')):.4f}")
    print(f"  R_time   = {info.get('R_time',   float('nan')):.4f}")
    print(f"  t_final  = {info.get('t_final',  float('nan')):.3f} s")


if __name__ == "__main__":
    main()
