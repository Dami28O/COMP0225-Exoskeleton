"""
PPO training for minimal-control TSA hardware parameter optimisation.

Optimises theta = [L, t0, t1, t2, t3] using the reward components from
ppo_wrapper_min_control.py:
    reference tracking + VAS effort reduction + TSA effort regularisation + stability.

Run:
    mjpython train_ppo_min_control.py
    mjpython train_ppo_min_control.py --timesteps 2000 --out logs/ppo_min_control_quick
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_CTRL_DIR = _ROOT_DIR / "ctrl_optim"
_TSA_DIR = _ROOT_DIR / "tsa_modelling"
_MYO_DIR = _ROOT_DIR / "myoassist"

for _p in [str(_CTRL_DIR), str(_TSA_DIR), str(_MYO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ppo_config import TSAOptimConfig
from ppo_wrapper_min_control import TSAOptimMinControlEnv
import numpy as np




               


class MinControlCallback(BaseCallback):
    """Logs reward components, sampled actions, and tracks the best sampled theta."""

    def __init__(self, csv_path: Path, best_json_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.best_json_path = best_json_path
        self._csv_file = None
        self._csv_writer = None
        self._episode = 0
        self.best_reward = -float("inf")
        self.best_info = None

    def _json_safe(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)

        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]

        return obj

    def _on_training_start(self) -> None:
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        self._csv_writer.writerow([
            "episode",
            "reward",
            "success",
            "steps_used",
            "final_phase",
            "t_final",
            "R_success",
            "R_track",
            "R_track_pos",
            "R_track_vel",
            "R_muscle",
            "R_tsa",
            "R_stability",
            "R_time",
            "L",
            "t0",
            "t1",
            "t2",
            "t3",
        ])
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            self._episode += 1

            reward = float(info.get("reward", float("nan")))

            # ------------------------------------------------------------
            # Always log this episode
            # ------------------------------------------------------------
            row = [
                self._episode,
                reward,
                info.get("success", float("nan")),
                info.get("steps_used", float("nan")),
                info.get("final_phase", float("nan")),
                info.get("t_final", float("nan")),
                info.get("R_success", float("nan")),
                info.get("R_track", float("nan")),
                info.get("R_track_pos", float("nan")),
                info.get("R_track_vel", float("nan")),
                info.get("R_muscle", float("nan")),
                info.get("R_tsa", float("nan")),
                info.get("R_stability", float("nan")),
                info.get("R_time", float("nan")),
                info.get("L", float("nan")),
                info.get("t0", float("nan")),
                info.get("t1", float("nan")),
                info.get("t2", float("nan")),
                info.get("t3", float("nan")),
            ]

            if self._csv_writer is not None:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            # ------------------------------------------------------------
            # Print occasionally for sanity
            # ------------------------------------------------------------
            if self._episode <= 5 or self._episode % 100 == 0:
                print(
                    f"[EP {self._episode}] "
                    f"reward={reward:.4f} "
                    f"success={info.get('success', float('nan'))} "
                    f"steps={info.get('steps_used', float('nan'))} "
                    f"phase={info.get('final_phase', float('nan'))} "
                    f"t_final={info.get('t_final', float('nan'))} "
                    f"L={info.get('L', float('nan')):.4f} "
                    f"t=[{info.get('t0', float('nan')):.3f}, "
                    f"{info.get('t1', float('nan')):.3f}, "
                    f"{info.get('t2', float('nan')):.3f}, "
                    f"{info.get('t3', float('nan')):.3f}]"
                )

            # ------------------------------------------------------------
            # Save best sampled theta
            # ------------------------------------------------------------
            if np.isfinite(reward) and reward > self.best_reward:
                self.best_reward = reward
                self.best_info = dict(info)

                safe_payload = self._json_safe({
                    "episode": self._episode,
                    "reward": reward,
                    "info": self.best_info,
                })

                with open(self.best_json_path, "w") as f:
                    json.dump(safe_payload, f, indent=2)

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


def _load_config(path: str | None) -> TSAOptimConfig:
    if path is None:
        return TSAOptimConfig()
    with open(path) as f:
        d = json.load(f)
    cfg = TSAOptimConfig()
    for key, val in d.items():
        if key in ("env_params", "reward_params", "ppo_params"):
            sub = getattr(cfg, key)
            for sub_key, sub_val in val.items():
                if hasattr(sub, sub_key):
                    setattr(sub, sub_key, sub_val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal-control PPO TSA parameter optimiser")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
        
        
    print("\n[CONFIG CHECK]")
    print(f"  requested timesteps = {args.timesteps}")
    print(f"  config.total_timesteps = {config.total_timesteps}")
    print(f"  env max_steps = {config.env_params.max_steps}")
    print(f"  ppo n_steps = {config.ppo_params.n_steps}")
    print(f"  ppo batch_size = {config.ppo_params.batch_size}")


    import time as _time
    out_tag = args.out or f"logs/ppo_min_control_{int(_time.time())}"
    out_dir = _ROOT_DIR / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard"
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    mj_env = gym.make("TorsoLegs")
    env = TSAOptimMinControlEnv(mj_env, config)
    print(env.mj_env.sim.model.opt.timestep)

    try:
        import tensorboard  # noqa: F401
        tb_log = str(tb_dir)
    except ImportError:
        tb_log = None
        print("[warn] tensorboard not installed — TensorBoard logging disabled; CSV log still active")

    p = config.ppo_params
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=p.learning_rate,
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        ent_coef=p.ent_coef,
        clip_range=p.clip_range,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        verbose=1,
        device=p.device,
        tensorboard_log=tb_log,
    )

    checkpoint_cb = CheckpointCallback(save_freq=200, save_path=str(ck_dir), name_prefix="ppo_min_control")
    reward_csv = out_dir / "min_control_components.csv"
    best_json = out_dir / "best_sampled_theta.json"
    reward_cb = MinControlCallback(csv_path=reward_csv, best_json_path=best_json)

    print(f"\nStarting minimal-control PPO training — {config.total_timesteps} episodes")
    print(f"Action space: {env.action_space}")
    print(f"Output dir:   {out_dir}\n")

    model.learn(total_timesteps=config.total_timesteps, callback=[checkpoint_cb, reward_cb], progress_bar=True)

    final_path = out_dir / "ppo_tsa_min_control_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}.zip")
    print(f"Best sampled theta saved to {best_json}")

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)
    print("\nFinal deterministic policy theta:")
    print(f"  L      = {info.get('L', float('nan')):.4f} m")
    for i in range(4):
        print(f"  t{i}     = {info.get(f't{i}', float('nan')):.4f} s")
    print(f"  Reward = {reward:.4f}")
    print(f"  Success = {bool(info.get('success', 0.0))}")
    print(f"  R_track_pos = {info.get('R_track_pos', float('nan')):.4f}")
    print(f"  R_track_vel = {info.get('R_track_vel', float('nan')):.4f}")
    print(f"  R_muscle    = {info.get('R_muscle', float('nan')):.4f}")
    print(f"  R_tsa       = {info.get('R_tsa', float('nan')):.4f}")
    print(f"  R_stability = {info.get('R_stability', float('nan')):.4f}")
    print(f"  t_final     = {info.get('t_final', float('nan')):.3f} s")


if __name__ == "__main__":
    main()
