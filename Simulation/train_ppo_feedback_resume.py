from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
_ROOT_DIR = Path(__file__).resolve().parent
_CTRL_DIR = _ROOT_DIR / "ctrl_optim"
_TSA_DIR = _ROOT_DIR / "tsa_modelling"
_MYO_DIR  = _ROOT_DIR / "myoassist"

for p in [str(_ROOT_DIR), str(_CTRL_DIR), str(_TSA_DIR),str(_MYO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from ppo_feedback_wrapper import FeedbackAssistConfig, STSFeedbackAssistEnv


# ---------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------

class FeedbackEpisodeCallback(BaseCallback):
    """
    Logs completed feedback-PPO episodes and saves the current model immediately
    when a better Phase-4-reaching episode is observed.

    This is important because checkpoints are saved by timestep, not by episode,
    so the usual checkpoint may not correspond to the exact policy that produced
    a good Phase 4 episode.
    """

    def __init__(self, out_dir: Path, target_phase: int = 4, verbose: int = 0):
        super().__init__(verbose)

        self.out_dir = Path(out_dir)
        self.target_phase = int(target_phase)

        self.csv_path = self.out_dir / "feedback_training_episodes.csv"
        self.best_json_path = self.out_dir / "best_phase4_episode.json"
        self.best_model_path = self.out_dir / "best_phase4_model.zip"
        self.best_trace_path = self.out_dir / "best_phase4_action_trace.csv"

        self._csv_file = None
        self._csv_writer = None

        self._episode = 0
        self.best_score_tuple = None
        self.best_info = None

        # For action trace of the current episode.
        self._current_trace: list[dict[str, Any]] = []

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
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        self._csv_writer.writerow([
            "episode",
            "episode_reward",
            "success",
            "failed",
            "phase",
            "steps_used",
            "t_final",
            "mean_muscle",
            "mean_abs_assist_tau",
            "rms_assist_tau",
            "max_assist_tau",
            "contact_bad_frac",

            "r_progress",
            "r_posture",
            "r_velocity",
            "r_muscle",
            "r_assist",
            "r_contact",
            "r_symmetry",

            "mean_leg_activation",
            "mean_vas_activation",
            "pelvis_y",
            "torso_y",
            "root_pitch",
            "lean",
            "knee_avg",
            "hip_avg",
            "ankle_avg",
            "pelvis_to_feet_x",
            "grounded",
            "seat_contact",

            "assist_r_cmd",
            "assist_l_cmd",
            "assist_tau_r",
            "assist_tau_l",
        ])
        self._csv_file.flush()

    def _record_trace_step(self) -> None:
        """
        Record one action/info row during training.

        This gives an approximate action trace for the episode. It is especially
        useful if a good stochastic rollout is found and the deterministic policy
        later does not exactly reproduce it.
        """
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)

        if not infos:
            return

        info = infos[0]

        if actions is None:
            action_r = float(info.get("assist_r_cmd", np.nan))
            action_l = float(info.get("assist_l_cmd", np.nan))
        else:
            arr = np.asarray(actions)
            # Stable-Baselines3 usually stores actions as shape (n_envs, action_dim).
            if arr.ndim == 2:
                action_r = float(arr[0, 0])
                action_l = float(arr[0, 1])
            elif arr.ndim == 1 and len(arr) >= 2:
                action_r = float(arr[0])
                action_l = float(arr[1])
            else:
                action_r = float(info.get("assist_r_cmd", np.nan))
                action_l = float(info.get("assist_l_cmd", np.nan))

        self._current_trace.append({
            "step_in_episode": len(self._current_trace),
            "time": float(info.get("t_final", np.nan)),
            "phase": float(info.get("phase", np.nan)),
            "reward_so_far": float(info.get("episode_reward", np.nan)),
            "action_r": action_r,
            "action_l": action_l,
            "assist_r_cmd": float(info.get("assist_r_cmd", np.nan)),
            "assist_l_cmd": float(info.get("assist_l_cmd", np.nan)),
            "assist_tau_r": float(info.get("assist_tau_r", np.nan)),
            "assist_tau_l": float(info.get("assist_tau_l", np.nan)),
            "pelvis_y": float(info.get("pelvis_y", np.nan)),
            "torso_y": float(info.get("torso_y", np.nan)),
            "root_pitch": float(info.get("root_pitch", np.nan)),
            "lean": float(info.get("lean", np.nan)),
            "knee_avg": float(info.get("knee_avg", np.nan)),
            "hip_avg": float(info.get("hip_avg", np.nan)),
            "ankle_avg": float(info.get("ankle_avg", np.nan)),
            "mean_vas_activation": float(info.get("mean_vas_activation", np.nan)),
            "mean_leg_activation": float(info.get("mean_leg_activation", np.nan)),
        })

    def _save_trace_csv(self, path: Path) -> None:
        if not self._current_trace:
            return

        keys = list(self._current_trace[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self._current_trace:
                writer.writerow(row)

    def _on_step(self) -> bool:
        self._record_trace_step()

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            self._episode += 1

            episode_reward = float(info.get("episode_reward", info.get("reward", -1e9)))
            phase = int(info.get("phase", info.get("final_phase", 1)))
            failed = int(info.get("failed", 0))

            # Treat Phase 4 as success if target_phase = 4, even if older wrapper info is wrong.
            success = int(info.get("success", 0))
            phase_success = int(phase >= self.target_phase)
            success_for_ranking = int(success or phase_success)

            row = [
                self._episode,
                episode_reward,
                success_for_ranking,
                failed,
                phase,
                info.get("steps_used", np.nan),
                info.get("t_final", np.nan),
                info.get("mean_muscle", np.nan),
                info.get("mean_abs_assist_tau", np.nan),
                info.get("rms_assist_tau", np.nan),
                info.get("max_assist_tau", np.nan),
                info.get("contact_bad_frac", np.nan),

                info.get("r_progress", np.nan),
                info.get("r_posture", np.nan),
                info.get("r_velocity", np.nan),
                info.get("r_muscle", np.nan),
                info.get("r_assist", np.nan),
                info.get("r_contact", np.nan),
                info.get("r_symmetry", np.nan),

                info.get("mean_leg_activation", np.nan),
                info.get("mean_vas_activation", np.nan),
                info.get("pelvis_y", np.nan),
                info.get("torso_y", np.nan),
                info.get("root_pitch", np.nan),
                info.get("lean", np.nan),
                info.get("knee_avg", np.nan),
                info.get("hip_avg", np.nan),
                info.get("ankle_avg", np.nan),
                info.get("pelvis_to_feet_x", np.nan),
                info.get("grounded", np.nan),
                info.get("seat_contact", np.nan),

                info.get("assist_r_cmd", np.nan),
                info.get("assist_l_cmd", np.nan),
                info.get("assist_tau_r", np.nan),
                info.get("assist_tau_l", np.nan),
            ]

            if self._csv_writer is not None:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            # Best ranking:
            # 1. successful Phase 4+ episodes first
            # 2. higher final phase next
            # 3. non-failed preferred
            # 4. higher episode reward
            score_tuple = (
                success_for_ranking,
                phase,
                int(not failed),
                episode_reward,
            )

            if self.best_score_tuple is None or score_tuple > self.best_score_tuple:
                self.best_score_tuple = score_tuple
                self.best_info = dict(info)

                # Save exact current model immediately.
                self.model.save(str(self.best_model_path))

                # Save action trace for this best episode.
                self._save_trace_csv(self.best_trace_path)

                payload = {
                    "episode": self._episode,
                    "score_tuple": score_tuple,
                    "target_phase": self.target_phase,
                    "success_for_ranking": success_for_ranking,
                    "phase": phase,
                    "failed": failed,
                    "episode_reward": episode_reward,
                    "info": dict(info),
                    "model_path": str(self.best_model_path),
                    "trace_path": str(self.best_trace_path),
                }

                with open(self.best_json_path, "w") as f:
                    json.dump(self._json_safe(payload), f, indent=2)

                print(
                    f"\n[BEST PHASE-{self.target_phase} MODEL SAVED] "
                    f"episode={self._episode} "
                    f"phase={phase} "
                    f"success={success_for_ranking} "
                    f"reward={episode_reward:.3f} "
                    f"path={self.best_model_path}\n"
                )

            if self._episode <= 5 or self._episode % 10 == 0:
                print(
                    f"[EP {self._episode}] "
                    f"reward={episode_reward:.2f} "
                    f"phase={phase} "
                    f"success={success_for_ranking} "
                    f"failed={failed} "
                    f"steps={info.get('steps_used', np.nan)} "
                    f"max_tau={info.get('max_assist_tau', np.nan):.2f} "
                    f"mean_vas={info.get('mean_vas_activation', np.nan):.4f}"
                )

            # New episode starts after reset, so clear trace.
            self._current_trace = []

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Resume/train feedback PPO for STS assistance.")

    parser.add_argument("--resume", default=None, help="Path to PPO checkpoint/model .zip to resume from.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Additional timesteps to train.")
    parser.add_argument("--out", default=None, help="Output directory.")
    parser.add_argument("--reset-timesteps", action="store_true", help="Reset SB3 timestep counter when resuming.")

    parser.add_argument("--target-phase", type=int, default=4, help="Phase counted as success.")
    parser.add_argument("--max-steps", type=int, default=3500)
    parser.add_argument("--leg-scale", type=float, default=0.90)
    parser.add_argument("--tau-max", type=float, default=60.0)
    parser.add_argument("--assist-sign", type=float, default=1.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--checkpoint-freq", type=int, default=25000)

    args = parser.parse_args()

    import time as _time
    out_dir = Path(args.out or f"logs/ppo_feedback_phase{args.target_phase}_{int(_time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = FeedbackAssistConfig()
    cfg.max_steps = int(args.max_steps)
    cfg.leg_scale = float(args.leg_scale)
    cfg.tau_max = float(args.tau_max)
    cfg.assist_sign = float(args.assist_sign)

    # Works with the amended wrapper. If your class has no declared field,
    # this still adds it dynamically and the callback still ranks Phase 4 as success.
    cfg.target_phase = int(args.target_phase)

    # Save run config.
    run_config = vars(args).copy()
    run_config["out_dir"] = str(out_dir)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    mj_env = gym.make("TorsoLegs")
    env = STSFeedbackAssistEnv(
        mj_env,
        fb_cfg=cfg,
        debug=False,
        log_to_csv=False,
        log_tag=None,
    )

    tensorboard_dir = out_dir / "tensorboard"

    if args.resume is not None:
        print(f"\nLoading PPO model from checkpoint:\n  {args.resume}\n")
        model = PPO.load(args.resume, env=env, device=args.device)
        model.tensorboard_log = str(tensorboard_dir)
    else:
        print("\nCreating new PPO model.\n")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=int(args.checkpoint_freq),
        save_path=str(ckpt_dir),
        name_prefix="ppo_feedback",
    )

    episode_cb = FeedbackEpisodeCallback(
        out_dir=out_dir,
        target_phase=int(args.target_phase),
    )

    print("\n[TRAINING CONFIG]")
    print(f"out_dir       = {out_dir}")
    print(f"resume        = {args.resume}")
    print(f"timesteps     = {args.timesteps}")
    print(f"target_phase  = {args.target_phase}")
    print(f"max_steps     = {args.max_steps}")
    print(f"leg_scale     = {args.leg_scale}")
    print(f"tau_max       = {args.tau_max} Nm")
    print(f"assist_sign   = {args.assist_sign}")
    print(f"n_steps       = {args.n_steps}")
    print(f"batch_size    = {args.batch_size}")
    print()

    model.learn(
        total_timesteps=int(args.timesteps),
        callback=[checkpoint_cb, episode_cb],
        progress_bar=True,
        reset_num_timesteps=bool(args.reset_timesteps),
    )

    final_path = out_dir / "ppo_feedback_final"
    model.save(str(final_path))

    print(f"\nFinal model saved to:")
    print(f"  {final_path}.zip")

    print(f"\nBest Phase-{args.target_phase} model saved to:")
    print(f"  {out_dir / 'best_phase4_model.zip'}")

    print(f"\nBest episode metadata:")
    print(f"  {out_dir / 'best_phase4_episode.json'}")

    print(f"\nBest episode action trace:")
    print(f"  {out_dir / 'best_phase4_action_trace.csv'}")


if __name__ == "__main__":
    main()