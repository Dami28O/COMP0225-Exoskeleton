"""
Train feedback PPO for idealised sit-to-stand knee assistance.

Unlike train_ppo.py / train_ppo_min_control.py, this is a multi-step RL setup:
PPO observes skeleton state at every timestep and outputs right/left knee-assist
commands. The biological STS controller is run at 70% leg-control capacity by
default, and PPO learns assistive knee torque on top of it.

Run examples:
    mjpython train_ppo_feedback.py --timesteps 100000 --out logs/ppo_feedback_100k
    mjpython train_ppo_feedback.py --timesteps 20000 --tau-max 40 --out logs/ppo_feedback_test

If assistance makes things worse immediately, try flipping the torque sign:
    mjpython train_ppo_feedback.py --assist-sign -1 --timesteps 20000 --out logs/ppo_feedback_signflip
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent
_CTRL_DIR = _ROOT_DIR / "ctrl_optim"
_TSA_DIR = _ROOT_DIR / "tsa_modelling"
_MYO_DIR = _ROOT_DIR / "myoassist"

for _p in [str(_CTRL_DIR), str(_TSA_DIR), str(_MYO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import myosuite.envs.myo.myobase  # noqa: F401,E402
from myosuite.utils import gym  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from ppo_feedback_wrapper import FeedbackAssistConfig, STSFeedbackAssistEnv  # noqa: E402


class FeedbackTrainingCallback(BaseCallback):
    """Logs terminal episode summaries and saves the best successful checkpoint metadata."""

    def __init__(self, csv_path: Path, best_json_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.best_json_path = best_json_path
        self._csv_file = None
        self._csv_writer = None
        self._episode = 0
        self.best_ep_reward = -float("inf")
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
            "episode_reward",
            "success",
            "failed",
            "steps_used",
            "t_final",
            "phase",
            "mean_muscle",
            "mean_leg_activation",
            "mean_vas_activation",
            "mean_abs_assist_tau",
            "rms_assist_tau",
            "max_assist_tau",
            "contact_bad_frac",
            "pelvis_y",
            "torso_y",
            "root_pitch",
            "lean",
            "knee_avg",
            "hip_avg",
            "ankle_avg",
            "pelvis_to_feet_x",
        ])
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            self._episode += 1
            ep_reward = float(info.get("episode_reward", float("nan")))

            row = [
                self._episode,
                ep_reward,
                info.get("success", float("nan")),
                info.get("failed", float("nan")),
                info.get("steps_used", float("nan")),
                info.get("t_final", float("nan")),
                info.get("phase", float("nan")),
                info.get("mean_muscle", float("nan")),
                info.get("mean_leg_activation", float("nan")),
                info.get("mean_vas_activation", float("nan")),
                info.get("mean_abs_assist_tau", float("nan")),
                info.get("rms_assist_tau", float("nan")),
                info.get("max_assist_tau", float("nan")),
                info.get("contact_bad_frac", float("nan")),
                info.get("pelvis_y", float("nan")),
                info.get("torso_y", float("nan")),
                info.get("root_pitch", float("nan")),
                info.get("lean", float("nan")),
                info.get("knee_avg", float("nan")),
                info.get("hip_avg", float("nan")),
                info.get("ankle_avg", float("nan")),
                info.get("pelvis_to_feet_x", float("nan")),
            ]

            if self._csv_writer is not None:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            self.logger.record("feedback/episode_reward", ep_reward)
            self.logger.record("feedback/success", float(info.get("success", 0.0)))
            self.logger.record("feedback/steps_used", float(info.get("steps_used", 0.0)))
            self.logger.record("feedback/mean_muscle", float(info.get("mean_muscle", 0.0)))
            self.logger.record("feedback/mean_abs_assist_tau", float(info.get("mean_abs_assist_tau", 0.0)))

            if self._episode <= 5 or self._episode % 10 == 0:
                print(
                    f"[EP {self._episode}] "
                    f"R={ep_reward:.3f} "
                    f"success={info.get('success', 0)} "
                    f"steps={info.get('steps_used', np.nan):.0f} "
                    f"phase={info.get('phase', np.nan):.0f} "
                    f"meanVAS={info.get('mean_vas_activation', np.nan):.3f} "
                    f"meanTau={info.get('mean_abs_assist_tau', np.nan):.2f} "
                    f"maxTau={info.get('max_assist_tau', np.nan):.2f}"
                )

            # Prefer successful episodes. If no success yet, still track highest reward.
            is_candidate = np.isfinite(ep_reward) and (
                ep_reward > self.best_ep_reward
                or (float(info.get("success", 0.0)) > 0.5 and self.best_info is not None and float(self.best_info.get("success", 0.0)) < 0.5)
            )
            if is_candidate:
                self.best_ep_reward = ep_reward
                self.best_info = dict(info)
                payload = self._json_safe({
                    "episode": self._episode,
                    "episode_reward": ep_reward,
                    "info": self.best_info,
                })
                with open(self.best_json_path, "w") as f:
                    json.dump(payload, f, indent=2)

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


def make_env(args) -> STSFeedbackAssistEnv:
    mj_env = gym.make("TorsoLegs")
    fb_cfg = FeedbackAssistConfig(
        max_steps=args.max_steps,
        leg_scale=args.leg_scale,
        tau_max=args.tau_max,
        assist_sign=args.assist_sign,
        target_duration=args.target_duration,
    )
    env = STSFeedbackAssistEnv(
        mj_env,
        fb_cfg=fb_cfg,
        debug=args.debug_env,
        log_to_csv=args.log_rollout_csv,
        log_tag=args.log_tag,
    )
    return env


def evaluate_policy_once(model: PPO, env: STSFeedbackAssistEnv) -> dict:
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    last_info = {}

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        last_info = dict(info)

    last_info["eval_total_reward"] = float(total_reward)
    return last_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback PPO knee-assist controller for reduced-control STS")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-steps", type=int, default=3500)
    parser.add_argument("--leg-scale", type=float, default=0.90)
    parser.add_argument("--tau-max", type=float, default=60.0)
    parser.add_argument("--assist-sign", type=float, default=1.0)
    parser.add_argument("--target-duration", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--debug-env", action="store_true")
    parser.add_argument("--log-rollout-csv", action="store_true", help="Enable sts_ctrl diagnostic CSV logging during training; slow.")
    parser.add_argument("--log-tag", default="ppo_feedback_train")
    args = parser.parse_args()

    out_tag = args.out or f"logs/ppo_feedback_{int(time.time())}"
    out_dir = _ROOT_DIR / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard"

    with open(out_dir / "feedback_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    env = make_env(args)
    env = Monitor(env)

    try:
        import tensorboard  # noqa: F401
        tb_log = str(tb_dir)
    except ImportError:
        tb_log = None
        print("[warn] tensorboard not installed — TensorBoard logging disabled; CSV log still active")

    print("\n[FEEDBACK PPO TRAINING]")
    print(f"  timesteps       = {args.timesteps}")
    print(f"  max_steps/ep    = {args.max_steps}")
    print(f"  leg_scale       = {args.leg_scale}")
    print(f"  tau_max         = {args.tau_max} Nm")
    print(f"  assist_sign     = {args.assist_sign}")
    print(f"  action_space    = {env.action_space}")
    print(f"  observation     = {env.observation_space}")
    print(f"  out_dir         = {out_dir}\n")

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
        verbose=1,
        device=args.device,
        tensorboard_log=tb_log,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.n_steps * 5, 1000),
        save_path=str(ck_dir),
        name_prefix="ppo_feedback",
    )
    train_csv = out_dir / "feedback_training_episodes.csv"
    best_json = out_dir / "best_feedback_episode.json"
    feedback_cb = FeedbackTrainingCallback(csv_path=train_csv, best_json_path=best_json)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, feedback_cb],
        progress_bar=True,
    )

    final_path = out_dir / "ppo_feedback_final"
    model.save(str(final_path))
    print(f"\nFinal feedback PPO model saved to {final_path}.zip")
    print(f"Episode log saved to {train_csv}")
    print(f"Best episode metadata saved to {best_json}")

    print("\n[DETERMINISTIC EVALUATION]")
    eval_env = make_env(args)
    info = evaluate_policy_once(model, eval_env)
    eval_env.close()

    for key in [
        "eval_total_reward", "success", "failed", "steps_used", "t_final", "phase",
        "mean_muscle", "mean_vas_activation", "mean_abs_assist_tau", "max_assist_tau",
        "pelvis_y", "torso_y", "root_pitch", "lean", "knee_avg", "hip_avg", "ankle_avg",
    ]:
        val = info.get(key, float("nan"))
        if isinstance(val, (int, float, np.floating)):
            print(f"  {key:22s} = {float(val):.5f}")
        else:
            print(f"  {key:22s} = {val}")

    with open(out_dir / "deterministic_eval.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in info.items()}, f, indent=2)


if __name__ == "__main__":
    main()
