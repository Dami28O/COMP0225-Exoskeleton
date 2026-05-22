from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CTRL_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CTRL_DIR.parent
_TSA_DIR = _ROOT_DIR / "tsa_modelling"
_MYO_DIR = _ROOT_DIR / "myoassist"

for _p in [str(_CTRL_DIR), str(_TSA_DIR), str(_MYO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym

from ppo_config import TSAOptimConfig
from ppo_wrapper_min_control import TSAOptimMinControlEnv


def load_best_action(best_json_path: Path) -> np.ndarray:
    with open(best_json_path) as f:
        d = json.load(f)

    info = d["info"]

    action = np.array(
        [
            float(info["L"]),
            float(info["t0"]),
            float(info["t1"]),
            float(info["t2"]),
            float(info["t3"]),
        ],
        dtype=np.float32,
    )

    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="Path to PPO run directory, e.g. logs/ppo_min_control_30k",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    best_json_path = run_dir / "best_sampled_theta.json"

    if not best_json_path.exists():
        raise FileNotFoundError(f"Could not find {best_json_path}")

    action = load_best_action(best_json_path)

    print("\n[EVAL BEST SAMPLED THETA]")
    print(f"Loaded from: {best_json_path}")
    print(f"L  = {action[0]:.5f}")
    print(f"t0 = {action[1]:.5f}")
    print(f"t1 = {action[2]:.5f}")
    print(f"t2 = {action[3]:.5f}")
    print(f"t3 = {action[4]:.5f}")

    config = TSAOptimConfig()

    # Match your training setup if you changed this.
    config.env_params.max_steps = 3500

    mj_env = gym.make("TorsoLegs")
    env = TSAOptimMinControlEnv(mj_env, config)


    obs, _ = env.reset()
    _, reward, terminated, truncated, info = env.step(action)
    
    

    print("\n[RESULT]")
    print(f"reward       = {reward:.5f}")
    print(f"success      = {info.get('success')}")
    print(f"final_phase  = {info.get('final_phase')}")
    print(f"steps_used   = {info.get('steps_used')}")
    print(f"t_final      = {info.get('t_final'):.5f}")

    print("\n[REWARD COMPONENTS]")
    for k in [
        "R_success",
        "R_track",
        "R_track_pos",
        "R_track_vel",
        "R_muscle",
        "R_tsa",
        "R_stability",
        "R_time",
    ]:
        print(f"{k:14s} = {info.get(k, float('nan')):.5f}")

    print("\n[ACTION USED]")
    print(f"L  = {info.get('L', float('nan')):.5f}")
    print(f"t0 = {info.get('t0', float('nan')):.5f}")
    print(f"t1 = {info.get('t1', float('nan')):.5f}")
    print(f"t2 = {info.get('t2', float('nan')):.5f}")
    print(f"t3 = {info.get('t3', float('nan')):.5f}")


if __name__ == "__main__":
    main()