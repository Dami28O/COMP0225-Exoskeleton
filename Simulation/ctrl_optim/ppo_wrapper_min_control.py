"""
Minimal-control TSA PPO wrapper.

Same one-step design as ppo_wrapper.py but reward tracks the baseline STS
kinematics while reducing VAS effort rather than targeting torque delivery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

# Ensure ctrl_optim is on the path when imported from train_ppo.py at root.
_CTRL_DIR = Path(__file__).resolve().parent
if str(_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTRL_DIR))

from ppo_config import TSAOptimConfig
from sts_ctrl import SitToStandSim, STSReflexParams
from tsa_integration_full import MotorConfig

import imageio
# ---------------------------------------------------------------------------
# Seated-pose helper
# ---------------------------------------------------------------------------

def _set_seated_pose(sim) -> None:
    model = sim.model
    data = sim.data
    name2jid = {model.joint(i).name: i for i in range(model.njnt)}

    def _set(name: str, val: float) -> None:
        jid = name2jid[name]
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = val

    _set("root_x", 0.0)
    _set("root_z", 0.0)
    _set("root_pitch", 0.0)
    _set("hip_flexion_r", 1.57)
    _set("hip_flexion_l", 1.57)
    _set("knee_angle_r", 1.75)
    _set("knee_angle_l", 1.75)
    _set("ankle_angle_r", 0.0)
    _set("ankle_angle_l", 0.0)

    data.qvel[:] = 0.0
    sim.forward()


class TSAOptimMinControlEnv(gymnasium.Env):
    """One-step env: θ = [L, t0..t3]. Reward tracks baseline STS kinematics + penalises VAS effort."""

    metadata = {"render_modes": []}

    # Features used for phase-wise reference tracking.
    POS_FEATURES: Tuple[str, ...] = (
        "hip_avg",
        "knee_avg",
        "ankle_avg",
        "root_pitch",
        "trunk_lean_rel",
        "pelvis_y",
        "torso_y",
        "pelvis_to_feet_x",
    )

    VEL_FEATURES: Tuple[str, ...] = (
        "hip_avg_vel",
        "knee_avg_vel",
        "ankle_avg_vel",
        "root_pitch_rel_vel",
        "trunk_lean_vel",
        "pelvis_y_vel",
        "torso_y_vel",
    )

    # Rough normalisation scales so one feature cannot dominate just because of units.
    POS_SCALE: Dict[str, float] = {
        "hip_avg": 0.50,
        "knee_avg": 0.60,
        "ankle_avg": 0.35,
        "root_pitch": 0.60,
        "trunk_lean_rel": 0.50,
        "pelvis_y": 0.25,
        "torso_y": 0.25,
        "pelvis_to_feet_x": 0.20,
    }

    VEL_SCALE: Dict[str, float] = {
        "hip_avg_vel": 2.0,
        "knee_avg_vel": 2.0,
        "ankle_avg_vel": 2.0,
        "root_pitch_rel_vel": 2.5,
        "trunk_lean_vel": 2.5,
        "pelvis_y_vel": 0.8,
        "torso_y_vel": 0.8,
    }

    def __init__(self, mujoco_env, config: TSAOptimConfig):
        super().__init__()
        self.mj_env = mujoco_env
        self.cfg = config.env_params
        self.rcfg = config.reward_params

        n_dim = 5 if self.cfg.symmetric else 9
        low = np.array([self.cfg.L_min] + [self.cfg.t_min] * (n_dim - 1), dtype=np.float32)
        high = np.array([self.cfg.L_max] + [self.cfg.t_max] * (n_dim - 1), dtype=np.float32)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Box(
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )

        self._target_muscle_ids: Optional[List[int]] = None
        self._reference = self._build_reference_trajectory()

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        L = float(np.clip(action[0], self.cfg.L_min, self.cfg.L_max))
        t_raw = np.clip(action[1:], self.cfg.t_min, self.cfg.t_max)
        t_sorted = np.sort(t_raw).astype(float).tolist()

        # Same four-motor symmetric layout as the original wrapper.
        offsets = [0.0, 0.0, 8.0, -8.0]
        cfgs = [
            MotorConfig(lateral_offset_deg=off, activation_time=float(t), name=f"M{i}")
            for i, (off, t) in enumerate(zip(offsets, t_sorted))
        ]

        self.mj_env.reset(seed=0)
        _set_seated_pose(self.mj_env.sim)

        params = STSReflexParams(tsa_string_length=L)
        sts = SitToStandSim(
            self.mj_env.sim,
            self.mj_env,
            params=params,
            debug=True,
            use_tsa_full=True,
            tsa_motor_configs_r=cfgs,
            tsa_motor_configs_l=cfgs,
            log_to_csv=True,
            log_tag="ppo_best_eval",
        )
        
        LEG_REDUCTION = 0.9
        for group in ["GLU", "VAS", "SOL", "TA", "HFL", "HAM"]:
            params.group_scale[group] *= LEG_REDUCTION
        
        # sts = SitToStandSim(
        #     self.mj_env.sim,
        #     self.mj_env,
        #     params=params,
        #     debug=False,
        #     use_tsa_full=True,
        #     tsa_motor_configs_r=cfgs,
        #     tsa_motor_configs_l=cfgs,
        #     log_to_csv=False,
        # )
        
        # sts = SitToStandSim(
        #     self.mj_env.sim,
        #     self.mj_env,
        #     params=params,
        #     debug=True,
        #     use_tsa_full=False,
        #     log_to_csv= True,
        #     log_tag="baseline",
        # )
        sts.reset_filters()
        sts.get_observation()
        sts.capture_phase1_hold_pose()
        sts.reset_phase(1)
        if sts.tsa is not None:
            sts.tsa.reset()
        frames = []
        # for step in range(3500):
        #     obs = sts.get_observation()
        #     phase = sts.get_phase()
        #     if phase <=3:
        #         sts.step(None, phase)
                

        #         frame = self.mj_env.mj_render()
        #         frames.append(frame)


        # imageio.mimsave("ppo_assisted_sts.mp4", frames, fps=30)

        if self._target_muscle_ids is None:
            self._target_muscle_ids = self._resolve_target_muscle_ids(sts)

        components = self._run_episode(sts)
        components.update(
            {
                "L": float(L),
                "t0": float(t_sorted[0]),
                "t1": float(t_sorted[1]),
                "t2": float(t_sorted[2]),
                "t3": float(t_sorted[3]),
            }
        )

        return np.zeros(1, dtype=np.float32), float(components["reward"]), True, False, components

    # ------------------------------------------------------------------

    def _make_baseline_sts(self) -> SitToStandSim:
        self.mj_env.reset(seed=0)
        _set_seated_pose(self.mj_env.sim)
        sts = SitToStandSim(
            self.mj_env.sim,
            self.mj_env,
            params=STSReflexParams(),
            debug=False,
            use_tsa_full=False,
            log_to_csv=False,
        )
        sts.reset_filters()
        sts.get_observation()
        sts.capture_phase1_hold_pose()
        sts.reset_phase(1)
        return sts

    def _build_reference_trajectory(self) -> Dict[int, List[Dict[str, float]]]:
        """Run the known-good controller once without TSA and store phase-wise targets."""
        sts = self._make_baseline_sts()
        ref: Dict[int, List[Dict[str, float]]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        for _ in range(int(self.cfg.max_steps)):
            obs = sts.get_observation()
            phase = int(sts.get_phase())
            row = {"phase_elapsed": float(obs.get("phase_elapsed", 0.0))}

            for key in self.POS_FEATURES + self.VEL_FEATURES:
                val = float(obs.get(key, np.nan))
                row[key] = val if np.isfinite(val) else 0.0

            ref.setdefault(phase, []).append(row)

            if phase >= 4:
                break

            sts.step(None, phase)

        sts.close()
        self.mj_env.reset(seed=0)
        _set_seated_pose(self.mj_env.sim)
        return ref

    def _target_for(self, phase: int, phase_elapsed: float) -> Optional[Dict[str, float]]:
        rows = self._reference.get(int(phase), [])
        if not rows:
            rows = self._reference.get(4, []) or self._reference.get(3, [])
        if not rows:
            return None

        times = np.asarray([r["phase_elapsed"] for r in rows], dtype=float)
        idx = int(np.argmin(np.abs(times - float(phase_elapsed))))
        return rows[idx]

    def _resolve_target_muscle_ids(self, sts: SitToStandSim) -> List[int]:
        # VAS only — RF is biarticular and model naming is inconsistent.
        ids: List[int] = []
        for group_name in ("VAS_r", "VAS_l"):
            ids.extend(sts.muscle_group_ids.get(group_name, []))
        return sorted(set(int(i) for i in ids))

    # ------------------------------------------------------------------

    def _safe_activation_mean_sq(self, sts: SitToStandSim, ids: List[int]) -> float:
        if not ids:
            return 0.0

        vals = []
        for i in ids:
            try:
                vals.append(float(sts.sim.data.act[i]))
            except Exception:
                vals.append(float(sts.sim.data.ctrl[i]))
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr * arr))

    def _tracking_errors(self, obs: Dict[str, float], phase: int) -> Tuple[float, float]:
        target = self._target_for(phase, float(obs.get("phase_elapsed", 0.0)))
        if target is None:
            return 0.0, 0.0

        pos_err = 0.0
        for key in self.POS_FEATURES:
            scale = self.POS_SCALE.get(key, 1.0)
            y = float(obs.get(key, 0.0))
            y_ref = float(target.get(key, 0.0))
            if np.isfinite(y) and np.isfinite(y_ref):
                pos_err += ((y - y_ref) / max(scale, 1e-6)) ** 2

        vel_err = 0.0
        for key in self.VEL_FEATURES:
            scale = self.VEL_SCALE.get(key, 1.0)
            y = float(obs.get(key, 0.0))
            y_ref = float(target.get(key, 0.0))
            if np.isfinite(y) and np.isfinite(y_ref):
                vel_err += ((y - y_ref) / max(scale, 1e-6)) ** 2

        return float(pos_err / len(self.POS_FEATURES)), float(vel_err / len(self.VEL_FEATURES))

    def _run_episode(self, sts: SitToStandSim) -> Dict[str, float]:
        dt_default = float(sts.sim.model.opt.timestep)
        T_ep = float(self.cfg.max_steps) * dt_default

        pos_track_acc = 0.0
        vel_track_acc = 0.0
        muscle_acc = 0.0
        tsa_effort_acc = 0.0
        stability_penalty_acc = 0.0
        t_final = T_ep
        success = False
        final_obs: Dict[str, float] = {}

        target_ids = self._target_muscle_ids or []
        tau_ref = float(getattr(self.rcfg, "torque_ref", 100.0))

        for _step in range(int(self.cfg.max_steps)):
            obs = sts.get_observation()
            phase = int(sts.get_phase())
            dt = float(obs.get("dt", dt_default))
            final_obs = dict(obs)

            if phase >= 4:
                success = True
                t_final = float(sts.sim.data.time)
                break

            pos_err, vel_err = self._tracking_errors(obs, phase)
            pos_track_acc += pos_err * dt
            vel_track_acc += vel_err * dt

            muscle_acc += self._safe_activation_mean_sq(sts, target_ids) * dt

            if sts.tsa is not None:
                state = sts.tsa.last_state
                tau_r = float(state.get("r", {}).get("torque", 0.0))
                tau_l = float(state.get("l", {}).get("torque", 0.0))
                tsa_effort_acc += ((tau_r / tau_ref) ** 2 + (tau_l / tau_ref) ** 2) * dt

            if phase >= 2 and bool(obs.get("seat_contact", False)) and float(obs.get("pelvis_y", 0.0)) > 0.70:
                stability_penalty_acc += 1.0 * dt
            if not bool(obs.get("left_foot_contact", False)) or not bool(obs.get("right_foot_contact", False)):
                stability_penalty_acc += 0.5 * dt
            if float(obs.get("root_pitch", 0.0)) < -1.35:
                stability_penalty_acc += 1.0 * dt

            sts.step(None, phase)

        t_elapsed = max(float(t_final), dt_default)

        R_track_pos = -(pos_track_acc / t_elapsed)
        R_track_vel = -(vel_track_acc / t_elapsed)
        R_muscle = -(muscle_acc / t_elapsed)
        R_tsa = -(tsa_effort_acc / t_elapsed)
        R_stability = -(stability_penalty_acc / t_elapsed)

        R_success = 1.0 if success else -1.0

        ref_end = self._reference_end_time(default=T_ep)
        R_time = -abs(t_final - ref_end) / max(ref_end, dt_default)

        w_success = float(getattr(self.rcfg, "w_success", 4.0))
        w_track_pos = float(getattr(self.rcfg, "w_track_pos", 1.4))
        w_track_vel = float(getattr(self.rcfg, "w_track_vel", 0.25))
        w_muscle = float(getattr(self.rcfg, "w_muscle", 1.0))
        w_tsa = float(getattr(self.rcfg, "w_tsa", 0.05))
        w_stability = float(getattr(self.rcfg, "w_stability", 1.2))
        w_time = float(getattr(self.rcfg, "w_time", 0.25))

        reward = (
            w_success * R_success
            + w_track_pos * R_track_pos
            + w_track_vel * R_track_vel
            + w_muscle * R_muscle
            + w_tsa * R_tsa
            + w_stability * R_stability
            + w_time * R_time
        )

        return {
            "reward": float(reward),
            "success": float(success),
            "R_success": float(R_success),
            "R_track_pos": float(R_track_pos),
            "R_track_vel": float(R_track_vel),
            "R_muscle": float(R_muscle),
            "R_tsa": float(R_tsa),
            "R_stability": float(R_stability),
            "R_time": float(R_time),
            "t_final": float(t_final),
            "ref_t_final": float(ref_end),
            "final_phase": float(final_obs.get("phase", -1)),
            "final_pelvis_y": float(final_obs.get("pelvis_y", np.nan)),
            "final_torso_y": float(final_obs.get("torso_y", np.nan)),
            "final_root_pitch": float(final_obs.get("root_pitch", np.nan)),
            "final_lean": float(final_obs.get("trunk_lean_rel", np.nan)),
            "final_knee": float(final_obs.get("knee_avg", np.nan)),
            "final_hip": float(final_obs.get("hip_avg", np.nan)),
        }

    def _reference_end_time(self, default: float) -> float:
        rows = self._reference.get(5, []) or self._reference.get(4, []) or self._reference.get(3, [])
        if not rows:
            return float(default)
        total = 0.0
        for ph in sorted(self._reference.keys()):
            ph_rows = self._reference.get(ph, [])
            if ph_rows:
                total += float(ph_rows[-1].get("phase_elapsed", 0.0))
        return max(total, 1e-6)


# Backwards-compatible alias: train scripts can import TSAOptimEnv if desired.
TSAOptimEnv = TSAOptimMinControlEnv
