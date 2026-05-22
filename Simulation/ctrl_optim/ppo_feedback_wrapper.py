"""
Multi-step feedback PPO environment for STS assistance.

PPO observes skeleton state each timestep and outputs idealised knee-extension
torques. Biological STS controller is kept; leg output is scaled to simulate
reduced control capacity.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass,field
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium
from gymnasium import spaces
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: works when this file is at project root or inside ctrl_optim.
# ---------------------------------------------------------------------------



_THIS_DIR = Path(__file__).resolve().parent
for _p in [_THIS_DIR, _THIS_DIR / "ctrl_optim", _THIS_DIR.parent, _THIS_DIR.parent / "ctrl_optim"]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sts_ctrl import SitToStandSim, STSReflexParams  # noqa: E402


# ---------------------------------------------------------------------------
# Initial seated pose helper
# ---------------------------------------------------------------------------

def set_seated_pose(sim) -> None:
    """Mirror the seated pose used by your previous PPO wrappers."""
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


def apply_leg_control_reduction(params: STSReflexParams, scale: float = 0.90) -> STSReflexParams:
    """Scale reflex stimulation for leg groups — not muscle force params."""
    for group in ["GLU", "VAS", "SOL", "TA", "HFL", "HAM"]:
        if group in params.group_scale:
            params.group_scale[group] *= float(scale)
    return params


@dataclass
class FeedbackRewardWeights:
    progress: float = 0.20
    posture: float = 0.35
    velocity: float = 0.06
    muscle: float = 0.10
    assist: float = 0.035
    contact: float = 0.35
    symmetry: float = 0.04
    success_bonus: float = 20.0
    fall_penalty: float = 20.0


@dataclass
class FeedbackAssistConfig:
    max_steps: int = 3500
    leg_scale: float = 0.90
    tau_max: float = 60.0
    assist_sign: float = 1.0
    assist_phases: Tuple[int, ...] = (2, 3, 4)
    phase4_scale: float = 0.35
    target_duration: float = 3.0
    reward_weights: FeedbackRewardWeights = field(default_factory=FeedbackRewardWeights)


class STSFeedbackAssistEnv(gymnasium.Env):
    """Feedback PPO env: action = [assist_r, assist_l] in [0,1], obs = normalised skeleton state."""

    metadata = {"render_modes": []}

    # Observation keys and rough scale factors for normalisation.
    OBS_SPECS = [
        ("phase", 5.0),
        ("phase_elapsed", 3.0),
        ("pelvis_y", 1.0),
        ("pelvis_y_vel", 1.0),
        ("torso_y", 1.0),
        ("torso_y_vel", 1.0),
        ("root_pitch", 2.0),
        ("root_pitch_rel_vel", 8.0),
        ("trunk_lean_rel", 1.5),
        ("trunk_lean_vel", 8.0),
        ("hip_avg", 2.0),
        ("hip_avg_vel", 8.0),
        ("knee_avg", 2.0),
        ("knee_avg_vel", 8.0),
        ("ankle_avg", 1.5),
        ("ankle_avg_vel", 8.0),
        ("pelvis_to_feet_x", 1.0),
        ("left_foot_contact", 1.0),
        ("right_foot_contact", 1.0),
        ("left_foot_load", 1.0),
        ("right_foot_load", 1.0),
        ("seat_contact", 1.0),
    ]

    def __init__(
        self,
        mujoco_env,
        fb_cfg: Optional[FeedbackAssistConfig] = None,
        debug: bool = False,
        log_to_csv: bool = False,
        log_tag: Optional[str] = None,
    ):
        super().__init__()
        self.mj_env = mujoco_env
        self.sim = mujoco_env.sim
        self.fb_cfg = fb_cfg if fb_cfg is not None else FeedbackAssistConfig()
        self.debug = bool(debug)
        self.log_to_csv = bool(log_to_csv)
        self.log_tag = log_tag

        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(len(self.OBS_SPECS),),
            dtype=np.float32,
        )

        self.sts: Optional[SitToStandSim] = None
        self._step_count = 0
        self._last_obs_dict: Dict[str, float] = {}
        self._prev_pelvis_y = 0.0
        self._prev_phase = 1
        self._knee_dadr: Dict[str, int] = {}

        # Episode accumulators for terminal info.
        self._ep_reward = 0.0
        self._ep_muscle_acc = 0.0
        self._ep_assist_abs_acc = 0.0
        self._ep_assist_sq_acc = 0.0
        self._ep_assist_max = 0.0
        self._ep_contact_bad_steps = 0
        self._ep_steps = 0

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            self.mj_env.reset(seed=seed if seed is not None else 0)
        except TypeError:
            self.mj_env.reset()

        set_seated_pose(self.sim)

        params = STSReflexParams()
        apply_leg_control_reduction(params, scale=self.fb_cfg.leg_scale)

        self.sts = SitToStandSim(
            self.sim,
            self.mj_env,
            params=params,
            debug=self.debug,
            use_tsa=False,
            use_tsa_full=False,
            log_to_csv=self.log_to_csv,
            log_tag=self.log_tag,
        )
        self.sts.reset_filters()
        self.sts.get_observation()
        self.sts.capture_phase1_hold_pose()
        self.sts.reset_phase(1)

        self._resolve_knee_dofs()

        self._step_count = 0
        self._last_obs_dict = self.sts.get_observation()
        self._prev_pelvis_y = float(self._last_obs_dict.get("pelvis_y", 0.0))
        self._prev_phase = int(self._last_obs_dict.get("phase", 1))

        self._ep_reward = 0.0
        self._ep_muscle_acc = 0.0
        self._ep_assist_abs_acc = 0.0
        self._ep_assist_sq_acc = 0.0
        self._ep_assist_max = 0.0
        self._ep_contact_bad_steps = 0
        self._ep_steps = 0

        return self._make_obs(self._last_obs_dict), {}

    def close(self) -> None:
        if self.sts is not None:
            self.sts.close()
            self.sts = None

    # ------------------------------------------------------------------
    # Main Gym step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        if self.sts is None:
            raise RuntimeError("Environment must be reset before stepping.")

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        obs_before = self.sts.get_observation()
        phase_before = int(self.sts.get_phase())

        # One full MuJoCo/controller step with PPO torque inserted before env.step.
        tau_r, tau_l = self._controller_step_with_assist(action, phase_before)

        obs_after = self.sts.get_observation()
        phase_after = int(self.sts.get_phase())
        self._last_obs_dict = obs_after

        reward, reward_info = self._compute_reward(obs_after, phase_after, action, tau_r, tau_l)
        self._ep_reward += float(reward)
        self._step_count += 1
        self._ep_steps += 1

        terminated = bool(phase_after >= 4)
        truncated = bool(self._step_count >= int(self.fb_cfg.max_steps))
        failed = self._is_bad_failure(obs_after, phase_after)
        if failed:
            reward -= float(self.fb_cfg.reward_weights.fall_penalty)
            self._ep_reward += -float(self.fb_cfg.reward_weights.fall_penalty)
            terminated = True

        info = {
            **reward_info,
            "phase": float(phase_after),
            "success": float(phase_after >= 4),
            "failed": float(failed),
            "steps_used": float(self._step_count),
            "t_final": float(self.sim.data.time),
            "episode_reward": float(self._ep_reward),
            "mean_muscle": float(self._ep_muscle_acc / max(self._ep_steps, 1)),
            "mean_abs_assist_tau": float(self._ep_assist_abs_acc / max(self._ep_steps, 1)),
            "rms_assist_tau": float(np.sqrt(self._ep_assist_sq_acc / max(self._ep_steps, 1))),
            "max_assist_tau": float(self._ep_assist_max),
            "contact_bad_frac": float(self._ep_contact_bad_steps / max(self._ep_steps, 1)),
            "assist_r_cmd": float(action[0]),
            "assist_l_cmd": float(action[1]),
            "assist_tau_r": float(tau_r),
            "assist_tau_l": float(tau_l),
        }

        return self._make_obs(obs_after), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Controller step with PPO torque injection
    # ------------------------------------------------------------------

    def _controller_step_with_assist(self, action: np.ndarray, phase: int) -> Tuple[float, float]:
        # Can't use sts.step() here — it clears qfrc_applied before we can inject PPO torque.
        assert self.sts is not None
        sts = self.sts

        if phase == 1 and sts.p1_hold_qpos is None:
            sts.capture_phase1_hold_pose()

        if not sts.foot_anchor_pos:
            sts.capture_foot_anchors()

        if not sts.heel_anchor_z:
            sts.capture_heel_anchors()

        bio_action = sts.compute_action(sts.obs, phase)

        if hasattr(sts.sim.data, "qfrc_applied"):
            sts.sim.data.qfrc_applied[:] = 0.0
        if hasattr(sts.sim.data, "xfrc_applied"):
            sts.sim.data.xfrc_applied[:] = 0.0

        if hasattr(sts.sim.data, "qfrc_applied"):
            if phase == 1:
                if sts.params.hold_root_in_phase1:
                    sts.apply_phase1_root_pd_hold()
                if sts.params.hold_legs_in_phase1:
                    sts.apply_phase1_leg_pd_hold()

            sts.apply_root_pitch_forward_brake(phase)
            sts.apply_heel_down_pd(phase)

            # if phase == 4:
            #     sts.apply_phase4_hip_forward_push()

            if phase in (2, 3):
                self._apply_left_right_sync_forces(sts)

        if int(phase) in sts.params.anchor_feet_in_phases:
            sts.apply_foot_anchor_pd(dt=float(sts.obs.get("dt", sts.model.opt.timestep)))

        tau_r, tau_l = self._apply_ppo_knee_assist(action, phase)

        self._safe_env_step(bio_action)
        return tau_r, tau_l

    def _apply_left_right_sync_forces(self, sts: SitToStandSim) -> None:
        sync_specs = {
            "hip": ("hip_flexion_r", "hip_flexion_l", 45.0, 5.0, 35.0),
            "knee": ("knee_angle_r", "knee_angle_l", 180.0, 18.0, 120.0),
            "ankle": ("ankle_angle_r", "ankle_angle_l", 50.0, 6.0, 45.0),
        }

        for _, (jr, jl, kp, kd, limit) in sync_specs.items():
            jid_r = sts.name2jid[jr]
            jid_l = sts.name2jid[jl]

            qadr_r = sts.model.jnt_qposadr[jid_r]
            qadr_l = sts.model.jnt_qposadr[jid_l]
            dadr_r = sts.model.jnt_dofadr[jid_r]
            dadr_l = sts.model.jnt_dofadr[jid_l]

            qr = float(sts.sim.data.qpos[qadr_r])
            ql = float(sts.sim.data.qpos[qadr_l])
            qdr = float(sts.sim.data.qvel[dadr_r])
            qdl = float(sts.sim.data.qvel[dadr_l])

            q_avg = 0.5 * (qr + ql)
            qd_avg = 0.5 * (qdr + qdl)

            tau_r = kp * (q_avg - qr) + kd * (qd_avg - qdr)
            tau_l = kp * (q_avg - ql) + kd * (qd_avg - qdl)

            sts.sim.data.qfrc_applied[dadr_r] += float(np.clip(tau_r, -limit, limit))
            sts.sim.data.qfrc_applied[dadr_l] += float(np.clip(tau_l, -limit, limit))

    def _apply_ppo_knee_assist(self, action: np.ndarray, phase: int) -> Tuple[float, float]:
        if not hasattr(self.sim.data, "qfrc_applied"):
            return 0.0, 0.0

        if int(phase) not in self.fb_cfg.assist_phases:
            return 0.0, 0.0

        phase_scale = self.fb_cfg.phase4_scale if int(phase) == 4 else 1.0
        tau_r = float(action[0]) * float(self.fb_cfg.tau_max) * phase_scale
        tau_l = float(action[1]) * float(self.fb_cfg.tau_max) * phase_scale

        sign = float(self.fb_cfg.assist_sign)
        self.sim.data.qfrc_applied[self._knee_dadr["r"]] += sign * tau_r
        self.sim.data.qfrc_applied[self._knee_dadr["l"]] += sign * tau_l

        tau_total = abs(tau_r) + abs(tau_l)
        self._ep_assist_abs_acc += tau_total
        self._ep_assist_sq_acc += tau_total * tau_total
        self._ep_assist_max = max(self._ep_assist_max, tau_total)

        return sign * tau_r, sign * tau_l

    def _safe_env_step(self, action: np.ndarray) -> None:
        _ = self.mj_env.step(action)

    # ------------------------------------------------------------------
    # Observation/reward helpers
    # ------------------------------------------------------------------

    def _resolve_knee_dofs(self) -> None:
        assert self.sts is not None
        name2jid = {self.sts.model.joint(i).name: i for i in range(self.sts.model.njnt)}
        self._knee_dadr = {
            "r": int(self.sts.model.jnt_dofadr[name2jid["knee_angle_r"]]),
            "l": int(self.sts.model.jnt_dofadr[name2jid["knee_angle_l"]]),
        }

    def _make_obs(self, obs: Dict[str, float]) -> np.ndarray:
        vals = []
        for key, scale in self.OBS_SPECS:
            v = obs.get(key, 0.0)
            if isinstance(v, bool):
                v = float(v)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            vals.append(v / float(scale))
        return np.clip(np.asarray(vals, dtype=np.float32), -5.0, 5.0)

    def _mean_leg_activation(self) -> float:
        assert self.sts is not None
        ids = []
        for group in ["GLU_r", "GLU_l", "VAS_r", "VAS_l", "SOL_r", "SOL_l", "TA_r", "TA_l", "HFL_r", "HFL_l", "HAM_r", "HAM_l"]:
            ids.extend(self.sts.muscle_group_ids.get(group, []))
        if not ids:
            return 0.0
        return float(np.mean([self.sim.data.act[i] for i in ids]))

    def _mean_vas_activation(self) -> float:
        assert self.sts is not None
        ids = []
        for group in ["VAS_r", "VAS_l"]:
            ids.extend(self.sts.muscle_group_ids.get(group, []))
        if not ids:
            return 0.0
        return float(np.mean([self.sim.data.act[i] for i in ids]))

    def _compute_reward(
        self,
        obs: Dict[str, float],
        phase: int,
        action: np.ndarray,
        tau_r: float,
        tau_l: float,
    ) -> Tuple[float, Dict[str, float]]:
        w = self.fb_cfg.reward_weights

        pelvis_y = float(obs.get("pelvis_y", 0.0))
        torso_y = float(obs.get("torso_y", 0.0))
        pelvis_vy = float(obs.get("pelvis_y_vel", 0.0))
        torso_vy = float(obs.get("torso_y_vel", 0.0))
        root_pitch = float(obs.get("root_pitch", 0.0))
        lean = float(obs.get("trunk_lean_rel", 0.0))
        knee = float(obs.get("knee_avg", 0.0))
        hip = float(obs.get("hip_avg", 0.0))
        ankle = float(obs.get("ankle_avg", 0.0))
        pelvis_to_feet = float(obs.get("pelvis_to_feet_x", 0.0))
        grounded = bool(obs.get("grounded", False))
        seat_contact = bool(obs.get("seat_contact", False))

        dy = pelvis_y - self._prev_pelvis_y
        r_progress = 15.0 * dy + 0.02 * max(phase - self._prev_phase, 0)
        self._prev_pelvis_y = pelvis_y
        self._prev_phase = phase

        if phase <= 1:
            pelvis_target = 0.55
            torso_target = 0.62
            lean_target = 0.50
            knee_target = 1.75
            hip_target = 1.57
        elif phase == 2:
            pelvis_target = 0.66
            torso_target = 0.90
            lean_target = 0.45
            knee_target = 1.05
            hip_target = 0.55
        elif phase == 3:
            pelvis_target = 0.82
            torso_target = 0.82
            lean_target = 0.55
            knee_target = 0.18
            hip_target = -0.05
        else:
            pelvis_target = 0.84
            torso_target = 0.84
            lean_target = 0.10
            knee_target = 0.06
            hip_target = -0.15

        r_posture = -(
            3.0 * (pelvis_y - pelvis_target) ** 2
            + 2.5 * (torso_y - torso_target) ** 2
            + 0.8 * (knee - knee_target) ** 2
            + 0.5 * (hip - hip_target) ** 2
            + 0.5 * (lean - lean_target) ** 2
            + 0.8 * max(-1.18 - root_pitch, 0.0) ** 2
            + 0.8 * max(-0.08 - pelvis_to_feet, 0.0) ** 2
            + 0.4 * max(-0.50 - ankle, 0.0) ** 2
        )

        r_velocity = -(0.4 * pelvis_vy ** 2 + 0.4 * torso_vy ** 2)
        if phase in (2, 3) and pelvis_vy < -0.03:
            r_velocity -= 0.5 * abs(pelvis_vy)
        if phase in (2, 3, 4) and torso_vy < -0.04:
            r_velocity -= 0.5 * abs(torso_vy)

        mean_leg = self._mean_leg_activation()
        mean_vas = self._mean_vas_activation()
        muscle_metric = 0.65 * mean_vas + 0.35 * mean_leg
        r_muscle = -muscle_metric
        self._ep_muscle_acc += muscle_metric

        tau_norm = (abs(tau_r) + abs(tau_l)) / max(float(self.fb_cfg.tau_max), 1e-6)
        r_assist = -(tau_norm ** 2)

        r_contact = 0.0
        if phase >= 2 and not grounded:
            r_contact -= 1.0
            self._ep_contact_bad_steps += 1
        if phase >= 3 and seat_contact:
            r_contact -= 0.5

        r_symmetry = -float((float(action[0]) - float(action[1])) ** 2)

        reward = (
            w.progress * r_progress
            + w.posture * r_posture
            + w.velocity * r_velocity
            + w.muscle * r_muscle
            + w.assist * r_assist
            + w.contact * r_contact
            + w.symmetry * r_symmetry
        )
        
        phase_progress_bonus = {
            1: 0.0,
            2: 2.0,
            3: 6.0,
            4: 12.0,
            5: 25.0,
        }
        R_phase = phase_progress_bonus.get(int(phase), 0.0)
        reward+= R_phase

        if phase >= 4:
            t = float(self.sim.data.time)
            r_time_target = -((t - float(self.fb_cfg.target_duration)) / max(float(self.fb_cfg.target_duration), 1e-6)) ** 2
            reward += float(w.success_bonus) + 1.5 * r_time_target

        info = {
            "r_progress": float(r_progress),
            "r_posture": float(r_posture),
            "r_velocity": float(r_velocity),
            "r_muscle": float(r_muscle),
            "r_assist": float(r_assist),
            "r_contact": float(r_contact),
            "r_symmetry": float(r_symmetry),
            "mean_leg_activation": float(mean_leg),
            "mean_vas_activation": float(mean_vas),
            "pelvis_y": float(pelvis_y),
            "torso_y": float(torso_y),
            "root_pitch": float(root_pitch),
            "lean": float(lean),
            "knee_avg": float(knee),
            "hip_avg": float(hip),
            "ankle_avg": float(ankle),
            "pelvis_to_feet_x": float(pelvis_to_feet),
            "grounded": float(grounded),
            "seat_contact": float(seat_contact),
        }
        return float(reward), info

    def _is_bad_failure(self, obs: Dict[str, float], phase: int) -> bool:
        pelvis_y = float(obs.get("pelvis_y", 0.0))
        torso_y = float(obs.get("torso_y", 0.0))
        root_pitch = float(obs.get("root_pitch", 0.0))
        lean = float(obs.get("trunk_lean_rel", 0.0))
        grounded = bool(obs.get("grounded", False))

        if self._step_count < 20:
            return False
        if phase >= 2 and not grounded:
            return False
        if pelvis_y < 0.35 and phase >= 3:
            return True
        if torso_y < 0.45 and phase >= 3:
            return True
        if root_pitch < -1.65:
            return True
        if lean > 1.25 and phase >= 3:
            return True
        return False
