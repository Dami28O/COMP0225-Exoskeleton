"""
One-step Gymnasium wrapper for TSA hardware parameter optimisation.

step(θ) runs a full STS episode and returns a scalar reward.
R = w_torque·R_torque + w_muscle·R_muscle + w_time·R_time
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium
from gymnasium import spaces

# Ensure ctrl_optim is on the path when imported from train_ppo.py at root.
_CTRL_DIR = Path(__file__).resolve().parent
if str(_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTRL_DIR))

from ppo_config import TSAOptimConfig
from sts_ctrl import SitToStandSim, STSReflexParams
from tsa_integration_full import MotorConfig


# ---------------------------------------------------------------------------
# Seated-pose helper (mirrors base_code.py set_seated_pose)
# ---------------------------------------------------------------------------

def _set_seated_pose(sim) -> None:
    model = sim.model
    data  = sim.data
    name2jid = {model.joint(i).name: i for i in range(model.njnt)}

    def _set(name, val):
        jid  = name2jid[name]
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = val

    _set("root_x",        0.0)
    _set("root_z",        0.0)
    _set("root_pitch",    0.0)
    _set("hip_flexion_r", 1.57)
    _set("hip_flexion_l", 1.57)
    _set("knee_angle_r",  1.75)
    _set("knee_angle_l",  1.75)
    _set("ankle_angle_r", -0.15)
    _set("ankle_angle_l", -0.15)

    data.qvel[:] = 0.0
    sim.forward()


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class TSAOptimEnv(gymnasium.Env):
    """One-step episodic env — each step() runs a full STS episode with fixed θ."""

    metadata = {"render_modes": []}

    def __init__(self, mujoco_env, config: TSAOptimConfig):
        super().__init__()
        self.mj_env = mujoco_env
        self.cfg    = config.env_params
        self.rcfg   = config.reward_params

        n_dim = 5 if self.cfg.symmetric else 9
        low  = np.array(
            [self.cfg.L_min] + [self.cfg.t_min] * (n_dim - 1), dtype=np.float32
        )
        high = np.array(
            [self.cfg.L_max] + [self.cfg.t_max] * (n_dim - 1), dtype=np.float32
        )
        self.action_space      = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Box(
            np.zeros(1, dtype=np.float32),
            np.ones(1,  dtype=np.float32),
        )

        # Quad muscle actuator IDs — resolved on first step (model available then).
        self._quad_ids: Optional[list] = None

        # Knee DOF addresses for reading τ_demand from qfrc_bias.
        self._knee_dadr: Optional[dict] = None

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Unpack and clip parameter vector.
        L = float(np.clip(action[0], self.cfg.L_min, self.cfg.L_max))
        t_raw = np.clip(action[1:], self.cfg.t_min, self.cfg.t_max)
        t_sorted = np.sort(t_raw).tolist()   # enforce t0 ≤ t1 ≤ t2 ≤ t3

        # Build motor configs (symmetric: same for both legs).
        offsets = [0.0, 0.0, 8.0, -8.0]
        cfgs = [
            MotorConfig(lateral_offset_deg=off, activation_time=float(t), name=f"M{i}")
            for i, (off, t) in enumerate(zip(offsets, t_sorted))
        ]

        # Reset MuJoCo and build a fresh SitToStandSim with these params.
        self.mj_env.reset(seed=0)
        # _set_seated_pose(self.mj_env.sim)

        params = STSReflexParams(tsa_string_length=L)
        sts = SitToStandSim(
            self.mj_env.sim, self.mj_env,
            params=params,
            debug=False,
            use_tsa_full=True,
            tsa_motor_configs_r=cfgs,
            tsa_motor_configs_l=cfgs,
            log_to_csv=False,
        )
        sts.reset_filters()
        sts.get_observation()
        sts.capture_phase1_hold_pose()
        sts.reset_phase(1)
        sts.tsa.reset()   # re-latch knee_angle_initial, clear motor state

        # Resolve muscle IDs on first run.
        if self._quad_ids is None:
            self._resolve_muscle_ids(sts)

        components = self._run_episode(sts)
        reward = components["reward"]
        return np.zeros(1, dtype=np.float32), float(reward), True, False, components

    # ------------------------------------------------------------------

    def _resolve_muscle_ids(self, sts: SitToStandSim) -> None:
        """Resolve quad actuator IDs from muscle names in config."""
        act_map = {sts.model.actuator(i).name: i for i in range(sts.model.nu)}
        self._quad_ids = [
            act_map[name]
            for name in self.rcfg.quad_muscle_names
            if name in act_map
        ]
        if not self._quad_ids:
            import warnings
            warnings.warn(
                "No quad muscle names matched actuators in model. "
                "w_muscle reward will be zero. "
                f"Names tried: {self.rcfg.quad_muscle_names}"
            )
        name2jid = {sts.model.joint(i).name: i for i in range(sts.model.njnt)}
        self._knee_dadr = {
            side: int(sts.model.jnt_dofadr[name2jid[f"knee_angle_{side}"]])
            for side in ('r', 'l')
        }

    # ------------------------------------------------------------------

    def _run_episode(self, sts: SitToStandSim) -> float:
        dt      = float(sts.sim.model.opt.timestep)
        T_ep    = self.cfg.max_steps * dt

        torque_deficit_acc = 0.0
        muscle_acc = 0.0
        t_final = T_ep
        sts_completed = False

        for _step in range(self.cfg.max_steps):
            obs = sts.get_observation()
            phase = sts.get_phase()
            sts.step(None, phase)

            if phase >= 4:
                t_final = float(sts.sim.data.time)
                sts_completed = True
                break

            # One-sided deficit: only penalise under-delivery so late/no activation always costs.
            tsa_state = sts.tsa.last_state
            tau_del = (
                float(tsa_state.get('r', {}).get('torque', 0.0))
                + float(tsa_state.get('l', {}).get('torque', 0.0))
            )
            if self._knee_dadr is not None:
                tau_dem = sum(
                    abs(float(sts.sim.data.qfrc_bias[dadr]))
                    for dadr in self._knee_dadr.values()
                )
            else:
                tau_dem = 0.0
            tau_target = self.rcfg.support_fraction * tau_dem
            torque_deficit_acc += max(tau_target - tau_del, 0.0) * dt

            if self._quad_ids:
                quad_act = float(np.mean([
                    sts.sim.data.act[i] for i in self._quad_ids
                ]))
                muscle_acc += quad_act * dt

        t_elapsed = max(t_final, dt)
        ref = self.rcfg.torque_ref
        R_torque = -(torque_deficit_acc / (ref * t_elapsed + 1e-9))
        R_muscle = -(muscle_acc / (t_elapsed + 1e-9))
        R_time   = -(t_final / (self.rcfg.t_max_episode + 1e-9))

        R_completion = 5.0 if sts_completed else 0.0

        reward = (
            self.rcfg.w_torque * R_torque
            + self.rcfg.w_muscle * R_muscle
            + self.rcfg.w_time   * R_time
            + R_completion
        )
        return {
            "reward":       float(reward),
            "R_torque":     float(R_torque),
            "R_muscle":     float(R_muscle),
            "R_time":       float(R_time),
            "R_completion": float(R_completion),
            "t_final":      float(t_final),
        }
