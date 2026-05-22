"""
TSA Integration — 4-motor-per-leg configuration.

Extends tsa_integration.py to four independent TSAActuators per leg.
Each motor has a frontal-plane offset α; sagittal torque contribution is
T_i × d × cos(α_i). Symmetric ±α pairs cancel frontal-plane moments.

Control modes: 'full_power' (each active motor runs at τ_stall, recommended)
or 'demand_share' (distributes demanded torque equally; under-delivers early).
Drop-in replacement for TSAIntegration — same step()/reset() signature.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_TSA_DIR = Path(__file__).resolve().parent.parent / "tsa_modelling"
if str(_TSA_DIR) not in sys.path:
    sys.path.insert(0, str(_TSA_DIR))

from actuator import TSAActuator
from model_v2 import TSASimulator

# ---------------------------------------------------------------------------
# Physical constants (identical to tsa_integration.py — do not diverge)
# ---------------------------------------------------------------------------

MOMENT_ARM: float = 0.045
"""Perp. distance from knee joint axis to cable line of action [m] — confirmed 4.5 cm."""

CABLE_PATH_ARM: float = 0.025
"""Effective cable path radius at the knee for the geometric slack model [m].
Smaller than MOMENT_ARM because the shin attachment is proximal to the
torque application point, reducing path length change per unit knee rotation.
Used only in X_geom = X0 + CABLE_PATH_ARM * Δangle; MOMENT_ARM is still
used for torque calculation."""

RESISTANCE_SCALE: float = 0.002
"""Fraction of raw joint torque the TSA must resist (muscles bear the rest).
Keep in sync with tsa_integration.py.  See F_resist Design Summary in
HANDOFF_v3.md before changing."""

_PAYLOAD_MASS_INIT: float = 10.0
"""Initial payload_mass for TSASimulator at construction time.
Overwritten per-step by _SHIN_CABLE_MASS — this value is only used
during construction."""

_SHIN_CABLE_MASS: float = 10.0
"""Cable-side equivalent mass [kg] used in the motor EOM each step.
Physical basis: I_shin_about_knee / d²
  Typical human shin+foot: I ≈ 0.10 kg·m², d = 0.045 m → m ≈ 49 kg
Using M_kk/d² (full-body generalised inertia) is incorrect for a
supplementary actuator — it makes the motor take ~7 s to spin up, far
beyond the 0.8 s STS window.  Tune this value if the motor is still
too slow (decrease) or unrealistically fast (increase)."""

_FULL_POWER_T_DES: float = 1e6
"""Sentinel desired tension for 'full_power' mode [N].
Guaranteed to exceed tau_avail/J at any operating point, so the actuator
always clamps tau_cmd = tau_avail (full motor effort)."""


# ---------------------------------------------------------------------------
# Motor configuration
# ---------------------------------------------------------------------------

@dataclass
class MotorConfig:
    """Static configuration for one TSA motor unit."""

    lateral_offset_deg: float = 0.0
    """Frontal-plane offset angle α [degrees].
    Positive = dextral (towards right for right leg, left for left leg).
    Negative = sinistral.
    Effective sagittal moment arm: d_nominal × cos(α).
    For |α| < 15° the torque correction is < 3.5%."""

    activation_time: float = 0.0
    """Sim time [s] at which this motor activates.  Set to np.inf to disable."""

    name: str = ""
    """Human-readable label used in log_str()."""


def build_default_motor_configs(
    t_stagger: float = 0.5,
    alpha_deg: float = 8.0,
) -> List[MotorConfig]:
    """
    Four-motor layout: two central, one dextral, one sinistral.

    Activation is staggered in time so each new motor adds torque as the
    previous one's contraction builds.

    Parameters
    ----------
    t_stagger : float
        Time gap [s] between successive motor activations.
        Set to 0 to activate all four simultaneously.
    alpha_deg : float
        Frontal-plane offset of the outer two motors [degrees].
        Typical hardware value: 5–15°.
    """
    return [
        MotorConfig(lateral_offset_deg=0.0,        activation_time=0 * t_stagger, name="M0_center"),
        MotorConfig(lateral_offset_deg=0.0,        activation_time=1 * t_stagger, name="M1_center"),
        MotorConfig(lateral_offset_deg=+alpha_deg, activation_time=2 * t_stagger, name="M2_dextral"),
        MotorConfig(lateral_offset_deg=-alpha_deg, activation_time=3 * t_stagger, name="M3_sinistral"),
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_tsa_sim(sim_id: int, L: float) -> TSASimulator:
    """TSASimulator with exo knee parameters (gravity_along_string=False)."""
    return TSASimulator(
        id=sim_id,
        L=L,
        radius=0.004,
        payload_mass=_PAYLOAD_MASS_INIT,
        I_motor=5e-5,
        pretension_theta=10 * np.pi,   # 5 turns
        max_motor_torque=0.1668,   # MP motor: 1.7 kg·cm → 0.1668 N·m
        no_load_speed=800 * 2 * np.pi / 60,   # 800 RPM → 83.8 rad/s
        b_theta=1e-4,
        b_X=0.0,
        gravity_along_string=False,
        max_contraction_ratio=0.30,
    )


# ---------------------------------------------------------------------------
# MotorUnit
# ---------------------------------------------------------------------------

class MotorUnit:
    """
    Single TSA motor with activation state and frontal-plane geometry.

    The TSAActuator is built with moment_arm = d_nominal.  The cos(α)
    geometry correction is NOT baked into the actuator's moment_arm field;
    instead it is applied to the torque output in MultiMotorLeg.step() so
    that the actuator's internal stall / EOM / Jacobian logic always uses
    d_nominal consistently.
    """

    def __init__(
        self,
        actuator: TSAActuator,
        config: MotorConfig,
        d_nominal: float,
    ) -> None:
        self.actuator  = actuator
        self.config    = config
        self.active    = False
        self._alpha    = np.radians(config.lateral_offset_deg)
        self._cos_a    = float(np.cos(self._alpha))
        self._d_nom    = d_nominal

    # ------------------------------------------------------------------

    @property
    def d_eff(self) -> float:
        """Effective sagittal moment arm d × cos(α) [m]."""
        return self._d_nom * self._cos_a

    # ------------------------------------------------------------------

    def try_activate(self, t: float) -> bool:
        """Activate if t >= activation_time.  Returns whether currently active."""
        if not self.active and t >= self.config.activation_time:
            self.active = True
        return self.active

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset actuator physics only.  Preserves activation state."""
        self.actuator.reset()

    def full_reset(self) -> None:
        """Reset actuator physics AND activation state (for new trial)."""
        self.actuator.reset()
        self.active = False


# ---------------------------------------------------------------------------
# MultiMotorLeg
# ---------------------------------------------------------------------------

class MultiMotorLeg:
    """
    Four-motor TSA assembly for one leg.

    Each motor steps independently.  Load sharing, geometry correction, and
    cable-slack zeroing are applied at this level; the summed knee torque is
    returned to TSAIntegrationFull for injection into MuJoCo.
    """

    def __init__(
        self,
        side: str,
        motor_configs: List[MotorConfig],
        L: float,
        control_mode: str = 'full_power',
    ) -> None:
        """
        Parameters
        ----------
        side         : 'right' or 'left'
        motor_configs: list of exactly 4 MotorConfig objects
        L            : untwisted string length [m]
        control_mode : 'full_power' or 'demand_share'
        """
        if len(motor_configs) != 4:
            raise ValueError(f"Expected 4 MotorConfigs, got {len(motor_configs)}")

        self.side         = side
        self.control_mode = control_mode
        self.motors: List[MotorUnit] = []

        # Unique sim_id per actuator: right = 0–3, left = 4–7
        id_offset = 0 if side == 'right' else 4
        for i, cfg in enumerate(motor_configs):
            tsa  = _build_tsa_sim(id_offset + i, L)
            act  = TSAActuator(tsa, side=side, moment_arm=MOMENT_ARM, name=cfg.name)
            unit = MotorUnit(act, cfg, d_nominal=MOMENT_ARM)
            self.motors.append(unit)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        for m in self.motors:
            m.full_reset()

    # ------------------------------------------------------------------

    def step(
        self,
        t: float,
        dt: float,
        tau_demand: float,
        F_resist_total: float,
        knee_angle: float,
        knee_angle_initial: float,
    ) -> Dict:
        """
        Advance all active motors and return the summed knee extension torque.

        Parameters
        ----------
        t, dt              : current sim time and timestep [s]
        tau_demand         : desired knee extension torque [N·m] (>= 0)
        F_resist_total     : total cable-side resistance force [N]
                             = RESISTANCE_SCALE × (bias + constraint) / d_nominal
        knee_angle         : current knee angle [rad]
        knee_angle_initial : knee angle at start of trial [rad]
                             — reference for X_geometric cable-slack check

        Returns
        -------
        dict with keys:
            torque       — total knee extension torque [N·m]
            N_active     — number of currently active motors
            motor_states — list of per-motor result dicts from actuator.step()
                           (each has 'motor_name', 'tension', 'torque', 'X', etc.)
        """
        active   = [m for m in self.motors if m.try_activate(t)]
        N_active = len(active)

        # Geometric contraction threshold: cable is taut only when X_motor > X_geom.
        # X_geom grows as the knee extends (knee_angle < knee_angle_initial) because
        # the anterior cable path shortens; the motor must wind at least X_geom just
        # to keep the string taut.  All motors share the same X_geom (same pretension
        # theta, same MOMENT_ARM, same knee angles).
        X0_common = self.motors[0].actuator.tsa._contraction(
            self.motors[0].actuator.tsa.theta_pretension
        )
        X_geom = X0_common + CABLE_PATH_ARM * max(0.0, knee_angle_initial - knee_angle)

        # Always return one state per motor in fixed order so CSV columns are
        # stable regardless of how many motors are currently active.
        motor_states = [
            {
                'motor_name': m.config.name,
                'active':     0,
                'tension':    0.0,
                'torque':     0.0,
                'X':          float(m.actuator.X),
                'theta':      float(m.actuator.theta),
                'theta_dot':  0.0,
                'saturated':  0,
            }
            for m in self.motors
        ]
        _idx = {id(m): i for i, m in enumerate(self.motors)}

        if N_active == 0:
            return {'torque': 0.0, 'N_active': 0, 'motor_states': motor_states,
                    'X_geom': X_geom}

        # Each motor bears a proportional share of the joint resistance.
        F_resist_per = F_resist_total / N_active

        # Compute per-motor desired tension based on control mode.
        if self.control_mode == 'full_power':
            # Sentinel value — always clamped to tau_avail (motor stall torque
            # at current speed) by the actuator's internal clip.
            T_des_list = [_FULL_POWER_T_DES] * N_active
        else:
            # 'demand_share': distribute demanded torque evenly, then convert
            # to cable tension using each motor's effective moment arm.
            T_des_list = []
            for m in active:
                if m.d_eff > 1e-9:
                    T = max(0.0, tau_demand / N_active) / m.d_eff
                else:
                    T = 0.0
                T_des_list.append(T)

        total_torque = 0.0

        for motor, T_des in zip(active, T_des_list):
            motor.actuator.tsa.m = _SHIN_CABLE_MASS

            # When the cable is currently slack the motor has no external load.
            # Pass zero resistance so the EOM accelerates theta_dot toward the
            # no-load speed rather than incorrectly stalling against F_resist.
            is_slack_now = motor.actuator.X < X_geom
            f_res = 0.0 if is_slack_now else F_resist_per

            result = motor.actuator.step(
                t, dt,
                desired_tension        = T_des,
                joint_resistance_force = f_res,
            )

            if result['X'] < X_geom:
                result['tension'] = 0.0
                result['torque']  = 0.0
            else:
                result['torque'] = result['tension'] * motor.d_eff

            total_torque += result['torque']
            motor_states[_idx[id(motor)]] = {
                'motor_name': motor.config.name, 'active': 1, **result
            }

        return {
            'torque':       total_torque,
            'N_active':     N_active,
            'motor_states': motor_states,
            'X_geom':       X_geom,
        }


# ---------------------------------------------------------------------------
# TSAIntegrationFull — public interface
# ---------------------------------------------------------------------------

class TSAIntegrationFull:
    """
    Four-motor-per-leg TSA integration.  Drop-in replacement for TSAIntegration.

    Usage in sts_ctrl.py (replace the single-motor constructor):

        # Before:
        self.tsa_integration = TSAIntegration(sim, L=0.50)

        # After:
        from tsa_integration_full import TSAIntegrationFull, build_default_motor_configs
        configs = build_default_motor_configs(t_stagger=0.5, alpha_deg=8.0)
        self.tsa_integration = TSAIntegrationFull(sim, L=0.50,
                                                   motor_configs=configs,
                                                   control_mode='full_power')

    All step() and reset() call sites are unchanged.

    Logging
    -------
    log_str() returns a compact summary (N_active per leg, total torque).
    For per-motor CSV logging, read last_state[side]['motor_states'] directly
    and extend the sts_ctrl.py log writer.
    """

    def __init__(
        self,
        sim,
        L: float = 0.50,
        motor_configs: Optional[List[MotorConfig]] = None,
        motor_configs_r: Optional[List[MotorConfig]] = None,
        motor_configs_l: Optional[List[MotorConfig]] = None,
        control_mode: str = 'full_power',
    ) -> None:
        """
        Parameters
        ----------
        sim            : MjSim — live MuJoCo simulation handle
        L              : untwisted string length [m] (same for all motors)
        motor_configs  : list of 4 MotorConfig objects shared for both legs;
                         None → default layout (two central at t=0 and t=0.5 s,
                         dextral at t=1.0 s, sinistral at t=1.5 s, offset 8°)
        motor_configs_r: right-leg specific configs; overrides motor_configs for right
        motor_configs_l: left-leg specific configs; overrides motor_configs for left
        control_mode   : 'full_power' (recommended) or 'demand_share'
        """
        self.sim          = sim
        self.control_mode = control_mode

        default = motor_configs or build_default_motor_configs(t_stagger=0.5, alpha_deg=8.0)
        cfgs_r = motor_configs_r or default
        cfgs_l = motor_configs_l or cfgs_r

        self.legs: Dict[str, MultiMotorLeg] = {
            'r': MultiMotorLeg('right', cfgs_r, L, control_mode),
            'l': MultiMotorLeg('left',  cfgs_l, L, control_mode),
        }

        # Cache DOF addresses once — identical approach to TSAIntegration
        name2jid = {sim.model.joint(i).name: i for i in range(sim.model.njnt)}

        self._knee_dadr: Dict[str, int] = {
            'r': int(sim.model.jnt_dofadr[name2jid['knee_angle_r']]),
            'l': int(sim.model.jnt_dofadr[name2jid['knee_angle_l']]),
        }
        self._knee_qadr: Dict[str, int] = {
            'r': int(sim.model.jnt_qposadr[name2jid['knee_angle_r']]),
            'l': int(sim.model.jnt_qposadr[name2jid['knee_angle_l']]),
        }

        # Initial knee angles — θ_seated reference for L(θ)=L₀+r·θ slack check
        self._knee_angle_initial: Dict[str, float] = {
            side: float(sim.data.qpos[self._knee_qadr[side]])
            for side in ('r', 'l')
        }

        self.last_state: Dict[str, Dict] = {'r': {}, 'l': {}}

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all motors (including activation) and re-latch knee angles."""
        for leg in self.legs.values():
            leg.reset()
        self.last_state = {'r': {}, 'l': {}}
        self._knee_angle_initial = {
            side: float(self.sim.data.qpos[self._knee_qadr[side]])
            for side in ('r', 'l')
        }

    # ------------------------------------------------------------------

    def step(
        self,
        t: float,
        dt: float,
        tau_knee_r: float,
        tau_knee_l: float,
    ) -> None:
        """
        Advance both legs by one timestep and inject torques into MuJoCo.

        Parameters — identical to TSAIntegration.step().
        tau_knee_r/l : desired knee extension torque [N·m] per side (>= 0).
        """
        for side, tau_knee in (('r', tau_knee_r), ('l', tau_knee_l)):
            dadr = self._knee_dadr[side]

            F_resist_total = self._get_resistance(side)
            knee_angle     = float(self.sim.data.qpos[self._knee_qadr[side]])

            result = self.legs[side].step(
                t                  = t,
                dt                 = dt,
                tau_demand         = max(0.0, float(tau_knee)),
                F_resist_total     = F_resist_total,
                knee_angle         = knee_angle,
                knee_angle_initial = self._knee_angle_initial[side],
            )

            self.last_state[side] = result

            # Negative injection: TSA extends the knee (negative generalised-
            # force direction for this DOF).
            self.sim.data.qfrc_applied[dadr] -= result['torque']

    # ------------------------------------------------------------------

    def _get_resistance(self, side: str) -> float:
        """
        Total cable-side resistance force [N] for this leg.

        Uses MOMENT_ARM (nominal d) for torque→force conversion — the same
        as tsa_integration.py.  MultiMotorLeg.step() divides by N_active.
        """
        dadr       = self._knee_dadr[side]
        tau_bias   = float(self.sim.data.qfrc_bias[dadr])
        tau_constr = float(self.sim.data.qfrc_constraint[dadr])
        tau_net    = RESISTANCE_SCALE * max(0.0, tau_bias + tau_constr)
        return tau_net / MOMENT_ARM

    # ------------------------------------------------------------------

    def log_str(self) -> str:
        """One-line debug summary (N_active and total torque per leg)."""
        parts = []
        for side in ('r', 'l'):
            s = self.last_state.get(side, {})
            if s:
                parts.append(
                    f"TSA_{side}(N={s.get('N_active', 0)}): "
                    f"tau={s.get('torque', 0.0):.2f}Nm"
                )
        return " | ".join(parts)

    def log_str_verbose(self) -> str:
        """Per-motor debug summary."""
        parts = []
        for side in ('r', 'l'):
            s = self.last_state.get(side, {})
            for ms in s.get('motor_states', []):
                parts.append(
                    f"{ms.get('motor_name','?')}: "
                    f"T={ms.get('tension', 0.0):.1f}N "
                    f"tau={ms.get('torque', 0.0):.3f}Nm "
                    f"X={ms.get('X', 0.0)*1e3:.1f}mm"
                )
        return " | ".join(parts)
