"""
TSA integration for SitToStandSim — single actuator per leg.

Assistive torque is injected as negative qfrc_applied at the knee DOF
(positive = flexion in MuJoCo, so extension assist is negative).
payload_mass is updated each step from the (knee,knee) inertia diagonal;
F_resist = (qfrc_bias + qfrc_constraint) / d.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# tsa_modelling/ contains actuator.py and model_v2.py; add it to sys.path
# so that actuator.py's own `from model_v2 import TSASimulator` resolves.
_TSA_DIR = Path(__file__).resolve().parent.parent / "tsa_modelling"
if str(_TSA_DIR) not in sys.path:
    sys.path.insert(0, str(_TSA_DIR))

from actuator import TSAActuator   
from model_v2 import TSASimulator  

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MOMENT_ARM: float = 0.045  # m — perp. distance from knee joint axis to cable line of action (confirmed 4.5 cm)

# Scales the raw (bias + constraint) knee torque down to the range the TSA
# motor can meaningfully resist (~50–100 N cable force).  The full body-weight
# load at d = 0.28 m is ~2500 N — far beyond motor capability — so any value
# that keeps F_resist above the motor's stall-tension ceiling (~64 N) correctly
# places the motor in the stall branch (T = T_des).  0.02 gives ~50 N at peak
# seated load and tapers naturally toward zero as the knee extends.
RESISTANCE_SCALE: float = 0.002  # rescaled for d = 0.03 m to keep F_resist ≈ 50 N peak

_PAYLOAD_MASS_INIT: float = 10.0  # kg — used only at construction time

# Cable-side equivalent mass [kg] set every step in the motor EOM.
# Physical basis: I_shin_about_knee / d²
#   Typical human shin+foot: I ≈ 0.10 kg·m², d = 0.045 m → m ≈ 49 kg
# Keep in sync with tsa_integration_full.py.  Tune if motor speed is wrong.
_SHIN_CABLE_MASS: float = 10.0  # kg


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _build_tsa_sim(sim_id: int, L: float) -> TSASimulator:
    """TSASimulator configured for exo knee use (gravity_along_string=False)."""
    return TSASimulator(
        id=sim_id,
        L=L,
        radius=0.004,              # 4 mm bundle radius
        payload_mass=_PAYLOAD_MASS_INIT,  # overwritten per-step from M diagonal
        I_motor=5e-5,
        pretension_theta=20 * np.pi,   # 10 turns → X₀≈68mm, J≈2.33mm/rad at L=0.5m
        max_motor_torque=0.1668,   # MP motor: 1.7 kg·cm → 0.1668 N·m
        no_load_speed=800 * 2 * np.pi / 60,   # 800 RPM → 83.8 rad/s
        b_theta=1e-4,
        b_X=0.0,
        gravity_along_string=False,  # load supplied per-step via F_resist
        max_contraction_ratio=0.30,
    )


def build_tsa_actuator(side: str, L: float) -> TSAActuator:
    sim_id = 0 if side == 'right' else 1
    tsa    = _build_tsa_sim(sim_id, L)
    return TSAActuator(tsa, side=side, moment_arm=MOMENT_ARM,
                       name=f"TSA_{side}")


# ---------------------------------------------------------------------------
# Integration class
# ---------------------------------------------------------------------------

class TSAIntegration:
    """Two TSAActuator instances (right + left knee) wired into MuJoCo."""

    def __init__(self, sim, L: float = 0.50) -> None:
        """
        Parameters
        ----------
        sim : MjSim
            Live MuJoCo simulation handle.
        L   : float
            String length [m] for both actuators.  Placeholder — to be
            optimised across 4 motors in the next phase of work.
        """
        self.sim = sim

        self.actuators: Dict[str, TSAActuator] = {
            'r': build_tsa_actuator('right', L=L),
            'l': build_tsa_actuator('left',  L=L),
        }

        # Cache knee DOF addresses once so we avoid dict lookups each step
        name2jid = {sim.model.joint(i).name: i for i in range(sim.model.njnt)}
        self._knee_dadr: Dict[str, int] = {
            'r': int(sim.model.jnt_dofadr[name2jid['knee_angle_r']]),
            'l': int(sim.model.jnt_dofadr[name2jid['knee_angle_l']]),
        }

        # Joint position addresses for reading knee angle each step.
        self._knee_qadr: Dict[str, int] = {
            'r': int(sim.model.jnt_qposadr[name2jid['knee_angle_r']]),
            'l': int(sim.model.jnt_qposadr[name2jid['knee_angle_l']]),
        }

        # Knee angles at reset — θ_seated reference for L(θ)=L₀+r·θ slack check.
        # X_geom = X₀ + r·(θ_initial − θ_current); taut when X_motor ≥ X_geom.
        self._knee_angle_initial: Dict[str, float] = {
            side: float(sim.data.qpos[self._knee_qadr[side]])
            for side in ('r', 'l')
        }

        self.last_state: Dict[str, Dict] = {'r': {}, 'l': {}}

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset both actuators to pretension initial state."""
        for act in self.actuators.values():
            act.reset()
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
        Advance both TSAs by one timestep and inject torques into MuJoCo.

        Parameters
        ----------
        t, dt        : current sim time and timestep [s]
        tau_knee_r/l : desired knee extension torque [N·m] per side.
                       Must be >= 0.  Converted to cable tension internally.
        """
        for side, tau_knee in (('r', tau_knee_r), ('l', tau_knee_l)):
            act = self.actuators[side]
            d   = act.get_moment_arm()

            act.tsa.m = _SHIN_CABLE_MASS

            T_des    = max(0.0, float(tau_knee)) / d
            F_resist = self._get_resistance(side, d)

            result = act.step(t, dt, T_des, F_resist)

            # Cable-slack check using L(θ) = L₀ + r·θ (anterior knee geometry).
            # String is taut at the start (θ_initial) with pretension X₀.
            # As the knee extends by Δθ the path shortens by r·Δθ; motor must
            # wind that shortening PLUS hold the pretension to stay taut.
            # Taut when X_motor ≥ X₀ + r·(θ_initial − θ_current).
            knee_angle  = float(self.sim.data.qpos[self._knee_qadr[side]])
            X0          = act.tsa._contraction(act.tsa.theta_pretension)
            X_geometric = X0 + d * max(0.0, self._knee_angle_initial[side] - knee_angle)
            if result['X'] < X_geometric:
                result['tension'] = 0.0
                result['torque']  = 0.0

            self.last_state[side] = result

            # Negative injection: TSA extends the knee (decreases knee_angle),
            # which is the negative generalised-force direction for this DOF.
            dadr = self._knee_dadr[side]
            self.sim.data.qfrc_applied[dadr] -= result['torque']

    # ------------------------------------------------------------------

    def _get_resistance(self, side: str, moment_arm: float) -> float:
        """
        Cable resistance force opposing contraction [N].

        During STS the TSA cable pulls the shin against the whole-body load
        (body weight above the knee is transmitted through the foot contact).
        The physically correct resistance is therefore:

            tau_net = qfrc_bias   (gravity + Coriolis at knee DOF)
                    + qfrc_constraint  (body-weight reaction through foot GRF)

        qfrc_actuator (muscle force) is intentionally excluded: including it
        drives the net to zero (muscles always over-deliver) and collapses
        F_resist to 0, leaving the TSA with nothing to push against.

        The raw tau_net at d = 0.28 m gives ~2500 N — beyond motor capability.
        RESISTANCE_SCALE (0.02) brings it to ~50 N peak, which is within the
        motor's stall-tension range and reliably keeps the motor in the stall
        branch so that T = T_des throughout the motion.
        """
        dadr       = self._knee_dadr[side]
        tau_bias   = float(self.sim.data.qfrc_bias[dadr])
        tau_constr = float(self.sim.data.qfrc_constraint[dadr])
        tau_net    = RESISTANCE_SCALE * max(0.0, tau_bias + tau_constr)
        return tau_net / moment_arm

    # ------------------------------------------------------------------

    def log_str(self) -> str:
        """One-line debug summary of the last TSA step."""
        parts = []
        for side in ('r', 'l'):
            s = self.last_state.get(side, {})
            if s:
                parts.append(
                    f"TSA_{side}: T={s.get('tension', 0.0):.1f}N "
                    f"tau={s.get('torque', 0.0):.2f}Nm "
                    f"X={s.get('X', 0.0)*1e3:.1f}mm "
                    f"sat={s.get('torque_saturated', False)}"
                )
        return " | ".join(parts)
