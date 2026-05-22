"""
4-phase reflex STS controller for MyoSuite.

Handles slide root_x/root_z + hinge root_pitch joints (not free-joint quaternion).
Load-based foot contact; feet anchored phases 1-3; root held only in phase 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from tsa_integration import TSAIntegration
    _TSA_AVAILABLE = True
except ImportError:
    _TSA_AVAILABLE = False

try:
    from tsa_integration_full import TSAIntegrationFull, build_default_motor_configs
    _TSA_FULL_AVAILABLE = True
except ImportError:
    _TSA_FULL_AVAILABLE = False


# =============================================================================
# Helpers
# =============================================================================


def _pos(x: float) -> float:
    """Positive part, used like np.maximum(..., 0) in reflex modules."""
    return float(max(float(x), 0.0))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(float(x), lo, hi))


class AccEstimator:
    """Low-pass filtered finite-difference acceleration estimator."""

    def __init__(self, size: int, alpha: float = 0.2):
        self.prev_qvel: Optional[np.ndarray] = None
        self.prev_qacc = np.zeros(size, dtype=float)
        self.alpha = float(alpha)

    def reset(self) -> None:
        self.prev_qvel = None
        self.prev_qacc[:] = 0.0

    def update(self, qvel: np.ndarray, dt: float) -> np.ndarray:
        qvel = np.asarray(qvel, dtype=float)
        if self.prev_qvel is None or dt <= 1e-8:
            qacc = np.zeros_like(qvel)
        else:
            raw = (qvel - self.prev_qvel) / dt
            qacc = self.alpha * raw + (1.0 - self.alpha) * self.prev_qacc

        self.prev_qvel = qvel.copy()
        self.prev_qacc = qacc.copy()
        return qacc


# =============================================================================
# Parameters
# =============================================================================


@dataclass
class STSPhaseState:
    phase: int = 1
    previous_phase: int = 1
    phase_start_time: float = 0.0
    phase_elapsed: float = 0.0
    phase_changed: bool = False
    p1_done: bool = False
    p2_done: bool = False
    p3_done: bool = False


@dataclass
class STSReflexParams:
    # -------------------------------------------------------------------------
    # Phase timings
    # -------------------------------------------------------------------------
    p1_min_time: float = 0.5
    p1_timeout: float = 0.6
    # Phase 2 is deliberately longer now so it can build upward pelvis velocity.
    p2_min_time: float = 0.6
    p2_timeout: float = 0.5

    p3_min_time: float = 0.5
    p3_timeout: float = 1.20

    # -------------------------------------------------------------------------
    # Phase transition thresholds
    # -------------------------------------------------------------------------
    p1_lean_ready: float = 0.30
    p1_forward_ready: float = 0.12
    p1_hip_safe: float = 0.70
    p1_knee_safe: float = 1.15

    # Phase 2 should not enter Phase 3 unless the torso is not already collapsing.
    p2_pelvis_rise_vel: float = 0.045
    p2_pelvis_lift_height: float = 0.58
    p2_hip_safe: float = 0.70
    p2_knee_safe: float = 1.20

    # New stricter P2 -> P3 guards.
    p2_to_p3_lean_max: float = 0.58
    p2_to_p3_min_torso_gap: float = 0.035
    p2_to_p3_min_torso_z_vel: float = -0.03
    p2_to_p3_min_hip: float = 0.25
    
    p2_to_p3_min_pelvis_vz: float = 0.15


    p3_pelvis_stand_height: float = 0.78


    

    p3_knee_stand: float = 0.12
    p3_hip_stand: float = -0.05
    p3_torso_upright: float = -0.12

    # -------------------------------------------------------------------------
    # Targets / limits
    # -------------------------------------------------------------------------
    p1_lean_target: float = 0.5
    p1_lean_hard_limit: float = 0.28
    p1_forward_target: float = 0.13

    p2_pelvis_target: float = 0.66
    p2_lean_limit: float = 0.48
    p2_knee_stop: float = 0.95
    p2_hip_stop: float = 0.55
    
    p3_pelvis_target: float = 0.82
    p3_torso_target_z: float = 0.88
    p3_lean_limit: float = 0.75
    p3_knee_stop: float = 0.18
    p3_hip_stop: float = -0.08
    p3_torso_rise_vel_min: float = 0.005
    p3_lean_hard_limit: float = 0.42

    # Phase 3 pelvis upward-velocity rescue.
    p3_pelvis_vz_target: float = 0.16
    p3_min_upward_drive_until_height: float = 0.86
    k_p3_pelvis_vz_glu: float = 2.40
    k_p3_pelvis_vz_torso: float = 0.55
    p3_vas_support_scale: float = 0.65
    p3_sol_support_scale: float = 0.55
    p3_min_vas_support: float = 0.60
    p3_min_vas_support_hip_spent: float = 0.65
    p3_min_sol_support_hip_spent: float = 0.35

    p4_lean_target: float = 0.00
    p4_ankle_target: float = -0.10


    # -------------------------------------------------------------------------
    # Phase 1 planted-leg/root support
    # -------------------------------------------------------------------------
    hold_legs_in_phase1: bool = True
    p1_leg_hold_kp: float = 80.0
    p1_leg_hold_kd: float = 8.0

    hold_root_in_phase1: bool = True
    p1_root_x_hold_kp: float = 40.0
    p1_root_x_hold_kd: float = 6.0
    p1_root_z_hold_kp: float = 260.0
    p1_root_z_hold_kd: float = 30.0
    p1_root_pitch_hold_kp: float = 70.0
    p1_root_pitch_hold_kd: float = 10.0
    p1_root_tau_limit: float = 260.0

    # -------------------------------------------------------------------------
    # Foot contact and anchoring
    # -------------------------------------------------------------------------
    foot_contact_load_threshold: float = 0.07

    # Keep feet planted through Phase 3. Do not root-hold in Phase 2/3.
    anchor_feet_in_phases: Tuple[int, ...] = (1,2,3)
    foot_anchor_kp: float = 2200.0
    foot_anchor_kd: float = 150.0
    foot_anchor_force_limit: float = 600.0
    foot_anchor_vertical_only: bool = False

    # Debug option.
    p1_zero_all_action_debug: bool = False

    # -------------------------------------------------------------------------
    # Baseline activation
    # -------------------------------------------------------------------------
    tonic: float = 0.01

    # -------------------------------------------------------------------------
    # Reflex module gains
    # -------------------------------------------------------------------------
    k_p1_torso_flex_lean: float = 0.025
    k_p1_torso_flex_forward: float = 0.010
    k_p1_torso_brake: float = 2.20
    k_p1_ta: float = 0.0
    k_p1_sol_brake: float = 0.0

    # Phase 2 is now hip-dominant.
    k_p2_glu_lift: float = 3.60
    k_p2_vas_lift: float = 1.25
    k_p2_sol_lift: float = 0.55
    k_p2_torso_brake: float = 1.40
    k_p2_vas_inhibit: float = 0.90
    k_p2_glu_inhibit: float = 0.70

    k_p3_trunk_lift: float = 1.40


    k_p3_glu_extend: float = 0.85
    k_p3_vas_extend: float = 0.75
    k_p3_sol_support: float = 0.85
    k_p3_torso_extend: float = 1.80


    k_p3_vas_inhibit: float = 1.00
    k_p3_glu_inhibit: float = 0.75

    k_p4_hip: float = 0.30
    k_p4_knee: float = 0.30
    k_p4_ankle: float = 0.65
    k_p4_torso: float = 0.35
    k_p4_cocontract: float = 0.035
    
    # -------------------------------------------------------------------------
    # Phase 4 rise continuation before final stabilization
    # -------------------------------------------------------------------------
    p4_pelvis_target: float = 0.84
    p4_pelvis_vz_target: float = 0.08
    p4_rise_until_height: float = 0.82
    p4_release_anchor_height: float = 0.82
    p4_release_anchor_vz_min: float = -0.02
    
    k_p4_glu_rise: float = 1.20
    k_p4_vas_support: float = 0.75
    k_p4_sol_support: float = 0.45
    k_p4_torso_rise: float = 0.80

    p4_min_vas_support: float = 0.55
    p4_min_sol_support: float = 0.18
        
        # -------------------------------------------------------------------------
    # Root pitch braking after seat-off
    # -------------------------------------------------------------------------
    brake_root_pitch_after_p1: bool = True

    p23_root_pitch_max_forward: float = -0.72
    p23_root_pitch_brake_kp: float = 520.0
    p23_root_pitch_brake_kd: float = 70.0
    p23_root_pitch_brake_tau_limit: float = 520.0

    p4_root_pitch_max_forward: float = -0.95
    p4_root_pitch_brake_kp: float = 260.0
    p4_root_pitch_brake_kd: float = 32.0
    p4_root_pitch_brake_tau_limit: float = 260.0
    
    # -------------------------------------------------------------------------
    # Phase 4 pelvis/hip forward recovery
    # -------------------------------------------------------------------------
    p4_hip_forward_target: float = 0.10      # desired pelvis ahead of current value
    p4_forward_min: float = 0.14             # if forward_metric below this, push pelvis forward
    p4_forward_good: float = 0.22            # enough forward placement
    p4_root_x_push_kp: float = 220.0
    p4_root_x_push_kd: float = 35.0
    p4_root_x_push_limit: float = 180.0

    # Phase 4 torso anti-backbend
    # STSReflexParams
    p4_max_torso_ext_normal: float = 0.20
    p4_max_torso_ext_recovery: float = 0.34
    p4_hip_target: float = -0.18
    p4_knee_target: float = 0.02
    p4_backlean_root_pitch: float = -0.75    # less negative than this = leaning too far back/upright
    p4_backlean_lean_rel: float = 0.35
        
    
    p3_min_torso_above_pelvis: float = 0.00
    k_p3_torso_pelvis_gap: float = 2.50
    k_p3_torso_height: float = 0.80
    k_p3_lean_recovery: float = 2.20
    
    # Phase 1 strict lower-limb lock
    p1_leg_q_tol: float = 0.025       # radians, allowed joint drift
    p1_leg_qd_tol: float = 0.20       # rad/s
    p1_leg_lock_kp: float = 420.0
    p1_leg_lock_kd: float = 45.0
    p1_leg_lock_tau_limit: float = 220.0
    
    p1_flatfoot_tol: float = 0.015
    p1_seat_height_tol: float = 0.020

    # Functional-group output scales.
    group_scale: Dict[str, float] = field(
        default_factory=lambda: {
            "GLU": 1.00,
            "VAS": 0.55,
            "SOL": 0.38,
            "TA": 0.35,
            "HFL": 0.25,
            "HAM": 0.12,
            "TORSO_EXT": 1.00,
            "TORSO_FLEX": 0.45,
        }
    )

    tsa_string_length: float = 0.50


# =============================================================================
# Reflex-module STS controller
# =============================================================================


class SitToStandSim:
    """Simulation wrapper and reflex-module controller."""

    m_keys = ["GLU", "VAS", "SOL", "TA", "HFL", "HAM", "TORSO_EXT", "TORSO_FLEX"]

    def __init__(self, sim, env, params: Optional[STSReflexParams] = None, debug: bool = True,
                 use_tsa: bool = False, use_tsa_full: bool = False,
                 tsa_motor_configs_r: Optional[List] = None,
                 tsa_motor_configs_l: Optional[List] = None,
                 log_to_csv: bool = True,
                 log_tag: Optional[str] = None):
        self.env = env
        self.sim = sim
        self.model = sim.model
        self.params = params if params is not None else STSReflexParams()
        self.debug = bool(debug)

        self.joints_of_interest = [
            "flex_extension",
            "hip_flexion_r",
            "hip_flexion_l",
            "knee_angle_r",
            "knee_angle_l",
            "ankle_angle_r",
            "ankle_angle_l",
        ]

        self.name2jid = {self.model.joint(i).name: i for i in range(self.model.njnt)}

        required_root_joints = ["root_x", "root_z", "root_pitch"]
        missing_root = [j for j in required_root_joints if j not in self.name2jid]
        if missing_root:
            raise KeyError(
                f"Missing expected root joints {missing_root}. "
                "This code assumes root_x, root_z, and root_pitch joints."
            )

        missing = [j for j in self.joints_of_interest if j not in self.name2jid]
        if missing:
            raise KeyError(f"Missing expected joints in model: {missing}")

        self.jids_of_interest = [self.name2jid[j] for j in self.joints_of_interest]
        self.acc_est = AccEstimator(size=len(self.jids_of_interest), alpha=0.20)
        self.acc_est_xz = AccEstimator(size=4, alpha=0.20)

        self.prev_time: Optional[float] = None
        self.prev_xz_pos: Optional[np.ndarray] = None

        self.initial_root_pitch: Optional[float] = None
        self.prev_root_pitch_rel: Optional[float] = None
        self.filtered_root_pitch_rel_vel = 0.0

        self.initial_trunk_lean: Optional[float] = None
        self.prev_trunk_lean_rel: Optional[float] = None
        self.filtered_trunk_lean_vel = 0.0

        self.obs: Dict[str, float] = {}
        self.phase_state = STSPhaseState(
            phase=1,
            previous_phase=1,
            phase_start_time=float(self.sim.data.time),
        )
        self.phase = 1

        self.act_name_to_id = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.muscle_groups = self._build_muscle_groups()
        self.muscle_group_ids = self._compile_muscle_groups(self.muscle_groups)
        if self.debug:
            for base in ["GLU", "VAS", "SOL", "TA", "HFL", "HAM"]:
                r_count = len(self.muscle_group_ids.get(f"{base}_r", []))
                l_count = len(self.muscle_group_ids.get(f"{base}_l", []))
                print(f"[GROUP CHECK] {base}: R={r_count}, L={l_count}")

        self.module_outputs: Dict[str, float] = {}
        self.last_stim: Dict[str, float] = {}

        self.p1_hold_qpos: Optional[np.ndarray] = None
        self.p1_hold_root: Optional[Dict[str, float]] = None

        self.foot_anchor_pos: Dict[str, np.ndarray] = {}
        self.heel_anchor_z: Dict[str, float] = {}
        self.prev_foot_anchor_pos: Dict[str, np.ndarray] = {}

        # TSA exoskeleton integration.
        self.tsa: Optional["TSAIntegrationFull"] = None
        self._tsa_csv_file = None
        self._tsa_csv_writer = None
        self._tsa_step_count: int = 0
        self._is_tsa_full: bool = False
        if use_tsa_full:
            if not _TSA_FULL_AVAILABLE:
                raise ImportError(
                    "use_tsa_full=True but TSAIntegrationFull could not be imported. "
                    "Check that ctrl_optim/tsa_integration_full.py is present."
                )
            self._is_tsa_full = True
            default_cfgs = build_default_motor_configs(t_stagger=0.5, alpha_deg=8.0)
            cfgs_r = tsa_motor_configs_r if tsa_motor_configs_r is not None else default_cfgs
            cfgs_l = tsa_motor_configs_l if tsa_motor_configs_l is not None else default_cfgs
            self.tsa = TSAIntegrationFull(
                sim, L=self.params.tsa_string_length,
                motor_configs_r=cfgs_r,
                motor_configs_l=cfgs_l,
                control_mode='full_power',
            )
            if log_to_csv:
                _base = Path(__file__).resolve().parent.parent / "logs"
                logs_dir = (_base / "optimised") if log_tag else (_base / "full")
                logs_dir.mkdir(parents=True, exist_ok=True)
                _stem = f"{log_tag}_tsa_log_full.csv" if log_tag else "tsa_log_full.csv"
                csv_path = logs_dir / _stem
                self._tsa_csv_file = open(csv_path, "w", newline="")
                self._tsa_csv_writer = csv.writer(self._tsa_csv_file)
                motor_cols: List[str] = []
                for _side in ('r', 'l'):
                    for _mi in range(4):
                        _tag = f"{_side}_m{_mi}"
                        motor_cols += [
                            f"{_tag}_active", f"{_tag}_tension_N", f"{_tag}_torque_Nm",
                            f"{_tag}_X_mm", f"{_tag}_theta_rad", f"{_tag}_theta_dot_rads", f"{_tag}_sat",
                        ]
                self._tsa_csv_writer.writerow([
                    "step", "time", "phase",
                    "knee_r_rad", "knee_l_rad",
                    "tau_demand_r_Nm", "tau_demand_l_Nm",
                    "N_active_r", "N_active_l",
                    "total_torque_r_Nm", "total_torque_l_Nm",
                    "F_resist_r_N", "F_resist_l_N",
                    "X_geom_r_mm", "X_geom_l_mm",
                    *motor_cols,
                ])
        elif use_tsa:
            if not _TSA_AVAILABLE:
                raise ImportError(
                    "use_tsa=True but TSAIntegration could not be imported. "
                    "Check that tsa_modelling/ is on the path."
                )
            self.tsa = TSAIntegration(sim, L=self.params.tsa_string_length)  # type: ignore[assignment]
            if log_to_csv:
                csv_path = Path(__file__).resolve().parent.parent / "logs" / "tsa_log.csv"
                self._tsa_csv_file = open(csv_path, "w", newline="")
                self._tsa_csv_writer = csv.writer(self._tsa_csv_file)
                self._tsa_csv_writer.writerow([
                    "step", "time", "phase",
                    "knee_r_rad", "knee_l_rad",
                    "tau_demand_r_Nm", "tau_demand_l_Nm",
                    "tension_r_N", "torque_r_Nm", "X_r_mm", "theta_r_rad", "theta_dot_r_rads", "sat_r",
                    "tension_l_N", "torque_l_Nm", "X_l_mm", "theta_l_rad", "theta_dot_l_rads", "sat_l",
                    "payload_mass_r_kg", "payload_mass_l_kg",
                    "F_resist_r_N", "F_resist_l_N",
                ])

        if log_to_csv:
            _base = Path(__file__).resolve().parent.parent / "logs"
            _log_dir = (_base / "optimised") if log_tag else _base
            _log_dir.mkdir(parents=True, exist_ok=True)
            _stem = f"{log_tag}_diag.csv" if log_tag else "diag_p4.csv"
            diag_path = _log_dir / _stem
            self._diag_csv_file = open(diag_path, "w", newline="")
            self._diag_csv_writer = csv.writer(self._diag_csv_file)
            self._diag_csv_writer.writerow([
                "step", "time", "phase",
                "root_pitch", "root_pitch_rel_vel",
                "flex_extension",
                "lean",
                "pelvis_z", "pelvis_z_vel",
                "hip", "knee_r", "knee_l", "ankle",
                "S_P4_TORSO_EXT", "S_P4_TORSO_FLEX",
                "S_P4_GLU", "S_P4_VAS", "S_P4_HAM", "S_P4_HFL",
                "stim_TORSO_EXT", "stim_TORSO_FLEX",
                "stim_GLU", "stim_VAS", "stim_HAM", "stim_HFL",
            ])
        else:
            self._diag_csv_file = None
            self._diag_csv_writer = None
        self._diag_step = 0

    # ------------------------------------------------------------------
    # Muscle grouping
    # ------------------------------------------------------------------

    def _build_muscle_groups(self) -> Dict[str, List[str]]:
        return {
            "GLU_r": ["glmax1_r", "glmax2_r", "glmax3_r", "glutmax_r"],
            "GLU_l": ["glmax1_l", "glmax2_l", "glmax3_l", "glutmax_l"],

            "VAS_r": ["vasint_r", "vaslat_r", "vasmed_r", "vasti_r"],
            "VAS_l": ["vasint_l", "vaslat_l", "vasmed_l", "vasti_l"],

            "SOL_r": ["soleus_r", "tibpost_r", "perlong_r", "perbrev_r"],
            "SOL_l": ["soleus_l", "tibpost_l", "perlong_l", "perbrev_l"],

            "TA_r": ["tibant_r", "edl_r", "ehl_r"],
            "TA_l": ["tibant_l", "edl_l", "ehl_l"],

            "HFL_r": ["iliacus_r", "psoas_r", "sart_r", "tfl_r", "iliopsoas_r"],
            "HFL_l": ["iliacus_l", "psoas_l", "sart_l", "tfl_l", "iliopsoas_l"],

            "HAM_r": ["semimem_r", "semiten_r", "bflh_r", "hamstrings_r"],
            "HAM_l": ["semimem_l", "semiten_l", "bflh_l", "hamstrings_l"],

            "TORSO_FLEX": [
                "rect_abd_l", "rect_abd_r",
                "EO1_l", "EO1_r",
                "IO1_l", "IO1_r",
            ],
            "TORSO_EXT": [
                "MF_m1s_l", "MF_m1s_r",
                "Ps_L4_L5_IVD_l", "Ps_L4_L5_IVD_r",
                "LTpL_L1_l", "LTpL_L1_r",
            ],
        }

    def _compile_muscle_groups(self, groups: Dict[str, Iterable[str]]) -> Dict[str, List[int]]:
        compiled: Dict[str, List[int]] = {}
        for group_name, muscle_names in groups.items():
            compiled[group_name] = [
                self.act_name_to_id[m]
                for m in muscle_names
                if m in self.act_name_to_id
            ]
        return compiled

    def _add_group(self, ctrl: np.ndarray, group_name: str, amount: float) -> None:
        amount = max(float(amount), 0.0)
        if amount <= 0.0:
            return

        for idx in self.muscle_group_ids.get(group_name, []):
            ctrl[idx] += amount

    def _add_bilateral(self, ctrl: np.ndarray, base_group: str, amount: float) -> None:
        self._add_group(ctrl, f"{base_group}_r", amount)
        self._add_group(ctrl, f"{base_group}_l", amount)

    # ------------------------------------------------------------------
    # Joint helpers
    # ------------------------------------------------------------------

    def _joint_qpos(self, joint_name: str) -> float:
        jid = self.name2jid[joint_name]
        qadr = self.model.jnt_qposadr[jid]
        return float(self.sim.data.qpos[qadr])

    def _joint_qvel(self, joint_name: str) -> float:
        jid = self.name2jid[joint_name]
        dadr = self.model.jnt_dofadr[jid]
        return float(self.sim.data.qvel[dadr])

    def get_root_pitch(self) -> float:
        return self._joint_qpos("root_pitch")

    def get_root_pitch_relative_to_initial(self) -> float:
        pitch_now = self.get_root_pitch()

        if self.initial_root_pitch is None:
            self.initial_root_pitch = pitch_now

        return float(pitch_now - self.initial_root_pitch)

    def get_joint_angle_vel(self, jid: int) -> Tuple[float, float]:
        qpos_idx = self.model.jnt_qposadr[jid]
        qvel_idx = self.model.jnt_dofadr[jid]
        return float(self.sim.data.qpos[qpos_idx]), float(self.sim.data.qvel[qvel_idx])

    # ------------------------------------------------------------------
    # Body helpers
    # ------------------------------------------------------------------
    
    def apply_phase4_hip_forward_push(self) -> None:
        """
        Push pelvis/root forward during late Phase 3 and Phase 4.

        The important posture metric is pelvis relative to the feet/support base.
        If pelvis is behind the feet, the model remains a diagonal plank.
        """
        phase = int(self.phase_state.phase)

        if phase not in (3, 4):
            return

        if not hasattr(self.sim.data, "qfrc_applied"):
            return

        p = self.params
        obs = self.obs

        pelvis_to_feet_x = float(obs.get("pelvis_to_feet_x", float("nan")))
        root_x_vel = self._joint_qvel("root_x")

        if not np.isfinite(pelvis_to_feet_x):
            return

        # Target: pelvis should move closer to/over the support base.
        # Tune this from logs. Start modestly.
        if phase == 3:
            target = -0.5
            kp = 620.0
            kd = 70.0
            limit = 360.0
        else:
            target = 0.02
            kp = 260.0
            kd = 40.0
            limit = 180.0

        forward_error = target - pelvis_to_feet_x

        # Only push forward. Do not pull backward.
        if forward_error <= 0.0:
            return

        jid = self.name2jid["root_x"]
        dadr = self.model.jnt_dofadr[jid]

        tau = kp * forward_error - kd * root_x_vel
        tau = float(np.clip(tau, 0.0, limit))

        # Positive should push root/pelvis forward.
        # If the model moves backward, flip this sign.
        self.sim.data.qfrc_applied[dadr] += tau
    
    def apply_root_pitch_forward_brake(self, phase: int) -> None:
        """
        Prevents root/pelvis pitch and torso flexion from running away forward.

        Phase 3 special case:
        - root_pitch brake keeps the whole body from pitching forward;
        - flex_extension brake keeps the visible back/torso from folding.
        """
        if not self.params.brake_root_pitch_after_p1:
            return

        if not hasattr(self.sim.data, "qfrc_applied"):
            return

        if int(phase) not in (2, 3, 4):
            return

        p = self.params

        # ------------------------------------------------------------------
        # Root pitch brake
        # ------------------------------------------------------------------
        if int(phase) == 4:
            pitch_limit = p.p4_root_pitch_max_forward
            kp = p.p4_root_pitch_brake_kp
            kd = p.p4_root_pitch_brake_kd
            tau_limit = p.p4_root_pitch_brake_tau_limit

        elif int(phase) == 3:
            # Phase 3: resist forward body pitch, but do not yank it forward
            # with negative torque when velocity changes sign.
            pitch_limit = -0.20
            kp = 500.0
            kd = 22.0
            tau_limit = 380.0

        else:
            pitch_limit = p.p23_root_pitch_max_forward
            kp = p.p23_root_pitch_brake_kp
            kd = p.p23_root_pitch_brake_kd
            tau_limit = p.p23_root_pitch_brake_tau_limit

        jid = self.name2jid["root_pitch"]
        qadr = self.model.jnt_qposadr[jid]
        dadr = self.model.jnt_dofadr[jid]

        q = float(self.sim.data.qpos[qadr])
        qd = float(self.sim.data.qvel[dadr])

        if q < pitch_limit:
            tau = kp * (pitch_limit - q) - kd * qd

            if int(phase) == 3:
                # Important: Phase 3 should only push the torso/root backward/upright.
                # Do not allow the damping term to create forward torque.
                tau = float(np.clip(tau, 0.0, tau_limit))
            else:
                tau = float(np.clip(tau, -tau_limit, tau_limit))

            self.sim.data.qfrc_applied[dadr] += tau

            if self.debug and int(phase) == 3:
                print(
                    f"[P3 ROOT BRAKE] q={q:.3f} qd={qd:.3f} "
                    f"limit={pitch_limit:.3f} tau={tau:.3f}"
                )

        # ------------------------------------------------------------------
        # Phase 3 visible spine/back straightening
        # ------------------------------------------------------------------
        if int(phase) == 3:
            jid = self.name2jid["flex_extension"]
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]

            q = float(self.sim.data.qpos[qadr])
            qd = float(self.sim.data.qvel[dadr])

            # Start with this target. If the back still bends forward, make this
            # slightly more negative. If it arches backward, make it closer to 0.
            torso_target = -0.1

            torso_kp = 260.0
            torso_kd = 28.0
            torso_tau_limit = 220.0

            tau = torso_kp * (torso_target - q) - torso_kd * qd
            tau = float(np.clip(tau, -torso_tau_limit, torso_tau_limit))

            self.sim.data.qfrc_applied[dadr] += tau

            if self.debug:
                print(
                    f"[P3 TORSO JOINT] flex_extension={q:.3f} "
                    f"qd={qd:.3f} target={torso_target:.3f} tau={tau:.3f}"
                )

    def _body_id(self, body_name: str) -> Optional[int]:
        try:
            return self.model.body(body_name).id
        except Exception:
            return None

    def _body_pos(self, body_name: str) -> Optional[np.ndarray]:
        bid = self._body_id(body_name)
        if bid is None:
            return None
        return self.sim.data.xpos[bid].copy()

    def _body_pos_required(self, body_name: str) -> np.ndarray:
        pos = self._body_pos(body_name)
        if pos is None:
            raise KeyError(f"Could not find body '{body_name}'.")
        return pos

    def get_trunk_lean_from_positions_raw(self) -> float:
        torso_pos = self._body_pos("torso")
        pelvis_pos = self._body_pos("pelvis")

        if torso_pos is None or pelvis_pos is None:
            return 0.0

        dx = pelvis_pos[0] - torso_pos[0]
        dz = torso_pos[2] - pelvis_pos[2]
        dz_safe = dz if abs(dz) > 1e-6 else np.sign(dz if dz != 0 else 1.0) * 1e-6
        return float(np.arctan2(dx, dz_safe))

    def get_trunk_lean_relative_to_initial(self) -> float:
        lean_raw = self.get_trunk_lean_from_positions_raw()

        if self.initial_trunk_lean is None:
            self.initial_trunk_lean = lean_raw

        return float(lean_raw - self.initial_trunk_lean)

    def get_xz_kinematics(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torso_pos = self._body_pos("torso")
        pelvis_pos = self._body_pos("pelvis")

        if torso_pos is None or pelvis_pos is None:
            xz_pos = np.zeros(4, dtype=float)
        else:
            xz_pos = np.array(
                [torso_pos[0], torso_pos[2], pelvis_pos[0], pelvis_pos[2]],
                dtype=float,
            )

        if self.prev_xz_pos is None or dt <= 1e-8:
            xz_vel = np.zeros_like(xz_pos)
        else:
            xz_vel = (xz_pos - self.prev_xz_pos) / dt

        xz_acc = self.acc_est_xz.update(xz_vel, dt)
        self.prev_xz_pos = xz_pos.copy()

        return xz_pos, xz_vel, xz_acc

    def _mean_existing_body_x(self, names: Iterable[str]) -> Optional[float]:
        xs = []
        for name in names:
            pos = self._body_pos(name)
            if pos is not None:
                xs.append(float(pos[0]))

        return float(np.mean(xs)) if xs else None

    def _feet_support_x(self) -> Optional[float]:
        return self._mean_existing_body_x(
            [
                "calcn_r", "calcn_l",
                "talus_r", "talus_l",
                "toes_r", "toes_l",
                "r_foot", "l_foot",
                "r_bofoot", "l_bofoot",
                "foot_r", "foot_l",
            ]
        )

    def _foot_positions(self) -> Tuple[float, float]:
        names = [
            "calcn_r", "calcn_l",
            "talus_r", "talus_l",
            "toes_r", "toes_l",
            "r_foot", "l_foot",
            "r_bofoot", "l_bofoot",
            "foot_r", "foot_l",
        ]

        xs, zs = [], []
        for name in names:
            pos = self._body_pos(name)
            if pos is not None:
                xs.append(float(pos[0]))
                zs.append(float(pos[2]))

        return (
            float(np.mean(xs)) if xs else float("nan"),
            float(np.mean(zs)) if zs else float("nan"),
        )

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def _geom_ids(self, names: Iterable[str]) -> List[int]:
        out = []
        for name in names:
            try:
                out.append(self.model.geom(name).id)
            except Exception:
                pass
        return out

    def _floor_geom_ids(self) -> List[int]:
        names = ["floor", "ground", "terrain", "plane"]
        ids = self._geom_ids(names)
        if ids:
            return ids

        out = []
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if name is not None and any(k in name.lower() for k in ["floor", "ground", "terrain", "plane"]):
                out.append(gid)
        return out

    def check_geom_contact(self, geom_ids_a: Iterable[int], geom_ids_b: Iterable[int]) -> bool:
        a, b = set(geom_ids_a), set(geom_ids_b)
        if not a or not b:
            return False

        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in a and g2 in b) or (g2 in a and g1 in b):
                return True
        return False

    def _sensor_value_safe(self, sensor_name: str) -> Optional[float]:
        try:
            return float(self.sim.data.sensor(sensor_name).data[0])
        except Exception:
            return None

    def _body_weight(self) -> float:
        return float(np.sum(self.model.body_mass) * 9.81)

    def _foot_load_from_sensors(self, side: str) -> Optional[float]:
        names = [f"{side}_foot", f"{side}_toes"]
        vals = []
        for name in names:
            v = self._sensor_value_safe(name)
            if v is not None:
                vals.append(v)

        if not vals:
            return None

        return float(np.sum(vals) / max(self._body_weight(), 1e-8))

    def _site_z_safe(self, site_name: str) -> Optional[float]:
        try:
            return float(self.sim.data.site(site_name).xpos[2])
        except Exception:
            return None

    def _min_foot_site_z(self, side: str) -> float:
        site_names = [
            f"{side}_heel_btm",
            f"{side}_toe_btm",
            f"{side}_foot_touch",
            f"{side}_toes_touch",
        ]

        zs = []
        for name in site_names:
            z = self._site_z_safe(name)
            if z is not None:
                zs.append(z)

        return float(np.min(zs)) if zs else float("nan")

    def _floor_height(self) -> float:
        floor_ids = self._floor_geom_ids()
        zs = []
        for gid in floor_ids:
            try:
                zs.append(float(self.sim.data.geom_xpos[gid][2]))
            except Exception:
                pass
        return float(np.max(zs)) if zs else 0.0

    def _foot_clearance_side(self, side: str) -> float:
        foot_z = self._min_foot_site_z(side)
        if np.isfinite(foot_z):
            return float(foot_z - self._floor_height())

        if side == "l":
            geom_names = [
                "l_foot_col1", "l_foot_col2", "l_foot_col3", "l_foot_col4",
                "l_bofoot_col1", "l_bofoot_col2",
            ]
        else:
            geom_names = [
                "r_foot_col1", "r_foot_col2", "r_foot_col3", "r_foot_col4",
                "r_bofoot_col1", "r_bofoot_col2",
            ]

        clearances = []
        for name in geom_names:
            try:
                gid = self.model.geom(name).id
                geom_z = float(self.sim.data.geom_xpos[gid][2])
                geom_radius = float(np.max(self.model.geom_size[gid]))
                clearances.append(geom_z - geom_radius - self._floor_height())
            except Exception:
                pass

        return float(np.min(clearances)) if clearances else float("nan")

    def _foot_floor_geom_contact_side(self, side: str) -> bool:
        if side == "l":
            foot_geoms = self._geom_ids([
                "l_foot_col1", "l_foot_col2", "l_foot_col3", "l_foot_col4",
                "l_bofoot_col1", "l_bofoot_col2",
            ])
        else:
            foot_geoms = self._geom_ids([
                "r_foot_col1", "r_foot_col2", "r_foot_col3", "r_foot_col4",
                "r_bofoot_col1", "r_bofoot_col2",
            ])

        floor_geoms = self._floor_geom_ids()
        return self.check_geom_contact(foot_geoms, floor_geoms)
    def _foot_geom_bottom_z_side(self, side: str) -> float:
        if side == "l":
            geom_names = [
                "l_foot_col1", "l_foot_col2", "l_foot_col3", "l_foot_col4",
                "l_bofoot_col1", "l_bofoot_col2",
            ]
        else:
            geom_names = [
                "r_foot_col1", "r_foot_col2", "r_foot_col3", "r_foot_col4",
                "r_bofoot_col1", "r_bofoot_col2",
            ]

        bottoms = []
        for name in geom_names:
            try:
                gid = self.model.geom(name).id
                geom_z = float(self.sim.data.geom_xpos[gid][2])
                size = np.asarray(self.model.geom_size[gid], dtype=float)
                radius = float(np.max(size))
                bottoms.append(geom_z - radius)
            except Exception:
                pass

        return float(np.min(bottoms)) if bottoms else float("nan")

    def _actual_foot_geom_clearance_side(self, side: str) -> float:
        bottom_z = self._foot_geom_bottom_z_side(side)
        if not np.isfinite(bottom_z):
            return float("nan")
        return float(bottom_z - self._floor_height())
    def debug_contacts(self) -> None:
        print("\n[CONTACT DEBUG]")
        print(f"ncon = {self.sim.data.ncon}")

        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            name1 = self.model.geom(g1).name
            name2 = self.model.geom(g2).name
            print(f"  contact {i}: {g1}={name1} <-> {g2}={name2}")

    def get_contacts(self) -> None:
        seat_geoms = self._geom_ids(["seat", "back", "chair", "chair_seat"])

        pelvis_thigh_geoms = self._geom_ids(
            [
                "l_pelvis", "l_pelvis_col",
                "r_pelvis", "r_pelvis_col",
                "sacrum", "sacrum_geom_1",
                "l_femur1_col", "l_femur2_col",
                "r_femur1_col", "r_femur2_col",
            ]
        )

        self.obs["seat_contact"] = bool(
            self.check_geom_contact(pelvis_thigh_geoms, seat_geoms)
        )

        l_load = self._foot_load_from_sensors("l")
        r_load = self._foot_load_from_sensors("r")

        self.obs["left_foot_load"] = float(l_load) if l_load is not None else float("nan")
        self.obs["right_foot_load"] = float(r_load) if r_load is not None else float("nan")

        self.obs["left_foot_clearance"] = self._foot_clearance_side("l")
        self.obs["right_foot_clearance"] = self._foot_clearance_side("r")

        self.obs["left_foot_geom_contact"] = self._foot_floor_geom_contact_side("l")
        self.obs["right_foot_geom_contact"] = self._foot_floor_geom_contact_side("r")

        # Primary truth source: load-based foot contact.
        # Keep clearance in debug only because site/floor references can be inconsistent.
        
        self.obs["left_geom_bottom_z"] = self._foot_geom_bottom_z_side("l")
        self.obs["right_geom_bottom_z"] = self._foot_geom_bottom_z_side("r")
        self.obs["left_actual_geom_clearance"] = self._actual_foot_geom_clearance_side("l")
        self.obs["right_actual_geom_clearance"] = self._actual_foot_geom_clearance_side("r")
        l_geom_close = (
            np.isfinite(self.obs["left_actual_geom_clearance"])
            and self.obs["left_actual_geom_clearance"] < 0.015
        )
        r_geom_close = (
            np.isfinite(self.obs["right_actual_geom_clearance"])
            and self.obs["right_actual_geom_clearance"] < 0.015
        )

                # Temporary practical contact logic:
        # Use load OR actual MuJoCo geom contact for phase progression.
        # Do not use actual_geom_clearance yet because its floor reference is currently wrong.
        if l_load is not None and r_load is not None:
            self.obs["left_foot_contact"] = bool(
                l_load > self.params.foot_contact_load_threshold
                or self.obs["left_foot_geom_contact"]
            )
            self.obs["right_foot_contact"] = bool(
                r_load > self.params.foot_contact_load_threshold
                or self.obs["right_foot_geom_contact"]
            )
        else:
            self.obs["left_foot_contact"] = bool(self.obs["left_foot_geom_contact"])
            self.obs["right_foot_contact"] = bool(self.obs["right_foot_geom_contact"])

        self.obs["grounded"] = bool(
            self.obs["left_foot_contact"] and self.obs["right_foot_contact"]
        )


    def _foot_clearance_from_ground(self) -> float:
        vals = [self._foot_clearance_side("l"), self._foot_clearance_side("r")]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.min(vals)) if vals else float("nan")

    # ------------------------------------------------------------------
    # Foot anchoring
    # ------------------------------------------------------------------
    
    def _foot_contact_body_candidates(self, side: str) -> List[str]:
        return [
            f"calcn_{side}",
            f"talus_{side}",
            f"toes_{side}",
            f"{side}_foot",
            f"{side}_bofoot",
            f"foot_{side}",
        ]

    def _best_foot_anchor_body(self, side: str) -> Optional[str]:
        candidates = [
            f"calcn_{side}",   # heel/back-foot first
            f"{side}_foot",
            f"{side}_bofoot",
            f"foot_{side}",
            f"talus_{side}",   # talus last
        ]
        for name in candidates:
            if self._body_id(name) is not None:
                return name

        return None
    
    def capture_heel_anchors(self) -> None:
        self.heel_anchor_z = {}

        for side in ["l", "r"]:
            body_name = f"calcn_{side}"
            pos = self._body_pos(body_name)

            if pos is None:
                if self.debug:
                    print(f"[HEEL ANCHOR] No calcn body found for {side}")
                continue

            self.heel_anchor_z[side] = float(pos[2])

            if self.debug:
                print(
                    f"[HEEL ANCHOR] Captured {side} heel z from {body_name}: "
                    f"{pos[2]:.4f}"
                )
    def apply_heel_down_pd(self, phase: int) -> None:
        """
        Prevents the heel/calcn from lifting during Phase 1/2.
        This specifically targets the tippy-toe issue.
        """
        if int(phase) not in (1, 2):
            return

        if not hasattr(self.sim.data, "xfrc_applied"):
            return

        kp = 3500.0
        kd = 180.0
        force_limit = 650.0

        for side in ["l", "r"]:
            if side not in self.heel_anchor_z:
                continue

            body_name = f"calcn_{side}"
            bid = self._body_id(body_name)
            pos = self._body_pos(body_name)

            if bid is None or pos is None:
                continue

            current_z = float(pos[2])
            target_z = float(self.heel_anchor_z[side])

            try:
                vz = float(self.sim.data.cvel[bid][5])
            except Exception:
                vz = 0.0

            # Positive error means heel is below target.
            # Negative error means heel has lifted above target.
            err_z = target_z - current_z

            force_z = kp * err_z - kd * vz
            force_z = float(np.clip(force_z, -force_limit, force_limit))

            # Only push down. Never pull heel upward.
            force_z = min(force_z, 0.0)

            self.sim.data.xfrc_applied[bid, 2] += force_z

            if self.debug and abs(err_z) > 0.015:
                print(
                    f"[HEEL PD] phase={phase} side={side} "
                    f"target_z={target_z:.4f} current_z={current_z:.4f} "
                    f"err={err_z:.4f} Fz={force_z:.1f}"
                )
    def capture_foot_anchors(self) -> None:
        self.foot_anchor_pos = {}
        self.prev_foot_anchor_pos = {}

        for side in ["l", "r"]:
            body_names = [
                f"calcn_{side}",
                f"talus_{side}",
                f"toes_{side}",
            ]

            self.foot_anchor_pos[side] = {}
            self.prev_foot_anchor_pos[side] = {}

            for body_name in body_names:
                if self._body_id(body_name) is None:
                    continue

                pos = self._body_pos_required(body_name)
                self.foot_anchor_pos[side][body_name] = pos.copy()
                self.prev_foot_anchor_pos[side][body_name] = pos.copy()

                if self.debug:
                    print(
                        f"[FOOT ANCHOR] Captured {side}:{body_name}: "
                        f"x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}"
                    )

            if not self.foot_anchor_pos[side] and self.debug:
                print(f"[FOOT ANCHOR] No anchor bodies found for side '{side}'.")

    def apply_foot_anchor_pd(self, dt: float) -> None:
        if not self.foot_anchor_pos:
            return

        p = self.params
        
        phase = int(self.phase_state.phase)

        anchor_kp = p.foot_anchor_kp
        anchor_kd = p.foot_anchor_kd
        anchor_limit = p.foot_anchor_force_limit
        
        if phase == 2:
            anchor_kp *= 0.45
            anchor_kd *= 0.55
            anchor_limit *= 0.55

        elif phase == 4:
            pelvis_high = self.obs.get("pelvis_y", 0.0) >= p.p4_release_anchor_height
            pelvis_not_falling = self.obs.get("pelvis_y_vel", -999.0) > p.p4_release_anchor_vz_min
            torso_not_falling = self.obs.get("torso_y_vel", -999.0) > -0.03
            seat_clear = not bool(self.obs.get("seat_contact", False))

            if pelvis_high and pelvis_not_falling and torso_not_falling and seat_clear:
                anchor_kp *= 0.15
                anchor_kd *= 0.25
                anchor_limit *= 0.30
            else:
                anchor_kp *= 0.35
                anchor_kd *= 0.45
                anchor_limit *= 0.50
        elif phase == 3:
            # Phase 3: keep the feet from drifting/lifting while still allowing rise.
            # Your current anchor gets too weak and the feet slide/pivot.
            anchor_kp = 0.35 * p.foot_anchor_kp
            anchor_kd = 0.35 * p.foot_anchor_kd
            anchor_limit = 0.40 * p.foot_anchor_force_limit
            l_load = float(self.obs.get("left_foot_load", 0.0))
            r_load = float(self.obs.get("right_foot_load", 0.0))

            # If either foot is becoming unloaded, strengthen vertical anchoring.
            # This prevents one leg from becoming effectively airborne while the other
            # continues to support the body.

        else:
            anchor_kp = p.foot_anchor_kp
            anchor_kd = p.foot_anchor_kd

        if not hasattr(self.sim.data, "xfrc_applied"):
            if self.debug:
                print("[FOOT ANCHOR] sim.data.xfrc_applied unavailable; cannot apply body anchor.")
            return

        for side in ["l", "r"]:
            if side not in self.foot_anchor_pos:
                continue

            anchors = self.foot_anchor_pos[side]

            # New multi-anchor format.
            if isinstance(anchors, dict):
                items = anchors.items()
            else:
                body_name = self._best_foot_anchor_body(side)
                if body_name is None:
                    continue
                items = [(body_name, anchors)]

            for body_name, target in items:
                bid = self._body_id(body_name)
                if bid is None:
                    continue

                pos = self._body_pos_required(body_name)

                if isinstance(self.prev_foot_anchor_pos.get(side, None), dict):
                    prev = self.prev_foot_anchor_pos[side].get(body_name, pos.copy())
                else:
                    prev = pos.copy()

                if dt <= 1e-8:
                    vel = np.zeros(3, dtype=float)
                else:
                    vel = (pos - prev) / dt

                err = np.asarray(target) - pos

                side_load = float(
                    self.obs.get(
                        "left_foot_load" if side == "l" else "right_foot_load",
                        0.0,
                    )
                )

                side_anchor_kp = anchor_kp
                side_anchor_kd = anchor_kd
                side_anchor_limit = anchor_limit

                if phase == 3 and side_load < 0.18:
                    side_anchor_kp *= 3.0
                    side_anchor_kd *= 1.8
                    side_anchor_limit *= 2.5

                force = np.zeros(3, dtype=float)

                if phase == 1:
                    # Phase 1: hold full foot pose, including x and z.
                    force[0] = side_anchor_kp * err[0] - side_anchor_kd * vel[0]
                    force[2] = side_anchor_kp * err[2] - side_anchor_kd * vel[2]
                elif p.foot_anchor_vertical_only or phase == 3:
                    force[2] = side_anchor_kp * err[2] - side_anchor_kd * vel[2]
                else:
                    force[0] = side_anchor_kp * err[0] - side_anchor_kd * vel[0]
                    force[2] = side_anchor_kp * err[2] - side_anchor_kd * vel[2]

                force = np.clip(force, -side_anchor_limit, side_anchor_limit)
                self.sim.data.xfrc_applied[bid, 0:3] += force

                # Save previous body position.
                if isinstance(self.prev_foot_anchor_pos.get(side, None), dict):
                    self.prev_foot_anchor_pos[side][body_name] = pos.copy()

    def _anchor_error(self, side: str) -> float:
        """
        Works with either:
        old format: self.foot_anchor_pos[side] = np.array([x,y,z])
        new format: self.foot_anchor_pos[side] = {"calcn_l": pos, "talus_l": pos, ...}
        """
        if side not in self.foot_anchor_pos:
            return float("nan")

        anchors = self.foot_anchor_pos[side]

        # New multi-body anchor format.
        if isinstance(anchors, dict):
            errs = []
            for body_name, target in anchors.items():
                pos = self._body_pos(body_name)
                if pos is not None:
                    errs.append(float(np.linalg.norm(np.asarray(target) - pos)))
            return float(np.mean(errs)) if errs else float("nan")

        # Old single-body format.
        body_name = self._best_foot_anchor_body(side)
        if body_name is None:
            return float("nan")

        pos = self._body_pos(body_name)
        if pos is None:
            return float("nan")

        return float(np.linalg.norm(np.asarray(anchors) - pos))

    # ------------------------------------------------------------------
    # Initial pose / Phase 1 hold helpers
    # ------------------------------------------------------------------
    
    def _foot_flatness_debug_side(self, side: str) -> Dict[str, float]:
        """
        Debug whether the foot is flat or pivoting onto toes.

        Measures heel/talus/toe positions separately. The exact body names may vary,
        so this tries common MyoSuite names.
        """
        if side == "l":
            heel_names = ["calcn_l", "l_heel", "heel_l"]
            toe_names = ["toes_l", "l_toes", "toe_l", "l_bofoot"]
            talus_names = ["talus_l"]
        else:
            heel_names = ["calcn_r", "r_heel", "heel_r"]
            toe_names = ["toes_r", "r_toes", "toe_r", "r_bofoot"]
            talus_names = ["talus_r"]

        def first_pos(names):
            for name in names:
                pos = self._body_pos(name)
                if pos is not None:
                    return name, pos
            return None, None

        heel_name, heel = first_pos(heel_names)
        toe_name, toe = first_pos(toe_names)
        talus_name, talus = first_pos(talus_names)

        out = {
            f"{side}_heel_name": heel_name,
            f"{side}_toe_name": toe_name,
            f"{side}_talus_name": talus_name,
            f"{side}_heel_z": float("nan"),
            f"{side}_toe_z": float("nan"),
            f"{side}_talus_z": float("nan"),
            f"{side}_heel_x": float("nan"),
            f"{side}_toe_x": float("nan"),
            f"{side}_heel_minus_toe_z": float("nan"),
            f"{side}_talus_minus_toe_z": float("nan"),
            f"{side}_foot_pitch_proxy": float("nan"),
        }

        if heel is not None:
            out[f"{side}_heel_z"] = float(heel[2])
            out[f"{side}_heel_x"] = float(heel[0])

        if toe is not None:
            out[f"{side}_toe_z"] = float(toe[2])
            out[f"{side}_toe_x"] = float(toe[0])

        if talus is not None:
            out[f"{side}_talus_z"] = float(talus[2])

        if heel is not None and toe is not None:
            dx = float(toe[0] - heel[0])
            dz = float(toe[2] - heel[2])
            out[f"{side}_heel_minus_toe_z"] = float(heel[2] - toe[2])
            out[f"{side}_foot_pitch_proxy"] = float(np.arctan2(dz, abs(dx) + 1e-6))

        if talus is not None and toe is not None:
            out[f"{side}_talus_minus_toe_z"] = float(talus[2] - toe[2])

        return out

    def reset_filters(self) -> None:
        self.acc_est.reset()
        self.acc_est_xz.reset()

        self.prev_time = None
        self.prev_xz_pos = None

        self.initial_root_pitch = None
        self.prev_root_pitch_rel = None
        self.filtered_root_pitch_rel_vel = 0.0

        self.initial_trunk_lean = None
        self.prev_trunk_lean_rel = None
        self.filtered_trunk_lean_vel = 0.0

        self.foot_anchor_pos = {}
        self.prev_foot_anchor_pos = {}

    def capture_phase1_hold_pose(self) -> None:
        self.p1_hold_qpos = self.sim.data.qpos.copy()
        self.p1_hold_root = {
            "root_x": self._joint_qpos("root_x"),
            "root_z": self._joint_qpos("root_z"),
            "root_pitch": self._joint_qpos("root_pitch"),
        }

        if self.debug:
            print(
                "[P1 HOLD] Captured seated lower-limb/root pose: "
                f"root_x={self.p1_hold_root['root_x']:.4f}, "
                f"root_z={self.p1_hold_root['root_z']:.4f}, "
                f"root_pitch={self.p1_hold_root['root_pitch']:.4f}"
            )

    def apply_phase1_root_pd_hold(self) -> None:
        if self.p1_hold_root is None:
            return

        p = self.params
        hold_specs = {
            "root_x": (
                self.p1_hold_root["root_x"],
                p.p1_root_x_hold_kp,
                p.p1_root_x_hold_kd,
            ),
            "root_z": (
                self.p1_hold_root["root_z"],
                p.p1_root_z_hold_kp,
                p.p1_root_z_hold_kd,
            ),
            "root_pitch": (
                self.p1_hold_root["root_pitch"],
                p.p1_root_pitch_hold_kp,
                p.p1_root_pitch_hold_kd,
            ),
        }

        for joint_name, (q0, kp, kd) in hold_specs.items():
            jid = self.name2jid[joint_name]
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]

            q = float(self.sim.data.qpos[qadr])
            qd = float(self.sim.data.qvel[dadr])

            tau = kp * (q0 - q) - kd * qd
            tau = float(np.clip(tau, -p.p1_root_tau_limit, p.p1_root_tau_limit))
            self.sim.data.qfrc_applied[dadr] += tau
            
    def phase1_lower_limb_still(self, obs: Dict[str, float]) -> bool:
        """
        Returns True only if hip/knee/ankle joints have stayed close to the
        captured seated pose during Phase 1.
        """
        if self.p1_hold_qpos is None:
            return True

        p = self.params

        joint_names = [
            "hip_flexion_r", "hip_flexion_l",
            "knee_angle_r", "knee_angle_l",
            "ankle_angle_r", "ankle_angle_l",
        ]

        max_q_err = 0.0
        max_qd = 0.0

        for name in joint_names:
            jid = self.name2jid[name]
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]

            q = float(self.sim.data.qpos[qadr])
            q0 = float(self.p1_hold_qpos[qadr])
            qd = float(self.sim.data.qvel[dadr])

            max_q_err = max(max_q_err, abs(q - q0))
            max_qd = max(max_qd, abs(qd))

        obs["p1_max_leg_q_err"] = float(max_q_err)
        obs["p1_max_leg_qd"] = float(max_qd)

        return max_q_err <= p.p1_leg_q_tol and max_qd <= p.p1_leg_qd_tol

    def apply_phase1_leg_pd_hold(self) -> None:
        """
        Phase 1: keep knees from drifting upward and keep ankles reasonably stable,
        but DO NOT lock hip flexion. The forward lean should come mainly from the hip
        hinge, not from excessive torso/spine folding.
        """
        if self.p1_hold_qpos is None:
            return

        if not hasattr(self.sim.data, "qfrc_applied"):
            return

        hold_specs = {
            "knee_angle_r": (420.0, 45.0, 220.0),
            "knee_angle_l": (420.0, 45.0, 220.0),
            "ankle_angle_r": (140.0, 16.0, 80.0),
            "ankle_angle_l": (140.0, 16.0, 80.0),
        }

        for name, (kp, kd, tau_limit) in hold_specs.items():
            jid = self.name2jid[name]
            qadr = self.model.jnt_qposadr[jid]
            dadr = self.model.jnt_dofadr[jid]

            q = float(self.sim.data.qpos[qadr])
            q0 = float(self.p1_hold_qpos[qadr])
            qd = float(self.sim.data.qvel[dadr])

            tau = kp * (q0 - q) - kd * qd
            tau = float(np.clip(tau, -tau_limit, tau_limit))
            self.sim.data.qfrc_applied[dadr] += tau

            if self.debug and "knee" in name:
                print(
                    f"[P1 HOLD] {name} q={q:.4f} q0={q0:.4f} "
                    f"err={q0-q:.4f} qd={qd:.4f} tau={tau:.2f}"
                )
    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> Dict[str, float]:
        current_time = float(self.sim.data.time)
        dt = self.model.opt.timestep if self.prev_time is None else current_time - self.prev_time

        if dt <= 1e-8:
            dt = float(self.model.opt.timestep)

        self.prev_time = current_time

        joint_pos, joint_vel = [], []
        for jid in self.jids_of_interest:
            pos, vel = self.get_joint_angle_vel(jid)
            joint_pos.append(pos)
            joint_vel.append(vel)

        joint_pos = np.asarray(joint_pos, dtype=float)
        joint_vel = np.asarray(joint_vel, dtype=float)
        joint_acc = self.acc_est.update(joint_vel, dt)

        torso_pos, torso_vel, torso_acc = joint_pos[0], joint_vel[0], joint_acc[0]

        hip_r, hip_l = joint_pos[1], joint_pos[2]
        knee_r, knee_l = joint_pos[3], joint_pos[4]
        ankle_r, ankle_l = joint_pos[5], joint_pos[6]

        hip_r_vel, hip_l_vel = joint_vel[1], joint_vel[2]
        knee_r_vel, knee_l_vel = joint_vel[3], joint_vel[4]
        ankle_r_vel, ankle_l_vel = joint_vel[5], joint_vel[6]

        xz_pos, xz_vel, xz_acc = self.get_xz_kinematics(dt)

        torso_x, torso_z, pelvis_x, pelvis_z = xz_pos
        torso_x_vel, torso_z_vel, pelvis_x_vel, pelvis_z_vel = xz_vel
        torso_x_acc, torso_z_acc, pelvis_x_acc, pelvis_z_acc = xz_acc

        trunk_lean_rel = self.get_trunk_lean_relative_to_initial()

        if self.prev_trunk_lean_rel is None:
            trunk_lean_vel_raw = 0.0
        else:
            trunk_lean_vel_raw = (trunk_lean_rel - self.prev_trunk_lean_rel) / dt

        self.prev_trunk_lean_rel = trunk_lean_rel
        self.filtered_trunk_lean_vel = (
            0.12 * trunk_lean_vel_raw
            + 0.88 * self.filtered_trunk_lean_vel
        )

        root_pitch_rel = self.get_root_pitch_relative_to_initial()

        if self.prev_root_pitch_rel is None:
            root_pitch_rel_vel_raw = 0.0
        else:
            root_pitch_rel_vel_raw = (root_pitch_rel - self.prev_root_pitch_rel) / dt

        self.prev_root_pitch_rel = root_pitch_rel
        self.filtered_root_pitch_rel_vel = (
            0.12 * root_pitch_rel_vel_raw
            + 0.88 * self.filtered_root_pitch_rel_vel
        )

        root_x = self._joint_qpos("root_x")
        root_z = self._joint_qpos("root_z")
        root_pitch = self._joint_qpos("root_pitch")

        feet_x = self._feet_support_x()
        pelvis_to_feet_x = (
            float(pelvis_x - feet_x)
            if feet_x is not None and np.isfinite(feet_x)
            else float("nan")
        )
        foot_x, foot_z = self._foot_positions()
        l_flat = self._foot_flatness_debug_side("l")
        r_flat = self._foot_flatness_debug_side("r")
        

        forward_metric = float(pelvis_x - torso_x)

        self.obs = {
            "time": current_time,
            "dt": float(dt),

            "torso_pos": float(torso_pos),
            "torso_vel": float(torso_vel),
            "torso_acc": float(torso_acc),

            "hip_r": float(hip_r),
            "hip_l": float(hip_l),
            "hip_avg": float((hip_r + hip_l) / 2.0),
            "hip_r_vel": float(hip_r_vel),
            "hip_l_vel": float(hip_l_vel),
            "hip_avg_vel": float((hip_r_vel + hip_l_vel) / 2.0),

            "knee_r": float(knee_r),
            "knee_l": float(knee_l),
            "knee_avg": float((knee_r + knee_l) / 2.0),
            "knee_r_vel": float(knee_r_vel),
            "knee_l_vel": float(knee_l_vel),
            "knee_avg_vel": float((knee_r_vel + knee_l_vel) / 2.0),

            "ankle_r": float(ankle_r),
            "ankle_l": float(ankle_l),
            "ankle_avg": float((ankle_r + ankle_l) / 2.0),
            "ankle_r_vel": float(ankle_r_vel),
            "ankle_l_vel": float(ankle_l_vel),
            "ankle_avg_vel": float((ankle_r_vel + ankle_l_vel) / 2.0),
            "l_heel_z": float(l_flat["l_heel_z"]),
            "l_toe_z": float(l_flat["l_toe_z"]),
            "l_talus_z": float(l_flat["l_talus_z"]),
            "l_heel_minus_toe_z": float(l_flat["l_heel_minus_toe_z"]),
            "l_talus_minus_toe_z": float(l_flat["l_talus_minus_toe_z"]),
            "l_foot_pitch_proxy": float(l_flat["l_foot_pitch_proxy"]),

            "r_heel_z": float(r_flat["r_heel_z"]),
            "r_toe_z": float(r_flat["r_toe_z"]),
            "r_talus_z": float(r_flat["r_talus_z"]),
            "r_heel_minus_toe_z": float(r_flat["r_heel_minus_toe_z"]),
            "r_talus_minus_toe_z": float(r_flat["r_talus_minus_toe_z"]),
            "r_foot_pitch_proxy": float(r_flat["r_foot_pitch_proxy"]),

            "torso_x": float(torso_x),
            "torso_y": float(torso_z),
            "torso_z": float(torso_z),
            "torso_x_vel": float(torso_x_vel),
            "torso_y_vel": float(torso_z_vel),
            "torso_z_vel": float(torso_z_vel),
            "torso_x_acc": float(torso_x_acc),
            "torso_y_acc": float(torso_z_acc),
            "torso_z_acc": float(torso_z_acc),

            "pelvis_x": float(pelvis_x),
            "pelvis_y": float(pelvis_z),
            "pelvis_z": float(pelvis_z),
            "pelvis_x_vel": float(pelvis_x_vel),
            "pelvis_y_vel": float(pelvis_z_vel),
            "pelvis_z_vel": float(pelvis_z_vel),
            "pelvis_x_acc": float(pelvis_x_acc),
            "pelvis_y_acc": float(pelvis_z_acc),
            "pelvis_z_acc": float(pelvis_z_acc),

            "root_x": float(root_x),
            "root_z": float(root_z),
            "root_pitch": float(root_pitch),
            "root_pitch_rel": float(root_pitch_rel),
            "root_pitch_rel_vel": float(np.clip(self.filtered_root_pitch_rel_vel, -8.0, 8.0)),
            "root_pitch_rel_vel_raw": float(root_pitch_rel_vel_raw),

            "trunk_lean_rel": float(trunk_lean_rel),
            "trunk_lean_vel": float(np.clip(self.filtered_trunk_lean_vel, -8.0, 8.0)),
            "trunk_lean_vel_raw": float(trunk_lean_vel_raw),

            "feet_x": float(feet_x) if feet_x is not None else float("nan"),
            "foot_x": float(foot_x),
            "foot_z": float(foot_z),

            "forward_metric": forward_metric,
            "torso_over_feet": bool(feet_x is not None and torso_x >= feet_x - 0.02),

            "phase": int(self.phase_state.phase),
            "phase_elapsed": float(self.phase_state.phase_elapsed),
            "pelvis_to_feet_x": pelvis_to_feet_x,
            
            
        }
        self.obs["left_heel_toe_dz"] = l_flat["l_heel_z"] - l_flat["l_toe_z"]
        self.obs["right_heel_toe_dz"] = r_flat["r_heel_z"] - r_flat["r_toe_z"]

        self.get_contacts()
        return self.obs

    # ------------------------------------------------------------------
    # Phase logic
    # ------------------------------------------------------------------

    def reset_phase(self, phase: int = 1) -> None:
        now = float(self.sim.data.time)
        self.phase_state = STSPhaseState(
            phase=phase,
            previous_phase=phase,
            phase_start_time=now,
        )
        self.phase = int(phase)

    def _set_phase(self, new_phase: int) -> None:
        if new_phase == self.phase_state.phase:
            self.phase_state.phase_changed = False
            return

        now = float(self.sim.data.time)
        old = self.phase_state.phase

        self.phase_state.previous_phase = old
        self.phase_state.phase = int(new_phase)
        self.phase_state.phase_start_time = now
        self.phase_state.phase_elapsed = 0.0
        self.phase_state.phase_changed = True
        self.phase = int(new_phase)

        if self.debug:
            print(f"[PHASE] {old} -> {new_phase} at t={now:.3f}")

    def update_phase(self, obs: Optional[Dict[str, float]] = None) -> int:
        if obs is None:
            obs = self.obs if self.obs else self.get_observation()

        p = self.params
        st = self.phase_state

        st.phase_elapsed = float(obs.get("time", self.sim.data.time) - st.phase_start_time)
        st.phase_changed = False

        obs["phase_elapsed"] = float(st.phase_elapsed)
        obs["phase"] = int(st.phase)

        feet_grounded = bool(
            obs.get("left_foot_contact", False)
            and obs.get("right_foot_contact", False)
        )

        if st.phase == 1:
            knees_ok = True

            if self.p1_hold_qpos is not None:
                kr0 = float(self.p1_hold_qpos[self.model.jnt_qposadr[self.name2jid["knee_angle_r"]]])
                kl0 = float(self.p1_hold_qpos[self.model.jnt_qposadr[self.name2jid["knee_angle_l"]]])

                knee_r_err = abs(obs["knee_r"] - kr0)
                knee_l_err = abs(obs["knee_l"] - kl0)

                obs["p1_knee_r_err"] = knee_r_err
                obs["p1_knee_l_err"] = knee_l_err

                knees_ok = (knee_r_err < 0.12 and knee_l_err < 0.12)

            lean_reached = obs["trunk_lean_rel"] <= p.p1_lean_target
            # print(obs["trunk_lean_rel"])
            too_much_lean = obs["trunk_lean_rel"] <= p.p1_lean_hard_limit

            ready = lean_reached

            if st.phase_elapsed >= p.p1_min_time and ready:
                st.p1_done = True
                self._set_phase(2)

        elif st.phase == 2:
            # With load-based contact, this should usually be true while anchored.
            # Do not return immediately if feet_grounded is false, because bad contact
            # debugging should not completely freeze the phase machine.
            torso_over_feet = bool(obs.get("torso_over_feet", False))
            torso_not_falling = obs["torso_y_vel"] > -0.015

            pelvis_has_upward_momentum = obs["pelvis_y_vel"] > p.p2_pelvis_rise_vel
            pelvis_has_started_lifting = obs["pelvis_y"] > p.p2_pelvis_lift_height
            hip_still_has_extension_range = obs["hip_avg"] > p.p2_hip_stop

            torso_gap = obs["torso_y"] - obs["pelvis_y"]
            lean_safe_for_p3 = (-obs["trunk_lean_rel"]) < p.p2_to_p3_lean_max
            torso_gap_safe_for_p3 = torso_gap > p.p2_to_p3_min_torso_gap
            torso_not_collapsing_for_p3 = obs["torso_y_vel"] > p.p2_to_p3_min_torso_z_vel
            hip_range_available_for_p3 = obs["hip_avg"] > p.p2_to_p3_min_hip
            
            root_pitch_safe_for_p3 = obs["root_pitch"] > -1.25

            phase2_ready = (
                torso_over_feet
                and torso_not_falling
                and pelvis_has_upward_momentum
                and pelvis_has_started_lifting
                and hip_still_has_extension_range
                and lean_safe_for_p3
                and torso_gap_safe_for_p3
                and torso_not_collapsing_for_p3
                and hip_range_available_for_p3
                and root_pitch_safe_for_p3
            )
                        
            early_p3_due_to_good_lift = (
                st.phase_elapsed >= p.p2_min_time
                and obs["pelvis_y_vel"] > p.p2_to_p3_min_pelvis_vz
                and obs["hip_avg"] > p.p2_to_p3_min_hip
                and obs["trunk_lean_rel"] < 0.72
            )

            hip_is_being_spent = (
                st.phase_elapsed >= p.p2_min_time
                and obs["pelvis_y_vel"] > 0.04
                and obs["hip_avg"] <= 0.25
            )
            
            seat_off_and_lifted = (
                st.phase_elapsed >= p.p2_min_time
                and not obs["seat_contact"]
                and obs["pelvis_y"] > 0.57
                and obs["root_pitch"] > -1.25
            )
            
            #print(early_p3_due_to_good_lift, hip_is_being_spent, st.phase_elapsed >= p.p2_timeout, obs["pelvis_y_vel"] > 0.05)

            real_seat_off = (
                not obs["seat_contact"]
                and obs["grounded"]
                and obs["pelvis_y"] >= 0.66
                and obs["pelvis_y_vel"] > -0.03
            )

            p2_safe_timeout = (
                st.phase_elapsed >= p.p2_timeout
                and obs["grounded"]
                and obs["pelvis_y"] >= 0.68
                and obs["seat_contact"] is False
            )

            if st.phase_elapsed >= p.p2_min_time and (real_seat_off or p2_safe_timeout):
                st.p2_done = True
                self._set_phase(3)

        elif st.phase == 3:
            knee_extended = obs["knee_avg"] <= p.p3_knee_stand
            hip_extended = obs["hip_avg"] <= p.p3_hip_stand

            pelvis_high_enough = obs["pelvis_y"] >= 0.70
            torso_high_enough = obs["torso_y"] >= 0.68

            torso_pelvis_gap = obs["torso_y"] - obs["pelvis_y"]

            height_ok = (
                obs["pelvis_y"] >= 0.80
                and obs["torso_y"] >= 0.80
            )

            pitch_ok = obs["root_pitch"] > -1.3

            # Keep this relaxed. Your lean metric stays high even when the body is high.
            lean_ok = obs["trunk_lean_rel"] < 0.95

            vel_ok = (
                abs(obs["pelvis_y_vel"]) < 0.025
                and abs(obs["torso_y_vel"]) < 0.025
            )

            gap_ok = torso_pelvis_gap > -0.035

            feet_ok = (
                obs["left_foot_contact"]
                and obs["right_foot_contact"]
                and obs["grounded"]
            )
            
            leg_compliance_ok = (
                obs["knee_avg"] > 0.04
                and obs["hip_avg"] > -0.25
            )
            # print(obs["left_foot_contact"],obs["right_foot_contact"],obs["grounded"])

            seat_off = not obs["seat_contact"]
            
            pelvis_forward_enough = (
                np.isfinite(obs.get("pelvis_to_feet_x", float("nan")))
                and obs["pelvis_to_feet_x"] > -0.05
            )

            # Normal successful Phase 3 exit.
            p3_good_posture = (
                st.phase_elapsed > p.p3_min_time
                and seat_off
                and height_ok
                and pitch_ok
                and lean_ok
                and vel_ok
                and gap_ok
                and pelvis_forward_enough
                and leg_compliance_ok
            )

            # Important fallback:
            # If the body is high, seat is off, feet are loaded, and motion has stalled,
            # Phase 3 is finished even if lean is still ugly.
            p3_spent_escape = (
                st.phase_elapsed > 0.70
                and seat_off
                and obs["pelvis_y"] >= 0.80
                and obs["torso_y"] >= 0.78
            )
            # print(p3_good_posture,st.phase_elapsed > p.p3_min_time,seat_off,height_ok,pitch_ok,lean_ok,vel_ok,gap_ok,feet_ok)

            # Safety fallback:
            # If Phase 3 has run for too long and the model is no longer improving,
            # pass control to Phase 4 recovery instead of hanging forever.
            
            not_catastrophically_locked = (
                obs["knee_avg"] > -0.04
                and obs["hip_avg"] > -0.45
            )
            p3_timeout_escape = (
                st.phase_elapsed >= p.p3_timeout
                and not_catastrophically_locked
                and seat_off
                and obs["pelvis_y"] >= 0.74
                and obs["torso_y"] >= 0.74
                and obs["root_pitch"] > -1.30
                and obs.get("pelvis_to_feet_x", -999.0) > -0.08
            )
            
            # print(st.phase_elapsed >= p.p3_timeout,seat_off,obs["torso_y"],obs["pelvis_y"],obs["root_pitch"])
            # print(p3_timeout_escape)
            


            if st.phase_elapsed >= 1.4:
                st.p3_done = True
                self._set_phase(4)

            elif self.debug and st.phase_elapsed > p.p3_min_time:
                print(
                    "[P3 HOLD]",
                    f"height={height_ok}",
                    f"pitch={pitch_ok}",
                    f"lean_ok={lean_ok}",
                    f"vel_ok={vel_ok}",
                    f"gap_ok={gap_ok}",
                    f"feet={feet_ok}",
                    f"escape={p3_spent_escape}",
                    f"timeout_escape={p3_timeout_escape}",
                    f"seat={obs['seat_contact']}",
                    f"pelvis_z={obs['pelvis_y']:.3f}",
                    f"torso_z={obs['torso_y']:.3f}",
                    f"gap={torso_pelvis_gap:.3f}",
                    f"lean={obs['trunk_lean_rel']:.3f}",
                    f"root_pitch={obs['root_pitch']:.3f}",
                    f"pitch_vel={obs['root_pitch_rel_vel']:.3f}",
                    f"Lfoot={obs['left_foot_contact']}",
                    f"Rfoot={obs['right_foot_contact']}",
                    f"Lload={obs['left_foot_load']:.3f}",
                    f"Rload={obs['right_foot_load']:.3f}",
                    
                    f"ankle_r={obs['ankle_r']:.3f} ",
                    f"ankle_l={obs['ankle_l']:.3f} ",
                    f"ankle_vel={obs['ankle_avg_vel']:.3f} ",
                    f"Lload={obs['left_foot_load']:.3f} ",
                    f"Rload={obs['right_foot_load']:.3f} ",
                    f"Lclear={obs['left_foot_clearance']:.4f} ",
                    f"Rclear={obs['right_foot_clearance']:.4f} ",

                )
        elif st.phase == 4:
            stable_enough = (
                obs["pelvis_y"] > 0.80
                and obs["torso_y"] > 0.80
                and obs["pelvis_y_vel"] > -0.035
                and obs["torso_y_vel"] > -0.035
                and obs["root_pitch"] > -1.18
                and obs["trunk_lean_rel"] < 0.80
                and obs["forward_metric"] > 0.18
                and obs["left_foot_contact"]
                and obs["right_foot_contact"]
                and not obs["seat_contact"]
            )

            if st.phase_elapsed >= 0.35 and stable_enough:
                self._set_phase(5)

        return self.phase_state.phase

    def get_phase(self) -> int:
        if not self.obs:
            self.get_observation()
        return self.update_phase(self.obs)

    # ------------------------------------------------------------------
    # Reflex modules and stimulation
    # ------------------------------------------------------------------

    def _phase_gates(self, phase: int) -> Dict[str, float]:
        return {
            "p1": float(phase == 1),
            "p2": float(phase == 2),
            "p3": float(phase == 3),
            "p4": float(phase == 4),
        }

    def _compute_reflex_modules(self, obs: Dict[str, float], phase: int) -> Dict[str, float]:
        p = self.params
        g = self._phase_gates(phase)
        t = float(obs.get("phase_elapsed", 0.0))

        lean = obs["trunk_lean_rel"]
        dlean = obs["trunk_lean_vel"]
        forward = obs["forward_metric"]

        pelvis_z = obs["pelvis_y"]
        torso_z = obs["torso_y"]

        hip = obs["hip_avg"]
        knee = obs["knee_avg"]
        ankle = obs["ankle_avg"]

        dhip = obs["hip_avg_vel"]
        dknee = obs["knee_avg_vel"]
        dankle = obs["ankle_avg_vel"]

        # ---------------------------------------------------------------------
        # Phase 1: flexion momentum
        # ---------------------------------------------------------------------
        p1_ramp = _clip(t / 0.35, 0.0, 1.0)

        S_P1_TORSO_FLEX = g["p1"] * p1_ramp * (
            p.k_p1_torso_flex_lean * _pos(p.p1_lean_target - lean)
            + p.k_p1_torso_flex_forward * _pos(p.p1_forward_target - forward)
        )

        S_P1_TORSO_EXT = g["p1"] * (
            p.k_p1_torso_brake * _pos(lean - p.p1_lean_hard_limit)
            + 0.08 * _pos(dlean)
        )

        S_P1_TA = 0.0
        S_P1_SOL = 0.0

        # ---------------------------------------------------------------------
        # Phase 2: hip-driven momentum transfer / seat-off
        # ---------------------------------------------------------------------
        if phase == 2:
            burst = 1.45 if t < 0.22 else 0.80
        else:
            burst = 0.0

        height_error_p2 = _pos(p.p2_pelvis_target - pelvis_z)
        pelvis_vz_target_p2 = 0.14
        pelvis_rise_vel_error_p2 = _pos(pelvis_vz_target_p2 - obs["pelvis_z_vel"])

        hip_available = _pos(hip - p.p2_hip_stop)
        hip_overextended = _pos(p.p2_hip_stop - hip)

        knee_available = _pos(knee - p.p2_knee_stop)
        knee_overextended = _pos(p.p2_knee_stop - knee)

        lean_ready_gate = _clip((lean - 0.18) / 0.18, 0.0, 1.0)

        # Important change: do not suppress GLU too aggressively as lean increases.
        # The previous version reduced hip drive exactly when the trunk was starting
        # to collapse forward.
        lean_safe_gate = _clip((p.p2_lean_limit + 0.20 - lean) / 0.25, 0.65, 1.0)

        hip_lift_demand = (
            1.10 * hip_available
            + 0.65 * height_error_p2
            + 1.10 * pelvis_rise_vel_error_p2
        )

        raw_glu2 = g["p2"] * burst * lean_ready_gate * lean_safe_gate * (
            p.k_p2_glu_lift * hip_lift_demand
            - 0.025 * dhip
        )

        raw_vas2 = g["p2"] * burst * (
            p.k_p2_vas_lift * (
                0.55 * knee_available
                + 0.20 * height_error_p2
                + 0.15 * pelvis_rise_vel_error_p2
            )
            - 0.025 * dknee
        )

        S_P2_SOL = g["p2"] * burst * _pos(
            0.35 * p.k_p2_sol_lift * (
                0.20 * height_error_p2
                + 0.10 * pelvis_rise_vel_error_p2
            )
            - 0.010 * dankle
        )

        S_P2_GLU_INHIBIT = g["p2"] * p.k_p2_glu_inhibit * hip_overextended
        S_P2_VAS_INHIBIT = g["p2"] * p.k_p2_vas_inhibit * knee_overextended

        S_P2_GLU = _pos(raw_glu2 - S_P2_GLU_INHIBIT)
        S_P2_VAS = _pos(raw_vas2 - S_P2_VAS_INHIBIT)

        S_P2_TORSO_EXT = g["p2"] * (
            p.k_p2_torso_brake * _pos(lean - p.p2_lean_limit)
            + 0.04 * _pos(dlean)
        )

        # Catch torso collapse before Phase 3.
        if phase == 2 and obs["torso_z_vel"] < -0.08:
            S_P2_GLU += 0.15
            S_P2_TORSO_EXT += 0.20
            S_P2_VAS = max(S_P2_VAS, 0.20)

        # ---------------------------------------------------------------------
        # Phase 3: extension / hip-driven rise continuation
        # ---------------------------------------------------------------------
        phase3_time = t if phase == 3 else 0.0
        
        S_P3_HAM_BRAKE = g["p3"] * (
            0.45 * _pos(0.08 - knee)
            + 0.20 * _pos(-0.12 - hip)
        )

        pelvis_height_error = _pos(p.p3_pelvis_target - pelvis_z)
        torso_height_error = _pos(p.p3_torso_target_z - torso_z)

        pelvis_vz_error_p3 = _pos(p.p3_pelvis_vz_target - obs["pelvis_z_vel"])
        pelvis_not_high_gate_p3 = _clip(
            (p.p3_min_upward_drive_until_height - pelvis_z) / 0.16,
            0.0,
            1.0,
        )
        hip_rise_boost_p3 = 1.75 * pelvis_not_high_gate_p3 * pelvis_vz_error_p3

        torso_pelvis_gap = torso_z - pelvis_z
        torso_gap_error = _pos(p.p3_min_torso_above_pelvis - torso_pelvis_gap)

        lean_excess_p3 = _pos(lean - 0.35)

        hip_available_p3 = _pos(hip - p.p3_hip_stop)
        knee_available_p3 = _pos(knee - p.p3_knee_stop)

        hip_overextended_p3 = _pos(p.p3_hip_stop - hip)
        knee_overextended_p3 = _pos(p.p3_knee_stop - knee)
        
        knee_lock_risk_p3 = _pos(0.08 - knee)
        hip_lock_risk_p3 = _pos(-0.10 - hip)

        lean_knee_gate = _clip((0.72 - lean) / 0.28, 0.15, 1.00)
        lean_hip_gate = _clip((0.78 - lean) / 0.30, 0.35, 1.00)

        early_p3_support = 1.15 if (phase == 3 and phase3_time < 0.30) else 1.0

        trunk_lift_demand = (
            p.k_p3_torso_height * torso_height_error
            + p.k_p3_torso_pelvis_gap * torso_gap_error
            + 2.20 * lean_excess_p3
            + 0.65 * _pos(dlean)
            + 0.80 * _pos(-obs["root_pitch"] - 0.90)
        )

        S_P3_TORSO_EXT = g["p3"] * trunk_lift_demand

        S_P3_GLU_EXT_RAW = g["p3"] * early_p3_support * lean_hip_gate * (
            1.05 * hip_available_p3
            + 0.95 * pelvis_height_error
            + 0.45 * torso_height_error
            + 0.30 * torso_gap_error
            + p.k_p3_pelvis_vz_glu * hip_rise_boost_p3
            - 0.035 * dhip
        )

        S_P3_VAS_EXT_RAW = (
            g["p3"]
            * early_p3_support
            * lean_knee_gate
            * (
                p.p3_vas_support_scale
                * (
                    0.55 * knee_available_p3
                    + 0.45 * pelvis_height_error
                    + 0.15 * torso_height_error
                )
                - 0.04 * dknee
            )
        )

        S_P3_SOL_SUPPORT = 0.0

        S_P3_GLU_INHIBIT = g["p3"] * p.k_p3_glu_inhibit * hip_overextended_p3
        S_P3_VAS_INHIBIT = g["p3"] * p.k_p3_vas_inhibit * knee_overextended_p3

        S_P3_GLU_EXT = _pos(S_P3_GLU_EXT_RAW - S_P3_GLU_INHIBIT)
        S_P3_VAS_EXT = _pos(S_P3_VAS_EXT_RAW - S_P3_VAS_INHIBIT)
        
        if phase == 3:
            # Avoid handing Phase 4 a locked-leg posture.
            # If knee is nearly straight, reduce VAS and use HAM lightly to preserve compliance.
            if knee < 0.08:
                S_P3_VAS_EXT *= 0.45
                # S_P3_SOL_SUPPORT *= 0.75

            # If hip is already very extended, stop using GLU as a rise motor.
            if hip < -0.10:
                S_P3_GLU_EXT *= 0.35

        # Minimum knee support during Phase 3. This is support only, not the
        # primary pelvis-rise drive.
        if phase == 3 and pelvis_z < p.p3_pelvis_target:
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, p.p3_min_vas_support)
            S_P3_SOL_SUPPORT = 0.0

        # Emergency: pelvis rises but torso does not.
        if phase == 3 and obs["pelvis_z_vel"] > 0.02 and obs["torso_z_vel"] < p.p3_torso_rise_vel_min:
            S_P3_TORSO_EXT += 0.18
            S_P3_GLU_EXT += 0.06
            # S_P3_SOL_SUPPORT += 0.05
            S_P3_VAS_EXT *= 0.90

        # Emergency: trunk too far forward.
        if phase == 3 and lean > p.p3_lean_hard_limit:
            S_P3_TORSO_EXT += 0.22
            S_P3_GLU_EXT += 0.08
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.40)

        torso_falling = phase == 3 and obs["torso_z_vel"] < -0.005
        torso_stalled = phase == 3 and -0.005 <= obs["torso_z_vel"] < 0.015

        if torso_falling:
            S_P3_TORSO_EXT += 0.35 + 0.8 * abs(obs["torso_z_vel"])
            S_P3_GLU_EXT += 0.10
            # S_P3_SOL_SUPPORT += 0.08
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.42)

        elif torso_stalled:
            S_P3_TORSO_EXT += 0.18
            S_P3_GLU_EXT += 0.06
            # S_P3_SOL_SUPPORT += 0.04
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.38)

        if phase == 3 and torso_z < 0.68:
            height_deficit = 0.68 - torso_z
            S_P3_TORSO_EXT += 0.45 + 1.50 * height_deficit
            S_P3_GLU_EXT += 0.08 + 0.35 * height_deficit
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.40)
            # S_P3_SOL_SUPPORT = max(S_P3_SOL_SUPPORT, 0.12)

        # Rescue: pelvis vertical velocity is collapsing before standing height.
        # This is now hip-range-aware, because once hip is already overextended,
        # more GLU mostly rotates/collapses rather than lifting.
        if phase == 3 and pelvis_z < p.p3_min_upward_drive_until_height and obs["pelvis_z_vel"] < 0.08:
            vz_deficit = p.p3_pelvis_vz_target - obs["pelvis_z_vel"]
            hip_can_still_extend = _clip((hip - 0.12) / 0.35, 0.0, 1.0)

            S_P3_GLU_EXT += hip_can_still_extend * (0.45 + 2.20 * _pos(vz_deficit))
            S_P3_TORSO_EXT += 0.12 + p.k_p3_pelvis_vz_torso * _pos(vz_deficit)

            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.42)
            # S_P3_SOL_SUPPORT = max(S_P3_SOL_SUPPORT, 0.16)

        # If hip range is spent, stop relying on GLU alone and add structural support.
        if phase == 3 and hip < 0.05:
            S_P3_GLU_EXT = min(S_P3_GLU_EXT, 0.10)
            S_P3_VAS_EXT = max(S_P3_VAS_EXT, 0.65)
            # S_P3_SOL_SUPPORT = max(S_P3_SOL_SUPPORT, 0.30)
            S_P3_TORSO_EXT += 0.35 * _pos(-torso_pelvis_gap)
                # ---------------------------------------------------------------------
        # Phase 4: stabilization
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # Phase 4: rise continuation + stabilization
        # ---------------------------------------------------------------------
        S_P4_COCON = g["p4"] * p.k_p4_cocontract

        # If Phase 4 starts before pelvis has fully settled at standing height,
        # keep a reduced version of the Phase 3 rise support active.
        p4_pelvis_height_error = _pos(p.p4_pelvis_target - pelvis_z)
        p4_pelvis_vz_error = _pos(p.p4_pelvis_vz_target - obs["pelvis_z_vel"])

        p4_rise_gate = g["p4"] * _clip(
            (p.p4_rise_until_height - pelvis_z) / 0.12,
            0.0,
            1.0,
        )

        # Hip range gate: use GLU only if hip can still meaningfully extend.
        p4_hip_range_gate = _clip((hip - 0.08) / 0.30, 0.0, 1.0)

        S_P4_GLU_RISE = p4_rise_gate * p4_hip_range_gate * (
            p.k_p4_glu_rise * (
                0.80 * p4_pelvis_height_error
                + 0.90 * p4_pelvis_vz_error
            )
        )

        S_P4_VAS_RISE = p4_rise_gate * p.k_p4_vas_support * (
            0.65 * p4_pelvis_height_error
            + 0.35 * p4_pelvis_vz_error
        )

        S_P4_SOL_RISE = p4_rise_gate * p.k_p4_sol_support * (
            0.55 * p4_pelvis_height_error
            + 0.30 * p4_pelvis_vz_error
        )

        S_P4_TORSO_RISE = p4_rise_gate * p.k_p4_torso_rise * (
            0.35 * _pos(p.p3_torso_target_z - torso_z)
            + 0.25 * _pos(-obs["torso_z_vel"])
            + 0.12 * _pos(lean - p.p4_lean_target)
        )

        # Normal standing stabilization.
        S_P4_GLU_STAB = g["p4"] * p.k_p4_hip * _pos(hip - p.p4_hip_target)
        S_P4_HFL = g["p4"] * p.k_p4_hip * _pos(p.p4_hip_target - hip)

        S_P4_VAS_STAB = g["p4"] * p.k_p4_knee * _pos(knee - p.p4_knee_target)
        S_P4_HAM = g["p4"] * p.k_p4_knee * _pos(p.p4_knee_target - knee)

        S_P4_SOL_STAB = g["p4"] * p.k_p4_ankle * _pos(ankle - p.p4_ankle_target)
        S_P4_TA = g["p4"] * p.k_p4_ankle * _pos(p.p4_ankle_target - ankle)

        S_P4_TORSO_EXT_STAB = g["p4"] * p.k_p4_torso * _pos(lean - p.p4_lean_target)
        S_P4_TORSO_FLEX = g["p4"] * p.k_p4_torso * _pos(p.p4_lean_target - lean)

        S_P4_GLU = S_P4_GLU_STAB + S_P4_GLU_RISE
        S_P4_VAS = S_P4_VAS_STAB + S_P4_VAS_RISE
        S_P4_SOL = S_P4_SOL_STAB + S_P4_SOL_RISE
        S_P4_TORSO_EXT = S_P4_TORSO_EXT_STAB + S_P4_TORSO_RISE
        
        
        # ---------------------------------------------------------------------
# Phase 4 anti-overextension / posture recovery
# ---------------------------------------------------------------------
        p4_knee_overextended = _pos(p.p4_knee_target - knee)
        p4_hip_overextended = _pos(p.p4_hip_target - hip)
        p4_ankle_too_plantarflexed = _pos(p.p4_ankle_target - ankle)
        

        if phase == 4:
            # If knee is too straight / hyperextended, use HAM to unlock it slightly.
            S_P4_HAM = max(S_P4_HAM, 0.45 * p4_knee_overextended)

            # If hip is too extended, use HFL to recover pelvis/trunk relation.
            S_P4_HFL = max(S_P4_HFL, 0.35 * p4_hip_overextended)

            # If ankle is too plantarflexed, use TA to pull it back.
            S_P4_TA = max(S_P4_TA, 0.55 * p4_ankle_too_plantarflexed)

        # Minimum support while pelvis is still below standing height.
        if phase == 4 and pelvis_z < p.p4_rise_until_height:
            S_P4_VAS = max(S_P4_VAS, p.p4_min_vas_support)
            S_P4_SOL = max(S_P4_SOL, p.p4_min_sol_support)
            
        if phase == 4 and pelvis_z < 0.86:
            S_P4_VAS = max(S_P4_VAS, 0.55)
            S_P4_SOL = max(S_P4_SOL, 0.20)
            if torso_pelvis_gap < -0.02 or obs["torso_z_vel"] < -0.02:
                S_P4_TORSO_EXT = max(S_P4_TORSO_EXT, 0.75)
        
        if phase == 4 and torso_pelvis_gap > -0.02 and obs["torso_z_vel"] > -0.02:
            S_P4_TORSO_EXT = min(S_P4_TORSO_EXT, 0.45)

        modules = {
            "S_P1_TORSO_FLEX": S_P1_TORSO_FLEX,
            "S_P1_TORSO_EXT": S_P1_TORSO_EXT,
            "S_P1_TA": S_P1_TA,
            "S_P1_SOL": S_P1_SOL,

            "S_P2_GLU": S_P2_GLU,
            "S_P2_VAS": S_P2_VAS,
            "S_P2_SOL": S_P2_SOL,
            "S_P2_TORSO_EXT": S_P2_TORSO_EXT,
            "S_P2_GLU_INHIBIT": S_P2_GLU_INHIBIT,
            "S_P2_VAS_INHIBIT": S_P2_VAS_INHIBIT,

            "S_P3_GLU": S_P3_GLU_EXT,
            "S_P3_VAS": S_P3_VAS_EXT,
            "S_P3_SOL": S_P3_SOL_SUPPORT,
            "S_P3_TORSO_EXT": S_P3_TORSO_EXT,
            "S_P3_GLU_INHIBIT": S_P3_GLU_INHIBIT,
            "S_P3_VAS_INHIBIT": S_P3_VAS_INHIBIT,

            "S_P4_COCON": S_P4_COCON,
            "S_P4_GLU": S_P4_GLU,
            "S_P4_HFL": S_P4_HFL,
            "S_P4_VAS": S_P4_VAS,
            "S_P4_HAM": S_P4_HAM,
            "S_P4_SOL": S_P4_SOL,
            "S_P4_TA": S_P4_TA,
            "S_P4_TORSO_EXT": S_P4_TORSO_EXT,
            "S_P4_TORSO_FLEX": S_P4_TORSO_FLEX,
            "S_P3_HAM": S_P3_HAM_BRAKE
        }

        return {k: _clip(v, 0.0, 1.5) for k, v in modules.items()}

    def _modules_to_stim(self, modules: Dict[str, float], phase: Optional[int] = None) -> Dict[str, float]:
        p = self.params
        sc = p.group_scale

        stim = {k: 0.0 for k in self.m_keys}

        if phase == 1:
            stim["TORSO_FLEX"] = p.tonic
            stim["TORSO_EXT"] = p.tonic
        else:
            for k in self.m_keys:
                stim[k] = p.tonic
                
        

        # P1
        stim["TORSO_FLEX"] += sc["TORSO_FLEX"] * modules["S_P1_TORSO_FLEX"]
        stim["TORSO_EXT"] += sc["TORSO_EXT"] * modules["S_P1_TORSO_EXT"]
        stim["TA"] += sc["TA"] * modules["S_P1_TA"]
        stim["SOL"] += sc["SOL"] * modules["S_P1_SOL"]

        # P2
        stim["GLU"] += sc["GLU"] * modules["S_P2_GLU"]
        stim["VAS"] += sc["VAS"] * modules["S_P2_VAS"]
        stim["SOL"] += sc["SOL"] * modules["S_P2_SOL"]
        stim["TORSO_EXT"] += sc["TORSO_EXT"] * modules["S_P2_TORSO_EXT"]

        # P3
        stim["GLU"] += sc["GLU"] * modules["S_P3_GLU"]
        stim["VAS"] += sc["VAS"] * modules["S_P3_VAS"]
        stim["SOL"] += sc["SOL"] * modules["S_P3_SOL"]
        stim["HAM"] += sc["HAM"] * modules["S_P3_HAM"]
        stim["TORSO_EXT"] += sc["TORSO_EXT"] * modules["S_P3_TORSO_EXT"]

        # P4
        stim["GLU"] += sc["GLU"] * (modules["S_P4_GLU"] + modules["S_P4_COCON"])
        stim["HFL"] += sc["HFL"] * (modules["S_P4_HFL"] + modules["S_P4_COCON"])
        stim["VAS"] += sc["VAS"] * (modules["S_P4_VAS"] + modules["S_P4_COCON"])
        stim["HAM"] += sc["HAM"] * (modules["S_P4_HAM"] + modules["S_P4_COCON"])
        stim["SOL"] += sc["SOL"] * (modules["S_P4_SOL"] + modules["S_P4_COCON"])
        stim["TA"] += sc["TA"] * (modules["S_P4_TA"] + modules["S_P4_COCON"])
        stim["TORSO_EXT"] += sc["TORSO_EXT"] * modules["S_P4_TORSO_EXT"]
        stim["TORSO_FLEX"] += sc["TORSO_FLEX"] * modules["S_P4_TORSO_FLEX"]

        if phase == 1:
            out = {}
            for k, v in stim.items():
                if k in ["TORSO_FLEX", "TORSO_EXT"]:
                    out[k] = _clip(v, 0.01, 1.0)
                else:
                    out[k] = _clip(v, 0.0, 1.0)
            return out
        
        elif phase == 3:
            stim["TORSO_EXT"] = min(stim["TORSO_EXT"], 0.52)
            
        

        return {k: _clip(v, 0.01, 1.0) for k, v in stim.items()}

    def _zero_leg_actuators(self, ctrl: np.ndarray) -> None:
        leg_groups = [
            "GLU_r", "GLU_l",
            "VAS_r", "VAS_l",
            "SOL_r", "SOL_l",
            "TA_r", "TA_l",
            "HFL_r", "HFL_l",
            "HAM_r", "HAM_l",
        ]

        for group_name in leg_groups:
            for idx in self.muscle_group_ids.get(group_name, []):
                ctrl[idx] = 0.0

    def _stim_to_action(self, stim: Dict[str, float], phase: Optional[int] = None) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=float)

        self._add_bilateral(ctrl, "GLU", stim["GLU"])
        self._add_bilateral(ctrl, "VAS", stim["VAS"])
        self._add_bilateral(ctrl, "SOL", stim["SOL"])
        self._add_bilateral(ctrl, "TA", stim["TA"])
        self._add_bilateral(ctrl, "HFL", stim["HFL"])
        self._add_bilateral(ctrl, "HAM", stim["HAM"])

        self._add_group(ctrl, "TORSO_EXT", stim["TORSO_EXT"])
        self._add_group(ctrl, "TORSO_FLEX", stim["TORSO_FLEX"])

        phase = int(self.phase_state.phase)

        if phase == 1:
            ctrl *= 0.1
        elif phase == 2:
            ctrl *= 0.9
        elif phase == 3:
            ctrl *= 0.9
        elif phase == 4:
            ctrl *= 0.80

        ctrl = np.clip(ctrl, 0.0, 1.0)

        return ctrl

    def compute_reflex_stim(
        self,
        obs: Optional[Dict[str, float]] = None,
        phase: Optional[int] = None,
    ) -> Dict[str, float]:
        if obs is None:
            obs = self.obs if self.obs else self.get_observation()

        if phase is None:
            phase = self.phase_state.phase

        modules = self._compute_reflex_modules(obs, int(phase))
        stim = self._modules_to_stim(modules, phase=int(phase))
        
        # ------------------------------------------------------------
        # Phase 4 recovery support
        # ------------------------------------------------------------
        st = self.phase_state
        falling_in_p4 = (
            st.phase == 4
            and (
                obs["pelvis_y_vel"] < -0.04
                or obs["torso_y_vel"] < -0.04
                or obs["pelvis_y"] < 0.78
                or obs["seat_contact"]
            )
        )

        p4_recovery = 1.0 if falling_in_p4 else 0.0
        
        if st.phase in (1, 2):
            heel_lift = max(
                obs.get("l_heel_minus_toe_z", 0.0),
                obs.get("r_heel_minus_toe_z", 0.0),
            )

            # If heel is much higher than toe, reduce plantarflexion and add TA.
            if heel_lift > 0.055:
                excess = heel_lift - 0.055

                stim["SOL"] *= 0.35
                stim["TA"] = max(stim["TA"], 0.08 + 2.0 * excess)

                if self.debug:
                    print(
                        f"[P{st.phase} ANTI TOE] "
                        f"heel_lift={heel_lift:.4f} "
                        f"SOL={stim['SOL']:.3f} TA={stim['TA']:.3f}"
                    )
        
        if st.phase == 4:
            legs_locked = (
                obs["knee_avg"] < 0.06
                or obs["hip_avg"] < -0.25
            )

            if legs_locked:
                stim["GLU"] *= 0.35
                stim["VAS"] *= 0.55
                stim["SOL"] *= 0.70
                stim["HAM"] = max(stim["HAM"], 0.08)
                stim["HFL"] = max(stim["HFL"], 0.08)
            hip_forward_gate = _clip((obs["hip_avg"] + 0.35) / 0.30, 0.0, 1.0)

            stim["GLU"] += p4_recovery * (0.22 if obs["hip_avg"] > -0.15 else 0.06)
            stim["VAS"] += p4_recovery * 0.30
            stim["SOL"] += p4_recovery * 0.16

            forward_deficit = _pos(self.params.p4_forward_good - obs["forward_metric"])
            stim["GLU"] += hip_forward_gate * 0.35 * forward_deficit
            stim["VAS"] += 0.18 * forward_deficit
            stim["SOL"] += 0.08 * forward_deficit
            ankle_avg = obs["ankle_avg"]

            if ankle_avg < -0.35:
                toe_excess = -0.35 - ankle_avg

                stim["SOL"] *= 0.35
                stim["TA"] = max(stim["TA"], 0.18 + 1.00 * toe_excess)
            
        elif st.phase == 3:
            # ------------------------------------------------------------
            # Phase 3 anti-overdrive / posture management
            # ------------------------------------------------------------
            # Do NOT add torso flexion when the model is already leaning forward.
            # That was making the folded posture worse. Use extension to recover
            # the trunk, but taper leg drive once the pelvis/torso have reached
            # the Phase-4 handoff region.

            lean_excess = max(obs["trunk_lean_rel"] - 0.62, 0.0)

            if lean_excess > 0.0:
                stim["TORSO_EXT"] += 1.20 * lean_excess
                stim["TORSO_FLEX"] = 0.0

            # Taper P3 extension once the body is already near the handoff region.
            # This prevents P3 from endlessly pushing while waiting for impossible
            # exit conditions.
            lift_progress = np.clip(
                (obs["pelvis_y"] - 0.620) / (0.700 - 0.620),
                0.0,
                1.0,
            )

            stim["GLU"] *= (1.0 - 0.35 * lift_progress)
            stim["VAS"] *= (1.0 - 0.10 * lift_progress)
            stim["SOL"] *= (1.0 - 0.35 * lift_progress)

            # Keep enough support to avoid collapse.
            # Keep knee/torso support, but do NOT force SOL support.
# SOL causes plantarflexion/toe-rise in Phase 3.
            stim["VAS"] = max(stim["VAS"], 0.42)
            stim["SOL"] = min(stim["SOL"], 0.02)
            stim["TA"] = max(stim["TA"], 0.16)
            stim["TORSO_EXT"] = np.clip(stim["TORSO_EXT"], 0.18, 0.52)
            stim["TORSO_FLEX"] = 0.0

        
            
        stim = {k: _clip(v, 0.01, 1.0) for k, v in stim.items()}
        if st.phase == 3:
            stim["TORSO_FLEX"] = 0.0
        if st.phase == 3:
            ankle_avg = obs["ankle_avg"]
            pelvis_z = obs["pelvis_z"]
            knee_avg = obs["knee_avg"]

            # Hard rule: Phase 3 is not allowed to push through SOL.
            stim["SOL"] = min(stim["SOL"], 0.02)

            # Base dorsiflexor support to resist toe-rise.
            stim["TA"] = max(stim["TA"], 0.16)

            if ankle_avg < -0.38:
                toe_excess = -0.38 - ankle_avg

                # Pull the foot back toward flat.
                stim["TA"] = max(stim["TA"], 0.20 + 1.60 * toe_excess)

                # Reduce vertical drive if the body is converting knee extension into toe pivot.
                if pelvis_z > 0.82:
                    stim["VAS"] *= 0.80

                if pelvis_z > 0.90 and knee_avg < 0.65:
                    stim["VAS"] *= 0.55

                if pelvis_z > 0.95 and knee_avg < 0.35:
                    stim["VAS"] *= 0.30
                    stim["GLU"] *= 0.70

            # Avoid handing Phase 4 fully locked legs.
            if knee_avg < 0.18:
                stim["VAS"] *= 0.45
                stim["HAM"] = max(stim["HAM"], 0.06)
                    
        if st.phase == 3 and obs["pelvis_z"] > 0.95 and obs["root_pitch"] > -0.80:
            stim["TORSO_EXT"] *= 0.55
            
        if st.phase in (3, 4) and obs["ankle_avg"] < -0.45:
            print(
                f"[ANTI TOE] phase={st.phase} "
                f"ankle={obs['ankle_avg']:.3f} "
                f"SOL={stim['SOL']:.3f} "
                f"TA={stim['TA']:.3f}"
            )

        self.module_outputs = dict(modules)
        self.last_stim = dict(stim)

        return stim

    def compute_action(
        self,
        obs: Optional[Dict[str, float]] = None,
        phase: Optional[int] = None,
    ) -> np.ndarray:
        stim = self.compute_reflex_stim(obs=obs, phase=phase)
        return self._stim_to_action(stim, phase=phase)

    def _debug_phase1_leg_ctrl_sum(self, action: np.ndarray) -> float:
        leg_sum = 0.0
        for group_name in [
            "GLU_r", "GLU_l",
            "VAS_r", "VAS_l",
            "SOL_r", "SOL_l",
            "TA_r", "TA_l",
            "HFL_r", "HFL_l",
            "HAM_r", "HAM_l",
        ]:
            for idx in self.muscle_group_ids.get(group_name, []):
                leg_sum += abs(float(action[idx]))
        return float(leg_sum)

    def step(self, control_u: Optional[Dict[str, float]] = None, phase: Optional[int] = None):
    # control_u is ignored intentionally; accepted only to preserve old API.

        # Always refresh observations before computing phase/action.
        obs = self.get_observation()

        if phase is None:
            phase = self.update_phase(obs)

        phase = int(phase)

        if phase == 1 and self.p1_hold_qpos is None:
            self.capture_phase1_hold_pose()

        if not self.foot_anchor_pos:
            self.capture_foot_anchors()
        
        if not self.heel_anchor_z:
            self.capture_heel_anchors()

        action = self.compute_action(self.obs, phase)

        # Clear old applied generalized/body forces.
        if hasattr(self.sim.data, "qfrc_applied"):
            self.sim.data.qfrc_applied[:] = 0.0

        if hasattr(self.sim.data, "xfrc_applied"):
            self.sim.data.xfrc_applied[:] = 0.0

        # Root hold is Phase 1 only. Do not hold root_z in Phase 2/3.
        if hasattr(self.sim.data, "qfrc_applied"):
            if phase == 1:
                if self.params.hold_root_in_phase1:
                    self.apply_phase1_root_pd_hold()

                if self.params.hold_legs_in_phase1:
                    self.apply_phase1_leg_pd_hold()

            self.apply_root_pitch_forward_brake(phase)

            self.apply_heel_down_pd(phase)

            if phase == 4:
                self.apply_phase4_hip_forward_push()
            if phase in (2, 3):
                sync_specs = {
                    "hip": ("hip_flexion_r", "hip_flexion_l", 45.0, 5.0, 35.0),
                    "knee": ("knee_angle_r", "knee_angle_l", 180.0, 18.0, 120.0),
                    "ankle": ("ankle_angle_r", "ankle_angle_l", 50.0, 6.0, 45.0),
                }

                for _, (jr, jl, kp, kd, limit) in sync_specs.items():
                    jid_r = self.name2jid[jr]
                    jid_l = self.name2jid[jl]

                    qadr_r = self.model.jnt_qposadr[jid_r]
                    qadr_l = self.model.jnt_qposadr[jid_l]
                    dadr_r = self.model.jnt_dofadr[jid_r]
                    dadr_l = self.model.jnt_dofadr[jid_l]

                    qr = float(self.sim.data.qpos[qadr_r])
                    ql = float(self.sim.data.qpos[qadr_l])
                    qdr = float(self.sim.data.qvel[dadr_r])
                    qdl = float(self.sim.data.qvel[dadr_l])

                    q_avg = 0.5 * (qr + ql)
                    qd_avg = 0.5 * (qdr + qdl)

                    tau_r = kp * (q_avg - qr) + kd * (qd_avg - qdr)
                    tau_l = kp * (q_avg - ql) + kd * (qd_avg - qdl)

                    tau_r = float(np.clip(tau_r, -limit, limit))
                    tau_l = float(np.clip(tau_l, -limit, limit))

                    self.sim.data.qfrc_applied[dadr_r] += tau_r
                    self.sim.data.qfrc_applied[dadr_l] += tau_l

        # Feet remain anchored in Phases 1-3, and softly anchored in Phase 4.
        if int(phase) in self.params.anchor_feet_in_phases:
            self.apply_foot_anchor_pd(dt=float(self.obs.get("dt", self.model.opt.timestep)))

        if self.debug:
            obs = self.obs
            stim = self.last_stim
            mods = self.module_outputs

            torso_gap = obs["torso_y"] - obs["pelvis_y"]
            leg_sum = self._debug_phase1_leg_ctrl_sum(action) if phase == 1 else float("nan")
            anchor_active = int(phase) in self.params.anchor_feet_in_phases

            print(
                f"[P{phase}] "
                f"seat={obs.get('seat_contact')} "
                f"Lfoot={obs.get('left_foot_contact')} "
                f"Rfoot={obs.get('right_foot_contact')} "
                f"grounded={obs.get('grounded')} "
                f"anchor_phase={anchor_active} "
                f"root_x={obs['root_x']:.3f} "
                f"root_z={obs['root_z']:.3f} "
                f"root_pitch={obs['root_pitch']:.3f} "
                f"torso_z={obs['torso_y']:.3f} "
                f"pelvis_z={obs['pelvis_y']:.3f} "
                f"pelvis_z_vel={obs['pelvis_y_vel']:.3f} "
                f"lean={obs['trunk_lean_rel']:.3f} "
                f"forward={obs['forward_metric']:.3f} "
                f"hip={obs['hip_avg']:.3f} "
                f"knee={obs['knee_avg']:.3f} "
                f"ankle={obs['ankle_avg']:.3f} "
                
                f"hip_diff={(obs['hip_r'] - obs['hip_l']):.3f} "
                f"knee_diff={(obs['knee_r'] - obs['knee_l']):.3f} "
                f"ankle_diff={(obs['ankle_r'] - obs['ankle_l']):.3f} "
                f"foot_x={obs.get('foot_x', np.nan):.3f} "
                f"foot_clearance={self._foot_clearance_from_ground():.4f} "
                f"Lclear={obs.get('left_foot_clearance', np.nan):.4f} "
                f"Rclear={obs.get('right_foot_clearance', np.nan):.4f} "
                f"Lgeom={obs.get('left_foot_geom_contact')} "
                f"Rgeom={obs.get('right_foot_geom_contact')} "
                f"Lload={obs.get('left_foot_load', np.nan):.3f} "
                f"Rload={obs.get('right_foot_load', np.nan):.3f} "
                f"Lanchor_err={self._anchor_error('l'):.4f} "
                f"Ranchor_err={self._anchor_error('r'):.4f} "
                f"torso_gap={torso_gap:.3f} "
                f"torso_z_vel={obs['torso_y_vel']:.3f} "
                f"hip_range_p3={_clip((obs['hip_avg'] - 0.12) / 0.35, 0.0, 1.0):.3f} "
                f"P1_leg_ctrl_sum={leg_sum:.6f} "
                f"stim={{GLU:{stim['GLU']:.3f}, VAS:{stim['VAS']:.3f}, "
                f"SOL:{stim['SOL']:.3f}, TA:{stim['TA']:.3f}, "
                f"TExt:{stim['TORSO_EXT']:.3f}, TFlex:{stim['TORSO_FLEX']:.3f}}} "
                f"mods={{P1_TFlex:{mods.get('S_P1_TORSO_FLEX', 0):.3f}, "
                f"P2_GLU:{mods.get('S_P2_GLU', 0):.3f}, "
                f"P2_VAS:{mods.get('S_P2_VAS', 0):.3f}, "
                f"P3_GLU:{mods.get('S_P3_GLU', 0):.3f}, "
                f"P3_VAS:{mods.get('S_P3_VAS', 0):.3f}, "
                f"P3_SOL:{mods.get('S_P3_SOL', 0):.3f}}}"
                f"root_pitch_vel={obs['root_pitch_rel_vel']:.3f} "
                f"pitch_bad={obs['root_pitch'] < -0.95} "
                f"LheelZ={obs.get('l_heel_z', np.nan):.4f} "
                f"LtoeZ={obs.get('l_toe_z', np.nan):.4f} "
                f"LheelToeDz={obs.get('l_heel_minus_toe_z', np.nan):.4f} "
                f"LfootPitch={obs.get('l_foot_pitch_proxy', np.nan):.3f} "
                f"RheelZ={obs.get('r_heel_z', np.nan):.4f} "
                f"RtoeZ={obs.get('r_toe_z', np.nan):.4f} "
                f"RheelToeDz={obs.get('r_heel_minus_toe_z', np.nan):.4f} "
                f"RfootPitch={obs.get('r_foot_pitch_proxy', np.nan):.3f} "
                f"LgeomClear={obs.get('left_actual_geom_clearance', np.nan):.4f} "
                f"RgeomClear={obs.get('right_actual_geom_clearance', np.nan):.4f} "
                f"LgeomBottom={obs.get('left_geom_bottom_z', np.nan):.4f} "
                f"RgeomBottom={obs.get('right_geom_bottom_z', np.nan):.4f} "
                f"foot_mode="
                f"{'HEEL_UP' if obs.get('left_heel_toe_dz', 0.0) > 0.035 or obs.get('right_heel_toe_dz', 0.0) > 0.035 else 'FLAT_ENOUGH'} "
                f"p1_leg_err={obs.get('p1_max_leg_q_err', np.nan):.4f} "
                f"p1_leg_qd={obs.get('p1_max_leg_qd', np.nan):.4f} "

            )

        # Diagnostic CSV logging for phases 3+.
        if phase >= 3 and self._diag_csv_writer is not None:
            obs = self.obs
            mods = self.module_outputs
            stim = self.last_stim
            jid_fe = self.name2jid["flex_extension"]
            fe_q = float(self.sim.data.qpos[self.model.jnt_qposadr[jid_fe]])
            jid_kr = self.name2jid["knee_angle_r"]
            jid_kl = self.name2jid["knee_angle_l"]
            self._diag_csv_writer.writerow([
                self._diag_step,
                round(float(obs.get("time", self.sim.data.time)), 4),
                phase,
                round(float(obs.get("root_pitch", 0.0)), 5),
                round(float(obs.get("root_pitch_rel_vel", 0.0)), 5),
                round(fe_q, 5),
                round(float(obs.get("trunk_lean_rel", 0.0)), 5),
                round(float(obs.get("pelvis_y", 0.0)), 5),
                round(float(obs.get("pelvis_y_vel", 0.0)), 5),
                round(float(obs.get("hip_avg", 0.0)), 5),
                round(float(self.sim.data.qpos[self.model.jnt_qposadr[jid_kr]]), 5),
                round(float(self.sim.data.qpos[self.model.jnt_qposadr[jid_kl]]), 5),
                round(float(obs.get("ankle_avg", 0.0)), 5),
                round(float(mods.get("S_P4_TORSO_EXT", 0.0)), 5),
                round(float(mods.get("S_P4_TORSO_FLEX", 0.0)), 5),
                round(float(mods.get("S_P4_GLU", 0.0)), 5),
                round(float(mods.get("S_P4_VAS", 0.0)), 5),
                round(float(mods.get("S_P4_HAM", 0.0)), 5),
                round(float(mods.get("S_P4_HFL", 0.0)), 5),
                round(float(stim.get("TORSO_EXT", 0.0)), 5),
                round(float(stim.get("TORSO_FLEX", 0.0)), 5),
                round(float(stim.get("GLU", 0.0)), 5),
                round(float(stim.get("VAS", 0.0)), 5),
                round(float(stim.get("HAM", 0.0)), 5),
                round(float(stim.get("HFL", 0.0)), 5),
            ])
            self._diag_csv_file.flush()
        self._diag_step += 1

        # TSA exoskeleton: inject knee extension torques into qfrc_applied.
        if self._is_tsa_full and self.tsa is not None:
            t_now = float(self.sim.data.time)
            dt_now = float(self.obs.get("dt", self.model.opt.timestep))
            tau_knee_r = abs(float(self.sim.data.qfrc_bias[self.tsa._knee_dadr['r']]))
            tau_knee_l = abs(float(self.sim.data.qfrc_bias[self.tsa._knee_dadr['l']]))
            self.tsa.step(t_now, dt_now, tau_knee_r=tau_knee_r, tau_knee_l=tau_knee_l)
            if self._tsa_csv_writer is not None:
                r = self.tsa.last_state.get('r', {})
                l = self.tsa.last_state.get('l', {})
                r_states = r.get('motor_states', [])
                l_states = l.get('motor_states', [])
                knee_qadr_r = self.model.jnt_qposadr[self.name2jid['knee_angle_r']]
                knee_qadr_l = self.model.jnt_qposadr[self.name2jid['knee_angle_l']]
                motor_row = []
                for ms_list in (r_states, l_states):
                    for ms in ms_list:
                        motor_row += [
                            int(ms.get('active', 0)),
                            round(float(ms.get('tension', 0.0)), 4),
                            round(float(ms.get('torque', 0.0)), 4),
                            round(float(ms.get('X', 0.0)) * 1000, 3),
                            round(float(ms.get('theta', 0.0)), 4),
                            round(float(ms.get('theta_dot', 0.0)), 4),
                            int(ms.get('saturated', 0)),
                        ]
                self._tsa_csv_writer.writerow([
                    self._tsa_step_count,
                    round(t_now, 4),
                    phase,
                    round(float(self.sim.data.qpos[knee_qadr_r]), 5),
                    round(float(self.sim.data.qpos[knee_qadr_l]), 5),
                    round(tau_knee_r, 4), round(tau_knee_l, 4),
                    r.get('N_active', 0), l.get('N_active', 0),
                    round(float(r.get('torque', 0.0)), 4),
                    round(float(l.get('torque', 0.0)), 4),
                    0.0, 0.0,
                    round(float(r.get('X_geom', 0.0)) * 1000, 3),
                    round(float(l.get('X_geom', 0.0)) * 1000, 3),
                    *motor_row,
                ])
            self._tsa_step_count += 1

        return self.env.step(action)

    def close(self) -> None:
        if self._tsa_csv_file is not None:
            self._tsa_csv_file.flush()
            self._tsa_csv_file.close()
            self._tsa_csv_file = None
            self._tsa_csv_writer = None
        if self._diag_csv_file is not None:
            self._diag_csv_file.flush()
            self._diag_csv_file.close()
            self._diag_csv_file = None
            self._diag_csv_writer = None


# =============================================================================
# Compatibility shell
# =============================================================================


class Controller:
    """Old API compatibility. The real controller is inside SitToStandSim."""

    def __init__(self, params: Optional[STSReflexParams] = None, debug: bool = False):
        self.params = params if params is not None else STSReflexParams()
        self.debug = bool(debug)
        self.last_modules: Dict[str, float] = {}

    @staticmethod
    def _blank_command() -> Dict[str, float]:
        return {
            "ankle_u": 0.0,
            "hip_u": 0.0,
            "knee_u": 0.0,
            "torso_u": 0.0,
        }

    def controller(self, obs: Dict[str, float], phase: int) -> Dict[str, float]:
        p = self.params
        u = self._blank_command()

        if int(phase) == 1:
            u["torso_u"] = -_clip(
                p.p1_lean_target - obs["trunk_lean_rel"],
                0.0,
                1.0,
            )

        elif int(phase) == 2:
            h = _pos(p.p2_pelvis_target - obs["pelvis_y"])
            u["hip_u"] = _clip(_pos(obs["hip_avg"] - p.p2_hip_stop) + 0.25 * h)
            u["knee_u"] = _clip(_pos(obs["knee_avg"] - p.p2_knee_stop) + 0.25 * h)
            u["ankle_u"] = _clip(0.25 * h)
            u["torso_u"] = _clip(_pos(obs["trunk_lean_rel"] - p.p2_lean_limit))

        elif int(phase) == 3:
            h = _pos(p.p3_pelvis_target - obs["pelvis_y"])
            u["hip_u"] = _clip(_pos(obs["hip_avg"] - p.p3_hip_stop) + h)
            u["knee_u"] = _clip(_pos(obs["knee_avg"] - p.p3_knee_stop) + h)
            u["ankle_u"] = _clip(0.4 * h)
            u["torso_u"] = _clip(_pos(obs["trunk_lean_rel"] - p.p3_lean_limit))

        else:
            u["hip_u"] = _clip(obs["hip_avg"] - p.p4_hip_target, -1.0, 1.0)
            u["knee_u"] = _clip(obs["knee_avg"] - p.p4_knee_target, -1.0, 1.0)
            u["ankle_u"] = _clip(obs["ankle_avg"] - p.p4_ankle_target, -1.0, 1.0)
            u["torso_u"] = _clip(obs["trunk_lean_rel"] - p.p4_lean_target, -1.0, 1.0)

        self.last_modules = dict(u)

        if self.debug:
            print(f"[Controller compatibility P{phase}] diagnostic_u={u}")

        return u