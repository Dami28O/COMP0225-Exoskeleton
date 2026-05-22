"""
Run the STS simulation with PPO-optimised TSA motor parameters.

Also runs a headless baseline (no exoskeleton) with identical STS controller
params and compares quadriceps muscle activation between the two conditions.

Set POLICY_PATH to a saved ppo_tsa_final.zip to load optimised params,
or set it to None to use the default motor configs.

Run:
    mjpython run_optimal.py
"""

import os
import signal
import sys
from pathlib import Path

import numpy as np

MYOASSIST_DIR = Path(__file__).resolve().parent / "myoassist"
if str(MYOASSIST_DIR) not in sys.path:
    sys.path.insert(0, str(MYOASSIST_DIR))

CTRL_OPTIM_DIR = Path(__file__).resolve().parent / "ctrl_optim"
if str(CTRL_OPTIM_DIR) not in sys.path:
    sys.path.insert(0, str(CTRL_OPTIM_DIR))

import myosuite.envs.myo.myobase  # noqa: F401
from myosuite.utils import gym
from sts_ctrl import SitToStandSim, STSReflexParams
from tsa_integration_full import MotorConfig

# ── Policy path ───────────────────────────────────────────────────────────────
POLICY_PATH = "logs/ppo_v13/ppo_tsa_final"
# POLICY_PATH = None

# ── Quad muscle actuator names (must match model) ─────────────────────────────
_QUAD_NAMES = [
    "vaslat_r", "vasmed_r", "vasint_r", "recfem_r",
    "vaslat_l", "vasmed_l", "vasint_l", "recfem_l",
]
_QUAD_NAMES_R = [n for n in _QUAD_NAMES if n.endswith("_r")]
_QUAD_NAMES_L = [n for n in _QUAD_NAMES if n.endswith("_l")]

_LATERAL_OFFSETS = [0.0, 0.0, 8.0, -8.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_policy_params(policy_path: str):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("stable_baselines3 is required to load a PPO policy")

    model = PPO.load(policy_path)
    action, _ = model.predict(np.zeros(1, dtype=np.float32), deterministic=True)
    L, *t_vals = action.tolist()
    t_sorted = sorted(t_vals)

    cfgs = [
        MotorConfig(lateral_offset_deg=off, activation_time=float(t), name=f"M{i}")
        for i, (off, t) in enumerate(zip(_LATERAL_OFFSETS, t_sorted))
    ]

    print(f"\nLoaded policy: {policy_path}")
    print(f"  L   = {L:.4f} m")
    for i, t in enumerate(t_sorted):
        print(f"  t{i}  = {t:.4f} s")
    print()

    return cfgs, L


def _get_quad_ids(model, names):
    act_map = {model.actuator(i).name: i for i in range(model.nu)}
    ids = [act_map[n] for n in names if n in act_map]
    if not ids:
        print(f"[warn] No quad actuators matched in model. Names tried: {names}")
    return ids


def set_seated_pose(sim):
    model = sim.model
    data  = sim.data
    name2jid = {model.joint(i).name: i for i in range(model.njnt)}

    def set_joint(name, val):
        jid  = name2jid[name]
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr] = val

    set_joint("root_x",        0.0)
    set_joint("root_z",        0.0)
    set_joint("root_pitch",    0.0)
    set_joint("hip_flexion_r", 1.57)
    set_joint("hip_flexion_l", 1.57)
    set_joint("knee_angle_r",  1.75)
    set_joint("knee_angle_l",  1.75)
    set_joint("ankle_angle_r", -0.15)
    set_joint("ankle_angle_l", -0.15)

    data.qvel[:] = 0.0
    sim.forward()


def _run_headless(env, sts_params, use_tsa_full, motor_cfgs_r, motor_cfgs_l,
                  quad_ids_all, quad_ids_r, quad_ids_l,
                  max_steps=6000, max_time: float = 8.0):
    """
    Run one complete STS episode headlessly. Terminates on phase >= 4,
    max_steps, OR sim time exceeding max_time (whichever comes first).

    Returns:
        times       : (T,)   simulation time at each step
        phases      : (T,)   STS phase at each step
        acts_all    : (T, N) per-muscle activation (all quads)
        acts_r      : (T, N) right-leg quads only
        acts_l      : (T, N) left-leg quads only
        completed   : bool   True if Phase 4 was reached
    """
    env.reset(seed=0)
    # set_seated_pose(env.sim)

    sts_h = SitToStandSim(
        env.sim, env,
        params              = sts_params,
        debug               = False,
        use_tsa_full        = use_tsa_full,
        use_tsa             = False,
        tsa_motor_configs_r = motor_cfgs_r if use_tsa_full else None,
        tsa_motor_configs_l = motor_cfgs_l if use_tsa_full else None,
        log_to_csv          = False,
    )
    sts_h.reset_filters()
    sts_h.get_observation()
    sts_h.capture_phase1_hold_pose()
    sts_h.reset_phase(1)
    if sts_h.tsa is not None:
        sts_h.tsa.reset()

    times, phases = [], []
    acts_all, acts_r, acts_l = [], [], []
    completed = False

    for _ in range(max_steps):
        sts_h.get_observation()
        phase = sts_h.get_phase()
        sts_h.step(None, phase)

        t_now = float(env.sim.data.time)

        if phase >= 4:
            completed = True
            break
        if t_now >= max_time:
            break

        times.append(t_now)
        phases.append(int(phase))
        acts_all.append([float(env.sim.data.act[i]) for i in quad_ids_all])
        acts_r.append([float(env.sim.data.act[i]) for i in quad_ids_r])
        acts_l.append([float(env.sim.data.act[i]) for i in quad_ids_l])

    sts_h.close()

    return (
        np.array(times),
        np.array(phases),
        np.array(acts_all),
        np.array(acts_r),
        np.array(acts_l),
        completed,
    )


def _print_summary(t_base, ph_base, acts_base, base_completed,
                   t_exo,  ph_exo,  acts_exo,  exo_completed):
    """Print a console comparison of mean quad activation over a matched time window."""
    t_base_end = float(t_base[-1]) if len(t_base) else float("nan")
    t_exo_end  = float(t_exo[-1])  if len(t_exo)  else float("nan")

    # Align comparison to the shorter window so we're comparing the same task interval.
    t_window = min(t_base_end, t_exo_end)
    mask_base = t_base <= t_window
    mask_exo  = t_exo  <= t_window

    acts_base_w = acts_base[mask_base]
    acts_exo_w  = acts_exo [mask_exo]
    ph_base_w   = ph_base[mask_base]
    ph_exo_w    = ph_exo [mask_exo]

    print("\n" + "=" * 68)
    print("QUADRICEPS ACTIVATION COMPARISON  (Phases 1–3 only)")
    print("=" * 68)
    print(f"  Baseline: duration={t_base_end:.2f}s  completed={'YES' if base_completed else 'NO — stuck in Phase ' + str(ph_base[-1])}")
    print(f"  TSA Exo:  duration={t_exo_end:.2f}s  completed={'YES' if exo_completed else 'NO — stopped at Phase ' + str(ph_exo[-1])}")
    print(f"  Comparison window: 0 – {t_window:.2f}s")
    print("-" * 68)
    print(f"{'':14s}  {'Baseline':>10s}  {'TSA Exo':>10s}  {'Δ exo−base':>10s}  {'Duration base':>13s}  {'Duration exo':>12s}")
    print("-" * 68)

    m_base = acts_base_w.mean()
    m_exo  = acts_exo_w.mean()
    pct    = 100 * (m_exo - m_base) / (m_base + 1e-9)
    print(f"{'Overall':14s}  {m_base:10.4f}  {m_exo:10.4f}  {pct:+9.1f}%")

    for p in sorted(set(ph_base) | set(ph_exo)):
        mb = acts_base_w[ph_base_w == p].mean() if (ph_base_w == p).any() else float("nan")
        me = acts_exo_w [ph_exo_w  == p].mean() if (ph_exo_w  == p).any() else float("nan")
        pct_p = 100 * (me - mb) / (mb + 1e-9) if np.isfinite(mb) and np.isfinite(me) else float("nan")
        # Phase durations (full episode, not windowed)
        dur_b = float((t_base[ph_base == p][-1] - t_base[ph_base == p][0])) if (ph_base == p).any() else float("nan")
        dur_e = float((t_exo [ph_exo  == p][-1] - t_exo [ph_exo  == p][0])) if (ph_exo  == p).any() else float("nan")
        b_str = f"{mb:10.4f}" if np.isfinite(mb) else "       n/a"
        e_str = f"{me:10.4f}" if np.isfinite(me) else "       n/a"
        r_str = f"{pct_p:+9.1f}%" if np.isfinite(pct_p) else "      n/a"
        d_b   = f"{dur_b:8.2f}s" if np.isfinite(dur_b) else "      n/a"
        d_e   = f"{dur_e:8.2f}s" if np.isfinite(dur_e) else "      n/a"
        print(f"  Phase {p}        {b_str}  {e_str}  {r_str}  {d_b:>13s}  {d_e:>12s}")

    print("=" * 68 + "\n")


def _plot_comparison(t_base, ph_base, acts_r_base, acts_l_base,
                     t_exo,  ph_exo,  acts_r_exo,  acts_l_exo,
                     out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available — skipping comparison plot")
        return

    PHASE_COLORS = {1: "#aec6e8", 2: "#b5e0b5", 3: "#f5c6a0", 4: "#d4b8e0"}

    def _shade_phases(ax, t_arr, ph_arr):
        for p, c in PHASE_COLORS.items():
            mask = ph_arr == p
            if not mask.any():
                continue
            t_p = t_arr[mask]
            ax.axvspan(t_p[0], t_p[-1], alpha=0.25, color=c, label=f"P{p}" if ax == axes[0] else "")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Quadriceps Activation: Baseline vs TSA Exoskeleton  (Phases 1–3)",
                 fontsize=13, fontweight="bold")

    datasets = [
        # (ax, title, r_base, l_base, r_exo, l_exo)
        (axes[0], "Mean — All Quads (bilateral)",
         acts_r_base, acts_l_base, acts_r_exo, acts_l_exo),
        (axes[1], "Mean — Right Leg Quads",
         acts_r_base, None, acts_r_exo, None),
        (axes[2], "Mean — Left Leg Quads",
         acts_l_base, None, acts_l_exo, None),
    ]

    # Compute matched time window for the reduction label.
    t_window = min(t_base[-1] if len(t_base) else 0, t_exo[-1] if len(t_exo) else 0)

    for ax, title, r_base, l_base, r_exo, l_exo in datasets:
        def _mean(*arrs):
            valid = [a for a in arrs if a is not None]
            return np.concatenate(valid, axis=1).mean(axis=1) if valid else np.array([])

        m_base = _mean(r_base, l_base)
        m_exo  = _mean(r_exo,  l_exo)

        _shade_phases(ax, t_base, ph_base)
        ax.plot(t_base, m_base, color="steelblue",  lw=1.8, label="Baseline (no exo)")
        ax.plot(t_exo,  m_exo,  color="darkorange", lw=1.8, label="With TSA exo")

        # Reduction over matched window only.
        mb_w = m_base[t_base <= t_window].mean() if (t_base <= t_window).any() else float("nan")
        me_w = m_exo [t_exo  <= t_window].mean() if (t_exo  <= t_window).any() else float("nan")
        overall_red = 100 * (me_w - mb_w) / (mb_w + 1e-9) if np.isfinite(mb_w) and np.isfinite(me_w) else float("nan")
        red_str = f"{overall_red:+.1f}%" if np.isfinite(overall_red) else "n/a"
        ax.set_title(f"{title}   (matched-window reduction: {red_str})", fontsize=10)
        ax.set_ylabel("Activation", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    # Phase legend on top panel only
    from matplotlib.patches import Patch
    phase_handles = [Patch(facecolor=c, alpha=0.4, label=f"Phase {p}")
                     for p, c in PHASE_COLORS.items()]
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + phase_handles,
                   labels=axes[0].get_legend_handles_labels()[1] + [f"Phase {p}" for p in PHASE_COLORS],
                   fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved → {out_path}")


# ── Resolve motor configs ─────────────────────────────────────────────────────

if POLICY_PATH is not None:
    _motor_cfgs, _tsa_L = _load_policy_params(POLICY_PATH)
    _sts_params  = STSReflexParams(tsa_string_length=_tsa_L)
    _tsa_motor_r = _motor_cfgs
    _tsa_motor_l = _motor_cfgs
    _p = Path(POLICY_PATH)
    _log_tag = f"{_p.parent.name}_{_p.stem}"
else:
    _sts_params  = STSReflexParams()
    _tsa_motor_r = None
    _tsa_motor_l = None
    _log_tag     = None

# ── Environment ───────────────────────────────────────────────────────────────

env = gym.make('TorsoLegs')
env.reset(seed=0)

max_steps = 50000

# Resolve quad actuator IDs once (model is available after gym.make).
_quad_ids_all = _get_quad_ids(env.sim.model, _QUAD_NAMES)
_quad_ids_r   = _get_quad_ids(env.sim.model, _QUAD_NAMES_R)
_quad_ids_l   = _get_quad_ids(env.sim.model, _QUAD_NAMES_L)

# ── Baseline run (headless) ───────────────────────────────────────────────────

print("Running baseline (no exo) headlessly...")
t_base, ph_base, acts_base_all, acts_base_r, acts_base_l, _base_done = _run_headless(
    env,
    sts_params    = _sts_params,
    use_tsa_full  = False,
    motor_cfgs_r  = None,
    motor_cfgs_l  = None,
    quad_ids_all  = _quad_ids_all,
    quad_ids_r    = _quad_ids_r,
    quad_ids_l    = _quad_ids_l,
    max_time      = 8.0,
)
_base_status = "completed" if _base_done else f"did NOT complete (stuck in Phase {ph_base[-1]})"
print(f"Baseline {_base_status} — duration: {t_base[-1]:.2f}s, phases reached: {sorted(set(ph_base))}")

# ── Exo run setup ─────────────────────────────────────────────────────────────

env.reset(seed=0)

sts = SitToStandSim(
    env.sim, env,
    params              = _sts_params,
    debug               = True,
    use_tsa_full        = True,
    use_tsa             = False,
    tsa_motor_configs_r = _tsa_motor_r,
    tsa_motor_configs_l = _tsa_motor_l,
    log_tag             = _log_tag,
)

# set_seated_pose(env.sim)
sts.reset_filters()
sts.get_observation()
sts.capture_phase1_hold_pose()
sts.reset_phase(1)
if sts.tsa is not None:
    sts.tsa.reset()

# Exo run data collectors.
_t_exo:   list = []
_ph_exo:  list = []
_acts_exo_all: list = []
_acts_exo_r:   list = []
_acts_exo_l:   list = []


def _shutdown(_sig=None, _frame=None):
    try:
        sts.close()
    except Exception:
        pass

    t_exo       = np.array(_t_exo)
    ph_exo      = np.array(_ph_exo)
    acts_exo_all = np.array(_acts_exo_all) if _acts_exo_all else np.zeros((1, len(_quad_ids_all)))
    acts_exo_r   = np.array(_acts_exo_r)   if _acts_exo_r   else np.zeros((1, len(_quad_ids_r)))
    acts_exo_l   = np.array(_acts_exo_l)   if _acts_exo_l   else np.zeros((1, len(_quad_ids_l)))

    _exo_done = len(ph_exo) > 0 and int(ph_exo[-1]) >= 4

    if len(t_exo) > 0:
        _print_summary(t_base, ph_base, acts_base_all, _base_done,
                       t_exo,  ph_exo,  acts_exo_all,  _exo_done)

        plot_out = Path(__file__).resolve().parent / "logs" / "optimised" / f"{_log_tag or 'default'}_quad_comparison.png"
        _plot_comparison(
            t_base, ph_base, acts_base_r, acts_base_l,
            t_exo,  ph_exo,  acts_exo_r,  acts_exo_l,
            plot_out,
        )

    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ── Exo visual run ────────────────────────────────────────────────────────────

print("\nStarting exo visual run (close window or Ctrl+C to exit and plot)...")

try:
    for step_i in range(max_steps):
        env.mj_render()
        obs   = sts.get_observation()
        phase = sts.get_phase()
        sts.step(None, phase)

        if phase < 4:
            _t_exo.append(float(env.sim.data.time))
            _ph_exo.append(int(phase))
            _acts_exo_all.append([float(env.sim.data.act[i]) for i in _quad_ids_all])
            _acts_exo_r.append([float(env.sim.data.act[i]) for i in _quad_ids_r])
            _acts_exo_l.append([float(env.sim.data.act[i]) for i in _quad_ids_l])

except (KeyboardInterrupt, SystemExit):
    pass

_shutdown()
