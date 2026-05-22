from myosuite.utils import gym
import numpy as np
import csv
import os
from pathlib import Path

# ============================================================
# Environment
# ============================================================
env = gym.make("TorsoLegs")
env.reset()
sim = env.sim
model = sim.model
data = sim.data

print("model nq  =", model.nq)
print("model nv  =", model.nv)
print("model nu  =", model.nu)
print("model njnt=", model.njnt)
print("model nsite=", model.nsite)
print("model nbody=", model.nbody)

log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "sts_muscle_log.npz"
ENABLE_STS_LOGGING = os.environ.get("STS_LOGGING", "0") == "0"

actuator_names = [model.actuator(i).name for i in range(model.nu)]


class STSLogger:
    def __init__(self, log_path, actuator_names, enabled=True):
        self.log_path = Path(log_path)
        self.csv_path = self.log_path.with_suffix(".csv")
        self.actuator_names = np.array(actuator_names, dtype=object)
        self.enabled = enabled
        self.frames = []
        self.times = []
        self.activations = []
        self.forces = []
        self.controls = []

    def record(self, sim):
        if not self.enabled:
            return
        self.frames.append(len(self.frames))
        self.times.append(float(sim.data.time))
        self.activations.append(sim.data.act.copy() if sim.model.na > 0 else np.array([], dtype=float))
        self.forces.append(sim.data.actuator_force.copy())
        self.controls.append(sim.data.ctrl.copy())

    def save(self):
        if not self.enabled:
            return

        frames = np.asarray(self.frames, dtype=int)
        times = np.asarray(self.times, dtype=float)
        activations = np.asarray(self.activations, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        act_width = activations.shape[1] if activations.ndim == 2 else 0
        force_width = forces.shape[1] if forces.ndim == 2 else 0
        ctrl_width = controls.shape[1] if controls.ndim == 2 else 0

        np.savez_compressed(
            self.log_path,
            frame=frames,
            time=times,
            act=activations,
            actuator_force=forces,
            ctrl=controls,
            actuator_names=self.actuator_names,
        )

        header = ["frame", "time"]
        header += [f"act_{index}" for index in range(act_width)]
        header += [f"force_{index}" for index in range(force_width)]
        header += [f"force_abs_{index}" for index in range(force_width)]
        header += [f"ctrl_{index}" for index in range(ctrl_width)]

        with self.csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for index, frame in enumerate(frames):
                row = [int(frame), float(times[index])]
                row.extend(activations[index].tolist() if activations.ndim > 1 else [])
                row.extend(forces[index].tolist())
                row.extend(np.abs(forces[index]).tolist())
                row.extend(controls[index].tolist())
                writer.writerow(row)

        print(f"Saved muscle log to {self.log_path}")
        print(f"Saved readable CSV log to {self.csv_path}")


sts_logger = STSLogger(log_path, actuator_names, enabled=ENABLE_STS_LOGGING)

# ============================================================
# Utilities
# ============================================================
def lerp(a, b, t):
    return (1.0 - t) * a + t * b

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3

def deg(x):
    return np.deg2rad(x)

# Joint mapping
name2jid = {model.joint(i).name: i for i in range(model.njnt)}
qadr = model.jnt_qposadr

def has_joint(name):
    return name in name2jid

def qidx(name):
    return int(qadr[name2jid[name]])

def set_joint(q, name, value):
    if has_joint(name):
        q[qidx(name)] = value

def clamp_limits(q):
    for i in range(model.njnt):
        if model.jnt_limited[i]:
            adr = int(model.jnt_qposadr[i])
            lo, hi = model.jnt_range[i]
            q[adr] = np.clip(q[adr], lo, hi)
    return q

# ============================================================
# Parse qpos safely
# ============================================================
def parse_qpos_string(qpos_str, nq, label):
    vals = np.fromstring(qpos_str, sep=" ")
    if len(vals) == nq:
        return vals

    # common copy/paste issue: one extra trailing zero
    if len(vals) == nq + 1 and abs(vals[-1]) < 1e-12:
        print(f"[warn] {label}: had {len(vals)} values, trimming trailing zero to match nq={nq}")
        return vals[:-1]

    raise ValueError(
        f"{label} has length {len(vals)}, but model.nq = {nq}. "
        f"Please fix the qpos string."
    )

# ============================================================
# Your sit / stand qpos
# NOTE:
# - Sitting qpos is from your Python snippet
# - Standing qpos is from your XML keyframe
# - If one has one extra trailing zero, parser trims it
# ============================================================
q_sit_str = """
-0.025 0.25 0.75 0.707388 0 0 -0.706825
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0
-0.025 0.25 0.75 0.707388 0 0 -0.706825
1.57 0 0 0 0 1.3 0 0 0 0 0 0 0 0 1.57 0 0 0 0 1.3 0 0 0 0 0 0 0 0
"""

q_stand_str = """
-0.025 0.1 0.935 0.707388 0 0 -0.706825
0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0
-0.025 0.1 0.935 0.707388 0 0 -0.706825
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""

q_sit = parse_qpos_string(q_sit_str, model.nq, "q_sit")
q_stand = parse_qpos_string(q_stand_str, model.nq, "q_stand")

print("len(q_sit)   =", len(q_sit))
print("len(q_stand) =", len(q_stand))

# ============================================================
# Root z indices for floating-base lift correction
# We detect all free joints and store their z positions.
# For a free joint, qpos layout is [x y z qw qx qy qz].
# ============================================================
FREE_JOINT = 0  # MuJoCo mjJNT_FREE

root_z_indices = []
for j in range(model.njnt):
    if model.jnt_type[j] == FREE_JOINT:
        adr = int(model.jnt_qposadr[j])
        root_z_indices.append(adr + 2)

print("Detected free-joint z indices:", root_z_indices)

if not root_z_indices:
    # fallback: assume first free root z is qpos[2]
    root_z_indices = [2]
    print("[warn] No free joint detected from model; falling back to qpos[2] for root z correction")

# ============================================================
# Optional torso / lumbar shaping
# Negative values bend forward in your model
# ============================================================
def add_forward_lean(q, torso_deg=None, lumbar_total_deg=None):
    q = q.copy()

    if torso_deg is not None and has_joint("flex_extension"):
        set_joint(q, "flex_extension", deg(torso_deg))

    if lumbar_total_deg is not None:
        lumbar_joints = ["L1_L2_FE", "L2_L3_FE", "L3_L4_FE", "L4_L5_FE"]
        each = deg(lumbar_total_deg) / 4.0
        for jn in lumbar_joints:
            if has_joint(jn):
                set_joint(q, jn, each)

    return clamp_limits(q)

# ============================================================
# Candidate foot sites / bodies for floor penetration detection
# Auto-detect by name substrings so you don't have to hardcode blindly
# ============================================================
site_names = [model.site(i).name for i in range(model.nsite)]
body_names = [model.body(i).name for i in range(model.nbody)]

# print("\nAvailable site names:")
# for s in site_names:
#     print("  ", s)

# print("\nAvailable body names:")
# for b in body_names:
#     print("  ", b)

def find_candidate_sites():
    wanted = ["heel", "toe", "foot", "ankle", "mtp"]
    out = []
    for s in site_names:
        sl = s.lower()
        if any(w in sl for w in wanted):
            out.append(s)
    return out

def find_candidate_bodies():
    wanted = ["talus", "calcn", "foot", "toe", "ankle", "mtp"]
    out = []
    for b in body_names:
        bl = b.lower()
        if any(w in bl for w in wanted):
            out.append(b)
    return out

foot_site_candidates = find_candidate_sites()
foot_body_candidates = find_candidate_bodies()

# print("\nDetected foot site candidates:", foot_site_candidates)
# print("Detected foot body candidates:", foot_body_candidates)

support_root_name = None
if has_joint("root"):
    support_root_name = "root"
else:
    for joint_name, joint_id in name2jid.items():
        if model.jnt_type[joint_id] == FREE_JOINT:
            support_root_name = joint_name
            break

torso_root_name = "full_body" if has_joint("full_body") else None

# ============================================================
# Foot penetration correction
# - evaluates current pose
# - finds lowest foot site/body z
# - lifts free root(s) if below floor
# ============================================================
def get_min_foot_z():
    min_z = np.inf

    # Prefer sites if present
    for s in foot_site_candidates:
        try:
            sid = model.site_name2id(s)
            z = data.site_xpos[sid][2]
            min_z = min(min_z, z)
        except Exception:
            pass

    # Fallback to body origins if no useful sites
    if not np.isfinite(min_z):
        for b in foot_body_candidates:
            try:
                bid = model.body_name2id(b)
                z = data.xpos[bid][2]
                min_z = min(min_z, z)
            except Exception:
                pass

    return min_z

def lift_root_if_feet_penetrate(state, floor_z=0.0, margin=0.003):
    env.set_env_state(state)
    sim.forward()

    min_z = get_min_foot_z()

    if np.isfinite(min_z) and min_z < floor_z + margin:
        dz = (floor_z + margin) - min_z
        for idx in root_z_indices:
            state["qpos"][idx] += dz

    return state

# ============================================================
# Build 4 STS phase keyframes
# We interpolate the FULL qpos to include translation.
# Then we add torso/lumbar shaping for realism.
# ============================================================
u1 = 0.20   # end phase 1: momentum forward
u2 = 0.38   # end phase 2: momentum transfer / seat-off
u3 = 0.88   # end phase 3: extension
u4 = 1.00   # end phase 4: stabilisation

q_phase0 = q_sit.copy()
q_phase1 = lerp(q_sit, q_stand, u1)
q_phase2 = lerp(q_sit, q_stand, u2)
q_phase3 = lerp(q_sit, q_stand, u3)
q_phase4 = q_stand.copy()

# Forward-lean shaping for upper body (negative = forward)
q_phase1 = add_forward_lean(q_phase1, torso_deg=-15, lumbar_total_deg=-12)
q_phase2 = add_forward_lean(q_phase2, torso_deg=-30, lumbar_total_deg=-20)
q_phase3 = add_forward_lean(q_phase3, torso_deg=-8,  lumbar_total_deg=-6)
q_phase4 = add_forward_lean(q_phase4, torso_deg=0,   lumbar_total_deg=0)

# Optional: keep initial upright sit posture exactly upright in torso
q_phase0 = add_forward_lean(q_phase0, torso_deg=0, lumbar_total_deg=0)

def copy_free_joint_pose(qpos, source_name, target_name):
    if source_name is None or target_name is None:
        return qpos

    source_start = qidx(source_name)
    target_start = qidx(target_name)
    qpos[target_start:target_start + 7] = qpos[source_start:source_start + 7]
    return qpos

# Clamp any limited joints
for q in [q_phase0, q_phase1, q_phase2, q_phase3, q_phase4]:
    clamp_limits(q)

# ============================================================
# Animation helper
# ============================================================
state = env.get_env_state()
state["qvel"] = np.zeros_like(state["qvel"])

def play_segment(q_start, q_end, n_frames, floor_z=0.0, margin=0.003):
    for t in range(n_frames):
        u = smoothstep(t / (n_frames - 1))
        state["qpos"] = lerp(q_start, q_end, u)
        state["qvel"] = np.zeros_like(state["qvel"])

        state["qpos"] = copy_free_joint_pose(state["qpos"], support_root_name, torso_root_name)

        # Prevent feet sinking into floor
        state_copy = {
            "qpos": state["qpos"].copy(),
            "qvel": state["qvel"].copy(),
            "act": state["act"].copy() if "act" in state else np.zeros(model.na),
            "ctrl": state["ctrl"].copy() if "ctrl" in state else np.zeros(model.nu),
            "time": state["time"] if "time" in state else 0.0,
            "site_pos": state["site_pos"].copy() if "site_pos" in state else model.site_pos.copy(),
            "site_quat": state["site_quat"].copy() if "site_quat" in state else model.site_quat.copy(),
            "body_pos": state["body_pos"].copy() if "body_pos" in state else model.body_pos.copy(),
            "body_quat": state["body_quat"].copy() if "body_quat" in state else model.body_quat.copy(),
        }

        state_copy = lift_root_if_feet_penetrate(state_copy, floor_z=floor_z, margin=margin)
        env.set_env_state(state_copy)
        sim.forward()
        sts_logger.record(sim)
        env.mj_render()

# ============================================================
# Run 4-phase sit-to-stand
# Slowed down a bit
# ============================================================
# Start seated
state["qpos"] = q_phase0.copy()
state["qvel"] = np.zeros_like(state["qvel"])

state["qpos"] = copy_free_joint_pose(state["qpos"], support_root_name, torso_root_name)

env.set_env_state(state)
sim.forward()
sts_logger.record(sim)

# Hold seated briefly
for _ in range(150):
    sts_logger.record(sim)
    env.mj_render()

# Phase 1: momentum forward
play_segment(q_phase0, q_phase1, n_frames=500)

# Phase 2: momentum transfer / seat-off
play_segment(q_phase1, q_phase2, n_frames=450)

# Phase 3: extension
play_segment(q_phase2, q_phase3, n_frames=1100)

# Phase 4: stabilisation
play_segment(q_phase3, q_phase4, n_frames=600)

# Hold standing
for _ in range(300):
    sts_logger.record(sim)
    env.mj_render()

sts_logger.save()
env.close()