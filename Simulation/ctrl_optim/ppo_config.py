"""
TSA PPO optimisation config.

Action vector (symmetric, 5-D): θ = [L, t0, t1, t2, t3]
t0 ≤ t1 ≤ t2 ≤ t3 enforced by sorting in the wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TSAOptimConfig:
    """Top-level config for TSA PPO optimisation."""

    total_timesteps: int = 5000

    # ------------------------------------------------------------------

    @dataclass
    class EnvParams:
        symmetric: bool = True  # True → 5-D action shared across legs; False → 9-D per-leg

        max_steps: int = 600  # ~6 s at dt=0.01 s

        L_min: float = 0.25  # string length bounds [m]
        L_max: float = 0.65

        t_min: float = 0.0   # motor activation time bounds [s]
        # Capped at 0.65 s so all motors fire before phase 4; beyond ~0.6 s activations
        # miss the episode and game the reward.
        t_max: float = 0.65

        num_envs: int = 1

    # ------------------------------------------------------------------

    @dataclass
    class RewardParams:
        w_torque: float = 0.0
        w_muscle: float = 1.0
        w_time: float = 0.0

        support_fraction: float = 0.1   # TSA target = α × knee demand
        torque_ref: float = 20.0        # [N·m] normalisation reference

        quad_muscle_names: List[str] = field(default_factory=lambda: [
            "vaslat_r", "vasmed_r", "vasint_r",
            "vaslat_l", "vasmed_l", "vasint_l",
            "recfem_r", "recfem_l",
        ])

        t_max_episode: float = 6.0  # [s] episode duration cap for time penalty

        slack_penalty: float = 0.1
        wall_penalty: float = 0.0

    # ------------------------------------------------------------------

    @dataclass
    class PPOParams:
        learning_rate: float = 3e-4
        n_steps: int = 64       # must satisfy: n_steps * num_envs >= batch_size
        batch_size: int = 32
        n_epochs: int = 10
        gamma: float = 0.99     # irrelevant for one-step MDP but SB3 requires it
        gae_lambda: float = 0.95
        ent_coef: float = 0.01
        clip_range: float = 0.2
        vf_coef: float = 0.5
        max_grad_norm: float = 0.5
        device: str = "cpu"

    # ------------------------------------------------------------------

    env_params: EnvParams = field(default_factory=EnvParams)
    reward_params: RewardParams = field(default_factory=RewardParams)
    ppo_params: PPOParams = field(default_factory=PPOParams)
