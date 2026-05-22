import numpy as np
from typing import Callable, Dict


class TSASimulator:
    """Single-unit TSA simulator with full forward dynamics (RK4). Tension ≥ 0."""

    def __init__(
        self,
        id: int,
        L: float,
        radius: float,
        payload_mass: float = 12.24,        # kg  (maps to ~120 N gravity load)
        I_motor: float = 5e-5,              # kg·m²
        pretension_theta: float = 2 * np.pi,
        max_motor_torque: float = 0.1275,   # Pololu 9.7:1 LP 12V stall torque
        no_load_speed: float = 520.0,       # rad/s  (~4966 RPM)
        b_theta: float = 1e-4,             # motor viscous friction [N·m·s/rad]
        b_X: float = 0.0,                  # payload viscous friction [N·s/m]
        gravity_along_string: bool = True,
        max_contraction_ratio: float = 0.30,
    ):
        self.id = id
        self.L = L
        self.r = radius
        self.m = payload_mass
        self.I = I_motor
        self.theta_pretension = pretension_theta
        self.max_motor_torque = max_motor_torque
        self.no_load_speed = no_load_speed
        self.b_theta = b_theta
        self.b_X = b_X
        self.g = 9.81
        self.gravity_along_string = gravity_along_string
        self.max_contraction = max_contraction_ratio * L

    # ------------------------------------------------------------------
    # Kinematics  (Section 1.1 / 1.4)
    # ------------------------------------------------------------------

    def _contraction(self, theta: float) -> float:
        """X(θ) = L - √(L² - θ²r²)  —  forward kinematics eq. 1.3"""
        arg = self.L**2 - (theta * self.r)**2
        arg = max(arg, 0.0)
        return min(self.L - np.sqrt(arg), self.max_contraction)

    def _jacobian(self, X: float) -> float:
        """J(X) = r·√[(2L-X)X] / (L-X)  —  task-space Jacobian Section 1.4"""
        num = self.r * np.sqrt(max((2*self.L - X) * X, 0.0))
        den = max(self.L - X, 1e-10)
        return num / den

    def _jacobian_dot(self, J: float, X: float, X_dot: float) -> float:
        """J̇ = (L²r²)/(L-X)³ · J⁻¹ · Ẋ  —  Section 1.4"""
        den = max((self.L - X)**3, 1e-20)
        J_inv = 1.0 / J if abs(J) > 1e-12 else 0.0
        return (self.L**2 * self.r**2) / den * J_inv * X_dot

    # ------------------------------------------------------------------
    # Motor torque-speed curve
    # ------------------------------------------------------------------

    def _tau_available(self, theta_dot: float) -> float:
        """Linear torque-speed characteristic (stall → no-load)."""
        speed = abs(theta_dot)
        if self.no_load_speed <= 1e-10:
            return self.max_motor_torque
        return self.max_motor_torque * max(0.0, 1.0 - speed / self.no_load_speed)

    # ------------------------------------------------------------------
    # Equations of motion  (Section 1.3 / 1.32)
    # ------------------------------------------------------------------

    def _eom(
        self,
        theta: float,
        theta_dot: float,
        tau_cmd: float,
        F_ext_override: float = None,
    ) -> tuple[float, float, float]:
        """
        Solve for θ̈, T, Ẍ. D_θ = I + mJ² (eq. 1.22). F_ext_override bypasses
        gravity_along_string so callers can pass joint resistance directly.
        """
        X     = self._contraction(theta)
        J     = self._jacobian(X)
        X_dot = J * theta_dot
        J_dot = self._jacobian_dot(J, X, X_dot)

        if F_ext_override is not None:
            F_ext = F_ext_override
        else:
            F_ext = self.m * self.g if self.gravity_along_string else 0.0

        # Effective inertia D_θ = I + m·J²  (eq. 1.22)
        D_theta = self.I + self.m * J**2

        # Right-hand side
        rhs = (tau_cmd
               - J * (self.m * J_dot * theta_dot + F_ext + self.b_X * J * theta_dot)
               - self.b_theta * theta_dot)

        theta_ddot = rhs / D_theta

        # Payload acceleration and tension
        X_ddot = J * theta_ddot + J_dot * theta_dot
        T = self.m * X_ddot + F_ext + self.b_X * X_dot
        T = max(T, 0.0)   # strings can only pull

        return theta_ddot, T, X, J, X_dot, J_dot, X_ddot

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        tau_input: Callable[[float], float],
        t_end: float,
        dt: float = 0.001,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """RK4 integration of motor dynamics. Returns dict of time-series arrays."""
        n = int(np.ceil(t_end / dt)) + 1
        t_arr      = np.linspace(0.0, t_end, n)

        theta_arr  = np.zeros(n)
        tdot_arr   = np.zeros(n)
        X_arr      = np.zeros(n)
        Xdot_arr   = np.zeros(n)
        Xddot_arr  = np.zeros(n)
        J_arr      = np.zeros(n)
        Jdot_arr   = np.zeros(n)
        tau_cmd_arr   = np.zeros(n)
        tau_avail_arr = np.zeros(n)
        tau_req_arr   = np.zeros(n)
        T_arr      = np.zeros(n)
        stalled    = np.zeros(n, dtype=bool)

        theta_arr[0] = self.theta_pretension
        tdot_arr[0]  = 0.0
        X0 = self._contraction(self.theta_pretension)
        J0 = self._jacobian(X0)
        X_arr[0]     = X0
        J_arr[0]     = J0
        tau_avail_arr[0] = self._tau_available(0.0)
        # Required torque to hold the load statically: τ = J·F_ext
        F_ext0 = self.m * self.g if self.gravity_along_string else 0.0
        tau_req_arr[0] = J0 * F_ext0
        T_arr[0]     = F_ext0   # static: T = mg

        def derivatives(theta: float, theta_dot: float, tau: float):
            """Return (dθ/dt, dθ̇/dt) = (θ̇, θ̈)."""
            tau_a = self._tau_available(theta_dot)
            tau_c = min(tau, tau_a)          # motor cannot exceed available torque
            tau_c = max(tau_c, 0.0)          # motor does not pull backwards here
            tddot, T, X, J, Xdot, Jdot, Xddot = self._eom(theta, theta_dot, tau_c)
            return theta_dot, tddot, T, X, J, Xdot, Jdot, Xddot, tau_c, tau_a

        for i in range(1, n):
            t_now  = t_arr[i - 1]
            th     = theta_arr[i - 1]
            thd    = tdot_arr[i - 1]
            tau    = tau_input(t_now)

            # -- k1 --
            _, k1_thd, T1, X1, J1, Xd1, Jd1, Xdd1, tc1, ta1 = derivatives(th, thd, tau)
            k1_th = thd

            # -- k2 --
            th2  = th  + 0.5*dt*k1_th
            thd2 = thd + 0.5*dt*k1_thd
            th2  = max(th2, self.theta_pretension)
            _, k2_thd, T2, X2, J2, Xd2, Jd2, Xdd2, tc2, ta2 = derivatives(th2, thd2, tau)
            k2_th = thd2

            # -- k3 --
            th3  = th  + 0.5*dt*k2_th
            thd3 = thd + 0.5*dt*k2_thd
            th3  = max(th3, self.theta_pretension)
            _, k3_thd, T3, X3, J3, Xd3, Jd3, Xdd3, tc3, ta3 = derivatives(th3, thd3, tau)
            k3_th = thd3

            # -- k4 --
            th4  = th  + dt*k3_th
            thd4 = thd + dt*k3_thd
            th4  = max(th4, self.theta_pretension)
            _, k4_thd, T4, X4, J4, Xd4, Jd4, Xdd4, tc4, ta4 = derivatives(th4, thd4, tau)
            k4_th = thd4

            # -- Weighted average --
            theta_new = th  + (dt/6)*(k1_th  + 2*k2_th  + 2*k3_th  + k4_th)
            tdot_new  = thd + (dt/6)*(k1_thd + 2*k2_thd + 2*k3_thd + k4_thd)

            # Physical constraints
            theta_new = max(theta_new, self.theta_pretension)
            tdot_new  = max(tdot_new,  0.0)   # no unwinding

            # Final state quantities at new point
            _, _, T_f, X_f, J_f, Xd_f, Jd_f, Xdd_f, tc_f, ta_f = derivatives(
                theta_new, tdot_new, tau
            )

            theta_arr[i] = theta_new
            tdot_arr[i]  = tdot_new
            X_arr[i]     = X_f
            Xdot_arr[i]  = Xd_f
            Xddot_arr[i] = Xdd_f
            J_arr[i]     = J_f
            Jdot_arr[i]  = Jd_f
            T_arr[i]     = T_f
            tau_cmd_arr[i]   = tc_f
            tau_avail_arr[i] = ta_f

            F_ext_i = self.m * self.g if self.gravity_along_string else 0.0
            tau_req_arr[i] = J_f * F_ext_i   # quasi-static required torque

            stalled[i] = ta_f <= tau_req_arr[i]

            if verbose and i % max(1, n // 10) == 0:
                print(
                    f"t={t_arr[i]:.3f}s | θ={theta_new:.2f}rad "
                    f"({theta_new/(2*np.pi):.1f} rot) | "
                    f"X={X_f*1e3:.2f}mm | T={T_f:.1f}N | "
                    f"τ_cmd={tc_f:.4f} τ_avail={ta_f:.4f} Nm"
                )

        return {
            't':             t_arr,
            'theta':         theta_arr,
            'theta_dot':     tdot_arr,
            'X':             X_arr,
            'X_dot':         Xdot_arr,
            'X_ddot':        Xddot_arr,
            'J':             J_arr,
            'J_dot':         Jdot_arr,
            'tau_cmd':       tau_cmd_arr,
            'tau_available': tau_avail_arr,
            'tau_required':  tau_req_arr,
            'T':             T_arr,
            'stalled':       stalled,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self, results: Dict[str, np.ndarray], title_suffix: str = ""):
        import matplotlib.pyplot as plt
        theta = results['theta']
        t     = results['t']

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(
            f"TSA Simulation Results  {title_suffix}\n"
            f"L={self.L*100:.0f}cm  r={self.r*1e3:.1f}mm  "
            f"m={self.m:.1f}kg  I={self.I:.1e}kg·m²",
            fontsize=12, fontweight='bold'
        )

        # Row 0: kinematics vs time
        axes[0, 0].plot(t, results['X']*1e3, lw=1.8, color='steelblue')
        axes[0, 0].set_ylabel('Contraction (mm)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_title('Contraction vs Time')
        axes[0, 0].grid(True, alpha=0.4)

        axes[0, 1].plot(t, results['theta']/(2*np.pi), lw=1.8, color='darkorange')
        axes[0, 1].set_ylabel('Motor angle (rotations)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_title('Motor Angle vs Time')
        axes[0, 1].grid(True, alpha=0.4)

        # Row 1: velocity and torque vs time
        axes[1, 0].plot(t, results['theta_dot'], lw=1.8, color='green')
        axes[1, 0].set_ylabel('Motor angular velocity (rad/s)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_title('Motor Speed vs Time')
        axes[1, 0].grid(True, alpha=0.4)

        axes[1, 1].plot(t, results['tau_required']*1e3,  lw=1.8, label='Required (static)', color='steelblue')
        axes[1, 1].plot(t, results['tau_available']*1e3, lw=1.8, label='Available',          color='darkorange')
        axes[1, 1].plot(t, results['tau_cmd']*1e3,       lw=1.5, label='Commanded',          color='purple', ls='--')
        axes[1, 1].axhline(self.max_motor_torque*1e3, color='r', ls='--', lw=1, label='Stall torque')
        axes[1, 1].set_ylabel('Torque (mN·m)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_title('Torque vs Time')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.4)

        # Row 2: tension and tension vs contraction
        stall_idx = np.where(results['stalled'])[0]
        axes[2, 0].plot(t, results['T'], lw=1.8, color='crimson', label='Tension T')
        if stall_idx.size:
            axes[2, 0].scatter(
                t[stall_idx], results['T'][stall_idx],
                color='k', s=10, zorder=5, label='Torque saturated'
            )
        axes[2, 0].set_ylabel('String Tension (N)')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_title('String Tension vs Time')
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.4)
        axes[2, 0].axhline(0, color='k', lw=0.8, ls='--')

        axes[2, 1].plot(results['X']*1e3, results['T'], lw=1.8, color='teal')
        axes[2, 1].set_xlabel('Contraction (mm)')
        axes[2, 1].set_ylabel('String Tension (N)')
        axes[2, 1].set_title('Tension vs Contraction  (optimisation target)')
        axes[2, 1].grid(True, alpha=0.4)
        axes[2, 1].axhline(0, color='k', lw=0.8, ls='--')

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_constant_torque():
    sim = TSASimulator(
        id=1,
        L=0.5,
        radius=0.004,
        # payload_mass=12.24,
        payload_mass=5.0,
        I_motor=5e-5,
        pretension_theta=2*np.pi,
        max_motor_torque=0.1275,
        no_load_speed=520.0,
        b_theta=1e-4,
        b_X=0.0,
        gravity_along_string=True,
        max_contraction_ratio=0.30,
    )

    tau_const = lambda t: 0.1275

    results = sim.simulate(tau_const, t_end=3.0, dt=0.001, verbose=True)

    print(f"\n--- Summary ---")
    print(f"Max contraction:  {results['X'].max()*1e3:.2f} mm  "
          f"(limit {sim.max_contraction*1e3:.1f} mm)")
    print(f"Peak tension:     {results['T'].max():.2f} N")
    print(f"Peak motor speed: {results['theta_dot'].max():.1f} rad/s")
    print(f"Peak τ_required:  {results['tau_required'].max()*1e3:.2f} mN·m")
    print(f"Torque saturated: {results['stalled'].any()}")

    import matplotlib.pyplot as plt
    fig = sim.plot_results(results, "(Constant stall torque, 3 s)")
    fig.savefig("images/tsa_constant_torque.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    test_constant_torque()