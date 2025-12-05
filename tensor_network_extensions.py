"""
Advanced Extensions for Tensor Network Simulations

This module extends the basic t-DMRG implementation with:
1. Higher-order Trotter decomposition (4th order)
2. Adaptive time stepping
3. Non-linear ramp protocols
4. Additional models (XXZ, Hubbard)
5. Finite temperature methods
6. Finite-size scaling analysis

Author: Claude
Date: 2024
"""

import numpy as np
from scipy.linalg import expm, svd, eigh
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm

from ising_adiabatic_dmrg import (
    MPS, DMRGParameters, Measurements, TransverseFieldIsing,
    GroundStateDMRG, TEBD
)


# ============================================================================
# 1. HIGHER-ORDER TROTTER DECOMPOSITION
# ============================================================================

class TEBDFourthOrder(TEBD):
    """
    Time evolution with 4th-order Trotter-Suzuki decomposition

    The 4th-order decomposition has error O(dt^5) vs O(dt^3) for 2nd order:

    exp(-iHdt) = exp(-ip₁H₁dt)exp(-ip₂H₂dt)...exp(-ip₂H₂dt)exp(-ip₁H₁dt) + O(dt⁵)

    where H = H₁ + H₂ (even/odd bonds) and coefficients are:
    p₁ = 1/(4 - 4^(1/3))
    p₂ = 1 - 4p₁
    """

    def __init__(self, model, params: DMRGParameters):
        super().__init__(model, params)

        # 4th-order Suzuki coefficients
        self.p1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
        self.p2 = 1.0 - 4.0 * self.p1

    def evolve_step(self, mps: MPS, h: float, dt: float):
        """
        Perform one time step with 4th-order Trotter decomposition

        This gives significantly better accuracy at the cost of more gates.
        """
        L = mps.L

        # Define the sequence of evolution steps
        sequence = [
            (self.p1, 'even'),
            (self.p1, 'odd'),
            (self.p2, 'even'),
            (self.p2, 'odd'),
            (self.p2, 'even'),
            (self.p1, 'odd'),
            (self.p1, 'even')
        ]

        for coeff, bond_type in sequence:
            if bond_type == 'even':
                # Even bonds (0, 2, 4, ...)
                for site in range(0, L - 1, 2):
                    U = self._time_evolution_operator(h, coeff * dt, site)
                    self._apply_two_site_gate(mps, U, site)
            else:
                # Odd bonds (1, 3, 5, ...)
                for site in range(1, L - 1, 2):
                    U = self._time_evolution_operator(h, coeff * dt, site)
                    self._apply_two_site_gate(mps, U, site)

        # Normalize to prevent numerical overflow
        norm = mps.norm()
        if norm > 1e-10:
            for i in range(L):
                mps.tensors[i] /= norm**(1.0/L)


# ============================================================================
# 2. ADAPTIVE TIME STEPPING
# ============================================================================

class AdaptiveTEBD(TEBD):
    """
    Time evolution with adaptive time stepping

    The time step is automatically adjusted based on:
    - Bond dimension growth rate (indicates entanglement growth)
    - Distance from critical point
    - Energy change rate
    """

    def __init__(self, model, params: DMRGParameters,
                 dt_min: float = 0.001, dt_max: float = 0.1,
                 target_chi_growth: float = 0.1):
        super().__init__(model, params)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.target_chi_growth = target_chi_growth

    def adaptive_dt(self, mps: MPS, h: float,
                   old_chi: Optional[int] = None,
                   old_energy: Optional[float] = None) -> float:
        """
        Compute adaptive time step based on system state

        Parameters:
        -----------
        mps : MPS
            Current state
        h : float
            Current field value
        old_chi : int, optional
            Previous bond dimension
        old_energy : float, optional
            Previous energy

        Returns:
        --------
        dt : float
            Adapted time step
        """
        dt = self.params.dt

        # Factor 1: Distance from critical point
        h_c = 1.0
        distance_factor = min(abs(h - h_c) + 0.1, 1.0)
        dt *= distance_factor

        # Factor 2: Bond dimension growth
        if old_chi is not None:
            current_chi = max(mps.get_bond_dimensions())
            chi_growth = abs(current_chi - old_chi) / max(old_chi, 1)
            if chi_growth > self.target_chi_growth:
                dt *= 0.5  # Reduce time step if chi growing too fast
            elif chi_growth < self.target_chi_growth / 2:
                dt *= 1.5  # Increase time step if chi stable

        # Factor 3: Energy change
        if old_energy is not None:
            current_energy = Measurements.measure_energy(mps, self.model, h)
            energy_change = abs(current_energy - old_energy)
            if energy_change > 1.0:  # Large energy change
                dt *= 0.7

        # Enforce bounds
        dt = np.clip(dt, self.dt_min, self.dt_max)

        return dt


# ============================================================================
# 3. NON-LINEAR RAMP PROTOCOLS
# ============================================================================

@dataclass
class RampProtocol:
    """Base class for field ramping protocols"""
    h_initial: float
    h_final: float
    T_total: float

    def h_of_t(self, t: float) -> float:
        """Return field value at time t"""
        raise NotImplementedError


class LinearRamp(RampProtocol):
    """Linear ramping: h(t) = h_i + (h_f - h_i) * t/T"""

    def h_of_t(self, t: float) -> float:
        return self.h_initial + (self.h_final - self.h_initial) * t / self.T_total


class TanhRamp(RampProtocol):
    """
    Tanh ramping: slower near critical point

    h(t) = h_c + Δh * tanh(a*(t/T - 0.5))

    This spends more time near h_c, reducing excitations.
    """

    def __init__(self, h_initial: float, h_final: float, T_total: float,
                 steepness: float = 3.0):
        super().__init__(h_initial, h_final, T_total)
        self.h_c = 1.0  # Critical point
        self.steepness = steepness

    def h_of_t(self, t: float) -> float:
        # Normalize time to [-1, 1] range
        tau = 2 * t / self.T_total - 1
        # Tanh function centered at t = T/2
        h_center = (self.h_initial + self.h_final) / 2
        h_amplitude = (self.h_final - self.h_initial) / 2
        return h_center + h_amplitude * np.tanh(self.steepness * tau)


class PolynomialRamp(RampProtocol):
    """
    Polynomial ramping: h(t) = h_i + (h_f - h_i) * (t/T)^n

    n > 1: Fast initially, slow near end
    n < 1: Slow initially, fast near end
    """

    def __init__(self, h_initial: float, h_final: float, T_total: float,
                 exponent: float = 2.0):
        super().__init__(h_initial, h_final, T_total)
        self.exponent = exponent

    def h_of_t(self, t: float) -> float:
        tau = t / self.T_total
        return self.h_initial + (self.h_final - self.h_initial) * tau**self.exponent


class OptimalRamp(RampProtocol):
    """
    Optimal ramping based on Kibble-Zurek theory

    Near the critical point, slow down as:
    v(t) ~ |δh|^(1+zν)

    This minimizes defect density.
    """

    def __init__(self, h_initial: float, h_final: float, T_total: float):
        super().__init__(h_initial, h_final, T_total)
        self.h_c = 1.0
        self.z = 1.0  # Dynamical exponent for TFIM
        self.nu = 1.0  # Correlation length exponent for TFIM
        self._build_trajectory()

    def _build_trajectory(self):
        """Pre-compute the optimal trajectory"""
        n_points = 1000
        t_arr = np.linspace(0, self.T_total, n_points)
        h_arr = np.zeros(n_points)

        h_arr[0] = self.h_initial

        for i in range(1, n_points):
            dt = t_arr[i] - t_arr[i-1]
            h_current = h_arr[i-1]

            # Velocity scaling near critical point
            delta_h = abs(h_current - self.h_c)
            if delta_h < 0.01:
                delta_h = 0.01  # Regularization

            # v ~ |δh|^(1+zν)
            velocity = (delta_h)**(1 + self.z * self.nu)
            velocity *= np.sign(self.h_final - self.h_initial)

            # Normalize to reach h_final at T_total
            velocity *= abs(self.h_final - self.h_initial) / self.T_total

            h_arr[i] = h_current + velocity * dt

        # Store interpolated function
        self.t_arr = t_arr
        self.h_arr = h_arr

    def h_of_t(self, t: float) -> float:
        """Interpolate the pre-computed trajectory"""
        return np.interp(t, self.t_arr, self.h_arr)


# ============================================================================
# 4. ADDITIONAL MODELS
# ============================================================================

class XXZModel:
    """
    XXZ Spin-1/2 Chain

    H = Σᵢ [Jₓ(σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ) + Jz σᵢᶻσᵢ₊₁ᶻ + h σᵢᶻ]

    Special cases:
    - Jx = Jz = 1: Heisenberg model
    - Jx = 1, Jz = Δ: XXZ model
    - Jz >> Jx: Ising limit
    - Jx >> Jz: XY limit
    """

    def __init__(self, L: int, Jx: float = 1.0, Jz: float = 1.0):
        self.L = L
        self.Jx = Jx
        self.Jz = Jz

        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)

        # Spin raising/lowering operators
        self.sigma_plus = (self.sigma_x + 1j * self.sigma_y) / 2
        self.sigma_minus = (self.sigma_x - 1j * self.sigma_y) / 2

    def two_site_hamiltonian(self, h: float, site: int) -> np.ndarray:
        """
        Two-site Hamiltonian for XXZ model

        H = Jₓ(S⁺S⁻ + S⁻S⁺) + Jz SᶻSᶻ + (h/2)(Sᶻ⊗I + I⊗Sᶻ)
        """
        # XX + YY terms = 2 * (S⁺S⁻ + S⁻S⁺)
        H = self.Jx * (np.kron(self.sigma_plus, self.sigma_minus) +
                       np.kron(self.sigma_minus, self.sigma_plus))

        # ZZ term
        H += self.Jz * np.kron(self.sigma_z, self.sigma_z)

        # Magnetic field (split between two sites)
        H += (h / 2) * np.kron(self.sigma_z, self.identity)
        H += (h / 2) * np.kron(self.identity, self.sigma_z)

        return H


class FermiHubbardModel:
    """
    Fermi-Hubbard Model on 1D chain

    H = -t Σᵢ,σ (c†ᵢ,σ cᵢ₊₁,σ + h.c.) + U Σᵢ nᵢ,↑ nᵢ,↓ - μ Σᵢ,σ nᵢ,σ

    Parameters:
    - t: hopping amplitude
    - U: onsite interaction
    - μ: chemical potential

    Note: This requires a 4-dimensional local Hilbert space:
    |0⟩ (empty), |↑⟩, |↓⟩, |↑↓⟩ (doubly occupied)
    """

    def __init__(self, L: int, t: float = 1.0, U: float = 4.0, mu: float = 0.0):
        self.L = L
        self.t = t
        self.U = U
        self.mu = mu
        self.d = 4  # Local Hilbert space dimension

        # Basis ordering: |0⟩, |↑⟩, |↓⟩, |↑↓⟩
        # Fermionic operators
        self._build_operators()

    def _build_operators(self):
        """Build fermionic creation/annihilation operators"""
        # Creation operator for spin up
        self.c_up = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=complex)

        # Creation operator for spin down (with Jordan-Wigner string)
        self.c_down = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, -1, 0, 0]  # Note the minus sign for anticommutation
        ], dtype=complex)

        # Number operators
        self.n_up = self.c_up.conj().T @ self.c_up
        self.n_down = self.c_down.conj().T @ self.c_down
        self.n_total = self.n_up + self.n_down

        # Identity
        self.identity = np.eye(4, dtype=complex)

    def two_site_hamiltonian(self, site: int) -> np.ndarray:
        """
        Two-site Hamiltonian for Hubbard model

        H = -t(c†ᵢ,↑cᵢ₊₁,↑ + c†ᵢ,↓cᵢ₊₁,↓ + h.c.) +
            U(nᵢ,↑nᵢ,↓ + nᵢ₊₁,↑nᵢ₊₁,↓) -
            μ(nᵢ + nᵢ₊₁)
        """
        # Hopping terms
        H = -self.t * (
            np.kron(self.c_up.conj().T, self.c_up) +
            np.kron(self.c_up, self.c_up.conj().T) +
            np.kron(self.c_down.conj().T, self.c_down) +
            np.kron(self.c_down, self.c_down.conj().T)
        )

        # Interaction terms
        H += self.U * (
            np.kron(self.n_up @ self.n_down, self.identity) +
            np.kron(self.identity, self.n_up @ self.n_down)
        )

        # Chemical potential
        H -= self.mu * (
            np.kron(self.n_total, self.identity) +
            np.kron(self.identity, self.n_total)
        )

        return H


# ============================================================================
# 5. FINITE TEMPERATURE METHODS
# ============================================================================

class FiniteTemperatureMPS:
    """
    Finite temperature simulations using purification method

    A thermal state ρ = exp(-βH)/Z can be represented as a pure state
    in a doubled Hilbert space: |ψ(β)⟩ = Σᵢ √pᵢ |i⟩⊗|i⟩

    We evolve in imaginary time: ∂|ψ⟩/∂τ = -H|ψ⟩
    """

    def __init__(self, model, params: DMRGParameters):
        self.model = model
        self.params = params

    def thermal_state(self, beta: float, h: float,
                     initial_mps: Optional[MPS] = None) -> MPS:
        """
        Compute thermal state at inverse temperature β = 1/T

        Starting from infinite temperature (β=0), evolve in imaginary time:
        |ψ(β)⟩ = exp(-βH/2)|ψ(0)⟩

        Parameters:
        -----------
        beta : float
            Inverse temperature β = 1/(k_B T)
        h : float
            Transverse field
        initial_mps : MPS, optional
            Initial state (default: infinite temperature)

        Returns:
        --------
        mps : MPS
            Thermal state at temperature T = 1/β
        """
        L = self.model.L

        if initial_mps is None:
            # Start from infinite temperature: maximally mixed state
            # In purification: |ψ⟩ = (1/√2)^L Σ |σ⟩⊗|σ⟩
            mps = MPS(L, chi=self.params.chi_init, random_init=True)
        else:
            mps = initial_mps.copy()

        # Imaginary time evolution
        n_steps = int(beta / self.params.dt)
        d_tau = beta / n_steps

        print(f"Imaginary time evolution: β={beta:.3f}, steps={n_steps}")

        for step in tqdm(range(n_steps)):
            self._imaginary_time_step(mps, h, d_tau)

            # Normalize (important for imaginary time)
            norm = mps.norm()
            for i in range(L):
                mps.tensors[i] /= norm**(1.0/L)

        return mps

    def _imaginary_time_step(self, mps: MPS, h: float, d_tau: float):
        """
        Single imaginary time step: exp(-Hτ)

        This is similar to real time evolution but with purely real
        exponentials (no oscillations).
        """
        L = mps.L

        # Second-order Trotter for imaginary time
        # exp(-Hτ) ≈ exp(-H_even τ/2) exp(-H_odd τ) exp(-H_even τ/2)

        # Even bonds
        for site in range(0, L - 1, 2):
            U = self._imaginary_time_gate(h, d_tau / 2, site)
            self._apply_gate(mps, U, site)

        # Odd bonds
        for site in range(1, L - 1, 2):
            U = self._imaginary_time_gate(h, d_tau, site)
            self._apply_gate(mps, U, site)

        # Even bonds again
        for site in range(0, L - 1, 2):
            U = self._imaginary_time_gate(h, d_tau / 2, site)
            self._apply_gate(mps, U, site)

    def _imaginary_time_gate(self, h: float, d_tau: float, site: int) -> np.ndarray:
        """Imaginary time evolution operator: exp(-H*d_tau)"""
        H_bond = self.model.two_site_hamiltonian(h, site)
        U = expm(-H_bond * d_tau)  # No imaginary unit!
        return U.reshape(2, 2, 2, 2)

    def _apply_gate(self, mps: MPS, U: np.ndarray, site: int):
        """Apply two-site gate with SVD truncation"""
        # Get tensors
        A_left = mps.tensors[site]
        A_right = mps.tensors[site + 1]

        chi_left = A_left.shape[0]
        chi_right = A_right.shape[2]

        # Combine
        theta = np.tensordot(A_left, A_right, axes=(2, 0))

        # Apply gate
        theta_new = np.tensordot(U, theta, axes=([2, 3], [1, 2]))
        theta_new = theta_new.transpose(1, 0, 2, 3)

        # SVD
        theta_mat = theta_new.reshape(chi_left * 2, 2 * chi_right)
        U_svd, S, Vt = svd(theta_mat, full_matrices=False)

        # Truncate
        chi_new = min(len(S), self.params.chi_max)
        U_svd = U_svd[:, :chi_new]
        S = S[:chi_new]
        Vt = Vt[:chi_new, :]

        # Update
        mps.tensors[site] = U_svd.reshape(chi_left, 2, chi_new)
        mps.tensors[site + 1] = (np.diag(S) @ Vt).reshape(chi_new, 2, chi_right)


# ============================================================================
# 6. FINITE-SIZE SCALING ANALYSIS
# ============================================================================

class FiniteSizeScaling:
    """
    Finite-size scaling analysis for quantum phase transitions

    At a quantum critical point, observables scale as:
    O(L, δ) = L^(x/ν) f(δ L^(1/ν))

    where:
    - L is the system size
    - δ = (h - h_c)/h_c is the distance from criticality
    - ν is the correlation length exponent
    - x is the scaling dimension of observable O
    """

    def __init__(self, h_c: float = 1.0, nu: float = 1.0):
        self.h_c = h_c
        self.nu = nu

    def scaling_collapse(self, L_values: List[int],
                        h_values: np.ndarray,
                        observables: Dict[int, np.ndarray]) -> Dict:
        """
        Perform data collapse for finite-size scaling

        Parameters:
        -----------
        L_values : list of int
            System sizes
        h_values : np.ndarray
            Field values
        observables : dict
            observables[L] = array of observable values at different h

        Returns:
        --------
        results : dict
            Scaled data and collapse quality
        """
        results = {
            'scaled_x': {},
            'scaled_y': {},
            'raw_x': {},
            'raw_y': {}
        }

        for L in L_values:
            # Scaling variable: x = (h - h_c) * L^(1/ν)
            delta_h = (h_values - self.h_c) / self.h_c
            x_scaled = delta_h * L**(1.0 / self.nu)

            # Assume observable scales as O ~ L^α (need to fit α)
            # For residual energy: α ≈ 0 (intensive)
            # For correlation length: α ≈ 1
            y_scaled = observables[L]  # / L^α if needed

            results['scaled_x'][L] = x_scaled
            results['scaled_y'][L] = y_scaled
            results['raw_x'][L] = h_values
            results['raw_y'][L] = observables[L]

        return results

    def plot_data_collapse(self, results: Dict,
                          observable_name: str = "Observable",
                          save_path: Optional[str] = None):
        """Plot finite-size scaling collapse"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Raw data
        for L in results['raw_x'].keys():
            ax1.plot(results['raw_x'][L], results['raw_y'][L],
                    'o-', label=f'L={L}', markersize=6, linewidth=2)

        ax1.axvline(x=self.h_c, color='r', linestyle='--',
                   linewidth=2, label=f'h_c={self.h_c}')
        ax1.set_xlabel('Transverse field h', fontsize=12)
        ax1.set_ylabel(observable_name, fontsize=12)
        ax1.set_title('Raw Data', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Scaled data (collapse)
        for L in results['scaled_x'].keys():
            ax2.plot(results['scaled_x'][L], results['scaled_y'][L],
                    'o-', label=f'L={L}', markersize=6, linewidth=2, alpha=0.7)

        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel(r'$(h - h_c) L^{1/\nu}$', fontsize=12)
        ax2.set_ylabel(f'{observable_name} (scaled)', fontsize=12)
        ax2.set_title('Data Collapse', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved finite-size scaling plot to {save_path}")

        plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_trotter_orders(model, params: DMRGParameters,
                          h_initial: float, h_final: float,
                          T: float, exact_energy: float):
    """
    Compare 2nd-order vs 4th-order Trotter accuracy

    Returns residual energies for both methods.
    """
    from ising_adiabatic_dmrg import AdiabaticPassage

    # 2nd order
    tebd2 = TEBD(model, params)
    passage2 = AdiabaticPassage(model, params)
    passage2.tebd = tebd2
    result2 = passage2.simulate(h_initial, h_final, T, save_every=10)

    # 4th order
    tebd4 = TEBDFourthOrder(model, params)
    passage4 = AdiabaticPassage(model, params)
    passage4.tebd = tebd4
    result4 = passage4.simulate(h_initial, h_final, T, save_every=10)

    delta_E_2nd = result2['final_energy'] - exact_energy
    delta_E_4th = result4['final_energy'] - exact_energy

    print(f"\nTrotter Comparison (T={T:.2f}):")
    print(f"  2nd order: δE = {delta_E_2nd:.6e}")
    print(f"  4th order: δE = {delta_E_4th:.6e}")
    print(f"  Improvement: {abs(delta_E_2nd / delta_E_4th):.2f}x")

    return delta_E_2nd, delta_E_4th


def compare_ramp_protocols(model, params: DMRGParameters,
                           h_initial: float, h_final: float,
                           T: float, exact_energy: float):
    """
    Compare different ramping protocols

    Returns residual energies for each protocol.
    """
    from ising_adiabatic_dmrg import AdiabaticPassage

    protocols = {
        'Linear': LinearRamp(h_initial, h_final, T),
        'Tanh': TanhRamp(h_initial, h_final, T),
        'Polynomial (n=2)': PolynomialRamp(h_initial, h_final, T, exponent=2.0),
        'Optimal': OptimalRamp(h_initial, h_final, T)
    }

    results = {}

    for name, protocol in protocols.items():
        print(f"\nTesting {name} ramp...")

        # Find initial ground state
        dmrg = GroundStateDMRG(model, params)
        mps, _ = dmrg.run(h_initial)

        # Time evolution with custom ramp
        tebd = TEBD(model, params)
        n_steps = int(T / params.dt)

        for step in tqdm(range(n_steps), desc=name):
            t = step * params.dt
            h_t = protocol.h_of_t(t)
            tebd.evolve_step(mps, h_t, params.dt)

        # Final energy
        h_f = protocol.h_of_t(T)
        E_final = Measurements.measure_energy(mps, model, h_f)
        delta_E = E_final - exact_energy

        results[name] = delta_E
        print(f"  Residual energy: δE = {delta_E:.6e}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    energies = [abs(results[n]) for n in names]

    ax.bar(names, energies, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('|Residual Energy| δE', fontsize=12)
    ax.set_title(f'Comparison of Ramp Protocols (T={T:.2f})', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('results/ramp_protocol_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comparison plot to results/ramp_protocol_comparison.png")
    plt.show()

    return results


if __name__ == '__main__':
    print("Tensor Network Extensions Module")
    print("="*70)
    print("\nAvailable features:")
    print("  1. TEBDFourthOrder - 4th-order Trotter decomposition")
    print("  2. AdaptiveTEBD - Adaptive time stepping")
    print("  3. RampProtocol classes - Non-linear ramps")
    print("  4. XXZModel - XXZ spin chain")
    print("  5. FermiHubbardModel - Fermi-Hubbard model")
    print("  6. FiniteTemperatureMPS - Thermal states")
    print("  7. FiniteSizeScaling - FSS analysis")
    print("\nImport this module to use these features!")
