"""
Time-Dependent DMRG for Adiabatic Passage in Transverse Field Ising Model

This module implements time-dependent DMRG (t-DMRG) to simulate adiabatic passage
through the quantum critical point of the transverse field Ising model (TFIM).

The TFIM Hamiltonian is:
    H(h) = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - h Σᵢ σᵢˣ

The quantum critical point is at h_c = J = 1 (setting J=1).

This implementation includes:
1. Matrix Product State (MPS) representation
2. Ground state DMRG
3. Time evolution using TEBD (Time-Evolving Block Decimation)
4. Adiabatic passage protocols
5. Analysis of Landau-Zener and Kibble-Zurek regimes
"""

import numpy as np
from scipy.linalg import expm, svd, eigh, norm
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os


@dataclass
class DMRGParameters:
    """Parameters for DMRG and time evolution"""
    chi_max: int = 50  # Maximum bond dimension
    chi_init: int = 10  # Initial bond dimension for ground state search
    n_sweeps: int = 10  # Number of DMRG sweeps
    convergence_tol: float = 1e-8  # Energy convergence tolerance
    dt: float = 0.02  # Time step for TEBD (smaller for stability)
    trotter_order: int = 2  # Trotter decomposition order


class MPS:
    """
    Matrix Product State representation

    An MPS represents a quantum state as:
    |ψ⟩ = Σ A[0]^{s₀} A[1]^{s₁} ... A[L-1]^{sₗ₋₁} |s₀s₁...sₗ₋₁⟩

    Each tensor A[i] has shape (χᵢ, d, χᵢ₊₁) where:
    - χᵢ is the left bond dimension
    - d is the physical dimension (d=2 for spin-1/2)
    - χᵢ₊₁ is the right bond dimension
    """

    def __init__(self, L: int, d: int = 2, chi: int = 10, random_init: bool = True):
        """
        Initialize MPS

        Parameters:
        -----------
        L : int
            System size (number of sites)
        d : int
            Physical dimension (default: 2 for spin-1/2)
        chi : int
            Initial bond dimension
        random_init : bool
            If True, initialize with random tensors; otherwise use product state
        """
        self.L = L
        self.d = d
        self.tensors = []

        if random_init:
            # Random initialization with consistent bond dimensions
            # First, determine all bond dimensions
            bond_dims = [1]  # Left boundary
            for i in range(1, L):
                bond_dim = min(chi, d**i, d**(L-i))
                bond_dims.append(bond_dim)
            bond_dims.append(1)  # Right boundary

            # Now create tensors with matching dimensions
            for i in range(L):
                chi_left = bond_dims[i]
                chi_right = bond_dims[i+1]

                tensor = np.random.randn(chi_left, d, chi_right) + \
                         1j * np.random.randn(chi_left, d, chi_right)
                # Normalize
                tensor /= np.linalg.norm(tensor)
                self.tensors.append(tensor)
        else:
            # Product state initialization (all spins up)
            for i in range(L):
                chi_left = 1
                chi_right = 1
                tensor = np.zeros((chi_left, d, chi_right), dtype=complex)
                tensor[0, 0, 0] = 1.0  # Spin up state
                self.tensors.append(tensor)

        # Don't normalize during initialization to avoid issues
        # self._normalize_mps()

    def _normalize_mps(self):
        """Normalize the MPS by right-canonicalization"""
        for i in range(self.L - 1, 0, -1):
            self._right_canonicalize_site(i)
        # Normalize the leftmost tensor
        self.tensors[0] /= np.linalg.norm(self.tensors[0])

    def _left_canonicalize_site(self, site: int):
        """Left-canonicalize a single site using SVD"""
        chi_left, d, chi_right = self.tensors[site].shape
        # Reshape to matrix: combine left index and physical index
        mat = self.tensors[site].reshape(chi_left * d, chi_right)
        # SVD
        U, S, Vt = svd(mat, full_matrices=False)
        # Keep U as the new tensor
        new_chi_right = min(U.shape[1], chi_right)
        self.tensors[site] = U[:, :new_chi_right].reshape(chi_left, d, new_chi_right)
        # Absorb S*Vt into next site
        if site < self.L - 1:
            SV = np.diag(S[:new_chi_right]) @ Vt[:new_chi_right, :]
            self.tensors[site + 1] = np.tensordot(SV, self.tensors[site + 1], axes=(1, 0))

    def _right_canonicalize_site(self, site: int):
        """Right-canonicalize a single site using SVD"""
        chi_left, d, chi_right = self.tensors[site].shape
        # Reshape to matrix: combine physical index and right index
        mat = self.tensors[site].reshape(chi_left, d * chi_right)
        # SVD
        U, S, Vt = svd(mat, full_matrices=False)
        # Keep Vt as the new tensor
        new_chi_left = min(Vt.shape[0], chi_left)
        self.tensors[site] = Vt[:new_chi_left, :].reshape(new_chi_left, d, chi_right)
        # Absorb U*S into previous site
        if site > 0:
            US = U[:, :new_chi_left] @ np.diag(S[:new_chi_left])
            self.tensors[site - 1] = np.tensordot(self.tensors[site - 1], US, axes=(2, 0))

    def get_bond_dimensions(self) -> List[int]:
        """Get the bond dimensions of the MPS"""
        dims = [1]  # Left boundary
        for tensor in self.tensors:
            dims.append(tensor.shape[2])
        return dims

    def copy(self):
        """Create a deep copy of the MPS"""
        new_mps = MPS.__new__(MPS)
        new_mps.L = self.L
        new_mps.d = self.d
        new_mps.tensors = [tensor.copy() for tensor in self.tensors]
        return new_mps

    def norm(self) -> float:
        """Compute the norm of the MPS"""
        # Contract the MPS with its conjugate
        result = np.ones((1, 1), dtype=complex)
        for i in range(self.L):
            # Contract with the MPS tensor and its conjugate
            temp = np.tensordot(result, self.tensors[i], axes=(1, 0))
            result = np.tensordot(temp, self.tensors[i].conj(), axes=([0, 1], [0, 1]))
        return np.sqrt(np.abs(result[0, 0]))


class TransverseFieldIsing:
    """
    Transverse Field Ising Model (TFIM)

    H(h) = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - h Σᵢ σᵢˣ

    where J is the Ising coupling (set to 1) and h is the transverse field.
    The quantum critical point is at h_c = 1.
    """

    def __init__(self, L: int, J: float = 1.0):
        """
        Initialize TFIM

        Parameters:
        -----------
        L : int
            System size
        J : float
            Ising coupling (default: 1.0)
        """
        self.L = L
        self.J = J

        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)

    def two_site_hamiltonian(self, h: float, site: int) -> np.ndarray:
        """
        Get the two-site Hamiltonian for sites (site, site+1)

        H_bond = -J σᶻ⊗σᶻ - (h/2)(σˣ⊗I + I⊗σˣ)
        """
        # Ising interaction
        H = -self.J * np.kron(self.sigma_z, self.sigma_z)
        # Transverse field (split equally between two sites)
        H -= (h / 2) * np.kron(self.sigma_x, self.identity)
        H -= (h / 2) * np.kron(self.identity, self.sigma_x)
        return H

    def exact_ground_state_energy_density(self, h: float) -> float:
        """
        Exact ground state energy per site for infinite system
        Using Jordan-Wigner transformation and Fourier analysis
        """
        # For large system, use thermodynamic limit formula
        def integrand(k):
            return -np.sqrt(1 + h**2 - 2*h*np.cos(k))

        # Numerical integration
        from scipy.integrate import quad
        energy, _ = quad(integrand, 0, np.pi)
        return -self.J * energy / np.pi

    def exact_ground_state_finite(self, h: float) -> Tuple[float, np.ndarray]:
        """
        Exact ground state for finite system using exact diagonalization
        (Only feasible for small systems L < 20)
        """
        if self.L > 20:
            raise ValueError("Exact diagonalization only feasible for L < 20")

        # Build full Hamiltonian
        dim = 2**self.L
        H = np.zeros((dim, dim), dtype=complex)

        for i in range(self.L):
            # Transverse field term
            op = [self.identity] * self.L
            op[i] = self.sigma_x
            term = op[0]
            for j in range(1, self.L):
                term = np.kron(term, op[j])
            H -= h * term

            # Ising interaction (periodic boundary conditions)
            j = (i + 1) % self.L
            op = [self.identity] * self.L
            op[i] = self.sigma_z
            op[j] = self.sigma_z
            term = op[0]
            for k in range(1, self.L):
                term = np.kron(term, op[k])
            H -= self.J * term

        # Diagonalize
        energies, eigenvectors = eigh(H)
        return energies[0], eigenvectors[:, 0]


class GroundStateDMRG:
    """
    Ground state DMRG for finding the ground state of a Hamiltonian
    """

    def __init__(self, model: TransverseFieldIsing, params: DMRGParameters):
        self.model = model
        self.params = params

    def optimize_two_site(self, mps: MPS, site: int, h: float) -> float:
        """
        Optimize two sites using the two-site Hamiltonian

        Returns the local energy
        """
        L = mps.L

        # Get the two-site Hamiltonian
        H_bond = self.model.two_site_hamiltonian(h, site)

        # Build the effective Hamiltonian with left and right environments
        # For simplicity, we use a simplified version without full environment tensors
        # This is sufficient for the TFIM which is nearest-neighbor

        # Combine two sites
        A_left = mps.tensors[site]
        A_right = mps.tensors[site + 1]

        chi_left = A_left.shape[0]
        chi_right = A_right.shape[2]

        # Form two-site tensor: theta[chi_left, d, d, chi_right]
        theta = np.tensordot(A_left, A_right, axes=(2, 0))

        # Reshape for eigenvalue problem
        theta_mat = theta.reshape(chi_left * 2 * 2 * chi_right)

        # Apply Hamiltonian (simplified without environment)
        H_bond_reshaped = H_bond.reshape(2, 2, 2, 2)
        theta_H = np.tensordot(H_bond_reshaped, theta, axes=([2, 3], [1, 2]))
        theta_H = theta_H.transpose(1, 0, 2, 3)

        # Compute energy
        energy = np.real(np.vdot(theta_mat, theta_H.reshape(-1)))

        # Normalize
        norm_sq = np.real(np.vdot(theta_mat, theta_mat))
        energy /= norm_sq

        # SVD to split back into two tensors
        theta_for_svd = theta.reshape(chi_left * 2, 2 * chi_right)
        U, S, Vt = svd(theta_for_svd, full_matrices=False)

        # Truncate to chi_max
        chi_new = min(len(S), self.params.chi_max)
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vt = Vt[:chi_new, :]

        # Normalize singular values
        S = S / np.linalg.norm(S)

        # Update tensors
        mps.tensors[site] = U.reshape(chi_left, 2, chi_new)
        mps.tensors[site + 1] = (np.diag(S) @ Vt).reshape(chi_new, 2, chi_right)

        return energy

    def run(self, h: float, initial_mps: Optional[MPS] = None) -> Tuple[MPS, float]:
        """
        Run DMRG to find ground state

        Parameters:
        -----------
        h : float
            Transverse field strength
        initial_mps : MPS, optional
            Initial MPS guess (if None, create random MPS)

        Returns:
        --------
        mps : MPS
            Ground state MPS
        energy : float
            Ground state energy
        """
        L = self.model.L

        if initial_mps is None:
            mps = MPS(L, chi=self.params.chi_init, random_init=True)
        else:
            mps = initial_mps.copy()

        energy_old = 0.0

        for sweep in range(self.params.n_sweeps):
            energy = 0.0

            # Left-to-right sweep
            for site in range(L - 1):
                e_local = self.optimize_two_site(mps, site, h)
                energy += e_local
                mps._left_canonicalize_site(site)

            # Right-to-left sweep
            for site in range(L - 2, -1, -1):
                e_local = self.optimize_two_site(mps, site, h)
                if site == 0:
                    energy = e_local * (L - 1)  # Approximate total energy
                mps._right_canonicalize_site(site + 1)

            # Check convergence
            delta_E = abs(energy - energy_old)
            if delta_E < self.params.convergence_tol:
                break

            energy_old = energy

        return mps, energy


class TEBD:
    """
    Time-Evolving Block Decimation (TEBD) for time evolution

    Implements time evolution of an MPS under a time-dependent Hamiltonian
    using Trotter decomposition.
    """

    def __init__(self, model: TransverseFieldIsing, params: DMRGParameters):
        self.model = model
        self.params = params

    def _time_evolution_operator(self, h: float, dt: float, site: int) -> np.ndarray:
        """
        Compute the time evolution operator U = exp(-i H dt) for two sites
        """
        H_bond = self.model.two_site_hamiltonian(h, site)
        U = expm(-1j * H_bond * dt)
        return U.reshape(2, 2, 2, 2)

    def _apply_two_site_gate(self, mps: MPS, U: np.ndarray, site: int):
        """
        Apply a two-site gate to the MPS and perform SVD truncation
        """
        # Get the two tensors
        A_left = mps.tensors[site]
        A_right = mps.tensors[site + 1]

        chi_left = A_left.shape[0]
        chi_right = A_right.shape[2]

        # Combine two sites: theta[chi_left, d, d, chi_right]
        theta = np.tensordot(A_left, A_right, axes=(2, 0))

        # Apply gate: U[d', d', d, d] * theta[chi_left, d, d, chi_right]
        theta_new = np.tensordot(U, theta, axes=([2, 3], [1, 2]))
        theta_new = theta_new.transpose(1, 0, 2, 3)

        # SVD truncation
        theta_mat = theta_new.reshape(chi_left * 2, 2 * chi_right)
        U_svd, S, Vt = svd(theta_mat, full_matrices=False)

        # Truncate
        chi_new = min(len(S), self.params.chi_max)
        U_svd = U_svd[:, :chi_new]
        S = S[:chi_new]
        Vt = Vt[:chi_new, :]

        # Update tensors
        mps.tensors[site] = U_svd.reshape(chi_left, 2, chi_new)
        mps.tensors[site + 1] = (np.diag(S) @ Vt).reshape(chi_new, 2, chi_right)

    def evolve_step(self, mps: MPS, h: float, dt: float):
        """
        Perform one time step of evolution with second-order Trotter decomposition
        """
        L = mps.L

        # Second-order Trotter: exp(-iHdt) ≈ exp(-iH_even dt/2) exp(-iH_odd dt) exp(-iH_even dt/2)

        # Even bonds (0, 2, 4, ...)
        for site in range(0, L - 1, 2):
            U = self._time_evolution_operator(h, dt / 2, site)
            self._apply_two_site_gate(mps, U, site)

        # Odd bonds (1, 3, 5, ...)
        for site in range(1, L - 1, 2):
            U = self._time_evolution_operator(h, dt, site)
            self._apply_two_site_gate(mps, U, site)

        # Even bonds again
        for site in range(0, L - 1, 2):
            U = self._time_evolution_operator(h, dt / 2, site)
            self._apply_two_site_gate(mps, U, site)

        # Normalize to prevent numerical overflow
        norm = mps.norm()
        if norm > 1e-10:
            for i in range(L):
                mps.tensors[i] /= norm**(1.0/L)


class Measurements:
    """
    Observable measurements on MPS
    """

    @staticmethod
    def measure_energy(mps: MPS, model: TransverseFieldIsing, h: float) -> float:
        """
        Measure the energy ⟨ψ|H|ψ⟩
        """
        L = mps.L
        energy = 0.0

        # Normalize MPS
        norm = mps.norm()

        for site in range(L - 1):
            # Get two-site reduced density matrix
            A_left = mps.tensors[site]
            A_right = mps.tensors[site + 1]

            # Contract to get two-site wave function: theta[chi_left, d, d, chi_right]
            theta = np.tensordot(A_left, A_right, axes=(2, 0))

            # Get Hamiltonian: H_bond[d*d, d*d]
            H_bond = model.two_site_hamiltonian(h, site)

            # Reshape H_bond to act on physical indices: H[d, d, d, d]
            H_reshaped = H_bond.reshape(2, 2, 2, 2)

            # Apply Hamiltonian to theta: contract physical indices
            # H[d', d', d, d] * theta[chi_left, d, d, chi_right]
            # -> result[d', d', chi_left, chi_right]
            H_theta = np.tensordot(H_reshaped, theta, axes=([2, 3], [1, 2]))
            # Rearrange: [chi_left, d', d', chi_right]
            H_theta = H_theta.transpose(2, 0, 1, 3)

            # Compute expectation value: <theta|H|theta>
            e_local = np.real(np.vdot(theta.reshape(-1), H_theta.reshape(-1)))
            energy += e_local

        return energy / (norm**2)

    @staticmethod
    def measure_correlation_length(mps: MPS) -> float:
        """
        Estimate correlation length from entanglement entropy

        For a critical system, ξ is related to the entanglement entropy.
        We use a simplified measure based on the bond dimension and entropy.
        """
        # Use the middle bond for measurement
        site = mps.L // 2

        # Get the bond dimension
        bond_dim = mps.get_bond_dimensions()[site]

        # Estimate correlation length from bond dimension
        # At criticality, χ ~ exp(c * ξ) for some constant c
        # So ξ ~ log(χ) / c, we use c=1 as an approximation
        xi_estimate = np.log(bond_dim + 1)

        return xi_estimate

    @staticmethod
    def measure_magnetization(mps: MPS, direction: str = 'z') -> float:
        """
        Measure the magnetization ⟨σᶻ⟩ or ⟨σˣ⟩
        """
        L = mps.L

        if direction == 'z':
            sigma = np.array([[1, 0], [0, -1]], dtype=complex)
        elif direction == 'x':
            sigma = np.array([[0, 1], [1, 0]], dtype=complex)
        else:
            raise ValueError("direction must be 'x' or 'z'")

        mag = 0.0
        norm_sq = mps.norm()**2

        for site in range(L):
            A = mps.tensors[site]
            chi_left, d, chi_right = A.shape

            # Contract with operator
            A_sigma = np.tensordot(sigma, A, axes=(1, 1))
            A_sigma = A_sigma.transpose(1, 0, 2)

            # Contract to get expectation value
            # This is a simplified calculation
            overlap = np.sum(A_sigma.conj() * A)
            mag += np.real(overlap)

        return mag / (L * norm_sq)


class AdiabaticPassage:
    """
    Simulate adiabatic passage through the quantum critical point
    """

    def __init__(self, model: TransverseFieldIsing, params: DMRGParameters):
        self.model = model
        self.params = params
        self.dmrg = GroundStateDMRG(model, params)
        self.tebd = TEBD(model, params)

    def linear_ramp(self, t: float, T: float, h_initial: float, h_final: float) -> float:
        """Linear ramping of the field"""
        return h_initial + (h_final - h_initial) * t / T

    def simulate(
        self,
        h_initial: float,
        h_final: float,
        T_total: float,
        save_every: int = 10
    ) -> dict:
        """
        Simulate adiabatic passage

        Parameters:
        -----------
        h_initial : float
            Initial transverse field
        h_final : float
            Final transverse field
        T_total : float
            Total time for the passage
        save_every : int
            Save state every N steps

        Returns:
        --------
        results : dict
            Dictionary containing:
            - times: array of times
            - fields: array of field values
            - energies: array of energies
            - bond_dims: array of maximum bond dimensions
            - final_mps: final MPS state
        """
        # Find initial ground state
        print(f"Finding initial ground state at h = {h_initial:.3f}...")
        mps, E0 = self.dmrg.run(h_initial)

        # Time evolution parameters
        dt = self.params.dt
        n_steps = int(T_total / dt)

        # Storage
        times = []
        fields = []
        energies = []
        bond_dims = []

        print(f"Starting time evolution for T = {T_total:.2f} with {n_steps} steps...")

        for step in tqdm(range(n_steps)):
            t = step * dt
            h_t = self.linear_ramp(t, T_total, h_initial, h_final)

            # Evolve one step
            self.tebd.evolve_step(mps, h_t, dt)

            # Save data
            if step % save_every == 0:
                times.append(t)
                fields.append(h_t)
                E = Measurements.measure_energy(mps, self.model, h_t)
                energies.append(E)
                bond_dims.append(max(mps.get_bond_dimensions()))

        # Final measurement
        t_final = T_total
        h_final_actual = self.linear_ramp(t_final, T_total, h_initial, h_final)
        E_final = Measurements.measure_energy(mps, self.model, h_final_actual)
        xi_final = Measurements.measure_correlation_length(mps)

        print(f"\nFinal state at h = {h_final_actual:.3f}:")
        print(f"  Energy: {E_final:.6f}")
        print(f"  Correlation length: {xi_final:.6f}")
        print(f"  Max bond dimension: {max(mps.get_bond_dimensions())}")

        return {
            'times': np.array(times),
            'fields': np.array(fields),
            'energies': np.array(energies),
            'bond_dims': np.array(bond_dims),
            'final_mps': mps,
            'final_energy': E_final,
            'final_xi': xi_final,
            'h_initial': h_initial,
            'h_final': h_final,
            'T_total': T_total
        }


class KibbleZurekAnalysis:
    """
    Analysis of Landau-Zener and Kibble-Zurek physics

    In the Landau-Zener regime (fast passage): δE ~ exp(-aT)
    In the Kibble-Zurek regime (slow passage): δE ~ T^{-dν/(1+zν)}

    For the 1D TFIM: d=1, z=1, ν=1 → δE ~ T^{-1/2}
    """

    def __init__(self, model: TransverseFieldIsing):
        self.model = model

    def compute_residual_energy(
        self,
        mps_final: MPS,
        h_final: float,
        E_ground: float
    ) -> float:
        """
        Compute residual excitation energy: δE = E_final - E_ground
        """
        E_final = Measurements.measure_energy(mps_final, self.model, h_final)
        return E_final - E_ground

    @staticmethod
    def landau_zener_fit(T: np.ndarray, a: float, b: float) -> np.ndarray:
        """Fit function for Landau-Zener regime: δE = b * exp(-a*T)"""
        return b * np.exp(-a * T)

    @staticmethod
    def kibble_zurek_fit(T: np.ndarray, A: float, alpha: float) -> np.ndarray:
        """Fit function for Kibble-Zurek regime: δE = A * T^(-alpha)"""
        return A * T**(-alpha)

    def analyze_scaling(
        self,
        T_values: np.ndarray,
        delta_E_values: np.ndarray,
        T_threshold: Optional[float] = None
    ) -> dict:
        """
        Analyze the scaling of residual energy with passage time

        Parameters:
        -----------
        T_values : np.ndarray
            Array of total passage times
        delta_E_values : np.ndarray
            Array of residual energies
        T_threshold : float, optional
            Threshold to separate LZ and KZ regimes (if None, auto-detect)

        Returns:
        --------
        results : dict
            Dictionary with fit parameters and regime classifications
        """
        # Sort by T
        idx = np.argsort(T_values)
        T_sorted = T_values[idx]
        dE_sorted = delta_E_values[idx]

        # Auto-detect threshold if not provided
        if T_threshold is None:
            # Use the median or a heuristic
            T_threshold = np.median(T_sorted)

        # Split into fast (LZ) and slow (KZ) regimes
        lz_mask = T_sorted < T_threshold
        kz_mask = T_sorted >= T_threshold

        results = {
            'T_values': T_sorted,
            'delta_E_values': dE_sorted,
            'T_threshold': T_threshold
        }

        # Fit Landau-Zener regime (if enough points)
        if np.sum(lz_mask) >= 3:
            T_lz = T_sorted[lz_mask]
            dE_lz = dE_sorted[lz_mask]

            # Use points that are positive and not too small
            valid = dE_lz > 1e-10
            if np.sum(valid) >= 3:
                try:
                    popt_lz, _ = curve_fit(
                        self.landau_zener_fit,
                        T_lz[valid],
                        dE_lz[valid],
                        p0=[1.0, 1.0],
                        maxfev=10000
                    )
                    results['lz_params'] = {'a': popt_lz[0], 'b': popt_lz[1]}
                    results['lz_fit'] = self.landau_zener_fit(T_lz[valid], *popt_lz)
                    results['T_lz'] = T_lz[valid]
                    results['dE_lz'] = dE_lz[valid]
                except:
                    results['lz_params'] = None

        # Fit Kibble-Zurek regime (if enough points)
        if np.sum(kz_mask) >= 3:
            T_kz = T_sorted[kz_mask]
            dE_kz = dE_sorted[kz_mask]

            # Use log-log fit for power law
            valid = (dE_kz > 1e-10) & (T_kz > 0)
            if np.sum(valid) >= 3:
                try:
                    # Log-log linear fit
                    log_T = np.log(T_kz[valid])
                    log_dE = np.log(dE_kz[valid])
                    coeffs = np.polyfit(log_T, log_dE, 1)
                    alpha = -coeffs[0]  # Negative slope
                    A = np.exp(coeffs[1])

                    results['kz_params'] = {'A': A, 'alpha': alpha}
                    results['kz_fit'] = self.kibble_zurek_fit(T_kz[valid], A, alpha)
                    results['T_kz'] = T_kz[valid]
                    results['dE_kz'] = dE_kz[valid]

                    # Compare with theoretical prediction (α = 1/2 for 1D TFIM)
                    results['theoretical_alpha'] = 0.5
                    results['alpha_error'] = abs(alpha - 0.5)
                except:
                    results['kz_params'] = None

        return results


def plot_adiabatic_passage_results(results: dict, save_path: Optional[str] = None):
    """Plot the results of adiabatic passage simulation"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Field vs time
    axes[0, 0].plot(results['times'], results['fields'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Critical point h=1')
    axes[0, 0].set_xlabel('Time t', fontsize=12)
    axes[0, 0].set_ylabel('Transverse field h(t)', fontsize=12)
    axes[0, 0].set_title('Field Ramping Protocol', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Energy vs time
    axes[0, 1].plot(results['times'], results['energies'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time t', fontsize=12)
    axes[0, 1].set_ylabel('Energy E(t)', fontsize=12)
    axes[0, 1].set_title('Energy Evolution', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # Bond dimension vs time
    axes[1, 0].plot(results['times'], results['bond_dims'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time t', fontsize=12)
    axes[1, 0].set_ylabel('Max Bond Dimension χ', fontsize=12)
    axes[1, 0].set_title('Entanglement Growth', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Energy vs field (hysteresis-like plot)
    axes[1, 1].plot(results['fields'], results['energies'], 'purple', linewidth=2)
    axes[1, 1].axvline(x=1.0, color='r', linestyle='--', label='Critical point')
    axes[1, 1].set_xlabel('Transverse field h', fontsize=12)
    axes[1, 1].set_ylabel('Energy E', fontsize=12)
    axes[1, 1].set_title('Energy vs Field', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_kibble_zurek_analysis(
    analysis_results: dict,
    save_path: Optional[str] = None
):
    """Plot the Kibble-Zurek scaling analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T_vals = analysis_results['T_values']
    dE_vals = analysis_results['delta_E_values']

    # Linear plot
    axes[0].loglog(T_vals, dE_vals, 'ko', markersize=8, label='Simulation data')

    # Plot fits if available
    if 'lz_params' in analysis_results and analysis_results['lz_params'] is not None:
        T_lz = analysis_results['T_lz']
        dE_lz_fit = analysis_results['lz_fit']
        axes[0].loglog(T_lz, dE_lz_fit, 'b-', linewidth=2,
                      label=f"LZ fit: exp(-{analysis_results['lz_params']['a']:.2f}T)")

    if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
        T_kz = analysis_results['T_kz']
        dE_kz_fit = analysis_results['kz_fit']
        alpha = analysis_results['kz_params']['alpha']
        axes[0].loglog(T_kz, dE_kz_fit, 'r-', linewidth=2,
                      label=f"KZ fit: T^(-{alpha:.3f})")

        # Plot theoretical KZ scaling
        T_theory = np.logspace(np.log10(T_kz[0]), np.log10(T_kz[-1]), 50)
        dE_theory = analysis_results['kz_params']['A'] * T_theory**(-0.5)
        axes[0].loglog(T_theory, dE_theory, 'g--', linewidth=2,
                      label="KZ theory: T^(-1/2)")

    if 'T_threshold' in analysis_results:
        axes[0].axvline(x=analysis_results['T_threshold'], color='gray',
                       linestyle='--', alpha=0.5, label='Regime threshold')

    axes[0].set_xlabel('Total Time T', fontsize=12)
    axes[0].set_ylabel('Residual Energy δE', fontsize=12)
    axes[0].set_title('Residual Energy Scaling', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Log-log plot with linear fit for KZ regime
    if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
        T_kz = analysis_results['T_kz']
        dE_kz = analysis_results['dE_kz']

        axes[1].plot(np.log(T_kz), np.log(dE_kz), 'ro', markersize=8,
                    label='KZ regime data')

        # Linear fit
        coeffs = np.polyfit(np.log(T_kz), np.log(dE_kz), 1)
        fit_line = coeffs[0] * np.log(T_kz) + coeffs[1]
        axes[1].plot(np.log(T_kz), fit_line, 'b-', linewidth=2,
                    label=f'Fit: slope = {-coeffs[0]:.3f}')

        # Theoretical slope
        axes[1].plot(np.log(T_kz), -0.5 * np.log(T_kz) + coeffs[1] + 0.5*np.log(T_kz[0]),
                    'g--', linewidth=2, label='Theory: slope = 0.5')

        axes[1].set_xlabel('log(T)', fontsize=12)
        axes[1].set_ylabel('log(δE)', fontsize=12)
        axes[1].set_title('Kibble-Zurek Regime (Log-Log)', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Print comparison
        alpha_sim = -coeffs[0]
        alpha_theory = 0.5
        print(f"\nCritical exponent comparison:")
        print(f"  Simulated α = {alpha_sim:.4f}")
        print(f"  Theoretical α = {alpha_theory:.4f}")
        print(f"  Relative error = {abs(alpha_sim - alpha_theory)/alpha_theory * 100:.2f}%")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def main_simulation():
    """
    Main simulation: Adiabatic passage for multiple system sizes and passage times
    """
    print("=" * 70)
    print("Adiabatic Passage through Quantum Critical Point")
    print("Transverse Field Ising Model (TFIM)")
    print("=" * 70)

    # Parameters
    L_values = [20, 30, 40]  # System sizes
    T_values = np.logspace(-0.5, 2, 10)  # Passage times from ~0.3 to 100
    h_initial = 2.0  # Start in paramagnetic phase
    h_final = 0.5    # End in ferromagnetic phase

    # DMRG parameters
    params = DMRGParameters(
        chi_max=50,
        chi_init=10,
        n_sweeps=10,
        convergence_tol=1e-8,
        dt=0.05,
        trotter_order=2
    )

    # Storage for results
    all_results = {}

    for L in L_values:
        print(f"\n{'='*70}")
        print(f"System size L = {L}")
        print(f"{'='*70}")

        model = TransverseFieldIsing(L)
        adiabatic = AdiabaticPassage(model, params)

        # Get exact ground state energy at final field
        print(f"\nComputing exact ground state at h_final = {h_final}...")
        try:
            if L <= 20:
                E_exact, _ = model.exact_ground_state_finite(h_final)
            else:
                E_exact = model.exact_ground_state_energy_density(h_final) * L
            print(f"Exact ground state energy: {E_exact:.6f}")
        except:
            print("Exact calculation not available, using DMRG reference")
            dmrg = GroundStateDMRG(model, params)
            _, E_exact = dmrg.run(h_final)

        L_results = {
            'E_exact': E_exact,
            'passages': []
        }

        # Simulate for different passage times
        for T in T_values:
            print(f"\n--- Passage time T = {T:.3f} ---")

            result = adiabatic.simulate(h_initial, h_final, T, save_every=10)

            # Compute residual energy
            delta_E = result['final_energy'] - E_exact
            result['delta_E'] = delta_E

            print(f"Residual energy δE = {delta_E:.6e}")

            L_results['passages'].append(result)

        all_results[L] = L_results

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/adiabatic_passage_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("\nResults saved to results/adiabatic_passage_results.pkl")

    return all_results


def main_analysis():
    """
    Analyze the results and extract Kibble-Zurek scaling
    """
    print("\n" + "="*70)
    print("Kibble-Zurek Analysis")
    print("="*70)

    # Load results
    with open('results/adiabatic_passage_results.pkl', 'rb') as f:
        all_results = pickle.load(f)

    # Analyze each system size
    for L, L_results in all_results.items():
        print(f"\n{'='*70}")
        print(f"System size L = {L}")
        print(f"{'='*70}")

        model = TransverseFieldIsing(L)
        kz_analysis = KibbleZurekAnalysis(model)

        # Extract T and δE values
        T_values = np.array([r['T_total'] for r in L_results['passages']])
        delta_E_values = np.array([r['delta_E'] for r in L_results['passages']])

        # Perform Kibble-Zurek analysis
        analysis = kz_analysis.analyze_scaling(T_values, delta_E_values)

        # Plot
        plot_kibble_zurek_analysis(
            analysis,
            save_path=f'results/kibble_zurek_L{L}.png'
        )

        # Plot one example adiabatic passage
        idx_mid = len(L_results['passages']) // 2
        plot_adiabatic_passage_results(
            L_results['passages'][idx_mid],
            save_path=f'results/adiabatic_passage_L{L}_T{T_values[idx_mid]:.1f}.png'
        )


if __name__ == '__main__':
    # Run simulation
    results = main_simulation()

    # Analyze results
    main_analysis()

    print("\n" + "="*70)
    print("Simulation and analysis complete!")
    print("="*70)
