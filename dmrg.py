"""
Time-Evolving Density Matrix Renormalization Group (t-DMRG)
时间演化密度矩阵重整化群

This module implements t-DMRG for imaginary time evolution to find ground states.
Specific implementation for the transverse field Ising model.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.linalg import expm
from mps import MPS
from tensor import Tensor


class IsingModel:
    """
    Transverse Field Ising Model Hamiltonian.

    H = -J Σ_i σ^z_i σ^z_{i+1} - h Σ_i σ^x_i

    Where:
    - J is the coupling strength (ferromagnetic if J > 0)
    - h is the transverse field strength
    - σ^x, σ^z are Pauli matrices
    """

    def __init__(self, L: int, J: float = 1.0, h: float = 0.5):
        """
        Initialize Ising model.

        Parameters:
        -----------
        L : int
            Number of sites
        J : float
            Coupling strength (default: 1.0)
        h : float
            Transverse field strength (default: 0.5)
        """
        self.L = L
        self.J = J
        self.h = h

        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)

        # Two-site Hamiltonian terms (for TEBD/t-DMRG)
        self._build_local_terms()

    def _build_local_terms(self):
        """Build local Hamiltonian terms."""
        # Two-site interaction: -J σ^z ⊗ σ^z
        self.h_zz = -self.J * np.kron(self.sigma_z, self.sigma_z)

        # Single-site field: -h σ^x
        self.h_x = -self.h * self.sigma_x

        # Two-site Hamiltonian for bonds (including half of neighboring field terms)
        # H_{i,i+1} = -J σ^z_i σ^z_{i+1} - (h/2) σ^x_i ⊗ I - (h/2) I ⊗ σ^x_{i+1}
        self.h_bond = (
            self.h_zz
            - (self.h / 2) * np.kron(self.sigma_x, self.identity)
            - (self.h / 2) * np.kron(self.identity, self.sigma_x)
        )

        # For boundary sites, we need to include full field term
        # Left boundary (site 0): includes full h σ^x_0
        self.h_bond_left = (
            -self.J * np.kron(self.sigma_z, self.sigma_z)
            - self.h * np.kron(self.sigma_x, self.identity)
            - (self.h / 2) * np.kron(self.identity, self.sigma_x)
        )

        # Right boundary (site L-1): includes full h σ^x_{L-1}
        self.h_bond_right = (
            -self.J * np.kron(self.sigma_z, self.sigma_z)
            - (self.h / 2) * np.kron(self.sigma_x, self.identity)
            - self.h * np.kron(self.identity, self.sigma_x)
        )

    def get_bond_hamiltonian(self, site: int) -> np.ndarray:
        """
        Get the two-site Hamiltonian for a bond.

        Parameters:
        -----------
        site : int
            Left site of the bond (acts on sites i and i+1)

        Returns:
        --------
        np.ndarray
            4×4 Hamiltonian matrix
        """
        if site == 0:
            return self.h_bond_left
        elif site == self.L - 2:
            return self.h_bond_right
        else:
            return self.h_bond

    def get_energy(self, mps: MPS) -> float:
        """
        Compute total energy of the Ising model for a given MPS.

        Parameters:
        -----------
        mps : MPS
            The quantum state

        Returns:
        --------
        float
            Total energy ⟨ψ|H|ψ⟩
        """
        if mps.L != self.L:
            raise ValueError(f"MPS length {mps.L} doesn't match model length {self.L}")

        energy = 0.0

        # Two-site terms: -J σ^z_i σ^z_{i+1}
        sigma_zz = np.kron(self.sigma_z, self.sigma_z).reshape(2, 2, 2, 2)
        for i in range(self.L - 1):
            exp_val = mps.expect_nn(sigma_zz, i)
            energy += -self.J * exp_val.real

        # One-site terms: -h σ^x_i
        for i in range(self.L):
            exp_val = mps.expect_local(self.sigma_x, i)
            energy += -self.h * exp_val.real

        return energy

    def exact_diagonalization(self) -> Tuple[float, np.ndarray]:
        """
        Compute ground state energy and wavefunction using exact diagonalization.

        Only feasible for small systems (L ≤ 20).

        Returns:
        --------
        energy : float
            Ground state energy
        wavefunction : np.ndarray
            Ground state wavefunction (1D array of length 2^L)
        """
        if self.L > 20:
            raise ValueError(f"Exact diagonalization not feasible for L={self.L} > 20")

        dim = 2 ** self.L
        H = np.zeros((dim, dim), dtype=complex)

        # Build full Hamiltonian
        # Single-site terms
        for i in range(self.L):
            op = 1
            for j in range(self.L):
                if j == i:
                    op = np.kron(op, self.sigma_x)
                else:
                    op = np.kron(op, self.identity)
            H += -self.h * op

        # Two-site terms
        for i in range(self.L - 1):
            op = 1
            for j in range(self.L):
                if j == i or j == i + 1:
                    op = np.kron(op, self.sigma_z)
                else:
                    op = np.kron(op, self.identity)
            H += -self.J * op

        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        ground_energy = eigenvalues[0].real
        ground_state = eigenvectors[:, 0]

        return ground_energy, ground_state


class DMRG:
    """
    Time-evolving DMRG (t-DMRG) implementation.

    Uses imaginary time evolution to find ground states.
    """

    def __init__(self, model: IsingModel, max_bond_dim: int = 50, cutoff: float = 1e-10):
        """
        Initialize DMRG.

        Parameters:
        -----------
        model : IsingModel
            The Hamiltonian
        max_bond_dim : int
            Maximum bond dimension
        cutoff : float
            Truncation cutoff for singular values
        """
        self.model = model
        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff

    def imaginary_time_evolution(
        self,
        mps: MPS,
        tau: float,
        n_steps: int,
        callback: Optional[Callable[[int, MPS, float], None]] = None,
        normalize_each_step: bool = True
    ) -> Tuple[MPS, List[float]]:
        """
        Perform imaginary time evolution using Trotter decomposition.

        Evolves: |ψ(τ)⟩ = e^{-τH} |ψ(0)⟩

        Uses second-order Trotter: e^{-τH} ≈ e^{-τH_even/2} e^{-τH_odd} e^{-τH_even/2}

        Parameters:
        -----------
        mps : MPS
            Initial state
        tau : float
            Time step size
        n_steps : int
            Number of time steps
        callback : callable, optional
            Function called after each step: callback(step, mps, energy)
        normalize_each_step : bool
            Whether to normalize after each step (default: True)

        Returns:
        --------
        final_mps : MPS
            Final evolved state
        energies : list of float
            Energy at each step
        """
        mps = mps.copy()
        energies = []

        # Build time evolution operators for each bond
        U_bonds = []
        for i in range(self.model.L - 1):
            H_bond = self.model.get_bond_hamiltonian(i)
            U = expm(-tau * H_bond)  # Imaginary time evolution operator
            U_bonds.append(U)

        for step in range(n_steps):
            # Second-order Trotter decomposition
            # Sweep 1: Even bonds (0, 2, 4, ...)
            for i in range(0, self.model.L - 1, 2):
                mps = self._apply_two_site_gate(mps, U_bonds[i], i)

            # Sweep 2: Odd bonds (1, 3, 5, ...)
            for i in range(1, self.model.L - 1, 2):
                mps = self._apply_two_site_gate(mps, U_bonds[i], i)

            # Sweep 3: Even bonds again (for symmetric Trotter)
            for i in range(0, self.model.L - 1, 2):
                mps = self._apply_two_site_gate(mps, U_bonds[i], i)

            # Normalize
            if normalize_each_step:
                mps.normalize()

            # Compute energy
            energy = self.model.get_energy(mps)
            energies.append(energy)

            # Callback
            if callback is not None:
                callback(step, mps, energy)

        return mps, energies

    def _apply_two_site_gate(self, mps: MPS, gate: np.ndarray, site: int) -> MPS:
        """
        Apply a two-site gate and truncate using SVD.

        Parameters:
        -----------
        mps : MPS
            Input MPS
        gate : np.ndarray
            Two-site gate (4×4 matrix or 2×2×2×2 tensor)
        site : int
            Left site index

        Returns:
        --------
        MPS
            Updated MPS
        """
        if site < 0 or site >= mps.L - 1:
            raise ValueError(f"Site {site} out of range")

        # Get two-site tensors
        A1 = mps.tensors[site].data      # (D_left, d, D_mid)
        A2 = mps.tensors[site + 1].data  # (D_mid, d, D_right)

        D_left, d1, D_mid = A1.shape
        D_mid2, d2, D_right = A2.shape

        assert D_mid == D_mid2, "Bond dimension mismatch"

        # Reshape gate if needed
        if gate.shape == (d1 * d2, d1 * d2):
            gate = gate.reshape(d1, d2, d1, d2)

        # Contract A1 and A2: (D_left, d1, D_mid) × (D_mid, d2, D_right) -> (D_left, d1, d2, D_right)
        theta = np.tensordot(A1, A2, axes=([2], [0]))

        # Apply gate: (d1, d2, d1', d2') × (D_left, d1', d2', D_right) -> (D_left, d1, d2, D_right)
        theta = np.tensordot(gate, theta, axes=([2, 3], [1, 2]))
        theta = np.transpose(theta, (2, 0, 1, 3))  # Reorder to (D_left, d1, d2, D_right)

        # Reshape for SVD: (D_left * d1, d2 * D_right)
        theta_mat = theta.reshape(D_left * d1, d2 * D_right)

        # SVD with truncation
        U, S, Vt = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate
        n_keep = min(len(S), self.max_bond_dim)
        # Also apply cutoff
        n_keep = min(n_keep, np.sum(S > self.cutoff * S[0]))
        n_keep = max(1, n_keep)

        U = U[:, :n_keep]
        S = S[:n_keep]
        Vt = Vt[:n_keep, :]

        # Absorb S into Vt (could also split or put in U)
        Vt = np.diag(S) @ Vt

        # Reshape back to MPS tensors
        A1_new = U.reshape(D_left, d1, n_keep)
        A2_new = Vt.reshape(n_keep, d2, D_right)

        # Update MPS
        mps_new = mps.copy()
        mps_new.tensors[site] = Tensor(A1_new)
        mps_new.tensors[site + 1] = Tensor(A2_new)
        mps_new.bond_dims[site + 1] = n_keep

        return mps_new

    def run_ground_state(
        self,
        initial_mps: Optional[MPS] = None,
        tau: float = 0.1,
        n_steps: int = 100,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 10,
        verbose: bool = True
    ) -> Tuple[MPS, float, List[float]]:
        """
        Find ground state using imaginary time evolution.

        Parameters:
        -----------
        initial_mps : MPS, optional
            Initial state (if None, creates random MPS)
        tau : float
            Time step size
        n_steps : int
            Number of steps per iteration
        convergence_threshold : float
            Convergence criterion for energy
        max_iterations : int
            Maximum number of iterations
        verbose : bool
            Print progress

        Returns:
        --------
        ground_mps : MPS
            Ground state MPS
        ground_energy : float
            Ground state energy
        energy_history : list of float
            Energy at each step
        """
        # Initialize MPS if not provided
        if initial_mps is None:
            if verbose:
                print("Creating random initial MPS...")
            mps = MPS.random(L=self.model.L, d=2, D=self.max_bond_dim, seed=42)
            mps.normalize()
        else:
            mps = initial_mps.copy()

        energy_history = []
        prev_energy = float('inf')

        if verbose:
            print(f"Starting imaginary time evolution:")
            print(f"  tau = {tau}, n_steps per iteration = {n_steps}")
            print(f"  max_bond_dim = {self.max_bond_dim}, cutoff = {self.cutoff}")
            print()

        for iteration in range(max_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{max_iterations}")

            # Perform imaginary time evolution
            def step_callback(step, mps_current, energy):
                energy_history.append(energy)
                if verbose and (step + 1) % 10 == 0:
                    print(f"  Step {step + 1}/{n_steps}: E = {energy:.10f}")

            mps, iter_energies = self.imaginary_time_evolution(
                mps, tau, n_steps, callback=step_callback
            )

            # Check convergence
            current_energy = iter_energies[-1]
            energy_change = abs(current_energy - prev_energy)

            if verbose:
                print(f"  Energy change: {energy_change:.2e}")
                print()

            if energy_change < convergence_threshold:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break

            prev_energy = current_energy

        ground_energy = self.model.get_energy(mps)

        return mps, ground_energy, energy_history


def compare_with_exact(model: IsingModel, dmrg_energy: float, dmrg_mps: MPS, verbose: bool = True):
    """
    Compare DMRG result with exact diagonalization.

    Parameters:
    -----------
    model : IsingModel
        The Hamiltonian
    dmrg_energy : float
        DMRG ground state energy
    dmrg_mps : MPS
        DMRG ground state
    verbose : bool
        Print comparison

    Returns:
    --------
    dict
        Comparison results
    """
    if model.L > 20:
        if verbose:
            print(f"Exact diagonalization not feasible for L={model.L}")
        return None

    exact_energy, exact_wavefunction = model.exact_diagonalization()

    results = {
        'exact_energy': exact_energy,
        'dmrg_energy': dmrg_energy,
        'energy_error': abs(dmrg_energy - exact_energy),
        'relative_error': abs(dmrg_energy - exact_energy) / abs(exact_energy)
    }

    if verbose:
        print("=" * 70)
        print("Comparison with Exact Diagonalization")
        print("=" * 70)
        print(f"Exact energy:         {exact_energy:.10f}")
        print(f"DMRG energy:          {dmrg_energy:.10f}")
        print(f"Absolute error:       {results['energy_error']:.2e}")
        print(f"Relative error:       {results['relative_error']:.2e}")
        print("=" * 70)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("t-DMRG for Transverse Field Ising Model")
    print("=" * 70)

    # Setup
    L = 10
    J = 1.0
    h = 0.5

    print(f"\nModel parameters:")
    print(f"  L = {L} (number of sites)")
    print(f"  J = {J} (coupling)")
    print(f"  h = {h} (transverse field)")
    print()

    # Create model
    model = IsingModel(L=L, J=J, h=h)

    # Run DMRG
    dmrg = DMRG(model, max_bond_dim=50, cutoff=1e-10)

    ground_mps, ground_energy, energy_history = dmrg.run_ground_state(
        tau=0.1,
        n_steps=50,
        max_iterations=5,
        convergence_threshold=1e-8,
        verbose=True
    )

    print(f"\nFinal ground state energy: {ground_energy:.10f}")
    print(f"Final MPS bond dimensions: {ground_mps.bond_dims}")

    # Compare with exact diagonalization (if feasible)
    if L <= 12:
        print("\n")
        compare_with_exact(model, ground_energy, ground_mps, verbose=True)

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
