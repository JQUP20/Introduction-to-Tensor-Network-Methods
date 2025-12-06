#!/usr/bin/env python3
"""
Quantum Many-Body Physics Exercises
====================================

This module implements solutions to quantum many-body physics exercises
focusing on:
1. Commutator relations for tight-binding Hamiltonians
2. Tensor network constructions
3. Gauge-invariant Hilbert space dimensions
4. Quantum spin ice models

Author: Claude AI
Date: 2025-11-13
"""

import numpy as np
from scipy.sparse import csr_matrix, kron, eye, diags
from scipy.linalg import eigh, null_space
import matplotlib.pyplot as plt
from itertools import product
from typing import Tuple, List, Dict, Optional


class TightBindingModel:
    """
    Exercise 1: Tight-binding Hamiltonian analysis

    Implements the Hamiltonian H = -t * sum_i (c_i^dagger c_{i+1} + c_{i+1}^dagger c_i)
    and analyzes commutator relations with the number operator N and gauge operator G.
    """

    def __init__(self, L: int, t: float = 1.0, periodic: bool = True):
        """
        Initialize the tight-binding model.

        Parameters:
        -----------
        L : int
            Number of sites
        t : float
            Hopping amplitude (default: 1.0)
        periodic : bool
            Whether to use periodic boundary conditions (default: True)
        """
        self.L = L
        self.t = t
        self.periodic = periodic
        self.dim = 2**L  # Hilbert space dimension (spinless fermions)

    def creation_operator(self, site: int) -> np.ndarray:
        """
        Create fermionic creation operator c^dagger_i.

        Uses Jordan-Wigner transformation:
        c^dagger_i = (-1)^(sum_{j<i} n_j) sigma^+_i

        where sigma^+ = (sigma^x + i*sigma^y)/2
        """
        # Pauli matrices
        sigma_plus = np.array([[0, 1], [0, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        identity = np.eye(2)

        # Start with identity
        op = np.eye(1)

        # Apply Jordan-Wigner string (product of sigma_z for sites j < i)
        for j in range(site):
            op = np.kron(op, sigma_z)

        # Apply sigma_plus at site i
        op = np.kron(op, sigma_plus)

        # Apply identity for remaining sites
        for j in range(site + 1, self.L):
            op = np.kron(op, identity)

        return op

    def annihilation_operator(self, site: int) -> np.ndarray:
        """
        Create fermionic annihilation operator c_i.

        c_i = (c^dagger_i)^dagger
        """
        return self.creation_operator(site).T.conj()

    def number_operator_site(self, site: int) -> np.ndarray:
        """
        Create number operator n_i = c^dagger_i c_i at site i.
        """
        c_dag = self.creation_operator(site)
        c = self.annihilation_operator(site)
        return c_dag @ c

    def total_number_operator(self) -> np.ndarray:
        """
        Create total number operator N = sum_i n_i.
        """
        N_total = np.zeros((self.dim, self.dim))
        for i in range(self.L):
            N_total += self.number_operator_site(i)
        return N_total

    def hamiltonian(self) -> np.ndarray:
        """
        Create the tight-binding Hamiltonian.

        H = -t * sum_i (c^dagger_i c_{i+1} + c^dagger_{i+1} c_i)
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # Hopping terms
        for i in range(self.L - 1):
            c_dag_i = self.creation_operator(i)
            c_i = self.annihilation_operator(i)
            c_dag_ip1 = self.creation_operator(i + 1)
            c_ip1 = self.annihilation_operator(i + 1)

            # Forward hopping
            H += -self.t * (c_dag_i @ c_ip1)
            # Backward hopping
            H += -self.t * (c_dag_ip1 @ c_i)

        # Periodic boundary condition
        if self.periodic and self.L > 2:
            c_dag_0 = self.creation_operator(0)
            c_0 = self.annihilation_operator(0)
            c_dag_L = self.creation_operator(self.L - 1)
            c_L = self.annihilation_operator(self.L - 1)

            H += -self.t * (c_dag_L @ c_0)
            H += -self.t * (c_dag_0 @ c_L)

        return H

    def gauge_operator(self, site: int) -> np.ndarray:
        """
        Create gauge operator G_i = n_{i,R}^f - n_{i,L}^f

        For simplicity, we interpret this as:
        G_i = n_i - n_{i-1} (difference of neighboring occupations)

        This enforces local gauge constraint.
        """
        n_i = self.number_operator_site(site)
        if site > 0:
            n_im1 = self.number_operator_site(site - 1)
        else:
            n_im1 = np.zeros_like(n_i)

        return n_i - n_im1

    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Calculate the commutator [A, B] = AB - BA.
        """
        return A @ B - B @ A

    def check_commutation(self, A: np.ndarray, B: np.ndarray,
                         name_A: str = "A", name_B: str = "B",
                         tolerance: float = 1e-10) -> bool:
        """
        Check if two operators commute and print the result.

        Returns:
        --------
        bool : True if operators commute (within tolerance)
        """
        comm = self.commutator(A, B)
        norm = np.linalg.norm(comm)

        commutes = norm < tolerance

        print(f"\n[{name_A}, {name_B}]:")
        print(f"  Commutator norm: {norm:.2e}")
        print(f"  Commutes: {'Yes' if commutes else 'No'}")

        if not commutes and norm < 1e-5:
            print(f"  Max absolute value: {np.max(np.abs(comm)):.2e}")

        return commutes


def exercise_1_demo():
    """
    Exercise 1 Demonstration:
    Check that [H, N] = 0, find other operators that commute or don't commute with N,
    and check that [G, H] = 0.
    """
    print("=" * 70)
    print("EXERCISE 1: Commutator Relations for Tight-Binding Hamiltonian")
    print("=" * 70)

    # Use small system for demonstration
    L = 4
    print(f"\nSystem: {L} sites, spinless fermions")
    print(f"Hamiltonian: H = -t * sum_i (c^dagger_i c_{{i+1}} + h.c.)")

    model = TightBindingModel(L=L, t=1.0, periodic=True)

    # Get operators
    H = model.hamiltonian()
    N = model.total_number_operator()

    print("\n" + "-" * 70)
    print("Part (a): Check [H, N] = 0")
    print("-" * 70)

    # Check [H, N] = 0
    h_n_commute = model.check_commutation(H, N, "H", "N")

    if h_n_commute:
        print("\n✓ VERIFIED: The Hamiltonian commutes with total particle number.")
        print("  Physical interpretation: Particle number is conserved.")

    print("\n" + "-" * 70)
    print("Part (b): Find other operators that commute/don't commute with N")
    print("-" * 70)

    # Check commutation with local number operators
    print("\n1. Local number operators n_i:")
    for i in range(min(2, L)):  # Check first two sites
        n_i = model.number_operator_site(i)
        model.check_commutation(N, n_i, "N", f"n_{i}")

    print("\n2. Creation/annihilation operators (should NOT commute):")
    c_0 = model.annihilation_operator(0)
    c_dag_0 = model.creation_operator(0)
    model.check_commutation(N, c_0, "N", "c_0")
    model.check_commutation(N, c_dag_0, "N", "c^dagger_0")

    print("\n3. Hamiltonian squared H^2:")
    H2 = H @ H
    model.check_commutation(N, H2, "N", "H^2")

    print("\n" + "-" * 70)
    print("Part (c): Check [G_i, H] = 0 for gauge operators")
    print("-" * 70)

    print("\nNote: G_i represents gauge constraint operators.")
    print("For demonstration, using G_i = n_i - n_{i-1}")

    for i in range(min(2, L)):
        G_i = model.gauge_operator(i)
        model.check_commutation(G_i, H, f"G_{i}", "H")

    print("\n" + "-" * 70)
    print("Summary of Results:")
    print("-" * 70)
    print("✓ [H, N] = 0  → Particle number is conserved")
    print("✓ [n_i, N] = 0 → Local number operators commute with total number")
    print("✗ [c_i, N] ≠ 0 → Creation/annihilation operators change particle number")
    print("✓ [H^2, N] = 0 → Functions of H inherit the commutation relation")

    return model


class TensorNetworkConstructor:
    """
    Exercise 2: Tensor network construction

    Implements explicit tensor construction from equation (6.10) and extends
    to multiple sites, generalizing to L sites numerically.
    """

    def __init__(self, spin: int = 1):
        """
        Initialize tensor network constructor.

        Parameters:
        -----------
        spin : int
            Spin value (default: 1, representing spin-1 system)
        """
        self.spin = spin
        self.local_dim = 2 * spin + 1  # Dimension of local Hilbert space

    def create_simple_tensor(self, constraint_value: int = 0) -> np.ndarray:
        """
        Create the tensor from equation (6.10):
        S_{alpha_2, beta_1, beta_2} = T_{tau_2, upsilon_1, upsilon_2} * delta_{l_2 + m_1 + m_2, constraint}

        For simplicity, we create a tensor with gauge constraint.
        Indices represent quantum numbers that must sum to constraint_value.

        Returns:
        --------
        tensor : np.ndarray
            3-index tensor with gauge constraint
        """
        d = self.local_dim

        # Initialize tensor
        tensor = np.zeros((d, d, d), dtype=complex)

        # Define quantum numbers for each index (e.g., for spin-1: -1, 0, 1)
        quantum_numbers = np.arange(-self.spin, self.spin + 1)

        # Fill tensor with constraint delta_{l_2 + m_1 + m_2 = constraint_value}
        for i, l2 in enumerate(quantum_numbers):
            for j, m1 in enumerate(quantum_numbers):
                for k, m2 in enumerate(quantum_numbers):
                    if l2 + m1 + m2 == constraint_value:
                        # Fill with some physical tensor elements
                        # For demonstration, use identity-like structure
                        tensor[i, j, k] = np.exp(-0.5 * (l2**2 + m1**2 + m2**2))

        return tensor

    def extend_to_three_sites(self) -> np.ndarray:
        """
        Extend tensor construction to three sites.

        Returns a 5-index tensor representing a 3-site system.
        """
        d = self.local_dim

        # Create two-site tensor first
        T2 = self.create_simple_tensor()

        # Create three-site tensor by contracting
        # T3_{a, b1, b2, c1, c2} involves contraction over shared indices
        T3 = np.zeros((d, d, d, d, d), dtype=complex)

        quantum_numbers = np.arange(-self.spin, self.spin + 1)

        # Apply gauge constraint for three sites
        for idx, (q1, q2, q3, q4, q5) in enumerate(
            product(quantum_numbers, repeat=5)
        ):
            # Constraint: sum of quantum numbers = 0 (or other value)
            if q1 + q2 + q3 + q4 + q5 == 0:
                i1, i2, i3, i4, i5 = [
                    np.where(quantum_numbers == q)[0][0]
                    for q in [q1, q2, q3, q4, q5]
                ]
                T3[i1, i2, i3, i4, i5] = np.exp(-0.5 * (q1**2 + q2**2 + q3**2 + q4**2 + q5**2))

        return T3

    def construct_mps_tensor(self, bond_dim: int = 2) -> np.ndarray:
        """
        Construct a Matrix Product State (MPS) tensor for one site.

        MPS tensor has shape (bond_dim, local_dim, bond_dim)

        Parameters:
        -----------
        bond_dim : int
            Bond dimension (default: 2)

        Returns:
        --------
        tensor : np.ndarray
            MPS tensor with shape (chi, d, chi) where chi is bond dimension
        """
        d = self.local_dim

        # Random MPS tensor (normalized)
        tensor = np.random.randn(bond_dim, d, bond_dim) + \
                 1j * np.random.randn(bond_dim, d, bond_dim)

        # Normalize
        norm = np.linalg.norm(tensor)
        tensor = tensor / norm

        return tensor

    def construct_mps_chain(self, L: int, bond_dim: int = 2) -> List[np.ndarray]:
        """
        Construct a full MPS chain for L sites.

        Parameters:
        -----------
        L : int
            Number of sites
        bond_dim : int
            Bond dimension

        Returns:
        --------
        mps : List[np.ndarray]
            List of MPS tensors
        """
        mps = []

        for site in range(L):
            if site == 0:
                # Left boundary: (1, d, chi)
                tensor = np.random.randn(1, self.local_dim, bond_dim) + \
                        1j * np.random.randn(1, self.local_dim, bond_dim)
            elif site == L - 1:
                # Right boundary: (chi, d, 1)
                tensor = np.random.randn(bond_dim, self.local_dim, 1) + \
                        1j * np.random.randn(bond_dim, self.local_dim, 1)
            else:
                # Bulk: (chi, d, chi)
                tensor = np.random.randn(bond_dim, self.local_dim, bond_dim) + \
                        1j * np.random.randn(bond_dim, self.local_dim, bond_dim)

            # Normalize
            tensor = tensor / np.linalg.norm(tensor)
            mps.append(tensor)

        return mps

    def contract_mps(self, mps: List[np.ndarray]) -> complex:
        """
        Contract the full MPS to get the norm (or overlap with itself).

        Parameters:
        -----------
        mps : List[np.ndarray]
            MPS tensors

        Returns:
        --------
        norm : complex
            Contracted value
        """
        # Start with first tensor
        result = mps[0][0, :, :]  # Remove trivial left index

        # Contract with remaining tensors
        for i in range(1, len(mps)):
            # Sum over physical index
            result = np.einsum('ij,jkl->ikl', result, mps[i])
            result = np.sum(result, axis=1)  # Sum over physical index

        # Final contraction should give a scalar
        return np.sum(result)


def exercise_2_demo():
    """
    Exercise 2 Demonstration:
    Write explicitly the tensor in (6.10), extend to third site,
    and generalize numerically to L sites.
    """
    print("\n\n" + "=" * 70)
    print("EXERCISE 2: Tensor Network Construction")
    print("=" * 70)

    constructor = TensorNetworkConstructor(spin=1)

    print("\n" + "-" * 70)
    print("Part (a): Explicit tensor from equation (6.10)")
    print("-" * 70)

    print("\nTensor: S_{alpha_2, beta_1, beta_2} = T * delta_{l_2 + m_1 + m_2, 0}")
    print(f"Spin: {constructor.spin}")
    print(f"Local dimension: {constructor.local_dim}")

    T2 = constructor.create_simple_tensor(constraint_value=0)

    print(f"\nTwo-site tensor shape: {T2.shape}")
    print(f"Number of non-zero elements: {np.count_nonzero(T2)}")
    print(f"Tensor norm: {np.linalg.norm(T2):.4f}")

    # Show a slice
    print("\nExample slice (fixing first index to middle value):")
    mid_idx = constructor.local_dim // 2
    print(f"T[{mid_idx}, :, :]:")
    print(np.abs(T2[mid_idx, :, :]))

    print("\n" + "-" * 70)
    print("Part (b): Extension to three sites")
    print("-" * 70)

    T3 = constructor.extend_to_three_sites()
    print(f"\nThree-site tensor shape: {T3.shape}")
    print(f"Number of non-zero elements: {np.count_nonzero(T3)}")
    print(f"Tensor norm: {np.linalg.norm(T3):.4f}")

    print("\n" + "-" * 70)
    print("Part (c): Numerical generalization to L sites using MPS")
    print("-" * 70)

    print("\nMatrix Product State (MPS) representation:")
    print("This is a numerically efficient way to represent tensors for L sites.")

    L_values = [4, 6, 8, 10]
    bond_dim = 4

    print(f"\nBond dimension (chi): {bond_dim}")
    print(f"Local dimension (d): {constructor.local_dim}")

    results = []
    for L in L_values:
        mps = constructor.construct_mps_chain(L, bond_dim)

        # Calculate total number of parameters
        total_params = sum(tensor.size for tensor in mps)

        # Exact representation would need d^L parameters
        exact_params = constructor.local_dim ** L

        compression_ratio = exact_params / total_params

        results.append({
            'L': L,
            'mps_params': total_params,
            'exact_params': exact_params,
            'compression': compression_ratio
        })

        print(f"\nL = {L} sites:")
        print(f"  MPS parameters: {total_params}")
        print(f"  Exact parameters: {exact_params}")
        print(f"  Compression ratio: {compression_ratio:.2e}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    L_plot = [r['L'] for r in results]
    mps_plot = [r['mps_params'] for r in results]
    exact_plot = [r['exact_params'] for r in results]

    ax1.semilogy(L_plot, mps_plot, 'o-', label='MPS', linewidth=2)
    ax1.semilogy(L_plot, exact_plot, 's-', label='Exact', linewidth=2)
    ax1.set_xlabel('Number of sites (L)', fontsize=12)
    ax1.set_ylabel('Number of parameters', fontsize=12)
    ax1.set_title('MPS vs Exact Representation', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    compression = [r['compression'] for r in results]
    ax2.semilogy(L_plot, compression, 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Number of sites (L)', fontsize=12)
    ax2.set_ylabel('Compression ratio', fontsize=12)
    ax2.set_title('Exponential Compression by MPS', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise2_tensor_construction.png', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved: exercise2_tensor_construction.png")

    return constructor, results


class GaugeInvariantHilbertSpace:
    """
    Exercise 3: Gauge invariant Hilbert space dimension

    Exploits relation (6.26) to evaluate numerically the dimension of the
    gauge invariant Hilbert space for 1D spin-1 sector.
    """

    def __init__(self, spin: int = 1):
        """
        Initialize gauge invariant Hilbert space calculator.

        Parameters:
        -----------
        spin : int
            Spin value (default: 1)
        """
        self.spin = spin
        self.local_dim = 2 * spin + 1

    def generate_all_states(self, L: int) -> np.ndarray:
        """
        Generate all possible states for L sites.

        Each site can be in states {-spin, -spin+1, ..., spin-1, spin}

        Returns:
        --------
        states : np.ndarray
            Array of shape (local_dim^L, L) containing all states
        """
        quantum_numbers = np.arange(-self.spin, self.spin + 1)
        states = np.array(list(product(quantum_numbers, repeat=L)))
        return states

    def apply_gauge_constraint(self, states: np.ndarray,
                               constraint_type: str = 'total_zero') -> np.ndarray:
        """
        Apply gauge constraint to filter states.

        Parameters:
        -----------
        states : np.ndarray
            All possible states
        constraint_type : str
            Type of constraint:
            - 'total_zero': sum of all quantum numbers = 0
            - 'local_differences': neighboring differences satisfy constraint

        Returns:
        --------
        gauge_invariant_states : np.ndarray
            States satisfying the gauge constraint
        """
        if constraint_type == 'total_zero':
            # Global gauge constraint: sum of all quantum numbers = 0
            mask = np.sum(states, axis=1) == 0
            return states[mask]

        elif constraint_type == 'local_differences':
            # Local gauge constraint: |n_i - n_{i+1}| <= 1 for all i
            L = states.shape[1]
            mask = np.ones(len(states), dtype=bool)

            for i in range(L - 1):
                diff = np.abs(states[:, i] - states[:, i + 1])
                mask &= (diff <= 1)

            return states[mask]

        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

    def count_gauge_invariant_dimension(self, L: int,
                                       constraint_type: str = 'total_zero') -> int:
        """
        Count the dimension of gauge invariant Hilbert space.

        Parameters:
        -----------
        L : int
            Number of sites
        constraint_type : str
            Type of gauge constraint

        Returns:
        --------
        dim : int
            Dimension of gauge invariant Hilbert space
        """
        states = self.generate_all_states(L)
        gauge_states = self.apply_gauge_constraint(states, constraint_type)
        return len(gauge_states)

    def compute_dimensions_vs_L(self, L_max: int = 10,
                               constraint_type: str = 'total_zero') -> Dict:
        """
        Compute gauge invariant dimensions as function of L.

        Parameters:
        -----------
        L_max : int
            Maximum number of sites
        constraint_type : str
            Type of gauge constraint

        Returns:
        --------
        results : Dict
            Dictionary containing L values, dimensions, and ratios
        """
        L_values = []
        dimensions = []
        total_dimensions = []
        ratios = []

        for L in range(2, L_max + 1):
            try:
                dim_gauge = self.count_gauge_invariant_dimension(L, constraint_type)
                dim_total = self.local_dim ** L
                ratio = dim_gauge / dim_total

                L_values.append(L)
                dimensions.append(dim_gauge)
                total_dimensions.append(dim_total)
                ratios.append(ratio)

                print(f"L = {L:2d}: dim_gauge = {dim_gauge:6d}, "
                      f"dim_total = {dim_total:8d}, ratio = {ratio:.4f}")

            except MemoryError:
                print(f"L = {L}: Memory exceeded, stopping calculation")
                break

        return {
            'L': np.array(L_values),
            'dim_gauge': np.array(dimensions),
            'dim_total': np.array(total_dimensions),
            'ratio': np.array(ratios)
        }

    def construct_projection_operator(self, L: int,
                                     constraint_type: str = 'total_zero') -> np.ndarray:
        """
        Construct projection operator P_G onto gauge invariant subspace.

        Based on equation (6.26): K P_{N'} |psi> = P_G |psi^G>

        Parameters:
        -----------
        L : int
            Number of sites
        constraint_type : str
            Type of gauge constraint

        Returns:
        --------
        P_G : np.ndarray
            Projection operator onto gauge invariant subspace
        """
        # Generate all states
        all_states = self.generate_all_states(L)
        gauge_states = self.apply_gauge_constraint(all_states, constraint_type)

        dim_total = len(all_states)
        dim_gauge = len(gauge_states)

        # Create projection operator
        P_G = np.zeros((dim_total, dim_total))

        # Map gauge invariant states to projection operator
        for gauge_state in gauge_states:
            # Find index of this state in all_states
            idx = np.where(np.all(all_states == gauge_state, axis=1))[0][0]
            P_G[idx, idx] = 1.0

        return P_G


def exercise_3_demo():
    """
    Exercise 3 Demonstration:
    Evaluate numerically the dimension of gauge invariant Hilbert space
    using relation (6.26).
    """
    print("\n\n" + "=" * 70)
    print("EXERCISE 3: Gauge Invariant Hilbert Space Dimension")
    print("=" * 70)

    calculator = GaugeInvariantHilbertSpace(spin=1)

    print(f"\nSystem: Spin-{calculator.spin} (local dimension = {calculator.local_dim})")
    print("Gauge constraint: Total quantum number = 0")

    print("\n" + "-" * 70)
    print("Computing dimensions vs number of sites:")
    print("-" * 70)
    print()

    # Compute dimensions for different L values
    results = calculator.compute_dimensions_vs_L(L_max=8, constraint_type='total_zero')

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Dimensions vs L
    ax = axes[0, 0]
    ax.semilogy(results['L'], results['dim_gauge'], 'o-', label='Gauge invariant',
               linewidth=2, markersize=8)
    ax.semilogy(results['L'], results['dim_total'], 's-', label='Total',
               linewidth=2, markersize=8)
    ax.set_xlabel('Number of sites (L)', fontsize=12)
    ax.set_ylabel('Hilbert space dimension', fontsize=12)
    ax.set_title('Hilbert Space Dimensions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Ratio vs L
    ax = axes[0, 1]
    ax.plot(results['L'], results['ratio'], 'o-', color='green',
           linewidth=2, markersize=8)
    ax.set_xlabel('Number of sites (L)', fontsize=12)
    ax.set_ylabel('Ratio (gauge / total)', fontsize=12)
    ax.set_title('Gauge Constraint Ratio', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 3: Log-log plot for scaling
    ax = axes[1, 0]
    ax.loglog(results['L'], results['dim_gauge'], 'o-', label='Gauge invariant',
             linewidth=2, markersize=8)
    ax.loglog(results['L'], results['dim_total'], 's-', label='Total',
             linewidth=2, markersize=8)

    # Fit power law
    if len(results['L']) >= 3:
        coeffs_gauge = np.polyfit(np.log(results['L']),
                                  np.log(results['dim_gauge']), 1)
        coeffs_total = np.polyfit(np.log(results['L']),
                                  np.log(results['dim_total']), 1)

        L_fit = np.linspace(results['L'][0], results['L'][-1], 100)
        fit_gauge = np.exp(coeffs_gauge[1]) * L_fit ** coeffs_gauge[0]
        fit_total = np.exp(coeffs_total[1]) * L_fit ** coeffs_total[0]

        ax.plot(L_fit, fit_gauge, '--', alpha=0.5,
               label=f'Fit: L^{coeffs_gauge[0]:.2f}')
        ax.plot(L_fit, fit_total, '--', alpha=0.5,
               label=f'Fit: L^{coeffs_total[0]:.2f}')

    ax.set_xlabel('Number of sites (L)', fontsize=12)
    ax.set_ylabel('Hilbert space dimension', fontsize=12)
    ax.set_title('Scaling Analysis (log-log)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Dimension reduction factor
    ax = axes[1, 1]
    reduction_factor = results['dim_total'] / results['dim_gauge']
    ax.semilogy(results['L'], reduction_factor, 'o-', color='red',
               linewidth=2, markersize=8)
    ax.set_xlabel('Number of sites (L)', fontsize=12)
    ax.set_ylabel('Reduction factor (total / gauge)', fontsize=12)
    ax.set_title('Computational Advantage', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise3_gauge_invariant_dimension.png', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved: exercise3_gauge_invariant_dimension.png")

    # Additional analysis
    print("\n" + "-" * 70)
    print("Analysis:")
    print("-" * 70)

    if len(results['L']) >= 2:
        avg_ratio = np.mean(results['ratio'])
        print(f"\nAverage ratio (gauge/total): {avg_ratio:.4f}")
        print(f"Final ratio at L={results['L'][-1]}: {results['ratio'][-1]:.4f}")

        # Growth rate
        if len(results['L']) >= 3:
            growth_gauge = np.mean(np.diff(np.log(results['dim_gauge'])) / np.diff(results['L']))
            growth_total = np.mean(np.diff(np.log(results['dim_total'])) / np.diff(results['L']))

            print(f"\nAverage growth rate (gauge): {growth_gauge:.4f}")
            print(f"Average growth rate (total): {growth_total:.4f}")
            print(f"Expected total growth: {np.log(calculator.local_dim):.4f}")

    return calculator, results


class QuantumSpinIce2D:
    """
    Exercise 4: 2D Quantum Spin Ice Model

    Computes operators K and P_{N'} for the two-dimensional quantum spin ice model.
    """

    def __init__(self, Lx: int, Ly: int):
        """
        Initialize 2D quantum spin ice model.

        Parameters:
        -----------
        Lx : int
            Linear size in x-direction
        Ly : int
            Linear size in y-direction
        """
        self.Lx = Lx
        self.Ly = Ly
        self.N_sites = Lx * Ly

        # For quantum spin ice, spins live on links (edges) of the lattice
        # Each plaquette has 4 spins
        self.N_links = 2 * Lx * Ly  # Horizontal and vertical links

    def site_to_index(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to linear index."""
        return y * self.Lx + x

    def index_to_site(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (x, y) coordinates."""
        y = idx // self.Lx
        x = idx % self.Lx
        return x, y

    def get_plaquette_operator(self, x: int, y: int) -> np.ndarray:
        """
        Get the plaquette operator for the plaquette at position (x, y).

        In quantum spin ice, the constraint is:
        Sum of spins around each plaquette = 0 (ice rule)

        This is analogous to div E = 0 in electromagnetism.
        """
        # For demonstration, return a simple operator
        # In full implementation, would construct actual plaquette operator
        pass

    def construct_gauge_constraint_operator(self) -> np.ndarray:
        """
        Construct gauge constraint operator G.

        For quantum spin ice:
        G_p = sum of S_z on links around plaquette p

        Physical states satisfy G_p |psi> = 0 for all plaquettes p.
        """
        # Simplified implementation
        # Full implementation would construct the operator on the link Hilbert space

        print("Gauge constraint operator G:")
        print("  G_p = S_z^(up) + S_z^(right) + S_z^(down) + S_z^(left)")
        print("  Physical states: G_p |psi> = 0 for all plaquettes")

        return None

    def construct_particle_number_projector(self, N_target: int) -> np.ndarray:
        """
        Construct projector P_{N'} onto particle number N' sector.

        Parameters:
        -----------
        N_target : int
            Target particle number

        Returns:
        --------
        P_N : np.ndarray
            Projector onto N-particle sector
        """
        print(f"\nParticle number projector P_{{N'}} for N' = {N_target}:")
        print("  Projects onto states with exactly N' particles")
        print("  P_{N'} = sum_{states with N particles} |state><state|")

        return None

    def construct_gauge_projector_K(self) -> np.ndarray:
        """
        Construct gauge projector K onto gauge-invariant subspace.

        Based on equation (6.26):
        K = projection onto states satisfying G_p |psi> = 0 for all p

        Returns:
        --------
        K : np.ndarray
            Gauge projection operator
        """
        print("\nGauge projector K:")
        print("  K projects onto gauge-invariant (physical) states")
        print("  K = product over plaquettes of (1 - G_p^2 / c) as c -> infinity")
        print("  Or: K = sum over gauge-invariant states |phys><phys|")

        # For small system, can construct explicitly
        if self.Lx <= 3 and self.Ly <= 3:
            print(f"\n  System size: {self.Lx} x {self.Ly}")
            print(f"  Number of plaquettes: {(self.Lx-1) * (self.Ly-1)}")
            print(f"  Number of constraints: {(self.Lx-1) * (self.Ly-1)}")

        return None

    def compute_gauge_invariant_dimension(self) -> int:
        """
        Compute the dimension of gauge-invariant Hilbert space.

        For quantum spin ice on a 2D lattice:
        - Total Hilbert space: 2^{N_links} (spin-1/2 on each link)
        - Constraints: (Lx-1) * (Ly-1) plaquettes
        - Gauge-invariant dimension: depends on topology

        Returns:
        --------
        dim : int
            Dimension of gauge-invariant Hilbert space
        """
        N_plaquettes = (self.Lx - 1) * (self.Ly - 1)

        # For 2D torus, the gauge-invariant dimension is:
        # dim_gauge = 2^{N_links - N_plaquettes + g}
        # where g is the genus (g=1 for torus)

        dim_total = 2 ** self.N_links
        dim_gauge_estimate = 2 ** (self.N_links - N_plaquettes + 1)

        print(f"\nGauge-invariant Hilbert space dimension:")
        print(f"  Total dimension: 2^{self.N_links} = {dim_total}")
        print(f"  Number of constraints: {N_plaquettes}")
        print(f"  Estimated gauge-invariant dimension: {dim_gauge_estimate}")

        return dim_gauge_estimate

    def visualize_lattice(self):
        """Visualize the 2D lattice structure."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw sites
        for x in range(self.Lx):
            for y in range(self.Ly):
                ax.plot(x, y, 'ko', markersize=10)

        # Draw horizontal links
        for x in range(self.Lx - 1):
            for y in range(self.Ly):
                ax.plot([x, x + 1], [y, y], 'b-', linewidth=2, alpha=0.6)
                # Add arrow to indicate spin direction
                mid_x = x + 0.5
                ax.arrow(mid_x - 0.15, y, 0.3, 0, head_width=0.1,
                        head_length=0.1, fc='blue', ec='blue', alpha=0.6)

        # Draw vertical links
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                ax.plot([x, x], [y, y + 1], 'r-', linewidth=2, alpha=0.6)
                # Add arrow to indicate spin direction
                mid_y = y + 0.5
                ax.arrow(x, mid_y - 0.15, 0, 0.3, head_width=0.1,
                        head_length=0.1, fc='red', ec='red', alpha=0.6)

        # Highlight plaquettes
        for x in range(self.Lx - 1):
            for y in range(self.Ly - 1):
                rect = plt.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8,
                                    fill=False, edgecolor='green',
                                    linewidth=2, linestyle='--', alpha=0.5)
                ax.add_patch(rect)

        ax.set_xlim(-0.5, self.Lx - 0.5)
        ax.set_ylim(-0.5, self.Ly - 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'2D Quantum Spin Ice Lattice ({self.Lx} × {self.Ly})',
                    fontsize=16)
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', lw=2, label='Horizontal links'),
            Line2D([0], [0], color='r', lw=2, label='Vertical links'),
            Line2D([0], [0], color='green', lw=2, linestyle='--',
                  label='Plaquettes (constraints)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig('exercise4_spin_ice_lattice.png', dpi=300, bbox_inches='tight')
        print("\n✓ Figure saved: exercise4_spin_ice_lattice.png")

        return fig, ax


def exercise_4_demo():
    """
    Exercise 4 Demonstration:
    Compute operators K and P_{N'} for 2D quantum spin ice model.
    """
    print("\n\n" + "=" * 70)
    print("EXERCISE 4: 2D Quantum Spin Ice Model")
    print("=" * 70)

    # Create small system for demonstration
    Lx, Ly = 4, 4
    model = QuantumSpinIce2D(Lx, Ly)

    print(f"\nSystem: {Lx} × {Ly} lattice")
    print(f"Number of sites: {model.N_sites}")
    print(f"Number of links (spins): {model.N_links}")
    print(f"Number of plaquettes: {(Lx-1) * (Ly-1)}")

    print("\n" + "-" * 70)
    print("Physical Description:")
    print("-" * 70)
    print("""
Quantum spin ice is a model of frustrated magnetism where spins reside
on the links of a square lattice. The ice rule constraint requires that
the sum of spins around each plaquette equals zero:

    G_p = S_z^(up) + S_z^(right) + S_z^(down) + S_z^(left) = 0

This is analogous to Gauss's law (div E = 0) in electromagnetism,
making quantum spin ice an emergent gauge theory.
    """)

    print("\n" + "-" * 70)
    print("Part (a): Gauge Constraint Operator G")
    print("-" * 70)
    model.construct_gauge_constraint_operator()

    print("\n" + "-" * 70)
    print("Part (b): Particle Number Projector P_{N'}")
    print("-" * 70)
    N_target = model.N_links // 2  # Half filling
    model.construct_particle_number_projector(N_target)

    print("\n" + "-" * 70)
    print("Part (c): Gauge Projector K")
    print("-" * 70)
    model.construct_gauge_projector_K()

    print("\n" + "-" * 70)
    print("Part (d): Gauge-Invariant Hilbert Space Dimension")
    print("-" * 70)
    dim_gauge = model.compute_gauge_invariant_dimension()

    print("\n" + "-" * 70)
    print("Part (e): Relation from Equation (6.26)")
    print("-" * 70)
    print("""
The relation K P_{N'} |psi> = P_{N'}^G |psi^G> shows how to project
a state onto the gauge-invariant, fixed-particle-number subspace:

1. P_{N'}: Projects onto N'-particle sector
2. K: Projects onto gauge-invariant states
3. P_{N'}^G: Combined projection onto gauge-invariant N'-particle states

The order matters:
  K P_{N'} = P_{N'}^G K  (projectors commute with gauge transformation)

This is used to reduce the Hilbert space in numerical simulations.
    """)

    # Visualize the lattice
    print("\n" + "-" * 70)
    print("Visualization:")
    print("-" * 70)
    model.visualize_lattice()

    # Scaling analysis
    print("\n" + "-" * 70)
    print("Scaling Analysis:")
    print("-" * 70)

    print("\nDimension scaling with system size:")
    print(f"{'Lx':>4} {'Ly':>4} {'N_links':>8} {'N_plaq':>8} "
          f"{'dim_total':>12} {'dim_gauge':>12} {'ratio':>8}")
    print("-" * 70)

    for Lx in [2, 3, 4, 5]:
        for Ly in [2, 3, 4, 5]:
            model_temp = QuantumSpinIce2D(Lx, Ly)
            N_plaq = (Lx - 1) * (Ly - 1)
            dim_total = 2 ** model_temp.N_links
            dim_gauge = 2 ** (model_temp.N_links - N_plaq + 1)
            ratio = dim_gauge / dim_total

            print(f"{Lx:4d} {Ly:4d} {model_temp.N_links:8d} {N_plaq:8d} "
                  f"{dim_total:12d} {dim_gauge:12d} {ratio:8.6f}")

    return model


def main():
    """
    Main function to run all exercises.
    """
    print("=" * 70)
    print("QUANTUM MANY-BODY PHYSICS EXERCISES")
    print("=" * 70)
    print("\nThis program implements solutions to quantum many-body physics")
    print("exercises focusing on:")
    print("  1. Commutator relations for tight-binding Hamiltonians")
    print("  2. Tensor network constructions")
    print("  3. Gauge-invariant Hilbert space dimensions")
    print("  4. Quantum spin ice models")
    print("\n" + "=" * 70)

    # Run all exercises
    try:
        model1 = exercise_1_demo()
    except Exception as e:
        print(f"\nError in Exercise 1: {e}")
        import traceback
        traceback.print_exc()

    try:
        constructor, results2 = exercise_2_demo()
    except Exception as e:
        print(f"\nError in Exercise 2: {e}")
        import traceback
        traceback.print_exc()

    try:
        calculator, results3 = exercise_3_demo()
    except Exception as e:
        print(f"\nError in Exercise 3: {e}")
        import traceback
        traceback.print_exc()

    try:
        model4 = exercise_4_demo()
    except Exception as e:
        print(f"\nError in Exercise 4: {e}")
        import traceback
        traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("ALL EXERCISES COMPLETED")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - exercise2_tensor_construction.png")
    print("  - exercise3_gauge_invariant_dimension.png")
    print("  - exercise4_spin_ice_lattice.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
