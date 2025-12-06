"""
Comprehensive Demo: t-DMRG for Transverse Field Ising Model
完整演示：横场伊辛模型的 t-DMRG

This script demonstrates:
1. Tensor class operations (contraction, SVD, compression)
2. MPS operations (norm, expectation values)
3. t-DMRG ground state search
4. Comparison with exact diagonalization
5. Phase transition investigation (varying h/J)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Tuple

from tensor import Tensor
from mps import MPS
from dmrg import IsingModel, DMRG, compare_with_exact


def demo_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("=" * 80)
    print("PART 1: Tensor Class Operations")
    print("=" * 80)

    print("\n1.1 Tensor Creation and Contraction")
    print("-" * 80)
    A = Tensor.random((3, 4, 5), seed=42)
    B = Tensor.random((4, 5, 6), seed=43)
    print(f"Tensor A: shape {A.shape}")
    print(f"Tensor B: shape {B.shape}")

    C = A.contract(B, [1, 2], [0, 1])
    print(f"Contraction C = A[i,j,k] * B[j,k,l]: shape {C.shape}")
    print(f"Result norm: {C.norm():.6f}")

    print("\n1.2 Singular Value Decomposition")
    print("-" * 80)
    T = Tensor.random((6, 8, 10), seed=44)
    print(f"Original tensor: shape {T.shape}")

    U, S, Vt = T.svd([0], [1, 2])
    print(f"SVD decomposition:")
    print(f"  U shape: {U.shape}")
    print(f"  Singular values (first 5): {S[:5]}")
    print(f"  Vt shape: {Vt.shape}")

    # Reconstruction error
    S_diag = Tensor(np.diag(S))
    T_reconstructed = U.contract(S_diag, 1, 0).contract(Vt, 1, 0)
    error = np.linalg.norm(T.data - T_reconstructed.data)
    print(f"  Reconstruction error: {error:.2e}")

    print("\n1.3 Tensor Compression")
    print("-" * 80)
    T_large = Tensor.random((10, 10, 10), seed=45)
    print(f"Original tensor: shape {T_large.shape}, norm {T_large.norm():.6f}")

    for max_bond in [8, 5, 3]:
        left, right = T_large.compress([0], [1, 2], max_bond_dim=max_bond)
        T_approx = left.contract(right, -1, 0)
        rel_error = np.linalg.norm(T_large.data - T_approx.data) / T_large.norm()
        print(f"  Bond dim {max_bond}: shapes {left.shape}, {right.shape}, "
              f"relative error {rel_error:.2e}")


def demo_mps_operations():
    """Demonstrate MPS operations."""
    print("\n" + "=" * 80)
    print("PART 2: Matrix Product State (MPS) Operations")
    print("=" * 80)

    print("\n2.1 Random MPS Creation and Norm")
    print("-" * 80)
    L = 8
    d = 2
    D = 10

    mps = MPS.random(L=L, d=d, D=D, seed=42)
    print(f"Random MPS: {mps}")
    print(f"Norm: {mps.norm():.6f}")

    original_norm = mps.norm()
    mps.normalize()
    print(f"After normalization: {mps.norm():.6f}")

    print("\n2.2 Product State")
    print("-" * 80)
    spin_up = np.array([1.0, 0.0], dtype=complex)
    spin_down = np.array([0.0, 1.0], dtype=complex)

    # All spins up
    mps_up = MPS.product_state([spin_up] * L)
    print(f"All-up state: norm = {mps_up.norm():.6f}")

    # Alternating spins
    states = [spin_up if i % 2 == 0 else spin_down for i in range(L)]
    mps_alt = MPS.product_state(states)
    print(f"Alternating state: norm = {mps_alt.norm():.6f}")

    print("\n2.3 Local Operator Expectation Values")
    print("-" * 80)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    print("All-up state ⟨σ_z⟩:")
    for i in range(min(4, L)):
        exp_z = mps_up.expect_local(sigma_z, i)
        print(f"  Site {i}: {exp_z.real:.6f}")

    print("\nAlternating state ⟨σ_z⟩:")
    for i in range(min(4, L)):
        exp_z = mps_alt.expect_local(sigma_z, i)
        print(f"  Site {i}: {exp_z.real:.6f}")

    print("\n2.4 Nearest-Neighbor Correlations")
    print("-" * 80)
    sigma_zz = np.kron(sigma_z, sigma_z).reshape(2, 2, 2, 2)

    print("All-up state ⟨σ_z σ_z⟩:")
    for i in range(min(3, L - 1)):
        exp_zz = mps_up.expect_nn(sigma_zz, i)
        print(f"  Bond ({i},{i+1}): {exp_zz.real:.6f}")


def run_dmrg_single(L: int, J: float, h: float, max_bond_dim: int = 50,
                   verbose: bool = True) -> Tuple[float, MPS, IsingModel]:
    """
    Run DMRG for a single set of parameters.

    Returns:
    --------
    energy : float
        Ground state energy
    mps : MPS
        Ground state MPS
    model : IsingModel
        The Hamiltonian
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Running DMRG: L={L}, J={J:.2f}, h={h:.2f}")
        print(f"{'=' * 80}")

    model = IsingModel(L=L, J=J, h=h)
    dmrg = DMRG(model, max_bond_dim=max_bond_dim, cutoff=1e-10)

    ground_mps, ground_energy, energy_history = dmrg.run_ground_state(
        tau=0.1,
        n_steps=50,
        max_iterations=5,
        convergence_threshold=1e-8,
        verbose=verbose
    )

    if verbose:
        print(f"\nFinal ground state energy: {ground_energy:.10f}")
        print(f"Final bond dimensions: {ground_mps.bond_dims}")

    return ground_energy, ground_mps, model


def demo_dmrg_ground_state():
    """Demonstrate DMRG ground state calculation."""
    print("\n" + "=" * 80)
    print("PART 3: t-DMRG Ground State Calculation")
    print("=" * 80)

    # Small system for exact comparison
    L = 10
    J = 1.0
    h = 0.5

    energy, mps, model = run_dmrg_single(L, J, h, max_bond_dim=50, verbose=True)

    # Compare with exact
    if L <= 12:
        print("\n")
        results = compare_with_exact(model, energy, mps, verbose=True)

    return energy, mps, model


def phase_transition_study():
    """Study the quantum phase transition by varying h/J."""
    print("\n" + "=" * 80)
    print("PART 4: Phase Transition Study (varying h/J)")
    print("=" * 80)

    L = 10
    J = 1.0
    h_values = np.linspace(0.1, 2.0, 10)

    print(f"\nSystem size: L = {L}")
    print(f"Coupling: J = {J}")
    print(f"Transverse field range: h = {h_values[0]:.2f} to {h_values[-1]:.2f}")
    print()

    dmrg_energies = []
    exact_energies = []
    magnetizations = []

    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    for i, h in enumerate(h_values):
        print(f"\n[{i + 1}/{len(h_values)}] h = {h:.3f}")
        print("-" * 80)

        # Run DMRG
        energy, mps, model = run_dmrg_single(L, J, h, max_bond_dim=50, verbose=False)
        dmrg_energies.append(energy)

        # Compute magnetization ⟨σ_z⟩
        mag = np.mean([mps.expect_local(sigma_z, site).real for site in range(L)])
        magnetizations.append(abs(mag))

        print(f"  Ground state energy: {energy:.10f}")
        print(f"  Magnetization |⟨σ_z⟩|: {abs(mag):.6f}")

        # Exact diagonalization for comparison
        if L <= 10:
            exact_energy, _ = model.exact_diagonalization()
            exact_energies.append(exact_energy)
            error = abs(energy - exact_energy)
            print(f"  Exact energy: {exact_energy:.10f}")
            print(f"  Error: {error:.2e}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Energy plot
    ax1.plot(h_values, dmrg_energies, 'o-', label='DMRG', linewidth=2, markersize=6)
    if exact_energies:
        ax1.plot(h_values, exact_energies, 's--', label='Exact', linewidth=2, markersize=4)
    ax1.set_xlabel('Transverse field h', fontsize=12)
    ax1.set_ylabel('Ground state energy', fontsize=12)
    ax1.set_title(f'Ising Model Ground State Energy (L={L}, J={J})', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Magnetization plot
    ax2.plot(h_values, magnetizations, 'o-', color='red', linewidth=2, markersize=6)
    ax2.axvline(x=J, color='gray', linestyle='--', alpha=0.5, label='Critical point (h=J)')
    ax2.set_xlabel('Transverse field h', fontsize=12)
    ax2.set_ylabel('Magnetization |⟨σ_z⟩|', fontsize=12)
    ax2.set_title(f'Order Parameter (L={L}, J={J})', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ising_phase_transition.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved as: ising_phase_transition.png")

    return h_values, dmrg_energies, magnetizations


def scaling_analysis():
    """Analyze how DMRG scales with system size."""
    print("\n" + "=" * 80)
    print("PART 5: Scaling Analysis (varying L)")
    print("=" * 80)

    J = 1.0
    h = 0.5
    L_values = [6, 8, 10, 12]

    print(f"\nFixed parameters: J = {J}, h = {h}")
    print(f"System sizes: L = {L_values}")
    print()

    results = []

    for L in L_values:
        print(f"\nL = {L}")
        print("-" * 80)

        energy, mps, model = run_dmrg_single(L, J, h, max_bond_dim=50, verbose=False)

        # Energy per site
        energy_per_site = energy / L

        result = {
            'L': L,
            'energy': energy,
            'energy_per_site': energy_per_site,
            'max_bond_dim': max(mps.bond_dims)
        }
        results.append(result)

        print(f"  Energy: {energy:.10f}")
        print(f"  Energy per site: {energy_per_site:.10f}")
        print(f"  Max bond dimension: {result['max_bond_dim']}")

        # Compare with exact if feasible
        if L <= 12:
            exact_energy, _ = model.exact_diagonalization()
            error = abs(energy - exact_energy)
            print(f"  Exact energy: {exact_energy:.10f}")
            print(f"  Error: {error:.2e}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    L_vals = [r['L'] for r in results]
    energies = [r['energy'] for r in results]
    energies_per_site = [r['energy_per_site'] for r in results]
    bond_dims = [r['max_bond_dim'] for r in results]

    ax1.plot(L_vals, energies_per_site, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('System size L', fontsize=12)
    ax1.set_ylabel('Energy per site', fontsize=12)
    ax1.set_title(f'Energy Scaling (J={J}, h={h})', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(L_vals, bond_dims, 's-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('System size L', fontsize=12)
    ax2.set_ylabel('Maximum bond dimension', fontsize=12)
    ax2.set_title('Bond Dimension Growth', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ising_scaling_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved as: ising_scaling_analysis.png")

    return results


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 15 + "t-DMRG for Transverse Field Ising Model" + " " * 23 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    # Part 1: Tensor operations
    demo_tensor_operations()

    # Part 2: MPS operations
    demo_mps_operations()

    # Part 3: DMRG ground state
    demo_dmrg_ground_state()

    # Part 4: Phase transition
    phase_transition_study()

    # Part 5: Scaling analysis
    scaling_analysis()

    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "ALL DEMONSTRATIONS COMPLETE!" + " " * 29 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print("\nGenerated files:")
    print("  - ising_phase_transition.png")
    print("  - ising_scaling_analysis.png")
    print()


if __name__ == "__main__":
    main()
