"""
Simple example demonstrating the TFIM implementation.

This script shows basic usage of the transverse field Ising model
implementation for each of the three main tasks.

Author: Claude
Date: 2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
from ising_model_tfim import (
    TFIMParameters,
    TransverseFieldIsingModel,
    compute_structure_factor_order_parameter
)


def example_1_basic_usage():
    """Example 1: Basic TFIM computation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Create a small TFIM system
    params = TFIMParameters(L=6, J=1.0, h=0.8, periodic=False)
    tfim = TransverseFieldIsingModel(params)

    print(f"\nSystem parameters:")
    print(f"  System size L = {params.L}")
    print(f"  Coupling strength J = {params.J}")
    print(f"  Transverse field h = {params.h}")
    print(f"  Boundary conditions: {'Periodic' if params.periodic else 'Open'}")

    # Build and diagonalize
    print("\nBuilding Hamiltonian...")
    tfim.build_hamiltonian()

    print("Diagonalizing...")
    eigenvalues, _ = tfim.diagonalize()

    # Compute observables
    E0 = tfim.ground_state_energy()
    m = tfim.magnetization()

    print(f"\nGround state energy: E₀ = {E0:.6f}")
    print(f"Energy per site: E₀/L = {E0/params.L:.6f}")
    print(f"Magnetization: m = {m:.6f}")

    print(f"\nLowest 5 energy eigenvalues:")
    for i in range(min(5, len(eigenvalues))):
        print(f"  E_{i} = {eigenvalues[i]:.6f}")


def example_2_phase_transition():
    """Example 2: Scanning across the phase transition."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Phase Transition Scan")
    print("=" * 70)

    L = 8
    h_values = np.linspace(0.5, 1.5, 21)

    print(f"\nSystem size L = {L}")
    print(f"Scanning h from {h_values[0]:.2f} to {h_values[-1]:.2f}")
    print(f"Critical point: h_c ≈ 1.0")

    magnetizations = []
    energies = []

    print("\nComputing...")
    for h in h_values:
        params = TFIMParameters(L=L, J=1.0, h=h)
        tfim = TransverseFieldIsingModel(params)
        tfim.build_hamiltonian()
        tfim.diagonalize(k=1)

        m = abs(tfim.magnetization())
        E = tfim.ground_state_energy() / L

        magnetizations.append(m)
        energies.append(E)

    magnetizations = np.array(magnetizations)
    energies = np.array(energies)

    # Print results near critical point
    print("\nResults near critical point:")
    print("-" * 50)
    print(f"{'h':>8s} {'|m|':>12s} {'E/L':>12s}")
    print("-" * 50)
    for i, h in enumerate(h_values):
        if 0.8 <= h <= 1.2:
            print(f"{h:8.3f} {magnetizations[i]:12.6f} {energies[i]:12.6f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h_values, magnetizations, 'o-', markersize=6)
    ax1.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax1.set_xlabel('Transverse field h', fontsize=11)
    ax1.set_ylabel('Magnetization |m|', fontsize=11)
    ax1.set_title('Order Parameter', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(h_values, energies, 's-', markersize=6, color='green')
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax2.set_xlabel('Transverse field h', fontsize=11)
    ax2.set_ylabel('Energy per site E/L', fontsize=11)
    ax2.set_title('Ground State Energy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_phase_transition.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as: example_phase_transition.png")


def example_3_correlations():
    """Example 3: Correlation functions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Correlation Functions")
    print("=" * 70)

    L = 10
    h_values = [0.5, 1.0, 1.5]  # Ordered, critical, disordered

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, h in enumerate(h_values):
        print(f"\nTransverse field h = {h}")

        params = TFIMParameters(L=L, J=1.0, h=h)
        tfim = TransverseFieldIsingModel(params)
        tfim.build_hamiltonian()
        tfim.diagonalize(k=1)

        # Compute correlation function from site 0
        correlations = []
        for i in range(L):
            corr = tfim.correlation_function(0, i)
            correlations.append(corr)
            if i < 5:
                print(f"  C(0, {i}) = {corr:.6f}")

        # Plot
        ax = axes[idx]
        ax.plot(range(L), correlations, 'o-', markersize=6)
        ax.set_xlabel('Distance r', fontsize=10)
        ax.set_ylabel('C(0, r)', fontsize=10)
        if h < 1.0:
            phase = "Ordered"
        elif h == 1.0:
            phase = "Critical"
        else:
            phase = "Disordered"
        ax.set_title(f'h = {h:.1f} ({phase})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig('example_correlations.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as: example_correlations.png")


def example_4_structure_factor():
    """Example 4: Structure factor comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Structure Factor Method")
    print("=" * 70)

    L = 8
    h = 0.7

    print(f"\nSystem size L = {L}")
    print(f"Transverse field h = {h}")

    # Compute order parameter both ways
    m_direct, m_structure = compute_structure_factor_order_parameter(L, h)

    print(f"\nResults:")
    print(f"  Direct magnetization:        m = {m_direct:.8f}")
    print(f"  Structure factor method:     m = {m_structure:.8f}")
    print(f"  Relative difference:           = {abs(m_direct - m_structure)/m_direct * 100:.4f}%")

    # Also compute and show the structure factor itself
    params = TFIMParameters(L=L, J=1.0, h=h)
    tfim = TransverseFieldIsingModel(params)
    tfim.build_hamiltonian()
    tfim.diagonalize(k=1)

    q_values, S_q = tfim.structure_factor()

    print(f"\nStructure factor S(q):")
    print("-" * 40)
    print(f"{'q/π':>10s} {'S(q)':>15s}")
    print("-" * 40)
    for i in range(min(L, 8)):
        print(f"{q_values[i]/np.pi:10.4f} {S_q[i]:15.8f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(q_values/np.pi, S_q, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('Momentum q/π', fontsize=11)
    ax.set_ylabel('Structure factor S(q)', fontsize=11)
    ax.set_title(f'Structure Factor (L={L}, h={h})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2])

    plt.tight_layout()
    plt.savefig('example_structure_factor.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as: example_structure_factor.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("TRANSVERSE FIELD ISING MODEL - EXAMPLES")
    print("=" * 70)

    example_1_basic_usage()
    example_2_phase_transition()
    example_3_correlations()
    example_4_structure_factor()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nGenerated figures:")
    print("  - example_phase_transition.png")
    print("  - example_correlations.png")
    print("  - example_structure_factor.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
