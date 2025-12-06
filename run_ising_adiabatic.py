"""
Example script to run adiabatic passage simulation for the TFIM

This script demonstrates:
1. Single adiabatic passage simulation
2. Parameter sweep over passage times
3. Kibble-Zurek scaling analysis
4. Critical exponent extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from ising_adiabatic_dmrg import (
    TransverseFieldIsing,
    GroundStateDMRG,
    AdiabaticPassage,
    KibbleZurekAnalysis,
    DMRGParameters,
    Measurements,
    plot_adiabatic_passage_results,
    plot_kibble_zurek_analysis
)
import os


def example_single_passage():
    """
    Example 1: Single adiabatic passage simulation
    """
    print("\n" + "="*70)
    print("Example 1: Single Adiabatic Passage")
    print("="*70)

    # System parameters
    L = 20
    h_initial = 2.0
    h_final = 0.5
    T_total = 10.0

    # DMRG parameters
    params = DMRGParameters(
        chi_max=40,
        chi_init=10,
        n_sweeps=8,
        convergence_tol=1e-7,
        dt=0.05
    )

    # Create model and run simulation
    model = TransverseFieldIsing(L)
    adiabatic = AdiabaticPassage(model, params)

    print(f"\nSystem size: L = {L}")
    print(f"Initial field: h_i = {h_initial}")
    print(f"Final field: h_f = {h_final}")
    print(f"Total time: T = {T_total}")
    print(f"Time step: dt = {params.dt}")
    print(f"Max bond dimension: χ_max = {params.chi_max}")

    # Run simulation
    result = adiabatic.simulate(h_initial, h_final, T_total, save_every=5)

    # Get exact ground state for comparison
    print("\nComputing exact ground state...")
    dmrg_exact = GroundStateDMRG(model, params)
    mps_exact, E_exact = dmrg_exact.run(h_final)
    xi_exact = Measurements.measure_correlation_length(mps_exact)

    print("\n" + "-"*70)
    print("Comparison with exact ground state:")
    print("-"*70)
    print(f"Final energy (adiabatic):  {result['final_energy']:.8f}")
    print(f"Ground state energy:       {E_exact:.8f}")
    print(f"Energy difference:         {result['final_energy'] - E_exact:.8e}")
    print(f"\nFinal correlation length (adiabatic): {result['final_xi']:.6f}")
    print(f"Ground state correlation length:      {xi_exact:.6f}")
    print(f"Correlation length difference:        {result['final_xi'] - xi_exact:.6f}")

    # Plot results
    os.makedirs('results', exist_ok=True)
    plot_adiabatic_passage_results(
        result,
        save_path='results/example_single_passage.png'
    )

    return result, E_exact


def example_kibble_zurek_scaling():
    """
    Example 2: Kibble-Zurek scaling analysis
    """
    print("\n" + "="*70)
    print("Example 2: Kibble-Zurek Scaling Analysis")
    print("="*70)

    # System parameters
    L = 20
    h_initial = 2.0
    h_final = 0.5

    # Range of passage times (log-spaced)
    T_values = np.logspace(-0.3, 1.5, 8)  # From ~0.5 to ~30

    # DMRG parameters
    params = DMRGParameters(
        chi_max=40,
        chi_init=10,
        n_sweeps=6,
        convergence_tol=1e-6,
        dt=0.05
    )

    # Create model
    model = TransverseFieldIsing(L)
    adiabatic = AdiabaticPassage(model, params)

    # Get exact ground state energy
    print("\nComputing exact ground state...")
    dmrg_exact = GroundStateDMRG(model, params)
    _, E_exact = dmrg_exact.run(h_final)
    print(f"Exact ground state energy: {E_exact:.8f}")

    # Run simulations for different T values
    print(f"\nRunning {len(T_values)} simulations with different passage times...")
    print(f"T values: {T_values}")

    delta_E_values = []

    for i, T in enumerate(T_values):
        print(f"\n[{i+1}/{len(T_values)}] Simulating with T = {T:.3f}...")

        result = adiabatic.simulate(h_initial, h_final, T, save_every=10)

        # Compute residual energy
        delta_E = result['final_energy'] - E_exact
        delta_E_values.append(delta_E)

        print(f"  Final energy: {result['final_energy']:.8f}")
        print(f"  Residual energy: δE = {delta_E:.6e}")

    delta_E_values = np.array(delta_E_values)

    # Kibble-Zurek analysis
    print("\n" + "-"*70)
    print("Kibble-Zurek Analysis")
    print("-"*70)

    kz_analysis = KibbleZurekAnalysis(model)
    analysis_results = kz_analysis.analyze_scaling(T_values, delta_E_values)

    # Print results
    if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
        alpha_sim = analysis_results['kz_params']['alpha']
        alpha_theory = 0.5  # Theoretical value for 1D TFIM

        print(f"\nKibble-Zurek regime detected:")
        print(f"  Fitted exponent: α = {alpha_sim:.4f}")
        print(f"  Theoretical prediction: α = {alpha_theory:.4f}")
        print(f"  Relative error: {abs(alpha_sim - alpha_theory)/alpha_theory * 100:.2f}%")

        print(f"\nTheoretical background:")
        print(f"  For the 1D TFIM at criticality:")
        print(f"    - Dynamical exponent: z = 1")
        print(f"    - Correlation length exponent: ν = 1")
        print(f"    - Dimension: d = 1")
        print(f"    - KZ scaling: δE ~ T^(-dν/(1+zν)) = T^(-1/2)")

    if 'lz_params' in analysis_results and analysis_results['lz_params'] is not None:
        print(f"\nLandau-Zener regime detected:")
        print(f"  Exponential decay rate: a = {analysis_results['lz_params']['a']:.4f}")
        print(f"  This regime corresponds to fast passages where δE ~ exp(-aT)")

    # Plot results
    os.makedirs('results', exist_ok=True)
    plot_kibble_zurek_analysis(
        analysis_results,
        save_path='results/example_kibble_zurek.png'
    )

    return T_values, delta_E_values, analysis_results


def example_system_size_comparison():
    """
    Example 3: Comparison of different system sizes
    """
    print("\n" + "="*70)
    print("Example 3: System Size Comparison")
    print("="*70)

    # System parameters
    L_values = [10, 15, 20]
    h_initial = 2.0
    h_final = 0.5
    T_values = np.logspace(0, 1.5, 6)  # From 1 to ~30

    # DMRG parameters
    params = DMRGParameters(
        chi_max=40,
        chi_init=10,
        n_sweeps=6,
        convergence_tol=1e-6,
        dt=0.05
    )

    # Storage
    all_results = {}

    for L in L_values:
        print(f"\n{'='*70}")
        print(f"System size L = {L}")
        print(f"{'='*70}")

        model = TransverseFieldIsing(L)
        adiabatic = AdiabaticPassage(model, params)

        # Get exact ground state
        dmrg_exact = GroundStateDMRG(model, params)
        _, E_exact = dmrg_exact.run(h_final)

        delta_E_values = []

        for i, T in enumerate(T_values):
            print(f"\n[{i+1}/{len(T_values)}] T = {T:.2f}...")

            result = adiabatic.simulate(h_initial, h_final, T, save_every=10)
            delta_E = result['final_energy'] - E_exact
            delta_E_values.append(delta_E)

            print(f"  δE = {delta_E:.6e}")

        all_results[L] = {
            'T_values': T_values,
            'delta_E_values': np.array(delta_E_values),
            'E_exact': E_exact
        }

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for L in L_values:
        T_vals = all_results[L]['T_values']
        dE_vals = all_results[L]['delta_E_values']

        ax1.loglog(T_vals, dE_vals, 'o-', label=f'L = {L}', linewidth=2, markersize=8)

        # Also plot normalized by system size
        ax2.loglog(T_vals, dE_vals / L, 'o-', label=f'L = {L}', linewidth=2, markersize=8)

    # Add theoretical scaling
    T_theory = np.logspace(0, 1.5, 50)
    dE_theory = 0.1 * T_theory**(-0.5)
    ax1.loglog(T_theory, dE_theory, 'k--', linewidth=2, label='T^(-1/2) (theory)')
    ax2.loglog(T_theory, dE_theory / 15, 'k--', linewidth=2, label='T^(-1/2) (theory)')

    ax1.set_xlabel('Passage Time T', fontsize=12)
    ax1.set_ylabel('Residual Energy δE', fontsize=12)
    ax1.set_title('Absolute Residual Energy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Passage Time T', fontsize=12)
    ax2.set_ylabel('δE / L', fontsize=12)
    ax2.set_title('Energy Density', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/example_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to results/example_size_comparison.png")
    plt.show()

    return all_results


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("ADIABATIC PASSAGE THROUGH QUANTUM CRITICAL POINT")
    print("Transverse Field Ising Model - t-DMRG Simulation")
    print("="*70)

    # Example 1: Single passage
    result_single, E_exact = example_single_passage()

    # Example 2: Kibble-Zurek scaling
    T_vals, dE_vals, analysis = example_kibble_zurek_scaling()

    # Example 3: System size comparison
    size_comparison = example_system_size_comparison()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nResults saved to the 'results/' directory:")
    print("  - example_single_passage.png")
    print("  - example_kibble_zurek.png")
    print("  - example_size_comparison.png")


if __name__ == '__main__':
    main()
