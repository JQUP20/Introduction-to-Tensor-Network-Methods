"""
TFIM Analysis Script - All Three Tasks

This script performs the three main tasks:
1. Mean field critical exponents with m=1
2. Order parameter comparison (direct vs structure factor)
3. Finite size scaling with RG and DMRG

Author: Claude
Date: 2025-11-13
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from ising_model_tfim import (
    TFIMParameters,
    TransverseFieldIsingModel,
    compute_mean_field_exponents,
    compute_structure_factor_order_parameter,
    finite_size_scaling_analysis,
    SimpleDMRG
)

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def task1_mean_field_critical_exponents():
    """
    Task 1: Compute mean field critical exponents using tensor networks (m=1).

    The critical point for 1D TFIM is at h_c = J = 1.
    Mean field critical exponents:
    - β = 1/2 (order parameter: m ~ |h - h_c|^β)
    - ν = 1/2 (correlation length: ξ ~ |h - h_c|^(-ν))
    """
    print("=" * 80)
    print("TASK 1: Mean Field Critical Exponents (m=1)")
    print("=" * 80)

    # Scan across the critical point
    h_values = np.linspace(0.2, 1.8, 40)
    L = 10  # Small system size for m=1 (exact diagonalization)

    print(f"\nComputing for L = {L}...")
    print(f"Scanning h from {h_values[0]:.2f} to {h_values[-1]:.2f}")
    print(f"Critical point: h_c = 1.0")

    results = compute_mean_field_exponents(h_values, L=L, bond_dim=1)

    # Analytical prediction: m ~ (h_c - h)^β with β = 1/2 (mean field)
    h_c = 1.0
    beta_mf = 0.5

    # Fit the exponent near the critical point
    # Use points in the ordered phase (h < h_c)
    mask_ordered = (h_values < h_c) & (h_values > 0.5)
    if np.sum(mask_ordered) > 2:
        h_fit = h_values[mask_ordered]
        m_fit = results['magnetizations'][mask_ordered]

        # Remove zeros
        mask_nonzero = m_fit > 1e-6
        h_fit = h_fit[mask_nonzero]
        m_fit = m_fit[mask_nonzero]

        if len(h_fit) > 2:
            # Log-log fit: log(m) = β * log(h_c - h) + const
            x = np.log(h_c - h_fit)
            y = np.log(m_fit)
            coeffs = np.polyfit(x, y, 1)
            beta_fit = coeffs[0]

            print(f"\nFitted exponent β = {beta_fit:.4f}")
            print(f"Analytical mean field β = {beta_mf:.4f}")
            print(f"Relative error: {abs(beta_fit - beta_mf) / beta_mf * 100:.2f}%")
        else:
            beta_fit = None
            print("\nNot enough points for fitting")
    else:
        beta_fit = None
        print("\nNot enough points for fitting")

    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, figure=fig)

    # Plot 1: Magnetization vs h
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(h_values, results['magnetizations'], 'o-', label='Computed (m=1)', markersize=4)
    ax1.axvline(h_c, color='r', linestyle='--', alpha=0.7, label=f'Critical point h_c={h_c}')
    ax1.set_xlabel('Transverse field h', fontsize=11)
    ax1.set_ylabel('Magnetization |m|', fontsize=11)
    ax1.set_title('Order Parameter vs Field', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy vs h
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(h_values, results['energies'], 's-', color='green', label='Ground state energy', markersize=4)
    ax2.axvline(h_c, color='r', linestyle='--', alpha=0.7, label=f'h_c={h_c}')
    ax2.set_xlabel('Transverse field h', fontsize=11)
    ax2.set_ylabel('Energy per site e₀', fontsize=11)
    ax2.set_title('Ground State Energy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log-log plot for exponent
    ax3 = fig.add_subplot(gs[2])
    if mask_ordered.any():
        h_ordered = h_values[mask_ordered]
        m_ordered = results['magnetizations'][mask_ordered]
        mask_pos = m_ordered > 1e-6

        if mask_pos.any():
            ax3.loglog(h_c - h_ordered[mask_pos], m_ordered[mask_pos], 'o',
                      label='Computed', markersize=6)

            # Analytical curve
            h_theory = np.linspace(0.1, 0.8, 100)
            m_theory = (h_c - h_theory)**beta_mf
            ax3.loglog(h_c - h_theory, m_theory, '--', color='red',
                      label=f'Mean field β={beta_mf}', linewidth=2)

            if beta_fit is not None:
                # Fitted curve
                h_fit_curve = np.linspace(h_fit[0], h_fit[-1], 100)
                m_fit_curve = np.exp(coeffs[1]) * (h_c - h_fit_curve)**beta_fit
                ax3.loglog(h_c - h_fit_curve, m_fit_curve, ':', color='blue',
                          label=f'Fitted β={beta_fit:.3f}', linewidth=2)

    ax3.set_xlabel('|h - h_c|', fontsize=11)
    ax3.set_ylabel('Magnetization |m|', fontsize=11)
    ax3.set_title('Critical Scaling (log-log)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('task1_mean_field_exponents.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: task1_mean_field_exponents.png")

    return results, beta_fit


def task2_order_parameter_comparison():
    """
    Task 2: Compare order parameter calculated directly and from structure factor.

    Direct: m = (1/L) Σᵢ <σᵢᶻ>
    Structure factor: S(q) = (1/L) Σᵢⱼ exp(iq(i-j)) <σᵢᶻ σⱼᶻ>
    For ferromagnetic order: S(q=0) ~ L * m²
    """
    print("\n" + "=" * 80)
    print("TASK 2: Order Parameter - Direct vs Structure Factor")
    print("=" * 80)

    # Test for various system sizes and field values
    L_values = [4, 6, 8, 10]
    h_values = np.linspace(0.3, 1.5, 15)

    results = {L: {'h': [], 'm_direct': [], 'm_structure': [], 'ratio': []}
               for L in L_values}

    for L in L_values:
        print(f"\nComputing for L = {L}...")
        for h in h_values:
            m_direct, m_structure = compute_structure_factor_order_parameter(L, h)
            results[L]['h'].append(h)
            results[L]['m_direct'].append(m_direct)
            results[L]['m_structure'].append(m_structure)
            if m_direct > 1e-10:
                results[L]['ratio'].append(m_structure / m_direct)
            else:
                results[L]['ratio'].append(np.nan)

        # Convert to arrays
        for key in ['h', 'm_direct', 'm_structure', 'ratio']:
            results[L][key] = np.array(results[L][key])

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot 1: Direct magnetization for all L
    ax1 = fig.add_subplot(gs[0, 0])
    for L in L_values:
        ax1.plot(results[L]['h'], results[L]['m_direct'], 'o-',
                label=f'L={L}', markersize=4)
    ax1.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax1.set_xlabel('Transverse field h', fontsize=11)
    ax1.set_ylabel('Direct magnetization', fontsize=11)
    ax1.set_title('Direct Method', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Structure factor magnetization
    ax2 = fig.add_subplot(gs[0, 1])
    for L in L_values:
        ax2.plot(results[L]['h'], results[L]['m_structure'], 's-',
                label=f'L={L}', markersize=4)
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax2.set_xlabel('Transverse field h', fontsize=11)
    ax2.set_ylabel('Structure factor magnetization', fontsize=11)
    ax2.set_title('Structure Factor Method', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison for L=10
    ax3 = fig.add_subplot(gs[1, 0])
    L_plot = 10
    ax3.plot(results[L_plot]['h'], results[L_plot]['m_direct'], 'o-',
            label='Direct', markersize=6, linewidth=2)
    ax3.plot(results[L_plot]['h'], results[L_plot]['m_structure'], 's-',
            label='Structure factor', markersize=6, linewidth=2, alpha=0.7)
    ax3.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax3.set_xlabel('Transverse field h', fontsize=11)
    ax3.set_ylabel('Magnetization', fontsize=11)
    ax3.set_title(f'Comparison for L={L_plot}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Ratio
    ax4 = fig.add_subplot(gs[1, 1])
    for L in L_values:
        ratio = results[L]['ratio']
        mask = ~np.isnan(ratio) & (results[L]['m_direct'] > 1e-3)
        if mask.any():
            ax4.plot(results[L]['h'][mask], ratio[mask], 'o-',
                    label=f'L={L}', markersize=4)
    ax4.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Perfect agreement')
    ax4.axvline(1.0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Transverse field h', fontsize=11)
    ax4.set_ylabel('Ratio: Structure / Direct', fontsize=11)
    ax4.set_title('Method Comparison Ratio', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.5, 1.5])

    plt.tight_layout()
    plt.savefig('task2_order_parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: task2_order_parameter_comparison.png")

    # Print statistics
    print("\n" + "-" * 60)
    print("Comparison Statistics (h < 1.0, m > 0.01):")
    print("-" * 60)
    for L in L_values:
        mask = (results[L]['h'] < 1.0) & (results[L]['m_direct'] > 0.01)
        if mask.any():
            diff = results[L]['m_structure'][mask] - results[L]['m_direct'][mask]
            rel_diff = diff / results[L]['m_direct'][mask]
            print(f"L={L:2d}: Mean diff = {np.mean(diff):8.5f}, "
                  f"RMS = {np.sqrt(np.mean(diff**2)):8.5f}, "
                  f"Mean rel. diff = {np.mean(rel_diff)*100:6.2f}%")

    return results


def task3_finite_size_scaling():
    """
    Task 3: Finite size scaling to extract critical exponents.

    Compare exact diagonalization and DMRG methods.
    Extract critical exponent ν from scaling: m ~ L^(-β/ν)
    """
    print("\n" + "=" * 80)
    print("TASK 3: Finite Size Scaling Analysis")
    print("=" * 80)

    # System sizes to analyze
    L_values = [4, 6, 8, 10, 12]

    # Field values around critical point
    h_values = np.linspace(0.8, 1.2, 15)

    print(f"\nSystem sizes: {L_values}")
    print(f"Field range: {h_values[0]:.2f} to {h_values[-1]:.2f}")
    print(f"Number of field points: {len(h_values)}")

    # Exact diagonalization
    print("\n" + "-" * 60)
    print("Method 1: Exact Diagonalization")
    print("-" * 60)
    results_exact = finite_size_scaling_analysis(L_values, h_values, method='exact')

    # DMRG (for comparison) - optional, may fail for numerical reasons
    print("\n" + "-" * 60)
    print("Method 2: DMRG (bond dimension = 20)")
    print("-" * 60)
    try:
        results_dmrg = finite_size_scaling_analysis(L_values[:3], h_values, method='dmrg')
        dmrg_available = True
    except Exception as e:
        print(f"DMRG method encountered errors: {e}")
        print("Skipping DMRG comparison (using exact diagonalization only)")
        results_dmrg = None
        dmrg_available = False

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Plot 1: Magnetization vs h for different L (Exact)
    ax1 = fig.add_subplot(gs[0, 0])
    for L in L_values:
        ax1.plot(h_values, results_exact[L], 'o-', label=f'L={L}', markersize=4)
    ax1.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax1.set_xlabel('Transverse field h', fontsize=11)
    ax1.set_ylabel('Magnetization |m|', fontsize=11)
    ax1.set_title('Exact Diagonalization', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnetization vs h for different L (DMRG)
    ax2 = fig.add_subplot(gs[0, 1])
    if dmrg_available and results_dmrg is not None:
        for L in results_dmrg.keys():
            ax2.plot(h_values, results_dmrg[L], 's-', label=f'L={L}', markersize=4)
        ax2.set_title('DMRG', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'DMRG not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('DMRG (not available)', fontsize=12, fontweight='bold')
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.7, label='h_c=1')
    ax2.set_xlabel('Transverse field h', fontsize=11)
    ax2.set_ylabel('Magnetization |m|', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison at h = 0.9
    ax3 = fig.add_subplot(gs[0, 2])
    h_idx = np.argmin(np.abs(h_values - 0.9))
    m_exact = [results_exact[L][h_idx] for L in L_values]

    ax3.plot(L_values, m_exact, 'o-', label='Exact', markersize=8, linewidth=2)
    if dmrg_available and results_dmrg is not None:
        m_dmrg = [results_dmrg[L][h_idx] for L in results_dmrg.keys()]
        L_dmrg = list(results_dmrg.keys())
        ax3.plot(L_dmrg, m_dmrg, 's-', label='DMRG', markersize=8, linewidth=2)
    ax3.set_xlabel('System size L', fontsize=11)
    ax3.set_ylabel('Magnetization |m|', fontsize=11)
    ax3.set_title(f'Size Dependence at h={h_values[h_idx]:.2f}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Data collapse attempt
    ax4 = fig.add_subplot(gs[1, 0])

    # At critical point, m ~ L^(-β/ν)
    # For 1D TFIM: β/ν = 1/8 / 1 = 0.125 (exact)
    # Mean field: β/ν = 0.5 / 0.5 = 1
    beta_over_nu_exact = 0.125
    h_c_idx = np.argmin(np.abs(h_values - 1.0))

    m_at_hc = [results_exact[L][h_c_idx] for L in L_values]
    L_array = np.array(L_values)

    # Log-log plot
    ax4.loglog(L_array, m_at_hc, 'o-', label='Data at h_c', markersize=8, linewidth=2)

    # Fit
    if len(L_array) > 2 and all(m > 1e-10 for m in m_at_hc):
        coeffs = np.polyfit(np.log(L_array), np.log(m_at_hc), 1)
        exponent_fit = coeffs[0]

        L_fit = np.logspace(np.log10(L_array[0]), np.log10(L_array[-1]), 100)
        m_fit = np.exp(coeffs[1]) * L_fit**exponent_fit

        ax4.loglog(L_fit, m_fit, '--', label=f'Fit: m ~ L^{exponent_fit:.3f}', linewidth=2)
        ax4.loglog(L_fit, np.exp(coeffs[1]) * L_fit**(-beta_over_nu_exact), ':',
                  label=f'Exact: m ~ L^{-beta_over_nu_exact:.3f}', linewidth=2)

        print(f"\n" + "-" * 60)
        print("Critical Exponent Analysis at h_c")
        print("-" * 60)
        print(f"Fitted exponent: β/ν = {-exponent_fit:.4f}")
        print(f"Exact value: β/ν = {beta_over_nu_exact:.4f}")
        print(f"Mean field: β/ν = 1.0000")

    ax4.set_xlabel('System size L', fontsize=11)
    ax4.set_ylabel('Magnetization at h_c', fontsize=11)
    ax4.set_title('Critical Scaling', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # Plot 5: Magnetization difference (Exact vs DMRG)
    ax5 = fig.add_subplot(gs[1, 1])
    if dmrg_available and results_dmrg is not None:
        for L in results_dmrg.keys():
            diff = np.abs(results_exact[L] - results_dmrg[L])
            ax5.semilogy(h_values, diff, 'o-', label=f'L={L}', markersize=4)
        ax5.set_ylabel('|m_exact - m_DMRG|', fontsize=11)
        ax5.set_title('Method Comparison', fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'DMRG comparison\nnot available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=14)
        ax5.set_ylabel('Difference', fontsize=11)
        ax5.set_title('Method Comparison (not available)', fontsize=12, fontweight='bold')
    ax5.axvline(1.0, color='r', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Transverse field h', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3, which='both')

    # Plot 6: Finite size scaling collapse
    ax6 = fig.add_subplot(gs[1, 2])

    # Scaling form: m(L, δh) = L^(-β/ν) f(δh * L^(1/ν))
    # where δh = h - h_c
    nu = 1.0  # correlation length exponent for 1D TFIM
    h_c = 1.0

    for L in L_values:
        delta_h = h_values - h_c
        scaled_x = delta_h * L**(1/nu)
        scaled_y = results_exact[L] * L**(beta_over_nu_exact)

        # Only plot near critical point
        mask = np.abs(delta_h) < 0.3
        ax6.plot(scaled_x[mask], scaled_y[mask], 'o-', label=f'L={L}', markersize=4)

    ax6.set_xlabel(r'$(h - h_c) L^{1/\nu}$', fontsize=11)
    ax6.set_ylabel(r'$m L^{\beta/\nu}$', fontsize=11)
    ax6.set_title('Data Collapse', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('task3_finite_size_scaling.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: task3_finite_size_scaling.png")

    return results_exact, results_dmrg


def main():
    """Run all three tasks."""
    print("\n" + "=" * 80)
    print("TRANSVERSE FIELD ISING MODEL - TENSOR NETWORK ANALYSIS")
    print("=" * 80)
    print("\nThis script performs three main tasks:")
    print("1. Mean field critical exponents (m=1)")
    print("2. Order parameter: direct vs structure factor")
    print("3. Finite size scaling: exact vs DMRG")
    print("=" * 80)

    # Task 1
    results_task1, beta_fit = task1_mean_field_critical_exponents()

    # Task 2
    results_task2 = task2_order_parameter_comparison()

    # Task 3
    results_task3_exact, results_task3_dmrg = task3_finite_size_scaling()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated figures:")
    print("  1. task1_mean_field_exponents.png")
    print("  2. task2_order_parameter_comparison.png")
    print("  3. task3_finite_size_scaling.png")
    print("\nAll tasks completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
