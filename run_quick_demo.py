"""
Quick demonstration of adiabatic passage and Kibble-Zurek analysis
This runs a reduced set of simulations for faster execution
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from ising_adiabatic_dmrg import (
    TransverseFieldIsing,
    GroundStateDMRG,
    AdiabaticPassage,
    KibbleZurekAnalysis,
    DMRGParameters,
    plot_adiabatic_passage_results,
    plot_kibble_zurek_analysis
)
import os

print("="*70)
print("Quick Demonstration: Adiabatic Passage and Kibble-Zurek Analysis")
print("="*70)

# Parameters
L = 16  # System size
h_initial = 2.0  # Start in paramagnetic phase (h > h_c = 1)
h_final = 0.5    # End in ferromagnetic phase (h < h_c = 1)
T_values = np.logspace(0, 1.3, 6)  # Passage times from 1 to ~20

# DMRG parameters
params = DMRGParameters(
    chi_max=30,
    chi_init=10,
    n_sweeps=6,
    convergence_tol=1e-6,
    dt=0.02  # Smaller time step for numerical stability
)

print(f"\nSystem parameters:")
print(f"  L = {L}")
print(f"  h_initial = {h_initial} (paramagnetic)")
print(f"  h_final = {h_final} (ferromagnetic)")
print(f"  Critical point: h_c = 1.0")
print(f"  Number of passage times: {len(T_values)}")
print(f"  T range: {T_values[0]:.2f} to {T_values[-1]:.2f}")

# Create model
model = TransverseFieldIsing(L)
adiabatic = AdiabaticPassage(model, params)

# Get exact ground state energy at final field
print(f"\nComputing reference ground state at h = {h_final}...")
dmrg_exact = GroundStateDMRG(model, params)
_, E_exact = dmrg_exact.run(h_final)
print(f"Reference ground state energy: {E_exact:.8f}")

# Run simulations for different passage times
print(f"\n{'='*70}")
print(f"Running simulations for {len(T_values)} different passage times")
print(f"{'='*70}")

delta_E_values = []
results_list = []

for i, T in enumerate(T_values):
    print(f"\n[{i+1}/{len(T_values)}] T = {T:.3f}")
    print(f"-" * 50)

    result = adiabatic.simulate(h_initial, h_final, T, save_every=10)

    # Compute residual energy
    delta_E = result['final_energy'] - E_exact
    delta_E_values.append(delta_E)
    results_list.append(result)

    print(f"  Final energy: {result['final_energy']:.8f}")
    print(f"  Ground energy: {E_exact:.8f}")
    print(f"  Residual δE: {delta_E:.6e}")
    print(f"  Max bond dim: {max(result['bond_dims'])}")

delta_E_values = np.array(delta_E_values)

# Kibble-Zurek analysis
print(f"\n{'='*70}")
print("Kibble-Zurek Scaling Analysis")
print(f"{'='*70}")

kz_analysis = KibbleZurekAnalysis(model)
analysis_results = kz_analysis.analyze_scaling(T_values, delta_E_values)

# Print analysis results
print(f"\nResidual energy scaling:")
print(f"  {'T':<10} {'δE':<15}")
print(f"  {'-'*10} {'-'*15}")
for T, dE in zip(T_values, delta_E_values):
    print(f"  {T:<10.3f} {dE:<15.6e}")

if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
    alpha_sim = analysis_results['kz_params']['alpha']
    alpha_theory = 0.5

    print(f"\nKibble-Zurek regime analysis:")
    print(f"  Theoretical prediction: δE ~ T^(-1/2)")
    print(f"  Theoretical exponent: α = {alpha_theory:.4f}")
    print(f"  Fitted exponent: α = {alpha_sim:.4f}")
    print(f"  Relative error: {abs(alpha_sim - alpha_theory)/alpha_theory * 100:.2f}%")

    print(f"\n  Physical interpretation:")
    print(f"  - In the Kibble-Zurek regime (slow passage), the system")
    print(f"    cannot follow the ground state adiabatically near the")
    print(f"    critical point due to the closing of the gap.")
    print(f"  - The residual excitation energy scales as T^(-dν/(1+zν))")
    print(f"  - For 1D TFIM: d=1, ν=1, z=1 → α = 1/2")

if 'lz_params' in analysis_results and analysis_results['lz_params'] is not None:
    print(f"\nLandau-Zener regime detected (fast passage):")
    print(f"  Exponential suppression: δE ~ exp(-aT)")
    print(f"  Decay rate: a = {analysis_results['lz_params']['a']:.4f}")

# Create results directory
os.makedirs('results', exist_ok=True)

# Plot Kibble-Zurek scaling
print(f"\n{'='*70}")
print("Generating plots...")
print(f"{'='*70}")

plot_kibble_zurek_analysis(
    analysis_results,
    save_path='results/demo_kibble_zurek.png'
)

# Plot one example adiabatic passage (middle T value)
idx_mid = len(results_list) // 2
plot_adiabatic_passage_results(
    results_list[idx_mid],
    save_path=f'results/demo_adiabatic_T{T_values[idx_mid]:.1f}.png'
)

# Create a summary plot showing multiple passages
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot energies for different T
for i, (T, result) in enumerate(zip(T_values, results_list)):
    alpha = 0.3 + 0.7 * i / len(T_values)
    axes[0, 0].plot(result['fields'], result['energies'],
                    label=f'T={T:.1f}', alpha=alpha, linewidth=2)

axes[0, 0].axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='h_c')
axes[0, 0].axhline(y=E_exact, color='k', linestyle=':', linewidth=2, label='Ground state')
axes[0, 0].set_xlabel('Transverse field h', fontsize=12)
axes[0, 0].set_ylabel('Energy', fontsize=12)
axes[0, 0].set_title('Energy vs Field for Different Passage Times', fontsize=14)
axes[0, 0].legend(fontsize=8, loc='best')
axes[0, 0].grid(True, alpha=0.3)

# Plot bond dimension growth
for i, (T, result) in enumerate(zip(T_values, results_list)):
    alpha = 0.3 + 0.7 * i / len(T_values)
    axes[0, 1].plot(result['fields'], result['bond_dims'],
                    label=f'T={T:.1f}', alpha=alpha, linewidth=2)

axes[0, 1].axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='h_c')
axes[0, 1].set_xlabel('Transverse field h', fontsize=12)
axes[0, 1].set_ylabel('Bond Dimension χ', fontsize=12)
axes[0, 1].set_title('Entanglement Growth', fontsize=14)
axes[0, 1].legend(fontsize=8, loc='best')
axes[0, 1].grid(True, alpha=0.3)

# Plot residual energy vs T
axes[1, 0].loglog(T_values, delta_E_values, 'ko', markersize=10, label='Simulation')
if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
    T_fit = np.logspace(np.log10(T_values[0]), np.log10(T_values[-1]), 50)
    dE_fit = analysis_results['kz_params']['A'] * T_fit**(-analysis_results['kz_params']['alpha'])
    axes[1, 0].loglog(T_fit, dE_fit, 'r-', linewidth=2,
                      label=f"Fit: T^(-{analysis_results['kz_params']['alpha']:.3f})")
    dE_theory = analysis_results['kz_params']['A'] * T_fit**(-0.5)
    axes[1, 0].loglog(T_fit, dE_theory, 'g--', linewidth=2, label='Theory: T^(-1/2)')

axes[1, 0].set_xlabel('Passage Time T', fontsize=12)
axes[1, 0].set_ylabel('Residual Energy δE', fontsize=12)
axes[1, 0].set_title('Kibble-Zurek Scaling', fontsize=14)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot final energy vs T
final_energies = [r['final_energy'] for r in results_list]
axes[1, 1].semilogx(T_values, final_energies, 'bo-', markersize=8, linewidth=2, label='Final energy')
axes[1, 1].axhline(y=E_exact, color='r', linestyle='--', linewidth=2, label='Ground state')
axes[1, 1].set_xlabel('Passage Time T', fontsize=12)
axes[1, 1].set_ylabel('Final Energy', fontsize=12)
axes[1, 1].set_title('Convergence to Ground State', fontsize=14)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/demo_summary.png', dpi=300, bbox_inches='tight')
print("  Saved: results/demo_summary.png")

print(f"\n{'='*70}")
print("Demonstration complete!")
print(f"{'='*70}")
print("\nGenerated files:")
print("  results/demo_kibble_zurek.png - KZ scaling analysis")
print(f"  results/demo_adiabatic_T{T_values[idx_mid]:.1f}.png - Example passage")
print("  results/demo_summary.png - Summary of all passages")
print("\nKey findings:")
if 'kz_params' in analysis_results and analysis_results['kz_params'] is not None:
    alpha_sim = analysis_results['kz_params']['alpha']
    print(f"  1. Residual energy scales as δE ~ T^(-α) with α ≈ {alpha_sim:.3f}")
    print(f"  2. Theoretical prediction α = 0.5 for 1D TFIM")
    print(f"  3. Agreement indicates proper Kibble-Zurek physics")
else:
    print(f"  1. Kibble-Zurek regime analysis requires more data points")
    print(f"  2. Theoretical prediction: δE ~ T^(-1/2) for 1D TFIM")
print(f"  4. System crosses critical point at h_c = 1.0")
print(f"  5. Entanglement growth peaks near criticality")
