"""
Comprehensive Examples Demonstrating Advanced Features

This script demonstrates all the extended features:
1. 4th-order Trotter vs 2nd-order
2. Adaptive time stepping
3. Non-linear ramp protocols
4. XXZ model simulations
5. Finite temperature states
6. Finite-size scaling analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from ising_adiabatic_dmrg import (
    TransverseFieldIsing, GroundStateDMRG, AdiabaticPassage,
    DMRGParameters, Measurements, MPS
)

from tensor_network_extensions import (
    TEBDFourthOrder, AdaptiveTEBD,
    LinearRamp, TanhRamp, PolynomialRamp, OptimalRamp,
    XXZModel, FiniteTemperatureMPS, FiniteSizeScaling,
    compare_trotter_orders, compare_ramp_protocols
)

# Create results directory
os.makedirs('results', exist_ok=True)


# ============================================================================
# EXAMPLE 1: Trotter Order Comparison
# ============================================================================

def example_trotter_comparison():
    """
    Compare 2nd-order and 4th-order Trotter decompositions

    The 4th-order method should show significantly better accuracy,
    especially for larger time steps.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Trotter Order Comparison")
    print("="*70)

    L = 16
    h_initial = 2.0
    h_final = 0.5
    T = 5.0

    # Test different time steps
    dt_values = [0.05, 0.02, 0.01]

    results_2nd = []
    results_4th = []

    for dt in dt_values:
        print(f"\nTime step dt = {dt}")

        params = DMRGParameters(
            chi_max=30,
            chi_init=10,
            n_sweeps=5,
            convergence_tol=1e-6,
            dt=dt
        )

        model = TransverseFieldIsing(L)

        # Get exact ground state
        dmrg = GroundStateDMRG(model, params)
        _, E_exact = dmrg.run(h_final)

        # Compare
        delta_2, delta_4 = compare_trotter_orders(
            model, params, h_initial, h_final, T, E_exact
        )

        results_2nd.append(abs(delta_2))
        results_4th.append(abs(delta_4))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.loglog(dt_values, results_2nd, 'o-', label='2nd-order',
             markersize=10, linewidth=2)
    ax.loglog(dt_values, results_4th, 's-', label='4th-order',
             markersize=10, linewidth=2)

    # Add theoretical scaling lines
    dt_theory = np.array(dt_values)
    ax.loglog(dt_theory, 0.1 * dt_theory**3, 'k--',
             label=r'$\propto dt^3$ (2nd-order theory)', linewidth=2, alpha=0.5)
    ax.loglog(dt_theory, 0.01 * dt_theory**5, 'k:',
             label=r'$\propto dt^5$ (4th-order theory)', linewidth=2, alpha=0.5)

    ax.set_xlabel('Time step dt', fontsize=12)
    ax.set_ylabel('|Residual Energy| |δE|', fontsize=12)
    ax.set_title('Trotter Error Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/trotter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/trotter_comparison.png")

    return results_2nd, results_4th


# ============================================================================
# EXAMPLE 2: Adaptive Time Stepping
# ============================================================================

def example_adaptive_timestepping():
    """
    Demonstrate adaptive time stepping

    The time step automatically adjusts based on system state,
    using smaller dt near the critical point.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Adaptive Time Stepping")
    print("="*70)

    L = 16
    h_initial = 2.0
    h_final = 0.5
    T = 10.0

    params = DMRGParameters(
        chi_max=30,
        chi_init=10,
        n_sweeps=5,
        convergence_tol=1e-6,
        dt=0.05  # Initial dt (will be adapted)
    )

    model = TransverseFieldIsing(L)

    # Fixed dt evolution
    print("\n1. Fixed time step:")
    passage_fixed = AdiabaticPassage(model, params)
    result_fixed = passage_fixed.simulate(h_initial, h_final, T, save_every=5)

    # Adaptive dt evolution
    print("\n2. Adaptive time step:")

    # Find initial ground state
    dmrg = GroundStateDMRG(model, params)
    mps, _ = dmrg.run(h_initial)

    # Adaptive TEBD
    adaptive_tebd = AdaptiveTEBD(model, params,
                                 dt_min=0.005, dt_max=0.1,
                                 target_chi_growth=0.1)

    # Manual evolution loop to track dt
    times = []
    fields = []
    energies = []
    dt_history = []
    bond_dims = []

    t = 0.0
    old_chi = None
    old_energy = None

    print("Starting adaptive evolution...")
    pbar = tqdm(total=T)

    while t < T:
        # Adapt time step
        dt_adapted = adaptive_tebd.adaptive_dt(mps, h_initial + (h_final - h_initial) * t / T,
                                               old_chi, old_energy)

        # Don't overshoot
        if t + dt_adapted > T:
            dt_adapted = T - t

        # Evolve
        h_t = h_initial + (h_final - h_initial) * t / T
        adaptive_tebd.evolve_step(mps, h_t, dt_adapted)

        # Record
        if len(times) == 0 or t - times[-1] >= 0.1:  # Save every 0.1 time units
            times.append(t)
            fields.append(h_t)
            energies.append(Measurements.measure_energy(mps, model, h_t))
            dt_history.append(dt_adapted)
            bond_dims.append(max(mps.get_bond_dimensions()))

        old_chi = max(mps.get_bond_dimensions())
        old_energy = Measurements.measure_energy(mps, model, h_t)

        t += dt_adapted
        pbar.update(dt_adapted)

    pbar.close()

    result_adaptive = {
        'times': np.array(times),
        'fields': np.array(fields),
        'energies': np.array(energies),
        'dt_history': np.array(dt_history),
        'bond_dims': np.array(bond_dims),
        'final_energy': energies[-1]
    }

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time step history
    axes[0, 0].plot(result_adaptive['times'], result_adaptive['dt_history'],
                    'b-', linewidth=2)
    axes[0, 0].axvline(x=T/2, color='r', linestyle='--', alpha=0.5, label='Cross h_c')
    axes[0, 0].set_xlabel('Time t', fontsize=12)
    axes[0, 0].set_ylabel('Time step dt', fontsize=12)
    axes[0, 0].set_title('Adaptive Time Step', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Field vs time
    axes[0, 1].plot(result_fixed['times'], result_fixed['fields'],
                    'g-', linewidth=2, label='Fixed dt', alpha=0.7)
    axes[0, 1].plot(result_adaptive['times'], result_adaptive['fields'],
                    'b--', linewidth=2, label='Adaptive dt')
    axes[0, 1].axhline(y=1.0, color='r', linestyle=':', label='h_c')
    axes[0, 1].set_xlabel('Time t', fontsize=12)
    axes[0, 1].set_ylabel('Field h(t)', fontsize=12)
    axes[0, 1].set_title('Field Evolution', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Energy comparison
    axes[1, 0].plot(result_fixed['times'], result_fixed['energies'],
                    'g-', linewidth=2, label='Fixed dt', alpha=0.7)
    axes[1, 0].plot(result_adaptive['times'], result_adaptive['energies'],
                    'b--', linewidth=2, label='Adaptive dt')
    axes[1, 0].set_xlabel('Time t', fontsize=12)
    axes[1, 0].set_ylabel('Energy E(t)', fontsize=12)
    axes[1, 0].set_title('Energy Evolution', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Bond dimension comparison
    axes[1, 1].plot(result_fixed['times'], result_fixed['bond_dims'],
                    'g-', linewidth=2, label='Fixed dt', alpha=0.7)
    axes[1, 1].plot(result_adaptive['times'], result_adaptive['bond_dims'],
                    'b--', linewidth=2, label='Adaptive dt')
    axes[1, 1].set_xlabel('Time t', fontsize=12)
    axes[1, 1].set_ylabel('Max Bond Dimension χ', fontsize=12)
    axes[1, 1].set_title('Entanglement Growth', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/adaptive_timestepping.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/adaptive_timestepping.png")

    # Compare final energies
    print(f"\nFinal energies:")
    print(f"  Fixed dt:    {result_fixed['final_energy']:.6f}")
    print(f"  Adaptive dt: {result_adaptive['final_energy']:.6f}")
    print(f"  Difference:  {abs(result_fixed['final_energy'] - result_adaptive['final_energy']):.6e}")

    return result_fixed, result_adaptive


# ============================================================================
# EXAMPLE 3: Non-Linear Ramp Protocols
# ============================================================================

def example_nonlinear_ramps():
    """
    Compare different ramping protocols

    Non-linear ramps can reduce defect formation by spending more
    time near the critical point.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Non-Linear Ramp Protocols")
    print("="*70)

    L = 16
    h_initial = 2.0
    h_final = 0.5
    T = 5.0

    params = DMRGParameters(
        chi_max=30,
        chi_init=10,
        n_sweeps=5,
        convergence_tol=1e-6,
        dt=0.02
    )

    model = TransverseFieldIsing(L)

    # Get exact ground state
    dmrg = GroundStateDMRG(model, params)
    _, E_exact = dmrg.run(h_final)

    # Compare all protocols
    results = compare_ramp_protocols(model, params, h_initial, h_final, T, E_exact)

    # Plot the ramp functions themselves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t_arr = np.linspace(0, T, 500)
    protocols = {
        'Linear': LinearRamp(h_initial, h_final, T),
        'Tanh': TanhRamp(h_initial, h_final, T),
        'Polynomial (n=2)': PolynomialRamp(h_initial, h_final, T, exponent=2.0),
        'Optimal': OptimalRamp(h_initial, h_final, T)
    }

    for name, protocol in protocols.items():
        h_arr = [protocol.h_of_t(t) for t in t_arr]
        ax1.plot(t_arr, h_arr, linewidth=2, label=name)

    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Critical point h_c')
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Field h(t)', fontsize=12)
    ax1.set_title('Ramp Protocols', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Velocity (dh/dt)
    for name, protocol in protocols.items():
        h_arr = np.array([protocol.h_of_t(t) for t in t_arr])
        velocity = np.gradient(h_arr, t_arr)
        ax2.plot(h_arr, np.abs(velocity), linewidth=2, label=name)

    ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='h_c')
    ax2.set_xlabel('Field h', fontsize=12)
    ax2.set_ylabel('|Velocity| |dh/dt|', fontsize=12)
    ax2.set_title('Ramp Velocity', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('results/ramp_protocols.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/ramp_protocols.png")

    return results


# ============================================================================
# EXAMPLE 4: XXZ Model Ground State
# ============================================================================

def example_xxz_model():
    """
    Ground state of XXZ model for different anisotropies

    The XXZ model interpolates between Ising (Δ→∞) and XY (Δ=0) limits.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: XXZ Model Ground State")
    print("="*70)

    L = 20
    h = 0.5  # Magnetic field

    # Different anisotropies
    Delta_values = [0.5, 1.0, 1.5, 2.0, 3.0]  # Jz/Jx ratio

    params = DMRGParameters(
        chi_max=40,
        chi_init=10,
        n_sweeps=8,
        convergence_tol=1e-7,
        dt=0.02
    )

    energies = []
    correlations = []

    for Delta in Delta_values:
        print(f"\nΔ = Jz/Jx = {Delta:.2f}")

        # Create XXZ model with Jx = 1, Jz = Delta
        model = XXZModel(L, Jx=1.0, Jz=Delta)

        # DMRG ground state (need to adapt GroundStateDMRG for XXZ)
        # For now, just create a simple optimization loop
        mps = MPS(L, chi=params.chi_init, random_init=True)

        # Simple imaginary time evolution to find ground state
        from tensor_network_extensions import FiniteTemperatureMPS
        ft_mps = FiniteTemperatureMPS(model, params)

        # Use imaginary time to cool to ground state
        beta = 10.0  # Large β → ground state
        mps_gs = ft_mps.thermal_state(beta, h, initial_mps=mps)

        # Measure energy
        E = Measurements.measure_energy(mps_gs, model, h)
        energies.append(E)

        # Measure correlation length
        xi = Measurements.measure_correlation_length(mps_gs)
        correlations.append(xi)

        print(f"  Energy: {E:.6f}")
        print(f"  Correlation length: {xi:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(Delta_values, energies, 'o-', markersize=10, linewidth=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', alpha=0.5,
                label='Heisenberg point')
    ax1.set_xlabel('Anisotropy Δ = Jz/Jx', fontsize=12)
    ax1.set_ylabel('Ground State Energy', fontsize=12)
    ax1.set_title(f'XXZ Model (L={L}, h={h})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(Delta_values, correlations, 's-', markersize=10,
            linewidth=2, color='green')
    ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.5,
                label='Heisenberg point')
    ax2.set_xlabel('Anisotropy Δ = Jz/Jx', fontsize=12)
    ax2.set_ylabel('Correlation Length ξ', fontsize=12)
    ax2.set_title('Correlations vs Anisotropy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/xxz_model.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/xxz_model.png")

    return energies, correlations


# ============================================================================
# EXAMPLE 5: Finite Temperature
# ============================================================================

def example_finite_temperature():
    """
    Thermal states at different temperatures

    Using imaginary time evolution to prepare thermal states.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Finite Temperature States")
    print("="*70)

    L = 12  # Smaller system for thermal states
    h = 1.0  # At critical point

    # Different temperatures
    T_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_values = [1.0/T for T in T_values]

    params = DMRGParameters(
        chi_max=30,
        chi_init=10,
        n_sweeps=5,
        convergence_tol=1e-6,
        dt=0.01
    )

    model = TransverseFieldIsing(L)
    ft_mps = FiniteTemperatureMPS(model, params)

    energies = []
    entropies = []  # Estimated from bond dimension

    for i, (T, beta) in enumerate(zip(T_values, beta_values)):
        print(f"\nTemperature T = {T:.2f} (β = {beta:.2f})")

        # Prepare thermal state
        mps_thermal = ft_mps.thermal_state(beta, h)

        # Measure energy
        E = Measurements.measure_energy(mps_thermal, model, h)
        energies.append(E / L)  # Energy per site

        # Estimate entropy from bond dimension
        chi_avg = np.mean(mps_thermal.get_bond_dimensions())
        S_est = np.log(chi_avg)  # Rough estimate
        entropies.append(S_est)

        print(f"  Energy per site: {E/L:.6f}")
        print(f"  Average bond dim: {chi_avg:.2f}")
        print(f"  Entropy estimate: {S_est:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(T_values, energies, 'o-', markersize=10, linewidth=2)
    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel('Energy per site E/L', fontsize=12)
    ax1.set_title(f'Thermal Energy (L={L}, h={h})', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(T_values, entropies, 's-', markersize=10,
            linewidth=2, color='green')
    ax2.set_xlabel('Temperature T', fontsize=12)
    ax2.set_ylabel('Entropy estimate S', fontsize=12)
    ax2.set_title('Thermal Entropy', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/finite_temperature.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/finite_temperature.png")

    return energies, entropies


# ============================================================================
# EXAMPLE 6: Finite-Size Scaling
# ============================================================================

def example_finite_size_scaling():
    """
    Finite-size scaling analysis near quantum critical point

    Demonstrates data collapse for different system sizes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Finite-Size Scaling Analysis")
    print("="*70)

    L_values = [10, 12, 16, 20]
    h_values = np.linspace(0.7, 1.3, 15)  # Around h_c = 1.0

    params = DMRGParameters(
        chi_max=40,
        chi_init=10,
        n_sweeps=6,
        convergence_tol=1e-6,
        dt=0.02
    )

    # Collect ground state energies
    observables = {}

    for L in L_values:
        print(f"\nSystem size L = {L}")
        model = TransverseFieldIsing(L)
        dmrg = GroundStateDMRG(model, params)

        energies_per_site = []

        for h in tqdm(h_values, desc=f"L={L}"):
            _, E = dmrg.run(h)
            energies_per_site.append(E / L)

        observables[L] = np.array(energies_per_site)

    # Finite-size scaling analysis
    fss = FiniteSizeScaling(h_c=1.0, nu=1.0)
    results = fss.scaling_collapse(L_values, h_values, observables)

    # Plot
    fss.plot_data_collapse(results,
                          observable_name="Energy per site E/L",
                          save_path='results/finite_size_scaling.png')

    print("\nFinite-size scaling analysis complete!")

    return results


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

def main():
    """Run all advanced feature examples"""

    print("\n" + "="*70)
    print("ADVANCED TENSOR NETWORK FEATURES - COMPREHENSIVE EXAMPLES")
    print("="*70)

    examples_to_run = [
        ("Trotter Comparison", example_trotter_comparison),
        ("Adaptive Timestepping", example_adaptive_timestepping),
        ("Non-Linear Ramps", example_nonlinear_ramps),
        ("XXZ Model", example_xxz_model),
        ("Finite Temperature", example_finite_temperature),
        ("Finite-Size Scaling", example_finite_size_scaling)
    ]

    results = {}

    for name, func in examples_to_run:
        try:
            print(f"\n\nRunning: {name}")
            print("-" * 70)
            result = func()
            results[name] = result
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated files in results/:")
    print("  - trotter_comparison.png")
    print("  - adaptive_timestepping.png")
    print("  - ramp_protocols.png")
    print("  - ramp_protocol_comparison.png")
    print("  - xxz_model.png")
    print("  - finite_temperature.png")
    print("  - finite_size_scaling.png")

    return results


if __name__ == '__main__':
    main()
