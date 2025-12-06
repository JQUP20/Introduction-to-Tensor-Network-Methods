"""
测试量子谐振子模块
Test Quantum Harmonic Oscillator Module
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端 / Use non-interactive backend
import matplotlib.pyplot as plt

from quantum_harmonic_oscillator import (
    QuantumHarmonicOscillator,
    TimeDependentHarmonicOscillator
)


def test_ground_state_energy():
    """测试基态能量计算"""
    print("\n" + "="*70)
    print("测试1: 基态能量计算 / Test 1: Ground State Energy")
    print("="*70)

    qho = QuantumHarmonicOscillator(N=500, L=10.0)

    # 精确基态波函数
    psi0 = qho.exact_ground_state_wavefunction()

    # 计算能量期望值
    E0_simpson = qho.compute_energy_expectation(psi0, integration_method='simpson')
    E0_exact = qho.exact_energy(0)

    error = abs(E0_simpson - E0_exact)

    print(f"精确能量 / Exact energy:     E₀ = {E0_exact:.10f}")
    print(f"数值能量 / Numerical energy: E₀ = {E0_simpson:.10f}")
    print(f"误差 / Error:                    {error:.2e}")

    # 验证 (对于N=500,误差在1e-3级别是可接受的)
    assert error < 1e-3, f"Error too large: {error}"
    print("✓ 测试通过 / Test passed!")

    return True


def test_eigenvalue_solver():
    """测试特征值求解器"""
    print("\n" + "="*70)
    print("测试2: 特征值求解器 / Test 2: Eigenvalue Solver")
    print("="*70)

    qho = QuantumHarmonicOscillator(N=500, L=10.0)
    qho.build_hamiltonian_matrix()
    eigenvalues, eigenvectors = qho.diagonalize()

    # 检查前5个能级
    print(f"\n{'n':>3s} {'Exact':>12s} {'Numerical':>12s} {'Error':>12s}")
    print("-"*45)

    max_error = 0
    for n in range(5):
        exact = qho.exact_energy(n)
        numerical = eigenvalues[n]
        error = abs(numerical - exact)
        max_error = max(max_error, error)

        print(f"{n:3d} {exact:12.6f} {numerical:12.6f} {error:12.2e}")

    # 验证 (对于N=500,误差在5e-3级别是可接受的)
    assert max_error < 5e-3, f"Error too large: {max_error}"
    print("\n✓ 测试通过 / Test passed!")

    # 验证正交归一性
    print("\n验证正交归一性 / Verify orthonormality:")
    passed = qho.verify_orthonormality(k_max=3)
    assert passed, "Orthonormality check failed"

    return True


def test_time_evolution():
    """测试时间演化"""
    print("\n" + "="*70)
    print("测试3: 时间演化 / Test 3: Time Evolution")
    print("="*70)

    td_qho = TimeDependentHarmonicOscillator(N=300, L=10.0)
    psi0 = td_qho.exact_ground_state_wavefunction()

    # 测试短时间演化
    T = 1.0
    print(f"\n测试时间演化 T = {T}")
    psi_final, history = td_qho.time_evolution_split_step(psi0, T, Nt=100, save_interval=50)

    # 检查归一化
    norm = np.sum(np.abs(psi_final)**2) * td_qho.dx
    print(f"\n最终波函数归一化 / Final wavefunction normalization: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-3, f"Normalization error: {norm}"

    # 计算跃迁概率
    probs = td_qho.compute_transition_probabilities(psi_final, n_max=5)
    print(f"\n跃迁概率 / Transition probabilities:")
    for n, p in enumerate(probs):
        print(f"  P_{n} = {p:.6f}")

    # 验证概率之和
    total_prob = np.sum(probs)
    print(f"总概率 / Total probability: {total_prob:.6f}")
    assert total_prob >= 0 and total_prob <= 1.1, f"Invalid probability sum: {total_prob}"

    print("\n✓ 测试通过 / Test passed!")

    return True


def test_error_convergence():
    """测试误差收敛性"""
    print("\n" + "="*70)
    print("测试4: 误差收敛性 / Test 4: Error Convergence")
    print("="*70)

    qho = QuantumHarmonicOscillator(N=1000, L=10.0)
    N_values = [100, 200, 400]

    results = qho.error_analysis(N_values=N_values)

    # 检查误差是否随N增加而减小
    simpson_errors = results['errors']['simpson']

    print(f"\nSimpson方法误差 / Simpson method errors:")
    for N, error in zip(N_values, simpson_errors):
        print(f"  N = {N:4d}: error = {error:.2e}")

    # 验证误差递减
    for i in range(len(simpson_errors) - 1):
        assert simpson_errors[i] > simpson_errors[i+1], \
            f"Error not decreasing: {simpson_errors[i]} >= {simpson_errors[i+1]}"

    print("\n✓ 测试通过 / Test passed!")

    return True


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║          量子谐振子模块测试                                         ║
    ║          Quantum Harmonic Oscillator Module Test                  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True

    try:
        test_ground_state_energy()
    except Exception as e:
        print(f"✗ 测试1失败 / Test 1 failed: {e}")
        all_passed = False

    try:
        test_eigenvalue_solver()
    except Exception as e:
        print(f"✗ 测试2失败 / Test 2 failed: {e}")
        all_passed = False

    try:
        test_time_evolution()
    except Exception as e:
        print(f"✗ 测试3失败 / Test 3 failed: {e}")
        all_passed = False

    try:
        test_error_convergence()
    except Exception as e:
        print(f"✗ 测试4失败 / Test 4 failed: {e}")
        all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("所有测试通过！ / All tests passed!")
    else:
        print("部分测试失败 / Some tests failed")
    print("="*70)
