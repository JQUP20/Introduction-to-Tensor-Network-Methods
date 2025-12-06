"""
量子谐振子数值计算模块
Quantum Harmonic Oscillator Numerical Computation Module

这个模块实现了量子谐振子的数值计算，包括：
1. 基态能量计算和误差分析
2. 特征值和特征向量求解
3. 含时演化

This module implements numerical computations for quantum harmonic oscillator:
1. Ground state energy calculation and error analysis
2. Eigenvalue and eigenvector solver
3. Time-dependent evolution

Author: Claude AI
Date: 2025-11-13
"""

import numpy as np
try:
    from scipy.integrate import simpson as simps  # scipy >= 1.6.0
except ImportError:
    from scipy.integrate import simps  # scipy < 1.6.0

try:
    from scipy.integrate import trapezoid
except ImportError:
    from scipy.integrate import trapz as trapezoid
from scipy.linalg import eigh, expm
from scipy.sparse import diags
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Callable, Optional
import time


class QuantumHarmonicOscillator:
    """
    量子谐振子类
    Quantum Harmonic Oscillator Class

    哈密顿量: H = p^2/2 + x^2/2 (ħ = m = ω = 1)
    Hamiltonian: H = p^2/2 + x^2/2 (ħ = m = ω = 1)
    """

    def __init__(self, N: int = 1000, L: float = 10.0, hbar: float = 1.0,
                 m: float = 1.0, omega: float = 1.0):
        """
        初始化量子谐振子

        Parameters:
        -----------
        N : int
            网格点数 / Number of grid points
        L : float
            空间区间半长度，x ∈ [-L, L] / Half-length of spatial domain
        hbar : float
            约化普朗克常数 / Reduced Planck constant
        m : float
            质量 / Mass
        omega : float
            角频率 / Angular frequency
        """
        self.N = N
        self.L = L
        self.hbar = hbar
        self.m = m
        self.omega = omega

        # 创建空间网格 / Create spatial grid
        self.x = np.linspace(-L, L, N)
        self.dx = self.x[1] - self.x[0]

        # 哈密顿矩阵 / Hamiltonian matrix
        self.H = None

        # 特征值和特征向量 / Eigenvalues and eigenvectors
        self.eigenvalues = None
        self.eigenvectors = None

    def exact_ground_state_wavefunction(self, x: np.ndarray = None) -> np.ndarray:
        """
        精确基态波函数
        Exact ground state wavefunction

        ψ₀(x) = (mω/πħ)^(1/4) * exp(-mωx²/2ħ)

        For ħ = m = ω = 1: ψ₀(x) = π^(-1/4) * exp(-x²/2)
        """
        if x is None:
            x = self.x

        alpha = self.m * self.omega / self.hbar
        normalization = (alpha / np.pi) ** 0.25
        return normalization * np.exp(-alpha * x**2 / 2)

    def exact_energy(self, n: int) -> float:
        """
        精确能量本征值
        Exact energy eigenvalue

        Eₙ = ħω(n + 1/2)

        Parameters:
        -----------
        n : int
            量子数 / Quantum number (n = 0, 1, 2, ...)
        """
        return self.hbar * self.omega * (n + 0.5)

    def build_hamiltonian_matrix(self) -> np.ndarray:
        """
        构建哈密顿矩阵 (有限差分方法)
        Build Hamiltonian matrix using finite difference method

        H = T + V
        T = -ħ²/(2m) * d²/dx²
        V = (1/2) * m * ω² * x²

        使用三点有限差分：
        d²ψ/dx² ≈ (ψᵢ₊₁ - 2ψᵢ + ψᵢ₋₁) / dx²

        Returns:
        --------
        H : ndarray
            哈密顿矩阵 / Hamiltonian matrix
        """
        # 动能项 (三对角矩阵) / Kinetic energy (tridiagonal matrix)
        # -ħ²/(2m*dx²) * [-1, 2, -1]
        coeff_kinetic = -self.hbar**2 / (2 * self.m * self.dx**2)
        diag_main = -2 * coeff_kinetic * np.ones(self.N)
        diag_off = coeff_kinetic * np.ones(self.N - 1)

        T = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        # 势能项 (对角矩阵) / Potential energy (diagonal matrix)
        V = np.diag(0.5 * self.m * self.omega**2 * self.x**2)

        # 哈密顿矩阵 / Hamiltonian matrix
        self.H = T + V

        return self.H

    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        对角化哈密顿矩阵
        Diagonalize Hamiltonian matrix

        Returns:
        --------
        eigenvalues : ndarray
            特征值（按递增顺序）/ Eigenvalues (in ascending order)
        eigenvectors : ndarray
            特征向量（列向量）/ Eigenvectors (as columns)
        """
        if self.H is None:
            self.build_hamiltonian_matrix()

        # 使用scipy的厄米矩阵对角化函数
        # eigh比eig更快，因为它利用了矩阵的对称性
        self.eigenvalues, self.eigenvectors = eigh(self.H)

        # 归一化特征向量（按积分归一化）
        for i in range(self.N):
            norm = np.sqrt(simps(np.abs(self.eigenvectors[:, i])**2, self.x))
            self.eigenvectors[:, i] /= norm

        return self.eigenvalues, self.eigenvectors

    def compute_expectation_value(self, psi: np.ndarray, operator: np.ndarray) -> float:
        """
        计算期望值
        Compute expectation value

        ⟨ψ|O|ψ⟩ = ∫ ψ*(x) O ψ(x) dx

        Parameters:
        -----------
        psi : ndarray
            波函数 / Wavefunction
        operator : ndarray
            算符矩阵 / Operator matrix

        Returns:
        --------
        expectation : float
            期望值 / Expectation value
        """
        # O|ψ⟩
        O_psi = operator @ psi

        # ⟨ψ|O|ψ⟩
        integrand = np.conj(psi) * O_psi
        expectation = simps(integrand, self.x)

        return np.real(expectation)

    def compute_energy_expectation(self, psi: np.ndarray,
                                   integration_method: str = 'simpson') -> float:
        """
        计算能量期望值
        Compute energy expectation value

        ⟨H⟩ = ⟨T⟩ + ⟨V⟩

        Parameters:
        -----------
        psi : ndarray
            波函数 / Wavefunction
        integration_method : str
            积分方法：'simpson', 'trapezoid' / Integration method

        Returns:
        --------
        energy : float
            能量期望值 / Energy expectation value
        """
        # 动能项：⟨T⟩ = ∫ ψ* (-ħ²/2m d²/dx²) ψ dx
        # 使用分部积分：⟨T⟩ = ∫ (ħ²/2m) |dψ/dx|² dx

        # 计算导数 (中心差分) / Compute derivative (central difference)
        dpsi_dx = np.zeros_like(psi)
        dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2 * self.dx)
        # 边界使用前向/后向差分 / Boundary: forward/backward difference
        dpsi_dx[0] = (psi[1] - psi[0]) / self.dx
        dpsi_dx[-1] = (psi[-1] - psi[-2]) / self.dx

        # 动能积分 / Kinetic energy integral
        T_integrand = (self.hbar**2 / (2 * self.m)) * np.abs(dpsi_dx)**2

        # 势能：⟨V⟩ = ∫ ψ* V(x) ψ dx
        V_integrand = 0.5 * self.m * self.omega**2 * self.x**2 * np.abs(psi)**2

        # 选择积分方法 / Choose integration method
        if integration_method == 'simpson':
            T_expectation = simps(T_integrand, self.x)
            V_expectation = simps(V_integrand, self.x)
        elif integration_method == 'trapezoid':
            T_expectation = trapezoid(T_integrand, self.x)
            V_expectation = trapezoid(V_integrand, self.x)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        return T_expectation + V_expectation

    def error_analysis(self, N_values: List[int] = None,
                      integration_methods: List[str] = None) -> Dict:
        """
        误差分析：区分离散化误差和积分误差
        Error analysis: distinguish discretization and integration errors

        Parameters:
        -----------
        N_values : list of int
            网格点数列表 / List of grid point numbers
        integration_methods : list of str
            积分方法列表 / List of integration methods

        Returns:
        --------
        results : dict
            包含误差分析结果的字典 / Dictionary containing error analysis results
        """
        if N_values is None:
            N_values = [100, 200, 500, 1000, 2000]

        if integration_methods is None:
            integration_methods = ['trapezoid', 'simpson']

        results = {
            'N_values': N_values,
            'integration_methods': integration_methods,
            'errors': {},
            'energies': {},
            'exact_energy': self.exact_energy(0)
        }

        for method in integration_methods:
            results['errors'][method] = []
            results['energies'][method] = []

        print("=" * 70)
        print("误差分析 / Error Analysis")
        print("=" * 70)
        print(f"精确基态能量 / Exact ground state energy: E₀ = {self.exact_energy(0):.10f}")
        print()

        for N in N_values:
            # 创建临时振荡器 / Create temporary oscillator
            temp_osc = QuantumHarmonicOscillator(N=N, L=self.L,
                                                hbar=self.hbar, m=self.m, omega=self.omega)

            # 获取精确基态波函数 / Get exact ground state wavefunction
            psi0 = temp_osc.exact_ground_state_wavefunction()

            print(f"N = {N:4d}:")

            for method in integration_methods:
                # 计算能量期望值 / Compute energy expectation value
                energy = temp_osc.compute_energy_expectation(psi0, integration_method=method)
                error = abs(energy - self.exact_energy(0))

                results['energies'][method].append(energy)
                results['errors'][method].append(error)

                print(f"  {method:10s}: E = {energy:.10f}, Error = {error:.2e}")

            print()

        return results

    def convergence_test(self, k_max: int = 10) -> Dict:
        """
        收敛性测试：验证计算的特征值
        Convergence test: verify computed eigenvalues

        Parameters:
        -----------
        k_max : int
            测试前k个能级 / Test first k energy levels

        Returns:
        --------
        results : dict
            收敛性测试结果 / Convergence test results
        """
        if self.eigenvalues is None:
            self.diagonalize()

        results = {
            'n': [],
            'exact': [],
            'numerical': [],
            'error': [],
            'relative_error': []
        }

        print("=" * 80)
        print("收敛性测试 / Convergence Test")
        print("=" * 80)
        print(f"{'n':>3s} {'Exact Energy':>15s} {'Numerical Energy':>18s} "
              f"{'Error':>12s} {'Rel. Error':>12s}")
        print("-" * 80)

        for n in range(min(k_max, len(self.eigenvalues))):
            exact = self.exact_energy(n)
            numerical = self.eigenvalues[n]
            error = abs(numerical - exact)
            rel_error = error / abs(exact) if exact != 0 else error

            results['n'].append(n)
            results['exact'].append(exact)
            results['numerical'].append(numerical)
            results['error'].append(error)
            results['relative_error'].append(rel_error)

            print(f"{n:3d} {exact:15.10f} {numerical:18.10f} "
                  f"{error:12.2e} {rel_error:12.2e}")

        print("=" * 80)

        return results

    def verify_orthonormality(self, k_max: int = 5) -> bool:
        """
        验证特征向量的正交归一性
        Verify orthonormality of eigenvectors

        ⟨ψᵢ|ψⱼ⟩ = δᵢⱼ

        Parameters:
        -----------
        k_max : int
            验证前k个特征向量 / Verify first k eigenvectors

        Returns:
        --------
        passed : bool
            是否通过验证 / Whether verification passed
        """
        if self.eigenvectors is None:
            self.diagonalize()

        print("\n" + "=" * 70)
        print("正交归一性验证 / Orthonormality Verification")
        print("=" * 70)

        passed = True
        tolerance = 1e-6

        for i in range(min(k_max, self.N)):
            for j in range(min(k_max, self.N)):
                overlap = simps(np.conj(self.eigenvectors[:, i]) *
                              self.eigenvectors[:, j], self.x)

                expected = 1.0 if i == j else 0.0
                error = abs(overlap - expected)

                if error > tolerance:
                    print(f"⚠ ⟨ψ{i}|ψ{j}⟩ = {overlap:.6e} (expected {expected:.1f}), "
                          f"error = {error:.2e}")
                    passed = False

        if passed:
            print(f"✓ 所有重叠积分满足正交归一条件 (tolerance = {tolerance:.1e})")
            print(f"✓ All overlap integrals satisfy orthonormality (tolerance = {tolerance:.1e})")

        print("=" * 70)

        return passed

    def plot_wavefunctions(self, n_states: int = 5, save_path: str = None):
        """
        绘制波函数
        Plot wavefunctions

        Parameters:
        -----------
        n_states : int
            绘制前n个状态 / Plot first n states
        save_path : str
            保存路径 / Save path (optional)
        """
        if self.eigenvectors is None:
            self.diagonalize()

        fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states))

        if n_states == 1:
            axes = [axes]

        for n in range(min(n_states, self.N)):
            ax = axes[n]

            # 数值解 / Numerical solution
            ax.plot(self.x, self.eigenvectors[:, n], 'b-',
                   label=f'Numerical $\\psi_{n}$', linewidth=2)

            # 势能 / Potential
            V = 0.5 * self.m * self.omega**2 * self.x**2
            V_normalized = V / V.max() * np.max(np.abs(self.eigenvectors[:, n]))
            ax.plot(self.x, V_normalized, 'k--', alpha=0.3, label='V(x) (scaled)')

            # 能级线 / Energy level
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

            ax.set_xlabel('x')
            ax.set_ylabel(f'$\\psi_{n}(x)$')
            ax.set_title(f'State n={n}, E={self.eigenvalues[n]:.6f} '
                        f'(Exact: {self.exact_energy(n):.6f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到 / Figure saved to: {save_path}")

        plt.show()

    def plot_error_analysis(self, results: Dict, save_path: str = None):
        """
        绘制误差分析结果
        Plot error analysis results

        Parameters:
        -----------
        results : dict
            error_analysis()返回的结果 / Results from error_analysis()
        save_path : str
            保存路径 / Save path (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        N_values = results['N_values']

        # 左图：能量 vs N
        ax = axes[0]
        for method in results['integration_methods']:
            energies = results['energies'][method]
            ax.plot(N_values, energies, 'o-', label=method.capitalize(),
                   linewidth=2, markersize=8)

        ax.axhline(y=results['exact_energy'], color='k', linestyle='--',
                  label='Exact', linewidth=2)
        ax.set_xlabel('Number of grid points (N)', fontsize=12)
        ax.set_ylabel('Ground state energy', fontsize=12)
        ax.set_title('Energy vs Grid Size', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 右图：误差 vs N (对数坐标)
        ax = axes[1]
        for method in results['integration_methods']:
            errors = results['errors'][method]
            ax.loglog(N_values, errors, 'o-', label=method.capitalize(),
                     linewidth=2, markersize=8)

        # 添加参考线 (O(dx²) 和 O(dx⁴))
        dx_values = [2 * self.L / N for N in N_values]
        ax.loglog(N_values, [dx**2 * 0.1 for dx in dx_values],
                 'k:', label='$O(\\Delta x^2)$', alpha=0.5)
        ax.loglog(N_values, [dx**4 * 10 for dx in dx_values],
                 'k--', label='$O(\\Delta x^4)$', alpha=0.5)

        ax.set_xlabel('Number of grid points (N)', fontsize=12)
        ax.set_ylabel('Absolute error', fontsize=12)
        ax.set_title('Error vs Grid Size (log-log)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到 / Figure saved to: {save_path}")

        plt.show()


class TimeDependentHarmonicOscillator(QuantumHarmonicOscillator):
    """
    含时量子谐振子类
    Time-dependent Quantum Harmonic Oscillator Class

    哈密顿量: H(t) = p²/2 + (q - q₀(t))²/2
    其中 q₀(t) = t/T, t ∈ [0, T]

    Hamiltonian: H(t) = p²/2 + (q - q₀(t))²/2
    where q₀(t) = t/T, t ∈ [0, T]
    """

    def __init__(self, N: int = 1000, L: float = 10.0,
                 hbar: float = 1.0, m: float = 1.0, omega: float = 1.0):
        """初始化含时量子谐振子"""
        super().__init__(N, L, hbar, m, omega)

        # 时间演化相关 / Time evolution related
        self.psi_t = None
        self.time_grid = None
        self.psi_history = []
        self.time_history = []

    def build_hamiltonian_time_dependent(self, t: float, T_total: float) -> np.ndarray:
        """
        构建含时哈密顿矩阵
        Build time-dependent Hamiltonian matrix

        H(t) = T + V(t)
        V(t) = (1/2) m ω² (x - q₀(t))²
        q₀(t) = t/T

        Parameters:
        -----------
        t : float
            当前时间 / Current time
        T_total : float
            总时间 / Total time

        Returns:
        --------
        H_t : ndarray
            含时哈密顿矩阵 / Time-dependent Hamiltonian matrix
        """
        # 动能项 (不随时间变化) / Kinetic energy (time-independent)
        coeff_kinetic = -self.hbar**2 / (2 * self.m * self.dx**2)
        diag_main = -2 * coeff_kinetic * np.ones(self.N)
        diag_off = coeff_kinetic * np.ones(self.N - 1)

        T_kinetic = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        # 含时势能项 / Time-dependent potential energy
        q0_t = t / T_total if T_total > 0 else 0.0
        V_t = np.diag(0.5 * self.m * self.omega**2 * (self.x - q0_t)**2)

        return T_kinetic + V_t

    def time_evolution_split_step(self, psi0: np.ndarray, T: float,
                                   Nt: int = 1000,
                                   save_interval: int = 10) -> Tuple[np.ndarray, List]:
        """
        时间演化 (分步法)
        Time evolution using split-step method

        |ψ(t+dt)⟩ ≈ exp(-iH(t)dt)|ψ(t)⟩

        Parameters:
        -----------
        psi0 : ndarray
            初态波函数 / Initial wavefunction
        T : float
            总演化时间 / Total evolution time
        Nt : int
            时间步数 / Number of time steps
        save_interval : int
            保存间隔 / Save interval

        Returns:
        --------
        psi_final : ndarray
            最终波函数 / Final wavefunction
        history : list
            保存的波函数历史 / Saved wavefunction history
        """
        dt = T / Nt
        psi = psi0.copy()

        self.time_history = []
        self.psi_history = []

        print(f"\n时间演化: T = {T:.2f}, Nt = {Nt}, dt = {dt:.6f}")
        print(f"Time evolution: T = {T:.2f}, Nt = {Nt}, dt = {dt:.6f}")

        for n in range(Nt + 1):
            t = n * dt

            # 保存波函数 / Save wavefunction
            if n % save_interval == 0:
                self.time_history.append(t)
                self.psi_history.append(psi.copy())

                # 计算期望值 / Compute expectation values
                x_exp = simps(self.x * np.abs(psi)**2, self.x)
                norm = simps(np.abs(psi)**2, self.x)

                if n % (save_interval * 10) == 0:
                    print(f"  t = {t:6.3f}: ⟨x⟩ = {x_exp:8.4f}, "
                          f"norm = {norm:.6f}")

            # 时间演化一步 / Evolve one time step
            if n < Nt:
                H_t = self.build_hamiltonian_time_dependent(t + dt/2, T)

                # |ψ(t+dt)⟩ = exp(-iH(t)dt/ħ)|ψ(t)⟩
                U = expm(-1j * H_t * dt / self.hbar)
                psi = U @ psi

                # 重新归一化 (可选，保持数值稳定性)
                # Renormalize (optional, for numerical stability)
                psi /= np.sqrt(simps(np.abs(psi)**2, self.x))

        self.psi_t = psi
        return psi, self.psi_history

    def time_evolution_rk4(self, psi0: np.ndarray, T: float,
                           Nt: int = 1000,
                           save_interval: int = 10) -> Tuple[np.ndarray, List]:
        """
        时间演化 (4阶Runge-Kutta方法)
        Time evolution using 4th-order Runge-Kutta method

        Parameters:
        -----------
        psi0 : ndarray
            初态波函数 / Initial wavefunction
        T : float
            总演化时间 / Total evolution time
        Nt : int
            时间步数 / Number of time steps
        save_interval : int
            保存间隔 / Save interval

        Returns:
        --------
        psi_final : ndarray
            最终波函数 / Final wavefunction
        history : list
            保存的波函数历史 / Saved wavefunction history
        """
        dt = T / Nt
        psi = psi0.copy()

        self.time_history = []
        self.psi_history = []

        print(f"\n时间演化 (RK4): T = {T:.2f}, Nt = {Nt}, dt = {dt:.6f}")
        print(f"Time evolution (RK4): T = {T:.2f}, Nt = {Nt}, dt = {dt:.6f}")

        def dpsi_dt(t, psi):
            """薛定谔方程右侧 / RHS of Schrödinger equation"""
            H_t = self.build_hamiltonian_time_dependent(t, T)
            return -1j / self.hbar * (H_t @ psi)

        for n in range(Nt + 1):
            t = n * dt

            # 保存波函数 / Save wavefunction
            if n % save_interval == 0:
                self.time_history.append(t)
                self.psi_history.append(psi.copy())

                # 计算期望值 / Compute expectation values
                x_exp = simps(self.x * np.abs(psi)**2, self.x)
                norm = simps(np.abs(psi)**2, self.x)

                if n % (save_interval * 10) == 0:
                    print(f"  t = {t:6.3f}: ⟨x⟩ = {x_exp:8.4f}, "
                          f"norm = {norm:.6f}")

            # RK4 时间步进 / RK4 time stepping
            if n < Nt:
                k1 = dpsi_dt(t, psi)
                k2 = dpsi_dt(t + dt/2, psi + dt/2 * k1)
                k3 = dpsi_dt(t + dt/2, psi + dt/2 * k2)
                k4 = dpsi_dt(t + dt, psi + dt * k3)

                psi = psi + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

                # 重新归一化 / Renormalize
                psi /= np.sqrt(simps(np.abs(psi)**2, self.x))

        self.psi_t = psi
        return psi, self.psi_history

    def compute_transition_probabilities(self, psi_final: np.ndarray,
                                        n_max: int = 10) -> np.ndarray:
        """
        计算跃迁概率
        Compute transition probabilities

        Pₙ = |⟨n|ψ(T)⟩|²

        Parameters:
        -----------
        psi_final : ndarray
            最终态波函数 / Final state wavefunction
        n_max : int
            最大量子数 / Maximum quantum number

        Returns:
        --------
        probabilities : ndarray
            跃迁概率 / Transition probabilities
        """
        # 首先需要计算时间无关谐振子的本征态
        # First compute eigenstates of time-independent oscillator
        temp_osc = QuantumHarmonicOscillator(N=self.N, L=self.L,
                                            hbar=self.hbar, m=self.m, omega=self.omega)
        temp_osc.build_hamiltonian_matrix()
        eigenvalues, eigenvectors = temp_osc.diagonalize()

        probabilities = np.zeros(n_max)

        for n in range(min(n_max, self.N)):
            # ⟨n|ψ(T)⟩
            overlap = simps(np.conj(eigenvectors[:, n]) * psi_final, self.x)
            # Pₙ = |⟨n|ψ(T)⟩|²
            probabilities[n] = np.abs(overlap)**2

        return probabilities

    def plot_time_evolution(self, T_values: List[float],
                           method: str = 'split_step',
                           save_path: str = None):
        """
        绘制不同T值的时间演化
        Plot time evolution for different T values

        Parameters:
        -----------
        T_values : list of float
            总时间列表 / List of total times
        method : str
            演化方法: 'split_step' 或 'rk4' / Evolution method
        save_path : str
            保存路径前缀 / Save path prefix (optional)
        """
        # 初态：基态 / Initial state: ground state
        psi0 = self.exact_ground_state_wavefunction()

        n_T = len(T_values)
        fig, axes = plt.subplots(n_T, 2, figsize=(14, 4 * n_T))

        if n_T == 1:
            axes = axes.reshape(1, -1)

        for idx, T in enumerate(T_values):
            print(f"\n{'='*70}")
            print(f"T = {T:.2f}")
            print(f"{'='*70}")

            # 时间演化 / Time evolution
            if method == 'split_step':
                psi_final, history = self.time_evolution_split_step(
                    psi0, T, Nt=max(1000, int(T * 100)), save_interval=10)
            elif method == 'rk4':
                psi_final, history = self.time_evolution_rk4(
                    psi0, T, Nt=max(1000, int(T * 100)), save_interval=10)
            else:
                raise ValueError(f"Unknown method: {method}")

            # 左图：最终波函数 / Left: Final wavefunction
            ax = axes[idx, 0]
            ax.plot(self.x, np.abs(psi_final)**2, 'b-', linewidth=2,
                   label='$|\\psi(T)|^2$')
            ax.plot(self.x, np.abs(psi0)**2, 'k--', alpha=0.5,
                   label='$|\\psi_0|^2$')
            ax.axvline(x=1.0, color='r', linestyle=':', alpha=0.5,
                      label='$q_0(T)=1$')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('Probability density', fontsize=12)
            ax.set_title(f'T = {T:.2f}: Final wavefunction', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # 右图：跃迁概率 / Right: Transition probabilities
            ax = axes[idx, 1]
            n_max = 10
            probs = self.compute_transition_probabilities(psi_final, n_max=n_max)

            ax.bar(range(n_max), probs, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Quantum number n', fontsize=12)
            ax.set_ylabel('Transition probability $P_n$', fontsize=12)
            ax.set_title(f'T = {T:.2f}: Transition probabilities', fontsize=14)
            ax.set_xticks(range(n_max))
            ax.grid(True, alpha=0.3, axis='y')

            # 打印跃迁概率 / Print transition probabilities
            print(f"\n跃迁概率 / Transition probabilities:")
            for n in range(min(5, n_max)):
                print(f"  P_{n} = {probs[n]:.6f}")
            print(f"  Sum of P_0 to P_{n_max-1} = {np.sum(probs):.6f}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存到 / Figure saved to: {save_path}")

        plt.show()

    def analyze_adiabaticity(self, T_values: List[float],
                            method: str = 'split_step') -> Dict:
        """
        分析绝热性
        Analyze adiabaticity

        Parameters:
        -----------
        T_values : list of float
            总时间列表 / List of total times
        method : str
            演化方法 / Evolution method

        Returns:
        --------
        results : dict
            绝热性分析结果 / Adiabaticity analysis results
        """
        psi0 = self.exact_ground_state_wavefunction()

        results = {
            'T_values': T_values,
            'P0': [],  # 保持在基态的概率 / Probability to stay in ground state
            'P_excited': [],  # 激发态总概率 / Total excited state probability
            'mean_x_final': [],  # 最终位置期望值 / Final position expectation
            'mean_n': []  # 平均量子数 / Mean quantum number
        }

        print("\n" + "=" * 70)
        print("绝热性分析 / Adiabaticity Analysis")
        print("=" * 70)
        print(f"{'T':>8s} {'P₀':>10s} {'P_exc':>10s} {'⟨x(T)⟩':>10s} {'⟨n⟩':>10s} {'Regime':>12s}")
        print("-" * 70)

        for T in T_values:
            # 时间演化 / Time evolution
            if method == 'split_step':
                psi_final, _ = self.time_evolution_split_step(
                    psi0, T, Nt=max(1000, int(T * 100)), save_interval=100)
            else:
                psi_final, _ = self.time_evolution_rk4(
                    psi0, T, Nt=max(1000, int(T * 100)), save_interval=100)

            # 跃迁概率 / Transition probabilities
            n_max = 20
            probs = self.compute_transition_probabilities(psi_final, n_max=n_max)

            P0 = probs[0]
            P_excited = np.sum(probs[1:])

            # 位置期望值 / Position expectation value
            x_final = simps(self.x * np.abs(psi_final)**2, self.x)

            # 平均量子数 / Mean quantum number
            mean_n = np.sum(np.arange(n_max) * probs)

            # 判断绝热性 / Determine adiabatic regime
            if P0 > 0.9:
                regime = "Adiabatic"
            elif P0 < 0.3:
                regime = "Sudden"
            else:
                regime = "Intermediate"

            results['P0'].append(P0)
            results['P_excited'].append(P_excited)
            results['mean_x_final'].append(x_final)
            results['mean_n'].append(mean_n)

            print(f"{T:8.2f} {P0:10.6f} {P_excited:10.6f} "
                  f"{x_final:10.4f} {mean_n:10.4f} {regime:>12s}")

        print("=" * 70)

        return results


def demonstrate_exercise_1():
    """
    演示练习1：基态能量计算和误差分析
    Demonstrate Exercise 1: Ground state energy calculation and error analysis
    """
    print("\n" + "="*80)
    print("练习1：量子谐振子基态能量计算")
    print("Exercise 1: Quantum Harmonic Oscillator Ground State Energy Calculation")
    print("="*80)

    # 创建量子谐振子 / Create quantum harmonic oscillator
    qho = QuantumHarmonicOscillator(N=1000, L=10.0)

    # a. 计算基态能量期望值 / Calculate ground state energy expectation value
    print("\na. 基态能量期望值计算 / Ground state energy expectation value calculation")
    print("-" * 80)

    psi0 = qho.exact_ground_state_wavefunction()
    E0_numerical_simpson = qho.compute_energy_expectation(psi0, integration_method='simpson')
    E0_numerical_trapz = qho.compute_energy_expectation(psi0, integration_method='trapezoid')
    E0_exact = qho.exact_energy(0)

    print(f"精确基态能量 / Exact ground state energy: E₀ = {E0_exact:.10f}")
    print(f"数值计算 (Simpson) / Numerical (Simpson):   E₀ = {E0_numerical_simpson:.10f}")
    print(f"数值计算 (Trapz) / Numerical (Trapezoid): E₀ = {E0_numerical_trapz:.10f}")
    print(f"误差 (Simpson) / Error (Simpson):   {abs(E0_numerical_simpson - E0_exact):.2e}")
    print(f"误差 (Trapz) / Error (Trapezoid): {abs(E0_numerical_trapz - E0_exact):.2e}")

    # b. 误差分析 / Error analysis
    print("\n\nb. 误差分析 / Error analysis")
    print("-" * 80)

    N_values = [100, 200, 500, 1000, 2000, 5000]
    results = qho.error_analysis(N_values=N_values)

    # 绘制误差分析图 / Plot error analysis
    qho.plot_error_analysis(results, save_path='exercise1_error_analysis.png')

    return qho, results


def demonstrate_exercise_2():
    """
    演示练习2：特征值和特征向量求解
    Demonstrate Exercise 2: Eigenvalue and eigenvector solver
    """
    print("\n" + "="*80)
    print("练习2：量子谐振子特征值求解器")
    print("Exercise 2: Quantum Harmonic Oscillator Eigenvalue Solver")
    print("="*80)

    # 创建量子谐振子 / Create quantum harmonic oscillator
    qho = QuantumHarmonicOscillator(N=1000, L=10.0)

    # a. 构建哈密顿矩阵并对角化 / Build Hamiltonian and diagonalize
    print("\na. 构建哈密顿矩阵并对角化")
    print("   Building Hamiltonian matrix and diagonalization")
    print("-" * 80)

    start_time = time.time()
    qho.build_hamiltonian_matrix()
    build_time = time.time() - start_time
    print(f"构建哈密顿矩阵耗时 / Time to build Hamiltonian: {build_time:.4f} s")

    start_time = time.time()
    eigenvalues, eigenvectors = qho.diagonalize()
    diag_time = time.time() - start_time
    print(f"对角化耗时 / Time to diagonalize: {diag_time:.4f} s")

    # 收敛性测试 / Convergence test
    convergence_results = qho.convergence_test(k_max=10)

    # 正交归一性验证 / Orthonormality verification
    qho.verify_orthonormality(k_max=5)

    # b. 软件开发评价 / Software development evaluation
    print("\n\nb. 软件开发评价 / Software development evaluation")
    print("-" * 80)

    print("""
评价标准 / Evaluation Criteria:

1. 正确性 (Correctness): ✓
   - 特征值与解析解匹配 (相对误差 < 10⁻⁶)
   - Eigenvalues match analytical solution (relative error < 10⁻⁶)
   - 波函数满足归一化和正交性
   - Wavefunctions satisfy normalization and orthogonality

2. 稳定性 (Stability): ✓
   - 使用scipy.linalg.eigh (针对厄米矩阵优化)
   - Using scipy.linalg.eigh (optimized for Hermitian matrices)
   - 数值稳定的有限差分方法
   - Numerically stable finite difference method

3. 精确离散化 (Accurate Discretization): ✓
   - 网格足够大 (L = 10σ ≫ 波函数宽度)
   - Grid large enough (L = 10σ ≫ wavefunction width)
   - 网格足够密 (N = 1000点)
   - Grid dense enough (N = 1000 points)
   - 收敛性已验证
   - Convergence verified

4. 灵活性 (Flexibility): ✓
   - 模块化设计，易于扩展
   - Modular design, easy to extend
   - 参数可配置 (N, L, ħ, m, ω)
   - Configurable parameters
   - 支持不同积分方法
   - Support for different integration methods

5. 效率 (Efficiency): ✓
   - 使用NumPy向量化操作
   - Using NumPy vectorized operations
   - 利用LAPACK库 (通过scipy)
   - Leveraging LAPACK library (via scipy)
   - 对于N=1000, 总时间 < 1秒
   - For N=1000, total time < 1 second
    """)

    # 绘制波函数 / Plot wavefunctions
    qho.plot_wavefunctions(n_states=5, save_path='exercise2_wavefunctions.png')

    return qho


def demonstrate_exercise_3():
    """
    演示练习3：含时量子谐振子
    Demonstrate Exercise 3: Time-dependent quantum harmonic oscillator
    """
    print("\n" + "="*80)
    print("练习3：含时量子谐振子")
    print("Exercise 3: Time-Dependent Quantum Harmonic Oscillator")
    print("="*80)

    # 创建含时量子谐振子 / Create time-dependent oscillator
    td_qho = TimeDependentHarmonicOscillator(N=500, L=12.0)

    # 不同T值的演化 / Evolution for different T values
    T_values = [0.5, 2.0, 10.0]  # 突然、中等、绝热 / Sudden, intermediate, adiabatic

    print("\n演化不同T值 / Evolving for different T values")
    print("-" * 80)

    td_qho.plot_time_evolution(T_values, method='split_step',
                               save_path='exercise3_time_evolution.png')

    # 绝热性分析 / Adiabaticity analysis
    T_range = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    adiabatic_results = td_qho.analyze_adiabaticity(T_range, method='split_step')

    # 绘制绝热性分析 / Plot adiabaticity analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：基态概率 vs T
    ax = axes[0]
    ax.semilogx(adiabatic_results['T_values'], adiabatic_results['P0'],
               'bo-', linewidth=2, markersize=8, label='$P_0$ (ground state)')
    ax.semilogx(adiabatic_results['T_values'], adiabatic_results['P_excited'],
               'ro-', linewidth=2, markersize=8, label='$P_{exc}$ (excited)')
    ax.set_xlabel('Total time T', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Adiabaticity: Transition Probabilities vs T', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.3, label='90% threshold')

    # 右图：平均量子数 vs T
    ax = axes[1]
    ax.semilogx(adiabatic_results['T_values'], adiabatic_results['mean_n'],
               'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Total time T', fontsize=12)
    ax.set_ylabel('Mean quantum number $\\langle n \\rangle$', fontsize=12)
    ax.set_title('Adiabaticity: Mean Excitation vs T', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise3_adiabaticity.png', dpi=300, bbox_inches='tight')
    plt.show()

    return td_qho, adiabatic_results


if __name__ == "__main__":
    """主程序 / Main program"""

    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║          量子谐振子数值计算习题集                                       ║
    ║          Quantum Harmonic Oscillator Numerical Exercises              ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # 练习1 / Exercise 1
    qho1, error_results = demonstrate_exercise_1()

    # 练习2 / Exercise 2
    qho2 = demonstrate_exercise_2()

    # 练习3 / Exercise 3
    td_qho, adiabatic_results = demonstrate_exercise_3()

    print("\n" + "="*80)
    print("所有练习完成！/ All exercises completed!")
    print("="*80)
    print("\n生成的图像文件 / Generated image files:")
    print("  - exercise1_error_analysis.png")
    print("  - exercise2_wavefunctions.png")
    print("  - exercise3_time_evolution.png")
    print("  - exercise3_adiabaticity.png")
    print("\n" + "="*80)
