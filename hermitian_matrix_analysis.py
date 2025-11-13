"""
随机厄米矩阵特征值分析
Random Hermitian Matrix Eigenvalue Analysis

这个模块实现了随机矩阵理论中的经典问题，用于研究：
- 能级统计和能级排斥
- LU 分解的标度行为
- 归一化间距的统计性质
- Wigner-Dyson 统计
"""

import numpy as np
import scipy.linalg as la
from scipy.stats import expon
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class HermitianMatrixAnalyzer:
    """随机厄米矩阵分析器"""

    def __init__(self, N: int, seed: int = None):
        """
        初始化分析器

        Parameters:
        -----------
        N : int
            矩阵大小
        seed : int, optional
            随机数种子，用于可重复性
        """
        self.N = N
        if seed is not None:
            np.random.seed(seed)
        self.A = None
        self.eigenvalues = None
        self.eigenvectors = None

    def generate_hermitian_matrix(self) -> np.ndarray:
        """
        生成随机厄米矩阵

        对于高斯酉系综(GUE)，矩阵元素从以下分布中采样：
        - 对角元素: 实数，方差为 2
        - 非对角元素: 复数，实部和虚部独立，方差各为 1

        Returns:
        --------
        A : np.ndarray
            N×N 厄米矩阵
        """
        # 生成随机复矩阵
        # 实部和虚部都是从标准正态分布采样
        real_part = np.random.randn(self.N, self.N)
        imag_part = np.random.randn(self.N, self.N)

        # 组合成复矩阵
        random_matrix = real_part + 1j * imag_part

        # 构造厄米矩阵: A = (M + M†)/2
        # 这确保了 A = A†
        self.A = (random_matrix + random_matrix.conj().T) / 2

        return self.A

    def perform_lu_decomposition(self, measure_time: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        执行 LU 分解

        Parameters:
        -----------
        measure_time : bool
            是否测量执行时间

        Returns:
        --------
        P : np.ndarray
            置换矩阵
        L : np.ndarray
            下三角矩阵
        U : np.ndarray
            上三角矩阵
        elapsed_time : float
            执行时间（秒）
        """
        if self.A is None:
            raise ValueError("请先生成厄米矩阵")

        start_time = time.time() if measure_time else 0

        # 执行 LU 分解: PA = LU
        P, L, U = la.lu(self.A)

        elapsed_time = time.time() - start_time if measure_time else 0

        return P, L, U, elapsed_time

    def diagonalize_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        对角化矩阵并按递增顺序排列特征值

        Returns:
        --------
        eigenvalues : np.ndarray
            排序后的特征值（递增顺序）
        eigenvectors : np.ndarray
            对应的特征向量
        """
        if self.A is None:
            raise ValueError("请先生成厄米矩阵")

        # 对厄米矩阵使用 eigh（更快且数值稳定）
        eigenvalues, eigenvectors = la.eigh(self.A)

        # eigh 已经返回按递增顺序排列的特征值
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        return self.eigenvalues, self.eigenvectors

    def calculate_normalized_spacings(self,
                                     window_sizes: List[int] = None,
                                     method: str = 'global') -> Dict[str, np.ndarray]:
        """
        计算归一化特征值间距

        Parameters:
        -----------
        window_sizes : List[int], optional
            用于局部平均的窗口大小列表
            如果为 None，则使用 [N/100, N/50, N/10, N/5, N]
        method : str
            'global': 使用全局平均间距
            'local': 使用局部平均间距

        Returns:
        --------
        results : Dict[str, np.ndarray]
            包含不同方法的归一化间距
        """
        if self.eigenvalues is None:
            raise ValueError("请先对角化矩阵")

        # 计算原始间距 Δλᵢ = λᵢ₊₁ - λᵢ
        raw_spacings = np.diff(self.eigenvalues)

        results = {}

        if method == 'global':
            # 全局平均间距
            mean_spacing = np.mean(raw_spacings)
            normalized_spacings = raw_spacings / mean_spacing
            results['global'] = normalized_spacings

        if window_sizes is None:
            window_sizes = [
                max(2, self.N // 100),
                max(2, self.N // 50),
                max(2, self.N // 10),
                max(2, self.N // 5),
                self.N
            ]

        # 局部平均分析
        for window_size in window_sizes:
            local_normalized = self._local_average_normalization(raw_spacings, window_size)
            results[f'local_window_{window_size}'] = local_normalized

        return results

    def _local_average_normalization(self, spacings: np.ndarray, window_size: int) -> np.ndarray:
        """
        使用局部平均进行归一化

        Parameters:
        -----------
        spacings : np.ndarray
            原始间距
        window_size : int
            窗口大小

        Returns:
        --------
        normalized : np.ndarray
            归一化间距
        """
        n = len(spacings)
        normalized = np.zeros_like(spacings)

        for i in range(n):
            # 确定窗口范围
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window)

            # 计算局部平均
            local_mean = np.mean(spacings[start:end])
            normalized[i] = spacings[i] / local_mean if local_mean > 0 else 0

        return normalized

    def wigner_dyson_distribution(self, s: np.ndarray, ensemble: str = 'GOE') -> np.ndarray:
        """
        计算 Wigner-Dyson 分布（理论预测）

        Parameters:
        -----------
        s : np.ndarray
            归一化间距值
        ensemble : str
            'GOE': 高斯正交系综 (实对称矩阵)
            'GUE': 高斯酉系综 (复厄米矩阵)
            'GSE': 高斯辛系综

        Returns:
        --------
        P(s) : np.ndarray
            概率密度
        """
        if ensemble == 'GOE':
            # P(s) = (π/2) s exp(-πs²/4)
            return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
        elif ensemble == 'GUE':
            # P(s) = (32/π²) s² exp(-4s²/π)
            return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
        elif ensemble == 'GSE':
            # P(s) = (2^18/3^6π³) s⁴ exp(-64s²/9π)
            return (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(-64 * s**2 / (9 * np.pi))
        else:
            raise ValueError(f"未知的系综类型: {ensemble}")

    def poisson_distribution(self, s: np.ndarray) -> np.ndarray:
        """
        泊松分布（无能级排斥的情况，对应于随机数序列）

        Parameters:
        -----------
        s : np.ndarray
            归一化间距值

        Returns:
        --------
        P(s) : np.ndarray
            概率密度 P(s) = exp(-s)
        """
        return np.exp(-s)


def analyze_lu_scaling(N_values: List[int], num_trials: int = 3, seed: int = 42) -> Dict:
    """
    分析 LU 分解的标度行为

    Parameters:
    -----------
    N_values : List[int]
        要测试的矩阵大小列表
    num_trials : int
        每个 N 重复的次数
    seed : int
        随机数种子

    Returns:
    --------
    results : Dict
        包含 N 值和对应的平均时间
    """
    times = []

    for N in N_values:
        trial_times = []
        for trial in range(num_trials):
            analyzer = HermitianMatrixAnalyzer(N, seed=seed + trial)
            analyzer.generate_hermitian_matrix()
            _, _, _, elapsed = analyzer.perform_lu_decomposition()
            trial_times.append(elapsed)

        avg_time = np.mean(trial_times)
        std_time = np.std(trial_times)
        times.append((avg_time, std_time))
        print(f"N = {N:4d}: {avg_time:.6f} ± {std_time:.6f} 秒")

    return {
        'N_values': N_values,
        'mean_times': [t[0] for t in times],
        'std_times': [t[1] for t in times]
    }


def plot_spacing_distribution(normalized_spacings: np.ndarray,
                              ensemble: str = 'GUE',
                              bins: int = 50,
                              title: str = None):
    """
    绘制归一化间距分布并与理论对比

    Parameters:
    -----------
    normalized_spacings : np.ndarray
        归一化间距数据
    ensemble : str
        理论分布类型
    bins : int
        直方图的箱数
    title : str
        图表标题
    """
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    counts, bin_edges, _ = plt.hist(normalized_spacings, bins=bins, density=True,
                                     alpha=0.6, label='数值结果', edgecolor='black')

    # 理论曲线
    s_theory = np.linspace(0, np.max(normalized_spacings), 200)
    analyzer = HermitianMatrixAnalyzer(100)  # 临时实例用于调用分布函数

    wd_dist = analyzer.wigner_dyson_distribution(s_theory, ensemble)
    poisson_dist = analyzer.poisson_distribution(s_theory)

    plt.plot(s_theory, wd_dist, 'r-', linewidth=2,
             label=f'Wigner-Dyson ({ensemble})')
    plt.plot(s_theory, poisson_dist, 'g--', linewidth=2,
             label='Poisson (无排斥)')

    plt.xlabel('归一化间距 s', fontsize=12)
    plt.ylabel('概率密度 P(s)', fontsize=12)
    plt.title(title or f'特征值间距分布 ({ensemble})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_lu_scaling(scaling_results: Dict):
    """
    绘制 LU 分解的标度行为

    Parameters:
    -----------
    scaling_results : Dict
        从 analyze_lu_scaling 返回的结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    N_values = np.array(scaling_results['N_values'])
    mean_times = np.array(scaling_results['mean_times'])
    std_times = np.array(scaling_results['std_times'])

    # 线性标度图
    ax1.errorbar(N_values, mean_times, yerr=std_times,
                 fmt='o-', capsize=5, label='实际时间')

    # 拟合 O(N³) 曲线
    if len(N_values) > 1:
        # 使用前两个点来估计系数
        coeff = mean_times[1] / (N_values[1]**3)
        fitted = coeff * N_values**3
        ax1.plot(N_values, fitted, 'r--', label=r'$O(N^3)$ 拟合')

    ax1.set_xlabel('矩阵大小 N', fontsize=12)
    ax1.set_ylabel('时间 (秒)', fontsize=12)
    ax1.set_title('LU 分解时间 vs 矩阵大小', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 对数标度图
    ax2.loglog(N_values, mean_times, 'o-', label='实际时间')

    if len(N_values) > 1:
        ax2.loglog(N_values, fitted, 'r--', label=r'$O(N^3)$ 拟合')

        # 添加参考线
        slope_2 = mean_times[0] * (N_values / N_values[0])**2
        slope_3 = mean_times[0] * (N_values / N_values[0])**3
        ax2.loglog(N_values, slope_2, 'g:', alpha=0.5, label=r'$O(N^2)$ 参考')
        ax2.loglog(N_values, slope_3, 'b:', alpha=0.5, label=r'$O(N^3)$ 参考')

    ax2.set_xlabel('矩阵大小 N', fontsize=12)
    ax2.set_ylabel('时间 (秒)', fontsize=12)
    ax2.set_title('LU 分解标度行为 (对数图)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    return fig


def compare_local_averaging_methods(analyzer: HermitianMatrixAnalyzer,
                                    window_sizes: List[int] = None):
    """
    比较不同局部平均窗口大小的效果

    Parameters:
    -----------
    analyzer : HermitianMatrixAnalyzer
        已经对角化的分析器实例
    window_sizes : List[int]
        窗口大小列表
    """
    if window_sizes is None:
        N = analyzer.N
        window_sizes = [
            max(2, N // 100),
            max(2, N // 50),
            max(2, N // 10),
            max(2, N // 5),
            N
        ]

    results = analyzer.calculate_normalized_spacings(window_sizes=window_sizes)

    n_plots = len(window_sizes) + 1  # +1 for global
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # 理论曲线
    s_theory = np.linspace(0, 5, 200)
    wd_dist = analyzer.wigner_dyson_distribution(s_theory, 'GUE')

    # 绘制全局平均
    if 'global' in results:
        ax = axes[0]
        ax.hist(results['global'], bins=50, density=True, alpha=0.6,
                edgecolor='black', label='数值')
        ax.plot(s_theory, wd_dist, 'r-', linewidth=2, label='Wigner-Dyson')
        ax.set_xlabel('归一化间距 s')
        ax.set_ylabel('P(s)')
        ax.set_title('全局平均')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 绘制不同窗口大小
    for idx, window_size in enumerate(window_sizes):
        ax = axes[idx + 1]
        key = f'local_window_{window_size}'
        if key in results:
            ax.hist(results[key], bins=50, density=True, alpha=0.6,
                   edgecolor='black', label='数值')
            ax.plot(s_theory, wd_dist, 'r-', linewidth=2, label='Wigner-Dyson')
            ax.set_xlabel('归一化间距 s')
            ax.set_ylabel('P(s)')
            ax.set_title(f'局部窗口: {window_size} ({window_size/analyzer.N:.1%} N)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("随机厄米矩阵特征值分析")
    print("Random Hermitian Matrix Eigenvalue Analysis")
    print("=" * 60)

    # 任务 a: LU 分解和标度分析
    print("\n任务 a: LU 分解标度分析")
    print("-" * 60)
    N_values = [50, 100, 200, 400, 800]
    scaling_results = analyze_lu_scaling(N_values, num_trials=3)

    # 任务 b, c, d: 特征值分析
    print("\n任务 b-d: 特征值间距分析")
    print("-" * 60)

    N = 1000
    print(f"\n使用 N = {N} 的随机厄米矩阵")

    analyzer = HermitianMatrixAnalyzer(N, seed=42)

    print("生成随机厄米矩阵...")
    A = analyzer.generate_hermitian_matrix()
    print(f"矩阵形状: {A.shape}")
    print(f"矩阵是厄米的: {np.allclose(A, A.conj().T)}")

    print("\n对角化矩阵...")
    eigenvalues, eigenvectors = analyzer.diagonalize_matrix()
    print(f"特征值范围: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    print(f"特征值个数: {len(eigenvalues)}")

    print("\n计算归一化间距...")
    results = analyzer.calculate_normalized_spacings()

    print("\n归一化间距统计:")
    for key, spacings in results.items():
        print(f"  {key}:")
        print(f"    平均值: {np.mean(spacings):.4f}")
        print(f"    标准差: {np.std(spacings):.4f}")
        print(f"    最小值: {np.min(spacings):.4f}")
        print(f"    最大值: {np.max(spacings):.4f}")

    print("\n分析完成!")
    print("使用 Jupyter notebook 查看可视化结果")
