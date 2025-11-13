"""
练习 2：间距分布 P(s) 的研究
Exercise 2: Study of Spacing Distribution P(s)

研究归一化特征值间距的统计分布，比较：
- 随机厄米矩阵（Wigner-Dyson 统计）
- 对角矩阵（泊松统计）
- 理论预测与拟合
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from typing import List, Tuple, Dict
import time

from hermitian_matrix_analysis import HermitianMatrixAnalyzer


class SpacingDistributionAnalyzer:
    """间距分布分析器"""

    def __init__(self, seed: int = None):
        """
        初始化分析器

        Parameters:
        -----------
        seed : int, optional
            随机数种子
        """
        if seed is not None:
            np.random.seed(seed)
        self.hermitian_spacings = []
        self.diagonal_spacings = []

    def collect_hermitian_spacings(self,
                                   N_values: List[int],
                                   num_matrices: int = 10) -> np.ndarray:
        """
        收集多个随机厄米矩阵的归一化间距

        Parameters:
        -----------
        N_values : List[int]
            矩阵大小列表
        num_matrices : int
            每个大小生成的矩阵数量

        Returns:
        --------
        all_spacings : np.ndarray
            所有矩阵的归一化间距
        """
        all_spacings = []

        print(f"\n收集随机厄米矩阵的间距分布...")
        for N in N_values:
            print(f"  N = {N}: 生成 {num_matrices} 个矩阵...", end=" ")
            start_time = time.time()

            for i in range(num_matrices):
                analyzer = HermitianMatrixAnalyzer(N)
                analyzer.generate_hermitian_matrix()
                analyzer.diagonalize_matrix()

                # 计算归一化间距
                results = analyzer.calculate_normalized_spacings(method='global')
                spacings = results['global']
                all_spacings.extend(spacings)

            elapsed = time.time() - start_time
            print(f"完成 ({elapsed:.2f}秒)")

        self.hermitian_spacings = np.array(all_spacings)
        print(f"总共收集 {len(self.hermitian_spacings)} 个间距值")

        return self.hermitian_spacings

    def collect_diagonal_spacings(self,
                                 N_values: List[int],
                                 num_matrices: int = 10) -> np.ndarray:
        """
        收集多个对角矩阵的归一化间距

        对角矩阵的特征值就是对角元素，应该服从泊松统计（无能级排斥）

        Parameters:
        -----------
        N_values : List[int]
            矩阵大小列表
        num_matrices : int
            每个大小生成的矩阵数量

        Returns:
        --------
        all_spacings : np.ndarray
            所有矩阵的归一化间距
        """
        all_spacings = []

        print(f"\n收集对角矩阵的间距分布...")
        for N in N_values:
            print(f"  N = {N}: 生成 {num_matrices} 个矩阵...", end=" ")
            start_time = time.time()

            for i in range(num_matrices):
                # 生成随机对角矩阵（对角元素为随机实数）
                diagonal_elements = np.random.randn(N)

                # 排序（模拟特征值）
                eigenvalues = np.sort(diagonal_elements)

                # 计算间距
                raw_spacings = np.diff(eigenvalues)

                # 归一化
                mean_spacing = np.mean(raw_spacings)
                if mean_spacing > 0:
                    normalized_spacings = raw_spacings / mean_spacing
                    all_spacings.extend(normalized_spacings)

            elapsed = time.time() - start_time
            print(f"完成 ({elapsed:.2f}秒)")

        self.diagonal_spacings = np.array(all_spacings)
        print(f"总共收集 {len(self.diagonal_spacings)} 个间距值")

        return self.diagonal_spacings

    @staticmethod
    def general_spacing_distribution(s: np.ndarray, a: float, alpha: float,
                                    b: float, beta: float) -> np.ndarray:
        """
        通用间距分布函数: P(s) = a * s^α * exp(-b * s^β)

        Parameters:
        -----------
        s : np.ndarray
            归一化间距值
        a : float
            归一化常数
        alpha : float
            低能量行为的幂指数
        b : float
            衰减速率
        beta : float
            高能量衰减的幂指数

        Returns:
        --------
        P(s) : np.ndarray
            概率密度
        """
        return a * np.power(s, alpha) * np.exp(-b * np.power(s, beta))

    def fit_spacing_distribution(self,
                                spacings: np.ndarray,
                                initial_guess: Tuple[float, float, float, float] = None,
                                s_range: Tuple[float, float] = None) -> Dict:
        """
        拟合间距分布到函数 P(s) = a * s^α * exp(-b * s^β)

        Parameters:
        -----------
        spacings : np.ndarray
            归一化间距数据
        initial_guess : Tuple[float, float, float, float]
            初始参数猜测 (a, alpha, b, beta)
            如果为 None，自动选择
        s_range : Tuple[float, float]
            拟合的间距范围

        Returns:
        --------
        results : Dict
            包含拟合参数、协方差、拟合质量等
        """
        # 创建直方图
        bins = 100
        counts, bin_edges = np.histogram(spacings, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 选择拟合范围
        if s_range is not None:
            mask = (bin_centers >= s_range[0]) & (bin_centers <= s_range[1])
            bin_centers = bin_centers[mask]
            counts = counts[mask]

        # 只拟合非零的 bins
        mask = counts > 0
        bin_centers = bin_centers[mask]
        counts = counts[mask]

        # 初始参数猜测
        if initial_guess is None:
            # 根据数据特性自动猜测
            max_count = np.max(counts)
            max_s = bin_centers[np.argmax(counts)]

            if max_s < 0.1:  # 泊松型
                initial_guess = (1.0, 0.0, 1.0, 1.0)
            else:  # Wigner-Dyson 型
                initial_guess = (max_count, 1.0, 1.0, 2.0)

        try:
            # 执行拟合
            popt, pcov = curve_fit(
                self.general_spacing_distribution,
                bin_centers,
                counts,
                p0=initial_guess,
                maxfev=50000,
                bounds=([0, 0, 0, 0], [np.inf, 10, np.inf, 10]),
                ftol=1e-6,
                xtol=1e-6
            )

            # 计算拟合质量
            fitted_values = self.general_spacing_distribution(bin_centers, *popt)
            residuals = counts - fitted_values
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((counts - np.mean(counts))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 计算标准误差
            perr = np.sqrt(np.diag(pcov))

            results = {
                'a': popt[0],
                'alpha': popt[1],
                'b': popt[2],
                'beta': popt[3],
                'a_err': perr[0],
                'alpha_err': perr[1],
                'b_err': perr[2],
                'beta_err': perr[3],
                'covariance': pcov,
                'r_squared': r_squared,
                'bin_centers': bin_centers,
                'counts': counts,
                'fitted_values': fitted_values,
                'success': True
            }

        except Exception as e:
            print(f"拟合失败: {e}")
            results = {
                'success': False,
                'error': str(e)
            }

        return results

    @staticmethod
    def goe_distribution(s: np.ndarray) -> np.ndarray:
        """GOE 理论预测: P(s) = (π/2) s exp(-πs²/4)"""
        return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    @staticmethod
    def gue_distribution(s: np.ndarray) -> np.ndarray:
        """GUE 理论预测: P(s) = (32/π²) s² exp(-4s²/π)"""
        return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

    @staticmethod
    def poisson_distribution(s: np.ndarray) -> np.ndarray:
        """泊松分布: P(s) = exp(-s)"""
        return np.exp(-s)

    def compare_with_theory(self,
                           spacings: np.ndarray,
                           theory_type: str = 'GUE') -> Dict:
        """
        与理论分布比较

        Parameters:
        -----------
        spacings : np.ndarray
            归一化间距数据
        theory_type : str
            'GOE', 'GUE', 或 'Poisson'

        Returns:
        --------
        comparison : Dict
            包含 KS 统计量、p 值等
        """
        # 生成理论分布的样本（通过接受-拒绝采样）
        n_samples = len(spacings)
        s_max = np.max(spacings) * 2

        if theory_type == 'GOE':
            theory_func = self.goe_distribution
        elif theory_type == 'GUE':
            theory_func = self.gue_distribution
        elif theory_type == 'Poisson':
            theory_func = self.poisson_distribution
        else:
            raise ValueError(f"未知的理论类型: {theory_type}")

        # KS 检验（使用经验累积分布函数）
        # 注意：这里我们比较的是直方图，不是直接的样本
        # 更准确的方法是比较累积分布函数

        from scipy.stats import kstest

        # 定义累积分布函数
        def theory_cdf(s):
            s_vals = np.linspace(0, s, 1000)
            return np.trapz(theory_func(s_vals), s_vals)

        # 对每个数据点计算理论 CDF
        theory_cdf_values = np.array([theory_cdf(s) for s in np.sort(spacings)])

        # 简化版本：直接比较直方图
        bins = 100
        s_max = min(5, np.max(spacings))
        hist_data, bin_edges = np.histogram(spacings, bins=bins, density=True,
                                            range=(0, s_max))
        # 使用 bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist_theory = theory_func(bin_centers)

        # 计算均方误差
        mse = np.mean((hist_data - hist_theory)**2)

        return {
            'theory_type': theory_type,
            'mse': mse,
            'mean_spacing': np.mean(spacings),
            'std_spacing': np.std(spacings)
        }


def plot_spacing_comparison(analyzer: SpacingDistributionAnalyzer,
                           hermitian_fit: Dict = None,
                           diagonal_fit: Dict = None,
                           save_path: str = None):
    """
    绘制间距分布的全面比较图

    Parameters:
    -----------
    analyzer : SpacingDistributionAnalyzer
        包含数据的分析器
    hermitian_fit : Dict
        厄米矩阵的拟合结果
    diagonal_fit : Dict
        对角矩阵的拟合结果
    save_path : str, optional
        保存图像的路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 理论曲线的 s 值
    s_theory = np.linspace(0, 5, 500)
    goe_theory = SpacingDistributionAnalyzer.goe_distribution(s_theory)
    gue_theory = SpacingDistributionAnalyzer.gue_distribution(s_theory)
    poisson_theory = SpacingDistributionAnalyzer.poisson_distribution(s_theory)

    # 图 1: 随机厄米矩阵的间距分布
    ax1 = axes[0, 0]
    if len(analyzer.hermitian_spacings) > 0:
        ax1.hist(analyzer.hermitian_spacings, bins=80, density=True,
                alpha=0.6, edgecolor='black', label='数值结果', range=(0, 5))
        ax1.plot(s_theory, gue_theory, 'r-', linewidth=2.5, label='GUE 理论')
        ax1.plot(s_theory, goe_theory, 'g--', linewidth=2, label='GOE 理论', alpha=0.7)

        if hermitian_fit and hermitian_fit.get('success'):
            s_fit = np.linspace(0, 5, 500)
            fit_curve = SpacingDistributionAnalyzer.general_spacing_distribution(
                s_fit, hermitian_fit['a'], hermitian_fit['alpha'],
                hermitian_fit['b'], hermitian_fit['beta']
            )
            ax1.plot(s_fit, fit_curve, 'b:', linewidth=2, label='拟合曲线')

            # 添加拟合参数文本
            param_text = (f"$\\alpha$ = {hermitian_fit['alpha']:.3f} ± {hermitian_fit['alpha_err']:.3f}\n"
                         f"$\\beta$ = {hermitian_fit['beta']:.3f} ± {hermitian_fit['beta_err']:.3f}\n"
                         f"$R^2$ = {hermitian_fit['r_squared']:.4f}")
            ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)

    ax1.set_xlabel('归一化间距 s', fontsize=11)
    ax1.set_ylabel('概率密度 P(s)', fontsize=11)
    ax1.set_title('(a) 随机厄米矩阵 - Wigner-Dyson 统计', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)

    # 图 2: 对角矩阵的间距分布
    ax2 = axes[0, 1]
    if len(analyzer.diagonal_spacings) > 0:
        ax2.hist(analyzer.diagonal_spacings, bins=80, density=True,
                alpha=0.6, edgecolor='black', label='数值结果', range=(0, 5), color='orange')
        ax2.plot(s_theory, poisson_theory, 'm-', linewidth=2.5, label='泊松理论')

        if diagonal_fit and diagonal_fit.get('success'):
            s_fit = np.linspace(0, 5, 500)
            fit_curve = SpacingDistributionAnalyzer.general_spacing_distribution(
                s_fit, diagonal_fit['a'], diagonal_fit['alpha'],
                diagonal_fit['b'], diagonal_fit['beta']
            )
            ax2.plot(s_fit, fit_curve, 'c:', linewidth=2, label='拟合曲线')

            # 添加拟合参数文本
            param_text = (f"$\\alpha$ = {diagonal_fit['alpha']:.3f} ± {diagonal_fit['alpha_err']:.3f}\n"
                         f"$\\beta$ = {diagonal_fit['beta']:.3f} ± {diagonal_fit['beta_err']:.3f}\n"
                         f"$R^2$ = {diagonal_fit['r_squared']:.4f}")
            ax2.text(0.95, 0.95, param_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=9)

    ax2.set_xlabel('归一化间距 s', fontsize=11)
    ax2.set_ylabel('概率密度 P(s)', fontsize=11)
    ax2.set_title('(b) 对角矩阵 - 泊松统计', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)

    # 图 3: 对数尺度比较（能级排斥）
    ax3 = axes[1, 0]
    if len(analyzer.hermitian_spacings) > 0:
        counts_h, bins_h, _ = ax3.hist(analyzer.hermitian_spacings, bins=80,
                                        density=True, alpha=0.6,
                                        edgecolor='black', label='厄米矩阵',
                                        range=(0, 5))
    if len(analyzer.diagonal_spacings) > 0:
        counts_d, bins_d, _ = ax3.hist(analyzer.diagonal_spacings, bins=80,
                                        density=True, alpha=0.6,
                                        edgecolor='black', label='对角矩阵',
                                        range=(0, 5), color='orange')

    ax3.plot(s_theory, gue_theory, 'r-', linewidth=2, label='GUE', alpha=0.8)
    ax3.plot(s_theory, poisson_theory, 'm-', linewidth=2, label='泊松', alpha=0.8)

    ax3.set_yscale('log')
    ax3.set_xlabel('归一化间距 s', fontsize=11)
    ax3.set_ylabel('概率密度 P(s) [对数]', fontsize=11)
    ax3.set_title('(c) 对数尺度 - 能级排斥效应', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(0, 5)
    ax3.set_ylim(1e-4, 2)

    # 图 4: 小间距区域放大（展示能级排斥）
    ax4 = axes[1, 1]
    if len(analyzer.hermitian_spacings) > 0:
        ax4.hist(analyzer.hermitian_spacings, bins=100, density=True,
                alpha=0.6, edgecolor='black', label='厄米矩阵', range=(0, 1))
    if len(analyzer.diagonal_spacings) > 0:
        ax4.hist(analyzer.diagonal_spacings, bins=100, density=True,
                alpha=0.6, edgecolor='black', label='对角矩阵',
                range=(0, 1), color='orange')

    s_small = np.linspace(0, 1, 200)
    ax4.plot(s_small, SpacingDistributionAnalyzer.gue_distribution(s_small),
            'r-', linewidth=2.5, label='GUE: $P(s) \\sim s^2$')
    ax4.plot(s_small, SpacingDistributionAnalyzer.poisson_distribution(s_small),
            'm-', linewidth=2.5, label='泊松: $P(0) = 1$')

    ax4.set_xlabel('归一化间距 s', fontsize=11)
    ax4.set_ylabel('概率密度 P(s)', fontsize=11)
    ax4.set_title('(d) 小间距区域 - 能级排斥', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    return fig


def print_fit_results(fit_results: Dict, matrix_type: str):
    """
    打印拟合结果

    Parameters:
    -----------
    fit_results : Dict
        拟合结果字典
    matrix_type : str
        矩阵类型描述
    """
    print(f"\n{'='*60}")
    print(f"{matrix_type}的拟合结果")
    print(f"{'='*60}")

    if not fit_results.get('success'):
        print(f"拟合失败: {fit_results.get('error', '未知错误')}")
        return

    print(f"\n拟合函数: P(s) = a * s^α * exp(-b * s^β)")
    print(f"\n参数估计:")
    print(f"  a     = {fit_results['a']:.6f} ± {fit_results['a_err']:.6f}")
    print(f"  α     = {fit_results['alpha']:.6f} ± {fit_results['alpha_err']:.6f}")
    print(f"  b     = {fit_results['b']:.6f} ± {fit_results['b_err']:.6f}")
    print(f"  β     = {fit_results['beta']:.6f} ± {fit_results['beta_err']:.6f}")
    print(f"\n拟合质量:")
    print(f"  R²    = {fit_results['r_squared']:.6f}")

    # 与理论值比较
    print(f"\n与理论预测比较:")
    if matrix_type == "随机厄米矩阵":
        print(f"  GUE 理论: α = 2, β = 2")
        print(f"  GOE 理论: α = 1, β = 2")
        alpha_diff_gue = abs(fit_results['alpha'] - 2.0)
        beta_diff_gue = abs(fit_results['beta'] - 2.0)
        alpha_diff_goe = abs(fit_results['alpha'] - 1.0)
        beta_diff_goe = abs(fit_results['beta'] - 2.0)
        print(f"  与 GUE 的偏差: Δα = {alpha_diff_gue:.4f}, Δβ = {beta_diff_gue:.4f}")
        print(f"  与 GOE 的偏差: Δα = {alpha_diff_goe:.4f}, Δβ = {beta_diff_goe:.4f}")
    elif matrix_type == "对角矩阵":
        print(f"  泊松理论: α = 0, β = 1")
        alpha_diff = abs(fit_results['alpha'] - 0.0)
        beta_diff = abs(fit_results['beta'] - 1.0)
        print(f"  与泊松的偏差: Δα = {alpha_diff:.4f}, Δβ = {beta_diff:.4f}")


if __name__ == "__main__":
    print("="*70)
    print("练习 2：间距分布 P(s) 的研究")
    print("Exercise 2: Study of Spacing Distribution P(s)")
    print("="*70)

    # 初始化分析器
    analyzer = SpacingDistributionAnalyzer(seed=42)

    # 任务 a: 收集随机厄米矩阵的间距
    print("\n任务 (a): 计算随机厄米矩阵的 P(s)")
    print("-"*70)
    N_values_hermitian = [100, 200, 300, 400, 500]
    num_matrices = 5
    hermitian_spacings = analyzer.collect_hermitian_spacings(
        N_values_hermitian, num_matrices
    )

    # 任务 b: 收集对角矩阵的间距
    print("\n任务 (b): 计算对角矩阵的 P(s)")
    print("-"*70)
    N_values_diagonal = [100, 200, 300, 400, 500]
    diagonal_spacings = analyzer.collect_diagonal_spacings(
        N_values_diagonal, num_matrices
    )

    # 任务 c: 拟合分布函数
    print("\n任务 (c): 拟合分布函数 P(s) = a·s^α·exp(-b·s^β)")
    print("-"*70)

    # 拟合厄米矩阵数据
    print("\n拟合随机厄米矩阵...")
    hermitian_fit = analyzer.fit_spacing_distribution(
        hermitian_spacings,
        initial_guess=(1.0, 2.0, 1.0, 2.0),  # 接近 GUE
        s_range=(0.1, 4.0)
    )
    print_fit_results(hermitian_fit, "随机厄米矩阵")

    # 拟合对角矩阵数据
    print("\n拟合对角矩阵...")
    diagonal_fit = analyzer.fit_spacing_distribution(
        diagonal_spacings,
        initial_guess=(1.0, 0.0, 1.0, 1.0),  # 接近泊松
        s_range=(0.0, 4.0)
    )
    print_fit_results(diagonal_fit, "对角矩阵")

    # 任务 d: 与随机矩阵理论预测比较
    print("\n任务 (d): 与随机矩阵理论预测比较")
    print("-"*70)

    hermitian_comparison = analyzer.compare_with_theory(hermitian_spacings, 'GUE')
    print(f"\n随机厄米矩阵 vs GUE 理论:")
    print(f"  均方误差: {hermitian_comparison['mse']:.6f}")
    print(f"  平均间距: {hermitian_comparison['mean_spacing']:.6f}")
    print(f"  标准差:   {hermitian_comparison['std_spacing']:.6f}")

    diagonal_comparison = analyzer.compare_with_theory(diagonal_spacings, 'Poisson')
    print(f"\n对角矩阵 vs 泊松理论:")
    print(f"  均方误差: {diagonal_comparison['mse']:.6f}")
    print(f"  平均间距: {diagonal_comparison['mean_spacing']:.6f}")
    print(f"  标准差:   {diagonal_comparison['std_spacing']:.6f}")

    # 绘制综合比较图
    print("\n生成可视化图表...")
    fig = plot_spacing_comparison(
        analyzer,
        hermitian_fit,
        diagonal_fit,
        save_path='exercise2_spacing_distribution.png'
    )

    print("\n练习 2 完成!")
    print("="*70)
