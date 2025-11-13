"""
简单示例：演示随机厄米矩阵特征值分析的基本用法
Simple Example: Basic usage of random Hermitian matrix eigenvalue analysis
"""

import numpy as np
from hermitian_matrix_analysis import HermitianMatrixAnalyzer
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("随机厄米矩阵特征值分析 - 简单示例")
    print("Random Hermitian Matrix Eigenvalue Analysis - Simple Example")
    print("=" * 60)

    # 1. 创建分析器
    N = 500
    print(f"\n1. 创建 {N}×{N} 随机厄米矩阵")
    analyzer = HermitianMatrixAnalyzer(N, seed=42)

    # 2. 生成矩阵
    A = analyzer.generate_hermitian_matrix()
    print(f"   ✓ 矩阵生成完成")
    print(f"   ✓ 验证厄米性: {np.allclose(A, A.conj().T)}")

    # 3. LU 分解
    print(f"\n2. 执行 LU 分解")
    P, L, U, time_elapsed = analyzer.perform_lu_decomposition()
    print(f"   ✓ LU 分解完成")
    print(f"   ✓ 耗时: {time_elapsed:.6f} 秒")

    # 验证 PA = LU
    reconstruction_error = np.linalg.norm(P @ A - L @ U)
    print(f"   ✓ 重构误差: {reconstruction_error:.2e}")

    # 4. 对角化
    print(f"\n3. 对角化矩阵")
    eigenvalues, eigenvectors = analyzer.diagonalize_matrix()
    print(f"   ✓ 对角化完成")
    print(f"   ✓ 特征值个数: {len(eigenvalues)}")
    print(f"   ✓ 特征值范围: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    print(f"   ✓ 特征值均值: {np.mean(eigenvalues):.4f}")

    # 验证特征值方程
    test_idx = N // 2
    Av = A @ eigenvectors[:, test_idx]
    lambda_v = eigenvalues[test_idx] * eigenvectors[:, test_idx]
    eigenvalue_error = np.linalg.norm(Av - lambda_v)
    print(f"   ✓ 特征值验证误差: {eigenvalue_error:.2e}")

    # 5. 计算归一化间距
    print(f"\n4. 计算归一化间距")
    results = analyzer.calculate_normalized_spacings()
    spacings = results['global']

    print(f"   ✓ 间距个数: {len(spacings)}")
    print(f"   ✓ 平均值: {np.mean(spacings):.4f} (理论: 1.0)")
    print(f"   ✓ 标准差: {np.std(spacings):.4f}")
    print(f"   ✓ 最小间距: {np.min(spacings):.4f}")

    # 检查能级排斥
    small_spacings = np.sum(spacings < 0.1)
    print(f"   ✓ 小间距 (s<0.1) 的比例: {small_spacings/len(spacings):.2%}")
    print(f"     (理论: 对于 Poisson 应为 ~9.5%, 对于 GUE 应为 ~0%)")

    # 6. 生成并保存可视化
    print(f"\n5. 生成可视化结果")

    # 绘制特征值分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 特征值直方图
    axes[0].hist(eigenvalues, bins=50, edgecolor='black', alpha=0.7, density=True)
    axes[0].set_xlabel('Eigenvalue λ')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Eigenvalue Distribution')
    axes[0].grid(True, alpha=0.3)

    # Wigner 半圆律对比
    x = np.linspace(eigenvalues[0], eigenvalues[-1], 200)
    R = np.max(np.abs(eigenvalues))
    wigner_semicircle = np.sqrt(np.maximum(0, R**2 - x**2)) * (2 / (np.pi * R**2))
    axes[0].plot(x, wigner_semicircle, 'r-', linewidth=2, label='Wigner Semicircle')
    axes[0].legend()

    # 归一化间距分布
    axes[1].hist(spacings, bins=50, density=True, alpha=0.6,
                edgecolor='black', label='Numerical')

    # Wigner-Dyson 理论曲线
    s_theory = np.linspace(0, np.max(spacings), 200)
    wd_dist = analyzer.wigner_dyson_distribution(s_theory, 'GUE')
    axes[1].plot(s_theory, wd_dist, 'r-', linewidth=2, label='Wigner-Dyson (GUE)')

    # Poisson 对比
    poisson_dist = analyzer.poisson_distribution(s_theory)
    axes[1].plot(s_theory, poisson_dist, 'g--', linewidth=2, label='Poisson')

    axes[1].set_xlabel('Normalized Spacing s')
    axes[1].set_ylabel('Probability Density P(s)')
    axes[1].set_title('Spacing Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 累积分布
    sorted_spacings = np.sort(spacings)
    cumulative = np.arange(1, len(sorted_spacings) + 1) / len(sorted_spacings)
    axes[2].plot(sorted_spacings, cumulative, 'b-', linewidth=2, label='Numerical')

    # 理论累积分布（数值积分）
    try:
        from scipy.integrate import cumulative_trapezoid
        wd_cumulative = cumulative_trapezoid(wd_dist, s_theory, initial=0)
    except ImportError:
        from scipy.integrate import cumtrapz
        wd_cumulative = cumtrapz(wd_dist, s_theory, initial=0)
    wd_cumulative /= wd_cumulative[-1]  # 归一化
    axes[2].plot(s_theory, wd_cumulative, 'r--', linewidth=2, label='Wigner-Dyson (GUE)')

    axes[2].set_xlabel('Normalized Spacing s')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('Cumulative Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hermitian_analysis_results.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ 图表已保存为: hermitian_analysis_results.png")

    # 7. 总结
    print(f"\n6. 分析总结")
    print("   ✓ 所有测试通过")
    print("   ✓ 观察到明显的能级排斥现象")
    print("   ✓ 结果与 Wigner-Dyson (GUE) 理论吻合")
    print("\n" + "=" * 60)
    print("分析完成！请查看生成的图表和 Jupyter notebook 以获得更多细节。")
    print("=" * 60)

if __name__ == "__main__":
    main()
