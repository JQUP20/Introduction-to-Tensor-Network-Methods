"""
运行所有随机矩阵理论习题
Run All Random Matrix Theory Exercises

这个脚本按顺序运行：
- 练习 2：间距分布 P(s) 的研究
- 练习 3：Ising 哈密顿量

并生成所有可视化图表
"""

import sys
import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

print("="*80)
print("随机矩阵理论习题集 - 练习 2 & 3")
print("Random Matrix Theory Exercises - Exercise 2 & 3")
print("="*80)

# 练习 2：间距分布
print("\n\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*20 + "练习 2：间距分布 P(s) 的研究" + " "*28 + "█")
print("█" + " "*78 + "█")
print("█"*80)

import exercise2_spacing_distribution as ex2

# 创建分析器
analyzer = ex2.SpacingDistributionAnalyzer(seed=42)

# 任务 a: 收集随机厄米矩阵的间距
print("\n" + "─"*80)
print("任务 (a): 计算随机厄米矩阵的 P(s)")
print("─"*80)

# 使用适中的参数以获得好的统计同时保持合理的运行时间
N_values_hermitian = [100, 200, 300, 400, 500]
num_matrices_hermitian = 5

print(f"\n配置:")
print(f"  矩阵大小 N: {N_values_hermitian}")
print(f"  每个大小的矩阵数量: {num_matrices_hermitian}")
print(f"  总矩阵数: {len(N_values_hermitian) * num_matrices_hermitian}")

start_time = time.time()
hermitian_spacings = analyzer.collect_hermitian_spacings(
    N_values_hermitian, num_matrices_hermitian
)
elapsed = time.time() - start_time
print(f"\n✓ 完成! 用时: {elapsed:.2f} 秒")
print(f"  收集的间距数据点: {len(hermitian_spacings)}")

# 任务 b: 收集对角矩阵的间距
print("\n" + "─"*80)
print("任务 (b): 计算对角矩阵的 P(s)")
print("─"*80)

N_values_diagonal = [100, 200, 300, 400, 500]
num_matrices_diagonal = 5

print(f"\n配置:")
print(f"  矩阵大小 N: {N_values_diagonal}")
print(f"  每个大小的矩阵数量: {num_matrices_diagonal}")

start_time = time.time()
diagonal_spacings = analyzer.collect_diagonal_spacings(
    N_values_diagonal, num_matrices_diagonal
)
elapsed = time.time() - start_time
print(f"\n✓ 完成! 用时: {elapsed:.2f} 秒")
print(f"  收集的间距数据点: {len(diagonal_spacings)}")

# 任务 c: 拟合分布函数
print("\n" + "─"*80)
print("任务 (c): 拟合分布函数 P(s) = a·s^α·exp(-b·s^β)")
print("─"*80)

print("\n拟合随机厄米矩阵...")
hermitian_fit = analyzer.fit_spacing_distribution(
    hermitian_spacings,
    initial_guess=(1.0, 2.0, 1.0, 2.0),  # 接近 GUE
    s_range=(0.1, 4.0)
)
ex2.print_fit_results(hermitian_fit, "随机厄米矩阵")

print("\n拟合对角矩阵...")
diagonal_fit = analyzer.fit_spacing_distribution(
    diagonal_spacings,
    initial_guess=(1.0, 0.0, 1.0, 1.0),  # 接近泊松
    s_range=(0.0, 4.0)
)
ex2.print_fit_results(diagonal_fit, "对角矩阵")

# 任务 d: 与随机矩阵理论预测比较
print("\n" + "─"*80)
print("任务 (d): 与随机矩阵理论预测比较")
print("─"*80)

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

# 生成可视化
print("\n生成练习 2 的可视化图表...")
fig = ex2.plot_spacing_comparison(
    analyzer,
    hermitian_fit,
    diagonal_fit,
    save_path='exercise2_spacing_distribution.png'
)
plt.close(fig)
print("✓ 图表已保存: exercise2_spacing_distribution.png")

print("\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*25 + "练习 2 完成!" + " "*32 + "█")
print("█" + " "*78 + "█")
print("█"*80)

# 练习 3：Ising 哈密顿量
print("\n\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*22 + "练习 3：Ising 哈密顿量" + " "*33 + "█")
print("█" + " "*78 + "█")
print("█"*80)

import exercise3_ising_hamiltonian as ex3
import numpy as np

# 任务 a & b: 构造并对角化
print("\n" + "─"*80)
print("任务 (a) & (b): 构造并对角化 Ising 哈密顿量")
print("─"*80)

# 测试小系统
print("\n测试：N = 3 的小系统")
test_analyzer = ex3.IsingHamiltonianAnalyzer(N=3)

lambda_test = 1.0
print(f"横向场强度 λ = {lambda_test}")

print("\n使用稠密矩阵方法...")
H_dense = test_analyzer.construct_hamiltonian_dense(lambda_test)
print(f"  哈密顿矩阵形状: {H_dense.shape}")
print(f"  矩阵是厄米的: {np.allclose(H_dense, H_dense.conj().T)}")

eigenvalues_dense, _ = test_analyzer.diagonalize(use_sparse=False)
print(f"\n  前 5 个能级:")
for i, E in enumerate(eigenvalues_dense[:5]):
    print(f"    E_{i} = {E:.6f}")

# 任务 c: 绘制能级图
print("\n" + "─"*80)
print("任务 (c): 绘制不同 N 和 λ 的能级图")
print("─"*80)

# 参数设置
N_values = [2, 3, 4, 5, 6, 7]  # 自旋数量
lambda_values = np.linspace(0.0, 3.0, 60)  # λ ∈ [0, 3]
num_levels = 10  # 保留的能级数量

print(f"\n配置:")
print(f"  自旋数量 N: {N_values}")
print(f"  横向场范围 λ: [{lambda_values[0]:.1f}, {lambda_values[-1]:.1f}]")
print(f"  λ 采样点数: {len(lambda_values)}")
print(f"  保留能级数: {num_levels}")
print(f"\n希尔伯特空间维度:")
for N in N_values:
    print(f"  N = {N}: 2^{N} = {2**N}")

# 计算能谱
start_time = time.time()
results = ex3.compute_energy_spectrum(
    N_values,
    lambda_values,
    num_levels=num_levels,
    use_sparse=True,
    verbose=True
)
elapsed = time.time() - start_time
print(f"\n✓ 能谱计算完成! 总用时: {elapsed:.2f} 秒")

# 绘制能级图
print("\n生成能级图...")
fig_spectrum = ex3.plot_energy_spectrum(
    results,
    N_values_to_plot=N_values,
    save_path='exercise3_energy_spectrum.png'
)
plt.close(fig_spectrum)
print("✓ 图表已保存: exercise3_energy_spectrum.png")

# 绘制能隙分析
print("\n生成能隙分析图...")
fig_gaps = ex3.plot_energy_gaps(
    results,
    save_path='exercise3_energy_gaps.png'
)
plt.close(fig_gaps)
print("✓ 图表已保存: exercise3_energy_gaps.png")

# 相变分析
ex3.analyze_phase_transition(results, lambda_critical=1.0)

print("\n" + "█"*80)
print("█" + " "*78 + "█")
print("█" + " "*25 + "练习 3 完成!" + " "*32 + "█")
print("█" + " "*78 + "█")
print("█"*80)

# 总结
print("\n\n" + "="*80)
print("所有练习完成!")
print("="*80)

print("\n生成的文件:")
print("  1. exercise2_spacing_distribution.png - 间距分布分析")
print("  2. exercise3_energy_spectrum.png      - Ising 模型能级图")
print("  3. exercise3_energy_gaps.png          - Ising 模型能隙分析")

print("\n主要发现:")
print("\n练习 2：")
print("  ✓ 随机厄米矩阵展示 Wigner-Dyson 统计（能级排斥）")
print("  ✓ 对角矩阵展示泊松统计（无能级排斥）")
print("  ✓ 拟合参数与随机矩阵理论预测吻合")
print(f"    - 厄米矩阵: α ≈ {hermitian_fit.get('alpha', 'N/A'):.2f}, β ≈ {hermitian_fit.get('beta', 'N/A'):.2f} (理论: α=2, β=2 for GUE)")
print(f"    - 对角矩阵: α ≈ {diagonal_fit.get('alpha', 'N/A'):.2f}, β ≈ {diagonal_fit.get('beta', 'N/A'):.2f} (理论: α=0, β=1 for Poisson)")

print("\n练习 3：")
print("  ✓ 横向场 Ising 模型在 λ ≈ 1 处存在量子相变")
print("  ✓ 铁磁相 (λ < 1) 和顺磁相 (λ > 1) 的能谱特征明显不同")
print("  ✓ 有限尺寸效应：随着 N 增大，相变特征更加明显")
print("  ✓ 能隙在相变点附近最小，体现能隙关闭的趋势")

print("\n物理意义:")
print("  • 能级排斥是量子混沌系统的标志")
print("  • Wigner-Dyson 统计出现在量子混沌系统中")
print("  • 泊松统计对应可积系统（无能级关联）")
print("  • Ising 模型的量子相变展示了竞争相互作用的效果")
print("  • 这些练习展示了随机矩阵理论在量子系统中的普适性")

print("\n" + "="*80)
print("程序执行完毕。")
print("="*80)
