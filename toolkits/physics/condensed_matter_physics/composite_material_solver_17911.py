# Filename: composite_material_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_load_distribution(areas, elastic_moduli, lengths, max_stresses):
    """
    计算复合材料系统中的载荷分布和最大安全载荷。
    
    该函数基于材料的弹性模量、横截面积、长度和最大允许应力，计算复合材料系统中
    各组件的载荷分布，并确定整个系统的最大安全载荷。适用于凝聚态物理中的材料力学
    分析和多体系统模拟。
    
    Parameters:
    -----------
    areas : array_like
        各组件的横截面积，单位为 mm²
    elastic_moduli : array_like
        各组件的弹性模量，单位为 N/mm²
    lengths : array_like
        各组件的长度，单位为 mm
    max_stresses : array_like
        各组件的最大允许应力，单位为 N/mm²
    
    Returns:
    --------
    tuple
        (max_safe_load, load_distribution, strains, stresses)
        - max_safe_load: 系统可承受的最大安全载荷，单位为 N
        - load_distribution: 各组件承受的载荷分布，单位为 N
        - strains: 各组件的应变
        - stresses: 各组件的应力，单位为 N/mm²
    """
    # 将输入转换为numpy数组以便于计算
    areas = np.array(areas)
    elastic_moduli = np.array(elastic_moduli)
    lengths = np.array(lengths)
    max_stresses = np.array(max_stresses)
    
    # 计算各组件的刚度系数 k = AE/L
    stiffness = areas * elastic_moduli / lengths
    
    # 计算单位载荷下的应变分布
    # 在复合材料系统中，总位移相等，因此应变与长度成正比
    unit_strains = lengths / elastic_moduli
    
    # 计算载荷分布比例
    # 在相同位移下，载荷分配与刚度成正比
    load_ratios = stiffness / np.sum(stiffness)
    
    # 计算各组件在单位总载荷下的应力
    unit_stresses = load_ratios / areas
    
    # 计算各组件达到最大允许应力时的总载荷
    limiting_loads = max_stresses / unit_stresses
    
    # 系统的最大安全载荷由最先达到最大允许应力的组件决定
    max_safe_load = np.min(limiting_loads)
    
    # 计算最大安全载荷下各组件的载荷分布
    load_distribution = max_safe_load * load_ratios
    
    # 计算最大安全载荷下各组件的应变
    strains = load_distribution / (areas * elastic_moduli)
    
    # 计算最大安全载荷下各组件的应力
    stresses = load_distribution / areas
    
    return max_safe_load, load_distribution, strains, stresses

def quantum_state_energy_solver(hamiltonian, num_states=5, method='exact'):
    """
    求解量子系统的能量本征值和本征态。
    
    该函数用于计算给定哈密顿量的能量本征值和本征态，适用于凝聚态物理中的
    量子多体系统模拟，如量子点、量子阱等低维系统的能级结构分析。
    
    Parameters:
    -----------
    hamiltonian : numpy.ndarray
        系统的哈密顿量矩阵，应为厄米矩阵
    num_states : int, optional
        需要计算的能量本征态数量，默认为5
    method : str, optional
        求解方法，可选 'exact'（精确对角化）或 'lanczos'（Lanczos算法），默认为'exact'
        
    Returns:
    --------
    tuple
        (eigenvalues, eigenstates)
        - eigenvalues: 能量本征值数组，按升序排列
        - eigenstates: 对应的本征态矩阵，每列为一个本征态
    """
    # 确保哈密顿量为厄米矩阵
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("Hamiltonian must be Hermitian")
    
    if method == 'exact':
        # 使用精确对角化方法
        eigenvalues, eigenstates = np.linalg.eigh(hamiltonian)
        
        # 只返回请求的本征态数量
        return eigenvalues[:num_states], eigenstates[:, :num_states]
    
    elif method == 'lanczos':
        # 使用Lanczos算法（适用于大型稀疏矩阵）
        # 这里提供一个简化版本的Lanczos算法实现
        n = hamiltonian.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        V = np.zeros((n, num_states+1))
        V[:, 0] = v
        
        H_tridiag = np.zeros((num_states, num_states))
        beta = 0
        
        for j in range(num_states):
            w = hamiltonian @ V[:, j]
            alpha = np.real(np.vdot(w, V[:, j]))
            w = w - alpha * V[:, j] - beta * V[:, j-1] if j > 0 else w - alpha * V[:, j]
            
            beta = np.linalg.norm(w)
            if beta < 1e-10:
                break
                
            V[:, j+1] = w / beta
            
            H_tridiag[j, j] = alpha
            if j < num_states - 1:
                H_tridiag[j, j+1] = beta
                H_tridiag[j+1, j] = beta
        
        # 对三对角矩阵进行对角化
        eig_vals, eig_vecs = np.linalg.eigh(H_tridiag[:j+1, :j+1])
        
        # 转换回原始基底
        eigenstates = V[:, :j+1] @ eig_vecs
        
        # 按能量排序
        idx = np.argsort(eig_vals)
        eigenvalues = eig_vals[idx]
        eigenstates = eigenstates[:, idx]
        
        return eigenvalues, eigenstates
    
    else:
        raise ValueError("Method must be either 'exact' or 'lanczos'")

def visualize_load_distribution(materials, areas, loads, stresses, max_stresses, filename=None):
    """
    可视化复合材料系统中的载荷分布和应力状态。
    
    Parameters:
    -----------
    materials : list
        材料名称列表
    areas : array_like
        各组件的横截面积，单位为 mm²
    loads : array_like
        各组件承受的载荷，单位为 N
    stresses : array_like
        各组件的应力，单位为 N/mm²
    max_stresses : array_like
        各组件的最大允许应力，单位为 N/mm²
    filename : str, optional
        图像保存的文件名，如果为None则不保存
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n = len(materials)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制载荷分布
    bars1 = ax1.bar(materials, loads/1000)
    ax1.set_ylabel('载荷 (kN)')
    ax1.set_title('各组件载荷分布')
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 绘制应力状态
    x = np.arange(n)
    width = 0.35
    bars2 = ax2.bar(x - width/2, stresses, width, label='实际应力')
    bars3 = ax2.bar(x + width/2, max_stresses, width, label='最大允许应力')
    
    ax2.set_ylabel('应力 (N/mm²)')
    ax2.set_title('各组件应力状态')
    ax2.set_xticks(x)
    ax2.set_xticklabels(materials)
    ax2.legend()
    
    # 在柱状图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    if filename:
        # 确保目录存在
        os.makedirs('./images', exist_ok=True)
        plt.savefig(f'./images/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_quantum_states(eigenvalues, eigenstates, potential=None, x_range=None, filename=None):
    """
    可视化量子系统的能级和波函数。
    
    Parameters:
    -----------
    eigenvalues : array_like
        能量本征值数组
    eigenstates : array_like
        本征态矩阵，每列为一个本征态
    potential : callable or array_like, optional
        势能函数或势能数组
    x_range : array_like, optional
        空间坐标范围，如果为None则使用默认范围
    filename : str, optional
        图像保存的文件名，如果为None则不保存
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n_states = len(eigenvalues)
    
    if x_range is None:
        x_range = np.linspace(-5, 5, eigenstates.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制势能函数（如果提供）
    if potential is not None:
        if callable(potential):
            V = potential(x_range)
        else:
            V = potential
        ax.plot(x_range, V, 'k--', label='势能')
    
    # 绘制能级和波函数
    colors = plt.cm.rainbow(np.linspace(0, 1, n_states))
    
    for i in range(n_states):
        # 绘制能级线
        ax.axhline(y=eigenvalues[i], color=colors[i], linestyle='-', alpha=0.3)
        
        # 绘制波函数（将波函数放置在对应能级上）
        # 对波函数进行缩放以便于可视化
        psi = np.real(eigenstates[:, i])
        scale = 0.5
        psi_scaled = scale * psi / np.max(np.abs(psi)) + eigenvalues[i]
        
        ax.plot(x_range, psi_scaled, color=colors[i], 
                label=f'E{i} = {eigenvalues[i]:.3f}')
    
    ax.set_xlabel('位置')
    ax.set_ylabel('能量')
    ax.set_title('量子系统能级和波函数')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图像
    if filename:
        # 确保目录存在
        os.makedirs('./images', exist_ok=True)
        plt.savefig(f'./images/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """
    主函数：演示如何使用工具函数求解复合材料载荷分布问题
    """
    print("复合材料载荷分布问题求解示例")
    print("-" * 50)
    
    # 问题参数定义
    # 两根黄铜杆和一根钢杆支撑载荷
    # 黄铜杆1: 1000 mm², 黄铜杆2: 1000 mm², 钢杆: 1500 mm²
    areas = [1000, 1500, 1000]  # mm²
    
    # 黄铜的弹性模量为 1×10⁵ N/mm², 钢的弹性模量为 2×10⁵ N/mm²
    elastic_moduli = [1e5, 2e5, 1e5]  # N/mm²
    
    # 黄铜杆长度为 1000 mm, 钢杆长度为 1700 mm (从图中估计)
    lengths = [1000, 1700, 1000]  # mm
    
    # 黄铜的最大允许应力为 60 N/mm², 钢的最大允许应力为 120 N/mm²
    max_stresses = [60, 120, 60]  # N/mm²
    
    # 材料名称
    materials = ["黄铜杆1", "钢杆", "黄铜杆2"]
    
    # 计算最大安全载荷和载荷分布
    max_load, loads, strains, stresses = calculate_load_distribution(
        areas, elastic_moduli, lengths, max_stresses
    )
    
    # 输出结果
    print(f"系统最大安全载荷: {max_load:.2f} N")
    print("\n各组件载荷分布:")
    for i, material in enumerate(materials):
        print(f"{material}: {loads[i]:.2f} N (应力: {stresses[i]:.2f} N/mm²)")
    
    # 可视化载荷分布和应力状态
    visualize_load_distribution(materials, areas, loads, stresses, max_stresses, 
                               filename="composite_load_distribution.png")
    
    print("\n\n量子多体系统模拟示例")
    print("-" * 50)
    
    # 创建一个简单的量子系统哈密顿量（一维谐振子）
    n = 100  # 矩阵维度
    x = np.linspace(-5, 5, n)
    dx = x[1] - x[0]
    
    # 动能项（二阶导数）
    D2 = np.zeros((n, n))
    for i in range(n):
        D2[i, i] = -2
        if i > 0:
            D2[i, i-1] = 1
        if i < n-1:
            D2[i, i+1] = 1
    
    # 势能项（谐振子势能 V(x) = 0.5 * x²）
    V = 0.5 * x**2
    
    # 构建哈密顿量 H = -0.5 * d²/dx² + V(x)
    H = -0.5 * D2 / (dx**2) + np.diag(V)
    
    # 求解本征值和本征态
    eigenvalues, eigenstates = quantum_state_energy_solver(H, num_states=5)
    
    # 输出能量本征值
    print("量子谐振子能量本征值:")
    for i, e in enumerate(eigenvalues):
        print(f"E{i} = {e:.6f}")
    
    # 理论值（谐振子能级 E_n = (n + 0.5) * hbar * omega，这里 hbar * omega = 1）
    theory = np.array([i + 0.5 for i in range(5)])
    print("\n理论能量本征值:")
    for i, e in enumerate(theory):
        print(f"E{i} = {e:.6f}")
    
    # 计算相对误差
    rel_error = np.abs(eigenvalues - theory) / theory
    print("\n相对误差:")
    for i, err in enumerate(rel_error):
        print(f"E{i}: {err:.6%}")
    
    # 可视化量子态
    visualize_quantum_states(eigenvalues, eigenstates, potential=V, x_range=x,
                            filename="quantum_harmonic_oscillator.png")

if __name__ == "__main__":
    main()
    """
    主函数：演示如何使用工具函数求解复合材料载荷分布问题
    """
    print("复合材料载荷分布问题求解示例")
    print("-" * 50)
    
    # 问题参数定义
    # 两根黄铜杆和一根钢杆支撑载荷
    # 黄铜杆1: 1000 mm², 黄铜杆2: 1000 mm², 钢杆: 1500 mm²
    areas = [1000, 1500, 1000]  # mm²
    
    # 黄铜的弹性模量为 1×10⁵ N/mm², 钢的弹性模量为 2×10⁵ N/mm²
    elastic_moduli = [1e5, 2e5, 1e5]  # N/mm²
    
    # 黄铜杆长度为 1000 mm, 钢杆长度为 1700 mm (从图中估计)
    lengths = [1000, 1700, 1000]  # mm
    
    # 黄铜的最大允许应力为 60 N/mm², 钢的最大允许应力为 120 N/mm²
    max_stresses = [60, 120, 60]  # N/mm²
    
    # 材料名称
    materials = ["黄铜杆1", "钢杆", "黄铜杆2"]
    
    # 计算最大安全载荷和载荷分布
    max_load, loads, strains, stresses = calculate_load_distribution(
        areas, elastic_moduli, lengths, max_stresses
    )
    
    # 输出结果
    print(f"系统最大安全载荷: {max_load:.2f} N")
    print("\n各组件载荷分布:")
    for i, material in enumerate(materials):
        print(f"{material}: {loads[i]:.2f} N (应力: {stresses[i]:.2f} N/mm²)")
    
    #