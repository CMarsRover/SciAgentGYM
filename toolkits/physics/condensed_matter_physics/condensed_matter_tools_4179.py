# Filename: condensed_matter_tools.py

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os

# 确保图片保存目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def construct_hamiltonian(system_size, interaction_matrix, potential=None, periodic=False):
    """
    构建量子多体系统的哈密顿量矩阵（稀疏格式）
    
    该函数用于构建一维或多维量子系统的哈密顿量，支持自定义相互作用和势能项。
    基于紧束缚模型(tight-binding model)的理论框架。
    
    Parameters:
    -----------
    system_size : tuple of int
        系统尺寸，如(Nx, Ny)表示二维晶格，(N,)表示一维链
    interaction_matrix : ndarray
        相互作用矩阵，描述粒子间的相互作用强度
    potential : ndarray or callable, optional
        外部势能场，可以是数组或函数
    periodic : bool, optional
        是否使用周期性边界条件，默认为False
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        系统哈密顿量的稀疏矩阵表示
    """
    # 统一 system_size 表示为元组，便于处理一维/二维
    if isinstance(system_size, tuple):
        size_tuple = system_size
    else:
        size_tuple = (int(system_size),)

    # 计算系统总格点数
    total_sites = int(np.prod(size_tuple))
    
    # 初始化稀疏矩阵
    H = sp.lil_matrix((total_sites, total_sites), dtype=np.float64)
    
    # 构建相互作用项
    if len(size_tuple) == 1:  # 一维系统 (N,)
        for i in range(total_sites):
            # 对角项（on-site能量）
            if potential is not None:
                if callable(potential):
                    H[i, i] = potential(i)
                else:
                    H[i, i] = potential[i]
            
            # 最近邻相互作用
            if i < total_sites - 1:  # 右邻居
                hop = interaction_matrix[0] if hasattr(interaction_matrix, '__len__') else interaction_matrix
                H[i, i+1] = hop
                H[i+1, i] = hop  # 厄米共轭
            
            # 周期性边界条件
            if periodic and i == total_sites - 1:
                hop = interaction_matrix[0] if hasattr(interaction_matrix, '__len__') else interaction_matrix
                H[i, 0] = hop
                H[0, i] = hop
    
    elif len(size_tuple) == 2:  # 二维系统 (Nx, Ny)
        nx, ny = size_tuple
        for ix in range(nx):
            for iy in range(ny):
                i = ix * ny + iy  # 线性索引
                
                # 对角项
                if potential is not None:
                    if callable(potential):
                        H[i, i] = potential(ix, iy)
                    else:
                        H[i, i] = potential[ix, iy]
                
                # x方向相互作用
                if ix < nx - 1:  # 右邻居
                    j = (ix + 1) * ny + iy
                    H[i, j] = interaction_matrix[0, 0]
                    H[j, i] = interaction_matrix[0, 0]
                elif periodic:  # 周期性边界
                    j = 0 * ny + iy
                    H[i, j] = interaction_matrix[0, 0]
                    H[j, i] = interaction_matrix[0, 0]
                
                # y方向相互作用
                if iy < ny - 1:  # 上邻居
                    j = ix * ny + (iy + 1)
                    H[i, j] = interaction_matrix[0, 1]
                    H[j, i] = interaction_matrix[0, 1]
                elif periodic:  # 周期性边界
                    j = ix * ny + 0
                    H[i, j] = interaction_matrix[0, 1]
                    H[j, i] = interaction_matrix[0, 1]
    
    return H.tocsr()  # 转换为CSR格式以提高计算效率

def solve_eigensystem(hamiltonian, k=6, which='SA'):
    """
    求解哈密顿量的本征值和本征态
    
    使用稀疏矩阵方法高效求解大型哈密顿量的低能本征态，
    适用于量子多体系统的基态和激发态计算。
    
    Parameters:
    -----------
    hamiltonian : scipy.sparse.csr_matrix
        系统的哈密顿量（稀疏矩阵格式）
    k : int, optional
        需要计算的本征值数量，默认为6
    which : str, optional
        指定计算哪些本征值：
        'SA' - 代数值最小的k个本征值（默认）
        'LA' - 代数值最大的k个本征值
        'SM' - 绝对值最小的k个本征值
        'LM' - 绝对值最大的k个本征值
        
    Returns:
    --------
    eigenvalues : ndarray
        计算得到的本征值数组
    eigenvectors : ndarray
        对应的本征向量数组，每列是一个本征向量
    """
    # 检查哈密顿量是否为厄米矩阵
    if not np.allclose((hamiltonian - hamiltonian.getH()).data, 0):
        print("警告：哈密顿量不是厄米矩阵，结果可能不物理")
    
    # 使用Arnoldi迭代法求解稀疏矩阵本征问题
    eigenvalues, eigenvectors = spla.eigsh(hamiltonian, k=k, which=which)
    
    # 按能量排序
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

def calculate_density_matrix(eigenvector, trace_subsystem=None):
    """
    计算量子态的密度矩阵或约化密度矩阵
    
    可用于计算纯态的密度矩阵，或通过部分迹运算得到子系统的约化密度矩阵，
    是量子纠缠和量子信息研究的基础工具。
    
    Parameters:
    -----------
    eigenvector : ndarray
        量子态的向量表示
    trace_subsystem : tuple, optional
        需要追踪的子系统维度，如果为None则计算完整密度矩阵
        
    Returns:
    --------
    density_matrix : ndarray
        密度矩阵或约化密度矩阵
    """
    # 确保eigenvector是列向量
    if eigenvector.ndim == 1:
        eigenvector = eigenvector.reshape(-1, 1)
    
    # 计算完整密度矩阵 ρ = |ψ⟩⟨ψ|
    full_dm = np.outer(eigenvector, eigenvector.conj())
    
    # 如果不需要部分迹，直接返回完整密度矩阵
    if trace_subsystem is None:
        return full_dm
    
    # 计算约化密度矩阵（通过部分迹）
    # 这里实现一个简化版本，假设系统可以分为两个子系统A和B
    dim_a, dim_b = trace_subsystem
    reduced_dm = np.zeros((dim_a, dim_a), dtype=complex)
    
    # 执行部分迹运算
    for i in range(dim_b):
        # 提取对应于B系统第i个基矢的子块
        block = full_dm[i::dim_b, i::dim_b]
        reduced_dm += block
    
    return reduced_dm

def monte_carlo_ising(lattice_size, temperature, num_steps, J=1.0):
    """
    使用Metropolis算法进行二维Ising模型的蒙特卡洛模拟
    
    实现经典的Metropolis算法模拟二维Ising模型的热力学行为，
    可用于研究相变、临界现象和标度行为。
    
    Parameters:
    -----------
    lattice_size : tuple of int
        晶格尺寸，如(L, L)表示LxL的二维方格
    temperature : float
        系统温度，单位为J/kB
    num_steps : int
        蒙特卡洛步数
    J : float, optional
        自旋间相互作用强度，J>0表示铁磁性，默认为1.0
        
    Returns:
    --------
    dict
        包含模拟结果的字典，包括：
        - 'lattice': 最终晶格构型
        - 'energy': 能量时间序列
        - 'magnetization': 磁化强度时间序列
        - 'specific_heat': 比热
        - 'susceptibility': 磁化率
    """
    # 初始化随机晶格构型
    lattice = np.random.choice([-1, 1], size=lattice_size)
    
    # 预分配结果数组
    energy_samples = np.zeros(num_steps)
    mag_samples = np.zeros(num_steps)
    
    # 计算初始能量
    energy = 0.0
    for i in range(lattice_size[0]):
        for j in range(lattice_size[1]):
            spin = lattice[i, j]
            # 考虑最近邻相互作用（周期性边界条件）
            neighbors = (
                lattice[(i+1)%lattice_size[0], j] + 
                lattice[i, (j+1)%lattice_size[1]] + 
                lattice[(i-1)%lattice_size[0], j] + 
                lattice[i, (j-1)%lattice_size[1]]
            )
            energy += -J * spin * neighbors
    # 避免重复计算，除以2
    energy /= 2.0
    
    # 计算初始磁化强度
    magnetization = np.sum(lattice)
    
    # Metropolis算法主循环
    for step in range(num_steps):
        # 随机选择一个格点
        i, j = np.random.randint(0, lattice_size[0]), np.random.randint(0, lattice_size[1])
        spin = lattice[i, j]
        
        # 计算翻转自旋导致的能量变化
        neighbors = (
            lattice[(i+1)%lattice_size[0], j] + 
            lattice[i, (j+1)%lattice_size[1]] + 
            lattice[(i-1)%lattice_size[0], j] + 
            lattice[i, (j-1)%lattice_size[1]]
        )
        delta_E = 2 * J * spin * neighbors
        
        # Metropolis接受/拒绝准则
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
            # 接受翻转
            lattice[i, j] = -spin
            energy += delta_E
            magnetization -= 2 * spin
        
        # 记录样本
        energy_samples[step] = energy
        mag_samples[step] = magnetization
    
    # 计算热力学量
    # 丢弃前20%的样本作为热化阶段
    discard = int(0.2 * num_steps)
    energy_samples = energy_samples[discard:]
    mag_samples = mag_samples[discard:]
    
    # 计算平均能量、磁化强度、比热和磁化率
    N = np.prod(lattice_size)  # 总格点数
    avg_energy = np.mean(energy_samples) / N
    avg_mag = np.mean(np.abs(mag_samples)) / N  # 使用绝对值避免不同域的抵消
    
    # 比热: C = (⟨E²⟩ - ⟨E⟩²) / (kB*T²*N)
    specific_heat = (np.mean(energy_samples**2) - np.mean(energy_samples)**2) / (temperature**2 * N)
    
    # 磁化率: χ = (⟨M²⟩ - ⟨M⟩²) / (kB*T*N)
    susceptibility = (np.mean(mag_samples**2) - np.mean(mag_samples)**2) / (temperature * N)
    
    return {
        'lattice': lattice,
        'energy': energy_samples,
        'magnetization': mag_samples,
        'avg_energy': avg_energy,
        'avg_magnetization': avg_mag,
        'specific_heat': specific_heat,
        'susceptibility': susceptibility
    }

def plot_phase_diagram(temp_range, results, observable='specific_heat', title=None):
    """
    绘制相变图，展示物理量随温度的变化
    
    可视化相变过程中物理量的变化，帮助识别临界点和相变行为。
    
    Parameters:
    -----------
    temp_range : array_like
        温度范围数组
    results : list of dict
        每个温度点的模拟结果字典列表
    observable : str, optional
        要绘制的物理量，可选值：'specific_heat', 'susceptibility', 
        'avg_energy', 'avg_magnetization'，默认为'specific_heat'
    title : str, optional
        图表标题，如果为None则自动生成
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    # 提取要绘制的物理量
    y_values = [result[observable] for result in results]
    
    # 设置图表标题
    if title is None:
        observable_names = {
            'specific_heat': '比热',
            'susceptibility': '磁化率',
            'avg_energy': '平均能量',
            'avg_magnetization': '平均磁化强度'
        }
        title = f"Ising模型 - {observable_names.get(observable, observable)}随温度的变化"
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, y_values, 'o-', linewidth=2)
    ax.set_xlabel('温度 (J/k_B)', fontsize=12)
    ax.set_ylabel(observable_names.get(observable, observable), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    filename = f"./images/phase_diagram_{observable}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {filename}")
    
    return fig

def visualize_lattice(lattice, title=None):
    """
    可视化二维晶格构型
    
    用于展示Ising模型的自旋构型，帮助直观理解系统状态。
    
    Parameters:
    -----------
    lattice : ndarray
        二维晶格数组，值为+1或-1
    title : str, optional
        图表标题，如果为None则使用默认标题
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用imshow显示晶格，+1为白色，-1为黑色
    cmap = plt.cm.binary
    im = ax.imshow(lattice, cmap=cmap, vmin=-1, vmax=1)
    
    # 设置标题
    if title is None:
        title = "Ising模型晶格构型"
    ax.set_title(title, fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['自旋向下', '自旋向上'])
    
    # 保存图表
    filename = "./images/ising_lattice.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"晶格图已保存至: {filename}")
    
    return fig

def calculate_correlation_function(lattice, max_distance=None):
    """
    计算二维Ising模型的自旋关联函数
    
    关联函数描述了不同距离自旋之间的相关性，是研究临界现象的重要工具。
    
    Parameters:
    -----------
    lattice : ndarray
        二维晶格数组，值为+1或-1
    max_distance : int, optional
        计算关联函数的最大距离，默认为晶格尺寸的一半
        
    Returns:
    --------
    distances : ndarray
        距离数组
    correlation : ndarray
        对应距离的关联函数值
    """
    L = lattice.shape[0]
    if max_distance is None:
        max_distance = L // 2
    
    # 初始化结果数组
    distances = np.arange(0, max_distance + 1)
    correlation = np.zeros(max_distance + 1)
    
    # 计算平均磁化强度
    avg_spin = np.mean(lattice)
    
    # 计算关联函数
    # C(r) = <s_i * s_j> - <s_i><s_j>，其中r是距离
    for r in distances:
        # 统计所有间隔为r的自旋对
        count = 0
        corr_sum = 0
        
        for i in range(L):
            for j in range(L):
                # 水平方向
                if j + r < L:
                    corr_sum += lattice[i, j] * lattice[i, j+r]
                    count += 1
                
                # 垂直方向
                if i + r < L:
                    corr_sum += lattice[i, j] * lattice[i+r, j]
                    count += 1
        
        if count > 0:
            correlation[r] = corr_sum / count - avg_spin**2
    
    return distances, correlation

def main():
    """
    主函数：演示如何使用工具函数求解凝聚态物理中的关键问题
    """
    print("凝聚态物理计算工具演示")
    print("=" * 50)
    
    # 示例1：量子多体系统模拟
    print("\n1. 量子多体系统模拟 - 一维紧束缚模型")
    # 系统参数
    system_size = 20  # 一维链长度
    hopping = -1.0    # 跃迁积分
    
    # 构建相互作用矩阵
    interaction = np.array([hopping])
    
    # 构建哈密顿量
    H = construct_hamiltonian(system_size, interaction, periodic=True)
    print(f"哈密顿量构建完成，维度: {H.shape}")
    
    # 求解本征值和本征态
    eigenvalues, eigenvectors = solve_eigensystem(H, k=6)
    print("最低的6个能级:")
    for i, e in enumerate(eigenvalues):
        print(f"  E{i} = {e:.6f}")
    
    # 计算基态密度矩阵
    ground_state = eigenvectors[:, 0]
    density_matrix = calculate_density_matrix(ground_state)
    print(f"基态密度矩阵迹: {np.trace(density_matrix):.6f}")
    
    # 可视化基态波函数
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(system_size), np.abs(ground_state)**2, 'o-')
    plt.xlabel('格点位置')
    plt.ylabel('概率密度')
    plt.title('一维紧束缚模型基态波函数')
    plt.grid(True)
    plt.savefig('./images/ground_state_wavefunction.png', dpi=300)
    print("基态波函数图已保存至: ./images/ground_state_wavefunction.png")
    
    # 示例2：相变与临界现象模拟
    print("\n2. 相变与临界现象模拟 - 二维Ising模型")
    
    # 系统参数
    lattice_size = (20, 20)
    temperatures = np.linspace(1.5, 3.5, 5)  # 温度范围，包含临界温度Tc≈2.269
    mc_steps = 10000  # 蒙特卡洛步数
    
    # 存储不同温度的结果
    results = []
    
    # 对每个温度进行模拟
    for temp in temperatures:
        print(f"  模拟温度 T = {temp:.2f}...")
        result = monte_carlo_ising(lattice_size, temp, mc_steps)
        results.append(result)
        
        # 如果是接近临界温度的模拟，可视化晶格构型
        if abs(temp - 2.27) < 0.1:
            visualize_lattice(result['lattice'], 
                             f"Ising模型晶格构型 (T ≈ Tc ≈ {temp:.2f})")
    
    # 绘制相变图
    plot_phase_diagram(temperatures, results, 'specific_heat')
    plot_phase_diagram(temperatures, results, 'avg_magnetization')
    
    # 计算并绘制关联函数（在临界温度附近）
    critical_idx = np.argmin(np.abs(temperatures - 2.27))
    critical_lattice = results[critical_idx]['lattice']
    distances, correlation = calculate_correlation_function(critical_lattice)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(distances, np.abs(correlation), 'o-')
    plt.xlabel('距离 r')
    plt.ylabel('关联函数 |C(r)|')
    plt.title(f'Ising模型关联函数 (T ≈ Tc ≈ {temperatures[critical_idx]:.2f})')
    plt.grid(True)
    plt.savefig('./images/correlation_function.png', dpi=300)
    print("关联函数图已保存至: ./images/correlation_function.png")
    
    print("\n计算完成！所有结果已保存到images目录")

if __name__ == "__main__":
    main()
   