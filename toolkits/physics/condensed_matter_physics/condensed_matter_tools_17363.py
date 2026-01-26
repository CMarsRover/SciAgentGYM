# Filename: condensed_matter_tools.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import os

# 确保图像目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_bond_angle(v1, v2):
    """
    计算两个键向量之间的夹角
    
    Parameters:
    -----------
    v1 : numpy.ndarray
        第一个键向量
    v2 : numpy.ndarray
        第二个键向量
    
    Returns:
    --------
    float
        两个键向量之间的夹角（以度为单位）
    """
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_v1 * norm_v2)
    
    # 由于浮点数精度问题，确保cos_angle在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 计算夹角（以度为单位）
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def generate_diamond_lattice(size=2):
    """
    生成金刚石晶格结构
    
    Parameters:
    -----------
    size : int, optional
        晶格的大小（单位晶胞的数量），默认为2
    
    Returns:
    --------
    tuple
        (positions, bonds) 其中positions是原子位置的数组，bonds是键连接的索引对列表
    """
    # 金刚石晶格的基本单位晶胞
    a = 1.0  # 晶格常数
    
    # FCC晶格点
    fcc_points = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # 金刚石晶格点（FCC + 基础移位）
    diamond_basis = np.array([
        [0, 0, 0],
        [0.25, 0.25, 0.25]
    ]) * a
    
    # 生成完整的金刚石晶格
    positions = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k]) * a
                for fcc_point in fcc_points:
                    for basis_point in diamond_basis:
                        position = fcc_point + basis_point + offset
                        positions.append(position)
    
    positions = np.array(positions)
    
    # 生成键连接
    bonds = []
    # 键长阈值（金刚石的键长约为a*sqrt(3)/4）
    bond_threshold = a * np.sqrt(3) / 4 * 1.1
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < bond_threshold:
                bonds.append((i, j))
    
    return positions, bonds

def calculate_diamond_bond_angles():
    """
    计算金刚石晶格中键之间的夹角
    
    Returns:
    --------
    float
        金刚石晶格中任意两个键之间的夹角（以度为单位）
    """
    # 金刚石晶格中心原子到四个最近邻的向量
    # 这些向量指向正四面体的四个顶点
    vectors = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)
    
    # 计算任意两个键之间的夹角
    angles = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            angle = calculate_bond_angle(vectors[i], vectors[j])
            angles.append(angle)
    
    # 所有夹角应该相同，返回平均值
    return np.mean(angles)

def visualize_diamond_lattice(size=2, show=True, save_path=None):
    """
    可视化金刚石晶格结构
    
    Parameters:
    -----------
    size : int, optional
        晶格的大小，默认为2
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径，如果为None则不保存
    
    Returns:
    --------
    None
    """
    positions, bonds = generate_diamond_lattice(size)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原子
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               color='gray', s=100, alpha=0.7)
    
    # 绘制键
    for bond in bonds:
        i, j = bond
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                [positions[i, 2], positions[j, 2]], 'k-', lw=1)
    
    # 突出显示一个原子及其四个最近邻
    # 选择一个位于晶格中心附近的原子
    center_idx = np.argmin(np.sum((positions - np.mean(positions, axis=0))**2, axis=1))
    ax.scatter(positions[center_idx, 0], positions[center_idx, 1], positions[center_idx, 2], 
               color='red', s=150)
    
    # 找到与中心原子相连的原子
    neighbor_indices = [j if i == center_idx else i for i, j in bonds if i == center_idx or j == center_idx]
    
    # 突出显示这些原子和键
    for idx in neighbor_indices:
        ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], 
                   color='blue', s=120)
        ax.plot([positions[center_idx, 0], positions[idx, 0]],
                [positions[center_idx, 1], positions[idx, 1]],
                [positions[center_idx, 2], positions[idx, 2]], 'r-', lw=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('金刚石晶格结构')
    
    # 设置坐标轴范围
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_zlim([0, size])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def simulate_quantum_system(hamiltonian, num_states=5):
    """
    模拟量子多体系统，求解哈密顿量的本征值和本征态
    
    Parameters:
    -----------
    hamiltonian : numpy.ndarray
        系统的哈密顿量矩阵
    num_states : int, optional
        需要计算的能级数量，默认为5
    
    Returns:
    --------
    tuple
        (eigenvalues, eigenvectors) 本征值和对应的本征态
    """
    # 确保哈密顿量是厄米矩阵
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("哈密顿量必须是厄米矩阵")
    
    # 求解本征值问题
    eigenvalues, eigenvectors = la.eigh(hamiltonian)
    
    # 返回最低的num_states个能级和对应的本征态
    return eigenvalues[:num_states], eigenvectors[:, :num_states]

def plot_energy_spectrum(eigenvalues, labels=None, show=True, save_path=None):
    """
    绘制能谱图
    
    Parameters:
    -----------
    eigenvalues : numpy.ndarray
        能级值数组
    labels : list, optional
        能级标签
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径，如果为None则不保存
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制能级
    for i, energy in enumerate(eigenvalues):
        plt.plot([0, 1], [energy, energy], 'b-', linewidth=2)
        
        # 添加标签
        if labels and i < len(labels):
            plt.text(1.1, energy, labels[i], fontsize=12)
        else:
            plt.text(1.1, energy, f"E{i}", fontsize=12)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(min(eigenvalues) - 0.5, max(eigenvalues) + 0.5)
    plt.ylabel('能量')
    plt.title('量子系统能谱')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def monte_carlo_ising_step(lattice, beta, J=1.0):
    """
    执行一步蒙特卡洛模拟（Metropolis算法）用于伊辛模型
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        表示伊辛模型的晶格，元素为+1或-1
    beta : float
        逆温度参数 β = 1/(k_B*T)
    J : float, optional
        自旋间相互作用强度，默认为1.0（铁磁性）
    
    Returns:
    --------
    numpy.ndarray
        更新后的晶格
    """
    L = lattice.shape[0]
    
    # 随机选择一个格点
    i, j = np.random.randint(0, L, size=2)
    
    # 计算翻转自旋前后的能量变化
    s = lattice[i, j]
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
    delta_E = 2 * J * s * neighbors
    
    # Metropolis准则
    if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
        lattice[i, j] = -s
    
    return lattice

def simulate_ising_model(L=50, T=2.27, steps=10000, equilibration=1000, J=1.0):
    """
    模拟二维伊辛模型的相变
    
    Parameters:
    -----------
    L : int, optional
        晶格大小，默认为50
    T : float, optional
        温度（单位为J/k_B），默认为2.27（接近临界温度）
    steps : int, optional
        模拟步数，默认为10000
    equilibration : int, optional
        平衡步数，默认为1000
    J : float, optional
        自旋间相互作用强度，默认为1.0
    
    Returns:
    --------
    tuple
        (lattice, magnetizations, energies) 最终晶格状态、磁化强度历史和能量历史
    """
    # 初始化随机晶格
    lattice = np.random.choice([-1, 1], size=(L, L))
    
    # 计算逆温度
    beta = 1.0 / T
    
    # 记录磁化强度和能量
    magnetizations = []
    energies = []
    
    # 模拟主循环
    for step in range(steps + equilibration):
        # 执行L^2次自旋翻转尝试（一个蒙特卡洛步）
        for _ in range(L*L):
            lattice = monte_carlo_ising_step(lattice, beta, J)
        
        # 平衡后开始记录
        if step >= equilibration:
            # 计算磁化强度
            m = np.abs(np.mean(lattice))
            magnetizations.append(m)
            
            # 计算能量
            e = 0
            for i in range(L):
                for j in range(L):
                    e -= J * lattice[i, j] * (lattice[(i+1)%L, j] + lattice[i, (j+1)%L])
            energies.append(e / (L*L))
    
    return lattice, np.array(magnetizations), np.array(energies)

def plot_ising_results(lattice, magnetizations, energies, T, show=True, save_path=None):
    """
    绘制伊辛模型模拟结果
    
    Parameters:
    -----------
    lattice : numpy.ndarray
        最终晶格状态
    magnetizations : numpy.ndarray
        磁化强度历史
    energies : numpy.ndarray
        能量历史
    T : float
        模拟温度
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径前缀，如果为None则不保存
    
    Returns:
    --------
    None
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制晶格状态
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(lattice, cmap='coolwarm')
    ax1.set_title(f'伊辛模型晶格 (T={T:.2f})')
    plt.colorbar(im, ax=ax1, label='自旋')
    
    # 绘制磁化强度历史
    ax2 = fig.add_subplot(132)
    ax2.plot(magnetizations, 'b-')
    ax2.set_xlabel('蒙特卡洛步')
    ax2.set_ylabel('磁化强度 |m|')
    ax2.set_title('磁化强度演化')
    ax2.grid(True)
    
    # 绘制能量历史
    ax3 = fig.add_subplot(133)
    ax3.plot(energies, 'r-')
    ax3.set_xlabel('蒙特卡洛步')
    ax3.set_ylabel('能量密度 e')
    ax3.set_title('能量演化')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_T{T:.2f}.png", dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    """
    主函数：演示如何使用工具函数求解凝聚态物理中的关键问题
    """
    print("凝聚态物理计算工具演示")
    print("=" * 50)
    
    # 示例1：金刚石晶格键角计算
    print("\n1. 金刚石晶格键角计算")
    print("-" * 50)
    
    # 计算金刚石晶格中键之间的夹角
    bond_angle = calculate_diamond_bond_angles()
    print(f"金刚石晶格中任意两个键之间的夹角: {bond_angle:.2f}°")
    
    # 可视化金刚石晶格结构
    print("正在生成金刚石晶格结构可视化...")
    visualize_diamond_lattice(size=2, show=False, save_path="./images/diamond_lattice.png")
    print("金刚石晶格结构图已保存至 ./images/diamond_lattice.png")
    
    # 示例2：量子多体系统模拟
    print("\n2. 量子多体系统模拟")
    print("-" * 50)
    
    # 创建一个简单的量子系统哈密顿量（例如，一维紧束缚模型）
    print("模拟一维紧束缚模型...")
    N = 10  # 系统大小
    t = 1.0  # 跃迁强度
    
    # 构建哈密顿量
    H = np.zeros((N, N))
    for i in range(N-1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    
    # 求解本征值和本征态
    eigenvalues, eigenvectors = simulate_quantum_system(H, num_states=N)
    
    print(f"系统的最低5个能级:")
    for i in range(min(5, len(eigenvalues))):
        print(f"E{i} = {eigenvalues[i]:.4f}")
    
    # 绘制能谱
    print("正在生成能谱图...")
    plot_energy_spectrum(eigenvalues, show=False, save_path="./images/energy_spectrum.png")
    print("能谱图已保存至 ./images/energy_spectrum.png")
    
    

if __name__ == "__main__":
    main()
    