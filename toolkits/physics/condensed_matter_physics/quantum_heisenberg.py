# quantum_heisenberg.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from tqdm import tqdm
import argparse

def create_heisenberg_hamiltonian(L, J = 1.0, periodic = True):
    """
    创建二维正方晶格上的量子海森堡模型哈密顿量。

    Parameters:
    -----------
    L : int
        晶格的线性尺寸
    J : float
        交换耦合强度
    periodic : bool
        是否使用周期性边界条件
    Returns:
    H : scipy.sparse.csr_matrix
        哈密顿量
    --------
    """
    N = L * L
    dim = 2**N
    
    # 创建邻居列表
    G = nx.grid_2d_graph(L, L, periodic=periodic)
    edges = list(G.edges())
    
    row, col, data = [], [], []
    
    for i in range(dim):
        # 对角项：S_zS_z
        diag_term = 0.0
        for u, v in edges:
            site1 = u[0] * L + u[1]
            site2 = v[0] * L + v[1]
            
            s1z = 1 if (i & (1 << site1)) else -1
            s2z = 1 if (i & (1 << site2)) else -1
            diag_term += J * s1z * s2z
        
        row.append(i)
        col.append(i)
        data.append(diag_term)
        
        # 非对角项：S_xS_x + S_yS_y
        for u, v in edges:
            site1 = u[0] * L + u[1]
            site2 = v[0] * L + v[1]
            
            # 只有当两个自旋相反时才翻转
            if ((i >> site1) & 1) != ((i >> site2) & 1):
                j = i ^ (1 << site1) ^ (1 << site2)
                row.append(i)
                col.append(j)
                data.append(J)  # 正确的系数
    
    return csr_matrix((data, (row, col)), shape=(dim, dim))

def calculate_ground_state(L, J = 1.0, periodic = True):
    """
    计算哈密顿量的基态和基态能量。
    
    Parameters:
    -----------
    L : int
        晶格的线性尺寸
    J : float
        交换耦合强度
    periodic : bool
        是否使用周期性边界条件
    Returns:
    --------
    tuple
        (基态能量, 基态波函数)
    """
    H = create_heisenberg_hamiltonian(L, J, periodic)
    # 计算最小特征值和对应的特征向量
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
    return eigenvalues[0], eigenvectors[:, 0]

def calculate_magnetization(state, L):
    """
    计算给定量子态的磁化强度。
    
    Parameters:
    -----------
    state : List[float]
        量子态向量
    L : int
        晶格的线性尺寸
        
    Returns:
    --------
    float
        平均每个格点的磁化强度
    numpy.ndarray
        每个格点的磁化强度分布
    """
    N = L * L
    dim = 2**N
    
    # 计算每个格点的磁化强度
    site_magnetization = np.zeros(N)
    
    for i in range(dim):
        prob = abs(state[i])**2
        for site in range(N):
            # 计算第site位的自旋z分量
            sz = 1 if (i & (1 << site)) else -1
            site_magnetization[site] += sz * prob
    
    # 重塑为二维数组以便可视化
    mag_2d = site_magnetization.reshape(L, L)
    avg_mag = np.mean(np.abs(site_magnetization))
    
    return avg_mag, mag_2d

def simulate_temperature_dependence(L, J=1.0, T_range=np.linspace(0.1, 5.0, 20).tolist(), periodic=True):
    """
    模拟不同温度下的磁化行为。
    
    Parameters:
    -----------
    L : int
        晶格的线性尺寸
    J : float
        交换耦合强度
    T_range : List[float]
        温度范围
    periodic : bool
        是否使用周期性边界条件
        
    Returns:
    --------
    tuple
        (温度数组, 磁化强度数组)
    """
    # 构建哈密顿量
    H = create_heisenberg_hamiltonian(L, J, periodic)
    T_range = np.array(T_range)
    
    # 计算所有特征值和特征向量（对于小系统）
    if L <= 3:  # 限制系统大小以避免内存溢出
        eigenvalues, eigenvectors = eigsh(H, k=min(20, H.shape[0]-1), which='SA')
        
        magnetizations = []
        for T in tqdm(T_range, desc="Simulating temperatures"):
            # 计算玻尔兹曼权重
            if T > 0:
                weights = np.exp(-eigenvalues / T)
                weights /= np.sum(weights)
            else:
                # T=0时只考虑基态
                weights = np.zeros_like(eigenvalues)
                weights[0] = 1.0
            
            # 计算热平均磁化强度
            mag = 0
            for i, vec in enumerate(eigenvectors.T):
                m, _ = calculate_magnetization(vec.tolist(), L)
                mag += weights[i] * m
            
            magnetizations.append(mag)
    else:
        # 对于大系统，只计算基态
        magnetizations = [calculate_magnetization(calculate_ground_state(H)[1].tolist(), L)[0]] * len(T_range)
        print("Warning: System too large, only ground state magnetization calculated")
    
    return T_range, np.array(magnetizations)

def visualize_magnetization(mag_2d, title="Magnetization Distribution", save_path=None):
    """
    可视化磁化强度分布。
    
    Parameters:
    -----------
    mag_2d : ndarray
        二维磁化强度分布
    title : str
        图表标题
    save_path : str
        保存路径，如果为None则显示图形
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mag_2d, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Magnetization $\\langle S_z \\rangle$')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Magnetization plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_temperature_dependence(T_range, magnetizations, J, save_path=None):
    """
    可视化磁化强度随温度的变化。
    
    Parameters:
    -----------
    T_range : List[float]
        温度数组
    magnetizations : List[float]
        磁化强度数组
    J : float
        交换耦合强度
    save_path : str
        保存路径，如果为None则显示图形
    """
    plt.figure(figsize=(8, 6))
    plt.plot(T_range, magnetizations, 'o-', linewidth=2)
    plt.xlabel('Temperature $T/J$')
    plt.ylabel('Average Magnetization $\\langle |M| \\rangle$')
    plt.title(f'Magnetization vs Temperature (J={J})')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temperature dependence plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Quantum Heisenberg Model Simulation')
    parser.add_argument('--L', type=int, default=3, help='Linear lattice size')
    parser.add_argument('--J', type=float, default=1.0, help='Exchange coupling strength')
    parser.add_argument('--periodic', action='store_true', help='Use periodic boundary conditions')
    parser.add_argument('--T_min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--T_max', type=float, default=5.0, help='Maximum temperature')
    parser.add_argument('--T_steps', type=int, default=20, help='Number of temperature steps')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print(f"Simulating Heisenberg model on {args.L}x{args.L} lattice with J={args.J}")
    
    # 构建哈密顿量
    H = create_heisenberg_hamiltonian(args.L, args.J, args.periodic)
    
    # 计算基态
    print("Calculating ground state...")
    E0, psi0 = calculate_ground_state(args.L, args.J, args.periodic)
    print(f"Ground state energy: {E0:.6f}")
    
    # 计算基态磁化强度
    avg_mag, mag_2d = calculate_magnetization(psi0, args.L)
    print(f"Average ground state magnetization: {avg_mag:.6f}")
    
    if args.visualize:
        # 可视化基态磁化强度分布
        mag_plot_path = f"scienceqa-physics-1256_magnetization_L{args.L}_J{args.J}.png"
        visualize_magnetization(mag_2d, title=f"Ground State Magnetization (L={args.L}, J={args.J})", save_path=mag_plot_path)
        
        # 模拟温度依赖性
        print("Simulating temperature dependence...")
        T_range = np.linspace(args.T_min, args.T_max, args.T_steps)
        T, M = simulate_temperature_dependence(args.L, args.J, T_range, args.periodic)
        
        # 可视化温度依赖性
        temp_plot_path = f"scienceqa-physics-1256_temperature_dependence_L{args.L}_J{args.J}.png"
        visualize_temperature_dependence(T, M, args.J, save_path=temp_plot_path) 
        
    
    return {
        'isTrue': True,
        'answer': {
            'ground_state_energy': E0,
            'average_magnetization': avg_mag,
            'magnetization_distribution': mag_2d.tolist()
        }
    }

if __name__ == "__main__":
    main()