# quantum_spin_chain.py
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import networkx as nx
import argparse

def create_heisenberg_hamiltonian(N, J, h, periodic=True):
    """
    创建一维海森堡自旋链哈密顿量
    
    Parameters:
    -----------
    N : int
        自旋链长度（量子比特数）
    J : float
        自旋耦合强度
    h : float
        外部磁场强度
    periodic : bool, optional
        是否为周期性边界条件，默认为True
    
    Returns:
    --------
    qutip.Qobj
        系统的哈密顿量
    """
    # 初始化哈密顿量
    H = 0
    
    # 泡利矩阵
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    
    # 构建耦合项
    for i in range(N):
        # 横向磁场项
        h_term = h * tensor([sz if j == i else qeye(2) for j in range(N)])
        H += h_term
        
        # 最近邻相互作用
        if i < N-1 or periodic:
            j = (i + 1) % N  # 周期性边界条件
            
            # 构建相互作用项 J(σ_x^i σ_x^j + σ_y^i σ_y^j + σ_z^i σ_z^j)
            x_term = tensor([sx if k == i or k == j else qeye(2) for k in range(N)])
            y_term = tensor([sy if k == i or k == j else qeye(2) for k in range(N)])
            z_term = tensor([sz if k == i or k == j else qeye(2) for k in range(N)])
            
            H += J * (x_term + y_term + z_term)
    
    return H

def calculate_ground_state(H):
    """
    计算哈密顿量的基态和能量
    
    Parameters:
    -----------
    H : qutip.Qobj
        系统哈密顿量
    
    Returns:
    --------
    tuple
        (基态向量, 基态能量)
    """
    # 计算最低的特征值和特征向量
    eigen_energies, eigen_states = H.eigenstates(sparse=True, sort='low', eigvals=1)
    
    return eigen_states[0], eigen_energies[0]

def calculate_entanglement_entropy(state, site, N):
    """
    计算量子态的纠缠熵
    
    Parameters:
    -----------
    state : qutip.Qobj
        量子态向量
    site : int
        计算纠缠熵的切分点
    N : int
        系统总大小
    
    Returns:
    --------
    float
        纠缠熵值
    """
    # 创建切分
    subsystems = list(range(N))
    subsystem_A = subsystems[:site]
    
    # 计算约化密度矩阵
    rho_A = ptrace(state, subsystem_A)
    
    # 计算冯诺依曼熵
    return entropy_vn(rho_A)

def simulate_time_evolution(H, psi0, times):
    """
    模拟量子系统的时间演化
    
    Parameters:
    -----------
    H : qutip.Qobj
        系统哈密顿量
    psi0 : qutip.Qobj
        初始量子态
    times : numpy.ndarray
        时间点数组
    
    Returns:
    --------
    qutip.solver.Result
        时间演化结果
    """
    # 使用Schrödinger方程求解器
    result = sesolve(H, psi0, times, [])
    return result

def plot_magnetization_dynamics(result, N, times):
    """
    绘制磁化强度随时间的演化
    
    Parameters:
    -----------
    result : qutip.solver.Result
        时间演化结果
    N : int
        系统大小
    times : numpy.ndarray
        时间点数组
    
    Returns:
    --------
    matplotlib.figure.Figure
        绘制的图形
    """
    # 计算每个时间点的磁化强度
    sz_list = []
    for t_idx, psi_t in enumerate(result.states):
        # 计算每个格点的<σz>
        sz_expect = []
        for i in range(N):
            sz_i = tensor([sigmaz() if j == i else qeye(2) for j in range(N)])
            sz_expect.append(expect(sz_i, psi_t))
        sz_list.append(sz_expect)
    
    # 绘制热图
    plt.figure(figsize=(10, 6))
    plt.imshow(np.array(sz_list).T, aspect='auto', origin='lower', 
               extent=[times[0], times[-1], 0, N-1], cmap='RdBu_r')
    plt.colorbar(label=r'$\langle\sigma^z\rangle$')
    plt.xlabel('Time')
    plt.ylabel('Site')
    plt.title('Spin Magnetization Dynamics')
    
    return plt.gcf()

def plot_entanglement_entropy(J_values, N):
    """
    绘制不同耦合强度下的纠缠熵
    
    Parameters:
    -----------
    J_values : numpy.ndarray
        耦合强度值数组
    N : int
        系统大小
    
    Returns:
    --------
    matplotlib.figure.Figure
        绘制的图形
    """
    entropies = []
    
    for J in J_values:
        H = create_heisenberg_hamiltonian(N, J, 1.0)
        ground_state, _ = calculate_ground_state(H)
        entropy = calculate_entanglement_entropy(ground_state, N//2, N)
        entropies.append(entropy)
    
    plt.figure(figsize=(8, 6))
    plt.plot(J_values, entropies, 'o-')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point')
    plt.xlabel('Coupling Strength (J)')
    plt.ylabel('Entanglement Entropy')
    plt.title('Entanglement Entropy vs. Coupling Strength')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def parse_args():
    """
    解析命令行参数
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='Quantum Spin Chain Simulation')
    
    parser.add_argument('--N', type=int, default=8,
                        help='Number of spins in the chain')
    parser.add_argument('--J', type=float, default=1.0,
                        help='Coupling strength')
    parser.add_argument('--h', type=float, default=1.0,
                        help='External field strength')
    parser.add_argument('--periodic', action='store_true',
                        help='Use periodic boundary conditions')
    parser.add_argument('--evolution', action='store_true',
                        help='Simulate time evolution')
    parser.add_argument('--tmax', type=float, default=10.0,
                        help='Maximum simulation time')
    parser.add_argument('--phase_diagram', default=True,action='store_true',
                        help='Generate phase diagram')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 创建哈密顿量
    H = create_heisenberg_hamiltonian(args.N, args.J, args.h, args.periodic)
    
    # 计算基态
    ground_state, ground_energy = calculate_ground_state(H)
    print(f"Ground state energy: {ground_energy}")
    
    # 计算纠缠熵
    entropy = calculate_entanglement_entropy(ground_state, args.N//2, args.N)
    print(f"Entanglement entropy: {entropy}")
    
    # 时间演化模拟
    if args.evolution:
        # 创建初始状态 (所有自旋向上)
        psi0 = tensor([basis(2, 0) for _ in range(args.N)])
        
        # 设置时间点
        times = np.linspace(0, args.tmax, 100)
        
        # 模拟时间演化
        result = simulate_time_evolution(H, psi0, times)
        
        # 绘制磁化强度动力学
        fig = plot_magnetization_dynamics(result, args.N, times)
        fig.savefig('magnetization_dynamics.png')
        print("Magnetization dynamics saved to 'magnetization_dynamics.png'")
    
    # 生成相图
    if args.phase_diagram:
        J_values = np.linspace(0.1, 2.0, 20)
        fig = plot_entanglement_entropy(J_values, args.N)
        fig.savefig('entanglement_phase_diagram.png')
        print("Entanglement phase diagram saved to 'entanglement_phase_diagram.png'")