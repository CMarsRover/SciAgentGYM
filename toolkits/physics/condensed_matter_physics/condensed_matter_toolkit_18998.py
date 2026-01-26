# Filename: condensed_matter_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os

# 确保图像目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def electron_motion_in_field(electric_field, effective_mass, time_span, initial_velocity=0):
    """
    计算电子在恒定电场下的运动
    
    基于经典力学和半导体物理学原理，计算电子在恒定电场下的加速运动。
    F = qE = m*a，其中q是电子电荷，E是电场强度，m是有效质量，a是加速度。
    
    Parameters:
    -----------
    electric_field : float
        电场强度，单位为V/m
    effective_mass : float
        电子的有效质量，以电子静止质量(m_e)为单位
    time_span : float or tuple
        如果是float，表示从0到该值的时间范围；如果是tuple，表示(t_start, t_end)
    initial_velocity : float, optional
        初始速度，单位为m/s，默认为0
        
    Returns:
    --------
    tuple
        (时间数组, 速度数组, 位置数组)，单位分别为s, m/s, m
    """
    # 电子电荷（C）
    q = const.e
    # 电子静止质量（kg）
    m_e = const.m_e
    # 有效质量（kg）
    m_eff = effective_mass * m_e
    
    # 加速度（m/s^2）
    acceleration = q * electric_field / m_eff
    
    # 处理时间范围
    if isinstance(time_span, (int, float)):
        t_start, t_end = 0, time_span
    else:
        t_start, t_end = time_span
    
    # 创建时间数组
    t = np.linspace(t_start, t_end, 1000)
    
    # 计算速度和位置
    v = initial_velocity + acceleration * t
    x = initial_velocity * t + 0.5 * acceleration * t**2
    
    return t, v, x

def band_structure_1d(k_range, band_params, band_type='parabolic'):
    """
    计算一维晶体的能带结构
    
    基于不同的能带模型（抛物线、紧束缚等）计算能带结构。
    
    Parameters:
    -----------
    k_range : tuple or array
        如果是tuple，表示(k_min, k_max, num_points)；如果是array，直接使用该k点数组
    band_params : dict
        能带参数，根据band_type不同而不同：
        - 'parabolic': {'effective_mass': float, 'band_edge': float}
        - 'tight_binding': {'hopping': float, 'lattice_constant': float}
        - 'kronig_penney': {'V0': float, 'a': float, 'b': float}
    band_type : str, optional
        能带模型类型，可选'parabolic'（抛物线近似）, 'tight_binding'（紧束缚模型）, 
        'kronig_penney'（Kronig-Penney模型）
        
    Returns:
    --------
    tuple
        (k点数组, 能量数组)，单位分别为1/m和eV
    """
    # 处理k范围
    if isinstance(k_range, tuple):
        k_min, k_max, num_points = k_range
        k = np.linspace(k_min, k_max, num_points)
    else:
        k = k_range
    
    # 根据不同模型计算能带
    if band_type == 'parabolic':
        # E = ħ^2*k^2/(2*m*)
        hbar = const.hbar
        m_eff = band_params['effective_mass'] * const.m_e
        band_edge = band_params.get('band_edge', 0)  # 能带边缘，默认为0
        
        # 计算能量（J）
        E = band_edge + (hbar**2 * k**2) / (2 * m_eff)
        # 转换为eV
        E = E / const.e
        
    elif band_type == 'tight_binding':
        # E = -2t*cos(ka)
        t = band_params['hopping']  # 跃迁积分（eV）
        a = band_params['lattice_constant']  # 晶格常数（m）
        
        E = -2 * t * np.cos(k * a)
        
    elif band_type == 'kronig_penney':
        # 使用Kronig-Penney模型的解析解
        V0 = band_params['V0']  # 势垒高度（eV）
        a = band_params['a']    # 势垒宽度（m）
        b = band_params['b']    # 势阱宽度（m）
        
        # 这里简化处理，实际应该求解超越方程
        # 这里仅作示意性计算
        E = V0 * np.abs(np.sin(k * (a + b)))
    
    else:
        raise ValueError(f"不支持的能带模型类型: {band_type}")
    
    return k, E

def monte_carlo_phase_transition(lattice_size, temperature_range, J=1.0, num_steps=10000, equilibration=1000):
    """
    使用蒙特卡洛方法模拟二维Ising模型的相变
    
    基于Metropolis算法实现的蒙特卡洛模拟，用于研究二维Ising模型的相变现象。
    
    Parameters:
    -----------
    lattice_size : int
        晶格大小（N x N）
    temperature_range : tuple or array
        如果是tuple，表示(T_min, T_max, num_points)；如果是array，直接使用该温度数组
    J : float, optional
        交换耦合常数，默认为1.0
    num_steps : int, optional
        每个温度点的蒙特卡洛步数，默认为10000
    equilibration : int, optional
        平衡化步数（在计算物理量前的热化步数），默认为1000
        
    Returns:
    --------
    tuple
        (温度数组, 能量数组, 磁化强度数组, 比热容数组, 磁化率数组)
    """
    # 处理温度范围
    if isinstance(temperature_range, tuple):
        T_min, T_max, num_points = temperature_range
        temperatures = np.linspace(T_min, T_max, num_points)
    else:
        temperatures = temperature_range
    
    # 初始化结果数组
    energies = np.zeros(len(temperatures))
    magnetizations = np.zeros(len(temperatures))
    specific_heats = np.zeros(len(temperatures))
    susceptibilities = np.zeros(len(temperatures))
    
    # 玻尔兹曼常数（设为1，相当于用约化单位）
    k_B = 1.0
    
    for t_idx, T in enumerate(temperatures):
        # 初始化随机自旋构型
        spins = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
        
        # 计算初始能量
        energy = 0
        for i in range(lattice_size):
            for j in range(lattice_size):
                energy += -J * spins[i, j] * (
                    spins[(i+1)%lattice_size, j] + 
                    spins[i, (j+1)%lattice_size]
                )
        
        # 存储能量和磁化强度的历史记录（用于计算涨落）
        energy_history = []
        mag_history = []
        
        # 蒙特卡洛循环
        for step in range(num_steps):
            # 随机选择一个格点
            i, j = np.random.randint(0, lattice_size, 2)
            
            # 计算翻转自旋的能量变化
            neighbors = (
                spins[(i+1)%lattice_size, j] + 
                spins[(i-1)%lattice_size, j] + 
                spins[i, (j+1)%lattice_size] + 
                spins[i, (j-1)%lattice_size]
            )
            delta_E = 2 * J * spins[i, j] * neighbors
            
            # Metropolis接受准则
            if delta_E <= 0 or np.random.random() < np.exp(-delta_E / (k_B * T)):
                spins[i, j] *= -1
                energy += delta_E
            
            # 平衡后开始收集数据
            if step >= equilibration:
                energy_history.append(energy)
                mag_history.append(np.abs(np.sum(spins)))
        
        # 计算物理量
        energy_history = np.array(energy_history)
        mag_history = np.array(mag_history)
        
        energies[t_idx] = np.mean(energy_history) / (lattice_size**2)
        magnetizations[t_idx] = np.mean(mag_history) / (lattice_size**2)
        
        # 比热容 C = (⟨E²⟩ - ⟨E⟩²) / (k_B * T² * N)
        specific_heats[t_idx] = (np.mean(energy_history**2) - np.mean(energy_history)**2) / (k_B * T**2 * lattice_size**2)
        
        # 磁化率 χ = (⟨M²⟩ - ⟨M⟩²) / (k_B * T * N)
        susceptibilities[t_idx] = (np.mean(mag_history**2) - np.mean(mag_history)**2) / (k_B * T * lattice_size**2)
    
    return temperatures, energies, magnetizations, specific_heats, susceptibilities

def quantum_state_evolution(hamiltonian, initial_state, time_span, method='exact'):
    """
    计算量子态在给定哈密顿量下的时间演化
    
    基于量子力学的薛定谔方程，计算量子态随时间的演化。
    
    Parameters:
    -----------
    hamiltonian : numpy.ndarray
        哈密顿量矩阵，形状为(n, n)
    initial_state : numpy.ndarray
        初始量子态，形状为(n,)
    time_span : tuple or array
        如果是tuple，表示(t_start, t_end, num_points)；如果是array，直接使用该时间数组
    method : str, optional
        演化方法，可选'exact'（精确对角化）或'runge_kutta'（龙格-库塔法），默认为'exact'
        
    Returns:
    --------
    tuple
        (时间数组, 量子态数组)，量子态数组形状为(time_points, n)
    """
    # 处理时间范围
    if isinstance(time_span, tuple):
        t_start, t_end, num_points = time_span
        times = np.linspace(t_start, t_end, num_points)
    else:
        times = time_span
    
    # 归一化初始态
    psi0 = initial_state / np.linalg.norm(initial_state)
    
    # 普朗克常数（约化）
    hbar = 1.0  # 使用自然单位
    
    if method == 'exact':
        # 对角化哈密顿量
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        
        # 将初始态展开到本征态基底
        coefficients = np.dot(eigenvectors.conj().T, psi0)
        
        # 时间演化
        states = np.zeros((len(times), len(psi0)), dtype=complex)
        for i, t in enumerate(times):
            # 应用时间演化算符 exp(-i*E*t/ħ)
            evolved_coeffs = coefficients * np.exp(-1j * eigenvalues * t / hbar)
            # 转换回原始基底
            states[i] = np.dot(eigenvectors, evolved_coeffs)
    
    elif method == 'runge_kutta':
        # 定义薛定谔方程的右侧
        def schrodinger_eq(t, psi):
            # i*ħ*dψ/dt = H*ψ  =>  dψ/dt = -i*H*ψ/ħ
            return -1j * np.dot(hamiltonian, psi) / hbar
        
        # 使用scipy的ODE求解器
        solution = solve_ivp(
            schrodinger_eq, 
            (times[0], times[-1]), 
            psi0, 
            t_eval=times,
            method='RK45'
        )
        
        states = solution.y.T
    
    else:
        raise ValueError(f"不支持的演化方法: {method}")
    
    return times, states

def main():
    """
    主函数：演示如何使用工具函数求解凝聚态物理问题
    """
    print("凝聚态物理工具包演示")
    print("-" * 50)
    
    # 示例1：电子在电场中的运动（半导体物理）
    print("\n示例1：GaAs中电子在电场中的运动")
    # 问题参数
    electric_field = 100 * 100  # 100 V/cm 转换为 V/m
    effective_mass = 0.067      # GaAs中电子的有效质量
    time_span = 0.1e-12         # 0.1 ps
    
    # 计算电子运动
    t, v, x = electron_motion_in_field(electric_field, effective_mass, time_span)
    
    # 输出结果
    final_velocity = v[-1]
    print(f"电场强度: {electric_field} V/m")
    print(f"电子有效质量: {effective_mass} m_e")
    print(f"加速时间: {time_span*1e12} ps")
    print(f"最终速度: {final_velocity:.2e} m/s")
    
    # 可视化电子运动
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t*1e12, v, 'b-')
    plt.xlabel('时间 (ps)')
    plt.ylabel('速度 (m/s)')
    plt.title('GaAs中电子在电场中的速度随时间变化')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t*1e12, x*1e9, 'r-')
    plt.xlabel('时间 (ps)')
    plt.ylabel('位移 (nm)')
    plt.title('GaAs中电子在电场中的位移随时间变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./images/electron_motion.png')
    
    # 示例2：一维晶体的能带结构
    print("\n示例2：一维晶体的能带结构")
    
    # 布里渊区范围
    k_range = (-5*np.pi, 5*np.pi, 1000)
    a = 5e-10  # 晶格常数 (5 Å)
    
    # 计算不同模型的能带结构
    k_parabolic, E_parabolic = band_structure_1d(
        k_range, 
        {'effective_mass': 0.067, 'band_edge': 0}, 
        'parabolic'
    )
    
    k_tight_binding, E_tight_binding = band_structure_1d(
        k_range, 
        {'hopping': 2.0, 'lattice_constant': a}, 
        'tight_binding'
    )
    
    # 可视化能带结构
    plt.figure(figsize=(10, 6))
    plt.plot(k_parabolic*a/np.pi, E_parabolic, 'b-', label='抛物线近似')
    plt.plot(k_tight_binding*a/np.pi, E_tight_binding, 'r-', label='紧束缚模型')
    
    plt.xlabel(r'波矢 $k$ ($\pi/a$)')
    plt.ylabel('能量 (eV)')
    plt.title('一维晶体的能带结构')
    plt.grid(True)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 标记布里渊区边界
    for i in range(-5, 6):
        if i != 0:
            plt.axvline(x=i, color='g', linestyle=':', alpha=0.5)
    
    plt.savefig('./images/band_structure.png')
    
    # 示例3：二维Ising模型的相变
    print("\n示例3：二维Ising模型的相变")
    
    # 模拟参数
    lattice_size = 20
    temperature_range = (1.0, 3.5, 20)
    
    # 运行蒙特卡洛模拟
    print("正在运行蒙特卡洛模拟，这可能需要一些时间...")
    T, E, M, C, X = monte_carlo_phase_transition(
        lattice_size, 
        temperature_range, 
        J=1.0, 
        num_steps=5000,  # 减少步数以加快演示
        equilibration=1000
    )
    
    # 可视化相变结果
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(T, E, 'o-')
    plt.xlabel('温度 (T/J)')
    plt.ylabel('能量 (E/J)')
    plt.title('能量随温度变化')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(T, M, 'o-')
    plt.xlabel('温度 (T/J)')
    plt.ylabel('磁化强度 (M)')
    plt.title('磁化强度随温度变化')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(T, C, 'o-')
    plt.xlabel('温度 (T/J)')
    plt.ylabel('比热容 (C)')
    plt.title('比热容随温度变化')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(T, X, 'o-')
    plt.xlabel('温度 (T/J)')
    plt.ylabel('磁化率 (χ)')
    plt.title('磁化率随温度变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./images/ising_phase_transition.png')
    
    # 示例4：量子态演化
    print("\n示例4：量子态演化")
    
    # 创建一个简单的二能级系统哈密顿量
    H = np.array([
        [1.0, 0.5],
        [0.5, -1.0]
    ])
    
    # 初始态（|0⟩状态）
    psi0 = np.array([1.0, 0.0])
    
    # 时间范围
    time_span = (0, 10, 200)
    
    # 计算量子态演化
    times, states = quantum_state_evolution(H, psi0, time_span)
    
    # 计算观测量：|0⟩和|1⟩态的概率
    prob_0 = np.abs(states[:, 0])**2
    prob_1 = np.abs(states[:, 1])**2
    
    # 可视化量子态演化
    plt.figure(figsize=(10, 6))
    plt.plot(times, prob_0, 'b-', label='|0⟩态概率')
    plt.plot(times, prob_1, 'r-', label='|1⟩态概率')
    plt.xlabel('时间 (ħ/E)')
    plt.ylabel('概率')
    plt.title('二能级系统的量子态演化')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('./images/quantum_evolution.png')
    
    print("\n所有示例计算完成，结果已保存到images目录")

if __name__ == "__main__":
    main()
    """
    主函数：演示如何使用工具函数求解凝聚态物理问题
    """
    print("凝聚态物理工具包演示")
    print("-" * 50)
    
    # 示例1：电子在电场中的运动（半导体物理）
    print("\n示例1：GaAs中电子在电场中的运动")
    # 问题参数
    electric_field = 100 * 100  # 100 V/cm 转换为 V/m
    effective_mass = 0.067      # GaAs中电子的有效质量
    time_span = 0.1e-12         # 0.1 ps
    
    # 计算电子运动
    t, v, x = electron_motion_in_field(electric_field, effective_mass, time_span)
    
    # 输出结果
    final_velocity = v[-1]
    print(f"电场强度: {electric_field} V/m")
    print(f"电子有效质量: {effective_mass} m_e")
    print(f"加速时间: {time_span*1e12} ps")
    print(f"最终速度: {final_velocity:.2e} m/s")