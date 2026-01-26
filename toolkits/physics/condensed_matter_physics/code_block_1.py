#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子多体系统中的磁性相互作用研究 - 海森堡链模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import qutip as qt
from magnetic_interaction import calculate_magnetic_moment

def create_heisenberg_hamiltonian(n_sites, J=1.0, h=0.0):
    """
    创建一维海森堡链的哈密顿量
    
    Parameters:
    -----------
    n_sites : int
        链中的格点数量
    J : float
        交换相互作用强度，J>0表示反铁磁耦合，J<0表示铁磁耦合
    h : float
        外部磁场强度
        
    Returns:
    --------
    qutip.Qobj
        系统的哈密顿量
    """
    # 创建空的哈密顿量
    H = qt.tensor([qt.qeye(2) for _ in range(n_sites)]) * 0
    
    # 泡利矩阵
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    
    # 添加近邻交换相互作用
    for i in range(n_sites-1):
        # 创建局部算符
        sx_list = [qt.qeye(2) for _ in range(n_sites)]
        sy_list = [qt.qeye(2) for _ in range(n_sites)]
        sz_list = [qt.qeye(2) for _ in range(n_sites)]
        
        # 在位置i和i+1放置泡利矩阵
        sx_list[i] = sx
        sx_list[i+1] = sx
        sx_term = qt.tensor(sx_list)
        
        sy_list[i] = sy
        sy_list[i+1] = sy
        sy_term = qt.tensor(sy_list)
        
        sz_list[i] = sz
        sz_list[i+1] = sz
        sz_term = qt.tensor(sz_list)
        
        # 添加到哈密顿量
        H += J * (sx_term + sy_term + sz_term)
    
    # 添加外部磁场项
    if h != 0:
        for i in range(n_sites):
            sz_list = [qt.qeye(2) for _ in range(n_sites)]
            sz_list[i] = sz
            H += h * qt.tensor(sz_list)
    
    return H

def calculate_susceptibility(n_sites, T_range, J=1.0, h_small=0.01):
    """
    计算不同温度下的磁化率
    
    Parameters:
    -----------
    n_sites : int
        链中的格点数量
    T_range : List[float]
        温度范围
    J : float
        交换相互作用强度
    h_small : float
        用于数值微分的小磁场
        
    Returns:
    --------
    array_like
        不同温度下的磁化率
    """
    susceptibility = []
    kb = 1.0  # 玻尔兹曼常数，这里设为1

    def calculate_magnetization(state, n_sites: int):
        sz = qt.sigmaz()
        magnetization = 0
        
        for i in range(n_sites):
            op_list = [qt.qeye(2) for _ in range(n_sites)]
            op_list[i] = sz
            sz_i = qt.tensor(op_list)
            magnetization += qt.expect(sz_i, state)
        
        return magnetization / n_sites
    
    for T in T_range:
        # 创建零磁场哈密顿量
        H0 = create_heisenberg_hamiltonian(n_sites, J, 0)
        
        # 创建小磁场哈密顿量
        H_h = create_heisenberg_hamiltonian(n_sites, J, h_small)
        
        # 计算零磁场的玻尔兹曼分布
        beta = 1.0 / (kb * T)
        rho0 = (-beta * H0).expm()
        rho0 = rho0 / rho0.tr()
        
        # 计算小磁场的玻尔兹曼分布
        rho_h = (-beta * H_h).expm()
        rho_h = rho_h / rho_h.tr()
        
        # 计算磁化强度
        m0 = calculate_magnetization(rho0, n_sites)
        mh = calculate_magnetization(rho_h, n_sites)
        
        # 计算磁化率 χ = ∂M/∂h
        chi = (mh - m0) / h_small
        susceptibility.append(chi)
    
    return np.array(susceptibility)

def main():
    # 参数设置
    n_sites = 6  # 格点数量
    J = 1.0      # 交换相互作用强度（反铁磁）
    T_range = np.linspace(0.1, 5.0, 50)  # 温度范围
    
    # 计算磁化率
    susceptibility = calculate_susceptibility(n_sites, T_range, J)
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(T_range, susceptibility, 'o-', linewidth=2)
    plt.xlabel('Temperature (K)', fontsize=14)
    plt.ylabel('Magnetic Susceptibility', fontsize=14)
    plt.title(f'Magnetic Susceptibility of {n_sites}-site Heisenberg Chain', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 分析临界行为
    # 在低温区域寻找磁化率的峰值
    peak_idx = np.argmax(susceptibility)
    T_critical = T_range[peak_idx]
    
    print(f"Estimated critical temperature: {T_critical:.3f} K")
    plt.axvline(x=T_critical, color='r', linestyle='--', 
                label=f'Critical T ≈ {T_critical:.3f} K')
    plt.legend()
    
    # 使用对数坐标分析临界指数
    plt.figure(figsize=(10, 6))
    # 只选择临界温度附近的数据点
    near_critical = (T_range > T_critical * 0.8) & (T_range < T_critical * 1.2)
    plt.loglog(abs(T_range[near_critical] - T_critical), 
              susceptibility[near_critical], 'o-')
    plt.xlabel('|T - T_c|', fontsize=14)
    plt.ylabel('Susceptibility', fontsize=14)
    plt.title('Critical Behavior Analysis', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    plt.show()
    
    # 使用前面创建的磁矩计算函数
    # 假设每个格点的有效磁矩与温度相关
    effective_volume = 1e-9  # m^3
    T_selected = 1.0  # 选择一个特定温度
    idx = np.abs(T_range - T_selected).argmin()
    magnetization_value = susceptibility[idx] * 1e5  # 转换为合适的单位
    
    magnetic_moment = calculate_magnetic_moment(effective_volume, magnetization_value)
    print(f"At T = {T_selected} K, the effective magnetic moment is: {magnetic_moment:.3e} A·m²")

if __name__ == "__main__":
    main()