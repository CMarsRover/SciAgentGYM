#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子自旋链模拟工具

该模块提供模拟一维海森堡量子自旋链的函数，用于研究量子多体系统中的磁性相变。
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import curve_fit
import argparse
from typing import Tuple, List, Dict, Union, Optional

def create_heisenberg_hamiltonian(
    n_sites: int, 
    j_coupling: float, 
    h_field: float, 
    boundary_condition: str = 'periodic'
) -> qt.Qobj:
    """
    创建一维海森堡模型的哈密顿量。
    
    Parameters:
    -----------
    n_sites : int
        自旋链中的格点数量
    j_coupling : float
        相邻自旋之间的交换耦合强度，J>0为反铁磁耦合，J<0为铁磁耦合
    h_field : float
        外部磁场强度
    boundary_condition : str, optional
        边界条件，可选 'periodic'（周期性）或 'open'（开放式）
        
    Returns:
    --------
    qt.Qobj
        表示系统哈密顿量的QuTiP量子对象
    """
    # 初始化哈密顿量
    hamiltonian = 0
    
    # 创建自旋算符
    sx_list = [qt.tensor([qt.sigmax() if i==k else qt.qeye(2) for i in range(n_sites)]) for k in range(n_sites)]
    sy_list = [qt.tensor([qt.sigmay() if i==k else qt.qeye(2) for i in range(n_sites)]) for k in range(n_sites)]
    sz_list = [qt.tensor([qt.sigmaz() if i==k else qt.qeye(2) for i in range(n_sites)]) for k in range(n_sites)]
    
    # 添加交换相互作用项
    for i in range(n_sites):
        if i < n_sites - 1:
            # 相邻格点之间的交换相互作用
            hamiltonian += j_coupling * (sx_list[i] * sx_list[i+1] + 
                                         sy_list[i] * sy_list[i+1] + 
                                         sz_list[i] * sz_list[i+1])
        elif boundary_condition == 'periodic':
            # 周期性边界条件下的首尾相互作用
            hamiltonian += j_coupling * (sx_list[i] * sx_list[0] + 
                                         sy_list[i] * sy_list[0] + 
                                         sz_list[i] * sz_list[0])
    
    # 添加外部磁场项
    for i in range(n_sites):
        hamiltonian += -h_field * sz_list[i]
    
    return hamiltonian

def calculate_thermal_expectation(
    hamiltonian: qt.Qobj, 
    operator: qt.Qobj, 
    temperature: float
) -> float:
    """
    计算给定温度下量子算符的热力学期望值。
    
    Parameters:
    -----------
    hamiltonian : qt.Qobj
        系统的哈密顿量
    operator : qt.Qobj
        要计算期望值的量子算符
    temperature : float
        系统温度，单位为能量单位（假设kB=1）
        
    Returns:
    --------
    float
        算符的热力学期望值
    """
    # 对于T=0的特殊情况（基态）
    if temperature == 0:
        eigvals, eigvecs = hamiltonian.eigenstates()
        ground_state = eigvecs[0]
        return qt.expect(operator, ground_state)
    
    # 计算密度矩阵 ρ = exp(-H/kT)/Z
    beta = 1.0 / temperature
    rho = (-beta * hamiltonian).expm()
    # 归一化（除以配分函数Z）
    rho = rho / rho.tr()
    
    # 计算期望值 <O> = Tr(ρO)
    expectation_value = qt.expect(operator, rho)
    return expectation_value

def calculate_magnetization(
    n_sites: int, 
    j_coupling: float, 
    h_field: float, 
    temperature: float,
    boundary_condition: str = 'periodic'
) -> float:
    """
    计算给定参数下自旋链的平均磁化强度。
    
    Parameters:
    -----------
    n_sites : int
        自旋链中的格点数量
    j_coupling : float
        相邻自旋之间的交换耦合强度
    h_field : float
        外部磁场强度
    temperature : float
        系统温度
    boundary_condition : str, optional
        边界条件
        
    Returns:
    --------
    float
        每个格点的平均磁化强度
    """
    # 创建哈密顿量
    hamiltonian = create_heisenberg_hamiltonian(n_sites, j_coupling, h_field, boundary_condition)
    
    # 创建总磁化算符（z方向）
    sz_total = sum([qt.tensor([qt.sigmaz() if i==k else qt.qeye(2) for i in range(n_sites)]) 
                   for k in range(n_sites)])
    
    # 计算磁化强度的热力学期望值
    magnetization = calculate_thermal_expectation(hamiltonian, sz_total, temperature)
    
    # 返回每个格点的平均磁化强度
    return magnetization / n_sites

def simulate_phase_transition(
    n_sites: int,
    j_coupling: float,
    h_fields: np.ndarray,
    temperatures: np.ndarray,
    boundary_condition: str = 'periodic'
) -> Dict[str, np.ndarray]:
    """
    模拟不同磁场和温度下的磁化行为，研究相变现象。
    
    Parameters:
    -----------
    n_sites : int
        自旋链中的格点数量
    j_coupling : float
        相邻自旋之间的交换耦合强度
    h_fields : np.ndarray
        要扫描的外部磁场强度数组
    temperatures : np.ndarray
        要扫描的温度数组
    boundary_condition : str, optional
        边界条件
        
    Returns:
    --------
    Dict[str, np.ndarray]
        包含模拟结果的字典，键包括：
        - 'h_fields': 磁场强度数组
        - 'temperatures': 温度数组
        - 'magnetization': 磁化强度数组（形状为[len(temperatures), len(h_fields)]）
        - 'susceptibility': 磁化率数组
    """
    # 初始化结果数组
    magnetization = np.zeros((len(temperatures), len(h_fields)))
    susceptibility = np.zeros((len(temperatures), len(h_fields)-1))
    
    # 对每个温度进行计算
    for i, temp in enumerate(temperatures):
        for j, h in enumerate(h_fields):
            magnetization[i, j] = calculate_magnetization(n_sites, j_coupling, h, temp, boundary_condition)
        
        # 计算磁化率 χ = dM/dH
        susceptibility[i] = np.diff(magnetization[i]) / np.diff(h_fields)
    
    return {
        'h_fields': h_fields,
        'temperatures': temperatures,
        'magnetization': magnetization,
        'susceptibility': susceptibility
    }

def find_critical_point(
    h_fields: np.ndarray,
    susceptibility: np.ndarray
) -> float:
    """
    通过寻找磁化率峰值来确定临界点。
    
    Parameters:
    -----------
    h_fields : np.ndarray
        磁场强度数组
    susceptibility : np.ndarray
        对应的磁化率数组
        
    Returns:
    --------
    float
        估计的临界磁场强度
    """
    # 使用磁化率的中点位置
    h_midpoints = (h_fields[1:] + h_fields[:-1]) / 2
    
    # 找到磁化率最大值对应的磁场
    max_idx = np.argmax(np.abs(susceptibility))
    critical_h = h_midpoints[max_idx]
    
    return critical_h

def plot_results(results: Dict[str, np.ndarray], save_path: Optional[str] = None):
    """
    绘制模拟结果。
    
    Parameters:
    -----------
    results : Dict[str, np.ndarray]
        模拟结果字典
    save_path : str, optional
        保存图像的路径，如果为None则显示图像
    """
    h_fields = results['h_fields']
    temperatures = results['temperatures']
    magnetization = results['magnetization']
    susceptibility = results['susceptibility']
    
    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制磁化强度
    for i, temp in enumerate(temperatures):
        ax1.plot(h_fields, magnetization[i], '-o', label=f'T = {temp:.2f}')
    
    ax1.set_xlabel('磁场强度 (H)')
    ax1.set_ylabel('磁化强度 (M)')
    ax1.set_title('磁化强度随磁场变化')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制磁化率
    h_midpoints = (h_fields[1:] + h_fields[:-1]) / 2
    for i, temp in enumerate(temperatures):
        ax2.plot(h_midpoints, np.abs(susceptibility[i]), '-o', label=f'T = {temp:.2f}')
    
    ax2.set_xlabel('磁场强度 (H)')
    ax2.set_ylabel('磁化率 |χ|')
    ax2.set_title('磁化率随磁场变化')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='量子自旋链相变模拟工具')
    parser.add_argument('--n_sites', type=int, default=6, help='自旋链中的格点数量')
    parser.add_argument('--j_coupling', type=float, default=1.0, help='交换耦合强度')
    parser.add_argument('--h_min', type=float, default=0.0, help='最小磁场强度')
    parser.add_argument('--h_max', type=float, default=4.0, help='最大磁场强度')
    parser.add_argument('--h_steps', type=int, default=20, help='磁场扫描步数')
    parser.add_argument('--t_min', type=float, default=0.1, help='最小温度')
    parser.add_argument('--t_max', type=float, default=2.0, help='最大温度')
    parser.add_argument('--t_steps', type=int, default=4, help='温度扫描步数')
    parser.add_argument('--boundary', choices=['periodic', 'open'], default='periodic', help='边界条件')
    parser.add_argument('--save_path', type=str, help='保存图像的路径')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 生成磁场和温度数组
    h_fields = np.linspace(args.h_min, args.h_max, args.h_steps)
    temperatures = np.linspace(args.t_min, args.t_max, args.t_steps)
    
    # 运行模拟
    results = simulate_phase_transition(
        args.n_sites, 
        args.j_coupling, 
        h_fields, 
        temperatures, 
        args.boundary
    )
    
    # 寻找临界点
    for i, temp in enumerate(temperatures):
        critical_h = find_critical_point(h_fields, results['susceptibility'][i])
        print(f"温度 T = {temp:.2f} 的估计临界磁场: H_c = {critical_h:.4f}")
    
    # 绘制结果
    plot_results(results, args.save_path)