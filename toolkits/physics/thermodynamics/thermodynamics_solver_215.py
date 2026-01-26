# Filename: thermodynamics_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import constants

def ideal_gas_state(P, V, n=1, R=constants.R):
    """
    计算理想气体状态方程的温度
    
    Parameters:
    -----------
    P : float
        压力，单位：Pa
    V : float
        体积，单位：m³
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float
        温度，单位：K
    """
    return P * V / (n * R)

def isothermal_process(V, T, n=1, R=constants.R):
    """
    计算等温过程中的压力
    
    Parameters:
    -----------
    V : float or array_like
        体积，单位：m³
    T : float
        温度，单位：K
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float or array_like
        压力，单位：Pa
    """
    return n * R * T / V

def isochoric_process(T, V, n=1, R=constants.R):
    """
    计算等容过程中的压力
    
    Parameters:
    -----------
    T : float or array_like
        温度，单位：K
    V : float
        体积，单位：m³
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float or array_like
        压力，单位：Pa
    """
    return n * R * T / V

def isobaric_process(V, P, n=1, R=constants.R):
    """
    计算等压过程中的温度
    
    Parameters:
    -----------
    V : float or array_like
        体积，单位：m³
    P : float
        压力，单位：Pa
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float or array_like
        温度，单位：K
    """
    return P * V / (n * R)

def calculate_work_heat(process_type, initial_state, final_state, n=1, R=constants.R):
    """
    计算热力学过程中的功和热量
    
    Parameters:
    -----------
    process_type : str
        过程类型，可选值：'isothermal', 'isochoric', 'isobaric', 'adiabatic'
    initial_state : tuple
        初始状态 (P, V, T)，单位分别为：Pa, m³, K
    final_state : tuple
        终止状态 (P, V, T)，单位分别为：Pa, m³, K
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    tuple
        (功, 热量)，单位：J
        注意：系统对外做功为正
    """
    P1, V1, T1 = initial_state
    P2, V2, T2 = final_state
    
    if process_type == 'isothermal':
        # 等温过程
        work = n * R * T1 * np.log(V2/V1)
        heat = work  # 等温过程中，热量等于功
    elif process_type == 'isochoric':
        # 等容过程
        work = 0  # 等容过程不做功
        heat = n * R * (T2 - T1)  # 单原子理想气体，Cv = 3R/2
    elif process_type == 'isobaric':
        # 等压过程
        work = P1 * (V2 - V1)
        heat = n * R * (T2 - T1) * 5/2  # 单原子理想气体，Cp = 5R/2
    elif process_type == 'adiabatic':
        # 绝热过程
        work = n * R * (T1 - T2) / (5/3 - 1)  # 单原子理想气体，γ = 5/3
        heat = 0  # 绝热过程无热量交换
    else:
        raise ValueError("不支持的过程类型")
    
    return work, heat

def calculate_cycle_efficiency(states, processes, n=1, R=constants.R):
    """
    计算热力学循环的效率
    
    Parameters:
    -----------
    states : list of tuples
        循环中各状态点 [(P1, V1, T1), (P2, V2, T2), ...]，单位分别为：Pa, m³, K
    processes : list of str
        各过程类型 ['isothermal', 'isochoric', ...]
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float
        循环效率 η = W_net / Q_in
    """
    total_work = 0
    heat_in = 0
    heat_out = 0
    
    for i in range(len(states)):
        initial_state = states[i]
        final_state = states[(i+1) % len(states)]
        process_type = processes[i]
        
        work, heat = calculate_work_heat(process_type, initial_state, final_state, n, R)
        total_work += work
        
        if heat > 0:
            heat_in += heat
        else:
            heat_out += abs(heat)
    
    efficiency = total_work / heat_in if heat_in != 0 else 0
    return efficiency

def solve_thermodynamic_cycle(T1, T3, n=1, R=constants.R):
    """
    求解由两条等容线和两条等压线构成的循环过程中状态b的温度
    
    Parameters:
    -----------
    T1 : float
        状态a的温度，单位：K
    T3 : float
        状态c的温度，单位：K
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    float
        状态b的温度，单位：K
    """
    # 设定初始条件（可以任意设定，因为我们关心的是比例关系）
    P1 = 1.0  # 状态a的压力
    V1 = 1.0  # 状态a的体积
    
    # 状态a: (P1, V1, T1)
    # 状态b: (P2, V1, T2) - 等容过程a→b
    # 状态c: (P2, V2, T3) - 等压过程b→c
    # 状态d: (P1, V2, T4) - 等容过程c→d
    # 状态a: (P1, V1, T1) - 等压过程d→a
    
    # 计算其他状态点的参数
    P2 = P1 * T3 / T1  # 等压过程d→a和b→c，P2/P1 = T3/T1
    V2 = V1 * T3 / T1  # 等容过程a→b和c→d，V2/V1 = T3/T1
    
    # 状态b和d在同一等温线上，即P2*V1 = P1*V2
    # 代入上面的关系，得到T2 = sqrt(T1*T3)
    T2 = np.sqrt(T1 * T3)
    
    return T2

def plot_thermodynamic_cycle(T1, T3, n=1, R=constants.R):
    """
    绘制热力学循环的P-V图
    
    Parameters:
    -----------
    T1 : float
        状态a的温度，单位：K
    T3 : float
        状态c的温度，单位：K
    n : float, optional
        物质的量，默认为1 mol
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    
    Returns:
    --------
    None
    """
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设定初始条件
    P1 = 1.0  # 状态a的压力
    V1 = 1.0  # 状态a的体积
    
    # 计算T2和其他状态点
    T2 = solve_thermodynamic_cycle(T1, T3, n, R)
    P2 = P1 * T3 / T1
    V2 = V1 * T3 / T1
    T4 = T1 * V2 / V1
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 绘制循环路径
    plt.plot([V1, V1], [P1, P2], 'b-', label='等容过程a→b')
    plt.plot([V1, V2], [P2, P2], 'r-', label='等压过程b→c')
    plt.plot([V2, V2], [P2, P1], 'g-', label='等容过程c→d')
    plt.plot([V2, V1], [P1, P1], 'm-', label='等压过程d→a')
    
    # 标记状态点
    plt.scatter([V1, V1, V2, V2], [P1, P2, P2, P1], color='black')
    plt.text(V1-0.05, P1-0.05, 'a', fontsize=12)
    plt.text(V1-0.05, P2+0.05, 'b', fontsize=12)
    plt.text(V2+0.05, P2+0.05, 'c', fontsize=12)
    plt.text(V2+0.05, P1-0.05, 'd', fontsize=12)
    
    # 绘制等温线
    V_range = np.linspace(V1*0.5, V2*1.5, 100)
    plt.plot(V_range, n*R*T2/(V_range), 'k--', label=f'等温线 T={T2:.1f}K')
    
    # 设置坐标轴
    plt.xlabel('体积 V (m³)')
    plt.ylabel('压力 P (Pa)')
    plt.title('理想气体循环过程P-V图')
    plt.grid(True)
    plt.legend()
    
    # 显示温度信息
    info_text = f"状态a温度: {T1}K\n状态b温度: {T2:.4f}K\n状态c温度: {T3}K\n状态d温度: {T4:.4f}K"
    plt.figtext(0.15, 0.15, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示如何使用工具函数求解当前问题
    """
    # 问题参数定义
    T1 = 300  # 状态a的温度，单位：K
    T3 = 600  # 状态c的温度，单位：K
    
    # 求解状态b的温度
    T2 = solve_thermodynamic_cycle(T1, T3)
    
    # 输出结果
    print(f"状态a的温度 T1 = {T1} K")
    print(f"状态c的温度 T3 = {T3} K")
    print(f"状态b的温度 T2 = {T2} K = √(T1·T3) = √({T1}·{T3}) = {T2} K")
    
    # 可选：绘制热力学循环图
    plot_thermodynamic_cycle(T1, T3)

if __name__ == "__main__":
    main()