# Filename: thermodynamic_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import CoolProp.CoolProp as CP
from scipy.integrate import solve_ivp

def ideal_gas_expansion(p0, V_ratio=2.0, gamma=1.4):
    """
    计算理想气体绝热自由膨胀过程的最终状态和热力学参数。
    
    在绝热自由膨胀过程中，气体从初始体积膨胀到更大体积，
    系统不与外界交换热量(绝热)，且不对外做功(自由膨胀)。
    
    Parameters:
    -----------
    p0 : float
        初始压强，单位：Pa
    V_ratio : float, optional
        膨胀后体积与初始体积的比值，默认为2.0
    gamma : float, optional
        气体的绝热指数(Cp/Cv)，默认为1.4(双原子分子)
    
    Returns:
    --------
    dict
        包含以下键值对的字典:
        - 'p_final': 最终压强，单位：Pa
        - 'T_ratio': 最终温度与初始温度的比值，无量纲
        - 'work': 系统对外做功，单位：J (对于自由膨胀为0)
        - 'entropy_change': 熵变，单位：J/K
    """
    # 自由膨胀是不可逆过程，最终压强为初始压强除以体积比
    p_final = p0 / V_ratio
    
    # 自由膨胀过程中系统不对外做功
    work = 0.0
    
    # 对于理想气体，温度比等于压强比的(gamma-1)/gamma次方
    # 但在自由膨胀中，由于是绝热且不可逆，最终温度与初始温度相同
    T_ratio = 1.0
    
    # 熵变计算 (理想气体自由膨胀的熵增)
    entropy_change = p0 * np.log(V_ratio)
    
    return {
        'p_final': p_final,
        'T_ratio': T_ratio,
        'work': work,
        'entropy_change': entropy_change
    }

def rankine_cycle_efficiency(T_high, P_high, P_low, fluid='Water', eta_turbine=1.0, eta_pump=1.0):
    """
    计算理想朗肯循环的热效率和关键状态点参数。
    
    朗肯循环是蒸汽动力装置的理想热力循环，包括等熵压缩、等压加热、
    等熵膨胀和等压冷凝四个过程。
    
    Parameters:
    -----------
    T_high : float
        锅炉出口蒸汽温度，单位：K
    P_high : float
        锅炉出口蒸汽压力，单位：Pa
    P_low : float
        冷凝器压力，单位：Pa
    fluid : str, optional
        工质名称，默认为'Water'
    eta_turbine : float, optional
        汽轮机等熵效率，范围[0,1]，默认为1.0(理想)
    eta_pump : float, optional
        泵的等熵效率，范围[0,1]，默认为1.0(理想)
    
    Returns:
    --------
    dict
        包含以下键值对的字典:
        - 'efficiency': 循环热效率，无量纲
        - 'w_turbine': 单位质量汽轮机做功，单位：J/kg
        - 'w_pump': 单位质量泵耗功，单位：J/kg
        - 'q_in': 单位质量加热量，单位：J/kg
        - 'states': 包含各状态点热力学参数的字典
    """
    try:
        # 状态点1：冷凝器出口(饱和液体)
        h1 = CP.PropsSI('H', 'P', P_low, 'Q', 0, fluid)
        s1 = CP.PropsSI('S', 'P', P_low, 'Q', 0, fluid)
        T1 = CP.PropsSI('T', 'P', P_low, 'Q', 0, fluid)
        
        # 状态点2s：泵出口(等熵压缩后)
        h2s = CP.PropsSI('H', 'P', P_high, 'S', s1, fluid)
        
        # 状态点2：考虑泵效率
        h2 = h1 + (h2s - h1) / eta_pump
        T2 = CP.PropsSI('T', 'P', P_high, 'H', h2, fluid)
        s2 = CP.PropsSI('S', 'P', P_high, 'H', h2, fluid)
        
        # 状态点3：锅炉出口(过热蒸汽)
        h3 = CP.PropsSI('H', 'P', P_high, 'T', T_high, fluid)
        s3 = CP.PropsSI('S', 'P', P_high, 'T', T_high, fluid)
        T3 = CP.PropsSI('T', 'P', P_high, 'H', h3, fluid)
        
        # 状态点4s：汽轮机出口(等熵膨胀后)
        s4s = s3
        h4s = CP.PropsSI('H', 'P', P_low, 'S', s4s, fluid)
        
        # 状态点4：考虑汽轮机效率
        h4 = h3 - eta_turbine * (h3 - h4s)
        T4 = CP.PropsSI('T', 'P', P_low, 'H', h4, fluid)
        s4 = CP.PropsSI('S', 'P', P_low, 'H', h4, fluid)
        
        # 计算各过程的热量和功量
        w_turbine = h3 - h4  # 汽轮机单位质量做功
        w_pump = h2 - h1     # 泵单位质量耗功
        q_in = h3 - h2       # 锅炉单位质量加热量
        
        # 计算循环热效率
        efficiency = (w_turbine - w_pump) / q_in
        
        # 收集各状态点参数
        states = {
            '1': {'T': T1, 'P': P_low, 'h': h1, 's': s1},
            '2': {'T': T2, 'P': P_high, 'h': h2, 's': s2},
            '3': {'T': T3, 'P': P_high, 'h': h3, 's': s3},
            '4': {'T': T4, 'P': P_low, 'h': h4, 's': s4}
        }
        
        return {
            'efficiency': efficiency,
            'w_turbine': w_turbine,
            'w_pump': w_pump,
            'q_in': q_in,
            'states': states
        }
    except Exception as e:
        print(f"计算错误: {e}")
        return None

def heat_diffusion_solver(L, nx, T_left, T_right, k, rho, c_p, t_final, nt):
    """
    求解一维热传导方程的数值解。
    
    使用有限差分法求解一维非稳态热传导方程：
    ρc_p(∂T/∂t) = k(∂²T/∂x²)
    
    Parameters:
    -----------
    L : float
        计算域长度，单位：m
    nx : int
        空间网格点数
    T_left : float
        左边界温度，单位：K
    T_right : float
        右边界温度，单位：K
    k : float
        热导率，单位：W/(m·K)
    rho : float
        密度，单位：kg/m³
    c_p : float
        比热容，单位：J/(kg·K)
    t_final : float
        模拟总时间，单位：s
    nt : int
        时间步数
    
    Returns:
    --------
    dict
        包含以下键值对的字典:
        - 'T': 温度场数组，形状为(nt+1, nx)
        - 'x': 空间坐标数组，形状为(nx,)
        - 't': 时间坐标数组，形状为(nt+1,)
        - 'alpha': 热扩散系数，单位：m²/s
    """
    # 计算热扩散系数
    alpha = k / (rho * c_p)
    
    # 空间和时间离散化
    dx = L / (nx - 1)
    dt = t_final / nt
    
    # 检查数值稳定性条件
    stability = alpha * dt / (dx**2)
    if stability > 0.5:
        print(f"警告: 数值不稳定! 稳定性参数 = {stability:.4f} > 0.5")
    
    # 初始化空间和时间网格
    x = np.linspace(0, L, nx)
    t = np.linspace(0, t_final, nt+1)
    
    # 初始化温度场 (初始条件为线性温度分布)
    T = np.zeros((nt+1, nx))
    T[0, :] = T_left + (T_right - T_left) * x / L
    
    # 显式时间推进求解
    for n in range(nt):
        for i in range(1, nx-1):
            T[n+1, i] = T[n, i] + alpha * dt / (dx**2) * (T[n, i+1] - 2*T[n, i] + T[n, i-1])
        
        # 边界条件
        T[n+1, 0] = T_left
        T[n+1, -1] = T_right
    
    return {
        'T': T,
        'x': x,
        't': t,
        'alpha': alpha
    }

def phase_field_simulation(nx, ny, dx, dy, D, time_steps, dt, initial_radius=10):
    """
    使用相场方法模拟材料相变过程。
    
    基于Allen-Cahn方程模拟相场演化:
    ∂φ/∂t = D∇²φ + φ(1-φ)(φ-0.5)
    
    Parameters:
    -----------
    nx, ny : int
        x和y方向的网格点数
    dx, dy : float
        x和y方向的网格间距，单位：无量纲
    D : float
        扩散系数，单位：无量纲
    time_steps : int
        模拟的时间步数
    dt : float
        时间步长，单位：无量纲
    initial_radius : float, optional
        初始相场圆形区域的半径，默认为10
    
    Returns:
    --------
    dict
        包含以下键值对的字典:
        - 'phi': 最终相场分布，形状为(ny, nx)
        - 'phi_history': 相场演化历史，形状为(time_steps//10+1, ny, nx)
        - 'interface_length': 界面长度随时间的变化，形状为(time_steps//10+1,)
    """
    # 初始化网格
    x = np.linspace(0, nx*dx, nx)
    y = np.linspace(0, ny*dy, ny)
    X, Y = np.meshgrid(x, y)
    
    # 初始化相场变量 (中心区域为1，外部为0)
    phi = np.zeros((ny, nx))
    center_x, center_y = nx*dx/2, ny*dy/2
    phi[(X-center_x)**2 + (Y-center_y)**2 < initial_radius**2] = 1.0
    
    # 存储相场演化历史
    history_interval = 10  # 每10步保存一次
    phi_history = np.zeros((time_steps//history_interval+1, ny, nx))
    phi_history[0] = phi.copy()
    
    # 存储界面长度
    interface_length = np.zeros(time_steps//history_interval+1)
    
    # 计算初始界面长度
    gradient_x, gradient_y = np.gradient(phi)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    interface_length[0] = np.sum(gradient_magnitude) * dx
    
    # 时间推进求解
    for step in range(time_steps):
        # 计算拉普拉斯算子
        laplacian = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) + 
                     np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 4*phi) / (dx*dy)
        
        # Allen-Cahn方程
        dphi_dt = D * laplacian + phi * (1 - phi) * (phi - 0.5)
        
        # 显式欧拉法时间推进
        phi += dt * dphi_dt
        
        # 保存历史数据
        if (step+1) % history_interval == 0:
            idx = (step+1) // history_interval
            phi_history[idx] = phi.copy()
            
            # 计算界面长度
            gradient_x, gradient_y = np.gradient(phi)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            interface_length[idx] = np.sum(gradient_magnitude) * dx
    
    return {
        'phi': phi,
        'phi_history': phi_history,
        'interface_length': interface_length
    }

def main():
    """
    主函数：演示如何使用工具函数求解自由膨胀问题
    """
    print("理想气体自由膨胀问题求解")
    print("-" * 50)
    
    # 问题参数
    p0 = 101325  # 初始压强，单位：Pa (1个标准大气压)
    
    # 求解自由膨胀问题
    result = ideal_gas_expansion(p0)
    print(result)
    
    # # 输出结果
    # print(f"初始压强: {p0} Pa")
    # print(f"最终压强: {result['p_final']:.2f} Pa")
    # print(f"温度比(T_final/T_initial): {result['T_ratio']:.2f}") 

if __name__ == '__main__':
    main()
