# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, exp, pi, integrate, sqrt, simplify

def KineticEnergyIntegral(v1, v2, m=28 * 1.67e-27, T=300, N=6.022e23, k=1.38e-23, verbose=True, precision=2):
    """
    计算指定速度区间内的分子动能积分
    
    基于Maxwell-Boltzmann分布计算指定速度区间内分子的总动能，包括精确积分计算
    和近似方法验证，同时计算该区间内的分子数比例和平均动能。
    
    Parameters:
    -----------
    v1 : float
        速度区间下限，单位为m/s
    v2 : float
        速度区间上限，单位为m/s
    m : float, optional
        分子质量，单位为kg，默认为28个原子质量单位
    T : float, optional
        温度，单位为K，默认为300K
    N : float, optional
        总分子数，默认为阿伏伽德罗常数
    k : float, optional
        玻尔兹曼常数，单位为J/K，默认为1.38e-23
    verbose : bool, optional
        是否显示详细输出，默认为True
    precision : int, optional
        数值显示精度，默认为2位小数
    
    Returns:
    --------
    dict
        包含以下键值对的计算结果字典：
        - N_ratio : float
            区间内分子数比例
        - N_interval : float
            区间内分子数
        - E_avg : float
            区间内总平均动能，单位为J
        - total_E : float
            精确积分总动能，单位为J
        - relative_error : float
            近似方法与精确方法的相对误差百分比
    """
    
    coeff = 4 * np.pi * N * (m / (2 * np.pi * k * T)) ** 1.5

    def integrand(v):
        """
        计算动能积分被积函数
        
        计算在给定速度下的动能密度函数，用于数值积分计算指定速度区间内的总动能。
        
        Parameters:
        -----------
        v : float
            分子速度，单位为m/s
        
        Returns:
        --------
        float
            动能密度函数值，单位为J·s/m
        """
        if v < 0:
            return 0
        return 0.5 * m * v**2 * coeff * v**2 * np.exp(-m * v**2 / (2 * k * T))

    total_E, _ = quad(integrand, v1, v2)
    # 验证：使用近似方法
    # 在区间内取平均速度 v_avg = (v1 + v2) / 2
    
    v_avg = (v1 + v2) / 2
    E_avg = 0.5 * m * v_avg**2  # 平均动能
    
    if verbose:
        print(f"区间内平均速度: {v_avg:.0f} m/s")
        print(f"平均动能: {E_avg:.{precision}e} J")
    
    # 计算该区间内分子数比例（使用Maxwell-Boltzmann分布）
    coeff_dist = 4 * np.pi * (m / (2 * np.pi * k * T)) ** 1.5
    
    def f_v(v):
        """
        计算Maxwell-Boltzmann速度分布函数
        
        计算在给定速度下的分子数密度分布，用于统计力学分析。
        
        Parameters:
        -----------
        v : float
            分子速度，单位为m/s
        
        Returns:
        --------
        float
            速度分布函数值，无量纲
        """
        return coeff_dist * v**2 * np.exp(-m * v**2 / (2 * k * T))
    
    # 数值积分计算分子数比例
    N_ratio, _ = quad(f_v, v1, v2)
    N_interval = N * N_ratio
    results = {
      
        'N_ratio': N_ratio,
        'N_interval': N_interval,
        'E_avg': E_avg*N_interval,
        'total_E': total_E,
        'relative_error': abs(total_E - E_avg * N_interval) / total_E * 100
    }

    if verbose:
        print(f"区间内分子数比例: {N_ratio:.4f}")
        print(f"区间内分子数: {N_interval:.{precision}e}")
        print(f"近似结果 (E_avg × N_interval): {E_avg * N_interval:.{precision}e} J")
        print(f"精确积分结果: {total_E:.{precision}e} J")
        print(f"相对误差: {abs(total_E - E_avg * N_interval) / total_E * 100:.2f}%")
    return results

def visual_func_mb_distribution_and_integral(v1, v2, m=28 * 1.67e-27, T=300, N=6.022e23, k=1.38e-23, figsize=(10, 6), v_max=2000, n_points=500):
    """
    生成Maxwell-Boltzmann分布和动能积分的可视化图表
    
    创建动能密度分布图，显示指定速度区间内的积分区域，用于直观理解
    Maxwell-Boltzmann分布和动能积分计算过程。
    
    Parameters:
    -----------
    v1 : float
        速度区间下限，单位为m/s
    v2 : float
        速度区间上限，单位为m/s
    m : float, optional
        分子质量，单位为kg，默认为28个原子质量单位
    T : float, optional
        温度，单位为K，默认为300K
    N : float, optional
        总分子数，默认为阿伏伽德罗常数
    k : float, optional
        玻尔兹曼常数，单位为J/K，默认为1.38e-23
    figsize : tuple, optional
        图表尺寸，默认为(10, 6)
    v_max : float, optional
        速度显示上限，单位为m/s，默认为2000
    n_points : int, optional
        绘图点数，默认为500
    
    Returns:
    --------
    None
        显示图表
    """
    
    coeff = 4 * np.pi * (m / (2 * np.pi * k * T)) ** 1.5

    v = np.linspace(0, v_max, n_points)
    f_v = coeff * v**2 * np.exp(-m * v**2 / (2 * k * T))  # 分布密度
    kinetic_density = 0.5 * m * v**2 * N * f_v  # 动能密度函数

    plt.figure(figsize=figsize)
    plt.plot(v, kinetic_density, 'b-', lw=2, label=r'$\frac{1}{2}mv^2 N f(v)$')
    v_fill = v[(v >= v1) & (v <= v2)]
    kd_fill = kinetic_density[(v >= v1) & (v <= v2)]
    plt.fill_between(v_fill, kd_fill, color='orange', alpha=0.5, label=f'Integration Range [{v1}, {v2}]')

    plt.axvline(x=v1, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=v2, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel("Velocity v (m/s)")
    plt.ylabel("Kinetic Energy Density (J/m/s)")
    plt.title("Kinetic Energy Density Distribution in Velocity Interval")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def math_func_mb_kinetic_integral(verbose=True):
    """
    计算Maxwell-Boltzmann动能积分的数学表达式
    
    使用符号计算推导Maxwell-Boltzmann分布下的动能积分数学表达式，
    展示理论推导过程和不定积分结果。
    
    Parameters:
    -----------
    verbose : bool, optional
        是否显示详细输出，默认为True。控制是否打印数学表达式和积分结果
    
    Returns:
    --------
    sympy expression
        动能积分被积函数的符号表达式
    """
    
    v, m, k, T = symbols('v m k T', positive=True)
    f_v = 4 * pi * v**2 * (m / (2 * pi * k * T))**(3/2) * exp(-m * v**2 / (2 * k * T))
    kinetic_integrand = (1/2) * m * v**2 * f_v
    result = integrate(kinetic_integrand, v)
    
    if verbose:
        print("Mathematical Expression: Average Kinetic Energy Density per Unit Velocity Interval (Unnormalized)")
        print(f"Integrand: (1/2)mv² × f(v) = {simplify(kinetic_integrand)}")
        print(f"Indefinite Integral: {result}")
    
    return kinetic_integrand

# --- 示例调用 ---
if __name__ == "__main__":
    # 用户提供的具体参数
    v1, v2 = 400, 600  # m/s
    m = 5.0e-26        # kg (分子质量)
    T = 300            # K (温度)
    N = 6.02e23        # 总分子数
    
    print("=== 问题参数 ===")
    print(f"分子质量 m = {m:.1e} kg")
    print(f"温度 T = {T} K")
    print(f"总分子数 N = {N:.2e}")
    print(f"速度区间 v₁ = {v1} m/s, v₂ = {v2} m/s")
    print()
    
    # 计算积分值
    results = KineticEnergyIntegral(v1, v2, m=m, T=T, N=N)
    print(f"=== 计算结果 ===")
    print(results)
    # 数学表达式
    math_func_mb_kinetic_integral()
    
    # 可视化
    visual_func_mb_distribution_and_integral(v1, v2, m=m, T=T, N=N)

   
