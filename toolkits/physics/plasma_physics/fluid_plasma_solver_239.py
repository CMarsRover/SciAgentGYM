# Filename: fluid_plasma_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

def solve_river_crossing(angle_outbound, angle_return, v_boat=1.0):
    """
    计算河流中船只往返时的水流速度比值。
    
    该问题是流体动力学中相对运动的典型应用，可类比于等离子体中带电粒子在电磁场中的运动轨迹计算。
    
    Parameters:
    -----------
    angle_outbound : float
        去程时船只与河岸的夹角（度），指船只航向与河岸法线的夹角
    angle_return : float
        返程时船只与河岸的夹角（度），指船只航向与河岸法线的夹角
    v_boat : float, optional
        船只在静水中的速度，默认为1.0（单位长度/单位时间）
    
    Returns:
    --------
    float
        返程时河水流速与去程时河水流速的比值
    """
    # 将角度转换为弧度
    angle_out_rad = np.radians(angle_outbound)
    angle_ret_rad = np.radians(angle_return)
    
    # 根据几何关系，计算去程时的水流速度与船速的比值
    # 在去程中，船只能够恰好到达对岸码头，说明水流速度与船速、航向角有特定关系
    v_flow1_ratio = np.sin(angle_out_rad)
    
    # 根据几何关系，计算返程时的水流速度与船速的比值
    # 在返程中，船只同样能够恰好到达对岸码头
    v_flow2_ratio = np.sin(angle_ret_rad)
    
    # 计算返程与去程的水流速度比值
    flow_ratio = v_flow2_ratio / v_flow1_ratio
    
    return flow_ratio

def simulate_plasma_particle_trajectory(E_field, B_field, q_m_ratio, initial_position, initial_velocity, t_span, t_points=1000):
    """
    模拟带电粒子在电磁场中的运动轨迹。
    
    使用洛伦兹力方程计算带电粒子在电磁场中的运动，这是等离子体物理中的基础问题。
    
    Parameters:
    -----------
    E_field : callable or array-like
        电场向量函数 E(t, r) 或常数电场向量 [Ex, Ey, Ez]
    B_field : callable or array-like
        磁场向量函数 B(t, r) 或常数磁场向量 [Bx, By, Bz]
    q_m_ratio : float
        粒子电荷与质量的比值 (q/m)
    initial_position : array-like
        初始位置向量 [x0, y0, z0]
    initial_velocity : array-like
        初始速度向量 [vx0, vy0, vz0]
    t_span : tuple
        时间积分区间 (t_start, t_end)
    t_points : int, optional
        时间积分点数，默认为1000
    
    Returns:
    --------
    tuple
        (t, positions, velocities) 其中:
        - t: 时间点数组
        - positions: 位置数组，形状为(t_points, 3)
        - velocities: 速度数组，形状为(t_points, 3)
    """
    def lorentz_force(t, state):
        """洛伦兹力方程的右端项"""
        r = state[:3]  # 位置
        v = state[3:]  # 速度
        
        # 获取电场和磁场
        if callable(E_field):
            E = E_field(t, r)
        else:
            E = E_field
            
        if callable(B_field):
            B = B_field(t, r)
        else:
            B = B_field
        
        # 计算洛伦兹力
        F = q_m_ratio * (E + np.cross(v, B))
        
        return np.concatenate([v, F])
    
    # 初始状态
    initial_state = np.concatenate([initial_position, initial_velocity])
    
    # 求解微分方程
    t = np.linspace(t_span[0], t_span[1], t_points)
    solution = solve_ivp(lorentz_force, t_span, initial_state, t_eval=t, method='RK45')
    
    # 提取结果
    positions = solution.y[:3].T
    velocities = solution.y[3:].T
    
    return solution.t, positions, velocities

def solve_mhd_equilibrium(pressure_gradient, j_current, B_field, grid_shape=(50, 50, 50), domain_size=(1.0, 1.0, 1.0)):
    """
    求解磁流体力学(MHD)平衡方程。
    
    在理想MHD中，平衡状态满足: ∇p = j × B，其中p是压力，j是电流密度，B是磁场。
    这是等离子体约束和稳定性研究中的基本问题。
    
    Parameters:
    -----------
    pressure_gradient : callable
        压力梯度函数 ∇p(x, y, z)，返回向量 [dpx, dpy, dpz]
    j_current : callable
        电流密度函数 j(x, y, z)，返回向量 [jx, jy, jz]
    B_field : callable
        磁场函数 B(x, y, z)，返回向量 [Bx, By, Bz]
    grid_shape : tuple, optional
        计算网格的形状 (nx, ny, nz)，默认为(50, 50, 50)
    domain_size : tuple, optional
        计算域的大小 (Lx, Ly, Lz)，默认为(1.0, 1.0, 1.0)
    
    Returns:
    --------
    tuple
        (coordinates, force_balance_error)，其中:
        - coordinates: 网格点坐标 (X, Y, Z)
        - force_balance_error: 力平衡误差 |∇p - j×B|
    """
    # 创建计算网格
    x = np.linspace(0, domain_size[0], grid_shape[0])
    y = np.linspace(0, domain_size[1], grid_shape[1])
    z = np.linspace(0, domain_size[2], grid_shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 计算每个网格点上的压力梯度、电流密度和磁场
    grad_p = np.zeros((*grid_shape, 3))
    j = np.zeros((*grid_shape, 3))
    B = np.zeros((*grid_shape, 3))
    
    for i in range(grid_shape[0]):
        for j_idx in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                point = (X[i, j_idx, k], Y[i, j_idx, k], Z[i, j_idx, k])
                grad_p[i, j_idx, k] = pressure_gradient(*point)
                j[i, j_idx, k] = j_current(*point)
                B[i, j_idx, k] = B_field(*point)
    
    # 计算j×B
    j_cross_B = np.zeros((*grid_shape, 3))
    j_cross_B[..., 0] = j[..., 1] * B[..., 2] - j[..., 2] * B[..., 1]
    j_cross_B[..., 1] = j[..., 2] * B[..., 0] - j[..., 0] * B[..., 2]
    j_cross_B[..., 2] = j[..., 0] * B[..., 1] - j[..., 1] * B[..., 0]
    
    # 计算力平衡误差
    force_balance_error = np.sqrt(np.sum((grad_p - j_cross_B)**2, axis=3))
    
    return (X, Y, Z), force_balance_error

def calculate_plasma_turbulence_spectrum(velocity_field, domain_size, resolution):
    """
    计算等离子体湍流的能量谱。
    
    湍流是等离子体物理中的重要现象，能量谱分析可以揭示湍流的尺度分布和能量级联过程。
    
    Parameters:
    -----------
    velocity_field : callable or array-like
        速度场函数 v(x, y, z) 或预先计算的速度场数组
    domain_size : tuple
        计算域的物理尺寸 (Lx, Ly, Lz)
    resolution : tuple
        计算网格的分辨率 (nx, ny, nz)
    
    Returns:
    --------
    tuple
        (k, E(k))，其中:
        - k: 波数数组
        - E(k): 对应的能量谱密度
    """
    # 创建计算网格
    x = np.linspace(0, domain_size[0], resolution[0])
    y = np.linspace(0, domain_size[1], resolution[1])
    z = np.linspace(0, domain_size[2], resolution[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 获取速度场
    if callable(velocity_field):
        v = np.zeros((*resolution, 3))
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                for k in range(resolution[2]):
                    v[i, j, k] = velocity_field(X[i, j, k], Y[i, j, k], Z[i, j, k])
    else:
        v = velocity_field
    
    # 计算速度场的傅里叶变换
    v_hat = np.zeros_like(v, dtype=complex)
    for i in range(3):
        v_hat[..., i] = np.fft.fftn(v[..., i])
    
    # 创建波数网格
    kx = 2 * np.pi * np.fft.fftfreq(resolution[0], domain_size[0]/resolution[0])
    ky = 2 * np.pi * np.fft.fftfreq(resolution[1], domain_size[1]/resolution[1])
    kz = 2 * np.pi * np.fft.fftfreq(resolution[2], domain_size[2]/resolution[2])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # 计算能量谱
    energy_density = 0.5 * (np.abs(v_hat[..., 0])**2 + np.abs(v_hat[..., 1])**2 + np.abs(v_hat[..., 2])**2)
    
    # 将能量密度按波数大小分bin
    k_max = np.max(K_mag)
    dk = np.min([kx[1] if len(kx) > 1 else kx[0], 
                 ky[1] if len(ky) > 1 else ky[0], 
                 kz[1] if len(kz) > 1 else kz[0]])
    k_bins = np.arange(0, k_max + dk, dk)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # 计算每个bin中的平均能量密度
    spectrum = np.zeros_like(k_centers)
    for i in range(len(k_centers)):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])
        if np.any(mask):
            spectrum[i] = np.mean(energy_density[mask])
    
    return k_centers, spectrum

def main():
    """
    主函数：演示如何使用工具函数求解流体等离子体相关问题
    """
    print("=== 流体等离子体计算工具演示 ===\n")
    
    # 示例1：河流穿越问题（流体相对运动）
    print("示例1：河流穿越问题（流体相对运动）")
    angle_outbound = 60  # 去程与河岸夹角
    angle_return = 30    # 返程与河岸夹角
    
    flow_ratio = solve_river_crossing(angle_outbound, angle_return)
    print(f"返程时河水流速是去程时的 {flow_ratio:.6f} 倍")
    print(f"数学表达式: √3/3 ≈ {np.sqrt(3)/3:.6f}")
    
    # 可视化河流穿越问题
    plt.figure(figsize=(10, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 河岸
    plt.plot([-1, 1], [0, 0], 'k-', linewidth=2, label='河岸')
    plt.plot([-1, 1], [1, 1], 'k-', linewidth=2)
    
    # 去程轨迹
    v_boat = 1.0
    v_flow1 = v_boat * np.sin(np.radians(angle_outbound))
    vx1 = v_boat * np.cos(np.radians(angle_outbound))
    vy1 = v_boat * np.sin(np.radians(angle_outbound))
    
    t1 = 1 / vy1  # 到达对岸的时间
    x1 = v_flow1 * t1  # 水平位移
    
    plt.arrow(0, 0, x1, 1, head_width=0.05, head_length=0.1, fc='blue', ec='blue', label='去程轨迹')
    
    # 返程轨迹
    v_flow2 = v_flow1 * flow_ratio
    vx2 = v_boat * np.cos(np.radians(angle_return))
    vy2 = v_boat * np.sin(np.radians(angle_return))
    
    t2 = 1 / vy2  # 到达对岸
      # 返程轨迹
    v_flow2 = v_flow1 * flow_ratio
    vx2 = v_boat * np.cos(np.radians(angle_return))
    vy2 = v_boat * np.sin(np.radians(angle_return))
    
    t2 = 1 / vy2  # 到达对岸的时间
    x2 = x1 - v_flow2 * t2  # 水平位移
    
    plt.arrow(x1, 1, -x1, -1, head_width=0.05, head_length=0.1, fc='red', ec='red', label='返程轨迹')
    
    # 水流方向
    for x in np.linspace(-0.8, 0.8, 5):
        plt.arrow(x, 0.2, 0, 0.1, head_width=0.02, head_length=0.04, fc='cyan', ec='cyan')
        plt.arrow(x, 0.7, 0, 0.1, head_width=0.02, head_length=0.04, fc='cyan', ec='cyan')
    
    plt.text(-0.1, 0.5, '流水', color='cyan', fontsize=12)
    plt.text(0.05, 0.3, f'去程角度: {angle_outbound}°', color='blue', fontsize=10)
    plt.text(0.05, 0.7, f'返程角度: {angle_return}°', color='red', fontsize=10)
    plt.text(-0.9, 0.1, '码头', fontsize=10)
    plt.text(0.8, 0.1, '码头', fontsize=10)
    
    plt.xlim(-1, 1)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title('河流穿越问题示意图')
    plt.axis('equal')
    
    # 示例2：带电粒子在电磁场中的运动（等离子体基本行为）
    print("\n示例2：带电粒子在电磁场中的运动（等离子体基本行为）")
    
    # 设置参数
    q_m_ratio = -1.76e11  # 电子电荷质量比 (C/kg)
    B_field = np.array([0, 0, 0.01])  # 均匀磁场 (T)
    E_field = np.array([0, 0, 0])     # 无电场
    
    initial_position = np.array([0, 0, 0])  # 初始位置
    initial_velocity = np.array([1e6, 0, 0])  # 初始速度 (m/s)
    
    t_span = (0, 1e-9)  # 时间范围 (s)
    
    # 计算轨迹
    t, positions, velocities = simulate_plasma_particle_trajectory(
        E_field, B_field, q_m_ratio, initial_position, initial_velocity, t_span)
    
    print(f"粒子在磁场中的回旋半径: {np.max(positions[:, 1]):.2e} m")
    print(f"回旋周期: {t[-1]*2:.2e} s")
    
    # 示例3：MHD平衡分析（磁流体力学）
    print("\n示例3：MHD平衡分析（磁流体力学）")
    
    # 定义简化的圆柱形等离子体平衡
    def pressure_gradient(x, y, z):
        r = np.sqrt(x**2 + y**2)
        if r < 1e-10:
            return np.array([0, 0, 0])
        # 简化的压力梯度模型
        dp_dr = -2 * r * np.exp(-r**2)
        return np.array([dp_dr * x/r, dp_dr * y/r, 0])
    
    def j_current(x, y, z):
        r = np.sqrt(x**2 + y**2)
        if r < 1e-10:
            return np.array([0, 0, 1])
        # 简化的电流密度模型
        j_theta = r * np.exp(-r**2)
        return np.array([-j_theta * y/r, j_theta * x/r, 0])
    
    def B_field_func(x, y, z):
        r = np.sqrt(x**2 + y**2)
        # 简化的磁场模型
        B_theta = 0.5 * r * (1 - np.exp(-r**2))
        B_z = np.sqrt(1 - 0.5 * r**2) if r < 1.0 else 0.5
        return np.array([-B_theta * y/r if r > 1e-10 else 0, 
                         B_theta * x/r if r > 1e-10 else 0, 
                         B_z])
    
    # 计算MHD平衡
    grid_shape = (20, 20, 1)  # 2D平面
    domain_size = (2.0, 2.0, 0.1)
    
    coords, force_error = solve_mhd_equilibrium(
        pressure_gradient, j_current, B_field_func, grid_shape, domain_size)
    
    print(f"MHD平衡最大误差: {np.max(force_error):.4e}")
    print(f"MHD平衡平均误差: {np.mean(force_error):.4e}")
    
    # 示例4：等离子体湍流能量谱（湍流传输）
    print("\n示例4：等离子体湍流能量谱（湍流传输）")
    
    # 生成简化的湍流速度场（使用随机相位的傅里叶模式）
    def generate_turbulent_field(resolution, domain_size, k0=5, spectral_slope=-5/3):
        """生成具有指定能谱的湍流场"""
        # 创建波数网格
        kx = 2 * np.pi * np.fft.fftfreq(resolution[0], domain_size[0]/resolution[0])
        ky = 2 * np.pi * np.fft.fftfreq(resolution[1], domain_size[1]/resolution[1])
        kz = 2 * np.pi * np.fft.fftfreq(resolution[2], domain_size[2]/resolution[2])
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # 生成随机相位
        phases = 2 * np.pi * np.random.random((*resolution, 3))
        
        # 生成速度场的傅里叶分量
        v_hat = np.zeros((*resolution, 3), dtype=complex)
        
        # 设置能量谱
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                for k in range(resolution[2]):
                    if K_mag[i, j, k] > 0:
                        # 能量谱 E(k) ~ k^spectral_slope
                        amplitude = (K_mag[i, j, k] / k0) ** (spectral_slope/2)
                        
                        # 确保速度场是无散的 (k·v = 0)
                        k_vec = np.array([KX[i, j, k], KY[i, j, k], KZ[i, j, k]])
                        if np.linalg.norm(k_vec) > 0:
                            k_unit = k_vec / np.linalg.norm(k_vec)
                            
                            # 生成随机向量
                            random_vec = np.random.random(3) - 0.5
                            
                            # 使其垂直于k
                            v_perp = random_vec - np.dot(random_vec, k_unit) * k_unit
                            if np.linalg.norm(v_perp) > 0:
                                v_perp = v_perp / np.linalg.norm(v_perp)
                                
                                # 设置振幅和相位
                                for d in range(3):
                                    v_hat[i, j, k, d] = amplitude * v_perp[d] * np.exp(1j * phases[i, j, k, d])
        
        # 反傅里叶变换得到实空间速度场
        v = np.zeros((*resolution, 3))
        for d in range(3):
            v[..., d] = np.real(np.fft.ifftn(v_hat[..., d]))
        
        return v
    
    # 生成湍流场并计算能量谱
    resolution = (64, 64, 64)
    domain_size = (2*np.pi, 2*np.pi, 2*np.pi)
    
    # 生成具有Kolmogorov能谱的湍流场
    v_field = generate_turbulent_field(resolution, domain_size, k0=5, spectral_slope=-5/3)
    
    # 计算能量谱
    k, E_k = calculate_plasma_turbulence_spectrum(v_field, domain_size, resolution)
    
    # 输出结果
    k_inertial = (k > 10) & (k < 30)
    if np.any(k_inertial):
        # 在惯性区计算谱指数
        log_k = np.log(k[k_inertial])
        log_E = np.log(E_k[k_inertial])
        slope, _, _, _, _ = np.polyfit(log_k, log_E, 1, full=True)
        print(f"湍流能量谱在惯性区的谱指数: {slope} (理论值: -5/3 ≈ -1.667)")
    
    print("\n=== 计算完成 ===")

if __name__ == "__main__":
    main()