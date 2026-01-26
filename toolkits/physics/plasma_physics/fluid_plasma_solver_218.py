# Filename: fluid_plasma_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import constants
import warnings

# 设置matplotlib后端和中文显示
import matplotlib
matplotlib.use('Agg')  # 或者使用 'Qt5Agg'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def pipe_flow_velocity(radius, position, flow_rate, viscosity=1.0, pressure_gradient=1.0):
    """
    计算圆管内层流(泊肃叶流动)的速度分布
    
    基于Hagen-Poiseuille方程，计算圆管内任意位置的流速。
    对于不可压缩流体在圆管内的层流，速度分布呈抛物线形。
    
    Parameters:
    -----------
    radius : float
        管道半径，单位：米
    position : float or array_like
        距离管道中心的径向距离，单位：米
    flow_rate : float
        体积流量，单位：立方米/秒
    viscosity : float, optional
        流体动力粘度，单位：Pa·s，默认为1.0
    pressure_gradient : float, optional
        沿管道轴向的压力梯度，单位：Pa/m，默认为1.0
        
    Returns:
    --------
    float or ndarray
        指定位置的流速，单位：米/秒
    """
    # 确保position不超过radius
    if np.any(np.abs(position) > radius):
        warnings.warn("Some positions are outside the pipe radius")
        position = np.clip(position, -radius, radius)
    
    # 泊肃叶流动的速度分布公式
    velocity = (pressure_gradient * (radius**2 - position**2)) / (4 * viscosity)
    
    # 如果提供了流量，则归一化速度分布
    if flow_rate is not None:
        # 计算理论总流量
        theoretical_flow = np.pi * pressure_gradient * radius**4 / (8 * viscosity)
        # 归一化系数
        scale_factor = flow_rate / theoretical_flow
        velocity *= scale_factor
        
    return velocity

def mhd_flow_velocity(radius, position, B_field, current_density, fluid_density, 
                      viscosity=1.0, pressure_gradient=1.0):
    """
    计算磁流体动力学(MHD)条件下圆管内的速度分布
    
    考虑洛伦兹力影响下的导电流体在圆管内的流动。
    
    Parameters:
    -----------
    radius : float
        管道半径，单位：米
    position : float or array_like
        距离管道中心的径向距离，单位：米
    B_field : float
        垂直于流动方向的磁场强度，单位：特斯拉
    current_density : float
        电流密度，单位：安培/平方米
    fluid_density : float
        流体密度，单位：千克/立方米
    viscosity : float, optional
        流体动力粘度，单位：Pa·s，默认为1.0
    pressure_gradient : float, optional
        沿管道轴向的压力梯度，单位：Pa/m，默认为1.0
        
    Returns:
    --------
    float or ndarray
        指定位置的流速，单位：米/秒
    """
    # 基础流速(无磁场)
    base_velocity = pipe_flow_velocity(radius, position, None, viscosity, pressure_gradient)
    
    # 计算哈特曼数 (Hartmann number)
    hartmann = B_field * radius * np.sqrt(constants.sigma / viscosity)
    
    # 计算洛伦兹力修正项
    lorentz_term = current_density * B_field / fluid_density
    
    # 修正后的速度
    modified_velocity = base_velocity
    
    # 对于强磁场，速度分布会从抛物线变为更平坦的形状
    if hartmann > 1:
        # 简化模型：哈特曼数越大，速度分布越平坦
        flatness_factor = 1 - np.exp(-position**2 / (radius**2 / hartmann))
        modified_velocity *= flatness_factor
    
    # 添加洛伦兹力的影响
    modified_velocity += lorentz_term * (radius**2 - position**2) / (2 * viscosity)
    
    return modified_velocity

def plasma_kinetic_simulation(n_particles, temperature, B_field, E_field, 
                             mass=constants.m_p, charge=constants.e, 
                             time_span=(0, 1e-6), time_steps=1000):
    """
    简化的等离子体粒子运动模拟
    
    使用Boris算法模拟带电粒子在电磁场中的运动。
    
    Parameters:
    -----------
    n_particles : int
        模拟粒子数量
    temperature : float
        等离子体温度，单位：开尔文
    B_field : array_like, shape (3,)
        磁场矢量，单位：特斯拉
    E_field : array_like, shape (3,)
        电场矢量，单位：伏特/米
    mass : float, optional
        粒子质量，单位：千克，默认为质子质量
    charge : float, optional
        粒子电荷，单位：库仑，默认为电子电荷
    time_span : tuple, optional
        模拟时间范围，单位：秒，默认为(0, 1e-6)
    time_steps : int, optional
        时间步数，默认为1000
        
    Returns:
    --------
    tuple
        (times, positions, velocities) 包含时间点、粒子位置和速度的数组
    """
    # 将输入转换为numpy数组
    B_field = np.array(B_field, dtype=float)
    E_field = np.array(E_field, dtype=float)
    
    # 初始化粒子位置和速度
    # 假设初始位置在原点附近随机分布
    positions = np.random.normal(0, 1e-6, (n_particles, 3))
    
    # 基于Maxwell-Boltzmann分布初始化速度
    thermal_velocity = np.sqrt(constants.k * temperature / mass)
    velocities = np.random.normal(0, thermal_velocity, (n_particles, 3))
    
    # 时间步长
    dt = (time_span[1] - time_span[0]) / time_steps
    times = np.linspace(time_span[0], time_span[1], time_steps)
    
    # 存储结果
    position_history = np.zeros((time_steps, n_particles, 3))
    velocity_history = np.zeros((time_steps, n_particles, 3))
    
    # Boris算法的辅助函数
    def boris_push(v, pos, dt, E, B, q, m):
        # 半步电场加速
        v_minus = v + q * E * dt / (2 * m)
        
        # 磁场旋转
        t = q * B * dt / (2 * m)
        s = 2 * t / (1 + np.sum(t**2))
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        
        # 半步电场加速
        v_new = v_plus + q * E * dt / (2 * m)
        
        # 更新位置
        pos_new = pos + v_new * dt
        
        return v_new, pos_new
    
    # 主循环
    for i in range(time_steps):
        for j in range(n_particles):
            velocities[j], positions[j] = boris_push(
                velocities[j], positions[j], dt, E_field, B_field, charge, mass
            )
        
        position_history[i] = positions.copy()
        velocity_history[i] = velocities.copy()
    
    return times, position_history, velocity_history

def turbulent_plasma_spectrum(k_range, L_outer, L_inner, magnetic_field=0.0, 
                             density=1.0, temperature=1e6):
    """
    计算湍流等离子体的能量谱
    
    基于Kolmogorov-理论和磁流体动力学修正，计算湍流等离子体的能量谱。
    
    Parameters:
    -----------
    k_range : array_like
        波数范围，单位：1/米
    L_outer : float
        外尺度(能量注入尺度)，单位：米
    L_inner : float
        内尺度(耗散尺度)，单位：米
    magnetic_field : float, optional
        磁场强度，单位：特斯拉，默认为0.0
    density : float, optional
        等离子体密度，单位：kg/m^3，默认为1.0
    temperature : float, optional
        等离子体温度，单位：开尔文，默认为1e6
        
    Returns:
    --------
    ndarray
        能量谱密度，单位：J/(m^3·(1/m))
    """
    # 确保k_range是numpy数组
    k = np.asarray(k_range)
    
    # 计算波数对应的尺度范围
    k_outer = 2 * np.pi / L_outer
    k_inner = 2 * np.pi / L_inner
    
    # 初始化能量谱
    spectrum = np.zeros_like(k, dtype=float)
    
    # 计算阿尔文速度
    alfven_velocity = 0
    if magnetic_field > 0:
        # 阿尔文速度 = B/sqrt(μ₀ρ)
        alfven_velocity = magnetic_field / np.sqrt(constants.mu_0 * density)
    
    # 计算声速
    sound_speed = np.sqrt(constants.k * temperature / (density / constants.m_p))
    
    # 计算能量谱
    for i, k_val in enumerate(k):
        if k_val < k_outer:
            # 能量注入区域
            spectrum[i] = k_val**2
        elif k_val < k_inner:
            # 惯性区域 - Kolmogorov谱
            if magnetic_field > 0 and alfven_velocity > sound_speed:
                # 强磁场下的MHD湍流 - Iroshnikov-Kraichnan谱
                spectrum[i] = k_val**(-3/2)
            else:
                # 标准Kolmogorov谱
                spectrum[i] = k_val**(-5/3)
        else:
            # 耗散区域
            spectrum[i] = k_val**(-3) * np.exp(-(k_val/k_inner - 1))
    
    # 归一化谱
    max_val = np.max(spectrum)
    if max_val > 0:
        spectrum /= max_val
    
    return spectrum

def continuity_equation_solver(velocity_field, density_field, grid_spacing, dt, 
                              boundary_condition='periodic'):
    """
    求解流体连续性方程
    
    使用有限差分方法求解连续性方程 ∂ρ/∂t + ∇·(ρv) = 0
    
    Parameters:
    -----------
    velocity_field : ndarray
        速度场，形状为(nx, ny, nz, 3)或(nx, ny, 3)
    density_field : ndarray
        密度场，形状为(nx, ny, nz)或(nx, ny)
    grid_spacing : float or tuple
        网格间距，单位：米
    dt : float
        时间步长，单位：秒
    boundary_condition : str, optional
        边界条件，可选'periodic'或'fixed'，默认为'periodic'
        
    Returns:
    --------
    ndarray
        更新后的密度场
    """
    # 确定问题维度
    ndim = density_field.ndim
    
    # 将grid_spacing转换为每个维度的间距
    if isinstance(grid_spacing, (int, float)):
        dx = dy = dz = float(grid_spacing)
    else:
        if ndim == 2:
            dx, dy = grid_spacing
            dz = 1.0  # 不使用
        else:
            dx, dy, dz = grid_spacing
    
    # 创建新的密度场
    new_density = density_field.copy()
    
    # 计算密度通量
    if ndim == 2:
        # 2D情况
        flux_x = density_field * velocity_field[..., 0]
        flux_y = density_field * velocity_field[..., 1]
        
        # 计算散度 ∇·(ρv)
        if boundary_condition == 'periodic':
            # 周期性边界条件
            div_x = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2 * dx)
            div_y = (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2 * dy)
        else:
            # 固定边界条件
            div_x = np.zeros_like(density_field)
            div_y = np.zeros_like(density_field)
            
            # 内部点使用中心差分
            div_x[1:-1, :] = (flux_x[2:, :] - flux_x[:-2, :]) / (2 * dx)
            div_y[:, 1:-1] = (flux_y[:, 2:] - flux_y[:, :-2]) / (2 * dy)
            
            # 边界点使用前向/后向差分
            div_x[0, :] = (flux_x[1, :] - flux_x[0, :]) / dx
            div_x[-1, :] = (flux_x[-1, :] - flux_x[-2, :]) / dx
            div_y[:, 0] = (flux_y[:, 1] - flux_y[:, 0]) / dy
            div_y[:, -1] = (flux_y[:, -1] - flux_y[:, -2]) / dy
        
        divergence = div_x + div_y
        
    else:
        # 3D情况
        flux_x = density_field * velocity_field[..., 0]
        flux_y = density_field * velocity_field[..., 1]
        flux_z = density_field * velocity_field[..., 2]
        
        # 计算散度 ∇·(ρv)
        if boundary_condition == 'periodic':
       
            # 周期性边界条件
            div_x = (np.roll(flux_x, -1, axis=0) - np.roll(flux_x, 1, axis=0)) / (2 * dx)
            div_y = (np.roll(flux_y, -1, axis=1) - np.roll(flux_y, 1, axis=1)) / (2 * dy)
            div_z = (np.roll(flux_z, -1, axis=2) - np.roll(flux_z, 1, axis=2)) / (2 * dz)
        else:
            # 固定边界条件
            div_x = np.zeros_like(density_field)
            div_y = np.zeros_like(density_field)
            div_z = np.zeros_like(density_field)
            
            # 内部点使用中心差分
            div_x[1:-1, :, :] = (flux_x[2:, :, :] - flux_x[:-2, :, :]) / (2 * dx)
            div_y[:, 1:-1, :] = (flux_y[:, 2:, :] - flux_y[:, :-2, :]) / (2 * dy)
            div_z[:, :, 1:-1] = (flux_z[:, :, 2:] - flux_z[:, :, :-2]) / (2 * dz)
            
            # 边界点使用前向/后向差分
            div_x[0, :, :] = (flux_x[1, :, :] - flux_x[0, :, :]) / dx
            div_x[-1, :, :] = (flux_x[-1, :, :] - flux_x[-2, :, :]) / dx
            div_y[:, 0, :] = (flux_y[:, 1, :] - flux_y[:, 0, :]) / dy
            div_y[:, -1, :] = (flux_y[:, -1, :] - flux_y[:, -2, :]) / dy
            div_z[:, :, 0] = (flux_z[:, :, 1] - flux_z[:, :, 0]) / dz
            div_z[:, :, -1] = (flux_z[:, :, -1] - flux_z[:, :, -2]) / dz
        
        divergence = div_x + div_y + div_z
    
    # 更新密度场
    new_density = density_field - dt * divergence
    
    # 确保密度非负
    new_density = np.maximum(new_density, 0.0)
    
    return new_density

def main():
    """
    主函数：演示如何使用工具函数求解流体等离子体问题
    """
    print("流体等离子体计算工具演示")
    print("-" * 50)
    
    # 示例1：圆管内流动速度分布
    print("\n示例1：圆管内流动速度分布")
    # 设置参数
    pipe_radius = 0.05  # 管道半径，单位：米
    flow_rate = 0.001   # 流量，单位：立方米/秒
    
    # 计算A点和B点的流速（根据图中所示，A点在上半部分，B点在下半部分）
    # 假设A点距离中心线0.02米，B点距离中心线0.01米
    r_A = 0.02
    r_B = 0.01
    
    # 计算流速
    v_A = pipe_flow_velocity(pipe_radius, r_A, flow_rate)
    v_B = pipe_flow_velocity(pipe_radius, r_B, flow_rate)
    
    print(f"A点流速: {v_A:.4f} m/s")
    print(f"B点流速: {v_B:.4f} m/s")
    
    if v_A < v_B:
        print("结论: v_A < v_B，因为A点距离管道中心更远，流速较小")
    else:
        print("结论: v_A > v_B，因为A点距离管道中心更近，流速较大")
    
    # 可视化速度分布
    r_values = np.linspace(-pipe_radius, pipe_radius, 100)
    v_values = pipe_flow_velocity(pipe_radius, r_values, flow_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, v_values)
    plt.scatter([r_A, r_B], [v_A, v_B], color=['red', 'blue'], s=100)
    plt.annotate('A', (r_A, v_A), xytext=(r_A+0.005, v_A+0.01), fontsize=12)
    plt.annotate('B', (r_B, v_B), xytext=(r_B+0.005, v_B+0.01), fontsize=12)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('管道径向位置 (m)')
    plt.ylabel('流速 (m/s)')
    plt.title('圆管内流体速度分布')
    
    # 保存第一个图形
    plt.savefig('pipe_flow_velocity.png', dpi=300, bbox_inches='tight')
    print("图形1已保存为 'pipe_flow_velocity.png'")
    
    # 示例2：磁流体动力学流动
    print("\n\n示例2：磁流体动力学流动")
    # 设置参数
    B_field = 0.5  # 磁场强度，单位：特斯拉
    current_density = 1000  # 电流密度，单位：A/m²
    fluid_density = 1000  # 流体密度，单位：kg/m³
    
    # 计算MHD条件下的流速
    v_A_mhd = mhd_flow_velocity(pipe_radius, r_A, B_field, current_density, fluid_density)
    v_B_mhd = mhd_flow_velocity(pipe_radius, r_B, B_field, current_density, fluid_density)
    
    print(f"MHD条件下A点流速: {v_A_mhd:.4f} m/s")
    print(f"MHD条件下B点流速: {v_B_mhd:.4f} m/s")
    
    # 可视化MHD速度分布
    v_mhd_values = mhd_flow_velocity(pipe_radius, r_values, B_field, current_density, fluid_density)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, v_values, label='无磁场')
    plt.plot(r_values, v_mhd_values, label='有磁场')
    plt.scatter([r_A, r_B], [v_A_mhd, v_B_mhd], color=['red', 'blue'], s=100)
    plt.annotate('A (MHD)', (r_A, v_A_mhd), xytext=(r_A+0.005, v_A_mhd+0.01), fontsize=12)
    plt.annotate('B (MHD)', (r_B, v_B_mhd), xytext=(r_B+0.005, v_B_mhd+0.01), fontsize=12)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('管道径向位置 (m)')
    plt.ylabel('流速 (m/s)')
    plt.title('磁流体动力学条件下的速度分布')
    plt.legend()
    
    # 保存第二个图形
    plt.savefig('mhd_flow_velocity.png', dpi=300, bbox_inches='tight')
    print("图形2已保存为 'mhd_flow_velocity.png'")
    
    # 示例3：湍流等离子体能量谱
    print("\n\n示例3：湍流等离子体能量谱")
    # 设置参数
    k_range = np.logspace(-3, 3, 1000)  # 波数范围
    L_outer = 1.0  # 外尺度，单位：米
    L_inner = 0.001  # 内尺度，单位：米
    
    # 计算不同磁场强度下的湍流谱
    spectrum_B0 = turbulent_plasma_spectrum(k_range, L_outer, L_inner, magnetic_field=0.0)
    spectrum_B1 = turbulent_plasma_spectrum(k_range, L_outer, L_inner, magnetic_field=0.1)
    spectrum_B5 = turbulent_plasma_spectrum(k_range, L_outer, L_inner, magnetic_field=0.5)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k_range, spectrum_B0, label='B = 0 T')
    plt.loglog(k_range, spectrum_B1, label='B = 0.1 T')
    plt.loglog(k_range, spectrum_B5, label='B = 0.5 T')
    
    # 标记理论幂律
    k_mid = np.sqrt(2*np.pi/L_outer * 2*np.pi/L_inner)
    y_mid = spectrum_B0[np.abs(k_range - k_mid).argmin()]
    
    plt.loglog([k_mid, k_mid*10], [y_mid, y_mid*10**(-5/3)], 'k--', label='k^(-5/3)')
    plt.loglog([k_mid, k_mid*10], [y_mid, y_mid*10**(-3/2)], 'k:', label='k^(-3/2)')
    
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.xlabel('波数 k (1/m)')
    plt.ylabel('能量谱密度 E(k)')
    plt.title('湍流等离子体能量谱')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存第三个图形
    plt.savefig('turbulent_plasma_spectrum.png', dpi=300, bbox_inches='tight')
    print("图形3已保存为 'turbulent_plasma_spectrum.png'")
    
    # 尝试显示图形
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图形: {e}")
        print("所有图形已保存到文件中")

if __name__ == "__main__":
    main()