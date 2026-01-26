# Filename: electromagnetic_field_calculator.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.constants import mu_0, pi

def calculate_mutual_inductance_wire_loop(geometry_func, wire_position, current_direction, 
                                         integration_limits, num_points=1000):
    """
    计算直导线与任意形状闭合回路之间的互感系数。
    
    基于比奥-萨伐尔定律和法拉第电磁感应定律，计算直导线产生的磁场
    通过闭合回路的磁通量，从而得到互感系数。
    
    Parameters:
    -----------
    geometry_func : callable
        描述闭合回路几何形状的函数，接收参数t(参数方程的参数)，
        返回回路上对应点的坐标(x,y,z)和切向量(dx,dy,dz)
    wire_position : tuple
        直导线的位置坐标 (x, y, z)
    current_direction : tuple
        直导线的电流方向单位向量 (dx, dy, dz)
    integration_limits : tuple
        积分参数的上下限 (t_min, t_max)
    num_points : int, optional
        数值积分使用的点数，默认为1000
        
    Returns:
    --------
    float
        互感系数，单位为亨利(H)
    """
    t_min, t_max = integration_limits
    t_values = np.linspace(t_min, t_max, num_points)
    dt = (t_max - t_min) / (num_points - 1)
    
    # 归一化电流方向
    current_dir = np.array(current_direction)
    current_dir = current_dir / np.linalg.norm(current_dir)
    
    # 计算每个点的贡献
    mutual_inductance = 0
    for t in t_values:
        # 获取回路上点的位置和方向
        point, direction = geometry_func(t)
        
        # 计算从直导线到回路上点的向量
        r_vec = np.array(point) - np.array(wire_position)
        r_mag = np.linalg.norm(r_vec)
        
        # 计算直导线在该点产生的磁场方向
        B_direction = np.cross(current_dir, r_vec/r_mag)
        
        # 计算磁场强度系数 (不包含μ0/4π和电流I)
        B_magnitude = 1 / r_mag
        
        # 计算磁场与回路微元的标量积
        dl = np.array(direction) * dt
        contribution = np.dot(B_direction, dl) * B_magnitude
        
        mutual_inductance += contribution
    
    # 乘以系数 μ0/4π
    mutual_inductance *= mu_0 / (4 * pi)
    
    return mutual_inductance

def calculate_mutual_inductance_wire_triangle(d, a, analytical=True):
    """
    计算无限长直导线与等边三角形线框之间的互感。
    
    Parameters:
    -----------
    d : float
        直导线到三角形最近顶点的距离，单位为米
    a : float
        等边三角形的边长，单位为米
    analytical : bool, optional
        是否使用解析解，默认为True。若为False则使用数值积分
        
    Returns:
    --------
    float
        互感系数，单位为亨利(H)
    """
    if analytical:
        # 解析解公式
        M = (np.sqrt(3) * mu_0) / (3 * pi) * (d * np.log((2*d + np.sqrt(3)*a)/d) - (np.sqrt(3)/2) * a)
        return M
    else:
        # 数值积分方法
        def triangle_geometry(t):
            """参数方程描述等边三角形，t∈[0,3]对应三条边"""
            if t < 1:  # 第一条边
                x = d + t * a
                y = 0
                dx = a
                dy = 0
            elif t < 2:  # 第二条边
                t = t - 1
                x = d + a - t * (a/2)
                y = t * (a * np.sqrt(3)/2)
                dx = -a/2
                dy = a * np.sqrt(3)/2
            else:  # 第三条边
                t = t - 2
                x = d + a/2 - t * (a/2)
                y = a * np.sqrt(3)/2 - t * (a * np.sqrt(3)/2)
                dx = -a/2
                dy = -a * np.sqrt(3)/2
            
            return (x, y, 0), (dx, dy, 0)
        
        wire_position = (0, 0, 0)
        current_direction = (0, 0, 1)  # z方向
        integration_limits = (0, 3)
        
        return calculate_mutual_inductance_wire_loop(
            triangle_geometry, wire_position, current_direction, integration_limits)

def calculate_electromagnetic_field(source_func, observation_points, params):
    """
    计算电磁场分布。
    
    Parameters:
    -----------
    source_func : callable
        描述电磁场源的函数，接收参数和观测点坐标，返回场强
    observation_points : ndarray
        观测点的坐标数组，形状为(n, 3)，表示n个点的(x,y,z)坐标
    params : dict
        传递给source_func的参数字典
        
    Returns:
    --------
    ndarray
        场强数组，形状与observation_points相同
    """
    field = np.zeros_like(observation_points)
    for i, point in enumerate(observation_points):
        field[i] = source_func(point, **params)
    return field

def infinite_wire_magnetic_field(point, wire_position, current_direction, current=1.0):
    """
    计算无限长直导线在给定点产生的磁场。
    
    Parameters:
    -----------
    point : tuple or ndarray
        观测点的坐标 (x, y, z)
    wire_position : tuple or ndarray
        导线的位置坐标 (x, y, z)，表示导线上的一点
    current_direction : tuple or ndarray
        导线的方向单位向量 (dx, dy, dz)
    current : float, optional
        电流大小，单位为安培，默认为1.0
        
    Returns:
    --------
    ndarray
        磁场向量 (Bx, By, Bz)，单位为特斯拉
    """
    r_vec = np.array(point) - np.array(wire_position)
    
    # 计算从导线到观测点的垂直距离
    current_dir = np.array(current_direction)
    current_dir = current_dir / np.linalg.norm(current_dir)
    
    # 投影r_vec到current_direction上
    proj = np.dot(r_vec, current_dir) * current_dir
    
    # 计算垂直分量
    r_perp = r_vec - proj
    r_perp_mag = np.linalg.norm(r_perp)
    
    if r_perp_mag < 1e-10:  # 避免除以零
        return np.zeros(3)
    
    # 计算磁场方向和大小
    B_direction = np.cross(current_dir, r_perp) / r_perp_mag
    B_magnitude = (mu_0 * current) / (2 * pi * r_perp_mag)
    
    return B_magnitude * B_direction / np.linalg.norm(B_direction)

def calculate_electromagnetic_scattering(incident_field_func, scatterer_func, 
                                        observation_points, params):
    """
    计算电磁波散射问题。
    
    Parameters:
    -----------
    incident_field_func : callable
        入射场函数，接收观测点坐标和参数，返回入射场
    scatterer_func : callable
        散射体函数，接收入射场、观测点和参数，返回散射场
    observation_points : ndarray
        观测点的坐标数组
    params : dict
        参数字典
        
    Returns:
    --------
    tuple
        (入射场, 散射场, 总场)
    """
    incident_field = np.zeros_like(observation_points, dtype=complex)
    scattered_field = np.zeros_like(observation_points, dtype=complex)
    
    for i, point in enumerate(observation_points):
        incident_field[i] = incident_field_func(point, **params)
        scattered_field[i] = scatterer_func(incident_field[i], point, **params)
    
    total_field = incident_field + scattered_field
    return incident_field, scattered_field, total_field

def calculate_coupled_electromagnetic_thermal(em_field_func, thermal_response_func, 
                                             material_properties, geometry, time_steps):
    """
    计算电磁场与热场耦合问题。
    
    Parameters:
    -----------
    em_field_func : callable
        电磁场计算函数
    thermal_response_func : callable
        热响应计算函数
    material_properties : dict
        材料属性字典
    geometry : dict
        几何形状描述
    time_steps : ndarray
        时间步长数组
        
    Returns:
    --------
    tuple
        (电磁场随时间演化, 温度场随时间演化)
    """
    em_fields = []
    thermal_fields = []
    
    # 初始条件
    current_thermal = np.zeros(geometry['mesh_size'])
    
    for t in time_steps:
        # 计算当前电磁场
        current_em = em_field_func(geometry, material_properties, current_thermal)
        
        # 计算热响应
        current_thermal = thermal_response_func(current_em, current_thermal, 
                                               material_properties, t)
        
        em_fields.append(current_em.copy())
        thermal_fields.append(current_thermal.copy())
    
    return np.array(em_fields), np.array(thermal_fields)

def main():
    """
    主函数：演示如何使用工具函数求解互感问题
    """
    # 问题参数定义
    d = 0.1  # 直导线到三角形最近顶点的距离，单位为米
    a = 0.2  # 等边三角形的边长，单位为米
    
    # 计算互感系数
    M_analytical = calculate_mutual_inductance_wire_triangle(d, a, analytical=True)
    M_numerical = calculate_mutual_inductance_wire_triangle(d, a, analytical=False)
    
    print(f"解析解计算的互感系数: {M_analytical:.6e} H")
    print(f"数值积分计算的互感系数: {M_numerical:.6e} H")
    
    # 验证公式: M = (sqrt(3)*μ0)/(3π) * (d*ln((2d+sqrt(3)*a)/d) - (sqrt(3)/2)*a)
    formula = (np.sqrt(3) * mu_0) / (3 * pi) * (d * np.log((2*d + np.sqrt(3)*a)/d) - (np.sqrt(3)/2) * a)
    print(f"公式计算的互感系数: {formula:.6e} H")
    
    # 可视化磁场分布（可选）
    visualize_magnetic_field = True
    if visualize_magnetic_field:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建网格
        x = np.linspace(-0.1, 0.5, 50)
        y = np.linspace(-0.2, 0.4, 50)
        X, Y = np.meshgrid(x, y)
        
        # 计算每个点的磁场
        B_field = np.zeros((len(y), len(x), 3))
        for i in range(len(y)):
            for j in range(len(x)):
                B_field[i, j] = infinite_wire_magnetic_field(
                    (X[i, j], Y[i, j], 0), 
                    (0, 0, 0), 
                    (0, 0, 1), 
                    current=1.0
                )
        
        # 计算磁场强度和方向
        B_magnitude = np.sqrt(B_field[:,:,0]**2 + B_field[:,:,1]**2 + B_field[:,:,2]**2)
        B_x = B_field[:,:,0]
        B_y = B_field[:,:,1]
        
        # 绘制磁场
        plt.figure(figsize=(10, 8))
        plt.streamplot(X, Y, B_x, B_y, density=1.5, color=B_magnitude, 
                      linewidth=1, cmap='viridis', arrowsize=1.5)
        
        # 绘制导线位置
        plt.plot([0, 0], [-0.2, 0.4], 'k-', linewidth=2, label='无限长直导线')
        
        # 绘制三角形
        triangle_x = [d, d+a, d+a/2, d]
        triangle_y = [0, 0, a*np.sqrt(3)/2, 0]
        plt.plot(triangle_x, triangle_y, 'r-', linewidth=2, label='等边三角形线框')
        
        plt.colorbar(label='磁场强度 (T)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('无限长直导线与等边三角形线框的磁场分布')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()