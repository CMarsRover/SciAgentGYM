# Filename: electromagnetic_field_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def calculate_conical_conductor_potential(theta, alpha, phi_0=1.0):
    """
    计算无限长导体圆锥在导体平面上的电位分布。
    
    该函数基于拉普拉斯方程在球坐标系中的解，适用于轴对称电位分布问题。
    对于圆锥电极-平面电极系统，电位仅依赖于极角θ。
    
    Parameters:
    -----------
    theta : float or ndarray
        极角，单位为弧度，有效范围为[alpha, pi/2]
    alpha : float
        圆锥半夹角，单位为弧度，范围(0, pi/2)
    phi_0 : float, optional
        圆锥表面的电位值，默认为1.0
        
    Returns:
    --------
    float or ndarray
        给定极角处的电位值
    """
    # 确保输入参数在有效范围内
    if np.any(theta < alpha) or np.any(theta > np.pi/2):
        raise ValueError("极角theta必须在[alpha, pi/2]范围内")
    if alpha <= 0 or alpha >= np.pi/2:
        raise ValueError("圆锥半夹角alpha必须在(0, pi/2)范围内")
    
    # 计算电位分布
    potential = phi_0 * np.log(np.tan(theta/2)) / np.log(np.tan(alpha/2))
    return potential

def derive_potential_constants(alpha, phi_0=1.0, verbose=False):
    """
    标准推理过程：
    1) 从轴对称拉普拉斯角向方程出发： (1/sinθ) d/dθ ( sinθ dφ/dθ ) = 0
    2) 一次积分： sinθ dφ/dθ = C1 ⇒ dφ/dθ = C1 cscθ
    3) 二次积分： φ(θ) = C1 ∫cscθ dθ + C2 = C1 ln|tan(θ/2)| + C2
    4) 用边界条件 φ(π/2)=0, φ(α)=φ0 求 C1、C2。

    为体现“解常数”过程，不直接写死结果，而是以线性方程组形式解：
        [ ln tan(π/4)   1 ] [C1] = [ 0 ]
        [ ln tan(α/2)   1 ] [C2]   [ φ0]

    Parameters
    ----------
    alpha : float
        圆锥半夹角，弧度，(0, π/2)
    phi_0 : float
        圆锥表面的电位值
    verbose : bool
        若 True，打印推导与线性求解细节

    Returns
    -------
    tuple[float, float]
        (A, B) ≡ (C1, C2)
    """
    if alpha <= 0 or alpha >= np.pi/2:
        raise ValueError("圆锥半夹角alpha必须在(0, pi/2)范围内")

    # 构造两条边界条件：θ1=π/2, φ1=0；θ2=α, φ2=φ0
    t1 = np.log(np.tan(np.pi/4.0))  # = 0
    t2 = np.log(np.tan(alpha / 2.0))

    A_mat = np.array([[t1, 1.0],
                      [t2, 1.0]], dtype=float)
    b_vec = np.array([0.0, float(phi_0)], dtype=float)

    # 解线性方程组
    C1, C2 = np.linalg.solve(A_mat, b_vec)

    if verbose:
        print("推导过程：")
        print("(1/sinθ) d/dθ ( sinθ dφ/dθ ) = 0 ⇒ dφ/dθ = C1 cscθ ⇒ φ = C1 ln|tan(θ/2)| + C2")
        print("边界条件：φ(π/2)=0, φ(α)=φ0 ⇒ 线性方程 A·C=b")
        print(f"A=\n{A_mat}\nb={b_vec}")
        print(f"解得：C1={C1:.6g}, C2={C2:.6g}")

    return C1, C2

def potential_via_integration(theta, alpha, phi_0=1.0, verbose=False):
    """
    使用显式积分推导得到的解析式计算电位，并可选打印推导步骤与常数。

    Parameters
    ----------
    theta : float or ndarray
        极角（弧度），取值区间 [alpha, π/2]
    alpha : float
        圆锥半夹角（弧度），(0, π/2)
    phi_0 : float
        圆锥表面的电位值
    verbose : bool
        若为 True，则打印积分推导关键步骤与 A、B 常数

    Returns
    -------
    float or ndarray
        电位 φ(θ)
    """
    if np.any(theta < alpha) or np.any(theta > np.pi/2):
        raise ValueError("极角theta必须在[alpha, pi/2]范围内")

    A, B = derive_potential_constants(alpha, phi_0, verbose=verbose)

    return A * np.log(np.tan(theta / 2.0)) + B

def calculate_electric_field(r, theta, alpha, phi_0=1.0):
    """
    计算无限长导体圆锥在导体平面上的电场分布。
    
    电场是电位的负梯度，在球坐标系中分为径向和角向分量。
    由于电位只与θ有关，电场只有径向和角向分量。
    
    Parameters:
    -----------
    r : float or ndarray
        径向距离，单位为米
    theta : float or ndarray
        极角，单位为弧度，有效范围为[alpha, pi/2]
    alpha : float
        圆锥半夹角，单位为弧度，范围(0, pi/2)
    phi_0 : float, optional
        圆锥表面的电位值，默认为1.0
        
    Returns:
    --------
    tuple (E_r, E_theta)
        径向和角向电场分量
    """
    # 确保输入参数在有效范围内
    if np.any(theta < alpha) or np.any(theta > np.pi/2):
        raise ValueError("极角theta必须在[alpha, pi/2]范围内")
    
    # 计算电位对θ的导数
    d_phi_d_theta = phi_0 / (2 * np.sin(theta/2) * np.cos(theta/2) * np.log(np.tan(alpha/2)))

    # 将 r 广播到与 theta 相同的形状，确保返回的分量与 theta 同形
    theta_shape = np.shape(theta)
    r_array = np.asarray(r)
    if r_array.shape == ():
        r_b = np.ones(theta_shape, dtype=float) * float(r_array)
    else:
        r_b = np.broadcast_to(r_array, theta_shape)

    # 计算电场分量
    # 电场E = -∇φ，在球坐标系中:
    # E_r = -∂φ/∂r = 0 (因为φ不依赖于r)
    # E_theta = -(1/r)·∂φ/∂θ
    E_r = np.zeros_like(r_b, dtype=float)  # 径向分量为零
    E_theta = -d_phi_d_theta / r_b        # 角向分量
    
    return E_r, E_theta

def calculate_charge_density(r, theta, alpha, phi_0=1.0, epsilon_0=8.85e-12):
    """
    计算导体表面的电荷密度分布。
    
    根据高斯定理，导体表面的电荷密度等于ε₀乘以电场的法向分量。
    
    Parameters:
    -----------
    r : float or ndarray
        径向距离，单位为米
    theta : float or ndarray
        极角，单位为弧度
    alpha : float
        圆锥半夹角，单位为弧度
    phi_0 : float, optional
        圆锥表面的电位值，默认为1.0
    epsilon_0 : float, optional
        真空介电常数，默认为8.85e-12 F/m
        
    Returns:
    --------
    float or ndarray
        表面电荷密度，单位为C/m²
    """
    # 计算圆锥表面的电场
    _, E_theta_cone = calculate_electric_field(r, alpha, alpha, phi_0)
    
    # 计算圆锥表面的法向电场分量
    E_normal_cone = E_theta_cone * np.sin(np.pi/2 - alpha)
    
    # 计算平面表面的电场
    _, E_theta_plane = calculate_electric_field(r, np.pi/2, alpha, phi_0)
    
    # 计算平面表面的法向电场分量
    E_normal_plane = E_theta_plane
    
    # 根据给定的theta值确定返回哪个表面的电荷密度
    if np.isclose(theta, alpha):
        return epsilon_0 * E_normal_cone
    elif np.isclose(theta, np.pi/2):
        return epsilon_0 * E_normal_plane
    else:
        raise ValueError("theta必须等于alpha(圆锥表面)或pi/2(平面表面)")

def plot_potential_distribution(alpha, phi_0=1.0, resolution=100):
    """
    绘制电位分布图。
    
    Parameters:
    -----------
    alpha : float
        圆锥半夹角，单位为弧度
    phi_0 : float, optional
        圆锥表面的电位值，默认为1.0
    resolution : int, optional
        网格分辨率，默认为100
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建网格
    r_max = 10.0
    r = np.linspace(0.1, r_max, resolution)
    theta = np.linspace(alpha, np.pi/2, resolution)
    r_grid, theta_grid = np.meshgrid(r, theta)
    
    # 计算电位
    phi = calculate_conical_conductor_potential(theta_grid, alpha, phi_0)
    
    # 转换为直角坐标
    x = r_grid * np.sin(theta_grid)
    z = r_grid * np.cos(theta_grid)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制电位分布
    surf = ax.plot_surface(x, z, phi, cmap=cm.viridis, linewidth=0, antialiased=True)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='电位 (V)')
    
    # 设置标签
    ax.set_xlabel('r·sin(θ) (m)')
    ax.set_ylabel('r·cos(θ) (m)')
    ax.set_zlabel('电位 (V)')
    ax.set_title(f'圆锥-平面系统的电位分布 (α = {alpha:.2f} rad)')
    
    plt.tight_layout()
    plt.show()

def plot_electric_field(alpha, phi_0=1.0, resolution=20):
    """
    绘制电场分布图。
    
    Parameters:
    -----------
    alpha : float
        圆锥半夹角，单位为弧度
    phi_0 : float, optional
        圆锥表面的电位值，默认为1.0
    resolution : int, optional
        网格分辨率，默认为20
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建网格
    r = np.linspace(0.5, 5.0, resolution)
    theta = np.linspace(alpha + 0.01, np.pi/2 - 0.01, resolution)
    r_grid, theta_grid = np.meshgrid(r, theta)
    
    # 计算电场
    E_r, E_theta = calculate_electric_field(r_grid, theta_grid, alpha, phi_0)
    
    # 转换为直角坐标
    x = r_grid * np.sin(theta_grid)
    z = r_grid * np.cos(theta_grid)
    
    # 将电场分量从球坐标转换为直角坐标
    E_x = E_r * np.sin(theta_grid) + E_theta * np.cos(theta_grid)
    E_z = E_r * np.cos(theta_grid) - E_theta * np.sin(theta_grid)
    
    # 归一化电场向量以便可视化
    E_norm = np.sqrt(E_x**2 + E_z**2)
    E_x = E_x / E_norm
    E_z = E_z / E_norm
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制电场向量
    plt.quiver(x, z, E_x, E_z, E_norm, cmap=plt.cm.viridis, scale=30)
    
    # 绘制圆锥和平面
    cone_x = np.linspace(0, 5.0 * np.sin(alpha), 100)
    cone_z = np.linspace(0, 5.0 * np.cos(alpha), 100)
    plt.plot(cone_x, cone_z, 'r-', linewidth=2, label='圆锥表面')
    plt.plot([-5, 5], [0, 0], 'k-', linewidth=2, label='平面')
    
    # 设置图形属性
    plt.colorbar(label='电场强度 (V/m)')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title(f'圆锥-平面系统的电场分布 (α = {alpha:.2f} rad)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数：演示如何使用工具函数求解导体圆锥-平面系统的电位分布问题
    """
    # 问题参数定义
    alpha = np.pi/6  # 圆锥半夹角为30度
    phi_0 = 100.0    # 圆锥电位为100V
    
    # 计算并打印特定位置的电位值（两种方法对比）
    theta_values = np.array([alpha, np.pi/4, np.pi/3, np.pi/2])
    potential_values_formula = calculate_conical_conductor_potential(theta_values, alpha, phi_0)
    A_const, B_const = derive_potential_constants(alpha, phi_0, verbose=True)
    potential_values_integral = potential_via_integration(theta_values, alpha, phi_0)
    
    print("电位分布计算结果:")
    print("-" * 50)
    print(f"圆锥半夹角 α = {alpha:.4f} rad ({alpha*180/np.pi:.1f}°)")
    print(f"圆锥电位 φ₀ = {phi_0:.1f} V")
    print("-" * 50)
    print("θ (rad) | θ (°)   | φ(公式) (V) | φ(积分) (V) | 差值")
    print("-" * 50)
    for i, theta in enumerate(theta_values):
        diff = potential_values_formula[i] - potential_values_integral[i]
        print(f"{theta:.4f}   | {theta*180/np.pi:6.1f} | {potential_values_formula[i]:.4f}     | {potential_values_integral[i]:.4f}     | {diff: .2e}")
    
    # 计算并打印特定位置的电场强度
    r_value = 1.0  # 距离为1米
    E_r, E_theta = calculate_electric_field(r_value, theta_values, alpha, phi_0)
    
    print("\n电场强度计算结果 (r = 1.0 m):")
    print("-" * 50)
    print("θ (rad) | θ (°)   | E_r (V/m)  | E_θ (V/m)")
    print("-" * 50)
    for i, theta in enumerate(theta_values):
        print(f"{theta:.4f}   | {theta*180/np.pi:6.1f} | {E_r[i]:10.4f} | {E_theta[i]:10.4f}")
    
    # 可视化电位分布
    # 注释掉以下行以禁用可视化
    # plot_potential_distribution(alpha, phi_0)
    
    # 可视化电场分布
    # 注释掉以下行以禁用可视化
    # plot_electric_field(alpha, phi_0)

if __name__ == "__main__":
    main()