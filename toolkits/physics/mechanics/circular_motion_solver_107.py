# Filename: circular_motion_solver.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

def calculate_tension_circular_motion(m1, m2, L1, L2, omega):
    """
    计算两个质点在水平面上做匀速圆周运动时绳索上的张力。
    
    该函数基于牛顿第二定律和向心力原理，计算连接两个质点的绳索上的张力。
    在匀速圆周运动中，向心力由绳索的张力提供。
    
    Parameters:
    -----------
    m1 : float
        第一个质点的质量，单位：kg
    m2 : float
        第二个质点的质量，单位：kg
    L1 : float
        第一段绳索的长度（从固定点O到质点m1），单位：m
    L2 : float
        第二段绳索的长度（从质点m1到质点m2），单位：m
    omega : float
        角速度，单位：rad/s
        
    Returns:
    --------
    tuple (T1, T12)
        T1: 第一段绳索（O到m1）上的张力，单位：N
        T12: 第二段绳索（m1到m2）上的张力，单位：N
        注意：T12 = T21，即作用在m1和m2上的张力大小相等
    """
    # 计算各质点到旋转中心的距离
    r1 = L1  # m1到旋转中心O的距离
    r2 = L1 + L2  # m2到旋转中心O的距离
    
    # 计算向心加速度
    a1 = omega**2 * r1  # m1的向心加速度
    a2 = omega**2 * r2  # m2的向心加速度
    
    # 计算第二段绳索上的张力（作用在m1和m2之间）
    # 这个张力由m2的向心力决定
    T12 = m2 * omega**2 * r2
    
    # 计算第一段绳索上的张力
    # 这个张力需要提供m1的向心力和传递给m2的向心力
    T1 = m1 * omega**2 * r1 + T12
    
    return T1, T12

def simulate_circular_motion(m1, m2, L1, L2, omega, duration=10, fps=30):
    """
    模拟两个质点在水平面上做匀速圆周运动的运动轨迹。
    
    Parameters:
    -----------
    m1 : float
        第一个质点的质量，单位：kg
    m2 : float
        第二个质点的质量，单位：kg
    L1 : float
        第一段绳索的长度，单位：m
    L2 : float
        第二段绳索的长度，单位：m
    omega : float
        角速度，单位：rad/s
    duration : float, optional
        模拟持续时间，单位：s，默认为10秒
    fps : int, optional
        每秒帧数，默认为30
        
    Returns:
    --------
    tuple (positions_m1, positions_m2)
        positions_m1: 质点m1的位置坐标数组，形状为(n, 2)
        positions_m2: 质点m2的位置坐标数组，形状为(n, 2)
        其中n是时间步数，每行包含(x, y)坐标
    """
    # 计算时间步长和总步数
    dt = 1.0 / fps
    steps = int(duration * fps)
    
    # 初始化位置数组
    positions_m1 = np.zeros((steps, 2))
    positions_m2 = np.zeros((steps, 2))
    
    # 计算每个时间步的位置
    for i in range(steps):
        t = i * dt
        angle = omega * t
        
        # 计算m1的位置（绕原点旋转）
        positions_m1[i, 0] = L1 * np.cos(angle)
        positions_m1[i, 1] = L1 * np.sin(angle)
        
        # 计算m2的位置（绕原点旋转）
        positions_m2[i, 0] = (L1 + L2) * np.cos(angle)
        positions_m2[i, 1] = (L1 + L2) * np.sin(angle)
    
    return positions_m1, positions_m2

def analyze_forces(m1, m2, L1, L2, omega):
    """
    分析圆周运动中的力学关系，计算张力并验证力平衡。
    
    Parameters:
    -----------
    m1 : float
        第一个质点的质量，单位：kg
    m2 : float
        第二个质点的质量，单位：kg
    L1 : float
        第一段绳索的长度，单位：m
    L2 : float
        第二段绳索的长度，单位：m
    omega : float
        角速度，单位：rad/s
        
    Returns:
    --------
    dict
        包含力学分析结果的字典，包括张力、向心力等
    """
    # 计算张力
    T1, T12 = calculate_tension_circular_motion(m1, m2, L1, L2, omega)
    
    # 计算各质点到旋转中心的距离
    r1 = L1
    r2 = L1 + L2
    
    # 计算向心力
    F_centripetal_m1 = m1 * omega**2 * r1
    F_centripetal_m2 = m2 * omega**2 * r2
    
    # 验证力平衡
    # 对m1：T1 - T12 = m1 * omega^2 * r1
    balance_m1 = abs((T1 - T12) - F_centripetal_m1) < 1e-10
    
    # 对m2：T12 = m2 * omega^2 * r2
    balance_m2 = abs(T12 - F_centripetal_m2) < 1e-10
    
    return {
        'T1': T1,
        'T12': T12,
        'F_centripetal_m1': F_centripetal_m1,
        'F_centripetal_m2': F_centripetal_m2,
        'balance_m1': balance_m1,
        'balance_m2': balance_m2
    }

def visualize_motion(m1, m2, L1, L2, omega, duration=5, fps=30):
    """
    可视化两个质点的圆周运动。
    
    Parameters:
    -----------
    m1 : float
        第一个质点的质量，单位：kg
    m2 : float
        第二个质点的质量，单位：kg
    L1 : float
        第一段绳索的长度，单位：m
    L2 : float
        第二段绳索的长度，单位：m
    omega : float
        角速度，单位：rad/s
    duration : float, optional
        模拟持续时间，单位：s，默认为5秒
    fps : int, optional
        每秒帧数，默认为30
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算张力
    T1, T12 = calculate_tension_circular_motion(m1, m2, L1, L2, omega)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # 设置坐标轴范围
    max_radius = L1 + L2 + 1
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    
    # 绘制固定点O
    ax.plot(0, 0, 'ko', markersize=8, label='固定点O')
    
    # 绘制圆形轨迹
    circle1 = Circle((0, 0), L1, fill=False, linestyle='--', color='gray')
    circle2 = Circle((0, 0), L1 + L2, fill=False, linestyle='--', color='gray')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # 初始化质点和绳索
    point_m1, = ax.plot([], [], 'rs', markersize=10, label=f'm1 = {m1} kg')
    point_m2, = ax.plot([], [], 'bs', markersize=10, label=f'm2 = {m2} kg')
    line1, = ax.plot([], [], 'k-', lw=2, label=f'T1 = {T1:.2f} N')
    line2, = ax.plot([], [], 'k-', lw=2, label=f'T12 = {T12:.2f} N')
    
    # 添加标题和图例
    ax.set_title(f'两质点匀速圆周运动模拟 (ω = {omega} rad/s)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # 添加坐标轴标签
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    
    # 动画更新函数
    def update(frame):
        t = frame / fps
        angle = omega * t
        
        # 更新m1的位置
        x1 = L1 * np.cos(angle)
        y1 = L1 * np.sin(angle)
        point_m1.set_data(x1, y1)
        
        # 更新m2的位置
        x2 = (L1 + L2) * np.cos(angle)
        y2 = (L1 + L2) * np.sin(angle)
        point_m2.set_data(x2, y2)
        
        # 更新绳索
        line1.set_data([0, x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        
        return point_m1, point_m2, line1, line2
    
    # 创建动画
    frames = int(duration * fps)
    ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def main():
    """
    主函数：演示如何使用工具函数求解当前问题
    """
    # 问题参数定义
    m1 = 2.0  # 质点m1的质量，单位：kg
    m2 = 3.0  # 质点m2的质量，单位：kg
    L1 = 1.0  # 第一段绳索长度，单位：m
    L2 = 1.5  # 第二段绳索长度，单位：m
    omega = 2.0  # 角速度，单位：rad/s
    
    # 计算张力
    T1, T12 = calculate_tension_circular_motion(m1, m2, L1, L2, omega)
    
    # 输出结果
    print("\n圆周运动张力计算结果:")
    print("-" * 40)
    print(f"质点m1质量: {m1} kg")
    print(f"质点m2质量: {m2} kg")
    print(f"第一段绳索长度L1: {L1} m")
    print(f"第二段绳索长度L2: {L2} m")
    print(f"角速度ω: {omega} rad/s")
    print("-" * 40)
    print(f"第一段绳索张力T1: {T1:.4f} N")
    print(f"第二段绳索张力T12: {T12:.4f} N")
    print("-" * 40)
    
    # 验证计算结果与理论公式
    T1_theory = omega**2 * (m1*L1 + m2*(L1+L2))
    T12_theory = omega**2 * m2 * (L1+L2)
    
    print("\n验证计算结果与理论公式:")
    print(f"理论公式T1 = ω²[m₁L₁ + m₂(L₁+L₂)] = {T1_theory:.4f} N")
    print(f"理论公式T12 = ω²m₂(L₁+L₂) = {T12_theory:.4f} N")
    print(f"计算结果与理论公式误差T1: {abs(T1-T1_theory):.10f} N")
    print(f"计算结果与理论公式误差T12: {abs(T12-T12_theory):.10f} N")
    
    # 进行力学分析
    force_analysis = analyze_forces(m1, m2, L1, L2, omega)
    print("\n力学平衡分析:")
    print(f"m1的向心力: {force_analysis['F_centripetal_m1']:.4f} N")
    print(f"m2的向心力: {force_analysis['F_centripetal_m2']:.4f} N")
    print(f"m1力平衡验证: {'通过' if force_analysis['balance_m1'] else '不通过'}")
    print(f"m2力平衡验证: {'通过' if force_analysis['balance_m2'] else '不通过'}")
    
    # # 可视化模拟（可选，取消注释以运行）
    # visualize_motion(m1, m2, L1, L2, omega)

if __name__ == "__main__":
    main()