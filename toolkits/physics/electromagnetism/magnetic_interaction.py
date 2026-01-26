#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
磁体相互作用计算模块

该模块提供了计算不同形状磁体之间磁力的函数。
适用于凝聚态物理中磁性材料的相互作用研究。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def calculate_magnetic_moment(volume, magnetization):
    """
    计算磁体的磁矩
    
    Parameters:
    -----------
    volume : float
        磁体的体积，单位为m³
    magnetization : float
        磁体的磁化强度，单位为A/m
        
    Returns:
    --------
    float
        磁矩，单位为A·m²
    """
    return volume * magnetization


def calculate_magnetic_force(m1, m2, r, mu0=4*np.pi*1e-7):
    """
    计算两个磁偶极子之间的力
    
    Parameters:
    -----------
    m1 : ndarray
        第一个磁体的磁矩向量，单位为A·m²
    m2 : ndarray
        第二个磁体的磁矩向量，单位为A·m²
    r : ndarray
        从第一个磁体指向第二个磁体的位置向量，单位为m
    mu0 : float
        真空磁导率，默认为4π×10⁻⁷ H/m
        
    Returns:
    --------
    numpy.ndarray
        磁力向量，单位为N。正值表示吸引力，负值表示排斥力
    """
    m1 = np.array(m1)
    m2 = np.array(m2)
    r = np.array(r)
    r_mag = np.linalg.norm(r)
    r_hat = r / r_mag
    
    # 磁偶极子相互作用力公式
    # F = (3*mu0/(4*pi*r^4)) * [r(m1·m2) - 5r(r·m1)(r·m2) + (r·m2)m1 + (r·m1)m2]
    # 简化为向量形式
    
    m1_dot_m2 = np.dot(m1, m2)
    m1_dot_r = np.dot(m1, r_hat)
    m2_dot_r = np.dot(m2, r_hat)
    
    term1 = m1_dot_m2 * r_hat
    term2 = -5 * m1_dot_r * m2_dot_r * r_hat
    term3 = m2_dot_r * m1
    term4 = m1_dot_r * m2
    
    force = (3 * mu0 / (4 * np.pi * r_mag**4)) * (term1 + term2 + term3 + term4)
    
    return force


def compare_magnetic_pairs(shape1_pair, shape2_pair, distance):
    """
    比较两对磁体之间的磁力大小
    
    Parameters:
    -----------
    shape1_pair : tuple
        第一对磁体的描述 (形状, 体积1, 体积2, 磁化强度1, 磁化强度2, 相对方向)
        形状可以是 'bar', 'disc', 'sphere' 等
    shape2_pair : tuple
        第二对磁体的描述，格式同上
    distance : float
        每对磁体之间的距离，单位为m
        
    Returns:
    --------
    dict
        包含两对磁体的磁力大小比较结果
    """
    # 提取参数
    shape1, vol1_1, vol1_2, mag1_1, mag1_2, align1 = shape1_pair
    shape2, vol2_1, vol2_2, mag2_1, mag2_2, align2 = shape2_pair
    
    # 计算磁矩
    m1_1 = calculate_magnetic_moment(vol1_1, mag1_1)
    m1_2 = calculate_magnetic_moment(vol1_2, mag1_2)
    m2_1 = calculate_magnetic_moment(vol2_1, mag2_1)
    m2_2 = calculate_magnetic_moment(vol2_2, mag2_2)
    
    # 设置磁矩方向
    # 假设磁体沿z轴排列
    m1_1_vec = np.array([0, 0, m1_1])
    m1_2_vec = np.array([0, 0, m1_2 * align1])  # align1为1表示同向，-1表示反向
    
    m2_1_vec = np.array([0, 0, m2_1])
    m2_2_vec = np.array([0, 0, m2_2 * align2])
    
    # 计算位置向量
    r1 = np.array([0, 0, distance])
    r2 = np.array([0, 0, distance])
    
    # 计算力
    force1 = calculate_magnetic_force(m1_1_vec, m1_2_vec, r1)
    force2 = calculate_magnetic_force(m2_1_vec, m2_2_vec, r2)
    
    # 计算力的大小
    force1_mag = np.linalg.norm(force1)
    force2_mag = np.linalg.norm(force2)
    
    # 判断力的性质（吸引或排斥）
    force1_type = "吸引" if force1[2] > 0 else "排斥"
    force2_type = "吸引" if force2[2] > 0 else "排斥"
    
    return {
        "pair1": {
            "shape": shape1,
            "force_magnitude": force1_mag,
            "force_type": force1_type
        },
        "pair2": {
            "shape": shape2,
            "force_magnitude": force2_mag,
            "force_type": force2_type
        },
        "comparison": "pair1" if force1_mag > force2_mag else "pair2"
    }


def visualize_magnetic_interaction(shape, volume, magnetization, distance, alignment, save_path=None):
    """
    可视化磁体之间的相互作用
    
    Parameters:
    -----------
    shape : str
        磁体的形状，如'bar', 'disc', 'sphere'等
    volume : tuple
        两个磁体的体积，单位为m³
    magnetization : tuple
        两个磁体的磁化强度，单位为A/m
    distance : float
        磁体之间的距离，单位为m
    alignment : int
        磁体的相对方向，1表示同向，-1表示反向
    save_path : str
        保存路径，如果为None则显示图形
        
    Returns:
    --------
    None
        显示磁体和磁场线的3D图像
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置磁体位置
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([0, 0, distance])
    
    # 根据形状绘制不同的磁体
    if shape == 'bar':
        # 条形磁体 - 绘制矩形
        bar_length = distance * 0.2
        bar_width = distance * 0.1
        # 磁体1
        x1 = [pos1[0] - bar_width/2, pos1[0] + bar_width/2, pos1[0] + bar_width/2, pos1[0] - bar_width/2]
        y1 = [pos1[1] - bar_length/2, pos1[1] - bar_length/2, pos1[1] + bar_length/2, pos1[1] + bar_length/2]
        z1 = [pos1[2]] * 4
        ax.plot(x1 + [x1[0]], y1 + [y1[0]], z1 + [z1[0]], 'r-', linewidth=3, label='Bar Magnet 1')
        # 磁体2
        x2 = [pos2[0] - bar_width/2, pos2[0] + bar_width/2, pos2[0] + bar_width/2, pos2[0] - bar_width/2]
        y2 = [pos2[1] - bar_length/2, pos2[1] - bar_length/2, pos2[1] + bar_length/2, pos2[1] + bar_length/2]
        z2 = [pos2[2]] * 4
        ax.plot(x2 + [x2[0]], y2 + [y2[0]], z2 + [z2[0]], 'b-', linewidth=3, label='Bar Magnet 2')
        
    elif shape == 'disc':
        # 圆盘形磁体 - 绘制圆形
        radius = distance * 0.15
        theta = np.linspace(0, 2*np.pi, 50)
        # 磁体1
        x1 = pos1[0] + radius * np.cos(theta)
        y1 = pos1[1] + radius * np.sin(theta)
        z1 = [pos1[2]] * len(theta)
        ax.plot(x1, y1, z1, 'r-', linewidth=3, label='Disc Magnet 1')
        # 磁体2
        x2 = pos2[0] + radius * np.cos(theta)
        y2 = pos2[1] + radius * np.sin(theta)
        z2 = [pos2[2]] * len(theta)
        ax.plot(x2, y2, z2, 'b-', linewidth=3, label='Disc Magnet 2')
        
    elif shape == 'sphere':
        # 球形磁体 - 绘制球形
        radius = distance * 0.1
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        # 磁体1
        x1 = pos1[0] + radius * np.outer(np.cos(u), np.sin(v))
        y1 = pos1[1] + radius * np.outer(np.sin(u), np.sin(v))
        z1 = pos1[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x1, y1, z1, color='red', alpha=0.7, label='Sphere Magnet 1')
        # 磁体2
        x2 = pos2[0] + radius * np.outer(np.cos(u), np.sin(v))
        y2 = pos2[1] + radius * np.outer(np.sin(u), np.sin(v))
        z2 = pos2[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x2, y2, z2, color='blue', alpha=0.7, label='Sphere Magnet 2')
        
    else:
        # 默认用点表示
        ax.scatter(pos1[0], pos1[1], pos1[2], color='red', s=100, label=f'{shape} Magnet 1')
        ax.scatter(pos2[0], pos2[1], pos2[2], color='blue', s=100, label=f'{shape} Magnet 2')
    
    # 计算磁矩
    m1 = calculate_magnetic_moment(volume[0], magnetization[0])
    m2 = calculate_magnetic_moment(volume[1], magnetization[1]) * alignment
    
    # 绘制磁矩方向
    arrow_length = distance * 0.3
    ax.quiver(pos1[0], pos1[1], pos1[2], 0, 0, arrow_length, color='red')
    ax.quiver(pos2[0], pos2[1], pos2[2], 0, 0, arrow_length * np.sign(alignment), color='blue')
    
    # 计算一些磁场线点（简化）
    phi = np.linspace(0, 2*np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    
    # 绘制简化的磁场线
    for p in np.linspace(0.2, 0.8, 5):
        mid_point = pos1 + p * (pos2 - pos1)
        radius = distance * 0.2
        
        x = mid_point[0] + radius * np.outer(np.cos(phi), np.sin(theta))
        y = mid_point[1] + radius * np.outer(np.sin(phi), np.sin(theta))
        z = mid_point[2] + radius * np.outer(np.ones(np.size(phi)), np.cos(theta))
        
        ax.plot_surface(x, y, z, color='cyan', alpha=0.1)
    
    # 设置图形属性
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Magnetic Interaction - {shape} magnets')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Magnetic interaction plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def parse_args():
    """
    解析命令行参数
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='磁体相互作用计算工具')
    
    parser.add_argument('--shape1', type=str, default='bar', help='第一对磁体的形状')
    parser.add_argument('--shape2', type=str, default='disc', help='第二对磁体的形状')
    parser.add_argument('--volume1', type=float, nargs=2, default=[1e-6, 1e-6], 
                        help='第一对磁体的体积 (m³)')
    parser.add_argument('--volume2', type=float, nargs=2, default=[1e-6, 1e-6], 
                        help='第二对磁体的体积 (m³)')
    parser.add_argument('--magnetization1', type=float, nargs=2, default=[1e6, 1e6], 
                        help='第一对磁体的磁化强度 (A/m)')
    parser.add_argument('--magnetization2', type=float, nargs=2, default=[1e6, 1e6], 
                        help='第二对磁体的磁化强度 (A/m)')
    parser.add_argument('--align1', type=int, default=1, choices=[-1, 1], 
                        help='第一对磁体的相对方向 (1:同向, -1:反向)')
    parser.add_argument('--align2', type=int, default=1, choices=[-1, 1], 
                        help='第二对磁体的相对方向 (1:同向, -1:反向)')
    parser.add_argument('--distance', type=float, default=0.01, 
                        help='每对磁体之间的距离 (m)')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否可视化结果')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 构建磁体对的描述
    shape1_pair = (args.shape1, args.volume1[0], args.volume1[1], 
                  args.magnetization1[0], args.magnetization1[1], args.align1)
    
    shape2_pair = (args.shape2, args.volume2[0], args.volume2[1], 
                  args.magnetization2[0], args.magnetization2[1], args.align2)
    
    # 比较磁力
    result = compare_magnetic_pairs(shape1_pair, shape2_pair, args.distance)
    
    # 输出结果
    print("\n磁体相互作用比较结果:")
    print(f"第一对 ({args.shape1}): 力大小 = {result['pair1']['force_magnitude']:.4e} N, 类型: {result['pair1']['force_type']}")
    print(f"第二对 ({args.shape2}): 力大小 = {result['pair2']['force_magnitude']:.4e} N, 类型: {result['pair2']['force_type']}")
    print(f"结论: {'第一对' if result['comparison'] == 'pair1' else '第二对'}磁体之间的力更强\n")
    
    # 可视化
    if args.visualize:
        print("正在生成第一对磁体的可视化...")
        plot1_path = f"scienceqa-physics-1291_{args.shape1}_magnets_L{args.align1}.png"
        visualize_magnetic_interaction(args.shape1, args.volume1, args.magnetization1, 
                                      args.distance, args.align1, save_path=plot1_path)
        
        print("正在生成第二对磁体的可视化...")
        plot2_path = f"scienceqa-physics-1291_{args.shape2}_magnets_L{args.align2}.png"
        visualize_magnetic_interaction(args.shape2, args.volume2, args.magnetization2, 
                                      args.distance, args.align2, save_path=plot2_path)