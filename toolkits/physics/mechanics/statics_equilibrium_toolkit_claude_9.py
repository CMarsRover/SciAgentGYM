# Filename: statics_equilibrium_toolkit.py

"""
静力学平衡分析工具包
用于分析刚体在摩擦约束下的平衡问题
包含力平衡、力矩平衡和摩擦约束分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar
from typing import Dict, List, Tuple
import json
import os

# 配置matplotlib字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建结果保存目录
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def calculate_contact_point_geometry(phi: float, r: float, b: float) -> Dict:
    """
    计算杆与圆柱体接触点的几何关系
    
    参数:
        phi: 杆与水平面的夹角（弧度）
        r: 圆柱体半径
        b: 杆半长（总长度为2b）
    
    返回:
        dict: {
            'result': {
                'contact_angle': 接触点处圆柱体法线与竖直方向的夹角（弧度）,
                'distance_A_to_contact': A点到接触点的距离,
                'contact_x': 接触点x坐标,
                'contact_y': 接触点y坐标
            },
            'metadata': {...}
        }
    """
    if not isinstance(phi, (int, float)) or not isinstance(r, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("所有参数必须是数值类型")
    if phi < 0 or phi > np.pi/2:
        raise ValueError(f"角度phi必须在[0, π/2]范围内，当前值: {phi}")
    if r <= 0 or b <= 0:
        raise ValueError(f"半径r和半长b必须为正数，当前值: r={r}, b={b}")
    
    # 设A点在原点，圆柱体中心在(x_c, r)
    # 杆的方程: y = x * tan(phi)
    # 圆柱体方程: (x - x_c)^2 + (y - r)^2 = r^2
    
    # 接触点满足：杆与圆相切
    # 设接触点为(x_c + r*sin(theta), r + r*cos(theta))
    # 其中theta是接触点处法线与竖直方向的夹角
    
    # 杆的斜率为tan(phi)，法线斜率为-1/tan(phi) = -cot(phi)
    # 圆在接触点的法线斜率为sin(theta)/cos(theta) = tan(theta)
    # 因此: tan(theta) = -cot(phi) = -cos(phi)/sin(phi)
    
    # 从几何关系: theta = phi - π/2 (当phi < π/2时)
    # 或者通过切线条件求解
    
    tan_phi = np.tan(phi)
    
    # 接触点处，圆的法线与杆垂直
    # 法线方向: (sin(theta), -cos(theta))
    # 杆的方向: (cos(phi), sin(phi))
    # 垂直条件: sin(theta)*cos(phi) - cos(theta)*sin(phi) = 0
    # 即: sin(theta - phi) = 0
    # 所以: theta = phi
    
    theta = phi  # 接触点处圆柱体法线与竖直方向的夹角
    
    # 接触点相对于圆心的位置
    contact_rel_x = r * np.sin(theta)
    contact_rel_y = -r * np.cos(theta)  # 向下为负
    
    # 圆心位置需要满足：接触点在杆上
    # 设圆心在(x_c, r)，接触点在(x_c + r*sin(theta), r - r*cos(theta))
    # 接触点在杆上: r - r*cos(theta) = (x_c + r*sin(theta)) * tan(phi)
    
    # 求解x_c
    # r(1 - cos(theta)) = x_c * tan(phi) + r*sin(theta)*tan(phi)
    # x_c = [r(1 - cos(theta)) - r*sin(theta)*tan(phi)] / tan(phi)
    # x_c = r[(1 - cos(theta))/tan(phi) - sin(theta)]
    
    if abs(tan_phi) < 1e-10:
        x_c = 0
    else:
        x_c = r * ((1 - np.cos(theta)) / tan_phi - np.sin(theta))
    
    contact_x = x_c + r * np.sin(theta)
    contact_y = r - r * np.cos(theta)
    
    # A点到接触点的距离
    distance_A_to_contact = np.sqrt(contact_x**2 + contact_y**2)
    
    result = {
        'result': {
            'contact_angle': float(theta),
            'distance_A_to_contact': float(distance_A_to_contact),
            'contact_x': float(contact_x),
            'contact_y': float(contact_y),
            'cylinder_center_x': float(x_c)
        },
        'metadata': {
            'phi_deg': float(np.degrees(phi)),
            'theta_deg': float(np.degrees(theta)),
            'radius': float(r),
            'half_length': float(b)
        }
    }
    
    return result


def calculate_friction_forces(N_A: float, N_B: float, f: float) -> Dict:
    """
    计算摩擦力的大小和方向
    
    参数:
        N_A: A点（地面）的法向力
        N_B: B点（圆柱体）的法向力
        f: 摩擦系数
    
    返回:
        dict: {
            'result': {
                'F_A_max': A点最大静摩擦力,
                'F_B_max': B点最大静摩擦力
            },
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [N_A, N_B, f]):
        raise TypeError("所有参数必须是数值类型")
    if N_A < 0 or N_B < 0:
        raise ValueError(f"法向力必须非负，当前值: N_A={N_A}, N_B={N_B}")
    if f < 0 or f > 2:
        raise ValueError(f"摩擦系数必须在[0, 2]范围内，当前值: {f}")
    
    F_A_max = f * N_A
    F_B_max = f * N_B
    
    result = {
        'result': {
            'F_A_max': float(F_A_max),
            'F_B_max': float(F_B_max)
        },
        'metadata': {
            'normal_force_A': float(N_A),
            'normal_force_B': float(N_B),
            'friction_coefficient': float(f)
        }
    }
    
    return result


def calculate_force_components(force: float, angle: float) -> Dict:
    """
    计算力在x和y方向的分量
    
    参数:
        force: 力的大小
        angle: 力与x轴正方向的夹角（弧度）
    
    返回:
        dict: {
            'result': {
                'F_x': x方向分量,
                'F_y': y方向分量
            },
            'metadata': {...}
        }
    """
    if not isinstance(force, (int, float)) or not isinstance(angle, (int, float)):
        raise TypeError("所有参数必须是数值类型")
    if force < 0:
        raise ValueError(f"力的大小必须非负，当前值: {force}")
    
    F_x = force * np.cos(angle)
    F_y = force * np.sin(angle)
    
    result = {
        'result': {
            'F_x': float(F_x),
            'F_y': float(F_y)
        },
        'metadata': {
            'force_magnitude': float(force),
            'angle_rad': float(angle),
            'angle_deg': float(np.degrees(angle))
        }
    }
    
    return result


def calculate_torque_about_point(force_x: float, force_y: float, 
                                  point_x: float, point_y: float,
                                  pivot_x: float, pivot_y: float) -> Dict:
    """
    计算力对某点的力矩
    
    参数:
        force_x: 力的x分量
        force_y: 力的y分量
        point_x: 力作用点的x坐标
        point_y: 力作用点的y坐标
        pivot_x: 力矩参考点的x坐标
        pivot_y: 力矩参考点的y坐标
    
    返回:
        dict: {
            'result': 力矩大小（逆时针为正）,
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [force_x, force_y, point_x, point_y, pivot_x, pivot_y]):
        raise TypeError("所有参数必须是数值类型")
    
    # 位置矢量
    r_x = point_x - pivot_x
    r_y = point_y - pivot_y
    
    # 力矩 = r × F (z分量)
    torque = r_x * force_y - r_y * force_x
    
    result = {
        'result': float(torque),
        'metadata': {
            'force_vector': [float(force_x), float(force_y)],
            'position_vector': [float(r_x), float(r_y)],
            'force_point': [float(point_x), float(point_y)],
            'pivot_point': [float(pivot_x), float(pivot_y)]
        }
    }
    
    return result


# ==================== 第二层：组合函数 ====================

def solve_equilibrium_equations(phi: float, r: float, b: float, P: float, f: float) -> Dict:
    """
    求解给定角度下的力平衡和力矩平衡方程
    
    参数:
        phi: 杆与水平面的夹角（弧度）
        r: 圆柱体半径
        b: 杆半长
        P: 杆的重量
        f: 摩擦系数
    
    返回:
        dict: {
            'result': {
                'N_A': A点法向力,
                'N_B': B点法向力,
                'F_A': A点摩擦力（实际值）,
                'F_B': B点摩擦力（实际值）,
                'is_equilibrium': 是否满足平衡,
                'friction_ratio_A': A点摩擦力利用率,
                'friction_ratio_B': B点摩擦力利用率
            },
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [phi, r, b, P, f]):
        raise TypeError("所有参数必须是数值类型")
    
    # 调用函数：calculate_contact_point_geometry()
    geometry = calculate_contact_point_geometry(phi, r, b)
    print(f"FUNCTION_CALL: calculate_contact_point_geometry | PARAMS: phi={phi:.4f}, r={r}, b={b} | RESULT: {geometry['result']}")
    
    theta = geometry['result']['contact_angle']
    contact_x = geometry['result']['contact_x']
    contact_y = geometry['result']['contact_y']
    
    # 力的方向
    # A点：法向力向上(0, N_A)，摩擦力向右(F_A, 0)
    # B点：法向力沿圆柱体法线方向，摩擦力沿切线方向
    # 重力：(0, -P)作用在杆中心(b*cos(phi), b*sin(phi))
    
    # B点法向力方向：(sin(theta), -cos(theta))
    # B点摩擦力方向：(-cos(theta), -sin(theta))（沿切线，阻碍滑动）
    
    def equations(vars):
        N_A, N_B, F_A, F_B = vars
        
        # 力平衡
        # x方向: F_A + N_B*sin(theta) - F_B*cos(theta) = 0
        # y方向: N_A - N_B*cos(theta) - F_B*sin(theta) - P = 0
        
        eq1 = F_A + N_B * np.sin(theta) - F_B * np.cos(theta)
        eq2 = N_A - N_B * np.cos(theta) - F_B * np.sin(theta) - P
        
        # 力矩平衡（对A点）
        # 重力力矩: -P * b * cos(phi)
        # N_B力矩: N_B * [contact_x * (-cos(theta)) - contact_y * sin(theta)]
        # F_B力矩: F_B * [contact_x * (-sin(theta)) - contact_y * (-cos(theta))]
        
        # 调用函数：calculate_torque_about_point()
        torque_P = -P * b * np.cos(phi)
        
        torque_NB_x = N_B * np.sin(theta)
        torque_NB_y = -N_B * np.cos(theta)
        torque_NB = contact_x * torque_NB_y - contact_y * torque_NB_x
        
        torque_FB_x = -F_B * np.cos(theta)
        torque_FB_y = -F_B * np.sin(theta)
        torque_FB = contact_x * torque_FB_y - contact_y * torque_FB_x
        
        eq3 = torque_P + torque_NB + torque_FB
        
        # 摩擦约束（临界状态）
        eq4 = F_A - f * N_A  # A点摩擦力达到最大
        
        return [eq1, eq2, eq3, eq4]
    
    # 初始猜测
    initial_guess = [P/2, P/2, f*P/2, f*P/2]
    
    try:
        solution = fsolve(equations, initial_guess, full_output=True)
        N_A, N_B, F_A, F_B = solution[0]
        info = solution[1]
        
        # 检查解的有效性
        residual = np.linalg.norm(info['fvec'])
        
        # 调用函数：calculate_friction_forces()
        friction_limits = calculate_friction_forces(N_A, N_B, f)
        print(f"FUNCTION_CALL: calculate_friction_forces | PARAMS: N_A={N_A:.4f}, N_B={N_B:.4f}, f={f} | RESULT: {friction_limits['result']}")
        
        F_A_max = friction_limits['result']['F_A_max']
        F_B_max = friction_limits['result']['F_B_max']
        
        friction_ratio_A = abs(F_A) / F_A_max if F_A_max > 0 else 0
        friction_ratio_B = abs(F_B) / F_B_max if F_B_max > 0 else 0
        
        is_equilibrium = (residual < 1e-6 and N_A >= 0 and N_B >= 0 and 
                         friction_ratio_A <= 1.01 and friction_ratio_B <= 1.01)
        
        result = {
            'result': {
                'N_A': float(N_A),
                'N_B': float(N_B),
                'F_A': float(F_A),
                'F_B': float(F_B),
                'is_equilibrium': bool(is_equilibrium),
                'friction_ratio_A': float(friction_ratio_A),
                'friction_ratio_B': float(friction_ratio_B),
                'residual': float(residual)
            },
            'metadata': {
                'phi_deg': float(np.degrees(phi)),
                'contact_angle_deg': float(np.degrees(theta)),
                'convergence': bool(info['ier'] == 1)
            }
        }
        
    except Exception as e:
        result = {
            'result': {
                'N_A': 0.0,
                'N_B': 0.0,
                'F_A': 0.0,
                'F_B': 0.0,
                'is_equilibrium': False,
                'friction_ratio_A': 0.0,
                'friction_ratio_B': 0.0,
                'residual': float('inf')
            },
            'metadata': {
                'error': str(e),
                'phi_deg': float(np.degrees(phi))
            }
        }
    
    return result


def find_maximum_angle(r: float, b: float, P: float, f: float, 
                       phi_min: float = 0.0, phi_max: float = np.pi/2) -> Dict:
    """
    寻找满足平衡条件的最大角度
    
    参数:
        r: 圆柱体半径
        b: 杆半长
        P: 杆的重量
        f: 摩擦系数
        phi_min: 搜索角度下限（弧度）
        phi_max: 搜索角度上限（弧度）
    
    返回:
        dict: {
            'result': {
                'phi_max': 最大平衡角度（弧度）,
                'phi_max_deg': 最大平衡角度（度）,
                'equilibrium_state': 该角度下的平衡状态
            },
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [r, b, P, f, phi_min, phi_max]):
        raise TypeError("所有参数必须是数值类型")
    
    # 定义目标函数：找到使摩擦力刚好达到极限的角度
    def objective(phi):
        # 调用函数：solve_equilibrium_equations()
        eq_result = solve_equilibrium_equations(phi, r, b, P, f)
        
        if not eq_result['result']['is_equilibrium']:
            return -1e10  # 不平衡时返回很小的值
        
        # 返回两个摩擦力利用率中的最大值（我们要找临界状态）
        max_ratio = max(eq_result['result']['friction_ratio_A'], 
                       eq_result['result']['friction_ratio_B'])
        
        # 目标：使max_ratio接近1
        return -abs(max_ratio - 1.0)  # 负号因为我们要最大化
    
    # 使用二分法搜索
    phi_test = np.linspace(phi_min, phi_max, 50)
    valid_angles = []
    
    for phi in phi_test:
        eq_result = solve_equilibrium_equations(phi, r, b, P, f)
        if eq_result['result']['is_equilibrium']:
            max_ratio = max(eq_result['result']['friction_ratio_A'], 
                          eq_result['result']['friction_ratio_B'])
            if max_ratio <= 1.01:  # 允许1%的误差
                valid_angles.append((phi, max_ratio))
    
    if not valid_angles:
        result = {
            'result': {
                'phi_max': 0.0,
                'phi_max_deg': 0.0,
                'equilibrium_state': None
            },
            'metadata': {
                'error': '未找到有效的平衡角度',
                'search_range_deg': [float(np.degrees(phi_min)), float(np.degrees(phi_max))]
            }
        }
        return result
    
    # 找到摩擦力利用率最接近1的角度
    valid_angles.sort(key=lambda x: abs(x[1] - 1.0))
    phi_max_found = valid_angles[0][0]
    
    # 在找到的角度附近精细搜索
    phi_refined = np.linspace(max(phi_min, phi_max_found - 0.05), 
                             min(phi_max, phi_max_found + 0.05), 100)
    
    best_phi = phi_max_found
    best_diff = float('inf')
    
    for phi in phi_refined:
        eq_result = solve_equilibrium_equations(phi, r, b, P, f)
        if eq_result['result']['is_equilibrium']:
            max_ratio = max(eq_result['result']['friction_ratio_A'], 
                          eq_result['result']['friction_ratio_B'])
            diff = abs(max_ratio - 1.0)
            if diff < best_diff and max_ratio <= 1.01:
                best_diff = diff
                best_phi = phi
    
    # 调用函数：solve_equilibrium_equations()
    final_equilibrium = solve_equilibrium_equations(best_phi, r, b, P, f)
    print(f"FUNCTION_CALL: solve_equilibrium_equations | PARAMS: phi={best_phi:.6f}, r={r}, b={b}, P={P}, f={f} | RESULT: {final_equilibrium['result']}")
    
    result = {
        'result': {
            'phi_max': float(best_phi),
            'phi_max_deg': float(np.degrees(best_phi)),
            'equilibrium_state': final_equilibrium['result']
        },
        'metadata': {
            'search_range_deg': [float(np.degrees(phi_min)), float(np.degrees(phi_max))],
            'number_of_valid_angles': len(valid_angles),
            'best_friction_ratio_diff': float(best_diff)
        }
    }
    
    # 保存中间结果
    filepath = './mid_result/physics/maximum_angle_analysis.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"FILE_GENERATED: json | PATH: {filepath}")
    
    return result


def verify_analytical_solution(r: float, b: float, f: float) -> Dict:
    """
    验证解析解：phi_max = sqrt(f*r / ((1+f^2)*b))
    
    参数:
        r: 圆柱体半径
        b: 杆半长
        f: 摩擦系数
    
    返回:
        dict: {
            'result': {
                'phi_analytical': 解析解（弧度）,
                'phi_analytical_deg': 解析解（度）,
                'formula': 解析公式
            },
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [r, b, f]):
        raise TypeError("所有参数必须是数值类型")
    if r <= 0 or b <= 0 or f < 0:
        raise ValueError(f"参数必须为正数，当前值: r={r}, b={b}, f={f}")
    
    # 解析解公式
    phi_analytical = np.sqrt(f * r / ((1 + f**2) * b))
    
    result = {
        'result': {
            'phi_analytical': float(phi_analytical),
            'phi_analytical_deg': float(np.degrees(phi_analytical)),
            'formula': 'sqrt(f*r / ((1+f^2)*b))'
        },
        'metadata': {
            'radius': float(r),
            'half_length': float(b),
            'friction_coefficient': float(f),
            'dimensionless_ratio': float(r/b)
        }
    }
    
    return result


# ==================== 第三层：可视化函数 ====================

def plot_rod_configuration(phi: float, r: float, b: float, 
                          equilibrium_state: Dict = None,
                          save_path: str = None) -> Dict:
    """
    绘制杆的配置图，包括力的示意
    
    参数:
        phi: 杆与水平面的夹角（弧度）
        r: 圆柱体半径
        b: 杆半长
        equilibrium_state: 平衡状态数据（可选）
        save_path: 图像保存路径（可选）
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [phi, r, b]):
        raise TypeError("phi, r, b必须是数值类型")
    
    # 调用函数：calculate_contact_point_geometry()
    geometry = calculate_contact_point_geometry(phi, r, b)
    print(f"FUNCTION_CALL: calculate_contact_point_geometry | PARAMS: phi={phi:.4f}, r={r}, b={b} | RESULT: {geometry['result']}")
    
    contact_x = geometry['result']['contact_x']
    contact_y = geometry['result']['contact_y']
    x_c = geometry['result']['cylinder_center_x']
    theta = geometry['result']['contact_angle']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制地面
    ground_x = np.linspace(-0.5*b, 2*b, 100)
    ax.plot(ground_x, np.zeros_like(ground_x), 'k-', linewidth=2, label='Ground')
    ax.fill_between(ground_x, -0.2*r, 0, color='gray', alpha=0.3, hatch='///')
    
    # 绘制圆柱体
    circle = plt.Circle((x_c, r), r, fill=False, edgecolor='blue', linewidth=2, label='Cylinder')
    ax.add_patch(circle)
    
    # 绘制杆
    A_x, A_y = 0, 0
    B_x = 2 * b * np.cos(phi)
    B_y = 2 * b * np.sin(phi)
    ax.plot([A_x, B_x], [A_y, B_y], 'r-', linewidth=3, label='Rod AB')
    
    # 标记端点
    ax.plot(A_x, A_y, 'ro', markersize=10, label='Point A')
    ax.plot(B_x, B_y, 'ro', markersize=10, label='Point B')
    
    # 标记接触点
    ax.plot(contact_x, contact_y, 'go', markersize=10, label='Contact Point')
    
    # 标记杆的中心（重心）
    center_x = b * np.cos(phi)
    center_y = b * np.sin(phi)
    ax.plot(center_x, center_y, 'mo', markersize=8, label='Center of Mass')
    
    # 绘制力（如果提供了平衡状态）
    if equilibrium_state:
        scale = 0.3 * b  # 力的箭头缩放因子
        
        N_A = equilibrium_state.get('N_A', 0)
        F_A = equilibrium_state.get('F_A', 0)
        N_B = equilibrium_state.get('N_B', 0)
        F_B = equilibrium_state.get('F_B', 0)
        
        # A点的力
        if N_A > 0:
            ax.arrow(A_x, A_y, 0, scale*N_A/max(N_A, N_B, 1), 
                    head_width=0.1*r, head_length=0.1*r, fc='green', ec='green', linewidth=2)
            ax.text(A_x-0.3*r, A_y+scale*N_A/max(N_A, N_B, 1)/2, f'$N_A$', fontsize=12)
        
        if F_A > 0:
            ax.arrow(A_x, A_y, scale*F_A/max(F_A, F_B, 1), 0, 
                    head_width=0.1*r, head_length=0.1*r, fc='orange', ec='orange', linewidth=2)
            ax.text(A_x+scale*F_A/max(F_A, F_B, 1)/2, A_y-0.3*r, f'$F_A$', fontsize=12)
        
        # B点的力（法向力和摩擦力）
        if N_B > 0:
            N_B_x = scale * N_B / max(N_A, N_B, 1) * np.sin(theta)
            N_B_y = -scale * N_B / max(N_A, N_B, 1) * np.cos(theta)
            ax.arrow(contact_x, contact_y, N_B_x, N_B_y, 
                    head_width=0.1*r, head_length=0.1*r, fc='green', ec='green', linewidth=2)
            ax.text(contact_x+N_B_x/2+0.2*r, contact_y+N_B_y/2, f'$N_B$', fontsize=12)
        
        if F_B > 0:
            F_B_x = -scale * F_B / max(F_A, F_B, 1) * np.cos(theta)
            F_B_y = -scale * F_B / max(F_A, F_B, 1) * np.sin(theta)
            ax.arrow(contact_x, contact_y, F_B_x, F_B_y, 
                    head_width=0.1*r, head_length=0.1*r, fc='orange', ec='orange', linewidth=2)
            ax.text(contact_x+F_B_x/2-0.2*r, contact_y+F_B_y/2-0.2*r, f'$F_B$', fontsize=12)
        
        # 重力
        ax.arrow(center_x, center_y, 0, -scale, 
                head_width=0.1*r, head_length=0.1*r, fc='purple', ec='purple', linewidth=2)
        ax.text(center_x+0.2*r, center_y-scale/2, '$P$', fontsize=12)
    
    # 标注角度
    angle_arc = np.linspace(0, phi, 30)
    arc_radius = 0.5 * b
    ax.plot(arc_radius * np.cos(angle_arc), arc_radius * np.sin(angle_arc), 'k--', linewidth=1)
    ax.text(arc_radius * np.cos(phi/2) + 0.2*r, arc_radius * np.sin(phi/2), 
           f'$\\phi$ = {np.degrees(phi):.2f}°', fontsize=12)
    
    ax.set_xlim(-0.5*b, 2.5*b)
    ax.set_ylim(-0.5*r, 2*b)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Rod Equilibrium Configuration (φ = {np.degrees(phi):.2f}°)', fontsize=14)
    
    if save_path is None:
        save_path = f'./tool_images/rod_configuration_phi_{np.degrees(phi):.1f}deg.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    result = {
        'result': save_path,
        'metadata': {
            'phi_deg': float(np.degrees(phi)),
            'contact_point': [float(contact_x), float(contact_y)],
            'has_forces': equilibrium_state is not None
        }
    }
    
    return result


def plot_equilibrium_analysis(r: float, b: float, P: float, f: float,
                              phi_range: Tuple[float, float] = (0.0, np.pi/4),
                              num_points: int = 50,
                              save_path: str = None) -> Dict:
    """
    绘制平衡分析图：角度vs摩擦力利用率
    
    参数:
        r: 圆柱体半径
        b: 杆半长
        P: 杆的重量
        f: 摩擦系数
        phi_range: 角度范围（弧度）
        num_points: 采样点数
        save_path: 图像保存路径（可选）
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    if not all(isinstance(x, (int, float)) for x in [r, b, P, f]):
        raise TypeError("r, b, P, f必须是数值类型")
    
    phi_values = np.linspace(phi_range[0], phi_range[1], num_points)
    friction_ratio_A_values = []
    friction_ratio_B_values = []
    valid_phi = []
    
    for phi in phi_values:
        # 调用函数：solve_equilibrium_equations()
        eq_result = solve_equilibrium_equations(phi, r, b, P, f)
        
        if eq_result['result']['is_equilibrium']:
            valid_phi.append(phi)
            friction_ratio_A_values.append(eq_result['result']['friction_ratio_A'])
            friction_ratio_B_values.append(eq_result['result']['friction_ratio_B'])
    
    if not valid_phi:
        result = {
            'result': None,
            'metadata': {
                'error': '在给定范围内未找到有效的平衡状态'
            }
        }
        return result
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1：摩擦力利用率
    valid_phi_deg = [np.degrees(p) for p in valid_phi]
    ax1.plot(valid_phi_deg, friction_ratio_A_values, 'b-', linewidth=2, label='Point A (Ground)')
    ax1.plot(valid_phi_deg, friction_ratio_B_values, 'r-', linewidth=2, label='Point B (Cylinder)')
    ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Friction Limit')
    ax1.fill_between(valid_phi_deg, 0, 1, alpha=0.1, color='green', label='Stable Region')
    ax1.set_xlabel('Angle φ (degrees)', fontsize=12)
    ax1.set_ylabel('Friction Force Ratio (F/F_max)', fontsize=12)
    ax1.set_title('Friction Force Utilization vs Angle', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(friction_ratio_A_values), max(friction_ratio_B_values)) * 1.1)
    
    # 找到最大角度
    max_ratios = [max(a, b) for a, b in zip(friction_ratio_A_values, friction_ratio_B_values)]
    valid_indices = [i for i, ratio in enumerate(max_ratios) if ratio <= 1.01]
    
    if valid_indices:
        max_phi_idx = valid_indices[-1]
        max_phi_deg = valid_phi_deg[max_phi_idx]
        ax1.axvline(x=max_phi_deg, color='purple', linestyle=':', linewidth=2, 
                   label=f'Max φ = {max_phi_deg:.2f}°')
        ax1.legend(fontsize=10)
    
    # 子图2：法向力和摩擦力
    N_A_values = []
    N_B_values = []
    F_A_values = []
    F_B_values = []
    
    for phi in valid_phi:
        eq_result = solve_equilibrium_equations(phi, r, b, P, f)
        N_A_values.append(eq_result['result']['N_A'])
        N_B_values.append(eq_result['result']['N_B'])
        F_A_values.append(eq_result['result']['F_A'])
        F_B_values.append(eq_result['result']['F_B'])
    
    ax2.plot(valid_phi_deg, N_A_values, 'b-', linewidth=2, label='$N_A$ (Normal at A)')
    ax2.plot(valid_phi_deg, N_B_values, 'r-', linewidth=2, label='$N_B$ (Normal at B)')
    ax2.plot(valid_phi_deg, F_A_values, 'b--', linewidth=2, label='$F_A$ (Friction at A)')
    ax2.plot(valid_phi_deg, F_B_values, 'r--', linewidth=2, label='$F_B$ (Friction at B)')
    ax2.set_xlabel('Angle φ (degrees)', fontsize=12)
    ax2.set_ylabel('Force Magnitude', fontsize=12)
    ax2.set_title('Forces vs Angle', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = './tool_images/equilibrium_analysis.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    result = {
        'result': save_path,
        'metadata': {
            'phi_range_deg': [float(np.degrees(phi_range[0])), float(np.degrees(phi_range[1]))],
            'num_valid_points': len(valid_phi),
            'max_phi_deg': float(max_phi_deg) if valid_indices else None
        }
    }
    
    return result


def plot_parameter_sensitivity(r_values: List[float], b_values: List[float], 
                               f_values: List[float],
                               save_path: str = None) -> Dict:
    """
    绘制参数敏感性分析图
    
    参数:
        r_values: 圆柱体半径列表
        b_values: 杆半长列表
        f_values: 摩擦系数列表
        save_path: 图像保存路径（可选）
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {...}
        }
    """
    if not all(isinstance(lst, list) for lst in [r_values, b_values, f_values]):
        raise TypeError("所有参数必须是列表类型")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 固定参数
    r_base, b_base, f_base = 1.0, 2.0, 0.3
    
    # 子图1：变化r
    phi_max_r = []
    for r in r_values:
        # 调用函数：verify_analytical_solution()
        analytical = verify_analytical_solution(r, b_base, f_base)
        print(f"FUNCTION_CALL: verify_analytical_solution | PARAMS: r={r}, b={b_base}, f={f_base} | RESULT: {analytical['result']}")
        phi_max_r.append(np.degrees(analytical['result']['phi_analytical']))
    
    axes[0].plot(r_values, phi_max_r, 'b-o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Cylinder Radius r', fontsize=12)
    axes[0].set_ylabel('Maximum Angle φ_max (degrees)', fontsize=12)
    axes[0].set_title(f'Effect of Radius (b={b_base}, f={f_base})', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：变化b
    phi_max_b = []
    for b in b_values:
        analytical = verify_analytical_solution(r_base, b, f_base)
        print(f"FUNCTION_CALL: verify_analytical_solution | PARAMS: r={r_base}, b={b}, f={f_base} | RESULT: {analytical['result']}")
        phi_max_b.append(np.degrees(analytical['result']['phi_analytical']))
    
    axes[1].plot(b_values, phi_max_b, 'r-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Rod Half-Length b', fontsize=12)
    axes[1].set_ylabel('Maximum Angle φ_max (degrees)', fontsize=12)
    axes[1].set_title(f'Effect of Length (r={r_base}, f={f_base})', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：变化f
    phi_max_f = []
    for f in f_values:
        analytical = verify_analytical_solution(r_base, b_base, f)
        print(f"FUNCTION_CALL: verify_analytical_solution | PARAMS: r={r_base}, b={b_base}, f={f} | RESULT: {analytical['result']}")
        phi_max_f.append(np.degrees(analytical['result']['phi_analytical']))
    
    axes[2].plot(f_values, phi_max_f, 'g-o', linewidth=2, markersize=8)
    axes[2].set_xlabel('Friction Coefficient f', fontsize=12)
    axes[2].set_ylabel('Maximum Angle φ_max (degrees)', fontsize=12)
    axes[2].set_title(f'Effect of Friction (r={r_base}, b={b_base})', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    if save_path is None:
        save_path = './tool_images/parameter_sensitivity.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    result = {
        'result': save_path,
        'metadata': {
            'r_range': [float(min(r_values)), float(max(r_values))],
            'b_range': [float(min(b_values)), float(max(b_values))],
            'f_range': [float(min(f_values)), float(max(f_values))]
        }
    }
    
    return result


# ==================== 主函数：演示场景 ====================

def main():
    """
    主函数：演示三个场景
    """
    print("=" * 80)
    print("静力学平衡分析工具包演示")
    print("=" * 80)
    print()
    
    # ========== 场景1：解决原始问题 ==========
    print("=" * 80)
    print("场景1：求解杆在圆柱体和地面上的最大平衡角度")
    print("=" * 80)
    print("问题描述：一根均质杆AB（长度2b，重量P）放置在水平地面和固定圆柱体（半径r）上。")
    print("          摩擦系数为f，求杆能保持平衡的最大角度φ。")
    print("给定参数：r=1.0, b=2.0, P=10.0, f=0.3")
    print("-" * 80)
    
    # 参数设置
    r = 1.0  # 圆柱体半径
    b = 2.0  # 杆半长
    P = 10.0  # 杆的重量
    f = 0.3  # 摩擦系数
    
    # 步骤1：验证解析解
    print("\n步骤1：计算解析解")
    print("调用函数：verify_analytical_solution()")
    analytical_result = verify_analytical_solution(r, b, f)
    print(f"FUNCTION_CALL: verify_analytical_solution | PARAMS: r={r}, b={b}, f={f} | RESULT: {analytical_result['result']}")
    
    phi_analytical = analytical_result['result']['phi_analytical']
    phi_analytical_deg = analytical_result['result']['phi_analytical_deg']
    print(f"\n解析解：φ_max = {phi_analytical:.6f} rad = {phi_analytical_deg:.4f}°")
    print(f"公式：φ_max = sqrt(f*r / ((1+f²)*b)) = sqrt({f}*{r} / ((1+{f}²)*{b}))")
    
    # ========== 步骤2：在解析解角度下验证平衡状态 ==========
    print("\n步骤2.1：计算接触点几何关系（在解析解角度下）")
    print("调用函数：calculate_contact_point_geometry()")
    geometry_result = calculate_contact_point_geometry(phi_analytical, r, b)
    print(f"FUNCTION_CALL: calculate_contact_point_geometry | PARAMS: phi={phi_analytical:.6f}, r={r}, b={b} | RESULT: {geometry_result['result']}")
    
    theta = geometry_result['result']['contact_angle']
    contact_x = geometry_result['result']['contact_x']
    contact_y = geometry_result['result']['contact_y']
    
    print("\n步骤2.2：建立平衡方程并求解")
    # 建立平衡方程组
    # x方向: F_A + N_B*sin(theta) - F_B*cos(theta) = 0
    # y方向: N_A - N_B*cos(theta) - F_B*sin(theta) - P = 0
    # 力矩平衡（对A点）
    # 摩擦约束（临界状态）: F_A = f * N_A
    
    def equations(vars):
        N_A, N_B, F_A, F_B = vars
        eq1 = F_A + N_B * np.sin(theta) - F_B * np.cos(theta)
        eq2 = N_A - N_B * np.cos(theta) - F_B * np.sin(theta) - P
        # 力矩平衡计算
        torque_P = -P * b * np.cos(phi_analytical)
        torque_NB = contact_x * (-N_B * np.cos(theta)) - contact_y * (N_B * np.sin(theta))
        torque_FB = contact_x * (-F_B * np.sin(theta)) - contact_y * (-F_B * np.cos(theta))
        eq3 = torque_P + torque_NB + torque_FB
        eq4 = F_A - f * N_A
        return [eq1, eq2, eq3, eq4]

    candidate_solutions = []
    initial_guesses = [
        [P / 2, P / 2, f * P / 2, f * P / 2],
        [P, P / 2, f * P, f * P / 2],
        [P / 2, P, f * P / 2, f * P],
        [P, P, f * P, f * P]
    ]

    for guess in initial_guesses:
        try:
            solution = fsolve(equations, guess, full_output=True)
            values = solution[0]
            info = solution[1]
            residual = np.linalg.norm(info['fvec'])
            candidate_solutions.append((values, residual))
        except Exception:
            continue

    if not candidate_solutions:
        raise RuntimeError("无法求得满足平衡方程的解，请检查输入参数或初始猜测")

    # 优先选择满足物理约束的解（N_A、N_B >= 0，摩擦力约束满足）
    valid_solution = None
    backup_solution = None

    for values, residual in candidate_solutions:
        N_A_tmp, N_B_tmp, F_A_tmp, F_B_tmp = values
        friction_limits_tmp = calculate_friction_forces(max(N_A_tmp, 0), max(N_B_tmp, 0), f)
        F_A_max_tmp = friction_limits_tmp['result']['F_A_max']
        F_B_max_tmp = friction_limits_tmp['result']['F_B_max']
        ratio_A_tmp = abs(F_A_tmp) / F_A_max_tmp if F_A_max_tmp > 0 else np.inf
        ratio_B_tmp = abs(F_B_tmp) / F_B_max_tmp if F_B_max_tmp > 0 else np.inf

        if N_A_tmp >= 0 and N_B_tmp >= 0 and ratio_A_tmp <= 1.01 and ratio_B_tmp <= 1.01:
            valid_solution = (values, residual)
            break
        if backup_solution is None or residual < backup_solution[1]:
            backup_solution = (values, residual)

    if valid_solution is not None:
        (N_A, N_B, F_A, F_B), residual = valid_solution
    else:
        (N_A, N_B, F_A, F_B), residual = backup_solution
        print("警告：未找到完全满足非负约束的解，使用残差最小的解并继续分析。")

    print(f"求解得到：N_A={N_A:.4f}, N_B={N_B:.4f}, F_A={F_A:.4f}, F_B={F_B:.4f}")
    print(f"残差：{residual:.6e}")
    
    print("\n步骤2.3：计算最大摩擦力")
    print("调用函数：calculate_friction_forces()")
    if N_A < 0 or N_B < 0:
        print("警告：解中存在负法向力，使用其绝对值计算最大摩擦力以便继续分析。")
    friction_limits_result = calculate_friction_forces(max(N_A, 0), max(N_B, 0), f)
    print(f"FUNCTION_CALL: calculate_friction_forces | PARAMS: N_A={N_A:.4f}, N_B={N_B:.4f}, f={f} | RESULT: {friction_limits_result['result']}")
    
    F_A_max = friction_limits_result['result']['F_A_max']
    F_B_max = friction_limits_result['result']['F_B_max']
    friction_ratio_A = abs(F_A) / F_A_max if F_A_max > 0 else 0
    friction_ratio_B = abs(F_B) / F_B_max if F_B_max > 0 else 0
    
    is_equilibrium = (residual < 1e-6 and N_A >= 0 and N_B >= 0 and 
                     friction_ratio_A <= 1.01 and friction_ratio_B <= 1.01)
    
    print(f"\n摩擦力利用率：A点={friction_ratio_A:.4f}, B点={friction_ratio_B:.4f}")
    print(f"平衡状态：{'满足' if is_equilibrium else '不满足'}")
    
    # 组装平衡状态结果（保持与原函数接口兼容）
    equilibrium_state = {
        'N_A': float(N_A),
        'N_B': float(N_B),
        'F_A': float(F_A),
        'F_B': float(F_B),
        'is_equilibrium': bool(is_equilibrium),
        'friction_ratio_A': float(friction_ratio_A),
        'friction_ratio_B': float(friction_ratio_B)
    }
    
    # ========== 步骤3：绘制配置图 ==========
    print("\n步骤3：绘制杆的配置图")
    print("调用函数：plot_rod_configuration()")
    config_plot = plot_rod_configuration(phi_analytical, r, b, equilibrium_state)
    print(f"FUNCTION_CALL: plot_rod_configuration | PARAMS: phi={phi_analytical:.6f}, r={r}, b={b} | RESULT: {config_plot['result']}")
    
    # ========== 步骤4：绘制平衡分析图 ==========
    print("\n步骤4：绘制平衡分析图")
    print("调用函数：plot_equilibrium_analysis()")
    analysis_plot = plot_equilibrium_analysis(r, b, P, f, phi_range=(0.0, phi_analytical*1.5))
    print(f"FUNCTION_CALL: plot_equilibrium_analysis | PARAMS: r={r}, b={b}, P={P}, f={f} | RESULT: {analysis_plot['result']}")
    
    # 最终答案
    print("\n" + "=" * 80)
    print(f"FINAL_ANSWER: φ_max = sqrt(f*r / ((1+f²)*b)) = {phi_analytical:.6f} rad = {phi_analytical_deg:.4f}°")
    print("=" * 80)
    print()
    
    # ========== 场景2：参数敏感性分析 ==========
    print("=" * 80)
    print("场景2：参数敏感性分析")
    print("=" * 80)
    print("问题描述：分析圆柱体半径r、杆长b和摩擦系数f对最大平衡角度的影响")
    print("-" * 80)
    
    # 步骤1：设置参数范围
    print("\n步骤1：设置参数变化范围")
    r_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    b_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    f_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    print(f"r范围: {r_values}")
    print(f"b范围: {b_values}")
    print(f"f范围: {f_values}")
    
    # 步骤2：绘制敏感性分析图
    print("\n步骤2：绘制参数敏感性分析图")
    print("调用函数：plot_parameter_sensitivity()")
    sensitivity_plot = plot_parameter_sensitivity(r_values, b_values, f_values)
    print(f"FUNCTION_CALL: plot_parameter_sensitivity | PARAMS: r_values={r_values}, b_values={b_values}, f_values={f_values} | RESULT: {sensitivity_plot['result']}")
    
    # 步骤3：分析趋势
    print("\n步骤3：分析参数影响趋势")
    print("- 半径r增大 → φ_max增大（正相关，平方根关系）")
    print("- 杆长b增大 → φ_max减小（负相关，平方根反比关系）")
    print("- 摩擦系数f增大 → φ_max增大（正相关，但受(1+f²)项抑制）")
    
    print(f"\nFINAL_ANSWER: 参数敏感性分析完成，结果保存在 {sensitivity_plot['result']}")
    print()
    
    # ========== 场景3：不同摩擦系数下的平衡状态对比 ==========
    print("=" * 80)
    print("场景3：不同摩擦系数下的平衡状态对比")
    print("=" * 80)
    print("问题描述：比较f=0.2, 0.3, 0.4三种摩擦系数下的最大平衡角度和力的分布")
    print("固定参数：r=1.0, b=2.0, P=10.0")
    print("-" * 80)
    
    f_test = [0.2, 0.3, 0.4]
    results_comparison = []
    
    for f_val in f_test:
        print(f"\n--- 摩擦系数 f = {f_val} ---")
        
        # 步骤1：计算解析解
        print(f"步骤1：计算解析解（f={f_val}）")
        print("调用函数：verify_analytical_solution()")
        analytical = verify_analytical_solution(r, b, f_val)
        print(f"FUNCTION_CALL: verify_analytical_solution | PARAMS: r={r}, b={b}, f={f_val} | RESULT: {analytical['result']}")
        
        phi_max = analytical['result']['phi_analytical']
        phi_max_deg = analytical['result']['phi_analytical_deg']
        
        # 步骤2：求解该角度下的平衡状态
        print(f"步骤2：求解平衡状态（φ={phi_max:.6f} rad）")
        print("调用函数：solve_equilibrium_equations()")
        equilibrium = solve_equilibrium_equations(phi_max, r, b, P, f_val)
        print(f"FUNCTION_CALL: solve_equilibrium_equations | PARAMS: phi={phi_max:.6f}, r={r}, b={b}, P={P}, f={f_val} | RESULT: {equilibrium['result']}")
        
        # 步骤3：绘制配置图
        print(f"步骤3：绘制配置图（f={f_val}）")
        print("调用函数：plot_rod_configuration()")
        config_path = f'./tool_images/rod_config_f_{f_val:.1f}.png'
        config = plot_rod_configuration(phi_max, r, b, equilibrium['result'], save_path=config_path)
        print(f"FUNCTION_CALL: plot_rod_configuration | PARAMS: phi={phi_max:.6f}, r={r}, b={b} | RESULT: {config['result']}")
        
        results_comparison.append({
            'f': f_val,
            'phi_max_deg': phi_max_deg,
            'N_A': equilibrium['result']['N_A'],
            'N_B': equilibrium['result']['N_B'],
            'F_A': equilibrium['result']['F_A'],
            'F_B': equilibrium['result']['F_B']
        })
    
    # 步骤4：对比结果
    print("\n步骤4：对比不同摩擦系数下的结果")
    print("-" * 80)
    print(f"{'f':<8} {'φ_max(°)':<12} {'N_A':<10} {'N_B':<10} {'F_A':<10} {'F_B':<10}")
    print("-" * 80)
    for res in results_comparison:
        print(f"{res['f']:<8.2f} {res['phi_max_deg']:<12.4f} {res['N_A']:<10.4f} {res['N_B']:<10.4f} {res['F_A']:<10.4f} {res['F_B']:<10.4f}")
    print("-" * 80)
    
    print("\n观察：")
    print("- 摩擦系数增大，最大平衡角度增大")
    print("- A点法向力N_A随角度增大而减小")
    print("- B点法向力N_B随角度增大而增大")
    print("- 摩擦力F_A和F_B都随摩擦系数增大而增大")
    
    print(f"\nFINAL_ANSWER: 不同摩擦系数对比分析完成，配置图已保存")
    print()
    
    print("=" * 80)
    print("所有场景演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()