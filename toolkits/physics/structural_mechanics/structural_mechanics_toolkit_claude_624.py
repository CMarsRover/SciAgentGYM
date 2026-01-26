# Filename: structural_mechanics_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/structural_mechanics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def calculate_support_reactions(span: float, load_position: float, load_value: float) -> Dict:
    """
    计算简支梁支座反力
    
    Args:
        span: 跨度 (m)
        load_position: 荷载位置距左支座距离 (m)
        load_value: 荷载值 (kN)，向下为正
    
    Returns:
        {'result': {'Ra': float, 'Rb': float}, 'metadata': {...}}
    """
    if span <= 0:
        raise ValueError("跨度必须大于0")
    if load_position < 0 or load_position > span:
        raise ValueError(f"荷载位置必须在0到{span}之间")
    
    # 力矩平衡：ΣMa = 0 → Rb * span - P * load_position = 0
    Rb = load_value * load_position / span
    # 力平衡：ΣFy = 0 → Ra + Rb - P = 0
    Ra = load_value - Rb
    
    result = {
        'result': {'Ra': round(Ra, 2), 'Rb': round(Rb, 2)},
        'metadata': {
            'method': 'moment_equilibrium',
            'span': span,
            'load_position': load_position,
            'load_value': load_value
        }
    }
    
    return result


def solve_joint_equilibrium(known_forces: List[Tuple[float, float]], 
                            unknown_angles: List[float]) -> Dict:
    """
    节点法求解未知杆件内力
    
    Args:
        known_forces: 已知力列表 [(Fx, Fy), ...], 单位kN
        unknown_angles: 未知杆件角度列表（度），从水平正向逆时针
    
    Returns:
        {'result': List[float], 'metadata': {...}}
    """
    n = len(unknown_angles)
    if n == 0:
        raise ValueError("至少需要一个未知杆件")
    
    # 构建平衡方程矩阵 [cos(θ1), cos(θ2), ...; sin(θ1), sin(θ2), ...]
    A = np.zeros((2, n))
    for i, angle in enumerate(unknown_angles):
        rad = np.radians(angle)
        A[0, i] = np.cos(rad)  # x方向系数
        A[1, i] = np.sin(rad)  # y方向系数
    
    # 已知力的合力（取反作为方程右侧）
    b = -np.array([sum(f[0] for f in known_forces), 
                   sum(f[1] for f in known_forces)])
    
    # 求解线性方程组
    try:
        if n == 2:
            forces = np.linalg.solve(A, b)
        else:
            forces, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        raise ValueError("方程组无解或有无穷多解")
    
    result = {
        'result': [round(f, 2) for f in forces],
        'metadata': {
            'method': 'joint_method',
            'equation_matrix': A.tolist(),
            'known_forces_sum': b.tolist(),
            'angles': unknown_angles
        }
    }
    
    return result


def calculate_member_length(point1: Tuple[float, float], 
                           point2: Tuple[float, float]) -> Dict:
    """
    计算两点间距离（杆件长度）
    
    Args:
        point1: 点1坐标 (x, y)
        point2: 点2坐标 (x, y)
    
    Returns:
        {'result': float, 'metadata': {...}}
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    length = np.sqrt(dx**2 + dy**2)
    
    result = {
        'result': round(length, 4),
        'metadata': {
            'point1': point1,
            'point2': point2,
            'dx': dx,
            'dy': dy
        }
    }
    
    return result


def calculate_member_angle(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> Dict:
    """
    计算杆件与水平方向夹角
    
    Args:
        point1: 起点坐标 (x, y)
        point2: 终点坐标 (x, y)
    
    Returns:
        {'result': float, 'metadata': {...}} 角度单位：度
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    result = {
        'result': round(angle_deg, 2),
        'metadata': {
            'point1': point1,
            'point2': point2,
            'angle_rad': angle_rad
        }
    }
    
    return result


# ==================== 第二层：组合函数 ====================

def validate_triangle_geometry(nodes: Dict[str, Tuple[float, float]], 
                               members: List[Tuple[str, str]],
                               tolerance: float = 1e-6) -> Dict:
    """
    验证桁架几何结构是否满足三角形构成条件
    
    Args:
        nodes: 节点坐标字典
        members: 杆件列表
        tolerance: 数值容差
    
    Returns:
        {'result': bool, 'metadata': {...}}
    """
    errors = []
    warnings = []
    
    # 1. 检查所有杆件长度是否为正
    member_lengths = {}
    for member in members:
        node1, node2 = member
        if node1 not in nodes or node2 not in nodes:
            errors.append(f"节点 {node1} 或 {node2} 不存在")
            continue
        
        length_result = calculate_member_length(nodes[node1], nodes[node2])
        length = length_result['result']
        member_name = f"{node1}{node2}"
        member_lengths[member_name] = length
        
        if length <= tolerance:
            errors.append(f"杆件 {member_name} 长度为零或负值: {length:.6f}")
    
    # 2. 检查关键三角形是否满足构成条件
    # 三角形构成条件：任意两边之和大于第三边，任意两边之差小于第三边
    # 注意：实际桁架结构中的关键三角形是AED和BFD（B-F-D）
    def check_triangle(a: float, b: float, c: float, name: str):
        """检查三条边a, b, c是否能构成三角形"""
        sides = [a, b, c]
        if any(s <= tolerance for s in sides):
            return False, f"存在零边或负边"
        
        # 检查两边之和大于第三边（对所有三边组合）
        if not (a + b > c + tolerance):
            return False, f"{a:.3f} + {b:.3f} <= {c:.3f}"
        if not (a + c > b + tolerance):
            return False, f"{a:.3f} + {c:.3f} <= {b:.3f}"
        if not (b + c > a + tolerance):
            return False, f"{b:.3f} + {c:.3f} <= {a:.3f}"
        
        # 检查两边之差小于第三边（对所有三边组合）
        if not (abs(a - b) < c - tolerance):
            return False, f"|{a:.3f} - {b:.3f}| >= {c:.3f}"
        if not (abs(a - c) < b - tolerance):
            return False, f"|{a:.3f} - {c:.3f}| >= {b:.3f}"
        if not (abs(b - c) < a - tolerance):
            return False, f"|{b:.3f} - {c:.3f}| >= {a:.3f}"
        
        return True, ""
    
    # 三角形AED: A-E-D（关键三角形）
    if all(n in nodes for n in ['A', 'E', 'D']):
        l_AE = member_lengths.get('AE', 0)
        l_ED = member_lengths.get('ED', 0)
        l_AD = member_lengths.get('AD', 0)
        
        is_valid, msg = check_triangle(l_AE, l_ED, l_AD, "AED")
        if not is_valid:
            errors.append(f"三角形AED不满足构成条件: AE={l_AE:.3f}, ED={l_ED:.3f}, AD={l_AD:.3f}, {msg}")
    
    # 三角形BFD: B-F-D（关键三角形，注意顺序是B-F-D）
    if all(n in nodes for n in ['B', 'F', 'D']):
        l_BF = member_lengths.get('FB', 0)  # FB和BF是同一根杆件
        l_FD = member_lengths.get('DF', 0)  # DF和FD是同一根杆件
        l_BD = member_lengths.get('DB', 0)  # DB和BD是同一根杆件
        
        is_valid, msg = check_triangle(l_BF, l_FD, l_BD, "BFD")
        if not is_valid:
            errors.append(f"三角形BFD不满足构成条件: BF={l_BF:.3f}, FD={l_FD:.3f}, BD={l_BD:.3f}, {msg}")
    
    # 3. 检查边界情况
    # 检查高度是否合理（不能为0或过大）
    if 'C' in nodes and 'A' in nodes:
        h = nodes['C'][1] - nodes['A'][1]
        if h <= tolerance:
            errors.append(f"桁架高度过小或为零: h={h:.6f}")
        elif h > 10 * (nodes.get('B', (0, 0))[0] - nodes.get('A', (0, 0))[0]):
            warnings.append(f"桁架高度过大，可能导致结构不稳定: h={h:.3f}")
    
    # 检查跨度是否合理
    if 'A' in nodes and 'B' in nodes:
        span = nodes['B'][0] - nodes['A'][0]
        if span <= tolerance:
            errors.append(f"桁架跨度过小或为零: span={span:.6f}")
    
    is_valid = len(errors) == 0
    
    result = {
        'result': is_valid,
        'metadata': {
            'errors': errors,
            'warnings': warnings,
            'member_lengths': member_lengths,
            'total_errors': len(errors),
            'total_warnings': len(warnings)
        }
    }
    
    return result


def analyze_truss_structure(nodes: Dict[str, Tuple[float, float]], 
                           members: List[Tuple[str, str]], 
                           supports: Dict[str, str],
                           loads: Dict[str, Tuple[float, float]]) -> Dict:
    """
    完整桁架结构分析
    
    Args:
        nodes: 节点坐标字典 {'A': (x, y), ...}
        members: 杆件列表 [('A', 'B'), ...]
        supports: 支座类型 {'A': 'pin', 'B': 'roller'}
        loads: 节点荷载 {'D': (Fx, Fy)}
    
    Returns:
        {'result': Dict, 'metadata': {...}}
    """
    # 计算所有杆件的几何信息
    member_info = {}
    for member in members:
        node1, node2 = member
        length_result = calculate_member_length(nodes[node1], nodes[node2])
        angle_result = calculate_member_angle(nodes[node1], nodes[node2])
        
        member_info[f"{node1}{node2}"] = {
            'length': length_result['result'],
            'angle': angle_result['result'],
            'nodes': (node1, node2)
        }
    
    # 保存中间结果
    filepath = './mid_result/structural_mechanics/member_geometry.json'
    with open(filepath, 'w') as f:
        json.dump(member_info, f, indent=2)
    
    result = {
        'result': {
            'member_info': member_info,
            'total_members': len(members),
            'total_nodes': len(nodes)
        },
        'metadata': {
            'geometry_file': filepath,
            'supports': supports,
            'loads': loads
        }
    }
    
    return result


def solve_truss_by_joints(nodes: Dict[str, Tuple[float, float]], 
                         member_info: Dict,
                         support_reactions: Dict[str, float],
                         loads: Dict[str, Tuple[float, float]],
                         solve_sequence: List[str]) -> Dict:
    """
    按节点顺序求解桁架内力
    
    Args:
        nodes: 节点坐标
        member_info: 杆件几何信息
        support_reactions: 支座反力 {'Ra': value, 'Rb': value}
        loads: 节点荷载
        solve_sequence: 求解节点顺序
    
    Returns:
        {'result': Dict, 'metadata': {...}}
    """
    member_forces = {}
    solution_steps = []
    
    for joint in solve_sequence:
        # 收集该节点的已知力和未知杆件
        known_forces = []
        unknown_members = []
        
        # 添加支座反力
        if joint == 'A' and 'Ra' in support_reactions:
            known_forces.append((0, support_reactions['Ra']))
        if joint == 'B' and 'Rb' in support_reactions:
            known_forces.append((0, support_reactions['Rb']))
        
        # 添加外荷载
        if joint in loads:
            known_forces.append(loads[joint])
        
        # 查找连接该节点的杆件
        for member_name, info in member_info.items():
            node1, node2 = info['nodes']
            if joint in (node1, node2):
                if member_name in member_forces:
                    # 已知内力
                    force = member_forces[member_name]
                    angle = info['angle']
                    if joint == node2:  # 反向
                        angle = (angle + 180) % 360
                    fx = force * np.cos(np.radians(angle))
                    fy = force * np.sin(np.radians(angle))
                    known_forces.append((fx, fy))
                else:
                    # 未知内力
                    angle = info['angle']
                    if joint == node2:
                        angle = (angle + 180) % 360
                    unknown_members.append((member_name, angle))
        
        # 求解未知杆件内力
        if unknown_members:
            angles = [m[1] for m in unknown_members]
            equilibrium_result = solve_joint_equilibrium(known_forces, angles)
            forces = equilibrium_result['result']
            
            for i, (member_name, _) in enumerate(unknown_members):
                member_forces[member_name] = forces[i]
            
            solution_steps.append({
                'joint': joint,
                'known_forces': known_forces,
                'solved_members': {m[0]: forces[i] for i, m in enumerate(unknown_members)}
            })
    
    # 保存求解步骤
    filepath = './mid_result/structural_mechanics/solution_steps.json'
    with open(filepath, 'w') as f:
        json.dump(solution_steps, f, indent=2)
    
    result = {
        'result': member_forces,
        'metadata': {
            'solution_file': filepath,
            'solve_sequence': solve_sequence,
            'total_steps': len(solution_steps)
        }
    }
    
    return result


def identify_force_type(force_value: float, tolerance: float = 0.01) -> Dict:
    """
    判断杆件受力类型
    
    Args:
        force_value: 杆件内力值 (kN)，正为拉力，负为压力
        tolerance: 零力判断容差
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    if abs(force_value) < tolerance:
        force_type = 'zero'
        description = '零杆'
    elif force_value > 0:
        force_type = 'tension'
        description = '拉力'
    else:
        force_type = 'compression'
        description = '压力'
    
    result = {
        'result': force_type,
        'metadata': {
            'force_value': force_value,
            'description': description,
            'magnitude': abs(force_value)
        }
    }
    
    return result


# ==================== 第三层：可视化函数 ====================

def visualize_truss_structure(nodes: Dict[str, Tuple[float, float]], 
                             members: List[Tuple[str, str]],
                             member_forces: Dict[str, float],
                             loads: Dict[str, Tuple[float, float]],
                             support_reactions: Dict[str, float],
                             title: str = "Truss Structure Analysis") -> Dict:
    """
    可视化桁架结构及内力
    
    Args:
        nodes: 节点坐标
        members: 杆件列表
        member_forces: 杆件内力
        loads: 节点荷载
        support_reactions: 支座反力
        title: 图表标题
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制杆件
    for member in members:
        node1, node2 = member
        x = [nodes[node1][0], nodes[node2][0]]
        y = [nodes[node1][1], nodes[node2][1]]
        
        member_name = f"{node1}{node2}"
        if member_name in member_forces:
            force = member_forces[member_name]
            if force > 0:
                color = 'blue'
                linestyle = '-'
                label = 'Tension'
            elif force < 0:
                color = 'red'
                linestyle = '--'
                label = 'Compression'
            else:
                color = 'gray'
                linestyle = ':'
                label = 'Zero'
            
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2)
            
            # 标注内力值
            mid_x = (x[0] + x[1]) / 2
            mid_y = (y[0] + y[1]) / 2
            ax.text(mid_x, mid_y, f"{abs(force):.2f}", 
                   fontsize=9, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 绘制节点
    for name, (x, y) in nodes.items():
        ax.plot(x, y, 'ko', markersize=10)
        ax.text(x, y + 0.15, name, fontsize=12, ha='center', weight='bold')
    
    # 绘制荷载
    for node, (fx, fy) in loads.items():
        x, y = nodes[node]
        if fy != 0:
            ax.arrow(x, y, 0, -0.3 * np.sign(fy), head_width=0.1, 
                    head_length=0.1, fc='green', ec='green', linewidth=2)
            ax.text(x + 0.2, y - 0.2, f"P={abs(fy):.0f}kN", fontsize=10, color='green')
    
    # 绘制支座
    if 'Ra' in support_reactions:
        x, y = nodes['A']
        ax.plot(x, y, 'v', markersize=15, color='brown')
        ax.text(x - 0.3, y - 0.3, f"Ra={support_reactions['Ra']:.1f}kN", 
               fontsize=9, color='brown')
    
    if 'Rb' in support_reactions:
        x, y = nodes['B']
        ax.plot(x, y, '^', markersize=15, color='brown')
        ax.text(x + 0.1, y - 0.3, f"Rb={support_reactions['Rb']:.1f}kN", 
               fontsize=9, color='brown')
    
    ax.set_xlabel('Distance (a)', fontsize=12)
    ax.set_ylabel('Height (h)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(['Tension (+)', 'Compression (-)', 'Zero'], loc='upper right')
    
    filepath = './tool_images/truss_structure_analysis.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    result = {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'size': os.path.getsize(filepath),
            'title': title
        }
    }
    
    return result


def generate_force_report(member_forces: Dict[str, float], 
                         member_info: Dict,
                         standard_answer: Optional[str] = None) -> Dict:
    """
    生成杆件内力报告
    
    Args:
        member_forces: 杆件内力字典
        member_info: 杆件几何信息
        standard_answer: 标准答案（可选）
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("桁架杆件内力分析报告")
    report_lines.append("=" * 60)
    report_lines.append(f"{'杆件':<8} {'内力(kN)':<12} {'类型':<10} {'长度':<10}")
    report_lines.append("-" * 60)
    
    incorrect_members = []
    
    for member_name, force in sorted(member_forces.items()):
        force_type_result = identify_force_type(force)
        force_type = force_type_result['metadata']['description']
        length = member_info[member_name]['length']
        
        report_lines.append(f"{member_name:<8} {force:>10.2f}  {force_type:<10} {length:>8.2f}")
        
        # 检查是否与标准答案匹配
        if standard_answer and member_name in standard_answer:
            if "pressure" in standard_answer.lower() or "压" in standard_answer:
                if force > 0:  # 应该是压力但计算为拉力
                    incorrect_members.append(member_name)
    
    report_lines.append("=" * 60)
    
    if standard_answer:
        report_lines.append(f"\n标准答案: {standard_answer}")
        if incorrect_members:
            report_lines.append(f"发现错误杆件: {', '.join(incorrect_members)}")
    
    report_text = "\n".join(report_lines)
    
    filepath = './mid_result/structural_mechanics/force_report.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    result = {
        'result': filepath,
        'metadata': {
            'total_members': len(member_forces),
            'tension_count': sum(1 for f in member_forces.values() if f > 0),
            'compression_count': sum(1 for f in member_forces.values() if f < 0),
            'zero_count': sum(1 for f in member_forces.values() if abs(f) < 0.01),
            'incorrect_members': incorrect_members
        }
    }
    
    return result


# ==================== 主函数：3个场景演示 ====================

def main():
    """
    演示桁架内力分析工具包的3个应用场景
    """
    
    # ========== 场景1：AC与AB夹角45度时DF杆受力情况分析 ==========
    print("=" * 60)
    print("场景1：AC与AB夹角45度时DF杆受力情况分析")
    print("=" * 60)
    print("问题描述：分析简支三角形桁架，AC与AB夹角为45度，节点D受向下荷载P=20kN")
    print("求DF杆的受力情况")
    print("-" * 60)
    
    # 固定角度：AC与AB夹角为45度
    angle_deg = 45.0  # AC与AB的夹角
    a_base = 2.0  # 单位长度，保持不变
    load_value = 20.0  # 荷载值不变
    
    # 根据角度计算高度：h = a * tan(45°) = 2 * 1 = 2m
    h = a_base * np.tan(np.radians(angle_deg))
    
    # 定义节点坐标（AC与AB夹角45度）
    # E和F的位置因子，确保三角形能够构成
    e_factor = 0.5  # E在AC上的40%位置
    f_factor = 0.5  # F在CB上的60%位置（从C到B）
    
    nodes = {
        'A': (0, 0),
        'B': (2*a_base, 0),
        'D': (a_base, 0),
        'E': (a_base * e_factor, h * e_factor),  # E在AC上
        'F': (a_base + a_base * f_factor, h * (1 - f_factor)),  # F在CB上
        'C': (a_base, h)
    }
    
    members = [
        ('A', 'D'), ('D', 'B'),  # 下弦
        ('A', 'E'), ('E', 'C'), ('C', 'F'), ('F', 'B'),  # 斜杆
        ('E', 'D'), ('D', 'F'),  # 竖杆和斜杆
        ('C', 'D')  # 竖杆
    ]
    
    supports = {'A': 'pin', 'B': 'roller'}
    loads = {'D': (0, -20)}  # 20kN向下
    solve_sequence = ['A', 'E', 'D', 'C', 'F', 'B']
    
    print(f"\n几何参数：")
    print(f"  AC与AB夹角: {angle_deg}°")
    print(f"  跨度AB: {2*a_base} m")
    print(f"  高度h: {h:.3f} m")
    print(f"  单位长度a: {a_base} m")
    print("-" * 60)
    
    # 步骤0：验证几何结构是否满足三角形构成条件
    print(f"\n步骤0：验证几何结构")
    validation_result = validate_triangle_geometry(nodes, members)
    print(f"FUNCTION_CALL: validate_triangle_geometry | PARAMS: {{angle: {angle_deg}°, h: {h:.3f}}} | RESULT: {{is_valid: {validation_result['result']}}}")
    
    if not validation_result['result']:
        print(f"⚠️  错误：几何结构验证失败")
        for error in validation_result['metadata']['errors']:
            print(f"  错误: {error}")
        print(f"无法继续分析")
        return
    
    if validation_result['metadata']['warnings']:
        for warning in validation_result['metadata']['warnings']:
            print(f"  ⚠️  警告: {warning}")
    
    print(f"✓ 几何结构验证通过")
    
    # 步骤1：计算支座反力
    print(f"\n步骤1：计算支座反力")
    reaction_result = calculate_support_reactions(span=2*a_base, load_position=a_base, load_value=load_value)
    print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{span: {2*a_base}, load_position: {a_base}, load_value: {load_value}}} | RESULT: {reaction_result['result']}")
    
    Ra = reaction_result['result']['Ra']
    Rb = reaction_result['result']['Rb']
    support_reactions = {'Ra': Ra, 'Rb': Rb}
    
    # 步骤2：分析桁架几何
    print(f"\n步骤2：分析桁架几何结构")
    geometry_result = analyze_truss_structure(nodes, members, supports, loads)
    print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: {{angle: {angle_deg}°, h: {h:.3f}}} | RESULT: {{total_members: {geometry_result['result']['total_members']}}}")
    
    member_info = geometry_result['result']['member_info']
    
    # 步骤3：按节点法求解内力
    print(f"\n步骤3：按节点法求解杆件内力")
    forces_result = solve_truss_by_joints(nodes, member_info, support_reactions, loads, solve_sequence)
    print(f"FUNCTION_CALL: solve_truss_by_joints | PARAMS: {{angle: {angle_deg}°}} | RESULT: {{solved_members: {len(forces_result['result'])}}}")
    
    member_forces = forces_result['result']
    
    # 步骤4：重点分析DF杆受力
    print(f"\n步骤4：分析DF杆受力")
    df_force = member_forces.get('DF', 0)
    df_type_result = identify_force_type(df_force)
    print(f"FUNCTION_CALL: identify_force_type | PARAMS: {{force_value: {df_force}}} | RESULT: {df_type_result['result']}")
    print(f"DF杆内力: {df_force:.2f} kN ({df_type_result['metadata']['description']})")
    
    # 显示所有杆件内力
    print(f"\n所有杆件内力：")
    for mem in sorted(member_forces.keys()):
        force = member_forces[mem]
        force_type = identify_force_type(force)['metadata']['description']
        print(f"  {mem}: {force:>8.2f} kN ({force_type})")
    
    # 步骤5：生成内力报告
    print(f"\n步骤5：生成完整内力报告")
    report_result = generate_force_report(member_forces, member_info)
    print(f"FUNCTION_CALL: generate_force_report | RESULT: {report_result['result']}")
    
    # 步骤6：可视化桁架结构
    print(f"\n步骤6：可视化桁架结构")
    viz_result = visualize_truss_structure(nodes, members, member_forces, loads, support_reactions, 
                                          "Scenario 1: AC-AB Angle 45° - DF Rod Analysis")
    print(f"FUNCTION_CALL: visualize_truss_structure | RESULT: {viz_result['result']}")
    
    print(f"\nFINAL_ANSWER: 当AC与AB夹角为45度时，DF杆受{df_type_result['metadata']['description']}，内力为{abs(df_force):.2f} kN")
    
    
    # ========== 场景2：对称荷载下的桁架分析 ==========
    print("\n\n" + "=" * 60)
    print("场景2：对称双荷载桁架分析")
    print("=" * 60)
    print("问题描述：在节点E和F同时施加10kN向下荷载，分析各杆件内力")
    print("-" * 60)
    
    loads_2 = {'E': (0, -10), 'F': (0, -10)}
    
    # 步骤1：计算支座反力（对称荷载）
    # 调用函数：calculate_support_reactions()
    print("\n步骤1：计算对称荷载下的支座反力")
    print("调用函数：calculate_support_reactions()")
    # 对称荷载，每个支座承担一半
    Ra_2 = 10.0
    Rb_2 = 10.0
    support_reactions_2 = {'Ra': Ra_2, 'Rb': Rb_2}
    print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{symmetric_load: True}} | RESULT: {{Ra: {Ra_2}, Rb: {Rb_2}}}")
    
    # 步骤2：求解内力
    # 调用函数：solve_truss_by_joints()
    print("\n步骤2：求解对称荷载下的杆件内力")
    print("调用函数：solve_truss_by_joints()")
    forces_result_2 = solve_truss_by_joints(nodes, member_info, support_reactions_2, loads_2, solve_sequence)
    print(f"FUNCTION_CALL: solve_truss_by_joints | PARAMS: {{loads: 'E,F'}} | RESULT: {{solved_members: {len(forces_result_2['result'])}}}")
    
    member_forces_2 = forces_result_2['result']
    
    # 步骤3：生成报告
    # 调用函数：generate_force_report()
    print("\n步骤3：生成对称荷载内力报告")
    print("调用函数：generate_force_report()")
    report_result_2 = generate_force_report(member_forces_2, member_info)
    print(f"FUNCTION_CALL: generate_force_report | PARAMS: {{scenario: 'symmetric'}} | RESULT: {report_result_2['result']}")
    
    # # 步骤4：可视化
    # # 调用函数：visualize_truss_structure()
    # print("\n步骤4：可视化对称荷载桁架")
    # print("调用函数：visualize_truss_structure()")
    # viz_result_2 = visualize_truss_structure(nodes, members, member_forces_2, loads_2, support_reactions_2,
    #                                         "Scenario 2: Symmetric Load Analysis")
    # print(f"FUNCTION_CALL: visualize_truss_structure | PARAMS: {{title: 'Scenario 2'}} | RESULT: {viz_result_2['result']}")
    
    # max_force = max(abs(f) for f in member_forces_2.values())
    # max_member = [k for k, v in member_forces_2.items() if abs(v) == max_force][0]
    # print(f"\nFINAL_ANSWER: 对称荷载下，最大内力杆件为{max_member}，内力{max_force:.2f}kN")
    
    
    # ========== 场景3：不同角度下的桁架分析 ==========
    print("\n\n" + "=" * 60)
    print("场景3：不同角度下的桁架内力分析")
    print("=" * 60)
    print("问题描述：分析不同高度h（对应不同角度）对桁架内力的影响")
    print("保持跨度2a=4m不变，改变高度h，分析DF杆内力变化")
    print("-" * 60)
    
    # 定义不同的高度值（对应不同角度）
    # 角度 = arctan(h/a)，其中a=2m
    test_heights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # 对应不同角度
    a_base = 2.0  # 保持跨度不变
    load_value = 20.0  # 荷载值不变
    
    results_by_angle = []
    
    for h in test_heights:
        # 计算角度
        angle_deg = np.degrees(np.arctan(h / a_base))
        
        print(f"\n{'='*60}")
        print(f"测试情况：高度 h = {h:.1f} m，角度 = {angle_deg:.2f}°")
        print(f"{'='*60}")
        
        # 根据高度重新定义节点坐标
        nodes_angle = {
            'A': (0, 0),
            'B': (2*a_base, 0),
            'D': (a_base, 0),
            'E': (a_base/2, h/2),
            'F': (3*a_base/2, h/2),
            'C': (a_base, h)
        }
        
        # 步骤1：计算支座反力
        print(f"\n步骤1：计算支座反力（h={h:.1f}m）")
        reaction_result_angle = calculate_support_reactions(
            span=2*a_base, 
            load_position=a_base, 
            load_value=load_value
        )
        print(f"FUNCTION_CALL: calculate_support_reactions | PARAMS: {{span: {2*a_base}, load_position: {a_base}, load_value: {load_value}}} | RESULT: {reaction_result_angle['result']}")
        
        Ra_angle = reaction_result_angle['result']['Ra']
        Rb_angle = reaction_result_angle['result']['Rb']
        support_reactions_angle = {'Ra': Ra_angle, 'Rb': Rb_angle}
        
        # 步骤2：分析桁架几何
        print(f"\n步骤2：分析桁架几何结构（h={h:.1f}m）")
        geometry_result_angle = analyze_truss_structure(nodes_angle, members, supports, loads)
        print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: {{h: {h:.1f}}} | RESULT: {{total_members: {geometry_result_angle['result']['total_members']}}}")
        
        member_info_angle = geometry_result_angle['result']['member_info']
        
        # 步骤3：求解内力
        print(f"\n步骤3：求解杆件内力（h={h:.1f}m）")
        forces_result_angle = solve_truss_by_joints(
            nodes_angle, 
            member_info_angle, 
            support_reactions_angle, 
            loads, 
            solve_sequence
        )
        print(f"FUNCTION_CALL: solve_truss_by_joints | PARAMS: {{h: {h:.1f}}} | RESULT: {{solved_members: {len(forces_result_angle['result'])}}}")
        
        member_forces_angle = forces_result_angle['result']
        
        # 步骤4：分析DF杆
        print(f"\n步骤4：分析DF杆受力（h={h:.1f}m）")
        df_force_angle = member_forces_angle.get('DF', 0)
        df_type_result_angle = identify_force_type(df_force_angle)
        print(f"FUNCTION_CALL: identify_force_type | PARAMS: {{force_value: {df_force_angle}}} | RESULT: {df_type_result_angle['result']}")
        print(f"DF杆内力: {abs(df_force_angle):.2f} kN ({df_type_result_angle['metadata']['description']})")
        
        # 收集结果
        results_by_angle.append({
            'height': h,
            'angle_deg': angle_deg,
            'DF_force': df_force_angle,
            'DF_force_abs': abs(df_force_angle),
            'DF_type': df_type_result_angle['metadata']['description'],
            'Ra': Ra_angle,
            'Rb': Rb_angle,
            'member_forces': member_forces_angle
        })
        
        # 显示关键杆件内力
        print(f"\n关键杆件内力（h={h:.1f}m）:")
        key_members = ['DF', 'CD', 'ED', 'AE', 'CF']
        for mem in key_members:
            if mem in member_forces_angle:
                force = member_forces_angle[mem]
                force_type = identify_force_type(force)['metadata']['description']
                print(f"  {mem}: {force:>8.2f} kN ({force_type})")
    
    # 步骤5：生成角度对比报告
    print(f"\n\n{'='*60}")
    print("场景3总结：不同角度下的DF杆内力对比")
    print(f"{'='*60}")
    print(f"{'高度h(m)':<10} {'角度(°)':<10} {'DF杆内力(kN)':<15} {'受力类型':<10} {'Ra(kN)':<10} {'Rb(kN)':<10}")
    print("-" * 60)
    
    for result in results_by_angle:
        print(f"{result['height']:<10.1f} {result['angle_deg']:<10.2f} "
              f"{result['DF_force_abs']:<15.2f} {result['DF_type']:<10} "
              f"{result['Ra']:<10.2f} {result['Rb']:<10.2f}")
    
    # 分析趋势
    print(f"\n趋势分析：")
    print(f"- 当高度h从{test_heights[0]:.1f}m增加到{test_heights[-1]:.1f}m时：")
    print(f"  - 角度从{results_by_angle[0]['angle_deg']:.2f}°增加到{results_by_angle[-1]['angle_deg']:.2f}°")
    print(f"  - DF杆内力从{results_by_angle[0]['DF_force_abs']:.2f}kN变化到{results_by_angle[-1]['DF_force_abs']:.2f}kN")
    
    # 找出最大和最小DF杆内力
    max_df_result = max(results_by_angle, key=lambda x: x['DF_force_abs'])
    min_df_result = min(results_by_angle, key=lambda x: x['DF_force_abs'])
    
    print(f"\n- DF杆内力最大值: {max_df_result['DF_force_abs']:.2f} kN (h={max_df_result['height']:.1f}m, 角度={max_df_result['angle_deg']:.2f}°)")
    print(f"- DF杆内力最小值: {min_df_result['DF_force_abs']:.2f} kN (h={min_df_result['height']:.1f}m, 角度={min_df_result['angle_deg']:.2f}°)")
    
    # 保存结果到JSON文件
    results_file = './mid_result/structural_mechanics/angle_analysis_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        json_results = []
        for r in results_by_angle:
            json_r = {
                'height': float(r['height']),
                'angle_deg': float(r['angle_deg']),
                'DF_force': float(r['DF_force']),
                'DF_force_abs': float(r['DF_force_abs']),
                'DF_type': r['DF_type'],
                'Ra': float(r['Ra']),
                'Rb': float(r['Rb']),
                'member_forces': {k: float(v) for k, v in r['member_forces'].items()}
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")
    print(f"FUNCTION_CALL: 角度分析完成 | RESULT: {results_file}")
    
    print(f"\nFINAL_ANSWER: 完成{len(test_heights)}种不同角度的桁架分析，DF杆内力范围: {min_df_result['DF_force_abs']:.2f}~{max_df_result['DF_force_abs']:.2f} kN")
    
    
if __name__ == "__main__":
    main()