# Filename: structural_mechanics_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/structural_mechanics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def count_truss_elements(node_coords: List[List[float]], 
                        member_connections: List[List[int]]) -> Dict:
    """
    统计桁架结构的基本元素数量
    
    Args:
        node_coords: 节点坐标列表 [[x1,y1], [x2,y2], ...]
        member_connections: 杆件连接关系 [[node_i, node_j], ...]
    
    Returns:
        {'result': {'joints': int, 'members': int}, 'metadata': {...}}
    """
    if not node_coords or not member_connections:
        raise ValueError("节点坐标和杆件连接不能为空")
    
    num_joints = len(node_coords)
    num_members = len(member_connections)
    
    # 验证连接关系的有效性
    for conn in member_connections:
        if len(conn) != 2:
            raise ValueError(f"杆件连接必须是两个节点: {conn}")
        if conn[0] >= num_joints or conn[1] >= num_joints or conn[0] < 0 or conn[1] < 0:
            raise ValueError(f"节点索引超出范围: {conn}")
    
    return {
        'result': {
            'joints': num_joints,
            'members': num_members
        },
        'metadata': {
            'description': '桁架基本元素统计',
            'node_count': num_joints,
            'member_count': num_members
        }
    }


def identify_support_reactions(support_types: List[Dict]) -> Dict:
    """
    识别支座类型并计算约束反力数
    
    Args:
        support_types: 支座信息列表 [{'node': int, 'type': str}, ...]
                      type可选: 'fixed' (固定铰支座, 2个反力), 
                               'roller' (滚动支座, 1个反力),
                               'fixed_end' (固定端, 3个反力)
    
    Returns:
        {'result': int, 'metadata': {...}}
    """
    reaction_map = {
        'fixed': 2,      # 固定铰支座：水平和竖向反力
        'roller': 1,     # 滚动支座：仅竖向反力
        'fixed_end': 3   # 固定端：水平、竖向反力和弯矩
    }
    
    total_reactions = 0
    support_details = []
    
    for support in support_types:
        s_type = support.get('type', '').lower()
        if s_type not in reaction_map:
            raise ValueError(f"不支持的支座类型: {s_type}. 可选: {list(reaction_map.keys())}")
        
        reactions = reaction_map[s_type]
        total_reactions += reactions
        support_details.append({
            'node': support['node'],
            'type': s_type,
            'reactions': reactions
        })
    
    return {
        'result': total_reactions,
        'metadata': {
            'description': '支座约束反力统计',
            'support_count': len(support_types),
            'support_details': support_details
        }
    }


def calculate_degree_of_indeterminacy(members: int, reactions: int, joints: int) -> Dict:
    """
    计算结构的超静定次数
    
    公式: n = m + r - 2j
    其中: m=杆件数, r=支座反力数, j=节点数
    
    Args:
        members: 杆件数量
        reactions: 支座约束反力数
        joints: 节点数量
    
    Returns:
        {'result': int, 'metadata': {...}}
    """
    if members < 0 or reactions < 0 or joints < 0:
        raise ValueError("杆件数、反力数和节点数必须为非负整数")
    
    n = members + reactions - 2 * joints
    
    if n > 0:
        structure_type = "超静定结构"
        stability = "稳定"
    elif n == 0:
        structure_type = "静定结构"
        stability = "需进一步检查几何稳定性"
    else:
        structure_type = "几何可变体系"
        stability = "不稳定"
    
    return {
        'result': n,
        'metadata': {
            'description': '超静定次数计算',
            'formula': 'n = m + r - 2j',
            'members': members,
            'reactions': reactions,
            'joints': joints,
            'structure_type': structure_type,
            'stability': stability
        }
    }


def check_geometric_stability(node_coords: List[List[float]], 
                              member_connections: List[List[int]],
                              support_nodes: List[int]) -> Dict:
    """
    检查结构的几何稳定性（简化判断）
    
    Args:
        node_coords: 节点坐标
        member_connections: 杆件连接
        support_nodes: 支座节点索引列表
    
    Returns:
        {'result': bool, 'metadata': {...}}
    """
    num_joints = len(node_coords)
    num_members = len(member_connections)
    
    # 检查是否有足够的支座
    if len(support_nodes) < 1:
        return {
            'result': False,
            'metadata': {
                'description': '几何稳定性检查',
                'reason': '缺少支座约束',
                'is_stable': False
            }
        }
    
    # 检查连通性（简化：检查是否所有节点都通过杆件连接）
    connected_nodes = set()
    for conn in member_connections:
        connected_nodes.add(conn[0])
        connected_nodes.add(conn[1])
    
    is_connected = len(connected_nodes) == num_joints
    
    return {
        'result': is_connected,
        'metadata': {
            'description': '几何稳定性检查',
            'connected_nodes': len(connected_nodes),
            'total_nodes': num_joints,
            'is_fully_connected': is_connected
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_truss_structure(node_coords: List[List[float]], 
                           member_connections: List[List[int]],
                           support_types: List[Dict]) -> Dict:
    """
    综合分析桁架结构的超静定特性
    
    Args:
        node_coords: 节点坐标
        member_connections: 杆件连接
        support_types: 支座类型信息
    
    Returns:
        {'result': {...}, 'metadata': {...}}
    """
    # 步骤1：统计基本元素
    elements = count_truss_elements(node_coords, member_connections)
    joints = elements['result']['joints']
    members = elements['result']['members']
    
    # 步骤2：计算支座反力
    reactions_result = identify_support_reactions(support_types)
    reactions = reactions_result['result']
    
    # 步骤3：计算超静定次数
    indeterminacy = calculate_degree_of_indeterminacy(members, reactions, joints)
    
    # 步骤4：检查几何稳定性
    support_nodes = [s['node'] for s in support_types]
    stability = check_geometric_stability(node_coords, member_connections, support_nodes)
    
    return {
        'result': {
            'degree_of_indeterminacy': indeterminacy['result'],
            'structure_type': indeterminacy['metadata']['structure_type'],
            'is_stable': stability['result'],
            'joints': joints,
            'members': members,
            'reactions': reactions
        },
        'metadata': {
            'description': '桁架结构综合分析',
            'elements_info': elements['metadata'],
            'support_info': reactions_result['metadata'],
            'indeterminacy_info': indeterminacy['metadata'],
            'stability_info': stability['metadata']
        }
    }


def save_analysis_report(analysis_result: Dict, filename: str = 'truss_analysis_report.json') -> Dict:
    """
    保存分析报告到文件
    
    Args:
        analysis_result: 分析结果字典
        filename: 输出文件名
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    filepath = os.path.join('./mid_result/structural_mechanics', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(filepath)
    
    return {
        'result': filepath,
        'metadata': {
            'description': '分析报告已保存',
            'file_type': 'json',
            'size': file_size,
            'path': filepath
        }
    }


# ==================== 第三层：可视化函数 ====================

def visualize_truss_structure(node_coords: List[List[float]], 
                              member_connections: List[List[int]],
                              support_types: List[Dict],
                              analysis_result: Optional[Dict] = None,
                              filename: str = 'truss_structure.png') -> Dict:
    """
    可视化桁架结构
    
    Args:
        node_coords: 节点坐标
        member_connections: 杆件连接
        support_types: 支座类型
        analysis_result: 分析结果（可选）
        filename: 输出文件名
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制杆件
    for conn in member_connections:
        x = [node_coords[conn[0]][0], node_coords[conn[1]][0]]
        y = [node_coords[conn[0]][1], node_coords[conn[1]][1]]
        ax.plot(x, y, 'b-', linewidth=2)
    
    # 绘制节点
    for i, coord in enumerate(node_coords):
        ax.plot(coord[0], coord[1], 'ko', markersize=8)
        ax.text(coord[0], coord[1] + 0.15, f'{i}', ha='center', fontsize=9)
    
    # 绘制支座
    support_markers = {
        'fixed': '^',
        'roller': 's',
        'fixed_end': 'D'
    }
    support_colors = {
        'fixed': 'red',
        'roller': 'green',
        'fixed_end': 'purple'
    }
    
    for support in support_types:
        node_idx = support['node']
        s_type = support['type']
        coord = node_coords[node_idx]
        marker = support_markers.get(s_type, 'o')
        color = support_colors.get(s_type, 'red')
        ax.plot(coord[0], coord[1], marker, markersize=12, color=color, 
                label=f'{s_type.capitalize()} Support' if s_type not in [s['type'] for s in support_types[:support_types.index(support)]] else '')
    
    # 添加分析结果文本
    if analysis_result:
        result_text = f"Degree of Indeterminacy: {analysis_result['result']['degree_of_indeterminacy']}\n"
        result_text += f"Structure Type: {analysis_result['result']['structure_type']}\n"
        result_text += f"Joints: {analysis_result['result']['joints']}, "
        result_text += f"Members: {analysis_result['result']['members']}, "
        result_text += f"Reactions: {analysis_result['result']['reactions']}"
        
        ax.text(0.02, 0.98, result_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Truss Structure Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right')
    
    filepath = os.path.join('./tool_images', filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'description': '桁架结构可视化',
            'file_type': 'png',
            'path': filepath
        }
    }


def create_comparison_chart(scenarios: List[Dict], filename: str = 'indeterminacy_comparison.png') -> Dict:
    """
    创建多个场景的超静定次数对比图
    
    Args:
        scenarios: 场景列表 [{'name': str, 'degree': int, 'members': int, 'reactions': int, 'joints': int}, ...]
        filename: 输出文件名
    
    Returns:
        {'result': str, 'metadata': {...}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    names = [s['name'] for s in scenarios]
    degrees = [s['degree'] for s in scenarios]
    
    # 图1：超静定次数对比
    colors = ['green' if d == 0 else 'orange' if d > 0 else 'red' for d in degrees]
    bars = ax1.bar(names, degrees, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel('Degree of Indeterminacy', fontsize=11)
    ax1.set_title('Comparison of Structural Indeterminacy', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, degree in zip(bars, degrees):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{degree}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # 图2：结构参数对比
    x = np.arange(len(names))
    width = 0.25
    
    members = [s['members'] for s in scenarios]
    reactions = [s['reactions'] for s in scenarios]
    joints = [s['joints'] for s in scenarios]
    
    ax2.bar(x - width, members, width, label='Members (m)', alpha=0.8)
    ax2.bar(x, reactions, width, label='Reactions (r)', alpha=0.8)
    ax2.bar(x + width, joints, width, label='Joints (j)', alpha=0.8)
    
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Structural Parameters Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    filepath = os.path.join('./tool_images', filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'description': '超静定次数对比图',
            'file_type': 'png',
            'scenarios_count': len(scenarios),
            'path': filepath
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    演示三个场景的桁架结构分析
    """
    
    print("=" * 60)
    print("场景1：原始问题 - 双层桁架结构超静定次数分析")
    print("=" * 60)
    print("问题描述：分析图中所示的双层桁架结构的超静定次数")
    print("结构特征：上下两层水平杆件，通过斜杆和竖杆连接，左侧两个固定支座")
    print("-" * 60)
    
    # 根据图片分析构建结构模型
    # 节点编号：上层0-3（从左到右），下层4-7（从左到右）
    node_coords_1 = [
        [0, 1], [1, 1], [2, 1],     # 上层节点 0-3
        [0, 0], [1, 0], [2, 0], [3, 0]       # 下层节点 4-7
    ]
    
    # 杆件连接（根据图片的三角形桁架单元）
    member_connections_1 = [
        # 上层水平杆
        [0, 1], [1, 2], 
        # 下层水平杆
        [3, 4], [4, 5], [5, 6],
        # 竖向杆
        [1, 4], [2, 5],
        # 斜杆（形成三角形）
        [0, 4], [1, 3], [1, 5], [2, 4], [2, 6]
    ]
    
    # 支座：左侧上下各一个固定铰支座
    support_types_1 = [
        {'node': 0, 'type': 'fixed'},  # 左上固定支座
        {'node': 3, 'type': 'fixed'}   # 左下固定支座
    ]
    
    # 步骤1：统计结构元素
    print("\n步骤1：统计结构基本元素")
    print("调用函数：count_truss_elements()")
    elements_1 = count_truss_elements(node_coords_1, member_connections_1)
    print(f"FUNCTION_CALL: count_truss_elements | PARAMS: nodes={len(node_coords_1)}, connections={len(member_connections_1)} | RESULT: {elements_1['result']}")
    
    # 步骤2：识别支座反力
    print("\n步骤2：识别支座约束反力")
    print("调用函数：identify_support_reactions()")
    reactions_1 = identify_support_reactions(support_types_1)
    print(f"FUNCTION_CALL: identify_support_reactions | PARAMS: {support_types_1} | RESULT: {reactions_1['result']}")
    
    # 步骤3：计算超静定次数
    print("\n步骤3：计算超静定次数")
    print("调用函数：calculate_degree_of_indeterminacy()")
    params_1 = {
        'members': elements_1['result']['members'],
        'reactions': reactions_1['result'],
        'joints': elements_1['result']['joints']
    }
    indeterminacy_1 = calculate_degree_of_indeterminacy(**params_1)
    print(f"FUNCTION_CALL: calculate_degree_of_indeterminacy | PARAMS: {params_1} | RESULT: {indeterminacy_1['result']}")
    print(f"结构类型: {indeterminacy_1['metadata']['structure_type']}")
    
    # 步骤4：综合分析
    print("\n步骤4：综合结构分析")
    print("调用函数：analyze_truss_structure()")
    analysis_1 = analyze_truss_structure(node_coords_1, member_connections_1, support_types_1)
    print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: nodes={len(node_coords_1)}, members={len(member_connections_1)}, supports={len(support_types_1)} | RESULT: {analysis_1['result']}")
    
    # 步骤5：保存分析报告
    print("\n步骤5：保存分析报告")
    print("调用函数：save_analysis_report()")
    report_1 = save_analysis_report(analysis_1, 'scenario1_report.json')
    print(f"FUNCTION_CALL: save_analysis_report | PARAMS: filename='scenario1_report.json' | RESULT: {report_1['result']}")
    
    # 步骤6：可视化结构
    print("\n步骤6：可视化桁架结构")
    print("调用函数：visualize_truss_structure()")
    viz_1 = visualize_truss_structure(node_coords_1, member_connections_1, support_types_1, analysis_1, 'scenario1_structure.png')
    print(f"FUNCTION_CALL: visualize_truss_structure | PARAMS: filename='scenario1_structure.png' | RESULT: {viz_1['result']}")
    
    print(f"\nFINAL_ANSWER: {indeterminacy_1['result']}")
    
    
    print("\n" + "=" * 60)
    print("场景2：简单三角形桁架 - 静定结构验证")
    print("=" * 60)
    print("问题描述：分析一个简单的三角形桁架，验证静定结构的特征")
    print("-" * 60)
    
    # 简单三角形桁架：3个节点，3个杆件，1个固定支座+1个滚动支座
    node_coords_2 = [
        [0, 0], [2, 0], [1, 1.5]
    ]
    
    member_connections_2 = [
        [0, 1], [1, 2], [2, 0]
    ]
    
    support_types_2 = [
        {'node': 0, 'type': 'fixed'},
        {'node': 1, 'type': 'roller'}
    ]
    
    print("\n步骤1：统计结构元素")
    print("调用函数：count_truss_elements()")
    elements_2 = count_truss_elements(node_coords_2, member_connections_2)
    print(f"FUNCTION_CALL: count_truss_elements | PARAMS: nodes={len(node_coords_2)}, connections={len(member_connections_2)} | RESULT: {elements_2['result']}")
    
    print("\n步骤2：计算超静定次数")
    print("调用函数：analyze_truss_structure()")
    analysis_2 = analyze_truss_structure(node_coords_2, member_connections_2, support_types_2)
    print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: nodes=3, members=3, supports=2 | RESULT: {analysis_2['result']}")
    
    print("\n步骤3：可视化结构")
    print("调用函数：visualize_truss_structure()")
    viz_2 = visualize_truss_structure(node_coords_2, member_connections_2, support_types_2, analysis_2, 'scenario2_structure.png')
    print(f"FUNCTION_CALL: visualize_truss_structure | PARAMS: filename='scenario2_structure.png' | RESULT: {viz_2['result']}")
    
    print(f"\nFINAL_ANSWER: {analysis_2['result']['degree_of_indeterminacy']}")
    
    
    print("\n" + "=" * 60)
    print("场景3：复杂桁架对比分析 - 不同支座配置的影响")
    print("=" * 60)
    print("问题描述：对比分析相同杆件布局但不同支座配置的超静定次数变化")
    print("-" * 60)
    
    # 使用场景1的节点和杆件，但改变支座配置
    # 配置A：3个固定支座（过约束）
    support_types_3a = [
        {'node': 0, 'type': 'fixed'},
        {'node': 4, 'type': 'fixed'},
        {'node': 7, 'type': 'fixed'}
    ]
    
    # 配置B：1个固定支座 + 1个滚动支座（欠约束）
    support_types_3b = [
        {'node': 0, 'type': 'fixed'},
        {'node': 7, 'type': 'roller'}
    ]
    
    print("\n步骤1：分析配置A（3个固定支座）")
    print("调用函数：analyze_truss_structure()")
    analysis_3a = analyze_truss_structure(node_coords_1, member_connections_1, support_types_3a)
    print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: supports=3_fixed | RESULT: {analysis_3a['result']}")
    
    print("\n步骤2：分析配置B（1固定+1滚动）")
    print("调用函数：analyze_truss_structure()")
    analysis_3b = analyze_truss_structure(node_coords_1, member_connections_1, support_types_3b)
    print(f"FUNCTION_CALL: analyze_truss_structure | PARAMS: supports=1_fixed+1_roller | RESULT: {analysis_3b['result']}")
    
    print("\n步骤3：创建对比图表")
    print("调用函数：create_comparison_chart()")
    scenarios_data = [
        {
            'name': 'Original\n(2 Fixed)',
            'degree': analysis_1['result']['degree_of_indeterminacy'],
            'members': analysis_1['result']['members'],
            'reactions': analysis_1['result']['reactions'],
            'joints': analysis_1['result']['joints']
        },
        {
            'name': 'Config A\n(3 Fixed)',
            'degree': analysis_3a['result']['degree_of_indeterminacy'],
            'members': analysis_3a['result']['members'],
            'reactions': analysis_3a['result']['reactions'],
            'joints': analysis_3a['result']['joints']
        },
        {
            'name': 'Config B\n(1F+1R)',
            'degree': analysis_3b['result']['degree_of_indeterminacy'],
            'members': analysis_3b['result']['members'],
            'reactions': analysis_3b['result']['reactions'],
            'joints': analysis_3b['result']['joints']
        }
    ]
    
    comparison = create_comparison_chart(scenarios_data, 'scenario3_comparison.png')
    print(f"FUNCTION_CALL: create_comparison_chart | PARAMS: scenarios=3 | RESULT: {comparison['result']}")
    
    print("\n对比结果总结：")
    print(f"- 原始配置（2个固定支座）：超静定次数 = {analysis_1['result']['degree_of_indeterminacy']}")
    print(f"- 配置A（3个固定支座）：超静定次数 = {analysis_3a['result']['degree_of_indeterminacy']}")
    print(f"- 配置B（1固定+1滚动）：超静定次数 = {analysis_3b['result']['degree_of_indeterminacy']}")
    
    print(f"\nFINAL_ANSWER: Original={analysis_1['result']['degree_of_indeterminacy']}, ConfigA={analysis_3a['result']['degree_of_indeterminacy']}, ConfigB={analysis_3b['result']['degree_of_indeterminacy']}")


if __name__ == "__main__":
    main()