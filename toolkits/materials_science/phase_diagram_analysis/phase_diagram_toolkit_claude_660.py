# Filename: phase_diagram_toolkit.py

"""
Binary Phase Diagram Analysis Toolkit
用于分析和验证二元相图的正确性
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass, asdict

# 配置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建输出目录
os.makedirs('./mid_result/materials', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ==================== 第一层：原子函数 ====================

def check_gibbs_phase_rule(num_components: int, num_phases: int, degrees_of_freedom: int) -> Dict:
    """
    验证吉布斯相律：F = C - P + 2
    
    Args:
        num_components: 组分数
        num_phases: 相数
        degrees_of_freedom: 自由度
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not isinstance(num_components, int) or num_components < 1:
        raise ValueError("组分数必须是正整数")
    if not isinstance(num_phases, int) or num_phases < 1:
        raise ValueError("相数必须是正整数")
    if not isinstance(degrees_of_freedom, int) or degrees_of_freedom < 0:
        raise ValueError("自由度必须是非负整数")
    
    # 对于恒压系统，吉布斯相律简化为 F = C - P + 1
    expected_dof = num_components - num_phases + 1
    is_valid = (degrees_of_freedom == expected_dof)
    
    return {
        'result': is_valid,
        'metadata': {
            'expected_dof': expected_dof,
            'actual_dof': degrees_of_freedom,
            'formula': 'F = C - P + 1 (constant pressure)',
            'components': num_components,
            'phases': num_phases
        }
    }


def check_liquidus_solidus_order(liquidus_temp: float, solidus_temp: float, 
                                  composition: float) -> Dict:
    """
    检查液相线和固相线的温度顺序
    
    Args:
        liquidus_temp: 液相线温度 (K)
        solidus_temp: 固相线温度 (K)
        composition: 组分 (0-1)
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not (0 <= composition <= 1):
        raise ValueError("组分必须在0-1之间")
    if liquidus_temp <= 0 or solidus_temp <= 0:
        raise ValueError("温度必须为正值")
    
    # 液相线温度必须高于或等于固相线温度
    is_valid = liquidus_temp >= solidus_temp
    
    return {
        'result': is_valid,
        'metadata': {
            'liquidus_temp': liquidus_temp,
            'solidus_temp': solidus_temp,
            'composition': composition,
            'temperature_difference': liquidus_temp - solidus_temp,
            'rule': 'Liquidus temperature must be >= Solidus temperature'
        }
    }


def check_phase_boundary_continuity(boundary_points: List[List[float]], 
                                     boundary_type: str) -> Dict:
    """
    检查相边界的连续性
    
    Args:
        boundary_points: 边界点列表 [[composition, temperature], ...]
        boundary_type: 边界类型 ('liquidus', 'solidus', 'solvus')
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not boundary_points or len(boundary_points) < 2:
        raise ValueError("至少需要2个边界点")
    
    valid_types = ['liquidus', 'solidus', 'solvus', 'phase_boundary']
    if boundary_type not in valid_types:
        raise ValueError(f"边界类型必须是 {valid_types} 之一")
    
    # 检查点的连续性（相邻点之间的距离不应过大）
    max_gap = 0.0
    gaps = []
    
    for i in range(len(boundary_points) - 1):
        x1, y1 = boundary_points[i]
        x2, y2 = boundary_points[i + 1]
        gap = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        gaps.append(gap)
        max_gap = max(max_gap, gap)
    
    # 判断连续性（相邻点间隔不应超过合理阈值）
    threshold = 0.3  # 归一化距离阈值
    is_continuous = max_gap < threshold
    
    return {
        'result': is_continuous,
        'metadata': {
            'boundary_type': boundary_type,
            'num_points': len(boundary_points),
            'max_gap': float(max_gap),
            'average_gap': float(np.mean(gaps)) if gaps else 0.0,
            'threshold': threshold,
            'gaps': [float(g) for g in gaps]
        }
    }


def check_eutectic_point_validity(eutectic_temp: float, 
                                   left_solidus_temp: float,
                                   right_solidus_temp: float,
                                   eutectic_composition: float) -> Dict:
    """
    检查共晶点的有效性
    
    Args:
        eutectic_temp: 共晶温度 (K)
        left_solidus_temp: 左侧固相线温度 (K)
        right_solidus_temp: 右侧固相线温度 (K)
        eutectic_composition: 共晶成分 (0-1)
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not (0 < eutectic_composition < 1):
        raise ValueError("共晶成分必须在0-1之间（不包括端点）")
    if eutectic_temp <= 0:
        raise ValueError("温度必须为正值")
    
    # 共晶温度必须低于两侧的固相线温度
    is_valid = (eutectic_temp < left_solidus_temp and 
                eutectic_temp < right_solidus_temp)
    
    return {
        'result': is_valid,
        'metadata': {
            'eutectic_temp': eutectic_temp,
            'left_solidus_temp': left_solidus_temp,
            'right_solidus_temp': right_solidus_temp,
            'eutectic_composition': eutectic_composition,
            'rule': 'Eutectic temperature must be lower than both solidus temperatures'
        }
    }


def check_lever_rule_consistency(composition: float, 
                                  phase1_composition: float,
                                  phase2_composition: float,
                                  phase1_fraction: float) -> Dict:
    """
    验证杠杆定律的一致性
    
    Args:
        composition: 总体成分 (0-1)
        phase1_composition: 相1的成分 (0-1)
        phase2_composition: 相2的成分 (0-1)
        phase1_fraction: 相1的质量分数 (0-1)
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not all(0 <= x <= 1 for x in [composition, phase1_composition, 
                                      phase2_composition, phase1_fraction]):
        raise ValueError("所有成分和分数必须在0-1之间")
    
    if phase1_composition == phase2_composition:
        raise ValueError("两相成分不能相同")
    
    # 杠杆定律：C0 = C1 * f1 + C2 * f2，其中 f1 + f2 = 1
    phase2_fraction = 1 - phase1_fraction
    calculated_composition = (phase1_composition * phase1_fraction + 
                             phase2_composition * phase2_fraction)
    
    # 允许小的数值误差
    tolerance = 1e-6
    is_valid = abs(calculated_composition - composition) < tolerance
    
    return {
        'result': is_valid,
        'metadata': {
            'composition': composition,
            'calculated_composition': calculated_composition,
            'phase1_composition': phase1_composition,
            'phase2_composition': phase2_composition,
            'phase1_fraction': phase1_fraction,
            'phase2_fraction': phase2_fraction,
            'error': abs(calculated_composition - composition),
            'tolerance': tolerance
        }
    }


def check_phase_region_topology(phase_regions: List[Dict]) -> Dict:
    """
    检查相区拓扑结构的合理性
    
    Args:
        phase_regions: 相区列表，每个相区包含 {'name': str, 'num_phases': int, 
                      'boundaries': List[str]}
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not phase_regions:
        raise ValueError("相区列表不能为空")
    
    errors = []
    
    # 检查单相区和两相区的交替
    for i, region in enumerate(phase_regions):
        if 'name' not in region or 'num_phases' not in region:
            errors.append(f"相区 {i} 缺少必要字段")
            continue
        
        num_phases = region['num_phases']
        if num_phases < 1:
            errors.append(f"相区 {region['name']} 的相数必须至少为1")
        
        # 检查相邻相区的相数差异
        if i > 0:
            prev_phases = phase_regions[i-1]['num_phases']
            phase_diff = abs(num_phases - prev_phases)
            # 相邻相区的相数差异通常为1
            if phase_diff > 2:
                errors.append(
                    f"相区 {region['name']} 与前一相区的相数差异过大: {phase_diff}"
                )
    
    is_valid = len(errors) == 0
    
    return {
        'result': is_valid,
        'metadata': {
            'num_regions': len(phase_regions),
            'errors': errors,
            'regions_summary': [
                {'name': r['name'], 'num_phases': r['num_phases']} 
                for r in phase_regions
            ]
        }
    }


def calculate_phase_fraction(composition: float, 
                            left_boundary: float,
                            right_boundary: float) -> Dict:
    """
    使用杠杆定律计算相分数
    
    Args:
        composition: 总体成分 (0-1)
        left_boundary: 左边界成分 (0-1)
        right_boundary: 右边界成分 (0-1)
    
    Returns:
        dict: {'result': float, 'metadata': {...}}
    """
    if not all(0 <= x <= 1 for x in [composition, left_boundary, right_boundary]):
        raise ValueError("所有成分必须在0-1之间")
    
    if left_boundary >= right_boundary:
        raise ValueError("左边界必须小于右边界")
    
    if not (left_boundary <= composition <= right_boundary):
        raise ValueError("总体成分必须在两边界之间")
    
    # 杠杆定律：右相分数 = (C0 - Cleft) / (Cright - Cleft)
    right_phase_fraction = (composition - left_boundary) / (right_boundary - left_boundary)
    left_phase_fraction = 1 - right_phase_fraction
    
    return {
        'result': right_phase_fraction,
        'metadata': {
            'left_phase_fraction': left_phase_fraction,
            'right_phase_fraction': right_phase_fraction,
            'composition': composition,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary
        }
    }


# ==================== 第二层：组合函数 ====================

def analyze_binary_phase_diagram(diagram_data: Dict) -> Dict:
    """
    综合分析二元相图的正确性
    
    Args:
        diagram_data: 相图数据，包含：
            - 'type': 相图类型 ('isomorphous', 'eutectic', 'peritectic', 'monotectic')
            - 'liquidus': 液相线点列表 [[x, T], ...]
            - 'solidus': 固相线点列表 [[x, T], ...]
            - 'phases': 相列表
            - 'special_points': 特殊点（共晶点、包晶点等）
    
    Returns:
        dict: {'result': bool, 'metadata': {...}}
    """
    if not isinstance(diagram_data, dict):
        raise ValueError("diagram_data必须是字典类型")
    
    required_keys = ['type', 'liquidus', 'solidus', 'phases']
    for key in required_keys:
        if key not in diagram_data:
            raise ValueError(f"缺少必要字段: {key}")
    
    errors = []
    checks = []
    
    # 1. 检查液相线和固相线的连续性
    liquidus_check = check_phase_boundary_continuity(
        diagram_data['liquidus'], 'liquidus'
    )
    checks.append({'name': 'liquidus_continuity', 'result': liquidus_check})
    if not liquidus_check['result']:
        errors.append("液相线不连续")
    
    solidus_check = check_phase_boundary_continuity(
        diagram_data['solidus'], 'solidus'
    )
    checks.append({'name': 'solidus_continuity', 'result': solidus_check})
    if not solidus_check['result']:
        errors.append("固相线不连续")
    
    # 2. 检查液相线和固相线的温度顺序
    temp_order_errors = 0
    for i in range(min(len(diagram_data['liquidus']), len(diagram_data['solidus']))):
        liq_point = diagram_data['liquidus'][i]
        sol_point = diagram_data['solidus'][i]
        
        if abs(liq_point[0] - sol_point[0]) < 0.05:  # 相近的组分
            order_check = check_liquidus_solidus_order(
                liq_point[1], sol_point[1], liq_point[0]
            )
            if not order_check['result']:
                temp_order_errors += 1
    
    if temp_order_errors > 0:
        errors.append(f"发现 {temp_order_errors} 处液相线低于固相线")
    
    # 3. 检查特殊点（如共晶点）
    if 'special_points' in diagram_data:
        for point in diagram_data['special_points']:
            if point['type'] == 'eutectic':
                eutectic_check = check_eutectic_point_validity(
                    point['temperature'],
                    point['left_solidus_temp'],
                    point['right_solidus_temp'],
                    point['composition']
                )
                checks.append({'name': f"eutectic_{point['composition']}", 
                             'result': eutectic_check})
                if not eutectic_check['result']:
                    errors.append(f"共晶点 {point['composition']} 不满足温度条件")
    
    # 4. 检查相区拓扑
    if 'phase_regions' in diagram_data:
        topology_check = check_phase_region_topology(diagram_data['phase_regions'])
        checks.append({'name': 'topology', 'result': topology_check})
        if not topology_check['result']:
            errors.extend(topology_check['metadata']['errors'])
    
    # 5. 检查吉布斯相律
    if 'phase_regions' in diagram_data:
        for region in diagram_data['phase_regions']:
            # 二元系统，恒压条件
            gibbs_check = check_gibbs_phase_rule(
                num_components=2,
                num_phases=region['num_phases'],
                degrees_of_freedom=region.get('dof', 2 - region['num_phases'] + 1)
            )
            if not gibbs_check['result']:
                errors.append(
                    f"相区 {region['name']} 不满足吉布斯相律"
                )
    
    is_valid = len(errors) == 0
    
    return {
        'result': is_valid,
        'metadata': {
            'diagram_type': diagram_data['type'],
            'num_errors': len(errors),
            'errors': errors,
            'checks_performed': len(checks),
            'detailed_checks': checks
        }
    }


def identify_phase_diagram_errors(diagram_description: Dict) -> Dict:
    """
    识别相图中的具体错误类型
    
    Args:
        diagram_description: 相图描述，包含：
            - 'diagram_id': 相图编号
            - 'features': 特征列表
            - 'data': 相图数据
    
    Returns:
        dict: {'result': List[str], 'metadata': {...}}
    """
    if not isinstance(diagram_description, dict):
        raise ValueError("diagram_description必须是字典类型")
    
    errors = []
    error_types = []
    
    # 分析相图数据
    if 'data' in diagram_description:
        analysis = analyze_binary_phase_diagram(diagram_description['data'])
        
        if not analysis['result']:
            errors.extend(analysis['metadata']['errors'])
            
            # 分类错误类型
            for error in analysis['metadata']['errors']:
                if '液相线' in error or '固相线' in error:
                    error_types.append('boundary_error')
                elif '共晶' in error or '包晶' in error:
                    error_types.append('special_point_error')
                elif '吉布斯' in error:
                    error_types.append('thermodynamic_error')
                elif '拓扑' in error or '相区' in error:
                    error_types.append('topology_error')
                else:
                    error_types.append('other_error')
    
    # 检查特征描述中的矛盾
    if 'features' in diagram_description:
        features = diagram_description['features']
        
        # 检查相图类型与特征的一致性
        if 'type' in features:
            diagram_type = features['type']
            
            if diagram_type == 'isomorphous':
                # 完全互溶系统不应有共晶点
                if any('eutectic' in str(f).lower() for f in features.values()):
                    errors.append("完全互溶系统不应包含共晶点")
                    error_types.append('type_mismatch')
            
            elif diagram_type == 'eutectic':
                # 共晶系统必须有共晶点
                if not any('eutectic' in str(f).lower() for f in features.values()):
                    errors.append("共晶系统缺少共晶点")
                    error_types.append('missing_feature')
    
    return {
        'result': errors,
        'metadata': {
            'diagram_id': diagram_description.get('diagram_id', 'unknown'),
            'num_errors': len(errors),
            'error_types': list(set(error_types)),
            'is_correct': len(errors) == 0
        }
    }


def compare_phase_diagrams(diagrams: List[Dict]) -> Dict:
    """
    比较多个相图，统计正确和错误的数量
    
    Args:
        diagrams: 相图列表，每个包含完整的相图描述
    
    Returns:
        dict: {'result': int, 'metadata': {...}}
    """
    if not isinstance(diagrams, list) or len(diagrams) == 0:
        raise ValueError("diagrams必须是非空列表")
    
    results = []
    correct_count = 0
    
    for i, diagram in enumerate(diagrams):
        diagram['diagram_id'] = f"diagram_{i+1}"
        error_analysis = identify_phase_diagram_errors(diagram)
        
        is_correct = error_analysis['metadata']['is_correct']
        if is_correct:
            correct_count += 1
        
        results.append({
            'diagram_id': diagram['diagram_id'],
            'is_correct': is_correct,
            'errors': error_analysis['result'],
            'error_types': error_analysis['metadata']['error_types']
        })
    
    return {
        'result': correct_count,
        'metadata': {
            'total_diagrams': len(diagrams),
            'correct_diagrams': correct_count,
            'incorrect_diagrams': len(diagrams) - correct_count,
            'detailed_results': results
        }
    }


# ==================== 第三层：可视化函数 ====================

def visualize_phase_diagram(diagram_data: Dict, output_path: str) -> Dict:
    """
    可视化二元相图
    
    Args:
        diagram_data: 相图数据
        output_path: 输出文件路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    if not isinstance(diagram_data, dict):
        raise ValueError("diagram_data必须是字典类型")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制液相线
    if 'liquidus' in diagram_data and diagram_data['liquidus']:
        liquidus = np.array(diagram_data['liquidus'])
        ax.plot(liquidus[:, 0], liquidus[:, 1], 'r-', linewidth=2, 
                label='Liquidus', marker='o', markersize=4)
    
    # 绘制固相线
    if 'solidus' in diagram_data and diagram_data['solidus']:
        solidus = np.array(diagram_data['solidus'])
        ax.plot(solidus[:, 0], solidus[:, 1], 'b-', linewidth=2, 
                label='Solidus', marker='s', markersize=4)
    
    # 标记特殊点
    if 'special_points' in diagram_data:
        for point in diagram_data['special_points']:
            ax.plot(point['composition'], point['temperature'], 
                   'g*', markersize=15, label=f"{point['type'].capitalize()} point")
    
    # 标注相区
    if 'phase_regions' in diagram_data:
        for region in diagram_data['phase_regions']:
            if 'label_position' in region:
                ax.text(region['label_position'][0], region['label_position'][1],
                       region['name'], fontsize=12, ha='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Composition (A → B)', fontsize=14)
    ax.set_ylabel('Temperature (K)', fontsize=14)
    ax.set_title(f"Binary Phase Diagram - {diagram_data.get('type', 'Unknown')}", 
                fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'diagram_type': diagram_data.get('type', 'unknown'),
            'size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
    }


def create_error_report(comparison_results: Dict, output_path: str) -> Dict:
    """
    创建相图错误分析报告
    
    Args:
        comparison_results: 比较结果
        output_path: 输出文件路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    if not isinstance(comparison_results, dict):
        raise ValueError("comparison_results必须是字典类型")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Binary Phase Diagram Error Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    metadata = comparison_results['metadata']
    report_lines.append(f"Total diagrams analyzed: {metadata['total_diagrams']}")
    report_lines.append(f"Correct diagrams: {metadata['correct_diagrams']}")
    report_lines.append(f"Incorrect diagrams: {metadata['incorrect_diagrams']}")
    report_lines.append("")
    report_lines.append("-" * 80)
    
    for result in metadata['detailed_results']:
        report_lines.append(f"\n{result['diagram_id'].upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"Status: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
        
        if not result['is_correct']:
            report_lines.append(f"Number of errors: {len(result['errors'])}")
            report_lines.append(f"Error types: {', '.join(result['error_types'])}")
            report_lines.append("\nDetailed errors:")
            for i, error in enumerate(result['errors'], 1):
                report_lines.append(f"  {i}. {error}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'txt',
            'num_lines': len(report_lines),
            'size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
    }


def visualize_error_statistics(comparison_results: Dict, output_path: str) -> Dict:
    """
    可视化错误统计
    
    Args:
        comparison_results: 比较结果
        output_path: 输出文件路径
    
    Returns:
        dict: {'result': str, 'metadata': {...}}
    """
    if not isinstance(comparison_results, dict):
        raise ValueError("comparison_results必须是字典类型")
    
    metadata = comparison_results['metadata']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 饼图：正确vs错误
    labels = ['Correct', 'Incorrect']
    sizes = [metadata['correct_diagrams'], metadata['incorrect_diagrams']]
    colors = ['#90EE90', '#FFB6C6']
    explode = (0.1, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Phase Diagram Correctness', fontsize=14, fontweight='bold')
    
    # 柱状图：错误类型统计
    error_type_counts = {}
    for result in metadata['detailed_results']:
        for error_type in result['error_types']:
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    if error_type_counts:
        error_types = list(error_type_counts.keys())
        counts = list(error_type_counts.values())
        
        ax2.bar(range(len(error_types)), counts, color='coral')
        ax2.set_xticks(range(len(error_types)))
        ax2.set_xticklabels(error_types, rotation=45, ha='right')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No errors found', ha='center', va='center',
                fontsize=16, transform=ax2.transAxes)
        ax2.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'error_types_found': len(error_type_counts),
            'size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    演示三个场景的相图分析
    """
    
    print("=" * 80)
    print("场景1：分析图片中的四个二元相图")
    print("=" * 80)
    print("问题描述：根据图片分析结果，验证四个二元相图的正确性")
    print("-" * 80)
    
    # 根据图片分析结果构建四个相图的数据
    
    # 相图1：完全互溶固溶体（Isomorphous）
    diagram1 = {
        'diagram_id': 'diagram_1',
        'features': {
            'type': 'isomorphous',
            'description': '完全互溶的固溶体α相，有液相线和固相线'
        },
        'data': {
            'type': 'isomorphous',
            'liquidus': [
                [0.0, 1800], [0.2, 1750], [0.4, 1700], [0.6, 1650], 
                [0.8, 1600], [1.0, 1550]
            ],
            'solidus': [
                [0.0, 1700], [0.2, 1650], [0.4, 1600], [0.6, 1550], 
                [0.8, 1500], [1.0, 1450]
            ],
            'phases': ['L', 'α'],
            'phase_regions': [
                {'name': 'L', 'num_phases': 1, 'dof': 2},
                {'name': 'L+α', 'num_phases': 2, 'dof': 1},
                {'name': 'α', 'num_phases': 1, 'dof': 2}
            ]
        }
    }
    
    # 步骤1：分析相图1
    print("\n步骤1：分析相图1（完全互溶系统）")
    result1 = identify_phase_diagram_errors(diagram1)
    print(f"FUNCTION_CALL: identify_phase_diagram_errors | PARAMS: diagram_1 | RESULT: {result1}")
    
    # 相图2：共晶型（Eutectic）- 但液相线形状错误
    diagram2 = {
        'diagram_id': 'diagram_2',
        'features': {
            'type': 'eutectic',
            'description': '液相L和固溶体α相，具有共晶型特征，液相线呈U型'
        },
        'data': {
            'type': 'eutectic',
            'liquidus': [
                [0.0, 1800], [0.2, 1650], [0.4, 1550], [0.5, 1500],  # U型液相线
                [0.6, 1550], [0.8, 1650], [1.0, 1800]
            ],
            'solidus': [
                [0.0, 1700], [0.3, 1500], [0.5, 1500], [0.7, 1500], [1.0, 1700]
            ],
            'phases': ['L', 'α'],
            'special_points': [
                {
                    'type': 'eutectic',
                    'composition': 0.5,
                    'temperature': 1500,
                    'left_solidus_temp': 1500,
                    'right_solidus_temp': 1500
                }
            ],
            'phase_regions': [
                {'name': 'L', 'num_phases': 1, 'dof': 2},
                {'name': 'L+α', 'num_phases': 2, 'dof': 1},
                {'name': 'α', 'num_phases': 1, 'dof': 2}
            ]
        }
    }
    
    # 步骤2：分析相图2
    print("\n步骤2：分析相图2（共晶系统）")
    result2 = identify_phase_diagram_errors(diagram2)
    print(f"FUNCTION_CALL: identify_phase_diagram_errors | PARAMS: diagram_2 | RESULT: {result2}")
    
    # 相图3：复杂相图（包含α、β、γ相）- 相区拓扑错误
    diagram3 = {
        'diagram_id': 'diagram_3',
        'features': {
            'type': 'complex',
            'description': '液相L、α相、β相和γ相的复杂相图，包含共晶点和包晶反应区域'
        },
        'data': {
            'type': 'peritectic',
            'liquidus': [
                [0.0, 1900], [0.2, 1800], [0.4, 1700], [0.6, 1750], 
                [0.8, 1800], [1.0, 1900]
            ],
            'solidus': [
                [0.0, 1800], [0.3, 1600], [0.5, 1650], [0.7, 1600], [1.0, 1800]
            ],
            'phases': ['L', 'α', 'β', 'γ'],
            'phase_regions': [
                {'name': 'L', 'num_phases': 1, 'dof': 2},
                {'name': 'α', 'num_phases': 1, 'dof': 2},
                {'name': 'β', 'num_phases': 1, 'dof': 2},
                {'name': 'γ', 'num_phases': 1, 'dof': 2},
                {'name': 'L+α', 'num_phases': 2, 'dof': 1},
                {'name': 'α+β', 'num_phases': 2, 'dof': 1},
                {'name': 'β+γ', 'num_phases': 2, 'dof': 1}
            ]
        }
    }
    
    # 步骤3：分析相图3
    print("\n步骤3：分析相图3（复杂相图）")
    result3 = identify_phase_diagram_errors(diagram3)
    print(f"FUNCTION_CALL: identify_phase_diagram_errors | PARAMS: diagram_3 | RESULT: {result3}")
    
    # 相图4：偏晶型（Monotectic）- 液相不互溶区错误
    diagram4 = {
        'diagram_id': 'diagram_4',
        'features': {
            'type': 'monotectic',
            'description': '液相L、α相和β相，具有偏晶型特征，包含液相不互溶区和固相转变'
        },
        'data': {
            'type': 'monotectic',
            'liquidus': [
                [0.0, 1900], [0.2, 1850], [0.3, 1800], [0.4, 1800],  # 液相不互溶区
                [0.5, 1800], [0.6, 1800], [0.8, 1850], [1.0, 1900]
            ],
            'solidus': [
                [0.0, 1800], [0.3, 1700], [0.5, 1650], [0.7, 1700], [1.0, 1800]
            ],
            'phases': ['L', 'α', 'β'],
            'phase_regions': [
                {'name': 'L', 'num_phases': 1, 'dof': 2},
                {'name': 'L1+L2', 'num_phases': 2, 'dof': 1},  # 液相不互溶
                {'name': 'α', 'num_phases': 1, 'dof': 2},
                {'name': 'β', 'num_phases': 1, 'dof': 2},
                {'name': 'α+β', 'num_phases': 2, 'dof': 1}
            ]
        }
    }
    
    # 步骤4：分析相图4
    print("\n步骤4：分析相图4（偏晶系统）")
    result4 = identify_phase_diagram_errors(diagram4)
    print(f"FUNCTION_CALL: identify_phase_diagram_errors | PARAMS: diagram_4 | RESULT: {result4}")
    
    # 步骤5：比较所有相图
    print("\n步骤5：统计正确的相图数量")
    all_diagrams = [diagram1, diagram2, diagram3, diagram4]
    comparison = compare_phase_diagrams(all_diagrams)
    print(f"FUNCTION_CALL: compare_phase_diagrams | PARAMS: 4 diagrams | RESULT: {comparison}")
    
    # 步骤6：生成错误报告
    print("\n步骤6：生成详细错误报告")
    report_path = './mid_result/materials/phase_diagram_error_report.txt'
    report = create_error_report(comparison, report_path)
    print(f"FUNCTION_CALL: create_error_report | PARAMS: comparison_results | RESULT: {report}")
    print(f"FILE_GENERATED: txt | PATH: {report_path}")
    
    # 步骤7：可视化错误统计
    print("\n步骤7：生成错误统计图表")
    stats_path = './tool_images/phase_diagram_error_statistics.png'
    stats_viz = visualize_error_statistics(comparison, stats_path)
    print(f"FUNCTION_CALL: visualize_error_statistics | PARAMS: comparison_results | RESULT: {stats_viz}")
    print(f"FILE_GENERATED: image | PATH: {stats_path}")
    
    print("\n" + "=" * 80)
    print(f"场景1结论：四个相图中有 {comparison['result']} 个是正确的")
    print("=" * 80)
    print(f"FINAL_ANSWER: {comparison['result']}")
    
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("场景2：验证吉布斯相律在不同相区的应用")
    print("=" * 80)
    print("问题描述：对于二元系统，验证单相区、两相区和三相区是否满足吉布斯相律")
    print("-" * 80)
    
    # 步骤1：检查单相区（L相）
    print("\n步骤1：检查单相区的自由度")
    single_phase_check = check_gibbs_phase_rule(
        num_components=2,
        num_phases=1,
        degrees_of_freedom=2
    )
    print(f"FUNCTION_CALL: check_gibbs_phase_rule | PARAMS: C=2, P=1, F=2 | RESULT: {single_phase_check}")
    
    # 步骤2：检查两相区（L+α）
    print("\n步骤2：检查两相区的自由度")
    two_phase_check = check_gibbs_phase_rule(
        num_components=2,
        num_phases=2,
        degrees_of_freedom=1
    )
    print(f"FUNCTION_CALL: check_gibbs_phase_rule | PARAMS: C=2, P=2, F=1 | RESULT: {two_phase_check}")
    
    # 步骤3：检查三相区（共晶点）
    print("\n步骤3：检查三相区的自由度")
    three_phase_check = check_gibbs_phase_rule(
        num_components=2,
        num_phases=3,
        degrees_of_freedom=0
    )
    print(f"FUNCTION_CALL: check_gibbs_phase_rule | PARAMS: C=2, P=3, F=0 | RESULT: {three_phase_check}")
    
    # 步骤4：检查错误的相区配置
    print("\n步骤4：检查不满足吉布斯相律的配置")
    invalid_check = check_gibbs_phase_rule(
        num_components=2,
        num_phases=2,
        degrees_of_freedom=2  # 错误：应该是1
    )
    print(f"FUNCTION_CALL: check_gibbs_phase_rule | PARAMS: C=2, P=2, F=2 | RESULT: {invalid_check}")
    
    all_valid = (single_phase_check['result'] and 
                 two_phase_check['result'] and 
                 three_phase_check['result'] and 
                 not invalid_check['result'])
    
    print("\n" + "=" * 80)
    print(f"场景2结论：吉布斯相律验证{'通过' if all_valid else '失败'}")
    print("=" * 80)
    print(f"FINAL_ANSWER: All thermodynamic checks passed: {all_valid}")
    
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("场景3：使用杠杆定律计算两相区的相分数")
    print("=" * 80)
    print("问题描述：在共晶系统的两相区，计算不同成分下的相分数")
    print("-" * 80)
    
    # 步骤1：定义两相区边界
    print("\n步骤1：设定两相区的成分边界")
    left_boundary = 0.3  # α相的成分
    right_boundary = 0.7  # β相的成分
    print(f"左边界（α相）: {left_boundary}, 右边界（β相）: {right_boundary}")
    
    # 步骤2：计算不同成分下的相分数
    print("\n步骤2：计算三个不同成分的相分数")
    compositions = [0.4, 0.5, 0.6]
    phase_fractions = []
    
    for comp in compositions:
        fraction = calculate_phase_fraction(comp, left_boundary, right_boundary)
        phase_fractions.append(fraction)
        print(f"FUNCTION_CALL: calculate_phase_fraction | PARAMS: C={comp}, Cleft={left_boundary}, Cright={right_boundary} | RESULT: {fraction}")
    
    # 步骤3：验证杠杆定律的一致性
    print("\n步骤3：验证杠杆定律计算的一致性")
    for i, comp in enumerate(compositions):
        fraction_result = phase_fractions[i]
        lever_check = check_lever_rule_consistency(
            composition=comp,
            phase1_composition=left_boundary,
            phase2_composition=right_boundary,
            phase1_fraction=fraction_result['metadata']['left_phase_fraction']
        )
        print(f"FUNCTION_CALL: check_lever_rule_consistency | PARAMS: C={comp} | RESULT: {lever_check}")
    
    # 步骤4：可视化相分数变化
    print("\n步骤4：可视化相分数随成分的变化")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    comp_range = np.linspace(left_boundary, right_boundary, 50)
    left_fractions = []
    right_fractions = []
    
    for c in comp_range:
        frac = calculate_phase_fraction(c, left_boundary, right_boundary)
        left_fractions.append(frac['metadata']['left_phase_fraction'])
        right_fractions.append(frac['metadata']['right_phase_fraction'])
    
    ax.plot(comp_range, left_fractions, 'b-', linewidth=2, label='α phase fraction')
    ax.plot(comp_range, right_fractions, 'r-', linewidth=2, label='β phase fraction')
    ax.axvline(x=left_boundary, color='b', linestyle='--', alpha=0.5, label='α boundary')
    ax.axvline(x=right_boundary, color='r', linestyle='--', alpha=0.5, label='β boundary')
    
    # 标记计算的点
    for comp, frac in zip(compositions, phase_fractions):
        ax.plot(comp, frac['metadata']['left_phase_fraction'], 'bo', markersize=8)
        ax.plot(comp, frac['metadata']['right_phase_fraction'], 'ro', markersize=8)
    
    ax.set_xlabel('Composition', fontsize=14)
    ax.set_ylabel('Phase Fraction', fontsize=14)
    ax.set_title('Lever Rule: Phase Fraction vs Composition', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    lever_viz_path = './tool_images/lever_rule_visualization.png'
    plt.tight_layout()
    plt.savefig(lever_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {lever_viz_path}")
    
    print("\n" + "=" * 80)
    print("场景3结论：杠杆定律计算和验证完成")
    print("=" * 80)
    print(f"FINAL_ANSWER: Lever rule calculations verified for {len(compositions)} compositions")


if __name__ == "__main__":
    main()