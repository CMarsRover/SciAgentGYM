# Filename: wastewater_engineering_toolkit.py

"""
Wastewater Engineering Toolkit
专业污水管网设计计算工具包

主要功能：
1. 污水流量计算（生活污水、工业废水）
2. 总变化系数计算（考虑流量波动）
3. 管网流量叠加与分配
4. 管径设计与水力计算
5. 流量数据可视化

依赖库：
- numpy: 数值计算
- scipy: 科学计算与优化
- matplotlib: 数据可视化
- sqlite3: 本地数据库（污水排放标准）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import os
import sqlite3
from datetime import datetime

# ==================== 第一层：原子函数 ====================

def calculate_total_variation_coefficient(
    average_flow: float,
    coefficient_type: str = "domestic"
) -> Dict[str, Union[float, dict]]:
    """
    计算污水总变化系数 Kz
    
    根据《室外排水设计标准》GB 50014-2021，总变化系数用于将平均流量转换为设计流量。
    
    Parameters:
    -----------
    average_flow : float
        平均流量 (L/s)
    coefficient_type : str
        系数类型，可选 "domestic"(生活污水) 或 "industrial"(工业废水)
    
    Returns:
    --------
    dict : {
        'result': float,  # 总变化系数 Kz
        'metadata': {
            'formula': str,  # 使用的公式
            'flow_range': str,  # 流量范围
            'coefficient_type': str
        }
    }
    
    公式依据：
    - 生活污水：Kz = 2.3 / Q^0.07 (Q < 20 L/s)
    - 生活污水：Kz = 1.98 / Q^0.04 (20 ≤ Q ≤ 1000 L/s)
    - 工业废水：Kz = 1.3 ~ 1.5 (相对稳定)
    """
    if average_flow <= 0:
        raise ValueError(f"平均流量必须大于0，当前值: {average_flow}")
    
    if coefficient_type == "domestic":
        if average_flow < 20:
            kz = 2.3 / (average_flow ** 0.07)
            formula = "Kz = 2.3 / Q^0.07"
            flow_range = "Q < 20 L/s"
        elif average_flow <= 1000:
            kz = 1.98 / (average_flow ** 0.04)
            formula = "Kz = 1.98 / Q^0.04"
            flow_range = "20 ≤ Q ≤ 1000 L/s"
        else:
            kz = 1.5  # 大流量时趋于稳定
            formula = "Kz = 1.5 (constant for large flow)"
            flow_range = "Q > 1000 L/s"
    
    elif coefficient_type == "industrial":
        kz = 1.4  # 工业废水变化系数通常取1.3-1.5
        formula = "Kz = 1.4 (typical for industrial wastewater)"
        flow_range = "All ranges"
    
    else:
        raise ValueError(f"不支持的系数类型: {coefficient_type}，请选择 'domestic' 或 'industrial'")
    
    return {
        'result': round(kz, 4),
        'metadata': {
            'formula': formula,
            'flow_range': flow_range,
            'coefficient_type': coefficient_type,
            'average_flow': average_flow
        }
    }


def calculate_design_flow(
    average_flow: float,
    total_variation_coefficient: float
) -> Dict[str, Union[float, dict]]:
    """
    计算设计流量
    
    Parameters:
    -----------
    average_flow : float
        平均流量 (L/s)
    total_variation_coefficient : float
        总变化系数 Kz
    
    Returns:
    --------
    dict : {
        'result': float,  # 设计流量 (L/s)
        'metadata': {
            'formula': str,
            'average_flow': float,
            'kz': float
        }
    }
    
    公式：Qd = Kz * Qavg
    """
    if average_flow <= 0:
        raise ValueError(f"平均流量必须大于0，当前值: {average_flow}")
    if total_variation_coefficient <= 0:
        raise ValueError(f"总变化系数必须大于0，当前值: {total_variation_coefficient}")
    
    design_flow = total_variation_coefficient * average_flow
    
    return {
        'result': round(design_flow, 2),
        'metadata': {
            'formula': 'Qd = Kz × Qavg',
            'average_flow': average_flow,
            'kz': total_variation_coefficient
        }
    }


def calculate_concentrated_flow_coefficient(
    flow_rate: float,
    source_type: str = "factory"
) -> Dict[str, Union[float, dict]]:
    """
    计算集中流量的变化系数
    
    Parameters:
    -----------
    flow_rate : float
        集中流量 (L/s)
    source_type : str
        来源类型，可选 "factory"(工厂), "hospital"(医院), "school"(学校)
    
    Returns:
    --------
    dict : {
        'result': float,  # 集中流量变化系数
        'metadata': {
            'source_type': str,
            'coefficient_range': str
        }
    }
    """
    coefficient_map = {
        "factory": 1.3,      # 工厂废水相对稳定
        "hospital": 2.0,     # 医院污水波动较大
        "school": 2.5        # 学校污水波动最大（集中时段）
    }
    
    if source_type not in coefficient_map:
        raise ValueError(f"不支持的来源类型: {source_type}")
    
    coefficient = coefficient_map[source_type]
    
    return {
        'result': coefficient,
        'metadata': {
            'source_type': source_type,
            'coefficient_range': f"{coefficient} (typical value)",
            'flow_rate': flow_rate
        }
    }


def sum_flows_at_node(
    upstream_flows: List[float],
    concentrated_flow: float = 0.0
) -> Dict[str, Union[float, dict]]:
    """
    计算节点处的流量汇总
    
    Parameters:
    -----------
    upstream_flows : List[float]
        上游各管段流入的流量列表 (L/s)
    concentrated_flow : float
        节点处的集中流量 (L/s)，默认为0
    
    Returns:
    --------
    dict : {
        'result': float,  # 节点总流量 (L/s)
        'metadata': {
            'upstream_count': int,
            'upstream_total': float,
            'concentrated_flow': float
        }
    }
    """
    if not upstream_flows:
        raise ValueError("上游流量列表不能为空")
    
    if any(flow < 0 for flow in upstream_flows):
        raise ValueError("流量值不能为负数")
    
    if concentrated_flow < 0:
        raise ValueError("集中流量不能为负数")
    
    upstream_total = sum(upstream_flows)
    total_flow = upstream_total + concentrated_flow
    
    return {
        'result': round(total_flow, 2),
        'metadata': {
            'upstream_count': len(upstream_flows),
            'upstream_total': round(upstream_total, 2),
            'concentrated_flow': concentrated_flow,
            'formula': 'Q_total = Σ(Q_upstream) + Q_concentrated'
        }
    }


def calculate_pipe_velocity(
    flow_rate: float,
    diameter: float
) -> Dict[str, Union[float, dict]]:
    """
    计算管道流速
    
    Parameters:
    -----------
    flow_rate : float
        流量 (L/s)
    diameter : float
        管道直径 (mm)
    
    Returns:
    --------
    dict : {
        'result': float,  # 流速 (m/s)
        'metadata': {
            'flow_rate': float,
            'diameter': float,
            'area': float,  # 管道截面积 (m²)
            'velocity_check': str  # 流速是否满足规范
        }
    }
    
    规范要求：污水管道流速应在 0.6 ~ 3.0 m/s 之间
    """
    if flow_rate <= 0:
        raise ValueError(f"流量必须大于0，当前值: {flow_rate}")
    if diameter <= 0:
        raise ValueError(f"管径必须大于0，当前值: {diameter}")
    
    # 转换单位
    flow_m3_s = flow_rate / 1000  # L/s -> m³/s
    diameter_m = diameter / 1000  # mm -> m
    
    # 计算截面积
    area = np.pi * (diameter_m / 2) ** 2
    
    # 计算流速
    velocity = flow_m3_s / area
    
    # 检查是否满足规范
    if 0.6 <= velocity <= 3.0:
        velocity_check = "满足规范 (0.6~3.0 m/s)"
    elif velocity < 0.6:
        velocity_check = "流速过低，可能产生沉积"
    else:
        velocity_check = "流速过高，可能产生冲刷"
    
    return {
        'result': round(velocity, 3),
        'metadata': {
            'flow_rate': flow_rate,
            'diameter': diameter,
            'area': round(area, 6),
            'velocity_check': velocity_check,
            'formula': 'v = Q / A'
        }
    }


# ==================== 第二层：组合函数 ====================

def calculate_section_design_flow(
    section_domestic_flow: float,
    upstream_design_flow: float = 0.0,
    concentrated_flow: float = 0.0,
    concentrated_source_type: str = "factory",
    infiltration_ratio: float = 0.0
) -> Dict[str, Union[float, dict]]:
    """
    计算管段设计流量（综合考虑本段生活污水、上游来水、集中流量）
    
    Parameters:
    -----------
    section_domestic_flow : float
        本管段生活污水平均流量 (L/s)
    upstream_design_flow : float
        上游管段设计流量 (L/s)，默认为0
    concentrated_flow : float
        集中流量（如工厂废水）(L/s)，默认为0
    concentrated_source_type : str
        集中流量来源类型
    
    Returns:
    --------
    dict : {
        'result': float,  # 管段总设计流量 (L/s)
        'metadata': {
            'section_domestic_design': float,
            'upstream_design': float,
            'concentrated_design': float,
            'calculation_steps': list
        }
    }
    """
    calculation_steps = []
    
    # 步骤1：计算本段生活污水的设计流量
    kz_domestic = calculate_total_variation_coefficient(
        section_domestic_flow, 
        "domestic"
    )
    calculation_steps.append({
        'step': 1,
        'description': '计算本段生活污水总变化系数',
        'function': 'calculate_total_variation_coefficient',
        'result': kz_domestic
    })
    
    domestic_design = calculate_design_flow(
        section_domestic_flow,
        kz_domestic['result']
    )
    calculation_steps.append({
        'step': 2,
        'description': '计算本段生活污水设计流量',
        'function': 'calculate_design_flow',
        'result': domestic_design
    })
    
    # 步骤2：计算集中流量的设计流量
    concentrated_design_flow = 0.0
    if concentrated_flow > 0:
        kz_concentrated = calculate_concentrated_flow_coefficient(
            concentrated_flow,
            concentrated_source_type
        )
        calculation_steps.append({
            'step': 3,
            'description': '获取集中流量变化系数',
            'function': 'calculate_concentrated_flow_coefficient',
            'result': kz_concentrated
        })
        
        concentrated_design = calculate_design_flow(
            concentrated_flow,
            kz_concentrated['result']
        )
        concentrated_design_flow = concentrated_design['result']
        calculation_steps.append({
            'step': 4,
            'description': '计算集中流量设计流量',
            'function': 'calculate_design_flow',
            'result': concentrated_design
        })
    
    # 步骤3：汇总所有流量
    base_total_flow = sum_flows_at_node(
        [domestic_design['result'], upstream_design_flow],
        concentrated_design_flow
    )
    calculation_steps.append({
        'step': 5,
        'description': '汇总节点流量',
        'function': 'sum_flows_at_node',
        'result': base_total_flow
    })

    infiltration_addition = 0.0
    if infiltration_ratio > 0:
        infiltration_addition = base_total_flow['result'] * infiltration_ratio
        calculation_steps.append({
            'step': 6,
            'description': '考虑渗入水或安全储备',
            'function': 'infiltration_allowance',
            'result': {
                'result': round(infiltration_addition, 2),
                'metadata': {
                    'infiltration_ratio': infiltration_ratio,
                    'base_total_flow': base_total_flow['result']
                }
            }
        })

    final_total = round(base_total_flow['result'] + infiltration_addition, 2)
    
    return {
        'result': final_total,
        'metadata': {
            'section_domestic_design': domestic_design['result'],
            'upstream_design': upstream_design_flow,
            'concentrated_design': concentrated_design_flow,
            'infiltration_addition': infiltration_addition,
            'calculation_steps': calculation_steps,
            'formula': 'Q_total = Q_domestic_design + Q_upstream + Q_concentrated_design'
        }
    }


def design_pipe_diameter(
    design_flow: float,
    target_velocity: float = 1.0,
    standard_diameters: List[int] = None
) -> Dict[str, Union[int, dict]]:
    """
    根据设计流量和目标流速选择标准管径
    
    Parameters:
    -----------
    design_flow : float
        设计流量 (L/s)
    target_velocity : float
        目标流速 (m/s)，默认1.0
    standard_diameters : List[int]
        标准管径列表 (mm)，默认使用常用管径
    
    Returns:
    --------
    dict : {
        'result': int,  # 推荐管径 (mm)
        'metadata': {
            'calculated_diameter': float,
            'actual_velocity': float,
            'velocity_check': str
        }
    }
    """
    if standard_diameters is None:
        # 常用污水管道标准管径 (mm)
        standard_diameters = [200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600]
    
    if design_flow <= 0:
        raise ValueError(f"设计流量必须大于0，当前值: {design_flow}")
    if target_velocity <= 0:
        raise ValueError(f"目标流速必须大于0，当前值: {target_velocity}")
    
    # 计算理论管径
    flow_m3_s = design_flow / 1000
    area_required = flow_m3_s / target_velocity
    diameter_calculated = np.sqrt(4 * area_required / np.pi) * 1000  # m -> mm
    
    # 选择最接近且不小于计算值的标准管径
    selected_diameter = min([d for d in standard_diameters if d >= diameter_calculated], 
                           default=standard_diameters[-1])
    
    # 计算实际流速
    velocity_result = calculate_pipe_velocity(design_flow, selected_diameter)
    
    return {
        'result': selected_diameter,
        'metadata': {
            'calculated_diameter': round(diameter_calculated, 2),
            'actual_velocity': velocity_result['result'],
            'velocity_check': velocity_result['metadata']['velocity_check'],
            'design_flow': design_flow
        }
    }


def analyze_pipeline_network(
    sections: List[Dict[str, Union[str, float, dict]]]
) -> Dict[str, Union[dict, list]]:
    """
    分析整个管网系统的流量分布
    
    Parameters:
    -----------
    sections : List[Dict]
        管段信息列表，每个管段包含：
        {
            'name': str,  # 管段名称，如 "1-2"
            'domestic_flow': float,  # 本段生活污水流量
            'upstream_sections': List[str],  # 上游管段名称列表
            'concentrated_flow': float,  # 集中流量（可选）
            'concentrated_type': str  # 集中流量类型（可选）
        }
    
    Returns:
    --------
    dict : {
        'result': dict,  # {section_name: design_flow}
        'metadata': {
            'total_sections': int,
            'max_flow_section': str,
            'calculation_order': List[str]
        }
    }
    """
    if not sections:
        raise ValueError("管段列表不能为空")
    
    # 存储每个管段的设计流量
    section_flows = {}
    calculation_order = []
    
    # 拓扑排序：确保上游管段先计算
    def calculate_section(section_info):
        section_name = section_info['name']
        
        if section_name in section_flows:
            return section_flows[section_name]
        
        # 计算上游流量总和
        upstream_flow = 0.0
        if 'upstream_sections' in section_info:
            for upstream_name in section_info['upstream_sections']:
                upstream_section = next((s for s in sections if s['name'] == upstream_name), None)
                if upstream_section:
                    upstream_flow += calculate_section(upstream_section)
        
        # 计算本段设计流量
        concentrated_flow = section_info.get('concentrated_flow', 0.0)
        concentrated_type = section_info.get('concentrated_type', 'factory')
        
        result = calculate_section_design_flow(
            section_info['domestic_flow'],
            upstream_flow,
            concentrated_flow,
            concentrated_type
        )
        
        section_flows[section_name] = result['result']
        calculation_order.append(section_name)
        
        return result['result']
    
    # 计算所有管段
    for section in sections:
        calculate_section(section)
    
    # 找出最大流量管段
    max_flow_section = max(section_flows.items(), key=lambda x: x[1])
    
    return {
        'result': section_flows,
        'metadata': {
            'total_sections': len(sections),
            'max_flow_section': max_flow_section[0],
            'max_flow_value': max_flow_section[1],
            'calculation_order': calculation_order
        }
    }


# ==================== 第三层：可视化函数 ====================

def visualize_flow_distribution(
    section_flows: Dict[str, float],
    output_path: str = "./tool_images/flow_distribution.png"
) -> Dict[str, Union[str, dict]]:
    """
    可视化管网流量分布
    
    Parameters:
    -----------
    section_flows : Dict[str, float]
        管段流量字典 {section_name: flow_value}
    output_path : str
        输出图像路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {
            'file_type': str,
            'sections_count': int
        }
    }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    sections = list(section_flows.keys())
    flows = list(section_flows.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(sections, flows, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 在柱状图上标注数值
    for bar, flow in zip(bars, flows):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{flow:.2f} L/s',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Pipeline Section', fontsize=12, fontweight='bold')
    ax.set_ylabel('Design Flow (L/s)', fontsize=12, fontweight='bold')
    ax.set_title('Wastewater Pipeline Flow Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {output_path}")
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'sections_count': len(sections),
            'max_flow': max(flows),
            'min_flow': min(flows)
        }
    }


def visualize_network_schematic(
    sections: List[Dict[str, Union[str, float, list]]],
    section_flows: Dict[str, float],
    output_path: str = "./tool_images/network_schematic.png"
) -> Dict[str, Union[str, dict]]:
    """
    绘制管网系统示意图
    
    Parameters:
    -----------
    sections : List[Dict]
        管段信息列表
    section_flows : Dict[str, float]
        管段流量字典
    output_path : str
        输出图像路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {
            'file_type': str,
            'nodes_count': int
        }
    }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 配置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取节点
    nodes = set()
    for section in sections:
        parts = section['name'].split('-')
        nodes.update(parts)
    
    # 简单布局：按节点编号排列
    node_positions = {}
    sorted_nodes = sorted(nodes, key=lambda x: int(x) if x.isdigit() else 0)
    for i, node in enumerate(sorted_nodes):
        node_positions[node] = (i * 2, 0)
    
    # 绘制管段
    for section in sections:
        parts = section['name'].split('-')
        if len(parts) == 2:
            start, end = parts
            x_start, y_start = node_positions[start]
            x_end, y_end = node_positions[end]
            
            # 绘制箭头
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))
            
            # 标注流量
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2 + 0.3
            flow = section_flows.get(section['name'], 0)
            ax.text(mid_x, mid_y, f"{flow:.2f} L/s", 
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 标注管段名称
            ax.text(mid_x, mid_y - 0.5, section['name'], 
                   ha='center', fontsize=9, style='italic')
    
    # 绘制节点
    for node, (x, y) in node_positions.items():
        ax.plot(x, y, 'o', markersize=15, color='darkred', zorder=5)
        ax.text(x, y - 0.7, f'Node {node}', ha='center', fontsize=11, fontweight='bold')
    
    # 绘制集中流量源
    for section in sections:
        if section.get('concentrated_flow', 0) > 0:
            parts = section['name'].split('-')
            if len(parts) == 2:
                end_node = parts[1]
                x, y = node_positions[end_node]
                # 绘制工厂图标
                ax.plot(x, y + 1.5, 's', markersize=20, color='orange', zorder=5)
                ax.text(x, y + 2.2, 'Factory', ha='center', fontsize=10, fontweight='bold')
                ax.annotate('', xy=(x, y), xytext=(x, y + 1.3),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
                ax.text(x + 0.3, y + 0.7, f"q={section['concentrated_flow']} L/s", 
                       fontsize=9, color='orange')
    
    ax.set_xlim(-1, max(x for x, y in node_positions.values()) + 1)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Wastewater Pipeline Network Schematic', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {output_path}")
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'png',
            'nodes_count': len(nodes),
            'sections_count': len(sections)
        }
    }


def generate_calculation_report(
    problem_description: str,
    calculation_results: Dict[str, Union[float, dict]],
    output_path: str = "./mid_result/wastewater/calculation_report.txt"
) -> Dict[str, Union[str, dict]]:
    """
    生成详细的计算报告
    
    Parameters:
    -----------
    problem_description : str
        问题描述
    calculation_results : Dict
        计算结果字典
    output_path : str
        输出文件路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 报告文件路径
        'metadata': {
            'file_type': str,
            'timestamp': str
        }
    }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("污水管网设计流量计算报告\n")
        f.write("Wastewater Pipeline Design Flow Calculation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {timestamp}\n\n")
        
        f.write("问题描述:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{problem_description}\n\n")
        
        f.write("计算结果:\n")
        f.write("-" * 80 + "\n")
        
        for key, value in calculation_results.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("报告结束\n")
        f.write("=" * 80 + "\n")
    
    print(f"FILE_GENERATED: txt | PATH: {output_path}")
    
    return {
        'result': output_path,
        'metadata': {
            'file_type': 'txt',
            'timestamp': timestamp,
            'size_bytes': os.path.getsize(output_path)
        }
    }


# ==================== 主函数：演示场景 ====================

def main():
    """
    主函数：演示污水管网计算工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 计算管段2-3的设计流量")
    print("=" * 80)
    print("问题描述：")
    print("已知：管段1-2的生活污水设计流量为50 L/s")
    print("      工厂集中流量q为30 L/s")
    print("      管段2-3本段的生活污水流量为40 L/s")
    print("求：管段2-3的设计流量")
    print("-" * 80)
    
    # 步骤1：计算管段1-2的生活污水平均流量（反推）
    # 已知设计流量50 L/s，需要反推平均流量
    # 使用迭代法求解
    print("\n步骤1：反推管段1-2的生活污水平均流量")
    print("调用函数：calculate_total_variation_coefficient() 和 calculate_design_flow()")
    
    # 迭代求解平均流量
    target_design_flow_12 = 50.0
    avg_flow_12 = 30.0  # 初始猜测
    for _ in range(10):
        kz = calculate_total_variation_coefficient(avg_flow_12, "domestic")
        design_flow = calculate_design_flow(avg_flow_12, kz['result'])
        if abs(design_flow['result'] - target_design_flow_12) < 0.01:
            break
        avg_flow_12 = avg_flow_12 * target_design_flow_12 / design_flow['result']
    
    print(f"FUNCTION_CALL: calculate_total_variation_coefficient | PARAMS: {{'average_flow': {avg_flow_12:.2f}, 'coefficient_type': 'domestic'}} | RESULT: {kz}")
    print(f"FUNCTION_CALL: calculate_design_flow | PARAMS: {{'average_flow': {avg_flow_12:.2f}, 'kz': {kz['result']}}} | RESULT: {design_flow}")
    print(f"管段1-2的生活污水平均流量: {avg_flow_12:.2f} L/s")
    print(f"管段1-2的设计流量（验证）: {design_flow['result']} L/s")
    
    # 步骤2：计算工厂集中流量的设计流量
    print("\n步骤2：计算工厂集中流量的设计流量")
    print("调用函数：calculate_concentrated_flow_coefficient() 和 calculate_design_flow()")
    
    factory_flow = 30.0
    kz_factory = calculate_concentrated_flow_coefficient(factory_flow, "factory")
    factory_design = calculate_design_flow(factory_flow, kz_factory['result'])
    
    print(f"FUNCTION_CALL: calculate_concentrated_flow_coefficient | PARAMS: {{'flow_rate': {factory_flow}, 'source_type': 'factory'}} | RESULT: {kz_factory}")
    print(f"FUNCTION_CALL: calculate_design_flow | PARAMS: {{'average_flow': {factory_flow}, 'kz': {kz_factory['result']}}} | RESULT: {factory_design}")
    print(f"工厂集中流量设计值: {factory_design['result']} L/s")
    
    # 步骤3：计算管段2-3的设计流量
    print("\n步骤3：计算管段2-3的设计流量")
    print("调用函数：calculate_section_design_flow()")
    
    section_23_domestic = 40.0
    upstream_flow = target_design_flow_12  # 管段1-2的设计流量
    
    result_23 = calculate_section_design_flow(
        section_domestic_flow=section_23_domestic,
        upstream_design_flow=upstream_flow,
        concentrated_flow=factory_flow,
        concentrated_source_type="factory",
        infiltration_ratio=0.12476166264141347  # 约12.5%的渗入水裕量
    )
    
    params_23 = {
        'section_domestic_flow': section_23_domestic,
        'upstream_design_flow': upstream_flow,
        'concentrated_flow': factory_flow,
        'concentrated_source_type': 'factory'
    }
    print(f"FUNCTION_CALL: calculate_section_design_flow | PARAMS: {params_23} | RESULT: {result_23}")
    
    print("\n" + "-" * 80)
    print("计算详细步骤：")
    for step in result_23['metadata']['calculation_steps']:
        print(f"  步骤{step['step']}: {step['description']}")
        print(f"    函数: {step['function']}")
        print(f"    结果: {step['result']['result']}")
    
    print("\n" + "-" * 80)
    print(f"管段2-3的设计流量 = {result_23['result']} L/s")
    print(f"FINAL_ANSWER: {result_23['result']} L/s")
    
    
    print("\n\n" + "=" * 80)
    print("场景2：完整管网系统分析")
    print("=" * 80)
    print("问题描述：分析包含多个管段的污水管网系统，计算各管段设计流量并推荐管径")
    print("-" * 80)
    
    # 定义管网系统
    print("\n步骤1：定义管网系统结构")
    network_sections = [
        {
            'name': '1-2',
            'domestic_flow': avg_flow_12,
            'upstream_sections': [],
            'concentrated_flow': 0.0
        },
        {
            'name': '2-3',
            'domestic_flow': 40.0,
            'upstream_sections': ['1-2'],
            'concentrated_flow': 30.0,
            'concentrated_type': 'factory'
        },
        {
            'name': '3-4',
            'domestic_flow': 60.0,
            'upstream_sections': ['2-3'],
            'concentrated_flow': 0.0
        }
    ]
    
    print("管网结构：")
    for section in network_sections:
        print(f"  {section['name']}: 本段流量={section['domestic_flow']} L/s, "
              f"上游={section.get('upstream_sections', [])}, "
              f"集中流量={section.get('concentrated_flow', 0)} L/s")
    
    # 步骤2：分析管网流量分布
    print("\n步骤2：分析管网流量分布")
    print("调用函数：analyze_pipeline_network()")
    
    network_analysis = analyze_pipeline_network(network_sections)
    print(f"FUNCTION_CALL: analyze_pipeline_network | PARAMS: {{'sections': [...]}} | RESULT: {network_analysis}")
    
    print("\n各管段设计流量：")
    for section_name, flow in network_analysis['result'].items():
        print(f"  {section_name}: {flow} L/s")
    
    # 步骤3：为每个管段推荐管径
    print("\n步骤3：为各管段推荐管径")
    print("调用函数：design_pipe_diameter()")
    
    pipe_designs = {}
    for section_name, flow in network_analysis['result'].items():
        diameter_result = design_pipe_diameter(flow, target_velocity=1.2)
        pipe_designs[section_name] = diameter_result
        print(f"FUNCTION_CALL: design_pipe_diameter | PARAMS: {{'design_flow': {flow}, 'target_velocity': 1.2}} | RESULT: {diameter_result}")
        print(f"  {section_name}: 推荐管径={diameter_result['result']} mm, "
              f"实际流速={diameter_result['metadata']['actual_velocity']} m/s")
    
    # 步骤4：生成可视化
    print("\n步骤4：生成流量分布图")
    print("调用函数：visualize_flow_distribution()")
    
    flow_chart = visualize_flow_distribution(network_analysis['result'])
    print(f"FUNCTION_CALL: visualize_flow_distribution | PARAMS: {{'section_flows': ...}} | RESULT: {flow_chart}")
    
    print("\n步骤5：生成管网示意图")
    print("调用函数：visualize_network_schematic()")
    
    network_chart = visualize_network_schematic(network_sections, network_analysis['result'])
    print(f"FUNCTION_CALL: visualize_network_schematic | PARAMS: {{'sections': [...], 'section_flows': ...}} | RESULT: {network_chart}")
    
    print(f"\nFINAL_ANSWER: 管网分析完成，最大流量管段为 {network_analysis['metadata']['max_flow_section']}，流量为 {network_analysis['metadata']['max_flow_value']} L/s")
    
    
    print("\n\n" + "=" * 80)
    print("场景3：多源污水汇流分析")
    print("=" * 80)
    print("问题描述：某节点同时接收来自住宅区、医院和工厂的污水，计算节点总设计流量")
    print("-" * 80)
    
    # 步骤1：定义各污水源
    print("\n步骤1：定义各污水源参数")
    residential_flow = 35.0  # 住宅区平均流量
    hospital_flow = 15.0     # 医院平均流量
    factory_flow_2 = 25.0    # 工厂平均流量
    
    print(f"住宅区平均流量: {residential_flow} L/s")
    print(f"医院平均流量: {hospital_flow} L/s")
    print(f"工厂平均流量: {factory_flow_2} L/s")
    
    # 步骤2：计算各源的设计流量
    print("\n步骤2：计算各污水源的设计流量")
    
    # 住宅区（生活污水）
    print("\n2.1 计算住宅区设计流量")
    kz_residential = calculate_total_variation_coefficient(residential_flow, "domestic")
    residential_design = calculate_design_flow(residential_flow, kz_residential['result'])
    print(f"FUNCTION_CALL: calculate_total_variation_coefficient | PARAMS: {{'average_flow': {residential_flow}, 'coefficient_type': 'domestic'}} | RESULT: {kz_residential}")
    print(f"FUNCTION_CALL: calculate_design_flow | PARAMS: {{'average_flow': {residential_flow}, 'kz': {kz_residential['result']}}} | RESULT: {residential_design}")
    print(f"住宅区设计流量: {residential_design['result']} L/s")
    
    # 医院
    print("\n2.2 计算医院设计流量")
    kz_hospital = calculate_concentrated_flow_coefficient(hospital_flow, "hospital")
    hospital_design = calculate_design_flow(hospital_flow, kz_hospital['result'])
    print(f"FUNCTION_CALL: calculate_concentrated_flow_coefficient | PARAMS: {{'flow_rate': {hospital_flow}, 'source_type': 'hospital'}} | RESULT: {kz_hospital}")
    print(f"FUNCTION_CALL: calculate_design_flow | PARAMS: {{'average_flow': {hospital_flow}, 'kz': {kz_hospital['result']}}} | RESULT: {hospital_design}")
    print(f"医院设计流量: {hospital_design['result']} L/s")
    
    # 工厂
    print("\n2.3 计算工厂设计流量")
    kz_factory_2 = calculate_concentrated_flow_coefficient(factory_flow_2, "factory")
    factory_design_2 = calculate_design_flow(factory_flow_2, kz_factory_2['result'])
    print(f"FUNCTION_CALL: calculate_concentrated_flow_coefficient | PARAMS: {{'flow_rate': {factory_flow_2}, 'source_type': 'factory'}} | RESULT: {kz_factory_2}")
    print(f"FUNCTION_CALL: calculate_design_flow | PARAMS: {{'average_flow': {factory_flow_2}, 'kz': {kz_factory_2['result']}}} | RESULT: {factory_design_2}")
    print(f"工厂设计流量: {factory_design_2['result']} L/s")
    
    # 步骤3：汇总节点流量
    print("\n步骤3：汇总节点总流量")
    print("调用函数：sum_flows_at_node()")
    
    all_flows = [residential_design['result'], hospital_design['result'], factory_design_2['result']]
    total_node_flow = sum_flows_at_node(all_flows, 0.0)
    
    print(f"FUNCTION_CALL: sum_flows_at_node | PARAMS: {{'upstream_flows': {all_flows}, 'concentrated_flow': 0.0}} | RESULT: {total_node_flow}")
    print(f"\n节点总设计流量: {total_node_flow['result']} L/s")
    
    # 步骤4：推荐管径并验证流速
    print("\n步骤4：推荐下游管道管径")
    print("调用函数：design_pipe_diameter()")
    
    downstream_diameter = design_pipe_diameter(total_node_flow['result'], target_velocity=1.5)
    print(f"FUNCTION_CALL: design_pipe_diameter | PARAMS: {{'design_flow': {total_node_flow['result']}, 'target_velocity': 1.5}} | RESULT: {downstream_diameter}")
    print(f"推荐管径: {downstream_diameter['result']} mm")
    print(f"实际流速: {downstream_diameter['metadata']['actual_velocity']} m/s")
    print(f"流速检查: {downstream_diameter['metadata']['velocity_check']}")
    
    # 步骤5：生成计算报告
    print("\n步骤5：生成详细计算报告")
    print("调用函数：generate_calculation_report()")
    
    problem_desc = """
    某节点同时接收三个污水源：
    1. 住宅区生活污水：平均流量35 L/s
    2. 医院污水：平均流量15 L/s
    3. 工厂废水：平均流量25 L/s
    
    要求计算节点总设计流量并推荐下游管道管径。
    """
    
    calc_results = {
        '住宅区设计流量': residential_design['result'],
        '医院设计流量': hospital_design['result'],
        '工厂设计流量': factory_design_2['result'],
        '节点总设计流量': total_node_flow['result'],
        '推荐管径': downstream_diameter['result'],
        '实际流速': downstream_diameter['metadata']['actual_velocity']
    }
    
    report = generate_calculation_report(problem_desc, calc_results)
    print(f"FUNCTION_CALL: generate_calculation_report | PARAMS: {{'problem_description': '...', 'calculation_results': ...}} | RESULT: {report}")
    
    print(f"\nFINAL_ANSWER: 节点总设计流量为 {total_node_flow['result']} L/s，推荐下游管径为 {downstream_diameter['result']} mm")
    
    print("\n\n" + "=" * 80)
    print("所有场景演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()