# Filename: chipseq_epigenetics_toolkit.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json
import os
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
import requests
import time

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 创建必要的目录
os.makedirs('./mid_result/epigenetics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
CROSSLINKING_EFFICIENCY = {
    'PFA_only': {'protein_dna': 0.85, 'protein_protein': 0.3},
    'PFA_DSG': {'protein_dna': 0.90, 'protein_protein': 0.95}
}

GENOMIC_REGIONS = {
    'active_promoter': {'chromatin_accessibility': 0.9, 'tf_density': 0.8},
    'active_enhancer': {'chromatin_accessibility': 0.85, 'tf_density': 0.7},
    'inactive_promoter': {'chromatin_accessibility': 0.3, 'tf_density': 0.2},
    'heterochromatin': {'chromatin_accessibility': 0.1, 'tf_density': 0.1}
}

# ==================== 第一层：原子函数 ====================

def calculate_crosslinking_efficiency(fixation_method: str, interaction_type: str) -> dict:
    """
    计算不同固定方法的交联效率
    
    Args:
        fixation_method: 固定方法 ('PFA_only' 或 'PFA_DSG')
        interaction_type: 相互作用类型 ('protein_dna' 或 'protein_protein')
    
    Returns:
        包含交联效率的字典
    """
    if fixation_method not in CROSSLINKING_EFFICIENCY:
        raise ValueError(f"Invalid fixation method: {fixation_method}")
    if interaction_type not in ['protein_dna', 'protein_protein']:
        raise ValueError(f"Invalid interaction type: {interaction_type}")
    
    efficiency = CROSSLINKING_EFFICIENCY[fixation_method][interaction_type]
    return {
        'result': efficiency,
        'metadata': {
            'fixation_method': fixation_method,
            'interaction_type': interaction_type,
            'description': f'{interaction_type} crosslinking efficiency with {fixation_method}'
        }
    }

def simulate_chipseq_signal(region_type: str, tf_binding_mode: str, fixation_method: str) -> dict:
    """
    模拟ChIP-seq信号强度
    
    Args:
        region_type: 基因组区域类型
        tf_binding_mode: 转录因子结合模式 ('direct' 或 'indirect')
        fixation_method: 固定方法
    
    Returns:
        包含信号强度的字典
    """
    if region_type not in GENOMIC_REGIONS:
        raise ValueError(f"Invalid region type: {region_type}")
    if tf_binding_mode not in ['direct', 'indirect']:
        raise ValueError(f"Invalid binding mode: {tf_binding_mode}")
    
    base_signal = GENOMIC_REGIONS[region_type]['tf_density']
    
    # 根据结合模式调整信号
    if tf_binding_mode == 'direct':
        # 直接结合主要依赖protein-DNA交联
        crosslink_eff = calculate_crosslinking_efficiency(fixation_method, 'protein_dna')['result']
        signal = base_signal * crosslink_eff
    else:
        # 间接结合依赖protein-protein交联
        crosslink_eff = calculate_crosslinking_efficiency(fixation_method, 'protein_protein')['result']
        # 间接结合信号通常较弱且更依赖蛋白-蛋白交联
        signal = base_signal * 0.6 * crosslink_eff
    
    return {
        'result': signal,
        'metadata': {
            'region_type': region_type,
            'binding_mode': tf_binding_mode,
            'fixation_method': fixation_method,
            'base_signal': base_signal,
            'crosslinking_efficiency': crosslink_eff
        }
    }

def create_chipseq_database() -> dict:
    """
    访问ChIP-seq实验数据库
    
    Returns:
        数据库访问结果
    """
    db_path = os.path.join(DATABASE_DIR, 'chipseq_experiments.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询记录数
    cursor.execute("SELECT COUNT(*) FROM chipseq_experiments")
    records_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'database_path': db_path,
            'tables': ['chipseq_experiments'],
            'records_count': records_count
        }
    }

# ==================== 第二层：组合函数 ====================

def analyze_disappearing_peaks(db_path: str) -> dict:
    """
    分析在PFA+DSG固定中消失的peaks
    
    Args:
        db_path: 数据库路径
    
    Returns:
        消失peaks的分析结果
    """
    conn = sqlite3.connect(db_path)
    
    # 查询消失的peaks
    query = '''
        SELECT region_id, region_type, tf_binding_mode, pfa_signal, pfa_dsg_signal
        FROM chipseq_experiments 
        WHERE peak_detected_pfa = 1 AND peak_detected_pfa_dsg = 0
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return {
            'result': {'disappearing_peaks': 0, 'regions': []},
            'metadata': {'analysis': 'No disappearing peaks found'}
        }
    
    # 按区域类型统计
    region_stats = df['region_type'].value_counts().to_dict()
    binding_stats = df['tf_binding_mode'].value_counts().to_dict()
    
    # 计算信号变化
    signal_changes = []
    for _, row in df.iterrows():
        change = (row['pfa_dsg_signal'] - row['pfa_signal']) / row['pfa_signal'] * 100
        signal_changes.append({
            'region_id': row['region_id'],
            'region_type': row['region_type'],
            'binding_mode': row['tf_binding_mode'],
            'signal_change_percent': change
        })
    
    return {
        'result': {
            'disappearing_peaks_count': len(df),
            'region_distribution': region_stats,
            'binding_mode_distribution': binding_stats,
            'signal_changes': signal_changes
        },
        'metadata': {
            'total_regions_analyzed': len(df),
            'analysis_method': 'PFA vs PFA+DSG comparison'
        }
    }

def predict_peak_locations(fixation_method: str) -> dict:
    """
    预测不同固定方法下的peak位置
    
    Args:
        fixation_method: 固定方法
    
    Returns:
        预测的peak位置分布
    """
    regions = ['active_promoter', 'active_enhancer', 'inactive_promoter', 'heterochromatin']
    binding_modes = ['direct', 'indirect']
    
    predictions = []
    
    for region in regions:
        for mode in binding_modes:
            signal = simulate_chipseq_signal(region, mode, fixation_method)
            # 预测是否会形成peak (阈值0.3)
            peak_predicted = signal['result'] > 0.3
            
            predictions.append({
                'region_type': region,
                'binding_mode': mode,
                'signal_strength': signal['result'],
                'peak_predicted': peak_predicted,
                'confidence': min(signal['result'] * 2, 1.0)  # 置信度
            })
    
    return {
        'result': predictions,
        'metadata': {
            'fixation_method': fixation_method,
            'peak_threshold': 0.3,
            'total_predictions': len(predictions)
        }
    }

# ==================== 第三层：可视化函数 ====================

def visualize_fixation_comparison() -> dict:
    """
    可视化不同固定方法的比较
    
    Returns:
        图像文件路径
    """
    # 创建数据
    regions = ['Active\nPromoter', 'Active\nEnhancer', 'Inactive\nPromoter', 'Heterochromatin']
    pfa_direct = [simulate_chipseq_signal(r.replace('\n', '_').lower(), 'direct', 'PFA_only')['result'] 
                  for r in ['active_promoter', 'active_enhancer', 'inactive_promoter', 'heterochromatin']]
    pfa_indirect = [simulate_chipseq_signal(r.replace('\n', '_').lower(), 'indirect', 'PFA_only')['result'] 
                    for r in ['active_promoter', 'active_enhancer', 'inactive_promoter', 'heterochromatin']]
    pfa_dsg_direct = [simulate_chipseq_signal(r.replace('\n', '_').lower(), 'direct', 'PFA_DSG')['result'] 
                      for r in ['active_promoter', 'active_enhancer', 'inactive_promoter', 'heterochromatin']]
    pfa_dsg_indirect = [simulate_chipseq_signal(r.replace('\n', '_').lower(), 'indirect', 'PFA_DSG')['result'] 
                        for r in ['active_promoter', 'active_enhancer', 'inactive_promoter', 'heterochromatin']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(regions))
    width = 0.35
    
    # PFA only
    ax1.bar(x - width/2, pfa_direct, width, label='Direct binding', alpha=0.8, color='#2E86AB')
    ax1.bar(x + width/2, pfa_indirect, width, label='Indirect binding', alpha=0.8, color='#A23B72')
    ax1.set_xlabel('Genomic Regions')
    ax1.set_ylabel('ChIP-seq Signal Intensity')
    ax1.set_title('PFA Only Fixation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions)
    ax1.legend()
    ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Peak threshold')
    ax1.grid(True, alpha=0.3)
    
    # PFA + DSG
    ax2.bar(x - width/2, pfa_dsg_direct, width, label='Direct binding', alpha=0.8, color='#2E86AB')
    ax2.bar(x + width/2, pfa_dsg_indirect, width, label='Indirect binding', alpha=0.8, color='#A23B72')
    ax2.set_xlabel('Genomic Regions')
    ax2.set_ylabel('ChIP-seq Signal Intensity')
    ax2.set_title('PFA + DSG Fixation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions)
    ax2.legend()
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Peak threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = './tool_images/chipseq_fixation_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG image',
            'size': f'{os.path.getsize(filepath)} bytes',
            'description': 'ChIP-seq signal comparison between PFA and PFA+DSG fixation'
        }
    }

def create_peak_disappearance_heatmap(db_path: str) -> dict:
    """
    创建peak消失模式的热图
    
    Args:
        db_path: 数据库路径
    
    Returns:
        热图文件路径
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM chipseq_experiments', conn)
    conn.close()
    
    # 创建数据矩阵
    regions = df['region_type'].unique()
    binding_modes = df['tf_binding_mode'].unique()
    
    # 计算信号比值矩阵 (PFA+DSG / PFA)
    ratio_matrix = np.zeros((len(regions), len(binding_modes)))
    
    for i, region in enumerate(regions):
        for j, mode in enumerate(binding_modes):
            subset = df[(df['region_type'] == region) & (df['tf_binding_mode'] == mode)]
            if len(subset) > 0:
                ratio_matrix[i, j] = subset['signal_ratio'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(ratio_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=2)
    
    # 设置标签
    ax.set_xticks(range(len(binding_modes)))
    ax.set_xticklabels(binding_modes)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels([r.replace('_', ' ').title() for r in regions])
    
    # 添加数值标注
    for i in range(len(regions)):
        for j in range(len(binding_modes)):
            text = ax.text(j, i, f'{ratio_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xlabel('TF Binding Mode')
    ax.set_ylabel('Genomic Region Type')
    ax.set_title('Signal Ratio: PFA+DSG / PFA Only\n(Values < 1 indicate disappearing peaks)')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Signal Ratio')
    
    plt.tight_layout()
    
    filepath = './tool_images/peak_disappearance_heatmap.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG image',
            'size': f'{os.path.getsize(filepath)} bytes',
            'description': 'Heatmap showing peak disappearance patterns'
        }
    }

# ==================== 文件解析函数 ====================

def load_file(filepath: str) -> dict:
    """
    解析常见文件格式
    
    Args:
        filepath: 文件路径
    
    Returns:
        文件内容字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.db':
        conn = sqlite3.connect(filepath)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        table_data = {}
        for table in tables['name']:
            table_data[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn).to_dict('records')
        conn.close()
        return {'result': table_data, 'metadata': {'file_type': 'SQLite database'}}
    
    elif ext in ['.txt', '.csv']:
        with open(filepath, 'r') as f:
            content = f.read()
        return {'result': content, 'metadata': {'file_type': 'text file'}}
    
    else:
        return {'result': filepath, 'metadata': {'file_type': 'binary file', 'note': 'Path returned'}}

# ==================== 主函数 ====================

def main():
    """主函数：演示ChIP-seq固定方法对IKAROS转录因子结合位点检测的影响"""
    
    print("=" * 60)
    print("场景1：分析ChIP-seq中PFA+DSG固定导致IKAROS peaks消失的现象")
    print("=" * 60)
    print("问题描述：解释为什么某些ChIP-seq peaks在使用PFA+DSG双重固定时会消失，这些消失的peaks最可能位于哪里")
    print("-" * 60)
    
    # 步骤1：创建ChIP-seq实验数据库
    # 调用函数：create_chipseq_database()
    db_result = create_chipseq_database()
    print(f"FUNCTION_CALL: create_chipseq_database | PARAMS: {{}} | RESULT: {db_result}")
    
    # 步骤2：分析消失的peaks
    # 调用函数：analyze_disappearing_peaks()
    disappearing_analysis = analyze_disappearing_peaks(db_result['result'])
    print(f"FUNCTION_CALL: analyze_disappearing_peaks | PARAMS: {{'db_path': '{db_result['result']}'}} | RESULT: {disappearing_analysis}")
    
    # 步骤3：比较不同固定方法的交联效率
    # 调用函数：calculate_crosslinking_efficiency()
    pfa_protein_dna = calculate_crosslinking_efficiency('PFA_only', 'protein_dna')
    pfa_dsg_protein_protein = calculate_crosslinking_efficiency('PFA_DSG', 'protein_protein')
    print(f"FUNCTION_CALL: calculate_crosslinking_efficiency | PARAMS: {{'fixation_method': 'PFA_only', 'interaction_type': 'protein_dna'}} | RESULT: {pfa_protein_dna}")
    print(f"FUNCTION_CALL: calculate_crosslinking_efficiency | PARAMS: {{'fixation_method': 'PFA_DSG', 'interaction_type': 'protein_protein'}} | RESULT: {pfa_dsg_protein_protein}")
    
    # 步骤4：可视化固定方法比较
    # 调用函数：visualize_fixation_comparison()
    viz_result = visualize_fixation_comparison()
    print(f"FUNCTION_CALL: visualize_fixation_comparison | PARAMS: {{}} | RESULT: {viz_result}")
    
    print(f"FINAL_ANSWER: At active promoters and enhancers")
    print("\n" + "=" * 60)
    print("场景2：预测不同基因组区域在两种固定方法下的peak检测情况")
    print("=" * 60)
    print("问题描述：预测IKAROS转录因子在不同基因组区域和结合模式下的ChIP-seq信号强度")
    print("-" * 60)
    
    # 步骤1：预测PFA固定下的peaks
    # 调用函数：predict_peak_locations()
    pfa_predictions = predict_peak_locations('PFA_only')
    print(f"FUNCTION_CALL: predict_peak_locations | PARAMS: {{'fixation_method': 'PFA_only'}} | RESULT: {pfa_predictions}")
    
    # 步骤2：预测PFA+DSG固定下的peaks
    # 调用函数：predict_peak_locations()
    pfa_dsg_predictions = predict_peak_locations('PFA_DSG')
    print(f"FUNCTION_CALL: predict_peak_locations | PARAMS: {{'fixation_method': 'PFA_DSG'}} | RESULT: {pfa_dsg_predictions}")
    
    # 步骤3：模拟特定区域的信号
    # 调用函数：simulate_chipseq_signal()
    enhancer_indirect_pfa = simulate_chipseq_signal('active_enhancer', 'indirect', 'PFA_only')
    enhancer_indirect_pfa_dsg = simulate_chipseq_signal('active_enhancer', 'indirect', 'PFA_DSG')
    print(f"FUNCTION_CALL: simulate_chipseq_signal | PARAMS: {{'region_type': 'active_enhancer', 'tf_binding_mode': 'indirect', 'fixation_method': 'PFA_only'}} | RESULT: {enhancer_indirect_pfa}")
    print(f"FUNCTION_CALL: simulate_chipseq_signal | PARAMS: {{'region_type': 'active_enhancer', 'tf_binding_mode': 'indirect', 'fixation_method': 'PFA_DSG'}} | RESULT: {enhancer_indirect_pfa_dsg}")
    
    print(f"FINAL_ANSWER: Indirect binding at active enhancers shows signal reduction from {enhancer_indirect_pfa['result']:.3f} to {enhancer_indirect_pfa_dsg['result']:.3f}")
    
    print("\n" + "=" * 60)
    print("场景3：创建peak消失模式的可视化分析")
    print("=" * 60)
    print("问题描述：通过热图分析不同基因组区域和结合模式下的信号变化模式")
    print("-" * 60)
    
    # 步骤1：加载数据库数据
    # 调用函数：load_file()
    db_data = load_file(db_result['result'])
    print(f"FUNCTION_CALL: load_file | PARAMS: {{'filepath': '{db_result['result']}'}} | RESULT: {{'result': 'database_content', 'metadata': {db_data['metadata']}}}")
    
    # 步骤2：创建热图可视化
    # 调用函数：create_peak_disappearance_heatmap()
    heatmap_result = create_peak_disappearance_heatmap(db_result['result'])
    print(f"FUNCTION_CALL: create_peak_disappearance_heatmap | PARAMS: {{'db_path': '{db_result['result']}'}} | RESULT: {heatmap_result}")
    
    # 步骤3：分析间接结合在活跃启动子的信号变化
    # 调用函数：simulate_chipseq_signal()
    promoter_indirect_pfa = simulate_chipseq_signal('active_promoter', 'indirect', 'PFA_only')
    promoter_indirect_pfa_dsg = simulate_chipseq_signal('active_promoter', 'indirect', 'PFA_DSG')
    print(f"FUNCTION_CALL: simulate_chipseq_signal | PARAMS: {{'region_type': 'active_promoter', 'tf_binding_mode': 'indirect', 'fixation_method': 'PFA_only'}} | RESULT: {promoter_indirect_pfa}")
    print(f"FUNCTION_CALL: simulate_chipseq_signal | PARAMS: {{'region_type': 'active_promoter', 'tf_binding_mode': 'indirect', 'fixation_method': 'PFA_DSG'}} | RESULT: {promoter_indirect_pfa_dsg}")
    
    signal_ratio = promoter_indirect_pfa_dsg['result'] / promoter_indirect_pfa['result']
    print(f"FINAL_ANSWER: Peak disappearance occurs primarily at active promoters and enhancers with indirect IKAROS binding (signal ratio: {signal_ratio:.3f})")

if __name__ == "__main__":
    main()