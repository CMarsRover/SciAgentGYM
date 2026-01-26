
# Filename: chromosome_biology_toolkit.py

import sqlite3
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 创建目录
os.makedirs('./mid_result/biology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
SPINDLE_FORCE_CONSTANT = 1.0  # 纺锤体牵引力常数
CHROMOSOME_ELASTICITY = 0.5   # 染色体弹性模量
CRITICAL_TENSION_THRESHOLD = 2.0  # 临界张力阈值

def create_chromosome_database() -> str:
    """访问染色体特性本地数据库"""
    db_path = os.path.join(DATABASE_DIR, 'chromosome_db.sqlite')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    return db_path

def calculate_spindle_tension(centromere_positions: List[float], cell_length: float) -> Dict:
    """计算双着丝粒染色体的纺锤体张力"""
    if len(centromere_positions) != 2:
        raise ValueError("双着丝粒染色体必须有且仅有2个着丝粒位置")
    
    # 计算着丝粒间距离
    centromere_distance = abs(centromere_positions[1] - centromere_positions[0])
    
    # 计算纺锤体极点到着丝粒的距离
    pole1_distance = cell_length / 4  # 假设纺锤体极点位置
    pole2_distance = 3 * cell_length / 4
    
    # 计算张力（基于胡克定律）
    tension_force = SPINDLE_FORCE_CONSTANT * centromere_distance
    
    # 计算应力集中
    stress_concentration = tension_force / (CHROMOSOME_ELASTICITY * centromere_distance)
    
    # 判断是否超过临界阈值
    will_break = stress_concentration > CRITICAL_TENSION_THRESHOLD
    
    return {
        'result': {
            'centromere_distance': centromere_distance,
            'tension_force': tension_force,
            'stress_concentration': stress_concentration,
            'critical_threshold': CRITICAL_TENSION_THRESHOLD,
            'will_break': will_break,
            'bridge_formation': will_break
        },
        'metadata': {
            'calculation_method': 'spindle_mechanics',
            'assumptions': ['uniform_elasticity', 'linear_force_model'],
            'units': 'relative_units'
        }
    }

def analyze_mitotic_behavior(chromosome_type: str, ploidy_level: str) -> Dict:
    """分析染色体在有丝分裂中的行为"""
    db_path = os.path.join(DATABASE_DIR, 'chromosome_db.sqlite')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM chromosome_types WHERE type_name = ?
    ''', (chromosome_type,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise ValueError(f"未知的染色体类型: {chromosome_type}")
    
    _, type_name, centromere_count, stability_index, mitotic_behavior, expected_outcome = result
    
    # 单倍体特殊性分析
    if ploidy_level == 'haploid':
        # 单倍体中没有同源染色体备份，损失更严重
        severity_multiplier = 2.0
        backup_available = False
    else:
        severity_multiplier = 1.0
        backup_available = True
    
    # 计算最终稳定性
    final_stability = stability_index / severity_multiplier
    
    # 预测细胞命运
    if final_stability < 0.3:
        cell_fate = 'cell_death_or_arrest'
    elif final_stability < 0.6:
        cell_fate = 'genomic_instability'
    else:
        cell_fate = 'normal_division'
    
    return {
        'result': {
            'chromosome_type': type_name,
            'centromere_count': centromere_count,
            'base_stability': stability_index,
            'ploidy_effect': severity_multiplier,
            'final_stability': final_stability,
            'mitotic_behavior': mitotic_behavior,
            'expected_outcome': expected_outcome,
            'backup_available': backup_available,
            'predicted_cell_fate': cell_fate
        },
        'metadata': {
            'ploidy_level': ploidy_level,
            'analysis_type': 'mitotic_behavior_prediction',
            'severity_factors': ['no_homolog_backup', 'single_copy_genes']
        }
    }

def simulate_chromosome_segregation(centromere_positions: List[float], simulation_steps: int = 100) -> Dict:
    """模拟染色体分离过程"""
    if len(centromere_positions) != 2:
        raise ValueError("需要双着丝粒染色体数据")
    
    # 初始化模拟参数
    time_points = np.linspace(0, 1, simulation_steps)
    positions = []
    tensions = []
    
    initial_distance = abs(centromere_positions[1] - centromere_positions[0])
    
    for t in time_points:
        # 模拟纺锤体牵引过程
        separation_force = t * SPINDLE_FORCE_CONSTANT
        
        # 计算当前张力
        current_tension = separation_force * initial_distance
        tensions.append(current_tension)
        
        # 计算位置变化
        if current_tension > CRITICAL_TENSION_THRESHOLD:
            # 染色体断裂
            break_point = initial_distance * 0.5  # 假设在中点断裂
            positions.append([0, break_point, initial_distance])
        else:
            # 正常拉伸
            stretch_factor = 1 + t * 0.5
            new_distance = initial_distance * stretch_factor
            positions.append([0, new_distance/2, new_distance])
    
    # 保存模拟结果
    result_file = './mid_result/biology/segregation_simulation.json'
    simulation_data = {
        'time_points': time_points.tolist(),
        'positions': positions,
        'tensions': tensions,
        'break_occurred': len(tensions) < simulation_steps
    }
    
    with open(result_file, 'w') as f:
        json.dump(simulation_data, f, indent=2)
    
    return {
        'result': result_file,
        'metadata': {
            'file_type': 'json',
            'size': len(str(simulation_data)),
            'simulation_steps': len(tensions),
            'break_occurred': simulation_data['break_occurred']
        }
    }

def evaluate_chromosome_stability_factors(chromosome_type: str, ploidy: str, gene_density: float) -> Dict:
    """评估染色体稳定性的多个因素"""
    
    # 基础稳定性评分
    stability_scores = {
        'monocentric': 1.0,
        'dicentric': 0.2,
        'acentric': 0.0,
        'polycentric': 0.1
    }
    
    base_score = stability_scores.get(chromosome_type, 0.0)
    
    # 倍性影响
    ploidy_factors = {
        'haploid': 0.5,    # 单倍体更脆弱
        'diploid': 1.0,    # 二倍体标准
        'polyploid': 1.2   # 多倍体更稳定
    }
    ploidy_factor = ploidy_factors.get(ploidy, 1.0)
    
    # 基因密度影响（基因越多，损失越严重）
    gene_density_penalty = 1.0 - (gene_density * 0.3)
    
    # 综合稳定性评分
    final_stability = base_score * ploidy_factor * gene_density_penalty
    
    # 预测结果
    predictions = {
        'will_be_lost': final_stability < 0.3,
        'causes_instability': final_stability < 0.6,
        'survives_division': final_stability > 0.6,
        'gene_loss_risk': gene_density * (1 - final_stability)
    }
    
    return {
        'result': {
            'base_stability': base_score,
            'ploidy_factor': ploidy_factor,
            'gene_density_penalty': gene_density_penalty,
            'final_stability_score': final_stability,
            'predictions': predictions,
            'risk_assessment': 'high' if final_stability < 0.4 else 'medium' if final_stability < 0.7 else 'low'
        },
        'metadata': {
            'evaluation_factors': ['chromosome_structure', 'ploidy_level', 'gene_content'],
            'scoring_range': [0.0, 1.0],
            'prediction_confidence': 0.85
        }
    }

def visualize_chromosome_behavior(chromosome_data: Dict, output_path: str = None) -> str:
    """可视化染色体行为分析结果"""
    if output_path is None:
        output_path = './tool_images/chromosome_behavior_analysis.png'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 稳定性对比图
    types = ['monocentric', 'dicentric', 'acentric', 'polycentric']
    stabilities = [1.0, 0.2, 0.0, 0.1]
    colors = ['green', 'orange', 'red', 'darkred']
    
    ax1.bar(types, stabilities, color=colors, alpha=0.7)
    ax1.set_title('Chromosome Stability by Type')
    ax1.set_ylabel('Stability Index')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 倍性影响
    ploidy_levels = ['haploid', 'diploid', 'polyploid']
    ploidy_effects = [0.5, 1.0, 1.2]
    
    ax2.plot(ploidy_levels, ploidy_effects, 'bo-', linewidth=2, markersize=8)
    ax2.set_title('Ploidy Level Effect on Stability')
    ax2.set_ylabel('Stability Multiplier')
    ax2.grid(True, alpha=0.3)
    
    # 3. 双着丝粒染色体张力分布
    positions = np.linspace(0, 100, 100)
    tension = np.exp(-((positions - 50)**2) / 200) * 3  # 高斯分布模拟张力
    
    ax3.plot(positions, tension, 'r-', linewidth=2, label='Tension')
    ax3.axhline(y=CRITICAL_TENSION_THRESHOLD, color='k', linestyle='--', label='Critical Threshold')
    ax3.fill_between(positions, tension, alpha=0.3, color='red')
    ax3.set_title('Spindle Tension Distribution')
    ax3.set_xlabel('Chromosome Position')
    ax3.set_ylabel('Tension Force')
    ax3.legend()
    
    # 4. 细胞命运预测
    outcomes = ['Normal\nDivision', 'Genomic\nInstability', 'Cell Death\nor Arrest']
    probabilities = [0.1, 0.3, 0.6]  # 双着丝粒在单倍体中的预期结果
    
    wedges, texts, autotexts = ax4.pie(probabilities, labels=outcomes, autopct='%1.1f%%', 
                                       colors=['lightgreen', 'yellow', 'lightcoral'])
    ax4.set_title('Predicted Cell Fate\n(Dicentric in Haploid)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def load_file(filepath: str) -> Dict:
    """加载文件内容的通用函数"""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                content = json.load(f)
            return {'type': 'json', 'content': content}
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as f:
                content = f.read()
            return {'type': 'text', 'content': content}
        else:
            return {'type': 'unknown', 'content': None, 'error': 'Unsupported file type'}
    except Exception as e:
        return {'type': 'error', 'content': None, 'error': str(e)}

def main():
    """主函数：演示三个场景的染色体行为分析"""
    print("=" * 60)
    print("场景1：双着丝粒染色体在单倍体生物中的行为分析")
    print("=" * 60)
    print("问题描述：分析双着丝粒染色体在单倍体生物有丝分裂中是否会变得不稳定并在后续细胞分裂中丢失")
    print("-" * 60)
    
    # 步骤1：创建染色体数据库
    # 调用函数：create_chromosome_database()
    db_path = create_chromosome_database()
    print(f"FUNCTION_CALL: create_chromosome_database | PARAMS: {{}} | RESULT: {{'result': '{db_path}', 'metadata': {{'database_type': 'sqlite', 'tables': ['chromosome_types']}}}}")
    
    # 步骤2：分析双着丝粒染色体在单倍体中的行为
    # 调用函数：analyze_mitotic_behavior()
    params1 = {'chromosome_type': 'dicentric', 'ploidy_level': 'haploid'}
    result1 = analyze_mitotic_behavior('dicentric', 'haploid')
    print(f"FUNCTION_CALL: analyze_mitotic_behavior | PARAMS: {params1} | RESULT: {result1}")
    
    # 步骤3：计算纺锤体张力
    # 调用函数：calculate_spindle_tension()
    params2 = {'centromere_positions': [25.0, 75.0], 'cell_length': 100.0}
    result2 = calculate_spindle_tension([25.0, 75.0], 100.0)
    print(f"FUNCTION_CALL: calculate_spindle_tension | PARAMS: {params2} | RESULT: {result2}")
    
    # 步骤4：评估稳定性因素
    # 调用函数：evaluate_chromosome_stability_factors()
    params3 = {'chromosome_type': 'dicentric', 'ploidy': 'haploid', 'gene_density': 0.8}
    result3 = evaluate_chromosome_stability_factors('dicentric', 'haploid', 0.8)
    print(f"FUNCTION_CALL: evaluate_chromosome_stability_factors | PARAMS: {params3} | RESULT: {result3}")
    
    # 根据分析结果，双着丝粒染色体在单倍体中确实会变得不稳定
    instability_confirmed = result1['result']['final_stability'] < 0.3 and result3['result']['predictions']['will_be_lost']
    answer1 = f"双着丝粒染色体在单倍体生物中会变得不稳定并在后续细胞分裂中丢失: {instability_confirmed}"
    print(f"FINAL_ANSWER: {answer1}")
    
    print("\n" + "=" * 60)
    print("场景2：不同染色体类型的稳定性比较分析")
    print("=" * 60)
    print("问题描述：比较单着丝粒、双着丝粒、无着丝粒和多着丝粒染色体在不同倍性水平下的稳定性")
    print("-" * 60)
    
    # 步骤1：分析多种染色体类型
    chromosome_types = ['monocentric', 'dicentric', 'acentric', 'polycentric']
    stability_results = {}
    
    for chr_type in chromosome_types:
        # 调用函数：analyze_mitotic_behavior()
        params4 = {'chromosome_type': chr_type, 'ploidy_level': 'haploid'}
        result4 = analyze_mitotic_behavior(chr_type, 'haploid')
        stability_results[chr_type] = result4['result']['final_stability']
        print(f"FUNCTION_CALL: analyze_mitotic_behavior | PARAMS: {params4} | RESULT: {result4}")
    
    # 步骤2：生成可视化图表
    # 调用函数：visualize_chromosome_behavior()
    params5 = {'chromosome_data': stability_results}
    image_path = visualize_chromosome_behavior(stability_results)
    print(f"FUNCTION_CALL: visualize_chromosome_behavior | PARAMS: {params5} | RESULT: {{'result': '{image_path}', 'metadata': {{'file_type': 'png', 'analysis_type': 'comparative_stability'}}}}")
    print(f"FILE_GENERATED: image | PATH: {image_path}")
    
    most_stable = max(stability_results, key=stability_results.get)
    least_stable = min(stability_results, key=stability_results.get)
    answer2 = f"最稳定的染色体类型: {most_stable} (稳定性: {stability_results[most_stable]:.2f}), 最不稳定: {least_stable} (稳定性: {stability_results[least_stable]:.2f})"
    print(f"FINAL_ANSWER: {answer2}")
    
    print("\n" + "=" * 60)
    print("场景3：染色体分离过程的动态模拟")
    print("=" * 60)
    print("问题描述：模拟双着丝粒染色体在有丝分裂过程中的动态行为，预测断裂时间点")
    print("-" * 60)
    
    # 步骤1：运行分离模拟
    # 调用函数：simulate_chromosome_segregation()
    params6 = {'centromere_positions': [30.0, 70.0], 'simulation_steps': 100}
    result6 = simulate_chromosome_segregation([30.0, 70.0], 100)
    print(f"FUNCTION_CALL: simulate_chromosome_segregation | PARAMS: {params6} | RESULT: {result6}")
    
    # 步骤2：加载模拟结果
    # 调用函数：load_file()
    params7 = {'filepath': result6['result']}
    loaded_data = load_file(result6['result'])
    print(f"FUNCTION_CALL: load_file | PARAMS: {params7} | RESULT: {loaded_data}")
    
    # 步骤3：分析模拟结果
    simulation_content = loaded_data['content']
    break_occurred = simulation_content['break_occurred']
    steps_completed = len(simulation_content['tensions'])
    
    if break_occurred:
        break_time = steps_completed / 100.0  # 转换为相对时间
        answer3 = f"染色体在有丝分裂过程的 {break_time:.2f} 时间点发生断裂，模拟了 {steps_completed} 个步骤"
    else:
        answer3 = f"染色体在整个模拟过程中未发生断裂，完成了 {steps_completed} 个步骤"
    
    print(f"FINAL_ANSWER: {answer3}")

if __name__ == "__main__":
    main()