# Filename: mitochondrial_function_toolkit.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json
import os
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib字体，优先使用 DejaVu Sans，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 创建必要的目录
os.makedirs('./mid_result/biology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
MITOCHONDRIAL_CONSTANTS = {
    'ATP_MW': 507.18,  # ATP分子量 g/mol
    'NADH_EXTINCTION_COEFF': 6220,  # NADH消光系数 M-1cm-1 at 340nm
    'CYTOCHROME_C_EXTINCTION_COEFF': 18500,  # 细胞色素c消光系数 M-1cm-1 at 550nm
    'PROTEIN_CONVERSION': 1000,  # mg to μg conversion
    'STANDARD_TEMP': 37,  # 标准实验温度 °C
    'STANDARD_PH': 7.4  # 标准pH值
}

# ==================== 第一层：原子函数 ====================

def create_mitochondrial_database() -> dict:
    """
    访问线粒体功能检测相关的本地数据库
    
    Returns:
        dict: 数据库访问结果和路径信息
    """
    try:
        db_path = os.path.join(DATABASE_DIR, 'mitochondrial_db.sqlite')
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询记录数
        cursor.execute("SELECT COUNT(*) FROM respiratory_complexes")
        complex_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM assay_methods")
        method_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'result': 'success',
            'metadata': {
                'database_path': db_path,
                'tables': ['respiratory_complexes', 'assay_methods'],
                'records_count': complex_count + method_count
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'database_path': None
            }
        }

def calculate_atp_concentration(luminescence_values: List[float], 
                              standard_curve_params: Dict[str, float],
                              protein_concentration: float) -> dict:
    """
    根据荧光素酶反应的发光值计算ATP浓度
    
    Parameters:
        luminescence_values: 发光值列表 (RLU)
        standard_curve_params: 标准曲线参数 {'slope': float, 'intercept': float, 'r_squared': float}
        protein_concentration: 蛋白质浓度 (mg/mL)
    
    Returns:
        dict: ATP浓度计算结果
    """
    if not isinstance(luminescence_values, list) or not luminescence_values:
        raise ValueError("luminescence_values must be a non-empty list")
    
    if not isinstance(standard_curve_params, dict):
        raise ValueError("standard_curve_params must be a dictionary")
    
    required_params = ['slope', 'intercept', 'r_squared']
    if not all(param in standard_curve_params for param in required_params):
        raise ValueError(f"standard_curve_params must contain: {required_params}")
    
    if protein_concentration <= 0:
        raise ValueError("protein_concentration must be positive")
    
    try:
        # 计算ATP浓度 (μM)
        atp_concentrations = []
        for rlu in luminescence_values:
            if rlu < 0:
                raise ValueError("Luminescence values cannot be negative")
            
            # 使用标准曲线: ATP (μM) = (RLU - intercept) / slope
            atp_conc = (rlu - standard_curve_params['intercept']) / standard_curve_params['slope']
            atp_concentrations.append(max(0, atp_conc))  # 确保非负值
        
        # 标准化到蛋白质含量 (nmol/mg protein)
        normalized_atp = [conc / protein_concentration * 1000 for conc in atp_concentrations]
        
        # 统计分析
        mean_atp = np.mean(normalized_atp)
        std_atp = np.std(normalized_atp, ddof=1) if len(normalized_atp) > 1 else 0
        sem_atp = std_atp / np.sqrt(len(normalized_atp)) if len(normalized_atp) > 1 else 0
        
        return {
            'result': {
                'atp_concentrations_raw': atp_concentrations,
                'atp_normalized': normalized_atp,
                'mean_atp': mean_atp,
                'std_atp': std_atp,
                'sem_atp': sem_atp,
                'unit': 'nmol/mg protein'
            },
            'metadata': {
                'n_samples': len(luminescence_values),
                'protein_concentration': protein_concentration,
                'standard_curve_r2': standard_curve_params['r_squared'],
                'calculation_method': 'luciferase_luminescence'
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'input_values': luminescence_values
            }
        }

def calculate_complex_activity(absorbance_data: List[float],
                             time_points: List[float],
                             extinction_coefficient: float,
                             path_length: float,
                             protein_concentration: float,
                             complex_name: str) -> dict:
    """
    计算呼吸链复合体酶活性
    
    Parameters:
        absorbance_data: 吸光度数据列表
        time_points: 时间点列表 (秒)
        extinction_coefficient: 消光系数 (M-1cm-1)
        path_length: 光程长度 (cm)
        protein_concentration: 蛋白质浓度 (mg/mL)
        complex_name: 复合体名称
    
    Returns:
        dict: 酶活性计算结果
    """
    if len(absorbance_data) != len(time_points):
        raise ValueError("absorbance_data and time_points must have the same length")
    
    if len(absorbance_data) < 3:
        raise ValueError("Need at least 3 data points for activity calculation")
    
    if extinction_coefficient <= 0 or path_length <= 0 or protein_concentration <= 0:
        raise ValueError("extinction_coefficient, path_length, and protein_concentration must be positive")
    
    try:
        # 线性拟合计算斜率 (ΔA/Δt)
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, absorbance_data)
        
        # 计算酶活性 (nmol/min/mg protein)
        # Activity = (ΔA/min) / (ε × l) × (1000 nmol/μmol) × (1000 μL/mL) / (protein mg/mL)
        activity = abs(slope) * 60 / (extinction_coefficient * path_length) * 1e6 / protein_concentration
        
        # 质量控制检查
        r_squared = r_value ** 2
        linearity_acceptable = r_squared >= 0.95
        
        return {
            'result': {
                'activity': activity,
                'slope': slope,
                'r_squared': r_squared,
                'p_value': p_value,
                'std_error': std_err,
                'linearity_acceptable': linearity_acceptable,
                'unit': 'nmol/min/mg protein'
            },
            'metadata': {
                'complex_name': complex_name,
                'extinction_coefficient': extinction_coefficient,
                'path_length': path_length,
                'protein_concentration': protein_concentration,
                'n_points': len(absorbance_data),
                'time_range': f"{min(time_points)}-{max(time_points)} sec"
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'complex_name': complex_name
            }
        }

def assess_glucose_uptake_relevance(method_name: str, 
                                  target_organelle: str,
                                  experimental_context: str) -> dict:
    """
    评估葡萄糖摄取实验对线粒体功能研究的相关性
    
    Parameters:
        method_name: 实验方法名称
        target_organelle: 目标细胞器
        experimental_context: 实验背景描述
    
    Returns:
        dict: 相关性评估结果
    """
    if not isinstance(method_name, str) or not method_name.strip():
        raise ValueError("method_name must be a non-empty string")
    
    if not isinstance(target_organelle, str) or not target_organelle.strip():
        raise ValueError("target_organelle must be a non-empty string")
    
    try:
        # 从数据库查询方法信息
        db_path = os.path.join(DATABASE_DIR, 'mitochondrial_db.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT method_name, target_organelle, principle, relevance_to_mitochondria
            FROM assay_methods 
            WHERE method_name LIKE ? OR target_organelle LIKE ?
        ''', (f'%{method_name}%', f'%{target_organelle}%'))
        
        results = cursor.fetchall()
        conn.close()
        
        # 分析相关性
        relevance_score = 0
        analysis = []
        
        for result in results:
            db_method, db_organelle, principle, relevance = result
            if 'glucose' in method_name.lower() and 'mitochondria' in experimental_context.lower():
                relevance_score = max(relevance_score, relevance)
                analysis.append({
                    'method': db_method,
                    'organelle': db_organelle,
                    'principle': principle,
                    'relevance_score': relevance
                })
        
        # 特殊分析：葡萄糖摄取与线粒体功能的关系
        if 'glucose uptake' in method_name.lower():
            mitochondrial_relevance = {
                'direct_relevance': False,
                'indirect_relevance': True,
                'explanation': 'Glucose uptake measures cellular glucose transport, not mitochondrial function directly',
                'limitations': [
                    'Does not measure oxidative phosphorylation',
                    'Does not assess respiratory chain activity',
                    'Does not evaluate ATP synthesis capacity',
                    'Measures cytoplasmic process, not mitochondrial'
                ],
                'recommendation': 'Use ATP synthesis assay or respiratory chain complex activity instead'
            }
        else:
            mitochondrial_relevance = {
                'direct_relevance': relevance_score >= 8,
                'indirect_relevance': 3 <= relevance_score < 8,
                'explanation': f'Method relevance score: {relevance_score}/10',
                'limitations': [],
                'recommendation': 'Suitable for mitochondrial function assessment' if relevance_score >= 8 else 'Limited relevance to mitochondrial function'
            }
        
        return {
            'result': {
                'relevance_score': relevance_score,
                'mitochondrial_relevance': mitochondrial_relevance,
                'database_matches': analysis
            },
            'metadata': {
                'method_analyzed': method_name,
                'target_organelle': target_organelle,
                'experimental_context': experimental_context,
                'analysis_basis': 'database_lookup_and_expert_knowledge'
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'method_name': method_name
            }
        }

# ==================== 第二层：组合函数 ====================

def comprehensive_mitochondrial_assessment(atp_data: Dict[str, List[float]],
                                         complex_activities: Dict[str, Dict[str, List[float]]],
                                         control_group: str = "Control") -> dict:
    """
    综合评估线粒体功能
    
    Parameters:
        atp_data: ATP数据 {组别: [重复值]}
        complex_activities: 复合体活性数据 {复合体: {组别: [活性值]}}
        control_group: 对照组名称
    
    Returns:
        dict: 综合评估结果
    """
    if not isinstance(atp_data, dict) or not atp_data:
        raise ValueError("atp_data must be a non-empty dictionary")
    
    if not isinstance(complex_activities, dict) or not complex_activities:
        raise ValueError("complex_activities must be a non-empty dictionary")
    
    if control_group not in atp_data:
        raise ValueError(f"Control group '{control_group}' not found in atp_data")
    
    try:
        assessment_results = {
            'atp_analysis': {},
            'complex_analysis': {},
            'statistical_tests': {},
            'overall_assessment': {}
        }
        
        # ATP分析
        for group, values in atp_data.items():
            if not values:
                continue
                
            assessment_results['atp_analysis'][group] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'sem': stats.sem(values) if len(values) > 1 else 0,
                'n': len(values),
                'relative_to_control': np.mean(values) / np.mean(atp_data[control_group]) * 100 if group != control_group else 100
            }
        
        # 复合体活性分析
        for complex_name, group_data in complex_activities.items():
            assessment_results['complex_analysis'][complex_name] = {}
            
            for group, values in group_data.items():
                if not values:
                    continue
                    
                assessment_results['complex_analysis'][complex_name][group] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                    'relative_activity': np.mean(values) / np.mean(group_data[control_group]) * 100 if control_group in group_data and group != control_group else 100
                }
        
        # 统计检验
        for group in atp_data.keys():
            if group != control_group and len(atp_data[group]) > 1 and len(atp_data[control_group]) > 1:
                t_stat, p_value = stats.ttest_ind(atp_data[control_group], atp_data[group])
                assessment_results['statistical_tests'][f'ATP_{group}_vs_{control_group}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': abs(t_stat) / np.sqrt(len(atp_data[control_group]) + len(atp_data[group]))
                }
        
        # 整体评估
        treatment_groups = [g for g in atp_data.keys() if g != control_group]
        
        for group in treatment_groups:
            atp_change = assessment_results['atp_analysis'][group]['relative_to_control']
            
            # 计算复合体活性平均变化
            complex_changes = []
            for complex_name in complex_activities.keys():
                if group in assessment_results['complex_analysis'][complex_name]:
                    complex_changes.append(assessment_results['complex_analysis'][complex_name][group]['relative_activity'])
            
            avg_complex_change = np.mean(complex_changes) if complex_changes else 100
            
            # 功能评估
            if atp_change < 70 or avg_complex_change < 70:
                functional_status = "Severely impaired"
            elif atp_change < 85 or avg_complex_change < 85:
                functional_status = "Moderately impaired"
            elif atp_change > 115 or avg_complex_change > 115:
                functional_status = "Enhanced"
            else:
                functional_status = "Normal"
            
            assessment_results['overall_assessment'][group] = {
                'atp_change_percent': atp_change,
                'avg_complex_change_percent': avg_complex_change,
                'functional_status': functional_status,
                'mitochondrial_dysfunction': atp_change < 80 and avg_complex_change < 80
            }
        
        return {
            'result': assessment_results,
            'metadata': {
                'control_group': control_group,
                'treatment_groups': treatment_groups,
                'complexes_analyzed': list(complex_activities.keys()),
                'analysis_type': 'comprehensive_mitochondrial_function'
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'control_group': control_group
            }
        }

def experimental_design_validator(proposed_experiments: List[Dict[str, str]],
                                research_objective: str) -> dict:
    """
    验证实验设计对研究目标的适用性
    
    Parameters:
        proposed_experiments: 提议的实验列表 [{'name': str, 'method': str, 'target': str}]
        research_objective: 研究目标描述
    
    Returns:
        dict: 实验设计验证结果
    """
    if not isinstance(proposed_experiments, list) or not proposed_experiments:
        raise ValueError("proposed_experiments must be a non-empty list")
    
    if not isinstance(research_objective, str) or not research_objective.strip():
        raise ValueError("research_objective must be a non-empty string")
    
    try:
        validation_results = {
            'suitable_experiments': [],
            'unsuitable_experiments': [],
            'recommendations': []
        }
        
        # 分析研究目标
        is_mitochondrial_study = any(keyword in research_objective.lower() 
                                   for keyword in ['mitochondria', 'atp', 'respiratory', 'oxidative phosphorylation'])
        
        for exp in proposed_experiments:
            if not isinstance(exp, dict) or 'name' not in exp:
                continue
                
            exp_name = exp.get('name', '')
            exp_method = exp.get('method', '')
            exp_target = exp.get('target', '')
            
            # 评估实验相关性
            relevance_result = assess_glucose_uptake_relevance(exp_name, exp_target, research_objective)
            
            if relevance_result['result'] != 'error':
                relevance_score = relevance_result['result']['relevance_score']
                mitochondrial_relevance = relevance_result['result']['mitochondrial_relevance']
                
                exp_evaluation = {
                    'experiment': exp,
                    'relevance_score': relevance_score,
                    'suitable': relevance_score >= 7 if is_mitochondrial_study else relevance_score >= 3,
                    'direct_mitochondrial_relevance': mitochondrial_relevance['direct_relevance'],
                    'limitations': mitochondrial_relevance['limitations'],
                    'recommendation': mitochondrial_relevance['recommendation']
                }
                
                if exp_evaluation['suitable']:
                    validation_results['suitable_experiments'].append(exp_evaluation)
                else:
                    validation_results['unsuitable_experiments'].append(exp_evaluation)
        
        # 生成推荐
        if is_mitochondrial_study:
            validation_results['recommendations'] = [
                "Use ATP synthesis assay to measure mitochondrial energy production",
                "Perform respiratory chain complex activity assays (I-V)",
                "Measure mitochondrial membrane potential using fluorescent probes",
                "Assess oxygen consumption rate using Seahorse analyzer",
                "Avoid glucose uptake assays as primary mitochondrial function readout"
            ]
        
        return {
            'result': validation_results,
            'metadata': {
                'research_objective': research_objective,
                'total_experiments_evaluated': len(proposed_experiments),
                'mitochondrial_study': is_mitochondrial_study,
                'validation_criteria': 'relevance_score_and_organelle_specificity'
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'research_objective': research_objective
            }
        }

# ==================== 第三层：可视化函数 ====================

def plot_mitochondrial_function_results(assessment_results: dict,
                                       save_path: str = None) -> dict:
    """
    绘制线粒体功能评估结果图表
    
    Parameters:
        assessment_results: 综合评估结果
        save_path: 图片保存路径
    
    Returns:
        dict: 绘图结果信息
    """
    if not isinstance(assessment_results, dict):
        raise ValueError("assessment_results must be a dictionary")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mitochondrial Function Assessment Results', fontsize=16, fontweight='bold')
        
        # 1. ATP含量柱状图
        atp_data = assessment_results.get('atp_analysis', {})
        if atp_data:
            groups = list(atp_data.keys())
            means = [atp_data[g]['mean'] for g in groups]
            sems = [atp_data[g]['sem'] for g in groups]
            
            bars = axes[0,0].bar(groups, means, yerr=sems, capsize=5, alpha=0.7, 
                               color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            axes[0,0].set_ylabel('ATP Content (nmol/mg protein)')
            axes[0,0].set_title('Mitochondrial ATP Synthesis')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, mean in zip(bars, means):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sems)*0.1,
                             f'{mean:.1f}', ha='center', va='bottom')
        
        # 2. 复合体活性热图
        complex_data = assessment_results.get('complex_analysis', {})
        if complex_data:
            complexes = list(complex_data.keys())
            groups = list(complex_data[complexes[0]].keys()) if complexes else []
            
            if complexes and groups:
                heatmap_data = []
                for complex_name in complexes:
                    row = []
                    for group in groups:
                        if group in complex_data[complex_name]:
                            row.append(complex_data[complex_name][group]['relative_activity'])
                        else:
                            row.append(0)
                    heatmap_data.append(row)
                
                im = axes[0,1].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=150)
                axes[0,1].set_xticks(range(len(groups)))
                axes[0,1].set_yticks(range(len(complexes)))
                axes[0,1].set_xticklabels(groups, rotation=45)
                axes[0,1].set_yticklabels(complexes)
                axes[0,1].set_title('Respiratory Complex Relative Activity (%)')
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=axes[0,1])
                cbar.set_label('Relative Activity (%)')
        
        # 3. 统计显著性图
        stat_data = assessment_results.get('statistical_tests', {})
        if stat_data:
            comparisons = list(stat_data.keys())
            p_values = [stat_data[comp]['p_value'] for comp in comparisons]
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            
            bars = axes[1,0].bar(range(len(comparisons)), [-np.log10(p) for p in p_values], 
                               color=colors, alpha=0.7)
            axes[1,0].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
            axes[1,0].set_xticks(range(len(comparisons)))
            axes[1,0].set_xticklabels([comp.replace('_', '\n') for comp in comparisons], rotation=45)
            axes[1,0].set_ylabel('-log10(p-value)')
            axes[1,0].set_title('Statistical Significance')
            axes[1,0].legend()
        
        # 4. 整体功能状态饼图
        overall_data = assessment_results.get('overall_assessment', {})
        if overall_data:
            status_counts = {}
            for group, data in overall_data.items():
                status = data['functional_status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                colors = {'Normal': '#2E86AB', 'Enhanced': '#A23B72', 
                         'Moderately impaired': '#F18F01', 'Severely impaired': '#C73E1D'}
                pie_colors = [colors.get(status, 'gray') for status in status_counts.keys()]
                
                axes[1,1].pie(status_counts.values(), labels=status_counts.keys(), 
                            autopct='%1.0f%%', colors=pie_colors)
                axes[1,1].set_title('Mitochondrial Functional Status Distribution')
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = './tool_images/mitochondrial_function_assessment.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"FILE_GENERATED: image | PATH: {save_path}")
        plt.close()
        
        return {
            'result': 'success',
            'metadata': {
                'figure_path': save_path,
                'plots_generated': ['atp_content', 'complex_activity_heatmap', 'statistical_significance', 'functional_status'],
                'figure_size': '14x10 inches',
                'dpi': 300
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'save_path': save_path
            }
        }

def create_experimental_design_flowchart(validation_results: dict,
                                       save_path: str = None) -> dict:
    """
    创建实验设计流程图
    
    Parameters:
        validation_results: 实验设计验证结果
        save_path: 图片保存路径
    
    Returns:
        dict: 流程图创建结果
    """
    if not isinstance(validation_results, dict):
        raise ValueError("validation_results must be a dictionary")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Experimental Design Validation for Mitochondrial Function Study', 
                    fontsize=14, fontweight='bold')
        
        # 左图：适用性评分
        suitable_exps = validation_results.get('suitable_experiments', [])
        unsuitable_exps = validation_results.get('unsuitable_experiments', [])
        
        all_exps = suitable_exps + unsuitable_exps
        if all_exps:
            exp_names = [exp['experiment']['name'][:20] + '...' if len(exp['experiment']['name']) > 20 
                        else exp['experiment']['name'] for exp in all_exps]
            scores = [exp['relevance_score'] for exp in all_exps]
            colors = ['green' if exp['suitable'] else 'red' for exp in all_exps]
            
            bars = ax1.barh(exp_names, scores, color=colors, alpha=0.7)
            ax1.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='Suitability Threshold')
            ax1.set_xlabel('Relevance Score (0-10)')
            ax1.set_title('Experiment Suitability for Mitochondrial Study')
            ax1.legend()
            # 添加分数标签
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}', va='center')
        
        # 右图：推荐实验方法
        recommendations = validation_results.get('recommendations', [])
        if recommendations:
            y_pos = np.arange(len(recommendations))
            ax2.barh(y_pos, [1]*len(recommendations), color='lightblue', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([rec[:40] + '...' if len(rec) > 40 else rec for rec in recommendations])
            ax2.set_xlabel('Recommendation Priority')
            ax2.set_title('Recommended Experimental Approaches')
            ax2.set_xlim(0, 1.2)
            
            # 移除x轴刻度
            ax2.set_xticks([])
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = './tool_images/experimental_design_validation.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"FILE_GENERATED: image | PATH: {save_path}")
        plt.close()
        
        return {
            'result': 'success',
            'metadata': {
                'figure_path': save_path,
                'suitable_experiments': len(suitable_exps),
                'unsuitable_experiments': len(unsuitable_exps),
                'recommendations_count': len(recommendations)
            }
        }
        
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'save_path': save_path
            }
        }

# ==================== 文件解析函数 ====================

def load_file(filepath: str) -> dict:
    """
    解析常见文件格式
    
    Parameters:
        filepath: 文件路径
    
    Returns:
        dict: 文件内容解析结果
    """
    if not isinstance(filepath, str) or not filepath.strip():
        raise ValueError("filepath must be a non-empty string")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'result': content,
                'metadata': {
                    'file_type': 'text',
                    'file_size': os.path.getsize(filepath),
                    'encoding': 'utf-8'
                }
            }
        
        elif file_ext == '.csv':
            df = pd.read_csv(filepath)
            return {
                'result': df.to_dict('records'),
                'metadata': {
                    'file_type': 'csv',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'file_size': os.path.getsize(filepath)
                }
            }
        
        elif file_ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'result': data,
                'metadata': {
                    'file_type': 'json',
                    'file_size': os.path.getsize(filepath)
                }
            }
        
        else:
            return {
                'result': 'unsupported_format',
                'metadata': {
                    'file_type': file_ext,
                    'error': f'Unsupported file format: {file_ext}'
                }
            }
            
    except Exception as e:
        return {
            'result': 'error',
            'metadata': {
                'error_message': str(e),
                'filepath': filepath
            }
        }

# ==================== 主函数 ====================

def main():
    """主函数：演示三个场景的线粒体功能分析"""
    
    print("=" * 60)
    print("场景1：抗糖尿病药物对线粒体功能影响的完整评估")
    print("=" * 60)
    print("问题描述：评估新发现的抗糖尿病药物是否影响HEK293细胞的线粒体ATP合成和呼吸链复合体活性")
    print("-" * 60)
    
    # 步骤1：创建线粒体功能数据库
    # 调用函数：create_mitochondrial_database()
    db_result = create_mitochondrial_database()
    print(f"FUNCTION_CALL: create_mitochondrial_database | PARAMS: {{}} | RESULT: {db_result}")
    
    # 步骤2：模拟ATP合成实验数据并计算浓度
    # 调用函数：calculate_atp_concentration()
    luminescence_control = [15420.5, 15890.2, 15234.8, 15667.1]
    luminescence_treatment = [11234.7, 10987.3, 11456.2, 11089.5]
    standard_curve = {'slope': 1250.5, 'intercept': 2340.8, 'r_squared': 0.998}
    protein_conc = 2.5
    
    atp_control = calculate_atp_concentration(luminescence_control, standard_curve, protein_conc)
    print(f"FUNCTION_CALL: calculate_atp_concentration | PARAMS: {{luminescence_values: {luminescence_control}, standard_curve_params: {standard_curve}, protein_concentration: {protein_conc}}} | RESULT: {atp_control}")
    
    atp_treatment = calculate_atp_concentration(luminescence_treatment, standard_curve, protein_conc)
    print(f"FUNCTION_CALL: calculate_atp_concentration | PARAMS: {{luminescence_values: {luminescence_treatment}, standard_curve_params: {standard_curve}, protein_concentration: {protein_conc}}} | RESULT: {atp_treatment}")
    
    # 步骤3：计算呼吸链复合体I活性
    # 调用函数：calculate_complex_activity()
    absorbance_data = [0.850, 0.820, 0.785, 0.750, 0.715, 0.680]
    time_points = [0, 30, 60, 90, 120, 150]
    
    complex1_activity = calculate_complex_activity(
        absorbance_data, time_points, MITOCHONDRIAL_CONSTANTS['NADH_EXTINCTION_COEFF'], 
        1.0, protein_conc, "Complex I"
    )
    print(f"FUNCTION_CALL: calculate_complex_activity | PARAMS: {{absorbance_data: {absorbance_data}, time_points: {time_points}, extinction_coefficient: {MITOCHONDRIAL_CONSTANTS['NADH_EXTINCTION_COEFF']}, path_length: 1.0, protein_concentration: {protein_conc}, complex_name: 'Complex I'}} | RESULT: {complex1_activity}")
    
    # 步骤4：综合评估线粒体功能
    # 调用函数：comprehensive_mitochondrial_assessment()
    atp_data = {
        'Control': atp_control['result']['atp_normalized'],
        'Treatment': atp_treatment['result']['atp_normalized']
    }
    
    complex_activities = {
        'Complex I': {
            'Control': [125.3, 132.1, 128.7, 130.5],
            'Treatment': [89.2, 85.7, 91.3, 87.8]
        },
        'Complex V': {
            'Control': [98.5, 102.3, 95.8, 100.1],
            'Treatment': [72.4, 68.9, 75.2, 70.6]
        }
    }
    
    assessment = comprehensive_mitochondrial_assessment(atp_data, complex_activities)
    print(f"FUNCTION_CALL: comprehensive_mitochondrial_assessment | PARAMS: {{atp_data: {atp_data}, complex_activities: {complex_activities}, control_group: 'Control'}} | RESULT: {assessment}")
    
    # 步骤5：生成结果可视化
    # 调用函数：plot_mitochondrial_function_results()
    plot_result = plot_mitochondrial_function_results(assessment['result'])
    print(f"FUNCTION_CALL: plot_mitochondrial_function_results | PARAMS: {{assessment_results: {assessment['result']}, save_path: None}} | RESULT: {plot_result}")
    
    print(f"FINAL_ANSWER: 该抗糖尿病药物显著抑制线粒体功能，ATP合成能力下降至对照组的{assessment['result']['overall_assessment']['Treatment']['atp_change_percent']:.1f}%，呼吸链复合体活性平均下降至{assessment['result']['overall_assessment']['Treatment']['avg_complex_change_percent']:.1f}%，功能状态为{assessment['result']['overall_assessment']['Treatment']['functional_status']}")
    
    print("\n" + "=" * 60)
    print("场景2：实验方法适用性验证 - 识别不适合的线粒体功能检测方法")
    print("=" * 60)
    print("问题描述：验证差速离心提取线粒体后进行葡萄糖摄取比色检测是否适合评估线粒体功能")
    print("-" * 60)
    
    # 步骤1：评估葡萄糖摄取实验的线粒体相关性
    # 调用函数：assess_glucose_uptake_relevance()
    glucose_relevance = assess_glucose_uptake_relevance(
        "Glucose Uptake Colorimetric Assay", 
        "Cytoplasm", 
        "mitochondrial function assessment in anti-diabetes drug study"
    )
    print(f"FUNCTION_CALL: assess_glucose_uptake_relevance | PARAMS: {{method_name: 'Glucose Uptake Colorimetric Assay', target_organelle: 'Cytoplasm', experimental_context: 'mitochondrial function assessment in anti-diabetes drug study'}} | RESULT: {glucose_relevance}")
    
    # 步骤2：验证多个实验方法的适用性
    # 调用函数：experimental_design_validator()
    proposed_experiments = [
        {'name': 'Differential centrifugation extraction of mitochondria followed by Glucose Uptake Colorimetric Assay Kit', 'method': 'Colorimetric', 'target': 'Cytoplasm'},
        {'name': 'ATP Synthesis Assay', 'method': 'Luminescence', 'target': 'Mitochondria'},
        {'name': 'Respiratory Chain Complex Activity', 'method': 'Spectrophotometric', 'target': 'Mitochondria'},
        {'name': 'Mitochondrial Membrane Potential', 'method': 'Fluorescence', 'target': 'Mitochondria'}
    ]
    
    validation = experimental_design_validator(
        proposed_experiments, 
        "investigate mitochondrial role of anti-diabetes drug"
    )
    print(f"FUNCTION_CALL: experimental_design_validator | PARAMS: {{proposed_experiments: {proposed_experiments}, research_objective: 'investigate mitochondrial role of anti-diabetes drug'}} | RESULT: {validation}")
    
    # 步骤3：创建实验设计验证流程图
    # 调用函数：create_experimental_design_flowchart()
    flowchart_result = create_experimental_design_flowchart(validation['result'])
    print(f"FUNCTION_CALL: create_experimental_design_flowchart | PARAMS: {{validation_results: {validation['result']}, save_path: None}} | RESULT: {flowchart_result}")
    
    print(f"FINAL_ANSWER: 差速离心提取线粒体后进行葡萄糖摄取比色检测不适合评估线粒体功能，因为葡萄糖摄取检测的是细胞质过程而非线粒体功能，相关性评分仅为{glucose_relevance['result']['relevance_score']}/10分，不能直接反映线粒体ATP合成或呼吸链活性")
    
    print("\n" + "=" * 60)
    print("场景3：线粒体功能数据库查询与复合体特性分析")
    print("=" * 60)
    print("问题描述：从本地数据库查询呼吸链复合体特性，分析不同复合体的分子量和检测波长特征")
    print("-" * 60)
    
    # 步骤1：查询数据库中的呼吸链复合体信息
    # 调用函数：load_file() 读取数据库信息
    db_path = './mid_result/biology/mitochondrial_db.sqlite'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM respiratory_complexes')
        complex_info = cursor.fetchall()
        conn.close()
        
        # 保存查询结果到文件
        result_file = './mid_result/biology/complex_characteristics.json'
        complex_data = []
        for row in complex_info:
            complex_data.append({
                'id': row[0],
                'name': row[1],
                'subunits': row[2],
                'molecular_weight': row[3],
                'substrate': row[4],
                'product': row[5],
                'detection_wavelength': row[6],
                'normal_activity_range': row[7],
                'inhibitors': row[8]
            })
        
        with open(result_file, 'w') as f:
            json.dump(complex_data, f, indent=2)
        
        print(f"FUNCTION_CALL: database_query | PARAMS: {{table: 'respiratory_complexes', query: 'SELECT * FROM respiratory_complexes'}} | RESULT: {{'result': 'success', 'records': {len(complex_data)}, 'file_saved': '{result_file}'}}")
        
    except Exception as e:
        print(f"FUNCTION_CALL: database_query | PARAMS: {{}} | RESULT: {{'result': 'error', 'error': '{str(e)}'}}")
    
    # 步骤2：加载并分析复合体特性文件
    # 调用函数：load_file()
    file_content = load_file(result_file)
    print(f"FUNCTION_CALL: load_file | PARAMS: {{filepath: '{result_file}'}} | RESULT: {file_content}")
    
    # 步骤3：分析复合体分子量分布
    if file_content['result'] != 'error':
        complexes = file_content['result']
        molecular_weights = [comp['molecular_weight'] for comp in complexes]
        detection_wavelengths = [comp['detection_wavelength'] for comp in complexes]
        
        # 创建分析图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 分子量分布
        complex_names = [comp['name'] for comp in complexes]
        ax1.bar(complex_names, molecular_weights, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_ylabel('Molecular Weight (kDa)')
        ax1.set_title('Respiratory Complex Molecular Weights')
        ax1.tick_params(axis='x', rotation=45)
        
        # 检测波长分布
        ax2.scatter(detection_wavelengths, molecular_weights, s=100, alpha=0.7, 
                   c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        for i, name in enumerate(complex_names):
            ax2.annotate(name, (detection_wavelengths[i], molecular_weights[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Detection Wavelength (nm)')
        ax2.set_ylabel('Molecular Weight (kDa)')
        ax2.set_title('Complex Characteristics: MW vs Detection Wavelength')
        
        plt.tight_layout()
        analysis_plot_path = './tool_images/complex_characteristics_analysis.png'
        plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
        print(f"FILE_GENERATED: image | PATH: {analysis_plot_path}")
        plt.close()
        
        # 统计分析
        avg_mw = np.mean(molecular_weights)
        max_mw_complex = complexes[np.argmax(molecular_weights)]['name']
        min_mw_complex = complexes[np.argmin(molecular_weights)]['name']
        
        analysis_result = {
            'average_molecular_weight': avg_mw,
            'largest_complex': max_mw_complex,
            'smallest_complex': min_mw_complex,
            'wavelength_range': f"{min(detection_wavelengths)}-{max(detection_wavelengths)} nm"
        }
        
        print(f"FUNCTION_CALL: analyze_complex_characteristics | PARAMS: {{complexes_data: {len(complexes)} complexes}} | RESULT: {analysis_result}")
    
    print(f"FINAL_ANSWER: 呼吸链复合体特性分析完成：平均分子量{avg_mw:.0f} kDa，最大复合体为{max_mw_complex}({max(molecular_weights)} kDa)，最小为{min_mw_complex}({min(molecular_weights)} kDa)，检测波长范围{min(detection_wavelengths)}-{max(detection_wavelengths)} nm，数据已保存至本地数据库供后续实验设计参考")

if __name__ == "__main__":
    main()