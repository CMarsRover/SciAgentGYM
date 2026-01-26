import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import json
import csv
import os
from typing import Dict, List, Any, Optional, Union
warnings.filterwarnings("ignore")

# ================== 1. 改进的数学逻辑模块（Math Func）==================
class ReactivityAnalyzer:
    
    def __init__(self, compounds_file: Optional[str] = None):
        """
        初始化反应性分析器
        
        Args:
            compounds_file: 可选的化合物数据库文件路径（JSON或CSV格式）
        """
        # 默认物质库
        self.compounds = {
            'Ni2O3': {
                'formula': r'$\mathrm{Ni}_{2}\mathrm{O}_{3}$',
                'color_solid': 'black',
                'react_HCl': {
                    'gas': None,  # Ni2O3 不产生Cl2
                    'ion': 'Ni²⁺',
                    'solution_color': 'green'  # 实际为绿色，非蓝色
                },
                'dilution_effect': 'stable',
                'KSCN_acetone': None,
                'notes': 'Ni²⁺ 与SCN⁻在丙酮中不显蓝色'
            },
            'Co2O3': {
                'formula': r'$\mathrm{Co}_{2}\mathrm{O}_{3}$',
                'color_solid': 'black',
                'react_HCl': {
                    'gas': 'Cl₂',  # 强氧化性，氧化HCl
                    'ion': 'Co²⁺',
                    'solution_color': 'blue'  # CoCl₄²⁻ 在浓HCl中为蓝色
                },
                'dilution_effect': 'pale pink',  # 稀释后[Co(H₂O)₆]²⁺为粉红色
                'KSCN_acetone': {
                    'color': 'deep blue',  # Co²⁺ + SCN⁻ → [Co(SCN)₄]²⁻（在丙酮中萃取显深蓝）
                    'condition': 'acetone'
                },
                'notes': '符合所有现象'
            },
            'Fe2O3': {
                'formula': r'$\mathrm{Fe}_{2}\mathrm{O}_{3}$',
                'color_solid': 'red-brown',
                'react_HCl': {
                    'gas': None,
                    'ion': 'Fe³⁺',
                    'solution_color': 'yellow/brown'
                },
                'dilution_effect': 'light yellow',
                'KSCN_acetone': {
                    'color': 'blood red',  # Fe³⁺ + SCN⁻ → [Fe(SCN)]²⁺（水相红色）
                    'condition': 'aqueous'
                },
                'notes': '固体非黑色，溶液非蓝色'
            },
            'MnO2': {
                'formula': r'$\mathrm{MnO}_{2}$',
                'color_solid': 'black',
                'react_HCl': {
                    'gas': 'Cl₂',  # MnO₂ + 4HCl → MnCl₂ + Cl₂ + 2H₂O
                    'ion': 'Mn²⁺',
                    'solution_color': 'pale pink'  # Mn²⁺极淡粉，近乎无色
                },
                'dilution_effect': 'colorless',
                'KSCN_acetone': None,
                'notes': '溶液不呈蓝色，稀释不变红'
            }
        }
        
        # 如果提供了文件路径，尝试加载
        if compounds_file:
            self.load_compounds_from_file(compounds_file)
    
    def add_compound(self, name: str, compound_data: Dict[str, Any]) -> None:
        """
        动态添加新化合物
        
        Args:
            name: 化合物名称
            compound_data: 化合物数据字典
        """
        self.compounds[name] = compound_data
        print(f"✅ 已添加化合物: {name}")
    
    def remove_compound(self, name: str) -> bool:
        """
        移除化合物
        
        Args:
            name: 化合物名称
            
        Returns:
            bool: 是否成功移除
        """
        if name in self.compounds:
            del self.compounds[name]
            print(f"✅ 已移除化合物: {name}")
            return True
        else:
            print(f"❌ 化合物 {name} 不存在")
            return False
    
    def update_compound(self, name: str, compound_data: Dict[str, Any]) -> bool:
        """
        更新化合物信息
        
        Args:
            name: 化合物名称
            compound_data: 新的化合物数据
            
        Returns:
            bool: 是否成功更新
        """
        if name in self.compounds:
            self.compounds[name].update(compound_data)
            print(f"✅ 已更新化合物: {name}")
            return True
        else:
            print(f"❌ 化合物 {name} 不存在，无法更新")
            return False
    
    def load_compounds_from_file(self, file_path: str) -> bool:
        """
        从文件加载化合物数据库
        
        Args:
            file_path: 文件路径（支持JSON和CSV格式）
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_compounds = json.load(f)
                self.compounds.update(new_compounds)
                print(f"✅ 从JSON文件加载了 {len(new_compounds)} 个化合物")
                
            elif file_path.endswith('.csv'):
                new_compounds = {}
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # 解析CSV行数据为化合物格式
                        compound_data = self._parse_csv_row_to_compound(row)
                        if compound_data:
                            new_compounds[row['name']] = compound_data
                
                self.compounds.update(new_compounds)
                print(f"✅ 从CSV文件加载了 {len(new_compounds)} 个化合物")
                
            else:
                print(f"❌ 不支持的文件格式: {file_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return False
    
    def save_compounds_to_file(self, file_path: str) -> bool:
        """
        保存化合物数据库到文件
        
        Args:
            file_path: 文件路径（支持JSON和CSV格式）
            
        Returns:
            bool: 是否成功保存
        """
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.compounds, f, ensure_ascii=False, indent=2)
                print(f"✅ 化合物数据库已保存到: {file_path}")
                
            elif file_path.endswith('.csv'):
                # 将化合物数据转换为CSV格式
                csv_data = self._convert_compounds_to_csv()
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
                print(f"✅ 化合物数据库已保存到: {file_path}")
                
            else:
                print(f"❌ 不支持的文件格式: {file_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            return False
    
    def _parse_csv_row_to_compound(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        将CSV行数据解析为化合物格式
        
        Args:
            row: CSV行数据
            
        Returns:
            化合物数据字典或None
        """
        try:
            # 这里需要根据实际的CSV格式进行调整
            compound_data = {
                'formula': row.get('formula', ''),
                'color_solid': row.get('color_solid', ''),
                'react_HCl': {
                    'gas': row.get('gas', None) if row.get('gas') != '' else None,
                    'ion': row.get('ion', ''),
                    'solution_color': row.get('solution_color', '')
                },
                'dilution_effect': row.get('dilution_effect', ''),
                'KSCN_acetone': {
                    'color': row.get('kscn_color', ''),
                    'condition': row.get('kscn_condition', '')
                } if row.get('kscn_color') else None,
                'notes': row.get('notes', '')
            }
            return compound_data
        except Exception as e:
            print(f"❌ 解析CSV行失败: {e}")
            return None
    
    def _convert_compounds_to_csv(self) -> List[Dict[str, str]]:
        """
        将化合物数据转换为CSV格式
        
        Returns:
            CSV格式的数据列表
        """
        csv_data = []
        for name, data in self.compounds.items():
            row = {
                'name': name,
                'formula': data.get('formula', ''),
                'color_solid': data.get('color_solid', ''),
                'gas': data.get('react_HCl', {}).get('gas', ''),
                'ion': data.get('react_HCl', {}).get('ion', ''),
                'solution_color': data.get('react_HCl', {}).get('solution_color', ''),
                'dilution_effect': data.get('dilution_effect', ''),
                'kscn_color': data.get('KSCN_acetone', {}).get('color', '') if data.get('KSCN_acetone') else '',
                'kscn_condition': data.get('KSCN_acetone', {}).get('condition', '') if data.get('KSCN_acetone') else '',
                'notes': data.get('notes', '')
            }
            csv_data.append(row)
        return csv_data
    
    def search_compounds(self, criteria: Dict[str, Any]) -> List[str]:
        """
        根据条件搜索化合物
        
        Args:
            criteria: 搜索条件字典
            
        Returns:
            匹配的化合物名称列表
        """
        matches = []
        for name, data in self.compounds.items():
            match = True
            for key, value in criteria.items():
                if key == 'color_solid':
                    if data.get('color_solid') != value:
                        match = False
                        break
                elif key == 'gas':
                    if data.get('react_HCl', {}).get('gas') != value:
                        match = False
                        break
                elif key == 'solution_color':
                    if value not in str(data.get('react_HCl', {}).get('solution_color', '')):
                        match = False
                        break
                elif key == 'ion':
                    if value not in str(data.get('react_HCl', {}).get('ion', '')):
                        match = False
                        break
            
            if match:
                matches.append(name)
        
        return matches
    
    def get_compound_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取化合物详细信息
        
        Args:
            name: 化合物名称
            
        Returns:
            化合物信息字典或None
        """
        return self.compounds.get(name)
    
    def list_all_compounds(self) -> List[str]:
        """
        列出所有化合物名称
        
        Returns:
            化合物名称列表
        """
        return list(self.compounds.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        total = len(self.compounds)
        colors = {}
        gases = {}
        ions = set()
        
        for data in self.compounds.values():
            # 统计固体颜色
            color = data.get('color_solid', 'unknown')
            colors[color] = colors.get(color, 0) + 1
            
            # 统计气体产物
            gas = data.get('react_HCl', {}).get('gas', 'none')
            gases[gas] = gases.get(gas, 0) + 1
            
            # 统计离子类型
            ion = data.get('react_HCl', {}).get('ion', '')
            if ion:
                ions.add(ion)
        
        return {
            'total_compounds': total,
            'color_distribution': colors,
            'gas_distribution': gases,
            'unique_ions': list(ions),
            'ion_count': len(ions)
        }

    def match_phenomenon(self, observed_phenomena):
        
        scores = {}
        for name, data in self.compounds.items():
            score = 0
            total_criteria = 0

            # 1. 固体颜色
            total_criteria += 1
            if data['color_solid'] == observed_phenomena['solid_color']:
                score += 1

            # 2. 与HCl反应产生气体
            if 'gas' in observed_phenomena:
                total_criteria += 1
                if (observed_phenomena['gas'] == 'Cl₂' and 
                    data['react_HCl']['gas'] == 'Cl₂'):
                    score += 1
                elif observed_phenomena['gas'] is None and data['react_HCl']['gas'] is None:
                    score += 1

            # 3. 反应后溶液颜色
            total_criteria += 1
            if observed_phenomena['solution_color_conc'] in data['react_HCl']['solution_color']:
                score += 1

            # 4. 稀释后颜色
            total_criteria += 1
            if (observed_phenomena['solution_color_dilute'] in 
                str(data['dilution_effect'])):
                score += 1

            # 5. KSCN + 丙酮现象
            if observed_phenomena['KSCN_acetone']:
                total_criteria += 1
                if (data['KSCN_acetone'] and 
                    data['KSCN_acetone']['color'] == observed_phenomena['KSCN_acetone']):
                    score += 1

            scores[name] = score / total_criteria if total_criteria > 0 else 0
        
        return scores

    def predict_compound(self, observed_phenomena):
        
        scores = self.match_phenomenon(observed_phenomena)
        best = max(scores, key=scores.get)
        return best, scores[best], scores

# ================== 2. 编码功能模块（Coding Func）==================
def solve_chemical_puzzle():
    
    analyzer = ReactivityAnalyzer()
    
    # 观察到的现象
    observed = {
        'solid_color': 'black',
        'gas': 'Cl₂',
        'solution_color_conc': 'blue',
        'solution_color_dilute': 'pale pink',  # "淡红"即粉红
        'KSCN_acetone': 'deep blue'
    }
    
    prediction, confidence, all_scores = analyzer.predict_compound(observed)
    
    result = {
        'predicted_compound': prediction,
        'formula': analyzer.compounds[prediction]['formula'],
        'confidence': confidence,
        'all_scores': all_scores,
        'explanation': analyzer.compounds[prediction]['notes']
    }
    
    return result

# ================== 3. 可视化模块（Visual Func）==================
def visualize_analysis(result):
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # 子图1：预测结果柱状图
    ax1 = fig.add_subplot(gs[0, :])
    compounds = list(result['all_scores'].keys())
    scores = [result['all_scores'][c] for c in compounds]
    colors = ['lightcoral' if c != result['predicted_compound'] else 'lightgreen' for c in compounds]
    
    bars = ax1.bar(compounds, scores, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("Matching Score for Each Candidate Compound", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Match Score (0-1)")
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{score:.2f}', ha='center', va='bottom')

    # 子图2：颜色变化路径
    ax2 = fig.add_subplot(gs[1, 0])
    steps = [
        ("Solid A (Black)", "black"),
        ("+ HCl → Gas (Yellow-Green)", "yellowgreen"),
        ("Conc. Solution (Blue)", "blue"),
        ("After Dilution (Pale Pink)", "pink"),
        ("+ KSCN/Acetone (Deep Blue in Acetone)", "deepskyblue")
    ]
    for i, (label, color) in enumerate(steps):
        rect = Rectangle((0.1, 0.7 - i*0.15), 0.3, 0.1, facecolor=color, edgecolor='black')
        ax2.add_patch(rect)
        ax2.text(0.5, 0.7 - i*0.15 + 0.05, label, va='center', ha='left', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.9)
    ax2.axis('off')
    ax2.set_title("Observed Color Changes", fontweight='bold')

    # 子图3：候选化合物信息对比
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    title = f"✅ Best Match: {result['predicted_compound']} → {result['formula']} Confidence: {result['confidence']:.2f}"
    ax3.text(0.1, 0.9, title, fontsize=12, fontweight='bold', color='darkgreen')

    explanation = result['explanation']
    ax3.text(0.1, 0.7, f"Explanation: {explanation}", fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 子图4：各化合物反应性特征热力图
    ax4 = fig.add_subplot(gs[2, :])
    features = ['Solid Color', 'Gas (Cl₂)', 'Sol. Color', 'Dilution', 'KSCN+Acetone']
    data = []
    for c in compounds:
        row = []
        d = analyzer.compounds[c]
        row.append(1 if d['color_solid'] == 'black' else 0)
        row.append(1 if d['react_HCl']['gas'] == 'Cl₂' else 0)
        row.append(1 if 'blue' in str(d['react_HCl']['solution_color']) else 0)
        row.append(1 if 'pink' in str(d['dilution_effect']) else 0)
        row.append(1 if d['KSCN_acetone'] and d['KSCN_acetone']['color']=='deep blue' else 0)
        data.append(row)
    
    im = ax4.imshow(data, cmap='Blues', aspect='auto')
    ax4.set_xticks(np.arange(len(features)))
    ax4.set_yticks(np.arange(len(compounds)))
    ax4.set_xticklabels(features, rotation=45)
    ax4.set_yticklabels([analyzer.compounds[c]['formula'] for c in compounds])
    ax4.set_title("Reactivity Feature Matrix (1 = matches observation)")
    for i in range(len(compounds)):
        for j in range(len(features)):
            text = ax4.text(j, i, data[i][j], ha="center", va="center", color="w" if data[i][j] else "black")

    plt.suptitle("Chemical Reactivity Analysis: Identification of Black Solid A", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    return fig

# ================== 主程序执行 ==================
if __name__ == "__main__":
    print("🔬 改进的反应性分析器演示")
    print("=" * 50)
    
    # 初始化分析器
    analyzer = ReactivityAnalyzer()
    
    # 1. 显示当前数据库统计
    print("\n📊 当前数据库统计:")
    stats = analyzer.get_statistics()
    print(f"总化合物数: {stats['total_compounds']}")
    print(f"颜色分布: {stats['color_distribution']}")
    print(f"气体产物分布: {stats['gas_distribution']}")
    print(f"离子类型: {stats['unique_ions']}")
    
    # 2. 动态添加新化合物
    print("\n➕ 添加新化合物 CuO:")
    new_compound = {
        'formula': r'$\mathrm{CuO}$',
        'color_solid': 'black',
        'react_HCl': {
            'gas': None,
            'ion': 'Cu²⁺',
            'solution_color': 'blue-green'
        },
        'dilution_effect': 'blue',
        'KSCN_acetone': None,
        'notes': 'Cu²⁺ 在溶液中呈蓝色'
    }
    analyzer.add_compound('CuO', new_compound)
    
    # 3. 搜索特定条件的化合物
    print("\n🔍 搜索黑色固体:")
    black_solids = analyzer.search_compounds({'color_solid': 'black'})
    print(f"黑色固体化合物: {black_solids}")
    
    print("\n🔍 搜索产生Cl₂气体的化合物:")
    cl2_producers = analyzer.search_compounds({'gas': 'Cl₂'})
    print(f"产生Cl₂的化合物: {cl2_producers}")
    
    # 4. 获取化合物详细信息
    print("\n📋 获取Co2O3的详细信息:")
    co2o3_info = analyzer.get_compound_info('Co2O3')
    if co2o3_info:
        print(f"化学式: {co2o3_info['formula']}")
        print(f"固体颜色: {co2o3_info['color_solid']}")
        print(f"与HCl反应: {co2o3_info['react_HCl']}")
    
    # 5. 保存数据库到文件
    print("\n💾 保存数据库到文件:")
    analyzer.save_compounds_to_file('compounds_database.json')
    analyzer.save_compounds_to_file('compounds_database.csv')
    
    # 6. 解题（原有功能）
    print("\n🧪 化学推断分析:")
    result = solve_chemical_puzzle()
    
    # 打印结果
    print("🔍 化学推断结果")
    print(f"最可能的化合物: {result['predicted_compound']}")
    print(f"化学式: {result['formula']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"解释: {result['explanation']}")
    print("\n各候选得分:")
    for c, s in result['all_scores'].items():
        print(f"  {c}: {s:.2f}")
    
    # 7. 显示所有化合物
    print(f"\n📚 数据库中的所有化合物 ({len(analyzer.list_all_compounds())}):")
    for compound in analyzer.list_all_compounds():
        print(f"  - {compound}")
    
    # 可视化
    print("\n📊 生成可视化图表...")
    fig = visualize_analysis(result)
