#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版反应性分析器演示
展示新增的灵活功能
"""

import json
import csv
from typing import Dict, List, Any, Optional

class ReactivityAnalyzer:
    """改进的反应性分析器 - 支持动态操作和文件导入导出"""
    
    def __init__(self, compounds_file: Optional[str] = None):
        """初始化分析器，可选择从文件加载化合物数据库"""
        # 默认物质库
        self.compounds = {
            'Ni2O3': {
                'formula': 'Ni₂O₃',
                'color_solid': 'black',
                'react_HCl': {
                    'gas': None,
                    'ion': 'Ni²⁺',
                    'solution_color': 'green'
                },
                'dilution_effect': 'stable',
                'KSCN_acetone': None,
                'notes': 'Ni²⁺ 与SCN⁻在丙酮中不显蓝色'
            },
            'Co2O3': {
                'formula': 'Co₂O₃',
                'color_solid': 'black',
                'react_HCl': {
                    'gas': 'Cl₂',
                    'ion': 'Co²⁺',
                    'solution_color': 'blue'
                },
                'dilution_effect': 'pale pink',
                'KSCN_acetone': {
                    'color': 'deep blue',
                    'condition': 'acetone'
                },
                'notes': '符合所有现象'
            },
            'Fe2O3': {
                'formula': 'Fe₂O₃',
                'color_solid': 'red-brown',
                'react_HCl': {
                    'gas': None,
                    'ion': 'Fe³⁺',
                    'solution_color': 'yellow/brown'
                },
                'dilution_effect': 'light yellow',
                'KSCN_acetone': {
                    'color': 'blood red',
                    'condition': 'aqueous'
                },
                'notes': '固体非黑色，溶液非蓝色'
            }
        }
        
        # 如果提供了文件路径，尝试加载
        if compounds_file:
            self.load_compounds_from_file(compounds_file)
    
    def add_compound(self, name: str, compound_data: Dict[str, Any]) -> None:
        """动态添加新化合物"""
        self.compounds[name] = compound_data
        print(f"✅ 已添加化合物: {name}")
    
    def remove_compound(self, name: str) -> bool:
        """移除化合物"""
        if name in self.compounds:
            del self.compounds[name]
            print(f"✅ 已移除化合物: {name}")
            return True
        else:
            print(f"❌ 化合物 {name} 不存在")
            return False
    
    def update_compound(self, name: str, compound_data: Dict[str, Any]) -> bool:
        """更新化合物信息"""
        if name in self.compounds:
            self.compounds[name].update(compound_data)
            print(f"✅ 已更新化合物: {name}")
            return True
        else:
            print(f"❌ 化合物 {name} 不存在，无法更新")
            return False
    
    def search_compounds(self, criteria: Dict[str, Any]) -> List[str]:
        """根据条件搜索化合物"""
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
                elif key == 'ion':
                    if value not in str(data.get('react_HCl', {}).get('ion', '')):
                        match = False
                        break
            
            if match:
                matches.append(name)
        
        return matches
    
    def get_compound_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取化合物详细信息"""
        return self.compounds.get(name)
    
    def list_all_compounds(self) -> List[str]:
        """列出所有化合物名称"""
        return list(self.compounds.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
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
    
    def save_compounds_to_file(self, file_path: str) -> bool:
        """保存化合物数据库到文件"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.compounds, f, ensure_ascii=False, indent=2)
                print(f"✅ 化合物数据库已保存到: {file_path}")
                return True
            else:
                print(f"❌ 不支持的文件格式: {file_path}")
                return False
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            return False
    
    def load_compounds_from_file(self, file_path: str) -> bool:
        """从文件加载化合物数据库"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_compounds = json.load(f)
                self.compounds.update(new_compounds)
                print(f"✅ 从JSON文件加载了 {len(new_compounds)} 个化合物")
                return True
            else:
                print(f"❌ 不支持的文件格式: {file_path}")
                return False
        except Exception as e:
            print(f"❌ 加载文件失败: {e}")
            return False

def main():
    """主函数 - 演示所有新功能"""
    print("🔬 改进的反应性分析器演示")
    print("=" * 50)
    
    # 1. 初始化分析器
    analyzer = ReactivityAnalyzer()
    
    # 2. 显示当前数据库统计
    print("\n📊 当前数据库统计:")
    stats = analyzer.get_statistics()
    print(f"总化合物数: {stats['total_compounds']}")
    print(f"颜色分布: {stats['color_distribution']}")
    print(f"气体产物分布: {stats['gas_distribution']}")
    print(f"离子类型: {stats['unique_ions']}")
    
    # 3. 动态添加新化合物
    print("\n➕ 添加新化合物 CuO:")
    new_compound = {
        'formula': 'CuO',
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
    
    # 4. 搜索特定条件的化合物
    print("\n🔍 搜索黑色固体:")
    black_solids = analyzer.search_compounds({'color_solid': 'black'})
    print(f"黑色固体化合物: {black_solids}")
    
    print("\n🔍 搜索产生Cl₂气体的化合物:")
    cl2_producers = analyzer.search_compounds({'gas': 'Cl₂'})
    print(f"产生Cl₂的化合物: {cl2_producers}")
    
    # 5. 获取化合物详细信息
    print("\n📋 获取Co2O3的详细信息:")
    co2o3_info = analyzer.get_compound_info('Co2O3')
    if co2o3_info:
        print(f"化学式: {co2o3_info['formula']}")
        print(f"固体颜色: {co2o3_info['color_solid']}")
        print(f"与HCl反应: {co2o3_info['react_HCl']}")
    
    # 6. 保存数据库到文件
    print("\n💾 保存数据库到文件:")
    analyzer.save_compounds_to_file('compounds_database.json')
    
    # 7. 显示所有化合物
    print(f"\n📚 数据库中的所有化合物 ({len(analyzer.list_all_compounds())}):")
    for compound in analyzer.list_all_compounds():
        print(f"  - {compound}")
    
    # 8. 演示更新功能
    print("\n🔄 更新化合物信息:")
    analyzer.update_compound('CuO', {'notes': 'Cu²⁺ 在溶液中呈蓝色，可形成配合物'})
    
    # 9. 演示移除功能
    print("\n🗑️ 移除化合物:")
    analyzer.remove_compound('CuO')
    
    print("\n🎉 演示完成！")

if __name__ == "__main__":
    main()
