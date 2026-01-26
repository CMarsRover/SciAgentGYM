from rdkit import Chem 
from rdkit.Chem import Descriptors
try:
    import cirpy  # type: ignore
except Exception:
    cirpy = None
try:
    import pubchempy as pcp  # type: ignore
except Exception:
    pcp = None
import re
from collections import defaultdict

_PREDEFINED_NAME_TO_SMILES = {
    # common polycyclic aromatics that frequently appear in benchmark questions
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc3ccccc23",
    "acenaphthylene": "c1ccc2c(c1)C=Cc3ccccc23",
    "fluorene": "c1ccc2c(c1)Cc3ccccc23",
}

def chemical_name_to_molecular_formula(chemical_name:str):
    """
    化学名称转SMILES字符串工具
    
    基于CIR（Chemical Identifier Resolver）在线服务，将化学物质的常用名称、
    IUPAC名称或其他标识符转换为SMILES分子表示形式。
    
    Parameters:
    -----------
    chemical_name : str
        化学物质的名称，支持：
        - 常用名：如 'aspirin', 'ethanol', 'caffeine'
        - IUPAC名称：如 '2-acetoxybenzoic acid'
        - CAS号等其他标识符
        
    Returns:
    --------
    str or None
        SMILES字符串表示，如果无法解析则返回None
        
    Examples:
    ---------
    >>> # 使用常用名
    >>> chemical_name_to_molecular_formula('aspirin')
    'CC(=O)Oc1ccccc1C(=O)O'
    
    >>> # 使用IUPAC名称
    >>> chemical_name_to_molecular_formula('ethanol')
    'CCO'
    
    >>> # 无效名称返回None
    >>> chemical_name_to_molecular_formula('unknown_chemical')
    None
    
    Note:
    -----
    此函数需要网络连接访问CIR在线服务
    """
    if not chemical_name:
        return None

    query = chemical_name.strip()
    if not query:
        return None

    # Already a valid SMILES, return canonical form
    try:
        mol = Chem.MolFromSmiles(query)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass

    # # Quick lookup table for well-known molecules
    # preset = _PREDEFINED_NAME_TO_SMILES.get(query.lower())
    # if preset:
    #     return preset

    # Use PubChem if available
    if pcp is not None:
        try:
            compounds = pcp.get_compounds(query, 'name')
            # 确保 compounds 不为 None 且是列表
            if compounds:
                for compound in compounds:
                    # 尝试获取 canonical_smiles，如果不存在则尝试 isomeric_smiles
                    smiles = getattr(compound, "canonical_smiles", None)
                    if not smiles:
                        smiles = getattr(compound, "isomeric_smiles", None)
                    if smiles:
                        return smiles
        except Exception as e:
            # 静默处理错误，继续尝试其他方法
            pass

    # Fall back to CIR if installed
    if cirpy is not None:
        try:
            smiles = cirpy.resolve(query, 'smiles')
            if smiles:
                return smiles
        except Exception:
            pass

    return None 

    
def smiles_to_formula(smiles: str) -> dict:
    """
    从SMILES获取化学式
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        包含化学式信息的字典，格式如：
        {
            'smiles': 'COCC',
            'canonical_smiles': 'CCOC',
            'molecular_formula': 'C3H8O',
            'molecular_weight': 60.096,
            'formal_charge': 0,
            'is_charged': False,
            'valid': True,
            'warning': None
        }
    """
    result = {
        'smiles': smiles,
        'canonical_smiles': None,
        'molecular_formula': None,
        'molecular_weight': None,
        'formal_charge': None,
        'is_charged': False,
        'valid': False,
        'warning': None
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            result['valid'] = True
            result['canonical_smiles'] = Chem.MolToSmiles(mol, canonical=True)
            result['molecular_formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
            result['molecular_weight'] = round(Descriptors.MolWt(mol), 3)
            
            # 计算总电荷
            formal_charge = Chem.GetFormalCharge(mol)
            result['formal_charge'] = formal_charge
            result['is_charged'] = (formal_charge != 0)
            
            print(f"SMILES: {smiles}")
            print(f"标准SMILES: {result['canonical_smiles']}")
            print(f"化学式: {result['molecular_formula']}")
            print(f"分子量: {result['molecular_weight']}")
            
            # 如果是带电分子，给出警告
            if result['is_charged']:
                charge_str = f"+{formal_charge}" if formal_charge > 0 else str(formal_charge)
                result['warning'] = f"这是带电分子（总电荷: {charge_str}），可能无法在PubChem中查询"
                print(f"⚠️  {result['warning']}")
        else:
            print(f"⚠️  无效的SMILES: {smiles}")
    except Exception as e:
        print(f"❌ 解析SMILES时出错: {str(e)}")
        result['error'] = str(e)
    
    return result
def parse_molecular_formula(formula: str) -> dict:
    """
    解析分子式，提取各元素的个数
    
    将化学分子式（如C9H11FN2O5）解析为各元素及其数量的字典。
    支持标准的分子式格式，包括元素符号后的数字。
    
    Parameters:
    -----------
    formula : str
        分子式字符串
        例如: 'C9H11FN2O5', 'C6H12O6', 'H2O', 'CH4'
        
    Returns:
    --------
    dict
        元素及其数量的字典，格式如：
        {
            'element_composition': {'C': 9, 'H': 11, 'F': 1, 'N': 2, 'O': 5},
            'total_atoms': 28,
            'num_elements': 5,
            'formula': 'C9H11FN2O5'
        }
        
    Examples:
    ---------
    >>> # 解析水分子
    >>> result = parse_molecular_formula('H2O')
    >>> print(result['element_composition'])
    {'H': 2, 'O': 1}
    
    >>> # 解析葡萄糖
    >>> result = parse_molecular_formula('C6H12O6')
    >>> print(result['element_composition'])
    {'C': 6, 'H': 12, 'O': 6}
    >>> print(f"总原子数: {result['total_atoms']}")
    总原子数: 24
    
    >>> # 解析复杂分子
    >>> result = parse_molecular_formula('C9H11FN2O5')
    >>> for element, count in result['element_composition'].items():
    ...     print(f"{element}: {count}")
    C: 9
    H: 11
    F: 1
    N: 2
    O: 5
    
    Note:
    -----
    - 支持标准化学分子式格式
    - 元素符号后的数字表示该元素的个数
    - 没有数字的元素默认个数为1
    - 返回的字典按元素符号排序
    """
    # 移除空格
    formula = formula.strip()
    
    # 使用正则表达式匹配元素和数量
    # 匹配模式: 大写字母 + 可选的小写字母 + 可选的数字
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    element_dict = {}
    
    for element, count in matches:
        if element:  # 确保元素不为空
            # 如果没有数字，默认为1
            count = int(count) if count else 1
            # 累加（处理可能的重复元素）
            if element in element_dict:
                element_dict[element] += count
            else:
                element_dict[element] = count
    
    # 计算总原子数
    total_atoms = sum(element_dict.values())
    
    # 按元素符号排序（C优先，然后H，然后其他按字母顺序）
    element_order = []
    if 'C' in element_dict:
        element_order.append('C')
    if 'H' in element_dict:
        element_order.append('H')
    
    # 添加其他元素（按字母顺序）
    other_elements = sorted([e for e in element_dict.keys() if e not in ['C', 'H']])
    element_order.extend(other_elements)
    
    # 创建有序字典
    sorted_element_dict = {e: element_dict[e] for e in element_order}
    
    return {
        'element_composition': sorted_element_dict,
        'total_atoms': total_atoms,
        'num_elements': len(sorted_element_dict),
        'formula': formula
    }


def smiles_to_element_composition(smiles: str) -> dict:
    """
    从SMILES获取元素组成信息
    
    将SMILES字符串转换为分子式，然后解析出各元素的个数。
    这是一个集成函数，组合了SMILES→分子式和分子式→元素组成两步。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES字符串
        例如: 'CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O'
        
    Returns:
    --------
    dict
        包含元素组成和分子信息的完整字典：
        {
            'smiles': str,
            'canonical_smiles': str,
            'molecular_formula': str,
            'molecular_weight': float,
            'element_composition': dict,  # {'C': n, 'H': m, ...}
            'total_atoms': int,
            'num_elements': int,
            'valid': bool
        }
        
    Examples:
    ---------
    >>> # 分析乙醇
    >>> result = smiles_to_element_composition('CCO')
    >>> print(f"分子式: {result['molecular_formula']}")
    分子式: C2H6O
    >>> print(f"元素组成: {result['element_composition']}")
    元素组成: {'C': 2, 'H': 6, 'O': 1}
    
    >>> # 分析阿司匹林
    >>> result = smiles_to_element_composition('CC(=O)Oc1ccccc1C(=O)O')
    >>> print(f"分子式: {result['molecular_formula']}")
    >>> print("元素组成:")
    >>> for element, count in result['element_composition'].items():
    ...     print(f"  {element}: {count}")
    
    >>> # 快速查看分子组成
    >>> result = smiles_to_element_composition('CN1C=NC2=C1C(=O)N(C(=O)N2C)C')
    >>> print(f"咖啡因含有{result['num_elements']}种元素")
    >>> print(f"总共{result['total_atoms']}个原子")
    
    Note:
    -----
    - 自动处理无效的SMILES（返回valid=False）
    - 元素按C、H优先，其他按字母顺序排列
    - 包含完整的分子信息（分子量、规范SMILES等）
    """
    result = {
        'smiles': smiles,
        'canonical_smiles': None,
        'molecular_formula': None,
        'molecular_weight': None,
        'element_composition': None,
        'total_atoms': None,
        'num_elements': None,
        'valid': False,
        'error': None
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            result['valid'] = True
            result['canonical_smiles'] = Chem.MolToSmiles(mol, canonical=True)
            
            # 获取分子式
            molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            result['molecular_formula'] = molecular_formula
            result['molecular_weight'] = round(Descriptors.MolWt(mol), 2)
            
            # 解析元素组成
            composition_data = parse_molecular_formula(molecular_formula)
            result['element_composition'] = composition_data['element_composition']
            result['total_atoms'] = composition_data['total_atoms']
            result['num_elements'] = composition_data['num_elements']
            
            # 打印信息
            print(f"SMILES: {smiles}")
            print(f"标准SMILES: {result['canonical_smiles']}")
            print(f"分子式: {result['molecular_formula']}")
            print(f"分子量: {result['molecular_weight']}")
            print(f"元素组成:")
            for element, count in result['element_composition'].items():
                print(f"  {element}: {count}")
            print(f"总原子数: {result['total_atoms']}")
            print(f"元素种类: {result['num_elements']}")
        else:
            result['error'] = "无效的SMILES字符串"
            print(f"⚠️  无效的SMILES: {smiles}")
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ 解析SMILES时出错: {str(e)}")
    
    return result


if __name__ == "__main__":

    
    # 测试2: SMILES转化学式
    print("=== 测试2: SMILES转化学式 ===")
    
    # 测试乙基甲基醚（CH₃CH₂OCH₃）
    print("\n--- 乙基甲基醚 ---")
    result1 = smiles_to_formula("COCC")  # 或者 CCOC
    

    
    print("\n--- 乙酸 ---")
    result3 = smiles_to_formula("CCOCC(=O)O")
    
    print("\n--- 苯 ---")
    result4 = smiles_to_formula("[CH2-]CCCC")
    
    print("\n--- 葡萄糖 ---")
    result5 = smiles_to_formula("C=[C+]C")
    
    # 新功能测试
    print("\n" + "="*70)
    print("=== 测试3: 元素组成分析 ===")
    print("="*70)
    
    # 测试解析分子式
    print("\n--- 测试1: 解析分子式 ---")
    formula_result = parse_molecular_formula('C9H11FN2O5')
    print(f"分子式: {formula_result['formula']}")
    print(f"元素组成: {formula_result['element_composition']}")
    print(f"总原子数: {formula_result['total_atoms']}")
    
    # 测试从SMILES获取元素组成
    print("\n--- 测试2: 从SMILES获取元素组成 ---")
    
    test_molecules = {
        '乙醇': 'CCO',
        '阿司匹林': 'CC(=O)Oc1ccccc1C(=O)O',
        '咖啡因': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        '5-氟脱氧尿苷': 'OC[C@H]1O[C@H](C[C@@H]1O)N2C=C(F)C(=O)NC2=O'
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n{name}:")
        result = smiles_to_element_composition(smiles)
        if result['valid']:
            print(f"  分子式: {result['molecular_formula']}")
            print(f"  元素: {', '.join([f'{e}{n}' for e, n in result['element_composition'].items()])}")
            print(f"  总原子: {result['total_atoms']}")
        print("-" * 60)
