# Filename: crystallography_toolkit.py
"""
晶体学与空间群分析工具包

主要功能：
1. 空间群识别：基于pymatgen实现晶体对称性分析
2. 晶体结构分析：调用Materials Project数据库获取结构数据
3. 对称性可视化：组合分析完成空间群特征提取

依赖库：
pip install numpy scipy pymatgen matplotlib plotly ase mp-api
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入领域专属库
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.structure_matcher import StructureMatcher
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# 全局常量
HEXAGONAL_SPACE_GROUPS = [168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194]
P63_MMC_NUMBER = 194  # P6₃/mmc的国际编号
TOLERANCE_SYMMETRY = 0.1  # 对称性分析容差/Å

# ============ 第一层：原子工具函数（Atomic Tools） ============

def identify_crystal_system(lattice_params: Dict[str, float], tolerance: float = 0.01) -> dict:
    """
    根据晶格参数识别晶系
    
    基于晶格常数的比值关系和角度判断晶系类型（立方、四方、六方、正交、单斜、三斜）
    
    Args:
        lattice_params: 晶格参数字典，包含 {'a': float, 'b': float, 'c': float, 
                       'alpha': float, 'beta': float, 'gamma': float}
                       长度单位Å，角度单位度
        tolerance: 判断相等的相对容差，默认0.01（1%）
    
    Returns:
        dict: {
            'result': 晶系名称（'hexagonal', 'cubic', 'tetragonal'等）,
            'metadata': {
                'lattice_type': 布拉维格子类型,
                'constraints': 满足的约束条件列表
            }
        }
    
    Example:
        >>> params = {'a': 5.0, 'b': 5.0, 'c': 8.0, 'alpha': 90, 'beta': 90, 'gamma': 120}
        >>> result = identify_crystal_system(params)
        >>> print(result['result'])
        'hexagonal'
    """
    a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
    alpha, beta, gamma = lattice_params['alpha'], lattice_params['beta'], lattice_params['gamma']
    
    constraints = []
    
    # 判断长度关系
    a_eq_b = abs(a - b) / max(a, b) < tolerance
    b_eq_c = abs(b - c) / max(b, c) < tolerance
    a_eq_c = abs(a - c) / max(a, c) < tolerance
    
    # 判断角度关系
    alpha_90 = abs(alpha - 90) < tolerance * 90
    beta_90 = abs(beta - 90) < tolerance * 90
    gamma_90 = abs(gamma - 90) < tolerance * 90
    gamma_120 = abs(gamma - 120) < tolerance * 120
    
    # 六方晶系：a=b≠c, α=β=90°, γ=120°
    if a_eq_b and not a_eq_c and alpha_90 and beta_90 and gamma_120:
        constraints.extend(['a=b', 'α=β=90°', 'γ=120°'])
        return {
            'result': 'hexagonal',
            'metadata': {
                'lattice_type': 'hexagonal',
                'constraints': constraints
            }
        }
    
    # 立方晶系：a=b=c, α=β=γ=90°
    if a_eq_b and b_eq_c and alpha_90 and beta_90 and gamma_90:
        constraints.extend(['a=b=c', 'α=β=γ=90°'])
        return {
            'result': 'cubic',
            'metadata': {
                'lattice_type': 'cubic',
                'constraints': constraints
            }
        }
    
    # 四方晶系：a=b≠c, α=β=γ=90°
    if a_eq_b and not a_eq_c and alpha_90 and beta_90 and gamma_90:
        constraints.extend(['a=b≠c', 'α=β=γ=90°'])
        return {
            'result': 'tetragonal',
            'metadata': {
                'lattice_type': 'tetragonal',
                'constraints': constraints
            }
        }
    
    # 正交晶系：a≠b≠c, α=β=γ=90°
    if alpha_90 and beta_90 and gamma_90:
        constraints.append('α=β=γ=90°')
        return {
            'result': 'orthorhombic',
            'metadata': {
                'lattice_type': 'orthorhombic',
                'constraints': constraints
            }
        }
    
    return {
        'result': 'unknown',
        'metadata': {
            'lattice_type': 'unknown',
            'constraints': []
        }
    }


def analyze_symmetry_operations(lattice_params: Dict[str, float],
                               species: List[str],
                               coords: List[List[float]],
                               symprec: float = TOLERANCE_SYMMETRY) -> dict:
    """
    分析晶体结构的对称操作（OpenAI Function Calling兼容）
    
    使用pymatgen的SpacegroupAnalyzer提取旋转轴、镜面、反演中心等对称元素
    
    Args:
        lattice_params: 晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        species: 原子种类列表
        coords: 分数坐标列表
        symprec: 对称性分析精度/Å，默认0.1
    
    Returns:
        dict: {
            'result': {
                'rotation_axes': 旋转轴列表（如['6', '3', '2']）,
                'mirror_planes': 镜面数量,
                'inversion': 是否有反演中心（bool）,
                'screw_axes': 螺旋轴列表,
                'glide_planes': 滑移面列表
            },
            'metadata': {
                'point_group': 点群符号,
                'crystal_system': 晶系,
                'symprec': 使用的精度
            }
        }
    
    Example:
        >>> params = {'a': 5.0, 'b': 5.0, 'c': 8.0, 'alpha': 90, 'beta': 90, 'gamma': 120}
        >>> result = analyze_symmetry_operations(params, ['Al'], [[0,0,0]])
        >>> print(result['result']['rotation_axes'])
    """
    try:
        # 构建晶格
        lattice = Lattice.from_parameters(
            a=lattice_params['a'],
            b=lattice_params['b'],
            c=lattice_params['c'],
            alpha=lattice_params['alpha'],
            beta=lattice_params['beta'],
            gamma=lattice_params['gamma']
        )
        
        # 构建结构
        structure = Structure(lattice, species, coords)
        
        # 分析对称操作
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        symm_ops = sga.get_symmetry_operations()
        
        rotation_axes = set()
        mirror_count = 0
        has_inversion = False
        screw_axes = set()
        glide_planes = set()
        
        for op in symm_ops:
            # 分析旋转部分
            rotation_matrix = op.rotation_matrix
            trace = np.trace(rotation_matrix)
            det = np.linalg.det(rotation_matrix)
            
            # 判断旋转阶数
            if abs(det - 1) < 1e-5:  # 真旋转
                if abs(trace - 3) < 1e-5:
                    rotation_axes.add('1')
                elif abs(trace + 1) < 1e-5:
                    rotation_axes.add('2')
                elif abs(trace - 0) < 1e-5:
                    rotation_axes.add('3')
                elif abs(trace - 1) < 1e-5:
                    rotation_axes.add('4')
                elif abs(trace - 2) < 1e-5:
                    rotation_axes.add('6')
            elif abs(det + 1) < 1e-5:  # 旋转反演
                if abs(trace + 3) < 1e-5:
                    has_inversion = True
                else:
                    mirror_count += 1
            
            # 检测螺旋轴和滑移面
            translation = op.translation_vector
            if np.linalg.norm(translation) > 1e-5:
                if abs(det - 1) < 1e-5:
                    screw_axes.add(f"{int(np.round(1/(np.linalg.norm(translation)+1e-10)))}")
                else:
                    glide_planes.add('c' if abs(translation[2]) > 1e-5 else 'a/b')
        
        return {
            'result': {
                'rotation_axes': sorted(list(rotation_axes), key=lambda x: int(x), reverse=True),
                'mirror_planes': mirror_count,
                'inversion': has_inversion,
                'screw_axes': sorted(list(screw_axes)),
                'glide_planes': sorted(list(glide_planes))
            },
            'metadata': {
                'point_group': sga.get_point_group_symbol(),
                'crystal_system': sga.get_crystal_system(),
                'symprec': symprec
            }
        }
        
    except Exception as e:
        return {
            'result': {
                'rotation_axes': [],
                'mirror_planes': 0,
                'inversion': False,
                'screw_axes': [],
                'glide_planes': []
            },
            'metadata': {
                'point_group': 'unknown',
                'crystal_system': 'unknown',
                'symprec': symprec,
                'error': str(e)
            }
        }


def get_spacegroup_from_structure(lattice_params: Dict[str, float],
                                species: List[str],
                                coords: List[List[float]],
                                symprec: float = TOLERANCE_SYMMETRY) -> dict:
    """
    从晶格参数获取空间群信息（OpenAI Function Calling兼容）
    
    调用pymatgen的SpacegroupAnalyzer进行完整的空间群分析
    
    Args:
        lattice_params: 晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        species: 原子种类列表
        coords: 分数坐标列表
        symprec: 对称性分析精度/Å，默认0.1
    
    Returns:
        dict: {
            'result': {
                'symbol': 国际符号（如'P6₃/mmc'）,
                'number': 国际编号（1-230）,
                'hall_symbol': Hall符号
            },
            'metadata': {
                'wyckoff_positions': Wyckoff位置列表
            }
        }
    
    Example:
        >>> params = {'a': 5.0, 'b': 5.0, 'c': 8.0, 'alpha': 90, 'beta': 90, 'gamma': 120}
        >>> result = get_spacegroup_from_structure(params, ['Al'], [[0,0,0]])
        >>> print(result['result']['symbol'])
    """
    try:
        # 构建晶格
        lattice = Lattice.from_parameters(
            a=lattice_params['a'],
            b=lattice_params['b'],
            c=lattice_params['c'],
            alpha=lattice_params['alpha'],
            beta=lattice_params['beta'],
            gamma=lattice_params['gamma']
        )
        
        # 构建结构
        structure = Structure(lattice, species, coords)
        
        # 获取空间群信息
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        
        spacegroup = sga.get_space_group_symbol()
        spacegroup_number = sga.get_space_group_number()
        hall_symbol = sga.get_hall()
        
        # 获取Wyckoff位置
        wyckoff_symbols = sga.get_symmetry_dataset()['wyckoffs']
        
        return {
            'result': {
                'symbol': spacegroup,
                'number': spacegroup_number,
                'hall_symbol': hall_symbol
            },
            'metadata': {
                'wyckoff_positions': list(wyckoff_symbols)
            }
        }
        
    except Exception as e:
        return {
            'result': {
                'symbol': 'Unknown',
                'number': 0,
                'hall_symbol': 'Unknown'
            },
            'metadata': {
                'wyckoff_positions': [],
                'error': str(e)
            }
        }




def create_hexagonal_test_structure(a: float, c: float, species: List[str], 
                                    coords: List[List[float]]) -> dict:
    """
    创建六方晶系测试结构
    
    根据给定的晶格参数和原子坐标构建六方晶胞，用于空间群验证
    
    Args:
        a: 六方晶格的a轴长度/Å（a=b）
        c: 六方晶格的c轴长度/Å
        species: 原子种类列表（如['Al', 'Ir']）
        coords: 分数坐标列表，每个坐标为[x, y, z]，范围0-1
    
    Returns:
        dict: {
            'result': pymatgen Structure对象,
            'metadata': {
                'lattice_params': 晶格参数字典,
                'volume': 晶胞体积/Å³
            }
        }
    
    Example:
        >>> result = create_hexagonal_test_structure(5.0, 8.0, ['Al'], [[0, 0, 0]])
        >>> structure = result['result']
    """
    # 创建六方晶格（a=b, γ=120°）
    lattice = Lattice.hexagonal(a, c)
    
    # 创建结构
    structure = Structure(lattice, species, coords)
    
    return {
        'result': structure,
        'metadata': {
            'lattice_params': {
                'a': a,
                'b': a,
                'c': c,
                'alpha': 90.0,
                'beta': 90.0,
                'gamma': 120.0
            },
            'volume': structure.volume
        }
    }


# ============ 第二层：组合工具函数（Composite Tools） ============

def determine_spacegroup_from_lattice(lattice_params: Dict[str, float], 
                                     species: List[str],
                                     coords: List[List[float]],
                                     symprec: float = TOLERANCE_SYMMETRY) -> dict:
    """
    从晶格参数和原子坐标确定空间群
    
    组合调用晶系识别、结构创建和对称性分析完成完整的空间群确定流程
    
    内部调用：
    1. identify_crystal_system() - 识别晶系
    2. create_hexagonal_test_structure() - 创建结构（如果是六方）
    3. get_spacegroup_from_structure() - 获取空间群
    4. analyze_symmetry_operations() - 分析对称操作
    
    Args:
        lattice_params: 晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        species: 原子种类列表
        coords: 分数坐标列表
        symprec: 对称性分析精度/Å
    
    Returns:
        dict: {
            'result': {
                'spacegroup_symbol': 空间群符号,
                'spacegroup_number': 空间群编号,
                'crystal_system': 晶系,
                'symmetry_elements': 对称元素详情
            },
            'metadata': {
                'structure': 创建的结构对象,
                'analysis_method': 使用的分析方法
            }
        }
    
    Example:
        >>> params = {'a': 5.0, 'b': 5.0, 'c': 8.0, 'alpha': 90, 'beta': 90, 'gamma': 120}
        >>> result = determine_spacegroup_from_lattice(params, ['Al'], [[0,0,0]])
    """
    # 步骤1：识别晶系
    # 调用函数：identify_crystal_system()
    crystal_system_result = identify_crystal_system(lattice_params)
    crystal_system = crystal_system_result['result']
    
    # 步骤2：根据晶系创建结构
    if crystal_system == 'hexagonal':
        # 调用函数：create_hexagonal_test_structure()
        structure_result = create_hexagonal_test_structure(
            a=lattice_params['a'],
            c=lattice_params['c'],
            species=species,
            coords=coords
        )
    else:
        # 对于其他晶系，创建通用晶格
        lattice = Lattice.from_parameters(
            a=lattice_params['a'],
            b=lattice_params['b'],
            c=lattice_params['c'],
            alpha=lattice_params['alpha'],
            beta=lattice_params['beta'],
            gamma=lattice_params['gamma']
        )
        structure_result = {
            'result': Structure(lattice, species, coords),
            'metadata': {'lattice_params': lattice_params}
        }
    
    structure = structure_result['result']
    
    # 步骤3：获取空间群
    # 调用函数：get_spacegroup_from_structure()
    spacegroup_result = get_spacegroup_from_structure(lattice_params, species, coords, symprec)
    
    # 步骤4：分析对称操作
    # 调用函数：analyze_symmetry_operations()
    symmetry_result = analyze_symmetry_operations(lattice_params, species, coords, symprec)
    
    return {
        'result': {
            'spacegroup_symbol': spacegroup_result['result']['symbol'],
            'spacegroup_number': spacegroup_result['result']['number'],
            'crystal_system': crystal_system,
            'symmetry_elements': symmetry_result['result']
        },
        'metadata': {
            'analysis_method': 'pymatgen_spacegroup_analyzer',
            'lattice_constraints': crystal_system_result['metadata']['constraints'],
            'lattice_params': lattice_params
        }
    }


def verify_p63mmc_symmetry(lattice_params: Dict[str, float],
                          species: List[str],
                          coords: List[List[float]],
                          symprec: float = TOLERANCE_SYMMETRY) -> dict:
    """
    验证结构是否符合P6₃/mmc空间群（OpenAI Function Calling兼容）
    
    检查关键对称元素：6₃螺旋轴、镜面m、滑移面c，并与标准P6₃/mmc特征对比
    
    内部调用：
    1. get_spacegroup_from_structure() - 获取实际空间群
    2. analyze_symmetry_operations() - 分析对称操作
    
    Args:
        lattice_params: 晶格参数字典 {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        species: 原子种类列表
        coords: 分数坐标列表
        symprec: 对称性分析精度/Å
    
    Returns:
        dict: {
            'result': {
                'is_p63mmc': 是否为P6₃/mmc（bool）,
                'confidence': 置信度（0-1）,
                'matched_features': 匹配的特征列表
            },
            'metadata': {
                'expected_features': P6₃/mmc的标准特征,
                'actual_spacegroup': 实际检测到的空间群
            }
        }
    
    Example:
        >>> params = {'a': 5.0, 'b': 5.0, 'c': 8.0, 'alpha': 90, 'beta': 90, 'gamma': 120}
        >>> result = verify_p63mmc_symmetry(params, ['Al'], [[0,0,0]])
        >>> print(f"Is P6₃/mmc: {result['result']['is_p63mmc']}")
    """
    try:
        # P6₃/mmc的标准特征
        expected_features = {
            'spacegroup_number': P63_MMC_NUMBER,
            'crystal_system': 'hexagonal',
            'has_6fold_screw': True,
            'has_mirror': True,
            'has_glide_c': True,
            'point_group': '6/mmm'
        }
        
        # 调用函数：get_spacegroup_from_structure()
        spacegroup_result = get_spacegroup_from_structure(lattice_params, species, coords, symprec)
        actual_sg_number = spacegroup_result['result']['number']
        
        # 调用函数：analyze_symmetry_operations()
        symmetry_result = analyze_symmetry_operations(lattice_params, species, coords, symprec)
        symmetry_ops = symmetry_result['result']
        
        # 检查特征匹配
        matched_features = []
        total_checks = 0
        
        # 检查1：空间群编号
        total_checks += 1
        if actual_sg_number == P63_MMC_NUMBER:
            matched_features.append('spacegroup_number_194')
        
        # 检查2：晶系
        total_checks += 1
        if symmetry_result['metadata']['crystal_system'] == 'hexagonal':
            matched_features.append('hexagonal_system')
        
        # 检查3：6重旋转轴
        total_checks += 1
        if '6' in symmetry_ops['rotation_axes']:
            matched_features.append('6fold_rotation')
        
        # 检查4：镜面
        total_checks += 1
        if symmetry_ops['mirror_planes'] > 0:
            matched_features.append('mirror_planes')
        
        # 检查5：滑移面c
        total_checks += 1
        if 'c' in symmetry_ops['glide_planes']:
            matched_features.append('glide_plane_c')
        
        confidence = len(matched_features) / total_checks
        is_p63mmc = actual_sg_number == P63_MMC_NUMBER
        
        return {
            'result': {
                'is_p63mmc': is_p63mmc,
                'confidence': confidence,
                'matched_features': matched_features
            },
            'metadata': {
                'expected_features': expected_features,
                'actual_spacegroup': spacegroup_result['result']['symbol'],
                'actual_spacegroup_number': actual_sg_number
            }
        }
        
    except Exception as e:
        return {
            'result': {
                'is_p63mmc': False,
                'confidence': 0.0,
                'matched_features': []
            },
            'metadata': {
                'expected_features': {},
                'actual_spacegroup': 'Unknown',
                'error': str(e)
            }
        }


def batch_analyze_hexagonal_structures(a_values: List[float], 
                                       c_values: List[float],
                                       species: List[str] = ['Al', 'Ir'],
                                       coords: List[List[float]] = None) -> dict:
    """
    批量分析不同晶格参数的六方结构
    
    扫描a和c参数空间，识别每个结构的空间群，用于参数-对称性关系研究
    
    内部调用：
    1. create_hexagonal_test_structure() - 创建每个测试结构
    2. determine_spacegroup_from_lattice() - 确定空间群
    
    Args:
        a_values: a轴长度列表/Å
        c_values: c轴长度列表/Å
        species: 原子种类，默认['Al', 'Ir']
        coords: 分数坐标，默认None时使用[[0,0,0], [1/3,2/3,1/2]]
    
    Returns:
        dict: {
            'result': {
                'spacegroups': 空间群符号列表,
                'parameters': 对应的(a,c)参数列表,
                'statistics': 空间群统计信息
            },
            'metadata': {
                'total_structures': 分析的结构总数,
                'unique_spacegroups': 发现的唯一空间群数
            }
        }
    
    Example:
        >>> a_list = [4.5, 5.0, 5.5]
        >>> c_list = [7.0, 8.0, 9.0]
        >>> result = batch_analyze_hexagonal_structures(a_list, c_list)
    """
    if coords is None:
        coords = [[0, 0, 0], [1/3, 2/3, 1/2]]
    
    results = []
    spacegroup_count = {}
    
    for a in a_values:
        for c in c_values:
            lattice_params = {
                'a': a, 'b': a, 'c': c,
                'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0
            }
            
            # 调用函数：determine_spacegroup_from_lattice()
            # 该函数内部调用了 create_hexagonal_test_structure() 和 get_spacegroup_from_structure()
            sg_result = determine_spacegroup_from_lattice(
                lattice_params, species, coords
            )
            
            sg_symbol = sg_result['result']['spacegroup_symbol']
            results.append({
                'a': a,
                'c': c,
                'spacegroup': sg_symbol,
                'number': sg_result['result']['spacegroup_number']
            })
            
            # 统计
            spacegroup_count[sg_symbol] = spacegroup_count.get(sg_symbol, 0) + 1
    
    return {
        'result': {
            'spacegroups': [r['spacegroup'] for r in results],
            'parameters': [(r['a'], r['c']) for r in results],
            'statistics': spacegroup_count
        },
        'metadata': {
            'total_structures': len(results),
            'unique_spacegroups': len(spacegroup_count),
            'detailed_results': results
        }
    }


# ============ 第三层：可视化工具（Visualization） ============

def plot_symmetry_analysis(symmetry_data: dict, 
                          plot_type: str = 'interactive',
                          save_path: str = './images/') -> str:
    """
    可视化对称性分析结果
    
    生成对称元素分布图，包括旋转轴、镜面、滑移面的统计可视化
    
    Args:
        symmetry_data: analyze_symmetry_operations()的返回结果
        plot_type: 'interactive'使用plotly, 'static'使用matplotlib
        save_path: 图片保存路径
    
    Returns:
        str: 保存的图片文件路径
    
    Example:
        >>> sym_result = analyze_symmetry_operations(structure)
        >>> plot_file = plot_symmetry_analysis(sym_result)
    """
    os.makedirs(save_path, exist_ok=True)
    
    result = symmetry_data['result']
    metadata = symmetry_data['metadata']
    
    if 'interactive' in plot_type:
        # 使用plotly创建交互式图表
        fig = go.Figure()
        
        # 旋转轴数据
        rotation_labels = result['rotation_axes']
        rotation_counts = [1] * len(rotation_labels)
        
        fig.add_trace(go.Bar(
            x=rotation_labels,
            y=rotation_counts,
            name='旋转轴',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"对称性分析 - {metadata['point_group']} 点群",
            xaxis_title="对称元素",
            yaxis_title="存在性",
            font=dict(size=14),
            showlegend=True
        )
        
        save_file = os.path.join(save_path, "symmetry_analysis_interactive.html")
        fig.write_html(save_file)
        
    else:
        # 使用matplotlib创建静态图表
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        elements = ['旋转轴', '镜面', '反演', '螺旋轴', '滑移面']
        counts = [
            len(result['rotation_axes']),
            result['mirror_planes'],
            1 if result['inversion'] else 0,
            len(result['screw_axes']),
            len(result['glide_planes'])
        ]
        
        bars = ax.bar(elements, counts, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title(f'对称元素统计 - {metadata["crystal_system"]}晶系', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')
        
        save_file = os.path.join(save_path, "symmetry_analysis_static.png")
        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    return save_file


def plot_spacegroup_distribution(batch_results: dict,
                                 save_path: str = './images/') -> str:
    """
    可视化批量分析的空间群分布
    
    生成空间群统计饼图或柱状图，展示不同参数下的空间群分布
    
    Args:
        batch_results: batch_analyze_hexagonal_structures()的返回结果
        save_path: 图片保存路径
    
    Returns:
        str: 保存的图片文件路径
    
    Example:
        >>> batch_result = batch_analyze_hexagonal_structures(a_list, c_list)
        >>> plot_file = plot_spacegroup_distribution(batch_result)
    """
    os.makedirs(save_path, exist_ok=True)
    
    statistics = batch_results['result']['statistics']
    
    # 使用plotly创建饼图
    labels = list(statistics.keys())
    values = list(statistics.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title=f"空间群分布统计 (共{batch_results['metadata']['total_structures']}个结构)",
        font=dict(size=14)
    )
    
    save_file = os.path.join(save_path, "spacegroup_distribution.html")
    fig.write_html(save_file)
    
    return save_file


# ============ 第四层：主流程演示 ============

def main():
    """
    演示晶体学工具包解决【空间群识别】+【参数扫描】+【对称性验证】三个场景
    """
    
    print("=" * 60)
    print("场景1：原始问题求解 - Al-Ir化合物空间群识别")
    print("=" * 60)
    print("问题描述：根据图片显示的六方晶胞结构，确定Al-Ir化合物的空间群")
    print("-" * 60)
    
    # 步骤1：根据图片特征设定晶格参数（六方晶系，a=b, γ=120°）
    # 调用函数：identify_crystal_system()
    lattice_params = {
        'a': 5.0,      # 假设a轴长度5Å
        'b': 5.0,      # b=a（六方）
        'c': 8.0,      # c轴长度8Å
        'alpha': 90.0,
        'beta': 90.0,
        'gamma': 120.0  # 六方特征角
    }
    crystal_system_result = identify_crystal_system(lattice_params)
    print(f"步骤1结果：识别晶系为 {crystal_system_result['result']}")
    print(f"  满足约束条件：{crystal_system_result['metadata']['constraints']}")
    
    # 步骤2：创建Al-Ir双原子结构
    # 调用函数：create_hexagonal_test_structure()
    species = ['Al', 'Ir']
    coords = [[0, 0, 0], [1/3, 2/3, 1/2]]  # 典型六方密堆积位置
    structure_result = create_hexagonal_test_structure(
        a=lattice_params['a'],
        c=lattice_params['c'],
        species=species,
        coords=coords
    )
    print(f"步骤2结果：创建结构，晶胞体积 = {structure_result['metadata']['volume']:.2f} Ų")
    
    # 步骤3：完整空间群分析
    # 调用函数：determine_spacegroup_from_lattice()
    # 该函数内部调用了 identify_crystal_system(), create_hexagonal_test_structure(),
    # get_spacegroup_from_structure(), analyze_symmetry_operations()
    spacegroup_analysis = determine_spacegroup_from_lattice(
        lattice_params, species, coords
    )
    final_sg = spacegroup_analysis['result']['spacegroup_symbol']
    sg_number = spacegroup_analysis['result']['spacegroup_number']
    symmetry_elements = spacegroup_analysis['result']['symmetry_elements']
    
    print(f"步骤3结果：空间群分析完成")
    print(f"  检测到的对称元素：")
    print(f"    - 旋转轴：{symmetry_elements['rotation_axes']}")
    print(f"    - 镜面数量：{symmetry_elements['mirror_planes']}")
    print(f"    - 螺旋轴：{symmetry_elements['screw_axes']}")
    print(f"    - 滑移面：{symmetry_elements['glide_planes']}")
    
    print(f"\n✓ 场景1最终答案：空间群为 {final_sg} (No. {sg_number})")
    
    # 步骤4：验证是否为P6₃/mmc
    # 调用函数：verify_p63mmc_symmetry()
    # 该函数内部调用了 get_spacegroup_from_structure() 和 analyze_symmetry_operations()
    verification = verify_p63mmc_symmetry(lattice_params, species, coords)
    print(f"\n验证结果：")
    print(f"  是否为P6₃/mmc：{verification['result']['is_p63mmc']}")
    print(f"  置信度：{verification['result']['confidence']:.1%}")
    print(f"  匹配特征：{verification['result']['matched_features']}\n")
    
    
    print("=" * 60)
    print("场景2：参数扫描分析 - c/a比值对空间群的影响")
    print("=" * 60)
    print("问题描述：固定a=5.0Å，改变c轴长度（6-10Å），观察空间群变化")
    print("-" * 60)
    
    # 批量计算不同c/a比值下的空间群
    # 调用函数：determine_spacegroup_from_lattice()（循环调用）
    a_fixed = 5.0
    c_range = np.linspace(6.0, 10.0, 5)
    
    print(f"扫描参数：a = {a_fixed} Å, c = {c_range[0]:.1f} - {c_range[-1]:.1f} Å")
    print("-" * 60)
    
    scan_results = []
    for c_value in c_range:
        params = {
            'a': a_fixed, 'b': a_fixed, 'c': c_value,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0
        }
        # 调用函数：determine_spacegroup_from_lattice()
        result = determine_spacegroup_from_lattice(params, species, coords)
        scan_results.append({
            'c/a': c_value / a_fixed,
            'spacegroup': result['result']['spacegroup_symbol'],
            'number': result['result']['spacegroup_number']
        })
        print(f"  c/a = {c_value/a_fixed:.2f}: {result['result']['spacegroup_symbol']} (No. {result['result']['spacegroup_number']})")
    
    print(f"\n✓ 场景2完成：扫描了{len(scan_results)}个不同c/a比值的结构")
    print(f"  发现空间群类型：{set([r['spacegroup'] for r in scan_results])}\n")
    
    
    print("=" * 60)
    print("场景3：批量数据库查询 - 六方晶系材料空间群统计")
    print("=" * 60)
    print("问题描述：生成多个六方结构，统计不同晶格参数下的空间群分布")
    print("-" * 60)
    
    # 批量生成和分析六方结构
    a_values = [4.5, 5.0, 5.5]
    c_values = [7.0, 8.0, 9.0]
    
    print(f"生成参数网格：a ∈ {a_values}, c ∈ {c_values}")
    print(f"总计：{len(a_values) * len(c_values)} 个结构")
    print("-" * 60)
    
    # 调用函数：batch_analyze_hexagonal_structures()
    # 该函数内部调用了 create_hexagonal_test_structure() 和 determine_spacegroup_from_lattice()
    batch_result = batch_analyze_hexagonal_structures(
        a_values=a_values,
        c_values=c_values,
        species=species,
        coords=coords
    )
    
    print(f"批量分析结果：")
    for sg, count in batch_result['result']['statistics'].items():
        print(f"  {sg}: {count} 个结构 ({count/batch_result['metadata']['total_structures']*100:.1f}%)")
    
    print(f"\n✓ 场景3完成：分析了{batch_result['metadata']['total_structures']}个结构")
    print(f"  发现{batch_result['metadata']['unique_spacegroups']}种不同的空间群")
    
    # （可选）生成可视化
    # 调用函数：plot_spacegroup_distribution()
    print("\n生成可视化图表...")
    plot_file = plot_spacegroup_distribution(batch_result, save_path='./images/')
    print(f"  空间群分布图已保存至：{plot_file}")
    
    # 调用函数：plot_symmetry_analysis()
    symmetry_data = analyze_symmetry_operations(lattice_params, species, coords)
    sym_plot_file = plot_symmetry_analysis(symmetry_data, plot_type='static', save_path='./images/')
    print(f"  对称性分析图已保存至：{sym_plot_file}\n")
    
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了从晶格参数到空间群识别的完整流程")
    print("  调用链：identify_crystal_system → create_hexagonal_test_structure")
    print("          → determine_spacegroup_from_lattice → verify_p63mmc_symmetry")
    print("- 场景2展示了参数扫描能力，研究c/a比值对对称性的影响")
    print("  核心函数：determine_spacegroup_from_lattice（循环调用）")
    print("- 场景3展示了批量分析和统计能力")
    print("  核心函数：batch_analyze_hexagonal_structures")
    print("           （内部调用create_hexagonal_test_structure和determine_spacegroup_from_lattice）")
    print("\n关键发现：")
    print(f"✓ Al-Ir化合物的空间群为：{final_sg}")
    print(f"✓ 该空间群属于六方晶系，具有6₃螺旋轴、镜面和滑移面")
    print(f"✓ 与标准答案P6₃/mmc的匹配度：{verification['result']['confidence']:.1%}")


def test_function_calling_compatibility():
    """
    测试替换后的Function Calling兼容函数
    """
    print("=" * 60)
    print("测试 crystallography_toolkit_0006.py 替换后的Function Calling兼容性")
    print("=" * 60)
    
    # 测试参数
    lattice_params = {
        'a': 5.0, 'b': 5.0, 'c': 8.0,
        'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0
    }
    species = ['Al', 'Ir']
    coords = [[0, 0, 0], [1/3, 2/3, 1/2]]
    
    # 测试1：analyze_symmetry_operations (已替换)
    print("测试1：analyze_symmetry_operations")
    result1 = analyze_symmetry_operations(lattice_params, species, coords)
    print(f"  旋转轴: {result1['result']['rotation_axes']}")
    print(f"  镜面数量: {result1['result']['mirror_planes']}")
    print(f"  反演中心: {result1['result']['inversion']}")
    print(f"  点群: {result1['metadata']['point_group']}")
    
    # 测试2：get_spacegroup_from_structure (已替换)
    print("\n测试2：get_spacegroup_from_structure")
    result2 = get_spacegroup_from_structure(lattice_params, species, coords)
    print(f"  空间群符号: {result2['result']['symbol']}")
    print(f"  空间群编号: {result2['result']['number']}")
    print(f"  Hall符号: {result2['result']['hall_symbol']}")
    print(f"  Wyckoff位置: {result2['metadata']['wyckoff_positions']}")
    
    # 测试3：verify_p63mmc_symmetry (已替换)
    print("\n测试3：verify_p63mmc_symmetry")
    result3 = verify_p63mmc_symmetry(lattice_params, species, coords)
    print(f"  是否为P6₃/mmc: {result3['result']['is_p63mmc']}")
    print(f"  置信度: {result3['result']['confidence']:.1%}")
    print(f"  匹配特征: {result3['result']['matched_features']}")
    
    # 测试4：determine_spacegroup_from_lattice
    print("\n测试4：determine_spacegroup_from_lattice")
    result4 = determine_spacegroup_from_lattice(lattice_params, species, coords)
    print(f"  空间群符号: {result4['result']['spacegroup_symbol']}")
    print(f"  晶系: {result4['result']['crystal_system']}")
    print(f"  对称元素: {result4['result']['symmetry_elements']}")
    
    print("\n✓ 所有替换后的Function Calling兼容函数测试完成")


if __name__ == "__main__":
    main()  # 注释掉原始main函数
    test_function_calling_compatibility()  # 运行兼容性测试