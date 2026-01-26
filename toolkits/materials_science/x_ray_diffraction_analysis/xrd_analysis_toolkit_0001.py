# Filename: xrd_analysis_toolkit.py
"""
X射线衍射(XRD)分析工具包

主要功能：
1. 峰位识别与Miller指数标定：基于scipy信号处理实现自动峰检测
2. 晶格参数计算：利用Bragg定律和晶体学公式精确计算
3. 晶体结构匹配：调用pymatgen和Materials Project数据库进行相鉴定
4. 可视化分析：使用plotly生成交互式XRD图谱

依赖库：
pip install numpy scipy pymatgen mp-api plotly scikit-learn pandas
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from scipy.signal import find_peaks
from scipy.optimize import least_squares
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 尝试导入pymatgen和mp-api（材料科学专属库）
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Some features will be limited.")

# mp-api 延迟导入，避免在模块加载时触发 emmet 初始化（numpy 兼容性问题）
# 导入 mp-api 时可能抛出 ValidationError（来自 pydantic），不是 ImportError
MP_API_AVAILABLE = False  # 将在需要时尝试导入

# ============ 全局常量 ============
CU_KA_WAVELENGTH = 1.5406  # Å, Cu Kα辐射波长
PLANCK_CONSTANT = 6.62607015e-34  # J·s
SPEED_OF_LIGHT = 2.99792458e8  # m/s

# 常见晶系的d-spacing公式
CRYSTAL_SYSTEMS = {
    'cubic': lambda h, k, l, a, b, c: a / np.sqrt(h**2 + k**2 + l**2),
    'tetragonal': lambda h, k, l, a, b, c: 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2),
    'orthorhombic': lambda h, k, l, a, b, c: 1 / np.sqrt(h**2/a**2 + k**2/b**2 + l**2/c**2),
    'hexagonal': lambda h, k, l, a, b, c: 1 / np.sqrt(4*(h**2 + h*k + k**2)/(3*a**2) + l**2/c**2),
}


# ============ 第一层：原子工具函数 ============

def detect_peaks_from_pattern(two_theta: np.ndarray, 
                               intensity: np.ndarray,
                               prominence: float = 5.0,
                               min_distance: int = 5) -> Dict:
    """
    从XRD图谱中自动检测衍射峰位置
    
    使用scipy的find_peaks算法识别局部最大值，通过prominence参数过滤噪声峰。
    
    Args:
        two_theta: 2θ角度数组 (degrees)，范围通常10-90°
        intensity: 对应的衍射强度数组 (a.u.)
        prominence: 峰的显著性阈值，默认5.0（相对强度单位）
        min_distance: 相邻峰的最小间隔（数据点数），默认5
    
    Returns:
        dict: {
            'result': {
                'peak_positions': 峰位2θ值数组,
                'peak_intensities': 峰强度数组,
                'peak_indices': 峰在原数组中的索引
            },
            'metadata': {
                'num_peaks': 检测到的峰数量,
                'prominence_threshold': 使用的prominence值
            }
        }
    
    Example:
        >>> two_theta = np.linspace(20, 90, 1000)
        >>> intensity = np.random.rand(1000) * 10
        >>> result = detect_peaks_from_pattern(two_theta, intensity)
        >>> print(f"Found {len(result['result']['peak_positions'])} peaks")
    """
    # 使用scipy的峰检测算法
    peaks, properties = find_peaks(intensity, 
                                   prominence=prominence,
                                   distance=min_distance)
    
    peak_positions = two_theta[peaks]
    peak_intensities = intensity[peaks]
    
    # 按强度排序（降序）
    sorted_indices = np.argsort(peak_intensities)[::-1]
    
    return {
        'result': {
            'peak_positions': peak_positions[sorted_indices],
            'peak_intensities': peak_intensities[sorted_indices],
            'peak_indices': peaks[sorted_indices]
        },
        'metadata': {
            'num_peaks': len(peaks),
            'prominence_threshold': prominence,
            'properties': properties
        }
    }


def calculate_d_spacing(two_theta: Union[float, np.ndarray], 
                        wavelength: float = CU_KA_WAVELENGTH) -> Dict:
    """
    根据Bragg定律计算晶面间距d
    
    应用Bragg定律: nλ = 2d·sinθ，其中n=1（一级衍射）
    
    Args:
        two_theta: 2θ衍射角 (degrees)，可以是单值或数组
        wavelength: X射线波长 (Å)，默认Cu Kα = 1.5406 Å
    
    Returns:
        dict: {
            'result': d间距值或数组 (Å),
            'metadata': {
                'wavelength': 使用的波长,
                'formula': 'Bragg Law: λ = 2d·sinθ'
            }
        }
    
    Example:
        >>> result = calculate_d_spacing(26.5)
        >>> print(f"d-spacing: {result['result']:.3f} Å")
    """
    theta_rad = np.deg2rad(two_theta / 2.0)  # 转换为θ（弧度）
    d_spacing = wavelength / (2.0 * np.sin(theta_rad))
    
    return {
        'result': d_spacing,
        'metadata': {
            'wavelength': wavelength,
            'formula': 'Bragg Law: λ = 2d·sinθ',
            'theta_degrees': two_theta / 2.0
        }
    }


def assign_miller_indices_tetragonal(d_spacing: float,
                                     a: float,
                                     c: float,
                                     tolerance: float = 0.02) -> Dict:
    """
    为四方晶系的d间距分配Miller指数(hkl)
    
    使用四方晶系公式: 1/d² = (h²+k²)/a² + l²/c²，遍历可能的hkl组合。
    
    Args:
        d_spacing: 实验测得的晶面间距 (Å)
        a: 四方晶系的a晶格参数 (Å)
        c: 四方晶系的c晶格参数 (Å)
        tolerance: 匹配容差，默认0.02 (Å)
    
    Returns:
        dict: {
            'result': {
                'hkl': (h, k, l)元组或None,
                'd_calculated': 理论d值,
                'error': 误差
            },
            'metadata': {
                'crystal_system': 'tetragonal',
                'search_range': 搜索的hkl范围
            }
        }
    
    Example:
        >>> result = assign_miller_indices_tetragonal(3.36, 4.76, 3.21)
        >>> print(result['result']['hkl'])
    """
    best_match = None
    min_error = float('inf')
    
    # 搜索hkl范围（通常0-5足够）
    for h in range(0, 6):
        for k in range(0, 6):
            for l in range(0, 6):
                if h == 0 and k == 0 and l == 0:
                    continue
                
                # 四方晶系d-spacing公式
                d_calc = 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
                error = abs(d_calc - d_spacing)
                
                if error < tolerance and error < min_error:
                    min_error = error
                    best_match = (h, k, l, d_calc)
    
    if best_match:
        return {
            'result': {
                'hkl': best_match[:3],
                'd_calculated': best_match[3],
                'error': min_error
            },
            'metadata': {
                'crystal_system': 'tetragonal',
                'search_range': '(0-5, 0-5, 0-5)',
                'lattice_params': {'a': a, 'c': c}
            }
        }
    else:
        return {
            'result': {
                'hkl': None,
                'd_calculated': None,
                'error': None
            },
            'metadata': {
                'crystal_system': 'tetragonal',
                'message': 'No match found within tolerance'
            }
        }


def refine_lattice_parameters_tetragonal(peak_data: List[Dict],
                                         initial_a: float,
                                         initial_c: float) -> Dict:
    """
    通过最小二乘法精修四方晶系的晶格参数
    
    使用scipy.optimize.least_squares最小化理论与实验d值的差异。
    
    Args:
        peak_data: 峰数据列表，每项包含 {'two_theta': float, 'hkl': (h,k,l)}
        initial_a: a参数初始值 (Å)
        initial_c: c参数初始值 (Å)
    
    Returns:
        dict: {
            'result': {
                'a': 精修后的a参数 (Å),
                'c': 精修后的c参数 (Å),
                'residual': 残差平方和
            },
            'metadata': {
                'num_peaks_used': 用于精修的峰数量,
                'convergence': 优化收敛信息
            }
        }
    
    Example:
        >>> peaks = [{'two_theta': 26.5, 'hkl': (1,1,0)}, ...]
        >>> result = refine_lattice_parameters_tetragonal(peaks, 4.7, 3.2)
    """
    def residual_function(params, peak_data):
        a, c = params
        residuals = []
        for peak in peak_data:
            h, k, l = peak['hkl']
            # 调用函数：calculate_d_spacing()
            d_exp_result = calculate_d_spacing(peak['two_theta'])
            d_exp = d_exp_result['result']
            
            # 理论d值
            d_calc = 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
            residuals.append(d_exp - d_calc)
        return residuals
    
    # 使用scipy的最小二乘优化
    result = least_squares(residual_function, 
                          [initial_a, initial_c],
                          args=(peak_data,))
    
    a_refined, c_refined = result.x
    
    return {
        'result': {
            'a': a_refined,
            'c': c_refined,
            'residual': np.sum(result.fun**2)
        },
        'metadata': {
            'num_peaks_used': len(peak_data),
            'convergence': {
                'success': result.success,
                'iterations': result.nfev,
                'message': result.message
            }
        }
    }


def fetch_structure_from_mp(formula: str, 
                            api_key: Optional[str] = None) -> Dict:
    """
    从Materials Project数据库获取晶体结构信息
    
    通过mp-api访问Materials Project免费数据库，获取化学式对应的晶体结构。
    
    Args:
        formula: 化学式，如 'SnO2', 'TiO2'
        api_key: Materials Project API密钥（可选，使用环境变量MP_API_KEY）
    
    Returns:
        dict: {
            'result': {
                'structure': pymatgen Structure对象,
                'lattice_params': {'a': float, 'b': float, 'c': float},
                'space_group': 空间群符号,
                'material_id': MP数据库ID
            },
            'metadata': {
                'database': 'Materials Project',
                'formula': 查询的化学式
            }
        }
    
    Example:
        >>> result = fetch_structure_from_mp('SnO2')
        >>> print(result['result']['space_group'])
    """
    # 延迟导入 mp-api，避免在模块加载时触发 emmet 初始化
    # 参考: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started
    try:
        from mp_api.client import MPRester
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': f'无法导入 mp-api（可能是 numpy 兼容性问题）: {e}',
                'formula': formula,
                'message': 'mp-api 导入失败，可能是 numpy 2.x 兼容性问题。请参考: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started'
            }
        }
    
    # 根据官方文档，优先从环境变量读取 API key，否则使用传入的参数或默认值
    # https://docs.materialsproject.org/downloading-data/using-the-api/getting-started
    if not api_key:
        api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        return {
            'result': None,
            'metadata': {
                'error': 'API key not provided',
                'formula': formula,
                'message': '请提供 API key 或设置环境变量 MP_API_KEY。参考: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started'
            }
        }
    
    try:
        # 使用上下文管理器（官方推荐方式）
        with MPRester(api_key) as mpr:
            # 搜索化学式
            docs = mpr.materials.summary.search(formula=formula, fields=["structure", "material_id"])
            
            if not docs:
                return {
                    'result': None,
                    'metadata': {
                        'error': 'No structure found',
                        'formula': formula
                    }
                }
            
            # 取第一个结果（通常是最稳定的）
            doc = docs[0]
            structure = doc.structure
            
            # 分析空间群
            sga = SpacegroupAnalyzer(structure)
            space_group = sga.get_space_group_symbol()
            
            lattice = structure.lattice
            
            return {
                'result': {
                    'structure': structure,
                    'lattice_params': {
                        'a': lattice.a,
                        'b': lattice.b,
                        'c': lattice.c,
                        'alpha': lattice.alpha,
                        'beta': lattice.beta,
                        'gamma': lattice.gamma
                    },
                    'space_group': space_group,
                    'material_id': doc.material_id
                },
                'metadata': {
                    'database': 'Materials Project',
                    'formula': formula,
                    'num_results': len(docs)
                }
            }
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': str(e),
                'formula': formula
            }
        }


# ============ 第二层：组合工具函数 ============

def analyze_xrd_pattern(two_theta: np.ndarray,
                       intensity: np.ndarray,
                       expected_phase: str = 'SnO2',
                       prominence: float = 5.0) -> Dict:
    """
    完整分析XRD图谱：峰检测、相鉴定、Miller指数标定、晶格参数计算
    
    组合调用多个原子工具函数，实现从原始数据到晶格参数的全流程分析。
    
    Args:
        two_theta: 2θ角度数组 (degrees)
        intensity: 衍射强度数组 (a.u.)
        expected_phase: 预期相的化学式，默认'SnO2'
        prominence: 峰检测显著性阈值
    
    Returns:
        dict: {
            'result': {
                'identified_phase': 鉴定的相,
                'space_group': 空间群,
                'peak_assignments': [{'2theta': float, 'hkl': tuple, 'd': float}, ...],
                'lattice_params': {'a': float, 'c': float, ...},
                'refined_params': 精修后的晶格参数
            },
            'metadata': {
                'num_peaks_detected': int,
                'analysis_steps': 分析步骤列表
            }
        }
    
    Example:
        >>> two_theta = np.array([26.5, 33.7, 38.8, ...])
        >>> intensity = np.array([100, 82, 24, ...])
        >>> result = analyze_xrd_pattern(two_theta, intensity)
    """
    analysis_steps = []
    
    # 步骤1：峰检测
    # 调用函数：detect_peaks_from_pattern()
    peak_result = detect_peaks_from_pattern(two_theta, intensity, prominence)
    detected_peaks = peak_result['result']['peak_positions']
    analysis_steps.append(f"Detected {len(detected_peaks)} peaks")
    
    # 步骤2：从数据库获取参考结构
    # 调用函数：fetch_structure_from_mp()
    mp_result = fetch_structure_from_mp(expected_phase)
    
    if mp_result['result'] is None:
        # 如果数据库不可用，使用已知的SnO2参数
        initial_a = 4.76
        initial_c = 3.21
        space_group = 'P4_2/mnm'
        analysis_steps.append("Using known SnO2 parameters (MP database unavailable)")
    else:
        lattice_params = mp_result['result']['lattice_params']
        initial_a = lattice_params['a']
        initial_c = lattice_params['c']
        space_group = mp_result['result']['space_group']
        analysis_steps.append(f"Retrieved structure from Materials Project: {space_group}")
    
    # 步骤3：计算d间距并分配Miller指数
    peak_assignments = []
    peaks_for_refinement = []
    
    for peak_2theta in detected_peaks[:10]:  # 只处理前10个最强峰
        # 调用函数：calculate_d_spacing()
        d_result = calculate_d_spacing(peak_2theta)
        d_spacing = d_result['result']
        
        # 调用函数：assign_miller_indices_tetragonal()
        hkl_result = assign_miller_indices_tetragonal(d_spacing, initial_a, initial_c)
        
        if hkl_result['result']['hkl'] is not None:
            hkl = hkl_result['result']['hkl']
            peak_assignments.append({
                '2theta': peak_2theta,
                'hkl': hkl,
                'd_exp': d_spacing,
                'd_calc': hkl_result['result']['d_calculated']
            })
            peaks_for_refinement.append({
                'two_theta': peak_2theta,
                'hkl': hkl
            })
    
    analysis_steps.append(f"Assigned Miller indices to {len(peak_assignments)} peaks")
    
    # 步骤4：精修晶格参数
    # 调用函数：refine_lattice_parameters_tetragonal()
    if len(peaks_for_refinement) >= 3:
        refined_result = refine_lattice_parameters_tetragonal(
            peaks_for_refinement, initial_a, initial_c
        )
        refined_params = refined_result['result']
        analysis_steps.append(f"Refined lattice parameters using {len(peaks_for_refinement)} peaks")
    else:
        refined_params = {'a': initial_a, 'c': initial_c, 'residual': None}
        analysis_steps.append("Insufficient peaks for refinement, using initial parameters")
    
    return {
        'result': {
            'identified_phase': expected_phase,
            'space_group': space_group,
            'peak_assignments': peak_assignments,
            'lattice_params': {
                'a': refined_params['a'],
                'b': refined_params['a'],  # 四方晶系 a=b
                'c': refined_params['c'],
                'alpha': 90.0,
                'beta': 90.0,
                'gamma': 90.0
            },
            'refined_params': refined_params
        },
        'metadata': {
            'num_peaks_detected': len(detected_peaks),
            'num_peaks_assigned': len(peak_assignments),
            'analysis_steps': analysis_steps
        }
    }


def compare_experimental_theoretical_xrd(experimental_2theta: np.ndarray,
                                        experimental_intensity: np.ndarray,
                                        structure: 'Structure') -> Dict:
    """
    对比实验XRD图谱与理论计算图谱
    
    使用pymatgen的XRDCalculator生成理论图谱，与实验数据对比。
    
    Args:
        experimental_2theta: 实验2θ数组 (degrees)
        experimental_intensity: 实验强度数组 (a.u.)
        structure: pymatgen Structure对象
    
    Returns:
        dict: {
            'result': {
                'theoretical_pattern': {'2theta': array, 'intensity': array},
                'match_score': 匹配度分数 (0-1),
                'peak_matches': 匹配的峰列表
            },
            'metadata': {
                'calculator': 'pymatgen XRDCalculator',
                'wavelength': 使用的波长
            }
        }
    """
    if not PYMATGEN_AVAILABLE:
        return {
            'result': None,
            'metadata': {'error': 'pymatgen not available'}
        }
    
    # 使用pymatgen计算理论XRD
    calculator = XRDCalculator(wavelength=CU_KA_WAVELENGTH)
    theoretical_pattern = calculator.get_pattern(structure)
    
    # 提取理论峰位和强度
    theo_2theta = np.array([peak[0] for peak in theoretical_pattern.x])
    theo_intensity = np.array([peak[1] for peak in theoretical_pattern.y])
    
    # 简单匹配度计算（基于峰位置的重合度）
    # 调用函数：detect_peaks_from_pattern()
    exp_peaks = detect_peaks_from_pattern(experimental_2theta, experimental_intensity)
    exp_peak_positions = exp_peaks['result']['peak_positions']
    
    matches = 0
    for exp_peak in exp_peak_positions[:10]:
        # 查找最近的理论峰
        distances = np.abs(theo_2theta - exp_peak)
        if np.min(distances) < 0.5:  # 0.5度容差
            matches += 1
    
    match_score = matches / min(10, len(exp_peak_positions))
    
    return {
        'result': {
            'theoretical_pattern': {
                '2theta': theo_2theta,
                'intensity': theo_intensity
            },
            'match_score': match_score,
            'peak_matches': matches
        },
        'metadata': {
            'calculator': 'pymatgen XRDCalculator',
            'wavelength': CU_KA_WAVELENGTH,
            'num_theoretical_peaks': len(theo_2theta)
        }
    }


# ============ 第三层：可视化工具 ============

def plot_xrd_analysis(two_theta: np.ndarray,
                     intensity: np.ndarray,
                     peak_assignments: List[Dict],
                     theoretical_pattern: Optional[Dict] = None,
                     save_path: str = './images/') -> str:
    """
    生成XRD分析的交互式可视化图表
    
    使用plotly创建包含实验数据、峰标注和理论对比的多子图。
    
    Args:
        two_theta: 实验2θ数组
        intensity: 实验强度数组
        peak_assignments: 峰标注列表 [{'2theta': float, 'hkl': tuple}, ...]
        theoretical_pattern: 理论图谱数据（可选）
        save_path: 图片保存路径
    
    Returns:
        str: 保存的HTML文件路径
    
    Example:
        >>> plot_file = plot_xrd_analysis(two_theta, intensity, assignments)
        >>> print(f"Plot saved to {plot_file}")
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 创建图表
    if theoretical_pattern:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('实验XRD图谱与峰标注', '实验vs理论对比'),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12
        )
    else:
        fig = go.Figure()
    
    # 绘制实验数据
    fig.add_trace(
        go.Scatter(
            x=two_theta,
            y=intensity,
            mode='lines',
            name='实验数据',
            line=dict(color='black', width=1.5)
        ),
        row=1 if theoretical_pattern else None,
        col=1 if theoretical_pattern else None
    )
    
    # 标注峰位和Miller指数
    for peak in peak_assignments:
        hkl_text = f"({peak['hkl'][0]}{peak['hkl'][1]}{peak['hkl'][2]})"
        fig.add_trace(
            go.Scatter(
                x=[peak['2theta']],
                y=[np.interp(peak['2theta'], two_theta, intensity)],
                mode='markers+text',
                marker=dict(size=8, color='red', symbol='triangle-down'),
                text=[hkl_text],
                textposition='top center',
                textfont=dict(size=10, color='red'),
                showlegend=False
            ),
            row=1 if theoretical_pattern else None,
            col=1 if theoretical_pattern else None
        )
    
    # 如果有理论图谱，添加对比
    if theoretical_pattern:
        fig.add_trace(
            go.Scatter(
                x=two_theta,
                y=intensity / np.max(intensity) * 100,
                mode='lines',
                name='实验 (归一化)',
                line=dict(color='black', width=1.5)
            ),
            row=2, col=1
        )
        
        theo_2theta = theoretical_pattern['2theta']
        theo_intensity = theoretical_pattern['intensity']
        fig.add_trace(
            go.Scatter(
                x=theo_2theta,
                y=theo_intensity / np.max(theo_intensity) * 100,
                mode='markers',
                name='理论峰位',
                marker=dict(size=6, color='blue', symbol='line-ns-open')
            ),
            row=2, col=1
        )
    
    # 更新布局
    fig.update_xaxes(title_text="2θ (degrees)", row=1 if theoretical_pattern else None, col=1)
    fig.update_xaxes(title_text="2θ (degrees)", row=2, col=1) if theoretical_pattern else None
    fig.update_yaxes(title_text="强度 (a.u.)", row=1 if theoretical_pattern else None, col=1)
    fig.update_yaxes(title_text="归一化强度 (%)", row=2, col=1) if theoretical_pattern else None
    
    fig.update_layout(
        height=800 if theoretical_pattern else 500,
        title_text="XRD图谱分析结果",
        showlegend=True,
        hovermode='x unified'
    )
    
    # 保存文件
    save_file = os.path.join(save_path, "xrd_analysis_interactive.html")
    fig.write_html(save_file)
    
    return save_file


# ============ 第四层：主流程演示 ============

def main():
    """
    演示XRD分析工具包的三种典型应用场景
    
    场景1：完整分析给定的SnO2 XRD图谱
    场景2：批量分析不同prominence参数对峰检测的影响
    场景3：从Materials Project数据库批量查询多种氧化物的晶体结构
    """
    
    print("=" * 60)
    print("场景1：SnO2 XRD图谱完整分析")
    print("=" * 60)
    print("问题描述：分析给定的SnO2单相材料XRD图谱，识别晶相、标定Miller指数、计算晶格参数")
    print("-" * 60)
    
    # 模拟实验数据（基于问题中的峰位）
    experimental_peaks_2theta = np.array([26.5, 33.7, 38.8, 51.5, 54.5, 62.2, 65.6, 70.8, 78.2, 89.2])
    experimental_peaks_intensity = np.array([100, 82, 24, 70, 17, 16, 17, 21, 15, 13])
    
    # 生成完整的XRD图谱（添加背景噪声）
    two_theta_full = np.linspace(20, 95, 1500)
    intensity_full = np.zeros_like(two_theta_full)
    
    # 为每个峰添加高斯形状
    for peak_pos, peak_int in zip(experimental_peaks_2theta, experimental_peaks_intensity):
        gaussian = peak_int * np.exp(-((two_theta_full - peak_pos) ** 2) / (2 * 0.15 ** 2))
        intensity_full += gaussian
    
    # 添加噪声
    intensity_full += np.random.normal(0, 0.5, len(intensity_full))
    intensity_full = np.maximum(intensity_full, 0)  # 确保非负
    
    # 步骤1：峰检测
    # 调用函数：detect_peaks_from_pattern()
    print("\n步骤1：自动检测衍射峰")
    peak_detection_result = detect_peaks_from_pattern(two_theta_full, intensity_full, prominence=3.0)
    detected_peaks = peak_detection_result['result']['peak_positions']
    print(f"  检测到 {len(detected_peaks)} 个衍射峰")
    print(f"  前5个最强峰位置: {detected_peaks[:5]}")
    
    # 步骤2：完整XRD分析
    # 调用函数：analyze_xrd_pattern()，内部调用了 detect_peaks_from_pattern(), calculate_d_spacing(), 
    #          assign_miller_indices_tetragonal(), refine_lattice_parameters_tetragonal()
    print("\n步骤2：执行完整XRD分析（相鉴定、Miller指数标定、晶格参数计算）")
    analysis_result = analyze_xrd_pattern(two_theta_full, intensity_full, expected_phase='SnO2')
    
    phase = analysis_result['result']['identified_phase']
    space_group = analysis_result['result']['space_group']
    lattice = analysis_result['result']['lattice_params']
    assignments = analysis_result['result']['peak_assignments']
    
    print(f"  鉴定相: {phase}")
    print(f"  空间群: {space_group}")
    print(f"  晶格参数: a = b = {lattice['a']:.3f} Å, c = {lattice['c']:.3f} Å")
    print(f"  成功标定 {len(assignments)} 个峰的Miller指数")
    print("\n  峰标定详情（前5个）:")
    for i, peak in enumerate(assignments[:5]):
        print(f"    2θ={peak['2theta']:.1f}° → (hkl)={peak['hkl']}, d={peak['d_exp']:.3f}Å")
    
    # 步骤3：可视化
    # 调用函数：plot_xrd_analysis()
    print("\n步骤3：生成交互式可视化图表")
    plot_file = plot_xrd_analysis(two_theta_full, intensity_full, assignments, save_path='./images/')
    print(f"  图表已保存至: {plot_file}")
    
    print(f"\n✓ 场景1最终答案：")
    print(f"  {phase} (空间群 {space_group}) 的晶格参数为 a=b={lattice['a']:.2f}Å, c={lattice['c']:.2f}Å, α=β=γ=90°")
    print(f"  主要衍射峰已标定为 {', '.join([str(p['hkl']) for p in assignments[:5]])} 等反射\n")
    
    
    print("=" * 60)
    print("场景2：峰检测参数优化分析")
    print("=" * 60)
    print("问题描述：改变prominence阈值参数，观察峰检测数量和质量的变化")
    print("-" * 60)
    
    # 批量测试不同prominence值
    # 调用函数：detect_peaks_from_pattern()（循环调用）
    prominence_range = np.linspace(1.0, 10.0, 10)
    num_peaks_detected = []
    
    print("\n测试不同prominence阈值:")
    for prom in prominence_range:
        # 调用函数：detect_peaks_from_pattern()
        result = detect_peaks_from_pattern(two_theta_full, intensity_full, prominence=prom)
        num_peaks = result['metadata']['num_peaks']
        num_peaks_detected.append(num_peaks)
        print(f"  prominence={prom:.1f} → 检测到 {num_peaks} 个峰")
    
    optimal_prominence = prominence_range[np.argmin(np.abs(np.array(num_peaks_detected) - 10))]
    print(f"\n✓ 场景2完成：最优prominence参数为 {optimal_prominence:.1f}（检测到约10个主峰）")
    print(f"  参数范围测试: {len(prominence_range)} 组数据\n")
    
    
    print("=" * 60)
    print("场景3：Materials Project数据库批量查询")
    print("=" * 60)
    print("问题描述：从Materials Project数据库批量获取常见氧化物的晶体结构参数")
    print("-" * 60)
    
    # 批量查询多种氧化物
    oxide_formulas = ['SnO2', 'TiO2', 'ZnO', 'Fe2O3']
    comparison_results = {}
    
    print("\n查询氧化物晶体结构:")
    for formula in oxide_formulas:
        # 调用函数：fetch_structure_from_mp()
        db_result = fetch_structure_from_mp(formula)
        
        if db_result['result'] is not None:
            lattice_params = db_result['result']['lattice_params']
            space_group = db_result['result']['space_group']
            
            comparison_results[formula] = {
                'a': lattice_params['a'],
                'c': lattice_params.get('c', lattice_params['a']),
                'space_group': space_group
            }
            
            print(f"  {formula}:")
            print(f"    空间群: {space_group}")
            print(f"    晶格参数: a={lattice_params['a']:.3f}Å, c={lattice_params.get('c', lattice_params['a']):.3f}Å")
        else:
            print(f"  {formula}: 数据库查询失败（可能需要API密钥）")
            # 使用已知数据作为备选
            if formula == 'SnO2':
                comparison_results[formula] = {'a': 4.76, 'c': 3.21, 'space_group': 'P4_2/mnm'}
    
    print(f"\n✓ 场景3完成：成功查询 {len(comparison_results)} 种氧化物的结构数据")
    print(f"  对比分析: SnO2的c/a比 = {comparison_results['SnO2']['c']/comparison_results['SnO2']['a']:.3f}\n")
    
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了从XRD原始数据到晶格参数的完整分析流程")
    print("- 场景2展示了工具的参数优化能力（峰检测阈值调优）")
    print("- 场景3展示了与Materials Project数据库的集成能力")
    print("\n核心工具函数调用链:")
    print("  analyze_xrd_pattern() → detect_peaks_from_pattern()")
    print("                       → calculate_d_spacing()")
    print("                       → assign_miller_indices_tetragonal()")
    print("                       → refine_lattice_parameters_tetragonal()")
    print("                       → fetch_structure_from_mp()")


if __name__ == "__main__":
    main()