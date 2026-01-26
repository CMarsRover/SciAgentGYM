# Filename: materials_toolkit.py
"""
材料科学计算工具包

主要功能：
1. XRD相鉴定与峰匹配：基于pymatgen与scipy实现衍射峰匹配与角度漂移校准
2. 元素与数据库访问：使用mendeleev获取元素物性，示例集成Materials Project（mp-api）
3. 组合分析与可视化：多材料混合谱的相识别与Plotly交互式可视化

依赖库：
pip install numpy scipy pymatgen mendeleev plotly mp-api
"""

import os
import json
import math
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

# 尝试导入可选依赖
try:
    from mendeleev import element
    MENDELEEV_AVAILABLE = True
except ImportError:
    MENDELEEV_AVAILABLE = False
    print("Warning: mendeleev not available. Install with: pip install mendeleev")

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Install with: pip install pymatgen")

# ======== 全局常量（避免魔法数，集中管理） ========
MID_SAVE_DIR = "./mid_result/materials"
TOOL_IMAGE_DIR = "./tool_images"
DEFAULT_WAVELENGTH_CUKA = 1.5406  # Å, Cu Kα
DEFAULT_RANGE = (10.0, 90.0)      # 2θ范围，单位度
DEFAULT_TOLERANCE = 0.25          # 峰匹配容差（度）
MAX_SHIFT_DEG = 1.0               # 允许的整体角度漂移（度）
PLOT_WIDTH = 900
PLOT_HEIGHT = 500

os.makedirs(MID_SAVE_DIR, exist_ok=True)
os.makedirs(TOOL_IMAGE_DIR, exist_ok=True)

# ======== 演示所用的参考图谱（从题目图片读出的典型峰，作为常量管理） ========
# 注意：这些参考峰仅用于工具演示与教学，实际工程请用标准数据库（PDF卡片/ICSD/MP）生成或导入
REF_AG2O = {
    "name": "Ag2O",
    "peaks_2theta": [32.5, 38.0, 54.6, 65.0, 90.0],
    "intensity":   [100, 42, 36, 37, 16]
}
REF_ALN = {
    "name": "AlN",
    "peaks_2theta": [35.0, 36.7, 49.6, 59.0, 65.3, 70.8, 72.1, 80.5, 85.7],
    "intensity":   [100, 92, 33, 61, 55, 10, 43, 7, 2]
}
REF_BAS = {
    "name": "BAs",
    "peaks_2theta": [32.3, 37.2, 54.2, 64.1, 69.0, 79.0, 88.5],
    "intensity":   [100, 40, 40, 40, 21, 17, 16]
}
REF_YSF = {
    "name": "YSF",
    "peaks_2theta": [27.0, 32.0, 33.0, 43.0, 47.5, 49.0, 55.5, 60.0, 63.0, 68.0, 69.5, 79.5, 86.5, 88.5],
    "intensity":   [5, 39, 100, 89, 49, 49, 20, 11, 15, 3, 28, 8, 22, 7]
}
REF_ACOF = {
    "name": "AcOF",
    "peaks_2theta": [32.0, 36.9, 54.0, 64.0, 68.8, 79.0, 88.5],
    "intensity":   [100, 40, 40, 40, 21, 17, 16]
}

# 综合谱（第一幅图）抽取的主要峰位（仅用于演示）
COMPOSITE_PATTERN = {
    "peaks_2theta": [12.5, 26.0, 27.2, 31.8, 32.3, 36.8, 37.5, 38.7, 43.2, 47.5, 49.0, 54.1, 55.0, 56.0, 60.5, 64.5, 65.0, 68.5, 70.0, 71.0, 73.5, 78.5, 79.5, 85.8, 88.5, 90.0],
    "intensity":   [1, 5, 12, 50, 100, 10, 6, 5, 12, 7, 6, 1, 20, 36, 4, 20, 37, 9, 21, 15, 13, 7, 8, 3, 9, 16]
}


# ============ 第一层：原子工具函数（Atomic Tools） ============
def save_json_data(filename: str, data: Dict) -> dict:
    """
    将字典数据保存为JSON文件（Function Calling兼容的简单持久化工具）
    
    原理与说明：
    - 将中间结果（峰表、参数、得分等）JSON序列化保存，便于组合函数复用与审计
    - 路径统一到 ./mid_result/materials，便于管理与后续加载
    
    ### 🔧 更新后的代码质量检查清单
    - [x] 所有函数参数类型为可JSON序列化
    - [x] Python对象构建逻辑在函数内部
    - [x] 支持多种输入格式（字典）
    - [x] 示例使用基础类型
    
    Args:
        filename: 文件名（不含路径），如 'peaks.json'
        data: 读图上重要的信息需要保存的数据字典
    
    Returns:
        dict: {'result': filepath, 'metadata': {'size': bytes}}
    
    Example:
        >>> save_json_data('demo.json', {'a': 1})
    """
    if not isinstance(filename, str):
        raise TypeError("filename必须是字符串")
    if not isinstance(data, dict):
        raise TypeError("data必须是字典")

    filepath = os.path.join(MID_SAVE_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size = os.path.getsize(filepath)
    return {'result': filepath, 'metadata': {'size_bytes': size}}


def get_element_properties(symbol: str) -> dict:
    """
    获取元素基础物性（来自mendeleev数据库）
    
    原理与说明：
    - 通过mendeleev快速查询元素的原子序数、原子量、范德华半径等；用于估算散射能力或材料性质初筛
    - 数据库为本地打包数据，访问稳定
    
    Args:
        symbol: 元素符号，示例'Ag','O','B','As'
    
    Returns:
        dict: {'result': {...元素属性...}, 'metadata': {'source': 'mendeleev'}}
    
    Example:
        >>> get_element_properties('Ag')
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol必须为字符串")
    
    if not MENDELEEV_AVAILABLE:
        # 提供基本的元素数据作为后备
        basic_elements = {
            'Ag': {'symbol': 'Ag', 'name': 'Silver', 'atomic_number': 47, 'atomic_weight': 107.868, 'density': 10.5, 'vdw_radius': 1.72, 'group_id': 11, 'period': 5},
            'O': {'symbol': 'O', 'name': 'Oxygen', 'atomic_number': 8, 'atomic_weight': 15.999, 'density': 0.0014, 'vdw_radius': 1.52, 'group_id': 16, 'period': 2},
            'B': {'symbol': 'B', 'name': 'Boron', 'atomic_number': 5, 'atomic_weight': 10.811, 'density': 2.34, 'vdw_radius': 1.92, 'group_id': 13, 'period': 2},
            'As': {'symbol': 'As', 'name': 'Arsenic', 'atomic_number': 33, 'atomic_weight': 74.922, 'density': 5.78, 'vdw_radius': 1.85, 'group_id': 15, 'period': 4},
            'Y': {'symbol': 'Y', 'name': 'Yttrium', 'atomic_number': 39, 'atomic_weight': 88.906, 'density': 4.47, 'vdw_radius': 2.27, 'group_id': 3, 'period': 5},
            'S': {'symbol': 'S', 'name': 'Sulfur', 'atomic_number': 16, 'atomic_weight': 32.065, 'density': 2.07, 'vdw_radius': 1.80, 'group_id': 16, 'period': 3},
            'F': {'symbol': 'F', 'name': 'Fluorine', 'atomic_number': 9, 'atomic_weight': 18.998, 'density': 0.0017, 'vdw_radius': 1.47, 'group_id': 17, 'period': 2}
        }
        if symbol in basic_elements:
            return {'result': basic_elements[symbol], 'metadata': {'source': 'basic_data'}}
        else:
            return {'result': None, 'metadata': {'error': f'Element {symbol} not in basic database', 'source': 'basic_data'}}
    
    try:
        e = element(symbol)
        props = {
            "symbol": e.symbol,
            "name": e.name,
            "atomic_number": e.atomic_number,
            "atomic_weight": float(e.atomic_weight),
            "density": float(e.density) if e.density else None,
            "vdw_radius": float(e.vdw_radius) if e.vdw_radius else None,
            "group_id": e.group_id,
            "period": e.period
        }
        return {'result': props, 'metadata': {'source': 'mendeleev'}}
    except Exception as exc:
        return {'result': None, 'metadata': {'error': str(exc), 'source': 'mendeleev'}}


def simulate_xrd_from_structure(identifier: str,
                                wavelength: float = DEFAULT_WAVELENGTH_CUKA,
                                two_theta_range: Tuple[float, float] = DEFAULT_RANGE,
                                source: str = "cif") -> dict:
    """
    用pymatgen从结构文件或材料ID生成XRD图谱（返回峰位与相对强度）
    
    原理与说明：
    - 使用pymatgen的XRDCalculator计算粉末衍射，支持给定波长与2θ范围
    - 结构可来源于CIF文件或Materials Project（需网络与MP API Key）
    
    Args:
        identifier: 结构来源标识；当source='cif'时是CIF文件路径；当source='mp'时是材料ID（如'mp-1234'）
        wavelength: X射线波长（Å），默认Cu Kα 1.5406 Å
        two_theta_range: 2θ范围 (min_deg, max_deg)
        source: 'cif'或'mp'
    
    Returns:
        dict: {
            'result': {'two_theta': List[float], 'intensity': List[float]},
            'metadata': {'source': source, 'identifier': identifier}
        }
    
    Example:
        >>> simulate_xrd_from_structure('./example.cif')
    """
    if not isinstance(identifier, str):
        raise TypeError("identifier必须为字符串")
    if not isinstance(wavelength, (int, float)) or wavelength <= 0:
        raise ValueError("wavelength必须为正数")
    if not isinstance(two_theta_range, (list, tuple)) or len(two_theta_range) != 2:
        raise ValueError("two_theta_range必须为长度为2的序列")
    tmin, tmax = float(two_theta_range[0]), float(two_theta_range[1])
    if tmin >= tmax:
        raise ValueError("two_theta_range必须满足min < max")

    if not PYMATGEN_AVAILABLE:
        return {'result': {'two_theta': [], 'intensity': []}, 'metadata': {'error': 'pymatgen not available', 'source': source, 'identifier': identifier}}
    
    try:
        if source.lower() == "cif":
            if not os.path.isfile(identifier):
                raise FileNotFoundError(f"CIF文件不存在: {identifier}")
            structure = Structure.from_file(identifier)
        elif source.lower() == "mp":
            # 尝试从Materials Project下载（可能需要网络与API Key）
            try:
                from mp_api.client import MPRester
                with MPRester() as mpr:
                    doc = mpr.materials.summary.get_data_by_id(identifier)
                    if not doc or not getattr(doc[0], "structure", None):
                        raise ValueError("未从MP获得结构数据")
                    structure = doc[0].structure
            except ImportError:
                raise ValueError("mp-api not available")
        else:
            raise ValueError("source必须为'cif'或'mp'")

        calc = XRDCalculator(wavelength=wavelength)
        pattern = calc.get_pattern(structure, two_theta_range=(tmin, tmax))
        # 保证可序列化
        res = {'two_theta': list(pattern.x), 'intensity': list(pattern.y)}
        return {'result': res, 'metadata': {'source': source, 'identifier': identifier}}
    except Exception as exc:
        return {'result': {'two_theta': [], 'intensity': []}, 'metadata': {'error': str(exc), 'source': source, 'identifier': identifier}}


def normalize_intensity(intensity: List[float]) -> dict:
    """
    将强度归一化到最大值为100
    
    原理与说明：
    - XRD相对强度通常统一到100，便于跨样品比较与匹配评分
    - 防止零向量与负值，安全归一化
    
    Args:
        intensity: 强度数组（list），非负
    
    Returns:
        dict: {'result': List[float], 'metadata': {'max_before': float}}
    
    Example:
        >>> normalize_intensity([10, 50, 100])
    """
    if not isinstance(intensity, list):
        raise TypeError("intensity必须是list")
    if len(intensity) == 0:
        return {'result': [], 'metadata': {'max_before': 0.0}}
    arr = np.array(intensity, dtype=float)
    if np.any(arr < 0):
        raise ValueError("强度必须非负")
    m = float(np.max(arr))
    res = list((arr / m * 100.0) if m > 0 else arr)
    return {'result': res, 'metadata': {'max_before': m}}


def peak_matching_score(observed_2theta: List[float],
                        candidate_2theta: List[float],
                        tolerance: float = DEFAULT_TOLERANCE,
                        allow_shift: bool = True,
                        max_shift: float = MAX_SHIFT_DEG) -> dict:
    """
    计算候选相与观测峰的匹配得分（考虑整体角度漂移校准）
    
    原理与说明：
    - 使用最近邻匹配统计命中比例，并用scipy.optimize对全局角度漂移Δ进行最优校准
    - 得分定义：命中数 / 候选峰数，范围0-1；并给出最佳漂移Δ
    
    Args:
        observed_2theta: 观测峰位数组（度）
        candidate_2theta: 候选材料峰位数组（度）
        tolerance: 匹配容差（度）
        allow_shift: 是否允许整体漂移优化
        max_shift: 漂移范围（绝对值最大度数）
    
    Returns:
        dict: {
            'result': {'score': float, 'shift_deg': float, 'matches': List[Tuple[float, float]]},
            'metadata': {'tolerance': float}
        }
    
    Example:
        >>> peak_matching_score([32.3, 37.2], [32.5, 38.0])
    """
    for arr in (observed_2theta, candidate_2theta):
        if not isinstance(arr, list):
            raise TypeError("输入必须为list")
        if any([not isinstance(x, (int, float)) for x in arr]):
            raise TypeError("峰位必须为数值")
    if tolerance <= 0:
        raise ValueError("tolerance必须为正数")

    obs = np.array(observed_2theta, dtype=float)
    cand = np.array(candidate_2theta, dtype=float)

    def score_with_shift(shift: float):
        shifted = cand + shift
        matches = 0
        matched_pairs = []
        for c in shifted:
            diffs = np.abs(obs - c)
            min_d = float(np.min(diffs)) if len(diffs) > 0 else math.inf
            if min_d <= tolerance:
                matches += 1
                matched_pairs.append((c, float(obs[np.argmin(diffs)])))
        return -matches, matched_pairs  # 负号用于最小化

    best_shift = 0.0
    best_pairs = []
    if allow_shift:
        res = minimize(lambda s: score_with_shift(float(s))[0],
                       x0=0.0,
                       bounds=[(-max_shift, max_shift)],
                       method='L-BFGS-B')
        best_shift = float(res.x[0])
        _, best_pairs = score_with_shift(best_shift)
    else:
        _, best_pairs = score_with_shift(0.0)

    score = len(best_pairs) / max(1, len(cand))
    return {'result': {'score': float(score), 'shift_deg': best_shift, 'matches': best_pairs},
            'metadata': {'tolerance': tolerance, 'allow_shift': allow_shift}}


# ============ 第二层：组合工具函数（Composite Tools） ============
def identify_phases_by_matching(observed_2theta: List[float],
                                observed_intensity: Optional[List[float]],
                                candidate_refs: List[Dict],
                                tolerance: float = DEFAULT_TOLERANCE,
                                allow_shift: bool = True) -> dict:
    """
    从候选参考中识别最可能的晶相（峰匹配与综合评分）
    
    科学原理：
    - XRD相鉴定基于峰位与相对强度对比；峰位主导匹配，强度作为次级加权
    - 采用最近邻峰位匹配与全局角度漂移校准，提高仪器偏差或应力导致的整体偏移下的鲁棒性
    
    Args:
        observed_2theta: 观测峰位（度）
        observed_intensity: 观测相对强度（可选，用于强度加权）；若None则仅峰位评分
        candidate_refs: 候选参考列表，每个包含{'name','peaks_2theta','intensity'}
        tolerance: 峰位匹配容差（度）
        allow_shift: 是否允许整体角度漂移优化
    
    Returns:
        dict: {
            'result': {'ranking': List[Dict]},    # 每个包含 name, score, shift_deg
            'metadata': {'tolerance': tolerance}
        }
    """
    if not isinstance(observed_2theta, list):
        raise TypeError("observed_2theta必须为list")
    if observed_intensity is not None and not isinstance(observed_intensity, list):
        raise TypeError("observed_intensity必须为list或None")
    if not isinstance(candidate_refs, list):
        raise TypeError("candidate_refs必须为list")

    # 归一化观测强度（如果提供）
    if observed_intensity is not None and len(observed_intensity) > 0:
        norm_obs = normalize_intensity(observed_intensity)['result']
    else:
        norm_obs = None

    results = []
    for ref in candidate_refs:
        name = ref.get("name", "Unknown")
        cand_peaks = ref.get("peaks_2theta", [])
        cand_intens = ref.get("intensity", [])
        # === using atomic tool: peak_matching_score(), and get ** returns
        pm = peak_matching_score(observed_2theta, cand_peaks, tolerance=tolerance, allow_shift=allow_shift)
        score = pm['result']['score']
        shift = pm['result']['shift_deg']
        # 强度加权：若观测强度可用，计算匹配对的强度差惩罚
        if norm_obs is not None and len(cand_intens) == len(cand_peaks):
            # 为每个匹配的观测峰找到观测强度（最近邻）
            penalty = 0.0
            for c_shifted, o in pm['result']['matches']:
                # 找到观测峰的索引
                idx_obs = np.argmin(np.abs(np.array(observed_2theta) - o))
                I_obs = norm_obs[idx_obs]
                # 找到候选峰的原始索引（反向匹配）
                idx_cand = np.argmin(np.abs((np.array(cand_peaks) + shift) - c_shifted))
                I_cand = normalize_intensity(cand_intens)['result'][idx_cand]
                penalty += abs(I_obs - I_cand) / 100.0
            # 强度差惩罚越小越好，将其转换为奖励因子
            intensity_factor = math.exp(-penalty)
            score = score * intensity_factor
        results.append({"name": name, "score": round(float(score), 4), "shift_deg": round(float(shift), 4)})

    ranking = sorted(results, key=lambda x: x['score'], reverse=True)
    return {'result': {'ranking': ranking}, 'metadata': {'tolerance': tolerance, 'allow_shift': allow_shift}}


def merge_patterns(patterns: List[Dict]) -> dict:
    """
    将多个参考图谱合成为混合谱（简单叠加）
    
    原理与说明：
    - 将多个相的峰位合并并强度叠加，模拟混合样品的衍射图谱
    - 强度采用线性叠加后归一化
    
    Args:
        patterns: [{'two_theta': List[float], 'intensity': List[float]}]
    
    Returns:
        dict: {'result': {'two_theta': List[float], 'intensity': List[float]}, 'metadata': {}}
    """
    if not isinstance(patterns, list):
        raise TypeError("patterns必须为list")
    merged = {}
    for pat in patterns:
        tt = pat.get('two_theta', [])
        I = pat.get('intensity', [])
        for t, val in zip(tt, I):
            t_round = round(float(t), 2)
            merged[t_round] = merged.get(t_round, 0.0) + float(val)
    # 排序并归一
    two_theta = sorted(merged.keys())
    intensity = [merged[t] for t in two_theta]
    intensity = normalize_intensity(intensity)['result']
    return {'result': {'two_theta': two_theta, 'intensity': intensity}, 'metadata': {'count': len(two_theta)}}


# ============ 第三层：可视化工具（Visualization） ============
def plot_xrd_pattern(patterns: Dict[str, Dict],
                     title: str,
                     filename: Optional[str] = None) -> dict:
    """
    使用Plotly绘制XRD图谱（可叠加多条曲线），自动保存到 ./tool_images
    
    Args:
        patterns: 形如 {'Composite': {'two_theta': List, 'intensity': List}, 'Ag2O': {...}}
        title: 图标题
        filename: 自定义文件名（不含路径和扩展名），默认自动生成
    
    Returns:
        dict: {'result': filepath, 'metadata': {'curves': list(patterns.keys())}}
    """
    try:
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
        print("Warning: plotly not available. Install with: pip install plotly")

    if filename is None:
        safe_title = "".join([c if c.isalnum() else "_" for c in title])
        filename = f"{safe_title}.png"
    filepath = os.path.join(TOOL_IMAGE_DIR, filename)

    if not PLOTLY_AVAILABLE:
        # 使用matplotlib作为后备
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, pat in patterns.items():
            tt = pat.get('two_theta', [])
            I = pat.get('intensity', [])
            ax.plot(tt, I, 'o-', label=name, linewidth=2, markersize=4)
        
        ax.set_xlabel('2 Theta (degrees)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FILE_GENERATED: Plot (matplotlib) | PATH: {filepath}")
        return {'result': filepath, 'metadata': {'curves': list(patterns.keys()), 'backend': 'matplotlib'}}

    try:
        fig = go.Figure()
        for name, pat in patterns.items():
            tt = pat.get('two_theta', [])
            I = pat.get('intensity', [])
            fig.add_trace(go.Bar(x=tt, y=I, name=name, opacity=0.7))
        fig.update_layout(
            title=title,
            xaxis_title="2 Theta (degrees)",
            yaxis_title="Intensity (a.u.)",
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.write_image(filepath)
        print(f"FILE_GENERATED: Plot (plotly) | PATH: {filepath}")
        return {'result': filepath, 'metadata': {'curves': list(patterns.keys()), 'backend': 'plotly'}}
    except Exception as e:
        print(f"Plotly failed: {e}, falling back to matplotlib")
        # 使用matplotlib作为后备
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, pat in patterns.items():
            tt = pat.get('two_theta', [])
            I = pat.get('intensity', [])
            ax.plot(tt, I, 'o-', label=name, linewidth=2, markersize=4)
        
        ax.set_xlabel('2 Theta (degrees)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"FILE_GENERATED: Plot (matplotlib) | PATH: {filepath}")
        return {'result': filepath, 'metadata': {'curves': list(patterns.keys()), 'backend': 'matplotlib'}}


# ============ 第四层：主流程演示 ============
def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    
    ⚠️ 必须严格按照以下格式编写：
    """
    print("=" * 60)
    print("场景1：原始问题求解 - XRD混合谱相鉴定")
    print("=" * 60)
    print("问题描述：基于给定的混合XRD峰与五种候选材料参考图谱，识别复合材料中包含的三种晶相。")
    print("-" * 60)

    # 步骤1：保存观测数据以便审计和复用
    # 调用函数：save_json_data()
    res_save = save_json_data("composite_peaks.json", COMPOSITE_PATTERN)
    print(f"FUNCTION_CALL: save_json_data | PARAMS: filename='composite_peaks.json' | RESULT: {res_save['result']}")

    # 步骤2：执行相识别（峰匹配 + 漂移校准）
    # 调用函数：identify_phases_by_matching()
    candidates = [REF_AG2O, REF_ALN, REF_BAS, REF_YSF, REF_ACOF]
    res_id = identify_phases_by_matching(
        observed_2theta=COMPOSITE_PATTERN['peaks_2theta'],
        observed_intensity=COMPOSITE_PATTERN['intensity'],
        candidate_refs=candidates,
        tolerance=DEFAULT_TOLERANCE,
        allow_shift=True
    )
    print(f"FUNCTION_CALL: identify_phases_by_matching | PARAMS: tolerance={DEFAULT_TOLERANCE}, allow_shift=True | RESULT: {res_id['result']['ranking']}")
    top3 = [r['name'] for r in res_id['result']['ranking'][:3]]

    # 步骤3：可视化混合谱与前三名匹配参考
    # 调用函数：plot_xrd_pattern()
    composite_curve = {
        'two_theta': COMPOSITE_PATTERN['peaks_2theta'],
        'intensity': normalize_intensity(COMPOSITE_PATTERN['intensity'])['result']
    }
    ref_curves = {}
    for ref in res_id['result']['ranking'][:3]:
        name = ref['name']
        ref_data = next(item for item in candidates if item['name'] == name)
        # 将参考峰转换为细棒图曲线
        ref_curves[name] = {'two_theta': ref_data['peaks_2theta'],
                            'intensity': normalize_intensity(ref_data['intensity'])['result']}
    vis_input = {"Composite": composite_curve}
    vis_input.update(ref_curves)
    res_plot = plot_xrd_pattern(vis_input, title="场景1_混合谱与前三匹配参考")
    print(f"FUNCTION_CALL: plot_xrd_pattern | PARAMS: title='场景1_混合谱与前三匹配参考' | RESULT: {res_plot['result']}")

    print(f"✓ 场景1最终答案：识别到的三种晶相候选为 {', '.join(top3)}\n")

    print("=" * 60)
    print("场景2：参数扫描 - 容差对识别稳定性的影响")
    print("=" * 60)
    print("问题描述：在不同峰位匹配容差下（0.10-0.50度），评估识别结果的鲁棒性。")
    print("-" * 60)

    tolerances = [0.10, 0.20, 0.30, 0.40, 0.50]
    scan_results = []
    for tol in tolerances:
        # 调用函数：identify_phases_by_matching()
        res_scan = identify_phases_by_matching(
            observed_2theta=COMPOSITE_PATTERN['peaks_2theta'],
            observed_intensity=COMPOSITE_PATTERN['intensity'],
            candidate_refs=candidates,
            tolerance=tol,
            allow_shift=True
        )
        ranking = [r['name'] for r in res_scan['result']['ranking'][:3]]
        scan_results.append({'tolerance': tol, 'top3': ranking})
        print(f"FUNCTION_CALL: identify_phases_by_matching | PARAMS: tolerance={tol}, allow_shift=True | RESULT: {ranking}")

    # 保存扫描结果
    res_save_scan = save_json_data("tolerance_scan.json", {"scan_results": scan_results})
    print(f"FUNCTION_CALL: save_json_data | PARAMS: filename='tolerance_scan.json' | RESULT: {res_save_scan['result']}")
    print("✓ 场景2完成：容差扫描结果已生成并保存\n")

    # print("=" * 60)
    # print("场景3：数据库集成 - 元素物性查询与结构谱模拟示例")
    # print("=" * 60)
    # print("问题描述：查询Ag、B、As、Y、S、F的元素属性，并演示从CIF/MP结构生成XRD谱的流程。")
    # print("-" * 60)

    # # 步骤1：元素属性查询
    # # 调用函数：get_element_properties()
    # elems = ['Ag', 'B', 'As', 'Y', 'S', 'F']
    # elem_props = {}
    # for e in elems:
    #     res_e = get_element_properties(e)
    #     elem_props[e] = res_e['result']
    #     print(f"FUNCTION_CALL: get_element_properties | PARAMS: symbol='{e}' | RESULT: {res_e['result'] and res_e['result'].get('atomic_number')}")

    # res_save_elems = save_json_data("element_props.json", elem_props)
    # print(f"FUNCTION_CALL: save_json_data | PARAMS: filename='element_props.json' | RESULT: {res_save_elems['result']}")

    # # 步骤2：结构谱模拟（演示接口，若无文件或网络则返回空谱）
    # # 调用函数：simulate_xrd_from_structure()
    # demo_cif_path = "./example.cif"  # 演示路径；若不存在将触发安全错误处理
    # res_sim = simulate_xrd_from_structure(demo_cif_path, wavelength=DEFAULT_WAVELENGTH_CUKA, source="cif")
    # print(f"FUNCTION_CALL: simulate_xrd_from_structure | PARAMS: identifier='{demo_cif_path}', source='cif' | RESULT: len(two_theta)={len(res_sim['result']['two_theta'])}")

    # # 保存模拟谱或者空谱
    # res_save_sim = save_json_data("simulated_xrd.json", res_sim['result'])
    # print(f"FUNCTION_CALL: save_json_data | PARAMS: filename='simulated_xrd.json' | RESULT: {res_save_sim['result']}")
    # print("✓ 场景3完成：元素属性与结构谱接口演示已完成\n")

    # print("=" * 60)
    # print("工具包演示完成")
    # print("=" * 60)
    # print("总结：")
    # print("- 场景1展示了解决原始问题的完整流程")
    # print("- 场景2展示了工具的参数泛化能力")
    # print("- 场景3展示了工具与数据库的集成能力")

    # # 原始题目正确答案校准输出（来自校准推理过程）
    # final_answer = "Ag₂O, BAs, YSF"
    # print(f"FINAL_ANSWER: {final_answer}")


if __name__ == "__main__":
    main()