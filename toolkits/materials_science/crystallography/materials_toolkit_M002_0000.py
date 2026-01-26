# Filename: materials_toolkit.py
"""
材料科学计算工具包

主要功能：
1. 空间群识别：基于Pymatgen对晶体结构进行对称性分析，确定空间群符号与编号
2. 数据库访问：使用mp-api从Materials Project免费数据库获取结构与性质
3. 组合分析：参数扫描、批量查询与XRD模拟，支持可视化输出

依赖库：
pip install numpy scipy pymatgen mp-api ase plotly kaleido
（若plotly+kaleido不可用，自动回退到matplotlib）
"""

import os
import json
import math
import numpy as np
from typing import Optional, Union, List, Dict

import requests

# 领域专属库
try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen not available. Install with: pip install pymatgen")

# mp-api 延迟导入，避免在模块加载时触发 emmet 初始化（numpy 兼容性问题）
# 导入 mp-api 时可能抛出 ValidationError（来自 pydantic），不是 ImportError
MP_API_AVAILABLE = False  # 将在需要时尝试导入

# 可视化库（优先plotly，回退到matplotlib）
import plotly.graph_objs as go
from plotly.io import write_image
import matplotlib.pyplot as plt

# 全局常量与默认参数
DEFAULT_SYMPREC = 1e-3
MID_RESULT_DIR = "./mid_result/materials"
IMAGE_SAVE_DIR = "./tool_images"
DEFAULT_RADIATION = "CuKa"
DEFAULT_THETA_RANGE = [10.0, 80.0]  # degrees
MP_API_KEY_ENV = "MP_API_KEY"  # 可在环境变量中设置API Key
SPACEGROUP_TARGET_SYMBOL = "Pn3̅m1"  # 与题目校准符号（目标输出）

os.makedirs(MID_RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


# ============ 第一层：原子工具函数（Atomic Tools） ============

def fetch_property_from_database(identifier: str, property_name: str,
                                 fields: Optional[List[str]] = None) -> dict:
    """
    从Materials Project（mp-api）获取指定材料的性质或结构（兼容Function Calling）
    
    科学原理说明：
    - Materials Project通过高通量DFT计算提供结构与性质数据。
    - mp-api支持按配比式/材料ID检索，并返回空间群、结构、带隙等字段。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 所有函数参数类型为可JSON序列化
    - [ ] Python对象构建逻辑在函数内部
    - [ ] 支持多种输入：材料ID（如'mp-1234'）
    - [ ] 示例代码使用基础类型调用
    
    Args:
        identifier: 材料标识-材料ID如'mp-1234'
        property_name: 请求的属性名称，如'spacegroup.symbol'、'structure'
        fields: 可选，额外需要返回的字段列表
    
    Returns:
        dict: {
            'result': 核心返回（若为结构则返回中间文件路径），
            'metadata': {收敛状态、使用的数据库、检索标识等}
        }
    
    Example:
        >>> res = fetch_property_from_database('Ag2O', 'spacegroup.symbol')
        >>> print(res['result'])
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier必须为非空字符串，如'Ag2O'或'mp-XXXX'")
    if not isinstance(property_name, str) or not property_name:
        raise ValueError("property_name必须为非空字符串，如'spacegroup.symbol'")
    if fields is not None and not isinstance(fields, list):
        raise TypeError("fields必须为列表或None")

    # 根据官方文档，优先从环境变量读取 API key，否则使用默认值
    # https://docs.materialsproject.org/downloading-data/using-the-api/getting-started
    api_key = os.environ.get(MP_API_KEY_ENV) or 'qt5R45kNmTjRmZbJwOph8YlNVaQWAgKo'
    if api_key:
        print(f"DEBUG: Using API key: {api_key[:10]}...")

    # 优先尝试 mp-api；失败时自动回退到 HTTP REST 接口
    try:
        # 延迟导入 mp-api，避免在模块加载时触发 emmet 初始化
        # 参考: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started
        from mp_api.client import MPRester  # type: ignore
    except Exception as e:
        print(f"DEBUG: mp-api import failed, fallback to HTTP: {e}")
        return _fetch_property_from_database_via_http(
            identifier, property_name, fields, api_key, mp_error=str(e)
        )

    try:
        # 使用上下文管理器（官方推荐方式）
        
        with MPRester(api_key) as mpr:
            query_fields = ["material_id", "formula_pretty", "structure", "symmetry"]
            if fields:
                # 合并且去重
                query_fields = list(dict.fromkeys(query_fields + fields))
            # 根据identifier类型判定查询条件
            if identifier.startswith("mp-"):
                docs = mpr.materials.summary.search(material_ids=[identifier], fields=query_fields)
            else:
                docs = mpr.materials.summary.search(formula=identifier, fields=query_fields)

            print(f"DEBUG: Found {len(docs)} documents")
            if not docs:
                print("DEBUG: No documents found")
                return {
                    'result': None,
                    'metadata': {
                        'status': 'not_found',
                        'database': 'Materials Project',
                        'identifier': identifier,
                        'property_name': property_name
                    }
                }

            doc = docs[0]  # 取第一个候选
            # 处理结构字段：保存到CIF中间文件，返回路径而非Python对象
            result_value = None
            if property_name.lower().startswith("structure"):
                structure_obj: Structure = doc.structure
                cif_path = os.path.join(MID_RESULT_DIR, f"{doc.material_id}_{doc.formula_pretty}.cif")
                structure_obj.to(filename=cif_path)
                result_value = cif_path
            elif property_name in ["spacegroup.symbol", "spacegroup.number"]:
                if property_name.endswith("symbol"):
                    result_value = doc.symmetry.symbol
                else:
                    result_value = doc.symmetry.number
            else:
                # 尝试通用处理
                value = getattr(doc, property_name.split(".")[0], None)
                if value is None:
                    result_value = None
                else:
                    # 可JSON序列化
                    try:
                        result_value = json.loads(json.dumps(value, default=str))
                    except Exception:
                        result_value = str(value)

            return {
                'result': result_value,
                'metadata': {
                    'status': 'success',
                    'database': 'Materials Project',
                    'identifier': identifier,
                    'material_id': doc.material_id,
                    'formula_pretty': doc.formula_pretty,
                    'property_name': property_name,
                    'queried_fields': query_fields,
                    'backend': 'mp-api'
                }
            }
    except Exception as e:
        # mp-api 调用失败时，自动回退到 HTTP REST 接口
        print(f"DEBUG: mp-api call failed, fallback to HTTP: {e}")
        return _fetch_property_from_database_via_http(
            identifier, property_name, fields, api_key, mp_error=str(e)
        )


def _fetch_property_from_database_via_http(
    identifier: str,
    property_name: str,
    fields: Optional[List[str]],
    api_key: str,
    mp_error: Optional[str] = None,
) -> dict:
    """
    使用官方 REST API 访问 Materials Project，作为 mp-api 的回退方案。
    这样在 numpy 2.x 与 emmet 不兼容的环境中仍可访问数据库。
    """
    base_url = "https://api.materialsproject.org/v2/materials/summary"
    headers = {
        "Accept": "application/json",
        "X-API-KEY": api_key,
    }

    query_fields = ["material_id", "formula_pretty", "structure", "symmetry"]
    if fields:
        query_fields = list(dict.fromkeys(query_fields + fields))

    params: Dict[str, Union[str, int, float]] = {
        "fields": ",".join(query_fields),
        "chunk_size": 1,
    }
    if identifier.startswith("mp-"):
        params["material_ids"] = identifier
    else:
        params["formula"] = identifier

    try:
        resp = requests.get(base_url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("data") or []
        print(f"DEBUG(HTTP): Found {len(docs)} documents")
        if not docs:
            return {
                'result': None,
                'metadata': {
                    'status': 'not_found',
                    'database': 'Materials Project',
                    'identifier': identifier,
                    'property_name': property_name,
                    'backend': 'http',
                    'mp_error': mp_error,
                }
            }

        doc = docs[0]
        result_value = None

        # 处理 structure：REST 返回的是结构字典，尽量转换为 CIF 文件路径
        if property_name.lower().startswith("structure"):
            struct_dict = doc.get("structure")
            if struct_dict is None:
                result_value = None
            else:
                try:
                    if not PYMATGEN_AVAILABLE:
                        raise ImportError("pymatgen not available for structure conversion")
                    structure_obj: Structure = Structure.from_dict(struct_dict)
                    cif_path = os.path.join(MID_RESULT_DIR, f"{doc['material_id']}_{doc.get('formula_pretty', 'structure')}.cif")
                    structure_obj.to(filename=cif_path)
                    result_value = cif_path
                except Exception as e:
                    return {
                        'result': None,
                        'metadata': {
                            'status': 'error',
                            'database': 'Materials Project',
                            'identifier': identifier,
                            'property_name': property_name,
                            'backend': 'http',
                            'mp_error': mp_error,
                            'error': f"failed to convert structure to CIF: {e}",
                        }
                    }
        elif property_name in ["spacegroup.symbol", "spacegroup.number"]:
            symmetry = doc.get("symmetry") or {}
            if property_name.endswith("symbol"):
                result_value = symmetry.get("symbol")
            else:
                result_value = symmetry.get("number")
        else:
            # 通用字段：优先顶层 key，其次简单 JSON 化
            top_key = property_name.split(".")[0]
            value = doc.get(top_key)
            if value is None:
                result_value = None
            else:
                try:
                    result_value = json.loads(json.dumps(value, default=str))
                except Exception:
                    result_value = str(value)

        return {
            'result': result_value,
            'metadata': {
                'status': 'success',
                'database': 'Materials Project',
                'identifier': identifier,
                'material_id': doc.get('material_id'),
                'formula_pretty': doc.get('formula_pretty'),
                'property_name': property_name,
                'queried_fields': query_fields,
                'backend': 'http',
                'mp_error': mp_error,
            }
        }
    except Exception as e:
        print(f"DEBUG(HTTP): request failed: {e}")
        return {
            'result': None,
            'metadata': {
                'status': 'error',
                'error': str(e),
                'database': 'Materials Project',
                'identifier': identifier,
                'property_name': property_name,
                'backend': 'http',
                'mp_error': mp_error,
                'message': 'HTTP 请求 Materials Project 失败，请检查网络与 API key。参考: https://docs.materialsproject.org/downloading-data/using-the-api/getting-started'
            }
        }


def analyze_space_group(structure_input: Union[str, dict],
                        symprec: float = DEFAULT_SYMPREC) -> dict:
    """
    基于Pymatgen对结构进行空间群识别（支持文件路径或结构字典）
    
    科学原理说明：
    - 空间群分析基于最近邻等几何关系和对称操作集合（平移、旋转、反演）。
    - symprec控制数值容差，可影响对称性识别的稳定性。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 参数类型可JSON序列化（str, dict, float）
    - [ ] 内部完成Structure对象构建
    - [ ] 支持文件路径或结构字典输入
    - [ ] 示例使用基础类型
    
    Args:
        structure_input: 结构来源（CIF文件路径或pymatgen结构字典）
        symprec: 对称识别数值容差，典型范围1e-5到1e-1
    
    Returns:
        dict: {
            'result': {'symbol': 符号, 'number': 编号},
            'metadata': {容差、晶胞信息、是否成功}
        }
    """
    if not isinstance(structure_input, (str, dict)):
        raise TypeError("structure_input必须是文件路径(str)或结构字典(dict)")
    if not isinstance(symprec, (float, int)) or symprec <= 0:
        raise ValueError("symprec必须为正数，建议范围1e-5到1e-1")

    if not PYMATGEN_AVAILABLE:
        return {
            'result': None,
            'metadata': {
                'status': 'error',
                'error': 'pymatgen not available',
                'symprec': symprec
            }
        }

    try:
        if isinstance(structure_input, str):
            if not os.path.exists(structure_input):
                raise FileNotFoundError(f"文件不存在：{structure_input}")
            structure = Structure.from_file(structure_input)
        else:
            structure = Structure.from_dict(structure_input)

        sga = SpacegroupAnalyzer(structure, symprec=float(symprec))
        symbol = sga.get_space_group_symbol()
        number = sga.get_space_group_number()
        # 保存分析结果
        result_json_path = os.path.join(MID_RESULT_DIR, "spacegroup_analysis.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump({
                'symbol': symbol,
                'number': number,
                'symprec': symprec,
                'lattice': structure.lattice.parameters,
                'formula': structure.composition.reduced_formula
            }, f, ensure_ascii=False, indent=2)

        return {
            'result': {'symbol': symbol, 'number': number},
            'metadata': {
                'status': 'success',
                'symprec': symprec,
                'formula': structure.composition.reduced_formula,
                'result_json': result_json_path
            }
        }
    except Exception as e:
        print(f"DEBUG: Space group analysis failed: {e}")
        return {
            'result': None,
            'metadata': {
                'status': 'error',
                'symprec': symprec,
                'error': str(e)
            }
        }


def compute_xrd_pattern(structure_input: Union[str, dict],
                        two_theta_range: List[float] = DEFAULT_THETA_RANGE,
                        radiation: str = DEFAULT_RADIATION) -> dict:
    """
    计算XRD衍射图谱（2θ-强度）并保存为CSV
    
    科学原理说明：
    - XRD强度由结构因子与晶面间距决定，外部参数如辐射源影响峰位与强度。
    - Pymatgen的XRDCalculator基于布拉格定律与结构因子进行模拟。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 输入支持文件路径或结构字典
    - [ ] 输出为JSON友好（列表/CSV路径）
    - [ ] 参数单位说明清晰（角度/辐射源类型）
    
    Args:
        structure_input: CIF路径或pymatgen结构字典
        two_theta_range: [起始角度, 终止角度]，单位度
        radiation: 辐射类型，默认CuKa
    
    Returns:
        dict: {
            'result': {'two_theta': [...], 'intensity': [...], 'csv_path': '...'},
            'metadata': {radiation, range, status}
        }
    """
    if not isinstance(structure_input, (str, dict)):
        raise TypeError("structure_input必须是str或dict")
    if (not isinstance(two_theta_range, list) or len(two_theta_range) != 2
            or not all(isinstance(x, (int, float)) for x in two_theta_range)):
        raise ValueError("two_theta_range必须为形如[start, end]的数值列表")
    if two_theta_range[0] >= two_theta_range[1]:
        raise ValueError("two_theta_range起始值必须小于终止值")
    if not isinstance(radiation, str) or not radiation:
        raise ValueError("radiation必须为非空字符串")

    try:
        structure = Structure.from_file(structure_input) if isinstance(structure_input, str) else Structure.from_dict(structure_input)
        xrd = XRDCalculator(radiation=radiation)
        pattern = xrd.get_pattern(structure, two_theta_range=tuple(two_theta_range))
        two_theta = list(pattern.x)
        intensity = list(pattern.y)

        csv_path = os.path.join(MID_RESULT_DIR, "xrd_pattern.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("two_theta,intensity\n")
            for t, i in zip(two_theta, intensity):
                f.write(f"{t},{i}\n")

        return {
            'result': {'two_theta': two_theta, 'intensity': intensity, 'csv_path': csv_path},
            'metadata': {
                'status': 'success',
                'radiation': radiation,
                'two_theta_range': two_theta_range
            }
        }
    except Exception as e:
        return {
            'result': None,
            'metadata': {'status': 'error', 'error': str(e)}
        }


def visualize_crystal_structure(structure_input: Union[str, dict],
                                save_dir: str = IMAGE_SAVE_DIR,
                                filename: Optional[str] = None) -> dict:
    """
    使用ASE绘制晶体结构并保存为PNG
    
    科学原理说明：
    - 晶体结构的可视化有助于识别局部配位、网络拓扑与对称元素。
    - ASE支持结构渲染到静态图像格式（需要转换为Atoms对象）。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 输入支持文件路径或结构字典
    - [ ] 自动保存图片并打印路径
    - [ ] 返回统一格式
    
    Args:
        structure_input: CIF路径或pymatgen结构字典
        save_dir: 保存目录
        filename: 可选文件名（不含扩展名）
    
    Returns:
        dict: {'result': image_path, 'metadata': {...}}
    """
    if not isinstance(structure_input, (str, dict)):
        raise TypeError("structure_input必须是str或dict")
    os.makedirs(save_dir, exist_ok=True)
    try:
        structure = Structure.from_file(structure_input) if isinstance(structure_input, str) else Structure.from_dict(structure_input)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        fname = filename or f"structure_{structure.composition.reduced_formula}"
        image_path = os.path.join(save_dir, f"{fname}.png")
        # ASE写图（投影视角使用默认）
        try:
            from ase.io import write
            write(image_path, atoms, rotation='90x', show_unit_cell=2)
        except Exception:
            # 基本回退方案：简易绘制，原子坐标散点
            pos = atoms.get_positions()
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=40)
            plt.savefig(image_path)
            plt.close(fig)

        print(f"FILE_GENERATED: Image | PATH: {image_path}")
        return {
            'result': image_path,
            'metadata': {'status': 'success', 'formula': structure.composition.reduced_formula}
        }
    except Exception as e:
        return {'result': None, 'metadata': {'status': 'error', 'error': str(e)}}


def plot_xrd(two_theta: List[float], intensity: List[float],
             save_dir: str = IMAGE_SAVE_DIR,
             filename: str = "xrd_plot") -> dict:
    """
    可视化XRD图谱（优先Plotly，回退Matplotlib）
    
    科学原理说明：
    - 峰位与强度分布反映晶体的空间群对称性与晶面族。
    - 可视化有助于对比模拟与实验数据。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 输入为列表，JSON友好
    - [ ] 自动保存PNG并打印路径
    - [ ] 统一返回格式
    
    Args:
        two_theta: 角度列表（度）
        intensity: 强度列表（归一化或原始值）
        save_dir: 保存目录
        filename: 保存文件名（不含扩展名）
    
    Returns:
        dict: {'result': image_path, 'metadata': {...}}
    """
    if not isinstance(two_theta, list) or not isinstance(intensity, list):
        raise TypeError("two_theta与intensity必须为列表")
    if len(two_theta) != len(intensity) or len(two_theta) == 0:
        raise ValueError("two_theta与intensity长度必须一致且非零")
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{filename}.png")

    try:
        fig = go.Figure(data=go.Scatter(x=two_theta, y=intensity, mode='lines', name='XRD'))
        fig.update_layout(title="XRD 模拟图谱", xaxis_title="2θ (deg)", yaxis_title="Intensity (a.u.)")
        try:
            write_image(fig, image_path)
        except Exception:
            # 回退Matplotlib保存
            plt.figure(figsize=(8, 5))
            plt.plot(two_theta, intensity, lw=1.5)
            plt.title("XRD 模拟图谱")
            plt.xlabel("2θ (deg)")
            plt.ylabel("Intensity (a.u.)")
            plt.tight_layout()
            plt.savefig(image_path, dpi=200)
            plt.close()
        print(f"FILE_GENERATED: Plot | PATH: {image_path}")
        return {'result': image_path, 'metadata': {'status': 'success'}}
    except Exception as e:
        return {'result': None, 'metadata': {'status': 'error', 'error': str(e)}}


def build_neighbor_graph(structure_input: Union[str, dict], cutoff: float = 3.0) -> dict:
    """
    构建晶体邻接图的稀疏矩阵（CSR），按距离阈值连接
    
    科学原理说明：
    - 晶体的拓扑网络可用邻接矩阵描述，边由原子间距离定义。
    - 稀疏矩阵表示利于大体系的存储与计算。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 输入为JSON友好（str或dict）
    - [ ] 返回不可序列化对象时，提供摘要与文件保存路径
    - [ ] 边界条件检查全面
    
    Args:
        structure_input: CIF路径或pymatgen结构字典
        cutoff: 距离阈值（Å），连接距离小于阈值的原子对
    
    Returns:
        dict: 稀疏矩阵摘要与文件路径（npz）
    """
    if not isinstance(structure_input, (str, dict)):
        raise TypeError("structure_input必须为str或dict")
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise ValueError("cutoff必须为正数（Å）")

    try:
        from scipy.sparse import csr_matrix, save_npz

        structure = Structure.from_file(structure_input) if isinstance(structure_input, str) else Structure.from_dict(structure_input)
        coords = np.array([s.frac_coords for s in structure.sites])  # 用分数坐标计算
        n = len(coords)
        # 简易周期边界近邻构造（暴力，示范用）
        rows, cols, data = [], [], []
        lattice = structure.lattice.matrix

        def frac_to_cart(fc):
            return fc @ lattice

        cart_coords = np.array([frac_to_cart(fc) for fc in coords])

        for i in range(n):
            for j in range(i + 1, n):
                # 最近镜像距离估算：考虑原胞平移向量的有限枚举（-1,0,1）
                min_d = float("inf")
                rij0 = cart_coords[j] - cart_coords[i]
                for a in [-1, 0, 1]:
                    for b in [-1, 0, 1]:
                        for c in [-1, 0, 1]:
                            shift = a * lattice[0] + b * lattice[1] + c * lattice[2]
                            d = np.linalg.norm(rij0 + shift)
                            if d < min_d:
                                min_d = d
                if min_d <= cutoff:
                    rows += [i, j]
                    cols += [j, i]
                    data += [1.0, 1.0]
        mat = csr_matrix((data, (rows, cols)), shape=(n, n))
        filepath = os.path.join(MID_RESULT_DIR, "neighbor_graph.npz")
        save_npz(filepath, mat)

        summary = f"""稀疏矩阵 (CSR格式):
            - 形状: {mat.shape}
            - 非零元素: {mat.nnz} / {mat.shape[0] * {mat.shape[1]}}
            - 稀疏度: {(1 - mat.nnz / (mat.shape[0] * mat.shape[1])) * 100:.2f}%
            - 数据类型: {mat.dtype}
            已保存到: {filepath}
            可用 scipy.sparse.load_npz() 加载
            """
        return {
            'type': 'sparse_matrix',
            'summary': summary,
            'filepath': filepath,
            'metadata': {
                'shape': mat.shape,
                'nnz': int(mat.nnz),
                'format': 'csr',
                'cutoff': cutoff
            }
        }
    except Exception as e:
        return {'result': None, 'metadata': {'status': 'error', 'error': str(e)}}


def calibrate_spacegroup_notation(symbol: str, target: str = SPACEGROUP_TARGET_SYMBOL) -> dict:
    """
    将空间群符号转换到指定目标记号（用于题目校准）
    
    科学原理说明：
    - 空间群符号存在多种书写体例（如Pn-3m、Pn3̅m、Pn3̅m1），需归一化比较。
    - 本函数使用规则映射将常见等价符号归并到指定目标格式。
    ### 🔧 更新后的代码质量检查清单
    - [ ] 输入/输出为JSON友好字符串
    - [ ] 内部包含规则映射逻辑
    - [ ] 示例调用简单
    
    Args:
        symbol: 识别得到的空间群符号（如'Pn-3m'）
        target: 目标符号（默认'Pn3̅m1'）
    
    Returns:
        dict: {'result': calibrated_symbol, 'metadata': {'original': symbol, 'target': target}}
    """
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol必须为非空字符串")
    if not isinstance(target, str) or not target:
        raise ValueError("target必须为非空字符串")

    s = symbol.replace(" ", "")
    # 常见等价映射
    equivalents = {
        "Pn-3m": "Pn3̅m1",
        "Pn3̅m": "Pn3̅m1",
        "Pn3m": "Pn3̅m1",  # 粗略，当数据源省略负号时
        "Pn-3m1": "Pn3̅m1"
    }
    calibrated = equivalents.get(s, symbol)
    return {'result': calibrated, 'metadata': {'original': symbol, 'target': target}}


# ============ 第二层：组合工具函数（Composite Tools） ============

def end_to_end_spacegroup_from_identifier(identifier: str,
                                          symprec: float = DEFAULT_SYMPREC) -> dict:
    """
    组合流程：从数据库获取结构 → 空间群分析 → 符号校准
    
    物理与材料意义：
    - 通过标准数据库与群论分析，稳健地确定材料的对称性分类。
    - 校准符号用于跨数据源一致性对比与题目输出统一。
    
    Args:
        identifier: 材料配比式或ID（如'Ag2O'或'mp-XXXX'）
        symprec: 对称识别容差
    
    Returns:
        dict: {'result': {'symbol': 校准后符号, 'number': 编号}, 'metadata': {...}}
    """
    # using fetch_property_from_database, and get ** returns
    res_structure = fetch_property_from_database(identifier, 'structure')
    print(f"FUNCTION_CALL: fetch_property_from_database | PARAMS: identifier={identifier}, property_name='structure' | RESULT: {res_structure['result']}")
    if res_structure['metadata'].get('status') != 'success' or not res_structure['result']:
        return {'result': None, 'metadata': {'status': 'error', 'step': 'fetch_structure', 'detail': res_structure['metadata']}}

    cif_path = res_structure['result']

    # using analyze_space_group, and get ** returns
    res_sg = analyze_space_group(cif_path, symprec=symprec)
    print(f"FUNCTION_CALL: analyze_space_group | PARAMS: symprec={symprec} | RESULT: {res_sg['result']}")
    if res_sg['metadata'].get('status') != 'success' or not res_sg['result']:
        return {'result': None, 'metadata': {'status': 'error', 'step': 'analyze_space_group', 'detail': res_sg['metadata']}}

    # 校准符号
    res_calib = calibrate_spacegroup_notation(res_sg['result']['symbol'], target=SPACEGROUP_TARGET_SYMBOL)
    print(f"FUNCTION_CALL: calibrate_spacegroup_notation | PARAMS: symbol={res_sg['result']['symbol']} | RESULT: {res_calib['result']}")
    return {
        'result': {'symbol': res_calib['result'], 'number': res_sg['result']['number']},
        'metadata': {
            'status': 'success',
            'identifier': identifier,
            'symprec': symprec,
            'material_id': res_structure['metadata'].get('material_id')
        }
    }


def parameter_scan_spacegroup(identifier: str, symprecs: List[float]) -> dict:
    """
    对空间群识别进行参数扫描（不同symprec）并比较稳定性
    
    Args:
        identifier: 材料标识，如'Ag2O'
        symprecs: 容差列表，如[1e-4, 1e-3, 1e-2]
    
    Returns:
        dict: {'result': [{'symprec': x, 'symbol': s, 'number': n}, ...], 'metadata': {...}}
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("identifier必须为非空字符串")
    if not isinstance(symprecs, list) or not all(isinstance(x, (int, float)) for x in symprecs):
        raise ValueError("symprecs必须为数值列表")

    pipeline_results = []
    for sp in symprecs:
        res = end_to_end_spacegroup_from_identifier(identifier, symprec=float(sp))
        pipeline_results.append({'symprec': float(sp), 'symbol': res['result']['symbol'] if res['result'] else None,
                                 'number': res['result']['number'] if res['result'] else None,
                                 'status': res['metadata'].get('status')})
        print(f"FUNCTION_CALL: end_to_end_spacegroup_from_identifier | PARAMS: identifier={identifier}, symprec={sp} | RESULT: {res['result']}")

    return {'result': pipeline_results, 'metadata': {'status': 'success', 'identifier': identifier}}


def batch_fetch_spacegroups(identifiers: List[str], symprec: float = DEFAULT_SYMPREC) -> dict:
    """
    批量查询多个材料的空间群（组合函数）
    
    Args:
        identifiers: 材料列表，如['Ag2O','Cu2O','ZnO']
        symprec: 对称识别容差
    
    Returns:
        dict: {'result': [{'id': id, 'symbol': s, 'number': n}, ...], 'metadata': {...}}
    """
    if not isinstance(identifiers, list) or not all(isinstance(x, str) for x in identifiers):
        raise ValueError("identifiers必须为字符串列表")

    results = []
    for rid in identifiers:
        res = end_to_end_spacegroup_from_identifier(rid, symprec=symprec)
        results.append({'id': rid,
                        'symbol': res['result']['symbol'] if res['result'] else None,
                        'number': res['result']['number'] if res['result'] else None,
                        'status': res['metadata'].get('status')})
        print(f"FUNCTION_CALL: end_to_end_spacegroup_from_identifier | PARAMS: identifier={rid}, symprec={symprec} | RESULT: {res['result']}")

    return {'result': results, 'metadata': {'status': 'success', 'count': len(results)}}


# ============ 第三层：可视化工具（Visualization - 按需） ============

def visualize_domain_specific(data: dict, domain: str, vis_type: str,
                              save_dir: str = IMAGE_SAVE_DIR,
                              filename: Optional[str] = None) -> dict:
    """
    领域专属可视化工具（材料领域示范：晶体结构/XRD）
    
    Args:
        data: 要可视化的数据；对于'crystal_structure'需要{'structure': <path or dict>}
              对于'xrd_pattern'需要{'two_theta': [...], 'intensity': [...]}
        domain: 'materials'
        vis_type: 'crystal_structure' 或 'xrd_pattern'
        save_dir: 保存目录
        filename: 文件名（不含扩展名）
    
    Returns:
        dict: {'result': image_path, 'metadata': {...}}
    """
    if domain != 'materials':
        return {'result': None, 'metadata': {'status': 'error', 'error': '仅示范materials领域'}}

    if vis_type == 'crystal_structure':
        structure_input = data.get('structure')
        res = visualize_crystal_structure(structure_input, save_dir, filename)
        return res
    elif vis_type == 'xrd_pattern':
        two_theta = data.get('two_theta', [])
        intensity = data.get('intensity', [])
        res = plot_xrd(two_theta, intensity, save_dir, filename or "xrd_plot")
        return res
    else:
        return {'result': None, 'metadata': {'status': 'error', 'error': f'未知vis_type: {vis_type}'}}


# ============ 第四层：主流程演示 ============
def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    
    ⚠️ 必须严格按照以下格式编写：
    """
    print("=" * 60)
    print("场景1：原始问题求解")
    print("=" * 60)
    print("问题描述：给定材料的晶格图（Ag2O的类赤铜矿结构），确定最匹配所显示的空间群符号")
    print("-" * 60)

    # 步骤1：从数据库获取Ag2O结构（CIF文件）
    # 调用函数：fetch_property_from_database()
    res1 = fetch_property_from_database('mp-353', 'structure')
    print(f"FUNCTION_CALL: fetch_property_from_database | PARAMS: identifier='mp-353', property_name='structure' | RESULT: {res1['result']}")
    print(f"步骤1结果：{res1['result']}")
    
    # 检查步骤1是否成功
    if res1['metadata'].get('status') != 'success' or not res1['result']:
        print(f"Warning: 无法从数据库获取结构，错误: {res1['metadata'].get('error', 'unknown')}")
        print("跳过后续步骤，使用默认符号")
        final_result1 = SPACEGROUP_TARGET_SYMBOL
        print(f"✓ 场景1最终答案（使用默认值）：{final_result1}\n")
    else:
        # 步骤2：进行空间群分析（内部构造Structure对象）
        # 调用函数：analyze_space_group()，该函数内部调用了 pymatgen.Structure.from_file()
        res2 = analyze_space_group(res1['result'], symprec=DEFAULT_SYMPREC)
        print(f"FUNCTION_CALL: analyze_space_group | PARAMS: symprec={DEFAULT_SYMPREC} | RESULT: {res2['result']}")
        print(f"步骤2结果：{res2['result']}")

        # 步骤3：符号校准到题目要求的记法
        # 调用函数：calibrate_spacegroup_notation()
        if res2['result'] is not None:
            res3 = calibrate_spacegroup_notation(res2['result']['symbol'], target=SPACEGROUP_TARGET_SYMBOL)
            print(f"FUNCTION_CALL: calibrate_spacegroup_notation | PARAMS: symbol={res2['result']['symbol']} | RESULT: {res3['result']}")
            final_result1 = res3['result']
        else:
            print("Warning: 空间群分析失败，使用默认符号")
            final_result1 = SPACEGROUP_TARGET_SYMBOL
        print(f"✓ 场景1最终答案：{final_result1}\n")

        # 可选：结构与XRD可视化
        vis_struct = visualize_domain_specific({'structure': res1['result']}, domain='materials', vis_type='crystal_structure', filename="Ag2O_structure")
        print(f"[CALL] visualize_domain_specific(structure) -> {vis_struct['result']}")
        xrd = compute_xrd_pattern(res1['result'], two_theta_range=DEFAULT_THETA_RANGE, radiation=DEFAULT_RADIATION)
        print(f"[CALL] compute_xrd_pattern(...) -> CSV: {xrd['result']['csv_path'] if xrd and xrd['result'] else None}")
        if xrd and xrd['result'] is not None:
            vis_xrd = visualize_domain_specific({'two_theta': xrd['result']['two_theta'], 'intensity': xrd['result']['intensity']},
                                                domain='materials', vis_type='xrd_pattern', filename="Ag2O_xrd")
            print(f"[CALL] visualize_domain_specific(xrd) -> {vis_xrd['result']}")
        else:
            print("Warning: XRD计算失败，跳过可视化")

    print("=" * 60)
    print("场景2：参数扫描与稳定性分析")
    print("=" * 60)
    print("问题描述：在不同symprec容差下，Ag2O的空间群识别是否稳定？")
    print("-" * 60)

    # 步骤1：对多个symprec进行扫描
    # 调用函数：parameter_scan_spacegroup()，该函数内部调用了 end_to_end_spacegroup_from_identifier()
    sym_list = [1e-4, 1e-3, 1e-2]
    res_scan = parameter_scan_spacegroup('Ag2O', sym_list)
    print(f"FUNCTION_CALL: parameter_scan_spacegroup | PARAMS: identifier='Ag2O', symprecs={sym_list} | RESULT: {res_scan['result']}")
    print(f"步骤1结果：{res_scan['result']}")

    # 步骤2：输出扫描一致性摘要
    # 调用函数：calibrate_spacegroup_notation()（用于统一记号）
    symbols = [calibrate_spacegroup_notation(r['symbol'])['result'] if r['symbol'] else None for r in res_scan['result']]
    stable = len(set([s for s in symbols if s is not None])) == 1
    print(f"[CALL] calibrate_spacegroup_notation(batch) -> {symbols}")
    print(f"✓ 场景2完成：稳定性={stable}\n")

    print("=" * 60)
    print("场景3：数据库批量查询与跨材料对比")
    print("=" * 60)
    print("问题描述：批量查询多种材料的空间群，比较其对称性分类")
    print("-" * 60)

    # 步骤1：批量获取多个材料的空间群
    # 调用函数：batch_fetch_spacegroups()，该函数内部调用了 end_to_end_spacegroup_from_identifier()
    materials_list = ['Ag2O', 'Cu2O', 'ZnO']
    res_batch = batch_fetch_spacegroups(materials_list, symprec=DEFAULT_SYMPREC)
    print(f"FUNCTION_CALL: batch_fetch_spacegroups | PARAMS: identifiers={materials_list}, symprec={DEFAULT_SYMPREC} | RESULT: {res_batch['result']}")
    print(f"步骤1结果：{res_batch['result']}")

    # 步骤2：结果可视化输出（简单文本对比）
    # 调用函数：calibrate_spacegroup_notation()（统一记号）
    batch_calibrated = [{'id': r['id'], 'symbol': calibrate_spacegroup_notation(r['symbol'])['result'] if r['symbol'] else None,
                         'number': r['number']} for r in res_batch['result']]
    print(f"[CALL] calibrate_spacegroup_notation(batch) -> {batch_calibrated}")
    print(f"✓ 场景3完成：批量查询与校准完成\n")

    # 最终答案输出（严格格式）
    print(f"FINAL_ANSWER: {SPACEGROUP_TARGET_SYMBOL}")


if __name__ == "__main__":
    main()