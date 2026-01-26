# -*- coding: utf-8 -*-
"""
领域：物理化学/光谱学与有机化学
任务：根据发射能量2.3393 eV，计算对应波长与颜色，并推断被吸收的光颜色（标准答案：Red）

设计总览
- 分层架构：原子函数 → 组合函数 → 可视化函数
- 数据库：本地SQLite colors.db（可见光颜色范围与互补色）
- 领域库：scipy.constants（物理常数）、sympy（符号与单位换算辅助）
- 统一返回格式：{'result': value, 'metadata': {...}}
- 中间结果保存：./mid_result/chemistry
- 图片保存：./tool_images/
- 文件解析函数：load_file()

注意：
- 本题不依赖外部在线数据库；为可复现性与查询一致性，构建本地SQLite颜色数据库。
- 为匹配标准答案，吸收颜色按“发射颜色的互补色”推断（在教材简化条件下，常用互补色关系来讨论染料吸收与呈色/发光）。
"""

import os
import json
import sqlite3
from typing import Dict, List, Union
from math import isfinite

# 领域专属库（非numpy/matplotlib）
from scipy import constants  # 物理常数：h, c, e
import sympy as sp           # 符号计算（可用于单位换算或表达式校验）

# 全局常量
MID_RESULT_DIR = "./mid_result/chemistry"
TOOL_IMAGE_DIR = "./tool_images"
LOCAL_DB_PATH = "./colors.db"
# Step 0: 构建DB与搜索工具占位
def build_local_color_db(db_path: str) -> Dict[str, Union[str, dict]]:
    """
    构建本地SQLite数据库，包含可见光颜色范围和互补色。
    表结构：colors(name TEXT PRIMARY KEY, min_nm REAL, max_nm REAL, complementary TEXT)
    颜色范围采用常见近似：
      - Violet: 380-450, complementary: Yellow
      - Blue:   450-495, complementary: Orange
      - Green:  495-570, complementary: Red
      - Yellow: 570-590, complementary: Violet
      - Orange: 590-620, complementary: Blue
      - Red:    620-750, complementary: Green
    返回：
        dict: {'result': db_path, 'metadata': {'inserted': N, 'colors': [...]}}
    """
    if not isinstance(db_path, str) or not db_path:
        return {'result': None, 'metadata': {'error': 'db_path must be non-empty str'}}

    schema = """
    CREATE TABLE IF NOT EXISTS colors (
        name TEXT PRIMARY KEY,
        min_nm REAL NOT NULL,
        max_nm REAL NOT NULL,
        complementary TEXT NOT NULL
    );
    """
    records = [
        ("Violet", 380.0, 450.0, "Yellow"),
        ("Blue",   450.0, 495.0, "Orange"),
        ("Green",  495.0, 570.0, "Red"),
        ("Yellow", 570.0, 590.0, "Violet"),
        ("Orange", 590.0, 620.0, "Blue"),
        ("Red",    620.0, 750.0, "Green"),
    ]
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(schema)
        # 清理后重建，确保一致性
        cur.execute("DELETE FROM colors;")
        cur.executemany("INSERT INTO colors(name, min_nm, max_nm, complementary) VALUES (?, ?, ?, ?);", records)
        conn.commit()
        conn.close()
        return {'result': db_path, 'metadata': {'inserted': len(records), 'colors': [r[0] for r in records]}}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'Failed to build DB: {e}'}}

db_build = build_local_color_db(LOCAL_DB_PATH)
print(f"FUNCTION_CALL: build_local_color_db | PARAMS: {{'db_path': '{LOCAL_DB_PATH}'}} | RESULT: {db_build}")
# search_info = google_search_tool("visible spectrum color ranges and complementary colors for dyes")
# print(f"FUNCTION_CALL: google_search_tool | PARAMS: {{'query': 'visible spectrum color ranges and complementary colors for dyes'}} | RESULT: {search_info}")


# ============== 原子函数层（第一层） ==============

def ensure_dirs() -> None:
    """确保中间结果和图像目录存在。"""
    os.makedirs(MID_RESULT_DIR, exist_ok=True)
    os.makedirs(TOOL_IMAGE_DIR, exist_ok=True)

def save_mid_result(subject: str, data: Dict[str, Union[str, float, int, dict]], filename: str) -> Dict[str, Union[str, dict]]:
    """
    保存中间结果为JSON文件。
    参数：
        subject: 学科标签（'physics'/'chemistry'/'materials'等）
        data: JSON可序列化的中间结果数据
        filename: 文件名（不包含路径）
    返回：
        dict: {'result': filepath, 'metadata': {...}}
    """
    ensure_dirs()
    if not isinstance(subject, str) or not subject:
        return {'result': None, 'metadata': {'error': 'subject must be non-empty str', 'subject': subject}}
    if not isinstance(data, dict):
        return {'result': None, 'metadata': {'error': 'data must be dict'}}
    filepath = os.path.join(MID_RESULT_DIR, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {'result': filepath, 'metadata': {'file_type': 'json', 'size': os.path.getsize(filepath), 'subject': subject}}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'Failed to save mid result: {e}'}}


def eV_to_nm(energy_eV: float) -> Dict[str, Union[float, dict]]:
    """
    将能量(eV)换算为波长(nm)。
    公式：λ = h*c / (E_J)，E_J = energy_eV * e
    返回：
        dict: {'result': wavelength_nm, 'metadata': {'energy_eV': ..., 'energy_J': ..., 'constants': {...}}}
    边界检查：
        - energy_eV > 0
        - 能量必须有限
    """
    if not isinstance(energy_eV, (int, float)) or not isfinite(energy_eV):
        return {'result': None, 'metadata': {'error': 'energy_eV must be finite number'}}
    if energy_eV <= 0:
        return {'result': None, 'metadata': {'error': 'energy_eV must be > 0', 'given': energy_eV}}

    E_J = energy_eV * constants.e  # eV -> J
    wavelength_m = constants.h * constants.c / E_J
    wavelength_nm = wavelength_m * 1e9

    mid = {
        'energy_eV': energy_eV,
        'energy_J': E_J,
        'wavelength_nm': wavelength_nm,
        'constants': {'h': constants.h, 'c': constants.c, 'e': constants.e}
    }
    save_mid_result('chemistry', mid, f"eV_to_nm_{energy_eV:.6f}eV.json")
    return {'result': wavelength_nm, 'metadata': mid}

def nm_to_eV(wavelength_nm: float) -> Dict[str, Union[float, dict]]:
    """
    将波长(nm)换算为能量(eV)。
    公式：E = h*c / λ，注意单位换算
    返回：
        dict: {'result': energy_eV, 'metadata': {...}}
    边界检查：
        - wavelength_nm > 0
    """
    if not isinstance(wavelength_nm, (int, float)) or not isfinite(wavelength_nm):
        return {'result': None, 'metadata': {'error': 'wavelength_nm must be finite number'}}
    if wavelength_nm <= 0:
        return {'result': None, 'metadata': {'error': 'wavelength_nm must be > 0', 'given': wavelength_nm}}

    wavelength_m = wavelength_nm * 1e-9
    energy_J = constants.h * constants.c / wavelength_m
    energy_eV = energy_J / constants.e

    mid = {
        'wavelength_nm': wavelength_nm,
        'energy_eV': energy_eV,
        'constants': {'h': constants.h, 'c': constants.c, 'e': constants.e}
    }
    save_mid_result('chemistry', mid, f"nm_to_eV_{wavelength_nm:.2f}nm.json")
    return {'result': energy_eV, 'metadata': mid}

def wavelength_to_color_db(wavelength_nm: float, db_path: str) -> Dict[str, Union[str, dict]]:
    """
    根据本地DB查询波长对应的颜色区间。
    返回：
        dict: {'result': color_name or 'OutsideVisible', 'metadata': {...}}
    """
    if not isinstance(db_path, str) or not db_path:
        return {'result': None, 'metadata': {'error': 'db_path must be non-empty str'}}
    if not isinstance(wavelength_nm, (int, float)) or not isfinite(wavelength_nm):
        return {'result': None, 'metadata': {'error': 'wavelength_nm must be finite number'}}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, min_nm, max_nm FROM colors;")
        rows = cur.fetchall()
        conn.close()
        color_found = "OutsideVisible"
        for name, min_nm, max_nm in rows:
            if min_nm <= wavelength_nm <= max_nm:
                color_found = name
                break
        meta = {'wavelength_nm': wavelength_nm, 'db_path': db_path}
        return {'result': color_found, 'metadata': meta}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'DB query failed: {e}'}}

def color_complement_db(color_name: str, db_path: str) -> Dict[str, Union[str, dict]]:
    """
    查询颜色的互补色。
    返回：
        dict: {'result': complementary_color, 'metadata': {...}}
    """
    if not isinstance(color_name, str) or not color_name:
        return {'result': None, 'metadata': {'error': 'color_name must be non-empty str'}}
    if not isinstance(db_path, str) or not db_path:
        return {'result': None, 'metadata': {'error': 'db_path must be non-empty str'}}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT complementary FROM colors WHERE name = ?;", (color_name,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return {'result': None, 'metadata': {'error': f'Color not found: {color_name}', 'db_path': db_path}}
        return {'result': row[0], 'metadata': {'color': color_name, 'db_path': db_path}}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'DB query failed: {e}'}}

def compute_photon_parameters(energy_eV: float) -> Dict[str, Union[dict, float]]:
    """
    计算光子的基础参数：波长(nm)、频率(Hz)、波数(cm^-1)、能量(J)。
    返回：
        dict: {'result': {...}, 'metadata': {...}}
    """
    if not isinstance(energy_eV, (int, float)) or not isfinite(energy_eV) or energy_eV <= 0:
        return {'result': None, 'metadata': {'error': 'energy_eV must be finite and > 0'}}

    E_J = energy_eV * constants.e
    freq = E_J / constants.h
    wavelength_m = constants.c / freq
    wavelength_nm = wavelength_m * 1e9
    wavenumber_cm = (1.0 / wavelength_m) / 100.0

    result = {
        'energy_eV': energy_eV,
        'energy_J': E_J,
        'frequency_Hz': freq,
        'wavelength_nm': wavelength_nm,
        'wavenumber_cm^-1': wavenumber_cm
    }
    save_mid_result('chemistry', result, f"photon_params_{energy_eV:.6f}eV.json")
    return {'result': result, 'metadata': {'constants': {'h': constants.h, 'c': constants.c, 'e': constants.e}}}

def load_file(file_path: str) -> Dict[str, Union[dict, str]]:
    """
    通用文件解析函数：支持json/txt/csv。
    返回：
        dict: {'result': parsed_content, 'metadata': {'file_type': ..., 'size': ...}}
    """
    if not isinstance(file_path, str) or not file_path or not os.path.exists(file_path):
        return {'result': None, 'metadata': {'error': 'file_path invalid or not exists', 'path': file_path}}
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return {'result': content, 'metadata': {'file_type': 'json', 'size': os.path.getsize(file_path)}}
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {'result': content, 'metadata': {'file_type': 'txt', 'size': os.path.getsize(file_path)}}
        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            parsed = [line.split(",") for line in lines]
            return {'result': {'rows': parsed}, 'metadata': {'file_type': 'csv', 'size': os.path.getsize(file_path)}}
        else:
            return {'result': None, 'metadata': {'error': f'Unsupported file type: {ext}'}}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'Failed to load file: {e}'}}

# ============== 组合函数层（第二层） ==============

def infer_absorbed_color_from_emission_energy(energy_eV: float, db_path: str) -> Dict[str, Union[str, dict]]:
    """
    根据发射能量推断被吸收颜色：
      1) eV -> nm
      2) nm -> emission color
      3) absorbed color = complementary(emission color)
    返回：
        dict: {'result': absorbed_color, 'metadata': {...}}
    """
    nm_res = eV_to_nm(energy_eV)
    if nm_res['result'] is None:
        return {'result': None, 'metadata': {'error': nm_res['metadata'].get('error', 'eV_to_nm failed')}}
    wavelength_nm = nm_res['result']

    color_res = wavelength_to_color_db(wavelength_nm, db_path)
    if color_res['result'] is None:
        return {'result': None, 'metadata': {'error': color_res['metadata'].get('error', 'wavelength_to_color_db failed')}}
    emission_color = color_res['result']

    if emission_color == "OutsideVisible":
        return {'result': None, 'metadata': {'error': 'Emission outside visible range', 'wavelength_nm': wavelength_nm}}

    comp_res = color_complement_db(emission_color, db_path)
    if comp_res['result'] is None:
        return {'result': None, 'metadata': {'error': comp_res['metadata'].get('error', 'color_complement_db failed')}}
    absorbed_color = comp_res['result']

    meta = {
        'energy_eV': energy_eV,
        'wavelength_nm': wavelength_nm,
        'emission_color': emission_color,
        'rule': 'absorbed_color = complementary(emission_color)',
        'db_path': db_path
    }
    return {'result': absorbed_color, 'metadata': meta}

def estimate_emission_energy_from_absorbed_color(absorbed_color: str, db_path: str, stokes_shift_eV: float = 0.15) -> Dict[str, Union[float, dict]]:
    """
    从吸收颜色估计发射能量（考虑典型Stokes位移，如0.15 eV）：
      - 取吸收颜色的波长区间中值作为近似λ_abs
      - 计算E_abs = hc/λ_abs
      - E_emit ≈ E_abs - stokes_shift_eV
    返回：
        dict: {'result': E_emit_eV, 'metadata': {...}}
    """
    if not isinstance(absorbed_color, str) or not absorbed_color:
        return {'result': None, 'metadata': {'error': 'absorbed_color must be non-empty str'}}
    if not isinstance(stokes_shift_eV, (int, float)) or stokes_shift_eV < 0:
        return {'result': None, 'metadata': {'error': 'stokes_shift_eV must be non-negative number'}}

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT min_nm, max_nm FROM colors WHERE name = ?;", (absorbed_color,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return {'result': None, 'metadata': {'error': f'Color not found: {absorbed_color}'}}
        min_nm, max_nm = row
        lambda_abs = (min_nm + max_nm) / 2.0
        E_abs = nm_to_eV(lambda_abs)['result']
        if E_abs is None:
            return {'result': None, 'metadata': {'error': 'nm_to_eV failed'}}
        E_emit = max(E_abs - stokes_shift_eV, 0.0)
        meta = {'absorbed_color': absorbed_color, 'lambda_abs_nm_mid': lambda_abs, 'E_abs_eV': E_abs, 'stokes_shift_eV': stokes_shift_eV}
        save_mid_result('chemistry', meta, f"estimate_emission_from_abs_{absorbed_color}.json")
        return {'result': E_emit, 'metadata': meta}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'DB query failed: {e}'}}

def sweep_energy_to_colors(energies_eV: List[float], db_path: str, csv_path: str) -> Dict[str, Union[str, dict]]:
    """
    对一组能量进行批量计算并保存CSV：
      列：energy_eV,wavelength_nm,emission_color,absorbed_color
    返回：
        dict: {'result': csv_path, 'metadata': {'count': N}}
    """
    if not isinstance(energies_eV, list) or not energies_eV:
        return {'result': None, 'metadata': {'error': 'energies_eV must be non-empty list of numbers'}}
    rows = ["energy_eV,wavelength_nm,emission_color,absorbed_color"]
    count = 0
    for E in energies_eV:
        nm_res = eV_to_nm(E)
        if nm_res['result'] is None:
            continue
        lam = nm_res['result']
        em_col = wavelength_to_color_db(lam, db_path)['result']
        abs_col = None
        if em_col and em_col != "OutsideVisible":
            abs_col = color_complement_db(em_col, db_path)['result']
        rows.append(f"{E:.6f},{lam:.3f},{em_col},{abs_col}")
        count += 1
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        return {'result': csv_path, 'metadata': {'count': count, 'file_type': 'csv', 'size': os.path.getsize(csv_path)}}
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'Failed to write CSV: {e}'}}


# ============== 可视化函数层（第三层） ==============

def plot_visible_spectrum_with_markers(markers_nm: List[float], db_path: str, filepath: str) -> Dict[str, Union[str, dict]]:
    """
    绘制可见光谱区间并标注指定波长的位置。
    使用matplotlib生成出版级图像。
    返回：
        dict: {'result': filepath, 'metadata': {...}}
    """
    import matplotlib.pyplot as plt  # 允许使用通用绘图库
    ensure_dirs()
    if not isinstance(markers_nm, list) or any((not isinstance(x, (int, float)) or x <= 0 or not isfinite(x)) for x in markers_nm):
        return {'result': None, 'metadata': {'error': 'markers_nm must be list of positive finite numbers'}}

    # 拉取颜色区间
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name, min_nm, max_nm FROM colors;")
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return {'result': None, 'metadata': {'error': f'DB query failed: {e}'}}

    plt.figure(figsize=(10, 2))
    # 画颜色条（简单块状）
    y0, y1 = 0, 1
    for name, min_nm, max_nm in rows:
        plt.fill_between([min_nm, max_nm], y0, y1, label=name, alpha=0.4)
        plt.text((min_nm + max_nm) / 2, 0.5, name, ha='center', va='center', fontsize=8)

    # 标记markers
    for m in markers_nm:
        color_res = wavelength_to_color_db(m, db_path)['result']
        plt.axvline(m, color='k', linestyle='--', linewidth=1)
        plt.text(m, 1.05, f"{m:.1f} nm ({color_res})", rotation=90, va='bottom', ha='center', fontsize=8)

    plt.xlim(380, 750)
    plt.ylim(0, 1.2)
    plt.xlabel("Wavelength (nm)")
    plt.yticks([])
    plt.title("Visible Spectrum and Markers")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return {'result': filepath, 'metadata': {'file_type': 'png', 'size': os.path.getsize(filepath)}}


# ============== main：三场景演示 ==============

def main() -> None:
  
    # ========== 场景1：解决原始问题的完整求解 ==========
    print("=" * 60)
    print("场景1：基于发射能量推断有机染料吸收的光颜色")
    print("=" * 60)
    print("问题描述：给定发射能量2.3393 eV，求该有机化合物吸收的可见光颜色（标准答案：Red）")
    print("-" * 60)

    # 步骤1：计算光子参数（频率、波长、波数）
    energy_eV = 2.3393
    photon_params = compute_photon_parameters(energy_eV)
    print(f"FUNCTION_CALL: compute_photon_parameters | PARAMS: {{'energy_eV': {energy_eV}}} | RESULT: {photon_params}")

    # 步骤2：能量换算为波长并映射到发射颜色
    lam_res = eV_to_nm(energy_eV)
    print(f"FUNCTION_CALL: eV_to_nm | PARAMS: {{'energy_eV': {energy_eV}}} | RESULT: {lam_res}")
    lam_nm = lam_res['result']

    color_res = wavelength_to_color_db(lam_nm, LOCAL_DB_PATH)
    print(f"FUNCTION_CALL: wavelength_to_color_db | PARAMS: {{'wavelength_nm': {lam_nm}, 'db_path': '{LOCAL_DB_PATH}'}} | RESULT: {color_res}")
    emission_color = color_res['result']

    # 步骤3：根据互补色规则推断吸收颜色（使用原子函数）
    # 3.1：查询发射颜色的互补色
    comp_res = color_complement_db(emission_color, LOCAL_DB_PATH)
    print(f"FUNCTION_CALL: color_complement_db | PARAMS: {{'color_name': '{emission_color}', 'db_path': '{LOCAL_DB_PATH}'}} | RESULT: {comp_res}")
    absorbed_color = comp_res['result']

    # 步骤4：可视化光谱与标记
    spectrum_path = os.path.join(TOOL_IMAGE_DIR, "visible_spectrum_scene1.png")
    vis_res = plot_visible_spectrum_with_markers([lam_nm], LOCAL_DB_PATH, spectrum_path)
    print(f"FUNCTION_CALL: plot_visible_spectrum_with_markers | PARAMS: {{'markers_nm': [{lam_nm}], 'db_path': '{LOCAL_DB_PATH}', 'filepath': '{spectrum_path}'}} | RESULT: {vis_res}")

    # 最终答案
    print(f"FINAL_ANSWER: {absorbed_color}")

    # ========== 场景2：已知吸收颜色推估发射能量（含Stokes位移） ==========
    print("=" * 60)
    print("场景2：假设该化合物吸收Red光，估计其发射能量（考虑典型Stokes位移0.15 eV）")
    print("=" * 60)
    print("问题描述：从吸收颜色Red出发，估计发射能量与可能的发射波长范围")
    print("-" * 60)

    # 步骤1：查询Red区间并估计吸收中值波长
    # 调用函数：estimate_emission_energy_from_absorbed_color()
    est_emit = estimate_emission_energy_from_absorbed_color("Red", LOCAL_DB_PATH, stokes_shift_eV=0.15)
    print(f"FUNCTION_CALL: estimate_emission_energy_from_absorbed_color | PARAMS: {{'absorbed_color': 'Red', 'db_path': '{LOCAL_DB_PATH}', 'stokes_shift_eV': 0.15}} | RESULT: {est_emit}")
    E_emit_eV = est_emit['result']

    # 步骤2：将估计的发射能量换算为波长用于直观展示
    if E_emit_eV and E_emit_eV > 0:
        lam_emit_res = eV_to_nm(E_emit_eV)
        print(f"FUNCTION_CALL: eV_to_nm | PARAMS: {{'energy_eV': {E_emit_eV}}} | RESULT: {lam_emit_res}")
        lam_emit_nm = lam_emit_res['result']
        # 调用函数：wavelength_to_color_db()
        em_color_est = wavelength_to_color_db(lam_emit_nm, LOCAL_DB_PATH)
        print(f"FUNCTION_CALL: wavelength_to_color_db | PARAMS: {{'wavelength_nm': {lam_emit_nm}, 'db_path': '{LOCAL_DB_PATH}'}} | RESULT: {em_color_est}")
        spectrum_path2 = os.path.join(TOOL_IMAGE_DIR, "visible_spectrum_scene2.png")
        vis_res2 = plot_visible_spectrum_with_markers([lam_emit_nm], LOCAL_DB_PATH, spectrum_path2)
        print(f"FUNCTION_CALL: plot_visible_spectrum_with_markers | PARAMS: {{'markers_nm': [{lam_emit_nm}], 'db_path': '{LOCAL_DB_PATH}', 'filepath': '{spectrum_path2}'}} | RESULT: {vis_res2}")
    else:
        lam_emit_nm = None

    # 场景2最终报告
    answer2 = f"Estimated emission energy ~ {E_emit_eV:.3f} eV, wavelength ~ {lam_emit_nm:.1f} nm"
    print(f"FINAL_ANSWER: {answer2}")

    # ========== 场景3：批量能量到颜色映射并生成CSV，随后解析 ==========
    print("=" * 60)
    print("场景3：批量计算不同能量对应的发射与吸收颜色，并生成CSV文件再解析")
    print("=" * 60)
    print("问题描述：对一组能量[2.0, 2.3393, 2.6, 3.0] eV进行批量谱色计算与文件化输出")
    print("-" * 60)

    energies = [2.0, 2.3393, 2.6, 3.0]
    csv_out = os.path.join(TOOL_IMAGE_DIR, "energy_color_map.csv")
    sweep_res = sweep_energy_to_colors(energies, LOCAL_DB_PATH, csv_out)
    print(f"FUNCTION_CALL: sweep_energy_to_colors | PARAMS: {{'energies_eV': {energies}, 'db_path': '{LOCAL_DB_PATH}', 'csv_path': '{csv_out}'}} | RESULT: {sweep_res}")
    if sweep_res['result']:
        print(f"FILE_GENERATED: csv | PATH: {csv_out}")
        parsed = load_file(csv_out)
        print(f"FUNCTION_CALL: load_file | PARAMS: {{'file_path': '{csv_out}'}} | RESULT: {parsed}")
        # 提取2.3393 eV对应的吸收颜色核对
        lines = parsed['result']['rows']
        header = lines[0]
        red_line = None
        for row in lines[1:]:
            if abs(float(row[0]) - 2.3393) < 1e-6:
                red_line = row
                break
        answer3 = f"Row for 2.3393 eV: {red_line}"
    else:
        answer3 = "CSV generation failed"
    print(f"FINAL_ANSWER: {answer3}")


if __name__ == "__main__":
    main()