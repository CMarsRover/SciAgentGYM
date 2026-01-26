# 任务目标
基于同一领域内的多道问题与其参考答案（问题-答案对），生成可用工具集，构建一个可泛化到该领域内多个任务的计算与推理工具包库（Python），同时支持：
# 要求考虑
1. 通用算法与数学原理的抽象与复用；
    多问题批量解析、求解与校验（与答案对比）；
    可选的数据知识库（公式库、常量库、经验规则库）构建与调用；
    针对多个子领域场景的可扩展模块化设计（至少三个关键子领域/任务场景）。
2. 输入与适配范围
    输入包括：成组的问题-答案对（Q-A pairs）、该领域的背景知识要点、可用的外部工具/库清单、可能的单位与数据规范。
    目标领域为上轮对话指定的关键领域；需面向该领域内的至少三个子方向构建统一的工具包（例如：电化学定量、物理化学热力学、分子结构与MO理论；或“同领域内的三个典型计算场景”）。
3. 工具包需在 main 中展示对多道问题的批量求解过程。
# 执行要求
严格按照下面流程输出完整的 Python 代码文件（集成工具包），并确保可运行。
1. 数学与领域原理剖析（面向多问题）
统一梳理该领域的核心数学与物理化学原理、公式与定律，形成“公式库/规则库”数据结构（可序列化或字典化存储）。
针对不同子场景抽象共性求解路径；明确每个计算步骤需要的数学工具与单位规范。
形成面向多问题的推理框架（例如：解析→归类→单位校准→参数提取→计算→不确定度/数值稳定性检查→结果校验）。
2. 核心算法抽象（复用与扩展）
识别通用计算原语（如：能斯特方程、亥斯定律、理想气体状态方程、摩尔/电量换算、分子轨道电子计数与键级等），设计可复用的函数。
设计面向多问题的解析与调度层：支持批量处理、根据问题类型自动选择计算模块、并输出结构化结果。
分析与采用必要的 Python 科学计算库（如 numpy、scipy）；可视化为可选且最简。
避免类过度设计，优先以函数+轻量数据结构（dict, dataclass 可选）组织。
3. 代码实现标准（工具化与通用化）
编写高质量工具函数：完整文档字符串、参数通用化、无硬编码特定数值。
函数需包含：参数说明（单位/范围/物理意义）、返回值结构说明（含单位与异常约定）、使用示例。
设计统一的单位处理与校验工具（至少包含：体积、压力、温度、电荷、浓度、能量、电位等常用维度的转换）。
设计最小可用的公式/常量数据知识库（字典或 JSON 可序列化），供多个模块调用与扩展。
4. main 函数中：
展示批处理求解流程；
将计算结果与给定答案进行对比（容差可配置），并输出评估摘要；
可选：对一个子任务给出最简可视化（如一条曲线或柱状图），并确保字体兼容设置可用但默认不开启。
📌 代码模板规范
# Filename: <tool_name>.py
# （注意：<tool_name> 必须与该领域工具主题相称，如 physical_chemistry_multi_solver）
```python
## physical_chemistry_multi_solver.py
from typing import Dict, Any, List, Tuple, Optional

# ========== 0. 常量与配置 ==========
FORMULAE_DB = {
    # 示例：可扩展的公式库/规则库
    "ideal_gas": "n = P*V/(R*T)",
    "faraday": "n_moles = Q/(z*F)",
    "nernst": "E = E0 + (RT/(zF)) * ln(activity)",
    "hess": "DeltaH_total = sum(DeltaH_steps)",
    "bond_order": "BO = (N_bonding - N_antibonding)/2",
}
CONSTANTS = {
    "R": 8.314462618,      # J/(mol·K)
    "F": 96485.33212,      # C/mol
    "ln10": 2.302585093,
    "T_STD": 298.15,       # K
}
DEFAULTS = {
    "tolerance_rel": 1e-3,
    "tolerance_abs": 1e-6,
    "nernst_log_base": 10,  # 或  e，自适配
}

# ========== 1. 单位与数值工具 ==========
def to_si_pressure(value: float, unit: str = "Pa") -> float:
    """
    将压力转换为 Pa。
    支持单位：Pa, kPa, bar, atm, torr, mmHg。
    """
    unit = unit.lower()
    factors = {"pa":1.0, "kpa":1e3, "bar":1e5, "atm":101325.0, "torr":133.322368, "mmhg":133.322368}
    if unit not in factors:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value * factors[unit]

def to_si_volume(value: float, unit: str = "m3") -> float:
    """
    将体积转换为 m^3。
    支持单位：m3, L, mL, uL, nL, pL, cm3。
    """
    unit = unit.lower()
    factors = {"m3":1.0, "l":1e-3, "ml":1e-6, "ul":1e-9, "nl":1e-12, "pl":1e-15, "cm3":1e-6}
    if unit not in factors:
        raise ValueError(f"Unsupported volume unit: {unit}")
    return value * factors[unit]

def to_si_temperature(value: float, unit: str = "K") -> float:
    """
    将温度转换为 K。
    支持单位：K, C。
    """
    unit = unit.lower()
    if unit == "k":
        return value
    if unit == "c":
        return value + 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")

def close_enough(x: float, y: float, tol_abs: float = DEFAULTS["tolerance_abs"], tol_rel: float = DEFAULTS["tolerance_rel"]) -> bool:
    """
    判断两个数值是否在给定容差内相等。
    """
    return abs(x - y) <= max(tol_abs, tol_rel * max(1.0, abs(y)))

# ========== 2. 通用计算原语 ==========
def ideal_gas_moles(P: float, V: float, T: float, P_unit="Pa", V_unit="m3", T_unit="K", R: float = CONSTANTS["R"]) -> float:
    """
    理想气体方程：n = P*V/(R*T)
    Parameters:
    -----------
    P : float
        压力数值
    V : float
        体积数值
    T : float
        温度数值
    P_unit, V_unit, T_unit : str
        单位（支持见各自转换函数）
    R : float
        气体常数，单位 J/(mol·K)
    Returns:
    --------
    float
        物质的量（mol）
    """
    P_si = to_si_pressure(P, P_unit)
    V_si = to_si_volume(V, V_unit)
    T_si = to_si_temperature(T, T_unit)
    return (P_si * V_si) / (R * T_si)

def faraday_moles(Q: float, z: int = 1, F: float = CONSTANTS["F"]) -> float:
    """
    法拉第定律：根据电量计算物质的量。
    Parameters:
    -----------
    Q : float
        电荷量（C）
    z : int
        电子转移数
    F : float
        法拉第常数（C/mol）
    Returns:
    --------
    float
        物质的量（mol）
    """
    if z <= 0:
        raise ValueError("z must be positive.")
    return Q / (z * F)

def nernst_potential(E0: float, z: int, activity: float, T: float = CONSTANTS["T_STD"], log_base: str = "10") -> float:
    """
    能斯特方程计算电极电位。
    E = E0 + (RT/(zF)) * ln(a) 或 E0 + (0.05916/z) * log10(a) 在 298 K。
    Parameters:
    -----------
    E0 : float
        标准电极电位（V）
    z : int
        电子数
    activity : float
        活度或有效浓度（无量纲）
    T : float
        温度（K）
    log_base : {"e","10"}
        选择自然对数或常用对数形式
    Returns:
    --------
    float
        电位（V）
    """
    if z == 0:
        raise ValueError("z must be nonzero.")
    if activity <= 0:
        raise ValueError("activity must be positive.")
    if log_base == "e":
        from math import log
        return E0 + (CONSTANTS["R"] * T) / (z * CONSTANTS["F"]) * log(activity)
    else:
        # 默认 298 K 的便捷常数
        return E0 + (0.05916 / z) * __import__("math").log10(activity)

def hess_sum(delta_h_steps: List[float]) -> float:
    """
    亥斯定律：总焓变等于各步焓变之和。
    """
    return sum(delta_h_steps)

def bond_order(n_bonding: int, n_antibonding: int) -> float:
    """
    分子轨道法键级计算。
    """
    return 0.5 * (n_bonding - n_antibonding)

# ========== 3. 问题解析与调度 ==========
def classify_problem(problem_text: str) -> str:
    """
    将问题文本粗分类为某一子任务类型。
    返回类型标签，例如："ideal_gas", "faraday", "nernst", "hess", "bond_order"。
    规则可扩展：基于关键词或轻量正则。
    """
    t = problem_text.lower()
    if any(k in t for k in ["kpa", "atm", "torr", "cm3", "ideal gas", "容器", "气体方程", "压力", "体积"]):
        return "ideal_gas"
    if any(k in t for k in ["stripping", "faraday", "电量", "库仑", "charge", "coulomb"]):
        return "faraday"
    if "nernst" in t or "能斯特" in t or "log" in t and "电位" in t:
        return "nernst"
    if "Δh" in t.lower() or "hess" in t or "焓" in t:
        return "hess"
    if "bond order" in t or "键级" in t or "mo" in t:
        return "bond_order"
    return "unknown"

def solve_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一求解入口。
    problem 需包含：
      - "text": 问题文本
      - "data": 参数字典（可选）
      - "answer": 参考答案（可选，用于校验）
    返回结构化结果：{"type","result","units","ok","details"}
    """
    ptype = classify_problem(problem.get("text", ""))
    data = problem.get("data", {})
    result = None
    units = None
    details: Dict[str, Any] = {"ptype": ptype}

    try:
        if ptype == "ideal_gas":
            # 期望 keys: P, V, T, P_unit, V_unit, T_unit
            n = ideal_gas_moles(
                data["P"], data["V"], data["T"],
                data.get("P_unit","Pa"), data.get("V_unit","m3"), data.get("T_unit","K")
            )
            result, units = n, "mol"
        elif ptype == "faraday":
            # 期望 keys: Q, z
            n = faraday_moles(data["Q"], data.get("z", 1))
            if "V_particle" in data:
                # 可选计算浓度
                c = n / to_si_volume(data["V_particle"], data.get("V_unit","m3"))  # mol/m3
                details["concentration_mol_per_m3"] = c
                details["concentration_M"] = c / 1000.0
            result, units = n, "mol"
        elif ptype == "nernst":
            E = nernst_potential(
                data["E0"], data["z"], data.get("activity", data.get("C", 1.0)),
                data.get("T", CONSTANTS["T_STD"]), data.get("log_base","10")
            )
            # 过电位修正（可选）
            if "overpotential" in data:
                E_eff = E - data["overpotential"]
                details["E_eff"] = E_eff
                result = E_eff
            else:
                result = E
            units = "V"
        elif ptype == "hess":
            result = hess_sum(data["DeltaH_steps"])
            units = "kJ" if data.get("units","kJ") == "kJ" else data.get("units","J")
        elif ptype == "bond_order":
            result = bond_order(data["N_bonding"], data["N_antibonding"])
            units = "dimensionless"
        else:
            details["note"] = "Unknown problem type; no computation performed."
            result, units = None, None
    except KeyError as e:
        details["error"] = f"Missing parameter: {e}"
    except Exception as e:
        details["error"] = str(e)

    # 校验
    ref = problem.get("answer", None)
    ok = None
    if ref is not None and isinstance(result, (int, float)):
        ok = close_enough(result, float(ref))
    return {"type": ptype, "result": result, "units": units, "ok": ok, "details": details}

# ========== 4. 示例数据集与批量运行 ==========
def example_dataset() -> List[Dict[str, Any]]:
    """
    构造覆盖多个子领域的示例问题集（问题-答案对可选）。
    """
    return [
        {
            "text": "Ideal gas: NO collected in 250.0 cm3 at 24.5 kPa and 19.5 C. Find moles.",
            "data": {"P":24.5, "P_unit":"kPa", "V":250.0, "V_unit":"cm3", "T":19.5, "T_unit":"C"},
            "answer": 2.51e-3
        },
        {
            "text": "Faraday: Li stripping Q=85 nC, z=1, particle volume 1 pL. Compute moles and concentration.",
            "data": {"Q":85e-9, "z":1, "V_particle":1.0, "V_unit":"pL"},
            "answer": 85e-9/96485.33212
        },
        {
            "text": "Nernst with overpotential: Cu2+ at 1.0 M, E0=0.34 V, z=2, T=298.15 K, overpotential=0.05 V.",
            "data": {"E0":0.34, "z":2, "C":1.0, "T":298.15, "overpotential":0.05},
            "answer": 0.34 - 0.05
        },
        {
            "text": "Hess law: Step enthalpies -98.3 kJ and -104 kJ. Total?",
            "data": {"DeltaH_steps":[-98.3, -104.0], "units":"kJ"},
            "answer": -202.3
        },
        {
            "text": "MO bond order: Nb=8, Na=5 (e.g., F2- valence).",
            "data": {"N_bonding":8, "N_antibonding":5},
            "answer": 1.5
        },
    ]

# ========== 5. 可选可视化（默认不启用） ==========
def simple_bar_plot(values: Dict[str, float], title: str = "Results", show: bool = False) -> None:
    """
    最简可视化：柱状图。默认不显示，设置 show=True 时启用。
    """
    if not show:
        return
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    names = list(values.keys())
    vals = list(values.values())
    plt.figure(figsize=(6,3))
    plt.bar(names, vals)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ========== 6. main：批处理与评估 ==========
def main():
    """
    主函数：批量演示如何使用工具函数求解多问题，并进行答案对比评估。
    """
    problems = example_dataset()
    results: List[Dict[str, Any]] = []
    summary_ok = 0

    for i, prob in enumerate(problems, 1):
        out = solve_problem(prob)
        results.append(out)
        ok_str = "✓" if out.get("ok") or out.get("ok") is None else "×"
        print(f"[{i}] Type={out['type']:<10} Result={out['result']} {out['units'] or ''}  OK={ok_str}  Details={out['details']}")

        if out.get("ok"):
            summary_ok += 1

    print(f"\nValidated {summary_ok}/{len([p for p in problems if p.get('answer') is not None])} problems within tolerance.")

    # 可选可视化演示（默认关闭）
    # simple_bar_plot({f'Q{i+1}': r['result'] for i, r in enumerate(results) if isinstance(r.get('result'), (int, float))}, show=False)

if __name__ == "__main__":
    main()
```
输出格式
直接输出为 .py 文件内容：
# Filename: <tool_name>.py
# 完整的工具函数代码
## 注意事项
1. 代码必须完整可运行，文件开头必须有 # Filename: <tool_name>.py 注释。
2. 函数命名与参数设计需具有通用性，避免与特定题目强耦合。
3. 架构须体现“代码即工具”：数据知识库、单位工具、通用原语、问题调度均可复用与扩展。
4. 可视化为可选且最简；需包含字体设置示例以确保标签可显示。
5. 需面向同一领域内至少三个关键子任务构建模块，并在 main 中进行多问题的批量演示与自动评估。
