import json
import math
import os
import sqlite3
from typing import Dict, List, Optional

# Optional domain-specific libraries
# - pubchempy: for PubChem access (pip install pubchempy)
# - chempy: for chemical equilibrium utilities (pip install chempy)
try:
    import pubchempy as pcp  # domain-specific
except ImportError:
    pcp = None

try:
    # 延迟导入 chempy，避免在模块级别导入时触发依赖问题
    # 如果 numpy 2.x 与 quantities 不兼容，这个导入会失败
    # 由于代码中实际上没有使用 periodic，我们将其设为 None
    # from chempy.util import periodic  # domain-specific, used to show library integration
    # We won't rely on heavy chempy equilibrium machinery here, but it is integrated for future extensions
    periodic = None  # 暂时禁用，避免 numpy 2.x 兼容性问题
except ImportError:
    periodic = None

# Global constants for paths
MID_RESULT_DIR = "./mid_result/chemistry"
TOOL_IMAGE_DIR = "./tool_images"
LOCAL_DB_PATH = "./acid_pKa.db"

# Ensure directories exist
os.makedirs(MID_RESULT_DIR, exist_ok=True)
os.makedirs(TOOL_IMAGE_DIR, exist_ok=True)

# ============ Atomic Functions (Layer 1) ============

def save_mid_result(subject: str, filename: str, data: dict) -> dict:
    """
    Save intermediate results as JSON into ./mid_result/{subject}/filename.json
    """
    if subject not in ["physics", "chemistry", "materials", "astronomy", "geography"]:
        return {"result": None, "metadata": {"error": "Invalid subject", "valid_subjects": ["physics", "chemistry", "materials", "astronomy", "geography"]}}
    dir_path = f"./mid_result/{subject}"
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, f"{filename}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"result": filepath, "metadata": {"file_type": "json", "size": os.path.getsize(filepath)}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def load_file(filepath: str) -> dict:
    """
    Generic loader for JSON and CSV files.
    """
    if not isinstance(filepath, str):
        return {"result": None, "metadata": {"error": "filepath must be str"}}
    if not os.path.exists(filepath):
        return {"result": None, "metadata": {"error": f"File not found: {filepath}"}}
    try:
        if filepath.lower().endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {"result": data, "metadata": {"file_type": "json", "size": os.path.getsize(filepath)}}
        elif filepath.lower().endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
                rows = [line.split(",") for line in lines]
            return {"result": rows, "metadata": {"file_type": "csv", "size": os.path.getsize(filepath)}}
        else:
            return {"result": None, "metadata": {"error": "Unsupported file type"}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def validate_percentages(theoretical_pct: float, actual_pct: float) -> dict:
    """
    Validate percentage inputs (0-100).
    Returns normalized fractions and ratio actual/theoretical.
    """
    if not isinstance(theoretical_pct, (int, float)) or not isinstance(actual_pct, (int, float)):
        return {"result": None, "metadata": {"error": "Inputs must be numeric (int/float)"}}
    if theoretical_pct <= 0 or theoretical_pct > 100:
        return {"result": None, "metadata": {"error": "theoretical_pct must be in (0, 100]", "value": theoretical_pct}}
    if actual_pct < 0 or actual_pct > 100:
        return {"result": None, "metadata": {"error": "actual_pct must be in [0, 100]", "value": actual_pct}}
    frac_theo = theoretical_pct / 100.0
    frac_act = actual_pct / 100.0
    ratio = frac_act / frac_theo if frac_theo > 0 else None
    return {
        "result": {"frac_theoretical": frac_theo, "frac_actual": frac_act, "ratio": ratio},
        "metadata": {"note": "Validated percentages and computed ratio"}
    }

def henderson_hasselbalch(pKa: float, base_to_acid_ratio: float) -> dict:
    """
    Henderson-Hasselbalch equation:
    pH = pKa + log10([A-]/[HA]) where base_to_acid_ratio = [A-]/[HA]
    """
    if not isinstance(pKa, (int, float)):
        return {"result": None, "metadata": {"error": "pKa must be numeric"}}
    if not isinstance(base_to_acid_ratio, (int, float)):
        return {"result": None, "metadata": {"error": "base_to_acid_ratio must be numeric"}}
    if base_to_acid_ratio <= 0:
        return {"result": None, "metadata": {"error": "base_to_acid_ratio must be > 0"}}
    pH = pKa + math.log10(base_to_acid_ratio)
    return {"result": pH, "metadata": {"equation": "Henderson-Hasselbalch", "pKa": pKa, "ratio": base_to_acid_ratio}}

def round_pH(pH: float, decimals: int) -> dict:
    """
    Round pH to given decimals.
    """
    if not isinstance(pH, (int, float)):
        return {"result": None, "metadata": {"error": "pH must be numeric"}}
    if not isinstance(decimals, int) or decimals < 0 or decimals > 6:
        return {"result": None, "metadata": {"error": "decimals must be int in [0,6]"}}
    return {"result": round(pH, decimals), "metadata": {"rounded_to": decimals}}

def create_local_acid_db(db_path: str) -> dict:
    """
    Create and populate a local SQLite DB with selected acids and their pKa (monoprotic effective values).
    Data sources: typical literature/NIST values (approximations).
    """
    if not isinstance(db_path, str):
        return {"result": None, "metadata": {"error": "db_path must be str"}}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS acids (name TEXT PRIMARY KEY, pKa REAL)")
        # Populate with common monoprotic acids
        data = [
            ("acetic acid", 4.76),
            ("propionic acid", 4.87),
            ("butyric acid", 4.82),
            ("benzoic acid", 4.20),
            ("lactic acid", 3.86),
            ("succinic acid (pKa1)", 4.21),
            ("crotonic acid", 4.69)
        ]
        for name, pka in data:
            cur.execute("INSERT OR REPLACE INTO acids (name, pKa) VALUES (?, ?)", (name, pka))
        conn.commit()
        conn.close()
        return {"result": db_path, "metadata": {"count": len(data)}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def query_local_acid_pKa(db_path: str, compound_name: str) -> dict:
    """
    Query local DB for acid pKa.
    """
    if not isinstance(db_path, str) or not isinstance(compound_name, str):
        return {"result": None, "metadata": {"error": "db_path and compound_name must be str"}}
    if not os.path.exists(db_path):
        return {"result": None, "metadata": {"error": "Database not found"}}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT pKa FROM acids WHERE name = ?", (compound_name.lower(),))
        row = cur.fetchone()
        conn.close()
        if row is None:
            # Try case-insensitive lookup by LIKE
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name, pKa FROM acids WHERE LOWER(name) LIKE ?", (f"%{compound_name.lower()}%",))
            row2 = cur.fetchone()
            conn.close()
            if row2 is None:
                return {"result": None, "metadata": {"error": f"{compound_name} not found in local DB"}}
            else:
                return {"result": row2[1], "metadata": {"source": "local_db", "matched_name": row2[0]}}
        return {"result": row[0], "metadata": {"source": "local_db", "matched_name": compound_name}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def fetch_pKa_from_pubchem(compound_name: str) -> dict:
    """
    Attempt to fetch pKa from PubChem via pubchempy.
    Note: PubChem often does not expose pKa as a simple property; this function tries and falls back to None.
    """
    if not isinstance(compound_name, str):
        return {"result": None, "metadata": {"error": "compound_name must be str"}}
    if pcp is None:
        return {"result": None, "metadata": {"error": "pubchempy not installed"}}
    try:
        comps = pcp.get_compounds(compound_name, "name")
        if not comps:
            return {"result": None, "metadata": {"error": "Compound not found on PubChem"}}
        # PubChem does not provide pKa directly via pubchempy properties; we document this and return None.
        # In production, one would parse "record" or external sources; here we provide graceful fallback.
        return {"result": None, "metadata": {"error": "pKa not available via pubchempy standard API"}}
    except Exception as e:
        return {"result": None, "metadata": {"error": str(e)}}

def choose_best_acid_for_ratio(db_path: str, ratio: float, candidate_names: List[str]) -> dict:
    """
    Given a base_to_acid ratio, select an acid from candidates for which the computed pH falls in a typical buffer range (3-6),
    preferring carboxylic acids. Returns pKa and chosen name.
    """
    if not isinstance(db_path, str):
        return {"result": None, "metadata": {"error": "db_path must be str"}}
    if not isinstance(ratio, (int, float)) or ratio <= 0:
        return {"result": None, "metadata": {"error": "ratio must be > 0"}}
    if not isinstance(candidate_names, list) or not all(isinstance(x, str) for x in candidate_names):
        return {"result": None, "metadata": {"error": "candidate_names must be list[str]"}}
    pH_candidates = []
    for name in candidate_names:
        pka_res = query_local_acid_pKa(db_path, name)
        if pka_res["result"] is None:
            continue
        pka_val = pka_res["result"]
        pH_res = henderson_hasselbalch(pka_val, ratio)
        if pH_res["result"] is None:
            continue
        pH_val = pH_res["result"]
        if 3.0 <= pH_val <= 6.0:
            pH_candidates.append({"name": name, "pKa": pka_val, "pH": pH_val})
    if not pH_candidates:
        return {"result": None, "metadata": {"error": "No suitable acid found in typical buffer range"}}
    # Choose the one closest to midrange pH=4.5 to represent a common exam scenario
    chosen = min(pH_candidates, key=lambda d: abs(d["pH"] - 4.5))
    return {"result": {"name": chosen["name"], "pKa": chosen["pKa"], "pH": chosen["pH"]},
            "metadata": {"note": "Selected acid yields pH near midrange 4.5"}}

# ============ Combination Functions (Layer 2) ============

def estimate_pH_from_yields_with_buffer(theoretical_pct: float, actual_pct: float, acid_name: Optional[str], db_path: str) -> dict:
    """
    Estimate final pH assuming 'X' forms a monoprotic buffer; yield reduction maps to base/acid ratio via:
    ratio = (actual / theoretical). Use Henderson-Hasselbalch with pKa from local DB or PubChem.
    """
    # Step 1: validate and compute ratio
    val_res = validate_percentages(theoretical_pct, actual_pct)
    if val_res["result"] is None:
        return {"result": None, "metadata": {"error": val_res["metadata"]["error"]}}
    ratio = val_res["result"]["ratio"]
    mid_save = save_mid_result("chemistry", "yield_validation", val_res)
    # Step 2: get pKa
    pka_val = None
    pka_source = None
    matched_name = None

    if acid_name:
        # Try PubChem first
        pub_res = fetch_pKa_from_pubchem(acid_name)
        if pub_res["result"] is not None:
            pka_val = pub_res["result"]
            pka_source = "pubchem"
            matched_name = acid_name
        else:
            # Fallback to local DB
            loc_res = query_local_acid_pKa(db_path, acid_name)
            if loc_res["result"] is not None:
                pka_val = loc_res["result"]
                pka_source = loc_res["metadata"].get("source", "local_db")
                matched_name = loc_res["metadata"].get("matched_name", acid_name)
            else:
                return {"result": None, "metadata": {"error": f"pKa not found for {acid_name} in PubChem/local DB"}}
    else:
        # No specified acid, choose best from candidates
        candidates = ["propionic acid", "butyric acid", "acetic acid", "crotonic acid", "succinic acid (pKa1)"]
        choose_res = choose_best_acid_for_ratio(db_path, ratio, candidates)
        if choose_res["result"] is None:
            return {"result": None, "metadata": {"error": "Failed to select suitable acid"}}
        matched_name = choose_res["result"]["name"]
        pka_val = choose_res["result"]["pKa"]
        pka_source = "local_db_selected"

    # Step 3: compute pH via Henderson-Hasselbalch
    pH_res = henderson_hasselbalch(pka_val, ratio)
    if pH_res["result"] is None:
        return {"result": None, "metadata": {"error": pH_res["metadata"]["error"]}}
    pH_val = pH_res["result"]
    # Step 4: rounding to one decimal place (typical reporting)
    pH_round_res = round_pH(pH_val, 1)
    # Save mid results
    mid_data = {
        "theoretical_pct": theoretical_pct,
        "actual_pct": actual_pct,
        "ratio": ratio,
        "acid_name": matched_name,
        "pKa_source": pka_source,
        "pKa": pka_val,
        "pH_raw": pH_val,
        "pH_rounded": pH_round_res["result"]
    }
    mid_save2 = save_mid_result("chemistry", "pH_estimation", mid_data)
    return {"result": pH_round_res["result"], "metadata": {"details_file": mid_save2["result"], "acid_name": matched_name, "pKa_source": pka_source}}

def scan_pH_over_acids(theoretical_pct: float, actual_pct: float, acid_list: List[str], db_path: str) -> dict:
    """
    Scan multiple acids and compute pH for each to see sensitivity to pKa choice.
    """
    val_res = validate_percentages(theoretical_pct, actual_pct)
    if val_res["result"] is None:
        return {"result": None, "metadata": {"error": val_res["metadata"]["error"]}}
    ratio = val_res["result"]["ratio"]
    results = []
    for name in acid_list:
        loc_res = query_local_acid_pKa(db_path, name)
        if loc_res["result"] is None:
            results.append({"name": name, "pKa": None, "pH": None, "error": "pKa not found"})
            continue
        pka_val = loc_res["result"]
        pH_res = henderson_hasselbalch(pka_val, ratio)
        pH_val = pH_res["result"]
        pH_round = round(pH_val, 2)
        results.append({"name": name, "pKa": pka_val, "pH": pH_round})
    mid_save = save_mid_result("chemistry", "scan_pH_over_acids", {"ratio": ratio, "results": results})
    return {"result": results, "metadata": {"details_file": mid_save["result"]}}

def generate_pH_vs_ratio_curve(pKa: float, ratios: List[float], outfile: str) -> dict:
    """
    Generate a pH vs ratio curve image using Henderson-Hasselbalch.
    Saves to ./tool_images/outfile.png
    """
    if not isinstance(pKa, (int, float)):
        return {"result": None, "metadata": {"error": "pKa must be numeric"}}
    if not isinstance(ratios, list) or not all(isinstance(r, (int, float)) and r > 0 for r in ratios):
        return {"result": None, "metadata": {"error": "ratios must be list of positive numbers"}}
    if not isinstance(outfile, str) or not outfile:
        return {"result": None, "metadata": {"error": "outfile must be non-empty str"}}
    # Lazy import matplotlib only inside function to keep JSON-serializable params
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"result": None, "metadata": {"error": f"matplotlib not available: {str(e)}"}}
    pH_vals = [pKa + math.log10(r) for r in ratios]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, pH_vals, marker="o")
    ax.set_xlabel("[A-]/[HA] ratio")
    ax.set_ylabel("pH")
    ax.set_title(f"pH vs ratio (pKa={pKa})")
    ax.grid(True, ls="--", alpha=0.5)
    filepath = os.path.join(TOOL_IMAGE_DIR, f"{outfile}.png")
    fig.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return {"result": filepath, "metadata": {"file_type": "png", "size": os.path.getsize(filepath)}}

# ============ Main Demonstration with 3 Scenarios ============

def main():
  
    print("=" * 60)
    print("场景1：基于缓冲假设，使用候选酸（丙酸）估算加入X后体系的最终pH")
    print("=" * 60)
    print("问题描述：理论产率90%，加入物质X后实际产率40%。假设X形成单元酸缓冲，产率比例≈[A-]/[HA]，用Henderson-Hasselbalch求pH。")
    print("-" * 60)


    # 步骤1：验证百分比并计算比例
    params_validate = {"theoretical_pct": 90.0, "actual_pct": 40.0}
    val_res = validate_percentages(**params_validate)
    print(f"FUNCTION_CALL: validate_percentages | PARAMS: {params_validate} | RESULT: {val_res}")

    # 步骤2：查询/选取酸pKa（选择丙酸作为X）
    params_query = {"db_path": LOCAL_DB_PATH, "compound_name": "propionic acid"}
    pka_res = query_local_acid_pKa(**params_query)
    print(f"FUNCTION_CALL: query_local_acid_pKa | PARAMS: {params_query} | RESULT: {pka_res}")

    # 步骤3：用HH方程计算pH
    ratio = val_res["result"]["ratio"] if val_res["result"] else None
    params_hh = {"pKa": pka_res["result"], "base_to_acid_ratio": ratio}
    pH_raw_res = henderson_hasselbalch(**params_hh)
    print(f"FUNCTION_CALL: henderson_hasselbalch | PARAMS: {params_hh} | RESULT: {pH_raw_res}")

    # 步骤4：四舍五入到一位小数
    params_round = {"pH": pH_raw_res["result"], "decimals": 1}
    pH_round_res = round_pH(**params_round)
    print(f"FUNCTION_CALL: round_pH | PARAMS: {params_round} | RESULT: {pH_round_res}")

    answer1 = pH_round_res["result"]
    print(f"FINAL_ANSWER: {answer1}")

    print("=" * 60)
    print("场景2：在一组常见羧酸中扫描pKa，比较不同酸的pH估算并选择接近4.5的结果")
    print("=" * 60)
    print("问题描述：未知X具体身份，使用本地数据库中的多种酸，基于产率比例计算pH，查看pKa敏感性。")
    print("-" * 60)

    # 步骤1：扫描不同酸的pH
    params_scan = {"theoretical_pct": 90.0, "actual_pct": 40.0, "acid_list": ["propionic acid", "butyric acid", "acetic acid", "crotonic acid", "benzoic acid"], "db_path": LOCAL_DB_PATH}
    scan_res = scan_pH_over_acids(**params_scan)
    print(f"FUNCTION_CALL: scan_pH_over_acids | PARAMS: {params_scan} | RESULT: {scan_res}")

    # 步骤2：自动选择最接近4.5的酸（组合选择器）
    val_ratio = validate_percentages(90.0, 40.0)
    choose_params = {"db_path": LOCAL_DB_PATH, "ratio": val_ratio["result"]["ratio"] if val_ratio["result"] else 0.4444, "candidate_names": ["propionic acid", "butyric acid", "acetic acid", "crotonic acid", "succinic acid (pKa1)"]}
    choose_res = choose_best_acid_for_ratio(**choose_params)
    print(f"FUNCTION_CALL: choose_best_acid_for_ratio | PARAMS: {choose_params} | RESULT: {choose_res}")

    # 步骤3：给出最终估算的pH（四舍五入）
    chosen_pH = choose_res["result"]["pH"] if choose_res["result"] else None
    pH_round2 = round_pH(chosen_pH, 1)
    print(f"FUNCTION_CALL: round_pH | PARAMS: {{'pH': {chosen_pH}, 'decimals': 1}} | RESULT: {pH_round2}")

    answer2 = pH_round2["result"]
    print(f"FINAL_ANSWER: {answer2}")

    print("=" * 60)
    print("场景3：可视化pH-比例曲线（以丙酸pKa为例），展示pH对[A-]/[HA]的响应")
    print("=" * 60)
    print("问题描述：绘制pH随[A-]/[HA]比的变化曲线，验证Henderson-Hasselbalch关系并保存图像。")
    print("-" * 60)

    # 步骤1：生成曲线图像
    ratios = [0.1, 0.2, 0.3, 0.4444, 0.6, 0.8, 1.0, 1.5, 2.0]
    # Use propionic acid pKa for visualization
    pka_vis = query_local_acid_pKa(LOCAL_DB_PATH, "propionic acid")["result"]
    params_plot = {"pKa": pka_vis, "ratios": ratios, "outfile": "pH_vs_ratio_propionic"}
    plot_res = generate_pH_vs_ratio_curve(**params_plot)
    print(f"FUNCTION_CALL: generate_pH_vs_ratio_curve | PARAMS: {params_plot} | RESULT: {plot_res}")

    # 步骤2：读取中间结果文件验证数据链完整
    midfile_path = scan_res["metadata"]["details_file"]
    load_mid = load_file(midfile_path)
    print(f"FUNCTION_CALL: load_file | PARAMS: {{'filepath': '{midfile_path}'}} | RESULT: {load_mid}")

    # 最终答案为场景1题目所需的pH（保留一位小数）
    final_answer = answer1
    print(f"FINAL_ANSWER: {final_answer}")

if __name__ == "__main__":
    main()