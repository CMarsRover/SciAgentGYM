# Task Objective
Based on multiple problems and their reference answers (problem-answer pairs) within the same domain, generate available tool sets and build a computational and reasoning toolkit library (Python) that can be generalized to multiple tasks within that domain, while supporting:
# Requirements to Consider
1. Abstraction and reuse of general algorithms and mathematical principles;
    Batch parsing, solving, and validation of multiple problems (comparison with answers);
    Optional construction and invocation of data knowledge bases (formula library, constant library, empirical rule library);
    Extensible modular design for multiple sub-domain scenarios (at least three key sub-domains/task scenarios).
2. Input and Adaptation Range
    Input includes: grouped problem-answer pairs (Q-A pairs), background knowledge points for the domain, available external tools/library lists, possible units and data specifications.
    Target domain is the key domain specified in the previous conversation; need to build a unified toolkit for at least three sub-directions within that domain (e.g., electrochemical quantification, physical chemistry thermodynamics, molecular structure and MO theory; or "three typical computational scenarios within the same domain").
3. Toolkit must demonstrate batch solving process for multiple problems in main.
# Execution Requirements
Strictly output complete Python code files (integrated toolkit) following the process below, and ensure it is runnable.
1. Mathematical and Domain Principle Analysis (for multiple problems)
Unified review of core mathematical and physical chemistry principles, formulas, and laws in the domain, forming "formula library/rule library" data structures (serializable or dictionary storage).
Abstract common solution paths for different sub-scenarios; clarify mathematical tools and unit specifications needed for each calculation step.
Form a reasoning framework for multiple problems (e.g., parse → classify → unit calibration → parameter extraction → calculate → uncertainty/numerical stability check → result validation).
2. Core Algorithm Abstraction (Reuse and Extension)
Identify general computational primitives (e.g., Nernst equation, Hess's law, ideal gas state equation, mole/charge conversion, molecular orbital electron counting and bond order, etc.), design reusable functions.
Design parsing and scheduling layer for multiple problems: support batch processing, automatically select computational modules based on problem type, and output structured results.
Analyze and adopt necessary Python scientific computing libraries (such as numpy, scipy); visualization is optional and minimal.
Avoid excessive class design, prioritize function + lightweight data structures (dict, dataclass optional) organization.
3. Code Implementation Standards (Toolization and Generalization)
Write high-quality tool functions: complete docstrings, generalized parameters, no hardcoded specific values.
Functions must include: parameter descriptions (units/ranges/physical meaning), return value structure descriptions (including units and exception conventions), usage examples.
Design unified unit handling and validation tools (at least include conversion for common dimensions: volume, pressure, temperature, charge, concentration, energy, potential, etc.).
Design minimal usable formula/constant data knowledge base (dictionary or JSON serializable) for multiple modules to call and extend.
4. In main function:
Demonstrate batch solving workflow;
Compare calculation results with given answers (tolerance configurable), and output evaluation summary;
Optional: Provide minimal visualization for one sub-task (such as a curve or bar chart), and ensure font compatibility settings are available but not enabled by default.
📌 Code Template Specification
# Filename: <tool_name>.py
# (Note: <tool_name> must match the domain tool theme, e.g., physical_chemistry_multi_solver)
```python
## physical_chemistry_multi_solver.py
from typing import Dict, Any, List, Tuple, Optional

# ========== 0. Constants and Configuration ==========
FORMULAE_DB = {
    # Example: Extensible formula library/rule library
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
    "nernst_log_base": 10,  # or e, self-adapt
}

# ========== 1. Unit and Numerical Tools ==========
def to_si_pressure(value: float, unit: str = "Pa") -> float:
    """
    Convert pressure to Pa.
    Supported units: Pa, kPa, bar, atm, torr, mmHg.
    """
    unit = unit.lower()
    factors = {"pa":1.0, "kpa":1e3, "bar":1e5, "atm":101325.0, "torr":133.322368, "mmhg":133.322368}
    if unit not in factors:
        raise ValueError(f"Unsupported pressure unit: {unit}")
    return value * factors[unit]

def to_si_volume(value: float, unit: str = "m3") -> float:
    """
    Convert volume to m^3.
    Supported units: m3, L, mL, uL, nL, pL, cm3.
    """
    unit = unit.lower()
    factors = {"m3":1.0, "l":1e-3, "ml":1e-6, "ul":1e-9, "nl":1e-12, "pl":1e-15, "cm3":1e-6}
    if unit not in factors:
        raise ValueError(f"Unsupported volume unit: {unit}")
    return value * factors[unit]

def to_si_temperature(value: float, unit: str = "K") -> float:
    """
    Convert temperature to K.
    Supported units: K, C.
    """
    unit = unit.lower()
    if unit == "k":
        return value
    if unit == "c":
        return value + 273.15
    raise ValueError(f"Unsupported temperature unit: {unit}")

def close_enough(x: float, y: float, tol_abs: float = DEFAULTS["tolerance_abs"], tol_rel: float = DEFAULTS["tolerance_rel"]) -> bool:
    """
    Determine if two values are equal within given tolerance.
    """
    return abs(x - y) <= max(tol_abs, tol_rel * max(1.0, abs(y)))

# ========== 2. General Computational Primitives ==========
def ideal_gas_moles(P: float, V: float, T: float, P_unit="Pa", V_unit="m3", T_unit="K", R: float = CONSTANTS["R"]) -> float:
    """
    Ideal gas equation: n = P*V/(R*T)
    Parameters:
    -----------
    P : float
        Pressure value
    V : float
        Volume value
    T : float
        Temperature value
    P_unit, V_unit, T_unit : str
        Units (supported units see respective conversion functions)
    R : float
        Gas constant, unit J/(mol·K)
    Returns:
    --------
    float
        Amount of substance (mol)
    """
    P_si = to_si_pressure(P, P_unit)
    V_si = to_si_volume(V, V_unit)
    T_si = to_si_temperature(T, T_unit)
    return (P_si * V_si) / (R * T_si)

def faraday_moles(Q: float, z: int = 1, F: float = CONSTANTS["F"]) -> float:
    """
    Faraday's law: Calculate amount of substance based on charge.
    Parameters:
    -----------
    Q : float
        Charge quantity (C)
    z : int
        Number of electron transfers
    F : float
        Faraday constant (C/mol)
    Returns:
    --------
    float
        Amount of substance (mol)
    """
    if z <= 0:
        raise ValueError("z must be positive.")
    return Q / (z * F)

def nernst_potential(E0: float, z: int, activity: float, T: float = CONSTANTS["T_STD"], log_base: str = "10") -> float:
    """
    Nernst equation to calculate electrode potential.
    E = E0 + (RT/(zF)) * ln(a) or E0 + (0.05916/z) * log10(a) at 298 K.
    Parameters:
    -----------
    E0 : float
        Standard electrode potential (V)
    z : int
        Number of electrons
    activity : float
        Activity or effective concentration (dimensionless)
    T : float
        Temperature (K)
    log_base : {"e","10"}
        Choose natural logarithm or common logarithm form
    Returns:
    --------
    float
        Potential (V)
    """
    if z == 0:
        raise ValueError("z must be nonzero.")
    if activity <= 0:
        raise ValueError("activity must be positive.")
    if log_base == "e":
        from math import log
        return E0 + (CONSTANTS["R"] * T) / (z * CONSTANTS["F"]) * log(activity)
    else:
        # Default convenient constant for 298 K
        return E0 + (0.05916 / z) * __import__("math").log10(activity)

def hess_sum(delta_h_steps: List[float]) -> float:
    """
    Hess's law: Total enthalpy change equals sum of step enthalpy changes.
    """
    return sum(delta_h_steps)

def bond_order(n_bonding: int, n_antibonding: int) -> float:
    """
    Molecular orbital method bond order calculation.
    """
    return 0.5 * (n_bonding - n_antibonding)

# ========== 3. Problem Parsing and Scheduling ==========
def classify_problem(problem_text: str) -> str:
    """
    Roughly classify problem text into a sub-task type.
    Returns type label, e.g., "ideal_gas", "faraday", "nernst", "hess", "bond_order".
    Rules are extensible: based on keywords or lightweight regex.
    """
    t = problem_text.lower()
    if any(k in t for k in ["kpa", "atm", "torr", "cm3", "ideal gas", "container", "gas equation", "pressure", "volume"]):
        return "ideal_gas"
    if any(k in t for k in ["stripping", "faraday", "charge", "coulomb"]):
        return "faraday"
    if "nernst" in t or "log" in t and "potential" in t:
        return "nernst"
    if "Δh" in t.lower() or "hess" in t or "enthalpy" in t:
        return "hess"
    if "bond order" in t or "mo" in t:
        return "bond_order"
    return "unknown"

def solve_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified solving entry.
    problem must contain:
      - "text": Problem text
      - "data": Parameter dictionary (optional)
      - "answer": Reference answer (optional, for validation)
    Returns structured result: {"type","result","units","ok","details"}
    """
    ptype = classify_problem(problem.get("text", ""))
    data = problem.get("data", {})
    result = None
    units = None
    details: Dict[str, Any] = {"ptype": ptype}

    try:
        if ptype == "ideal_gas":
            # Expected keys: P, V, T, P_unit, V_unit, T_unit
            n = ideal_gas_moles(
                data["P"], data["V"], data["T"],
                data.get("P_unit","Pa"), data.get("V_unit","m3"), data.get("T_unit","K")
            )
            result, units = n, "mol"
        elif ptype == "faraday":
            # Expected keys: Q, z
            n = faraday_moles(data["Q"], data.get("z", 1))
            if "V_particle" in data:
                # Optional concentration calculation
                c = n / to_si_volume(data["V_particle"], data.get("V_unit","m3"))  # mol/m3
                details["concentration_mol_per_m3"] = c
                details["concentration_M"] = c / 1000.0
            result, units = n, "mol"
        elif ptype == "nernst":
            E = nernst_potential(
                data["E0"], data["z"], data.get("activity", data.get("C", 1.0)),
                data.get("T", CONSTANTS["T_STD"]), data.get("log_base","10")
            )
            # Overpotential correction (optional)
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

    # Validation
    ref = problem.get("answer", None)
    ok = None
    if ref is not None and isinstance(result, (int, float)):
        ok = close_enough(result, float(ref))
    return {"type": ptype, "result": result, "units": units, "ok": ok, "details": details}

# ========== 4. Example Dataset and Batch Run ==========
def example_dataset() -> List[Dict[str, Any]]:
    """
    Construct example problem set covering multiple sub-domains (problem-answer pairs optional).
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

# ========== 5. Optional Visualization (disabled by default) ==========
def simple_bar_plot(values: Dict[str, float], title: str = "Results", show: bool = False) -> None:
    """
    Minimal visualization: Bar chart. Not displayed by default, enable with show=True.
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

# ========== 6. main: Batch Processing and Evaluation ==========
def main():
    """
    Main function: Batch demonstration of how to use tool functions to solve multiple problems, and perform answer comparison evaluation.
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

    # Optional visualization demo (disabled by default)
    # simple_bar_plot({f'Q{i+1}': r['result'] for i, r in enumerate(results) if isinstance(r.get('result'), (int, float))}, show=False)

if __name__ == "__main__":
    main()
```
Output Format
Directly output as .py file content:
# Filename: <tool_name>.py
# Complete tool function code
## Notes
1. Code must be complete and runnable, file header must have # Filename: <tool_name>.py comment.
2. Function naming and parameter design must be generalizable, avoid strong coupling with specific problems.
3. Architecture must reflect "code as tool": data knowledge base, unit tools, general primitives, problem scheduling are all reusable and extensible.
4. Visualization is optional and minimal; must include font setting examples to ensure labels are displayable.
5. Must build modules for at least three key sub-tasks within the same domain, and demonstrate batch processing and automatic evaluation of multiple problems in main.
