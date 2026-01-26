# Filename: circuit_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

def calculate_series_resistance(resistances):
    """
    Calculate the equivalent resistance of resistors connected in series.
    
    In a series connection, the equivalent resistance is the sum of all individual resistances.
    Formula: R_eq = R1 + R2 + ... + Rn
    
    Parameters:
    -----------
    resistances : list or numpy.ndarray
        List of resistance values in ohms (Ω)
    
    Returns:
    --------
    float
        Equivalent resistance in ohms (Ω)
    
    Examples:
    ---------
    >>> calculate_series_resistance([4.0, 7.0])
    11.0
    """
    return np.sum(resistances)

def calculate_parallel_resistance(resistances):
    """
    Calculate the equivalent resistance of resistors connected in parallel.
    
    In a parallel connection, the reciprocal of the equivalent resistance is the sum of the 
    reciprocals of all individual resistances.
    Formula: 1/R_eq = 1/R1 + 1/R2 + ... + 1/Rn
    
    Parameters:
    -----------
    resistances : list or numpy.ndarray
        List of resistance values in ohms (Ω)
    
    Returns:
    --------
    float
        Equivalent resistance in ohms (Ω)
    
    Examples:
    ---------
    >>> calculate_parallel_resistance([6.0, 3.0])
    2.0
    """
    return 1.0 / np.sum(1.0 / np.array(resistances))

def solve_wheatstone_bridge(r1, r2, r3, r4):
    """
    Calculate the equivalent resistance of a Wheatstone bridge circuit.
    
    A Wheatstone bridge consists of four resistors arranged in a diamond pattern.
    This function calculates the equivalent resistance between two opposite corners.
    
    Parameters:
    -----------
    r1, r2, r3, r4 : float
        Resistance values of the four resistors in ohms (Ω)
        Arranged as:
            r1
        a---/\/\/---c
        |           |
        |           |
        r4          r2
        |           |
        |           |
        b---/\/\/---d
            r3
    
    Returns:
    --------
    float
        Equivalent resistance between points a and b in ohms (Ω)
    """
    # Calculate the product and sum terms
    product = r1 * r3 + r2 * r4
    sum_term = (r1 + r2) * (r3 + r4)
    
    # Calculate the equivalent resistance
    r_eq = product / sum_term
    
    return r_eq


    """
    Solve a complex resistor circuit by recursively applying series and parallel rules.
    
    This function takes a description of the circuit structure and the resistance values,
    then calculates the equivalent resistance by applying the appropriate combination rules.
    
    Parameters:
    -----------
    circuit_structure : list
        Nested list describing the circuit structure. Each element can be:
        - An integer index (referring to resistances list)
        - A list with 's' prefix for series connection: ['s', elem1, elem2, ...]
        - A list with 'p' prefix for parallel connection: ['p', elem1, elem2, ...]
    
    resistances : list
        List of resistance values in ohms (Ω)
    
    Returns:
    --------
    float
        Equivalent resistance of the entire circuit in ohms (Ω)
    
    Examples:
    ---------
    >>> # Circuit: R1 and R2 in series, in parallel with R3
    >>> structure = ['p', ['s', 0, 1], 2]
    >>> resistances = [10.0, 20.0, 15.0]
    >>> solve_complex_circuit(structure, resistances)
    8.57
    """
    if isinstance(circuit_structure, int):
        # Base case: single resistor
        return resistances[circuit_structure]
    
    connection_type = circuit_structure[0]
    components = circuit_structure[1:]
    
    # Calculate equivalent resistances for all sub-components
    component_resistances = [solve_complex_circuit(comp, resistances) for comp in components]
    
    # Apply the appropriate combination rule
    if connection_type == 's':
        return calculate_series_resistance(component_resistances)
    elif connection_type == 'p':
        return calculate_parallel_resistance(component_resistances)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

def essential_circuit_analysis_guide():
    """
    🔧 Equivalent Circuit Analysis Tool - Essential Reading Guide
    This is a multimodal analysis tool to help you correctly analyze circuit diagrams and calculate equivalent resistance
    """
    
    return """
        🎯 **Equivalent Circuit Analysis - Essential Reading Guide**
        📋 **Instructions**:
        1. Upload circuit diagram or describe circuit structure
        2. Follow the step-by-step analysis below

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔍 **STEP 1: Image Recognition & Structural Analysis**

        📸 **Key Points for Circuit Reading**:
        • Identify all resistor components (zigzag wave symbols)
        • Trace wire connection paths (straight lines)
        • Mark current entry and exit points
        • Focus on actual connections, don't be misled by layout

        🏷️ **Node Marking Method**:
        • Mark all wire junction points as nodes A, B, C...
        • All connection points at the same node have equal potential
        • Draw a simplified node connection diagram

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔍 **STEP 2: Series-Parallel Relationship Identification**

        ⚡ **Series Identification**:
        ✅ Unique current path → Series
        ✅ Components connected end-to-end → Series
        ✅ No branching points → Series

        ⚡ **Parallel Identification**:
        ✅ Share the same two nodes → Parallel
        ✅ Current has multiple branch paths → Parallel
        ✅ Components "head-to-head, tail-to-tail" → Parallel

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔍 **STEP 3: Equivalent Calculation Strategy**

        🎯 **Analysis Order** (Inside-Out):
        1️⃣ Find the most obvious series or parallel combination
        2️⃣ Calculate the equivalent resistance of that combination
        3️⃣ Replace the original combination with equivalent resistance
        4️⃣ Redraw the simplified circuit diagram
        5️⃣ Repeat steps until you get a single equivalent resistance

        💡 **Calculation Formulas**:
        Series: R_eq = R1 + R2 + R3 + ...
        Parallel: 1/R_eq = 1/R1 + 1/R2 + 1/R3 + ...

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔍 **STEP 4: Special Circuit Recognition**

        🔺 **Delta Networks** (Delta-Y Transform):
        • Three resistors forming triangular connections
        • Requires special transformation formulas
        • Or use nodal analysis method

        🌉 **Bridge Circuits**:
        • Five resistors arranged in bridge configuration
        • Middle resistor connects diagonal nodes
        • Requires circuit theorems for solution

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔍 **STEP 5: Verification & Checking**

        ✔️ **Result Verification**:
        • Unit check: Is result in ohms (Ω)?
        • Numerical reasonableness: Should not exceed sum of all resistors
        • Limit check: Behavior when resistance → 0 or → ∞

        ✔️ **Common Mistakes**:
        ❌ Position misleading: Adjacent ≠ Series, Vertical ≠ Parallel
        ❌ Wire neglect: Ignoring actual connection paths
        ❌ Node confusion: Electrical connection ≠ Physical position

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        after this manual, you should give like '[R6] + [R5 || (R1||R2 + R3 + R4)]'output(|| is parallel ,and + is series)
        """

def main():
    """
    主函数：演示如何使用工具函数求解电路问题
    """
    print("=== 电路分析工具演示 ===")
    
    # 示例1：解决电阻网络问题
    print("\n示例1：计算复杂电阻网络的等效电阻")
    
    # 定义电路参数 - 使用图中的电阻值
    r1 = 4.0  # 左侧电阻
    r2 = 7.0  # 上方电阻
    r3 = 9.0  # 右侧电阻
    r_bottom = 18.0  # 下方电阻R（假设值，图中未给出具体值）
    
    # 计算c-d路径的等效电阻（r2和r_bottom并联）
    r_cd_parallel = calculate_parallel_resistance([r2, r_bottom])
    print(f"c-d路径的等效电阻: {r_cd_parallel:.2f} Ω")
    
    # 计算a-b的等效电阻（r1和r3与c-d路径串联）
    r_equivalent = calculate_series_resistance([r1, r_cd_parallel, r3])
    print(f"a-b间的等效电阻: {r_equivalent:.2f} Ω")
    
    
if __name__ == "__main__":
    main()