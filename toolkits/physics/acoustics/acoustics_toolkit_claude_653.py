# Filename: acoustics_toolkit.py

"""
Acoustics Toolkit for Sound Pressure Level Calculations
========================================================
This toolkit provides functions for calculating total sound pressure levels
from multiple sound sources using logarithmic addition and lookup tables.

Layers:
1. Atomic functions: Basic SPL calculations and table lookups
2. Composite functions: Multi-source SPL combination
3. Visualization functions: SPL distribution and comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import json
import os

# Configure matplotlib for proper font display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# Global constants
REFERENCE_PRESSURE = 20e-6  # 20 μPa (reference sound pressure)
SPL_LOOKUP_TABLE = {
    3: 1.8,
    4: 1.5,
    5: 1.2,
    6: 1.0,
    7: 0.8
}

# Create output directories
os.makedirs('./mid_result/acoustics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)


# ============================================================================
# LAYER 1: ATOMIC FUNCTIONS
# ============================================================================

def convert_spl_to_pressure(spl_db: float) -> Dict[str, Union[float, dict]]:
    """
    Convert sound pressure level (dB) to actual sound pressure (Pa).
    
    Formula: p = p0 * 10^(SPL/20)
    where p0 = 20 μPa (reference pressure)
    
    Args:
        spl_db: Sound pressure level in decibels
        
    Returns:
        Dictionary with pressure value and metadata
        
    Raises:
        ValueError: If spl_db is not a valid number
    """
    if not isinstance(spl_db, (int, float)):
        raise ValueError(f"SPL must be a number, got {type(spl_db)}")
    
    pressure_pa = REFERENCE_PRESSURE * (10 ** (spl_db / 20))
    
    return {
        'result': pressure_pa,
        'metadata': {
            'unit': 'Pa',
            'spl_input': spl_db,
            'reference_pressure': REFERENCE_PRESSURE,
            'formula': 'p = p0 * 10^(SPL/20)'
        }
    }


def convert_pressure_to_spl(pressure_pa: float) -> Dict[str, Union[float, dict]]:
    """
    Convert sound pressure (Pa) to sound pressure level (dB).
    
    Formula: SPL = 20 * log10(p/p0)
    where p0 = 20 μPa (reference pressure)
    
    Args:
        pressure_pa: Sound pressure in Pascals
        
    Returns:
        Dictionary with SPL value and metadata
        
    Raises:
        ValueError: If pressure is not positive
    """
    if not isinstance(pressure_pa, (int, float)) or pressure_pa <= 0:
        raise ValueError(f"Pressure must be a positive number, got {pressure_pa}")
    
    spl_db = 20 * np.log10(pressure_pa / REFERENCE_PRESSURE)
    
    return {
        'result': spl_db,
        'metadata': {
            'unit': 'dB',
            'pressure_input': pressure_pa,
            'reference_pressure': REFERENCE_PRESSURE,
            'formula': 'SPL = 20 * log10(p/p0)'
        }
    }


def get_spl_increase_from_table(spl_difference: float, 
                                 lookup_table: Dict[int, float] = None) -> Dict[str, Union[float, dict]]:
    """
    Get the increase value from SPL lookup table based on the difference.
    
    Uses linear interpolation for values between table entries.
    For differences > 7 dB, returns 0.8 dB (conservative estimate).
    For differences < 3 dB, uses extrapolation.
    
    Args:
        spl_difference: Absolute difference between two SPL values (dB)
        lookup_table: Optional custom lookup table (default uses global table)
        
    Returns:
        Dictionary with increase value and metadata
        
    Raises:
        ValueError: If spl_difference is negative
    """
    if not isinstance(spl_difference, (int, float)) or spl_difference < 0:
        raise ValueError(f"SPL difference must be non-negative, got {spl_difference}")
    
    if lookup_table is None:
        lookup_table = SPL_LOOKUP_TABLE
    
    # Sort table keys
    differences = sorted(lookup_table.keys())
    increases = [lookup_table[d] for d in differences]
    
    # Handle edge cases
    if spl_difference >= differences[-1] or spl_difference <= differences[0]:
        # Use exact logarithmic formula outside table range for accuracy
        increase = 10 * np.log10(1 + 10 ** (-spl_difference / 10))
        method = 'formula_fallback'
    else:
        # Linear interpolation within table range
        increase = np.interp(spl_difference, differences, increases)
        method = 'interpolation'
    
    return {
        'result': float(increase),
        'metadata': {
            'unit': 'dB',
            'spl_difference': spl_difference,
            'method': method,
            'lookup_table': lookup_table
        }
    }


def add_two_spl_values(spl1_db: float, spl2_db: float, 
                       use_table: bool = True) -> Dict[str, Union[float, dict]]:
    """
    Add two sound pressure levels using either table lookup or exact formula.
    
    Table method: SPL_total = SPL_higher + increase_value
    Exact method: SPL_total = 10 * log10(10^(SPL1/10) + 10^(SPL2/10))
    
    Args:
        spl1_db: First sound pressure level (dB)
        spl2_db: Second sound pressure level (dB)
        use_table: If True, use lookup table; if False, use exact formula
        
    Returns:
        Dictionary with total SPL and metadata
        
    Raises:
        ValueError: If SPL values are not valid numbers
    """
    if not all(isinstance(x, (int, float)) for x in [spl1_db, spl2_db]):
        raise ValueError("Both SPL values must be numbers")
    
    if use_table:
        # Table-based method
        spl_higher = max(spl1_db, spl2_db)
        spl_lower = min(spl1_db, spl2_db)
        difference = spl_higher - spl_lower
        
        increase_result = get_spl_increase_from_table(difference)
        increase = increase_result['result']
        total_spl = spl_higher + increase
        method = 'table_lookup'
        
    else:
        # Exact logarithmic formula
        intensity_sum = 10 ** (spl1_db / 10) + 10 ** (spl2_db / 10)
        total_spl = 10 * np.log10(intensity_sum)
        method = 'exact_formula'
        increase = total_spl - max(spl1_db, spl2_db)
    
    return {
        'result': float(total_spl),
        'metadata': {
            'unit': 'dB',
            'spl1': spl1_db,
            'spl2': spl2_db,
            'method': method,
            'increase_applied': float(increase),
            'formula': '10*log10(10^(SPL1/10) + 10^(SPL2/10))' if not use_table else 'table_lookup'
        }
    }


# ============================================================================
# LAYER 2: COMPOSITE FUNCTIONS
# ============================================================================

def calculate_total_spl_multiple_sources(spl_list: List[float], 
                                         use_table: bool = True) -> Dict[str, Union[float, dict]]:
    """
    Calculate total sound pressure level from multiple sound sources.
    
    Process:
    1. Sort SPL values in descending order
    2. Iteratively combine pairs using table lookup or exact formula
    3. Return final total SPL with detailed calculation steps
    
    Args:
        spl_list: List of sound pressure levels (dB)
        use_table: If True, use lookup table; if False, use exact formula
        
    Returns:
        Dictionary with total SPL and detailed calculation steps
        
    Raises:
        ValueError: If spl_list is empty or contains invalid values
    """
    if not isinstance(spl_list, list) or len(spl_list) == 0:
        raise ValueError("spl_list must be a non-empty list")
    
    if not all(isinstance(x, (int, float)) for x in spl_list):
        raise ValueError("All SPL values must be numbers")
    
    # Sort in descending order
    sorted_spl = sorted(spl_list, reverse=True)
    
    calculation_steps = []
    current_spl = sorted_spl[0]
    
    for i, next_spl in enumerate(sorted_spl[1:], 1):
        result = add_two_spl_values(current_spl, next_spl, use_table)
        step_info = {
            'step': i,
            'spl_current': float(current_spl),
            'spl_added': float(next_spl),
            'difference': float(current_spl - next_spl),
            'increase': result['metadata']['increase_applied'],
            'result': result['result']
        }
        calculation_steps.append(step_info)
        current_spl = result['result']
    
    # Save detailed calculation to file
    output_file = './mid_result/acoustics/spl_calculation_steps.json'
    with open(output_file, 'w') as f:
        json.dump({
            'input_spl_values': spl_list,
            'sorted_spl_values': sorted_spl,
            'method': 'table_lookup' if use_table else 'exact_formula',
            'calculation_steps': calculation_steps,
            'final_total_spl': float(current_spl)
        }, f, indent=2)
    
    return {
        'result': float(current_spl),
        'metadata': {
            'unit': 'dB',
            'input_count': len(spl_list),
            'input_values': spl_list,
            'sorted_values': sorted_spl,
            'method': 'table_lookup' if use_table else 'exact_formula',
            'calculation_steps': calculation_steps,
            'steps_file': output_file
        }
    }


def compare_table_vs_exact_methods(spl_list: List[float]) -> Dict[str, Union[dict, str]]:
    """
    Compare results from table lookup method vs exact formula method.
    
    Args:
        spl_list: List of sound pressure levels (dB)
        
    Returns:
        Dictionary with comparison results and analysis
    """
    if not isinstance(spl_list, list) or len(spl_list) == 0:
        raise ValueError("spl_list must be a non-empty list")
    
    # Calculate using both methods
    table_result = calculate_total_spl_multiple_sources(spl_list, use_table=True)
    exact_result = calculate_total_spl_multiple_sources(spl_list, use_table=False)
    
    table_spl = table_result['result']
    exact_spl = exact_result['result']
    difference = abs(table_spl - exact_spl)
    relative_error = (difference / exact_spl) * 100
    
    comparison = {
        'table_method': {
            'total_spl': float(table_spl),
            'steps': table_result['metadata']['calculation_steps']
        },
        'exact_method': {
            'total_spl': float(exact_spl),
            'steps': exact_result['metadata']['calculation_steps']
        },
        'comparison': {
            'absolute_difference': float(difference),
            'relative_error_percent': float(relative_error),
            'agreement': 'excellent' if difference < 0.5 else 'good' if difference < 1.0 else 'acceptable'
        }
    }
    
    # Save comparison to file
    output_file = './mid_result/acoustics/method_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return {
        'result': comparison,
        'metadata': {
            'input_values': spl_list,
            'comparison_file': output_file
        }
    }


def calculate_spl_at_distance(source_spl: float, 
                              source_distance: float,
                              target_distance: float,
                              environment: str = 'free_field') -> Dict[str, Union[float, dict]]:
    """
    Calculate SPL at a different distance from the source.
    
    Free field: SPL2 = SPL1 - 20*log10(r2/r1)
    Diffuse field: SPL2 = SPL1 - 10*log10(r2/r1)
    
    Args:
        source_spl: SPL at source distance (dB)
        source_distance: Distance where source_spl is measured (m)
        target_distance: Distance where SPL is to be calculated (m)
        environment: 'free_field' or 'diffuse_field'
        
    Returns:
        Dictionary with SPL at target distance and metadata
        
    Raises:
        ValueError: If distances are not positive or environment is invalid
    """
    if not all(isinstance(x, (int, float)) and x > 0 for x in [source_distance, target_distance]):
        raise ValueError("Distances must be positive numbers")
    
    if environment not in ['free_field', 'diffuse_field']:
        raise ValueError("Environment must be 'free_field' or 'diffuse_field'")
    
    distance_ratio = target_distance / source_distance
    
    if environment == 'free_field':
        attenuation = 20 * np.log10(distance_ratio)
    else:  # diffuse_field
        attenuation = 10 * np.log10(distance_ratio)
    
    target_spl = source_spl - attenuation
    
    return {
        'result': float(target_spl),
        'metadata': {
            'unit': 'dB',
            'source_spl': source_spl,
            'source_distance': source_distance,
            'target_distance': target_distance,
            'environment': environment,
            'attenuation': float(attenuation),
            'formula': f'{20 if environment == "free_field" else 10}*log10(r2/r1)'
        }
    }


# ============================================================================
# LAYER 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_spl_combination_process(spl_list: List[float], 
                                 use_table: bool = True,
                                 save_path: str = None,
                                 precomputed_steps: List[Dict[str, float]] = None,
                                 sorted_spl: List[float] = None,
                                 total_spl: float = None) -> Dict[str, Union[str, dict]]:
    """
    Visualize the step-by-step SPL combination process.
    
    Args:
        spl_list: List of sound pressure levels (dB)
        use_table: If True, use lookup table; if False, use exact formula
        save_path: Optional custom save path
        
    Returns:
        Dictionary with image path and metadata
    """
    if not isinstance(spl_list, list) or len(spl_list) == 0:
        raise ValueError("spl_list must be a non-empty list")
    
    # Calculate total SPL
    if precomputed_steps is not None:
        steps = precomputed_steps
        sorted_values = sorted_spl if sorted_spl is not None else sorted(spl_list, reverse=True)
        total_spl_value = total_spl if total_spl is not None else (steps[-1]['result'] if steps else sorted_values[0])
    else:
        result = calculate_total_spl_multiple_sources(spl_list, use_table)
        steps = result['metadata']['calculation_steps']
        sorted_values = result['metadata']['sorted_values']
        total_spl_value = result['result']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Step-by-step combination
    step_numbers = [0] + [s['step'] for s in steps]
    initial_value = (sorted_values[0] if 'sorted_values' in locals() else sorted(spl_list, reverse=True)[0])
    spl_values = [initial_value] + [s['result'] for s in steps]
    
    ax1.plot(step_numbers, spl_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Combination Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative SPL (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('SPL Combination Process', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(step_numbers)
    
    # Add annotations
    for i, (step, spl) in enumerate(zip(step_numbers, spl_values)):
        ax1.annotate(f'{spl:.2f} dB', 
                    xy=(step, spl), 
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9)
    
    # Plot 2: Individual sources vs total
    sources = [f'Source {i+1}' for i in range(len(spl_list))] + ['Total']
    values = spl_list + [total_spl_value]
    colors = ['#A23B72'] * len(spl_list) + ['#F18F01']
    
    bars = ax2.bar(sources, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('SPL (dB)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Sources vs Total SPL', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, max(values) * 1.15])
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = './tool_images/spl_combination_process.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    metadata_total = total_spl_value if total_spl_value is not None else values[-1]
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'image/png',
            'input_values': spl_list,
            'total_spl': metadata_total,
            'method': 'table_lookup' if use_table else 'exact_formula',
            'steps_count': len(steps)
        }
    }


def plot_spl_distance_attenuation(source_spl: float,
                                  source_distance: float,
                                  max_distance: float = 100.0,
                                  save_path: str = None) -> Dict[str, Union[str, dict]]:
    """
    Plot SPL attenuation with distance for both free field and diffuse field.
    
    Args:
        source_spl: SPL at source distance (dB)
        source_distance: Distance where source_spl is measured (m)
        max_distance: Maximum distance to plot (m)
        save_path: Optional custom save path
        
    Returns:
        Dictionary with image path and metadata
    """
    if not all(isinstance(x, (int, float)) and x > 0 for x in [source_distance, max_distance]):
        raise ValueError("Distances must be positive numbers")
    
    # Generate distance array
    distances = np.linspace(source_distance, max_distance, 200)
    
    # Calculate SPL for both environments
    spl_free_field = []
    spl_diffuse_field = []
    
    for dist in distances:
        result_free = calculate_spl_at_distance(source_spl, source_distance, dist, 'free_field')
        result_diffuse = calculate_spl_at_distance(source_spl, source_distance, dist, 'diffuse_field')
        spl_free_field.append(result_free['result'])
        spl_diffuse_field.append(result_diffuse['result'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(distances, spl_free_field, '-', linewidth=2, label='Free Field (-20 dB per decade)', color='#2E86AB')
    ax.plot(distances, spl_diffuse_field, '--', linewidth=2, label='Diffuse Field (-10 dB per decade)', color='#A23B72')
    ax.axvline(source_distance, color='red', linestyle=':', linewidth=1.5, label=f'Source Distance ({source_distance} m)')
    
    ax.set_xlabel('Distance (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SPL (dB)', fontsize=12, fontweight='bold')
    ax.set_title(f'SPL Attenuation with Distance\n(Source: {source_spl} dB at {source_distance} m)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = './tool_images/spl_distance_attenuation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'image/png',
            'source_spl': source_spl,
            'source_distance': source_distance,
            'max_distance': max_distance,
            'distance_points': len(distances)
        }
    }


def plot_lookup_table_visualization(save_path: str = None) -> Dict[str, Union[str, dict]]:
    """
    Visualize the SPL lookup table with interpolation curve.
    
    Args:
        save_path: Optional custom save path
        
    Returns:
        Dictionary with image path and metadata
    """
    differences = list(SPL_LOOKUP_TABLE.keys())
    increases = list(SPL_LOOKUP_TABLE.values())
    
    # Generate interpolated curve
    diff_interp = np.linspace(min(differences), max(differences), 100)
    inc_interp = [get_spl_increase_from_table(d)['result'] for d in diff_interp]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(diff_interp, inc_interp, '-', linewidth=2, color='#2E86AB', label='Interpolated Curve')
    ax.plot(differences, increases, 'o', markersize=10, color='#F18F01', 
            label='Table Values', zorder=5)
    
    # Add value annotations
    for diff, inc in zip(differences, increases):
        ax.annotate(f'{inc} dB', 
                   xy=(diff, inc), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold')
    
    ax.set_xlabel('Sound Pressure Level Difference (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Increase Value (dB)', fontsize=12, fontweight='bold')
    ax.set_title('SPL Addition Lookup Table', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([2.5, 7.5])
    ax.set_ylim([0.5, 2.0])
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = './tool_images/spl_lookup_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'image/png',
            'table_entries': len(SPL_LOOKUP_TABLE),
            'difference_range': [min(differences), max(differences)],
            'increase_range': [min(increases), max(increases)]
        }
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstrate the acoustics toolkit with three scenarios.
    """
    
    print("=" * 80)
    print("ACOUSTICS TOOLKIT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SCENARIO 1: Original Problem - Calculate Total SPL from Three Sources
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 1: Calculate Total SPL from Three Sound Sources")
    print("=" * 80)
    print("Problem: Three sound sources at point s have SPLs of 70 dB, 75 dB, and 65 dB.")
    print("Find the total sound pressure level using the provided lookup table.")
    print("-" * 80)
    
    # Input data
    spl_sources = [70, 75, 65]
    print(f"Input SPL values: {spl_sources} dB")
    print()
    
    # Step 1: Calculate total SPL using table lookup method (原子函数迭代)
    print("Step 1: 使用原子函数逐步叠加声压级（表格法）")
    sorted_values = sorted(spl_sources, reverse=True)
    print(f"Sorted SPL values (descending): {sorted_values}")
    
    calculation_steps = []
    current_spl = sorted_values[0]
    
    for idx, next_spl in enumerate(sorted_values[1:], 1):
        pair_result = add_two_spl_values(current_spl, next_spl, use_table=True)
        step_info = {
            'step': idx,
            'spl_current': float(current_spl),
            'spl_added': float(next_spl),
            'difference': float(current_spl - next_spl),
            'increase': pair_result['metadata']['increase_applied'],
            'result': pair_result['result']
        }
        calculation_steps.append(step_info)
        current_spl = pair_result['result']
    
    total_spl_table = current_spl
    print(f"\nTotal SPL (table method): {total_spl_table:.1f} dB\n")
    
    steps_file = './mid_result/acoustics/spl_calculation_steps_manual.json'
    with open(steps_file, 'w') as f:
        json.dump({
            'input_spl_values': spl_sources,
            'sorted_spl_values': sorted_values,
            'calculation_steps': calculation_steps,
            'final_total_spl': float(total_spl_table),
            'method': 'table_lookup'
        }, f, indent=2)
    print(f"Calculation steps saved to: {steps_file}\n")
    
    # Step 2: Show detailed calculation steps
    print("Step 2: 逐步叠加过程")
    for step in calculation_steps:
        print(f"  Step {step['step']}: Combine {step['spl_current']:.1f} dB + {step['spl_added']:.1f} dB")
        print(f"    Difference: {step['difference']:.1f} dB → Increase: {step['increase']:.2f} dB")
        print(f"    Result: {step['result']:.2f} dB")
    print()
    
    # Step 3: Verify with exact formula using原子函数
    print("Step 3: 使用原子函数验证精确对数公式")
    pressures = []
    for spl in spl_sources:
        pressure_info = convert_spl_to_pressure(spl)
        pressures.append(pressure_info['result'])
    
    # 对于互不相关的声源，等效总压为均方根和
    total_effective_pressure = np.sqrt(np.sum(np.square(pressures)))
    total_spl_exact = convert_pressure_to_spl(total_effective_pressure)['result']
    
    print(f"Total SPL (exact formula): {total_spl_exact:.1f} dB")
    print(f"Difference between methods: {abs(total_spl_table - total_spl_exact):.3f} dB\n")
    
    # Step 4: Visualize the combination process
    print("Step 4: 可视化叠加过程")
    print("Function call: plot_spl_combination_process()")
    viz_result = plot_spl_combination_process(
        spl_sources, 
        use_table=True,
        precomputed_steps=calculation_steps,
        sorted_spl=sorted_values,
        total_spl=total_spl_table
    )
    print(f"FUNCTION_CALL: plot_spl_combination_process | PARAMS: {{'spl_list': {spl_sources}, 'use_table': True, 'precomputed_steps': 'provided'}} | RESULT: {viz_result}")
    print()
    
    print("-" * 80)
    print(f"FINAL_ANSWER: {total_spl_table:.1f} dB")
    print("=" * 80)
    print()
    print()
    
    # ========================================================================
    # SCENARIO 2: Compare Table vs Exact Methods for Multiple Configurations
    # ========================================================================
    print("=" * 80)
    print("SCENARIO 2: Method Comparison for Different SPL Configurations")
    print("=" * 80)
    print("Problem: Compare table lookup vs exact formula for various SPL combinations")
    print("to understand the accuracy and applicability of the lookup table method.")
    print("-" * 80)
    
    # Test configurations
    test_configs = [
        [80, 80, 80],  # Equal sources
        [90, 85, 75],  # Large differences
        [70, 69, 68],  # Small differences
    ]
    
    print("Test configurations:")
    for i, config in enumerate(test_configs, 1):
        print(f"  Config {i}: {config} dB")
    print()
    
    comparison_results = []
    for i, config in enumerate(test_configs, 1):
        print(f"Configuration {i}: {config} dB")
        print("Function call: compare_table_vs_exact_methods()")
        comp_result = compare_table_vs_exact_methods(config)
        print(f"FUNCTION_CALL: compare_table_vs_exact_methods | PARAMS: {{'spl_list': {config}}} | RESULT: {comp_result}")
        print()
        
        comparison = comp_result['result']['comparison']
        table_spl = comp_result['result']['table_method']['total_spl']
        exact_spl = comp_result['result']['exact_method']['total_spl']
        
        print(f"  Table method: {table_spl:.2f} dB")
        print(f"  Exact method: {exact_spl:.2f} dB")
        print(f"  Difference: {comparison['absolute_difference']:.3f} dB ({comparison['relative_error_percent']:.3f}%)")
        print(f"  Agreement: {comparison['agreement']}")
        print()
        
        comparison_results.append({
            'config': config,
            'table_spl': table_spl,
            'exact_spl': exact_spl,
            'difference': comparison['absolute_difference']
        })
    
if __name__ == "__main__":
    main()