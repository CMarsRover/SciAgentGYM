# Filename: waste_calorific_value_toolkit.py

"""
Municipal Solid Waste (MSW) Calorific Value Calculation Toolkit

This toolkit provides functions for calculating the weighted average calorific value
of municipal solid waste based on composition data. It follows a three-layer architecture:
1. Atomic functions: Basic calculations and data validation
2. Composite functions: Multi-step calculations combining atomic functions
3. Visualization functions: Data presentation and analysis plots

All functions return standardized dict format: {'result': value, 'metadata': {...}}
"""

import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# 配置matplotlib字体，优先使用 DejaVu Sans，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# Global constants
MID_RESULT_DIR = "./mid_result/environmental_engineering"
IMAGE_DIR = "./tool_images"

# Create directories if they don't exist
os.makedirs(MID_RESULT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ============================================================================
# LAYER 1: ATOMIC FUNCTIONS - Basic calculations and data validation
# ============================================================================

def validate_waste_component(component_name: str, weight_percentage: float, 
                            heat_value: float) -> dict:
    """
    Validate a single waste component's data.
    
    Args:
        component_name: Name of the waste component
        weight_percentage: Weight percentage (0-100)
        heat_value: Calorific value in kJ/kg (>= 0)
    
    Returns:
        dict: {'result': bool, 'metadata': {'errors': list}}
    
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    if not isinstance(component_name, str) or not component_name.strip():
        errors.append("Component name must be a non-empty string")
    
    if not isinstance(weight_percentage, (int, float)):
        errors.append("Weight percentage must be numeric")
    elif not 0 <= weight_percentage <= 100:
        errors.append(f"Weight percentage must be between 0 and 100, got {weight_percentage}")
    
    if not isinstance(heat_value, (int, float)):
        errors.append("Heat value must be numeric")
    elif heat_value < 0:
        errors.append(f"Heat value must be non-negative, got {heat_value}")
    
    is_valid = len(errors) == 0
    
    return {
        'result': is_valid,
        'metadata': {
            'component': component_name,
            'errors': errors,
            'weight_percentage': weight_percentage,
            'heat_value': heat_value
        }
    }


def calculate_component_contribution(weight_percentage: float, heat_value: float) -> dict:
    """
    Calculate the calorific value contribution of a single component.
    
    Formula: Contribution = (weight_percentage / 100) * heat_value
    
    Args:
        weight_percentage: Weight percentage of the component (0-100)
        heat_value: Calorific value in kJ/kg
    
    Returns:
        dict: {
            'result': float (contribution in kJ/kg),
            'metadata': {
                'weight_percentage': float,
                'heat_value': float,
                'formula': str
            }
        }
    """
    if not 0 <= weight_percentage <= 100:
        raise ValueError(f"Weight percentage must be between 0 and 100, got {weight_percentage}")
    
    if heat_value < 0:
        raise ValueError(f"Heat value must be non-negative, got {heat_value}")
    
    contribution = (weight_percentage / 100.0) * heat_value
    
    return {
        'result': contribution,
        'metadata': {
            'weight_percentage': weight_percentage,
            'heat_value': heat_value,
            'formula': 'contribution = (weight_percentage / 100) * heat_value'
        }
    }


def sum_contributions(contributions: List[float]) -> dict:
    """
    Sum all component contributions to get total calorific value.
    
    Args:
        contributions: List of individual component contributions in kJ/kg
    
    Returns:
        dict: {
            'result': float (total calorific value in kJ/kg),
            'metadata': {
                'num_components': int,
                'individual_contributions': list,
                'formula': str
            }
        }
    """
    if not isinstance(contributions, list):
        raise TypeError("Contributions must be a list")
    
    if not all(isinstance(c, (int, float)) for c in contributions):
        raise TypeError("All contributions must be numeric")
    
    if not all(c >= 0 for c in contributions):
        raise ValueError("All contributions must be non-negative")
    
    total = sum(contributions)
    
    return {
        'result': total,
        'metadata': {
            'num_components': len(contributions),
            'individual_contributions': contributions,
            'formula': 'total = sum(all_contributions)'
        }
    }


def validate_total_percentage(weight_percentages: List[float], 
                             tolerance: float = 0.01) -> dict:
    """
    Validate that total weight percentages sum to approximately 100%.
    
    Args:
        weight_percentages: List of weight percentages
        tolerance: Acceptable deviation from 100% (default: 0.01)
    
    Returns:
        dict: {
            'result': bool (True if valid),
            'metadata': {
                'total_percentage': float,
                'expected': float,
                'deviation': float,
                'is_valid': bool
            }
        }
    """
    total = sum(weight_percentages)
    deviation = abs(total - 100.0)
    is_valid = deviation <= tolerance
    
    return {
        'result': is_valid,
        'metadata': {
            'total_percentage': total,
            'expected': 100.0,
            'deviation': deviation,
            'tolerance': tolerance,
            'is_valid': is_valid
        }
    }


# ============================================================================
# LAYER 2: COMPOSITE FUNCTIONS - Multi-step calculations
# ============================================================================

def calculate_waste_calorific_value(waste_data: List[Dict[str, float]]) -> dict:
    """
    Calculate the weighted average calorific value of municipal solid waste.
    
    This function combines validation, contribution calculation, and summation
    to compute the overall calorific value.
    
    Args:
        waste_data: List of dicts, each containing:
            - 'component': str (component name)
            - 'weight_percentage': float (0-100)
            - 'heat_value': float (kJ/kg, >= 0 for combustible, 0 for incombustible)
    
    Returns:
        dict: {
            'result': float (total calorific value in kJ/kg),
            'metadata': {
                'num_components': int,
                'total_weight_percentage': float,
                'combustible_percentage': float,
                'incombustible_percentage': float,
                'component_contributions': list,
                'validation_passed': bool
            }
        }
    """
    # Step 1: Validate all components
    all_valid = True
    weight_percentages = []
    contributions = []
    component_details = []
    
    for component_data in waste_data:
        component_name = component_data['component']
        weight_pct = component_data['weight_percentage']
        heat_val = component_data['heat_value']
        
        # Validate component
        validation = validate_waste_component(component_name, weight_pct, heat_val)
        if not validation['result']:
            all_valid = False
            raise ValueError(f"Validation failed for {component_name}: {validation['metadata']['errors']}")
        
        weight_percentages.append(weight_pct)
        
        # Calculate contribution
        contrib = calculate_component_contribution(weight_pct, heat_val)
        contributions.append(contrib['result'])
        
        component_details.append({
            'component': component_name,
            'weight_percentage': weight_pct,
            'heat_value': heat_val,
            'contribution': contrib['result']
        })
    
    # Step 2: Validate total percentage
    percentage_validation = validate_total_percentage(weight_percentages)
    if not percentage_validation['result']:
        raise ValueError(f"Total weight percentage validation failed: {percentage_validation['metadata']}")
    
    # Step 3: Sum contributions
    total_result = sum_contributions(contributions)
    
    # Calculate combustible vs incombustible percentages
    combustible_pct = sum(d['weight_percentage'] for d in waste_data if d['heat_value'] > 0)
    incombustible_pct = sum(d['weight_percentage'] for d in waste_data if d['heat_value'] == 0)
    
    return {
        'result': total_result['result'],
        'metadata': {
            'num_components': len(waste_data),
            'total_weight_percentage': percentage_validation['metadata']['total_percentage'],
            'combustible_percentage': combustible_pct,
            'incombustible_percentage': incombustible_pct,
            'component_contributions': component_details,
            'validation_passed': all_valid
        }
    }


def analyze_waste_composition(waste_data: List[Dict[str, float]]) -> dict:
    """
    Perform comprehensive analysis of waste composition.
    
    Args:
        waste_data: List of waste component dictionaries
    
    Returns:
        dict: {
            'result': dict with analysis results,
            'metadata': {
                'total_calorific_value': float,
                'combustible_components': list,
                'incombustible_components': list,
                'highest_contributor': dict,
                'lowest_contributor': dict
            }
        }
    """
    # Calculate total calorific value
    calorific_result = calculate_waste_calorific_value(waste_data)
    
    # Separate combustible and incombustible
    combustible = [d for d in waste_data if d['heat_value'] > 0]
    incombustible = [d for d in waste_data if d['heat_value'] == 0]
    
    # Find highest and lowest contributors (among combustible)
    contributions = []
    for comp in combustible:
        contrib = calculate_component_contribution(
            comp['weight_percentage'], 
            comp['heat_value']
        )
        contributions.append({
            'component': comp['component'],
            'contribution': contrib['result'],
            'weight_percentage': comp['weight_percentage'],
            'heat_value': comp['heat_value']
        })
    
    if contributions:
        highest = max(contributions, key=lambda x: x['contribution'])
        lowest = min(contributions, key=lambda x: x['contribution'])
    else:
        highest = lowest = None
    
    analysis = {
        'total_calorific_value_kJ_per_kg': calorific_result['result'],
        'num_combustible_components': len(combustible),
        'num_incombustible_components': len(incombustible),
        'combustible_weight_percentage': sum(c['weight_percentage'] for c in combustible),
        'incombustible_weight_percentage': sum(c['weight_percentage'] for c in incombustible)
    }
    
    return {
        'result': analysis,
        'metadata': {
            'total_calorific_value': calorific_result['result'],
            'combustible_components': combustible,
            'incombustible_components': incombustible,
            'highest_contributor': highest,
            'lowest_contributor': lowest,
            'detailed_contributions': calorific_result['metadata']['component_contributions']
        }
    }


def save_calculation_report(waste_data: List[Dict[str, float]], 
                           output_filename: str = "waste_calorific_calculation.json") -> dict:
    """
    Save detailed calculation report to JSON file.
    
    Args:
        waste_data: List of waste component dictionaries
        output_filename: Name of output file
    
    Returns:
        dict: {
            'result': str (filepath),
            'metadata': {
                'file_type': str,
                'size': int (bytes),
                'num_components': int
            }
        }
    """
    # Perform analysis
    analysis = analyze_waste_composition(waste_data)
    
    # Prepare report
    report = {
        'summary': analysis['result'],
        'detailed_contributions': analysis['metadata']['detailed_contributions'],
        'highest_contributor': analysis['metadata']['highest_contributor'],
        'lowest_contributor': analysis['metadata']['lowest_contributor'],
        'raw_data': waste_data
    }
    
    # Save to file
    filepath = os.path.join(MID_RESULT_DIR, output_filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(filepath)
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'json',
            'size': file_size,
            'num_components': len(waste_data)
        }
    }


# ============================================================================
# LAYER 3: VISUALIZATION FUNCTIONS - Data presentation
# ============================================================================

def plot_waste_composition_pie(waste_data: List[Dict[str, float]], 
                               output_filename: str = "waste_composition_pie.png") -> dict:
    """
    Create pie chart showing waste composition by weight percentage.
    
    Args:
        waste_data: List of waste component dictionaries
        output_filename: Name of output image file
    
    Returns:
        dict: {
            'result': str (filepath),
            'metadata': {
                'file_type': str,
                'chart_type': str,
                'num_components': int
            }
        }
    """
    components = [d['component'] for d in waste_data]
    percentages = [d['weight_percentage'] for d in waste_data]
    
    # Create color map (combustible vs incombustible)
    colors = ['#ff9999' if d['heat_value'] > 0 else '#cccccc' for d in waste_data]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        percentages, 
        labels=components, 
        autopct='%1.2f%%',
        colors=colors,
        startangle=90
    )
    
    # Improve text readability
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)
        autotext.set_weight('bold')
    
    ax.set_title('Municipal Solid Waste Composition by Weight Percentage', 
                 fontsize=14, weight='bold')
    
    # Add legend
    legend_labels = ['Combustible Components', 'Incombustible Components']
    legend_colors = ['#ff9999', '#cccccc']
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    filepath = os.path.join(IMAGE_DIR, output_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'chart_type': 'pie_chart',
            'num_components': len(waste_data)
        }
    }


def plot_calorific_contribution_bar(waste_data: List[Dict[str, float]], 
                                   output_filename: str = "calorific_contribution_bar.png") -> dict:
    """
    Create bar chart showing calorific value contribution of each component.
    
    Args:
        waste_data: List of waste component dictionaries
        output_filename: Name of output image file
    
    Returns:
        dict: {
            'result': str (filepath),
            'metadata': {
                'file_type': str,
                'chart_type': str,
                'total_calorific_value': float
            }
        }
    """
    # Calculate contributions
    components = []
    contributions = []
    
    for comp in waste_data:
        contrib = calculate_component_contribution(
            comp['weight_percentage'], 
            comp['heat_value']
        )
        components.append(comp['component'])
        contributions.append(contrib['result'])
    
    # Sort by contribution (descending)
    sorted_data = sorted(zip(components, contributions), key=lambda x: x[1], reverse=True)
    components_sorted, contributions_sorted = zip(*sorted_data) if sorted_data else ([], [])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if c > 0 else '#95a5a6' for c in contributions_sorted]
    bars = ax.bar(range(len(components_sorted)), contributions_sorted, color=colors)
    
    ax.set_xlabel('Waste Component', fontsize=12, weight='bold')
    ax.set_ylabel('Calorific Value Contribution (kJ/kg)', fontsize=12, weight='bold')
    ax.set_title('Calorific Value Contribution by Waste Component', 
                 fontsize=14, weight='bold')
    ax.set_xticks(range(len(components_sorted)))
    ax.set_xticklabels(components_sorted, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, contributions_sorted)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(IMAGE_DIR, output_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    total_calorific = sum(contributions)
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'chart_type': 'bar_chart',
            'total_calorific_value': total_calorific
        }
    }


def plot_heat_value_vs_percentage(waste_data: List[Dict[str, float]], 
                                 output_filename: str = "heat_value_vs_percentage.png") -> dict:
    """
    Create scatter plot showing relationship between heat value and weight percentage.
    
    Args:
        waste_data: List of waste component dictionaries
        output_filename: Name of output image file
    
    Returns:
        dict: {
            'result': str (filepath),
            'metadata': {
                'file_type': str,
                'chart_type': str,
                'num_combustible': int
            }
        }
    """
    # Filter combustible components only
    combustible = [d for d in waste_data if d['heat_value'] > 0]
    
    if not combustible:
        raise ValueError("No combustible components found in waste data")
    
    heat_values = [d['heat_value'] for d in combustible]
    percentages = [d['weight_percentage'] for d in combustible]
    components = [d['component'] for d in combustible]
    
    # Calculate contributions for bubble size
    contributions = []
    for comp in combustible:
        contrib = calculate_component_contribution(
            comp['weight_percentage'], 
            comp['heat_value']
        )
        contributions.append(contrib['result'])
    
    # Normalize contributions for bubble size (50-500 range)
    max_contrib = max(contributions) if contributions else 1
    bubble_sizes = [50 + 450 * (c / max_contrib) for c in contributions]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(percentages, heat_values, s=bubble_sizes, 
                        alpha=0.6, c=contributions, cmap='YlOrRd', edgecolors='black')
    
    # Add component labels
    for i, comp in enumerate(components):
        ax.annotate(comp, (percentages[i], heat_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Weight Percentage (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Heat Value (kJ/kg)', fontsize=12, weight='bold')
    ax.set_title('Heat Value vs Weight Percentage (Bubble size = Contribution)', 
                 fontsize=14, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Calorific Contribution (kJ/kg)', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(IMAGE_DIR, output_filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'chart_type': 'scatter_plot',
            'num_combustible': len(combustible)
        }
    }


# ============================================================================
# FILE LOADING UTILITIES
# ============================================================================

def load_json_file(filepath: str) -> dict:
    """
    Load and parse JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: {
            'result': dict (parsed JSON content),
            'metadata': {
                'filepath': str,
                'file_size': int
            }
        }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    file_size = os.path.getsize(filepath)
    
    return {
        'result': content,
        'metadata': {
            'filepath': filepath,
            'file_size': file_size
        }
    }


# ============================================================================
# MAIN FUNCTION - Demonstration scenarios
# ============================================================================

def main():
    """
    Demonstrate the waste calorific value calculation toolkit with three scenarios.
    """
    
    # ========================================================================
    # Scenario 1: Calculate calorific value for the original problem
    # ========================================================================
    print("=" * 60)
    print("Scenario 1: Calculate Municipal Solid Waste Calorific Value")
    print("=" * 60)
    print("Problem: Calculate the weighted average calorific value of waste")
    print("based on composition data from a Chinese city.")
    print("-" * 60)
    
    # Define waste composition data from the table
    waste_composition = [
        # Combustible components
        {'component': 'Kitchen Waste, Fruit Peels', 'weight_percentage': 30.01, 'heat_value': 4650},
        {'component': 'Wood and Grass', 'weight_percentage': 2.50, 'heat_value': 6510},
        {'component': 'Paper', 'weight_percentage': 1.76, 'heat_value': 16750},
        {'component': 'Leather, Plastic, Rubber, Cotton', 'weight_percentage': 2.39, 'heat_value': 32560},
        # Incombustible components (heat value = 0)
        {'component': 'Ash, Clay', 'weight_percentage': 61.38, 'heat_value': 0},
        {'component': 'Ceramic Brick, Stone', 'weight_percentage': 1.96, 'heat_value': 0}
    ]
    
    # Step 1: Validate total percentage
    print("\nStep 1: Validate total weight percentage")
    weight_percentages = [comp['weight_percentage'] for comp in waste_composition]
    validation_result = validate_total_percentage(weight_percentages)
    print(f"FUNCTION_CALL: validate_total_percentage | PARAMS: {weight_percentages} | RESULT: {validation_result}")
    
    # Step 2: Calculate calorific value
    print("\nStep 2: Calculate weighted average calorific value")
    calorific_result = calculate_waste_calorific_value(waste_composition)
    print(f"FUNCTION_CALL: calculate_waste_calorific_value | PARAMS: {len(waste_composition)} components | RESULT: {calorific_result}")
    
    # Step 3: Display detailed breakdown
    print("\nStep 3: Display component contributions")
    for contrib in calorific_result['metadata']['component_contributions']:
        print(f"  {contrib['component']}: {contrib['contribution']:.3f} kJ/kg")
    
    print(f"\nTotal Calorific Value: {calorific_result['result']:.3f} kJ/kg")
    print(f"Standard Answer: 2631.199 kJ/kg")
    print(f"Difference: {abs(calorific_result['result'] - 2631.199):.6f} kJ/kg")
    
    print(f"\nFINAL_ANSWER: {calorific_result['result']:.3f} kJ/kg")
    
    # ========================================================================
    # Scenario 2: Comprehensive waste composition analysis with visualization
    # ========================================================================
    print("\n" + "=" * 60)
    print("Scenario 2: Comprehensive Waste Composition Analysis")
    print("=" * 60)
    print("Problem: Analyze waste composition, identify key contributors,")
    print("and generate visual reports.")
    print("-" * 60)
    
    # Step 1: Perform comprehensive analysis
    print("\nStep 1: Analyze waste composition")
    analysis_result = analyze_waste_composition(waste_composition)
    print(f"FUNCTION_CALL: analyze_waste_composition | PARAMS: {len(waste_composition)} components | RESULT: {analysis_result}")
    
    print("\nAnalysis Summary:")
    print(f"  Total Calorific Value: {analysis_result['result']['total_calorific_value_kJ_per_kg']:.3f} kJ/kg")
    print(f"  Combustible Components: {analysis_result['result']['num_combustible_components']}")
    print(f"  Incombustible Components: {analysis_result['result']['num_incombustible_components']}")
    print(f"  Combustible Weight %: {analysis_result['result']['combustible_weight_percentage']:.2f}%")
    print(f"  Incombustible Weight %: {analysis_result['result']['incombustible_weight_percentage']:.2f}%")
    
    # Step 2: Identify highest and lowest contributors
    print("\nStep 2: Identify key contributors")
    highest = analysis_result['metadata']['highest_contributor']
    lowest = analysis_result['metadata']['lowest_contributor']
    print(f"  Highest Contributor: {highest['component']} ({highest['contribution']:.3f} kJ/kg)")
    print(f"  Lowest Contributor: {lowest['component']} ({lowest['contribution']:.3f} kJ/kg)")
    
    # Step 3: Generate pie chart
    print("\nStep 3: Generate composition pie chart")
    pie_result = plot_waste_composition_pie(waste_composition)
    print(f"FUNCTION_CALL: plot_waste_composition_pie | PARAMS: {len(waste_composition)} components | RESULT: {pie_result}")
    
    # Step 4: Generate bar chart
    print("\nStep 4: Generate calorific contribution bar chart")
    bar_result = plot_calorific_contribution_bar(waste_composition)
    print(f"FUNCTION_CALL: plot_calorific_contribution_bar | PARAMS: {len(waste_composition)} components | RESULT: {bar_result}")
    
    # Step 5: Generate scatter plot
    print("\nStep 5: Generate heat value vs percentage scatter plot")
    scatter_result = plot_heat_value_vs_percentage(waste_composition)
    print(f"FUNCTION_CALL: plot_heat_value_vs_percentage | PARAMS: combustible components only | RESULT: {scatter_result}")
    
    # Step 6: Save calculation report
    print("\nStep 6: Save detailed calculation report")
    report_result = save_calculation_report(waste_composition)
    print(f"FUNCTION_CALL: save_calculation_report | PARAMS: {len(waste_composition)} components | RESULT: {report_result}")
    
    print(f"\nFINAL_ANSWER: Analysis complete. Total calorific value: {analysis_result['result']['total_calorific_value_kJ_per_kg']:.3f} kJ/kg")
    
    # ========================================================================
    # Scenario 3: Sensitivity analysis - Impact of composition changes
    # ========================================================================
    print("\n" + "=" * 60)
    print("Scenario 3: Sensitivity Analysis - Composition Variation Impact")
    print("=" * 60)
    print("Problem: Analyze how changes in waste composition affect")
    print("total calorific value (e.g., increased plastic content).")
    print("-" * 60)
    
    # Step 1: Create modified composition (increase plastic by 5%, decrease ash)
    print("\nStep 1: Create modified waste composition")
    print("Modification: Increase plastic content by 5%, decrease ash by 5%")
    
    modified_composition = [
        {'component': 'Kitchen Waste, Fruit Peels', 'weight_percentage': 30.01, 'heat_value': 4650},
        {'component': 'Wood and Grass', 'weight_percentage': 2.50, 'heat_value': 6510},
        {'component': 'Paper', 'weight_percentage': 1.76, 'heat_value': 16750},
        {'component': 'Leather, Plastic, Rubber, Cotton', 'weight_percentage': 7.39, 'heat_value': 32560},  # +5%
        {'component': 'Ash, Clay', 'weight_percentage': 56.38, 'heat_value': 0},  # -5%
        {'component': 'Ceramic Brick, Stone', 'weight_percentage': 1.96, 'heat_value': 0}
    ]
    
    # Step 2: Calculate new calorific value
    print("\nStep 2: Calculate calorific value for modified composition")
    modified_result = calculate_waste_calorific_value(modified_composition)
    print(f"FUNCTION_CALL: calculate_waste_calorific_value | PARAMS: modified composition | RESULT: {modified_result}")
    
    # Step 3: Compare with original
    print("\nStep 3: Compare original vs modified composition")
    original_value = calorific_result['result']
    modified_value = modified_result['result']
    difference = modified_value - original_value
    percent_change = (difference / original_value) * 100
    
    print(f"  Original Calorific Value: {original_value:.3f} kJ/kg")
    print(f"  Modified Calorific Value: {modified_value:.3f} kJ/kg")
    print(f"  Absolute Difference: {difference:.3f} kJ/kg")
    print(f"  Percent Change: {percent_change:.2f}%")
    
    # Step 4: Calculate contribution change for plastic component
    print("\nStep 4: Analyze plastic component contribution change")
    original_plastic = next(c for c in waste_composition if 'Plastic' in c['component'])
    modified_plastic = next(c for c in modified_composition if 'Plastic' in c['component'])
    
    original_plastic_contrib = calculate_component_contribution(
        original_plastic['weight_percentage'], 
        original_plastic['heat_value']
    )
    modified_plastic_contrib = calculate_component_contribution(
        modified_plastic['weight_percentage'], 
        modified_plastic['heat_value']
    )
    
    print(f"FUNCTION_CALL: calculate_component_contribution | PARAMS: original plastic | RESULT: {original_plastic_contrib}")
    print(f"FUNCTION_CALL: calculate_component_contribution | PARAMS: modified plastic | RESULT: {modified_plastic_contrib}")
    
    plastic_contrib_change = modified_plastic_contrib['result'] - original_plastic_contrib['result']
    print(f"  Plastic Contribution Change: {plastic_contrib_change:.3f} kJ/kg")
    
    # Step 5: Generate comparison visualization
    print("\nStep 5: Generate comparison bar chart")
    comparison_data = [
        {'component': 'Original Composition', 'weight_percentage': 100, 'heat_value': original_value},
        {'component': 'Modified Composition (+5% Plastic)', 'weight_percentage': 100, 'heat_value': modified_value}
    ]
    comparison_result = plot_calorific_contribution_bar(
        comparison_data, 
        output_filename="composition_comparison.png"
    )
    print(f"FUNCTION_CALL: plot_calorific_contribution_bar | PARAMS: comparison data | RESULT: {comparison_result}")
    
    print(f"\nFINAL_ANSWER: Increasing plastic content by 5% increases calorific value by {difference:.3f} kJ/kg ({percent_change:.2f}%)")


if __name__ == "__main__":
    main()