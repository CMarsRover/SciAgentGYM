# Filename: magnetic_materials_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

def extract_bh_curve_data(material_name, curve_points=None):
    """
    Extract B-H curve data for a specific material.
    
    This function either uses provided curve points or returns predefined data
    for common magnetic materials. The B-H curve represents the relationship
    between magnetic flux density (B) and magnetic field strength (H).
    
    Parameters:
    -----------
    material_name : str
        Name of the magnetic material (e.g., 'mild steel', 'cast iron')
    curve_points : dict, optional
        Dictionary containing 'H' and 'B' arrays with measured data points.
        If None, predefined curves will be used.
    
    Returns:
    --------
    tuple (H, B)
        H: numpy array of magnetic field strength values (A/m)
        B: numpy array of corresponding flux density values (T)
    """
    # Predefined B-H curve data points for common materials
    # These are approximate values extracted from typical B-H curves
    predefined_curves = {
        'mild steel': {
            'H': np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]),
            'B': np.array([0, 1.0, 1.25, 1.35, 1.4, 1.43, 1.45, 1.5, 1.52, 1.54])
        },
        'cast iron': {
            'H': np.array([0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]),
            'B': np.array([0, 0.2, 0.35, 0.5, 0.6, 0.67, 0.72, 0.76, 0.79])
        },
        'silicon iron': {
            'H': np.array([0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]),
            'B': np.array([0, 0.5, 0.9, 1.3, 1.5, 1.55, 1.58, 1.6, 1.62, 1.63])
        },
        'cast steel': {
            'H': np.array([0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]),
            'B': np.array([0, 0.5, 0.8, 1.2, 1.4, 1.55, 1.62, 1.67, 1.7, 1.72, 1.73])
        }
    }
    
    if curve_points is not None:
        return curve_points['H'], curve_points['B']
    
    material_name = material_name.lower()
    if material_name in predefined_curves:
        return predefined_curves[material_name]['H'], predefined_curves[material_name]['B']
    else:
        raise ValueError(f"Material '{material_name}' not found in predefined curves. Please provide curve_points.")

def calculate_flux_density(material_name, h_value, curve_points=None):
    """
    Calculate the magnetic flux density (B) for a given material and magnetic field strength (H).
    
    This function uses interpolation on the B-H curve to find the flux density
    at a specific field strength value.
    
    Parameters:
    -----------
    material_name : str
        Name of the magnetic material
    h_value : float
        Magnetic field strength (H) in A/m
    curve_points : dict, optional
        Dictionary containing 'H' and 'B' arrays with measured data points.
        If None, predefined curves will be used.
    
    Returns:
    --------
    float
        Magnetic flux density (B) in Tesla
    """
    H, B = extract_bh_curve_data(material_name, curve_points)
    
    # Create interpolation function
    interp_func = interp1d(H, B, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Calculate B for the given H
    b_value = float(interp_func(h_value))
    
    return b_value

def calculate_permeability(material_name, h_value, curve_points=None):
    """
    Calculate the relative permeability (μr) for a given material and magnetic field strength (H).
    
    Relative permeability is calculated as μr = B/(μ0*H), where μ0 is the permeability of free space.
    
    Parameters:
    -----------
    material_name : str
        Name of the magnetic material
    h_value : float
        Magnetic field strength (H) in A/m
    curve_points : dict, optional
        Dictionary containing 'H' and 'B' arrays with measured data points.
        If None, predefined curves will be used.
    
    Returns:
    --------
    float
        Relative permeability (μr), dimensionless
    """
    # Permeability of free space (μ0) in H/m
    mu_0 = 4 * np.pi * 1e-7
    
    # Calculate B for the given H
    b_value = calculate_flux_density(material_name, h_value, curve_points)
    
    # Calculate relative permeability
    if h_value == 0:
        # Avoid division by zero
        return None
    else:
        mu_r = b_value / (mu_0 * h_value)
        return mu_r

def plot_bh_curves(materials, h_range=(0, 7000), save_path=None):
    """
    Plot B-H curves for multiple materials.
    
    Parameters:
    -----------
    materials : list
        List of material names to plot
    h_range : tuple, optional
        Range of H values to plot (min, max)
    save_path : str, optional
        Path to save the plot. If None, the plot will not be saved.
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(10, 6))
    
    # Set up fonts for proper display of labels
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Generate H values for smooth curves
    H = np.linspace(h_range[0], h_range[1], 1000)
    
    for material in materials:
        try:
            # Get original data points
            H_orig, B_orig = extract_bh_curve_data(material)
            
            # Create interpolation function
            interp_func = interp1d(H_orig, B_orig, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # Calculate B values for smooth curve
            B = interp_func(H)
            
            # Plot the curve
            plt.plot(H, B, label=material.title())
        except ValueError as e:
            print(f"Error plotting {material}: {e}")
    
    plt.title('B-H Curves for Different Magnetic Materials')
    plt.xlabel('Magnetic Field Strength, H (A/m)')
    plt.ylabel('Flux Density, B (T)')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def analyze_material_permeability(material_name, h_values, curve_points=None):
    """
    Analyze the permeability of a material across different field strengths.
    
    Parameters:
    -----------
    material_name : str
        Name of the magnetic material
    h_values : array-like
        Array of magnetic field strength (H) values in A/m
    curve_points : dict, optional
        Dictionary containing 'H' and 'B' arrays with measured data points.
        If None, predefined curves will be used.
    
    Returns:
    --------
    dict
        Dictionary containing 'H', 'B', and 'μr' arrays
    """
    h_array = np.array(h_values)
    b_array = np.zeros_like(h_array, dtype=float)
    mu_r_array = np.zeros_like(h_array, dtype=float)
    
    for i, h in enumerate(h_array):
        b_array[i] = calculate_flux_density(material_name, h, curve_points)
        if h > 0:  # Avoid division by zero
            mu_r_array[i] = calculate_permeability(material_name, h, curve_points)
    
    return {
        'H': h_array,
        'B': b_array,
        'μr': mu_r_array
    }

def main():
    """
    Main function: Demonstrate the use of magnetic material analysis tools.
    """
    print("Magnetic Materials Analyzer")
    print("==========================")
    
    # Example 1: Calculate flux density for mild steel at H = 2500 A/m
    material = "mild steel"
    h_value = 2500
    b_value = calculate_flux_density(material, h_value)
    print(f"\nExample 1: Flux density calculation")
    print(f"Material: {material.title()}")
    print(f"Magnetic field strength (H): {h_value} A/m")
    print(f"Flux density (B): {b_value:.4f} T")
    
    # Example 2: Calculate permeability
    mu_r = calculate_permeability(material, h_value)
    print(f"\nExample 2: Permeability calculation")
    print(f"Relative permeability (μr) at H = {h_value} A/m: {mu_r:.2f}")
    
    # Example 3: Plot B-H curves for common materials
    materials = ['mild steel', 'cast iron', 'silicon iron', 'cast steel']
    plot_bh_curves(materials, save_path="./images/bh_curves.png")
    
    # Example 4: Analyze permeability across different field strengths
    h_values = np.linspace(100, 6000, 20)
    analysis = analyze_material_permeability(material, h_values)
    
    print(f"\nExample 4: Permeability analysis for {material.title()}")
    print("H (A/m)\tB (T)\tμr")
    print("------------------------")
    for i in range(len(h_values)):
        print(f"{analysis['H'][i]:.0f}\t{analysis['B'][i]:.4f}\t{analysis['μr'][i]:.2f}")

if __name__ == "__main__":
    main()