# Filename: pulley_system_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle, Arrow
from typing import Tuple, Dict, List, Optional, Union

def decompose_force(force_magnitude: float, angle_degrees: float) -> Tuple[float, float]:
    """
    Decompose a force vector into its x and y components.
    
    Parameters:
    -----------
    force_magnitude : float
        The magnitude of the force in Newtons (N)
    angle_degrees : float
        The angle in degrees measured from the positive x-axis (counterclockwise is positive)
    
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the x and y components of the force (Fx, Fy) in Newtons
    """
    angle_radians = np.radians(angle_degrees)
    force_x = force_magnitude * np.cos(angle_radians)
    force_y = force_magnitude * np.sin(angle_radians)
    return force_x, force_y

def calculate_pulley_system_acceleration(
    applied_force: float,
    applied_force_angle: float,
    mass_1: float,
    mass_2: float,
    friction_coefficient: float = 0.0,
    gravity: float = 9.8
) -> Dict[str, float]:
    """
    Calculate the acceleration and tension in a pulley system with two masses.
    
    This function solves for the dynamics of a pulley system where one mass (mass_1) 
    is on a horizontal surface (potentially with friction) and connected via a rope 
    and pulley to a hanging mass (mass_2). An external force can be applied to mass_1
    at an angle.
    
    Parameters:
    -----------
    applied_force : float
        The magnitude of the external force applied to mass_1 in Newtons (N)
    applied_force_angle : float
        The angle of the applied force in degrees, measured from the positive x-axis
    mass_1 : float
        The mass of the object on the horizontal surface in kilograms (kg)
    mass_2 : float
        The mass of the hanging object in kilograms (kg)
    friction_coefficient : float, optional
        The coefficient of kinetic friction between mass_1 and the surface (default: 0.0)
    gravity : float, optional
        The acceleration due to gravity in m/s² (default: 9.8)
    
    Returns:
    --------
    Dict[str, float]
        A dictionary containing:
        - 'acceleration': The magnitude of the acceleration of the system in m/s²
        - 'tension': The tension in the rope in Newtons (N)
        - 'normal_force': The normal force on mass_1 in Newtons (N)
        - 'friction_force': The friction force on mass_1 in Newtons (N)
    """
    # Decompose the applied force into x and y components
    force_x, force_y = decompose_force(applied_force, applied_force_angle)
    
    # Calculate the normal force on mass_1
    normal_force = mass_1 * gravity + force_y
    
    # Calculate the friction force (if any)
    friction_force = friction_coefficient * normal_force
    
    # Set up the equations of motion
    # For mass_1: F_x - T - f = m_1 * a
    # For mass_2: T - m_2 * g = m_2 * a
    
    # Solve for acceleration
    # From the two equations: F_x - friction - (m_2 * g + m_2 * a) = m_1 * a
    # Rearranging: F_x - friction - m_2 * g = m_1 * a + m_2 * a = (m_1 + m_2) * a
    
    acceleration = (force_x - friction_force - mass_2 * gravity) / (mass_1 + mass_2)
    
    # If the calculated acceleration would cause the system to move in the wrong direction,
    # then the system is in static equilibrium
    if acceleration < 0 and friction_coefficient > 0:
        # Recalculate with static friction (which would be just enough to prevent motion)
        acceleration = 0
        tension = mass_2 * gravity
        friction_force = force_x - tension
    else:
        # Calculate the tension using the equation for mass_2
        tension = mass_2 * (gravity + acceleration)
    
    return {
        'acceleration': acceleration,
        'tension': tension,
        'normal_force': normal_force,
        'friction_force': friction_force
    }

def visualize_pulley_system(
    applied_force: float,
    applied_force_angle: float,
    mass_1: float,
    mass_2: float,
    results: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the pulley system with forces, masses, and calculated results.
    
    Parameters:
    -----------
    applied_force : float
        The magnitude of the external force applied to mass_1 in Newtons (N)
    applied_force_angle : float
        The angle of the applied force in degrees, measured from the positive x-axis
    mass_1 : float
        The mass of the object on the horizontal surface in kilograms (kg)
    mass_2 : float
        The mass of the hanging object in kilograms (kg)
    results : Dict[str, float]
        Dictionary containing the calculated results from calculate_pulley_system_acceleration
    save_path : str, optional
        Path to save the visualization image. If None, the image is displayed but not saved.
    
    Returns:
    --------
    None
    """
    # Set up the figure
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Set up the axes
    ax = plt.gca()
    ax.set_xlim(-2, 10)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # Draw the surface
    plt.plot([-2, 10], [0, 0], 'k-', linewidth=2)
    
    # Draw mass_1 (on the surface)
    mass1_width, mass1_height = 2, 1.5
    mass1_x, mass1_y = 3, 0
    mass1_rect = Rectangle((mass1_x, mass1_y), mass1_width, mass1_height, 
                          facecolor='lightblue', edgecolor='black')
    ax.add_patch(mass1_rect)
    plt.text(mass1_x + mass1_width/2, mass1_y + mass1_height/2, f"{mass_1} kg", 
             ha='center', va='center')
    
    # Draw the pulley
    pulley_x, pulley_y = 7, 2
    pulley_radius = 0.5
    pulley = Circle((pulley_x, pulley_y), pulley_radius, facecolor='lightgray', edgecolor='black')
    ax.add_patch(pulley)
    
    # Draw the rope
    plt.plot([mass1_x + mass1_width, pulley_x], [mass1_y + mass1_height/2, pulley_y], 'k-', linewidth=1.5)
    plt.plot([pulley_x, pulley_x], [pulley_y, -4], 'k-', linewidth=1.5)
    
    # Draw mass_2 (hanging)
    mass2_width, mass2_height = 1.5, 1.5
    mass2_x, mass2_y = pulley_x - mass2_width/2, -5.5
    mass2_rect = Rectangle((mass2_x, mass2_y), mass2_width, mass2_height, 
                          facecolor='lightgreen', edgecolor='black')
    ax.add_patch(mass2_rect)
    plt.text(mass2_x + mass2_width/2, mass2_y + mass2_height/2, f"{mass_2} kg", 
             ha='center', va='center')
    
    # Draw the applied force vector
    force_length = 2
    force_x = mass1_x + mass1_width/2
    force_y = mass1_y + mass1_height/2
    force_dx = force_length * np.cos(np.radians(applied_force_angle))
    force_dy = force_length * np.sin(np.radians(applied_force_angle))
    plt.arrow(force_x, force_y, force_dx, force_dy, head_width=0.3, head_length=0.3, 
              fc='red', ec='red', linewidth=2)
    plt.text(force_x + force_dx/2 + 0.3, force_y + force_dy/2 + 0.3, 
             f"F = {applied_force} N", color='red', ha='center', va='center')
    
    # Add the calculated results as text
    result_text = (
        f"Acceleration: {results['acceleration']:.1f} m/s²\n"
        f"Tension: {results['tension']:.1f} N\n"
        f"Normal Force: {results['normal_force']:.1f} N\n"
        f"Friction Force: {results['friction_force']:.1f} N"
    )
    plt.text(0, 4, result_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title and labels
    plt.title("Pulley System Analysis")
    plt.xlabel("Position (m)")
    plt.ylabel("Position (m)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show the figure
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def solve_connected_masses_problem(
    force: float,
    force_angle: float,
    mass_ratio: Union[float, Tuple[float, float]],
    mass_value: Optional[float] = None,
    friction_coefficient: float = 0.0,
    gravity: float = 9.8,
    visualize: bool = False,
    save_path: Optional[str] = None,
    round_digits: int = 1
) -> Dict[str, float]:
    """
    Solve problems involving connected masses in a pulley system.
    
    This function handles different ways of specifying the masses:
    - Either by providing a ratio (e.g., 2:1) and a value for one of the masses
    - Or by directly providing both masses as a tuple
    
    Parameters:
    -----------
    force : float
        The magnitude of the external force in Newtons (N)
    force_angle : float
        The angle of the applied force in degrees, measured from the positive x-axis
    mass_ratio : Union[float, Tuple[float, float]]
        Either the ratio of mass_1 to mass_2 (as a float), or a tuple of (mass_1, mass_2)
    mass_value : float, optional
        If mass_ratio is a ratio, this is the value of mass_2 in kg
    friction_coefficient : float, optional
        The coefficient of kinetic friction (default: 0.0)
    gravity : float, optional
        The acceleration due to gravity in m/s² (default: 9.8)
    visualize : bool, optional
        Whether to generate a visualization (default: False)
    save_path : str, optional
        Path to save the visualization if visualize is True
    round_digits : int, optional
        Number of decimal places to round the results (default: 1)
    
    Returns:
    --------
    Dict[str, float]
        A dictionary containing the calculated results
    """
    # Determine the masses based on input
    if isinstance(mass_ratio, tuple):
        mass_1, mass_2 = mass_ratio
    else:
        if mass_value is None:
            raise ValueError("When providing a mass ratio, mass_value must be specified")
        mass_2 = mass_value
        mass_1 = mass_ratio * mass_2
    
    # Calculate the system dynamics
    results = calculate_pulley_system_acceleration(
        applied_force=force,
        applied_force_angle=force_angle,
        mass_1=mass_1,
        mass_2=mass_2,
        friction_coefficient=friction_coefficient,
        gravity=gravity
    )
    
    # Round the results
    for key in results:
        results[key] = round(results[key], round_digits)
    
    # Visualize if requested
    if visualize:
        if save_path is None:
            save_path = "./images/pulley_system.png"
        visualize_pulley_system(
            applied_force=force,
            applied_force_angle=force_angle,
            mass_1=mass_1,
            mass_2=mass_2,
            results=results,
            save_path=save_path
        )
    
    return results

def main():
    """
    Main function to demonstrate the use of the pulley system solver.
    Solves the specific problem from the prompt.
    """
    # Problem parameters
    force = 47.4  # N
    force_angle = 30  # degrees
    mass = 2.6  # kg
    
    # In this problem, the mass on the surface is 2M and the hanging mass is M
    # So the ratio is 2:1, and the value of M is 2.6 kg
    
    # Solve the problem
    results = solve_connected_masses_problem(
        force=force,
        force_angle=force_angle,
        mass_ratio=(2*mass, mass),  # (2M, M)
        friction_coefficient=0.0,  # frictionless
        visualize=True,
        save_path="./images/pulley_system_solution.png"
    )
    
    # Print the results
    print("\nPulley System Analysis Results:")
    print(f"Applied Force: {force} N at {force_angle}°")
    print(f"Mass on surface (2M): {2*mass} kg")
    print(f"Hanging mass (M): {mass} kg")
    print(f"Acceleration of the system: {results['acceleration']} m/s²")
    print(f"Tension in the rope: {results['tension']} N")
    print(f"Normal force on the surface mass: {results['normal_force']} N")
    
    # Verify the answer matches the expected result
    expected_acceleration = 2.0  # m/s²
    calculated_acceleration = results['acceleration']
    
    print(f"\nExpected acceleration: {expected_acceleration} m/s²")
    print(f"Calculated acceleration: {calculated_acceleration} m/s²")
    
    if abs(calculated_acceleration - expected_acceleration) < 0.1:
        print("✓ The calculated result matches the expected answer.")
    else:
        print("✗ The calculated result does not match the expected answer.")

if __name__ == "__main__":
    main()