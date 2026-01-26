# Filename: pulley_system_solver.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import os

def calculate_pulley_system_acceleration(m1, m2, external_force, g=9.8, angle=0, friction_coef=0):
    """
    Calculate the acceleration of a pulley system with two masses connected by an inextensible string.
    
    This function solves for the acceleration in a generalized pulley system where one mass may be 
    on a surface (possibly inclined) and another mass may be hanging. The masses are connected by
    an inextensible string passing over an ideal (massless, frictionless) pulley.
    
    Parameters:
    -----------
    m1 : float
        Mass of the first object (kg), typically the hanging mass
    m2 : float
        Mass of the second object (kg), typically the mass on the surface
    external_force : float
        External force applied to m2 (N). Positive value means force is in the direction of potential motion.
    g : float, optional
        Acceleration due to gravity (m/s²), default is 9.8
    angle : float, optional
        Angle of the inclined surface in degrees, default is 0 (horizontal)
    friction_coef : float, optional
        Coefficient of kinetic friction between m2 and the surface, default is 0 (frictionless)
    
    Returns:
    --------
    float
        The magnitude of the acceleration of the system (m/s²)
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Component of gravity parallel to the incline for m2
    g_parallel = g * np.sin(angle_rad)
    
    # Component of gravity perpendicular to the incline for m2
    g_perpendicular = g * np.cos(angle_rad)
    
    # Friction force (if any)
    friction_force = friction_coef * m2 * g_perpendicular
    
    # For a system where m1 is hanging and m2 is on a surface:
    # The net force on the system is:
    # F_net = external_force + m1*g - friction_force - m2*g_parallel
    # This must equal (m1 + m2) * a
    
    # Solving for acceleration:
    acceleration = (external_force + m1*g - friction_force - m2*g_parallel) / (m1 + m2)
    
    return acceleration

def calculate_tension(m1, m2, acceleration, g=9.8):
    """
    Calculate the tension in the string connecting the two masses in a pulley system.
    
    Parameters:
    -----------
    m1 : float
        Mass of the first object (kg), typically the hanging mass
    m2 : float
        Mass of the second object (kg), typically the mass on the surface
    acceleration : float
        Acceleration of the system (m/s²)
    g : float, optional
        Acceleration due to gravity (m/s²), default is 9.8
    
    Returns:
    --------
    float
        The tension in the string (N)
    """
    # For m1 (hanging mass): T - m1*g = -m1*a
    # Therefore: T = m1*(g - a)
    tension = m1 * (g - acceleration)
    
    return tension

def visualize_pulley_system(m1, m2, external_force, acceleration, tension, save_path=None):
    """
    Visualize the pulley system with forces and acceleration.
    
    Parameters:
    -----------
    m1 : float
        Mass of the first object (kg), typically the hanging mass
    m2 : float
        Mass of the second object (kg), typically the mass on the surface
    external_force : float
        External force applied to m2 (N)
    acceleration : float
        Calculated acceleration of the system (m/s²)
    tension : float
        Calculated tension in the string (N)
    save_path : str, optional
        Path to save the figure, if None the figure is displayed but not saved
    
    Returns:
    --------
    None
    """
    # Set up the figure
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create axis
    ax = plt.gca()
    ax.set_xlim(-5, 15)
    ax.set_ylim(-8, 5)
    ax.set_aspect('equal')
    
    # Draw the table
    table = Rectangle((-5, -3), 20, 1, color='burlywood')
    ax.add_patch(table)
    
    # Draw the pulley
    pulley = Circle((0, 0), 0.8, color='lightgray', zorder=3)
    ax.add_patch(pulley)
    
    # Draw m2 (block on surface)
    m2_block = Rectangle((5, -2), 2, 2, color='skyblue')
    ax.add_patch(m2_block)
    
    # Draw m1 (hanging mass)
    m1_block = Rectangle((-2, -7), 2, 2, color='sandybrown')
    ax.add_patch(m1_block)
    
    # Draw the string
    plt.plot([0, 5], [0, 0], 'k-', linewidth=1.5)  # Horizontal part
    plt.plot([0, 0], [0, -5], 'k-', linewidth=1.5)  # Vertical part to m1
    plt.plot([0, -1], [0, -5], 'k-', linewidth=1.5)  # Connection to m1
    
    # Draw external force arrow
    force_arrow = FancyArrowPatch((7, -1), (10, -1), 
                                 arrowstyle='->', 
                                 color='blue',
                                 mutation_scale=20,
                                 linewidth=2)
    ax.add_patch(force_arrow)
    
    # Draw acceleration arrow
    acc_arrow = FancyArrowPatch((7, -3.5), (9, -3.5), 
                               arrowstyle='->', 
                               color='red',
                               mutation_scale=15,
                               linewidth=1.5)
    ax.add_patch(acc_arrow)
    
    # Add text labels
    plt.text(6, -1, f'$F_x = {external_force}N$', fontsize=12)
    plt.text(-1, -6, f'$m_1 = {m1}kg$', fontsize=12)
    plt.text(6, -1.5, f'$m_2 = {m2}kg$', fontsize=12)
    plt.text(7, -4, f'$a = {acceleration:.1f}m/s^2$', fontsize=12, color='red')
    plt.text(2, 0.5, f'$T = {tension:.1f}N$', fontsize=12)
    
    # Remove axis ticks
    plt.axis('off')
    
    # Add title
    plt.title('Pulley System Analysis', fontsize=14)
    
    # Save or show the figure
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def analyze_pulley_system(m1, m2, external_force, g=9.8, angle=0, friction_coef=0, visualize=True):
    """
    Perform a complete analysis of a pulley system.
    
    This function calculates the acceleration and tension in a pulley system and
    optionally visualizes the results.
    
    Parameters:
    -----------
    m1 : float
        Mass of the first object (kg), typically the hanging mass
    m2 : float
        Mass of the second object (kg), typically the mass on the surface
    external_force : float
        External force applied to m2 (N)
    g : float, optional
        Acceleration due to gravity (m/s²), default is 9.8
    angle : float, optional
        Angle of the inclined surface in degrees, default is 0 (horizontal)
    friction_coef : float, optional
        Coefficient of kinetic friction between m2 and the surface, default is 0 (frictionless)
    visualize : bool, optional
        Whether to visualize the system, default is True
    
    Returns:
    --------
    dict
        A dictionary containing the calculated acceleration and tension
    """
    # Calculate acceleration
    acceleration = calculate_pulley_system_acceleration(
        m1, m2, external_force, g, angle, friction_coef
    )
    
    # Calculate tension
    tension = calculate_tension(m1, m2, acceleration, g)
    
    # Visualize if requested
    if visualize:
        save_path = "./images/pulley_system.png"
        visualize_pulley_system(m1, m2, external_force, acceleration, tension, save_path)
    
    # Return results
    return {
        "acceleration": acceleration,
        "tension": tension
    }

def main():
    """
    Main function to demonstrate the use of the pulley system solver.
    """
    # Problem parameters
    m1 = 5.0  # kg
    m2 = 5.0  # kg
    external_force = 180.0  # N
    g = 9.8  # m/s²
    
    # Analyze the system
    results = analyze_pulley_system(m1, m2, external_force, g)
    
    # Print results
    print("\nPulley System Analysis Results:")
    print("-------------------------------")
    print(f"Mass 1 (hanging): {m1} kg")
    print(f"Mass 2 (on surface): {m2} kg")
    print(f"External force: {external_force} N")
    print(f"Acceleration: {results['acceleration']:.1f} m/s²")
    print(f"Tension in string: {results['tension']:.1f} N")
    
    # Verify with the expected answer
    expected_acceleration = 13.1  # m/s²
    print(f"\nExpected acceleration: {expected_acceleration} m/s²")
    print(f"Calculated acceleration: {results['acceleration']:.1f} m/s²")
    print(f"Difference: {abs(results['acceleration'] - expected_acceleration):.2f} m/s²")

if __name__ == "__main__":
    main()