# Filename: pulley_system_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_tension(mass, acceleration, gravity=9.8, angle=0, friction_coef=0):
    """
    Calculate the tension in a string/rope in a pulley system.
    
    This function calculates the tension force in a string or rope that is 
    connected to a mass in various pulley configurations, considering the 
    effects of gravity, acceleration, and optional friction.
    
    Parameters:
    -----------
    mass : float
        The mass of the object in kilograms (kg)
    acceleration : float
        The acceleration of the object in meters per second squared (m/s²)
        Positive values indicate acceleration in the direction of motion
    gravity : float, optional
        The gravitational acceleration in m/s² (default is 9.8)
    angle : float, optional
        The angle in degrees between the horizontal and the direction of the tension
        (0 degrees is horizontal, 90 degrees is vertical upward)
    friction_coef : float, optional
        The coefficient of friction between the mass and its surface (default is 0)
    
    Returns:
    --------
    float
        The tension force in Newtons (N)
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)
    
    # Calculate the components of gravity
    gravity_parallel = mass * gravity * np.sin(angle_rad)
    gravity_normal = mass * gravity * np.cos(angle_rad)
    
    # Calculate friction force
    friction_force = friction_coef * gravity_normal
    
    # Calculate tension based on Newton's second law
    # For horizontal motion (angle = 0): T = m*a + friction
    # For vertical motion (angle = 90): T = m*a + m*g
    # For angled motion: T = m*a + m*g*sin(angle) + friction
    tension = mass * acceleration + gravity_parallel + friction_force
    
    return tension

def calculate_acceleration_in_pulley_system(m1, m2, gravity=9.8, friction_coef=0, pulley_efficiency=1.0):
    """
    Calculate the acceleration for a block-on-table (m1) connected to a hanging mass (m2).
    String is massless/inextensible, pulley ideal (efficiency scales result). Table friction: μ*m1*g.
    
    Parameters:
    -----------
    m1 : float
        The mass of the first object in kilograms (kg)
    m2 : float
        The mass of the second object in kilograms (kg)
    gravity : float, optional
        The gravitational acceleration in m/s² (default is 9.8)
    friction_coef : float, optional
        The coefficient of friction between the masses and their surfaces (default is 0)
    pulley_efficiency : float, optional
        The efficiency of the pulley system (1.0 means ideal pulley with no friction)
        Values between 0 and 1, where 1 represents a perfect pulley
    
    Returns:
    --------
    tuple of float
        (a1, a2) - The accelerations of the first and second masses in m/s²
        Note: The signs indicate the direction of acceleration
    """
    # Net driving force (downward on m2) minus kinetic friction on m1
    driving = m2 * gravity
    opposing = friction_coef * m1 * gravity
    net = driving - opposing
    if net <= 0:
        return 0.0, 0.0
    a = pulley_efficiency * net / (m1 + m2)
    return a, a

def calculate_mass_from_acceleration(known_mass, known_acceleration, gravity=9.8, pulley_efficiency=1.0, friction_coef=0.0):
    """
    Calculate the unknown mass in a pulley system given the known mass and its acceleration.
    
    This function determines the mass of one object in a pulley system when the other
    mass and its acceleration are known, using the principles of an Atwood machine.
    
    Parameters:
    -----------
    known_mass : float
        The mass of the known object in kilograms (kg)
    known_acceleration : float
        The acceleration of the known mass in meters per second squared (m/s²)
        The sign indicates direction (positive or negative)
    gravity : float, optional
        The gravitational acceleration in m/s² (default is 9.8)
    pulley_efficiency : float, optional
        The efficiency of the pulley system (1.0 means ideal pulley with no friction)
        Values between 0 and 1, where 1 represents a perfect pulley
    
    Returns:
    --------
    float
        The calculated unknown mass in kilograms (kg)
    """
    # Standard reasoning for block-on-table + hanging mass (no pulley inertia):
    # T = m1*a ;  m2*g - T = m2*a  =>  m2*(g - a) = m1*a
    # With efficiency and friction μ on table: m2*(g_eff - a) = m1*(a + μ*g)
    # => m2 = m1*(a + μ*g) / (g_eff - a)
    g_eff = gravity * pulley_efficiency
    a = float(known_acceleration)
    if a >= g_eff:
        raise ValueError("Acceleration cannot be >= effective gravity")
    m1 = float(known_mass)
    m2 = m1 * (a + friction_coef * gravity) / (g_eff - a)
    return m2

def visualize_pulley_system(m1, m2, a1, a2=None, save_path=None):
    """
    Visualize a pulley system with two masses and their accelerations.
    
    Parameters:
    -----------
    m1 : float
        The mass of the first object in kilograms (kg)
    m2 : float
        The mass of the second object in kilograms (kg)
    a1 : float
        The acceleration of the first mass in m/s²
    a2 : float, optional
        The acceleration of the second mass in m/s²
        If None, it's assumed to be equal in magnitude but opposite in direction to a1
    save_path : str, optional
        Path to save the visualization image. If None, the image is displayed but not saved.
    
    Returns:
    --------
    None
    """
    # Set up fonts for plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If a2 is not provided, calculate it
    if a2 is None:
        a2 = -a1
    
    # Draw pulley
    circle = plt.Circle((5, 8), 0.5, fill=False, color='black')
    ax.add_patch(circle)
    
    # Draw string
    plt.plot([3, 5, 7], [6, 8, 6], 'k-', linewidth=1)
    
    # Draw masses (size proportional to mass)
    m1_size = min(1 + m1/5, 3)  # Limit size for very large masses
    m2_size = min(1 + m2/5, 3)
    
    # Draw m1 (left mass)
    rect1 = plt.Rectangle((2, 5), m1_size, m1_size, color='skyblue')
    ax.add_patch(rect1)
    plt.text(2 + m1_size/2, 5 + m1_size/2, f'm₁={m1} kg', ha='center', va='center')
    
    # Draw m2 (right mass)
    rect2 = plt.Rectangle((7, 5 - m2_size), m2_size, m2_size, color='lightcoral')
    ax.add_patch(rect2)
    plt.text(7 + m2_size/2, 5 - m2_size/2, f'm₂={m2} kg', ha='center', va='center')
    
    # Draw acceleration arrows
    if a1 != 0:
        a1_dir = 1 if a1 > 0 else -1
        plt.arrow(2 + m1_size/2, 4, a1_dir * min(abs(a1), 1), 0, 
                 head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        plt.text(2 + m1_size/2 + a1_dir * min(abs(a1), 1) * 1.2, 4, 
                f'a₁={a1} m/s²', ha='center', color='blue')
    
    if a2 != 0:
        a2_dir = 1 if a2 > 0 else -1
        plt.arrow(7 + m2_size/2, 3, 0, a2_dir * min(abs(a2), 1), 
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
        plt.text(7 + m2_size/2 + 1, 3 + a2_dir * min(abs(a2), 1) * 0.5, 
                f'a₂={a2} m/s²', va='center', color='red')
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Add title
    plt.title('Pulley System Visualization')
    
    # Save or display the figure
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to demonstrate the use of the pulley system solver.
    
    This function shows how to use the various functions in this module
    to solve a specific pulley system problem.
    """
    # Problem parameters
    m1 = 5.0  # kg
    a1 = 0.25  # m/s²
    g = 9.8  # m/s²
    
    # Step 1: Calculate the tension in the string
    tension = calculate_tension(m1, a1)
    print(f"Tension in the string: {tension:.2f} N")
    
    # Step 2: Calculate the unknown mass m2
    m2 = calculate_mass_from_acceleration(m1, a1)
    print(f"Mass m2: {m2:.3f} kg")
    
    # Step 3: Verify the result by calculating the accelerations
    a1_calc, a2_calc = calculate_acceleration_in_pulley_system(m1, m2)
    print(f"Calculated acceleration of m1: {a1_calc:.3f} m/s²")
    print(f"Calculated acceleration of m2: {a2_calc:.3f} m/s²")
    
    # Step 4: Visualize the pulley system
    visualize_pulley_system(m1, m2, a1, -a1, save_path="./images/pulley_system.png")
    
    print("\nVerification:")
    print(f"Given: m1 = {m1} kg, a1 = {a1} m/s²")
    print(f"Calculated: m2 = {m2:.3f} kg")
    print(f"Expected answer: m2 = 0.131 kg")
    print(f"Difference: {abs(m2 - 0.131):.6f} kg")

if __name__ == "__main__":
    main()