# Filename: pulley_system_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

def calculate_forces_on_inclined_block(mass, angle_deg, mu_k=0, g=9.8):
    """
    Calculate the forces acting on a block on an inclined plane.
    
    This function computes the weight components (parallel and perpendicular to the incline),
    normal force, and friction force for a block on an inclined plane.
    
    Parameters:
    -----------
    mass : float
        Mass of the block in kg
    angle_deg : float
        Angle of the incline in degrees (measured from horizontal)
    mu_k : float, optional
        Coefficient of kinetic friction between the block and the incline (dimensionless)
    g : float, optional
        Acceleration due to gravity in m/s²
    
    Returns:
    --------
    dict
        Dictionary containing the calculated forces in Newtons:
        - 'weight': Total weight force (mg)
        - 'weight_parallel': Component of weight parallel to the incline
        - 'weight_perpendicular': Component of weight perpendicular to the incline
        - 'normal': Normal force exerted by the incline on the block
        - 'friction': Friction force opposing motion along the incline
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Calculate weight and its components
    weight = mass * g
    weight_parallel = weight * np.sin(angle_rad)
    weight_perpendicular = weight * np.cos(angle_rad)
    
    # Calculate normal force (equal to perpendicular component of weight)
    normal_force = weight_perpendicular
    
    # Calculate friction force
    friction_force = mu_k * normal_force
    
    return {
        'weight': weight,
        'weight_parallel': weight_parallel,
        'weight_perpendicular': weight_perpendicular,
        'normal': normal_force,
        'friction': friction_force
    }

def solve_pulley_system_with_incline(m2, angle_deg, mu_k, constant_speed=True, g=9.8):
    """
    Solve for the unknown mass in a pulley system with one block on an inclined plane.
    
    This function calculates the mass of a hanging block (m1) connected by a rope over a 
    pulley to another block (m2) on an inclined plane, given that the system is moving 
    at constant speed.
    
    Parameters:
    -----------
    m2 : float
        Mass of the block on the inclined plane in kg
    angle_deg : float
        Angle of the incline in degrees (measured from horizontal)
    mu_k : float
        Coefficient of kinetic friction between block m2 and the incline
    constant_speed : bool, optional
        If True, assumes the system is moving at constant speed (net force = 0)
    g : float, optional
        Acceleration due to gravity in m/s²
    
    Returns:
    --------
    float
        The calculated mass m1 of the hanging block in kg
    """
    # Calculate forces on the inclined block
    forces_m2 = calculate_forces_on_inclined_block(m2, angle_deg, mu_k, g)
    
    # For block moving up the incline at constant speed:
    # Tension = weight_parallel + friction
    if constant_speed:
        tension = forces_m2['weight_parallel'] + forces_m2['friction']
    else:
        # For a general case (not used in this specific problem)
        # Would need to account for acceleration
        tension = forces_m2['weight_parallel'] + forces_m2['friction']
    
    # For the hanging block at constant speed:
    # m1 * g = tension
    m1 = tension / g
    
    return m1

def analyze_pulley_system(m1, m2, angle_deg, mu_k, g=9.8):
    """
    Analyze the dynamics of a pulley system with one block on an inclined plane.
    
    This function calculates all forces and determines whether the system will
    accelerate, remain stationary, or move at constant speed.
    
    Parameters:
    -----------
    m1 : float
        Mass of the hanging block in kg
    m2 : float
        Mass of the block on the inclined plane in kg
    angle_deg : float
        Angle of the incline in degrees (measured from horizontal)
    mu_k : float
        Coefficient of kinetic friction between block m2 and the incline
    g : float, optional
        Acceleration due to gravity in m/s²
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'forces_m1': Forces acting on m1
        - 'forces_m2': Forces acting on m2
        - 'tension': Tension in the rope
        - 'net_force': Net force on the system
        - 'acceleration': Acceleration of the system
        - 'state': Description of system's motion state
    """
    # Calculate forces on the inclined block
    forces_m2 = calculate_forces_on_inclined_block(m2, angle_deg, mu_k, g)
    
    # Calculate weight of hanging block
    weight_m1 = m1 * g
    
    # Calculate net force (positive means m1 going down, m2 going up the incline)
    net_force = weight_m1 - forces_m2['weight_parallel'] - forces_m2['friction']
    
    # Calculate acceleration
    acceleration = net_force / (m1 + m2)
    
    # Determine tension in the rope
    if acceleration > 0:  # m1 accelerating downward
        tension = m1 * (g - acceleration)
    elif acceleration < 0:  # m1 accelerating upward
        tension = m1 * (g + abs(acceleration))
    else:  # constant speed
        tension = m1 * g
    
    # Determine state of motion
    if abs(acceleration) < 1e-10:  # Effectively zero
        state = "Constant speed"
    elif acceleration > 0:
        state = "Accelerating (m1 down, m2 up)"
    else:
        state = "Accelerating (m1 up, m2 down)"
    
    return {
        'forces_m1': {'weight': weight_m1, 'tension': tension},
        'forces_m2': forces_m2,
        'tension': tension,
        'net_force': net_force,
        'acceleration': acceleration,
        'state': state
    }

def visualize_pulley_system(m1, m2, angle_deg, mu_k, show_forces=True, save_path=None):
    """
    Visualize the pulley system with forces.
    
    Parameters:
    -----------
    m1 : float
        Mass of the hanging block in kg
    m2 : float
        Mass of the block on the inclined plane in kg
    angle_deg : float
        Angle of the incline in degrees (measured from horizontal)
    mu_k : float
        Coefficient of kinetic friction
    show_forces : bool, optional
        Whether to show force vectors
    save_path : str, optional
        Path to save the figure, if None, the figure is displayed
    
    Returns:
    --------
    None
    """
    # Set up figure
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Calculate system dynamics
    analysis = analyze_pulley_system(m1, m2, angle_deg, mu_k)
    
    # Set up coordinate system
    ax = plt.gca()
    ax.set_xlim(-5, 15)
    ax.set_ylim(-2, 12)
    ax.set_aspect('equal')
    
    # Draw inclined plane
    incline_length = 12
    incline_x = [0, incline_length * np.cos(np.radians(90-angle_deg))]
    incline_y = [0, incline_length * np.sin(np.radians(90-angle_deg))]
    plt.plot(incline_x, incline_y, 'k-', lw=2)
    
    # Draw horizontal line
    plt.plot([-5, incline_x[0]], [0, 0], 'k-', lw=2)
    
    # Draw pulley
    pulley_x = incline_x[1]
    pulley_y = incline_y[1] + 1
    pulley = Circle((pulley_x, pulley_y), 0.5, fill=True, color='gray')
    ax.add_patch(pulley)
    
    # Draw blocks
    block_size = 1.0
    
    # Block on incline (m2)
    angle_rad = np.radians(angle_deg)
    block2_x = incline_x[1] - 3 * np.cos(np.radians(90-angle_deg))
    block2_y = incline_y[1] - 3 * np.sin(np.radians(90-angle_deg))
    
    # Create rotated rectangle for block 2
    block2 = plt.Rectangle((block2_x, block2_y), block_size, block_size, 
                          angle=angle_deg, color='skyblue', alpha=0.7)
    ax.add_patch(block2)
    plt.text(block2_x + 0.5, block2_y + 0.5, f"m₂", 
             ha='center', va='center', fontsize=12)
    
    # Block hanging (m1)
    block1_x = pulley_x - 3
    block1_y = 5
    block1 = Rectangle((block1_x-block_size/2, block1_y-block_size), 
                       block_size, block_size, color='lightcoral', alpha=0.7)
    ax.add_patch(block1)
    plt.text(block1_x, block1_y - 0.5, f"m₁", 
             ha='center', va='center', fontsize=12)
    
    # Draw rope
    rope_points = [
        (block1_x, block1_y),
        (block1_x, pulley_y),
        (pulley_x, pulley_y),
        (block2_x + block_size/2, block2_y + block_size/2)
    ]
    
    # Draw rope segments
    plt.plot([rope_points[0][0], rope_points[1][0]], 
             [rope_points[0][1], rope_points[1][1]], 'k-', lw=1)
    
    # Arc over pulley
    arc_theta = np.linspace(180, 270, 30)
    arc_x = pulley_x + 0.5 * np.cos(np.radians(arc_theta))
    arc_y = pulley_y + 0.5 * np.sin(np.radians(arc_theta))
    plt.plot(arc_x, arc_y, 'k-', lw=1)
    
    # Rope from pulley to block 2
    plt.plot([arc_x[-1], rope_points[3][0]], 
             [arc_y[-1], rope_points[3][1]], 'k-', lw=1)
    
    # Draw angle marker
    angle_radius = 1.0
    angle_theta = np.linspace(0, angle_deg, 30)
    angle_x = angle_radius * np.cos(np.radians(angle_theta))
    angle_y = angle_radius * np.sin(np.radians(angle_theta))
    plt.plot(angle_x, angle_y, 'k-', lw=1)
    plt.text(angle_radius/2 * np.cos(np.radians(angle_deg/2)), 
             angle_radius/2 * np.sin(np.radians(angle_deg/2)), 
             f"θ", ha='center', va='center', fontsize=12)
    
    # Show forces if requested
    if show_forces:
        # Scale factor for force vectors
        scale = 0.5
        
        # Forces on m1
        # Weight
        plt.arrow(block1_x, block1_y - 0.5, 0, -scale * analysis['forces_m1']['weight'],
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
        plt.text(block1_x + 0.3, block1_y - 1, "m₁g", color='red', fontsize=10)
        
        # Tension
        plt.arrow(block1_x, block1_y - 0.5, 0, scale * analysis['tension'],
                 head_width=0.2, head_length=0.2, fc='green', ec='green')
        plt.text(block1_x - 0.3, block1_y - 0.2, "T", color='green', fontsize=10)
        
        # Forces on m2
        # Weight components
        weight_parallel_x = -scale * analysis['forces_m2']['weight_parallel'] * np.cos(angle_rad)
        weight_parallel_y = -scale * analysis['forces_m2']['weight_parallel'] * np.sin(angle_rad)
        
        plt.arrow(block2_x + block_size/2, block2_y + block_size/2, 
                 weight_parallel_x, weight_parallel_y,
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
        plt.text(block2_x + block_size/2 + weight_parallel_x/2, 
                block2_y + block_size/2 + weight_parallel_y/2 - 0.3, 
                "m₂g sinθ", color='red', fontsize=10)
        
        # Normal force
        normal_x = scale * analysis['forces_m2']['normal'] * np.sin(angle_rad)
        normal_y = scale * analysis['forces_m2']['normal'] * np.cos(angle_rad)
        
        plt.arrow(block2_x + block_size/2, block2_y + block_size/2, 
                 normal_x, normal_y,
                 head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        plt.text(block2_x + block_size/2 + normal_x/2, 
                block2_y + block_size/2 + normal_y/2 + 0.3, 
                "N", color='blue', fontsize=10)
        
        # Friction
        friction_x = -scale * analysis['forces_m2']['friction'] * np.cos(angle_rad)
        friction_y = -scale * analysis['forces_m2']['friction'] * np.sin(angle_rad)
        
        if analysis['state'] == "Constant speed":
            # If moving up, friction points down the incline
            plt.arrow(block2_x + block_size/2, block2_y + block_size/2, 
                     friction_x, friction_y,
                     head_width=0.2, head_length=0.2, fc='purple', ec='purple')
            plt.text(block2_x + block_size/2 + friction_x/2, 
                    block2_y + block_size/2 + friction_y/2 - 0.3, 
                    "f", color='purple', fontsize=10)
        
        # Tension on m2
        tension_x = scale * analysis['tension'] * np.cos(angle_rad)
        tension_y = scale * analysis['tension'] * np.sin(angle_rad)
        
        plt.arrow(block2_x + block_size/2, block2_y + block_size/2, 
                 tension_x, tension_y,
                 head_width=0.2, head_length=0.2, fc='green', ec='green')
        plt.text(block2_x + block_size/2 + tension_x/2, 
                block2_y + block_size/2 + tension_y/2 + 0.3, 
                "T", color='green', fontsize=10)
    
    # Add legend for system state
    plt.text(0, 11, f"System State: {analysis['state']}", fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add parameters text
    params_text = (
        f"Parameters:\n"
        f"m₁ = {m1:.2f} kg\n"
        f"m₂ = {m2:.2f} kg\n"
        f"θ = {angle_deg}°\n"
        f"μₖ = {mu_k}\n"
        f"Tension = {analysis['tension']:.2f} N"
    )
    plt.text(-4.5, 8, params_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title("Pulley System with Block on Inclined Plane")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('off')
    
    # Save or show the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function: Solve the pulley system problem and visualize the results.
    """
    # Problem parameters
    m2 = 11.3  # kg
    mu_k = 0.200  # coefficient of kinetic friction
    angle_deg = 27.5  # degrees
    
    # Calculate m1 for constant speed
    m1 = solve_pulley_system_with_incline(m2, angle_deg, mu_k)
    print(f"Mass of block 1 (m1): {m1:.2f} kg")
    
    # Analyze the system
    analysis = analyze_pulley_system(m1, m2, angle_deg, mu_k)
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print(f"Forces on block 1 (m1):")
    print(f"  Weight: {analysis['forces_m1']['weight']:.2f} N")
    print(f"  Tension: {analysis['forces_m1']['tension']:.2f} N")
    
    print(f"\nForces on block 2 (m2):")
    print(f"  Weight: {analysis['forces_m2']['weight']:.2f} N")
    print(f"  Weight parallel to incline: {analysis['forces_m2']['weight_parallel']:.2f} N")
    print(f"  Weight perpendicular to incline: {analysis['forces_m2']['weight_perpendicular']:.2f} N")
    print(f"  Normal force: {analysis['forces_m2']['normal']:.2f} N")
    print(f"  Friction force: {analysis['forces_m2']['friction']:.2f} N")
    
    print(f"\nSystem dynamics:")
    print(f"  Tension in rope: {analysis['tension']:.2f} N")
    print(f"  Net force: {analysis['net_force']:.2f} N")
    print(f"  Acceleration: {analysis['acceleration']:.2f} m/s²")
    print(f"  State: {analysis['state']}")
    
    # Visualize the system
    visualize_pulley_system(m1, m2, angle_deg, mu_k, save_path="./images/pulley_system.png")
    
    # Verify our answer matches the expected result
    expected_m1 = 7.22  # kg
    print(f"\nVerification:")
    print(f"Calculated m1: {m1:.2f} kg")
    print(f"Expected m1: {expected_m1} kg")
    print(f"Difference: {abs(m1 - expected_m1):.4f} kg")
    
    # Demonstrate how the tool can be used for different parameters
    print("\nDemonstrating tool with different parameters:")
    test_m2 = 10.0  # kg
    test_angle = 30.0  # degrees
    test_mu = 0.15  # coefficient of friction
    
    test_m1 = solve_pulley_system_with_incline(test_m2, test_angle, test_mu)
    print(f"For m2 = {test_m2} kg, angle = {test_angle}°, μₖ = {test_mu}:")
    print(f"Required m1 for constant speed: {test_m1:.2f} kg")

if __name__ == "__main__":
    main()