# Filename: pulley_friction_system_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

def calculate_forces_in_pulley_system(m1, m2, angle_deg, mu_s=None, mu_k=None, g=9.8, is_moving=False):
    """
    Calculate forces in a pulley system with two masses connected by a cable.
    
    This function analyzes a system where one mass is on an inclined plane and another
    is hanging freely, connected by a cable through a frictionless pulley. It calculates
    all relevant forces including tension, friction, and determines the system's state.
    
    Parameters:
    -----------
    m1 : float
        Mass of the object on the inclined plane (kg)
    m2 : float
        Mass of the hanging object (kg)
    angle_deg : float
        Angle of the inclined plane in degrees
    mu_s : float, optional
        Coefficient of static friction between m1 and the inclined plane
    mu_k : float, optional
        Coefficient of kinetic friction between m1 and the inclined plane
    g : float, optional
        Acceleration due to gravity (m/s²), default is 9.8
    is_moving : bool, optional
        Whether the system is in motion (True) or at rest (False)
        
    Returns:
    --------
    dict
        Dictionary containing calculated forces and system state:
        - 'tension': Cable tension (N)
        - 'normal_force': Normal force on m1 (N)
        - 'friction_force': Friction force on m1 (N)
        - 'friction_direction': Direction of friction force ('up' or 'down' the ramp)
        - 'max_static_friction': Maximum possible static friction (N) if applicable
        - 'net_force': Net force on the system (N)
        - 'acceleration': System acceleration (m/s²)
        - 'will_move': Boolean indicating if system would move with given parameters
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate components of gravitational force for m1
    m1_weight = m1 * g
    m1_weight_parallel = m1_weight * np.sin(angle_rad)  # Component parallel to the incline
    m1_weight_normal = m1_weight * np.cos(angle_rad)    # Component normal to the incline
    
    # Calculate weight of m2
    m2_weight = m2 * g
    
    # Calculate normal force on m1
    normal_force = m1_weight_normal
    
    # If the system is not moving (static case)
    if not is_moving:
        if mu_s is None:
            raise ValueError("Static friction coefficient (mu_s) must be provided for static analysis")
        
        # In static equilibrium, tension equals weight of m2
        tension = m2_weight
        
        # Calculate the required friction force for equilibrium
        # Friction must balance the difference between tension and parallel component of m1's weight
        required_friction = tension - m1_weight_parallel
        
        # Determine direction of static friction
        if required_friction > 0:
            friction_direction = "up"  # Friction points up the ramp
        elif required_friction < 0:
            friction_direction = "down"  # Friction points down the ramp
            required_friction = abs(required_friction)  # Make positive for calculation
        else:
            friction_direction = "none"  # No friction needed
            
        # Calculate maximum possible static friction
        max_static_friction = mu_s * normal_force
        
        # Check if static friction is sufficient to maintain equilibrium
        will_move = abs(required_friction) > max_static_friction
        
        # Actual friction force is the required amount, capped by maximum static friction
        friction_force = min(required_friction, max_static_friction) if required_friction > 0 else 0
        
        # Net force and acceleration are zero in static equilibrium (if friction is sufficient)
        net_force = 0 if not will_move else (required_friction - max_static_friction)
        acceleration = 0 if not will_move else (net_force / (m1 + m2))
        
    # If the system is moving (kinetic case)
    else:
        if mu_k is None:
            raise ValueError("Kinetic friction coefficient (mu_k) must be provided for dynamic analysis")
        
        # Calculate kinetic friction force
        friction_force = mu_k * normal_force
        
        # Determine the net force parallel to the incline
        # For m1: component of weight parallel to incline + friction force (opposing motion)
        # For m2: full weight
        
        # We need to determine the direction of motion to know friction direction
        # If m2_weight > m1_weight_parallel, system moves with m1 up the incline
        # If m2_weight < m1_weight_parallel, system moves with m1 down the incline
        
        if m2_weight > m1_weight_parallel:
            # m1 moves up the incline, friction opposes this (points down)
            friction_direction = "down"
            net_force = m2_weight - m1_weight_parallel - friction_force
        else:
            # m1 moves down the incline, friction opposes this (points up)
            friction_direction = "up"
            net_force = m1_weight_parallel - m2_weight - friction_force
            
        # Calculate acceleration and tension
        acceleration = net_force / (m1 + m2)
        
        # Tension calculation depends on direction of motion
        if friction_direction == "down":
            tension = m2_weight - m2 * acceleration
        else:
            tension = m2_weight + m2 * acceleration
            
        will_move = True  # System is already defined as moving
        max_static_friction = None  # Not applicable in kinetic case
    
    # Return all calculated values
    return {
        'tension': tension,
        'normal_force': normal_force,
        'friction_force': friction_force,
        'friction_direction': friction_direction,
        'max_static_friction': max_static_friction,
        'net_force': net_force,
        'acceleration': acceleration,
        'will_move': will_move
    }

def analyze_pulley_system_equilibrium(m, M_factor, angle_deg, mu_s, mu_k=None, g=9.8):
    """
    Analyze the equilibrium conditions of a pulley system with specified mass ratio.
    
    This function specifically analyzes a system where one mass (M) is on an inclined plane
    and is related to the hanging mass (m) by a factor (M = M_factor * m).
    
    Parameters:
    -----------
    m : float
        Mass of the hanging object (kg)
    M_factor : float
        Factor relating the mass on the incline to the hanging mass (M = M_factor * m)
    angle_deg : float
        Angle of the inclined plane in degrees
    mu_s : float
        Coefficient of static friction between M and the inclined plane
    mu_k : float, optional
        Coefficient of kinetic friction between M and the inclined plane
    g : float, optional
        Acceleration due to gravity (m/s²), default is 9.8
        
    Returns:
    --------
    dict
        Dictionary containing analysis results:
        - 'is_in_equilibrium': Whether the system can remain in static equilibrium
        - 'required_friction_coefficient': Minimum coefficient of static friction needed
        - 'friction_direction': Direction of static friction force
        - All forces from calculate_forces_in_pulley_system
    """
    # Calculate mass on incline
    M = M_factor * m
    
    # Calculate forces in static condition
    forces = calculate_forces_in_pulley_system(M, m, angle_deg, mu_s, mu_k, g, is_moving=False)
    
    # Calculate the minimum coefficient of static friction required for equilibrium
    angle_rad = np.deg2rad(angle_deg)
    
    # The required friction to balance the system
    tension = m * g
    M_parallel_weight = M * g * np.sin(angle_rad)
    required_friction = abs(tension - M_parallel_weight)
    normal_force = M * g * np.cos(angle_rad)
    
    required_friction_coefficient = required_friction / normal_force
    
    # Determine if the system is in equilibrium
    is_in_equilibrium = mu_s >= required_friction_coefficient
    
    # Add additional analysis results
    result = forces.copy()
    result.update({
        'is_in_equilibrium': is_in_equilibrium,
        'required_friction_coefficient': required_friction_coefficient
    })
    
    return result

def visualize_pulley_system(m, M, angle_deg, results, save_path=None):
    """
    Visualize the pulley system with forces.
    
    Parameters:
    -----------
    m : float
        Mass of the hanging object (kg)
    M : float
        Mass of the object on the inclined plane (kg)
    angle_deg : float
        Angle of the inclined plane in degrees
    results : dict
        Dictionary containing the calculated forces and system state
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    """
    # Set up figure
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Set up coordinate system
    max_coord = max(m, M) * 2
    plt.xlim(-max_coord, max_coord)
    plt.ylim(-max_coord/2, max_coord)
    
    # Draw inclined plane
    incline_length = max_coord * 1.5
    incline_x = np.array([0, incline_length * np.cos(np.pi/2 - angle_rad)])
    incline_y = np.array([0, incline_length * np.sin(np.pi/2 - angle_rad)])
    plt.plot(incline_x, incline_y, 'k-', linewidth=2)
    
    # Draw horizontal line
    plt.plot([-max_coord, incline_x[1] + 1], [0, 0], 'k-', linewidth=1)
    
    # Calculate positions
    # Position of mass M on the incline
    M_distance_along_incline = max_coord * 0.4
    M_x = M_distance_along_incline * np.cos(np.pi/2 - angle_rad)
    M_y = M_distance_along_incline * np.sin(np.pi/2 - angle_rad)
    
    # Size of the boxes proportional to their masses
    M_size = 0.15 * max_coord * np.sqrt(M/max(m, M))
    m_size = 0.15 * max_coord * np.sqrt(m/max(m, M))
    
    # Position of the pulley
    pulley_x = incline_x[1] * 0.8
    pulley_y = incline_y[1] * 0.8
    pulley_radius = 0.08 * max_coord
    
    # Position of mass m (hanging)
    m_x = pulley_x + pulley_radius * 1.5
    m_y = pulley_y - max_coord * 0.5
    
    # Draw pulley
    pulley = plt.Circle((pulley_x, pulley_y), pulley_radius, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(pulley)
    
    # Draw masses
    M_box = plt.Rectangle((M_x - M_size/2, M_y), M_size, M_size, angle=angle_deg, 
                         color='blue', alpha=0.7)
    m_box = plt.Rectangle((m_x - m_size/2, m_y), m_size, m_size, color='red', alpha=0.7)
    plt.gca().add_patch(M_box)
    plt.gca().add_patch(m_box)
    
    # Draw rope
    # From M to pulley
    plt.plot([M_x, pulley_x], [M_y + M_size/2, pulley_y], 'k-', linewidth=1.5)
    # From pulley to m
    plt.plot([pulley_x, m_x], [pulley_y, m_y + m_size], 'k-', linewidth=1.5)
    
    # Draw forces
    force_scale = max_coord * 0.2
    
    # Normal force on M
    normal_x = -results['normal_force'] * np.sin(angle_rad) * force_scale / (M * 9.8)
    normal_y = results['normal_force'] * np.cos(angle_rad) * force_scale / (M * 9.8)
    plt.arrow(M_x, M_y + M_size/2, normal_x, normal_y, head_width=0.05*max_coord, 
              head_length=0.07*max_coord, fc='green', ec='green', label='Normal Force')
    
    # Weight of M
    plt.arrow(M_x, M_y + M_size/2, 0, -force_scale, head_width=0.05*max_coord, 
              head_length=0.07*max_coord, fc='purple', ec='purple', label='Weight M')
    
    # Weight of m
    plt.arrow(m_x, m_y + m_size/2, 0, -force_scale * m/M, head_width=0.05*max_coord, 
              head_length=0.07*max_coord, fc='purple', ec='purple', label='Weight m')
    
    # Tension force on M (along the rope)
    tension_angle = np.arctan2(pulley_y - (M_y + M_size/2), pulley_x - M_x)
    tension_x = results['tension'] * np.cos(tension_angle) * force_scale / (M * 9.8)
    tension_y = results['tension'] * np.sin(tension_angle) * force_scale / (M * 9.8)
    plt.arrow(M_x, M_y + M_size/2, tension_x, tension_y, head_width=0.05*max_coord, 
              head_length=0.07*max_coord, fc='orange', ec='orange', label='Tension')
    
    # Friction force on M
    if results['friction_direction'] == 'up':
        friction_angle = angle_rad
        friction_x = results['friction_force'] * np.cos(friction_angle) * force_scale / (M * 9.8)
        friction_y = results['friction_force'] * np.sin(friction_angle) * force_scale / (M * 9.8)
    else:  # 'down'
        friction_angle = angle_rad + np.pi
        friction_x = results['friction_force'] * np.cos(friction_angle) * force_scale / (M * 9.8)
        friction_y = results['friction_force'] * np.sin(friction_angle) * force_scale / (M * 9.8)
    
    plt.arrow(M_x, M_y + M_size/2, friction_x, friction_y, head_width=0.05*max_coord, 
              head_length=0.07*max_coord, fc='red', ec='red', label='Friction')
    
    # Add labels and angle
    plt.text(M_x, M_y + M_size + 0.1*max_coord, f'M = {M:.1f} kg', ha='center')
    plt.text(m_x, m_y - 0.1*max_coord, f'm = {m:.1f} kg', ha='center')
    plt.text(incline_x[1]/3, incline_y[1]/6, f'θ = {angle_deg}°', ha='center')
    
    # Add force information
    info_x = -max_coord * 0.8
    info_y = max_coord * 0.8
    plt.text(info_x, info_y, 
             f"Normal Force: {results['normal_force']:.2f} N\n"
             f"Tension: {results['tension']:.2f} N\n"
             f"Friction: {results['friction_force']:.2f} N ({results['friction_direction']})\n"
             f"Equilibrium: {'Yes' if not results['will_move'] else 'No'}",
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Normal Force'),
        Line2D([0], [0], color='purple', lw=2, label='Weight'),
        Line2D([0], [0], color='orange', lw=2, label='Tension'),
        Line2D([0], [0], color='red', lw=2, label='Friction')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.title('Pulley System Force Analysis')
    plt.axis('equal')
    plt.grid(True)
    
    # Save or show the figure
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to demonstrate the use of the pulley system solver.
    Solves the specific problem from the task and visualizes the results.
    """
    # Problem parameters
    m = 1.0  # kg (arbitrary value for m)
    M_factor = 2.0  # M = 2m
    angle_deg = 25.0  # degrees (< 30 degrees as specified)
    mu_s = 0.5  # arbitrary value for static friction coefficient
    mu_k = 0.3  # arbitrary value for kinetic friction coefficient
    
    # Calculate M
    M = M_factor * m
    
    print("Analyzing pulley system with:")
    print(f"m = {m} kg, M = {M} kg (M = {M_factor}m)")
    print(f"Incline angle = {angle_deg} degrees")
    print(f"Static friction coefficient = {mu_s}")
    print(f"Kinetic friction coefficient = {mu_k}")
    print("\n")
    
    # Analyze the system
    results = analyze_pulley_system_equilibrium(m, M_factor, angle_deg, mu_s, mu_k)
    
    # Print results
    print("Analysis Results:")
    print(f"Normal force on M: {results['normal_force']:.2f} N")
    print(f"Tension in the rope: {results['tension']:.2f} N")
    print(f"Friction force: {results['friction_force']:.2f} N")
    print(f"Direction of friction: {results['friction_direction']} the ramp")
    print(f"Required friction coefficient for equilibrium: {results['required_friction_coefficient']:.4f}")
    print(f"Is the system in equilibrium? {'Yes' if results['is_in_equilibrium'] else 'No'}")
    
    # Visualize the system
    visualize_pulley_system(m, M, angle_deg, results, save_path="./images/pulley_system.png")
    print("\nVisualization saved to ./images/pulley_system.png")
    
    # Verify the answer to the specific problem
    print("\nAnswer to the specific problem:")
    print(f"The direction of the static friction force is: {results['friction_direction']} the ramp")

if __name__ == "__main__":
    main()