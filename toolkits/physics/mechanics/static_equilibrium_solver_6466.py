# Filename: static_equilibrium_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle, Arrow

def calculate_force_from_torque(mass, gravity=9.8, pivot_point=(0, 0), force_points=None, 
                               weight_point=None, precision=3):
    """
    Calculate forces in a static equilibrium system using torque balance principles.
    
    This function solves for unknown forces in a system where torque and force equilibrium 
    must be maintained. It's applicable to various physics problems including levers, 
    see-saws, and body mechanics problems like push-ups.
    
    Parameters:
    -----------
    mass : float
        Mass of the object or body in kg
    gravity : float, optional
        Gravitational acceleration in m/s², default is 9.8
    pivot_point : tuple of (float, float), optional
        Coordinates (x, y) of the pivot point or fulcrum in meters
    force_points : list of tuples, optional
        List of (x, y) coordinates where forces are applied in meters
    weight_point : tuple of (float, float), optional
        Coordinates (x, y) of the center of gravity in meters
    precision : int, optional
        Number of significant digits in the result, default is 3
        
    Returns:
    --------
    dict
        Dictionary containing calculated forces at each force point in Newtons,
        with keys corresponding to the indices of force_points
    """
    # Calculate weight force
    weight = mass * gravity
    
    # For 1D problems (forces along a line), simplify to distance calculations
    if all(point[1] == pivot_point[1] for point in force_points + [weight_point]):
        # Convert to distances from pivot point
        force_distances = [point[0] - pivot_point[0] for point in force_points]
        weight_distance = weight_point[0] - pivot_point[0]
        
        # Set up the system of equations for static equilibrium
        # 1. Sum of forces = 0
        # 2. Sum of torques = 0
        
        # For a simple case with one unknown force (like a push-up problem)
        if len(force_points) == 1:
            # Calculate force from torque balance
            force = (weight * weight_distance) / force_distances[0]
            
            # Round to specified precision
            force = round(force, precision)
            
            return {0: force}
        
        # For more complex systems, we would need to set up and solve a system of equations
        # This would be implemented here for multiple unknown forces
    
    # For 2D problems, we would need a more complex torque calculation
    # This would involve cross products and would be implemented here
    
    # Return empty dict if no solution found
    return {}

def visualize_static_equilibrium(mass, pivot_point, force_points, weight_point, forces=None, 
                                title="Static Equilibrium System", save_path=None):
    """
    Visualize a static equilibrium system with forces, pivot points, and center of gravity.
    
    Parameters:
    -----------
    mass : float
        Mass of the object or body in kg
    pivot_point : tuple of (float, float)
        Coordinates (x, y) of the pivot point or fulcrum
    force_points : list of tuples
        List of (x, y) coordinates where forces are applied
    weight_point : tuple of (float, float)
        Coordinates (x, y) of the center of gravity
    forces : dict, optional
        Dictionary of calculated forces at each force point
    title : str, optional
        Title for the visualization
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    None
    """
    # Set up fonts for proper display of text
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine plot limits based on points
    all_points = [pivot_point, weight_point] + force_points
    x_values = [p[0] for p in all_points]
    y_values = [p[1] for p in all_points]
    
    x_min, x_max = min(x_values) - 0.5, max(x_values) + 0.5
    y_min, y_max = min(y_values) - 0.5, max(y_values) + 0.5
    
    # Plot the system components
    # Pivot point
    ax.plot(pivot_point[0], pivot_point[1], 'ko', markersize=10, label='Pivot Point')
    
    # Center of gravity
    ax.plot(weight_point[0], weight_point[1], 'ro', markersize=8, label='Center of Gravity')
    weight = mass * 9.8
    ax.arrow(weight_point[0], weight_point[1], 0, -0.2, head_width=0.05, 
             head_length=0.1, fc='r', ec='r', label=f'Weight: {weight:.1f} N')
    ax.text(weight_point[0] + 0.1, weight_point[1] - 0.15, f'W = {weight:.1f} N', fontsize=10)
    
    # Force points and forces
    for i, point in enumerate(force_points):
        ax.plot(point[0], point[1], 'bo', markersize=8)
        if forces and i in forces:
            force = forces[i]
            ax.arrow(point[0], point[1], 0, 0.2, head_width=0.05, 
                     head_length=0.1, fc='b', ec='b')
            ax.text(point[0] + 0.1, point[1] + 0.1, f'F = {force:.1f} N', fontsize=10)
        ax.text(point[0] - 0.1, point[1] - 0.3, f'Point {i+1}', fontsize=10)
    
    # Draw the body or beam
    if all(point[1] == pivot_point[1] for point in force_points + [weight_point]):
        # For 1D problems, draw a horizontal line
        ax.plot([min(x_values), max(x_values)], [pivot_point[1], pivot_point[1]], 'k-', linewidth=2)
    else:
        # For 2D problems, connect the points
        # This is a simplified representation
        points = [pivot_point] + force_points + [weight_point]
        for i in range(len(points)-1):
            ax.plot([points[i][0], points[i+1][0]], 
                    [points[i][1], points[i+1][1]], 'k-', linewidth=1)
    
    # Add distances
    for i, point in enumerate(force_points + [weight_point]):
        if point[0] != pivot_point[0]:
            y_offset = -0.4 if i % 2 == 0 else -0.5
            distance = abs(point[0] - pivot_point[0])
            midpoint = (pivot_point[0] + point[0]) / 2
            ax.annotate(f'{distance:.2f} m', 
                        xy=(midpoint, pivot_point[1] + y_offset),
                        xytext=(midpoint, pivot_point[1] + y_offset),
                        arrowprops=dict(arrowstyle='<->', color='gray'),
                        ha='center', va='center', fontsize=9)
    
    # Set plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def visualize_pushup(mass, cg_distance, hand_distance, force=None, save_path=None):
    """
    Visualize a push-up scenario with forces and distances.
    
    Parameters:
    -----------
    mass : float
        Mass of the person in kg
    cg_distance : float
        Distance from feet to center of gravity in meters
    hand_distance : float
        Distance from feet to hands in meters
    force : float, optional
        Calculated force exerted by hands in Newtons
    save_path : str, optional
        Path to save the visualization image
        
    Returns:
    --------
    None
    """
    # Set up fonts for proper display of text
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Define positions
    feet_pos = (0, 0)
    cg_pos = (cg_distance, 0.3)
    hand_pos = (hand_distance, 0)
    
    # Draw the ground
    ax.add_patch(Rectangle((-0.5, -0.1), hand_distance + 1, 0.1, color='lightgray'))
    
    # Draw a simplified body shape
    body_x = [feet_pos[0], feet_pos[0], cg_pos[0], hand_pos[0], hand_pos[0]]
    body_y = [feet_pos[1], feet_pos[1] + 0.1, cg_pos[1], hand_pos[1] + 0.1, hand_pos[1]]
    ax.plot(body_x, body_y, 'k-', linewidth=2)
    
    # Draw center of gravity
    ax.plot(cg_pos[0], cg_pos[1], 'ro', markersize=8)
    ax.text(cg_pos[0] - 0.2, cg_pos[1] + 0.1, 'CG', fontsize=10)
    
    # Draw weight force
    weight = mass * 9.8
    ax.arrow(cg_pos[0], cg_pos[1], 0, -0.2, head_width=0.05, 
             head_length=0.1, fc='r', ec='r')
    ax.text(cg_pos[0] + 0.1, cg_pos[1] - 0.15, f'W = {weight:.1f} N', fontsize=10)
    
    # Draw reaction forces
    if force:
        # Hand force
        ax.arrow(hand_pos[0], hand_pos[1], 0, 0.2, head_width=0.05, 
                 head_length=0.1, fc='b', ec='b')
        ax.text(hand_pos[0] + 0.1, hand_pos[1] + 0.1, f'F = {force:.1f} N', fontsize=10)
        
        # Feet force (calculated from force balance)
        feet_force = weight - force
        ax.arrow(feet_pos[0], feet_pos[1], 0, 0.2, head_width=0.05, 
                 head_length=0.1, fc='g', ec='g')
        ax.text(feet_pos[0] - 0.4, feet_pos[1] + 0.1, f'R = {feet_force:.1f} N', fontsize=10)
    
    # Add distance annotations
    ax.annotate(f'{cg_distance:.2f} m', 
                xy=(cg_distance/2, -0.2),
                xytext=(cg_distance/2, -0.2),
                arrowprops=dict(arrowstyle='<->', color='gray'),
                ha='center', va='center', fontsize=9)
    
    ax.annotate(f'{hand_distance:.2f} m', 
                xy=(hand_distance/2, -0.3),
                xytext=(hand_distance/2, -0.3),
                arrowprops=dict(arrowstyle='<->', color='gray'),
                ha='center', va='center', fontsize=9)
    
    # Add mass information
    ax.text(cg_pos[0], cg_pos[1] + 0.3, f'm = {mass} kg', fontsize=12, ha='center')
    
    # Set plot properties
    ax.set_xlim(-0.5, hand_distance + 0.5)
    ax.set_ylim(-0.5, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Push-up Force Analysis')
    ax.grid(True)
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate the use of the static equilibrium solver
    for the push-up problem.
    """
    # Problem parameters for the push-up example
    mass = 53.0  # kg
    cg_distance = 0.89  # m (distance from feet to center of gravity)
    hand_distance = 1.48  # m (distance from feet to hands)
    
    # Calculate the force exerted by the hands
    pivot_point = (0, 0)  # feet as pivot
    force_points = [(hand_distance, 0)]  # hands position
    weight_point = (cg_distance, 0)  # center of gravity position
    
    # Calculate the force
    forces = calculate_force_from_torque(
        mass=mass,
        pivot_point=pivot_point,
        force_points=force_points,
        weight_point=weight_point
    )
    
    # Extract the force value
    hand_force = forces[0]
    
    # Print the result
    print(f"Mass: {mass} kg")
    print(f"Distance from feet to center of gravity: {cg_distance} m")
    print(f"Distance from feet to hands: {hand_distance} m")
    print(f"Force exerted by hands: {hand_force} N")
    
    # Verify with torque calculation
    weight = mass * 9.8
    torque_due_to_weight = weight * cg_distance
    torque_due_to_hands = hand_force * hand_distance
    print(f"\nVerification:")
    print(f"Torque due to weight: {torque_due_to_weight:.1f} N·m")
    print(f"Torque due to hands: {torque_due_to_hands:.1f} N·m")
    print(f"Difference: {abs(torque_due_to_weight - torque_due_to_hands):.1f} N·m")
    
    # Visualize the system
    visualize_pushup(
        mass=mass,
        cg_distance=cg_distance,
        hand_distance=hand_distance,
        force=hand_force,
        save_path="./images/pushup_analysis.png"
    )
    
    # Also demonstrate the general visualization function
    visualize_static_equilibrium(
        mass=mass,
        pivot_point=pivot_point,
        force_points=force_points,
        weight_point=weight_point,
        forces=forces,
        title="Push-up Force Analysis",
        save_path="./images/static_equilibrium.png"
    )

if __name__ == "__main__":
    main()