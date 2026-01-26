# Filename: conical_motion_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

def calculate_normal_force(mass, height, cone_angle, gravity=9.8):
    """
    Calculate the normal force acting on an object moving in a horizontal circle 
    on the inner surface of a frictionless cone.
    
    This function applies principles of circular motion and force balance to determine
    the normal force required to maintain an object in horizontal circular motion
    inside a conical surface.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms
    height : float
        Height of the object above the cone's tip in meters
    cone_angle : float
        Half-angle of the cone in degrees (angle between the cone's axis and its surface)
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8
        
    Returns:
    --------
    float
        Magnitude of the normal force in Newtons
    """
    # Convert angle to radians
    theta_rad = np.radians(cone_angle)
    
    # Calculate the normal force using force balance equations
    # In vertical direction: N*cos(theta) = m*g
    normal_force = mass * gravity / np.cos(theta_rad)
    
    return normal_force

def calculate_circular_path_radius(height, cone_angle):
    """
    Calculate the radius of the horizontal circular path for an object moving
    inside a cone at a given height.
    
    Parameters:
    -----------
    height : float
        Height of the object above the cone's tip in meters
    cone_angle : float
        Half-angle of the cone in degrees
        
    Returns:
    --------
    float
        Radius of the circular path in meters
    """
    # Convert angle to radians
    theta_rad = np.radians(cone_angle)
    
    # Calculate radius using trigonometry
    radius = height * np.tan(theta_rad)
    
    return radius

def calculate_velocity(mass, height, cone_angle, gravity=9.8):
    """
    Calculate the velocity of an object moving in a horizontal circle 
    on the inner surface of a frictionless cone.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms
    height : float
        Height of the object above the cone's tip in meters
    cone_angle : float
        Half-angle of the cone in degrees
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8
        
    Returns:
    --------
    float
        Velocity of the object in m/s
    """
    # Convert angle to radians
    theta_rad = np.radians(cone_angle)
    
    # Calculate the radius of the circular path
    radius = calculate_circular_path_radius(height, cone_angle)
    
    # Calculate velocity using the condition for horizontal circular motion
    # The centripetal force is provided by the horizontal component of the normal force
    # N*sin(theta) = m*v²/r
    # Combined with N*cos(theta) = m*g, we get:
    # v² = r*g*tan(theta)
    velocity = np.sqrt(radius * gravity * np.tan(theta_rad))
    
    return velocity

def analyze_conical_motion(mass, height, cone_angle, gravity=9.8):
    """
    Perform a comprehensive analysis of an object's motion in a horizontal circle
    on the inner surface of a frictionless cone.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms
    height : float
        Height of the object above the cone's tip in meters
    cone_angle : float
        Half-angle of the cone in degrees
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8
        
    Returns:
    --------
    dict
        Dictionary containing calculated values:
        - 'normal_force': Normal force in Newtons
        - 'radius': Radius of circular path in meters
        - 'velocity': Velocity of the object in m/s
        - 'centripetal_acceleration': Centripetal acceleration in m/s²
    """
    # Calculate normal force
    normal_force = calculate_normal_force(mass, height, cone_angle, gravity)
    
    # Calculate radius of circular path
    radius = calculate_circular_path_radius(height, cone_angle)
    
    # Calculate velocity
    velocity = calculate_velocity(mass, height, cone_angle, gravity)
    
    # Calculate centripetal acceleration
    centripetal_acceleration = velocity**2 / radius
    
    return {
        'normal_force': normal_force,
        'radius': radius,
        'velocity': velocity,
        'centripetal_acceleration': centripetal_acceleration
    }

def visualize_conical_motion(height, cone_angle, save_path=None):
    """
    Create a 3D visualization of the conical surface and the circular path of the object.
    
    Parameters:
    -----------
    height : float
        Height of the object above the cone's tip in meters
    cone_angle : float
        Half-angle of the cone in degrees
    save_path : str, optional
        Path to save the visualization image. If None, the image is displayed but not saved.
        
    Returns:
    --------
    None
    """
    # Convert angle to radians
    theta_rad = np.radians(cone_angle)
    
    # Calculate the radius of the circular path
    radius = calculate_circular_path_radius(height, cone_angle)
    
    # Calculate the height of the cone needed to reach the specified height
    cone_height = height / np.cos(theta_rad)
    
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up font for Chinese characters if needed
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create the cone
    # Generate points for the cone surface
    z = np.linspace(0, cone_height, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    z_grid, theta_grid = np.meshgrid(z, theta)
    
    # Calculate x and y coordinates for the cone surface
    x_cone = z_grid * np.tan(theta_rad) * np.cos(theta_grid)
    y_cone = z_grid * np.tan(theta_rad) * np.sin(theta_grid)
    z_cone = cone_height - z_grid  # Invert the cone
    
    # Plot the cone surface
    ax.plot_surface(x_cone, y_cone, z_cone, alpha=0.3, color='gray')
    
    # Create the circular path
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = radius * np.cos(theta_circle)
    y_circle = radius * np.sin(theta_circle)
    z_circle = np.ones_like(theta_circle) * (cone_height - height)
    
    # Plot the circular path
    ax.plot(x_circle, y_circle, z_circle, 'r-', linewidth=3, label='Circular Path')
    
    # Add a point representing the object
    ax.scatter([radius], [0], [cone_height - height], color='blue', s=100, label='Object')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Object Moving in a Horizontal Circle Inside an Inverted Cone')
    
    # Add a legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save the figure if a path is provided
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to demonstrate the use of the conical motion solver for the given problem.
    """
    # Problem parameters
    mass = 8.0  # kg
    height = 5.0  # m
    cone_angle = 30.0  # degrees
    gravity = 9.8  # m/s²
    
    # Calculate the normal force
    normal_force = calculate_normal_force(mass, height, cone_angle, gravity)
    
    # Perform a comprehensive analysis
    results = analyze_conical_motion(mass, height, cone_angle, gravity)
    
    # Display the results
    print("\nConical Motion Analysis Results:")
    print("=" * 40)
    print(f"Mass: {mass} kg")
    print(f"Height above cone tip: {height} m")
    print(f"Cone angle: {cone_angle}°")
    print("-" * 40)
    print(f"Normal force: {results['normal_force']:.2f} N")
    print(f"Radius of circular path: {results['radius']:.2f} m")
    print(f"Velocity: {results['velocity']:.2f} m/s")
    print(f"Centripetal acceleration: {results['centripetal_acceleration']:.2f} m/s²")
    print("=" * 40)
    
    # Create a visualization of the conical motion
    visualize_conical_motion(height, cone_angle, save_path="./images/conical_motion.png")

if __name__ == "__main__":
    main()