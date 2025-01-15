import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

def wrench_analyze(urdf_path, grasp_point):
    # Connect to PyBullet
    p.connect(p.DIRECT)  # Or p.GUI for graphical interface

    # Load an object and get its ID
    objectId = p.loadURDF(urdf_path)

    # Get the mass and other dynamics info of the object
    mass, _, _, _, _, _, _, _ = p.getDynamicsInfo(objectId, -1)

    gravity = -9.81  # m/s^2, the direction is usually along the negative z-axis
    gravitational_force = mass * gravity  # F = m * g

    # Get the base position and orientation of the object in world coordinates
    position, orientation = p.getBasePositionAndOrientation(objectId)

    # Convert to numpy arrays for vector operations
    position = np.array(position)
    grasp_point = np.array(grasp_point)
    force_vector = np.array([0, 0, gravitational_force])  # Force vector in world coordinates

    # Position vector from the specific point to the center of mass
    r = position - grasp_point

    # Calculate the torque: τ = r × F
    torque = np.cross(r, force_vector)

def calculate_wrench(center_mass, frame_b_pos, frame_b_ori, mass):
    # Constants
    g = 9.81  # Acceleration due to gravity (m/s^2)

    orn_b = R.from_quat(frame_b_ori)  # Quaternion for the new frame

    # Force due to gravity in world frame
    F_g_world = np.array([0, 0, -mass * g])

    # Convert force into the new frame's coordinates
    force_b = orn_b.apply(F_g_world)

    # Position vector from new frame origin to center of mass in world frame
    r_world = center_mass - frame_b_pos

    # Convert position vector into the new frame's coordinates
    r_new_frame = orn_b.apply(r_world)

    # Calculate torque in new frame
    torque_b = np.cross(r_new_frame, force_b)
    
    return {'force': force_b, 'torque': torque_b}

def main():
    # Example usage
    frame_a_pos = (0, 0, 0)  # Center of mass of the object in frame a
    frame_a_ori = (0, 0, 0, 1)  # Orientation of frame a
    frame_b_pos = (0, 0, 0)  # Position of frame b
    frame_b_ori = (0, 0, 0, 1)  # Orientation of frame b
    mass = 10  # Mass of the object at the center of frame a
    wrench_at_b = calculate_wrench(frame_a_pos, frame_a_ori, frame_b_pos, frame_b_ori, mass)
    print("Wrench at frame b due to gravity at center of mass in frame a:")
    print("Force:", wrench_at_b['force'])
    print("Torque:", wrench_at_b['torque'])

if __name__ == "__main__":
    main()