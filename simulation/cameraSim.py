

import math
import random

import cv2
from bullet.random_pointcloud import select_random_obj
from bullet.random_pointcloud import calculate_dimensions
from bullet.random_pointcloud import generate_urdf
from bullet.random_pointcloud import isObjectStable
from bullet.random_pointcloud import write_point_cloud_to_obj
from bullet.point_projection import project
from bullet.point_projection import read_obj
from utils.distance_function import distance_function
from utils.find_nearby_vertices import find_nearby_vertices
from utils.best_fitting_plane import best_fitting_plane
import numpy as np
import alphashape
from shapely.geometry import Point
from bullet.wrench_test import calculate_wrench
from scipy.spatial.transform import Rotation as R

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from utils.translate_rotate import rotate_translate_vertices

from pyquaternion import Quaternion

import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import os
import random
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
import time
from scipy.interpolate import CubicSpline

from kinovaGripper import Kinova

from tensorflow.keras.models import load_model

# Load the trained model (if saved previously)
model = load_model('model/improved_grasp_quality_cnn_model.h5')


# Adjusted FOV and translation values
visualize = True
fov = 0.5  # Field of view in degrees
translation_z = -22  # Move the camera further away

iteration_over_object = 1 # was 100
iteration_over_grasp = 1 # was 50

fingerTipSize = 1.3
perimeterSpringRes = 3
perimeterSpringWeight = 100
flexionSpringRes = 6
flexionSpringWeight = 100
# connectionSpringRes = 6
# connectionSpringWeight = 100

V = 15 # suction pump force
friction_coefficient_low = 0.1
friction_coefficient_high = 0.7

# v = True

cup_size = 0.005
object_directories = "objects/database/OMG_objects/024_bowl/"
object_directories = "/home/elyar/thesis/DualGripNet/dataset/objects/database/OMG_objects/007_tuna_fish_can"

def interpolate_positions(start_pos, end_pos, num_steps):
    t = np.linspace(0, 1, num_steps)
    cs = CubicSpline([0, 1], np.vstack([start_pos, end_pos]), axis=0)
    t = np.linspace(0, 1, num_steps)
    cs = CubicSpline([0, 1], np.vstack([start_pos, end_pos]), axis=0)
    # interpolated_positions = cs(t)
    
    # # Plot the interpolated positions
    # fig, ax = plt.subplots()
    # ax.plot(t, interpolated_positions[:, 0], label='X')
    # ax.plot(t, interpolated_positions[:, 1], label='Y')
    # ax.plot(t, interpolated_positions[:, 2], label='Z')
    # ax.set_xlabel('Interpolation parameter t')
    # ax.set_ylabel('Position')
    # ax.set_title('Interpolated Positions using Cubic Splines')
    # ax.legend()
    # plt.grid(True)
    # plt.show()
    return cs(t)

def slerp(q0, q1, t):
    q0 = np.array(q0)
    q1 = np.array(q1)
    dot_product = np.dot(q0, q1)
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product
    if dot_product > 0.9995:
        return q0 + t * (q1 - q0)

    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.cos(theta_t) - dot_product * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return (s0 * q0) + (s1 * q1)

def interpolate_orientations(start_orn, end_orn, num_steps):
    t = np.linspace(0, 1, num_steps)
    return [slerp(start_orn, end_orn, ti) for ti in t]
def camera_to_world(x_cam, y_cam, z_cam):
    # Position of the camera in the world coordinates
    camera_position_world = np.array([0, 0, 0.6])
    
    # Convert camera coordinates to world coordinates
    x_world = x_cam
    y_world = y_cam
    z_world = camera_position_world[2] - z_cam
    z_world =  z_cam - camera_position_world[2] 
    
    return x_world, y_world, z_world

def get_current_pose(robotId, end_effector_index):
    state = p.getLinkState(robotId, end_effector_index, computeLinkVelocity=True)
    position = state[0]
    orientation = state[5]
    return position, orientation

def rotate_depth_data(depth_data, u_grasp, v_grasp, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)
    
    # Get the shape of the depth data
    h, w = depth_data.shape
    
    # Create a grid of x and y coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # Center the coordinates around the middle of the image
    cx, cy = u_grasp, v_grasp
    xx = xx - cx
    yy = yy - cy
    
    # Define the rotation matrix for the z-axis
    Rz = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    
    # Stack the x and y coordinates
    points = np.vstack((xx.flatten(), yy.flatten()))
    
    # Apply the rotation matrix to each point
    rotated_points = np.dot(Rz, points)
    
    # Get the rotated x and y coordinates
    rotated_xx = rotated_points[0].reshape(h, w) + cx
    rotated_yy = rotated_points[1].reshape(h, w) + cy
    
    # Interpolate the depth values at the rotated coordinates
    rotated_depth_data = map_coordinates(depth_data, [rotated_yy, rotated_xx], order=1, mode='nearest')
    
    return rotated_depth_data

def vector_angles(v):
    A, B, C = v

    # Calculate the magnitude of the vector
    magnitude = math.sqrt(A**2 + B**2 + C**2)
    
    # Calculate the magnitude of the projection on the x-y plane
    magnitude_xy = math.sqrt(A**2 + B**2)

    # Angle with x-axis in the x-y plane (cos(theta_xy) = A / |v_xy|)
    if magnitude_xy == 0:
        angle_y = None  # Undefined for zero vector in x-y plane
    else:
        angle_radians = math.atan2(A, B)
    
        # Convert the angle to degrees
        angle_y = math.degrees(angle_radians)

    # Angle with z-axis (cos(theta_z) = C / |v|)
    if magnitude == 0:
        angle_z = None  # Undefined for zero vector
    else:
        cos_theta_z = C / magnitude
        angle_z = math.degrees(math.acos(cos_theta_z))

    return angle_y, angle_z

def find_quaternion_from_z_axis(z_axis):
    # Normalize the given Z-axis vector
    z_b = np.array(z_axis) / np.linalg.norm(z_axis)
    
    # Assume an arbitrary X-axis that is not aligned with Z-axis
    x_axis = np.array([1, 0, 0]) if z_b[0] == 0 and z_b[1] == 0 else np.array([0, 0, 1])
    
    # Calculate the Y-axis using the cross product of Z-axis and X-axis
    y_b = np.cross(z_b, x_axis)
    y_b /= np.linalg.norm(y_b)  # Normalize the Y-axis
    
    # Re-calculate the X-axis using the cross product of Y-axis and Z-axis
    x_b = np.cross(y_b, z_b)
    x_b /= np.linalg.norm(x_b)  # Normalize the X-axis
    
    # Construct the rotation matrix from the world frame to the new frame
    rotation_matrix = np.vstack([x_b, y_b, z_b]).T
    
    # Create a rotation object from the rotation matrix
    rotation = R.from_matrix(rotation_matrix)
    
    # Convert the rotation object to a quaternion
    quaternion = rotation.as_quat()  # This will give the quaternion (x, y, z, w)
    return quaternion

def filter_points_in_cylinder(points, center, direction, radius):
    """
    Filters points that are inside a cylinder defined by a center, direction, and radius.

    :param points: np.array of shape (N, 3) containing 3D coordinates of points.
    :param center: np.array of shape (3,) representing the center of the cylinder's base.
    :param direction: np.array of shape (3,) representing the axis direction of the cylinder.
    :param radius: float representing the radius of the cylinder.
    :return: np.array of shape (M, 3) containing points within the cylinder.
    """
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Vector from the center to the points
    vec_to_points = points - center

    # Project vec_to_points onto the direction vector
    proj_lengths = np.dot(vec_to_points, direction)
    proj_vectors = np.outer(proj_lengths, direction)

    # Calculate the perpendicular distances to the cylinder axis
    perp_vectors = vec_to_points - proj_vectors
    perp_distances = np.linalg.norm(perp_vectors, axis=1)

    # Filter points based on perpendicular distance
    inside_cylinder_mask = perp_distances <= radius
    filtered_points = points[inside_cylinder_mask]

    return filtered_points

def handle_key_events():
    global rotate_left_right, rotate_up_down
    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        rotate_left_right += 1
    if keys[K_DOWN]:
        rotate_left_right -= 1
    if keys[K_LEFT]:
        rotate_up_down += 1
    if keys[K_RIGHT]:
        rotate_up_down -= 1

# Initialize rotation variables
rotate_left_right = 0
rotate_up_down = 0

obj_iter = 0
# while obj_iter < iteration_over_object:
def generate(file_pos, file_orn, file_path, depth_image):
    # file_pos, file_orn, file_path, depth_image = gen_ran_obj(object_directories)


    vertices = read_obj("point_cloud.obj")
    obj_vertices = read_obj(file_path)

    q = Quaternion([file_orn[3], file_orn[0],file_orn[1],file_orn[2]])
    obj_vertices = np.array([q.rotate(vec) for vec in obj_vertices])

    selected_vertex_index = random.sample(range(len(vertices)), 1)[0]

    # Given values
    fov_retrieve = 60  # Example FOV
    width, height = 640, 480  # Example dimensions
    x_grasp, y_grasp, z_grasp = vertices[selected_vertex_index]  # Assuming x, y, z are numpy arrays or values you have

    # Calculate u and v
    u_grasp = (x_grasp * (width / 2) / (z_grasp * np.tan(np.deg2rad(fov_retrieve / 2)))) + (width / 2)
    v_grasp = (y_grasp * (height / 2) / (z_grasp * np.tan(np.deg2rad(fov_retrieve / 2)))) + (height / 2)
    
    


    nearby_vertices = find_nearby_vertices(vertices, selected_vertex_index, cup_size)
    plane_params = best_fitting_plane(selected_vertex_index, nearby_vertices, vertices)
    A, B, C, D = plane_params
    graspDirection = -np.array([A, B, C])
    print(f"u_grasp is {u_grasp}, and v_grasp is {v_grasp}")
    return(x_grasp, y_grasp, z_grasp, graspDirection)
    







obj_path = select_random_obj(object_directories)
length, width, height = calculate_dimensions(obj_path)
urdf_path = generate_urdf(obj_path, length, width, height)

# pos, orn, depth = load_urdf_fall_simulate(urdf_path)

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load URDF files
p.resetDebugVisualizerCamera(1.3, 180.0, -41.0, [-0.35, -0.58, -0.88])

p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0.0])


# Instantiate the Kinova class
# Assuming init_joints is an optional parameter. If you want to initialize with specific joint positions, provide them.
init_joints = np.array([0, 0, 0, 0, 0, 0, 0, 'open'], dtype=object) 
base_shift = [-0.3, 0.0, 0.2]  # Example base shift

# Create an instance of the Kinova robot
kinova_robot = Kinova(init_joints=init_joints, base_shift=base_shift)

num_joints = p.getNumJoints(kinova_robot.robot)
robot_id = kinova_robot.robot

max_iterations = 20
position_threshold = 0.005
orientation_threshold = 0.001
# cube_position, cube_orientation = p.getBasePositionAndOrientation(cube_id)
# grasp_orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi/2])  # Adjust as needed

# Generate random Euler angles
roll = random.uniform(0, 2 * np.pi)
pitch = random.uniform(0, 2 * np.pi)
yaw = random.uniform(0, 2 * np.pi)

# Convert Euler angles to a quaternion
random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

object_id = p.loadURDF(urdf_path, basePosition=[0,0,0.2], baseOrientation=random_orientation)
p.changeDynamics(object_id, linkIndex=-1, lateralFriction=0.2, rollingFriction=0.005)




# Intrinsic parameters
cx = 255.5
cy = 191.75
fx = 552.5
fy = 552.5
width = 516
height = 386

# Camera position and orientation
camera_distance = 0.6  # 60 cm
target_position = [0, 0, 0]  # Point the camera is looking at
up_vector = [0, 1, 0]        # Up direction for the camera

# Position the camera at the specified distance along the z-axis
camera_position = [target_position[0], target_position[1], target_position[2] + camera_distance]

# Calculate the view matrix
view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)

# Aspect ratio
aspect = width / height

# Near and far clipping planes
near = 0.01
far = 100

# Field of view (FOV) in radians
fov = 2 * np.arctan(height / (2 * fy)) * (180 / np.pi)  # Convert to degrees

# Calculate the projection matrix
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

for i in range(10):
    p.stepSimulation()
    time.sleep(0.01)
start_time = time.time()  # Record the start time
while not isObjectStable(object_id):
    p.stepSimulation()
    time.sleep(0.01)
    if time.time() - start_time > 3:
        print("Timeout: Stopped waiting for object to stabilize.")
        break
width, height, rgb_img, depth_img, segImg = p.getCameraImage(
    width,
    height,
    viewMatrix=view_matrix,
    projectionMatrix=projection_matrix,
    renderer=p.ER_TINY_RENDERER  # Change to p.ER_BULLET_HARDWARE_OPENGL if needed
)

mask = (segImg == object_id)

# Apply the mask to the depth data
masked_depth = np.zeros_like(depth_img)
masked_depth[mask] = depth_img[mask]

# Calculate the 3D coordinates
u, v = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
z = masked_depth
x = (u - width / 2) / (width / 2) * z * np.tan(np.deg2rad(fov / 2))
y = (v - height / 2) / (height / 2) * z * np.tan(np.deg2rad(fov / 2))

# Reshape into Nx3 points
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

write_point_cloud_to_obj(points, 'point_cloud.obj')



pos, orn = p.getBasePositionAndOrientation(object_id)
print("Final position:", pos)
print("Final orientation (quaternion):", orn)

x_grasp, y_grasp, z_grasp, graspDirection = generate(pos, orn, obj_path, depth_img)
x_world, y_world, z_world = camera_to_world(x_grasp, y_grasp, z_grasp)
print(x_grasp, y_grasp, z_grasp, graspDirection)
# This part needs to be worked on. Grasp location is not reflected



# Normalize the grasp direction
grasp_direction_normalized = graspDirection / np.linalg.norm(graspDirection)

# Default direction (z-axis)
default_direction = np.array([0, 0, 1])

# Calculate the rotation axis and angle
rotation_axis = np.cross(default_direction, [-grasp_direction_normalized[0],grasp_direction_normalized[1],grasp_direction_normalized[2]])
rotation_angle = np.arccos(np.dot(default_direction, grasp_direction_normalized))
rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)

# Function to convert axis-angle to quaternion

half_angle = rotation_angle / 2
w = np.cos(half_angle)
x, y, z = rotation_axis_normalized * np.sin(half_angle)
graspQuaternion = np.array([x, y, z, w])


for i in np.linspace(0, 0.5, 20):
    # Add a static sphere to the simulation
    # sphere_start_pos = [x_world, y_world, z_world]  # Position in the air at a reachable height
    rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
    transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)*i
    sphere_start_pos = np.array([x_world, y_world, z_world]) + transformed_offset
    sphere_start_orientation = graspQuaternion
    sphere_radius = 0.003  # Radius of the sphere

    # Create visual shape for the sphere
    sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 1])


    sphere_id = p.createMultiBody(
        baseMass=0,  # Mass of zero to keep it static
        # baseCollisionShapeIndex=sphere_collision_shape_id,  # Uncomment if collision detection is needed
        baseVisualShapeIndex=sphere_visual_shape_id,
        basePosition=sphere_start_pos,
        baseOrientation=sphere_start_orientation
    )


for _ in range(max_iterations):

    start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)


    rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
    transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
    end_pos = np.array([x_world, y_world, z_world]) + transformed_offset
    end_orn = graspQuaternion

    num_steps = 300
    interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)
    interpolated_orientations = interpolate_orientations(start_orn, end_orn, num_steps)

    for i in range(num_steps):
        kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], interpolated_orientations[i]), dtype=object)
        target_pos = np.insert(kinova_angles[:7], 7, 'close')
        kinova_robot.setTargetPositions(target_pos)
        p.stepSimulation()
        time.sleep(1./70.)

    print("*******************************************************")
    print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
    print("desired pos: ", [x_world, y_world, z_world] + transformed_offset)
    print("*******************************************************")
    print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
    print("desired orn: ", end_orn)
    print("*******************************************************")

    for _ in range(100):
        # Step the simulation
        p.stepSimulation()
        time.sleep(1/140)

    # Get current end-effector pose
    current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

    # Calculate position and orientation errors
    position_error = np.linalg.norm(([x_world, y_world, z_world] + transformed_offset) - (current_position))
    graspQuaternion = np.array(graspQuaternion)
    current_orientation = np.array(current_orientation)
    q1 = graspQuaternion / np.linalg.norm(graspQuaternion)
    q2 = current_orientation / np.linalg.norm(current_orientation)

    dot_product = np.dot(q1, q2)

    orientation_error = 1 - np.abs(dot_product)
    print("position_error: ", position_error)
    print("orientation_error: ", orientation_error)
    # Check if the end-effector is within the desired thresholds
    if position_error < position_threshold and orientation_error < orientation_threshold:
        print(f"Reached target pose in {_} iterations.")
        break

for _ in range(100):
    # Step the simulation
    p.stepSimulation()
    time.sleep(1/140)

for _ in range(max_iterations):

    start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)


    rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
    transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
    end_pos = np.array([x_world, y_world, z_world]) + 0.52 * transformed_offset
    end_orn = graspQuaternion

    num_steps = 300
    interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

    for i in range(num_steps):
        kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], graspQuaternion), dtype=object)
        target_pos = np.insert(kinova_angles[:7], 7, 'close')
        kinova_robot.setTargetPositions(target_pos)
        p.stepSimulation()
        time.sleep(1./30.)

    print("*******************************************************")
    print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
    print("desired pos: ", [x_world, y_world, z_world] + 0.52 * transformed_offset)
    print("*******************************************************")
    print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
    print("desired orn: ", end_orn)
    print("*******************************************************")

    for _ in range(100):
        # Step the simulation
        p.stepSimulation()
        time.sleep(1/140)

    # Get current end-effector pose
    current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

    # Calculate position and orientation errors
    position_error = np.linalg.norm(([x_world, y_world, z_world] + 0.52 * transformed_offset) - (current_position))
    graspQuaternion = np.array(graspQuaternion)
    current_orientation = np.array(current_orientation)
    q1 = graspQuaternion / np.linalg.norm(graspQuaternion)
    q2 = current_orientation / np.linalg.norm(current_orientation)

    dot_product = np.dot(q1, q2)

    orientation_error = 1 - np.abs(dot_product)
    print("position_error: ", position_error)
    print("orientation_error: ", orientation_error)
    # Check if the end-effector is within the desired thresholds
    if position_error < position_threshold and orientation_error < orientation_threshold:
        print(f"Reached target pose in {_} iterations.")
        break

for _ in range(1000):
    # Step the simulation
    p.stepSimulation()
    time.sleep(1/140)

# for _ in range(max_iterations):
#     # Compute inverse kinematics
    
    
#     # kinova_angles = np.array(kinova_robot.solveInverseKinematics([cube_position[0],cube_position[1],cube_position[2]], grasp_orientation), dtype=object)
#     kinova_angles = np.array(kinova_robot.solveInverseKinematics([x_world, y_world, z_world], graspQuaternion), dtype=object)
#     target_pos = np.insert(kinova_angles[:7], 7, 'close')
#     kinova_robot.setTargetPositions(target_pos)

#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1./140.)

#     # Get current end-effector pose
#     current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

#     # Calculate position and orientation errors
#     position_error = np.linalg.norm(([x_world, y_world, z_world] + transformed_offset) - (current_position))
#     orientation_error = np.linalg.norm(np.array(graspQuaternion) - np.array(current_orientation))
#     print("position_error: ", position_error)
#     print("orientation_error: ", orientation_error)
#     # Check if the end-effector is within the desired thresholds
#     if position_error < position_threshold and orientation_error < orientation_threshold:
#         print(f"Reached target pose in {_} iterations.")
#         break

for _ in range(2000):
    # Step the simulation
    p.stepSimulation()
    time.sleep(1/140)

p.disconnect()






