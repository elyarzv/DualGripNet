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

from graspPlan import run_grasping_policy

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)




CAM = False

# Adjusted FOV and translation values
visualize = True
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
object_1_directories = "dataset/objects/database/OMG_objects/009_gelatin_box"
object_2_directories = "dataset/objects/database/OMG_objects/009_gelatin_box"
object_1_directories = "dataset/objects/database/OMG_objects/007_tuna_fish_can"
# object_directories = "dataset/objects/database/OMG_objects/008_pudding_box"
# object_directories = "dataset/objects/database/OMG_objects/010_potted_meat_can"

# object_directories = "/home/elyar/thesis/DualGripNet/dataset/objects/database/dexnet_objects"


gripper_close = 0.673
gripper_open = -0.35

def interpolate_positions(start_pos, end_pos, num_steps):
    t = np.linspace(0, 1, num_steps)
    cs = CubicSpline([0, 1], np.vstack([start_pos, end_pos]), axis=0)
    t = np.linspace(0, 1, num_steps)
    cs = CubicSpline([0, 1], np.vstack([start_pos, end_pos]), axis=0)
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
    y_world = -y_cam
    z_world = camera_position_world[2] - z_cam
    
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

def close_gripper(robot_id, joint_indices, target_positions, forces):
    num_steps = 100
    for step in range(num_steps):
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[pos * step / num_steps for pos in target_positions],
            forces=forces
        )
        p.stepSimulation()
        time.sleep(1/140)

        # Check for contact with the object
        contact_points = p.getContactPoints(bodyA=robot_id, bodyB=object_id_1)
        if contact_points:
            print("Contact detected!")
            break



# Initialize rotation variables
rotate_left_right = 0
rotate_up_down = 0

obj_iter = 0
# while obj_iter < iteration_over_object:
def generate(grasp_pos):
    [u, v, z] = grasp_pos
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    center = [x, y, z]

    return(center[0], center[1], center[2])
    

obj_1_path = select_random_obj(object_1_directories)
obj_2_path = select_random_obj(object_2_directories)
obj_length, obj_width, obj_height = calculate_dimensions(obj_1_path)
urdf_path_1 = generate_urdf(obj_1_path, obj_length, obj_width, obj_height)
urdf_path_2 = generate_urdf(obj_2_path, obj_length, obj_width, obj_height)

# pos, orn, depth = load_urdf_fall_simulate(urdf_path)

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load URDF files

# Set the initial camera parameters
camera_distance = 0.5  # Distance from the camera to the target
camera_yaw = 30      # Yaw angle in degrees
camera_pitch = -20   # Pitch angle in degrees
camera_target_position = [0, 0, 0.3]  # The XYZ position the camera is looking at

# Reset the camera view at the beginning of the simulation
p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                            cameraYaw=camera_yaw,
                            cameraPitch=camera_pitch,
                            cameraTargetPosition=camera_target_position)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0.0])


# Instantiate the Kinova class
# Assuming init_joints is an optional parameter. If you want to initialize with specific joint positions, provide them.
init_joints = np.array([0, 0, 0, 0, 0, 0, 0, gripper_open], dtype=object) 
base_shift = [-0.4, 0.0, 0.2]  # Example base shift

# Create an instance of the Kinova robot
kinova_robot = Kinova(init_joints=init_joints, base_shift=base_shift)

num_joints = p.getNumJoints(kinova_robot.robot)
robot_id = kinova_robot.robot

# num_joints = p.getNumJoints(robot_id)
# links = []
# for i in range(num_joints):
#     joint_info = p.getJointInfo(robot_id, i)
#     link_index = joint_info[0]
#     link_name = joint_info[12].decode('utf-8')  # The name of the link
#     links.append((link_index, link_name))
#     print(f"Link Index: {link_index}, Link Name: {link_name}")

# breakpoint()

# Define the link indices for the gripper fingers
left_finger_index = 11  # Replace with the actual link index for the left finger
right_finger_index = 13  # Replace with the actual link index for the right finger

# Change the dynamics properties of the left finger
p.changeDynamics(robot_id, left_finger_index, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.05)

# Change the dynamics properties of the right finger
p.changeDynamics(robot_id, right_finger_index, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.05)

p.setCollisionFilterPair(planeId, robot_id, -1, -1, enableCollision=False)


if CAM is False:
    max_iterations = 20
    position_threshold = 0.005
    orientation_threshold = 0.001




    # Generate random Euler angles
    roll = random.uniform(0, 2 * np.pi)
    pitch = random.uniform(0, 2 * np.pi)
    yaw = random.uniform(0, 2 * np.pi)

    # Convert Euler angles to a quaternion
    random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    object_id_1 = p.loadURDF(urdf_path_1, basePosition=[0.0, 0.0, 0.2], baseOrientation=random_orientation)

    # Change dynamics properties
    p.changeDynamics(object_id_1, linkIndex=-1,
                    mass=0.01,
                    lateralFriction=1,  # Reasonable lateral friction
                    spinningFriction=0.005,  # Reasonable spinning friction
                    rollingFriction=0.005)  # Reasonable rolling friction

    # Enable gravity
    p.setGravity(0, 0, -9.81)



    current_time = time.time()
    while time.time()< current_time+3:
        p.stepSimulation()
        time.sleep(1/140)


# Set the camera settings.
camera_target_position = [0, 0, 0]
camera_distance = 0.6 # Distance from the object
camera_pitch = -90 # Top down view
camera_yaw = 0
camera_roll = 0
up_axis_index = 2
nearVal = 0.01
farVal = 1.0



# Intrinsic parameters
cx = 255.5
cy = 191.75
fx = 552.5
fy = 552.5
width = 516
height = 386

# Intrinsic camera matrix
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Camera position and orientation
camera_position = [0, 0, 0.6]  # Example position
target_position = [0, 0, 0]  # Point the camera is looking at
up_vector = [0, 1, 0]        # Up direction for the camera

view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)

# Aspect ratio
aspect = width / height

# Near and far clipping planes
nearVal = 0.01
farVal = 100

# Calculate horizontal FOV (in radians)
fov_x_rad = 2 * np.arctan(width / (2 * fx))

# Calculate vertical FOV (in radians)
fov_y_rad = 2 * np.arctan(height / (2 * fy))

# Convert radians to degrees
fov_x_deg = np.degrees(fov_x_rad)
fov_y_deg = np.degrees(fov_y_rad)

# projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect = 1.0, nearVal = 0.01, farVal = 1.0)
# Calculate the projection matrix
projection_matrix = p.computeProjectionMatrixFOV(fov_y_deg, aspect, nearVal, farVal)


if CAM is False:
    for i in range(10):
        p.stepSimulation()
        time.sleep(0.01)
    start_time = time.time()  # Record the start time
    while not isObjectStable(object_id_1):
        p.stepSimulation()
        time.sleep(0.01)
        if time.time() - start_time > 3:
            print("Timeout: Stopped waiting for object to stabilize.")
            break
    _, _, rgb_img, depth_img, segImg = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    # Convert depth pixels to real depth values
    Realdepth = farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)

    np.save("simulation/depth_image", Realdepth)

    # segImg = np.array(segImg).reshape(height, width)
    # Create a mask for the target object ID
    object_mask = (segImg == object_id_1).astype(np.uint8) * 255  # Convert to 0-255 range

        # Save the mask as a PNG image
    cv2.imwrite("simulation/object_mask.png", object_mask)
    segmask = BinaryImage.open("simulation/object_mask.png")

# FROM HERE
if CAM == True:
    import pyzed.sl as sl
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the intrinsic parameters and target resolution from the simulation
    sim_cx = 255.5
    sim_cy = 191.75
    sim_fx = 552.5
    sim_fy = 552.5
    sim_width = 516
    sim_height = 386

    # Define the cropping ratio (e.g., 0.5 means crop to 50% of the original dimensions)
    crop_ratio = 0.7

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use QUALITY depth mode for better depth accuracy
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER  # Set the units to meters
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use a resolution that is closest to your target

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open camera")
        exit()

    # Set runtime parameters with a confidence threshold
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 100  # Set a higher threshold for better depth quality

    # Capture one frame
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the depth map
        depth_map = sl.Mat()
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        # Convert the depth map to a numpy array
        depth_data = depth_map.get_data()

        # Resize the depth data to match the simulated camera's resolution
        resized_depth_data = cv2.resize(depth_data, (sim_width, sim_height), interpolation=cv2.INTER_LINEAR)

        # Calculate the cropping dimensions
        crop_width = int(sim_width * crop_ratio)
        crop_height = int(sim_height * crop_ratio)

        # Calculate the top-left corner of the cropping rectangle
        start_x = (sim_width - crop_width) // 2
        start_y = (sim_height - crop_height) // 2

        # Crop the center of the image
        cropped_depth_data = resized_depth_data[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # Resize the cropped image back to the simulation dimensions
        final_depth_data = cv2.resize(cropped_depth_data, (sim_width, sim_height), interpolation=cv2.INTER_LINEAR)

        # Save the final resized depth data as a .npy file
        np.save("final_resized_depth.npy", final_depth_data)
        print("Final resized depth data saved as final_resized_depth.npy")

        # Plot the final resized depth image
        plt.imshow(final_depth_data, cmap='gray')
        plt.colorbar(label='Depth (meters)')
        plt.title('Final Resized Depth Image')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.show()

    # Close the camera
    zed.close()

    action_pj = run_grasping_policy(
        model_name="FC-GQCNN-4.0-PJ",
        depth_image=final_depth_data,
        segmask=None,
        camera_intr="simulation/gqcnn/data/calib/primesense/primesense.intr",
        model_dir="simulation/gqcnn/models",
        config_filename="simulation/gqcnn/cfg/examples/replication/dex-net_4.0_fc_pj.yaml",
        fully_conv=True
    )
    # print("Planned Action:", action_pj)

    action_suction = run_grasping_policy(
        model_name="FC-GQCNN-4.0-SUCTION",
        depth_image=final_depth_data,
        segmask=None,
        camera_intr="simulation/gqcnn/data/calib/primesense/primesense.intr",
        model_dir="simulation/gqcnn/models",
        config_filename="simulation/gqcnn/cfg/examples/replication/dex-net_4.0_fc_suction.yaml",
        fully_conv=True
    )
    print("Planned Action:", action_suction)
# UP TO HERE
elif CAM == False:
        
    action_pj = run_grasping_policy(
        model_name="FC-GQCNN-4.0-PJ",
        depth_image=Realdepth,
        segmask=None,
        camera_intr="simulation/gqcnn/data/calib/primesense/primesense.intr",
        model_dir="simulation/gqcnn/models",
        config_filename="simulation/gqcnn/cfg/examples/replication/dex-net_4.0_fc_pj.yaml",
        fully_conv=True
    )
    # print("Planned Action:", action_pj)

    action_suction = run_grasping_policy(
        model_name="FC-GQCNN-4.0-SUCTION",
        depth_image=Realdepth,
        segmask=None,
        camera_intr="simulation/gqcnn/data/calib/primesense/primesense.intr",
        model_dir="simulation/gqcnn/models",
        config_filename="simulation/gqcnn/cfg/examples/replication/dex-net_4.0_fc_suction.yaml",
        fully_conv=True
    )
    print("Planned Action:", action_suction)

    # Convert the segmentation image to a numpy array and create mask
    # segImg = np.array(segImg).reshape(height, width)  # Reshape to match image dimensions
    mask = (segImg == object_id_1)

    # Apply the mask to the depth data
    masked_depth = np.zeros_like(Realdepth)
    masked_depth[mask] = Realdepth[mask]

    # Calculate the 3D coordinates
    u, v = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    z = masked_depth
    # z = Realdepth
    x = (u - width / 2) / (width / 2) * z * np.tan(fov_x_rad / 2)
    y = (v - height / 2) / (height / 2) * z * np.tan(fov_y_rad / 2)

    # Reshape into Nx3 points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    write_point_cloud_to_obj(points, 'point_cloud.obj')


if action_suction.q_value > action_pj.q_value:
    suction = True
    graspPos = np.array([action_suction.grasp.center[0],action_suction.grasp.center[1],action_suction.grasp.depth])
    x_grasp, y_grasp, z_grasp = generate(graspPos)
    x_world, y_world, z_world = camera_to_world(x_grasp, y_grasp, z_grasp)
    goal_pose = np.array([x_world, y_world, z_world])

    angle1 = action_suction.grasp.angle  # Rotation around z-axis in degrees
    angle2 = action_suction.grasp.approach_angle  # Rotation around local x-axis in degrees
    initial_quaternion = R.from_euler('x', np.pi)
    rotation_z = R.from_euler('z', -angle1)
    quaternion_after_z = rotation_z * initial_quaternion
    rotation_x_local = R.from_euler('y', -angle2)

    # Apply the rotation around the local x-axis
    final_quaternion =  rotation_x_local * quaternion_after_z

    # Get the quaternion representation
    graspQuaternion = final_quaternion.as_quat()
    goal_orn = graspQuaternion
else:
    suction = False
    graspPos = np.array([action_pj.grasp.center[0],action_pj.grasp.center[1],action_pj.grasp.depth])
    x_grasp, y_grasp, z_grasp = generate(graspPos)
    x_world, y_world, z_world = camera_to_world(x_grasp, y_grasp, z_grasp)

    goal_pose = np.array([x_world, y_world, z_world])

    angle1 = action_pj.grasp.angle  # Rotation around z-axis in degrees

    initial_quaternion = R.from_euler('x', np.pi)
    rotation_z = R.from_euler('z', -angle1)
    final_quaternion = rotation_z  * initial_quaternion
    graspQuaternion = final_quaternion.as_quat()
    goal_orn = graspQuaternion
    
if CAM is True:
    kinova_robot.dualGripGraspCAM(suction, goal_pose, goal_orn)
elif CAM is False:
    kinova_robot.dualGripGrasp(suction, goal_pose, goal_orn, object_id_1)

# # for i in np.linspace(0, 0.5, 20):
# #     # Add a static sphere to the simulation
# #     # sphere_start_pos = [x_world, y_world, z_world]  # Position in the air at a reachable height
# #     rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
# #     transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)*i
# #     sphere_start_pos = np.array([x_world, y_world, z_world]) + transformed_offset
# #     sphere_start_orientation = graspQuaternion
# #     sphere_radius = 0.003  # Radius of the sphere

# #     # Create visual shape for the sphere
# #     sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 1])


# #     sphere_id = p.createMultiBody(
# #         baseMass=0,  # Mass of zero to keep it static
# #         # baseCollisionShapeIndex=sphere_collision_shape_id,  # Uncomment if collision detection is needed
# #         baseVisualShapeIndex=sphere_visual_shape_id,
# #         basePosition=sphere_start_pos,
# #         baseOrientation=sphere_start_orientation
# #     )


# for _ in range(max_iterations):

#     start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)


#     rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
#     transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
#     end_pos = np.array([x_world, y_world, z_world]) + transformed_offset
#     end_orn = graspQuaternion

#     num_steps = 300
#     interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)
#     interpolated_orientations = interpolate_orientations(start_orn, end_orn, num_steps)

#     for i in range(num_steps):
#         kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], interpolated_orientations[i]), dtype=object)
#         target_pos = np.array(kinova_angles[:7])
#         kinova_robot.setTargetPositions(target_pos)
#         p.stepSimulation()
#         time.sleep(1./140)

#     print("*******************************************************")
#     print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
#     print("desired pos: ", [x_world, y_world, z_world] + transformed_offset)
#     print("*******************************************************")
#     print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
#     print("desired orn: ", end_orn)
#     print("*******************************************************")

#     for _ in range(100):
#         # Step the simulation
#         p.stepSimulation()
#         time.sleep(1/140)

#     # Get current end-effector pose
#     current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

#     # Calculate position and orientation errors
#     position_error = np.linalg.norm(([x_world, y_world, z_world] + transformed_offset) - (current_position))
#     graspQuaternion = np.array(graspQuaternion)
#     current_orientation = np.array(current_orientation)
#     q1 = graspQuaternion / np.linalg.norm(graspQuaternion)
#     q2 = current_orientation / np.linalg.norm(current_orientation)

#     dot_product = np.dot(q1, q2)

#     orientation_error = 1 - np.abs(dot_product)
#     print("position_error: ", position_error)
#     print("orientation_error: ", orientation_error)
#     # Check if the end-effector is within the desired thresholds
#     if position_error < position_threshold and orientation_error < orientation_threshold:
#         # print(f"Reached target pose in {_} iterations.")
#         break

# for _ in range(100):
#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1/140)

# for _ in range(max_iterations):

#     start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)


#     rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
#     transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
#     end_pos = np.array([x_world, y_world, z_world]) + 0.52 * transformed_offset
#     end_orn = graspQuaternion

#     num_steps = 200
#     interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

#     for i in range(num_steps):
#         kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], graspQuaternion), dtype=object)
#         if suction:
#             target_pos = np.array(kinova_angles[:7])
#         else:
#             target_pos = np.array(kinova_angles[:7])
#         kinova_robot.setTargetPositions(target_pos)
#         p.stepSimulation()
#         time.sleep(1./140)

#     print("*******************************************************")
#     print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
#     print("desired pos: ", [x_world, y_world, z_world] + 0.52 * transformed_offset)
#     print("*******************************************************")
#     print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
#     print("desired orn: ", end_orn)
#     print("*******************************************************")

#     for _ in range(100):
#         # Step the simulation
#         p.stepSimulation()
#         time.sleep(1/140)

#     # Get current end-effector pose
#     current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

#     # Calculate position and orientation errors
#     position_error = np.linalg.norm(([x_world, y_world, z_world] + 0.52 * transformed_offset) - (current_position))
#     graspQuaternion = np.array(graspQuaternion)
#     current_orientation = np.array(current_orientation)
#     q1 = graspQuaternion / np.linalg.norm(graspQuaternion)
#     q2 = current_orientation / np.linalg.norm(current_orientation)

#     dot_product = np.dot(q1, q2)

#     orientation_error = 1 - np.abs(dot_product)
#     # print("position_error: ", position_error)
#     # print("orientation_error: ", orientation_error)
#     # Check if the end-effector is within the desired thresholds
#     if position_error < position_threshold and orientation_error < orientation_threshold:
#         print(f"Reached target pose in {_} iterations.")
#         break

# for _ in range(100):
#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1/140)

# for _ in range(max_iterations):

#     start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)


#     rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
#     transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
#     end_pos = np.array([x_world, y_world, z_world]) + 0.38 * transformed_offset
#     end_orn = graspQuaternion

#     num_steps = 100
#     interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

#     for i in range(num_steps):
#         kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], graspQuaternion), dtype=object)
#         if suction:
#             target_pos = np.array(kinova_angles[:7])
#         else:
#             target_pos = np.array(kinova_angles[:7])
#         kinova_robot.setTargetPositions(target_pos)
#         p.stepSimulation()
#         time.sleep(1./140)

#     print("*******************************************************")
#     print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
#     print("desired pos: ", [x_world, y_world, z_world] + 0.38 * transformed_offset)
#     print("*******************************************************")
#     print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
#     print("desired orn: ", end_orn)
#     print("*******************************************************")

#     for _ in range(100):
#         # Step the simulation
#         p.stepSimulation()
#         time.sleep(1/140)

#     # Get current end-effector pose
#     current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

#     # Calculate position and orientation errors
#     position_error = np.linalg.norm(([x_world, y_world, z_world] + 0.38 * transformed_offset) - (current_position))
#     graspQuaternion = np.array(graspQuaternion)
#     current_orientation = np.array(current_orientation)
#     q1 = graspQuaternion / np.linalg.norm(graspQuaternion)
#     q2 = current_orientation / np.linalg.norm(current_orientation)

#     dot_product = np.dot(q1, q2)

#     orientation_error = 1 - np.abs(dot_product)
#     # print("position_error: ", position_error)
#     # print("orientation_error: ", orientation_error)
#     # Check if the end-effector is within the desired thresholds
#     if position_error < position_threshold and orientation_error < orientation_threshold:
#         print(f"Reached target pose in {_} iterations.")
#         break

# for _ in range(100):
#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1/140)




# start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)



# end_pos = np.array([x_world, y_world, z_world]) + 0.38 * transformed_offset
# end_orn = start_orn

# # p.setJointMotorControlArray(
# #             bodyUniqueId=robot_id,
# #             jointIndices=[8],
# #             controlMode=p.POSITION_CONTROL,
# #             targetPositions=[gripper_close],
# #             forces=[850])
# # p.stepSimulation()
# finish_time = time.time() + 2.0
# while time.time() < finish_time:
#     p.stepSimulation()
#     p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=8, controlMode=p.VELOCITY_CONTROL, targetVelocity=4.0, force=5)
#     p.setJointMotorControlArray(
#             bodyUniqueId=robot_id,
#             jointIndices=[11],
#             controlMode=p.POSITION_CONTROL,
#             targetPositions=[-0.05],
#             forces=[20])
#     time.sleep(1/140)



# # for i in range(p.getNumJoints(robot_id)):
# #     info = p.getJointInfo(robot_id, i)
# #     joint_name = info[1].decode('utf-8')
# #     joint_index = info[0]
# #     link_name = info[12].decode('utf-8')
# #     link_index = i
# #     print(f"Joint name: {joint_name}, Joint index: {joint_index}, Link name: {link_name}, Link index: {link_index}")



# # start_pos, start_orn = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

# # close_gripper(robot_id, [8], [gripper_close], [100])

# rotation_matrix = np.array(p.getMatrixFromQuaternion(graspQuaternion)).reshape(3, 3)
# transformed_offset = rotation_matrix.dot(kinova_robot.endEffectorOffset)
# end_pos = np.array([start_pos[0], start_pos[1], start_pos[2]+0.3])
# end_orn = start_orn

# num_steps = 100
# interpolated_positions = interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

# for i in range(num_steps):
#     kinova_angles = np.array(kinova_robot.solveInverseKinematics( interpolated_positions[i], graspQuaternion), dtype=object)
#     if suction:
#         target_pos = np.array(kinova_angles[:7])
#     else:
#         target_pos = np.array(kinova_angles[:7])
#     kinova_robot.setTargetPositions(target_pos)
#     p.stepSimulation()
#     time.sleep(1./140)

# print("*******************************************************")
# print('Current pos: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[0])
# print("desired pos: ", [x_world, y_world, z_world+0.3] + 0.42 * transformed_offset)
# print("*******************************************************")
# print('Current orn: ', get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)[1])
# print("desired orn: ", end_orn)
# print("*******************************************************")


# for _ in range(1000):
#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1/140)

p.disconnect()

