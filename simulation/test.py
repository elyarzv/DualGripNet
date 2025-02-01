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
object_directories = "dataset/objects/database/OMG_objects/007_tuna_fish_can"
object_directories = "dataset/objects/database/OMG_objects/009_gelatin_box"
# object_directories = "/home/elyar/thesis/DualGripNet/dataset/objects/database/dexnet_objects"


gripper_close = 0.673
gripper_open = -0.35


    

obj_path = select_random_obj(object_directories)
obj_length, obj_width, obj_height = calculate_dimensions(obj_path)
urdf_path = generate_urdf(obj_path, obj_length, obj_width, obj_height)

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



# Define the link indices for the gripper fingers
left_finger_index = 11  # Replace with the actual link index for the left finger
right_finger_index = 13  # Replace with the actual link index for the right finger

# Change the dynamics properties of the left finger
p.changeDynamics(robot_id, left_finger_index, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.05)

# Change the dynamics properties of the right finger
p.changeDynamics(robot_id, right_finger_index, lateralFriction=1.0, rollingFriction=0.01, spinningFriction=0.05)

p.setCollisionFilterPair(planeId, robot_id, -1, -1, enableCollision=False)

max_iterations = 20
position_threshold = 0.005
orientation_threshold = 0.001


# Generate random Euler angles
roll = random.uniform(0, 2 * np.pi)
pitch = random.uniform(0, 2 * np.pi)
yaw = random.uniform(0, 2 * np.pi)

# Convert Euler angles to a quaternion
random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

object_id = p.loadURDF(urdf_path, basePosition=[0.0, 0.0, 0.4], baseOrientation=random_orientation)


# Generate a random initial orientation
random_orientation = p.getQuaternionFromEuler(np.random.uniform(low=[-np.pi, -np.pi, -np.pi], high=[np.pi, np.pi, np.pi]))

# Set the random initial position and orientation
p.resetBasePositionAndOrientation(object_id, [0.0, 0.0, 0.6], (0.6439643885176298, 0.3374784977831525, 0.6668226309341145, -0.16360228827631776))

# Enable gravity
p.setGravity(0, 0, -9.81)

# Define the dimensions of the box (half extents)
half_extents = [0.02, 0.05, 0.05]  # This creates a 1x1x1 box

# Create a collision shape for the box
collision_shape_id = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=half_extents
)

# Create a visual shape for the box (optional)
visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=half_extents,
    rgbaColor=[1, 0, 0, 1]  # Red color
)

# Create the box in the simulation
box_id = p.createMultiBody(
    baseMass=1,  # Mass of the box
    baseCollisionShapeIndex=collision_shape_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0.2, 0, 0.3]  # Initial position of the box
)

for i in range(400):
    p.stepSimulation()
    print(i)
    time.sleep(0.01)


# Set the camera settings.
camera_target_position = [0, 0, 0]
camera_distance = 0.6 # Distance from the object
camera_pitch = -90 # Top down view
camera_yaw = 0
camera_roll = 0
up_axis_index = 2
nearVal = 0.01
farVal = 1.0







print("Grasping")
finish_time = time.time() + 10.0
while time.time() < finish_time:
    p.stepSimulation()
    p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=8, controlMode=p.VELOCITY_CONTROL, targetVelocity=6.8, force=100)
    time.sleep(1/140)


p.disconnect()




