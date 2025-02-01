import pybullet as p
import pybullet_data
import time
from kinovaGripper import Kinova
import numpy as np
from bullet.random_pointcloud import select_random_obj
from bullet.random_pointcloud import calculate_dimensions
from bullet.random_pointcloud import generate_urdf

gripper_close = 0.673
gripper_open = -0.35

def get_current_pose(robotId, end_effector_index):
    state = p.getLinkState(robotId, end_effector_index)
    position = state[4]
    orientation = state[5]
    return position, orientation

# Connect to PyBullet and create a simulation environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load URDF files

# Load a plane for reference
p.loadURDF("plane.urdf")




# Instantiate the Kinova class
# Assuming init_joints is an optional parameter. If you want to initialize with specific joint positions, provide them.
init_joints = np.array([0, 0, 0, 0, 0, 0, 0, 'open'], dtype=object) 
base_shift = [0, 0, 0]  # Example base shift

# Create an instance of the Kinova robot
kinova_robot = Kinova(init_joints=init_joints, base_shift=base_shift)

num_joints = p.getNumJoints(kinova_robot.robot)
robot_id = kinova_robot.robot

# Define the link indices for the gripper fingers
left_finger_index = 11  # Replace with the actual link index for the left finger
right_finger_index = 13  # Replace with the actual link index for the right finger

# Change the dynamics properties of the left finger
p.changeDynamics(robot_id, left_finger_index, lateralFriction=1.0, rollingFriction=1, spinningFriction=5)

# Change the dynamics properties of the right finger
p.changeDynamics(robot_id, right_finger_index, lateralFriction=1.0, rollingFriction=1, spinningFriction=5)

# Add a static cube to the simulation
cube_start_pos = [0.5, 0.325, 0.1]  # Position in the air at a reachable height
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.017, 0.017, 0.017], rgbaColor=[1, 0, 0, 1])
cube_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.017, 0.017, 0.017])

cube_id = p.createMultiBody(
    baseMass=0,  # Mass of zero to keep it static
    baseCollisionShapeIndex=cube_collision_shape_id,
    baseVisualShapeIndex=cube_visual_shape_id,
    basePosition=cube_start_pos,
    baseOrientation=cube_start_orientation
)


max_iterations = 10000
position_threshold = 0.17
orientation_threshold = 2
cube_position, cube_orientation = p.getBasePositionAndOrientation(cube_id)
grasp_orientation = p.getQuaternionFromEuler([np.pi, 0, np.pi/2])  # Adjust as needed
# for _ in range(max_iterations):
#     # Compute inverse kinematics
    
    
#     kinova_angles = np.array(kinova_robot.solveInverseKinematics([cube_position[0],cube_position[1],cube_position[2]], grasp_orientation), dtype=object)
#     target_pos = np.insert(kinova_angles[:7], 7, 'open')
#     kinova_robot.setTargetPositions(target_pos)

#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1./140.)

#     # Get current end-effector pose
#     current_position, current_orientation = get_current_pose(robot_id, kinova_robot.kinovaEndEffectorIndex)

#     # Calculate position and orientation errors
#     position_error = np.linalg.norm(np.array(cube_position) - np.array(current_position))
#     orientation_error = np.linalg.norm(np.array(grasp_orientation) - np.array(current_orientation))
#     print(position_error)
#     print(orientation_error)
#     # Check if the end-effector is within the desired thresholds
#     if position_error < position_threshold and orientation_error < orientation_threshold:
#         print(f"Reached target pose in {_} iterations.")
#         break

# for _ in range(1000):
#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1/140)
    

for j in range(100):

    # Run the simulation to see the robot in action

    # Get the cube's position and orientation
    cube_position, cube_orientation = p.getBasePositionAndOrientation(cube_id)
    print(cube_position, cube_orientation)

    # Define the end-effector orientation for grasping the cube from the top
    grasp_orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # Adjust as needed

    # Solve inverse kinematics to find joint positions for the target end-effector position and orientation
    kinova_angles = np.array(kinova_robot.solveInverseKinematics([cube_position[0],cube_position[1],cube_position[2]+0.16], grasp_orientation), dtype=object)
    print(kinova_angles[:7])

    # Move the robot to the target joint positions to approach the cube
    target_pos = np.array(kinova_angles[:7])
    kinova_robot.setTargetPositions(target_pos)


    for i in range(100):
        target_pos = np.array(kinova_angles[:7])
        kinova_robot.setTargetPositions(target_pos)
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[gripper_open])
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[11],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[0])
        p.stepSimulation()
        time.sleep(1/140)

    # Close the gripper to grasp the cube
    for i in range(100):
        target_pos = np.array(kinova_angles[:7])
        kinova_robot.setTargetPositions(target_pos)
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[gripper_close])
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[11],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[-0.05])
        p.stepSimulation()
        time.sleep(1/140)

# Disconnect from PyBullet
p.disconnect()
