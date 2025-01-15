import os
import random
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
import time


def write_point_cloud_to_obj(points, file_name):
    with open(file_name, 'w') as file:
        for p in points:
            if p[2]>0:
                # Write each point as a vertex in the OBJ file
                file.write(f"v {p[0]} {p[1]} {p[2]}\n")


def select_random_obj(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.obj')]
    if not files:
        raise ValueError("No .obj files found in the directory.")
    selected_file = random.choice(files)
    print("Selected random object is: ", selected_file)
    return os.path.join(directory, selected_file)

def calculate_dimensions(obj_path):
    mesh = trimesh.load(obj_path, force='mesh')
    bounding_box = mesh.bounding_box_oriented.vertices
    length = np.max(bounding_box[:, 0]) - np.min(bounding_box[:, 0])
    width = np.max(bounding_box[:, 1]) - np.min(bounding_box[:, 1])
    height = np.max(bounding_box[:, 2]) - np.min(bounding_box[:, 2])
    return length, width, height

def calculate_inertia(length, width, height, mass):
    Ixx = (1 / 12) * mass * (width ** 2 + height ** 2)
    Iyy = (1 / 12) * mass * (length ** 2 + height ** 2)
    Izz = (1 / 12) * mass * (length ** 2 + width ** 2)
    return Ixx, Iyy, Izz

def generate_urdf(obj_path, length, width, height, mass=0.2):
    Ixx, Iyy, Izz = calculate_inertia(length, width, height, mass)
    
    robot = ET.Element('robot', name='generated_object')
    link = ET.SubElement(robot, 'link', name='base_link')
    
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value=str(mass))
    ET.SubElement(inertial, 'inertia', ixx=str(Ixx), ixy="0", ixz="0",
                  iyy=str(Iyy), iyz="0", izz=str(Izz))
    
    visual = ET.SubElement(link, 'visual')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'mesh', filename=obj_path, scale='1 1 1')
    
    collision = ET.SubElement(link, 'collision')
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'mesh', filename=obj_path, scale='1 1 1')
    
    tree = ET.ElementTree(robot)
    urdf_path = os.path.splitext(obj_path)[0] + '.urdf'
    tree.write(urdf_path)
    return urdf_path

def isObjectStable(object_id, linear_threshold=0.01, angular_threshold=0.01):
    velocity, angular_velocity = p.getBaseVelocity(object_id)
    linear_speed = np.linalg.norm(velocity)
    angular_speed = np.linalg.norm(angular_velocity)
    return linear_speed < linear_threshold and angular_speed < angular_threshold

def capture_depth_image(physicsClientId, camera_target_position, width=640, height=480):
    camera_distance = 2.0
    camera_pitch = -30
    camera_yaw = 50
    fov = 60
    aspect = width / height
    near = 0.1
    far = 100

    view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target_position, camera_distance, camera_yaw, camera_pitch, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    _, _, _, depth_img, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, physicsClientId=physicsClientId)

    depth_buffer = np.reshape(depth_img, (height, width))
    depth = far * near / (far - (far - near) * depth_buffer)
    return depth

def capture_depth_image(physicsClientId, camera_target_position, width=640, height=480):
    camera_distance = 2.0
    camera_pitch = -90  # Looking straight down
    camera_yaw = 0
    fov = 60
    aspect = width / height
    near = 0.1
    far = 100

    view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target_position, camera_distance, camera_yaw, camera_pitch, 0, 2)
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    _, _, _, depth_img, _ = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, physicsClientId=physicsClientId)
    depth_buffer = np.reshape(depth_img, (height, width))
    depth = far * near / (far - (far - near) * depth_buffer)
    return depth

def load_urdf_fall_simulate(urdf_path):
    physicsClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1.3, 180.0, -41.0, [-0.35, -0.58, -0.88])

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0.0])

    # Generate random Euler angles
    roll = random.uniform(0, 2 * np.pi)
    pitch = random.uniform(0, 2 * np.pi)
    yaw = random.uniform(0, 2 * np.pi)

    # Convert Euler angles to a quaternion
    random_orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    object_id = p.loadURDF(urdf_path, basePosition=[0,0,0.3], baseOrientation=random_orientation)
    p.changeDynamics(object_id, linkIndex=-1, lateralFriction=0.2, rollingFriction=0.005)

    # Set the initial camera parameters
    camera_distance = 0.6  # Distance from the camera to the target
    camera_yaw = 0      # Yaw angle in degrees
    camera_pitch = -40   # Pitch angle in degrees
    camera_target_position = [0, 0, 0]  # The XYZ position the camera is looking at

    # Reset the camera view at the beginning of the simulation
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                             cameraYaw=camera_yaw,
                             cameraPitch=camera_pitch,
                             cameraTargetPosition=camera_target_position)

    # Set the camera settings.
    camera_target_position = [0, 0, 0]
    camera_distance = 0.4  # Distance from the object
    camera_pitch = -90  # Top down view
    camera_yaw = 0
    camera_roll = 0
    up_axis_index = 2
    aspect = 1.0
    nearVal = 0.01
    farVal = 1.0
    fov = 60

    view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target_position, camera_distance, camera_yaw, camera_pitch, camera_roll, up_axis_index)
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect = 1.0, nearVal = 0.01, farVal = 1.0)

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
    width, height, rgb_img, depth_img, segImg = p.getCameraImage(width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix)

    # Convert depth pixels to real depth values
    depth = farVal * nearVal / (farVal - (farVal - nearVal) * depth_img)
    
    # Convert the segmentation image to a numpy array and create mask
    segImg = np.array(segImg).reshape(height, width)  # Reshape to match image dimensions
    mask = (segImg == object_id)

    # Apply the mask to the depth data
    masked_depth = np.zeros_like(depth)
    masked_depth[mask] = depth[mask]

    # Calculate the 3D coordinates
    u, v = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height))
    z = masked_depth
    x = (u - width / 2) / (width / 2) * z * np.tan(np.deg2rad(fov / 2))
    y = (v - height / 2) / (height / 2) * z * np.tan(np.deg2rad(fov / 2))

    # Reshape into Nx3 points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    write_point_cloud_to_obj(points, 'point_cloud.obj')

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


    pos, orn = p.getBasePositionAndOrientation(object_id)
    print("Final position:", pos)
    print("Final orientation (quaternion):", orn)
    p.disconnect()
    return (pos, orn, depth_img)

def gen_ran_obj(directory = "objects/database/OMG_objects/003_cracker_box/"):
    try:
        obj_path = select_random_obj(directory)
        length, width, height = calculate_dimensions(obj_path)
        urdf_path = generate_urdf(obj_path, length, width, height)
        print(f"Generated URDF at {urdf_path}")
        pos, orn, depth = load_urdf_fall_simulate(urdf_path)
        os.remove(urdf_path)
        return pos, orn, obj_path, depth
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    gen_ran_obj()
