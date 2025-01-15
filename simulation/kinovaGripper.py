import pybullet as p
import pybullet_data
import time
import os
import IPython
import numpy as np
from scipy.interpolate import CubicSpline

def quat_to_euler(quat):
    return p.getEulerFromQuaternion(quat)

def euler_to_quat(euler):
    return p.getQuaternionFromEuler(euler)

def compute_relative_transform(link_pos, link_orn, object_pos, object_orn):
    link_inv_pos, link_inv_orn = p.invertTransform(link_pos, link_orn)
    rel_pos, rel_orn = p.multiplyTransforms(link_inv_pos, link_inv_orn, object_pos, object_orn)
    return rel_pos, rel_orn

class Kinova:
    def __init__(self, init_joints=None, base_shift=[0, 0, 0]):
        self.position_threshold = 0.005
        self.orientation_threshold = 0.001
        p_gain = 0.01
        self.position_control_gain_p =  [
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain,
            p_gain
        ]
        d_gain = 0.1*10
        self.position_control_gain_d = [
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain,
            d_gain
        ]
        f_max = 250*3
        self.max_torque = [
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max,
            f_max
        ]
        self.robot = p.loadURDF(
            "/home/elyar/thesis/DualGripNet/simulation/ROBOTIQ85.URDF",
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self._base_position = [
            0.0 + base_shift[0],
            0.0 + base_shift[1],
            0.0 + base_shift[2],
        ]
        self.kinovaUid = self.robot
        self.dof = p.getNumJoints(self.robot)
        # gripper constraints
        finger_joint_index = 8  # Replace with the actual index of 'finger_joint'
        left_inner_knuckle_joint_index = 10  # Replace with the actual index of 'left_inner_knuckle_joint'
        left_inner_finger_joint_index = 11  # Replace with the actual index of 'left_inner_finger_joint'
        right_inner_knuckle_joint_index = 12  # Replace with the actual index of 'right_inner_knuckle_joint'
        right_inner_finger_joint_index = 13  # Replace with the actual index of 'right_inner_finger_joint'
        right_outer_knuckle_joint_index = 14  # Replace with the actual index of 'right_outer_knuckle_joint'

        # Define the axis of rotation for the gear constraints (assuming Y-axis for example)
        joint_axis = [0, -1, 0]
        right_joint_axis = [0, 1, 0]  # Right joints have opposite axis

        # Positions in the parent and child link frames from URDF
        parent_frame_position_finger_joint = [0.0306011444260539, 0, 0.0627920162695395]  # from 'finger_joint'
        child_frame_position_left_inner_knuckle_joint = [0.0127000000001501, 0, 0.0693074999999639]  # from 'left_inner_knuckle_joint'
        child_frame_position_left_inner_finger_joint = [0.034585310861294, 0, 0.0454970193817975]  # from 'left_inner_finger_joint'
        child_frame_position_right_inner_knuckle_joint = [-0.0126999999998499, 0, 0.0693075000000361]  # from 'right_inner_knuckle_joint'
        child_frame_position_right_inner_finger_joint = [0.0341060475457406, 0, 0.0458573878541688]  # from 'right_inner_finger_joint'
        child_frame_position_right_outer_knuckle_joint = [-0.0306011444258893, 0, 0.0627920162695395]  # from 'right_outer_knuckle_joint'

        # Create the gear constraints
        # Mimic 'finger_joint' with 'left_inner_knuckle_joint'
        gear_constraint_1 = p.createConstraint(
            self.kinovaUid,
            finger_joint_index,
            self.kinovaUid,
            left_inner_knuckle_joint_index,
            p.JOINT_GEAR,
            joint_axis,
            parent_frame_position_finger_joint,
            child_frame_position_left_inner_knuckle_joint
        )
        p.changeConstraint(gear_constraint_1, gearRatio=-1.0, erp = 1, maxForce = 50)

        gear_constraint_2 = p.createConstraint(
            self.kinovaUid,
            finger_joint_index,
            self.kinovaUid,
            right_inner_knuckle_joint_index,
            p.JOINT_GEAR,
            right_joint_axis,
            parent_frame_position_finger_joint,
            child_frame_position_right_inner_knuckle_joint
        )
        p.changeConstraint(gear_constraint_2, gearRatio=1.0, erp = 1, maxForce = 50)

        # Mimic 'finger_joint' with 'right_outer_knuckle_joint' (in opposite direction)
        gear_constraint_3 = p.createConstraint(
            self.kinovaUid,
            finger_joint_index,
            self.kinovaUid,
            right_outer_knuckle_joint_index,
            p.JOINT_GEAR,
            right_joint_axis,
            parent_frame_position_finger_joint,
            child_frame_position_right_outer_knuckle_joint
        )
        p.changeConstraint(gear_constraint_3, gearRatio=1.0, erp = 1, maxForce = 50)


        # Mimic 'finger_joint' with 'right_inner_finger_joint'
        gear_constraint_4 = p.createConstraint(
            self.kinovaUid,
            left_inner_finger_joint_index,
            self.kinovaUid,
            right_inner_finger_joint_index,
            p.JOINT_GEAR,
            right_joint_axis,
            parent_frame_position_finger_joint,
            child_frame_position_right_inner_finger_joint
        )
        p.changeConstraint(gear_constraint_4, gearRatio=-1.0, erp = 0.8, maxForce = 50)

        


        # List of non-fixed joint indices
        self.non_fixed_joint_indices = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14]
        
        # Initialize the lists
        self.kinova_joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = [0.0, -0.9, 0.0, 2.5, 0.0, 0.5, -1.57]
        self.target_torque = []
        self.kinovaEndEffectorIndex = 7
        self.endEffectorOffset = np.array([0, 0, -0.4])

        # Loop through the specified non-fixed joints
        for j in range(7):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
            joint_info = p.getJointInfo(self.robot, j)
            self.kinova_joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_torque.append(0.0)

        self.reset(init_joints)

    def reset(self, joints=None):
        self.t = 0.0
        self.control_mode = "torque"
        p.resetBasePositionAndOrientation(
            self.kinovaUid, self._base_position, [0.000000, 0.000000, 0.000000, 1.000000]
        )
        if joints is None:
            self.target_pos = [0.0, -0.9, 0.0, 2.5, 0.0, 0.5, -1.57]
            for j in range(7):
                self.target_torque[j] = 0.0
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j])
        else:
            joints = self.append(joints)
            for j in range(7):
                self.target_pos[j] = joints[j]
                self.target_torque[j] = 0.0
                p.resetJointState(self.robot, j, targetValue=self.target_pos[j])

        self.resetController()
        self.setTargetPositions(self.target_pos)

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    def resetController(self):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.kinova_joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for i in range(7)],
        )

    def append(self, target_pos):

        # if len(target_pos) == 8:
        #     # if target_pos[7] == 'open':
        #     #     target_pos[7] = -0.35
        #     # elif target_pos[7] == 'close':
        #     #     target_pos[7] = 0.673
        #     target_pos = np.insert(target_pos, 7, 0)
        #     target_pos = np.insert(target_pos, 9, 0)
        #     target_pos = np.insert(target_pos, 10, target_pos[8])
        #     target_pos = np.insert(target_pos, 11, -target_pos[8])
        #     target_pos = np.insert(target_pos, 12, -target_pos[8])
        #     target_pos = np.insert(target_pos, 13, target_pos[8])
        #     target_pos = np.insert(target_pos, 14, -target_pos[8])
        #     target_pos = np.insert(target_pos, 15, 0)
        return target_pos

    def setTargetPositions(self, target_pos):
        self.target_pos = self.append(target_pos)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.kinova_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos,
            forces=self.max_torque[:7],
            positionGains=self.position_control_gain_p[:7],
            velocityGains=self.position_control_gain_d[:7],
        )

    # def setTargetPositions(self, target_pos):
    #     self.target_pos = self.append(target_pos)
    #     p.setJointMotorControlArray(
    #         bodyUniqueId=self.robot,
    #         jointIndices=self.joints,
    #         controlMode=p.POSITION_CONTROL,
    #         targetPositions=self.target_pos,
    #     )

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot, self.kinova_joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel
    
    def solveInverseDynamics(self, pos, vel, acc):
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        rotation_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
        modified_pos = np.array(pos) + transformed_offset
        # return list(p.calculateInverseKinematics(self.robot, self.kinovaEndEffectorIndex, pos, ori, maxNumIterations=100, residualThreshold=0.01,))
        return list(p.calculateInverseKinematics(self.robot, self.kinovaEndEffectorIndex, pos, ori))
    
    def interpolate_positions(self, start_pos, end_pos, num_steps):
        t = np.linspace(0, 1, num_steps)
        start_pos = np.array(start_pos).flatten()
        end_pos = np.array(end_pos).flatten()
        cs = CubicSpline([0, 1], np.vstack([start_pos, end_pos]), axis=0)
        return cs(t)

    def interpolate_orientations(self, start_orn, end_orn, num_steps):
        t = np.linspace(0, 1, num_steps)
        return [self.slerp(start_orn, end_orn, ti) for ti in t]
    def slerp(self, q0, q1, t):
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
    
    def get_current_pose(self):
        state = p.getLinkState(self.robot, self.kinovaEndEffectorIndex, computeLinkVelocity=True)
        position = state[0]
        orientation = state[5]
        return position, orientation
    
    def dualGripGrasp(self, suction, goal_pose, goal_orn, object_id):
        

        #########################################
        ########      TOWARD OBJECT     #########
        #########################################

        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()
            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            end_pos = goal_pose + transformed_offset
            end_orn = goal_orn

            num_steps = 300

            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)
            interpolated_orientations = self.interpolate_orientations(start_orn, end_orn, num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], interpolated_orientations[i]), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)

            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - (current_position))
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
            print("position_error: ", position_error)
            print("orientation_error: ", orientation_error)
            # Check if the end-effector is within the desired thresholds
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        


        #########################################
        ########     FIRST APPROACH     #########
        #########################################
        if suction:
            finish_time = time.time() + 2.0
            print("CLOSING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[-0.35, -0.05],
                        forces=[100,20])
                time.sleep(1/140)
        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()


            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            end_pos = goal_pose + 0.54 * transformed_offset
            end_orn = goal_orn

            num_steps = 200
            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], goal_orn), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)


            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - current_position)
            goal_orn = np.array(goal_orn)
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
            # print("position_error: ", position_error)
            # print("orientation_error: ", orientation_error)
            # Check if the end-effector is within the desired thresholds
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        if suction is False:
            finish_time = time.time() + 2.0
            print("OPENING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[0.3,0],
                        forces=[100,20])
                time.sleep(1/140)
                

        ####################################
        ######     FINAL APPROACH     ######
        ####################################

    
        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()


            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            if suction:
                end_pos = np.array(goal_pose) + 0.46 * transformed_offset
            else:
                end_pos = np.array(goal_pose) + 0.40 * transformed_offset
            end_orn = goal_orn

            num_steps = 100
            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], goal_orn), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)


            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - current_position)
            goal_orn = np.array(goal_orn)
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
    
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        ####################################
        ######     LIFTING OBJECT     ######
        ####################################
        if suction:
            for _ in range(300):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

        start_pos, start_orn = self.get_current_pose()

        if suction is False:
            finish_time = time.time() + 4.0
            print("CLOSING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[-0.35, -0.05],
                        forces=[100,20])
                time.sleep(1/140)

    def dualGripGraspCAM(self, suction, goal_pose, goal_orn):
        

        #########################################
        ########      TOWARD OBJECT     #########
        #########################################

        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()
            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            end_pos = goal_pose + transformed_offset
            end_orn = goal_orn

            num_steps = 300

            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)
            interpolated_orientations = self.interpolate_orientations(start_orn, end_orn, num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], interpolated_orientations[i]), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)

            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - (current_position))
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
            print("position_error: ", position_error)
            print("orientation_error: ", orientation_error)
            # Check if the end-effector is within the desired thresholds
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        


        #########################################
        ########     FIRST APPROACH     #########
        #########################################
        if suction:
            finish_time = time.time() + 2.0
            print("CLOSING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[-0.35, -0.05],
                        forces=[100,20])
                time.sleep(1/140)
        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()


            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            end_pos = goal_pose + 0.54 * transformed_offset
            end_orn = goal_orn

            num_steps = 200
            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], goal_orn), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)


            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - current_position)
            goal_orn = np.array(goal_orn)
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
            # print("position_error: ", position_error)
            # print("orientation_error: ", orientation_error)
            # Check if the end-effector is within the desired thresholds
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        if suction is False:
            finish_time = time.time() + 2.0
            print("OPENING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[0.3,0],
                        forces=[100,20])
                time.sleep(1/140)
                

        ####################################
        ######     FINAL APPROACH     ######
        ####################################

    
        for _ in range(100):

            start_pos, start_orn = self.get_current_pose()


            rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
            transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
            if suction:
                end_pos = np.array(goal_pose) + 0.46 * transformed_offset
            else:
                end_pos = np.array(goal_pose) + 0.40 * transformed_offset
            end_orn = goal_orn

            num_steps = 100
            interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

            for i in range(num_steps):
                kinova_angles = np.array(self.solveInverseKinematics( interpolated_positions[i], goal_orn), dtype=object)
                target_pos = np.array(kinova_angles[:7])
                self.setTargetPositions(target_pos)
                p.stepSimulation()
                time.sleep(1./140)


            for _ in range(100):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

            # Get current end-effector pose
            current_position, current_orientation = self.get_current_pose()

            # Calculate position and orientation errors
            position_error = np.linalg.norm(end_pos - current_position)
            goal_orn = np.array(goal_orn)
            current_orientation = np.array(current_orientation)
            q1 = goal_orn / np.linalg.norm(goal_orn)
            q2 = current_orientation / np.linalg.norm(current_orientation)

            dot_product = np.dot(q1, q2)

            orientation_error = 1 - np.abs(dot_product)
    
            if position_error < self.position_threshold and orientation_error < self.orientation_threshold:
                print(f"Reached target pose in {_} iterations.")
                break

        for _ in range(100):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

        ####################################
        ######     LIFTING OBJECT     ######
        ####################################
        if suction:
            for _ in range(300):
                # Step the simulation
                p.stepSimulation()
                time.sleep(1/140)

        start_pos, start_orn = self.get_current_pose()

        if suction is False:
            finish_time = time.time() + 2.0
            print("CLOSING GRIPPER")
            while time.time() < finish_time:
                p.stepSimulation()
                p.setJointMotorControlArray(
                        bodyUniqueId=self.robot,
                        jointIndices=[14,11],
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=[-0.35, -0.05],
                        forces=[100,20])
                time.sleep(1/140)


        rotation_matrix = np.array(p.getMatrixFromQuaternion(goal_orn)).reshape(3, 3)
        transformed_offset = rotation_matrix.dot(self.endEffectorOffset)
        end_pos = np.array([start_pos[0], start_pos[1], start_pos[2]+0.3])
        end_orn = start_orn

        num_steps = 100
        interpolated_positions = self.interpolate_positions(np.array(start_pos), np.array(end_pos), num_steps)

        link_index = 15
        initial_link_pos, initial_link_orn = p.getLinkState(self.robot, link_index)[4:6]

        p.stepSimulation()
        time.sleep(1./140)
                


        for _ in range(1000):
            # Step the simulation
            p.stepSimulation()
            time.sleep(1/140)

if __name__ == "__main__":
    robot = Kinova()

    def gripper(action):
        if action:
            targetPosition = -0.35
        else:
            targetPosition = 0.673
        for i in range(num_joints):
            # Set the target position for the gripper joints
            if(i==8):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition)
            if(i==10):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition)
            if(i==11):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, -targetPosition)
            if(i==12):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, -targetPosition)
            if(i==13):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition)
            if(i==14):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, -targetPosition)


    # Connect to PyBullet
    p.connect(p.GUI)

    # Set the search path to find URDF files
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load the plane and Robotiq 85 gripper URDF
    planeId = p.loadURDF("plane.urdf")
    robotId = p.loadURDF("/home/elyar/thesis/DualGripNet/simulation/ROBOTIQ85.URDF", basePosition=[0, 0, 0.2], useFixedBase=True)

    # Set gravity
    p.setGravity(0, 0, -9.8)

    # Get the number of joints in the gripper
    num_joints = p.getNumJoints(robotId)
    print(f"Number of joints in the gripper: {num_joints}")

    # List the joint information
    for i in range(num_joints):
        joint_info = p.getJointInfo(robotId, i)
        print(f"Joint {i}: {joint_info}")

    # Control the gripper (example: close the gripper)
    open = False
    while True:
        for i in range(100):
            open = False
            gripper(open)
            p.stepSimulation()
            time.sleep(1./240.)
        for i in range(100):
            open = True
            gripper(open)
            p.stepSimulation()
            time.sleep(1./240.)
            
        

    # Disconnect from PyBullet
    p.disconnect()
