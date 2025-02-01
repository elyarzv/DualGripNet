import numpy as np
import time

def rotate_translate_vertices(vertices, destination_point, approach_vector):

    # Ensure vector is normalized
    normalized_approach_vector = approach_vector / np.linalg.norm(approach_vector)  

    # Define the z-axis vector
    v1 = np.array([0, 0, 1])

    # Calculate rotation axis
    N = np.cross(v1, normalized_approach_vector)
    # print("N is:", N)
    # time.sleep(5)
    # Calculate rotation angle
    dot_product = np.dot(v1, normalized_approach_vector)
    # print("dot product is:" , dot_product)
    # time.sleep(5)
    angle = np.arccos(dot_product)
    # print("angle is:" , angle)
    # time.sleep(5)

    if np.isclose(angle, 0):
        # No rotation is needed
        rotated_vertices = vertices

        # Calculate translation vector
        translation_vector = destination_point - rotated_vertices[0]

        # Translate the rotated vertices
        translated_vertices = rotated_vertices + translation_vector

        # return the translated vertices
        return translated_vertices

    # Normalize rotation axis
    if np.linalg.norm(N) != 0:
        # print("first condition")
        # time.sleep(5)
        N /= np.linalg.norm(N)
    # elif np.array_equal(normalized_approach_vector , np.array([0., 2., 2.])):
    #     print("second condition")
    #     time.sleep(5)
    #     N = np.array([1, 0, 0])  # Use a different axis if N is a zero vector
    elif np.array_equal(normalized_approach_vector , np.array([-0., -0., -1.])):
        # print("third condition")
        # time.sleep(5)
        N = np.array([-1, 0, 0])  # Use a different axis if N is a zero vector
    else:
        # print("none of the conditions")
        # time.sleep(5)
        N = np.array([-1, 0, 0])
    
    # Skew-symmetric matrix
    K = np.array([
        [0, -N[2], N[1]],
        [N[2], 0, -N[0]],
        [-N[1], N[0], 0]
    ])

    # Rodrigues' rotation formula
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Rotate vertices using the obtained rotation matrix
    rotated_vertices = np.dot(R, vertices.T).T

    # Calculate translation vector
    translation_vector = destination_point - rotated_vertices[0]

    # Translate the rotated vertices
    translated_vertices = rotated_vertices + translation_vector
    return translated_vertices


