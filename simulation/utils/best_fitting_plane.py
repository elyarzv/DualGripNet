import numpy as np

def best_fitting_plane(center, neighbors, vertices):
    # Finding neighbor vertices' (x, y, z) of grasp center vertex
    neighbor_vertices = []
    for indices in neighbors:
        neighbor_vertices.append(vertices[indices][:3])

    # Center the points
    centroid = np.mean(neighbor_vertices, axis=0)
    centered_vertices = neighbor_vertices - centroid

    # Use PCA to find the normal vector of the best-fitting plane
    covariance_matrix = np.cov(centered_vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # The normal vector corresponds to the smallest eigenvalue
    normal_vector = eigenvectors[:, 0]

    # normalize the normal vector to have unit length
    normal_vector /= np.linalg.norm(normal_vector)

    # Check orientation of normal vector using dot product to make sure normal vector points outward from the object
    dot_product = normal_vector[0] * center[0] + normal_vector[1] * center[1] + normal_vector[2] * center[2]
    if dot_product < 0:
        normal_vector = - normal_vector

    A, B, C = normal_vector
    D = -np.dot(normal_vector, centroid)
    return A, B, C, D