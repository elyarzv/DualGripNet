import numpy as np

def rms_distance(points, plane_params):
    # Extract plane parameters (A, B, C, D) from the input
    A, B, C, D = plane_params
    N = len(points)
    distances = 0
    for point in points:    
        # Calculate signed distances
        distances = distances + np.abs(A * point[0] + B * point[1] + C * point[2] + D) / np.sqrt(A**2 + B**2 + C**2)

    # Calculate the root mean square (RMS) distance
    rms = np.sqrt(np.sum(distances**2) / N)
    rms = rms * (10 ** 3)
    return rms