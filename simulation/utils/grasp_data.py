# Define a class to store grasp vertices with their neighbor indices, rms, and approaching vector
class GraspData:
    def __init__(self, center_index, neighbor_indices, plane_params, rms):
        self.center_index = center_index
        self.neighbor_indices = neighbor_indices
        self.plane_params = plane_params
        self.rms = rms