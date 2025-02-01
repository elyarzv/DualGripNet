from utils.distance_function import distance_function

# Function to find nearby vertices
def find_nearby_vertices(vertices, target_point, big_distance):
    nearby_vertices = [i for i, vertex in enumerate(vertices) if  distance_function(vertex[:3], target_point) < big_distance ]
    return nearby_vertices