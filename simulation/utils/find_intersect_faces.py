import numpy as np
from utils.check_triangle import check_line_triangle_intersection
from utils.distance_function import distance_function

def find_intersecting_faces(vertices, faces, point_on_line, line_direction):
    intersecting_faces = []

    for face in faces:
        vertex1 = np.array(vertices[face[0] - 1][:3])
        vertex2 = np.array(vertices[face[1] - 1][:3])
        vertex3 = np.array(vertices[face[2] - 1][:3])

        if check_line_triangle_intersection(point_on_line, line_direction, vertex1, vertex2, vertex3):
            intersecting_faces.append(face)
    
    if len(intersecting_faces) > 1:
        dist = []
        for face in intersecting_faces:
            dist.append(distance_function(vertices[face[0]-1][:3], point_on_line)+distance_function(vertices[face[1]-1][:3], point_on_line)+distance_function(vertices[face[2]-1][:3], point_on_line))

    print(dist)
    index_for_min = dist.index(min(dist))
    return [intersecting_faces[index_for_min]]