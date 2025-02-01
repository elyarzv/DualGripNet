import numpy as np
import alphashape
from shapely.geometry import Point

def read_obj(filename):
    """Reads vertices from an OBJ file."""
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) > 0 and parts[0] == 'v':  # This line describes a vertex
                vertices.append(np.array(list(map(float, parts[1:4]))))
    return np.array(vertices)

def distance_from_line(line_point, line_vector, vertex):
    """Calculates the perpendicular distance from a vertex to a line."""
    point_to_vertex = vertex - line_point
    proj_length = np.dot(point_to_vertex, line_vector)
    proj_point = line_point + proj_length * line_vector
    return np.linalg.norm(proj_point - vertex), proj_point
    # return np.linalg.norm(proj_point - vertex) + 0.1 * np.linalg.norm(proj_point - line_point), proj_point

def find_closest_vertices(vertices, line_point, line_vector, num_vertices=3):
    """Finds the closest vertices to the line and returns their indices."""
    distances = []
    for index, vertex in enumerate(vertices):
        distance, point = distance_from_line(line_point, line_vector, vertex)
        distances.append((distance, index, vertex, point))
    # Sort by distance and return the closest num_vertices vertices
    distances.sort()
    in_triangle = point_in_triangle(distances[0], distances[1], distances[2])
    # breakpoint()
    if in_triangle == True:
        # print("************* in triangle")
        return [(idx, vert, point) for _, idx, vert, point in distances[:num_vertices]]
    else:
        # print("################## not in triangle")
        # return None
        return [(idx, vert, point) for _, idx, vert, point in distances[:num_vertices]]


def point_in_triangle(point1, point2, point3):
    # Helper function to compute cross product of vectors AB and AC
    def cross_product(xA, yA, xB, yB, xC, yC):
        ABx = xB - xA
        ABy = yB - yA
        ACx = xC - xA
        ACy = yC - yA
        return ABx * ACy - ABy * ACx

    # Check the sign of the cross product for vectors formed with the point and triangle vertices
    x, y, _ = point1[3]
    x1, y1, _ = point1[2]
    x2, y2, _ = point2[2]
    x3, y3, _ = point3[2]

    sign1 = cross_product(x1, y1, x2, y2, x, y)
    sign2 = cross_product(x2, y2, x3, y3, x, y)
    sign3 = cross_product(x3, y3, x1, y1, x, y)

    # The point is inside the triangle if the cross products are all positive or all negative
    is_same_sign = (sign1 >= 0 and sign2 >= 0 and sign3 >= 0) or (sign1 <= 0 and sign2 <= 0 and sign3 <= 0)

    # Optionally, you could handle the case where the point lies exactly on an edge
    # This would depend on the inclusion of zero in the comparisons (e.g., >= 0 or > 0 and <= 0 or < 0)
    
    return is_same_sign

def project(point, direction, vertices, alpha_shape):
    # Path to your OBJ file and line definition
    obj_filename = 'point_cloud.obj'
    direction /= np.linalg.norm(direction)  # Normalize the line direction vector

    # Find the three closest vertices to the line
    closest_vertices = find_closest_vertices(vertices, point, direction)

    if closest_vertices != None:
        proj_point = closest_vertices[0][2]
        test_point = Point(proj_point[0], proj_point[1])
        is_inside = alpha_shape.contains(test_point)
        if is_inside:
            # Print the closest vertices and their indices
            # print("The projected point on the surface is:", closest_vertices[0][2])
            return (closest_vertices[0][2])
        else:
            return None
    
if __name__ == "__main__":
    project()

