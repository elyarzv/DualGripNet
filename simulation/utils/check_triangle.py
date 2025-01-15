import numpy as np
from utils.distance_function import distance_function

def check_line_triangle_intersection(point_on_line, line_direction, vertex1, vertex2, vertex3):
    try:
    
        # Define vectors for the triangle sides
        side1 = vertex2 - vertex1
        side2 = vertex3 - vertex1

        # Calculate the normal vector to the triangle
        normal_vector = np.cross(side1, side2)

        # Check if the line is parallel to the triangle
        dot_product = np.dot(normal_vector, line_direction)
        

        
        # if np.abs(dot_product) == 0:
        #     # Line is parallel to the triangle, check if it lies in the same plane
        #     if np.abs(np.dot(normal_vector, point_on_line - vertex1)) == 0:
        #         # Line lies in the same plane as the triangle, consider it intersects

        #         return True
        #     else:

        #         return False  # Line is parallel but not in the same plane, no intersection

        # Calculate the parameter t for the intersection point
        if dot_product != 0:
            t = np.dot(normal_vector, vertex1 - point_on_line) / dot_product
        # Calculate the intersection point
        intersection_point = point_on_line + t * line_direction

        # Check if the intersection point lies within the triangle
        u = np.dot(np.cross(intersection_point - vertex1, side2), normal_vector) / np.dot(normal_vector, normal_vector)

        v = np.dot(np.cross(side1, intersection_point - vertex1), normal_vector) / np.dot(normal_vector, normal_vector)


        # Check if the intersection point is inside the triangle (u, v, and u+v are between 0 and 1)
        if 0 <= u <=1 and 0 <= v <=1 and u+v <=1 :
            return True, intersection_point
        else:
            return False
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
    except TypeError:
        print("Error: Unsupported types for division.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
def find_intersecting_face(vertices, faces, point_on_line, line_direction):
    
    try:
        intersecting_faces = []

        for face in faces:
            vertex1 = np.array(vertices[face[0] - 1][:3])
            vertex2 = np.array(vertices[face[1] - 1][:3])
            vertex3 = np.array(vertices[face[2] - 1][:3])

            if check_line_triangle_intersection(point_on_line, line_direction, vertex1, vertex2, vertex3):
                intersecting_point = check_line_triangle_intersection(point_on_line, line_direction, vertex1, vertex2, vertex3)[1]
                intersecting_faces.append((face,intersecting_point))
        if len(intersecting_faces) > 1:
            dist = []
            for face in intersecting_faces:
                dist.append(distance_function(vertices[face[0][0]-1][:3], point_on_line)+distance_function(vertices[face[0][1]-1][:3], point_on_line)+distance_function(vertices[face[0][2]-1][:3], point_on_line))

            index_for_min = dist.index(min(dist))
            return [intersecting_faces[index_for_min]]
        elif len(intersecting_faces) == 1:
            return [intersecting_faces[0]]

    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
    except TypeError:
        print("Error: Unsupported types for division.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")