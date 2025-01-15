# Function to load OBJ file
# def load_obj(file_path):
#     vertices = []
#     faces = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             if line.startswith('v '):
#                 vertex = list(map(float, line[2:].split()))
#                 vertices.append(vertex)
#     return vertices

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.split()
            if not tokens:
                continue  # Skip empty lines

            if tokens[0] == 'v':
                # Vertex information
                vertex = list(map(float, tokens[1:]))
                vertices.append(vertex)
            elif tokens[0] == 'f':
                # Face information
                face = [int(vertex.split('/')[0]) for vertex in tokens[1:]]
                faces.append(face)

    return vertices, faces