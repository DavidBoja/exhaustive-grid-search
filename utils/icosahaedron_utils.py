
import numpy as np

def create_ico():
    r = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1.0,   r, 0.0],
        [ 1.0,   r, 0.0],
        [-1.0,  -r, 0.0],
        [ 1.0,  -r, 0.0],
        [0.0, -1.0,   r],
        [0.0,  1.0,   r],
        [0.0, -1.0,  -r],
        [0.0,  1.0,  -r],
        [  r, 0.0, -1.0],
        [  r, 0.0,  1.0],
        [ -r, 0.0, -1.0],
        [ -r, 0.0,  1.0],
        ], dtype=float)

    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [5, 4, 9],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
        ])

    return vertices, faces

def scale_ico(vertices,scale=1):
    '''
    scale icosahaedron so its vertices 
    lie on a sphere of radius scale
    input: vertices: numpy array N x 3
    return scaled_vertices: numpy array N x 3
    '''

    vertices_scaled = vertices.copy()
    vertices_scaled = vertices_scaled / np.linalg.norm(vertices_scaled,axis=1).reshape(-1,1) * scale
    return vertices_scaled


def split_middle_point(vertices,index1,index2):
    '''
    splits vertices[index1] and vertices[index2] with middle point
    and projects the point to sphere of radius 1
    
    adds index of point in middle_point_cache
    adds point into vertices
    '''
    
    # split point
    v1 = vertices[index1,:]
    v2 = vertices[index2,:]
    
    middle_point = (v1+v2)/2
    # project point to sphere radius 1
    l = np.sqrt(np.sum(middle_point **2))
    middle_point_projected = middle_point / l
    vertices = np.vstack([vertices,middle_point_projected])

    index = len(vertices) - 1

    return vertices, index
    

def split_icosahaedron(vertices,faces):
    '''
    From each face, create 4 new faces
    by splitting each edge and creating an additional point
    The triangle then looks like this, with 3 new additiaonl points
    denoted as new_index0, new_index1, new_index2
                    triangle_index2
                /                     \
        new_index1                 new_index2
            /                             \
    triangle_index0------new_index0------traignel_index1

    4 new traignels are created by replacing the face
    (triangle_index0,triangle_index1,triangle_index2)
    with 4 faces:
    (triangle_index0,new_index0,new_index1)
    (new_index0,traignel_index1,new_index2)
    (new_index1,new_index2,triangle_index2)
    (new_index1,new_index0,new_index2)

    return: vertices: numpy array Nx3 with added points from splited edges
            faces: numpy array Nx3 with new faces added as explained above
    '''
    
    middle_points_created = {}
    new_faces = []

    for i,triangle in enumerate(faces):
        triangle_index0 = triangle[0]
        triangle_index1 = triangle[1]
        triangle_index2 = triangle[2]

        new_indices = []

        for index1, index2 in [(triangle_index0,triangle_index1),
                               (triangle_index0,triangle_index2),
                               (triangle_index1,triangle_index2)]:
            # check if edge already splitted
            smaller_index = min(index1,index2)
            greater_index = max(index1,index2)

            key = f'{smaller_index}-{greater_index}'
            if key in middle_points_created.keys():
                new_index = middle_points_created[key]
                new_indices.append(new_index)
            else:
                # split edge
                vertices, new_index = split_middle_point(vertices, 
                                                        index1, 
                                                        index2)
                middle_points_created[key] = new_index
                new_indices.append(new_index)
        
        
        # add newly created vertices to faces
        new_faces.append([triangle_index0, new_indices[0], new_indices[1]]) 
        new_faces.append([new_indices[0], triangle_index1, new_indices[2]]) 
        new_faces.append([new_indices[1], new_indices[2], triangle_index2]) 
        new_faces.append([new_indices[1], new_indices[0], new_indices[2]]) 
        
    
    return vertices, np.array(new_faces)