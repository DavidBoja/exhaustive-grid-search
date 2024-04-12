
import numpy as np
import open3d as o3d
import torch
# import custom_ext as _C



def check_angles(normals1,normals2,thr=np.pi/2):
    '''
    checks if the angle in radians between 3D 
    vectors normals1[i] and normals2[i] is greater than given threshold
    Input: normals1: numpy Nx3 array - representing normals of point cloud for scale1
           normals2: numpy Nx3 array - representing normals of point cloud for scale2
           thr: number - threshold to return condition if 
    Returns: mask: numpy (N,) array of True/False where mask[i] indicates
                   angle(normals1[i], normals2[i]) > thr
    '''

    # normalize 
    norm_normals1 = normals1 / np.linalg.norm(normals1,ord=2,axis=1).reshape(-1,1)
    norm_normals2 = normals2 / np.linalg.norm(normals2,ord=2,axis=1).reshape(-1,1)

    # row-wise dot product between normals
    # example
    # a=np.array([[1,2,3],[3,4,5]])
    # b=np.array([[1,2,3],[1,2,3]])
    # row_wise_dot_prod = np.sum(a*b, axis=1)
    row_wise_dot_prod = np.sum(norm_normals1*norm_normals2, axis=1)

    # find angles
    angles = np.arccos(np.clip(row_wise_dot_prod, -1.0,1.0))

    # threshold 
    return (angles > thr)

def get_normals_by_radius(pc,r):
    rad = o3d.geometry.KDTreeSearchParamRadius(radius=r)

    pc.estimate_normals(search_param = rad)
    normals = np.asarray(pc.normals).copy()

    return normals


def check_normal_ambiguity(small_normals,big_normals):
    na = check_angles(small_normals,big_normals)
    # if np.sum(na) == 0:
    #     print('No ambiguities.')
    # else:
    #     print('Ambiguities')
    
    return na