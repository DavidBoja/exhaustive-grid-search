
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

import operator
import open3d as o3d

from utils.subsampling import get_normals_by_radius
import time



def unravel_index_pytorch(flat_index, shape): 
    flat_index = operator.index(flat_index) 
    res = [] 

    # Short-circuits on zero dim tensors 
    if shape == torch.Size([]): 
        return 0 

    for size in shape[::-1]: 
        res.append(flat_index % size) 
        flat_index = flat_index // size 

    if len(res) == 1: 
        return res[0] 

    return tuple(res[::-1])

####################################################################################
#       BATCH VOXELIZATON
####################################################################################

def voxelize_batch(points, voxel_size, fill_positive=1, fill_negative=0):
    """
    Voxelize multiple point clouds and batch the voxelized volumes.
    Because of batching, each point cloud needs to be voxelized into 
    same volume dimensions.
  
    Input:  points: (torch) BxNx3 points to voxelize 
            voxel_size: (int) scalar that determines size of one voxel
            fill_positive: (int) number put in place of filled voxels
            fill_negative: (int) number put in place of emtpy voxels
    Returns: voxels (torch): voxelized points of dim 
                            B?? x NR_VOXELS[0] x NR_VOXELS[1] x NR_VOXELS[2]
             NR_VOXELS: (torch) tensor of voxel dimensions, dim3
    """

    # max of input by ax
    # tt1 = time.time()
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    # print_time_elapsed(tt1,'max_ax_input','green')

    # tt2 = time.time()
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()
    # print_time_elapsed(tt2,'NR_VOXELS','green')

    # tt3 = time.time()
    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)
    # print_time_elapsed(tt3,'dims','green')

    # tt4 = time.time()
    voxels = torch.zeros(dims) + fill_negative
    # print_time_elapsed(tt4,'create voxels and fill negative','green')

    # tt5 = time.time()
    voxel_indices = torch.floor(points / voxel_size).type(torch.int16)
    # print_time_elapsed(tt5,'voxel_indices','green')

    # reshape tako da bude N x 4
    # tt6 = time.time()
    batch_index = torch.arange(B,dtype=torch.int16).repeat(N,1).transpose(0,1)
    # print_time_elapsed(tt6,'batch_index','green')

    # tt7 = time.time()
    # batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),
    #                                  voxel_indices.int()],dim=2)
    batch_voxel_indices = torch.cat([batch_index.unsqueeze(-1),
                                     voxel_indices],dim=2)
    # print_time_elapsed(tt7,'batch_voxel_indices','green')

    # tt8 = time.time()
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()
    # print_time_elapsed(tt8,'batch_voxel_indices','green')

    # tt9 = time.time()
    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive
    # print_time_elapsed(tt9,'voxels fill positive','green')

    return voxels, NR_VOXELS

def voxelize_batchRound(points, voxel_size, fill_positive=1, fill_negative=0):
    '''
    points: torch BxNx3 pts
    voxel_size: int
    '''

    # max of input by ax
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 2).type(torch.int64).tolist()

    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)

    voxels = torch.zeros(dims) + fill_negative
    voxel_indices = torch.round(points / voxel_size)


    # reshape tako da bude N x 4
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()



    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive
    
    return voxels, NR_VOXELS


def voxelize_batchImportance(points, important_pts_indices, voxel_size, fill_positive=1, 
                             fill_intermediate=0.5, fill_negative=0):
    '''
    points: torch BxNx3 pts

    Batch version of voxelizeImportance

    Put fill_postivie in voxels where important_points are located, 
    fill_intermediate on all other full voxels and fill_negative on empty voxels
    '''

    # max of input by ax
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()

    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)

    # fill empty voxels with negative value
    voxels = torch.zeros(dims) + fill_negative
    voxel_indices = torch.floor(points / voxel_size)


    # reshape tako da bude N x 4
    # fill full voxels with intermediate value
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()

    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_intermediate


    # update voxels where important points are located with fill_positive value
    # NOTE this works because all the points in the batch are actually the same 
    # just rotated, so the important_pts_indices is same for all batch examples
    important_points = points[:,important_pts_indices]
    voxel_indices = torch.floor(important_points / voxel_size)
    N = important_points.shape[1]
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()

    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive
    
    return voxels, NR_VOXELS


def voxelize_batchLayering(points, voxel_size, layering_indices, layering_indices_behind,  
                         fill_positive=1, fill_negative=0, fill_layer=-1):
    '''
    points: torch BxNx3 pts
    TODO Explain this mister.
    '''

    # max of input by ax
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()

    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)

    voxels = torch.zeros(dims) + fill_negative
    voxel_indices = torch.floor(points / voxel_size)


    # reshape tako da bude N x 4
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()



    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive

    # correct for layering indices
    layering_indices = np.hstack([layering_indices,layering_indices_behind])
    layer_points = points[:,layering_indices]
    batch_index = torch.arange(B).repeat(layer_points.shape[1],1).transpose(0,1)
    voxel_indices = torch.floor(layer_points / voxel_size)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()

    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_layer

    
    return voxels, NR_VOXELS


def voxelize_batchImportanceAndLayering(points, important_pts_indices, voxel_size, 
                                        layering_indices, layering_indices_behind,  
                                        fill_positive=1, fill_negative=0, 
                                        fill_layer=-1, fill_intermediate=0.5):

    # max of input by ax
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()

    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)

    voxels = torch.zeros(dims) + fill_negative
    voxel_indices = torch.floor(points / voxel_size)


    # reshape tako da bude N x 4
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()



    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_intermediate

    # correct for layering indices
    layering_indices = np.hstack([layering_indices,layering_indices_behind])
    layer_points = points[:,layering_indices]
    voxel_indices = torch.floor(layer_points / voxel_size)
    N = layer_points.shape[1]
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()

    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_layer

    # update voxels where important points are located with fill_positive value
    important_points = points[:,important_pts_indices]
    voxel_indices = torch.floor(important_points / voxel_size)
    N = important_points.shape[1]
    batch_index = torch.arange(B).repeat(N,1).transpose(0,1)
    batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),voxel_indices.int()],dim=2)
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()

    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive

    
    return voxels, NR_VOXELS
    

def voxelize_batch_timed(points, voxel_size, fill_positive=1, fill_negative=0):
    '''
    points: torch BxNx3 pts
    voxel_size: int
    '''
    timings = []

    # max of input by ax
    tt1 = time.time()
    max_ax_input = torch.max(torch.max(points,dim=1)[0],dim=0)[0]
    timings.append(time.time()-tt1) # max ax input time
    # print_time_elapsed(tt1,'max_ax_input','green')

    tt2 = time.time()
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64).tolist()
    timings.append(time.time()-tt2) # get nr voxels time
    # print_time_elapsed(tt2,'NR_VOXELS','green')

    B = points.shape[0]
    N = points.shape[1]
    dims = tuple([B] + NR_VOXELS)

    tt4 = time.time()
    voxels = torch.zeros(dims) + fill_negative # create voxels and fill negative time
    timings.append(time.time()-tt4)
    # print_time_elapsed(tt4,'create voxels and fill negative','green')

    tt5 = time.time()
    voxel_indices = torch.floor(points / voxel_size).type(torch.int16)
    timings.append(time.time()-tt5) # get voxels indices time
    # print_time_elapsed(tt5,'voxel_indices','green')

    # reshape tako da bude N x 4
    tt6 = time.time()
    batch_index = torch.arange(B,dtype=torch.int16).repeat(N,1).transpose(0,1)
    timings.append(time.time()-tt6) # create indices time
    # print_time_elapsed(tt6,'batch_index','green')

    tt7 = time.time()
    # batch_voxel_indices = torch.cat([batch_index[:,:,None].int(),
    #                                  voxel_indices.int()],dim=2)
    batch_voxel_indices = torch.cat([batch_index.unsqueeze(-1),
                                     voxel_indices],dim=2) # batch indices time
    timings.append(time.time()-tt7)
    # print_time_elapsed(tt7,'batch_voxel_indices','green')

    tt8 = time.time()
    batch_voxel_indices = batch_voxel_indices.reshape(-1,4).long()
    timings.append(time.time()-tt8) # reshape batch indices time
    # print_time_elapsed(tt8,'batch_voxel_indices','green')

    tt9 = time.time()
    voxels[batch_voxel_indices[:,0],
           batch_voxel_indices[:,1],
           batch_voxel_indices[:,2],
           batch_voxel_indices[:,3]] = fill_positive # fill positive time
    timings.append(time.time()-tt9)
    # print_time_elapsed(tt9,'voxels fill positive','green')

    return voxels, NR_VOXELS, timings

####################################################################################
#               SINGLE VOXELIZATION
####################################################################################

def voxelize(points, voxel_size, fill_positive=1, fill_negative=0):
    """
    Voxelize points to voxel_size.
  
    Input:  points: (torch) Nx3 points to voxelize 
            voxel_size: (int) scalar that determines size of one voxel
            fill_positive: (int) number put in place of filled voxels
            fill_negative: (int) number put in place of emtpy voxels
    Returns: voxels (torch): voxelized points of dim 
                            NR_VOXELS[0] x NR_VOXELS[1] x NR_VOXELS[2]
             NR_VOXELS: (torch) tensor of voxel dimensions, dim3
    """

    # max of input by ax
    max_ax_input = torch.max(points,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive
    
    return voxels, NR_VOXELS

def voxelizeLayeringPrecomputed(points, voxel_size, 
                                points_front_inds, points_behind_inds, normals,
                                fill_positive=1, fill_negative=0, fill_layer=-1
                                # radius_l=0.03
                                ):
    
    # create o3d object 
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.numpy())

    # find normals
    # big_normals = get_normals_by_radius(pcd, radius_l) # np arrays

    # new points defined by moving each point by its normal direction for voxel_size distance
    # in both directions of the normal
    points_front   = points[points_front_inds] + voxel_size * normals[points_front_inds]
    points_behind  = points[points_behind_inds]  - voxel_size * normals[points_behind_inds]

    all_pts = torch.vstack([points,
                            points_front,
                            points_behind])

    # voxelize normally the points
    max_ax_input = torch.max(all_pts,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive


    # re-voxelize by changing the value of voxels of the layer points
    voxel_indices = torch.floor(points_front / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer

    voxel_indices = torch.floor(points_behind / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer
    
    return voxels, NR_VOXELS
    

def voxelizeLayering(points, voxel_size, fill_positive=1, fill_negative=0, fill_layer=-1,
                     radius_s=0.01, radius_l=0.03, quantile_thr=0.25):
    '''
    Put fill_positive in voxels that have points, fill_negative in voxels without points
    and fill_layer in neighboring voxels
    '''

    # create o3d object 
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(points)

    # find normals
    small_normals = get_normals_by_radius(pc_o3d, radius_s)  #np arrays
    big_normals = get_normals_by_radius(pc_o3d, radius_l) # np arrays
    #ambiguities = check_normal_ambiguity(small_normals, big_normals)

    # find difference of normals
    don = (small_normals - big_normals) / 2
    norms = np.linalg.norm(don, ord=2, axis=1)

    # threshold to identify flat surfaces -- the ones with small norm of don
    thr = np.quantile(norms, quantile_thr)
    flat_surface_indices = np.where(norms < thr)[0]

    # new points defined by moving each point by its normal direction for voxel_size distance
    # in both directions of the normal
    new_pts = points[flat_surface_indices].numpy() + voxel_size * big_normals[flat_surface_indices]
    new_pts_behind = points[flat_surface_indices].numpy() - voxel_size * big_normals[flat_surface_indices]

    # if there are points in the neighborhood of the new defined points
    # remove them -- we want that point to be isolated in its own voxel
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points.numpy())
    distances, _ = neigh.kneighbors(new_pts, return_distance=True)
    distances_behind, _ = neigh.kneighbors(new_pts_behind, return_distance=True)

    # remove close points
    # dist_thr = voxel_size
    choose_pts = np.where(distances > voxel_size)[0]
    new_pts = new_pts[choose_pts]

    choose_pts_behind = np.where(distances_behind > voxel_size)[0]
    new_pts_behind = new_pts_behind[choose_pts_behind]

    all_pts = torch.vstack([points,
                            torch.from_numpy(new_pts),
                            torch.from_numpy(new_pts_behind)])

    # voxelize normally the points
    max_ax_input = torch.max(all_pts,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive


    # re-voxelize by changing the value of voxels of the layer points
    voxel_indices = torch.floor(torch.from_numpy(new_pts) / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer

    voxel_indices = torch.floor(torch.from_numpy(new_pts_behind) / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer
    
    return voxels, NR_VOXELS


def voxelizeImportance(points, important_pts_indices, voxel_size, fill_positive=1, 
                       fill_intermediate=0.5, fill_negative=0):
    '''
    Put fill_postivie in voxels where important_points are located, 
    fill_intermediate on all other full voxels and fill_negative on empty voxels
    '''

    # max of input by ax
    max_ax_input = torch.max(points,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    # fill all voxels with negative value
    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative

    # fill all full voxels with intermediate value
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_intermediate

    # update some full voxels to positive value
    important_points = points[important_pts_indices]
    voxel_indices = torch.floor(important_points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive
    
    return voxels, NR_VOXELS


def voxelizeImportanceAndLayering(points, important_pts_indices, voxel_size, fill_positive=1, fill_negative=0, fill_layer=-1,
                                  fill_intermediate=0.5, radius_s=0.01, radius_l=0.03, quantile_thr=0.25):

    # create o3d object 
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(points)

    # find normals
    small_normals = get_normals_by_radius(pc_o3d, radius_s)  #np arrays
    big_normals = get_normals_by_radius(pc_o3d, radius_l) # np arrays
    #ambiguities = check_normal_ambiguity(small_normals, big_normals)

    # find difference of normals
    don = (small_normals - big_normals) / 2
    norms = np.linalg.norm(don, ord=2, axis=1)

    # threshold to identify flat surfaces -- the ones with small norm of don
    thr = np.quantile(norms, quantile_thr)
    flat_surface_indices = np.where(norms < thr)[0]

    # new points defined by moving each point by its normal direction for voxel_size distance
    # in both directions of the normal
    new_pts = points[flat_surface_indices].numpy() + voxel_size * big_normals[flat_surface_indices]
    new_pts_behind = points[flat_surface_indices].numpy() - voxel_size * big_normals[flat_surface_indices]

    # if there are points in the neighborhood of the new defined points
    # remove them -- we want that point to be isolated in its own voxel
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points.numpy())
    distances, _ = neigh.kneighbors(new_pts, return_distance=True)
    distances_behind, _ = neigh.kneighbors(new_pts_behind, return_distance=True)

    # remove close points
    # dist_thr = voxel_size
    choose_pts = np.where(distances > voxel_size)[0]
    new_pts = new_pts[choose_pts]

    choose_pts_behind = np.where(distances_behind > voxel_size)[0]
    new_pts_behind = new_pts_behind[choose_pts_behind]

    all_pts = torch.vstack([points,
                            torch.from_numpy(new_pts),
                            torch.from_numpy(new_pts_behind)])

    # voxelize normally the points
    max_ax_input = torch.max(all_pts,dim=0)[0]
    NR_VOXELS = (torch.floor(max_ax_input/voxel_size) + 1).type(torch.int64)

    voxels = torch.zeros(tuple(NR_VOXELS.tolist())) + fill_negative
    voxel_indices = torch.floor(points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_intermediate


    # re-voxelize by changing the value of voxels of the layer points
    voxel_indices = torch.floor(torch.from_numpy(new_pts) / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer

    voxel_indices = torch.floor(torch.from_numpy(new_pts_behind) / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_layer

    
    # update some full voxels to positive value
    important_points = points[important_pts_indices]
    voxel_indices = torch.floor(important_points / voxel_size).long()
    voxels[voxel_indices[:,0],
           voxel_indices[:,1],
           voxel_indices[:,2]] = fill_positive
    
    return voxels, NR_VOXELS


def get_voxel_overlap_for_same_padding(input_voxel_shape, template_voxel_shape, 
                                       central_voxel_templ):
    '''
    Calc real overlap of the output when doing 
    Input: input_voxel_shape: Torch Size tensor -- get by input_voxels.shape
            templtae_voxel_shape: Torch Size tnesor --get by template_voxel.shape
            central_voxel_templ: the central voxel of the template

    Returns: overlap when doing cross correlation with template over input


    Example:
    input_voxel = torch.Size(5,5,5)
    template_voxel = torch.Size(3,3,3)
    central_voxel_templ = tensor([1, 1, 1])
    should return 
    tensor([[[ 8., 12., 12., 12.,  8.],
         [12., 18., 18., 18., 12.],
         [12., 18., 18., 18., 12.],
         [12., 18., 18., 18., 12.],
         [ 8., 12., 12., 12.,  8.]],

        [[12., 18., 18., 18., 12.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [12., 18., 18., 18., 12.]],

        [[12., 18., 18., 18., 12.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [12., 18., 18., 18., 12.]],

        [[12., 18., 18., 18., 12.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [18., 27., 27., 27., 18.],
         [12., 18., 18., 18., 12.]],

        [[ 8., 12., 12., 12.,  8.],
         [12., 18., 18., 18., 12.],
         [12., 18., 18., 18., 12.],
         [12., 18., 18., 18., 12.],
         [ 8., 12., 12., 12.,  8.]]])

    '''

    Ix, Iy, Iz = input_voxel_shape
    Tx, Ty, Tz = template_voxel_shape
    Cx, Cy, Cz = central_voxel_templ.tolist()

    #NR_VOXELS_TEMPL = torch.Tensor([Tx,Ty,Tz])

    # all i,j,k coordinate pairs of the input voxel
    i_coord, j_coord, k_coord = torch.meshgrid(torch.arange(Ix), 
                                                torch.arange(Iy),
                                                torch.arange(Iz))

    # put the in a long vector
    i_coord_raveled = i_coord.ravel()
    j_coord_raveled = j_coord.ravel()
    k_coord_raveled = k_coord.ravel()

    N = i_coord_raveled.shape[0]

    # seems like the most innefficient method
    # to put all these in a list
    # but somehow its faster
    # at least according to %timeit in jupyter
    # its because later I use torch.minimum
    # which wants a vector cant process an int
    # torch.where was also slower
    Ix_vector = torch.Tensor([Ix]*N)
    Iy_vector = torch.Tensor([Iy]*N)
    Iz_vector = torch.Tensor([Iz]*N)

    Cx_vector = torch.Tensor([Cx]*N)
    Cy_vector = torch.Tensor([Cy]*N)
    Cz_vector = torch.Tensor([Cz]*N)

    Tx_vector = torch.Tensor([Tx]*N)
    Ty_vector = torch.Tensor([Ty]*N)
    Tz_vector = torch.Tensor([Tz]*N)

    # find how many points to left-right, 
    # above-below, and in front- backwards of every
    # index i,j,k
    # it is the minimum of the template and input
    # in the dimensions x,y,z when set at i,j,k
    Ix_minus_i = Ix_vector - i_coord_raveled
    Tx_minus_Cx = Tx_vector - Cx_vector

    Iy_minus_j = Iy_vector - j_coord_raveled
    Ty_minus_Cy = Ty_vector - Cy_vector

    Iz_minus_k = Iz_vector - k_coord_raveled
    Tz_minus_Cz = Tz_vector - Cz_vector

    minx1 = torch.minimum(Ix_minus_i -1, Tx_minus_Cx-1)
    minx2 = torch.minimum(i_coord_raveled, Cx_vector)
    range_x = (minx1 + minx2 + 1)

    miny1 = torch.minimum(Iy_minus_j-1, Ty_minus_Cy-1)
    miny2 = torch.minimum(j_coord_raveled, Cy_vector)
    range_y = (miny1 + miny2 + 1)

    minz1 = torch.minimum(Iz_minus_k-1, Tz_minus_Cz-1)
    minz2 = torch.minimum(k_coord_raveled, Cz_vector)
    range_z = (minz1 + minz2 + 1)


    overlap = torch.mul(torch.mul(range_x,range_y),range_z).reshape(Ix,Iy,Iz)

    return overlap


