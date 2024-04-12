
from torch.utils.data import DataLoader, Dataset
import torch
from utils.subsampling import get_normals_by_radius
from utils.pc_utils import (voxelize_batch, voxelize, voxelize_batch_timed, 
                            voxelize_batchImportance, voxelize_batchImportanceAndLayering, 
                            voxelize_batchLayering, voxelize_batchRound, voxelizeImportance,
                            voxelizeLayeringPrecomputed)
from functools import partial
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from utils.utils import print_time_elapsed
import time

#####################################################################
# NO PREPROCESSING VERSION
# DATA IS GIVEN AS ALREADY ROTATED EXAMPLES- COLLATOR ONLY VOXELIZES BATCH

class CustomDataset(Dataset):
    '''
    Used to take rotated pcj in K rotations and give
    in batches to collator MyCollator
    '''
    
    def __init__(self, data):
        '''
        data: B x N x 3 torch Tensor of data
        '''
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        pts = self.data[idx]
        
        return pts

class MyCollator(object):
    
    def __init__(self, voxel_size, pp, dtype_of_tensors):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.dtype_of_tensors = dtype_of_tensors
        
    def __call__(self,batch):
        
        pts = torch.stack(batch,dim=0)
        
        d, _ = voxelize_batch(pts, self.voxel_size)
        print(f'Orig shape {d.shape}')
        d_padded = torch.nn.functional.pad(d.type(self.dtype_of_tensors), 
                                           self.pp, mode='constant', value=0)

        return d_padded.unsqueeze(1)

#####################################################################

#####################################################################
# PREPROCESSING BATCH VERSION
# preprocess pc in getitem -- voxelize in collator
# minimas are exposed because it doesent work when saving to self.minimas

class ThreeDMatchFullResBatch(Dataset):
    '''
    Used to give one rotated example (computed in real time)
    - when using full resolution scans its not feasible to use 
    CustomDataset because all the examples need to be 
    calculated at once which are enourmous tensors
    '''

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]


        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]
        # self.make_positive = torch.empty(self.K,3)
        self.layering_indices = None
        self.layering_indices_behind = None

        if not isinstance(voxelization_option, type(None)):
            if ('Layering' in voxelization_option) or ('ImportanceAndLayering' in voxelization_option):
                # find normals
                # create o3d object 
                pc_o3d = o3d.geometry.PointCloud()
                pc_o3d.points = o3d.utility.Vector3dVector(pts)

                # find normals
                small_normals = get_normals_by_radius(pc_o3d, 0.01)  #np arrays
                big_normals = get_normals_by_radius(pc_o3d, 0.03) # np arrays
                #ambiguities = check_normal_ambiguity(small_normals, big_normals)

                # find difference of normals
                don = (small_normals - big_normals) / 2
                norms = np.linalg.norm(don, ord=2, axis=1)

                # threshold to identify flat surfaces -- the ones with small norm of don
                thr = np.quantile(norms, 0.25)
                flat_surface_indices = np.where(norms < thr)[0]

                # new points defined by moving each point by its normal direction for voxel_size distance
                # in both directions of the normal
                new_pts = pts[flat_surface_indices].numpy() + voxel_size * big_normals[flat_surface_indices]
                new_pts_behind = pts[flat_surface_indices].numpy() - voxel_size * big_normals[flat_surface_indices]

                # if there are points in the neighborhood of the new defined points
                # remove them -- we want that point to be isolated in its own voxel
                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(pts.numpy())
                distances, _ = neigh.kneighbors(new_pts, return_distance=True)
                distances_behind, _ = neigh.kneighbors(new_pts_behind, return_distance=True)

                # remove close points
                # dist_thr = voxel_size
                choose_pts = np.where(distances > voxel_size)[0]
                new_pts = new_pts[choose_pts]

                choose_pts_behind = np.where(distances_behind > voxel_size)[0]
                new_pts_behind = new_pts_behind[choose_pts_behind]

                self.points_preprocessed = torch.vstack([self.pts,
                                                         torch.from_numpy(new_pts),
                                                         torch.from_numpy(new_pts_behind)])

                self.layering_indices = np.arange(self.pts.shape[0],
                                                  self.pts.shape[0]+new_pts.shape[0])
                self.layering_indices_behind = np.arange(self.pts.shape[0]+new_pts.shape[0],
                                                self.pts.shape[0]+new_pts.shape[0]+new_pts_behind.shape[0])

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        # returns rotated + centered
        return points, minima

class ThreeDMatchFullResCollatorV2(object):
    
    def __init__(self, voxel_size, pp, dtype_of_tensors, padding_constant=0, voxelization_option='1and0',
                 important_pts_indices=None, layering_indices=None, layering_indices_behind=None):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.dtype_of_tensors = dtype_of_tensors
        self.p_const = padding_constant
        if 'and' in voxelization_option:
            fill_positive, fill_negative = [float(x) for x in voxelization_option.split('and')]
            self.voxelization_function = partial(voxelize_batch, 
                                                 voxel_size=self.voxel_size,
                                                 fill_positive=fill_positive, 
                                                 fill_negative=fill_negative)

        elif 'ImportanceAndLayering' in voxelization_option:
            _, fill_positive, fill_intermediate, fill_layer, fill_negative = voxelization_option.split('_')
            fill_positive = float(fill_positive)
            fill_intermediate = float(fill_intermediate) 
            fill_layer = float(fill_layer)
            fill_negative = float(fill_negative)

            self.voxelization_function = partial(voxelize_batchImportanceAndLayering,
                                                     important_pts_indices = important_pts_indices,
                                                     voxel_size=self.voxel_size,
                                                     layering_indices=layering_indices,
                                                     layering_indices_behind=layering_indices_behind,
                                                     fill_positive=fill_positive, 
                                                     fill_intermediate=fill_intermediate,
                                                     fill_layer=fill_layer,
                                                     fill_negative=fill_negative
                                                    )

        elif 'Importance' in  voxelization_option:
                _, fill_positive, fill_intermediate, fill_negative = voxelization_option.split('_')
                fill_positive = float(fill_positive)
                fill_intermediate = float(fill_intermediate) 
                fill_negative = float(fill_negative)

                self.voxelization_function = partial(voxelize_batchImportance,
                                                     important_pts_indices = important_pts_indices,
                                                     voxel_size=self.voxel_size,
                                                     fill_positive=fill_positive, 
                                                     fill_intermediate=fill_intermediate,
                                                     fill_negative=fill_negative
                                                    )


        elif 'Layering' in voxelization_option:
            _, fill_positive, fill_layer, fill_negative = voxelization_option.split('_')
            fill_positive = float(fill_positive)
            fill_layer = float(fill_layer) 
            fill_negative = float(fill_negative)
            self.voxelization_function = partial(voxelize_batchLayering,
                                                    layering_indices = layering_indices,
                                                    layering_indices_behind= layering_indices_behind,
                                                    voxel_size=self.voxel_size,
                                                    fill_positive=fill_positive, 
                                                    fill_layer=fill_layer,
                                                    fill_negative=fill_negative
                                                    )

        elif 'Round' in voxelization_option:
            _, fill_positive, fill_negative = voxelization_option.split('_')
            fill_positive = float(fill_positive)
            fill_negative = float(fill_negative)
            self.voxelization_function = partial(voxelize_batchRound,
                                                voxel_size=self.voxel_size,
                                                fill_positive=fill_positive,
                                                fill_negative=fill_negative
                                                )
        else:
            RuntimeError('No such voxelization option!')
        
    def __call__(self,batch):
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        # d, _ = voxelize_batch(pts, self.voxel_size)
        # d, orig_shape = voxelize_batch(pts, self.voxel_size)
        d, orig_shape = self.voxelization_function(pts)
        d_padded = torch.nn.functional.pad(d.type(self.dtype_of_tensors), 
                                        self.pp, mode='constant', value=self.p_const)

        return d_padded.unsqueeze(1), minimas#, orig_shape


#####################################################################
# PREPROCESSING BATCH=1 VERSION -- FASTEST WHEN num_workers=0
# preprocess and voxelize pc in getitem 
# NO NEED FOR COLLATOR

class ThreeDMatchFinalSingle(Dataset):

    def __init__(self, pts, R_batch, pp, center=torch.zeros(3), 
                 voxel_size=0.06, 
                 fill_positive=5, fill_negative=-1, fill_padding=-1,
                 subsampling_indices=None):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

        #self.make_positive = torch.empty(self.K,3)
        # self.dtype_of_tensors = dtype_of_tensors
        self.pp = pp

        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding

    def __len__(self):
        return self.K

    def __getitem__(self, idx):

        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # voxelize
        d, orig_shape = voxelize(points, self.voxel_size, 
                                 self.fill_positive, self.fill_negative)

        # pad
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                           self.pp, 
                                           mode='constant', 
                                           value=self.fill_padding)
        
        return d_padded.unsqueeze(0), minima#, orig_shape


########################################################################################
# FINAL IMPLEMMENTATIONS FOR RUNNING FASTER
########################################################################################

class ThreeDMatchFinalBatch(Dataset):
    '''
    For final implmentation that should run faster without any ifs, fors ,...
    Used to give one rotated example (computed in real time)
    - when using full resolution scans its not feasible to use 
    CustomDataset because all the examples need to be 
    calculated at once which are enourmous tensors
    '''

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]


        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima

class ThreeDMatchFinalCollator(object):
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
    def __call__(self,batch):
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        voxelized_batch, orig_shape = voxelize_batch(pts, 
                                        self.voxel_size,
                                        self.fill_positive,
                                        self.fill_negative)

        voxelized_batch_padded = torch.nn.functional.pad(voxelized_batch.type(torch.int32), 
                                        self.pp, mode='constant', 
                                        value=self.fill_padding)
        # d_padded = d_padded.to(self.device)
        return voxelized_batch_padded.unsqueeze(1), minimas#, orig_shape


class ThreeDMatchFinalBatchTimed(Dataset):
    '''
    For final implmentation that should run faster without any ifs, fors ,...
    Used to give one rotated example (computed in real time)
    - when using full resolution scans its not feasible to use 
    CustomDataset because all the examples need to be 
    calculated at once which are enourmous tensors
    '''

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]


        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        dataloader_time = time.time()
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        dataloader_time = time.time() - dataloader_time

        # returns rotated + centered + "positived"
        return points, minima, dataloader_time


class ThreeDMatchFinalCollatorTimed(object):
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
    def __call__(self,batch):
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        time_stack = time.time()
        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)
        time_dataload = np.mean([b[2] for b in batch])
        time_stack = time.time()-time_stack

        d, orig_shape, time_vox = voxelize_batch_timed(pts, 
                                                self.voxel_size,
                                                self.fill_positive,
                                                self.fill_negative)

        time_pad = time.time()
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                        self.pp, mode='constant', 
                                        value=self.fill_padding)
        # d_padded = d_padded.to(self.device)
        time_pad = time.time() - time_pad

        return d_padded.unsqueeze(1), minimas, time_dataload, time_stack, time_vox, time_pad


########################################################################################
# FAUST
########################################################################################

class FAUSTBatch(Dataset):

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3)):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]


        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.clone().T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima

class FAUSTCollator(object):
    
    def __init__(self, voxel_size, pp):
        
        self.voxel_size = voxel_size
        self.pp = pp
        
    def __call__(self,batch):
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        d, orig_shape = voxelize_batch(pts, self.voxel_size,5,-1)
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                        self.pp, mode='constant', value=-1)

        return d_padded.unsqueeze(1), minimas#, orig_shape


########################################################################################
# ETH
########################################################################################

class ETHFinalBatch(Dataset):

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima

class ETHFinalCollatorTimed(object):
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1,timings_pth=None):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
        # self.timings_stack = []
        # self.timings_voxelize = []
        # self.timings_pad = []
        
    def __call__(self,batch):
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        time_stack = time.time()
        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)
        # self.timings_stack.append(time.time()-time_stack)
        time_stack = time.time()-time_stack

        
        d, orig_shape, time_vox = voxelize_batch_timed(pts, 
                                                self.voxel_size,
                                                self.fill_positive,
                                                self.fill_negative)
        # self.timings_voxelize += timings

        time_pad = time.time()
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                        self.pp, mode='constant', 
                                        value=self.fill_padding)
        # self.timings_pad.append(time.time()-time_pad)
        time_pad = time.time() - time_pad
        # d_padded = d_padded.to(self.device)
        return d_padded.unsqueeze(1), minimas, time_stack, time_vox, time_pad 

class ETHFinalCollator(object):
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        

        
    def __call__(self,batch):
       
 

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        
        d, orig_shape, = voxelize_batch(pts, 
                                        self.voxel_size,
                                        self.fill_positive,
                                        self.fill_negative)

        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                        self.pp, mode='constant', 
                                        value=self.fill_padding)
        
        return d_padded.unsqueeze(1), minimas

########################################################################################
# KITTI
########################################################################################

class KITTIFinalBatch(Dataset):

    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        # self.make_positive[idx] = torch.min(points,dim=0)[0]
        minima = torch.min(points,dim=0)[0]
        # points = points - self.make_positive[idx]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima

class KITTIFinalCollator(object):
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1,timings_pth=None):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
        # self.timings_stack = []
        # self.timings_voxelize = []
        # self.timings_pad = []
        
    def __call__(self,batch):

        # tttt = time.time()
       
        # NOTE be wary using this -- batch must have pc-s of same nr points
        # this is not a problem because its the same point cloud rotated
        # in many ways -- thats why stack can work!
        # minimas = torch.stack()

        # time_stack = time.time()
        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)
        # self.timings_stack.append(time.time()-time_stack)
        # time_stack = time.time()-time_stack

        
        # d, orig_shape, time_vox = voxelize_batch_timed(pts, 
        #                                         self.voxel_size,
        #                                         self.fill_positive,
        #                                         self.fill_negative)

        d, orig_shape = voxelize_batch(pts, 
                                        self.voxel_size,
                                        self.fill_positive,
                                        self.fill_negative)

        # self.timings_voxelize += timings


        # time_pad = time.time()
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                        self.pp, mode='constant', 
                                        value=self.fill_padding)
        # self.timings_pad.append(time.time()-time_pad)
        # time_pad = time.time() - time_pad
        # d_padded = d_padded.to(self.device)

        # print_time_elapsed(tttt, ' 1 COLLATOR CALL','green')
        return d_padded.unsqueeze(1), minimas#, time_stack, time_vox, time_pad 

class KITTIFinalB1(Dataset):

    def __init__(self, pts, R_batch, pp, center=torch.zeros(3), 
                 voxel_size=0.06, 
                 fill_positive=5, fill_negative=-1, fill_padding=-1,
                 subsampling_indices=None,device=None):
        '''
        data: N x 3 point cloud that will be rotated with
              R_Batch
        R_Batch: K x 3 x 3 tensor of K rotations
        '''

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

        self.minimas = torch.empty(self.K,3)
        # self.dtype_of_tensors = dtype_of_tensors
        self.pp = pp

        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding

        self.device = device

    def __len__(self):
        return self.K

    def __getitem__(self, idx):

        init_process_data = time.time()
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive
        minima = torch.min(points,dim=0)[0]
        self.minimas[idx] = minima
        points = points - minima

        # voxelize
        d, orig_shape = voxelize(points, self.voxel_size, 
                                 self.fill_positive, self.fill_negative)

        # pad
        d_padded = torch.nn.functional.pad(d.type(torch.int32), 
                                           self.pp, 
                                           mode='constant', 
                                           value=self.fill_padding)
        
        print_time_elapsed(init_process_data,'DATALOADER 1 EXAMPLE TIME')
        # return d_padded.unsqueeze(0), minima#, orig_shape
        return d_padded.unsqueeze(0)


########################################################################################
# PROCESSING DURING REGISTRATION
########################################################################################

class RotatePC(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,voxel_size=0.06):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]

        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # returns rotated + centered + "positived"
        return points, minima

class RotatePCcollator(object):
    """
    Collator that for a batch of points with same size, voxelizes and pads 
    them together -- __call__ returns torch tensor of  1 x N x Vx x Vy x Vz dims
    """
    
    def __init__(self, voxel_size, pp, fill_positive=5, fill_negative=-1,
                 fill_padding=-1):
        
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
    def __call__(self,batch):
       
        # NOTE -- batch must have pc-s of same nr points
        # we use the same point cloud rotated in multiple ways so we good

        minimas = torch.stack([b[1] for b in batch],dim=0)
        pts = torch.stack([b[0] for b in batch],dim=0)

        voxelized_batch, _ = voxelize_batch(pts, 
                                        self.voxel_size,
                                        self.fill_positive,
                                        self.fill_negative) # N x Vx x Vy x Vz

        voxelized_batch_padded = torch.nn.functional.pad(voxelized_batch.type(torch.int32), 
                                                        self.pp, mode='constant', 
                                                        value=self.fill_padding)

        return voxelized_batch_padded.unsqueeze(1), minimas

class RotatePCB1(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, voxel_size, pp, 
                 subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,
                 fill_positive=5, fill_negative=-1, fill_padding=-1):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # voxelize
        voxelized_pts, orig_shape = voxelize(points, 
                                             self.voxel_size, 
                                             self.fill_positive, 
                                             self.fill_negative) # Vx x Vy x Vz

        # pad
        voxelized_pts_padded = torch.nn.functional.pad(voxelized_pts.type(torch.int32), 
                                                        self.pp, 
                                                        mode='constant', 
                                                        value=self.fill_padding) # Vx x Vy x Vz
        
        return voxelized_pts_padded.unsqueeze(0), minima, orig_shape # 1 x Vx x Vy x Vz

class RotatePCB1Timed(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, voxel_size, pp, 
                 subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,
                 fill_positive=5, fill_negative=-1, fill_padding=-1):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        
        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

        # self.rotating_time_init = np.zeros((self.K))
        # self.rotating_time_finish = np.zeros((self.K))
        # self.make_positive_time = np.zeros((self.K))
        # self.voxelize_time_init = np.zeros((self.K))
        # self.voxelize_time_finish = np.zeros((self.K))
        # self.pad_time = np.zeros((self.K))

    def __len__(self):
        return self.K

    def __getitem__(self, idx):

        # rotate
        rot_time = time.time() # self.rotating_time_init[idx] = time.time()
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N
        rot_time = time.time() - rot_time # self.rotating_time_finish[idx] = time.time()

        # make positive by translating min bounding box point to origin
        make_positive_time = time.time()
        minima = torch.min(points,dim=0)[0]
        points = points - minima
        make_positive_time = time.time() - make_positive_time

        # voxelize
        voxelize_time = time.time()
        voxelized_pts, orig_shape = voxelize(points, 
                                             self.voxel_size, 
                                             self.fill_positive, 
                                             self.fill_negative) # Vx x Vy x Vz
        voxelize_time = time.time() - voxelize_time

        # pad
        pad_time = time.time()
        voxelized_pts_padded = torch.nn.functional.pad(voxelized_pts.type(torch.int32), 
                                                        self.pp, 
                                                        mode='constant', 
                                                        value=self.fill_padding) # Vx x Vy x Vz
        pad_time = time.time() - pad_time
        
        return voxelized_pts_padded.unsqueeze(0), minima, idx, rot_time, make_positive_time, voxelize_time, pad_time # 1 x Vx x Vy x Vz
                                  
class RotatePCB1Importance(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, voxel_size, pp, important_indices,
                 subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,
                 fill_positive=5, fill_intermediate=2, fill_negative=-1, fill_padding=-1
                 ):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_intermediate = fill_intermediate
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        self.important_indices = important_indices
        
        
        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # voxelize
        voxelized_pts, orig_shape = voxelizeImportance(points, 
                                                          self.important_indices, 
                                                          self.voxel_size, 
                                                          fill_positive=self.fill_positive, 
                                                          fill_intermediate=self.fill_intermediate, 
                                                          fill_negative=self.fill_negative)

        # pad
        voxelized_pts_padded = torch.nn.functional.pad(voxelized_pts.type(torch.int32), 
                                                        self.pp, 
                                                        mode='constant', 
                                                        value=self.fill_padding) # Vx x Vy x Vz
        
        return voxelized_pts_padded.unsqueeze(0), minima, idx # 1 x Vx x Vy x Vz
    
class RotatePCB1Layering(Dataset):
    """
    Rotates a given centered point cloud by a given batch of rotations. 
    __getitem__ first rotates the point cloud and then makes it positive by
    translating the minimal bounding box point to the origin.
    """
    
    def __init__(self, pts, R_batch, voxel_size, pp,
                 points_front_inds, points_behind_inds, normals,
                 subsampling_indices=None, center=torch.zeros(3),
                 voxelization_option=None,
                 fill_positive=5, fill_layer=2, fill_negative=-1, fill_padding=-1,
                #  radius_l=0.03
                 ):

        self.pts = pts
        self.R_batch = R_batch
        self.K = self.R_batch.shape[0]
        self.voxel_size = voxel_size
        self.pp = pp
        self.fill_positive = fill_positive
        self.fill_layer = fill_layer
        self.fill_negative = fill_negative
        self.fill_padding = fill_padding
        self.points_front_inds = points_front_inds
        self.points_behind_inds = points_behind_inds
        # self.radius_l = radius_l
        self.normals_preprocessed = normals
        
        
        self.center = torch.mean(self.pts,axis=0) - center
        self.points_preprocessed = self.pts - self.center
        if not isinstance(subsampling_indices, type(None)):
            self.points_preprocessed = self.points_preprocessed[subsampling_indices]
            self.normals_preprocessed = self.normals_preprocessed[subsampling_indices]

    def __len__(self):
        return self.K

    def __getitem__(self, idx):
       
        # rotate
        points = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.points_preprocessed.T).T # 3 X N
        
        normals = torch.matmul(self.R_batch[idx], #  3 x 3
                              self.normals_preprocessed.T).T # 3 X N

        # make positive by translating min bounding box point to origin
        minima = torch.min(points,dim=0)[0]
        points = points - minima

        # voxelize
        voxelized_pts, orig_shape = voxelizeLayeringPrecomputed(
                                            points=points, 
                                            voxel_size=self.voxel_size,
                                            points_front_inds=self.points_front_inds,
                                            points_behind_inds=self.points_behind_inds,
                                            fill_positive=self.fill_positive, 
                                            fill_negative=self.fill_negative,
                                            fill_layer=self.fill_layer, 
                                            # radius_l=self.radius_l
                                            normals=normals
                                            )

        # pad
        voxelized_pts_padded = torch.nn.functional.pad(voxelized_pts.type(torch.int32), 
                                                        self.pp, 
                                                        mode='constant', 
                                                        value=self.fill_padding) # Vx x Vy x Vz
        
        return voxelized_pts_padded.unsqueeze(0), minima, idx # 1 x Vx x Vy x Vz

def preprocess_pcj(pcj, R_batch, dataset_name, voxel_size, pp, batch_size, num_workers,
                    fill_positive,fill_negative,fill_padding):
    """
    Create a dataloader that loads batch_size batches of rotated, voxelized  and padded pcj 
    points for a given dataset. 
    
    The batches are voxelized and paded pcj rotated with a number of
    rotations from R_batch.
  
    Input:  pcj: (torch) Nx3 points 
            R_batch: (torch) NX3x3 rotations
            dataset_name: (str) dataset name
            voxel_size: (float) size of voxel side
            pp: (tuple) 6dim tuple for padding -- deterimend in padding.padding_options
            batch_size: (scalar) torch dataloader size of batch
            num_workers: (scalar) torch dataloader num workers
    Returns: my_data: (torch.Dataset) dataset that returns rotated pcj for given index of R_batch
             my_dataloader: (torch.DataLoader) datalodaer that loads batch_size batches of rotated, 
                                                voxelized  and padded pcj points for a given dataset. 
    """
    
    my_data = RotatePC(pcj, R_batch)
    my_data_collator = RotatePCcollator(voxel_size, pp,
                                        fill_positive=fill_positive, 
                                        fill_negative=fill_negative, 
                                        fill_padding=fill_padding)
    my_dataloader = DataLoader(my_data, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=my_data_collator,
                                num_workers=num_workers
                                    )


    # if dataset_name.upper() == '3DMATCH':

    #     my_data = ThreeDMatchFinalBatch(pcj, R_batch)
    #     my_data_collator = ThreeDMatchFinalCollator(voxel_size, pp)
    #     my_dataloader = DataLoader(my_data, 
    #                                batch_size=batch_size, 
    #                                shuffle=False, 
    #                                collate_fn=my_data_collator,
    #                                num_workers=num_workers
    #                                 )
        
    # elif dataset_name.upper() == 'KITTI':
    #     my_data = KITTIFinalBatch(pcj, R_batch)
    #     my_data_collator = KITTIFinalCollator(voxel_size, pp)
    #     my_dataloader = DataLoader(my_data, 
    #                                batch_size=batch_size, 
    #                                shuffle=False, 
    #                                collate_fn=my_data_collator,
    #                                num_workers=num_workers
    #                                 )

    # elif dataset_name.upper() == 'ETH':
    #     my_data = ETHFinalBatch(pcj, R_batch)
    #     my_data_collator = ETHFinalCollatorTimed(voxel_size, pp)
    #     my_dataloader = DataLoader(my_data, 
    #                                batch_size=batch_size, 
    #                                shuffle=False, 
    #                                collate_fn=my_data_collator,
    #                                num_workers=num_workers
    #                                 )


    # elif 'FP-' in dataset_name.upper():
    #     my_data = FAUSTBatch(pcj, R_batch)
    #     my_data_collator = FAUSTCollator(voxel_size, pp)
    #     my_dataloader = DataLoader(my_data, 
    #                                batch_size=batch_size, 
    #                                shuffle=False, 
    #                                collate_fn=my_data_collator,
    #                                num_workers=num_workers
    #                                 )
    # else:
    #     raise NotImplementedError('Cant load this dataset!')

    return my_data, my_dataloader

def preprocess_pcj_B1(pcj, R_batch, voxel_size, pp, num_workers,
                     fill_positive, fill_negative, fill_padding, **kwargs):
    """
    Create a dataloader that loads batch_size batches of rotated, voxelized  and padded pcj 
    points for a given dataset. 
    
    The batches are voxelized and paded pcj rotated with a number of
    rotations from R_batch.
  
    Input:  pcj: (torch) Nx3 points 
            R_batch: (torch) NX3x3 rotations
            dataset_name: (str) dataset name
            voxel_size: (float) size of voxel side
            pp: (tuple) 6dim tuple for padding -- deterimend in padding.padding_options
            batch_size: (scalar) torch dataloader size of batch
            num_workers: (scalar) torch dataloader num workers
            kwargs: voxelization_type: (str) -- chose from "normal" voxelization, importance voxelization or
                                                layering voxelization
    Returns: my_data: (torch.Dataset) dataset that returns rotated pcj for given index of R_batch
             my_dataloader: (torch.DataLoader) datalodaer that loads batch_size batches of rotated, 
                                                voxelized  and padded pcj points for a given dataset. 
    """


    if "voxelization_type" not in kwargs.keys():
        voxelization_type = "weighted"
    else:
        voxelization_type = kwargs["voxelization_type"].lower()



    if voxelization_type == "weighted":
        rot_pbb1_data = RotatePCB1(pcj, R_batch, voxel_size, pp,
                                fill_positive=fill_positive, 
                                fill_negative=fill_negative, 
                                fill_padding=fill_padding)
    elif voxelization_type == "importance":
        important_indices = kwargs["important_indices"]
        fill_intermediate = kwargs["fill_intermediate"]
        rot_pbb1_data = RotatePCB1Importance(pcj, R_batch, voxel_size, pp, 
                                        important_indices,
                                        fill_positive=fill_positive, 
                                        fill_intermediate=fill_intermediate,
                                        fill_negative=fill_negative, 
                                        fill_padding=fill_padding)
    elif voxelization_type == "layering":
        layering_inds = kwargs["layering_inds"]
        points_front_inds = layering_inds["front"]
        points_behind_inds = layering_inds["behind"]
        fill_layer = kwargs["fill_layer"]
        # radius_l = kwargs["radius_l"]
        normals = kwargs["normals"]

        rot_pbb1_data = RotatePCB1Layering(
                                pts=pcj, 
                                R_batch=R_batch, 
                                voxel_size=voxel_size, 
                                pp=pp, 
                                points_front_inds=points_front_inds,
                                points_behind_inds=points_behind_inds,
                                fill_positive=fill_positive, 
                                fill_layer=fill_layer,
                                fill_negative=fill_negative, 
                                fill_padding=fill_padding,
                                # radius_l=radius_l
                                normals=normals
                                )
    else:
        raise NotImplementedError("This voxelization is not implemented.")
  

    rot_pcb1_loader = DataLoader(rot_pbb1_data, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=num_workers
                                )

    return rot_pbb1_data, rot_pcb1_loader
            