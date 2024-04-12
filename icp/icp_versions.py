
import open3d as o3d
import time
from datetime import timedelta
import numpy as np

class ICP:
    '''
    Run generalized, p2point or p2plane icp.
    '''

    def __init__(self,version_choice, max_iter, quantile_distance):
        '''
        Run generalized, p2point or p2plane icp.
        Input:  version_choice: type of icp -- can be generalized, p2point or p2plane
                max_iter: scalar -- number of maximum iterations of icp
                quantile_distance: scalar in [0,1] range -- the quantile of all the
                                    distances on the source pc to threshold the
                                    inlier ratio
        '''

        choices = {"generalized": self.generalized,
                   "p2point":self.p2point,
                   "p2plane":self.p2plane}
        self.choice = choices[version_choice]

        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        self.quantile_distance = quantile_distance

        self.name = version_choice
        
    def run_icp(self,src,tgt,dist_thr,init_transform=np.eye(4)):
        '''
        Input:  src: open3d.PointCloud -- source point cloud
                tgt: open3d.PointCloud -- target point cloud
                dist_thr: scalar -- distance threshold for inlier rate
                init_transform: np.array -- 4x4 initial transformation
        '''
        runtime, new_T_estim = self.choice(src,tgt,dist_thr,init_transform)
        return runtime, new_T_estim 

    def generalized(self,src,tgt,dist_thr,init_transform):
            
        runtime = time.time() 
        reg_o3d = o3d.pipelines.registration.registration_generalized_icp(
                            src,tgt,
                            dist_thr,
                            init_transform,
                            criteria=self.criteria
                            )   
        runtime = timedelta(seconds=time.time()-runtime)
        new_T_estim = reg_o3d.transformation

        return runtime, new_T_estim
    
    def p2point(self,src,tgt,dist_thr,init_transform):

        runtime = time.time() 
        reg_o3d = o3d.pipelines.registration.registration_icp(
                            src, tgt, 
                            max_correspondence_distance=dist_thr, 
                            init=init_transform,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            criteria=self.criteria
                            )
    
        runtime = timedelta(seconds=time.time()-runtime)
        new_T_estim = reg_o3d.transformation

        return runtime, new_T_estim
    
    def p2plane(self,src,tgt,dist_thr,init_transform):

        src.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        tgt.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
        runtime = time.time() 
        reg_o3d = o3d.pipelines.registration.registration_icp(
                            src, tgt, 
                            max_correspondence_distance=dist_thr, 
                            init=init_transform,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                            criteria=self.criteria
                            )
    
        runtime = timedelta(seconds=time.time()-runtime)
        new_T_estim = reg_o3d.transformation

        return runtime, new_T_estim