

import numpy as np
import torch
from scipy.spatial.transform import Rotation as scRot

def RRE(R_gt,R_estim):
    '''
    R_gt: numpy array dim (3,3)
    R_estim: np array dim (3,3)
    Returns: angle measurement in degrees
    '''

    # tnp = np.matmul(R_estim.T,R_gt)
    tnp = np.matmul(np.linalg.inv(R_estim),R_gt)
    tnp = (np.trace(tnp) -1) /2
    tnp = np.clip(tnp, -1, 1)
    tnp = np.arccos(tnp) * (180/np.pi)
    return tnp

def RRE_batch(R_batch_estim,R_gt):
    '''
    R_batch_gt: numpy array dim (N,3,3)
    R_estim: np array dim (3,3)
    Returns: angle measurement in degrees
    '''

    tnp = np.matmul(np.transpose(R_batch_estim,(0,2,1)),R_gt)
    tnp = (np.trace(tnp,axis1=1,axis2=2) -1) /2
    tnp = np.clip(tnp, -1, 1)
    tnp = np.arccos(tnp) * (180/np.pi)

    return np.min(tnp), np.argmin(tnp)


def RRE_KITTI(R_gt, R_estim):
    '''
    Metric used by GeDi
    '''
    tnp = np.matmul(np.linalg.inv(R_estim),R_gt)
    sr = scRot.from_matrix(tnp)
    sr = sr.as_euler('xyz')
    return np.sum(np.abs(sr))

def RTE(t_gt,t_estim):
    '''
    t_gt: np array dim (3,)
    t_estim: np array dim (3,)
    '''

    return np.linalg.norm(t_gt - t_estim,ord=2)

def ADD(A,B):
    return np.mean(np.sqrt(np.sum((A - B)**2,axis=1)))

def euclid_dist(A,B):
    return torch.sqrt(torch.sum((A - B)**2, axis=1))

def euclid_dist_np(A,B):
    return np.sqrt(np.sum((A - B)**2, axis=1))

def eval_from_csv(data,thr_rot=15,thr_trans=0.3,M=1724,rre_col='RRE',printout=True):

    true_positives = (data['RTE'] < thr_trans) & (data[rre_col] < thr_rot)
    if printout: print(f'Registered {np.sum(true_positives)}/{data.shape[0]} examples.')
    rr = np.sum(true_positives)/data.shape[0]
    if printout: print(f'RR: {rr:.4f}')

    rre_mean = np.mean(data.loc[true_positives,rre_col])
    if printout: print(f'RRE: {rre_mean:.4f} degrees')

    rte_mean = np.mean(data.loc[true_positives,'RTE']) * 100
    if printout: print(f'RTE: {rte_mean:.4f} cm')

    #add_mean = np.mean(data['ADD']) * 100
    #print(f'ADD: {add_mean:.4f} cm')

    N = data.shape[0]
    if printout: print(f'Results obtained on {(N/M)*100}% of benchmark examples')

    return rr, rre_mean, rte_mean
