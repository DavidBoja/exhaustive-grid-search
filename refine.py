import os
import os.path as osp
import argparse
import json
import yaml
import time
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from datetime import timedelta
import pandas as pd
import pickle

from utils.load_input import load_dataset, load_point_clouds, sort_eval_pairs, process_log
from utils.rot_utils import homo_matmul
from utils.utils import prepare_for_saving_results
from utils.eval_utils import RRE, RTE
from icp.icp_versions import ICP


def register_icp(options):

    DATASET_NAME = options['DATASET-NAME']
    CONTINUE_RUN = options['CONTINUE-RUN']
    CONTINUE_RUN_PATH = options['CONTINUE-RUN-PATH']
    RESULTS_PATH = options['RESULTS-PATH']
    QUANTILE_THR = options['MAX-CORRESPONDENCE-DISTANCE-QUANTILE']
    MAX_ITER = options['MAX-ITERATION']
    ICP_VERSION = options['ICP-VERSION']
    SUBSAMPLING_INDS_PATH = options['SUBSAMPLING-INDS-PATH']
    SUBSAMPLE = False
    if not isinstance(SUBSAMPLING_INDS_PATH, type(None)):
        if not os.path.exists(SUBSAMPLING_INDS_PATH):
            raise ValueError(f"Path {SUBSAMPLING_INDS_PATH} does not exist.")
        SUBSAMPLE = True

    # set icp
    icp = ICP(version_choice=ICP_VERSION,
              max_iter=MAX_ITER,
              quantile_distance=QUANTILE_THR)


    # create results paths and files
    now = CONTINUE_RUN_PATH.split("/")[1] if CONTINUE_RUN else time.strftime('%y%m%d%H%M')

    cols = ['folder',
            'examples',
            'time',
            'RRE',
            'RTE']
    results_vars = prepare_for_saving_results(now,CONTINUE_RUN,options,cols)
    results_folder_path, results_df_path, result_columns, results_df = results_vars

    # load inputs
    data_dict, folder_names = load_dataset(options)
    if SUBSAMPLE: subsample_dict = pickle.load(open(SUBSAMPLING_INDS_PATH,"rb"))

    for fname in folder_names:

        print(f'{icp.name} for {fname}')

        N_point_clouds_folder = data_dict[fname]['N']  
        full_data_path = data_dict[fname]['full_data_path']
        
        eval_pairs = list(data_dict[fname]['eval'].keys()) 
        eval_pairs = sort_eval_pairs(eval_pairs, DATASET_NAME)
        
        T_GT = data_dict[fname]['eval']
        name = fname.split('.ply')[0] # special case for faust-partial that iterates over examples
        T_ESTIM_BASELINE = process_log(osp.join(RESULTS_PATH,f'{name}.log'))

        log_path = osp.join(results_folder_path,f'{name}.log')  

        for ep in tqdm(eval_pairs):

            if CONTINUE_RUN:
                if results_df[(results_df['folder'] == fname) & 
                              (results_df['examples'] ==ep)].shape[0]>0:
                    continue

            # init_time = time.time()
            # pci is target if following paper
            # pcj is source if following paper
            # solve rotation on source
            # solve translation on target
            # goal is to register pcj onto pci

            ind_i, ind_j = ep.split(' ')
            
            pci, pcj = load_point_clouds(ind_i, 
                                         ind_j, 
                                         DATASET_NAME, 
                                         full_data_path,
                                         fname,
                                         data_dict)
            
            if SUBSAMPLE:
                pci.points = o3d.utility.Vector3dVector(np.asarray(pci.points)[subsample_dict[fname][ind_i],:])
                pcj.points = o3d.utility.Vector3dVector(np.asarray(pcj.points)[subsample_dict[fname][ind_j],:])
            
            # source is pcj
            # target is pci   
            pci_np = np.asarray(pci.points).copy()
            pcj_np = np.asarray(pcj.points).copy()    
            pcj_np_estim = homo_matmul(pcj_np,T_ESTIM_BASELINE[ep])
            
            # find distance threshold
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(pci_np)
            dist, _ = neigh.kneighbors(pcj_np_estim)
            adaptive_thr = np.quantile(dist,QUANTILE_THR) # threshold so QUANTILE_THR% pts in
            
            runtime, new_T_estim = icp.run_icp(pcj,pci,adaptive_thr,T_ESTIM_BASELINE[ep])

            #### EVAL ###################################################################
            R_est = new_T_estim[:3,:3]
            t_est = new_T_estim[:3,3]
            T_gt = T_GT[ep]
            R_gt = T_gt[:3,:3]
            t_gt = T_gt[:3,3]
            rre = RRE(R_gt,R_est)
            rte = RTE(t_gt,t_est)

            #### SAVE RESULTS  ##########################################################   
            # save transformation into log
            with open(log_path,'a+') as f:
                f.write('{} \t {}\t {}\n'.format(ind_i, ind_j, N_point_clouds_folder))
                f.write(str(new_T_estim.tolist()).replace('], [','\n').replace(',','\t')[2:-2])
                f.write('\n')          

            # save evaluation results
            current_results = pd.Series([fname,
                                        ep,
                                        runtime.__str__(),
                                        rre,
                                        rte
                                        ],
                                        index=result_columns)
            results_df = pd.concat([results_df,current_results.to_frame().T],
                                   ignore_index=True)
            results_df.to_csv(results_df_path, index=False)



if __name__ == '__main__':

    # choose which results to evaluate
    possible_results_folder_names = os.listdir('results')
    possible_results_folder_names = [osp.join('results',x) for x in possible_results_folder_names]

    parser = argparse.ArgumentParser()
    parser.add_argument("-R","--results_folder_path", 
                        required=True,
                        type=str, 
                        choices=possible_results_folder_names,
                        help='Path to results folder')
    parser.add_argument("-C","--continue_run_folder_path", 
                        type=str, 
                        default=None,
                        choices=possible_results_folder_names,
                        help='Path to results where running gen-icp stopped.')
    args = parser.parse_args()

    # parse choice and load existing config file
    results_path = args.results_folder_path
    f = open (osp.join(results_path,'options.json'), "r")
    options = json.loads(f.read())
    options['RESULTS-PATH'] = results_path

    options['CONTINUE-RUN'] = True if args.continue_run_folder_path else False
    options['CONTINUE-RUN-PATH'] = args.continue_run_folder_path


    # load dataset variables
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)
    dataset_name = options['DATASET-NAME']
    if ('FP' not in options['DATASET-NAME']) and ('KITTI' not in options['DATASET-NAME']):
        options['OVERLAP-CSV-PATH'] = config[f'REGISTER-{dataset_name}']['OVERLAP-CSV-PATH']

    # load icp variables
    icp_vars = config['REFINE']
    options.update(icp_vars)
    options['METHOD-REFINEMENT-NAME'] = icp_vars['ICP-VERSION']

    register_icp(options)