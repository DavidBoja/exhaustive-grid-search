
import argparse
import os.path as osp
import yaml
import time
import torch
from pprint import pprint
import pandas as pd
from datetime import timedelta
import numpy as np
from tqdm import tqdm
import os
import pickle
from utils.eval_utils import RRE, RTE

from fft_conv_pytorch import fft_conv

from utils.load_input import load_dataset, load_point_clouds, sort_eval_pairs
from utils.pc_utils import voxelize, unravel_index_pytorch
from utils.data_utils import preprocess_pcj_B1
from utils.rot_utils import create_T_estim_matrix, load_rotations
from utils.utils import set_seeds, prepare_for_saving_results
from utils.padding import padding_options


def register(config):
    '''
    Register selected dataset
    '''

    pprint(config)
    DEVICE = torch.device('cuda:{}'.format(config['GPU-INDEX']))
    CONTINUE_RUN = config['CONTINUE-RUN']
    DATASET_NAME = config['DATASET-NAME']
    PADDING = config['PADDING']
    BATCH_SIZE = 1
    VOXEL_SIZE = config['VOXEL-SIZE']
    PV = config['PV']
    NV = config['NV']
    PPV = config['PPV']
    NUM_WORKERS = config['NUM-WORKERS']
    ROTATION_CHOICE = config['ROTATION-OPTION']
    SUBSAMPLING_INDS_PATH = config['SUBSAMPLING-INDS-PATH']
    SUBSAMPLE = False
    if not isinstance(SUBSAMPLING_INDS_PATH, type(None)):
        if not os.path.exists(SUBSAMPLING_INDS_PATH):
            raise ValueError(f"Path {SUBSAMPLING_INDS_PATH} does not exist.")
        SUBSAMPLE = True



    # create results paths and files
    now = CONTINUE_RUN if CONTINUE_RUN else time.strftime('%y%m%d%H%M')
   
    results_vars = prepare_for_saving_results(now,CONTINUE_RUN,config)
    results_folder_path, results_df_path, result_columns, results_df = results_vars


    # load inputs
    data_dict, folder_names = load_dataset(config)
    R_batch = load_rotations(ROTATION_CHOICE)
    if SUBSAMPLE: subsample_dict = pickle.load(open(SUBSAMPLING_INDS_PATH,"rb"))

    for fname in folder_names:

        print(f'Register {fname}')

        N_point_clouds_folder = data_dict[fname]['N']  
        full_data_path = data_dict[fname]['full_data_path']
        
        eval_pairs = list(data_dict[fname]['eval'].keys()) 
        eval_pairs = sort_eval_pairs(eval_pairs, DATASET_NAME)

        name = fname.split('.ply')[0] # special case for faust-partial that iterates over examples
        log_path = osp.join(results_folder_path,f'{name}.log')   

        for ep in tqdm(eval_pairs):

            if CONTINUE_RUN:
                if results_df[(results_df['folder'] == fname) & 
                              (results_df['examples'] ==ep)].shape[0]>0:
                    continue
            
            init_time = time.time()
            # pci = target
            # pcj = source
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

            pci = torch.from_numpy(np.asarray(pci.points))
            pcj = torch.from_numpy(np.asarray(pcj.points))

            if SUBSAMPLE:
                pci = pci[subsample_dict[fname][ind_i],:]
                pcj = pcj[subsample_dict[fname][ind_j],:]
            
            #### PREPROCESS pci ##########################################################
            # 1. make pci positive for voxelization
            make_pci_posit_translation = torch.min(pci,axis=0)[0]
            pci = pci - make_pci_posit_translation
            
            # 2. voxelize pci
            pci_voxel, NR_VOXELS_PCI = voxelize(pci, VOXEL_SIZE,
                                                fill_positive=PV,
                                                fill_negative=NV)

            # find indices of the pci central voxel 
            CENTRAL_VOXEL_PCI = torch.where(NR_VOXELS_PCI % 2 == 0, # check if even
                                              (NR_VOXELS_PCI / 2) -1, # if even take one voxel to the left 
                                              torch.floor(NR_VOXELS_PCI / 2)).int() # else just take middle voxel
            # find central voxel in xyz coordinates
            central_voxel_center =  CENTRAL_VOXEL_PCI * VOXEL_SIZE + (0.5*VOXEL_SIZE)

            # 3. move pci on cuda -- dims needed 1 x 1 x Vx x Vy x Vz
            weight_to_fftconv3d = pci_voxel.type(torch.int32).to(DEVICE)[None,None,:,:,:]

            #### PREPROCESS pcj = target ##########################################################
            # define padding (z,y,x) axis is the order for padding
            pp, pp_xyz = padding_options(PADDING,
                                         CENTRAL_VOXEL_PCI,
                                         NR_VOXELS_PCI)

            # batch pcj voxelized data
            # FIXME use **kwargs to write this function?
            my_data, my_dataloader = preprocess_pcj_B1(pcj, 
                                                        R_batch, 
                                                        VOXEL_SIZE, 
                                                        pp, 
                                                        NUM_WORKERS,
                                                        PV,
                                                        NV,
                                                        PPV)

            preprocess_time = time.time() - init_time

            #### PROCESS (FFT) #####################################################################
            maxes = []
            argmaxes = []
            shapes = []
            minimas = torch.empty(R_batch.shape[0],3)

            fft_iter_time = time.time()
            for ind_dataloader,(voxelized_pts_padded,mins,orig_input_shape) in enumerate(my_dataloader):
                minimas[ind_dataloader,:] = mins
                
                input_to_fftconv3d = voxelized_pts_padded.to(DEVICE)
                
                out = fft_conv(input_to_fftconv3d, 
                               weight_to_fftconv3d, bias=None)
                
                maxes.append(torch.max(out))
                argmaxes.append(torch.argmax(out))
                shapes.append(out.shape)

            fft_iter_time = time.time() - fft_iter_time

            #### POST-PROCESS ####################################################################
            post_process_time = time.time()
            # 1. find voxel location with biggest cross-correlation value
            m_index = torch.argmax(torch.stack(maxes)) # tells us which batch had max response
            ind0, _, ind1, ind2, ind3 = unravel_index_pytorch(argmaxes[m_index], 
                                                              shapes[m_index])

            # when batch_size = 1, this equals to m_index
            rotation_index = m_index * BATCH_SIZE + ind0
            R = R_batch[rotation_index]

            # translation -- translate for padding pp_xyz and CENTRAL_VOXEL_PCI
            t = torch.Tensor([-(pp_xyz[0] * VOXEL_SIZE) + 
                              ((CENTRAL_VOXEL_PCI[0]) * VOXEL_SIZE) +
                              (ind1 * VOXEL_SIZE) + 
                              (0.5 * VOXEL_SIZE),
                              
                              -(pp_xyz[2] * VOXEL_SIZE) + 
                              ((CENTRAL_VOXEL_PCI[1]) * VOXEL_SIZE) +
                              (ind2 * VOXEL_SIZE) + 
                              (0.5 * VOXEL_SIZE),
                              
                              -(pp_xyz[4] * VOXEL_SIZE) + 
                              ((CENTRAL_VOXEL_PCI[2]) * VOXEL_SIZE) +
                              (ind3 * VOXEL_SIZE) + 
                              (0.5 * VOXEL_SIZE)
                              ])

            center_pcj_translation = my_data.center
            make_pcj_posit_translation = minimas[rotation_index]
            estim_T_baseline = create_T_estim_matrix(center_pcj_translation,
                                                     R,
                                                     make_pcj_posit_translation,
                                                     central_voxel_center,
                                                     t,
                                                     make_pci_posit_translation
                                                     )
                                                     
            post_process_time = time.time() - post_process_time
            baseline_times = time.time() - init_time

            #### EVAL ###################################################################
            R_est = estim_T_baseline[:3,:3].numpy()
            t_est = estim_T_baseline[:3,3].numpy()
            T_gt = data_dict[fname]['eval'][ep]
            R_gt = T_gt[:3,:3]
            t_gt = T_gt[:3,3]
            rre = RRE(R_gt,R_est)
            rte = RTE(t_gt,t_est)

            #### SAVE RESULTS  ##########################################################   
            # save transformation
            with open(log_path,'a+') as f:
                f.write('{} \t {}\t {}\n'.format(ind_i, ind_j, N_point_clouds_folder))
                f.write(str(estim_T_baseline.tolist()).replace('], [','\n').replace(',','\t')[2:-2])
                f.write('\n')          

            # save results
            current_results = pd.Series([fname,
                                        ep,
                                        timedelta(seconds=preprocess_time).__str__(),# preprocess weight time
                                        timedelta(seconds=fft_iter_time).__str__(),# fft for loop time
                                        timedelta(seconds=post_process_time).__str__(),# post_process_time time
                                        timedelta(seconds=baseline_times).__str__(),# baseline time
                                        rre,
                                        rte
                                        ],
                                        index=result_columns)
            results_df = pd.concat([results_df,current_results.to_frame().T],
                                ignore_index=True)
            results_df.to_csv(results_df_path, index=False)
    
    print('DONE!!')




if __name__ == '__main__':


    possible_datasets = ['3DMATCH','KITTI','ETH','FPv1','3DLOMATCH']
    for param in ['R','T','O']:
        for hardness in ['E','M','H']:
            possible_datasets.append(f'FP-{param}-{hardness}')


    parser = argparse.ArgumentParser()
    parser.add_argument("-D","--dataset_name",
                        required=False,
                        type=str, 
                        choices=possible_datasets,
                        default='3DMATCH',
                        help='Dataset name')
    parser.add_argument("-C","--config_option_name", 
                        required=False,
                        type=str, 
                        default='',
                        help='Specific options from the config.yaml file')
    args = parser.parse_args()

    # set options
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)


    if args.config_option_name in config.keys():
        registration_options = args.config_option_name
    else:
        if args.config_option_name != '':
            print('There is no such option.')
        registration_options = f'REGISTER-{args.dataset_name.upper()}'
        print(f'Using {registration_options} options from config.yaml')
    
 
    config = config[registration_options]

    if config['SET-SEED']:
        set_seeds()

    register(config)