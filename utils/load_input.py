
import numpy as np
import os
import pandas as pd
import pickle
import os.path as osp
import open3d as o3d
from tqdm import tqdm

try:
    from utils.rot_utils import homo_matmul, T_COLS
except:
    from rot_utils import homo_matmul, T_COLS
    

def count_registration_examples(data_dict,printout=False):
    counter = 0
    for fname in data_dict.keys():
        counter += len(data_dict[fname]['eval'])
    if printout:
        print(f'There are {counter} examples')
    return counter

def process_log(log_path):
    """
    Process .log file given in 3DMatch dataset.
  
    Input:   log_path: (str) location of .log file
    Returns: log_dict: (dict) of data prepared for regsitration in format:
                        "i j" pairs and GT transforamtion
                        {"i j": Tji}
    """
    
    log_dict = {}
    
    # read gt.log
    with open(log_path, 'r') as f:
        log = f.read()
        
    # split every line
    log = log.split('\n')
    
    # iterate over every 5 lines
    # first line says which point clouds (i,j) have a match
    # next 4 lines is the 4x4 transfomration matrix that alignts j to i
    for line in range(0,len(log)-5,5):
        
        # name is 'i j' -- index of pc i and index of pc j separated by space
        name = log[line].split('\t')[0] + log[line].split('\t')[1] 
        name = name.replace('  ',' ')
        
        # get transformation matrix 4x4 from next 4 lines
        transformation = np.zeros((4,4))
        transformation[0,:] = [float(x) for x in log[line+1].strip().split('\t')]
        transformation[1,:] = [float(x) for x in log[line+2].strip().split('\t')]
        transformation[2,:] = [float(x) for x in log[line+3].strip().split('\t')]
        transformation[3,:] = [float(x) for x in log[line+4].strip().split('\t')]
        
        log_dict[name] = transformation
        
    return log_dict

def load_3DMATCH(data_path):
    """
    Create 3DMatch data dictionary for registration.
  
    Input:   data_path: (str) location of 3DMatch folders
    Returns: fragments_dict: (dict) of data prepared for regsitration in format:
                            folder_name:
                                full_data_path: full path of folder
                                N: number of point clouds per folder
                                eval: dict of registration pairs with
                                      "i j" pairs and GT transforamtion
                                      {"i j": Tji}
    """
    
    # folder names
    fragment_folders = ['7-scenes-redkitchen',
                        'sun3d-mit_76_studyroom-76-1studyroom2',
                        'sun3d-home_at-home_at_scan1_2013_jan_1',
                        'sun3d-home_md-home_md_scan9_2012_sep_30',
                        'sun3d-hotel_uc-scan3',
                        'sun3d-hotel_umd-maryland_hotel1',
                        'sun3d-hotel_umd-maryland_hotel3',
                        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
                        ]

    fragments_dict = {}
    for ff in fragment_folders:
        fragments_dict[ff] = {}    
        
        # full data paths
        ff_path = osp.join(data_path,ff)
        fragments_dict[ff]['full_data_path'] = ff_path
        
        # count number of point clouds per folder
        pcs = [x for x in os.listdir(ff_path) if x.endswith('.ply')]
        fragments_dict[ff]['N'] = len(pcs)

        
        # process gt.log transformation matrices
        ff_eval_path = osp.join(data_path,ff + '-evaluation')
        log_path = osp.join(ff_eval_path,'gt.log')
        gt_log = process_log(log_path)
        fragments_dict[ff]['eval'] = gt_log

    return fragments_dict, fragment_folders

def load_3DLoMATCH(data_path):
    
    fragment_folders = ['7-scenes-redkitchen', 
                        'sun3d-home_at-home_at_scan1_2013_jan_1',
                        'sun3d-home_md-home_md_scan9_2012_sep_30', 
                        'sun3d-hotel_uc-scan3',
                        'sun3d-hotel_umd-maryland_hotel1',
                        'sun3d-hotel_umd-maryland_hotel3',
                        'sun3d-mit_76_studyroom-76-1studyroom2',
                        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
    fragments_dict = {}

    
    pkl_path = osp.join(data_path,"3DLoMatch.pkl")
    with open(pkl_path,"rb") as f:
        infos = pickle.load(f)

    scenes = np.array([x.split("/")[1] for x in infos["src"]])
    for ff in fragment_folders:
        fragments_dict[ff] = {}
        fragments_dict[ff]['full_data_path'] = f"/data/3DMatch/{ff}"
        fragments_dict[ff]['N'] = sum(scenes == ff)

        # scene_inds = (scenes == ff)
        fragments_dict[ff]['eval'] = {}

    N = len(infos["src"])
    rotations = infos["rot"]
    translations = infos["trans"]
    sources = infos["src"]
    targets = infos["tgt"]
    # overlaps = infos["overlap"]

    for i in range(N):
        gt_trans = np.eye(4)
        gt_trans[:3,:3] = rotations[i]
        gt_trans[:3,3] = translations[i].reshape(3)

        ff = sources[i].split("/")[1]
        if targets[i].split("/")[1] != ff:
            raise ValueError("The keys are not matching!")
        if ff not in fragment_folders:
            raise ValueError("This scene should not exist!")
        
        src_index = sources[i].split("/")[-1].split("cloud_bin_")[1].split(".")[0]
        tgt_index = targets[i].split("/")[-1].split("cloud_bin_")[1].split(".")[0]
        ep = f"{tgt_index} {src_index}"
        fragments_dict[ff]['eval'][ep] = gt_trans

    return fragments_dict, fragment_folders

def load_ETH(data_path):
    """
    Create ETH data dictionary for registration.
  
    Input:   data_path: (str) location of ETH folders
    Returns: fragments_dict: (dict) of data prepared for regsitration in format:
                            folder_name:
                                full_data_path: full path of folder
                                N: number of point clouds per folder
                                eval: dict of registration pairs with
                                      "i j" pairs and GT transforamtion
                                      {"i j": Tji}
    """
    
    fragment_folders = ['gazebo_summer',  
                        'gazebo_winter',
                        'wood_autmn',
                        'wood_summer']

    fragments_dict = {}
    for ff in fragment_folders:
        fragments_dict[ff] = {}    
        
        # full data paths
        ff_location = os.path.join(data_path,ff)
        fragments_dict[ff]['full_data_path'] = ff_location
        
        # count uniquie ply files
        pcs = [x for x in os.listdir(ff_location) if x.endswith('.ply')]
        fragments_dict[ff]['N'] = len(pcs)
        
        # process transformation matrices
        log_path = os.path.join(ff_location,'gt.log')
        gt_log = process_log(log_path)
        fragments_dict[ff]['eval'] = gt_log

    return fragments_dict, fragment_folders

def process_gt_eth_overlap(data_path='/data/ETH'):
    '''
    Each folder of ETH dataset has overlap computed by Perfect Match paper.
    Process that overlap into csv with columns folder,i,j,overlap as used
    for other datasets
    '''

    fragment_folders = ['gazebo_summer',  
                        'gazebo_winter',
                        'wood_autmn',
                        'wood_summer']

    cols = ['folder','i','j','overlap']
    overlap_df = pd.DataFrame(columns=cols)
    save_to = '/registration-baseline/data/overlaps/knn3cm/eth_overlap_perfect_match.csv'

    folder_list = []
    i_list = []
    j_list = []
    overlap_list = []

    for ff in tqdm(fragment_folders):

        overlap_pth = os.path.join(data_path,ff,'overlapMatrix.csv')
        overlaps_f = pd.read_csv(overlap_pth,header=None)

        N_rows, N_cols = overlaps_f.shape

        for i in range(N_rows):
            for j in range(N_cols):
                if i != j:
                    folder_list.append(ff)
                    i_list.append(i)
                    j_list.append(j)
                    overlap_list.append(overlaps_f.loc[i, j].item())
                    
                
    overlap_df['folder'] = folder_list
    overlap_df['i'] = i_list
    overlap_df['j'] = j_list
    overlap_df['overlap'] = overlap_list

    overlap_df.to_csv(save_to,index=False)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_KITTI(data_path):
    """
    Create KITTI data dictionary for registration.
  
    Input:   data_path: (str) location of KITTI folders
    Returns: fragments_dict: (dict) of data prepared for regsitration in format:
                            folder_name:
                                full_data_path: full path of folder
                                N: number of point clouds per folder
                                eval: dict of registration pairs with
                                      "i j" pairs and GT transforamtion
                                      {"i j": Tji}
    """
    
    # folder names
    fragment_folders = ['08',  
                        '09',
                        '10']
    
    # load dict for test examples -- taken from GeoTransformer
    test_examples = load_pickle(f'{data_path}/test.pkl')
    
    fragments_dict = {}
    for ff in fragment_folders:
        fragments_dict[ff] = {}
        fragments_dict[ff]['data'] = []
        
    for info in test_examples:
        padded_folder_name = f'{info["seq_id"]:02d}'
        
        fragments_dict[padded_folder_name]['full_data_path'] = f'{data_path}/{padded_folder_name}/velodyne'
        
        name = f'{info["frame0"]:06d} {info["frame1"]:06d}'
        if 'eval' in fragments_dict[padded_folder_name].keys():
            fragments_dict[padded_folder_name]['eval'][name] = info['transform']
        else:
            fragments_dict[padded_folder_name]['eval'] = {}
            fragments_dict[padded_folder_name]['eval'][name] = info['transform']

        # fragments_dict[padded_folder_name]['data'].append(f'{info["frame0"]:06d}.bin')
        # fragments_dict[padded_folder_name]['data'].append(f'{info["frame1"]:06d}.bin')

    for ff in fragment_folders:
        # fragments_dict[ff]['data'] = np.unique(fragments_dict[ff]['data'])
        fragments_dict[ff]['N'] = len(np.unique(fragments_dict[ff]['data']))
        
    return fragments_dict, fragment_folders

def load_FAUSTpartial(data_path,benchmark_path,indices_pth):
    """
    Create FAUST-partial data dictionary for registration.
  
    Input:   data_path: (str) location of FAUST training scans
             benchmark_path: (str) location of benchmark csv with columns 
                            Scan,Viewpoint_i,Viewpoint_j,overlap,T00,T01,T02,T03,
                            T10,T11,T12,T13,T20,T21,T22,T23,T30,T31,T32,T33
                            that tells us which viewpoints of the Scan needs to 
                            be registered, and Tij are the elements of the 4x4 
                            transformation matrix
            indices_path: (str) location of the indices for the partial viewpoints
                                that correspond to the viewpoints in the csv from
                                benchmark_path
    Returns: fragments_dict: (dict) of data prepared for regsitration in format:
                            folder_name:
                                full_data_path: full path of folder
                                N: number of point clouds per folder
                                eval: dict of registration pairs with
                                      "i j" pairs and GT transforamtion
                                      {"i j": Tji}
    """

    benchmark = pd.read_csv(benchmark_path,index_col=None)
    
    fragment_folders = [f'tr_scan_{x:03d}.ply' for x in range(100)]

    fragments_dict = {}
    for ff in fragment_folders:
        fragments_dict[ff] = {}  

        name = ff.split('.ply')[0]  
        
        # full data paths
        fragments_dict[ff]['full_data_path'] = data_path
        fragments_dict[ff]['indices_pth'] = osp.join(indices_pth,f'indices_{name}.pickle')
        
        # store ply data filenames
        # sorted_data = sorted([x for x in os.listdir(data_path)])
        # fragments_dict[ff]['data'] = sorted_data
        
        # process gt.log transformation matrices
        benchmark_filtered = benchmark[benchmark['Scan'] == name]
        benchmark_filtered.reset_index(inplace=True,drop=True)
        vi = benchmark_filtered['Viewpoint_i'].tolist()
        vj = benchmark_filtered['Viewpoint_j'].tolist()
        gt_log = {f'{vi[i]} {vj[i]}': np.array(benchmark_filtered.loc[i,T_COLS]).astype('float64').reshape(4,4)  
                        for i in range(len(vi)) }
        fragments_dict[ff]['eval'] = gt_log

        fragments_dict[ff]['N'] = 1

    return fragments_dict, fragment_folders

def load_dataset(config):
    """
    Load 3DMatch or KITTI or ETH or FAUST-partial dataset.
  
    Input:   config: (dictionary) with options:
                    DATASET-NAME -- name of dataset to load
                    OVERLAP-CSV-PATH -- path of csv that defines the overlap,
                                        used only for 3DMatch and ETH
                    BENCHMARK-PATH -- path to benchmark csv for Faust-partial only
                    SCANS-PATH -- path to scans of FAUSt-partial
    Returns: data_dict: (dict) of data prepared for regsitration in format:
                            folder_name:
                                full_data_path: full path of folder
                                N: number of point clouds per folder
                                eval: dict of registration pairs with
                                      "i j" pairs and GT transforamtion
                                      {"i j": Tji}
            folder_names: (list) names of folders in the dataset
    """
    
    dataset_name = config['DATASET-NAME']

    if dataset_name.upper() == '3DMATCH':
        data_dict, folder_names = load_3DMATCH(config['DATASET-PATH'])
        overlap_csv_path = config['OVERLAP-CSV-PATH']
        data_dict = filter_overlap(data_dict,overlap_csv_path,threshold=0.30)
    elif dataset_name.upper() == '3DLOMATCH':
        data_dict, folder_names = load_3DLoMATCH(config['DATASET-PATH'])
    elif dataset_name.upper() == 'KITTI':
        data_dict, folder_names = load_KITTI(config['DATASET-PATH'])

    elif dataset_name.upper() == 'ETH':
        data_dict, folder_names = load_ETH(config['DATASET-PATH'])
        overlap_csv_path = config['OVERLAP-CSV-PATH']
        data_dict = filter_overlap(data_dict,overlap_csv_path,threshold=0.30)

    elif 'FP' in dataset_name.upper():
        benchmark_root = config['BENCHMARK-PATH']
        benchmark_csv_path = osp.join(benchmark_root,dataset_name,f'BENCHMARK-{dataset_name}.csv')
        indices_path = osp.join(benchmark_root,'indices')
        data_dict, folder_names = load_FAUSTpartial(config['SCANS-PATH'], 
                                                    benchmark_csv_path, 
                                                    indices_path)
    else:
        raise NotImplementedError('Cant load this dataset!')

    return data_dict, folder_names

def skip_examples_condition(ind_i, ind_j, dataset_name, data_dict, fname):

    # skip_examples = False
    
    # if dataset_name.lower() == '3dmatch':
    #     ind_i, ind_j = int(ind_i), int(ind_j)

    #     # NOTE removed this condition -- see
    #     # https://github.com/qinzheng93/GeoTransformer/issues/57
    #     # if not (ind_i + 1 < ind_j):
    #     #     skip_examples = True

    # # for kitti we dont skip examples so this can be omitted
    # elif dataset_name.lower() == 'kitti':
    #     pass

    # elif dataset_name.lower() == 'eth':
    #     ind_i, ind_j = int(ind_i), int(ind_j)

    #     # overlap = data_dict[fname]['overlap']
    #     # if overlap.loc[ind_i, ind_j] < 0.3:
    #     #     skip_examples=True

    # elif dataset_name.lower() == 'faust-partial':
    #     pass

    # return skip_examples
    pass

def filter_overlap(data_dict,overlap_csv_path,threshold=0.30):
    """
    Filter data dictionary used for registration with an overlap dataframe.
    Each example that has overlap < threshold is discarded.
  
    Input:   data_dict: (dict) dictionary of data returned by one of the functions 
                        load_3DMATCH or load_ETH -- see functions for formatting
             overlap_csv_path: (str) path of the overlap csv file with columns
                                [folder,i,j,overlap] that says that for each example from folder
                                in the registration the examples i,j have overlap overlap
             threshold: (float) examples with overlap < threhsold are discarded
    Returns: data_dict: (dict) of filtered data_dict with only examples with overlap > thr in format:
                        "i j" pairs and GT transforamtion
                        {"i j": Tji}
    """

    overlaps = pd.read_csv(overlap_csv_path,index_col=None)
    
    for fname in data_dict.keys():

        overlap_f = overlaps[overlaps['folder'] == fname]
        overlap_f.reset_index(inplace=True,drop=True)

        gt_log = data_dict[fname]['eval']
        new_gt_log = {}

        for key,transformation in gt_log.items():
            ind_i, ind_j = key.split(' ')
            ind_i, ind_j = int(ind_i), int(ind_j)

            mask = (overlap_f['i'] == ind_i) & (overlap_f['j'] == ind_j)

            if (overlap_f.loc[mask,'overlap'] >= threshold).item():
                new_gt_log[key] = transformation

        data_dict[fname]['eval'] = new_gt_log

    return data_dict

def load_point_clouds(ind_i,ind_j,dataset_name,full_data_path,fname, data_dict):
    """
    Load registration example (ind_i, ind_j) for dataset dataset_name.
  
    Input:  ind_i: (int/str) index of first point cloud
            ind_j: (int/str) index of second point cloud
            dataset_name: (str) name of dataset
            full_data_path: (str) folder path to the point clouds
            fname: (str) folder name of the dataset
            data_dict: (dict) dictionary obatined from load_dataset
    Returns: pci: (o3d) open3d point cloud from ind_i
             pcj: (o3d) open3d point cloud from ind_j
    """

    if dataset_name.upper() in ['3DMATCH','3DLOMATCH']:

        ind_i, ind_j = int(ind_i), int(ind_j)
        pci_name = f'cloud_bin_{ind_i}.ply'
        pci_loc = osp.join(full_data_path, pci_name)
        pci = o3d.io.read_point_cloud(pci_loc) 

        pcj_name = f'cloud_bin_{ind_j}.ply'
        pcj_loc = os.path.join(full_data_path, pcj_name)
        pcj = o3d.io.read_point_cloud(pcj_loc)

    elif dataset_name.upper() == 'KITTI':

        pci_name = f'{ind_i}.bin'
        pci_loc = osp.join(full_data_path, pci_name)
        pci_loaded = np.fromfile(pci_loc, dtype=np.float32).reshape(-1, 4)
        pci_loaded = pci_loaded[:, :3].astype(np.float64)
        pci = o3d.geometry.PointCloud()
        pci.points = o3d.utility.Vector3dVector(pci_loaded)

        pcj_name = f'{ind_j}.bin'
        pcj_loc = os.path.join(full_data_path, pcj_name)
        pcj_loaded = np.fromfile(pcj_loc, dtype=np.float32).reshape(-1, 4)
        pcj_loaded = pcj_loaded[:, :3].astype(np.float64)
        pcj = o3d.geometry.PointCloud()
        pcj.points = o3d.utility.Vector3dVector(pcj_loaded)

    elif dataset_name.upper() == 'ETH':

        ind_i, ind_j = int(ind_i), int(ind_j)
        pci_name = f'Hokuyo_{ind_i}.ply'
        pci_loc = osp.join(full_data_path, pci_name)
        pci = o3d.io.read_point_cloud(pci_loc)

        pcj_name = f'Hokuyo_{ind_j}.ply'
        pcj_loc = os.path.join(full_data_path, pcj_name)
        pcj = o3d.io.read_point_cloud(pcj_loc)

    elif 'FP' in dataset_name.upper():
        pc = o3d.io.read_point_cloud(osp.join(full_data_path,fname))
        pc = np.asarray(pc.points)

        with open(data_dict[fname]['indices_pth'],'rb') as f:
            indices = pickle.load(f) 

        vi = indices[ind_i]
        vj = indices[ind_j]

        pci_orig = pc[vi]
        pcj_orig = pc[vj]

        T_gt = data_dict[fname]['eval'][f'{ind_i} {ind_j}']
        pci_orig_T_gt = homo_matmul(pci_orig,T_gt)

        pci = o3d.geometry.PointCloud()
        pci.points = o3d.utility.Vector3dVector(pci_orig_T_gt)

        pcj = o3d.geometry.PointCloud()
        pcj.points = o3d.utility.Vector3dVector(pcj_orig)

    return pci, pcj

def sort_eval_pairs(eval_pairs, dataset_name):

    if dataset_name.upper() in ['3DMATCH','3DLOMATCH','KITTI','ETH']:
       sorted_pairs = sorted(eval_pairs, 
                    key=lambda x: tuple(int(i) for i in x.split(' ')))
    elif 'FP' in dataset_name.upper():
        sorted_pairs = eval_pairs
    else:
        raise NotImplementedError('Cant load this dataset!')

    return sorted_pairs



