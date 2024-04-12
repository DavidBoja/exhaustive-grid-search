
import argparse
import yaml
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import open3d as o3d
import pandas as pd
import pickle
from dataset_statistics_utils import calc_resolution, calc_size

def get_fragment_folders(dataset_name):
    if dataset_name == "3DMATCH":
        fragment_folders = ['7-scenes-redkitchen', 
                        'sun3d-home_at-home_at_scan1_2013_jan_1',
                        'sun3d-home_md-home_md_scan9_2012_sep_30', 
                        'sun3d-hotel_uc-scan3',
                        'sun3d-hotel_umd-maryland_hotel1',
                        'sun3d-hotel_umd-maryland_hotel3',
                        'sun3d-mit_76_studyroom-76-1studyroom2',
                        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']
    elif dataset_name == "ETH":
        fragment_folders = ['gazebo_summer',  
                            'gazebo_winter',
                            'wood_autmn',
                            'wood_summer']
    elif dataset_name == "KITTI":
        fragment_folders = ['08',  
                            '09',
                            '10']
    elif "FP" in  dataset_name:
        fragment_folders = [f'tr_scan_{x:03d}.ply' for x in range(100)]

    return fragment_folders


def get_folder_pc_names(dataset_name, folder_path, **kwargs):
    if dataset_name == "3DMATCH":
        pc_paths = glob(os.path.join(folder_path,"*.ply"))
    elif dataset_name == "ETH":
        pc_paths = glob(os.path.join(folder_path,"*.ply"))
    elif dataset_name == "KITTI":
        folder_path = os.path.join(folder_path,"velodyne")
        pc_paths = glob(os.path.join(folder_path,"*.bin"))
    elif "FP" in dataset_name:
        # dataset_name je FP-R-E recimo, folder_path je 
        # folder_path scans/tr_scan_000.ply
        # u kwargs stavljam benchmark csv
        # ovdje cu mu vratiti unique viewpoints
        benchmark = kwargs["benchmark"]
        indices_path = kwargs["INDICES_PATH"]
        scan_name = folder_path.split("/")[-1].split(".ply")[0]
        benchmark_subset = benchmark[benchmark["Scan"] == scan_name]
        unique_viewpoints = np.unique(benchmark_subset["Viewpoint_i"].tolist() + 
                                      benchmark_subset["Viewpoint_j"].tolist()).tolist()
        scan_viewpoint_indices = os.path.join(indices_path,f'indices_{scan_name}.pickle')
        viewpoint_indices = pickle.load(open(scan_viewpoint_indices,"rb"))
        unique_viewpoint_inds = [viewpoint_indices[uv] for uv in unique_viewpoints] 

        # NOTE: hacky 
        # instead of returing pc_paths for FP
        # the folder_name is the scan path /scans/tr_scan_000.ply
        # the pc_paths are the viewpoint inds already loaded
        pc_paths = unique_viewpoint_inds


    return pc_paths


def get_pc(dataset_name, pc_path, **kwargs):
    if dataset_name == "3DMATCH":
        pc = np.asarray(o3d.io.read_point_cloud(pc_path).points)
    elif dataset_name == "ETH":
        pc = np.asarray(o3d.io.read_point_cloud(pc_path).points)
    elif dataset_name == "KITTI":
        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        pc = pc[:, :3].astype(np.float64)
    elif "FP" in dataset_name:
        # NOTE: hacky
        # pc_path are the partial indices
        # kwargs["folder_path"] is the scan path
        subsample_inds = pc_path
        pc = np.asarray(o3d.io.read_point_cloud(kwargs["folder_path"]).points)
        pc = pc[subsample_inds]

    return pc


def get_stats(config):

    STATS = {}
    STATS["NR_PCS"] = 0
    STATS["NR_PTS"] = []
    STATS["RESOLUTION"] = []
    STATS["SIZE_X"] = []
    STATS["SIZE_Y"] = []
    STATS["SIZE_Z"] = []

    DATASET_NAME = config["DATASET-NAME"]
    DATASET_PATH = config["DATASET-PATH"]

    fragment_folders = get_fragment_folders(DATASET_NAME)
    if "FP" in DATASET_NAME:
        BENCHMARK_PATH = config["BENCHMARK-PATH"]
        INDICES_PATH = os.path.join(BENCHMARK_PATH,"indices")
        BENCHMARK_CSV_PATH = os.path.join(BENCHMARK_PATH,
                                            DATASET_NAME,
                                            f"BENCHMARK-{DATASET_NAME}.csv")
        benchmark = pd.read_csv(BENCHMARK_CSV_PATH,index_col=None)
        print(f"Loaded {BENCHMARK_CSV_PATH}")
        fp_config = dict(benchmark=benchmark,
                         INDICES_PATH=INDICES_PATH)
    else:
        fp_config = None


    for f in fragment_folders:
        print(f'Processing folder {f}')

        folder_path = os.path.join(DATASET_PATH,f)
        pc_paths = get_folder_pc_names(DATASET_NAME,
                                       folder_path, 
                                       **fp_config)
                
        for pc_path in tqdm(pc_paths):
            STATS["NR_PCS"] = STATS["NR_PCS"] + 1

            pc = get_pc(DATASET_NAME, pc_path,
                        folder_path=folder_path)

            STATS["NR_PTS"].append(pc.shape[0])
            pc_res = calc_resolution(pc)
            STATS["RESOLUTION"].append(pc_res)
            pc_size = calc_size(pc)
            STATS["SIZE_X"].append(pc_size[0])
            STATS["SIZE_Y"].append(pc_size[1])
            STATS["SIZE_Z"].append(pc_size[2])

    nr_pcs = STATS["NR_PCS"]
    avg_nr_pts = np.mean(STATS["NR_PTS"])
    avg_res = np.mean(STATS["RESOLUTION"])
    avg_size_x = np.mean(STATS["SIZE_X"])
    avg_size_y = np.mean(STATS["SIZE_Y"])
    avg_size_z = np.mean(STATS["SIZE_Z"])

    print(f"Dataset statistics for {DATASET_NAME}")
    print(f"Nr point clouds: {nr_pcs}")
    print(f"Avg. # pts:\t {avg_nr_pts}")
    print(f"Avg. resolution:\t {avg_res}")
    print(f"Avg. size x-ax:\t {avg_size_x}")
    print(f"Avg. size y-ax:\t {avg_size_y}")
    print(f"Avg. size z-ax:\t {avg_size_z}")

        



if __name__ == "__main__":
    possible_datasets = ['3DMATCH','KITTI','ETH','FPv1']
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
    args = parser.parse_args()

    # set options
    with open('../config.yaml','r') as f:
        config = yaml.safe_load(f)

    if "FP" in args.dataset_name.upper():
        dataset_vars = config['DATASET-VARS']["FP"]["FIXED-E"][args.dataset_name.upper()]
    else:
        dataset_vars = config['DATASET-VARS'][args.dataset_name.upper()]
    bench_config = config[f'REGISTER-{args.dataset_name.upper()}-B1']
    bench_config.update(dataset_vars)
    if "FP" in args.dataset_name.upper():
        bench_config["DATASET-PATH"] = bench_config["SCANS-PATH"]

    get_stats(bench_config)

