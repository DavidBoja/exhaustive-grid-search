
from utils.load_input import load_FAUSTpartial, load_point_clouds, sort_eval_pairs
import argparse
import os.path as osp


if __name__ == "__main__":

    possible_names = []
    for param in ['R','T','O']:
        for hardness in ['E','M','H']:
            possible_names.append(f'FP-{param}-{hardness}')

    parser = argparse.ArgumentParser()
    parser.add_argument("--faust_scans_path",
                        # required=True,
                        type=str, 
                        default="/data/FAUST/training/scans",
                        help='Path to FAUST training scans.')
    parser.add_argument("--benchmark_name",
                        # required=True,
                        type=str, 
                        default="FP-R-E",
                        choices=possible_names,
                        help='Name of benchmark. One of the following FP-{R,T,O}-{E,M,H}.')
    parser.add_argument("--benchmark_root_path",
                        # required=True,
                        type=str, 
                        default="/data/FAUST-partial",
                        help='Path to all the FAUST-partial benchmarks.')
    args = parser.parse_args()



    # load dataset
    benchmark_root_path = osp.join(args.benchmark_root_path,"ICO-12-FIXED-E")
    benchmark_csv_path = osp.join(benchmark_root_path,
                                  args.benchmark_name,
                                  f'BENCHMARK-{args.benchmark_name}.csv')
    benchmark_indices_path = osp.join(benchmark_root_path,'indices')

    data_dict, folder_names = load_FAUSTpartial(args.faust_scans_path,
                                                benchmark_csv_path,
                                                benchmark_indices_path)

    # iterate over dataset registration pairs
    # register pcj (source point cloud) onto pci (target point cloud)
    # given T_gt as the ground truth 4x4 affine transformation matrix
    for fname in folder_names:

        full_data_path = data_dict[fname]['full_data_path']
        
        eval_pairs = list(data_dict[fname]['eval'].keys()) 
        eval_pairs = sort_eval_pairs(eval_pairs, args.benchmark_name)

        name = fname.split('.ply')[0]
        # log_path = osp.join(results_folder_path,f'{name}.log')   

        for ep in eval_pairs:
            # pci is target point cloud
            # pcj is source point cloud
            # goal is to register pcj (source) onto pci (target)

            ind_i, ind_j = ep.split(' ')
            
            pci, pcj = load_point_clouds(ind_i, 
                                         ind_j, 
                                         args.benchmark_name, 
                                         full_data_path,
                                         fname,
                                         data_dict)

            T_gt = data_dict[fname]['eval'][ep]

            #NOTE: fill the rest of the script with your own code