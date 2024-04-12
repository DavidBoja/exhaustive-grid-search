import pandas as pd
from utils.eval_utils import eval_from_csv
import yaml
import argparse
import os
import os.path as osp
import json


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
    parser.add_argument("--dataset_name",
                        required=False,
                        type=str,
                        help="When eval old results, the parameter dataset-name is missing")
    parser.add_argument("-P","--partial_results_nr_examples", 
                        required=False,
                        type=int, 
                        default=None,
                        help='Evaluate part of results by giving index to split the result csv.')
    
    args = parser.parse_args()

    # parse choice
    results_path = args.results_folder_path
    f = open (osp.join(results_path,'options.json'), "r")
    options = json.loads(f.read())
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = options['DATASET-NAME']


    # load dataset variables
    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    if 'FP' in dataset_name:
        benchmark_path = options['BENCHMARK-PATH']
        benchmark_type = benchmark_path.split('/')[-1].split('-')
        # case for FPv1
        if len(benchmark_type) == 1:
            benchmark_type = 'FPV1'
        # and case for FP-{R,T,O}-{E,M,H}
        else:
            benchmark_type = f'{benchmark_type[-2]}-{benchmark_type[-1]}' # FIXED-E or FIXED-M

        config = config['DATASET-VARS']['FP']
        THR_ROT = config['THR-ROT']
        THR_TRANS = config['THR-TRANS']
        NR_EXAMPLES = config[benchmark_type][dataset_name]['N']
    else:
        config = config['DATASET-VARS'][dataset_name]
        THR_ROT = config['THR-ROT']
        THR_TRANS = config['THR-TRANS']
        NR_EXAMPLES = config['N']

    results_df = pd.read_csv(osp.join(results_path,'results.csv'))

    if args.partial_results_nr_examples:
        results_df = results_df.loc[:args.partial_results_nr_examples-1,:]

    eval_from_csv(data=results_df,
                thr_rot= THR_ROT,
                thr_trans= THR_TRANS,
                M = NR_EXAMPLES)