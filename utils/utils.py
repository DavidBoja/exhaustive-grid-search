
from datetime import timedelta
from termcolor import colored
import time
import numpy as np
import torch
import random
import os
import os.path as osp
import pandas as pd
import json

def print_time_elapsed(start,msg,color='red'):
    elapsed = (time.time() - start)
    print(colored("{}: {}".format(msg, timedelta(seconds=elapsed)), color))
    return elapsed

def set_seeds():
    seed_nr = 42
    random.seed(seed_nr)
    np.random.seed(seed_nr)
    torch.manual_seed(seed_nr)

def prepare_for_saving_results(result_folder_name, continue_run, config, cols=None):
    """
    Prepare for saving registration results.
    1. Create new directory for saving results in results/ directory
    2. Create pandas dataframe where save results that will 
       be saved in the new directory under name results.csv
    3. Save config file to the new directory

    Input:  result_folder_name: (str) folder name that will 
                                be created in results/
            config: (dict) dictionary with information 
                    CONTINUE-RUN -- if True, reusing existing results
                                    dataframe
    Returns: results_folder_path: (str) path to results folder
             results_df_path: (str) path to results.csv
             result_columns: (list) list of strings of columns in results.csv
             results_df: (pd.DataFrame) dataframe of results (empty if no 
                          continuing run)
    """

    results_folder_path = osp.join('results',str(result_folder_name))
    results_df_path = osp.join(results_folder_path,'results.csv')

    if cols:
        result_columns = cols
    else:
        result_columns = ['folder',
                        'examples',
                        'preprocess_time',
                        'fft_time',
                        'postprocess_time',
                        'baseline_time',
                        'RRE',
                        'RTE']

    if continue_run:
        results_df = pd.read_csv(results_df_path,
                                 dtype={"folder":str} # important for KITTI only
                                 )
    else:
        os.mkdir(results_folder_path)
        results_df = pd.DataFrame(columns=result_columns)

    # save options from config into results folder
    opts_path = osp.join(results_folder_path,'options.json')
    with open(opts_path,'w+') as f:
        json.dump(config,f)

    return results_folder_path, results_df_path, result_columns, results_df
