
import yaml
import pandas as pd
import os.path as osp
import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
from tqdm import tqdm
import sys


T_names = ['T00','T01','T02','T03',
            'T10','T11','T12','T13',
            'T20','T21','T22','T23',
            'T30','T31','T32','T33']

hardness_mapper = {'E':'EASY',
                   'M':'MEDIUM',
                   'H':'HARD'}


def parse_dataset_name_to_function_name(name):

    name = name.lower()
    name = name.replace('-','_')
    name = f'create_{name}'

    return name

def filter_overlap(config,hardness):

    root_dir = config['SAVE-TO']

    overlaps = pd.read_csv(osp.join(root_dir, 'overlaps.csv'),
                            index_col=False)

    LB = config[f'OVERLAP-{hardness_mapper[hardness]}'][0]
    UB = config[f'OVERLAP-{hardness_mapper[hardness]}'][1]

    if hardness == 'E':
        mask = (overlaps['overlap'] >= LB) & (overlaps['overlap'] <= UB)
    else:
        mask = (overlaps['overlap'] >= LB) & (overlaps['overlap'] < UB)
        
    overlaps_f = overlaps[mask]
    overlaps_f.reset_index(inplace=True,drop=True)

    saving_name = osp.join(root_dir,f'overlaps-filtered-{hardness}.csv')
    overlaps_f.to_csv(saving_name,index=False)

    return overlaps

def create_intervals(config):

    intervals = {
        'ROTATION-EASY': {'XZ':config['ROTATION-EASY-XZ'], 'Y': config['ROTATION-EASY-Y']},
        'ROTATION-MEDIUM': {'XZ':config['ROTATION-MEDIUM-XZ'], 'Y': config['ROTATION-MEDIUM-Y']},
        'ROTATION-HARD': {'XZ':config['ROTATION-HARD-XZ'], 'Y': config['ROTATION-HARD-Y']},

        'TRANSLATION-EASY': config['TRANSLATION-EASY'],
        'TRANSLATION-MEDIUM': config['TRANSLATION-MEDIUM'],
        'TRANSLATION-HARD': config['TRANSLATION-HARD'],
    }

    return intervals
    
def sample_rotation(ri):

    # for xz range
    # randomly chose from which intervals to sample rotation
    ri_n_xz = len(ri['XZ'])
    ri_choice_xz = np.random.choice(ri_n_xz,2).tolist()
    ri_sample_xz = np.array(ri['XZ'])[ri_choice_xz].tolist()

    # for y range
    ri_n_y = len(ri['Y'])
    ri_choice_y = np.random.choice(ri_n_y,1).tolist()
    ri_sample_y = np.array(ri['Y'])[ri_choice_y].tolist()


    # uniformly sample from these intervals the euler angles
    ri_sample = [ri_sample_xz[0],ri_sample_y[0],ri_sample_xz[1]]
    euler_angles = [np.random.uniform(*x) for x in ri_sample]
    euler_angles = np.array(euler_angles)


    # old code
    # # randomly chose from which intervals to sample rotation
    # ri_choice = np.random.choice(ri_n,3).tolist()
    # ri_sample = np.array(ri)[ri_choice].tolist()
    # # uniformly sample from these intervals the euler angles
    # euler_angles = np.array([np.random.uniform(*x) for x in ri_sample])

    R_rand = ScipyRot.from_euler('xyz',
                                euler_angles,
                                degrees=True).as_matrix()

    return R_rand

def sample_translation(ti):

    # chose random point in [-1,1] cube
    ti_point = np.random.uniform(-1,1,3)
    # project onto unit sphere and then scale by translation desired range
    ti_scale = random.uniform(*ti)
    rt = ti_point / np.linalg.norm(ti_point) * ti_scale

    return rt

def pack_rot_trans(R,t):

    T_rand = np.eye(4)
    T_rand[:3,:3] = R
    T_rand[:3,3] = t

    return T_rand

def sample_transformations(overlaps_filtered,ri,ti):

    generated_transformations = pd.DataFrame(columns=T_names)


    for row_index in tqdm(range(overlaps_filtered.shape[0])):

        R_rand = sample_rotation(ri)
        t_rand = sample_translation(ti)

        T_rand = pack_rot_trans(R_rand, t_rand)

        T_rand_series = pd.Series(T_rand.ravel(),index=T_names)
        generated_transformations.loc[row_index,T_names] = T_rand_series

    return generated_transformations

def create_rotation_benchmark(config,intervals,hardness):

    root_dir = config['SAVE-TO']
    benchmark_path = osp.join(root_dir,f'FP-R-{hardness}')

    # load easy overlap hardness
    overlaps_filtered_path = osp.join(root_dir,f'overlaps-filtered-E.csv')
    overlaps_filtered = pd.read_csv(overlaps_filtered_path, index_col=False)
    
    # set intervals
    ri = intervals[f'ROTATION-{hardness_mapper[hardness]}']
    ti = intervals['TRANSLATION-EASY']


    generated_transformations = sample_transformations(overlaps_filtered,
                                                       ri,
                                                       ti)

    benchmark = pd.concat([overlaps_filtered, generated_transformations], 
                            axis=1, join="inner")

    benchmark.to_csv(osp.join(benchmark_path,f'BENCHMARK-FP-R-{hardness}.csv'),
                     index=False)

def create_translation_benchmark(config,intervals,hardness):

    root_dir = config['SAVE-TO']
    benchmark_path = osp.join(root_dir,f'FP-T-{hardness}')

    # load easy overlap hardness
    overlaps_filtered_path = osp.join(root_dir,f'overlaps-filtered-E.csv')
    overlaps_filtered = pd.read_csv(overlaps_filtered_path, index_col=False)
    
    # set intervals
    ri = intervals[f'ROTATION-EASY']
    ti = intervals[f'TRANSLATION-{hardness_mapper[hardness]}']


    generated_transformations = sample_transformations(overlaps_filtered,
                                                       ri,
                                                       ti)

    benchmark = pd.concat([overlaps_filtered, generated_transformations], 
                            axis=1, join="inner")

    benchmark.to_csv(osp.join(benchmark_path,f'BENCHMARK-FP-T-{hardness}.csv'),
                     index=False)

def create_overlap_benchmark(config,intervals,hardness):

    root_dir = config['SAVE-TO']
    benchmark_path = osp.join(root_dir,f'FP-O-{hardness}')

    # load medium overlap hardness
    overlaps_filtered_path = osp.join(root_dir,f'overlaps-filtered-{hardness}.csv')
    overlaps_filtered = pd.read_csv(overlaps_filtered_path, index_col=False)
    
    # set intervals
    ri = intervals[f'ROTATION-EASY']
    ti = intervals[f'TRANSLATION-EASY']

    generated_transformations = sample_transformations(overlaps_filtered,
                                                       ri,
                                                       ti)

    benchmark = pd.concat([overlaps_filtered, generated_transformations], 
                            axis=1, join="inner")

    benchmark.to_csv(osp.join(benchmark_path,f'BENCHMARK-FP-O-{hardness}.csv'),
                     index=False)

def create_benchmark(config,param,hardness):

    if hardness not in ['E','M','H']:
        sys.exit('Hardness level must be E,M or H')

    root_dir = config['SAVE-TO']
    benchmark_path = osp.join(root_dir,f'FP-{param}-{hardness}')
    config['BENCHMARK-PATH'] = benchmark_path

    if not osp.exists(benchmark_path):
        os.mkdir(benchmark_path)

    intervals = create_intervals(config)

    # create benchmark
    if param == 'R':
        create_rotation_benchmark(config,intervals,hardness)
    elif param == 'T':
        create_translation_benchmark(config,intervals,hardness)
    elif param == 'O':
        create_overlap_benchmark(config,intervals,hardness)
    else:
        msg = f'Cannot create benchmark for parameter {param}. Choices: [R,T,O]'
        raise NotImplementedError(msg)

    print(f'Benchmark FP-{param}-{hardness} created!')


if __name__ == '__main__':

    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    config = config['CREATE-BENCHMARK']
    dataset_name = config['DATASET-NAME']
    root_dir = config['SAVE-TO']

    print('Creating overlaps for 3 hardness levels.')
    for hardness in ['E','M','H']:
        overlaps_filtered_path = osp.join(root_dir,f'overlaps-filtered-{hardness}.csv')
        if osp.exists(overlaps_filtered_path):
            print(f'Filtered overlap {hardness} csv already created.')
            user_input = input('overwrite (O) or skip (S) this process? [O/S]?: ')
            if user_input.lower() == 'o':
                filter_overlap(config,hardness)
        else:
            filter_overlap(config,hardness)

    print('Create benchmark.')
    if dataset_name == 'ALL':
        for param in ['R','T','O']:
            for hardness in ['E','M','H']:
                create_benchmark(config,param,hardness)
    else:
        _, param, hardness = dataset_name.split('-')
        create_benchmark(config,param,hardness)