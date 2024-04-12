
from tqdm import tqdm
import open3d as o3d
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipyRot
import yaml
from sklearn.neighbors import NearestNeighbors


try:
    from utils.load_input import load_dataset, load_point_clouds
except:
    from load_input import load_dataset, load_point_clouds

def kde_over_hist_for_nr_points(data_dict,dataset_name,binwidth=10000):

    dataset_name = dataset_name.lower()
            
    shapes = []
    for fname in data_dict.keys():
        file_pths = data_dict[fname]['full_data_path']

        if (dataset_name == '3dmatch') or (dataset_name == 'eth'):

            for fl in tqdm(data_dict[fname]['data']):
                pcd = o3d.io.read_point_cloud(f'{file_pths}/{fl}')
                N = np.asarray(pcd.points).shape[0]
                shapes.append(N)

        elif dataset_name == 'kitti':

            for fl in tqdm(data_dict[fname]['data']):

                pc_loc = f'{file_pths}/{fl}'
                pc = np.fromfile(pc_loc, dtype=np.float32).reshape(-1, 4)
                pc = pc[:, :3].astype(np.float64)
                N = pc.shape[0]
                shapes.append(N)

        else:
            raise NotImplementedError('This dataset has not been implemented!')

    
    axx = sns.displot(pd.DataFrame(shapes,columns=['Nr points']), 
                     x="Nr points", 
                     binwidth=binwidth,
                     kde=True,
                     rug=False,
                     legend=False,
                     fill=False,
                    )
    
    x = axx.axes.flat[0].lines[0].get_xdata()
    y = axx.axes.flat[0].lines[0].get_ydata()
    plt.close()

    return x,y,shapes

def kde_over_hist_for_angle_range(data_dict,binwidth=10,split=False):
    
    scene_euler = {}

    for fname in data_dict.keys():
        
        scene_euler[fname] = {}
        
        eval_examples = data_dict[fname]['eval']
        
        all_rot_matrices_from_scene = [eval_examples[k][None,:3,:3] for k in eval_examples.keys()]
        all_rot_matrices_from_scene = np.concatenate(all_rot_matrices_from_scene,axis=0) # N x 3 x 3
        
        r = scipyRot.from_matrix(all_rot_matrices_from_scene)
        returned_rot = r.as_euler('xyz', degrees=True)
        
        scene_euler[fname]['x'] = returned_rot[:,0]
        scene_euler[fname]['y'] = returned_rot[:,1]
        scene_euler[fname]['z'] = returned_rot[:,2]

    all_angle_ranges_x = []
    all_angle_ranges_y = []
    all_angle_ranges_z = []

    for k in scene_euler.keys():
        all_angle_ranges_x += scene_euler[k]['x'].tolist()
        all_angle_ranges_y += scene_euler[k]['y'].tolist()
        all_angle_ranges_z += scene_euler[k]['z'].tolist()

    kdes = {}

    if split:

        for d1,d2 in zip(['x','y','z'],
                [all_angle_ranges_x,all_angle_ranges_y,all_angle_ranges_z]):

            d2_range = np.array(d2)
            d2_pos = d2_range[d2_range>0]
            d2_neg = d2_range[d2_range<0]

            axx = sns.displot(pd.DataFrame(d2_pos,columns=['xxx']), 
                            x="xxx", 
                            binwidth=binwidth,
                            kde=True,
                            rug=False,
                            legend=False,
                            fill=False,
                            )
            
            x_ax = axx.axes.flat[0].lines[0].get_xdata()
            y_ax = axx.axes.flat[0].lines[0].get_ydata()
            plt.close()

            kdes[d1+'_pos'] = (x_ax,y_ax)

            axx = sns.displot(pd.DataFrame(d2_neg,columns=['xxx']), 
                            x="xxx", 
                            binwidth=binwidth,
                            kde=True,
                            rug=False,
                            legend=False,
                            fill=False,
                            )
            
            x_ax = axx.axes.flat[0].lines[0].get_xdata()
            y_ax = axx.axes.flat[0].lines[0].get_ydata()
            plt.close()

            kdes[d1+'_neg'] = (x_ax,y_ax)

    else:
        for d1,d2 in zip(['x','y','z'],
                [all_angle_ranges_x,all_angle_ranges_y,all_angle_ranges_z]):
            axx = sns.displot(pd.DataFrame(d2,columns=['xxx']), 
                            x="xxx", 
                            binwidth=binwidth,
                            kde=True,
                            rug=False,
                            legend=False,
                            fill=False,
                            )
            
            x_ax = axx.axes.flat[0].lines[0].get_xdata()
            y_ax = axx.axes.flat[0].lines[0].get_ydata()
            plt.close()

            kdes[d1] = (x_ax,y_ax)
        
    return all_angle_ranges_x, all_angle_ranges_y, all_angle_ranges_z, kdes

def kde_over_hist_for_translation_range(data_dict,binwidth=0.10):
    
    scene_translation = {}

    for scene in data_dict.keys():
        
        eval_examples = data_dict[scene]['eval']
        
        all_trans_vectors_from_scene = [eval_examples[k][None,:3,3] for k in eval_examples.keys()]
        all_trans_vectors_from_scene = np.concatenate(all_trans_vectors_from_scene,axis=0) # N x 3
        
        norms = np.sqrt(np.sum(all_trans_vectors_from_scene**2,axis=1))
        
        scene_translation[scene] = norms

    all_norms = np.concatenate([scene_translation[k] for k in scene_translation.keys()])

    
    axx = sns.displot(pd.DataFrame(all_norms,columns=['xxx']), 
                        x="xxx", 
                        binwidth=binwidth,
                        kde=True,
                        rug=False,
                        legend=False,
                        fill=False,
                        )

    x_ax = axx.axes.flat[0].lines[0].get_xdata()
    y_ax = axx.axes.flat[0].lines[0].get_ydata()
    plt.close()

    return x_ax, y_ax, all_norms

def kde_over_hist_for_dimension_range(data_dict,dataset_name,binwidth=0.1):

    dataset_name = dataset_name.lower()

    ranges = {}

    for fname in data_dict.keys():
        file_pths = data_dict[fname]['full_data_path']
        
        ranges[fname] = {}
        ranges[fname]['x'] = []
        ranges[fname]['y'] = []
        ranges[fname]['z'] = []

        if (dataset_name == '3dmatch') or (dataset_name == 'eth'):

            for fl in tqdm(data_dict[fname]['data']):
                pcd = o3d.io.read_point_cloud(f'{file_pths}/{fl}')
                pts = np.asarray(pcd.points)
                rr = np.abs(np.max(pts,axis=0) - np.min(pts,axis=0))
                ranges[fname]['x'].append(rr[0]) 
                ranges[fname]['y'].append(rr[1]) 
                ranges[fname]['z'].append(rr[2])

        elif dataset_name == 'kitti':

            for fl in tqdm(data_dict[fname]['data']):

                pc_loc = f'{file_pths}/{fl}'
                pts = np.fromfile(pc_loc, dtype=np.float32).reshape(-1, 4)
                pts = pts[:, :3].astype(np.float64)
                rr = np.abs(np.max(pts,axis=0) - np.min(pts,axis=0))
                ranges[fname]['x'].append(rr[0]) 
                ranges[fname]['y'].append(rr[1]) 
                ranges[fname]['z'].append(rr[2])

        else:
            raise NotImplementedError('This dataset has not been implemented!')

    all_ranges_x = []
    all_ranges_y = []
    all_ranges_z = []

    for k in ranges.keys():
        all_ranges_x += ranges[k]['x']
        all_ranges_y += ranges[k]['y']
        all_ranges_z += ranges[k]['z']
    

    kdes = {}

    for d1,d2 in zip(['x','y','z'],
            [all_ranges_x,all_ranges_y,all_ranges_z]):
        axx = sns.displot(pd.DataFrame(d2,columns=['xxx']), 
                        x="xxx", 
                        binwidth=binwidth,
                        kde=True,
                        rug=False,
                        legend=False,
                        fill=False,
                        )
        
        x_ax = axx.axes.flat[0].lines[0].get_xdata()
        y_ax = axx.axes.flat[0].lines[0].get_ydata()
        plt.close()

        kdes[d1] = (x_ax,y_ax)
        
    return all_ranges_x, all_ranges_y, all_ranges_z, kdes

def kde_over_hist_for_overlaps(data_list,binwidth=0.1):
    
    axx = sns.displot(pd.DataFrame(data_list,columns=['xxx']), 
                        x="xxx", 
                        binwidth=binwidth,
                        kde=True,
                        rug=False,
                        legend=False,
                        fill=False,
                        )

    x_ax = axx.axes.flat[0].lines[0].get_xdata()
    y_ax = axx.axes.flat[0].lines[0].get_ydata()
    plt.close()

    return x_ax, y_ax


# taken from PREDATOR paper
# in scripts/cal_overlap/get_overlap_ratio
def get_overlap_ratio(source,target,threshold=0.03):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    match_count=0
    for i, point in enumerate(source.points):
        # search_radius_vector_3d returns 
        # [nr points closer than threshold, their indices, their distance]
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio

def calc_median_resolution(pc):
    '''
    find the median resolution of the point cloud
    '''
    
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(pc)
    distances, indices = neigh.kneighbors(pc, return_distance=True)
    
    return np.median(distances[:,1])

def calc_quantile_resolution(pc,quant):
    '''
    find the median resolution of the point cloud
    '''
    
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(pc)
    distances, _ = neigh.kneighbors(pc, return_distance=True)
    
    return np.quantile(distances[:,1],quant)

def calc_resolution(pc):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(pc)
    distances, _ = neigh.kneighbors(pc, return_distance=True)
    
    return np.mean(distances[:,1])

def calc_size(pc):    
    x_range, y_range, z_range = np.abs(np.max(pc,axis=0) - 
                                       np.min(pc,axis=0))
    return x_range, y_range, z_range

           
def calc_overlap(config):

    # SETUP ##################################################################
    dataset_name = config['DATASET']

    cols = ['folder','i','j','overlap','threshold']
    overlap_df = pd.DataFrame(columns=cols)
    save_to = config['SAVE-TO']
    save_to = f'{save_to}/{dataset_name.lower()}_overlap_auto_thr.csv'

    #########################################################################
    data_dict, folder_names = load_dataset(config)

    for fname in folder_names:
        # data = data_dict[fname]['data']
        # N_point_clouds_folder = len(data)  

        full_data_path = data_dict[fname]['full_data_path']
        
        eval_pairs = list(data_dict[fname]['eval'].keys()) 
        # sort pairs in ascending order by first and then second index
        eval_pairs = sorted(eval_pairs, key=lambda x: tuple(int(i) for i in x.split(' ')))
        eval_T_gt = data_dict[fname]['eval']

        for ep in tqdm(eval_pairs):
            ind_i, ind_j = ep.split(' ')
            # ind_i, ind_j = int(ind_i), int(ind_j)

            # skip_examples = skip_examples_condition(ind_i, 
            #                                         ind_j, 
            #                                         dataset_name, 
            #                                         data_dict, 
            #                                         fname)

            # if skip_examples:
            #     continue

            pci, pcj = load_point_clouds(ind_i, 
                                         ind_j, 
                                         dataset_name, 
                                         full_data_path,
                                         fname,
                                         data_dict)

            # register pcj onto pci
            # need to be o3d point clouds
            T_gt = eval_T_gt[ep]
            pcj.transform(T_gt)

            src = pcj
            tgt = pci

            # threshold = 0.06
            threshold = 3 * calc_median_resolution(src.points)

            overlap_ratio = get_overlap_ratio(src,tgt,threshold)

            current_results = pd.Series([fname,
                                        ind_i,
                                        ind_j,
                                        overlap_ratio,
                                        threshold
                                        ],
                                        index=cols)
            overlap_df = pd.concat([overlap_df,current_results.to_frame().T],
                                    ignore_index=True)
            overlap_df.to_csv(save_to, index=False)



if __name__ == '__main__':

    with open('config.yaml','r') as f:
        config = yaml.safe_load(f)

    dataset_name = config['CALC-OVERLAP']['DATASET']
    dataset_vars = config['DATASET-VARS'][dataset_name.upper()]

    config = config['CALC-OVERLAP']

    for k,v in dataset_vars.items():
        config[k] = v

    calc_overlap(config)