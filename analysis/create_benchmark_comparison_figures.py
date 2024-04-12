
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

import sys
sys.path.append("../utils")
from dataset_statistics_utils import kde_over_hist_for_overlaps, kde_over_hist_for_translation_range, kde_over_hist_for_angle_range
from load_input import load_3DMATCH, load_ETH, load_KITTI, filter_overlap, load_FAUSTpartial # type: ignore


def compare_benchmark_rotations(path_3dmatch, path_ETH, path_KITTI, path_FAUST_scans, path_FP):
    binwidth=10

    # load data
    data_dict, _ = load_3DMATCH(path_3dmatch)
    data_dict = filter_overlap(data_dict,'/exhaustive-grid-search/data/overlaps/3dmatch_overlap_auto_thr.csv',threshold=0.30)
    ranges_3dmatch = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_ETH(path_ETH)
    data_dict = filter_overlap(data_dict,'/exhaustive-grid-search/data/overlaps/eth_overlap_auto_thr.csv',threshold=0.30)
    ranges_eth = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_KITTI(path_KITTI)
    ranges_kitti = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth)


    benchmark_root = os.path.join(path_FP,"ICO-12-FIXED-E")
    indices_path = os.path.join(benchmark_root,'indices')

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-R-E/BENCHMARK-FP-R-E.csv'),
                                indices_path)
    ranges_FP_E = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-R-M/BENCHMARK-FP-R-M.csv'),
                                indices_path)
    ranges_FP_M = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth,split=True)

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-R-H/BENCHMARK-FP-R-H.csv'),
                                indices_path)
    ranges_FP_H = kde_over_hist_for_angle_range(data_dict,binwidth=binwidth,split=True)

    # create figure
    fig = make_subplots(rows=6, cols=1, row_heights=[10, 10]*3, shared_xaxes=True,
                        subplot_titles=("Angle range density x-ax","", 
                                        "Angle range density y-ax", "", 
                                        "Angle range density z-ax", ""),
                        vertical_spacing=0.05)

    colors = {'3DMatch':'red',
                'ETH':'blue',
                'KITTI':'green'}

    data_packed = [('3DMatch',ranges_3dmatch),
                    ('ETH', ranges_eth),
                    ('KITTI',ranges_kitti)]

    for dataset_name,dataset_ranges in data_packed:
        range_x,range_y,range_z,kdes = dataset_ranges
        
        # plot density curves for x,y,z axes
        showlegend=True
        for ax_name, plot_nr in zip(['x','y','z'],[1,3,5]):
            
            fig.add_trace(go.Scatter(x=kdes[ax_name][0],
                                    y=kdes[ax_name][1] / np.max(kdes[ax_name][1]),
                                    mode='lines',
                                    name=dataset_name,
                                    legendgroup=dataset_name,
                                    showlegend = showlegend,
                                    line_width=3,
                                    line_color=colors[dataset_name]), row=plot_nr,col=1)
            showlegend=False
            
        # plot carpets
        for ax_name, plot_nr,ax_range in zip(['x','y','z'],[2,4,6],[range_x,range_y,range_z]):
            fig.add_trace(go.Scatter(x=ax_range,
                                    y=[dataset_name]*len(ax_range),
                                    mode='markers',
                                    marker=dict(symbol='line-ns-open',color=colors[dataset_name]),
                                    text=dataset_name,
                                    name=dataset_name,
                                    showlegend=showlegend,
                                    legendgroup=dataset_name
                                ),
                    row=plot_nr,col=1)
        
        
    # plot faust partial
    data_packed = [('FP-R-E', ranges_FP_E, 'solid'),
                    ('FP-R-M', ranges_FP_M, 'dash'),
                    ('FP-R-H', ranges_FP_H, 'dot')]

    for dataset_name,dataset_ranges,line_style in data_packed:
        range_x,range_y,range_z,kdes = dataset_ranges
        
        # plot density curves for x,y,z axes
        showlegend=True
        for ax_name, plot_nr in zip(['x','y','z'],[1,3,5]):
            
            if len(kdes.keys()) > 3:
                ax_name_pos = ax_name + '_pos'
                fig.add_trace(go.Scatter(x=kdes[ax_name_pos][0],
                                        y=kdes[ax_name_pos][1] / np.max(kdes[ax_name_pos][1]),
                                        mode='lines',
                                        name=dataset_name,
                                        legendgroup=dataset_name,
                                        showlegend = showlegend,
                                        line = dict(color='black', width=3, dash=line_style)
                                        ), row=plot_nr,col=1)
                ax_name_neg = ax_name+'_neg'
                fig.add_trace(go.Scatter(x=kdes[ax_name_neg][0],
                                        y=kdes[ax_name_neg][1] / np.max(kdes[ax_name_neg][1]),
                                        mode='lines',
                                        name=dataset_name,
                                        legendgroup=dataset_name,
                                        showlegend = False,
                                        line = dict(color='black', width=3, dash=line_style)
                                        ), row=plot_nr,col=1)
                
            else:
            
                fig.add_trace(go.Scatter(x=kdes[ax_name][0],
                                        y=kdes[ax_name][1] / np.max(kdes[ax_name][1]),
                                        mode='lines',
                                        name=dataset_name,
                                        legendgroup=dataset_name,
                                        showlegend = showlegend,
                                        line = dict(color='black', width=3, dash=line_style)
                                        ), row=plot_nr,col=1)
            showlegend=False
            
        # plot carpets
        for ax_name, plot_nr,ax_range in zip(['x','y','z'],[2,4,6],[range_x,range_y,range_z]):
            fig.add_trace(go.Scatter(x=ax_range,
                                    y=[dataset_name]*len(ax_range),
                                    mode='markers',
                                    marker=dict(symbol='line-ns-open',color='black'),
                                    text=dataset_name,
                                    name=dataset_name,
                                    showlegend=showlegend,
                                    legendgroup=dataset_name
                                ),
                    row=plot_nr,col=1)
        
        

        
    fig.update_xaxes(tickangle=-90,
                    tickmode = 'array',
                    tickvals = np.arange(-180,185,10)
                    )

    fig.update_layout(width=1000,height=1000)

    fig.update_layout(xaxis_showticklabels=True, 
                    xaxis3_showticklabels=True,
                    xaxis5_showticklabels=True,
                    xaxis6_showticklabels=True,
                    xaxis6_title_text='degrees',
                    xaxis6_tickfont=dict(color="rgba(0,0,0,0)"),
                    xaxis6_title_standoff=0
                    )



    fig.update_layout(yaxis_tickmode='array',yaxis_tickvals=np.arange(0,1.1,0.2),
                    yaxis3_tickmode='array',yaxis3_tickvals=np.arange(0,1.1,0.2),
                    yaxis5_tickmode='array',yaxis5_tickvals=np.arange(0,1.1,0.2))

    fig.update_annotations(xshift=-490,yshift=-220,textangle=-90)

    # horizontal version
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=0.8
    ))

    # wanted_width_in_mm = 79.9 # 15.99cm is the textwidth for MVA --> half of that (because fitting into one column is )
    # 17780.0 is to convert to 300dpi
    fig.write_image("../assets/angle_ranges_300dpi.png", scale = 4.2)
    fig.write_image("../assets/angle_ranges_300dpi.pdf", scale = 4.2)


def compare_benchmark_translations(path_3dmatch, path_ETH, path_KITTI, path_FAUST_scans, path_FP):
    binwidth=0.1

    # load data
    data_dict, _ = load_3DMATCH(path_3dmatch)
    data_dict = filter_overlap(data_dict, '/exhaustive-grid-search/data/overlaps/3dmatch_overlap_auto_thr.csv',threshold=0.30)
    transl_3dmatch = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_ETH(path_ETH)
    data_dict = filter_overlap(data_dict,'/exhaustive-grid-search/data/overlaps/eth_overlap_auto_thr.csv',threshold=0.30)
    transl_eth = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_KITTI(path_KITTI)
    transl_kitti = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    benchmark_root = os.path.join(path_FP,'ICO-12-FIXED-E')
    indices_path = os.path.join(benchmark_root,'indices')

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-T-E/BENCHMARK-FP-T-E.csv'),
                                indices_path)
    transl_fp_e = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-T-M/BENCHMARK-FP-T-M.csv'),
                                indices_path)
    transl_fp_m = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    data_dict, _ = load_FAUSTpartial(path_FAUST_scans,
                                os.path.join(benchmark_root,'FP-T-H/BENCHMARK-FP-T-H.csv'),
                                indices_path)
    transl_fp_h = kde_over_hist_for_translation_range(data_dict,binwidth=binwidth)

    # create figure
    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.2], shared_xaxes=True,vertical_spacing=0.05,
                        subplot_titles=("Translation distance density","",))

    colors = {'3DMatch':'red',
              'ETH':'blue',
              'KITTI':'green'}

    data_packed = [('3DMatch',transl_3dmatch),
                    ('ETH', transl_eth),
                    ('KITTI',transl_kitti)]

    for dataset_name,dd in data_packed:
        
        x_pts,y_pts,shapes = dd
        
        showlegend=True

        # plot density curve
        fig.add_trace(go.Scatter(x=x_pts,
                                y=y_pts / np.max(y_pts),
                                mode='lines',
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name,
                                line_width=3,
                                line_color=colors[dataset_name]), row=1,col=1)
        showlegend=False
        
        # plot carpet
        fig.add_trace(go.Scatter(x=shapes,
                                y=[dataset_name]*len(shapes),
                                mode='markers',
                                marker=dict(symbol='line-ns-open',color=colors[dataset_name]),
                                text=dataset_name,
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name
                                ),
                    row=2,col=1)
        
    # plot faust partial
    data_packed = [('FP-T-E', transl_fp_e, 'solid'),
                    ('FP-T-M', transl_fp_m, 'dash'),
                    ('FP-T-H', transl_fp_h, 'dot')]

    for dataset_name,dd,line_style in data_packed:
        
        x_pts,y_pts,shapes = dd
        
        showlegend=True

        # plot density curve
        fig.add_trace(go.Scatter(x=x_pts,
                                y=y_pts / np.max(y_pts),
                                mode='lines',
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name,
                                line = dict(color='black', width=3, dash=line_style)
                                ), row=1,col=1)
        showlegend=False
        
        # plot carpet
        fig.add_trace(go.Scatter(x=shapes,
                                y=[dataset_name]*len(shapes),
                                mode='markers',
                                marker=dict(symbol='line-ns-open',color='black'),
                                text=dataset_name,
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name
                                ),
                    row=2,col=1)


    fig.update_xaxes(#tickangle=-90,
                    tickmode = 'array',
                    tickvals = np.arange(0,10.1,0.5)
                    )

    fig.update_layout(xaxis_showticklabels=True, 
                    xaxis2_showticklabels=True,
                    xaxis2_title='meters',
                    xaxis2_tickfont=dict(color="rgba(0,0,0,0)"),
                    xaxis2_title_standoff=0)

    fig.update_annotations(xshift=-330,yshift=-280,textangle=-90)

    # horizontal version
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=0.95
    ))

    fig.write_image("../assets/translation_ranges_300dpi.png", scale = 4.2)
    fig.write_image("../assets/translation_ranges_300dpi.pdf", scale = 4.2)


def compare_benchmark_overlaps(path_FP):
    overlap_3dmatch = pd.read_csv('/exhaustive-grid-search/data/overlaps/3dmatch_overlap_auto_thr.csv')
    overlap_eth = pd.read_csv('/exhaustive-grid-search/data/overlaps/eth_overlap_auto_thr.csv')
    overlap_kitti = pd.read_csv('/exhaustive-grid-search/data/overlaps/kitti_overlap_auto_thr.csv')
    overlap_faust_e = pd.read_csv(os.path.join(path_FP,"ICO-12-FIXED-E/FP-O-E/BENCHMARK-FP-O-E.csv"))
    overlap_faust_m = pd.read_csv(os.path.join(path_FP,"ICO-12-FIXED-E/FP-O-M/BENCHMARK-FP-O-M.csv"))
    overlap_faust_h = pd.read_csv(os.path.join(path_FP,"ICO-12-FIXED-E/FP-O-H/BENCHMARK-FP-O-H.csv"))

    overlap_faust_e['overlap'] = overlap_faust_e['overlap'] / 100
    overlap_faust_m['overlap'] = overlap_faust_m['overlap'] / 100
    overlap_faust_h['overlap'] = overlap_faust_h['overlap'] / 100

    binwidth=0.1
    overlap_3dmatch_range = kde_over_hist_for_overlaps(overlap_3dmatch.loc[overlap_3dmatch['overlap'] >=0.3,'overlap'].tolist(),binwidth=binwidth)
    overlap_eth_range = kde_over_hist_for_overlaps(overlap_eth.loc[overlap_eth['overlap'] >=0.3,'overlap'].tolist(),binwidth=binwidth)
    overlap_kitti_range = kde_over_hist_for_overlaps(overlap_kitti['overlap'].tolist(),binwidth=binwidth)
    overlap_fp_e = kde_over_hist_for_overlaps(overlap_faust_e['overlap'].tolist(),binwidth=binwidth)
    overlap_fp_m = kde_over_hist_for_overlaps(overlap_faust_m['overlap'].tolist(),binwidth=binwidth)
    overlap_fp_h = kde_over_hist_for_overlaps(overlap_faust_h['overlap'].tolist(),binwidth=binwidth)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.2], shared_xaxes=True,vertical_spacing=0.1,
                    subplot_titles=("Overlap percentage density","",))

    colors = {'3DMatch':'red',
              'ETH':'blue',
              'KITTI':'green'}

    data_packed = [('3DMatch',overlap_3dmatch_range,overlap_3dmatch.loc[overlap_3dmatch['overlap'] >=0.3,'overlap'].tolist()),
                    ('ETH', overlap_eth_range,overlap_eth.loc[overlap_eth['overlap'] >=0.3,'overlap'].tolist()),
                    ('KITTI',overlap_kitti_range,overlap_kitti['overlap'].tolist())]

    for dataset_name, overlap_ranges, shapes in data_packed:
        
        x_pts, y_pts = overlap_ranges
        
        showlegend=True

        # plot density curve
        fig.add_trace(go.Scatter(x=x_pts,
                                y=y_pts / np.max(y_pts),
                                mode='lines',
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name,
                                line_width=3,
                                line_color=colors[dataset_name]), row=1,col=1)
        showlegend=False
        
        # plot carpet
        fig.add_trace(go.Scatter(x=shapes,
                                y=[dataset_name]*len(shapes),
                                mode='markers',
                                marker=dict(symbol='line-ns-open',color=colors[dataset_name]),
                                text=dataset_name,
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name
                                ),
                    row=2,col=1)
        
        
    # PLOT FAUST-partial
    data_packed = [('FP-O-E',overlap_fp_e,overlap_faust_e['overlap'].tolist(),'solid'),
                    ('FP-O-M', overlap_fp_m,overlap_faust_m['overlap'].tolist(),'dash'),
                    ('FP-O-H',overlap_fp_h,overlap_faust_h['overlap'].tolist(),'dot')]

    for dataset_name,overlap_ranges,shapes,line_style in data_packed:
        
        x_pts, y_pts = overlap_ranges
        
        showlegend=True

        # plot density curve
        fig.add_trace(go.Scatter(x=x_pts,
                                y=y_pts / np.max(y_pts),
                                mode='lines',
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name,
                                line = dict(color='black', width=3, dash=line_style)
                                ), row=1,col=1)
        showlegend=False
        
        # plot carpet
        fig.add_trace(go.Scatter(x=shapes,
                                y=[dataset_name]*len(shapes),
                                mode='markers',
                                marker=dict(symbol='line-ns-open',color='black'),
                                text=dataset_name,
                                name=dataset_name,
                                showlegend=showlegend,
                                legendgroup=dataset_name
                                ),
                    row=2,col=1)


    fig.update_xaxes( #tickangle=-90,
                    tickmode = 'array',
                    tickvals = np.arange(0,1.1,0.05),
                    ticktext = np.arange(0,106,5)
                    )


    fig.update_layout(xaxis_showticklabels=True, 
                    xaxis2_showticklabels=True,
                    xaxis2_title='percentage',
                    xaxis2_tickfont=dict(color="rgba(0,0,0,0)"),
                    xaxis2_title_standoff=0)

    fig.update_annotations(xshift=-330,yshift=-280,textangle=-90)

    # horizontal version
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="right",
        x=0.96
    ))

    fig.write_image("../assets/overlap_ranges_300dpi.png", scale = 4.2)
    fig.write_image("../assets/overlap_ranges_300dpi.pdf", scale = 4.2)




if __name__ == '__main__':

    choices = ["rotation","translation","overlap"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-P","--param",
                        required=True,
                        type=str, 
                        choices=choices,
                        default='rotation',
                        help='Create benchmark comparison figure from paper.')
    parser.add_argument("--path_3DMatch",
                        required=False,
                        type=str, 
                        default='/data/3DMatch',
                        help='Path to 3DMatch dataset.')
    parser.add_argument("--path_ETH",
                        required=False,
                        type=str, 
                        default='/data/ETH',
                        help='Path to ETH dataset.')
    parser.add_argument("--path_KITTI",
                        required=False,
                        type=str, 
                        default='/data/KITTI',
                        help='Path to KITTI dataset.')
    parser.add_argument("--path_FP",
                        required=False,
                        type=str, 
                        default='/data/FAUST-partial',
                        help='Path to FAUST-partial dataset.')
    parser.add_argument("--path_FAUST_scans",
                        required=False,
                        type=str, 
                        default='/data/FAUST/training/scans',
                        help='Path to FAUST training scans.')
    args = parser.parse_args()

    if args.param == "rotation":
        compare_benchmark_rotations(args.path_3DMatch, 
                                    args.path_ETH, 
                                    args.path_KITTI, 
                                    args.path_FAUST_scans, 
                                    args.path_FP)
    elif  args.param == "translation":
        compare_benchmark_translations(args.path_3DMatch, 
                                       args.path_ETH, 
                                       args.path_KITTI, 
                                       args.path_FAUST_scans, 
                                       args.path_FP)
    elif args.param == "overlap":
        compare_benchmark_overlaps(args.path_FP)
    
    print("Image saved in /exhaustive-grid-search/assets")