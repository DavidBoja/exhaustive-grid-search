

import numpy as np
from pprint import pprint
import torch

import plotly.graph_objects as go
import plotly.express as px
colors = px.colors.qualitative.Alphabet
better_colors = (colors[:8] + colors[9:14] + colors[15:])


def vizz(scans,names,title,axes=[],filters=[],scan_sizes=[],pts=[],pts_names=[],
         coloring=better_colors, save_to=None, show=True, opacity=[], remove_bg=False,
         segments_between=[]):

    fig = go.Figure()
    
    if not filters:
        filters = [range(x.shape[0])[::1] for x in scans]
    else:
        filters = [range(x.shape[0])[::filters[i]] for i,x in enumerate(scans)]

    if not scan_sizes:
        scan_sizes = [1] * len(scans)

    if not opacity:
        opacity = [1] * len(scans)

    for i,scan in enumerate(scans):
        # fig.add_trace(go.Scatter3d(x = scan[filters[i],0], 
        #                            y = scan[filters[i],1], 
        #                            z = scan[filters[i],2], 
        #                     mode='markers',
        #                     marker=dict(size=scan_sizes[i],
        #                                 color=coloring[i],
        #                                 opacity=opacity[i]),
        #                     name=names[i]))
        fig.add_trace(go.Scatter3d(x = scan[filters[i],0], 
                                   y = scan[filters[i],1], 
                                   z = scan[filters[i],2], 
                            mode='markers',
                            marker=dict(size=scan_sizes[i],
                                        color=coloring[i],
                                        opacity=opacity[i],
                                        line=dict(color='black',width=1)),
                            name=names[i]))

    for i,segm in enumerate(segments_between):
        pc1 = segments_between[0][0]
        pc2 = segments_between[0][1]

        xevi = []
        yiloni = []
        zevi = []

        for npc_seg_idx in range(pc1.shape[0]):
            xevi.append(pc1[npc_seg_idx,0])
            xevi.append(pc2[npc_seg_idx,0])
            xevi.append(None)

            yiloni.append(pc1[npc_seg_idx,1])
            yiloni.append(pc2[npc_seg_idx,1])
            yiloni.append(None)

            zevi.append(pc1[npc_seg_idx,2])
            zevi.append(pc2[npc_seg_idx,2])
            zevi.append(None)


        fig.add_trace(go.Scatter3d(x=xevi, 
                                    y=yiloni, 
                                    z=zevi,
                                    marker=dict(
                                        size=4,
                                        color="rgba(0,0,0,0)",
                                    ),
                                    line=dict(
                                        color='violet',
                                        width=3)
                                    ))


        
    ax_colors = ['red','green','blue']
    if axes:
        for i,a in enumerate(axes):
            # a is 3x3 matrix, row 0 shows new x ax, row 1 shows new y ax,..
            centroid = np.mean(scans[i].vertices,0)
#             xes = []
#             yilons = []
#             zs = []
            for direction in [0,1,2]:
                xes = [centroid[0],centroid[0]+a[direction,0]*2,None]
                yilons = [centroid[1],centroid[1]+a[direction,1]*2,None]
                zs = [centroid[2],centroid[2]+a[direction,2]*2,None]
            
                fig.add_trace(go.Scatter3d(x = xes,
                                           y = yilons, 
                                           z = zs, 
                                mode='lines',
                                marker=dict(size=1,color=ax_colors[i],opacity=1),
                               name='Coord system {} - ax {}'.format(i,direction)
    #                            legendgroup='part {}'.format(body_part),
                                ))
            
    if pts:
        if not pts_names:
            pts_names = ['point {}'.format(k) for k in range(len(pts))]
        for i,p in enumerate(pts):
            fig.add_trace(go.Scatter3d(x = [p[0]],
                                       y = [p[1]], 
                                       z = [p[2]], 
                                mode='markers',
                                marker=dict(size=5,color='red',opacity=1),
                               name=pts_names[i]
                                ))

    fig.update_layout(scene_aspectmode='data',
                      width=1000, height=700,
                     title=title,
                     )

    if remove_bg:
        fig.update_scenes(xaxis_visible=False, 
                        yaxis_visible=False,
                        zaxis_visible=False)

    if save_to:
        fig.write_html(save_to)

    if show:
        fig.show()
    else:
        del fig


def visualize_voxels_plotly(data,thr='mean'):
    
    fig = go.Figure()
    # imagine points are given on 3d grid with voxel coordinates
    Vx = data.shape[0]
    Vy = data.shape[1]
    Vz = data.shape[2]
    
    X, Y, Z = np.mgrid[0:Vx:Vx*1j, 
                       0:Vy:Vy*1j, 
                       0:Vz:Vz*1j]
        
    normalized_data = data.flatten() / np.max(data.flatten())
    if thr == 'mean':
        thr = np.mean(normalized_data)
    elif thr == 'quantile':
        thr = np.quantile(normalized_data,0.95)
    colorscale_list = [f"rgba(0,255,0,{x})" if x > thr else f"rgba(0,255,0,0)" for x in normalized_data]
    
    # version 1 -- visualize volume
#     fig.add_trace(go.Volume(
#         x = X.flatten(),
#         y = Y.flatten(),
#         z = Z.flatten(),
#         value = data.flatten(),
# #         colorscale='jet',
#         colorscale=colorscale_list,
#         opacity = 0.2,
#     ))


    # version 2 -- visualize centers of voxels
    # transparet points have low values, green have high
    fig.add_trace(go.Scatter3d(x = X.flatten(),
                                y = Y.flatten(),
                                z = Z.flatten(), 
                                mode='markers',
                                marker=dict(size=2,
#                                             color=data.flatten() / np.max(data.flatten()),
                                            color=colorscale_list,
                                            #opacity=0.5,
#                                            colorscale=[
#                                                         [0.0, "rgba(0,1,0,0.5)"],
#                                                         [1.0, "rgba(0,1,0,1)"]
#                                                        ]
                                           )
                                ))
    
    #TODO add visualization of boxes
    
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
    
    fig.show()
    
# this implementation works better
def visualize_voxels_plotly2(data,color_values=[],color_colors=[]):
    '''
    data: torch tensor 3d volume representing voxelized point cloud
    color_values: list of numbers that are present in data. 
                  Used for coloring these values using the color in colors_color
    colors_color: list of rgba(0,0,0,0) values to color each color_values with
    '''
    fig = go.Figure()

    # preproecss unique voxels
    # imagine points are given on 3d grid with voxel coordinates
    Vx = data.shape[0]
    Vy = data.shape[1]
    Vz = data.shape[2]

    X, Y, Z = np.mgrid[0:Vx:Vx*1j, 
                    0:Vy:Vy*1j, 
                    0:Vz:Vz*1j]

    if color_values:
        colorscale_list = ['rgba(0,0,0,0)'] * len(data.flatten())
        for data_ind,x in enumerate(data.flatten()):
            if x in color_values:
                ind = color_values.index(x)
                colorscale_list[data_ind] = color_colors[ind]

    # colorscale_list = []
    # if color_values:
    #     for x in data.flatten():
    #         if x in color_values:
    #             ind = color_values.index(x)
    #             colorscale_list.append(color_colors[ind])
    #         else:
    #             colorscale_list.append('rgba(0,0,0,0)')
    # else:
    #     for x in data.flatten():
    #         if x == 0:
    #             colorscale_list.append('rgba(0,0,0,0)')
    #         else:
    #             colorscale_list.append('rgba(0,0,255,1)')


    fig.add_trace(go.Scatter3d(x = X.flatten(),
                                y = Y.flatten(),
                                z = Z.flatten(), 
                                mode='markers',
                                marker=dict(size=3,
                                            color=colorscale_list
                                        )
                                ))

    fig.update_layout(scene_aspectmode='data',
                        width=1000, height=700,
                        title='Visualize voxels',
                        )


    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    fig.show()



def visualize_cc_volume(data,voxel_size,isomin=0,pci=None,pcj=None,gt_center_pci=None,
                         show=True, save_to=None, title=None, visualize_top_k_cc=0, 
                         fig=None,width=1000, height=700, pci_estim=None):
    '''
    Visualize cross-correlation output. Draws the output of the cross-corr (data) as a volume
    Additioanlly, plot the pci and pcj (need to be given in right space prior to the visualization)
    Additiaonly plot a point in the maximal cross-correlation and where the maximal cross correlation 
    result should be (gt_center_pci)

    Input: data: numpy of dim Vx x Vy x Vz
           voxel_size: int, the voxelization volume with which the data was voxelized
           isomin: int, the minimal value plottet in the cross correlation volume -- if CC below that value
                    then it appears transparent on the plot
           pci: torch Tensor  dim N x 3, point cloud (template from CC)
           pcj: torch Tensor dim M x 3, point cloud (input from CC)
           gt_center_pci: torch Tensor / numpy array dim 3, vector representing the point 
                          where the central voxel of the 
                          pc_i should be when the pc's are aligned with GT transformation
    '''
    if isinstance(fig,type(None)):
        fig = go.Figure()
    # imagine points are given on 3d grid with voxel coordinates
    Vx = data.shape[0]
    Vy = data.shape[1]
    Vz = data.shape[2]
    
    isomax=np.max(data)
    
    # if isinstance(pcj,type(None)):
    X, Y, Z = np.mgrid[0:Vx-1:Vx*1j, 
                       0:Vy-1:Vy*1j, 
                       0:Vz-1:Vz*1j]
    # turn to real coordaintes and not voxel indices
    X = X * voxel_size + 0.5 * voxel_size
    Y = Y * voxel_size + 0.5 * voxel_size
    Z = Z * voxel_size + 0.5 * voxel_size
    # else:
    #     d1,d2,d3 = ((torch.max(pcj,dim=0)[0] / voxel_size) * voxel_size) + voxel_size
    #     X, Y, Z = np.mgrid[0:d1.item():Vx*1j, 
    #                        0:d2.item():Vy*1j, 
    #                        0:d3.item():Vz*1j]
    
    fig.add_trace(go.Volume(
                        x=X.flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        value=data.flatten(),
                        isomin=isomin,
                        isomax=isomax,
                        opacity=0.08, # needs to be small to see through all surfaces
                        surface_count=17, # needs to be a large number for good volume rendering
                        name='CC',
                        showlegend=True
                    ))
    
    
    # plot maximal CC point
    max_cc_voxel = np.unravel_index(np.argmax(data),
                                    data.shape)
    max_cc_voxel_coords = np.array(max_cc_voxel) * voxel_size + 0.5 * voxel_size
    max_cc_value = data[max_cc_voxel[0],max_cc_voxel[1],max_cc_voxel[2]].item()
    
    fig.add_trace(go.Scatter3d(x = [max_cc_voxel_coords[0]], 
                               y = [max_cc_voxel_coords[1]],
                               z = [max_cc_voxel_coords[2]],
                                mode='markers',
                                marker=dict(size=8,
                                            color='cyan',
                                            opacity=1),
                                name=f'MAX CC - value {max_cc_value}'))
    
    
    # plot real points on this
    if not isinstance(pci,type(None)):
        fig.add_trace(go.Scatter3d(x = pci[::35,0], 
                                   y = pci[::35,1],
                                   z = pci[::35,2],
                            mode='markers',
                            marker=dict(size=2,
                                        color='red',
                                        opacity=1),
                            name='Point cloud i'))

    if not isinstance(pci_estim,type(None)):
        fig.add_trace(go.Scatter3d(x = pci_estim[::35,0], 
                                   y = pci_estim[::35,1],
                                   z = pci_estim[::35,2],
                            mode='markers',
                            marker=dict(size=2,
                                        color='blue',
                                        opacity=1),
                            name='Point cloud i ESTIM'))
        
    if not isinstance(pcj,type(None)):
        fig.add_trace(go.Scatter3d(x = pcj[::35,0], 
                                   y = pcj[::35,1],
                                   z = pcj[::35,2],
                            mode='markers',
                            marker=dict(size=2,
                                        color='green',
                                        opacity=1),
                            name='Point cloud j'))
        
    if not isinstance(gt_center_pci,type(None)):

        voxel_of_GT = torch.floor(gt_center_pci / voxel_size).int()
        cc_value_GT = data[voxel_of_GT[0],voxel_of_GT[1],voxel_of_GT[2]].item()

        fig.add_trace(go.Scatter3d(x = [gt_center_pci[0]], 
                                   y = [gt_center_pci[1]],
                                   z = [gt_center_pci[2]],
                            mode='markers',
                            marker=dict(size=8,
                                        color='chartreuse',
                                        opacity=1),
                            name=f'GT - CC {cc_value_GT}'))

    if visualize_top_k_cc:
        acc_maxes, aac_argmaxes = torch.topk(torch.from_numpy(data).ravel(),visualize_top_k_cc)
        acc_maxes = acc_maxes.numpy()
        aac_argmaxes = aac_argmaxes.numpy()


        for argmax_ind, argmax_i in enumerate(aac_argmaxes):
            top_k_cc_voxel = np.unravel_index(argmax_i,
                                            data.shape)
            top_k_cc_voxel_coords = np.array(top_k_cc_voxel) * voxel_size + 0.5 * voxel_size
            fig.add_trace(go.Scatter3d(x = [top_k_cc_voxel_coords[0]], 
                                        y = [top_k_cc_voxel_coords[1]],
                                        z = [top_k_cc_voxel_coords[2]],
                            mode='markers',
                            marker=dict(size=8,
                                        symbol='x',
                                        color=colors[argmax_ind],
                                        opacity=1),
                            name=f'TOP {argmax_ind} CC {acc_maxes[argmax_ind]}'))

    if isinstance(title,type(None)):
        title = 'CC analysis'
    else:
        title = f'CC analysis - {title}'

    fig.update_layout(scene_aspectmode='data',
                      width=width, height=height,
                     title=title,
                     )
    
    fig.update_layout(legend=dict(
                                xanchor="left",
                                x=0.01
                            )
                     )
    fig.update_scenes(xaxis_visible=False, 
                     yaxis_visible=False,
                     zaxis_visible=False)
    
    if save_to:
        fig.write_html(save_to)

    if show:
        fig.show()
    else:
        del fig
        # return fig


def visualize_cc_volume_notSamePadding(data,voxel_size,padding_displacement_coords,isomin=0,
                                        pci=None,pcj=None,gt_center_pci=None,
                                        show=True, save_to=None, title=None, visualize_top_k_cc=0, 
                                        fig=None,width=1000, height=700):
    '''
    Visualize cross-correlation output. Draws the output of the cross-corr (data) as a volume
    Additioanlly, plot the pci and pcj (need to be given in right space prior to the visualization)
    Additiaonly plot a point in the maximal cross-correlation and where the maximal cross correlation 
    result should be (gt_center_pci)

    Input: data: numpy of dim Vx x Vy x Vz
           voxel_size: int, the voxelization volume with which the data was voxelized
           isomin: int, the minimal value plottet in the cross correlation volume -- if CC below that value
                    then it appears transparent on the plot
           pci: torch Tensor  dim N x 3, point cloud (template from CC)
           pcj: torch Tensor dim M x 3, point cloud (input from CC)
           gt_center_pci: torch Tensor / numpy array dim 3, vector representing the point 
                          where the central voxel of the 
                          pc_i should be when the pc's are aligned with GT transformation
    '''
    if isinstance(fig,type(None)):
        fig = go.Figure()
    # imagine points are given on 3d grid with voxel coordinates
    Vx = data.shape[0]
    Vy = data.shape[1]
    Vz = data.shape[2]
    
    isomax=np.max(data)
    
    #d1,d2,d3 = ((torch.max(pcj,dim=0)[0] / voxel_size) * voxel_size) + voxel_size
    # displace the cc volume because of the uneven padding
    start_x = - padding_displacement_coords[0] + (0.5 * voxel_size)
    start_y = - padding_displacement_coords[1] + (0.5 * voxel_size)
    start_z = - padding_displacement_coords[2] + (0.5 * voxel_size)

    end_x = - padding_displacement_coords[0] + (0.5 * voxel_size) + (Vx * voxel_size)
    end_y = - padding_displacement_coords[1] + (0.5 * voxel_size) + (Vy * voxel_size)
    end_z = - padding_displacement_coords[2] + (0.5 * voxel_size) + (Vz * voxel_size)

    X, Y, Z = np.mgrid[start_x.item():end_x.item():Vx*1j, 
                       start_y.item():end_y.item():Vy*1j, 
                       start_z.item():end_z.item():Vz*1j]
    
    fig.add_trace(go.Volume(
                        x=X.flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        value=data.flatten(),
                        isomin=isomin,
                        isomax=isomax,
                        opacity=0.08, # needs to be small to see through all surfaces
                        surface_count=17, # needs to be a large number for good volume rendering
                        name='CC',
                        showlegend=True
                    ))
    
    
    # plot maximal CC point
    max_cc_voxel = np.unravel_index(np.argmax(data),
                                    data.shape)
    max_cc_voxel_coords = np.array(max_cc_voxel) * voxel_size
    max_cc_voxel_coords = max_cc_voxel_coords + (0.5 * voxel_size) - padding_displacement_coords.numpy()
    max_cc_value = data[max_cc_voxel[0],max_cc_voxel[1],max_cc_voxel[2]].item()
    
    fig.add_trace(go.Scatter3d(x = [max_cc_voxel_coords[0]], 
                               y = [max_cc_voxel_coords[1]],
                               z = [max_cc_voxel_coords[2]],
                                mode='markers',
                                marker=dict(size=8,
                                            color='cyan',
                                            opacity=1),
                                name=f'MAX CC - value {max_cc_value}'))
    
    
    # plot real points on this
    if not isinstance(pci,type(None)):
        fig.add_trace(go.Scatter3d(x = pci[::35,0], 
                                   y = pci[::35,1],
                                   z = pci[::35,2],
                            mode='markers',
                            marker=dict(size=2,
                                        color='red',
                                        opacity=1),
                            name='Point cloud i'))
        
    if not isinstance(pcj,type(None)):
        fig.add_trace(go.Scatter3d(x = pcj[::35,0], 
                                   y = pcj[::35,1],
                                   z = pcj[::35,2],
                            mode='markers',
                            marker=dict(size=2,
                                        color='green',
                                        opacity=1),
                            name='Point cloud j'))
        
    if not isinstance(gt_center_pci,type(None)):

        voxel_of_GT = torch.floor(gt_center_pci / voxel_size).int()
        cc_value_GT = data[voxel_of_GT[0],voxel_of_GT[1],voxel_of_GT[2]].item()

        fig.add_trace(go.Scatter3d(x = [gt_center_pci[0]], 
                                   y = [gt_center_pci[1]],
                                   z = [gt_center_pci[2]],
                            mode='markers',
                            marker=dict(size=8,
                                        color='chartreuse',
                                        opacity=1),
                            name=f'GT - CC {cc_value_GT}'))

    if visualize_top_k_cc:
        acc_maxes, aac_argmaxes = torch.topk(torch.from_numpy(data).ravel(),visualize_top_k_cc)
        acc_maxes = acc_maxes.numpy()
        aac_argmaxes = aac_argmaxes.numpy()


        for argmax_ind, argmax_i in enumerate(aac_argmaxes):
            top_k_cc_voxel = np.unravel_index(argmax_i,
                                            data.shape)
            top_k_cc_voxel_coords = np.array(top_k_cc_voxel) * voxel_size
            top_k_cc_voxel_coords = top_k_cc_voxel_coords + (0.5 * voxel_size) - padding_displacement_coords.numpy()
            fig.add_trace(go.Scatter3d(x = [top_k_cc_voxel_coords[0]], 
                                        y = [top_k_cc_voxel_coords[1]],
                                        z = [top_k_cc_voxel_coords[2]],
                            mode='markers',
                            marker=dict(size=8,
                                        symbol='x',
                                        color=colors[argmax_ind],
                                        opacity=1),
                            name=f'TOP {argmax_ind} CC {acc_maxes[argmax_ind]}'))

    if isinstance(title,type(None)):
        title = 'CC analysis'
    else:
        title = f'CC analysis - {title}'

    fig.update_layout(scene_aspectmode='data',
                      width=width, height=height,
                     title=title,
                     )
    
    fig.update_layout(legend=dict(
                                xanchor="left",
                                x=0.01
                            )
                     )
    fig.update_scenes(xaxis_visible=False, 
                     yaxis_visible=False,
                     zaxis_visible=False)
    
    if save_to:
        fig.write_html(save_to)

    if show:
        fig.show()
    else:
        # del fig
        return fig

def visualize_cc_signals_by_rotation(max_ccs,closest2RGTindex,ordering_of_closestR2GT_by_cc=None,
                                    show=False,save_to=None,title=None):

    if isinstance(ordering_of_closestR2GT_by_cc,type(None)):
        ordering_of_closestR2GT_by_cc = '?'
    if isinstance(title,type(None)):
        title='Max CC for rotation matrix'
    else:
        title=f'Max CC for rotation matrix - {title}'

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.arange(len(max_ccs)), y=max_ccs,
                            mode='markers',
                            marker=dict(size=3),
                            name='max ccs by rotation'))

    # fig.add_trace(go.Scatter(x=np.arange(len(max_ccs)), y=max_ccs,
    #                         mode='lines+markers',
    #                         marker=dict(size=3),
    #                         line=dict(width=1),
    #                         name='max ccs by rotation'))

    fig.add_trace(go.Scatter(x=[closest2RGTindex],y=[max_ccs[closest2RGTindex]],
                            mode='markers',
                            name=f'closest2RGT CC {max_ccs[closest2RGTindex]:.4f}' +
                                f'- order {ordering_of_closestR2GT_by_cc}/{len(max_ccs)}',
                            marker=dict(color='red',size=10)))
    fig.add_trace(go.Scatter(x=[0,len(max_ccs)],y=[max_ccs[closest2RGTindex]]*2,
                            mode='lines',
                            name=f'closest2RGT thr',
                            line=dict(color='red')
                            ))

    mask_pts_greater_cc = np.where(max_ccs > max_ccs[closest2RGTindex])[0]
    fig.add_trace(go.Scatter(x=mask_pts_greater_cc, y=max_ccs[mask_pts_greater_cc],
                            mode='markers',
                            marker=dict(size=3,color='green'),
                            name='pts with greater CC'))

    fig.update_xaxes(title_text='Rotation matrix index')
    fig.update_yaxes(title_text='CC')

    title += f'- closest2RGT CC {max_ccs[closest2RGTindex]:.4f}'
    title += f'- order {ordering_of_closestR2GT_by_cc}/{len(max_ccs)}'

    fig.update_layout(scene_aspectmode='data',
                      width=1500, height=700,
                     title=title,
                     )



    if not isinstance(save_to,type(None)):
        fig.write_html(save_to)

    if show:
        fig.show()
    else:
        return fig