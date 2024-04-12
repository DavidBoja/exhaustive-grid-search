
import plotly.graph_objects as go
import numpy as np


def draw_spheres(size, fig, clr='#ffff00', dist=0,name='Sphere',opacity=0.5): 
    '''
    size is diametar
    clr is color '#ffff00'
    dist is center
    '''
    
    # Set up 100 points. First, do angles
    theta = np.linspace(0,2*np.pi,100)
    phi = np.linspace(0,np.pi,100)
    
    # Set up coordinates for points on the sphere
    x0 = dist[0] + size * np.outer(np.cos(theta),np.sin(phi))
    y0 = dist[1] + size * np.outer(np.sin(theta),np.sin(phi))
    z0 = dist[2] + size * np.outer(np.ones(100),np.cos(phi))
    
    # Set up trace
    trace= go.Surface(x=x0, y=y0, z=z0, 
                      colorscale=[[0,clr], [1,clr]],opacity=opacity,
                      name=name)
    trace.update(showscale=False,showlegend=True)

    fig.add_trace(trace)
    
    return fig
    
    
def draw_xyz_axs(fig,k=1):
    
    # xax
    fig.add_trace(go.Scatter3d(x = [0,k],
                               y = [0,0],
                               z = [0,0],
                                mode='markers+lines',
                                marker=dict(size=5,color='red',opacity=1),
                                name='axes',
                                text=['x'],
                                textposition='top right',
                                legendgroup='axes',
                                showlegend=True)
                                    )
    
    # yax
    fig.add_trace(go.Scatter3d(x = [0,0],
                               y = [0,k],
                               z = [0,0],
                                mode='markers+lines',
                                marker=dict(size=5,color='green',opacity=1),
                                name='axes',
                                text=['y'],
                                textposition='top right',
                                legendgroup='axes',
                                showlegend=True)
                                    )
    
    # zax
    fig.add_trace(go.Scatter3d(x = [0,0],
                               y = [0,0],
                               z = [0,k],
                                mode='markers+lines',
                                marker=dict(size=5,color='blue',opacity=1),
                                name='axes',
#                                 text=['y'],
                                textposition='top right',
                                legendgroup='axes',
                                showlegend=True)
                                    )
    
    
    fig.add_trace(go.Scatter3d(x = [0.5*k,0,0],
                               y = [0,0.5*k,0],
                               z = [0,0,0.5*k],
                                mode='text',
                                marker=dict(size=5,color='rgba(1,0,0,1)',opacity=1),
                                name='axes',
                                text=['x','y','z'],
                                legendgroup='axes',
                                showlegend=True)
                                    )
    
    return fig

def draw_viewpoints(fig,ico):

    for ci,camera in enumerate(ico):        

        fig.add_trace(go.Scatter3d(x = [camera[0]],
                                   y = [camera[1]], 
                                   z = [camera[2]], 
                            mode='markers',
                            marker=dict(size=10,color='red',opacity=1),
                            name=f'Camera {ci}')
                                )
    return fig

def draw_partial_pointclouds(fig,ico,pts,colors_pts,indices):

    for ci,camera in enumerate(ico):
        pt_map = indices[f'viewpoint{ci}-{camera}']              
                
        fig.add_trace(go.Scatter3d(x = pts[pt_map[::100],0],
                                   y = pts[pt_map[::100],1], 
                                   z = pts[pt_map[::100],2], 
                            mode='markers',
                            marker=dict(size=2,color=colors_pts[pt_map[::100]],opacity=1),
                            name=f'Partial {ci}')
                                )

    return fig

def visualize_connected_faces(points,faces,fig,color='blue',name='Edges',marker_size=10,
                              mode='marker+lines',line_width=2):
    
    for i,triangle in enumerate(faces):
        p1 = points[triangle[0]]
        p2 = points[triangle[1]]
        p3 = points[triangle[2]]
        
        xes = [p1[0],p2[0],None,p1[0],p3[0],None,p2[0],p3[0],None]
        ys = [p1[1],p2[1],None,p1[1],p3[1],None,p2[1],p3[1],None]
        zs = [p1[2],p2[2],None,p1[2],p3[2],None,p2[2],p3[2],None]
        
        if i == 0:
        
            fig.add_trace(go.Scatter3d(x = xes,
                                       y = ys,
                                       z = zs,
                                        mode=mode,
                                        marker=dict(size=marker_size,color=color,opacity=1),
                                        line=dict(width=line_width),
                                        name=name,
                                        legendgroup=name,
                                        showlegend=True)
                                    )
        else:
            fig.add_trace(go.Scatter3d(x = xes,
                                       y = ys,
                                       z = zs,
                                        mode=mode,
                                        marker=dict(size=marker_size,color=color,opacity=1),
                                        line=dict(width=line_width),
                                        name=name,
                                        legendgroup=name,
                                        showlegend=False)
                                    )
        
    return fig

def draw_icosahaedron(fig,ico_v,ico_f):
    # draw whole icosahaedron
    fig.add_trace(go.Scatter3d(x = ico_v[:,0],
                               y = ico_v[:,1], 
                               z = ico_v[:,2], 
                            mode='markers',
                            marker=dict(size=2,color='blue',opacity=1),
                            name='icosahaedron')
                                )
    
    # draw edges
    fig = visualize_connected_faces(ico_v,
                                    ico_f,
                                    fig,
                                    'blue',
                                    name='icosahaedron edges',
                                    mode='lines')

    return fig

def circle_pts_under_floor(fig,ico):
    ico_neg = ico[ico[:,1] < 0]
    fig.add_trace(go.Scatter3d(x = ico_neg[:,0],
                               y = ico_neg[:,1], 
                               z = ico_neg[:,2], 
                            mode='markers',
                            marker=dict(
                                        color='rgba(1,0,0,0)',
                                        size=8,
                                        line=dict(
                                            color='red',
                                            width=5
                                        )),
                            name='icosah. pts below floor (removed)')
                                )
    return fig