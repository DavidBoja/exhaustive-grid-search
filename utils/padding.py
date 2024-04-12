

def padding_options(padding_choice,CENTRAL_VOXEL,NR_VOXELS):
    """
    Pad voxelized volume. Padding order is (z,y,x) axes f
    or both sides of volume -- therefore define padding 
    as (z1,z2, y1,y2, x1,x2).
    
    From an existing volume with CENTRAL_VOXEL and NR_VOXELS, 
    determine how much to pad a given volume to achieve one of the
    padding_choice-s
  
    Input:  padding_choice: (str) choice of voxel padding
            CENTRAL_VOXEL: (tuple)  3dim tuple that determines the index of 
                            the central voxel of an existing volume
            NR_VOXELS: (tuple) 3dim tuple that deemines the size of
                        an existing volume (in number of voxels)
    Returns: pp (tuple): 6dim tuple with scalars that detirmine how much to 
                pad each side of the volume -- in order (z1,z2, y1,y2, x1,x2)
    """
    
    if padding_choice == 'same':
        # same padding -- same as "same padding" for convolutions
        # 1D example:
        #                pad|                                      |pad
        #    inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
        #                |________________|
        #                               |_________________|
        #                                              |________________|
        pp = (CENTRAL_VOXEL[2].item(),
                NR_VOXELS[2].item() - 1 - CENTRAL_VOXEL[2].item(),
                CENTRAL_VOXEL[1].item(),
                NR_VOXELS[1].item() - 1 - CENTRAL_VOXEL[1].item(),
                CENTRAL_VOXEL[0].item(),
                NR_VOXELS[0].item() - 1 - CENTRAL_VOXEL[0].item(),
            )
        pp_xyz = (
            CENTRAL_VOXEL[0].item(), 
            NR_VOXELS[0].item() - 1 - CENTRAL_VOXEL[0].item(), 
            CENTRAL_VOXEL[1].item(),
            NR_VOXELS[1].item() - 1 - CENTRAL_VOXEL[1].item(),
            CENTRAL_VOXEL[2].item(),
            NR_VOXELS[2].item() - 1 - CENTRAL_VOXEL[2].item()
            )
    elif padding_choice == '2thirds':
        # 2thirds padding -- adding two thirds of the NR_VOXELS to each side
        pp = (int((2/3) * NR_VOXELS[2].item()),
                int((2/3) * NR_VOXELS[2].item()),
                int((2/3) * NR_VOXELS[1].item()),
                int((2/3) * NR_VOXELS[1].item()),
                int((2/3) * NR_VOXELS[0].item()),
                int((2/3) * NR_VOXELS[0].item())
            )
        pp_xyz = (
            int((2/3) * NR_VOXELS[0].item()),
            int((2/3) * NR_VOXELS[0].item()),
            int((2/3) * NR_VOXELS[1].item()),
            int((2/3) * NR_VOXELS[1].item()),
            int((2/3) * NR_VOXELS[2].item()),
            int((2/3) * NR_VOXELS[2].item()),
            )
    elif padding_choice == 'all':
        # all padding -- adding NR_VOXELS to each side
        pp = (int( NR_VOXELS[2].item()),
                int( NR_VOXELS[2].item()),
                int( NR_VOXELS[1].item()),
                int( NR_VOXELS[1].item()),
                int( NR_VOXELS[0].item()),
                int( NR_VOXELS[0].item())
            )
        pp_xyz = (
            int( NR_VOXELS[0].item()),
            int( NR_VOXELS[0].item()),
            int( NR_VOXELS[1].item()),
            int( NR_VOXELS[1].item()),
            int( NR_VOXELS[2].item()),
            int( NR_VOXELS[2].item()),
            )
    else:
        msg = 'Viable options for padding: same, 2thirds, all.'
        raise NotImplementedError(msg)


    return pp, pp_xyz