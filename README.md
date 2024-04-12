
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-eth-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-eth-trained-on?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-fp-o-m)](https://paperswithcode.com/sota/point-cloud-registration-on-fp-o-m?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-fp-t-m)](https://paperswithcode.com/sota/point-cloud-registration-on-fp-t-m?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-fp-o-h)](https://paperswithcode.com/sota/point-cloud-registration-on-fp-o-h?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-kitti-trained-on)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti-trained-on?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-3dmatch-at-least-1)](https://paperswithcode.com/sota/point-cloud-registration-on-3dmatch-at-least-1?p=addressing-the-generalization-of-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/addressing-the-generalization-of-3d/point-cloud-registration-on-kitti-fcgf)](https://paperswithcode.com/sota/point-cloud-registration-on-kitti-fcgf?p=addressing-the-generalization-of-3d)


# Exhaustive Grid Search

This Github presents the code for the paper [Addressing the generalization of 3D registration methods with a featureless baseline and an unbiased benchmark](https://link.springer.com/article/10.1007/s00138-024-01510-w) published in the MVA journal.

<p align="center">
  <img src="https://github.com/DavidBoja/exhaustive-grid-search/blob/master/assets/pipeline-image.png" width="900">
</p>

## TL;DR

We propose the following:
1. Simple 3D registration baseline model that outperforms most state-of-the-art learning-based methods
2. Methodology to create a 3D registration benchmark from a point cloud dataset that provides a more informative evaluation than previous benchmarks
3. Porpose the FAUST-partial benchmark created using this methodology
4. Provide an evaluation of a great number of state-of-the-art methods on 4 benchmarks: 3DMatch, KITTI, ETH and FAUST-partial
5. Provide a new rotation space subsampling

<br>

## 🔨 Gettng started


### Using Docker
We provide a Dockerfile to facilitate running the code. Run in terminal:

```bash
cd docker
sh docker_build.sh
sh docker_run.sh CODE_PATH DATA_PATH
```
by adjusting the `CODE_PATH` and (optionally) `DATA_PATH` which are added to the docker container as volumes.

You can attach to the container using
```bash
docker exec -it egs-container /bin/bash
```

<br>

### Not using Docker
If you do not want to use Docker, you can install the python packages listed in `docker/requirements.txt` into your own environment. We tested the code under ubuntu20.04, python3.8, cuda 11.2.2 and cudnn8.

<br>

## 🏃 Running

### Register
You can run:
```bash
python register.py --dataset_name <name>
```

where `dataset_name` can be 3DMATCH, KITTI, ETH or any of the FP-{R,T,O}-{E,M,H} benchmarks (see paper for more details). The script saves the registration results in `results/timestamp`, where `timestamp` changes according to the time of script execution.

<br>

To configure the parameters of the baseline for the dataset you want to register, adjust the `REGISTER-{dataset-name}` option in `config.yaml`. The parameters are:

- `DATASET-NAME`: (str) name of the dataset in uppercase
- `DATASET-PATH`: (str) path of the dataset in uppercase
- `OVERLAP-CSV-PATH`: (str) path to the already processed overlap statistics from `data/overlaps`
- `METHOD-NAME`: (str) name of your experiment for easier experiment tracking
- `SET-SEED`: (bool) remove any randomness in the code (just in case)
- `GPU-INDEX`: (int) the gpu to use to run the code, set as `cuda:{GPU-INDEX}`, if you have multiple gpus
- `DTYPE-REGISTRATION`: (str) torch type of data used
- `VOXEL-SIZE`: (float) size of voxel cube usually in meters (depends on the data used!)
- `CONTINUE-RUN`: (str) if registering a whole dataset breaks, give the path to the `results/timestamp` you want to continue running
- `ROTATION-OPTION`: (str) the precomputed rotations (see paper, Sec. 5.3). See [Notes](##-📝-Notes) for more details.
- `PADDING`: (str) padding of the voxelized source point cloud prior to registration (see paper, Sec. 3). Only `same` padding supported.
- `PV`: (int) fill the voxels that have points inside them with this value
- `NV`: (int) fill the empty voxels that do not have points inside them with this value
- `PPV`: (int) fill the padding voxels with this value
- `NUM-WORKERS`: (int) number of pytorch dataloader workers that prepare the data
- `SUBSAMPLING-INDS-PATH`: (str) path to subsampling indices.See [Notes](##-📝-Notes) for more details.

<br>



###  Refine
To refine the results from the registration above, run:
```bash
python refine.py -R <results/timestamp>
```
where timestamp should be changed to the results path from the previous step you want to refine.

<br>

To configure the parameters of the refinement, adjust the `REFINE` option in `config.yaml`. The parameters are:

- `ICP-VERSION`: (str) ICP version to run. Choices are `generalized`, `p2point`, `p2plane`
- `MAX-ITERATION`: (int) maximum number of refining iterations
- `MAX-CORRESPONDENCE-DISTANCE-QUANTILE`: (float) The distance chosen for the inliers as the quantile of all the point distances.
- `SUBSAMPLING-INDS-PATH`: (str) path to subsampling indices. See [Notes](##-📝-Notes) for more details.

<br>

### Evaluate
To evaluate the registration you can run:
```bash
python evaluate.py -R <results/timestamp>
```
where timestamp should be changed accordingly to indicate your results.

<br>

## 💿 Demo: Register + refine
If you want to register and refine a pair of scans (and not a whole dataset) you can run:
```bash
python demo.py --pc_target_path <path/to/target/pc.ply> --pc_source_path <path/to/source/pc.ply>
```

where:
- `pc_target_path`: (str) is the path to the target point cloud in .ply format
- `pc_source_path`: (str) is the path to the source point cloud in .ply format

<br>

To configure the parameters of the baseline and refinement, adjust the `DEMO` option in `config.yaml`. The parameters are:

- `METHOD-NAME`: (str) name of your experiment for easier experiment tracking
- `SET-SEED`: (bool) remove any randomness in the code (just in case)
- `GPU-INDEX`: (int) the gpu to use to run the code, set as `cuda:{GPU-INDEX}`, if you have multiple gpus
- `VOXEL-SIZE`: (float) size of voxel cube usually in meters (depends on the data used!)
- `ROTATION-OPTION`: (str) the precomputed rotations (see paper, Sec. 5.3). See [Notes](##-📝-Notes) for more details.
- `PADDING`: (str) padding of the voxelized source point cloud prior to registration (see paper, Sec. 3). Only `same` padding supported.
- `PV`: (int) fill the voxels that have points inside them with this value
- `NV`: (int) fill the empty voxels that do not have points inside them with this value
- `PPV`: (int) fill the padding voxels with this value
- `NUM-WORKERS`: (int) number of pytorch dataloader workers that prepare the data
- `REFINE-NAME`: (str) is the optional name of refinement algorithm - either p2point icp or p2plane icp or generalized icp
- `REFINE-MAX-ITER`: (int) maximum number of refining iterations
- `REFINE-MAX-CORRESPONDENCE-DISTANCE-QUANTILE`: (float) The distance chosen for the inliers as the quantile of all the point distances.

<br>


## 🖼️ Paper Figures and Tables

To facilitate reproducibility and comparison, we provide the python code to create Figures 3, 4 and 5, and provide Latex code to create Tables 3, 4, 5, 6 and 7.

### Figures
To create Figures 3,4 or 5 run the following: 
```bash
cd analysis
python create_benchmark_comparison_figures.py --param <rotation-translation-or-overlap> --path_3DMatch <path/to/3DMatch> --path_ETH <path/to/ETH> --path_KITTI <path/to/KITTI>  --path_FP <path/to/FAUST-partial> --path_FAUST_scans <path/to/FAUST/scans>
```
where `param` can be either `rotation` (Figure 3), `translation` (Figure 4) or `overlap` (Figure 5). You can alter the code in order to add your own benchmark to the figures and compare the rotation, translation and overlap parameter distribution.

<p align="center">
  <img src="https://github.com/DavidBoja/exhaustive-grid-search/blob/master/assets/param_comparisons.png" width="900">
</p>

<br>

### Tables
We provide the Tables 3, 4, 5, 6 and 7 in the folder `latex_tables` so you can more easily use them in your work.


<br>


## 💻 Data

### 3DMatch
Download the testing examples from [here](https://3dmatch.cs.princeton.edu/) under the title `Geometric Registration Benchmark` --> `Downloads`. There are 8 scenes that are used for testing. In total, there are 16 folders, two for each scene with names `{folder_name}` and `{folder_name}-evaluation`:
```
7-scenes-redkitchen
7-scenes-redkitchen-evaluation
sun3d-home_at-home_at_scan1_2013_jan_1
sun3d-home_at-home_at_scan1_2013_jan_1-evaluation
sun3d-home_md-home_md_scan9_2012_sep_30
sun3d-home_md-home_md_scan9_2012_sep_30-evaluation
sun3d-hotel_uc-scan3
sun3d-hotel_uc-scan3-evaluation
sun3d-hotel_umd-maryland_hotel1
sun3d-hotel_umd-maryland_hotel1-evaluation
sun3d-hotel_umd-maryland_hotel3
sun3d-hotel_umd-maryland_hotel3-evaluation
sun3d-mit_76_studyroom-76-1studyroom2
sun3d-mit_76_studyroom-76-1studyroom2-evaluation
sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation
```
We use the overlaps from PREDATOR [1] (found in `data/overlaps`) to filter the data and use only those with overlap > 30%.

<br>

### KITTI
Download the testing data from [here](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) under `Download odometry data set (velodyne laser data, 80 GB)`. 3 scenes are used for testing:
```
08
09
10
```
Download the `test.pkl` from GeoTransformer [3] [here](https://github.com/qinzheng93/GeoTransformer/blob/main/data/Kitti/metadata/test.pkl) and put it in the same directory where the scenes are located.


<br>

### ETH
Download the testing data from [here](https://github.com/zgojcic/3DSmoothNet#eth). There are 4 scenes that are used for testing:
```
gazeebo_summer
gazeebo_winter
wood_autumn
wood_summer
```
We use the overlaps from Perfect Match [2] (found in `data/overlaps`) to filter the data and use only those with overlap > 30%. We obtain the overlaps from their `overlapMatrix.csv` in each scene.

<br>

### FAUST-partial

Download the FAUST scans from [here](http://faust.is.tue.mpg.de/challenge/Inter-subject_challenge/datasets). There are 100 scans in the training dataset named `tr_scan_xxx.ply` that are used for the registration benchmark. Download the `FAUST-partial` benchmark from [here](https://ferhr-my.sharepoint.com/:f:/g/personal/dbojanic_fer_hr/EgH5iaoUDp1PmL1K8xBDnCQBXU82ZlrSG_PiZmlIEK7dwQ?e=2U9EtJ).

To use the benchmark with your own method, we provide a script that facilitates loading the benchmark:
```bash
python load_faust_partial.py --faust_scans_path <path/to/FAUST/scans> --benchmark_name <FP-{R,T,O}-{E,M,H}> --benchmark_root_path <path/to/FAUST-partial/benchmarks>
```

where:
- `faust_scans_path`: (str) is the path to the FAUST training scans
- `benchmark_name`: (str) is the name of the FAUST-partial benchmark. Choices are FP-{R,T,O}-{E,M,H}
- `benchmark_root_path`: (str) is the path to the FAUST-partial benchmarks

The script loads the data and iterates over the registration pairs and provides the `source point cloud`, `target point cloud` and `ground truth 4x4 transformation`. Complete the rest of the script with your own method in order to use it.

<br>


⚠️ The first version of the [benchmark](https://github.com/DavidBoja/FAUST-partial) was created in the paper ["Challenging universal representation of deep models for 3D point cloud registration"](https://github.com/DavidBoja/greedy-grid-search) and is different from this version of the benchmark. We denominate the first version as `FPv1` and the new version as `ICO-12-FIXED-E`. We make both benchmarks available under the same download link. Read the appropriate papers to get more details about the benchmark differences. ⚠️

<br>


# 🏋🏼 Creating your own benchmark from a point cloud dataset

In the paper, we present a methodology for creating a 3D registration benchmark that provides better evaluations with more insights. We provide the code in `data/FAUST-partial` with which we create the FAUST-partial benchmark, but we note that it can be used on any point cloud dataset.

First, we create the viewpoints using the points of an icosahaedron (see paper for more details). To do so, you can setup the variables in `data/FAUST-partial/config.yaml`, under `CREATE-INDICES-VIEWPOINTS-OVERLAP`:

1. `FAUST-DATA-PATH`: path to training scans of FAUST dataset
2. `SAVE-TO`: path to save the newly created dataset
3. `VISUALIZE`: visualize the creation of partial benchmark (see Fig. 2 from paper)
4. `ICOSAHEDRON-SCALE`: points of icosahaedron (that act as viewpoints) lie on sphere of radius `ICOSAHEDRON-SCALE`
5. `ICOSAHAEDRON-NR-DIVISIONS`: number of splits of the icosahaedron edges - the icsoahaedron starts with 12 points, and then by splitting the edges it results in 42, 162, 642,.. points

and run:

```python
python create_indices_viewpoints_overlap.py
```

to get the partial scan viewpoints, indices and overlaps.

Next, we create the benchmark by sampling random rotation and translations depending on the dataset difficulty. To do so, setup the variables in `data/FAUST-partial/config.yaml`, under `CREATE-BENCHMARK`:

1. `DATASET-NAME`: which dataset to create (FP-R-E, FP-O-M, ... or "All" to create all of them)
2. `SAVE-TO`: path to save the newly created benchmark
3. `ROTATION-EASY-XZ`: rotation parmaeter range for easy difficulty (in degrees) for x and z axes. Given as list of bounds.
4. `ROTATION-MEDIUM-XZ`: rotation parmaeter range for medium difficulty (in degrees) for x and z axes. Given as list of two bounds.
5. `ROTATION-HARD-XZ`: rotation parmaeter range for hard difficulty (in degrees) for x and z axes. Given as list of bounds. Lower and upper bounds are limited to -180, 180.
6. `ROTATION-EASY-Y`: rotation parmaeter range for easy difficulty (in degrees) for y ax. Given as list of bounds.
7. `ROTATION-MEDIUM-Y`: rotation parmaeter range for medium difficulty (in degrees) for y ax. Given as list of bounds.
7. `ROTATION-HARD-Y`: rotation parmaeter range for hard difficulty (in degrees) for y ax. Given as list of bounds. Lower and upper bounds are limited to -90, 90.
8. `TRANSLATION-EASY`: translation parameter range for easy difficulty (in meters) given as list of bounds
9. `TRANSLATION-MEDIUM`: translation parameter range for medium difficulty (in meters) given as list of bounds
10. `TRANSLATION-HARD`: translation parameter range for hard difficulty (in meters) given as list of bounds. Lower bound is limited to 0.
11. `OVERLAP-EASY`: overlap parameter range for easy difficulty (in percentage) given as list of bounds
12. `OVERLAP-MEDIUM`: overlap parameter range for medium difficulty (in percentage) given as list of bounds
13. `OVERLAP-HARD`: overlap parameter range for hard difficulty (in percentage) given as list of bounds. Lower and upper bounds are limited to 0, 100.

```python
python create_benchmarks.py
```

This creates 9 benchmarks: for each parameter (rotation, translation and overlap) 3 difficulty levels, which are saved in: `{SAVE-TO}/FP-{R,T,E}-{E,M,H}`.

<br>


## 📝 Notes

### Differences from our previous work

This work is a continuation of our previous work [greedy-grid-search](https://github.com/davidboja/greedy-grid-search), where the main difference is the computation of the rotation subsampling and the translation estimation, which provide much better results; along with some minor adjustments. The main difference from the previous benchmark [FAUST-partial](https://github.com/davidboja/FAUST-partial) is that we propose a general methodology to create a new benchmark from a point cloud dataset that provides more insights into the 3D registration evaluation.

<br>

### Rotation options

The different tested rotations from Table 8 (see paper) are listed below with the corresponding row from the table


| # from Table 8 | Name                          | N    | Description                                                                                                      |
|----------------|-------------------------------|------|------------------------------------------------------------------------------------------------------------------|
| #Ref           | AA_ICO162_S10                 | 3536 | Our angle axis sampling from the icosahaedron with 162 vertices and angle step 10                                |
| #10            | EULER-S=15-DUPLICATES         | 6363 | Euler angles (from -180, 180) with angle step 15 with removed duplicate rotations                                |
| #11            | EULER-S=15-LIMITED-DUPLICATES | 1886 | Euler angles (from -90, 90) with angle step 15 with removed duplicate rotations                                  |
| #12            | EULER-S=10-LIMITED-DUPLICATES | 6177 | Euler angles (from -90, 90) with angle step 10 with removed duplicate rotations                                  |
| #13            | HEALPIX                       | 4608 | Healpix sampling from Implicit-PDF paper [4]                                                                     |
| #14            | SUPER-FIBONACCI               | 3536 | Super-fibonacci sampling from [5]                                                                                |
| #15            | AA_ICO42_S15                  | 281  | Our angle axis sampling from the icosahaedron with 42 vertices and angle step 15                                 |
| #16            | AA_ICO42_S10                  | 913  | Our angle axis sampling from the icosahaedron with 42 vertices and angle step 10                                 |
| #17            | AA_ICO162_S15                 | 2289 | Our angle axis sampling from the icosahaedron with 162 vertices and angle step 15                                |
| #18            | AA_ICO162_S24_positive        | 4531 | Our angle axis sampling from the icosahaedron with 162 vertices (only from the one hemisphere) and angle step 24 |
| #19            | AA_ICO642_S30                 | 4368 | Our angle axis sampling from the icosahaedron with 642 points and angle step 30                                  |

<br>

### Subsampling options
We provide the subsampling indices [here](https://ferhr-my.sharepoint.com/:f:/g/personal/dbojanic_fer_hr/El0KXDFLFZ9Esmdrt5jxguEBgSVmp9iZOjlaORsCWO_6qA?e=sVHLbg).

<br>

## Citation

If you use our work, please reference our paper:

```
@article{Bojanic24,
  title = {Addressing the generalization of 3D registration methods with a featureless baseline and an unbiased benchmark},
  volume = {35},
  ISSN = {1432-1769},
  url = {http://dx.doi.org/10.1007/s00138-024-01510-w},
  DOI = {10.1007/s00138-024-01510-w},
  number = {3},
  journal = {Machine Vision and Applications},
  publisher = {Springer Science and Business Media LLC},
  author = {Bojanić,  David and Bartol,  Kristijan and Forest,  Josep and Petković,  Tomislav and Pribanić,  Tomislav},
  year = {2024},
  month = mar 
}
```

<br>

## References 
[1] [PREDATOR](https://github.com/prs-eth/OverlapPredator) <br>
[2] [Perfect Match](https://github.com/zgojcic/3DSmoothNet) <br>
[3] [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) <br>
[4] [Implicit-PDF](https://implicit-pdf.github.io/) <br>
[5] [Super-fibonacci](https://github.com/marcalexa/superfibonacci)

<br>

## Acknowledgements
We reuse parts of the [fft-conv-pytorch](https://github.com/fkodom/fft-conv-pytorch) repository for computing the cross-correlation on the GPU.
