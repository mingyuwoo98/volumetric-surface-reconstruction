# Volumetric Surface Reconstruction

Volumetric surface reconstruction is the reconstruction of  the 3D projection from 
 existing data sets with known camera positions. For example, relatively easy data sets by
Steven Seitz et al. [University of Washington] or real outdoor sequences by Carl
Olsson [Lund University]. 

In this project we constructed and compared visual hull and photonsistency-based
reconstruction. For visual hull you need to compute silhouettes (background segmentaion) at each image. 
You can use automatic segmentation or semi-supervised segmentation (e.g. interactive graph cuts). Feel
free to use smaller subsets of the data collections. Keep in mind that larger
distances between cameras make photoconsistency less reliable, you may want to
play with robust versions of photo-consistency loss.

Please refer to [our paper](Report.pdf) for more detail

## Environment

**Please note that Anaconda is required for environment setup**

Please find the anaconda environment yml file here: [VSR_environment.yml](extlibs/VSR_environment.yml)
The following command can be used to create the environment with Python 3.8.
```
conda env create -f extlibs/VSR_environment.yml
```
The following command activates the project environment. 
```
conda activate VSR
```

The extra package used in the project is as followed:
```
- alphashape==1.3.1
- matplotlib==3.5.1
- notebook==6.4.11
- numpy==1.22.3
- opencv-python==4.5.5.64
- pickle-mixin==1.0.2
- pillow==9.1.0
- scikit-image==0.19.2
- scikit-learn==1.0.2
- scipy==1.8.0
```

## To-Do List

* Mike (mgwoo) & Leon (lclyao) 
  * Mar 22: Research about current library / paper (Done)
  * Mar 29: Assignment 2 Method + Depth Method feasibility test (On-going)
  * Apr 7: Fixing Assignment 2 Method (Dual/Multiple camera view point)
* Jack (s362xu) 
  * Mar 22: Construct the Github README, and also the input loader (Done)
  * Mar 29: Photoshop actions, mesh algorithm (Done)
  * Apr 7: Recreation from paper
