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

