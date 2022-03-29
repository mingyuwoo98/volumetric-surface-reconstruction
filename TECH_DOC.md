## Topic

Project 3 (+++) - volumetric surface reconstruction. You can use existing data
sets with known camera positions. For example, relatively easy data sets by
Steven Seitz et al. [University of Washington] or real outdoor sequences by Carl
Olsson [Lund University]. Compare visual hull and photonsistency-based
reconstruction (discussed in Topic 9). For visual hull you need to compute
silhouettes (background segmentaion) at each image. You can use automatic
segmentation or semi-supervised segmentation (e.g. interactive graph cuts). Feel
free to use smaller subsets of the data collections. Keep in mind that larger
distances between cameras make photoconsistency less reliable, you may want to
play with robust versions of photo-consistency loss.

## To-Do List

* Mike (mgwoo) & Leon (lclyao) 
  * Mar 22: Research about current library / paper
  * Mar 29: Assignment 2 Method + Depth Method feasibility test
* Jack (s362xu) 
  * Mar 22: Construct the Github README, and also the input loader
  * Mar 29: Photoshop actions, mesh algorithm
