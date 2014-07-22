cmm
===

Reference implementation as well as all scripts to reproduce the results from the paper "Compressed Manifold Modes for Mesh Processing", conditionally accepted to SGP 2014.

Right now, this is a partial release containing the core optimization procedure as well as one example script (teaser.py). Since I'm very busy right now, I will put the remaining code online later (including all the scripts to generate the results/figures from the paper). The code still needs some serious cleanup.

External Dependencies:
 - numpy
 - scipy
 - scikits.sparse.cholmod
 - mayavi + traits + vtk + tvtk + pyface (only for 3d visualization)
 - matplotlib (only for 2d visualization)

Description of scripts:
 - teaser.py - compute CMM, MH and Varimax basis on the bumpy cube mesh from the teaser
