cmm
===

Reference implementation as well as all scripts to reproduce the results from the paper "Compressed Manifold Modes for Mesh Processing", as presented at SGP 2014.

## External Dependencies:

The code is written in Python 2.7 and requires

 - numpy
 - scipy
 - scikits.sparse.cholmod
 - mayavi + traits + vtk + tvtk + pyface (only for 3d visualization)
 - matplotlib (only for 2d visualization)
 - plac (only for "compute_cmm.py" and "export_cmm_dataset.sh")
 - joblib (only for "reproduce_fig14_functionalmaps.py")

### Installation

#### Ubuntu
```
$ sudo apt-get install python-numpy python-scipy mayavi2 python-matplotlib
```
Additional dependencies:
```
$ pip install plac joblib
```

#### Windows

There are many free scientific Python distributions such as [Python(x,y)](https://code.google.com/p/pythonxy/) or [Anaconda](http://continuum.io/downloads). These should come with all the dependencies listed above. [WinPython](http://winpython.sourceforge.net/) is also nice but requires additional installation of some of those dependencies (such as mayavi, plac, joblib).

#### Mac OS

AFAIK, all the dependencies can be installed using pip and homebrew, or you can use the [Anaconda](http://continuum.io/downloads) Python distribution.

## Description of scripts:

### reproduce_*.py
There are scripts to reproduce the various figures in the paper. For example, "reproduce_fig1_teaser.py" reproduces the teaser figure. It displays the computed CMMs as well as manifold harmonics and varimax modes as a 3D visualization. It is possible to rotate around the meshes. To change to the next eigenfunction use the "idx" slider on the bottom of the window. There are some other controls next to that slider that should be self-explanatory. You can also make a screenshot of all eigenfunctions after a good camera viewpoint was found with the button "Save all".

### compute_cmm.py
General-purpose script to compute CMM basis on a given mesh (OBJ or OFF format).
Example that computes K=6 basis functions with mu=2.0 on the hand mesh:
```
$ python compute_cmm.py meshes/hand_868.obj 6 2.0 -v
```
The "-v" flag toggles interactive visualization of the resulting eigenfunctions.
To output the results in various formats (simple Text-File, Matlab, HDF5, NPY9 you have to specify an output directory with the "-o" option. You can also output visualizations in PLY and OFF (with "-off" and "-ply"):
```
$ mkdir /tmp/test_hand
$ python compute_cmm.py meshes/hand_868.obj 6 2.0 -o /tmp/test_hand -ply -off
```
