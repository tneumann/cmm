cmm
===

Reference implementation as well as all scripts to reproduce the results from the paper "Compressed Manifold Modes for Mesh Processing", as presented at SGP 2014.

More infos (including the paper) can be found on the [project page](http://www.drematrix.de/?portfolio=english-compressed-manifold-modes-for-mesh-processing&lang=en).

![Compressed Manifold Modes Teaser](http://www.drematrix.de/wordpress/wp-content/uploads/2014/07/cmm_teaser.jpg)

## External Dependencies

The code is written in Python 2.7 and requires

 - numpy
 - scipy
 - scikits.sparse.cholmod
 - mayavi + traits + vtk + tvtk + pyface (only for 3d visualization)
 - matplotlib (only for 2d visualization)
 - plac (only for "compute_cmm.py" and "export_cmm_dataset.sh")
 - joblib (only for "reproduce_fig14_functionalmaps.py")

### Installation on Ubuntu

Add "sudo" when required:

```
$ apt-get install python-numpy python-scipy mayavi2 python-matplotlib libsuitesparse-dev
$ pip install scikits.sparse
```

Additional dependencies:
```
$ pip install plac joblib
```

### Installation on Windows

There are many free scientific Python distributions such as [Python(x,y)](https://code.google.com/p/pythonxy/) or [Anaconda](http://continuum.io/downloads). These should come with all the dependencies listed above. [WinPython](http://winpython.sourceforge.net/) is also nice but requires additional installation of some of those dependencies (such as mayavi, plac, joblib).

### Installation on Mac OS

AFAIK, all the dependencies can be installed using pip and homebrew, or you can use the [Anaconda](http://continuum.io/downloads) Python distribution.

## Description of scripts

### reproduce_*.py
There are scripts to reproduce the various figures in the paper. For example, "reproduce_fig1_teaser.py" reproduces the teaser figure. It displays the computed CMMs as well as manifold harmonics and varimax modes as a 3D visualization. It is possible to rotate around the meshes. To change to the next eigenfunction use the "idx" slider on the bottom of the window. There are some other controls next to that slider that should be self-explanatory. You can also make a screenshot of all eigenfunctions after a good camera viewpoint was found with the button "Save all".

### compute_cmm.py
General-purpose script to compute CMM basis on a given mesh (OBJ or OFF format).
Example that computes K=6 basis functions with mu=2.0 on the hand mesh:
```
$ python compute_cmm.py meshes/hand_868.obj 6 2.0 -v
```
The "-v" flag toggles interactive visualization of the resulting eigenfunctions.
To output the results in various formats (simple Text-File, Matlab, HDF5, NPY) you have to specify an output directory with the "-o" option. You can also output visualizations as PLY and OFF (with "-off" and "-ply"):
```
$ mkdir /tmp/test_hand
$ python compute_cmm.py meshes/hand_868.obj 6 2.0 -o /tmp/test_hand -ply -off
```

## Bibtex Entry

If you use this code in research, please cite:
```
@article {CMM2014,
  author = {Neumann, T. and Varanasi, K. and Theobalt, C. and Magnor, M. and Wacker, M.},
  title = {Compressed Manifold Modes for Mesh Processing},
  journal = {Computer Graphics Forum},
  volume = {33},
  number = {5},
  issn = {1467-8659},
  pages = {35--44},
  year = {2014},
}
```
