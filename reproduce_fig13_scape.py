import sys
from os import path
import numpy as np

from cmmlib import cmm
from cmmlib.inout import load_mesh
from cmmlib.vis.weights import show_many_weights


scape_dir = path.join('meshes', 'scape')
if not path.exists(scape_dir) or not path.exists(path.join(scape_dir, 'mesh000.off')):
    print "SCAPE dataset not found. You need to get it from James Davis."
    print "Instructions on this website: "
    print "http://robotics.stanford.edu/~drago/Projects/scape/scape.html"
    sys.exit(1)

meshes = [load_mesh(path.join(scape_dir, 'mesh%03d.off' % i), normalize=True)
          for i in [0, 7, 10]]

results = []
for verts, tris in meshes:
    Phi_cmh = cmm.compressed_manifold_modes(
        verts, tris, 10, mu=12.5, scaled=True, maxiter=2000)
    results.append((verts, tris, Phi_cmh))

show_many_weights(results)
